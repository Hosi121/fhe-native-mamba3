"""OpenFHE CKKS backend for minimal MIMO recurrence smoke tests."""

from __future__ import annotations

import random
import time
from dataclasses import asdict, dataclass
from typing import Any, Literal

from fhe_native_mamba3.backends.base import FHEBackend
from fhe_native_mamba3.backends.openfhe import OpenFheCkksBackend

ReadoutStrategy = Literal["slotwise", "rank-reduce", "rank-local"]
InputMode = Literal["server-bx", "client-update", "encrypted-dynamic-bc"]


@dataclass(frozen=True)
class OpenFheRecurrenceProblem:
    """Plain inputs for an encrypted MIMO recurrence run."""

    rank_inputs: tuple[tuple[float, ...], ...]
    decay: tuple[float, ...]
    b: tuple[tuple[float, ...], ...]
    c: tuple[tuple[float, ...], ...]
    decay_by_token: tuple[tuple[float, ...], ...] | None = None
    decay_state_by_token: tuple[tuple[tuple[float, ...], ...], ...] | None = None
    b_by_token: tuple[tuple[tuple[float, ...], ...], ...] | None = None
    c_by_token: tuple[tuple[tuple[float, ...], ...], ...] | None = None
    d_skip: tuple[float, ...] | None = None

    @property
    def seq_len(self) -> int:
        return len(self.rank_inputs)

    @property
    def d_state(self) -> int:
        if self.b_by_token is not None and self.b_by_token:
            return len(self.b_by_token[0])
        return len(self.b)

    @property
    def mimo_rank(self) -> int:
        return len(self.decay)


@dataclass(frozen=True)
class OpenFheRecurrenceResult:
    """Result of an encrypted OpenFHE recurrence run."""

    problem: OpenFheRecurrenceProblem
    decrypted_outputs: tuple[tuple[float, ...], ...]
    expected_outputs: tuple[tuple[float, ...], ...]
    max_abs_error: float
    ring_dimension: int
    batch_size: int
    multiplicative_depth: int
    rotations: tuple[int, ...]
    backend_stats: dict[str, Any]
    latency_sec_per_token: float
    readout_strategy: str
    input_mode: str
    client_plaintext_public_weight_multiplies: int

    def to_json_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["problem"] = asdict(self.problem)
        return payload


def make_demo_problem(
    *,
    seq_len: int = 3,
    d_state: int = 2,
    mimo_rank: int = 2,
    seed: int = 7,
) -> OpenFheRecurrenceProblem:
    """Create a deterministic low-noise recurrence problem."""

    if seq_len <= 0 or d_state <= 0 or mimo_rank <= 0:
        msg = "seq_len, d_state, and mimo_rank must be positive"
        raise ValueError(msg)

    rng = random.Random(seed)
    rank_inputs = tuple(
        tuple(round(rng.uniform(-1.25, 1.25), 4) for _ in range(mimo_rank)) for _ in range(seq_len)
    )
    decay = tuple(round(0.35 + 0.1 * r + rng.uniform(-0.02, 0.02), 4) for r in range(mimo_rank))
    b = tuple(
        tuple(round(rng.uniform(-0.55, 0.55), 4) for _ in range(mimo_rank)) for _ in range(d_state)
    )
    c = tuple(
        tuple(round(rng.uniform(-0.55, 0.55), 4) for _ in range(mimo_rank)) for _ in range(d_state)
    )
    return OpenFheRecurrenceProblem(rank_inputs=rank_inputs, decay=decay, b=b, c=c)


def plaintext_static_recurrence(
    problem: OpenFheRecurrenceProblem,
) -> tuple[tuple[float, ...], ...]:
    """Reference static recurrence in plaintext."""

    state = [[0.0 for _ in range(problem.mimo_rank)] for _ in range(problem.d_state)]
    outputs: list[tuple[float, ...]] = []
    for t, rank_input in enumerate(problem.rank_inputs):
        b_matrix = problem.b_by_token[t] if problem.b_by_token is not None else problem.b
        c_matrix = problem.c_by_token[t] if problem.c_by_token is not None else problem.c
        for n in range(problem.d_state):
            for r in range(problem.mimo_rank):
                if problem.decay_state_by_token is not None:
                    decay = problem.decay_state_by_token[t][n][r]
                elif problem.decay_by_token is not None:
                    decay = problem.decay_by_token[t][r]
                else:
                    decay = problem.decay[r]
                state[n][r] = decay * state[n][r] + b_matrix[n][r] * rank_input[r]
        outputs.append(
            tuple(
                sum(c_matrix[n][r] * state[n][r] for n in range(problem.d_state))
                + ((problem.d_skip[r] * rank_input[r]) if problem.d_skip is not None else 0.0)
                for r in range(problem.mimo_rank)
            )
        )
    return tuple(outputs)


def _flat_state_by_rank(
    matrix: tuple[tuple[float, ...], ...], d_state: int, rank: int
) -> list[float]:
    return [matrix[n][r] for r in range(rank) for n in range(d_state)]


def _expanded_rank_input(rank_input: tuple[float, ...], d_state: int, rank: int) -> list[float]:
    return [rank_input[r] for r in range(rank) for _ in range(d_state)]


def _expanded_update(
    rank_input: tuple[float, ...],
    b_matrix: tuple[tuple[float, ...], ...],
    d_state: int,
    rank: int,
) -> list[float]:
    return [b_matrix[n][r] * rank_input[r] for r in range(rank) for n in range(d_state)]


def _matrix_at_token(
    token_matrices: tuple[tuple[tuple[float, ...], ...], ...] | None,
    fallback: tuple[tuple[float, ...], ...],
    token_index: int,
) -> tuple[tuple[float, ...], ...]:
    if token_matrices is None:
        return fallback
    return token_matrices[token_index]


def _validate_token_matrices(
    matrices: tuple[tuple[tuple[float, ...], ...], ...],
    *,
    seq_len: int,
    d_state: int,
    rank: int,
    name: str,
) -> None:
    if len(matrices) != seq_len:
        msg = f"{name} length must match seq_len"
        raise ValueError(msg)
    for token_index, matrix in enumerate(matrices):
        if len(matrix) != d_state:
            msg = f"{name}[{token_index}] must have d_state={d_state} rows"
            raise ValueError(msg)
        bad_rows = [len(row) for row in matrix if len(row) != rank]
        if bad_rows:
            msg = f"each {name}[{token_index}] row must match mimo_rank={rank}"
            raise ValueError(msg)


def _d_skip_output_vector(
    *,
    rank_input: tuple[float, ...],
    d_skip: tuple[float, ...],
    output_slots: tuple[int, ...],
    batch_size: int,
) -> list[float]:
    values = [0.0] * batch_size
    for r, slot in enumerate(output_slots):
        values[slot] = d_skip[r] * rank_input[r]
    return values


def required_readout_rotations(
    *,
    d_state: int,
    mimo_rank: int,
    readout_strategy: ReadoutStrategy = "slotwise",
) -> tuple[int, ...]:
    """Rotations needed to reduce state slots into per-rank output slots."""

    if readout_strategy in {"rank-reduce", "rank-local"}:
        reduce_steps = {2**stage for stage in range(max(0, (d_state - 1).bit_length()))}
        scatter_steps = (
            set()
            if readout_strategy == "rank-local"
            else {r * d_state - r for r in range(mimo_rank) if r * d_state - r != 0}
        )
        return tuple(sorted(reduce_steps | scatter_steps))
    if readout_strategy != "slotwise":
        msg = f"unsupported readout_strategy: {readout_strategy}"
        raise ValueError(msg)

    return tuple(
        sorted(
            {
                r * d_state + n - r
                for r in range(mimo_rank)
                for n in range(d_state)
                if r * d_state + n - r != 0
            }
        )
    )


def readout_output_slots(
    *,
    d_state: int,
    mimo_rank: int,
    readout_strategy: ReadoutStrategy,
) -> tuple[int, ...]:
    """Slots that contain the per-rank readout after the selected layout."""

    if readout_strategy in {"slotwise", "rank-reduce"}:
        return tuple(range(mimo_rank))
    if readout_strategy == "rank-local":
        return tuple(rank * d_state for rank in range(mimo_rank))
    msg = f"unsupported readout_strategy: {readout_strategy}"
    raise ValueError(msg)


def _readout_ciphertext(
    *,
    backend: FHEBackend,
    contrib_ct: Any,
    d_state: int,
    rank: int,
    readout_strategy: ReadoutStrategy,
) -> Any:
    if readout_strategy in {"rank-reduce", "rank-local"}:
        return _readout_rank_reduce(
            backend=backend,
            contrib_ct=contrib_ct,
            d_state=d_state,
            rank=rank,
            dense_output=readout_strategy == "rank-reduce",
        )
    if readout_strategy != "slotwise":
        msg = f"unsupported readout_strategy: {readout_strategy}"
        raise ValueError(msg)
    return _readout_slotwise(
        backend=backend,
        contrib_ct=contrib_ct,
        d_state=d_state,
        rank=rank,
    )


def _readout_slotwise(
    *,
    backend: FHEBackend,
    contrib_ct: Any,
    d_state: int,
    rank: int,
) -> Any:
    output_ct = backend.encrypt([0.0] * backend.batch_size)
    for r in range(rank):
        for n in range(d_state):
            source = r * d_state + n
            target = r
            mask = [0.0] * backend.batch_size
            mask[source] = 1.0
            term = backend.mul_plain(contrib_ct, backend.encode(mask))
            shift = source - target
            if shift:
                term = backend.rotate(term, shift)
            output_ct = backend.add(output_ct, term)
    return output_ct


def _readout_rank_reduce(
    *,
    backend: FHEBackend,
    contrib_ct: Any,
    d_state: int,
    rank: int,
    dense_output: bool,
) -> Any:
    reduced = contrib_ct
    stage = 0
    step = 1
    while step < d_state:
        mask = []
        for _r in range(rank):
            for n in range(d_state):
                mask.append(1.0 if n + step < d_state and n % (2 * step) == 0 else 0.0)
        rotated = backend.rotate(reduced, step)
        reduced = backend.add(reduced, backend.mul_plain(rotated, backend.encode(mask)))
        stage += 1
        step = 2**stage

    if not dense_output:
        return reduced

    output_ct = backend.encrypt([0.0] * backend.batch_size)
    for r in range(rank):
        source = r * d_state
        target = r if dense_output else source
        mask = [0.0] * backend.batch_size
        mask[source] = 1.0
        term = backend.mul_plain(reduced, backend.encode(mask))
        shift = source - target
        if shift:
            term = backend.rotate(term, shift)
        output_ct = backend.add(output_ct, term)
    return output_ct


def _d_skip_from_input_ciphertext(
    *,
    backend: FHEBackend,
    input_ct: Any,
    d_skip: tuple[float, ...],
    d_state: int,
    rank: int,
    output_slots: tuple[int, ...],
) -> Any:
    if all(output_slots[r] == r * d_state for r in range(rank)):
        mask = [0.0] * backend.batch_size
        for r in range(rank):
            mask[r * d_state] = d_skip[r]
        return backend.mul_plain(input_ct, backend.encode(mask))

    output_ct = backend.encrypt([0.0] * backend.batch_size)
    for r in range(rank):
        source = r * d_state
        target = output_slots[r]
        mask = [0.0] * backend.batch_size
        mask[source] = d_skip[r]
        term = backend.mul_plain(input_ct, backend.encode(mask))
        shift = source - target
        if shift:
            term = backend.rotate(term, shift)
        output_ct = backend.add(output_ct, term)
    return output_ct


def run_static_mimo_recurrence_with_backend(
    problem: OpenFheRecurrenceProblem,
    *,
    backend: FHEBackend,
    multiplicative_depth: int,
    readout_strategy: ReadoutStrategy = "slotwise",
    input_mode: InputMode = "client-update",
) -> OpenFheRecurrenceResult:
    """Evaluate the encrypted static MIMO recurrence with a backend."""

    d_state = problem.d_state
    rank = problem.mimo_rank
    slots = d_state * rank
    if input_mode not in {"server-bx", "client-update", "encrypted-dynamic-bc"}:
        msg = f"unsupported input_mode: {input_mode}"
        raise ValueError(msg)
    if backend.batch_size < slots:
        msg = f"backend batch_size={backend.batch_size} is smaller than required slots={slots}"
        raise ValueError(msg)
    if problem.d_skip is not None and len(problem.d_skip) != rank:
        msg = f"d_skip length must match mimo_rank={rank}"
        raise ValueError(msg)
    if problem.decay_by_token is not None:
        if len(problem.decay_by_token) != problem.seq_len:
            msg = "decay_by_token length must match seq_len"
            raise ValueError(msg)
        bad_rows = [len(row) for row in problem.decay_by_token if len(row) != rank]
        if bad_rows:
            msg = f"each decay_by_token row must match mimo_rank={rank}"
            raise ValueError(msg)
    if problem.decay_state_by_token is not None:
        _validate_token_matrices(
            problem.decay_state_by_token,
            seq_len=problem.seq_len,
            d_state=d_state,
            rank=rank,
            name="decay_state_by_token",
        )
    if problem.b_by_token is not None:
        _validate_token_matrices(
            problem.b_by_token,
            seq_len=problem.seq_len,
            d_state=d_state,
            rank=rank,
            name="b_by_token",
        )
    if problem.c_by_token is not None:
        _validate_token_matrices(
            problem.c_by_token,
            seq_len=problem.seq_len,
            d_state=d_state,
            rank=rank,
            name="c_by_token",
        )
    if input_mode == "encrypted-dynamic-bc" and (
        problem.b_by_token is None or problem.c_by_token is None
    ):
        msg = "encrypted-dynamic-bc input mode requires b_by_token and c_by_token"
        raise ValueError(msg)

    started = time.perf_counter()
    state_ct = backend.encrypt([0.0] * backend.batch_size)
    decay_pt = backend.encode([problem.decay[r] for r in range(rank) for _ in range(d_state)])
    b_pt = backend.encode(_flat_state_by_rank(problem.b, d_state, rank))
    c_pt = backend.encode(_flat_state_by_rank(problem.c, d_state, rank))

    decrypted_outputs: list[tuple[float, ...]] = []
    output_slots = readout_output_slots(
        d_state=d_state,
        mimo_rank=rank,
        readout_strategy=readout_strategy,
    )
    client_plaintext_public_weight_multiplies = 0
    for t, rank_input in enumerate(problem.rank_inputs):
        b_matrix = _matrix_at_token(problem.b_by_token, problem.b, t)
        c_matrix = _matrix_at_token(problem.c_by_token, problem.c, t)
        input_ct = None
        if input_mode == "encrypted-dynamic-bc":
            input_ct = backend.encrypt(_expanded_rank_input(rank_input, d_state, rank))
            b_ct = backend.encrypt(_flat_state_by_rank(b_matrix, d_state, rank))
            update_ct = backend.mul_ct(input_ct, b_ct)
        elif input_mode == "server-bx":
            input_ct = backend.encrypt(_expanded_rank_input(rank_input, d_state, rank))
            if problem.b_by_token is None:
                update_ct = backend.mul_plain(input_ct, b_pt)
            else:
                update_ct = backend.mul_plain(
                    input_ct,
                    backend.encode(_flat_state_by_rank(b_matrix, d_state, rank)),
                )
        else:
            update_ct = backend.encrypt(_expanded_update(rank_input, b_matrix, d_state, rank))
            client_plaintext_public_weight_multiplies += slots
        if problem.decay_state_by_token is not None:
            decay_ct = backend.encrypt(
                _flat_state_by_rank(problem.decay_state_by_token[t], d_state, rank)
            )
            decayed_state_ct = backend.mul_ct(state_ct, decay_ct)
        elif problem.decay_by_token is None:
            decayed_state_ct = backend.mul_plain(state_ct, decay_pt)
        else:
            decay_ct = backend.encrypt(
                [problem.decay_by_token[t][r] for r in range(rank) for _ in range(d_state)]
            )
            decayed_state_ct = backend.mul_ct(state_ct, decay_ct)
        state_ct = backend.add(
            decayed_state_ct,
            update_ct,
        )
        if input_mode == "encrypted-dynamic-bc":
            c_ct = backend.encrypt(_flat_state_by_rank(c_matrix, d_state, rank))
            contrib_ct = backend.mul_ct(state_ct, c_ct)
        elif problem.c_by_token is None:
            contrib_ct = backend.mul_plain(state_ct, c_pt)
        else:
            contrib_ct = backend.mul_plain(
                state_ct,
                backend.encode(_flat_state_by_rank(c_matrix, d_state, rank)),
            )
        output_ct = _readout_ciphertext(
            backend=backend,
            contrib_ct=contrib_ct,
            d_state=d_state,
            rank=rank,
            readout_strategy=readout_strategy,
        )
        if problem.d_skip is not None:
            if input_ct is None:
                d_skip_ct = backend.encrypt(
                    _d_skip_output_vector(
                        rank_input=rank_input,
                        d_skip=problem.d_skip,
                        output_slots=output_slots,
                        batch_size=backend.batch_size,
                    )
                )
                client_plaintext_public_weight_multiplies += rank
            else:
                d_skip_ct = _d_skip_from_input_ciphertext(
                    backend=backend,
                    input_ct=input_ct,
                    d_skip=problem.d_skip,
                    d_state=d_state,
                    rank=rank,
                    output_slots=output_slots,
                )
            output_ct = backend.add(output_ct, d_skip_ct)
        output_values = backend.decrypt(output_ct, length=backend.batch_size)
        decrypted_outputs.append(tuple(output_values[slot] for slot in output_slots))
    eval_seconds = time.perf_counter() - started
    backend.stats().eval_seconds += eval_seconds

    expected_outputs = plaintext_static_recurrence(problem)
    max_abs_error = max(
        abs(actual - expected)
        for actual_row, expected_row in zip(decrypted_outputs, expected_outputs, strict=True)
        for actual, expected in zip(actual_row, expected_row, strict=True)
    )

    return OpenFheRecurrenceResult(
        problem=problem,
        decrypted_outputs=tuple(decrypted_outputs),
        expected_outputs=expected_outputs,
        max_abs_error=max_abs_error,
        ring_dimension=backend.ring_dimension,
        batch_size=backend.batch_size,
        multiplicative_depth=multiplicative_depth,
        rotations=required_readout_rotations(
            d_state=d_state,
            mimo_rank=rank,
            readout_strategy=readout_strategy,
        ),
        backend_stats=backend.stats().to_json_dict(),
        latency_sec_per_token=eval_seconds / problem.seq_len,
        readout_strategy=readout_strategy,
        input_mode=input_mode,
        client_plaintext_public_weight_multiplies=client_plaintext_public_weight_multiplies,
    )


def run_openfhe_static_recurrence(
    problem: OpenFheRecurrenceProblem,
    *,
    multiplicative_depth: int | None = None,
    scaling_mod_size: int = 50,
    readout_strategy: ReadoutStrategy = "slotwise",
    input_mode: InputMode = "client-update",
) -> OpenFheRecurrenceResult:
    """Encrypt inputs and evaluate the static MIMO recurrence with OpenFHE CKKS."""

    depth = multiplicative_depth or max(8, problem.seq_len + 5)
    backend = OpenFheCkksBackend(
        batch_size=problem.d_state * problem.mimo_rank,
        multiplicative_depth=depth,
        scaling_mod_size=scaling_mod_size,
        rotations=required_readout_rotations(
            d_state=problem.d_state,
            mimo_rank=problem.mimo_rank,
            readout_strategy=readout_strategy,
        ),
    )
    return run_static_mimo_recurrence_with_backend(
        problem,
        backend=backend,
        multiplicative_depth=depth,
        readout_strategy=readout_strategy,
        input_mode=input_mode,
    )
