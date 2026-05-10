"""OpenFHE CKKS backend for minimal MIMO recurrence smoke tests."""

from __future__ import annotations

import random
import time
from dataclasses import asdict, dataclass
from typing import Any, Literal

from fhe_native_mamba3.backends.base import FHEBackend
from fhe_native_mamba3.backends.openfhe import OpenFheBootstrapConfig, OpenFheCkksBackend
from fhe_native_mamba3.layout import (
    ReadoutStrategy,
    readout_reduce_mask,
    readout_reduce_steps,
    readout_scatter_mask,
    readout_scatter_shifts,
    state_slot,
)
from fhe_native_mamba3.layout import (
    readout_output_slots as layout_readout_output_slots,
)
from fhe_native_mamba3.layout import (
    required_readout_rotations as layout_required_readout_rotations,
)

InputMode = Literal["server-bx", "client-update", "encrypted-dynamic-bc"]
CiphertextTraceOutputLayout = Literal["readout", "expanded-rank-input"]


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
    bootstrap_after_tokens: tuple[int, ...] = ()

    def to_json_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["problem"] = asdict(self.problem)
        return payload


@dataclass(frozen=True)
class OpenFheRecurrenceCiphertextTrace:
    """Ciphertext-only recurrence trace with no output decryption."""

    output_ciphertexts: tuple[Any, ...]
    output_slots: tuple[int, ...]
    ring_dimension: int
    batch_size: int
    multiplicative_depth: int
    rotations: tuple[int, ...]
    backend_stats: dict[str, Any]
    latency_sec_per_token: float
    readout_strategy: str
    input_mode: str
    client_plaintext_public_weight_multiplies: int = 0
    bootstrap_after_tokens: tuple[int, ...] = ()


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


def plaintext_recurrence_trace(problem: OpenFheRecurrenceProblem) -> dict[str, float]:
    """Plaintext range trace for a recurrence problem."""

    state = [[0.0 for _ in range(problem.mimo_rank)] for _ in range(problem.d_state)]
    update_abs_max = 0.0
    state_abs_max = 0.0
    output_abs_max = 0.0
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
                update = b_matrix[n][r] * rank_input[r]
                update_abs_max = max(update_abs_max, abs(update))
                state[n][r] = decay * state[n][r] + update
                state_abs_max = max(state_abs_max, abs(state[n][r]))
        for r in range(problem.mimo_rank):
            output = sum(c_matrix[n][r] * state[n][r] for n in range(problem.d_state))
            if problem.d_skip is not None:
                output += problem.d_skip[r] * rank_input[r]
            output_abs_max = max(output_abs_max, abs(output))
    return {
        "update_abs_max": update_abs_max,
        "state_abs_max": state_abs_max,
        "output_abs_max": output_abs_max,
    }


def scale_recurrence_state(
    problem: OpenFheRecurrenceProblem, state_scale: float
) -> OpenFheRecurrenceProblem:
    """Apply the equivalent state gauge transform h' = state_scale * h."""

    return scale_recurrence_state_and_output(
        problem,
        state_scale=state_scale,
        output_scale=1.0,
    )


def scale_recurrence_state_and_output(
    problem: OpenFheRecurrenceProblem,
    *,
    state_scale: float,
    output_scale: float,
) -> OpenFheRecurrenceProblem:
    """Apply h' = state_scale * h and y' = output_scale * y."""

    if state_scale <= 0:
        msg = "state_scale must be positive"
        raise ValueError(msg)
    if output_scale <= 0:
        msg = "output_scale must be positive"
        raise ValueError(msg)
    if state_scale == 1.0 and output_scale == 1.0:
        return problem
    c_scale = output_scale / state_scale
    return OpenFheRecurrenceProblem(
        rank_inputs=problem.rank_inputs,
        decay=problem.decay,
        decay_by_token=problem.decay_by_token,
        decay_state_by_token=problem.decay_state_by_token,
        b=_scale_matrix(problem.b, state_scale),
        c=_scale_matrix(problem.c, c_scale),
        b_by_token=_scale_token_matrices(problem.b_by_token, state_scale),
        c_by_token=_scale_token_matrices(problem.c_by_token, c_scale),
        d_skip=_scale_vector(problem.d_skip, output_scale),
    )


def _scale_vector(values: tuple[float, ...] | None, scale: float) -> tuple[float, ...] | None:
    if values is None:
        return None
    return tuple(scale * value for value in values)


def _scale_matrix(
    matrix: tuple[tuple[float, ...], ...], scale: float
) -> tuple[tuple[float, ...], ...]:
    return tuple(tuple(scale * value for value in row) for row in matrix)


def _scale_token_matrices(
    matrices: tuple[tuple[tuple[float, ...], ...], ...] | None,
    scale: float,
) -> tuple[tuple[tuple[float, ...], ...], ...] | None:
    if matrices is None:
        return None
    return tuple(_scale_matrix(matrix, scale) for matrix in matrices)


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

    return layout_required_readout_rotations(
        d_state=d_state,
        mimo_rank=mimo_rank,
        readout_strategy=readout_strategy,
    )


def readout_output_slots(
    *,
    d_state: int,
    mimo_rank: int,
    readout_strategy: ReadoutStrategy,
) -> tuple[int, ...]:
    """Slots that contain the per-rank readout after the selected layout."""

    return layout_readout_output_slots(
        d_state=d_state,
        mimo_rank=mimo_rank,
        readout_strategy=readout_strategy,
    )


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
            source = state_slot(d_state=d_state, rank_index=r, state_index=n)
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
    for step in readout_reduce_steps(d_state):
        mask = readout_reduce_mask(
            d_state=d_state,
            mimo_rank=rank,
            step=step,
            batch_size=backend.batch_size,
        )
        rotated = backend.rotate(reduced, step)
        reduced = backend.add(reduced, backend.mul_plain(rotated, backend.encode(mask)))

    if not dense_output:
        return reduced

    output_ct = backend.encrypt([0.0] * backend.batch_size)
    for r, shift in enumerate(
        readout_scatter_shifts(d_state=d_state, mimo_rank=rank, dense_output=dense_output)
    ):
        mask = readout_scatter_mask(
            d_state=d_state,
            mimo_rank=rank,
            rank_index=r,
            batch_size=backend.batch_size,
        )
        term = backend.mul_plain(reduced, backend.encode(mask))
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


def _expand_readout_ciphertext_to_rank_input_slots(
    *,
    backend: FHEBackend,
    output_ct: Any,
    output_slots: tuple[int, ...],
    d_state: int,
    rank: int,
) -> Any:
    """Expand rank readout slots to the repeated rank-input state-slot layout."""

    expanded_ct = backend.encrypt([0.0] * backend.batch_size)
    for r, source in enumerate(output_slots):
        for n in range(d_state):
            target = r * d_state + n
            mask = [0.0] * backend.batch_size
            mask[source] = 1.0
            term = backend.mul_plain(output_ct, backend.encode(mask))
            shift = source - target
            if shift:
                term = backend.rotate(term, shift)
            expanded_ct = backend.add(expanded_ct, term)
    return expanded_ct


def run_static_mimo_recurrence_with_backend(
    problem: OpenFheRecurrenceProblem,
    *,
    backend: FHEBackend,
    multiplicative_depth: int,
    readout_strategy: ReadoutStrategy = "slotwise",
    input_mode: InputMode = "client-update",
    bootstrap_every_tokens: int = 0,
    bootstrap_after_tokens: tuple[int, ...] = (),
    rank_input_ciphertexts: tuple[Any, ...] | None = None,
) -> OpenFheRecurrenceResult:
    """Evaluate the encrypted static MIMO recurrence with a backend."""

    trace = run_static_mimo_recurrence_ciphertexts_with_backend(
        problem,
        backend=backend,
        multiplicative_depth=multiplicative_depth,
        readout_strategy=readout_strategy,
        input_mode=input_mode,
        bootstrap_every_tokens=bootstrap_every_tokens,
        bootstrap_after_tokens=bootstrap_after_tokens,
        rank_input_ciphertexts=rank_input_ciphertexts,
    )
    decrypted_outputs: list[tuple[float, ...]] = []
    for output_ct in trace.output_ciphertexts:
        output_values = backend.decrypt(output_ct, length=backend.batch_size)
        decrypted_outputs.append(tuple(output_values[slot] for slot in trace.output_slots))

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
        ring_dimension=trace.ring_dimension,
        batch_size=trace.batch_size,
        multiplicative_depth=trace.multiplicative_depth,
        rotations=trace.rotations,
        backend_stats=backend.stats().to_json_dict(),
        latency_sec_per_token=trace.latency_sec_per_token,
        readout_strategy=trace.readout_strategy,
        input_mode=trace.input_mode,
        client_plaintext_public_weight_multiplies=(trace.client_plaintext_public_weight_multiplies),
        bootstrap_after_tokens=trace.bootstrap_after_tokens,
    )


def run_static_mimo_recurrence_ciphertexts_with_backend(
    problem: OpenFheRecurrenceProblem,
    *,
    backend: FHEBackend,
    multiplicative_depth: int,
    readout_strategy: ReadoutStrategy = "slotwise",
    input_mode: InputMode = "client-update",
    bootstrap_every_tokens: int = 0,
    bootstrap_after_tokens: tuple[int, ...] = (),
    rank_input_ciphertexts: tuple[Any, ...] | None = None,
    output_layout: CiphertextTraceOutputLayout = "readout",
) -> OpenFheRecurrenceCiphertextTrace:
    """Evaluate recurrence and return ciphertext outputs without decrypting."""

    d_state = problem.d_state
    rank = problem.mimo_rank
    slots = d_state * rank
    bootstrap_tokens = _resolve_bootstrap_after_tokens(
        seq_len=problem.seq_len,
        every=bootstrap_every_tokens,
        explicit=bootstrap_after_tokens,
    )
    if input_mode not in {"server-bx", "client-update", "encrypted-dynamic-bc"}:
        msg = f"unsupported input_mode: {input_mode}"
        raise ValueError(msg)
    if output_layout not in {"readout", "expanded-rank-input"}:
        msg = f"unsupported output_layout: {output_layout}"
        raise ValueError(msg)
    if rank_input_ciphertexts is not None:
        if len(rank_input_ciphertexts) != problem.seq_len:
            msg = "rank_input_ciphertexts length must match seq_len"
            raise ValueError(msg)
        if input_mode == "client-update":
            msg = "rank_input_ciphertexts require server-bx or encrypted-dynamic-bc input mode"
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

    output_ciphertexts: list[Any] = []
    readout_slots = readout_output_slots(
        d_state=d_state,
        mimo_rank=rank,
        readout_strategy=readout_strategy,
    )
    output_slots = (
        tuple(r * d_state for r in range(rank))
        if output_layout == "expanded-rank-input"
        else readout_slots
    )
    client_plaintext_public_weight_multiplies = 0
    for t, rank_input in enumerate(problem.rank_inputs):
        b_matrix = _matrix_at_token(problem.b_by_token, problem.b, t)
        c_matrix = _matrix_at_token(problem.c_by_token, problem.c, t)
        if rank_input_ciphertexts is not None:
            input_ct = rank_input_ciphertexts[t]
        elif input_mode == "client-update":
            input_ct = None
        else:
            input_ct = backend.encrypt(_expanded_rank_input(rank_input, d_state, rank))
        if input_mode == "encrypted-dynamic-bc":
            b_ct = backend.encrypt(_flat_state_by_rank(b_matrix, d_state, rank))
            update_ct = backend.mul_ct(input_ct, b_ct)
        elif input_mode == "server-bx":
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
        state_ct = backend.add(decayed_state_ct, update_ct)
        token_number = t + 1
        if token_number in bootstrap_tokens:
            state_ct = backend.bootstrap(state_ct)
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
                        output_slots=readout_slots,
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
                    output_slots=readout_slots,
                )
            output_ct = backend.add(output_ct, d_skip_ct)
        if output_layout == "expanded-rank-input":
            output_ct = _expand_readout_ciphertext_to_rank_input_slots(
                backend=backend,
                output_ct=output_ct,
                output_slots=readout_slots,
                d_state=d_state,
                rank=rank,
            )
        output_ciphertexts.append(output_ct)
    eval_seconds = time.perf_counter() - started
    backend.stats().eval_seconds += eval_seconds

    return OpenFheRecurrenceCiphertextTrace(
        output_ciphertexts=tuple(output_ciphertexts),
        output_slots=output_slots,
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
        bootstrap_after_tokens=tuple(sorted(bootstrap_tokens)),
    )


def run_openfhe_static_recurrence(
    problem: OpenFheRecurrenceProblem,
    *,
    multiplicative_depth: int | None = None,
    scaling_mod_size: int = 50,
    readout_strategy: ReadoutStrategy = "slotwise",
    input_mode: InputMode = "client-update",
    bootstrap_every_tokens: int = 0,
    bootstrap_after_tokens: tuple[int, ...] = (),
    bootstrap_config: OpenFheBootstrapConfig | None = None,
    ring_dimension: int | None = None,
) -> OpenFheRecurrenceResult:
    """Encrypt inputs and evaluate the static MIMO recurrence with OpenFHE CKKS."""

    depth = multiplicative_depth or max(8, problem.seq_len + 5)
    backend = OpenFheCkksBackend(
        batch_size=problem.d_state * problem.mimo_rank,
        multiplicative_depth=depth,
        scaling_mod_size=scaling_mod_size,
        bootstrap_config=bootstrap_config,
        ring_dimension=ring_dimension,
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
        bootstrap_every_tokens=bootstrap_every_tokens,
        bootstrap_after_tokens=bootstrap_after_tokens,
    )


def _resolve_bootstrap_after_tokens(
    *,
    seq_len: int,
    every: int,
    explicit: tuple[int, ...],
) -> set[int]:
    if every < 0:
        msg = "bootstrap_every_tokens must be non-negative"
        raise ValueError(msg)
    if any(token <= 0 or token > seq_len for token in explicit):
        msg = f"bootstrap_after_tokens must be in [1, {seq_len}]"
        raise ValueError(msg)
    tokens = set(explicit)
    if every:
        tokens.update(range(every, seq_len + 1, every))
    return tokens
