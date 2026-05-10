"""OpenFHE CKKS backend for minimal MIMO recurrence smoke tests."""

from __future__ import annotations

import random
import time
from dataclasses import asdict, dataclass, replace
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
CiphertextTraceOutputLayout = Literal["readout", "expanded-rank-input", "visible-output"]


@dataclass(frozen=True)
class CiphertextLayoutContract:
    """Slot-layout metadata attached to ciphertext trace outputs."""

    output_layout: CiphertextTraceOutputLayout
    d_state: int
    mimo_rank: int
    readout_strategy: ReadoutStrategy
    output_slots: tuple[int, ...]
    required_rotations: tuple[int, ...]


class LayoutBoundCiphertexts(tuple):
    """Tuple of ciphertexts carrying their slot-layout contract."""

    layout_contract: CiphertextLayoutContract

    def __new__(
        cls,
        values: tuple[Any, ...],
        *,
        layout_contract: CiphertextLayoutContract,
    ) -> LayoutBoundCiphertexts:
        instance = super().__new__(cls, values)
        instance.layout_contract = layout_contract
        return instance


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
    output_layout: CiphertextTraceOutputLayout
    layout_contract: CiphertextLayoutContract
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


@dataclass(frozen=True)
class OpenFheRecurrenceCiphertextChainResult:
    """Result for a recurrence chain with ciphertext handoff between layers."""

    decrypted_outputs: tuple[tuple[float, ...], ...]
    expected_outputs: tuple[tuple[float, ...], ...]
    max_abs_error: float
    layer_count: int
    seq_len: int
    ring_dimension: int
    batch_size: int
    multiplicative_depth: int
    rotations: tuple[int, ...]
    backend_stats: dict[str, Any]
    latency_sec_per_token: float
    readout_strategy: str
    input_mode: str
    bootstrap_after_layers: tuple[int, ...] = ()
    intermediate_decrypt_count: int = 0
    ciphertext_chain: bool = True
    encrypted_chain: bool = False
    full_layer_correctness_claimed: bool = False

    def to_json_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class _RecurrenceRunPlan:
    d_state: int
    rank: int
    slots: int
    bootstrap_tokens: frozenset[int]


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


def _prepare_recurrence_run_plan(
    problem: OpenFheRecurrenceProblem,
    *,
    backend: FHEBackend,
    input_mode: str,
    bootstrap_every_tokens: int,
    bootstrap_after_tokens: tuple[int, ...],
    rank_input_ciphertexts: tuple[Any, ...] | None,
    output_layout: str,
) -> _RecurrenceRunPlan:
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
        _validate_rank_input_ciphertext_contract(
            rank_input_ciphertexts,
            d_state=d_state,
            rank=rank,
        )
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
    return _RecurrenceRunPlan(
        d_state=d_state,
        rank=rank,
        slots=slots,
        bootstrap_tokens=frozenset(bootstrap_tokens),
    )


def _validate_rank_input_ciphertext_contract(
    rank_input_ciphertexts: tuple[Any, ...],
    *,
    d_state: int,
    rank: int,
) -> None:
    contract = getattr(rank_input_ciphertexts, "layout_contract", None)
    if contract is None:
        msg = "rank_input_ciphertexts must carry a ciphertext layout contract"
        raise ValueError(msg)
    if contract.output_layout != "expanded-rank-input":
        msg = "rank_input_ciphertexts must use expanded-rank-input output_layout"
        raise ValueError(msg)
    if contract.d_state != d_state or contract.mimo_rank != rank:
        msg = (
            "rank_input_ciphertexts layout contract must match d_state and mimo_rank; "
            f"got d_state={contract.d_state}, mimo_rank={contract.mimo_rank}"
        )
        raise ValueError(msg)
    expected_slots = tuple(r * d_state for r in range(rank))
    if contract.output_slots != expected_slots:
        msg = (
            "rank_input_ciphertexts expanded-rank-input contract has incompatible output_slots; "
            f"expected {expected_slots}, got {contract.output_slots}"
        )
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


def required_recurrence_chain_rotations(
    *,
    d_state: int,
    mimo_rank: int,
    readout_strategy: ReadoutStrategy = "rank-local",
) -> tuple[int, ...]:
    """Rotations needed for recurrence readout plus inter-layer input expansion."""

    rotations = set(
        required_readout_rotations(
            d_state=d_state,
            mimo_rank=mimo_rank,
            readout_strategy=readout_strategy,
        )
    )
    rotations.update(
        _expanded_rank_input_rotations(
            d_state=d_state,
            rank=mimo_rank,
            readout_strategy=readout_strategy,
        )
    )
    return tuple(sorted(rotations))


def _output_layout_rotations(
    *,
    d_state: int,
    rank: int,
    readout_strategy: ReadoutStrategy,
    output_layout: CiphertextTraceOutputLayout,
) -> tuple[int, ...]:
    if output_layout == "expanded-rank-input":
        return required_recurrence_chain_rotations(
            d_state=d_state,
            mimo_rank=rank,
            readout_strategy=readout_strategy,
        )
    return required_readout_rotations(
        d_state=d_state,
        mimo_rank=rank,
        readout_strategy=readout_strategy,
    )


def _bind_ciphertext_layout(
    ciphertexts: tuple[Any, ...],
    *,
    layout_contract: CiphertextLayoutContract,
) -> LayoutBoundCiphertexts:
    return LayoutBoundCiphertexts(ciphertexts, layout_contract=layout_contract)


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


def _expanded_rank_input_rotations(
    *,
    d_state: int,
    rank: int,
    readout_strategy: ReadoutStrategy,
) -> tuple[int, ...]:
    output_slots = readout_output_slots(
        d_state=d_state,
        mimo_rank=rank,
        readout_strategy=readout_strategy,
    )
    rotations: set[int] = set()
    for r, source in enumerate(output_slots):
        for n in range(d_state):
            target = r * d_state + n
            shift = source - target
            if shift:
                rotations.add(shift)
    return tuple(sorted(rotations))


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


def run_static_mimo_recurrence_ciphertext_chain_with_backend(
    problems: tuple[OpenFheRecurrenceProblem, ...],
    *,
    backend: FHEBackend,
    multiplicative_depth: int,
    readout_strategy: ReadoutStrategy = "rank-local",
    input_mode: InputMode = "server-bx",
    bootstrap_after_layers: tuple[int, ...] = (),
) -> OpenFheRecurrenceCiphertextChainResult:
    """Run recurrence layers as a ciphertext handoff chain.

    This helper is intentionally narrower than a full Mamba layer: it chains the
    recurrence readout of layer ``i`` into the rank input slots of layer
    ``i + 1`` without decrypting intermediate outputs. It is the low-level
    contract needed before wiring gate/out-projection/residual handoff for real
    checkpoint layers.
    """

    _validate_recurrence_chain(problems)
    if input_mode not in {"server-bx", "encrypted-dynamic-bc"}:
        msg = "ciphertext recurrence chain requires server-bx or encrypted-dynamic-bc input_mode"
        raise ValueError(msg)
    bootstrap_after = _validate_bootstrap_after_layers(
        bootstrap_after_layers,
        layer_count=len(problems),
    )
    chain_rotations = required_recurrence_chain_rotations(
        d_state=problems[0].d_state,
        mimo_rank=problems[0].mimo_rank,
        readout_strategy=readout_strategy,
    )

    started_decrypts = backend.stats().decrypt_count
    started = time.perf_counter()
    trace: OpenFheRecurrenceCiphertextTrace | None = None
    rank_input_ciphertexts: tuple[Any, ...] | None = None
    for layer_index, problem in enumerate(problems):
        is_last = layer_index == len(problems) - 1
        trace = run_static_mimo_recurrence_ciphertexts_with_backend(
            problem,
            backend=backend,
            multiplicative_depth=multiplicative_depth,
            readout_strategy=readout_strategy,
            input_mode=input_mode,
            rank_input_ciphertexts=rank_input_ciphertexts,
            output_layout="readout" if is_last else "expanded-rank-input",
        )
        rank_input_ciphertexts = trace.output_ciphertexts
        if layer_index + 1 in bootstrap_after:
            rank_input_ciphertexts = _bind_ciphertext_layout(
                tuple(backend.bootstrap(ct) for ct in rank_input_ciphertexts),
                layout_contract=trace.layout_contract,
            )
    if trace is None:
        msg = "problems must not be empty"
        raise ValueError(msg)

    eval_seconds = time.perf_counter() - started
    decrypted_output_rows: list[tuple[float, ...]] = []
    for output_ct in trace.output_ciphertexts:
        output_values = backend.decrypt(output_ct, length=backend.batch_size)
        decrypted_output_rows.append(tuple(output_values[slot] for slot in trace.output_slots))
    decrypted_outputs = tuple(decrypted_output_rows)
    expected_outputs = _plaintext_recurrence_chain_outputs(problems)
    max_abs_error = max(
        (
            abs(actual - expected)
            for actual_token, expected_token in zip(
                decrypted_outputs,
                expected_outputs,
                strict=True,
            )
            for actual, expected in zip(actual_token, expected_token, strict=True)
        ),
        default=0.0,
    )
    return OpenFheRecurrenceCiphertextChainResult(
        decrypted_outputs=decrypted_outputs,
        expected_outputs=expected_outputs,
        max_abs_error=max_abs_error,
        layer_count=len(problems),
        seq_len=problems[0].seq_len,
        ring_dimension=trace.ring_dimension,
        batch_size=trace.batch_size,
        multiplicative_depth=multiplicative_depth,
        rotations=chain_rotations,
        backend_stats=backend.stats().to_json_dict(),
        latency_sec_per_token=eval_seconds / problems[0].seq_len,
        readout_strategy=readout_strategy,
        input_mode=input_mode,
        bootstrap_after_layers=bootstrap_after,
        intermediate_decrypt_count=backend.stats().decrypt_count
        - started_decrypts
        - len(decrypted_outputs),
        ciphertext_chain=True,
        encrypted_chain=backend.encrypted,
    )


def _validate_recurrence_chain(problems: tuple[OpenFheRecurrenceProblem, ...]) -> None:
    if not problems:
        msg = "problems must not be empty"
        raise ValueError(msg)
    seq_len = problems[0].seq_len
    rank = problems[0].mimo_rank
    d_state = problems[0].d_state
    if seq_len <= 0:
        msg = "recurrence chain problems must have positive seq_len"
        raise ValueError(msg)
    if rank <= 0:
        msg = "recurrence chain problems must have positive mimo_rank"
        raise ValueError(msg)
    for index, problem in enumerate(problems):
        if problem.seq_len != seq_len:
            msg = "all recurrence chain problems must share seq_len"
            raise ValueError(msg)
        if problem.mimo_rank != rank:
            msg = "all recurrence chain problems must share mimo_rank"
            raise ValueError(msg)
        if problem.d_state != d_state:
            msg = "all recurrence chain problems must share d_state"
            raise ValueError(msg)
        if any(len(rank_input) != rank for rank_input in problem.rank_inputs):
            msg = f"problem {index} rank_inputs rows must match mimo_rank={rank}"
            raise ValueError(msg)


def _validate_bootstrap_after_layers(
    bootstrap_after_layers: tuple[int, ...],
    *,
    layer_count: int,
) -> tuple[int, ...]:
    bootstrap_after = tuple(sorted(set(bootstrap_after_layers)))
    if any(layer <= 0 or layer >= layer_count for layer in bootstrap_after):
        msg = f"bootstrap_after_layers must be in [1, {layer_count - 1}]"
        raise ValueError(msg)
    return bootstrap_after


def _plaintext_recurrence_chain_outputs(
    problems: tuple[OpenFheRecurrenceProblem, ...],
) -> tuple[tuple[float, ...], ...]:
    _validate_recurrence_chain(problems)
    rank_inputs = problems[0].rank_inputs
    outputs: tuple[tuple[float, ...], ...] = ()
    for problem in problems:
        bound_problem = replace(problem, rank_inputs=rank_inputs)
        outputs = plaintext_static_recurrence(bound_problem)
        rank_inputs = outputs
    return outputs


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

    plan = _prepare_recurrence_run_plan(
        problem,
        backend=backend,
        input_mode=input_mode,
        bootstrap_every_tokens=bootstrap_every_tokens,
        bootstrap_after_tokens=bootstrap_after_tokens,
        rank_input_ciphertexts=rank_input_ciphertexts,
        output_layout=output_layout,
    )
    d_state = plan.d_state
    rank = plan.rank
    slots = plan.slots
    bootstrap_tokens = plan.bootstrap_tokens

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
    rotations = _output_layout_rotations(
        d_state=d_state,
        rank=rank,
        readout_strategy=readout_strategy,
        output_layout=output_layout,
    )
    layout_contract = CiphertextLayoutContract(
        output_layout=output_layout,
        d_state=d_state,
        mimo_rank=rank,
        readout_strategy=readout_strategy,
        output_slots=output_slots,
        required_rotations=rotations,
    )

    return OpenFheRecurrenceCiphertextTrace(
        output_ciphertexts=_bind_ciphertext_layout(
            tuple(output_ciphertexts),
            layout_contract=layout_contract,
        ),
        output_slots=output_slots,
        output_layout=output_layout,
        layout_contract=layout_contract,
        ring_dimension=backend.ring_dimension,
        batch_size=backend.batch_size,
        multiplicative_depth=multiplicative_depth,
        rotations=rotations,
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
