"""OpenFHE CKKS backend for the minimal static MIMO recurrence."""

from __future__ import annotations

import random
import time
from dataclasses import asdict, dataclass
from typing import Any

from fhe_native_mamba3.backends.base import FHEBackend
from fhe_native_mamba3.backends.openfhe import OpenFheCkksBackend


@dataclass(frozen=True)
class OpenFheRecurrenceProblem:
    """Plain inputs for an encrypted static MIMO recurrence run."""

    rank_inputs: tuple[tuple[float, ...], ...]
    decay: tuple[float, ...]
    b: tuple[tuple[float, ...], ...]
    c: tuple[tuple[float, ...], ...]

    @property
    def seq_len(self) -> int:
        return len(self.rank_inputs)

    @property
    def d_state(self) -> int:
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
    for rank_input in problem.rank_inputs:
        for n in range(problem.d_state):
            for r in range(problem.mimo_rank):
                state[n][r] = problem.decay[r] * state[n][r] + problem.b[n][r] * rank_input[r]
        outputs.append(
            tuple(
                sum(problem.c[n][r] * state[n][r] for n in range(problem.d_state))
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


def required_readout_rotations(*, d_state: int, mimo_rank: int) -> tuple[int, ...]:
    """Rotations needed to reduce state slots into per-rank output slots."""

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


def _readout_ciphertext(
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


def run_static_mimo_recurrence_with_backend(
    problem: OpenFheRecurrenceProblem,
    *,
    backend: FHEBackend,
    multiplicative_depth: int,
) -> OpenFheRecurrenceResult:
    """Evaluate the encrypted static MIMO recurrence with a backend."""

    d_state = problem.d_state
    rank = problem.mimo_rank
    slots = d_state * rank
    if backend.batch_size < slots:
        msg = f"backend batch_size={backend.batch_size} is smaller than required slots={slots}"
        raise ValueError(msg)

    started = time.perf_counter()
    state_ct = backend.encrypt([0.0] * backend.batch_size)
    decay_pt = backend.encode([problem.decay[r] for r in range(rank) for _ in range(d_state)])
    b_pt = backend.encode(_flat_state_by_rank(problem.b, d_state, rank))
    c_pt = backend.encode(_flat_state_by_rank(problem.c, d_state, rank))

    decrypted_outputs: list[tuple[float, ...]] = []
    for rank_input in problem.rank_inputs:
        input_ct = backend.encrypt(_expanded_rank_input(rank_input, d_state, rank))
        state_ct = backend.add(
            backend.mul_plain(state_ct, decay_pt),
            backend.mul_plain(input_ct, b_pt),
        )
        contrib_ct = backend.mul_plain(state_ct, c_pt)
        output_ct = _readout_ciphertext(
            backend=backend,
            contrib_ct=contrib_ct,
            d_state=d_state,
            rank=rank,
        )
        decrypted_outputs.append(backend.decrypt(output_ct, length=rank))
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
        rotations=required_readout_rotations(d_state=d_state, mimo_rank=rank),
        backend_stats=backend.stats().to_json_dict(),
        latency_sec_per_token=eval_seconds / problem.seq_len,
    )


def run_openfhe_static_recurrence(
    problem: OpenFheRecurrenceProblem,
    *,
    multiplicative_depth: int | None = None,
    scaling_mod_size: int = 50,
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
        ),
    )
    return run_static_mimo_recurrence_with_backend(
        problem,
        backend=backend,
        multiplicative_depth=depth,
    )
