"""Stage 1 grouped static MIMO recurrence execution helpers."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from math import ceil
from typing import Any

from fhe_native_mamba3.backends.base import FHEBackend
from fhe_native_mamba3.backends.tracking import TrackingBackend
from fhe_native_mamba3.layout import ReadoutStrategy
from fhe_native_mamba3.openfhe_backend import (
    InputMode,
    OpenFheRecurrenceProblem,
    plaintext_static_recurrence,
    run_static_mimo_recurrence_with_backend,
)


@dataclass(frozen=True)
class Stage1GroupedRecurrenceGroup:
    """One rank-pack recurrence group."""

    group_index: int
    start_rank: int
    stop_rank: int
    pack_size: int
    rotations: tuple[int, ...]

    def to_json_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class Stage1GroupedRecurrenceSmokeResult:
    """Result for a grouped static MIMO recurrence smoke."""

    stage: str
    measurement_scope: dict[str, Any]
    d_state: int
    mimo_rank: int
    pack_size: int
    group_count: int
    seq_len: int
    backend: str
    encrypted: bool
    readout_strategy: ReadoutStrategy
    input_mode: InputMode
    shared_rotations: tuple[int, ...]
    group_rotation_counts: tuple[int, ...]
    max_abs_error: float
    atol: float
    passed: bool
    decrypted_outputs: tuple[tuple[float, ...], ...]
    expected_outputs: tuple[tuple[float, ...], ...]
    groups: tuple[Stage1GroupedRecurrenceGroup, ...]
    backend_stats: dict[str, Any]

    def to_json_dict(self) -> dict[str, Any]:
        return {
            "stage": self.stage,
            "measurement_scope": dict(self.measurement_scope),
            "d_state": self.d_state,
            "mimo_rank": self.mimo_rank,
            "pack_size": self.pack_size,
            "group_count": self.group_count,
            "seq_len": self.seq_len,
            "backend": self.backend,
            "encrypted": self.encrypted,
            "readout_strategy": self.readout_strategy,
            "input_mode": self.input_mode,
            "shared_rotations": list(self.shared_rotations),
            "rotation_count": len(self.shared_rotations),
            "group_rotation_counts": list(self.group_rotation_counts),
            "max_abs_error": self.max_abs_error,
            "atol": self.atol,
            "passed": self.passed,
            "decrypted_outputs": [list(row) for row in self.decrypted_outputs],
            "expected_outputs": [list(row) for row in self.expected_outputs],
            "groups": [group.to_json_dict() for group in self.groups],
            "backend_stats": dict(self.backend_stats),
        }


def slice_recurrence_problem_by_rank(
    problem: OpenFheRecurrenceProblem,
    *,
    start_rank: int,
    stop_rank: int,
) -> OpenFheRecurrenceProblem:
    """Return a rank-local slice of a static MIMO recurrence problem."""

    _validate_rank_slice(problem, start_rank=start_rank, stop_rank=stop_rank)
    return OpenFheRecurrenceProblem(
        rank_inputs=tuple(row[start_rank:stop_rank] for row in problem.rank_inputs),
        decay=problem.decay[start_rank:stop_rank],
        decay_by_token=(
            None
            if problem.decay_by_token is None
            else tuple(row[start_rank:stop_rank] for row in problem.decay_by_token)
        ),
        decay_state_by_token=_slice_token_matrices(
            problem.decay_state_by_token,
            start_rank=start_rank,
            stop_rank=stop_rank,
        ),
        b=_slice_matrix(problem.b, start_rank=start_rank, stop_rank=stop_rank),
        c=_slice_matrix(problem.c, start_rank=start_rank, stop_rank=stop_rank),
        b_by_token=_slice_token_matrices(
            problem.b_by_token,
            start_rank=start_rank,
            stop_rank=stop_rank,
        ),
        c_by_token=_slice_token_matrices(
            problem.c_by_token,
            start_rank=start_rank,
            stop_rank=stop_rank,
        ),
        d_skip=(None if problem.d_skip is None else problem.d_skip[start_rank:stop_rank]),
    )


def run_stage1_grouped_static_recurrence_smoke(
    problem: OpenFheRecurrenceProblem,
    *,
    pack_size: int,
    backend: FHEBackend | None = None,
    multiplicative_depth: int = 8,
    readout_strategy: ReadoutStrategy = "rank-local",
    input_mode: InputMode = "server-bx",
    atol: float = 1e-9,
) -> Stage1GroupedRecurrenceSmokeResult:
    """Run a rank-packed grouped recurrence and compare to monolithic plaintext.

    Groups are independent rank slices of the same recurrence. This helper
    decrypts each group only for the final comparison, so it proves the exact
    rank grouping contract without claiming a full Mamba layer or speedup.
    """

    if pack_size <= 0:
        msg = "pack_size must be positive"
        raise ValueError(msg)
    if atol < 0:
        msg = "atol must be non-negative"
        raise ValueError(msg)

    resolved_backend = backend or TrackingBackend(
        batch_size=max(problem.d_state * min(pack_size, problem.mimo_rank), 1)
    )
    actual_rows = [[0.0 for _ in range(problem.mimo_rank)] for _ in range(problem.seq_len)]
    groups: list[Stage1GroupedRecurrenceGroup] = []
    shared_rotations: set[int] = set()
    for group_index, start_rank in enumerate(range(0, problem.mimo_rank, pack_size)):
        stop_rank = min(start_rank + pack_size, problem.mimo_rank)
        sliced = slice_recurrence_problem_by_rank(
            problem,
            start_rank=start_rank,
            stop_rank=stop_rank,
        )
        group_result = run_static_mimo_recurrence_with_backend(
            sliced,
            backend=resolved_backend,
            multiplicative_depth=multiplicative_depth,
            readout_strategy=readout_strategy,
            input_mode=input_mode,
        )
        shared_rotations.update(group_result.rotations)
        groups.append(
            Stage1GroupedRecurrenceGroup(
                group_index=group_index,
                start_rank=start_rank,
                stop_rank=stop_rank,
                pack_size=stop_rank - start_rank,
                rotations=group_result.rotations,
            )
        )
        for token_index, row in enumerate(group_result.decrypted_outputs):
            actual_rows[token_index][start_rank:stop_rank] = row

    decrypted_outputs = tuple(tuple(row) for row in actual_rows)
    expected_outputs = plaintext_static_recurrence(problem)
    max_abs_error = max(
        (
            abs(actual - expected)
            for actual_row, expected_row in zip(
                decrypted_outputs,
                expected_outputs,
                strict=True,
            )
            for actual, expected in zip(actual_row, expected_row, strict=True)
        ),
        default=0.0,
    )
    return Stage1GroupedRecurrenceSmokeResult(
        stage="stage1-grouped-static-recurrence-smoke",
        measurement_scope={
            "benchmark": bool(resolved_backend.encrypted),
            "encrypted": bool(resolved_backend.encrypted),
            "planning_only": False,
            "exact_math_preserved": True,
            "full_model_correctness_claimed": False,
            "claim": (
                "Grouped static MIMO recurrence smoke: rank slices are evaluated "
                "independently and reassembled at the final rank-output boundary; "
                "this does not include full Mamba gate/out-projection/residual."
            ),
        },
        d_state=problem.d_state,
        mimo_rank=problem.mimo_rank,
        pack_size=pack_size,
        group_count=ceil(problem.mimo_rank / pack_size),
        seq_len=problem.seq_len,
        backend=resolved_backend.stats().backend,
        encrypted=bool(resolved_backend.encrypted),
        readout_strategy=readout_strategy,
        input_mode=input_mode,
        shared_rotations=tuple(sorted(shared_rotations)),
        group_rotation_counts=tuple(len(group.rotations) for group in groups),
        max_abs_error=max_abs_error,
        atol=atol,
        passed=max_abs_error <= atol,
        decrypted_outputs=decrypted_outputs,
        expected_outputs=expected_outputs,
        groups=tuple(groups),
        backend_stats=resolved_backend.stats().to_json_dict(),
    )


def _validate_rank_slice(
    problem: OpenFheRecurrenceProblem,
    *,
    start_rank: int,
    stop_rank: int,
) -> None:
    if start_rank < 0 or stop_rank > problem.mimo_rank or start_rank >= stop_rank:
        msg = (
            "rank slice must satisfy 0 <= start_rank < stop_rank <= mimo_rank; "
            f"got start_rank={start_rank}, stop_rank={stop_rank}, "
            f"mimo_rank={problem.mimo_rank}"
        )
        raise ValueError(msg)


def _slice_matrix(
    matrix: tuple[tuple[float, ...], ...],
    *,
    start_rank: int,
    stop_rank: int,
) -> tuple[tuple[float, ...], ...]:
    return tuple(row[start_rank:stop_rank] for row in matrix)


def _slice_token_matrices(
    matrices: tuple[tuple[tuple[float, ...], ...], ...] | None,
    *,
    start_rank: int,
    stop_rank: int,
) -> tuple[tuple[tuple[float, ...], ...], ...] | None:
    if matrices is None:
        return None
    return tuple(
        _slice_matrix(matrix, start_rank=start_rank, stop_rank=stop_rank) for matrix in matrices
    )


__all__ = [
    "Stage1GroupedRecurrenceGroup",
    "Stage1GroupedRecurrenceSmokeResult",
    "run_stage1_grouped_static_recurrence_smoke",
    "slice_recurrence_problem_by_rank",
]
