"""Stage 1 grouped static MIMO recurrence execution helpers."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from math import ceil
from typing import Any

import torch

from fhe_native_mamba3.backends.base import FHEBackend
from fhe_native_mamba3.backends.tracking import TrackingBackend
from fhe_native_mamba3.checkpoint_pre_recurrence import (
    slot_linear_bsgs_rotation_steps,
    slot_linear_ciphertext,
)
from fhe_native_mamba3.layout import ReadoutStrategy
from fhe_native_mamba3.openfhe_backend import (
    InputMode,
    OpenFheRecurrenceProblem,
    plaintext_static_recurrence,
    readout_output_slots,
    required_readout_rotations,
    run_static_mimo_recurrence_ciphertexts_with_backend,
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


@dataclass(frozen=True)
class Stage1GroupedFullLayerLiftSmokeResult:
    """Result for grouped recurrence plus gate/out-projection lifting."""

    stage: str
    measurement_scope: dict[str, Any]
    d_state: int
    mimo_rank: int
    visible_dim: int
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
            "visible_dim": self.visible_dim,
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


def run_stage1_grouped_full_layer_lift_smoke(
    problem: OpenFheRecurrenceProblem,
    *,
    gate_by_token: tuple[tuple[float, ...], ...],
    out_proj_weight: tuple[tuple[float, ...], ...],
    residual_by_token: tuple[tuple[float, ...], ...],
    pack_size: int,
    backend: FHEBackend | None = None,
    multiplicative_depth: int = 8,
    readout_strategy: ReadoutStrategy = "rank-local",
    input_mode: InputMode = "server-bx",
    atol: float = 1e-9,
) -> Stage1GroupedFullLayerLiftSmokeResult:
    """Run grouped recurrence, gated rank output, and visible projection.

    This is the next executable Stage 1 contract after grouped recurrence:
    each rank pack computes its recurrence output, applies the matching gate
    slice, projects through the matching out-projection columns, and the visible
    contributions are summed before adding the residual. It intentionally uses
    synthetic tensors and does not claim checkpoint/full-model correctness.
    """

    if pack_size <= 0:
        msg = "pack_size must be positive"
        raise ValueError(msg)
    if atol < 0:
        msg = "atol must be non-negative"
        raise ValueError(msg)
    _validate_full_layer_lift_inputs(
        problem,
        gate_by_token=gate_by_token,
        out_proj_weight=out_proj_weight,
        residual_by_token=residual_by_token,
    )

    visible_dim = len(out_proj_weight)
    resolved_backend = backend or TrackingBackend(
        batch_size=max(visible_dim, problem.d_state * min(pack_size, problem.mimo_rank), 1)
    )
    visible_cts = tuple(
        resolved_backend.encrypt([0.0] * resolved_backend.batch_size)
        for _ in range(problem.seq_len)
    )
    groups: list[Stage1GroupedRecurrenceGroup] = []
    shared_rotations: set[int] = set()
    for group_index, start_rank in enumerate(range(0, problem.mimo_rank, pack_size)):
        stop_rank = min(start_rank + pack_size, problem.mimo_rank)
        sliced = slice_recurrence_problem_by_rank(
            problem,
            start_rank=start_rank,
            stop_rank=stop_rank,
        )
        trace = run_static_mimo_recurrence_ciphertexts_with_backend(
            sliced,
            backend=resolved_backend,
            multiplicative_depth=multiplicative_depth,
            readout_strategy=readout_strategy,
            input_mode=input_mode,
        )
        projection_rotations = slot_linear_bsgs_rotation_steps(
            source_slots=trace.output_slots,
            output_dim=visible_dim,
        )
        group_rotations = tuple(sorted({*trace.rotations, *projection_rotations}))
        shared_rotations.update(group_rotations)
        groups.append(
            Stage1GroupedRecurrenceGroup(
                group_index=group_index,
                start_rank=start_rank,
                stop_rank=stop_rank,
                pack_size=stop_rank - start_rank,
                rotations=group_rotations,
            )
        )
        weight_slice = torch.tensor(
            [row[start_rank:stop_rank] for row in out_proj_weight],
            dtype=torch.float64,
        )
        for token_index, recurrence_ct in enumerate(trace.output_ciphertexts):
            gate_ct = resolved_backend.encrypt(
                _rank_values_to_slots(
                    gate_by_token[token_index][start_rank:stop_rank],
                    output_slots=trace.output_slots,
                    batch_size=resolved_backend.batch_size,
                )
            )
            gated_ct = resolved_backend.mul_ct(recurrence_ct, gate_ct)
            contribution_ct = slot_linear_ciphertext(
                gated_ct,
                source_slots=trace.output_slots,
                weight=weight_slice,
                bias=[0.0] * visible_dim,
                output_dim=visible_dim,
                backend=resolved_backend,
            )
            visible_cts = _replace_tuple_item(
                visible_cts,
                token_index,
                resolved_backend.add(visible_cts[token_index], contribution_ct),
            )

    output_ciphertexts = tuple(
        resolved_backend.add(
            visible_ct,
            resolved_backend.encrypt(list(residual_by_token[token_index])),
        )
        for token_index, visible_ct in enumerate(visible_cts)
    )
    decrypted_outputs = tuple(
        resolved_backend.decrypt(output_ct, length=visible_dim) for output_ct in output_ciphertexts
    )
    expected_outputs = grouped_full_layer_lift_plaintext(
        problem,
        gate_by_token=gate_by_token,
        out_proj_weight=out_proj_weight,
        residual_by_token=residual_by_token,
    )
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
    return Stage1GroupedFullLayerLiftSmokeResult(
        stage="stage1-grouped-full-layer-lift-smoke",
        measurement_scope={
            "benchmark": bool(resolved_backend.encrypted),
            "encrypted": bool(resolved_backend.encrypted),
            "planning_only": False,
            "exact_math_preserved": True,
            "full_model_correctness_claimed": False,
            "claim": (
                "Grouped full-layer lift smoke: rank-pack recurrence outputs are "
                "gated, projected with matching out-projection columns, summed in "
                "visible slots, and residual is added. This uses synthetic tensors "
                "and does not claim checkpoint/full-model correctness."
            ),
        },
        d_state=problem.d_state,
        mimo_rank=problem.mimo_rank,
        visible_dim=visible_dim,
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


def grouped_full_layer_lift_plaintext(
    problem: OpenFheRecurrenceProblem,
    *,
    gate_by_token: tuple[tuple[float, ...], ...],
    out_proj_weight: tuple[tuple[float, ...], ...],
    residual_by_token: tuple[tuple[float, ...], ...],
) -> tuple[tuple[float, ...], ...]:
    """Plaintext reference for grouped recurrence full-layer lifting."""

    _validate_full_layer_lift_inputs(
        problem,
        gate_by_token=gate_by_token,
        out_proj_weight=out_proj_weight,
        residual_by_token=residual_by_token,
    )
    rank_outputs = plaintext_static_recurrence(problem)
    return tuple(
        tuple(
            residual_by_token[token_index][visible_index]
            + sum(
                out_proj_weight[visible_index][rank_index]
                * rank_outputs[token_index][rank_index]
                * gate_by_token[token_index][rank_index]
                for rank_index in range(problem.mimo_rank)
            )
            for visible_index in range(len(out_proj_weight))
        )
        for token_index in range(problem.seq_len)
    )


def make_demo_full_layer_lift_inputs(
    *,
    seq_len: int,
    mimo_rank: int,
    visible_dim: int,
    seed: int = 23,
) -> tuple[
    tuple[tuple[float, ...], ...],
    tuple[tuple[float, ...], ...],
    tuple[tuple[float, ...], ...],
]:
    """Create deterministic synthetic gate/out-projection/residual tensors."""

    if seq_len <= 0 or mimo_rank <= 0 or visible_dim <= 0:
        msg = "seq_len, mimo_rank, and visible_dim must be positive"
        raise ValueError(msg)
    gate_by_token = tuple(
        tuple(round(0.45 + 0.03 * ((token + rank + seed) % 7), 4) for rank in range(mimo_rank))
        for token in range(seq_len)
    )
    out_proj_weight = tuple(
        tuple(
            round((((visible + 1) * (rank + 3 + seed)) % 17 - 8) / 31.0, 4)
            for rank in range(mimo_rank)
        )
        for visible in range(visible_dim)
    )
    residual_by_token = tuple(
        tuple(
            round((((token + 2) * (visible + 5 + seed)) % 13 - 6) / 29.0, 4)
            for visible in range(visible_dim)
        )
        for token in range(seq_len)
    )
    return gate_by_token, out_proj_weight, residual_by_token


def required_grouped_full_layer_lift_rotations(
    *,
    d_state: int,
    mimo_rank: int,
    pack_size: int,
    visible_dim: int,
    readout_strategy: ReadoutStrategy = "rank-local",
) -> tuple[int, ...]:
    """Shared rotation-key inventory for grouped full-layer lift smoke."""

    if pack_size <= 0:
        msg = "pack_size must be positive"
        raise ValueError(msg)
    rotations: set[int] = set()
    for start_rank in range(0, mimo_rank, pack_size):
        local_rank = min(pack_size, mimo_rank - start_rank)
        output_slots = readout_output_slots(
            d_state=d_state,
            mimo_rank=local_rank,
            readout_strategy=readout_strategy,
        )
        rotations.update(
            required_readout_rotations(
                d_state=d_state,
                mimo_rank=local_rank,
                readout_strategy=readout_strategy,
            )
        )
        rotations.update(
            slot_linear_bsgs_rotation_steps(
                source_slots=output_slots,
                output_dim=visible_dim,
            )
        )
    return tuple(sorted(rotations))


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


def _validate_full_layer_lift_inputs(
    problem: OpenFheRecurrenceProblem,
    *,
    gate_by_token: tuple[tuple[float, ...], ...],
    out_proj_weight: tuple[tuple[float, ...], ...],
    residual_by_token: tuple[tuple[float, ...], ...],
) -> None:
    if len(gate_by_token) != problem.seq_len:
        msg = "gate_by_token length must match seq_len"
        raise ValueError(msg)
    if len(residual_by_token) != problem.seq_len:
        msg = "residual_by_token length must match seq_len"
        raise ValueError(msg)
    if not out_proj_weight:
        msg = "out_proj_weight must not be empty"
        raise ValueError(msg)
    visible_dim = len(out_proj_weight)
    if any(len(row) != problem.mimo_rank for row in gate_by_token):
        msg = "each gate_by_token row must match mimo_rank"
        raise ValueError(msg)
    if any(len(row) != problem.mimo_rank for row in out_proj_weight):
        msg = "each out_proj_weight row must match mimo_rank"
        raise ValueError(msg)
    if any(len(row) != visible_dim for row in residual_by_token):
        msg = "each residual_by_token row must match visible_dim"
        raise ValueError(msg)


def _rank_values_to_slots(
    values: tuple[float, ...],
    *,
    output_slots: tuple[int, ...],
    batch_size: int,
) -> list[float]:
    if len(values) != len(output_slots):
        msg = "values length must match output_slots"
        raise ValueError(msg)
    vector = [0.0] * batch_size
    for output_slot, value in zip(output_slots, values, strict=True):
        vector[output_slot] = float(value)
    return vector


def _replace_tuple_item(values: tuple[Any, ...], index: int, value: Any) -> tuple[Any, ...]:
    items = list(values)
    items[index] = value
    return tuple(items)


__all__ = [
    "Stage1GroupedFullLayerLiftSmokeResult",
    "Stage1GroupedRecurrenceGroup",
    "Stage1GroupedRecurrenceSmokeResult",
    "grouped_full_layer_lift_plaintext",
    "make_demo_full_layer_lift_inputs",
    "required_grouped_full_layer_lift_rotations",
    "run_stage1_grouped_full_layer_lift_smoke",
    "run_stage1_grouped_static_recurrence_smoke",
    "slice_recurrence_problem_by_rank",
]
