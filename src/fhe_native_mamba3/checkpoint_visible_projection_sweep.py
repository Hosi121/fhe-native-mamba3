"""Visible-projection scaling sweeps for checkpoint full-layer gates."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import asdict, dataclass
from typing import Any, Literal

import torch
from torch import Tensor

from fhe_native_mamba3.backends.base import FHEBackend
from fhe_native_mamba3.backends.tracking import TrackingBackend
from fhe_native_mamba3.checkpoint_correctness import (
    required_full_layer_visible_rotations,
    run_checkpoint_full_layer_ciphertext_gate,
)
from fhe_native_mamba3.layout import ReadoutStrategy
from fhe_native_mamba3.mamba_checkpoint import plan_mamba_checkpoint
from fhe_native_mamba3.mamba_reference import run_mamba_source_layer
from fhe_native_mamba3.openfhe_backend import InputMode

ProjectionSweepStatus = Literal["passed", "failed", "skipped", "error"]
ProjectionBackendFactory = Callable[[int, tuple[int, ...]], FHEBackend]


@dataclass(frozen=True)
class CheckpointVisibleProjectionSweepRow:
    """One visible-dimension row in an OpenFHE/checkpoint projection sweep."""

    requested_visible_dim: int | None
    checked_visible_dim: int
    d_model: int
    full_visible_output: bool
    rotation_key_count: int
    status: ProjectionSweepStatus
    passed: bool
    reason: str
    backend: str | None = None
    encrypted: bool | None = None
    max_abs_error: float | None = None
    operation_counts: dict[str, int] | None = None
    timing: dict[str, float] | None = None
    ring_dimension: int | None = None
    batch_size: int | None = None

    def to_json_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class CheckpointVisibleProjectionSweepResult:
    """Summary for visible projection scaling over one checkpoint layer."""

    layer_index: int
    seq_len: int
    d_model: int
    d_state: int
    mimo_rank: int
    input_mode: str
    readout_strategy: str
    row_count: int
    passed_count: int
    failed_count: int
    skipped_count: int
    error_count: int
    max_checked_visible_dim_passed: int | None
    first_non_passed_visible_dim: int | None
    bottleneck: str
    rows: tuple[CheckpointVisibleProjectionSweepRow, ...]
    measurement_scope: dict[str, Any]

    @property
    def passed(self) -> bool:
        return self.failed_count == 0 and self.error_count == 0

    def to_json_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["passed"] = self.passed
        payload["rows"] = [row.to_json_dict() for row in self.rows]
        return payload


def run_checkpoint_visible_projection_sweep(
    state_dict: dict[str, Tensor],
    initial_layer_input: Tensor,
    *,
    visible_dim_limits: tuple[int | None, ...],
    layer_index: int = 0,
    d_state: int | None = None,
    mimo_rank: int | None = None,
    backend_factory: ProjectionBackendFactory | None = None,
    max_rotation_keys: int | None = None,
    input_mode: InputMode = "encrypted-dynamic-bc",
    readout_strategy: ReadoutStrategy = "rank-local",
    multiplicative_depth: int = 16,
    atol: float = 1e-6,
    norm_eps: float = 1e-5,
) -> CheckpointVisibleProjectionSweepResult:
    """Measure how visible output width changes full-layer gate cost/failure."""

    _validate_layer_input(initial_layer_input)
    if not visible_dim_limits:
        msg = "visible_dim_limits must not be empty"
        raise ValueError(msg)
    if max_rotation_keys is not None and max_rotation_keys <= 0:
        msg = "max_rotation_keys must be positive when provided"
        raise ValueError(msg)

    plan = plan_mamba_checkpoint(state_dict)
    if layer_index >= len(plan.layers):
        msg = f"layer_index {layer_index} is not present in the state_dict"
        raise ValueError(msg)
    resolved_d_state = d_state if d_state is not None else plan.inferred_d_state
    resolved_rank = mimo_rank if mimo_rank is not None else plan.inferred_mimo_rank
    if resolved_d_state is None or resolved_rank is None:
        msg = "d_state and mimo_rank must be provided when they cannot be inferred"
        raise ValueError(msg)
    if resolved_d_state <= 0 or resolved_rank <= 0:
        msg = "d_state and mimo_rank must be positive"
        raise ValueError(msg)

    x = _source_propagate_to_layer(
        state_dict,
        initial_layer_input,
        layer_index=layer_index,
        d_state=resolved_d_state,
        mimo_rank=resolved_rank,
        norm_eps=norm_eps,
    )
    d_model = int(x.shape[-1])
    factory = backend_factory or _tracking_backend_factory
    rows = tuple(
        _run_projection_row(
            state_dict,
            x,
            requested_visible_dim=requested,
            layer_index=layer_index,
            d_model=d_model,
            d_state=resolved_d_state,
            mimo_rank=resolved_rank,
            backend_factory=factory,
            max_rotation_keys=max_rotation_keys,
            input_mode=input_mode,
            readout_strategy=readout_strategy,
            multiplicative_depth=multiplicative_depth,
            atol=atol,
            norm_eps=norm_eps,
        )
        for requested in visible_dim_limits
    )
    return _summarize_projection_sweep(
        rows,
        layer_index=layer_index,
        seq_len=int(x.shape[1]),
        d_model=d_model,
        d_state=resolved_d_state,
        mimo_rank=resolved_rank,
        input_mode=input_mode,
        readout_strategy=readout_strategy,
    )


def _run_projection_row(
    state_dict: dict[str, Tensor],
    layer_input: Tensor,
    *,
    requested_visible_dim: int | None,
    layer_index: int,
    d_model: int,
    d_state: int,
    mimo_rank: int,
    backend_factory: ProjectionBackendFactory,
    max_rotation_keys: int | None,
    input_mode: InputMode,
    readout_strategy: ReadoutStrategy,
    multiplicative_depth: int,
    atol: float,
    norm_eps: float,
) -> CheckpointVisibleProjectionSweepRow:
    checked_visible_dim = _resolve_checked_visible_dim(
        d_model=d_model,
        visible_dim_limit=requested_visible_dim,
    )
    rotations = required_full_layer_visible_rotations(
        d_model=d_model,
        d_state=d_state,
        mimo_rank=mimo_rank,
        readout_strategy=readout_strategy,
        visible_dim_limit=requested_visible_dim,
    )
    if max_rotation_keys is not None and len(rotations) > max_rotation_keys:
        return CheckpointVisibleProjectionSweepRow(
            requested_visible_dim=requested_visible_dim,
            checked_visible_dim=checked_visible_dim,
            d_model=d_model,
            full_visible_output=checked_visible_dim == d_model,
            rotation_key_count=len(rotations),
            status="skipped",
            passed=False,
            reason=(
                f"rotation_key_count={len(rotations)} exceeds max_rotation_keys={max_rotation_keys}"
            ),
        )

    try:
        backend = backend_factory(max(d_state * mimo_rank, checked_visible_dim), rotations)
        gate = run_checkpoint_full_layer_ciphertext_gate(
            state_dict,
            layer_input,
            layer_index=layer_index,
            d_state=d_state,
            mimo_rank=mimo_rank,
            backend=backend,
            input_mode=input_mode,
            readout_strategy=readout_strategy,
            multiplicative_depth=multiplicative_depth,
            atol=atol,
            norm_eps=norm_eps,
            visible_dim_limit=requested_visible_dim,
        )
    except Exception as exc:
        return CheckpointVisibleProjectionSweepRow(
            requested_visible_dim=requested_visible_dim,
            checked_visible_dim=checked_visible_dim,
            d_model=d_model,
            full_visible_output=checked_visible_dim == d_model,
            rotation_key_count=len(rotations),
            status="error",
            passed=False,
            reason=f"{type(exc).__name__}: {exc}",
        )

    stats = gate.backend_stats
    return CheckpointVisibleProjectionSweepRow(
        requested_visible_dim=requested_visible_dim,
        checked_visible_dim=checked_visible_dim,
        d_model=d_model,
        full_visible_output=checked_visible_dim == d_model,
        rotation_key_count=len(rotations),
        status="passed" if gate.passed else "failed",
        passed=gate.passed,
        reason="" if gate.passed else f"max_abs_error={gate.max_abs_error} > atol={gate.atol}",
        backend=gate.backend,
        encrypted=gate.encrypted,
        max_abs_error=gate.max_abs_error,
        operation_counts={
            "ct_ct_mul": int(stats["ct_ct_mul_count"]),
            "ct_pt_mul": int(stats["ct_pt_mul_count"]),
            "add": int(stats["add_count"]),
            "rotations": int(stats["rotation_count"]),
            "bootstraps": int(stats["bootstrap_count"]),
            "encrypt": int(stats["encrypt_count"]),
            "decrypt": int(stats["decrypt_count"]),
            "encode": int(stats["encode_count"]),
        },
        timing={
            "setup_seconds": float(stats["setup_seconds"]),
            "eval_seconds": float(stats["eval_seconds"]),
        },
        ring_dimension=int(backend.ring_dimension),
        batch_size=int(backend.batch_size),
    )


def _summarize_projection_sweep(
    rows: tuple[CheckpointVisibleProjectionSweepRow, ...],
    *,
    layer_index: int,
    seq_len: int,
    d_model: int,
    d_state: int,
    mimo_rank: int,
    input_mode: str,
    readout_strategy: str,
) -> CheckpointVisibleProjectionSweepResult:
    passed_rows = tuple(row for row in rows if row.status == "passed")
    first_non_passed = next((row for row in rows if row.status != "passed"), None)
    return CheckpointVisibleProjectionSweepResult(
        layer_index=layer_index,
        seq_len=seq_len,
        d_model=d_model,
        d_state=d_state,
        mimo_rank=mimo_rank,
        input_mode=input_mode,
        readout_strategy=readout_strategy,
        row_count=len(rows),
        passed_count=sum(1 for row in rows if row.status == "passed"),
        failed_count=sum(1 for row in rows if row.status == "failed"),
        skipped_count=sum(1 for row in rows if row.status == "skipped"),
        error_count=sum(1 for row in rows if row.status == "error"),
        max_checked_visible_dim_passed=max(
            (row.checked_visible_dim for row in passed_rows),
            default=None,
        ),
        first_non_passed_visible_dim=(
            first_non_passed.checked_visible_dim if first_non_passed is not None else None
        ),
        bottleneck=_infer_bottleneck(rows),
        rows=rows,
        measurement_scope={
            "source_style_full_layer_formula": True,
            "official_mamba_parity": False,
            "full_model_correctness_claimed": False,
            "pre_recurrence_stages_plaintext_precomputed": True,
            "claim": (
                "visible-projection scaling for the source-style checkpoint full-layer gate; "
                "not a full encrypted model claim"
            ),
        },
    )


def _infer_bottleneck(rows: tuple[CheckpointVisibleProjectionSweepRow, ...]) -> str:
    first = next((row for row in rows if row.status != "passed"), None)
    if first is None:
        return "none_observed"
    if first.status == "skipped" and "rotation_key_count" in first.reason:
        return "rotation_key_guard"
    if first.status == "failed":
        return "accuracy"
    if "ring dimension" in first.reason.lower() or "he standards" in first.reason.lower():
        return "ckks_ring_dimension"
    return "runtime_error"


def _source_propagate_to_layer(
    state_dict: dict[str, Tensor],
    initial_layer_input: Tensor,
    *,
    layer_index: int,
    d_state: int,
    mimo_rank: int,
    norm_eps: float,
) -> Tensor:
    x = initial_layer_input
    with torch.inference_mode():
        for current_layer in range(layer_index):
            x = run_mamba_source_layer(
                state_dict,
                x,
                layer_index=current_layer,
                d_state=d_state,
                mimo_rank=mimo_rank,
                norm_eps=norm_eps,
            )
    return x


def _tracking_backend_factory(batch_size: int, rotations: tuple[int, ...]) -> FHEBackend:
    del rotations
    return TrackingBackend(batch_size=batch_size)


def _resolve_checked_visible_dim(*, d_model: int, visible_dim_limit: int | None) -> int:
    if d_model <= 0:
        msg = "d_model must be positive"
        raise ValueError(msg)
    if visible_dim_limit is None:
        return d_model
    if visible_dim_limit <= 0:
        msg = "visible_dim_limit must be positive"
        raise ValueError(msg)
    return min(d_model, visible_dim_limit)


def _validate_layer_input(layer_input: Tensor) -> None:
    if layer_input.ndim != 3:
        msg = "initial_layer_input must have shape [batch, seq_len, d_model]"
        raise ValueError(msg)
    if int(layer_input.shape[0]) != 1:
        msg = "visible projection sweep currently supports batch size 1"
        raise ValueError(msg)
    if int(layer_input.shape[1]) <= 0:
        msg = "seq_len must be positive"
        raise ValueError(msg)
    if int(layer_input.shape[2]) <= 0:
        msg = "d_model must be positive"
        raise ValueError(msg)
