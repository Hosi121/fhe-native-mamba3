"""Layer-sweep harness for checkpoint-derived full-layer ciphertext gates."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import asdict, dataclass
from typing import Any

import torch
from torch import Tensor

from fhe_native_mamba3.backends.base import FHEBackend
from fhe_native_mamba3.backends.tracking import TrackingBackend
from fhe_native_mamba3.checkpoint_correctness import (
    CheckpointFullLayerCiphertextGate,
    required_full_layer_visible_rotations,
    run_checkpoint_full_layer_ciphertext_gate,
)
from fhe_native_mamba3.layout import ReadoutStrategy
from fhe_native_mamba3.mamba_checkpoint import plan_mamba_checkpoint
from fhe_native_mamba3.mamba_reference import run_mamba_source_layer
from fhe_native_mamba3.openfhe_backend import InputMode

BackendFactory = Callable[[int, tuple[int, ...]], FHEBackend]


@dataclass(frozen=True)
class CheckpointFullLayerSweepLayer:
    """One layer result in a checkpoint full-layer ciphertext sweep."""

    layer_index: int
    passed: bool
    max_abs_error: float
    d_model: int
    checked_visible_dim: int
    d_state: int
    mimo_rank: int
    seq_len: int
    backend: str
    encrypted: bool
    rotation_key_count: int
    operation_counts: dict[str, int]
    timing: dict[str, float]
    plaintext_precomputed_stages: tuple[str, ...]

    def to_json_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["plaintext_precomputed_stages"] = list(self.plaintext_precomputed_stages)
        return payload


@dataclass(frozen=True)
class CheckpointFullLayerSweepResult:
    """Summary for a source-propagated checkpoint full-layer sweep."""

    layer_count: int
    seq_len: int
    d_model: int
    d_state: int
    mimo_rank: int
    input_mode: str
    readout_strategy: str
    passed: bool
    max_abs_error_max: float
    failing_layers: tuple[int, ...]
    layers: tuple[CheckpointFullLayerSweepLayer, ...]
    measurement_scope: dict[str, Any]

    def to_json_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["failing_layers"] = list(self.failing_layers)
        payload["layers"] = [layer.to_json_dict() for layer in self.layers]
        return payload


def run_checkpoint_full_layer_ciphertext_sweep(
    state_dict: dict[str, Tensor],
    initial_layer_input: Tensor,
    *,
    layer_count: int | None = None,
    d_state: int | None = None,
    mimo_rank: int | None = None,
    backend_factory: BackendFactory | None = None,
    input_mode: InputMode = "encrypted-dynamic-bc",
    readout_strategy: ReadoutStrategy = "rank-local",
    multiplicative_depth: int = 12,
    atol: float = 1e-6,
    norm_eps: float = 1e-5,
    visible_dim_limit: int | None = None,
) -> CheckpointFullLayerSweepResult:
    """Sweep source-style full visible-layer gates over consecutive layers.

    This is a Stage 0 coverage harness, not a full encrypted model execution:
    the visible output ciphertext of each checked layer is compared at the layer
    boundary, and the next layer input is propagated with the transparent
    source-style PyTorch formula.
    """

    _validate_initial_layer_input(initial_layer_input)
    plan = plan_mamba_checkpoint(state_dict)
    resolved_d_state = d_state if d_state is not None else plan.inferred_d_state
    resolved_rank = mimo_rank if mimo_rank is not None else plan.inferred_mimo_rank
    if resolved_d_state is None or resolved_rank is None:
        msg = "d_state and mimo_rank must be provided when they cannot be inferred"
        raise ValueError(msg)
    if resolved_d_state <= 0 or resolved_rank <= 0:
        msg = "d_state and mimo_rank must be positive"
        raise ValueError(msg)
    resolved_layer_count = layer_count if layer_count is not None else plan.complete_layer_count
    if resolved_layer_count <= 0:
        msg = "layer_count must be positive"
        raise ValueError(msg)
    if resolved_layer_count > len(plan.layers):
        msg = (
            f"layer_count={resolved_layer_count} exceeds checkpoint layer count {len(plan.layers)}"
        )
        raise ValueError(msg)

    factory = backend_factory or _tracking_backend_factory
    x = initial_layer_input
    layer_results: list[CheckpointFullLayerSweepLayer] = []
    with torch.inference_mode():
        for layer_index in range(resolved_layer_count):
            rotations = required_full_layer_visible_rotations(
                d_model=int(x.shape[-1]),
                d_state=resolved_d_state,
                mimo_rank=resolved_rank,
                readout_strategy=readout_strategy,
                visible_dim_limit=visible_dim_limit,
            )
            backend = factory(
                max(
                    resolved_d_state * resolved_rank,
                    _resolve_checked_visible_dim(int(x.shape[-1]), visible_dim_limit),
                ),
                rotations,
            )
            gate = run_checkpoint_full_layer_ciphertext_gate(
                state_dict,
                x,
                layer_index=layer_index,
                d_state=resolved_d_state,
                mimo_rank=resolved_rank,
                backend=backend,
                input_mode=input_mode,
                readout_strategy=readout_strategy,
                multiplicative_depth=multiplicative_depth,
                atol=atol,
                norm_eps=norm_eps,
                visible_dim_limit=visible_dim_limit,
            )
            layer_results.append(_sweep_layer_from_gate(gate, rotation_key_count=len(rotations)))
            if layer_index + 1 < resolved_layer_count:
                x = run_mamba_source_layer(
                    state_dict,
                    x,
                    layer_index=layer_index,
                    d_state=resolved_d_state,
                    mimo_rank=resolved_rank,
                    norm_eps=norm_eps,
                )

    failing_layers = tuple(layer.layer_index for layer in layer_results if not layer.passed)
    max_abs_error_max = max((layer.max_abs_error for layer in layer_results), default=0.0)
    return CheckpointFullLayerSweepResult(
        layer_count=resolved_layer_count,
        seq_len=int(initial_layer_input.shape[1]),
        d_model=int(initial_layer_input.shape[-1]),
        d_state=resolved_d_state,
        mimo_rank=resolved_rank,
        input_mode=input_mode,
        readout_strategy=readout_strategy,
        passed=not failing_layers,
        max_abs_error_max=max_abs_error_max,
        failing_layers=failing_layers,
        layers=tuple(layer_results),
        measurement_scope={
            "source_style_full_layer_formula": True,
            "official_mamba_parity": False,
            "full_model_correctness_claimed": False,
            "inter_layer_ciphertext_handoff": False,
            "layer_inputs_plaintext_propagated": True,
            "visible_dim_limit": visible_dim_limit,
            "claim": (
                "per-layer source-style full visible output sweep; next-layer "
                "inputs are propagated in plaintext between checked layers"
            ),
        },
    )


def _tracking_backend_factory(batch_size: int, rotations: tuple[int, ...]) -> FHEBackend:
    del rotations
    return TrackingBackend(batch_size=batch_size)


def _sweep_layer_from_gate(
    gate: CheckpointFullLayerCiphertextGate,
    *,
    rotation_key_count: int,
) -> CheckpointFullLayerSweepLayer:
    stats = gate.backend_stats
    return CheckpointFullLayerSweepLayer(
        layer_index=gate.layer_index,
        passed=gate.passed,
        max_abs_error=gate.max_abs_error,
        d_model=gate.d_model,
        checked_visible_dim=gate.checked_visible_dim,
        d_state=gate.d_state,
        mimo_rank=gate.mimo_rank,
        seq_len=gate.seq_len,
        backend=gate.backend,
        encrypted=gate.encrypted,
        rotation_key_count=rotation_key_count,
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
        plaintext_precomputed_stages=gate.plaintext_precomputed_stages,
    )


def _validate_initial_layer_input(initial_layer_input: Tensor) -> None:
    if initial_layer_input.ndim != 3:
        msg = "initial_layer_input must have shape [batch, seq_len, d_model]"
        raise ValueError(msg)
    if int(initial_layer_input.shape[0]) != 1:
        msg = "checkpoint full-layer sweep currently supports batch size 1"
        raise ValueError(msg)
    if int(initial_layer_input.shape[1]) <= 0:
        msg = "seq_len must be positive"
        raise ValueError(msg)
    if int(initial_layer_input.shape[2]) <= 0:
        msg = "d_model must be positive"
        raise ValueError(msg)


def _resolve_checked_visible_dim(d_model: int, visible_dim_limit: int | None) -> int:
    if visible_dim_limit is None:
        return d_model
    if visible_dim_limit <= 0:
        msg = "visible_dim_limit must be positive"
        raise ValueError(msg)
    return min(d_model, visible_dim_limit)
