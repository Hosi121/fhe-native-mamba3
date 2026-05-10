"""Correctness gates for checkpoint-derived encrypted recurrence paths."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from torch import Tensor

from fhe_native_mamba3.backends.base import FHEBackend
from fhe_native_mamba3.backends.tracking import TrackingBackend
from fhe_native_mamba3.ciphertext_handoff import (
    CiphertextHandoffLayer,
    matrix_to_cyclic_diagonals,
    required_handoff_rotations,
    run_ciphertext_handoff_chain,
)
from fhe_native_mamba3.layout import ReadoutStrategy
from fhe_native_mamba3.mamba_checkpoint import plan_mamba_checkpoint
from fhe_native_mamba3.mamba_reference import (
    build_mamba_source_recurrence_problem,
    compare_mamba_layer_reference,
)
from fhe_native_mamba3.openfhe_backend import InputMode, run_static_mimo_recurrence_with_backend


@dataclass(frozen=True)
class CheckpointRecurrenceCorrectnessGate:
    """Pass/fail gate for one checkpoint-derived recurrence layer."""

    layer_index: int
    d_state: int
    mimo_rank: int
    seq_len: int
    backend: str
    encrypted: bool
    input_mode: str
    readout_strategy: str
    recurrence_max_abs_error: float
    recurrence_atol: float
    recurrence_passed: bool
    reference_max_exact_stage_error: float | None
    reference_atol: float | None
    reference_passed: bool | None
    visible_handoff_checked: bool
    visible_handoff_passed: bool | None
    visible_handoff_max_abs_error: float | None
    visible_handoff_metadata: dict[str, Any]
    full_layer_correctness_claimed: bool
    passed: bool
    backend_stats: dict[str, Any]
    notes: tuple[str, ...]

    def to_json_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["notes"] = list(self.notes)
        return payload


def run_checkpoint_recurrence_correctness_gate(
    state_dict: dict[str, Tensor],
    layer_input: Tensor,
    *,
    layer_index: int = 0,
    d_state: int | None = None,
    mimo_rank: int | None = None,
    backend: FHEBackend | None = None,
    input_mode: InputMode = "encrypted-dynamic-bc",
    readout_strategy: ReadoutStrategy = "rank-local",
    multiplicative_depth: int = 8,
    recurrence_atol: float = 1e-8,
    reference_atol: float | None = 1e-8,
    include_reference_gate: bool = True,
    include_visible_handoff_gate: bool = False,
    visible_handoff_backend: FHEBackend | None = None,
) -> CheckpointRecurrenceCorrectnessGate:
    """Run a one-layer checkpoint recurrence gate against a backend.

    The recurrence problem is extracted from the source-style Mamba layer:
    RMSNorm, SiLU causal convolution, token-dependent B/C, and token-dependent
    state-rank decay are evaluated in plaintext to produce the encrypted
    recurrence inputs. The backend result is then compared against the plaintext
    recurrence reference used by the same problem.
    """

    if recurrence_atol < 0:
        msg = "recurrence_atol must be non-negative"
        raise ValueError(msg)
    if reference_atol is not None and reference_atol < 0:
        msg = "reference_atol must be non-negative"
        raise ValueError(msg)
    problem = build_mamba_source_recurrence_problem(
        state_dict,
        layer_input,
        layer_index=layer_index,
        d_state=d_state,
        mimo_rank=mimo_rank,
    )
    resolved_backend = backend or TrackingBackend(batch_size=problem.d_state * problem.mimo_rank)
    result = run_static_mimo_recurrence_with_backend(
        problem,
        backend=resolved_backend,
        multiplicative_depth=multiplicative_depth,
        readout_strategy=readout_strategy,
        input_mode=input_mode,
    )

    reference_max_exact_stage_error: float | None = None
    reference_passed: bool | None = None
    if include_reference_gate:
        reference = compare_mamba_layer_reference(
            state_dict,
            layer_input,
            layer_index=layer_index,
            d_state=problem.d_state,
            mimo_rank=problem.mimo_rank,
        )
        exact_errors = (
            reference.projected_rank_input_max_abs_error,
            reference.causal_conv_output_max_abs_error,
            reference.dt_hidden_max_abs_error or 0.0,
            reference.dt_max_abs_error or 0.0,
            reference.decay_by_token_max_abs_error or 0.0,
            reference.recurrence_rank_output_max_abs_error,
        )
        reference_max_exact_stage_error = max(exact_errors, default=0.0)
        reference_passed = (
            True if reference_atol is None else reference_max_exact_stage_error <= reference_atol
        )

    recurrence_passed = result.max_abs_error <= recurrence_atol
    visible_handoff_passed: bool | None = None
    visible_handoff_max_abs_error: float | None = None
    visible_handoff_metadata: dict[str, Any] = {}
    notes: list[str] = [
        "recurrence gate compares backend output to the checkpoint-derived "
        "plaintext recurrence problem",
    ]
    if include_visible_handoff_gate:
        visible = _validate_visible_handoff_readiness(
            state_dict,
            layer_input,
            layer_index=layer_index,
            d_state=problem.d_state,
            mimo_rank=problem.mimo_rank,
            backend=visible_handoff_backend,
        )
        visible_handoff_passed = bool(visible["passed"])
        visible_handoff_max_abs_error = visible["max_abs_error"]
        visible_handoff_metadata = visible["metadata"]
        notes.append(
            "visible handoff gate validates gate/out-projection/residual shape metadata "
            "and a handoff helper probe only"
        )
    else:
        notes.append(
            "full gate/out-projection/residual ciphertext handoff remains a separate Stage 0 item"
        )

    passed = (
        recurrence_passed
        and (reference_passed is not False)
        and (visible_handoff_passed is not False)
    )
    return CheckpointRecurrenceCorrectnessGate(
        layer_index=layer_index,
        d_state=problem.d_state,
        mimo_rank=problem.mimo_rank,
        seq_len=problem.seq_len,
        backend=result.backend_stats["backend"],
        encrypted=bool(result.backend_stats["encrypted"]),
        input_mode=input_mode,
        readout_strategy=readout_strategy,
        recurrence_max_abs_error=result.max_abs_error,
        recurrence_atol=recurrence_atol,
        recurrence_passed=recurrence_passed,
        reference_max_exact_stage_error=reference_max_exact_stage_error,
        reference_atol=reference_atol if include_reference_gate else None,
        reference_passed=reference_passed,
        visible_handoff_checked=include_visible_handoff_gate,
        visible_handoff_passed=visible_handoff_passed,
        visible_handoff_max_abs_error=visible_handoff_max_abs_error,
        visible_handoff_metadata=visible_handoff_metadata,
        full_layer_correctness_claimed=False,
        passed=passed,
        backend_stats=result.backend_stats,
        notes=tuple(notes),
    )


def _validate_visible_handoff_readiness(
    state_dict: dict[str, Tensor],
    layer_input: Tensor,
    *,
    layer_index: int,
    d_state: int,
    mimo_rank: int,
    backend: FHEBackend | None,
) -> dict[str, Any]:
    if layer_input.ndim != 3:
        msg = "layer_input must have shape [batch, seq_len, d_model]"
        raise ValueError(msg)

    plan = plan_mamba_checkpoint(state_dict)
    if layer_index >= len(plan.layers):
        msg = f"layer_index {layer_index} is not present in the state_dict"
        raise ValueError(msg)
    layer = plan.layers[layer_index]
    d_model = int(layer_input.shape[-1])
    batch = int(layer_input.shape[0])
    seq_len = int(layer_input.shape[1])

    in_proj_shape = (
        tuple(int(dim) for dim in state_dict[layer.in_proj_key].shape)
        if layer.in_proj_key is not None
        else None
    )
    out_proj_shape = (
        tuple(int(dim) for dim in state_dict[layer.out_proj_key].shape)
        if layer.out_proj_key is not None
        else None
    )
    gate_ready = (
        in_proj_shape is not None
        and len(in_proj_shape) == 2
        and in_proj_shape[0] >= (2 * mimo_rank)
    )
    out_projection_ready = (
        out_proj_shape is not None
        and len(out_proj_shape) == 2
        and out_proj_shape[0] >= d_model
        and out_proj_shape[1] >= mimo_rank
    )
    residual_ready = batch > 0 and seq_len > 0 and d_model > 0
    readiness = {
        "gate": gate_ready,
        "out_projection": out_projection_ready,
        "residual": residual_ready,
    }

    metadata: dict[str, Any] = {
        "source": "checkpoint-visible-output-handoff",
        "visible_width": d_model,
        "recurrence_width": mimo_rank,
        "d_state": d_state,
        "seq_len": seq_len,
        "residual_shape": [batch, seq_len, d_model],
        "gate_shape": [batch, seq_len, mimo_rank] if gate_ready else None,
        "out_projection_shape": list(out_proj_shape) if out_proj_shape is not None else None,
        "required_rotations": list(required_handoff_rotations(d_model)) if d_model > 0 else [],
        "readiness": readiness,
        "ready_for_gate_out_residual": all(readiness.values()),
        "full_layer_correctness_claimed": False,
        "claim": (
            "shape/metadata readiness only; recurrence correctness does not imply "
            "full gate/out-projection/residual correctness"
        ),
    }
    if not metadata["ready_for_gate_out_residual"]:
        missing = tuple(name for name, ready in readiness.items() if not ready)
        metadata["missing"] = list(missing)
        return {"passed": False, "max_abs_error": None, "metadata": metadata}

    handoff_backend = backend or TrackingBackend(batch_size=d_model)
    if handoff_backend.batch_size != d_model:
        msg = (
            "visible handoff backend batch_size must equal visible width "
            f"{d_model}; got {handoff_backend.batch_size}"
        )
        raise ValueError(msg)
    zero_update = tuple(tuple(0.0 for _ in range(d_model)) for _ in range(d_model))
    handoff = run_ciphertext_handoff_chain(
        backend=handoff_backend,
        input_values=tuple(float(value) for value in layer_input[0, 0].detach().cpu().tolist()),
        layers=(CiphertextHandoffLayer(matrix_to_cyclic_diagonals(zero_update)),),
    )
    metadata["handoff_backend_stats"] = handoff.backend_stats
    return {
        "passed": handoff.max_abs_error == 0.0,
        "max_abs_error": handoff.max_abs_error,
        "metadata": metadata,
    }
