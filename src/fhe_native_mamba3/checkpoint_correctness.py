"""Correctness gates for checkpoint-derived encrypted recurrence paths."""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
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
    MambaSourceVisibleHandoffTensors,
    build_mamba_source_recurrence_problem,
    build_mamba_source_visible_handoff_tensors,
    compare_mamba_layer_reference,
)
from fhe_native_mamba3.openfhe_backend import (
    InputMode,
    readout_output_slots,
    required_readout_rotations,
    run_static_mimo_recurrence_ciphertexts_with_backend,
    run_static_mimo_recurrence_with_backend,
)


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


@dataclass(frozen=True)
class CheckpointFullLayerCiphertextGate:
    """Pass/fail gate for checkpoint-derived full visible-layer arithmetic."""

    layer_index: int
    d_model: int
    checked_visible_dim: int
    d_state: int
    mimo_rank: int
    seq_len: int
    backend: str
    encrypted: bool
    input_mode: str
    readout_strategy: str
    max_abs_error: float
    atol: float
    passed: bool
    recurrence_ciphertext: bool
    visible_handoff_ciphertext: bool
    no_intermediate_decrypt: bool
    full_layer_formula_checked: bool
    official_mamba_parity: bool
    full_model_correctness_claimed: bool
    plaintext_precomputed_stages: tuple[str, ...]
    backend_stats: dict[str, Any]
    notes: tuple[str, ...]

    def to_json_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["plaintext_precomputed_stages"] = list(self.plaintext_precomputed_stages)
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


def run_checkpoint_full_layer_ciphertext_gate(
    state_dict: dict[str, Tensor],
    layer_input: Tensor,
    *,
    layer_index: int = 0,
    d_state: int | None = None,
    mimo_rank: int | None = None,
    backend: FHEBackend | None = None,
    input_mode: InputMode = "encrypted-dynamic-bc",
    readout_strategy: ReadoutStrategy = "rank-local",
    multiplicative_depth: int = 12,
    atol: float = 1e-6,
    norm_eps: float = 1e-5,
    visible_dim_limit: int | None = None,
) -> CheckpointFullLayerCiphertextGate:
    """Check source-style full-layer output through encrypted rank handoff.

    This is still a Stage 0 gate: RMSNorm, convolution, dynamic B/C, decay, and
    gate values are produced by the transparent PyTorch source-style reference.
    The encrypted path covers recurrence, skip addition, gate multiply,
    out-projection, and residual addition, then decrypts only final visible
    token outputs.
    """

    if atol < 0:
        msg = "atol must be non-negative"
        raise ValueError(msg)
    problem = build_mamba_source_recurrence_problem(
        state_dict,
        layer_input,
        layer_index=layer_index,
        d_state=d_state,
        mimo_rank=mimo_rank,
        norm_eps=norm_eps,
    )
    visible = build_mamba_source_visible_handoff_tensors(
        state_dict,
        layer_input,
        layer_index=layer_index,
        d_state=problem.d_state,
        mimo_rank=problem.mimo_rank,
        norm_eps=norm_eps,
    )
    checked_visible_dim = _resolve_visible_dim_limit(
        d_model=visible.d_model,
        visible_dim_limit=visible_dim_limit,
    )
    batch_size = max(problem.d_state * problem.mimo_rank, checked_visible_dim)
    resolved_backend = backend or TrackingBackend(batch_size=batch_size)
    if resolved_backend.batch_size < batch_size:
        msg = (
            "full-layer ciphertext gate backend batch_size must cover recurrence and visible "
            f"slots; need at least {batch_size}, got {resolved_backend.batch_size}"
        )
        raise ValueError(msg)

    started_decrypts = resolved_backend.stats().decrypt_count
    trace = run_static_mimo_recurrence_ciphertexts_with_backend(
        replace(problem, d_skip=None),
        backend=resolved_backend,
        multiplicative_depth=multiplicative_depth,
        readout_strategy=readout_strategy,
        input_mode=input_mode,
    )
    decrypted_outputs = []
    for token_index, recurrence_ct in enumerate(trace.output_ciphertexts):
        final_ct = _visible_output_ciphertext(
            backend=resolved_backend,
            recurrence_ct=recurrence_ct,
            output_slots=trace.output_slots,
            visible=visible,
            checked_visible_dim=checked_visible_dim,
            token_index=token_index,
        )
        decrypted_outputs.append(resolved_backend.decrypt(final_ct, length=checked_visible_dim))

    expected_rows = tuple(
        tuple(
            float(value)
            for value in visible.expected_final_output[0, token_index, :checked_visible_dim]
            .detach()
            .cpu()
        )
        for token_index in range(visible.seq_len)
    )
    actual_rows = tuple(tuple(row) for row in decrypted_outputs)
    max_abs_error = max(
        (
            abs(actual - expected)
            for actual_row, expected_row in zip(actual_rows, expected_rows, strict=True)
            for actual, expected in zip(actual_row, expected_row, strict=True)
        ),
        default=0.0,
    )
    no_intermediate_decrypt = (
        resolved_backend.stats().decrypt_count - started_decrypts == visible.seq_len
    )
    return CheckpointFullLayerCiphertextGate(
        layer_index=layer_index,
        d_model=visible.d_model,
        checked_visible_dim=checked_visible_dim,
        d_state=problem.d_state,
        mimo_rank=problem.mimo_rank,
        seq_len=problem.seq_len,
        backend=resolved_backend.stats().backend,
        encrypted=bool(resolved_backend.stats().encrypted),
        input_mode=input_mode,
        readout_strategy=readout_strategy,
        max_abs_error=max_abs_error,
        atol=atol,
        passed=max_abs_error <= atol and no_intermediate_decrypt,
        recurrence_ciphertext=True,
        visible_handoff_ciphertext=True,
        no_intermediate_decrypt=no_intermediate_decrypt,
        full_layer_formula_checked=True,
        official_mamba_parity=False,
        full_model_correctness_claimed=False,
        plaintext_precomputed_stages=(
            "rms_norm",
            "causal_conv_silu",
            "dynamic_b",
            "dynamic_c",
            "state_rank_decay",
            "gate_values",
        ),
        backend_stats=resolved_backend.stats().to_json_dict(),
        notes=(
            "checks source-style full-layer visible output, not official fused kernel parity",
            "input-dependent pre-recurrence tensors are still plaintext-precomputed in Stage 0",
            "visible output may be truncated when visible_dim_limit is set",
        ),
    )


def _visible_output_ciphertext(
    *,
    backend: FHEBackend,
    recurrence_ct: Any,
    output_slots: tuple[int, ...],
    visible: MambaSourceVisibleHandoffTensors,
    checked_visible_dim: int,
    token_index: int,
) -> Any:
    rank_ct = backend.add(
        recurrence_ct,
        backend.encrypt(
            _rank_slot_vector(
                visible.skip_update[0, token_index],
                output_slots=output_slots,
                batch_size=backend.batch_size,
            )
        ),
    )
    gated_ct = backend.mul_ct(
        rank_ct,
        backend.encrypt(
            _rank_slot_vector(
                visible.gate[0, token_index],
                output_slots=output_slots,
                batch_size=backend.batch_size,
            )
        ),
    )
    projected_ct = _project_rank_slots_to_visible(
        backend=backend,
        rank_ct=gated_ct,
        output_slots=output_slots,
        out_proj_weight=visible.out_proj_weight,
        checked_visible_dim=checked_visible_dim,
    )
    return backend.add(
        projected_ct,
        backend.encrypt(
            [
                float(value)
                for value in visible.residual[0, token_index, :checked_visible_dim].detach().cpu()
            ]
        ),
    )


def _rank_slot_vector(
    values: Tensor,
    *,
    output_slots: tuple[int, ...],
    batch_size: int,
) -> list[float]:
    if len(output_slots) != int(values.numel()):
        msg = "output_slots length must match rank values"
        raise ValueError(msg)
    vector = [0.0] * batch_size
    for slot, value in zip(output_slots, values.detach().cpu(), strict=True):
        vector[slot] = float(value)
    return vector


def _project_rank_slots_to_visible(
    *,
    backend: FHEBackend,
    rank_ct: Any,
    output_slots: tuple[int, ...],
    out_proj_weight: Tensor,
    checked_visible_dim: int,
) -> Any:
    if backend.batch_size < checked_visible_dim:
        msg = (
            f"backend batch_size={backend.batch_size} is smaller than "
            f"checked_visible_dim={checked_visible_dim}"
        )
        raise ValueError(msg)
    if int(out_proj_weight.shape[0]) < checked_visible_dim:
        msg = "out_proj_weight first dimension must cover checked_visible_dim"
        raise ValueError(msg)
    if int(out_proj_weight.shape[1]) != len(output_slots):
        msg = "out_proj_weight second dimension must match recurrence rank"
        raise ValueError(msg)

    output_ct = backend.encrypt([0.0] * backend.batch_size)
    weights = out_proj_weight.detach().cpu()
    for visible_index in range(checked_visible_dim):
        for rank_index, source in enumerate(output_slots):
            weight = float(weights[visible_index, rank_index])
            if weight == 0.0:
                continue
            mask = [0.0] * backend.batch_size
            mask[source] = weight
            term = backend.mul_plain(rank_ct, backend.encode(mask))
            shift = source - visible_index
            if shift:
                term = backend.rotate(term, shift)
            output_ct = backend.add(output_ct, term)
    return output_ct


def required_full_layer_visible_rotations(
    *,
    d_model: int,
    d_state: int,
    mimo_rank: int,
    readout_strategy: ReadoutStrategy = "rank-local",
    visible_dim_limit: int | None = None,
) -> tuple[int, ...]:
    """Rotations for recurrence readout plus rank-to-visible projection."""

    if d_model <= 0:
        msg = "d_model must be positive"
        raise ValueError(msg)
    checked_visible_dim = _resolve_visible_dim_limit(
        d_model=d_model,
        visible_dim_limit=visible_dim_limit,
    )
    output_slots = readout_output_slots(
        d_state=d_state,
        mimo_rank=mimo_rank,
        readout_strategy=readout_strategy,
    )
    rotations = set(
        required_readout_rotations(
            d_state=d_state,
            mimo_rank=mimo_rank,
            readout_strategy=readout_strategy,
        )
    )
    for visible_index in range(checked_visible_dim):
        for source in output_slots:
            shift = source - visible_index
            if shift:
                rotations.add(shift)
    return tuple(sorted(rotations))


def _resolve_visible_dim_limit(*, d_model: int, visible_dim_limit: int | None) -> int:
    if d_model <= 0:
        msg = "d_model must be positive"
        raise ValueError(msg)
    if visible_dim_limit is None:
        return d_model
    if visible_dim_limit <= 0:
        msg = "visible_dim_limit must be positive"
        raise ValueError(msg)
    return min(d_model, visible_dim_limit)


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
