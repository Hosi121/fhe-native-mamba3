"""Correctness gates for checkpoint-derived encrypted recurrence paths."""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from typing import Any

from torch import Tensor

from fhe_native_mamba3.backends.base import FHEBackend
from fhe_native_mamba3.backends.tracking import TrackingBackend
from fhe_native_mamba3.checkpoint_pre_recurrence import (
    RmsNormMode,
    StateDecayMode,
    run_checkpoint_pre_recurrence_ciphertexts_with_backend,
    slot_linear_bsgs_baby_step,
    slot_linear_bsgs_rotation_steps,
    slot_linear_ciphertext,
)
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
    run_mamba_source_layer,
)
from fhe_native_mamba3.openfhe_backend import (
    CiphertextLayoutContract,
    InputMode,
    LayoutBoundCiphertexts,
    plaintext_static_recurrence,
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
    full_visible_output_checked: bool
    partial_visible_output_checked: bool
    d_state: int
    mimo_rank: int
    seq_len: int
    backend: str
    encrypted: bool
    input_mode: str
    readout_strategy: str
    visible_output_scale: float
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
    pre_recurrence_ciphertext: bool = False
    pre_recurrence_depth_estimate: int | None = None

    def to_json_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["plaintext_precomputed_stages"] = list(self.plaintext_precomputed_stages)
        payload["notes"] = list(self.notes)
        return payload


@dataclass(frozen=True)
class CheckpointFullLayerCiphertextTrace:
    """Ciphertext trace for source-style full visible-layer arithmetic."""

    layer_index: int
    d_model: int
    checked_visible_dim: int
    full_visible_output_checked: bool
    partial_visible_output_checked: bool
    d_state: int
    mimo_rank: int
    seq_len: int
    backend: str
    encrypted: bool
    input_mode: str
    readout_strategy: str
    visible_output_scale: float
    output_layout: str
    output_slots: tuple[int, ...]
    layout_contract: CiphertextLayoutContract
    required_rotations: tuple[int, ...]
    output_ciphertexts: tuple[Any, ...]
    expected_outputs: tuple[tuple[float, ...], ...]
    backend_handle: FHEBackend
    recurrence_ciphertext: bool
    visible_handoff_ciphertext: bool
    decrypt_count_delta: int
    plaintext_precomputed_stages: tuple[str, ...]
    backend_stats: dict[str, Any]
    notes: tuple[str, ...]
    pre_recurrence_depth_estimate: int | None = None

    def to_json_dict(self) -> dict[str, Any]:
        return {
            "layer_index": self.layer_index,
            "d_model": self.d_model,
            "checked_visible_dim": self.checked_visible_dim,
            "full_visible_output_checked": self.full_visible_output_checked,
            "partial_visible_output_checked": self.partial_visible_output_checked,
            "d_state": self.d_state,
            "mimo_rank": self.mimo_rank,
            "seq_len": self.seq_len,
            "backend": self.backend,
            "encrypted": self.encrypted,
            "input_mode": self.input_mode,
            "readout_strategy": self.readout_strategy,
            "visible_output_scale": self.visible_output_scale,
            "output_layout": self.output_layout,
            "output_slots": list(self.output_slots),
            "layout_contract": {
                "output_layout": self.layout_contract.output_layout,
                "d_state": self.layout_contract.d_state,
                "mimo_rank": self.layout_contract.mimo_rank,
                "readout_strategy": self.layout_contract.readout_strategy,
                "output_slots": list(self.layout_contract.output_slots),
                "required_rotations": list(self.layout_contract.required_rotations),
            },
            "required_rotations": list(self.required_rotations),
            "output_ciphertext_count": len(self.output_ciphertexts),
            "expected_outputs": [list(row) for row in self.expected_outputs],
            "recurrence_ciphertext": self.recurrence_ciphertext,
            "visible_handoff_ciphertext": self.visible_handoff_ciphertext,
            "decrypt_count_delta": self.decrypt_count_delta,
            "plaintext_precomputed_stages": list(self.plaintext_precomputed_stages),
            "pre_recurrence_depth_estimate": self.pre_recurrence_depth_estimate,
            "backend_stats": self.backend_stats,
            "notes": list(self.notes),
        }


@dataclass(frozen=True)
class CheckpointEncryptedPreRecurrenceRecurrenceGate:
    """Gate for encrypted pre-recurrence outputs feeding encrypted recurrence."""

    layer_index: int
    d_state: int
    mimo_rank: int
    seq_len: int
    backend: str
    encrypted: bool
    input_mode: str
    readout_strategy: str
    pre_recurrence_ciphertext: bool
    recurrence_ciphertext: bool
    no_intermediate_decrypt: bool
    max_abs_error: float
    atol: float
    passed: bool
    pre_recurrence_depth_estimate: int
    backend_stats: dict[str, Any]
    notes: tuple[str, ...]

    def to_json_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["notes"] = list(self.notes)
        return payload


@dataclass(frozen=True)
class CheckpointFullLayerCiphertextChainGate:
    """Pass/fail gate for encrypted full-layer outputs chained across layers."""

    layer_count: int
    d_model: int
    checked_visible_dim: int
    full_visible_output_checked: bool
    partial_visible_output_checked: bool
    d_state: int
    mimo_rank: int
    seq_len: int
    backend: str
    encrypted: bool
    readout_strategy: str
    max_abs_error: float
    atol: float
    passed: bool
    inter_layer_ciphertext_handoff: bool
    no_intermediate_decrypt: bool
    final_decrypt_count: int
    plaintext_precomputed_stages: tuple[str, ...]
    layer_depth_estimates: tuple[int | None, ...]
    backend_stats: dict[str, Any]
    notes: tuple[str, ...]

    def to_json_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["plaintext_precomputed_stages"] = list(self.plaintext_precomputed_stages)
        payload["layer_depth_estimates"] = list(self.layer_depth_estimates)
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


def run_checkpoint_encrypted_pre_recurrence_recurrence_gate(
    state_dict: dict[str, Tensor],
    layer_input: Tensor,
    *,
    layer_index: int = 0,
    d_state: int | None = None,
    mimo_rank: int | None = None,
    backend: FHEBackend | None = None,
    readout_strategy: ReadoutStrategy = "rank-local",
    multiplicative_depth: int = 28,
    norm_eps: float = 1e-5,
    polynomial_degree: int = 13,
    polynomial_range: float = 6.0,
    rms_norm_mode: RmsNormMode = "newton-invsqrt",
    newton_iterations: int = 2,
    newton_range: tuple[float, float] = (0.25, 0.5),
    state_decay_mode: StateDecayMode = "poly-composed",
    decay_polynomial_degree: int = 5,
    decay_polynomial_range: tuple[float, float] = (-0.5, 0.5),
    atol: float = 2e-2,
) -> CheckpointEncryptedPreRecurrenceRecurrenceGate:
    """Feed encrypted pre-recurrence stage outputs into encrypted recurrence."""

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
    batch_size = max(problem.d_state * problem.mimo_rank, problem.mimo_rank)
    resolved_backend = backend or TrackingBackend(batch_size=batch_size)
    if resolved_backend.batch_size < batch_size:
        msg = (
            "encrypted pre-recurrence recurrence gate backend batch_size is too small; "
            f"need at least {batch_size}, got {resolved_backend.batch_size}"
        )
        raise ValueError(msg)

    started_decrypts = resolved_backend.stats().decrypt_count
    pre_trace = run_checkpoint_pre_recurrence_ciphertexts_with_backend(
        state_dict,
        layer_input,
        layer_index=layer_index,
        d_state=problem.d_state,
        mimo_rank=problem.mimo_rank,
        backend=resolved_backend,
        norm_eps=norm_eps,
        polynomial_degree=polynomial_degree,
        polynomial_range=polynomial_range,
        rms_norm_mode=rms_norm_mode,
        newton_iterations=newton_iterations,
        newton_range=newton_range,
        state_decay_mode=state_decay_mode,
        decay_polynomial_degree=decay_polynomial_degree,
        decay_polynomial_range=decay_polynomial_range,
        atol=atol,
    )
    rank_input_ciphertexts = _bind_expanded_rank_input_ciphertexts(
        tuple(
            _expand_rank_ciphertext_to_state_slots(
                ciphertext,
                d_state=problem.d_state,
                rank=problem.mimo_rank,
                backend=resolved_backend,
            )
            for ciphertext in pre_trace.causal_conv_post_silu_ciphertexts
        ),
        d_state=problem.d_state,
        rank=problem.mimo_rank,
        readout_strategy=readout_strategy,
    )
    b_ciphertexts = tuple(
        _expand_state_vector_ciphertext_to_state_slots(
            ciphertext,
            d_state=problem.d_state,
            rank=problem.mimo_rank,
            backend=resolved_backend,
        )
        for ciphertext in pre_trace.dynamic_b_ciphertexts
    )
    c_ciphertexts = tuple(
        _expand_state_vector_ciphertext_to_state_slots(
            ciphertext,
            d_state=problem.d_state,
            rank=problem.mimo_rank,
            backend=resolved_backend,
        )
        for ciphertext in pre_trace.dynamic_c_ciphertexts
    )
    recurrence_problem = replace(problem, d_skip=None)
    recurrence_trace = run_static_mimo_recurrence_ciphertexts_with_backend(
        recurrence_problem,
        backend=resolved_backend,
        multiplicative_depth=multiplicative_depth,
        readout_strategy=readout_strategy,
        input_mode="encrypted-dynamic-bc",
        rank_input_ciphertexts=rank_input_ciphertexts,
        b_ciphertexts=b_ciphertexts,
        c_ciphertexts=c_ciphertexts,
        decay_state_ciphertexts=pre_trace.state_rank_decay_ciphertexts,
    )
    intermediate_decrypts = resolved_backend.stats().decrypt_count - started_decrypts
    actual_rows = tuple(
        tuple(decrypted[slot] for slot in recurrence_trace.output_slots)
        for decrypted in (
            resolved_backend.decrypt(output_ct, length=resolved_backend.batch_size)
            for output_ct in recurrence_trace.output_ciphertexts
        )
    )
    expected_rows = plaintext_static_recurrence(recurrence_problem)
    max_abs_error = max(
        (
            abs(actual - expected)
            for actual_row, expected_row in zip(actual_rows, expected_rows, strict=True)
            for actual, expected in zip(actual_row, expected_row, strict=True)
        ),
        default=0.0,
    )
    final_decrypts = resolved_backend.stats().decrypt_count - started_decrypts
    no_intermediate_decrypt = (
        intermediate_decrypts == 0 and final_decrypts == recurrence_problem.seq_len
    )
    return CheckpointEncryptedPreRecurrenceRecurrenceGate(
        layer_index=layer_index,
        d_state=problem.d_state,
        mimo_rank=problem.mimo_rank,
        seq_len=problem.seq_len,
        backend=resolved_backend.stats().backend,
        encrypted=bool(resolved_backend.stats().encrypted),
        input_mode="encrypted-dynamic-bc",
        readout_strategy=readout_strategy,
        pre_recurrence_ciphertext=True,
        recurrence_ciphertext=True,
        no_intermediate_decrypt=no_intermediate_decrypt,
        max_abs_error=max_abs_error,
        atol=atol,
        passed=max_abs_error <= atol and no_intermediate_decrypt,
        pre_recurrence_depth_estimate=pre_trace.depth_estimate,
        backend_stats=resolved_backend.stats().to_json_dict(),
        notes=(
            "encrypted pre-recurrence ciphertexts feed encrypted recurrence",
            "gate decrypts only final recurrence readout ciphertexts",
            "visible out-projection, residual, and final lm_head are not included",
        ),
    )


def run_checkpoint_encrypted_pre_recurrence_full_layer_ciphertexts_with_backend(
    state_dict: dict[str, Tensor],
    layer_input: Tensor,
    *,
    layer_index: int = 0,
    d_state: int | None = None,
    mimo_rank: int | None = None,
    backend: FHEBackend | None = None,
    readout_strategy: ReadoutStrategy = "rank-local",
    multiplicative_depth: int = 28,
    norm_eps: float = 1e-5,
    polynomial_degree: int = 13,
    polynomial_range: float = 6.0,
    rms_norm_mode: RmsNormMode = "newton-invsqrt",
    newton_iterations: int = 2,
    newton_range: tuple[float, float] = (0.25, 0.5),
    state_decay_mode: StateDecayMode = "poly-composed",
    decay_polynomial_degree: int = 5,
    decay_polynomial_range: tuple[float, float] = (-0.5, 0.5),
    visible_dim_limit: int | None = None,
    visible_output_scale: float = 1.0,
    input_ciphertexts: tuple[Any, ...] | None = None,
    allow_partial_input_ciphertext_handoff: bool = False,
) -> CheckpointFullLayerCiphertextTrace:
    """Return encrypted pre-recurrence full-layer visible outputs as ciphertexts."""

    if visible_output_scale <= 0:
        msg = "visible_output_scale must be positive"
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
    batch_size = max(visible.d_model, problem.d_state * problem.mimo_rank, checked_visible_dim)
    resolved_backend = backend or TrackingBackend(batch_size=batch_size)
    if resolved_backend.batch_size < batch_size:
        msg = (
            "encrypted pre-recurrence full-layer trace backend batch_size is too small; "
            f"need at least {batch_size}, got {resolved_backend.batch_size}"
        )
        raise ValueError(msg)
    if input_ciphertexts is not None:
        _validate_visible_input_ciphertexts(
            input_ciphertexts,
            d_model=visible.d_model,
            seq_len=visible.seq_len,
        )
        if checked_visible_dim != visible.d_model and not allow_partial_input_ciphertext_handoff:
            msg = "input ciphertext handoff requires full visible output, not visible_dim_limit"
            raise ValueError(msg)

    started_decrypts = resolved_backend.stats().decrypt_count
    pre_trace = run_checkpoint_pre_recurrence_ciphertexts_with_backend(
        state_dict,
        layer_input,
        layer_index=layer_index,
        d_state=problem.d_state,
        mimo_rank=problem.mimo_rank,
        backend=resolved_backend,
        norm_eps=norm_eps,
        polynomial_degree=polynomial_degree,
        polynomial_range=polynomial_range,
        rms_norm_mode=rms_norm_mode,
        newton_iterations=newton_iterations,
        newton_range=newton_range,
        state_decay_mode=state_decay_mode,
        decay_polynomial_degree=decay_polynomial_degree,
        decay_polynomial_range=decay_polynomial_range,
        input_ciphertexts=input_ciphertexts,
    )
    rank_input_ciphertexts = _bind_expanded_rank_input_ciphertexts(
        tuple(
            _expand_rank_ciphertext_to_state_slots(
                ciphertext,
                d_state=problem.d_state,
                rank=problem.mimo_rank,
                backend=resolved_backend,
            )
            for ciphertext in pre_trace.causal_conv_post_silu_ciphertexts
        ),
        d_state=problem.d_state,
        rank=problem.mimo_rank,
        readout_strategy=readout_strategy,
    )
    b_ciphertexts = tuple(
        _expand_state_vector_ciphertext_to_state_slots(
            ciphertext,
            d_state=problem.d_state,
            rank=problem.mimo_rank,
            backend=resolved_backend,
        )
        for ciphertext in pre_trace.dynamic_b_ciphertexts
    )
    c_ciphertexts = tuple(
        _expand_state_vector_ciphertext_to_state_slots(
            ciphertext,
            d_state=problem.d_state,
            rank=problem.mimo_rank,
            backend=resolved_backend,
        )
        for ciphertext in pre_trace.dynamic_c_ciphertexts
    )
    recurrence_trace = run_static_mimo_recurrence_ciphertexts_with_backend(
        replace(problem, d_skip=None),
        backend=resolved_backend,
        multiplicative_depth=multiplicative_depth,
        readout_strategy=readout_strategy,
        input_mode="encrypted-dynamic-bc",
        rank_input_ciphertexts=rank_input_ciphertexts,
        b_ciphertexts=b_ciphertexts,
        c_ciphertexts=c_ciphertexts,
        decay_state_ciphertexts=pre_trace.state_rank_decay_ciphertexts,
    )
    output_ciphertexts = tuple(
        _encrypted_pre_recurrence_visible_output_ciphertext(
            backend=resolved_backend,
            recurrence_ct=recurrence_ct,
            rank_input_ct=rank_input_ct,
            gate_ct=gate_ct,
            output_slots=recurrence_trace.output_slots,
            d_skip=problem.d_skip,
            visible=visible,
            checked_visible_dim=checked_visible_dim,
            token_index=token_index,
            visible_output_scale=visible_output_scale,
            residual_ct=_visible_residual_ciphertext_for_checked_dim(
                backend=resolved_backend,
                ciphertexts=input_ciphertexts,
                token_index=token_index,
                checked_visible_dim=checked_visible_dim,
            ),
        )
        for token_index, (recurrence_ct, rank_input_ct, gate_ct) in enumerate(
            zip(
                recurrence_trace.output_ciphertexts,
                pre_trace.causal_conv_post_silu_ciphertexts,
                pre_trace.gate_post_silu_ciphertexts,
                strict=True,
            )
        )
    )
    expected_rows = tuple(
        tuple(
            visible_output_scale * float(value)
            for value in visible.expected_final_output[0, token_index, :checked_visible_dim]
            .detach()
            .cpu()
        )
        for token_index in range(visible.seq_len)
    )
    output_slots = tuple(range(checked_visible_dim))
    required_rotations = required_full_layer_visible_rotations(
        d_model=visible.d_model,
        d_state=problem.d_state,
        mimo_rank=problem.mimo_rank,
        readout_strategy=readout_strategy,
        visible_dim_limit=visible_dim_limit,
    )
    layout_contract = CiphertextLayoutContract(
        output_layout="visible-output",
        d_state=problem.d_state,
        mimo_rank=problem.mimo_rank,
        readout_strategy=readout_strategy,
        output_slots=output_slots,
        required_rotations=required_rotations,
    )
    plaintext_precomputed_stages = []
    if input_ciphertexts is None:
        plaintext_precomputed_stages.append("residual_input")
    if rms_norm_mode == "plaintext-exact":
        plaintext_precomputed_stages.append("rms_norm_output")
    if state_decay_mode == "plaintext-exact":
        plaintext_precomputed_stages.append("state_rank_decay")
    notes = [
        "encrypted pre-recurrence ciphertexts feed recurrence and visible projection",
        "gate and skip input come from encrypted pre-recurrence ciphertexts",
    ]
    if input_ciphertexts is None:
        notes.append("residual input is encrypted per token by this trace constructor")
    else:
        notes.append("residual input is reused from caller-provided visible input ciphertexts")
    if visible_output_scale != 1.0:
        notes.append("visible output ciphertext and expected output are scaled before decoding")
    return CheckpointFullLayerCiphertextTrace(
        layer_index=layer_index,
        d_model=visible.d_model,
        checked_visible_dim=checked_visible_dim,
        full_visible_output_checked=checked_visible_dim == visible.d_model,
        partial_visible_output_checked=checked_visible_dim < visible.d_model,
        d_state=problem.d_state,
        mimo_rank=problem.mimo_rank,
        seq_len=visible.seq_len,
        backend=resolved_backend.stats().backend,
        encrypted=bool(resolved_backend.stats().encrypted),
        input_mode="encrypted-dynamic-bc",
        readout_strategy=readout_strategy,
        visible_output_scale=visible_output_scale,
        output_layout="visible-output",
        output_slots=output_slots,
        layout_contract=layout_contract,
        required_rotations=required_rotations,
        output_ciphertexts=LayoutBoundCiphertexts(
            output_ciphertexts,
            layout_contract=layout_contract,
        ),
        expected_outputs=expected_rows,
        backend_handle=resolved_backend,
        recurrence_ciphertext=True,
        visible_handoff_ciphertext=True,
        decrypt_count_delta=resolved_backend.stats().decrypt_count - started_decrypts,
        plaintext_precomputed_stages=tuple(plaintext_precomputed_stages),
        backend_stats=resolved_backend.stats().to_json_dict(),
        notes=tuple(notes),
        pre_recurrence_depth_estimate=pre_trace.depth_estimate,
    )


def run_checkpoint_encrypted_pre_recurrence_partial_visible_chain_proxy(
    state_dict: dict[str, Tensor],
    layer_input: Tensor,
    *,
    layer_count: int,
    visible_dim_limit: int,
    d_state: int | None = None,
    mimo_rank: int | None = None,
    backend: FHEBackend | None = None,
    readout_strategy: ReadoutStrategy = "rank-local",
    multiplicative_depth: int = 28,
    norm_eps: float = 1e-5,
    polynomial_degree: int = 13,
    polynomial_range: float = 6.0,
    rms_norm_mode: RmsNormMode = "newton-invsqrt",
    newton_iterations: int = 2,
    newton_range: tuple[float, float] = (0.25, 0.5),
    state_decay_mode: StateDecayMode = "poly-composed",
    decay_polynomial_degree: int = 5,
    decay_polynomial_range: tuple[float, float] = (-0.5, 0.5),
    atol: float = 5e-2,
) -> CheckpointFullLayerCiphertextChainGate:
    """Chain the first K visible dimensions and inject plaintext remainder.

    This is an explicit Stage 0 proxy for real-checkpoint handoff. It does not
    claim a full visible-output chain because dimensions ``K:d_model`` are
    supplied from the plaintext reference after every layer.
    """

    if layer_count <= 0:
        msg = "layer_count must be positive"
        raise ValueError(msg)
    if layer_input.ndim != 3 or layer_input.shape[0] != 1:
        msg = "layer_input must have shape [1, seq_len, d_model]"
        raise ValueError(msg)
    if atol < 0:
        msg = "atol must be non-negative"
        raise ValueError(msg)
    if rms_norm_mode == "plaintext-exact":
        msg = (
            "encrypted partial-visible chain proxy cannot use plaintext-exact RMSNorm "
            "because each layer consumes visible-output ciphertexts"
        )
        raise ValueError(msg)
    d_model = int(layer_input.shape[-1])
    checked_visible_dim = _resolve_visible_dim_limit(
        d_model=d_model,
        visible_dim_limit=visible_dim_limit,
    )
    if checked_visible_dim == d_model:
        msg = "partial-visible proxy requires visible_dim_limit smaller than d_model"
        raise ValueError(msg)

    first_problem = build_mamba_source_recurrence_problem(
        state_dict,
        layer_input,
        layer_index=0,
        d_state=d_state,
        mimo_rank=mimo_rank,
        norm_eps=norm_eps,
    )
    batch_size = max(d_model, first_problem.d_state * first_problem.mimo_rank)
    resolved_backend = backend or TrackingBackend(batch_size=batch_size)
    if resolved_backend.batch_size < batch_size:
        msg = (
            "encrypted partial-visible chain backend batch_size is too small; "
            f"need at least {batch_size}, got {resolved_backend.batch_size}"
        )
        raise ValueError(msg)

    started_decrypts = resolved_backend.stats().decrypt_count
    hidden = layer_input
    input_ciphertexts = _visible_tensor_ciphertexts(
        hidden,
        backend=resolved_backend,
        d_state=first_problem.d_state,
        mimo_rank=first_problem.mimo_rank,
        readout_strategy=readout_strategy,
    )
    layer_depths: list[int | None] = []
    traces: list[CheckpointFullLayerCiphertextTrace] = []
    for layer_index in range(layer_count):
        trace = run_checkpoint_encrypted_pre_recurrence_full_layer_ciphertexts_with_backend(
            state_dict,
            hidden,
            layer_index=layer_index,
            d_state=first_problem.d_state,
            mimo_rank=first_problem.mimo_rank,
            backend=resolved_backend,
            readout_strategy=readout_strategy,
            multiplicative_depth=multiplicative_depth,
            norm_eps=norm_eps,
            polynomial_degree=polynomial_degree,
            polynomial_range=polynomial_range,
            rms_norm_mode=rms_norm_mode,
            newton_iterations=newton_iterations,
            newton_range=newton_range,
            state_decay_mode=state_decay_mode,
            decay_polynomial_degree=decay_polynomial_degree,
            decay_polynomial_range=decay_polynomial_range,
            visible_dim_limit=checked_visible_dim,
            input_ciphertexts=input_ciphertexts,
            allow_partial_input_ciphertext_handoff=True,
        )
        traces.append(trace)
        layer_depths.append(trace.pre_recurrence_depth_estimate)
        hidden = run_mamba_source_layer(
            state_dict,
            hidden,
            layer_index=layer_index,
            d_state=first_problem.d_state,
            mimo_rank=first_problem.mimo_rank,
            norm_eps=norm_eps,
        )
        input_ciphertexts = _inject_plaintext_visible_remainder(
            trace.output_ciphertexts,
            hidden,
            checked_visible_dim=checked_visible_dim,
            backend=resolved_backend,
            d_state=first_problem.d_state,
            mimo_rank=first_problem.mimo_rank,
            readout_strategy=readout_strategy,
        )

    final_started_decrypts = resolved_backend.stats().decrypt_count
    actual_rows = tuple(
        resolved_backend.decrypt(output_ct, length=checked_visible_dim)
        for output_ct in traces[-1].output_ciphertexts
    )
    expected_rows = tuple(
        tuple(float(value) for value in row[:checked_visible_dim])
        for row in hidden[0].detach().cpu().float().tolist()
    )
    max_abs_error = max(
        (
            abs(actual - expected)
            for actual_row, expected_row in zip(actual_rows, expected_rows, strict=True)
            for actual, expected in zip(actual_row, expected_row, strict=True)
        ),
        default=0.0,
    )
    final_decrypt_count = resolved_backend.stats().decrypt_count - final_started_decrypts
    no_intermediate_decrypt = (
        all(trace.decrypt_count_delta == 0 for trace in traces)
        and final_decrypt_count == int(layer_input.shape[1])
        and final_started_decrypts == started_decrypts
    )
    plaintext_precomputed = tuple(
        sorted(
            {
                "visible_plaintext_remainder",
                *(stage for trace in traces for stage in trace.plaintext_precomputed_stages),
            }
        )
    )
    return CheckpointFullLayerCiphertextChainGate(
        layer_count=layer_count,
        d_model=d_model,
        checked_visible_dim=checked_visible_dim,
        full_visible_output_checked=False,
        partial_visible_output_checked=True,
        d_state=first_problem.d_state,
        mimo_rank=first_problem.mimo_rank,
        seq_len=int(layer_input.shape[1]),
        backend=resolved_backend.stats().backend,
        encrypted=bool(resolved_backend.stats().encrypted),
        readout_strategy=readout_strategy,
        max_abs_error=max_abs_error,
        atol=atol,
        passed=max_abs_error <= atol and no_intermediate_decrypt,
        inter_layer_ciphertext_handoff=True,
        no_intermediate_decrypt=no_intermediate_decrypt,
        final_decrypt_count=final_decrypt_count,
        plaintext_precomputed_stages=plaintext_precomputed,
        layer_depth_estimates=tuple(layer_depths),
        backend_stats=resolved_backend.stats().to_json_dict(),
        notes=(
            "partial-visible proxy chains only the checked visible prefix as ciphertext",
            "the unchecked visible suffix is injected from plaintext reference after each layer",
            "this is not a full visible-output or full-model correctness claim",
        ),
    )


def run_checkpoint_encrypted_pre_recurrence_full_layer_gate(
    state_dict: dict[str, Tensor],
    layer_input: Tensor,
    *,
    layer_index: int = 0,
    d_state: int | None = None,
    mimo_rank: int | None = None,
    backend: FHEBackend | None = None,
    readout_strategy: ReadoutStrategy = "rank-local",
    multiplicative_depth: int = 28,
    norm_eps: float = 1e-5,
    polynomial_degree: int = 13,
    polynomial_range: float = 6.0,
    rms_norm_mode: RmsNormMode = "newton-invsqrt",
    newton_iterations: int = 2,
    newton_range: tuple[float, float] = (0.25, 0.5),
    state_decay_mode: StateDecayMode = "poly-composed",
    decay_polynomial_degree: int = 5,
    decay_polynomial_range: tuple[float, float] = (-0.5, 0.5),
    atol: float = 5e-2,
    visible_dim_limit: int | None = None,
    visible_output_scale: float = 1.0,
) -> CheckpointFullLayerCiphertextGate:
    """Check visible output with encrypted pre-recurrence and recurrence tensors."""

    if atol < 0:
        msg = "atol must be non-negative"
        raise ValueError(msg)
    trace = run_checkpoint_encrypted_pre_recurrence_full_layer_ciphertexts_with_backend(
        state_dict,
        layer_input,
        layer_index=layer_index,
        d_state=d_state,
        mimo_rank=mimo_rank,
        backend=backend,
        readout_strategy=readout_strategy,
        multiplicative_depth=multiplicative_depth,
        norm_eps=norm_eps,
        polynomial_degree=polynomial_degree,
        polynomial_range=polynomial_range,
        rms_norm_mode=rms_norm_mode,
        newton_iterations=newton_iterations,
        newton_range=newton_range,
        state_decay_mode=state_decay_mode,
        decay_polynomial_degree=decay_polynomial_degree,
        decay_polynomial_range=decay_polynomial_range,
        visible_dim_limit=visible_dim_limit,
        visible_output_scale=visible_output_scale,
    )
    resolved_backend = trace.backend_handle
    started_decrypts = resolved_backend.stats().decrypt_count
    actual_rows = tuple(
        resolved_backend.decrypt(output_ct, length=trace.checked_visible_dim)
        for output_ct in trace.output_ciphertexts
    )
    max_abs_error = max(
        (
            abs(actual - expected)
            for actual_row, expected_row in zip(actual_rows, trace.expected_outputs, strict=True)
            for actual, expected in zip(actual_row, expected_row, strict=True)
        ),
        default=0.0,
    )
    final_decrypts = resolved_backend.stats().decrypt_count - started_decrypts
    no_intermediate_decrypt = trace.decrypt_count_delta == 0 and final_decrypts == trace.seq_len
    return CheckpointFullLayerCiphertextGate(
        layer_index=layer_index,
        d_model=trace.d_model,
        checked_visible_dim=trace.checked_visible_dim,
        full_visible_output_checked=trace.full_visible_output_checked,
        partial_visible_output_checked=trace.partial_visible_output_checked,
        d_state=trace.d_state,
        mimo_rank=trace.mimo_rank,
        seq_len=trace.seq_len,
        backend=resolved_backend.stats().backend,
        encrypted=bool(resolved_backend.stats().encrypted),
        input_mode="encrypted-dynamic-bc",
        readout_strategy=readout_strategy,
        visible_output_scale=trace.visible_output_scale,
        max_abs_error=max_abs_error,
        atol=atol,
        passed=max_abs_error <= atol and no_intermediate_decrypt,
        recurrence_ciphertext=True,
        visible_handoff_ciphertext=True,
        no_intermediate_decrypt=no_intermediate_decrypt,
        full_layer_formula_checked=trace.full_visible_output_checked,
        official_mamba_parity=False,
        full_model_correctness_claimed=False,
        plaintext_precomputed_stages=trace.plaintext_precomputed_stages,
        backend_stats=resolved_backend.stats().to_json_dict(),
        notes=(*trace.notes, "final lm_head/client decoding is not included"),
        pre_recurrence_ciphertext=True,
        pre_recurrence_depth_estimate=trace.pre_recurrence_depth_estimate,
    )


def run_checkpoint_encrypted_pre_recurrence_full_layer_chain_gate(
    state_dict: dict[str, Tensor],
    layer_input: Tensor,
    *,
    layer_count: int,
    d_state: int | None = None,
    mimo_rank: int | None = None,
    backend: FHEBackend | None = None,
    readout_strategy: ReadoutStrategy = "rank-local",
    multiplicative_depth: int = 28,
    norm_eps: float = 1e-5,
    polynomial_degree: int = 13,
    polynomial_range: float = 6.0,
    rms_norm_mode: RmsNormMode = "newton-invsqrt",
    newton_iterations: int = 2,
    newton_range: tuple[float, float] = (0.25, 0.5),
    state_decay_mode: StateDecayMode = "poly-composed",
    decay_polynomial_degree: int = 5,
    decay_polynomial_range: tuple[float, float] = (-0.5, 0.5),
    atol: float = 5e-2,
) -> CheckpointFullLayerCiphertextChainGate:
    """Chain encrypted source-style full-layer outputs across multiple layers."""

    if layer_count <= 0:
        msg = "layer_count must be positive"
        raise ValueError(msg)
    if layer_input.ndim != 3 or layer_input.shape[0] != 1:
        msg = "layer_input must have shape [1, seq_len, d_model]"
        raise ValueError(msg)
    if atol < 0:
        msg = "atol must be non-negative"
        raise ValueError(msg)
    if rms_norm_mode == "plaintext-exact":
        msg = (
            "encrypted full-layer chain cannot use plaintext-exact RMSNorm because each "
            "layer consumes visible-output ciphertexts"
        )
        raise ValueError(msg)

    first_problem = build_mamba_source_recurrence_problem(
        state_dict,
        layer_input,
        layer_index=0,
        d_state=d_state,
        mimo_rank=mimo_rank,
        norm_eps=norm_eps,
    )
    d_model = int(layer_input.shape[-1])
    batch_size = max(d_model, first_problem.d_state * first_problem.mimo_rank)
    resolved_backend = backend or TrackingBackend(batch_size=batch_size)
    if resolved_backend.batch_size < batch_size:
        msg = (
            "encrypted full-layer chain backend batch_size is too small; "
            f"need at least {batch_size}, got {resolved_backend.batch_size}"
        )
        raise ValueError(msg)

    started_decrypts = resolved_backend.stats().decrypt_count
    hidden = layer_input
    input_ciphertexts = _visible_tensor_ciphertexts(
        hidden,
        backend=resolved_backend,
        d_state=first_problem.d_state,
        mimo_rank=first_problem.mimo_rank,
        readout_strategy=readout_strategy,
    )
    layer_depths: list[int | None] = []
    traces: list[CheckpointFullLayerCiphertextTrace] = []
    for layer_index in range(layer_count):
        trace = run_checkpoint_encrypted_pre_recurrence_full_layer_ciphertexts_with_backend(
            state_dict,
            hidden,
            layer_index=layer_index,
            d_state=first_problem.d_state,
            mimo_rank=first_problem.mimo_rank,
            backend=resolved_backend,
            readout_strategy=readout_strategy,
            multiplicative_depth=multiplicative_depth,
            norm_eps=norm_eps,
            polynomial_degree=polynomial_degree,
            polynomial_range=polynomial_range,
            rms_norm_mode=rms_norm_mode,
            newton_iterations=newton_iterations,
            newton_range=newton_range,
            state_decay_mode=state_decay_mode,
            decay_polynomial_degree=decay_polynomial_degree,
            decay_polynomial_range=decay_polynomial_range,
            visible_dim_limit=None,
            input_ciphertexts=input_ciphertexts,
        )
        traces.append(trace)
        layer_depths.append(trace.pre_recurrence_depth_estimate)
        input_ciphertexts = trace.output_ciphertexts
        hidden = run_mamba_source_layer(
            state_dict,
            hidden,
            layer_index=layer_index,
            d_state=first_problem.d_state,
            mimo_rank=first_problem.mimo_rank,
            norm_eps=norm_eps,
        )

    final_started_decrypts = resolved_backend.stats().decrypt_count
    actual_rows = tuple(
        resolved_backend.decrypt(output_ct, length=d_model) for output_ct in input_ciphertexts
    )
    expected_rows = tuple(
        tuple(float(value) for value in row) for row in hidden[0].detach().cpu().float().tolist()
    )
    max_abs_error = max(
        (
            abs(actual - expected)
            for actual_row, expected_row in zip(actual_rows, expected_rows, strict=True)
            for actual, expected in zip(actual_row, expected_row, strict=True)
        ),
        default=0.0,
    )
    final_decrypt_count = resolved_backend.stats().decrypt_count - final_started_decrypts
    no_intermediate_decrypt = (
        all(trace.decrypt_count_delta == 0 for trace in traces)
        and final_decrypt_count == int(layer_input.shape[1])
        and final_started_decrypts == started_decrypts
    )
    plaintext_precomputed = tuple(
        sorted({stage for trace in traces for stage in trace.plaintext_precomputed_stages})
    )
    return CheckpointFullLayerCiphertextChainGate(
        layer_count=layer_count,
        d_model=d_model,
        checked_visible_dim=d_model,
        full_visible_output_checked=True,
        partial_visible_output_checked=False,
        d_state=first_problem.d_state,
        mimo_rank=first_problem.mimo_rank,
        seq_len=int(layer_input.shape[1]),
        backend=resolved_backend.stats().backend,
        encrypted=bool(resolved_backend.stats().encrypted),
        readout_strategy=readout_strategy,
        max_abs_error=max_abs_error,
        atol=atol,
        passed=max_abs_error <= atol and no_intermediate_decrypt,
        inter_layer_ciphertext_handoff=layer_count > 1,
        no_intermediate_decrypt=no_intermediate_decrypt,
        final_decrypt_count=final_decrypt_count,
        plaintext_precomputed_stages=plaintext_precomputed,
        layer_depth_estimates=tuple(layer_depths),
        backend_stats=resolved_backend.stats().to_json_dict(),
        notes=(
            "visible-output ciphertexts are reused as the next layer input",
            "only final chained visible outputs are decrypted for correctness",
            "source-style formula parity is checked; official fused-kernel parity is not claimed",
            "final lm_head/client decoding is not included",
        ),
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
    visible_output_scale: float = 1.0,
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
    trace = run_checkpoint_full_layer_ciphertexts_with_backend(
        state_dict,
        layer_input,
        layer_index=layer_index,
        d_state=d_state,
        mimo_rank=mimo_rank,
        backend=backend,
        input_mode=input_mode,
        readout_strategy=readout_strategy,
        multiplicative_depth=multiplicative_depth,
        norm_eps=norm_eps,
        visible_dim_limit=visible_dim_limit,
        visible_output_scale=visible_output_scale,
    )
    resolved_backend = trace.backend_handle
    started_decrypts = resolved_backend.stats().decrypt_count
    decrypted_outputs = [
        resolved_backend.decrypt(output_ct, length=trace.checked_visible_dim)
        for output_ct in trace.output_ciphertexts
    ]

    actual_rows = tuple(tuple(row) for row in decrypted_outputs)
    max_abs_error = max(
        (
            abs(actual - expected)
            for actual_row, expected_row in zip(actual_rows, trace.expected_outputs, strict=True)
            for actual, expected in zip(actual_row, expected_row, strict=True)
        ),
        default=0.0,
    )
    no_intermediate_decrypt = (
        trace.decrypt_count_delta == 0
        and resolved_backend.stats().decrypt_count - started_decrypts == trace.seq_len
    )
    return CheckpointFullLayerCiphertextGate(
        layer_index=layer_index,
        d_model=trace.d_model,
        checked_visible_dim=trace.checked_visible_dim,
        full_visible_output_checked=trace.full_visible_output_checked,
        partial_visible_output_checked=trace.partial_visible_output_checked,
        d_state=trace.d_state,
        mimo_rank=trace.mimo_rank,
        seq_len=trace.seq_len,
        backend=resolved_backend.stats().backend,
        encrypted=bool(resolved_backend.stats().encrypted),
        input_mode=input_mode,
        readout_strategy=readout_strategy,
        visible_output_scale=trace.visible_output_scale,
        max_abs_error=max_abs_error,
        atol=atol,
        passed=max_abs_error <= atol and no_intermediate_decrypt,
        recurrence_ciphertext=True,
        visible_handoff_ciphertext=True,
        no_intermediate_decrypt=no_intermediate_decrypt,
        full_layer_formula_checked=trace.full_visible_output_checked,
        official_mamba_parity=False,
        full_model_correctness_claimed=False,
        plaintext_precomputed_stages=trace.plaintext_precomputed_stages,
        backend_stats=resolved_backend.stats().to_json_dict(),
        notes=trace.notes,
    )


def run_checkpoint_full_layer_ciphertexts_with_backend(
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
    norm_eps: float = 1e-5,
    visible_dim_limit: int | None = None,
    visible_output_scale: float = 1.0,
) -> CheckpointFullLayerCiphertextTrace:
    """Return full-layer visible output ciphertexts without decrypting them."""

    if visible_output_scale <= 0:
        msg = "visible_output_scale must be positive"
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
    output_ciphertexts = []
    for token_index, recurrence_ct in enumerate(trace.output_ciphertexts):
        final_ct = _visible_output_ciphertext(
            backend=resolved_backend,
            recurrence_ct=recurrence_ct,
            output_slots=trace.output_slots,
            visible=visible,
            checked_visible_dim=checked_visible_dim,
            token_index=token_index,
            visible_output_scale=visible_output_scale,
        )
        output_ciphertexts.append(final_ct)

    expected_rows = tuple(
        tuple(
            visible_output_scale * float(value)
            for value in visible.expected_final_output[0, token_index, :checked_visible_dim]
            .detach()
            .cpu()
        )
        for token_index in range(visible.seq_len)
    )
    full_visible_output_checked = checked_visible_dim == visible.d_model
    partial_visible_output_checked = checked_visible_dim < visible.d_model
    output_slots = tuple(range(checked_visible_dim))
    required_rotations = required_full_layer_visible_rotations(
        d_model=visible.d_model,
        d_state=problem.d_state,
        mimo_rank=problem.mimo_rank,
        readout_strategy=readout_strategy,
        visible_dim_limit=visible_dim_limit,
    )
    layout_contract = CiphertextLayoutContract(
        output_layout="visible-output",
        d_state=problem.d_state,
        mimo_rank=problem.mimo_rank,
        readout_strategy=readout_strategy,
        output_slots=output_slots,
        required_rotations=required_rotations,
    )
    notes = [
        "checks source-style full-layer visible output, not official fused kernel parity",
        "input-dependent pre-recurrence tensors are still plaintext-precomputed in Stage 0",
    ]
    if partial_visible_output_checked:
        notes.append(
            "visible output is partially checked because visible_dim_limit is smaller than d_model"
        )
    else:
        notes.append("visible output check covers the full d_model")
    if visible_output_scale != 1.0:
        notes.append("visible output ciphertext and expected output are scaled before decoding")
    return CheckpointFullLayerCiphertextTrace(
        layer_index=layer_index,
        d_model=visible.d_model,
        checked_visible_dim=checked_visible_dim,
        full_visible_output_checked=full_visible_output_checked,
        partial_visible_output_checked=partial_visible_output_checked,
        d_state=problem.d_state,
        mimo_rank=problem.mimo_rank,
        seq_len=problem.seq_len,
        backend=resolved_backend.stats().backend,
        encrypted=bool(resolved_backend.stats().encrypted),
        input_mode=input_mode,
        readout_strategy=readout_strategy,
        visible_output_scale=visible_output_scale,
        output_layout="visible-output",
        output_slots=output_slots,
        layout_contract=layout_contract,
        required_rotations=required_rotations,
        output_ciphertexts=LayoutBoundCiphertexts(
            tuple(output_ciphertexts),
            layout_contract=layout_contract,
        ),
        expected_outputs=expected_rows,
        backend_handle=resolved_backend,
        recurrence_ciphertext=True,
        visible_handoff_ciphertext=True,
        decrypt_count_delta=resolved_backend.stats().decrypt_count - started_decrypts,
        plaintext_precomputed_stages=(
            "rms_norm",
            "causal_conv_silu",
            "dynamic_b",
            "dynamic_c",
            "state_rank_decay",
            "gate_values",
        ),
        backend_stats=resolved_backend.stats().to_json_dict(),
        notes=tuple(notes),
    )


def _visible_output_ciphertext(
    *,
    backend: FHEBackend,
    recurrence_ct: Any,
    output_slots: tuple[int, ...],
    visible: MambaSourceVisibleHandoffTensors,
    checked_visible_dim: int,
    token_index: int,
    visible_output_scale: float = 1.0,
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
        out_proj_weight=visible.out_proj_weight * visible_output_scale,
        checked_visible_dim=checked_visible_dim,
    )
    return backend.add(
        projected_ct,
        backend.encrypt(
            [
                visible_output_scale * float(value)
                for value in visible.residual[0, token_index, :checked_visible_dim].detach().cpu()
            ]
        ),
    )


def _encrypted_pre_recurrence_visible_output_ciphertext(
    *,
    backend: FHEBackend,
    recurrence_ct: Any,
    rank_input_ct: Any,
    gate_ct: Any,
    output_slots: tuple[int, ...],
    d_skip: tuple[float, ...] | None,
    visible: MambaSourceVisibleHandoffTensors,
    checked_visible_dim: int,
    token_index: int,
    visible_output_scale: float = 1.0,
    residual_ct: Any | None = None,
) -> Any:
    skip_ct = _rank_ciphertext_to_output_slots(
        backend=backend,
        rank_ct=rank_input_ct,
        output_slots=output_slots,
        weights=d_skip if d_skip is not None else tuple(1.0 for _ in output_slots),
    )
    rank_ct = backend.add(recurrence_ct, skip_ct)
    aligned_gate_ct = _rank_ciphertext_to_output_slots(
        backend=backend,
        rank_ct=gate_ct,
        output_slots=output_slots,
        weights=tuple(1.0 for _ in output_slots),
    )
    gated_ct = backend.mul_ct(rank_ct, aligned_gate_ct)
    projected_ct = _project_rank_slots_to_visible(
        backend=backend,
        rank_ct=gated_ct,
        output_slots=output_slots,
        out_proj_weight=visible.out_proj_weight * visible_output_scale,
        checked_visible_dim=checked_visible_dim,
    )
    resolved_residual_ct = residual_ct
    if resolved_residual_ct is None:
        resolved_residual_ct = backend.encrypt(
            [
                visible_output_scale * float(value)
                for value in visible.residual[0, token_index, :checked_visible_dim].detach().cpu()
            ]
        )
    elif visible_output_scale != 1.0:
        resolved_residual_ct = backend.mul_plain(
            resolved_residual_ct,
            backend.encode([visible_output_scale] * checked_visible_dim),
        )
    return backend.add(projected_ct, resolved_residual_ct)


def _rank_ciphertext_to_output_slots(
    *,
    backend: FHEBackend,
    rank_ct: Any,
    output_slots: tuple[int, ...],
    weights: tuple[float, ...],
) -> Any:
    if len(weights) != len(output_slots):
        msg = "weights length must match output_slots"
        raise ValueError(msg)
    output_ct = backend.encrypt([0.0] * backend.batch_size)
    for rank_index, (output_slot, weight) in enumerate(zip(output_slots, weights, strict=True)):
        if weight == 0.0:
            continue
        mask = [0.0] * backend.batch_size
        mask[rank_index] = weight
        selected = backend.mul_plain(rank_ct, backend.encode(mask))
        shift = rank_index - output_slot
        term = selected if shift == 0 else backend.rotate(selected, shift)
        output_ct = backend.add(output_ct, term)
    return output_ct


def _bind_expanded_rank_input_ciphertexts(
    ciphertexts: tuple[Any, ...],
    *,
    d_state: int,
    rank: int,
    readout_strategy: ReadoutStrategy,
) -> LayoutBoundCiphertexts:
    output_slots = tuple(r * d_state for r in range(rank))
    layout_contract = CiphertextLayoutContract(
        output_layout="expanded-rank-input",
        d_state=d_state,
        mimo_rank=rank,
        readout_strategy=readout_strategy,
        output_slots=output_slots,
        required_rotations=(),
    )
    return LayoutBoundCiphertexts(ciphertexts, layout_contract=layout_contract)


def _visible_tensor_ciphertexts(
    values: Tensor,
    *,
    backend: FHEBackend,
    d_state: int,
    mimo_rank: int,
    readout_strategy: ReadoutStrategy,
) -> LayoutBoundCiphertexts:
    d_model = int(values.shape[-1])
    output_slots = tuple(range(d_model))
    layout_contract = CiphertextLayoutContract(
        output_layout="visible-output",
        d_state=d_state,
        mimo_rank=mimo_rank,
        readout_strategy=readout_strategy,
        output_slots=output_slots,
        required_rotations=(),
    )
    ciphertexts = tuple(
        backend.encrypt([float(value) for value in row])
        for row in values[0].detach().cpu().float().tolist()
    )
    return LayoutBoundCiphertexts(ciphertexts, layout_contract=layout_contract)


def _visible_residual_ciphertext_for_checked_dim(
    *,
    backend: FHEBackend,
    ciphertexts: tuple[Any, ...] | None,
    token_index: int,
    checked_visible_dim: int,
) -> Any | None:
    if ciphertexts is None:
        return None
    ciphertext = ciphertexts[token_index]
    if checked_visible_dim >= backend.batch_size:
        return ciphertext
    mask = [0.0] * backend.batch_size
    mask[:checked_visible_dim] = [1.0] * checked_visible_dim
    return backend.mul_plain(ciphertext, backend.encode(mask))


def _inject_plaintext_visible_remainder(
    partial_ciphertexts: tuple[Any, ...],
    values: Tensor,
    *,
    checked_visible_dim: int,
    backend: FHEBackend,
    d_state: int,
    mimo_rank: int,
    readout_strategy: ReadoutStrategy,
) -> LayoutBoundCiphertexts:
    d_model = int(values.shape[-1])
    if checked_visible_dim >= d_model:
        msg = "checked_visible_dim must be smaller than d_model for remainder injection"
        raise ValueError(msg)
    output_slots = tuple(range(d_model))
    layout_contract = CiphertextLayoutContract(
        output_layout="visible-output",
        d_state=d_state,
        mimo_rank=mimo_rank,
        readout_strategy=readout_strategy,
        output_slots=output_slots,
        required_rotations=(),
    )
    rows = values[0].detach().cpu().float().tolist()
    if len(partial_ciphertexts) != len(rows):
        msg = "partial_ciphertexts length must match seq_len"
        raise ValueError(msg)
    full_ciphertexts = []
    for ciphertext, row in zip(partial_ciphertexts, rows, strict=True):
        remainder = [0.0] * checked_visible_dim + [
            float(value) for value in row[checked_visible_dim:]
        ]
        full_ciphertexts.append(backend.add(ciphertext, backend.encrypt(remainder)))
    return LayoutBoundCiphertexts(tuple(full_ciphertexts), layout_contract=layout_contract)


def _validate_visible_input_ciphertexts(
    ciphertexts: tuple[Any, ...],
    *,
    d_model: int,
    seq_len: int,
) -> None:
    if len(ciphertexts) != seq_len:
        msg = "input_ciphertexts length must match seq_len"
        raise ValueError(msg)
    contract = getattr(ciphertexts, "layout_contract", None)
    if contract is None:
        return
    if contract.output_layout != "visible-output":
        msg = "input_ciphertexts must use visible-output layout"
        raise ValueError(msg)
    expected_slots = tuple(range(d_model))
    if tuple(contract.output_slots[:d_model]) != expected_slots:
        msg = "input_ciphertexts visible-output contract must cover contiguous d_model slots"
        raise ValueError(msg)


def _expand_rank_ciphertext_to_state_slots(
    ciphertext: Any,
    *,
    d_state: int,
    rank: int,
    backend: FHEBackend,
) -> Any:
    return _sparse_bsgs_slot_linear_ciphertext(
        ciphertext,
        source_slots=tuple(range(rank)),
        output_dim=d_state * rank,
        mappings=(
            (rank_index * d_state + state_index, rank_index, 1.0)
            for rank_index in range(rank)
            for state_index in range(d_state)
        ),
        backend=backend,
    )


def _expand_state_vector_ciphertext_to_state_slots(
    ciphertext: Any,
    *,
    d_state: int,
    rank: int,
    backend: FHEBackend,
) -> Any:
    return _sparse_bsgs_slot_linear_ciphertext(
        ciphertext,
        source_slots=tuple(range(d_state)),
        output_dim=d_state * rank,
        mappings=(
            (rank_index * d_state + state_index, state_index, 1.0)
            for rank_index in range(rank)
            for state_index in range(d_state)
        ),
        backend=backend,
    )


def expand_rank_to_state_bsgs_rotation_steps(*, d_state: int, rank: int) -> tuple[int, ...]:
    """Rotation-key inventory for exact rank-vector expansion into state slots."""

    return _sparse_bsgs_slot_linear_rotation_steps(
        source_slots=tuple(range(rank)),
        output_dim=d_state * rank,
        mappings=(
            (rank_index * d_state + state_index, rank_index)
            for rank_index in range(rank)
            for state_index in range(d_state)
        ),
    )


def expand_state_vector_to_state_bsgs_rotation_steps(
    *,
    d_state: int,
    rank: int,
) -> tuple[int, ...]:
    """Rotation-key inventory for exact state-vector expansion into state slots."""

    return _sparse_bsgs_slot_linear_rotation_steps(
        source_slots=tuple(range(d_state)),
        output_dim=d_state * rank,
        mappings=(
            (rank_index * d_state + state_index, state_index)
            for rank_index in range(rank)
            for state_index in range(d_state)
        ),
    )


def _sparse_bsgs_slot_linear_ciphertext(
    input_ct: Any,
    *,
    source_slots: tuple[int, ...],
    output_dim: int,
    mappings: Any,
    backend: FHEBackend,
) -> Any:
    """Evaluate a sparse slot-linear map using the same BSGS schedule as dense matmul."""

    baby_step = slot_linear_bsgs_baby_step(source_slots=source_slots, output_dim=output_dim)
    output_ct = backend.encrypt([0.0] * backend.batch_size)
    baby_cts: dict[int, Any] = {}
    grouped_masks: dict[tuple[int, int], list[float]] = {}
    for output_index, source_slot, coefficient in mappings:
        if coefficient == 0.0:
            continue
        giant_index, baby_index = divmod(source_slot - output_index, baby_step)
        key = (giant_index, baby_index)
        mask = grouped_masks.setdefault(key, [0.0] * backend.batch_size)
        mask[(source_slot - baby_index) % backend.batch_size] += float(coefficient)

    for giant_index, baby_index in sorted(grouped_masks):
        if baby_index not in baby_cts:
            baby_cts[baby_index] = (
                input_ct if baby_index == 0 else backend.rotate(input_ct, baby_index)
            )
        term = backend.mul_plain(
            baby_cts[baby_index],
            backend.encode(grouped_masks[(giant_index, baby_index)]),
        )
        giant_shift = giant_index * baby_step
        if giant_shift:
            term = backend.rotate(term, giant_shift)
        output_ct = backend.add(output_ct, term)
    return output_ct


def _sparse_bsgs_slot_linear_rotation_steps(
    *,
    source_slots: tuple[int, ...],
    output_dim: int,
    mappings: Any,
) -> tuple[int, ...]:
    baby_step = slot_linear_bsgs_baby_step(source_slots=source_slots, output_dim=output_dim)
    rotations: set[int] = set()
    for output_index, source_slot in mappings:
        giant_index, baby_index = divmod(source_slot - output_index, baby_step)
        if baby_index:
            rotations.add(baby_index)
        giant_shift = giant_index * baby_step
        if giant_shift:
            rotations.add(giant_shift)
    return tuple(sorted(rotations))


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
    return slot_linear_ciphertext(
        rank_ct,
        source_slots=output_slots,
        weight=out_proj_weight.detach().cpu().float(),
        bias=[0.0] * checked_visible_dim,
        output_dim=checked_visible_dim,
        backend=backend,
    )


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
    rotations.update(
        slot_linear_bsgs_rotation_steps(
            source_slots=output_slots,
            output_dim=checked_visible_dim,
        )
    )
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
