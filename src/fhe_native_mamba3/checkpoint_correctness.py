"""Correctness gates for checkpoint-derived encrypted recurrence paths."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from torch import Tensor

from fhe_native_mamba3.backends.base import FHEBackend
from fhe_native_mamba3.backends.tracking import TrackingBackend
from fhe_native_mamba3.layout import ReadoutStrategy
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
    passed = recurrence_passed and (reference_passed is not False)
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
        passed=passed,
        backend_stats=result.backend_stats,
        notes=(
            "recurrence gate compares backend output to the checkpoint-derived "
            "plaintext recurrence problem",
            "full gate/out-projection/residual ciphertext handoff remains a separate Stage 0 item",
        ),
    )
