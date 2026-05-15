"""Merge plaintext LoRA range-tuned rank/gate projections back into payloads."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

from fhe_native_mamba3.range_finetune import (
    LoRAConfig,
    RangeLossConfig,
    merged_linear_weight_bias,
)
from fhe_native_mamba3.stage1_rank_gate_payload import (
    Stage1RankGatePayload,
    _as_float64_array,
    _evaluate_power_polynomial_numpy,
    _evaluate_state_major_polynomial_numpy,
    _state_major_from_rank_vector,
    _state_major_from_state_vector,
)
from fhe_native_mamba3.stage2_lora_range_smoke import (
    RankGateProjectionModule,
    Stage2LoRARangeSmokeResult,
    train_lora_range_model,
)


@dataclass(frozen=True)
class Stage2LoRAPayloadMergeMetrics:
    """Detached diagnostics for one LoRA-to-payload merge."""

    effective_rank_weight_delta_max_abs: float
    gate_weight_delta_max_abs: float
    conv_bias_delta_max_abs: float
    reference_conv_pre_delta_max_abs: float
    reference_gate_pre_delta_max_abs: float
    reference_output_model_poly_delta_max_abs: float
    rank_input_poly_vs_exact_max_abs_error: float
    gate_poly_vs_exact_max_abs_error: float
    output_model_poly_vs_original_exact_max_abs_error: float

    def to_json_dict(self) -> dict[str, float]:
        return asdict(self)


@dataclass(frozen=True)
class Stage2LoRAPayloadMergeResult:
    """Result metadata for a LoRA-trained payload merge."""

    passed: bool
    training: Stage2LoRARangeSmokeResult
    metrics: Stage2LoRAPayloadMergeMetrics
    measurement_scope: dict[str, Any]

    def to_json_dict(self) -> dict[str, Any]:
        return {
            "passed": self.passed,
            "training": self.training.to_json_dict(),
            "metrics": self.metrics.to_json_dict(),
            "measurement_scope": self.measurement_scope,
        }


def train_and_merge_lora_range_payload(
    payload: Stage1RankGatePayload,
    *,
    sample_count: int = 64,
    noise_scale: float = 0.01,
    steps: int = 100,
    learning_rate: float = 1e-2,
    lora_config: LoRAConfig | None = None,
    range_loss_config: RangeLossConfig | None = None,
    seed: int = 0,
    device: str = "cpu",
) -> tuple[Stage1RankGatePayload, Stage2LoRAPayloadMergeResult]:
    """Train LoRA adapters and return a payload with merged public weights."""

    model, training = train_lora_range_model(
        payload,
        sample_count=sample_count,
        noise_scale=noise_scale,
        steps=steps,
        learning_rate=learning_rate,
        lora_config=lora_config,
        range_loss_config=range_loss_config,
        seed=seed,
        device=device,
    )
    merged_payload, metrics = merge_lora_range_payload(payload, model)
    result = Stage2LoRAPayloadMergeResult(
        passed=training.passed
        and metrics.reference_output_model_poly_delta_max_abs > 0.0
        and training.after.max_excess <= training.before.max_excess,
        training=training,
        metrics=metrics,
        measurement_scope={
            "stage2_lora_payload_merge": True,
            "lora_training_executed": True,
            "merged_public_rank_gate_weights": True,
            "encrypted_execution": False,
            "exact_reference_preserved": False,
            "full_model_correctness_claimed": False,
            "claim": (
                "Trains plaintext LoRA adapters on the rank/gate projection boundary "
                "and merges them into a Stage 1 payload for later encrypted replay. "
                "Only polynomial payload references are recomputed; original exact "
                "checkpoint references are retained for drift diagnostics."
            ),
        },
    )
    return merged_payload, result


def merge_lora_range_payload(
    payload: Stage1RankGatePayload,
    model: RankGateProjectionModule,
) -> tuple[Stage1RankGatePayload, Stage2LoRAPayloadMergeMetrics]:
    """Return a payload whose rank/gate weights include trained LoRA deltas."""

    rank_weight, rank_bias = merged_linear_weight_bias(model.rank)
    gate_weight, gate_bias = merged_linear_weight_bias(model.gate)
    if rank_bias is None:
        msg = "rank projection must have a bias"
        raise ValueError(msg)
    if gate_bias is not None:
        msg = "gate projection must remain bias-free"
        raise ValueError(msg)

    arrays = {
        name: np.array(value, dtype=np.float64, copy=True) for name, value in payload.arrays.items()
    }
    old_effective_rank_weight = arrays["effective_rank_weight"].copy()
    old_conv_bias = arrays["conv_bias"].copy()
    old_gate_weight = arrays["gate_weight"].copy()
    old_reference_conv_pre = arrays["reference_conv_pre"].copy()
    old_reference_gate_pre = arrays["reference_gate_pre"].copy()
    old_reference_output_model_poly = arrays["reference_output_model_poly"].copy()

    arrays["effective_rank_weight"] = _as_float64_array(rank_weight.detach().cpu().numpy())
    arrays["conv_bias"] = _as_float64_array(rank_bias.detach().cpu().numpy())
    arrays["gate_weight"] = _as_float64_array(gate_weight.detach().cpu().numpy())
    _recompute_payload_references(payload, arrays)

    merged_payload = Stage1RankGatePayload(
        config=payload.config,
        layer_index=payload.layer_index,
        prompt_token=payload.prompt_token,
        norm_eps=payload.norm_eps,
        arrays=arrays,
    )
    metrics = Stage2LoRAPayloadMergeMetrics(
        effective_rank_weight_delta_max_abs=_max_abs_delta(
            arrays["effective_rank_weight"],
            old_effective_rank_weight,
        ),
        gate_weight_delta_max_abs=_max_abs_delta(arrays["gate_weight"], old_gate_weight),
        conv_bias_delta_max_abs=_max_abs_delta(arrays["conv_bias"], old_conv_bias),
        reference_conv_pre_delta_max_abs=_max_abs_delta(
            arrays["reference_conv_pre"],
            old_reference_conv_pre,
        ),
        reference_gate_pre_delta_max_abs=_max_abs_delta(
            arrays["reference_gate_pre"],
            old_reference_gate_pre,
        ),
        reference_output_model_poly_delta_max_abs=_max_abs_delta(
            arrays["reference_output_model_poly"],
            old_reference_output_model_poly,
        ),
        rank_input_poly_vs_exact_max_abs_error=_max_abs_delta(
            arrays["reference_rank_input_poly"],
            arrays["reference_rank_input"],
        ),
        gate_poly_vs_exact_max_abs_error=_max_abs_delta(
            arrays["reference_gate_poly"],
            arrays["reference_gate"],
        ),
        output_model_poly_vs_original_exact_max_abs_error=_max_abs_delta(
            arrays["reference_output_model_poly"],
            arrays["reference_output_model_exact"],
        ),
    )
    return merged_payload, metrics


def _recompute_payload_references(
    payload: Stage1RankGatePayload,
    arrays: dict[str, np.ndarray],
) -> None:
    config = payload.config
    rms_input = arrays["rms_input"]
    arrays["reference_conv_pre"] = _as_float64_array(
        arrays["effective_rank_weight"] @ rms_input + arrays["conv_bias"]
    )
    arrays["reference_gate_pre"] = _as_float64_array(arrays["gate_weight"] @ rms_input)
    arrays["reference_rank_input"] = _as_float64_array(_silu_numpy(arrays["reference_conv_pre"]))
    arrays["reference_gate"] = _as_float64_array(_silu_numpy(arrays["reference_gate_pre"]))
    arrays["reference_skip_update"] = _as_float64_array(
        arrays["reference_rank_input"] * arrays["d_skip"]
    )
    arrays["reference_rank_input_poly"] = _as_float64_array(
        _evaluate_power_polynomial_numpy(
            arrays["reference_conv_pre"],
            arrays["rank_silu_coefficients"],
        )
    )
    arrays["reference_gate_poly"] = _as_float64_array(
        _evaluate_power_polynomial_numpy(
            arrays["reference_gate_pre"],
            arrays["gate_silu_coefficients"],
        )
    )
    arrays["reference_skip_update_poly"] = _as_float64_array(
        arrays["reference_rank_input_poly"] * arrays["d_skip"]
    )
    arrays["reference_b_vec_poly"] = _as_float64_array(
        arrays["b_weight"] @ arrays["reference_rank_input_poly"]
    )
    arrays["reference_c_vec_poly"] = _as_float64_array(
        arrays["c_weight"] @ arrays["reference_rank_input_poly"]
    )
    arrays["reference_b_state_major_poly"] = _as_float64_array(
        _state_major_from_state_vector(arrays["reference_b_vec_poly"], config=config)
    )
    arrays["reference_c_state_major_poly"] = _as_float64_array(
        _state_major_from_state_vector(arrays["reference_c_vec_poly"], config=config)
    )
    arrays["reference_dt_hidden_poly"] = _as_float64_array(
        arrays["dt_in_weight"] @ arrays["reference_rank_input_poly"]
    )
    arrays["reference_dt_pre_poly"] = _as_float64_array(
        arrays["dt_proj_weight"] @ arrays["reference_dt_hidden_poly"] + arrays["dt_proj_bias"]
    )
    arrays["reference_dt_state_major_poly"] = _as_float64_array(
        _state_major_from_rank_vector(arrays["reference_dt_pre_poly"], config=config)
    )
    arrays["reference_decay_state_major_poly"] = _as_float64_array(
        _evaluate_state_major_polynomial_numpy(
            arrays["reference_dt_state_major_poly"],
            arrays["decay_coefficients"],
        )
    )
    rank_input_state_major = _state_major_from_rank_vector(
        arrays["reference_rank_input_poly"],
        config=config,
    )
    arrays["reference_state_new_poly"] = _as_float64_array(
        arrays["reference_decay_state_major_poly"] * arrays["previous_state"]
        + arrays["reference_b_state_major_poly"] * rank_input_state_major
    )
    arrays["reference_readout_rank_poly"] = _as_float64_array(
        np.sum(arrays["reference_c_state_major_poly"] * arrays["reference_state_new_poly"], axis=0)
    )
    arrays["reference_rank_output_poly"] = _as_float64_array(
        arrays["reference_readout_rank_poly"] + arrays["reference_skip_update_poly"]
    )
    arrays["reference_rank_payload_poly"] = _as_float64_array(
        arrays["reference_rank_output_poly"] * arrays["reference_gate_poly"]
    )
    arrays["reference_output_model_poly"] = _as_float64_array(
        arrays["residual_input"] + arrays["w_out"] @ arrays["reference_rank_payload_poly"]
    )


def _silu_numpy(values: np.ndarray) -> np.ndarray:
    clipped = np.clip(-np.asarray(values, dtype=np.float64), -60.0, 60.0)
    return np.asarray(values, dtype=np.float64) / (1.0 + np.exp(clipped))


def _max_abs_delta(lhs: np.ndarray, rhs: np.ndarray) -> float:
    lhs_array = np.asarray(lhs, dtype=np.float64)
    rhs_array = np.asarray(rhs, dtype=np.float64)
    if lhs_array.shape != rhs_array.shape:
        msg = f"shape mismatch: {lhs_array.shape} vs {rhs_array.shape}"
        raise ValueError(msg)
    if lhs_array.size == 0:
        return 0.0
    return float(np.max(np.abs(lhs_array - rhs_array)))


__all__ = [
    "Stage2LoRAPayloadMergeMetrics",
    "Stage2LoRAPayloadMergeResult",
    "merge_lora_range_payload",
    "train_and_merge_lora_range_payload",
]
