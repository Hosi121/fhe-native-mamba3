"""Stage 0 status report assembly from measured JSON artifacts."""

from __future__ import annotations

from typing import Any


def build_stage0_status_report(
    *,
    version: str,
    bootstrap_latency: dict[str, Any] | None = None,
    stack_latency_estimate: dict[str, Any] | None = None,
    checkpoint_bootstrap_smoke: dict[str, Any] | None = None,
    checkpoint_source_profile: dict[str, Any] | None = None,
    range_scale_plan: dict[str, Any] | None = None,
    checkpoint_full_layer_gate: dict[str, Any] | None = None,
    client_decode_smoke: dict[str, Any] | None = None,
    segment_samples: dict[str, Any] | None = None,
    all_layer_recurrence: dict[str, Any] | None = None,
    ciphertext_handoff: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a compact Stage 0 progress report from existing measurements."""

    measurements = {
        "bootstrap_latency": _bootstrap_latency_summary(bootstrap_latency),
        "stack_latency_estimate": _stack_latency_summary(stack_latency_estimate),
        "checkpoint_bootstrap_smoke": _checkpoint_smoke_summary(checkpoint_bootstrap_smoke),
        "checkpoint_source_profile": _checkpoint_source_profile_summary(checkpoint_source_profile),
        "range_scale_plan": _range_scale_plan_summary(range_scale_plan),
        "checkpoint_full_layer_gate": _checkpoint_full_layer_gate_summary(
            checkpoint_full_layer_gate
        ),
        "client_decode_smoke": _client_decode_smoke_summary(client_decode_smoke),
        "segment_samples": _segment_sample_summary(segment_samples),
        "all_layer_recurrence": _all_layer_recurrence_summary(all_layer_recurrence),
        "ciphertext_handoff": _ciphertext_handoff_summary(ciphertext_handoff),
    }
    completed_items = _completed_items(measurements)
    remaining_items = _remaining_items(measurements)
    bottlenecks = _bottleneck_assessment(measurements)
    return {
        "version": version,
        "stage": "stage0-status-report",
        "stage0_complete": False,
        "measurement_scope": {
            "stage0_status_report": True,
            "encrypted": False,
            "full_model_correctness_claimed": False,
            "non_success_probe": True,
            "claim": (
                "aggregated Stage 0 status and bottleneck report; not a benchmark success "
                "claim and not full encrypted model correctness"
            ),
        },
        "completed_items": completed_items,
        "remaining_items": remaining_items,
        "measurements": measurements,
        "bottlenecks": bottlenecks,
        "next_bottleneck": _next_bottleneck(measurements, bottlenecks),
    }


def _remaining_items(measurements: dict[str, dict[str, Any]]) -> list[str]:
    all_layer = measurements["all_layer_recurrence"]
    full_gate = measurements["checkpoint_full_layer_gate"]
    if all_layer.get("actual_scheduled_bootstraps") and all_layer.get("bootstrap_probe_only"):
        first_item = (
            "connect scheduled bootstrap probe to true inter-layer ciphertext chain"
            " using full-layer ciphertext trace"
            if full_gate.get("visible_handoff_ciphertext")
            else "connect scheduled bootstrap probe to true inter-layer ciphertext chain"
        )
    elif all_layer.get("actual_scheduled_bootstraps"):
        if full_gate.get("visible_handoff_ciphertext"):
            first_item = (
                "connect full-layer visible ciphertext trace to encrypted next-layer pre-recurrence"
            )
        elif measurements["ciphertext_handoff"].get("no_intermediate_decrypt"):
            first_item = "wire checkpoint gate/out-projection/residual into ciphertext handoff"
        else:
            first_item = (
                "connect scheduled boundary bootstrap smoke to true inter-layer ciphertext handoff"
            )
    else:
        first_item = "run 24-layer encrypted recurrence with scheduled inter-layer bootstraps"
    items = [
        first_item,
        "measure 1024-token average latency or a documented smaller proxy if cost is too high",
        "compare encrypted recurrence outputs against plaintext baseline across all sampled layers",
        "record profiler breakdown for encode/encrypt/eval/bootstrap/decrypt",
    ]
    if not measurements["client_decode_smoke"].get("passed"):
        items.append("include client-side decoding smoke for an inference-shaped path")
    return items


def _bootstrap_latency_summary(payload: dict[str, Any] | None) -> dict[str, Any]:
    if not payload:
        return {"available": False}
    return {
        "available": bool(payload.get("available")),
        "mean_latency_sec": payload.get("mean_latency_sec"),
        "batch_size": payload.get("batch_size"),
        "ring_dimension": payload.get("ring_dimension"),
        "setup_seconds": payload.get("operation_counts", {}).get("setup_seconds"),
        "error_type": payload.get("error_type"),
        "reason": payload.get("reason"),
    }


def _stack_latency_summary(payload: dict[str, Any] | None) -> dict[str, Any]:
    if not payload:
        return {"available": False}
    group = (payload.get("groups") or [{}])[0]
    return {
        "available": True,
        "estimated_sec_per_token": payload.get("max_estimated_latency_sec_per_token"),
        "bootstrap_sec": payload.get("bootstrap_sec"),
        "arithmetic_sec_per_token": group.get("arithmetic_latency_sec_per_token"),
        "bootstrap_sec_per_token": group.get("bootstrap_latency_sec_per_token"),
        "bootstraps": group.get("bootstraps"),
        "sample_count": group.get("sample_count"),
    }


def _checkpoint_smoke_summary(payload: dict[str, Any] | None) -> dict[str, Any]:
    if not payload:
        return {"available": False}
    return {
        "available": True,
        "backend": payload.get("backend"),
        "encrypted": payload.get("encrypted"),
        "latency_sec_per_token": payload.get("latency_sec_per_token"),
        "max_abs_error": payload.get("max_abs_error"),
        "bootstraps": payload.get("operation_counts", {}).get("bootstraps"),
        "state_slots": payload.get("model", {}).get("state_slots"),
        "batch_size": payload.get("ckks", {}).get("batch_size"),
        "ring_dimension": payload.get("ckks", {}).get("ring_dimension"),
        "bootstrap_after_tokens": payload.get("ckks", {}).get("bootstrap_after_tokens"),
    }


def _checkpoint_source_profile_summary(payload: dict[str, Any] | None) -> dict[str, Any]:
    if not payload:
        return {"available": False}
    result = payload.get("result", {})
    global_maxima = result.get("global_maxima", {})
    token_ids = result.get("token_ids", [])
    range_stage = None
    range_layer = None
    if result.get("layers"):
        worst_layer = max(
            result["layers"],
            key=lambda layer: float(layer.get("range_score", 0.0)),
        )
        range_layer = worst_layer.get("layer_index")
        range_stage = worst_layer.get("range_score_stage")
    return {
        "available": True,
        "passed": payload.get("passed"),
        "encrypted": payload.get("measurement_scope", {}).get("encrypted"),
        "full_model_correctness_claimed": payload.get("measurement_scope", {}).get(
            "full_model_correctness_claimed"
        ),
        "seq_len": len(token_ids),
        "layer_count": result.get("layer_count"),
        "d_model": result.get("d_model"),
        "d_state": result.get("d_state"),
        "mimo_rank": result.get("mimo_rank"),
        "top1_token": result.get("top1_token"),
        "top1_top2_gap": result.get("top1_top2_gap"),
        "decay_abs_max": global_maxima.get("decay_abs_max"),
        "high_decay_burst_len": global_maxima.get("high_decay_burst_len"),
        "update_abs_max": global_maxima.get("update_abs_max"),
        "range_score": global_maxima.get("range_score"),
        "range_score_layer": range_layer,
        "range_score_stage": range_stage,
        "logits_abs_max": global_maxima.get("logits_abs_max"),
        "elapsed_sec": result.get("elapsed_sec"),
    }


def _range_scale_plan_summary(payload: dict[str, Any] | None) -> dict[str, Any]:
    if not payload:
        return {"available": False}
    plan = payload.get("scale_plan", {})
    layers = plan.get("layers") or []
    min_output_scale = min((float(layer.get("output_scale", 1.0)) for layer in layers), default=1.0)
    worst_activation = max(
        (float(layer.get("max_activation_abs", 0.0)) for layer in layers),
        default=0.0,
    )
    return {
        "available": True,
        "layer_count": plan.get("layer_count"),
        "activation_target": plan.get("activation_target"),
        "state_target": plan.get("state_target"),
        "encoded_target": plan.get("encoded_target"),
        "activation_tuning_layer_count": plan.get("activation_tuning_layer_count"),
        "state_scaled_layer_count": plan.get("state_scaled_layer_count"),
        "output_scaled_layer_count": plan.get("output_scaled_layer_count"),
        "max_encoded_input_abs": plan.get("max_encoded_input_abs"),
        "max_encoded_delta_abs": plan.get("max_encoded_delta_abs"),
        "max_encoded_output_abs": plan.get("max_encoded_output_abs"),
        "worst_activation_abs": worst_activation,
        "min_output_scale": min_output_scale,
    }


def _checkpoint_full_layer_gate_summary(payload: dict[str, Any] | None) -> dict[str, Any]:
    if not payload:
        return {"available": False}
    result = payload.get("result", {})
    scope = payload.get("measurement_scope", {})
    model = payload.get("model", {})
    ckks = payload.get("ckks", {})
    operation_counts = payload.get("operation_counts", {})
    return {
        "available": True,
        "backend": payload.get("backend"),
        "encrypted": payload.get("encrypted"),
        "passed": payload.get("passed"),
        "max_abs_error": payload.get("max_abs_error", result.get("max_abs_error")),
        "d_model": model.get("d_model", result.get("d_model")),
        "checked_visible_dim": model.get(
            "checked_visible_dim",
            result.get("checked_visible_dim"),
        ),
        "d_state": model.get("d_state", result.get("d_state")),
        "mimo_rank": model.get("mimo_rank", result.get("mimo_rank")),
        "seq_len": model.get("seq_len", result.get("seq_len")),
        "input_mode": model.get("input_mode", result.get("input_mode")),
        "readout_strategy": model.get("readout_strategy", result.get("readout_strategy")),
        "visible_output_scale": model.get(
            "visible_output_scale",
            result.get("visible_output_scale", 1.0),
        ),
        "scale_plan": model.get("scale_plan"),
        "source_style_full_layer_formula": bool(scope.get("source_style_full_layer_formula")),
        "full_visible_output_checked": bool(scope.get("full_visible_output_checked")),
        "partial_visible_output_checked": bool(scope.get("partial_visible_output_checked")),
        "recurrence_ciphertext": bool(result.get("recurrence_ciphertext")),
        "visible_handoff_ciphertext": bool(result.get("visible_handoff_ciphertext")),
        "no_intermediate_decrypt": bool(result.get("no_intermediate_decrypt")),
        "full_model_correctness_claimed": bool(scope.get("full_model_correctness_claimed")),
        "plaintext_precomputed_stages": scope.get("plaintext_precomputed_stages", []),
        "rotation_count": ckks.get("rotation_count"),
        "decrypt_count": operation_counts.get("decrypt"),
    }


def _scaled_full_gate_validated(full_gate: dict[str, Any]) -> bool:
    scale = full_gate.get("visible_output_scale", 1.0)
    return (
        bool(full_gate.get("passed"))
        and bool(full_gate.get("full_visible_output_checked"))
        and isinstance(scale, int | float)
        and 0.0 < float(scale) < 1.0
    )


def _client_decode_smoke_summary(payload: dict[str, Any] | None) -> dict[str, Any]:
    if not payload:
        return {"available": False}
    result = payload.get("result", {})
    decode_steps = result.get("decode_steps", [])
    first_step = decode_steps[0] if decode_steps else {}
    return {
        "available": True,
        "passed": payload.get("passed"),
        "layer_count": result.get("layer_count"),
        "d_model": result.get("d_model"),
        "d_state": result.get("d_state"),
        "mimo_rank": result.get("mimo_rank"),
        "vocab_size": result.get("vocab_size"),
        "new_token_ids": result.get("new_token_ids"),
        "top1_top2_gap": first_step.get("top1_top2_gap"),
        "hidden_abs_max": result.get("hidden_abs_max"),
        "logits_abs_max": result.get("logits_abs_max"),
        "client_side_lm_head": result.get("client_side_lm_head"),
        "client_side_argmax": result.get("client_side_argmax"),
        "encrypted_argmax": result.get("encrypted_argmax"),
        "full_model_correctness_claimed": result.get("full_model_correctness_claimed"),
        "elapsed_sec": result.get("elapsed_sec"),
    }


def _segment_sample_summary(payload: dict[str, Any] | None) -> dict[str, Any]:
    if not payload:
        return {"available": False}
    results = payload.get("results", [])
    successful = [result for result in results if result.get("returncode") == 0]
    bootstrap_counts = [
        result.get("operation_counts", {}).get("bootstraps", 0) for result in successful
    ]
    return {
        "available": True,
        "sample_count": payload.get("sample_count", len(results)),
        "success_count": payload.get("success_count", len(successful)),
        "bootstrap_enabled_sample_count": sum(1 for count in bootstrap_counts if count),
        "max_latency_sec_per_token": max(
            (float(result["latency_sec_per_token"]) for result in successful),
            default=None,
        ),
        "max_abs_error": max(
            (float(result["max_abs_error"]) for result in successful),
            default=None,
        ),
    }


def _all_layer_recurrence_summary(payload: dict[str, Any] | None) -> dict[str, Any]:
    if not payload:
        return {"available": False}
    summary = payload.get("summary", {})
    scope = payload.get("measurement_scope", {})
    return {
        "available": True,
        "layer_count": summary.get("layer_count"),
        "success_count": summary.get("success_count"),
        "failure_count": summary.get("failure_count"),
        "arithmetic_sec_per_token": summary.get("arithmetic_sec_per_token"),
        "scheduled_bootstraps": summary.get("scheduled_bootstraps"),
        "bootstrap_sec_per_token": summary.get("bootstrap_sec_per_token"),
        "estimated_scheduled_sec_per_token": summary.get("estimated_scheduled_sec_per_token"),
        "actual_scheduled_bootstraps": summary.get("actual_scheduled_bootstraps"),
        "actual_bootstrap_sec_per_token": summary.get("actual_bootstrap_sec_per_token"),
        "actual_scheduled_sec_per_token": summary.get("actual_scheduled_sec_per_token"),
        "actual_bootstrap_max_abs_error": summary.get("actual_bootstrap_max_abs_error"),
        "max_abs_error": summary.get("max_abs_error"),
        "encrypted_chain": bool(scope.get("encrypted_chain", False)),
        "bootstrap_probe_only": bool(scope.get("bootstrap_probe_only", False)),
        "layer_inputs_plaintext_precomputed": bool(
            scope.get("layer_inputs_plaintext_precomputed", False)
        ),
        "full_layer_correctness_claimed": bool(scope.get("full_layer_correctness_claimed", False)),
        "full_model_correctness_claimed": bool(scope.get("full_model_correctness_claimed", False)),
        "claim": scope.get("claim"),
    }


def _ciphertext_handoff_summary(payload: dict[str, Any] | None) -> dict[str, Any]:
    if not payload:
        return {"available": False}
    result = payload.get("result", {})
    stats = result.get("backend_stats", {})
    return {
        "available": True,
        "backend": payload.get("backend"),
        "encrypted": payload.get("encrypted"),
        "no_intermediate_decrypt": payload.get("no_intermediate_decrypt"),
        "layer_count": result.get("layer_count"),
        "bootstrap_after_layers": result.get("bootstrap_after_layers"),
        "decrypt_count": stats.get("decrypt_count"),
        "bootstrap_count": stats.get("bootstrap_count"),
        "latency_sec": result.get("latency_sec"),
        "max_abs_error": result.get("max_abs_error"),
    }


def _completed_items(measurements: dict[str, dict[str, Any]]) -> list[str]:
    items = [
        "import real Mamba checkpoint into recurrence smoke path",
        "run encrypted OpenFHE recurrence correctness smoke",
    ]
    if measurements["bootstrap_latency"].get("available"):
        items.append("measure OpenFHE CKKS bootstrap latency at Stage 0 slot scale")
    if measurements["stack_latency_estimate"].get("available"):
        items.append("estimate 24-layer recurrence stack latency from segment samples")
    if measurements["checkpoint_bootstrap_smoke"].get("bootstraps"):
        items.append("insert and execute an actual bootstrap in real-checkpoint recurrence")
    if measurements["checkpoint_source_profile"].get("available"):
        items.append("profile real checkpoint ranges, decay bursts, updates, and logit gaps")
    if measurements["range_scale_plan"].get("available"):
        items.append("derive source-profile range scale plan for recurrence and residual encoding")
    full_gate = measurements["checkpoint_full_layer_gate"]
    if (
        full_gate.get("passed")
        and full_gate.get("visible_handoff_ciphertext")
        and full_gate.get("no_intermediate_decrypt")
    ):
        items.append(
            "run source-style full-layer ciphertext gate through recurrence, skip, gate, "
            "out-projection, and residual"
        )
    if full_gate.get("passed") and _scaled_full_gate_validated(full_gate):
        items.append("validate range-scaled full-layer visible output ciphertext")
    if measurements["client_decode_smoke"].get("passed"):
        items.append("run real checkpoint source-style client-side decode smoke")
    if measurements["segment_samples"].get("bootstrap_enabled_sample_count"):
        items.append("sample representative recurrence segment with bootstrap enabled")
    all_layer = measurements["all_layer_recurrence"]
    if all_layer.get("layer_count") and all_layer.get("success_count") == all_layer.get(
        "layer_count"
    ):
        items.append("measure OpenFHE recurrence arithmetic for every selected layer")
    scheduled_bootstraps = all_layer.get("scheduled_bootstraps")
    if (
        scheduled_bootstraps
        and all_layer.get("actual_scheduled_bootstraps") == scheduled_bootstraps
    ):
        if all_layer.get("bootstrap_probe_only"):
            items.append("execute scheduled bootstrap probe for the 24-layer recurrence plan")
        elif all_layer.get("encrypted_chain"):
            items.append(
                "execute all scheduled boundary bootstraps in the 24-layer encrypted chain"
            )
        else:
            items.append("execute all scheduled boundary bootstraps in the 24-layer probe")
    handoff = measurements["ciphertext_handoff"]
    if handoff.get("encrypted") and handoff.get("no_intermediate_decrypt"):
        items.append("run encrypted ciphertext handoff smoke without intermediate decrypts")
    return items


def _bottleneck_assessment(measurements: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    bottlenecks: list[dict[str, Any]] = []
    profile = measurements["checkpoint_source_profile"]
    full_gate = measurements["checkpoint_full_layer_gate"]
    if profile.get("available"):
        range_score = profile.get("range_score")
        range_stage = profile.get("range_score_stage")
        if isinstance(range_score, int | float):
            range_target = 6.0
            if range_score > range_target:
                residual_stage = range_stage in {"final_block_delta", "final_block_output"}
                if not (residual_stage and _scaled_full_gate_validated(full_gate)):
                    reason = (
                        "source-style residual/output range exceeds the current encoded target"
                        if residual_stage
                        else "source-style activation range exceeds the current polynomial/FHE "
                        "planning target"
                    )
                    next_action = (
                        "apply the range scale plan and validate scaled recurrence/residual "
                        "artifacts"
                        if residual_stage
                        else (
                            "run range-aware calibration or LoRA before claiming full block "
                            "stability"
                        )
                    )
                    bottlenecks.append(
                        {
                            "name": "range",
                            "severity": "high" if range_score > 100.0 else "medium",
                            "value": range_score,
                            "threshold": range_target,
                            "stage": range_stage,
                            "reason": reason,
                            "next_action": next_action,
                        }
                    )
    scale_plan = measurements["range_scale_plan"]
    activation_count = scale_plan.get("activation_tuning_layer_count")
    worst_activation = scale_plan.get("worst_activation_abs")
    activation_target = scale_plan.get("activation_target")
    if (
        scale_plan.get("available")
        and isinstance(activation_count, int)
        and activation_count > 0
        and isinstance(worst_activation, int | float)
        and isinstance(activation_target, int | float)
    ):
        bottlenecks.append(
            {
                "name": "activation_tuning",
                "severity": "high" if worst_activation > 4 * activation_target else "medium",
                "value": worst_activation,
                "threshold": activation_target,
                "affected_layers": activation_count,
                "reason": "some nonlinear inputs remain outside the polynomial target range",
                "next_action": "run range-aware LoRA or activation calibration on affected layers",
            }
        )
    if profile.get("available"):
        burst_len = profile.get("high_decay_burst_len")
        seq_len = profile.get("seq_len")
        if isinstance(burst_len, int | float) and isinstance(seq_len, int | float):
            if seq_len >= 16 and burst_len >= seq_len:
                bottlenecks.append(
                    {
                        "name": "decay_burst",
                        "severity": "high",
                        "value": burst_len,
                        "threshold": seq_len,
                        "reason": "near-1 decay burst spans the profiled prompt",
                        "next_action": (
                            "do not claim a small fixed effective window without contraction "
                            "fine-tuning or broader prompt profiling"
                        ),
                    }
                )
            elif burst_len >= max(8.0, 0.5 * seq_len):
                bottlenecks.append(
                    {
                        "name": "decay_burst",
                        "severity": "medium",
                        "value": burst_len,
                        "threshold": max(8.0, 0.5 * seq_len),
                        "reason": "near-1 decay burst is large relative to the prompt length",
                        "next_action": "profile longer prompts and adversarial repetitions",
                    }
                )

    estimate = measurements["stack_latency_estimate"]
    arithmetic = estimate.get("arithmetic_sec_per_token")
    bootstrap = estimate.get("bootstrap_sec_per_token")
    if (
        isinstance(arithmetic, int | float)
        and isinstance(bootstrap, int | float)
        and bootstrap > arithmetic
    ):
        bottlenecks.append(
            {
                "name": "bootstrap_latency",
                "severity": "high" if bootstrap > 2 * arithmetic else "medium",
                "value": bootstrap,
                "threshold": arithmetic,
                "reason": "bootstrap latency dominates the current Stage 0 recurrence estimate",
                "next_action": (
                    "connect measured bootstrap cost to scheduled ciphertext-chain execution"
                ),
            }
        )
    if not measurements["checkpoint_bootstrap_smoke"].get("bootstraps"):
        bottlenecks.append(
            {
                "name": "bootstrap_execution",
                "severity": "medium",
                "value": 0,
                "threshold": 1,
                "reason": "no real-checkpoint recurrence smoke with an actual bootstrap is present",
                "next_action": (
                    "execute a real-checkpoint recurrence smoke with an actual bootstrap"
                ),
            }
        )
    all_layer = measurements["all_layer_recurrence"]
    if all_layer.get("actual_scheduled_bootstraps") and all_layer.get("bootstrap_probe_only"):
        bottlenecks.append(
            {
                "name": "ciphertext_chain",
                "severity": "high",
                "value": all_layer.get("actual_scheduled_bootstraps"),
                "threshold": all_layer.get("scheduled_bootstraps"),
                "reason": (
                    "scheduled bootstraps are still probe-only, not a true ciphertext handoff chain"
                ),
                "next_action": (
                    "connect scheduled bootstrap probe to true inter-layer ciphertext chain"
                ),
            }
        )
    decode = measurements["client_decode_smoke"]
    if decode.get("client_side_argmax") and not decode.get("encrypted_argmax"):
        bottlenecks.append(
            {
                "name": "decoding",
                "severity": "medium",
                "value": "client-side-argmax",
                "threshold": "encrypted-argmax",
                "reason": "generation currently relies on client-side lm_head/argmax",
                "next_action": "keep interactive decoding as baseline and defer CutMax to Stage 2",
            }
        )
    order = {"high": 0, "medium": 1, "low": 2}
    return sorted(bottlenecks, key=lambda item: order.get(str(item["severity"]), 3))


def _next_bottleneck(
    measurements: dict[str, dict[str, Any]],
    bottlenecks: list[dict[str, Any]],
) -> str:
    if bottlenecks:
        return str(bottlenecks[0]["reason"])
    if not measurements["checkpoint_bootstrap_smoke"].get("bootstraps"):
        return "execute a real-checkpoint recurrence smoke with an actual bootstrap"
    return "connect token/layer bootstrap probes to full 24-layer scheduled execution"
