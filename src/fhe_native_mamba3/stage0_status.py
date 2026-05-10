"""Stage 0 status report assembly from measured JSON artifacts."""

from __future__ import annotations

from typing import Any


def build_stage0_status_report(
    *,
    version: str,
    bootstrap_latency: dict[str, Any] | None = None,
    stack_latency_estimate: dict[str, Any] | None = None,
    checkpoint_bootstrap_smoke: dict[str, Any] | None = None,
    segment_samples: dict[str, Any] | None = None,
    all_layer_recurrence: dict[str, Any] | None = None,
    ciphertext_handoff: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a compact Stage 0 progress report from existing measurements."""

    measurements = {
        "bootstrap_latency": _bootstrap_latency_summary(bootstrap_latency),
        "stack_latency_estimate": _stack_latency_summary(stack_latency_estimate),
        "checkpoint_bootstrap_smoke": _checkpoint_smoke_summary(checkpoint_bootstrap_smoke),
        "segment_samples": _segment_sample_summary(segment_samples),
        "all_layer_recurrence": _all_layer_recurrence_summary(all_layer_recurrence),
        "ciphertext_handoff": _ciphertext_handoff_summary(ciphertext_handoff),
    }
    completed_items = _completed_items(measurements)
    remaining_items = _remaining_items(measurements)
    return {
        "version": version,
        "stage": "stage0-status-report",
        "stage0_complete": False,
        "completed_items": completed_items,
        "remaining_items": remaining_items,
        "measurements": measurements,
        "next_bottleneck": _next_bottleneck(measurements),
    }


def _remaining_items(measurements: dict[str, dict[str, Any]]) -> list[str]:
    all_layer = measurements["all_layer_recurrence"]
    first_item = (
        "wire checkpoint gate/out-projection/residual into ciphertext handoff"
        if measurements["ciphertext_handoff"].get("no_intermediate_decrypt")
        else "connect scheduled boundary bootstrap smoke to true inter-layer ciphertext handoff"
    )
    first_item = (
        first_item
        if all_layer.get("actual_scheduled_bootstraps")
        else "run 24-layer encrypted recurrence with scheduled inter-layer bootstraps"
    )
    return [
        first_item,
        "measure 1024-token average latency or a documented smaller proxy if cost is too high",
        "compare encrypted recurrence outputs against plaintext baseline across all sampled layers",
        "record profiler breakdown for encode/encrypt/eval/bootstrap/decrypt",
        "include client-side decoding smoke for an inference-shaped path",
    ]


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
    if measurements["segment_samples"].get("bootstrap_enabled_sample_count"):
        items.append("sample representative recurrence segment with bootstrap enabled")
    all_layer = measurements["all_layer_recurrence"]
    if all_layer.get("layer_count") and all_layer.get("success_count") == all_layer.get(
        "layer_count"
    ):
        items.append("measure OpenFHE recurrence arithmetic for every selected layer")
    if all_layer.get("scheduled_bootstraps") is not None and all_layer.get(
        "actual_scheduled_bootstraps"
    ) == all_layer.get("scheduled_bootstraps"):
        items.append("execute all scheduled boundary bootstraps in the 24-layer probe")
    handoff = measurements["ciphertext_handoff"]
    if handoff.get("encrypted") and handoff.get("no_intermediate_decrypt"):
        items.append("run encrypted ciphertext handoff smoke without intermediate decrypts")
    return items


def _next_bottleneck(measurements: dict[str, dict[str, Any]]) -> str:
    estimate = measurements["stack_latency_estimate"]
    arithmetic = estimate.get("arithmetic_sec_per_token")
    bootstrap = estimate.get("bootstrap_sec_per_token")
    if (
        isinstance(arithmetic, int | float)
        and isinstance(bootstrap, int | float)
        and bootstrap > arithmetic
    ):
        return "bootstrap latency dominates the current Stage 0 recurrence estimate"
    if not measurements["checkpoint_bootstrap_smoke"].get("bootstraps"):
        return "execute a real-checkpoint recurrence smoke with an actual bootstrap"
    return "connect token/layer bootstrap probes to full 24-layer scheduled execution"
