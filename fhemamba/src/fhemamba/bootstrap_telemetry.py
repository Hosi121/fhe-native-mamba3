"""Analysis helpers for native Mamba-2 bootstrap event telemetry."""

from __future__ import annotations

import re
from collections import defaultdict
from typing import Any

_LAYER_PREFIX = re.compile(r"^t\d+\.L\d+\.")
_TOKEN_PREFIX = re.compile(r"^t\d+\.")
_STATE_SUFFIX = re.compile(r"(state(?:_post)?|fifo)\d+$")


def bootstrap_checkpoint_family(checkpoint: str) -> str:
    """Collapse token/layer and state-slot suffixes into a stable family."""
    family = _LAYER_PREFIX.sub("", checkpoint)
    family = _TOKEN_PREFIX.sub("", family)
    return _STATE_SUFFIX.sub(r"\1", family)


def bootstrap_parent_phase(family: str) -> str | None:
    """Return the inclusive phase timer that contains this refresh family."""
    if family.startswith("gated_"):
        return "gated_norm"
    if family.startswith("decay_sq"):
        return "decay_exp_poly"
    if family.startswith("rms_newton"):
        return "block_norm"
    if family.startswith("final_rms_newton"):
        return "final_norm"
    return None


def build_bootstrap_telemetry_report(payload: dict[str, Any]) -> dict[str, Any]:
    """Summarize event cost and trigger margins from a native artifact."""
    parameters = payload.get("parameters")
    measurements = payload.get("measurements")
    timing = payload.get("timing")
    if not isinstance(parameters, dict) or not isinstance(measurements, dict):
        raise ValueError("artifact must include parameters and measurements")
    depth = parameters.get("multiplicative_depth")
    events = measurements.get("bootstrap_events")
    if not isinstance(depth, int) or depth <= 0:
        raise ValueError("artifact must include a positive multiplicative depth")
    if not isinstance(events, list) or not events:
        raise ValueError("artifact must include non-empty bootstrap_events")

    families: dict[str, dict[str, Any]] = defaultdict(
        lambda: {
            "events": 0,
            "physical_bootstraps": 0,
            "seconds": 0.0,
            "min_trigger_gap": None,
            "max_trigger_gap": None,
            "min_refresh_gain": None,
            "max_refresh_gain": None,
        }
    )
    normalized_events: list[dict[str, Any]] = []
    for raw in events:
        if not isinstance(raw, dict):
            raise ValueError("bootstrap event entries must be objects")
        checkpoint = str(raw["checkpoint"])
        level_before = int(raw["level_before"])
        level_after = int(raw["level_after"])
        requirement = int(raw["requirement"])
        headroom = int(raw["policy_headroom"])
        physical = int(raw["physical_bootstraps"])
        seconds = float(raw["seconds"])
        bound = float(raw["bound"])
        meta_bts = bool(raw["meta_bts"])
        if physical <= 0 or seconds < 0.0:
            raise ValueError(
                "bootstrap event physical count must be positive and latency nonnegative"
            )
        family = bootstrap_checkpoint_family(checkpoint)
        required_with_policy = requirement + headroom + int(meta_bts)
        available_before = depth - level_before
        trigger_gap = required_with_policy - available_before
        refresh_gain = level_before - level_after
        summary = families[family]
        summary["events"] += 1
        summary["physical_bootstraps"] += physical
        summary["seconds"] += seconds
        for prefix, value in (("trigger_gap", trigger_gap), ("refresh_gain", refresh_gain)):
            minimum = summary[f"min_{prefix}"]
            maximum = summary[f"max_{prefix}"]
            summary[f"min_{prefix}"] = value if minimum is None else min(minimum, value)
            summary[f"max_{prefix}"] = value if maximum is None else max(maximum, value)
        normalized_events.append(
            {
                "checkpoint": checkpoint,
                "family": family,
                "level_before": level_before,
                "level_after": level_after,
                "requirement": requirement,
                "policy_headroom": headroom,
                "available_before": available_before,
                "trigger_gap": trigger_gap,
                "refresh_gain": refresh_gain,
                "physical_bootstraps": physical,
                "seconds": seconds,
                "bound": bound,
                "carried": bool(raw["carried"]),
                "meta_bts": meta_bts,
            }
        )

    ordered_families = sorted(
        ({"family": family, **summary} for family, summary in families.items()),
        key=lambda item: (-float(item["seconds"]), str(item["family"])),
    )
    physical_total = sum(int(event["physical_bootstraps"]) for event in normalized_events)
    recorded_total = int(measurements.get("executed_bootstrap_count", physical_total))
    per_token_counts = measurements.get("per_token_bootstrap_count")
    recorded_logical_total = (
        sum(int(count) for count in per_token_counts)
        if isinstance(per_token_counts, list)
        else len(normalized_events)
    )
    seconds_total = sum(float(event["seconds"]) for event in normalized_events)
    recorded_seconds = (
        float(timing["bootstrap_eval_seconds"])
        if isinstance(timing, dict) and "bootstrap_eval_seconds" in timing
        else seconds_total
    )
    logical_count_matches = len(normalized_events) == recorded_logical_total
    physical_count_matches = physical_total == recorded_total
    seconds_match = abs(seconds_total - recorded_seconds) <= 1e-6 * max(1.0, recorded_seconds)
    phase_timings = payload.get("phase_timings")
    phase_accounting: dict[str, Any] | None = None
    if isinstance(phase_timings, dict):
        nested_by_phase: dict[str, float] = defaultdict(float)
        for event in normalized_events:
            parent = bootstrap_parent_phase(str(event["family"]))
            if parent is not None:
                nested_by_phase[parent] += float(event["seconds"])
        adjustments = []
        inclusive_total = 0.0
        exclusive_total = 0.0
        for phase, raw_seconds in phase_timings.items():
            inclusive = float(raw_seconds)
            nested = nested_by_phase.get(str(phase), 0.0)
            exclusive = inclusive - nested
            inclusive_total += inclusive
            exclusive_total += exclusive
            if nested > 0.0:
                adjustments.append(
                    {
                        "phase": str(phase),
                        "inclusive_seconds": inclusive,
                        "nested_bootstrap_seconds": nested,
                        "exclusive_seconds": exclusive,
                    }
                )
        eval_seconds = (
            float(timing["eval_seconds"])
            if isinstance(timing, dict) and "eval_seconds" in timing
            else exclusive_total
        )
        phase_accounting = {
            "inclusive_seconds": inclusive_total,
            "nested_bootstrap_seconds": sum(nested_by_phase.values()),
            "exclusive_seconds": exclusive_total,
            "recorded_eval_seconds": eval_seconds,
            "unattributed_seconds": eval_seconds - exclusive_total,
            "exclusive_reconciles_eval": abs(eval_seconds - exclusive_total)
            <= 1e-3 * max(1.0, eval_seconds),
            "adjustments": adjustments,
        }
    phase_reconciled = phase_accounting is None or bool(
        phase_accounting["exclusive_reconciles_eval"]
    )
    return {
        "multiplicative_depth": depth,
        "event_count": len(normalized_events),
        "recorded_logical_events": recorded_logical_total,
        "logical_count_matches": logical_count_matches,
        "physical_bootstraps": physical_total,
        "recorded_physical_bootstraps": recorded_total,
        "physical_count_matches": physical_count_matches,
        "seconds": seconds_total,
        "recorded_seconds": recorded_seconds,
        "seconds_match": seconds_match,
        "telemetry_reconciled": (
            logical_count_matches and physical_count_matches and seconds_match and phase_reconciled
        ),
        "phase_accounting": phase_accounting,
        "families": ordered_families,
        "events": normalized_events,
        "measurement_scope": {
            "global_bootstrap_placement_optimized": False,
            "claim": (
                "Profiles native bootstrap trigger margins, costs, and nested phase "
                "overlap; it does not claim that any checkpoint can yet be removed."
            ),
        },
    }
