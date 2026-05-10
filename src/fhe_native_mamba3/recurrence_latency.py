"""Latency estimates built from recurrence sweeps and OpenFHE segment samples."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from statistics import mean
from typing import Any


@dataclass(frozen=True)
class RecurrenceStackLatencyGroup:
    """Latency estimate for one recurrence-source/sequence group."""

    recurrence_source: str
    seq_len: int
    input_mode: str
    readout_strategy: str
    layer_count: int
    segment_count: int
    bootstraps: int
    sample_count: int
    sample_layers: tuple[int, ...]
    mean_layer_latency_sec_per_token: float
    min_layer_latency_sec_per_token: float
    max_layer_latency_sec_per_token: float
    arithmetic_latency_sec_per_token: float
    bootstrap_latency_sec_per_token: float
    estimated_latency_sec_per_token: float
    operation_counts: dict[str, int]

    def to_json_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["sample_layers"] = list(self.sample_layers)
        return payload


def estimate_recurrence_stack_latency(
    sweep_payload: dict[str, Any],
    samples_payload: dict[str, Any],
    *,
    bootstrap_sec: float,
) -> dict[str, Any]:
    """Estimate full-stack recurrence latency from sweep metadata and samples."""

    if bootstrap_sec < 0:
        msg = "bootstrap_sec must be non-negative"
        raise ValueError(msg)
    groups = [
        _estimate_group(
            group,
            sweep_rows=sweep_payload["rows"],
            sample_results=samples_payload["results"],
            bootstrap_sec=bootstrap_sec,
        )
        for group in sweep_payload["summary"]["bootstrap_schedules"]["groups"]
    ]
    return {
        "stage": "recurrence-stack-latency-estimate",
        "sweep_stage": sweep_payload.get("stage"),
        "sample_stage": samples_payload.get("stage"),
        "bootstrap_sec": bootstrap_sec,
        "group_count": len(groups),
        "max_estimated_latency_sec_per_token": max(
            (group.estimated_latency_sec_per_token for group in groups),
            default=0.0,
        ),
        "groups": [group.to_json_dict() for group in groups],
    }


def _estimate_group(
    group: dict[str, Any],
    *,
    sweep_rows: list[dict[str, Any]],
    sample_results: list[dict[str, Any]],
    bootstrap_sec: float,
) -> RecurrenceStackLatencyGroup:
    group_key = _group_key(group)
    rows = [row for row in sweep_rows if _row_key(row) == group_key]
    samples = [
        sample
        for sample in sample_results
        if _sample_key(sample) == group_key and sample.get("returncode") == 0
    ]
    if not samples:
        msg = f"no successful OpenFHE segment samples for group={group_key}"
        raise ValueError(msg)

    sample_latencies = [float(sample["latency_sec_per_token"]) for sample in samples]
    layer_count = len(group["layer_indices"])
    mean_layer_latency = mean(sample_latencies)
    arithmetic_latency = layer_count * mean_layer_latency
    bootstrap_latency = int(group["bootstraps"]) * bootstrap_sec
    return RecurrenceStackLatencyGroup(
        recurrence_source=str(group["recurrence_source"]),
        seq_len=int(group["seq_len"]),
        input_mode=str(group["input_mode"]),
        readout_strategy=str(group["readout_strategy"]),
        layer_count=layer_count,
        segment_count=int(group["segment_count"]),
        bootstraps=int(group["bootstraps"]),
        sample_count=len(samples),
        sample_layers=tuple(int(sample["layer_index"]) for sample in samples),
        mean_layer_latency_sec_per_token=mean_layer_latency,
        min_layer_latency_sec_per_token=min(sample_latencies),
        max_layer_latency_sec_per_token=max(sample_latencies),
        arithmetic_latency_sec_per_token=arithmetic_latency,
        bootstrap_latency_sec_per_token=bootstrap_latency,
        estimated_latency_sec_per_token=arithmetic_latency + bootstrap_latency,
        operation_counts=_sum_operation_counts(rows),
    )


def _sum_operation_counts(rows: list[dict[str, Any]]) -> dict[str, int]:
    totals: dict[str, int] = {}
    for row in rows:
        for name, value in row["operation_counts"].items():
            if isinstance(value, int):
                totals[name] = totals.get(name, 0) + value
    return totals


def _group_key(payload: dict[str, Any]) -> tuple[str, int, str, str]:
    return (
        str(payload["recurrence_source"]),
        int(payload["seq_len"]),
        str(payload["input_mode"]),
        str(payload["readout_strategy"]),
    )


def _row_key(row: dict[str, Any]) -> tuple[str, int, str, str]:
    return (
        str(row["recurrence_source"]),
        int(row["seq_len"]),
        str(row["input_mode"]),
        str(row["readout_strategy"]),
    )


def _sample_key(sample: dict[str, Any]) -> tuple[str, int, str, str]:
    return (
        str(sample["recurrence_source"]),
        int(sample["seq_len"]),
        str(sample["input_mode"]),
        str(sample["readout_strategy"]),
    )
