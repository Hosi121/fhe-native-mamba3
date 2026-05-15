"""Compare Stage 1 model-layout handoff artifacts across model shapes."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class Stage1ModelHandoffArtifactSummary:
    """Normalized fields from one native/FIDESlib model-layout handoff artifact."""

    d_model: int | None
    mimo_rank: int | None
    d_state: int | None
    rank_pad: int | None
    payload_count: int
    eval_seconds: float | None
    setup_seconds: float | None
    fixed_seconds: float
    peak_rss_gib: float | None
    required_application_rotation_key_count: int | None
    max_abs_error: float | None
    diagnostic_max_abs_error: float | None
    model_layout_handoff_max_abs_error: float | None
    payload_chain_reference_max_abs_error: float | None
    output_model_poly_vs_exact_max_abs_error: float | None
    output_model_poly_vs_exact_reference_steps: int | None
    operation_counts: dict[str, float]


@dataclass(frozen=True)
class Stage1ModelHandoffScalingReport:
    """Artifact-level comparison for same-depth model-layout handoff scaling."""

    stage: str
    passed: bool
    base: Stage1ModelHandoffArtifactSummary
    scaled: Stage1ModelHandoffArtifactSummary
    operation_count_deltas: dict[str, float]
    operation_count_ratios: dict[str, float | None]
    measurement_scope: dict[str, Any]

    def to_json_dict(self) -> dict[str, Any]:
        return asdict(self)


def build_stage1_model_handoff_scaling_report(
    *,
    base_payload: dict[str, Any],
    scaled_payload: dict[str, Any],
) -> Stage1ModelHandoffScalingReport:
    """Build a report comparing two same-depth model-layout handoff artifacts."""

    base = _summarize_artifact(base_payload)
    scaled = _summarize_artifact(scaled_payload)
    if base.payload_count != scaled.payload_count:
        msg = (
            "base and scaled artifacts must have the same payload_count; "
            f"got {base.payload_count} and {scaled.payload_count}"
        )
        raise ValueError(msg)
    op_deltas = _operation_deltas(base.operation_counts, scaled.operation_counts)
    op_ratios = _operation_ratios(base.operation_counts, scaled.operation_counts)
    return Stage1ModelHandoffScalingReport(
        stage="stage1-model-layout-handoff-scaling-report",
        passed=bool(base_payload.get("passed", True)) and bool(scaled_payload.get("passed", True)),
        base=base,
        scaled=scaled,
        operation_count_deltas=op_deltas,
        operation_count_ratios=op_ratios,
        measurement_scope={
            "artifact_level_report": True,
            "stage1_model_layout_handoff_scaling_report": True,
            "full_model_correctness_claimed": False,
            "payload_count": base.payload_count,
            "claim": (
                "Compares existing native/FIDESlib model-layout handoff artifacts "
                "with the same payload_count; this report does not execute FHE."
            ),
        },
    )


def _summarize_artifact(payload: dict[str, Any]) -> Stage1ModelHandoffArtifactSummary:
    return Stage1ModelHandoffArtifactSummary(
        d_model=_int_value(payload, "d_model"),
        mimo_rank=_int_value(payload, "mimo_rank"),
        d_state=_int_value(payload, "d_state"),
        rank_pad=_int_value(payload, "rank_pad"),
        payload_count=_payload_count(payload),
        eval_seconds=_float_value(payload, "eval_seconds"),
        setup_seconds=_float_value(payload, "setup_seconds"),
        fixed_seconds=_fixed_seconds(payload),
        peak_rss_gib=_float_value(payload, "peak_rss_gib"),
        required_application_rotation_key_count=_int_value(
            payload,
            "required_application_rotation_key_count",
        ),
        max_abs_error=_float_value(payload, "max_abs_error"),
        diagnostic_max_abs_error=_float_value(payload, "diagnostic_max_abs_error"),
        model_layout_handoff_max_abs_error=_float_value(
            payload,
            "model_layout_handoff_max_abs_error",
        ),
        payload_chain_reference_max_abs_error=_float_value(
            payload,
            "payload_chain_reference_max_abs_error",
        ),
        output_model_poly_vs_exact_max_abs_error=_float_value(
            payload,
            "output_model_poly_vs_exact_max_abs_error",
        ),
        output_model_poly_vs_exact_reference_steps=_int_value(
            payload,
            "output_model_poly_vs_exact_reference_steps",
        ),
        operation_counts=_operation_counts(payload),
    )


def _payload_count(payload: dict[str, Any]) -> int:
    value = _raw_value(payload, "payload_count")
    if value is None:
        artifact = payload.get("artifact")
        if isinstance(artifact, dict):
            payloads = artifact.get("payloads")
            if isinstance(payloads, list):
                return len(payloads)
    if value is None:
        msg = "payload_count is required"
        raise ValueError(msg)
    return int(value)


def _fixed_seconds(payload: dict[str, Any]) -> float:
    total = 0.0
    found = False
    for key in (
        "setup_seconds",
        "keygen_seconds",
        "rotate_keygen_seconds",
        "load_context_seconds",
    ):
        value = _float_value(payload, key)
        if value is not None:
            total += value
            found = True
    return total if found else 0.0


def _int_value(payload: dict[str, Any], key: str) -> int | None:
    value = _raw_value(payload, key)
    return None if value is None else int(value)


def _float_value(payload: dict[str, Any], key: str) -> float | None:
    value = _raw_value(payload, key)
    return None if value is None else float(value)


def _raw_value(payload: dict[str, Any], key: str) -> Any:
    if payload.get(key) is not None:
        return payload[key]
    for section_name in ("parameters", "measurements", "timing", "config"):
        section = payload.get(section_name)
        if isinstance(section, dict) and section.get(key) is not None:
            return section[key]
    artifact = payload.get("artifact")
    if isinstance(artifact, dict):
        if artifact.get(key) is not None:
            return artifact[key]
        payloads = artifact.get("payloads")
        if isinstance(payloads, list) and payloads:
            first = payloads[0]
            if isinstance(first, dict):
                config = first.get("config")
                if isinstance(config, dict) and config.get(key) is not None:
                    return config[key]
    return None


def _operation_counts(payload: dict[str, Any]) -> dict[str, float]:
    counts = payload.get("operation_counts")
    if not isinstance(counts, dict):
        return {}
    return {
        str(key): float(value) for key, value in counts.items() if isinstance(value, int | float)
    }


def _operation_deltas(
    base_counts: dict[str, float],
    scaled_counts: dict[str, float],
) -> dict[str, float]:
    return {
        key: scaled_counts.get(key, 0.0) - base_counts.get(key, 0.0)
        for key in _operation_keys(base_counts, scaled_counts)
    }


def _operation_ratios(
    base_counts: dict[str, float],
    scaled_counts: dict[str, float],
) -> dict[str, float | None]:
    ratios: dict[str, float | None] = {}
    for key in _operation_keys(base_counts, scaled_counts):
        base_value = base_counts.get(key, 0.0)
        ratios[key] = None if base_value == 0.0 else scaled_counts.get(key, 0.0) / base_value
    return ratios


def _operation_keys(
    base_counts: dict[str, float],
    scaled_counts: dict[str, float],
) -> tuple[str, ...]:
    return tuple(
        sorted(
            set(base_counts)
            | set(scaled_counts)
            | {"rotations", "ct_pt_mul", "ct_ct_mul", "adds", "bootstraps"}
        )
    )


__all__ = [
    "Stage1ModelHandoffArtifactSummary",
    "Stage1ModelHandoffScalingReport",
    "build_stage1_model_handoff_scaling_report",
]
