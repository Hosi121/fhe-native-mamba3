"""Reports over client-side checkpoint decode smoke artifacts."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class ClientDecodeReportRow:
    """One client-side decode artifact summary."""

    source: str
    prompt_token_ids: tuple[int, ...]
    new_token_ids: tuple[int, ...]
    selected_token: int | None
    top1_top2_gap: float | None
    top1_score: float | None
    top2_score: float | None
    output_payload_width: int | None
    client_decrypt_count: int
    elapsed_sec: float
    hidden_abs_max: float
    logits_abs_max: float
    layer_count: int
    vocab_size: int
    client_side_lm_head: bool
    client_side_argmax: bool
    encrypted_argmax: bool
    passed: bool

    def to_json_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["prompt_token_ids"] = list(self.prompt_token_ids)
        payload["new_token_ids"] = list(self.new_token_ids)
        return payload


@dataclass(frozen=True)
class ClientDecodeReport:
    """Aggregate report for PBI-S2-010."""

    stage: str
    measurement_scope: dict[str, Any]
    passed: bool
    row_count: int
    prompt_count: int
    total_client_decrypt_count: int
    min_top1_top2_gap: float | None
    max_logits_abs_max: float
    max_output_payload_width: int | None
    encrypted_argmax_claimed: bool
    rows: tuple[ClientDecodeReportRow, ...]

    def to_json_dict(self) -> dict[str, Any]:
        return {
            "stage": self.stage,
            "measurement_scope": dict(self.measurement_scope),
            "passed": self.passed,
            "row_count": self.row_count,
            "prompt_count": self.prompt_count,
            "total_client_decrypt_count": self.total_client_decrypt_count,
            "min_top1_top2_gap": self.min_top1_top2_gap,
            "max_logits_abs_max": self.max_logits_abs_max,
            "max_output_payload_width": self.max_output_payload_width,
            "encrypted_argmax_claimed": self.encrypted_argmax_claimed,
            "rows": [row.to_json_dict() for row in self.rows],
        }


def build_client_decode_report(
    artifacts: tuple[tuple[str, dict[str, Any]], ...],
) -> ClientDecodeReport:
    """Build a compact matrix over client-side decode smoke artifacts."""

    rows = tuple(_row_from_payload(source, payload) for source, payload in artifacts)
    if not rows:
        msg = "at least one client decode artifact is required"
        raise ValueError(msg)
    gaps = tuple(row.top1_top2_gap for row in rows if row.top1_top2_gap is not None)
    widths = tuple(row.output_payload_width for row in rows if row.output_payload_width is not None)
    encrypted_argmax_claimed = any(row.encrypted_argmax for row in rows)
    return ClientDecodeReport(
        stage="stage2-client-decode-report",
        measurement_scope={
            "report_only": True,
            "source_style_layers": True,
            "client_side_lm_head": all(row.client_side_lm_head for row in rows),
            "client_side_argmax": all(row.client_side_argmax for row in rows),
            "encrypted_argmax": encrypted_argmax_claimed,
            "full_model_correctness_claimed": False,
            "claim": (
                "Report over real-checkpoint client-side decode smoke artifacts. "
                "It records prompt/token outputs, top1-top2 logit gaps, payload width, "
                "and client decryption accounting; it is not encrypted argmax evidence."
            ),
        },
        passed=all(row.passed for row in rows) and not encrypted_argmax_claimed,
        row_count=len(rows),
        prompt_count=len({row.prompt_token_ids for row in rows}),
        total_client_decrypt_count=sum(row.client_decrypt_count for row in rows),
        min_top1_top2_gap=min(gaps) if gaps else None,
        max_logits_abs_max=max(row.logits_abs_max for row in rows),
        max_output_payload_width=max(widths) if widths else None,
        encrypted_argmax_claimed=encrypted_argmax_claimed,
        rows=rows,
    )


def client_decode_report_markdown(report: ClientDecodeReport) -> str:
    """Render a compact Markdown client decode report."""

    lines = [
        "# Stage 2 Client Decode Report",
        "",
        f"- Rows: `{report.row_count}`",
        f"- Total client decryptions: `{report.total_client_decrypt_count}`",
        f"- Min top1/top2 gap: `{_md_float(report.min_top1_top2_gap)}`",
        f"- Max payload width: `{report.max_output_payload_width or ''}`",
        "",
        "| source | prompt | selected | gap | decrypts | payload |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for row in report.rows:
        lines.append(
            "| "
            f"{row.source} | "
            f"{list(row.prompt_token_ids)} | "
            f"{row.selected_token if row.selected_token is not None else ''} | "
            f"{_md_float(row.top1_top2_gap)} | "
            f"{row.client_decrypt_count} | "
            f"{row.output_payload_width if row.output_payload_width is not None else ''} |"
        )
    lines.extend(["", "Scope: client-side lm_head/argmax baseline; no encrypted argmax claim."])
    return "\n".join(lines)


def _row_from_payload(source: str, payload: dict[str, Any]) -> ClientDecodeReportRow:
    if payload.get("stage") != "mamba-checkpoint-client-decode-smoke":
        msg = f"{source} is not a client decode smoke artifact"
        raise ValueError(msg)
    result = _required_dict(payload, "result")
    step = _first_decode_step(result)
    return ClientDecodeReportRow(
        source=source,
        prompt_token_ids=tuple(_int_items(result.get("prompt_token_ids", []))),
        new_token_ids=tuple(_int_items(result.get("new_token_ids", []))),
        selected_token=_int_or_none(step.get("selected_token")),
        top1_top2_gap=_float_or_none(step.get("top1_top2_gap")),
        top1_score=_float_or_none(step.get("top1_score")),
        top2_score=_float_or_none(step.get("top2_score")),
        output_payload_width=_int_or_none(step.get("output_payload_width")),
        client_decrypt_count=_int_or_zero(step.get("client_decrypt_count")),
        elapsed_sec=_required_float(result, "elapsed_sec"),
        hidden_abs_max=_required_float(result, "hidden_abs_max"),
        logits_abs_max=_required_float(result, "logits_abs_max"),
        layer_count=_required_int(result, "layer_count"),
        vocab_size=_required_int(result, "vocab_size"),
        client_side_lm_head=bool(result.get("client_side_lm_head")),
        client_side_argmax=bool(result.get("client_side_argmax")),
        encrypted_argmax=bool(result.get("encrypted_argmax")),
        passed=bool(payload.get("passed")) and bool(result.get("passed")),
    )


def _first_decode_step(result: dict[str, Any]) -> dict[str, Any]:
    steps = result.get("decode_steps")
    if isinstance(steps, list) and steps and isinstance(steps[0], dict):
        return steps[0]
    msg = "result.decode_steps must contain at least one object"
    raise ValueError(msg)


def _required_dict(payload: dict[str, Any], key: str) -> dict[str, Any]:
    value = payload.get(key)
    if isinstance(value, dict):
        return value
    msg = f"{key} must be a JSON object"
    raise ValueError(msg)


def _required_int(payload: dict[str, Any], key: str) -> int:
    value = payload.get(key)
    if isinstance(value, int) and not isinstance(value, bool):
        return value
    msg = f"{key} must be an integer"
    raise ValueError(msg)


def _required_float(payload: dict[str, Any], key: str) -> float:
    value = _float_or_none(payload.get(key))
    if value is not None:
        return value
    msg = f"{key} must be numeric"
    raise ValueError(msg)


def _int_items(value: Any) -> tuple[int, ...]:
    if not isinstance(value, list):
        return ()
    return tuple(item for item in value if isinstance(item, int) and not isinstance(item, bool))


def _int_or_none(value: Any) -> int | None:
    return value if isinstance(value, int) and not isinstance(value, bool) else None


def _int_or_zero(value: Any) -> int:
    parsed = _int_or_none(value)
    return 0 if parsed is None else parsed


def _float_or_none(value: Any) -> float | None:
    return float(value) if isinstance(value, int | float) and not isinstance(value, bool) else None


def _md_float(value: float | None) -> str:
    if value is None:
        return ""
    return f"{value:.6g}"


__all__ = [
    "ClientDecodeReport",
    "ClientDecodeReportRow",
    "build_client_decode_report",
    "client_decode_report_markdown",
]
