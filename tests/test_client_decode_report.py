from __future__ import annotations

import pytest

import fhe_native_mamba3 as fhm3
from fhe_native_mamba3.artifact_validation import validate_benchmark_artifact
from fhe_native_mamba3.client_decode_report import (
    build_client_decode_report,
    client_decode_report_markdown,
)


def test_client_decode_report_summarizes_decode_artifacts() -> None:
    report = build_client_decode_report(
        (
            ("runs/natural.json", _decode_payload(prompt=(1, 2), token=10, gap=1.25)),
            ("runs/repeat.json", _decode_payload(prompt=(7, 7), token=11, gap=4.5)),
        )
    )
    payload = {"version": "0.0.0", "repo_commit": "abc", **report.to_json_dict()}

    assert report.stage == "stage2-client-decode-report"
    assert report.passed is True
    assert report.row_count == 2
    assert report.total_client_decrypt_count == 2
    assert report.min_top1_top2_gap == 1.25
    assert report.max_output_payload_width == 32
    assert report.measurement_scope["client_side_argmax"] is True
    assert report.encrypted_argmax_claimed is False
    assert validate_benchmark_artifact(payload).valid is True


def test_client_decode_report_markdown_renders_table() -> None:
    report = build_client_decode_report(
        (("runs/natural.json", _decode_payload(prompt=(1,), token=10, gap=1.25)),)
    )

    markdown = client_decode_report_markdown(report)

    assert "# Stage 2 Client Decode Report" in markdown
    assert "| runs/natural.json | [1] | 10 | 1.25 | 1 | 32 |" in markdown
    assert "no encrypted argmax claim" in markdown


def test_client_decode_report_rejects_wrong_stage() -> None:
    with pytest.raises(ValueError, match="not a client decode"):
        build_client_decode_report((("runs/nope.json", {"stage": "other"}),))


def test_client_decode_report_is_public_api() -> None:
    report = fhm3.build_client_decode_report(
        (("runs/natural.json", _decode_payload(prompt=(1,), token=10, gap=1.25)),)
    )

    assert isinstance(report, fhm3.ClientDecodeReport)
    assert fhm3.client_decode_report_markdown(report).startswith("# Stage 2 Client Decode Report")


def _decode_payload(*, prompt: tuple[int, ...], token: int, gap: float) -> dict[str, object]:
    return {
        "stage": "mamba-checkpoint-client-decode-smoke",
        "passed": True,
        "measurement_scope": {
            "client_side_lm_head": True,
            "client_side_argmax": True,
            "encrypted_argmax": False,
        },
        "result": {
            "prompt_token_ids": list(prompt),
            "new_token_ids": [token],
            "decode_steps": [
                {
                    "selected_token": token,
                    "top1_top2_gap": gap,
                    "top1_score": 10.0,
                    "top2_score": 10.0 - gap,
                    "output_payload_width": 32,
                    "client_decrypt_count": 1,
                }
            ],
            "elapsed_sec": 0.25,
            "hidden_abs_max": 2.0,
            "logits_abs_max": 10.0,
            "layer_count": 2,
            "vocab_size": 32,
            "client_side_lm_head": True,
            "client_side_argmax": True,
            "encrypted_argmax": False,
            "passed": True,
        },
    }
