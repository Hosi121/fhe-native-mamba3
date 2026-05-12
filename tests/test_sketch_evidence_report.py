from __future__ import annotations

import pytest

import fhe_native_mamba3 as fhm3
from fhe_native_mamba3.artifact_validation import validate_benchmark_artifact
from fhe_native_mamba3.sketch_evidence_report import (
    build_sketch_evidence_report,
    sketch_evidence_report_markdown,
)


def test_sketch_evidence_report_summarizes_matrix_artifact() -> None:
    report = build_sketch_evidence_report(_matrix_payload(), source="runs/matrix.json")
    payload = {"version": "0.0.0", "repo_commit": "abc", **report.to_json_dict()}

    assert report.stage == "stage2-checkpoint-sketch-evidence-report"
    assert report.passed is True
    assert report.row_count == 2
    assert report.recommended_sketch_size_counts == {8: 1, 16: 1}
    assert report.max_recommended_sketch_size == 16
    assert report.worst_product_norm_error == 0.02
    assert report.min_recommended_pass_rate == 1.0
    assert report.measurement_scope["encrypted"] is False
    assert "Non-scalar" in report.recurrence_type_caveats[0]
    assert validate_benchmark_artifact(payload).valid is True


def test_sketch_evidence_report_markdown_renders_summary() -> None:
    report = build_sketch_evidence_report(_matrix_payload(), source="runs/matrix.json")

    markdown = sketch_evidence_report_markdown(report)

    assert "# Stage 2 Sketch Evidence Report" in markdown
    assert "| 0 | short | first:2 | rank-state | 16 | 1.000 | 0.01 |" in markdown
    assert "no encrypted or perplexity claim" in markdown


def test_sketch_evidence_report_rejects_wrong_stage() -> None:
    with pytest.raises(ValueError, match="stage='mamba-checkpoint-sketch-matrix'"):
        build_sketch_evidence_report({"stage": "other", "rows": []}, source="runs/nope.json")


def test_sketch_evidence_report_is_public_api() -> None:
    report = fhm3.build_sketch_evidence_report(_matrix_payload(), source="runs/matrix.json")

    assert isinstance(report, fhm3.SketchEvidenceReport)
    assert fhm3.sketch_evidence_report_markdown(report).startswith(
        "# Stage 2 Sketch Evidence Report"
    )


def _matrix_payload() -> dict[str, object]:
    return {
        "stage": "mamba-checkpoint-sketch-matrix",
        "passed": True,
        "rows": [
            _matrix_row(
                layer_index=0,
                prompt_name="short",
                rank_strategy="first:2",
                decay_kind="rank-state",
                rank_indices=[0, 1],
                recommended_sketch_size=16,
                recommended_error=0.01,
            ),
            _matrix_row(
                layer_index=1,
                prompt_name="repeat",
                rank_strategy="stride:2:4",
                decay_kind="rank-state",
                rank_indices=[0, 4],
                recommended_sketch_size=8,
                recommended_error=0.02,
            ),
        ],
    }


def _matrix_row(
    *,
    layer_index: int,
    prompt_name: str,
    rank_strategy: str,
    decay_kind: str,
    rank_indices: list[int],
    recommended_sketch_size: int,
    recommended_error: float,
) -> dict[str, object]:
    return {
        "layer_index": layer_index,
        "prompt_name": prompt_name,
        "rank_strategy": rank_strategy,
        "decay_kind": decay_kind,
        "rank_indices": rank_indices,
        "recommended_sketch_size": recommended_sketch_size,
        "seed_sweep": {
            "recommended_sketch_size": recommended_sketch_size,
            "rows": [
                _seed_row(4, pass_rate=0.0, all_passed=False, error=0.5),
                _seed_row(8, pass_rate=1.0, all_passed=True, error=0.02),
                _seed_row(16, pass_rate=1.0, all_passed=True, error=recommended_error),
            ],
        },
    }


def _seed_row(
    sketch_size: int,
    *,
    pass_rate: float,
    all_passed: bool,
    error: float,
) -> dict[str, object]:
    return {
        "sketch_size": sketch_size,
        "pass_rate": pass_rate,
        "all_passed": all_passed,
        "max_pairnorm_l2_error": error,
        "max_relative_l2_error": error * 2,
        "max_pairnorm_p95_abs_error": error * 1.5,
        "recurrence_compat_available": False,
        "max_recurrence_compat_abs_error": 0.0,
    }
