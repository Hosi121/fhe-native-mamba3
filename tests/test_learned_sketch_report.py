from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from fhe_native_mamba3 import __version__
from fhe_native_mamba3.learned_sketch_report import build_learned_sketch_report

ROOT = Path(__file__).resolve().parents[1]


def test_learned_sketch_report_summarizes_matrix_payload() -> None:
    report = build_learned_sketch_report(_matrix_payload(), source="matrix.json")

    assert report.stage == "stage2-learned-sketch-report"
    assert report.passed is True
    assert report.row_count == 2
    assert report.learned_recommended_sketch_size_counts == {4: 2}
    assert report.srht_recommended_sketch_size_counts == {8: 1, 16: 1}
    assert report.worst_learned_recommended_pairnorm_l2_error == 0.03
    assert report.min_srht_recommended_pass_rate == 0.8
    assert report.measurement_scope["plaintext_offline_training"] is True


def test_learned_sketch_report_script_runs(tmp_path) -> None:
    matrix_json = tmp_path / "matrix.json"
    output_json = tmp_path / "report.json"
    matrix_json.write_text(json.dumps(_matrix_payload()), encoding="utf-8")

    completed = subprocess.run(
        [
            sys.executable,
            "scripts/build_stage2_learned_sketch_report.py",
            "--matrix-json",
            str(matrix_json),
            "--output-json",
            str(output_json),
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )

    payload = json.loads(completed.stdout)
    persisted = json.loads(output_json.read_text(encoding="utf-8"))

    assert payload["version"] == __version__
    assert payload["stage"] == "stage2-learned-sketch-report"
    assert payload["passed"] is True
    assert payload["measurements"]["row_count"] == 2
    assert persisted["rows"] == payload["rows"]


def _matrix_payload() -> dict[str, object]:
    return {
        "stage": "mamba-checkpoint-learned-sketch-matrix",
        "passed": True,
        "rows": [
            _matrix_row(layer_index=0, learned_error=0.02, srht_size=16, srht_error=1e-8),
            _matrix_row(layer_index=1, learned_error=0.03, srht_size=8, srht_error=0.2),
        ],
    }


def _matrix_row(
    *,
    layer_index: int,
    learned_error: float,
    srht_size: int,
    srht_error: float,
) -> dict[str, object]:
    return {
        "layer_index": layer_index,
        "prompt_name": "short",
        "rank_strategy": "first:8",
        "learned_baseline": {
            "recommended_sketch_size": 4,
            "learned_rows": [
                {"sketch_size": 4, "readout_pairnorm_l2_error": learned_error},
                {"sketch_size": 8, "readout_pairnorm_l2_error": 1e-4},
            ],
            "srht_seed_sweep": {
                "recommended_sketch_size": srht_size,
                "rows": [
                    {
                        "sketch_size": 8,
                        "pass_rate": 0.8,
                        "max_pairnorm_l2_error": 0.2,
                    },
                    {
                        "sketch_size": 16,
                        "pass_rate": 1.0,
                        "max_pairnorm_l2_error": srht_error,
                    },
                ],
            },
        },
    }
