from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_stage2_range_lora_decision_script_emits_artifact(tmp_path) -> None:
    scale_plan = tmp_path / "scale-plan.json"
    learned = tmp_path / "learned.json"
    correctness = tmp_path / "correctness.json"
    output = tmp_path / "decision.json"
    scale_plan.write_text(
        json.dumps(
            {
                "scale_plan": {
                    "activation_tuning_layer_count": 2,
                    "state_scaled_layer_count": 3,
                    "output_scaled_layer_count": 3,
                    "max_encoded_input_abs": 32.0,
                    "max_encoded_delta_abs": 32.0,
                    "max_encoded_output_abs": 32.0,
                }
            }
        ),
        encoding="utf-8",
    )
    learned.write_text(
        json.dumps(
            {
                "measurements": {
                    "learned_recommended_sketch_size_counts": {"4": 12},
                    "worst_learned_recommended_pairnorm_l2_error": 0.03,
                }
            }
        ),
        encoding="utf-8",
    )
    correctness.write_text(json.dumps({"passed": True, "max_abs_error": 0.01}), encoding="utf-8")

    completed = subprocess.run(
        [
            sys.executable,
            "scripts/build_stage2_range_lora_decision.py",
            "--scale-plan-json",
            str(scale_plan),
            "--learned-sketch-report-json",
            str(learned),
            "--correctness-json",
            str(correctness),
            "--output-json",
            str(output),
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    payload = json.loads(output.read_text(encoding="utf-8"))

    assert completed.stdout
    assert payload["stage"] == "stage2-range-lora-decision"
    assert payload["passed"] is True
    assert payload["lora_recommended_now"] is False
    assert payload["recommended_action"] == "defer_lora_use_deterministic_calibration"
    assert payload["inputs"]["correctness_json"] == str(correctness)
    assert payload["measurement_scope"]["lora_training_executed"] is False
