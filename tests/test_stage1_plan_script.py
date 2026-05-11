from __future__ import annotations

import json
import subprocess
import sys

from fhe_native_mamba3 import __version__


def test_build_stage1_plan_script_accepts_source_profile(tmp_path) -> None:
    profile_json = tmp_path / "profile.json"
    output_json = tmp_path / "stage1-plan.json"
    profile_json.write_text(
        json.dumps(
            {
                "result": {
                    "d_model": 8,
                    "d_state": 2,
                    "mimo_rank": 8,
                    "token_ids": [1, 2, 3, 4],
                    "layers": [
                        {
                            "recurrence": {
                                "head_count": 8,
                                "high_decay_bursts": [
                                    {
                                        "head": 7,
                                        "update_abs_max": 3.0,
                                        "decay_abs_max": 0.99,
                                    }
                                ],
                                "worst_cases": {
                                    "update_abs_max": {"head": 3, "value": 4.0},
                                    "decay_abs_max": {"head": 5, "value": 0.98},
                                },
                            }
                        }
                    ],
                }
            }
        ),
        encoding="utf-8",
    )

    completed = subprocess.run(
        [
            sys.executable,
            "scripts/build_stage1_plan.py",
            "--source-profile-json",
            str(profile_json),
            "--scan-len",
            "16",
            "--window",
            "8",
            "--slot-count",
            "128",
            "--candidate-pack-sizes",
            "2,4,8",
            "--matmul-diagonal-stride",
            "2",
            "--bootstrap-internal-key-count",
            "2",
            "--key-size-mb",
            "32",
            "--output-json",
            str(output_json),
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    payload = json.loads(completed.stdout)
    persisted = json.loads(output_json.read_text(encoding="utf-8"))

    assert payload["version"] == __version__
    assert payload["stage"] == "stage1-plan"
    assert payload["head_count"] == 8
    assert payload["d_state"] == 2
    assert payload["d_model"] == 8
    assert payload["window"] == 8
    assert payload["recommended_candidate"]["pack_size"] == 4
    assert payload["profile_hints"]["known_head_range_count"] == 2
    assert persisted["recommended_candidate"] == payload["recommended_candidate"]


def test_build_stage1_plan_script_accepts_explicit_shape_without_profile(tmp_path) -> None:
    output_json = tmp_path / "stage1-plan-explicit.json"

    completed = subprocess.run(
        [
            sys.executable,
            "scripts/build_stage1_plan.py",
            "--head-count",
            "32",
            "--d-state",
            "64",
            "--d-model",
            "768",
            "--scan-len",
            "256",
            "--window",
            "64",
            "--slot-count",
            "32768",
            "--candidate-pack-sizes",
            "4,8,16,32",
            "--output-json",
            str(output_json),
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    payload = json.loads(completed.stdout)

    assert payload["version"] == __version__
    assert payload["profile_hints"] is None
    assert payload["recommended_candidate"]["pack_size"] == 4
    assert payload["recommended_candidate"]["requires_cross_ciphertext_carry"] is True
    assert payload["recommended_candidate"]["estimated_total_scan_depth"] == 7
    assert any(candidate["requires_cross_ciphertext_carry"] for candidate in payload["candidates"])
