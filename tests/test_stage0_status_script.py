from __future__ import annotations

import json
import subprocess
import sys


def test_build_stage0_status_report_script_accepts_profile_and_decode_artifacts(tmp_path) -> None:
    profile_json = tmp_path / "profile.json"
    decode_json = tmp_path / "decode.json"
    output_json = tmp_path / "status.json"
    profile_json.write_text(
        json.dumps(
            {
                "passed": True,
                "measurement_scope": {
                    "encrypted": False,
                    "full_model_correctness_claimed": False,
                },
                "result": {
                    "token_ids": [1] * 32,
                    "layer_count": 24,
                    "d_model": 768,
                    "d_state": 16,
                    "mimo_rank": 1536,
                    "global_maxima": {
                        "range_score": 1000.0,
                        "high_decay_burst_len": 32,
                    },
                    "layers": [
                        {
                            "layer_index": 7,
                            "range_score": 1000.0,
                            "range_score_stage": "gate_post_silu",
                        }
                    ],
                },
            }
        ),
        encoding="utf-8",
    )
    decode_json.write_text(
        json.dumps(
            {
                "passed": True,
                "result": {
                    "new_token_ids": [42],
                    "decode_steps": [{"top1_top2_gap": 0.5}],
                    "client_side_lm_head": True,
                    "client_side_argmax": True,
                    "encrypted_argmax": False,
                },
            }
        ),
        encoding="utf-8",
    )

    completed = subprocess.run(
        [
            sys.executable,
            "scripts/build_stage0_status_report.py",
            "--checkpoint-source-profile-json",
            str(profile_json),
            "--client-decode-smoke-json",
            str(decode_json),
            "--output-json",
            str(output_json),
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    payload = json.loads(completed.stdout)
    assert payload["version"] == "0.2.84"
    assert payload["measurements"]["checkpoint_source_profile"]["range_score_layer"] == 7
    assert payload["measurements"]["client_decode_smoke"]["new_token_ids"] == [42]
    assert payload["bottlenecks"][0]["name"] == "range"
    assert json.loads(output_json.read_text(encoding="utf-8"))["next_bottleneck"]
