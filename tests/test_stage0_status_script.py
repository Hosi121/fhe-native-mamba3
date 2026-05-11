from __future__ import annotations

import json
import subprocess
import sys

from fhe_native_mamba3 import __version__


def test_build_stage0_status_report_script_accepts_profile_and_decode_artifacts(tmp_path) -> None:
    profile_json = tmp_path / "profile.json"
    scale_plan_json = tmp_path / "scale-plan.json"
    full_layer_json = tmp_path / "full-layer.json"
    pre_sweep_json = tmp_path / "pre-sweep.json"
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
    full_layer_json.write_text(
        json.dumps(
            {
                "backend": "tracking",
                "encrypted": False,
                "passed": True,
                "max_abs_error": 0.0,
                "model": {
                    "seq_len": 2,
                    "d_model": 8,
                    "checked_visible_dim": 8,
                    "d_state": 2,
                    "mimo_rank": 4,
                },
                "measurement_scope": {
                    "source_style_full_layer_formula": True,
                    "full_visible_output_checked": True,
                    "partial_visible_output_checked": False,
                    "full_model_correctness_claimed": False,
                    "plaintext_precomputed_stages": ["rms_norm", "gate_values"],
                },
                "result": {
                    "recurrence_ciphertext": True,
                    "visible_handoff_ciphertext": True,
                    "no_intermediate_decrypt": True,
                },
                "ckks": {"rotation_count": 9},
                "operation_counts": {"decrypt": 2},
            }
        ),
        encoding="utf-8",
    )
    scale_plan_json.write_text(
        json.dumps(
            {
                "scale_plan": {
                    "activation_target": 6.0,
                    "state_target": 32.0,
                    "encoded_target": 32.0,
                    "layer_count": 1,
                    "activation_tuning_layer_count": 1,
                    "state_scaled_layer_count": 1,
                    "output_scaled_layer_count": 1,
                    "max_encoded_input_abs": 8.0,
                    "max_encoded_delta_abs": 32.0,
                    "max_encoded_output_abs": 32.0,
                    "layers": [{"output_scale": 0.5, "max_activation_abs": 12.0}],
                },
            }
        ),
        encoding="utf-8",
    )
    pre_sweep_json.write_text(
        json.dumps(
            {
                "stage": "tracking-24-layer-encrypted-pre-full-gate-summary",
                "layer_count": 2,
                "passed_count": 2,
                "failed_count": 0,
                "max_abs_error": 1e-3,
                "max_abs_error_layer": 1,
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
            "--range-scale-plan-json",
            str(scale_plan_json),
            "--checkpoint-full-layer-gate-json",
            str(full_layer_json),
            "--checkpoint-pre-recurrence-layer-sweep-json",
            str(pre_sweep_json),
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
    assert payload["version"] == __version__
    assert payload["repo_commit"]
    assert payload["measurements"]["checkpoint_source_profile"]["range_score_layer"] == 7
    assert payload["measurements"]["range_scale_plan"]["activation_tuning_layer_count"] == 1
    assert payload["measurements"]["checkpoint_full_layer_gate"]["rotation_count"] == 9
    assert payload["measurements"]["checkpoint_pre_recurrence_layer_sweep"]["passed_count"] == 2
    assert payload["measurements"]["client_decode_smoke"]["new_token_ids"] == [42]
    assert payload["bottlenecks"][0]["name"] == "range"
    assert json.loads(output_json.read_text(encoding="utf-8"))["next_bottleneck"]
