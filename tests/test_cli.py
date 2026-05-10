from __future__ import annotations

import json
import subprocess
import sys
from dataclasses import asdict

import torch

from fhe_native_mamba3.model import FheMamba3Config, FheMamba3ForCausalLM


def test_inspect_cli_outputs_json() -> None:
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "fhe_native_mamba3.cli",
            "inspect",
            "--d-model",
            "16",
            "--d-state",
            "4",
            "--mimo-rank",
            "2",
            "--seq-len",
            "8",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    payload = json.loads(completed.stdout)
    assert payload["version"] == "0.2.50"
    assert payload["cost_per_block"]["seq_len"] == 8


def test_cost_model_cli_outputs_ckks_payload() -> None:
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "fhe_native_mamba3.cli",
            "cost-model",
            "--d-model",
            "16",
            "--d-state",
            "4",
            "--mimo-rank",
            "2",
            "--n-layers",
            "2",
            "--seq-len",
            "8",
            "--effective-window",
            "4",
            "--scan-mode",
            "ssd",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    payload = json.loads(completed.stdout)
    assert payload["version"] == "0.2.50"
    assert payload["integrated_cost"]["effective_window"] == 4
    assert payload["integrated_cost"]["head_packing"]["heads_per_ciphertext"] >= 1
    assert payload["integrated_cost"]["block_cost"]["rotations"] == 2


def test_openfhe_recurrence_cli_encrypts_inputs() -> None:
    try:
        __import__("openfhe")
    except ImportError:
        return

    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "fhe_native_mamba3.cli",
            "openfhe-recurrence",
            "--seq-len",
            "2",
            "--d-state",
            "2",
            "--mimo-rank",
            "2",
            "--seed",
            "11",
            "--multiplicative-depth",
            "8",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    payload = json.loads(completed.stdout)
    assert payload["backend"] == "openfhe-ckks"
    assert payload["max_abs_error"] < 1e-6


def test_stage0_tracking_cli_outputs_benchmark_json() -> None:
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "fhe_native_mamba3.cli",
            "stage0-mimo",
            "--backend",
            "tracking",
            "--seq-len",
            "3",
            "--d-state",
            "2",
            "--mimo-rank",
            "2",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    payload = json.loads(completed.stdout)
    assert payload["version"] == "0.2.50"
    assert payload["stage"] == "0"
    assert payload["backend"] == "tracking"
    assert payload["encrypted"] is False
    assert payload["model"]["input_mode"] == "client-update"
    assert payload["max_abs_error"] < 1e-12
    assert payload["operation_counts"]["client_plaintext_public_weight_multiplies"] == 12
    assert payload["operation_counts"]["rotations"] == 9


def test_stage0_sweep_cli_outputs_summary() -> None:
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "fhe_native_mamba3.cli",
            "stage0-sweep",
            "--backend",
            "tracking",
            "--seq-lens",
            "2",
            "--d-states",
            "2,4",
            "--mimo-ranks",
            "2",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    payload = json.loads(completed.stdout)
    assert payload["version"] == "0.2.50"
    assert payload["result_count"] == 4
    assert payload["summary"]["max_abs_error_max"] < 1e-12


def test_stage0_rank_local_cli_outputs_benchmark_json() -> None:
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "fhe_native_mamba3.cli",
            "stage0-mimo",
            "--backend",
            "tracking",
            "--seq-len",
            "2",
            "--d-state",
            "4",
            "--mimo-rank",
            "4",
            "--readout-strategy",
            "rank-local",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    payload = json.loads(completed.stdout)
    assert payload["model"]["readout_strategy"] == "rank-local"
    assert payload["ckks"]["rotations"] == [1, 2]
    assert payload["operation_counts"]["ct_pt_mul"] == 8
    assert payload["operation_counts"]["rotations"] == 4
    assert payload["max_abs_error"] < 1e-12


def test_profile_synthetic_cli_outputs_profile() -> None:
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "fhe_native_mamba3.cli",
            "profile-synthetic",
            "--batch-size",
            "2",
            "--seq-len",
            "8",
            "--d-model",
            "16",
            "--d-state",
            "3",
            "--mimo-rank",
            "2",
            "--n-layers",
            "1",
            "--beta-grid",
            "0.5,1.0",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    payload = json.loads(completed.stdout)
    assert payload["version"] == "0.2.50"
    assert payload["profile"]["seq_len"] == 8
    assert payload["profile"]["blocks"][0]["lambda_by_beta"]["0.5"] >= 0.0


def test_planning_cli_commands_output_json() -> None:
    commands = [
        ["backend-capabilities"],
        ["decoding-policy", "--mode", "client-side"],
        [
            "rotation-inventory",
            "--scan-len",
            "8",
            "--d-state",
            "4",
            "--d-model",
            "8",
        ],
        ["weight-calibrate", "--values", "0.25,-2.0,0.5"],
    ]
    for command in commands:
        completed = subprocess.run(
            [sys.executable, "-m", "fhe_native_mamba3.cli", *command],
            check=True,
            capture_output=True,
            text=True,
        )
        payload = json.loads(completed.stdout)
        assert payload["version"] == "0.2.50"


def test_weight_bundle_cli_exports_and_inspects_manifest(tmp_path) -> None:
    bundle_dir = tmp_path / "bundle"
    export_completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "fhe_native_mamba3.cli",
            "weight-bundle-export",
            "--output-dir",
            str(bundle_dir),
            "--vocab-size",
            "16",
            "--d-model",
            "8",
            "--n-layers",
            "1",
            "--d-state",
            "2",
            "--mimo-rank",
            "2",
            "--max-seq-len",
            "8",
            "--scan-mode",
            "ssd",
            "--effective-window",
            "8",
            "--seed",
            "13",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    export_payload = json.loads(export_completed.stdout)
    assert export_payload["version"] == "0.2.50"
    assert export_payload["summary"]["tensor_count"] > 0
    assert export_payload["summary"]["parameter_count"] > 0
    assert (bundle_dir / "manifest.json").exists()
    assert (bundle_dir / "weights.pt").exists()

    inspect_completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "fhe_native_mamba3.cli",
            "weight-bundle-inspect",
            str(bundle_dir),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    inspect_payload = json.loads(inspect_completed.stdout)
    assert inspect_payload["version"] == "0.2.50"
    assert inspect_payload["summary"] == export_payload["summary"]
    assert inspect_payload["weight_bundle"]["model_config"]["scan_mode"] == "ssd"


def test_weight_bundle_eval_cli_runs_loaded_bundle(tmp_path) -> None:
    bundle_dir = tmp_path / "bundle"
    subprocess.run(
        [
            sys.executable,
            "-m",
            "fhe_native_mamba3.cli",
            "weight-bundle-export",
            "--output-dir",
            str(bundle_dir),
            "--vocab-size",
            "16",
            "--d-model",
            "8",
            "--n-layers",
            "1",
            "--d-state",
            "2",
            "--mimo-rank",
            "2",
            "--max-seq-len",
            "8",
            "--seed",
            "21",
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "fhe_native_mamba3.cli",
            "weight-bundle-eval",
            str(bundle_dir),
            "--batch-size",
            "2",
            "--seq-len",
            "6",
            "--seed",
            "21",
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    payload = json.loads(completed.stdout)
    assert payload["version"] == "0.2.50"
    assert payload["bundle_dir"] == str(bundle_dir)
    assert payload["input_shape"] == [2, 6]
    assert payload["logits_shape"] == [2, 6, 16]
    assert len(payload["client_side_next_tokens"]) == 2
    assert payload["loss"] > 0


def test_weight_bundle_generate_cli_runs_client_side_argmax(tmp_path) -> None:
    bundle_dir = tmp_path / "bundle"
    subprocess.run(
        [
            sys.executable,
            "-m",
            "fhe_native_mamba3.cli",
            "weight-bundle-export",
            "--output-dir",
            str(bundle_dir),
            "--vocab-size",
            "16",
            "--d-model",
            "8",
            "--n-layers",
            "1",
            "--d-state",
            "2",
            "--mimo-rank",
            "2",
            "--max-seq-len",
            "8",
            "--seed",
            "31",
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "fhe_native_mamba3.cli",
            "weight-bundle-generate",
            str(bundle_dir),
            "--prompt",
            "1,2,3",
            "--steps",
            "3",
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    payload = json.loads(completed.stdout)
    assert payload["version"] == "0.2.50"
    assert payload["decoding_mode"] == "client-side-argmax"
    assert payload["prompt_token_ids"] == [1, 2, 3]
    assert len(payload["new_token_ids"]) == 3
    assert len(payload["generated_token_ids"]) == 6
    assert all(0 <= token < 16 for token in payload["generated_token_ids"])


def test_weight_bundle_recurrence_cli_runs_tracking_backend(tmp_path) -> None:
    bundle_dir = tmp_path / "bundle"
    output_json = tmp_path / "recurrence-full.json"
    subprocess.run(
        [
            sys.executable,
            "-m",
            "fhe_native_mamba3.cli",
            "weight-bundle-export",
            "--output-dir",
            str(bundle_dir),
            "--vocab-size",
            "16",
            "--d-model",
            "8",
            "--n-layers",
            "1",
            "--d-state",
            "2",
            "--mimo-rank",
            "2",
            "--max-seq-len",
            "8",
            "--seed",
            "41",
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "fhe_native_mamba3.cli",
            "weight-bundle-recurrence",
            str(bundle_dir),
            "--backend",
            "tracking",
            "--prompt",
            "1,2,3",
            "--readout-strategy",
            "rank-local",
            "--max-output-values",
            "2",
            "--output-json",
            str(output_json),
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    payload = json.loads(completed.stdout)
    assert payload["version"] == "0.2.50"
    assert payload["stage"] == "bundle-recurrence"
    assert payload["source"] == "weight-bundle"
    assert payload["backend"] == "tracking"
    assert payload["encrypted"] is False
    assert payload["model"]["seq_len"] == 3
    assert payload["model"]["state_slots"] == 4
    assert payload["max_abs_error"] < 1e-12
    assert payload["operation_counts"]["encrypt"] > 0
    assert "decrypted_outputs" not in payload
    assert payload["output_summary"]["decrypted_outputs"]["included_value_count"] == 2
    assert payload["output_summary"]["decrypted_outputs"]["truncated"] is True
    assert payload["extracted_problem_summary"]["problem"]["state_slots"] == 4
    full_payload = json.loads(output_json.read_text(encoding="utf-8"))
    assert "decrypted_outputs" in full_payload
    assert full_payload["decrypted_outputs"] == full_payload["expected_outputs"]


def test_mamba_checkpoint_to_bundle_cli_adapts_common_checkpoint(tmp_path) -> None:
    checkpoint_path = tmp_path / "mamba.pt"
    bundle_dir = tmp_path / "mamba-bundle"
    torch.save({"model": _fake_mamba_state_dict()}, checkpoint_path)

    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "fhe_native_mamba3.cli",
            "mamba-checkpoint-to-bundle",
            str(checkpoint_path),
            "--output-dir",
            str(bundle_dir),
            "--d-state",
            "2",
            "--mimo-rank",
            "3",
            "--n-layers",
            "1",
            "--max-seq-len",
            "8",
            "--max-statuses",
            "4",
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    payload = json.loads(completed.stdout)
    assert payload["version"] == "0.2.50"
    assert payload["state_dict_key"] == "model"
    assert payload["adapter_shape"] == {"source": "cli", "d_state": 2, "mimo_rank": 3}
    assert payload["summary"]["tensor_count"] > 0
    assert payload["mamba_checkpoint_plan"]["complete_layer_count"] == 1
    assert payload["mamba_checkpoint_plan"]["inferred_d_state"] == 3
    assert payload["adapter_report"]["adapted_layers"] == 1
    assert payload["adapter_report"]["adapted_count"] >= 4
    assert (bundle_dir / "manifest.json").exists()


def test_mamba_checkpoint_to_bundle_cli_can_infer_adapter_shape(tmp_path) -> None:
    checkpoint_path = tmp_path / "mamba.pt"
    bundle_dir = tmp_path / "mamba-bundle"
    torch.save({"model": _fake_mamba_state_dict()}, checkpoint_path)

    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "fhe_native_mamba3.cli",
            "mamba-checkpoint-to-bundle",
            str(checkpoint_path),
            "--output-dir",
            str(bundle_dir),
            "--infer-shape",
            "--n-layers",
            "1",
            "--max-seq-len",
            "8",
            "--max-statuses",
            "1",
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    payload = json.loads(completed.stdout)
    assert payload["version"] == "0.2.50"
    assert payload["adapter_shape"] == {"source": "checkpoint", "d_state": 3, "mimo_rank": 6}
    assert payload["weight_bundle"]["model_config"]["d_state"] == 3
    assert payload["weight_bundle"]["model_config"]["mimo_rank"] == 6
    assert payload["mamba_checkpoint_plan"]["inferred_d_state"] == 3
    assert payload["mamba_checkpoint_plan"]["inferred_mimo_rank"] == 6


def test_mamba_checkpoint_plan_cli_reports_detected_layout(tmp_path) -> None:
    checkpoint_path = tmp_path / "mamba.pt"
    torch.save({"model": _fake_mamba_state_dict()}, checkpoint_path)

    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "fhe_native_mamba3.cli",
            "mamba-checkpoint-plan",
            str(checkpoint_path),
            "--max-layers",
            "1",
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    payload = json.loads(completed.stdout)
    plan = payload["mamba_checkpoint_plan"]
    assert payload["version"] == "0.2.50"
    assert payload["state_dict_key"] == "model"
    assert plan["embedding_key"] == "backbone.embedding.weight"
    assert plan["final_norm_key"] == "backbone.norm_f.weight"
    assert plan["vocab_size"] == 11
    assert plan["d_model"] == 8
    assert plan["inferred_layers"] == 1
    assert plan["complete_layer_count"] == 1
    assert plan["inferred_d_state"] == 3
    assert plan["inferred_mimo_rank"] == 6
    assert len(plan["layers"]) == 1
    assert plan["layers"][0]["inferred_dt_rank"] == 2


def test_mamba_checkpoint_recurrence_smoke_cli_runs_tracking_backend(tmp_path) -> None:
    checkpoint_path = tmp_path / "mamba.pt"
    bundle_dir = tmp_path / "mamba-smoke-bundle"
    output_json = tmp_path / "mamba-smoke-full.json"
    torch.save({"model": _fake_mamba_state_dict()}, checkpoint_path)

    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "fhe_native_mamba3.cli",
            "mamba-checkpoint-recurrence-smoke",
            str(checkpoint_path),
            "--output-dir",
            str(bundle_dir),
            "--backend",
            "tracking",
            "--d-state",
            "2",
            "--mimo-rank",
            "2",
            "--n-layers",
            "1",
            "--max-seq-len",
            "8",
            "--prompt",
            "1,2",
            "--max-statuses",
            "4",
            "--max-output-values",
            "3",
            "--output-json",
            str(output_json),
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    payload = json.loads(completed.stdout)
    assert payload["version"] == "0.2.50"
    assert payload["stage"] == "mamba-checkpoint-recurrence-smoke"
    assert payload["backend"] == "tracking"
    assert payload["encrypted"] is False
    assert payload["adapter_shape"] == {"source": "cli", "d_state": 2, "mimo_rank": 2}
    assert payload["mamba_checkpoint_plan"]["complete_layer_count"] == 1
    assert payload["mamba_checkpoint_plan"]["inferred_mimo_rank"] == 6
    assert payload["adapter_report"]["adapted_layers"] == 1
    assert payload["model"]["seq_len"] == 2
    assert payload["max_abs_error"] == 0
    assert "decrypted_outputs" not in payload
    assert payload["output_summary"]["expected_outputs"]["included_value_count"] == 3
    assert payload["extracted_problem_summary"]["problem"]["state_slots"] == 4
    assert json.loads(output_json.read_text(encoding="utf-8"))["decrypted_outputs"]
    assert (bundle_dir / "weights.pt").exists()


def test_mamba_checkpoint_recurrence_smoke_cli_runs_source_dynamic_bc(tmp_path) -> None:
    checkpoint_path = tmp_path / "mamba.pt"
    bundle_dir = tmp_path / "mamba-source-dynamic-bundle"
    torch.save({"model": _fake_mamba_state_dict()}, checkpoint_path)

    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "fhe_native_mamba3.cli",
            "mamba-checkpoint-recurrence-smoke",
            str(checkpoint_path),
            "--output-dir",
            str(bundle_dir),
            "--infer-shape",
            "--n-layers",
            "1",
            "--max-seq-len",
            "8",
            "--prompt",
            "1,2",
            "--backend",
            "tracking",
            "--readout-strategy",
            "rank-local",
            "--input-mode",
            "encrypted-dynamic-bc",
            "--recurrence-source",
            "source-dynamic",
            "--state-scale",
            "0.5",
            "--output-scale",
            "0.25",
            "--max-statuses",
            "2",
            "--max-plan-layers",
            "1",
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    payload = json.loads(completed.stdout)
    assert payload["version"] == "0.2.50"
    assert payload["model"]["recurrence_source"] == "source-dynamic"
    assert payload["model"]["input_mode"] == "encrypted-dynamic-bc"
    assert payload["model"]["input_propagation"] == "source"
    assert payload["model"]["state_scale"] == 0.5
    assert payload["model"]["output_scale"] == 0.25
    assert payload["model"]["c_scale_from_state"] == 0.5
    assert payload["operation_counts"]["ct_ct_mul"] == 3 * payload["model"]["seq_len"]
    assert payload["max_abs_error"] < 1e-12
    assert payload["scaled_problem_summary"]["plaintext_trace"]["state_abs_max"] > 0
    assert payload["extracted_problem_summary"]["problem"]["b_by_token_abs_max"] is not None
    assert payload["extracted_problem_summary"]["problem"]["c_by_token_abs_max"] is not None


def test_mamba_checkpoint_recurrence_sweep_cli_reports_rows(tmp_path) -> None:
    checkpoint_path = tmp_path / "mamba.pt"
    bundle_dir = tmp_path / "mamba-sweep-bundle"
    output_json = tmp_path / "sweep.json"
    torch.save({"model": _fake_mamba_state_dict()}, checkpoint_path)

    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "fhe_native_mamba3.cli",
            "mamba-checkpoint-recurrence-sweep",
            str(checkpoint_path),
            "--output-dir",
            str(bundle_dir),
            "--infer-shape",
            "--n-layers",
            "1",
            "--max-seq-len",
            "8",
            "--prompt",
            "1,2",
            "--seq-lens",
            "1,2",
            "--layer-indices",
            "0",
            "--all-layers",
            "--recurrence-sources",
            "adapter-static,source-dynamic",
            "--max-statuses",
            "2",
            "--max-plan-layers",
            "1",
            "--output-json",
            str(output_json),
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    payload = json.loads(completed.stdout)
    assert payload["version"] == "0.2.50"
    assert payload["stage"] == "mamba-checkpoint-recurrence-sweep"
    assert payload["summary"]["row_count"] == 4
    assert payload["summary"]["layer_count"] == 1
    assert payload["summary"]["seq_lens"] == [1, 2]
    assert payload["sweep_config"]["all_layers"] is True
    assert payload["summary"]["max_ct_ct_mul"] == 3 * 2
    assert payload["summary"]["by_layer"][0]["layer_index"] == 0
    assert payload["summary"]["by_layer"][0]["max_ct_ct_mul"] == 3 * 2
    assert payload["summary"]["top_range_cases"]
    assert {row["recurrence_source"] for row in payload["rows"]} == {
        "adapter-static",
        "source-dynamic",
    }
    dynamic_rows = [row for row in payload["rows"] if row["recurrence_source"] == "source-dynamic"]
    assert dynamic_rows[-1]["operation_counts"]["ct_ct_mul"] == 3 * 2
    assert dynamic_rows[-1]["problem"]["b_by_token_abs_max"] is not None
    assert json.loads(output_json.read_text(encoding="utf-8"))["summary"]["row_count"] == 4


def test_mamba_checkpoint_compare_reference_cli_reports_stage_errors(tmp_path) -> None:
    checkpoint_path = tmp_path / "mamba.pt"
    torch.save({"model": _fake_mamba_state_dict()}, checkpoint_path)

    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "fhe_native_mamba3.cli",
            "mamba-checkpoint-compare-reference",
            str(checkpoint_path),
            "--infer-shape",
            "--n-layers",
            "1",
            "--max-seq-len",
            "8",
            "--prompt",
            "1,2",
            "--max-statuses",
            "4",
            "--max-plan-layers",
            "1",
            "--include-source-delta",
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    payload = json.loads(completed.stdout)
    assert payload["version"] == "0.2.50"
    assert payload["stage"] == "mamba-checkpoint-compare-reference"
    assert payload["state_dict_key"] == "model"
    assert payload["adapter_shape"] == {"source": "checkpoint", "d_state": 3, "mimo_rank": 6}
    assert payload["prompt_token_ids"] == [1, 2]
    assert payload["comparison"]["scope"] == "adapter-compatible-reference"
    assert payload["comparison"]["official_mamba_parity"] is False
    assert payload["comparison"]["passed"] is True
    assert payload["comparison"]["max_exact_stage_error"] <= payload["comparison"]["atol"]
    assert payload["comparison"]["exact_stage_errors"]["recurrence_rank_output_max_abs_error"] == 0
    assert payload["model"]["include_source_delta"] is True
    assert payload["source_delta"]["scope"] == "source-style-delta"
    assert payload["source_delta"]["official_mamba_parity"] is False
    assert (
        payload["source_delta"]["reference_formula"] == "source-style-dynamic-bc-state-rank-decay"
    )


def test_mamba_checkpoint_source_diagnostics_cli_reports_ranges(tmp_path) -> None:
    checkpoint_path = tmp_path / "mamba.pt"
    output_json = tmp_path / "source-diagnostics.json"
    torch.save({"model": _fake_mamba_state_dict()}, checkpoint_path)

    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "fhe_native_mamba3.cli",
            "mamba-checkpoint-source-diagnostics",
            str(checkpoint_path),
            "--infer-shape",
            "--n-layers",
            "1",
            "--max-seq-len",
            "8",
            "--prompt",
            "1,2",
            "--seq-lens",
            "1,2",
            "--all-layers",
            "--input-propagation",
            "source",
            "--range-target",
            "0.1",
            "--range-warn",
            "1.0",
            "--range-fail",
            "100.0",
            "--max-statuses",
            "2",
            "--max-plan-layers",
            "1",
            "--output-json",
            str(output_json),
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    payload = json.loads(completed.stdout)
    assert payload["version"] == "0.2.50"
    assert payload["stage"] == "mamba-checkpoint-source-diagnostics"
    assert payload["summary"]["row_count"] == 2
    assert payload["summary"]["layer_count"] == 1
    assert payload["summary"]["seq_lens"] == [1, 2]
    assert payload["summary"]["max_range_score"] > 0
    assert sum(payload["summary"]["status_counts"].values()) == 2
    assert sum(payload["summary"]["group_status_counts"]["activation"].values()) == 2
    assert payload["summary"]["top_activation_cases"]
    assert payload["summary"]["top_recurrence_cases"]
    assert payload["summary"]["mitigation_plan"]["activation"]["top_targets"]
    assert payload["summary"]["mitigation_plan"]["recurrence"]["top_targets"]
    assert payload["diagnostics_config"]["input_propagation"] == "source"
    assert payload["summary"]["by_layer"][0]["layer_index"] == 0
    assert payload["summary"]["by_layer"][0]["range_status_at_max"] in {
        "ok",
        "target-exceeded",
        "warn",
        "fail",
    }
    assert payload["summary"]["top_range_cases"]
    assert payload["rows"][0]["ranges"]["causal_conv_post_silu"]["abs_max"] >= 0
    assert payload["rows"][0]["range_score_stage"] in payload["rows"][0]["ranges"]
    assert payload["rows"][0]["range_status"] in {"ok", "target-exceeded", "warn", "fail"}
    assert payload["rows"][0]["range_groups"]["activation"]["range_score_stage"] in {
        "rms_norm_output",
        "causal_conv_pre_silu",
        "gate_pre_silu",
    }
    assert json.loads(output_json.read_text(encoding="utf-8"))["summary"]["row_count"] == 2


def test_source_diagnostics_scale_plan_cli_reports_layer_scales(tmp_path) -> None:
    diagnostics_json = tmp_path / "diagnostics.json"
    output_json = tmp_path / "scale-plan.json"
    diagnostics_json.write_text(
        json.dumps(
            {
                "rows": [
                    {
                        "layer_index": 0,
                        "ranges": {
                            "layer_input": {"abs_max": 10.0},
                            "final_block_delta": {"abs_max": 200.0},
                            "final_block_output": {"abs_max": 150.0},
                            "recurrence_rank_output": {"abs_max": 80.0},
                        },
                        "range_groups": {
                            "activation": {"range_score": 12.0},
                            "recurrence": {"range_score": 200.0},
                            "residual": {"range_score": 150.0},
                        },
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "fhe_native_mamba3.cli",
            "source-diagnostics-scale-plan",
            str(diagnostics_json),
            "--activation-target",
            "6",
            "--state-target",
            "32",
            "--encoded-target",
            "32",
            "--output-json",
            str(output_json),
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    payload = json.loads(completed.stdout)
    assert payload["version"] == "0.2.50"
    assert payload["stage"] == "source-diagnostics-scale-plan"
    scale_plan = payload["scale_plan"]
    assert scale_plan["layer_count"] == 1
    assert scale_plan["activation_tuning_layer_count"] == 1
    assert scale_plan["state_scaled_layer_count"] == 1
    assert scale_plan["layers"][0]["output_scale"] == 32.0 / 200.0
    assert json.loads(output_json.read_text(encoding="utf-8"))["scale_plan"]["layer_count"] == 1


def test_weight_bundle_cli_converts_checkpoint(tmp_path) -> None:
    config = FheMamba3Config(vocab_size=16, d_model=8, n_layers=1, d_state=2, mimo_rank=2)
    model = FheMamba3ForCausalLM(config)
    checkpoint_path = tmp_path / "checkpoint.pt"
    bundle_dir = tmp_path / "bundle-from-checkpoint"
    torch.save(
        {
            "version": "test",
            "config": asdict(config),
            "model": model.state_dict(),
            "last_loss": 0.0,
        },
        checkpoint_path,
    )

    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "fhe_native_mamba3.cli",
            "weight-bundle-from-checkpoint",
            str(checkpoint_path),
            "--output-dir",
            str(bundle_dir),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    payload = json.loads(completed.stdout)
    assert payload["version"] == "0.2.50"
    assert payload["summary"]["tensor_count"] == len(model.state_dict())
    assert payload["weight_bundle"]["model_config"]["vocab_size"] == 16


def test_checkpoint_inspect_cli_outputs_tensor_shapes(tmp_path) -> None:
    checkpoint_path = tmp_path / "checkpoint.pt"
    torch.save({"model": {"b.weight": torch.zeros(2, 3), "a.bias": torch.ones(4)}}, checkpoint_path)

    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "fhe_native_mamba3.cli",
            "checkpoint-inspect",
            str(checkpoint_path),
            "--max-tensors",
            "1",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    payload = json.loads(completed.stdout)
    inspection = payload["checkpoint_inspection"]
    assert payload["version"] == "0.2.50"
    assert inspection["state_dict_key"] == "model"
    assert inspection["tensor_count"] == 2
    assert inspection["parameter_count"] == 10
    assert len(inspection["tensors"]) == 1
    assert inspection["tensors"][0]["name"] == "a.bias"


def test_checkpoint_map_report_cli_compares_against_target_model(tmp_path) -> None:
    config = FheMamba3Config(vocab_size=16, d_model=8, n_layers=1, d_state=2, mimo_rank=2)
    model = FheMamba3ForCausalLM(config)
    checkpoint_path = tmp_path / "checkpoint.pt"
    torch.save({"model": model.state_dict()}, checkpoint_path)

    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "fhe_native_mamba3.cli",
            "checkpoint-map-report",
            str(checkpoint_path),
            "--vocab-size",
            "16",
            "--d-model",
            "8",
            "--n-layers",
            "1",
            "--d-state",
            "2",
            "--mimo-rank",
            "2",
            "--max-seq-len",
            "256",
            "--max-statuses",
            "2",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    payload = json.loads(completed.stdout)
    report = payload["mapping_report"]
    assert payload["version"] == "0.2.50"
    assert report["is_complete"] is True
    assert report["mapped_count"] == len(model.state_dict())
    assert len(report["statuses"]) == 2


def test_checkpoint_map_template_cli_writes_reusable_rules_json(tmp_path) -> None:
    config = FheMamba3Config(vocab_size=17, d_model=8, n_layers=1, d_state=2, mimo_rank=2)
    model = FheMamba3ForCausalLM(config)
    source_state_dict = dict(model.state_dict())
    source_state_dict["external.in_rank.weight"] = source_state_dict.pop("blocks.0.in_rank.weight")
    checkpoint_path = tmp_path / "checkpoint.pt"
    rules_path = tmp_path / "draft-rules.json"
    bundle_dir = tmp_path / "mapped-bundle"
    torch.save({"model": source_state_dict}, checkpoint_path)

    template_completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "fhe_native_mamba3.cli",
            "checkpoint-map-template",
            str(checkpoint_path),
            "--output-json",
            str(rules_path),
            "--vocab-size",
            "17",
            "--d-model",
            "8",
            "--n-layers",
            "1",
            "--d-state",
            "2",
            "--mimo-rank",
            "2",
            "--max-seq-len",
            "256",
            "--max-entries",
            "2",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    template_payload = json.loads(template_completed.stdout)
    template = template_payload["mapping_template"]
    assert template_payload["version"] == "0.2.50"
    assert template["unique_shape_count"] == 1
    assert rules_path.exists()

    bundle_completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "fhe_native_mamba3.cli",
            "checkpoint-map-to-bundle",
            str(checkpoint_path),
            "--output-dir",
            str(bundle_dir),
            "--rules-json",
            str(rules_path),
            "--vocab-size",
            "17",
            "--d-model",
            "8",
            "--n-layers",
            "1",
            "--d-state",
            "2",
            "--mimo-rank",
            "2",
            "--max-seq-len",
            "256",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    bundle_payload = json.loads(bundle_completed.stdout)
    assert bundle_payload["mapping_report"]["is_complete"] is True
    assert bundle_payload["summary"]["tensor_count"] == len(model.state_dict())


def test_checkpoint_map_to_bundle_cli_exports_complete_mapping(tmp_path) -> None:
    config = FheMamba3Config(vocab_size=16, d_model=8, n_layers=1, d_state=2, mimo_rank=2)
    model = FheMamba3ForCausalLM(config)
    checkpoint_path = tmp_path / "checkpoint.pt"
    bundle_dir = tmp_path / "mapped-bundle"
    torch.save({"model": model.state_dict()}, checkpoint_path)

    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "fhe_native_mamba3.cli",
            "checkpoint-map-to-bundle",
            str(checkpoint_path),
            "--output-dir",
            str(bundle_dir),
            "--vocab-size",
            "16",
            "--d-model",
            "8",
            "--n-layers",
            "1",
            "--d-state",
            "2",
            "--mimo-rank",
            "2",
            "--max-seq-len",
            "256",
            "--max-statuses",
            "2",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    payload = json.loads(completed.stdout)
    assert payload["version"] == "0.2.50"
    assert payload["mapping_report"]["is_complete"] is True
    assert payload["summary"]["tensor_count"] == len(model.state_dict())
    assert (bundle_dir / "manifest.json").exists()


def _fake_mamba_state_dict() -> dict[str, torch.Tensor]:
    return {
        "backbone.embedding.weight": torch.arange(88, dtype=torch.float32).view(11, 8) / 100.0,
        "backbone.layers.0.norm.weight": torch.full((8,), 2.0),
        "backbone.layers.0.mixer.in_proj.weight": torch.arange(
            96,
            dtype=torch.float32,
        ).view(12, 8)
        / 100.0,
        "backbone.layers.0.mixer.x_proj.weight": torch.arange(
            48,
            dtype=torch.float32,
        ).view(8, 6)
        / 100.0,
        "backbone.layers.0.mixer.dt_proj.weight": torch.arange(
            12,
            dtype=torch.float32,
        ).view(6, 2)
        / 100.0,
        "backbone.layers.0.mixer.dt_proj.bias": torch.linspace(-0.2, 0.1, 6),
        "backbone.layers.0.mixer.out_proj.weight": torch.arange(
            48,
            dtype=torch.float32,
        ).view(8, 6)
        / 100.0,
        "backbone.layers.0.mixer.D": torch.linspace(0.1, 0.6, 6),
        "backbone.layers.0.mixer.conv1d.weight": torch.arange(
            18,
            dtype=torch.float32,
        ).view(6, 1, 3)
        / 50.0,
        "backbone.layers.0.mixer.conv1d.bias": torch.linspace(-0.1, 0.1, 6),
        "backbone.layers.0.mixer.A_log": torch.zeros(6, 3),
        "backbone.norm_f.weight": torch.ones(8),
        "lm_head.weight": torch.arange(88, dtype=torch.float32).view(11, 8) / 200.0,
    }
