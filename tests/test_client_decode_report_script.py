from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_build_client_decode_report_script(tmp_path) -> None:
    first_json = tmp_path / "first.json"
    second_json = tmp_path / "second.json"
    output_json = tmp_path / "report.json"
    output_markdown = tmp_path / "report.md"
    first_json.write_text(json.dumps(_decode_payload(token=10, gap=1.0)), encoding="utf-8")
    second_json.write_text(json.dumps(_decode_payload(token=11, gap=2.0)), encoding="utf-8")

    completed = subprocess.run(
        [
            sys.executable,
            "scripts/build_client_decode_report.py",
            "--client-decode-json",
            str(first_json),
            "--client-decode-json",
            str(second_json),
            "--output-json",
            str(output_json),
            "--output-markdown",
            str(output_markdown),
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )

    payload = json.loads(output_json.read_text(encoding="utf-8"))
    assert completed.stdout
    assert payload["stage"] == "stage2-client-decode-report"
    assert payload["row_count"] == 2
    assert payload["total_client_decrypt_count"] == 2
    assert payload["passed"] is True
    assert output_markdown.read_text(encoding="utf-8").startswith("# Stage 2 Client Decode Report")


def _decode_payload(*, token: int, gap: float) -> dict[str, object]:
    return {
        "stage": "mamba-checkpoint-client-decode-smoke",
        "passed": True,
        "result": {
            "prompt_token_ids": [1],
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
