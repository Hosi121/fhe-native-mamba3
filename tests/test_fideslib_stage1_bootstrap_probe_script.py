from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from fhe_native_mamba3.artifact_validation import validate_benchmark_artifact


def test_fideslib_stage1_bootstrap_probe_wrapper_adds_metadata(tmp_path: Path) -> None:
    fake_binary = _write_fake_binary(tmp_path, returncode=0)
    output_json = tmp_path / "wrapped.json"

    subprocess.run(
        [
            sys.executable,
            "scripts/run_fideslib_stage1_bootstrap_probe.py",
            "--binary",
            str(fake_binary),
            "--output-json",
            str(output_json),
            "--ring-dim",
            "65536",
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    payload = json.loads(output_json.read_text(encoding="utf-8"))
    assert payload["stage"] == "fideslib-gpu-stage1-bootstrap-latency"
    assert payload["version"]
    assert payload["repo_commit"]
    assert payload["native_returncode"] == 0
    assert payload["measurement_scope"]["stage1_target_compatible"] is True
    assert validate_benchmark_artifact(payload, require_commit=True).valid is True


def test_fideslib_stage1_bootstrap_probe_wrapper_records_failure(tmp_path: Path) -> None:
    fake_binary = _write_fake_binary(tmp_path, returncode=2)
    output_json = tmp_path / "wrapped.json"

    completed = subprocess.run(
        [
            sys.executable,
            "scripts/run_fideslib_stage1_bootstrap_probe.py",
            "--binary",
            str(fake_binary),
            "--output-json",
            str(output_json),
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    payload = json.loads(output_json.read_text(encoding="utf-8"))
    assert completed.returncode == 1
    assert payload["passed"] is False
    assert payload["available"] is False
    assert payload["measurement_scope"]["non_success_probe"] is True
    assert validate_benchmark_artifact(payload, require_commit=True).valid is True


def _write_fake_binary(tmp_path: Path, *, returncode: int) -> Path:
    path = tmp_path / "fake_fides_probe.py"
    path.write_text(
        "\n".join(
            [
                "#!/usr/bin/env python3",
                "from __future__ import annotations",
                "import json",
                "import sys",
                "if " + str(returncode) + " != 0:",
                "    print('native failure', file=sys.stderr)",
                "    raise SystemExit(" + str(returncode) + ")",
                "output = sys.argv[sys.argv.index('--output-json') + 1]",
                "payload = {",
                "    'stage': 'fideslib-gpu-stage1-bootstrap-latency',",
                "    'backend': 'fideslib-gpu',",
                "    'available': True,",
                "    'encrypted': True,",
                "    'passed': True,",
                "    'config': {'input_mode': 'bootstrap-probe'},",
                "    'latencies_sec': [0.25],",
                "    'mean_latency_sec': 0.25,",
                "    'operation_counts': {",
                "        'bootstraps': 1, 'rotations': 0, 'ct_ct_mul': 0,",
                "        'ct_pt_mul': 0, 'encrypt': 1, 'decrypt': 1,",
                "    },",
                "    'measurement_scope': {",
                "        'bootstrap_latency_probe': True,",
                "        'gpu_bootstrap': True,",
                "        'stage1_target_compatible': True,",
                "        'full_model_correctness_claimed': False,",
                "        'claim': 'fake successful native FIDESlib bootstrap probe',",
                "    },",
                "}",
                "open(output, 'w', encoding='utf-8').write(json.dumps(payload))",
            ]
        ),
        encoding="utf-8",
    )
    path.chmod(0o755)
    return path
