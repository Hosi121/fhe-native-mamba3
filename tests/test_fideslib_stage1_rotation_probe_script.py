from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from fhe_native_mamba3.artifact_validation import validate_benchmark_artifact


def test_fideslib_stage1_rotation_probe_wrapper_adds_metadata(tmp_path: Path) -> None:
    fake_binary = _write_fake_binary(tmp_path, returncode=0)
    output_json = tmp_path / "wrapped.json"

    subprocess.run(
        [
            sys.executable,
            "scripts/run_fideslib_stage1_rotation_probe.py",
            "--binary",
            str(fake_binary),
            "--output-json",
            str(output_json),
            "--rotations-csv",
            "2,-1,2",
            "--ring-dimension",
            "131072",
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    payload = json.loads(output_json.read_text(encoding="utf-8"))
    assert payload["stage"] == "fideslib-gpu-stage1-state-major-rotation-probe"
    assert payload["version"]
    assert payload["repo_commit"]
    assert payload["native_returncode"] == 0
    assert "--ring-dim" in payload["native_command"]
    assert "131072" in payload["native_command"]
    assert payload["required_application_rotations"] == [-1, 2]
    assert payload["required_application_rotation_key_count"] == 2
    assert payload["measurement_scope"]["stage1_state_major_target_compatible"] is True
    assert validate_benchmark_artifact(payload, require_commit=True).valid is True


def test_fideslib_stage1_rotation_probe_wrapper_records_failure(tmp_path: Path) -> None:
    fake_binary = _write_fake_binary(tmp_path, returncode=2)
    output_json = tmp_path / "wrapped.json"

    completed = subprocess.run(
        [
            sys.executable,
            "scripts/run_fideslib_stage1_rotation_probe.py",
            "--binary",
            str(fake_binary),
            "--output-json",
            str(output_json),
            "--rotations-csv",
            "1,2",
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


def test_fideslib_stage1_rotation_probe_wrapper_accepts_rotation_artifact(
    tmp_path: Path,
) -> None:
    fake_binary = _write_fake_binary(tmp_path, returncode=0)
    rotation_artifact = tmp_path / "rotations.json"
    output_json = tmp_path / "wrapped.json"
    rotation_artifact.write_text(
        json.dumps({"required_application_rotations": [8, -4, 8]}),
        encoding="utf-8",
    )

    subprocess.run(
        [
            sys.executable,
            "scripts/run_fideslib_stage1_rotation_probe.py",
            "--binary",
            str(fake_binary),
            "--output-json",
            str(output_json),
            "--rotation-artifact",
            str(rotation_artifact),
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    payload = json.loads(output_json.read_text(encoding="utf-8"))
    assert payload["required_application_rotations"] == [-4, 8]
    assert payload["measurements"]["requested_rotation_key_count"] == 2


def _write_fake_binary(tmp_path: Path, *, returncode: int) -> Path:
    path = tmp_path / "fake_fides_rotation_probe.py"
    path.write_text(
        "\n".join(
            [
                "#!/usr/bin/env python3",
                "from __future__ import annotations",
                "import json",
                "import sys",
                f"if {returncode} != 0:",
                "    print('native failure', file=sys.stderr)",
                f"    raise SystemExit({returncode})",
                "output = sys.argv[sys.argv.index('--output-json') + 1]",
                "rotations = sys.argv[sys.argv.index('--rotations-csv') + 1]",
                "payload = {",
                "    'stage': 'fideslib-gpu-stage1-state-major-rotation-probe',",
                "    'backend': 'fideslib-gpu',",
                "    'available': True,",
                "    'encrypted': True,",
                "    'passed': True,",
                "    'config': {'input_mode': 'state-major-rotation-probe'},",
                "    'required_application_rotations': [int(x) for x in rotations.split(',')],",
                "    'executed_rotations': [int(x) for x in rotations.split(',')],",
                "    'latencies_sec': [0.5],",
                "    'mean_latency_sec': 0.5,",
                "    'measurements': {",
                "        'requested_rotation_key_count': len(rotations.split(',')),",
                "        'executed_rotation_count': len(rotations.split(',')),",
                "        'stage1_state_major_target_compatible': True,",
                "    },",
                "    'operation_counts': {",
                "        'bootstraps': 0, 'rotations': len(rotations.split(',')),",
                "        'ct_ct_mul': 0, 'ct_pt_mul': len(rotations.split(',')),",
                "        'encrypt': 1, 'decrypt': 0,",
                "    },",
                "    'measurement_scope': {",
                "        'stage1_fideslib_rotation_probe': True,",
                "        'state_major_layout': True,",
                "        'rank_pack_first': True,",
                "        'key_memory_probe': True,",
                "        'stage1_state_major_target_compatible': True,",
                "        'full_model_correctness_claimed': False,",
                "        'claim': 'fake successful native FIDESlib rotation probe',",
                "    },",
                "}",
                "open(output, 'w', encoding='utf-8').write(json.dumps(payload))",
            ]
        ),
        encoding="utf-8",
    )
    path.chmod(0o755)
    return path
