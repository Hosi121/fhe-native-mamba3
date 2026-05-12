from __future__ import annotations

import json
import subprocess
import sys

import pytest

from fhe_native_mamba3.artifact_validation import validate_benchmark_artifact
from fhe_native_mamba3.backends.tracking import TrackingBackend
from fhe_native_mamba3.ciphertext_handoff import (
    CiphertextHandoffLayer,
    apply_handoff_bootstrap_schedule,
    matrix_to_cyclic_diagonals,
    plaintext_handoff_chain,
    required_handoff_rotations,
    run_ciphertext_handoff_chain,
)


def test_ciphertext_handoff_chain_decrypts_only_after_final_layer() -> None:
    first = CiphertextHandoffLayer(
        diagonals=matrix_to_cyclic_diagonals(
            (
                (0.10, 0.01, -0.02, 0.00),
                (0.03, 0.20, 0.04, -0.01),
                (-0.01, 0.02, 0.15, 0.03),
                (0.02, 0.00, -0.02, 0.12),
            )
        ),
        residual_scale=0.95,
    )
    second = CiphertextHandoffLayer(
        diagonals=matrix_to_cyclic_diagonals(
            (
                (0.05, -0.04, 0.02, 0.01),
                (0.01, 0.07, -0.03, 0.02),
                (0.02, 0.03, 0.06, -0.01),
                (-0.03, 0.01, 0.02, 0.04),
            )
        ),
        residual_scale=1.05,
        bootstrap_after=True,
    )

    result = run_ciphertext_handoff_chain(
        backend=TrackingBackend(batch_size=4),
        input_values=(0.5, -0.25, 0.125, -0.375),
        layers=(first, second),
    )

    assert result.max_abs_error < 1e-12
    assert result.bootstrap_after_layers == (2,)
    assert result.decrypted_output == plaintext_handoff_chain(
        input_values=(0.5, -0.25, 0.125, -0.375),
        layers=(first, second),
    )
    assert result.backend_stats["encrypt_count"] == 1
    assert result.backend_stats["decrypt_count"] == 1
    assert result.backend_stats["bootstrap_count"] == 1
    assert result.backend_stats["rotation_count"] == 6


def test_handoff_bootstrap_schedule_runs_24_layer_boundary_smoke() -> None:
    matrix = (
        (0.020, -0.005, 0.001, 0.000),
        (0.003, 0.015, -0.002, 0.001),
        (0.000, 0.004, 0.018, -0.003),
        (-0.002, 0.000, 0.005, 0.017),
    )
    layers = tuple(
        CiphertextHandoffLayer(
            diagonals=matrix_to_cyclic_diagonals(matrix),
            residual_scale=0.98 + 0.001 * (layer_index % 3),
        )
        for layer_index in range(24)
    )

    scheduled = apply_handoff_bootstrap_schedule(
        layers,
        bootstrap_before_layers=(4, 8, 12, 16, 20),
    )
    result = run_ciphertext_handoff_chain(
        backend=TrackingBackend(batch_size=4),
        input_values=(0.5, -0.25, 0.125, -0.375),
        layers=scheduled,
    )

    assert result.layer_count == 24
    assert result.bootstrap_after_layers == (4, 8, 12, 16, 20)
    assert result.backend_stats["encrypt_count"] == 1
    assert result.backend_stats["decrypt_count"] == 1
    assert result.backend_stats["bootstrap_count"] == 5
    assert result.max_abs_error < 1e-12


def test_handoff_bootstrap_schedule_rejects_unrepresentable_first_boundary() -> None:
    layer = CiphertextHandoffLayer(diagonals=((1.0,),))

    with pytest.raises(ValueError, match="first layer boundary"):
        apply_handoff_bootstrap_schedule((layer,), bootstrap_before_layers=(0,))


def test_required_handoff_rotations_are_cyclic_diagonal_shifts() -> None:
    assert required_handoff_rotations(4) == (1, 2, 3)


def test_matrix_to_cyclic_diagonals_rejects_non_square_matrix() -> None:
    with pytest.raises(ValueError, match="square"):
        matrix_to_cyclic_diagonals(((1.0, 2.0),))


def test_handoff_rejects_layer_width_mismatch() -> None:
    layer = CiphertextHandoffLayer(diagonals=((1.0,),))

    with pytest.raises(ValueError, match="does not match"):
        run_ciphertext_handoff_chain(
            backend=TrackingBackend(batch_size=2),
            input_values=(1.0, 2.0),
            layers=(layer,),
        )


def test_smoke_script_rejects_openfhe_non_power_of_two_width_early() -> None:
    completed = subprocess.run(
        [
            sys.executable,
            "scripts/run_ciphertext_handoff_smoke.py",
            "--backend",
            "openfhe",
            "--width",
            "3",
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode != 0
    assert "requires --width to be a power of two" in completed.stderr


def test_smoke_script_accepts_bootstrap_before_layer_schedule() -> None:
    completed = subprocess.run(
        [
            sys.executable,
            "scripts/run_ciphertext_handoff_smoke.py",
            "--backend",
            "tracking",
            "--width",
            "4",
            "--layers",
            "24",
            "--bootstrap-before-layers",
            "4,8,12,16,20",
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    assert '"bootstrap_before_layers": [' in completed.stdout
    assert '"bootstrap_after_layers": [' in completed.stdout
    assert '"bootstrap_count": 5' in completed.stdout
    assert '"no_intermediate_decrypt": true' in completed.stdout
    payload = json.loads(completed.stdout)
    assert payload["measurement_scope"]["full_model_correctness_claimed"] is False
    assert payload["operation_counts"]["bootstraps"] == 5
    assert payload["passed"] is True
    assert validate_benchmark_artifact(payload).valid is True
