from __future__ import annotations

import subprocess
import sys

import pytest

from fhe_native_mamba3.backends.tracking import TrackingBackend
from fhe_native_mamba3.ciphertext_handoff import (
    CiphertextHandoffLayer,
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

    assert result.max_abs_error == 0
    assert result.bootstrap_after_layers == (2,)
    assert result.decrypted_output == plaintext_handoff_chain(
        input_values=(0.5, -0.25, 0.125, -0.375),
        layers=(first, second),
    )
    assert result.backend_stats["encrypt_count"] == 1
    assert result.backend_stats["decrypt_count"] == 1
    assert result.backend_stats["bootstrap_count"] == 1
    assert result.backend_stats["rotation_count"] == 6


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
