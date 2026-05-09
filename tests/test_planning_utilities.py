from __future__ import annotations

from fhe_native_mamba3.backends.capabilities import backend_capability_matrix
from fhe_native_mamba3.decoding import client_side_argmax, get_decoding_policy
from fhe_native_mamba3.rotation_inventory import build_rotation_inventory
from fhe_native_mamba3.weight_encoding import (
    WeightEncodingConfig,
    apply_weight_rescale,
    calibrate_weight_values,
)


def test_backend_capability_matrix_marks_fideslib_as_gpu_bootstrap_candidate() -> None:
    matrix = {entry["name"]: entry for entry in backend_capability_matrix()}
    assert matrix["fideslib-gpu"]["gpu"] is True
    assert matrix["fideslib-gpu"]["bootstrap"] is True
    assert matrix["fideslib-gpu"]["status"] == "verified-on-b200-stage0-native"
    assert matrix["phantom-fhe"]["bootstrap"] is False


def test_rotation_inventory_estimates_unique_keys_and_memory() -> None:
    inventory = build_rotation_inventory(
        scan_len=8,
        d_state=4,
        d_model=8,
        head_pack_sizes=(4, 8),
        key_size_mb=64.0,
    )
    assert 1 in inventory.unique_steps
    assert 4 in inventory.unique_steps
    assert inventory.unique_key_count > 0
    assert inventory.estimated_key_memory_gib == inventory.unique_key_count * 64.0 / 1024.0


def test_decoding_policy_defaults_to_client_side_generation() -> None:
    policy = get_decoding_policy("client-side")
    assert policy.interactive is True
    assert policy.encrypted_argmax is False
    assert client_side_argmax([0.1, 0.9, -1.0]) == 1


def test_weight_calibration_rescales_large_fp32_weights() -> None:
    calibration = calibrate_weight_values(
        [0.25, -2.0, 0.5],
        WeightEncodingConfig(scale_bits=40, target_max_abs=1.0),
    )
    assert calibration.source_dtype == "fp32"
    assert calibration.max_abs == 2.0
    assert calibration.rescale_factor == 0.5
    assert apply_weight_rescale([0.25, -2.0], calibration) == (0.125, -1.0)
