from __future__ import annotations

import json
import subprocess

from fhe_native_mamba3.layout import (
    readout_output_slots,
    readout_reduce_mask,
    readout_reduce_steps,
    readout_scatter_mask,
    readout_scatter_shifts,
    required_readout_rotations,
    state_slots,
)


def test_native_stage0_layout_cpp_unit_tests(tmp_path) -> None:
    build_dir = tmp_path / "build"
    subprocess.run(
        [
            "cmake",
            "-S",
            "native/fideslib_stage0",
            "-B",
            str(build_dir),
            "-DFHE_STAGE0_BUILD_KERNEL=OFF",
            "-DFHE_STAGE0_BUILD_TESTS=ON",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    subprocess.run(
        ["cmake", "--build", str(build_dir), "-j", "2"],
        check=True,
        capture_output=True,
        text=True,
    )
    subprocess.run(
        ["ctest", "--test-dir", str(build_dir), "--output-on-failure"],
        check=True,
        capture_output=True,
        text=True,
    )
    dumped = subprocess.run(
        [str(build_dir / "test_stage0_layout"), "--dump-json"],
        check=True,
        capture_output=True,
        text=True,
    )
    payload = json.loads(dumped.stdout)
    assert payload["state_slots_4x4"] == state_slots(4, 4)
    assert payload["readout_rotations_dense_4x4"] == list(
        required_readout_rotations(d_state=4, mimo_rank=4, readout_strategy="rank-reduce")
    )
    assert payload["readout_rotations_rank_local_4x4"] == list(
        required_readout_rotations(d_state=4, mimo_rank=4, readout_strategy="rank-local")
    )
    assert payload["reduce_steps_5"] == list(readout_reduce_steps(5))
    assert payload["reduce_mask_4x2_step1"] == list(
        readout_reduce_mask(d_state=4, mimo_rank=2, step=1)
    )
    assert payload["reduce_mask_4x2_step2"] == list(
        readout_reduce_mask(d_state=4, mimo_rank=2, step=2)
    )
    assert payload["scatter_mask_4x2_rank1"] == list(
        readout_scatter_mask(d_state=4, mimo_rank=2, rank_index=1)
    )
    assert payload["scatter_shifts_dense_4x4"] == list(
        readout_scatter_shifts(d_state=4, mimo_rank=4, dense_output=True)
    )
    assert payload["scatter_shifts_rank_local_4x4"] == list(
        readout_scatter_shifts(d_state=4, mimo_rank=4, dense_output=False)
    )
    assert payload["output_slots_dense_4x4"] == list(
        readout_output_slots(d_state=4, mimo_rank=4, readout_strategy="rank-reduce")
    )
    assert payload["output_slots_rank_local_4x4"] == list(
        readout_output_slots(d_state=4, mimo_rank=4, readout_strategy="rank-local")
    )
