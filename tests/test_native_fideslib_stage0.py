from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_fideslib_stage0_native_kernel_is_repo_owned() -> None:
    source = ROOT / "native" / "fideslib_stage0" / "src" / "stage0_static_mimo.cpp"
    cmake = ROOT / "native" / "fideslib_stage0" / "CMakeLists.txt"
    slurm = ROOT / "slurm" / "fideslib_stage0.sbatch"
    sweep_slurm = ROOT / "slurm" / "fideslib_stage0_sweep.sbatch"

    assert source.exists()
    assert cmake.exists()
    assert slurm.exists()
    assert sweep_slurm.exists()

    source_text = source.read_text()
    assert "fideslib-static-mimo-recurrence" in source_text
    assert "EvalMult(h_cipher" in source_text
    assert "EvalMult(encrypted_inputs" in source_text
    assert '"client-update"' in source_text
    assert '"rank-reduce"' in source_text
    assert '"rank-local"' in source_text
    assert "EvalRotateKeyGen" in source_text
    assert "rank_reduce_readout" in source_text
    assert "readout_rotation_keys" in source_text
    assert "make_output_slots" in source_text
    assert "output_has_nonfinite" in source_text
    assert "client_plaintext_public_weight_multiplies" in source_text
    assert "level_alignment_unity_multiplies" in source_text

    slurm_text = slurm.read_text()
    assert "stage0_static_mimo" in slurm_text
    assert "fideslib_stage0_${SLURM_JOB_ID:-manual}" in slurm_text
    assert "INPUT_MODE:-client-update" in slurm_text
    assert "READOUT_MODE:-rank-reduce" in slurm_text

    sweep_text = sweep_slurm.read_text()
    assert "fideslib_stage0_sweep_${SLURM_JOB_ID:-manual}" in sweep_text
    assert "MIMO_RANKS" in sweep_text
    assert "Run sweep" in sweep_text
