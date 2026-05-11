from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_fideslib_stage0_native_kernel_is_repo_owned() -> None:
    source = ROOT / "native" / "fideslib_stage0" / "src" / "stage0_static_mimo.cpp"
    cmake = ROOT / "native" / "fideslib_stage0" / "CMakeLists.txt"
    slurm = ROOT / "slurm" / "fideslib_stage0.sbatch"
    sweep_slurm = ROOT / "slurm" / "fideslib_stage0_sweep.sbatch"
    checkpoint_openfhe_slurm = ROOT / "slurm" / "mamba_checkpoint_openfhe_smoke.sbatch"
    bootstrap_openfhe_slurm = ROOT / "slurm" / "openfhe_bootstrap_latency.sbatch"
    segment_openfhe_slurm = ROOT / "slurm" / "openfhe_segment_samples.sbatch"
    all_layer_openfhe_slurm = ROOT / "slurm" / "openfhe_all_layer_recurrence.sbatch"
    full_layer_gate_slurm = ROOT / "slurm" / "mamba_checkpoint_full_layer_gate.sbatch"
    encrypted_pre_full_layer_gate_slurm = (
        ROOT / "slurm" / "mamba_checkpoint_encrypted_pre_recurrence_full_layer_gate.sbatch"
    )
    encrypted_pre_full_layer_chain_slurm = (
        ROOT / "slurm" / "mamba_checkpoint_encrypted_pre_recurrence_full_layer_chain.sbatch"
    )
    synthetic_pre_full_layer_chain_slurm = (
        ROOT / "slurm" / "synthetic_encrypted_pre_recurrence_full_layer_chain.sbatch"
    )
    full_layer_sweep_slurm = ROOT / "slurm" / "mamba_checkpoint_full_layer_sweep.sbatch"
    source_profile_slurm = ROOT / "slurm" / "mamba_checkpoint_source_profile.sbatch"
    client_decode_slurm = ROOT / "slurm" / "mamba_checkpoint_client_decode_smoke.sbatch"
    visible_projection_sweep_slurm = (
        ROOT / "slurm" / "mamba_checkpoint_visible_projection_sweep.sbatch"
    )
    handoff_openfhe_slurm = ROOT / "slurm" / "openfhe_ciphertext_handoff.sbatch"
    recurrence_chain_openfhe_slurm = ROOT / "slurm" / "openfhe_recurrence_chain.sbatch"
    submit_stage0_jobs = ROOT / "scripts" / "submit_stage0_high_jobs.sh"

    assert source.exists()
    assert cmake.exists()
    assert slurm.exists()
    assert sweep_slurm.exists()
    assert checkpoint_openfhe_slurm.exists()
    assert bootstrap_openfhe_slurm.exists()
    assert segment_openfhe_slurm.exists()
    assert all_layer_openfhe_slurm.exists()
    assert full_layer_gate_slurm.exists()
    assert encrypted_pre_full_layer_gate_slurm.exists()
    assert encrypted_pre_full_layer_chain_slurm.exists()
    assert synthetic_pre_full_layer_chain_slurm.exists()
    assert full_layer_sweep_slurm.exists()
    assert source_profile_slurm.exists()
    assert client_decode_slurm.exists()
    assert visible_projection_sweep_slurm.exists()
    assert handoff_openfhe_slurm.exists()
    assert recurrence_chain_openfhe_slurm.exists()
    assert submit_stage0_jobs.exists()

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

    checkpoint_openfhe_text = checkpoint_openfhe_slurm.read_text()
    assert "mamba-checkpoint-recurrence-smoke" in checkpoint_openfhe_text
    assert "--backend openfhe" in checkpoint_openfhe_text
    assert "INSTALL_OPENFHE" in checkpoint_openfhe_text
    assert "BOOTSTRAP_AFTER_TOKENS" in checkpoint_openfhe_text
    assert "BOOTSTRAP_CORRECTION_FACTOR" in checkpoint_openfhe_text
    assert "recommended_multiplicative_depth" in checkpoint_openfhe_text

    bootstrap_openfhe_text = bootstrap_openfhe_slurm.read_text()
    assert "measure_openfhe_bootstrap_latency.py" in bootstrap_openfhe_text
    assert "BOOTSTRAP_LEVEL_BUDGET" in bootstrap_openfhe_text
    assert "INSTALL_OPENFHE" in bootstrap_openfhe_text

    segment_openfhe_text = segment_openfhe_slurm.read_text()
    assert "run_openfhe_segment_samples.py" in segment_openfhe_text
    assert "BOOTSTRAP_AFTER_TOKENS" in segment_openfhe_text
    assert "MULTIPLICATIVE_DEPTH_OVERRIDE" in segment_openfhe_text

    all_layer_openfhe_text = all_layer_openfhe_slurm.read_text()
    assert "run_openfhe_all_layer_recurrence.py" in all_layer_openfhe_text
    assert "--all-layers" in all_layer_openfhe_text
    assert "BOOTSTRAP_SEC" in all_layer_openfhe_text
    assert "EXECUTE_SCHEDULED_BOOTSTRAP" in all_layer_openfhe_text
    assert "BOOTSTRAP_MULTIPLICATIVE_DEPTH" in all_layer_openfhe_text

    full_layer_gate_text = full_layer_gate_slurm.read_text()
    assert "mamba-checkpoint-full-layer-gate" in full_layer_gate_text
    assert "MAX_ROTATION_KEYS" in full_layer_gate_text
    assert "VISIBLE_DIM_LIMIT" in full_layer_gate_text
    assert "full-layer gate" in full_layer_gate_text

    encrypted_pre_full_layer_gate_text = encrypted_pre_full_layer_gate_slurm.read_text()
    assert "run_checkpoint_encrypted_pre_recurrence_full_layer_gate.py" in (
        encrypted_pre_full_layer_gate_text
    )
    assert "#SBATCH --mem=512G" in encrypted_pre_full_layer_gate_text
    assert "RMS_NORM_MODE" in encrypted_pre_full_layer_gate_text
    assert "--decay-polynomial-range=" in encrypted_pre_full_layer_gate_text
    assert "pre_recurrence_depth_estimate" in encrypted_pre_full_layer_gate_text

    full_layer_sweep_text = full_layer_sweep_slurm.read_text()
    assert "run_checkpoint_full_layer_sweep.py" in full_layer_sweep_text
    assert "LAYER_COUNT" in full_layer_sweep_text
    assert "VISIBLE_DIM_LIMIT" in full_layer_sweep_text
    assert "full-layer sweep" in full_layer_sweep_text

    source_profile_text = source_profile_slurm.read_text()
    assert "run_checkpoint_source_profile.py" in source_profile_text
    assert "PROFILE_ALL_LAYERS" in source_profile_text
    assert "top1_top2_gap" in source_profile_text

    client_decode_text = client_decode_slurm.read_text()
    assert "run_checkpoint_client_decode_smoke.py" in client_decode_text
    assert "DECODE_ALL_LAYERS" in client_decode_text
    assert "client_side_argmax" in client_decode_text

    encrypted_pre_full_layer_chain_text = encrypted_pre_full_layer_chain_slurm.read_text()
    assert "run_checkpoint_encrypted_pre_recurrence_full_layer_chain.py" in (
        encrypted_pre_full_layer_chain_text
    )
    assert "inter_layer_ciphertext_handoff" in encrypted_pre_full_layer_chain_text
    assert "N_LAYERS" in encrypted_pre_full_layer_chain_text

    synthetic_pre_full_layer_chain_text = synthetic_pre_full_layer_chain_slurm.read_text()
    assert "run_synthetic_encrypted_pre_recurrence_full_layer_chain.py" in (
        synthetic_pre_full_layer_chain_text
    )
    assert "reduced_proxy" in synthetic_pre_full_layer_chain_text
    assert "D_MODEL" in synthetic_pre_full_layer_chain_text

    visible_projection_sweep_text = visible_projection_sweep_slurm.read_text()
    assert "run_checkpoint_visible_projection_sweep.py" in visible_projection_sweep_text
    assert "VISIBLE_DIM_LIMITS" in visible_projection_sweep_text
    assert "MAX_ROTATION_KEYS" in visible_projection_sweep_text

    handoff_openfhe_text = handoff_openfhe_slurm.read_text()
    assert "run_ciphertext_handoff_smoke.py" in handoff_openfhe_text
    assert "BOOTSTRAP_AFTER_LAYERS" in handoff_openfhe_text
    assert "no_intermediate_decrypt" in handoff_openfhe_text

    recurrence_chain_openfhe_text = recurrence_chain_openfhe_slurm.read_text()
    assert "run_openfhe_recurrence_chain_smoke.py" in recurrence_chain_openfhe_text
    assert "BOOTSTRAP_AFTER_LAYERS" in recurrence_chain_openfhe_text
    assert "ciphertext_chain" in recurrence_chain_openfhe_text

    submit_jobs_text = submit_stage0_jobs.read_text()
    assert "SUBMIT_FULL_LAYER_GATE" in submit_jobs_text
    assert "SUBMIT_ENCRYPTED_PRE_RECURRENCE_FULL_LAYER_GATE" in submit_jobs_text
    assert "SUBMIT_ENCRYPTED_PRE_RECURRENCE_FULL_LAYER_CHAIN" in submit_jobs_text
    assert "mamba_checkpoint_full_layer_gate.sbatch" in submit_jobs_text
    assert "mamba_checkpoint_encrypted_pre_recurrence_full_layer_gate.sbatch" in (submit_jobs_text)
    assert "mamba_checkpoint_encrypted_pre_recurrence_full_layer_chain.sbatch" in (submit_jobs_text)
    assert "openfhe_all_layer_recurrence.sbatch" in submit_jobs_text
    assert "mamba_checkpoint_source_profile.sbatch" in submit_jobs_text
    assert "mamba_checkpoint_client_decode_smoke.sbatch" in submit_jobs_text
