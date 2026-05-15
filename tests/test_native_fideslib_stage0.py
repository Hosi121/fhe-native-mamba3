from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_fideslib_stage0_native_kernel_is_repo_owned() -> None:
    source = ROOT / "native" / "fideslib_stage0" / "src" / "stage0_static_mimo.cpp"
    stage1_bootstrap_source = (
        ROOT / "native" / "fideslib_stage0" / "src" / "stage1_bootstrap_probe.cpp"
    )
    stage1_rotation_source = (
        ROOT / "native" / "fideslib_stage0" / "src" / "stage1_rotation_probe.cpp"
    )
    stage1_rank_gate_payload_eval_source = (
        ROOT / "native" / "fideslib_stage0" / "src" / "stage1_rank_gate_payload_eval.cpp"
    )
    stage1_rank_gate_fideslib_source = (
        ROOT / "native" / "fideslib_stage0" / "src" / "stage1_rank_gate_fideslib.cpp"
    )
    cmake = ROOT / "native" / "fideslib_stage0" / "CMakeLists.txt"
    slurm = ROOT / "slurm" / "fideslib_stage0.sbatch"
    sweep_slurm = ROOT / "slurm" / "fideslib_stage0_sweep.sbatch"
    stage1_bootstrap_slurm = ROOT / "slurm" / "fideslib_stage1_bootstrap_probe.sbatch"
    stage1_rotation_slurm = ROOT / "slurm" / "fideslib_stage1_rotation_probe.sbatch"
    stage1_rank_gate_slurm = ROOT / "slurm" / "fideslib_stage1_rank_gate_projection.sbatch"
    checkpoint_openfhe_slurm = ROOT / "slurm" / "mamba_checkpoint_openfhe_smoke.sbatch"
    bootstrap_openfhe_slurm = ROOT / "slurm" / "openfhe_bootstrap_latency.sbatch"
    segment_openfhe_slurm = ROOT / "slurm" / "openfhe_segment_samples.sbatch"
    all_layer_openfhe_slurm = ROOT / "slurm" / "openfhe_all_layer_recurrence.sbatch"
    full_layer_gate_slurm = ROOT / "slurm" / "mamba_checkpoint_full_layer_gate.sbatch"
    encrypted_pre_full_layer_gate_slurm = (
        ROOT / "slurm" / "mamba_checkpoint_encrypted_pre_recurrence_full_layer_gate.sbatch"
    )
    encrypted_pre_full_layer_gate_tracking_slurm = (
        ROOT / "slurm" / "mamba_checkpoint_encrypted_pre_recurrence_full_layer_gate_tracking.sbatch"
    )
    encrypted_pre_full_layer_chain_slurm = (
        ROOT / "slurm" / "mamba_checkpoint_encrypted_pre_recurrence_full_layer_chain.sbatch"
    )
    encrypted_pre_full_layer_chain_tracking_slurm = (
        ROOT
        / "slurm"
        / "mamba_checkpoint_encrypted_pre_recurrence_full_layer_chain_tracking.sbatch"
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
    assert stage1_bootstrap_source.exists()
    assert stage1_rotation_source.exists()
    assert stage1_rank_gate_payload_eval_source.exists()
    assert stage1_rank_gate_fideslib_source.exists()
    assert cmake.exists()
    assert slurm.exists()
    assert sweep_slurm.exists()
    assert stage1_bootstrap_slurm.exists()
    assert stage1_rotation_slurm.exists()
    assert stage1_rank_gate_slurm.exists()
    assert checkpoint_openfhe_slurm.exists()
    assert bootstrap_openfhe_slurm.exists()
    assert segment_openfhe_slurm.exists()
    assert all_layer_openfhe_slurm.exists()
    assert full_layer_gate_slurm.exists()
    assert encrypted_pre_full_layer_gate_slurm.exists()
    assert encrypted_pre_full_layer_gate_tracking_slurm.exists()
    assert encrypted_pre_full_layer_chain_slurm.exists()
    assert encrypted_pre_full_layer_chain_tracking_slurm.exists()
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

    cmake_text = cmake.read_text()
    assert "add_executable(stage1_bootstrap_probe" in cmake_text
    assert "add_executable(stage1_rotation_probe" in cmake_text
    assert "add_executable(stage1_rank_gate_payload_eval" in cmake_text
    assert "add_executable(stage1_rank_gate_fideslib" in cmake_text

    stage1_bootstrap_source_text = stage1_bootstrap_source.read_text()
    assert "fideslib-gpu-stage1-bootstrap-latency" in stage1_bootstrap_source_text
    assert '\\"input_mode\\":\\"bootstrap-probe\\"' in stage1_bootstrap_source_text
    assert '\\"stage1_target_compatible\\"' in stage1_bootstrap_source_text
    assert '\\"ring_dimension\\"' in stage1_bootstrap_source_text
    assert '\\"batch_size\\"' in stage1_bootstrap_source_text

    stage1_rotation_source_text = stage1_rotation_source.read_text()
    assert "fideslib-gpu-stage1-state-major-rotation-probe" in stage1_rotation_source_text
    assert "state-major-rotation-probe" in stage1_rotation_source_text
    assert "state-major-opmix-probe" in stage1_rotation_source_text
    assert "EvalRotateKeyGen" in stage1_rotation_source_text
    assert "EvalRotate(ciphertext" in stage1_rotation_source_text
    assert '\\"requested_rotation_key_count\\"' in stage1_rotation_source_text
    assert '\\"peak_rss_gib\\"' in stage1_rotation_source_text
    assert "config.multiplicative_depth >= 48" in stage1_rotation_source_text
    assert "target_ct_pt_muls" in stage1_rotation_source_text
    assert "fideslib-gpu-stage1-state-major-opmix-probe" in stage1_rotation_source_text

    rank_gate_payload_eval_text = stage1_rank_gate_payload_eval_source.read_text()
    assert "stage1-rank-gate-payload-native-eval" in rank_gate_payload_eval_text
    assert "pre_recurrence_rank_gate_only" in rank_gate_payload_eval_text

    rank_gate_fideslib_text = stage1_rank_gate_fideslib_source.read_text()
    assert "stage1-rank-gate-fideslib-projection" in rank_gate_fideslib_text
    assert "--artifact-version" in rank_gate_fideslib_text
    assert "--repo-commit" in rank_gate_fideslib_text
    assert '\\"version\\"' in rank_gate_fideslib_text
    assert '\\"repo_commit\\"' in rank_gate_fideslib_text
    assert (
        '\\"config\\":{\\"input_mode\\":\\"stage1-rank-gate-fideslib-projection\\"}'
        in rank_gate_fideslib_text
    )
    assert "pre_recurrence_rank_gate_projection" in rank_gate_fideslib_text
    assert "pre_recurrence_dynamic_bc" in rank_gate_fideslib_text
    assert "pre_recurrence_decay" in rank_gate_fideslib_text
    assert "recurrence_tail_executed" in rank_gate_fideslib_text
    assert "full_one_layer_polynomial_output_checked" in rank_gate_fideslib_text
    assert "diagnostic_max_abs_error" in rank_gate_fideslib_text
    assert "ckks_levels" in rank_gate_fideslib_text
    assert "ckks_level_telemetry" in rank_gate_fideslib_text
    assert "decrypt_failure_artifact" in rank_gate_fideslib_text
    assert "json_escape" in rank_gate_fideslib_text
    assert "chain_steps" in rank_gate_fideslib_text
    assert "ciphertext_recurrent_state_chain" in rank_gate_fideslib_text
    assert "build_repeated_chain_reference" in rank_gate_fideslib_text
    assert "bootstrap_before_chain_steps" in rank_gate_fideslib_text
    assert "EvalBootstrapSetup" in rank_gate_fideslib_text
    assert "EvalBootstrap(state_new_poly_ct)" in rank_gate_fideslib_text
    assert "write_runtime_failure_payload" in rank_gate_fideslib_text
    assert "previous_state_nonzero" in rank_gate_fideslib_text
    assert "state_new_poly" in rank_gate_fideslib_text
    assert "state_vector_to_state_major_ciphertext" in rank_gate_fideslib_text

    stage1_bootstrap_slurm_text = stage1_bootstrap_slurm.read_text()
    assert "stage1_bootstrap_probe" in stage1_bootstrap_slurm_text
    assert "run_fideslib_stage1_bootstrap_probe.py" in stage1_bootstrap_slurm_text
    assert "RING_DIM:-65536" in stage1_bootstrap_slurm_text
    assert "NUM_SLOTS:-32768" in stage1_bootstrap_slurm_text
    assert "export OUTPUT_JSON" in stage1_bootstrap_slurm_text

    stage1_rotation_slurm_text = stage1_rotation_slurm.read_text()
    assert "stage1_rotation_probe" in stage1_rotation_slurm_text
    assert "run_fideslib_stage1_rotation_probe.py" in stage1_rotation_slurm_text
    assert "RING_DIM:-131072" in stage1_rotation_slurm_text
    assert "MULTIPLICATIVE_DEPTH:-48" in stage1_rotation_slurm_text
    assert "SCALING_MOD_SIZE:-40" in stage1_rotation_slurm_text
    assert "ROTATION_ARTIFACT" in stage1_rotation_slurm_text
    assert "ROTATION_LIMIT" in stage1_rotation_slurm_text
    assert "TARGET_CT_PT_MULS" in stage1_rotation_slurm_text

    stage1_rank_gate_slurm_text = stage1_rank_gate_slurm.read_text()
    assert "stage1_rank_gate_fideslib" in stage1_rank_gate_slurm_text
    assert "export_stage1_rank_gate_payload.py" in stage1_rank_gate_slurm_text
    assert "ARTIFACT_VERSION" in stage1_rank_gate_slurm_text
    assert "REPO_COMMIT" in stage1_rank_gate_slurm_text
    assert "--artifact-version" in stage1_rank_gate_slurm_text
    assert "--repo-commit" in stage1_rank_gate_slurm_text
    assert "DECAY_POLYNOMIAL_DEGREE" in stage1_rank_gate_slurm_text
    assert "DT_PROJECTION_SCALE" in stage1_rank_gate_slurm_text
    assert "PREVIOUS_STATE_SCALE" in stage1_rank_gate_slurm_text
    assert "CHAIN_STEPS" in stage1_rank_gate_slurm_text
    assert "BOOTSTRAP_BEFORE_CHAIN_STEPS" in stage1_rank_gate_slurm_text
    assert "--chain-steps" in stage1_rank_gate_slurm_text
    assert "--bootstrap-before-chain-steps" in stage1_rank_gate_slurm_text
    assert "--previous-state-scale" in stage1_rank_gate_slurm_text

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
    assert 'BACKEND="${BACKEND:-openfhe}"' in encrypted_pre_full_layer_gate_text
    assert "RMS_NORM_MODE" in encrypted_pre_full_layer_gate_text
    assert "VISIBLE_OUTPUT_SCALE" in encrypted_pre_full_layer_gate_text
    assert "SCALE_PLAN_JSON" in encrypted_pre_full_layer_gate_text
    assert "--decay-polynomial-range=" in encrypted_pre_full_layer_gate_text
    assert "MAX_ESTIMATED_ROTATION_KEY_MEMORY_GIB" in encrypted_pre_full_layer_gate_text
    assert "pre_recurrence_depth_estimate" in encrypted_pre_full_layer_gate_text

    encrypted_pre_full_layer_gate_tracking_text = (
        encrypted_pre_full_layer_gate_tracking_slurm.read_text()
    )
    assert "#SBATCH --mem=8G" in encrypted_pre_full_layer_gate_tracking_text
    assert "--backend tracking" in encrypted_pre_full_layer_gate_tracking_text
    assert "run_checkpoint_encrypted_pre_recurrence_full_layer_gate.py" in (
        encrypted_pre_full_layer_gate_tracking_text
    )

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
    assert "MAX_ESTIMATED_ROTATION_KEY_MEMORY_GIB" in encrypted_pre_full_layer_chain_text
    assert 'BACKEND="${BACKEND:-openfhe}"' in encrypted_pre_full_layer_chain_text

    encrypted_pre_full_layer_chain_tracking_text = (
        encrypted_pre_full_layer_chain_tracking_slurm.read_text()
    )
    assert "#SBATCH --mem=8G" in encrypted_pre_full_layer_chain_tracking_text
    assert "--backend tracking" in encrypted_pre_full_layer_chain_tracking_text
    assert "run_checkpoint_encrypted_pre_recurrence_full_layer_chain.py" in (
        encrypted_pre_full_layer_chain_tracking_text
    )

    synthetic_pre_full_layer_chain_text = synthetic_pre_full_layer_chain_slurm.read_text()
    assert "run_synthetic_encrypted_pre_recurrence_full_layer_chain.py" in (
        synthetic_pre_full_layer_chain_text
    )
    assert "reduced_proxy" in synthetic_pre_full_layer_chain_text
    assert "D_MODEL" in synthetic_pre_full_layer_chain_text
    assert "WEIGHT_SCALE" in synthetic_pre_full_layer_chain_text

    visible_projection_sweep_text = visible_projection_sweep_slurm.read_text()
    assert "run_checkpoint_visible_projection_sweep.py" in visible_projection_sweep_text
    assert "VISIBLE_DIM_LIMITS" in visible_projection_sweep_text
    assert "MAX_ROTATION_KEYS" in visible_projection_sweep_text
    assert "MAX_OPENFHE_CHECKED_VISIBLE_DIM" in visible_projection_sweep_text

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
    assert "mamba_checkpoint_encrypted_pre_recurrence_full_layer_gate_tracking.sbatch" in (
        submit_jobs_text
    )
    assert "mamba_checkpoint_encrypted_pre_recurrence_full_layer_chain_tracking.sbatch" in (
        submit_jobs_text
    )
    assert "openfhe_all_layer_recurrence.sbatch" in submit_jobs_text
    assert "mamba_checkpoint_source_profile.sbatch" in submit_jobs_text
    assert "mamba_checkpoint_client_decode_smoke.sbatch" in submit_jobs_text
