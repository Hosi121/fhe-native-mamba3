from __future__ import annotations

from fhe_native_mamba3.benchmarks.stage0_mimo import Stage0MimoConfig, run_stage0_mimo


def test_stage0_tracking_benchmark_counts_operations() -> None:
    result = run_stage0_mimo(
        Stage0MimoConfig(
            backend="tracking",
            seq_len=3,
            d_state=2,
            mimo_rank=2,
            seed=7,
        )
    )
    assert result["stage"] == "0"
    assert result["encrypted"] is False
    assert result["model"]["parameter_count"] == 10
    assert result["model"]["input_mode"] == "client-update"
    assert result["operation_counts"]["ct_pt_mul"] == 18
    assert result["operation_counts"]["client_plaintext_public_weight_multiplies"] == 12
    assert result["operation_counts"]["rotations"] == 9
    assert result["max_abs_error"] == 0


def test_stage0_server_bx_keeps_server_plaintext_weight_multiply() -> None:
    result = run_stage0_mimo(
        Stage0MimoConfig(
            backend="tracking",
            seq_len=3,
            d_state=2,
            mimo_rank=2,
            seed=7,
            input_mode="server-bx",
        )
    )
    assert result["model"]["input_mode"] == "server-bx"
    assert result["operation_counts"]["ct_pt_mul"] == 21
    assert result["operation_counts"]["client_plaintext_public_weight_multiplies"] == 0
    assert result["max_abs_error"] == 0


def test_stage0_rank_reduce_uses_fewer_rotations_for_larger_state() -> None:
    slotwise = run_stage0_mimo(
        Stage0MimoConfig(
            backend="tracking",
            seq_len=2,
            d_state=4,
            mimo_rank=2,
            readout_strategy="slotwise",
        )
    )
    rank_reduce = run_stage0_mimo(
        Stage0MimoConfig(
            backend="tracking",
            seq_len=2,
            d_state=4,
            mimo_rank=2,
            readout_strategy="rank-reduce",
        )
    )
    assert slotwise["max_abs_error"] < 1e-12
    assert rank_reduce["max_abs_error"] < 1e-12
    assert rank_reduce["operation_counts"]["rotations"] < slotwise["operation_counts"]["rotations"]


def test_stage0_rank_local_keeps_outputs_in_rank_groups() -> None:
    rank_reduce = run_stage0_mimo(
        Stage0MimoConfig(
            backend="tracking",
            seq_len=2,
            d_state=4,
            mimo_rank=4,
            readout_strategy="rank-reduce",
        )
    )
    rank_local = run_stage0_mimo(
        Stage0MimoConfig(
            backend="tracking",
            seq_len=2,
            d_state=4,
            mimo_rank=4,
            readout_strategy="rank-local",
        )
    )
    assert rank_local["max_abs_error"] < 1e-12
    assert rank_local["ckks"]["rotations"] == [1, 2]
    assert rank_reduce["ckks"]["rotations"] == [1, 2, 3, 6, 9]
    assert rank_local["decrypted_outputs"] == rank_reduce["decrypted_outputs"]
    assert (
        rank_local["operation_counts"]["rotations"] < rank_reduce["operation_counts"]["rotations"]
    )
    assert (
        rank_local["operation_counts"]["ct_pt_mul"] < rank_reduce["operation_counts"]["ct_pt_mul"]
    )
    assert rank_local["operation_counts"]["encrypt"] < rank_reduce["operation_counts"]["encrypt"]
