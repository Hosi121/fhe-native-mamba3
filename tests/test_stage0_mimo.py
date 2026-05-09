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
    assert result["operation_counts"]["ct_pt_mul"] == 21
    assert result["operation_counts"]["rotations"] == 9
    assert result["max_abs_error"] == 0
