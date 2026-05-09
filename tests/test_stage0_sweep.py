from __future__ import annotations

from fhe_native_mamba3.benchmarks.stage0_sweep import Stage0SweepConfig, run_stage0_sweep


def test_stage0_sweep_returns_summary() -> None:
    result = run_stage0_sweep(
        Stage0SweepConfig(
            backend="tracking",
            seq_lens=(2,),
            d_states=(2, 4),
            mimo_ranks=(2,),
            readout_strategies=("slotwise", "rank-reduce"),
            input_modes=("client-update", "server-bx"),
        )
    )
    assert result["stage"] == "0"
    assert result["result_count"] == 8
    assert result["summary"]["max_abs_error_max"] < 1e-12
    assert result["summary"]["lowest_rotations"]["readout_strategy"] == "rank-reduce"
    assert result["summary"]["lowest_ct_pt_mul"]["input_mode"] == "client-update"
