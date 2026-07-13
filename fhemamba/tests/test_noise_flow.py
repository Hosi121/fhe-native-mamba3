"""Noise-flow analysis: amplification measurement + horizon/K* arithmetic."""

import pytest
import torch
from fhemamba.noise_flow import (
    horizon,
    measure_amplification,
    measure_group_amplification,
    reanchor_cadence,
)

transformers = pytest.importorskip("transformers")


def _tiny():
    torch.manual_seed(19)
    config = transformers.Mamba2Config(
        vocab_size=97,
        hidden_size=32,
        expand=2,
        num_heads=4,
        head_dim=16,
        state_size=8,
        n_groups=1,
        num_hidden_layers=2,
        conv_kernel=4,
        chunk_size=8,
    )
    return transformers.Mamba2ForCausalLM(config).float().eval()


def test_amplification_is_positive_and_finite() -> None:
    model = _tiny()
    torch.manual_seed(5)
    ids = torch.randint(0, 97, (1, 8))
    amp = measure_amplification(model, ids, probes=2)
    assert len(amp["lambda_out"]) == 2
    for lo, lc in zip(amp["lambda_out"], amp["lambda_carry"], strict=True):
        assert 0.0 <= lo < 1e4
        # Mamba-2 carries a state perturbation through multiplication by an
        # exponential decay in (0, 1]. This assertion catches accidental
        # inclusion of the randn probe magnitude in the measured gain.
        assert 0.0 <= lc <= 1.01


def test_group_amplification_matches_packed_groups_and_scales() -> None:
    model = _tiny()
    ids = torch.arange(8).unsqueeze(0)
    scales = [[2.0, 3.0], [5.0, 7.0]]
    result = measure_group_amplification(
        model,
        ids,
        heads_per_group=2,
        probes=2,
        state_group_scales=scales,
    )

    assert result["groups_per_layer"] == [2, 2]
    assert len(result["records"]) == 4
    for record in result["records"]:
        assert record["head_end"] - record["head_start"] == 2
        assert 0.0 <= record["carry_gain"] <= 1.01
        assert record["boundary_gain"] >= 0.0
        assert record["final_gain"] >= 0.0
        assert record["normalized_state_output_gain"] == pytest.approx(
            record["final_gain"] * record["state_scale"]
        )


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"heads_per_group": 3}, "not divisible"),
        (
            {"heads_per_group": 2, "state_group_scales": [[1.0], [1.0]]},
            "must contain 2 values",
        ),
        (
            {"heads_per_group": 2, "state_group_scales": [[1.0, -1.0], [1.0, 1.0]]},
            "positive and finite",
        ),
    ],
)
def test_group_amplification_rejects_incompatible_geometry(kwargs, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        measure_group_amplification(_tiny(), torch.arange(8).unsqueeze(0), **kwargs)


def test_horizon_monotone_in_epsilon() -> None:
    lam_out, lam_carry = [1.0, 2.0], [0.9, 0.8]
    h_small = horizon(1e-6, lam_out, lam_carry)
    h_big = horizon(1e-3, lam_out, lam_carry)
    assert h_small["horizon_tokens"] >= h_big["horizon_tokens"]
    assert h_big["horizon_tokens"] >= 1


def test_horizon_matches_geometric_closed_form() -> None:
    # single layer, lambda_carry < 1: steady-state error = eps*lo/(1-lc)
    eps, lo, lc = 1e-3, 1.0, 0.5
    steady = eps * lo / (1 - lc)  # = 0.002
    h = horizon(eps, [lo], [lc], budget=steady * 1.01)
    assert h["horizon_tokens"] > 1000  # never crosses: horizon saturates
    h2 = horizon(eps, [lo], [lc], budget=steady * 0.6)
    assert h2["horizon_tokens"] <= 2


def test_reanchor_cost_decreases_with_horizon() -> None:
    strong_noise = reanchor_cadence(1e-2, [2.0], [1.2], budget=5e-2)
    weak_noise = reanchor_cadence(1e-5, [2.0], [1.2], budget=5e-2)
    assert weak_noise["K_star"] > strong_noise["K_star"]
    assert weak_noise["cost_factor_T128"] < strong_noise["cost_factor_T128"]
