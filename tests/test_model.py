from __future__ import annotations

import torch

from fhe_native_mamba3.ckks import CkksConfig, CkksTrace
from fhe_native_mamba3.cost import estimate_block_cost
from fhe_native_mamba3.model import FheMamba3Config, FheMamba3ForCausalLM


def test_static_model_forward_and_backward() -> None:
    config = FheMamba3Config(
        vocab_size=32,
        d_model=24,
        n_layers=2,
        d_state=4,
        mimo_rank=3,
        max_seq_len=16,
        bc_mode="static",
    )
    model = FheMamba3ForCausalLM(config)
    input_ids = torch.randint(1, config.vocab_size, (2, 12))
    output = model(input_ids, labels=input_ids)
    assert output["logits"].shape == (2, 12, config.vocab_size)
    output["loss"].backward()
    assert model.embed.weight.grad is not None


def test_dynamic_model_forward() -> None:
    config = FheMamba3Config(
        vocab_size=32,
        d_model=16,
        n_layers=1,
        d_state=3,
        mimo_rank=2,
        max_seq_len=16,
        bc_mode="dynamic",
        gate_mode="quadratic",
    )
    model = FheMamba3ForCausalLM(config)
    input_ids = torch.randint(1, config.vocab_size, (2, 10))
    output = model(input_ids)
    assert output["logits"].shape == (2, 10, config.vocab_size)


def test_windowed_scan_matches_sequential_when_window_covers_sequence() -> None:
    torch.manual_seed(1)
    base = FheMamba3Config(
        vocab_size=32,
        d_model=16,
        n_layers=1,
        d_state=3,
        mimo_rank=2,
        max_seq_len=16,
        bc_mode="static",
        decay_mode="scalar",
        scan_mode="sequential",
    )
    windowed = FheMamba3Config(
        vocab_size=32,
        d_model=16,
        n_layers=1,
        d_state=3,
        mimo_rank=2,
        max_seq_len=16,
        bc_mode="static",
        decay_mode="scalar",
        scan_mode="windowed",
        effective_window=16,
    )
    sequential_model = FheMamba3ForCausalLM(base)
    windowed_model = FheMamba3ForCausalLM(windowed)
    windowed_model.load_state_dict(sequential_model.state_dict())
    input_ids = torch.randint(1, base.vocab_size, (2, 10))
    assert torch.allclose(
        sequential_model(input_ids)["logits"],
        windowed_model(input_ids)["logits"],
        atol=1e-5,
        rtol=1e-5,
    )


def test_cost_static_is_lower_depth_than_dynamic() -> None:
    static = FheMamba3Config(d_model=32, d_state=4, mimo_rank=2, bc_mode="static")
    dynamic = FheMamba3Config(d_model=32, d_state=4, mimo_rank=2, bc_mode="dynamic")
    static_cost = estimate_block_cost(static, seq_len=8)
    dynamic_cost = estimate_block_cost(dynamic, seq_len=8)
    assert static_cost.multiplicative_depth < dynamic_cost.multiplicative_depth
    assert static_cost.ciphertext_ciphertext_mul < dynamic_cost.ciphertext_ciphertext_mul


def test_ckks_trace_bootstraps_before_level_underflow() -> None:
    trace = CkksTrace(CkksConfig(max_level=5, min_level=2))
    trace.ct_ct_mul(depth=2)
    assert trace.level == 3
    trace.ct_ct_mul(depth=2)
    assert trace.bootstraps == 1
    assert trace.level == 3
