from __future__ import annotations

import torch

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


def test_cost_static_is_lower_depth_than_dynamic() -> None:
    static = FheMamba3Config(d_model=32, d_state=4, mimo_rank=2, bc_mode="static")
    dynamic = FheMamba3Config(d_model=32, d_state=4, mimo_rank=2, bc_mode="dynamic")
    static_cost = estimate_block_cost(static, seq_len=8)
    dynamic_cost = estimate_block_cost(dynamic, seq_len=8)
    assert static_cost.multiplicative_depth < dynamic_cost.multiplicative_depth
    assert static_cost.ciphertext_ciphertext_mul < dynamic_cost.ciphertext_ciphertext_mul
