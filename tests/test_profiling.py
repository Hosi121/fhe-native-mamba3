from __future__ import annotations

import math

import pytest
import torch

from fhe_native_mamba3.model import FheMamba3Config, FheMamba3ForCausalLM
from fhe_native_mamba3.profiling import (
    estimate_cumulative_log_contraction,
    estimate_high_decay_burst_len,
    profile_model_batch,
)


def test_decay_trace_helpers_use_worst_decay_per_position() -> None:
    decay = torch.tensor(
        [
            [0.50, 0.80],
            [0.96, 0.70],
            [0.97, 0.60],
            [0.20, 0.99],
            [0.30, 0.40],
        ]
    )

    expected_per_position = torch.tensor([0.80, 0.96, 0.97, 0.99, 0.40])
    expected_log = torch.log(expected_per_position).cumsum(dim=0)

    assert estimate_cumulative_log_contraction(decay) == pytest.approx(
        tuple(float(value) for value in expected_log)
    )
    assert estimate_high_decay_burst_len(decay, threshold=0.95) == 3
    assert estimate_high_decay_burst_len(decay.unsqueeze(0), threshold=0.95, position_dim=1) == 3


def test_profile_model_batch_reports_position_and_worst_case_summaries() -> None:
    torch.manual_seed(13)
    config = FheMamba3Config(
        vocab_size=32,
        d_model=16,
        n_layers=2,
        d_state=3,
        mimo_rank=2,
        max_seq_len=16,
        bc_mode="static",
    )
    model = FheMamba3ForCausalLM(config)
    high_decay = 0.97
    with torch.no_grad():
        for block in model.blocks:
            block.decay_logits.fill_(math.log(high_decay / (1.0 - high_decay)))

    input_ids = torch.randint(1, config.vocab_size, (2, 9))
    profile = profile_model_batch(
        model,
        input_ids,
        labels=input_ids,
        beta_grid=(0.5,),
        position_bucket_count=3,
        high_decay_threshold=0.95,
    )
    payload = profile.to_json_dict()

    assert len(payload["position_buckets"]) == 3
    assert payload["position_buckets"][0]["start"] == 0
    assert payload["position_buckets"][0]["end"] == 3
    assert payload["position_buckets"][0]["token_count"] == 6

    first_block = payload["blocks"][0]
    assert first_block["lambda_by_beta"]["0.5"] >= 0.0
    assert len(first_block["position_buckets"]) == 3
    assert first_block["position_buckets"][0]["token_count"] == 6
    assert first_block["position_buckets"][0]["decay_abs_max"] == pytest.approx(high_decay)
    assert first_block["high_decay_burst_len"] == 9
    assert first_block["log_contraction_total"] == pytest.approx(9 * math.log(high_decay))

    assert payload["max_high_decay_burst_len"] == 9
    assert payload["global_maxima"]["high_decay_burst_len"] == 9.0
    assert payload["global_maxima"]["state_abs_max"] >= 0.0
    assert payload["worst_case_blocks"]["high_decay_burst_len"]["layer"] in {0, 1}
    assert payload["worst_case_blocks"]["state_abs_max"]["value"] >= 0.0


def test_profile_model_batch_handles_dynamic_state_rank_toy_model() -> None:
    torch.manual_seed(17)
    config = FheMamba3Config(
        vocab_size=24,
        d_model=12,
        n_layers=1,
        d_state=2,
        mimo_rank=3,
        max_seq_len=12,
        bc_mode="dynamic",
        decay_mode="state_rank",
        gate_mode="quadratic",
    )
    model = FheMamba3ForCausalLM(config)
    input_ids = torch.randint(1, config.vocab_size, (2, 6))

    payload = profile_model_batch(
        model,
        input_ids,
        position_bucket_count=2,
        high_decay_threshold=0.95,
    ).to_json_dict()

    assert len(payload["blocks"]) == 1
    assert len(payload["blocks"][0]["position_buckets"]) == 2
    assert payload["blocks"][0]["high_decay_burst_len"] == 0
    assert payload["global_maxima"]["update_abs_max"] >= 0.0
