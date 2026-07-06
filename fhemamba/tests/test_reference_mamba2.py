"""Mamba-2 reference forward vs the official transformers implementation."""

import pytest
import torch
from fhemamba.ops import Exact, PolyOps, RangeRecorder
from fhemamba.reference import model_forward

transformers = pytest.importorskip("transformers")


@pytest.fixture(scope="module")
def tiny_model():
    config = transformers.Mamba2Config(
        vocab_size=97,
        hidden_size=32,
        expand=2,  # intermediate = 64 = num_heads * head_dim
        num_heads=4,
        head_dim=16,
        state_size=8,
        n_groups=2,
        num_hidden_layers=2,
        conv_kernel=4,
        chunk_size=8,
    )
    torch.manual_seed(19)
    model = transformers.Mamba2ForCausalLM(config).float().eval()
    return model


@pytest.fixture(scope="module")
def token_ids():
    torch.manual_seed(23)
    return torch.randint(0, 97, (1, 33))  # odd length exercises chunk padding


def test_loop_scan_matches_official_forward(tiny_model, token_ids) -> None:
    with torch.no_grad():
        official = tiny_model(token_ids, output_hidden_states=True)
    ours = model_forward(tiny_model, token_ids, Exact(), scan="loop", output_hidden_states=True)
    assert len(ours["hidden_states"]) == len(official.hidden_states)
    for i, (theirs, mine) in enumerate(
        zip(official.hidden_states, ours["hidden_states"], strict=True)
    ):
        diff = float((theirs - mine).abs().max())
        # HF evaluates the chunked SSD algebra; the sequential loop is the same
        # recurrence reassociated, so agreement is fp-noise level.
        assert diff < 1e-4, f"hidden state {i} diverged by {diff}"
    assert torch.allclose(official.logits, ours["logits"], atol=1e-4)


def test_chunked_scan_matches_loop(tiny_model, token_ids) -> None:
    loop = model_forward(tiny_model, token_ids, Exact(), scan="loop")
    chunked = model_forward(tiny_model, token_ids, Exact(), scan="chunked")
    assert torch.allclose(loop["logits"], chunked["logits"], atol=1e-4)


def test_full_ladder_plumbing_on_mamba2(tiny_model, token_ids) -> None:
    recorder = RangeRecorder()
    model_forward(tiny_model, token_ids, recorder, scan="chunked")
    ranges = recorder.pooled_by_name()
    assert "gated_rms_invsqrt" in ranges  # Mamba-2's extra site is exercised
    poly_ops = PolyOps.fit(
        ranges_by_name=ranges,
        enabled=frozenset(ranges),
        degrees=dict.fromkeys(ranges, 31),
    )
    exact = model_forward(tiny_model, token_ids, Exact(), scan="chunked")
    poly = model_forward(tiny_model, token_ids, poly_ops, scan="chunked")
    max_diff = float((exact["logits"] - poly["logits"]).abs().max())
    assert max_diff < 0.05, f"poly substitution moved logits by {max_diff}"
