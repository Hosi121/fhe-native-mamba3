"""Reference forward vs the official transformers implementation."""

import pytest
import torch
from fhemamba.ops import Exact, PolyOps, RangeRecorder
from fhemamba.reference import chunked_scan, model_forward

transformers = pytest.importorskip("transformers")


@pytest.fixture(scope="module")
def tiny_model():
    config = transformers.MambaConfig(
        vocab_size=97,
        hidden_size=32,
        intermediate_size=64,
        state_size=8,
        num_hidden_layers=2,
        conv_kernel=4,
        time_step_rank=4,
        use_mambapy=False,
    )
    torch.manual_seed(7)
    model = transformers.MambaForCausalLM(config).float().eval()
    return model


@pytest.fixture(scope="module")
def token_ids():
    torch.manual_seed(11)
    return torch.randint(0, 97, (1, 33))  # odd length exercises chunk padding


def test_loop_scan_matches_official_forward(tiny_model, token_ids) -> None:
    with torch.no_grad():
        official = tiny_model(token_ids, output_hidden_states=True)
    ours = model_forward(tiny_model, token_ids, Exact(), scan="loop", output_hidden_states=True)
    assert len(ours["hidden_states"]) == len(official.hidden_states)
    for i, (theirs, mine) in enumerate(
        zip(official.hidden_states, ours["hidden_states"], strict=True)
    ):
        assert torch.allclose(theirs, mine, atol=1e-5), f"hidden state {i} diverged"
    assert torch.allclose(official.logits, ours["logits"], atol=1e-5)


def test_chunked_scan_matches_loop(tiny_model, token_ids) -> None:
    loop = model_forward(tiny_model, token_ids, Exact(), scan="loop")
    chunked = model_forward(tiny_model, token_ids, Exact(), scan="chunked")
    assert torch.allclose(loop["logits"], chunked["logits"], atol=1e-4)


def test_chunked_scan_against_direct_recurrence() -> None:
    torch.manual_seed(3)
    decay = torch.rand(2, 5, 50, 4) * 0.9 + 0.05
    update = torch.randn(2, 5, 50, 4)
    state = torch.zeros(2, 5, 4)
    expected = []
    for t in range(50):
        state = decay[:, :, t] * state + update[:, :, t]
        expected.append(state)
    expected = torch.stack(expected, dim=2)
    got = chunked_scan(decay, update, chunk=16)
    assert torch.allclose(got, expected, atol=1e-5)


def test_full_ladder_plumbing_stays_close_at_high_degree(tiny_model, token_ids) -> None:
    """Calibrate -> fit -> substitute every site; logits must stay near exact."""
    recorder = RangeRecorder()
    model_forward(tiny_model, token_ids, recorder, scan="chunked")
    poly_ops = PolyOps.fit(
        ranges_by_name=recorder.pooled_by_name(),
        enabled=frozenset(recorder.pooled_by_name()),
        degrees=dict.fromkeys(recorder.pooled_by_name(), 31),
    )
    exact = model_forward(tiny_model, token_ids, Exact(), scan="chunked")
    poly = model_forward(tiny_model, token_ids, poly_ops, scan="chunked")
    max_diff = (exact["logits"] - poly["logits"]).abs().max()
    assert float(max_diff) < 0.05, f"poly substitution moved logits by {float(max_diff)}"
    assert all(rate == 0.0 for rate in poly_ops.violation_summary().values())
