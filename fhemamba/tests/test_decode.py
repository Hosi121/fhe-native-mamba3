"""Stateful decode vs full-sequence forward, and greedy parity vs HF generate."""

import pytest
import torch
from fhemamba.generate import generate_greedy
from fhemamba.reference import init_states, model_forward

transformers = pytest.importorskip("transformers")


def _tiny_mamba1():
    torch.manual_seed(7)
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
    return transformers.MambaForCausalLM(config).float().eval()


def _tiny_mamba2():
    torch.manual_seed(19)
    config = transformers.Mamba2Config(
        vocab_size=97,
        hidden_size=32,
        expand=2,
        num_heads=4,
        head_dim=16,
        state_size=8,
        n_groups=2,
        num_hidden_layers=2,
        conv_kernel=4,
        chunk_size=8,
    )
    return transformers.Mamba2ForCausalLM(config).float().eval()


@pytest.mark.parametrize("factory", [_tiny_mamba1, _tiny_mamba2])
def test_token_by_token_decode_matches_full_forward(factory) -> None:
    model = factory()
    torch.manual_seed(29)
    ids = torch.randint(0, 97, (1, 17))
    full = model_forward(model, ids)["logits"]

    states = init_states(model)
    stepwise = []
    for t in range(ids.shape[1]):
        out = model_forward(model, ids[:, t : t + 1], states=states)
        stepwise.append(out["logits"][:, 0])
    stepwise = torch.stack(stepwise, dim=1)
    diff = float((full - stepwise).abs().max())
    assert diff < 1e-4, f"decode path diverged from full forward by {diff}"


@pytest.mark.parametrize("factory", [_tiny_mamba1, _tiny_mamba2])
def test_stateful_prefill_then_decode_matches_full_forward(factory) -> None:
    model = factory()
    torch.manual_seed(31)
    ids = torch.randint(0, 97, (1, 21))
    full = model_forward(model, ids)["logits"]

    states = init_states(model)
    prefill = model_forward(model, ids[:, :13], scan="chunked", states=states)
    tail = []
    for t in range(13, ids.shape[1]):
        out = model_forward(model, ids[:, t : t + 1], states=states)
        tail.append(out["logits"][:, 0])
    got_last = torch.stack(tail, dim=1)
    assert torch.allclose(full[:, :13], prefill["logits"], atol=1e-4)
    assert torch.allclose(full[:, 13:], got_last, atol=1e-4)


@pytest.mark.parametrize("factory", [_tiny_mamba1, _tiny_mamba2])
def test_greedy_generation_matches_hf_generate(factory) -> None:
    model = factory()
    torch.manual_seed(37)
    ids = torch.randint(0, 97, (1, 9))
    hf_tokens = model.generate(ids, max_new_tokens=6, do_sample=False, pad_token_id=0)[
        0, ids.shape[1] :
    ].tolist()
    ours = generate_greedy(model, ids, 6)
    # HF may stop early on the config's eos token; compare the overlap.
    assert len(hf_tokens) >= 3
    assert ours[: len(hf_tokens)] == hf_tokens
