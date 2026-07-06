"""Lowered decode step vs reference forward (numerics must match exactly)."""

import pytest
import torch
from fhemamba.lowering import Lowerer, lower_decode_step_mamba2
from fhemamba.ops import DEFAULT_DEGREES
from fhemamba.reference import init_states, model_forward

transformers = pytest.importorskip("transformers")


def test_lowered_decode_step_matches_reference() -> None:
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
    model = transformers.Mamba2ForCausalLM(config).float().eval()
    torch.manual_seed(23)
    ids = torch.randint(0, 97, (1, 6))

    ref_states = init_states(model)
    ref = model_forward(model, ids, states=ref_states, output_hidden_states=True)
    ref_final = ref["hidden_states"][-1][0, -1]

    low_states = init_states(model)
    lw = Lowerer(poly_degrees=dict(DEFAULT_DEGREES))
    final = None
    with torch.no_grad():
        embeds = model.backbone.embeddings(ids)[0]
    for t in range(ids.shape[1]):
        final = lower_decode_step_mamba2(model, embeds[t], low_states, lw)

    diff = float((final - ref_final).abs().max())
    assert diff < 1e-4, f"lowered decode diverged by {diff}"
    assert lw.c.ct_ct_mul > 0
    assert lw.c.rotations > 0
    assert any(name.endswith(".out") for name, _ in lw.c.stages)


def test_bsgs_rotation_bound() -> None:
    from fhemamba.lowering import _bsgs_rotations

    assert _bsgs_rotations(768) <= 2 * 28  # ~2*sqrt(768)
    assert _bsgs_rotations(1) == 0
