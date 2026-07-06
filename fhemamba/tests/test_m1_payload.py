"""Payload export round-trip on a tiny Mamba-2."""

import json

import numpy as np
import pytest
import torch
from fhemamba.m1_payload import export_m1_payload

transformers = pytest.importorskip("transformers")


class _IdTokenizer:
    def __call__(self, text, return_tensors=None):
        ids = torch.tensor([[(ord(c) % 90) + 3 for c in text[:64]]])

        class R:
            input_ids = ids

        return R()


def test_export_round_trip(tmp_path) -> None:
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
    out = export_m1_payload(model, _IdTokenizer(), tmp_path / "payload", n_test_tokens=3)

    meta = json.loads((out / "meta.json").read_text())
    assert meta["dims"]["d_inner"] == 64
    assert meta["polys"]["decay_exp"]["kind"] == "cheb-resquared"
    assert meta["polys"]["gated_rms_invsqrt"]["iterations"] == 14

    for name, shape in meta["tensors"].items():
        arr = np.fromfile(out / f"{name}.bin", dtype="<f4").reshape(shape)
        assert np.isfinite(arr).all(), name

    w = np.fromfile(out / "in_proj_w.bin", dtype="<f4").reshape(meta["tensors"]["in_proj_w"])
    assert np.allclose(w, model.backbone.layers[0].mixer.in_proj.weight.detach().numpy(), atol=1e-6)
    outs = np.fromfile(out / "test_layer_output.bin", dtype="<f4").reshape(
        meta["tensors"]["test_layer_output"]
    )
    assert outs.shape == (3, 32)


def test_chain_export(tmp_path) -> None:
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
    from fhemamba.m1_payload import export_chain_payload

    out = export_chain_payload(model, _IdTokenizer(), tmp_path / "chain", n_test_tokens=2)
    chain = json.loads((out / "chain.json").read_text())
    assert chain["n_layers"] == 2
    for d in chain["layer_dirs"]:
        assert (out / d / "meta.json").exists()
    finals = np.fromfile(out / "chain_expected_final.bin", dtype="<f4").reshape(
        chain["tensors"]["chain_expected_final"]
    )
    assert finals.shape == (2, 32)
    assert np.isfinite(finals).all()
