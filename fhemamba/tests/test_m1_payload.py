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
        n_groups=1,
        num_hidden_layers=2,
        conv_kernel=4,
        chunk_size=8,
    )
    model = transformers.Mamba2ForCausalLM(config).float().eval()
    out = export_m1_payload(model, _IdTokenizer(), tmp_path / "payload", n_test_tokens=3)

    meta = json.loads((out / "meta.json").read_text())
    assert meta["dims"]["d_inner"] == 64
    assert meta["polys"]["decay_exp"]["kind"] == "cheb-resquared"
    assert meta["polys"]["gated_rms_invsqrt"]["kind"] == "sq-poly-newton"
    assert len(meta["polys"]["conv_silu"]["coeffs"]) - 1 == 96
    assert meta["polys"]["gated_rms_invsqrt"]["iterations"] == 4
    cb = meta["carried_bounds"]
    assert cb["state_abs_max"] > 0.0
    assert len(cb["state_head_abs_max"]) == meta["dims"]["num_heads"]
    assert max(cb["state_head_abs_max"]) == pytest.approx(cb["state_abs_max"])
    assert cb["fifo_abs_max"] > 0.0
    assert cb["source"] == "calibration_text"
    assert cb["calibration_tokens"] > 0
    assert set(cb["checkpoint_abs_max"]) == {
        "residual",
        "proj",
        "conv_silu",
        "dt",
        "y",
        "output",
        "gated_variance",
        "gated_newton",
    }
    assert all(value > 0.0 for value in cb["checkpoint_abs_max"].values())

    for name, shape in meta["tensors"].items():
        arr = np.fromfile(out / f"{name}.bin", dtype="<f4").reshape(shape)
        assert np.isfinite(arr).all(), name

    w = np.fromfile(out / "in_proj_w.bin", dtype="<f4").reshape(meta["tensors"]["in_proj_w"])
    assert np.allclose(w, model.backbone.layers[0].mixer.in_proj.weight.detach().numpy(), atol=1e-6)
    outs = np.fromfile(out / "test_layer_output.bin", dtype="<f4").reshape(
        meta["tensors"]["test_layer_output"]
    )
    assert outs.shape == (3, 32)
    states = np.fromfile(out / "test_state_output.bin", dtype="<f4").reshape(
        meta["tensors"]["test_state_output"]
    )
    assert states.shape == (3, 4, 16, 8)


def test_chain_export(tmp_path, monkeypatch) -> None:
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
    model = transformers.Mamba2ForCausalLM(config).float().eval()
    import fhemamba.m1_payload as payload_module

    monkeypatch.setattr(payload_module, "_poly_ops_from_export", lambda *_args: None)
    out = payload_module.export_chain_payload(
        model,
        _IdTokenizer(),
        tmp_path / "chain",
        n_test_tokens=2,
        autoregressive_prompt_tokens=2,
        autoregressive_generate_tokens=4,
        gated_init_degree=15,
        gated_newton_iterations=3,
    )
    chain = json.loads((out / "chain.json").read_text())
    assert chain["n_layers"] == 2
    assert chain["gated_norm"] == {"init_degree": 15, "newton_iterations": 3}
    assert (
        chain["test_token_ids"]
        == _IdTokenizer()("The capital of France is", return_tensors="pt").input_ids[0, :2].tolist()
    )
    for d in chain["layer_dirs"]:
        meta = json.loads((out / d / "meta.json").read_text())
        assert "test_layer_output_poly" in meta["tensors"]
        assert meta["tensors"]["test_state_output"] == [2, 4, 16, 8]
        assert meta["tensors"]["test_state_output_poly"] == [2, 4, 16, 8]
        assert meta["tensors"]["autoregressive_poly_layer_output"] == [5, 32]
        assert meta["tensors"]["autoregressive_poly_state_output"] == [5, 4, 16, 8]
        assert meta["test_token_ids"] == chain["test_token_ids"]
        assert meta["polys"]["gated_rms_invsqrt"]["iterations"] == 3
        assert len(meta["polys"]["gated_rms_invsqrt"]["coeffs"]) == 16

    poly_final_path = out / "chain_expected_poly_final.bin"
    exported_poly_final = np.fromfile(poly_final_path, dtype="<f4").copy()
    np.zeros_like(exported_poly_final).tofile(poly_final_path)
    payload_module.export_state_debug_references(model, out)
    assert np.array_equal(
        np.fromfile(poly_final_path, dtype="<f4"),
        exported_poly_final,
    )

    # Simulate an old payload lacking token IDs. The incremental path recovers
    # them from exported embedding rows without recalibration.
    legacy_chain = json.loads((out / "chain.json").read_text())
    expected_token_ids = legacy_chain.pop("test_token_ids")
    (out / "chain.json").write_text(json.dumps(legacy_chain, indent=2))
    for directory in chain["layer_dirs"]:
        meta_path = out / directory / "meta.json"
        legacy_meta = json.loads(meta_path.read_text())
        legacy_meta.pop("test_token_ids")
        meta_path.write_text(json.dumps(legacy_meta, indent=2))
    payload_module.export_state_debug_references(model, out, tokens=1)
    refreshed_meta = json.loads((out / chain["layer_dirs"][0] / "meta.json").read_text())
    assert refreshed_meta["test_token_ids"] == expected_token_ids
    assert refreshed_meta["tensors"]["test_layer_output_poly"] == [1, 32]
    assert refreshed_meta["tensors"]["test_state_output_poly"] == [1, 4, 16, 8]
    assert sum("polynomial-circuit layer boundary" in note for note in refreshed_meta["notes"]) == 1
    assert sum("post-update recurrent state" in note for note in refreshed_meta["notes"]) == 1
    with pytest.raises(ValueError, match="within the exported chain length"):
        payload_module.export_state_debug_references(model, out, tokens=3)
    finals = np.fromfile(out / "chain_expected_final.bin", dtype="<f4").reshape(
        chain["tensors"]["chain_expected_final"]
    )
    assert finals.shape == (2, 32)
    assert np.isfinite(finals).all()
    poly_finals = np.fromfile(out / "chain_expected_poly_final.bin", dtype="<f4").reshape(
        chain["tensors"]["chain_expected_poly_final"]
    )
    assert poly_finals.shape == finals.shape
    assert np.isfinite(poly_finals).all()
    autoregressive = chain["autoregressive"]
    assert autoregressive["protocol"] == "client-in-loop-greedy-v1"
    assert autoregressive["prompt_tokens"] == 2
    assert autoregressive["generate_tokens"] == 4
    assert autoregressive["server_evaluations"] == 5
    assert len(autoregressive["poly_evaluated_ids"]) == 5
    assert len(autoregressive["poly_generated_ids"]) == 4
    autoregressive_finals = np.fromfile(
        out / "autoregressive_poly_expected_final.bin", dtype="<f4"
    ).reshape(chain["tensors"]["autoregressive_poly_expected_final"])
    assert np.isfinite(autoregressive_finals).all()
    embedding = np.fromfile(out / "client_embedding_w.bin", dtype="<f4").reshape(
        chain["tensors"]["client_embedding_w"]
    )
    lm_head = (
        embedding
        if autoregressive["embedding_lm_head_tied"]
        else np.fromfile(out / "client_lm_head_w.bin", dtype="<f4").reshape(
            chain["tensors"]["client_lm_head_w"]
        )
    )
    selected = []
    for final in autoregressive_finals[autoregressive["prompt_tokens"] - 1 :]:
        logits = lm_head @ final
        if autoregressive["client_lm_head_bias"]:
            logits += np.fromfile(out / "client_lm_head_b.bin", dtype="<f4")
        selected.append(int(logits.argmax()))
    assert selected == autoregressive["poly_generated_ids"]


def test_add_autoregressive_assets_to_existing_chain(tmp_path, monkeypatch) -> None:
    torch.manual_seed(23)
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
    model = transformers.Mamba2ForCausalLM(config).float().eval()
    import fhemamba.m1_payload as payload_module

    monkeypatch.setattr(payload_module, "_poly_ops_from_export", lambda *_args: None)
    out = payload_module.export_chain_payload(
        model,
        _IdTokenizer(),
        tmp_path / "chain",
        n_test_tokens=2,
    )
    before = json.loads((out / "chain.json").read_text())
    assert before["autoregressive"] is None
    before_tensors = dict(before["tensors"])

    payload_module.export_autoregressive_client_payload(
        model,
        _IdTokenizer(),
        out,
        prompt_tokens=2,
        generate_tokens=4,
    )

    after = json.loads((out / "chain.json").read_text())
    assert before_tensors.items() <= after["tensors"].items()
    assert after["autoregressive"]["server_evaluations"] == 5
    assert len(after["autoregressive"]["poly_generated_ids"]) == 4
    for name in (
        "client_embedding_w",
        "autoregressive_poly_embeddings",
        "autoregressive_poly_expected_final",
        "autoregressive_exact_expected_final",
    ):
        assert (out / f"{name}.bin").stat().st_size > 0
    for directory in after["layer_dirs"]:
        meta = json.loads((out / directory / "meta.json").read_text())
        assert meta["tensors"]["autoregressive_poly_layer_output"] == [5, 32]
        assert meta["tensors"]["autoregressive_poly_state_output"] == [5, 4, 16, 8]


def test_legacy_const_newton_payload_spec() -> None:
    import fhemamba.m1_payload as payload_module

    guess = 0.125
    iterations = 3
    operation = payload_module._poly_from_export_spec(
        {"kind": "const-newton", "guess": guess, "iterations": iterations}
    )
    values = torch.tensor([0.25, 1.0, 4.0])
    expected = torch.full_like(values, guess)
    for _ in range(iterations):
        expected = expected * (1.5 - 0.5 * values * expected * expected)
    assert torch.equal(operation(values), expected)


def test_export_rejects_native_incompatible_groups(tmp_path) -> None:
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
    with pytest.raises(ValueError, match="n_groups == 1"):
        export_m1_payload(model, _IdTokenizer(), tmp_path / "payload")


def test_carried_bounds_do_not_depend_on_evaluation_prompt(tmp_path) -> None:
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
    model = transformers.Mamba2ForCausalLM(config).float().eval()
    first = export_m1_payload(model, _IdTokenizer(), tmp_path / "first", prompt="first")
    second = export_m1_payload(model, _IdTokenizer(), tmp_path / "second", prompt="second")
    first_meta = json.loads((first / "meta.json").read_text())
    second_meta = json.loads((second / "meta.json").read_text())
    assert first_meta["carried_bounds"] == second_meta["carried_bounds"]
