from __future__ import annotations

import pytest
import torch

from fhe_native_mamba3.backends.tracking import TrackingBackend
from fhe_native_mamba3.bundle_recurrence import build_weight_bundle_recurrence_problem
from fhe_native_mamba3.model import FheMamba3Config, FheMamba3ForCausalLM
from fhe_native_mamba3.openfhe_backend import run_static_mimo_recurrence_with_backend
from fhe_native_mamba3.weight_bundle import save_weight_bundle


def test_weight_bundle_recurrence_problem_uses_saved_static_weights(tmp_path) -> None:
    torch.manual_seed(23)
    config = FheMamba3Config(
        vocab_size=16,
        d_model=8,
        n_layers=1,
        d_state=2,
        mimo_rank=3,
        max_seq_len=8,
        bc_mode="static",
        decay_mode="scalar",
    )
    model = FheMamba3ForCausalLM(config)
    save_weight_bundle(model, tmp_path)

    extracted = build_weight_bundle_recurrence_problem(
        tmp_path,
        token_ids=(1, 2, 3, 4),
    )

    assert extracted.problem.seq_len == 4
    assert extracted.problem.d_state == 2
    assert extracted.problem.mimo_rank == 3
    assert extracted.problem.b == tuple(
        tuple(float(value) for value in row) for row in model.blocks[0].b_static.tolist()
    )
    assert extracted.problem.c == tuple(
        tuple(float(value) for value in row) for row in model.blocks[0].c_static.tolist()
    )
    assert extracted.problem.d_skip == tuple(
        float(value) for value in model.blocks[0].d_skip.tolist()
    )

    result = run_static_mimo_recurrence_with_backend(
        extracted.problem,
        backend=TrackingBackend(batch_size=extracted.problem.d_state * extracted.problem.mimo_rank),
        multiplicative_depth=8,
        readout_strategy="rank-local",
    )

    assert result.max_abs_error == 0
    assert result.backend_stats["backend"] == "tracking"


def test_weight_bundle_recurrence_problem_rejects_invalid_token_ids(tmp_path) -> None:
    config = FheMamba3Config(vocab_size=8, d_model=8, n_layers=1, d_state=2, mimo_rank=2)
    save_weight_bundle(FheMamba3ForCausalLM(config), tmp_path)

    with pytest.raises(ValueError, match="out of range"):
        build_weight_bundle_recurrence_problem(tmp_path, token_ids=(1, 9))


def test_weight_bundle_recurrence_problem_rejects_unsupported_layer_modes(tmp_path) -> None:
    dynamic_config = FheMamba3Config(
        vocab_size=8,
        d_model=8,
        n_layers=1,
        d_state=2,
        mimo_rank=2,
        bc_mode="dynamic",
    )
    dynamic_dir = tmp_path / "dynamic"
    save_weight_bundle(FheMamba3ForCausalLM(dynamic_config), dynamic_dir)

    with pytest.raises(ValueError, match="static B/C"):
        build_weight_bundle_recurrence_problem(dynamic_dir, token_ids=(1, 2))

    state_rank_config = FheMamba3Config(
        vocab_size=8,
        d_model=8,
        n_layers=1,
        d_state=2,
        mimo_rank=2,
        decay_mode="state_rank",
    )
    state_rank_dir = tmp_path / "state-rank"
    save_weight_bundle(FheMamba3ForCausalLM(state_rank_config), state_rank_dir)

    with pytest.raises(ValueError, match="scalar decay"):
        build_weight_bundle_recurrence_problem(state_rank_dir, token_ids=(1, 2))


def test_weight_bundle_recurrence_problem_rejects_bad_layer_and_context(tmp_path) -> None:
    config = FheMamba3Config(
        vocab_size=8,
        d_model=8,
        n_layers=1,
        d_state=2,
        mimo_rank=2,
        max_seq_len=3,
    )
    save_weight_bundle(FheMamba3ForCausalLM(config), tmp_path)

    with pytest.raises(ValueError, match="token_ids must be non-empty"):
        build_weight_bundle_recurrence_problem(tmp_path, token_ids=())
    with pytest.raises(ValueError, match="layer_index"):
        build_weight_bundle_recurrence_problem(tmp_path, token_ids=(1, 2), layer_index=1)
    with pytest.raises(ValueError, match="max_seq_len"):
        build_weight_bundle_recurrence_problem(tmp_path, token_ids=(1, 2, 3, 4))
