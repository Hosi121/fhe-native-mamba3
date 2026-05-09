from __future__ import annotations

import pytest
import torch

from fhe_native_mamba3.ssd import sequential_static_scan, ssd_static_scan


def test_ssd_static_scan_matches_sequential_scalar_decay() -> None:
    torch.manual_seed(3)
    rank_input = torch.randn(2, 7, 3)
    b_terms = torch.randn(4, 3)
    c_terms = torch.randn(4, 3)
    decay = torch.sigmoid(torch.randn(1, 1, 3))

    sequential = sequential_static_scan(
        rank_input,
        b_terms,
        c_terms,
        decay,
        decay_mode="scalar",
    )
    ssd = ssd_static_scan(
        rank_input,
        b_terms,
        c_terms,
        decay,
        decay_mode="scalar",
    )
    assert torch.allclose(ssd, sequential, atol=1e-6, rtol=1e-6)


def test_ssd_static_scan_matches_sequential_state_rank_decay() -> None:
    torch.manual_seed(4)
    rank_input = torch.randn(2, 6, 2)
    b_terms = torch.randn(3, 2)
    c_terms = torch.randn(3, 2)
    decay = torch.sigmoid(torch.randn(1, 3, 2))

    sequential = sequential_static_scan(
        rank_input,
        b_terms,
        c_terms,
        decay,
        decay_mode="state_rank",
    )
    ssd = ssd_static_scan(
        rank_input,
        b_terms,
        c_terms,
        decay,
        decay_mode="state_rank",
    )
    assert torch.allclose(ssd, sequential, atol=1e-6, rtol=1e-6)


def test_ssd_static_scan_window_truncates_prefix() -> None:
    rank_input = torch.tensor([[[1.0], [2.0], [3.0], [4.0]]])
    b_terms = torch.tensor([[2.0]])
    c_terms = torch.tensor([[3.0]])
    decay = torch.tensor([[[0.5]]])

    ssd = ssd_static_scan(
        rank_input,
        b_terms,
        c_terms,
        decay,
        decay_mode="scalar",
        window=2,
    )
    expected = torch.tensor([[[6.0], [15.0], [24.0], [33.0]]])
    assert torch.allclose(ssd, expected)


def test_ssd_static_scan_rejects_invalid_shapes() -> None:
    with pytest.raises(ValueError, match="rank_input"):
        ssd_static_scan(
            torch.zeros(2, 3),
            torch.zeros(4, 2),
            torch.zeros(4, 2),
            torch.zeros(2),
            decay_mode="scalar",
        )
