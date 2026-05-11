from __future__ import annotations

import pytest
import torch

from fhe_native_mamba3.backends.tracking import TrackingBackend
from fhe_native_mamba3.ssd import sequential_static_scan, ssd_static_scan
from fhe_native_mamba3.ssd_prefix_scan import (
    PlaintextPrefixScanKernel,
    backend_hillis_steele_prefix_products,
    build_packed_prefix_scan_plan,
    build_prefix_scan_metadata,
    causal_decay_weights,
    packed_prefix_scan_rotation_steps,
    prefix_decay_products,
    ssd_prefix_scan,
    ssd_prefix_scan_prefill,
)


def test_prefix_decay_products_support_scalar_and_state_rank_decay() -> None:
    scalar_decay = torch.tensor([0.5, 0.25])
    scalar_prefix = prefix_decay_products(
        scalar_decay,
        seq_len=3,
        decay_mode="scalar",
        rank=2,
    )
    assert torch.allclose(
        scalar_prefix,
        torch.tensor(
            [
                [0.5, 0.25],
                [0.25, 0.0625],
                [0.125, 0.015625],
            ]
        ),
    )

    state_decay = torch.tensor([[0.5, 0.25], [0.2, 0.1]])
    state_prefix = prefix_decay_products(
        state_decay,
        seq_len=3,
        decay_mode="state_rank",
        d_state=2,
        rank=2,
    )
    assert state_prefix.shape == (3, 2, 2)
    assert torch.allclose(state_prefix[0], state_decay)
    assert torch.allclose(state_prefix[2], state_decay.pow(3))


def test_causal_decay_weights_cover_full_and_truncated_windows() -> None:
    decay = torch.tensor([0.5])
    full = causal_decay_weights(
        decay,
        seq_len=4,
        decay_mode="scalar",
        rank=1,
    ).squeeze(-1)
    assert torch.allclose(
        full,
        torch.tensor(
            [
                [1.0, 0.0, 0.0, 0.0],
                [0.5, 1.0, 0.0, 0.0],
                [0.25, 0.5, 1.0, 0.0],
                [0.125, 0.25, 0.5, 1.0],
            ]
        ),
    )

    truncated = causal_decay_weights(
        decay,
        seq_len=4,
        decay_mode="scalar",
        window=2,
        rank=1,
    ).squeeze(-1)
    assert torch.allclose(
        truncated,
        torch.tensor(
            [
                [1.0, 0.0, 0.0, 0.0],
                [0.5, 1.0, 0.0, 0.0],
                [0.0, 0.5, 1.0, 0.0],
                [0.0, 0.0, 0.5, 1.0],
            ]
        ),
    )


def test_prefix_scan_metadata_tracks_hillis_steele_and_blelloch_work() -> None:
    hillis_steele = build_prefix_scan_metadata(seq_len=8, algorithm="hillis_steele")
    assert hillis_steele.window == 8
    assert hillis_steele.scan_depth == 3
    assert hillis_steele.scan_work_items == 17
    assert tuple(step.stride for step in hillis_steele.steps) == (1, 2, 4)

    truncated = build_prefix_scan_metadata(seq_len=8, window=2, algorithm="hillis_steele")
    assert truncated.window == 2
    assert truncated.scan_depth == 1
    assert truncated.scan_work_items == 7

    blelloch = build_prefix_scan_metadata(seq_len=8, algorithm="blelloch")
    assert blelloch.scan_depth == 6
    assert blelloch.scan_work_items == 14
    assert tuple(step.phase for step in blelloch.steps) == (
        "up_sweep",
        "up_sweep",
        "up_sweep",
        "down_sweep",
        "down_sweep",
        "down_sweep",
    )


def test_packed_prefix_scan_plan_accounts_for_lane_stride_and_slot_capacity() -> None:
    plan = build_packed_prefix_scan_plan(
        seq_len=64,
        window=32,
        lanes=16,
        slot_count=64,
    )

    assert plan.tokens_per_ciphertext == 4
    assert plan.ciphertext_count == 16
    assert plan.in_ciphertext_window == 4
    assert plan.rotations == (16, 32)
    assert plan.scan_depth == 2
    assert plan.requires_cross_ciphertext_carry is True
    assert plan.to_json_dict()["lanes"] == 16

    assert packed_prefix_scan_rotation_steps(seq_len=8, lanes=3, window=5) == (
        3,
        6,
        12,
    )


def test_backend_hillis_steele_prefix_products_matches_packed_cumprod() -> None:
    values = torch.tensor(
        [
            [0.5, 0.25],
            [0.2, 0.4],
            [0.1, 0.5],
            [0.8, 0.2],
        ],
        dtype=torch.float32,
    )
    backend = TrackingBackend(batch_size=8)
    decay_ct = backend.encrypt(tuple(float(value) for value in values.flatten()))

    result = backend_hillis_steele_prefix_products(
        decay_ct,
        seq_len=4,
        lanes=2,
        backend=backend,
    )
    decrypted = torch.tensor(backend.decrypt(result.ciphertext, length=8)).view(4, 2)

    assert torch.allclose(decrypted, torch.cumprod(values, dim=0))
    assert result.plan.rotations == (2, 4)
    assert backend.stats().rotation_count == 2
    assert backend.stats().ct_ct_mul_count == 2


def test_backend_hillis_steele_prefix_products_rejects_cross_ciphertext_scan() -> None:
    backend = TrackingBackend(batch_size=4)
    decay_ct = backend.encrypt((0.5, 0.5, 0.5, 0.5))

    with pytest.raises(ValueError, match="fit in one ciphertext"):
        backend_hillis_steele_prefix_products(
            decay_ct,
            seq_len=3,
            lanes=2,
            backend=backend,
        )


def test_ssd_prefix_scan_prefill_matches_sequential_and_ssd_scalar_decay() -> None:
    torch.manual_seed(13)
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
    result = ssd_prefix_scan_prefill(
        rank_input,
        b_terms,
        c_terms,
        decay,
        decay_mode="scalar",
    )

    assert torch.allclose(result.output, sequential, atol=1e-6, rtol=1e-6)
    assert torch.allclose(result.output, ssd, atol=1e-6, rtol=1e-6)
    assert result.scan_depth == 3
    assert result.scan_work_items == 14
    assert result.window == 7
    assert result.decay_mode == "scalar"


def test_ssd_prefix_scan_prefill_matches_sequential_and_ssd_state_rank_decay() -> None:
    torch.manual_seed(17)
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
    result = ssd_prefix_scan_prefill(
        rank_input,
        b_terms,
        c_terms,
        decay,
        decay_mode="state_rank",
    )

    assert torch.allclose(result.output, sequential, atol=1e-6, rtol=1e-6)
    assert torch.allclose(result.output, ssd, atol=1e-6, rtol=1e-6)
    assert result.decay_mode == "state_rank"


def test_ssd_prefix_scan_prefill_matches_truncated_ssd_window() -> None:
    rank_input = torch.tensor([[[1.0], [2.0], [3.0], [4.0]]])
    b_terms = torch.tensor([[2.0]])
    c_terms = torch.tensor([[3.0]])
    decay = torch.tensor([[[0.5]]])

    result = ssd_prefix_scan(
        rank_input,
        b_terms,
        c_terms,
        decay,
        decay_mode="scalar",
        window=2,
    )
    expected = torch.tensor([[[6.0], [15.0], [24.0], [33.0]]])
    ssd = ssd_static_scan(
        rank_input,
        b_terms,
        c_terms,
        decay,
        decay_mode="scalar",
        window=2,
    )
    assert torch.allclose(result.output, expected)
    assert torch.allclose(result.output, ssd)
    assert result.window == 2
    assert result.scan_depth == 1
    assert result.scan_work_items == 3


def test_ssd_prefix_scan_prefill_matches_truncated_state_rank_ssd_window() -> None:
    torch.manual_seed(19)
    rank_input = torch.randn(1, 5, 2)
    b_terms = torch.randn(3, 2)
    c_terms = torch.randn(3, 2)
    decay = torch.sigmoid(torch.randn(1, 3, 2))

    result = ssd_prefix_scan_prefill(
        rank_input,
        b_terms,
        c_terms,
        decay,
        decay_mode="state_rank",
        window=3,
    )
    ssd = ssd_static_scan(
        rank_input,
        b_terms,
        c_terms,
        decay,
        decay_mode="state_rank",
        window=3,
    )
    assert torch.allclose(result.output, ssd, atol=1e-6, rtol=1e-6)
    assert result.window == 3


def test_plaintext_kernel_implements_protocol_shape() -> None:
    kernel = PlaintextPrefixScanKernel()
    decay = torch.tensor([0.5])
    assert torch.allclose(
        kernel.prefix_products(decay, seq_len=2, decay_mode="scalar", rank=1),
        torch.tensor([[0.5], [0.25]]),
    )
    assert kernel.causal_weights(decay, seq_len=2, decay_mode="scalar", rank=1).shape == (
        2,
        2,
        1,
    )


def test_ssd_prefix_scan_rejects_invalid_window_and_shapes() -> None:
    with pytest.raises(ValueError, match="window"):
        ssd_prefix_scan_prefill(
            torch.zeros(1, 2, 1),
            torch.zeros(1, 1),
            torch.zeros(1, 1),
            torch.ones(1),
            decay_mode="scalar",
            window=0,
        )

    with pytest.raises(ValueError, match="rank_input"):
        ssd_prefix_scan_prefill(
            torch.zeros(2, 3),
            torch.zeros(1, 1),
            torch.zeros(1, 1),
            torch.ones(1),
            decay_mode="scalar",
        )
