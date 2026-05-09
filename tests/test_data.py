from __future__ import annotations

import pytest
import torch

from fhe_native_mamba3.data import generate_modular_stream


def test_generate_modular_stream_is_deterministic_for_seed() -> None:
    first_x, first_y = generate_modular_stream(
        batch_size=2,
        seq_len=6,
        vocab_size=11,
        device="cpu",
        seed=17,
    )
    second_x, second_y = generate_modular_stream(
        batch_size=2,
        seq_len=6,
        vocab_size=11,
        device="cpu",
        seed=17,
    )

    assert torch.equal(first_x, second_x)
    assert torch.equal(first_y, second_y)


def test_generate_modular_stream_labels_follow_rule() -> None:
    x, y = generate_modular_stream(
        batch_size=3,
        seq_len=7,
        vocab_size=13,
        device="cpu",
        seed=23,
    )

    assert torch.equal(y[:, :2], x[:, :2])
    expected = (x[:, 1:-1] + 2 * x[:, :-2] + 3) % 12 + 1
    assert torch.equal(y[:, 2:], expected)


def test_generate_modular_stream_rejects_tiny_vocab() -> None:
    with pytest.raises(ValueError, match="vocab_size"):
        generate_modular_stream(
            batch_size=1,
            seq_len=4,
            vocab_size=7,
            device="cpu",
        )
