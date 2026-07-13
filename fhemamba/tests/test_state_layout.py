import numpy as np
import pytest
from fhemamba.state_layout import (
    direct_state_block_cost,
    normalized_recurrence_step,
    recurrent_state_group_scales,
    replicated_state_block_cost,
    replicated_state_blocks,
    state_block_reference,
)


def test_replicated_state_blocks_match_dense_mamba_layout() -> None:
    rng = np.random.default_rng(17)
    batch = 32768
    state_size = 128
    group_block = 256
    bc_base = 1536
    conv = np.zeros(batch)
    conv[:1792] = rng.standard_normal(1792)

    for base in (bc_base, bc_base + state_size):
        got = replicated_state_blocks(conv, base, bc_base, state_size, group_block)
        want = state_block_reference(conv, base, state_size, group_block)
        assert np.array_equal(got, want)


def test_replicated_state_blocks_reject_invalid_geometry() -> None:
    conv = np.zeros(64)
    try:
        replicated_state_blocks(conv, 8, 8, 4, 8)
    except ValueError:
        pass
    else:
        raise AssertionError("invalid partially filled geometry was accepted")


def test_replicated_schedule_reduces_mamba_bc_operations() -> None:
    direct = direct_state_block_cost(128, 4, 256)
    replicated = replicated_state_block_cost(128, 256)

    assert direct.ct_pt_mul == 128
    assert replicated.ct_pt_mul == 1
    assert direct.rotations == 43
    assert replicated.rotations == 16
    # Both B and C branches share one source mask.
    assert 1 + 2 * replicated.ct_pt_mul == 3


def test_persistent_normalized_recurrence_matches_original_coordinates() -> None:
    rng = np.random.default_rng(17)
    scale = 37.5
    state = rng.normal(size=32)
    normalized = state / scale
    for _ in range(6):
        decay = rng.uniform(0.7, 1.0, size=32)
        update = rng.normal(scale=0.2, size=32)
        readout = rng.normal(size=32)
        state = decay * state + update
        normalized, got = normalized_recurrence_step(normalized, decay, update, readout, scale)
        np.testing.assert_allclose(normalized, state / scale, rtol=1e-14, atol=1e-14)
        np.testing.assert_allclose(got, readout * state, rtol=1e-14, atol=1e-14)


def test_recurrent_state_group_scales_use_calibrated_group_maxima() -> None:
    scales = recurrent_state_group_scales([0.0, 2.0, 1.0, 4.0, 3.0, 0.5], 2)
    np.testing.assert_array_equal(scales, [2.0, 4.0, 3.0])
    assert recurrent_state_group_scales([0.0, 0.0], 2)[0] == 1e-6


@pytest.mark.parametrize(
    ("maxima", "group_heads", "message"),
    [
        ([], 1, "non-empty"),
        ([1.0, 2.0], 0, "divide"),
        ([1.0, 2.0, 3.0], 2, "divide"),
        ([float("nan")], 1, "finite"),
    ],
)
def test_recurrent_state_group_scales_reject_invalid_inputs(maxima, group_heads, message) -> None:
    with pytest.raises(ValueError, match=message):
        recurrent_state_group_scales(maxima, group_heads)
