import numpy as np
from fhemamba.state_layout import (
    direct_state_block_cost,
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
