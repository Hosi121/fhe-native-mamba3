from __future__ import annotations

import pytest

from fhe_native_mamba3.backends.tracking import TrackingBackend
from fhe_native_mamba3.composite_rotation import (
    CompositeRotationBackend,
    composite_rotation_basis_for_steps,
    decompose_rotation_steps,
    estimate_composite_rotation_basis,
    normalize_rotation_step,
    power_of_two_rotation_basis,
    rotate_composite,
)


def test_decompose_rotation_uses_short_signed_representative() -> None:
    assert normalize_rotation_step(13, batch_size=16) == -3
    assert decompose_rotation_steps(13, batch_size=16) == (1, -4)
    assert decompose_rotation_steps(7, batch_size=16) == (-1, 8)
    assert decompose_rotation_steps(16, batch_size=16) == ()


def test_power_of_two_basis_can_be_complete_or_requested_only() -> None:
    assert power_of_two_rotation_basis(batch_size=16) == (-8, -4, -2, -1, 1, 2, 4, 8)
    assert composite_rotation_basis_for_steps((3, 13), batch_size=16) == (-4, -1, 1, 4)
    assert composite_rotation_basis_for_steps((3, 13), batch_size=16, complete_basis=True) == (
        -8,
        -4,
        -2,
        -1,
        1,
        2,
        4,
        8,
    )


def test_rotate_composite_matches_direct_tracking_rotation() -> None:
    direct_backend = TrackingBackend(batch_size=16)
    composite_backend = TrackingBackend(batch_size=16)
    values = tuple(float(index) for index in range(16))

    direct_ct = direct_backend.encrypt(values)
    composite_ct = composite_backend.encrypt(values)
    direct = direct_backend.rotate(direct_ct, 13)
    composite = rotate_composite(composite_backend, composite_ct, 13, batch_size=16)

    assert composite_backend.decrypt(composite, length=16) == pytest.approx(
        direct_backend.decrypt(direct, length=16)
    )
    assert composite_backend.stats().rotation_count == 2


def test_composite_rotation_backend_delegates_other_ops() -> None:
    base = TrackingBackend(batch_size=16)
    backend = CompositeRotationBackend(base)
    ciphertext = backend.encrypt(tuple(range(16)))

    rotated = backend.rotate(ciphertext, 7)

    assert backend.name == "composite-rotation(tracking)"
    assert backend.encrypted is False
    assert backend.decrypt(rotated, length=4) == pytest.approx((7.0, 8.0, 9.0, 10.0))
    assert base.stats().rotation_count == 2


def test_estimate_composite_rotation_basis_reports_memory_reduction() -> None:
    estimate = estimate_composite_rotation_basis(
        tuple(range(1, 1112)),
        batch_size=32768,
        key_size_mb=200.0,
    )

    assert estimate.requested_rotation_key_count == 1111
    assert estimate.basis_rotation_key_count == 20
    assert estimate.key_reduction_factor == pytest.approx(55.55, rel=1e-3)
    assert estimate.requested_estimated_key_memory_gib == pytest.approx(216.9921875)
    assert estimate.basis_estimated_key_memory_gib == pytest.approx(3.90625)
    assert estimate.max_composition_length <= 6
