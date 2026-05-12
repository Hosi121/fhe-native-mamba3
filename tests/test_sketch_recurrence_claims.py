from __future__ import annotations

import fhe_native_mamba3 as fhm3
from fhe_native_mamba3.sketch_recurrence_claims import classify_sketch_recurrence_claim


def test_classify_scalar_recurrence_claim_exact() -> None:
    claim = classify_sketch_recurrence_claim(
        recurrence_type="scalar",
        recurrence_compat_available=True,
        recurrence_compat_max_abs_error=1e-12,
    )

    assert claim.compatibility_status == "exact"
    assert claim.exact_recurrence_claimed is True
    assert claim.readout_error_only is False


def test_classify_scalar_recurrence_claim_approximate() -> None:
    claim = classify_sketch_recurrence_claim(
        recurrence_type="rank-scalar",
        recurrence_compat_available=True,
        recurrence_compat_max_abs_error=1e-4,
    )

    assert claim.compatibility_status == "approximate"
    assert claim.approximate_recurrence_claimed is True


def test_classify_rank_state_recurrence_claim_unavailable() -> None:
    claim = classify_sketch_recurrence_claim(
        recurrence_type="rank-state",
        recurrence_compat_available=False,
        recurrence_compat_max_abs_error=None,
    )

    assert claim.compatibility_status == "unavailable"
    assert claim.exact_recurrence_claimed is False
    assert claim.readout_error_only is True
    assert "does not generally commute" in claim.caveat


def test_sketch_recurrence_claims_are_public_api() -> None:
    claim = fhm3.classify_sketch_recurrence_claim(
        recurrence_type="scalar",
        recurrence_compat_available=True,
        recurrence_compat_max_abs_error=0.0,
    )

    assert isinstance(claim, fhm3.SketchRecurrenceClaim)
