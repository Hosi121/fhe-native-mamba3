"""Claim scoping helpers for sketched SSM recurrence compatibility."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class SketchRecurrenceClaim:
    """Machine-readable scope for a sketch recurrence-compatibility claim."""

    recurrence_type: str
    compatibility_status: str
    exact_recurrence_claimed: bool
    approximate_recurrence_claimed: bool
    readout_error_only: bool
    recurrence_compat_available: bool
    recurrence_compat_max_abs_error: float | None
    tolerance: float
    caveat: str

    def to_json_dict(self) -> dict[str, Any]:
        return asdict(self)


def classify_sketch_recurrence_claim(
    *,
    recurrence_type: str,
    recurrence_compat_available: bool,
    recurrence_compat_max_abs_error: float | None,
    tolerance: float = 1e-9,
) -> SketchRecurrenceClaim:
    """Classify whether an SRHT-sketched recurrence claim is exact, approximate, or absent."""

    normalized = recurrence_type.strip().lower().replace("_", "-")
    if normalized in {"scalar", "rank-scalar"} and recurrence_compat_available:
        error = recurrence_compat_max_abs_error
        exact = error is not None and error <= tolerance
        return SketchRecurrenceClaim(
            recurrence_type=normalized,
            compatibility_status="exact" if exact else "approximate",
            exact_recurrence_claimed=exact,
            approximate_recurrence_claimed=not exact,
            readout_error_only=False,
            recurrence_compat_available=True,
            recurrence_compat_max_abs_error=error,
            tolerance=tolerance,
            caveat=(
                "Scalar decay commutes with the linear SRHT sketch; recurrence "
                "compatibility is exact within tolerance."
                if exact
                else "Scalar decay path was measured but exceeded the exactness tolerance."
            ),
        )
    if normalized == "rank-state":
        return SketchRecurrenceClaim(
            recurrence_type=normalized,
            compatibility_status="unavailable",
            exact_recurrence_claimed=False,
            approximate_recurrence_claimed=False,
            readout_error_only=True,
            recurrence_compat_available=False,
            recurrence_compat_max_abs_error=recurrence_compat_max_abs_error,
            tolerance=tolerance,
            caveat=(
                "Rank-state decay does not generally commute with an SRHT sketch; "
                "rows are readout/trajectory sketch-error evidence only."
            ),
        )
    return SketchRecurrenceClaim(
        recurrence_type=normalized or "unknown",
        compatibility_status="unavailable",
        exact_recurrence_claimed=False,
        approximate_recurrence_claimed=False,
        readout_error_only=True,
        recurrence_compat_available=recurrence_compat_available,
        recurrence_compat_max_abs_error=recurrence_compat_max_abs_error,
        tolerance=tolerance,
        caveat="Recurrence type is unknown or unsupported for exact sketched recurrence claims.",
    )


__all__ = [
    "SketchRecurrenceClaim",
    "classify_sketch_recurrence_claim",
]
