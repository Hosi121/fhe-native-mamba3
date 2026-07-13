import pytest
from fhemamba.gated_norm_sweep import parse_candidate


def test_parse_candidate() -> None:
    assert parse_candidate("31:4") == (31, 4)


@pytest.mark.parametrize("spec", ["31", "0:4", "31:0", "x:4"])
def test_parse_candidate_rejects_invalid_specs(spec: str) -> None:
    with pytest.raises((ValueError, TypeError)):
        parse_candidate(spec)
