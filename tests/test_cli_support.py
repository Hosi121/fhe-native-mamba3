from __future__ import annotations

import pytest

from fhe_native_mamba3.cli_support import parse_float_list, parse_int_list


def test_parse_int_list_skips_empty_parts() -> None:
    assert parse_int_list("1,2,,4,") == (1, 2, 4)


def test_parse_int_list_propagates_invalid_values() -> None:
    with pytest.raises(ValueError, match="invalid literal"):
        parse_int_list("1,nope")


def test_parse_float_list_skips_empty_parts() -> None:
    assert parse_float_list("0.5,,1.25,") == (0.5, 1.25)


def test_parse_float_list_propagates_invalid_values() -> None:
    with pytest.raises(ValueError, match="could not convert"):
        parse_float_list("0.5,nope")
