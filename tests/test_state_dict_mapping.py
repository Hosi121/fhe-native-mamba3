from __future__ import annotations

import torch

from fhe_native_mamba3.state_dict_mapping import (
    StateDictMappingRule,
    identity_mapping_rules,
    map_state_dict,
)


def test_identity_mapping_rules_follow_shared_keys() -> None:
    source = {"b": torch.zeros(2), "a": torch.ones(1), "extra": torch.zeros(1)}
    target = {"a": torch.zeros(1), "b": torch.zeros(2), "missing": torch.zeros(3)}

    rules = identity_mapping_rules(source, target)

    assert rules == (
        StateDictMappingRule(source="a", target="a"),
        StateDictMappingRule(source="b", target="b"),
    )


def test_map_state_dict_reports_missing_unexpected_and_shape_mismatch() -> None:
    source = {
        "a": torch.ones(2, dtype=torch.float64),
        "bad": torch.zeros(3),
        "extra": torch.zeros(1),
    }
    target = {
        "a": torch.zeros(2, dtype=torch.float32),
        "bad": torch.zeros(2),
        "unfilled": torch.zeros(1),
    }
    rules = (
        StateDictMappingRule(source="a", target="a"),
        StateDictMappingRule(source="bad", target="bad"),
        StateDictMappingRule(source="missing", target="unfilled"),
    )

    mapped, report = map_state_dict(source, target, rules)

    assert torch.equal(mapped["a"], torch.ones(2))
    assert mapped["a"].dtype == torch.float32
    assert report.mapped_count == 1
    assert report.dtype_cast_count == 1
    assert report.shape_mismatch_count == 1
    assert report.missing_source_keys == ("missing",)
    assert report.unexpected_source_keys == ("extra",)
    assert report.is_complete is False
