from __future__ import annotations

import torch

from fhe_native_mamba3.state_dict_mapping import (
    StateDictMappingRule,
    draft_mapping_rules,
    identity_mapping_rules,
    load_mapping_rules,
    map_state_dict,
    save_mapping_draft,
    save_mapping_rules,
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


def test_mapping_rules_round_trip_json(tmp_path) -> None:
    rules = (
        StateDictMappingRule(source="external.a", target="internal.a"),
        StateDictMappingRule(source="external.b", target="internal.b"),
    )
    path = tmp_path / "rules.json"

    save_mapping_rules(path, rules)

    assert load_mapping_rules(path) == rules


def test_draft_mapping_rules_uses_only_exact_and_unique_shape_matches() -> None:
    source = {
        "ambiguous.a": torch.zeros(4),
        "ambiguous.b": torch.zeros(4),
        "external.unique": torch.zeros(3),
        "same": torch.zeros(2),
        "unused": torch.zeros(5),
    }
    target = {
        "same": torch.zeros(2),
        "target.ambiguous": torch.zeros(4),
        "target.missing": torch.zeros(6),
        "target.unique": torch.zeros(3),
    }

    draft = draft_mapping_rules(source, target)

    assert draft.rules == (
        StateDictMappingRule(source="same", target="same"),
        StateDictMappingRule(source="external.unique", target="target.unique"),
    )
    statuses = {entry.target: entry.status for entry in draft.entries}
    assert statuses == {
        "same": "exact",
        "target.ambiguous": "ambiguous_shape",
        "target.missing": "unmatched",
        "target.unique": "unique_shape",
    }
    assert draft.unused_source_keys == ("ambiguous.a", "ambiguous.b", "unused")


def test_mapping_draft_json_can_be_reused_as_rules_json(tmp_path) -> None:
    source = {"external.a": torch.zeros(2)}
    target = {"internal.a": torch.zeros(2)}
    path = tmp_path / "draft.json"

    save_mapping_draft(path, draft_mapping_rules(source, target))

    assert load_mapping_rules(path) == (
        StateDictMappingRule(source="external.a", target="internal.a"),
    )
