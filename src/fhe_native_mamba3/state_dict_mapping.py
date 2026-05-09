"""State-dict mapping helpers for checkpoint adapters."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch


@dataclass(frozen=True)
class StateDictMappingRule:
    """Map one source tensor key to one target tensor key."""

    source: str
    target: str

    def to_json_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_json_dict(cls, payload: dict[str, Any]) -> StateDictMappingRule:
        return cls(source=str(payload["source"]), target=str(payload["target"]))


@dataclass(frozen=True)
class TensorMappingStatus:
    """Result for one attempted tensor mapping."""

    source: str
    target: str
    status: str
    source_shape: tuple[int, ...] | None
    target_shape: tuple[int, ...] | None
    dtype: str | None
    message: str

    def to_json_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["source_shape"] = list(self.source_shape) if self.source_shape is not None else None
        payload["target_shape"] = list(self.target_shape) if self.target_shape is not None else None
        return payload


@dataclass(frozen=True)
class StateDictMappingReport:
    """Summary of a source-to-target state-dict mapping attempt."""

    rule_count: int
    mapped_count: int
    missing_source_keys: tuple[str, ...]
    missing_target_keys: tuple[str, ...]
    unexpected_source_keys: tuple[str, ...]
    shape_mismatch_count: int
    dtype_cast_count: int
    statuses: tuple[TensorMappingStatus, ...]

    @property
    def is_complete(self) -> bool:
        return (
            self.mapped_count == self.rule_count
            and not self.missing_source_keys
            and not self.missing_target_keys
            and self.shape_mismatch_count == 0
        )

    def to_json_dict(self, *, max_statuses: int | None = None) -> dict[str, Any]:
        statuses = self.statuses[:max_statuses] if max_statuses is not None else self.statuses
        return {
            "rule_count": self.rule_count,
            "mapped_count": self.mapped_count,
            "missing_source_keys": list(self.missing_source_keys),
            "missing_target_keys": list(self.missing_target_keys),
            "unexpected_source_keys": list(self.unexpected_source_keys),
            "shape_mismatch_count": self.shape_mismatch_count,
            "dtype_cast_count": self.dtype_cast_count,
            "is_complete": self.is_complete,
            "statuses": [status.to_json_dict() for status in statuses],
        }


def identity_mapping_rules(
    source_state_dict: dict[str, torch.Tensor],
    target_state_dict: dict[str, torch.Tensor],
) -> tuple[StateDictMappingRule, ...]:
    """Create exact-name mapping rules for keys present on both sides."""

    return tuple(
        StateDictMappingRule(source=key, target=key)
        for key in sorted(set(source_state_dict) & set(target_state_dict))
    )


def load_mapping_rules(path: str | Path) -> tuple[StateDictMappingRule, ...]:
    """Load mapping rules from a JSON file."""

    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    rules_payload = payload["rules"] if isinstance(payload, dict) else payload
    if not isinstance(rules_payload, list):
        msg = "mapping rules JSON must be a list or an object with a 'rules' list"
        raise ValueError(msg)
    return tuple(StateDictMappingRule.from_json_dict(item) for item in rules_payload)


def save_mapping_rules(path: str | Path, rules: tuple[StateDictMappingRule, ...]) -> None:
    """Persist mapping rules as stable JSON."""

    payload = {"rules": [rule.to_json_dict() for rule in rules]}
    Path(path).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def map_state_dict(
    source_state_dict: dict[str, torch.Tensor],
    target_state_dict: dict[str, torch.Tensor],
    rules: tuple[StateDictMappingRule, ...],
) -> tuple[dict[str, torch.Tensor], StateDictMappingReport]:
    """Apply mapping rules into a target-shaped state dict."""

    mapped = {name: tensor.detach().clone() for name, tensor in target_state_dict.items()}
    used_source_keys: set[str] = set()
    used_target_keys: set[str] = set()
    statuses: list[TensorMappingStatus] = []
    mapped_count = 0
    dtype_cast_count = 0
    shape_mismatch_count = 0

    for rule in rules:
        used_source_keys.add(rule.source)
        used_target_keys.add(rule.target)
        if rule.source not in source_state_dict:
            statuses.append(
                TensorMappingStatus(
                    source=rule.source,
                    target=rule.target,
                    status="missing_source",
                    source_shape=None,
                    target_shape=_shape_or_none(target_state_dict.get(rule.target)),
                    dtype=None,
                    message="source key is absent",
                )
            )
            continue
        if rule.target not in target_state_dict:
            statuses.append(
                TensorMappingStatus(
                    source=rule.source,
                    target=rule.target,
                    status="missing_target",
                    source_shape=_shape_or_none(source_state_dict.get(rule.source)),
                    target_shape=None,
                    dtype=str(source_state_dict[rule.source].dtype).removeprefix("torch."),
                    message="target key is absent",
                )
            )
            continue

        source = source_state_dict[rule.source].detach()
        target = target_state_dict[rule.target]
        source_shape = tuple(int(dim) for dim in source.shape)
        target_shape = tuple(int(dim) for dim in target.shape)
        if source_shape != target_shape:
            shape_mismatch_count += 1
            statuses.append(
                TensorMappingStatus(
                    source=rule.source,
                    target=rule.target,
                    status="shape_mismatch",
                    source_shape=source_shape,
                    target_shape=target_shape,
                    dtype=str(source.dtype).removeprefix("torch."),
                    message="source and target shapes differ",
                )
            )
            continue

        casted = source.to(dtype=target.dtype, device=target.device).clone()
        if source.dtype != target.dtype:
            dtype_cast_count += 1
        mapped[rule.target] = casted
        mapped_count += 1
        statuses.append(
            TensorMappingStatus(
                source=rule.source,
                target=rule.target,
                status="mapped",
                source_shape=source_shape,
                target_shape=target_shape,
                dtype=str(casted.dtype).removeprefix("torch."),
                message="mapped exactly",
            )
        )

    missing_source_keys = tuple(
        sorted(rule.source for rule in rules if rule.source not in source_state_dict)
    )
    missing_target_keys = tuple(sorted(set(target_state_dict) - used_target_keys))
    unexpected_source_keys = tuple(sorted(set(source_state_dict) - used_source_keys))
    report = StateDictMappingReport(
        rule_count=len(rules),
        mapped_count=mapped_count,
        missing_source_keys=missing_source_keys,
        missing_target_keys=missing_target_keys,
        unexpected_source_keys=unexpected_source_keys,
        shape_mismatch_count=shape_mismatch_count,
        dtype_cast_count=dtype_cast_count,
        statuses=tuple(statuses),
    )
    return mapped, report


def _shape_or_none(tensor: torch.Tensor | None) -> tuple[int, ...] | None:
    if tensor is None:
        return None
    return tuple(int(dim) for dim in tensor.shape)
