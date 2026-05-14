"""Helpers for the Stage 1 FIDESlib state-major rotation probe."""

from __future__ import annotations

import json
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class Stage1FideslibRotationProbeConfig:
    """Shape and CKKS metadata for the bounded FIDESlib rotation probe."""

    d_model: int = 768
    d_model_pad: int = 1024
    d_state: int = 16
    mimo_rank: int = 1536
    rank_pad: int = 2048
    model_baby_step: int = 64
    rank_baby_step: int = 64
    pre_recurrence_mode: str = "rank-gate-bc-decay-bsgs-poly"
    layer_index: int = 0
    ring_dimension: int = 131072
    num_slots: int = 32768
    multiplicative_depth: int = 64
    scaling_mod_size: int = 30

    def to_json_dict(self) -> dict[str, Any]:
        return asdict(self)


def normalize_rotation_inventory(rotations: Iterable[int]) -> tuple[int, ...]:
    """Return the nonzero sorted unique rotation inventory."""

    normalized = tuple(sorted({int(rotation) for rotation in rotations if int(rotation) != 0}))
    if not normalized:
        msg = "rotation inventory must contain at least one nonzero rotation"
        raise ValueError(msg)
    return normalized


def rotations_to_csv(rotations: Iterable[int]) -> str:
    """Serialize a rotation inventory for the native probe CLI."""

    return ",".join(str(rotation) for rotation in normalize_rotation_inventory(rotations))


def load_rotation_inventory_from_artifact(path: str | Path) -> tuple[int, ...]:
    """Load required application rotations from a previous Stage 1 artifact."""

    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        msg = "rotation artifact must be a JSON object"
        raise ValueError(msg)
    rotations = payload.get("required_application_rotations")
    if rotations is None and isinstance(payload.get("measurements"), dict):
        rotations = payload["measurements"].get("required_application_rotations")
    if rotations is None:
        msg = "rotation artifact does not include required_application_rotations"
        raise ValueError(msg)
    if not isinstance(rotations, list | tuple):
        msg = "required_application_rotations must be a JSON array"
        raise ValueError(msg)
    return normalize_rotation_inventory(rotations)


def build_checkpoint_rotation_inventory(
    checkpoint: str | Path,
    *,
    state_dict_key: str | None = None,
    config: Stage1FideslibRotationProbeConfig | None = None,
) -> tuple[int, ...]:
    """Build the checkpoint state-major rotation inventory from source weights."""

    from fhe_native_mamba3.checkpoint import load_checkpoint_state_dict
    from fhe_native_mamba3.mamba_checkpoint import plan_mamba_checkpoint
    from fhe_native_mamba3.stage1_state_major_checkpoint import (
        StateMajorFullShapeConfig,
        required_state_major_checkpoint_layer_rotations,
    )

    resolved_config = config or Stage1FideslibRotationProbeConfig()
    state_dict, _ = load_checkpoint_state_dict(
        checkpoint,
        state_dict_key=state_dict_key,
    )
    checkpoint_plan = plan_mamba_checkpoint(state_dict)
    layer = checkpoint_plan.layers[resolved_config.layer_index]
    shape_config = StateMajorFullShapeConfig(
        d_model=resolved_config.d_model,
        d_model_pad=resolved_config.d_model_pad,
        mimo_rank=resolved_config.mimo_rank,
        rank_pad=resolved_config.rank_pad,
        d_state=resolved_config.d_state,
        model_baby_step=resolved_config.model_baby_step,
        rank_baby_step=resolved_config.rank_baby_step,
    )
    return normalize_rotation_inventory(
        required_state_major_checkpoint_layer_rotations(
            shape_config,
            pre_recurrence_mode=resolved_config.pre_recurrence_mode,
            dt_rank=layer.inferred_dt_rank,
        )
    )


__all__ = [
    "Stage1FideslibRotationProbeConfig",
    "build_checkpoint_rotation_inventory",
    "load_rotation_inventory_from_artifact",
    "normalize_rotation_inventory",
    "rotations_to_csv",
]
