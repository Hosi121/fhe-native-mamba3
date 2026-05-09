"""Generic PyTorch checkpoint inspection utilities."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch


@dataclass(frozen=True)
class CheckpointTensorSpec:
    """Shape metadata for one tensor in a checkpoint."""

    name: str
    shape: tuple[int, ...]
    dtype: str
    value_count: int

    def to_json_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["shape"] = list(self.shape)
        return payload


@dataclass(frozen=True)
class CheckpointInspection:
    """Summary of a PyTorch checkpoint state dict."""

    checkpoint: str
    state_dict_key: str
    tensor_count: int
    parameter_count: int
    tensors: tuple[CheckpointTensorSpec, ...]
    top_level_keys: tuple[str, ...]

    def to_json_dict(self, *, max_tensors: int | None = None) -> dict[str, Any]:
        tensors = self.tensors[:max_tensors] if max_tensors is not None else self.tensors
        return {
            "checkpoint": self.checkpoint,
            "state_dict_key": self.state_dict_key,
            "tensor_count": self.tensor_count,
            "parameter_count": self.parameter_count,
            "top_level_keys": list(self.top_level_keys),
            "tensors": [tensor.to_json_dict() for tensor in tensors],
        }


def inspect_checkpoint(
    checkpoint_path: str | Path,
    *,
    state_dict_key: str | None = None,
    map_location: str | torch.device = "cpu",
) -> CheckpointInspection:
    """Inspect a generic PyTorch checkpoint without model-specific assumptions."""

    checkpoint = _load_checkpoint(Path(checkpoint_path), map_location)
    state_dict, resolved_key = _extract_state_dict(checkpoint, state_dict_key)
    specs = tuple(
        CheckpointTensorSpec(
            name=name,
            shape=tuple(int(dim) for dim in tensor.shape),
            dtype=str(tensor.dtype).removeprefix("torch."),
            value_count=int(tensor.numel()),
        )
        for name, tensor in sorted(state_dict.items())
    )
    return CheckpointInspection(
        checkpoint=str(checkpoint_path),
        state_dict_key=resolved_key,
        tensor_count=len(specs),
        parameter_count=sum(spec.value_count for spec in specs),
        tensors=specs,
        top_level_keys=tuple(sorted(str(key) for key in checkpoint))
        if isinstance(checkpoint, dict)
        else (),
    )


def load_checkpoint_state_dict(
    checkpoint_path: str | Path,
    *,
    state_dict_key: str | None = None,
    map_location: str | torch.device = "cpu",
) -> tuple[dict[str, torch.Tensor], str]:
    """Load the tensor state_dict selected by the same rules as inspection."""

    checkpoint = _load_checkpoint(Path(checkpoint_path), map_location)
    return _extract_state_dict(checkpoint, state_dict_key)


def _load_checkpoint(path: Path, map_location: str | torch.device) -> Any:
    try:
        return torch.load(path, map_location=map_location, weights_only=True)
    except TypeError:  # pragma: no cover - compatibility with older torch.
        return torch.load(path, map_location=map_location)


def _extract_state_dict(
    checkpoint: Any,
    state_dict_key: str | None,
) -> tuple[dict[str, torch.Tensor], str]:
    if state_dict_key is not None:
        if not isinstance(checkpoint, dict) or state_dict_key not in checkpoint:
            msg = f"checkpoint does not contain state_dict_key={state_dict_key!r}"
            raise ValueError(msg)
        return _require_tensor_dict(checkpoint[state_dict_key]), state_dict_key

    if isinstance(checkpoint, dict):
        if _is_tensor_dict(checkpoint):
            return checkpoint, "<root>"
        for key in ("state_dict", "model", "module"):
            if key in checkpoint and _is_tensor_dict(checkpoint[key]):
                return checkpoint[key], key
    msg = "could not infer a tensor state_dict; pass --state-dict-key"
    raise ValueError(msg)


def _is_tensor_dict(value: Any) -> bool:
    return (
        isinstance(value, dict)
        and bool(value)
        and all(isinstance(item, torch.Tensor) for item in value.values())
    )


def _require_tensor_dict(value: Any) -> dict[str, torch.Tensor]:
    if not _is_tensor_dict(value):
        msg = "selected state_dict is empty or contains non-tensor values"
        raise ValueError(msg)
    return value
