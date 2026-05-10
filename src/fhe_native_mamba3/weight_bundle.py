"""Weight bundle manifest for OSS checkpoint import/export work."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch

from fhe_native_mamba3.model import FheMamba3Config, FheMamba3ForCausalLM
from fhe_native_mamba3.state_dict_mapping import (
    StateDictMappingReport,
    StateDictMappingRule,
    map_state_dict,
)
from fhe_native_mamba3.weight_encoding import (
    WeightCalibration,
    WeightEncodingConfig,
    calibrate_weight_tensor,
)

WEIGHT_BUNDLE_FORMAT_VERSION = "fhe-native-mamba3.weight-bundle.v1"
MANIFEST_NAME = "manifest.json"
WEIGHTS_NAME = "weights.pt"


@dataclass(frozen=True)
class TensorManifest:
    """Metadata for one tensor in a weight bundle."""

    name: str
    shape: tuple[int, ...]
    dtype: str
    calibration: WeightCalibration

    @property
    def value_count(self) -> int:
        return self.calibration.value_count

    def to_json_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["shape"] = list(self.shape)
        payload["calibration"] = self.calibration.to_json_dict()
        return payload

    @classmethod
    def from_json_dict(cls, payload: dict[str, Any]) -> TensorManifest:
        return cls(
            name=str(payload["name"]),
            shape=tuple(int(value) for value in payload["shape"]),
            dtype=str(payload["dtype"]),
            calibration=WeightCalibration(**payload["calibration"]),
        )


@dataclass(frozen=True)
class WeightBundleManifest:
    """JSON manifest stored next to a fp32 PyTorch state_dict."""

    format_version: str
    model_config: dict[str, Any]
    tensor_count: int
    parameter_count: int
    tensors: tuple[TensorManifest, ...]
    weights_file: str = WEIGHTS_NAME

    def to_json_dict(self) -> dict[str, Any]:
        return {
            "format_version": self.format_version,
            "model_config": self.model_config,
            "tensor_count": self.tensor_count,
            "parameter_count": self.parameter_count,
            "weights_file": self.weights_file,
            "tensors": [tensor.to_json_dict() for tensor in self.tensors],
        }

    @classmethod
    def from_json_dict(cls, payload: dict[str, Any]) -> WeightBundleManifest:
        return cls(
            format_version=str(payload["format_version"]),
            model_config=dict(payload["model_config"]),
            tensor_count=int(payload["tensor_count"]),
            parameter_count=int(payload["parameter_count"]),
            weights_file=str(payload.get("weights_file", WEIGHTS_NAME)),
            tensors=tuple(TensorManifest.from_json_dict(item) for item in payload["tensors"]),
        )


def build_weight_bundle_manifest(
    model: FheMamba3ForCausalLM,
    encoding_config: WeightEncodingConfig = WeightEncodingConfig(),
) -> WeightBundleManifest:
    """Build calibration metadata for a model state dict."""

    tensors = tuple(
        TensorManifest(
            name=name,
            shape=tuple(int(dim) for dim in tensor.shape),
            dtype="float32"
            if tensor.is_floating_point()
            else str(tensor.dtype).removeprefix("torch."),
            calibration=calibrate_weight_tensor(tensor, encoding_config),
        )
        for name, tensor in sorted(model.state_dict().items())
    )
    return WeightBundleManifest(
        format_version=WEIGHT_BUNDLE_FORMAT_VERSION,
        model_config=asdict(model.config),
        tensor_count=len(tensors),
        parameter_count=sum(tensor.value_count for tensor in tensors),
        tensors=tensors,
    )


def save_weight_bundle(
    model: FheMamba3ForCausalLM,
    output_dir: str | Path,
    encoding_config: WeightEncodingConfig = WeightEncodingConfig(),
) -> WeightBundleManifest:
    """Save a fp32 state_dict plus JSON calibration manifest."""

    bundle_dir = Path(output_dir)
    bundle_dir.mkdir(parents=True, exist_ok=True)
    manifest = build_weight_bundle_manifest(model, encoding_config)
    state_dict = {
        name: _master_weight_tensor(tensor) for name, tensor in sorted(model.state_dict().items())
    }
    torch.save(state_dict, bundle_dir / manifest.weights_file)
    (bundle_dir / MANIFEST_NAME).write_text(
        json.dumps(manifest.to_json_dict(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest


def load_weight_bundle_manifest(bundle_dir: str | Path) -> WeightBundleManifest:
    """Load the JSON manifest for a weight bundle."""

    payload = json.loads((Path(bundle_dir) / MANIFEST_NAME).read_text(encoding="utf-8"))
    manifest = WeightBundleManifest.from_json_dict(payload)
    if manifest.format_version != WEIGHT_BUNDLE_FORMAT_VERSION:
        msg = f"unsupported weight bundle format: {manifest.format_version}"
        raise ValueError(msg)
    return manifest


def load_weight_bundle_model(
    bundle_dir: str | Path,
    *,
    map_location: str | torch.device = "cpu",
) -> tuple[FheMamba3ForCausalLM, WeightBundleManifest]:
    """Reconstruct a model from a saved weight bundle."""

    manifest = load_weight_bundle_manifest(bundle_dir)
    config = FheMamba3Config(**manifest.model_config)
    model = FheMamba3ForCausalLM(config)
    state_dict = _load_state_dict(Path(bundle_dir) / manifest.weights_file, map_location)
    state_dict = _migrate_state_dict_for_model(state_dict, model)
    model.load_state_dict(state_dict)
    return model, manifest


def save_weight_bundle_from_checkpoint(
    checkpoint_path: str | Path,
    output_dir: str | Path,
    encoding_config: WeightEncodingConfig = WeightEncodingConfig(),
    *,
    map_location: str | torch.device = "cpu",
) -> WeightBundleManifest:
    """Convert a training checkpoint into a fp32 weight bundle."""

    checkpoint = _load_training_checkpoint(Path(checkpoint_path), map_location)
    config = FheMamba3Config(**checkpoint["config"])
    model = FheMamba3ForCausalLM(config)
    state_dict = checkpoint["model"]
    if not isinstance(state_dict, dict):
        msg = "checkpoint['model'] must be a state_dict"
        raise ValueError(msg)
    state_dict = _migrate_state_dict_for_model(state_dict, model)
    model.load_state_dict(state_dict)
    return save_weight_bundle(model, output_dir, encoding_config)


def save_weight_bundle_from_mapped_checkpoint(
    checkpoint_state_dict: dict[str, torch.Tensor],
    output_dir: str | Path,
    *,
    config: FheMamba3Config,
    rules: tuple[StateDictMappingRule, ...],
    encoding_config: WeightEncodingConfig = WeightEncodingConfig(),
    allow_partial: bool = False,
) -> tuple[WeightBundleManifest, StateDictMappingReport]:
    """Map an external state_dict into the prototype model and save a bundle."""

    model = FheMamba3ForCausalLM(config)
    mapped_state_dict, report = map_state_dict(checkpoint_state_dict, model.state_dict(), rules)
    if not allow_partial and not report.is_complete:
        msg = "state_dict mapping is incomplete; inspect mapping_report before bundling"
        raise ValueError(msg)
    model.load_state_dict(mapped_state_dict)
    manifest = save_weight_bundle(model, output_dir, encoding_config)
    return manifest, report


def _master_weight_tensor(tensor: torch.Tensor) -> torch.Tensor:
    detached = tensor.detach().cpu()
    if detached.is_floating_point():
        return detached.to(dtype=torch.float32).clone()
    return detached.clone()


def _load_state_dict(path: Path, map_location: str | torch.device) -> dict[str, torch.Tensor]:
    try:
        state_dict = torch.load(path, map_location=map_location, weights_only=True)
    except TypeError:  # pragma: no cover - compatibility with older torch.
        state_dict = torch.load(path, map_location=map_location)
    if not isinstance(state_dict, dict):
        msg = "weight bundle must contain a state_dict"
        raise ValueError(msg)
    return state_dict


def _migrate_state_dict_for_model(
    state_dict: dict[str, torch.Tensor],
    model: FheMamba3ForCausalLM,
) -> dict[str, torch.Tensor]:
    """Fill additive-compatible parameters absent from older local bundles."""

    model_state = model.state_dict()
    migrated = dict(state_dict)
    for name, tensor in model_state.items():
        if name.endswith(".d_skip") and name not in migrated:
            migrated[name] = tensor.detach().clone()
    return migrated


def _load_training_checkpoint(path: Path, map_location: str | torch.device) -> dict[str, Any]:
    try:
        checkpoint = torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:  # pragma: no cover - compatibility with older torch.
        checkpoint = torch.load(path, map_location=map_location)
    if not isinstance(checkpoint, dict):
        msg = "checkpoint must be a dictionary"
        raise ValueError(msg)
    missing = {"config", "model"} - set(checkpoint)
    if missing:
        msg = f"checkpoint is missing required keys: {sorted(missing)}"
        raise ValueError(msg)
    return checkpoint
