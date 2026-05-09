"""Adapters from common Mamba-family checkpoints into prototype bundles."""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch

from fhe_native_mamba3.model import FheMamba3Config, FheMamba3ForCausalLM
from fhe_native_mamba3.weight_bundle import WeightBundleManifest, save_weight_bundle
from fhe_native_mamba3.weight_encoding import WeightEncodingConfig

_LAYER_RE = re.compile(r"(?:^|\.)(?:backbone\.)?layers\.(\d+)\.")


@dataclass(frozen=True)
class AdapterTensorStatus:
    """Status for one tensor adapted from an external checkpoint."""

    target: str
    source: str | None
    status: str
    target_shape: tuple[int, ...]
    source_shape: tuple[int, ...] | None
    message: str

    def to_json_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["target_shape"] = list(self.target_shape)
        payload["source_shape"] = list(self.source_shape) if self.source_shape is not None else None
        return payload


@dataclass(frozen=True)
class MambaCheckpointAdapterReport:
    """Summary of a Mamba-family checkpoint adaptation."""

    source_format: str
    inferred_layers: int
    adapted_layers: int
    statuses: tuple[AdapterTensorStatus, ...]

    @property
    def adapted_count(self) -> int:
        return sum(1 for status in self.statuses if status.status == "adapted")

    @property
    def initialized_count(self) -> int:
        return sum(1 for status in self.statuses if status.status == "initialized")

    @property
    def skipped_count(self) -> int:
        return sum(1 for status in self.statuses if status.status == "skipped")

    def to_json_dict(self, *, max_statuses: int | None = None) -> dict[str, Any]:
        statuses = self.statuses[:max_statuses] if max_statuses is not None else self.statuses
        return {
            "source_format": self.source_format,
            "inferred_layers": self.inferred_layers,
            "adapted_layers": self.adapted_layers,
            "adapted_count": self.adapted_count,
            "initialized_count": self.initialized_count,
            "skipped_count": self.skipped_count,
            "statuses": [status.to_json_dict() for status in statuses],
        }


@dataclass(frozen=True)
class MambaLayerPlan:
    """Detected source tensor layout for one Mamba-family layer."""

    layer_index: int
    prefix: str
    norm_key: str | None
    in_proj_key: str | None
    x_proj_key: str | None
    dt_proj_weight_key: str | None
    dt_proj_bias_key: str | None
    out_proj_key: str | None
    d_key: str | None
    conv1d_weight_key: str | None
    conv1d_bias_key: str | None
    a_log_key: str | None
    source_inner_dim: int | None
    source_d_state: int | None
    inferred_dt_rank: int | None

    def to_json_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class MambaCheckpointPlan:
    """Read-only diagnosis of a Mamba-family checkpoint layout."""

    source_format: str
    embedding_key: str | None
    lm_head_key: str | None
    final_norm_key: str | None
    vocab_size: int | None
    d_model: int | None
    inferred_layers: int
    layers: tuple[MambaLayerPlan, ...]

    @property
    def complete_layer_count(self) -> int:
        return sum(
            1
            for layer in self.layers
            if layer.in_proj_key is not None
            and layer.x_proj_key is not None
            and layer.a_log_key is not None
        )

    @property
    def inferred_d_state(self) -> int | None:
        for layer in self.layers:
            if layer.source_d_state is not None:
                return layer.source_d_state
        return None

    @property
    def inferred_mimo_rank(self) -> int | None:
        for layer in self.layers:
            if layer.source_inner_dim is not None:
                return layer.source_inner_dim
        return None

    def to_json_dict(self, *, max_layers: int | None = None) -> dict[str, Any]:
        layers = self.layers[:max_layers] if max_layers is not None else self.layers
        return {
            "source_format": self.source_format,
            "embedding_key": self.embedding_key,
            "lm_head_key": self.lm_head_key,
            "final_norm_key": self.final_norm_key,
            "vocab_size": self.vocab_size,
            "d_model": self.d_model,
            "inferred_layers": self.inferred_layers,
            "complete_layer_count": self.complete_layer_count,
            "inferred_d_state": self.inferred_d_state,
            "inferred_mimo_rank": self.inferred_mimo_rank,
            "layers": [layer.to_json_dict() for layer in layers],
        }


def plan_mamba_checkpoint(
    source_state_dict: dict[str, torch.Tensor],
) -> MambaCheckpointPlan:
    """Inspect Mamba-family checkpoint keys without constructing a model."""

    embedding_key = _find_embedding_key(source_state_dict)
    embedding = (
        _require_matrix(source_state_dict[embedding_key], embedding_key)
        if embedding_key is not None
        else None
    )
    lm_head_key = _find_lm_head_key(source_state_dict)
    final_norm_key = _find_first_key(
        source_state_dict,
        ("backbone.norm_f.weight", "norm_f.weight", "norm.weight"),
    )
    inferred_layers = _infer_layer_count(source_state_dict)
    layers = tuple(
        _plan_layer(source_state_dict, layer_index=layer_index)
        for layer_index in range(inferred_layers)
    )
    return MambaCheckpointPlan(
        source_format="mamba-family-state-dict",
        embedding_key=embedding_key,
        lm_head_key=lm_head_key,
        final_norm_key=final_norm_key,
        vocab_size=int(embedding.shape[0]) if embedding is not None else None,
        d_model=int(embedding.shape[1]) if embedding is not None else None,
        inferred_layers=inferred_layers,
        layers=layers,
    )


def save_mamba_checkpoint_bundle(
    source_state_dict: dict[str, torch.Tensor],
    output_dir: str | Path,
    *,
    d_state: int,
    mimo_rank: int,
    n_layers: int | None = None,
    max_seq_len: int = 256,
    seed: int = 0,
    encoding_config: WeightEncodingConfig = WeightEncodingConfig(),
) -> tuple[WeightBundleManifest, MambaCheckpointAdapterReport]:
    """Adapt a common Mamba-family state dict into a prototype weight bundle."""

    model, report = adapt_mamba_state_dict_to_model(
        source_state_dict,
        d_state=d_state,
        mimo_rank=mimo_rank,
        n_layers=n_layers,
        max_seq_len=max_seq_len,
        seed=seed,
    )
    manifest = save_weight_bundle(model, output_dir, encoding_config)
    return manifest, report


def adapt_mamba_state_dict_to_model(
    source_state_dict: dict[str, torch.Tensor],
    *,
    d_state: int,
    mimo_rank: int,
    n_layers: int | None = None,
    max_seq_len: int = 256,
    seed: int = 0,
) -> tuple[FheMamba3ForCausalLM, MambaCheckpointAdapterReport]:
    """Create a prototype model initialized from Mamba-family checkpoint tensors."""

    if d_state <= 0 or mimo_rank <= 0:
        msg = "d_state and mimo_rank must be positive"
        raise ValueError(msg)
    embedding_key = _find_embedding_key(source_state_dict)
    if embedding_key is None:
        msg = "could not find a Mamba-family embedding weight"
        raise ValueError(msg)
    embedding = _require_matrix(source_state_dict[embedding_key], embedding_key)
    vocab_size, d_model = (int(embedding.shape[0]), int(embedding.shape[1]))
    inferred_layers = _infer_layer_count(source_state_dict)
    adapted_layers = n_layers if n_layers is not None else max(1, inferred_layers)
    if adapted_layers <= 0:
        msg = "n_layers must be positive when provided"
        raise ValueError(msg)

    generator_state = torch.random.get_rng_state()
    torch.manual_seed(seed)
    try:
        config = FheMamba3Config(
            vocab_size=vocab_size,
            d_model=d_model,
            n_layers=adapted_layers,
            d_state=d_state,
            mimo_rank=mimo_rank,
            max_seq_len=max_seq_len,
            bc_mode="static",
            decay_mode="scalar",
            scan_mode="sequential",
        )
        model = FheMamba3ForCausalLM(config)
    finally:
        torch.random.set_rng_state(generator_state)

    statuses: list[AdapterTensorStatus] = []
    with torch.no_grad():
        model.pos.zero_()
        _copy_exact_or_fit(
            model.embed.weight,
            embedding,
            target="embed.weight",
            source=embedding_key,
            statuses=statuses,
        )
        _adapt_lm_head(
            model,
            source_state_dict,
            embedding_key=embedding_key,
            statuses=statuses,
        )
        norm_key = _find_first_key(
            source_state_dict,
            ("backbone.norm_f.weight", "norm_f.weight", "norm.weight"),
        )
        if norm_key is not None:
            _copy_exact_or_fit(
                model.norm.weight,
                source_state_dict[norm_key],
                target="norm.weight",
                source=norm_key,
                statuses=statuses,
            )

        for layer_index, block in enumerate(model.blocks):
            _adapt_layer(
                source_state_dict,
                layer_index=layer_index,
                block=block,
                statuses=statuses,
            )

    return model, MambaCheckpointAdapterReport(
        source_format="mamba-family-state-dict",
        inferred_layers=inferred_layers,
        adapted_layers=adapted_layers,
        statuses=tuple(statuses),
    )


def _adapt_layer(
    source_state_dict: dict[str, torch.Tensor],
    *,
    layer_index: int,
    block: Any,
    statuses: list[AdapterTensorStatus],
) -> None:
    prefix = _layer_prefix(source_state_dict, layer_index)
    in_proj_key = _find_layer_key(
        source_state_dict, prefix, ("mixer.in_proj.weight", "in_proj.weight")
    )
    x_proj_key = _find_layer_key(
        source_state_dict, prefix, ("mixer.x_proj.weight", "x_proj.weight")
    )
    dt_proj_weight_key = _find_layer_key(
        source_state_dict,
        prefix,
        ("mixer.dt_proj.weight", "dt_proj.weight"),
    )
    dt_proj_bias_key = _find_layer_key(
        source_state_dict,
        prefix,
        ("mixer.dt_proj.bias", "dt_proj.bias"),
    )
    out_proj_key = _find_layer_key(
        source_state_dict,
        prefix,
        ("mixer.out_proj.weight", "out_proj.weight"),
    )
    d_key = _find_layer_key(source_state_dict, prefix, ("mixer.D", "D"))
    conv1d_weight_key = _find_layer_key(
        source_state_dict,
        prefix,
        ("mixer.conv1d.weight", "conv1d.weight"),
    )
    conv1d_bias_key = _find_layer_key(
        source_state_dict,
        prefix,
        ("mixer.conv1d.bias", "conv1d.bias"),
    )
    a_log_key = _find_layer_key(source_state_dict, prefix, ("mixer.A_log", "A_log"))
    norm_key = _find_layer_key(source_state_dict, prefix, ("norm.weight",))

    if norm_key is not None:
        _copy_exact_or_fit(
            block.in_norm.weight,
            source_state_dict[norm_key],
            target=f"blocks.{layer_index}.in_norm.weight",
            source=norm_key,
            statuses=statuses,
        )
    if in_proj_key is not None:
        _copy_exact_or_fit(
            block.in_rank.weight,
            source_state_dict[in_proj_key],
            target=f"blocks.{layer_index}.in_rank.weight",
            source=in_proj_key,
            statuses=statuses,
        )
    else:
        statuses.append(
            _initialized_status(f"blocks.{layer_index}.in_rank.weight", block.in_rank.weight)
        )
    block.in_rank.bias.zero_()
    statuses.append(_initialized_status(f"blocks.{layer_index}.in_rank.bias", block.in_rank.bias))

    if out_proj_key is not None:
        _copy_exact_or_fit(
            block.out_rank.weight,
            source_state_dict[out_proj_key],
            target=f"blocks.{layer_index}.out_rank.weight",
            source=out_proj_key,
            statuses=statuses,
        )
    else:
        statuses.append(
            _initialized_status(f"blocks.{layer_index}.out_rank.weight", block.out_rank.weight)
        )

    b_source, c_source = _extract_bc_sources(
        source_state_dict,
        x_proj_key=x_proj_key,
        a_log_key=a_log_key,
        d_state=block.config.d_state,
    )
    if b_source is not None:
        _copy_exact_or_fit(
            block.b_static,
            b_source[1],
            target=f"blocks.{layer_index}.b_static",
            source=b_source[0],
            statuses=statuses,
        )
    else:
        statuses.append(_initialized_status(f"blocks.{layer_index}.b_static", block.b_static))
    if c_source is not None:
        _copy_exact_or_fit(
            block.c_static,
            c_source[1],
            target=f"blocks.{layer_index}.c_static",
            source=c_source[0],
            statuses=statuses,
        )
    else:
        statuses.append(_initialized_status(f"blocks.{layer_index}.c_static", block.c_static))

    if a_log_key is not None:
        decay_logits = _decay_logits_from_a_log(
            source_state_dict[a_log_key],
            target_rank=block.config.mimo_rank,
        )
        _copy_exact_or_fit(
            block.decay_logits,
            decay_logits,
            target=f"blocks.{layer_index}.decay_logits",
            source=a_log_key,
            statuses=statuses,
        )
    else:
        block.decay_logits.zero_()
        statuses.append(
            _initialized_status(f"blocks.{layer_index}.decay_logits", block.decay_logits)
        )

    for key, target_name in (
        (dt_proj_weight_key, "dt_proj.weight"),
        (dt_proj_bias_key, "dt_proj.bias"),
        (d_key, "D"),
        (conv1d_weight_key, "conv1d.weight"),
        (conv1d_bias_key, "conv1d.bias"),
    ):
        if key is not None:
            statuses.append(
                _skipped_status(
                    target=f"blocks.{layer_index}.{target_name}",
                    source=key,
                    tensor=source_state_dict[key],
                    message=("not represented in the current FHE-native static recurrence adapter"),
                )
            )


def _extract_bc_sources(
    source_state_dict: dict[str, torch.Tensor],
    *,
    x_proj_key: str | None,
    a_log_key: str | None,
    d_state: int,
) -> tuple[tuple[str, torch.Tensor] | None, tuple[str, torch.Tensor] | None]:
    if x_proj_key is None:
        return None, None
    x_proj = _require_matrix(source_state_dict[x_proj_key], x_proj_key).detach().float().cpu()
    source_d_state = d_state
    if a_log_key is not None:
        a_log = source_state_dict[a_log_key]
        if a_log.ndim >= 2:
            source_d_state = int(a_log.shape[-1])
    dt_rank = max(0, int(x_proj.shape[0]) - 2 * source_d_state)
    b_start = min(dt_rank, int(x_proj.shape[0]))
    c_start = min(dt_rank + source_d_state, int(x_proj.shape[0]))
    b = x_proj[b_start : b_start + d_state]
    c = x_proj[c_start : c_start + d_state]
    return (f"{x_proj_key}[B]", b), (f"{x_proj_key}[C]", c)


def _decay_logits_from_a_log(a_log: torch.Tensor, *, target_rank: int) -> torch.Tensor:
    raw = a_log.detach().float().cpu()
    if raw.ndim >= 2:
        raw = raw.mean(dim=-1)
    raw = raw.reshape(-1)
    if raw.numel() == 0:
        return torch.zeros(target_rank)
    fitted = _fit_tensor(raw, (target_rank,))
    decay = torch.exp(-torch.exp(fitted)).clamp(min=1e-4, max=1 - 1e-4)
    return torch.log(decay / (1 - decay))


def _copy_exact_or_fit(
    target_tensor: torch.Tensor,
    source_tensor: torch.Tensor,
    *,
    target: str,
    source: str,
    statuses: list[AdapterTensorStatus],
) -> None:
    fitted = _fit_tensor(source_tensor.detach().float().cpu(), tuple(target_tensor.shape))
    target_tensor.copy_(fitted.to(dtype=target_tensor.dtype, device=target_tensor.device))
    source_shape = tuple(int(dim) for dim in source_tensor.shape)
    target_shape = tuple(int(dim) for dim in target_tensor.shape)
    statuses.append(
        AdapterTensorStatus(
            target=target,
            source=source,
            status="adapted",
            target_shape=target_shape,
            source_shape=source_shape,
            message="copied exactly"
            if source_shape == target_shape
            else "copied with slice/pad fit",
        )
    )


def _fit_tensor(source: torch.Tensor, target_shape: tuple[int, ...]) -> torch.Tensor:
    target = torch.zeros(target_shape, dtype=torch.float32)
    if source.ndim != len(target_shape):
        flat = source.reshape(-1)
        target.reshape(-1)[: min(flat.numel(), target.numel())] = flat[: target.numel()]
        return target
    slices = tuple(
        slice(0, min(int(source.shape[i]), target_shape[i])) for i in range(len(target_shape))
    )
    target[slices] = source[slices]
    return target


def _initialized_status(target: str, tensor: torch.Tensor | None) -> AdapterTensorStatus:
    shape = tuple(int(dim) for dim in tensor.shape) if tensor is not None else ()
    return AdapterTensorStatus(
        target=target,
        source=None,
        status="initialized",
        target_shape=shape,
        source_shape=None,
        message="kept prototype initialization",
    )


def _skipped_status(
    *,
    target: str,
    source: str,
    tensor: torch.Tensor,
    message: str,
) -> AdapterTensorStatus:
    return AdapterTensorStatus(
        target=target,
        source=source,
        status="skipped",
        target_shape=(),
        source_shape=tuple(int(dim) for dim in tensor.shape),
        message=message,
    )


def _adapt_lm_head(
    model: FheMamba3ForCausalLM,
    source_state_dict: dict[str, torch.Tensor],
    *,
    embedding_key: str,
    statuses: list[AdapterTensorStatus],
) -> None:
    lm_head_key = _find_lm_head_key(source_state_dict)
    if lm_head_key is None:
        statuses.append(
            AdapterTensorStatus(
                target="lm_head.weight",
                source=embedding_key,
                status="adapted",
                target_shape=tuple(int(dim) for dim in model.lm_head.weight.shape),
                source_shape=tuple(int(dim) for dim in source_state_dict[embedding_key].shape),
                message="prototype ties lm_head.weight to the adapted embedding",
            )
        )
        return

    lm_head = _require_matrix(source_state_dict[lm_head_key], lm_head_key)
    embedding = _require_matrix(source_state_dict[embedding_key], embedding_key)
    if tuple(lm_head.shape) == tuple(embedding.shape) and torch.allclose(
        lm_head.detach().float().cpu(),
        embedding.detach().float().cpu(),
    ):
        statuses.append(
            AdapterTensorStatus(
                target="lm_head.weight",
                source=lm_head_key,
                status="adapted",
                target_shape=tuple(int(dim) for dim in model.lm_head.weight.shape),
                source_shape=tuple(int(dim) for dim in lm_head.shape),
                message="source lm_head is tied to embedding; prototype keeps the shared weight",
            )
        )
        return

    statuses.append(
        _skipped_status(
            target="lm_head.weight",
            source=lm_head_key,
            tensor=lm_head,
            message="prototype lm_head is tied to embedding and cannot represent an untied head",
        )
    )


def _require_matrix(tensor: torch.Tensor, key: str) -> torch.Tensor:
    if tensor.ndim != 2:
        msg = f"{key} must be a rank-2 tensor"
        raise ValueError(msg)
    return tensor


def _find_first_key(
    state_dict: dict[str, torch.Tensor],
    candidates: tuple[str, ...],
) -> str | None:
    for key in candidates:
        if key in state_dict:
            return key
    return None


def _find_layer_key(
    state_dict: dict[str, torch.Tensor],
    prefix: str,
    suffixes: tuple[str, ...],
) -> str | None:
    for suffix in suffixes:
        key = f"{prefix}.{suffix}" if prefix else suffix
        if key in state_dict:
            return key
    return None


def _layer_prefix(state_dict: dict[str, torch.Tensor], layer_index: int) -> str:
    candidates = (
        f"backbone.layers.{layer_index}",
        f"layers.{layer_index}",
        f"model.layers.{layer_index}",
    )
    for prefix in candidates:
        if any(key.startswith(f"{prefix}.") for key in state_dict):
            return prefix
    return candidates[0]


def _infer_layer_count(state_dict: dict[str, torch.Tensor]) -> int:
    layer_indices = {
        int(match.group(1))
        for key in state_dict
        for match in (_LAYER_RE.search(key),)
        if match is not None
    }
    if not layer_indices:
        return 0
    return max(layer_indices) + 1


def _plan_layer(
    source_state_dict: dict[str, torch.Tensor],
    *,
    layer_index: int,
) -> MambaLayerPlan:
    prefix = _layer_prefix(source_state_dict, layer_index)
    norm_key = _find_layer_key(source_state_dict, prefix, ("norm.weight",))
    in_proj_key = _find_layer_key(
        source_state_dict,
        prefix,
        ("mixer.in_proj.weight", "in_proj.weight"),
    )
    x_proj_key = _find_layer_key(
        source_state_dict,
        prefix,
        ("mixer.x_proj.weight", "x_proj.weight"),
    )
    dt_proj_weight_key = _find_layer_key(
        source_state_dict,
        prefix,
        ("mixer.dt_proj.weight", "dt_proj.weight"),
    )
    dt_proj_bias_key = _find_layer_key(
        source_state_dict,
        prefix,
        ("mixer.dt_proj.bias", "dt_proj.bias"),
    )
    out_proj_key = _find_layer_key(
        source_state_dict,
        prefix,
        ("mixer.out_proj.weight", "out_proj.weight"),
    )
    d_key = _find_layer_key(source_state_dict, prefix, ("mixer.D", "D"))
    conv1d_weight_key = _find_layer_key(
        source_state_dict,
        prefix,
        ("mixer.conv1d.weight", "conv1d.weight"),
    )
    conv1d_bias_key = _find_layer_key(
        source_state_dict,
        prefix,
        ("mixer.conv1d.bias", "conv1d.bias"),
    )
    a_log_key = _find_layer_key(source_state_dict, prefix, ("mixer.A_log", "A_log"))
    source_d_state = _infer_source_d_state(source_state_dict, a_log_key=a_log_key)
    source_inner_dim = _infer_source_inner_dim(
        source_state_dict,
        in_proj_key=in_proj_key,
        x_proj_key=x_proj_key,
        a_log_key=a_log_key,
    )
    inferred_dt_rank = None
    if x_proj_key is not None and source_d_state is not None:
        inferred_dt_rank = max(0, int(source_state_dict[x_proj_key].shape[0]) - 2 * source_d_state)
    return MambaLayerPlan(
        layer_index=layer_index,
        prefix=prefix,
        norm_key=norm_key,
        in_proj_key=in_proj_key,
        x_proj_key=x_proj_key,
        dt_proj_weight_key=dt_proj_weight_key,
        dt_proj_bias_key=dt_proj_bias_key,
        out_proj_key=out_proj_key,
        d_key=d_key,
        conv1d_weight_key=conv1d_weight_key,
        conv1d_bias_key=conv1d_bias_key,
        a_log_key=a_log_key,
        source_inner_dim=source_inner_dim,
        source_d_state=source_d_state,
        inferred_dt_rank=inferred_dt_rank,
    )


def _find_embedding_key(source_state_dict: dict[str, torch.Tensor]) -> str | None:
    return _find_first_key(
        source_state_dict,
        (
            "backbone.embedding.weight",
            "backbone.embeddings.weight",
            "embedding.weight",
            "embeddings.weight",
            "embed_tokens.weight",
            "model.embed_tokens.weight",
            "embed.weight",
        ),
    )


def _find_lm_head_key(source_state_dict: dict[str, torch.Tensor]) -> str | None:
    return _find_first_key(
        source_state_dict,
        (
            "lm_head.weight",
            "backbone.lm_head.weight",
            "model.lm_head.weight",
        ),
    )


def _infer_source_d_state(
    source_state_dict: dict[str, torch.Tensor],
    *,
    a_log_key: str | None,
) -> int | None:
    if a_log_key is None:
        return None
    a_log = source_state_dict[a_log_key]
    if a_log.ndim >= 2:
        return int(a_log.shape[-1])
    return int(a_log.numel()) if a_log.numel() > 0 else None


def _infer_source_inner_dim(
    source_state_dict: dict[str, torch.Tensor],
    *,
    in_proj_key: str | None,
    x_proj_key: str | None,
    a_log_key: str | None,
) -> int | None:
    if a_log_key is not None and source_state_dict[a_log_key].ndim >= 1:
        return int(source_state_dict[a_log_key].shape[0])
    if x_proj_key is not None and source_state_dict[x_proj_key].ndim >= 2:
        return int(source_state_dict[x_proj_key].shape[1])
    if in_proj_key is not None and source_state_dict[in_proj_key].ndim >= 2:
        return max(1, int(source_state_dict[in_proj_key].shape[0]) // 2)
    return None
