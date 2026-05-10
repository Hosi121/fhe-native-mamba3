"""Official/HF parity probes for checkpoint-derived source-style layers."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
from torch import Tensor
from torch.nn import functional

from fhe_native_mamba3.checkpoint import load_checkpoint_state_dict
from fhe_native_mamba3.mamba_checkpoint import plan_mamba_checkpoint
from fhe_native_mamba3.mamba_reference import run_mamba_source_layer


@dataclass(frozen=True)
class OfficialMambaParityResult:
    """Result of comparing source-style layer output to an official model."""

    status: str
    reason: str
    checkpoint: str
    state_dict_key: str
    layer_index: int
    token_ids: tuple[int, ...]
    d_model: int | None
    d_state: int | None
    mimo_rank: int | None
    official_backend: str | None
    max_abs_error: float | None
    atol: float
    passed: bool
    source_style_output_shape: tuple[int, ...] | None
    official_output_shape: tuple[int, ...] | None
    notes: tuple[str, ...]

    def to_json_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["token_ids"] = list(self.token_ids)
        payload["source_style_output_shape"] = (
            list(self.source_style_output_shape)
            if self.source_style_output_shape is not None
            else None
        )
        payload["official_output_shape"] = (
            list(self.official_output_shape) if self.official_output_shape is not None else None
        )
        payload["notes"] = list(self.notes)
        return payload


def probe_official_mamba_parity(
    checkpoint: str | Path,
    *,
    token_ids: tuple[int, ...],
    state_dict_key: str | None = None,
    map_location: str = "cpu",
    layer_index: int = 0,
    d_state: int | None = None,
    mimo_rank: int | None = None,
    norm_eps: float = 1e-5,
    atol: float = 1e-5,
) -> OfficialMambaParityResult:
    """Probe whether source-style layer output matches an official model."""

    checkpoint_path = Path(checkpoint)
    source_state_dict, resolved_key = load_checkpoint_state_dict(
        checkpoint_path,
        state_dict_key=state_dict_key,
        map_location=map_location,
    )
    plan = plan_mamba_checkpoint(source_state_dict)
    if plan.embedding_key is None or plan.vocab_size is None or plan.d_model is None:
        return _skipped(
            checkpoint_path,
            resolved_key,
            layer_index=layer_index,
            token_ids=token_ids,
            reason="checkpoint does not expose an embedding tensor",
            atol=atol,
        )
    invalid = [token for token in token_ids if token < 0 or token >= plan.vocab_size]
    if invalid:
        msg = f"token ids out of range for vocab_size={plan.vocab_size}: {invalid}"
        raise ValueError(msg)
    resolved_d_state = d_state if d_state is not None else plan.inferred_d_state
    resolved_rank = mimo_rank if mimo_rank is not None else plan.inferred_mimo_rank
    if resolved_d_state is None or resolved_rank is None:
        msg = "d_state and mimo_rank must be provided when they cannot be inferred"
        raise ValueError(msg)

    source_output = _run_source_style_output(
        source_state_dict,
        token_ids=token_ids,
        embedding_key=plan.embedding_key,
        layer_index=layer_index,
        d_state=resolved_d_state,
        mimo_rank=resolved_rank,
        norm_eps=norm_eps,
    )
    config_path = checkpoint_path / "config.json" if checkpoint_path.is_dir() else None
    if config_path is None or not config_path.exists():
        return OfficialMambaParityResult(
            status="skipped",
            reason="official HF parity requires a checkpoint directory with config.json",
            checkpoint=str(checkpoint),
            state_dict_key=resolved_key,
            layer_index=layer_index,
            token_ids=token_ids,
            d_model=plan.d_model,
            d_state=resolved_d_state,
            mimo_rank=resolved_rank,
            official_backend=None,
            max_abs_error=None,
            atol=atol,
            passed=False,
            source_style_output_shape=tuple(int(dim) for dim in source_output.shape),
            official_output_shape=None,
            notes=(
                "source-style checkpoint arithmetic is available",
                "official parity is blocked before model construction",
            ),
        )

    try:
        official_output = _run_transformers_hidden_state(
            checkpoint_path,
            token_ids=token_ids,
            layer_index=layer_index,
        )
    except Exception as exc:  # keep parity probes non-destructive.
        return OfficialMambaParityResult(
            status="error",
            reason=f"{type(exc).__name__}: {exc}",
            checkpoint=str(checkpoint),
            state_dict_key=resolved_key,
            layer_index=layer_index,
            token_ids=token_ids,
            d_model=plan.d_model,
            d_state=resolved_d_state,
            mimo_rank=resolved_rank,
            official_backend="transformers",
            max_abs_error=None,
            atol=atol,
            passed=False,
            source_style_output_shape=tuple(int(dim) for dim in source_output.shape),
            official_output_shape=None,
            notes=("official model construction or forward failed",),
        )

    if tuple(source_output.shape) != tuple(official_output.shape):
        return OfficialMambaParityResult(
            status="failed",
            reason="source-style and official outputs have different shapes",
            checkpoint=str(checkpoint),
            state_dict_key=resolved_key,
            layer_index=layer_index,
            token_ids=token_ids,
            d_model=plan.d_model,
            d_state=resolved_d_state,
            mimo_rank=resolved_rank,
            official_backend="transformers",
            max_abs_error=None,
            atol=atol,
            passed=False,
            source_style_output_shape=tuple(int(dim) for dim in source_output.shape),
            official_output_shape=tuple(int(dim) for dim in official_output.shape),
            notes=("shape mismatch prevents numerical parity claim",),
        )

    max_abs_error = float((source_output - official_output).abs().max().item())
    passed = max_abs_error <= atol
    return OfficialMambaParityResult(
        status="passed" if passed else "failed",
        reason="" if passed else f"max_abs_error={max_abs_error} > atol={atol}",
        checkpoint=str(checkpoint),
        state_dict_key=resolved_key,
        layer_index=layer_index,
        token_ids=token_ids,
        d_model=plan.d_model,
        d_state=resolved_d_state,
        mimo_rank=resolved_rank,
        official_backend="transformers",
        max_abs_error=max_abs_error,
        atol=atol,
        passed=passed,
        source_style_output_shape=tuple(int(dim) for dim in source_output.shape),
        official_output_shape=tuple(int(dim) for dim in official_output.shape),
        notes=("compares source-style layer output to transformers hidden_states[layer+1]",),
    )


def _run_source_style_output(
    state_dict: dict[str, Tensor],
    *,
    token_ids: tuple[int, ...],
    embedding_key: str,
    layer_index: int,
    d_state: int,
    mimo_rank: int,
    norm_eps: float,
) -> Tensor:
    input_ids = torch.tensor([list(token_ids)], dtype=torch.long)
    embedding = state_dict[embedding_key].to(dtype=torch.float32)
    x = functional.embedding(input_ids, embedding)
    with torch.inference_mode():
        for current_layer in range(layer_index + 1):
            x = run_mamba_source_layer(
                state_dict,
                x,
                layer_index=current_layer,
                d_state=d_state,
                mimo_rank=mimo_rank,
                norm_eps=norm_eps,
            )
    return x.detach().cpu()


def _run_transformers_hidden_state(
    checkpoint_path: Path,
    *,
    token_ids: tuple[int, ...],
    layer_index: int,
) -> Tensor:
    from transformers import AutoModelForCausalLM  # type: ignore[import-not-found]

    model = AutoModelForCausalLM.from_pretrained(
        checkpoint_path,
        local_files_only=True,
        trust_remote_code=True,
    )
    model.eval()
    input_ids = torch.tensor([list(token_ids)], dtype=torch.long)
    with torch.inference_mode():
        outputs = model(input_ids, output_hidden_states=True, use_cache=False)
    hidden_states = getattr(outputs, "hidden_states", None)
    if hidden_states is None:
        msg = "official model did not return hidden_states"
        raise RuntimeError(msg)
    hidden_index = layer_index + 1
    if hidden_index >= len(hidden_states):
        msg = f"official hidden_states has no entry {hidden_index}"
        raise RuntimeError(msg)
    return hidden_states[hidden_index].detach().cpu().to(dtype=torch.float32)


def _skipped(
    checkpoint: Path,
    state_dict_key: str,
    *,
    layer_index: int,
    token_ids: tuple[int, ...],
    reason: str,
    atol: float,
) -> OfficialMambaParityResult:
    return OfficialMambaParityResult(
        status="skipped",
        reason=reason,
        checkpoint=str(checkpoint),
        state_dict_key=state_dict_key,
        layer_index=layer_index,
        token_ids=token_ids,
        d_model=None,
        d_state=None,
        mimo_rank=None,
        official_backend=None,
        max_abs_error=None,
        atol=atol,
        passed=False,
        source_style_output_shape=None,
        official_output_shape=None,
        notes=("official parity could not be evaluated",),
    )


def parity_result_to_json(result: OfficialMambaParityResult) -> str:
    """Return a stable pretty-printed JSON representation."""

    return json.dumps(result.to_json_dict(), indent=2, sort_keys=True)
