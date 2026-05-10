"""Build encrypted recurrence smoke tests from saved weight bundles."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch

from fhe_native_mamba3.openfhe_backend import OpenFheRecurrenceProblem
from fhe_native_mamba3.weight_bundle import WeightBundleManifest, load_weight_bundle_model


@dataclass(frozen=True)
class WeightBundleRecurrenceProblem:
    """A recurrence problem extracted from a real bundle layer."""

    bundle_dir: str
    layer_index: int
    token_ids: tuple[int, ...]
    problem: OpenFheRecurrenceProblem
    manifest: WeightBundleManifest

    def to_json_dict(self) -> dict[str, object]:
        return {
            "bundle_dir": self.bundle_dir,
            "layer_index": self.layer_index,
            "token_ids": list(self.token_ids),
            "problem": {
                "rank_inputs": [list(row) for row in self.problem.rank_inputs],
                "decay": list(self.problem.decay),
                "decay_by_token": [list(row) for row in self.problem.decay_by_token]
                if self.problem.decay_by_token is not None
                else None,
                "decay_state_by_token": [
                    [list(row) for row in matrix] for matrix in self.problem.decay_state_by_token
                ]
                if self.problem.decay_state_by_token is not None
                else None,
                "b": [list(row) for row in self.problem.b],
                "c": [list(row) for row in self.problem.c],
                "b_by_token": [[list(row) for row in matrix] for matrix in self.problem.b_by_token]
                if self.problem.b_by_token is not None
                else None,
                "c_by_token": [[list(row) for row in matrix] for matrix in self.problem.c_by_token]
                if self.problem.c_by_token is not None
                else None,
                "d_skip": list(self.problem.d_skip) if self.problem.d_skip is not None else None,
            },
        }


def build_weight_bundle_recurrence_problem(
    bundle_dir: str | Path,
    *,
    token_ids: tuple[int, ...],
    layer_index: int = 0,
    bc_mode: str = "static",
) -> WeightBundleRecurrenceProblem:
    """Extract a scalar MIMO recurrence problem from a saved bundle."""

    if not token_ids:
        msg = "token_ids must be non-empty"
        raise ValueError(msg)

    model, manifest = load_weight_bundle_model(bundle_dir, map_location="cpu")
    if layer_index < 0 or layer_index >= len(model.blocks):
        msg = f"layer_index must be in [0, {len(model.blocks) - 1}]"
        raise ValueError(msg)
    invalid = [token for token in token_ids if token < 0 or token >= model.config.vocab_size]
    if invalid:
        msg = f"token ids out of range for vocab_size={model.config.vocab_size}: {invalid}"
        raise ValueError(msg)
    if len(token_ids) > model.config.max_seq_len:
        msg = "token_ids length exceeds bundle max_seq_len"
        raise ValueError(msg)
    if model.config.decay_mode != "scalar":
        msg = "weight-bundle recurrence smoke currently supports scalar decay only"
        raise ValueError(msg)
    if bc_mode not in {"static", "dynamic"}:
        msg = f"unsupported recurrence bc_mode: {bc_mode}"
        raise ValueError(msg)
    if bc_mode == "static" and model.config.bc_mode != "static":
        msg = "static recurrence smoke requires a bundle with static B/C"
        raise ValueError(msg)
    if bc_mode == "dynamic" and model.config.bc_mode != "dynamic":
        msg = "dynamic recurrence smoke requires a bundle with dynamic B/C"
        raise ValueError(msg)

    model.eval()
    input_ids = torch.tensor([token_ids], dtype=torch.long)
    with torch.inference_mode():
        x = model.embed(input_ids) + model.pos[: len(token_ids)].unsqueeze(0)
        for block in model.blocks[:layer_index]:
            x = block(x)
        block = model.blocks[layer_index]
        x_norm = block.in_norm(x)
        rank_input = block._causal_rank_conv(block.in_rank(x_norm))[0].detach().cpu()
        decay = block._decay(dtype=rank_input.dtype, device=rank_input.device).view(-1)
        decay_by_token_tensor = block._decay_by_token(rank_input.unsqueeze(0), decay)
        decay_by_token = (
            decay_by_token_tensor[0].detach().cpu() if decay_by_token_tensor is not None else None
        )
        if bc_mode == "static":
            if block.b_static is None or block.c_static is None:
                msg = "selected block does not contain static B/C parameters"
                raise ValueError(msg)
            b_static = block.b_static.detach().cpu()
            c_static = block.c_static.detach().cpu()
            b_by_token = None
            c_by_token = None
        else:
            if block.b_dynamic is None or block.c_dynamic is None:
                msg = "selected block does not contain dynamic B/C projections"
                raise ValueError(msg)
            shape = (1, len(token_ids), model.config.d_state, model.config.mimo_rank)
            b_tensor = block.b_dynamic(x_norm).view(shape)[0].detach().cpu()
            c_tensor = block.c_dynamic(x_norm).view(shape)[0].detach().cpu()
            b_static = b_tensor.mean(dim=0)
            c_static = c_tensor.mean(dim=0)
            b_by_token = _tensor_matrices(b_tensor)
            c_by_token = _tensor_matrices(c_tensor)
        d_skip = block.d_skip.detach().cpu()

    problem = OpenFheRecurrenceProblem(
        rank_inputs=_tensor_rows(rank_input),
        decay=_tensor_vector(decay),
        decay_by_token=_tensor_rows(decay_by_token) if decay_by_token is not None else None,
        b=_tensor_rows(b_static),
        c=_tensor_rows(c_static),
        b_by_token=b_by_token,
        c_by_token=c_by_token,
        d_skip=_tensor_vector(d_skip),
    )
    return WeightBundleRecurrenceProblem(
        bundle_dir=str(bundle_dir),
        layer_index=layer_index,
        token_ids=token_ids,
        problem=problem,
        manifest=manifest,
    )


def _tensor_vector(tensor: torch.Tensor) -> tuple[float, ...]:
    return tuple(float(value) for value in tensor.reshape(-1).tolist())


def _tensor_rows(tensor: torch.Tensor) -> tuple[tuple[float, ...], ...]:
    return tuple(tuple(float(value) for value in row) for row in tensor.tolist())


def _tensor_matrices(tensor: torch.Tensor) -> tuple[tuple[tuple[float, ...], ...], ...]:
    return tuple(_tensor_rows(matrix) for matrix in tensor)
