"""Conservative multiplicative-depth estimates for recurrence lowering."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Literal

from fhe_native_mamba3.ckks import CkksConfig
from fhe_native_mamba3.cost import greedy_bootstrap_schedule
from fhe_native_mamba3.layout import ReadoutStrategy, readout_reduce_steps

InputMode = Literal["server-bx", "client-update", "encrypted-dynamic-bc"]


@dataclass(frozen=True)
class RecurrenceDepthEstimate:
    """Depth estimate for one lowered recurrence problem."""

    seq_len: int
    d_state: int
    input_mode: str
    readout_strategy: str
    state_depth: int
    contribution_depth: int
    readout_extra_depth: int
    output_depth: int
    d_skip_depth: int
    recommended_multiplicative_depth: int

    def to_json_dict(self) -> dict[str, int | str]:
        return asdict(self)


@dataclass(frozen=True)
class RecurrenceBootstrapGroup:
    """Bootstrap plan for one recurrence-source/sequence configuration."""

    recurrence_source: str
    seq_len: int
    input_mode: str
    readout_strategy: str
    layer_indices: tuple[int, ...]
    layer_depths: tuple[int, ...]
    bootstrap_before_layers: tuple[int, ...]
    final_level: int
    bootstraps: int

    def to_json_dict(self) -> dict[str, int | str | list[int]]:
        payload = asdict(self)
        payload["layer_indices"] = list(self.layer_indices)
        payload["layer_depths"] = list(self.layer_depths)
        payload["bootstrap_before_layers"] = list(self.bootstrap_before_layers)
        return payload


def estimate_recurrence_depth(
    *,
    seq_len: int,
    d_state: int,
    input_mode: InputMode,
    readout_strategy: ReadoutStrategy,
    has_d_skip: bool,
) -> RecurrenceDepthEstimate:
    """Estimate depth consumed by the current sequential CKKS recurrence lowering.

    The current lowering multiplies the carried state by plaintext or encrypted
    decay once per token, multiplies the final state by C once per token, and
    then performs readout reductions with plaintext masks. Plaintext
    multiplications still consume CKKS levels in OpenFHE, so they matter here.
    """

    if seq_len <= 0:
        msg = "seq_len must be positive"
        raise ValueError(msg)
    if d_state <= 0:
        msg = "d_state must be positive"
        raise ValueError(msg)
    if input_mode not in {"server-bx", "client-update", "encrypted-dynamic-bc"}:
        msg = f"unsupported input_mode: {input_mode}"
        raise ValueError(msg)

    state_depth = seq_len
    contribution_depth = state_depth + 1
    readout_extra_depth = _readout_extra_depth(
        d_state=d_state,
        readout_strategy=readout_strategy,
    )
    output_depth = contribution_depth + readout_extra_depth
    d_skip_depth = 1 if has_d_skip and input_mode != "client-update" else 0
    recommended = max(output_depth, d_skip_depth)
    return RecurrenceDepthEstimate(
        seq_len=seq_len,
        d_state=d_state,
        input_mode=input_mode,
        readout_strategy=readout_strategy,
        state_depth=state_depth,
        contribution_depth=contribution_depth,
        readout_extra_depth=readout_extra_depth,
        output_depth=output_depth,
        d_skip_depth=d_skip_depth,
        recommended_multiplicative_depth=recommended,
    )


def build_recurrence_bootstrap_plan(
    rows: list[dict],
    *,
    ckks_max_level: int,
    ckks_min_level: int,
) -> dict[str, int | list[dict[str, int | str | list[int]]]]:
    """Build greedy bootstrap schedules from recurrence sweep rows."""

    ckks = CkksConfig(max_level=ckks_max_level, min_level=ckks_min_level)
    groups = tuple(_recurrence_bootstrap_groups(rows, ckks=ckks))
    return {
        "ckks_max_level": ckks.max_level,
        "ckks_min_level": ckks.min_level,
        "group_count": len(groups),
        "max_bootstraps": max((group.bootstraps for group in groups), default=0),
        "groups": [group.to_json_dict() for group in groups],
    }


def _recurrence_bootstrap_groups(
    rows: list[dict],
    *,
    ckks: CkksConfig,
) -> list[RecurrenceBootstrapGroup]:
    grouped: dict[tuple[str, int, str, str], dict[int, int]] = {}
    for row in rows:
        key = (
            str(row["recurrence_source"]),
            int(row["seq_len"]),
            str(row["input_mode"]),
            str(row["readout_strategy"]),
        )
        layer_depths = grouped.setdefault(key, {})
        layer_index = int(row["layer_index"])
        depth = int(row["depth_advisory"]["recommended_multiplicative_depth"])
        layer_depths[layer_index] = max(depth, layer_depths.get(layer_index, 0))

    plans: list[RecurrenceBootstrapGroup] = []
    for key in sorted(grouped):
        recurrence_source, seq_len, input_mode, readout_strategy = key
        depths_by_layer = grouped[key]
        layer_indices = tuple(sorted(depths_by_layer))
        layer_depths = tuple(depths_by_layer[index] for index in layer_indices)
        schedule = greedy_bootstrap_schedule(layer_depths, ckks)
        plans.append(
            RecurrenceBootstrapGroup(
                recurrence_source=recurrence_source,
                seq_len=seq_len,
                input_mode=input_mode,
                readout_strategy=readout_strategy,
                layer_indices=layer_indices,
                layer_depths=layer_depths,
                bootstrap_before_layers=tuple(
                    layer_indices[position] for position in schedule.bootstrap_before_layers
                ),
                final_level=schedule.final_level,
                bootstraps=schedule.bootstraps,
            )
        )
    return plans


def _readout_extra_depth(
    *,
    d_state: int,
    readout_strategy: ReadoutStrategy,
) -> int:
    if readout_strategy == "rank-local":
        return len(readout_reduce_steps(d_state))
    if readout_strategy == "rank-reduce":
        return len(readout_reduce_steps(d_state)) + 1
    if readout_strategy == "slotwise":
        return 1
    msg = f"unsupported readout_strategy: {readout_strategy}"
    raise ValueError(msg)
