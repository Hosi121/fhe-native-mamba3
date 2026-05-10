"""Conservative multiplicative-depth estimates for recurrence lowering."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Literal

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
