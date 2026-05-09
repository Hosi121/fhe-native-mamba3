"""FHE operation estimates for the prototype block."""

from __future__ import annotations

from dataclasses import dataclass

from fhe_native_mamba3.ckks import CkksConfig, PackingPlan
from fhe_native_mamba3.model import FheMamba3Config


@dataclass(frozen=True)
class FheCostEstimate:
    """Approximate per-block inference cost for one encrypted sequence."""

    seq_len: int
    ciphertext_ciphertext_mul: int
    ciphertext_plaintext_mul: int
    additions: int
    rotations: int
    multiplicative_depth: int
    notes: tuple[str, ...]

    @property
    def total_mul(self) -> int:
        return self.ciphertext_ciphertext_mul + self.ciphertext_plaintext_mul


@dataclass(frozen=True)
class BootstrapSchedule:
    """Greedy bootstrap placement over a regular layer stack."""

    layer_depths: tuple[int, ...]
    bootstrap_before_layers: tuple[int, ...]
    final_level: int
    bootstraps: int


@dataclass(frozen=True)
class IntegratedCostEstimate:
    """Conjecture-style token cost model for the current research direction."""

    token_seconds: float
    scan_seconds_per_layer: float
    nonlinearity_seconds_per_layer: float
    amortized_bootstrap_seconds_per_layer: float
    effective_window: int
    head_packing: PackingPlan
    bootstrap_schedule: BootstrapSchedule
    block_cost: FheCostEstimate
    notes: tuple[str, ...]


def greedy_bootstrap_schedule(
    layer_depths: tuple[int, ...],
    ckks: CkksConfig,
) -> BootstrapSchedule:
    """Place bootstraps greedily before a layer would underflow the level budget."""

    level = ckks.max_level
    positions: list[int] = []
    for index, depth in enumerate(layer_depths):
        if depth < 0:
            msg = "layer depths must be non-negative"
            raise ValueError(msg)
        if level - depth < ckks.min_level:
            positions.append(index)
            level = ckks.max_level
        level -= depth
    return BootstrapSchedule(
        layer_depths=layer_depths,
        bootstrap_before_layers=tuple(positions),
        final_level=level,
        bootstraps=len(positions),
    )


def estimate_block_cost(config: FheMamba3Config, seq_len: int) -> FheCostEstimate:
    """Estimate dominant arithmetic for one block and one batch item.

    Counts are intentionally conservative and backend-agnostic. They are useful
    for comparing static-vs-dynamic B/C and gate variants before implementing a
    concrete CKKS/BFV lowering.
    """

    if seq_len <= 0:
        msg = "seq_len must be positive"
        raise ValueError(msg)

    d_model = config.d_model
    d_state = config.d_state
    rank = config.mimo_rank

    # Linear maps over encrypted activations with plaintext weights.
    ct_pt = seq_len * (d_model * rank + d_model * d_model + rank * d_model)
    adds = seq_len * (d_model * rank + d_model * d_model + rank * d_model)
    ct_ct = 0
    rotations = 0
    depth = 0
    notes: list[str] = []

    if config.scan_mode == "windowed":
        window = min(config.effective_window or seq_len, seq_len)
        rotations += seq_len * max(0, window.bit_length() - 1)
        notes.append(f"windowed SSD form uses effective_window={window}")
    else:
        window = seq_len

    if config.bc_mode == "static":
        recurrent_steps = seq_len * window if config.scan_mode == "windowed" else seq_len
        ct_pt += recurrent_steps * (2 * d_state * rank)
        adds += recurrent_steps * (2 * d_state * rank)
        notes.append("static B/C keeps the recurrent path at ciphertext-plaintext depth")
    else:
        ct_pt += seq_len * (2 * d_model * d_state * rank)
        ct_ct += seq_len * (2 * d_state * rank)
        adds += seq_len * (2 * d_model * d_state * rank + 2 * d_state * rank)
        depth += 2
        notes.append("dynamic B/C is closer to Mamba-3 MIMO but adds ct-ct products")

    if config.gate_mode == "linear":
        ct_pt += seq_len * d_model * d_model
        ct_ct += seq_len * d_model
        adds += seq_len * (d_model * d_model + d_model)
        depth += 1
        notes.append("linear polynomial gate adds one ciphertext-ciphertext layer")
    elif config.gate_mode == "quadratic":
        ct_pt += seq_len * d_model * d_model
        ct_ct += seq_len * (2 * d_model)
        adds += seq_len * (d_model * d_model + 2 * d_model)
        depth += 2
        notes.append("quadratic gate improves sigmoid fit but costs another depth level")
    else:
        notes.append("no gate is cheapest but reduces expressivity")

    return FheCostEstimate(
        seq_len=seq_len,
        ciphertext_ciphertext_mul=ct_ct,
        ciphertext_plaintext_mul=ct_pt,
        additions=adds,
        rotations=rotations,
        multiplicative_depth=depth,
        notes=tuple(notes),
    )


def estimate_integrated_cost(
    config: FheMamba3Config,
    *,
    seq_len: int,
    heads: int,
    requested_head_pack: int,
    ckks: CkksConfig,
    scan_step_ms: float = 1.0,
    nonlinearity_ms: float = 0.0,
    bootstrap_every_layers: int = 2,
) -> IntegratedCostEstimate:
    """Estimate the token cost model stated in the research memo.

    This is a symbolic model, not an OpenFHE measurement. It intentionally keeps
    the user's current assumptions explicit: lazy bootstrap scheduling,
    head-packed bootstrap, and a scan cost linear in the effective window.
    """

    if bootstrap_every_layers <= 0:
        msg = "bootstrap_every_layers must be positive"
        raise ValueError(msg)
    if scan_step_ms < 0 or nonlinearity_ms < 0:
        msg = "timing constants must be non-negative"
        raise ValueError(msg)

    block_cost = estimate_block_cost(config, seq_len=seq_len)
    effective_window = min(config.effective_window or seq_len, seq_len)
    packing = PackingPlan(
        heads=heads,
        state_size=config.d_state,
        mimo_rank=config.mimo_rank,
        slots=ckks.slots,
        requested_head_pack=requested_head_pack,
    )
    layer_depths = tuple(block_cost.multiplicative_depth for _ in range(config.n_layers))
    schedule = greedy_bootstrap_schedule(layer_depths, ckks)

    scan_seconds = effective_window * scan_step_ms / 1000.0
    nonlin_seconds = nonlinearity_ms / 1000.0
    bootstrap_seconds = (
        ckks.bootstrap_seconds
        / bootstrap_every_layers
        * packing.ciphertext_groups
        / max(1, packing.heads)
    )
    token_seconds = config.n_layers * (scan_seconds + nonlin_seconds + bootstrap_seconds)

    notes = [
        "symbolic CKKS model; no OpenFHE encryption is executed",
        "head-packed bootstrap is amortized across logical heads",
        "scan cost follows the memo's linear C_scan(tau) assumption",
    ]
    if schedule.bootstraps == 0:
        notes.append("greedy level schedule needs no intra-stack bootstrap for one token")
    else:
        notes.append(f"greedy level schedule inserts {schedule.bootstraps} bootstrap(s)")

    return IntegratedCostEstimate(
        token_seconds=token_seconds,
        scan_seconds_per_layer=scan_seconds,
        nonlinearity_seconds_per_layer=nonlin_seconds,
        amortized_bootstrap_seconds_per_layer=bootstrap_seconds,
        effective_window=effective_window,
        head_packing=packing,
        bootstrap_schedule=schedule,
        block_cost=block_cost,
        notes=tuple(notes),
    )
