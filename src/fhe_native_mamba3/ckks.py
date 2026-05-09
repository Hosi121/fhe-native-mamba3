"""Symbolic CKKS execution model.

This module does not encrypt data. It tracks the quantities that matter before
lowering to OpenFHE: levels, ciphertext-ciphertext products, rotations,
plaintext multiplications, and bootstrap placement.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import ceil


@dataclass(frozen=True)
class CkksConfig:
    """Backend parameters used by the symbolic CKKS cost model."""

    max_level: int = 30
    min_level: int = 3
    bootstrap_seconds: float = 2.0
    ct_ct_mul_ms: float = 1.0
    ct_pt_mul_ms: float = 0.02
    rotation_ms: float = 0.1
    add_ms: float = 0.001
    slots: int = 32768
    scale_bits: int = 40

    def __post_init__(self) -> None:
        if self.max_level <= self.min_level:
            msg = "max_level must be larger than min_level"
            raise ValueError(msg)
        if self.slots <= 0:
            msg = "slots must be positive"
            raise ValueError(msg)


@dataclass(frozen=True)
class PackingPlan:
    """Head/state packing plan for CKKS SIMD slots."""

    heads: int
    state_size: int
    mimo_rank: int
    slots: int = 32768
    requested_head_pack: int = 32

    def __post_init__(self) -> None:
        if self.heads <= 0:
            msg = "heads must be positive"
            raise ValueError(msg)
        if self.state_size <= 0:
            msg = "state_size must be positive"
            raise ValueError(msg)
        if self.mimo_rank <= 0:
            msg = "mimo_rank must be positive"
            raise ValueError(msg)
        if self.requested_head_pack <= 0:
            msg = "requested_head_pack must be positive"
            raise ValueError(msg)

    @property
    def slots_per_head(self) -> int:
        return self.state_size * self.mimo_rank

    @property
    def max_heads_by_slots(self) -> int:
        return max(1, self.slots // self.slots_per_head)

    @property
    def heads_per_ciphertext(self) -> int:
        return min(self.heads, self.requested_head_pack, self.max_heads_by_slots)

    @property
    def ciphertext_groups(self) -> int:
        return ceil(self.heads / self.heads_per_ciphertext)

    @property
    def slot_utilization(self) -> float:
        used = self.heads_per_ciphertext * self.slots_per_head
        return used / self.slots


@dataclass
class CkksTrace:
    """Mutable symbolic trace for one encrypted tensor stream."""

    config: CkksConfig
    level: int | None = None
    ciphertext_ciphertext_mul: int = 0
    ciphertext_plaintext_mul: int = 0
    additions: int = 0
    rotations: int = 0
    bootstraps: int = 0

    def __post_init__(self) -> None:
        if self.level is None:
            self.level = self.config.max_level

    def add(self, count: int = 1) -> None:
        self.additions += count

    def rotate(self, count: int = 1) -> None:
        self.rotations += count

    def ct_pt_mul(self, count: int = 1) -> None:
        self.ciphertext_plaintext_mul += count

    def ct_ct_mul(self, count: int = 1, depth: int = 1) -> None:
        self.ensure_depth(depth)
        self.ciphertext_ciphertext_mul += count
        if self.level is None:
            msg = "trace level is not initialized"
            raise RuntimeError(msg)
        self.level -= depth

    def ensure_depth(self, depth: int) -> None:
        if depth < 0:
            msg = "depth must be non-negative"
            raise ValueError(msg)
        if self.level is None:
            msg = "trace level is not initialized"
            raise RuntimeError(msg)
        if self.level - depth < self.config.min_level:
            self.bootstrap()

    def bootstrap(self, count: int = 1) -> None:
        self.bootstraps += count
        self.level = self.config.max_level

    @property
    def latency_seconds(self) -> float:
        return (
            self.bootstraps * self.config.bootstrap_seconds
            + self.ciphertext_ciphertext_mul * self.config.ct_ct_mul_ms / 1000.0
            + self.ciphertext_plaintext_mul * self.config.ct_pt_mul_ms / 1000.0
            + self.rotations * self.config.rotation_ms / 1000.0
            + self.additions * self.config.add_ms / 1000.0
        )
