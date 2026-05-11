"""Stage 1 head-pack/readout sweep utilities."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass
from typing import Any

from fhe_native_mamba3.backends.base import FHEBackend
from fhe_native_mamba3.layout import ReadoutStrategy
from fhe_native_mamba3.stage1_plan import Stage1CandidatePlan, build_stage1_plan
from fhe_native_mamba3.stage1_tiny_mimo import (
    build_tiny_mimo_block_problem,
    required_tiny_mimo_block_rotations,
    run_tiny_mimo_block_smoke,
)

BackendFactory = Callable[[int, tuple[int, ...]], FHEBackend]


@dataclass(frozen=True)
class Stage1PackSweepRow:
    """One measured head-pack/readout candidate row."""

    pack_size: int
    grouping_strategy: str
    backend: str
    encrypted: bool
    passed: bool
    max_abs_error: float
    eval_seconds: float
    tiny_rotation_key_count: int
    tiny_rotation_steps: tuple[int, ...]
    full_inventory_rotation_key_count: int
    estimated_key_memory_gib: float
    estimated_total_scan_depth: int
    packed_scan_depth: int
    cross_ciphertext_carry_depth: int
    scan_ciphertext_count: int
    tokens_per_scan_ciphertext: int
    estimated_bootstrap_amortization: float
    feasible_under_key_budget: bool | None
    operation_counts: dict[str, int]

    def to_json_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class Stage1PackSweepResult:
    """Pack sweep result with planning and tiny-block execution rows."""

    stage: str
    measurement_scope: dict[str, Any]
    head_count: int
    d_state: int
    d_model: int
    seq_len: int
    scan_len: int
    slot_count: int
    readout_strategy: ReadoutStrategy
    key_size_mb: float
    skipped_pack_sizes: tuple[int, ...]
    rows: tuple[Stage1PackSweepRow, ...]
    recommended_pack_size: int
    recommended_reason: str

    def to_json_dict(self) -> dict[str, Any]:
        return {
            "stage": self.stage,
            "measurement_scope": dict(self.measurement_scope),
            "head_count": self.head_count,
            "d_state": self.d_state,
            "d_model": self.d_model,
            "seq_len": self.seq_len,
            "scan_len": self.scan_len,
            "slot_count": self.slot_count,
            "readout_strategy": self.readout_strategy,
            "key_size_mb": self.key_size_mb,
            "skipped_pack_sizes": self.skipped_pack_sizes,
            "recommended_pack_size": self.recommended_pack_size,
            "recommended_reason": self.recommended_reason,
            "rows": [row.to_json_dict() for row in self.rows],
        }


def run_stage1_pack_sweep(
    *,
    backend_factory: BackendFactory,
    backend_name: str,
    encrypted: bool,
    head_count: int = 32,
    d_state: int = 64,
    d_model: int = 768,
    seq_len: int = 5,
    scan_len: int = 256,
    slot_count: int = 32768,
    candidate_pack_sizes: Sequence[int] = (4, 8, 16, 32),
    readout_strategy: ReadoutStrategy = "rank-local",
    key_size_mb: float = 200.0,
    max_key_memory_gib: float | None = 80.0,
    atol: float = 1e-10,
) -> Stage1PackSweepResult:
    """Run tiny-block measurements and attach full Stage 1 inventory estimates."""

    if not candidate_pack_sizes:
        msg = "candidate_pack_sizes must not be empty"
        raise ValueError(msg)
    plan = build_stage1_plan(
        head_count=head_count,
        d_state=d_state,
        d_model=d_model,
        scan_len=scan_len,
        slot_count=slot_count,
        candidate_pack_sizes=tuple(candidate_pack_sizes),
        grouping_strategies=("contiguous",),
        readout_strategy=readout_strategy,
        key_size_mb=key_size_mb,
        max_key_memory_gib=max_key_memory_gib,
    )
    candidates_by_pack = {
        candidate.pack_size: candidate
        for candidate in plan.candidates
        if candidate.grouping_strategy
    }
    measured_pack_sizes = tuple(
        pack_size for pack_size in candidate_pack_sizes if pack_size in candidates_by_pack
    )
    rows = tuple(
        _run_pack_sweep_row(
            pack_size=pack_size,
            candidate=candidates_by_pack[pack_size],
            backend_factory=backend_factory,
            backend_name=backend_name,
            encrypted=encrypted,
            d_state=d_state,
            seq_len=seq_len,
            slot_count=slot_count,
            atol=atol,
        )
        for pack_size in measured_pack_sizes
    )
    ranked = sorted(range(len(rows)), key=lambda index: _row_sort_key(rows[index]))
    recommended = rows[ranked[0]]
    return Stage1PackSweepResult(
        stage="stage1-head-pack-readout-sweep",
        measurement_scope={
            "tiny_block_execution": True,
            "full_inventory_estimate": True,
            "real_checkpoint_full_chain": False,
            "backend": backend_name,
            "encrypted": encrypted,
            "claim": (
                "Rows combine tiny MIMO/SSD block execution with full Stage 1 "
                "rotation-key inventory estimates; they are layout evidence, not "
                "real-checkpoint speedup claims."
            ),
        },
        head_count=head_count,
        d_state=d_state,
        d_model=d_model,
        seq_len=seq_len,
        scan_len=scan_len,
        slot_count=slot_count,
        readout_strategy=readout_strategy,
        key_size_mb=key_size_mb,
        skipped_pack_sizes=plan.skipped_pack_sizes,
        rows=rows,
        recommended_pack_size=recommended.pack_size,
        recommended_reason=(
            "lowest passing row by feasibility, estimated scan depth, measured tiny-block "
            "latency, key memory, and bootstrap amortization"
        ),
    )


def _run_pack_sweep_row(
    *,
    pack_size: int,
    candidate: Stage1CandidatePlan,
    backend_factory: BackendFactory,
    backend_name: str,
    encrypted: bool,
    d_state: int,
    seq_len: int,
    slot_count: int,
    atol: float,
) -> Stage1PackSweepRow:
    rotations = required_tiny_mimo_block_rotations(
        seq_len=seq_len,
        d_state=d_state,
        rank=pack_size,
        batch_size=slot_count,
    )
    backend = backend_factory(slot_count, rotations)
    if backend.batch_size != slot_count:
        msg = (
            "backend_factory returned batch_size="
            f"{backend.batch_size} for planned slot_count={slot_count}; "
            "pass the backend's normalized batch size to run_stage1_pack_sweep"
        )
        raise ValueError(msg)
    problem = build_tiny_mimo_block_problem(
        seq_len=seq_len,
        d_state=d_state,
        rank=pack_size,
    )
    result = run_tiny_mimo_block_smoke(problem, backend=backend)
    stats = result.backend_stats
    return Stage1PackSweepRow(
        pack_size=pack_size,
        grouping_strategy=candidate.grouping_strategy,
        backend=backend_name,
        encrypted=encrypted,
        passed=result.max_abs_error <= atol,
        max_abs_error=result.max_abs_error,
        eval_seconds=result.eval_seconds,
        tiny_rotation_key_count=len(rotations),
        tiny_rotation_steps=rotations,
        full_inventory_rotation_key_count=candidate.rotation_key_count,
        estimated_key_memory_gib=candidate.estimated_key_memory_gib,
        estimated_total_scan_depth=candidate.estimated_total_scan_depth,
        packed_scan_depth=candidate.packed_scan_depth,
        cross_ciphertext_carry_depth=candidate.cross_ciphertext_carry_depth,
        scan_ciphertext_count=candidate.scan_ciphertext_count,
        tokens_per_scan_ciphertext=candidate.tokens_per_scan_ciphertext,
        estimated_bootstrap_amortization=candidate.estimated_bootstrap_amortization,
        feasible_under_key_budget=candidate.feasible_under_key_budget,
        operation_counts={
            "ct_ct_mul": int(stats["ct_ct_mul_count"]),
            "ct_pt_mul": int(stats["ct_pt_mul_count"]),
            "add": int(stats["add_count"]),
            "rotations": int(stats["rotation_count"]),
            "bootstraps": int(stats["bootstrap_count"]),
            "encrypt": int(stats["encrypt_count"]),
            "decrypt": int(stats["decrypt_count"]),
            "encode": int(stats["encode_count"]),
        },
    )


def _row_sort_key(row: Stage1PackSweepRow) -> tuple[Any, ...]:
    feasible = row.feasible_under_key_budget
    return (
        not row.passed,
        feasible is False,
        row.estimated_total_scan_depth,
        row.eval_seconds,
        row.estimated_key_memory_gib,
        -row.estimated_bootstrap_amortization,
        row.pack_size,
    )


__all__ = [
    "BackendFactory",
    "Stage1PackSweepResult",
    "Stage1PackSweepRow",
    "run_stage1_pack_sweep",
]
