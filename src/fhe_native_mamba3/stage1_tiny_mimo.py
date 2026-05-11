"""Stage 1 tiny packed MIMO/SSD block smoke helpers."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor

from fhe_native_mamba3.backends.base import FHEBackend
from fhe_native_mamba3.ssd import sequential_static_scan
from fhe_native_mamba3.ssd_prefix_scan import (
    BackendPackedMimoReadoutResult,
    PackedPrefixScanPlan,
    backend_packed_static_mimo_readout,
    backend_segmented_hillis_steele_affine_scan,
    build_packed_prefix_scan_plan,
)


@dataclass(frozen=True)
class TinyMimoBlockProblem:
    """Small deterministic static MIMO block used for Stage 1 smoke tests."""

    rank_input: Tensor
    b_terms: Tensor
    c_terms: Tensor
    decay: Tensor

    @property
    def seq_len(self) -> int:
        return int(self.rank_input.shape[0])

    @property
    def d_state(self) -> int:
        return int(self.b_terms.shape[0])

    @property
    def rank(self) -> int:
        return int(self.rank_input.shape[1])


@dataclass(frozen=True)
class TinyMimoBlockSmokeResult:
    """End-to-end tiny packed MIMO block smoke result."""

    decoded_output: Tensor
    expected_output: Tensor
    max_abs_error: float
    d_state: int
    rank: int
    plan: PackedPrefixScanPlan
    readout: BackendPackedMimoReadoutResult
    backend_stats: dict[str, Any]
    eval_seconds: float

    def to_json_dict(self, *, atol: float) -> dict[str, Any]:
        stats = self.backend_stats
        return {
            "config": {
                "seq_len": self.plan.seq_len,
                "d_state": self.d_state,
                "rank": self.rank,
                "lanes": self.plan.lanes,
                "batch_size": stats.get("batch_size"),
            },
            "plan": self.plan.to_json_dict(),
            "readout": {
                "rotations": self.readout.rotations,
                "output_slots": self.readout.output_slots,
            },
            "operation_counts": {
                "ct_ct_mul": stats["ct_ct_mul_count"],
                "ct_pt_mul": stats["ct_pt_mul_count"],
                "add": stats["add_count"],
                "rotations": stats["rotation_count"],
                "bootstraps": stats["bootstrap_count"],
                "encrypt": stats["encrypt_count"],
                "decrypt": stats["decrypt_count"],
                "encode": stats["encode_count"],
            },
            "timing": {"eval_seconds": self.eval_seconds},
            "passed": self.max_abs_error <= atol,
            "max_abs_error": self.max_abs_error,
            "atol": atol,
        }


def build_tiny_mimo_block_problem(
    *,
    seq_len: int,
    d_state: int,
    rank: int,
) -> TinyMimoBlockProblem:
    """Build a deterministic low-noise static MIMO block problem."""

    if seq_len <= 0 or d_state <= 0 or rank <= 0:
        msg = "seq_len, d_state, and rank must be positive"
        raise ValueError(msg)
    rank_input = torch.linspace(
        -0.35,
        0.65,
        steps=seq_len * rank,
        dtype=torch.float64,
    ).view(seq_len, rank)
    b_terms = torch.linspace(
        0.15,
        0.55,
        steps=d_state * rank,
        dtype=torch.float64,
    ).view(d_state, rank)
    c_terms = torch.linspace(
        -0.30,
        0.45,
        steps=d_state * rank,
        dtype=torch.float64,
    ).view(d_state, rank)
    decay = torch.linspace(
        0.55,
        0.88,
        steps=d_state * rank,
        dtype=torch.float64,
    ).view(d_state, rank)
    return TinyMimoBlockProblem(
        rank_input=rank_input,
        b_terms=b_terms,
        c_terms=c_terms,
        decay=decay,
    )


def run_tiny_mimo_block_smoke(
    problem: TinyMimoBlockProblem,
    *,
    backend: FHEBackend,
) -> TinyMimoBlockSmokeResult:
    """Run a tiny static MIMO block through packed encrypted-style primitives."""

    lanes = problem.d_state * problem.rank
    plan = build_packed_prefix_scan_plan(
        seq_len=problem.seq_len,
        lanes=lanes,
        slot_count=backend.batch_size,
    )
    decay_values = _packed_decay_values(problem)
    update_values = _packed_update_values(problem)
    decay_ciphertexts = tuple(
        backend.encrypt(_pack_chunk(chunk, batch_size=backend.batch_size))
        for chunk in decay_values.split(plan.tokens_per_ciphertext, dim=0)
    )
    update_ciphertexts = tuple(
        backend.encrypt(_pack_chunk(chunk, batch_size=backend.batch_size))
        for chunk in update_values.split(plan.tokens_per_ciphertext, dim=0)
    )

    started = time.perf_counter()
    affine = backend_segmented_hillis_steele_affine_scan(
        decay_ciphertexts,
        update_ciphertexts,
        seq_len=problem.seq_len,
        lanes=lanes,
        backend=backend,
    )
    readout = backend_packed_static_mimo_readout(
        affine.state_ciphertexts,
        seq_len=problem.seq_len,
        d_state=problem.d_state,
        rank=problem.rank,
        c_terms=problem.c_terms,
        backend=backend,
    )
    eval_seconds = time.perf_counter() - started
    decoded = _decode_readout(
        readout,
        backend=backend,
        rank=problem.rank,
    )
    expected = sequential_static_scan(
        problem.rank_input.unsqueeze(0),
        problem.b_terms,
        problem.c_terms,
        problem.decay,
        decay_mode="state_rank",
    )
    max_abs_error = float((decoded - expected).abs().max().item())
    stats = backend.stats().to_json_dict()
    stats["batch_size"] = backend.batch_size
    return TinyMimoBlockSmokeResult(
        decoded_output=decoded,
        expected_output=expected,
        max_abs_error=max_abs_error,
        d_state=problem.d_state,
        rank=problem.rank,
        plan=affine.plan,
        readout=readout,
        backend_stats=stats,
        eval_seconds=eval_seconds,
    )


def required_tiny_mimo_block_rotations(
    *,
    seq_len: int,
    d_state: int,
    rank: int,
    batch_size: int,
) -> tuple[int, ...]:
    """Return rotation steps required by the tiny packed MIMO block."""

    if d_state <= 0 or rank <= 0:
        msg = "d_state and rank must be positive"
        raise ValueError(msg)
    plan = build_packed_prefix_scan_plan(
        seq_len=seq_len,
        lanes=d_state * rank,
        slot_count=batch_size,
    )
    rotations = set(plan.rotations)
    rotations.update(-rotation for rotation in plan.rotations)
    rotations.update(plan.carry_rotations)
    rotations.update(-rotation for rotation in plan.carry_rotations)
    rotations.update(range(1, d_state))
    return tuple(sorted(rotation for rotation in rotations if rotation))


def _packed_decay_values(problem: TinyMimoBlockProblem) -> Tensor:
    lanes = problem.d_state * problem.rank
    values = torch.empty(problem.seq_len, lanes, dtype=torch.float64)
    for token_index in range(problem.seq_len):
        for rank_index in range(problem.rank):
            for state_index in range(problem.d_state):
                slot = rank_index * problem.d_state + state_index
                values[token_index, slot] = problem.decay[state_index, rank_index]
    return values


def _packed_update_values(problem: TinyMimoBlockProblem) -> Tensor:
    lanes = problem.d_state * problem.rank
    values = torch.empty(problem.seq_len, lanes, dtype=torch.float64)
    for token_index in range(problem.seq_len):
        for rank_index in range(problem.rank):
            for state_index in range(problem.d_state):
                slot = rank_index * problem.d_state + state_index
                values[token_index, slot] = (
                    problem.rank_input[token_index, rank_index]
                    * problem.b_terms[state_index, rank_index]
                )
    return values


def _pack_chunk(chunk: Tensor, *, batch_size: int) -> tuple[float, ...]:
    flat = [float(value) for value in chunk.flatten()]
    if len(flat) > batch_size:
        msg = "chunk does not fit in batch_size"
        raise ValueError(msg)
    return tuple(flat + [0.0] * (batch_size - len(flat)))


def _decode_readout(
    readout: BackendPackedMimoReadoutResult,
    *,
    backend: FHEBackend,
    rank: int,
) -> Tensor:
    decoded_chunks: list[Tensor] = []
    for ciphertext, slots in zip(readout.ciphertexts, readout.output_slots, strict=True):
        values = backend.decrypt(ciphertext, length=backend.batch_size)
        decoded_chunks.append(
            torch.tensor([values[slot] for slot in slots], dtype=torch.float64).view(
                -1,
                rank,
            )
        )
    return torch.cat(decoded_chunks, dim=0).unsqueeze(0)


def payload_for_tiny_mimo_block_smoke(
    *,
    version: str,
    result: TinyMimoBlockSmokeResult,
    atol: float,
) -> dict[str, Any]:
    """Build the JSON payload emitted by the Stage 1 tiny block smoke."""

    payload = result.to_json_dict(atol=atol)
    payload.update(
        {
            "version": version,
            "stage": "stage1-tiny-mimo-block-smoke",
            "backend": result.backend_stats["backend"],
            "encrypted": result.backend_stats["encrypted"],
            "measurement_scope": {
                "packed_prefix_scan": True,
                "cross_ciphertext_carry": result.plan.requires_cross_ciphertext_carry,
                "static_mimo_recurrence": True,
                "packed_readout": True,
                "plaintext_reference": "sequential_static_scan",
            },
        }
    )
    return payload
