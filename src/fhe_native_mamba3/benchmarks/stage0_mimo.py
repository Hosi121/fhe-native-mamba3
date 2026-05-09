"""Stage 0 benchmark: tiny encrypted FHE-native MIMO recurrence."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Literal

from fhe_native_mamba3.backends.openfhe import OpenFheCkksBackend
from fhe_native_mamba3.backends.tracking import TrackingBackend
from fhe_native_mamba3.openfhe_backend import (
    make_demo_problem,
    required_readout_rotations,
    run_static_mimo_recurrence_with_backend,
)

Stage0Backend = Literal["openfhe", "tracking"]


@dataclass(frozen=True)
class Stage0MimoConfig:
    """Configuration for Stage 0 tiny MIMO benchmark."""

    backend: Stage0Backend = "openfhe"
    seq_len: int = 3
    d_state: int = 2
    mimo_rank: int = 2
    seed: int = 7
    multiplicative_depth: int = 8
    scaling_mod_size: int = 50

    @property
    def state_slots(self) -> int:
        return self.d_state * self.mimo_rank


def run_stage0_mimo(config: Stage0MimoConfig) -> dict[str, Any]:
    """Run Stage 0 and return OSS-stable benchmark JSON."""

    problem = make_demo_problem(
        seq_len=config.seq_len,
        d_state=config.d_state,
        mimo_rank=config.mimo_rank,
        seed=config.seed,
    )
    rotations = required_readout_rotations(
        d_state=config.d_state,
        mimo_rank=config.mimo_rank,
    )

    if config.backend == "openfhe":
        backend = OpenFheCkksBackend(
            batch_size=config.state_slots,
            multiplicative_depth=config.multiplicative_depth,
            scaling_mod_size=config.scaling_mod_size,
            rotations=rotations,
        )
    elif config.backend == "tracking":
        backend = TrackingBackend(batch_size=config.state_slots)
    else:
        msg = f"unsupported backend: {config.backend}"
        raise ValueError(msg)

    result = run_static_mimo_recurrence_with_backend(
        problem,
        backend=backend,
        multiplicative_depth=config.multiplicative_depth,
    )
    stats = result.backend_stats
    next_bottleneck = _next_bottleneck(stats)

    return {
        "stage": "0",
        "name": "tiny-fhe-native-mimo-recurrence",
        "backend": stats["backend"],
        "encrypted": stats["encrypted"],
        "model": {
            "seq_len": config.seq_len,
            "d_state": config.d_state,
            "mimo_rank": config.mimo_rank,
            "state_slots": config.state_slots,
            "parameter_count": config.mimo_rank + 2 * config.d_state * config.mimo_rank,
        },
        "ckks": {
            "multiplicative_depth": config.multiplicative_depth,
            "scaling_mod_size": config.scaling_mod_size,
            "ring_dimension": result.ring_dimension,
            "batch_size": result.batch_size,
            "rotations": list(result.rotations),
        },
        "latency_sec_per_token": result.latency_sec_per_token,
        "max_abs_error": result.max_abs_error,
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
        "timing": {
            "setup_seconds": stats["setup_seconds"],
            "eval_seconds": stats["eval_seconds"],
        },
        "problem": asdict(problem),
        "decrypted_outputs": result.decrypted_outputs,
        "expected_outputs": result.expected_outputs,
        "next_bottleneck": next_bottleneck,
    }


def _next_bottleneck(stats: dict[str, Any]) -> str:
    counts = {
        "ct_pt_mul": stats["ct_pt_mul_count"],
        "ct_ct_mul": stats["ct_ct_mul_count"],
        "rotations": stats["rotation_count"],
        "bootstrap": stats["bootstrap_count"],
    }
    name = max(counts, key=counts.get)
    if name == "ct_pt_mul":
        return (
            "plaintext-ciphertext linear/readout multiplies; next stage should add diagonal packing"
        )
    if name == "rotations":
        return (
            "slot rotations in readout; next stage should pack MIMO rank/head groups more carefully"
        )
    if name == "bootstrap":
        return "bootstrap scheduling; next stage should add lazy bootstrap and head packing"
    return (
        "ciphertext-ciphertext multiplies; next stage should reduce dynamic terms or sketch state"
    )
