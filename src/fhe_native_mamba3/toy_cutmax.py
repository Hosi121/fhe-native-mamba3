"""Toy encrypted CutMax/argmax smoke over a tiny packed vocabulary."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

from fhe_native_mamba3.backends.base import FHEBackend


@dataclass(frozen=True)
class ToyCutMaxSmokeResult:
    """Result of a tiny encrypted-style argmax approximation."""

    logits: tuple[float, ...]
    decoded_winner_mask: tuple[float, ...]
    expected_argmax: int
    selected_argmax: int
    winner_margin: float
    margin_scale: float
    passed: bool
    backend_stats: dict[str, Any]
    eval_seconds: float

    def to_json_dict(self) -> dict[str, Any]:
        stats = self.backend_stats
        return {
            "config": {
                "vocab_size": len(self.logits),
                "batch_size": stats.get("batch_size"),
                "margin_scale": self.margin_scale,
            },
            "logits": list(self.logits),
            "decoded_winner_mask": list(self.decoded_winner_mask),
            "expected_argmax": self.expected_argmax,
            "selected_argmax": self.selected_argmax,
            "winner_margin": self.winner_margin,
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
            "passed": self.passed,
        }


def run_toy_cutmax_smoke(
    *,
    backend: FHEBackend,
    logits: tuple[float, ...] = (0.75, 0.1, -0.2, -0.5),
    margin_scale: float = 1.5,
    mask_threshold: float = 0.5,
) -> ToyCutMaxSmokeResult:
    """Compute an approximate encrypted winner mask for a tiny packed vocab."""

    _validate_inputs(logits=logits, backend=backend, margin_scale=margin_scale)
    expected_argmax = max(range(len(logits)), key=lambda index: logits[index])
    sorted_logits = sorted(logits, reverse=True)
    winner_margin = sorted_logits[0] - sorted_logits[1] if len(sorted_logits) > 1 else 0.0

    started = time.perf_counter()
    logits_ct = backend.encrypt(logits)
    winner_mask_ct = backend.encrypt((1.0,) * backend.batch_size)
    for shift in range(1, len(logits)):
        diff = _sub(backend, logits_ct, backend.rotate(logits_ct, shift))
        gate = _smoothstep_gate(backend, diff, margin_scale=margin_scale)
        winner_mask_ct = backend.mul_ct(winner_mask_ct, gate)
    decoded_slots = backend.decrypt(winner_mask_ct, length=backend.batch_size)
    eval_seconds = time.perf_counter() - started

    decoded_mask = tuple(float(decoded_slots[index]) for index in range(len(logits)))
    selected_argmax = max(range(len(decoded_mask)), key=lambda index: decoded_mask[index])
    passed = selected_argmax == expected_argmax and decoded_mask[selected_argmax] > mask_threshold
    stats = backend.stats().to_json_dict()
    stats["batch_size"] = backend.batch_size
    return ToyCutMaxSmokeResult(
        logits=tuple(float(value) for value in logits),
        decoded_winner_mask=decoded_mask,
        expected_argmax=expected_argmax,
        selected_argmax=selected_argmax,
        winner_margin=winner_margin,
        margin_scale=margin_scale,
        passed=passed,
        backend_stats=stats,
        eval_seconds=eval_seconds,
    )


def required_toy_cutmax_rotations(vocab_size: int) -> tuple[int, ...]:
    """Return rotation steps required by the toy packed CutMax comparison."""

    if vocab_size <= 1:
        msg = "vocab_size must be greater than one"
        raise ValueError(msg)
    return tuple(range(1, vocab_size))


def payload_for_toy_cutmax_smoke(
    *,
    version: str,
    result: ToyCutMaxSmokeResult,
) -> dict[str, Any]:
    """Build an artifact payload for toy encrypted CutMax."""

    return {
        "version": version,
        "stage": "stage2-toy-encrypted-cutmax-smoke",
        "backend": result.backend_stats["backend"],
        "encrypted": bool(result.backend_stats["encrypted"]),
        "measurement_scope": {
            "toy_vocab_size": len(result.logits),
            "encrypted_cutmax": True,
            "encrypted_argmax": True,
            "client_side_argmax": False,
            "full_vocab_claimed": False,
            "full_model_correctness_claimed": False,
            "claim": (
                "Toy encrypted CutMax path over a tiny packed vocabulary. The backend "
                "computes an approximate winner mask using polynomial smoothstep "
                "comparisons; the client only decrypts the toy winner mask. This is "
                "not full-vocab generation evidence."
            ),
        },
        **result.to_json_dict(),
    }


def _smoothstep_gate(backend: FHEBackend, diff: Any, *, margin_scale: float) -> Any:
    # h(x) = 0.5 + 0.75x - 0.25x^3 maps [-1, 1] to [0, 1] with flat endpoints.
    scaled = backend.mul_plain(
        diff,
        backend.encode((1.0 / margin_scale,) * backend.batch_size),
    )
    squared = backend.mul_ct(scaled, scaled)
    cubed = backend.mul_ct(squared, scaled)
    linear = backend.mul_plain(scaled, backend.encode((0.75,) * backend.batch_size))
    cubic = backend.mul_plain(cubed, backend.encode((-0.25,) * backend.batch_size))
    half = backend.encrypt((0.5,) * backend.batch_size)
    return backend.add(half, backend.add(linear, cubic))


def _sub(backend: FHEBackend, left: Any, right: Any) -> Any:
    neg_one = backend.encode((-1.0,) * backend.batch_size)
    return backend.add(left, backend.mul_plain(right, neg_one))


def _validate_inputs(
    *,
    logits: tuple[float, ...],
    backend: FHEBackend,
    margin_scale: float,
) -> None:
    if len(logits) <= 1:
        msg = "logits must contain at least two values"
        raise ValueError(msg)
    if len(logits) != backend.batch_size:
        msg = f"logits length {len(logits)} must match backend batch_size={backend.batch_size}"
        raise ValueError(msg)
    if margin_scale <= 0:
        msg = "margin_scale must be positive"
        raise ValueError(msg)
    max_gap = max(abs(a - b) for a in logits for b in logits)
    if max_gap > margin_scale:
        msg = "all pairwise logit differences must fit within margin_scale"
        raise ValueError(msg)


__all__ = [
    "ToyCutMaxSmokeResult",
    "payload_for_toy_cutmax_smoke",
    "required_toy_cutmax_rotations",
    "run_toy_cutmax_smoke",
]
