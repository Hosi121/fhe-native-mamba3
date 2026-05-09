"""Decoding-path policy for encrypted generation."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Literal

DecodingMode = Literal["client-side", "encrypted-argmax", "scoring"]


@dataclass(frozen=True)
class DecodingPolicy:
    """Project policy for a decoding mode."""

    mode: DecodingMode
    interactive: bool
    encrypted_argmax: bool
    stage0_blocker: bool
    status: str
    notes: tuple[str, ...]

    def to_json_dict(self) -> dict[str, Any]:
        return asdict(self)


def decoding_policies() -> tuple[DecodingPolicy, ...]:
    """Return supported/planned decoding policies."""

    return (
        DecodingPolicy(
            mode="client-side",
            interactive=True,
            encrypted_argmax=False,
            stage0_blocker=False,
            status="default",
            notes=(
                "Server returns encrypted logits or scores; client decrypts and chooses the token.",
                "This is the default generation path for Stage 0/1.",
            ),
        ),
        DecodingPolicy(
            mode="encrypted-argmax",
            interactive=False,
            encrypted_argmax=True,
            stage0_blocker=False,
            status="research-branch",
            notes=(
                "Requires comparison/sign polynomial circuits over vocab scores.",
                "Tracked separately because it can dominate depth and latency.",
            ),
        ),
        DecodingPolicy(
            mode="scoring",
            interactive=False,
            encrypted_argmax=False,
            stage0_blocker=False,
            status="supported-task-scope",
            notes=(
                "Use encrypted scoring/classification instead of autoregressive sampling.",
                "Useful fallback if generation latency is dominated by decoding.",
            ),
        ),
    )


def get_decoding_policy(mode: DecodingMode) -> DecodingPolicy:
    """Return one decoding policy by mode."""

    for policy in decoding_policies():
        if policy.mode == mode:
            return policy
    msg = f"unknown decoding mode: {mode}"
    raise ValueError(msg)


def client_side_argmax(scores: tuple[float, ...] | list[float]) -> int:
    """Plain client-side argmax after decryption."""

    if not scores:
        msg = "scores must be non-empty"
        raise ValueError(msg)
    return max(range(len(scores)), key=lambda index: scores[index])
