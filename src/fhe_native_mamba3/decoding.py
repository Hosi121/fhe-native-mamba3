"""Decoding-path policy for encrypted generation."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Literal

from fhe_native_mamba3.backends.base import FHEBackend

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


@dataclass(frozen=True)
class ClientSideDecodeResult:
    """Client-side token choice after decrypting one server output payload."""

    decoding_mode: str
    selected_token: int
    output_payload_width: int
    client_decrypt_count: int
    top1_score: float
    top2_score: float | None
    top1_top2_gap: float | None
    scores_abs_max: float

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


def client_side_decode_scores(
    scores: tuple[float, ...] | list[float],
    *,
    client_decrypt_count: int = 1,
) -> ClientSideDecodeResult:
    """Return token selection metadata for the interactive decoding baseline."""

    if client_decrypt_count < 0:
        msg = "client_decrypt_count must be non-negative"
        raise ValueError(msg)
    selected = client_side_argmax(scores)
    score_values = tuple(float(score) for score in scores)
    sorted_scores = sorted(score_values, reverse=True)
    top2_score = sorted_scores[1] if len(sorted_scores) > 1 else None
    return ClientSideDecodeResult(
        decoding_mode="client-side-argmax",
        selected_token=selected,
        output_payload_width=len(score_values),
        client_decrypt_count=client_decrypt_count,
        top1_score=score_values[selected],
        top2_score=top2_score,
        top1_top2_gap=(score_values[selected] - top2_score) if top2_score is not None else None,
        scores_abs_max=max(abs(score) for score in score_values),
    )


def client_side_decode_ciphertext(
    backend: FHEBackend,
    encrypted_scores: Any,
    *,
    output_payload_width: int,
) -> ClientSideDecodeResult:
    """Decrypt one server output payload and run the client-side baseline."""

    if output_payload_width <= 0:
        msg = "output_payload_width must be positive"
        raise ValueError(msg)
    scores = backend.decrypt(encrypted_scores, length=output_payload_width)
    return client_side_decode_scores(scores, client_decrypt_count=1)
