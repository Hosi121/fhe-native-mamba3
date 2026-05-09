"""Simple FHE operation estimates for the prototype block."""

from __future__ import annotations

from dataclasses import dataclass

from fhe_native_mamba3.model import FheMamba3Config


@dataclass(frozen=True)
class FheCostEstimate:
    """Approximate per-block inference cost for one encrypted sequence."""

    seq_len: int
    ciphertext_ciphertext_mul: int
    ciphertext_plaintext_mul: int
    additions: int
    multiplicative_depth: int
    notes: tuple[str, ...]

    @property
    def total_mul(self) -> int:
        return self.ciphertext_ciphertext_mul + self.ciphertext_plaintext_mul


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
    depth = 0
    notes: list[str] = []

    if config.bc_mode == "static":
        ct_pt += seq_len * (2 * d_state * rank)
        adds += seq_len * (2 * d_state * rank)
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
        multiplicative_depth=depth,
        notes=tuple(notes),
    )
