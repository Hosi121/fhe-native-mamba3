"""OpenFHE CKKS backend for the minimal static MIMO recurrence."""

from __future__ import annotations

import random
from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class OpenFheRecurrenceProblem:
    """Plain inputs for an encrypted static MIMO recurrence run."""

    rank_inputs: tuple[tuple[float, ...], ...]
    decay: tuple[float, ...]
    b: tuple[tuple[float, ...], ...]
    c: tuple[tuple[float, ...], ...]

    @property
    def seq_len(self) -> int:
        return len(self.rank_inputs)

    @property
    def d_state(self) -> int:
        return len(self.b)

    @property
    def mimo_rank(self) -> int:
        return len(self.decay)


@dataclass(frozen=True)
class OpenFheRecurrenceResult:
    """Result of an encrypted OpenFHE recurrence run."""

    problem: OpenFheRecurrenceProblem
    decrypted_outputs: tuple[tuple[float, ...], ...]
    expected_outputs: tuple[tuple[float, ...], ...]
    max_abs_error: float
    ring_dimension: int
    batch_size: int
    multiplicative_depth: int
    rotations: tuple[int, ...]

    def to_json_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["problem"] = asdict(self.problem)
        return payload


def make_demo_problem(
    *,
    seq_len: int = 3,
    d_state: int = 2,
    mimo_rank: int = 2,
    seed: int = 7,
) -> OpenFheRecurrenceProblem:
    """Create a deterministic low-noise recurrence problem."""

    if seq_len <= 0 or d_state <= 0 or mimo_rank <= 0:
        msg = "seq_len, d_state, and mimo_rank must be positive"
        raise ValueError(msg)

    rng = random.Random(seed)
    rank_inputs = tuple(
        tuple(round(rng.uniform(-1.25, 1.25), 4) for _ in range(mimo_rank)) for _ in range(seq_len)
    )
    decay = tuple(round(0.35 + 0.1 * r + rng.uniform(-0.02, 0.02), 4) for r in range(mimo_rank))
    b = tuple(
        tuple(round(rng.uniform(-0.55, 0.55), 4) for _ in range(mimo_rank)) for _ in range(d_state)
    )
    c = tuple(
        tuple(round(rng.uniform(-0.55, 0.55), 4) for _ in range(mimo_rank)) for _ in range(d_state)
    )
    return OpenFheRecurrenceProblem(rank_inputs=rank_inputs, decay=decay, b=b, c=c)


def plaintext_static_recurrence(
    problem: OpenFheRecurrenceProblem,
) -> tuple[tuple[float, ...], ...]:
    """Reference static recurrence in plaintext."""

    state = [[0.0 for _ in range(problem.mimo_rank)] for _ in range(problem.d_state)]
    outputs: list[tuple[float, ...]] = []
    for rank_input in problem.rank_inputs:
        for n in range(problem.d_state):
            for r in range(problem.mimo_rank):
                state[n][r] = problem.decay[r] * state[n][r] + problem.b[n][r] * rank_input[r]
        outputs.append(
            tuple(
                sum(problem.c[n][r] * state[n][r] for n in range(problem.d_state))
                for r in range(problem.mimo_rank)
            )
        )
    return tuple(outputs)


def _flat_state_by_rank(
    matrix: tuple[tuple[float, ...], ...], d_state: int, rank: int
) -> list[float]:
    return [matrix[n][r] for r in range(rank) for n in range(d_state)]


def _expanded_rank_input(rank_input: tuple[float, ...], d_state: int, rank: int) -> list[float]:
    return [rank_input[r] for r in range(rank) for _ in range(d_state)]


def _readout_ciphertext(
    *,
    cc: Any,
    public_key: Any,
    contrib_ct: Any,
    d_state: int,
    rank: int,
) -> Any:
    slots = d_state * rank
    output_ct = cc.Encrypt(public_key, cc.MakeCKKSPackedPlaintext([0.0] * slots))
    for r in range(rank):
        for n in range(d_state):
            source = r * d_state + n
            target = r
            mask = [0.0] * slots
            mask[source] = 1.0
            term = cc.EvalMult(contrib_ct, cc.MakeCKKSPackedPlaintext(mask))
            shift = source - target
            if shift:
                term = cc.EvalRotate(term, shift)
            output_ct = cc.EvalAdd(output_ct, term)
    return output_ct


def run_openfhe_static_recurrence(
    problem: OpenFheRecurrenceProblem,
    *,
    multiplicative_depth: int | None = None,
    scaling_mod_size: int = 50,
) -> OpenFheRecurrenceResult:
    """Encrypt inputs and evaluate the static MIMO recurrence with OpenFHE CKKS."""

    try:
        from openfhe import (  # type: ignore[import-not-found]
            CCParamsCKKSRNS,
            GenCryptoContext,
            PKESchemeFeature,
        )
    except ImportError as exc:
        msg = "OpenFHE Python bindings are required. Install with: pip install '.[fhe]'"
        raise RuntimeError(msg) from exc

    d_state = problem.d_state
    rank = problem.mimo_rank
    slots = d_state * rank
    depth = multiplicative_depth or max(8, problem.seq_len + 5)

    params = CCParamsCKKSRNS()
    params.SetMultiplicativeDepth(depth)
    params.SetScalingModSize(scaling_mod_size)
    params.SetBatchSize(slots)
    cc = GenCryptoContext(params)
    cc.Enable(PKESchemeFeature.PKE)
    cc.Enable(PKESchemeFeature.KEYSWITCH)
    cc.Enable(PKESchemeFeature.LEVELEDSHE)

    keys = cc.KeyGen()
    cc.EvalMultKeyGen(keys.secretKey)
    rotations = tuple(
        sorted(
            {
                r * d_state + n - r
                for r in range(rank)
                for n in range(d_state)
                if r * d_state + n - r != 0
            }
        )
    )
    if rotations:
        cc.EvalRotateKeyGen(keys.secretKey, list(rotations))

    zero_pt = cc.MakeCKKSPackedPlaintext([0.0] * slots)
    state_ct = cc.Encrypt(keys.publicKey, zero_pt)
    decay_pt = cc.MakeCKKSPackedPlaintext(
        [problem.decay[r] for r in range(rank) for _ in range(d_state)]
    )
    b_pt = cc.MakeCKKSPackedPlaintext(_flat_state_by_rank(problem.b, d_state, rank))
    c_pt = cc.MakeCKKSPackedPlaintext(_flat_state_by_rank(problem.c, d_state, rank))

    decrypted_outputs: list[tuple[float, ...]] = []
    for rank_input in problem.rank_inputs:
        input_ct = cc.Encrypt(
            keys.publicKey,
            cc.MakeCKKSPackedPlaintext(_expanded_rank_input(rank_input, d_state, rank)),
        )
        state_ct = cc.EvalAdd(cc.EvalMult(state_ct, decay_pt), cc.EvalMult(input_ct, b_pt))
        contrib_ct = cc.EvalMult(state_ct, c_pt)
        output_ct = _readout_ciphertext(
            cc=cc,
            public_key=keys.publicKey,
            contrib_ct=contrib_ct,
            d_state=d_state,
            rank=rank,
        )
        decrypted = cc.Decrypt(output_ct, keys.secretKey)
        decrypted.SetLength(slots)
        values = decrypted.GetCKKSPackedValue()
        decrypted_outputs.append(tuple(float(values[r].real) for r in range(rank)))

    expected_outputs = plaintext_static_recurrence(problem)
    max_abs_error = max(
        abs(actual - expected)
        for actual_row, expected_row in zip(decrypted_outputs, expected_outputs, strict=True)
        for actual, expected in zip(actual_row, expected_row, strict=True)
    )

    return OpenFheRecurrenceResult(
        problem=problem,
        decrypted_outputs=tuple(decrypted_outputs),
        expected_outputs=expected_outputs,
        max_abs_error=max_abs_error,
        ring_dimension=cc.GetRingDimension(),
        batch_size=slots,
        multiplicative_depth=depth,
        rotations=rotations,
    )
