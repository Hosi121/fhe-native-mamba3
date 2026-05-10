from __future__ import annotations

from collections.abc import Callable

import pytest

from fhe_native_mamba3.backends.openfhe import (
    _resolve_ring_dimension,
    ckks_batch_size_for_slots,
    ckks_ring_dimension_for_batch_size,
)
from fhe_native_mamba3.backends.tracking import TrackingBackend
from fhe_native_mamba3.openfhe_backend import (
    OpenFheRecurrenceProblem,
    make_demo_problem,
    plaintext_recurrence_trace,
    readout_output_slots,
    required_readout_rotations,
    required_recurrence_chain_rotations,
    run_openfhe_static_recurrence,
    run_static_mimo_recurrence_ciphertext_chain_with_backend,
    run_static_mimo_recurrence_ciphertexts_with_backend,
    run_static_mimo_recurrence_with_backend,
    scale_recurrence_state,
    scale_recurrence_state_and_output,
)

_RECURRENCE_RUNNERS: tuple[Callable[..., object], ...] = (
    run_static_mimo_recurrence_with_backend,
    run_static_mimo_recurrence_ciphertexts_with_backend,
)


def _validation_problem(**overrides: object) -> OpenFheRecurrenceProblem:
    params: dict[str, object] = {
        "rank_inputs": ((1.0, -2.0), (0.5, 0.25)),
        "decay": (0.1, 0.2),
        "b": ((0.25, -0.5),),
        "c": ((2.0, -1.0),),
    }
    params.update(overrides)
    return OpenFheRecurrenceProblem(**params)


def test_ckks_batch_size_rounds_to_power_of_two() -> None:
    assert ckks_batch_size_for_slots(1) == 1
    assert ckks_batch_size_for_slots(4) == 4
    assert ckks_batch_size_for_slots(18) == 32
    with pytest.raises(ValueError, match="positive"):
        ckks_batch_size_for_slots(0)


def test_ckks_ring_dimension_scales_with_batch_size() -> None:
    assert ckks_ring_dimension_for_batch_size(1) == 32768
    assert ckks_ring_dimension_for_batch_size(16384) == 32768
    assert ckks_ring_dimension_for_batch_size(32768) == 65536
    with pytest.raises(ValueError, match="positive"):
        ckks_ring_dimension_for_batch_size(0)


def test_explicit_openfhe_ring_dimension_is_validated() -> None:
    assert _resolve_ring_dimension(batch_size=16, ring_dimension=65536) == 65536
    with pytest.raises(ValueError, match="host"):
        _resolve_ring_dimension(batch_size=16, ring_dimension=16)
    with pytest.raises(ValueError, match="power"):
        _resolve_ring_dimension(batch_size=16, ring_dimension=65535)


def test_openfhe_static_recurrence_matches_plaintext() -> None:
    pytest.importorskip("openfhe")
    problem = make_demo_problem(seq_len=2, d_state=2, mimo_rank=2, seed=11)
    result = run_openfhe_static_recurrence(problem, multiplicative_depth=8)
    assert result.max_abs_error < 1e-6
    assert result.batch_size == 4
    assert result.rotations == (1, 2)


def test_readout_layout_metadata_distinguishes_dense_and_rank_local() -> None:
    assert required_readout_rotations(
        d_state=4,
        mimo_rank=4,
        readout_strategy="rank-reduce",
    ) == (1, 2, 3, 6, 9)
    assert required_readout_rotations(
        d_state=4,
        mimo_rank=4,
        readout_strategy="rank-local",
    ) == (1, 2)
    assert readout_output_slots(
        d_state=4,
        mimo_rank=4,
        readout_strategy="rank-local",
    ) == (0, 4, 8, 12)
    assert required_recurrence_chain_rotations(
        d_state=4,
        mimo_rank=4,
        readout_strategy="rank-local",
    ) == (-3, -2, -1, 1, 2)


def test_dynamic_decay_uses_ciphertext_multiply_path() -> None:
    problem = OpenFheRecurrenceProblem(
        rank_inputs=((1.0, -2.0), (0.5, 0.25)),
        decay=(0.1, 0.2),
        decay_by_token=((0.5, 0.6), (0.7, 0.8)),
        b=((0.25, -0.5),),
        c=((2.0, -1.0),),
    )

    result = run_static_mimo_recurrence_with_backend(
        problem,
        backend=TrackingBackend(batch_size=2),
        multiplicative_depth=8,
        readout_strategy="rank-local",
    )

    assert result.max_abs_error == 0
    assert result.backend_stats["ct_ct_mul_count"] == problem.seq_len


def test_encrypted_dynamic_bc_uses_ciphertext_multiply_path() -> None:
    problem = OpenFheRecurrenceProblem(
        rank_inputs=((1.0, -2.0), (0.5, 0.25)),
        decay=(0.1, 0.2),
        b=((0.0, 0.0),),
        c=((0.0, 0.0),),
        b_by_token=(
            ((0.25, -0.5),),
            ((0.75, 0.125),),
        ),
        c_by_token=(
            ((2.0, -1.0),),
            ((-0.25, 0.5),),
        ),
    )

    result = run_static_mimo_recurrence_with_backend(
        problem,
        backend=TrackingBackend(batch_size=2),
        multiplicative_depth=8,
        readout_strategy="rank-local",
        input_mode="encrypted-dynamic-bc",
    )

    assert result.max_abs_error == 0
    assert result.backend_stats["ct_ct_mul_count"] == 2 * problem.seq_len


def test_recurrence_runner_can_bootstrap_state_after_tokens() -> None:
    problem = OpenFheRecurrenceProblem(
        rank_inputs=((1.0, -2.0), (0.5, 0.25), (0.75, -0.5)),
        decay=(0.1, 0.2),
        b=((0.25, -0.5),),
        c=((2.0, -1.0),),
    )

    result = run_static_mimo_recurrence_with_backend(
        problem,
        backend=TrackingBackend(batch_size=2),
        multiplicative_depth=8,
        readout_strategy="rank-local",
        bootstrap_after_tokens=(1,),
        bootstrap_every_tokens=2,
    )

    assert result.max_abs_error == 0
    assert result.bootstrap_after_tokens == (1, 2)
    assert result.backend_stats["bootstrap_count"] == 2


def test_ciphertext_recurrence_trace_can_handoff_without_decrypting() -> None:
    class NoDecryptTrackingBackend(TrackingBackend):
        def decrypt(self, value: object, *, length: int) -> tuple[float, ...]:
            raise AssertionError("handoff trace must not decrypt")

    backend = NoDecryptTrackingBackend(batch_size=4)
    first_problem = OpenFheRecurrenceProblem(
        rank_inputs=((1.0, -2.0), (0.5, 0.25)),
        decay=(0.0, 0.0),
        b=((1.0, 1.0), (0.0, 0.0)),
        c=((1.0, 1.0), (0.0, 0.0)),
    )
    second_problem = OpenFheRecurrenceProblem(
        rank_inputs=((0.0, 0.0), (0.0, 0.0)),
        decay=(0.0, 0.0),
        b=((0.5, -0.25), (1.5, 0.75)),
        c=((2.0, 3.0), (1.0, 2.0)),
    )

    first = run_static_mimo_recurrence_ciphertexts_with_backend(
        first_problem,
        backend=backend,
        multiplicative_depth=8,
        readout_strategy="rank-local",
        input_mode="server-bx",
        bootstrap_after_tokens=(1,),
        output_layout="expanded-rank-input",
    )
    second = run_static_mimo_recurrence_ciphertexts_with_backend(
        second_problem,
        backend=backend,
        multiplicative_depth=8,
        readout_strategy="rank-local",
        input_mode="server-bx",
        rank_input_ciphertexts=first.output_ciphertexts,
    )

    assert first.bootstrap_after_tokens == (1,)
    assert first.output_layout == "expanded-rank-input"
    assert first.output_slots == (0, 2)
    assert first.rotations == required_recurrence_chain_rotations(
        d_state=2,
        mimo_rank=2,
        readout_strategy="rank-local",
    )
    assert first.layout_contract.output_layout == "expanded-rank-input"
    assert first.layout_contract.required_rotations == first.rotations
    assert first.output_ciphertexts.layout_contract == first.layout_contract
    assert second.output_slots == (0, 2)
    assert backend.stats().decrypt_count == 0
    assert backend.stats().bootstrap_count == 1


def test_readout_trace_ciphertexts_cannot_be_used_as_rank_input_handoff() -> None:
    backend = TrackingBackend(batch_size=4)
    problem = OpenFheRecurrenceProblem(
        rank_inputs=((1.0, -2.0), (0.5, 0.25)),
        decay=(0.0, 0.0),
        b=((1.0, 1.0), (0.0, 0.0)),
        c=((1.0, 1.0), (0.0, 0.0)),
    )
    readout_trace = run_static_mimo_recurrence_ciphertexts_with_backend(
        problem,
        backend=backend,
        multiplicative_depth=8,
        readout_strategy="rank-local",
        input_mode="server-bx",
        output_layout="readout",
    )

    assert readout_trace.output_layout == "readout"
    assert readout_trace.rotations == required_readout_rotations(
        d_state=2,
        mimo_rank=2,
        readout_strategy="rank-local",
    )
    with pytest.raises(ValueError, match="expanded-rank-input"):
        run_static_mimo_recurrence_ciphertexts_with_backend(
            problem,
            backend=backend,
            multiplicative_depth=8,
            readout_strategy="rank-local",
            input_mode="server-bx",
            rank_input_ciphertexts=readout_trace.output_ciphertexts,
        )


def test_rank_local_handoff_expands_readout_slots_for_d_state_greater_than_one() -> None:
    backend = TrackingBackend(batch_size=4)
    first_problem = OpenFheRecurrenceProblem(
        rank_inputs=((2.0, 4.0),),
        decay=(0.0, 0.0),
        b=((1.0, 1.0), (0.0, 0.0)),
        c=((1.0, 1.0), (0.0, 0.0)),
    )
    second_problem = OpenFheRecurrenceProblem(
        rank_inputs=((2.0, 4.0),),
        decay=(0.0, 0.0),
        b=((1.0, 1.0), (1.0, 1.0)),
        c=((1.0, 1.0), (1.0, 1.0)),
    )

    first = run_static_mimo_recurrence_ciphertexts_with_backend(
        first_problem,
        backend=backend,
        multiplicative_depth=8,
        readout_strategy="rank-local",
        input_mode="server-bx",
        output_layout="expanded-rank-input",
    )
    second = run_static_mimo_recurrence_with_backend(
        second_problem,
        backend=backend,
        multiplicative_depth=8,
        readout_strategy="rank-local",
        input_mode="server-bx",
        rank_input_ciphertexts=first.output_ciphertexts,
    )

    assert second.decrypted_outputs == ((4.0, 8.0),)
    assert second.max_abs_error == 0


def test_ciphertext_recurrence_chain_runs_without_intermediate_decrypts() -> None:
    backend = TrackingBackend(batch_size=4)
    first_problem = OpenFheRecurrenceProblem(
        rank_inputs=((2.0, 4.0), (1.5, -2.0)),
        decay=(0.0, 0.0),
        b=((1.0, 1.0), (0.0, 0.0)),
        c=((1.0, 1.0), (0.0, 0.0)),
    )
    second_problem = OpenFheRecurrenceProblem(
        rank_inputs=((0.0, 0.0), (0.0, 0.0)),
        decay=(0.0, 0.0),
        b=((1.0, 1.0), (1.0, 1.0)),
        c=((1.0, 1.0), (1.0, 1.0)),
    )

    result = run_static_mimo_recurrence_ciphertext_chain_with_backend(
        (first_problem, second_problem),
        backend=backend,
        multiplicative_depth=8,
        readout_strategy="rank-local",
        input_mode="server-bx",
        bootstrap_after_layers=(1,),
    )

    assert result.decrypted_outputs == ((4.0, 8.0), (3.0, -4.0))
    assert result.expected_outputs == result.decrypted_outputs
    assert result.max_abs_error == 0
    assert result.ciphertext_chain is True
    assert result.encrypted_chain is False
    assert result.full_layer_correctness_claimed is False
    assert result.intermediate_decrypt_count == 0
    assert result.bootstrap_after_layers == (1,)
    assert result.backend_stats["decrypt_count"] == first_problem.seq_len
    assert result.backend_stats["bootstrap_count"] == first_problem.seq_len


def test_ciphertext_recurrence_chain_rejects_invalid_contracts() -> None:
    problem = make_demo_problem(seq_len=2, d_state=2, mimo_rank=2, seed=11)

    with pytest.raises(ValueError, match="problems must not be empty"):
        run_static_mimo_recurrence_ciphertext_chain_with_backend(
            (),
            backend=TrackingBackend(batch_size=4),
            multiplicative_depth=8,
        )
    with pytest.raises(ValueError, match="server-bx or encrypted-dynamic-bc"):
        run_static_mimo_recurrence_ciphertext_chain_with_backend(
            (problem,),
            backend=TrackingBackend(batch_size=4),
            multiplicative_depth=8,
            input_mode="client-update",
        )
    with pytest.raises(ValueError, match="bootstrap_after_layers"):
        run_static_mimo_recurrence_ciphertext_chain_with_backend(
            (problem, problem),
            backend=TrackingBackend(batch_size=4),
            multiplicative_depth=8,
            bootstrap_after_layers=(2,),
        )
    with pytest.raises(ValueError, match="share seq_len"):
        run_static_mimo_recurrence_ciphertext_chain_with_backend(
            (problem, make_demo_problem(seq_len=1, d_state=2, mimo_rank=2, seed=12)),
            backend=TrackingBackend(batch_size=4),
            multiplicative_depth=8,
        )


def test_client_update_trace_keeps_client_side_update_statistics() -> None:
    problem = OpenFheRecurrenceProblem(
        rank_inputs=((1.0, -2.0), (0.5, 0.25)),
        decay=(0.1, 0.2),
        b=((0.25, -0.5),),
        c=((2.0, -1.0),),
        d_skip=(0.5, 0.25),
    )

    result = run_static_mimo_recurrence_with_backend(
        problem,
        backend=TrackingBackend(batch_size=2),
        multiplicative_depth=8,
        readout_strategy="rank-local",
        input_mode="client-update",
    )

    assert result.max_abs_error == 0
    assert result.backend_stats["encrypt_count"] == 1 + 2 * problem.seq_len
    assert (
        result.client_plaintext_public_weight_multiplies
        == (problem.d_state * problem.mimo_rank + problem.mimo_rank) * problem.seq_len
    )


def test_recurrence_runner_rejects_invalid_bootstrap_token() -> None:
    problem = make_demo_problem(seq_len=2, d_state=2, mimo_rank=2, seed=11)

    with pytest.raises(ValueError, match="bootstrap_after_tokens"):
        run_static_mimo_recurrence_with_backend(
            problem,
            backend=TrackingBackend(batch_size=4),
            multiplicative_depth=8,
            bootstrap_after_tokens=(3,),
        )


@pytest.mark.parametrize("runner", _RECURRENCE_RUNNERS)
def test_recurrence_runners_reject_bad_decay_by_token_length(
    runner: Callable[..., object],
) -> None:
    problem = _validation_problem(decay_by_token=((0.5, 0.6),))

    with pytest.raises(ValueError, match="decay_by_token length must match seq_len"):
        runner(
            problem,
            backend=TrackingBackend(batch_size=2),
            multiplicative_depth=8,
            readout_strategy="rank-local",
        )


@pytest.mark.parametrize(
    ("field_name", "value", "message"),
    [
        (
            "decay_state_by_token",
            (((0.5, 0.6), (0.7, 0.8)), ((0.9, 1.0),)),
            r"decay_state_by_token\[0\] must have d_state=1 rows",
        ),
        (
            "b_by_token",
            (((0.25, -0.5),),),
            "b_by_token length must match seq_len",
        ),
        (
            "c_by_token",
            (((2.0,),), ((-1.0,),)),
            r"each c_by_token\[0\] row must match mimo_rank=2",
        ),
    ],
)
@pytest.mark.parametrize("runner", _RECURRENCE_RUNNERS)
def test_recurrence_runners_validate_token_matrices(
    runner: Callable[..., object],
    field_name: str,
    value: object,
    message: str,
) -> None:
    problem = _validation_problem(**{field_name: value})

    with pytest.raises(ValueError, match=message):
        runner(
            problem,
            backend=TrackingBackend(batch_size=2),
            multiplicative_depth=8,
            readout_strategy="rank-local",
        )


@pytest.mark.parametrize("runner", _RECURRENCE_RUNNERS)
def test_recurrence_runners_reject_rank_input_ciphertext_length(
    runner: Callable[..., object],
) -> None:
    problem = _validation_problem()

    with pytest.raises(ValueError, match="rank_input_ciphertexts length must match seq_len"):
        runner(
            problem,
            backend=TrackingBackend(batch_size=2),
            multiplicative_depth=8,
            readout_strategy="rank-local",
            input_mode="server-bx",
            rank_input_ciphertexts=(object(),),
        )


@pytest.mark.parametrize("runner", _RECURRENCE_RUNNERS)
def test_recurrence_runners_reject_rank_input_ciphertexts_with_client_update(
    runner: Callable[..., object],
) -> None:
    problem = _validation_problem()

    with pytest.raises(
        ValueError,
        match="rank_input_ciphertexts require server-bx or encrypted-dynamic-bc input mode",
    ):
        runner(
            problem,
            backend=TrackingBackend(batch_size=2),
            multiplicative_depth=8,
            readout_strategy="rank-local",
            input_mode="client-update",
            rank_input_ciphertexts=(object(), object()),
        )


def test_ciphertext_recurrence_trace_rejects_invalid_output_layout() -> None:
    problem = _validation_problem()

    with pytest.raises(ValueError, match="unsupported output_layout: dense"):
        run_static_mimo_recurrence_ciphertexts_with_backend(
            problem,
            backend=TrackingBackend(batch_size=2),
            multiplicative_depth=8,
            readout_strategy="rank-local",
            output_layout="dense",
        )


def test_state_rank_decay_adds_ciphertext_multiply_path() -> None:
    problem = OpenFheRecurrenceProblem(
        rank_inputs=((1.0, -2.0), (0.5, 0.25)),
        decay=(0.1, 0.2),
        decay_state_by_token=(
            ((0.5, 0.6),),
            ((0.7, 0.8),),
        ),
        b=((0.25, -0.5),),
        c=((2.0, -1.0),),
    )

    result = run_static_mimo_recurrence_with_backend(
        problem,
        backend=TrackingBackend(batch_size=2),
        multiplicative_depth=8,
        readout_strategy="rank-local",
    )

    assert result.max_abs_error == 0
    assert result.backend_stats["ct_ct_mul_count"] == problem.seq_len


def test_state_scale_preserves_outputs_and_reduces_plain_state_range() -> None:
    problem = OpenFheRecurrenceProblem(
        rank_inputs=((2.0, -1.0), (1.5, 0.25)),
        decay=(0.8, 0.7),
        b=((3.0, -2.0), (1.0, 4.0)),
        c=((0.5, -1.0), (2.0, 0.25)),
    )

    scaled = scale_recurrence_state(problem, 0.125)
    original_result = run_static_mimo_recurrence_with_backend(
        problem,
        backend=TrackingBackend(batch_size=4),
        multiplicative_depth=8,
        readout_strategy="rank-local",
    )
    scaled_result = run_static_mimo_recurrence_with_backend(
        scaled,
        backend=TrackingBackend(batch_size=4),
        multiplicative_depth=8,
        readout_strategy="rank-local",
    )

    assert scaled_result.expected_outputs == original_result.expected_outputs
    assert scaled_result.max_abs_error == 0
    assert plaintext_recurrence_trace(scaled)["state_abs_max"] == pytest.approx(
        0.125 * plaintext_recurrence_trace(problem)["state_abs_max"]
    )


def test_state_and_output_scale_bounds_c_weights_and_scales_outputs() -> None:
    problem = OpenFheRecurrenceProblem(
        rank_inputs=((2.0, -1.0), (1.5, 0.25)),
        decay=(0.8, 0.7),
        b=((3.0, -2.0), (1.0, 4.0)),
        c=((0.5, -1.0), (2.0, 0.25)),
        d_skip=(0.5, -0.25),
    )

    scaled = scale_recurrence_state_and_output(
        problem,
        state_scale=0.125,
        output_scale=0.25,
    )
    original_result = run_static_mimo_recurrence_with_backend(
        problem,
        backend=TrackingBackend(batch_size=4),
        multiplicative_depth=8,
        readout_strategy="rank-local",
    )
    scaled_result = run_static_mimo_recurrence_with_backend(
        scaled,
        backend=TrackingBackend(batch_size=4),
        multiplicative_depth=8,
        readout_strategy="rank-local",
    )

    for actual_row, expected_row in zip(
        scaled_result.expected_outputs,
        original_result.expected_outputs,
        strict=True,
    ):
        assert actual_row == pytest.approx(tuple(0.25 * value for value in expected_row))
    assert scaled_result.max_abs_error == 0
    assert scaled.c[0][0] == pytest.approx(2.0 * problem.c[0][0])
    assert scaled.d_skip == pytest.approx(tuple(0.25 * value for value in problem.d_skip))
