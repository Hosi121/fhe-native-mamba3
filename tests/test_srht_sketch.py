from __future__ import annotations

import json
from math import sqrt

import pytest
import torch

from fhe_native_mamba3.srht_sketch import (
    SrhtButterflyStage,
    apply_srht_sketch,
    build_srht_sketch_metadata,
    deterministic_rademacher_signs,
    normalized_walsh_hadamard,
    srht_sample_indices,
    srht_sampling_mask,
    srht_sketch_matrix,
    walsh_hadamard_butterfly_stages,
)


def test_deterministic_rademacher_signs_are_seeded_and_backend_neutral() -> None:
    signs = deterministic_rademacher_signs(8, seed=18, dtype=torch.float64)

    assert signs.tolist() == [-1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0, -1.0]
    assert torch.equal(signs, deterministic_rademacher_signs(8, seed=18, dtype=torch.float64))
    assert not torch.equal(signs, deterministic_rademacher_signs(8, seed=19, dtype=torch.float64))


def test_normalized_walsh_hadamard_matches_golden_matrix() -> None:
    values = torch.eye(4, dtype=torch.float64)
    transformed = normalized_walsh_hadamard(values)
    expected = torch.tensor(
        [
            [1.0, 1.0, 1.0, 1.0],
            [1.0, -1.0, 1.0, -1.0],
            [1.0, 1.0, -1.0, -1.0],
            [1.0, -1.0, -1.0, 1.0],
        ],
        dtype=torch.float64,
    ) / sqrt(4)

    assert torch.allclose(transformed, expected)
    assert torch.allclose(transformed @ transformed.T, torch.eye(4, dtype=torch.float64))


def test_srht_metadata_contains_fhe_friendly_golden_payload() -> None:
    metadata = build_srht_sketch_metadata(
        state_width=8,
        sketch_size=3,
        sign_seed=18,
        sample_seed=7,
    )

    assert metadata.signs == (-1, 1, -1, 1, -1, 1, -1, -1)
    assert metadata.sample_indices == (7, 0, 3)
    assert [stage.to_json_dict() for stage in metadata.butterfly_stages] == [
        {"stage_index": 0, "stride": 1},
        {"stage_index": 1, "stride": 2},
        {"stage_index": 2, "stride": 4},
    ]
    assert metadata.sampling_mask == (1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0)

    payload = metadata.to_json_dict()
    assert payload["normalization"] == "orthonormal"
    assert json.loads(metadata.to_json()) == payload


def test_srht_application_matches_explicit_sketch_matrix() -> None:
    metadata = build_srht_sketch_metadata(
        state_width=8,
        sketch_size=4,
        sign_seed=5,
        sample_seed=11,
        projection_scale=sqrt(8 / 4),
    )
    values = torch.arange(24, dtype=torch.float64).reshape(3, 8)

    sketched = apply_srht_sketch(values, metadata)
    matrix = srht_sketch_matrix(metadata, dtype=torch.float64)

    assert matrix.shape == (4, 8)
    assert torch.allclose(sketched, values @ matrix.T)


def test_sampling_helpers_reject_invalid_shapes_and_indices() -> None:
    with pytest.raises(ValueError, match="power of two"):
        normalized_walsh_hadamard(torch.zeros(3))

    with pytest.raises(ValueError, match="sketch_size cannot exceed"):
        srht_sample_indices(state_width=8, sketch_size=9, seed=1)

    with pytest.raises(ValueError, match="unique"):
        srht_sampling_mask(state_width=8, sample_indices=(1, 1))

    with pytest.raises(ValueError, match="outside"):
        srht_sampling_mask(state_width=8, sample_indices=(8,))

    with pytest.raises(ValueError, match="last dimension"):
        apply_srht_sketch(
            torch.zeros(2, 4),
            build_srht_sketch_metadata(
                state_width=8,
                sketch_size=2,
                sign_seed=1,
                sample_seed=1,
            ),
        )


def test_butterfly_stage_metadata_scales_by_powers_of_two() -> None:
    assert walsh_hadamard_butterfly_stages(16) == (
        SrhtButterflyStage(stage_index=0, stride=1),
        SrhtButterflyStage(stage_index=1, stride=2),
        SrhtButterflyStage(stage_index=2, stride=4),
        SrhtButterflyStage(stage_index=3, stride=8),
    )
