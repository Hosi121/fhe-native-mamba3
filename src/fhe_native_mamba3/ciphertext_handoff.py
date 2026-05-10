"""Ciphertext handoff smoke for layer-boundary execution."""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass
from typing import Any

from fhe_native_mamba3.backends.base import FHEBackend


@dataclass(frozen=True)
class CiphertextHandoffLayer:
    """One square plaintext linear update applied to an encrypted hidden vector."""

    diagonals: tuple[tuple[float, ...], ...]
    residual_scale: float = 1.0
    bootstrap_after: bool = False

    @property
    def width(self) -> int:
        return len(self.diagonals)


@dataclass(frozen=True)
class CiphertextHandoffResult:
    """Result for a no-intermediate-decrypt ciphertext handoff chain."""

    input_values: tuple[float, ...]
    decrypted_output: tuple[float, ...]
    expected_output: tuple[float, ...]
    max_abs_error: float
    layer_count: int
    bootstrap_after_layers: tuple[int, ...]
    backend_stats: dict[str, Any]
    latency_sec: float
    latency_sec_per_layer: float

    def to_json_dict(self) -> dict[str, Any]:
        return asdict(self)


def required_handoff_rotations(width: int) -> tuple[int, ...]:
    """Rotations needed for the square cyclic diagonal method."""

    if width <= 0:
        msg = "width must be positive"
        raise ValueError(msg)
    return tuple(range(1, width))


def matrix_to_cyclic_diagonals(
    matrix: tuple[tuple[float, ...], ...],
) -> tuple[tuple[float, ...], ...]:
    """Convert a square row-major matrix to cyclic diagonals for backend rotations."""

    width = _validate_square_matrix(matrix)
    return tuple(
        tuple(float(matrix[row][(row + shift) % width]) for row in range(width))
        for shift in range(width)
    )


def run_ciphertext_handoff_chain(
    *,
    backend: FHEBackend,
    input_values: tuple[float, ...],
    layers: tuple[CiphertextHandoffLayer, ...],
) -> CiphertextHandoffResult:
    """Run a layer chain while decrypting only after the final layer."""

    if not input_values:
        msg = "input_values must not be empty"
        raise ValueError(msg)
    if not layers:
        msg = "layers must not be empty"
        raise ValueError(msg)
    width = len(input_values)
    if backend.batch_size != width:
        msg = (
            "ciphertext handoff smoke requires "
            f"batch_size={backend.batch_size} to equal width={width}"
        )
        raise ValueError(msg)
    for layer in layers:
        _validate_layer(layer, width)

    started = time.perf_counter()
    hidden_ct = backend.encrypt(input_values)
    bootstrap_after_layers: list[int] = []
    for layer_index, layer in enumerate(layers):
        update_ct = _apply_diagonal_linear(
            backend=backend,
            hidden_ct=hidden_ct,
            diagonals=layer.diagonals,
            width=width,
        )
        hidden_ct = backend.add(
            backend.mul_plain(
                hidden_ct,
                backend.encode(_padded_constant(layer.residual_scale, width, backend.batch_size)),
            ),
            update_ct,
        )
        if layer.bootstrap_after:
            hidden_ct = backend.bootstrap(hidden_ct)
            bootstrap_after_layers.append(layer_index + 1)

    decrypted = backend.decrypt(hidden_ct, length=width)
    latency_sec = time.perf_counter() - started
    backend.stats().eval_seconds += latency_sec
    expected = plaintext_handoff_chain(input_values=input_values, layers=layers)
    max_abs_error = max(
        (
            abs(actual - expected_value)
            for actual, expected_value in zip(decrypted, expected, strict=True)
        ),
        default=0.0,
    )
    return CiphertextHandoffResult(
        input_values=input_values,
        decrypted_output=decrypted,
        expected_output=expected,
        max_abs_error=max_abs_error,
        layer_count=len(layers),
        bootstrap_after_layers=tuple(bootstrap_after_layers),
        backend_stats=backend.stats().to_json_dict(),
        latency_sec=latency_sec,
        latency_sec_per_layer=latency_sec / len(layers),
    )


def plaintext_handoff_chain(
    *,
    input_values: tuple[float, ...],
    layers: tuple[CiphertextHandoffLayer, ...],
) -> tuple[float, ...]:
    """Plaintext reference for the handoff chain."""

    hidden = tuple(float(value) for value in input_values)
    width = len(hidden)
    for layer in layers:
        _validate_layer(layer, width)
        update = _plaintext_diagonal_linear(hidden, layer.diagonals)
        hidden = tuple(
            layer.residual_scale * hidden[index] + update[index] for index in range(width)
        )
    return hidden


def _apply_diagonal_linear(
    *,
    backend: FHEBackend,
    hidden_ct: Any,
    diagonals: tuple[tuple[float, ...], ...],
    width: int,
) -> Any:
    output_ct = backend.mul_plain(
        hidden_ct,
        backend.encode(_pad_values(diagonals[0], backend.batch_size)),
    )
    for shift, diagonal in enumerate(diagonals[1:], start=1):
        rotated = backend.rotate(hidden_ct, shift)
        term = backend.mul_plain(rotated, backend.encode(_pad_values(diagonal, backend.batch_size)))
        output_ct = backend.add(output_ct, term)
    return output_ct


def _plaintext_diagonal_linear(
    hidden: tuple[float, ...],
    diagonals: tuple[tuple[float, ...], ...],
) -> tuple[float, ...]:
    width = len(hidden)
    return tuple(
        sum(diagonals[shift][row] * hidden[(row + shift) % width] for shift in range(width))
        for row in range(width)
    )


def _validate_layer(layer: CiphertextHandoffLayer, width: int) -> None:
    if layer.width != width:
        msg = f"layer width={layer.width} does not match input width={width}"
        raise ValueError(msg)
    for diagonal in layer.diagonals:
        if len(diagonal) != width:
            msg = "each diagonal must match layer width"
            raise ValueError(msg)


def _validate_square_matrix(matrix: tuple[tuple[float, ...], ...]) -> int:
    if not matrix:
        msg = "matrix must not be empty"
        raise ValueError(msg)
    width = len(matrix)
    if any(len(row) != width for row in matrix):
        msg = "matrix must be square"
        raise ValueError(msg)
    return width


def _pad_values(values: tuple[float, ...], batch_size: int) -> list[float]:
    if len(values) > batch_size:
        msg = f"got {len(values)} values for batch_size={batch_size}"
        raise ValueError(msg)
    return [float(value) for value in values] + [0.0] * (batch_size - len(values))


def _padded_constant(value: float, width: int, batch_size: int) -> list[float]:
    if width > batch_size:
        msg = f"width={width} exceeds batch_size={batch_size}"
        raise ValueError(msg)
    return [float(value)] * width + [0.0] * (batch_size - width)
