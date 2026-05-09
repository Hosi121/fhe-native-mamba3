"""Common backend protocol for FHE execution."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Protocol


@dataclass
class BackendStats:
    """Operation counters emitted by every backend."""

    backend: str
    encrypted: bool
    encode_count: int = 0
    encrypt_count: int = 0
    decrypt_count: int = 0
    add_count: int = 0
    ct_pt_mul_count: int = 0
    ct_ct_mul_count: int = 0
    rotation_count: int = 0
    bootstrap_count: int = 0
    setup_seconds: float = 0.0
    eval_seconds: float = 0.0

    def to_json_dict(self) -> dict[str, Any]:
        return asdict(self)


class FHEBackend(Protocol):
    """Minimal interface required by Stage 0 MIMO recurrence."""

    @property
    def name(self) -> str: ...

    @property
    def encrypted(self) -> bool: ...

    @property
    def batch_size(self) -> int: ...

    @property
    def ring_dimension(self) -> int: ...

    def encode(self, values: list[float] | tuple[float, ...]) -> Any: ...

    def encrypt(self, values: list[float] | tuple[float, ...]) -> Any: ...

    def decrypt(self, value: Any, *, length: int) -> tuple[float, ...]: ...

    def add(self, left: Any, right: Any) -> Any: ...

    def mul_plain(self, ciphertext: Any, plaintext: Any) -> Any: ...

    def mul_ct(self, left: Any, right: Any) -> Any: ...

    def rotate(self, ciphertext: Any, steps: int) -> Any: ...

    def bootstrap(self, ciphertext: Any) -> Any: ...

    def stats(self) -> BackendStats: ...
