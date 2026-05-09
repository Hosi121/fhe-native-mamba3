"""Backend capability matrix."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class BackendCapability:
    """Static capability description for a backend candidate."""

    name: str
    role: str
    encrypted_execution: bool
    ckks: bool
    gpu: bool
    bootstrap: bool
    openfhe_interop: bool
    status: str
    notes: tuple[str, ...]

    def to_json_dict(self) -> dict[str, Any]:
        return asdict(self)


def known_backend_capabilities() -> tuple[BackendCapability, ...]:
    """Return the project backend matrix."""

    return (
        BackendCapability(
            name="tracking",
            role="operation-count and correctness scaffold",
            encrypted_execution=False,
            ckks=False,
            gpu=False,
            bootstrap=False,
            openfhe_interop=False,
            status="implemented",
            notes=(
                "Runs plaintext values through the same backend protocol.",
                "Used for layout sweeps before expensive encrypted execution.",
            ),
        ),
        BackendCapability(
            name="openfhe-cpu",
            role="correctness baseline",
            encrypted_execution=True,
            ckks=True,
            gpu=False,
            bootstrap=True,
            openfhe_interop=True,
            status="implemented-for-stage0",
            notes=(
                "Python wheel is used for encrypted CKKS recurrence tests.",
                "GPU acceleration is not assumed for this backend.",
            ),
        ),
        BackendCapability(
            name="fideslib-gpu",
            role="GPU CKKS/bootstrap benchmark candidate",
            encrypted_execution=True,
            ckks=True,
            gpu=True,
            bootstrap=True,
            openfhe_interop=True,
            status="verified-on-b200-stage0-native",
            notes=(
                "Built against patched OpenFHE 1.4.2 on high/kra-120 with CUDA 12.8.",
                "GPU bootstrap smoke and repo-owned encrypted Stage 0 native MIMO recurrence pass.",
            ),
        ),
        BackendCapability(
            name="phantom-fhe",
            role="optional GPU primitive microbenchmark",
            encrypted_execution=True,
            ckks=True,
            gpu=True,
            bootstrap=False,
            openfhe_interop=False,
            status="deprioritized",
            notes=(
                "Not suitable for Stage 1 bootstrap scheduling measurements.",
                "Can still be useful for non-bootstrap primitive timing.",
            ),
        ),
    )


def backend_capability_matrix() -> list[dict[str, Any]]:
    """Return JSON-serializable backend capabilities."""

    return [capability.to_json_dict() for capability in known_backend_capabilities()]
