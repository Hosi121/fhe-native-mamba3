"""FHE-native Mamba-3 MIMO research prototype."""

from fhe_native_mamba3.cost import FheCostEstimate, estimate_block_cost
from fhe_native_mamba3.model import (
    FheMamba3Block,
    FheMamba3Config,
    FheMamba3ForCausalLM,
)

__version__ = "0.1.0"

__all__ = [
    "FheCostEstimate",
    "FheMamba3Block",
    "FheMamba3Config",
    "FheMamba3ForCausalLM",
    "__version__",
    "estimate_block_cost",
]
