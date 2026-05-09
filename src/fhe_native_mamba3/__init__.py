"""FHE-native Mamba-3 MIMO research prototype."""

from fhe_native_mamba3.ckks import CkksConfig, CkksTrace, PackingPlan
from fhe_native_mamba3.cost import (
    BootstrapSchedule,
    FheCostEstimate,
    IntegratedCostEstimate,
    estimate_block_cost,
    estimate_integrated_cost,
    greedy_bootstrap_schedule,
)
from fhe_native_mamba3.model import (
    FheMamba3Block,
    FheMamba3Config,
    FheMamba3ForCausalLM,
)

__version__ = "0.2.0"

__all__ = [
    "BootstrapSchedule",
    "CkksConfig",
    "CkksTrace",
    "FheCostEstimate",
    "FheMamba3Block",
    "FheMamba3Config",
    "FheMamba3ForCausalLM",
    "IntegratedCostEstimate",
    "PackingPlan",
    "__version__",
    "estimate_block_cost",
    "estimate_integrated_cost",
    "greedy_bootstrap_schedule",
]
