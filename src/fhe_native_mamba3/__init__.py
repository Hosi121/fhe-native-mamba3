"""FHE-native Mamba-3 MIMO research prototype."""

__version__ = "0.2.0"

_LAZY_IMPORTS = {
    "BootstrapSchedule": ("fhe_native_mamba3.cost", "BootstrapSchedule"),
    "CkksConfig": ("fhe_native_mamba3.ckks", "CkksConfig"),
    "CkksTrace": ("fhe_native_mamba3.ckks", "CkksTrace"),
    "BackendStats": ("fhe_native_mamba3.backends.base", "BackendStats"),
    "FHEBackend": ("fhe_native_mamba3.backends.base", "FHEBackend"),
    "FheCostEstimate": ("fhe_native_mamba3.cost", "FheCostEstimate"),
    "FheMamba3Block": ("fhe_native_mamba3.model", "FheMamba3Block"),
    "FheMamba3Config": ("fhe_native_mamba3.model", "FheMamba3Config"),
    "FheMamba3ForCausalLM": ("fhe_native_mamba3.model", "FheMamba3ForCausalLM"),
    "IntegratedCostEstimate": ("fhe_native_mamba3.cost", "IntegratedCostEstimate"),
    "OpenFheRecurrenceProblem": (
        "fhe_native_mamba3.openfhe_backend",
        "OpenFheRecurrenceProblem",
    ),
    "OpenFheRecurrenceResult": (
        "fhe_native_mamba3.openfhe_backend",
        "OpenFheRecurrenceResult",
    ),
    "OpenFheCkksBackend": ("fhe_native_mamba3.backends.openfhe", "OpenFheCkksBackend"),
    "PackingPlan": ("fhe_native_mamba3.ckks", "PackingPlan"),
    "Stage0MimoConfig": ("fhe_native_mamba3.benchmarks.stage0_mimo", "Stage0MimoConfig"),
    "TrackingBackend": ("fhe_native_mamba3.backends.tracking", "TrackingBackend"),
    "estimate_block_cost": ("fhe_native_mamba3.cost", "estimate_block_cost"),
    "estimate_integrated_cost": ("fhe_native_mamba3.cost", "estimate_integrated_cost"),
    "greedy_bootstrap_schedule": ("fhe_native_mamba3.cost", "greedy_bootstrap_schedule"),
    "make_demo_problem": ("fhe_native_mamba3.openfhe_backend", "make_demo_problem"),
    "plaintext_static_recurrence": (
        "fhe_native_mamba3.openfhe_backend",
        "plaintext_static_recurrence",
    ),
    "run_openfhe_static_recurrence": (
        "fhe_native_mamba3.openfhe_backend",
        "run_openfhe_static_recurrence",
    ),
    "run_stage0_mimo": ("fhe_native_mamba3.benchmarks.stage0_mimo", "run_stage0_mimo"),
}

__all__ = [
    "BackendStats",
    "BootstrapSchedule",
    "CkksConfig",
    "CkksTrace",
    "FHEBackend",
    "FheCostEstimate",
    "FheMamba3Block",
    "FheMamba3Config",
    "FheMamba3ForCausalLM",
    "IntegratedCostEstimate",
    "OpenFheCkksBackend",
    "OpenFheRecurrenceProblem",
    "OpenFheRecurrenceResult",
    "PackingPlan",
    "Stage0MimoConfig",
    "TrackingBackend",
    "__version__",
    "estimate_block_cost",
    "estimate_integrated_cost",
    "greedy_bootstrap_schedule",
    "make_demo_problem",
    "plaintext_static_recurrence",
    "run_openfhe_static_recurrence",
    "run_stage0_mimo",
]


def __getattr__(name: str) -> object:
    if name not in _LAZY_IMPORTS:
        msg = f"module {__name__!r} has no attribute {name!r}"
        raise AttributeError(msg)
    module_name, attr_name = _LAZY_IMPORTS[name]
    module = __import__(module_name, fromlist=[attr_name])
    value = getattr(module, attr_name)
    globals()[name] = value
    return value
