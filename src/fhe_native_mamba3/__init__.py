"""FHE-native Mamba-3 MIMO research prototype."""

__version__ = "0.2.2"

_LAZY_IMPORTS = {
    "BootstrapSchedule": ("fhe_native_mamba3.cost", "BootstrapSchedule"),
    "CkksConfig": ("fhe_native_mamba3.ckks", "CkksConfig"),
    "CkksTrace": ("fhe_native_mamba3.ckks", "CkksTrace"),
    "BackendStats": ("fhe_native_mamba3.backends.base", "BackendStats"),
    "BackendCapability": ("fhe_native_mamba3.backends.capabilities", "BackendCapability"),
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
    "Stage0SweepConfig": ("fhe_native_mamba3.benchmarks.stage0_sweep", "Stage0SweepConfig"),
    "TrackingBackend": ("fhe_native_mamba3.backends.tracking", "TrackingBackend"),
    "WeightCalibration": ("fhe_native_mamba3.weight_encoding", "WeightCalibration"),
    "WeightEncodingConfig": ("fhe_native_mamba3.weight_encoding", "WeightEncodingConfig"),
    "backend_capability_matrix": (
        "fhe_native_mamba3.backends.capabilities",
        "backend_capability_matrix",
    ),
    "build_rotation_inventory": (
        "fhe_native_mamba3.rotation_inventory",
        "build_rotation_inventory",
    ),
    "calibrate_weight_values": (
        "fhe_native_mamba3.weight_encoding",
        "calibrate_weight_values",
    ),
    "client_side_argmax": ("fhe_native_mamba3.decoding", "client_side_argmax"),
    "decoding_policies": ("fhe_native_mamba3.decoding", "decoding_policies"),
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
    "run_stage0_sweep": ("fhe_native_mamba3.benchmarks.stage0_sweep", "run_stage0_sweep"),
}

__all__ = [
    "BackendCapability",
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
    "Stage0SweepConfig",
    "TrackingBackend",
    "WeightCalibration",
    "WeightEncodingConfig",
    "__version__",
    "backend_capability_matrix",
    "build_rotation_inventory",
    "calibrate_weight_values",
    "client_side_argmax",
    "decoding_policies",
    "estimate_block_cost",
    "estimate_integrated_cost",
    "greedy_bootstrap_schedule",
    "make_demo_problem",
    "plaintext_static_recurrence",
    "run_openfhe_static_recurrence",
    "run_stage0_mimo",
    "run_stage0_sweep",
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
