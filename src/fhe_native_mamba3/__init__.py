"""FHE-native Mamba-3 MIMO research prototype."""

__version__ = "0.2.55"

_LAZY_IMPORTS = {
    "BootstrapSchedule": ("fhe_native_mamba3.cost", "BootstrapSchedule"),
    "CkksConfig": ("fhe_native_mamba3.ckks", "CkksConfig"),
    "CkksTrace": ("fhe_native_mamba3.ckks", "CkksTrace"),
    "CheckpointInspection": ("fhe_native_mamba3.checkpoint", "CheckpointInspection"),
    "CheckpointTensorSpec": ("fhe_native_mamba3.checkpoint", "CheckpointTensorSpec"),
    "BackendStats": ("fhe_native_mamba3.backends.base", "BackendStats"),
    "WeightBundleRecurrenceProblem": (
        "fhe_native_mamba3.bundle_recurrence",
        "WeightBundleRecurrenceProblem",
    ),
    "MambaCheckpointAdapterReport": (
        "fhe_native_mamba3.mamba_checkpoint",
        "MambaCheckpointAdapterReport",
    ),
    "MambaCheckpointPlan": ("fhe_native_mamba3.mamba_checkpoint", "MambaCheckpointPlan"),
    "MambaLayerPlan": ("fhe_native_mamba3.mamba_checkpoint", "MambaLayerPlan"),
    "MambaLayerReferenceResult": (
        "fhe_native_mamba3.mamba_reference",
        "MambaLayerReferenceResult",
    ),
    "MambaSourceDeltaResult": (
        "fhe_native_mamba3.mamba_reference",
        "MambaSourceDeltaResult",
    ),
    "MambaSourceLayerDiagnostics": (
        "fhe_native_mamba3.mamba_reference",
        "MambaSourceLayerDiagnostics",
    ),
    "MambaStageRange": (
        "fhe_native_mamba3.mamba_reference",
        "MambaStageRange",
    ),
    "run_mamba_source_layer": (
        "fhe_native_mamba3.mamba_reference",
        "run_mamba_source_layer",
    ),
    "LayerRangeScalePlan": (
        "fhe_native_mamba3.range_calibration",
        "LayerRangeScalePlan",
    ),
    "RangeScalePlan": (
        "fhe_native_mamba3.range_calibration",
        "RangeScalePlan",
    ),
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
    "StateDictMappingReport": ("fhe_native_mamba3.state_dict_mapping", "StateDictMappingReport"),
    "StateDictMappingRule": ("fhe_native_mamba3.state_dict_mapping", "StateDictMappingRule"),
    "StateDictMappingDraft": ("fhe_native_mamba3.state_dict_mapping", "StateDictMappingDraft"),
    "TrackingBackend": ("fhe_native_mamba3.backends.tracking", "TrackingBackend"),
    "WeightCalibration": ("fhe_native_mamba3.weight_encoding", "WeightCalibration"),
    "WeightBundleManifest": ("fhe_native_mamba3.weight_bundle", "WeightBundleManifest"),
    "WeightEncodingConfig": ("fhe_native_mamba3.weight_encoding", "WeightEncodingConfig"),
    "backend_capability_matrix": (
        "fhe_native_mamba3.backends.capabilities",
        "backend_capability_matrix",
    ),
    "build_rotation_inventory": (
        "fhe_native_mamba3.rotation_inventory",
        "build_rotation_inventory",
    ),
    "build_mamba_source_recurrence_problem": (
        "fhe_native_mamba3.mamba_reference",
        "build_mamba_source_recurrence_problem",
    ),
    "calibrate_weight_values": (
        "fhe_native_mamba3.weight_encoding",
        "calibrate_weight_values",
    ),
    "compare_mamba_layer_reference": (
        "fhe_native_mamba3.mamba_reference",
        "compare_mamba_layer_reference",
    ),
    "compare_mamba_source_delta": (
        "fhe_native_mamba3.mamba_reference",
        "compare_mamba_source_delta",
    ),
    "build_weight_bundle_manifest": (
        "fhe_native_mamba3.weight_bundle",
        "build_weight_bundle_manifest",
    ),
    "build_weight_bundle_recurrence_problem": (
        "fhe_native_mamba3.bundle_recurrence",
        "build_weight_bundle_recurrence_problem",
    ),
    "adapt_mamba_state_dict_to_model": (
        "fhe_native_mamba3.mamba_checkpoint",
        "adapt_mamba_state_dict_to_model",
    ),
    "calibrate_weight_tensor": (
        "fhe_native_mamba3.weight_encoding",
        "calibrate_weight_tensor",
    ),
    "ckks_batch_size_for_slots": (
        "fhe_native_mamba3.backends.openfhe",
        "ckks_batch_size_for_slots",
    ),
    "ckks_ring_dimension_for_batch_size": (
        "fhe_native_mamba3.backends.openfhe",
        "ckks_ring_dimension_for_batch_size",
    ),
    "client_side_argmax": ("fhe_native_mamba3.decoding", "client_side_argmax"),
    "decoding_policies": ("fhe_native_mamba3.decoding", "decoding_policies"),
    "draft_mapping_rules": ("fhe_native_mamba3.state_dict_mapping", "draft_mapping_rules"),
    "estimate_block_cost": ("fhe_native_mamba3.cost", "estimate_block_cost"),
    "estimate_integrated_cost": ("fhe_native_mamba3.cost", "estimate_integrated_cost"),
    "estimate_recurrence_depth": (
        "fhe_native_mamba3.recurrence_depth",
        "estimate_recurrence_depth",
    ),
    "build_recurrence_bootstrap_plan": (
        "fhe_native_mamba3.recurrence_depth",
        "build_recurrence_bootstrap_plan",
    ),
    "greedy_bootstrap_schedule": ("fhe_native_mamba3.cost", "greedy_bootstrap_schedule"),
    "inspect_checkpoint": ("fhe_native_mamba3.checkpoint", "inspect_checkpoint"),
    "load_mapping_rules": ("fhe_native_mamba3.state_dict_mapping", "load_mapping_rules"),
    "make_demo_problem": ("fhe_native_mamba3.openfhe_backend", "make_demo_problem"),
    "map_state_dict": ("fhe_native_mamba3.state_dict_mapping", "map_state_dict"),
    "plaintext_static_recurrence": (
        "fhe_native_mamba3.openfhe_backend",
        "plaintext_static_recurrence",
    ),
    "plaintext_recurrence_trace": (
        "fhe_native_mamba3.openfhe_backend",
        "plaintext_recurrence_trace",
    ),
    "plan_mamba_checkpoint": (
        "fhe_native_mamba3.mamba_checkpoint",
        "plan_mamba_checkpoint",
    ),
    "run_openfhe_static_recurrence": (
        "fhe_native_mamba3.openfhe_backend",
        "run_openfhe_static_recurrence",
    ),
    "scale_recurrence_state": (
        "fhe_native_mamba3.openfhe_backend",
        "scale_recurrence_state",
    ),
    "scale_recurrence_state_and_output": (
        "fhe_native_mamba3.openfhe_backend",
        "scale_recurrence_state_and_output",
    ),
    "run_stage0_mimo": ("fhe_native_mamba3.benchmarks.stage0_mimo", "run_stage0_mimo"),
    "run_stage0_sweep": ("fhe_native_mamba3.benchmarks.stage0_sweep", "run_stage0_sweep"),
    "load_weight_bundle_model": ("fhe_native_mamba3.weight_bundle", "load_weight_bundle_model"),
    "save_weight_bundle": ("fhe_native_mamba3.weight_bundle", "save_weight_bundle"),
    "save_weight_bundle_from_checkpoint": (
        "fhe_native_mamba3.weight_bundle",
        "save_weight_bundle_from_checkpoint",
    ),
    "save_weight_bundle_from_mapped_checkpoint": (
        "fhe_native_mamba3.weight_bundle",
        "save_weight_bundle_from_mapped_checkpoint",
    ),
    "save_mapping_draft": ("fhe_native_mamba3.state_dict_mapping", "save_mapping_draft"),
    "save_mamba_checkpoint_bundle": (
        "fhe_native_mamba3.mamba_checkpoint",
        "save_mamba_checkpoint_bundle",
    ),
}

__all__ = [
    "BackendCapability",
    "BackendStats",
    "BootstrapSchedule",
    "CheckpointInspection",
    "CheckpointTensorSpec",
    "CkksConfig",
    "CkksTrace",
    "FHEBackend",
    "FheCostEstimate",
    "FheMamba3Block",
    "FheMamba3Config",
    "FheMamba3ForCausalLM",
    "IntegratedCostEstimate",
    "MambaCheckpointAdapterReport",
    "MambaCheckpointPlan",
    "MambaLayerPlan",
    "MambaLayerReferenceResult",
    "MambaSourceDeltaResult",
    "OpenFheCkksBackend",
    "OpenFheRecurrenceProblem",
    "OpenFheRecurrenceResult",
    "PackingPlan",
    "Stage0MimoConfig",
    "Stage0SweepConfig",
    "StateDictMappingDraft",
    "StateDictMappingReport",
    "StateDictMappingRule",
    "TrackingBackend",
    "WeightBundleManifest",
    "WeightBundleRecurrenceProblem",
    "WeightCalibration",
    "WeightEncodingConfig",
    "__version__",
    "adapt_mamba_state_dict_to_model",
    "backend_capability_matrix",
    "build_mamba_source_recurrence_problem",
    "build_recurrence_bootstrap_plan",
    "build_rotation_inventory",
    "build_weight_bundle_manifest",
    "build_weight_bundle_recurrence_problem",
    "calibrate_weight_tensor",
    "calibrate_weight_values",
    "ckks_batch_size_for_slots",
    "ckks_ring_dimension_for_batch_size",
    "client_side_argmax",
    "compare_mamba_layer_reference",
    "compare_mamba_source_delta",
    "decoding_policies",
    "draft_mapping_rules",
    "estimate_block_cost",
    "estimate_integrated_cost",
    "estimate_recurrence_depth",
    "greedy_bootstrap_schedule",
    "inspect_checkpoint",
    "load_mapping_rules",
    "load_weight_bundle_model",
    "make_demo_problem",
    "map_state_dict",
    "plaintext_recurrence_trace",
    "plaintext_static_recurrence",
    "plan_mamba_checkpoint",
    "run_openfhe_static_recurrence",
    "run_stage0_mimo",
    "run_stage0_sweep",
    "save_mamba_checkpoint_bundle",
    "save_mapping_draft",
    "save_weight_bundle",
    "save_weight_bundle_from_checkpoint",
    "save_weight_bundle_from_mapped_checkpoint",
    "scale_recurrence_state",
    "scale_recurrence_state_and_output",
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
