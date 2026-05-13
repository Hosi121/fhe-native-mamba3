"""FHE-native Mamba-3 MIMO research prototype."""

import re
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

_PACKAGE_NAME = "fhe-native-mamba3"


def _source_tree_version() -> str | None:
    pyproject = Path(__file__).resolve().parents[2] / "pyproject.toml"
    if not pyproject.exists():
        return None
    match = re.search(
        r'(?m)^version\s*=\s*"([^"]+)"\s*$',
        pyproject.read_text(encoding="utf-8"),
    )
    return match.group(1) if match else None


try:
    __version__ = _source_tree_version() or version(_PACKAGE_NAME)
except PackageNotFoundError:
    __version__ = "0.0.0+unknown"

_LAZY_IMPORTS = {
    "BootstrapSchedule": ("fhe_native_mamba3.cost", "BootstrapSchedule"),
    "BootstrapExecutionBlockCost": (
        "fhe_native_mamba3.bootstrap_schedule",
        "BootstrapExecutionBlockCost",
    ),
    "BootstrapExecutionPolicy": (
        "fhe_native_mamba3.bootstrap_schedule",
        "BootstrapExecutionPolicy",
    ),
    "BootstrapExecutionSchedule": (
        "fhe_native_mamba3.bootstrap_schedule",
        "BootstrapExecutionSchedule",
    ),
    "BootstrapExecutionScheduleStep": (
        "fhe_native_mamba3.bootstrap_schedule",
        "BootstrapExecutionScheduleStep",
    ),
    "CkksConfig": ("fhe_native_mamba3.ckks", "CkksConfig"),
    "CkksTrace": ("fhe_native_mamba3.ckks", "CkksTrace"),
    "CheckpointInspection": ("fhe_native_mamba3.checkpoint", "CheckpointInspection"),
    "CheckpointTensorSpec": ("fhe_native_mamba3.checkpoint", "CheckpointTensorSpec"),
    "CheckpointClientDecodeSmoke": (
        "fhe_native_mamba3.checkpoint_decode",
        "CheckpointClientDecodeSmoke",
    ),
    "ClientDecodeReport": (
        "fhe_native_mamba3.client_decode_report",
        "ClientDecodeReport",
    ),
    "ClientDecodeReportRow": (
        "fhe_native_mamba3.client_decode_report",
        "ClientDecodeReportRow",
    ),
    "CheckpointSourceProfile": (
        "fhe_native_mamba3.checkpoint_profile",
        "CheckpointSourceProfile",
    ),
    "CheckpointSourceProfileLayer": (
        "fhe_native_mamba3.checkpoint_profile",
        "CheckpointSourceProfileLayer",
    ),
    "CheckpointSourceSketchTrace": (
        "fhe_native_mamba3.checkpoint_sketch_trace",
        "CheckpointSourceSketchTrace",
    ),
    "CheckpointSketchMatrixResult": (
        "fhe_native_mamba3.checkpoint_sketch_matrix",
        "CheckpointSketchMatrixResult",
    ),
    "CheckpointSketchMatrixRow": (
        "fhe_native_mamba3.checkpoint_sketch_matrix",
        "CheckpointSketchMatrixRow",
    ),
    "SketchEvidenceReport": (
        "fhe_native_mamba3.sketch_evidence_report",
        "SketchEvidenceReport",
    ),
    "SketchEvidenceReportRow": (
        "fhe_native_mamba3.sketch_evidence_report",
        "SketchEvidenceReportRow",
    ),
    "SketchRecurrenceClaim": (
        "fhe_native_mamba3.sketch_recurrence_claims",
        "SketchRecurrenceClaim",
    ),
    "CheckpointRecurrenceCorrectnessGate": (
        "fhe_native_mamba3.checkpoint_correctness",
        "CheckpointRecurrenceCorrectnessGate",
    ),
    "CheckpointPreRecurrenceStageGate": (
        "fhe_native_mamba3.checkpoint_pre_recurrence",
        "CheckpointPreRecurrenceStageGate",
    ),
    "CheckpointPreRecurrenceChainGate": (
        "fhe_native_mamba3.checkpoint_pre_recurrence",
        "CheckpointPreRecurrenceChainGate",
    ),
    "CheckpointPreRecurrenceCiphertextTrace": (
        "fhe_native_mamba3.checkpoint_pre_recurrence",
        "CheckpointPreRecurrenceCiphertextTrace",
    ),
    "CheckpointPreRecurrenceGroupedCiphertextTrace": (
        "fhe_native_mamba3.checkpoint_pre_recurrence",
        "CheckpointPreRecurrenceGroupedCiphertextTrace",
    ),
    "CheckpointPreRecurrenceRankPackTrace": (
        "fhe_native_mamba3.checkpoint_pre_recurrence",
        "CheckpointPreRecurrenceRankPackTrace",
    ),
    "PRE_RECURRENCE_STAGES": (
        "fhe_native_mamba3.checkpoint_pre_recurrence",
        "PRE_RECURRENCE_STAGES",
    ),
    "CheckpointFullLayerCiphertextGate": (
        "fhe_native_mamba3.checkpoint_correctness",
        "CheckpointFullLayerCiphertextGate",
    ),
    "CheckpointFullLayerCiphertextTrace": (
        "fhe_native_mamba3.checkpoint_correctness",
        "CheckpointFullLayerCiphertextTrace",
    ),
    "CheckpointFullLayerCiphertextChainGate": (
        "fhe_native_mamba3.checkpoint_correctness",
        "CheckpointFullLayerCiphertextChainGate",
    ),
    "CheckpointEncryptedPreRecurrenceRecurrenceGate": (
        "fhe_native_mamba3.checkpoint_correctness",
        "CheckpointEncryptedPreRecurrenceRecurrenceGate",
    ),
    "CheckpointFullLayerSweepLayer": (
        "fhe_native_mamba3.checkpoint_full_layer_sweep",
        "CheckpointFullLayerSweepLayer",
    ),
    "CheckpointFullLayerSweepResult": (
        "fhe_native_mamba3.checkpoint_full_layer_sweep",
        "CheckpointFullLayerSweepResult",
    ),
    "CheckpointVisibleProjectionSweepResult": (
        "fhe_native_mamba3.checkpoint_visible_projection_sweep",
        "CheckpointVisibleProjectionSweepResult",
    ),
    "CheckpointVisibleProjectionSweepRow": (
        "fhe_native_mamba3.checkpoint_visible_projection_sweep",
        "CheckpointVisibleProjectionSweepRow",
    ),
    "ArtifactValidationIssue": (
        "fhe_native_mamba3.artifact_validation",
        "ArtifactValidationIssue",
    ),
    "ArtifactValidationResult": (
        "fhe_native_mamba3.artifact_validation",
        "ArtifactValidationResult",
    ),
    "BackendPrefixScanResult": (
        "fhe_native_mamba3.ssd_prefix_scan",
        "BackendPrefixScanResult",
    ),
    "BackendSrhtSmokeResult": (
        "fhe_native_mamba3.backend_srht",
        "BackendSrhtSmokeResult",
    ),
    "BackendSegmentedPrefixScanResult": (
        "fhe_native_mamba3.ssd_prefix_scan",
        "BackendSegmentedPrefixScanResult",
    ),
    "BackendAffinePrefixScanResult": (
        "fhe_native_mamba3.ssd_prefix_scan",
        "BackendAffinePrefixScanResult",
    ),
    "BackendPackedMimoReadoutResult": (
        "fhe_native_mamba3.ssd_prefix_scan",
        "BackendPackedMimoReadoutResult",
    ),
    "BackendStats": ("fhe_native_mamba3.backends.base", "BackendStats"),
    "BsgsSchedule": ("fhe_native_mamba3.stage1_state_major_layout", "BsgsSchedule"),
    "CiphertextHandoffLayer": (
        "fhe_native_mamba3.ciphertext_handoff",
        "CiphertextHandoffLayer",
    ),
    "CiphertextHandoffResult": (
        "fhe_native_mamba3.ciphertext_handoff",
        "CiphertextHandoffResult",
    ),
    "CiphertextLayoutContract": (
        "fhe_native_mamba3.openfhe_backend",
        "CiphertextLayoutContract",
    ),
    "apply_handoff_bootstrap_schedule": (
        "fhe_native_mamba3.ciphertext_handoff",
        "apply_handoff_bootstrap_schedule",
    ),
    "apply_lora_to_linear_modules": (
        "fhe_native_mamba3.range_finetune",
        "apply_lora_to_linear_modules",
    ),
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
    "MambaSourceVisibleHandoffTensors": (
        "fhe_native_mamba3.mamba_reference",
        "MambaSourceVisibleHandoffTensors",
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
    "LazyBootstrapReport": ("fhe_native_mamba3.lazy_bootstrap", "LazyBootstrapReport"),
    "LazyBootstrapScheduleRow": (
        "fhe_native_mamba3.lazy_bootstrap",
        "LazyBootstrapScheduleRow",
    ),
    "RangeScalePlan": (
        "fhe_native_mamba3.range_calibration",
        "RangeScalePlan",
    ),
    "LoRAConfig": ("fhe_native_mamba3.range_finetune", "LoRAConfig"),
    "LoRALinear": ("fhe_native_mamba3.range_finetune", "LoRALinear"),
    "RangeLossConfig": ("fhe_native_mamba3.range_finetune", "RangeLossConfig"),
    "RangeLossResult": ("fhe_native_mamba3.range_finetune", "RangeLossResult"),
    "RangeLossTerm": ("fhe_native_mamba3.range_finetune", "RangeLossTerm"),
    "RecurrenceTraceProfile": ("fhe_native_mamba3.profiling", "RecurrenceTraceProfile"),
    "encrypted_pre_recurrence_logical_batch_size": (
        "fhe_native_mamba3.checkpoint_pre_recurrence",
        "encrypted_pre_recurrence_logical_batch_size",
    ),
    "PackedPrefixScanPlan": (
        "fhe_native_mamba3.ssd_prefix_scan",
        "PackedPrefixScanPlan",
    ),
    "SsdPrefixScanResult": (
        "fhe_native_mamba3.ssd_prefix_scan",
        "SsdPrefixScanResult",
    ),
    "BackendCapability": ("fhe_native_mamba3.backends.capabilities", "BackendCapability"),
    "CompositeRotationBackend": (
        "fhe_native_mamba3.composite_rotation",
        "CompositeRotationBackend",
    ),
    "CompositeRotationEstimate": (
        "fhe_native_mamba3.composite_rotation",
        "CompositeRotationEstimate",
    ),
    "FHEBackend": ("fhe_native_mamba3.backends.base", "FHEBackend"),
    "FheCostEstimate": ("fhe_native_mamba3.cost", "FheCostEstimate"),
    "FheMamba3Block": ("fhe_native_mamba3.model", "FheMamba3Block"),
    "FheMamba3Config": ("fhe_native_mamba3.model", "FheMamba3Config"),
    "FheMamba3ForCausalLM": ("fhe_native_mamba3.model", "FheMamba3ForCausalLM"),
    "GreedyBootstrapSchedule": (
        "fhe_native_mamba3.bootstrap_schedule",
        "GreedyBootstrapSchedule",
    ),
    "HeadPackCandidate": ("fhe_native_mamba3.head_packing", "HeadPackCandidate"),
    "HeadPackSweep": ("fhe_native_mamba3.head_packing", "HeadPackSweep"),
    "IntegratedCostEstimate": ("fhe_native_mamba3.cost", "IntegratedCostEstimate"),
    "OpenFheRecurrenceProblem": (
        "fhe_native_mamba3.openfhe_backend",
        "OpenFheRecurrenceProblem",
    ),
    "OpenFheRecurrenceResult": (
        "fhe_native_mamba3.openfhe_backend",
        "OpenFheRecurrenceResult",
    ),
    "NumpyTrackingBackend": ("fhe_native_mamba3.backends.tracking", "NumpyTrackingBackend"),
    "OpenFheRecurrenceCiphertextTrace": (
        "fhe_native_mamba3.openfhe_backend",
        "OpenFheRecurrenceCiphertextTrace",
    ),
    "OpenFheRecurrenceCiphertextChainResult": (
        "fhe_native_mamba3.openfhe_backend",
        "OpenFheRecurrenceCiphertextChainResult",
    ),
    "OpenFheBootstrapConfig": ("fhe_native_mamba3.backends.openfhe", "OpenFheBootstrapConfig"),
    "OpenFheBootstrapLatencyConfig": (
        "fhe_native_mamba3.bootstrap_latency",
        "OpenFheBootstrapLatencyConfig",
    ),
    "OpenFheCkksBackend": ("fhe_native_mamba3.backends.openfhe", "OpenFheCkksBackend"),
    "OfficialMambaParityResult": (
        "fhe_native_mamba3.official_parity",
        "OfficialMambaParityResult",
    ),
    "PackingPlan": ("fhe_native_mamba3.ckks", "PackingPlan"),
    "Stage0MimoConfig": ("fhe_native_mamba3.benchmarks.stage0_mimo", "Stage0MimoConfig"),
    "Stage0SweepConfig": ("fhe_native_mamba3.benchmarks.stage0_sweep", "Stage0SweepConfig"),
    "Stage1CandidatePlan": ("fhe_native_mamba3.stage1_plan", "Stage1CandidatePlan"),
    "Stage1ComparisonReport": (
        "fhe_native_mamba3.stage1_report",
        "Stage1ComparisonReport",
    ),
    "Stage1ComparisonRow": ("fhe_native_mamba3.stage1_report", "Stage1ComparisonRow"),
    "Stage1CheckpointCostReport": (
        "fhe_native_mamba3.stage1_checkpoint_cost_report",
        "Stage1CheckpointCostReport",
    ),
    "Stage1CheckpointCostRow": (
        "fhe_native_mamba3.stage1_checkpoint_cost_report",
        "Stage1CheckpointCostRow",
    ),
    "Stage1Dependency": ("fhe_native_mamba3.stage1_plan", "Stage1Dependency"),
    "Stage1GroupedChainInventoryReport": (
        "fhe_native_mamba3.stage1_grouped_chain",
        "Stage1GroupedChainInventoryReport",
    ),
    "Stage1GroupedChainInventoryRow": (
        "fhe_native_mamba3.stage1_grouped_chain",
        "Stage1GroupedChainInventoryRow",
    ),
    "Stage1CheckpointGroupedGateInventoryReport": (
        "fhe_native_mamba3.stage1_checkpoint_grouped_gate",
        "Stage1CheckpointGroupedGateInventoryReport",
    ),
    "Stage1CheckpointGroupedGateInventoryRow": (
        "fhe_native_mamba3.stage1_checkpoint_grouped_gate",
        "Stage1CheckpointGroupedGateInventoryRow",
    ),
    "Stage1CompositeRotationReport": (
        "fhe_native_mamba3.stage1_composite_rotation_report",
        "Stage1CompositeRotationReport",
    ),
    "Stage1CompositeRotationRow": (
        "fhe_native_mamba3.stage1_composite_rotation_report",
        "Stage1CompositeRotationRow",
    ),
    "StateMajorLayoutPlan": (
        "fhe_native_mamba3.stage1_state_major_layout",
        "StateMajorLayoutPlan",
    ),
    "StateMajorFullShapeConfig": (
        "fhe_native_mamba3.stage1_state_major_fullshape",
        "StateMajorFullShapeConfig",
    ),
    "StateMajorFullShapeResult": (
        "fhe_native_mamba3.stage1_state_major_fullshape",
        "StateMajorFullShapeResult",
    ),
    "SlotBsgsSchedule": (
        "fhe_native_mamba3.stage1_state_major_layout",
        "SlotBsgsSchedule",
    ),
    "StateMajorToyKernelResult": (
        "fhe_native_mamba3.stage1_state_major_kernel",
        "StateMajorToyKernelResult",
    ),
    "StateMajorToyProblem": (
        "fhe_native_mamba3.stage1_state_major_kernel",
        "StateMajorToyProblem",
    ),
    "Stage1GroupedFullLayerLiftSmokeResult": (
        "fhe_native_mamba3.stage1_grouped_recurrence",
        "Stage1GroupedFullLayerLiftSmokeResult",
    ),
    "Stage1GroupedRecurrenceGroup": (
        "fhe_native_mamba3.stage1_grouped_recurrence",
        "Stage1GroupedRecurrenceGroup",
    ),
    "Stage1GroupedRecurrenceSmokeResult": (
        "fhe_native_mamba3.stage1_grouped_recurrence",
        "Stage1GroupedRecurrenceSmokeResult",
    ),
    "Stage1Plan": ("fhe_native_mamba3.stage1_plan", "Stage1Plan"),
    "Stage1PackSweepResult": (
        "fhe_native_mamba3.stage1_pack_sweep",
        "Stage1PackSweepResult",
    ),
    "Stage1PackSweepRow": (
        "fhe_native_mamba3.stage1_pack_sweep",
        "Stage1PackSweepRow",
    ),
    "Stage1ProfileHints": ("fhe_native_mamba3.stage1_plan", "Stage1ProfileHints"),
    "Stage2SketchSweepResult": (
        "fhe_native_mamba3.stage2_sketch_sweep",
        "Stage2SketchSweepResult",
    ),
    "Stage2SketchSweepRow": (
        "fhe_native_mamba3.stage2_sketch_sweep",
        "Stage2SketchSweepRow",
    ),
    "Stage2SketchSeedSample": (
        "fhe_native_mamba3.stage2_sketch_seed_sweep",
        "Stage2SketchSeedSample",
    ),
    "Stage2SketchSeedSweepResult": (
        "fhe_native_mamba3.stage2_sketch_seed_sweep",
        "Stage2SketchSeedSweepResult",
    ),
    "Stage2SketchSeedSweepRow": (
        "fhe_native_mamba3.stage2_sketch_seed_sweep",
        "Stage2SketchSeedSweepRow",
    ),
    "TinyMimoBlockProblem": (
        "fhe_native_mamba3.stage1_tiny_mimo",
        "TinyMimoBlockProblem",
    ),
    "TinyMimoBlockSmokeResult": (
        "fhe_native_mamba3.stage1_tiny_mimo",
        "TinyMimoBlockSmokeResult",
    ),
    "ToyCutMaxSmokeResult": (
        "fhe_native_mamba3.toy_cutmax",
        "ToyCutMaxSmokeResult",
    ),
    "StateDictMappingReport": ("fhe_native_mamba3.state_dict_mapping", "StateDictMappingReport"),
    "StateDictMappingRule": ("fhe_native_mamba3.state_dict_mapping", "StateDictMappingRule"),
    "StateDictMappingDraft": ("fhe_native_mamba3.state_dict_mapping", "StateDictMappingDraft"),
    "SrhtButterflyStage": ("fhe_native_mamba3.srht_sketch", "SrhtButterflyStage"),
    "SrhtSketchMetadata": ("fhe_native_mamba3.srht_sketch", "SrhtSketchMetadata"),
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
    "build_checkpoint_source_sketch_trace": (
        "fhe_native_mamba3.checkpoint_sketch_trace",
        "build_checkpoint_source_sketch_trace",
    ),
    "build_client_decode_report": (
        "fhe_native_mamba3.client_decode_report",
        "build_client_decode_report",
    ),
    "build_sketch_evidence_report": (
        "fhe_native_mamba3.sketch_evidence_report",
        "build_sketch_evidence_report",
    ),
    "classify_sketch_recurrence_claim": (
        "fhe_native_mamba3.sketch_recurrence_claims",
        "classify_sketch_recurrence_claim",
    ),
    "client_decode_report_markdown": (
        "fhe_native_mamba3.client_decode_report",
        "client_decode_report_markdown",
    ),
    "run_checkpoint_sketch_matrix": (
        "fhe_native_mamba3.checkpoint_sketch_matrix",
        "run_checkpoint_sketch_matrix",
    ),
    "resolve_rank_strategy": (
        "fhe_native_mamba3.checkpoint_sketch_matrix",
        "resolve_rank_strategy",
    ),
    "sketch_evidence_report_markdown": (
        "fhe_native_mamba3.sketch_evidence_report",
        "sketch_evidence_report_markdown",
    ),
    "build_bootstrap_execution_schedule": (
        "fhe_native_mamba3.bootstrap_schedule",
        "build_bootstrap_execution_schedule",
    ),
    "build_lazy_bootstrap_report": (
        "fhe_native_mamba3.lazy_bootstrap",
        "build_lazy_bootstrap_report",
    ),
    "build_packed_prefix_scan_plan": (
        "fhe_native_mamba3.ssd_prefix_scan",
        "build_packed_prefix_scan_plan",
    ),
    "build_prefix_scan_metadata": (
        "fhe_native_mamba3.ssd_prefix_scan",
        "build_prefix_scan_metadata",
    ),
    "build_mamba_source_recurrence_problem": (
        "fhe_native_mamba3.mamba_reference",
        "build_mamba_source_recurrence_problem",
    ),
    "build_mamba_source_visible_handoff_tensors": (
        "fhe_native_mamba3.mamba_reference",
        "build_mamba_source_visible_handoff_tensors",
    ),
    "calibrate_weight_values": (
        "fhe_native_mamba3.weight_encoding",
        "calibrate_weight_values",
    ),
    "backend_hillis_steele_prefix_products": (
        "fhe_native_mamba3.ssd_prefix_scan",
        "backend_hillis_steele_prefix_products",
    ),
    "backend_segmented_hillis_steele_prefix_products": (
        "fhe_native_mamba3.ssd_prefix_scan",
        "backend_segmented_hillis_steele_prefix_products",
    ),
    "backend_hillis_steele_affine_scan": (
        "fhe_native_mamba3.ssd_prefix_scan",
        "backend_hillis_steele_affine_scan",
    ),
    "backend_segmented_hillis_steele_affine_scan": (
        "fhe_native_mamba3.ssd_prefix_scan",
        "backend_segmented_hillis_steele_affine_scan",
    ),
    "backend_packed_static_mimo_readout": (
        "fhe_native_mamba3.ssd_prefix_scan",
        "backend_packed_static_mimo_readout",
    ),
    "backend_apply_srht_masked": (
        "fhe_native_mamba3.backend_srht",
        "backend_apply_srht_masked",
    ),
    "build_tiny_mimo_block_problem": (
        "fhe_native_mamba3.stage1_tiny_mimo",
        "build_tiny_mimo_block_problem",
    ),
    "causal_decay_weights": (
        "fhe_native_mamba3.ssd_prefix_scan",
        "causal_decay_weights",
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
    "current_git_commit": (
        "fhe_native_mamba3.artifact_validation",
        "current_git_commit",
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
    "client_side_decode_scores": (
        "fhe_native_mamba3.decoding",
        "client_side_decode_scores",
    ),
    "client_side_decode_ciphertext": (
        "fhe_native_mamba3.decoding",
        "client_side_decode_ciphertext",
    ),
    "composite_rotation_basis_for_steps": (
        "fhe_native_mamba3.composite_rotation",
        "composite_rotation_basis_for_steps",
    ),
    "decoding_policies": ("fhe_native_mamba3.decoding", "decoding_policies"),
    "decompose_rotation_steps": (
        "fhe_native_mamba3.composite_rotation",
        "decompose_rotation_steps",
    ),
    "draft_mapping_rules": ("fhe_native_mamba3.state_dict_mapping", "draft_mapping_rules"),
    "estimate_block_cost": ("fhe_native_mamba3.cost", "estimate_block_cost"),
    "estimate_integrated_cost": ("fhe_native_mamba3.cost", "estimate_integrated_cost"),
    "estimate_composite_rotation_basis": (
        "fhe_native_mamba3.composite_rotation",
        "estimate_composite_rotation_basis",
    ),
    "estimate_cumulative_log_contraction": (
        "fhe_native_mamba3.profiling",
        "estimate_cumulative_log_contraction",
    ),
    "estimate_high_decay_burst_len": (
        "fhe_native_mamba3.profiling",
        "estimate_high_decay_burst_len",
    ),
    "evaluate_head_pack_candidate": (
        "fhe_native_mamba3.head_packing",
        "evaluate_head_pack_candidate",
    ),
    "estimate_recurrence_depth": (
        "fhe_native_mamba3.recurrence_depth",
        "estimate_recurrence_depth",
    ),
    "estimate_recurrence_stack_latency": (
        "fhe_native_mamba3.recurrence_latency",
        "estimate_recurrence_stack_latency",
    ),
    "build_stage0_status_report": (
        "fhe_native_mamba3.stage0_status",
        "build_stage0_status_report",
    ),
    "build_stage1_comparison_report": (
        "fhe_native_mamba3.stage1_report",
        "build_stage1_comparison_report",
    ),
    "build_stage1_checkpoint_cost_report": (
        "fhe_native_mamba3.stage1_checkpoint_cost_report",
        "build_stage1_checkpoint_cost_report",
    ),
    "build_stage1_grouped_chain_inventory": (
        "fhe_native_mamba3.stage1_grouped_chain",
        "build_stage1_grouped_chain_inventory",
    ),
    "build_stage1_checkpoint_grouped_gate_inventory": (
        "fhe_native_mamba3.stage1_checkpoint_grouped_gate",
        "build_stage1_checkpoint_grouped_gate_inventory",
    ),
    "build_stage1_composite_rotation_report": (
        "fhe_native_mamba3.stage1_composite_rotation_report",
        "build_stage1_composite_rotation_report",
    ),
    "build_state_major_layout_plan": (
        "fhe_native_mamba3.stage1_state_major_layout",
        "build_state_major_layout_plan",
    ),
    "build_slot_bsgs_schedule": (
        "fhe_native_mamba3.stage1_state_major_layout",
        "build_slot_bsgs_schedule",
    ),
    "checkpoint_grouped_gate_rotation_steps": (
        "fhe_native_mamba3.stage1_checkpoint_grouped_gate",
        "checkpoint_grouped_gate_rotation_steps",
    ),
    "checkpoint_monolithic_gate_rotation_steps": (
        "fhe_native_mamba3.stage1_checkpoint_grouped_gate",
        "checkpoint_monolithic_gate_rotation_steps",
    ),
    "build_stage1_plan": ("fhe_native_mamba3.stage1_plan", "build_stage1_plan"),
    "build_fixed_bsgs_schedule": (
        "fhe_native_mamba3.stage1_state_major_layout",
        "build_fixed_bsgs_schedule",
    ),
    "build_srht_sketch_metadata": (
        "fhe_native_mamba3.srht_sketch",
        "build_srht_sketch_metadata",
    ),
    "measure_openfhe_bootstrap_latency": (
        "fhe_native_mamba3.bootstrap_latency",
        "measure_openfhe_bootstrap_latency",
    ),
    "make_state_major_toy_problem": (
        "fhe_native_mamba3.stage1_state_major_kernel",
        "make_state_major_toy_problem",
    ),
    "normalize_rotation_step": (
        "fhe_native_mamba3.composite_rotation",
        "normalize_rotation_step",
    ),
    "extract_stage1_profile_hints": (
        "fhe_native_mamba3.stage1_plan",
        "extract_stage1_profile_hints",
    ),
    "build_recurrence_bootstrap_plan": (
        "fhe_native_mamba3.recurrence_depth",
        "build_recurrence_bootstrap_plan",
    ),
    "greedy_bootstrap_schedule": (
        "fhe_native_mamba3.bootstrap_schedule",
        "greedy_bootstrap_schedule",
    ),
    "grouped_full_layer_lift_plaintext": (
        "fhe_native_mamba3.stage1_grouped_recurrence",
        "grouped_full_layer_lift_plaintext",
    ),
    "inspect_checkpoint": ("fhe_native_mamba3.checkpoint", "inspect_checkpoint"),
    "load_mapping_rules": ("fhe_native_mamba3.state_dict_mapping", "load_mapping_rules"),
    "make_demo_problem": ("fhe_native_mamba3.openfhe_backend", "make_demo_problem"),
    "make_demo_full_layer_lift_inputs": (
        "fhe_native_mamba3.stage1_grouped_recurrence",
        "make_demo_full_layer_lift_inputs",
    ),
    "lazy_bootstrap_markdown": (
        "fhe_native_mamba3.lazy_bootstrap",
        "lazy_bootstrap_markdown",
    ),
    "map_state_dict": ("fhe_native_mamba3.state_dict_mapping", "map_state_dict"),
    "packed_prefix_scan_rotation_steps": (
        "fhe_native_mamba3.ssd_prefix_scan",
        "packed_prefix_scan_rotation_steps",
    ),
    "packed_prefix_scan_carry_rotation_steps": (
        "fhe_native_mamba3.ssd_prefix_scan",
        "packed_prefix_scan_carry_rotation_steps",
    ),
    "packed_mimo_readout_output_slots": (
        "fhe_native_mamba3.ssd_prefix_scan",
        "packed_mimo_readout_output_slots",
    ),
    "payload_for_tiny_mimo_block_smoke": (
        "fhe_native_mamba3.stage1_tiny_mimo",
        "payload_for_tiny_mimo_block_smoke",
    ),
    "plaintext_static_recurrence": (
        "fhe_native_mamba3.openfhe_backend",
        "plaintext_static_recurrence",
    ),
    "power_of_two_rotation_basis": (
        "fhe_native_mamba3.composite_rotation",
        "power_of_two_rotation_basis",
    ),
    "plaintext_recurrence_trace": (
        "fhe_native_mamba3.openfhe_backend",
        "plaintext_recurrence_trace",
    ),
    "prefix_decay_products": (
        "fhe_native_mamba3.ssd_prefix_scan",
        "prefix_decay_products",
    ),
    "profile_model_batch": ("fhe_native_mamba3.profiling", "profile_model_batch"),
    "profile_recurrence_traces": (
        "fhe_native_mamba3.profiling",
        "profile_recurrence_traces",
    ),
    "range_loss": ("fhe_native_mamba3.range_finetune", "range_loss"),
    "resolve_pre_recurrence_shape": (
        "fhe_native_mamba3.checkpoint_pre_recurrence",
        "resolve_pre_recurrence_shape",
    ),
    "fhe_aware_loss": ("fhe_native_mamba3.range_finetune", "fhe_aware_loss"),
    "group_checkpoint_pre_recurrence_trace_by_rank": (
        "fhe_native_mamba3.checkpoint_pre_recurrence",
        "group_checkpoint_pre_recurrence_trace_by_rank",
    ),
    "mark_only_lora_trainable": (
        "fhe_native_mamba3.range_finetune",
        "mark_only_lora_trainable",
    ),
    "lora_parameter_count": (
        "fhe_native_mamba3.range_finetune",
        "lora_parameter_count",
    ),
    "profile_checkpoint_source_layers": (
        "fhe_native_mamba3.checkpoint_profile",
        "profile_checkpoint_source_layers",
    ),
    "run_checkpoint_client_decode_smoke": (
        "fhe_native_mamba3.checkpoint_decode",
        "run_checkpoint_client_decode_smoke",
    ),
    "probe_official_mamba_parity": (
        "fhe_native_mamba3.official_parity",
        "probe_official_mamba_parity",
    ),
    "apply_srht_sketch": ("fhe_native_mamba3.srht_sketch", "apply_srht_sketch"),
    "plan_mamba_checkpoint": (
        "fhe_native_mamba3.mamba_checkpoint",
        "plan_mamba_checkpoint",
    ),
    "run_checkpoint_pre_recurrence_stage_gate": (
        "fhe_native_mamba3.checkpoint_pre_recurrence",
        "run_checkpoint_pre_recurrence_stage_gate",
    ),
    "run_checkpoint_pre_recurrence_chain_gate": (
        "fhe_native_mamba3.checkpoint_pre_recurrence",
        "run_checkpoint_pre_recurrence_chain_gate",
    ),
    "run_checkpoint_pre_recurrence_ciphertexts_with_backend": (
        "fhe_native_mamba3.checkpoint_pre_recurrence",
        "run_checkpoint_pre_recurrence_ciphertexts_with_backend",
    ),
    "run_openfhe_static_recurrence": (
        "fhe_native_mamba3.openfhe_backend",
        "run_openfhe_static_recurrence",
    ),
    "rotate_composite": ("fhe_native_mamba3.composite_rotation", "rotate_composite"),
    "run_checkpoint_recurrence_correctness_gate": (
        "fhe_native_mamba3.checkpoint_correctness",
        "run_checkpoint_recurrence_correctness_gate",
    ),
    "run_checkpoint_encrypted_pre_recurrence_recurrence_gate": (
        "fhe_native_mamba3.checkpoint_correctness",
        "run_checkpoint_encrypted_pre_recurrence_recurrence_gate",
    ),
    "run_checkpoint_encrypted_pre_recurrence_full_layer_gate": (
        "fhe_native_mamba3.checkpoint_correctness",
        "run_checkpoint_encrypted_pre_recurrence_full_layer_gate",
    ),
    "run_checkpoint_encrypted_pre_recurrence_full_layer_ciphertexts_with_backend": (
        "fhe_native_mamba3.checkpoint_correctness",
        "run_checkpoint_encrypted_pre_recurrence_full_layer_ciphertexts_with_backend",
    ),
    "run_checkpoint_encrypted_pre_recurrence_full_layer_chain_gate": (
        "fhe_native_mamba3.checkpoint_correctness",
        "run_checkpoint_encrypted_pre_recurrence_full_layer_chain_gate",
    ),
    "run_checkpoint_encrypted_pre_recurrence_partial_visible_chain_proxy": (
        "fhe_native_mamba3.checkpoint_correctness",
        "run_checkpoint_encrypted_pre_recurrence_partial_visible_chain_proxy",
    ),
    "run_checkpoint_full_layer_ciphertext_gate": (
        "fhe_native_mamba3.checkpoint_correctness",
        "run_checkpoint_full_layer_ciphertext_gate",
    ),
    "run_checkpoint_full_layer_ciphertexts_with_backend": (
        "fhe_native_mamba3.checkpoint_correctness",
        "run_checkpoint_full_layer_ciphertexts_with_backend",
    ),
    "run_checkpoint_full_layer_ciphertext_sweep": (
        "fhe_native_mamba3.checkpoint_full_layer_sweep",
        "run_checkpoint_full_layer_ciphertext_sweep",
    ),
    "run_checkpoint_grouped_encrypted_pre_recurrence_full_layer_gate": (
        "fhe_native_mamba3.checkpoint_correctness",
        "run_checkpoint_grouped_encrypted_pre_recurrence_full_layer_gate",
    ),
    "run_checkpoint_grouped_encrypted_pre_recurrence_full_layer_ciphertexts_with_backend": (
        "fhe_native_mamba3.checkpoint_correctness",
        "run_checkpoint_grouped_encrypted_pre_recurrence_full_layer_ciphertexts_with_backend",
    ),
    "run_checkpoint_grouped_encrypted_pre_recurrence_full_layer_chain_proxy": (
        "fhe_native_mamba3.checkpoint_correctness",
        "run_checkpoint_grouped_encrypted_pre_recurrence_full_layer_chain_proxy",
    ),
    "run_checkpoint_visible_projection_sweep": (
        "fhe_native_mamba3.checkpoint_visible_projection_sweep",
        "run_checkpoint_visible_projection_sweep",
    ),
    "run_backend_srht_smoke": (
        "fhe_native_mamba3.backend_srht",
        "run_backend_srht_smoke",
    ),
    "required_full_layer_visible_rotations": (
        "fhe_native_mamba3.checkpoint_correctness",
        "required_full_layer_visible_rotations",
    ),
    "required_grouped_full_layer_lift_rotations": (
        "fhe_native_mamba3.stage1_grouped_recurrence",
        "required_grouped_full_layer_lift_rotations",
    ),
    "required_backend_srht_rotations": (
        "fhe_native_mamba3.backend_srht",
        "required_backend_srht_rotations",
    ),
    "expand_rank_to_state_bsgs_rotation_steps": (
        "fhe_native_mamba3.checkpoint_correctness",
        "expand_rank_to_state_bsgs_rotation_steps",
    ),
    "expand_state_vector_to_state_bsgs_rotation_steps": (
        "fhe_native_mamba3.checkpoint_correctness",
        "expand_state_vector_to_state_bsgs_rotation_steps",
    ),
    "run_static_mimo_recurrence_ciphertexts_with_backend": (
        "fhe_native_mamba3.openfhe_backend",
        "run_static_mimo_recurrence_ciphertexts_with_backend",
    ),
    "run_static_mimo_recurrence_ciphertext_chain_with_backend": (
        "fhe_native_mamba3.openfhe_backend",
        "run_static_mimo_recurrence_ciphertext_chain_with_backend",
    ),
    "run_stage1_grouped_full_layer_lift_smoke": (
        "fhe_native_mamba3.stage1_grouped_recurrence",
        "run_stage1_grouped_full_layer_lift_smoke",
    ),
    "run_stage1_grouped_static_recurrence_smoke": (
        "fhe_native_mamba3.stage1_grouped_recurrence",
        "run_stage1_grouped_static_recurrence_smoke",
    ),
    "scale_recurrence_state": (
        "fhe_native_mamba3.openfhe_backend",
        "scale_recurrence_state",
    ),
    "scale_recurrence_state_and_output": (
        "fhe_native_mamba3.openfhe_backend",
        "scale_recurrence_state_and_output",
    ),
    "slice_recurrence_problem_by_rank": (
        "fhe_native_mamba3.stage1_grouped_recurrence",
        "slice_recurrence_problem_by_rank",
    ),
    "slot_bsgs_linear_block0": (
        "fhe_native_mamba3.slot_bsgs",
        "slot_bsgs_linear_block0",
    ),
    "slot_bsgs_pre_mask": (
        "fhe_native_mamba3.slot_bsgs",
        "slot_bsgs_pre_mask",
    ),
    "slot_bsgs_rotation_groups": (
        "fhe_native_mamba3.slot_bsgs",
        "slot_bsgs_rotation_groups",
    ),
    "ssd_prefix_scan": ("fhe_native_mamba3.ssd_prefix_scan", "ssd_prefix_scan"),
    "ssd_prefix_scan_prefill": (
        "fhe_native_mamba3.ssd_prefix_scan",
        "ssd_prefix_scan_prefill",
    ),
    "stage1_comparison_markdown": (
        "fhe_native_mamba3.stage1_report",
        "stage1_comparison_markdown",
    ),
    "stage1_checkpoint_cost_markdown": (
        "fhe_native_mamba3.stage1_checkpoint_cost_report",
        "stage1_checkpoint_cost_markdown",
    ),
    "state_axis_rotation_steps": (
        "fhe_native_mamba3.stage1_state_major_layout",
        "state_axis_rotation_steps",
    ),
    "state_major_slot": (
        "fhe_native_mamba3.stage1_state_major_kernel",
        "state_major_slot",
    ),
    "srht_sample_indices": ("fhe_native_mamba3.srht_sketch", "srht_sample_indices"),
    "srht_sampling_mask": ("fhe_native_mamba3.srht_sketch", "srht_sampling_mask"),
    "srht_sketch_matrix": ("fhe_native_mamba3.srht_sketch", "srht_sketch_matrix"),
    "sweep_head_pack_candidates": (
        "fhe_native_mamba3.head_packing",
        "sweep_head_pack_candidates",
    ),
    "run_stage0_mimo": ("fhe_native_mamba3.benchmarks.stage0_mimo", "run_stage0_mimo"),
    "run_stage0_sweep": ("fhe_native_mamba3.benchmarks.stage0_sweep", "run_stage0_sweep"),
    "run_stage1_pack_sweep": (
        "fhe_native_mamba3.stage1_pack_sweep",
        "run_stage1_pack_sweep",
    ),
    "run_stage2_sketch_sweep": (
        "fhe_native_mamba3.stage2_sketch_sweep",
        "run_stage2_sketch_sweep",
    ),
    "run_stage2_sketch_seed_sweep": (
        "fhe_native_mamba3.stage2_sketch_seed_sweep",
        "run_stage2_sketch_seed_sweep",
    ),
    "run_tiny_mimo_block_smoke": (
        "fhe_native_mamba3.stage1_tiny_mimo",
        "run_tiny_mimo_block_smoke",
    ),
    "required_tiny_mimo_block_rotations": (
        "fhe_native_mamba3.stage1_tiny_mimo",
        "required_tiny_mimo_block_rotations",
    ),
    "required_toy_cutmax_rotations": (
        "fhe_native_mamba3.toy_cutmax",
        "required_toy_cutmax_rotations",
    ),
    "run_toy_cutmax_smoke": (
        "fhe_native_mamba3.toy_cutmax",
        "run_toy_cutmax_smoke",
    ),
    "required_state_major_toy_kernel_rotations": (
        "fhe_native_mamba3.stage1_state_major_kernel",
        "required_state_major_toy_kernel_rotations",
    ),
    "run_state_major_toy_kernel": (
        "fhe_native_mamba3.stage1_state_major_kernel",
        "run_state_major_toy_kernel",
    ),
    "run_state_major_full_shape_tracking": (
        "fhe_native_mamba3.stage1_state_major_fullshape",
        "run_state_major_full_shape_tracking",
    ),
    "validate_artifact_file": (
        "fhe_native_mamba3.artifact_validation",
        "validate_artifact_file",
    ),
    "validate_benchmark_artifact": (
        "fhe_native_mamba3.artifact_validation",
        "validate_benchmark_artifact",
    ),
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
    "PRE_RECURRENCE_STAGES",
    "ArtifactValidationIssue",
    "ArtifactValidationResult",
    "BackendAffinePrefixScanResult",
    "BackendCapability",
    "BackendPackedMimoReadoutResult",
    "BackendPrefixScanResult",
    "BackendSegmentedPrefixScanResult",
    "BackendSrhtSmokeResult",
    "BackendStats",
    "BootstrapExecutionBlockCost",
    "BootstrapExecutionPolicy",
    "BootstrapExecutionSchedule",
    "BootstrapExecutionScheduleStep",
    "BootstrapSchedule",
    "BsgsSchedule",
    "CheckpointClientDecodeSmoke",
    "CheckpointEncryptedPreRecurrenceRecurrenceGate",
    "CheckpointFullLayerCiphertextChainGate",
    "CheckpointFullLayerCiphertextGate",
    "CheckpointFullLayerCiphertextTrace",
    "CheckpointFullLayerSweepLayer",
    "CheckpointFullLayerSweepResult",
    "CheckpointInspection",
    "CheckpointPreRecurrenceChainGate",
    "CheckpointPreRecurrenceCiphertextTrace",
    "CheckpointPreRecurrenceGroupedCiphertextTrace",
    "CheckpointPreRecurrenceRankPackTrace",
    "CheckpointPreRecurrenceStageGate",
    "CheckpointRecurrenceCorrectnessGate",
    "CheckpointSketchMatrixResult",
    "CheckpointSketchMatrixRow",
    "CheckpointSourceProfile",
    "CheckpointSourceProfileLayer",
    "CheckpointSourceSketchTrace",
    "CheckpointTensorSpec",
    "CheckpointVisibleProjectionSweepResult",
    "CheckpointVisibleProjectionSweepRow",
    "CiphertextHandoffLayer",
    "CiphertextHandoffResult",
    "CiphertextLayoutContract",
    "CkksConfig",
    "CkksTrace",
    "ClientDecodeReport",
    "ClientDecodeReportRow",
    "CompositeRotationBackend",
    "CompositeRotationEstimate",
    "FHEBackend",
    "FheCostEstimate",
    "FheMamba3Block",
    "FheMamba3Config",
    "FheMamba3ForCausalLM",
    "GreedyBootstrapSchedule",
    "HeadPackCandidate",
    "HeadPackSweep",
    "IntegratedCostEstimate",
    "LazyBootstrapReport",
    "LazyBootstrapScheduleRow",
    "LoRAConfig",
    "LoRALinear",
    "MambaCheckpointAdapterReport",
    "MambaCheckpointPlan",
    "MambaLayerPlan",
    "MambaLayerReferenceResult",
    "MambaSourceDeltaResult",
    "MambaSourceLayerDiagnostics",
    "MambaSourceVisibleHandoffTensors",
    "NumpyTrackingBackend",
    "OfficialMambaParityResult",
    "OpenFheBootstrapConfig",
    "OpenFheBootstrapLatencyConfig",
    "OpenFheCkksBackend",
    "OpenFheRecurrenceCiphertextChainResult",
    "OpenFheRecurrenceCiphertextTrace",
    "OpenFheRecurrenceProblem",
    "OpenFheRecurrenceResult",
    "PackedPrefixScanPlan",
    "PackingPlan",
    "RangeLossConfig",
    "RangeLossResult",
    "RangeLossTerm",
    "RecurrenceTraceProfile",
    "SketchEvidenceReport",
    "SketchEvidenceReportRow",
    "SketchRecurrenceClaim",
    "SlotBsgsSchedule",
    "SrhtButterflyStage",
    "SrhtSketchMetadata",
    "SsdPrefixScanResult",
    "Stage0MimoConfig",
    "Stage0SweepConfig",
    "Stage1CandidatePlan",
    "Stage1CheckpointCostReport",
    "Stage1CheckpointCostRow",
    "Stage1CheckpointGroupedGateInventoryReport",
    "Stage1CheckpointGroupedGateInventoryRow",
    "Stage1ComparisonReport",
    "Stage1ComparisonRow",
    "Stage1CompositeRotationReport",
    "Stage1CompositeRotationRow",
    "Stage1Dependency",
    "Stage1GroupedChainInventoryReport",
    "Stage1GroupedChainInventoryRow",
    "Stage1GroupedFullLayerLiftSmokeResult",
    "Stage1GroupedRecurrenceGroup",
    "Stage1GroupedRecurrenceSmokeResult",
    "Stage1PackSweepResult",
    "Stage1PackSweepRow",
    "Stage1Plan",
    "Stage1ProfileHints",
    "Stage2SketchSeedSample",
    "Stage2SketchSeedSweepResult",
    "Stage2SketchSeedSweepRow",
    "Stage2SketchSweepResult",
    "Stage2SketchSweepRow",
    "StateDictMappingDraft",
    "StateDictMappingReport",
    "StateDictMappingRule",
    "StateMajorFullShapeConfig",
    "StateMajorFullShapeResult",
    "StateMajorLayoutPlan",
    "StateMajorToyKernelResult",
    "StateMajorToyProblem",
    "TinyMimoBlockProblem",
    "TinyMimoBlockSmokeResult",
    "ToyCutMaxSmokeResult",
    "TrackingBackend",
    "WeightBundleManifest",
    "WeightBundleRecurrenceProblem",
    "WeightCalibration",
    "WeightEncodingConfig",
    "__version__",
    "adapt_mamba_state_dict_to_model",
    "apply_handoff_bootstrap_schedule",
    "apply_lora_to_linear_modules",
    "apply_srht_sketch",
    "backend_apply_srht_masked",
    "backend_capability_matrix",
    "backend_hillis_steele_affine_scan",
    "backend_hillis_steele_prefix_products",
    "backend_packed_static_mimo_readout",
    "backend_segmented_hillis_steele_affine_scan",
    "backend_segmented_hillis_steele_prefix_products",
    "build_bootstrap_execution_schedule",
    "build_checkpoint_source_sketch_trace",
    "build_client_decode_report",
    "build_fixed_bsgs_schedule",
    "build_lazy_bootstrap_report",
    "build_mamba_source_recurrence_problem",
    "build_mamba_source_visible_handoff_tensors",
    "build_packed_prefix_scan_plan",
    "build_prefix_scan_metadata",
    "build_recurrence_bootstrap_plan",
    "build_rotation_inventory",
    "build_sketch_evidence_report",
    "build_slot_bsgs_schedule",
    "build_srht_sketch_metadata",
    "build_stage0_status_report",
    "build_stage1_checkpoint_cost_report",
    "build_stage1_checkpoint_grouped_gate_inventory",
    "build_stage1_comparison_report",
    "build_stage1_composite_rotation_report",
    "build_stage1_grouped_chain_inventory",
    "build_stage1_plan",
    "build_state_major_layout_plan",
    "build_tiny_mimo_block_problem",
    "build_weight_bundle_manifest",
    "build_weight_bundle_recurrence_problem",
    "calibrate_weight_tensor",
    "calibrate_weight_values",
    "causal_decay_weights",
    "checkpoint_grouped_gate_rotation_steps",
    "checkpoint_monolithic_gate_rotation_steps",
    "ckks_batch_size_for_slots",
    "ckks_ring_dimension_for_batch_size",
    "classify_sketch_recurrence_claim",
    "client_decode_report_markdown",
    "client_side_argmax",
    "client_side_decode_ciphertext",
    "client_side_decode_scores",
    "compare_mamba_layer_reference",
    "compare_mamba_source_delta",
    "composite_rotation_basis_for_steps",
    "current_git_commit",
    "decoding_policies",
    "decompose_rotation_steps",
    "draft_mapping_rules",
    "encrypted_pre_recurrence_logical_batch_size",
    "estimate_block_cost",
    "estimate_composite_rotation_basis",
    "estimate_cumulative_log_contraction",
    "estimate_high_decay_burst_len",
    "estimate_integrated_cost",
    "estimate_recurrence_depth",
    "estimate_recurrence_stack_latency",
    "evaluate_head_pack_candidate",
    "expand_rank_to_state_bsgs_rotation_steps",
    "expand_state_vector_to_state_bsgs_rotation_steps",
    "extract_stage1_profile_hints",
    "fhe_aware_loss",
    "greedy_bootstrap_schedule",
    "group_checkpoint_pre_recurrence_trace_by_rank",
    "grouped_full_layer_lift_plaintext",
    "inspect_checkpoint",
    "lazy_bootstrap_markdown",
    "load_mapping_rules",
    "load_weight_bundle_model",
    "lora_parameter_count",
    "make_demo_full_layer_lift_inputs",
    "make_demo_problem",
    "make_state_major_toy_problem",
    "map_state_dict",
    "mark_only_lora_trainable",
    "measure_openfhe_bootstrap_latency",
    "normalize_rotation_step",
    "packed_mimo_readout_output_slots",
    "packed_prefix_scan_carry_rotation_steps",
    "packed_prefix_scan_rotation_steps",
    "payload_for_tiny_mimo_block_smoke",
    "plaintext_recurrence_trace",
    "plaintext_static_recurrence",
    "plan_mamba_checkpoint",
    "power_of_two_rotation_basis",
    "prefix_decay_products",
    "probe_official_mamba_parity",
    "profile_checkpoint_source_layers",
    "profile_model_batch",
    "profile_recurrence_traces",
    "range_loss",
    "required_backend_srht_rotations",
    "required_full_layer_visible_rotations",
    "required_grouped_full_layer_lift_rotations",
    "required_state_major_toy_kernel_rotations",
    "required_tiny_mimo_block_rotations",
    "required_toy_cutmax_rotations",
    "resolve_pre_recurrence_shape",
    "resolve_rank_strategy",
    "rotate_composite",
    "run_backend_srht_smoke",
    "run_checkpoint_client_decode_smoke",
    "run_checkpoint_encrypted_pre_recurrence_full_layer_chain_gate",
    "run_checkpoint_encrypted_pre_recurrence_full_layer_ciphertexts_with_backend",
    "run_checkpoint_encrypted_pre_recurrence_full_layer_gate",
    "run_checkpoint_encrypted_pre_recurrence_partial_visible_chain_proxy",
    "run_checkpoint_encrypted_pre_recurrence_recurrence_gate",
    "run_checkpoint_full_layer_ciphertext_gate",
    "run_checkpoint_full_layer_ciphertext_sweep",
    "run_checkpoint_full_layer_ciphertexts_with_backend",
    "run_checkpoint_grouped_encrypted_pre_recurrence_full_layer_chain_proxy",
    "run_checkpoint_grouped_encrypted_pre_recurrence_full_layer_ciphertexts_with_backend",
    "run_checkpoint_grouped_encrypted_pre_recurrence_full_layer_gate",
    "run_checkpoint_pre_recurrence_chain_gate",
    "run_checkpoint_pre_recurrence_ciphertexts_with_backend",
    "run_checkpoint_pre_recurrence_stage_gate",
    "run_checkpoint_recurrence_correctness_gate",
    "run_checkpoint_sketch_matrix",
    "run_checkpoint_visible_projection_sweep",
    "run_openfhe_static_recurrence",
    "run_stage0_mimo",
    "run_stage0_sweep",
    "run_stage1_grouped_full_layer_lift_smoke",
    "run_stage1_grouped_static_recurrence_smoke",
    "run_stage1_pack_sweep",
    "run_stage2_sketch_seed_sweep",
    "run_stage2_sketch_sweep",
    "run_state_major_full_shape_tracking",
    "run_state_major_toy_kernel",
    "run_static_mimo_recurrence_ciphertext_chain_with_backend",
    "run_static_mimo_recurrence_ciphertexts_with_backend",
    "run_tiny_mimo_block_smoke",
    "run_toy_cutmax_smoke",
    "save_mamba_checkpoint_bundle",
    "save_mapping_draft",
    "save_weight_bundle",
    "save_weight_bundle_from_checkpoint",
    "save_weight_bundle_from_mapped_checkpoint",
    "scale_recurrence_state",
    "scale_recurrence_state_and_output",
    "sketch_evidence_report_markdown",
    "slice_recurrence_problem_by_rank",
    "slot_bsgs_linear_block0",
    "slot_bsgs_pre_mask",
    "slot_bsgs_rotation_groups",
    "srht_sample_indices",
    "srht_sampling_mask",
    "srht_sketch_matrix",
    "ssd_prefix_scan",
    "ssd_prefix_scan_prefill",
    "stage1_checkpoint_cost_markdown",
    "stage1_comparison_markdown",
    "state_axis_rotation_steps",
    "state_major_slot",
    "sweep_head_pack_candidates",
    "validate_artifact_file",
    "validate_benchmark_artifact",
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
