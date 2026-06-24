"""
Public API for one hazard-conditioned functional message-passing layer.

Target path:
    urban_resilience_models/
        v2_hazard_conditioned_functional_ugnn/
            functional_message_passing/
                layer/
                    __init__.py

The package exposes the complete bounded V2.0 layer subsystem:

- immutable runtime, trace, residual, normalization, and computation schemas;
- parameter-free dropout and optional additive residual updates;
- optional post-residual feature-wise layer normalization;
- explicit tensor-free descriptive diagnostics;
- complete one-layer orchestration over the existing relation-transform,
  edge-normalization, exact-relation gate, edge-attention, message-builder,
  and target-node aggregation subsystems;
- public-output assembly, lineage validation, builders, and execution helpers.

Numerical stage order
---------------------
The bounded implementation follows:

    relation transform
        -> structural edge normalization
        -> optional relation gate
        -> optional edge attention
        -> edge-message construction
        -> existing target-node aggregation
        -> dropout
        -> optional additive residual
        -> optional post-residual layer normalization
        -> updated node state

Trace policy
------------
``none`` retains no optional internal trace.

``node`` retains aggregation, residual-update, and normalization stages.

``full`` retains the complete edge- and node-level stage chain and permits
construction of the historical public intermediate schema.

Limits
-----------------
Gates, attention values, messages, aggregates, state transitions, and
diagnostic summaries are descriptive model quantities. They do not
automatically establish causal importance, faithful explanation, calibrated
uncertainty, counterfactual effect, or mechanistic identifiability.

Importing this package does not construct model components, mutate registries,
run graph computation, or generate diagnostics.
"""

from __future__ import annotations

# ============================================================================
# Immutable schemas and compatibility exports
# ============================================================================

from .schemas import (
    LAYER_INPUTS_SCHEMA_VERSION,
    LAYER_TRACE_POLICY_SCHEMA_VERSION,
    LAYER_RESIDUAL_UPDATE_SCHEMA_VERSION,
    LAYER_NORMALIZATION_OUTPUT_SCHEMA_VERSION,
    LAYER_INTERMEDIATE_TRACE_SCHEMA_VERSION,
    LAYER_COMPUTATION_OUTPUT_SCHEMA_VERSION,
    LAYER_TRACE_NONE,
    LAYER_TRACE_NODE,
    LAYER_TRACE_FULL,
    CANONICAL_LAYER_TRACE_MODES,
    V2_0_IMPLEMENTED_LAYER_TRACE_MODES,
    LayerTracePolicy,
    LAYER_RESIDUAL_DISABLED,
    LAYER_RESIDUAL_ADDITIVE,
    CANONICAL_LAYER_RESIDUAL_MODES,
    V2_0_IMPLEMENTED_LAYER_RESIDUAL_MODES,
    LAYER_NORMALIZATION_NONE,
    LAYER_NORMALIZATION_LAYER_NORM,
    CANONICAL_LAYER_NORMALIZATION_MODES,
    V2_0_IMPLEMENTED_LAYER_NORMALIZATION_MODES,
    LAYER_NORMALIZATION_PRE_RESIDUAL,
    LAYER_NORMALIZATION_POST_RESIDUAL,
    CANONICAL_LAYER_NORMALIZATION_POSITIONS,
    V2_0_IMPLEMENTED_LAYER_NORMALIZATION_POSITIONS,
    LAYER_UPDATE_BRANCH_FORMULA,
    LAYER_ADDITIVE_RESIDUAL_FORMULA,
    LAYER_DISABLED_RESIDUAL_FORMULA,
    LAYER_POST_NORMALIZATION_FORMULA,
    LAYER_INPUT_LAYOUT,
    LAYER_AGGREGATE_LAYOUT,
    LAYER_OUTPUT_LAYOUT,
    LAYER_SCIENTIFIC_INTERPRETATION,
    FunctionalMessagePassingLayerInputs,
    LayerInputs,
    LayerResidualUpdateOutput,
    ResidualUpdateOutput,
    LayerNormalizationOutput,
    NormalizationOutput,
    FunctionalMessagePassingLayerTrace,
    LayerTrace,
    LayerComputationOutput,
    FunctionalMessagePassingLayerComputation,
    FunctionalMessagePassingLayerStages,
    LayerStages,
    validate_layer_stage_chain,
    build_public_layer_intermediates,
    validate_public_layer_output,
    layer_schema_architecture_dict,
    layer_schema_architecture_fingerprint,
    AggregationOutput,
    FunctionalMessagePassingIntermediates,
    FunctionalMessagePassingLayerOutput,
)

# ============================================================================
# Residual update
# ============================================================================

from .residual_update import (
    LAYER_RESIDUAL_UPDATER_SCHEMA_VERSION,
    LAYER_RESIDUAL_UPDATER_OPERATION,
    LAYER_RESIDUAL_UPDATER_OPERATION_ORDER,
    LAYER_RESIDUAL_UPDATER_PARAMETER_FREE,
    LAYER_RESIDUAL_UPDATER_BUFFER_FREE,
    LAYER_RESIDUAL_UPDATER_PROJECTION_OWNED_HERE,
    LAYER_RESIDUAL_UPDATER_AGGREGATION_OWNED_HERE,
    LAYER_RESIDUAL_UPDATER_NORMALIZATION_OWNED_HERE,
    LAYER_DROPOUT_SEMANTICS,
    LAYER_DISABLED_DROPOUT_IDENTITY_POLICY,
    LAYER_DISABLED_RESIDUAL_IDENTITY_POLICY,
    residual_mode_from_enabled,
    residual_enabled_from_mode,
    layer_residual_updater_architecture_dict,
    layer_residual_updater_architecture_fingerprint,
    apply_layer_update_dropout,
    apply_update_dropout,
    apply_layer_residual,
    apply_residual,
    build_layer_residual_update_output,
    apply_layer_residual_update,
    resolve_layer_residual_update,
    validate_layer_residual_update_output,
    layer_residual_update_diagnostic_summary,
    LayerResidualUpdater,
    ResidualUpdater,
    FunctionalResidualUpdater,
    MessagePassingResidualUpdater,
    build_layer_residual_updater,
    build_residual_updater,
    build_layer_residual_updater_from_flags,
    build_residual_updater_from_flags,
)

# ============================================================================
# Post-residual normalization
# ============================================================================

from .normalization import (
    LAYER_NORMALIZER_SCHEMA_VERSION,
    LAYER_NORMALIZER_OPERATION,
    LAYER_NORMALIZER_OPERATION_ORDER,
    LAYER_NORMALIZER_NORMALIZED_AXIS,
    LAYER_NORMALIZER_STATISTIC_SCOPE,
    LAYER_NORMALIZER_VARIANCE_ESTIMATOR,
    LAYER_NORMALIZER_AGGREGATION_OWNED_HERE,
    LAYER_NORMALIZER_DROPOUT_OWNED_HERE,
    LAYER_NORMALIZER_RESIDUAL_OWNED_HERE,
    LAYER_NORMALIZER_MULTI_LAYER_ITERATION_OWNED_HERE,
    LAYER_DISABLED_NORMALIZATION_IDENTITY_POLICY,
    LAYER_NORMALIZER_DEFAULT_EPSILON,
    LAYER_NORMALIZER_DEFAULT_ELEMENTWISE_AFFINE,
    LAYER_NORMALIZER_DEFAULT_BIAS_ENABLED,
    normalization_mode_from_enabled,
    normalization_enabled_from_mode,
    layer_normalizer_architecture_dict,
    layer_normalizer_architecture_fingerprint,
    apply_layer_normalization,
    apply_normalization,
    build_layer_normalization_output,
    normalize_layer_state,
    resolve_layer_normalization,
    validate_layer_normalization_output,
    layer_normalization_diagnostic_summary,
    LayerNormalizer,
    FunctionalLayerNormalizer,
    MessagePassingLayerNormalizer,
    PostResidualLayerNormalizer,
    build_layer_normalizer,
    build_normalizer,
    build_layer_normalizer_from_flag,
    build_normalizer_from_flag,
)

# ============================================================================
# Explicit descriptive diagnostics
# ============================================================================

from .diagnostics import (
    LAYER_DIAGNOSTICS_SCHEMA_VERSION,
    LAYER_DIAGNOSTICS_INTERPRETATION,
    LAYER_DIAGNOSTICS_OPERATION_ORDER,
    LAYER_DIAGNOSTICS_PARAMETER_FREE,
    LAYER_DIAGNOSTICS_BUFFER_FREE,
    LAYER_DIAGNOSTICS_IMPLICIT_FORWARD_EXECUTION,
    LAYER_DIAGNOSTIC_SECTION_STAGE_SUMMARIES,
    LAYER_DIAGNOSTIC_SECTION_GLOBAL,
    LAYER_DIAGNOSTIC_SECTION_BY_GRAPH,
    LAYER_DIAGNOSTIC_SECTION_TRACE,
    LAYER_DIAGNOSTIC_SECTION_REGULARIZATION,
    LAYER_DIAGNOSTIC_SECTION_LINEAGE,
    LAYER_DIAGNOSTIC_SECTION_ALERTS,
    LAYER_DIAGNOSTIC_REQUIRED_SECTIONS,
    LayerDiagnosticThresholds,
    DEFAULT_LAYER_DIAGNOSTIC_THRESHOLDS,
    scalar_tensor_statistics,
    matrix_statistics,
    state_transition_statistics,
    incoming_edge_count_statistics,
    aggregation_diagnostic_summary,
    graph_batch_diagnostics,
    layer_trace_diagnostic_summary,
    layer_lineage_summary,
    regularization_diagnostic_summary,
    derive_layer_alerts,
    layer_diagnostics_architecture_dict,
    layer_diagnostics_architecture_fingerprint,
    build_layer_diagnostic_report,
    layer_diagnostic_report,
    build_public_layer_diagnostic_report,
    public_layer_diagnostic_report,
    layer_diagnostic_report_fingerprint,
    validate_layer_diagnostic_report,
    LayerDiagnostics,
    FunctionalLayerDiagnostics,
    MessagePassingLayerDiagnostics,
    build_layer_diagnostics,
    build_functional_layer_diagnostics,
    build_message_passing_layer_diagnostics,
)

# ============================================================================
# Complete layer orchestration
# ============================================================================

from .layer import (
    FUNCTIONAL_MESSAGE_PASSING_LAYER_SCHEMA_VERSION,
    FUNCTIONAL_MESSAGE_PASSING_LAYER_OPERATION_ORDER,
    FUNCTIONAL_MESSAGE_PASSING_LAYER_SCIENTIFIC_INTERPRETATION,
    FUNCTIONAL_MESSAGE_PASSING_LAYER_OUTPUT_SCHEMA,
    FUNCTIONAL_MESSAGE_PASSING_LAYER_AGGREGATION_ORCHESTRATED_HERE,
    FUNCTIONAL_MESSAGE_PASSING_LAYER_AGGREGATION_MATH_OWNED_HERE,
    FUNCTIONAL_MESSAGE_PASSING_LAYER_STACKING_OWNED_HERE,
    FUNCTIONAL_MESSAGE_PASSING_LAYER_PREDICTION_OWNED_HERE,
    FUNCTIONAL_MESSAGE_PASSING_LAYER_DISABLED_GATE_REPRESENTATION,
    FUNCTIONAL_MESSAGE_PASSING_LAYER_DISABLED_ATTENTION_REPRESENTATION,
    FUNCTIONAL_MESSAGE_PASSING_LAYER_UNIFORM_ATTENTION_REPRESENTATION,
    FUNCTIONAL_MESSAGE_PASSING_LAYER_TRACE_AFFECTS_NUMERICS,
    FUNCTIONAL_MESSAGE_PASSING_LAYER_DIAGNOSTICS_AFFECT_NUMERICS,
    FunctionalMessagePassingLayerEdgeStages,
    LayerEdgeStages,
    FunctionalMessagePassingLayerNodeStages,
    LayerNodeStages,
    FunctionalMessagePassingLayerRun,
    LayerRun,
    FunctionalMessagePassingLayerRunWithDiagnostics,
    LayerRunWithDiagnostics,
    assemble_functional_message_passing_layer_output,
    validate_functional_message_passing_layer_run,
    FunctionalMessagePassingLayer,
    HazardConditionedFunctionalMessagePassingLayer,
    FunctionalLayer,
    MessagePassingLayer,
    build_functional_message_passing_layer,
    build_layer,
    build_functional_message_passing_layer_from_config,
    build_layer_from_config,
    run_functional_message_passing_layer,
    run_layer,
    run_functional_message_passing_layer_complete,
    run_layer_complete,
)


LAYER_PACKAGE_API_VERSION = "0.1"


__all__ = (
    "LAYER_PACKAGE_API_VERSION",
    "LAYER_INPUTS_SCHEMA_VERSION",
    "LAYER_TRACE_POLICY_SCHEMA_VERSION",
    "LAYER_RESIDUAL_UPDATE_SCHEMA_VERSION",
    "LAYER_NORMALIZATION_OUTPUT_SCHEMA_VERSION",
    "LAYER_INTERMEDIATE_TRACE_SCHEMA_VERSION",
    "LAYER_COMPUTATION_OUTPUT_SCHEMA_VERSION",
    "LAYER_TRACE_NONE",
    "LAYER_TRACE_NODE",
    "LAYER_TRACE_FULL",
    "CANONICAL_LAYER_TRACE_MODES",
    "V2_0_IMPLEMENTED_LAYER_TRACE_MODES",
    "LayerTracePolicy",
    "LAYER_RESIDUAL_DISABLED",
    "LAYER_RESIDUAL_ADDITIVE",
    "CANONICAL_LAYER_RESIDUAL_MODES",
    "V2_0_IMPLEMENTED_LAYER_RESIDUAL_MODES",
    "LAYER_NORMALIZATION_NONE",
    "LAYER_NORMALIZATION_LAYER_NORM",
    "CANONICAL_LAYER_NORMALIZATION_MODES",
    "V2_0_IMPLEMENTED_LAYER_NORMALIZATION_MODES",
    "LAYER_NORMALIZATION_PRE_RESIDUAL",
    "LAYER_NORMALIZATION_POST_RESIDUAL",
    "CANONICAL_LAYER_NORMALIZATION_POSITIONS",
    "V2_0_IMPLEMENTED_LAYER_NORMALIZATION_POSITIONS",
    "LAYER_UPDATE_BRANCH_FORMULA",
    "LAYER_ADDITIVE_RESIDUAL_FORMULA",
    "LAYER_DISABLED_RESIDUAL_FORMULA",
    "LAYER_POST_NORMALIZATION_FORMULA",
    "LAYER_INPUT_LAYOUT",
    "LAYER_AGGREGATE_LAYOUT",
    "LAYER_OUTPUT_LAYOUT",
    "LAYER_SCIENTIFIC_INTERPRETATION",
    "FunctionalMessagePassingLayerInputs",
    "LayerInputs",
    "LayerResidualUpdateOutput",
    "ResidualUpdateOutput",
    "LayerNormalizationOutput",
    "NormalizationOutput",
    "FunctionalMessagePassingLayerTrace",
    "LayerTrace",
    "LayerComputationOutput",
    "FunctionalMessagePassingLayerComputation",
    "FunctionalMessagePassingLayerStages",
    "LayerStages",
    "validate_layer_stage_chain",
    "build_public_layer_intermediates",
    "validate_public_layer_output",
    "layer_schema_architecture_dict",
    "layer_schema_architecture_fingerprint",
    "AggregationOutput",
    "FunctionalMessagePassingIntermediates",
    "FunctionalMessagePassingLayerOutput",
    "LAYER_RESIDUAL_UPDATER_SCHEMA_VERSION",
    "LAYER_RESIDUAL_UPDATER_OPERATION",
    "LAYER_RESIDUAL_UPDATER_OPERATION_ORDER",
    "LAYER_RESIDUAL_UPDATER_PARAMETER_FREE",
    "LAYER_RESIDUAL_UPDATER_BUFFER_FREE",
    "LAYER_RESIDUAL_UPDATER_PROJECTION_OWNED_HERE",
    "LAYER_RESIDUAL_UPDATER_AGGREGATION_OWNED_HERE",
    "LAYER_RESIDUAL_UPDATER_NORMALIZATION_OWNED_HERE",
    "LAYER_DROPOUT_SEMANTICS",
    "LAYER_DISABLED_DROPOUT_IDENTITY_POLICY",
    "LAYER_DISABLED_RESIDUAL_IDENTITY_POLICY",
    "residual_mode_from_enabled",
    "residual_enabled_from_mode",
    "layer_residual_updater_architecture_dict",
    "layer_residual_updater_architecture_fingerprint",
    "apply_layer_update_dropout",
    "apply_update_dropout",
    "apply_layer_residual",
    "apply_residual",
    "build_layer_residual_update_output",
    "apply_layer_residual_update",
    "resolve_layer_residual_update",
    "validate_layer_residual_update_output",
    "layer_residual_update_diagnostic_summary",
    "LayerResidualUpdater",
    "ResidualUpdater",
    "FunctionalResidualUpdater",
    "MessagePassingResidualUpdater",
    "build_layer_residual_updater",
    "build_residual_updater",
    "build_layer_residual_updater_from_flags",
    "build_residual_updater_from_flags",
    "LAYER_NORMALIZER_SCHEMA_VERSION",
    "LAYER_NORMALIZER_OPERATION",
    "LAYER_NORMALIZER_OPERATION_ORDER",
    "LAYER_NORMALIZER_NORMALIZED_AXIS",
    "LAYER_NORMALIZER_STATISTIC_SCOPE",
    "LAYER_NORMALIZER_VARIANCE_ESTIMATOR",
    "LAYER_NORMALIZER_AGGREGATION_OWNED_HERE",
    "LAYER_NORMALIZER_DROPOUT_OWNED_HERE",
    "LAYER_NORMALIZER_RESIDUAL_OWNED_HERE",
    "LAYER_NORMALIZER_MULTI_LAYER_ITERATION_OWNED_HERE",
    "LAYER_DISABLED_NORMALIZATION_IDENTITY_POLICY",
    "LAYER_NORMALIZER_DEFAULT_EPSILON",
    "LAYER_NORMALIZER_DEFAULT_ELEMENTWISE_AFFINE",
    "LAYER_NORMALIZER_DEFAULT_BIAS_ENABLED",
    "normalization_mode_from_enabled",
    "normalization_enabled_from_mode",
    "layer_normalizer_architecture_dict",
    "layer_normalizer_architecture_fingerprint",
    "apply_layer_normalization",
    "apply_normalization",
    "build_layer_normalization_output",
    "normalize_layer_state",
    "resolve_layer_normalization",
    "validate_layer_normalization_output",
    "layer_normalization_diagnostic_summary",
    "LayerNormalizer",
    "FunctionalLayerNormalizer",
    "MessagePassingLayerNormalizer",
    "PostResidualLayerNormalizer",
    "build_layer_normalizer",
    "build_normalizer",
    "build_layer_normalizer_from_flag",
    "build_normalizer_from_flag",
    "LAYER_DIAGNOSTICS_SCHEMA_VERSION",
    "LAYER_DIAGNOSTICS_INTERPRETATION",
    "LAYER_DIAGNOSTICS_OPERATION_ORDER",
    "LAYER_DIAGNOSTICS_PARAMETER_FREE",
    "LAYER_DIAGNOSTICS_BUFFER_FREE",
    "LAYER_DIAGNOSTICS_IMPLICIT_FORWARD_EXECUTION",
    "LAYER_DIAGNOSTIC_SECTION_STAGE_SUMMARIES",
    "LAYER_DIAGNOSTIC_SECTION_GLOBAL",
    "LAYER_DIAGNOSTIC_SECTION_BY_GRAPH",
    "LAYER_DIAGNOSTIC_SECTION_TRACE",
    "LAYER_DIAGNOSTIC_SECTION_REGULARIZATION",
    "LAYER_DIAGNOSTIC_SECTION_LINEAGE",
    "LAYER_DIAGNOSTIC_SECTION_ALERTS",
    "LAYER_DIAGNOSTIC_REQUIRED_SECTIONS",
    "LayerDiagnosticThresholds",
    "DEFAULT_LAYER_DIAGNOSTIC_THRESHOLDS",
    "scalar_tensor_statistics",
    "matrix_statistics",
    "state_transition_statistics",
    "incoming_edge_count_statistics",
    "aggregation_diagnostic_summary",
    "graph_batch_diagnostics",
    "layer_trace_diagnostic_summary",
    "layer_lineage_summary",
    "regularization_diagnostic_summary",
    "derive_layer_alerts",
    "layer_diagnostics_architecture_dict",
    "layer_diagnostics_architecture_fingerprint",
    "build_layer_diagnostic_report",
    "layer_diagnostic_report",
    "build_public_layer_diagnostic_report",
    "public_layer_diagnostic_report",
    "layer_diagnostic_report_fingerprint",
    "validate_layer_diagnostic_report",
    "LayerDiagnostics",
    "FunctionalLayerDiagnostics",
    "MessagePassingLayerDiagnostics",
    "build_layer_diagnostics",
    "build_functional_layer_diagnostics",
    "build_message_passing_layer_diagnostics",
    "FUNCTIONAL_MESSAGE_PASSING_LAYER_SCHEMA_VERSION",
    "FUNCTIONAL_MESSAGE_PASSING_LAYER_OPERATION_ORDER",
    "FUNCTIONAL_MESSAGE_PASSING_LAYER_SCIENTIFIC_INTERPRETATION",
    "FUNCTIONAL_MESSAGE_PASSING_LAYER_OUTPUT_SCHEMA",
    "FUNCTIONAL_MESSAGE_PASSING_LAYER_AGGREGATION_ORCHESTRATED_HERE",
    "FUNCTIONAL_MESSAGE_PASSING_LAYER_AGGREGATION_MATH_OWNED_HERE",
    "FUNCTIONAL_MESSAGE_PASSING_LAYER_STACKING_OWNED_HERE",
    "FUNCTIONAL_MESSAGE_PASSING_LAYER_PREDICTION_OWNED_HERE",
    "FUNCTIONAL_MESSAGE_PASSING_LAYER_DISABLED_GATE_REPRESENTATION",
    "FUNCTIONAL_MESSAGE_PASSING_LAYER_DISABLED_ATTENTION_REPRESENTATION",
    "FUNCTIONAL_MESSAGE_PASSING_LAYER_UNIFORM_ATTENTION_REPRESENTATION",
    "FUNCTIONAL_MESSAGE_PASSING_LAYER_TRACE_AFFECTS_NUMERICS",
    "FUNCTIONAL_MESSAGE_PASSING_LAYER_DIAGNOSTICS_AFFECT_NUMERICS",
    "FunctionalMessagePassingLayerEdgeStages",
    "LayerEdgeStages",
    "FunctionalMessagePassingLayerNodeStages",
    "LayerNodeStages",
    "FunctionalMessagePassingLayerRun",
    "LayerRun",
    "FunctionalMessagePassingLayerRunWithDiagnostics",
    "LayerRunWithDiagnostics",
    "assemble_functional_message_passing_layer_output",
    "validate_functional_message_passing_layer_run",
    "FunctionalMessagePassingLayer",
    "HazardConditionedFunctionalMessagePassingLayer",
    "FunctionalLayer",
    "MessagePassingLayer",
    "build_functional_message_passing_layer",
    "build_layer",
    "build_functional_message_passing_layer_from_config",
    "build_layer_from_config",
    "run_functional_message_passing_layer",
    "run_layer",
    "run_functional_message_passing_layer_complete",
    "run_layer_complete",
)
