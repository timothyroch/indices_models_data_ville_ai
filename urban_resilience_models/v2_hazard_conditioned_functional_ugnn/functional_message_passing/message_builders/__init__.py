"""
Public API for functional edge-message construction.

Target path:
    urban_resilience_models/
        v2_hazard_conditioned_functional_ugnn/
            functional_message_passing/
                message_builders/
                    __init__.py

This package exposes the complete bounded message-builder subsystem:

    relation-transformed source state
        + structural edge normalization
        + optional exact-relation gate
        + optional edge attention
        + explicit semantic-edge policy
            ↓
    resolved scalar edge coefficients
            ↓
    edge-aligned message composition
            ↓
    public EdgeMessageOutput

Frozen equation
---------------
For every stored directed edge ``e``:

    m_e
        = u_e
        * n_e
        * g_e
        * alpha_e
        * w_e

where:

``u_e``
    Exact relation-transformed source state.

``n_e``
    Structural edge-normalization coefficient.

``g_e``
    Exact relation-gate coefficient, or one when disabled.

``alpha_e``
    Reduced edge-attention coefficient, or one when disabled.

``w_e``
    Explicitly consumed semantic edge weight, or one.

The scalar product is broadcast only across the hidden-feature axis.

Package guarantees
------------------
- exact source-object lineage across every stage;
- exact tensor identity for enabled upstream factors;
- exact multiplicative identity for disabled mechanisms;
- zero-copy consumption of edge-aligned transformed source states;
- explicit semantic-edge policy;
- parameter-free and buffer-free execution;
- empty-edge, dtype, device, and autograd preservation;
- optional tensor-free descriptive diagnostics;
- no target-node aggregation;
- no residual update, dropout, or layer normalization;
- no causal or explanation-faithfulness claims.

Importing this package constructs no modules, registers no global state, and
mutates no registries.
"""

from __future__ import annotations


# =============================================================================
# Package identity
# =============================================================================


MESSAGE_BUILDERS_PACKAGE_API_VERSION = "0.1"


# =============================================================================
# Immutable schemas and vocabulary
# =============================================================================


from .schemas import (
    CANONICAL_MESSAGE_SEMANTIC_EDGE_POLICIES,
    IMPLEMENTED_MESSAGE_SEMANTIC_EDGE_POLICIES,
    MESSAGE_COMBINED_COEFFICIENT_FORMULA,
    MESSAGE_COMPOSITION_FORMULA,
    MESSAGE_COMPOSITION_OUTPUT_SCHEMA_VERSION,
    MESSAGE_DISABLED_FACTOR_POLICY,
    MESSAGE_FACTOR_EDGE_ATTENTION,
    MESSAGE_FACTOR_ORDER,
    MESSAGE_FACTOR_RELATION_GATE,
    MESSAGE_FACTOR_SEMANTIC_EDGE_WEIGHT,
    MESSAGE_FACTOR_STRUCTURAL_NORMALIZATION,
    MESSAGE_SEMANTIC_EDGE_POLICY_IGNORE,
    MESSAGE_SEMANTIC_EDGE_POLICY_USE_SOURCE_GRAPH,
    MESSAGE_TRANSFORM_INPUT_LAYOUT,
    RESOLVED_MESSAGE_COEFFICIENTS_SCHEMA_VERSION,
    EdgeMessageCompositionOutput,
    EdgeMessageOutput,
    MessageBuilderStages,
    MessageCoefficients,
    MessageCompositionOutput,
    ResolvedMessageCoefficients,
    validate_message_builder_stage_chain,
    validate_public_edge_message_output,
)


# =============================================================================
# Zero-copy relation-state boundary
# =============================================================================


from .relation_state_gather import (
    RELATION_STATE_GATHER_INDEXING_OWNED_HERE,
    RELATION_STATE_GATHER_INPUT_LAYOUT,
    RELATION_STATE_GATHER_OPERATION,
    RELATION_STATE_GATHER_OPERATION_ORDER,
    RELATION_STATE_GATHER_OUTPUT_LAYOUT,
    RELATION_STATE_GATHER_OWNER,
    RELATION_STATE_GATHER_PARAMETER_FREE,
    RELATION_STATE_GATHER_SCHEMA_VERSION,
    RELATION_STATE_GATHER_ZERO_COPY_REQUIRED,
    EdgeAlignedRelationStateResolver,
    RelationStateGather,
    RelationStateResolver,
    assert_zero_copy_relation_state,
    build_relation_state_gather,
    build_relation_state_resolver,
    gather_relation_state,
    relation_state_gather_architecture_dict,
    relation_state_gather_architecture_fingerprint,
    relation_state_gather_diagnostic_summary,
    resolve_edge_aligned_relation_state,
    resolve_relation_state,
    validate_edge_aligned_relation_state,
    validate_relation_state_gather_contract,
)


# =============================================================================
# Scalar coefficient resolution
# =============================================================================


from .coefficient_resolution import (
    MESSAGE_COEFFICIENT_RESOLUTION_BUFFER_FREE,
    MESSAGE_COEFFICIENT_RESOLUTION_INTERPRETATION,
    MESSAGE_COEFFICIENT_RESOLUTION_OPERATION_ORDER,
    MESSAGE_COEFFICIENT_RESOLUTION_PARAMETER_FREE,
    MESSAGE_COEFFICIENT_RESOLUTION_SCHEMA_VERSION,
    CoefficientResolver,
    EdgeMessageCoefficientResolver,
    MessageCoefficientResolver,
    OptionalFactorResolution,
    SemanticFactorResolution,
    build_coefficient_resolver,
    build_message_coefficient_resolver,
    coefficient_resolution_architecture_dict,
    coefficient_resolution_architecture_fingerprint,
    combine_message_coefficients,
    message_coefficient_diagnostic_summary,
    resolve_edge_attention_factor,
    resolve_edge_message_coefficients,
    resolve_message_coefficients,
    resolve_relation_gate_factor,
    resolve_semantic_edge_factor,
    resolve_structural_normalization_factor,
    validate_resolved_message_coefficients,
)


# =============================================================================
# Edge-message vector composition
# =============================================================================


from .message_composition import (
    MESSAGE_COMPOSER_AGGREGATION_OWNED_HERE,
    MESSAGE_COMPOSER_BROADCAST_AXIS,
    MESSAGE_COMPOSER_BUFFER_FREE,
    MESSAGE_COMPOSER_INPUT_COEFFICIENT_LAYOUT,
    MESSAGE_COMPOSER_INPUT_STATE_LAYOUT,
    MESSAGE_COMPOSER_OPERATION,
    MESSAGE_COMPOSER_OPERATION_ORDER,
    MESSAGE_COMPOSER_OUTPUT_LAYOUT,
    MESSAGE_COMPOSER_PARAMETER_FREE,
    MESSAGE_COMPOSER_SCHEMA_VERSION,
    EdgeMessageComposer,
    FunctionalMessageComposer,
    MessageComposer,
    build_edge_message_composer,
    build_message_composer,
    compose_edge_message_tensor,
    compose_edge_messages,
    compose_message_output,
    compose_message_vectors,
    message_composer_architecture_dict,
    message_composer_architecture_fingerprint,
    message_composition_diagnostic_summary,
    validate_message_composition_output,
)


# =============================================================================
# Tensor-free descriptive diagnostics
# =============================================================================


from .diagnostics import (
    DEFAULT_MESSAGE_BUILDER_DIAGNOSTIC_THRESHOLDS,
    MESSAGE_BUILDER_DIAGNOSTICS_BUFFER_FREE,
    MESSAGE_BUILDER_DIAGNOSTICS_INTERPRETATION,
    MESSAGE_BUILDER_DIAGNOSTICS_OPERATION_ORDER,
    MESSAGE_BUILDER_DIAGNOSTICS_PARAMETER_FREE,
    MESSAGE_BUILDER_DIAGNOSTICS_SCHEMA_VERSION,
    MESSAGE_BUILDER_DIAGNOSTIC_REQUIRED_SECTIONS,
    MESSAGE_BUILDER_DIAGNOSTIC_SECTION_ALERTS,
    MESSAGE_BUILDER_DIAGNOSTIC_SECTION_BY_GRAPH,
    MESSAGE_BUILDER_DIAGNOSTIC_SECTION_BY_RELATION,
    MESSAGE_BUILDER_DIAGNOSTIC_SECTION_GLOBAL,
    MESSAGE_BUILDER_DIAGNOSTIC_SECTION_LINEAGE,
    MESSAGE_BUILDER_DIAGNOSTIC_SECTION_STAGE_SUMMARIES,
    EdgeMessageDiagnostics,
    MessageBuilderDiagnostics,
    MessageBuilderDiagnosticThresholds,
    MessageDiagnostics,
    build_edge_message_diagnostics,
    build_message_builder_diagnostic_report,
    build_message_builder_diagnostics,
    build_public_edge_message_diagnostic_report,
    derive_message_builder_alerts,
    diagnostic_report_fingerprint,
    edge_vector_norm_statistics,
    exact_relation_diagnostics,
    factor_statistics,
    graph_batch_diagnostics,
    message_builder_diagnostic_report,
    message_builder_diagnostics_architecture_dict,
    message_builder_diagnostics_architecture_fingerprint,
    message_builder_lineage_summary,
    public_edge_message_diagnostic_report,
    scalar_tensor_statistics,
    validate_message_builder_diagnostic_report,
)


# =============================================================================
# Complete orchestration
# =============================================================================


from .message_builders import (
    MESSAGE_BUILDERS_AGGREGATION_OWNED_HERE,
    MESSAGE_BUILDERS_BUFFER_FREE,
    MESSAGE_BUILDERS_DISABLED_ATTENTION_REPRESENTATION,
    MESSAGE_BUILDERS_DISABLED_GATE_REPRESENTATION,
    MESSAGE_BUILDERS_OPERATION_ORDER,
    MESSAGE_BUILDERS_ORCHESTRATOR_SCHEMA_VERSION,
    MESSAGE_BUILDERS_OUTPUT_SCHEMA,
    MESSAGE_BUILDERS_PARAMETER_FREE,
    MESSAGE_BUILDERS_SCIENTIFIC_INTERPRETATION,
    EdgeMessageBuilder,
    FunctionalEdgeMessageBuilder,
    FunctionalMessageBuilder,
    MessageBuilder,
    MessageBuilderRun,
    MessageBuilderRunWithDiagnostics,
    assemble_edge_message_output,
    build_edge_message_builder,
    build_message_builder,
    build_message_builders,
    run_edge_message_builder,
    run_edge_message_builder_stages,
    run_message_builder,
    run_message_builder_stages,
    validate_complete_message_builder_run,
)


# =============================================================================
# Public API
# =============================================================================


__all__ = (
    # Package identity.
    "MESSAGE_BUILDERS_PACKAGE_API_VERSION",

    # Schema versions.
    "RESOLVED_MESSAGE_COEFFICIENTS_SCHEMA_VERSION",
    "MESSAGE_COMPOSITION_OUTPUT_SCHEMA_VERSION",

    # Semantic-edge policies.
    "MESSAGE_SEMANTIC_EDGE_POLICY_IGNORE",
    "MESSAGE_SEMANTIC_EDGE_POLICY_USE_SOURCE_GRAPH",
    "CANONICAL_MESSAGE_SEMANTIC_EDGE_POLICIES",
    "IMPLEMENTED_MESSAGE_SEMANTIC_EDGE_POLICIES",

    # Factor identity and equations.
    "MESSAGE_FACTOR_STRUCTURAL_NORMALIZATION",
    "MESSAGE_FACTOR_RELATION_GATE",
    "MESSAGE_FACTOR_EDGE_ATTENTION",
    "MESSAGE_FACTOR_SEMANTIC_EDGE_WEIGHT",
    "MESSAGE_FACTOR_ORDER",
    "MESSAGE_DISABLED_FACTOR_POLICY",
    "MESSAGE_TRANSFORM_INPUT_LAYOUT",
    "MESSAGE_COMBINED_COEFFICIENT_FORMULA",
    "MESSAGE_COMPOSITION_FORMULA",

    # Immutable schemas.
    "ResolvedMessageCoefficients",
    "MessageCoefficients",
    "MessageCompositionOutput",
    "EdgeMessageCompositionOutput",
    "MessageBuilderStages",
    "EdgeMessageOutput",

    # Schema validators.
    "validate_message_builder_stage_chain",
    "validate_public_edge_message_output",

    # Relation-state identity.
    "RELATION_STATE_GATHER_SCHEMA_VERSION",
    "RELATION_STATE_GATHER_OPERATION",
    "RELATION_STATE_GATHER_INPUT_LAYOUT",
    "RELATION_STATE_GATHER_OUTPUT_LAYOUT",
    "RELATION_STATE_GATHER_OWNER",
    "RELATION_STATE_GATHER_INDEXING_OWNED_HERE",
    "RELATION_STATE_GATHER_ZERO_COPY_REQUIRED",
    "RELATION_STATE_GATHER_PARAMETER_FREE",
    "RELATION_STATE_GATHER_OPERATION_ORDER",

    # Relation-state functional API.
    "relation_state_gather_architecture_dict",
    "relation_state_gather_architecture_fingerprint",
    "validate_edge_aligned_relation_state",
    "assert_zero_copy_relation_state",
    "resolve_edge_aligned_relation_state",
    "resolve_relation_state",
    "gather_relation_state",
    "relation_state_gather_diagnostic_summary",
    "validate_relation_state_gather_contract",

    # Relation-state module API.
    "RelationStateGather",
    "RelationStateResolver",
    "EdgeAlignedRelationStateResolver",
    "build_relation_state_gather",
    "build_relation_state_resolver",

    # Coefficient-resolution identity.
    "MESSAGE_COEFFICIENT_RESOLUTION_SCHEMA_VERSION",
    "MESSAGE_COEFFICIENT_RESOLUTION_OPERATION_ORDER",
    "MESSAGE_COEFFICIENT_RESOLUTION_INTERPRETATION",
    "MESSAGE_COEFFICIENT_RESOLUTION_PARAMETER_FREE",
    "MESSAGE_COEFFICIENT_RESOLUTION_BUFFER_FREE",

    # Coefficient-resolution contracts.
    "OptionalFactorResolution",
    "SemanticFactorResolution",

    # Individual coefficient factors.
    "resolve_structural_normalization_factor",
    "resolve_relation_gate_factor",
    "resolve_edge_attention_factor",
    "resolve_semantic_edge_factor",
    "combine_message_coefficients",

    # Complete coefficient resolution.
    "coefficient_resolution_architecture_dict",
    "coefficient_resolution_architecture_fingerprint",
    "resolve_message_coefficients",
    "resolve_edge_message_coefficients",
    "validate_resolved_message_coefficients",
    "message_coefficient_diagnostic_summary",

    # Coefficient-resolution module API.
    "MessageCoefficientResolver",
    "CoefficientResolver",
    "EdgeMessageCoefficientResolver",
    "build_message_coefficient_resolver",
    "build_coefficient_resolver",

    # Message-composer identity.
    "MESSAGE_COMPOSER_SCHEMA_VERSION",
    "MESSAGE_COMPOSER_OPERATION",
    "MESSAGE_COMPOSER_OPERATION_ORDER",
    "MESSAGE_COMPOSER_INPUT_STATE_LAYOUT",
    "MESSAGE_COMPOSER_INPUT_COEFFICIENT_LAYOUT",
    "MESSAGE_COMPOSER_OUTPUT_LAYOUT",
    "MESSAGE_COMPOSER_BROADCAST_AXIS",
    "MESSAGE_COMPOSER_PARAMETER_FREE",
    "MESSAGE_COMPOSER_BUFFER_FREE",
    "MESSAGE_COMPOSER_AGGREGATION_OWNED_HERE",

    # Message-composer architecture.
    "message_composer_architecture_dict",
    "message_composer_architecture_fingerprint",

    # Message-composer functional API.
    "compose_edge_message_tensor",
    "compose_message_vectors",
    "compose_message_output",
    "compose_edge_messages",
    "validate_message_composition_output",
    "message_composition_diagnostic_summary",

    # Message-composer module API.
    "EdgeMessageComposer",
    "MessageComposer",
    "FunctionalMessageComposer",
    "build_edge_message_composer",
    "build_message_composer",

    # Diagnostic identity.
    "MESSAGE_BUILDER_DIAGNOSTICS_SCHEMA_VERSION",
    "MESSAGE_BUILDER_DIAGNOSTICS_INTERPRETATION",
    "MESSAGE_BUILDER_DIAGNOSTICS_OPERATION_ORDER",
    "MESSAGE_BUILDER_DIAGNOSTICS_PARAMETER_FREE",
    "MESSAGE_BUILDER_DIAGNOSTICS_BUFFER_FREE",
    "MESSAGE_BUILDER_DIAGNOSTIC_SECTION_STAGE_SUMMARIES",
    "MESSAGE_BUILDER_DIAGNOSTIC_SECTION_GLOBAL",
    "MESSAGE_BUILDER_DIAGNOSTIC_SECTION_BY_RELATION",
    "MESSAGE_BUILDER_DIAGNOSTIC_SECTION_BY_GRAPH",
    "MESSAGE_BUILDER_DIAGNOSTIC_SECTION_LINEAGE",
    "MESSAGE_BUILDER_DIAGNOSTIC_SECTION_ALERTS",
    "MESSAGE_BUILDER_DIAGNOSTIC_REQUIRED_SECTIONS",

    # Diagnostic configuration.
    "MessageBuilderDiagnosticThresholds",
    "DEFAULT_MESSAGE_BUILDER_DIAGNOSTIC_THRESHOLDS",

    # Diagnostic statistics.
    "scalar_tensor_statistics",
    "edge_vector_norm_statistics",
    "factor_statistics",
    "exact_relation_diagnostics",
    "graph_batch_diagnostics",

    # Diagnostic lineage and alerts.
    "message_builder_lineage_summary",
    "derive_message_builder_alerts",

    # Diagnostic reports.
    "message_builder_diagnostics_architecture_dict",
    "message_builder_diagnostics_architecture_fingerprint",
    "build_message_builder_diagnostic_report",
    "message_builder_diagnostic_report",
    "build_public_edge_message_diagnostic_report",
    "public_edge_message_diagnostic_report",
    "diagnostic_report_fingerprint",
    "validate_message_builder_diagnostic_report",

    # Diagnostic module API.
    "MessageBuilderDiagnostics",
    "MessageDiagnostics",
    "EdgeMessageDiagnostics",
    "build_message_builder_diagnostics",
    "build_edge_message_diagnostics",

    # Complete orchestrator identity.
    "MESSAGE_BUILDERS_ORCHESTRATOR_SCHEMA_VERSION",
    "MESSAGE_BUILDERS_OPERATION_ORDER",
    "MESSAGE_BUILDERS_SCIENTIFIC_INTERPRETATION",
    "MESSAGE_BUILDERS_DISABLED_GATE_REPRESENTATION",
    "MESSAGE_BUILDERS_DISABLED_ATTENTION_REPRESENTATION",
    "MESSAGE_BUILDERS_PARAMETER_FREE",
    "MESSAGE_BUILDERS_BUFFER_FREE",
    "MESSAGE_BUILDERS_AGGREGATION_OWNED_HERE",
    "MESSAGE_BUILDERS_OUTPUT_SCHEMA",

    # Complete run contracts.
    "MessageBuilderRun",
    "MessageBuilderRunWithDiagnostics",

    # Public assembly and validation.
    "assemble_edge_message_output",
    "validate_complete_message_builder_run",

    # Complete orchestrator module API.
    "EdgeMessageBuilder",
    "MessageBuilder",
    "FunctionalMessageBuilder",
    "FunctionalEdgeMessageBuilder",

    # Complete orchestrator builders.
    "build_edge_message_builder",
    "build_message_builder",
    "build_message_builders",

    # Complete orchestrator execution.
    "run_edge_message_builder",
    "run_message_builder",
    "run_edge_message_builder_stages",
    "run_message_builder_stages",
)
