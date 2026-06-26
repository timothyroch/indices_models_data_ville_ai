"""
Public schema API for shared temporal memory.

Target path:
    urban_resilience_models/
        v2_hazard_conditioned_functional_ugnn/
            memory/
                schemas/
                    __init__.py

This package exposes the stable Phase 4 contracts for:

- neutral node, feature, source-data, architecture, parameter, and execution
  provenance;
- absolute and relative temporal coordinates;
- metadata-preserving historical inputs ``[N, T, D]``;
- sequence-preserving encoder outputs ``[N, T, H]``;
- hazard-independent temporal pooling ``[N, A, T] -> [N, P]``;
- urban-memory orchestration that always preserves the encoded sequence;
- query-neutral temporal retrieval;
- hazard-query alignment and fused hazard-conditioned memory.

Import-boundary policy
----------------------
The public package re-exports schema contracts only. It does not construct
encoders, pooling modules, cross-attention modules, fusion modules, model
components, registries, or configuration objects.

Dependency order is intentional:

    provenance
        -> temporal coordinates
        -> history inputs
        -> sequence encoding
        -> temporal pooling
        -> urban memory
        -> hazard-queried memory

The hazard-queried schema imports stable hazard result contracts but no
trainable retrieval implementation.

Interpretation
--------------------------
These objects preserve software identities, tensor alignment, explicit
missingness, temporal ordering, model-assigned weights, and execution lineage.
They do not by themselves establish causal attribution, calibrated risk,
mechanistic explanation, or data validity.
"""

from __future__ import annotations


# =============================================================================
# Neutral axes and provenance
# =============================================================================

from .provenance import (
    TEMPORAL_NODE_AXIS_SCHEMA_VERSION,
    TEMPORAL_FEATURE_AXIS_SCHEMA_VERSION,
    MEMORY_SOURCE_PROVENANCE_SCHEMA_VERSION,
    MEMORY_ARCHITECTURE_PROVENANCE_SCHEMA_VERSION,
    MEMORY_PARAMETER_SNAPSHOT_PROVENANCE_SCHEMA_VERSION,
    MEMORY_EXECUTION_LINEAGE_SCHEMA_VERSION,
    MEMORY_COMPUTATION_PROVENANCE_SCHEMA_VERSION,
    MEMORY_PARAMETER_SNAPSHOT_POLICY,
    MEMORY_PROVENANCE_SCIENTIFIC_INTERPRETATION,
    TemporalNodeAxis,
    TemporalFeatureAxis,
    NodeAxisIdentity,
    FeatureAxisIdentity,
    MemorySourceProvenance,
    MemoryArchitectureProvenance,
    MemoryParameterSnapshotProvenance,
    MemoryExecutionLineage,
    MemoryComputationProvenance,
    SourceDataProvenance,
    ArchitectureProvenance,
    ParameterSnapshotProvenance,
    ExecutionLineage,
    ComputationProvenance,
)

# =============================================================================
# Temporal-coordinate contracts
# =============================================================================

from .temporal_coordinates import (
    ABSOLUTE_TEMPORAL_COORDINATES_SCHEMA_VERSION,
    RELATIVE_TEMPORAL_COORDINATES_SCHEMA_VERSION,
    TEMPORAL_COORDINATE_CANONICAL_PADDING_VALUE,
    TEMPORAL_COORDINATE_SCIENTIFIC_INTERPRETATION,
    TemporalCoordinateKind,
    TemporalLayout,
    TemporalChronologicalOrder,
    TemporalPaddingDirection,
    TemporalDuplicatePolicy,
    AbsoluteTemporalReferenceKind,
    RelativeTemporalAnchor,
    CANONICAL_TEMPORAL_COORDINATE_KINDS,
    CANONICAL_TEMPORAL_LAYOUTS,
    CANONICAL_TEMPORAL_CHRONOLOGICAL_ORDERS,
    CANONICAL_TEMPORAL_PADDING_DIRECTIONS,
    CANONICAL_TEMPORAL_DUPLICATE_POLICIES,
    CANONICAL_ABSOLUTE_TEMPORAL_REFERENCE_KINDS,
    CANONICAL_RELATIVE_TEMPORAL_ANCHORS,
    AbsoluteTemporalCoordinates,
    RelativeTemporalCoordinates,
    TemporalCoordinates,
    validate_temporal_coordinates,
    temporal_coordinates_fingerprint,
)

# =============================================================================
# Historical input contract
# =============================================================================

from .history_inputs import (
    HISTORICAL_SEQUENCE_INPUTS_SCHEMA_VERSION,
    HISTORY_VALUE_SEMANTICS,
    TIMESTEP_MASK_SEMANTICS,
    FEATURE_OBSERVED_MASK_SEMANTICS,
    HISTORY_CANONICAL_PADDING_VALUE,
    HISTORY_INPUT_SCIENTIFIC_INTERPRETATION,
    HistoryMissingValuePolicy,
    HistoryZeroLengthPolicy,
    CANONICAL_HISTORY_MISSING_VALUE_POLICIES,
    CANONICAL_HISTORY_ZERO_LENGTH_POLICIES,
    HistoricalSequenceInputs,
    TemporalHistoryInputs,
    HistoryInputs,
)

# =============================================================================
# Shared sequence encoding
# =============================================================================

from .sequence_encoding import (
    TEMPORAL_SEQUENCE_ENCODING_SCHEMA_VERSION,
    TEMPORAL_SEQUENCE_ENCODING_VALUE_SEMANTICS,
    TEMPORAL_SEQUENCE_ENCODING_PADDING_POLICY,
    TEMPORAL_SEQUENCE_ENCODING_SCIENTIFIC_INTERPRETATION,
    TemporalSequenceEncoderKind,
    CANONICAL_TEMPORAL_SEQUENCE_ENCODER_KINDS,
    TemporalSequenceEncoding,
    SequenceEncoding,
    SharedTemporalSequenceEncoding,
)

# =============================================================================
# Hazard-independent temporal pooling
# =============================================================================

from .temporal_pooling import (
    TEMPORAL_POOLING_OUTPUT_SCHEMA_VERSION,
    TEMPORAL_POOLING_WEIGHT_SEMANTICS,
    TEMPORAL_POOLING_PADDING_POLICY,
    TEMPORAL_POOLING_NORMALIZATION_POLICY,
    TEMPORAL_POOLING_SCIENTIFIC_INTERPRETATION,
    TemporalPoolingKind,
    TemporalPoolingHeadReduction,
    TemporalPoolingZeroHistoryPolicy,
    CANONICAL_TEMPORAL_POOLING_KINDS,
    CANONICAL_TEMPORAL_POOLING_HEAD_REDUCTIONS,
    CANONICAL_TEMPORAL_POOLING_ZERO_HISTORY_POLICIES,
    TemporalPoolingOutput,
    TemporalMemoryPoolingOutput,
    PoolingOutput,
)

# =============================================================================
# Urban-memory orchestration
# =============================================================================

from .urban_memory import (
    URBAN_MEMORY_SCHEMA_VERSION,
    URBAN_MEMORY_SEQUENCE_PRESERVATION_POLICY,
    URBAN_MEMORY_POOLING_SCOPE,
    URBAN_MEMORY_SCIENTIFIC_INTERPRETATION,
    UrbanMemoryAssemblyPolicy,
    CANONICAL_URBAN_MEMORY_ASSEMBLY_POLICIES,
    UrbanMemory,
    UrbanTemporalMemory,
    SharedUrbanMemory,
)

# =============================================================================
# Temporal retrieval and hazard-conditioned memory
# =============================================================================

from .hazard_queried_memory import (
    TEMPORAL_QUERY_RETRIEVAL_OUTPUT_SCHEMA_VERSION,
    HAZARD_QUERIED_MEMORY_SCHEMA_VERSION,
    TEMPORAL_QUERY_RETRIEVAL_WEIGHT_SEMANTICS,
    TEMPORAL_QUERY_RETRIEVAL_PADDING_POLICY,
    TEMPORAL_QUERY_RETRIEVAL_NORMALIZATION_POLICY,
    HAZARD_QUERY_ALIGNMENT_POLICY,
    HAZARD_QUERIED_MEMORY_SCIENTIFIC_INTERPRETATION,
    TemporalQueryRetrievalKind,
    TemporalQueryRetrievalHeadReduction,
    TemporalQueryRetrievalZeroHistoryPolicy,
    HazardQueryAlignmentScope,
    HazardMemoryFusionPolicy,
    CANONICAL_TEMPORAL_QUERY_RETRIEVAL_KINDS,
    CANONICAL_TEMPORAL_QUERY_RETRIEVAL_HEAD_REDUCTIONS,
    CANONICAL_TEMPORAL_QUERY_RETRIEVAL_ZERO_HISTORY_POLICIES,
    CANONICAL_HAZARD_QUERY_ALIGNMENT_SCOPES,
    CANONICAL_HAZARD_MEMORY_FUSION_POLICIES,
    TemporalQueryRetrievalOutput,
    TemporalRetrievalOutput,
    QueryRetrievalOutput,
    hazard_query_alignment_fingerprint,
    HazardQueriedMemory,
    HazardConditionedMemory,
)


# =============================================================================
# Public API
# =============================================================================

__all__ = (
    "TEMPORAL_NODE_AXIS_SCHEMA_VERSION",
    "TEMPORAL_FEATURE_AXIS_SCHEMA_VERSION",
    "MEMORY_SOURCE_PROVENANCE_SCHEMA_VERSION",
    "MEMORY_ARCHITECTURE_PROVENANCE_SCHEMA_VERSION",
    "MEMORY_PARAMETER_SNAPSHOT_PROVENANCE_SCHEMA_VERSION",
    "MEMORY_EXECUTION_LINEAGE_SCHEMA_VERSION",
    "MEMORY_COMPUTATION_PROVENANCE_SCHEMA_VERSION",
    "MEMORY_PARAMETER_SNAPSHOT_POLICY",
    "MEMORY_PROVENANCE_SCIENTIFIC_INTERPRETATION",
    "TemporalNodeAxis",
    "TemporalFeatureAxis",
    "NodeAxisIdentity",
    "FeatureAxisIdentity",
    "MemorySourceProvenance",
    "MemoryArchitectureProvenance",
    "MemoryParameterSnapshotProvenance",
    "MemoryExecutionLineage",
    "MemoryComputationProvenance",
    "SourceDataProvenance",
    "ArchitectureProvenance",
    "ParameterSnapshotProvenance",
    "ExecutionLineage",
    "ComputationProvenance",
    "ABSOLUTE_TEMPORAL_COORDINATES_SCHEMA_VERSION",
    "RELATIVE_TEMPORAL_COORDINATES_SCHEMA_VERSION",
    "TEMPORAL_COORDINATE_CANONICAL_PADDING_VALUE",
    "TEMPORAL_COORDINATE_SCIENTIFIC_INTERPRETATION",
    "TemporalCoordinateKind",
    "TemporalLayout",
    "TemporalChronologicalOrder",
    "TemporalPaddingDirection",
    "TemporalDuplicatePolicy",
    "AbsoluteTemporalReferenceKind",
    "RelativeTemporalAnchor",
    "CANONICAL_TEMPORAL_COORDINATE_KINDS",
    "CANONICAL_TEMPORAL_LAYOUTS",
    "CANONICAL_TEMPORAL_CHRONOLOGICAL_ORDERS",
    "CANONICAL_TEMPORAL_PADDING_DIRECTIONS",
    "CANONICAL_TEMPORAL_DUPLICATE_POLICIES",
    "CANONICAL_ABSOLUTE_TEMPORAL_REFERENCE_KINDS",
    "CANONICAL_RELATIVE_TEMPORAL_ANCHORS",
    "AbsoluteTemporalCoordinates",
    "RelativeTemporalCoordinates",
    "TemporalCoordinates",
    "validate_temporal_coordinates",
    "temporal_coordinates_fingerprint",
    "HISTORICAL_SEQUENCE_INPUTS_SCHEMA_VERSION",
    "HISTORY_VALUE_SEMANTICS",
    "TIMESTEP_MASK_SEMANTICS",
    "FEATURE_OBSERVED_MASK_SEMANTICS",
    "HISTORY_CANONICAL_PADDING_VALUE",
    "HISTORY_INPUT_SCIENTIFIC_INTERPRETATION",
    "HistoryMissingValuePolicy",
    "HistoryZeroLengthPolicy",
    "CANONICAL_HISTORY_MISSING_VALUE_POLICIES",
    "CANONICAL_HISTORY_ZERO_LENGTH_POLICIES",
    "HistoricalSequenceInputs",
    "TemporalHistoryInputs",
    "HistoryInputs",
    "TEMPORAL_SEQUENCE_ENCODING_SCHEMA_VERSION",
    "TEMPORAL_SEQUENCE_ENCODING_VALUE_SEMANTICS",
    "TEMPORAL_SEQUENCE_ENCODING_PADDING_POLICY",
    "TEMPORAL_SEQUENCE_ENCODING_SCIENTIFIC_INTERPRETATION",
    "TemporalSequenceEncoderKind",
    "CANONICAL_TEMPORAL_SEQUENCE_ENCODER_KINDS",
    "TemporalSequenceEncoding",
    "SequenceEncoding",
    "SharedTemporalSequenceEncoding",
    "TEMPORAL_POOLING_OUTPUT_SCHEMA_VERSION",
    "TEMPORAL_POOLING_WEIGHT_SEMANTICS",
    "TEMPORAL_POOLING_PADDING_POLICY",
    "TEMPORAL_POOLING_NORMALIZATION_POLICY",
    "TEMPORAL_POOLING_SCIENTIFIC_INTERPRETATION",
    "TemporalPoolingKind",
    "TemporalPoolingHeadReduction",
    "TemporalPoolingZeroHistoryPolicy",
    "CANONICAL_TEMPORAL_POOLING_KINDS",
    "CANONICAL_TEMPORAL_POOLING_HEAD_REDUCTIONS",
    "CANONICAL_TEMPORAL_POOLING_ZERO_HISTORY_POLICIES",
    "TemporalPoolingOutput",
    "TemporalMemoryPoolingOutput",
    "PoolingOutput",
    "URBAN_MEMORY_SCHEMA_VERSION",
    "URBAN_MEMORY_SEQUENCE_PRESERVATION_POLICY",
    "URBAN_MEMORY_POOLING_SCOPE",
    "URBAN_MEMORY_SCIENTIFIC_INTERPRETATION",
    "UrbanMemoryAssemblyPolicy",
    "CANONICAL_URBAN_MEMORY_ASSEMBLY_POLICIES",
    "UrbanMemory",
    "UrbanTemporalMemory",
    "SharedUrbanMemory",
    "TEMPORAL_QUERY_RETRIEVAL_OUTPUT_SCHEMA_VERSION",
    "HAZARD_QUERIED_MEMORY_SCHEMA_VERSION",
    "TEMPORAL_QUERY_RETRIEVAL_WEIGHT_SEMANTICS",
    "TEMPORAL_QUERY_RETRIEVAL_PADDING_POLICY",
    "TEMPORAL_QUERY_RETRIEVAL_NORMALIZATION_POLICY",
    "HAZARD_QUERY_ALIGNMENT_POLICY",
    "HAZARD_QUERIED_MEMORY_SCIENTIFIC_INTERPRETATION",
    "TemporalQueryRetrievalKind",
    "TemporalQueryRetrievalHeadReduction",
    "TemporalQueryRetrievalZeroHistoryPolicy",
    "HazardQueryAlignmentScope",
    "HazardMemoryFusionPolicy",
    "CANONICAL_TEMPORAL_QUERY_RETRIEVAL_KINDS",
    "CANONICAL_TEMPORAL_QUERY_RETRIEVAL_HEAD_REDUCTIONS",
    "CANONICAL_TEMPORAL_QUERY_RETRIEVAL_ZERO_HISTORY_POLICIES",
    "CANONICAL_HAZARD_QUERY_ALIGNMENT_SCOPES",
    "CANONICAL_HAZARD_MEMORY_FUSION_POLICIES",
    "TemporalQueryRetrievalOutput",
    "TemporalRetrievalOutput",
    "QueryRetrievalOutput",
    "hazard_query_alignment_fingerprint",
    "HazardQueriedMemory",
    "HazardConditionedMemory",
)
