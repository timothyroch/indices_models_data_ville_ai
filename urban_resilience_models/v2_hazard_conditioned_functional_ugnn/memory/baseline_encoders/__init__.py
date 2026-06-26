"""
Public API for Phase 5 temporal baseline encoders and poolers.

Target path:
    urban_resilience_models/
        v2_hazard_conditioned_functional_ugnn/
            memory/
                baseline_encoders/
                    __init__.py

The package exposes three sequence-preserving baseline encoders:

- ``IdentitySequenceEncoder``;
- ``LinearProjectionSequenceEncoder``;
- ``PointwiseMLPSequenceEncoder``.

It also exposes two deterministic temporal poolers:

- ``MaskedMeanTemporalPooler``;
- ``LastValidTemporalPooler``.

Sequence encoding and temporal pooling remain distinct stages:

    HistoricalSequenceInputs
        -> TemporalSequenceEncoding
        -> TemporalPoolingOutput

The public API additionally includes:

- explicit builders and execution helpers;
- implemented/deferred capability vocabularies;
- type predicates and supported-module aliases;
- detached immutable diagnostics;
- descriptive component constants used in experiment provenance.

Private provenance mechanics from ``_provenance.py`` are intentionally not
re-exported. Importing this package constructs no modules, hashes no parameter
values, mutates no registries, and changes no global PyTorch state.
"""

from __future__ import annotations

# =============================================================================
# Identity sequence baseline
# =============================================================================

from .identity_sequence_encoder import (
    IDENTITY_SEQUENCE_ENCODER_COMPONENT_KIND,
    IDENTITY_SEQUENCE_ENCODER_COMPONENT_NAME,
    IDENTITY_SEQUENCE_ENCODER_FEATURE_OBSERVATION_POLICY,
    IDENTITY_SEQUENCE_ENCODER_IMPLEMENTATION_VERSION,
    IDENTITY_SEQUENCE_ENCODER_OPERATION_NAME,
    IDENTITY_SEQUENCE_ENCODER_PADDING_POLICY,
    IDENTITY_SEQUENCE_ENCODER_TEMPORAL_COORDINATE_POLICY,
    IDENTITY_SEQUENCE_ENCODER_TEMPORAL_INTERACTION,
    IdentitySequenceEncoder,
    IdentityTemporalSequenceEncoder,
)

# =============================================================================
# Linear-projection sequence baseline
# =============================================================================

from .linear_projection_sequence_encoder import (
    LINEAR_PROJECTION_SEQUENCE_ENCODER_ACTIVATION_POLICY,
    LINEAR_PROJECTION_SEQUENCE_ENCODER_COMPONENT_KIND,
    LINEAR_PROJECTION_SEQUENCE_ENCODER_COMPONENT_NAME,
    LINEAR_PROJECTION_SEQUENCE_ENCODER_FEATURE_OBSERVATION_POLICY,
    LINEAR_PROJECTION_SEQUENCE_ENCODER_IMPLEMENTATION_VERSION,
    LINEAR_PROJECTION_SEQUENCE_ENCODER_OPERATION_NAME,
    LINEAR_PROJECTION_SEQUENCE_ENCODER_PADDING_POLICY,
    LINEAR_PROJECTION_SEQUENCE_ENCODER_TEMPORAL_COORDINATE_POLICY,
    LINEAR_PROJECTION_SEQUENCE_ENCODER_TEMPORAL_INTERACTION,
    LinearProjectionSequenceEncoder,
    PerTimestepLinearSequenceEncoder,
)

# =============================================================================
# Pointwise-MLP sequence baseline
# =============================================================================

from .pointwise_mlp_sequence_encoder import (
    POINTWISE_MLP_SEQUENCE_ENCODER_COMPONENT_KIND,
    POINTWISE_MLP_SEQUENCE_ENCODER_COMPONENT_NAME,
    POINTWISE_MLP_SEQUENCE_ENCODER_FEATURE_OBSERVATION_POLICY,
    POINTWISE_MLP_SEQUENCE_ENCODER_IMPLEMENTATION_VERSION,
    POINTWISE_MLP_SEQUENCE_ENCODER_LAYER_POLICY,
    POINTWISE_MLP_SEQUENCE_ENCODER_OPERATION_NAME,
    POINTWISE_MLP_SEQUENCE_ENCODER_PADDING_POLICY,
    POINTWISE_MLP_SEQUENCE_ENCODER_TEMPORAL_COORDINATE_POLICY,
    POINTWISE_MLP_SEQUENCE_ENCODER_TEMPORAL_INTERACTION,
    PerTimestepMLPSequenceEncoder,
    PointwiseMLPSequenceEncoder,
    TemporalMLPSequenceEncoder,
)

# =============================================================================
# Deterministic masked-mean pooling
# =============================================================================

from .masked_mean_pooling import (
    MASKED_MEAN_POOLING_COMPONENT_KIND,
    MASKED_MEAN_POOLING_COMPONENT_NAME,
    MASKED_MEAN_POOLING_HAZARD_CONDITIONED,
    MASKED_MEAN_POOLING_IMPLEMENTATION_VERSION,
    MASKED_MEAN_POOLING_OPERATION_NAME,
    MASKED_MEAN_POOLING_PADDING_POLICY,
    MASKED_MEAN_POOLING_PROJECTION_POLICY,
    MASKED_MEAN_POOLING_TEMPORAL_INTERACTION,
    MASKED_MEAN_POOLING_WEIGHT_POLICY,
    MASKED_MEAN_POOLING_ZERO_HISTORY_POLICIES,
    MaskedMeanPooling,
    MaskedMeanTemporalPooler,
    MaskedMeanTemporalPooling,
)

# =============================================================================
# Deterministic last-valid pooling
# =============================================================================

from .last_valid_pooling import (
    LAST_VALID_POOLING_COMPONENT_KIND,
    LAST_VALID_POOLING_COMPONENT_NAME,
    LAST_VALID_POOLING_HAZARD_CONDITIONED,
    LAST_VALID_POOLING_IMPLEMENTATION_VERSION,
    LAST_VALID_POOLING_MISSINGNESS_POLICY,
    LAST_VALID_POOLING_OPERATION_NAME,
    LAST_VALID_POOLING_PADDING_POLICY,
    LAST_VALID_POOLING_PROJECTION_POLICY,
    LAST_VALID_POOLING_TEMPORAL_INTERACTION,
    LAST_VALID_POOLING_WEIGHT_POLICY,
    LAST_VALID_POOLING_ZERO_HISTORY_POLICIES,
    LastObservationTemporalPooler,
    LastValidPooling,
    LastValidTemporalPooler,
    LastValidTemporalPooling,
)

# =============================================================================
# Detached diagnostics
# =============================================================================

from .diagnostics import (
    BASELINE_DIAGNOSTICS_SCHEMA_VERSION,
    BASELINE_DIAGNOSTICS_SCIENTIFIC_INTERPRETATION,
    BaselineDiagnostics,
    BaselinePoolingDiagnostics,
    BaselineSequenceDiagnostics,
    collect_baseline_diagnostics,
    collect_pooling_baseline_diagnostics,
    collect_sequence_baseline_diagnostics,
    diagnose_baseline_pipeline,
    diagnose_pooling_baseline,
    diagnose_sequence_baseline,
)

# =============================================================================
# Builders, dispatch, predicates, and thin execution helpers
# =============================================================================

from .baseline_encoders import (
    BASELINE_ENCODER_DISPATCHER_IMPLEMENTATION_VERSION,
    BASELINE_ENCODER_DISPATCHER_SCIENTIFIC_INTERPRETATION,
    BASELINE_SEQUENCE_ENCODER_TYPES,
    BASELINE_TEMPORAL_POOLER_TYPES,
    IMPLEMENTED_BASELINE_SEQUENCE_ENCODER_KINDS,
    IMPLEMENTED_BASELINE_TEMPORAL_POOLING_KINDS,
    RECOGNIZED_UNIMPLEMENTED_BASELINE_TEMPORAL_POOLING_KINDS,
    BaselineSequenceEncoderModule,
    BaselineTemporalPoolerModule,
    build_baseline_sequence_encoder,
    build_baseline_temporal_pooler,
    build_sequence_encoder,
    build_temporal_pooler,
    encode_baseline_history,
    encode_history,
    is_baseline_sequence_encoder,
    is_baseline_temporal_pooler,
    pool_baseline_sequence,
    pool_sequence,
    run_baseline_pipeline,
    run_pipeline,
)


# =============================================================================
# Public API
# =============================================================================


__all__ = (
    # -------------------------------------------------------------------------
    # Identity sequence baseline.
    # -------------------------------------------------------------------------
    "IDENTITY_SEQUENCE_ENCODER_COMPONENT_NAME",
    "IDENTITY_SEQUENCE_ENCODER_COMPONENT_KIND",
    "IDENTITY_SEQUENCE_ENCODER_OPERATION_NAME",
    "IDENTITY_SEQUENCE_ENCODER_IMPLEMENTATION_VERSION",
    "IDENTITY_SEQUENCE_ENCODER_TEMPORAL_INTERACTION",
    "IDENTITY_SEQUENCE_ENCODER_FEATURE_OBSERVATION_POLICY",
    "IDENTITY_SEQUENCE_ENCODER_TEMPORAL_COORDINATE_POLICY",
    "IDENTITY_SEQUENCE_ENCODER_PADDING_POLICY",
    "IdentitySequenceEncoder",
    "IdentityTemporalSequenceEncoder",

    # -------------------------------------------------------------------------
    # Linear-projection sequence baseline.
    # -------------------------------------------------------------------------
    "LINEAR_PROJECTION_SEQUENCE_ENCODER_COMPONENT_NAME",
    "LINEAR_PROJECTION_SEQUENCE_ENCODER_COMPONENT_KIND",
    "LINEAR_PROJECTION_SEQUENCE_ENCODER_OPERATION_NAME",
    "LINEAR_PROJECTION_SEQUENCE_ENCODER_IMPLEMENTATION_VERSION",
    "LINEAR_PROJECTION_SEQUENCE_ENCODER_TEMPORAL_INTERACTION",
    "LINEAR_PROJECTION_SEQUENCE_ENCODER_FEATURE_OBSERVATION_POLICY",
    "LINEAR_PROJECTION_SEQUENCE_ENCODER_TEMPORAL_COORDINATE_POLICY",
    "LINEAR_PROJECTION_SEQUENCE_ENCODER_PADDING_POLICY",
    "LINEAR_PROJECTION_SEQUENCE_ENCODER_ACTIVATION_POLICY",
    "LinearProjectionSequenceEncoder",
    "PerTimestepLinearSequenceEncoder",

    # -------------------------------------------------------------------------
    # Pointwise-MLP sequence baseline.
    # -------------------------------------------------------------------------
    "POINTWISE_MLP_SEQUENCE_ENCODER_COMPONENT_NAME",
    "POINTWISE_MLP_SEQUENCE_ENCODER_COMPONENT_KIND",
    "POINTWISE_MLP_SEQUENCE_ENCODER_OPERATION_NAME",
    "POINTWISE_MLP_SEQUENCE_ENCODER_IMPLEMENTATION_VERSION",
    "POINTWISE_MLP_SEQUENCE_ENCODER_TEMPORAL_INTERACTION",
    "POINTWISE_MLP_SEQUENCE_ENCODER_FEATURE_OBSERVATION_POLICY",
    "POINTWISE_MLP_SEQUENCE_ENCODER_TEMPORAL_COORDINATE_POLICY",
    "POINTWISE_MLP_SEQUENCE_ENCODER_PADDING_POLICY",
    "POINTWISE_MLP_SEQUENCE_ENCODER_LAYER_POLICY",
    "PointwiseMLPSequenceEncoder",
    "PerTimestepMLPSequenceEncoder",
    "TemporalMLPSequenceEncoder",

    # -------------------------------------------------------------------------
    # Masked-mean pooling.
    # -------------------------------------------------------------------------
    "MASKED_MEAN_POOLING_COMPONENT_NAME",
    "MASKED_MEAN_POOLING_COMPONENT_KIND",
    "MASKED_MEAN_POOLING_OPERATION_NAME",
    "MASKED_MEAN_POOLING_IMPLEMENTATION_VERSION",
    "MASKED_MEAN_POOLING_TEMPORAL_INTERACTION",
    "MASKED_MEAN_POOLING_HAZARD_CONDITIONED",
    "MASKED_MEAN_POOLING_WEIGHT_POLICY",
    "MASKED_MEAN_POOLING_PADDING_POLICY",
    "MASKED_MEAN_POOLING_PROJECTION_POLICY",
    "MASKED_MEAN_POOLING_ZERO_HISTORY_POLICIES",
    "MaskedMeanTemporalPooler",
    "MaskedMeanPooling",
    "MaskedMeanTemporalPooling",

    # -------------------------------------------------------------------------
    # Last-valid pooling.
    # -------------------------------------------------------------------------
    "LAST_VALID_POOLING_COMPONENT_NAME",
    "LAST_VALID_POOLING_COMPONENT_KIND",
    "LAST_VALID_POOLING_OPERATION_NAME",
    "LAST_VALID_POOLING_IMPLEMENTATION_VERSION",
    "LAST_VALID_POOLING_TEMPORAL_INTERACTION",
    "LAST_VALID_POOLING_HAZARD_CONDITIONED",
    "LAST_VALID_POOLING_WEIGHT_POLICY",
    "LAST_VALID_POOLING_PADDING_POLICY",
    "LAST_VALID_POOLING_PROJECTION_POLICY",
    "LAST_VALID_POOLING_MISSINGNESS_POLICY",
    "LAST_VALID_POOLING_ZERO_HISTORY_POLICIES",
    "LastValidTemporalPooler",
    "LastValidPooling",
    "LastValidTemporalPooling",
    "LastObservationTemporalPooler",

    # -------------------------------------------------------------------------
    # Diagnostics.
    # -------------------------------------------------------------------------
    "BASELINE_DIAGNOSTICS_SCHEMA_VERSION",
    "BASELINE_DIAGNOSTICS_SCIENTIFIC_INTERPRETATION",
    "BaselineSequenceDiagnostics",
    "BaselinePoolingDiagnostics",
    "BaselineDiagnostics",
    "diagnose_sequence_baseline",
    "diagnose_pooling_baseline",
    "diagnose_baseline_pipeline",
    "collect_sequence_baseline_diagnostics",
    "collect_pooling_baseline_diagnostics",
    "collect_baseline_diagnostics",

    # -------------------------------------------------------------------------
    # Dispatch capability vocabulary.
    # -------------------------------------------------------------------------
    "BASELINE_ENCODER_DISPATCHER_IMPLEMENTATION_VERSION",
    "BASELINE_ENCODER_DISPATCHER_SCIENTIFIC_INTERPRETATION",
    "IMPLEMENTED_BASELINE_SEQUENCE_ENCODER_KINDS",
    "IMPLEMENTED_BASELINE_TEMPORAL_POOLING_KINDS",
    "RECOGNIZED_UNIMPLEMENTED_BASELINE_TEMPORAL_POOLING_KINDS",

    # -------------------------------------------------------------------------
    # Supported module types and predicates.
    # -------------------------------------------------------------------------
    "BaselineSequenceEncoderModule",
    "BaselineTemporalPoolerModule",
    "BASELINE_SEQUENCE_ENCODER_TYPES",
    "BASELINE_TEMPORAL_POOLER_TYPES",
    "is_baseline_sequence_encoder",
    "is_baseline_temporal_pooler",

    # -------------------------------------------------------------------------
    # Builders and explicit execution helpers.
    # -------------------------------------------------------------------------
    "build_baseline_sequence_encoder",
    "build_baseline_temporal_pooler",
    "encode_baseline_history",
    "pool_baseline_sequence",
    "run_baseline_pipeline",

    # -------------------------------------------------------------------------
    # Compact aliases.
    # -------------------------------------------------------------------------
    "build_sequence_encoder",
    "build_temporal_pooler",
    "encode_history",
    "pool_sequence",
    "run_pipeline",
)
