"""
Stable constants and controlled vocabularies for the V2 model family.

Target path:
    urban_resilience_models/
        v2_hazard_conditioned_functional_ugnn/
            constants.py

This module answers:

    Which names, fields, modes, versions, and semantic categories are valid?

It does not answer:

    Which valid options should a particular model or experiment use?

Runtime and model-construction choices belong in ``config.py``. This file
must not define hidden dimensions, dropout rates, layer counts, selected
algorithms, or other experiment-specific defaults.

Design rules
------------
1. Keep the canonical package spelling as ``ugnn``.
2. Numeric hazard and relation IDs belong in versioned registries.
3. String constants do not replace runtime validation.
4. Canonical vocabulary may include future capabilities.
5. Target V2.0 capabilities describe planned scope.
6. Implemented capability collections must only contain working code paths.
7. Update schema or registry versions when stable meanings change.
8. Do not import PyTorch or implement model behavior here.
"""

from typing import Final


# =============================================================================
# Model-family identity
# =============================================================================

MODEL_FAMILY_ID: Final[str] = "v2_hazard_conditioned_functional_ugnn"
MODEL_FAMILY_SHORT_NAME: Final[str] = (
    "V2 Hazard-Conditioned Functional UGNN"
)
MODEL_FAMILY_DISPLAY_NAME: Final[str] = (
    "Hazard-Conditioned Functional Urban Graph Neural Network"
)
MODEL_FAMILY_RESEARCH_NAME: Final[str] = (
    "Hazard-Conditioned Functional Message Passing"
)

MODEL_FAMILY_VERSION: Final[str] = "v2.0.0-dev"
MODEL_STATUS: Final[str] = "design_skeleton_early_implementation"


# =============================================================================
# Serialized version-field names
# =============================================================================

FIELD_MODEL_FAMILY_VERSION: Final[str] = "model_family_version"
FIELD_MODEL_CONFIG_VERSION: Final[str] = "model_config_version"
FIELD_BATCH_SCHEMA_VERSION: Final[str] = "batch_schema_version"
FIELD_RELATION_REGISTRY_VERSION: Final[str] = (
    "relation_registry_version"
)
FIELD_HAZARD_REGISTRY_VERSION: Final[str] = "hazard_registry_version"
FIELD_FEATURE_CONTRACT_VERSION: Final[str] = (
    "feature_contract_version"
)
FIELD_PREDICTION_SCHEMA_VERSION: Final[str] = (
    "prediction_schema_version"
)
FIELD_EXPLANATION_SCHEMA_VERSION: Final[str] = (
    "explanation_schema_version"
)


# =============================================================================
# Current contract and registry versions
# =============================================================================

MODEL_CONFIG_VERSION: Final[str] = "0.1"
BATCH_SCHEMA_VERSION: Final[str] = "0.2"
RELATION_REGISTRY_VERSION: Final[str] = "0.1"
HAZARD_REGISTRY_VERSION: Final[str] = "0.1"
FEATURE_CONTRACT_VERSION: Final[str] = "0.1"
PREDICTION_SCHEMA_VERSION: Final[str] = "0.1"
EXPLANATION_SCHEMA_VERSION: Final[str] = "0.1"

REQUIRED_CONTRACT_VERSION_FIELDS: Final[tuple[str, ...]] = (
    FIELD_MODEL_FAMILY_VERSION,
    FIELD_MODEL_CONFIG_VERSION,
    FIELD_BATCH_SCHEMA_VERSION,
    FIELD_RELATION_REGISTRY_VERSION,
    FIELD_HAZARD_REGISTRY_VERSION,
    FIELD_FEATURE_CONTRACT_VERSION,
    FIELD_PREDICTION_SCHEMA_VERSION,
    FIELD_EXPLANATION_SCHEMA_VERSION,
)

CURRENT_CONTRACT_VERSION_PAIRS: Final[
    tuple[tuple[str, str], ...]
] = (
    (FIELD_MODEL_FAMILY_VERSION, MODEL_FAMILY_VERSION),
    (FIELD_MODEL_CONFIG_VERSION, MODEL_CONFIG_VERSION),
    (FIELD_BATCH_SCHEMA_VERSION, BATCH_SCHEMA_VERSION),
    (FIELD_RELATION_REGISTRY_VERSION, RELATION_REGISTRY_VERSION),
    (FIELD_HAZARD_REGISTRY_VERSION, HAZARD_REGISTRY_VERSION),
    (FIELD_FEATURE_CONTRACT_VERSION, FEATURE_CONTRACT_VERSION),
    (FIELD_PREDICTION_SCHEMA_VERSION, PREDICTION_SCHEMA_VERSION),
    (FIELD_EXPLANATION_SCHEMA_VERSION, EXPLANATION_SCHEMA_VERSION),
)


# =============================================================================
# Dataset and execution splits
# =============================================================================

SPLIT_TRAIN: Final[str] = "train"
SPLIT_VALIDATION: Final[str] = "validation"
SPLIT_TEST: Final[str] = "test"
SPLIT_INFERENCE: Final[str] = "inference"

CANONICAL_SPLITS: Final[tuple[str, ...]] = (
    SPLIT_TRAIN,
    SPLIT_VALIDATION,
    SPLIT_TEST,
    SPLIT_INFERENCE,
)


# =============================================================================
# Shared scope vocabulary
#
# In the packed-batch contract:
#   graph scope means one value per graph/scenario instance: [B]
#   node scope means one value per packed node: [N]
#
# A separate "batch" scope is intentionally not defined. A tensor of shape
# [B] is graph-scoped because B is the number of graph/scenario instances.
# =============================================================================

SCOPE_GRAPH: Final[str] = "graph"
SCOPE_NODE: Final[str] = "node"

CANONICAL_SCOPES: Final[tuple[str, ...]] = (
    SCOPE_GRAPH,
    SCOPE_NODE,
)


# =============================================================================
# Hazard vocabulary
#
# These are stable string names. Numeric IDs belong in the versioned hazard
# registry.
# =============================================================================

HAZARD_FLOOD: Final[str] = "flood"
HAZARD_HEAT: Final[str] = "heat"
HAZARD_OUTAGE: Final[str] = "outage"
HAZARD_ROAD_DISRUPTION: Final[str] = "road_disruption"
HAZARD_CIVIL_SECURITY_EVENT: Final[str] = "civil_security_event"
HAZARD_ALL_HAZARD: Final[str] = "all_hazard"

CANONICAL_HAZARD_NAMES: Final[tuple[str, ...]] = (
    HAZARD_FLOOD,
    HAZARD_HEAT,
    HAZARD_OUTAGE,
    HAZARD_ROAD_DISRUPTION,
    HAZARD_CIVIL_SECURITY_EVENT,
    HAZARD_ALL_HAZARD,
)

V2_0_TARGET_HAZARD_NAMES: Final[tuple[str, ...]] = (
    HAZARD_FLOOD,
    HAZARD_HEAT,
    HAZARD_CIVIL_SECURITY_EVENT,
)

# Update only when corresponding registry entries and model/data paths work.
V2_0_IMPLEMENTED_HAZARD_NAMES: Final[tuple[str, ...]] = ()


# =============================================================================
# Overall hazard-conditioning modes
# =============================================================================

HAZARD_CONDITIONING_NONE: Final[str] = "none"
HAZARD_CONDITIONING_EMBEDDING: Final[str] = "hazard_embedding"
HAZARD_CONDITIONING_EMBEDDING_SCENARIO: Final[str] = (
    "hazard_and_scenario"
)
HAZARD_CONDITIONING_RELATION_GATE: Final[str] = "relation_gate"
HAZARD_CONDITIONING_GATE_ATTENTION: Final[str] = (
    "relation_gate_and_attention"
)
HAZARD_CONDITIONING_QUERIED_MEMORY: Final[str] = (
    "hazard_queried_memory"
)
HAZARD_CONDITIONING_FULL: Final[str] = "full"

CANONICAL_HAZARD_CONDITIONING_MODES: Final[tuple[str, ...]] = (
    HAZARD_CONDITIONING_NONE,
    HAZARD_CONDITIONING_EMBEDDING,
    HAZARD_CONDITIONING_EMBEDDING_SCENARIO,
    HAZARD_CONDITIONING_RELATION_GATE,
    HAZARD_CONDITIONING_GATE_ATTENTION,
    HAZARD_CONDITIONING_QUERIED_MEMORY,
    HAZARD_CONDITIONING_FULL,
)

V2_0_TARGET_HAZARD_CONDITIONING_MODES: Final[tuple[str, ...]] = (
    HAZARD_CONDITIONING_NONE,
    HAZARD_CONDITIONING_EMBEDDING,
    HAZARD_CONDITIONING_RELATION_GATE,
    HAZARD_CONDITIONING_GATE_ATTENTION,
    HAZARD_CONDITIONING_QUERIED_MEMORY,
)

V2_0_IMPLEMENTED_HAZARD_CONDITIONING_MODES: Final[
    tuple[str, ...]
] = (
    HAZARD_CONDITIONING_NONE,
)


# =============================================================================
# Node-type vocabulary
#
# Raw feature spaces may differ by type. The stable model-facing contract
# eventually projects all node types into a shared hidden space [N, H].
# =============================================================================

NODE_TYPE_URBAN_UNIT: Final[str] = "urban_unit"
NODE_TYPE_CENSUS_TRACT: Final[str] = "census_tract"
NODE_TYPE_CENSUS_DIVISION: Final[str] = "census_division"
NODE_TYPE_MUNICIPALITY: Final[str] = "municipality"
NODE_TYPE_ADMINISTRATIVE_REGION: Final[str] = (
    "administrative_region"
)

NODE_TYPE_WATER_BODY: Final[str] = "water_body"
NODE_TYPE_FLOOD_ZONE: Final[str] = "flood_zone"
NODE_TYPE_HEAT_ISLAND: Final[str] = "heat_island"
NODE_TYPE_GREEN_SPACE: Final[str] = "green_space"
NODE_TYPE_CATCHMENT: Final[str] = "catchment"

NODE_TYPE_ROAD_SEGMENT: Final[str] = "road_segment"
NODE_TYPE_HOSPITAL: Final[str] = "hospital"
NODE_TYPE_SERVICE_FACILITY: Final[str] = "service_facility"
NODE_TYPE_COOLING_CENTER: Final[str] = "cooling_center"
NODE_TYPE_SHELTER: Final[str] = "shelter"

NODE_TYPE_DRAINAGE_ASSET: Final[str] = "drainage_asset"
NODE_TYPE_DRAINAGE_ZONE: Final[str] = "drainage_zone"
NODE_TYPE_CRITICAL_INFRASTRUCTURE: Final[str] = (
    "critical_infrastructure"
)

NODE_TYPE_WEATHER_STATION: Final[str] = "weather_station"
NODE_TYPE_HYDROMETRIC_STATION: Final[str] = (
    "hydrometric_station"
)

CANONICAL_NODE_TYPE_NAMES: Final[tuple[str, ...]] = (
    NODE_TYPE_URBAN_UNIT,
    NODE_TYPE_CENSUS_TRACT,
    NODE_TYPE_CENSUS_DIVISION,
    NODE_TYPE_MUNICIPALITY,
    NODE_TYPE_ADMINISTRATIVE_REGION,
    NODE_TYPE_WATER_BODY,
    NODE_TYPE_FLOOD_ZONE,
    NODE_TYPE_HEAT_ISLAND,
    NODE_TYPE_GREEN_SPACE,
    NODE_TYPE_CATCHMENT,
    NODE_TYPE_ROAD_SEGMENT,
    NODE_TYPE_HOSPITAL,
    NODE_TYPE_SERVICE_FACILITY,
    NODE_TYPE_COOLING_CENTER,
    NODE_TYPE_SHELTER,
    NODE_TYPE_DRAINAGE_ASSET,
    NODE_TYPE_DRAINAGE_ZONE,
    NODE_TYPE_CRITICAL_INFRASTRUCTURE,
    NODE_TYPE_WEATHER_STATION,
    NODE_TYPE_HYDROMETRIC_STATION,
)

V2_0_TARGET_NODE_TYPE_NAMES: Final[tuple[str, ...]] = (
    NODE_TYPE_CENSUS_TRACT,
    NODE_TYPE_CENSUS_DIVISION,
    NODE_TYPE_WATER_BODY,
    NODE_TYPE_FLOOD_ZONE,
    NODE_TYPE_HEAT_ISLAND,
    NODE_TYPE_GREEN_SPACE,
    NODE_TYPE_DRAINAGE_ASSET,
)

V2_0_IMPLEMENTED_NODE_TYPE_NAMES: Final[tuple[str, ...]] = ()


# =============================================================================
# Relation semantic roles
# =============================================================================

RELATION_ROLE_CONTROL: Final[str] = "control"
RELATION_ROLE_STRUCTURAL: Final[str] = "structural"
RELATION_ROLE_SPATIAL: Final[str] = "spatial"
RELATION_ROLE_ADMINISTRATIVE: Final[str] = "administrative"
RELATION_ROLE_TEMPORAL: Final[str] = "temporal"
RELATION_ROLE_EXPOSURE: Final[str] = "exposure"
RELATION_ROLE_PROTECTION: Final[str] = "protection"
RELATION_ROLE_ACCESS: Final[str] = "access"
RELATION_ROLE_DEPENDENCY: Final[str] = "dependency"
RELATION_ROLE_SIMILARITY: Final[str] = "similarity"
RELATION_ROLE_CROSS_SCALE: Final[str] = "cross_scale"
RELATION_ROLE_MEMORY: Final[str] = "memory"

CANONICAL_RELATION_ROLES: Final[tuple[str, ...]] = (
    RELATION_ROLE_CONTROL,
    RELATION_ROLE_STRUCTURAL,
    RELATION_ROLE_SPATIAL,
    RELATION_ROLE_ADMINISTRATIVE,
    RELATION_ROLE_TEMPORAL,
    RELATION_ROLE_EXPOSURE,
    RELATION_ROLE_PROTECTION,
    RELATION_ROLE_ACCESS,
    RELATION_ROLE_DEPENDENCY,
    RELATION_ROLE_SIMILARITY,
    RELATION_ROLE_CROSS_SCALE,
    RELATION_ROLE_MEMORY,
)


# =============================================================================
# Relation-family vocabulary
#
# These are semantic names only. Numeric IDs and richer metadata belong in
# ``relations/relation_registry.py``.
# =============================================================================

# Control relations
REL_IDENTITY_NO_EDGE: Final[str] = "identity_no_edge"
REL_RANDOM_PLACEBO: Final[str] = "random_placebo"
REL_CENTROID_KNN: Final[str] = "centroid_knn"

# Structural, spatial, and administrative relations
REL_SPATIAL_ADJACENCY: Final[str] = "spatial_adjacency"
REL_ADMINISTRATIVE_MEMBERSHIP: Final[str] = (
    "administrative_membership"
)

# Temporal relations
REL_TEMPORAL_MEMORY: Final[str] = "temporal_memory"
REL_HISTORICAL_EVENT_PROPAGATION: Final[str] = (
    "historical_event_propagation"
)

# Exposure relations
REL_HYDROLOGICAL_EXPOSURE: Final[str] = (
    "hydrological_exposure"
)
REL_FLOOD_ZONE_EXPOSURE: Final[str] = "flood_zone_exposure"
REL_LOW_ELEVATION_EXPOSURE: Final[str] = (
    "low_elevation_exposure"
)
REL_HEAT_EXPOSURE: Final[str] = "heat_exposure"
REL_IMPERVIOUS_SURFACE_EXPOSURE: Final[str] = (
    "impervious_surface_exposure"
)

# Protection and mitigation relations
REL_CANOPY_PROTECTION: Final[str] = "canopy_protection"
REL_COOLING_ACCESS: Final[str] = "cooling_access"

# Access relations
REL_SERVICE_ACCESS: Final[str] = "service_access"
REL_ROAD_ACCESS: Final[str] = "road_access"

# Dependency relations
REL_DRAINAGE_DEPENDENCY: Final[str] = "drainage_dependency"
REL_INFRASTRUCTURE_DEPENDENCY: Final[str] = (
    "infrastructure_dependency"
)
REL_CRITICAL_FACILITY_DEPENDENCY: Final[str] = (
    "critical_facility_dependency"
)

# Similarity relations
REL_REPORTING_SIMILARITY: Final[str] = "reporting_similarity"
REL_SOCIOECONOMIC_SIMILARITY: Final[str] = (
    "socioeconomic_similarity"
)

# Cross-scale relations
REL_CROSS_SCALE_PARENT: Final[str] = "cross_scale_parent"
REL_CROSS_SCALE_CHILD: Final[str] = "cross_scale_child"

CANONICAL_RELATION_NAMES: Final[tuple[str, ...]] = (
    REL_IDENTITY_NO_EDGE,
    REL_RANDOM_PLACEBO,
    REL_CENTROID_KNN,
    REL_SPATIAL_ADJACENCY,
    REL_ADMINISTRATIVE_MEMBERSHIP,
    REL_TEMPORAL_MEMORY,
    REL_HISTORICAL_EVENT_PROPAGATION,
    REL_HYDROLOGICAL_EXPOSURE,
    REL_FLOOD_ZONE_EXPOSURE,
    REL_LOW_ELEVATION_EXPOSURE,
    REL_HEAT_EXPOSURE,
    REL_IMPERVIOUS_SURFACE_EXPOSURE,
    REL_CANOPY_PROTECTION,
    REL_COOLING_ACCESS,
    REL_SERVICE_ACCESS,
    REL_ROAD_ACCESS,
    REL_DRAINAGE_DEPENDENCY,
    REL_INFRASTRUCTURE_DEPENDENCY,
    REL_CRITICAL_FACILITY_DEPENDENCY,
    REL_REPORTING_SIMILARITY,
    REL_SOCIOECONOMIC_SIMILARITY,
    REL_CROSS_SCALE_PARENT,
    REL_CROSS_SCALE_CHILD,
)

CONTROL_RELATION_NAMES: Final[frozenset[str]] = frozenset(
    {
        REL_IDENTITY_NO_EDGE,
        REL_RANDOM_PLACEBO,
        REL_CENTROID_KNN,
    }
)

STRUCTURAL_RELATION_NAMES: Final[frozenset[str]] = frozenset(
    {
        REL_SPATIAL_ADJACENCY,
        REL_ADMINISTRATIVE_MEMBERSHIP,
    }
)

TEMPORAL_RELATION_NAMES: Final[frozenset[str]] = frozenset(
    {
        REL_TEMPORAL_MEMORY,
        REL_HISTORICAL_EVENT_PROPAGATION,
    }
)

FUNCTIONAL_RELATION_NAMES: Final[frozenset[str]] = frozenset(
    {
        REL_HYDROLOGICAL_EXPOSURE,
        REL_FLOOD_ZONE_EXPOSURE,
        REL_LOW_ELEVATION_EXPOSURE,
        REL_HEAT_EXPOSURE,
        REL_IMPERVIOUS_SURFACE_EXPOSURE,
        REL_CANOPY_PROTECTION,
        REL_COOLING_ACCESS,
        REL_SERVICE_ACCESS,
        REL_ROAD_ACCESS,
        REL_DRAINAGE_DEPENDENCY,
        REL_INFRASTRUCTURE_DEPENDENCY,
        REL_CRITICAL_FACILITY_DEPENDENCY,
    }
)

SIMILARITY_RELATION_NAMES: Final[frozenset[str]] = frozenset(
    {
        REL_REPORTING_SIMILARITY,
        REL_SOCIOECONOMIC_SIMILARITY,
    }
)

CROSS_SCALE_RELATION_NAMES: Final[frozenset[str]] = frozenset(
    {
        REL_CROSS_SCALE_PARENT,
        REL_CROSS_SCALE_CHILD,
    }
)

NON_CONTROL_RELATION_NAMES: Final[frozenset[str]] = frozenset(
    set(CANONICAL_RELATION_NAMES) - set(CONTROL_RELATION_NAMES)
)

V2_0_CANDIDATE_RELATION_NAMES: Final[tuple[str, ...]] = (
    REL_RANDOM_PLACEBO,
    REL_CENTROID_KNN,
    REL_SPATIAL_ADJACENCY,
    REL_TEMPORAL_MEMORY,
    REL_HYDROLOGICAL_EXPOSURE,
    REL_HEAT_EXPOSURE,
    REL_CANOPY_PROTECTION,
    REL_DRAINAGE_DEPENDENCY,
)

# Update only when graph builders, registry entries, validation, and model paths
# for the corresponding relation are working.
V2_0_IMPLEMENTED_RELATION_NAMES: Final[tuple[str, ...]] = ()


# =============================================================================
# Relation-gate scope vocabulary
# =============================================================================

RELATION_GATE_SCOPE_GRAPH: Final[str] = "graph"
RELATION_GATE_SCOPE_TARGET_NODE: Final[str] = "target_node"
RELATION_GATE_SCOPE_SOURCE_NODE: Final[str] = "source_node"
RELATION_GATE_SCOPE_SOURCE_TARGET: Final[str] = "source_target"

CANONICAL_RELATION_GATE_SCOPES: Final[tuple[str, ...]] = (
    RELATION_GATE_SCOPE_GRAPH,
    RELATION_GATE_SCOPE_TARGET_NODE,
    RELATION_GATE_SCOPE_SOURCE_NODE,
    RELATION_GATE_SCOPE_SOURCE_TARGET,
)

V2_0_TARGET_RELATION_GATE_SCOPES: Final[tuple[str, ...]] = (
    RELATION_GATE_SCOPE_GRAPH,
    RELATION_GATE_SCOPE_TARGET_NODE,
)

V2_0_IMPLEMENTED_RELATION_GATE_SCOPES: Final[
    tuple[str, ...]
] = ()


# =============================================================================
# Relation-gate activation vocabulary
# =============================================================================

RELATION_GATE_ACTIVATION_SIGMOID: Final[str] = "sigmoid"
RELATION_GATE_ACTIVATION_SOFTMAX: Final[str] = "softmax"
RELATION_GATE_ACTIVATION_SPARSEMAX: Final[str] = "sparsemax"
RELATION_GATE_ACTIVATION_ENTMAX: Final[str] = "entmax"
RELATION_GATE_ACTIVATION_HARD_CONCRETE: Final[str] = (
    "hard_concrete"
)

CANONICAL_RELATION_GATE_ACTIVATIONS: Final[tuple[str, ...]] = (
    RELATION_GATE_ACTIVATION_SIGMOID,
    RELATION_GATE_ACTIVATION_SOFTMAX,
    RELATION_GATE_ACTIVATION_SPARSEMAX,
    RELATION_GATE_ACTIVATION_ENTMAX,
    RELATION_GATE_ACTIVATION_HARD_CONCRETE,
)

V2_0_TARGET_RELATION_GATE_ACTIVATIONS: Final[tuple[str, ...]] = (
    RELATION_GATE_ACTIVATION_SIGMOID,
)

V2_0_IMPLEMENTED_RELATION_GATE_ACTIVATIONS: Final[
    tuple[str, ...]
] = ()


# =============================================================================
# Edge-attention vocabulary
# =============================================================================

ATTENTION_MODE_UNIFORM: Final[str] = "uniform"
ATTENTION_MODE_SEMANTIC_WEIGHT: Final[str] = "semantic_weight"
ATTENTION_MODE_HAZARD_BLIND: Final[str] = "hazard_blind"
ATTENTION_MODE_HAZARD_CONDITIONED: Final[str] = (
    "hazard_conditioned"
)
ATTENTION_MODE_MULTIHEAD_HAZARD_CONDITIONED: Final[str] = (
    "multihead_hazard_conditioned"
)

CANONICAL_ATTENTION_MODES: Final[tuple[str, ...]] = (
    ATTENTION_MODE_UNIFORM,
    ATTENTION_MODE_SEMANTIC_WEIGHT,
    ATTENTION_MODE_HAZARD_BLIND,
    ATTENTION_MODE_HAZARD_CONDITIONED,
    ATTENTION_MODE_MULTIHEAD_HAZARD_CONDITIONED,
)

V2_0_TARGET_ATTENTION_MODES: Final[tuple[str, ...]] = (
    ATTENTION_MODE_UNIFORM,
    ATTENTION_MODE_HAZARD_CONDITIONED,
)

V2_0_IMPLEMENTED_ATTENTION_MODES: Final[tuple[str, ...]] = ()


ATTENTION_NORMALIZATION_TARGET_NODE: Final[str] = "target_node"
ATTENTION_NORMALIZATION_TARGET_NODE_RELATION: Final[str] = (
    "target_node_and_relation"
)
ATTENTION_NORMALIZATION_GLOBAL_RELATION: Final[str] = (
    "global_relation"
)
ATTENTION_NORMALIZATION_UNNORMALIZED_SIGMOID: Final[str] = (
    "unnormalized_sigmoid"
)

CANONICAL_ATTENTION_NORMALIZATION_MODES: Final[
    tuple[str, ...]
] = (
    ATTENTION_NORMALIZATION_TARGET_NODE,
    ATTENTION_NORMALIZATION_TARGET_NODE_RELATION,
    ATTENTION_NORMALIZATION_GLOBAL_RELATION,
    ATTENTION_NORMALIZATION_UNNORMALIZED_SIGMOID,
)

V2_0_TARGET_ATTENTION_NORMALIZATION_MODES: Final[
    tuple[str, ...]
] = (
    ATTENTION_NORMALIZATION_TARGET_NODE,
    ATTENTION_NORMALIZATION_TARGET_NODE_RELATION,
)

V2_0_IMPLEMENTED_ATTENTION_NORMALIZATION_MODES: Final[
    tuple[str, ...]
] = ()


ATTENTION_HEAD_REDUCTION_MEAN: Final[str] = "mean"
ATTENTION_HEAD_REDUCTION_WEIGHTED_MEAN: Final[str] = (
    "weighted_mean"
)
ATTENTION_HEAD_REDUCTION_MAX: Final[str] = "max"
ATTENTION_HEAD_REDUCTION_NONE: Final[str] = "none"

CANONICAL_ATTENTION_HEAD_REDUCTIONS: Final[tuple[str, ...]] = (
    ATTENTION_HEAD_REDUCTION_MEAN,
    ATTENTION_HEAD_REDUCTION_WEIGHTED_MEAN,
    ATTENTION_HEAD_REDUCTION_MAX,
    ATTENTION_HEAD_REDUCTION_NONE,
)

V2_0_TARGET_ATTENTION_HEAD_REDUCTIONS: Final[tuple[str, ...]] = (
    ATTENTION_HEAD_REDUCTION_MEAN,
)

V2_0_IMPLEMENTED_ATTENTION_HEAD_REDUCTIONS: Final[
    tuple[str, ...]
] = ()


# =============================================================================
# Relation-transform vocabulary
# =============================================================================

RELATION_TRANSFORM_SHARED: Final[str] = "shared"
RELATION_TRANSFORM_EMBEDDING_MODULATED: Final[str] = (
    "relation_embedding"
)
RELATION_TRANSFORM_PER_RELATION: Final[str] = "relation_specific"
RELATION_TRANSFORM_BASIS: Final[str] = "basis_decomposition"
RELATION_TRANSFORM_LOW_RANK: Final[str] = "low_rank"
RELATION_TRANSFORM_TYPED_MLP: Final[str] = "typed_mlp"

CANONICAL_RELATION_TRANSFORM_TYPES: Final[tuple[str, ...]] = (
    RELATION_TRANSFORM_SHARED,
    RELATION_TRANSFORM_EMBEDDING_MODULATED,
    RELATION_TRANSFORM_PER_RELATION,
    RELATION_TRANSFORM_BASIS,
    RELATION_TRANSFORM_LOW_RANK,
    RELATION_TRANSFORM_TYPED_MLP,
)

V2_0_TARGET_RELATION_TRANSFORM_TYPES: Final[tuple[str, ...]] = (
    RELATION_TRANSFORM_SHARED,
    RELATION_TRANSFORM_PER_RELATION,
)

V2_0_IMPLEMENTED_RELATION_TRANSFORM_TYPES: Final[
    tuple[str, ...]
] = ()


# =============================================================================
# Aggregation vocabulary
# =============================================================================

AGGREGATION_SUM: Final[str] = "sum"
AGGREGATION_MEAN: Final[str] = "mean"
AGGREGATION_MAX: Final[str] = "max"
AGGREGATION_DEGREE_NORMALIZED_SUM: Final[str] = (
    "degree_normalized_sum"
)
AGGREGATION_RELATION_WISE_SUM: Final[str] = (
    "relation_wise_sum"
)
AGGREGATION_RELATION_WISE_FUSION: Final[str] = (
    "relation_wise_fusion"
)

CANONICAL_AGGREGATION_TYPES: Final[tuple[str, ...]] = (
    AGGREGATION_SUM,
    AGGREGATION_MEAN,
    AGGREGATION_MAX,
    AGGREGATION_DEGREE_NORMALIZED_SUM,
    AGGREGATION_RELATION_WISE_SUM,
    AGGREGATION_RELATION_WISE_FUSION,
)

V2_0_TARGET_AGGREGATION_TYPES: Final[tuple[str, ...]] = (
    AGGREGATION_SUM,
    AGGREGATION_MEAN,
    AGGREGATION_DEGREE_NORMALIZED_SUM,
)

V2_0_IMPLEMENTED_AGGREGATION_TYPES: Final[tuple[str, ...]] = ()


# =============================================================================
# Edge-normalization vocabulary
# =============================================================================

EDGE_NORMALIZATION_NONE: Final[str] = "none"
EDGE_NORMALIZATION_SOURCE_DEGREE: Final[str] = "source_degree"
EDGE_NORMALIZATION_TARGET_DEGREE: Final[str] = "target_degree"
EDGE_NORMALIZATION_SYMMETRIC_DEGREE: Final[str] = (
    "symmetric_degree"
)
EDGE_NORMALIZATION_RELATION_DEGREE: Final[str] = (
    "relation_specific_degree"
)
EDGE_NORMALIZATION_DISTANCE_DECAY: Final[str] = "distance_decay"
EDGE_NORMALIZATION_SEMANTIC_WEIGHT: Final[str] = (
    "semantic_weight_normalization"
)

CANONICAL_EDGE_NORMALIZATION_TYPES: Final[tuple[str, ...]] = (
    EDGE_NORMALIZATION_NONE,
    EDGE_NORMALIZATION_SOURCE_DEGREE,
    EDGE_NORMALIZATION_TARGET_DEGREE,
    EDGE_NORMALIZATION_SYMMETRIC_DEGREE,
    EDGE_NORMALIZATION_RELATION_DEGREE,
    EDGE_NORMALIZATION_DISTANCE_DECAY,
    EDGE_NORMALIZATION_SEMANTIC_WEIGHT,
)

V2_0_TARGET_EDGE_NORMALIZATION_TYPES: Final[tuple[str, ...]] = (
    EDGE_NORMALIZATION_NONE,
    EDGE_NORMALIZATION_TARGET_DEGREE,
    EDGE_NORMALIZATION_RELATION_DEGREE,
)

V2_0_IMPLEMENTED_EDGE_NORMALIZATION_TYPES: Final[
    tuple[str, ...]
] = ()


# =============================================================================
# Memory encoder vocabulary
# =============================================================================

MEMORY_ENCODER_NONE: Final[str] = "none"
MEMORY_ENCODER_LAG: Final[str] = "lag"
MEMORY_ENCODER_GRU: Final[str] = "gru"
MEMORY_ENCODER_LSTM: Final[str] = "lstm"
MEMORY_ENCODER_TRANSFORMER: Final[str] = "transformer"

CANONICAL_MEMORY_ENCODER_TYPES: Final[tuple[str, ...]] = (
    MEMORY_ENCODER_NONE,
    MEMORY_ENCODER_LAG,
    MEMORY_ENCODER_GRU,
    MEMORY_ENCODER_LSTM,
    MEMORY_ENCODER_TRANSFORMER,
)

V2_0_TARGET_MEMORY_ENCODER_TYPES: Final[tuple[str, ...]] = (
    MEMORY_ENCODER_NONE,
    MEMORY_ENCODER_LAG,
    MEMORY_ENCODER_GRU,
    MEMORY_ENCODER_LSTM,
)

V2_0_IMPLEMENTED_MEMORY_ENCODER_TYPES: Final[tuple[str, ...]] = (
    MEMORY_ENCODER_NONE,
)


# =============================================================================
# Memory-query vocabulary
# =============================================================================

MEMORY_QUERY_NONE: Final[str] = "none"
MEMORY_QUERY_POST_CONCAT: Final[str] = "post_memory_concat"
MEMORY_QUERY_HAZARD_ATTENTION: Final[str] = "hazard_attention"
MEMORY_QUERY_SCENARIO_ATTENTION: Final[str] = "scenario_attention"

CANONICAL_MEMORY_QUERY_TYPES: Final[tuple[str, ...]] = (
    MEMORY_QUERY_NONE,
    MEMORY_QUERY_POST_CONCAT,
    MEMORY_QUERY_HAZARD_ATTENTION,
    MEMORY_QUERY_SCENARIO_ATTENTION,
)

V2_0_TARGET_MEMORY_QUERY_TYPES: Final[tuple[str, ...]] = (
    MEMORY_QUERY_NONE,
    MEMORY_QUERY_POST_CONCAT,
    MEMORY_QUERY_HAZARD_ATTENTION,
)

V2_0_IMPLEMENTED_MEMORY_QUERY_TYPES: Final[tuple[str, ...]] = (
    MEMORY_QUERY_NONE,
)


# =============================================================================
# Prediction-head vocabulary
# =============================================================================

PREDICTION_HEAD_REGRESSION: Final[str] = "regression"
PREDICTION_HEAD_NONNEGATIVE_REGRESSION: Final[str] = (
    "nonnegative_regression"
)
PREDICTION_HEAD_POISSON_RATE: Final[str] = "poisson_rate"
PREDICTION_HEAD_NEGATIVE_BINOMIAL: Final[str] = (
    "negative_binomial"
)
PREDICTION_HEAD_RANKING: Final[str] = "ranking"
PREDICTION_HEAD_MULTI_HORIZON: Final[str] = "multi_horizon"

CANONICAL_PREDICTION_HEAD_TYPES: Final[tuple[str, ...]] = (
    PREDICTION_HEAD_REGRESSION,
    PREDICTION_HEAD_NONNEGATIVE_REGRESSION,
    PREDICTION_HEAD_POISSON_RATE,
    PREDICTION_HEAD_NEGATIVE_BINOMIAL,
    PREDICTION_HEAD_RANKING,
    PREDICTION_HEAD_MULTI_HORIZON,
)

V2_0_TARGET_PREDICTION_HEAD_TYPES: Final[tuple[str, ...]] = (
    PREDICTION_HEAD_REGRESSION,
    PREDICTION_HEAD_NONNEGATIVE_REGRESSION,
    PREDICTION_HEAD_POISSON_RATE,
)

V2_0_IMPLEMENTED_PREDICTION_HEAD_TYPES: Final[
    tuple[str, ...]
] = ()


# =============================================================================
# Uncertainty-head vocabulary
#
# These options are model-output heads.
# =============================================================================

UNCERTAINTY_HEAD_NONE: Final[str] = "none"
UNCERTAINTY_HEAD_HETEROSCEDASTIC: Final[str] = (
    "heteroscedastic_variance"
)
UNCERTAINTY_HEAD_QUANTILE: Final[str] = "quantile"

CANONICAL_UNCERTAINTY_HEAD_TYPES: Final[tuple[str, ...]] = (
    UNCERTAINTY_HEAD_NONE,
    UNCERTAINTY_HEAD_HETEROSCEDASTIC,
    UNCERTAINTY_HEAD_QUANTILE,
)

V2_0_TARGET_UNCERTAINTY_HEAD_TYPES: Final[tuple[str, ...]] = (
    UNCERTAINTY_HEAD_NONE,
)

V2_0_IMPLEMENTED_UNCERTAINTY_HEAD_TYPES: Final[
    tuple[str, ...]
] = (
    UNCERTAINTY_HEAD_NONE,
)


# =============================================================================
# Uncertainty-method vocabulary
#
# These options are inference, ensemble, or calibration procedures and may be
# combined with uncertainty heads.
# =============================================================================

UNCERTAINTY_METHOD_NONE: Final[str] = "none"
UNCERTAINTY_METHOD_MC_DROPOUT: Final[str] = "mc_dropout"
UNCERTAINTY_METHOD_ENSEMBLE: Final[str] = "ensemble"
UNCERTAINTY_METHOD_CONFORMAL: Final[str] = "conformal"

CANONICAL_UNCERTAINTY_METHOD_TYPES: Final[tuple[str, ...]] = (
    UNCERTAINTY_METHOD_NONE,
    UNCERTAINTY_METHOD_MC_DROPOUT,
    UNCERTAINTY_METHOD_ENSEMBLE,
    UNCERTAINTY_METHOD_CONFORMAL,
)

V2_0_TARGET_UNCERTAINTY_METHOD_TYPES: Final[tuple[str, ...]] = (
    UNCERTAINTY_METHOD_NONE,
)

V2_0_IMPLEMENTED_UNCERTAINTY_METHOD_TYPES: Final[
    tuple[str, ...]
] = (
    UNCERTAINTY_METHOD_NONE,
)


# =============================================================================
# Reporting-bias vocabulary
# =============================================================================

REPORTING_BIAS_NONE: Final[str] = "none"
REPORTING_BIAS_COVARIATES: Final[str] = "reporting_covariates"
REPORTING_BIAS_MULTITASK: Final[str] = "reporting_multitask"
REPORTING_BIAS_LATENT_DECOMPOSITION: Final[str] = (
    "latent_reporting_decomposition"
)

CANONICAL_REPORTING_BIAS_TYPES: Final[tuple[str, ...]] = (
    REPORTING_BIAS_NONE,
    REPORTING_BIAS_COVARIATES,
    REPORTING_BIAS_MULTITASK,
    REPORTING_BIAS_LATENT_DECOMPOSITION,
)

V2_0_TARGET_REPORTING_BIAS_TYPES: Final[tuple[str, ...]] = (
    REPORTING_BIAS_NONE,
)

V2_0_IMPLEMENTED_REPORTING_BIAS_TYPES: Final[
    tuple[str, ...]
] = (
    REPORTING_BIAS_NONE,
)


# =============================================================================
# Weight-kind vocabulary
# =============================================================================

WEIGHT_KIND_SEMANTIC_EDGE: Final[str] = "semantic_edge_weight"
WEIGHT_KIND_NORMALIZATION: Final[str] = "normalization_weight"
WEIGHT_KIND_ATTENTION: Final[str] = "attention_weight"
WEIGHT_KIND_RELATION_GATE: Final[str] = "relation_gate_weight"

CANONICAL_WEIGHT_KINDS: Final[tuple[str, ...]] = (
    WEIGHT_KIND_SEMANTIC_EDGE,
    WEIGHT_KIND_NORMALIZATION,
    WEIGHT_KIND_ATTENTION,
    WEIGHT_KIND_RELATION_GATE,
)


# =============================================================================
# Canonical batch-field names
# =============================================================================

FIELD_EXTERNAL_NODE_IDS: Final[str] = "external_node_ids"
FIELD_EXTERNAL_EDGE_IDS: Final[str] = "external_edge_ids"

FIELD_NODE_FEATURES: Final[str] = "node_features"
FIELD_NODE_STATE: Final[str] = "node_state"
FIELD_NODE_TYPE: Final[str] = "node_type"
FIELD_NODE_BATCH_INDEX: Final[str] = "node_batch_index"
FIELD_GRAPH_PTR: Final[str] = "graph_ptr"

FIELD_HISTORY_SEQUENCES: Final[str] = "history_sequences"
FIELD_HISTORY_MASK: Final[str] = "history_mask"

FIELD_HAZARD_IDS: Final[str] = "hazard_ids"
FIELD_HAZARD_FEATURES: Final[str] = "hazard_features"
FIELD_SCENARIO_FEATURES: Final[str] = "scenario_features"

FIELD_EDGE_INDEX: Final[str] = "edge_index"
FIELD_EDGE_RELATION_TYPE: Final[str] = "edge_relation_type"
FIELD_EDGE_ATTRIBUTES: Final[str] = "edge_attributes"
FIELD_EDGE_BATCH_INDEX: Final[str] = "edge_batch_index"
FIELD_SEMANTIC_EDGE_WEIGHT: Final[str] = "semantic_edge_weight"

FIELD_TARGETS: Final[str] = "targets"
FIELD_TARGET_MASK: Final[str] = "target_mask"

CANONICAL_BATCH_FIELDS: Final[tuple[str, ...]] = (
    FIELD_EXTERNAL_NODE_IDS,
    FIELD_EXTERNAL_EDGE_IDS,
    FIELD_NODE_FEATURES,
    FIELD_NODE_STATE,
    FIELD_NODE_TYPE,
    FIELD_NODE_BATCH_INDEX,
    FIELD_GRAPH_PTR,
    FIELD_HISTORY_SEQUENCES,
    FIELD_HISTORY_MASK,
    FIELD_HAZARD_IDS,
    FIELD_HAZARD_FEATURES,
    FIELD_SCENARIO_FEATURES,
    FIELD_EDGE_INDEX,
    FIELD_EDGE_RELATION_TYPE,
    FIELD_EDGE_ATTRIBUTES,
    FIELD_EDGE_BATCH_INDEX,
    FIELD_SEMANTIC_EDGE_WEIGHT,
    FIELD_TARGETS,
    FIELD_TARGET_MASK,
)

REQUIRED_INFERENCE_BATCH_FIELDS: Final[tuple[str, ...]] = (
    FIELD_EXTERNAL_NODE_IDS,
    FIELD_NODE_BATCH_INDEX,
    FIELD_HAZARD_IDS,
    FIELD_EDGE_INDEX,
    FIELD_EDGE_RELATION_TYPE,
)

SUPERVISION_BATCH_FIELDS: Final[tuple[str, ...]] = (
    FIELD_TARGETS,
    FIELD_TARGET_MASK,
)


# =============================================================================
# Temporal-causality and availability fields
# =============================================================================

FIELD_ORIGIN_TIME: Final[str] = "origin_time"
FIELD_HISTORY_START_TIME: Final[str] = "history_start_time"
FIELD_HISTORY_END_TIME: Final[str] = "history_end_time"
FIELD_FEATURE_AVAILABILITY_CUTOFF: Final[str] = (
    "feature_availability_cutoff"
)
FIELD_TARGET_START_TIME: Final[str] = "target_start_time"
FIELD_TARGET_END_TIME: Final[str] = "target_end_time"
FIELD_FORECAST_HORIZON: Final[str] = "forecast_horizon"

FIELD_EDGE_VALID_FROM: Final[str] = "edge_valid_from"
FIELD_EDGE_VALID_TO: Final[str] = "edge_valid_to"
FIELD_EDGE_OBSERVATION_TIME: Final[str] = "edge_observation_time"

CANONICAL_TEMPORAL_FIELDS: Final[tuple[str, ...]] = (
    FIELD_ORIGIN_TIME,
    FIELD_HISTORY_START_TIME,
    FIELD_HISTORY_END_TIME,
    FIELD_FEATURE_AVAILABILITY_CUTOFF,
    FIELD_TARGET_START_TIME,
    FIELD_TARGET_END_TIME,
    FIELD_FORECAST_HORIZON,
    FIELD_EDGE_VALID_FROM,
    FIELD_EDGE_VALID_TO,
    FIELD_EDGE_OBSERVATION_TIME,
)

REQUIRED_INFERENCE_TEMPORAL_FIELDS: Final[tuple[str, ...]] = (
    FIELD_ORIGIN_TIME,
    FIELD_HISTORY_START_TIME,
    FIELD_HISTORY_END_TIME,
    FIELD_FEATURE_AVAILABILITY_CUTOFF,
    FIELD_FORECAST_HORIZON,
)

REQUIRED_SUPERVISED_TEMPORAL_FIELDS: Final[tuple[str, ...]] = (
    FIELD_TARGET_START_TIME,
    FIELD_TARGET_END_TIME,
)


# =============================================================================
# Prediction-field names
# =============================================================================

FIELD_TARGET_NAMES: Final[str] = "target_names"
FIELD_PREDICTION_MEAN: Final[str] = "prediction_mean"
FIELD_COUNT_RATE: Final[str] = "count_rate"
FIELD_RANKING_SCORE: Final[str] = "ranking_score"
FIELD_OUTPUT_TRANSFORMATION: Final[str] = "output_transformation"
FIELD_HIGHER_IS_RISKIER: Final[str] = "higher_is_riskier"
FIELD_UNCERTAINTY: Final[str] = "uncertainty"

CANONICAL_PREDICTION_FIELDS: Final[tuple[str, ...]] = (
    FIELD_EXTERNAL_NODE_IDS,
    FIELD_NODE_BATCH_INDEX,
    FIELD_HAZARD_IDS,
    FIELD_ORIGIN_TIME,
    FIELD_FORECAST_HORIZON,
    FIELD_TARGET_NAMES,
    FIELD_PREDICTION_MEAN,
    FIELD_COUNT_RATE,
    FIELD_RANKING_SCORE,
    FIELD_OUTPUT_TRANSFORMATION,
    FIELD_HIGHER_IS_RISKIER,
    FIELD_UNCERTAINTY,
)


# =============================================================================
# Explanation-field names
# =============================================================================

FIELD_RELATION_GATES: Final[str] = "relation_gates"
FIELD_EDGE_ATTENTION: Final[str] = "edge_attention"
FIELD_TEMPORAL_ATTENTION: Final[str] = "temporal_attention"
FIELD_PATHWAY_SCORES: Final[str] = "pathway_scores"
FIELD_TOP_RELATION_FAMILIES: Final[str] = (
    "top_relation_families"
)
FIELD_TOP_EDGES: Final[str] = "top_edges"
FIELD_TOP_NEIGHBORS: Final[str] = "top_neighbors"
FIELD_TOP_HISTORY_PERIODS: Final[str] = "top_history_periods"
FIELD_COUNTERFACTUAL_DELTAS: Final[str] = (
    "counterfactual_deltas"
)
FIELD_ATTENTION_HEAD_REDUCTION: Final[str] = (
    "attention_head_reduction"
)
FIELD_NORMALIZATION_SCOPE: Final[str] = "normalization_scope"
FIELD_LAYER_INDEX: Final[str] = "layer_index"

CANONICAL_EXPLANATION_FIELDS: Final[tuple[str, ...]] = (
    FIELD_RELATION_GATES,
    FIELD_EDGE_ATTENTION,
    FIELD_TEMPORAL_ATTENTION,
    FIELD_PATHWAY_SCORES,
    FIELD_TOP_RELATION_FAMILIES,
    FIELD_TOP_EDGES,
    FIELD_TOP_NEIGHBORS,
    FIELD_TOP_HISTORY_PERIODS,
    FIELD_COUNTERFACTUAL_DELTAS,
    FIELD_ATTENTION_HEAD_REDUCTION,
    FIELD_NORMALIZATION_SCOPE,
    FIELD_LAYER_INDEX,
)


# =============================================================================
# Provenance-field names
# =============================================================================

FIELD_CHECKPOINT_ID: Final[str] = "checkpoint_id"
FIELD_RUN_ID: Final[str] = "run_id"
FIELD_EXPERIMENT_NAME: Final[str] = "experiment_name"
FIELD_EXPERIMENT_FAMILY: Final[str] = "experiment_family"
FIELD_DATASET_VERSION: Final[str] = "dataset_version"
FIELD_GRAPH_VERSION: Final[str] = "graph_version"
FIELD_RANDOM_SEED: Final[str] = "random_seed"
FIELD_CONFIG_HASH: Final[str] = "config_hash"

CANONICAL_PROVENANCE_FIELDS: Final[tuple[str, ...]] = (
    FIELD_CHECKPOINT_ID,
    FIELD_RUN_ID,
    FIELD_EXPERIMENT_NAME,
    FIELD_EXPERIMENT_FAMILY,
    FIELD_DATASET_VERSION,
    FIELD_GRAPH_VERSION,
    FIELD_RANDOM_SEED,
    FIELD_CONFIG_HASH,
    *REQUIRED_CONTRACT_VERSION_FIELDS,
)


# =============================================================================
# Implementation-state vocabulary
# =============================================================================

IMPLEMENTATION_STATE_CANONICAL: Final[str] = "canonical"
IMPLEMENTATION_STATE_TARGET: Final[str] = "target"
IMPLEMENTATION_STATE_IMPLEMENTED: Final[str] = "implemented"
IMPLEMENTATION_STATE_EXPERIMENTAL: Final[str] = "experimental"
IMPLEMENTATION_STATE_DEPRECATED: Final[str] = "deprecated"

CANONICAL_IMPLEMENTATION_STATES: Final[tuple[str, ...]] = (
    IMPLEMENTATION_STATE_CANONICAL,
    IMPLEMENTATION_STATE_TARGET,
    IMPLEMENTATION_STATE_IMPLEMENTED,
    IMPLEMENTATION_STATE_EXPERIMENTAL,
    IMPLEMENTATION_STATE_DEPRECATED,
)


# =============================================================================
# Notes for validators
# =============================================================================
#
# ``typing.Final`` communicates intent to type checkers but does not enforce
# runtime immutability or validity.
#
# Runtime validators in ``config.py``, ``schemas.py``, and the registries must:
#
# - reject unknown vocabulary values;
# - reject canonical-but-unimplemented modes in strict execution mode;
# - raise NotImplementedError for recognized but unavailable capabilities;
# - validate relation and hazard registry versions;
# - validate schema compatibility;
# - validate temporal causality and feature availability.
#
# This module intentionally defines no dynamic ``__all__``. Callers should use
# explicit imports rather than wildcard imports.
