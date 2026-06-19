"""
Canonical relation registry for the V2 hazard-conditioned functional UGNN.

Target path:
    urban_resilience_models/
        v2_hazard_conditioned_functional_ugnn/
            relations/
                relation_registry.py

This module owns:

- the concrete V2 edge-relation ontology;
- stable relation IDs;
- exact endpoint-pair contracts;
- current software capability declarations;
- immutable semantic and operational registry views;
- ontology hierarchy traversal;
- deterministic dense runtime relation indices;
- registry serialization and fingerprints;
- source-registry verification for compiled artifacts.

It does not own:

- hazard-relation priors;
- graph construction;
- edge-table or tensor validation;
- temporal leakage inspection of actual graph artifacts;
- message-passing implementations;
- filesystem persistence.

Identity contract
-----------------
``RelationSpec.relation_id`` is a stable ontology identity. It is never used
directly as a model tensor index.

A compiled registry assigns:

    relation_index = 0, 1, ..., R - 1

Model-facing tensors such as ``edge_relation_type`` and relation-gate outputs
must use ``relation_index``.

Every graph artifact, checkpoint, prediction artifact, and explanation artifact
that contains relation indices should preserve:

    relation_index -> stable relation_id -> canonical relation name

Semantic versus operational identity
------------------------------------
The semantic ontology and current software capabilities are intentionally
separated.

Semantic identity includes:

- relation meaning;
- endpoints;
- direction;
- hierarchy;
- construction mode;
- temporal mode;
- edge attributes.

Operational identity includes:

- current implementation state;
- current message-passing availability;
- current training availability;
- current explanation availability.

Adding an implementation for an existing relation changes the operational
fingerprint but not the semantic fingerprint.

No-edge and feature-only concepts
---------------------------------
``identity_no_edge`` is a topology mode describing the absence of edges. It is
not registered as an edge relation.

``impervious_surface_exposure`` is currently excluded from the edge registry.
Impervious fraction should remain a node feature until a legitimate source
entity such as a land-surface patch or impervious-surface zone is introduced.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field, fields
from enum import StrEnum
from hashlib import sha256
import json
from types import MappingProxyType
from typing import Any, Final, Iterator, Mapping, Sequence

from .. import constants as C
from .relation_types import (
    EdgeAttributeKind,
    EdgeAttributeSpec,
    MissingValuePolicy,
    RELATION_SPEC_SCHEMA_VERSION,
    TOPOLOGY_ONLY_NO_EDGE_NAME,
    RelationConstructionMode,
    RelationDirection,
    RelationEvidenceType,
    RelationExplanationPolicy,
    RelationLeakageRisk,
    RelationSpec,
    RelationTemporalMode,
    TEMPORAL_FIELD_EDGE_LAG,
    validate_relation_spec_collection,
)


# =============================================================================
# Registry schema identity
# =============================================================================


DEFAULT_RELATION_REGISTRY_NAME: Final[str] = (
    "v2_canonical_edge_relation_registry"
)

RELATION_REGISTRY_SNAPSHOT_SCHEMA_VERSION: Final[str] = "0.2"
RELATION_CAPABILITY_SCHEMA_VERSION: Final[str] = "0.1"
ENDPOINT_CONTRACT_SCHEMA_VERSION: Final[str] = "0.1"
COMPILED_RELATION_REGISTRY_SCHEMA_VERSION: Final[str] = "0.2"


# =============================================================================
# Explicitly excluded canonical concepts
# =============================================================================


REL_IMPERVIOUS_SURFACE_EXPOSURE: Final[str] = (
    "impervious_surface_exposure"
)

EDGE_REGISTRY_EXCLUDED_CANONICAL_NAMES: Final[frozenset[str]] = (
    frozenset(
        {
            TOPOLOGY_ONLY_NO_EDGE_NAME,
            REL_IMPERVIOUS_SURFACE_EXPOSURE,
        }
    )
)


# =============================================================================
# Compilation policy
# =============================================================================


class HierarchyCompilationPolicy(StrEnum):
    """
    Policy for selecting ontology parents and descendants together.

    REJECT_OVERLAP
        Reject any selection containing both an ancestor and descendant.

    LEAF_ONLY
        Remove selected ancestors whenever selected descendants exist.

    ALLOW_OVERLAP
        Preserve both. This requires an external guarantee that their edge
        sets do not duplicate the same mechanism.
    """

    REJECT_OVERLAP = "reject_overlap"
    LEAF_ONLY = "leaf_only"
    ALLOW_OVERLAP = "allow_overlap"


# =============================================================================
# Coordinated constants access
# =============================================================================


def _required_string_constant(name: str) -> str:
    if not hasattr(C, name):
        raise RuntimeError(
            f"constants.py is missing required constant {name!r}."
        )

    value = getattr(C, name)

    if not isinstance(value, str) or not value.strip():
        raise RuntimeError(
            f"constants.py constant {name!r} must be a non-empty string."
        )

    return value


def _required_string_tuple_constant(name: str) -> tuple[str, ...]:
    if not hasattr(C, name):
        raise RuntimeError(
            f"constants.py is missing required constant {name!r}."
        )

    value = getattr(C, name)

    if not isinstance(value, tuple):
        raise RuntimeError(
            f"constants.py constant {name!r} must be a tuple."
        )

    if any(
        not isinstance(item, str) or not item.strip()
        for item in value
    ):
        raise RuntimeError(
            f"constants.py constant {name!r} must contain only "
            "non-empty strings."
        )

    return value


def _ordered_unique(values: Sequence[str]) -> tuple[str, ...]:
    return tuple(dict.fromkeys(values))


# Relation names
REL_RANDOM_PLACEBO = _required_string_constant(
    "REL_RANDOM_PLACEBO"
)
REL_CENTROID_KNN = _required_string_constant(
    "REL_CENTROID_KNN"
)
REL_SPATIAL_ADJACENCY = _required_string_constant(
    "REL_SPATIAL_ADJACENCY"
)
REL_ADMINISTRATIVE_MEMBERSHIP = _required_string_constant(
    "REL_ADMINISTRATIVE_MEMBERSHIP"
)
REL_TEMPORAL_MEMORY = _required_string_constant(
    "REL_TEMPORAL_MEMORY"
)
REL_HISTORICAL_EVENT_PROPAGATION = _required_string_constant(
    "REL_HISTORICAL_EVENT_PROPAGATION"
)
REL_HYDROLOGICAL_EXPOSURE = _required_string_constant(
    "REL_HYDROLOGICAL_EXPOSURE"
)
REL_FLOOD_ZONE_EXPOSURE = _required_string_constant(
    "REL_FLOOD_ZONE_EXPOSURE"
)
REL_LOW_ELEVATION_EXPOSURE = _required_string_constant(
    "REL_LOW_ELEVATION_EXPOSURE"
)
REL_HEAT_EXPOSURE = _required_string_constant(
    "REL_HEAT_EXPOSURE"
)
REL_CANOPY_PROTECTION = _required_string_constant(
    "REL_CANOPY_PROTECTION"
)
REL_COOLING_ACCESS = _required_string_constant(
    "REL_COOLING_ACCESS"
)
REL_SERVICE_ACCESS = _required_string_constant(
    "REL_SERVICE_ACCESS"
)
REL_ROAD_ACCESS = _required_string_constant(
    "REL_ROAD_ACCESS"
)
REL_INFRASTRUCTURE_DEPENDENCY = _required_string_constant(
    "REL_INFRASTRUCTURE_DEPENDENCY"
)
REL_DRAINAGE_DEPENDENCY = _required_string_constant(
    "REL_DRAINAGE_DEPENDENCY"
)
REL_CRITICAL_FACILITY_DEPENDENCY = _required_string_constant(
    "REL_CRITICAL_FACILITY_DEPENDENCY"
)
REL_REPORTING_SIMILARITY = _required_string_constant(
    "REL_REPORTING_SIMILARITY"
)
REL_SOCIOECONOMIC_SIMILARITY = _required_string_constant(
    "REL_SOCIOECONOMIC_SIMILARITY"
)
REL_CROSS_SCALE_PARENT = _required_string_constant(
    "REL_CROSS_SCALE_PARENT"
)
REL_CROSS_SCALE_CHILD = _required_string_constant(
    "REL_CROSS_SCALE_CHILD"
)

# Semantic roles
ROLE_CONTROL = _required_string_constant(
    "RELATION_ROLE_CONTROL"
)
ROLE_SPATIAL = _required_string_constant(
    "RELATION_ROLE_SPATIAL"
)
ROLE_ADMINISTRATIVE = _required_string_constant(
    "RELATION_ROLE_ADMINISTRATIVE"
)
ROLE_MEMORY = _required_string_constant(
    "RELATION_ROLE_MEMORY"
)
ROLE_EXPOSURE = _required_string_constant(
    "RELATION_ROLE_EXPOSURE"
)
ROLE_PROTECTION = _required_string_constant(
    "RELATION_ROLE_PROTECTION"
)
ROLE_ACCESS = _required_string_constant(
    "RELATION_ROLE_ACCESS"
)
ROLE_DEPENDENCY = _required_string_constant(
    "RELATION_ROLE_DEPENDENCY"
)
ROLE_SIMILARITY = _required_string_constant(
    "RELATION_ROLE_SIMILARITY"
)
ROLE_CROSS_SCALE = _required_string_constant(
    "RELATION_ROLE_CROSS_SCALE"
)

# Node types
NODE_URBAN_UNIT = _required_string_constant(
    "NODE_TYPE_URBAN_UNIT"
)
NODE_CENSUS_TRACT = _required_string_constant(
    "NODE_TYPE_CENSUS_TRACT"
)
NODE_CENSUS_DIVISION = _required_string_constant(
    "NODE_TYPE_CENSUS_DIVISION"
)
NODE_MUNICIPALITY = _required_string_constant(
    "NODE_TYPE_MUNICIPALITY"
)
NODE_WATER_BODY = _required_string_constant(
    "NODE_TYPE_WATER_BODY"
)
NODE_FLOOD_ZONE = _required_string_constant(
    "NODE_TYPE_FLOOD_ZONE"
)
NODE_HEAT_ISLAND = _required_string_constant(
    "NODE_TYPE_HEAT_ISLAND"
)
NODE_GREEN_SPACE = _required_string_constant(
    "NODE_TYPE_GREEN_SPACE"
)
NODE_ROAD_SEGMENT = _required_string_constant(
    "NODE_TYPE_ROAD_SEGMENT"
)
NODE_HOSPITAL = _required_string_constant(
    "NODE_TYPE_HOSPITAL"
)
NODE_SERVICE_FACILITY = _required_string_constant(
    "NODE_TYPE_SERVICE_FACILITY"
)
NODE_DRAINAGE_ASSET = _required_string_constant(
    "NODE_TYPE_DRAINAGE_ASSET"
)
NODE_CRITICAL_INFRASTRUCTURE = _required_string_constant(
    "NODE_TYPE_CRITICAL_INFRASTRUCTURE"
)

# Implementation vocabulary
IMPLEMENTATION_STATE_TARGET = _required_string_constant(
    "IMPLEMENTATION_STATE_TARGET"
)
IMPLEMENTATION_STATE_IMPLEMENTED = _required_string_constant(
    "IMPLEMENTATION_STATE_IMPLEMENTED"
)
IMPLEMENTATION_STATE_DEPRECATED = _required_string_constant(
    "IMPLEMENTATION_STATE_DEPRECATED"
)

CANONICAL_IMPLEMENTATION_STATES = _required_string_tuple_constant(
    "CANONICAL_IMPLEMENTATION_STATES"
)


PREDICTION_UNIT_NODE_TYPES: Final[tuple[str, ...]] = (
    _ordered_unique(
        (
            NODE_URBAN_UNIT,
            NODE_CENSUS_TRACT,
            NODE_CENSUS_DIVISION,
            NODE_MUNICIPALITY,
        )
    )
)

HIERARCHICAL_SPATIAL_NODE_TYPES: Final[tuple[str, ...]] = (
    NODE_CENSUS_TRACT,
    NODE_MUNICIPALITY,
    NODE_CENSUS_DIVISION,
)

SERVICE_NODE_TYPES: Final[tuple[str, ...]] = (
    _ordered_unique(
        (
            NODE_HOSPITAL,
            NODE_SERVICE_FACILITY,
        )
    )
)


# =============================================================================
# Stable relation IDs
#
# IDs are intentionally sparse and grouped by semantic domain.
# Never reuse a removed or deprecated ID.
# =============================================================================


RELATION_ID_RANDOM_PLACEBO: Final[int] = 100
RELATION_ID_CENTROID_KNN: Final[int] = 110

RELATION_ID_SPATIAL_ADJACENCY: Final[int] = 200
RELATION_ID_ADMINISTRATIVE_MEMBERSHIP: Final[int] = 210

RELATION_ID_TEMPORAL_MEMORY: Final[int] = 300
RELATION_ID_HISTORICAL_EVENT_PROPAGATION: Final[int] = 310

RELATION_ID_HYDROLOGICAL_EXPOSURE: Final[int] = 400
RELATION_ID_FLOOD_ZONE_EXPOSURE: Final[int] = 410
RELATION_ID_LOW_ELEVATION_EXPOSURE: Final[int] = 420
RELATION_ID_HEAT_EXPOSURE: Final[int] = 430

RELATION_ID_CANOPY_PROTECTION: Final[int] = 500
RELATION_ID_COOLING_ACCESS: Final[int] = 510

RELATION_ID_SERVICE_ACCESS: Final[int] = 600
RELATION_ID_ROAD_ACCESS: Final[int] = 610

RELATION_ID_INFRASTRUCTURE_DEPENDENCY: Final[int] = 700
RELATION_ID_DRAINAGE_DEPENDENCY: Final[int] = 710
RELATION_ID_CRITICAL_FACILITY_DEPENDENCY: Final[int] = 720

RELATION_ID_REPORTING_SIMILARITY: Final[int] = 800
RELATION_ID_SOCIOECONOMIC_SIMILARITY: Final[int] = 810

RELATION_ID_CROSS_SCALE_PARENT: Final[int] = 900
RELATION_ID_CROSS_SCALE_CHILD: Final[int] = 910


# =============================================================================
# Reusable edge-attribute contracts
#
# Names use concise, unit-explicit vocabulary and should be treated as stable.
# =============================================================================


ATTRIBUTE_DISTANCE_M: Final[EdgeAttributeSpec] = EdgeAttributeSpec(
    name="distance_m",
    description=(
        "Spatial or network distance associated with the edge."
    ),
    kind=EdgeAttributeKind.FLOAT,
    unit="m",
    minimum=0.0,
)

ATTRIBUTE_SHARED_BOUNDARY_LENGTH_M: Final[EdgeAttributeSpec] = (
    EdgeAttributeSpec(
        name="shared_boundary_length_m",
        description=(
            "Length of the shared geographic boundary between adjacent "
            "spatial units."
        ),
        kind=EdgeAttributeKind.FLOAT,
        unit="m",
        minimum=0.0,
    )
)

ATTRIBUTE_KNN_RANK: Final[EdgeAttributeSpec] = EdgeAttributeSpec(
    name="knn_rank",
    description=(
        "One-based nearest-neighbor rank used to construct a kNN edge."
    ),
    kind=EdgeAttributeKind.INTEGER,
    minimum=1,
)

ATTRIBUTE_CONTROL_GENERATOR_ID: Final[EdgeAttributeSpec] = (
    EdgeAttributeSpec(
        name="control_generator_id",
        description=(
            "Identifier of the control-graph generation procedure."
        ),
        kind=EdgeAttributeKind.IDENTIFIER,
    )
)

ATTRIBUTE_EDGE_LAG: Final[EdgeAttributeSpec] = EdgeAttributeSpec(
    name=TEMPORAL_FIELD_EDGE_LAG,
    description=(
        "Positive temporal lag from source state to target state."
    ),
    kind=EdgeAttributeKind.INTEGER,
    unit="months",
    minimum=1,
)

ATTRIBUTE_OVERLAP_FRACTION: Final[EdgeAttributeSpec] = (
    EdgeAttributeSpec(
        name="overlap_fraction",
        description=(
            "Fraction of target geometry overlapped or influenced by "
            "the source geometry."
        ),
        kind=EdgeAttributeKind.FLOAT,
        minimum=0.0,
        maximum=1.0,
    )
)

ATTRIBUTE_EXPOSURE_INTENSITY: Final[EdgeAttributeSpec] = (
    EdgeAttributeSpec(
        name="exposure_intensity",
        description=(
            "Nonnegative source-to-target exposure intensity."
        ),
        kind=EdgeAttributeKind.FLOAT,
        minimum=0.0,
    )
)

ATTRIBUTE_ELEVATION_DIFFERENCE_M: Final[EdgeAttributeSpec] = (
    EdgeAttributeSpec(
        name="elevation_difference_m",
        description=(
            "Source elevation minus target elevation."
        ),
        kind=EdgeAttributeKind.FLOAT,
        unit="m",
    )
)

ATTRIBUTE_TRAVEL_TIME_MIN: Final[EdgeAttributeSpec] = (
    EdgeAttributeSpec(
        name="travel_time_min",
        description=(
            "Estimated source-to-target travel time under the declared "
            "transport assumptions."
        ),
        kind=EdgeAttributeKind.FLOAT,
        unit="min",
        minimum=0.0,
    )
)

ATTRIBUTE_ACCESSIBILITY_SCORE: Final[EdgeAttributeSpec] = (
    EdgeAttributeSpec(
        name="accessibility_score",
        description=(
            "Normalized service or transport accessibility score."
        ),
        kind=EdgeAttributeKind.FLOAT,
        minimum=0.0,
        maximum=1.0,
    )
)

ATTRIBUTE_DEPENDENCY_STRENGTH: Final[EdgeAttributeSpec] = (
    EdgeAttributeSpec(
        name="dependency_strength",
        description=(
            "Normalized functional dependency strength."
        ),
        kind=EdgeAttributeKind.FLOAT,
        minimum=0.0,
        maximum=1.0,
    )
)

ATTRIBUTE_CAPACITY_SCORE: Final[EdgeAttributeSpec] = EdgeAttributeSpec(
    name="capacity_score",
    description=(
        "Nonnegative standardized service or infrastructure capacity."
    ),
    kind=EdgeAttributeKind.FLOAT,
    minimum=0.0,
    missing_value_policy=(
        MissingValuePolicy.NULLABLE_RAW_MASK_REQUIRED
    ),
)

ATTRIBUTE_SIMILARITY_SCORE: Final[EdgeAttributeSpec] = (
    EdgeAttributeSpec(
        name="similarity_score",
        description=(
            "Normalized similarity between source and target units."
        ),
        kind=EdgeAttributeKind.FLOAT,
        minimum=0.0,
        maximum=1.0,
    )
)

ATTRIBUTE_MEMBERSHIP_WEIGHT: Final[EdgeAttributeSpec] = (
    EdgeAttributeSpec(
        name="membership_weight",
        description=(
            "Fractional membership weight used for aggregation or "
            "cross-scale transfer."
        ),
        kind=EdgeAttributeKind.FLOAT,
        minimum=0.0,
        maximum=1.0,
    )
)

ATTRIBUTE_ADMINISTRATIVE_LEVEL: Final[EdgeAttributeSpec] = (
    EdgeAttributeSpec(
        name="administrative_level",
        description=(
            "Administrative level of the target parent unit."
        ),
        kind=EdgeAttributeKind.CATEGORICAL,
        categorical_closed_vocabulary=True,
        categorical_values=(
            NODE_MUNICIPALITY,
            NODE_CENSUS_DIVISION,
        ),
    )
)


# =============================================================================
# Validation and serialization helpers
# =============================================================================


def _require_nonempty_string(name: str, value: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string.")


def _require_mapping(
    name: str,
    value: Any,
) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must be a mapping.")

    return value


def _reject_unknown_fields(
    object_type: type[Any],
    payload: Mapping[str, Any],
) -> None:
    allowed = {
        definition.name
        for definition in fields(object_type)
        if definition.init
    }
    unknown = sorted(set(payload) - allowed)

    if unknown:
        raise ValueError(
            f"Unknown fields for {object_type.__name__}: {unknown}."
        )


def _as_tuple(
    name: str,
    value: Any,
) -> tuple[Any, ...]:
    if isinstance(value, tuple):
        return value

    if isinstance(value, list):
        return tuple(value)

    raise TypeError(f"{name} must be a list or tuple.")


def _canonical_json(payload: Mapping[str, Any]) -> str:
    return json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )


def _fingerprint(payload: Mapping[str, Any]) -> str:
    return sha256(
        _canonical_json(payload).encode("utf-8")
    ).hexdigest()


def _same_type_pairs(
    node_types: Sequence[str],
) -> tuple[tuple[str, str], ...]:
    return tuple(
        (node_type, node_type)
        for node_type in node_types
    )


def _cartesian_pairs(
    source_types: Sequence[str],
    target_types: Sequence[str],
) -> tuple[tuple[str, str], ...]:
    return tuple(
        (source_type, target_type)
        for source_type in source_types
        for target_type in target_types
    )


# =============================================================================
# Exact endpoint-pair contract
# =============================================================================


@dataclass(slots=True, frozen=True)
class RelationEndpointContract:
    """
    Exact endpoint-pair restrictions for one relation.

    ``RelationSpec.source_node_types`` and ``target_node_types`` define broad
    admissible type sets. When ``allowed_pairs`` is nonempty, this contract
    narrows their Cartesian product to explicitly permitted pairs.

    This is especially important for hierarchical relations. For example:

        census_tract -> municipality
        municipality -> census_division

    must not imply:

        census_tract -> census_division
        municipality -> municipality
    """

    allowed_pairs: tuple[tuple[str, str], ...] = ()

    schema_version: str = ENDPOINT_CONTRACT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        _require_nonempty_string(
            "endpoint contract schema_version",
            self.schema_version,
        )

        normalized: list[tuple[str, str]] = []

        for index, pair in enumerate(self.allowed_pairs):
            if not isinstance(pair, tuple) or len(pair) != 2:
                raise TypeError(
                    f"allowed_pairs[{index}] must be a two-item tuple."
                )

            source_type, target_type = pair
            _require_nonempty_string(
                f"allowed_pairs[{index}][0]",
                source_type,
            )
            _require_nonempty_string(
                f"allowed_pairs[{index}][1]",
                target_type,
            )

            normalized.append(
                (source_type, target_type)
            )

        duplicates = sorted(
            pair
            for pair, count in Counter(normalized).items()
            if count > 1
        )

        if duplicates:
            raise ValueError(
                f"Duplicate endpoint pairs: {duplicates}."
            )

    @property
    def explicit(self) -> bool:
        return bool(self.allowed_pairs)

    def validate_against(
        self,
        specification: RelationSpec,
    ) -> None:
        if not isinstance(specification, RelationSpec):
            raise TypeError(
                "specification must be a RelationSpec."
            )

        for source_type, target_type in self.allowed_pairs:
            if not specification.supports_node_pair(
                source_type,
                target_type,
            ):
                raise ValueError(
                    f"Endpoint pair {(source_type, target_type)!r} is "
                    f"incompatible with relation "
                    f"{specification.name!r}."
                )

    def permits(
        self,
        specification: RelationSpec,
        source_node_type: str,
        target_node_type: str,
    ) -> bool:
        if not specification.supports_node_pair(
            source_node_type,
            target_node_type,
        ):
            return False

        if not self.allowed_pairs:
            return True

        if (
            source_node_type,
            target_node_type,
        ) in self.allowed_pairs:
            return True

        if specification.is_undirected:
            return (
                target_node_type,
                source_node_type,
            ) in self.allowed_pairs

        return False

    def to_dict(self) -> dict[str, Any]:
        return {
            "allowed_pairs": [
                [source_type, target_type]
                for source_type, target_type
                in self.allowed_pairs
            ],
            "schema_version": self.schema_version,
        }

    @classmethod
    def from_dict(
        cls,
        payload: Mapping[str, Any],
    ) -> RelationEndpointContract:
        mapping = dict(
            _require_mapping(
                "RelationEndpointContract",
                payload,
            )
        )
        _reject_unknown_fields(cls, mapping)

        if "allowed_pairs" in mapping:
            raw_pairs = _as_tuple(
                "allowed_pairs",
                mapping["allowed_pairs"],
            )
            mapping["allowed_pairs"] = tuple(
                tuple(
                    _as_tuple(
                        f"allowed_pairs[{index}]",
                        pair,
                    )
                )
                for index, pair in enumerate(raw_pairs)
            )

        return cls(**mapping)


# =============================================================================
# Current software capability declaration
# =============================================================================


@dataclass(slots=True, frozen=True)
class RelationCapability:
    """
    Current software-release capability for one semantic relation.

    This object is deliberately separate from ``RelationSpec`` so adding an
    implementation does not change the semantic registry fingerprint.
    """

    implementation_state: str = IMPLEMENTATION_STATE_TARGET

    message_passing_allowed: bool = False
    training_allowed: bool = False

    explanation_policy: RelationExplanationPolicy = (
        RelationExplanationPolicy.EXCLUDED
    )

    schema_version: str = RELATION_CAPABILITY_SCHEMA_VERSION

    def __post_init__(self) -> None:
        self.validate()

    @property
    def implemented(self) -> bool:
        return (
            self.implementation_state
            == IMPLEMENTATION_STATE_IMPLEMENTED
        )

    @property
    def deprecated(self) -> bool:
        return (
            self.implementation_state
            == IMPLEMENTATION_STATE_DEPRECATED
        )

    @property
    def available_for_message_passing(self) -> bool:
        return (
            self.implemented
            and not self.deprecated
            and self.message_passing_allowed
        )

    @property
    def available_for_training(self) -> bool:
        return (
            self.available_for_message_passing
            and self.training_allowed
        )

    def validate(self) -> None:
        if (
            self.implementation_state
            not in CANONICAL_IMPLEMENTATION_STATES
        ):
            raise ValueError(
                "Unknown implementation state "
                f"{self.implementation_state!r}."
            )

        if not isinstance(
            self.explanation_policy,
            RelationExplanationPolicy,
        ):
            raise TypeError(
                "explanation_policy must be a "
                "RelationExplanationPolicy."
            )

        _require_nonempty_string(
            "capability schema_version",
            self.schema_version,
        )

        if (
            not self.implemented
            and (
                self.message_passing_allowed
                or self.training_allowed
            )
        ):
            raise ValueError(
                "Unimplemented relations cannot be enabled for message "
                "passing or training."
            )

        if (
            self.training_allowed
            and not self.message_passing_allowed
        ):
            raise ValueError(
                "training_allowed=True requires "
                "message_passing_allowed=True."
            )

        if self.deprecated and (
            self.message_passing_allowed
            or self.training_allowed
        ):
            raise ValueError(
                "Deprecated relations cannot remain operational."
            )

    def validate_against(
        self,
        specification: RelationSpec,
    ) -> None:
        if (
            specification.is_control
            and self.explanation_policy
            == RelationExplanationPolicy.ALLOWED
        ):
            raise ValueError(
                f"Control relation {specification.name!r} cannot be "
                "available as an ordinary scientific explanation."
            )

        if (
            not self.available_for_message_passing
            and self.explanation_policy
            == RelationExplanationPolicy.ALLOWED
        ):
            raise ValueError(
                f"Relation {specification.name!r} cannot be an ordinary "
                "message-path explanation while message passing is "
                "unavailable."
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "implementation_state": self.implementation_state,
            "message_passing_allowed": (
                self.message_passing_allowed
            ),
            "training_allowed": self.training_allowed,
            "explanation_policy": self.explanation_policy.value,
            "schema_version": self.schema_version,
        }

    @classmethod
    def from_dict(
        cls,
        payload: Mapping[str, Any],
    ) -> RelationCapability:
        mapping = dict(
            _require_mapping(
                "RelationCapability",
                payload,
            )
        )
        _reject_unknown_fields(cls, mapping)

        if "explanation_policy" in mapping:
            mapping["explanation_policy"] = (
                RelationExplanationPolicy(
                    mapping["explanation_policy"]
                )
            )

        return cls(**mapping)


# =============================================================================
# Canonical registry entry
# =============================================================================


@dataclass(slots=True, frozen=True)
class RelationRegistryEntry:
    """
    One semantic relation, its precise endpoint contract, and current
    operational capability.
    """

    specification: RelationSpec
    endpoint_contract: RelationEndpointContract
    capability: RelationCapability

    def __post_init__(self) -> None:
        if not isinstance(
            self.specification,
            RelationSpec,
        ):
            raise TypeError(
                "specification must be a RelationSpec."
            )

        if not isinstance(
            self.endpoint_contract,
            RelationEndpointContract,
        ):
            raise TypeError(
                "endpoint_contract must be a "
                "RelationEndpointContract."
            )

        if not isinstance(
            self.capability,
            RelationCapability,
        ):
            raise TypeError(
                "capability must be a RelationCapability."
            )

        self.specification.validate()
        self.endpoint_contract.validate_against(
            self.specification
        )
        self.capability.validate_against(
            self.specification
        )

    @property
    def relation_id(self) -> int:
        return self.specification.relation_id

    @property
    def name(self) -> str:
        return self.specification.name

    @property
    def implemented(self) -> bool:
        return self.capability.implemented

    @property
    def available_for_message_passing(self) -> bool:
        return (
            self.capability.available_for_message_passing
        )

    @property
    def available_for_training(self) -> bool:
        return self.capability.available_for_training

    def permits_endpoint_pair(
        self,
        source_node_type: str,
        target_node_type: str,
    ) -> bool:
        return self.endpoint_contract.permits(
            self.specification,
            source_node_type,
            target_node_type,
        )

    def semantic_dict(self) -> dict[str, Any]:
        specification = self.specification.to_dict()

        # These fields exist in RelationSpec for backward compatibility with
        # earlier contracts, but operational capability is authoritative here.
        for field_name in (
            "implementation_state",
            "message_passing_allowed",
            "training_allowed",
            "explanation_policy",
        ):
            specification.pop(field_name, None)

        return {
            "specification": specification,
            "endpoint_contract": (
                self.endpoint_contract.to_dict()
            ),
        }

    def operational_dict(self) -> dict[str, Any]:
        return {
            **self.semantic_dict(),
            "capability": self.capability.to_dict(),
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "specification": self.specification.to_dict(),
            "endpoint_contract": (
                self.endpoint_contract.to_dict()
            ),
            "capability": self.capability.to_dict(),
        }

    @classmethod
    def from_dict(
        cls,
        payload: Mapping[str, Any],
    ) -> RelationRegistryEntry:
        mapping = dict(
            _require_mapping(
                "RelationRegistryEntry",
                payload,
            )
        )
        _reject_unknown_fields(cls, mapping)

        return cls(
            specification=RelationSpec.from_dict(
                _require_mapping(
                    "RelationRegistryEntry.specification",
                    mapping["specification"],
                )
            ),
            endpoint_contract=(
                RelationEndpointContract.from_dict(
                    _require_mapping(
                        (
                            "RelationRegistryEntry."
                            "endpoint_contract"
                        ),
                        mapping["endpoint_contract"],
                    )
                )
            ),
            capability=RelationCapability.from_dict(
                _require_mapping(
                    "RelationRegistryEntry.capability",
                    mapping["capability"],
                )
            ),
        )


# =============================================================================
# Semantic relation construction helpers
# =============================================================================


def _semantic_spec(
    *,
    relation_id: int,
    name: str,
    display_name: str,
    description: str,
    semantic_role: str,
    source_node_types: tuple[str, ...],
    target_node_types: tuple[str, ...],
    direction: RelationDirection,
    evidence_type: RelationEvidenceType,
    construction_mode: RelationConstructionMode,
    leakage_risk: RelationLeakageRisk,
    temporal_mode: RelationTemporalMode,
    parent_relation_name: str | None = None,
    reverse_relation_name: str | None = None,
    allows_self_loops: bool = False,
    required_edge_attributes: tuple[EdgeAttributeSpec, ...] = (),
    optional_edge_attributes: tuple[EdgeAttributeSpec, ...] = (),
    tags: tuple[str, ...] = (),
) -> RelationSpec:
    """
    Build a capability-neutral semantic specification.

    Current operational availability is declared separately in
    ``RelationCapability``.
    """

    is_control = (
        name in C.CONTROL_RELATION_NAMES
        and name != TOPOLOGY_ONLY_NO_EDGE_NAME
    )

    return RelationSpec(
        relation_id=relation_id,
        name=name,
        display_name=display_name,
        description=description,
        semantic_role=semantic_role,
        source_node_types=source_node_types,
        target_node_types=target_node_types,
        direction=direction,
        evidence_type=evidence_type,
        construction_mode=construction_mode,
        leakage_risk=leakage_risk,
        temporal_mode=temporal_mode,
        parent_relation_name=parent_relation_name,
        reverse_relation_name=reverse_relation_name,
        implementation_state=IMPLEMENTATION_STATE_TARGET,
        is_control=is_control,
        allow_any_node_type=False,
        allows_self_loops=allows_self_loops,
        message_passing_allowed=False,
        training_allowed=False,
        explanation_policy=(
            RelationExplanationPolicy.EXCLUDED
        ),
        required_edge_attributes=required_edge_attributes,
        optional_edge_attributes=optional_edge_attributes,
        tags=tags,
    )


def _current_capability(
    relation_name: str,
    *,
    is_control: bool,
) -> RelationCapability:
    implemented_names = frozenset(
        getattr(
            C,
            "V2_0_IMPLEMENTED_RELATION_NAMES",
            (),
        )
    )

    if relation_name not in implemented_names:
        return RelationCapability(
            implementation_state=IMPLEMENTATION_STATE_TARGET,
            message_passing_allowed=False,
            training_allowed=False,
            explanation_policy=(
                RelationExplanationPolicy.EXCLUDED
            ),
        )

    return RelationCapability(
        implementation_state=(
            IMPLEMENTATION_STATE_IMPLEMENTED
        ),
        message_passing_allowed=True,
        training_allowed=True,
        explanation_policy=(
            RelationExplanationPolicy.DIAGNOSTIC_ONLY
            if is_control
            else RelationExplanationPolicy.ALLOWED
        ),
    )


def _entry(
    *,
    specification: RelationSpec,
    allowed_pairs: tuple[tuple[str, str], ...],
) -> RelationRegistryEntry:
    return RelationRegistryEntry(
        specification=specification,
        endpoint_contract=RelationEndpointContract(
            allowed_pairs=allowed_pairs,
        ),
        capability=_current_capability(
            specification.name,
            is_control=specification.is_control,
        ),
    )


# =============================================================================
# Canonical registry construction
# =============================================================================


def build_default_relation_entries(
) -> tuple[RelationRegistryEntry, ...]:
    """
    Build the complete V2 edge-relation registry.

    Broad ontology families and specialized child relations may coexist in
    this canonical registry. Runtime compilation rejects simultaneous
    parent-child activation by default.
    """

    same_prediction_unit_pairs = _same_type_pairs(
        PREDICTION_UNIT_NODE_TYPES
    )

    hierarchical_parent_pairs = (
        (NODE_CENSUS_TRACT, NODE_MUNICIPALITY),
        (NODE_MUNICIPALITY, NODE_CENSUS_DIVISION),
    )

    hierarchical_child_pairs = tuple(
        (target_type, source_type)
        for source_type, target_type
        in hierarchical_parent_pairs
    )

    service_access_pairs = _cartesian_pairs(
        PREDICTION_UNIT_NODE_TYPES,
        SERVICE_NODE_TYPES,
    )

    entries = (
        # ------------------------------------------------------------------
        # Control relations
        # ------------------------------------------------------------------
        _entry(
            specification=_semantic_spec(
                relation_id=RELATION_ID_RANDOM_PLACEBO,
                name=REL_RANDOM_PLACEBO,
                display_name="Random placebo",
                description=(
                    "Synthetic type-preserving random edges used as a "
                    "topology placebo. They must never be interpreted as "
                    "urban mechanisms."
                ),
                semantic_role=ROLE_CONTROL,
                source_node_types=(
                    PREDICTION_UNIT_NODE_TYPES
                ),
                target_node_types=(
                    PREDICTION_UNIT_NODE_TYPES
                ),
                direction=RelationDirection.DIRECTED,
                evidence_type=(
                    RelationEvidenceType.SYNTHETIC_CONTROL
                ),
                construction_mode=(
                    RelationConstructionMode.SYNTHETIC_CONTROL
                ),
                leakage_risk=RelationLeakageRisk.NONE,
                temporal_mode=RelationTemporalMode.STATIC,
                optional_edge_attributes=(
                    ATTRIBUTE_CONTROL_GENERATOR_ID,
                ),
                tags=(
                    "control",
                    "placebo",
                    "type_preserving",
                    "topology_ablation",
                ),
            ),
            allowed_pairs=same_prediction_unit_pairs,
        ),
        _entry(
            specification=_semantic_spec(
                relation_id=RELATION_ID_CENTROID_KNN,
                name=REL_CENTROID_KNN,
                display_name="Centroid kNN",
                description=(
                    "Generic same-type nearest-neighbor edges among "
                    "urban prediction units, used as a geometric "
                    "topology control."
                ),
                semantic_role=ROLE_CONTROL,
                source_node_types=(
                    PREDICTION_UNIT_NODE_TYPES
                ),
                target_node_types=(
                    PREDICTION_UNIT_NODE_TYPES
                ),
                direction=RelationDirection.DIRECTED,
                evidence_type=RelationEvidenceType.DERIVED,
                construction_mode=(
                    RelationConstructionMode.GEOMETRIC
                ),
                leakage_risk=RelationLeakageRisk.LOW,
                temporal_mode=RelationTemporalMode.STATIC,
                required_edge_attributes=(
                    ATTRIBUTE_DISTANCE_M,
                    ATTRIBUTE_KNN_RANK,
                ),
                tags=(
                    "control",
                    "knn",
                    "same_type",
                    "geometric",
                    "topology_ablation",
                ),
            ),
            allowed_pairs=same_prediction_unit_pairs,
        ),

        # ------------------------------------------------------------------
        # Spatial relations
        # ------------------------------------------------------------------
        _entry(
            specification=_semantic_spec(
                relation_id=RELATION_ID_SPATIAL_ADJACENCY,
                name=REL_SPATIAL_ADJACENCY,
                display_name="Spatial adjacency",
                description=(
                    "Geographic contiguity between urban units at the "
                    "same spatial scale."
                ),
                semantic_role=ROLE_SPATIAL,
                source_node_types=(
                    PREDICTION_UNIT_NODE_TYPES
                ),
                target_node_types=(
                    PREDICTION_UNIT_NODE_TYPES
                ),
                direction=RelationDirection.UNDIRECTED,
                evidence_type=RelationEvidenceType.DERIVED,
                construction_mode=(
                    RelationConstructionMode.GEOMETRIC
                ),
                leakage_risk=RelationLeakageRisk.NONE,
                temporal_mode=RelationTemporalMode.STATIC,
                optional_edge_attributes=(
                    ATTRIBUTE_SHARED_BOUNDARY_LENGTH_M,
                    ATTRIBUTE_DISTANCE_M,
                ),
                tags=(
                    "spatial",
                    "contiguity",
                    "same_scale",
                    "structural",
                ),
            ),
            allowed_pairs=same_prediction_unit_pairs,
        ),

        # ------------------------------------------------------------------
        # Temporal and memory relations
        # ------------------------------------------------------------------
        _entry(
            specification=_semantic_spec(
                relation_id=RELATION_ID_TEMPORAL_MEMORY,
                name=REL_TEMPORAL_MEMORY,
                display_name="Temporal memory",
                description=(
                    "Directed lagged connection from an urban unit's "
                    "historical state to a later state of the same unit "
                    "type."
                ),
                semantic_role=ROLE_MEMORY,
                source_node_types=(
                    PREDICTION_UNIT_NODE_TYPES
                ),
                target_node_types=(
                    PREDICTION_UNIT_NODE_TYPES
                ),
                direction=RelationDirection.DIRECTED,
                evidence_type=RelationEvidenceType.DERIVED,
                construction_mode=(
                    RelationConstructionMode.AS_OF_ORIGIN
                ),
                leakage_risk=RelationLeakageRisk.LOW,
                temporal_mode=RelationTemporalMode.LAGGED,
                required_edge_attributes=(
                    ATTRIBUTE_EDGE_LAG,
                ),
                tags=(
                    "temporal",
                    "memory",
                    "lagged",
                    "same_type",
                ),
            ),
            allowed_pairs=same_prediction_unit_pairs,
        ),
        _entry(
            specification=_semantic_spec(
                relation_id=(
                    RELATION_ID_HISTORICAL_EVENT_PROPAGATION
                ),
                name=REL_HISTORICAL_EVENT_PROPAGATION,
                display_name="Historical event propagation",
                description=(
                    "Lagged propagation of historical event burden "
                    "between same-scale urban units, constructed only "
                    "from information available by the prediction "
                    "origin."
                ),
                semantic_role=ROLE_MEMORY,
                source_node_types=(
                    PREDICTION_UNIT_NODE_TYPES
                ),
                target_node_types=(
                    PREDICTION_UNIT_NODE_TYPES
                ),
                direction=RelationDirection.DIRECTED,
                evidence_type=(
                    RelationEvidenceType.LEARNED_FROM_DATA
                ),
                construction_mode=(
                    RelationConstructionMode.AS_OF_ORIGIN
                ),
                leakage_risk=RelationLeakageRisk.HIGH,
                temporal_mode=RelationTemporalMode.LAGGED,
                parent_relation_name=REL_TEMPORAL_MEMORY,
                required_edge_attributes=(
                    ATTRIBUTE_EDGE_LAG,
                ),
                optional_edge_attributes=(
                    ATTRIBUTE_EXPOSURE_INTENSITY,
                    ATTRIBUTE_DISTANCE_M,
                ),
                tags=(
                    "temporal",
                    "event_history",
                    "propagation",
                    "high_leakage_audit",
                ),
            ),
            allowed_pairs=same_prediction_unit_pairs,
        ),

        # ------------------------------------------------------------------
        # Hydrological exposure
        # ------------------------------------------------------------------
        _entry(
            specification=_semantic_spec(
                relation_id=(
                    RELATION_ID_HYDROLOGICAL_EXPOSURE
                ),
                name=REL_HYDROLOGICAL_EXPOSURE,
                display_name="Hydrological exposure",
                description=(
                    "Broad hydrological influence of water bodies or "
                    "flood-related zones on urban units."
                ),
                semantic_role=ROLE_EXPOSURE,
                source_node_types=(
                    NODE_WATER_BODY,
                    NODE_FLOOD_ZONE,
                ),
                target_node_types=(
                    PREDICTION_UNIT_NODE_TYPES
                ),
                direction=RelationDirection.DIRECTED,
                evidence_type=RelationEvidenceType.HYBRID,
                construction_mode=(
                    RelationConstructionMode.GEOMETRIC
                ),
                leakage_risk=RelationLeakageRisk.LOW,
                temporal_mode=RelationTemporalMode.STATIC,
                optional_edge_attributes=(
                    ATTRIBUTE_DISTANCE_M,
                    ATTRIBUTE_OVERLAP_FRACTION,
                    ATTRIBUTE_EXPOSURE_INTENSITY,
                ),
                tags=(
                    "flood",
                    "hydrology",
                    "exposure",
                    "abstract_family",
                ),
            ),
            allowed_pairs=_cartesian_pairs(
                (
                    NODE_WATER_BODY,
                    NODE_FLOOD_ZONE,
                ),
                PREDICTION_UNIT_NODE_TYPES,
            ),
        ),
        _entry(
            specification=_semantic_spec(
                relation_id=RELATION_ID_FLOOD_ZONE_EXPOSURE,
                name=REL_FLOOD_ZONE_EXPOSURE,
                display_name="Flood-zone exposure",
                description=(
                    "Geometric exposure of an urban unit to a mapped "
                    "flood-prone zone."
                ),
                semantic_role=ROLE_EXPOSURE,
                source_node_types=(NODE_FLOOD_ZONE,),
                target_node_types=(
                    PREDICTION_UNIT_NODE_TYPES
                ),
                direction=RelationDirection.DIRECTED,
                evidence_type=RelationEvidenceType.OBSERVED,
                construction_mode=(
                    RelationConstructionMode.GEOMETRIC
                ),
                leakage_risk=RelationLeakageRisk.LOW,
                temporal_mode=RelationTemporalMode.STATIC,
                parent_relation_name=(
                    REL_HYDROLOGICAL_EXPOSURE
                ),
                required_edge_attributes=(
                    ATTRIBUTE_OVERLAP_FRACTION,
                ),
                optional_edge_attributes=(
                    ATTRIBUTE_DISTANCE_M,
                    ATTRIBUTE_EXPOSURE_INTENSITY,
                ),
                tags=(
                    "flood",
                    "flood_zone",
                    "exposure",
                ),
            ),
            allowed_pairs=_cartesian_pairs(
                (NODE_FLOOD_ZONE,),
                PREDICTION_UNIT_NODE_TYPES,
            ),
        ),
        _entry(
            specification=_semantic_spec(
                relation_id=(
                    RELATION_ID_LOW_ELEVATION_EXPOSURE
                ),
                name=REL_LOW_ELEVATION_EXPOSURE,
                display_name="Low-elevation exposure",
                description=(
                    "Hydrological exposure associated with low "
                    "elevation relative to nearby water features."
                ),
                semantic_role=ROLE_EXPOSURE,
                source_node_types=(NODE_WATER_BODY,),
                target_node_types=(
                    PREDICTION_UNIT_NODE_TYPES
                ),
                direction=RelationDirection.DIRECTED,
                evidence_type=RelationEvidenceType.DERIVED,
                construction_mode=(
                    RelationConstructionMode.GEOMETRIC
                ),
                leakage_risk=RelationLeakageRisk.LOW,
                temporal_mode=RelationTemporalMode.STATIC,
                parent_relation_name=(
                    REL_HYDROLOGICAL_EXPOSURE
                ),
                required_edge_attributes=(
                    ATTRIBUTE_DISTANCE_M,
                    ATTRIBUTE_ELEVATION_DIFFERENCE_M,
                ),
                optional_edge_attributes=(
                    ATTRIBUTE_EXPOSURE_INTENSITY,
                ),
                tags=(
                    "flood",
                    "elevation",
                    "exposure",
                ),
            ),
            allowed_pairs=_cartesian_pairs(
                (NODE_WATER_BODY,),
                PREDICTION_UNIT_NODE_TYPES,
            ),
        ),

        # ------------------------------------------------------------------
        # Heat exposure and protection
        # ------------------------------------------------------------------
        _entry(
            specification=_semantic_spec(
                relation_id=RELATION_ID_HEAT_EXPOSURE,
                name=REL_HEAT_EXPOSURE,
                display_name="Heat exposure",
                description=(
                    "Heat-island exposure of urban units based on "
                    "time-valid observed or derived heat surfaces."
                ),
                semantic_role=ROLE_EXPOSURE,
                source_node_types=(NODE_HEAT_ISLAND,),
                target_node_types=(
                    PREDICTION_UNIT_NODE_TYPES
                ),
                direction=RelationDirection.DIRECTED,
                evidence_type=RelationEvidenceType.HYBRID,
                construction_mode=(
                    RelationConstructionMode.EXTERNAL_STATIC
                ),
                leakage_risk=RelationLeakageRisk.LOW,
                temporal_mode=RelationTemporalMode.SNAPSHOT,
                optional_edge_attributes=(
                    ATTRIBUTE_OVERLAP_FRACTION,
                    ATTRIBUTE_EXPOSURE_INTENSITY,
                ),
                tags=(
                    "heat",
                    "heat_island",
                    "exposure",
                ),
            ),
            allowed_pairs=_cartesian_pairs(
                (NODE_HEAT_ISLAND,),
                PREDICTION_UNIT_NODE_TYPES,
            ),
        ),
        _entry(
            specification=_semantic_spec(
                relation_id=RELATION_ID_CANOPY_PROTECTION,
                name=REL_CANOPY_PROTECTION,
                display_name="Canopy protection",
                description=(
                    "Protective influence of a green-space source on an "
                    "urban unit. Total tract-level canopy fraction "
                    "remains a node feature rather than being repeated "
                    "on every incoming edge."
                ),
                semantic_role=ROLE_PROTECTION,
                source_node_types=(NODE_GREEN_SPACE,),
                target_node_types=(
                    PREDICTION_UNIT_NODE_TYPES
                ),
                direction=RelationDirection.DIRECTED,
                evidence_type=RelationEvidenceType.HYBRID,
                construction_mode=(
                    RelationConstructionMode.GEOMETRIC
                ),
                leakage_risk=RelationLeakageRisk.LOW,
                temporal_mode=RelationTemporalMode.SNAPSHOT,
                optional_edge_attributes=(
                    ATTRIBUTE_DISTANCE_M,
                    ATTRIBUTE_OVERLAP_FRACTION,
                ),
                tags=(
                    "heat",
                    "green_space",
                    "protection",
                ),
            ),
            allowed_pairs=_cartesian_pairs(
                (NODE_GREEN_SPACE,),
                PREDICTION_UNIT_NODE_TYPES,
            ),
        ),

        # ------------------------------------------------------------------
        # Access relations
        # ------------------------------------------------------------------
        _entry(
            specification=_semantic_spec(
                relation_id=RELATION_ID_SERVICE_ACCESS,
                name=REL_SERVICE_ACCESS,
                display_name="Service access",
                description=(
                    "Accessibility from an urban unit to a health, "
                    "shelter, cooling, or other service facility."
                ),
                semantic_role=ROLE_ACCESS,
                source_node_types=(
                    PREDICTION_UNIT_NODE_TYPES
                ),
                target_node_types=SERVICE_NODE_TYPES,
                direction=RelationDirection.DIRECTED,
                evidence_type=RelationEvidenceType.HYBRID,
                construction_mode=(
                    RelationConstructionMode.GEOMETRIC
                ),
                leakage_risk=RelationLeakageRisk.LOW,
                temporal_mode=RelationTemporalMode.STATIC,
                optional_edge_attributes=(
                    ATTRIBUTE_DISTANCE_M,
                    ATTRIBUTE_TRAVEL_TIME_MIN,
                    ATTRIBUTE_ACCESSIBILITY_SCORE,
                    ATTRIBUTE_CAPACITY_SCORE,
                ),
                tags=(
                    "service",
                    "access",
                    "abstract_family",
                ),
            ),
            allowed_pairs=service_access_pairs,
        ),
        _entry(
            specification=_semantic_spec(
                relation_id=RELATION_ID_COOLING_ACCESS,
                name=REL_COOLING_ACCESS,
                display_name="Cooling access",
                description=(
                    "Access from an urban unit to a cooling or "
                    "heat-relief service facility."
                ),
                semantic_role=ROLE_ACCESS,
                source_node_types=(
                    PREDICTION_UNIT_NODE_TYPES
                ),
                target_node_types=(NODE_SERVICE_FACILITY,),
                direction=RelationDirection.DIRECTED,
                evidence_type=RelationEvidenceType.HYBRID,
                construction_mode=(
                    RelationConstructionMode.GEOMETRIC
                ),
                leakage_risk=RelationLeakageRisk.LOW,
                temporal_mode=(
                    RelationTemporalMode.INTERVAL_VALID
                ),
                parent_relation_name=REL_SERVICE_ACCESS,
                required_edge_attributes=(
                    ATTRIBUTE_TRAVEL_TIME_MIN,
                ),
                optional_edge_attributes=(
                    ATTRIBUTE_DISTANCE_M,
                    ATTRIBUTE_ACCESSIBILITY_SCORE,
                    ATTRIBUTE_CAPACITY_SCORE,
                ),
                tags=(
                    "heat",
                    "cooling",
                    "service_access",
                ),
            ),
            allowed_pairs=_cartesian_pairs(
                PREDICTION_UNIT_NODE_TYPES,
                (NODE_SERVICE_FACILITY,),
            ),
        ),
        _entry(
            specification=_semantic_spec(
                relation_id=RELATION_ID_ROAD_ACCESS,
                name=REL_ROAD_ACCESS,
                display_name="Road access",
                description=(
                    "Transport-network connection supplied by a road "
                    "segment to urban units or service facilities. This "
                    "is a root transport relation, not a subtype of "
                    "service access."
                ),
                semantic_role=ROLE_ACCESS,
                source_node_types=(NODE_ROAD_SEGMENT,),
                target_node_types=_ordered_unique(
                    (
                        *PREDICTION_UNIT_NODE_TYPES,
                        *SERVICE_NODE_TYPES,
                    )
                ),
                direction=RelationDirection.DIRECTED,
                evidence_type=RelationEvidenceType.OBSERVED,
                construction_mode=(
                    RelationConstructionMode.GEOMETRIC
                ),
                leakage_risk=RelationLeakageRisk.LOW,
                temporal_mode=(
                    RelationTemporalMode.INTERVAL_VALID
                ),
                optional_edge_attributes=(
                    ATTRIBUTE_DISTANCE_M,
                    ATTRIBUTE_TRAVEL_TIME_MIN,
                    ATTRIBUTE_ACCESSIBILITY_SCORE,
                ),
                tags=(
                    "transport",
                    "road",
                    "access",
                ),
            ),
            allowed_pairs=_cartesian_pairs(
                (NODE_ROAD_SEGMENT,),
                _ordered_unique(
                    (
                        *PREDICTION_UNIT_NODE_TYPES,
                        *SERVICE_NODE_TYPES,
                    )
                ),
            ),
        ),

        # ------------------------------------------------------------------
        # Infrastructure dependency
        # ------------------------------------------------------------------
        _entry(
            specification=_semantic_spec(
                relation_id=(
                    RELATION_ID_INFRASTRUCTURE_DEPENDENCY
                ),
                name=REL_INFRASTRUCTURE_DEPENDENCY,
                display_name="Infrastructure dependency",
                description=(
                    "Broad functional dependency of urban units or "
                    "services on supporting infrastructure."
                ),
                semantic_role=ROLE_DEPENDENCY,
                source_node_types=(
                    NODE_CRITICAL_INFRASTRUCTURE,
                    NODE_DRAINAGE_ASSET,
                ),
                target_node_types=_ordered_unique(
                    (
                        *PREDICTION_UNIT_NODE_TYPES,
                        *SERVICE_NODE_TYPES,
                    )
                ),
                direction=RelationDirection.DIRECTED,
                evidence_type=RelationEvidenceType.HYBRID,
                construction_mode=(
                    RelationConstructionMode.EXTERNAL_STATIC
                ),
                leakage_risk=RelationLeakageRisk.LOW,
                temporal_mode=(
                    RelationTemporalMode.INTERVAL_VALID
                ),
                optional_edge_attributes=(
                    ATTRIBUTE_DEPENDENCY_STRENGTH,
                    ATTRIBUTE_CAPACITY_SCORE,
                    ATTRIBUTE_DISTANCE_M,
                ),
                tags=(
                    "infrastructure",
                    "dependency",
                    "abstract_family",
                ),
            ),
            allowed_pairs=_cartesian_pairs(
                (
                    NODE_CRITICAL_INFRASTRUCTURE,
                    NODE_DRAINAGE_ASSET,
                ),
                _ordered_unique(
                    (
                        *PREDICTION_UNIT_NODE_TYPES,
                        *SERVICE_NODE_TYPES,
                    )
                ),
            ),
        ),
        _entry(
            specification=_semantic_spec(
                relation_id=RELATION_ID_DRAINAGE_DEPENDENCY,
                name=REL_DRAINAGE_DEPENDENCY,
                display_name="Drainage dependency",
                description=(
                    "Functional dependence of an urban unit on drainage "
                    "or sewer infrastructure."
                ),
                semantic_role=ROLE_DEPENDENCY,
                source_node_types=(NODE_DRAINAGE_ASSET,),
                target_node_types=(
                    PREDICTION_UNIT_NODE_TYPES
                ),
                direction=RelationDirection.DIRECTED,
                evidence_type=RelationEvidenceType.HYBRID,
                construction_mode=(
                    RelationConstructionMode.EXTERNAL_STATIC
                ),
                leakage_risk=RelationLeakageRisk.LOW,
                temporal_mode=(
                    RelationTemporalMode.INTERVAL_VALID
                ),
                parent_relation_name=(
                    REL_INFRASTRUCTURE_DEPENDENCY
                ),
                optional_edge_attributes=(
                    ATTRIBUTE_DEPENDENCY_STRENGTH,
                    ATTRIBUTE_CAPACITY_SCORE,
                    ATTRIBUTE_DISTANCE_M,
                ),
                tags=(
                    "flood",
                    "drainage",
                    "sewer",
                    "dependency",
                ),
            ),
            allowed_pairs=_cartesian_pairs(
                (NODE_DRAINAGE_ASSET,),
                PREDICTION_UNIT_NODE_TYPES,
            ),
        ),
        _entry(
            specification=_semantic_spec(
                relation_id=(
                    RELATION_ID_CRITICAL_FACILITY_DEPENDENCY
                ),
                name=REL_CRITICAL_FACILITY_DEPENDENCY,
                display_name="Critical-facility dependency",
                description=(
                    "Functional dependence of hospitals and service "
                    "facilities on critical infrastructure."
                ),
                semantic_role=ROLE_DEPENDENCY,
                source_node_types=(
                    NODE_CRITICAL_INFRASTRUCTURE,
                ),
                target_node_types=SERVICE_NODE_TYPES,
                direction=RelationDirection.DIRECTED,
                evidence_type=RelationEvidenceType.HYBRID,
                construction_mode=(
                    RelationConstructionMode.EXTERNAL_STATIC
                ),
                leakage_risk=RelationLeakageRisk.LOW,
                temporal_mode=(
                    RelationTemporalMode.INTERVAL_VALID
                ),
                parent_relation_name=(
                    REL_INFRASTRUCTURE_DEPENDENCY
                ),
                optional_edge_attributes=(
                    ATTRIBUTE_DEPENDENCY_STRENGTH,
                    ATTRIBUTE_CAPACITY_SCORE,
                ),
                tags=(
                    "critical_facility",
                    "infrastructure",
                    "dependency",
                ),
            ),
            allowed_pairs=_cartesian_pairs(
                (NODE_CRITICAL_INFRASTRUCTURE,),
                SERVICE_NODE_TYPES,
            ),
        ),

        # ------------------------------------------------------------------
        # Similarity relations
        # ------------------------------------------------------------------
        _entry(
            specification=_semantic_spec(
                relation_id=RELATION_ID_REPORTING_SIMILARITY,
                name=REL_REPORTING_SIMILARITY,
                display_name="Reporting similarity",
                description=(
                    "Similarity between same-scale urban units based on "
                    "historical reporting behavior fitted exclusively "
                    "from the permitted training history."
                ),
                semantic_role=ROLE_SIMILARITY,
                source_node_types=(
                    PREDICTION_UNIT_NODE_TYPES
                ),
                target_node_types=(
                    PREDICTION_UNIT_NODE_TYPES
                ),
                direction=RelationDirection.UNDIRECTED,
                evidence_type=(
                    RelationEvidenceType.LEARNED_FROM_DATA
                ),
                construction_mode=(
                    RelationConstructionMode.TRAINING_FITTED
                ),
                leakage_risk=RelationLeakageRisk.HIGH,
                temporal_mode=RelationTemporalMode.SNAPSHOT,
                required_edge_attributes=(
                    ATTRIBUTE_SIMILARITY_SCORE,
                ),
                tags=(
                    "reporting",
                    "similarity",
                    "training_fitted",
                    "high_leakage_audit",
                ),
            ),
            allowed_pairs=same_prediction_unit_pairs,
        ),
        _entry(
            specification=_semantic_spec(
                relation_id=(
                    RELATION_ID_SOCIOECONOMIC_SIMILARITY
                ),
                name=REL_SOCIOECONOMIC_SIMILARITY,
                display_name="Socioeconomic similarity",
                description=(
                    "Similarity between same-scale urban units based on "
                    "socioeconomic features available by the prediction "
                    "origin."
                ),
                semantic_role=ROLE_SIMILARITY,
                source_node_types=(
                    PREDICTION_UNIT_NODE_TYPES
                ),
                target_node_types=(
                    PREDICTION_UNIT_NODE_TYPES
                ),
                direction=RelationDirection.UNDIRECTED,
                evidence_type=RelationEvidenceType.DERIVED,
                construction_mode=(
                    RelationConstructionMode.AS_OF_ORIGIN
                ),
                leakage_risk=RelationLeakageRisk.MODERATE,
                temporal_mode=RelationTemporalMode.SNAPSHOT,
                required_edge_attributes=(
                    ATTRIBUTE_SIMILARITY_SCORE,
                ),
                tags=(
                    "socioeconomic",
                    "similarity",
                    "as_of_origin",
                ),
            ),
            allowed_pairs=same_prediction_unit_pairs,
        ),

        # ------------------------------------------------------------------
        # Cross-scale relations and administrative subtype
        # ------------------------------------------------------------------
        _entry(
            specification=_semantic_spec(
                relation_id=RELATION_ID_CROSS_SCALE_PARENT,
                name=REL_CROSS_SCALE_PARENT,
                display_name="Cross-scale parent",
                description=(
                    "Broad directed relation from a lower-scale spatial "
                    "unit to its higher-scale parent."
                ),
                semantic_role=ROLE_CROSS_SCALE,
                source_node_types=(
                    NODE_CENSUS_TRACT,
                    NODE_MUNICIPALITY,
                ),
                target_node_types=(
                    NODE_MUNICIPALITY,
                    NODE_CENSUS_DIVISION,
                ),
                direction=RelationDirection.DIRECTED,
                evidence_type=RelationEvidenceType.OBSERVED,
                construction_mode=(
                    RelationConstructionMode.EXTERNAL_STATIC
                ),
                leakage_risk=RelationLeakageRisk.NONE,
                temporal_mode=RelationTemporalMode.STATIC,
                reverse_relation_name=REL_CROSS_SCALE_CHILD,
                optional_edge_attributes=(
                    ATTRIBUTE_MEMBERSHIP_WEIGHT,
                ),
                tags=(
                    "cross_scale",
                    "parent",
                    "hierarchy",
                    "abstract_family",
                ),
            ),
            allowed_pairs=hierarchical_parent_pairs,
        ),
        _entry(
            specification=_semantic_spec(
                relation_id=RELATION_ID_CROSS_SCALE_CHILD,
                name=REL_CROSS_SCALE_CHILD,
                display_name="Cross-scale child",
                description=(
                    "Directed reverse relation from a higher-scale "
                    "spatial unit to a lower-scale child."
                ),
                semantic_role=ROLE_CROSS_SCALE,
                source_node_types=(
                    NODE_MUNICIPALITY,
                    NODE_CENSUS_DIVISION,
                ),
                target_node_types=(
                    NODE_CENSUS_TRACT,
                    NODE_MUNICIPALITY,
                ),
                direction=RelationDirection.DIRECTED,
                evidence_type=RelationEvidenceType.OBSERVED,
                construction_mode=(
                    RelationConstructionMode.EXTERNAL_STATIC
                ),
                leakage_risk=RelationLeakageRisk.NONE,
                temporal_mode=RelationTemporalMode.STATIC,
                reverse_relation_name=REL_CROSS_SCALE_PARENT,
                optional_edge_attributes=(
                    ATTRIBUTE_MEMBERSHIP_WEIGHT,
                ),
                tags=(
                    "cross_scale",
                    "child",
                    "hierarchy",
                ),
            ),
            allowed_pairs=hierarchical_child_pairs,
        ),
        _entry(
            specification=_semantic_spec(
                relation_id=(
                    RELATION_ID_ADMINISTRATIVE_MEMBERSHIP
                ),
                name=REL_ADMINISTRATIVE_MEMBERSHIP,
                display_name="Administrative membership",
                description=(
                    "Administrative subtype of cross-scale parent "
                    "membership."
                ),
                semantic_role=ROLE_ADMINISTRATIVE,
                source_node_types=(
                    NODE_CENSUS_TRACT,
                    NODE_MUNICIPALITY,
                ),
                target_node_types=(
                    NODE_MUNICIPALITY,
                    NODE_CENSUS_DIVISION,
                ),
                direction=RelationDirection.DIRECTED,
                evidence_type=RelationEvidenceType.OBSERVED,
                construction_mode=(
                    RelationConstructionMode.EXTERNAL_STATIC
                ),
                leakage_risk=RelationLeakageRisk.NONE,
                temporal_mode=RelationTemporalMode.STATIC,
                parent_relation_name=REL_CROSS_SCALE_PARENT,
                optional_edge_attributes=(
                    ATTRIBUTE_MEMBERSHIP_WEIGHT,
                    ATTRIBUTE_ADMINISTRATIVE_LEVEL,
                ),
                tags=(
                    "administrative",
                    "membership",
                    "cross_scale",
                ),
            ),
            allowed_pairs=hierarchical_parent_pairs,
        ),
    )

    ordered = tuple(
        sorted(
            entries,
            key=lambda entry: entry.relation_id,
        )
    )

    _validate_default_registry_coverage(ordered)
    _validate_temporal_attribute_contracts(ordered)

    return ordered


def _validate_default_registry_coverage(
    entries: Sequence[RelationRegistryEntry],
) -> None:
    canonical_names = set(
        getattr(
            C,
            "CANONICAL_RELATION_NAMES",
            (),
        )
    )

    expected = (
        canonical_names
        - set(EDGE_REGISTRY_EXCLUDED_CANONICAL_NAMES)
    )
    observed = {
        entry.name
        for entry in entries
    }

    missing = sorted(expected - observed)
    unexpected = sorted(observed - expected)

    if missing or unexpected:
        raise RuntimeError(
            "The canonical edge registry and constants.py disagree. "
            f"Missing specifications: {missing}; "
            f"unexpected specifications: {unexpected}; "
            "explicitly excluded non-edge concepts: "
            f"{sorted(EDGE_REGISTRY_EXCLUDED_CANONICAL_NAMES)}."
        )


def _validate_temporal_attribute_contracts(
    entries: Sequence[RelationRegistryEntry],
) -> None:
    for entry in entries:
        specification = entry.specification

        if (
            TEMPORAL_FIELD_EDGE_LAG
            in specification.required_temporal_fields
            and not specification.requires_attribute(
                TEMPORAL_FIELD_EDGE_LAG
            )
        ):
            raise ValueError(
                f"Relation {specification.name!r} requires temporal "
                f"field {TEMPORAL_FIELD_EDGE_LAG!r}, but it is not "
                "declared as a required edge attribute."
            )


# =============================================================================
# Immutable canonical registry
# =============================================================================


@dataclass(slots=True, frozen=True)
class RelationRegistry:
    """Immutable semantic and operational relation registry."""

    entries: tuple[RelationRegistryEntry, ...]

    registry_name: str = DEFAULT_RELATION_REGISTRY_NAME
    description: str = (
        "Canonical V2 edge-relation ontology and current capability "
        "manifest."
    )
    registry_version: str = C.RELATION_REGISTRY_VERSION

    snapshot_schema_version: str = (
        RELATION_REGISTRY_SNAPSHOT_SCHEMA_VERSION
    )

    _by_id: Mapping[int, RelationRegistryEntry] = field(
        init=False,
        repr=False,
        compare=False,
    )
    _by_name: Mapping[str, RelationRegistryEntry] = field(
        init=False,
        repr=False,
        compare=False,
    )

    def __post_init__(self) -> None:
        # Structural validation must permit historical versions to be loaded.
        self.validate(
            require_current_registry_version=False,
            require_current_spec_schema_version=False,
        )

        object.__setattr__(
            self,
            "_by_id",
            MappingProxyType(
                {
                    entry.relation_id: entry
                    for entry in self.entries
                }
            ),
        )
        object.__setattr__(
            self,
            "_by_name",
            MappingProxyType(
                {
                    entry.name: entry
                    for entry in self.entries
                }
            ),
        )

    @property
    def specifications(self) -> tuple[RelationSpec, ...]:
        return tuple(
            entry.specification
            for entry in self.entries
        )

    @property
    def by_id(self) -> Mapping[int, RelationRegistryEntry]:
        return self._by_id

    @property
    def by_name(self) -> Mapping[str, RelationRegistryEntry]:
        return self._by_name

    @property
    def relation_ids(self) -> tuple[int, ...]:
        return tuple(
            entry.relation_id
            for entry in self.entries
        )

    @property
    def relation_names(self) -> tuple[str, ...]:
        return tuple(
            entry.name
            for entry in self.entries
        )

    @property
    def spec_schema_version(self) -> str:
        observed = {
            entry.specification.spec_schema_version
            for entry in self.entries
        }

        if len(observed) != 1:
            raise RuntimeError(
                "Registry entries do not share one relation-spec "
                "schema version."
            )

        return next(iter(observed))

    def validate(
        self,
        *,
        require_current_registry_version: bool,
        require_current_spec_schema_version: bool,
    ) -> None:
        _require_nonempty_string(
            "registry_name",
            self.registry_name,
        )
        _require_nonempty_string(
            "description",
            self.description,
        )
        _require_nonempty_string(
            "registry_version",
            self.registry_version,
        )
        _require_nonempty_string(
            "snapshot_schema_version",
            self.snapshot_schema_version,
        )

        if not self.entries:
            raise ValueError(
                "A relation registry cannot be empty."
            )

        for index, entry in enumerate(self.entries):
            if not isinstance(
                entry,
                RelationRegistryEntry,
            ):
                raise TypeError(
                    f"entries[{index}] must be a "
                    "RelationRegistryEntry."
                )

        validate_relation_spec_collection(
            self.specifications,
            require_current_registry_version=(
                require_current_registry_version
            ),
            require_current_spec_schema_version=(
                require_current_spec_schema_version
            ),
        )

        relation_ids = tuple(
            entry.relation_id
            for entry in self.entries
        )
        relation_names = tuple(
            entry.name
            for entry in self.entries
        )

        if relation_ids != tuple(sorted(relation_ids)):
            raise ValueError(
                "Registry entries must be ordered by stable "
                "relation_id."
            )

        duplicate_ids = sorted(
            value
            for value, count in Counter(relation_ids).items()
            if count > 1
        )
        duplicate_names = sorted(
            value
            for value, count in Counter(
                relation_names
            ).items()
            if count > 1
        )

        if duplicate_ids:
            raise ValueError(
                f"Duplicate stable relation IDs: {duplicate_ids}."
            )

        if duplicate_names:
            raise ValueError(
                f"Duplicate relation names: {duplicate_names}."
            )

        observed_versions = {
            entry.specification.registry_version
            for entry in self.entries
        }

        if observed_versions != {self.registry_version}:
            raise ValueError(
                "Registry version does not match specification "
                f"versions. Registry={self.registry_version!r}; "
                f"specifications={sorted(observed_versions)}."
            )

        self._validate_reverse_endpoint_contracts()

    def assert_current_compatibility(self) -> None:
        self.validate(
            require_current_registry_version=True,
            require_current_spec_schema_version=True,
        )

        if (
            self.snapshot_schema_version
            != RELATION_REGISTRY_SNAPSHOT_SCHEMA_VERSION
        ):
            raise ValueError(
                "Registry snapshot schema is not current. "
                f"Observed {self.snapshot_schema_version!r}, expected "
                f"{RELATION_REGISTRY_SNAPSHOT_SCHEMA_VERSION!r}."
            )

    def _validate_reverse_endpoint_contracts(self) -> None:
        local_by_name = {
            entry.name: entry
            for entry in self.entries
        }

        for entry in self.entries:
            reverse_name = (
                entry.specification.reverse_relation_name
            )

            if reverse_name is None:
                continue

            reverse = local_by_name[reverse_name]

            expected_reverse_pairs = {
                (target_type, source_type)
                for source_type, target_type
                in entry.endpoint_contract.allowed_pairs
            }
            observed_reverse_pairs = set(
                reverse.endpoint_contract.allowed_pairs
            )

            if (
                entry.endpoint_contract.explicit
                != reverse.endpoint_contract.explicit
            ):
                raise ValueError(
                    f"Reverse relation pair {entry.name!r} and "
                    f"{reverse.name!r} must both use explicit endpoint "
                    "contracts or both use Cartesian contracts."
                )

            if (
                entry.endpoint_contract.explicit
                and expected_reverse_pairs
                != observed_reverse_pairs
            ):
                raise ValueError(
                    f"Reverse relation {reverse.name!r} does not reverse "
                    f"the explicit endpoint pairs of {entry.name!r}."
                )

    def __len__(self) -> int:
        return len(self.entries)

    def __iter__(self) -> Iterator[RelationRegistryEntry]:
        return iter(self.entries)

    def __contains__(self, value: object) -> bool:
        if isinstance(value, int) and not isinstance(value, bool):
            return value in self.by_id

        if isinstance(value, str):
            return value in self.by_name

        return False

    def get_entry_by_id(
        self,
        relation_id: int,
    ) -> RelationRegistryEntry:
        try:
            return self.by_id[relation_id]
        except KeyError as exc:
            raise KeyError(
                f"Unknown stable relation ID {relation_id}."
            ) from exc

    def get_entry_by_name(
        self,
        relation_name: str,
    ) -> RelationRegistryEntry:
        try:
            return self.by_name[relation_name]
        except KeyError as exc:
            raise KeyError(
                f"Unknown relation name {relation_name!r}."
            ) from exc

    def get_spec_by_id(
        self,
        relation_id: int,
    ) -> RelationSpec:
        return self.get_entry_by_id(
            relation_id
        ).specification

    def get_spec_by_name(
        self,
        relation_name: str,
    ) -> RelationSpec:
        return self.get_entry_by_name(
            relation_name
        ).specification

    # ------------------------------------------------------------------
    # Capability filtering
    # ------------------------------------------------------------------

    def implemented(
        self,
    ) -> tuple[RelationRegistryEntry, ...]:
        return tuple(
            entry
            for entry in self.entries
            if entry.implemented
        )

    def controls(
        self,
    ) -> tuple[RelationRegistryEntry, ...]:
        return tuple(
            entry
            for entry in self.entries
            if entry.specification.is_control
        )

    def non_controls(
        self,
    ) -> tuple[RelationRegistryEntry, ...]:
        return tuple(
            entry
            for entry in self.entries
            if not entry.specification.is_control
        )

    def available_for_message_passing(
        self,
    ) -> tuple[RelationRegistryEntry, ...]:
        return tuple(
            entry
            for entry in self.entries
            if entry.available_for_message_passing
        )

    def available_for_training(
        self,
    ) -> tuple[RelationRegistryEntry, ...]:
        return tuple(
            entry
            for entry in self.entries
            if entry.available_for_training
        )

    def available_for_explanation(
        self,
        *,
        include_diagnostic: bool = False,
    ) -> tuple[RelationRegistryEntry, ...]:
        policies = {
            RelationExplanationPolicy.ALLOWED,
        }

        if include_diagnostic:
            policies.add(
                RelationExplanationPolicy.DIAGNOSTIC_ONLY
            )

        return tuple(
            entry
            for entry in self.entries
            if (
                entry.available_for_message_passing
                and entry.capability.explanation_policy
                in policies
            )
        )

    # ------------------------------------------------------------------
    # Hierarchy
    # ------------------------------------------------------------------

    def parent_of(
        self,
        relation_name: str,
    ) -> RelationRegistryEntry | None:
        entry = self.get_entry_by_name(relation_name)
        parent_name = (
            entry.specification.parent_relation_name
        )

        if parent_name is None:
            return None

        return self.get_entry_by_name(parent_name)

    def children_of(
        self,
        relation_name: str,
        *,
        recursive: bool = False,
    ) -> tuple[RelationRegistryEntry, ...]:
        self.get_entry_by_name(relation_name)

        direct = tuple(
            entry
            for entry in self.entries
            if (
                entry.specification.parent_relation_name
                == relation_name
            )
        )

        if not recursive:
            return direct

        descendants: list[RelationRegistryEntry] = []
        frontier = list(direct)
        visited: set[str] = set()

        while frontier:
            child = frontier.pop(0)

            if child.name in visited:
                raise RuntimeError(
                    "Cycle detected during hierarchy traversal."
                )

            visited.add(child.name)
            descendants.append(child)
            frontier.extend(
                self.children_of(
                    child.name,
                    recursive=False,
                )
            )

        return tuple(descendants)

    def ancestors_of(
        self,
        relation_name: str,
    ) -> tuple[RelationRegistryEntry, ...]:
        ancestors: list[RelationRegistryEntry] = []
        visited: set[str] = set()
        current = self.parent_of(relation_name)

        while current is not None:
            if current.name in visited:
                raise RuntimeError(
                    "Cycle detected during hierarchy traversal."
                )

            visited.add(current.name)
            ancestors.append(current)
            current = self.parent_of(current.name)

        return tuple(ancestors)

    def roots(
        self,
    ) -> tuple[RelationRegistryEntry, ...]:
        return tuple(
            entry
            for entry in self.entries
            if entry.specification.is_root_relation
        )

    def family_of(
        self,
        relation_name: str,
    ) -> tuple[RelationRegistryEntry, ...]:
        ancestors = self.ancestors_of(relation_name)
        root = (
            ancestors[-1]
            if ancestors
            else self.get_entry_by_name(relation_name)
        )

        return (
            root,
            *self.children_of(
                root.name,
                recursive=True,
            ),
        )

    # ------------------------------------------------------------------
    # Temporal/construction provenance requirements
    # ------------------------------------------------------------------

    def provenance_requirements_for(
        self,
        relation_name: str,
    ) -> frozenset[str]:
        entry = self.get_entry_by_name(relation_name)
        specification = entry.specification

        requirements = set(
            specification.required_temporal_fields
        )

        if specification.requires_training_fit:
            requirements.update(
                {
                    "training_split_fingerprint",
                    "training_fit_cutoff",
                }
            )

        if specification.requires_as_of_time:
            requirements.add(
                "construction_as_of_time"
            )

        return frozenset(requirements)

    # ------------------------------------------------------------------
    # Runtime compilation
    # ------------------------------------------------------------------

    def compile(
        self,
        relation_names: Sequence[str],
        *,
        require_implemented: bool = True,
        require_message_passing: bool = True,
        require_training: bool = False,
        allow_control_relations: bool = True,
        hierarchy_policy: HierarchyCompilationPolicy = (
            HierarchyCompilationPolicy.REJECT_OVERLAP
        ),
    ) -> CompiledRelationRegistry:
        if not relation_names:
            raise ValueError(
                "At least one relation name is required."
            )

        if not isinstance(
            hierarchy_policy,
            HierarchyCompilationPolicy,
        ):
            raise TypeError(
                "hierarchy_policy must be a "
                "HierarchyCompilationPolicy."
            )

        duplicates = sorted(
            value
            for value, count in Counter(
                relation_names
            ).items()
            if count > 1
        )

        if duplicates:
            raise ValueError(
                f"Duplicate requested relations: {duplicates}."
            )

        forbidden_non_edges = sorted(
            set(relation_names)
            & set(EDGE_REGISTRY_EXCLUDED_CANONICAL_NAMES)
        )

        if forbidden_non_edges:
            raise ValueError(
                "The following concepts are not edge relations: "
                f"{forbidden_non_edges}."
            )

        unknown = sorted(
            set(relation_names)
            - set(self.relation_names)
        )

        if unknown:
            raise ValueError(
                f"Unknown requested relations: {unknown}."
            )

        selected = tuple(
            self.get_entry_by_name(name)
            for name in relation_names
        )

        if not allow_control_relations:
            controls = sorted(
                entry.name
                for entry in selected
                if entry.specification.is_control
            )

            if controls:
                raise ValueError(
                    "Control relations are forbidden: "
                    f"{controls}."
                )

        if require_implemented:
            unavailable = sorted(
                entry.name
                for entry in selected
                if not entry.implemented
            )

            if unavailable:
                raise NotImplementedError(
                    "Requested relations are not implemented: "
                    f"{unavailable}."
                )

        if require_message_passing:
            unavailable = sorted(
                entry.name
                for entry in selected
                if not entry.available_for_message_passing
            )

            if unavailable:
                raise ValueError(
                    "Requested relations are unavailable for message "
                    f"passing: {unavailable}."
                )

        if require_training:
            unavailable = sorted(
                entry.name
                for entry in selected
                if not entry.available_for_training
            )

            if unavailable:
                raise ValueError(
                    "Requested relations are unavailable for training: "
                    f"{unavailable}."
                )

        selected = self._apply_hierarchy_policy(
            selected,
            hierarchy_policy,
        )

        ordered = tuple(
            sorted(
                selected,
                key=lambda entry: entry.relation_id,
            )
        )

        return CompiledRelationRegistry(
            entries=ordered,
            source_registry_name=self.registry_name,
            source_registry_version=self.registry_version,
            source_semantic_fingerprint=(
                self.semantic_fingerprint()
            ),
            source_operational_fingerprint=(
                self.operational_fingerprint()
            ),
            hierarchy_policy=hierarchy_policy,
        )

    def compile_for_training(
        self,
        relation_names: Sequence[str],
        *,
        allow_control_relations: bool,
        hierarchy_policy: HierarchyCompilationPolicy = (
            HierarchyCompilationPolicy.REJECT_OVERLAP
        ),
    ) -> CompiledRelationRegistry:
        return self.compile(
            relation_names,
            require_implemented=True,
            require_message_passing=True,
            require_training=True,
            allow_control_relations=allow_control_relations,
            hierarchy_policy=hierarchy_policy,
        )

    def compile_for_inference(
        self,
        relation_names: Sequence[str],
        *,
        allow_control_relations: bool = False,
        hierarchy_policy: HierarchyCompilationPolicy = (
            HierarchyCompilationPolicy.REJECT_OVERLAP
        ),
    ) -> CompiledRelationRegistry:
        return self.compile(
            relation_names,
            require_implemented=True,
            require_message_passing=True,
            require_training=False,
            allow_control_relations=allow_control_relations,
            hierarchy_policy=hierarchy_policy,
        )

    def compile_for_explanation(
        self,
        relation_names: Sequence[str],
        *,
        include_diagnostic_controls: bool = False,
        hierarchy_policy: HierarchyCompilationPolicy = (
            HierarchyCompilationPolicy.REJECT_OVERLAP
        ),
    ) -> CompiledRelationRegistry:
        compiled = self.compile_for_inference(
            relation_names,
            allow_control_relations=(
                include_diagnostic_controls
            ),
            hierarchy_policy=hierarchy_policy,
        )

        forbidden = sorted(
            entry.name
            for entry in compiled.entries
            if (
                entry.capability.explanation_policy
                == RelationExplanationPolicy.EXCLUDED
                or (
                    entry.capability.explanation_policy
                    == RelationExplanationPolicy.DIAGNOSTIC_ONLY
                    and not include_diagnostic_controls
                )
            )
        )

        if forbidden:
            raise ValueError(
                "Requested relations are unavailable for explanation: "
                f"{forbidden}."
            )

        return compiled

    def _apply_hierarchy_policy(
        self,
        selected: Sequence[RelationRegistryEntry],
        policy: HierarchyCompilationPolicy,
    ) -> tuple[RelationRegistryEntry, ...]:
        selected_names = {
            entry.name
            for entry in selected
        }

        overlaps: list[tuple[str, str]] = []

        for entry in selected:
            for ancestor in self.ancestors_of(entry.name):
                if ancestor.name in selected_names:
                    overlaps.append(
                        (ancestor.name, entry.name)
                    )

        if (
            policy
            == HierarchyCompilationPolicy.REJECT_OVERLAP
            and overlaps
        ):
            formatted = ", ".join(
                f"{parent} -> {child}"
                for parent, child in overlaps
            )

            raise ValueError(
                "The compiled registry selects ontology parents and "
                "descendants together, which may double-count a "
                f"mechanism: {formatted}."
            )

        if policy == HierarchyCompilationPolicy.LEAF_ONLY:
            ancestors_to_remove = {
                parent
                for parent, _ in overlaps
            }

            return tuple(
                entry
                for entry in selected
                if entry.name not in ancestors_to_remove
            )

        return tuple(selected)

    # ------------------------------------------------------------------
    # Fingerprints and persistence
    # ------------------------------------------------------------------

    def semantic_dict(self) -> dict[str, Any]:
        return {
            "registry_name": self.registry_name,
            "description": self.description,
            "registry_version": self.registry_version,
            "snapshot_schema_version": (
                self.snapshot_schema_version
            ),
            "spec_schema_version": self.spec_schema_version,
            "excluded_canonical_concepts": sorted(
                EDGE_REGISTRY_EXCLUDED_CANONICAL_NAMES
            ),
            "entries": [
                entry.semantic_dict()
                for entry in self.entries
            ],
        }

    def operational_dict(self) -> dict[str, Any]:
        return {
            **self.semantic_dict(),
            "entries": [
                entry.operational_dict()
                for entry in self.entries
            ],
        }

    def semantic_fingerprint(self) -> str:
        return _fingerprint(self.semantic_dict())

    def operational_fingerprint(self) -> str:
        return _fingerprint(self.operational_dict())

    def fingerprint(self) -> str:
        """
        Backward-compatible alias for the semantic fingerprint.
        """

        return self.semantic_fingerprint()

    def to_dict(self) -> dict[str, Any]:
        return {
            "registry_name": self.registry_name,
            "description": self.description,
            "registry_version": self.registry_version,
            "snapshot_schema_version": (
                self.snapshot_schema_version
            ),
            "spec_schema_version": self.spec_schema_version,
            "semantic_fingerprint": (
                self.semantic_fingerprint()
            ),
            "operational_fingerprint": (
                self.operational_fingerprint()
            ),
            "entries": [
                entry.to_dict()
                for entry in self.entries
            ],
        }

    @classmethod
    def from_dict(
        cls,
        payload: Mapping[str, Any],
        *,
        require_current_version: bool = False,
        verify_serialized_fingerprints: bool = True,
    ) -> RelationRegistry:
        mapping = dict(
            _require_mapping(
                "RelationRegistry",
                payload,
            )
        )

        allowed = {
            "registry_name",
            "description",
            "registry_version",
            "snapshot_schema_version",
            "spec_schema_version",
            "semantic_fingerprint",
            "operational_fingerprint",
            "entries",
        }
        unknown = sorted(set(mapping) - allowed)

        if unknown:
            raise ValueError(
                f"Unknown RelationRegistry fields: {unknown}."
            )

        serialized_spec_schema = mapping.pop(
            "spec_schema_version",
            None,
        )
        serialized_semantic_fingerprint = mapping.pop(
            "semantic_fingerprint",
            None,
        )
        serialized_operational_fingerprint = mapping.pop(
            "operational_fingerprint",
            None,
        )

        raw_entries = mapping.pop("entries", None)

        if not isinstance(raw_entries, list):
            raise TypeError(
                "RelationRegistry.entries must be a list."
            )

        if not raw_entries:
            raise ValueError(
                "RelationRegistry.entries cannot be empty."
            )

        entries = tuple(
            RelationRegistryEntry.from_dict(
                _require_mapping(
                    f"entries[{index}]",
                    value,
                )
            )
            for index, value in enumerate(raw_entries)
        )

        registry = cls(
            entries=entries,
            **mapping,
        )

        registry.validate(
            require_current_registry_version=(
                require_current_version
            ),
            require_current_spec_schema_version=(
                require_current_version
            ),
        )

        if (
            serialized_spec_schema is not None
            and serialized_spec_schema
            != registry.spec_schema_version
        ):
            raise ValueError(
                "Serialized spec_schema_version does not match the "
                "reconstructed entries."
            )

        if verify_serialized_fingerprints:
            if (
                serialized_semantic_fingerprint is not None
                and serialized_semantic_fingerprint
                != registry.semantic_fingerprint()
            ):
                raise ValueError(
                    "Serialized semantic registry fingerprint does not "
                    "match the reconstructed registry."
                )

            if (
                serialized_operational_fingerprint is not None
                and serialized_operational_fingerprint
                != registry.operational_fingerprint()
            ):
                raise ValueError(
                    "Serialized operational registry fingerprint does "
                    "not match the reconstructed registry."
                )

        return registry


# =============================================================================
# Dense compiled registry
# =============================================================================


@dataclass(slots=True, frozen=True)
class CompiledRelationRegistry:
    """
    Dense runtime relation-index mapping for one model run.

    The position of each entry in ``entries`` is its model-facing
    ``relation_index``.
    """

    entries: tuple[RelationRegistryEntry, ...]

    source_registry_name: str
    source_registry_version: str

    source_semantic_fingerprint: str
    source_operational_fingerprint: str

    hierarchy_policy: HierarchyCompilationPolicy

    compiled_schema_version: str = (
        COMPILED_RELATION_REGISTRY_SCHEMA_VERSION
    )

    _index_by_name: Mapping[str, int] = field(
        init=False,
        repr=False,
        compare=False,
    )
    _index_by_id: Mapping[int, int] = field(
        init=False,
        repr=False,
        compare=False,
    )

    def __post_init__(self) -> None:
        self.validate()

        object.__setattr__(
            self,
            "_index_by_name",
            MappingProxyType(
                {
                    entry.name: relation_index
                    for relation_index, entry
                    in enumerate(self.entries)
                }
            ),
        )
        object.__setattr__(
            self,
            "_index_by_id",
            MappingProxyType(
                {
                    entry.relation_id: relation_index
                    for relation_index, entry
                    in enumerate(self.entries)
                }
            ),
        )

    @property
    def specifications(self) -> tuple[RelationSpec, ...]:
        return tuple(
            entry.specification
            for entry in self.entries
        )

    @property
    def relation_names(self) -> tuple[str, ...]:
        return tuple(
            entry.name
            for entry in self.entries
        )

    @property
    def stable_relation_ids(self) -> tuple[int, ...]:
        return tuple(
            entry.relation_id
            for entry in self.entries
        )

    @property
    def relation_index_by_name(self) -> Mapping[str, int]:
        return self._index_by_name

    @property
    def relation_index_by_id(self) -> Mapping[int, int]:
        return self._index_by_id

    def validate(self) -> None:
        if not self.entries:
            raise ValueError(
                "A compiled relation registry cannot be empty."
            )

        for index, entry in enumerate(self.entries):
            if not isinstance(
                entry,
                RelationRegistryEntry,
            ):
                raise TypeError(
                    f"entries[{index}] must be a "
                    "RelationRegistryEntry."
                )

        relation_ids = tuple(
            entry.relation_id
            for entry in self.entries
        )
        relation_names = tuple(
            entry.name
            for entry in self.entries
        )

        if relation_ids != tuple(sorted(relation_ids)):
            raise ValueError(
                "Compiled entries must be ordered by stable "
                "relation_id."
            )

        if len(set(relation_ids)) != len(relation_ids):
            raise ValueError(
                "Compiled stable relation IDs must be unique."
            )

        if len(set(relation_names)) != len(relation_names):
            raise ValueError(
                "Compiled relation names must be unique."
            )

        _require_nonempty_string(
            "source_registry_name",
            self.source_registry_name,
        )
        _require_nonempty_string(
            "source_registry_version",
            self.source_registry_version,
        )
        _require_nonempty_string(
            "source_semantic_fingerprint",
            self.source_semantic_fingerprint,
        )
        _require_nonempty_string(
            "source_operational_fingerprint",
            self.source_operational_fingerprint,
        )
        _require_nonempty_string(
            "compiled_schema_version",
            self.compiled_schema_version,
        )

        if not isinstance(
            self.hierarchy_policy,
            HierarchyCompilationPolicy,
        ):
            raise TypeError(
                "hierarchy_policy must be a "
                "HierarchyCompilationPolicy."
            )

    def __len__(self) -> int:
        return len(self.entries)

    def __iter__(self) -> Iterator[RelationRegistryEntry]:
        return iter(self.entries)

    def index_for_name(
        self,
        relation_name: str,
    ) -> int:
        try:
            return self.relation_index_by_name[
                relation_name
            ]
        except KeyError as exc:
            raise KeyError(
                f"Relation {relation_name!r} is not present in the "
                "compiled registry."
            ) from exc

    def index_for_id(
        self,
        relation_id: int,
    ) -> int:
        try:
            return self.relation_index_by_id[relation_id]
        except KeyError as exc:
            raise KeyError(
                f"Stable relation ID {relation_id} is not present in "
                "the compiled registry."
            ) from exc

    def entry_for_index(
        self,
        relation_index: int,
    ) -> RelationRegistryEntry:
        if (
            isinstance(relation_index, bool)
            or not isinstance(relation_index, int)
        ):
            raise TypeError(
                "relation_index must be an integer."
            )

        if not 0 <= relation_index < len(self.entries):
            raise IndexError(
                f"relation_index {relation_index} is outside "
                f"[0, {len(self.entries) - 1}]."
            )

        return self.entries[relation_index]

    def spec_for_index(
        self,
        relation_index: int,
    ) -> RelationSpec:
        return self.entry_for_index(
            relation_index
        ).specification

    def encode_names(
        self,
        relation_names: Sequence[str],
    ) -> tuple[int, ...]:
        return tuple(
            self.index_for_name(name)
            for name in relation_names
        )

    def encode_stable_ids(
        self,
        relation_ids: Sequence[int],
    ) -> tuple[int, ...]:
        return tuple(
            self.index_for_id(relation_id)
            for relation_id in relation_ids
        )

    def decode_indices_to_names(
        self,
        relation_indices: Sequence[int],
    ) -> tuple[str, ...]:
        return tuple(
            self.entry_for_index(index).name
            for index in relation_indices
        )

    def decode_indices_to_stable_ids(
        self,
        relation_indices: Sequence[int],
    ) -> tuple[int, ...]:
        return tuple(
            self.entry_for_index(index).relation_id
            for index in relation_indices
        )

    def permits_endpoint_pair(
        self,
        relation_index: int,
        source_node_type: str,
        target_node_type: str,
    ) -> bool:
        return self.entry_for_index(
            relation_index
        ).permits_endpoint_pair(
            source_node_type,
            target_node_type,
        )

    def runtime_entries(
        self,
    ) -> tuple[Mapping[str, Any], ...]:
        return tuple(
            MappingProxyType(
                {
                    "relation_index": relation_index,
                    "relation_id": entry.relation_id,
                    "relation_name": entry.name,
                    "display_name": (
                        entry.specification.display_name
                    ),
                    "semantic_role": (
                        entry.specification.semantic_role
                    ),
                    "is_control": (
                        entry.specification.is_control
                    ),
                    "implementation_state": (
                        entry.capability.implementation_state
                    ),
                }
            )
            for relation_index, entry
            in enumerate(self.entries)
        )

    def assert_matches_source_registry(
        self,
        source_registry: RelationRegistry,
        *,
        require_operational_match: bool = True,
    ) -> None:
        if not isinstance(
            source_registry,
            RelationRegistry,
        ):
            raise TypeError(
                "source_registry must be a RelationRegistry."
            )

        if (
            source_registry.registry_name
            != self.source_registry_name
        ):
            raise ValueError(
                "Compiled registry source name does not match the "
                "provided canonical registry."
            )

        if (
            source_registry.registry_version
            != self.source_registry_version
        ):
            raise ValueError(
                "Compiled registry source version does not match the "
                "provided canonical registry."
            )

        if (
            source_registry.semantic_fingerprint()
            != self.source_semantic_fingerprint
        ):
            raise ValueError(
                "Compiled registry semantic source fingerprint does not "
                "match the provided canonical registry."
            )

        if (
            require_operational_match
            and source_registry.operational_fingerprint()
            != self.source_operational_fingerprint
        ):
            raise ValueError(
                "Compiled registry operational source fingerprint does "
                "not match the provided canonical registry."
            )

        missing_entries = sorted(
            entry.name
            for entry in self.entries
            if entry.name not in source_registry
        )

        if missing_entries:
            raise ValueError(
                "Compiled registry contains relations absent from its "
                f"claimed source registry: {missing_entries}."
            )

        for compiled_entry in self.entries:
            source_entry = (
                source_registry.get_entry_by_name(
                    compiled_entry.name
                )
            )

            if (
                compiled_entry.semantic_dict()
                != source_entry.semantic_dict()
            ):
                raise ValueError(
                    f"Compiled relation {compiled_entry.name!r} does "
                    "not match the semantic source registry entry."
                )

            if (
                require_operational_match
                and compiled_entry.capability
                != source_entry.capability
            ):
                raise ValueError(
                    f"Compiled relation {compiled_entry.name!r} does "
                    "not match the operational source capability."
                )

    def to_dict(self) -> dict[str, Any]:
        return {
            "compiled_schema_version": (
                self.compiled_schema_version
            ),
            "source_registry_name": (
                self.source_registry_name
            ),
            "source_registry_version": (
                self.source_registry_version
            ),
            "source_semantic_fingerprint": (
                self.source_semantic_fingerprint
            ),
            "source_operational_fingerprint": (
                self.source_operational_fingerprint
            ),
            "hierarchy_policy": self.hierarchy_policy.value,
            "runtime_relations": [
                {
                    "relation_index": relation_index,
                    "entry": entry.to_dict(),
                }
                for relation_index, entry
                in enumerate(self.entries)
            ],
        }

    def fingerprint(self) -> str:
        return _fingerprint(self.to_dict())

    @classmethod
    def from_dict(
        cls,
        payload: Mapping[str, Any],
    ) -> CompiledRelationRegistry:
        mapping = dict(
            _require_mapping(
                "CompiledRelationRegistry",
                payload,
            )
        )

        allowed = {
            "compiled_schema_version",
            "source_registry_name",
            "source_registry_version",
            "source_semantic_fingerprint",
            "source_operational_fingerprint",
            "hierarchy_policy",
            "runtime_relations",
        }
        unknown = sorted(set(mapping) - allowed)

        if unknown:
            raise ValueError(
                "Unknown CompiledRelationRegistry fields: "
                f"{unknown}."
            )

        raw_runtime_relations = mapping.pop(
            "runtime_relations",
            None,
        )

        if not isinstance(raw_runtime_relations, list):
            raise TypeError(
                "runtime_relations must be a list."
            )

        if not raw_runtime_relations:
            raise ValueError(
                "runtime_relations cannot be empty."
            )

        entries: list[RelationRegistryEntry] = []

        for expected_index, raw_value in enumerate(
            raw_runtime_relations
        ):
            runtime_entry = dict(
                _require_mapping(
                    f"runtime_relations[{expected_index}]",
                    raw_value,
                )
            )

            unknown_fields = sorted(
                set(runtime_entry)
                - {
                    "relation_index",
                    "entry",
                }
            )

            if unknown_fields:
                raise ValueError(
                    "Unknown runtime-relation fields: "
                    f"{unknown_fields}."
                )

            observed_index = runtime_entry.get(
                "relation_index"
            )

            if observed_index != expected_index:
                raise ValueError(
                    "Runtime relation indices must be contiguous and "
                    "ordered from zero. "
                    f"Expected {expected_index}, observed "
                    f"{observed_index!r}."
                )

            entries.append(
                RelationRegistryEntry.from_dict(
                    _require_mapping(
                        (
                            "runtime_relations"
                            f"[{expected_index}].entry"
                        ),
                        runtime_entry.get("entry"),
                    )
                )
            )

        mapping["entries"] = tuple(entries)
        mapping["hierarchy_policy"] = (
            HierarchyCompilationPolicy(
                mapping["hierarchy_policy"]
            )
        )

        return cls(**mapping)


# =============================================================================
# Default immutable registry
# =============================================================================


def build_default_relation_registry() -> RelationRegistry:
    registry = RelationRegistry(
        entries=build_default_relation_entries(),
    )

    registry.assert_current_compatibility()
    return registry


DEFAULT_RELATION_ENTRIES: Final[
    tuple[RelationRegistryEntry, ...]
] = build_default_relation_entries()

DEFAULT_RELATION_REGISTRY: Final[RelationRegistry] = (
    RelationRegistry(
        entries=DEFAULT_RELATION_ENTRIES,
    )
)

DEFAULT_RELATION_REGISTRY.assert_current_compatibility()


def get_default_relation_registry() -> RelationRegistry:
    """Return the immutable canonical V2 relation registry."""

    return DEFAULT_RELATION_REGISTRY


__all__ = (
    "ATTRIBUTE_ACCESSIBILITY_SCORE",
    "ATTRIBUTE_ADMINISTRATIVE_LEVEL",
    "ATTRIBUTE_CAPACITY_SCORE",
    "ATTRIBUTE_CONTROL_GENERATOR_ID",
    "ATTRIBUTE_DEPENDENCY_STRENGTH",
    "ATTRIBUTE_DISTANCE_M",
    "ATTRIBUTE_EDGE_LAG",
    "ATTRIBUTE_ELEVATION_DIFFERENCE_M",
    "ATTRIBUTE_EXPOSURE_INTENSITY",
    "ATTRIBUTE_KNN_RANK",
    "ATTRIBUTE_MEMBERSHIP_WEIGHT",
    "ATTRIBUTE_OVERLAP_FRACTION",
    "ATTRIBUTE_SHARED_BOUNDARY_LENGTH_M",
    "ATTRIBUTE_SIMILARITY_SCORE",
    "ATTRIBUTE_TRAVEL_TIME_MIN",
    "COMPILED_RELATION_REGISTRY_SCHEMA_VERSION",
    "CompiledRelationRegistry",
    "DEFAULT_RELATION_ENTRIES",
    "DEFAULT_RELATION_REGISTRY",
    "DEFAULT_RELATION_REGISTRY_NAME",
    "EDGE_REGISTRY_EXCLUDED_CANONICAL_NAMES",
    "ENDPOINT_CONTRACT_SCHEMA_VERSION",
    "HierarchyCompilationPolicy",
    "RELATION_CAPABILITY_SCHEMA_VERSION",
    "RELATION_ID_ADMINISTRATIVE_MEMBERSHIP",
    "RELATION_ID_CANOPY_PROTECTION",
    "RELATION_ID_CENTROID_KNN",
    "RELATION_ID_COOLING_ACCESS",
    "RELATION_ID_CRITICAL_FACILITY_DEPENDENCY",
    "RELATION_ID_CROSS_SCALE_CHILD",
    "RELATION_ID_CROSS_SCALE_PARENT",
    "RELATION_ID_DRAINAGE_DEPENDENCY",
    "RELATION_ID_FLOOD_ZONE_EXPOSURE",
    "RELATION_ID_HEAT_EXPOSURE",
    "RELATION_ID_HISTORICAL_EVENT_PROPAGATION",
    "RELATION_ID_HYDROLOGICAL_EXPOSURE",
    "RELATION_ID_INFRASTRUCTURE_DEPENDENCY",
    "RELATION_ID_LOW_ELEVATION_EXPOSURE",
    "RELATION_ID_RANDOM_PLACEBO",
    "RELATION_ID_REPORTING_SIMILARITY",
    "RELATION_ID_ROAD_ACCESS",
    "RELATION_ID_SERVICE_ACCESS",
    "RELATION_ID_SOCIOECONOMIC_SIMILARITY",
    "RELATION_ID_SPATIAL_ADJACENCY",
    "RELATION_ID_TEMPORAL_MEMORY",
    "RELATION_REGISTRY_SNAPSHOT_SCHEMA_VERSION",
    "RelationCapability",
    "RelationEndpointContract",
    "RelationRegistry",
    "RelationRegistryEntry",
    "build_default_relation_entries",
    "build_default_relation_registry",
    "get_default_relation_registry",
)