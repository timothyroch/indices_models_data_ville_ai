"""
Semantic relation types for the V2 hazard-conditioned functional UGNN.

Target path:
    urban_resilience_models/
        v2_hazard_conditioned_functional_ugnn/
            relations/
                relation_types.py

This module defines immutable semantic building blocks for the relation
registry.

It owns:

- stable relation identities;
- relation hierarchy;
- endpoint constraints;
- directionality;
- temporal behavior;
- construction and leakage semantics;
- explanation policy;
- edge-attribute contracts;
- deterministic serialization and reconstruction;
- collection-level ontology validation.

It does not own:

- dense runtime relation indices;
- concrete registry entries;
- graph construction;
- edge-table or tensor validation;
- hazard-relation priors;
- message-passing behavior;
- filesystem persistence.

Identity model
--------------
``relation_id`` is a stable registry identity. It is not required to be
contiguous and must never be used directly as the dense relation dimension of
a model tensor.

The concrete registry will compile selected relations into:

    relation_index in [0, R - 1]

Model-facing tensors such as ``edge_relation_type`` and relation-gate outputs
must use that dense runtime index. Artifacts must preserve the mapping:

    relation_index -> relation_id -> canonical relation name

Python requirement
------------------
This module uses ``StrEnum`` and ``typing.Self`` and therefore requires
Python 3.11 or newer.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import (
    dataclass,
    fields,
    replace as dataclass_replace,
)
from enum import StrEnum
import math
from types import MappingProxyType
from typing import Any, Final, Mapping, Self, Sequence

from ..constants import (
    CANONICAL_IMPLEMENTATION_STATES,
    CANONICAL_NODE_TYPE_NAMES,
    CANONICAL_RELATION_NAMES,
    CANONICAL_RELATION_ROLES,
    CONTROL_RELATION_NAMES,
    IMPLEMENTATION_STATE_DEPRECATED,
    IMPLEMENTATION_STATE_IMPLEMENTED,
    IMPLEMENTATION_STATE_TARGET,
    RELATION_REGISTRY_VERSION,
    RELATION_ROLE_ACCESS,
    RELATION_ROLE_DEPENDENCY,
    RELATION_ROLE_EXPOSURE,
    RELATION_ROLE_PROTECTION,
)


# =============================================================================
# Stable local vocabulary
# =============================================================================


ANY_NODE_TYPE: Final[str] = "*"

RELATION_SPEC_SCHEMA_VERSION: Final[str] = "0.2"

# This is an experiment topology mode, not an edge relation. It remains here
# as an explicit guard until constants.py moves it out of relation vocabulary.
TOPOLOGY_ONLY_NO_EDGE_NAME: Final[str] = "identity_no_edge"


FUNCTIONAL_RELATION_ROLES: Final[frozenset[str]] = frozenset(
    {
        RELATION_ROLE_EXPOSURE,
        RELATION_ROLE_PROTECTION,
        RELATION_ROLE_ACCESS,
        RELATION_ROLE_DEPENDENCY,
    }
)


TEMPORAL_FIELD_EDGE_OBSERVATION_TIME: Final[str] = (
    "edge_observation_time"
)
TEMPORAL_FIELD_EDGE_VALID_FROM: Final[str] = "edge_valid_from"
TEMPORAL_FIELD_EDGE_VALID_TO: Final[str] = "edge_valid_to"
TEMPORAL_FIELD_EDGE_LAG: Final[str] = "edge_lag"


# =============================================================================
# Enumerations
# =============================================================================


class RelationDirection(StrEnum):
    """
    Semantic direction of a relation.

    An undirected relation may still be stored as reciprocal directed arcs by
    the graph loader. Both arcs remain members of the same relation.
    """

    DIRECTED = "directed"
    UNDIRECTED = "undirected"


class RelationEvidenceType(StrEnum):
    """How the existence or meaning of a relation is justified."""

    OBSERVED = "observed"
    DERIVED = "derived"
    HYBRID = "hybrid"
    EXPERT_DEFINED = "expert_defined"
    LEARNED_FROM_DATA = "learned_from_data"
    SYNTHETIC_CONTROL = "synthetic_control"


class RelationConstructionMode(StrEnum):
    """
    How graph edges for a relation are constructed.

    This is distinct from evidence type. For example, a geometrically derived
    edge may still be supported by observed geographic data.
    """

    EXTERNAL_STATIC = "external_static"
    GEOMETRIC = "geometric"
    TRAINING_FITTED = "training_fitted"
    AS_OF_ORIGIN = "as_of_origin"
    SYNTHETIC_CONTROL = "synthetic_control"


class RelationLeakageRisk(StrEnum):
    """Declared leakage risk requiring different validation intensity."""

    NONE = "none"
    LOW = "low"
    MODERATE = "moderate"
    HIGH = "high"


class RelationTemporalMode(StrEnum):
    """How relation validity changes through time."""

    STATIC = "static"
    SNAPSHOT = "snapshot"
    INTERVAL_VALID = "interval_valid"
    LAGGED = "lagged"


class RelationExplanationPolicy(StrEnum):
    """Whether a relation may appear in scientific explanations."""

    ALLOWED = "allowed"
    DIAGNOSTIC_ONLY = "diagnostic_only"
    EXCLUDED = "excluded"


class EdgeAttributeKind(StrEnum):
    """Logical type of an edge attribute."""

    FLOAT = "float"
    INTEGER = "integer"
    BOOLEAN = "boolean"
    CATEGORICAL = "categorical"
    IDENTIFIER = "identifier"
    TIMESTAMP = "timestamp"


class MissingValuePolicy(StrEnum):
    """
    Missingness contract across raw and model-facing representations.

    ``NULLABLE_RAW_MASK_REQUIRED`` means:

    - raw edge tables may contain nulls;
    - model-facing values must be imputed;
    - a separate missingness mask must be supplied;
    - uncontrolled NaN values are not accepted as the mask.
    """

    FORBIDDEN = "forbidden"
    NULLABLE_RAW_MASK_REQUIRED = "nullable_raw_mask_required"


# =============================================================================
# Validation helpers
# =============================================================================


def _require_nonempty_string(name: str, value: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string.")


def _require_nonnegative_int(name: str, value: int) -> None:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer.")

    if value < 0:
        raise ValueError(f"{name} must be nonnegative.")


def _require_finite_number(
    name: str,
    value: int | float,
) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be numeric.")

    converted = float(value)

    if not math.isfinite(converted):
        raise ValueError(f"{name} must be finite.")

    return converted


def _require_unique_strings(
    name: str,
    values: Sequence[str],
) -> None:
    for index, value in enumerate(values):
        _require_nonempty_string(f"{name}[{index}]", value)

    duplicates = sorted(
        value
        for value, count in Counter(values).items()
        if count > 1
    )

    if duplicates:
        raise ValueError(
            f"{name} contains duplicate values: {duplicates}."
        )


def _reject_unknown_fields(
    object_type: type[Any],
    payload: Mapping[str, Any],
) -> None:
    allowed = {
        definition.name
        for definition in fields(object_type)
    }
    unknown = sorted(set(payload) - allowed)

    if unknown:
        raise ValueError(
            f"Unknown fields for {object_type.__name__}: {unknown}."
        )


def _require_mapping(
    name: str,
    payload: Any,
) -> Mapping[str, Any]:
    if not isinstance(payload, Mapping):
        raise TypeError(f"{name} must be a mapping.")

    return payload


def _as_tuple(
    name: str,
    value: Any,
) -> tuple[Any, ...]:
    if isinstance(value, tuple):
        return value

    if isinstance(value, list):
        return tuple(value)

    raise TypeError(f"{name} must be a list or tuple.")


def _validate_node_type_constraint(
    name: str,
    node_types: Sequence[str],
    *,
    allow_any_node_type: bool,
) -> None:
    if not node_types:
        raise ValueError(
            f"{name} must contain at least one node type."
        )

    _require_unique_strings(name, node_types)

    contains_wildcard = ANY_NODE_TYPE in node_types

    if contains_wildcard and len(node_types) != 1:
        raise ValueError(
            f"{name} cannot combine {ANY_NODE_TYPE!r} with specific "
            "node types."
        )

    if contains_wildcard and not allow_any_node_type:
        raise ValueError(
            f"{name} uses the wildcard node type, but "
            "allow_any_node_type=False."
        )

    unknown = sorted(
        set(node_types)
        - set(CANONICAL_NODE_TYPE_NAMES)
        - {ANY_NODE_TYPE}
    )

    if unknown:
        raise ValueError(
            f"{name} contains unknown node types: {unknown}."
        )


def _node_type_sets_equal(
    first: Sequence[str],
    second: Sequence[str],
) -> bool:
    return frozenset(first) == frozenset(second)


# =============================================================================
# Edge-attribute specification
# =============================================================================


@dataclass(slots=True, frozen=True)
class EdgeAttributeSpec:
    """
    Immutable semantic contract for one edge attribute.

    Requiredness is determined by whether the attribute appears in a
    relation's required or optional attribute collection.

    Numeric limits apply to the raw semantic values before model scaling.
    Missing values are never represented by uncontrolled NaN values in
    model-facing tensors.
    """

    name: str
    description: str
    kind: EdgeAttributeKind

    unit: str | None = None

    missing_value_policy: MissingValuePolicy = (
        MissingValuePolicy.FORBIDDEN
    )
    model_tensor_requires_finite: bool = True

    minimum: float | None = None
    maximum: float | None = None

    categorical_closed_vocabulary: bool = True
    categorical_values: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        self.validate()

    @property
    def numeric(self) -> bool:
        return self.kind in {
            EdgeAttributeKind.FLOAT,
            EdgeAttributeKind.INTEGER,
        }

    @property
    def raw_values_may_be_missing(self) -> bool:
        return (
            self.missing_value_policy
            == MissingValuePolicy.NULLABLE_RAW_MASK_REQUIRED
        )

    @property
    def requires_missingness_mask(self) -> bool:
        return self.raw_values_may_be_missing

    def validate(self) -> None:
        _require_nonempty_string(
            "edge attribute name",
            self.name,
        )
        _require_nonempty_string(
            f"edge attribute {self.name!r} description",
            self.description,
        )

        if not isinstance(self.kind, EdgeAttributeKind):
            raise TypeError(
                "kind must be an EdgeAttributeKind instance."
            )

        if not isinstance(
            self.missing_value_policy,
            MissingValuePolicy,
        ):
            raise TypeError(
                "missing_value_policy must be a "
                "MissingValuePolicy instance."
            )

        if self.unit is not None:
            _require_nonempty_string(
                f"edge attribute {self.name!r} unit",
                self.unit,
            )

        if self.minimum is not None:
            _require_finite_number(
                f"edge attribute {self.name!r} minimum",
                self.minimum,
            )

        if self.maximum is not None:
            _require_finite_number(
                f"edge attribute {self.name!r} maximum",
                self.maximum,
            )

        if (
            self.minimum is not None
            and self.maximum is not None
            and self.minimum > self.maximum
        ):
            raise ValueError(
                f"Edge attribute {self.name!r} minimum cannot exceed "
                "its maximum."
            )

        if not self.numeric and (
            self.minimum is not None
            or self.maximum is not None
        ):
            raise ValueError(
                f"Non-numeric edge attribute {self.name!r} cannot "
                "define numeric limits."
            )

        if self.kind == EdgeAttributeKind.INTEGER:
            for bound_name, bound in (
                ("minimum", self.minimum),
                ("maximum", self.maximum),
            ):
                if bound is not None and not float(bound).is_integer():
                    raise ValueError(
                        f"Integer edge attribute {self.name!r} has a "
                        f"non-integer {bound_name}."
                    )

        if self.kind == EdgeAttributeKind.CATEGORICAL:
            if (
                self.categorical_closed_vocabulary
                and not self.categorical_values
            ):
                raise ValueError(
                    f"Closed categorical edge attribute {self.name!r} "
                    "must declare categorical_values."
                )

            if self.categorical_values:
                _require_unique_strings(
                    (
                        f"edge attribute {self.name!r} "
                        "categorical_values"
                    ),
                    self.categorical_values,
                )

        elif self.categorical_values:
            raise ValueError(
                "Only categorical attributes may declare "
                f"categorical_values; {self.name!r} has kind "
                f"{self.kind.value!r}."
            )

        elif not self.categorical_closed_vocabulary:
            raise ValueError(
                "categorical_closed_vocabulary is only meaningful for "
                "categorical attributes."
            )

        if self.kind in {
            EdgeAttributeKind.BOOLEAN,
            EdgeAttributeKind.IDENTIFIER,
            EdgeAttributeKind.TIMESTAMP,
        } and self.unit is not None:
            raise ValueError(
                f"Edge attribute {self.name!r} with kind "
                f"{self.kind.value!r} cannot define a unit."
            )

        if (
            self.raw_values_may_be_missing
            and not self.model_tensor_requires_finite
        ):
            raise ValueError(
                f"Nullable edge attribute {self.name!r} must require "
                "finite model-facing values after imputation."
            )

    def replace(self, **changes: Any) -> Self:
        return dataclass_replace(self, **changes)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "kind": self.kind.value,
            "unit": self.unit,
            "missing_value_policy": (
                self.missing_value_policy.value
            ),
            "model_tensor_requires_finite": (
                self.model_tensor_requires_finite
            ),
            "minimum": self.minimum,
            "maximum": self.maximum,
            "categorical_closed_vocabulary": (
                self.categorical_closed_vocabulary
            ),
            "categorical_values": list(
                self.categorical_values
            ),
        }

    @classmethod
    def from_dict(
        cls,
        payload: Mapping[str, Any],
    ) -> EdgeAttributeSpec:
        mapping = dict(
            _require_mapping("EdgeAttributeSpec", payload)
        )
        _reject_unknown_fields(cls, mapping)

        if "kind" in mapping:
            mapping["kind"] = EdgeAttributeKind(mapping["kind"])

        if "missing_value_policy" in mapping:
            mapping["missing_value_policy"] = MissingValuePolicy(
                mapping["missing_value_policy"]
            )

        if "categorical_values" in mapping:
            mapping["categorical_values"] = _as_tuple(
                "categorical_values",
                mapping["categorical_values"],
            )

        return cls(**mapping)


# =============================================================================
# Relation specification
# =============================================================================


@dataclass(slots=True, frozen=True)
class RelationSpec:
    """
    Immutable semantic definition of one relation.

    ``relation_id`` is a stable registry identity, not a tensor index.

    ``parent_relation_name`` defines an ontology hierarchy. The concrete
    registry is responsible for determining whether parent and child
    relations may be active simultaneously in one compiled runtime registry.

    Source and target node constraints may use:

        ("census_tract",)
        ("hospital", "service_facility")
        ("*",)

    Wildcards require ``allow_any_node_type=True`` and should be used only
    for relations whose semantics genuinely permit arbitrary endpoints.
    """

    relation_id: int
    name: str
    display_name: str
    description: str

    semantic_role: str

    source_node_types: tuple[str, ...]
    target_node_types: tuple[str, ...]

    direction: RelationDirection
    evidence_type: RelationEvidenceType
    construction_mode: RelationConstructionMode
    leakage_risk: RelationLeakageRisk
    temporal_mode: RelationTemporalMode

    parent_relation_name: str | None = None
    reverse_relation_name: str | None = None

    implementation_state: str = IMPLEMENTATION_STATE_TARGET

    is_control: bool = False
    allow_any_node_type: bool = False
    allows_self_loops: bool = False

    message_passing_allowed: bool = False
    training_allowed: bool = False

    explanation_policy: RelationExplanationPolicy = (
        RelationExplanationPolicy.EXCLUDED
    )

    required_edge_attributes: tuple[EdgeAttributeSpec, ...] = ()
    optional_edge_attributes: tuple[EdgeAttributeSpec, ...] = ()

    tags: tuple[str, ...] = ()

    registry_version: str = RELATION_REGISTRY_VERSION
    spec_schema_version: str = RELATION_SPEC_SCHEMA_VERSION

    def __post_init__(self) -> None:
        self.validate()

    # ------------------------------------------------------------------
    # Identity and hierarchy
    # ------------------------------------------------------------------

    @property
    def is_root_relation(self) -> bool:
        return self.parent_relation_name is None

    @property
    def required_attribute_names(self) -> tuple[str, ...]:
        return tuple(
            attribute.name
            for attribute in self.required_edge_attributes
        )

    @property
    def optional_attribute_names(self) -> tuple[str, ...]:
        return tuple(
            attribute.name
            for attribute in self.optional_edge_attributes
        )

    @property
    def allowed_attribute_names(self) -> frozenset[str]:
        return frozenset(
            self.required_attribute_names
            + self.optional_attribute_names
        )

    @property
    def is_functional(self) -> bool:
        return self.semantic_role in FUNCTIONAL_RELATION_ROLES

    @property
    def is_directed(self) -> bool:
        return self.direction == RelationDirection.DIRECTED

    @property
    def is_undirected(self) -> bool:
        return self.direction == RelationDirection.UNDIRECTED

    # ------------------------------------------------------------------
    # Implementation and availability
    # ------------------------------------------------------------------

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

    @property
    def available_for_explanation(self) -> bool:
        return (
            self.available_for_message_passing
            and self.explanation_policy
            != RelationExplanationPolicy.EXCLUDED
        )

    # ------------------------------------------------------------------
    # Construction and temporal requirements
    # ------------------------------------------------------------------

    @property
    def requires_training_fit(self) -> bool:
        return (
            self.construction_mode
            == RelationConstructionMode.TRAINING_FITTED
        )

    @property
    def requires_as_of_time(self) -> bool:
        return self.construction_mode in {
            RelationConstructionMode.TRAINING_FITTED,
            RelationConstructionMode.AS_OF_ORIGIN,
        }

    @property
    def required_temporal_fields(self) -> frozenset[str]:
        if self.temporal_mode == RelationTemporalMode.STATIC:
            return frozenset()

        if self.temporal_mode == RelationTemporalMode.SNAPSHOT:
            return frozenset(
                {TEMPORAL_FIELD_EDGE_OBSERVATION_TIME}
            )

        if self.temporal_mode == RelationTemporalMode.INTERVAL_VALID:
            return frozenset(
                {
                    TEMPORAL_FIELD_EDGE_VALID_FROM,
                    TEMPORAL_FIELD_EDGE_VALID_TO,
                }
            )

        if self.temporal_mode == RelationTemporalMode.LAGGED:
            return frozenset(
                {
                    TEMPORAL_FIELD_EDGE_OBSERVATION_TIME,
                    TEMPORAL_FIELD_EDGE_LAG,
                }
            )

        raise RuntimeError(
            f"Unhandled temporal mode {self.temporal_mode!r}."
        )

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def validate(self) -> None:
        _require_nonnegative_int(
            "relation_id",
            self.relation_id,
        )
        _require_nonempty_string(
            "relation name",
            self.name,
        )
        _require_nonempty_string(
            f"relation {self.name!r} display_name",
            self.display_name,
        )
        _require_nonempty_string(
            f"relation {self.name!r} description",
            self.description,
        )
        _require_nonempty_string(
            f"relation {self.name!r} registry_version",
            self.registry_version,
        )
        _require_nonempty_string(
            f"relation {self.name!r} spec_schema_version",
            self.spec_schema_version,
        )

        if self.name == TOPOLOGY_ONLY_NO_EDGE_NAME:
            raise ValueError(
                "'identity_no_edge' is a topology mode describing the "
                "absence of edges, not a RelationSpec. Move it to the "
                "experiment topology vocabulary."
            )

        if self.name not in CANONICAL_RELATION_NAMES:
            raise ValueError(
                f"Unknown canonical relation name {self.name!r}. "
                "Add it to constants.py before defining a RelationSpec."
            )

        if self.semantic_role not in CANONICAL_RELATION_ROLES:
            raise ValueError(
                f"Unknown semantic role {self.semantic_role!r} for "
                f"relation {self.name!r}."
            )

        if (
            self.implementation_state
            not in CANONICAL_IMPLEMENTATION_STATES
        ):
            raise ValueError(
                f"Unknown implementation state "
                f"{self.implementation_state!r}."
            )

        enum_fields = (
            ("direction", self.direction, RelationDirection),
            (
                "evidence_type",
                self.evidence_type,
                RelationEvidenceType,
            ),
            (
                "construction_mode",
                self.construction_mode,
                RelationConstructionMode,
            ),
            (
                "leakage_risk",
                self.leakage_risk,
                RelationLeakageRisk,
            ),
            (
                "temporal_mode",
                self.temporal_mode,
                RelationTemporalMode,
            ),
            (
                "explanation_policy",
                self.explanation_policy,
                RelationExplanationPolicy,
            ),
        )

        for field_name, value, expected_type in enum_fields:
            if not isinstance(value, expected_type):
                raise TypeError(
                    f"{field_name} must be a "
                    f"{expected_type.__name__} instance."
                )

        _validate_node_type_constraint(
            "source_node_types",
            self.source_node_types,
            allow_any_node_type=self.allow_any_node_type,
        )
        _validate_node_type_constraint(
            "target_node_types",
            self.target_node_types,
            allow_any_node_type=self.allow_any_node_type,
        )

        self._validate_hierarchy_declaration()
        self._validate_reverse_declaration()
        self._validate_control_semantics()
        self._validate_construction_semantics()
        self._validate_temporal_semantics()
        self._validate_attribute_contracts()
        self._validate_availability_policy()
        self._validate_self_loop_policy()

        _require_unique_strings(
            "relation tags",
            self.tags,
        )

    def _validate_hierarchy_declaration(self) -> None:
        if self.parent_relation_name is None:
            return

        _require_nonempty_string(
            "parent_relation_name",
            self.parent_relation_name,
        )

        if self.parent_relation_name == self.name:
            raise ValueError(
                f"Relation {self.name!r} cannot be its own parent."
            )

        if (
            self.parent_relation_name
            not in CANONICAL_RELATION_NAMES
        ):
            raise ValueError(
                f"Unknown parent relation "
                f"{self.parent_relation_name!r}."
            )

        if (
            self.parent_relation_name
            == TOPOLOGY_ONLY_NO_EDGE_NAME
        ):
            raise ValueError(
                "'identity_no_edge' cannot be a relation parent."
            )

    def _validate_reverse_declaration(self) -> None:
        if self.reverse_relation_name is None:
            return

        _require_nonempty_string(
            "reverse_relation_name",
            self.reverse_relation_name,
        )

        if not self.is_directed:
            raise ValueError(
                "reverse_relation_name is only valid for directed "
                "relations."
            )

        if self.reverse_relation_name == self.name:
            raise ValueError(
                "A relation cannot identify itself as its reverse."
            )

        if (
            self.reverse_relation_name
            not in CANONICAL_RELATION_NAMES
        ):
            raise ValueError(
                f"Unknown reverse relation "
                f"{self.reverse_relation_name!r}."
            )

        if (
            self.reverse_relation_name
            == TOPOLOGY_ONLY_NO_EDGE_NAME
        ):
            raise ValueError(
                "'identity_no_edge' cannot be a reverse relation."
            )

    def _validate_control_semantics(self) -> None:
        canonical_control = (
            self.name in CONTROL_RELATION_NAMES
            and self.name != TOPOLOGY_ONLY_NO_EDGE_NAME
        )

        if self.is_control != canonical_control:
            raise ValueError(
                f"Relation {self.name!r} has is_control="
                f"{self.is_control}, but constants.py classifies it as "
                f"is_control={canonical_control}."
            )

        if (
            self.evidence_type
            == RelationEvidenceType.SYNTHETIC_CONTROL
            and not self.is_control
        ):
            raise ValueError(
                "Synthetic-control evidence requires is_control=True."
            )

        if (
            self.construction_mode
            == RelationConstructionMode.SYNTHETIC_CONTROL
            and not self.is_control
        ):
            raise ValueError(
                "Synthetic-control construction requires "
                "is_control=True."
            )

        if (
            self.is_control
            and self.explanation_policy
            == RelationExplanationPolicy.ALLOWED
        ):
            raise ValueError(
                f"Control relation {self.name!r} cannot be marked as an "
                "ordinary scientific explanation. Use diagnostic_only "
                "or excluded."
            )

    def _validate_construction_semantics(self) -> None:
        if (
            self.evidence_type
            == RelationEvidenceType.LEARNED_FROM_DATA
            and self.construction_mode
            not in {
                RelationConstructionMode.TRAINING_FITTED,
                RelationConstructionMode.AS_OF_ORIGIN,
            }
        ):
            raise ValueError(
                "Relations learned from data must use training_fitted "
                "or as_of_origin construction."
            )

        if (
            self.construction_mode
            == RelationConstructionMode.TRAINING_FITTED
            and self.leakage_risk
            not in {
                RelationLeakageRisk.MODERATE,
                RelationLeakageRisk.HIGH,
            }
        ):
            raise ValueError(
                "Training-fitted relations must declare moderate or "
                "high leakage risk."
            )

        if (
            self.construction_mode
            == RelationConstructionMode.AS_OF_ORIGIN
            and self.leakage_risk == RelationLeakageRisk.NONE
        ):
            raise ValueError(
                "As-of-origin relations cannot declare no leakage risk."
            )

        if (
            self.construction_mode
            == RelationConstructionMode.SYNTHETIC_CONTROL
            and self.evidence_type
            != RelationEvidenceType.SYNTHETIC_CONTROL
        ):
            raise ValueError(
                "Synthetic-control construction requires synthetic-control "
                "evidence."
            )

    def _validate_temporal_semantics(self) -> None:
        if (
            self.temporal_mode == RelationTemporalMode.LAGGED
            and not self.is_directed
        ):
            raise ValueError(
                "Lagged temporal relations must be directed."
            )

        if (
            self.requires_as_of_time
            and self.temporal_mode == RelationTemporalMode.STATIC
        ):
            raise ValueError(
                "Relations requiring as-of-time construction cannot use "
                "the static temporal mode."
            )

    def _validate_attribute_contracts(self) -> None:
        required_names = self.required_attribute_names
        optional_names = self.optional_attribute_names

        _require_unique_strings(
            "required edge attribute names",
            required_names,
        )
        _require_unique_strings(
            "optional edge attribute names",
            optional_names,
        )

        overlap = sorted(
            set(required_names) & set(optional_names)
        )

        if overlap:
            raise ValueError(
                "Edge attributes cannot be both required and optional: "
                f"{overlap}."
            )

        for attribute in (
            self.required_edge_attributes
            + self.optional_edge_attributes
        ):
            if not isinstance(attribute, EdgeAttributeSpec):
                raise TypeError(
                    "Edge-attribute collections must contain "
                    "EdgeAttributeSpec objects."
                )

            attribute.validate()

    def _validate_availability_policy(self) -> None:
        if self.deprecated and (
            self.message_passing_allowed
            or self.training_allowed
        ):
            raise ValueError(
                f"Deprecated relation {self.name!r} cannot remain "
                "training- or message-passing-enabled."
            )

        if (
            self.training_allowed
            and not self.message_passing_allowed
        ):
            raise ValueError(
                "training_allowed=True requires "
                "message_passing_allowed=True."
            )

        if (
            not self.message_passing_allowed
            and self.explanation_policy
            == RelationExplanationPolicy.ALLOWED
        ):
            raise ValueError(
                f"Relation {self.name!r} cannot be an ordinary message "
                "path explanation when message passing is disabled."
            )

        if (
            not self.implemented
            and self.explanation_policy
            == RelationExplanationPolicy.ALLOWED
        ):
            raise ValueError(
                f"Unimplemented relation {self.name!r} cannot be marked "
                "as an available scientific explanation."
            )

    def _validate_self_loop_policy(self) -> None:
        if not self.allows_self_loops:
            return

        if (
            ANY_NODE_TYPE not in self.source_node_types
            and ANY_NODE_TYPE not in self.target_node_types
            and not (
                set(self.source_node_types)
                & set(self.target_node_types)
            )
        ):
            raise ValueError(
                f"Relation {self.name!r} permits self-loops but its "
                "source and target node-type constraints are disjoint."
            )

    # ------------------------------------------------------------------
    # Endpoint and attribute helpers
    # ------------------------------------------------------------------

    def supports_source_type(self, node_type: str) -> bool:
        _require_nonempty_string(
            "node_type",
            node_type,
        )

        return (
            ANY_NODE_TYPE in self.source_node_types
            or node_type in self.source_node_types
        )

    def supports_target_type(self, node_type: str) -> bool:
        _require_nonempty_string(
            "node_type",
            node_type,
        )

        return (
            ANY_NODE_TYPE in self.target_node_types
            or node_type in self.target_node_types
        )

    def supports_node_pair(
        self,
        source_node_type: str,
        target_node_type: str,
    ) -> bool:
        direct_match = (
            self.supports_source_type(source_node_type)
            and self.supports_target_type(target_node_type)
        )

        if direct_match:
            return True

        if not self.is_undirected:
            return False

        return (
            self.supports_source_type(target_node_type)
            and self.supports_target_type(source_node_type)
        )

    def attribute_spec(
        self,
        attribute_name: str,
    ) -> EdgeAttributeSpec:
        _require_nonempty_string(
            "attribute_name",
            attribute_name,
        )

        for attribute in (
            self.required_edge_attributes
            + self.optional_edge_attributes
        ):
            if attribute.name == attribute_name:
                return attribute

        raise KeyError(
            f"Relation {self.name!r} does not declare edge attribute "
            f"{attribute_name!r}."
        )

    def requires_attribute(self, attribute_name: str) -> bool:
        return attribute_name in self.required_attribute_names

    def permits_attribute(self, attribute_name: str) -> bool:
        return attribute_name in self.allowed_attribute_names

    # ------------------------------------------------------------------
    # Immutable updates and serialization
    # ------------------------------------------------------------------

    def replace(self, **changes: Any) -> Self:
        return dataclass_replace(self, **changes)

    def to_dict(self) -> dict[str, Any]:
        return {
            "relation_id": self.relation_id,
            "name": self.name,
            "display_name": self.display_name,
            "description": self.description,
            "semantic_role": self.semantic_role,
            "source_node_types": list(
                self.source_node_types
            ),
            "target_node_types": list(
                self.target_node_types
            ),
            "direction": self.direction.value,
            "evidence_type": self.evidence_type.value,
            "construction_mode": self.construction_mode.value,
            "leakage_risk": self.leakage_risk.value,
            "temporal_mode": self.temporal_mode.value,
            "parent_relation_name": self.parent_relation_name,
            "reverse_relation_name": self.reverse_relation_name,
            "implementation_state": self.implementation_state,
            "is_control": self.is_control,
            "allow_any_node_type": self.allow_any_node_type,
            "allows_self_loops": self.allows_self_loops,
            "message_passing_allowed": (
                self.message_passing_allowed
            ),
            "training_allowed": self.training_allowed,
            "explanation_policy": (
                self.explanation_policy.value
            ),
            "required_edge_attributes": [
                attribute.to_dict()
                for attribute in self.required_edge_attributes
            ],
            "optional_edge_attributes": [
                attribute.to_dict()
                for attribute in self.optional_edge_attributes
            ],
            "tags": list(self.tags),
            "registry_version": self.registry_version,
            "spec_schema_version": self.spec_schema_version,
        }

    @classmethod
    def from_dict(
        cls,
        payload: Mapping[str, Any],
    ) -> RelationSpec:
        mapping = dict(
            _require_mapping("RelationSpec", payload)
        )
        _reject_unknown_fields(cls, mapping)

        enum_builders = {
            "direction": RelationDirection,
            "evidence_type": RelationEvidenceType,
            "construction_mode": RelationConstructionMode,
            "leakage_risk": RelationLeakageRisk,
            "temporal_mode": RelationTemporalMode,
            "explanation_policy": RelationExplanationPolicy,
        }

        for field_name, enum_type in enum_builders.items():
            if field_name in mapping:
                mapping[field_name] = enum_type(
                    mapping[field_name]
                )

        for field_name in (
            "source_node_types",
            "target_node_types",
            "tags",
        ):
            if field_name in mapping:
                mapping[field_name] = _as_tuple(
                    field_name,
                    mapping[field_name],
                )

        for field_name in (
            "required_edge_attributes",
            "optional_edge_attributes",
        ):
            if field_name in mapping:
                values = _as_tuple(
                    field_name,
                    mapping[field_name],
                )
                mapping[field_name] = tuple(
                    EdgeAttributeSpec.from_dict(
                        _require_mapping(
                            f"{field_name}[{index}]",
                            value,
                        )
                    )
                    for index, value in enumerate(values)
                )

        return cls(**mapping)


# =============================================================================
# Collection validation
# =============================================================================


def validate_relation_spec_collection(
    specifications: Sequence[RelationSpec],
    *,
    require_current_registry_version: bool = True,
    require_current_spec_schema_version: bool = True,
) -> tuple[RelationSpec, ...]:
    """
    Validate a complete immutable relation ontology collection.

    Stable relation IDs are only required to be unique and nonnegative. Dense
    runtime indices are assigned later by ``RelationRegistry.compile()``.

    Validation includes:

    - stable-ID uniqueness;
    - canonical-name uniqueness;
    - registry-version consistency;
    - specification-schema consistency;
    - strict reciprocal reverse declarations;
    - reversed endpoint constraints;
    - compatible reverse metadata;
    - hierarchy parent existence;
    - hierarchy cycle detection.
    """

    if not specifications:
        raise ValueError(
            "A relation registry must contain at least one relation."
        )

    validated = tuple(specifications)

    for index, specification in enumerate(validated):
        if not isinstance(specification, RelationSpec):
            raise TypeError(
                f"specifications[{index}] must be a RelationSpec."
            )

        specification.validate()

    relation_ids = tuple(
        specification.relation_id
        for specification in validated
    )
    relation_names = tuple(
        specification.name
        for specification in validated
    )

    duplicate_ids = sorted(
        value
        for value, count in Counter(relation_ids).items()
        if count > 1
    )
    duplicate_names = sorted(
        value
        for value, count in Counter(relation_names).items()
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

    _validate_collection_versions(
        validated,
        require_current_registry_version=(
            require_current_registry_version
        ),
        require_current_spec_schema_version=(
            require_current_spec_schema_version
        ),
    )

    by_name = {
        specification.name: specification
        for specification in validated
    }

    _validate_reverse_relations(
        validated,
        by_name,
    )
    _validate_relation_hierarchy(
        validated,
        by_name,
    )

    return validated


def _validate_collection_versions(
    specifications: Sequence[RelationSpec],
    *,
    require_current_registry_version: bool,
    require_current_spec_schema_version: bool,
) -> None:
    registry_versions = {
        specification.registry_version
        for specification in specifications
    }
    schema_versions = {
        specification.spec_schema_version
        for specification in specifications
    }

    if len(registry_versions) != 1:
        raise ValueError(
            "All relation specifications in one registry must use the "
            "same registry_version. Observed: "
            f"{sorted(registry_versions)}."
        )

    if len(schema_versions) != 1:
        raise ValueError(
            "All relation specifications in one registry must use the "
            "same spec_schema_version. Observed: "
            f"{sorted(schema_versions)}."
        )

    if (
        require_current_registry_version
        and registry_versions != {RELATION_REGISTRY_VERSION}
    ):
        raise ValueError(
            "Relation specifications use an incompatible registry "
            f"version. Observed {sorted(registry_versions)}, expected "
            f"{RELATION_REGISTRY_VERSION!r}."
        )

    if (
        require_current_spec_schema_version
        and schema_versions != {RELATION_SPEC_SCHEMA_VERSION}
    ):
        raise ValueError(
            "Relation specifications use an incompatible specification "
            f"schema. Observed {sorted(schema_versions)}, expected "
            f"{RELATION_SPEC_SCHEMA_VERSION!r}."
        )


def _validate_reverse_relations(
    specifications: Sequence[RelationSpec],
    by_name: Mapping[str, RelationSpec],
) -> None:
    for specification in specifications:
        reverse_name = specification.reverse_relation_name

        if reverse_name is None:
            continue

        if reverse_name not in by_name:
            raise ValueError(
                f"Relation {specification.name!r} declares reverse "
                f"relation {reverse_name!r}, but it is absent."
            )

        reverse = by_name[reverse_name]

        if not reverse.is_directed:
            raise ValueError(
                f"Reverse relation {reverse.name!r} must be directed."
            )

        if reverse.reverse_relation_name != specification.name:
            raise ValueError(
                "Reverse relation declarations must be strictly "
                "reciprocal: "
                f"{specification.name!r} -> {reverse.name!r}, but "
                f"{reverse.name!r} -> "
                f"{reverse.reverse_relation_name!r}."
            )

        if not _endpoint_constraints_are_reversed(
            specification,
            reverse,
        ):
            raise ValueError(
                f"Reverse relation {reverse.name!r} does not reverse "
                f"the endpoint constraints of "
                f"{specification.name!r}."
            )

        _validate_reverse_metadata_compatibility(
            specification,
            reverse,
        )


def _endpoint_constraints_are_reversed(
    relation: RelationSpec,
    reverse: RelationSpec,
) -> bool:
    return (
        _node_type_sets_equal(
            relation.source_node_types,
            reverse.target_node_types,
        )
        and _node_type_sets_equal(
            relation.target_node_types,
            reverse.source_node_types,
        )
    )


def _validate_reverse_metadata_compatibility(
    relation: RelationSpec,
    reverse: RelationSpec,
) -> None:
    comparable_fields = (
        "registry_version",
        "spec_schema_version",
        "evidence_type",
        "construction_mode",
        "leakage_risk",
        "temporal_mode",
        "is_control",
    )

    mismatches = [
        field_name
        for field_name in comparable_fields
        if getattr(relation, field_name)
        != getattr(reverse, field_name)
    ]

    if mismatches:
        raise ValueError(
            f"Reverse relations {relation.name!r} and "
            f"{reverse.name!r} disagree on shared metadata fields: "
            f"{mismatches}."
        )


def _validate_relation_hierarchy(
    specifications: Sequence[RelationSpec],
    by_name: Mapping[str, RelationSpec],
) -> None:
    for specification in specifications:
        parent_name = specification.parent_relation_name

        if parent_name is None:
            continue

        if parent_name not in by_name:
            raise ValueError(
                f"Relation {specification.name!r} declares parent "
                f"{parent_name!r}, but that relation is absent."
            )

    visit_state: dict[str, int] = {
        specification.name: 0
        for specification in specifications
    }
    path: list[str] = []

    def visit(relation_name: str) -> None:
        state = visit_state[relation_name]

        if state == 2:
            return

        if state == 1:
            cycle_start = path.index(relation_name)
            cycle = path[cycle_start:] + [relation_name]
            raise ValueError(
                "Relation hierarchy contains a cycle: "
                + " -> ".join(cycle)
            )

        visit_state[relation_name] = 1
        path.append(relation_name)

        parent_name = by_name[
            relation_name
        ].parent_relation_name

        if parent_name is not None:
            visit(parent_name)

        path.pop()
        visit_state[relation_name] = 2

    for relation_name in by_name:
        visit(relation_name)


# =============================================================================
# Immutable indexes
# =============================================================================


def relation_specs_by_id(
    specifications: Sequence[RelationSpec],
) -> Mapping[int, RelationSpec]:
    """
    Return an immutable stable-ID index.

    This is not a dense runtime relation-index mapping.
    """

    validated = validate_relation_spec_collection(
        specifications,
        require_current_registry_version=False,
        require_current_spec_schema_version=False,
    )

    return MappingProxyType(
        {
            specification.relation_id: specification
            for specification in validated
        }
    )


def relation_specs_by_name(
    specifications: Sequence[RelationSpec],
) -> Mapping[str, RelationSpec]:
    """Return an immutable canonical-name index."""

    validated = validate_relation_spec_collection(
        specifications,
        require_current_registry_version=False,
        require_current_spec_schema_version=False,
    )

    return MappingProxyType(
        {
            specification.name: specification
            for specification in validated
        }
    )


__all__ = (
    "ANY_NODE_TYPE",
    "EdgeAttributeKind",
    "EdgeAttributeSpec",
    "FUNCTIONAL_RELATION_ROLES",
    "MissingValuePolicy",
    "RELATION_SPEC_SCHEMA_VERSION",
    "RelationConstructionMode",
    "RelationDirection",
    "RelationEvidenceType",
    "RelationExplanationPolicy",
    "RelationLeakageRisk",
    "RelationSpec",
    "RelationTemporalMode",
    "TEMPORAL_FIELD_EDGE_LAG",
    "TEMPORAL_FIELD_EDGE_OBSERVATION_TIME",
    "TEMPORAL_FIELD_EDGE_VALID_FROM",
    "TEMPORAL_FIELD_EDGE_VALID_TO",
    "TOPOLOGY_ONLY_NO_EDGE_NAME",
    "relation_specs_by_id",
    "relation_specs_by_name",
    "validate_relation_spec_collection",
)