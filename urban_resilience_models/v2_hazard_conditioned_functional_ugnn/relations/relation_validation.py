"""
Validation of concrete relation-edge artifacts for the V2 functional UGNN.

Target path:
    urban_resilience_models/
        v2_hazard_conditioned_functional_ugnn/
            relations/
                relation_validation.py

This module validates a PyTorch-normalized relation graph against a
``CompiledRelationRegistry``.

It owns:

- structural tensor validation;
- canonical node-type validation;
- packed-graph membership validation;
- compiled-registry and graph-artifact identity checks;
- dense relation-index and stable-ID alignment;
- exact endpoint-pair validation;
- relation-specific edge-attribute validation;
- raw-versus-model missingness contracts;
- masked temporal-field validation;
- temporal leakage checks;
- graph-scoped construction provenance;
- expected provenance fingerprint verification;
- control-graph reproducibility validation;
- duplicate-edge diagnostics;
- undirected reciprocal-storage diagnostics;
- relation-coverage diagnostics;
- structured reports suitable for artifacts and CI.

It does not own:

- relation ontology definitions;
- relation-registry compilation;
- graph construction;
- feature imputation or scaling;
- hazard-relation priors;
- message passing;
- filesystem persistence.

Execution policy
----------------
This is a strict artifact validator. It should normally run:

- on CPU;
- when a graph artifact is created;
- when a persisted artifact is loaded;
- before an experiment begins.

It should not run inside the training hot path or once per epoch.

Time contract
-------------
All comparable times use one declared numeric encoding, such as:

- integer month index;
- Unix seconds;
- ordinal day.

Calendar strings are not accepted by this validator. Raw adapters must parse
them before constructing ``RelationEdgeData``.

Mixed-relation temporal fields
------------------------------
Temporal values are represented by ``TemporalEdgeColumn`` objects containing:

- one aligned value vector ``[E]``;
- one Boolean applicability mask ``[E]``.

This permits one packed graph to contain static, snapshot, interval-valid, and
lagged relations without arbitrary NaN placeholders.

Missingness contract
--------------------
Raw edge attributes may contain null values only when permitted by their
``EdgeAttributeSpec``.

Model-facing values must:

- be imputed;
- remain valid under the attribute's logical type and bounds;
- be finite when required;
- carry a Boolean missingness mask when raw missingness was permitted.

A missingness mask records original missingness. It does not excuse an invalid
or non-finite imputed value.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass, field
from enum import StrEnum
import math
from types import MappingProxyType
from typing import Any, Final, Mapping, Sequence

import torch

from ..constants import CANONICAL_NODE_TYPE_NAMES
from .relation_registry import (
    CompiledRelationRegistry,
    RelationRegistry,
    RelationRegistryEntry,
)
from .relation_types import (
    EdgeAttributeKind,
    EdgeAttributeSpec,
    RelationConstructionMode,
    RelationDirection,
    RelationLeakageRisk,
    RelationTemporalMode,
    TEMPORAL_FIELD_EDGE_LAG,
    TEMPORAL_FIELD_EDGE_OBSERVATION_TIME,
    TEMPORAL_FIELD_EDGE_VALID_FROM,
    TEMPORAL_FIELD_EDGE_VALID_TO,
)


# =============================================================================
# Schema identity and provenance vocabulary
# =============================================================================


RELATION_VALIDATION_SCHEMA_VERSION: Final[str] = "0.2"
RELATION_PROVENANCE_SCHEMA_VERSION: Final[str] = "0.2"
CONTROL_PROVENANCE_SCHEMA_VERSION: Final[str] = "0.1"

PROVENANCE_TRAINING_SPLIT_FINGERPRINT: Final[str] = (
    "training_split_fingerprint"
)
PROVENANCE_TRAINING_FIT_CUTOFF: Final[str] = (
    "training_fit_cutoff"
)
PROVENANCE_CONSTRUCTION_AS_OF_TIME: Final[str] = (
    "construction_as_of_time"
)
PROVENANCE_SOURCE_ARTIFACT_FINGERPRINT: Final[str] = (
    "source_artifact_fingerprint"
)


# =============================================================================
# Validation vocabulary
# =============================================================================


class ValidationSeverity(StrEnum):
    ERROR = "error"
    WARNING = "warning"


class AttributeRepresentation(StrEnum):
    """
    Processing stage of edge-attribute values.

    RAW
        Permitted nulls may still be present.

    MODEL
        Values have been imputed and normalized for model consumption.
    """

    RAW = "raw"
    MODEL = "model"


class RelationValidationProfile(StrEnum):
    """
    Validation rigor profile.

    DEVELOPMENT
        Suitable for in-memory graph construction and unit tests. Registry
        fingerprints and source-registry verification are optional.

    PERSISTED_ARTIFACT
        Requires compiled-registry and graph-artifact fingerprints.

    PUBLICATION
        Requires artifact fingerprints, source-registry verification, expected
        provenance identities, CPU validation, and promotes warnings to errors.
    """

    DEVELOPMENT = "development"
    PERSISTED_ARTIFACT = "persisted_artifact"
    PUBLICATION = "publication"


class TimeEncoding(StrEnum):
    """Declared numerical encoding shared by every temporal value."""

    MONTH_INDEX = "month_index"
    UNIX_SECONDS = "unix_seconds"
    ORDINAL_DAY = "ordinal_day"
    INTEGER_PERIOD_INDEX = "integer_period_index"


class ControlGraphKind(StrEnum):
    """Reproducible control-graph construction category."""

    RANDOM_PLACEBO = "random_placebo"
    CENTROID_KNN = "centroid_knn"
    OTHER = "other"


# =============================================================================
# Generic helpers
# =============================================================================


def _require_nonempty_string(name: str, value: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string.")


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


def _is_integer_tensor(tensor: torch.Tensor) -> bool:
    return tensor.dtype in {
        torch.int8,
        torch.int16,
        torch.int32,
        torch.int64,
        torch.uint8,
    }


def _is_numeric_tensor(tensor: torch.Tensor) -> bool:
    return (
        tensor.dtype.is_floating_point
        or _is_integer_tensor(tensor)
    )


def _tensor_scalar(
    tensor: torch.Tensor,
    index: int,
) -> float:
    value = tensor[index]

    if value.ndim != 0:
        raise RuntimeError("Expected a scalar tensor value.")

    result = float(value.item())

    if not math.isfinite(result):
        raise ValueError("Numerical validation values must be finite.")

    return result


def _value_is_missing(value: Any) -> bool:
    if value is None:
        return True

    if isinstance(value, float):
        return math.isnan(value)

    return False


def _coerce_finite_number(value: Any) -> float | None:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
    ):
        return None

    converted = float(value)

    if not math.isfinite(converted):
        return None

    return converted


def _within_bounds(
    value: float,
    specification: EdgeAttributeSpec,
    *,
    tolerance: float,
) -> bool:
    if (
        specification.minimum is not None
        and value
        < float(specification.minimum) - tolerance
    ):
        return False

    if (
        specification.maximum is not None
        and value
        > float(specification.maximum) + tolerance
    ):
        return False

    return True


def _canonicalize_parameter_value(
    name: str,
    value: Any,
) -> str | int | float | bool:
    if isinstance(value, bool):
        return value

    if isinstance(value, int):
        return value

    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(
                f"Control parameter {name!r} must be finite."
            )
        return value

    if isinstance(value, str) and value:
        return value

    raise TypeError(
        f"Control parameter {name!r} must be a nonempty string, "
        "Boolean, integer, or finite float."
    )


# =============================================================================
# Validation findings and reports
# =============================================================================


@dataclass(slots=True, frozen=True)
class ValidationIssue:
    """One relation-data validation finding."""

    severity: ValidationSeverity
    code: str
    message: str

    relation_name: str | None = None
    graph_index: int | None = None
    edge_indices: tuple[int, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(
            self.severity,
            ValidationSeverity,
        ):
            raise TypeError(
                "severity must be a ValidationSeverity."
            )

        _require_nonempty_string("issue code", self.code)
        _require_nonempty_string("issue message", self.message)

        if self.relation_name is not None:
            _require_nonempty_string(
                "relation_name",
                self.relation_name,
            )

        if self.graph_index is not None:
            if (
                isinstance(self.graph_index, bool)
                or not isinstance(self.graph_index, int)
                or self.graph_index < 0
            ):
                raise ValueError(
                    "graph_index must be absent or a nonnegative integer."
                )

        for edge_index in self.edge_indices:
            if (
                isinstance(edge_index, bool)
                or not isinstance(edge_index, int)
                or edge_index < 0
            ):
                raise ValueError(
                    "edge_indices must contain nonnegative integers."
                )

    def to_dict(self) -> dict[str, Any]:
        return {
            "severity": self.severity.value,
            "code": self.code,
            "message": self.message,
            "relation_name": self.relation_name,
            "graph_index": self.graph_index,
            "edge_indices": list(self.edge_indices),
        }


@dataclass(slots=True, frozen=True)
class RelationValidationReport:
    """Immutable complete validation report."""

    issues: tuple[ValidationIssue, ...]

    profile: RelationValidationProfile
    schema_version: str

    num_nodes: int
    num_edges: int
    num_graphs: int
    num_relations_observed: int

    @property
    def errors(self) -> tuple[ValidationIssue, ...]:
        return tuple(
            issue
            for issue in self.issues
            if issue.severity == ValidationSeverity.ERROR
        )

    @property
    def warnings(self) -> tuple[ValidationIssue, ...]:
        return tuple(
            issue
            for issue in self.issues
            if issue.severity == ValidationSeverity.WARNING
        )

    @property
    def valid(self) -> bool:
        return not self.errors

    @property
    def counts_by_code(self) -> Mapping[str, int]:
        return MappingProxyType(
            dict(
                Counter(
                    issue.code
                    for issue in self.issues
                )
            )
        )

    @property
    def counts_by_relation(self) -> Mapping[str, int]:
        return MappingProxyType(
            dict(
                Counter(
                    (
                        issue.relation_name
                        if issue.relation_name is not None
                        else "<global>"
                    )
                    for issue in self.issues
                )
            )
        )

    @property
    def counts_by_severity(self) -> Mapping[str, int]:
        return MappingProxyType(
            dict(
                Counter(
                    issue.severity.value
                    for issue in self.issues
                )
            )
        )

    def raise_for_errors(self) -> None:
        if not self.valid:
            raise RelationValidationError(self)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "profile": self.profile.value,
            "valid": self.valid,
            "num_nodes": self.num_nodes,
            "num_edges": self.num_edges,
            "num_graphs": self.num_graphs,
            "num_relations_observed": (
                self.num_relations_observed
            ),
            "counts_by_severity": dict(
                self.counts_by_severity
            ),
            "counts_by_code": dict(self.counts_by_code),
            "counts_by_relation": dict(
                self.counts_by_relation
            ),
            "issues": [
                issue.to_dict()
                for issue in self.issues
            ],
        }


class RelationValidationError(ValueError):
    """Raised when a validation report contains errors."""

    def __init__(
        self,
        report: RelationValidationReport,
    ) -> None:
        self.report = report

        preview = report.errors[:8]
        rendered = "\n".join(
            f"- [{issue.code}] {issue.message}"
            for issue in preview
        )

        remaining = len(report.errors) - len(preview)

        if remaining:
            rendered += (
                f"\n- ... and {remaining} additional error(s)."
            )

        super().__init__(
            "Relation-edge validation failed with "
            f"{len(report.errors)} error(s):\n{rendered}"
        )


# =============================================================================
# Aligned edge columns
# =============================================================================


@dataclass(slots=True, frozen=True)
class EdgeAttributeColumn:
    """
    One aligned edge-attribute column.

    ``missing_mask[e] == True`` means that edge ``e`` was missing in the raw
    source and has been imputed in model-facing data.
    """

    values: torch.Tensor | Sequence[Any]
    missing_mask: torch.Tensor | None = None

    def __post_init__(self) -> None:
        if isinstance(self.values, torch.Tensor):
            if self.values.ndim != 1:
                raise ValueError(
                    "EdgeAttributeColumn tensor values must be "
                    "one-dimensional."
                )

        elif isinstance(self.values, Sequence) and not isinstance(
            self.values,
            (str, bytes),
        ):
            object.__setattr__(
                self,
                "values",
                tuple(self.values),
            )

        else:
            raise TypeError(
                "values must be a one-dimensional tensor or a "
                "non-string sequence."
            )

        if self.missing_mask is not None:
            if not isinstance(
                self.missing_mask,
                torch.Tensor,
            ):
                raise TypeError(
                    "missing_mask must be a tensor."
                )

            if (
                self.missing_mask.ndim != 1
                or self.missing_mask.dtype != torch.bool
            ):
                raise ValueError(
                    "missing_mask must be a one-dimensional Boolean "
                    "tensor."
                )

            if len(self) != int(
                self.missing_mask.shape[0]
            ):
                raise ValueError(
                    "missing_mask length must equal values length."
                )

    def __len__(self) -> int:
        if isinstance(self.values, torch.Tensor):
            return int(self.values.shape[0])

        return len(self.values)

    def value_at(self, index: int) -> Any:
        if isinstance(self.values, torch.Tensor):
            value = self.values[index]

            if value.ndim != 0:
                raise RuntimeError(
                    "Expected a scalar edge-attribute value."
                )

            return value.item()

        return self.values[index]

    def is_masked_missing(self, index: int) -> bool:
        if self.missing_mask is None:
            return False

        return bool(self.missing_mask[index].item())


@dataclass(slots=True, frozen=True)
class TemporalEdgeColumn:
    """
    One numeric temporal field aligned to every edge.

    Non-applicable edges are identified through ``applicable_mask`` rather
    than NaN or arbitrary temporal sentinels.

    Every stored numerical value must remain finite, including values at
    non-applicable positions.
    """

    values: torch.Tensor
    applicable_mask: torch.Tensor

    def __post_init__(self) -> None:
        if not isinstance(self.values, torch.Tensor):
            raise TypeError(
                "TemporalEdgeColumn.values must be a tensor."
            )

        if not isinstance(
            self.applicable_mask,
            torch.Tensor,
        ):
            raise TypeError(
                "TemporalEdgeColumn.applicable_mask must be a tensor."
            )

        if self.values.ndim != 1:
            raise ValueError(
                "TemporalEdgeColumn.values must be one-dimensional."
            )

        if (
            self.applicable_mask.ndim != 1
            or self.applicable_mask.dtype != torch.bool
        ):
            raise ValueError(
                "applicable_mask must be a one-dimensional Boolean "
                "tensor."
            )

        if (
            self.values.shape[0]
            != self.applicable_mask.shape[0]
        ):
            raise ValueError(
                "Temporal values and applicability mask must have the "
                "same length."
            )

        if not _is_numeric_tensor(self.values):
            raise ValueError(
                "Temporal values must use a numeric tensor dtype."
            )

        if (
            self.values.dtype.is_floating_point
            and not bool(
                torch.isfinite(self.values).all().item()
            )
        ):
            raise ValueError(
                "Temporal values cannot contain NaN or infinity."
            )

    def __len__(self) -> int:
        return int(self.values.shape[0])

    def applies(self, index: int) -> bool:
        return bool(
            self.applicable_mask[index].item()
        )

    def value_at(self, index: int) -> float:
        return _tensor_scalar(self.values, index)


# =============================================================================
# Construction provenance
# =============================================================================


@dataclass(slots=True, frozen=True)
class RelationConstructionProvenance:
    """Construction metadata for one relation in one packed graph."""

    relation_name: str
    graph_index: int

    construction_as_of_time: int | float | None = None
    training_fit_cutoff: int | float | None = None

    training_split_fingerprint: str | None = None
    source_artifact_fingerprint: str | None = None
    builder_version: str | None = None

    schema_version: str = (
        RELATION_PROVENANCE_SCHEMA_VERSION
    )

    def __post_init__(self) -> None:
        _require_nonempty_string(
            "relation_name",
            self.relation_name,
        )

        if (
            isinstance(self.graph_index, bool)
            or not isinstance(self.graph_index, int)
            or self.graph_index < 0
        ):
            raise ValueError(
                "graph_index must be a nonnegative integer."
            )

        for field_name, value in (
            (
                PROVENANCE_CONSTRUCTION_AS_OF_TIME,
                self.construction_as_of_time,
            ),
            (
                PROVENANCE_TRAINING_FIT_CUTOFF,
                self.training_fit_cutoff,
            ),
        ):
            if value is not None:
                _require_finite_number(
                    field_name,
                    value,
                )

        for field_name, value in (
            (
                PROVENANCE_TRAINING_SPLIT_FINGERPRINT,
                self.training_split_fingerprint,
            ),
            (
                PROVENANCE_SOURCE_ARTIFACT_FINGERPRINT,
                self.source_artifact_fingerprint,
            ),
            ("builder_version", self.builder_version),
            ("schema_version", self.schema_version),
        ):
            if value is not None:
                _require_nonempty_string(
                    field_name,
                    value,
                )

        if (
            self.training_fit_cutoff is not None
            and self.construction_as_of_time is not None
            and self.training_fit_cutoff
            > self.construction_as_of_time
        ):
            raise ValueError(
                "training_fit_cutoff cannot be later than "
                "construction_as_of_time."
            )


@dataclass(slots=True, frozen=True)
class ControlGraphProvenance:
    """Reproducibility metadata for one control relation and graph."""

    relation_name: str
    graph_index: int

    control_kind: ControlGraphKind

    generator_name: str
    generator_version: str

    source_graph_fingerprint: str

    random_seed: int | None = None

    preserves_edge_count: bool = False
    preserves_degree_distribution: bool = False
    preserves_node_types: bool = True

    parameters: Mapping[
        str,
        str | int | float | bool,
    ] = field(default_factory=dict)

    schema_version: str = CONTROL_PROVENANCE_SCHEMA_VERSION

    def __post_init__(self) -> None:
        _require_nonempty_string(
            "control relation_name",
            self.relation_name,
        )
        _require_nonempty_string(
            "generator_name",
            self.generator_name,
        )
        _require_nonempty_string(
            "generator_version",
            self.generator_version,
        )
        _require_nonempty_string(
            "source_graph_fingerprint",
            self.source_graph_fingerprint,
        )
        _require_nonempty_string(
            "control schema_version",
            self.schema_version,
        )

        if (
            isinstance(self.graph_index, bool)
            or not isinstance(self.graph_index, int)
            or self.graph_index < 0
        ):
            raise ValueError(
                "graph_index must be a nonnegative integer."
            )

        if not isinstance(
            self.control_kind,
            ControlGraphKind,
        ):
            raise TypeError(
                "control_kind must be a ControlGraphKind."
            )

        if self.random_seed is not None and (
            isinstance(self.random_seed, bool)
            or not isinstance(self.random_seed, int)
            or self.random_seed < 0
        ):
            raise ValueError(
                "random_seed must be absent or a nonnegative integer."
            )

        normalized_parameters: dict[
            str,
            str | int | float | bool,
        ] = {}

        for key, value in self.parameters.items():
            _require_nonempty_string(
                "control parameter name",
                key,
            )
            normalized_parameters[key] = (
                _canonicalize_parameter_value(
                    key,
                    value,
                )
            )

        object.__setattr__(
            self,
            "parameters",
            MappingProxyType(normalized_parameters),
        )

        if (
            self.control_kind
            == ControlGraphKind.RANDOM_PLACEBO
            and self.random_seed is None
        ):
            raise ValueError(
                "Random-placebo provenance requires random_seed."
            )

        if self.control_kind == ControlGraphKind.CENTROID_KNN:
            required_parameters = {
                "k",
                "distance_metric",
                "coordinate_reference_system",
                "tie_breaking_policy",
                "same_type_enforced",
            }
            missing = sorted(
                required_parameters
                - set(self.parameters)
            )

            if missing:
                raise ValueError(
                    "Centroid-kNN provenance is missing required "
                    f"parameters: {missing}."
                )

            if self.parameters["same_type_enforced"] is not True:
                raise ValueError(
                    "Centroid-kNN controls must enforce same-type "
                    "neighbors."
                )


# =============================================================================
# External expectations and validation options
# =============================================================================


@dataclass(slots=True, frozen=True)
class RelationValidationExpectations:
    """
    External identities expected by the current experiment.

    These fields transform provenance validation from presence checks into
    identity checks.
    """

    expected_compiled_registry_fingerprint: str | None = None
    expected_graph_artifact_fingerprint: str | None = None

    expected_training_split_fingerprint: str | None = None
    expected_source_artifact_fingerprint: str | None = None

    expected_time_encoding: TimeEncoding | None = None

    def __post_init__(self) -> None:
        for name, value in (
            (
                "expected_compiled_registry_fingerprint",
                self.expected_compiled_registry_fingerprint,
            ),
            (
                "expected_graph_artifact_fingerprint",
                self.expected_graph_artifact_fingerprint,
            ),
            (
                "expected_training_split_fingerprint",
                self.expected_training_split_fingerprint,
            ),
            (
                "expected_source_artifact_fingerprint",
                self.expected_source_artifact_fingerprint,
            ),
        ):
            if value is not None:
                _require_nonempty_string(name, value)

        if (
            self.expected_time_encoding is not None
            and not isinstance(
                self.expected_time_encoding,
                TimeEncoding,
            )
        ):
            raise TypeError(
                "expected_time_encoding must be a TimeEncoding."
            )


@dataclass(slots=True, frozen=True)
class RelationValidationOptions:
    """Validation profile and optional representation diagnostics."""

    profile: RelationValidationProfile = (
        RelationValidationProfile.DEVELOPMENT
    )

    reject_duplicate_edges: bool = True

    require_reciprocal_undirected_storage: bool = False
    warn_on_nonreciprocal_undirected_storage: bool = True

    require_all_compiled_relations_present: bool = False
    warn_on_missing_compiled_relations: bool = True

    reject_values_for_undeclared_attributes: bool = True

    validity_end_inclusive: bool = False

    require_cpu_tensors: bool = True
    promote_warnings_to_errors: bool | None = None

    numeric_tolerance: float = 1e-6
    max_edge_indices_per_issue: int = 20

    def __post_init__(self) -> None:
        if not isinstance(
            self.profile,
            RelationValidationProfile,
        ):
            raise TypeError(
                "profile must be a RelationValidationProfile."
            )

        tolerance = _require_finite_number(
            "numeric_tolerance",
            self.numeric_tolerance,
        )

        if tolerance < 0.0:
            raise ValueError(
                "numeric_tolerance must be nonnegative."
            )

        if (
            isinstance(
                self.max_edge_indices_per_issue,
                bool,
            )
            or not isinstance(
                self.max_edge_indices_per_issue,
                int,
            )
            or self.max_edge_indices_per_issue <= 0
        ):
            raise ValueError(
                "max_edge_indices_per_issue must be a positive integer."
            )

    @property
    def registry_fingerprint_required(self) -> bool:
        return self.profile in {
            RelationValidationProfile.PERSISTED_ARTIFACT,
            RelationValidationProfile.PUBLICATION,
        }

    @property
    def graph_artifact_fingerprint_required(self) -> bool:
        return self.profile in {
            RelationValidationProfile.PERSISTED_ARTIFACT,
            RelationValidationProfile.PUBLICATION,
        }

    @property
    def source_registry_required(self) -> bool:
        return (
            self.profile
            == RelationValidationProfile.PUBLICATION
        )

    @property
    def expected_provenance_required(self) -> bool:
        return (
            self.profile
            == RelationValidationProfile.PUBLICATION
        )

    @property
    def warnings_are_errors(self) -> bool:
        if self.promote_warnings_to_errors is not None:
            return self.promote_warnings_to_errors

        return (
            self.profile
            == RelationValidationProfile.PUBLICATION
        )

    @property
    def cpu_required(self) -> bool:
        return (
            self.require_cpu_tensors
            or self.profile
            == RelationValidationProfile.PUBLICATION
        )


# =============================================================================
# Concrete graph payload
# =============================================================================


@dataclass(slots=True, frozen=True)
class RelationEdgeData:
    """
    PyTorch-normalized semantic representation of one packed graph batch.

    Required
    --------
    edge_index:
        Integer tensor ``[2, E]``.

    edge_relation_index:
        Dense compiled relation indices ``[E]``.

    node_type_names:
        Canonical node-type name for each packed node.

    node_batch_index:
        Contiguous graph membership IDs ``[N]`` beginning at zero.

    Temporal columns
    ----------------
    Each optional temporal field is a ``TemporalEdgeColumn`` with an explicit
    applicability mask.

    Graph-scoped provenance
    -----------------------
    Construction and control provenance records are indexed semantically by
    ``(relation_name, graph_index)``.
    """

    edge_index: torch.Tensor
    edge_relation_index: torch.Tensor

    node_type_names: tuple[str, ...]
    node_batch_index: torch.Tensor

    attributes: Mapping[str, EdgeAttributeColumn] = field(
        default_factory=dict
    )
    attribute_representation: AttributeRepresentation = (
        AttributeRepresentation.MODEL
    )

    edge_ids: tuple[str, ...] = ()
    edge_stable_relation_id: torch.Tensor | None = None
    edge_batch_index: torch.Tensor | None = None

    origin_time_by_graph: torch.Tensor | None = None
    node_time: torch.Tensor | None = None

    edge_observation_time: TemporalEdgeColumn | None = None
    edge_valid_from: TemporalEdgeColumn | None = None
    edge_valid_to: TemporalEdgeColumn | None = None
    edge_lag: TemporalEdgeColumn | None = None

    construction_provenance: tuple[
        RelationConstructionProvenance,
        ...,
    ] = ()

    control_provenance: tuple[
        ControlGraphProvenance,
        ...,
    ] = ()

    compiled_registry_fingerprint: str | None = None
    graph_artifact_fingerprint: str | None = None

    time_encoding: TimeEncoding = TimeEncoding.MONTH_INDEX

    def __post_init__(self) -> None:
        if not isinstance(
            self.attribute_representation,
            AttributeRepresentation,
        ):
            raise TypeError(
                "attribute_representation must be an "
                "AttributeRepresentation."
            )

        if not isinstance(self.time_encoding, TimeEncoding):
            raise TypeError(
                "time_encoding must be a TimeEncoding."
            )

        object.__setattr__(
            self,
            "node_type_names",
            tuple(self.node_type_names),
        )
        object.__setattr__(
            self,
            "edge_ids",
            tuple(self.edge_ids),
        )
        object.__setattr__(
            self,
            "construction_provenance",
            tuple(self.construction_provenance),
        )
        object.__setattr__(
            self,
            "control_provenance",
            tuple(self.control_provenance),
        )

        normalized_attributes: dict[
            str,
            EdgeAttributeColumn,
        ] = {}

        for name, column in self.attributes.items():
            _require_nonempty_string(
                "edge attribute name",
                name,
            )

            if not isinstance(
                column,
                EdgeAttributeColumn,
            ):
                raise TypeError(
                    f"Attribute {name!r} must be an "
                    "EdgeAttributeColumn."
                )

            normalized_attributes[name] = column

        object.__setattr__(
            self,
            "attributes",
            MappingProxyType(normalized_attributes),
        )

        for fingerprint_name, fingerprint in (
            (
                "compiled_registry_fingerprint",
                self.compiled_registry_fingerprint,
            ),
            (
                "graph_artifact_fingerprint",
                self.graph_artifact_fingerprint,
            ),
        ):
            if fingerprint is not None:
                _require_nonempty_string(
                    fingerprint_name,
                    fingerprint,
                )


# =============================================================================
# Internal validation context
# =============================================================================


@dataclass(slots=True)
class _ValidationContext:
    data: RelationEdgeData
    registry: CompiledRelationRegistry
    options: RelationValidationOptions
    expectations: RelationValidationExpectations

    issues: list[ValidationIssue]

    num_nodes: int = 0
    num_edges: int = 0
    num_graphs: int = 0

    sources: tuple[int, ...] = ()
    targets: tuple[int, ...] = ()
    relation_indices: tuple[int, ...] = ()

    node_batch_indices: tuple[int, ...] = ()
    edge_batch_indices: tuple[int, ...] = ()

    edges_by_relation: dict[int, list[int]] = field(
        default_factory=dict
    )
    edges_by_relation_and_graph: dict[
        tuple[int, int],
        list[int],
    ] = field(default_factory=dict)

    construction_provenance_by_key: dict[
        tuple[str, int],
        RelationConstructionProvenance,
    ] = field(default_factory=dict)

    control_provenance_by_key: dict[
        tuple[str, int],
        ControlGraphProvenance,
    ] = field(default_factory=dict)

    def add_error(
        self,
        code: str,
        message: str,
        *,
        relation_name: str | None = None,
        graph_index: int | None = None,
        edge_indices: Sequence[int] = (),
    ) -> None:
        self._add_issue(
            ValidationSeverity.ERROR,
            code,
            message,
            relation_name=relation_name,
            graph_index=graph_index,
            edge_indices=edge_indices,
        )

    def add_warning(
        self,
        code: str,
        message: str,
        *,
        relation_name: str | None = None,
        graph_index: int | None = None,
        edge_indices: Sequence[int] = (),
    ) -> None:
        severity = (
            ValidationSeverity.ERROR
            if self.options.warnings_are_errors
            else ValidationSeverity.WARNING
        )

        self._add_issue(
            severity,
            code,
            message,
            relation_name=relation_name,
            graph_index=graph_index,
            edge_indices=edge_indices,
        )

    def _add_issue(
        self,
        severity: ValidationSeverity,
        code: str,
        message: str,
        *,
        relation_name: str | None,
        graph_index: int | None,
        edge_indices: Sequence[int],
    ) -> None:
        limited = tuple(edge_indices)[
            : self.options.max_edge_indices_per_issue
        ]

        self.issues.append(
            ValidationIssue(
                severity=severity,
                code=code,
                message=message,
                relation_name=relation_name,
                graph_index=graph_index,
                edge_indices=limited,
            )
        )


# =============================================================================
# Validator
# =============================================================================


class RelationValidator:
    """Validate concrete edge data against a compiled registry."""

    def __init__(
        self,
        *,
        options: RelationValidationOptions | None = None,
        expectations: RelationValidationExpectations | None = None,
    ) -> None:
        self.options = (
            options
            if options is not None
            else RelationValidationOptions()
        )
        self.expectations = (
            expectations
            if expectations is not None
            else RelationValidationExpectations()
        )

    def validate(
        self,
        data: RelationEdgeData,
        registry: CompiledRelationRegistry,
        *,
        source_registry: RelationRegistry | None = None,
    ) -> RelationValidationReport:
        if not isinstance(data, RelationEdgeData):
            raise TypeError(
                "data must be a RelationEdgeData."
            )

        if not isinstance(
            registry,
            CompiledRelationRegistry,
        ):
            raise TypeError(
                "registry must be a CompiledRelationRegistry."
            )

        registry.validate()

        context = _ValidationContext(
            data=data,
            registry=registry,
            options=self.options,
            expectations=self.expectations,
            issues=[],
        )

        self._validate_source_registry(
            context,
            source_registry,
        )

        structurally_usable = self._validate_structure(
            context
        )

        if structurally_usable:
            self._validate_artifact_identity(context)
            self._validate_endpoint_contracts(context)
            self._validate_edge_attributes(context)
            self._validate_temporal_contracts(context)
            self._validate_construction_provenance(context)
            self._validate_control_provenance(context)
            self._validate_duplicates(context)
            self._validate_undirected_storage(context)
            self._validate_relation_coverage(context)

        return RelationValidationReport(
            issues=tuple(context.issues),
            profile=self.options.profile,
            schema_version=(
                RELATION_VALIDATION_SCHEMA_VERSION
            ),
            num_nodes=context.num_nodes,
            num_edges=context.num_edges,
            num_graphs=context.num_graphs,
            num_relations_observed=len(
                context.edges_by_relation
            ),
        )

    # ------------------------------------------------------------------
    # Source-registry and artifact identity
    # ------------------------------------------------------------------

    def _validate_source_registry(
        self,
        context: _ValidationContext,
        source_registry: RelationRegistry | None,
    ) -> None:
        if (
            context.options.source_registry_required
            and source_registry is None
        ):
            context.add_error(
                "missing_source_registry",
                "Publication validation requires the canonical source "
                "registry."
            )
            return

        if source_registry is None:
            return

        try:
            context.registry.assert_matches_source_registry(
                source_registry,
                require_operational_match=True,
            )
        except (TypeError, ValueError) as exc:
            context.add_error(
                "source_registry_mismatch",
                str(exc),
            )

    def _validate_artifact_identity(
        self,
        context: _ValidationContext,
    ) -> None:
        data = context.data
        expectations = context.expectations

        if (
            context.options.registry_fingerprint_required
            and data.compiled_registry_fingerprint is None
        ):
            context.add_error(
                "missing_compiled_registry_fingerprint",
                "This validation profile requires a compiled-registry "
                "fingerprint in the graph artifact."
            )

        if (
            data.compiled_registry_fingerprint is not None
            and data.compiled_registry_fingerprint
            != context.registry.fingerprint()
        ):
            context.add_error(
                "compiled_registry_fingerprint_mismatch",
                "Graph data was built under a different compiled "
                "relation registry."
            )

        if (
            expectations
            .expected_compiled_registry_fingerprint
            is not None
            and data.compiled_registry_fingerprint
            != expectations
            .expected_compiled_registry_fingerprint
        ):
            context.add_error(
                "unexpected_compiled_registry_fingerprint",
                "Graph data does not match the experiment's expected "
                "compiled-registry fingerprint."
            )

        if (
            context.options
            .graph_artifact_fingerprint_required
            and data.graph_artifact_fingerprint is None
        ):
            context.add_error(
                "missing_graph_artifact_fingerprint",
                "This validation profile requires a graph-artifact "
                "fingerprint."
            )

        if (
            expectations.expected_graph_artifact_fingerprint
            is not None
            and data.graph_artifact_fingerprint
            != expectations.expected_graph_artifact_fingerprint
        ):
            context.add_error(
                "graph_artifact_fingerprint_mismatch",
                "Graph artifact fingerprint does not match the expected "
                "experiment artifact."
            )

        if (
            expectations.expected_time_encoding is not None
            and data.time_encoding
            != expectations.expected_time_encoding
        ):
            context.add_error(
                "time_encoding_mismatch",
                "Graph temporal values use a different numerical time "
                "encoding than the experiment expects."
            )

        if context.data.edge_stable_relation_id is not None:
            stable_ids = tuple(
                int(value)
                for value in (
                    context.data.edge_stable_relation_id
                    .detach()
                    .cpu()
                    .tolist()
                )
            )

            mismatched = [
                edge_index
                for edge_index, (
                    relation_index,
                    stable_id,
                ) in enumerate(
                    zip(
                        context.relation_indices,
                        stable_ids,
                    )
                )
                if (
                    context.registry.entry_for_index(
                        relation_index
                    ).relation_id
                    != stable_id
                )
            ]

            if mismatched:
                context.add_error(
                    "stable_relation_id_mismatch",
                    "Stable relation IDs do not agree with dense "
                    "relation indices.",
                    edge_indices=mismatched,
                )

    # ------------------------------------------------------------------
    # Structural validation
    # ------------------------------------------------------------------

    def _validate_structure(
        self,
        context: _ValidationContext,
    ) -> bool:
        data = context.data

        if context.options.cpu_required:
            self._validate_cpu_tensors(context)

        if not isinstance(data.edge_index, torch.Tensor):
            context.add_error(
                "edge_index_type",
                "edge_index must be a tensor.",
            )
            return False

        if (
            data.edge_index.ndim != 2
            or data.edge_index.shape[0] != 2
        ):
            context.add_error(
                "edge_index_shape",
                "edge_index must have shape [2, E].",
            )
            return False

        if not _is_integer_tensor(data.edge_index):
            context.add_error(
                "edge_index_dtype",
                "edge_index must use an integer dtype.",
            )
            return False

        context.num_edges = int(
            data.edge_index.shape[1]
        )
        context.num_nodes = len(
            data.node_type_names
        )

        if (
            context.num_nodes == 0
            and context.num_edges > 0
        ):
            context.add_error(
                "edge_without_nodes",
                "Edges cannot reference an empty node set.",
            )
            return False

        self._validate_node_types(context)

        if not self._validate_integer_vector(
            context,
            "edge_relation_index",
            data.edge_relation_index,
            expected_length=context.num_edges,
        ):
            return False

        if not self._validate_integer_vector(
            context,
            "node_batch_index",
            data.node_batch_index,
            expected_length=context.num_nodes,
        ):
            return False

        if data.edge_batch_index is not None:
            if not self._validate_integer_vector(
                context,
                "edge_batch_index",
                data.edge_batch_index,
                expected_length=context.num_edges,
            ):
                return False

        if data.edge_stable_relation_id is not None:
            if not self._validate_integer_vector(
                context,
                "edge_stable_relation_id",
                data.edge_stable_relation_id,
                expected_length=context.num_edges,
            ):
                return False

        if data.node_time is not None:
            if not self._validate_numeric_vector(
                context,
                "node_time",
                data.node_time,
                expected_length=context.num_nodes,
            ):
                return False

        if data.origin_time_by_graph is not None:
            if not self._validate_numeric_vector(
                context,
                "origin_time_by_graph",
                data.origin_time_by_graph,
                expected_length=None,
            ):
                return False

        for field_name, column in self._temporal_columns(
            data
        ).items():
            if len(column) != context.num_edges:
                context.add_error(
                    "temporal_column_length",
                    f"Temporal field {field_name!r} has length "
                    f"{len(column)}, expected {context.num_edges}.",
                )

        for name, column in data.attributes.items():
            if len(column) != context.num_edges:
                context.add_error(
                    "attribute_length",
                    f"Attribute {name!r} has length {len(column)}, "
                    f"expected {context.num_edges}.",
                )

        self._validate_edge_ids(context)

        if context.errors_present:
            return False

        context.sources = tuple(
            int(value)
            for value in (
                data.edge_index[0]
                .detach()
                .cpu()
                .tolist()
            )
        )
        context.targets = tuple(
            int(value)
            for value in (
                data.edge_index[1]
                .detach()
                .cpu()
                .tolist()
            )
        )
        context.relation_indices = tuple(
            int(value)
            for value in (
                data.edge_relation_index
                .detach()
                .cpu()
                .tolist()
            )
        )
        context.node_batch_indices = tuple(
            int(value)
            for value in (
                data.node_batch_index
                .detach()
                .cpu()
                .tolist()
            )
        )

        if not self._validate_graph_membership(context):
            return False

        invalid_node_edges = [
            edge_index
            for edge_index, (source, target)
            in enumerate(
                zip(
                    context.sources,
                    context.targets,
                )
            )
            if (
                source < 0
                or target < 0
                or source >= context.num_nodes
                or target >= context.num_nodes
            )
        ]

        if invalid_node_edges:
            context.add_error(
                "edge_endpoint_bounds",
                "edge_index contains node indices outside the packed "
                "node range.",
                edge_indices=invalid_node_edges,
            )
            return False

        invalid_relation_edges = [
            edge_index
            for edge_index, relation_index
            in enumerate(context.relation_indices)
            if (
                relation_index < 0
                or relation_index >= len(context.registry)
            )
        ]

        if invalid_relation_edges:
            context.add_error(
                "relation_index_bounds",
                "edge_relation_index contains indices outside the "
                "compiled registry.",
                edge_indices=invalid_relation_edges,
            )
            return False

        source_batches = tuple(
            context.node_batch_indices[source]
            for source in context.sources
        )
        target_batches = tuple(
            context.node_batch_indices[target]
            for target in context.targets
        )

        cross_graph = [
            edge_index
            for edge_index, (
                source_batch,
                target_batch,
            ) in enumerate(
                zip(source_batches, target_batches)
            )
            if source_batch != target_batch
        ]

        if cross_graph:
            context.add_error(
                "cross_graph_edge",
                "Every edge must remain inside one packed graph.",
                edge_indices=cross_graph,
            )
            return False

        if data.edge_batch_index is None:
            context.edge_batch_indices = target_batches
        else:
            context.edge_batch_indices = tuple(
                int(value)
                for value in (
                    data.edge_batch_index
                    .detach()
                    .cpu()
                    .tolist()
                )
            )

            mismatched = [
                edge_index
                for edge_index, (
                    observed,
                    expected,
                ) in enumerate(
                    zip(
                        context.edge_batch_indices,
                        target_batches,
                    )
                )
                if observed != expected
            ]

            if mismatched:
                context.add_error(
                    "edge_batch_mismatch",
                    "edge_batch_index must agree with endpoint graph "
                    "membership.",
                    edge_indices=mismatched,
                )
                return False

        edges_by_relation: dict[int, list[int]] = (
            defaultdict(list)
        )
        edges_by_relation_and_graph: dict[
            tuple[int, int],
            list[int],
        ] = defaultdict(list)

        for edge_index, (
            relation_index,
            graph_index,
        ) in enumerate(
            zip(
                context.relation_indices,
                context.edge_batch_indices,
            )
        ):
            edges_by_relation[relation_index].append(
                edge_index
            )
            edges_by_relation_and_graph[
                (relation_index, graph_index)
            ].append(edge_index)

        context.edges_by_relation = dict(
            edges_by_relation
        )
        context.edges_by_relation_and_graph = dict(
            edges_by_relation_and_graph
        )

        self._index_provenance(context)

        return not context.errors_present

    def _validate_cpu_tensors(
        self,
        context: _ValidationContext,
    ) -> None:
        tensors: list[
            tuple[str, torch.Tensor],
        ] = []

        data = context.data

        for name, value in (
            ("edge_index", data.edge_index),
            (
                "edge_relation_index",
                data.edge_relation_index,
            ),
            (
                "node_batch_index",
                data.node_batch_index,
            ),
            (
                "edge_stable_relation_id",
                data.edge_stable_relation_id,
            ),
            (
                "edge_batch_index",
                data.edge_batch_index,
            ),
            (
                "origin_time_by_graph",
                data.origin_time_by_graph,
            ),
            ("node_time", data.node_time),
        ):
            if isinstance(value, torch.Tensor):
                tensors.append((name, value))

        for name, column in data.attributes.items():
            if isinstance(column.values, torch.Tensor):
                tensors.append(
                    (f"attributes.{name}.values", column.values)
                )

            if column.missing_mask is not None:
                tensors.append(
                    (
                        f"attributes.{name}.missing_mask",
                        column.missing_mask,
                    )
                )

        for name, column in self._temporal_columns(
            data
        ).items():
            tensors.append(
                (f"{name}.values", column.values)
            )
            tensors.append(
                (
                    f"{name}.applicable_mask",
                    column.applicable_mask,
                )
            )

        non_cpu = [
            name
            for name, tensor in tensors
            if tensor.device.type != "cpu"
        ]

        if non_cpu:
            context.add_error(
                "non_cpu_validation_tensor",
                "Strict relation validation must run on CPU. "
                f"Non-CPU tensors: {non_cpu}."
            )

    def _validate_node_types(
        self,
        context: _ValidationContext,
    ) -> None:
        invalid_strings = [
            index
            for index, value
            in enumerate(context.data.node_type_names)
            if not isinstance(value, str) or not value
        ]

        if invalid_strings:
            context.add_error(
                "invalid_node_type_name",
                "Every node type must be a nonempty string.",
            )
            return

        unknown = sorted(
            set(context.data.node_type_names)
            - set(CANONICAL_NODE_TYPE_NAMES)
        )

        if unknown:
            context.add_error(
                "unknown_node_type",
                "node_type_names contains unknown canonical node types: "
                f"{unknown}."
            )

    def _validate_graph_membership(
        self,
        context: _ValidationContext,
    ) -> bool:
        graph_ids = context.node_batch_indices

        if any(graph_id < 0 for graph_id in graph_ids):
            context.add_error(
                "negative_node_batch_index",
                "node_batch_index cannot contain negative values.",
            )
            return False

        if not graph_ids:
            context.num_graphs = 0

            if (
                context.data.origin_time_by_graph is not None
                and context.data.origin_time_by_graph.numel() != 0
            ):
                context.add_error(
                    "origin_for_empty_batch",
                    "An empty node batch cannot declare graph origins."
                )
                return False

            return True

        observed = sorted(set(graph_ids))
        expected = list(
            range(max(observed) + 1)
        )

        if observed != expected:
            context.add_error(
                "noncontiguous_graph_indices",
                "node_batch_index graph IDs must be contiguous from "
                f"zero. Observed {observed}, expected {expected}."
            )
            return False

        context.num_graphs = len(expected)

        origin = context.data.origin_time_by_graph

        if (
            origin is not None
            and int(origin.shape[0]) != context.num_graphs
        ):
            context.add_error(
                "origin_graph_count_mismatch",
                "origin_time_by_graph must contain one value for every "
                "graph, including graphs containing only isolated nodes."
            )
            return False

        return True

    def _validate_edge_ids(
        self,
        context: _ValidationContext,
    ) -> None:
        edge_ids = context.data.edge_ids

        if not edge_ids:
            return

        if len(edge_ids) != context.num_edges:
            context.add_error(
                "edge_id_length",
                "edge_ids length must equal the number of edges.",
            )
            return

        invalid = [
            index
            for index, value in enumerate(edge_ids)
            if not isinstance(value, str) or not value
        ]

        if invalid:
            context.add_error(
                "invalid_edge_id",
                "Every edge ID must be a nonempty string.",
                edge_indices=invalid,
            )

        duplicates = sorted(
            value
            for value, count in Counter(edge_ids).items()
            if count > 1
        )

        if duplicates:
            context.add_error(
                "duplicate_edge_ids",
                f"edge_ids contains duplicates: {duplicates}.",
            )

    def _validate_integer_vector(
        self,
        context: _ValidationContext,
        name: str,
        tensor: Any,
        *,
        expected_length: int | None,
    ) -> bool:
        if not isinstance(tensor, torch.Tensor):
            context.add_error(
                f"{name}_type",
                f"{name} must be a tensor.",
            )
            return False

        if tensor.ndim != 1:
            context.add_error(
                f"{name}_shape",
                f"{name} must be one-dimensional.",
            )
            return False

        if not _is_integer_tensor(tensor):
            context.add_error(
                f"{name}_dtype",
                f"{name} must use an integer dtype.",
            )
            return False

        if (
            expected_length is not None
            and int(tensor.shape[0]) != expected_length
        ):
            context.add_error(
                f"{name}_length",
                f"{name} has length {int(tensor.shape[0])}; "
                f"expected {expected_length}.",
            )
            return False

        return True

    def _validate_numeric_vector(
        self,
        context: _ValidationContext,
        name: str,
        tensor: Any,
        *,
        expected_length: int | None,
    ) -> bool:
        if not isinstance(tensor, torch.Tensor):
            context.add_error(
                f"{name}_type",
                f"{name} must be a tensor.",
            )
            return False

        if tensor.ndim != 1:
            context.add_error(
                f"{name}_shape",
                f"{name} must be one-dimensional.",
            )
            return False

        if not _is_numeric_tensor(tensor):
            context.add_error(
                f"{name}_dtype",
                f"{name} must use a numeric dtype.",
            )
            return False

        if (
            expected_length is not None
            and int(tensor.shape[0]) != expected_length
        ):
            context.add_error(
                f"{name}_length",
                f"{name} has length {int(tensor.shape[0])}; "
                f"expected {expected_length}.",
            )
            return False

        if (
            tensor.dtype.is_floating_point
            and not bool(
                torch.isfinite(tensor).all().item()
            )
        ):
            context.add_error(
                f"{name}_nonfinite",
                f"{name} cannot contain NaN or infinity.",
            )
            return False

        return True

    def _index_provenance(
        self,
        context: _ValidationContext,
    ) -> None:
        construction_index: dict[
            tuple[str, int],
            RelationConstructionProvenance,
        ] = {}

        for provenance in (
            context.data.construction_provenance
        ):
            if not isinstance(
                provenance,
                RelationConstructionProvenance,
            ):
                context.add_error(
                    "construction_provenance_type",
                    "construction_provenance must contain "
                    "RelationConstructionProvenance objects."
                )
                continue

            key = (
                provenance.relation_name,
                provenance.graph_index,
            )

            if key in construction_index:
                context.add_error(
                    "duplicate_construction_provenance",
                    f"Duplicate construction provenance for {key}.",
                    relation_name=provenance.relation_name,
                    graph_index=provenance.graph_index,
                )
            else:
                construction_index[key] = provenance

        control_index: dict[
            tuple[str, int],
            ControlGraphProvenance,
        ] = {}

        for provenance in context.data.control_provenance:
            if not isinstance(
                provenance,
                ControlGraphProvenance,
            ):
                context.add_error(
                    "control_provenance_type",
                    "control_provenance must contain "
                    "ControlGraphProvenance objects."
                )
                continue

            key = (
                provenance.relation_name,
                provenance.graph_index,
            )

            if key in control_index:
                context.add_error(
                    "duplicate_control_provenance",
                    f"Duplicate control provenance for {key}.",
                    relation_name=provenance.relation_name,
                    graph_index=provenance.graph_index,
                )
            else:
                control_index[key] = provenance

        context.construction_provenance_by_key = (
            construction_index
        )
        context.control_provenance_by_key = control_index

    # ------------------------------------------------------------------
    # Endpoint validation
    # ------------------------------------------------------------------

    def _validate_endpoint_contracts(
        self,
        context: _ValidationContext,
    ) -> None:
        invalid_pairs: dict[str, list[int]] = defaultdict(
            list
        )
        invalid_self_loops: dict[
            str,
            list[int],
        ] = defaultdict(list)

        for edge_index, (
            source,
            target,
            relation_index,
        ) in enumerate(
            zip(
                context.sources,
                context.targets,
                context.relation_indices,
            )
        ):
            entry = context.registry.entry_for_index(
                relation_index
            )

            source_type = (
                context.data.node_type_names[source]
            )
            target_type = (
                context.data.node_type_names[target]
            )

            if not entry.permits_endpoint_pair(
                source_type,
                target_type,
            ):
                invalid_pairs[entry.name].append(
                    edge_index
                )

            if (
                source == target
                and not entry.specification.allows_self_loops
            ):
                invalid_self_loops[entry.name].append(
                    edge_index
                )

        for relation_name, edge_indices in (
            invalid_pairs.items()
        ):
            context.add_error(
                "invalid_endpoint_pair",
                f"Relation {relation_name!r} contains forbidden source/"
                "target node-type pairs.",
                relation_name=relation_name,
                edge_indices=edge_indices,
            )

        for relation_name, edge_indices in (
            invalid_self_loops.items()
        ):
            context.add_error(
                "forbidden_self_loop",
                f"Relation {relation_name!r} does not permit self-loops.",
                relation_name=relation_name,
                edge_indices=edge_indices,
            )

    # ------------------------------------------------------------------
    # Edge-attribute validation
    # ------------------------------------------------------------------

    def _validate_edge_attributes(
        self,
        context: _ValidationContext,
    ) -> None:
        all_attribute_names = set(
            context.data.attributes
        )

        for relation_index, edge_indices in (
            context.edges_by_relation.items()
        ):
            entry = context.registry.entry_for_index(
                relation_index
            )
            specification = entry.specification

            required = {
                attribute.name: attribute
                for attribute
                in specification.required_edge_attributes
            }
            optional = {
                attribute.name: attribute
                for attribute
                in specification.optional_edge_attributes
            }
            declared = {
                **required,
                **optional,
            }

            missing_columns = sorted(
                set(required) - all_attribute_names
            )

            if missing_columns:
                context.add_error(
                    "missing_required_attribute_column",
                    f"Relation {entry.name!r} requires missing columns: "
                    f"{missing_columns}.",
                    relation_name=entry.name,
                    edge_indices=edge_indices,
                )

            for attribute_name, attribute_spec in (
                declared.items()
            ):
                column = context.data.attributes.get(
                    attribute_name
                )

                if column is None:
                    continue

                self._validate_attribute_for_edges(
                    context,
                    relation_entry=entry,
                    attribute_spec=attribute_spec,
                    column=column,
                    edge_indices=edge_indices,
                    required=attribute_name in required,
                )

            if (
                context.options
                .reject_values_for_undeclared_attributes
            ):
                for attribute_name in sorted(
                    all_attribute_names - set(declared)
                ):
                    column = context.data.attributes[
                        attribute_name
                    ]

                    populated = [
                        edge_index
                        for edge_index in edge_indices
                        if not self._column_semantically_missing(
                            column,
                            edge_index,
                        )
                    ]

                    if populated:
                        context.add_error(
                            "undeclared_attribute_value",
                            f"Relation {entry.name!r} contains values for "
                            f"undeclared attribute {attribute_name!r}.",
                            relation_name=entry.name,
                            edge_indices=populated,
                        )

        self._validate_lag_attribute_alignment(context)

    def _validate_attribute_for_edges(
        self,
        context: _ValidationContext,
        *,
        relation_entry: RelationRegistryEntry,
        attribute_spec: EdgeAttributeSpec,
        column: EdgeAttributeColumn,
        edge_indices: Sequence[int],
        required: bool,
    ) -> None:
        invalid_values: list[int] = []
        forbidden_missing: list[int] = []
        mask_errors: list[int] = []

        if (
            context.data.attribute_representation
            == AttributeRepresentation.MODEL
            and attribute_spec.requires_missingness_mask
            and column.missing_mask is None
        ):
            context.add_error(
                "missing_required_missingness_mask",
                f"Model-facing nullable attribute "
                f"{attribute_spec.name!r} requires a missingness mask.",
                relation_name=relation_entry.name,
                edge_indices=edge_indices,
            )

        for edge_index in edge_indices:
            value = column.value_at(edge_index)
            raw_missing = _value_is_missing(value)
            masked_missing = column.is_masked_missing(
                edge_index
            )

            if (
                context.data.attribute_representation
                == AttributeRepresentation.MODEL
            ):
                # Every stored imputed value must remain valid—even when the
                # corresponding missingness mask is true.
                if raw_missing or not self._value_matches_spec(
                    value,
                    attribute_spec,
                    tolerance=context.options.numeric_tolerance,
                ):
                    invalid_values.append(edge_index)

                if (
                    masked_missing
                    and not attribute_spec
                    .requires_missingness_mask
                ):
                    mask_errors.append(edge_index)

                continue

            # RAW representation.
            if raw_missing:
                if not attribute_spec.raw_values_may_be_missing:
                    forbidden_missing.append(edge_index)
                elif (
                    required
                    and not attribute_spec.raw_values_may_be_missing
                ):
                    forbidden_missing.append(edge_index)

                continue

            if not self._value_matches_spec(
                value,
                attribute_spec,
                tolerance=context.options.numeric_tolerance,
            ):
                invalid_values.append(edge_index)

        if forbidden_missing:
            context.add_error(
                "forbidden_attribute_missingness",
                f"Attribute {attribute_spec.name!r} contains forbidden "
                "raw missing values.",
                relation_name=relation_entry.name,
                edge_indices=forbidden_missing,
            )

        if invalid_values:
            context.add_error(
                "invalid_attribute_value",
                f"Attribute {attribute_spec.name!r} violates its type, "
                "finite-value, vocabulary, or numerical-bound contract.",
                relation_name=relation_entry.name,
                edge_indices=invalid_values,
            )

        if mask_errors:
            context.add_error(
                "unexpected_missingness_mask",
                f"Attribute {attribute_spec.name!r} uses a missingness "
                "mask although raw missingness is forbidden.",
                relation_name=relation_entry.name,
                edge_indices=mask_errors,
            )

    def _validate_lag_attribute_alignment(
        self,
        context: _ValidationContext,
    ) -> None:
        temporal_lag = context.data.edge_lag
        attribute_lag = context.data.attributes.get(
            TEMPORAL_FIELD_EDGE_LAG
        )

        if temporal_lag is None or attribute_lag is None:
            return

        mismatched: list[int] = []

        for edge_index, relation_index in enumerate(
            context.relation_indices
        ):
            specification = (
                context.registry.entry_for_index(
                    relation_index
                ).specification
            )

            if (
                TEMPORAL_FIELD_EDGE_LAG
                not in specification.required_temporal_fields
            ):
                continue

            if not temporal_lag.applies(edge_index):
                continue

            attribute_value = attribute_lag.value_at(
                edge_index
            )

            if _value_is_missing(attribute_value):
                mismatched.append(edge_index)
                continue

            number = _coerce_finite_number(
                attribute_value
            )

            if (
                number is None
                or abs(
                    number
                    - temporal_lag.value_at(edge_index)
                )
                > context.options.numeric_tolerance
            ):
                mismatched.append(edge_index)

        if mismatched:
            context.add_error(
                "edge_lag_alignment",
                "The temporal edge-lag column disagrees with the "
                "relation edge_lag attribute on lagged edges.",
                edge_indices=mismatched,
            )

    def _column_semantically_missing(
        self,
        column: EdgeAttributeColumn,
        edge_index: int,
    ) -> bool:
        return (
            column.is_masked_missing(edge_index)
            or _value_is_missing(
                column.value_at(edge_index)
            )
        )

    def _value_matches_spec(
        self,
        value: Any,
        specification: EdgeAttributeSpec,
        *,
        tolerance: float,
    ) -> bool:
        kind = specification.kind

        if kind == EdgeAttributeKind.FLOAT:
            number = _coerce_finite_number(value)

            return (
                number is not None
                and _within_bounds(
                    number,
                    specification,
                    tolerance=tolerance,
                )
            )

        if kind == EdgeAttributeKind.INTEGER:
            number = _coerce_finite_number(value)

            if number is None:
                return False

            if abs(number - round(number)) > tolerance:
                return False

            return _within_bounds(
                number,
                specification,
                tolerance=tolerance,
            )

        if kind == EdgeAttributeKind.BOOLEAN:
            if isinstance(value, bool):
                return True

            number = _coerce_finite_number(value)

            return (
                number is not None
                and (
                    abs(number) <= tolerance
                    or abs(number - 1.0) <= tolerance
                )
            )

        if kind == EdgeAttributeKind.CATEGORICAL:
            if not isinstance(value, str) or not value:
                return False

            if specification.categorical_closed_vocabulary:
                return value in specification.categorical_values

            return True

        if kind == EdgeAttributeKind.IDENTIFIER:
            if isinstance(value, bool):
                return False

            return (
                isinstance(value, int)
                or (
                    isinstance(value, str)
                    and bool(value)
                )
            )

        if kind == EdgeAttributeKind.TIMESTAMP:
            # Calendar strings are intentionally rejected.
            return _coerce_finite_number(value) is not None

        raise RuntimeError(
            f"Unhandled edge attribute kind {kind!r}."
        )

    # ------------------------------------------------------------------
    # Temporal validation
    # ------------------------------------------------------------------

    def _temporal_columns(
        self,
        data: RelationEdgeData,
    ) -> Mapping[str, TemporalEdgeColumn]:
        columns: dict[str, TemporalEdgeColumn] = {}

        for field_name, column in (
            (
                TEMPORAL_FIELD_EDGE_OBSERVATION_TIME,
                data.edge_observation_time,
            ),
            (
                TEMPORAL_FIELD_EDGE_VALID_FROM,
                data.edge_valid_from,
            ),
            (
                TEMPORAL_FIELD_EDGE_VALID_TO,
                data.edge_valid_to,
            ),
            (
                TEMPORAL_FIELD_EDGE_LAG,
                data.edge_lag,
            ),
        ):
            if column is not None:
                columns[field_name] = column

        return MappingProxyType(columns)

    def _validate_temporal_contracts(
        self,
        context: _ValidationContext,
    ) -> None:
        columns = self._temporal_columns(
            context.data
        )

        for relation_index, edge_indices in (
            context.edges_by_relation.items()
        ):
            entry = context.registry.entry_for_index(
                relation_index
            )
            specification = entry.specification

            for field_name in (
                specification.required_temporal_fields
            ):
                column = columns.get(field_name)

                if column is None:
                    context.add_error(
                        "missing_temporal_column",
                        f"Relation {entry.name!r} requires temporal "
                        f"column {field_name!r}.",
                        relation_name=entry.name,
                        edge_indices=edge_indices,
                    )
                    continue

                missing_applicability = [
                    edge_index
                    for edge_index in edge_indices
                    if not column.applies(edge_index)
                ]

                if missing_applicability:
                    context.add_error(
                        "missing_temporal_applicability",
                        f"Temporal column {field_name!r} is not marked "
                        f"applicable for required {entry.name!r} edges.",
                        relation_name=entry.name,
                        edge_indices=missing_applicability,
                    )

            requires_origin = (
                specification.temporal_mode
                != RelationTemporalMode.STATIC
                or specification.requires_as_of_time
            )

            if (
                requires_origin
                and context.data.origin_time_by_graph is None
            ):
                context.add_error(
                    "missing_prediction_origin",
                    f"Relation {entry.name!r} requires graph prediction "
                    "origins.",
                    relation_name=entry.name,
                    edge_indices=edge_indices,
                )
                continue

            if (
                specification.temporal_mode
                == RelationTemporalMode.SNAPSHOT
            ):
                self._validate_snapshot_relation(
                    context,
                    entry,
                    edge_indices,
                )

            elif (
                specification.temporal_mode
                == RelationTemporalMode.INTERVAL_VALID
            ):
                self._validate_interval_relation(
                    context,
                    entry,
                    edge_indices,
                )

            elif (
                specification.temporal_mode
                == RelationTemporalMode.LAGGED
            ):
                self._validate_lagged_relation(
                    context,
                    entry,
                    edge_indices,
                )

    def _validate_snapshot_relation(
        self,
        context: _ValidationContext,
        entry: RelationRegistryEntry,
        edge_indices: Sequence[int],
    ) -> None:
        column = context.data.edge_observation_time

        if column is None:
            return

        leaked = [
            edge_index
            for edge_index in edge_indices
            if (
                column.applies(edge_index)
                and column.value_at(edge_index)
                > self._edge_origin(context, edge_index)
            )
        ]

        if leaked:
            context.add_error(
                "snapshot_after_origin",
                f"Snapshot relation {entry.name!r} contains observations "
                "after the prediction origin.",
                relation_name=entry.name,
                edge_indices=leaked,
            )

    def _validate_interval_relation(
        self,
        context: _ValidationContext,
        entry: RelationRegistryEntry,
        edge_indices: Sequence[int],
    ) -> None:
        valid_from = context.data.edge_valid_from
        valid_to = context.data.edge_valid_to

        if valid_from is None or valid_to is None:
            return

        invalid_intervals: list[int] = []
        inactive: list[int] = []

        for edge_index in edge_indices:
            if not (
                valid_from.applies(edge_index)
                and valid_to.applies(edge_index)
            ):
                continue

            start = valid_from.value_at(edge_index)
            end = valid_to.value_at(edge_index)
            origin = self._edge_origin(
                context,
                edge_index,
            )

            if start > end:
                invalid_intervals.append(edge_index)
                continue

            if context.options.validity_end_inclusive:
                active = start <= origin <= end
            else:
                active = start <= origin < end

            if not active:
                inactive.append(edge_index)

        if invalid_intervals:
            context.add_error(
                "invalid_validity_interval",
                f"Relation {entry.name!r} contains valid_from values "
                "later than valid_to.",
                relation_name=entry.name,
                edge_indices=invalid_intervals,
            )

        if inactive:
            context.add_error(
                "edge_inactive_at_origin",
                f"Relation {entry.name!r} contains edges that were not "
                "active at prediction time.",
                relation_name=entry.name,
                edge_indices=inactive,
            )

    def _validate_lagged_relation(
        self,
        context: _ValidationContext,
        entry: RelationRegistryEntry,
        edge_indices: Sequence[int],
    ) -> None:
        observation = context.data.edge_observation_time
        lag_column = context.data.edge_lag

        if observation is None or lag_column is None:
            return

        invalid_lags: list[int] = []
        observation_leakage: list[int] = []
        nonforward_edges: list[int] = []
        lag_mismatches: list[int] = []
        future_node_states: list[int] = []

        for edge_index in edge_indices:
            if not (
                observation.applies(edge_index)
                and lag_column.applies(edge_index)
            ):
                continue

            lag = lag_column.value_at(edge_index)
            origin = self._edge_origin(
                context,
                edge_index,
            )

            if lag <= 0:
                invalid_lags.append(edge_index)

            if observation.value_at(edge_index) > origin:
                observation_leakage.append(edge_index)

            if context.data.node_time is None:
                continue

            source_time = _tensor_scalar(
                context.data.node_time,
                context.sources[edge_index],
            )
            target_time = _tensor_scalar(
                context.data.node_time,
                context.targets[edge_index],
            )

            if source_time >= target_time:
                nonforward_edges.append(edge_index)

            if (
                abs(
                    (target_time - source_time) - lag
                )
                > context.options.numeric_tolerance
            ):
                lag_mismatches.append(edge_index)

            if (
                source_time > origin
                or target_time > origin
            ):
                future_node_states.append(edge_index)

        if invalid_lags:
            context.add_error(
                "invalid_edge_lag",
                f"Lagged relation {entry.name!r} requires positive "
                "edge lags.",
                relation_name=entry.name,
                edge_indices=invalid_lags,
            )

        if observation_leakage:
            context.add_error(
                "lagged_observation_after_origin",
                f"Lagged relation {entry.name!r} uses observations "
                "after prediction time.",
                relation_name=entry.name,
                edge_indices=observation_leakage,
            )

        if nonforward_edges:
            context.add_error(
                "nonforward_temporal_edge",
                f"Lagged relation {entry.name!r} must connect an earlier "
                "source state to a later target state.",
                relation_name=entry.name,
                edge_indices=nonforward_edges,
            )

        if lag_mismatches:
            context.add_error(
                "node_time_lag_mismatch",
                f"Lagged relation {entry.name!r} has lag values "
                "inconsistent with source and target node times.",
                relation_name=entry.name,
                edge_indices=lag_mismatches,
            )

        if future_node_states:
            context.add_error(
                "future_node_state",
                f"Lagged relation {entry.name!r} contains source or "
                "target states later than the prediction origin.",
                relation_name=entry.name,
                edge_indices=future_node_states,
            )

    def _edge_origin(
        self,
        context: _ValidationContext,
        edge_index: int,
    ) -> float:
        origin = context.data.origin_time_by_graph

        if origin is None:
            raise RuntimeError(
                "Prediction origin requested after missing-origin "
                "validation."
            )

        graph_index = context.edge_batch_indices[
            edge_index
        ]

        return _tensor_scalar(origin, graph_index)

    # ------------------------------------------------------------------
    # Construction provenance
    # ------------------------------------------------------------------

    def _validate_construction_provenance(
        self,
        context: _ValidationContext,
    ) -> None:
        expected_keys: set[tuple[str, int]] = set()

        for (
            relation_index,
            graph_index,
        ), edge_indices in (
            context.edges_by_relation_and_graph.items()
        ):
            entry = context.registry.entry_for_index(
                relation_index
            )
            specification = entry.specification

            if not (
                specification.requires_training_fit
                or specification.requires_as_of_time
            ):
                continue

            key = (entry.name, graph_index)
            expected_keys.add(key)

            provenance = (
                context.construction_provenance_by_key.get(
                    key
                )
            )

            if provenance is None:
                context.add_error(
                    "missing_construction_provenance",
                    f"Relation {entry.name!r} requires graph-scoped "
                    "construction provenance.",
                    relation_name=entry.name,
                    graph_index=graph_index,
                    edge_indices=edge_indices,
                )
                continue

            if graph_index >= context.num_graphs:
                context.add_error(
                    "provenance_graph_bounds",
                    "Construction provenance references an unknown "
                    "graph index.",
                    relation_name=entry.name,
                    graph_index=graph_index,
                )
                continue

            origin = self._graph_origin(
                context,
                graph_index,
            )

            if specification.requires_as_of_time:
                as_of = provenance.construction_as_of_time

                if as_of is None:
                    context.add_error(
                        "missing_construction_as_of_time",
                        f"Relation {entry.name!r} requires "
                        "construction_as_of_time.",
                        relation_name=entry.name,
                        graph_index=graph_index,
                        edge_indices=edge_indices,
                    )
                elif as_of > origin:
                    context.add_error(
                        "construction_after_origin",
                        f"Relation {entry.name!r} was constructed using "
                        "information later than prediction time.",
                        relation_name=entry.name,
                        graph_index=graph_index,
                        edge_indices=edge_indices,
                    )

            if specification.requires_training_fit:
                self._validate_training_fit_provenance(
                    context,
                    entry,
                    provenance,
                    graph_index,
                    origin,
                    edge_indices,
                )

            self._validate_expected_source_fingerprint(
                context,
                entry,
                provenance,
                graph_index,
                edge_indices,
            )

            if (
                specification.leakage_risk
                in {
                    RelationLeakageRisk.MODERATE,
                    RelationLeakageRisk.HIGH,
                }
                and not provenance.builder_version
            ):
                context.add_warning(
                    "missing_builder_version",
                    f"Leakage-sensitive relation {entry.name!r} lacks "
                    "builder-version provenance.",
                    relation_name=entry.name,
                    graph_index=graph_index,
                    edge_indices=edge_indices,
                )

        unused = sorted(
            set(context.construction_provenance_by_key)
            - expected_keys
        )

        if unused:
            context.add_warning(
                "unused_construction_provenance",
                "Construction provenance was supplied for relation/"
                f"graph pairs not requiring it: {unused}."
            )

    def _validate_training_fit_provenance(
        self,
        context: _ValidationContext,
        entry: RelationRegistryEntry,
        provenance: RelationConstructionProvenance,
        graph_index: int,
        origin: float,
        edge_indices: Sequence[int],
    ) -> None:
        fingerprint = (
            provenance.training_split_fingerprint
        )
        cutoff = provenance.training_fit_cutoff

        if not fingerprint:
            context.add_error(
                "missing_training_split_fingerprint",
                f"Training-fitted relation {entry.name!r} requires a "
                "training-split fingerprint.",
                relation_name=entry.name,
                graph_index=graph_index,
                edge_indices=edge_indices,
            )

        expected = (
            context.expectations
            .expected_training_split_fingerprint
        )

        if (
            context.options.expected_provenance_required
            and expected is None
        ):
            context.add_error(
                "missing_expected_training_split_fingerprint",
                "Publication validation requires an expected training-"
                "split fingerprint."
            )

        if (
            expected is not None
            and fingerprint != expected
        ):
            context.add_error(
                "training_split_fingerprint_mismatch",
                f"Training-fitted relation {entry.name!r} was built "
                "from an unexpected training split.",
                relation_name=entry.name,
                graph_index=graph_index,
                edge_indices=edge_indices,
            )

        if cutoff is None:
            context.add_error(
                "missing_training_fit_cutoff",
                f"Training-fitted relation {entry.name!r} requires a "
                "training-fit cutoff.",
                relation_name=entry.name,
                graph_index=graph_index,
                edge_indices=edge_indices,
            )
        elif cutoff > origin:
            context.add_error(
                "training_fit_after_origin",
                f"Training-fitted relation {entry.name!r} uses a cutoff "
                "later than prediction time.",
                relation_name=entry.name,
                graph_index=graph_index,
                edge_indices=edge_indices,
            )

    def _validate_expected_source_fingerprint(
        self,
        context: _ValidationContext,
        entry: RelationRegistryEntry,
        provenance: RelationConstructionProvenance,
        graph_index: int,
        edge_indices: Sequence[int],
    ) -> None:
        expected = (
            context.expectations
            .expected_source_artifact_fingerprint
        )

        if (
            context.options.expected_provenance_required
            and expected is None
        ):
            context.add_error(
                "missing_expected_source_artifact_fingerprint",
                "Publication validation requires an expected source-"
                "artifact fingerprint."
            )
            return

        if (
            expected is not None
            and provenance.source_artifact_fingerprint
            != expected
        ):
            context.add_error(
                "source_artifact_fingerprint_mismatch",
                f"Relation {entry.name!r} was constructed from an "
                "unexpected source artifact.",
                relation_name=entry.name,
                graph_index=graph_index,
                edge_indices=edge_indices,
            )

    def _graph_origin(
        self,
        context: _ValidationContext,
        graph_index: int,
    ) -> float:
        origin = context.data.origin_time_by_graph

        if origin is None:
            raise RuntimeError(
                "Graph origin requested after missing-origin validation."
            )

        return _tensor_scalar(origin, graph_index)

    # ------------------------------------------------------------------
    # Control provenance
    # ------------------------------------------------------------------

    def _validate_control_provenance(
        self,
        context: _ValidationContext,
    ) -> None:
        expected_control_keys: set[
            tuple[str, int]
        ] = set()

        for (
            relation_index,
            graph_index,
        ), edge_indices in (
            context.edges_by_relation_and_graph.items()
        ):
            entry = context.registry.entry_for_index(
                relation_index
            )

            if not entry.specification.is_control:
                continue

            key = (entry.name, graph_index)
            expected_control_keys.add(key)

            provenance = (
                context.control_provenance_by_key.get(
                    key
                )
            )

            if provenance is None:
                context.add_error(
                    "missing_control_provenance",
                    f"Control relation {entry.name!r} requires "
                    "reproducible graph-construction provenance.",
                    relation_name=entry.name,
                    graph_index=graph_index,
                    edge_indices=edge_indices,
                )
                continue

            expected_source = (
                context.expectations
                .expected_source_artifact_fingerprint
            )

            if (
                expected_source is not None
                and provenance.source_graph_fingerprint
                != expected_source
            ):
                context.add_error(
                    "control_source_fingerprint_mismatch",
                    f"Control relation {entry.name!r} was generated from "
                    "an unexpected source graph.",
                    relation_name=entry.name,
                    graph_index=graph_index,
                    edge_indices=edge_indices,
                )

            if (
                context.options.expected_provenance_required
                and expected_source is None
            ):
                context.add_error(
                    "missing_expected_control_source_fingerprint",
                    "Publication validation requires the expected source "
                    "graph fingerprint for controls."
                )

            if not provenance.preserves_node_types:
                context.add_error(
                    "control_node_type_not_preserved",
                    f"Control relation {entry.name!r} must preserve node "
                    "types.",
                    relation_name=entry.name,
                    graph_index=graph_index,
                    edge_indices=edge_indices,
                )

        unused = sorted(
            set(context.control_provenance_by_key)
            - expected_control_keys
        )

        if unused:
            context.add_warning(
                "unused_control_provenance",
                "Control provenance was supplied for unused relation/"
                f"graph pairs: {unused}."
            )

    # ------------------------------------------------------------------
    # Duplicate, reciprocal, and coverage diagnostics
    # ------------------------------------------------------------------

    def _validate_duplicates(
        self,
        context: _ValidationContext,
    ) -> None:
        if not context.options.reject_duplicate_edges:
            return

        identities = [
            (
                context.edge_batch_indices[edge_index],
                context.sources[edge_index],
                context.targets[edge_index],
                context.relation_indices[edge_index],
            )
            for edge_index in range(context.num_edges)
        ]

        counts = Counter(identities)
        duplicated = {
            identity
            for identity, count in counts.items()
            if count > 1
        }

        duplicate_edges = [
            edge_index
            for edge_index, identity
            in enumerate(identities)
            if identity in duplicated
        ]

        if duplicate_edges:
            context.add_error(
                "duplicate_relation_edges",
                "Duplicate edges share graph, source, target, and "
                "relation identity.",
                edge_indices=duplicate_edges,
            )

    def _validate_undirected_storage(
        self,
        context: _ValidationContext,
    ) -> None:
        if not (
            context.options
            .require_reciprocal_undirected_storage
            or context.options
            .warn_on_nonreciprocal_undirected_storage
        ):
            return

        edge_keys = {
            (
                context.edge_batch_indices[edge_index],
                context.sources[edge_index],
                context.targets[edge_index],
                context.relation_indices[edge_index],
            )
            for edge_index in range(context.num_edges)
        }

        missing_by_relation: dict[
            str,
            list[int],
        ] = defaultdict(list)

        for edge_index in range(context.num_edges):
            relation_index = (
                context.relation_indices[edge_index]
            )
            entry = context.registry.entry_for_index(
                relation_index
            )

            if (
                entry.specification.direction
                != RelationDirection.UNDIRECTED
            ):
                continue

            reverse_key = (
                context.edge_batch_indices[edge_index],
                context.targets[edge_index],
                context.sources[edge_index],
                relation_index,
            )

            if reverse_key not in edge_keys:
                missing_by_relation[entry.name].append(
                    edge_index
                )

        for relation_name, edge_indices in (
            missing_by_relation.items()
        ):
            message = (
                f"Undirected relation {relation_name!r} is not stored "
                "with reciprocal arcs for every edge."
            )

            if (
                context.options
                .require_reciprocal_undirected_storage
            ):
                context.add_error(
                    "missing_undirected_reciprocal",
                    message,
                    relation_name=relation_name,
                    edge_indices=edge_indices,
                )
            else:
                context.add_warning(
                    "missing_undirected_reciprocal",
                    message,
                    relation_name=relation_name,
                    edge_indices=edge_indices,
                )

    def _validate_relation_coverage(
        self,
        context: _ValidationContext,
    ) -> None:
        observed = set(context.edges_by_relation)
        expected = set(range(len(context.registry)))
        missing = sorted(expected - observed)

        if not missing:
            return

        missing_names = [
            context.registry.entry_for_index(
                relation_index
            ).name
            for relation_index in missing
        ]

        message = (
            "Compiled relations have no edges in this payload: "
            f"{missing_names}."
        )

        if (
            context.options
            .require_all_compiled_relations_present
        ):
            context.add_error(
                "missing_compiled_relation_edges",
                message,
            )
        elif (
            context.options
            .warn_on_missing_compiled_relations
        ):
            context.add_warning(
                "missing_compiled_relation_edges",
                message,
            )


# Add a compact property after the context class definition without exposing
# mutable issue internals.
def _context_errors_present(
    context: _ValidationContext,
) -> bool:
    return any(
        issue.severity == ValidationSeverity.ERROR
        for issue in context.issues
    )


_ValidationContext.errors_present = property(
    _context_errors_present
)


# =============================================================================
# Convenience API
# =============================================================================


def validate_relation_edge_data(
    data: RelationEdgeData,
    registry: CompiledRelationRegistry,
    *,
    source_registry: RelationRegistry | None = None,
    options: RelationValidationOptions | None = None,
    expectations: RelationValidationExpectations | None = None,
) -> RelationValidationReport:
    """Validate graph relation data and return the complete report."""

    return RelationValidator(
        options=options,
        expectations=expectations,
    ).validate(
        data,
        registry,
        source_registry=source_registry,
    )


def assert_valid_relation_edge_data(
    data: RelationEdgeData,
    registry: CompiledRelationRegistry,
    *,
    source_registry: RelationRegistry | None = None,
    options: RelationValidationOptions | None = None,
    expectations: RelationValidationExpectations | None = None,
) -> RelationValidationReport:
    """Validate graph relation data and raise on report errors."""

    report = validate_relation_edge_data(
        data,
        registry,
        source_registry=source_registry,
        options=options,
        expectations=expectations,
    )
    report.raise_for_errors()
    return report


__all__ = (
    "AttributeRepresentation",
    "CONTROL_PROVENANCE_SCHEMA_VERSION",
    "ControlGraphKind",
    "ControlGraphProvenance",
    "EdgeAttributeColumn",
    "RELATION_PROVENANCE_SCHEMA_VERSION",
    "RELATION_VALIDATION_SCHEMA_VERSION",
    "RelationConstructionProvenance",
    "RelationEdgeData",
    "RelationValidationError",
    "RelationValidationExpectations",
    "RelationValidationOptions",
    "RelationValidationProfile",
    "RelationValidationReport",
    "RelationValidator",
    "TemporalEdgeColumn",
    "TimeEncoding",
    "ValidationIssue",
    "ValidationSeverity",
    "assert_valid_relation_edge_data",
    "validate_relation_edge_data",
)