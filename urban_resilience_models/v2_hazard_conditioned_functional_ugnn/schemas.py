"""
Validated schemas for the V2 hazard-conditioned functional UGNN.

Target path:
    urban_resilience_models/
        v2_hazard_conditioned_functional_ugnn/
            schemas.py

This module defines stable structural contracts shared by:

- data adapters and graph loaders;
- batch collators;
- memory and hazard encoders;
- functional message-passing modules;
- prediction and uncertainty heads;
- explanation exporters;
- training and inference code.

The schemas enforce:

- tensor shapes and dtypes;
- graph membership;
- node and edge identity alignment;
- optional supervision semantics;
- temporal causality and feature availability;
- output provenance;
- schema and registry-version compatibility.

Registry membership itself is intentionally validated outside this module by
the hazard registry, relation registry, node-type registry, graph loaders, and
adapter-level validation utilities. This avoids circular imports while keeping
the schema layer reusable.

Mutation policy
---------------
Schema fields should not be mutated directly after construction.

Use validated methods such as:

    batch.replace(...)
    batch.to(...)

which reconstruct and revalidate the schema object.

This module does not implement model behavior.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace as dataclass_replace
from datetime import date, datetime, time
from typing import Any, Iterable, Mapping, Sequence

import torch
from torch import Tensor

from .constants import (
    ATTENTION_HEAD_REDUCTION_NONE,
    ATTENTION_NORMALIZATION_GLOBAL_RELATION,
    ATTENTION_NORMALIZATION_TARGET_NODE,
    ATTENTION_NORMALIZATION_TARGET_NODE_RELATION,
    ATTENTION_NORMALIZATION_UNNORMALIZED_SIGMOID,
    BATCH_SCHEMA_VERSION,
    CANONICAL_ATTENTION_HEAD_REDUCTIONS,
    CANONICAL_ATTENTION_NORMALIZATION_MODES,
    CANONICAL_RELATION_GATE_SCOPES,
    CANONICAL_SCOPES,
    CANONICAL_SPLITS,
    EXPLANATION_SCHEMA_VERSION,
    FEATURE_CONTRACT_VERSION,
    HAZARD_REGISTRY_VERSION,
    MODEL_CONFIG_VERSION,
    MODEL_FAMILY_ID,
    MODEL_FAMILY_VERSION,
    PREDICTION_SCHEMA_VERSION,
    RELATION_GATE_SCOPE_GRAPH,
    RELATION_GATE_SCOPE_SOURCE_NODE,
    RELATION_GATE_SCOPE_SOURCE_TARGET,
    RELATION_GATE_SCOPE_TARGET_NODE,
    RELATION_REGISTRY_VERSION,
    SCOPE_GRAPH,
    SCOPE_NODE,
)


TimePoint = datetime | date
NullableTimePoint = TimePoint | None
TensorMapping = Mapping[str, Tensor]


_INTEGER_DTYPES = frozenset(
    {
        torch.int8,
        torch.int16,
        torch.int32,
        torch.int64,
        torch.uint8,
    }
)


# =============================================================================
# Generic validation helpers
# =============================================================================


def _require_nonempty_string(name: str, value: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string.")


def _require_choice(
    name: str,
    value: str,
    allowed: Sequence[str],
) -> None:
    if value not in allowed:
        raise ValueError(
            f"Unknown {name} {value!r}. Expected one of {tuple(allowed)}."
        )


def _require_tensor(name: str, value: Tensor) -> None:
    if not isinstance(value, Tensor):
        raise TypeError(
            f"{name} must be a torch.Tensor, "
            f"got {type(value).__name__}."
        )


def _require_ndim(name: str, value: Tensor, ndim: int) -> None:
    _require_tensor(name, value)

    if value.ndim != ndim:
        raise ValueError(
            f"{name} must have rank {ndim}, "
            f"got shape {tuple(value.shape)}."
        )


def _require_integer_tensor(
    name: str,
    value: Tensor,
    *,
    ndim: int | None = None,
) -> None:
    _require_tensor(name, value)

    if ndim is not None and value.ndim != ndim:
        raise ValueError(
            f"{name} must have rank {ndim}, "
            f"got shape {tuple(value.shape)}."
        )

    if value.dtype not in _INTEGER_DTYPES:
        raise TypeError(
            f"{name} must use an integer dtype, got {value.dtype}."
        )


def _require_long_tensor(
    name: str,
    value: Tensor,
    *,
    ndim: int | None = None,
) -> None:
    _require_tensor(name, value)

    if ndim is not None and value.ndim != ndim:
        raise ValueError(
            f"{name} must have rank {ndim}, "
            f"got shape {tuple(value.shape)}."
        )

    if value.dtype != torch.long:
        raise TypeError(
            f"{name} must use torch.long, got {value.dtype}."
        )


def _require_float_tensor(
    name: str,
    value: Tensor,
    *,
    ndim: int | None = None,
    finite: bool = True,
) -> None:
    _require_tensor(name, value)

    if ndim is not None and value.ndim != ndim:
        raise ValueError(
            f"{name} must have rank {ndim}, "
            f"got shape {tuple(value.shape)}."
        )

    if not value.is_floating_point():
        raise TypeError(
            f"{name} must use a floating-point dtype, got {value.dtype}."
        )

    if finite and not bool(torch.isfinite(value).all()):
        raise ValueError(f"{name} contains NaN or infinite values.")


def _require_bool_tensor(
    name: str,
    value: Tensor,
    *,
    ndim: int | None = None,
) -> None:
    _require_tensor(name, value)

    if ndim is not None and value.ndim != ndim:
        raise ValueError(
            f"{name} must have rank {ndim}, "
            f"got shape {tuple(value.shape)}."
        )

    if value.dtype != torch.bool:
        raise TypeError(
            f"{name} must use torch.bool, got {value.dtype}."
        )


def _require_first_dimension(
    name: str,
    value: Tensor,
    expected: int,
) -> None:
    if value.shape[0] != expected:
        raise ValueError(
            f"{name} must have first dimension {expected}, "
            f"got shape {tuple(value.shape)}."
        )


def _require_same_shape(
    first_name: str,
    first: Tensor,
    second_name: str,
    second: Tensor,
) -> None:
    if first.shape != second.shape:
        raise ValueError(
            f"{first_name} and {second_name} must have identical shapes, "
            f"got {tuple(first.shape)} and {tuple(second.shape)}."
        )


def _require_nonnegative_tensor(name: str, value: Tensor) -> None:
    if value.numel() > 0 and bool((value < 0).any()):
        raise ValueError(
            f"{name} must contain only nonnegative values."
        )


def _require_unique_strings(
    name: str,
    values: Sequence[str],
) -> None:
    for index, value in enumerate(values):
        _require_nonempty_string(f"{name}[{index}]", value)

    duplicates = sorted(
        value
        for value in set(values)
        if values.count(value) > 1
    )

    if duplicates:
        raise ValueError(
            f"{name} contains duplicate values: {duplicates}."
        )


def _validate_same_device(
    tensors: Mapping[str, Tensor | None],
) -> None:
    devices = {
        tensor.device
        for tensor in tensors.values()
        if isinstance(tensor, Tensor)
    }

    if len(devices) > 1:
        details = ", ".join(
            f"{name}={tensor.device}"
            for name, tensor in tensors.items()
            if isinstance(tensor, Tensor)
        )
        raise ValueError(
            "All tensors in one schema object must be on the same device. "
            f"Observed: {details}."
        )


def _tensor_equal(first: Tensor, second: Tensor) -> bool:
    if first.shape != second.shape or first.dtype != second.dtype:
        return False

    return torch.equal(
        first.detach().cpu(),
        second.detach().cpu(),
    )


def _to_datetime(value: TimePoint, *, name: str) -> datetime:
    if isinstance(value, datetime):
        return value

    if isinstance(value, date):
        return datetime.combine(value, time.min)

    raise TypeError(
        f"{name} must be a datetime or date, "
        f"got {type(value).__name__}."
    )


def _validate_time_sequence(
    name: str,
    values: Sequence[TimePoint],
    expected_length: int,
) -> None:
    if len(values) != expected_length:
        raise ValueError(
            f"{name} must contain {expected_length} values, "
            f"got {len(values)}."
        )

    for index, value in enumerate(values):
        _to_datetime(value, name=f"{name}[{index}]")


def _validate_nullable_time_sequence(
    name: str,
    values: Sequence[NullableTimePoint],
    expected_length: int,
) -> None:
    if len(values) != expected_length:
        raise ValueError(
            f"{name} must contain {expected_length} values, "
            f"got {len(values)}."
        )

    for index, value in enumerate(values):
        if value is not None:
            _to_datetime(value, name=f"{name}[{index}]")


def _validate_time_matrix(
    name: str,
    values: Sequence[Sequence[TimePoint]],
    *,
    expected_rows: int,
    expected_columns: int,
) -> None:
    if len(values) != expected_rows:
        raise ValueError(
            f"{name} must have {expected_rows} graph rows, "
            f"got {len(values)}."
        )

    for row_index, row in enumerate(values):
        if len(row) != expected_columns:
            raise ValueError(
                f"{name}[{row_index}] must contain "
                f"{expected_columns} target values, got {len(row)}."
            )

        for column_index, value in enumerate(row):
            _to_datetime(
                value,
                name=f"{name}[{row_index}][{column_index}]",
            )


def _validate_regularization_terms(
    name: str,
    terms: TensorMapping,
) -> None:
    for term_name, term_value in terms.items():
        _require_nonempty_string(f"{name} key", term_name)
        _require_float_tensor(
            f"{name}[{term_name!r}]",
            term_value,
        )

        if term_value.numel() != 1:
            raise ValueError(
                f"{name}[{term_name!r}] must be scalar, "
                f"got shape {tuple(term_value.shape)}."
            )


def _validate_float_tensor_mapping(
    name: str,
    values: TensorMapping,
    *,
    first_dimension: int | None = None,
) -> None:
    for key, tensor in values.items():
        _require_nonempty_string(f"{name} key", key)
        _require_float_tensor(f"{name}[{key!r}]", tensor)

        if first_dimension is not None:
            _require_first_dimension(
                f"{name}[{key!r}]",
                tensor,
                first_dimension,
            )


def _move_tensor(
    value: Tensor | None,
    device: torch.device | str,
) -> Tensor | None:
    if value is None:
        return None

    return value.to(device)


# =============================================================================
# Contract and provenance schemas
# =============================================================================


@dataclass(slots=True, frozen=True)
class ContractVersions:
    """Versions required to interpret a model artifact safely."""

    model_family_version: str = MODEL_FAMILY_VERSION
    model_config_version: str = MODEL_CONFIG_VERSION
    batch_schema_version: str = BATCH_SCHEMA_VERSION
    relation_registry_version: str = RELATION_REGISTRY_VERSION
    hazard_registry_version: str = HAZARD_REGISTRY_VERSION
    feature_contract_version: str = FEATURE_CONTRACT_VERSION
    prediction_schema_version: str = PREDICTION_SCHEMA_VERSION
    explanation_schema_version: str = EXPLANATION_SCHEMA_VERSION

    @classmethod
    def current(cls) -> ContractVersions:
        return cls()

    def as_dict(self) -> dict[str, str]:
        return {
            "model_family_version": self.model_family_version,
            "model_config_version": self.model_config_version,
            "batch_schema_version": self.batch_schema_version,
            "relation_registry_version": self.relation_registry_version,
            "hazard_registry_version": self.hazard_registry_version,
            "feature_contract_version": self.feature_contract_version,
            "prediction_schema_version": self.prediction_schema_version,
            "explanation_schema_version": self.explanation_schema_version,
        }

    def validate(self, *, require_current: bool = False) -> None:
        observed = self.as_dict()

        for field_name, value in observed.items():
            _require_nonempty_string(field_name, value)

        if not require_current:
            return

        expected = ContractVersions.current().as_dict()

        mismatches = {
            key: (observed[key], expected[key])
            for key in expected
            if observed[key] != expected[key]
        }

        if mismatches:
            details = ", ".join(
                f"{key}: observed={actual!r}, expected={required!r}"
                for key, (actual, required) in mismatches.items()
            )
            raise ValueError(
                "Incompatible contract or registry versions. "
                f"{details}."
            )


@dataclass(slots=True, frozen=True)
class RunMetadata:
    """Run-level identity attached to predictions and explanations."""

    checkpoint_id: str
    run_id: str
    experiment_name: str
    random_seed: int

    dataset_version: str
    graph_version: str
    config_hash: str

    model_family_id: str = MODEL_FAMILY_ID

    def validate(self) -> None:
        _require_nonempty_string(
            "checkpoint_id",
            self.checkpoint_id,
        )
        _require_nonempty_string("run_id", self.run_id)
        _require_nonempty_string(
            "experiment_name",
            self.experiment_name,
        )
        _require_nonempty_string(
            "dataset_version",
            self.dataset_version,
        )
        _require_nonempty_string(
            "graph_version",
            self.graph_version,
        )
        _require_nonempty_string(
            "config_hash",
            self.config_hash,
        )

        if self.model_family_id != MODEL_FAMILY_ID:
            raise ValueError(
                "RunMetadata belongs to an incompatible model family. "
                f"Observed {self.model_family_id!r}, "
                f"expected {MODEL_FAMILY_ID!r}."
            )

        if self.random_seed < 0:
            raise ValueError("random_seed must be nonnegative.")


@dataclass(slots=True)
class BatchMetadata:
    """Typed dataset, split, and experiment metadata."""

    dataset_version: str
    graph_version: str

    split_by_graph: tuple[str, ...] = ()
    experiment_name: str | None = None
    extra_metadata: Mapping[str, Any] = field(default_factory=dict)

    def validate(self, batch_size: int) -> None:
        _require_nonempty_string(
            "dataset_version",
            self.dataset_version,
        )
        _require_nonempty_string(
            "graph_version",
            self.graph_version,
        )

        if self.experiment_name is not None:
            _require_nonempty_string(
                "experiment_name",
                self.experiment_name,
            )

        if self.split_by_graph:
            if len(self.split_by_graph) != batch_size:
                raise ValueError(
                    "split_by_graph must contain one split per graph. "
                    f"Expected {batch_size}, "
                    f"got {len(self.split_by_graph)}."
                )

            unknown = sorted(
                set(self.split_by_graph) - set(CANONICAL_SPLITS)
            )

            if unknown:
                raise ValueError(
                    "split_by_graph contains unknown values: "
                    f"{unknown}. Known splits: {CANONICAL_SPLITS}."
                )


# =============================================================================
# Temporal schemas
# =============================================================================


@dataclass(slots=True)
class TemporalMetadata:
    """
    Graph-level temporal causality and availability metadata.

    Target windows are represented by graph and output:

        target_start_time_by_graph_target[B][K]
        target_end_time_by_graph_target[B][K]

    This supports multi-horizon output from one graph instance.
    """

    origin_time: tuple[TimePoint, ...]
    history_start_time: tuple[TimePoint, ...]
    history_end_time: tuple[TimePoint, ...]
    feature_availability_cutoff: tuple[TimePoint, ...]

    history_time_points_by_graph: (
        tuple[tuple[TimePoint, ...], ...] | None
    ) = None

    target_start_time_by_graph_target: (
        tuple[tuple[TimePoint, ...], ...] | None
    ) = None
    target_end_time_by_graph_target: (
        tuple[tuple[TimePoint, ...], ...] | None
    ) = None

    edge_valid_from: tuple[NullableTimePoint, ...] | None = None
    edge_valid_to: tuple[NullableTimePoint, ...] | None = None
    edge_observation_time: (
        tuple[NullableTimePoint, ...] | None
    ) = None

    def validate(
        self,
        *,
        batch_size: int,
        num_targets: int,
        history_length: int | None,
        num_edges: int,
        supervised: bool,
    ) -> None:
        _validate_time_sequence(
            "origin_time",
            self.origin_time,
            batch_size,
        )
        _validate_time_sequence(
            "history_start_time",
            self.history_start_time,
            batch_size,
        )
        _validate_time_sequence(
            "history_end_time",
            self.history_end_time,
            batch_size,
        )
        _validate_time_sequence(
            "feature_availability_cutoff",
            self.feature_availability_cutoff,
            batch_size,
        )

        if self.history_time_points_by_graph is not None:
            if history_length is None:
                raise ValueError(
                    "history_time_points_by_graph cannot be provided "
                    "without history sequences."
                )

            if len(self.history_time_points_by_graph) != batch_size:
                raise ValueError(
                    "history_time_points_by_graph must contain one row "
                    "per graph."
                )

            for graph_index, time_points in enumerate(
                self.history_time_points_by_graph
            ):
                if len(time_points) != history_length:
                    raise ValueError(
                        "Each history-time row must match the padded "
                        f"history length {history_length}. "
                        f"Graph {graph_index} has {len(time_points)}."
                    )

                for time_index, value in enumerate(time_points):
                    _to_datetime(
                        value,
                        name=(
                            "history_time_points_by_graph"
                            f"[{graph_index}][{time_index}]"
                        ),
                    )

        starts_present = (
            self.target_start_time_by_graph_target is not None
        )
        ends_present = (
            self.target_end_time_by_graph_target is not None
        )

        if starts_present != ends_present:
            raise ValueError(
                "Target start and end matrices must either both be "
                "present or both be absent."
            )

        if supervised and not starts_present:
            raise ValueError(
                "Supervised batches require graph-by-target start and "
                "end windows."
            )

        if starts_present:
            if num_targets <= 0:
                raise ValueError(
                    "Target windows require at least one declared target."
                )

            _validate_time_matrix(
                "target_start_time_by_graph_target",
                self.target_start_time_by_graph_target,
                expected_rows=batch_size,
                expected_columns=num_targets,
            )
            _validate_time_matrix(
                "target_end_time_by_graph_target",
                self.target_end_time_by_graph_target,
                expected_rows=batch_size,
                expected_columns=num_targets,
            )

        for graph_index in range(batch_size):
            origin = _to_datetime(
                self.origin_time[graph_index],
                name=f"origin_time[{graph_index}]",
            )
            history_start = _to_datetime(
                self.history_start_time[graph_index],
                name=f"history_start_time[{graph_index}]",
            )
            history_end = _to_datetime(
                self.history_end_time[graph_index],
                name=f"history_end_time[{graph_index}]",
            )
            cutoff = _to_datetime(
                self.feature_availability_cutoff[graph_index],
                name=(
                    "feature_availability_cutoff"
                    f"[{graph_index}]"
                ),
            )

            try:
                if history_start > history_end:
                    raise ValueError(
                        f"Graph {graph_index}: history_start_time must "
                        "not exceed history_end_time."
                    )

                if history_end > origin:
                    raise ValueError(
                        f"Graph {graph_index}: history_end_time must "
                        "not exceed origin_time."
                    )

                if cutoff > origin:
                    raise ValueError(
                        f"Graph {graph_index}: "
                        "feature_availability_cutoff must not exceed "
                        "origin_time."
                    )

                if self.history_time_points_by_graph is not None:
                    for time_index, time_point in enumerate(
                        self.history_time_points_by_graph[graph_index]
                    ):
                        observed_time = _to_datetime(
                            time_point,
                            name=(
                                "history_time_points_by_graph"
                                f"[{graph_index}][{time_index}]"
                            ),
                        )

                        if observed_time > history_end:
                            raise ValueError(
                                f"Graph {graph_index}: history time point "
                                f"{time_index} occurs after "
                                "history_end_time."
                            )

                if starts_present:
                    for target_index in range(num_targets):
                        target_start = _to_datetime(
                            self.target_start_time_by_graph_target[
                                graph_index
                            ][target_index],
                            name=(
                                "target_start_time_by_graph_target"
                                f"[{graph_index}][{target_index}]"
                            ),
                        )
                        target_end = _to_datetime(
                            self.target_end_time_by_graph_target[
                                graph_index
                            ][target_index],
                            name=(
                                "target_end_time_by_graph_target"
                                f"[{graph_index}][{target_index}]"
                            ),
                        )

                        if target_start <= origin:
                            raise ValueError(
                                f"Graph {graph_index}, target "
                                f"{target_index}: target_start_time must "
                                "be strictly after origin_time."
                            )

                        if target_end < target_start:
                            raise ValueError(
                                f"Graph {graph_index}, target "
                                f"{target_index}: target_end_time must "
                                "not precede target_start_time."
                            )
            except TypeError as exc:
                raise ValueError(
                    "Temporal values within each graph must use "
                    "compatible timezone conventions."
                ) from exc

        self._validate_edge_array_lengths(num_edges)

    def _validate_edge_array_lengths(self, num_edges: int) -> None:
        if self.edge_valid_from is not None:
            _validate_nullable_time_sequence(
                "edge_valid_from",
                self.edge_valid_from,
                num_edges,
            )

        if self.edge_valid_to is not None:
            _validate_nullable_time_sequence(
                "edge_valid_to",
                self.edge_valid_to,
                num_edges,
            )

        if self.edge_observation_time is not None:
            _validate_nullable_time_sequence(
                "edge_observation_time",
                self.edge_observation_time,
                num_edges,
            )


# =============================================================================
# Typed feature schemas
# =============================================================================


@dataclass(slots=True)
class TypedNodeFeatureBlock:
    """
    Raw features belonging to one node type.

    The numeric node_type_id is checked against the packed node_type tensor.
    Registry membership for the ID/name pair is validated outside this module.
    """

    node_type_name: str
    node_type_id: int
    internal_node_index: Tensor
    features: Tensor
    feature_names: tuple[str, ...] = ()

    def validate(
        self,
        *,
        num_nodes: int,
        packed_node_type: Tensor,
    ) -> None:
        _require_nonempty_string(
            "node_type_name",
            self.node_type_name,
        )

        if self.node_type_id < 0:
            raise ValueError("node_type_id must be nonnegative.")

        _require_long_tensor(
            "internal_node_index",
            self.internal_node_index,
            ndim=1,
        )
        _require_float_tensor(
            "features",
            self.features,
            ndim=2,
        )
        _require_first_dimension(
            "features",
            self.features,
            self.internal_node_index.shape[0],
        )

        if self.internal_node_index.numel() > 0:
            minimum = int(self.internal_node_index.min().item())
            maximum = int(self.internal_node_index.max().item())

            if minimum < 0 or maximum >= num_nodes:
                raise ValueError(
                    "internal_node_index contains an out-of-range node "
                    f"index. Valid range is [0, {num_nodes - 1}]."
                )

            if (
                torch.unique(self.internal_node_index).numel()
                != self.internal_node_index.numel()
            ):
                raise ValueError(
                    "internal_node_index contains duplicate node indices."
                )

            observed_types = packed_node_type[
                self.internal_node_index
            ]

            if not bool(
                (observed_types == self.node_type_id).all()
            ):
                raise ValueError(
                    f"Feature block {self.node_type_name!r} with "
                    f"node_type_id={self.node_type_id} does not match "
                    "the packed node_type tensor."
                )

        if self.feature_names:
            _require_unique_strings(
                "feature_names",
                self.feature_names,
            )

            if len(self.feature_names) != self.features.shape[1]:
                raise ValueError(
                    "feature_names must match the feature width. "
                    f"Expected {self.features.shape[1]}, "
                    f"got {len(self.feature_names)}."
                )

        _validate_same_device(
            {
                "internal_node_index": self.internal_node_index,
                "features": self.features,
                "packed_node_type": packed_node_type,
            }
        )

    def to(
        self,
        device: torch.device | str,
    ) -> TypedNodeFeatureBlock:
        return dataclass_replace(
            self,
            internal_node_index=self.internal_node_index.to(device),
            features=self.features.to(device),
        )


# =============================================================================
# Canonical model-input batch
# =============================================================================


@dataclass(slots=True)
class UrbanGraphBatch:
    """
    Canonical packed graph input.

    Exactly one node-representation source must be provided:

    - node_features;
    - node_state;
    - node_feature_blocks.

    Targets are optional for inference and counterfactual analysis.
    """

    external_node_ids: tuple[str, ...]
    node_batch_index: Tensor

    hazard_ids: Tensor
    hazard_scope: str

    edge_index: Tensor
    edge_relation_type: Tensor

    temporal: TemporalMetadata
    contract_versions: ContractVersions
    metadata: BatchMetadata

    target_names: tuple[str, ...] = ()
    target_horizons: tuple[str, ...] = ()

    node_features: Tensor | None = None
    node_state: Tensor | None = None
    node_feature_blocks: tuple[TypedNodeFeatureBlock, ...] = ()
    node_type: Tensor | None = None

    history_sequences: Tensor | None = None
    history_mask: Tensor | None = None

    hazard_features: Tensor | None = None

    scenario_features: Tensor | None = None
    scenario_scope: str = SCOPE_GRAPH

    edge_attributes: Tensor | None = None
    semantic_edge_weight: Tensor | None = None

    external_edge_ids: tuple[str, ...] | None = None
    edge_batch_index: Tensor | None = None
    graph_ptr: Tensor | None = None

    targets: Tensor | None = None
    target_mask: Tensor | None = None

    feature_names: tuple[str, ...] = ()
    history_feature_names: tuple[str, ...] = ()
    edge_attribute_names: tuple[str, ...] = ()

    allow_cross_graph_edges: bool = False

    def __post_init__(self) -> None:
        self.validate()

    @property
    def num_nodes(self) -> int:
        return len(self.external_node_ids)

    @property
    def batch_size(self) -> int:
        return len(self.temporal.origin_time)

    @property
    def num_edges(self) -> int:
        # Safe after construction because validate() checks edge_index first.
        return int(self.edge_index.shape[1])

    @property
    def num_targets(self) -> int:
        return len(self.target_names)

    @property
    def is_supervised(self) -> bool:
        return self.targets is not None

    @property
    def source_index(self) -> Tensor:
        return self.edge_index[0]

    @property
    def target_index(self) -> Tensor:
        return self.edge_index[1]

    def validate(
        self,
        *,
        require_current_versions: bool = False,
    ) -> None:
        self.contract_versions.validate(
            require_current=require_current_versions
        )

        self._validate_edge_index_structure()

        num_nodes = self.num_nodes
        num_edges = self.num_edges
        batch_size = self.batch_size

        if num_nodes <= 0:
            raise ValueError(
                "UrbanGraphBatch must contain at least one node."
            )

        if batch_size <= 0:
            raise ValueError(
                "UrbanGraphBatch must contain at least one graph."
            )

        _require_unique_strings(
            "external_node_ids",
            self.external_node_ids,
        )

        self._validate_target_declarations()
        self._validate_graph_membership(
            num_nodes=num_nodes,
            batch_size=batch_size,
        )
        self._validate_node_representations(num_nodes=num_nodes)
        self._validate_history(num_nodes=num_nodes)
        self._validate_hazard_context(
            num_nodes=num_nodes,
            batch_size=batch_size,
        )
        self._validate_scenario_context(
            num_nodes=num_nodes,
            batch_size=batch_size,
        )
        self._validate_edges(
            num_nodes=num_nodes,
            num_edges=num_edges,
            batch_size=batch_size,
        )
        self._validate_supervision(num_nodes=num_nodes)

        history_length = (
            int(self.history_sequences.shape[1])
            if self.history_sequences is not None
            else None
        )

        self.temporal.validate(
            batch_size=batch_size,
            num_targets=self.num_targets,
            history_length=history_length,
            num_edges=num_edges,
            supervised=self.is_supervised,
        )
        self._validate_temporal_edges()

        self.metadata.validate(batch_size)

        self._validate_external_node_identity()
        self._validate_feature_name_alignment()
        self._validate_device_compatibility()

    def _validate_edge_index_structure(self) -> None:
        _require_long_tensor(
            "edge_index",
            self.edge_index,
            ndim=2,
        )

        if self.edge_index.shape[0] != 2:
            raise ValueError(
                "edge_index must have shape [2, E], "
                f"got {tuple(self.edge_index.shape)}."
            )

    def _validate_target_declarations(self) -> None:
        if bool(self.target_names) != bool(self.target_horizons):
            raise ValueError(
                "target_names and target_horizons must either both be "
                "declared or both be empty."
            )

        if self.target_names:
            _require_unique_strings(
                "target_names",
                self.target_names,
            )

            for index, horizon in enumerate(self.target_horizons):
                _require_nonempty_string(
                    f"target_horizons[{index}]",
                    horizon,
                )

            if len(self.target_names) != len(self.target_horizons):
                raise ValueError(
                    "target_names and target_horizons must have the "
                    "same length."
                )

    def _validate_graph_membership(
        self,
        *,
        num_nodes: int,
        batch_size: int,
    ) -> None:
        _require_long_tensor(
            "node_batch_index",
            self.node_batch_index,
            ndim=1,
        )
        _require_first_dimension(
            "node_batch_index",
            self.node_batch_index,
            num_nodes,
        )

        if self.node_batch_index.numel() > 0:
            minimum = int(self.node_batch_index.min().item())
            maximum = int(self.node_batch_index.max().item())

            if minimum < 0 or maximum >= batch_size:
                raise ValueError(
                    "node_batch_index contains invalid graph IDs. "
                    f"Valid range is [0, {batch_size - 1}]."
                )

            observed = set(
                int(value)
                for value in self.node_batch_index.detach()
                .cpu()
                .tolist()
            )
            expected = set(range(batch_size))

            if observed != expected:
                raise ValueError(
                    "Every graph must contain at least one node. "
                    f"Observed graph IDs: {sorted(observed)}, "
                    f"expected: {sorted(expected)}."
                )

        if self.graph_ptr is None:
            return

        _require_long_tensor(
            "graph_ptr",
            self.graph_ptr,
            ndim=1,
        )

        if self.graph_ptr.shape[0] != batch_size + 1:
            raise ValueError(
                "graph_ptr must have shape [B + 1]. "
                f"Expected {batch_size + 1}, "
                f"got {self.graph_ptr.shape[0]}."
            )

        pointer_values = [
            int(value)
            for value in self.graph_ptr.detach().cpu().tolist()
        ]

        if pointer_values[0] != 0:
            raise ValueError("graph_ptr must begin with 0.")

        if pointer_values[-1] != num_nodes:
            raise ValueError(
                "graph_ptr must end with the total number of nodes."
            )

        if any(
            pointer_values[index]
            > pointer_values[index + 1]
            for index in range(batch_size)
        ):
            raise ValueError(
                "graph_ptr must be monotonically nondecreasing."
            )

        for graph_index in range(batch_size):
            start = pointer_values[graph_index]
            end = pointer_values[graph_index + 1]

            expected = torch.full(
                (end - start,),
                graph_index,
                dtype=torch.long,
                device=self.node_batch_index.device,
            )

            if not torch.equal(
                self.node_batch_index[start:end],
                expected,
            ):
                raise ValueError(
                    "graph_ptr is inconsistent with node_batch_index. "
                    "graph_ptr requires nodes to be packed contiguously "
                    "by graph."
                )

    def _validate_node_representations(
        self,
        *,
        num_nodes: int,
    ) -> None:
        representation_count = sum(
            (
                self.node_features is not None,
                self.node_state is not None,
                bool(self.node_feature_blocks),
            )
        )

        if representation_count != 1:
            raise ValueError(
                "Provide exactly one node representation: "
                "node_features, node_state, or node_feature_blocks."
            )

        if self.node_features is not None:
            _require_float_tensor(
                "node_features",
                self.node_features,
                ndim=2,
            )
            _require_first_dimension(
                "node_features",
                self.node_features,
                num_nodes,
            )

        if self.node_state is not None:
            _require_float_tensor(
                "node_state",
                self.node_state,
                ndim=2,
            )
            _require_first_dimension(
                "node_state",
                self.node_state,
                num_nodes,
            )

        if self.node_type is not None:
            _require_long_tensor(
                "node_type",
                self.node_type,
                ndim=1,
            )
            _require_first_dimension(
                "node_type",
                self.node_type,
                num_nodes,
            )
            _require_nonnegative_tensor(
                "node_type",
                self.node_type,
            )

        if not self.node_feature_blocks:
            return

        if self.node_type is None:
            raise ValueError(
                "node_type is required when node_feature_blocks are used."
            )

        covered_indices: list[Tensor] = []

        for block_index, block in enumerate(
            self.node_feature_blocks
        ):
            try:
                block.validate(
                    num_nodes=num_nodes,
                    packed_node_type=self.node_type,
                )
            except (TypeError, ValueError) as exc:
                raise type(exc)(
                    f"node_feature_blocks[{block_index}]: {exc}"
                ) from exc

            covered_indices.append(block.internal_node_index)

        concatenated = torch.cat(covered_indices)

        if concatenated.numel() != num_nodes:
            raise ValueError(
                "Typed feature blocks must collectively cover all nodes "
                "exactly once."
            )

        sorted_indices = torch.sort(concatenated).values
        expected_indices = torch.arange(
            num_nodes,
            dtype=torch.long,
            device=concatenated.device,
        )

        if not torch.equal(sorted_indices, expected_indices):
            raise ValueError(
                "Typed feature blocks must cover every internal node "
                "index exactly once."
            )

    def _validate_history(self, *, num_nodes: int) -> None:
        if self.history_sequences is None:
            if self.history_mask is not None:
                raise ValueError(
                    "history_mask cannot be provided without "
                    "history_sequences."
                )

            if self.history_feature_names:
                raise ValueError(
                    "history_feature_names cannot be provided without "
                    "history_sequences."
                )
            return

        _require_float_tensor(
            "history_sequences",
            self.history_sequences,
            ndim=3,
        )
        _require_first_dimension(
            "history_sequences",
            self.history_sequences,
            num_nodes,
        )

        if self.history_mask is not None:
            _require_bool_tensor(
                "history_mask",
                self.history_mask,
                ndim=2,
            )

            expected_shape = self.history_sequences.shape[:2]

            if self.history_mask.shape != expected_shape:
                raise ValueError(
                    "history_mask must match [N, T]. "
                    f"Expected {tuple(expected_shape)}, "
                    f"got {tuple(self.history_mask.shape)}."
                )

        if self.history_feature_names:
            _require_unique_strings(
                "history_feature_names",
                self.history_feature_names,
            )

            if (
                len(self.history_feature_names)
                != self.history_sequences.shape[2]
            ):
                raise ValueError(
                    "history_feature_names must match the historical "
                    f"feature width {self.history_sequences.shape[2]}."
                )

    def _validate_hazard_context(
        self,
        *,
        num_nodes: int,
        batch_size: int,
    ) -> None:
        _require_choice(
            "hazard_scope",
            self.hazard_scope,
            CANONICAL_SCOPES,
        )

        _require_long_tensor(
            "hazard_ids",
            self.hazard_ids,
            ndim=1,
        )
        _require_nonnegative_tensor(
            "hazard_ids",
            self.hazard_ids,
        )

        expected = (
            batch_size
            if self.hazard_scope == SCOPE_GRAPH
            else num_nodes
        )

        _require_first_dimension(
            "hazard_ids",
            self.hazard_ids,
            expected,
        )

        if self.hazard_features is not None:
            _require_float_tensor(
                "hazard_features",
                self.hazard_features,
                ndim=2,
            )
            _require_first_dimension(
                "hazard_features",
                self.hazard_features,
                expected,
            )

    def _validate_scenario_context(
        self,
        *,
        num_nodes: int,
        batch_size: int,
    ) -> None:
        _require_choice(
            "scenario_scope",
            self.scenario_scope,
            CANONICAL_SCOPES,
        )

        if self.scenario_features is None:
            return

        _require_float_tensor(
            "scenario_features",
            self.scenario_features,
            ndim=2,
        )

        expected = (
            batch_size
            if self.scenario_scope == SCOPE_GRAPH
            else num_nodes
        )

        _require_first_dimension(
            "scenario_features",
            self.scenario_features,
            expected,
        )

    def _validate_edges(
        self,
        *,
        num_nodes: int,
        num_edges: int,
        batch_size: int,
    ) -> None:
        _require_long_tensor(
            "edge_relation_type",
            self.edge_relation_type,
            ndim=1,
        )
        _require_first_dimension(
            "edge_relation_type",
            self.edge_relation_type,
            num_edges,
        )
        _require_nonnegative_tensor(
            "edge_relation_type",
            self.edge_relation_type,
        )

        if num_edges > 0:
            minimum = int(self.edge_index.min().item())
            maximum = int(self.edge_index.max().item())

            if minimum < 0 or maximum >= num_nodes:
                raise ValueError(
                    "edge_index contains an invalid node index. "
                    f"Valid range is [0, {num_nodes - 1}]."
                )

        source_batch = self.node_batch_index[self.source_index]
        target_batch = self.node_batch_index[self.target_index]

        cross_graph_mask = source_batch != target_batch

        if (
            bool(cross_graph_mask.any())
            and not self.allow_cross_graph_edges
        ):
            invalid_edge = int(
                torch.nonzero(
                    cross_graph_mask,
                    as_tuple=False,
                )[0].item()
            )
            raise ValueError(
                "Cross-graph edge detected at edge "
                f"{invalid_edge}. Cross-graph edges are disabled."
            )

        if self.edge_batch_index is not None:
            _require_long_tensor(
                "edge_batch_index",
                self.edge_batch_index,
                ndim=1,
            )
            _require_first_dimension(
                "edge_batch_index",
                self.edge_batch_index,
                num_edges,
            )

            if self.edge_batch_index.numel() > 0:
                minimum = int(self.edge_batch_index.min().item())
                maximum = int(self.edge_batch_index.max().item())

                if minimum < 0 or maximum >= batch_size:
                    raise ValueError(
                        "edge_batch_index contains invalid graph IDs."
                    )

            if not torch.equal(
                self.edge_batch_index,
                source_batch,
            ):
                raise ValueError(
                    "edge_batch_index must identify each edge's source "
                    "graph."
                )

        if self.edge_attributes is not None:
            _require_float_tensor(
                "edge_attributes",
                self.edge_attributes,
                ndim=2,
            )
            _require_first_dimension(
                "edge_attributes",
                self.edge_attributes,
                num_edges,
            )

        if self.semantic_edge_weight is not None:
            _require_float_tensor(
                "semantic_edge_weight",
                self.semantic_edge_weight,
                ndim=1,
            )
            _require_first_dimension(
                "semantic_edge_weight",
                self.semantic_edge_weight,
                num_edges,
            )

        if self.external_edge_ids is not None:
            if len(self.external_edge_ids) != num_edges:
                raise ValueError(
                    "external_edge_ids must contain one ID per edge."
                )

            source_graphs = [
                int(value)
                for value in source_batch.detach().cpu().tolist()
            ]
            seen: set[tuple[int, str]] = set()

            for edge_position, edge_id in enumerate(
                self.external_edge_ids
            ):
                _require_nonempty_string(
                    f"external_edge_ids[{edge_position}]",
                    edge_id,
                )

                key = (
                    source_graphs[edge_position],
                    edge_id,
                )

                if key in seen:
                    raise ValueError(
                        "Duplicate external edge ID within one graph: "
                        f"graph={key[0]}, edge_id={key[1]!r}."
                    )

                seen.add(key)

        if self.edge_attribute_names:
            if self.edge_attributes is None:
                raise ValueError(
                    "edge_attribute_names require edge_attributes."
                )

            _require_unique_strings(
                "edge_attribute_names",
                self.edge_attribute_names,
            )

            if (
                len(self.edge_attribute_names)
                != self.edge_attributes.shape[1]
            ):
                raise ValueError(
                    "edge_attribute_names must match the edge-attribute "
                    f"width {self.edge_attributes.shape[1]}."
                )

    def _validate_supervision(self, *, num_nodes: int) -> None:
        if self.targets is None:
            if self.target_mask is not None:
                raise ValueError(
                    "target_mask cannot be provided without targets."
                )
            return

        if not self.target_names:
            raise ValueError(
                "target_names are required whenever targets are present."
            )

        _require_float_tensor(
            "targets",
            self.targets,
            ndim=2,
        )
        _require_first_dimension(
            "targets",
            self.targets,
            num_nodes,
        )

        if self.target_mask is None:
            raise ValueError(
                "target_mask is required when targets are provided."
            )

        _require_bool_tensor(
            "target_mask",
            self.target_mask,
            ndim=2,
        )
        _require_same_shape(
            "targets",
            self.targets,
            "target_mask",
            self.target_mask,
        )

        if self.targets.shape[1] != self.num_targets:
            raise ValueError(
                "Target tensor width must match target_names. "
                f"Expected {self.num_targets}, "
                f"got {self.targets.shape[1]}."
            )

    def _validate_temporal_edges(self) -> None:
        if self.num_edges == 0:
            return

        edge_graph = self.node_batch_index[
            self.source_index
        ].detach().cpu()

        for edge_position in range(self.num_edges):
            graph_index = int(edge_graph[edge_position].item())

            origin = _to_datetime(
                self.temporal.origin_time[graph_index],
                name=f"origin_time[{graph_index}]",
            )
            cutoff = _to_datetime(
                self.temporal.feature_availability_cutoff[
                    graph_index
                ],
                name=(
                    "feature_availability_cutoff"
                    f"[{graph_index}]"
                ),
            )

            valid_from = (
                self.temporal.edge_valid_from[edge_position]
                if self.temporal.edge_valid_from is not None
                else None
            )
            valid_to = (
                self.temporal.edge_valid_to[edge_position]
                if self.temporal.edge_valid_to is not None
                else None
            )
            observed_at = (
                self.temporal.edge_observation_time[edge_position]
                if self.temporal.edge_observation_time is not None
                else None
            )

            valid_from_dt = (
                _to_datetime(
                    valid_from,
                    name=f"edge_valid_from[{edge_position}]",
                )
                if valid_from is not None
                else None
            )
            valid_to_dt = (
                _to_datetime(
                    valid_to,
                    name=f"edge_valid_to[{edge_position}]",
                )
                if valid_to is not None
                else None
            )
            observed_at_dt = (
                _to_datetime(
                    observed_at,
                    name=f"edge_observation_time[{edge_position}]",
                )
                if observed_at is not None
                else None
            )

            if (
                valid_from_dt is not None
                and valid_to_dt is not None
                and valid_from_dt > valid_to_dt
            ):
                raise ValueError(
                    f"Edge {edge_position}: edge_valid_from must not "
                    "exceed edge_valid_to."
                )

            if (
                observed_at_dt is not None
                and observed_at_dt > cutoff
            ):
                raise ValueError(
                    f"Edge {edge_position}: edge observation occurs "
                    "after the feature-availability cutoff."
                )

            if (
                observed_at_dt is not None
                and observed_at_dt > origin
            ):
                raise ValueError(
                    f"Edge {edge_position}: edge observation occurs "
                    "after the prediction origin."
                )

            if (
                valid_from_dt is not None
                and valid_from_dt > origin
            ):
                raise ValueError(
                    f"Edge {edge_position} is not yet valid at origin."
                )

            if valid_to_dt is not None and origin >= valid_to_dt:
                raise ValueError(
                    f"Edge {edge_position} is no longer valid at origin."
                )

    def _validate_external_node_identity(self) -> None:
        graph_indices = [
            int(value)
            for value in self.node_batch_index.detach().cpu().tolist()
        ]

        seen: set[tuple[int, str]] = set()

        for internal_index, external_id in enumerate(
            self.external_node_ids
        ):
            key = (
                graph_indices[internal_index],
                external_id,
            )

            if key in seen:
                raise ValueError(
                    "Duplicate external node ID within one graph: "
                    f"graph={key[0]}, node_id={key[1]!r}."
                )

            seen.add(key)

    def _validate_feature_name_alignment(self) -> None:
        if self.feature_names:
            if self.node_features is None:
                raise ValueError(
                    "feature_names are only valid with node_features."
                )

            _require_unique_strings(
                "feature_names",
                self.feature_names,
            )

            if len(self.feature_names) != self.node_features.shape[1]:
                raise ValueError(
                    "feature_names must match node-feature width. "
                    f"Expected {self.node_features.shape[1]}, "
                    f"got {len(self.feature_names)}."
                )

    def _validate_device_compatibility(self) -> None:
        tensors: dict[str, Tensor | None] = {
            "node_batch_index": self.node_batch_index,
            "hazard_ids": self.hazard_ids,
            "edge_index": self.edge_index,
            "edge_relation_type": self.edge_relation_type,
            "node_features": self.node_features,
            "node_state": self.node_state,
            "node_type": self.node_type,
            "history_sequences": self.history_sequences,
            "history_mask": self.history_mask,
            "hazard_features": self.hazard_features,
            "scenario_features": self.scenario_features,
            "edge_attributes": self.edge_attributes,
            "semantic_edge_weight": self.semantic_edge_weight,
            "edge_batch_index": self.edge_batch_index,
            "graph_ptr": self.graph_ptr,
            "targets": self.targets,
            "target_mask": self.target_mask,
        }

        for index, block in enumerate(self.node_feature_blocks):
            tensors[
                f"node_feature_blocks[{index}].internal_node_index"
            ] = block.internal_node_index
            tensors[
                f"node_feature_blocks[{index}].features"
            ] = block.features

        _validate_same_device(tensors)

    def derived_edge_batch_index(self) -> Tensor:
        return self.node_batch_index[self.source_index]

    def validate_registry_membership(
        self,
        *,
        valid_hazard_ids: Iterable[int],
        valid_relation_ids: Iterable[int],
        valid_node_type_ids: Iterable[int] | None = None,
    ) -> None:
        """
        Validate IDs against externally supplied registry memberships.

        This method intentionally accepts ID sets rather than importing
        registries directly.
        """

        valid_hazards = set(valid_hazard_ids)
        valid_relations = set(valid_relation_ids)

        observed_hazards = set(
            int(value)
            for value in self.hazard_ids.detach().cpu().tolist()
        )
        observed_relations = set(
            int(value)
            for value in self.edge_relation_type.detach().cpu().tolist()
        )

        unknown_hazards = sorted(
            observed_hazards - valid_hazards
        )
        unknown_relations = sorted(
            observed_relations - valid_relations
        )

        if unknown_hazards:
            raise ValueError(
                f"Unknown hazard IDs: {unknown_hazards}."
            )

        if unknown_relations:
            raise ValueError(
                f"Unknown relation IDs: {unknown_relations}."
            )

        if self.node_type is not None and valid_node_type_ids is not None:
            valid_node_types = set(valid_node_type_ids)
            observed_node_types = set(
                int(value)
                for value in self.node_type.detach().cpu().tolist()
            )
            unknown_node_types = sorted(
                observed_node_types - valid_node_types
            )

            if unknown_node_types:
                raise ValueError(
                    f"Unknown node-type IDs: {unknown_node_types}."
                )

    def replace(self, **changes: Any) -> UrbanGraphBatch:
        """Return a reconstructed and revalidated batch."""

        return dataclass_replace(self, **changes)

    def to(
        self,
        device: torch.device | str,
    ) -> UrbanGraphBatch:
        """Return a reconstructed batch with all tensors moved."""

        moved_blocks = tuple(
            block.to(device)
            for block in self.node_feature_blocks
        )

        return dataclass_replace(
            self,
            node_batch_index=self.node_batch_index.to(device),
            hazard_ids=self.hazard_ids.to(device),
            edge_index=self.edge_index.to(device),
            edge_relation_type=self.edge_relation_type.to(device),
            node_features=_move_tensor(self.node_features, device),
            node_state=_move_tensor(self.node_state, device),
            node_feature_blocks=moved_blocks,
            node_type=_move_tensor(self.node_type, device),
            history_sequences=_move_tensor(
                self.history_sequences,
                device,
            ),
            history_mask=_move_tensor(self.history_mask, device),
            hazard_features=_move_tensor(
                self.hazard_features,
                device,
            ),
            scenario_features=_move_tensor(
                self.scenario_features,
                device,
            ),
            edge_attributes=_move_tensor(
                self.edge_attributes,
                device,
            ),
            semantic_edge_weight=_move_tensor(
                self.semantic_edge_weight,
                device,
            ),
            edge_batch_index=_move_tensor(
                self.edge_batch_index,
                device,
            ),
            graph_ptr=_move_tensor(self.graph_ptr, device),
            targets=_move_tensor(self.targets, device),
            target_mask=_move_tensor(self.target_mask, device),
        )


# =============================================================================
# Intermediate module outputs
# =============================================================================


@dataclass(slots=True)
class MemoryEncoderOutput:
    memory_state: Tensor
    temporal_states: Tensor | None = None
    temporal_attention: Tensor | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _require_float_tensor(
            "memory_state",
            self.memory_state,
            ndim=2,
        )

        num_nodes, hidden_dim = self.memory_state.shape

        if self.temporal_states is not None:
            _require_float_tensor(
                "temporal_states",
                self.temporal_states,
                ndim=3,
            )
            _require_first_dimension(
                "temporal_states",
                self.temporal_states,
                num_nodes,
            )

            if self.temporal_states.shape[2] != hidden_dim:
                raise ValueError(
                    "temporal_states hidden width must match "
                    "memory_state."
                )

        if self.temporal_attention is not None:
            _require_float_tensor(
                "temporal_attention",
                self.temporal_attention,
                ndim=2,
            )
            _require_first_dimension(
                "temporal_attention",
                self.temporal_attention,
                num_nodes,
            )
            _require_nonnegative_tensor(
                "temporal_attention",
                self.temporal_attention,
            )

            if (
                self.temporal_states is not None
                and self.temporal_attention.shape[1]
                != self.temporal_states.shape[1]
            ):
                raise ValueError(
                    "temporal_attention width must match "
                    "temporal_states time width."
                )

        _validate_same_device(
            {
                "memory_state": self.memory_state,
                "temporal_states": self.temporal_states,
                "temporal_attention": self.temporal_attention,
            }
        )


@dataclass(slots=True)
class TemporalAttentionOutput:
    context_state: Tensor
    attention_weight: Tensor

    attention_weight_by_head: Tensor | None = None
    head_reduction_policy: str = ATTENTION_HEAD_REDUCTION_NONE

    def __post_init__(self) -> None:
        _require_float_tensor(
            "context_state",
            self.context_state,
            ndim=2,
        )
        _require_float_tensor(
            "attention_weight",
            self.attention_weight,
            ndim=2,
        )
        _require_first_dimension(
            "attention_weight",
            self.attention_weight,
            self.context_state.shape[0],
        )
        _require_nonnegative_tensor(
            "attention_weight",
            self.attention_weight,
        )

        _require_choice(
            "head_reduction_policy",
            self.head_reduction_policy,
            CANONICAL_ATTENTION_HEAD_REDUCTIONS,
        )

        if self.attention_weight_by_head is not None:
            _require_float_tensor(
                "attention_weight_by_head",
                self.attention_weight_by_head,
                ndim=3,
            )

            if (
                self.attention_weight_by_head.shape[0]
                != self.context_state.shape[0]
                or self.attention_weight_by_head.shape[2]
                != self.attention_weight.shape[1]
            ):
                raise ValueError(
                    "attention_weight_by_head must have shape [N, A, T]."
                )

        _validate_same_device(
            {
                "context_state": self.context_state,
                "attention_weight": self.attention_weight,
                "attention_weight_by_head": (
                    self.attention_weight_by_head
                ),
            }
        )


@dataclass(slots=True)
class HazardEncoderOutput:
    node_hazard_context: Tensor

    graph_hazard_context: Tensor | None = None
    hazard_embedding: Tensor | None = None
    scenario_context: Tensor | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _require_float_tensor(
            "node_hazard_context",
            self.node_hazard_context,
            ndim=2,
        )

        if self.graph_hazard_context is not None:
            _require_float_tensor(
                "graph_hazard_context",
                self.graph_hazard_context,
                ndim=2,
            )

        if self.hazard_embedding is not None:
            _require_float_tensor(
                "hazard_embedding",
                self.hazard_embedding,
                ndim=2,
            )

        if self.scenario_context is not None:
            _require_float_tensor(
                "scenario_context",
                self.scenario_context,
                ndim=2,
            )

        _validate_same_device(
            {
                "node_hazard_context": self.node_hazard_context,
                "graph_hazard_context": self.graph_hazard_context,
                "hazard_embedding": self.hazard_embedding,
                "scenario_context": self.scenario_context,
            }
        )


@dataclass(slots=True)
class HazardQueriedMemoryOutput:
    hazard_memory_state: Tensor

    temporal_attention: Tensor | None = None
    temporal_attention_by_head: Tensor | None = None
    head_reduction_policy: str = ATTENTION_HEAD_REDUCTION_NONE

    def __post_init__(self) -> None:
        _require_float_tensor(
            "hazard_memory_state",
            self.hazard_memory_state,
            ndim=2,
        )

        if self.temporal_attention is not None:
            _require_float_tensor(
                "temporal_attention",
                self.temporal_attention,
                ndim=2,
            )
            _require_first_dimension(
                "temporal_attention",
                self.temporal_attention,
                self.hazard_memory_state.shape[0],
            )

        if self.temporal_attention_by_head is not None:
            _require_float_tensor(
                "temporal_attention_by_head",
                self.temporal_attention_by_head,
                ndim=3,
            )
            _require_first_dimension(
                "temporal_attention_by_head",
                self.temporal_attention_by_head,
                self.hazard_memory_state.shape[0],
            )

        _require_choice(
            "head_reduction_policy",
            self.head_reduction_policy,
            CANONICAL_ATTENTION_HEAD_REDUCTIONS,
        )

        _validate_same_device(
            {
                "hazard_memory_state": self.hazard_memory_state,
                "temporal_attention": self.temporal_attention,
                "temporal_attention_by_head": (
                    self.temporal_attention_by_head
                ),
            }
        )


@dataclass(slots=True)
class RelationGateOutput:
    gate_values: Tensor
    scope: str
    relation_registry_version: str

    gate_logits: Tensor | None = None
    relation_mask: Tensor | None = None
    regularization_terms: TensorMapping = field(default_factory=dict)
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _require_float_tensor(
            "gate_values",
            self.gate_values,
            ndim=2,
        )
        _require_nonnegative_tensor(
            "gate_values",
            self.gate_values,
        )

        _require_choice(
            "relation-gate scope",
            self.scope,
            CANONICAL_RELATION_GATE_SCOPES,
        )
        _require_nonempty_string(
            "relation_registry_version",
            self.relation_registry_version,
        )

        if self.gate_logits is not None:
            _require_float_tensor(
                "gate_logits",
                self.gate_logits,
                ndim=2,
            )
            _require_same_shape(
                "gate_values",
                self.gate_values,
                "gate_logits",
                self.gate_logits,
            )

        if self.relation_mask is not None:
            _require_bool_tensor(
                "relation_mask",
                self.relation_mask,
            )

            if self.relation_mask.ndim == 1:
                if (
                    self.relation_mask.shape[0]
                    != self.gate_values.shape[1]
                ):
                    raise ValueError(
                        "A one-dimensional relation_mask must have "
                        "length R."
                    )
            elif self.relation_mask.shape != self.gate_values.shape:
                raise ValueError(
                    "relation_mask must have shape [R] or match "
                    "gate_values."
                )

        _validate_regularization_terms(
            "regularization_terms",
            self.regularization_terms,
        )

        _validate_same_device(
            {
                "gate_values": self.gate_values,
                "gate_logits": self.gate_logits,
                "relation_mask": self.relation_mask,
                **dict(self.regularization_terms),
            }
        )

    def validate_alignment(
        self,
        *,
        num_graphs: int,
        num_nodes: int,
        num_edges: int,
        num_relations: int,
    ) -> None:
        expected_first_dimension = {
            RELATION_GATE_SCOPE_GRAPH: num_graphs,
            RELATION_GATE_SCOPE_TARGET_NODE: num_nodes,
            RELATION_GATE_SCOPE_SOURCE_NODE: num_nodes,
            RELATION_GATE_SCOPE_SOURCE_TARGET: num_edges,
        }[self.scope]

        expected_shape = (
            expected_first_dimension,
            num_relations,
        )

        if self.gate_values.shape != expected_shape:
            raise ValueError(
                f"Gate scope {self.scope!r} requires shape "
                f"{expected_shape}, got {tuple(self.gate_values.shape)}."
            )


@dataclass(slots=True)
class EdgeAttentionOutput:
    attention_weight: Tensor
    normalization_scope: str
    head_reduction_policy: str

    attention_logits: Tensor | None = None
    attention_weight_by_head: Tensor | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _require_float_tensor(
            "attention_weight",
            self.attention_weight,
            ndim=1,
        )
        _require_nonnegative_tensor(
            "attention_weight",
            self.attention_weight,
        )

        _require_choice(
            "normalization_scope",
            self.normalization_scope,
            CANONICAL_ATTENTION_NORMALIZATION_MODES,
        )
        _require_choice(
            "head_reduction_policy",
            self.head_reduction_policy,
            CANONICAL_ATTENTION_HEAD_REDUCTIONS,
        )

        if self.attention_logits is not None:
            _require_float_tensor(
                "attention_logits",
                self.attention_logits,
            )

            if self.attention_logits.ndim not in (1, 2):
                raise ValueError(
                    "attention_logits must have shape [E] or [E, A]."
                )

            _require_first_dimension(
                "attention_logits",
                self.attention_logits,
                self.attention_weight.shape[0],
            )

        if self.attention_weight_by_head is not None:
            _require_float_tensor(
                "attention_weight_by_head",
                self.attention_weight_by_head,
                ndim=2,
            )
            _require_first_dimension(
                "attention_weight_by_head",
                self.attention_weight_by_head,
                self.attention_weight.shape[0],
            )
            _require_nonnegative_tensor(
                "attention_weight_by_head",
                self.attention_weight_by_head,
            )

        _validate_same_device(
            {
                "attention_weight": self.attention_weight,
                "attention_logits": self.attention_logits,
                "attention_weight_by_head": (
                    self.attention_weight_by_head
                ),
            }
        )

    def validate_alignment(self, *, num_edges: int) -> None:
        _require_first_dimension(
            "attention_weight",
            self.attention_weight,
            num_edges,
        )

    def validate_normalization(
        self,
        *,
        edge_index: Tensor,
        edge_relation_type: Tensor,
        edge_mask: Tensor | None = None,
        atol: float = 1e-5,
    ) -> None:
        """
        Validate contextual normalization of reduced attention weights.

        Unnormalized sigmoid attention is intentionally exempt.
        """

        if (
            self.normalization_scope
            == ATTENTION_NORMALIZATION_UNNORMALIZED_SIGMOID
        ):
            return

        _require_long_tensor(
            "edge_index",
            edge_index,
            ndim=2,
        )

        if edge_index.shape[0] != 2:
            raise ValueError("edge_index must have shape [2, E].")

        num_edges = edge_index.shape[1]
        self.validate_alignment(num_edges=num_edges)

        _require_long_tensor(
            "edge_relation_type",
            edge_relation_type,
            ndim=1,
        )
        _require_first_dimension(
            "edge_relation_type",
            edge_relation_type,
            num_edges,
        )

        if edge_mask is None:
            active_mask = torch.ones(
                num_edges,
                dtype=torch.bool,
                device=edge_index.device,
            )
        else:
            _require_bool_tensor(
                "edge_mask",
                edge_mask,
                ndim=1,
            )
            _require_first_dimension(
                "edge_mask",
                edge_mask,
                num_edges,
            )
            active_mask = edge_mask

        if not bool(active_mask.any()):
            return

        weights = self.attention_weight[active_mask]
        target_index = edge_index[1, active_mask]
        relation_type = edge_relation_type[active_mask]

        if (
            self.normalization_scope
            == ATTENTION_NORMALIZATION_TARGET_NODE
        ):
            group_index = target_index

        elif (
            self.normalization_scope
            == ATTENTION_NORMALIZATION_TARGET_NODE_RELATION
        ):
            num_relations = int(relation_type.max().item()) + 1
            group_index = (
                target_index * num_relations + relation_type
            )

        elif (
            self.normalization_scope
            == ATTENTION_NORMALIZATION_GLOBAL_RELATION
        ):
            group_index = relation_type

        else:
            raise ValueError(
                "Unsupported contextual normalization scope "
                f"{self.normalization_scope!r}."
            )

        num_groups = int(group_index.max().item()) + 1

        sums = torch.zeros(
            num_groups,
            dtype=weights.dtype,
            device=weights.device,
        )
        counts = torch.zeros(
            num_groups,
            dtype=torch.long,
            device=weights.device,
        )

        sums.index_add_(0, group_index, weights)
        counts.index_add_(
            0,
            group_index,
            torch.ones_like(group_index),
        )

        observed_sums = sums[counts > 0]

        if not torch.allclose(
            observed_sums,
            torch.ones_like(observed_sums),
            atol=atol,
            rtol=0.0,
        ):
            raise ValueError(
                "Attention weights do not sum to one within the "
                f"declared scope {self.normalization_scope!r}."
            )


@dataclass(slots=True)
class MessageBuilderOutput:
    messages: Tensor
    message_norms: Tensor | None = None
    component_weights: TensorMapping = field(default_factory=dict)

    def __post_init__(self) -> None:
        _require_float_tensor(
            "messages",
            self.messages,
            ndim=2,
        )

        num_edges = self.messages.shape[0]

        if self.message_norms is not None:
            _require_float_tensor(
                "message_norms",
                self.message_norms,
                ndim=1,
            )
            _require_first_dimension(
                "message_norms",
                self.message_norms,
                num_edges,
            )
            _require_nonnegative_tensor(
                "message_norms",
                self.message_norms,
            )

        _validate_float_tensor_mapping(
            "component_weights",
            self.component_weights,
            first_dimension=num_edges,
        )

        _validate_same_device(
            {
                "messages": self.messages,
                "message_norms": self.message_norms,
                **dict(self.component_weights),
            }
        )


@dataclass(slots=True)
class AggregationOutput:
    aggregated_messages: Tensor
    relation_aggregates: Tensor | None = None
    incoming_counts: Tensor | None = None

    def __post_init__(self) -> None:
        _require_float_tensor(
            "aggregated_messages",
            self.aggregated_messages,
            ndim=2,
        )

        num_nodes, hidden_dim = self.aggregated_messages.shape

        if self.relation_aggregates is not None:
            _require_float_tensor(
                "relation_aggregates",
                self.relation_aggregates,
                ndim=3,
            )
            _require_first_dimension(
                "relation_aggregates",
                self.relation_aggregates,
                num_nodes,
            )

            if self.relation_aggregates.shape[2] != hidden_dim:
                raise ValueError(
                    "relation_aggregates hidden width must match "
                    "aggregated_messages."
                )

        if self.incoming_counts is not None:
            _require_integer_tensor(
                "incoming_counts",
                self.incoming_counts,
                ndim=1,
            )
            _require_first_dimension(
                "incoming_counts",
                self.incoming_counts,
                num_nodes,
            )
            _require_nonnegative_tensor(
                "incoming_counts",
                self.incoming_counts,
            )

        _validate_same_device(
            {
                "aggregated_messages": self.aggregated_messages,
                "relation_aggregates": self.relation_aggregates,
                "incoming_counts": self.incoming_counts,
            }
        )


@dataclass(slots=True)
class FunctionalMessagePassingOutput:
    updated_node_state: Tensor

    relation_gate_output: RelationGateOutput | None = None
    edge_attention_output: EdgeAttentionOutput | None = None
    aggregated_messages: Tensor | None = None
    normalization_weight: Tensor | None = None

    explanation_trace: ExplanationTrace | None = None
    regularization_terms: TensorMapping = field(default_factory=dict)

    def __post_init__(self) -> None:
        _require_float_tensor(
            "updated_node_state",
            self.updated_node_state,
            ndim=2,
        )

        num_nodes, hidden_dim = self.updated_node_state.shape

        if self.aggregated_messages is not None:
            _require_float_tensor(
                "aggregated_messages",
                self.aggregated_messages,
                ndim=2,
            )

            if self.aggregated_messages.shape != (
                num_nodes,
                hidden_dim,
            ):
                raise ValueError(
                    "aggregated_messages must match "
                    "updated_node_state."
                )

        if self.normalization_weight is not None:
            _require_float_tensor(
                "normalization_weight",
                self.normalization_weight,
                ndim=1,
            )
            _require_nonnegative_tensor(
                "normalization_weight",
                self.normalization_weight,
            )

        _validate_regularization_terms(
            "regularization_terms",
            self.regularization_terms,
        )

        _validate_same_device(
            {
                "updated_node_state": self.updated_node_state,
                "aggregated_messages": self.aggregated_messages,
                "normalization_weight": self.normalization_weight,
                **dict(self.regularization_terms),
            }
        )


# =============================================================================
# Prediction and explanation alignment
# =============================================================================


@dataclass(slots=True)
class PredictionAlignment:
    """Identity and provenance required to interpret model outputs."""

    external_node_ids: tuple[str, ...]
    node_batch_index: Tensor

    hazard_ids: Tensor
    hazard_scope: str

    origin_time: tuple[TimePoint, ...]
    target_names: tuple[str, ...]
    target_horizons: tuple[str, ...]

    run_metadata: RunMetadata
    contract_versions: ContractVersions

    target_start_time_by_graph_target: (
        tuple[tuple[TimePoint, ...], ...] | None
    ) = None
    target_end_time_by_graph_target: (
        tuple[tuple[TimePoint, ...], ...] | None
    ) = None

    def validate(
        self,
        *,
        num_nodes: int,
        num_targets: int,
        require_current_versions: bool = False,
    ) -> None:
        if len(self.external_node_ids) != num_nodes:
            raise ValueError(
                "Prediction alignment node IDs do not match the "
                "prediction node count."
            )

        _require_unique_strings(
            "external_node_ids",
            self.external_node_ids,
        )

        _require_long_tensor(
            "node_batch_index",
            self.node_batch_index,
            ndim=1,
        )
        _require_first_dimension(
            "node_batch_index",
            self.node_batch_index,
            num_nodes,
        )
        _require_nonnegative_tensor(
            "node_batch_index",
            self.node_batch_index,
        )

        batch_size = len(self.origin_time)

        if batch_size <= 0:
            raise ValueError(
                "Prediction alignment must contain at least one graph."
            )

        _validate_time_sequence(
            "origin_time",
            self.origin_time,
            batch_size,
        )

        _require_choice(
            "hazard_scope",
            self.hazard_scope,
            CANONICAL_SCOPES,
        )

        _require_long_tensor(
            "hazard_ids",
            self.hazard_ids,
            ndim=1,
        )

        expected_hazard_count = (
            batch_size
            if self.hazard_scope == SCOPE_GRAPH
            else num_nodes
        )

        _require_first_dimension(
            "hazard_ids",
            self.hazard_ids,
            expected_hazard_count,
        )

        _require_unique_strings(
            "target_names",
            self.target_names,
        )

        if len(self.target_names) != num_targets:
            raise ValueError(
                "target_names must match prediction width. "
                f"Expected {num_targets}, got {len(self.target_names)}."
            )

        if len(self.target_horizons) != num_targets:
            raise ValueError(
                "target_horizons must match prediction width. "
                f"Expected {num_targets}, "
                f"got {len(self.target_horizons)}."
            )

        for index, horizon in enumerate(self.target_horizons):
            _require_nonempty_string(
                f"target_horizons[{index}]",
                horizon,
            )

        if self.node_batch_index.numel() > 0:
            maximum = int(self.node_batch_index.max().item())

            if maximum >= batch_size:
                raise ValueError(
                    "node_batch_index references a graph without an "
                    "origin_time."
                )

        starts_present = (
            self.target_start_time_by_graph_target is not None
        )
        ends_present = (
            self.target_end_time_by_graph_target is not None
        )

        if starts_present != ends_present:
            raise ValueError(
                "Target start and end matrices must either both be "
                "present or both be absent."
            )

        if starts_present:
            _validate_time_matrix(
                "target_start_time_by_graph_target",
                self.target_start_time_by_graph_target,
                expected_rows=batch_size,
                expected_columns=num_targets,
            )
            _validate_time_matrix(
                "target_end_time_by_graph_target",
                self.target_end_time_by_graph_target,
                expected_rows=batch_size,
                expected_columns=num_targets,
            )

        self.run_metadata.validate()
        self.contract_versions.validate(
            require_current=require_current_versions
        )

        _validate_same_device(
            {
                "node_batch_index": self.node_batch_index,
                "hazard_ids": self.hazard_ids,
            }
        )

    def assert_equivalent(
        self,
        other: PredictionAlignment,
    ) -> None:
        mismatches: list[str] = []

        if self.external_node_ids != other.external_node_ids:
            mismatches.append("external_node_ids")

        if not _tensor_equal(
            self.node_batch_index,
            other.node_batch_index,
        ):
            mismatches.append("node_batch_index")

        if not _tensor_equal(self.hazard_ids, other.hazard_ids):
            mismatches.append("hazard_ids")

        if self.hazard_scope != other.hazard_scope:
            mismatches.append("hazard_scope")

        if self.origin_time != other.origin_time:
            mismatches.append("origin_time")

        if self.target_names != other.target_names:
            mismatches.append("target_names")

        if self.target_horizons != other.target_horizons:
            mismatches.append("target_horizons")

        if (
            self.target_start_time_by_graph_target
            != other.target_start_time_by_graph_target
        ):
            mismatches.append(
                "target_start_time_by_graph_target"
            )

        if (
            self.target_end_time_by_graph_target
            != other.target_end_time_by_graph_target
        ):
            mismatches.append(
                "target_end_time_by_graph_target"
            )

        if self.run_metadata != other.run_metadata:
            mismatches.append("run_metadata")

        if self.contract_versions != other.contract_versions:
            mismatches.append("contract_versions")

        if mismatches:
            raise ValueError(
                "Prediction alignments are not equivalent. "
                f"Mismatched fields: {mismatches}."
            )


@dataclass(slots=True)
class PredictionOutput:
    prediction_mean: Tensor
    alignment: PredictionAlignment

    count_rate: Tensor | None = None
    ranking_score: Tensor | None = None

    output_transformation: str = "identity"
    higher_is_riskier: bool = True
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _require_float_tensor(
            "prediction_mean",
            self.prediction_mean,
            ndim=2,
        )

        num_nodes, num_targets = self.prediction_mean.shape

        self.alignment.validate(
            num_nodes=num_nodes,
            num_targets=num_targets,
        )

        if self.count_rate is not None:
            _require_float_tensor(
                "count_rate",
                self.count_rate,
                ndim=2,
            )
            _require_same_shape(
                "prediction_mean",
                self.prediction_mean,
                "count_rate",
                self.count_rate,
            )
            _require_nonnegative_tensor(
                "count_rate",
                self.count_rate,
            )

        if self.ranking_score is not None:
            _require_float_tensor(
                "ranking_score",
                self.ranking_score,
                ndim=2,
            )
            _require_same_shape(
                "prediction_mean",
                self.prediction_mean,
                "ranking_score",
                self.ranking_score,
            )

        _require_nonempty_string(
            "output_transformation",
            self.output_transformation,
        )

        _validate_same_device(
            {
                "prediction_mean": self.prediction_mean,
                "count_rate": self.count_rate,
                "ranking_score": self.ranking_score,
                "alignment.node_batch_index": (
                    self.alignment.node_batch_index
                ),
                "alignment.hazard_ids": self.alignment.hazard_ids,
            }
        )


@dataclass(slots=True)
class UncertaintyOutput:
    method: str

    variance: Tensor | None = None
    standard_deviation: Tensor | None = None
    quantiles: Tensor | None = None
    quantile_levels: tuple[float, ...] = ()
    lower_bound: Tensor | None = None
    upper_bound: Tensor | None = None
    confidence_score: Tensor | None = None

    metadata: Mapping[str, Any] = field(default_factory=dict)

    def validate(
        self,
        *,
        num_nodes: int,
        num_targets: int,
    ) -> None:
        _require_nonempty_string("method", self.method)

        values = {
            "variance": self.variance,
            "standard_deviation": self.standard_deviation,
            "quantiles": self.quantiles,
            "lower_bound": self.lower_bound,
            "upper_bound": self.upper_bound,
            "confidence_score": self.confidence_score,
        }

        if not any(value is not None for value in values.values()):
            raise ValueError(
                "UncertaintyOutput must contain at least one tensor."
            )

        expected_shape = (num_nodes, num_targets)

        for name in (
            "variance",
            "standard_deviation",
            "lower_bound",
            "upper_bound",
            "confidence_score",
        ):
            tensor = values[name]

            if tensor is None:
                continue

            _require_float_tensor(name, tensor, ndim=2)

            if tensor.shape != expected_shape:
                raise ValueError(
                    f"{name} must have shape {expected_shape}, "
                    f"got {tuple(tensor.shape)}."
                )

        if self.variance is not None:
            _require_nonnegative_tensor(
                "variance",
                self.variance,
            )

        if self.standard_deviation is not None:
            _require_nonnegative_tensor(
                "standard_deviation",
                self.standard_deviation,
            )

        if self.quantiles is not None:
            _require_float_tensor(
                "quantiles",
                self.quantiles,
                ndim=3,
            )

            if self.quantiles.shape[:2] != expected_shape:
                raise ValueError(
                    "quantiles must have shape [N, K, Q]."
                )

            if not self.quantile_levels:
                raise ValueError(
                    "quantile_levels are required with quantile outputs."
                )

            if (
                len(self.quantile_levels)
                != self.quantiles.shape[2]
            ):
                raise ValueError(
                    "quantile_levels must match quantile width Q."
                )

            if any(
                not 0.0 < level < 1.0
                for level in self.quantile_levels
            ):
                raise ValueError(
                    "Every quantile level must lie strictly between 0 "
                    "and 1."
                )

            if tuple(sorted(self.quantile_levels)) != (
                self.quantile_levels
            ):
                raise ValueError(
                    "quantile_levels must be strictly increasing."
                )

            if len(set(self.quantile_levels)) != len(
                self.quantile_levels
            ):
                raise ValueError(
                    "quantile_levels must not contain duplicates."
                )

            if self.quantiles.shape[2] > 1 and bool(
                (
                    self.quantiles[..., 1:]
                    < self.quantiles[..., :-1]
                ).any()
            ):
                raise ValueError(
                    "Predicted quantiles must be nondecreasing."
                )

        elif self.quantile_levels:
            raise ValueError(
                "quantile_levels cannot be provided without quantiles."
            )

        if (self.lower_bound is None) != (
            self.upper_bound is None
        ):
            raise ValueError(
                "lower_bound and upper_bound must either both be "
                "present or both be absent."
            )

        if (
            self.lower_bound is not None
            and bool((self.lower_bound > self.upper_bound).any())
        ):
            raise ValueError(
                "lower_bound cannot exceed upper_bound."
            )

        _validate_same_device(values)


@dataclass(slots=True)
class ReportingBiasOutput:
    latent_disruption: Tensor
    reporting_propensity: Tensor
    observed_prediction: Tensor

    metadata: Mapping[str, Any] = field(default_factory=dict)

    def validate(
        self,
        *,
        num_nodes: int,
        num_targets: int,
    ) -> None:
        expected_shape = (num_nodes, num_targets)

        for name, value in (
            ("latent_disruption", self.latent_disruption),
            ("reporting_propensity", self.reporting_propensity),
            ("observed_prediction", self.observed_prediction),
        ):
            _require_float_tensor(name, value, ndim=2)

            if value.shape != expected_shape:
                raise ValueError(
                    f"{name} must have shape {expected_shape}, "
                    f"got {tuple(value.shape)}."
                )

        _require_nonnegative_tensor(
            "reporting_propensity",
            self.reporting_propensity,
        )

        _validate_same_device(
            {
                "latent_disruption": self.latent_disruption,
                "reporting_propensity": self.reporting_propensity,
                "observed_prediction": self.observed_prediction,
            }
        )


@dataclass(slots=True)
class ExplanationTrace:
    alignment: PredictionAlignment

    relation_gate_output: RelationGateOutput | None = None
    edge_attention_output: EdgeAttentionOutput | None = None
    temporal_attention: Tensor | None = None
    pathway_scores: Tensor | None = None

    edge_index: Tensor | None = None
    edge_relation_type: Tensor | None = None

    history_time_points_by_graph: (
        tuple[tuple[TimePoint, ...], ...] | None
    ) = None

    layer_index: int | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def validate(
        self,
        *,
        num_nodes: int,
        num_targets: int,
    ) -> None:
        self.alignment.validate(
            num_nodes=num_nodes,
            num_targets=num_targets,
        )

        if not any(
            value is not None
            for value in (
                self.relation_gate_output,
                self.edge_attention_output,
                self.temporal_attention,
                self.pathway_scores,
            )
        ):
            raise ValueError(
                "ExplanationTrace must contain at least one explanation "
                "object or tensor."
            )

        num_edges: int | None = None

        if self.edge_index is not None:
            _require_long_tensor(
                "edge_index",
                self.edge_index,
                ndim=2,
            )

            if self.edge_index.shape[0] != 2:
                raise ValueError(
                    "edge_index must have shape [2, E]."
                )

            num_edges = int(self.edge_index.shape[1])

        if self.edge_relation_type is not None:
            if num_edges is None:
                raise ValueError(
                    "edge_index is required with edge_relation_type."
                )

            _require_long_tensor(
                "edge_relation_type",
                self.edge_relation_type,
                ndim=1,
            )
            _require_first_dimension(
                "edge_relation_type",
                self.edge_relation_type,
                num_edges,
            )

        if self.edge_attention_output is not None:
            if num_edges is None:
                raise ValueError(
                    "edge_index is required when edge attention is "
                    "exported."
                )

            if self.edge_relation_type is None:
                raise ValueError(
                    "edge_relation_type is required when edge attention "
                    "is exported."
                )

            self.edge_attention_output.validate_alignment(
                num_edges=num_edges
            )

        if self.temporal_attention is not None:
            _require_float_tensor(
                "temporal_attention",
                self.temporal_attention,
                ndim=2,
            )
            _require_first_dimension(
                "temporal_attention",
                self.temporal_attention,
                num_nodes,
            )

            if self.history_time_points_by_graph is None:
                raise ValueError(
                    "history_time_points_by_graph is required when "
                    "temporal attention is exported."
                )

            batch_size = len(self.alignment.origin_time)

            if (
                len(self.history_time_points_by_graph)
                != batch_size
            ):
                raise ValueError(
                    "history_time_points_by_graph must contain one row "
                    "per graph."
                )

            history_length = self.temporal_attention.shape[1]

            for graph_index, time_points in enumerate(
                self.history_time_points_by_graph
            ):
                if len(time_points) != history_length:
                    raise ValueError(
                        "Each history-time row must match temporal "
                        f"attention width {history_length}. "
                        f"Graph {graph_index} has {len(time_points)}."
                    )

        if self.pathway_scores is not None:
            _require_float_tensor(
                "pathway_scores",
                self.pathway_scores,
            )
            _require_first_dimension(
                "pathway_scores",
                self.pathway_scores,
                num_nodes,
            )

        if self.layer_index is not None and self.layer_index < 0:
            raise ValueError("layer_index must be nonnegative.")

        _validate_same_device(
            {
                "temporal_attention": self.temporal_attention,
                "pathway_scores": self.pathway_scores,
                "edge_index": self.edge_index,
                "edge_relation_type": self.edge_relation_type,
                "relation_gate_values": (
                    self.relation_gate_output.gate_values
                    if self.relation_gate_output is not None
                    else None
                ),
                "edge_attention": (
                    self.edge_attention_output.attention_weight
                    if self.edge_attention_output is not None
                    else None
                ),
                "alignment.node_batch_index": (
                    self.alignment.node_batch_index
                ),
                "alignment.hazard_ids": self.alignment.hazard_ids,
            }
        )


# =============================================================================
# Typed intermediate states
# =============================================================================


@dataclass(slots=True)
class IntermediateStates:
    node_states: TensorMapping = field(default_factory=dict)
    edge_states: TensorMapping = field(default_factory=dict)
    graph_states: TensorMapping = field(default_factory=dict)
    temporal_states: TensorMapping = field(default_factory=dict)
    auxiliary_states: TensorMapping = field(default_factory=dict)

    def validate(
        self,
        *,
        num_nodes: int,
        num_edges: int,
        num_graphs: int,
    ) -> None:
        _validate_float_tensor_mapping(
            "node_states",
            self.node_states,
            first_dimension=num_nodes,
        )
        _validate_float_tensor_mapping(
            "edge_states",
            self.edge_states,
            first_dimension=num_edges,
        )
        _validate_float_tensor_mapping(
            "graph_states",
            self.graph_states,
            first_dimension=num_graphs,
        )
        _validate_float_tensor_mapping(
            "temporal_states",
            self.temporal_states,
            first_dimension=num_nodes,
        )
        _validate_float_tensor_mapping(
            "auxiliary_states",
            self.auxiliary_states,
        )

        for name, value in self.temporal_states.items():
            if value.ndim < 3:
                raise ValueError(
                    f"temporal_states[{name!r}] must have at least "
                    "shape [N, T, ...]."
                )

        _validate_same_device(self.tensor_items())

    def tensor_items(self) -> dict[str, Tensor]:
        values: dict[str, Tensor] = {}

        for group_name, mapping in (
            ("node", self.node_states),
            ("edge", self.edge_states),
            ("graph", self.graph_states),
            ("temporal", self.temporal_states),
            ("auxiliary", self.auxiliary_states),
        ):
            for key, tensor in mapping.items():
                values[f"{group_name}.{key}"] = tensor

        return values


# =============================================================================
# Full model output
# =============================================================================


@dataclass(slots=True)
class ModelOutput:
    predictions: PredictionOutput

    uncertainty: UncertaintyOutput | None = None
    reporting_bias: ReportingBiasOutput | None = None
    explanations: ExplanationTrace | None = None

    intermediate_states: IntermediateStates = field(
        default_factory=IntermediateStates
    )
    regularization_terms: TensorMapping = field(default_factory=dict)

    num_edges: int = 0

    def __post_init__(self) -> None:
        num_nodes, num_targets = (
            self.predictions.prediction_mean.shape
        )
        num_graphs = len(
            self.predictions.alignment.origin_time
        )

        if self.num_edges < 0:
            raise ValueError("num_edges must be nonnegative.")

        if self.uncertainty is not None:
            self.uncertainty.validate(
                num_nodes=num_nodes,
                num_targets=num_targets,
            )

        if self.reporting_bias is not None:
            self.reporting_bias.validate(
                num_nodes=num_nodes,
                num_targets=num_targets,
            )

        if self.explanations is not None:
            self.explanations.validate(
                num_nodes=num_nodes,
                num_targets=num_targets,
            )
            self.predictions.alignment.assert_equivalent(
                self.explanations.alignment
            )

        self.intermediate_states.validate(
            num_nodes=num_nodes,
            num_edges=self.num_edges,
            num_graphs=num_graphs,
        )

        _validate_regularization_terms(
            "regularization_terms",
            self.regularization_terms,
        )

        self._validate_cross_output_devices()

    def _validate_cross_output_devices(self) -> None:
        tensors: dict[str, Tensor | None] = {
            "predictions.prediction_mean": (
                self.predictions.prediction_mean
            ),
            "predictions.count_rate": self.predictions.count_rate,
            "predictions.ranking_score": (
                self.predictions.ranking_score
            ),
            "alignment.node_batch_index": (
                self.predictions.alignment.node_batch_index
            ),
            "alignment.hazard_ids": (
                self.predictions.alignment.hazard_ids
            ),
        }

        if self.uncertainty is not None:
            tensors.update(
                {
                    "uncertainty.variance": self.uncertainty.variance,
                    "uncertainty.standard_deviation": (
                        self.uncertainty.standard_deviation
                    ),
                    "uncertainty.quantiles": (
                        self.uncertainty.quantiles
                    ),
                    "uncertainty.lower_bound": (
                        self.uncertainty.lower_bound
                    ),
                    "uncertainty.upper_bound": (
                        self.uncertainty.upper_bound
                    ),
                    "uncertainty.confidence_score": (
                        self.uncertainty.confidence_score
                    ),
                }
            )

        if self.reporting_bias is not None:
            tensors.update(
                {
                    "reporting.latent_disruption": (
                        self.reporting_bias.latent_disruption
                    ),
                    "reporting.reporting_propensity": (
                        self.reporting_bias.reporting_propensity
                    ),
                    "reporting.observed_prediction": (
                        self.reporting_bias.observed_prediction
                    ),
                }
            )

        if self.explanations is not None:
            tensors.update(
                {
                    "explanations.temporal_attention": (
                        self.explanations.temporal_attention
                    ),
                    "explanations.pathway_scores": (
                        self.explanations.pathway_scores
                    ),
                    "explanations.edge_index": (
                        self.explanations.edge_index
                    ),
                    "explanations.edge_relation_type": (
                        self.explanations.edge_relation_type
                    ),
                    "explanations.relation_gate_values": (
                        self.explanations.relation_gate_output.gate_values
                        if self.explanations.relation_gate_output
                        is not None
                        else None
                    ),
                    "explanations.edge_attention": (
                        self.explanations.edge_attention_output
                        .attention_weight
                        if self.explanations.edge_attention_output
                        is not None
                        else None
                    ),
                }
            )

        tensors.update(self.intermediate_states.tensor_items())
        tensors.update(dict(self.regularization_terms))

        _validate_same_device(tensors)


__all__ = (
    "AggregationOutput",
    "BatchMetadata",
    "ContractVersions",
    "EdgeAttentionOutput",
    "ExplanationTrace",
    "FunctionalMessagePassingOutput",
    "HazardEncoderOutput",
    "HazardQueriedMemoryOutput",
    "IntermediateStates",
    "MemoryEncoderOutput",
    "MessageBuilderOutput",
    "ModelOutput",
    "PredictionAlignment",
    "PredictionOutput",
    "RelationGateOutput",
    "ReportingBiasOutput",
    "RunMetadata",
    "TemporalAttentionOutput",
    "TemporalMetadata",
    "TypedNodeFeatureBlock",
    "UncertaintyOutput",
    "UrbanGraphBatch",
)