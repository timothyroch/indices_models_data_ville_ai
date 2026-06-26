"""
Neutral provenance and axis-identity contracts for temporal memory.

Target path:
    urban_resilience_models/
        v2_hazard_conditioned_functional_ugnn/
            memory/
                schemas/
                    provenance.py

This module owns the immutable identities shared by every temporal-memory
stage:

- the node axis used by historical inputs and all memory outputs;
- the ordered feature axis of model-facing historical values;
- source-data and preprocessing provenance;
- encoder/pooling/retrieval architecture provenance;
- optional parameter-snapshot provenance;
- execution lineage for one memory transformation;
- a validated computation-provenance bundle.

The module is deliberately neutral with respect to:

- GRU, LSTM, Transformer, baseline, pooling, or retrieval implementations;
- hazard-query implementations;
- node-state fusion;
- graph message passing;
- model configuration dispatch;
- temporal-coordinate tensors, which belong in ``temporal_coordinates.py``;
- historical values and masks, which belong in ``history_inputs.py``.

Import-boundary rule
--------------------
This file must remain upstream of memory encoders and downstream consumers.
It therefore does not import:

- ``fusion.schemas.NodeAlignment``;
- hazard-query schemas;
- recurrent or Transformer implementations;
- the top-level model;
- PyTorch neural modules.

The memory node axis intentionally duplicates no fusion implementation object.
It provides a neutral identity that later fusion code may compare with or
adapt into its own alignment contract without introducing:

    memory -> fusion -> memory

Scientific interpretation
--------------------------
Fingerprints provide deterministic software identities for reproducibility,
alignment checking, and lineage auditing. They do not establish causal
provenance, explanation faithfulness, data quality, or semantic equivalence
between two independently produced artifacts.

Parameter snapshots are optional. Computing a complete parameter hash during
every training forward pass is neither required nor implied by these schemas.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from hashlib import sha256
import json
import math
from types import MappingProxyType
from typing import Any, Final, Mapping, Sequence

import torch


# =============================================================================
# Schema identity
# =============================================================================


TEMPORAL_NODE_AXIS_SCHEMA_VERSION: Final[str] = "0.1"
TEMPORAL_FEATURE_AXIS_SCHEMA_VERSION: Final[str] = "0.1"
MEMORY_SOURCE_PROVENANCE_SCHEMA_VERSION: Final[str] = "0.1"
MEMORY_ARCHITECTURE_PROVENANCE_SCHEMA_VERSION: Final[str] = "0.1"
MEMORY_PARAMETER_SNAPSHOT_PROVENANCE_SCHEMA_VERSION: Final[str] = "0.1"
MEMORY_EXECUTION_LINEAGE_SCHEMA_VERSION: Final[str] = "0.1"
MEMORY_COMPUTATION_PROVENANCE_SCHEMA_VERSION: Final[str] = "0.1"

MEMORY_PARAMETER_SNAPSHOT_POLICY: Final[str] = (
    "optional_artifact_or_evaluation_snapshot"
)

MEMORY_PROVENANCE_SCIENTIFIC_INTERPRETATION: Final[str] = (
    "deterministic_software_identity_not_causal_provenance"
)


# =============================================================================
# Generic validation and fingerprint helpers
# =============================================================================


def _require_nonempty_string(
    name: str,
    value: str,
) -> None:
    if not isinstance(
        value,
        str,
    ):
        raise TypeError(
            f"{name} must be a string."
        )

    if not value.strip():
        raise ValueError(
            f"{name} must be a non-empty string."
        )


def _require_optional_nonempty_string(
    name: str,
    value: str | None,
) -> None:
    if value is not None:
        _require_nonempty_string(
            name,
            value,
        )


def _require_positive_int(
    name: str,
    value: int,
) -> None:
    if (
        isinstance(
            value,
            bool,
        )
        or not isinstance(
            value,
            int,
        )
    ):
        raise TypeError(
            f"{name} must be an integer."
        )

    if value <= 0:
        raise ValueError(
            f"{name} must be strictly positive."
        )


def _require_nonnegative_int(
    name: str,
    value: int,
) -> None:
    if (
        isinstance(
            value,
            bool,
        )
        or not isinstance(
            value,
            int,
        )
    ):
        raise TypeError(
            f"{name} must be an integer."
        )

    if value < 0:
        raise ValueError(
            f"{name} must be nonnegative."
        )


def _require_optional_nonnegative_int(
    name: str,
    value: int | None,
) -> None:
    if value is not None:
        _require_nonnegative_int(
            name,
            value,
        )


def _require_unique_strings(
    name: str,
    values: Sequence[str],
    *,
    require_nonempty_collection: bool = False,
) -> None:
    if (
        require_nonempty_collection
        and not values
    ):
        raise ValueError(
            f"{name} must not be empty."
        )

    observed: set[str] = set()
    duplicates: set[str] = set()

    for index, value in enumerate(
        values
    ):
        _require_nonempty_string(
            f"{name}[{index}]",
            value,
        )

        if value in observed:
            duplicates.add(
                value
            )

        observed.add(
            value
        )

    if duplicates:
        raise ValueError(
            f"{name} contains duplicate values: "
            f"{sorted(duplicates)}."
        )


def _require_long_vector(
    name: str,
    value: torch.Tensor,
    *,
    length: int,
) -> None:
    if not isinstance(
        value,
        torch.Tensor,
    ):
        raise TypeError(
            f"{name} must be a tensor."
        )

    if value.ndim != 1:
        raise ValueError(
            f"{name} must have shape [{length}]; "
            f"observed {tuple(value.shape)}."
        )

    if value.dtype != torch.long:
        raise ValueError(
            f"{name} must use torch.long."
        )

    if int(
        value.shape[0]
    ) != length:
        raise ValueError(
            f"{name} must have length {length}; "
            f"observed {int(value.shape[0])}."
        )


def _to_plain_json_value(
    value: Any,
) -> Any:
    if isinstance(
        value,
        Mapping,
    ):
        return {
            str(key): _to_plain_json_value(
                child
            )
            for key, child in value.items()
        }

    if isinstance(
        value,
        (
            tuple,
            list,
        ),
    ):
        return [
            _to_plain_json_value(
                child
            )
            for child in value
        ]

    return value


def _canonical_json(
    payload: Mapping[str, Any],
) -> str:
    return json.dumps(
        _to_plain_json_value(
            payload
        ),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )


def _fingerprint(
    payload: Mapping[str, Any],
) -> str:
    return sha256(
        _canonical_json(
            payload
        ).encode(
            "utf-8"
        )
    ).hexdigest()


def _tensor_fingerprint(
    tensors: Mapping[
        str,
        torch.Tensor,
    ],
) -> str:
    digest = sha256()

    for name in sorted(
        tensors
    ):
        tensor = (
            tensors[name]
            .detach()
            .cpu()
            .contiguous()
        )

        digest.update(
            name.encode(
                "utf-8"
            )
        )
        digest.update(
            str(
                tensor.dtype
            ).encode(
                "utf-8"
            )
        )
        digest.update(
            json.dumps(
                list(
                    tensor.shape
                ),
                separators=(
                    ",",
                    ":",
                ),
            ).encode(
                "utf-8"
            )
        )
        digest.update(
            tensor.view(
                torch.uint8
            )
            .numpy()
            .tobytes()
        )

    return digest.hexdigest()


def _validate_json_value(
    name: str,
    value: Any,
) -> None:
    if value is None:
        return

    if isinstance(
        value,
        bool,
    ):
        return

    if isinstance(
        value,
        int,
    ):
        return

    if isinstance(
        value,
        float,
    ):
        if not math.isfinite(
            value
        ):
            raise ValueError(
                f"{name} must not contain NaN or infinity."
            )
        return

    if isinstance(
        value,
        str,
    ):
        return

    if isinstance(
        value,
        Mapping,
    ):
        for key, child in (
            value.items()
        ):
            _require_nonempty_string(
                f"{name} key",
                key,
            )
            _validate_json_value(
                f"{name}[{key!r}]",
                child,
            )
        return

    if isinstance(
        value,
        (
            tuple,
            list,
        ),
    ):
        for index, child in enumerate(
            value
        ):
            _validate_json_value(
                f"{name}[{index}]",
                child,
            )
        return

    raise TypeError(
        f"{name} contains unsupported value type "
        f"{type(value).__name__!r}. Expected JSON-compatible "
        "metadata without tensors or implementation objects."
    )


def _freeze_json_value(
    value: Any,
) -> Any:
    if isinstance(
        value,
        Mapping,
    ):
        return MappingProxyType(
            {
                str(key): _freeze_json_value(
                    child
                )
                for key, child
                in value.items()
            }
        )

    if isinstance(
        value,
        (
            tuple,
            list,
        ),
    ):
        return tuple(
            _freeze_json_value(
                child
            )
            for child in value
        )

    return value


def _immutable_json_mapping(
    name: str,
    values: Mapping[
        str,
        Any,
    ],
) -> Mapping[
    str,
    Any,
]:
    if not isinstance(
        values,
        Mapping,
    ):
        raise TypeError(
            f"{name} must be a mapping."
        )

    _validate_json_value(
        name,
        values,
    )

    return _freeze_json_value(
        dict(
            values
        )
    )


def _immutable_string_mapping(
    name: str,
    values: Mapping[
        str,
        str,
    ],
) -> Mapping[
    str,
    str,
]:
    if not isinstance(
        values,
        Mapping,
    ):
        raise TypeError(
            f"{name} must be a mapping."
        )

    copied: dict[
        str,
        str,
    ] = {}

    for key, value in (
        values.items()
    ):
        _require_nonempty_string(
            f"{name} key",
            key,
        )
        _require_nonempty_string(
            f"{name}[{key!r}]",
            value,
        )
        copied[key] = value

    return MappingProxyType(
        copied
    )


# =============================================================================
# Neutral node-axis identity
# =============================================================================


@dataclass(
    slots=True,
    frozen=True,
)
class TemporalNodeAxis:
    """
    Stable node-row and packed-graph identity for temporal memory.

    ``node_ids[index]`` identifies row ``index`` of every node-aligned memory
    tensor.

    ``node_batch_index[index]`` identifies the packed graph containing that
    node. Graph indices must be contiguous from zero and every declared graph
    must contain at least one node.

    ``graph_ids`` is optional because not every data pipeline exposes a stable
    graph/scenario identifier. When supplied, it is ordered by dense packed
    graph index and must have exactly ``graph_count`` unique entries.
    """

    node_ids: tuple[
        str,
        ...,
    ]
    node_batch_index: torch.Tensor
    graph_count: int

    graph_ids: tuple[
        str,
        ...,
    ] = ()

    source_fingerprint: str | None = None
    axis_name: str = (
        "temporal_node_axis"
    )

    schema_version: str = (
        TEMPORAL_NODE_AXIS_SCHEMA_VERSION
    )

    def __post_init__(
        self,
    ) -> None:
        _require_unique_strings(
            "node_ids",
            self.node_ids,
            require_nonempty_collection=True,
        )

        _require_positive_int(
            "graph_count",
            self.graph_count,
        )

        _require_long_vector(
            "node_batch_index",
            self.node_batch_index,
            length=self.node_count,
        )

        minimum = int(
            self
            .node_batch_index
            .min()
            .item()
        )
        maximum = int(
            self
            .node_batch_index
            .max()
            .item()
        )

        if minimum < 0:
            raise ValueError(
                "node_batch_index cannot contain negative graph IDs."
            )

        if maximum >= self.graph_count:
            raise ValueError(
                "node_batch_index contains a graph ID outside "
                "graph_count."
            )

        observed = torch.unique(
            self.node_batch_index,
            sorted=True,
        )
        expected = torch.arange(
            self.graph_count,
            dtype=torch.long,
            device=(
                self
                .node_batch_index
                .device
            ),
        )

        if not torch.equal(
            observed,
            expected,
        ):
            raise ValueError(
                "node_batch_index must contain contiguous graph IDs "
                "from zero and represent every packed graph."
            )

        _require_unique_strings(
            "graph_ids",
            self.graph_ids,
        )

        if (
            self.graph_ids
            and len(
                self.graph_ids
            ) != self.graph_count
        ):
            raise ValueError(
                "graph_ids must be empty or contain exactly one ID "
                "per packed graph."
            )

        _require_optional_nonempty_string(
            "source_fingerprint",
            self.source_fingerprint,
        )
        _require_nonempty_string(
            "axis_name",
            self.axis_name,
        )
        _require_nonempty_string(
            "schema_version",
            self.schema_version,
        )

    @property
    def node_count(
        self,
    ) -> int:
        return len(
            self.node_ids
        )

    @property
    def item_count(
        self,
    ) -> int:
        return self.node_count

    @property
    def device(
        self,
    ) -> torch.device:
        return (
            self
            .node_batch_index
            .device
        )

    @property
    def graph_aligned(
        self,
    ) -> bool:
        return True

    def semantic_dict(
        self,
    ) -> dict[
        str,
        Any,
    ]:
        return {
            "schema_version": (
                self.schema_version
            ),
            "axis_name": (
                self.axis_name
            ),
            "node_count": (
                self.node_count
            ),
            "node_ids": list(
                self.node_ids
            ),
            "graph_count": (
                self.graph_count
            ),
            "graph_ids": list(
                self.graph_ids
            ),
            "source_fingerprint": (
                self.source_fingerprint
            ),
        }

    def semantic_fingerprint(
        self,
    ) -> str:
        return _fingerprint(
            self.semantic_dict()
        )

    def value_fingerprint(
        self,
    ) -> str:
        return _tensor_fingerprint(
            {
                "node_batch_index": (
                    self
                    .node_batch_index
                ),
            }
        )

    def fingerprint(
        self,
    ) -> str:
        return _fingerprint(
            {
                "semantic_fingerprint": (
                    self
                    .semantic_fingerprint()
                ),
                "value_fingerprint": (
                    self
                    .value_fingerprint()
                ),
            }
        )

    def alignment_fingerprint(
        self,
    ) -> str:
        return self.fingerprint()

    def to(
        self,
        device: (
            torch.device
            | str
        ),
    ) -> "TemporalNodeAxis":
        return TemporalNodeAxis(
            node_ids=self.node_ids,
            node_batch_index=(
                self
                .node_batch_index
                .to(
                    device=device
                )
            ),
            graph_count=(
                self.graph_count
            ),
            graph_ids=(
                self.graph_ids
            ),
            source_fingerprint=(
                self.source_fingerprint
            ),
            axis_name=(
                self.axis_name
            ),
            schema_version=(
                self.schema_version
            ),
        )


# =============================================================================
# Ordered feature-axis identity
# =============================================================================


@dataclass(
    slots=True,
    frozen=True,
)
class TemporalFeatureAxis:
    """
    Ordered semantic identity of the historical feature axis.

    Feature order is part of the model contract. Two histories containing the
    same names in a different order therefore have different fingerprints.
    """

    feature_names: tuple[
        str,
        ...,
    ]

    source_fingerprint: str | None = None
    axis_name: str = (
        "temporal_feature_axis"
    )

    schema_version: str = (
        TEMPORAL_FEATURE_AXIS_SCHEMA_VERSION
    )

    def __post_init__(
        self,
    ) -> None:
        _require_unique_strings(
            "feature_names",
            self.feature_names,
            require_nonempty_collection=True,
        )
        _require_optional_nonempty_string(
            "source_fingerprint",
            self.source_fingerprint,
        )
        _require_nonempty_string(
            "axis_name",
            self.axis_name,
        )
        _require_nonempty_string(
            "schema_version",
            self.schema_version,
        )

    @property
    def feature_dim(
        self,
    ) -> int:
        return len(
            self.feature_names
        )

    def semantic_dict(
        self,
    ) -> dict[
        str,
        Any,
    ]:
        return {
            "schema_version": (
                self.schema_version
            ),
            "axis_name": (
                self.axis_name
            ),
            "feature_dim": (
                self.feature_dim
            ),
            "feature_names": list(
                self.feature_names
            ),
            "source_fingerprint": (
                self.source_fingerprint
            ),
        }

    def semantic_fingerprint(
        self,
    ) -> str:
        return _fingerprint(
            self.semantic_dict()
        )

    def fingerprint(
        self,
    ) -> str:
        return _fingerprint(
            {
                "semantic_fingerprint": (
                    self
                    .semantic_fingerprint()
                ),
            }
        )


# =============================================================================
# Source-data and preprocessing provenance
# =============================================================================


@dataclass(
    slots=True,
    frozen=True,
)
class MemorySourceProvenance:
    """
    Stable identity of the data and preprocessing used to build one history.

    The required ``source_fingerprint`` is the authoritative upstream artifact
    identity. More specific fingerprints are optional because some pipelines
    expose only one composite identity.

    ``upstream_fingerprints`` supports named immutable lineage such as:

    - raw panel snapshot;
    - feature table;
    - temporal-window extraction;
    - node-index artifact;
    - external imputation model.

    The mapping is descriptive and must contain only non-empty string keys and
    values.
    """

    source_name: str
    source_kind: str
    source_fingerprint: str

    dataset_version: str | None = None
    dataset_snapshot_fingerprint: (
        str
        | None
    ) = None
    extraction_fingerprint: (
        str
        | None
    ) = None
    preprocessing_fingerprint: (
        str
        | None
    ) = None
    imputation_fingerprint: (
        str
        | None
    ) = None

    upstream_fingerprints: Mapping[
        str,
        str,
    ] = field(
        default_factory=dict
    )

    schema_version: str = (
        MEMORY_SOURCE_PROVENANCE_SCHEMA_VERSION
    )

    def __post_init__(
        self,
    ) -> None:
        _require_nonempty_string(
            "source_name",
            self.source_name,
        )
        _require_nonempty_string(
            "source_kind",
            self.source_kind,
        )
        _require_nonempty_string(
            "source_fingerprint",
            self.source_fingerprint,
        )

        for name, value in (
            (
                "dataset_version",
                self.dataset_version,
            ),
            (
                "dataset_snapshot_fingerprint",
                self
                .dataset_snapshot_fingerprint,
            ),
            (
                "extraction_fingerprint",
                self
                .extraction_fingerprint,
            ),
            (
                "preprocessing_fingerprint",
                self
                .preprocessing_fingerprint,
            ),
            (
                "imputation_fingerprint",
                self
                .imputation_fingerprint,
            ),
        ):
            _require_optional_nonempty_string(
                name,
                value,
            )

        object.__setattr__(
            self,
            "upstream_fingerprints",
            _immutable_string_mapping(
                "upstream_fingerprints",
                self
                .upstream_fingerprints,
            ),
        )

        _require_nonempty_string(
            "schema_version",
            self.schema_version,
        )

    def provenance_dict(
        self,
    ) -> dict[
        str,
        Any,
    ]:
        return {
            "schema_version": (
                self.schema_version
            ),
            "source_name": (
                self.source_name
            ),
            "source_kind": (
                self.source_kind
            ),
            "source_fingerprint": (
                self.source_fingerprint
            ),
            "dataset_version": (
                self.dataset_version
            ),
            "dataset_snapshot_fingerprint": (
                self
                .dataset_snapshot_fingerprint
            ),
            "extraction_fingerprint": (
                self
                .extraction_fingerprint
            ),
            "preprocessing_fingerprint": (
                self
                .preprocessing_fingerprint
            ),
            "imputation_fingerprint": (
                self
                .imputation_fingerprint
            ),
            "upstream_fingerprints": dict(
                self
                .upstream_fingerprints
            ),
        }

    def fingerprint(
        self,
    ) -> str:
        return _fingerprint(
            self.provenance_dict()
        )

    def lineage_fingerprint(
        self,
    ) -> str:
        return self.fingerprint()


# =============================================================================
# Architecture identity
# =============================================================================


@dataclass(
    slots=True,
    frozen=True,
)
class MemoryArchitectureProvenance:
    """
    Stable identity of one configured temporal-memory operation.

    This contract describes architecture and operation semantics, not current
    parameter values.
    """

    component_name: str
    component_kind: str
    architecture_fingerprint: str

    configuration_fingerprint: (
        str
        | None
    ) = None
    implementation_version: (
        str
        | None
    ) = None

    architecture_metadata: Mapping[
        str,
        Any,
    ] = field(
        default_factory=dict
    )

    schema_version: str = (
        MEMORY_ARCHITECTURE_PROVENANCE_SCHEMA_VERSION
    )

    def __post_init__(
        self,
    ) -> None:
        _require_nonempty_string(
            "component_name",
            self.component_name,
        )
        _require_nonempty_string(
            "component_kind",
            self.component_kind,
        )
        _require_nonempty_string(
            "architecture_fingerprint",
            self.architecture_fingerprint,
        )
        _require_optional_nonempty_string(
            "configuration_fingerprint",
            self.configuration_fingerprint,
        )
        _require_optional_nonempty_string(
            "implementation_version",
            self.implementation_version,
        )

        object.__setattr__(
            self,
            "architecture_metadata",
            _immutable_json_mapping(
                "architecture_metadata",
                self
                .architecture_metadata,
            ),
        )

        _require_nonempty_string(
            "schema_version",
            self.schema_version,
        )

    def provenance_dict(
        self,
    ) -> dict[
        str,
        Any,
    ]:
        return {
            "schema_version": (
                self.schema_version
            ),
            "component_name": (
                self.component_name
            ),
            "component_kind": (
                self.component_kind
            ),
            "architecture_fingerprint": (
                self
                .architecture_fingerprint
            ),
            "configuration_fingerprint": (
                self
                .configuration_fingerprint
            ),
            "implementation_version": (
                self
                .implementation_version
            ),
            "architecture_metadata": (
                _to_plain_json_value(
                    self
                    .architecture_metadata
                )
            ),
        }

    def fingerprint(
        self,
    ) -> str:
        return _fingerprint(
            self.provenance_dict()
        )


# =============================================================================
# Optional parameter-snapshot identity
# =============================================================================


@dataclass(
    slots=True,
    frozen=True,
)
class MemoryParameterSnapshotProvenance:
    """
    Optional identity of one concrete parameter or checkpoint snapshot.

    Construct this object for artifact generation, evaluation, export, or
    checkpoint-specific auditing. Training-time outputs may omit it.
    """

    parameter_snapshot_fingerprint: str

    checkpoint_id: str | None = None
    checkpoint_fingerprint: (
        str
        | None
    ) = None
    training_step: int | None = None

    parameter_count: int | None = None
    trainable_parameter_count: (
        int
        | None
    ) = None

    schema_version: str = (
        MEMORY_PARAMETER_SNAPSHOT_PROVENANCE_SCHEMA_VERSION
    )

    def __post_init__(
        self,
    ) -> None:
        _require_nonempty_string(
            "parameter_snapshot_fingerprint",
            self
            .parameter_snapshot_fingerprint,
        )
        _require_optional_nonempty_string(
            "checkpoint_id",
            self.checkpoint_id,
        )
        _require_optional_nonempty_string(
            "checkpoint_fingerprint",
            self
            .checkpoint_fingerprint,
        )
        _require_optional_nonnegative_int(
            "training_step",
            self.training_step,
        )
        _require_optional_nonnegative_int(
            "parameter_count",
            self.parameter_count,
        )
        _require_optional_nonnegative_int(
            "trainable_parameter_count",
            self
            .trainable_parameter_count,
        )

        if (
            self.parameter_count
            is not None
            and self
            .trainable_parameter_count
            is not None
            and self
            .trainable_parameter_count
            > self.parameter_count
        ):
            raise ValueError(
                "trainable_parameter_count cannot exceed "
                "parameter_count."
            )

        _require_nonempty_string(
            "schema_version",
            self.schema_version,
        )

    def provenance_dict(
        self,
    ) -> dict[
        str,
        Any,
    ]:
        return {
            "schema_version": (
                self.schema_version
            ),
            "parameter_snapshot_fingerprint": (
                self
                .parameter_snapshot_fingerprint
            ),
            "checkpoint_id": (
                self.checkpoint_id
            ),
            "checkpoint_fingerprint": (
                self
                .checkpoint_fingerprint
            ),
            "training_step": (
                self.training_step
            ),
            "parameter_count": (
                self.parameter_count
            ),
            "trainable_parameter_count": (
                self
                .trainable_parameter_count
            ),
            "snapshot_policy": (
                MEMORY_PARAMETER_SNAPSHOT_POLICY
            ),
        }

    def fingerprint(
        self,
    ) -> str:
        return _fingerprint(
            self.provenance_dict()
        )


# =============================================================================
# Execution lineage
# =============================================================================


@dataclass(
    slots=True,
    frozen=True,
)
class MemoryExecutionLineage:
    """
    Deterministic lineage of one temporal-memory transformation.

    ``source_lineage_fingerprints`` identifies the exact upstream contracts.
    Most operations have one source. Fusion or assembly operations may have
    several.

    Axis fingerprints are optional at this neutral level because not every
    operation owns every axis. Concrete schemas may require the appropriate
    subset.
    """

    operation_name: str
    source_lineage_fingerprints: tuple[
        str,
        ...,
    ]

    architecture_fingerprint: (
        str
        | None
    ) = None
    parameter_snapshot_fingerprint: (
        str
        | None
    ) = None
    configuration_fingerprint: (
        str
        | None
    ) = None

    node_axis_fingerprint: (
        str
        | None
    ) = None
    temporal_axis_fingerprint: (
        str
        | None
    ) = None
    feature_axis_fingerprint: (
        str
        | None
    ) = None

    lineage_metadata: Mapping[
        str,
        Any,
    ] = field(
        default_factory=dict
    )

    schema_version: str = (
        MEMORY_EXECUTION_LINEAGE_SCHEMA_VERSION
    )

    def __post_init__(
        self,
    ) -> None:
        _require_nonempty_string(
            "operation_name",
            self.operation_name,
        )
        _require_unique_strings(
            "source_lineage_fingerprints",
            self
            .source_lineage_fingerprints,
            require_nonempty_collection=True,
        )

        for name, value in (
            (
                "architecture_fingerprint",
                self
                .architecture_fingerprint,
            ),
            (
                "parameter_snapshot_fingerprint",
                self
                .parameter_snapshot_fingerprint,
            ),
            (
                "configuration_fingerprint",
                self
                .configuration_fingerprint,
            ),
            (
                "node_axis_fingerprint",
                self
                .node_axis_fingerprint,
            ),
            (
                "temporal_axis_fingerprint",
                self
                .temporal_axis_fingerprint,
            ),
            (
                "feature_axis_fingerprint",
                self
                .feature_axis_fingerprint,
            ),
        ):
            _require_optional_nonempty_string(
                name,
                value,
            )

        object.__setattr__(
            self,
            "lineage_metadata",
            _immutable_json_mapping(
                "lineage_metadata",
                self
                .lineage_metadata,
            ),
        )

        _require_nonempty_string(
            "schema_version",
            self.schema_version,
        )

    def lineage_dict(
        self,
    ) -> dict[
        str,
        Any,
    ]:
        return {
            "schema_version": (
                self.schema_version
            ),
            "operation_name": (
                self.operation_name
            ),
            "source_lineage_fingerprints": list(
                self
                .source_lineage_fingerprints
            ),
            "architecture_fingerprint": (
                self
                .architecture_fingerprint
            ),
            "parameter_snapshot_fingerprint": (
                self
                .parameter_snapshot_fingerprint
            ),
            "configuration_fingerprint": (
                self
                .configuration_fingerprint
            ),
            "node_axis_fingerprint": (
                self
                .node_axis_fingerprint
            ),
            "temporal_axis_fingerprint": (
                self
                .temporal_axis_fingerprint
            ),
            "feature_axis_fingerprint": (
                self
                .feature_axis_fingerprint
            ),
            "lineage_metadata": (
                _to_plain_json_value(
                    self
                    .lineage_metadata
                )
            ),
        }

    def lineage_fingerprint(
        self,
    ) -> str:
        return _fingerprint(
            self.lineage_dict()
        )

    def fingerprint(
        self,
    ) -> str:
        return self.lineage_fingerprint()


# =============================================================================
# Validated computation-provenance bundle
# =============================================================================


@dataclass(
    slots=True,
    frozen=True,
)
class MemoryComputationProvenance:
    """
    Architecture, optional parameter snapshot, and execution lineage.

    The bundle prevents a downstream output from carrying mutually
    inconsistent provenance objects.
    """

    architecture: (
        MemoryArchitectureProvenance
    )
    lineage: MemoryExecutionLineage

    parameter_snapshot: (
        MemoryParameterSnapshotProvenance
        | None
    ) = None

    schema_version: str = (
        MEMORY_COMPUTATION_PROVENANCE_SCHEMA_VERSION
    )

    def __post_init__(
        self,
    ) -> None:
        if not isinstance(
            self.architecture,
            MemoryArchitectureProvenance,
        ):
            raise TypeError(
                "architecture must be a "
                "MemoryArchitectureProvenance."
            )

        if not isinstance(
            self.lineage,
            MemoryExecutionLineage,
        ):
            raise TypeError(
                "lineage must be a MemoryExecutionLineage."
            )

        if (
            self.parameter_snapshot
            is not None
            and not isinstance(
                self.parameter_snapshot,
                MemoryParameterSnapshotProvenance,
            )
        ):
            raise TypeError(
                "parameter_snapshot must be a "
                "MemoryParameterSnapshotProvenance or None."
            )

        if (
            self.lineage
            .architecture_fingerprint
            != self.architecture
            .architecture_fingerprint
        ):
            raise ValueError(
                "lineage.architecture_fingerprint must exactly match "
                "architecture.architecture_fingerprint."
            )

        if (
            self.architecture
            .configuration_fingerprint
            is not None
            and self.lineage
            .configuration_fingerprint
            is not None
            and self.architecture
            .configuration_fingerprint
            != self.lineage
            .configuration_fingerprint
        ):
            raise ValueError(
                "Architecture and lineage configuration fingerprints "
                "must match when both are supplied."
            )

        if (
            self.parameter_snapshot
            is None
        ):
            if (
                self.lineage
                .parameter_snapshot_fingerprint
                is not None
            ):
                raise ValueError(
                    "lineage cannot declare a parameter snapshot when "
                    "parameter_snapshot is None."
                )
        else:
            if (
                self.lineage
                .parameter_snapshot_fingerprint
                != self
                .parameter_snapshot
                .parameter_snapshot_fingerprint
            ):
                raise ValueError(
                    "lineage.parameter_snapshot_fingerprint must "
                    "exactly match the supplied parameter snapshot."
                )

        _require_nonempty_string(
            "schema_version",
            self.schema_version,
        )

    @property
    def architecture_fingerprint(
        self,
    ) -> str:
        return (
            self
            .architecture
            .architecture_fingerprint
        )

    @property
    def parameter_snapshot_fingerprint(
        self,
    ) -> str | None:
        if (
            self.parameter_snapshot
            is None
        ):
            return None

        return (
            self
            .parameter_snapshot
            .parameter_snapshot_fingerprint
        )

    @property
    def lineage_fingerprint(
        self,
    ) -> str:
        return (
            self
            .lineage
            .lineage_fingerprint()
        )

    def provenance_dict(
        self,
    ) -> dict[
        str,
        Any,
    ]:
        return {
            "schema_version": (
                self.schema_version
            ),
            "architecture": (
                self
                .architecture
                .provenance_dict()
            ),
            "parameter_snapshot": (
                self
                .parameter_snapshot
                .provenance_dict()
                if (
                    self.parameter_snapshot
                    is not None
                )
                else None
            ),
            "lineage": (
                self
                .lineage
                .lineage_dict()
            ),
            "scientific_interpretation": (
                MEMORY_PROVENANCE_SCIENTIFIC_INTERPRETATION
            ),
        }

    def fingerprint(
        self,
    ) -> str:
        return _fingerprint(
            self.provenance_dict()
        )


# =============================================================================
# Compact aliases
# =============================================================================


NodeAxisIdentity = TemporalNodeAxis
FeatureAxisIdentity = TemporalFeatureAxis
SourceDataProvenance = MemorySourceProvenance
ArchitectureProvenance = MemoryArchitectureProvenance
ParameterSnapshotProvenance = (
    MemoryParameterSnapshotProvenance
)
ExecutionLineage = MemoryExecutionLineage
ComputationProvenance = MemoryComputationProvenance


# =============================================================================
# Public API
# =============================================================================


__all__ = (
    # Schema versions.
    "TEMPORAL_NODE_AXIS_SCHEMA_VERSION",
    "TEMPORAL_FEATURE_AXIS_SCHEMA_VERSION",
    "MEMORY_SOURCE_PROVENANCE_SCHEMA_VERSION",
    "MEMORY_ARCHITECTURE_PROVENANCE_SCHEMA_VERSION",
    "MEMORY_PARAMETER_SNAPSHOT_PROVENANCE_SCHEMA_VERSION",
    "MEMORY_EXECUTION_LINEAGE_SCHEMA_VERSION",
    "MEMORY_COMPUTATION_PROVENANCE_SCHEMA_VERSION",

    # Interpretation and policy.
    "MEMORY_PARAMETER_SNAPSHOT_POLICY",
    "MEMORY_PROVENANCE_SCIENTIFIC_INTERPRETATION",

    # Axis identities.
    "TemporalNodeAxis",
    "TemporalFeatureAxis",
    "NodeAxisIdentity",
    "FeatureAxisIdentity",

    # Provenance contracts.
    "MemorySourceProvenance",
    "MemoryArchitectureProvenance",
    "MemoryParameterSnapshotProvenance",
    "MemoryExecutionLineage",
    "MemoryComputationProvenance",

    # Compact provenance aliases.
    "SourceDataProvenance",
    "ArchitectureProvenance",
    "ParameterSnapshotProvenance",
    "ExecutionLineage",
    "ComputationProvenance",
)
