"""
Metadata-preserving contracts for functional message passing.

Target path:
    urban_resilience_models/
        v2_hazard_conditioned_functional_ugnn/
            functional_message_passing/
                schemas.py

This module owns only immutable data contracts, validation, alignment views,
and reproducibility metadata for the functional message-passing subsystem.

It does not own:

- trainable relation transforms;
- segment/scatter implementations;
- structural-normalization mathematics;
- gate networks or prior integration;
- attention score functions or grouped softmax;
- message construction;
- aggregation;
- residual updates;
- multi-layer orchestration.

Bounded V2.0 decisions
----------------------
The initial implementation uses the exact compiled relation axis ``R`` for
relation gates. Semantic ontology families are preserved through
``RelationFamilyAlignment`` but do not create a second trainable gate axis.

The configured attention-normalization contract
``target_node_relation`` means exact grouping by:

    target node + dense compiled relation index

Graph-scoped hazard queries are expanded explicitly through
``node_batch_index`` at this input boundary. The complete
``HazardQueryEncoding`` remains attached for lineage and diagnostics.

The public FMP boundary consumes complete source contracts:

- ``UrbanGraphBatch``;
- ``NodeStateFusionOutput``;
- ``CompiledRelationRegistry``;
- optional ``RelationRegistry``-derived family alignment;
- optional ``HazardQueryEncoding``;
- optional ``CompiledHazardRelationPriors``.

Stable ontology IDs are never interpreted as dense runtime relation indices.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from hashlib import sha256
import json
import math
from types import MappingProxyType
from typing import Any, Final, Mapping, Sequence

import torch

from ..constants import (
    AGGREGATION_MEAN,
    ATTENTION_HEAD_REDUCTION_MEAN,
    ATTENTION_NORMALIZATION_TARGET_NODE_RELATION,
    EDGE_NORMALIZATION_NONE,
    RELATION_GATE_ACTIVATION_SIGMOID,
    RELATION_GATE_SCOPE_TARGET_NODE,
)
from ..fusion.schemas import (
    NodeAlignment,
    NodeStateFusionOutput,
)
from ..hazard.hazard_embeddings import (
    HazardEmbeddingLookup,
    NodeAlignedHazardEmbeddingLookup,
)
from ..hazard.hazard_query_encoder import HazardQueryEncoding
from ..relations.hazard_relation_priors import (
    CompiledHazardRelationPriors,
)
from ..relations.relation_registry import (
    CompiledRelationRegistry,
    RelationRegistry,
)
from ..schemas import UrbanGraphBatch


# =============================================================================
# Schema identity
# =============================================================================


RELATION_FAMILY_ALIGNMENT_SCHEMA_VERSION: Final[str] = "0.1"
FUNCTIONAL_MESSAGE_PASSING_NODE_STATE_SCHEMA_VERSION: Final[str] = "0.1"
FUNCTIONAL_MESSAGE_PASSING_INPUT_SCHEMA_VERSION: Final[str] = "0.1"
RELATION_TRANSFORM_OUTPUT_SCHEMA_VERSION: Final[str] = "0.1"
EDGE_NORMALIZATION_OUTPUT_SCHEMA_VERSION: Final[str] = "0.1"
RELATION_GATE_OUTPUT_SCHEMA_VERSION: Final[str] = "0.1"
EDGE_ATTENTION_OUTPUT_SCHEMA_VERSION: Final[str] = "0.1"
EDGE_MESSAGE_OUTPUT_SCHEMA_VERSION: Final[str] = "0.1"
AGGREGATION_OUTPUT_SCHEMA_VERSION: Final[str] = "0.1"
FMP_INTERMEDIATES_SCHEMA_VERSION: Final[str] = "0.1"
FMP_LAYER_OUTPUT_SCHEMA_VERSION: Final[str] = "0.1"
FMP_STACK_OUTPUT_SCHEMA_VERSION: Final[str] = "0.1"


# =============================================================================
# Generic helpers
# =============================================================================


def _require_nonempty_string(
    name: str,
    value: str,
) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(
            f"{name} must be a non-empty string."
        )


def _require_optional_nonempty_string(
    name: str,
    value: str | None,
) -> None:
    if value is not None:
        _require_nonempty_string(name, value)


def _require_positive_int(
    name: str,
    value: int,
) -> None:
    if (
        isinstance(value, bool)
        or not isinstance(value, int)
        or value <= 0
    ):
        raise ValueError(
            f"{name} must be a positive integer."
        )


def _require_nonnegative_int(
    name: str,
    value: int,
) -> None:
    if (
        isinstance(value, bool)
        or not isinstance(value, int)
        or value < 0
    ):
        raise ValueError(
            f"{name} must be a nonnegative integer."
        )


def _require_unique_strings(
    name: str,
    values: Sequence[str],
) -> None:
    seen: set[str] = set()
    duplicates: set[str] = set()

    for index, value in enumerate(values):
        _require_nonempty_string(
            f"{name}[{index}]",
            value,
        )
        if value in seen:
            duplicates.add(value)
        seen.add(value)

    if duplicates:
        raise ValueError(
            f"{name} contains duplicates: "
            f"{sorted(duplicates)}."
        )


def _require_unique_ints(
    name: str,
    values: Sequence[int],
) -> None:
    seen: set[int] = set()
    duplicates: set[int] = set()

    for index, value in enumerate(values):
        if (
            isinstance(value, bool)
            or not isinstance(value, int)
        ):
            raise TypeError(
                f"{name}[{index}] must be an integer."
            )
        if value < 0:
            raise ValueError(
                f"{name}[{index}] must be nonnegative."
            )
        if value in seen:
            duplicates.add(value)
        seen.add(value)

    if duplicates:
        raise ValueError(
            f"{name} contains duplicates: "
            f"{sorted(duplicates)}."
        )


def _require_tensor(
    name: str,
    value: torch.Tensor,
    *,
    ndim: int | None = None,
) -> None:
    if not isinstance(value, torch.Tensor):
        raise TypeError(
            f"{name} must be a tensor."
        )

    if ndim is not None and value.ndim != ndim:
        raise ValueError(
            f"{name} must have rank {ndim}; "
            f"observed shape {tuple(value.shape)}."
        )


def _require_float_tensor(
    name: str,
    value: torch.Tensor,
    *,
    ndim: int | None = None,
) -> None:
    _require_tensor(
        name,
        value,
        ndim=ndim,
    )

    if not value.dtype.is_floating_point:
        raise ValueError(
            f"{name} must use a floating-point dtype."
        )

    _assert_finite_tensor(name, value)


def _require_long_tensor(
    name: str,
    value: torch.Tensor,
    *,
    ndim: int | None = None,
) -> None:
    _require_tensor(
        name,
        value,
        ndim=ndim,
    )

    if value.dtype != torch.long:
        raise ValueError(
            f"{name} must use torch.long."
        )


def _require_bool_tensor(
    name: str,
    value: torch.Tensor,
    *,
    ndim: int | None = None,
) -> None:
    _require_tensor(
        name,
        value,
        ndim=ndim,
    )

    if value.dtype != torch.bool:
        raise ValueError(
            f"{name} must use torch.bool."
        )


def _assert_finite_tensor(
    name: str,
    value: torch.Tensor,
) -> None:
    if (
        value.dtype.is_floating_point
        and not bool(
            torch.isfinite(value)
            .all()
            .item()
        )
    ):
        raise ValueError(
            f"{name} must contain only finite values."
        )


def _require_shape(
    name: str,
    value: torch.Tensor,
    expected: tuple[int, ...],
) -> None:
    if tuple(value.shape) != expected:
        raise ValueError(
            f"{name} must have shape {expected}; "
            f"observed {tuple(value.shape)}."
        )


def _require_same_device(
    tensors: Mapping[str, torch.Tensor | None],
) -> torch.device | None:
    observed: dict[str, torch.device] = {
        name: value.device
        for name, value in tensors.items()
        if value is not None
    }

    if not observed:
        return None

    devices = set(observed.values())

    if len(devices) != 1:
        details = ", ".join(
            f"{name}={device}"
            for name, device in observed.items()
        )
        raise ValueError(
            "All tensors must share one device. "
            f"Observed: {details}."
        )

    return next(iter(devices))


def _require_same_float_dtype(
    tensors: Mapping[str, torch.Tensor | None],
) -> torch.dtype | None:
    observed: dict[str, torch.dtype] = {
        name: value.dtype
        for name, value in tensors.items()
        if (
            value is not None
            and value.dtype.is_floating_point
        )
    }

    if not observed:
        return None

    dtypes = set(observed.values())

    if len(dtypes) != 1:
        details = ", ".join(
            f"{name}={dtype}"
            for name, dtype in observed.items()
        )
        raise ValueError(
            "All floating-point FMP tensors must share one dtype. "
            f"Observed: {details}."
        )

    return next(iter(dtypes))


def _require_index_range(
    name: str,
    values: torch.Tensor,
    *,
    upper_bound: int,
) -> None:
    _require_nonnegative_int(
        "upper_bound",
        upper_bound,
    )

    if values.numel() == 0:
        return

    minimum = int(values.min().item())
    maximum = int(values.max().item())

    if minimum < 0 or maximum >= upper_bound:
        valid = (
            "empty"
            if upper_bound == 0
            else f"[0, {upper_bound - 1}]"
        )
        raise ValueError(
            f"{name} contains out-of-range indices. "
            f"Observed range [{minimum}, {maximum}]; "
            f"valid range is {valid}."
        )


def _canonical_json(
    payload: Mapping[str, Any],
) -> str:
    return json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )


def _fingerprint(
    payload: Mapping[str, Any],
) -> str:
    return sha256(
        _canonical_json(payload).encode("utf-8")
    ).hexdigest()


def _tensor_fingerprint(
    tensors: Mapping[str, torch.Tensor],
) -> str:
    digest = sha256()

    for name in sorted(tensors):
        tensor = (
            tensors[name]
            .detach()
            .cpu()
            .contiguous()
        )

        digest.update(
            name.encode("utf-8")
        )
        digest.update(
            str(tensor.dtype).encode("utf-8")
        )
        digest.update(
            json.dumps(
                list(tensor.shape),
                separators=(",", ":"),
            ).encode("utf-8")
        )
        digest.update(
            tensor.view(torch.uint8)
            .numpy()
            .tobytes()
        )

    return digest.hexdigest()


def _immutable_tensor_mapping(
    name: str,
    values: Mapping[str, torch.Tensor],
    *,
    scalar_only: bool = False,
    device: torch.device | None = None,
) -> Mapping[str, torch.Tensor]:
    if not isinstance(values, Mapping):
        raise TypeError(
            f"{name} must be a mapping."
        )

    copied: dict[str, torch.Tensor] = {}

    for key, tensor in values.items():
        _require_nonempty_string(
            f"{name} key",
            key,
        )
        _require_float_tensor(
            f"{name}[{key!r}]",
            tensor,
        )

        if scalar_only and tensor.numel() != 1:
            raise ValueError(
                f"{name}[{key!r}] must contain exactly one value."
            )

        if device is not None and tensor.device != device:
            raise ValueError(
                f"{name}[{key!r}] must be on {device}; "
                f"observed {tensor.device}."
            )

        copied[key] = tensor

    return MappingProxyType(copied)


def _default_tolerances(
    dtype: torch.dtype,
) -> tuple[float, float]:
    if dtype in (
        torch.float16,
        torch.bfloat16,
    ):
        return 5e-3, 5e-3

    if dtype == torch.float64:
        return 1e-10, 1e-10

    return 1e-5, 1e-5


# =============================================================================
# Semantic relation-family alignment
# =============================================================================


@dataclass(slots=True, frozen=True)
class RelationFamilyAlignment:
    """
    Semantic root-family metadata aligned to a compiled relation registry.

    The first V2.0 trainable gate axis remains the exact compiled relation
    axis. This object preserves ontology-family identity for diagnostics,
    ablations, and future hierarchical models.

    ``relation_family_index_by_relation[r]`` maps dense relation index ``r``
    to a dense family index. Families are ordered by ascending stable root
    relation ID.
    """

    family_names: tuple[str, ...]
    stable_family_ids: tuple[int, ...]

    relation_family_index_by_relation: torch.Tensor

    relation_names: tuple[str, ...]
    stable_relation_ids: tuple[int, ...]

    source_relation_registry_fingerprint: str
    compiled_relation_registry_fingerprint: str

    schema_version: str = (
        RELATION_FAMILY_ALIGNMENT_SCHEMA_VERSION
    )

    def __post_init__(self) -> None:
        _require_unique_strings(
            "family_names",
            self.family_names,
        )
        _require_unique_ints(
            "stable_family_ids",
            self.stable_family_ids,
        )
        _require_unique_strings(
            "relation_names",
            self.relation_names,
        )
        _require_unique_ints(
            "stable_relation_ids",
            self.stable_relation_ids,
        )

        if not self.family_names:
            raise ValueError(
                "At least one relation family is required."
            )

        if len(self.family_names) != len(
            self.stable_family_ids
        ):
            raise ValueError(
                "family_names and stable_family_ids must align."
            )

        if len(self.relation_names) != len(
            self.stable_relation_ids
        ):
            raise ValueError(
                "relation_names and stable_relation_ids must align."
            )

        _require_long_tensor(
            "relation_family_index_by_relation",
            self.relation_family_index_by_relation,
            ndim=1,
        )
        _require_shape(
            "relation_family_index_by_relation",
            self.relation_family_index_by_relation,
            (len(self.relation_names),),
        )
        _require_index_range(
            "relation_family_index_by_relation",
            self.relation_family_index_by_relation,
            upper_bound=len(self.family_names),
        )

        observed = set(
            int(value)
            for value in (
                self.relation_family_index_by_relation
                .detach()
                .cpu()
                .tolist()
            )
        )
        expected = set(
            range(len(self.family_names))
        )

        if observed != expected:
            raise ValueError(
                "Every declared relation family must be represented by "
                "at least one compiled relation."
            )

        _require_nonempty_string(
            "source_relation_registry_fingerprint",
            self.source_relation_registry_fingerprint,
        )
        _require_nonempty_string(
            "compiled_relation_registry_fingerprint",
            self.compiled_relation_registry_fingerprint,
        )
        _require_nonempty_string(
            "schema_version",
            self.schema_version,
        )

    @classmethod
    def from_registries(
        cls,
        *,
        source_registry: RelationRegistry,
        compiled_registry: CompiledRelationRegistry,
        device: torch.device | str | None = None,
    ) -> "RelationFamilyAlignment":
        if not isinstance(
            source_registry,
            RelationRegistry,
        ):
            raise TypeError(
                "source_registry must be a RelationRegistry."
            )

        if not isinstance(
            compiled_registry,
            CompiledRelationRegistry,
        ):
            raise TypeError(
                "compiled_registry must be a CompiledRelationRegistry."
            )

        compiled_registry.assert_matches_source_registry(
            source_registry,
            require_operational_match=False,
        )

        roots_by_relation: list[
            tuple[int, str]
        ] = []

        for entry in compiled_registry.entries:
            ancestors = tuple(
                source_registry.ancestors_of(
                    entry.name
                )
            )
            root = (
                ancestors[-1]
                if ancestors
                else source_registry.get_entry_by_name(
                    entry.name
                )
            )
            roots_by_relation.append(
                (
                    root.relation_id,
                    root.name,
                )
            )

        unique_roots = tuple(
            sorted(
                set(roots_by_relation),
                key=lambda pair: pair[0],
            )
        )
        family_index_by_identity = {
            identity: index
            for index, identity in enumerate(
                unique_roots
            )
        }

        mapping = torch.tensor(
            [
                family_index_by_identity[
                    identity
                ]
                for identity in roots_by_relation
            ],
            dtype=torch.long,
            device=device,
        )

        return cls(
            family_names=tuple(
                name
                for _, name in unique_roots
            ),
            stable_family_ids=tuple(
                relation_id
                for relation_id, _ in unique_roots
            ),
            relation_family_index_by_relation=mapping,
            relation_names=(
                compiled_registry.relation_names
            ),
            stable_relation_ids=(
                compiled_registry.stable_relation_ids
            ),
            source_relation_registry_fingerprint=(
                source_registry.semantic_fingerprint()
            ),
            compiled_relation_registry_fingerprint=(
                compiled_registry.fingerprint()
            ),
        )

    @property
    def num_relations(self) -> int:
        return len(self.relation_names)

    @property
    def num_families(self) -> int:
        return len(self.family_names)

    @property
    def device(self) -> torch.device:
        return (
            self.relation_family_index_by_relation
            .device
        )

    def semantic_dict(
        self,
    ) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "family_names": list(
                self.family_names
            ),
            "stable_family_ids": list(
                self.stable_family_ids
            ),
            "relation_names": list(
                self.relation_names
            ),
            "stable_relation_ids": list(
                self.stable_relation_ids
            ),
            "source_relation_registry_fingerprint": (
                self
                .source_relation_registry_fingerprint
            ),
            "compiled_relation_registry_fingerprint": (
                self
                .compiled_relation_registry_fingerprint
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
                "relation_family_index_by_relation": (
                    self
                    .relation_family_index_by_relation
                )
            }
        )

    def fingerprint(
        self,
    ) -> str:
        return _fingerprint(
            {
                "semantic_fingerprint": (
                    self.semantic_fingerprint()
                ),
                "value_fingerprint": (
                    self.value_fingerprint()
                ),
            }
        )



# =============================================================================
# Functional message-passing node state
# =============================================================================


FMP_NODE_STATE_SOURCE_LAYER_OUTPUT: Final[str] = "layer_output"


@dataclass(slots=True, frozen=True)
class FunctionalMessagePassingNodeState:
    """
    Node state produced by a completed functional message-passing layer.

    This contract lets a later stack depth consume an evolved node state
    without pretending that the tensor was produced by node-state fusion.
    """

    state: torch.Tensor
    alignment: NodeAlignment

    source_kind: str
    source_layer_index: int

    source_architecture_fingerprint: str
    source_lineage_fingerprint: str
    source_parameter_fingerprint: str | None = None

    schema_version: str = (
        FUNCTIONAL_MESSAGE_PASSING_NODE_STATE_SCHEMA_VERSION
    )

    def __post_init__(self) -> None:
        _require_float_tensor(
            "state",
            self.state,
            ndim=2,
        )

        if int(self.state.shape[0]) <= 0:
            raise ValueError(
                "state must contain at least one node."
            )

        if int(self.state.shape[1]) <= 0:
            raise ValueError(
                "state must contain at least one hidden feature."
            )

        if not isinstance(
            self.alignment,
            NodeAlignment,
        ):
            raise TypeError(
                "alignment must be a NodeAlignment."
            )

        if self.alignment.item_count != self.item_count:
            raise ValueError(
                "alignment.item_count must match the number of state rows."
            )

        if self.alignment.node_batch_index is None:
            raise ValueError(
                "alignment must preserve node_batch_index."
            )

        if (
            self.alignment.node_batch_index.device
            != self.state.device
        ):
            raise ValueError(
                "state and alignment.node_batch_index must share one "
                "device."
            )

        _require_nonempty_string(
            "source_kind",
            self.source_kind,
        )

        if self.source_kind != (
            FMP_NODE_STATE_SOURCE_LAYER_OUTPUT
        ):
            raise ValueError(
                "source_kind must be 'layer_output'."
            )

        _require_nonnegative_int(
            "source_layer_index",
            self.source_layer_index,
        )

        _require_nonempty_string(
            "source_architecture_fingerprint",
            self.source_architecture_fingerprint,
        )
        _require_nonempty_string(
            "source_lineage_fingerprint",
            self.source_lineage_fingerprint,
        )
        _require_optional_nonempty_string(
            "source_parameter_fingerprint",
            self.source_parameter_fingerprint,
        )
        _require_nonempty_string(
            "schema_version",
            self.schema_version,
        )

    @property
    def fused_state(self) -> torch.Tensor:
        return self.state

    @property
    def item_count(self) -> int:
        return int(self.state.shape[0])

    @property
    def output_dim(self) -> int:
        return int(self.state.shape[1])

    @property
    def device(self) -> torch.device:
        return self.state.device

    @property
    def dtype(self) -> torch.dtype:
        return self.state.dtype

    @property
    def encoder_architecture_fingerprint(
        self,
    ) -> str:
        return self.source_architecture_fingerprint

    def lineage_dict(
        self,
    ) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "source_kind": self.source_kind,
            "source_layer_index": (
                self.source_layer_index
            ),
            "source_architecture_fingerprint": (
                self.source_architecture_fingerprint
            ),
            "source_parameter_fingerprint": (
                self.source_parameter_fingerprint
            ),
            "source_lineage_fingerprint": (
                self.source_lineage_fingerprint
            ),
            "alignment_fingerprint": (
                self.alignment.fingerprint()
            ),
        }

    @property
    def lineage_fingerprint(self) -> str:
        return _fingerprint(
            self.lineage_dict()
        )

    def value_fingerprint(self) -> str:
        return _tensor_fingerprint(
            {
                "state": self.state,
            }
        )
    


# =============================================================================
# Functional message-passing inputs
# =============================================================================


@dataclass(slots=True, frozen=True)
class FunctionalMessagePassingInputs:
    """
    Complete metadata-preserving input to functional message passing.

    ``edge_relation_type`` in ``source_graph`` is interpreted as a dense
    relation index into ``compiled_relation_registry``. Stable ontology IDs
    are rejected when they fall outside the dense range.

    The cached ``node_hazard_query`` is always node-aligned ``[N, Q]``.
    Graph-scoped queries are expanded explicitly through
    ``source_graph.node_batch_index``.
    """

    source_graph: UrbanGraphBatch
    node_state: (
        NodeStateFusionOutput
        | FunctionalMessagePassingNodeState
    )
    compiled_relation_registry: CompiledRelationRegistry

    relation_families: (
        RelationFamilyAlignment | None
    ) = None
    hazard_query: HazardQueryEncoding | None = None
    compiled_relation_priors: (
        CompiledHazardRelationPriors | None
    ) = None

    source_fingerprint: str | None = None

    schema_version: str = (
        FUNCTIONAL_MESSAGE_PASSING_INPUT_SCHEMA_VERSION
    )

    _edge_batch_index: torch.Tensor = field(
        init=False,
        repr=False,
        compare=False,
    )
    _control_relation_mask: torch.Tensor = field(
        init=False,
        repr=False,
        compare=False,
    )
    _control_edge_mask: torch.Tensor = field(
        init=False,
        repr=False,
        compare=False,
    )
    _node_hazard_query: torch.Tensor | None = field(
        init=False,
        repr=False,
        compare=False,
    )

    def __post_init__(self) -> None:
        if not isinstance(
            self.source_graph,
            UrbanGraphBatch,
        ):
            raise TypeError(
                "source_graph must be an UrbanGraphBatch."
            )

        if not isinstance(
            self.node_state,
            (
                NodeStateFusionOutput,
                FunctionalMessagePassingNodeState,
            ),
        ):
            raise TypeError(
                "node_state must be a NodeStateFusionOutput or "
                "FunctionalMessagePassingNodeState."
            )

        if not isinstance(
            self.compiled_relation_registry,
            CompiledRelationRegistry,
        ):
            raise TypeError(
                "compiled_relation_registry must be a "
                "CompiledRelationRegistry."
            )

        self.source_graph.validate()
        self.compiled_relation_registry.validate()

        if self.source_graph.allow_cross_graph_edges:
            raise ValueError(
                "The bounded FMP baseline forbids cross-graph edges. "
                "Construct independent packed graphs instead."
            )

        if self.num_nodes <= 0:
            raise ValueError(
                "Functional message passing requires at least one node."
            )

        if self.node_state.item_count != self.num_nodes:
            raise ValueError(
                "Fused node-state rows do not match source_graph nodes."
            )

        alignment = self.node_state.alignment

        if not alignment.item_ids:
            raise ValueError(
                "NodeStateFusionOutput alignment must preserve stable "
                "item_ids before entering functional message passing."
            )

        if alignment.item_ids != (
            self.source_graph.external_node_ids
        ):
            raise ValueError(
                "Fused node-state item_ids differ from "
                "source_graph.external_node_ids."
            )

        if alignment.node_batch_index is None:
            raise ValueError(
                "NodeStateFusionOutput alignment must preserve "
                "node_batch_index for functional message passing."
            )

        if (
            alignment.node_batch_index.device
            != self.source_graph.node_batch_index.device
        ):
            raise ValueError(
                "Functional message-passing tensors must share one device."
            )

        if not torch.equal(
            alignment.node_batch_index,
            self.source_graph.node_batch_index,
        ):
            raise ValueError(
                "Fused node-state node_batch_index differs from "
                "source_graph.node_batch_index."
            )


        if alignment.graph_count != self.num_graphs:
            raise ValueError(
                "Fused node-state graph_count differs from the "
                "source graph batch size."
            )

        edge_index = self.source_graph.edge_index
        relation_index = (
            self.source_graph.edge_relation_type
        )
        node_batch_index = (
            self.source_graph.node_batch_index
        )

        _require_long_tensor(
            "source_graph.edge_index",
            edge_index,
            ndim=2,
        )
        _require_shape(
            "source_graph.edge_index",
            edge_index,
            (2, self.num_edges),
        )
        _require_long_tensor(
            "source_graph.edge_relation_type",
            relation_index,
            ndim=1,
        )
        _require_shape(
            "source_graph.edge_relation_type",
            relation_index,
            (self.num_edges,),
        )
        _require_long_tensor(
            "source_graph.node_batch_index",
            node_batch_index,
            ndim=1,
        )

        _require_index_range(
            "source_graph.edge_relation_type",
            relation_index,
            upper_bound=self.num_relations,
        )

        if self.num_edges > 0:
            _require_index_range(
                "source_graph.edge_index",
                edge_index,
                upper_bound=self.num_nodes,
            )

        derived_edge_batch = (
            node_batch_index[self.source_index]
        )

        target_batch = (
            node_batch_index[self.target_index]
        )

        if not torch.equal(
            derived_edge_batch,
            target_batch,
        ):
            raise ValueError(
                "Cross-graph edges are not permitted by the bounded "
                "functional message-passing baseline."
            )

        if (
            self.source_graph.edge_batch_index
            is not None
        ):
            if not torch.equal(
                self.source_graph.edge_batch_index,
                derived_edge_batch,
            ):
                raise ValueError(
                    "source_graph.edge_batch_index differs from edge "
                    "endpoint graph membership."
                )

        object.__setattr__(
            self,
            "_edge_batch_index",
            derived_edge_batch,
        )

        control_relation_mask = torch.tensor(
            [
                bool(
                    entry.specification.is_control
                )
                for entry
                in self.compiled_relation_registry.entries
            ],
            dtype=torch.bool,
            device=relation_index.device,
        )
        control_edge_mask = (
            control_relation_mask[relation_index]
        )

        object.__setattr__(
            self,
            "_control_relation_mask",
            control_relation_mask,
        )
        object.__setattr__(
            self,
            "_control_edge_mask",
            control_edge_mask,
        )

        if self.relation_families is not None:
            if not isinstance(
                self.relation_families,
                RelationFamilyAlignment,
            ):
                raise TypeError(
                    "relation_families must be a "
                    "RelationFamilyAlignment or None."
                )

            if (
                self.relation_families.relation_names
                != self.relation_names
            ):
                raise ValueError(
                    "Relation-family metadata relation ordering differs "
                    "from the compiled relation registry."
                )

            if (
                self
                .relation_families
                .stable_relation_ids
                != self.stable_relation_ids
            ):
                raise ValueError(
                    "Relation-family metadata stable relation IDs differ "
                    "from the compiled relation registry."
                )

            if (
                self
                .relation_families
                .compiled_relation_registry_fingerprint
                != self.compiled_relation_registry.fingerprint()
            ):
                raise ValueError(
                    "Relation-family metadata references a different "
                    "compiled relation registry."
                )

        node_hazard_query = (
            self._align_hazard_query()
        )
        object.__setattr__(
            self,
            "_node_hazard_query",
            node_hazard_query,
        )

        if (
            self.compiled_relation_priors
            is not None
        ):
            if not isinstance(
                self.compiled_relation_priors,
                CompiledHazardRelationPriors,
            ):
                raise TypeError(
                    "compiled_relation_priors must be a "
                    "CompiledHazardRelationPriors or None."
                )

            priors = self.compiled_relation_priors

            if priors.relation_names != (
                self.relation_names
            ):
                raise ValueError(
                    "Compiled hazard-prior relation ordering differs "
                    "from the compiled relation registry."
                )

            if priors.stable_relation_ids != (
                self.stable_relation_ids
            ):
                raise ValueError(
                    "Compiled hazard-prior stable relation IDs differ "
                    "from the compiled relation registry."
                )

            if (
                priors
                .source_compiled_relation_fingerprint
                != self.compiled_relation_registry.fingerprint()
            ):
                raise ValueError(
                    "Compiled hazard priors reference a different "
                    "compiled relation registry."
                )

        floating = {
            "node_state": (
                self.node_state.fused_state
            ),
            "edge_attributes": (
                self.source_graph.edge_attributes
            ),
            "semantic_edge_weight": (
                self
                .source_graph
                .semantic_edge_weight
            ),
            "node_hazard_query": node_hazard_query,
        }

        for name, value in floating.items():
            if value is not None:
                _require_float_tensor(name, value)

        _require_same_device(
            {
                "node_state": (
                    self.node_state.fused_state
                ),
                "node_batch_index": (
                    node_batch_index
                ),
                "edge_index": edge_index,
                "edge_relation_index": (
                    relation_index
                ),
                "edge_batch_index": (
                    derived_edge_batch
                ),
                "edge_attributes": (
                    self.source_graph.edge_attributes
                ),
                "semantic_edge_weight": (
                    self
                    .source_graph
                    .semantic_edge_weight
                ),
                "node_hazard_query": (
                    node_hazard_query
                ),
                "relation_family_index": (
                    self.relation_families
                    .relation_family_index_by_relation
                    if self.relation_families
                    is not None
                    else None
                ),
            }
        )

        _require_same_float_dtype(
            floating
        )

        _require_optional_nonempty_string(
            "source_fingerprint",
            self.source_fingerprint,
        )
        _require_nonempty_string(
            "schema_version",
            self.schema_version,
        )

    def _align_hazard_query(
        self,
    ) -> torch.Tensor | None:
        if self.hazard_query is None:
            return None

        if not isinstance(
            self.hazard_query,
            HazardQueryEncoding,
        ):
            raise TypeError(
                "hazard_query must be a HazardQueryEncoding or None."
            )

        query = self.hazard_query.query
        source = self.hazard_query.source_embedding

        if isinstance(
            source,
            NodeAlignedHazardEmbeddingLookup,
        ):
            if self.hazard_query.item_count != self.num_nodes:
                raise ValueError(
                    "Node-aligned hazard query rows must match the "
                    "number of graph nodes."
                )

            if not torch.equal(
                source.node_batch_index,
                self.source_graph.node_batch_index,
            ):
                raise ValueError(
                    "Node-aligned hazard-query graph membership differs "
                    "from source_graph.node_batch_index."
                )

            return query

        if isinstance(
            source,
            HazardEmbeddingLookup,
        ):
            if self.hazard_query.item_count != self.num_graphs:
                raise ValueError(
                    "Graph-scoped hazard query rows must match the "
                    "packed graph count."
                )

            return query[
                self.source_graph.node_batch_index
            ]

        raise TypeError(
            "HazardQueryEncoding.source_embedding has an unsupported "
            "alignment contract."
        )

    @property
    def num_nodes(self) -> int:
        return self.source_graph.num_nodes

    @property
    def num_edges(self) -> int:
        return self.source_graph.num_edges

    @property
    def num_graphs(self) -> int:
        return self.source_graph.batch_size

    @property
    def hidden_dim(self) -> int:
        return self.node_state.output_dim

    @property
    def num_relations(self) -> int:
        return len(
            self.compiled_relation_registry
        )

    @property
    def num_relation_families(
        self,
    ) -> int | None:
        return (
            self.relation_families.num_families
            if self.relation_families
            is not None
            else None
        )

    @property
    def device(self) -> torch.device:
        return self.node_state.fused_state.device

    @property
    def dtype(self) -> torch.dtype:
        return self.node_state.fused_state.dtype

    @property
    def source_index(self) -> torch.Tensor:
        return self.source_graph.edge_index[0]

    @property
    def target_index(self) -> torch.Tensor:
        return self.source_graph.edge_index[1]

    @property
    def edge_relation_index(
        self,
    ) -> torch.Tensor:
        return self.source_graph.edge_relation_type

    @property
    def edge_batch_index(
        self,
    ) -> torch.Tensor:
        return self._edge_batch_index

    @property
    def node_batch_index(
        self,
    ) -> torch.Tensor:
        return self.source_graph.node_batch_index

    @property
    def node_hazard_query(
        self,
    ) -> torch.Tensor | None:
        return self._node_hazard_query

    @property
    def relation_names(
        self,
    ) -> tuple[str, ...]:
        return (
            self
            .compiled_relation_registry
            .relation_names
        )

    @property
    def stable_relation_ids(
        self,
    ) -> tuple[int, ...]:
        return (
            self
            .compiled_relation_registry
            .stable_relation_ids
        )

    @property
    def control_relation_mask(
        self,
    ) -> torch.Tensor:
        return self._control_relation_mask

    @property
    def control_edge_mask(
        self,
    ) -> torch.Tensor:
        return self._control_edge_mask

    @property
    def edge_relation_family_index(
        self,
    ) -> torch.Tensor | None:
        if self.relation_families is None:
            return None

        return (
            self
            .relation_families
            .relation_family_index_by_relation[
                self.edge_relation_index
            ]
        )

    @property
    def attention_group_id(
        self,
    ) -> torch.Tensor:
        """
        Flat group ID for target-node + exact-relation attention.
        """

        return (
            self.target_index
            * self.num_relations
            + self.edge_relation_index
        )

    @property
    def attention_num_groups(
        self,
    ) -> int:
        return self.num_nodes * self.num_relations

    def lineage_dict(
        self,
    ) -> dict[str, Any]:
        values: dict[str, Any] = {
            "schema_version": self.schema_version,
            "node_state_lineage_fingerprint": (
                self.node_state.lineage_fingerprint
            ),
            "node_state_architecture_fingerprint": (
                self
                .node_state
                .encoder_architecture_fingerprint
            ),
            "compiled_relation_registry_fingerprint": (
                self
                .compiled_relation_registry
                .fingerprint()
            ),
            "relation_names": list(
                self.relation_names
            ),
            "stable_relation_ids": list(
                self.stable_relation_ids
            ),
            "source_fingerprint": (
                self.source_fingerprint
            ),
        }

        if self.relation_families is not None:
            values[
                "relation_family_fingerprint"
            ] = self.relation_families.fingerprint()

        if self.hazard_query is not None:
            values[
                "hazard_query_lineage_fingerprint"
            ] = self.hazard_query.lineage_fingerprint

        if (
            self.compiled_relation_priors
            is not None
        ):
            values[
                "compiled_relation_prior_fingerprint"
            ] = (
                self
                .compiled_relation_priors
                .fingerprint()
            )

        return values

    def lineage_fingerprint(
        self,
    ) -> str:
        return _fingerprint(
            self.lineage_dict()
        )

    def value_fingerprint(
        self,
    ) -> str:
        tensors = {
            "node_state": (
                self.node_state.fused_state
            ),
            "node_batch_index": (
                self.node_batch_index
            ),
            "edge_index": (
                self.source_graph.edge_index
            ),
            "edge_relation_index": (
                self.edge_relation_index
            ),
            "edge_batch_index": (
                self.edge_batch_index
            ),
            "control_relation_mask": (
                self.control_relation_mask
            ),
        }

        if self.node_hazard_query is not None:
            tensors["node_hazard_query"] = (
                self.node_hazard_query
            )

        if (
            self.source_graph.edge_attributes
            is not None
        ):
            tensors["edge_attributes"] = (
                self.source_graph.edge_attributes
            )

        if (
            self
            .source_graph
            .semantic_edge_weight
            is not None
        ):
            tensors["semantic_edge_weight"] = (
                self
                .source_graph
                .semantic_edge_weight
            )

        if self.relation_families is not None:
            tensors[
                "relation_family_index_by_relation"
            ] = (
                self
                .relation_families
                .relation_family_index_by_relation
            )

        return _tensor_fingerprint(tensors)


# =============================================================================
# Relation-transform output
# =============================================================================


@dataclass(slots=True, frozen=True)
class RelationTransformOutput:
    """Edge-aligned transformed source states."""

    transformed_source_state: torch.Tensor
    source_inputs: FunctionalMessagePassingInputs

    transform_mode: str
    encoder_architecture_fingerprint: str
    parameter_fingerprint: str | None = None

    relation_parameter_fingerprints: Mapping[
        str,
        str,
    ] = field(default_factory=dict)

    schema_version: str = (
        RELATION_TRANSFORM_OUTPUT_SCHEMA_VERSION
    )

    def __post_init__(self) -> None:
        _require_float_tensor(
            "transformed_source_state",
            self.transformed_source_state,
            ndim=2,
        )
        _require_shape(
            "transformed_source_state",
            self.transformed_source_state,
            (
                self.source_inputs.num_edges,
                self.source_inputs.hidden_dim,
            ),
        )

        if (
            self.transformed_source_state.device
            != self.source_inputs.device
        ):
            raise ValueError(
                "transformed_source_state and source_inputs must "
                "share one device."
            )

        if (
            self.transformed_source_state.dtype
            != self.source_inputs.dtype
        ):
            raise ValueError(
                "transformed_source_state and node_state must "
                "share one dtype."
            )

        _require_nonempty_string(
            "transform_mode",
            self.transform_mode,
        )
        _require_nonempty_string(
            "encoder_architecture_fingerprint",
            self.encoder_architecture_fingerprint,
        )
        _require_optional_nonempty_string(
            "parameter_fingerprint",
            self.parameter_fingerprint,
        )
        _require_nonempty_string(
            "schema_version",
            self.schema_version,
        )

        if not isinstance(
            self.relation_parameter_fingerprints,
            Mapping,
        ):
            raise TypeError(
                "relation_parameter_fingerprints must be a mapping."
            )

        copied: dict[str, str] = {}

        for name, value in (
            self
            .relation_parameter_fingerprints
            .items()
        ):
            _require_nonempty_string(
                "relation parameter name",
                name,
            )
            _require_nonempty_string(
                f"relation parameter fingerprint {name!r}",
                value,
            )
            copied[name] = value

        unexpected = sorted(
            set(copied)
            - set(self.source_inputs.relation_names)
        )
        if unexpected:
            raise ValueError(
                "relation_parameter_fingerprints contains relations "
                f"outside the compiled registry: {unexpected}."
            )

        object.__setattr__(
            self,
            "relation_parameter_fingerprints",
            MappingProxyType(copied),
        )

    @property
    def num_edges(self) -> int:
        return int(
            self.transformed_source_state.shape[0]
        )

    @property
    def hidden_dim(self) -> int:
        return int(
            self.transformed_source_state.shape[1]
        )


# =============================================================================
# Structural edge-normalization output
# =============================================================================


@dataclass(slots=True, frozen=True)
class StructuralEdgeNormalizationOutput:
    """Graph-structural scalar coefficient for every stored edge."""

    coefficients: torch.Tensor
    source_inputs: FunctionalMessagePassingInputs

    normalization_mode: str
    encoder_architecture_fingerprint: str

    source_degree: torch.Tensor | None = None
    target_degree: torch.Tensor | None = None

    schema_version: str = (
        EDGE_NORMALIZATION_OUTPUT_SCHEMA_VERSION
    )

    def __post_init__(self) -> None:
        _require_float_tensor(
            "coefficients",
            self.coefficients,
            ndim=1,
        )
        _require_shape(
            "coefficients",
            self.coefficients,
            (self.source_inputs.num_edges,),
        )

        if bool(
            (self.coefficients < 0)
            .any()
            .item()
        ):
            raise ValueError(
                "Structural normalization coefficients must be "
                "nonnegative."
            )

        for name, degree in (
            ("source_degree", self.source_degree),
            ("target_degree", self.target_degree),
        ):
            if degree is None:
                continue

            _require_long_tensor(
                name,
                degree,
                ndim=1,
            )
            _require_shape(
                name,
                degree,
                (self.source_inputs.num_nodes,),
            )

            if bool(
                (degree < 0).any().item()
            ):
                raise ValueError(
                    f"{name} must be nonnegative."
                )

        _require_same_device(
            {
                "coefficients": self.coefficients,
                "source_degree": self.source_degree,
                "target_degree": self.target_degree,
                "node_state": (
                    self
                    .source_inputs
                    .node_state
                    .fused_state
                ),
            }
        )

        if (
            self.coefficients.dtype
            != self.source_inputs.dtype
        ):
            raise ValueError(
                "Structural normalization coefficients and node state "
                "must share one dtype."
            )

        _require_nonempty_string(
            "normalization_mode",
            self.normalization_mode,
        )
        _require_nonempty_string(
            "encoder_architecture_fingerprint",
            self.encoder_architecture_fingerprint,
        )
        _require_nonempty_string(
            "schema_version",
            self.schema_version,
        )

        if (
            self.normalization_mode
            == EDGE_NORMALIZATION_NONE
        ):
            expected = torch.ones_like(
                self.coefficients
            )
            if not torch.equal(
                self.coefficients,
                expected,
            ):
                raise ValueError(
                    "edge_normalization_type='none' requires exact "
                    "multiplicative identity coefficients."
                )


# =============================================================================
# Relation-gate output
# =============================================================================


@dataclass(slots=True, frozen=True)
class RelationGateOutput:
    """
    Target-node gate values over the exact compiled relation axis.

    The baseline gate is sigmoid and independently activates every compiled
    relation channel. It is not a family-level softmax.
    """

    gate_logits: torch.Tensor
    gate_values: torch.Tensor
    edge_gate_values: torch.Tensor

    source_inputs: FunctionalMessagePassingInputs

    scope: str
    activation: str

    encoder_architecture_fingerprint: str
    parameter_fingerprint: str | None = None

    prior_logit_contribution: torch.Tensor | None = None
    regularization_terms: Mapping[
        str,
        torch.Tensor,
    ] = field(default_factory=dict)

    schema_version: str = (
        RELATION_GATE_OUTPUT_SCHEMA_VERSION
    )

    def __post_init__(self) -> None:
        expected_node_relation = (
            self.source_inputs.num_nodes,
            self.source_inputs.num_relations,
        )

        for name, tensor in (
            ("gate_logits", self.gate_logits),
            ("gate_values", self.gate_values),
        ):
            _require_float_tensor(
                name,
                tensor,
                ndim=2,
            )
            _require_shape(
                name,
                tensor,
                expected_node_relation,
            )

        _require_float_tensor(
            "edge_gate_values",
            self.edge_gate_values,
            ndim=1,
        )
        _require_shape(
            "edge_gate_values",
            self.edge_gate_values,
            (self.source_inputs.num_edges,),
        )

        if self.prior_logit_contribution is not None:
            _require_float_tensor(
                "prior_logit_contribution",
                self.prior_logit_contribution,
                ndim=2,
            )
            _require_shape(
                "prior_logit_contribution",
                self.prior_logit_contribution,
                expected_node_relation,
            )

        _require_same_device(
            {
                "gate_logits": self.gate_logits,
                "gate_values": self.gate_values,
                "edge_gate_values": (
                    self.edge_gate_values
                ),
                "prior_logit_contribution": (
                    self.prior_logit_contribution
                ),
                "node_state": (
                    self
                    .source_inputs
                    .node_state
                    .fused_state
                ),
            }
        )
        _require_same_float_dtype(
            {
                "gate_logits": self.gate_logits,
                "gate_values": self.gate_values,
                "edge_gate_values": (
                    self.edge_gate_values
                ),
                "prior_logit_contribution": (
                    self.prior_logit_contribution
                ),
                "node_state": (
                    self
                    .source_inputs
                    .node_state
                    .fused_state
                ),
            }
        )

        _require_nonempty_string(
            "scope",
            self.scope,
        )
        _require_nonempty_string(
            "activation",
            self.activation,
        )

        if self.scope != (
            RELATION_GATE_SCOPE_TARGET_NODE
        ):
            raise ValueError(
                "The bounded V2.0 gate schema currently supports only "
                f"{RELATION_GATE_SCOPE_TARGET_NODE!r} scope."
            )

        if self.activation != (
            RELATION_GATE_ACTIVATION_SIGMOID
        ):
            raise ValueError(
                "The bounded V2.0 gate schema currently supports only "
                f"{RELATION_GATE_ACTIVATION_SIGMOID!r} activation."
            )

        if bool(
            (
                (self.gate_values < 0)
                | (self.gate_values > 1)
            )
            .any()
            .item()
        ):
            raise ValueError(
                "Sigmoid gate values must lie in [0, 1]."
            )

        expected_edge = self.gate_values[
            self.source_inputs.target_index,
            self.source_inputs.edge_relation_index,
        ]
        atol, rtol = _default_tolerances(
            self.gate_values.dtype
        )

        if not torch.allclose(
            self.edge_gate_values,
            expected_edge,
            atol=atol,
            rtol=rtol,
        ):
            raise ValueError(
                "edge_gate_values must equal target-node gate lookup "
                "by dense relation index."
            )

        _require_nonempty_string(
            "encoder_architecture_fingerprint",
            self.encoder_architecture_fingerprint,
        )
        _require_optional_nonempty_string(
            "parameter_fingerprint",
            self.parameter_fingerprint,
        )
        _require_nonempty_string(
            "schema_version",
            self.schema_version,
        )

        object.__setattr__(
            self,
            "regularization_terms",
            _immutable_tensor_mapping(
                "regularization_terms",
                self.regularization_terms,
                scalar_only=True,
                device=self.gate_values.device,
            ),
        )

    @property
    def control_relation_mask(
        self,
    ) -> torch.Tensor:
        return (
            self
            .source_inputs
            .control_relation_mask
        )


# =============================================================================
# Edge-attention output
# =============================================================================


@dataclass(slots=True, frozen=True)
class EdgeAttentionOutput:
    """
    Edge attention with exact target-node + relation grouping metadata.
    """

    raw_scores_by_head: torch.Tensor
    normalized_weights_by_head: torch.Tensor
    edge_weights: torch.Tensor

    group_ids: torch.Tensor
    group_counts: torch.Tensor

    source_inputs: FunctionalMessagePassingInputs

    attention_mode: str
    normalization_mode: str
    head_reduction: str

    encoder_architecture_fingerprint: str
    parameter_fingerprint: str | None = None

    schema_version: str = (
        EDGE_ATTENTION_OUTPUT_SCHEMA_VERSION
    )

    def __post_init__(self) -> None:
        num_edges = self.source_inputs.num_edges

        for name, tensor in (
            (
                "raw_scores_by_head",
                self.raw_scores_by_head,
            ),
            (
                "normalized_weights_by_head",
                self.normalized_weights_by_head,
            ),
        ):
            _require_float_tensor(
                name,
                tensor,
                ndim=2,
            )

            if int(tensor.shape[0]) != num_edges:
                raise ValueError(
                    f"{name} rows must align with edges."
                )

            if int(tensor.shape[1]) <= 0:
                raise ValueError(
                    f"{name} must contain at least one attention head."
                )

        if (
            self.raw_scores_by_head.shape
            != self.normalized_weights_by_head.shape
        ):
            raise ValueError(
                "Raw and normalized head-level attention tensors must "
                "have equal shape."
            )

        _require_float_tensor(
            "edge_weights",
            self.edge_weights,
            ndim=1,
        )
        _require_shape(
            "edge_weights",
            self.edge_weights,
            (num_edges,),
        )

        _require_long_tensor(
            "group_ids",
            self.group_ids,
            ndim=1,
        )
        _require_shape(
            "group_ids",
            self.group_ids,
            (num_edges,),
        )
        _require_index_range(
            "group_ids",
            self.group_ids,
            upper_bound=(
                self.source_inputs
                .attention_num_groups
            ),
        )

        _require_long_tensor(
            "group_counts",
            self.group_counts,
            ndim=1,
        )
        _require_shape(
            "group_counts",
            self.group_counts,
            (
                self
                .source_inputs
                .attention_num_groups,
            ),
        )

        if bool(
            (self.group_counts < 0)
            .any()
            .item()
        ):
            raise ValueError(
                "group_counts must be nonnegative."
            )

        _require_same_device(
            {
                "raw_scores_by_head": (
                    self.raw_scores_by_head
                ),
                "normalized_weights_by_head": (
                    self.normalized_weights_by_head
                ),
                "edge_weights": self.edge_weights,
                "group_ids": self.group_ids,
                "group_counts": self.group_counts,
                "node_state": (
                    self
                    .source_inputs
                    .node_state
                    .fused_state
                ),
            }
        )
        _require_same_float_dtype(
            {
                "raw_scores_by_head": (
                    self.raw_scores_by_head
                ),
                "normalized_weights_by_head": (
                    self.normalized_weights_by_head
                ),
                "edge_weights": self.edge_weights,
                "node_state": (
                    self
                    .source_inputs
                    .node_state
                    .fused_state
                ),
            }
        )

        _require_nonempty_string(
            "attention_mode",
            self.attention_mode,
        )
        _require_nonempty_string(
            "normalization_mode",
            self.normalization_mode,
        )
        _require_nonempty_string(
            "head_reduction",
            self.head_reduction,
        )

        if self.normalization_mode != (
            ATTENTION_NORMALIZATION_TARGET_NODE_RELATION
        ):
            raise ValueError(
                "The bounded V2.0 attention schema currently supports "
                "only exact target-node + relation normalization."
            )

        expected_group_ids = (
            self
            .source_inputs
            .attention_group_id
        )

        if not torch.equal(
            self.group_ids,
            expected_group_ids,
        ):
            raise ValueError(
                "group_ids must encode target node + exact dense "
                "relation index."
            )

        expected_counts = torch.bincount(
            self.group_ids,
            minlength=(
                self
                .source_inputs
                .attention_num_groups
            ),
        )

        if not torch.equal(
            self.group_counts,
            expected_counts,
        ):
            raise ValueError(
                "group_counts do not match attention group_ids."
            )

        if bool(
            (
                self.normalized_weights_by_head
                < 0
            )
            .any()
            .item()
        ):
            raise ValueError(
                "Normalized attention weights must be nonnegative."
            )

        if bool(
            (self.edge_weights < 0)
            .any()
            .item()
        ):
            raise ValueError(
                "Reduced edge attention weights must be nonnegative."
            )

        num_groups = (
            self
            .source_inputs
            .attention_num_groups
        )
        num_heads = int(
            self
            .normalized_weights_by_head
            .shape[1]
        )
        sums = torch.zeros(
            (
                num_groups,
                num_heads,
            ),
            dtype=(
                self
                .normalized_weights_by_head
                .dtype
            ),
            device=(
                self
                .normalized_weights_by_head
                .device
            ),
        )
        sums.index_add_(
            0,
            self.group_ids,
            self.normalized_weights_by_head,
        )

        nonempty = self.group_counts > 0
        atol, rtol = _default_tolerances(
            self
            .normalized_weights_by_head
            .dtype
        )

        if bool(nonempty.any().item()):
            expected_ones = torch.ones_like(
                sums[nonempty]
            )
            if not torch.allclose(
                sums[nonempty],
                expected_ones,
                atol=atol,
                rtol=rtol,
            ):
                raise ValueError(
                    "Each nonempty target-node/relation attention group "
                    "must sum to one independently for every head."
                )

        if self.head_reduction == (
            ATTENTION_HEAD_REDUCTION_MEAN
        ):
            expected_reduced = (
                self
                .normalized_weights_by_head
                .mean(dim=1)
            )
            if not torch.allclose(
                self.edge_weights,
                expected_reduced,
                atol=atol,
                rtol=rtol,
            ):
                raise ValueError(
                    "Mean head reduction must equal the arithmetic mean "
                    "of head-level weights."
                )

        _require_nonempty_string(
            "encoder_architecture_fingerprint",
            self.encoder_architecture_fingerprint,
        )
        _require_optional_nonempty_string(
            "parameter_fingerprint",
            self.parameter_fingerprint,
        )
        _require_nonempty_string(
            "schema_version",
            self.schema_version,
        )

    @property
    def num_heads(self) -> int:
        return int(
            self.raw_scores_by_head.shape[1]
        )


# =============================================================================
# Edge-message output
# =============================================================================


@dataclass(slots=True, frozen=True)
class EdgeMessageOutput:
    """
    Final edge messages and every multiplicative factor used to build them.

    Disabled gate or attention mechanisms are represented by ``None`` and
    contribute the exact multiplicative identity one.
    """

    edge_messages: torch.Tensor

    relation_transform: RelationTransformOutput
    edge_normalization: StructuralEdgeNormalizationOutput

    relation_gate: RelationGateOutput | None
    edge_attention: EdgeAttentionOutput | None

    semantic_edge_weight: torch.Tensor | None

    encoder_architecture_fingerprint: str

    schema_version: str = (
        EDGE_MESSAGE_OUTPUT_SCHEMA_VERSION
    )

    def __post_init__(self) -> None:
        source_inputs = (
            self.relation_transform.source_inputs
        )

        if (
            self.edge_normalization.source_inputs
            is not source_inputs
        ):
            raise ValueError(
                "Relation transform and edge normalization must share "
                "the same FunctionalMessagePassingInputs object."
            )

        if (
            self.relation_gate is not None
            and self.relation_gate.source_inputs
            is not source_inputs
        ):
            raise ValueError(
                "Relation gate and relation transform must share the "
                "same FunctionalMessagePassingInputs object."
            )

        if (
            self.edge_attention is not None
            and self.edge_attention.source_inputs
            is not source_inputs
        ):
            raise ValueError(
                "Edge attention and relation transform must share the "
                "same FunctionalMessagePassingInputs object."
            )

        _require_float_tensor(
            "edge_messages",
            self.edge_messages,
            ndim=2,
        )
        _require_shape(
            "edge_messages",
            self.edge_messages,
            (
                source_inputs.num_edges,
                source_inputs.hidden_dim,
            ),
        )

        if self.semantic_edge_weight is not None:
            _require_float_tensor(
                "semantic_edge_weight",
                self.semantic_edge_weight,
                ndim=1,
            )
            _require_shape(
                "semantic_edge_weight",
                self.semantic_edge_weight,
                (source_inputs.num_edges,),
            )

        _require_same_device(
            {
                "edge_messages": self.edge_messages,
                "transformed_source_state": (
                    self
                    .relation_transform
                    .transformed_source_state
                ),
                "structural_normalization": (
                    self
                    .edge_normalization
                    .coefficients
                ),
                "edge_gate_values": (
                    self.relation_gate.edge_gate_values
                    if self.relation_gate
                    is not None
                    else None
                ),
                "edge_attention": (
                    self.edge_attention.edge_weights
                    if self.edge_attention
                    is not None
                    else None
                ),
                "semantic_edge_weight": (
                    self.semantic_edge_weight
                ),
            }
        )
        _require_same_float_dtype(
            {
                "edge_messages": self.edge_messages,
                "transformed_source_state": (
                    self
                    .relation_transform
                    .transformed_source_state
                ),
                "structural_normalization": (
                    self
                    .edge_normalization
                    .coefficients
                ),
                "edge_gate_values": (
                    self.relation_gate.edge_gate_values
                    if self.relation_gate
                    is not None
                    else None
                ),
                "edge_attention": (
                    self.edge_attention.edge_weights
                    if self.edge_attention
                    is not None
                    else None
                ),
                "semantic_edge_weight": (
                    self.semantic_edge_weight
                ),
            }
        )

        expected = (
            self
            .relation_transform
            .transformed_source_state
        )
        expected = expected * (
            self
            .edge_normalization
            .coefficients
            .unsqueeze(-1)
        )

        if self.relation_gate is not None:
            expected = expected * (
                self
                .relation_gate
                .edge_gate_values
                .unsqueeze(-1)
            )

        if self.edge_attention is not None:
            expected = expected * (
                self
                .edge_attention
                .edge_weights
                .unsqueeze(-1)
            )

        if self.semantic_edge_weight is not None:
            expected = expected * (
                self
                .semantic_edge_weight
                .unsqueeze(-1)
            )

        atol, rtol = _default_tolerances(
            self.edge_messages.dtype
        )

        if not torch.allclose(
            self.edge_messages,
            expected,
            atol=atol,
            rtol=rtol,
        ):
            raise ValueError(
                "edge_messages do not equal the explicit product of "
                "the transformed source state and enabled edge factors."
            )

        _require_nonempty_string(
            "encoder_architecture_fingerprint",
            self.encoder_architecture_fingerprint,
        )
        _require_nonempty_string(
            "schema_version",
            self.schema_version,
        )

    @property
    def source_inputs(
        self,
    ) -> FunctionalMessagePassingInputs:
        return self.relation_transform.source_inputs

    @property
    def structural_factor(
        self,
    ) -> torch.Tensor:
        return self.edge_normalization.coefficients

    @property
    def gate_factor(
        self,
    ) -> torch.Tensor | None:
        return (
            self.relation_gate.edge_gate_values
            if self.relation_gate is not None
            else None
        )

    @property
    def attention_factor(
        self,
    ) -> torch.Tensor | None:
        return (
            self.edge_attention.edge_weights
            if self.edge_attention is not None
            else None
        )


# =============================================================================
# Aggregation output
# =============================================================================


@dataclass(slots=True, frozen=True)
class AggregationOutput:
    """Target-node reduction of final edge messages."""

    node_aggregate: torch.Tensor
    incoming_edge_count: torch.Tensor

    source_messages: EdgeMessageOutput

    aggregation_mode: str
    encoder_architecture_fingerprint: str

    schema_version: str = (
        AGGREGATION_OUTPUT_SCHEMA_VERSION
    )

    def __post_init__(self) -> None:
        inputs = self.source_messages.source_inputs

        _require_float_tensor(
            "node_aggregate",
            self.node_aggregate,
            ndim=2,
        )
        _require_shape(
            "node_aggregate",
            self.node_aggregate,
            (
                inputs.num_nodes,
                inputs.hidden_dim,
            ),
        )

        _require_long_tensor(
            "incoming_edge_count",
            self.incoming_edge_count,
            ndim=1,
        )
        _require_shape(
            "incoming_edge_count",
            self.incoming_edge_count,
            (inputs.num_nodes,),
        )

        if bool(
            (self.incoming_edge_count < 0)
            .any()
            .item()
        ):
            raise ValueError(
                "incoming_edge_count must be nonnegative."
            )

        expected_counts = torch.bincount(
            inputs.target_index,
            minlength=inputs.num_nodes,
        )

        if not torch.equal(
            self.incoming_edge_count,
            expected_counts,
        ):
            raise ValueError(
                "incoming_edge_count does not match target-node edge "
                "membership."
            )

        if (
            self.node_aggregate.device
            != inputs.device
            or self.incoming_edge_count.device
            != inputs.device
        ):
            raise ValueError(
                "Aggregation tensors and source inputs must share one "
                "device."
            )

        if (
            self.node_aggregate.dtype
            != inputs.dtype
        ):
            raise ValueError(
                "node_aggregate and source node state must share one "
                "dtype."
            )

        _require_nonempty_string(
            "aggregation_mode",
            self.aggregation_mode,
        )

        if self.aggregation_mode != (
            AGGREGATION_MEAN
        ):
            raise ValueError(
                "The bounded V2.0 aggregation schema currently supports "
                f"only {AGGREGATION_MEAN!r}."
            )

        expected_sum = torch.zeros(
            (
                inputs.num_nodes,
                inputs.hidden_dim,
            ),
            dtype=(
                self
                .source_messages
                .edge_messages
                .dtype
            ),
            device=(
                self
                .source_messages
                .edge_messages
                .device
            ),
        )
        expected_sum.index_add_(
            0,
            inputs.target_index,
            self.source_messages.edge_messages,
        )

        denominator = (
            self
            .incoming_edge_count
            .clamp_min(1)
            .to(dtype=expected_sum.dtype)
            .unsqueeze(-1)
        )
        expected_mean = (
            expected_sum / denominator
        )

        atol, rtol = _default_tolerances(
            self.node_aggregate.dtype
        )

        if not torch.allclose(
            self.node_aggregate,
            expected_mean,
            atol=atol,
            rtol=rtol,
        ):
            raise ValueError(
                "node_aggregate does not match mean aggregation by "
                "target node."
            )

        isolated = (
            self.incoming_edge_count == 0
        )
        if bool(isolated.any().item()):
            if not torch.equal(
                self.node_aggregate[isolated],
                torch.zeros_like(
                    self.node_aggregate[isolated]
                ),
            ):
                raise ValueError(
                    "Isolated nodes must receive an exact zero aggregate."
                )

        _require_nonempty_string(
            "encoder_architecture_fingerprint",
            self.encoder_architecture_fingerprint,
        )
        _require_nonempty_string(
            "schema_version",
            self.schema_version,
        )


# =============================================================================
# Optional retained layer intermediates
# =============================================================================


@dataclass(slots=True, frozen=True)
class FunctionalMessagePassingIntermediates:
    """Full one-layer diagnostic trace retained only when requested."""

    relation_transform: RelationTransformOutput
    edge_normalization: StructuralEdgeNormalizationOutput
    relation_gate: RelationGateOutput | None
    edge_attention: EdgeAttentionOutput | None
    edge_messages: EdgeMessageOutput
    aggregation: AggregationOutput

    pre_residual_state: torch.Tensor
    post_residual_state: torch.Tensor

    schema_version: str = (
        FMP_INTERMEDIATES_SCHEMA_VERSION
    )

    def __post_init__(self) -> None:
        source = (
            self
            .relation_transform
            .source_inputs
        )

        if (
            self.edge_normalization.source_inputs
            is not source
        ):
            raise ValueError(
                "edge_normalization uses different source inputs."
            )

        if (
            self.edge_messages.source_inputs
            is not source
        ):
            raise ValueError(
                "edge_messages uses different source inputs."
            )

        if (
            self.aggregation.source_messages
            is not self.edge_messages
        ):
            raise ValueError(
                "aggregation must consume the retained edge_messages "
                "object."
            )

        if (
            self.relation_gate is not None
            and self.relation_gate.source_inputs
            is not source
        ):
            raise ValueError(
                "relation_gate uses different source inputs."
            )

        if (
            self.edge_attention is not None
            and self.edge_attention.source_inputs
            is not source
        ):
            raise ValueError(
                "edge_attention uses different source inputs."
            )

        for name, state in (
            (
                "pre_residual_state",
                self.pre_residual_state,
            ),
            (
                "post_residual_state",
                self.post_residual_state,
            ),
        ):
            _require_float_tensor(
                name,
                state,
                ndim=2,
            )
            _require_shape(
                name,
                state,
                (
                    source.num_nodes,
                    source.hidden_dim,
                ),
            )

            if state.device != source.device:
                raise ValueError(
                    f"{name} and source inputs must share one device."
                )

            if state.dtype != source.dtype:
                raise ValueError(
                    f"{name} and source node state must share one dtype."
                )

        _require_nonempty_string(
            "schema_version",
            self.schema_version,
        )


# =============================================================================
# One-layer output
# =============================================================================


@dataclass(slots=True, frozen=True)
class FunctionalMessagePassingLayerOutput:
    """
    One FMP layer output with optional high-cost edge-level diagnostics.
    """

    updated_node_state: torch.Tensor
    node_aggregate: torch.Tensor
    incoming_edge_count: torch.Tensor

    source_inputs: FunctionalMessagePassingInputs

    layer_index: int
    residual_enabled: bool
    layer_norm_enabled: bool

    encoder_architecture_fingerprint: str
    lineage_fingerprint: str

    intermediates: (
        FunctionalMessagePassingIntermediates
        | None
    ) = None

    regularization_terms: Mapping[
        str,
        torch.Tensor,
    ] = field(default_factory=dict)

    schema_version: str = (
        FMP_LAYER_OUTPUT_SCHEMA_VERSION
    )

    def __post_init__(self) -> None:
        _require_nonnegative_int(
            "layer_index",
            self.layer_index,
        )

        if not isinstance(
            self.residual_enabled,
            bool,
        ):
            raise TypeError(
                "residual_enabled must be a Boolean."
            )

        if not isinstance(
            self.layer_norm_enabled,
            bool,
        ):
            raise TypeError(
                "layer_norm_enabled must be a Boolean."
            )

        for name, state in (
            (
                "updated_node_state",
                self.updated_node_state,
            ),
            (
                "node_aggregate",
                self.node_aggregate,
            ),
        ):
            _require_float_tensor(
                name,
                state,
                ndim=2,
            )
            _require_shape(
                name,
                state,
                (
                    self.source_inputs.num_nodes,
                    self.source_inputs.hidden_dim,
                ),
            )

            if state.device != (
                self.source_inputs.device
            ):
                raise ValueError(
                    f"{name} and source inputs must share one device."
                )

            if state.dtype != (
                self.source_inputs.dtype
            ):
                raise ValueError(
                    f"{name} and source node state must share one dtype."
                )

        _require_long_tensor(
            "incoming_edge_count",
            self.incoming_edge_count,
            ndim=1,
        )
        _require_shape(
            "incoming_edge_count",
            self.incoming_edge_count,
            (self.source_inputs.num_nodes,),
        )

        expected_count = torch.bincount(
            self.source_inputs.target_index,
            minlength=self.source_inputs.num_nodes,
        )

        if not torch.equal(
            self.incoming_edge_count,
            expected_count,
        ):
            raise ValueError(
                "incoming_edge_count does not match source edge "
                "membership."
            )

        if (
            self.incoming_edge_count.device
            != self.source_inputs.device
        ):
            raise ValueError(
                "incoming_edge_count and source inputs must share one "
                "device."
            )

        if self.intermediates is not None:
            if not isinstance(
                self.intermediates,
                FunctionalMessagePassingIntermediates,
            ):
                raise TypeError(
                    "intermediates must be a "
                    "FunctionalMessagePassingIntermediates or None."
                )

            if (
                self
                .intermediates
                .relation_transform
                .source_inputs
                is not self.source_inputs
            ):
                raise ValueError(
                    "Retained intermediates use different source inputs."
                )

            atol, rtol = _default_tolerances(
                self.node_aggregate.dtype
            )

            if not torch.allclose(
                self.node_aggregate,
                self
                .intermediates
                .aggregation
                .node_aggregate,
                atol=atol,
                rtol=rtol,
            ):
                raise ValueError(
                    "node_aggregate differs from retained aggregation."
                )

        _require_nonempty_string(
            "encoder_architecture_fingerprint",
            self.encoder_architecture_fingerprint,
        )
        _require_nonempty_string(
            "lineage_fingerprint",
            self.lineage_fingerprint,
        )
        _require_nonempty_string(
            "schema_version",
            self.schema_version,
        )

        object.__setattr__(
            self,
            "regularization_terms",
            _immutable_tensor_mapping(
                "regularization_terms",
                self.regularization_terms,
                scalar_only=True,
                device=self.updated_node_state.device,
            ),
        )

    @property
    def num_nodes(self) -> int:
        return int(
            self.updated_node_state.shape[0]
        )

    @property
    def hidden_dim(self) -> int:
        return int(
            self.updated_node_state.shape[1]
        )


# =============================================================================
# Multi-layer stack output
# =============================================================================


@dataclass(slots=True, frozen=True)
class FunctionalMessagePassingStackOutput:
    """Final node state and optional retained per-layer outputs."""

    final_node_state: torch.Tensor
    source_inputs: FunctionalMessagePassingInputs

    num_layers: int
    layer_outputs: tuple[
        FunctionalMessagePassingLayerOutput,
        ...,
    ] = ()

    encoder_architecture_fingerprint: str = ""
    lineage_fingerprint: str = ""

    regularization_terms: Mapping[
        str,
        torch.Tensor,
    ] = field(default_factory=dict)

    schema_version: str = (
        FMP_STACK_OUTPUT_SCHEMA_VERSION
    )

    def __post_init__(self) -> None:
        _require_positive_int(
            "num_layers",
            self.num_layers,
        )
        _require_float_tensor(
            "final_node_state",
            self.final_node_state,
            ndim=2,
        )
        _require_shape(
            "final_node_state",
            self.final_node_state,
            (
                self.source_inputs.num_nodes,
                self.source_inputs.hidden_dim,
            ),
        )

        if (
            self.final_node_state.device
            != self.source_inputs.device
        ):
            raise ValueError(
                "final_node_state and source inputs must share one "
                "device."
            )

        if (
            self.final_node_state.dtype
            != self.source_inputs.dtype
        ):
            raise ValueError(
                "final_node_state and source node state must share one "
                "dtype."
            )

        if len(self.layer_outputs) not in (
            0,
            self.num_layers,
        ):
            raise ValueError(
                "layer_outputs must be empty when not retained, or "
                "contain exactly num_layers outputs."
            )

        for expected_index, output in enumerate(
            self.layer_outputs
        ):
            if not isinstance(
                output,
                FunctionalMessagePassingLayerOutput,
            ):
                raise TypeError(
                    f"layer_outputs[{expected_index}] must be a "
                    "FunctionalMessagePassingLayerOutput."
                )

            if output.layer_index != expected_index:
                raise ValueError(
                    "Retained layer outputs must be ordered by contiguous "
                    "zero-based layer_index."
                )

            if output.source_inputs is not (
                self.source_inputs
            ):
                raise ValueError(
                    "Every retained layer output must reference the same "
                    "source input contract."
                )

        if self.layer_outputs:
            atol, rtol = _default_tolerances(
                self.final_node_state.dtype
            )
            if not torch.allclose(
                self.final_node_state,
                self
                .layer_outputs[-1]
                .updated_node_state,
                atol=atol,
                rtol=rtol,
            ):
                raise ValueError(
                    "final_node_state differs from the final retained "
                    "layer output."
                )

        _require_nonempty_string(
            "encoder_architecture_fingerprint",
            self.encoder_architecture_fingerprint,
        )
        _require_nonempty_string(
            "lineage_fingerprint",
            self.lineage_fingerprint,
        )
        _require_nonempty_string(
            "schema_version",
            self.schema_version,
        )

        object.__setattr__(
            self,
            "regularization_terms",
            _immutable_tensor_mapping(
                "regularization_terms",
                self.regularization_terms,
                scalar_only=True,
                device=self.final_node_state.device,
            ),
        )


__all__ = (
    "AGGREGATION_OUTPUT_SCHEMA_VERSION",
    "EDGE_ATTENTION_OUTPUT_SCHEMA_VERSION",
    "EDGE_MESSAGE_OUTPUT_SCHEMA_VERSION",
    "EDGE_NORMALIZATION_OUTPUT_SCHEMA_VERSION",
    "FMP_INTERMEDIATES_SCHEMA_VERSION",
    "FMP_LAYER_OUTPUT_SCHEMA_VERSION",
    "FMP_NODE_STATE_SOURCE_LAYER_OUTPUT",
    "FMP_STACK_OUTPUT_SCHEMA_VERSION",
    "FUNCTIONAL_MESSAGE_PASSING_INPUT_SCHEMA_VERSION",
    "FUNCTIONAL_MESSAGE_PASSING_NODE_STATE_SCHEMA_VERSION",
    "RELATION_FAMILY_ALIGNMENT_SCHEMA_VERSION",
    "RELATION_GATE_OUTPUT_SCHEMA_VERSION",
    "RELATION_TRANSFORM_OUTPUT_SCHEMA_VERSION",
    "AggregationOutput",
    "EdgeAttentionOutput",
    "EdgeMessageOutput",
    "FunctionalMessagePassingInputs",
    "FunctionalMessagePassingIntermediates",
    "FunctionalMessagePassingLayerOutput",
    "FunctionalMessagePassingNodeState",
    "FunctionalMessagePassingStackOutput",
    "RelationFamilyAlignment",
    "RelationGateOutput",
    "RelationTransformOutput",
    "StructuralEdgeNormalizationOutput",
)
