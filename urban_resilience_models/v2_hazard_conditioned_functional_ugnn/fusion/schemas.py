"""
Metadata-preserving schemas for V2 node-state fusion.

Target path:
    urban_resilience_models/
        v2_hazard_conditioned_functional_ugnn/
            fusion/
                schemas.py

This module owns only the typed data contracts used by the fusion subsystem:

- ``NodeAlignment``;
- ``NodeStateComponent``;
- ``NodeStateFusionInputs``;
- ``NodeStateFusionOutput``;
- alignment, value, and lineage fingerprints;
- device-preserving reconstruction helpers;
- fusion schema-version constants.

It does not own trainable projections, fusion algorithms, configuration
dispatch, diagnostics, attribution, or uncertainty propagation.

Import boundary
---------------
The schemas retain complete upstream result objects whenever a dedicated
contract already exists:

- ``LagMemoryEncoding`` for temporal memory;
- ``HazardQueryEncoding`` for hazard context;
- ``NodeAlignedHazardEmbeddingLookup`` for packed-graph alignment checks.

Static state and future generic components use ``NodeStateComponent`` until
dedicated upstream result contracts exist.
"""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
from types import MappingProxyType
from typing import Any, Final, Mapping, Sequence, TYPE_CHECKING

import torch

from ..hazard.hazard_embeddings import (
    NodeAlignedHazardEmbeddingLookup,
)
from ..hazard.hazard_query_encoder import (
    HazardQueryEncoding,
)
from ..memory.lag_memory_encoder import (
    LagMemoryEncoding,
)

if TYPE_CHECKING:
    from .node_state_fusion import NodeStateFusionMode


# =============================================================================
# Schema identity
# =============================================================================


NODE_ALIGNMENT_SCHEMA_VERSION: Final[str] = "0.1"
NODE_STATE_COMPONENT_SCHEMA_VERSION: Final[str] = "0.1"
NODE_STATE_FUSION_INPUT_SCHEMA_VERSION: Final[str] = "0.1"
NODE_STATE_FUSION_OUTPUT_SCHEMA_VERSION: Final[str] = "0.1"


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
    observed: set[str] = set()
    duplicates: set[str] = set()

    for index, value in enumerate(values):
        _require_nonempty_string(
            f"{name}[{index}]",
            value,
        )

        if value in observed:
            duplicates.add(value)

        observed.add(value)

    if duplicates:
        raise ValueError(
            f"{name} contains duplicates: "
            f"{sorted(duplicates)}."
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


def _assert_finite_tensor(
    name: str,
    tensor: torch.Tensor,
) -> None:
    if (
        tensor.dtype.is_floating_point
        and not bool(
            torch.isfinite(tensor)
            .all()
            .item()
        )
    ):
        raise ValueError(
            f"{name} must contain only finite values."
        )


# =============================================================================
# Alignment contract
# =============================================================================


@dataclass(slots=True, frozen=True)
class NodeAlignment:
    """
    Item/node alignment metadata shared by every fused component.

    Parameters
    ----------
    item_count:
        Number of rows in every node/item-aligned state.

    item_ids:
        Optional stable identifiers aligned with rows.

    node_batch_index:
        Optional packed-graph membership tensor ``[item_count]``.

    graph_count:
        Required when ``node_batch_index`` is supplied. Graph IDs must be
        contiguous from zero and cover exactly ``graph_count`` graphs.

    source_fingerprint:
        Optional fingerprint of the batch or node-index artifact.
    """

    item_count: int

    item_ids: tuple[str, ...] = ()

    node_batch_index: torch.Tensor | None = None
    graph_count: int | None = None

    source_fingerprint: str | None = None
    alignment_name: str = "node_state_alignment"

    schema_version: str = (
        NODE_ALIGNMENT_SCHEMA_VERSION
    )

    def __post_init__(self) -> None:
        _require_nonnegative_int(
            "item_count",
            self.item_count,
        )

        _require_unique_strings(
            "item_ids",
            self.item_ids,
        )

        if (
            self.item_ids
            and len(self.item_ids) != self.item_count
        ):
            raise ValueError(
                "item_ids must align with item_count."
            )

        if self.node_batch_index is None:
            if self.graph_count is not None:
                raise ValueError(
                    "graph_count requires node_batch_index."
                )
        else:
            self._validate_graph_membership()

        if self.source_fingerprint is not None:
            _require_nonempty_string(
                "source_fingerprint",
                self.source_fingerprint,
            )

        _require_nonempty_string(
            "alignment_name",
            self.alignment_name,
        )
        _require_nonempty_string(
            "schema_version",
            self.schema_version,
        )

    def _validate_graph_membership(self) -> None:
        assert self.node_batch_index is not None

        if not isinstance(
            self.node_batch_index,
            torch.Tensor,
        ):
            raise TypeError(
                "node_batch_index must be a tensor or None."
            )

        if self.node_batch_index.ndim != 1:
            raise ValueError(
                "node_batch_index must have shape [item_count]."
            )

        if self.node_batch_index.dtype != torch.long:
            raise ValueError(
                "node_batch_index must use torch.long."
            )

        if int(
            self.node_batch_index.shape[0]
        ) != self.item_count:
            raise ValueError(
                "node_batch_index must align with item_count."
            )

        if self.graph_count is None:
            raise ValueError(
                "graph_count is required when node_batch_index "
                "is supplied."
            )

        _require_positive_int(
            "graph_count",
            self.graph_count,
        )

        if self.item_count == 0:
            raise ValueError(
                "A graph-aligned batch cannot have zero items."
            )

        minimum = int(
            self.node_batch_index.min().item()
        )
        maximum = int(
            self.node_batch_index.max().item()
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
            device=self.node_batch_index.device,
        )

        if not torch.equal(
            observed,
            expected,
        ):
            raise ValueError(
                "node_batch_index must contain contiguous graph IDs "
                "from zero and represent every packed graph."
            )

    @property
    def device(self) -> torch.device | None:
        if self.node_batch_index is None:
            return None

        return self.node_batch_index.device

    @property
    def graph_aligned(self) -> bool:
        return self.node_batch_index is not None

    def semantic_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "alignment_name": self.alignment_name,
            "item_count": self.item_count,
            "item_ids": list(self.item_ids),
            "graph_count": self.graph_count,
            "graph_aligned": self.graph_aligned,
            "source_fingerprint": (
                self.source_fingerprint
            ),
        }

    def semantic_fingerprint(self) -> str:
        return _fingerprint(
            self.semantic_dict()
        )

    def value_fingerprint(self) -> str:
        tensors: dict[str, torch.Tensor] = {}

        if self.node_batch_index is not None:
            tensors["node_batch_index"] = (
                self.node_batch_index
            )

        if not tensors:
            return _fingerprint(
                {
                    "no_tensor_alignment": True,
                    "item_count": self.item_count,
                }
            )

        return _tensor_fingerprint(tensors)

    def fingerprint(self) -> str:
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

    def to(
        self,
        device: torch.device | str,
    ) -> "NodeAlignment":
        return NodeAlignment(
            item_count=self.item_count,
            item_ids=self.item_ids,
            node_batch_index=(
                self.node_batch_index.to(
                    device=device
                )
                if self.node_batch_index is not None
                else None
            ),
            graph_count=self.graph_count,
            source_fingerprint=(
                self.source_fingerprint
            ),
            alignment_name=self.alignment_name,
            schema_version=self.schema_version,
        )


# =============================================================================
# Generic metadata-preserving component
# =============================================================================


@dataclass(slots=True, frozen=True)
class NodeStateComponent:
    """
    Generic metadata-preserving node/item state.

    This wrapper is used for static state and for future component types whose
    dedicated result object does not yet exist.
    """

    values: torch.Tensor
    component_name: str

    source_fingerprint: str | None = None
    alignment_fingerprint: str | None = None

    schema_version: str = (
        NODE_STATE_COMPONENT_SCHEMA_VERSION
    )

    def __post_init__(self) -> None:
        if not isinstance(
            self.values,
            torch.Tensor,
        ):
            raise TypeError(
                "values must be a tensor."
            )

        if self.values.ndim != 2:
            raise ValueError(
                "values must have shape [items, feature_dim]."
            )

        if not self.values.dtype.is_floating_point:
            raise ValueError(
                "values must use a floating-point dtype."
            )

        if int(self.values.shape[1]) <= 0:
            raise ValueError(
                "values must contain at least one feature."
            )

        _require_nonempty_string(
            "component_name",
            self.component_name,
        )

        _assert_finite_tensor(
            self.component_name,
            self.values,
        )

        if self.source_fingerprint is not None:
            _require_nonempty_string(
                "source_fingerprint",
                self.source_fingerprint,
            )

        if (
            self.alignment_fingerprint
            is not None
        ):
            _require_nonempty_string(
                "alignment_fingerprint",
                self.alignment_fingerprint,
            )

        _require_nonempty_string(
            "schema_version",
            self.schema_version,
        )

    @property
    def item_count(self) -> int:
        return int(
            self.values.shape[0]
        )

    @property
    def feature_dim(self) -> int:
        return int(
            self.values.shape[1]
        )

    @property
    def device(self) -> torch.device:
        return self.values.device

    def value_fingerprint(self) -> str:
        return _tensor_fingerprint(
            {
                self.component_name: (
                    self.values
                )
            }
        )

    def lineage_fingerprint(self) -> str:
        return _fingerprint(
            {
                "schema_version": self.schema_version,
                "component_name": self.component_name,
                "source_fingerprint": (
                    self.source_fingerprint
                ),
                "alignment_fingerprint": (
                    self.alignment_fingerprint
                ),
                "value_fingerprint": (
                    self.value_fingerprint()
                ),
            }
        )

    def to(
        self,
        device: torch.device | str,
    ) -> "NodeStateComponent":
        return NodeStateComponent(
            values=self.values.to(
                device=device
            ),
            component_name=self.component_name,
            source_fingerprint=(
                self.source_fingerprint
            ),
            alignment_fingerprint=(
                self.alignment_fingerprint
            ),
            schema_version=self.schema_version,
        )


# =============================================================================
# Complete fusion input
# =============================================================================


@dataclass(slots=True, frozen=True)
class NodeStateFusionInputs:
    """
    Metadata-preserving inputs for node-state fusion.

    ``memory_state`` and ``hazard_context`` retain their complete typed source
    objects. ``static_state`` and ``hazard_memory_state`` use
    ``NodeStateComponent`` until dedicated upstream result contracts exist.
    """

    alignment: NodeAlignment

    static_state: NodeStateComponent | None = None
    memory_state: LagMemoryEncoding | None = None
    hazard_memory_state: NodeStateComponent | None = None
    hazard_context: HazardQueryEncoding | None = None

    node_type_ids: torch.Tensor | None = None

    source_fingerprint: str | None = None
    schema_version: str = (
        NODE_STATE_FUSION_INPUT_SCHEMA_VERSION
    )

    def __post_init__(self) -> None:
        if not isinstance(
            self.alignment,
            NodeAlignment,
        ):
            raise TypeError(
                "alignment must be a NodeAlignment."
            )

        for name, value, expected_type in (
            (
                "static_state",
                self.static_state,
                NodeStateComponent,
            ),
            (
                "memory_state",
                self.memory_state,
                LagMemoryEncoding,
            ),
            (
                "hazard_memory_state",
                self.hazard_memory_state,
                NodeStateComponent,
            ),
            (
                "hazard_context",
                self.hazard_context,
                HazardQueryEncoding,
            ),
        ):
            if (
                value is not None
                and not isinstance(
                    value,
                    expected_type,
                )
            ):
                raise TypeError(
                    f"{name} must be a "
                    f"{expected_type.__name__} or None."
                )

        self._validate_node_types()
        self._validate_row_alignment()
        self._validate_graph_alignment()

        # Force eager cross-component device validation.
        _ = self.device

        if self.source_fingerprint is not None:
            _require_nonempty_string(
                "source_fingerprint",
                self.source_fingerprint,
            )

        _require_nonempty_string(
            "schema_version",
            self.schema_version,
        )

    def _component_item_counts(
        self,
    ) -> Mapping[str, int]:
        values: dict[str, int] = {}

        if self.static_state is not None:
            values["static_state"] = (
                self.static_state.item_count
            )

        if self.memory_state is not None:
            values["memory_state"] = (
                self.memory_state.item_count
            )

        if self.hazard_memory_state is not None:
            values["hazard_memory_state"] = (
                self.hazard_memory_state.item_count
            )

        if self.hazard_context is not None:
            values["hazard_context"] = (
                self.hazard_context.item_count
            )

        if self.node_type_ids is not None:
            values["node_type_ids"] = int(
                self.node_type_ids.shape[0]
            )

        return MappingProxyType(values)

    def _validate_row_alignment(self) -> None:
        expected = self.alignment.item_count

        mismatches = {
            name: count
            for name, count
            in self._component_item_counts().items()
            if count != expected
        }

        if mismatches:
            raise ValueError(
                "Node-state component row counts differ from "
                f"alignment.item_count={expected}: {mismatches}."
            )

        alignment_fingerprint = (
            self.alignment.fingerprint()
        )

        for name, component in (
            ("static_state", self.static_state),
            (
                "hazard_memory_state",
                self.hazard_memory_state,
            ),
        ):
            if (
                component is not None
                and component.alignment_fingerprint
                is not None
                and component.alignment_fingerprint
                != alignment_fingerprint
            ):
                raise ValueError(
                    f"{name} was created for a different node "
                    "alignment."
                )

    def _validate_graph_alignment(self) -> None:
        if self.hazard_context is None:
            return

        source = (
            self.hazard_context
            .source_embedding
        )

        if not isinstance(
            source,
            NodeAlignedHazardEmbeddingLookup,
        ):
            return

        if self.alignment.node_batch_index is None:
            raise ValueError(
                "Node-aligned hazard context requires "
                "alignment.node_batch_index."
            )

        if source.node_batch_index.device != (
            self.alignment
            .node_batch_index
            .device
        ):
            raise ValueError(
                "Hazard-context graph membership and NodeAlignment "
                "must share one device."
            )

        if not torch.equal(
            source.node_batch_index,
            self.alignment.node_batch_index,
        ):
            raise ValueError(
                "Hazard-context graph membership differs from "
                "NodeAlignment."
            )

    def _validate_node_types(self) -> None:
        if self.node_type_ids is None:
            return

        if not isinstance(
            self.node_type_ids,
            torch.Tensor,
        ):
            raise TypeError(
                "node_type_ids must be a tensor or None."
            )

        if self.node_type_ids.ndim != 1:
            raise ValueError(
                "node_type_ids must have shape [item_count]."
            )

        if self.node_type_ids.dtype != torch.long:
            raise ValueError(
                "node_type_ids must use torch.long."
            )

        if (
            self.alignment.device is not None
            and self.node_type_ids.device
            != self.alignment.device
        ):
            raise ValueError(
                "node_type_ids and NodeAlignment graph membership "
                "must share one device."
            )

    @property
    def item_count(self) -> int:
        return self.alignment.item_count

    @property
    def device(self) -> torch.device | None:
        devices: list[torch.device] = []

        if self.alignment.device is not None:
            devices.append(
                self.alignment.device
            )

        if self.static_state is not None:
            devices.append(
                self.static_state.device
            )

        if self.memory_state is not None:
            devices.append(
                self.memory_state.memory_state.device
            )

        if self.hazard_memory_state is not None:
            devices.append(
                self.hazard_memory_state.device
            )

        if self.hazard_context is not None:
            devices.append(
                self.hazard_context.query.device
            )

        if self.node_type_ids is not None:
            devices.append(
                self.node_type_ids.device
            )

        if not devices:
            return None

        first = devices[0]

        if any(
            device != first
            for device in devices[1:]
        ):
            raise ValueError(
                "All node-state fusion inputs must share one device."
            )

        return first

    def component_lineage_dict(
        self,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "schema_version": self.schema_version,
            "alignment_fingerprint": (
                self.alignment.fingerprint()
            ),
            "source_fingerprint": (
                self.source_fingerprint
            ),
        }

        if self.static_state is not None:
            payload["static_state"] = (
                self.static_state
                .lineage_fingerprint()
            )

        if self.memory_state is not None:
            payload["memory_state"] = (
                self.memory_state
                .lineage_fingerprint
            )

        if self.hazard_memory_state is not None:
            payload["hazard_memory_state"] = (
                self.hazard_memory_state
                .lineage_fingerprint()
            )

        if self.hazard_context is not None:
            payload["hazard_context"] = (
                self.hazard_context
                .lineage_fingerprint
            )

        if self.node_type_ids is not None:
            payload["node_type_ids"] = (
                _tensor_fingerprint(
                    {
                        "node_type_ids": (
                            self.node_type_ids
                        )
                    }
                )
            )

        return payload

    def lineage_fingerprint(self) -> str:
        return _fingerprint(
            self.component_lineage_dict()
        )

    def to(
        self,
        device: torch.device | str,
    ) -> "NodeStateFusionInputs":
        return NodeStateFusionInputs(
            alignment=self.alignment.to(
                device
            ),
            static_state=(
                self.static_state.to(device)
                if self.static_state is not None
                else None
            ),
            memory_state=(
                _move_lag_memory_encoding(
                    self.memory_state,
                    device=device,
                )
                if self.memory_state is not None
                else None
            ),
            hazard_memory_state=(
                self.hazard_memory_state.to(
                    device
                )
                if (
                    self.hazard_memory_state
                    is not None
                )
                else None
            ),
            hazard_context=(
                _move_hazard_query_encoding(
                    self.hazard_context,
                    device=device,
                )
                if self.hazard_context is not None
                else None
            ),
            node_type_ids=(
                self.node_type_ids.to(
                    device=device
                )
                if self.node_type_ids is not None
                else None
            ),
            source_fingerprint=(
                self.source_fingerprint
            ),
            schema_version=self.schema_version,
        )


def _move_lag_memory_encoding(
    encoding: LagMemoryEncoding,
    *,
    device: torch.device | str,
) -> LagMemoryEncoding:
    return LagMemoryEncoding(
        memory_state=(
            encoding.memory_state.to(
                device=device
            )
        ),
        source_batch=(
            encoding.source_batch.to(
                device
            )
        ),
        lag_feature_states=(
            encoding.lag_feature_states.to(
                device=device
            )
            if encoding.lag_feature_states is not None
            else None
        ),
        lag_weights=(
            encoding.lag_weights.to(
                device=device
            )
            if encoding.lag_weights is not None
            else None
        ),
        encoder_architecture_fingerprint=(
            encoding.encoder_architecture_fingerprint
        ),
        lineage_fingerprint=(
            encoding.lineage_fingerprint
        ),
        schema_version=(
            encoding.schema_version
        ),
    )


def _move_hazard_query_encoding(
    encoding: HazardQueryEncoding,
    *,
    device: torch.device | str,
) -> HazardQueryEncoding:
    """
    Preserve the query contract without fabricating source-embedding metadata.

    ``HazardQueryEncoding`` currently has no public ``to`` method. The full
    source embedding result must therefore already be moved by its owning
    hazard module before a query result can be reconstructed elsewhere.
    """

    requested_device = torch.device(device)

    if encoding.query.device != requested_device:
        raise ValueError(
            "Move the metadata-preserving hazard embedding/query "
            "upstream before moving NodeStateFusionInputs."
        )

    return encoding


# =============================================================================
# Fusion output
# =============================================================================


@dataclass(slots=True, frozen=True)
class NodeStateFusionOutput:
    """
    Fused node state plus source inputs and projected component states.

    ``fusion_mode`` is normalized lazily against ``NodeStateFusionMode`` to
    avoid a module-import cycle while preserving the public enum contract.
    """

    fused_state: torch.Tensor
    source_inputs: NodeStateFusionInputs

    projected_components: Mapping[
        str,
        torch.Tensor,
    ]

    fusion_mode: "NodeStateFusionMode | str"
    encoder_architecture_fingerprint: str
    lineage_fingerprint: str

    schema_version: str = (
        NODE_STATE_FUSION_OUTPUT_SCHEMA_VERSION
    )

    def __post_init__(self) -> None:
        if not isinstance(
            self.fused_state,
            torch.Tensor,
        ):
            raise TypeError(
                "fused_state must be a tensor."
            )

        if self.fused_state.ndim != 2:
            raise ValueError(
                "fused_state must have shape "
                "[items, output_dim]."
            )

        if not (
            self.fused_state
            .dtype
            .is_floating_point
        ):
            raise ValueError(
                "fused_state must use a floating-point dtype."
            )

        if int(
            self.fused_state.shape[0]
        ) != self.source_inputs.item_count:
            raise ValueError(
                "fused_state rows must align with source_inputs."
            )

        if (
            self.source_inputs.device is not None
            and self.fused_state.device
            != self.source_inputs.device
        ):
            raise ValueError(
                "fused_state and source_inputs must share one device."
            )

        _assert_finite_tensor(
            "fused_state",
            self.fused_state,
        )

        if not isinstance(
            self.projected_components,
            Mapping,
        ):
            raise TypeError(
                "projected_components must be a mapping."
            )

        copied: dict[str, torch.Tensor] = {}

        for name, tensor in (
            self.projected_components.items()
        ):
            _require_nonempty_string(
                "projected component name",
                name,
            )

            if not isinstance(
                tensor,
                torch.Tensor,
            ):
                raise TypeError(
                    "Projected components must be tensors."
                )

            if tensor.ndim != 2:
                raise ValueError(
                    f"Projected component {name!r} must have "
                    "shape [items, output_dim]."
                )

            if int(tensor.shape[0]) != (
                self.source_inputs.item_count
            ):
                raise ValueError(
                    f"Projected component {name!r} rows do not "
                    "align with source_inputs."
                )

            if tensor.device != (
                self.fused_state.device
            ):
                raise ValueError(
                    f"Projected component {name!r} and fused_state "
                    "must share one device."
                )

            if not tensor.dtype.is_floating_point:
                raise ValueError(
                    f"Projected component {name!r} must use a "
                    "floating-point dtype."
                )

            _assert_finite_tensor(
                f"projected component {name}",
                tensor,
            )
            copied[name] = tensor

        if not copied:
            raise ValueError(
                "At least one projected component is required."
            )

        object.__setattr__(
            self,
            "projected_components",
            MappingProxyType(copied),
        )

        # Local import prevents schemas.py <-> node_state_fusion.py from
        # forming an import-time cycle.
        from .node_state_fusion import (
            NodeStateFusionMode,
        )

        normalized_mode = (
            self.fusion_mode
            if isinstance(
                self.fusion_mode,
                NodeStateFusionMode,
            )
            else NodeStateFusionMode(
                self.fusion_mode
            )
        )

        object.__setattr__(
            self,
            "fusion_mode",
            normalized_mode,
        )

        for name, value in (
            (
                "encoder_architecture_fingerprint",
                self.encoder_architecture_fingerprint,
            ),
            (
                "lineage_fingerprint",
                self.lineage_fingerprint,
            ),
            ("schema_version", self.schema_version),
        ):
            _require_nonempty_string(
                name,
                value,
            )

    @property
    def item_count(self) -> int:
        return int(
            self.fused_state.shape[0]
        )

    @property
    def output_dim(self) -> int:
        return int(
            self.fused_state.shape[1]
        )

    @property
    def alignment(self) -> NodeAlignment:
        return self.source_inputs.alignment


__all__ = (
    "NODE_ALIGNMENT_SCHEMA_VERSION",
    "NODE_STATE_COMPONENT_SCHEMA_VERSION",
    "NODE_STATE_FUSION_INPUT_SCHEMA_VERSION",
    "NODE_STATE_FUSION_OUTPUT_SCHEMA_VERSION",
    "NodeAlignment",
    "NodeStateComponent",
    "NodeStateFusionInputs",
    "NodeStateFusionOutput",
)
