"""
Research-grade node-state fusion orchestrator for the V2 functional UGNN.

Target path:
    urban_resilience_models/
        v2_hazard_conditioned_functional_ugnn/
            fusion/
                node_state_fusion.py

This module is the stable public entry point for node-state fusion. It owns:

- canonical fusion-mode recognition;
- distinction between canonical and implemented modes;
- construction from ``ModelConfig``;
- enforcement of enabled/disabled component policies;
- extraction of aligned tensors from typed metadata-preserving inputs;
- optional node-type embedding lookup;
- dispatch to the selected fusion algorithm;
- assembly of ``NodeStateFusionOutput``;
- architecture, parameter, and lineage identities;
- compatibility helpers for checkpoints from the pre-split implementation.

It does not own:

- fusion schemas, which live in ``fusion.schemas``;
- component projection mathematics, which live in
  ``fusion.component_projection``;
- concat-projection mathematics, which live in
  ``fusion.concat_projection``;
- gated, FiLM, expert, attribution, or uncertainty fusion.

Public boundary
---------------
``NodeStateFusion.forward`` accepts only ``NodeStateFusionInputs``. It never
accepts a bare tensor, list, tuple, or untyped dictionary.

The orchestrator extracts active component tensors in an explicit canonical
order and dispatches them to ``ConcatProjectionFusion``. Complete upstream
objects remain attached to ``NodeStateFusionOutput.source_inputs``.

Canonical versus implemented modes
----------------------------------
All configuration-level fusion modes are recognized locally:

- ``concat_projection``;
- ``projected_sum``;
- ``gated_fusion``;
- ``film_conditioning``.

Only ``concat_projection`` is implemented in this version. Therefore:

- an unknown mode raises ``ValueError``;
- a known future mode raises ``NotImplementedError``.

Checkpoint migration
--------------------
The original monolithic implementation registered projectors and the fusion
network directly under ``NodeStateFusion``. The split implementation nests
those parameters under ``fusion_algorithm`` and gives projector layers stable
semantic names. ``upgrade_legacy_state_dict`` converts the old keys without
changing tensor values.
"""

from __future__ import annotations

from collections import OrderedDict
from enum import StrEnum
from hashlib import sha256
import json
from types import MappingProxyType
from typing import Any, Final, Mapping

import torch
from torch import nn

from .concat_projection import (
    CANONICAL_FUSION_COMPONENT_ORDER,
    FUSION_COMPONENT_HAZARD_CONTEXT,
    FUSION_COMPONENT_HAZARD_MEMORY_STATE,
    FUSION_COMPONENT_MEMORY_STATE,
    FUSION_COMPONENT_NODE_TYPE_EMBEDDING,
    FUSION_COMPONENT_STATIC_STATE,
    ConcatProjectionFusion,
)
from .schemas import (
    NodeStateFusionInputs,
    NodeStateFusionOutput,
)


# =============================================================================
# Schema and capability identity
# =============================================================================


NODE_STATE_FUSION_ENCODER_SCHEMA_VERSION: Final[str] = "0.2"
NODE_STATE_FUSION_LEGACY_STATE_DICT_SCHEMA_VERSION: Final[str] = "0.1"


class NodeStateFusionMode(StrEnum):
    """Canonical configuration-level node-state fusion modes."""

    CONCAT_PROJECTION = "concat_projection"
    PROJECTED_SUM = "projected_sum"
    GATED_FUSION = "gated_fusion"
    FILM_CONDITIONING = "film_conditioning"


CANONICAL_NODE_STATE_FUSION_MODES: Final[
    tuple[NodeStateFusionMode, ...]
] = tuple(NodeStateFusionMode)


IMPLEMENTED_NODE_STATE_FUSION_MODES: Final[
    tuple[NodeStateFusionMode, ...]
] = (
    NodeStateFusionMode.CONCAT_PROJECTION,
)


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


def _normalize_mode(
    value: NodeStateFusionMode | str,
) -> NodeStateFusionMode:
    if isinstance(
        value,
        NodeStateFusionMode,
    ):
        return value

    try:
        return NodeStateFusionMode(value)
    except ValueError as exc:
        raise ValueError(
            f"Unknown node-state fusion mode {value!r}. "
            "Expected one of "
            f"{tuple(mode.value for mode in NodeStateFusionMode)}."
        ) from exc


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
# Orchestrator
# =============================================================================


class NodeStateFusion(nn.Module):
    """
    Extract, validate, and fuse typed node-state components.

    ``concat_projection`` is the only implemented fusion algorithm in this
    version. The other canonical modes are recognized so configuration errors
    remain distinguishable from implementation gaps.
    """

    def __init__(
        self,
        *,
        mode: NodeStateFusionMode | str,
        output_dim: int,

        include_static_state: bool,
        static_input_dim: int | None,

        include_memory_state: bool,
        memory_input_dim: int | None,

        include_hazard_memory_state: bool,
        hazard_memory_input_dim: int | None,

        include_hazard_context: bool,
        hazard_context_input_dim: int | None,

        include_node_type_embedding: bool,
        node_type_count: int | None,
        node_type_embedding_dim: int = 16,

        dropout: float = 0.0,
        layer_norm: bool = True,
    ) -> None:
        super().__init__()

        self.mode = _normalize_mode(mode)

        if self.mode not in (
            IMPLEMENTED_NODE_STATE_FUSION_MODES
        ):
            raise NotImplementedError(
                f"Node-state fusion mode {self.mode.value!r} is "
                "canonical but not implemented. Implement and test "
                "its dedicated fusion module before enabling it."
            )

        _require_positive_int(
            "output_dim",
            output_dim,
        )

        for name, value in (
            (
                "include_static_state",
                include_static_state,
            ),
            (
                "include_memory_state",
                include_memory_state,
            ),
            (
                "include_hazard_memory_state",
                include_hazard_memory_state,
            ),
            (
                "include_hazard_context",
                include_hazard_context,
            ),
            (
                "include_node_type_embedding",
                include_node_type_embedding,
            ),
            ("layer_norm", layer_norm),
        ):
            if not isinstance(value, bool):
                raise TypeError(
                    f"{name} must be a Boolean."
                )

        if not any(
            (
                include_static_state,
                include_memory_state,
                include_hazard_memory_state,
                include_hazard_context,
                include_node_type_embedding,
            )
        ):
            raise ValueError(
                "NodeStateFusion must include at least one component."
            )

        self.output_dim = output_dim
        self.include_static_state = (
            include_static_state
        )
        self.include_memory_state = (
            include_memory_state
        )
        self.include_hazard_memory_state = (
            include_hazard_memory_state
        )
        self.include_hazard_context = (
            include_hazard_context
        )
        self.include_node_type_embedding = (
            include_node_type_embedding
        )
        self.layer_norm = layer_norm

        self.static_input_dim = (
            self._resolve_component_dimension(
                name="static_input_dim",
                included=include_static_state,
                value=static_input_dim,
            )
        )
        self.memory_input_dim = (
            self._resolve_component_dimension(
                name="memory_input_dim",
                included=include_memory_state,
                value=memory_input_dim,
            )
        )
        self.hazard_memory_input_dim = (
            self._resolve_component_dimension(
                name="hazard_memory_input_dim",
                included=(
                    include_hazard_memory_state
                ),
                value=hazard_memory_input_dim,
            )
        )
        self.hazard_context_input_dim = (
            self._resolve_component_dimension(
                name="hazard_context_input_dim",
                included=include_hazard_context,
                value=hazard_context_input_dim,
            )
        )

        if include_node_type_embedding:
            if node_type_count is None:
                raise ValueError(
                    "node_type_count is required when node-type "
                    "embedding is enabled."
                )

            _require_positive_int(
                "node_type_count",
                node_type_count,
            )
            _require_positive_int(
                "node_type_embedding_dim",
                node_type_embedding_dim,
            )
        else:
            if node_type_count is not None:
                raise ValueError(
                    "node_type_count must be None when node-type "
                    "embedding is disabled."
                )

            _require_positive_int(
                "node_type_embedding_dim",
                node_type_embedding_dim,
            )

        self.node_type_count = node_type_count
        self.node_type_embedding_dim = (
            node_type_embedding_dim
        )

        if include_node_type_embedding:
            assert node_type_count is not None
            self.node_type_embedding: (
                nn.Embedding | None
            ) = nn.Embedding(
                node_type_count,
                node_type_embedding_dim,
            )
        else:
            self.node_type_embedding = None

        component_input_dims = (
            self._build_component_input_dims()
        )

        self.fusion_algorithm = (
            ConcatProjectionFusion(
                component_input_dims=(
                    component_input_dims
                ),
                output_dim=output_dim,
                component_order=tuple(
                    component_input_dims
                ),
                dropout=dropout,
                layer_norm=layer_norm,
                retain_concatenated_state=False,
                record_input_fingerprint=False,
            )
        )

        self.dropout = (
            self.fusion_algorithm.dropout
        )
        self.component_order = (
            self.fusion_algorithm.component_order
        )
        self.component_count = (
            self.fusion_algorithm.component_count
        )

    @staticmethod
    def _resolve_component_dimension(
        *,
        name: str,
        included: bool,
        value: int | None,
    ) -> int | None:
        if included:
            if value is None:
                raise ValueError(
                    f"{name} is required when its component "
                    "is enabled."
                )

            _require_positive_int(
                name,
                value,
            )
            return value

        if value is not None:
            raise ValueError(
                f"{name} must be None when its component is "
                "disabled."
            )

        return None

    def _build_component_input_dims(
        self,
    ) -> OrderedDict[str, int]:
        dimensions: dict[str, int] = {}

        if self.static_input_dim is not None:
            dimensions[
                FUSION_COMPONENT_STATIC_STATE
            ] = self.static_input_dim

        if self.memory_input_dim is not None:
            dimensions[
                FUSION_COMPONENT_MEMORY_STATE
            ] = self.memory_input_dim

        if (
            self.hazard_memory_input_dim
            is not None
        ):
            dimensions[
                FUSION_COMPONENT_HAZARD_MEMORY_STATE
            ] = self.hazard_memory_input_dim

        if (
            self.hazard_context_input_dim
            is not None
        ):
            dimensions[
                FUSION_COMPONENT_HAZARD_CONTEXT
            ] = self.hazard_context_input_dim

        if self.include_node_type_embedding:
            dimensions[
                FUSION_COMPONENT_NODE_TYPE_EMBEDDING
            ] = self.node_type_embedding_dim

        return OrderedDict(
            (
                name,
                dimensions[name],
            )
            for name in CANONICAL_FUSION_COMPONENT_ORDER
            if name in dimensions
        )

    # ------------------------------------------------------------------
    # Construction from frozen configuration
    # ------------------------------------------------------------------

    @classmethod
    def from_config(
        cls,
        config: "ModelConfig",
    ) -> "NodeStateFusion":
        """
        Construct from the complete frozen ``ModelConfig``.

        The complete model configuration is required because source dimensions
        live in multiple sub-configurations.
        """

        from ..config import (
            CANONICAL_NODE_FUSION_MODES,
            NODE_FUSION_CONCAT_PROJECTION,
            NODE_FUSION_FILM,
            NODE_FUSION_GATED,
            NODE_FUSION_PROJECTED_SUM,
            ModelConfig,
        )

        if not isinstance(config, ModelConfig):
            raise TypeError(
                "config must be a ModelConfig."
            )

        config.validate()
        fusion = config.fusion

        local_values = tuple(
            mode.value
            for mode in NodeStateFusionMode
        )
        config_values = tuple(
            CANONICAL_NODE_FUSION_MODES
        )

        if local_values != config_values:
            raise RuntimeError(
                "Fusion-mode vocabulary drift detected between "
                "config.py and fusion.node_state_fusion. "
                f"Config: {config_values}; local: {local_values}."
            )

        expected_named_constants = (
            NODE_FUSION_CONCAT_PROJECTION,
            NODE_FUSION_PROJECTED_SUM,
            NODE_FUSION_GATED,
            NODE_FUSION_FILM,
        )

        if expected_named_constants != local_values:
            raise RuntimeError(
                "Named fusion constants in config.py differ from the "
                "local canonical enum."
            )

        mode = _normalize_mode(
            fusion.mode
        )

        if mode not in (
            IMPLEMENTED_NODE_STATE_FUSION_MODES
        ):
            raise NotImplementedError(
                f"Fusion mode {mode.value!r} is canonical but not "
                "implemented."
            )

        static_dim = (
            config.static_input_dim
            if fusion.include_static_state
            else None
        )

        if (
            fusion.include_static_state
            and static_dim is None
        ):
            raise ValueError(
                "static_input_dim must be resolved before "
                "NodeStateFusion construction."
            )

        if (
            fusion.include_memory_state
            and not config.memory.enabled
        ):
            raise ValueError(
                "include_memory_state=True requires an enabled "
                "memory encoder."
            )

        memory_dim = (
            config.memory.hidden_dim
            if fusion.include_memory_state
            else None
        )

        if (
            fusion.include_hazard_memory_state
            and not config.memory.enabled
        ):
            raise ValueError(
                "include_hazard_memory_state=True requires an "
                "enabled memory encoder."
            )

        if (
            fusion.include_hazard_memory_state
            and not config.hazard.enabled
        ):
            raise ValueError(
                "include_hazard_memory_state=True requires enabled "
                "hazard conditioning."
            )

        hazard_memory_dim = (
            config.memory.hidden_dim
            if (
                fusion
                .include_hazard_memory_state
            )
            else None
        )

        if (
            fusion.include_hazard_context
            and not config.hazard.enabled
        ):
            raise ValueError(
                "include_hazard_context=True requires enabled "
                "hazard conditioning."
            )

        hazard_context_dim = (
            config.hazard.output_dim
            if fusion.include_hazard_context
            else None
        )

        node_type_count = (
            config.node_type_count
            if (
                fusion
                .include_node_type_embedding
            )
            else None
        )

        if (
            fusion.include_node_type_embedding
            and node_type_count is None
        ):
            raise ValueError(
                "node_type_count must be resolved before "
                "NodeStateFusion construction."
            )

        return cls(
            mode=mode,
            output_dim=fusion.output_dim,
            include_static_state=(
                fusion.include_static_state
            ),
            static_input_dim=static_dim,
            include_memory_state=(
                fusion.include_memory_state
            ),
            memory_input_dim=memory_dim,
            include_hazard_memory_state=(
                fusion
                .include_hazard_memory_state
            ),
            hazard_memory_input_dim=(
                hazard_memory_dim
            ),
            include_hazard_context=(
                fusion.include_hazard_context
            ),
            hazard_context_input_dim=(
                hazard_context_dim
            ),
            include_node_type_embedding=(
                fusion
                .include_node_type_embedding
            ),
            node_type_count=node_type_count,
            node_type_embedding_dim=(
                fusion
                .node_type_embedding_dim
            ),
            dropout=fusion.dropout,
            layer_norm=fusion.layer_norm,
        )

    # ------------------------------------------------------------------
    # Compatibility properties
    # ------------------------------------------------------------------

    @property
    def component_input_dims(
        self,
    ) -> Mapping[str, int]:
        return (
            self.fusion_algorithm
            .component_input_dims
        )

    @property
    def component_projections(
        self,
    ) -> nn.ModuleDict:
        """
        Compatibility view of the split algorithm's component projectors.
        """

        return (
            self.fusion_algorithm
            .component_projections
        )

    @property
    def fusion_network(
        self,
    ) -> nn.Sequential:
        """
        Compatibility view of the split algorithm's final fusion network.
        """

        return (
            self.fusion_algorithm
            .fusion_network
        )

    @property
    def device(self) -> torch.device:
        return self.fusion_algorithm.device

    @property
    def dtype(self) -> torch.dtype:
        return self.fusion_algorithm.dtype

    # ------------------------------------------------------------------
    # Input-policy enforcement and component extraction
    # ------------------------------------------------------------------

    def _require_input_contract(
        self,
        inputs: NodeStateFusionInputs,
    ) -> None:
        contracts = (
            (
                FUSION_COMPONENT_STATIC_STATE,
                self.include_static_state,
                inputs.static_state,
            ),
            (
                FUSION_COMPONENT_MEMORY_STATE,
                self.include_memory_state,
                inputs.memory_state,
            ),
            (
                FUSION_COMPONENT_HAZARD_MEMORY_STATE,
                self.include_hazard_memory_state,
                inputs.hazard_memory_state,
            ),
            (
                FUSION_COMPONENT_HAZARD_CONTEXT,
                self.include_hazard_context,
                inputs.hazard_context,
            ),
            (
                "node_type_ids",
                self.include_node_type_embedding,
                inputs.node_type_ids,
            ),
        )

        for name, enabled, value in contracts:
            if enabled and value is None:
                raise ValueError(
                    f"{name} is required by the fusion "
                    "configuration."
                )

            if not enabled and value is not None:
                raise ValueError(
                    f"{name} was supplied, but its fusion "
                    "component is disabled."
                )

    def _validate_node_type_ids(
        self,
        node_type_ids: torch.Tensor,
    ) -> None:
        if self.node_type_count is None:
            raise RuntimeError(
                "Node-type embedding is unavailable."
            )

        if node_type_ids.device != self.device:
            raise ValueError(
                "node_type_ids and NodeStateFusion must share "
                "one device."
            )

        if node_type_ids.numel() == 0:
            return

        minimum = int(
            node_type_ids.min().item()
        )
        maximum = int(
            node_type_ids.max().item()
        )

        if (
            minimum < 0
            or maximum >= self.node_type_count
        ):
            raise IndexError(
                "node_type_ids contains values outside the "
                f"configured range [0, {self.node_type_count - 1}]."
            )

    def extract_component_tensors(
        self,
        inputs: NodeStateFusionInputs,
    ) -> Mapping[str, torch.Tensor]:
        """
        Extract active tensors in the exact algorithm component order.

        The returned mapping is immutable. No projection or fusion is
        performed by this method.
        """

        if not isinstance(
            inputs,
            NodeStateFusionInputs,
        ):
            raise TypeError(
                "inputs must be NodeStateFusionInputs."
            )

        self._require_input_contract(
            inputs
        )

        input_device = inputs.device

        if (
            input_device is not None
            and input_device != self.device
        ):
            raise ValueError(
                "Node-state inputs and NodeStateFusion must share "
                "one device."
            )

        extracted: OrderedDict[
            str,
            torch.Tensor,
        ] = OrderedDict()

        for component_name in self.component_order:
            if (
                component_name
                == FUSION_COMPONENT_STATIC_STATE
            ):
                if inputs.static_state is None:
                    raise RuntimeError(
                        "static_state passed contract validation but "
                        "is unavailable."
                    )
                values = inputs.static_state.values

            elif (
                component_name
                == FUSION_COMPONENT_MEMORY_STATE
            ):
                if inputs.memory_state is None:
                    raise RuntimeError(
                        "memory_state passed contract validation but "
                        "is unavailable."
                    )
                values = (
                    inputs.memory_state.memory_state
                )

            elif (
                component_name
                == FUSION_COMPONENT_HAZARD_MEMORY_STATE
            ):
                if inputs.hazard_memory_state is None:
                    raise RuntimeError(
                        "hazard_memory_state passed contract "
                        "validation but is unavailable."
                    )
                values = (
                    inputs
                    .hazard_memory_state
                    .values
                )

            elif (
                component_name
                == FUSION_COMPONENT_HAZARD_CONTEXT
            ):
                if inputs.hazard_context is None:
                    raise RuntimeError(
                        "hazard_context passed contract validation "
                        "but is unavailable."
                    )
                values = (
                    inputs.hazard_context.query
                )

            elif (
                component_name
                == FUSION_COMPONENT_NODE_TYPE_EMBEDDING
            ):
                if inputs.node_type_ids is None:
                    raise RuntimeError(
                        "node_type_ids passed contract validation "
                        "but are unavailable."
                    )

                if self.node_type_embedding is None:
                    raise RuntimeError(
                        "node_type_embedding is unavailable."
                    )

                self._validate_node_type_ids(
                    inputs.node_type_ids
                )
                values = self.node_type_embedding(
                    inputs.node_type_ids
                )

            else:
                raise RuntimeError(
                    f"Unhandled fusion component "
                    f"{component_name!r}."
                )

            if values.device != self.device:
                raise ValueError(
                    f"Fusion component {component_name!r} and "
                    "NodeStateFusion must share one device."
                )

            if not values.dtype.is_floating_point:
                raise ValueError(
                    f"Fusion component {component_name!r} must use "
                    "a floating-point dtype."
                )

            if int(values.shape[0]) != inputs.item_count:
                raise ValueError(
                    f"Fusion component {component_name!r} rows do "
                    "not align with NodeStateFusionInputs."
                )

            _assert_finite_tensor(
                f"fusion component {component_name}",
                values,
            )

            extracted[component_name] = values

        if tuple(extracted) != self.component_order:
            raise RuntimeError(
                "Extracted component order differs from the "
                "fusion architecture."
            )

        return MappingProxyType(
            dict(extracted)
        )

    # ------------------------------------------------------------------
    # Forward path
    # ------------------------------------------------------------------

    def forward(
        self,
        inputs: NodeStateFusionInputs,
    ) -> NodeStateFusionOutput:
        if not isinstance(
            inputs,
            NodeStateFusionInputs,
        ):
            raise TypeError(
                "NodeStateFusion requires NodeStateFusionInputs; "
                "bare tensors, lists, and dictionaries are not "
                "accepted."
            )

        components = (
            self.extract_component_tensors(
                inputs
            )
        )

        algorithm_output = (
            self.fusion_algorithm(
                components
            )
        )

        if algorithm_output.component_order != (
            self.component_order
        ):
            raise RuntimeError(
                "Fusion algorithm returned a different component "
                "order."
            )

        return NodeStateFusionOutput(
            fused_state=(
                algorithm_output.fused_state
            ),
            source_inputs=inputs,
            projected_components=(
                algorithm_output
                .projected_components
            ),
            fusion_mode=self.mode,
            encoder_architecture_fingerprint=(
                self.architecture_fingerprint()
            ),
            lineage_fingerprint=(
                self.lineage_fingerprint(
                    inputs
                )
            ),
        )

    # ------------------------------------------------------------------
    # Architecture and lineage identity
    # ------------------------------------------------------------------

    def architecture_dict(
        self,
    ) -> dict[str, Any]:
        return {
            "schema_version": (
                NODE_STATE_FUSION_ENCODER_SCHEMA_VERSION
            ),
            "mode": self.mode.value,
            "canonical_modes": [
                mode.value
                for mode
                in CANONICAL_NODE_STATE_FUSION_MODES
            ],
            "implemented_modes": [
                mode.value
                for mode
                in IMPLEMENTED_NODE_STATE_FUSION_MODES
            ],
            "output_dim": self.output_dim,
            "include_static_state": (
                self.include_static_state
            ),
            "static_input_dim": (
                self.static_input_dim
            ),
            "include_memory_state": (
                self.include_memory_state
            ),
            "memory_input_dim": (
                self.memory_input_dim
            ),
            "include_hazard_memory_state": (
                self.include_hazard_memory_state
            ),
            "hazard_memory_input_dim": (
                self.hazard_memory_input_dim
            ),
            "include_hazard_context": (
                self.include_hazard_context
            ),
            "hazard_context_input_dim": (
                self.hazard_context_input_dim
            ),
            "include_node_type_embedding": (
                self.include_node_type_embedding
            ),
            "node_type_count": (
                self.node_type_count
            ),
            "node_type_embedding_dim": (
                self.node_type_embedding_dim
            ),
            "component_order": list(
                self.component_order
            ),
            "dropout": self.dropout,
            "layer_norm": self.layer_norm,
            "fusion_algorithm": (
                self.fusion_algorithm
                .architecture_dict()
            ),
        }

    def architecture_fingerprint(
        self,
    ) -> str:
        return _fingerprint(
            self.architecture_dict()
        )

    def parameter_fingerprint(
        self,
    ) -> str:
        return _tensor_fingerprint(
            {
                name: tensor
                for name, tensor
                in self.state_dict().items()
            }
        )

    def lineage_fingerprint(
        self,
        inputs: NodeStateFusionInputs,
    ) -> str:
        if not isinstance(
            inputs,
            NodeStateFusionInputs,
        ):
            raise TypeError(
                "inputs must be NodeStateFusionInputs."
            )

        return _fingerprint(
            {
                "fusion_architecture_fingerprint": (
                    self.architecture_fingerprint()
                ),
                "input_lineage_fingerprint": (
                    inputs.lineage_fingerprint()
                ),
                "alignment_fingerprint": (
                    inputs.alignment.fingerprint()
                ),
            }
        )

    def component_architecture_fingerprints(
        self,
    ) -> Mapping[str, str]:
        return (
            self.fusion_algorithm
            .component_architecture_fingerprints()
        )

    def component_parameter_fingerprints(
        self,
    ) -> Mapping[str, str]:
        return (
            self.fusion_algorithm
            .component_parameter_fingerprints()
        )

    def assert_finite_parameters(
        self,
    ) -> None:
        for name, tensor in (
            self.state_dict().items()
        ):
            if (
                tensor.dtype.is_floating_point
                and not bool(
                    torch.isfinite(tensor)
                    .all()
                    .item()
                )
            ):
                raise ValueError(
                    f"NodeStateFusion tensor {name!r} contains "
                    "NaN or infinity."
                )

        self.fusion_algorithm.assert_finite_parameters()

    # ------------------------------------------------------------------
    # Legacy checkpoint migration
    # ------------------------------------------------------------------

    def upgrade_legacy_state_dict(
        self,
        legacy_state_dict: Mapping[
            str,
            torch.Tensor,
        ],
    ) -> OrderedDict[str, torch.Tensor]:
        """
        Convert keys from the pre-split monolithic implementation.

        Tensor values are not copied or transformed. Unknown keys are retained
        unchanged so PyTorch's strict loading diagnostics remain informative.
        """

        if not isinstance(
            legacy_state_dict,
            Mapping,
        ):
            raise TypeError(
                "legacy_state_dict must be a mapping."
            )

        upgraded: OrderedDict[
            str,
            torch.Tensor,
        ] = OrderedDict()

        for key, value in (
            legacy_state_dict.items()
        ):
            if not isinstance(key, str):
                raise TypeError(
                    "State-dict keys must be strings."
                )

            if not isinstance(
                value,
                torch.Tensor,
            ):
                raise TypeError(
                    f"State-dict value for {key!r} must be a tensor."
                )

            new_key = self._upgrade_legacy_key(
                key
            )

            if new_key in upgraded:
                raise ValueError(
                    "Legacy checkpoint conversion produced a key "
                    f"collision at {new_key!r}."
                )

            upgraded[new_key] = value

        return upgraded

    def _upgrade_legacy_key(
        self,
        key: str,
    ) -> str:
        if key.startswith(
            "fusion_algorithm."
        ):
            return key

        if key.startswith(
            "node_type_embedding."
        ):
            return key

        for component_name in self.component_order:
            old_linear_prefix = (
                f"component_projections."
                f"{component_name}.network.0."
            )
            if key.startswith(
                old_linear_prefix
            ):
                suffix = key[
                    len(old_linear_prefix):
                ]
                return (
                    "fusion_algorithm."
                    "component_projections."
                    f"{component_name}.linear."
                    f"{suffix}"
                )

            old_norm_prefix = (
                f"component_projections."
                f"{component_name}.network.2."
            )
            if key.startswith(
                old_norm_prefix
            ):
                suffix = key[
                    len(old_norm_prefix):
                ]
                return (
                    "fusion_algorithm."
                    "component_projections."
                    f"{component_name}."
                    "normalization_layer."
                    f"{suffix}"
                )

        legacy_fusion_mapping = (
            (
                "fusion_network.0.",
                (
                    "fusion_algorithm."
                    "fusion_network.linear_in."
                ),
            ),
            (
                "fusion_network.3.",
                (
                    "fusion_algorithm."
                    "fusion_network.linear_out."
                ),
            ),
            (
                "fusion_network.4.",
                (
                    "fusion_algorithm."
                    "fusion_network.normalization."
                ),
            ),
        )

        for old_prefix, new_prefix in (
            legacy_fusion_mapping
        ):
            if key.startswith(old_prefix):
                return (
                    new_prefix
                    + key[len(old_prefix):]
                )

        return key

    def load_legacy_state_dict(
        self,
        legacy_state_dict: Mapping[
            str,
            torch.Tensor,
        ],
        *,
        strict: bool = True,
    ) -> nn.modules.module._IncompatibleKeys:
        """
        Upgrade and load a pre-split monolithic checkpoint.
        """

        upgraded = self.upgrade_legacy_state_dict(
            legacy_state_dict
        )
        return self.load_state_dict(
            upgraded,
            strict=strict,
        )

    def extra_repr(
        self,
    ) -> str:
        return (
            f"mode={self.mode.value!r}, "
            f"output_dim={self.output_dim}, "
            f"component_order={self.component_order}"
        )


__all__ = (
    "CANONICAL_NODE_STATE_FUSION_MODES",
    "IMPLEMENTED_NODE_STATE_FUSION_MODES",
    "NODE_STATE_FUSION_ENCODER_SCHEMA_VERSION",
    "NODE_STATE_FUSION_LEGACY_STATE_DICT_SCHEMA_VERSION",
    "NodeStateFusion",
    "NodeStateFusionInputs",
    "NodeStateFusionMode",
    "NodeStateFusionOutput",
)
