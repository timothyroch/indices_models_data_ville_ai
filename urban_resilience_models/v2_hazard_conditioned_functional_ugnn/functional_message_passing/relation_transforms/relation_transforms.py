"""
Public relation-transform dispatcher for functional message passing.

Target path:
    urban_resilience_models/
        v2_hazard_conditioned_functional_ugnn/
            functional_message_passing/
                relation_transforms/
                    relation_transforms.py

This module is the metadata-preserving public entry point for relation
transforms. It selects one bounded implementation:

- ``shared``:
  one ``Linear(H, H)`` map shared by every relation;

- ``per_relation``:
  one independent ``Linear(H, H)`` map per dense compiled relation.

The implementation modules own the mathematics. This dispatcher owns:

- canonical versus implemented mode validation;
- construction from configuration;
- exact compiled-registry compatibility;
- implementation dispatch;
- metadata-bearing ``RelationTransformOutput`` construction;
- architecture, parameter, and relation-specific fingerprints;
- stable public state-dict structure.

It does not own:

- registry compilation or hierarchy selection;
- source-node fusion;
- structural edge normalization;
- hazard-conditioned relation gates;
- edge attention;
- message-factor multiplication;
- target-node aggregation;
- residual or layer-normalization updates.

Baseline equation
-----------------
For every edge ``e = (s_e -> t_e)`` with dense relation index ``r_e``:

Shared mode:

    u_e = W_shared h[s_e] + b_shared

Per-relation mode:

    u_e = W[r_e] h[s_e] + b[r_e]

The returned ``RelationTransformOutput`` preserves the complete
``FunctionalMessagePassingInputs`` contract and therefore the graph, node,
edge, relation, control, hazard-query, and lineage metadata.

No activation, normalization, dropout, gating, attention, aggregation, hidden
casting, device movement, fallback, or relation remapping occurs here.
"""

from __future__ import annotations

from hashlib import sha256
import json
from types import MappingProxyType
from typing import Any, Final, Mapping

import torch
from torch import nn

from ...config import (
    FunctionalMessagePassingConfig,
)
from ...constants import (
    CANONICAL_RELATION_TRANSFORM_TYPES,
    RELATION_TRANSFORM_PER_RELATION,
    RELATION_TRANSFORM_SHARED,
    V2_0_IMPLEMENTED_RELATION_TRANSFORM_TYPES,
)
from ...relations.relation_registry import (
    CompiledRelationRegistry,
)
from ..schemas import (
    FunctionalMessagePassingInputs,
    RelationTransformOutput,
)
from .per_relation_transform import (
    PerRelationTransform,
)
from .shared_transform import (
    SharedRelationTransform,
)


# =============================================================================
# Public identity
# =============================================================================


RELATION_TRANSFORMS_DISPATCHER_SCHEMA_VERSION: Final[str] = "0.1"

_IMPLEMENTATION_ATTRIBUTE: Final[str] = "implementation"


# =============================================================================
# Validation and fingerprint helpers
# =============================================================================


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


def _require_bool(
    name: str,
    value: bool,
) -> None:
    if not isinstance(value, bool):
        raise TypeError(
            f"{name} must be a Boolean."
        )


def _require_nonempty_string(
    name: str,
    value: str,
) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(
            f"{name} must be a non-empty string."
        )


def _normalize_mode(
    mode: str,
) -> str:
    if not isinstance(mode, str):
        raise TypeError(
            "relation transform mode must be a string."
        )

    normalized = mode.strip()

    if not normalized:
        raise ValueError(
            "relation transform mode must be a non-empty string."
        )

    if normalized not in (
        CANONICAL_RELATION_TRANSFORM_TYPES
    ):
        raise ValueError(
            "Unknown relation transform mode "
            f"{normalized!r}. Expected one of "
            f"{tuple(CANONICAL_RELATION_TRANSFORM_TYPES)!r}."
        )

    if normalized not in (
        V2_0_IMPLEMENTED_RELATION_TRANSFORM_TYPES
    ):
        raise NotImplementedError(
            "Relation transform mode "
            f"{normalized!r} is canonical but not implemented in V2.0."
        )

    return normalized


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


def _state_dict_fingerprint(
    state_dict: Mapping[str, torch.Tensor],
) -> str:
    digest = sha256()

    for name in sorted(state_dict):
        tensor = (
            state_dict[name]
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


# =============================================================================
# Dispatcher
# =============================================================================


class RelationTransforms(nn.Module):
    """
    Dispatch functional source-state transformation by configured mode.

    Parameters
    ----------
    mode:
        Canonical relation-transform mode.

    hidden_dim:
        Input and output width ``H``.

    compiled_relation_registry:
        Exact dense relation ordering consumed by graph edge relation indices.

    bias:
        Whether the selected linear transform implementation includes bias.

    Notes
    -----
    Even shared mode retains the compiled registry as architecture metadata.
    The shared map does not use relation IDs mathematically, but the complete
    model artifact must still preserve which edge ontology and dense ordering
    produced its transformed edge states.
    """

    mode: str
    hidden_dim: int
    bias: bool

    def __init__(
        self,
        *,
        mode: str,
        hidden_dim: int,
        compiled_relation_registry: (
            CompiledRelationRegistry
        ),
        bias: bool = True,
    ) -> None:
        super().__init__()

        normalized_mode = _normalize_mode(
            mode
        )
        _require_positive_int(
            "hidden_dim",
            hidden_dim,
        )
        _require_bool(
            "bias",
            bias,
        )

        if not isinstance(
            compiled_relation_registry,
            CompiledRelationRegistry,
        ):
            raise TypeError(
                "compiled_relation_registry must be a "
                "CompiledRelationRegistry."
            )

        compiled_relation_registry.validate()

        if len(
            compiled_relation_registry
        ) <= 0:
            raise ValueError(
                "At least one compiled relation is required."
            )

        relation_names = tuple(
            compiled_relation_registry
            .relation_names
        )
        stable_relation_ids = tuple(
            compiled_relation_registry
            .stable_relation_ids
        )

        if len(relation_names) != len(
            stable_relation_ids
        ):
            raise ValueError(
                "Compiled relation names and stable IDs must align."
            )

        if len(relation_names) != len(
            compiled_relation_registry
        ):
            raise ValueError(
                "Compiled relation metadata must align with registry "
                "entries."
            )

        for index, name in enumerate(
            relation_names
        ):
            _require_nonempty_string(
                f"relation_names[{index}]",
                name,
            )

        if len(set(relation_names)) != len(
            relation_names
        ):
            raise ValueError(
                "Compiled relation names must be unique."
            )

        for index, relation_id in enumerate(
            stable_relation_ids
        ):
            if (
                isinstance(relation_id, bool)
                or not isinstance(
                    relation_id,
                    int,
                )
            ):
                raise TypeError(
                    f"stable_relation_ids[{index}] must be an integer."
                )

            if relation_id < 0:
                raise ValueError(
                    f"stable_relation_ids[{index}] must be nonnegative."
                )

        if len(
            set(stable_relation_ids)
        ) != len(stable_relation_ids):
            raise ValueError(
                "Compiled stable relation IDs must be unique."
            )

        control_relation_mask = tuple(
            bool(
                entry
                .specification
                .is_control
            )
            for entry
            in compiled_relation_registry.entries
        )

        if len(control_relation_mask) != len(
            relation_names
        ):
            raise ValueError(
                "Compiled control-relation metadata must align with "
                "relation ordering."
            )

        self.mode = normalized_mode
        self.hidden_dim = hidden_dim
        self.bias = bias

        self.compiled_relation_registry = (
            compiled_relation_registry
        )
        self.relation_names = (
            relation_names
        )
        self.stable_relation_ids = (
            stable_relation_ids
        )
        self.control_relation_mask = (
            control_relation_mask
        )
        self.compiled_relation_registry_fingerprint = (
            compiled_relation_registry
            .fingerprint()
        )

        if self.mode == (
            RELATION_TRANSFORM_SHARED
        ):
            implementation: nn.Module = (
                SharedRelationTransform(
                    hidden_dim=hidden_dim,
                    bias=bias,
                )
            )
        elif self.mode == (
            RELATION_TRANSFORM_PER_RELATION
        ):
            implementation = (
                PerRelationTransform(
                    hidden_dim=hidden_dim,
                    relation_names=(
                        relation_names
                    ),
                    stable_relation_ids=(
                        stable_relation_ids
                    ),
                    control_relation_mask=(
                        control_relation_mask
                    ),
                    bias=bias,
                )
            )
        else:
            # _normalize_mode already rejects unknown and canonical
            # unimplemented modes. This is a defensive exhaustiveness check.
            raise RuntimeError(
                "Internal relation-transform dispatch is incomplete for "
                f"mode {self.mode!r}."
            )

        setattr(
            self,
            _IMPLEMENTATION_ATTRIBUTE,
            implementation,
        )

    # ------------------------------------------------------------------
    # Construction from configuration
    # ------------------------------------------------------------------

    @classmethod
    def from_config(
        cls,
        *,
        config: FunctionalMessagePassingConfig,
        hidden_dim: int,
        compiled_relation_registry: (
            CompiledRelationRegistry
        ),
        bias: bool = True,
    ) -> "RelationTransforms":
        """
        Build the dispatcher from the canonical FMP configuration.

        The configuration object remains the source of truth for the selected
        mode. ``hidden_dim`` and the compiled registry are runtime model
        contracts supplied by the surrounding model builder.
        """

        if not isinstance(
            config,
            FunctionalMessagePassingConfig,
        ):
            raise TypeError(
                "config must be a FunctionalMessagePassingConfig."
            )

        config.validate()

        if config.enabled:
            config.assert_implemented()

        return cls(
            mode=(
                config.relation_transform_type
            ),
            hidden_dim=hidden_dim,
            compiled_relation_registry=(
                compiled_relation_registry
            ),
            bias=bias,
        )

    # ------------------------------------------------------------------
    # Public implementation identity
    # ------------------------------------------------------------------

    @property
    def implementation(
        self,
    ) -> SharedRelationTransform | PerRelationTransform:
        module = self._modules[
            _IMPLEMENTATION_ATTRIBUTE
        ]

        if isinstance(
            module,
            (
                SharedRelationTransform,
                PerRelationTransform,
            ),
        ):
            return module

        raise RuntimeError(
            "Internal relation-transform implementation has an invalid "
            f"type {type(module)!r}."
        )

    @implementation.setter
    def implementation(
        self,
        module: nn.Module,
    ) -> None:
        self.add_module(
            _IMPLEMENTATION_ATTRIBUTE,
            module,
        )

    @property
    def input_dim(self) -> int:
        return self.hidden_dim

    @property
    def output_dim(self) -> int:
        return self.hidden_dim

    @property
    def relation_count(self) -> int:
        return len(self.relation_names)

    @property
    def device(self) -> torch.device:
        return self.implementation.device

    @property
    def dtype(self) -> torch.dtype:
        return self.implementation.dtype

    @property
    def parameter_count(self) -> int:
        return sum(
            parameter.numel()
            for parameter in self.parameters()
        )

    @property
    def trainable_parameter_count(
        self,
    ) -> int:
        return sum(
            parameter.numel()
            for parameter in self.parameters()
            if parameter.requires_grad
        )

    @property
    def is_shared(self) -> bool:
        return self.mode == (
            RELATION_TRANSFORM_SHARED
        )

    @property
    def is_per_relation(self) -> bool:
        return self.mode == (
            RELATION_TRANSFORM_PER_RELATION
        )

    # ------------------------------------------------------------------
    # Compatibility validation
    # ------------------------------------------------------------------

    def _validate_inputs(
        self,
        inputs: FunctionalMessagePassingInputs,
    ) -> None:
        if not isinstance(
            inputs,
            FunctionalMessagePassingInputs,
        ):
            raise TypeError(
                "inputs must be a FunctionalMessagePassingInputs."
            )

        if inputs.hidden_dim != self.hidden_dim:
            raise ValueError(
                "Functional message-passing node-state width differs "
                "from RelationTransforms.hidden_dim. "
                f"Observed {inputs.hidden_dim}; expected "
                f"{self.hidden_dim}."
            )

        if inputs.relation_names != (
            self.relation_names
        ):
            raise ValueError(
                "Input relation ordering differs from the relation "
                "transform's compiled registry."
            )

        if inputs.stable_relation_ids != (
            self.stable_relation_ids
        ):
            raise ValueError(
                "Input stable relation IDs differ from the relation "
                "transform's compiled registry."
            )

        input_registry_fingerprint = (
            inputs
            .compiled_relation_registry
            .fingerprint()
        )

        if input_registry_fingerprint != (
            self
            .compiled_relation_registry_fingerprint
        ):
            raise ValueError(
                "Inputs reference a different compiled relation registry."
            )

        observed_control_mask = tuple(
            bool(value)
            for value in (
                inputs
                .control_relation_mask
                .detach()
                .cpu()
                .tolist()
            )
        )

        if observed_control_mask != (
            self.control_relation_mask
        ):
            raise ValueError(
                "Input control-relation metadata differs from the "
                "relation transform's compiled registry."
            )

        if inputs.device != self.device:
            raise ValueError(
                "Functional message-passing inputs and relation-transform "
                "parameters must share one device. "
                f"Observed {inputs.device} and {self.device}."
            )

        if inputs.dtype != self.dtype:
            raise ValueError(
                "Functional message-passing node-state dtype must match "
                "relation-transform parameter dtype. "
                f"Observed {inputs.dtype} and {self.dtype}."
            )

    # ------------------------------------------------------------------
    # Architecture and parameter identity
    # ------------------------------------------------------------------

    def implementation_architecture_dict(
        self,
    ) -> dict[str, Any]:
        architecture_method = getattr(
            self.implementation,
            "architecture_dict",
            None,
        )

        if not callable(
            architecture_method
        ):
            raise RuntimeError(
                "Relation-transform implementation does not expose "
                "architecture_dict()."
            )

        result = architecture_method()

        if not isinstance(result, dict):
            raise RuntimeError(
                "Implementation architecture_dict() must return a dict."
            )

        return result

    def relation_architecture_fingerprints(
        self,
    ) -> Mapping[str, str]:
        if self.is_shared:
            return MappingProxyType({})

        implementation = self.implementation

        if not isinstance(
            implementation,
            PerRelationTransform,
        ):
            raise RuntimeError(
                "Per-relation mode has an invalid implementation type."
            )

        return (
            implementation
            .relation_architecture_fingerprints()
        )

    def relation_parameter_fingerprints(
        self,
    ) -> Mapping[str, str]:
        if self.is_shared:
            return MappingProxyType({})

        implementation = self.implementation

        if not isinstance(
            implementation,
            PerRelationTransform,
        ):
            raise RuntimeError(
                "Per-relation mode has an invalid implementation type."
            )

        return (
            implementation
            .relation_parameter_fingerprints()
        )

    def architecture_dict(
        self,
    ) -> dict[str, Any]:
        return {
            "schema_version": (
                RELATION_TRANSFORMS_DISPATCHER_SCHEMA_VERSION
            ),
            "mode": self.mode,
            "hidden_dim": self.hidden_dim,
            "bias": self.bias,
            "relation_count": (
                self.relation_count
            ),
            "relation_names": list(
                self.relation_names
            ),
            "stable_relation_ids": list(
                self.stable_relation_ids
            ),
            "control_relation_mask": list(
                self.control_relation_mask
            ),
            "compiled_relation_registry_fingerprint": (
                self
                .compiled_relation_registry_fingerprint
            ),
            "implementation_type": (
                type(self.implementation)
                .__name__
            ),
            "implementation_architecture": (
                self
                .implementation_architecture_dict()
            ),
            "relation_architecture_fingerprints": dict(
                self
                .relation_architecture_fingerprints()
            ),
            "output_schema": (
                "RelationTransformOutput"
            ),
            "operation_order": [
                "validate_fmp_inputs",
                "gather_source_node_state",
                "dispatch_transform_implementation",
                "construct_metadata_preserving_output",
            ],
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
        return _state_dict_fingerprint(
            self.state_dict()
        )

    def assert_finite_parameters(
        self,
    ) -> None:
        assertion = getattr(
            self.implementation,
            "assert_finite_parameters",
            None,
        )

        if not callable(assertion):
            raise RuntimeError(
                "Relation-transform implementation does not expose "
                "assert_finite_parameters()."
            )

        assertion()

    # ------------------------------------------------------------------
    # Mathematical dispatch
    # ------------------------------------------------------------------

    def transform_tensor(
        self,
        inputs: FunctionalMessagePassingInputs,
    ) -> torch.Tensor:
        """
        Return only transformed edge source states.

        This lower-level method remains metadata-aware by accepting the full
        ``FunctionalMessagePassingInputs`` contract. Public model code should
        normally call ``forward`` to receive ``RelationTransformOutput``.
        """

        self._validate_inputs(inputs)

        node_state = (
            inputs.node_state.fused_state
        )

        if self.is_shared:
            implementation = self.implementation

            if not isinstance(
                implementation,
                SharedRelationTransform,
            ):
                raise RuntimeError(
                    "Shared mode has an invalid implementation type."
                )

            transformed = implementation(
                node_state,
                inputs.source_index,
            )
        elif self.is_per_relation:
            implementation = self.implementation

            if not isinstance(
                implementation,
                PerRelationTransform,
            ):
                raise RuntimeError(
                    "Per-relation mode has an invalid implementation "
                    "type."
                )

            transformed = implementation(
                node_state,
                inputs.source_index,
                inputs.edge_relation_index,
            )
        else:
            raise RuntimeError(
                "Internal relation-transform dispatch reached an "
                f"unsupported mode {self.mode!r}."
            )

        expected_shape = (
            inputs.num_edges,
            self.hidden_dim,
        )

        if tuple(transformed.shape) != (
            expected_shape
        ):
            raise RuntimeError(
                "Relation-transform implementation returned shape "
                f"{tuple(transformed.shape)}; expected "
                f"{expected_shape}."
            )

        if transformed.device != (
            inputs.device
        ):
            raise RuntimeError(
                "Relation-transform implementation changed device."
            )

        if transformed.dtype != (
            inputs.dtype
        ):
            raise RuntimeError(
                "Relation-transform implementation changed dtype."
            )

        if not bool(
            torch.isfinite(transformed)
            .all()
            .item()
        ):
            raise FloatingPointError(
                "Relation-transform dispatcher received NaN or infinity "
                "from its implementation."
            )

        return transformed

    def forward(
        self,
        inputs: FunctionalMessagePassingInputs,
    ) -> RelationTransformOutput:
        """
        Transform edge source states and preserve complete input metadata.
        """

        transformed = self.transform_tensor(
            inputs
        )

        return RelationTransformOutput(
            transformed_source_state=(
                transformed
            ),
            source_inputs=inputs,
            transform_mode=self.mode,
            encoder_architecture_fingerprint=(
                self.architecture_fingerprint()
            ),
            parameter_fingerprint=(
                self.parameter_fingerprint()
            ),
            relation_parameter_fingerprints=(
                self
                .relation_parameter_fingerprints()
            ),
        )

    # ------------------------------------------------------------------
    # Representation
    # ------------------------------------------------------------------

    def extra_repr(self) -> str:
        return (
            f"mode={self.mode!r}, "
            f"hidden_dim={self.hidden_dim}, "
            f"relation_count={self.relation_count}, "
            f"control_relation_count="
            f"{sum(self.control_relation_mask)}, "
            f"bias={self.bias}"
        )


__all__ = (
    "RELATION_TRANSFORMS_DISPATCHER_SCHEMA_VERSION",
    "RelationTransforms",
)
