"""
Independently parameterized relation transforms for functional message passing.

Target path:
    urban_resilience_models/
        v2_hazard_conditioned_functional_ugnn/
            functional_message_passing/
                relation_transforms/
                    per_relation_transform.py

For every stored directed edge ``e``:

    h_source[e] = node_state[source_index[e]]
    transformed[e] =
        W[edge_relation_index[e]] h_source[e]
        + b[edge_relation_index[e]]

Every active compiled relation owns an independent ``Linear(H, H)`` module.
Dense relation index ``r`` is interpreted only through the constructor's
ordered relation metadata:

    r
    → relation_names[r]
    → stable_relation_ids[r]
    → relation_transforms[module_key[r]]

Stable ontology IDs are metadata and state-dict identity components. They are
never used directly as dense tensor indices.

This module owns:

- deterministic relation ordering;
- one independent linear map per compiled relation;
- control-relation identity metadata;
- empty edge groups;
- relation-specific architecture and parameter fingerprints;
- edge-aligned source gathering and transform application.

It does not own:

- registry compilation or hierarchy selection;
- relation-mode dispatch;
- structural normalization;
- hazard-conditioned gates or priors;
- edge attention;
- message multiplication;
- target-node aggregation;
- metadata-bearing ``RelationTransformOutput`` construction.

Contract
--------
- ``node_state`` has shape ``[N, H]`` and floating dtype.
- ``source_index`` and ``edge_relation_index`` have shape ``[E]`` and
  dtype ``torch.long``.
- source indices lie in ``[0, N - 1]``.
- relation indices lie in ``[0, R - 1]``.
- one ordered relation metadata entry exists for every parameter module.
- zero-edge batches and zero-edge relation groups are valid.
- tensors and module parameters share one device and floating dtype.
- finite inputs, parameters, and outputs are required.
- no hidden casting, device movement, activation, normalization, dropout,
  fallback, or relation remapping occurs.
"""

from __future__ import annotations

from hashlib import sha256
import json
from types import MappingProxyType
from typing import Any, Final, Mapping, Sequence

import torch
from torch import nn

from ...constants import (
    RELATION_TRANSFORM_PER_RELATION,
)
from ...relations.relation_registry import (
    CompiledRelationRegistry,
)


# =============================================================================
# Public identity
# =============================================================================


PER_RELATION_TRANSFORM_SCHEMA_VERSION: Final[str] = "0.1"


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


def _require_relation_names(
    relation_names: Sequence[str],
) -> tuple[str, ...]:
    if isinstance(
        relation_names,
        (str, bytes),
    ) or not isinstance(
        relation_names,
        Sequence,
    ):
        raise TypeError(
            "relation_names must be a sequence of strings."
        )

    resolved = tuple(relation_names)

    if not resolved:
        raise ValueError(
            "At least one relation is required."
        )

    for index, name in enumerate(resolved):
        _require_nonempty_string(
            f"relation_names[{index}]",
            name,
        )

    if len(set(resolved)) != len(resolved):
        raise ValueError(
            "relation_names must be unique."
        )

    return resolved


def _require_stable_relation_ids(
    stable_relation_ids: Sequence[int],
    *,
    relation_count: int,
) -> tuple[int, ...]:
    if isinstance(
        stable_relation_ids,
        (str, bytes),
    ) or not isinstance(
        stable_relation_ids,
        Sequence,
    ):
        raise TypeError(
            "stable_relation_ids must be a sequence of integers."
        )

    resolved = tuple(stable_relation_ids)

    if len(resolved) != relation_count:
        raise ValueError(
            "stable_relation_ids must align exactly with relation_names."
        )

    for index, relation_id in enumerate(
        resolved
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

    if len(set(resolved)) != len(resolved):
        raise ValueError(
            "stable_relation_ids must be unique."
        )

    return resolved


def _require_control_mask(
    control_relation_mask: Sequence[bool] | None,
    *,
    relation_count: int,
) -> tuple[bool, ...]:
    if control_relation_mask is None:
        return tuple(
            False
            for _ in range(relation_count)
        )

    if isinstance(
        control_relation_mask,
        (str, bytes),
    ) or not isinstance(
        control_relation_mask,
        Sequence,
    ):
        raise TypeError(
            "control_relation_mask must be a sequence of Booleans."
        )

    resolved = tuple(
        control_relation_mask
    )

    if len(resolved) != relation_count:
        raise ValueError(
            "control_relation_mask must align exactly with "
            "relation_names."
        )

    for index, value in enumerate(resolved):
        if not isinstance(value, bool):
            raise TypeError(
                f"control_relation_mask[{index}] must be a Boolean."
            )

    return resolved


def _require_node_state(
    node_state: torch.Tensor,
    *,
    hidden_dim: int,
    name: str = "node_state",
) -> None:
    if not isinstance(
        node_state,
        torch.Tensor,
    ):
        raise TypeError(
            f"{name} must be a tensor."
        )

    if node_state.ndim != 2:
        raise ValueError(
            f"{name} must have shape [M, H]; "
            f"observed {tuple(node_state.shape)}."
        )

    if int(node_state.shape[1]) != hidden_dim:
        raise ValueError(
            f"{name} feature width does not match the per-relation "
            f"transform hidden_dim. Observed "
            f"{int(node_state.shape[1])}; expected {hidden_dim}."
        )

    if not node_state.dtype.is_floating_point:
        raise ValueError(
            f"{name} must use a floating-point dtype."
        )

    if not bool(
        torch.isfinite(node_state)
        .all()
        .item()
    ):
        raise ValueError(
            f"{name} must contain only finite values."
        )


def _require_index_vector(
    name: str,
    value: torch.Tensor,
    *,
    item_count: int | None = None,
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
            f"{name} must have shape [E]; "
            f"observed {tuple(value.shape)}."
        )

    if value.dtype != torch.long:
        raise ValueError(
            f"{name} must use torch.long."
        )

    if (
        item_count is not None
        and int(value.shape[0]) != item_count
    ):
        raise ValueError(
            f"{name} length must equal {item_count}; "
            f"observed {int(value.shape[0])}."
        )


def _require_index_range(
    name: str,
    value: torch.Tensor,
    *,
    upper_bound: int,
) -> None:
    if value.numel() == 0:
        return

    if upper_bound <= 0:
        raise ValueError(
            f"{name} cannot be nonempty when its valid range is empty."
        )

    minimum = int(
        value.min().item()
    )
    maximum = int(
        value.max().item()
    )

    if minimum < 0 or maximum >= upper_bound:
        raise ValueError(
            f"{name} contains out-of-range indices. "
            f"Observed range [{minimum}, {maximum}]; "
            f"valid range is [0, {upper_bound - 1}]."
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


def _relation_module_key(
    *,
    relation_index: int,
    stable_relation_id: int,
) -> str:
    """
    Stable state-dict-safe key.

    Dense index is included because it defines tensor-axis position. Stable
    relation ID is included because it defines ontology identity.
    """

    return (
        f"relation_{relation_index:04d}"
        f"_id_{stable_relation_id}"
    )


# =============================================================================
# Per-relation transform
# =============================================================================


class PerRelationTransform(nn.Module):
    """
    Apply an independent linear map for every compiled relation.

    Parameters
    ----------
    hidden_dim:
        Input and output width ``H``.

    relation_names:
        Canonical relation names in dense compiled relation order.

    stable_relation_ids:
        Stable ontology IDs aligned to ``relation_names``.

    control_relation_mask:
        Optional Boolean metadata aligned to ``relation_names``. Control
        relations receive ordinary independent parameters but remain marked
        for diagnostics and explanation policy.

    bias:
        Whether each relation-specific linear map has an additive bias.
    """

    hidden_dim: int
    bias: bool
    relation_names: tuple[str, ...]
    stable_relation_ids: tuple[int, ...]
    control_relation_mask: tuple[bool, ...]
    relation_module_keys: tuple[str, ...]

    def __init__(
        self,
        *,
        hidden_dim: int,
        relation_names: Sequence[str],
        stable_relation_ids: Sequence[int],
        control_relation_mask: (
            Sequence[bool] | None
        ) = None,
        bias: bool = True,
    ) -> None:
        super().__init__()

        _require_positive_int(
            "hidden_dim",
            hidden_dim,
        )
        _require_bool(
            "bias",
            bias,
        )

        resolved_names = (
            _require_relation_names(
                relation_names
            )
        )
        resolved_ids = (
            _require_stable_relation_ids(
                stable_relation_ids,
                relation_count=len(
                    resolved_names
                ),
            )
        )
        resolved_controls = (
            _require_control_mask(
                control_relation_mask,
                relation_count=len(
                    resolved_names
                ),
            )
        )

        module_keys = tuple(
            _relation_module_key(
                relation_index=index,
                stable_relation_id=(
                    relation_id
                ),
            )
            for index, relation_id in enumerate(
                resolved_ids
            )
        )

        if len(set(module_keys)) != len(
            module_keys
        ):
            raise RuntimeError(
                "Internal relation module keys must be unique."
            )

        self.hidden_dim = hidden_dim
        self.bias = bias
        self.relation_names = (
            resolved_names
        )
        self.stable_relation_ids = (
            resolved_ids
        )
        self.control_relation_mask = (
            resolved_controls
        )
        self.relation_module_keys = (
            module_keys
        )

        self.relation_transforms = (
            nn.ModuleDict(
                OrderedRelationModules(
                    (
                        module_key,
                        nn.Linear(
                            hidden_dim,
                            hidden_dim,
                            bias=bias,
                        ),
                    )
                    for module_key
                    in module_keys
                )
            )
        )

        self._relation_index_by_name = (
            MappingProxyType(
                {
                    name: index
                    for index, name in enumerate(
                        resolved_names
                    )
                }
            )
        )
        self._relation_index_by_stable_id = (
            MappingProxyType(
                {
                    relation_id: index
                    for index, relation_id
                    in enumerate(
                        resolved_ids
                    )
                }
            )
        )

    # ------------------------------------------------------------------
    # Construction from the compiled source of truth
    # ------------------------------------------------------------------

    @classmethod
    def from_compiled_registry(
        cls,
        *,
        hidden_dim: int,
        compiled_relation_registry: (
            CompiledRelationRegistry
        ),
        bias: bool = True,
    ) -> "PerRelationTransform":
        if not isinstance(
            compiled_relation_registry,
            CompiledRelationRegistry,
        ):
            raise TypeError(
                "compiled_relation_registry must be a "
                "CompiledRelationRegistry."
            )

        compiled_relation_registry.validate()

        return cls(
            hidden_dim=hidden_dim,
            relation_names=(
                compiled_relation_registry
                .relation_names
            ),
            stable_relation_ids=(
                compiled_relation_registry
                .stable_relation_ids
            ),
            control_relation_mask=tuple(
                bool(
                    entry
                    .specification
                    .is_control
                )
                for entry
                in compiled_relation_registry.entries
            ),
            bias=bias,
        )

    # ------------------------------------------------------------------
    # Public identity
    # ------------------------------------------------------------------

    @property
    def transform_mode(self) -> str:
        return (
            RELATION_TRANSFORM_PER_RELATION
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
        return (
            next(
                self.parameters()
            ).device
        )

    @property
    def dtype(self) -> torch.dtype:
        return (
            next(
                self.parameters()
            ).dtype
        )

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
    def relation_index_by_name(
        self,
    ) -> Mapping[str, int]:
        return self._relation_index_by_name

    @property
    def relation_index_by_stable_id(
        self,
    ) -> Mapping[int, int]:
        return (
            self._relation_index_by_stable_id
        )

    def relation_index(
        self,
        relation: str | int,
    ) -> int:
        """
        Resolve a relation name or stable ontology ID to its dense index.

        A plain integer is interpreted as a stable ontology ID, never as an
        already dense relation index. Dense indices are used only in tensor
        inputs.
        """

        if isinstance(relation, bool):
            raise TypeError(
                "relation must be a canonical name or stable integer ID."
            )

        if isinstance(relation, str):
            try:
                return self._relation_index_by_name[
                    relation
                ]
            except KeyError as error:
                raise KeyError(
                    f"Unknown relation name {relation!r}."
                ) from error

        if isinstance(relation, int):
            try:
                return (
                    self
                    ._relation_index_by_stable_id[
                        relation
                    ]
                )
            except KeyError as error:
                raise KeyError(
                    f"Unknown stable relation ID {relation}."
                ) from error

        raise TypeError(
            "relation must be a canonical name or stable integer ID."
        )

    def module_for_relation_index(
        self,
        relation_index: int,
    ) -> nn.Linear:
        if (
            isinstance(relation_index, bool)
            or not isinstance(
                relation_index,
                int,
            )
        ):
            raise TypeError(
                "relation_index must be an integer."
            )

        if not (
            0
            <= relation_index
            < self.relation_count
        ):
            raise IndexError(
                "relation_index is outside the compiled relation range "
                f"[0, {self.relation_count - 1}]."
            )

        module_key = (
            self.relation_module_keys[
                relation_index
            ]
        )
        module = self.relation_transforms[
            module_key
        ]

        if not isinstance(module, nn.Linear):
            raise RuntimeError(
                "Internal relation transform is not an nn.Linear."
            )

        return module

    def module_for_relation(
        self,
        relation: str | int,
    ) -> nn.Linear:
        return self.module_for_relation_index(
            self.relation_index(relation)
        )

    # ------------------------------------------------------------------
    # Architecture and parameter identity
    # ------------------------------------------------------------------

    def relation_architecture_dict(
        self,
        relation_index: int,
    ) -> dict[str, Any]:
        resolved_index = self._validate_relation_index_scalar(
            relation_index
        )

        return {
            "schema_version": (
                PER_RELATION_TRANSFORM_SCHEMA_VERSION
            ),
            "transform_mode": (
                RELATION_TRANSFORM_PER_RELATION
            ),
            "relation_index": (
                resolved_index
            ),
            "relation_name": (
                self.relation_names[
                    resolved_index
                ]
            ),
            "stable_relation_id": (
                self.stable_relation_ids[
                    resolved_index
                ]
            ),
            "is_control": (
                self.control_relation_mask[
                    resolved_index
                ]
            ),
            "module_key": (
                self.relation_module_keys[
                    resolved_index
                ]
            ),
            "hidden_dim": self.hidden_dim,
            "bias": self.bias,
            "operation": "relation_specific_linear_transform",
            "activation": None,
            "normalization": None,
            "dropout": 0.0,
        }

    def relation_architecture_fingerprint(
        self,
        relation_index: int,
    ) -> str:
        return _fingerprint(
            self.relation_architecture_dict(
                relation_index
            )
        )

    def relation_architecture_fingerprints(
        self,
    ) -> Mapping[str, str]:
        return MappingProxyType(
            {
                relation_name: (
                    self
                    .relation_architecture_fingerprint(
                        relation_index
                    )
                )
                for relation_index, relation_name
                in enumerate(
                    self.relation_names
                )
            }
        )

    def relation_parameter_fingerprint(
        self,
        relation_index: int,
    ) -> str:
        module = (
            self.module_for_relation_index(
                relation_index
            )
        )
        return _state_dict_fingerprint(
            module.state_dict()
        )

    def relation_parameter_fingerprints(
        self,
    ) -> Mapping[str, str]:
        return MappingProxyType(
            {
                relation_name: (
                    self
                    .relation_parameter_fingerprint(
                        relation_index
                    )
                )
                for relation_index, relation_name
                in enumerate(
                    self.relation_names
                )
            }
        )

    def architecture_dict(
        self,
    ) -> dict[str, Any]:
        return {
            "schema_version": (
                PER_RELATION_TRANSFORM_SCHEMA_VERSION
            ),
            "transform_mode": (
                RELATION_TRANSFORM_PER_RELATION
            ),
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
            "relation_module_keys": list(
                self.relation_module_keys
            ),
            "parameter_sharing": (
                "independent_linear_map_per_relation"
            ),
            "operation_order": [
                "gather_source_node_state",
                "select_relation_specific_parameters",
                "relation_specific_linear_transform",
            ],
            "activation": None,
            "normalization": None,
            "dropout": 0.0,
            "relation_architecture_fingerprints": dict(
                self
                .relation_architecture_fingerprints()
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
        return _state_dict_fingerprint(
            self.state_dict()
        )

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def _validate_relation_index_scalar(
        self,
        relation_index: int,
    ) -> int:
        if (
            isinstance(relation_index, bool)
            or not isinstance(
                relation_index,
                int,
            )
        ):
            raise TypeError(
                "relation_index must be an integer."
            )

        if not (
            0
            <= relation_index
            < self.relation_count
        ):
            raise IndexError(
                "relation_index is outside the compiled relation range "
                f"[0, {self.relation_count - 1}]."
            )

        return relation_index

    def assert_finite_parameters(
        self,
    ) -> None:
        for name, parameter in (
            self.named_parameters()
        ):
            if not bool(
                torch.isfinite(parameter)
                .all()
                .item()
            ):
                raise ValueError(
                    "Per-relation transform parameter "
                    f"{name!r} contains NaN or infinity."
                )

    def _validate_source_state(
        self,
        source_state: torch.Tensor,
    ) -> None:
        _require_node_state(
            source_state,
            hidden_dim=self.hidden_dim,
            name="source_state",
        )

        if source_state.device != self.device:
            raise ValueError(
                "source_state and PerRelationTransform parameters must "
                "share one device. "
                f"Observed {source_state.device} and {self.device}."
            )

        if source_state.dtype != self.dtype:
            raise ValueError(
                "source_state dtype must match PerRelationTransform "
                f"parameter dtype. Observed {source_state.dtype} and "
                f"{self.dtype}."
            )

    def _validate_edge_relation_index(
        self,
        edge_relation_index: torch.Tensor,
        *,
        edge_count: int,
        expected_device: torch.device,
    ) -> None:
        _require_index_vector(
            "edge_relation_index",
            edge_relation_index,
            item_count=edge_count,
        )
        _require_index_range(
            "edge_relation_index",
            edge_relation_index,
            upper_bound=self.relation_count,
        )

        if (
            edge_relation_index.device
            != expected_device
        ):
            raise ValueError(
                "source states and edge_relation_index must share one "
                "device. "
                f"Observed {expected_device} and "
                f"{edge_relation_index.device}."
            )

    def _validate_forward_inputs(
        self,
        node_state: torch.Tensor,
        source_index: torch.Tensor,
        edge_relation_index: torch.Tensor,
    ) -> None:
        _require_node_state(
            node_state,
            hidden_dim=self.hidden_dim,
        )
        _require_index_vector(
            "source_index",
            source_index,
        )
        _require_index_range(
            "source_index",
            source_index,
            upper_bound=int(
                node_state.shape[0]
            ),
        )
        self._validate_edge_relation_index(
            edge_relation_index,
            edge_count=int(
                source_index.shape[0]
            ),
            expected_device=(
                node_state.device
            ),
        )

        if (
            node_state.device
            != source_index.device
        ):
            raise ValueError(
                "node_state and source_index must share one device. "
                f"Observed {node_state.device} and "
                f"{source_index.device}."
            )

        if node_state.device != self.device:
            raise ValueError(
                "node_state and PerRelationTransform parameters must "
                "share one device. "
                f"Observed {node_state.device} and {self.device}."
            )

        if node_state.dtype != self.dtype:
            raise ValueError(
                "node_state dtype must match PerRelationTransform "
                f"parameter dtype. Observed {node_state.dtype} and "
                f"{self.dtype}."
            )

    # ------------------------------------------------------------------
    # Mathematical operations
    # ------------------------------------------------------------------

    def gather_source_state(
        self,
        node_state: torch.Tensor,
        source_index: torch.Tensor,
        edge_relation_index: torch.Tensor,
    ) -> torch.Tensor:
        """
        Return edge-aligned source states after complete alignment validation.

        ``edge_relation_index`` is validated here even though it does not
        affect gathering, ensuring that the returned state is ready for
        relation-specific transformation without a second alignment contract.
        """

        self._validate_forward_inputs(
            node_state,
            source_index,
            edge_relation_index,
        )

        gathered = node_state.index_select(
            0,
            source_index,
        )

        expected_shape = (
            int(source_index.shape[0]),
            self.hidden_dim,
        )

        if tuple(gathered.shape) != (
            expected_shape
        ):
            raise RuntimeError(
                "Internal source-state gathering produced an invalid "
                f"shape {tuple(gathered.shape)}; expected "
                f"{expected_shape}."
            )

        return gathered

    def stacked_weight(
        self,
    ) -> torch.Tensor:
        """
        Return relation-aligned weights with shape ``[R, H, H]``.

        The returned tensor remains connected to every relation parameter.
        Relations with zero edges therefore receive explicit zero gradients
        rather than disappearing from the parameter graph.
        """

        return torch.stack(
            [
                self
                .module_for_relation_index(
                    relation_index
                )
                .weight
                for relation_index
                in range(
                    self.relation_count
                )
            ],
            dim=0,
        )

    def stacked_bias(
        self,
    ) -> torch.Tensor | None:
        """
        Return relation-aligned biases with shape ``[R, H]``.
        """

        if not self.bias:
            return None

        biases = [
            self
            .module_for_relation_index(
                relation_index
            )
            .bias
            for relation_index
            in range(
                self.relation_count
            )
        ]

        if any(
            bias is None
            for bias in biases
        ):
            raise RuntimeError(
                "Bias-enabled relation transform contains a missing "
                "relation bias."
            )

        return torch.stack(
            [
                bias
                for bias in biases
                if bias is not None
            ],
            dim=0,
        )

    def transform_source_state(
        self,
        source_state: torch.Tensor,
        edge_relation_index: torch.Tensor,
    ) -> torch.Tensor:
        """
        Apply relation-specific maps to an edge-aligned source state.

        Parameters
        ----------
        source_state:
            Tensor ``[E, H]``.

        edge_relation_index:
            Dense compiled relation index ``[E]``.
        """

        self._validate_source_state(
            source_state
        )
        self._validate_edge_relation_index(
            edge_relation_index,
            edge_count=int(
                source_state.shape[0]
            ),
            expected_device=(
                source_state.device
            ),
        )

        edge_count = int(
            source_state.shape[0]
        )

        if edge_count == 0:
            # Preserve parameter connectivity for consistent zero gradients.
            zero = (
                self.stacked_weight()
                .sum()
                * source_state.new_zeros(())
            )
            if self.bias:
                bias = self.stacked_bias()
                if bias is None:
                    raise RuntimeError(
                        "Bias-enabled transform is missing stacked bias."
                    )
                zero = (
                    zero
                    + bias.sum()
                    * source_state.new_zeros(())
                )

            return (
                source_state.new_empty(
                    (0, self.hidden_dim)
                )
                + zero
            )

        weights = self.stacked_weight()
        selected_weights = weights.index_select(
            0,
            edge_relation_index,
        )

        transformed = torch.bmm(
            selected_weights,
            source_state.unsqueeze(-1),
        ).squeeze(-1)

        biases = self.stacked_bias()

        if biases is not None:
            transformed = (
                transformed
                + biases.index_select(
                    0,
                    edge_relation_index,
                )
            )

        expected_shape = (
            edge_count,
            self.hidden_dim,
        )

        if tuple(transformed.shape) != (
            expected_shape
        ):
            raise RuntimeError(
                "Per-relation transformation produced an invalid shape "
                f"{tuple(transformed.shape)}; expected {expected_shape}."
            )

        if not bool(
            torch.isfinite(transformed)
            .all()
            .item()
        ):
            raise FloatingPointError(
                "Per-relation transformation produced NaN or infinity "
                "from finite inputs."
            )

        return transformed

    def forward(
        self,
        node_state: torch.Tensor,
        source_index: torch.Tensor,
        edge_relation_index: torch.Tensor,
    ) -> torch.Tensor:
        """
        Gather source-node states and apply independently parameterized maps.

        Returns
        -------
        torch.Tensor
            Edge-aligned transformed source states with shape ``[E, H]``.
        """

        source_state = self.gather_source_state(
            node_state,
            source_index,
            edge_relation_index,
        )

        return self.transform_source_state(
            source_state,
            edge_relation_index,
        )

    # ------------------------------------------------------------------
    # State and representation
    # ------------------------------------------------------------------

    def extra_repr(self) -> str:
        control_count = sum(
            self.control_relation_mask
        )

        return (
            f"hidden_dim={self.hidden_dim}, "
            f"relation_count={self.relation_count}, "
            f"control_relation_count={control_count}, "
            f"bias={self.bias}, "
            f"transform_mode={self.transform_mode!r}"
        )


class OrderedRelationModules(dict[str, nn.Linear]):
    """
    Tiny explicit ordered mapping used only to construct ``nn.ModuleDict``.

    Python mappings preserve insertion order. The named class makes the
    state-dict ordering intent visible without adding runtime behavior.
    """


__all__ = (
    "PER_RELATION_TRANSFORM_SCHEMA_VERSION",
    "PerRelationTransform",
)
