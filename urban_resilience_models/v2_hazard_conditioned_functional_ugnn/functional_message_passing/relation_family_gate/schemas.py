"""
Typed contracts for hazard-conditioned relation gating.

Target path:
    urban_resilience_models/
        v2_hazard_conditioned_functional_ugnn/
            functional_message_passing/
                relation_family_gate/
                    schemas.py

Despite the historical package name ``relation_family_gate``, the bounded
V2.0 trainable gate axis is the exact compiled relation axis ``R``. Semantic
relation-family metadata is retained for diagnostics and future hierarchical
models, but is never substituted for relation identity during gate lookup.

This module owns immutable, metadata-preserving contracts for:

- exact relation-axis ordering;
- stable relation identities and control masks;
- optional semantic family alignment;
- neural target-node relation logits;
- node-aligned prior logit contributions and prior traces;
- sigmoid activation outputs before edge lookup.

It does not own:

- neural gate prediction;
- activation mathematics;
- compilation of hazard-relation priors;
- prior-to-logit conversion;
- edge-aligned gate lookup;
- final ``RelationGateOutput`` orchestration;
- edge attention, message construction, or aggregation.

Bounded V2.0 mathematical contract
----------------------------------
For ``N`` target nodes and ``R`` compiled relation identities:

    neural_logits:       [N, R]
    prior_contribution:  [N, R]  (optional)
    combined_logits:     [N, R]
    gate_values:         [N, R]

When priors are absent:

    combined_logits = neural_logits

When priors are present:

    combined_logits = neural_logits + prior_contribution

The implemented activation is sigmoid:

    gate_values = sigmoid(combined_logits)

The later orchestration module performs exact edge lookup:

    edge_gate_values[e] =
        gate_values[target_index[e], edge_relation_index[e]]

Several relations may be active simultaneously. The relation channels do not
compete through softmax.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from hashlib import sha256
import json
import math
from types import MappingProxyType
from typing import Any, Final, Mapping, Sequence

import torch

from ...constants import (
    RELATION_GATE_ACTIVATION_SIGMOID,
    RELATION_GATE_SCOPE_TARGET_NODE,
)
from ..schemas import (
    FunctionalMessagePassingInputs,
    RelationGateOutput,
)


# =============================================================================
# Public schema identity
# =============================================================================


RELATION_GATE_AXIS_SCHEMA_VERSION: Final[str] = "0.1"
GATE_NETWORK_OUTPUT_SCHEMA_VERSION: Final[str] = "0.1"
RELATION_PRIOR_CONTRIBUTION_SCHEMA_VERSION: Final[str] = "0.1"
GATE_ACTIVATION_OUTPUT_SCHEMA_VERSION: Final[str] = "0.1"


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
        _require_nonempty_string(
            name,
            value,
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


def _require_unique_nonnegative_ints(
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


def _require_nonnegative_finite_float(
    name: str,
    value: float,
) -> None:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
    ):
        raise TypeError(
            f"{name} must be numeric."
        )

    converted = float(value)

    if not math.isfinite(converted):
        raise ValueError(
            f"{name} must be finite."
        )

    if converted < 0.0:
        raise ValueError(
            f"{name} must be nonnegative."
        )


def _require_tensor(
    name: str,
    value: torch.Tensor,
    *,
    ndim: int,
) -> None:
    if not isinstance(
        value,
        torch.Tensor,
    ):
        raise TypeError(
            f"{name} must be a tensor."
        )

    if value.ndim != ndim:
        raise ValueError(
            f"{name} must have rank {ndim}; "
            f"observed shape {tuple(value.shape)}."
        )


def _require_float_tensor(
    name: str,
    value: torch.Tensor,
    *,
    ndim: int,
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

    if not bool(
        torch.isfinite(value)
        .all()
        .item()
    ):
        raise ValueError(
            f"{name} must contain only finite values."
        )


def _require_long_tensor(
    name: str,
    value: torch.Tensor,
    *,
    ndim: int,
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
    ndim: int,
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


def _devices_match(
    first: torch.device | str,
    second: torch.device | str,
) -> bool:
    first_device = torch.device(first)
    second_device = torch.device(second)

    if first_device.type != second_device.type:
        return False

    if first_device.type != "cuda":
        return first_device == second_device

    first_index = (
        torch.cuda.current_device()
        if first_device.index is None
        else first_device.index
    )
    second_index = (
        torch.cuda.current_device()
        if second_device.index is None
        else second_device.index
    )

    return first_index == second_index


def _require_device(
    name: str,
    value: torch.Tensor,
    expected: torch.device | str,
) -> None:
    if not _devices_match(
        value.device,
        expected,
    ):
        raise ValueError(
            f"{name} must be on device {torch.device(expected)}; "
            f"observed {value.device}."
        )


def _require_dtype(
    name: str,
    value: torch.Tensor,
    expected: torch.dtype,
) -> None:
    if value.dtype != expected:
        raise ValueError(
            f"{name} must use dtype {expected}; "
            f"observed {value.dtype}."
        )


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
        tensor = tensors[name]
        if not isinstance(
            tensor,
            torch.Tensor,
        ):
            raise TypeError(
                f"{name} must be a tensor."
            )

        detached = (
            tensor
            .detach()
            .contiguous()
            .cpu()
        )
        raw = (
            detached
            .view(torch.uint8)
            .numpy()
            .tobytes()
        )

        digest.update(
            name.encode("utf-8")
        )
        digest.update(
            str(tuple(detached.shape)).encode(
                "utf-8"
            )
        )
        digest.update(
            str(detached.dtype).encode(
                "utf-8"
            )
        )
        digest.update(raw)

    return digest.hexdigest()


def _immutable_nonnegative_int_mapping(
    name: str,
    values: Mapping[str, int],
) -> Mapping[str, int]:
    if not isinstance(values, Mapping):
        raise TypeError(
            f"{name} must be a mapping."
        )

    copied: dict[str, int] = {}

    for key, value in values.items():
        _require_nonempty_string(
            f"{name} key",
            key,
        )

        if (
            isinstance(value, bool)
            or not isinstance(value, int)
        ):
            raise TypeError(
                f"{name}[{key!r}] must be an integer."
            )

        if value < 0:
            raise ValueError(
                f"{name}[{key!r}] must be nonnegative."
            )

        copied[key] = value

    return MappingProxyType(copied)


def _validate_node_relation_tensor(
    name: str,
    value: torch.Tensor,
    *,
    source_inputs: FunctionalMessagePassingInputs,
) -> None:
    _require_float_tensor(
        name,
        value,
        ndim=2,
    )
    _require_shape(
        name,
        value,
        (
            source_inputs.num_nodes,
            source_inputs.num_relations,
        ),
    )
    _require_device(
        name,
        value,
        source_inputs.device,
    )
    _require_dtype(
        name,
        value,
        source_inputs.dtype,
    )


def _validate_optional_node_relation_float_tensor(
    name: str,
    value: torch.Tensor | None,
    *,
    source_inputs: FunctionalMessagePassingInputs,
    probability: bool = False,
) -> None:
    if value is None:
        return

    _validate_node_relation_tensor(
        name,
        value,
        source_inputs=source_inputs,
    )

    if probability and bool(
        (
            (value < 0)
            | (value > 1)
        )
        .any()
        .item()
    ):
        raise ValueError(
            f"{name} must lie in [0, 1]."
        )


def _validate_optional_node_relation_bool_tensor(
    name: str,
    value: torch.Tensor | None,
    *,
    source_inputs: FunctionalMessagePassingInputs,
) -> None:
    if value is None:
        return

    _require_bool_tensor(
        name,
        value,
        ndim=2,
    )
    _require_shape(
        name,
        value,
        (
            source_inputs.num_nodes,
            source_inputs.num_relations,
        ),
    )
    _require_device(
        name,
        value,
        source_inputs.device,
    )


# =============================================================================
# Exact compiled relation-axis metadata
# =============================================================================


@dataclass(slots=True, frozen=True)
class RelationGateAxis:
    """
    Immutable exact-relation axis used by the V2.0 gate.

    ``relation_names[r]`` and ``stable_relation_ids[r]`` describe dense gate
    column ``r``. ``control_relation_mask[r]`` marks registry-declared control
    relations without removing or reordering them.

    Optional family metadata maps each relation column to an ontology root
    family. It is diagnostic only and never changes the trainable gate axis.
    """

    relation_names: tuple[str, ...]
    stable_relation_ids: tuple[int, ...]
    control_relation_mask: torch.Tensor

    compiled_relation_registry_fingerprint: str

    family_names: tuple[str, ...] = ()
    stable_family_ids: tuple[int, ...] = ()
    relation_family_index_by_relation: (
        torch.Tensor | None
    ) = None

    source_relation_registry_fingerprint: (
        str | None
    ) = None

    schema_version: str = (
        RELATION_GATE_AXIS_SCHEMA_VERSION
    )

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "relation_names",
            tuple(self.relation_names),
        )
        object.__setattr__(
            self,
            "stable_relation_ids",
            tuple(self.stable_relation_ids),
        )
        object.__setattr__(
            self,
            "family_names",
            tuple(self.family_names),
        )
        object.__setattr__(
            self,
            "stable_family_ids",
            tuple(self.stable_family_ids),
        )

        _require_unique_strings(
            "relation_names",
            self.relation_names,
        )
        _require_unique_nonnegative_ints(
            "stable_relation_ids",
            self.stable_relation_ids,
        )

        if not self.relation_names:
            raise ValueError(
                "At least one compiled relation is required."
            )

        if len(self.relation_names) != len(
            self.stable_relation_ids
        ):
            raise ValueError(
                "relation_names and stable_relation_ids must align."
            )

        _require_bool_tensor(
            "control_relation_mask",
            self.control_relation_mask,
            ndim=1,
        )
        _require_shape(
            "control_relation_mask",
            self.control_relation_mask,
            (self.num_relations,),
        )

        _require_nonempty_string(
            "compiled_relation_registry_fingerprint",
            self.compiled_relation_registry_fingerprint,
        )
        _require_optional_nonempty_string(
            "source_relation_registry_fingerprint",
            self.source_relation_registry_fingerprint,
        )
        _require_nonempty_string(
            "schema_version",
            self.schema_version,
        )

        has_any_family_metadata = bool(
            self.family_names
            or self.stable_family_ids
            or self.relation_family_index_by_relation
            is not None
            or self.source_relation_registry_fingerprint
            is not None
        )

        if not has_any_family_metadata:
            return

        _require_unique_strings(
            "family_names",
            self.family_names,
        )
        _require_unique_nonnegative_ints(
            "stable_family_ids",
            self.stable_family_ids,
        )

        if not self.family_names:
            raise ValueError(
                "Family metadata requires at least one family."
            )

        if len(self.family_names) != len(
            self.stable_family_ids
        ):
            raise ValueError(
                "family_names and stable_family_ids must align."
            )

        if (
            self.relation_family_index_by_relation
            is None
        ):
            raise ValueError(
                "Family metadata requires "
                "relation_family_index_by_relation."
            )

        _require_long_tensor(
            "relation_family_index_by_relation",
            self.relation_family_index_by_relation,
            ndim=1,
        )
        _require_shape(
            "relation_family_index_by_relation",
            self.relation_family_index_by_relation,
            (self.num_relations,),
        )
        _require_device(
            "relation_family_index_by_relation",
            self.relation_family_index_by_relation,
            self.device,
        )

        minimum = int(
            self
            .relation_family_index_by_relation
            .min()
            .item()
        )
        maximum = int(
            self
            .relation_family_index_by_relation
            .max()
            .item()
        )

        if (
            minimum < 0
            or maximum >= self.num_families
        ):
            raise ValueError(
                "relation_family_index_by_relation contains "
                "out-of-range family indices."
            )

        represented = set(
            int(value)
            for value in (
                self
                .relation_family_index_by_relation
                .detach()
                .cpu()
                .tolist()
            )
        )

        if represented != set(
            range(self.num_families)
        ):
            raise ValueError(
                "Every declared relation family must be represented by "
                "at least one compiled relation."
            )

        if (
            self.source_relation_registry_fingerprint
            is None
        ):
            raise ValueError(
                "Family metadata requires "
                "source_relation_registry_fingerprint."
            )

    @classmethod
    def from_inputs(
        cls,
        *,
        source_inputs: FunctionalMessagePassingInputs,
    ) -> "RelationGateAxis":
        if not isinstance(
            source_inputs,
            FunctionalMessagePassingInputs,
        ):
            raise TypeError(
                "source_inputs must be a "
                "FunctionalMessagePassingInputs."
            )

        families = source_inputs.relation_families

        return cls(
            relation_names=(
                source_inputs.relation_names
            ),
            stable_relation_ids=(
                source_inputs.stable_relation_ids
            ),
            control_relation_mask=(
                source_inputs.control_relation_mask
            ),
            compiled_relation_registry_fingerprint=(
                source_inputs
                .compiled_relation_registry
                .fingerprint()
            ),
            family_names=(
                families.family_names
                if families is not None
                else ()
            ),
            stable_family_ids=(
                families.stable_family_ids
                if families is not None
                else ()
            ),
            relation_family_index_by_relation=(
                families
                .relation_family_index_by_relation
                if families is not None
                else None
            ),
            source_relation_registry_fingerprint=(
                families
                .source_relation_registry_fingerprint
                if families is not None
                else None
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
        return self.control_relation_mask.device

    @property
    def has_family_metadata(self) -> bool:
        return (
            self.relation_family_index_by_relation
            is not None
        )

    @property
    def control_relation_names(
        self,
    ) -> tuple[str, ...]:
        mask = (
            self.control_relation_mask
            .detach()
            .cpu()
            .tolist()
        )
        return tuple(
            name
            for name, is_control
            in zip(
                self.relation_names,
                mask,
            )
            if bool(is_control)
        )

    def assert_matches_inputs(
        self,
        source_inputs: FunctionalMessagePassingInputs,
    ) -> None:
        if not isinstance(
            source_inputs,
            FunctionalMessagePassingInputs,
        ):
            raise TypeError(
                "source_inputs must be a "
                "FunctionalMessagePassingInputs."
            )

        if self.relation_names != (
            source_inputs.relation_names
        ):
            raise ValueError(
                "Relation-gate axis ordering differs from source inputs."
            )

        if self.stable_relation_ids != (
            source_inputs.stable_relation_ids
        ):
            raise ValueError(
                "Relation-gate stable relation IDs differ from source "
                "inputs."
            )

        if (
            self
            .compiled_relation_registry_fingerprint
            != source_inputs
            .compiled_relation_registry
            .fingerprint()
        ):
            raise ValueError(
                "Relation-gate axis references a different compiled "
                "relation registry."
            )

        if not _devices_match(
            self.control_relation_mask.device,
            source_inputs.device,
        ):
            raise ValueError(
                "Relation-gate axis and source inputs must share one "
                "device."
            )

        if not torch.equal(
            self.control_relation_mask,
            source_inputs.control_relation_mask,
        ):
            raise ValueError(
                "Relation-gate control mask differs from source inputs."
            )

        families = source_inputs.relation_families

        if families is None:
            if self.has_family_metadata:
                raise ValueError(
                    "Relation-gate axis contains family metadata but "
                    "source inputs do not."
                )
            return

        if not self.has_family_metadata:
            raise ValueError(
                "Source inputs contain relation-family metadata but the "
                "relation-gate axis does not."
            )

        if self.family_names != (
            families.family_names
        ):
            raise ValueError(
                "Relation-gate family ordering differs from source "
                "inputs."
            )

        if self.stable_family_ids != (
            families.stable_family_ids
        ):
            raise ValueError(
                "Relation-gate stable family IDs differ from source "
                "inputs."
            )

        if not _devices_match(
            self
            .relation_family_index_by_relation
            .device,
            families
            .relation_family_index_by_relation
            .device,
        ):
            raise ValueError(
                "Relation-to-family alignment and source inputs must "
                "share one device."
            )

        if not torch.equal(
            self.relation_family_index_by_relation,
            families.relation_family_index_by_relation,
        ):
            raise ValueError(
                "Relation-to-family alignment differs from source "
                "inputs."
            )

        if (
            self.source_relation_registry_fingerprint
            != families
            .source_relation_registry_fingerprint
        ):
            raise ValueError(
                "Relation-gate family metadata references a different "
                "source relation registry."
            )

    def semantic_dict(
        self,
    ) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "relation_names": list(
                self.relation_names
            ),
            "stable_relation_ids": list(
                self.stable_relation_ids
            ),
            "compiled_relation_registry_fingerprint": (
                self
                .compiled_relation_registry_fingerprint
            ),
            "family_names": list(
                self.family_names
            ),
            "stable_family_ids": list(
                self.stable_family_ids
            ),
            "source_relation_registry_fingerprint": (
                self
                .source_relation_registry_fingerprint
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
        tensors = {
            "control_relation_mask": (
                self.control_relation_mask
            ),
        }

        if (
            self.relation_family_index_by_relation
            is not None
        ):
            tensors[
                "relation_family_index_by_relation"
            ] = (
                self
                .relation_family_index_by_relation
            )

        return _tensor_fingerprint(tensors)

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
# Neural gate-network output
# =============================================================================


@dataclass(slots=True, frozen=True)
class GateNetworkOutput:
    """
    Neural target-node logits over the exact compiled relation axis.

    No activation or prior contribution has been applied.
    """

    logits: torch.Tensor
    source_inputs: FunctionalMessagePassingInputs
    axis: RelationGateAxis

    scope: str

    encoder_architecture_fingerprint: str
    parameter_fingerprint: str | None = None

    input_feature_names: tuple[str, ...] = ()

    schema_version: str = (
        GATE_NETWORK_OUTPUT_SCHEMA_VERSION
    )

    def __post_init__(self) -> None:
        if not isinstance(
            self.source_inputs,
            FunctionalMessagePassingInputs,
        ):
            raise TypeError(
                "source_inputs must be a "
                "FunctionalMessagePassingInputs."
            )

        if not isinstance(
            self.axis,
            RelationGateAxis,
        ):
            raise TypeError(
                "axis must be a RelationGateAxis."
            )

        self.axis.assert_matches_inputs(
            self.source_inputs
        )

        _validate_node_relation_tensor(
            "logits",
            self.logits,
            source_inputs=self.source_inputs,
        )

        _require_nonempty_string(
            "scope",
            self.scope,
        )

        if self.scope != (
            RELATION_GATE_SCOPE_TARGET_NODE
        ):
            raise ValueError(
                "The bounded V2.0 gate network supports only "
                f"{RELATION_GATE_SCOPE_TARGET_NODE!r} scope."
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
            "input_feature_names",
            tuple(self.input_feature_names),
        )
        _require_unique_strings(
            "input_feature_names",
            self.input_feature_names,
        )

    @property
    def num_nodes(self) -> int:
        return self.source_inputs.num_nodes

    @property
    def num_relations(self) -> int:
        return self.axis.num_relations

    @property
    def device(self) -> torch.device:
        return self.logits.device

    @property
    def dtype(self) -> torch.dtype:
        return self.logits.dtype

    @property
    def control_relation_mask(
        self,
    ) -> torch.Tensor:
        return self.axis.control_relation_mask

    def lineage_dict(
        self,
    ) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "scope": self.scope,
            "axis_fingerprint": (
                self.axis.fingerprint()
            ),
            "source_input_lineage_fingerprint": (
                self
                .source_inputs
                .lineage_fingerprint()
            ),
            "encoder_architecture_fingerprint": (
                self
                .encoder_architecture_fingerprint
            ),
            "parameter_fingerprint": (
                self.parameter_fingerprint
            ),
            "input_feature_names": list(
                self.input_feature_names
            ),
        }

    def lineage_fingerprint(
        self,
    ) -> str:
        return _fingerprint(
            self.lineage_dict()
        )

    def value_fingerprint(
        self,
    ) -> str:
        return _tensor_fingerprint(
            {"logits": self.logits}
        )

    def fingerprint(
        self,
    ) -> str:
        return _fingerprint(
            {
                "lineage_fingerprint": (
                    self.lineage_fingerprint()
                ),
                "value_fingerprint": (
                    self.value_fingerprint()
                ),
            }
        )


# =============================================================================
# Node-aligned hazard-relation prior contribution
# =============================================================================


@dataclass(slots=True, frozen=True)
class RelationPriorContribution:
    """
    Node-aligned prior contribution added to neural gate logits.

    The contribution is already expressed in logit space and aligned to the
    exact relation axis. Optional trace tensors preserve the probability,
    confidence, initialization, and regularization information used to derive
    it.
    """

    logit_contribution: torch.Tensor
    source_inputs: FunctionalMessagePassingInputs
    axis: RelationGateAxis

    strength: float
    source_compiled_prior_fingerprint: str

    prior_mean: torch.Tensor | None = None
    confidence: torch.Tensor | None = None
    initialization_mask: torch.Tensor | None = None
    regularization_mask: torch.Tensor | None = None

    resolution_summary: Mapping[
        str,
        int,
    ] = field(default_factory=dict)

    schema_version: str = (
        RELATION_PRIOR_CONTRIBUTION_SCHEMA_VERSION
    )

    def __post_init__(self) -> None:
        if not isinstance(
            self.source_inputs,
            FunctionalMessagePassingInputs,
        ):
            raise TypeError(
                "source_inputs must be a "
                "FunctionalMessagePassingInputs."
            )

        if not isinstance(
            self.axis,
            RelationGateAxis,
        ):
            raise TypeError(
                "axis must be a RelationGateAxis."
            )

        self.axis.assert_matches_inputs(
            self.source_inputs
        )

        _validate_node_relation_tensor(
            "logit_contribution",
            self.logit_contribution,
            source_inputs=self.source_inputs,
        )
        _require_nonnegative_finite_float(
            "strength",
            self.strength,
        )
        _require_nonempty_string(
            "source_compiled_prior_fingerprint",
            self.source_compiled_prior_fingerprint,
        )
        _require_nonempty_string(
            "schema_version",
            self.schema_version,
        )

        compiled_priors = (
            self.source_inputs
            .compiled_relation_priors
        )

        if compiled_priors is None:
            raise ValueError(
                "RelationPriorContribution requires "
                "source_inputs.compiled_relation_priors."
            )

        if (
            self.source_compiled_prior_fingerprint
            != compiled_priors.fingerprint()
        ):
            raise ValueError(
                "Relation prior contribution references a different "
                "compiled prior artifact."
            )

        _validate_optional_node_relation_float_tensor(
            "prior_mean",
            self.prior_mean,
            source_inputs=self.source_inputs,
            probability=True,
        )
        _validate_optional_node_relation_float_tensor(
            "confidence",
            self.confidence,
            source_inputs=self.source_inputs,
            probability=True,
        )
        _validate_optional_node_relation_bool_tensor(
            "initialization_mask",
            self.initialization_mask,
            source_inputs=self.source_inputs,
        )
        _validate_optional_node_relation_bool_tensor(
            "regularization_mask",
            self.regularization_mask,
            source_inputs=self.source_inputs,
        )

        if float(self.strength) == 0.0:
            if not torch.equal(
                self.logit_contribution,
                torch.zeros_like(
                    self.logit_contribution
                ),
            ):
                raise ValueError(
                    "Zero prior strength requires exact zero logit "
                    "contribution."
                )

        object.__setattr__(
            self,
            "resolution_summary",
            _immutable_nonnegative_int_mapping(
                "resolution_summary",
                self.resolution_summary,
            ),
        )

    @property
    def device(self) -> torch.device:
        return self.logit_contribution.device

    @property
    def dtype(self) -> torch.dtype:
        return self.logit_contribution.dtype

    def lineage_dict(
        self,
    ) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "axis_fingerprint": (
                self.axis.fingerprint()
            ),
            "source_input_lineage_fingerprint": (
                self
                .source_inputs
                .lineage_fingerprint()
            ),
            "source_compiled_prior_fingerprint": (
                self
                .source_compiled_prior_fingerprint
            ),
            "strength": float(self.strength),
            "resolution_summary": dict(
                self.resolution_summary
            ),
        }

    def lineage_fingerprint(
        self,
    ) -> str:
        return _fingerprint(
            self.lineage_dict()
        )

    def value_fingerprint(
        self,
    ) -> str:
        tensors: dict[str, torch.Tensor] = {
            "logit_contribution": (
                self.logit_contribution
            ),
        }

        for name, value in (
            ("prior_mean", self.prior_mean),
            ("confidence", self.confidence),
            (
                "initialization_mask",
                self.initialization_mask,
            ),
            (
                "regularization_mask",
                self.regularization_mask,
            ),
        ):
            if value is not None:
                tensors[name] = value

        return _tensor_fingerprint(tensors)

    def fingerprint(
        self,
    ) -> str:
        return _fingerprint(
            {
                "lineage_fingerprint": (
                    self.lineage_fingerprint()
                ),
                "value_fingerprint": (
                    self.value_fingerprint()
                ),
            }
        )


# =============================================================================
# Combined-logit activation output
# =============================================================================


@dataclass(slots=True, frozen=True)
class GateActivationOutput:
    """
    Combined logits and activated node-relation gate values.

    ``gate_logits`` must equal the neural logits plus the optional prior
    contribution. Under the bounded sigmoid contract, ``gate_values`` must
    equal ``torch.sigmoid(gate_logits)`` within dtype-appropriate tolerance.
    """

    gate_logits: torch.Tensor
    gate_values: torch.Tensor

    source_network_output: GateNetworkOutput
    prior_contribution: (
        RelationPriorContribution | None
    )

    activation: str

    encoder_architecture_fingerprint: str
    parameter_fingerprint: str | None = None

    schema_version: str = (
        GATE_ACTIVATION_OUTPUT_SCHEMA_VERSION
    )

    def __post_init__(self) -> None:
        if not isinstance(
            self.source_network_output,
            GateNetworkOutput,
        ):
            raise TypeError(
                "source_network_output must be a "
                "GateNetworkOutput."
            )

        source_inputs = self.source_inputs

        _validate_node_relation_tensor(
            "gate_logits",
            self.gate_logits,
            source_inputs=source_inputs,
        )
        _validate_node_relation_tensor(
            "gate_values",
            self.gate_values,
            source_inputs=source_inputs,
        )

        _require_nonempty_string(
            "activation",
            self.activation,
        )

        if self.activation != (
            RELATION_GATE_ACTIVATION_SIGMOID
        ):
            raise ValueError(
                "The bounded V2.0 gate activation supports only "
                f"{RELATION_GATE_ACTIVATION_SIGMOID!r}."
            )

        expected_logits = (
            self
            .source_network_output
            .logits
        )

        if self.prior_contribution is not None:
            if not isinstance(
                self.prior_contribution,
                RelationPriorContribution,
            ):
                raise TypeError(
                    "prior_contribution must be a "
                    "RelationPriorContribution or None."
                )

            if (
                self
                .prior_contribution
                .source_inputs
                is not source_inputs
            ):
                raise ValueError(
                    "Prior contribution and neural logits must reference "
                    "the exact same source_inputs object."
                )

            if (
                self
                .prior_contribution
                .axis
                .fingerprint()
                != self.axis.fingerprint()
            ):
                raise ValueError(
                    "Prior contribution and neural logits must use the "
                    "same relation-gate axis."
                )

            expected_logits = (
                expected_logits
                + self
                .prior_contribution
                .logit_contribution
            )

        atol, rtol = _default_tolerances(
            self.gate_logits.dtype
        )

        if not torch.allclose(
            self.gate_logits,
            expected_logits,
            atol=atol,
            rtol=rtol,
        ):
            raise ValueError(
                "gate_logits must equal neural logits plus the optional "
                "prior logit contribution."
            )

        expected_values = torch.sigmoid(
            self.gate_logits
        )

        if not torch.allclose(
            self.gate_values,
            expected_values,
            atol=atol,
            rtol=rtol,
        ):
            raise ValueError(
                "gate_values must equal sigmoid(gate_logits)."
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
    def source_inputs(
        self,
    ) -> FunctionalMessagePassingInputs:
        return (
            self
            .source_network_output
            .source_inputs
        )

    @property
    def axis(self) -> RelationGateAxis:
        return (
            self
            .source_network_output
            .axis
        )

    @property
    def scope(self) -> str:
        return (
            self
            .source_network_output
            .scope
        )

    @property
    def device(self) -> torch.device:
        return self.gate_values.device

    @property
    def dtype(self) -> torch.dtype:
        return self.gate_values.dtype

    @property
    def control_relation_mask(
        self,
    ) -> torch.Tensor:
        return self.axis.control_relation_mask

    def lineage_dict(
        self,
    ) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "activation": self.activation,
            "scope": self.scope,
            "source_network_fingerprint": (
                self
                .source_network_output
                .fingerprint()
            ),
            "prior_contribution_fingerprint": (
                self
                .prior_contribution
                .fingerprint()
                if self.prior_contribution
                is not None
                else None
            ),
            "encoder_architecture_fingerprint": (
                self
                .encoder_architecture_fingerprint
            ),
            "parameter_fingerprint": (
                self.parameter_fingerprint
            ),
        }

    def lineage_fingerprint(
        self,
    ) -> str:
        return _fingerprint(
            self.lineage_dict()
        )

    def value_fingerprint(
        self,
    ) -> str:
        return _tensor_fingerprint(
            {
                "gate_logits": (
                    self.gate_logits
                ),
                "gate_values": (
                    self.gate_values
                ),
            }
        )

    def fingerprint(
        self,
    ) -> str:
        return _fingerprint(
            {
                "lineage_fingerprint": (
                    self.lineage_fingerprint()
                ),
                "value_fingerprint": (
                    self.value_fingerprint()
                ),
            }
        )


__all__ = (
    "GATE_ACTIVATION_OUTPUT_SCHEMA_VERSION",
    "GATE_NETWORK_OUTPUT_SCHEMA_VERSION",
    "RELATION_GATE_AXIS_SCHEMA_VERSION",
    "RELATION_PRIOR_CONTRIBUTION_SCHEMA_VERSION",
    "GateActivationOutput",
    "GateNetworkOutput",
    "RelationGateAxis",
    "RelationGateOutput",
    "RelationPriorContribution",
)
