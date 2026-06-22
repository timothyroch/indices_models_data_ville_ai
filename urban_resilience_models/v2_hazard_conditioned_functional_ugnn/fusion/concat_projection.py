"""
Research-grade concat-projection fusion for the V2 functional UGNN.

Target path:
    urban_resilience_models/
        v2_hazard_conditioned_functional_ugnn/
            fusion/
                concat_projection.py

This module owns the mathematical implementation of the baseline
``concat_projection`` fusion algorithm:

1. validate one explicitly ordered mapping of component tensors;
2. project every component independently to a common width;
3. concatenate projected components in canonical order;
4. transform the concatenated state through a fusion MLP;
5. return the fused state and metadata-preserving intermediate tensors.

It does not own:

- extraction of tensors from ``NodeStateFusionInputs``;
- configuration dispatch across fusion modes;
- upstream alignment or lineage validation;
- hazard-conditioned gating;
- node-type experts;
- FiLM conditioning;
- component attribution;
- uncertainty propagation.

Why this baseline is explicit
-----------------------------
A concatenation baseline is scientifically useful only when its execution is
fully specified. This implementation therefore freezes:

- semantic component names;
- canonical ordering;
- per-component input widths;
- per-component projection architecture;
- final fusion architecture;
- optional retention of the concatenated representation;
- optional exact input-value fingerprinting;
- architecture and parameter identities.

The implementation deliberately avoids hidden component reordering, implicit
broadcasting, silent missing-component substitution, and unexplained tensor
lists.

Mathematical contract
---------------------
For active components ``c_1, ..., c_K`` with item-aligned states ``x_k``:

    z_k = Project_k(x_k)

    z = concat(z_1, ..., z_K)

    h = Norm(
            Linear_out(
                Dropout(
                    GELU(
                        Linear_in(z)
                    )
                )
            )
        )

Each ``z_k`` has width ``output_dim``. The concatenated state has width
``K * output_dim`` and the final fused state has width ``output_dim``.

This is intentionally a baseline, not an adaptive fusion rule. Hazard and
node-type states may be included as components, but they do not explicitly
gate or modulate other components here.
"""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from hashlib import sha256
import json
import math
from types import MappingProxyType
from typing import Any, Final, Mapping, Sequence

import torch
from torch import nn

from .component_projection import (
    ComponentProjection,
    ComponentProjectionActivation,
    ComponentProjectionNormalization,
)


# =============================================================================
# Schema identity
# =============================================================================


CONCAT_PROJECTION_FUSION_SCHEMA_VERSION: Final[str] = "0.1"
CONCAT_PROJECTION_OUTPUT_SCHEMA_VERSION: Final[str] = "0.1"


# =============================================================================
# Canonical component identity
# =============================================================================


FUSION_COMPONENT_STATIC_STATE: Final[str] = "static_state"
FUSION_COMPONENT_MEMORY_STATE: Final[str] = "memory_state"
FUSION_COMPONENT_HAZARD_MEMORY_STATE: Final[str] = (
    "hazard_memory_state"
)
FUSION_COMPONENT_HAZARD_CONTEXT: Final[str] = "hazard_context"
FUSION_COMPONENT_NODE_TYPE_EMBEDDING: Final[str] = (
    "node_type_embedding"
)


CANONICAL_FUSION_COMPONENT_ORDER: Final[tuple[str, ...]] = (
    FUSION_COMPONENT_STATIC_STATE,
    FUSION_COMPONENT_MEMORY_STATE,
    FUSION_COMPONENT_HAZARD_MEMORY_STATE,
    FUSION_COMPONENT_HAZARD_CONTEXT,
    FUSION_COMPONENT_NODE_TYPE_EMBEDDING,
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


def _require_probability(
    name: str,
    value: int | float,
) -> float:
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

    if not 0.0 <= converted < 1.0:
        raise ValueError(
            f"{name} must lie in [0, 1)."
        )

    return converted


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
    """
    Compute an exact deterministic fingerprint of named tensor values.

    This operation moves detached tensor values to CPU. It is therefore
    intentionally opt-in on the forward path.
    """

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


def canonical_component_order(
    component_names: Sequence[str],
) -> tuple[str, ...]:
    """
    Return canonical order for known fusion components.

    Unknown names are rejected because silently placing them after known
    components would make architecture identity depend on incidental mapping
    order. Callers using experimental components must provide an explicit
    order directly to ``ConcatProjectionFusion``.
    """

    names = tuple(component_names)
    _require_unique_strings(
        "component_names",
        names,
    )

    unknown = sorted(
        set(names)
        - set(CANONICAL_FUSION_COMPONENT_ORDER)
    )

    if unknown:
        raise ValueError(
            "Canonical ordering is undefined for experimental fusion "
            f"components: {unknown}. Supply component_order explicitly."
        )

    selected = set(names)

    return tuple(
        name
        for name in CANONICAL_FUSION_COMPONENT_ORDER
        if name in selected
    )


# =============================================================================
# Output contract
# =============================================================================


@dataclass(slots=True, frozen=True)
class ConcatProjectionFusionOutput:
    """
    Output of ``ConcatProjectionFusion``.

    ``projected_components`` is immutable and preserves the exact active
    component order. ``concatenated_state`` and ``input_value_fingerprint``
    are optional diagnostics controlled by the fusion architecture.
    """

    fused_state: torch.Tensor
    projected_components: Mapping[str, torch.Tensor]

    component_order: tuple[str, ...]
    architecture_fingerprint: str

    concatenated_state: torch.Tensor | None = None
    input_value_fingerprint: str | None = None

    schema_version: str = (
        CONCAT_PROJECTION_OUTPUT_SCHEMA_VERSION
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
                "fused_state must have shape [items, output_dim]."
            )

        if not self.fused_state.dtype.is_floating_point:
            raise ValueError(
                "fused_state must use a floating-point dtype."
            )

        _assert_finite_tensor(
            "fused_state",
            self.fused_state,
        )

        _require_unique_strings(
            "component_order",
            self.component_order,
        )

        if not self.component_order:
            raise ValueError(
                "component_order cannot be empty."
            )

        if not isinstance(
            self.projected_components,
            Mapping,
        ):
            raise TypeError(
                "projected_components must be a mapping."
            )

        observed_order = tuple(
            self.projected_components
        )

        if observed_order != self.component_order:
            raise ValueError(
                "projected_components order differs from "
                "component_order."
            )

        copied: dict[str, torch.Tensor] = {}
        output_dim = int(
            self.fused_state.shape[1]
        )
        item_count = int(
            self.fused_state.shape[0]
        )

        for name in self.component_order:
            tensor = self.projected_components[name]

            if not isinstance(
                tensor,
                torch.Tensor,
            ):
                raise TypeError(
                    f"Projected component {name!r} must be a tensor."
                )

            if tensor.ndim != 2:
                raise ValueError(
                    f"Projected component {name!r} must have shape "
                    "[items, output_dim]."
                )

            if tuple(tensor.shape) != (
                item_count,
                output_dim,
            ):
                raise ValueError(
                    f"Projected component {name!r} shape differs "
                    "from the fusion output contract."
                )

            if tensor.device != self.fused_state.device:
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

        object.__setattr__(
            self,
            "projected_components",
            MappingProxyType(copied),
        )

        if self.concatenated_state is not None:
            if not isinstance(
                self.concatenated_state,
                torch.Tensor,
            ):
                raise TypeError(
                    "concatenated_state must be a tensor or None."
                )

            expected_width = (
                len(self.component_order)
                * output_dim
            )

            if tuple(
                self.concatenated_state.shape
            ) != (
                item_count,
                expected_width,
            ):
                raise ValueError(
                    "concatenated_state shape differs from the "
                    "component-concatenation contract."
                )

            if (
                self.concatenated_state.device
                != self.fused_state.device
            ):
                raise ValueError(
                    "concatenated_state and fused_state must share "
                    "one device."
                )

            if not (
                self.concatenated_state
                .dtype
                .is_floating_point
            ):
                raise ValueError(
                    "concatenated_state must use a floating-point "
                    "dtype."
                )

            _assert_finite_tensor(
                "concatenated_state",
                self.concatenated_state,
            )

        _require_nonempty_string(
            "architecture_fingerprint",
            self.architecture_fingerprint,
        )
        _require_nonempty_string(
            "schema_version",
            self.schema_version,
        )

        if self.input_value_fingerprint is not None:
            _require_nonempty_string(
                "input_value_fingerprint",
                self.input_value_fingerprint,
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
    def component_count(self) -> int:
        return len(
            self.component_order
        )


# =============================================================================
# Concat-projection fusion
# =============================================================================


class ConcatProjectionFusion(nn.Module):
    """
    Independently project, concatenate, and fuse aligned component states.

    Parameters
    ----------
    component_input_dims:
        Mapping from stable component name to its incoming width.

    output_dim:
        Common projected width and final fused-state width.

    component_order:
        Explicit order of concatenation. When omitted, all component names
        must belong to ``CANONICAL_FUSION_COMPONENT_ORDER`` and that canonical
        ordering is used.

    dropout:
        Dropout probability used in each component projector and in the final
        fusion MLP.

    layer_norm:
        Applies LayerNorm after each component projection and after the final
        fusion projection. This reproduces the original baseline contract.

    retain_concatenated_state:
        When true, return the concatenated projected representation for
        diagnostics. This may retain a large activation tensor.

    record_input_fingerprint:
        When true, compute an exact CPU fingerprint of all input values during
        each forward pass. This is useful for audits but intentionally disabled
        by default because it creates synchronization and transfer overhead.
    """

    def __init__(
        self,
        *,
        component_input_dims: Mapping[str, int],
        output_dim: int,
        component_order: Sequence[str] | None = None,
        dropout: float = 0.0,
        layer_norm: bool = True,
        retain_concatenated_state: bool = False,
        record_input_fingerprint: bool = False,
    ) -> None:
        super().__init__()

        if not isinstance(
            component_input_dims,
            Mapping,
        ):
            raise TypeError(
                "component_input_dims must be a mapping."
            )

        if not component_input_dims:
            raise ValueError(
                "At least one fusion component is required."
            )

        _require_positive_int(
            "output_dim",
            output_dim,
        )

        if not isinstance(
            layer_norm,
            bool,
        ):
            raise TypeError(
                "layer_norm must be a Boolean."
            )

        if not isinstance(
            retain_concatenated_state,
            bool,
        ):
            raise TypeError(
                "retain_concatenated_state must be a Boolean."
            )

        if not isinstance(
            record_input_fingerprint,
            bool,
        ):
            raise TypeError(
                "record_input_fingerprint must be a Boolean."
            )

        resolved_dims: dict[str, int] = {}

        for name, width in (
            component_input_dims.items()
        ):
            _require_nonempty_string(
                "component name",
                name,
            )
            _require_positive_int(
                f"component_input_dims[{name!r}]",
                width,
            )
            resolved_dims[name] = width

        if component_order is None:
            resolved_order = canonical_component_order(
                tuple(resolved_dims)
            )
        else:
            resolved_order = tuple(
                component_order
            )
            _require_unique_strings(
                "component_order",
                resolved_order,
            )

            missing = sorted(
                set(resolved_dims)
                - set(resolved_order)
            )
            unexpected = sorted(
                set(resolved_order)
                - set(resolved_dims)
            )

            if missing or unexpected:
                raise ValueError(
                    "component_order and component_input_dims must "
                    "contain exactly the same component names. "
                    f"Missing from order: {missing}; "
                    f"unexpected in order: {unexpected}."
                )

        if not resolved_order:
            raise ValueError(
                "component_order cannot be empty."
            )

        self.component_input_dims = MappingProxyType(
            {
                name: resolved_dims[name]
                for name in resolved_order
            }
        )
        self.component_order = resolved_order
        self.output_dim = output_dim
        self.dropout = _require_probability(
            "dropout",
            dropout,
        )
        self.layer_norm = layer_norm
        self.retain_concatenated_state = (
            retain_concatenated_state
        )
        self.record_input_fingerprint = (
            record_input_fingerprint
        )

        normalization = (
            ComponentProjectionNormalization.LAYER_NORM
            if layer_norm
            else ComponentProjectionNormalization.NONE
        )

        self.component_projections = nn.ModuleDict(
            OrderedDict(
                (
                    name,
                    ComponentProjection(
                        input_dim=(
                            self.component_input_dims[name]
                        ),
                        output_dim=output_dim,
                        component_name=name,
                        activation=(
                            ComponentProjectionActivation.GELU
                        ),
                        normalization=normalization,
                        dropout=self.dropout,
                    ),
                )
                for name in self.component_order
            )
        )

        fusion_input_dim = (
            len(self.component_order)
            * output_dim
        )

        final_normalization: nn.Module = (
            nn.LayerNorm(output_dim)
            if layer_norm
            else nn.Identity()
        )

        self.fusion_network = nn.Sequential(
            OrderedDict(
                (
                    (
                        "linear_in",
                        nn.Linear(
                            fusion_input_dim,
                            output_dim,
                        ),
                    ),
                    ("activation", nn.GELU()),
                    (
                        "dropout",
                        nn.Dropout(self.dropout),
                    ),
                    (
                        "linear_out",
                        nn.Linear(
                            output_dim,
                            output_dim,
                        ),
                    ),
                    (
                        "normalization",
                        final_normalization,
                    ),
                )
            )
        )

    # ------------------------------------------------------------------
    # Public properties
    # ------------------------------------------------------------------

    @property
    def component_count(self) -> int:
        return len(
            self.component_order
        )

    @property
    def fusion_input_dim(self) -> int:
        return (
            self.component_count
            * self.output_dim
        )

    @property
    def device(self) -> torch.device:
        return (
            self.fusion_network
            .linear_in
            .weight
            .device
        )

    @property
    def dtype(self) -> torch.dtype:
        return (
            self.fusion_network
            .linear_in
            .weight
            .dtype
        )

    # ------------------------------------------------------------------
    # Input validation
    # ------------------------------------------------------------------

    def _validate_components(
        self,
        components: Mapping[str, torch.Tensor],
    ) -> tuple[int, torch.device]:
        if not isinstance(
            components,
            Mapping,
        ):
            raise TypeError(
                "components must be a mapping from stable component "
                "names to tensors."
            )

        observed_order = tuple(
            components
        )
        observed_names = set(
            observed_order
        )
        expected_names = set(
            self.component_order
        )

        missing = sorted(
            expected_names - observed_names
        )
        unexpected = sorted(
            observed_names - expected_names
        )

        if missing or unexpected:
            raise ValueError(
                "Fusion components differ from the architecture "
                f"contract. Missing: {missing}; unexpected: "
                f"{unexpected}."
            )

        if observed_order != self.component_order:
            raise ValueError(
                "Fusion component mapping order differs from the "
                "configured component_order. Rebuild the mapping in "
                "canonical order rather than relying on implicit "
                "reordering."
            )

        item_count: int | None = None
        shared_device: torch.device | None = None

        for name in self.component_order:
            values = components[name]

            if not isinstance(
                values,
                torch.Tensor,
            ):
                raise TypeError(
                    f"Fusion component {name!r} must be a tensor."
                )

            if values.ndim != 2:
                raise ValueError(
                    f"Fusion component {name!r} must have shape "
                    "[items, feature_dim]."
                )

            expected_width = (
                self.component_input_dims[name]
            )

            if int(
                values.shape[1]
            ) != expected_width:
                raise ValueError(
                    f"Fusion component {name!r} has width "
                    f"{int(values.shape[1])}; expected "
                    f"{expected_width}."
                )

            if not values.dtype.is_floating_point:
                raise ValueError(
                    f"Fusion component {name!r} must use a "
                    "floating-point dtype."
                )

            if values.device != self.device:
                raise ValueError(
                    f"Fusion component {name!r} and "
                    "ConcatProjectionFusion must share one device."
                )

            _assert_finite_tensor(
                f"fusion component {name}",
                values,
            )

            rows = int(
                values.shape[0]
            )

            if item_count is None:
                item_count = rows
            elif rows != item_count:
                raise ValueError(
                    "All fusion components must have the same "
                    "item count."
                )

            if shared_device is None:
                shared_device = values.device
            elif values.device != shared_device:
                raise ValueError(
                    "All fusion components must share one device."
                )

        assert item_count is not None
        assert shared_device is not None

        return item_count, shared_device

    # ------------------------------------------------------------------
    # Forward path
    # ------------------------------------------------------------------

    def forward(
        self,
        components: Mapping[str, torch.Tensor],
    ) -> ConcatProjectionFusionOutput:
        item_count, _ = self._validate_components(
            components
        )

        projected: dict[
            str,
            torch.Tensor,
        ] = {}

        for name in self.component_order:
            state = self.component_projections[name](
                components[name]
            )

            if tuple(state.shape) != (
                item_count,
                self.output_dim,
            ):
                raise RuntimeError(
                    f"Projection for component {name!r} returned a "
                    "shape that differs from the fusion contract."
                )

            projected[name] = state

        observed_projected_order = tuple(
            projected
        )

        if observed_projected_order != (
            self.component_order
        ):
            raise RuntimeError(
                "Projected component order differs from the "
                "constructed architecture."
            )

        concatenated = torch.cat(
            [
                projected[name]
                for name in self.component_order
            ],
            dim=-1,
        )

        if tuple(
            concatenated.shape
        ) != (
            item_count,
            self.fusion_input_dim,
        ):
            raise RuntimeError(
                "Concatenated state shape differs from the "
                "fusion architecture."
            )

        _assert_finite_tensor(
            "concatenated fusion state",
            concatenated,
        )

        fused_state = self.fusion_network(
            concatenated
        )

        if tuple(
            fused_state.shape
        ) != (
            item_count,
            self.output_dim,
        ):
            raise RuntimeError(
                "Fused state shape differs from the "
                "concat-projection contract."
            )

        _assert_finite_tensor(
            "fused state",
            fused_state,
        )

        input_value_fingerprint = (
            _tensor_fingerprint(
                {
                    name: components[name]
                    for name in self.component_order
                }
            )
            if self.record_input_fingerprint
            else None
        )

        return ConcatProjectionFusionOutput(
            fused_state=fused_state,
            projected_components=projected,
            component_order=self.component_order,
            architecture_fingerprint=(
                self.architecture_fingerprint()
            ),
            concatenated_state=(
                concatenated
                if self.retain_concatenated_state
                else None
            ),
            input_value_fingerprint=(
                input_value_fingerprint
            ),
        )

    # ------------------------------------------------------------------
    # Architecture and parameter identity
    # ------------------------------------------------------------------

    def architecture_dict(
        self,
    ) -> dict[str, Any]:
        return {
            "schema_version": (
                CONCAT_PROJECTION_FUSION_SCHEMA_VERSION
            ),
            "algorithm": "concat_projection",
            "component_order": list(
                self.component_order
            ),
            "component_input_dims": {
                name: (
                    self.component_input_dims[name]
                )
                for name in self.component_order
            },
            "component_count": (
                self.component_count
            ),
            "output_dim": self.output_dim,
            "fusion_input_dim": (
                self.fusion_input_dim
            ),
            "dropout": self.dropout,
            "layer_norm": self.layer_norm,
            "retain_concatenated_state": (
                self.retain_concatenated_state
            ),
            "record_input_fingerprint": (
                self.record_input_fingerprint
            ),
            "component_projectors": {
                name: (
                    self.component_projections[name]
                    .architecture_dict()
                )
                for name in self.component_order
            },
            "fusion_network": {
                "operation_order": [
                    "linear_in",
                    "gelu",
                    "dropout",
                    "linear_out",
                    (
                        "layer_norm"
                        if self.layer_norm
                        else "identity"
                    ),
                ],
                "input_dim": self.fusion_input_dim,
                "hidden_dim": self.output_dim,
                "output_dim": self.output_dim,
            },
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

    def component_architecture_fingerprints(
        self,
    ) -> Mapping[str, str]:
        return MappingProxyType(
            {
                name: (
                    self.component_projections[name]
                    .architecture_fingerprint()
                )
                for name in self.component_order
            }
        )

    def component_parameter_fingerprints(
        self,
    ) -> Mapping[str, str]:
        return MappingProxyType(
            {
                name: (
                    self.component_projections[name]
                    .parameter_fingerprint()
                )
                for name in self.component_order
            }
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
                    f"ConcatProjectionFusion tensor {name!r} "
                    "contains NaN or infinity."
                )

        for projection in (
            self.component_projections.values()
        ):
            if not isinstance(
                projection,
                ComponentProjection,
            ):
                raise RuntimeError(
                    "component_projections contains an unexpected "
                    "module type."
                )
            projection.assert_finite_parameters()

    def extra_repr(
        self,
    ) -> str:
        return (
            f"component_order={self.component_order}, "
            f"output_dim={self.output_dim}, "
            f"dropout={self.dropout}, "
            f"layer_norm={self.layer_norm}, "
            f"retain_concatenated_state="
            f"{self.retain_concatenated_state}, "
            f"record_input_fingerprint="
            f"{self.record_input_fingerprint}"
        )


__all__ = (
    "CANONICAL_FUSION_COMPONENT_ORDER",
    "CONCAT_PROJECTION_FUSION_SCHEMA_VERSION",
    "CONCAT_PROJECTION_OUTPUT_SCHEMA_VERSION",
    "ConcatProjectionFusion",
    "ConcatProjectionFusionOutput",
    "FUSION_COMPONENT_HAZARD_CONTEXT",
    "FUSION_COMPONENT_HAZARD_MEMORY_STATE",
    "FUSION_COMPONENT_MEMORY_STATE",
    "FUSION_COMPONENT_NODE_TYPE_EMBEDDING",
    "FUSION_COMPONENT_STATIC_STATE",
    "canonical_component_order",
)
