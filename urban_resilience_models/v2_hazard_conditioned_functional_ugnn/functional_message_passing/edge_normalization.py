"""
Structural edge normalization for functional message passing.

Target path:
    urban_resilience_models/
        v2_hazard_conditioned_functional_ugnn/
            functional_message_passing/
                edge_normalization.py

This module computes graph-structural scalar coefficients independently from:

- relation transforms;
- semantic edge weights supplied by data;
- hazard-conditioned relation gates;
- learned edge attention;
- message construction;
- target-node aggregation.

For every stored directed edge ``e = (s_e -> t_e)`` the structural
normalization stage returns one coefficient ``n_e``.

Bounded V2.0 baseline
---------------------
The implemented normalization mode is:

``none``
    ``n_e = 1`` for every edge.

This exact multiplicative identity is intentionally different from:

- enabled uniform attention, which may produce ``1 / group_size``;
- mean aggregation, which divides the final target-node edge sum by incoming
  edge count;
- semantic edge weights, which are external data coefficients.

Canonical future modes may be recognized by repository constants, but this
module raises ``NotImplementedError`` rather than silently substituting the
identity baseline.

Directed degree diagnostics
---------------------------
The output always retains:

``source_degree[i]``
    number of stored edges whose source is node ``i``;

``target_degree[i]``
    number of stored edges whose target is node ``i``.

These diagnostics do not alter the identity coefficient in V2.0. They make
future source-degree, target-degree, and symmetric normalization auditable
without conflating them with attention or aggregation.

Contract
--------
- input is a validated ``FunctionalMessagePassingInputs`` object;
- coefficients have shape ``[E]`` and match node-state dtype/device;
- source and target degrees have shape ``[N]`` and dtype ``torch.long``;
- zero-edge graphs return an empty coefficient tensor and zero degree vectors;
- isolated nodes have degree zero;
- no trainable parameters, hidden casting, device movement, edge insertion,
  clipping, semantic weighting, or fallback is performed.
"""

from __future__ import annotations

from hashlib import sha256
import json
from typing import Any, Final, Mapping

import torch
from torch import nn

from ..config import (
    FunctionalMessagePassingConfig,
)
from ..constants import (
    CANONICAL_EDGE_NORMALIZATION_TYPES,
    EDGE_NORMALIZATION_NONE,
    V2_0_IMPLEMENTED_EDGE_NORMALIZATION_TYPES,
)
from .schemas import (
    FunctionalMessagePassingInputs,
    StructuralEdgeNormalizationOutput,
)


# =============================================================================
# Public identity
# =============================================================================


EDGE_NORMALIZATION_SCHEMA_VERSION: Final[str] = "0.1"


# =============================================================================
# Validation and fingerprint helpers
# =============================================================================


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
            "edge normalization mode must be a string."
        )

    normalized = mode.strip()

    if not normalized:
        raise ValueError(
            "edge normalization mode must be a non-empty string."
        )

    if normalized not in (
        CANONICAL_EDGE_NORMALIZATION_TYPES
    ):
        raise ValueError(
            "Unknown edge normalization mode "
            f"{normalized!r}. Expected one of "
            f"{tuple(CANONICAL_EDGE_NORMALIZATION_TYPES)!r}."
        )

    if normalized not in (
        V2_0_IMPLEMENTED_EDGE_NORMALIZATION_TYPES
    ):
        raise NotImplementedError(
            "Edge normalization mode "
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


def _require_edge_index_vector(
    name: str,
    value: torch.Tensor,
    *,
    edge_count: int,
    node_count: int,
    device: torch.device,
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

    if int(value.shape[0]) != edge_count:
        raise ValueError(
            f"{name} length must equal the edge count {edge_count}; "
            f"observed {int(value.shape[0])}."
        )

    if not _devices_match(
        value.device,
        device,
    ):
        raise ValueError(
            f"{name} and functional message-passing inputs must share "
            f"one device. Observed {value.device} and {device}."
        )

    if value.numel() == 0:
        return

    if node_count <= 0:
        raise ValueError(
            f"{name} cannot be nonempty when the node count is zero."
        )

    minimum = int(
        value.min().item()
    )
    maximum = int(
        value.max().item()
    )

    if minimum < 0 or maximum >= node_count:
        raise ValueError(
            f"{name} contains out-of-range node indices. "
            f"Observed range [{minimum}, {maximum}]; "
            f"valid range is [0, {node_count - 1}]."
        )


# =============================================================================
# Structural normalization
# =============================================================================


class EdgeNormalization(nn.Module):
    """
    Compute graph-structural edge coefficients and directed degree metadata.

    Parameters
    ----------
    mode:
        Canonical edge-normalization mode.

    The bounded V2.0 implementation supports only ``none``.
    """

    mode: str

    def __init__(
        self,
        *,
        mode: str = EDGE_NORMALIZATION_NONE,
    ) -> None:
        super().__init__()

        self.mode = _normalize_mode(
            mode
        )

        if self.mode != EDGE_NORMALIZATION_NONE:
            # Defensive exhaustiveness check. _normalize_mode has already
            # rejected canonical unimplemented modes.
            raise RuntimeError(
                "Internal edge-normalization dispatch is incomplete for "
                f"mode {self.mode!r}."
            )

    # ------------------------------------------------------------------
    # Construction from configuration
    # ------------------------------------------------------------------

    @classmethod
    def from_config(
        cls,
        *,
        config: FunctionalMessagePassingConfig,
    ) -> "EdgeNormalization":
        """
        Build the structural-normalization stage from FMP configuration.
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
            mode=config.edge_normalization_type
        )

    # ------------------------------------------------------------------
    # Public identity
    # ------------------------------------------------------------------

    @property
    def parameter_count(self) -> int:
        return 0

    @property
    def trainable_parameter_count(
        self,
    ) -> int:
        return 0

    @property
    def is_identity(self) -> bool:
        return self.mode == (
            EDGE_NORMALIZATION_NONE
        )

    def architecture_dict(
        self,
    ) -> dict[str, Any]:
        return {
            "schema_version": (
                EDGE_NORMALIZATION_SCHEMA_VERSION
            ),
            "mode": self.mode,
            "implemented_formula": (
                "n_e = 1"
                if self.is_identity
                else None
            ),
            "parameter_count": 0,
            "relation_agnostic": True,
            "hazard_agnostic": True,
            "attention_independent": True,
            "semantic_edge_weight_independent": True,
            "aggregation_independent": True,
            "self_loop_policy": (
                "consume_stored_edges_without_insertion_or_removal"
            ),
            "degree_diagnostics": {
                "source_degree": (
                    "count_edges_by_source_node"
                ),
                "target_degree": (
                    "count_edges_by_target_node"
                ),
            },
            "operation_order": [
                "validate_fmp_inputs",
                "count_directed_source_degrees",
                "count_directed_target_degrees",
                "construct_identity_coefficients",
                "construct_metadata_preserving_output",
            ],
            "output_schema": (
                "StructuralEdgeNormalizationOutput"
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
        """
        Deterministic identity for a parameter-free module.
        """

        return _fingerprint(
            {
                "schema_version": (
                    EDGE_NORMALIZATION_SCHEMA_VERSION
                ),
                "module": type(self).__name__,
                "parameter_count": 0,
                "state_dict_keys": list(
                    self.state_dict()
                ),
            }
        )

    def assert_finite_parameters(
        self,
    ) -> None:
        """
        Parameter-free compatibility hook.

        Future trainable normalization modes must replace this no-op with
        explicit finite-parameter validation.
        """

        if self.parameter_count != 0:
            raise RuntimeError(
                "The bounded edge-normalization baseline must remain "
                "parameter-free."
            )

    # ------------------------------------------------------------------
    # Input validation and degree computation
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

        if inputs.num_nodes <= 0:
            raise ValueError(
                "Structural edge normalization requires at least one node."
            )

        if not inputs.dtype.is_floating_point:
            raise ValueError(
                "Functional message-passing node state must use a "
                "floating-point dtype."
            )

        _require_edge_index_vector(
            "source_index",
            inputs.source_index,
            edge_count=inputs.num_edges,
            node_count=inputs.num_nodes,
            device=inputs.device,
        )
        _require_edge_index_vector(
            "target_index",
            inputs.target_index,
            edge_count=inputs.num_edges,
            node_count=inputs.num_nodes,
            device=inputs.device,
        )

    def compute_degrees(
        self,
        inputs: FunctionalMessagePassingInputs,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Return directed source and target degree vectors.

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor]
            ``(source_degree, target_degree)``, each ``[N]`` and
            ``torch.long``.
        """

        self._validate_inputs(inputs)

        source_degree = torch.bincount(
            inputs.source_index,
            minlength=inputs.num_nodes,
        )
        target_degree = torch.bincount(
            inputs.target_index,
            minlength=inputs.num_nodes,
        )

        if tuple(source_degree.shape) != (
            inputs.num_nodes,
        ):
            raise RuntimeError(
                "Source-degree computation returned an invalid shape "
                f"{tuple(source_degree.shape)}; expected "
                f"{(inputs.num_nodes,)}."
            )

        if tuple(target_degree.shape) != (
            inputs.num_nodes,
        ):
            raise RuntimeError(
                "Target-degree computation returned an invalid shape "
                f"{tuple(target_degree.shape)}; expected "
                f"{(inputs.num_nodes,)}."
            )

        if (
            source_degree.dtype
            != torch.long
            or target_degree.dtype
            != torch.long
        ):
            raise RuntimeError(
                "Directed degree computation must return torch.long."
            )

        if (
            not _devices_match(
                source_degree.device,
                inputs.device,
            )
            or not _devices_match(
                target_degree.device,
                inputs.device,
            )
        ):
            raise RuntimeError(
                "Directed degree computation changed device."
            )

        if int(
            source_degree.sum().item()
        ) != inputs.num_edges:
            raise RuntimeError(
                "Source degrees do not sum to the stored edge count."
            )

        if int(
            target_degree.sum().item()
        ) != inputs.num_edges:
            raise RuntimeError(
                "Target degrees do not sum to the stored edge count."
            )

        return (
            source_degree,
            target_degree,
        )

    # ------------------------------------------------------------------
    # Mathematical dispatch
    # ------------------------------------------------------------------

    def compute_coefficients(
        self,
        inputs: FunctionalMessagePassingInputs,
    ) -> torch.Tensor:
        """
        Return edge-aligned structural coefficients ``[E]``.

        ``none`` returns the exact multiplicative identity one for every
        stored edge.
        """

        self._validate_inputs(inputs)

        if self.mode == (
            EDGE_NORMALIZATION_NONE
        ):
            coefficients = torch.ones(
                inputs.num_edges,
                dtype=inputs.dtype,
                device=inputs.device,
            )
        else:
            raise RuntimeError(
                "Internal edge-normalization dispatch reached an "
                f"unsupported mode {self.mode!r}."
            )

        expected_shape = (
            inputs.num_edges,
        )

        if tuple(coefficients.shape) != (
            expected_shape
        ):
            raise RuntimeError(
                "Edge-normalization implementation returned shape "
                f"{tuple(coefficients.shape)}; expected "
                f"{expected_shape}."
            )

        if coefficients.dtype != (
            inputs.dtype
        ):
            raise RuntimeError(
                "Edge-normalization implementation changed dtype."
            )

        if not _devices_match(
            coefficients.device,
            inputs.device,
        ):
            raise RuntimeError(
                "Edge-normalization implementation changed device."
            )

        if not bool(
            torch.isfinite(coefficients)
            .all()
            .item()
        ):
            raise FloatingPointError(
                "Edge normalization produced NaN or infinity."
            )

        if bool(
            (coefficients < 0)
            .any()
            .item()
        ):
            raise FloatingPointError(
                "Structural edge-normalization coefficients must be "
                "nonnegative."
            )

        if self.mode == (
            EDGE_NORMALIZATION_NONE
        ) and not torch.equal(
            coefficients,
            torch.ones_like(coefficients),
        ):
            raise RuntimeError(
                "Normalization mode 'none' must return exact identity "
                "coefficients."
            )

        return coefficients

    def forward(
        self,
        inputs: FunctionalMessagePassingInputs,
    ) -> StructuralEdgeNormalizationOutput:
        """
        Compute structural coefficients and preserve complete input metadata.
        """

        source_degree, target_degree = (
            self.compute_degrees(inputs)
        )
        coefficients = (
            self.compute_coefficients(inputs)
        )

        return StructuralEdgeNormalizationOutput(
            coefficients=coefficients,
            source_inputs=inputs,
            normalization_mode=self.mode,
            encoder_architecture_fingerprint=(
                self.architecture_fingerprint()
            ),
            source_degree=source_degree,
            target_degree=target_degree,
        )

    # ------------------------------------------------------------------
    # Representation
    # ------------------------------------------------------------------

    def extra_repr(self) -> str:
        return (
            f"mode={self.mode!r}, "
            "parameter_count=0, "
            "degree_diagnostics=True"
        )


__all__ = (
    "EDGE_NORMALIZATION_SCHEMA_VERSION",
    "EdgeNormalization",
)
