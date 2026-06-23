"""
Relation-gate activation dispatch.

Target path:
    urban_resilience_models/
        v2_hazard_conditioned_functional_ugnn/
            functional_message_passing/
                relation_family_gate/
                    activations.py

The bounded V2.0 relation gate uses independent sigmoid activations over the
exact compiled relation axis ``R``:

    gate_values[n, r] = sigmoid(gate_logits[n, r])

Relation channels do not compete through softmax. Several relations may be
active simultaneously for the same target node and hazard query.

This module owns:

- canonical-versus-implemented activation validation;
- optional prior-logit composition;
- sigmoid activation;
- parameter-free architecture and parameter fingerprints;
- metadata-preserving ``GateActivationOutput`` construction.

It does not own:

- neural gate-logit prediction;
- hazard-relation prior compilation;
- prior resolution or prior-to-logit conversion;
- edge-aligned relation lookup;
- message construction, attention, or aggregation.

Bounded V2.0 contract
---------------------
For neural logits ``L`` and an optional prior contribution ``P``:

    combined_logits = L                  when P is absent
    combined_logits = L + P              when P is present
    gate_values     = sigmoid(combined_logits)

All tensors have shape ``[N, R]``, where ``N`` is the number of target nodes
and ``R`` is the number of exact compiled relation identities.

Canonical future activation names may exist in repository constants, but this
module raises ``NotImplementedError`` rather than silently substituting
sigmoid.
"""

from __future__ import annotations

from hashlib import sha256
import json
from typing import Any, Final, Mapping

import torch
from torch import nn

from ...config import RelationConfig
from ...constants import (
    CANONICAL_RELATION_GATE_ACTIVATIONS,
    RELATION_GATE_ACTIVATION_SIGMOID,
    V2_0_IMPLEMENTED_RELATION_GATE_ACTIVATIONS,
)
from .schemas import (
    GateActivationOutput,
    GateNetworkOutput,
    RelationPriorContribution,
)


# =============================================================================
# Public identity
# =============================================================================


RELATION_GATE_ACTIVATIONS_SCHEMA_VERSION: Final[str] = "0.1"


# =============================================================================
# Validation and fingerprint helpers
# =============================================================================


def _normalize_activation(
    activation: str,
) -> str:
    if not isinstance(
        activation,
        str,
    ):
        raise TypeError(
            "relation-gate activation must be a string."
        )

    normalized = activation.strip()

    if not normalized:
        raise ValueError(
            "relation-gate activation must be a non-empty string."
        )

    if normalized not in (
        CANONICAL_RELATION_GATE_ACTIVATIONS
    ):
        raise ValueError(
            "Unknown relation-gate activation "
            f"{normalized!r}. Expected one of "
            f"{tuple(CANONICAL_RELATION_GATE_ACTIVATIONS)!r}."
        )

    if normalized not in (
        V2_0_IMPLEMENTED_RELATION_GATE_ACTIVATIONS
    ):
        raise NotImplementedError(
            "Relation-gate activation "
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
        _canonical_json(payload).encode(
            "utf-8"
        )
    ).hexdigest()


def _devices_match(
    first: torch.device | str,
    second: torch.device | str,
) -> bool:
    first_device = torch.device(first)
    second_device = torch.device(second)

    if first_device.type != (
        second_device.type
    ):
        return False

    if first_device.type != "cuda":
        return first_device == (
            second_device
        )

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


def _require_finite_float_matrix(
    name: str,
    value: torch.Tensor,
) -> None:
    if not isinstance(
        value,
        torch.Tensor,
    ):
        raise TypeError(
            f"{name} must be a tensor."
        )

    if value.ndim != 2:
        raise ValueError(
            f"{name} must have shape [N, R]; "
            f"observed {tuple(value.shape)}."
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


def _require_network_output(
    source_network_output: GateNetworkOutput,
) -> None:
    if not isinstance(
        source_network_output,
        GateNetworkOutput,
    ):
        raise TypeError(
            "source_network_output must be a GateNetworkOutput."
        )


def _require_compatible_prior(
    source_network_output: GateNetworkOutput,
    prior_contribution: (
        RelationPriorContribution | None
    ),
) -> None:
    if prior_contribution is None:
        return

    if not isinstance(
        prior_contribution,
        RelationPriorContribution,
    ):
        raise TypeError(
            "prior_contribution must be a "
            "RelationPriorContribution or None."
        )

    if (
        prior_contribution.source_inputs
        is not source_network_output.source_inputs
    ):
        raise ValueError(
            "Prior contribution and gate-network output must reference "
            "the exact same source_inputs object."
        )

    if (
        prior_contribution.axis.fingerprint()
        != source_network_output.axis.fingerprint()
    ):
        raise ValueError(
            "Prior contribution and gate-network output must use the "
            "same relation-gate axis."
        )

    if prior_contribution.logit_contribution.shape != (
        source_network_output.logits.shape
    ):
        raise ValueError(
            "Prior contribution and neural logits must have the same "
            "shape."
        )

    if prior_contribution.logit_contribution.dtype != (
        source_network_output.logits.dtype
    ):
        raise ValueError(
            "Prior contribution and neural logits must use the same "
            "dtype."
        )

    if not _devices_match(
        prior_contribution
        .logit_contribution
        .device,
        source_network_output.logits.device,
    ):
        raise ValueError(
            "Prior contribution and neural logits must share one device."
        )


def _validate_activation_result(
    *,
    logits: torch.Tensor,
    values: torch.Tensor,
) -> None:
    if not isinstance(
        values,
        torch.Tensor,
    ):
        raise RuntimeError(
            "Relation-gate activation must return a tensor."
        )

    if values.shape != logits.shape:
        raise RuntimeError(
            "Relation-gate activation changed shape. "
            f"Observed {tuple(values.shape)}; expected "
            f"{tuple(logits.shape)}."
        )

    if values.dtype != logits.dtype:
        raise RuntimeError(
            "Relation-gate activation changed dtype."
        )

    if not _devices_match(
        values.device,
        logits.device,
    ):
        raise RuntimeError(
            "Relation-gate activation changed device."
        )

    if not bool(
        torch.isfinite(values)
        .all()
        .item()
    ):
        raise FloatingPointError(
            "Relation-gate activation produced NaN or infinity."
        )

    if bool(
        (
            (values < 0)
            | (values > 1)
        )
        .any()
        .item()
    ):
        raise FloatingPointError(
            "Sigmoid relation-gate values must lie in [0, 1]."
        )


# =============================================================================
# Functional activation
# =============================================================================


def sigmoid_gate_activation(
    logits: torch.Tensor,
) -> torch.Tensor:
    """
    Apply independent sigmoid activation to node-relation logits.

    Parameters
    ----------
    logits:
        Finite floating-point tensor ``[N, R]``.

    Returns
    -------
    torch.Tensor
        Tensor ``[N, R]`` on the same device and with the same dtype.
    """

    _require_finite_float_matrix(
        "logits",
        logits,
    )

    values = torch.sigmoid(logits)

    _validate_activation_result(
        logits=logits,
        values=values,
    )

    return values


def apply_relation_gate_activation(
    logits: torch.Tensor,
    *,
    activation: str = (
        RELATION_GATE_ACTIVATION_SIGMOID
    ),
) -> torch.Tensor:
    """
    Dispatch a canonical relation-gate activation.

    Canonical-but-unimplemented modes raise ``NotImplementedError``.
    """

    normalized = _normalize_activation(
        activation
    )

    if normalized == (
        RELATION_GATE_ACTIVATION_SIGMOID
    ):
        return sigmoid_gate_activation(
            logits
        )

    raise RuntimeError(
        "Internal relation-gate activation dispatch is incomplete for "
        f"activation {normalized!r}."
    )


# =============================================================================
# Metadata-preserving activation module
# =============================================================================


class RelationGateActivation(nn.Module):
    """
    Combine neural and prior logits, then apply the configured activation.

    Parameters
    ----------
    activation:
        Canonical relation-gate activation. Bounded V2.0 supports only
        ``sigmoid``.
    """

    activation: str

    def __init__(
        self,
        *,
        activation: str = (
            RELATION_GATE_ACTIVATION_SIGMOID
        ),
    ) -> None:
        super().__init__()

        self.activation = (
            _normalize_activation(
                activation
            )
        )

        if self.activation != (
            RELATION_GATE_ACTIVATION_SIGMOID
        ):
            raise RuntimeError(
                "Internal relation-gate activation dispatch is "
                f"incomplete for activation {self.activation!r}."
            )

    # ------------------------------------------------------------------
    # Construction from configuration
    # ------------------------------------------------------------------

    @classmethod
    def from_config(
        cls,
        *,
        config: RelationConfig,
    ) -> "RelationGateActivation":
        """
        Build the activation stage from relation configuration.
        """

        if not isinstance(
            config,
            RelationConfig,
        ):
            raise TypeError(
                "config must be a RelationConfig."
            )

        config.validate()

        if config.gate_enabled:
            config.assert_implemented()

        return cls(
            activation=config.gate_activation
        )

    # ------------------------------------------------------------------
    # Public identity
    # ------------------------------------------------------------------

    @property
    def is_sigmoid(self) -> bool:
        return self.activation == (
            RELATION_GATE_ACTIVATION_SIGMOID
        )

    @property
    def parameter_count(self) -> int:
        return 0

    @property
    def trainable_parameter_count(
        self,
    ) -> int:
        return 0

    def architecture_dict(
        self,
    ) -> dict[str, Any]:
        return {
            "schema_version": (
                RELATION_GATE_ACTIVATIONS_SCHEMA_VERSION
            ),
            "activation": self.activation,
            "implemented_formula": (
                "gate_values = sigmoid(gate_logits)"
                if self.is_sigmoid
                else None
            ),
            "parameter_count": 0,
            "gate_axis": (
                "exact_compiled_relation_axis"
            ),
            "node_scope": "target_node",
            "relation_channels_compete": False,
            "prior_integration": (
                "additive_in_logit_space"
            ),
            "prior_absent_policy": (
                "reuse_neural_logits"
            ),
            "value_range": [0.0, 1.0],
            "operation_order": [
                "validate_gate_network_output",
                "validate_optional_prior_contribution",
                "add_prior_logit_contribution_when_present",
                "apply_independent_sigmoid",
                "construct_gate_activation_output",
            ],
            "output_schema": (
                "GateActivationOutput"
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
        return _fingerprint(
            {
                "schema_version": (
                    RELATION_GATE_ACTIVATIONS_SCHEMA_VERSION
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
        if self.parameter_count != 0:
            raise RuntimeError(
                "The bounded relation-gate activation must remain "
                "parameter-free."
            )

    # ------------------------------------------------------------------
    # Logit composition and activation
    # ------------------------------------------------------------------

    def combine_logits(
        self,
        source_network_output: GateNetworkOutput,
        prior_contribution: (
            RelationPriorContribution | None
        ) = None,
    ) -> torch.Tensor:
        """
        Return neural logits plus an optional prior contribution.

        When no prior is supplied, the exact neural-logit tensor is returned
        without cloning.
        """

        _require_network_output(
            source_network_output
        )
        _require_compatible_prior(
            source_network_output,
            prior_contribution,
        )

        if prior_contribution is None:
            combined = (
                source_network_output.logits
            )
        else:
            combined = (
                source_network_output.logits
                + prior_contribution
                .logit_contribution
            )

        _require_finite_float_matrix(
            "combined_logits",
            combined,
        )

        if combined.shape != (
            source_network_output.logits.shape
        ):
            raise RuntimeError(
                "Relation-gate logit composition changed shape."
            )

        if combined.dtype != (
            source_network_output.logits.dtype
        ):
            raise RuntimeError(
                "Relation-gate logit composition changed dtype."
            )

        if not _devices_match(
            combined.device,
            source_network_output.logits.device,
        ):
            raise RuntimeError(
                "Relation-gate logit composition changed device."
            )

        return combined

    def activate_tensor(
        self,
        logits: torch.Tensor,
    ) -> torch.Tensor:
        """
        Apply the configured activation to ``[N, R]`` logits.
        """

        return apply_relation_gate_activation(
            logits,
            activation=self.activation,
        )

    def forward(
        self,
        source_network_output: GateNetworkOutput,
        prior_contribution: (
            RelationPriorContribution | None
        ) = None,
    ) -> GateActivationOutput:
        """
        Compose logits, activate them, and preserve complete provenance.
        """

        combined_logits = self.combine_logits(
            source_network_output,
            prior_contribution,
        )
        gate_values = self.activate_tensor(
            combined_logits
        )

        return GateActivationOutput(
            gate_logits=combined_logits,
            gate_values=gate_values,
            source_network_output=(
                source_network_output
            ),
            prior_contribution=(
                prior_contribution
            ),
            activation=self.activation,
            encoder_architecture_fingerprint=(
                self.architecture_fingerprint()
            ),
            parameter_fingerprint=(
                self.parameter_fingerprint()
            ),
        )

    # ------------------------------------------------------------------
    # Representation
    # ------------------------------------------------------------------

    def extra_repr(self) -> str:
        return (
            f"activation={self.activation!r}, "
            "parameter_count=0, "
            "relation_channels_compete=False"
        )


# Compact alias for call sites that prefer the stage name.
GateActivation = RelationGateActivation


__all__ = (
    "RELATION_GATE_ACTIVATIONS_SCHEMA_VERSION",
    "GateActivation",
    "RelationGateActivation",
    "apply_relation_gate_activation",
    "sigmoid_gate_activation",
)
