"""
Hazard-conditioned orchestration of exact-relation gates.

Target path:
    urban_resilience_models/
        v2_hazard_conditioned_functional_ugnn/
            functional_message_passing/
                relation_family_gate/
                    relation_family_gate.py

Despite the historical package name ``relation_family_gate``, the bounded
V2.0 trainable gate axis is the exact compiled relation axis ``R``. Semantic
relation-family metadata is preserved by ``RelationGateAxis`` for diagnostics,
explanations, and future hierarchical models, but it is never used to pool,
collapse, or reorder relation channels.

This module coordinates the complete relation-gate path:

1. construct or validate the exact relation axis;
2. predict target-node neural relation logits;
3. optionally resolve compiled hazard-relation priors into logit space;
4. combine neural and prior logits;
5. apply independent sigmoid activation;
6. gather node-relation gate values onto stored edges;
7. construct the public ``RelationGateOutput`` contract.

It does not own:

- relation-registry compilation;
- hazard-query construction;
- hazard-relation prior compilation;
- neural gate-network internals;
- activation mathematics;
- edge attention;
- message construction;
- aggregation;
- training-loss definitions.

Bounded V2.0 contract
---------------------
For ``N`` nodes, ``R`` exact compiled relations, and ``E`` stored edges:

    neural_logits:       [N, R]
    prior_contribution:  [N, R]  (optional)
    gate_logits:         [N, R]
    gate_values:         [N, R]
    edge_gate_values:    [E]

The implemented equations are:

    gate_logits = neural_logits

when priors are absent, and:

    gate_logits = neural_logits + prior_logit_contribution

when priors are present. Activation is independent sigmoid:

    gate_values = sigmoid(gate_logits)

The exact edge lookup is:

    edge_gate_values[e] =
        gate_values[target_index[e], edge_relation_index[e]]

Several relation channels may be active simultaneously. No family pooling or
softmax competition is performed.
"""

from __future__ import annotations

from hashlib import sha256
import json
from typing import Any, Final, Mapping

import torch
from torch import nn

from ...config import RelationConfig
from ...constants import (
    RELATION_GATE_ACTIVATION_SIGMOID,
    RELATION_GATE_SCOPE_TARGET_NODE,
)
from ..schemas import (
    FunctionalMessagePassingInputs,
    RelationGateOutput,
)
from .activations import (
    RelationGateActivation,
)
from .gate_network import (
    RelationGateNetwork,
)
from .relation_priors import (
    RelationPriorContributionBuilder,
)
from .schemas import (
    GateActivationOutput,
    RelationGateAxis,
    RelationPriorContribution,
)


# =============================================================================
# Public identity
# =============================================================================


RELATION_FAMILY_GATE_SCHEMA_VERSION: Final[str] = "0.1"


# =============================================================================
# Validation and fingerprint helpers
# =============================================================================


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


def _require_inputs(
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

    if source_inputs.num_nodes <= 0:
        raise ValueError(
            "Relation gating requires at least one node."
        )

    if source_inputs.num_relations <= 0:
        raise ValueError(
            "Relation gating requires at least one relation."
        )

    if not source_inputs.dtype.is_floating_point:
        raise ValueError(
            "Relation gating requires a floating-point node-state dtype."
        )


def _require_network(
    value: RelationGateNetwork,
) -> None:
    if not isinstance(
        value,
        RelationGateNetwork,
    ):
        raise TypeError(
            "gate_network must be a RelationGateNetwork."
        )


def _require_activation(
    value: RelationGateActivation,
) -> None:
    if not isinstance(
        value,
        RelationGateActivation,
    ):
        raise TypeError(
            "activation must be a RelationGateActivation."
        )


def _require_optional_prior_builder(
    value: (
        RelationPriorContributionBuilder | None
    ),
) -> None:
    if value is not None and not isinstance(
        value,
        RelationPriorContributionBuilder,
    ):
        raise TypeError(
            "prior_builder must be a "
            "RelationPriorContributionBuilder or None."
        )


def _validate_edge_gate_values(
    edge_gate_values: torch.Tensor,
    *,
    source_inputs: FunctionalMessagePassingInputs,
) -> None:
    if not isinstance(
        edge_gate_values,
        torch.Tensor,
    ):
        raise RuntimeError(
            "Edge gate lookup must return a tensor."
        )

    expected_shape = (
        source_inputs.num_edges,
    )

    if tuple(edge_gate_values.shape) != (
        expected_shape
    ):
        raise RuntimeError(
            "Edge gate lookup changed shape. "
            f"Observed {tuple(edge_gate_values.shape)}; expected "
            f"{expected_shape}."
        )

    if not edge_gate_values.dtype.is_floating_point:
        raise RuntimeError(
            "Edge gate lookup must use a floating-point dtype."
        )

    if edge_gate_values.dtype != (
        source_inputs.dtype
    ):
        raise RuntimeError(
            "Edge gate lookup changed dtype."
        )

    if not _devices_match(
        edge_gate_values.device,
        source_inputs.device,
    ):
        raise RuntimeError(
            "Edge gate lookup changed device."
        )

    if not bool(
        torch.isfinite(edge_gate_values)
        .all()
        .item()
    ):
        raise FloatingPointError(
            "Edge gate lookup produced NaN or infinity."
        )

    if bool(
        (
            (edge_gate_values < 0)
            | (edge_gate_values > 1)
        )
        .any()
        .item()
    ):
        raise FloatingPointError(
            "Sigmoid edge gate values must lie in [0, 1]."
        )


# =============================================================================
# Relation-gate orchestration
# =============================================================================


class RelationFamilyGate(nn.Module):
    """
    Coordinate neural prediction, optional priors, activation, and edge lookup.

    The historical class name is retained for package continuity. Operationally,
    the module gates exact compiled relation identities rather than pooled
    semantic families.

    Parameters
    ----------
    gate_network:
        Neural predictor of target-node exact-relation logits.
    activation:
        Parameter-free relation-gate activation stage. Bounded V2.0 supports
        independent sigmoid activation.
    prior_builder:
        Optional adapter from compiled hazard-relation priors to node-aligned
        additive logit contributions.
    """

    gate_network: RelationGateNetwork
    activation: RelationGateActivation
    prior_builder: (
        RelationPriorContributionBuilder | None
    )

    def __init__(
        self,
        *,
        gate_network: RelationGateNetwork,
        activation: RelationGateActivation,
        prior_builder: (
            RelationPriorContributionBuilder | None
        ) = None,
    ) -> None:
        super().__init__()

        _require_network(
            gate_network
        )
        _require_activation(
            activation
        )
        _require_optional_prior_builder(
            prior_builder
        )

        if gate_network.scope != (
            RELATION_GATE_SCOPE_TARGET_NODE
        ):
            raise ValueError(
                "The bounded relation-family gate requires "
                f"{RELATION_GATE_SCOPE_TARGET_NODE!r} scope."
            )

        if activation.activation != (
            RELATION_GATE_ACTIVATION_SIGMOID
        ):
            raise ValueError(
                "The bounded relation-family gate requires "
                f"{RELATION_GATE_ACTIVATION_SIGMOID!r} activation."
            )

        self.gate_network = gate_network
        self.activation = activation
        self.prior_builder = prior_builder

    # ------------------------------------------------------------------
    # Construction from configuration
    # ------------------------------------------------------------------

    @classmethod
    def from_config(
        cls,
        *,
        config: RelationConfig,
        source_inputs: FunctionalMessagePassingInputs,
        use_node_state: bool = True,
        use_hazard_query: bool = True,
        layer_norm: bool = True,
        relation_bias: bool = True,
        prior_epsilon: float = 1e-4,
    ) -> "RelationFamilyGate":
        """
        Build a relation gate aligned to one FMP input contract.

        ``gate_enabled`` is interpreted by the caller that wires the larger
        model. This constructor mirrors the component constructors: it always
        validates the configuration and only enforces implemented-capability
        checks when the gate is enabled.
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

        _require_inputs(
            source_inputs
        )

        gate_network = (
            RelationGateNetwork.from_config(
                config=config,
                source_inputs=source_inputs,
                use_node_state=use_node_state,
                use_hazard_query=(
                    use_hazard_query
                ),
                layer_norm=layer_norm,
                relation_bias=relation_bias,
            )
        )
        activation = (
            RelationGateActivation.from_config(
                config=config
            )
        )
        prior_builder = (
            RelationPriorContributionBuilder.from_config(
                config=config,
                epsilon=prior_epsilon,
            )
            if config.use_relation_priors
            else None
        )

        return cls(
            gate_network=gate_network,
            activation=activation,
            prior_builder=prior_builder,
        )

    # ------------------------------------------------------------------
    # Public identity
    # ------------------------------------------------------------------

    @property
    def scope(self) -> str:
        return self.gate_network.scope

    @property
    def activation_name(self) -> str:
        return self.activation.activation

    @property
    def relation_names(
        self,
    ) -> tuple[str, ...]:
        return self.gate_network.relation_names

    @property
    def stable_relation_ids(
        self,
    ) -> tuple[int, ...]:
        return self.gate_network.stable_relation_ids

    @property
    def num_relations(self) -> int:
        return self.gate_network.num_relations

    @property
    def uses_relation_priors(self) -> bool:
        return self.prior_builder is not None

    @property
    def parameter_count(self) -> int:
        return sum(
            int(parameter.numel())
            for parameter
            in self.parameters()
        )

    @property
    def trainable_parameter_count(
        self,
    ) -> int:
        return sum(
            int(parameter.numel())
            for parameter
            in self.parameters()
            if parameter.requires_grad
        )

    def architecture_dict(
        self,
    ) -> dict[str, Any]:
        return {
            "schema_version": (
                RELATION_FAMILY_GATE_SCHEMA_VERSION
            ),
            "scope": self.scope,
            "activation": (
                self.activation_name
            ),
            "gate_axis": (
                "exact_compiled_relation_axis"
            ),
            "num_relations": (
                self.num_relations
            ),
            "relation_names": list(
                self.relation_names
            ),
            "stable_relation_ids": list(
                self.stable_relation_ids
            ),
            "uses_relation_priors": (
                self.uses_relation_priors
            ),
            "relation_channels_compete": False,
            "family_pooling": False,
            "edge_lookup": (
                "target_node_by_dense_relation_index"
            ),
            "gate_network": (
                self
                .gate_network
                .architecture_dict()
            ),
            "prior_builder": (
                self
                .prior_builder
                .architecture_dict()
                if self.prior_builder
                is not None
                else None
            ),
            "activation_stage": (
                self
                .activation
                .architecture_dict()
            ),
            "parameter_count": (
                self.parameter_count
            ),
            "operation_order": [
                "validate_functional_message_passing_inputs",
                "construct_or_validate_exact_relation_axis",
                "predict_neural_relation_logits",
                "resolve_optional_prior_logit_contribution",
                "combine_neural_and_prior_logits",
                "apply_independent_sigmoid",
                "gather_target_node_relation_values_onto_edges",
                "construct_relation_gate_output",
            ],
            "output_schema": (
                "RelationGateOutput"
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
                    RELATION_FAMILY_GATE_SCHEMA_VERSION
                ),
                "module": type(self).__name__,
                "gate_network_parameter_fingerprint": (
                    self
                    .gate_network
                    .parameter_fingerprint()
                ),
                "prior_builder_parameter_fingerprint": (
                    self
                    .prior_builder
                    .parameter_fingerprint()
                    if self.prior_builder
                    is not None
                    else None
                ),
                "activation_parameter_fingerprint": (
                    self
                    .activation
                    .parameter_fingerprint()
                ),
                "parameter_count": (
                    self.parameter_count
                ),
            }
        )

    def assert_finite_parameters(
        self,
    ) -> None:
        self.gate_network.assert_finite_parameters()
        self.activation.assert_finite_parameters()

        if self.prior_builder is not None:
            self.prior_builder.assert_finite_parameters()

        observed_parameter_count = sum(
            int(parameter.numel())
            for parameter
            in self.parameters()
        )

        if observed_parameter_count != (
            self.parameter_count
        ):
            raise RuntimeError(
                "Relation-family gate parameter counting is inconsistent."
            )

    # ------------------------------------------------------------------
    # Runtime contract validation
    # ------------------------------------------------------------------

    def _validate_source_inputs(
        self,
        source_inputs: FunctionalMessagePassingInputs,
    ) -> None:
        _require_inputs(
            source_inputs
        )

        if source_inputs.relation_names != (
            self.relation_names
        ):
            raise ValueError(
                "Relation-family gate relation ordering differs from "
                "source inputs."
            )

        if source_inputs.stable_relation_ids != (
            self.stable_relation_ids
        ):
            raise ValueError(
                "Relation-family gate stable relation IDs differ from "
                "source inputs."
            )

        if source_inputs.num_relations != (
            self.num_relations
        ):
            raise ValueError(
                "Relation-family gate relation count differs from source "
                "inputs."
            )

        if source_inputs.hidden_dim != (
            self.gate_network.node_state_dim
        ):
            raise ValueError(
                "Relation-family gate node-state width differs from source "
                "inputs."
            )

        if self.gate_network.use_hazard_query:
            hazard_query = (
                source_inputs.node_hazard_query
            )

            if hazard_query is None:
                raise ValueError(
                    "The configured relation-family gate requires a "
                    "node-aligned hazard query."
                )

            if hazard_query.ndim != 2:
                raise ValueError(
                    "source_inputs.node_hazard_query must have shape "
                    "[N, Q]."
                )

            if int(hazard_query.shape[1]) != (
                self.gate_network.hazard_query_dim
            ):
                raise ValueError(
                    "Relation-family gate hazard-query width differs from "
                    "source inputs."
                )

    def resolve_axis(
        self,
        source_inputs: FunctionalMessagePassingInputs,
        *,
        axis: RelationGateAxis | None = None,
    ) -> RelationGateAxis:
        """
        Construct or validate the exact compiled relation axis.
        """

        self._validate_source_inputs(
            source_inputs
        )

        resolved_axis = (
            RelationGateAxis.from_inputs(
                source_inputs=source_inputs
            )
            if axis is None
            else axis
        )

        if not isinstance(
            resolved_axis,
            RelationGateAxis,
        ):
            raise TypeError(
                "axis must be a RelationGateAxis or None."
            )

        resolved_axis.assert_matches_inputs(
            source_inputs
        )

        if resolved_axis.relation_names != (
            self.relation_names
        ):
            raise ValueError(
                "Relation-family gate network ordering differs from the "
                "runtime relation axis."
            )

        if resolved_axis.stable_relation_ids != (
            self.stable_relation_ids
        ):
            raise ValueError(
                "Relation-family gate network stable IDs differ from the "
                "runtime relation axis."
            )

        return resolved_axis

    # ------------------------------------------------------------------
    # Stage orchestration
    # ------------------------------------------------------------------

    def build_prior_contribution(
        self,
        source_inputs: FunctionalMessagePassingInputs,
        *,
        axis: RelationGateAxis,
    ) -> RelationPriorContribution | None:
        """
        Resolve an optional node-aligned prior contribution.
        """

        self._validate_source_inputs(
            source_inputs
        )

        if not isinstance(
            axis,
            RelationGateAxis,
        ):
            raise TypeError(
                "axis must be a RelationGateAxis."
            )

        axis.assert_matches_inputs(
            source_inputs
        )

        if self.prior_builder is None:
            return None

        return self.prior_builder(
            source_inputs,
            axis=axis,
        )

    def prior_regularization_weights(
        self,
        source_inputs: FunctionalMessagePassingInputs,
    ) -> torch.Tensor | None:
        """
        Return compiled node-relation regularization weights when enabled.

        This method exposes prior regularization metadata without inventing a
        training loss inside the message-passing orchestrator. A training-loss
        module may combine these weights with gate outputs later.
        """

        self._validate_source_inputs(
            source_inputs
        )

        if self.prior_builder is None:
            return None

        return self.prior_builder.regularization_weights(
            source_inputs
        )

    def lookup_edge_gate_values(
        self,
        activation_output: GateActivationOutput,
    ) -> torch.Tensor:
        """
        Gather target-node exact-relation gate values onto stored edges.

        The operation is differentiable and does not clone, detach, cast, or
        move the source gate tensor.
        """

        if not isinstance(
            activation_output,
            GateActivationOutput,
        ):
            raise TypeError(
                "activation_output must be a GateActivationOutput."
            )

        source_inputs = (
            activation_output.source_inputs
        )
        self._validate_source_inputs(
            source_inputs
        )

        if activation_output.scope != (
            self.scope
        ):
            raise ValueError(
                "Gate activation scope differs from the orchestrator scope."
            )

        if activation_output.activation != (
            self.activation_name
        ):
            raise ValueError(
                "Gate activation identity differs from the orchestrator "
                "activation."
            )

        if activation_output.axis.relation_names != (
            self.relation_names
        ):
            raise ValueError(
                "Gate activation relation ordering differs from the "
                "orchestrator."
            )

        if activation_output.axis.stable_relation_ids != (
            self.stable_relation_ids
        ):
            raise ValueError(
                "Gate activation stable relation IDs differ from the "
                "orchestrator."
            )

        edge_gate_values = (
            activation_output.gate_values[
                source_inputs.target_index,
                source_inputs.edge_relation_index,
            ]
        )

        _validate_edge_gate_values(
            edge_gate_values,
            source_inputs=source_inputs,
        )

        return edge_gate_values

    def forward(
        self,
        source_inputs: FunctionalMessagePassingInputs,
        *,
        axis: RelationGateAxis | None = None,
    ) -> RelationGateOutput:
        """
        Produce node-relation and edge-aligned hazard-conditioned gates.
        """

        self._validate_source_inputs(
            source_inputs
        )
        self.assert_finite_parameters()

        resolved_axis = self.resolve_axis(
            source_inputs,
            axis=axis,
        )
        network_output = self.gate_network(
            source_inputs,
            axis=resolved_axis,
        )
        prior_contribution = (
            self.build_prior_contribution(
                source_inputs,
                axis=resolved_axis,
            )
        )
        activation_output = self.activation(
            network_output,
            prior_contribution,
        )
        edge_gate_values = (
            self.lookup_edge_gate_values(
                activation_output
            )
        )

        return RelationGateOutput(
            gate_logits=(
                activation_output.gate_logits
            ),
            gate_values=(
                activation_output.gate_values
            ),
            edge_gate_values=(
                edge_gate_values
            ),
            source_inputs=source_inputs,
            scope=activation_output.scope,
            activation=(
                activation_output.activation
            ),
            encoder_architecture_fingerprint=(
                self.architecture_fingerprint()
            ),
            parameter_fingerprint=(
                self.parameter_fingerprint()
            ),
            prior_logit_contribution=(
                prior_contribution
                .logit_contribution
                if prior_contribution
                is not None
                else None
            ),
            regularization_terms={},
        )

    # ------------------------------------------------------------------
    # Representation
    # ------------------------------------------------------------------

    def extra_repr(self) -> str:
        return (
            f"scope={self.scope!r}, "
            f"activation={self.activation_name!r}, "
            f"num_relations={self.num_relations}, "
            f"uses_relation_priors={self.uses_relation_priors}, "
            f"parameter_count={self.parameter_count}, "
            "family_pooling=False"
        )


# Compact alias for call sites that prefer the operational stage name.
RelationGate = RelationFamilyGate


__all__ = (
    "RELATION_FAMILY_GATE_SCHEMA_VERSION",
    "RelationFamilyGate",
    "RelationGate",
)
