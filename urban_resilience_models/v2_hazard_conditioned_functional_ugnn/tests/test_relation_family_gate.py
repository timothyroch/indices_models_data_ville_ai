"""
Contract tests for the relation-family gate orchestrator.

Target repository path:
    urban_resilience_models/
        v2_hazard_conditioned_functional_ugnn/
            tests/
                test_relation_family_gate.py

Implementation under test:
    functional_message_passing/
        relation_family_gate/
            relation_family_gate.py

The gate-network, prior-integration, activation, and schema modules should have
focused suites of their own. This suite isolates the orchestration boundary and
tests:

- constructor and configuration wiring;
- exact compiled-relation-axis handling;
- optional prior dispatch;
- independent sigmoid activation flow;
- exact target-node/relation edge lookup;
- output provenance and fingerprints;
- parameter accounting and finite-parameter checks;
- relation-order, width, dtype, device, and value guards;
- empty-edge, autograd, float64, and optional CUDA behavior.
"""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
from typing import Any, Mapping

import pytest
import torch
from torch import nn

from urban_resilience_models.v2_hazard_conditioned_functional_ugnn.constants import (
    RELATION_GATE_ACTIVATION_SIGMOID,
    RELATION_GATE_SCOPE_TARGET_NODE,
)
from urban_resilience_models.v2_hazard_conditioned_functional_ugnn.functional_message_passing.relation_family_gate import (
    relation_family_gate as gate_module,
)


RELATION_FAMILY_GATE_SCHEMA_VERSION = (
    gate_module.RELATION_FAMILY_GATE_SCHEMA_VERSION
)
RelationFamilyGate = gate_module.RelationFamilyGate
RelationGate = gate_module.RelationGate


NODE_COUNT = 4
RELATION_NAMES = (
    "spatial_adjacency",
    "temporal_memory",
    "random_placebo",
)
STABLE_RELATION_IDS = (100, 200, 900)
RELATION_COUNT = len(RELATION_NAMES)
HIDDEN_DIM = 5
HAZARD_QUERY_DIM = 3
TARGET_INDEX = (1, 2, 3, 0, 1)
EDGE_RELATION_INDEX = (0, 1, 2, 2, 1)
EDGE_COUNT = len(TARGET_INDEX)


# =============================================================================
# Controlled upstream contracts
# =============================================================================


@dataclass
class FakeNodeState:
    fused_state: torch.Tensor


class FakeRelationConfig:
    def __init__(
        self,
        *,
        gate_enabled: bool = True,
        gate_scope: str = RELATION_GATE_SCOPE_TARGET_NODE,
        gate_activation: str = RELATION_GATE_ACTIVATION_SIGMOID,
        gate_hidden_dim: int = 7,
        use_relation_priors: bool = False,
        relation_prior_strength: float = 0.0,
        validation_error: Exception | None = None,
        implementation_error: Exception | None = None,
    ) -> None:
        self.gate_enabled = gate_enabled
        self.gate_scope = gate_scope
        self.gate_activation = gate_activation
        self.gate_hidden_dim = gate_hidden_dim
        self.use_relation_priors = use_relation_priors
        self.relation_prior_strength = relation_prior_strength
        self.validation_error = validation_error
        self.implementation_error = implementation_error
        self.validate_calls = 0
        self.assert_implemented_calls = 0

    def validate(self) -> None:
        self.validate_calls += 1
        if self.validation_error is not None:
            raise self.validation_error

    def assert_implemented(self) -> None:
        self.assert_implemented_calls += 1
        if self.implementation_error is not None:
            raise self.implementation_error


class FakeFunctionalMessagePassingInputs:
    def __init__(
        self,
        *,
        node_count: int = NODE_COUNT,
        relation_names: tuple[str, ...] = RELATION_NAMES,
        stable_relation_ids: tuple[int, ...] = STABLE_RELATION_IDS,
        hidden_dim: int = HIDDEN_DIM,
        hazard_query_dim: int = HAZARD_QUERY_DIM,
        target_index: torch.Tensor | None = None,
        edge_relation_index: torch.Tensor | None = None,
        dtype: torch.dtype = torch.float32,
        device: torch.device | str = "cpu",
        include_hazard_query: bool = True,
        hazard_query_rank: int = 2,
        num_nodes_override: int | None = None,
        num_relations_override: int | None = None,
        dtype_override: torch.dtype | None = None,
    ) -> None:
        resolved_device = torch.device(device)
        self.relation_names = tuple(relation_names)
        self.stable_relation_ids = tuple(stable_relation_ids)
        self.num_nodes = (
            node_count if num_nodes_override is None else num_nodes_override
        )
        self.num_relations = (
            len(relation_names)
            if num_relations_override is None
            else num_relations_override
        )
        self.hidden_dim = hidden_dim
        self.device = resolved_device
        self.dtype = dtype if dtype_override is None else dtype_override

        node_values = torch.arange(
            max(node_count, 0) * hidden_dim,
            dtype=dtype,
            device=resolved_device,
        ).reshape(max(node_count, 0), hidden_dim)
        self.node_state = FakeNodeState(fused_state=node_values)

        self.target_index = (
            torch.tensor(
                TARGET_INDEX,
                dtype=torch.long,
                device=resolved_device,
            )
            if target_index is None
            else target_index
        )
        self.edge_relation_index = (
            torch.tensor(
                EDGE_RELATION_INDEX,
                dtype=torch.long,
                device=resolved_device,
            )
            if edge_relation_index is None
            else edge_relation_index
        )
        self.num_edges = int(self.target_index.numel())

        if include_hazard_query:
            if hazard_query_rank == 2:
                self.node_hazard_query = torch.arange(
                    max(node_count, 0) * hazard_query_dim,
                    dtype=dtype,
                    device=resolved_device,
                ).reshape(max(node_count, 0), hazard_query_dim)
            elif hazard_query_rank == 1:
                self.node_hazard_query = torch.zeros(
                    max(node_count, 0),
                    dtype=dtype,
                    device=resolved_device,
                )
            else:
                self.node_hazard_query = torch.zeros(
                    (max(node_count, 0), hazard_query_dim, 1),
                    dtype=dtype,
                    device=resolved_device,
                )
        else:
            self.node_hazard_query = None


class FakeRelationGateAxis:
    from_inputs_calls = 0

    def __init__(
        self,
        *,
        source_inputs: FakeFunctionalMessagePassingInputs,
        relation_names: tuple[str, ...] | None = None,
        stable_relation_ids: tuple[int, ...] | None = None,
        mismatch_error: Exception | None = None,
    ) -> None:
        self.source_inputs = source_inputs
        self.relation_names = (
            source_inputs.relation_names
            if relation_names is None
            else tuple(relation_names)
        )
        self.stable_relation_ids = (
            source_inputs.stable_relation_ids
            if stable_relation_ids is None
            else tuple(stable_relation_ids)
        )
        self.num_relations = len(self.relation_names)
        self.mismatch_error = mismatch_error
        self.assert_matches_calls = 0

    @classmethod
    def from_inputs(
        cls,
        *,
        source_inputs: FakeFunctionalMessagePassingInputs,
    ) -> "FakeRelationGateAxis":
        cls.from_inputs_calls += 1
        return cls(source_inputs=source_inputs)

    def assert_matches_inputs(
        self,
        source_inputs: FakeFunctionalMessagePassingInputs,
    ) -> None:
        self.assert_matches_calls += 1
        if self.mismatch_error is not None:
            raise self.mismatch_error
        if source_inputs is not self.source_inputs:
            raise ValueError("Relation-gate axis references different inputs.")
        if self.relation_names != source_inputs.relation_names:
            raise ValueError("Relation-gate axis ordering differs from inputs.")
        if self.stable_relation_ids != source_inputs.stable_relation_ids:
            raise ValueError("Relation-gate stable IDs differ from inputs.")

    def fingerprint(self) -> str:
        return _fingerprint(
            {
                "relation_names": list(self.relation_names),
                "stable_relation_ids": list(self.stable_relation_ids),
            }
        )


@dataclass
class FakeGateNetworkOutput:
    logits: torch.Tensor
    source_inputs: FakeFunctionalMessagePassingInputs
    axis: FakeRelationGateAxis
    scope: str


@dataclass
class FakeRelationPriorContribution:
    logit_contribution: torch.Tensor
    source_inputs: FakeFunctionalMessagePassingInputs
    axis: FakeRelationGateAxis


class FakeGateActivationOutput:
    def __init__(
        self,
        *,
        gate_logits: torch.Tensor,
        gate_values: torch.Tensor,
        source_inputs: FakeFunctionalMessagePassingInputs,
        axis: FakeRelationGateAxis,
        scope: str = RELATION_GATE_SCOPE_TARGET_NODE,
        activation: str = RELATION_GATE_ACTIVATION_SIGMOID,
    ) -> None:
        self.gate_logits = gate_logits
        self.gate_values = gate_values
        self.source_inputs = source_inputs
        self.axis = axis
        self.scope = scope
        self.activation = activation


class FakeRelationGateOutput:
    def __init__(self, **values: Any) -> None:
        for name, value in values.items():
            setattr(self, name, value)


class FakeRelationGateNetwork(nn.Module):
    last_from_config: dict[str, Any] | None = None

    def __init__(
        self,
        *,
        relation_names: tuple[str, ...] = RELATION_NAMES,
        stable_relation_ids: tuple[int, ...] = STABLE_RELATION_IDS,
        node_state_dim: int = HIDDEN_DIM,
        hazard_query_dim: int = HAZARD_QUERY_DIM,
        scope: str = RELATION_GATE_SCOPE_TARGET_NODE,
        use_hazard_query: bool = True,
    ) -> None:
        super().__init__()
        self.relation_names = tuple(relation_names)
        self.stable_relation_ids = tuple(stable_relation_ids)
        self.node_state_dim = node_state_dim
        self.hazard_query_dim = hazard_query_dim
        self.scope = scope
        self.use_hazard_query = use_hazard_query
        self.weight = nn.Parameter(torch.tensor(0.25))
        self.forward_calls = 0
        self.finite_calls = 0
        self.last_axis: FakeRelationGateAxis | None = None

    @classmethod
    def from_config(
        cls,
        *,
        config: FakeRelationConfig,
        source_inputs: FakeFunctionalMessagePassingInputs,
        use_node_state: bool = True,
        use_hazard_query: bool = True,
        layer_norm: bool = True,
        relation_bias: bool = True,
    ) -> "FakeRelationGateNetwork":
        cls.last_from_config = {
            "config": config,
            "source_inputs": source_inputs,
            "use_node_state": use_node_state,
            "use_hazard_query": use_hazard_query,
            "layer_norm": layer_norm,
            "relation_bias": relation_bias,
        }
        return cls(
            relation_names=source_inputs.relation_names,
            stable_relation_ids=source_inputs.stable_relation_ids,
            node_state_dim=source_inputs.hidden_dim,
            hazard_query_dim=(
                int(source_inputs.node_hazard_query.shape[1])
                if source_inputs.node_hazard_query is not None
                and source_inputs.node_hazard_query.ndim == 2
                else 1
            ),
            scope=config.gate_scope,
            use_hazard_query=use_hazard_query,
        ).to(device=source_inputs.device, dtype=source_inputs.dtype)

    @property
    def num_relations(self) -> int:
        return len(self.relation_names)

    def architecture_dict(self) -> dict[str, Any]:
        return {
            "module": type(self).__name__,
            "relation_names": list(self.relation_names),
            "stable_relation_ids": list(self.stable_relation_ids),
            "scope": self.scope,
            "node_state_dim": self.node_state_dim,
            "hazard_query_dim": self.hazard_query_dim,
            "use_hazard_query": self.use_hazard_query,
            "parameter_count": sum(p.numel() for p in self.parameters()),
        }

    def parameter_fingerprint(self) -> str:
        return _fingerprint(
            {
                "weight": self.weight.detach().cpu().item(),
            }
        )

    def assert_finite_parameters(self) -> None:
        self.finite_calls += 1
        if not bool(torch.isfinite(self.weight).all().item()):
            raise FloatingPointError("Fake network parameter is non-finite.")

    def forward(
        self,
        source_inputs: FakeFunctionalMessagePassingInputs,
        *,
        axis: FakeRelationGateAxis | None = None,
    ) -> FakeGateNetworkOutput:
        self.forward_calls += 1
        if axis is None:
            axis = FakeRelationGateAxis.from_inputs(source_inputs=source_inputs)
        self.last_axis = axis

        base = torch.arange(
            source_inputs.num_nodes * source_inputs.num_relations,
            dtype=source_inputs.dtype,
            device=source_inputs.device,
        ).reshape(source_inputs.num_nodes, source_inputs.num_relations) / 10.0
        logits = base + self.weight.to(
            device=source_inputs.device,
            dtype=source_inputs.dtype,
        )
        return FakeGateNetworkOutput(
            logits=logits,
            source_inputs=source_inputs,
            axis=axis,
            scope=self.scope,
        )


class FakeRelationGateActivation(nn.Module):
    last_from_config: dict[str, Any] | None = None

    def __init__(
        self,
        *,
        activation: str = RELATION_GATE_ACTIVATION_SIGMOID,
    ) -> None:
        super().__init__()
        self.activation = activation
        self.forward_calls = 0
        self.finite_calls = 0
        self.last_network_output: FakeGateNetworkOutput | None = None
        self.last_prior: FakeRelationPriorContribution | None = None

    @classmethod
    def from_config(
        cls,
        *,
        config: FakeRelationConfig,
    ) -> "FakeRelationGateActivation":
        cls.last_from_config = {"config": config}
        return cls(activation=config.gate_activation)

    def architecture_dict(self) -> dict[str, Any]:
        return {
            "module": type(self).__name__,
            "activation": self.activation,
            "parameter_count": 0,
        }

    def parameter_fingerprint(self) -> str:
        return _fingerprint({"activation": self.activation, "parameters": []})

    def assert_finite_parameters(self) -> None:
        self.finite_calls += 1

    def forward(
        self,
        source_network_output: FakeGateNetworkOutput,
        prior_contribution: FakeRelationPriorContribution | None = None,
    ) -> FakeGateActivationOutput:
        self.forward_calls += 1
        self.last_network_output = source_network_output
        self.last_prior = prior_contribution
        logits = source_network_output.logits
        if prior_contribution is not None:
            logits = logits + prior_contribution.logit_contribution
        return FakeGateActivationOutput(
            gate_logits=logits,
            gate_values=torch.sigmoid(logits),
            source_inputs=source_network_output.source_inputs,
            axis=source_network_output.axis,
            scope=source_network_output.scope,
            activation=self.activation,
        )


class FakeRelationPriorContributionBuilder(nn.Module):
    last_from_config: dict[str, Any] | None = None

    def __init__(
        self,
        *,
        strength: float = 0.5,
        epsilon: float = 1e-4,
    ) -> None:
        super().__init__()
        self.strength = float(strength)
        self.epsilon = float(epsilon)
        self.forward_calls = 0
        self.regularization_calls = 0
        self.finite_calls = 0
        self.last_axis: FakeRelationGateAxis | None = None

    @classmethod
    def from_config(
        cls,
        *,
        config: FakeRelationConfig,
        epsilon: float = 1e-4,
    ) -> "FakeRelationPriorContributionBuilder":
        cls.last_from_config = {
            "config": config,
            "epsilon": epsilon,
        }
        return cls(
            strength=config.relation_prior_strength,
            epsilon=epsilon,
        )

    def architecture_dict(self) -> dict[str, Any]:
        return {
            "module": type(self).__name__,
            "strength": self.strength,
            "epsilon": self.epsilon,
            "parameter_count": 0,
        }

    def parameter_fingerprint(self) -> str:
        return _fingerprint(
            {
                "strength": self.strength,
                "epsilon": self.epsilon,
                "parameters": [],
            }
        )

    def assert_finite_parameters(self) -> None:
        self.finite_calls += 1

    def forward(
        self,
        source_inputs: FakeFunctionalMessagePassingInputs,
        *,
        axis: FakeRelationGateAxis | None = None,
    ) -> FakeRelationPriorContribution:
        self.forward_calls += 1
        if axis is None:
            axis = FakeRelationGateAxis.from_inputs(source_inputs=source_inputs)
        self.last_axis = axis
        contribution = torch.full(
            (source_inputs.num_nodes, source_inputs.num_relations),
            self.strength,
            dtype=source_inputs.dtype,
            device=source_inputs.device,
        )
        return FakeRelationPriorContribution(
            logit_contribution=contribution,
            source_inputs=source_inputs,
            axis=axis,
        )

    def regularization_weights(
        self,
        source_inputs: FakeFunctionalMessagePassingInputs,
    ) -> torch.Tensor:
        self.regularization_calls += 1
        return torch.full(
            (source_inputs.num_nodes, source_inputs.num_relations),
            0.75,
            dtype=source_inputs.dtype,
            device=source_inputs.device,
        )


# =============================================================================
# Shared helpers and patches
# =============================================================================


def _canonical_json(payload: Mapping[str, Any]) -> str:
    return json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )


def _fingerprint(payload: Mapping[str, Any]) -> str:
    return sha256(_canonical_json(payload).encode("utf-8")).hexdigest()


@pytest.fixture(autouse=True)
def _patch_orchestrator_contracts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    FakeRelationGateAxis.from_inputs_calls = 0
    FakeRelationGateNetwork.last_from_config = None
    FakeRelationGateActivation.last_from_config = None
    FakeRelationPriorContributionBuilder.last_from_config = None

    monkeypatch.setattr(gate_module, "RelationConfig", FakeRelationConfig)
    monkeypatch.setattr(
        gate_module,
        "FunctionalMessagePassingInputs",
        FakeFunctionalMessagePassingInputs,
    )
    monkeypatch.setattr(gate_module, "RelationGateAxis", FakeRelationGateAxis)
    monkeypatch.setattr(
        gate_module,
        "RelationGateNetwork",
        FakeRelationGateNetwork,
    )
    monkeypatch.setattr(
        gate_module,
        "RelationGateActivation",
        FakeRelationGateActivation,
    )
    monkeypatch.setattr(
        gate_module,
        "RelationPriorContributionBuilder",
        FakeRelationPriorContributionBuilder,
    )
    monkeypatch.setattr(
        gate_module,
        "RelationPriorContribution",
        FakeRelationPriorContribution,
    )
    monkeypatch.setattr(
        gate_module,
        "GateActivationOutput",
        FakeGateActivationOutput,
    )
    monkeypatch.setattr(
        gate_module,
        "RelationGateOutput",
        FakeRelationGateOutput,
    )


def _inputs(**kwargs: Any) -> FakeFunctionalMessagePassingInputs:
    return FakeFunctionalMessagePassingInputs(**kwargs)


def _gate(
    *,
    priors: bool = False,
    relation_names: tuple[str, ...] = RELATION_NAMES,
    stable_relation_ids: tuple[int, ...] = STABLE_RELATION_IDS,
    node_state_dim: int = HIDDEN_DIM,
    hazard_query_dim: int = HAZARD_QUERY_DIM,
    scope: str = RELATION_GATE_SCOPE_TARGET_NODE,
    activation: str = RELATION_GATE_ACTIVATION_SIGMOID,
    use_hazard_query: bool = True,
) -> RelationFamilyGate:
    return RelationFamilyGate(
        gate_network=FakeRelationGateNetwork(
            relation_names=relation_names,
            stable_relation_ids=stable_relation_ids,
            node_state_dim=node_state_dim,
            hazard_query_dim=hazard_query_dim,
            scope=scope,
            use_hazard_query=use_hazard_query,
        ),
        activation=FakeRelationGateActivation(
            activation=activation,
        ),
        prior_builder=(
            FakeRelationPriorContributionBuilder(strength=0.5)
            if priors
            else None
        ),
    )


def _activation_output(
    *,
    source_inputs: FakeFunctionalMessagePassingInputs | None = None,
    gate_values: torch.Tensor | None = None,
    scope: str = RELATION_GATE_SCOPE_TARGET_NODE,
    activation: str = RELATION_GATE_ACTIVATION_SIGMOID,
    axis: FakeRelationGateAxis | None = None,
) -> FakeGateActivationOutput:
    inputs = _inputs() if source_inputs is None else source_inputs
    resolved_axis = (
        FakeRelationGateAxis(source_inputs=inputs)
        if axis is None
        else axis
    )
    values = (
        torch.arange(
            inputs.num_nodes * inputs.num_relations,
            dtype=inputs.dtype,
            device=inputs.device,
        ).reshape(inputs.num_nodes, inputs.num_relations)
        / float(inputs.num_nodes * inputs.num_relations)
        if gate_values is None
        else gate_values
    )
    return FakeGateActivationOutput(
        gate_logits=torch.zeros_like(values),
        gate_values=values,
        source_inputs=inputs,
        axis=resolved_axis,
        scope=scope,
        activation=activation,
    )


# =============================================================================
# Public identity and constructor
# =============================================================================


def test_schema_version_is_nonempty() -> None:
    assert isinstance(RELATION_FAMILY_GATE_SCHEMA_VERSION, str)
    assert RELATION_FAMILY_GATE_SCHEMA_VERSION.strip()


def test_operational_alias_points_to_primary_class() -> None:
    assert RelationGate is RelationFamilyGate


def test_primary_class_is_torch_module() -> None:
    assert issubclass(RelationFamilyGate, nn.Module)


def test_constructor_preserves_components() -> None:
    network = FakeRelationGateNetwork()
    activation = FakeRelationGateActivation()
    prior = FakeRelationPriorContributionBuilder()

    gate = RelationFamilyGate(
        gate_network=network,
        activation=activation,
        prior_builder=prior,
    )

    assert gate.gate_network is network
    assert gate.activation is activation
    assert gate.prior_builder is prior
    assert gate.uses_relation_priors


@pytest.mark.parametrize("value", (None, object(), nn.Identity()))
def test_constructor_rejects_invalid_network(value: Any) -> None:
    with pytest.raises(TypeError, match="gate_network"):
        RelationFamilyGate(
            gate_network=value,
            activation=FakeRelationGateActivation(),
        )


@pytest.mark.parametrize("value", (None, object(), nn.Identity()))
def test_constructor_rejects_invalid_activation(value: Any) -> None:
    with pytest.raises(TypeError, match="activation"):
        RelationFamilyGate(
            gate_network=FakeRelationGateNetwork(),
            activation=value,
        )


def test_constructor_rejects_invalid_prior_builder() -> None:
    with pytest.raises(TypeError, match="prior_builder"):
        RelationFamilyGate(
            gate_network=FakeRelationGateNetwork(),
            activation=FakeRelationGateActivation(),
            prior_builder=object(),
        )


def test_constructor_rejects_non_target_node_scope() -> None:
    with pytest.raises(ValueError, match="target_node"):
        _gate(scope="graph")


def test_constructor_rejects_non_sigmoid_activation() -> None:
    with pytest.raises(ValueError, match="sigmoid"):
        _gate(activation="softmax")


# =============================================================================
# Construction from configuration
# =============================================================================


def test_from_config_constructs_all_required_stages() -> None:
    config = FakeRelationConfig(
        use_relation_priors=True,
        relation_prior_strength=0.4,
    )
    inputs = _inputs()

    gate = RelationFamilyGate.from_config(
        config=config,
        source_inputs=inputs,
        use_node_state=False,
        use_hazard_query=True,
        layer_norm=False,
        relation_bias=False,
        prior_epsilon=2e-4,
    )

    assert isinstance(gate.gate_network, FakeRelationGateNetwork)
    assert isinstance(gate.activation, FakeRelationGateActivation)
    assert isinstance(gate.prior_builder, FakeRelationPriorContributionBuilder)
    assert gate.prior_builder.strength == pytest.approx(0.4)
    assert gate.prior_builder.epsilon == pytest.approx(2e-4)
    assert config.validate_calls == 1
    assert config.assert_implemented_calls == 1

    network_call = FakeRelationGateNetwork.last_from_config
    assert network_call is not None
    assert network_call["config"] is config
    assert network_call["source_inputs"] is inputs
    assert network_call["use_node_state"] is False
    assert network_call["use_hazard_query"] is True
    assert network_call["layer_norm"] is False
    assert network_call["relation_bias"] is False

    prior_call = FakeRelationPriorContributionBuilder.last_from_config
    assert prior_call == {"config": config, "epsilon": 2e-4}


def test_from_config_omits_prior_stage_when_disabled() -> None:
    config = FakeRelationConfig(use_relation_priors=False)
    gate = RelationFamilyGate.from_config(
        config=config,
        source_inputs=_inputs(),
    )

    assert gate.prior_builder is None
    assert not gate.uses_relation_priors
    assert FakeRelationPriorContributionBuilder.last_from_config is None


def test_from_config_does_not_assert_implementation_for_disabled_gate() -> None:
    config = FakeRelationConfig(
        gate_enabled=False,
        implementation_error=RuntimeError("should not run"),
    )

    RelationFamilyGate.from_config(
        config=config,
        source_inputs=_inputs(),
    )

    assert config.validate_calls == 1
    assert config.assert_implemented_calls == 0


def test_from_config_propagates_validation_failure() -> None:
    config = FakeRelationConfig(validation_error=ValueError("invalid config"))

    with pytest.raises(ValueError, match="invalid config"):
        RelationFamilyGate.from_config(
            config=config,
            source_inputs=_inputs(),
        )


def test_from_config_propagates_implementation_failure() -> None:
    config = FakeRelationConfig(
        implementation_error=NotImplementedError("unsupported gate"),
    )

    with pytest.raises(NotImplementedError, match="unsupported gate"):
        RelationFamilyGate.from_config(
            config=config,
            source_inputs=_inputs(),
        )


def test_from_config_rejects_wrong_config_type() -> None:
    with pytest.raises(TypeError, match="RelationConfig"):
        RelationFamilyGate.from_config(
            config=object(),
            source_inputs=_inputs(),
        )


# =============================================================================
# Public metadata, parameters, and fingerprints
# =============================================================================


def test_public_properties_match_network_contract() -> None:
    gate = _gate(priors=True)

    assert gate.scope == RELATION_GATE_SCOPE_TARGET_NODE
    assert gate.activation_name == RELATION_GATE_ACTIVATION_SIGMOID
    assert gate.relation_names == RELATION_NAMES
    assert gate.stable_relation_ids == STABLE_RELATION_IDS
    assert gate.num_relations == RELATION_COUNT
    assert gate.uses_relation_priors


def test_parameter_counts_equal_registered_submodule_parameters() -> None:
    gate = _gate(priors=True)

    expected_total = sum(int(p.numel()) for p in gate.parameters())
    expected_trainable = sum(
        int(p.numel()) for p in gate.parameters() if p.requires_grad
    )

    assert gate.parameter_count == expected_total == 1
    assert gate.trainable_parameter_count == expected_trainable == 1


def test_architecture_dict_is_complete_and_json_serializable() -> None:
    gate = _gate(priors=True)
    architecture = gate.architecture_dict()

    assert architecture["schema_version"] == RELATION_FAMILY_GATE_SCHEMA_VERSION
    assert architecture["scope"] == RELATION_GATE_SCOPE_TARGET_NODE
    assert architecture["activation"] == RELATION_GATE_ACTIVATION_SIGMOID
    assert architecture["gate_axis"] == "exact_compiled_relation_axis"
    assert architecture["relation_names"] == list(RELATION_NAMES)
    assert architecture["stable_relation_ids"] == list(STABLE_RELATION_IDS)
    assert architecture["uses_relation_priors"] is True
    assert architecture["relation_channels_compete"] is False
    assert architecture["family_pooling"] is False
    assert architecture["edge_lookup"] == "target_node_by_dense_relation_index"
    assert architecture["output_schema"] == "RelationGateOutput"
    assert architecture["parameter_count"] == gate.parameter_count
    assert isinstance(architecture["gate_network"], dict)
    assert isinstance(architecture["prior_builder"], dict)
    assert isinstance(architecture["activation_stage"], dict)
    json.dumps(architecture, allow_nan=False)


def test_architecture_fingerprint_is_stable_and_architecture_sensitive() -> None:
    first = _gate(priors=False)
    second = _gate(priors=False)
    with_priors = _gate(priors=True)

    assert first.architecture_fingerprint() == second.architecture_fingerprint()
    assert first.architecture_fingerprint() != with_priors.architecture_fingerprint()


def test_parameter_fingerprint_changes_when_network_parameter_changes() -> None:
    gate = _gate()
    before = gate.parameter_fingerprint()

    with torch.no_grad():
        gate.gate_network.weight.add_(1.0)

    after = gate.parameter_fingerprint()
    assert before != after


def test_assert_finite_parameters_delegates_to_all_stages() -> None:
    gate = _gate(priors=True)

    gate.assert_finite_parameters()

    assert gate.gate_network.finite_calls == 1
    assert gate.activation.finite_calls == 1
    assert gate.prior_builder is not None
    assert gate.prior_builder.finite_calls == 1


def test_assert_finite_parameters_rejects_nonfinite_network_parameter() -> None:
    gate = _gate()
    with torch.no_grad():
        gate.gate_network.weight.fill_(float("nan"))

    with pytest.raises(FloatingPointError, match="non-finite"):
        gate.assert_finite_parameters()


def test_extra_repr_reports_core_architecture() -> None:
    text = _gate(priors=True).extra_repr()

    assert "target_node" in text
    assert "sigmoid" in text
    assert f"num_relations={RELATION_COUNT}" in text
    assert "uses_relation_priors=True" in text
    assert "family_pooling=False" in text


# =============================================================================
# Source-input and axis validation
# =============================================================================


def test_forward_rejects_wrong_input_type() -> None:
    with pytest.raises(TypeError, match="FunctionalMessagePassingInputs"):
        _gate()(object())


@pytest.mark.parametrize(
    ("kwargs", "message"),
    (
        ({"num_nodes_override": 0}, "at least one node"),
        ({"num_relations_override": 0}, "at least one relation"),
        ({"dtype_override": torch.long}, "floating-point"),
    ),
)
def test_forward_rejects_invalid_basic_input_contract(
    kwargs: dict[str, Any],
    message: str,
) -> None:
    with pytest.raises((ValueError, TypeError), match=message):
        _gate()(_inputs(**kwargs))


def test_forward_rejects_relation_order_mismatch() -> None:
    inputs = _inputs(relation_names=tuple(reversed(RELATION_NAMES)))

    with pytest.raises(ValueError, match="relation ordering"):
        _gate()(inputs)


def test_forward_rejects_stable_relation_id_mismatch() -> None:
    inputs = _inputs(stable_relation_ids=(101, 201, 901))

    with pytest.raises(ValueError, match="stable relation IDs"):
        _gate()(inputs)


def test_forward_rejects_relation_count_mismatch() -> None:
    inputs = _inputs(num_relations_override=RELATION_COUNT + 1)

    with pytest.raises(ValueError, match="relation count"):
        _gate()(inputs)


def test_forward_rejects_node_state_width_mismatch() -> None:
    inputs = _inputs(hidden_dim=HIDDEN_DIM + 1)

    with pytest.raises(ValueError, match="node-state width"):
        _gate()(inputs)


def test_forward_requires_hazard_query_when_network_uses_it() -> None:
    inputs = _inputs(include_hazard_query=False)

    with pytest.raises(ValueError, match="requires a node-aligned hazard query"):
        _gate(use_hazard_query=True)(inputs)


def test_forward_rejects_hazard_query_rank_mismatch() -> None:
    inputs = _inputs(hazard_query_rank=1)

    with pytest.raises(ValueError, match=r"\[N, Q\]"):
        _gate()(inputs)


def test_forward_rejects_hazard_query_width_mismatch() -> None:
    inputs = _inputs(hazard_query_dim=HAZARD_QUERY_DIM + 1)

    with pytest.raises(ValueError, match="hazard-query width"):
        _gate()(inputs)


def test_gate_can_be_hazard_query_blind_when_explicitly_configured() -> None:
    inputs = _inputs(include_hazard_query=False)
    output = _gate(use_hazard_query=False)(inputs)

    assert output.source_inputs is inputs


def test_resolve_axis_constructs_axis_when_absent() -> None:
    inputs = _inputs()
    gate = _gate()

    axis = gate.resolve_axis(inputs)

    assert isinstance(axis, FakeRelationGateAxis)
    assert axis.source_inputs is inputs
    assert FakeRelationGateAxis.from_inputs_calls == 1
    assert axis.assert_matches_calls == 1


def test_resolve_axis_reuses_and_validates_supplied_axis() -> None:
    inputs = _inputs()
    axis = FakeRelationGateAxis(source_inputs=inputs)

    resolved = _gate().resolve_axis(inputs, axis=axis)

    assert resolved is axis
    assert FakeRelationGateAxis.from_inputs_calls == 0
    assert axis.assert_matches_calls == 1


def test_resolve_axis_rejects_wrong_axis_type() -> None:
    with pytest.raises(TypeError, match="RelationGateAxis"):
        _gate().resolve_axis(_inputs(), axis=object())


def test_resolve_axis_propagates_axis_alignment_failure() -> None:
    inputs = _inputs()
    axis = FakeRelationGateAxis(
        source_inputs=inputs,
        mismatch_error=ValueError("axis mismatch"),
    )

    with pytest.raises(ValueError, match="axis mismatch"):
        _gate().resolve_axis(inputs, axis=axis)


# =============================================================================
# Prior stage and regularization metadata
# =============================================================================


def test_build_prior_contribution_returns_none_when_disabled() -> None:
    inputs = _inputs()
    axis = FakeRelationGateAxis(source_inputs=inputs)

    assert _gate(priors=False).build_prior_contribution(inputs, axis=axis) is None


def test_build_prior_contribution_delegates_with_same_axis() -> None:
    inputs = _inputs()
    axis = FakeRelationGateAxis(source_inputs=inputs)
    gate = _gate(priors=True)

    contribution = gate.build_prior_contribution(inputs, axis=axis)

    assert isinstance(contribution, FakeRelationPriorContribution)
    assert contribution.source_inputs is inputs
    assert contribution.axis is axis
    assert gate.prior_builder is not None
    assert gate.prior_builder.forward_calls == 1
    assert gate.prior_builder.last_axis is axis


def test_build_prior_contribution_rejects_wrong_axis_type() -> None:
    with pytest.raises(TypeError, match="RelationGateAxis"):
        _gate(priors=True).build_prior_contribution(_inputs(), axis=object())


def test_prior_regularization_weights_returns_none_when_disabled() -> None:
    assert _gate().prior_regularization_weights(_inputs()) is None


def test_prior_regularization_weights_delegates_without_defining_loss() -> None:
    inputs = _inputs(dtype=torch.float64)
    gate = _gate(priors=True).to(dtype=torch.float64)

    weights = gate.prior_regularization_weights(inputs)

    assert weights is not None
    assert weights.shape == (NODE_COUNT, RELATION_COUNT)
    assert weights.dtype == torch.float64
    assert weights.device == inputs.device
    assert torch.equal(weights, torch.full_like(weights, 0.75))
    assert gate.prior_builder is not None
    assert gate.prior_builder.regularization_calls == 1


# =============================================================================
# Exact edge lookup
# =============================================================================


def test_lookup_edge_gate_values_uses_target_node_and_dense_relation() -> None:
    inputs = _inputs()
    activation_output = _activation_output(source_inputs=inputs)

    observed = _gate().lookup_edge_gate_values(activation_output)
    expected = activation_output.gate_values[
        inputs.target_index,
        inputs.edge_relation_index,
    ]

    assert torch.equal(observed, expected)


def test_lookup_edge_gate_values_preserves_gradient() -> None:
    inputs = _inputs()
    values = torch.rand(
        NODE_COUNT,
        RELATION_COUNT,
        dtype=inputs.dtype,
        requires_grad=True,
    )
    activation_output = _activation_output(
        source_inputs=inputs,
        gate_values=values,
    )

    edge_values = _gate().lookup_edge_gate_values(activation_output)
    edge_values.sum().backward()

    assert values.grad is not None
    expected_counts = torch.zeros_like(values)
    expected_counts.index_put_(
        (inputs.target_index, inputs.edge_relation_index),
        torch.ones(EDGE_COUNT, dtype=values.dtype),
        accumulate=True,
    )
    assert torch.equal(values.grad, expected_counts)


def test_lookup_edge_gate_values_supports_empty_edges() -> None:
    empty = torch.empty(0, dtype=torch.long)
    inputs = _inputs(target_index=empty, edge_relation_index=empty)

    observed = _gate().lookup_edge_gate_values(
        _activation_output(source_inputs=inputs)
    )

    assert observed.shape == (0,)
    assert observed.dtype == inputs.dtype


def test_lookup_rejects_wrong_activation_output_type() -> None:
    with pytest.raises(TypeError, match="GateActivationOutput"):
        _gate().lookup_edge_gate_values(object())


def test_lookup_rejects_scope_mismatch() -> None:
    with pytest.raises(ValueError, match="scope differs"):
        _gate().lookup_edge_gate_values(
            _activation_output(scope="graph")
        )


def test_lookup_rejects_activation_identity_mismatch() -> None:
    with pytest.raises(ValueError, match="activation identity differs"):
        _gate().lookup_edge_gate_values(
            _activation_output(activation="softmax")
        )


def test_lookup_rejects_relation_order_mismatch() -> None:
    inputs = _inputs()
    axis = FakeRelationGateAxis(
        source_inputs=inputs,
        relation_names=tuple(reversed(RELATION_NAMES)),
    )
    output = _activation_output(source_inputs=inputs, axis=axis)

    with pytest.raises(ValueError, match="relation ordering differs"):
        _gate().lookup_edge_gate_values(output)


def test_lookup_rejects_nonfinite_values() -> None:
    inputs = _inputs()
    values = torch.zeros(NODE_COUNT, RELATION_COUNT)
    values[1, 0] = float("nan")

    with pytest.raises(FloatingPointError, match="NaN or infinity"):
        _gate().lookup_edge_gate_values(
            _activation_output(source_inputs=inputs, gate_values=values)
        )


def test_lookup_rejects_values_outside_sigmoid_range() -> None:
    inputs = _inputs()
    values = torch.zeros(NODE_COUNT, RELATION_COUNT)
    values[1, 0] = 1.1

    with pytest.raises(FloatingPointError, match=r"must lie in \[0, 1\]"):
        _gate().lookup_edge_gate_values(
            _activation_output(source_inputs=inputs, gate_values=values)
        )


# =============================================================================
# End-to-end orchestration boundary
# =============================================================================


def test_forward_without_priors_runs_each_stage_once() -> None:
    inputs = _inputs()
    gate = _gate(priors=False)

    output = gate(inputs)

    assert isinstance(output, FakeRelationGateOutput)
    assert gate.gate_network.forward_calls == 1
    assert gate.activation.forward_calls == 1
    assert gate.activation.last_prior is None
    assert output.prior_logit_contribution is None
    assert output.regularization_terms == {}


def test_forward_with_priors_combines_logits_before_sigmoid() -> None:
    inputs = _inputs()
    gate = _gate(priors=True)

    output = gate(inputs)
    network_output = gate.activation.last_network_output
    prior = gate.activation.last_prior

    assert network_output is not None
    assert prior is not None
    expected_logits = network_output.logits + prior.logit_contribution
    expected_values = torch.sigmoid(expected_logits)
    expected_edges = expected_values[
        inputs.target_index,
        inputs.edge_relation_index,
    ]

    assert torch.allclose(output.gate_logits, expected_logits)
    assert torch.allclose(output.gate_values, expected_values)
    assert torch.allclose(output.edge_gate_values, expected_edges)
    assert output.prior_logit_contribution is prior.logit_contribution
    assert gate.prior_builder is not None
    assert gate.prior_builder.forward_calls == 1


def test_forward_preserves_source_contract_and_public_metadata() -> None:
    inputs = _inputs()
    gate = _gate(priors=True)

    output = gate(inputs)

    assert output.source_inputs is inputs
    assert output.scope == RELATION_GATE_SCOPE_TARGET_NODE
    assert output.activation == RELATION_GATE_ACTIVATION_SIGMOID
    assert output.encoder_architecture_fingerprint == gate.architecture_fingerprint()
    assert output.parameter_fingerprint == gate.parameter_fingerprint()


def test_forward_passes_one_resolved_axis_to_network_and_prior() -> None:
    inputs = _inputs()
    gate = _gate(priors=True)

    gate(inputs)

    assert gate.gate_network.last_axis is not None
    assert gate.prior_builder is not None
    assert gate.prior_builder.last_axis is gate.gate_network.last_axis
    assert FakeRelationGateAxis.from_inputs_calls == 1


def test_forward_reuses_explicit_axis_for_all_stages() -> None:
    inputs = _inputs()
    axis = FakeRelationGateAxis(source_inputs=inputs)
    gate = _gate(priors=True)

    gate(inputs, axis=axis)

    assert gate.gate_network.last_axis is axis
    assert gate.prior_builder is not None
    assert gate.prior_builder.last_axis is axis
    assert FakeRelationGateAxis.from_inputs_calls == 0


def test_forward_is_differentiable_through_network_parameter() -> None:
    inputs = _inputs()
    gate = _gate(priors=True)

    output = gate(inputs)
    loss = output.edge_gate_values.sum() + output.gate_values.mean()
    loss.backward()

    gradient = gate.gate_network.weight.grad
    assert gradient is not None
    assert bool(torch.isfinite(gradient).all().item())
    assert float(gradient.abs().item()) > 0.0


def test_forward_supports_float64_without_silent_cast() -> None:
    inputs = _inputs(dtype=torch.float64)
    gate = _gate(priors=True).to(dtype=torch.float64)

    output = gate(inputs)

    assert output.gate_logits.dtype == torch.float64
    assert output.gate_values.dtype == torch.float64
    assert output.edge_gate_values.dtype == torch.float64


def test_forward_supports_empty_edges() -> None:
    empty = torch.empty(0, dtype=torch.long)
    inputs = _inputs(target_index=empty, edge_relation_index=empty)

    output = _gate(priors=True)(inputs)

    assert output.gate_logits.shape == (NODE_COUNT, RELATION_COUNT)
    assert output.gate_values.shape == (NODE_COUNT, RELATION_COUNT)
    assert output.edge_gate_values.shape == (0,)


@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA is unavailable.",
)
def test_cuda_results_match_cpu() -> None:
    cpu_inputs = _inputs(device="cpu")
    cuda_inputs = _inputs(device="cuda")
    cpu_gate = _gate(priors=True)
    cuda_gate = _gate(priors=True).to("cuda")
    cuda_gate.load_state_dict(cpu_gate.state_dict())

    cpu_output = cpu_gate(cpu_inputs)
    cuda_output = cuda_gate(cuda_inputs)

    assert cuda_output.gate_logits.device.type == "cuda"
    assert cuda_output.gate_values.device.type == "cuda"
    assert cuda_output.edge_gate_values.device.type == "cuda"
    assert torch.allclose(
        cpu_output.gate_logits,
        cuda_output.gate_logits.cpu(),
    )
    assert torch.allclose(
        cpu_output.gate_values,
        cuda_output.gate_values.cpu(),
    )
    assert torch.allclose(
        cpu_output.edge_gate_values,
        cuda_output.edge_gate_values.cpu(),
    )
