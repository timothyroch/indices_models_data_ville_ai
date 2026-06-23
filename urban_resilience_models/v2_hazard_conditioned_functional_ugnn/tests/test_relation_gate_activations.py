"""
Contract tests for relation-gate activation dispatch.

Target repository path:
    urban_resilience_models/
        v2_hazard_conditioned_functional_ugnn/
            tests/
                test_relation_gate_activations.py

Implementation under test:
    functional_message_passing/
        relation_family_gate/
            activations.py

The bounded V2.0 activation contract is independent sigmoid activation over
the exact compiled relation axis:

    combined_logits = neural_logits + optional_prior_logit_contribution
    gate_values = sigmoid(combined_logits)

Relation channels do not compete through softmax.

This suite covers:

- public exports and schema identity;
- canonical versus implemented activation dispatch;
- direct functional sigmoid behavior;
- construction from ``RelationConfig``;
- parameter-free architecture and fingerprints;
- no-prior and prior logit composition;
- exact metadata-preserving ``GateActivationOutput`` construction;
- source-input, relation-axis, shape, dtype, and device compatibility;
- internal activation-result guards;
- finite autograd and double-precision gradcheck;
- semantic CUDA-device comparison and optional CUDA execution.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any
from unittest.mock import patch

import pytest
import torch
from torch import nn

from urban_resilience_models.v2_hazard_conditioned_functional_ugnn.constants import (
    CANONICAL_RELATION_GATE_ACTIVATIONS,
    RELATION_GATE_ACTIVATION_SIGMOID,
    RELATION_GATE_SCOPE_TARGET_NODE,
    V2_0_IMPLEMENTED_RELATION_GATE_ACTIVATIONS,
)
from urban_resilience_models.v2_hazard_conditioned_functional_ugnn.functional_message_passing.relation_family_gate import (
    activations as activations_module,
)
from urban_resilience_models.v2_hazard_conditioned_functional_ugnn.functional_message_passing.relation_family_gate import (
    schemas as schemas_module,
)
from urban_resilience_models.v2_hazard_conditioned_functional_ugnn.functional_message_passing.relation_family_gate.activations import (
    RELATION_GATE_ACTIVATIONS_SCHEMA_VERSION,
    GateActivation,
    RelationGateActivation,
    apply_relation_gate_activation,
    sigmoid_gate_activation,
)
from urban_resilience_models.v2_hazard_conditioned_functional_ugnn.functional_message_passing.relation_family_gate.schemas import (
    GateActivationOutput,
    GateNetworkOutput,
    RelationGateAxis,
    RelationPriorContribution,
)


NODE_COUNT = 4
RELATION_COUNT = 3

RELATION_NAMES = (
    "spatial_adjacency",
    "temporal_lag",
    "random_placebo",
)
STABLE_RELATION_IDS = (
    100,
    200,
    900,
)
CONTROL_MASK_VALUES = (
    False,
    False,
    True,
)

FAMILY_NAMES = (
    "physical",
    "control",
)
STABLE_FAMILY_IDS = (
    10,
    90,
)
RELATION_FAMILY_INDICES = (
    0,
    0,
    1,
)

COMPILED_REGISTRY_FINGERPRINT = (
    "compiled-relation-registry"
)
SOURCE_REGISTRY_FINGERPRINT = (
    "source-relation-registry"
)
COMPILED_PRIOR_FINGERPRINT = (
    "compiled-relation-priors"
)


# =============================================================================
# Controlled upstream contracts
# =============================================================================


class FakeRelationConfig:
    def __init__(
        self,
        *,
        gate_activation: str = (
            RELATION_GATE_ACTIVATION_SIGMOID
        ),
        gate_enabled: bool = True,
        validation_error: Exception | None = None,
        implementation_error: Exception | None = None,
    ) -> None:
        self.gate_activation = gate_activation
        self.gate_enabled = gate_enabled
        self.validation_error = validation_error
        self.implementation_error = (
            implementation_error
        )
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


class FakeCompiledRegistry:
    def __init__(
        self,
        fingerprint: str = (
            COMPILED_REGISTRY_FINGERPRINT
        ),
    ) -> None:
        self._fingerprint = fingerprint

    def fingerprint(self) -> str:
        return self._fingerprint


class FakeCompiledPriors:
    def __init__(
        self,
        fingerprint: str = (
            COMPILED_PRIOR_FINGERPRINT
        ),
    ) -> None:
        self._fingerprint = fingerprint

    def fingerprint(self) -> str:
        return self._fingerprint


class FakeFunctionalMessagePassingInputs:
    def __init__(
        self,
        *,
        num_nodes: int = NODE_COUNT,
        relation_names: tuple[str, ...] = (
            RELATION_NAMES
        ),
        stable_relation_ids: tuple[int, ...] = (
            STABLE_RELATION_IDS
        ),
        dtype: torch.dtype = torch.float32,
        device: torch.device | str = "cpu",
        compiled_relation_priors: object | None = ...,
        lineage_fingerprint: str = (
            "functional-message-passing-inputs"
        ),
    ) -> None:
        resolved_device = torch.device(
            device
        )

        self.num_nodes = num_nodes
        self.relation_names = tuple(
            relation_names
        )
        self.stable_relation_ids = tuple(
            stable_relation_ids
        )
        self.num_relations = len(
            self.relation_names
        )
        self.dtype = dtype
        self.device = resolved_device

        self.control_relation_mask = (
            torch.tensor(
                CONTROL_MASK_VALUES,
                dtype=torch.bool,
                device=resolved_device,
            )
        )
        self.compiled_relation_registry = (
            FakeCompiledRegistry()
        )
        self.relation_families = (
            SimpleNamespace(
                family_names=FAMILY_NAMES,
                stable_family_ids=(
                    STABLE_FAMILY_IDS
                ),
                relation_family_index_by_relation=torch.tensor(
                    RELATION_FAMILY_INDICES,
                    dtype=torch.long,
                    device=resolved_device,
                ),
                source_relation_registry_fingerprint=(
                    SOURCE_REGISTRY_FINGERPRINT
                ),
            )
        )
        self.compiled_relation_priors = (
            FakeCompiledPriors()
            if compiled_relation_priors is ...
            else compiled_relation_priors
        )
        self._lineage_fingerprint = (
            lineage_fingerprint
        )

    def lineage_fingerprint(self) -> str:
        return self._lineage_fingerprint


@pytest.fixture(autouse=True)
def _patch_upstream_types(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        activations_module,
        "RelationConfig",
        FakeRelationConfig,
    )
    monkeypatch.setattr(
        schemas_module,
        "FunctionalMessagePassingInputs",
        FakeFunctionalMessagePassingInputs,
    )


# =============================================================================
# Helpers
# =============================================================================


def _inputs(
    **kwargs: Any,
) -> FakeFunctionalMessagePassingInputs:
    return FakeFunctionalMessagePassingInputs(
        **kwargs
    )


def _axis(
    *,
    inputs: (
        FakeFunctionalMessagePassingInputs
        | None
    ) = None,
    control_mask: torch.Tensor | None = None,
) -> RelationGateAxis:
    resolved_inputs = (
        _inputs()
        if inputs is None
        else inputs
    )

    return RelationGateAxis(
        relation_names=RELATION_NAMES,
        stable_relation_ids=(
            STABLE_RELATION_IDS
        ),
        control_relation_mask=(
            resolved_inputs
            .control_relation_mask
            if control_mask is None
            else control_mask
        ),
        compiled_relation_registry_fingerprint=(
            COMPILED_REGISTRY_FINGERPRINT
        ),
        family_names=FAMILY_NAMES,
        stable_family_ids=(
            STABLE_FAMILY_IDS
        ),
        relation_family_index_by_relation=(
            resolved_inputs
            .relation_families
            .relation_family_index_by_relation
        ),
        source_relation_registry_fingerprint=(
            SOURCE_REGISTRY_FINGERPRINT
        ),
    )


def _logits(
    *,
    dtype: torch.dtype = torch.float32,
    device: torch.device | str = "cpu",
    requires_grad: bool = False,
) -> torch.Tensor:
    value = torch.tensor(
        [
            [0.0, 0.5, -0.5],
            [0.2, -0.2, 0.1],
            [1.0, -1.0, 0.0],
            [0.3, 0.4, 0.5],
        ],
        dtype=dtype,
        device=device,
    )
    value.requires_grad_(
        requires_grad
    )
    return value


def _network(
    *,
    inputs: (
        FakeFunctionalMessagePassingInputs
        | None
    ) = None,
    axis: RelationGateAxis | None = None,
    logits: torch.Tensor | None = None,
) -> GateNetworkOutput:
    resolved_inputs = (
        _inputs()
        if inputs is None
        else inputs
    )
    resolved_axis = (
        _axis(
            inputs=resolved_inputs
        )
        if axis is None
        else axis
    )
    resolved_logits = (
        _logits(
            dtype=resolved_inputs.dtype,
            device=resolved_inputs.device,
        )
        if logits is None
        else logits
    )

    return GateNetworkOutput(
        logits=resolved_logits,
        source_inputs=resolved_inputs,
        axis=resolved_axis,
        scope=(
            RELATION_GATE_SCOPE_TARGET_NODE
        ),
        encoder_architecture_fingerprint=(
            "gate-network-architecture"
        ),
        parameter_fingerprint=(
            "gate-network-parameters"
        ),
        input_feature_names=(
            "node_state",
            "hazard_query",
        ),
    )


def _prior(
    *,
    inputs: (
        FakeFunctionalMessagePassingInputs
        | None
    ) = None,
    axis: RelationGateAxis | None = None,
    contribution: torch.Tensor | None = None,
    strength: float = 0.25,
) -> RelationPriorContribution:
    resolved_inputs = (
        _inputs()
        if inputs is None
        else inputs
    )
    resolved_axis = (
        _axis(
            inputs=resolved_inputs
        )
        if axis is None
        else axis
    )
    resolved_contribution = (
        torch.full(
            (
                resolved_inputs.num_nodes,
                resolved_inputs.num_relations,
            ),
            0.1,
            dtype=resolved_inputs.dtype,
            device=resolved_inputs.device,
        )
        if contribution is None
        else contribution
    )

    return RelationPriorContribution(
        logit_contribution=(
            resolved_contribution
        ),
        source_inputs=resolved_inputs,
        axis=resolved_axis,
        strength=strength,
        source_compiled_prior_fingerprint=(
            COMPILED_PRIOR_FINGERPRINT
        ),
        resolution_summary={
            "explicit": (
                resolved_inputs.num_nodes
                * resolved_inputs.num_relations
            ),
        },
    )


def _canonical_unimplemented_activation() -> (
    str | None
):
    for activation in (
        CANONICAL_RELATION_GATE_ACTIVATIONS
    ):
        if activation not in (
            V2_0_IMPLEMENTED_RELATION_GATE_ACTIVATIONS
        ):
            return activation
    return None


# =============================================================================
# Public identity and constructor
# =============================================================================


def test_schema_version_is_nonempty() -> None:
    assert isinstance(
        RELATION_GATE_ACTIVATIONS_SCHEMA_VERSION,
        str,
    )
    assert (
        RELATION_GATE_ACTIVATIONS_SCHEMA_VERSION
        .strip()
    )


def test_alias_points_to_activation_class() -> None:
    assert GateActivation is (
        RelationGateActivation
    )


def test_activation_class_is_module() -> None:
    assert issubclass(
        RelationGateActivation,
        nn.Module,
    )


def test_default_constructor_selects_sigmoid() -> None:
    activation = RelationGateActivation()

    assert activation.activation == (
        RELATION_GATE_ACTIVATION_SIGMOID
    )
    assert activation.is_sigmoid


def test_constructor_accepts_sigmoid() -> None:
    activation = RelationGateActivation(
        activation=(
            RELATION_GATE_ACTIVATION_SIGMOID
        )
    )

    assert activation.is_sigmoid


def test_constructor_strips_whitespace() -> None:
    activation = RelationGateActivation(
        activation=(
            f"  {RELATION_GATE_ACTIVATION_SIGMOID}  "
        )
    )

    assert activation.activation == (
        RELATION_GATE_ACTIVATION_SIGMOID
    )


@pytest.mark.parametrize(
    "activation",
    (
        "",
        " ",
        "\t",
    ),
)
def test_constructor_rejects_blank_activation(
    activation: str,
) -> None:
    with pytest.raises(
        ValueError,
        match="non-empty string",
    ):
        RelationGateActivation(
            activation=activation
        )


@pytest.mark.parametrize(
    "activation",
    (
        None,
        1,
        True,
        object(),
    ),
)
def test_constructor_rejects_nonstring_activation(
    activation: Any,
) -> None:
    with pytest.raises(
        TypeError,
        match="must be a string",
    ):
        RelationGateActivation(
            activation=activation
        )


def test_constructor_rejects_unknown_activation() -> None:
    with pytest.raises(
        ValueError,
        match="Unknown relation-gate activation",
    ):
        RelationGateActivation(
            activation="unknown"
        )


def test_constructor_rejects_canonical_unimplemented_activation() -> None:
    activation = (
        _canonical_unimplemented_activation()
    )

    if activation is None:
        pytest.skip(
            "No canonical unimplemented activation exists."
        )

    with pytest.raises(
        NotImplementedError,
        match="canonical but not implemented",
    ):
        RelationGateActivation(
            activation=activation
        )


def test_implemented_activations_are_canonical() -> None:
    assert set(
        V2_0_IMPLEMENTED_RELATION_GATE_ACTIVATIONS
    ).issubset(
        set(
            CANONICAL_RELATION_GATE_ACTIVATIONS
        )
    )
    assert RELATION_GATE_ACTIVATION_SIGMOID in (
        V2_0_IMPLEMENTED_RELATION_GATE_ACTIVATIONS
    )


# =============================================================================
# Construction from config
# =============================================================================


def test_from_config_builds_sigmoid_activation() -> None:
    config = FakeRelationConfig(
        gate_activation=(
            RELATION_GATE_ACTIVATION_SIGMOID
        ),
        gate_enabled=True,
    )

    activation = (
        RelationGateActivation.from_config(
            config=config
        )
    )

    assert config.validate_calls == 1
    assert (
        config.assert_implemented_calls
        == 1
    )
    assert activation.is_sigmoid


def test_from_disabled_config_skips_assert_implemented() -> None:
    config = FakeRelationConfig(
        gate_enabled=False
    )

    activation = (
        RelationGateActivation.from_config(
            config=config
        )
    )

    assert config.validate_calls == 1
    assert (
        config.assert_implemented_calls
        == 0
    )
    assert activation.is_sigmoid


def test_from_config_rejects_wrong_type() -> None:
    with pytest.raises(
        TypeError,
        match="RelationConfig",
    ):
        RelationGateActivation.from_config(
            config=object()  # type: ignore[arg-type]
        )


def test_from_config_propagates_validation_error() -> None:
    config = FakeRelationConfig(
        validation_error=RuntimeError(
            "invalid relation config"
        )
    )

    with pytest.raises(
        RuntimeError,
        match="invalid relation config",
    ):
        RelationGateActivation.from_config(
            config=config
        )


def test_from_config_propagates_implementation_error() -> None:
    config = FakeRelationConfig(
        implementation_error=(
            NotImplementedError(
                "gate activation unavailable"
            )
        )
    )

    with pytest.raises(
        NotImplementedError,
        match="gate activation unavailable",
    ):
        RelationGateActivation.from_config(
            config=config
        )


# =============================================================================
# Functional sigmoid activation
# =============================================================================


def test_sigmoid_gate_activation_exact_values() -> None:
    logits = torch.tensor(
        [
            [-2.0, 0.0, 2.0],
            [1.0, -1.0, 0.5],
        ]
    )

    observed = sigmoid_gate_activation(
        logits
    )

    assert torch.equal(
        observed,
        torch.sigmoid(logits),
    )


@pytest.mark.parametrize(
    "dtype",
    (
        torch.float32,
        torch.float64,
        torch.bfloat16,
    ),
)
def test_sigmoid_preserves_shape_dtype_and_device(
    dtype: torch.dtype,
) -> None:
    logits = _logits(dtype=dtype)

    observed = sigmoid_gate_activation(
        logits
    )

    assert observed.shape == logits.shape
    assert observed.dtype == dtype
    assert observed.device == logits.device


def test_sigmoid_values_lie_in_unit_interval() -> None:
    logits = torch.linspace(
        -100.0,
        100.0,
        steps=300,
    ).reshape(100, 3)

    observed = sigmoid_gate_activation(
        logits
    )

    assert bool(
        (observed >= 0).all().item()
    )
    assert bool(
        (observed <= 1).all().item()
    )


def test_sigmoid_channels_do_not_compete() -> None:
    logits = torch.full(
        (2, 3),
        2.0,
    )

    observed = sigmoid_gate_activation(
        logits
    )

    assert bool(
        (
            observed.sum(dim=-1)
            > 1.0
        )
        .all()
        .item()
    )


def test_sigmoid_is_elementwise_monotone() -> None:
    lower = torch.tensor(
        [[-2.0, 0.0, 1.0]]
    )
    upper = torch.tensor(
        [[-1.0, 0.5, 2.0]]
    )

    assert bool(
        (
            sigmoid_gate_activation(
                upper
            )
            > sigmoid_gate_activation(
                lower
            )
        )
        .all()
        .item()
    )


def test_sigmoid_zero_maps_to_half() -> None:
    observed = sigmoid_gate_activation(
        torch.zeros(2, 3)
    )

    assert torch.equal(
        observed,
        torch.full(
            (2, 3),
            0.5,
        ),
    )


def test_sigmoid_rejects_nontensor() -> None:
    with pytest.raises(
        TypeError,
        match="must be a tensor",
    ):
        sigmoid_gate_activation(
            [[0.0]]  # type: ignore[arg-type]
        )


@pytest.mark.parametrize(
    "logits",
    (
        torch.zeros(3),
        torch.zeros(1, 2, 3),
    ),
)
def test_sigmoid_rejects_invalid_rank(
    logits: torch.Tensor,
) -> None:
    with pytest.raises(
        ValueError,
        match=r"shape \[N, R\]",
    ):
        sigmoid_gate_activation(
            logits
        )


@pytest.mark.parametrize(
    "dtype",
    (
        torch.long,
        torch.int32,
        torch.bool,
    ),
)
def test_sigmoid_rejects_nonfloating_dtype(
    dtype: torch.dtype,
) -> None:
    with pytest.raises(
        ValueError,
        match="floating-point dtype",
    ):
        sigmoid_gate_activation(
            torch.zeros(
                2,
                3,
                dtype=dtype,
            )
        )


@pytest.mark.parametrize(
    "bad_value",
    (
        float("nan"),
        float("inf"),
        float("-inf"),
    ),
)
def test_sigmoid_rejects_nonfinite_logits(
    bad_value: float,
) -> None:
    logits = torch.zeros(2, 3)
    logits[0, 0] = bad_value

    with pytest.raises(
        ValueError,
        match="finite values",
    ):
        sigmoid_gate_activation(
            logits
        )


def test_apply_dispatch_matches_direct_sigmoid() -> None:
    logits = _logits()

    assert torch.equal(
        apply_relation_gate_activation(
            logits,
            activation=(
                RELATION_GATE_ACTIVATION_SIGMOID
            ),
        ),
        sigmoid_gate_activation(logits),
    )


def test_apply_dispatch_uses_default_sigmoid() -> None:
    logits = _logits()

    assert torch.equal(
        apply_relation_gate_activation(
            logits
        ),
        torch.sigmoid(logits),
    )


def test_apply_dispatch_rejects_unknown_activation() -> None:
    with pytest.raises(
        ValueError,
        match="Unknown relation-gate activation",
    ):
        apply_relation_gate_activation(
            _logits(),
            activation="unknown",
        )


def test_apply_dispatch_rejects_canonical_unimplemented() -> None:
    activation = (
        _canonical_unimplemented_activation()
    )

    if activation is None:
        pytest.skip(
            "No canonical unimplemented activation exists."
        )

    with pytest.raises(
        NotImplementedError,
        match="canonical but not implemented",
    ):
        apply_relation_gate_activation(
            _logits(),
            activation=activation,
        )


# =============================================================================
# Internal activation-result guards
# =============================================================================


def test_sigmoid_rejects_nontensor_result(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        activations_module.torch,
        "sigmoid",
        lambda logits: [0.5],
    )

    with pytest.raises(
        RuntimeError,
        match="must return a tensor",
    ):
        sigmoid_gate_activation(
            _logits()
        )


def test_sigmoid_rejects_changed_shape(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        activations_module.torch,
        "sigmoid",
        lambda logits: torch.zeros(
            logits.shape[0],
            logits.shape[1] + 1,
            dtype=logits.dtype,
            device=logits.device,
        ),
    )

    with pytest.raises(
        RuntimeError,
        match="changed shape",
    ):
        sigmoid_gate_activation(
            _logits()
        )


def test_sigmoid_rejects_changed_dtype(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        activations_module.torch,
        "sigmoid",
        lambda logits: torch.zeros(
            logits.shape,
            dtype=torch.float64,
            device=logits.device,
        ),
    )

    with pytest.raises(
        RuntimeError,
        match="changed dtype",
    ):
        sigmoid_gate_activation(
            _logits(
                dtype=torch.float32
            )
        )


def test_sigmoid_rejects_nonfinite_result(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_sigmoid(
        logits: torch.Tensor,
    ) -> torch.Tensor:
        result = torch.zeros_like(logits)
        result[0, 0] = float("nan")
        return result

    monkeypatch.setattr(
        activations_module.torch,
        "sigmoid",
        fake_sigmoid,
    )

    with pytest.raises(
        FloatingPointError,
        match="NaN or infinity",
    ):
        sigmoid_gate_activation(
            _logits()
        )


@pytest.mark.parametrize(
    "value",
    (
        -0.1,
        1.1,
    ),
)
def test_sigmoid_rejects_out_of_range_result(
    monkeypatch: pytest.MonkeyPatch,
    value: float,
) -> None:
    monkeypatch.setattr(
        activations_module.torch,
        "sigmoid",
        lambda logits: torch.full_like(
            logits,
            value,
        ),
    )

    with pytest.raises(
        FloatingPointError,
        match=r"\[0, 1\]",
    ):
        sigmoid_gate_activation(
            _logits()
        )


def test_sigmoid_rejects_semantic_device_change(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        activations_module,
        "_devices_match",
        lambda first, second: False,
    )

    with pytest.raises(
        RuntimeError,
        match="changed device",
    ):
        sigmoid_gate_activation(
            _logits()
        )


# =============================================================================
# Parameter-free identity and architecture
# =============================================================================


def test_activation_module_is_parameter_free() -> None:
    activation = RelationGateActivation()

    assert activation.parameter_count == 0
    assert (
        activation.trainable_parameter_count
        == 0
    )
    assert tuple(
        activation.parameters()
    ) == ()
    assert activation.state_dict() == {}


def test_architecture_dict_is_exact() -> None:
    activation = RelationGateActivation()

    assert activation.architecture_dict() == {
        "schema_version": (
            RELATION_GATE_ACTIVATIONS_SCHEMA_VERSION
        ),
        "activation": (
            RELATION_GATE_ACTIVATION_SIGMOID
        ),
        "implemented_formula": (
            "gate_values = sigmoid(gate_logits)"
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


def test_architecture_fingerprint_is_stable() -> None:
    first = RelationGateActivation()
    second = RelationGateActivation()

    assert first.architecture_dict() == (
        second.architecture_dict()
    )
    assert first.architecture_fingerprint() == (
        second.architecture_fingerprint()
    )


def test_parameter_fingerprint_is_stable() -> None:
    first = RelationGateActivation()
    second = RelationGateActivation()

    assert first.parameter_fingerprint() == (
        second.parameter_fingerprint()
    )


def test_architecture_and_parameter_fingerprints_are_distinct() -> None:
    activation = RelationGateActivation()

    assert (
        activation.architecture_fingerprint()
        != activation.parameter_fingerprint()
    )


def test_assert_finite_parameters_is_noop() -> None:
    RelationGateActivation().assert_finite_parameters()


def test_assert_finite_parameters_rejects_nonzero_contract() -> None:
    class InvalidActivation(
        RelationGateActivation
    ):
        @property
        def parameter_count(self) -> int:
            return 1

    with pytest.raises(
        RuntimeError,
        match="must remain parameter-free",
    ):
        InvalidActivation().assert_finite_parameters()


# =============================================================================
# Logit composition
# =============================================================================


def test_combine_without_prior_returns_exact_logits_object() -> None:
    network = _network()

    combined = (
        RelationGateActivation()
        .combine_logits(
            network,
            None,
        )
    )

    assert combined is network.logits


def test_combine_with_prior_is_exact_addition() -> None:
    inputs = _inputs()
    axis = _axis(inputs=inputs)
    network = _network(
        inputs=inputs,
        axis=axis,
    )
    prior = _prior(
        inputs=inputs,
        axis=axis,
    )

    combined = (
        RelationGateActivation()
        .combine_logits(
            network,
            prior,
        )
    )

    assert torch.equal(
        combined,
        (
            network.logits
            + prior.logit_contribution
        ),
    )


def test_combine_with_prior_preserves_shape_dtype_device() -> None:
    inputs = _inputs(
        dtype=torch.float64
    )
    axis = _axis(inputs=inputs)
    network = _network(
        inputs=inputs,
        axis=axis,
        logits=_logits(
            dtype=torch.float64
        ),
    )
    prior = _prior(
        inputs=inputs,
        axis=axis,
        contribution=torch.full(
            (NODE_COUNT, RELATION_COUNT),
            0.25,
            dtype=torch.float64,
        ),
    )

    combined = (
        RelationGateActivation()
        .combine_logits(
            network,
            prior,
        )
    )

    assert combined.shape == (
        NODE_COUNT,
        RELATION_COUNT,
    )
    assert combined.dtype == torch.float64
    assert combined.device == torch.device(
        "cpu"
    )


def test_combine_rejects_wrong_network_type() -> None:
    with pytest.raises(
        TypeError,
        match="GateNetworkOutput",
    ):
        RelationGateActivation().combine_logits(
            object()  # type: ignore[arg-type]
        )


def test_combine_rejects_wrong_prior_type() -> None:
    with pytest.raises(
        TypeError,
        match="RelationPriorContribution",
    ):
        RelationGateActivation().combine_logits(
            _network(),
            object(),  # type: ignore[arg-type]
        )


def test_combine_requires_same_source_inputs_object() -> None:
    network = _network()
    other_inputs = _inputs()
    prior = _prior(
        inputs=other_inputs,
        axis=_axis(
            inputs=other_inputs
        ),
    )

    with pytest.raises(
        ValueError,
        match="exact same source_inputs object",
    ):
        RelationGateActivation().combine_logits(
            network,
            prior,
        )


def test_combine_requires_same_relation_axis() -> None:
    inputs = _inputs()
    axis = _axis(inputs=inputs)
    network = _network(
        inputs=inputs,
        axis=axis,
    )
    prior = _prior(
        inputs=inputs,
        axis=axis,
    )

    changed_axis = RelationGateAxis(
        relation_names=RELATION_NAMES,
        stable_relation_ids=(
            STABLE_RELATION_IDS
        ),
        control_relation_mask=torch.tensor(
            [False, True, True],
            dtype=torch.bool,
        ),
        compiled_relation_registry_fingerprint=(
            COMPILED_REGISTRY_FINGERPRINT
        ),
        family_names=FAMILY_NAMES,
        stable_family_ids=(
            STABLE_FAMILY_IDS
        ),
        relation_family_index_by_relation=torch.tensor(
            RELATION_FAMILY_INDICES,
            dtype=torch.long,
        ),
        source_relation_registry_fingerprint=(
            SOURCE_REGISTRY_FINGERPRINT
        ),
    )
    object.__setattr__(
        prior,
        "axis",
        changed_axis,
    )

    with pytest.raises(
        ValueError,
        match="same relation-gate axis",
    ):
        RelationGateActivation().combine_logits(
            network,
            prior,
        )


def test_combine_rejects_prior_shape_mismatch() -> None:
    inputs = _inputs()
    axis = _axis(inputs=inputs)
    network = _network(
        inputs=inputs,
        axis=axis,
    )
    prior = _prior(
        inputs=inputs,
        axis=axis,
    )
    object.__setattr__(
        prior,
        "logit_contribution",
        torch.zeros(
            1,
            RELATION_COUNT,
        ),
    )

    with pytest.raises(
        ValueError,
        match="same shape",
    ):
        RelationGateActivation().combine_logits(
            network,
            prior,
        )


def test_combine_rejects_prior_dtype_mismatch() -> None:
    inputs = _inputs()
    axis = _axis(inputs=inputs)
    network = _network(
        inputs=inputs,
        axis=axis,
    )
    prior = _prior(
        inputs=inputs,
        axis=axis,
    )
    object.__setattr__(
        prior,
        "logit_contribution",
        prior
        .logit_contribution
        .to(torch.float64),
    )

    with pytest.raises(
        ValueError,
        match="same dtype",
    ):
        RelationGateActivation().combine_logits(
            network,
            prior,
        )


def test_combine_rejects_nonfinite_result_from_overflow() -> None:
    inputs = _inputs(
        dtype=torch.float16
    )
    axis = _axis(inputs=inputs)
    maximum = torch.finfo(
        torch.float16
    ).max
    network = _network(
        inputs=inputs,
        axis=axis,
        logits=torch.full(
            (
                NODE_COUNT,
                RELATION_COUNT,
            ),
            maximum,
            dtype=torch.float16,
        ),
    )
    prior = _prior(
        inputs=inputs,
        axis=axis,
        contribution=torch.full(
            (
                NODE_COUNT,
                RELATION_COUNT,
            ),
            maximum,
            dtype=torch.float16,
        ),
    )

    with pytest.raises(
        ValueError,
        match="finite values",
    ):
        RelationGateActivation().combine_logits(
            network,
            prior,
        )


def test_combine_rejects_prior_device_mismatch_semantically(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    inputs = _inputs()
    axis = _axis(inputs=inputs)
    network = _network(
        inputs=inputs,
        axis=axis,
    )
    prior = _prior(
        inputs=inputs,
        axis=axis,
    )

    original = (
        activations_module._devices_match
    )
    call_count = 0

    def fake_devices_match(
        first: torch.device | str,
        second: torch.device | str,
    ) -> bool:
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            return False
        return original(first, second)

    monkeypatch.setattr(
        activations_module,
        "_devices_match",
        fake_devices_match,
    )

    with pytest.raises(
        ValueError,
        match="share one device",
    ):
        RelationGateActivation().combine_logits(
            network,
            prior,
        )


# =============================================================================
# Module activation and forward output
# =============================================================================


def test_activate_tensor_matches_functional_dispatch() -> None:
    activation = RelationGateActivation()
    logits = _logits()

    assert torch.equal(
        activation.activate_tensor(
            logits
        ),
        apply_relation_gate_activation(
            logits
        ),
    )


def test_forward_without_prior_preserves_network_logits_object() -> None:
    network = _network()

    output = RelationGateActivation()(
        network,
        None,
    )

    assert isinstance(
        output,
        GateActivationOutput,
    )
    assert output.gate_logits is (
        network.logits
    )
    assert torch.equal(
        output.gate_values,
        torch.sigmoid(network.logits),
    )
    assert output.source_network_output is (
        network
    )
    assert output.prior_contribution is None


def test_forward_with_prior_constructs_complete_output() -> None:
    inputs = _inputs()
    axis = _axis(inputs=inputs)
    network = _network(
        inputs=inputs,
        axis=axis,
    )
    prior = _prior(
        inputs=inputs,
        axis=axis,
    )
    activation = RelationGateActivation()

    output = activation(
        network,
        prior,
    )

    expected_logits = (
        network.logits
        + prior.logit_contribution
    )

    assert isinstance(
        output,
        GateActivationOutput,
    )
    assert torch.equal(
        output.gate_logits,
        expected_logits,
    )
    assert torch.equal(
        output.gate_values,
        torch.sigmoid(expected_logits),
    )
    assert output.source_network_output is (
        network
    )
    assert output.prior_contribution is prior
    assert output.activation == (
        RELATION_GATE_ACTIVATION_SIGMOID
    )
    assert (
        output.encoder_architecture_fingerprint
        == activation.architecture_fingerprint()
    )
    assert (
        output.parameter_fingerprint
        == activation.parameter_fingerprint()
    )


def test_forward_is_deterministic() -> None:
    inputs = _inputs()
    axis = _axis(inputs=inputs)
    network = _network(
        inputs=inputs,
        axis=axis,
    )
    prior = _prior(
        inputs=inputs,
        axis=axis,
    )
    activation = RelationGateActivation()

    first = activation(network, prior)
    second = activation(network, prior)

    assert torch.equal(
        first.gate_logits,
        second.gate_logits,
    )
    assert torch.equal(
        first.gate_values,
        second.gate_values,
    )
    assert (
        first.encoder_architecture_fingerprint
        == second.encoder_architecture_fingerprint
    )
    assert (
        first.parameter_fingerprint
        == second.parameter_fingerprint
    )


# =============================================================================
# Autograd
# =============================================================================


def test_direct_sigmoid_backward_matches_derivative() -> None:
    logits = _logits(
        dtype=torch.float64,
        requires_grad=True,
    )

    values = sigmoid_gate_activation(
        logits
    )
    values.sum().backward()

    assert logits.grad is not None
    expected = (
        values.detach()
        * (
            1.0
            - values.detach()
        )
    )
    assert torch.allclose(
        logits.grad,
        expected,
        atol=1e-12,
        rtol=1e-12,
    )


def test_forward_without_prior_backpropagates_to_neural_logits() -> None:
    inputs = _inputs(
        dtype=torch.float64
    )
    axis = _axis(inputs=inputs)
    logits = _logits(
        dtype=torch.float64,
        requires_grad=True,
    )
    network = _network(
        inputs=inputs,
        axis=axis,
        logits=logits,
    )

    output = RelationGateActivation()(
        network,
        None,
    )
    output.gate_values.sum().backward()

    assert logits.grad is not None
    assert bool(
        torch.isfinite(
            logits.grad
        )
        .all()
        .item()
    )


def test_forward_with_prior_backpropagates_to_both_inputs() -> None:
    inputs = _inputs(
        dtype=torch.float64
    )
    axis = _axis(inputs=inputs)
    logits = _logits(
        dtype=torch.float64,
        requires_grad=True,
    )
    prior_values = torch.full(
        (
            NODE_COUNT,
            RELATION_COUNT,
        ),
        0.2,
        dtype=torch.float64,
        requires_grad=True,
    )
    network = _network(
        inputs=inputs,
        axis=axis,
        logits=logits,
    )
    prior = _prior(
        inputs=inputs,
        axis=axis,
        contribution=prior_values,
    )

    output = RelationGateActivation()(
        network,
        prior,
    )
    output.gate_values.sum().backward()

    assert logits.grad is not None
    assert prior_values.grad is not None
    assert torch.allclose(
        logits.grad,
        prior_values.grad,
        atol=1e-12,
        rtol=1e-12,
    )


def test_sigmoid_gradcheck() -> None:
    logits = torch.randn(
        3,
        4,
        dtype=torch.float64,
        requires_grad=True,
    )

    assert torch.autograd.gradcheck(
        sigmoid_gate_activation,
        (logits,),
        eps=1e-6,
        atol=1e-5,
        rtol=1e-3,
    )


# =============================================================================
# Semantic device helper
# =============================================================================


def test_devices_match_cpu() -> None:
    assert activations_module._devices_match(
        "cpu",
        torch.device("cpu"),
    )


def test_devices_match_rejects_cpu_cuda() -> None:
    assert not activations_module._devices_match(
        "cpu",
        "cuda:0",
    )


def test_devices_match_explicit_cuda_indices() -> None:
    assert activations_module._devices_match(
        "cuda:0",
        "cuda:0",
    )
    assert not activations_module._devices_match(
        "cuda:0",
        "cuda:1",
    )


def test_devices_match_resolves_current_cuda_index() -> None:
    with patch.object(
        torch.cuda,
        "current_device",
        return_value=0,
    ):
        assert activations_module._devices_match(
            "cuda",
            "cuda:0",
        )
        assert activations_module._devices_match(
            "cuda:0",
            "cuda",
        )
        assert not activations_module._devices_match(
            "cuda",
            "cuda:1",
        )


# =============================================================================
# Representation
# =============================================================================


def test_extra_repr_contains_contract_identity() -> None:
    representation = (
        RelationGateActivation()
        .extra_repr()
    )

    assert (
        RELATION_GATE_ACTIVATION_SIGMOID
        in representation
    )
    assert "parameter_count=0" in (
        representation
    )
    assert (
        "relation_channels_compete=False"
        in representation
    )


# =============================================================================
# Optional CUDA
# =============================================================================


@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA is unavailable.",
)
def test_cuda_functional_sigmoid_matches_cpu() -> None:
    cpu_logits = _logits()
    cuda_logits = _logits(
        device="cuda"
    )

    cpu_values = sigmoid_gate_activation(
        cpu_logits
    )
    cuda_values = sigmoid_gate_activation(
        cuda_logits
    )

    assert torch.allclose(
        cpu_values,
        cuda_values.cpu(),
        atol=1e-6,
        rtol=1e-6,
    )


@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA is unavailable.",
)
def test_cuda_forward_accepts_implicit_device_metadata() -> None:
    inputs = _inputs(
        device=torch.device("cuda")
    )
    axis = _axis(inputs=inputs)
    network = _network(
        inputs=inputs,
        axis=axis,
        logits=_logits(
            device="cuda"
        ),
    )
    prior = _prior(
        inputs=inputs,
        axis=axis,
        contribution=torch.full(
            (
                NODE_COUNT,
                RELATION_COUNT,
            ),
            0.1,
            device="cuda",
        ),
    )

    output = RelationGateActivation()(
        network,
        prior,
    )

    assert output.device.type == "cuda"
    assert output.gate_logits.device.type == (
        "cuda"
    )
    assert output.gate_values.device.type == (
        "cuda"
    )


@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA is unavailable.",
)
def test_cuda_backward_is_finite() -> None:
    inputs = _inputs(
        dtype=torch.float64,
        device=torch.device("cuda"),
    )
    axis = _axis(inputs=inputs)
    logits = _logits(
        dtype=torch.float64,
        device="cuda",
        requires_grad=True,
    )
    prior_values = torch.full(
        (
            NODE_COUNT,
            RELATION_COUNT,
        ),
        0.2,
        dtype=torch.float64,
        device="cuda",
        requires_grad=True,
    )
    network = _network(
        inputs=inputs,
        axis=axis,
        logits=logits,
    )
    prior = _prior(
        inputs=inputs,
        axis=axis,
        contribution=prior_values,
    )

    output = RelationGateActivation()(
        network,
        prior,
    )
    output.gate_values.square().mean().backward()

    assert logits.grad is not None
    assert prior_values.grad is not None
    assert bool(
        torch.isfinite(
            logits.grad
        )
        .all()
        .item()
    )
    assert bool(
        torch.isfinite(
            prior_values.grad
        )
        .all()
        .item()
    )


@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA is unavailable.",
)
def test_cuda_rejects_cpu_prior_contribution() -> None:
    inputs = _inputs(
        device=torch.device("cuda")
    )
    axis = _axis(inputs=inputs)
    network = _network(
        inputs=inputs,
        axis=axis,
        logits=_logits(
            device="cuda"
        ),
    )
    prior = _prior(
        inputs=inputs,
        axis=axis,
        contribution=torch.full(
            (
                NODE_COUNT,
                RELATION_COUNT,
            ),
            0.1,
            device="cuda",
        ),
    )
    object.__setattr__(
        prior,
        "logit_contribution",
        prior.logit_contribution.cpu(),
    )

    with pytest.raises(
        ValueError,
        match="share one device",
    ):
        RelationGateActivation().combine_logits(
            network,
            prior,
        )


@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA is unavailable.",
)
def test_cuda_rejects_activation_result_moved_to_cpu(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original_sigmoid = torch.sigmoid

    def fake_sigmoid(
        logits: torch.Tensor,
    ) -> torch.Tensor:
        return original_sigmoid(
            logits
        ).cpu()

    monkeypatch.setattr(
        activations_module.torch,
        "sigmoid",
        fake_sigmoid,
    )

    with pytest.raises(
        RuntimeError,
        match="changed device",
    ):
        sigmoid_gate_activation(
            _logits(
                device="cuda"
            )
        )
