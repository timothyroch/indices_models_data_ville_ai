"""
Contract tests for target-node relation-gate logit prediction.

Target repository path:
    urban_resilience_models/
        v2_hazard_conditioned_functional_ugnn/
            tests/
                test_relation_gate_network.py

Implementation under test:
    functional_message_passing/
        relation_family_gate/
            gate_network.py

The bounded V2.0 gate network consumes enabled node-aligned context features
and predicts one independent logit for every exact compiled relation:

    context[n] = concat(enabled node_state[n], enabled hazard_query[n])

    hidden[n] = optional_layer_norm(
        GELU(
            hidden_projection(
                GELU(
                    input_projection(context[n])
                )
            )
        )
    )

    logits[n, r] =
        dot(hidden[n], relation_embedding[r]) / sqrt(hidden_dim)
        + optional_relation_bias[r]

The trainable axis is the exact relation axis ``R``. Semantic family metadata
must never pool, collapse, or reorder relation channels.

This suite covers:

- constructor and configuration validation;
- module topology, initialization, and parameter counts;
- exact input-feature selection and concatenation;
- hidden-context encoding;
- exact relation scoring, scaling, and bias behavior;
- exact relation-axis alignment;
- architecture and parameter fingerprints;
- finite-parameter guards;
- metadata-preserving ``GateNetworkOutput`` construction;
- node and relation independence properties;
- gradients and double-precision gradcheck;
- semantic device matching and optional CUDA execution.
"""

from __future__ import annotations

import copy
import math
from types import SimpleNamespace
from typing import Any
from unittest.mock import patch

import pytest
import torch
from torch import nn
from torch.nn import functional as F

from urban_resilience_models.v2_hazard_conditioned_functional_ugnn.constants import (
    CANONICAL_RELATION_GATE_SCOPES,
    RELATION_GATE_SCOPE_TARGET_NODE,
    V2_0_IMPLEMENTED_RELATION_GATE_SCOPES,
)
from urban_resilience_models.v2_hazard_conditioned_functional_ugnn.functional_message_passing.relation_family_gate import (
    gate_network as gate_network_module,
)
from urban_resilience_models.v2_hazard_conditioned_functional_ugnn.functional_message_passing.relation_family_gate import (
    schemas as schemas_module,
)
from urban_resilience_models.v2_hazard_conditioned_functional_ugnn.functional_message_passing.relation_family_gate.gate_network import (
    RELATION_GATE_NETWORK_SCHEMA_VERSION,
    GateNetwork,
    RelationGateNetwork,
)
from urban_resilience_models.v2_hazard_conditioned_functional_ugnn.functional_message_passing.relation_family_gate.schemas import (
    GateNetworkOutput,
    RelationGateAxis,
)


NODE_COUNT = 5
NODE_STATE_DIM = 6
HAZARD_QUERY_DIM = 4
HIDDEN_DIM = 8
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


# =============================================================================
# Controlled upstream contracts
# =============================================================================


class FakeRelationConfig:
    def __init__(
        self,
        *,
        gate_hidden_dim: int = HIDDEN_DIM,
        gate_scope: str = (
            RELATION_GATE_SCOPE_TARGET_NODE
        ),
        gate_enabled: bool = True,
        validation_error: Exception | None = None,
        implementation_error: Exception | None = None,
    ) -> None:
        self.gate_hidden_dim = gate_hidden_dim
        self.gate_scope = gate_scope
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
        node_state: object | None = ...,
        node_hazard_query: (
            torch.Tensor | None
        ) = ...,
        dtype: torch.dtype = torch.float32,
        device: torch.device | str = "cpu",
        compiled_registry_fingerprint: str = (
            COMPILED_REGISTRY_FINGERPRINT
        ),
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

        if node_state is ...:
            self.node_state = (
                SimpleNamespace(
                    fused_state=torch.arange(
                        num_nodes
                        * NODE_STATE_DIM,
                        dtype=dtype,
                        device=resolved_device,
                    ).reshape(
                        num_nodes,
                        NODE_STATE_DIM,
                    ) / 10.0
                )
            )
        else:
            self.node_state = node_state

        if node_hazard_query is ...:
            self.node_hazard_query = (
                torch.arange(
                    num_nodes
                    * HAZARD_QUERY_DIM,
                    dtype=dtype,
                    device=resolved_device,
                ).reshape(
                    num_nodes,
                    HAZARD_QUERY_DIM,
                ) / 20.0
            )
        else:
            self.node_hazard_query = (
                node_hazard_query
            )

        self.control_relation_mask = (
            torch.tensor(
                CONTROL_MASK_VALUES,
                dtype=torch.bool,
                device=resolved_device,
            )
        )
        self.compiled_relation_registry = (
            FakeCompiledRegistry(
                compiled_registry_fingerprint
            )
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
        gate_network_module,
        "RelationConfig",
        FakeRelationConfig,
    )
    monkeypatch.setattr(
        gate_network_module,
        "FunctionalMessagePassingInputs",
        FakeFunctionalMessagePassingInputs,
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
) -> RelationGateAxis:
    resolved_inputs = (
        _inputs()
        if inputs is None
        else inputs
    )
    return RelationGateAxis.from_inputs(
        source_inputs=resolved_inputs
    )


def _network(
    *,
    node_state_dim: int = NODE_STATE_DIM,
    hazard_query_dim: int = HAZARD_QUERY_DIM,
    relation_names: tuple[str, ...] = (
        RELATION_NAMES
    ),
    stable_relation_ids: tuple[int, ...] = (
        STABLE_RELATION_IDS
    ),
    hidden_dim: int = HIDDEN_DIM,
    scope: str = (
        RELATION_GATE_SCOPE_TARGET_NODE
    ),
    use_node_state: bool = True,
    use_hazard_query: bool = True,
    layer_norm: bool = True,
    relation_bias: bool = True,
    dtype: torch.dtype = torch.float32,
    device: torch.device | str = "cpu",
    seed: int = 17,
) -> RelationGateNetwork:
    torch.manual_seed(seed)
    module = RelationGateNetwork(
        node_state_dim=node_state_dim,
        hazard_query_dim=(
            hazard_query_dim
        ),
        relation_names=relation_names,
        stable_relation_ids=(
            stable_relation_ids
        ),
        hidden_dim=hidden_dim,
        scope=scope,
        use_node_state=use_node_state,
        use_hazard_query=(
            use_hazard_query
        ),
        layer_norm=layer_norm,
        relation_bias=relation_bias,
    )
    return module.to(
        device=device,
        dtype=dtype,
    )


def _canonical_unimplemented_scope() -> (
    str | None
):
    for scope in (
        CANONICAL_RELATION_GATE_SCOPES
    ):
        if scope not in (
            V2_0_IMPLEMENTED_RELATION_GATE_SCOPES
        ):
            return scope
    return None


def _expected_parameter_count(
    *,
    input_dim: int,
    hidden_dim: int,
    num_relations: int,
    layer_norm: bool,
    relation_bias: bool,
) -> int:
    total = (
        hidden_dim * input_dim
        + hidden_dim
    )
    total += (
        hidden_dim * hidden_dim
        + hidden_dim
    )
    if layer_norm:
        total += 2 * hidden_dim
    total += (
        num_relations * hidden_dim
    )
    if relation_bias:
        total += num_relations
    return total


# =============================================================================
# Public identity and constructor
# =============================================================================


def test_schema_version_is_nonempty() -> None:
    assert isinstance(
        RELATION_GATE_NETWORK_SCHEMA_VERSION,
        str,
    )
    assert (
        RELATION_GATE_NETWORK_SCHEMA_VERSION
        .strip()
    )


def test_alias_points_to_network_class() -> None:
    assert GateNetwork is (
        RelationGateNetwork
    )


def test_network_is_torch_module() -> None:
    assert issubclass(
        RelationGateNetwork,
        nn.Module,
    )


def test_default_constructor_contract() -> None:
    network = _network()

    assert network.node_state_dim == (
        NODE_STATE_DIM
    )
    assert network.hazard_query_dim == (
        HAZARD_QUERY_DIM
    )
    assert network.hidden_dim == HIDDEN_DIM
    assert network.scope == (
        RELATION_GATE_SCOPE_TARGET_NODE
    )
    assert network.use_node_state
    assert network.use_hazard_query
    assert network.layer_norm_enabled
    assert network.relation_bias_enabled
    assert network.relation_names == (
        RELATION_NAMES
    )
    assert network.stable_relation_ids == (
        STABLE_RELATION_IDS
    )
    assert network.num_relations == (
        RELATION_COUNT
    )
    assert network.input_dim == (
        NODE_STATE_DIM
        + HAZARD_QUERY_DIM
    )
    assert network.input_feature_names == (
        "node_state",
        "hazard_query",
    )


@pytest.mark.parametrize(
    "field_name",
    (
        "node_state_dim",
        "hazard_query_dim",
        "hidden_dim",
    ),
)
@pytest.mark.parametrize(
    "value",
    (
        0,
        -1,
    ),
)
def test_constructor_rejects_nonpositive_dimensions(
    field_name: str,
    value: int,
) -> None:
    kwargs = {
        field_name: value,
    }

    with pytest.raises(
        ValueError,
        match="must be positive",
    ):
        _network(**kwargs)


@pytest.mark.parametrize(
    "field_name",
    (
        "node_state_dim",
        "hazard_query_dim",
        "hidden_dim",
    ),
)
@pytest.mark.parametrize(
    "value",
    (
        True,
        1.5,
        "8",
        None,
    ),
)
def test_constructor_rejects_noninteger_dimensions(
    field_name: str,
    value: Any,
) -> None:
    kwargs = {
        field_name: value,
    }

    with pytest.raises(
        TypeError,
        match="must be an integer",
    ):
        _network(**kwargs)


@pytest.mark.parametrize(
    "relation_names",
    (
        (),
        ("",),
        (" ",),
        ("a", "a"),
    ),
)
def test_constructor_rejects_invalid_relation_names(
    relation_names: tuple[str, ...],
) -> None:
    ids = tuple(
        range(len(relation_names))
    )

    with pytest.raises(ValueError):
        _network(
            relation_names=relation_names,
            stable_relation_ids=ids,
        )


@pytest.mark.parametrize(
    "stable_relation_ids",
    (
        (100, -1, 900),
        (100, 100, 900),
    ),
)
def test_constructor_rejects_invalid_stable_relation_ids(
    stable_relation_ids: tuple[int, ...],
) -> None:
    with pytest.raises(ValueError):
        _network(
            stable_relation_ids=(
                stable_relation_ids
            )
        )


@pytest.mark.parametrize(
    "stable_relation_ids",
    (
        (100, True, 900),
        (100, "200", 900),
    ),
)
def test_constructor_rejects_noninteger_stable_relation_ids(
    stable_relation_ids: Any,
) -> None:
    with pytest.raises(
        TypeError,
        match="must be an integer",
    ):
        _network(
            stable_relation_ids=(
                stable_relation_ids
            )
        )


def test_constructor_rejects_relation_metadata_length_mismatch() -> None:
    with pytest.raises(
        ValueError,
        match="must align",
    ):
        _network(
            stable_relation_ids=(
                100,
                200,
            )
        )


@pytest.mark.parametrize(
    "field_name",
    (
        "use_node_state",
        "use_hazard_query",
        "layer_norm",
        "relation_bias",
    ),
)
@pytest.mark.parametrize(
    "value",
    (
        1,
        "true",
        None,
    ),
)
def test_constructor_rejects_nonboolean_flags(
    field_name: str,
    value: Any,
) -> None:
    kwargs = {
        field_name: value,
    }

    with pytest.raises(
        TypeError,
        match="must be a bool",
    ):
        _network(**kwargs)


def test_constructor_requires_at_least_one_input_source() -> None:
    with pytest.raises(
        ValueError,
        match="At least one",
    ):
        _network(
            use_node_state=False,
            use_hazard_query=False,
        )


def test_constructor_strips_scope_whitespace() -> None:
    network = _network(
        scope=(
            f"  {RELATION_GATE_SCOPE_TARGET_NODE}  "
        )
    )

    assert network.scope == (
        RELATION_GATE_SCOPE_TARGET_NODE
    )


@pytest.mark.parametrize(
    "scope",
    (
        "",
        " ",
        "\t",
    ),
)
def test_constructor_rejects_blank_scope(
    scope: str,
) -> None:
    with pytest.raises(
        ValueError,
        match="non-empty string",
    ):
        _network(scope=scope)


@pytest.mark.parametrize(
    "scope",
    (
        None,
        1,
        True,
    ),
)
def test_constructor_rejects_nonstring_scope(
    scope: Any,
) -> None:
    with pytest.raises(
        TypeError,
        match="must be a string",
    ):
        _network(scope=scope)


def test_constructor_rejects_unknown_scope() -> None:
    with pytest.raises(
        ValueError,
        match="Unknown relation-gate scope",
    ):
        _network(
            scope="unknown_scope"
        )


def test_constructor_rejects_canonical_unimplemented_scope() -> None:
    scope = _canonical_unimplemented_scope()

    if scope is None:
        pytest.skip(
            "No canonical unimplemented relation-gate scope exists."
        )

    with pytest.raises(
        NotImplementedError,
        match="canonical but not implemented",
    ):
        _network(scope=scope)


def test_implemented_scopes_are_canonical() -> None:
    assert set(
        V2_0_IMPLEMENTED_RELATION_GATE_SCOPES
    ).issubset(
        set(
            CANONICAL_RELATION_GATE_SCOPES
        )
    )
    assert RELATION_GATE_SCOPE_TARGET_NODE in (
        V2_0_IMPLEMENTED_RELATION_GATE_SCOPES
    )


# =============================================================================
# Module topology and initialization
# =============================================================================


def test_module_topology_with_all_options() -> None:
    network = _network()

    assert isinstance(
        network.input_projection,
        nn.Linear,
    )
    assert network.input_projection.in_features == (
        NODE_STATE_DIM
        + HAZARD_QUERY_DIM
    )
    assert network.input_projection.out_features == (
        HIDDEN_DIM
    )
    assert isinstance(
        network.hidden_projection,
        nn.Linear,
    )
    assert network.hidden_projection.in_features == (
        HIDDEN_DIM
    )
    assert network.hidden_projection.out_features == (
        HIDDEN_DIM
    )
    assert isinstance(
        network.context_norm,
        nn.LayerNorm,
    )
    assert network.relation_embeddings.shape == (
        RELATION_COUNT,
        HIDDEN_DIM,
    )
    assert network.relation_bias is not None
    assert network.relation_bias.shape == (
        RELATION_COUNT,
    )


def test_module_topology_without_optional_components() -> None:
    network = _network(
        layer_norm=False,
        relation_bias=False,
    )

    assert isinstance(
        network.context_norm,
        nn.Identity,
    )
    assert network.relation_bias is None
    assert "relation_bias" not in dict(
        network.named_parameters()
    )


def test_reset_parameters_sets_all_biases_to_zero() -> None:
    network = _network()

    assert torch.equal(
        network.input_projection.bias,
        torch.zeros_like(
            network.input_projection.bias
        ),
    )
    assert torch.equal(
        network.hidden_projection.bias,
        torch.zeros_like(
            network.hidden_projection.bias
        ),
    )
    assert torch.equal(
        network.relation_bias,
        torch.zeros_like(
            network.relation_bias
        ),
    )


def test_reset_parameters_sets_layer_norm_identity() -> None:
    network = _network()

    assert isinstance(
        network.context_norm,
        nn.LayerNorm,
    )
    assert torch.equal(
        network.context_norm.weight,
        torch.ones_like(
            network.context_norm.weight
        ),
    )
    assert torch.equal(
        network.context_norm.bias,
        torch.zeros_like(
            network.context_norm.bias
        ),
    )


def test_initialized_weights_are_finite_and_nontrivial() -> None:
    network = _network()

    for name in (
        "input_projection.weight",
        "hidden_projection.weight",
        "relation_embeddings",
    ):
        value = dict(
            network.named_parameters()
        )[name]
        assert bool(
            torch.isfinite(value)
            .all()
            .item()
        )
        assert not torch.equal(
            value,
            torch.zeros_like(value),
        )


def test_reset_parameters_changes_weight_values() -> None:
    network = _network()
    before = (
        network.relation_embeddings
        .detach()
        .clone()
    )

    torch.manual_seed(99)
    network.reset_parameters()

    assert not torch.equal(
        before,
        network.relation_embeddings,
    )


# =============================================================================
# Construction from config
# =============================================================================


def test_from_config_builds_aligned_network() -> None:
    inputs = _inputs(
        dtype=torch.float64
    )
    config = FakeRelationConfig(
        gate_hidden_dim=11,
        gate_enabled=True,
    )

    network = (
        RelationGateNetwork.from_config(
            config=config,
            source_inputs=inputs,
        )
    )

    assert config.validate_calls == 1
    assert (
        config.assert_implemented_calls
        == 1
    )
    assert network.node_state_dim == (
        NODE_STATE_DIM
    )
    assert network.hazard_query_dim == (
        HAZARD_QUERY_DIM
    )
    assert network.hidden_dim == 11
    assert network.scope == (
        RELATION_GATE_SCOPE_TARGET_NODE
    )
    assert network.relation_names == (
        inputs.relation_names
    )
    assert network.stable_relation_ids == (
        inputs.stable_relation_ids
    )
    first_parameter = next(
        network.parameters()
    )
    assert first_parameter.dtype == (
        torch.float64
    )
    assert first_parameter.device == (
        torch.device("cpu")
    )


def test_from_disabled_config_skips_assert_implemented() -> None:
    inputs = _inputs()
    config = FakeRelationConfig(
        gate_enabled=False
    )

    RelationGateNetwork.from_config(
        config=config,
        source_inputs=inputs,
    )

    assert config.validate_calls == 1
    assert (
        config.assert_implemented_calls
        == 0
    )


def test_from_config_rejects_wrong_config_type() -> None:
    with pytest.raises(
        TypeError,
        match="RelationConfig",
    ):
        RelationGateNetwork.from_config(
            config=object(),  # type: ignore[arg-type]
            source_inputs=_inputs(),
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
        RelationGateNetwork.from_config(
            config=config,
            source_inputs=_inputs(),
        )


def test_from_config_propagates_implementation_error() -> None:
    config = FakeRelationConfig(
        implementation_error=(
            NotImplementedError(
                "gate network unavailable"
            )
        )
    )

    with pytest.raises(
        NotImplementedError,
        match="gate network unavailable",
    ):
        RelationGateNetwork.from_config(
            config=config,
            source_inputs=_inputs(),
        )


def test_from_config_rejects_wrong_input_type() -> None:
    with pytest.raises(
        TypeError,
        match="FunctionalMessagePassingInputs",
    ):
        RelationGateNetwork.from_config(
            config=FakeRelationConfig(),
            source_inputs=object(),  # type: ignore[arg-type]
        )


def test_from_config_rejects_invalid_node_state() -> None:
    inputs = _inputs(
        node_state=SimpleNamespace(
            fused_state=torch.zeros(
                NODE_COUNT,
                NODE_STATE_DIM,
                1,
            )
        )
    )

    with pytest.raises(
        ValueError,
        match=r"shape \[N, D\]",
    ):
        RelationGateNetwork.from_config(
            config=FakeRelationConfig(),
            source_inputs=inputs,
        )


def test_from_config_requires_hazard_query_when_enabled() -> None:
    inputs = _inputs(
        node_hazard_query=None
    )

    with pytest.raises(
        ValueError,
        match="use_hazard_query=True",
    ):
        RelationGateNetwork.from_config(
            config=FakeRelationConfig(),
            source_inputs=inputs,
            use_hazard_query=True,
        )


def test_from_config_rejects_invalid_hazard_query() -> None:
    inputs = _inputs(
        node_hazard_query=torch.zeros(
            NODE_COUNT,
            HAZARD_QUERY_DIM,
            1,
        )
    )

    with pytest.raises(
        ValueError,
        match=r"shape \[N, Q\]",
    ):
        RelationGateNetwork.from_config(
            config=FakeRelationConfig(),
            source_inputs=inputs,
        )


def test_from_config_uses_placeholder_query_width_when_disabled_and_absent() -> None:
    inputs = _inputs(
        node_hazard_query=None
    )

    network = (
        RelationGateNetwork.from_config(
            config=FakeRelationConfig(),
            source_inputs=inputs,
            use_hazard_query=False,
        )
    )

    assert network.hazard_query_dim == 1
    assert not network.use_hazard_query
    assert network.input_dim == (
        NODE_STATE_DIM
    )


def test_from_config_preserves_available_query_width_when_disabled() -> None:
    inputs = _inputs()

    network = (
        RelationGateNetwork.from_config(
            config=FakeRelationConfig(),
            source_inputs=inputs,
            use_hazard_query=False,
        )
    )

    assert network.hazard_query_dim == (
        HAZARD_QUERY_DIM
    )


# =============================================================================
# Properties, parameter counts, and fingerprints
# =============================================================================


@pytest.mark.parametrize(
    (
        "use_node_state",
        "use_hazard_query",
        "expected_dim",
        "expected_names",
    ),
    (
        (
            True,
            True,
            NODE_STATE_DIM
            + HAZARD_QUERY_DIM,
            (
                "node_state",
                "hazard_query",
            ),
        ),
        (
            True,
            False,
            NODE_STATE_DIM,
            ("node_state",),
        ),
        (
            False,
            True,
            HAZARD_QUERY_DIM,
            ("hazard_query",),
        ),
    ),
)
def test_input_properties_for_feature_modes(
    use_node_state: bool,
    use_hazard_query: bool,
    expected_dim: int,
    expected_names: tuple[str, ...],
) -> None:
    network = _network(
        use_node_state=use_node_state,
        use_hazard_query=(
            use_hazard_query
        ),
    )

    assert network.input_dim == (
        expected_dim
    )
    assert network.input_feature_names == (
        expected_names
    )


@pytest.mark.parametrize(
    ("layer_norm", "relation_bias"),
    (
        (True, True),
        (True, False),
        (False, True),
        (False, False),
    ),
)
def test_parameter_count_is_exact(
    layer_norm: bool,
    relation_bias: bool,
) -> None:
    network = _network(
        layer_norm=layer_norm,
        relation_bias=relation_bias,
    )
    expected = _expected_parameter_count(
        input_dim=network.input_dim,
        hidden_dim=network.hidden_dim,
        num_relations=(
            network.num_relations
        ),
        layer_norm=layer_norm,
        relation_bias=relation_bias,
    )

    assert network.parameter_count == (
        expected
    )
    assert (
        network.trainable_parameter_count
        == expected
    )


def test_relation_score_scale_is_exact() -> None:
    network = _network(
        hidden_dim=16
    )

    assert network.relation_score_scale == (
        1.0 / math.sqrt(16.0)
    )


def test_architecture_dict_is_exact() -> None:
    network = _network()

    assert network.architecture_dict() == {
        "schema_version": (
            RELATION_GATE_NETWORK_SCHEMA_VERSION
        ),
        "scope": (
            RELATION_GATE_SCOPE_TARGET_NODE
        ),
        "node_state_dim": (
            NODE_STATE_DIM
        ),
        "hazard_query_dim": (
            HAZARD_QUERY_DIM
        ),
        "input_dim": (
            NODE_STATE_DIM
            + HAZARD_QUERY_DIM
        ),
        "hidden_dim": HIDDEN_DIM,
        "num_relations": (
            RELATION_COUNT
        ),
        "relation_names": list(
            RELATION_NAMES
        ),
        "stable_relation_ids": list(
            STABLE_RELATION_IDS
        ),
        "input_feature_names": [
            "node_state",
            "hazard_query",
        ],
        "use_node_state": True,
        "use_hazard_query": True,
        "layer_norm": True,
        "relation_bias": True,
        "context_activation": "gelu",
        "context_depth": 2,
        "relation_identity_parameterization": (
            "learned_exact_relation_embeddings"
        ),
        "relation_score_formula": (
            "dot(context, relation_embedding) / sqrt(hidden_dim) "
            "+ optional_relation_bias"
        ),
        "relation_channels_compete": False,
        "family_pooling": False,
        "parameter_count": (
            network.parameter_count
        ),
        "output_schema": (
            "GateNetworkOutput"
        ),
    }


def test_architecture_fingerprint_is_stable_across_initializations() -> None:
    first = _network(seed=1)
    second = _network(seed=2)

    assert first.architecture_dict() == (
        second.architecture_dict()
    )
    assert first.architecture_fingerprint() == (
        second.architecture_fingerprint()
    )


def test_parameter_fingerprint_is_stable_for_identical_state() -> None:
    first = _network(seed=1)
    second = _network(seed=2)
    second.load_state_dict(
        first.state_dict()
    )

    assert first.parameter_fingerprint() == (
        second.parameter_fingerprint()
    )


def test_parameter_change_changes_parameter_fingerprint() -> None:
    network = _network()
    before = (
        network.parameter_fingerprint()
    )

    with torch.no_grad():
        network.relation_embeddings[
            0,
            0,
        ] += 1.0

    assert network.parameter_fingerprint() != (
        before
    )


def test_deepcopy_preserves_parameter_fingerprint() -> None:
    network = _network()
    copied = copy.deepcopy(network)

    assert network.parameter_fingerprint() == (
        copied.parameter_fingerprint()
    )


# =============================================================================
# Finite-parameter and parameter-device guards
# =============================================================================


def test_assert_finite_parameters_accepts_valid_network() -> None:
    _network().assert_finite_parameters()


@pytest.mark.parametrize(
    "bad_value",
    (
        float("nan"),
        float("inf"),
        float("-inf"),
    ),
)
def test_assert_finite_parameters_rejects_nonfinite_parameter(
    bad_value: float,
) -> None:
    network = _network()

    with torch.no_grad():
        network.relation_embeddings[
            0,
            0,
        ] = bad_value

    with pytest.raises(
        FloatingPointError,
        match="contains NaN or infinity",
    ):
        network.assert_finite_parameters()


def test_parameter_device_dtype_returns_common_contract() -> None:
    network = _network(
        dtype=torch.float64
    )

    device, dtype = (
        network._parameter_device_dtype()
    )

    assert device == torch.device("cpu")
    assert dtype == torch.float64


def test_parameter_device_dtype_rejects_empty_parameter_set() -> None:
    network = _network()

    with patch.object(
        network,
        "parameters",
        return_value=iter(()),
    ):
        with pytest.raises(
            RuntimeError,
            match="unexpectedly has no parameters",
        ):
            network._parameter_device_dtype()


def test_parameter_device_dtype_rejects_mixed_dtypes() -> None:
    network = _network(
        dtype=torch.float32
    )
    network.relation_embeddings = (
        nn.Parameter(
            network
            .relation_embeddings
            .detach()
            .to(torch.float64)
        )
    )

    with pytest.raises(
        RuntimeError,
        match="multiple dtypes",
    ):
        network._parameter_device_dtype()


# =============================================================================
# Relation-axis alignment
# =============================================================================


def test_validate_relation_axis_accepts_exact_axis() -> None:
    inputs = _inputs()
    network = _network()

    network._validate_relation_axis(
        inputs,
        _axis(inputs=inputs),
    )


def test_validate_relation_axis_rejects_wrong_type() -> None:
    with pytest.raises(
        TypeError,
        match="RelationGateAxis",
    ):
        _network()._validate_relation_axis(
            _inputs(),
            object(),  # type: ignore[arg-type]
        )


def test_validate_relation_axis_rejects_network_relation_order_mismatch() -> None:
    inputs = _inputs()
    axis = _axis(inputs=inputs)
    network = _network(
        relation_names=(
            "temporal_lag",
            "spatial_adjacency",
            "random_placebo",
        )
    )

    with pytest.raises(
        ValueError,
        match="relation ordering differs",
    ):
        network._validate_relation_axis(
            inputs,
            axis,
        )


def test_validate_relation_axis_rejects_network_stable_id_mismatch() -> None:
    inputs = _inputs()
    axis = _axis(inputs=inputs)
    network = _network(
        stable_relation_ids=(
            100,
            201,
            900,
        )
    )

    with pytest.raises(
        ValueError,
        match="stable relation IDs differ",
    ):
        network._validate_relation_axis(
            inputs,
            axis,
        )


def test_validate_relation_axis_rejects_network_relation_count_mismatch() -> None:
    inputs = _inputs()
    axis = _axis(inputs=inputs)
    network = _network(
        relation_names=(
            "spatial_adjacency",
            "temporal_lag",
        ),
        stable_relation_ids=(
            100,
            200,
        ),
    )

    with pytest.raises(
        ValueError,
        match="relation ordering differs",
    ):
        network._validate_relation_axis(
            inputs,
            axis,
        )


# =============================================================================
# Context assembly
# =============================================================================


def test_build_context_concatenates_node_state_then_hazard_query() -> None:
    inputs = _inputs()
    network = _network()

    observed = network.build_context(
        inputs
    )
    expected = torch.cat(
        [
            inputs.node_state.fused_state,
            inputs.node_hazard_query,
        ],
        dim=-1,
    )

    assert torch.equal(
        observed,
        expected,
    )


def test_build_context_node_state_only_returns_exact_tensor() -> None:
    inputs = _inputs()
    network = _network(
        use_node_state=True,
        use_hazard_query=False,
    )

    observed = network.build_context(
        inputs
    )

    assert observed is (
        inputs.node_state.fused_state
    )


def test_build_context_hazard_query_only_returns_exact_tensor() -> None:
    inputs = _inputs()
    network = _network(
        use_node_state=False,
        use_hazard_query=True,
    )

    observed = network.build_context(
        inputs
    )

    assert observed is (
        inputs.node_hazard_query
    )


def test_build_context_rejects_wrong_input_type() -> None:
    with pytest.raises(
        TypeError,
        match="FunctionalMessagePassingInputs",
    ):
        _network().build_context(
            object()  # type: ignore[arg-type]
        )


@pytest.mark.parametrize(
    ("field_name", "value", "message"),
    (
        (
            "num_nodes",
            0,
            "at least one node",
        ),
        (
            "num_relations",
            0,
            "at least one relation",
        ),
    ),
)
def test_build_context_rejects_nonpositive_input_counts(
    field_name: str,
    value: int,
    message: str,
) -> None:
    inputs = _inputs()
    setattr(
        inputs,
        field_name,
        value,
    )

    with pytest.raises(
        ValueError,
        match=message,
    ):
        _network().build_context(
            inputs
        )


def test_build_context_rejects_nonfloating_input_dtype() -> None:
    inputs = _inputs(
        dtype=torch.long,
        node_state=SimpleNamespace(
            fused_state=torch.zeros(
                NODE_COUNT,
                NODE_STATE_DIM,
                dtype=torch.long,
            )
        ),
        node_hazard_query=torch.zeros(
            NODE_COUNT,
            HAZARD_QUERY_DIM,
            dtype=torch.long,
        ),
    )

    with pytest.raises(
        ValueError,
        match="floating-point",
    ):
        _network().build_context(
            inputs
        )


def test_build_context_rejects_nontensor_node_state() -> None:
    inputs = _inputs(
        node_state=SimpleNamespace(
            fused_state=[[0.0]]
        )
    )

    with pytest.raises(
        TypeError,
        match="fused_state must be a tensor",
    ):
        _network().build_context(
            inputs
        )


@pytest.mark.parametrize(
    "value",
    (
        torch.zeros(
            NODE_COUNT,
            NODE_STATE_DIM,
            1,
        ),
        torch.zeros(
            NODE_COUNT - 1,
            NODE_STATE_DIM,
        ),
        torch.zeros(
            NODE_COUNT,
            NODE_STATE_DIM + 1,
        ),
    ),
)
def test_build_context_rejects_invalid_node_state_shape(
    value: torch.Tensor,
) -> None:
    inputs = _inputs(
        node_state=SimpleNamespace(
            fused_state=value
        )
    )

    with pytest.raises(
        ValueError,
        match="must have shape",
    ):
        _network().build_context(
            inputs
        )


def test_build_context_rejects_node_state_dtype_mismatch() -> None:
    inputs = _inputs(
        dtype=torch.float32,
        node_state=SimpleNamespace(
            fused_state=torch.zeros(
                NODE_COUNT,
                NODE_STATE_DIM,
                dtype=torch.float64,
            )
        ),
    )

    with pytest.raises(
        ValueError,
        match="must use dtype",
    ):
        _network().build_context(
            inputs
        )


def test_build_context_rejects_nonfinite_node_state() -> None:
    value = torch.zeros(
        NODE_COUNT,
        NODE_STATE_DIM,
    )
    value[0, 0] = float("nan")
    inputs = _inputs(
        node_state=SimpleNamespace(
            fused_state=value
        )
    )

    with pytest.raises(
        ValueError,
        match="finite values",
    ):
        _network().build_context(
            inputs
        )


def test_build_context_requires_hazard_query_when_enabled() -> None:
    inputs = _inputs(
        node_hazard_query=None
    )

    with pytest.raises(
        ValueError,
        match="use_hazard_query=True",
    ):
        _network().build_context(
            inputs
        )


@pytest.mark.parametrize(
    "value",
    (
        torch.zeros(
            NODE_COUNT,
            HAZARD_QUERY_DIM,
            1,
        ),
        torch.zeros(
            NODE_COUNT - 1,
            HAZARD_QUERY_DIM,
        ),
        torch.zeros(
            NODE_COUNT,
            HAZARD_QUERY_DIM + 1,
        ),
    ),
)
def test_build_context_rejects_invalid_hazard_query_shape(
    value: torch.Tensor,
) -> None:
    inputs = _inputs(
        node_hazard_query=value
    )

    with pytest.raises(
        ValueError,
        match="must have shape",
    ):
        _network().build_context(
            inputs
        )


def test_build_context_rejects_hazard_query_dtype_mismatch() -> None:
    inputs = _inputs(
        dtype=torch.float32,
        node_hazard_query=torch.zeros(
            NODE_COUNT,
            HAZARD_QUERY_DIM,
            dtype=torch.float64,
        ),
    )

    with pytest.raises(
        ValueError,
        match="must use dtype",
    ):
        _network().build_context(
            inputs
        )


def test_build_context_rejects_nonfinite_hazard_query() -> None:
    value = torch.zeros(
        NODE_COUNT,
        HAZARD_QUERY_DIM,
    )
    value[0, 0] = float("inf")
    inputs = _inputs(
        node_hazard_query=value
    )

    with pytest.raises(
        ValueError,
        match="finite values",
    ):
        _network().build_context(
            inputs
        )


def test_build_context_defensive_no_features_guard() -> None:
    network = _network()
    network.use_node_state = False
    network.use_hazard_query = False

    with pytest.raises(
        RuntimeError,
        match="no enabled input features",
    ):
        network.build_context(
            _inputs()
        )


# =============================================================================
# Context encoding
# =============================================================================


def test_encode_context_matches_explicit_formula_with_layer_norm() -> None:
    network = _network(
        dtype=torch.float64
    )
    context = _network(
        dtype=torch.float64
    ).build_context(
        _inputs(
            dtype=torch.float64
        )
    )
    # Use the tested network's own expected-width input.
    context = _inputs(
        dtype=torch.float64
    )
    context = torch.cat(
        [
            context.node_state.fused_state,
            context.node_hazard_query,
        ],
        dim=-1,
    )

    observed = network.encode_context(
        context
    )
    expected = network.context_norm(
        F.gelu(
            network.hidden_projection(
                F.gelu(
                    network.input_projection(
                        context
                    )
                )
            )
        )
    )

    assert torch.equal(
        observed,
        expected,
    )


def test_encode_context_matches_explicit_formula_without_layer_norm() -> None:
    network = _network(
        layer_norm=False,
        dtype=torch.float64,
    )
    inputs = _inputs(
        dtype=torch.float64
    )
    context = network.build_context(
        inputs
    )

    observed = network.encode_context(
        context
    )
    expected = F.gelu(
        network.hidden_projection(
            F.gelu(
                network.input_projection(
                    context
                )
            )
        )
    )

    assert torch.equal(
        observed,
        expected,
    )


def test_encode_context_rejects_nontensor() -> None:
    with pytest.raises(
        TypeError,
        match="context must be a tensor",
    ):
        _network().encode_context(
            [[0.0]]  # type: ignore[arg-type]
        )


def test_encode_context_rejects_invalid_rank() -> None:
    with pytest.raises(
        ValueError,
        match=r"shape \[N, input_dim\]",
    ):
        _network().encode_context(
            torch.zeros(
                2,
                3,
                4,
            )
        )


def test_encode_context_rejects_invalid_width() -> None:
    with pytest.raises(
        ValueError,
        match="context width differs",
    ):
        _network().encode_context(
            torch.zeros(
                NODE_COUNT,
                NODE_STATE_DIM
                + HAZARD_QUERY_DIM
                + 1,
            )
        )


def test_encode_context_rejects_nonfloating_dtype() -> None:
    with pytest.raises(
        ValueError,
        match="floating-point dtype",
    ):
        _network().encode_context(
            torch.zeros(
                NODE_COUNT,
                NODE_STATE_DIM
                + HAZARD_QUERY_DIM,
                dtype=torch.long,
            )
        )


def test_encode_context_rejects_nonfinite_input() -> None:
    context = torch.zeros(
        NODE_COUNT,
        NODE_STATE_DIM
        + HAZARD_QUERY_DIM,
    )
    context[0, 0] = float("nan")

    with pytest.raises(
        ValueError,
        match="finite values",
    ):
        _network().encode_context(
            context
        )


def test_encode_context_rejects_parameter_dtype_mismatch() -> None:
    network = _network(
        dtype=torch.float64
    )
    context = torch.zeros(
        NODE_COUNT,
        network.input_dim,
        dtype=torch.float32,
    )

    with pytest.raises(
        ValueError,
        match="must use one dtype",
    ):
        network.encode_context(
            context
        )


def test_encode_context_rejects_internal_shape_change(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    network = _network()

    monkeypatch.setattr(
        network.context_norm,
        "forward",
        lambda value: torch.zeros(
            value.shape[0],
            value.shape[1] + 1,
            dtype=value.dtype,
            device=value.device,
        ),
    )

    with pytest.raises(
        RuntimeError,
        match="has shape",
    ):
        network.encode_context(
            torch.zeros(
                NODE_COUNT,
                network.input_dim,
            )
        )


def test_encode_context_rejects_internal_dtype_change(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    network = _network()

    monkeypatch.setattr(
        network.context_norm,
        "forward",
        lambda value: value.to(
            torch.float64
        ),
    )

    with pytest.raises(
        RuntimeError,
        match="changed dtype",
    ):
        network.encode_context(
            torch.zeros(
                NODE_COUNT,
                network.input_dim,
            )
        )


def test_encode_context_rejects_internal_nonfinite_output(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    network = _network()

    def fake_norm(
        value: torch.Tensor,
    ) -> torch.Tensor:
        result = value.clone()
        result[0, 0] = float("nan")
        return result

    monkeypatch.setattr(
        network.context_norm,
        "forward",
        fake_norm,
    )

    with pytest.raises(
        FloatingPointError,
        match="NaN or infinity",
    ):
        network.encode_context(
            torch.zeros(
                NODE_COUNT,
                network.input_dim,
            )
        )


# =============================================================================
# Exact relation scoring
# =============================================================================


def test_score_relations_matches_exact_formula_with_bias() -> None:
    network = _network(
        hidden_dim=4,
        dtype=torch.float64,
    )

    with torch.no_grad():
        network.relation_embeddings.copy_(
            torch.tensor(
                [
                    [1.0, 0.0, 0.0, 0.0],
                    [0.0, 2.0, 0.0, 0.0],
                    [0.0, 0.0, 3.0, 0.0],
                ],
                dtype=torch.float64,
            )
        )
        network.relation_bias.copy_(
            torch.tensor(
                [0.1, 0.2, 0.3],
                dtype=torch.float64,
            )
        )

    encoded = torch.tensor(
        [
            [2.0, 3.0, 4.0, 5.0],
            [1.0, -1.0, 2.0, 0.0],
        ],
        dtype=torch.float64,
    )

    observed = network.score_relations(
        encoded
    )
    expected = (
        encoded
        @ network
        .relation_embeddings
        .transpose(0, 1)
    ) / 2.0
    expected = (
        expected
        + network.relation_bias
    )

    assert torch.equal(
        observed,
        expected,
    )


def test_score_relations_matches_exact_formula_without_bias() -> None:
    network = _network(
        hidden_dim=4,
        relation_bias=False,
        dtype=torch.float64,
    )
    encoded = torch.randn(
        2,
        4,
        dtype=torch.float64,
    )

    observed = network.score_relations(
        encoded
    )
    expected = (
        encoded
        @ network
        .relation_embeddings
        .transpose(0, 1)
    ) / 2.0

    assert torch.equal(
        observed,
        expected,
    )


def test_changing_one_relation_embedding_changes_only_its_column() -> None:
    network = _network(
        layer_norm=False,
        relation_bias=False,
    )
    encoded = torch.randn(
        NODE_COUNT,
        HIDDEN_DIM,
    )

    before = network.score_relations(
        encoded
    )

    with torch.no_grad():
        network.relation_embeddings[
            1
        ] += 1.0

    after = network.score_relations(
        encoded
    )

    assert torch.equal(
        before[:, 0],
        after[:, 0],
    )
    assert not torch.equal(
        before[:, 1],
        after[:, 1],
    )
    assert torch.equal(
        before[:, 2],
        after[:, 2],
    )


def test_changing_one_context_row_changes_only_its_logit_row() -> None:
    network = _network(
        relation_bias=False
    )
    encoded = torch.randn(
        NODE_COUNT,
        HIDDEN_DIM,
    )

    before = network.score_relations(
        encoded
    )
    changed = encoded.clone()
    changed[2] += 1.0
    after = network.score_relations(
        changed
    )

    assert torch.equal(
        before[:2],
        after[:2],
    )
    assert not torch.equal(
        before[2],
        after[2],
    )
    assert torch.equal(
        before[3:],
        after[3:],
    )


def test_same_family_relations_remain_distinct_channels() -> None:
    network = _network(
        hidden_dim=3,
        relation_bias=False,
    )

    with torch.no_grad():
        network.relation_embeddings.copy_(
            torch.eye(
                3,
                dtype=network
                .relation_embeddings
                .dtype,
            )
        )

    encoded = torch.tensor(
        [[1.0, 2.0, 3.0]]
    )
    logits = network.score_relations(
        encoded
    )

    # Relations 0 and 1 share one semantic family in the fixture, but their
    # exact relation channels remain distinct.
    assert logits[0, 0] != logits[0, 1]


def test_score_relations_rejects_nontensor() -> None:
    with pytest.raises(
        TypeError,
        match="encoded_context must be a tensor",
    ):
        _network().score_relations(
            [[0.0]]  # type: ignore[arg-type]
        )


def test_score_relations_rejects_invalid_rank() -> None:
    with pytest.raises(
        ValueError,
        match=r"shape \[N, hidden_dim\]",
    ):
        _network().score_relations(
            torch.zeros(
                2,
                3,
                4,
            )
        )


def test_score_relations_rejects_invalid_width() -> None:
    with pytest.raises(
        ValueError,
        match="width differs",
    ):
        _network().score_relations(
            torch.zeros(
                NODE_COUNT,
                HIDDEN_DIM + 1,
            )
        )


def test_score_relations_rejects_nonfloating_dtype() -> None:
    with pytest.raises(
        ValueError,
        match="floating-point dtype",
    ):
        _network().score_relations(
            torch.zeros(
                NODE_COUNT,
                HIDDEN_DIM,
                dtype=torch.long,
            )
        )


def test_score_relations_rejects_nonfinite_input() -> None:
    encoded = torch.zeros(
        NODE_COUNT,
        HIDDEN_DIM,
    )
    encoded[0, 0] = float("inf")

    with pytest.raises(
        ValueError,
        match="finite values",
    ):
        _network().score_relations(
            encoded
        )


def test_score_relations_rejects_parameter_dtype_mismatch() -> None:
    network = _network(
        dtype=torch.float64
    )

    with pytest.raises(
        ValueError,
        match="must use one dtype",
    ):
        network.score_relations(
            torch.zeros(
                NODE_COUNT,
                HIDDEN_DIM,
                dtype=torch.float32,
            )
        )


def test_score_relations_rejects_internal_shape_change(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    network = _network(
        relation_bias=False
    )

    monkeypatch.setattr(
        gate_network_module.torch,
        "matmul",
        lambda first, second: torch.zeros(
            first.shape[0],
            RELATION_COUNT + 1,
            dtype=first.dtype,
            device=first.device,
        ),
    )

    with pytest.raises(
        RuntimeError,
        match="has shape",
    ):
        network.score_relations(
            torch.zeros(
                NODE_COUNT,
                HIDDEN_DIM,
            )
        )


def test_score_relations_rejects_internal_dtype_change(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    network = _network(
        relation_bias=False
    )

    monkeypatch.setattr(
        gate_network_module.torch,
        "matmul",
        lambda first, second: torch.zeros(
            first.shape[0],
            RELATION_COUNT,
            dtype=torch.float64,
            device=first.device,
        ),
    )

    with pytest.raises(
        RuntimeError,
        match="changed dtype",
    ):
        network.score_relations(
            torch.zeros(
                NODE_COUNT,
                HIDDEN_DIM,
            )
        )


def test_score_relations_rejects_internal_nonfinite_output(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    network = _network(
        relation_bias=False
    )

    def fake_matmul(
        first: torch.Tensor,
        second: torch.Tensor,
    ) -> torch.Tensor:
        result = torch.zeros(
            first.shape[0],
            RELATION_COUNT,
            dtype=first.dtype,
            device=first.device,
        )
        result[0, 0] = float("nan")
        return result

    monkeypatch.setattr(
        gate_network_module.torch,
        "matmul",
        fake_matmul,
    )

    with pytest.raises(
        FloatingPointError,
        match="NaN or infinity",
    ):
        network.score_relations(
            torch.zeros(
                NODE_COUNT,
                HIDDEN_DIM,
            )
        )


# =============================================================================
# Forward output
# =============================================================================


def test_forward_constructs_complete_output() -> None:
    inputs = _inputs()
    axis = _axis(inputs=inputs)
    network = _network()

    output = network(
        inputs,
        axis=axis,
    )

    assert isinstance(
        output,
        GateNetworkOutput,
    )
    assert output.source_inputs is inputs
    assert output.axis is axis
    assert output.logits.shape == (
        NODE_COUNT,
        RELATION_COUNT,
    )
    assert output.logits.dtype == (
        inputs.dtype
    )
    assert output.logits.device == (
        inputs.device
    )
    assert output.scope == (
        RELATION_GATE_SCOPE_TARGET_NODE
    )
    assert output.input_feature_names == (
        "node_state",
        "hazard_query",
    )
    assert (
        output.encoder_architecture_fingerprint
        == network.architecture_fingerprint()
    )
    assert (
        output.parameter_fingerprint
        == network.parameter_fingerprint()
    )


def test_forward_builds_axis_when_absent() -> None:
    inputs = _inputs()
    output = _network()(inputs)

    assert isinstance(
        output.axis,
        RelationGateAxis,
    )
    output.axis.assert_matches_inputs(
        inputs
    )


def test_forward_is_deterministic() -> None:
    inputs = _inputs()
    network = _network()

    first = network(inputs)
    second = network(inputs)

    assert torch.equal(
        first.logits,
        second.logits,
    )
    assert (
        first.encoder_architecture_fingerprint
        == second.encoder_architecture_fingerprint
    )
    assert (
        first.parameter_fingerprint
        == second.parameter_fingerprint
    )


@pytest.mark.parametrize(
    (
        "use_node_state",
        "use_hazard_query",
        "expected_names",
    ),
    (
        (
            True,
            False,
            ("node_state",),
        ),
        (
            False,
            True,
            ("hazard_query",),
        ),
    ),
)
def test_forward_feature_modes(
    use_node_state: bool,
    use_hazard_query: bool,
    expected_names: tuple[str, ...],
) -> None:
    inputs = _inputs()
    network = _network(
        use_node_state=use_node_state,
        use_hazard_query=(
            use_hazard_query
        ),
    )

    output = network(inputs)

    assert output.logits.shape == (
        NODE_COUNT,
        RELATION_COUNT,
    )
    assert output.input_feature_names == (
        expected_names
    )


def test_forward_rejects_nonfinite_parameter_before_prediction() -> None:
    network = _network()

    with torch.no_grad():
        network.relation_embeddings[
            0,
            0,
        ] = float("nan")

    with pytest.raises(
        FloatingPointError,
        match="parameter",
    ):
        network(_inputs())


# =============================================================================
# Gradients
# =============================================================================


def test_forward_backpropagates_to_both_enabled_inputs_and_parameters() -> None:
    node_state = torch.randn(
        NODE_COUNT,
        NODE_STATE_DIM,
        requires_grad=True,
    )
    hazard_query = torch.randn(
        NODE_COUNT,
        HAZARD_QUERY_DIM,
        requires_grad=True,
    )
    inputs = _inputs(
        node_state=SimpleNamespace(
            fused_state=node_state
        ),
        node_hazard_query=(
            hazard_query
        ),
    )
    network = _network()

    output = network(inputs)
    output.logits.square().mean().backward()

    assert node_state.grad is not None
    assert hazard_query.grad is not None
    assert bool(
        torch.isfinite(node_state.grad)
        .all()
        .item()
    )
    assert bool(
        torch.isfinite(hazard_query.grad)
        .all()
        .item()
    )

    for parameter in network.parameters():
        assert parameter.grad is not None
        assert bool(
            torch.isfinite(parameter.grad)
            .all()
            .item()
        )


def test_node_state_only_does_not_require_hazard_query() -> None:
    node_state = torch.randn(
        NODE_COUNT,
        NODE_STATE_DIM,
        requires_grad=True,
    )
    inputs = _inputs(
        node_state=SimpleNamespace(
            fused_state=node_state
        ),
        node_hazard_query=None,
    )
    network = _network(
        use_hazard_query=False
    )

    network(
        inputs
    ).logits.sum().backward()

    assert node_state.grad is not None


def test_hazard_only_does_not_read_node_state() -> None:
    hazard_query = torch.randn(
        NODE_COUNT,
        HAZARD_QUERY_DIM,
        requires_grad=True,
    )
    inputs = _inputs(
        node_state=object(),
        node_hazard_query=(
            hazard_query
        ),
    )
    network = _network(
        use_node_state=False
    )

    network(
        inputs
    ).logits.sum().backward()

    assert hazard_query.grad is not None


def test_score_relations_gradcheck() -> None:
    network = _network(
        dtype=torch.float64,
        relation_bias=True,
    )
    encoded = torch.randn(
        3,
        HIDDEN_DIM,
        dtype=torch.float64,
        requires_grad=True,
    )

    assert torch.autograd.gradcheck(
        network.score_relations,
        (encoded,),
        eps=1e-6,
        atol=1e-5,
        rtol=1e-3,
    )


# =============================================================================
# Semantic device helper
# =============================================================================


def test_devices_match_cpu() -> None:
    assert gate_network_module._devices_match(
        "cpu",
        torch.device("cpu"),
    )


def test_devices_match_rejects_cpu_cuda() -> None:
    assert not gate_network_module._devices_match(
        "cpu",
        "cuda:0",
    )


def test_devices_match_explicit_cuda_indices() -> None:
    assert gate_network_module._devices_match(
        "cuda:0",
        "cuda:0",
    )
    assert not gate_network_module._devices_match(
        "cuda:0",
        "cuda:1",
    )


def test_devices_match_resolves_implicit_cuda_index() -> None:
    with patch.object(
        torch.cuda,
        "current_device",
        return_value=0,
    ):
        assert gate_network_module._devices_match(
            "cuda",
            "cuda:0",
        )
        assert gate_network_module._devices_match(
            "cuda:0",
            "cuda",
        )
        assert not gate_network_module._devices_match(
            "cuda",
            "cuda:1",
        )


# =============================================================================
# Representation
# =============================================================================


def test_extra_repr_contains_architecture_identity() -> None:
    representation = (
        _network().extra_repr()
    )

    for text in (
        f"node_state_dim={NODE_STATE_DIM}",
        f"hazard_query_dim={HAZARD_QUERY_DIM}",
        f"hidden_dim={HIDDEN_DIM}",
        f"num_relations={RELATION_COUNT}",
        "scope='target_node'",
        "use_node_state=True",
        "use_hazard_query=True",
        "layer_norm=True",
        "relation_bias=True",
    ):
        assert text in representation


# =============================================================================
# Optional CUDA
# =============================================================================


@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA is unavailable.",
)
def test_cuda_forward_accepts_implicit_device_metadata() -> None:
    inputs = _inputs(
        device=torch.device("cuda")
    )
    network = _network(
        device="cuda"
    )

    output = network(inputs)

    assert output.logits.device.type == (
        "cuda"
    )
    assert output.logits.shape == (
        NODE_COUNT,
        RELATION_COUNT,
    )


@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA is unavailable.",
)
def test_cuda_results_match_cpu_for_identical_state() -> None:
    cpu_inputs = _inputs()
    cuda_inputs = _inputs(
        device=torch.device("cuda")
    )
    cpu_network = _network()
    cuda_network = _network(
        device="cuda"
    )
    cuda_network.load_state_dict(
        cpu_network.state_dict()
    )

    cpu_output = cpu_network(
        cpu_inputs
    )
    cuda_output = cuda_network(
        cuda_inputs
    )

    assert torch.allclose(
        cpu_output.logits,
        cuda_output.logits.cpu(),
        atol=1e-5,
        rtol=1e-5,
    )


@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA is unavailable.",
)
def test_cuda_backward_is_finite() -> None:
    node_state = torch.randn(
        NODE_COUNT,
        NODE_STATE_DIM,
        device="cuda",
        requires_grad=True,
    )
    hazard_query = torch.randn(
        NODE_COUNT,
        HAZARD_QUERY_DIM,
        device="cuda",
        requires_grad=True,
    )
    inputs = _inputs(
        device=torch.device("cuda"),
        node_state=SimpleNamespace(
            fused_state=node_state
        ),
        node_hazard_query=(
            hazard_query
        ),
    )
    network = _network(
        device="cuda"
    )

    network(
        inputs
    ).logits.square().mean().backward()

    assert node_state.grad is not None
    assert hazard_query.grad is not None
    assert bool(
        torch.isfinite(node_state.grad)
        .all()
        .item()
    )
    assert bool(
        torch.isfinite(hazard_query.grad)
        .all()
        .item()
    )


@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA is unavailable.",
)
def test_cuda_rejects_cpu_node_state_against_cuda_metadata() -> None:
    inputs = _inputs(
        device=torch.device("cuda"),
        node_state=SimpleNamespace(
            fused_state=torch.zeros(
                NODE_COUNT,
                NODE_STATE_DIM,
                device="cpu",
            )
        ),
    )
    network = _network(
        device="cuda"
    )

    with pytest.raises(
        ValueError,
        match="must share",
    ):
        network.build_context(
            inputs
        )


@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA is unavailable.",
)
def test_cuda_rejects_cpu_context_against_cuda_parameters() -> None:
    network = _network(
        device="cuda"
    )
    context = torch.zeros(
        NODE_COUNT,
        network.input_dim,
        device="cpu",
    )

    with pytest.raises(
        ValueError,
        match="must share one device",
    ):
        network.encode_context(
            context
        )


@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA is unavailable.",
)
def test_cuda_rejects_cpu_encoded_context_against_cuda_parameters() -> None:
    network = _network(
        device="cuda"
    )
    encoded = torch.zeros(
        NODE_COUNT,
        HIDDEN_DIM,
        device="cpu",
    )

    with pytest.raises(
        ValueError,
        match="must share one device",
    ):
        network.score_relations(
            encoded
        )
