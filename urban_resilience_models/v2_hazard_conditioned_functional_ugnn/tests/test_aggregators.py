"""
Contract tests for target-node message aggregation.

Target repository path:
    urban_resilience_models/
        v2_hazard_conditioned_functional_ugnn/
            tests/
                test_aggregators.py

Implementation under test:
    urban_resilience_models/
        v2_hazard_conditioned_functional_ugnn/
            functional_message_passing/
                aggregators.py

This suite freezes the bounded V2.0 aggregation contract independently from
relation transforms, structural normalization, semantic edge weights,
hazard-conditioned gates, edge attention, residual updates, and layer
normalization.

Covered behavior:

- public alias and schema identity;
- canonical versus implemented aggregation modes;
- construction from functional-message-passing configuration;
- exact target-node incoming counts, sums, and means;
- denominator semantics based on retained incoming edge count;
- parallel edges, self-loops, repeated relations, isolated nodes, and
  zero-edge graphs;
- invariance to edge ordering and unrelated relation/hazard/attention metadata;
- parameter-free architecture and fingerprints;
- metadata-preserving ``AggregationOutput`` construction;
- strict message, target-index, dtype, device, shape, range, and finiteness
  validation;
- internal grouped-reduction output guards;
- finite autograd, exact mean gradients, and double-precision gradcheck;
- optional CUDA parity and device-failure checks.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pytest
import torch
from torch import nn

from urban_resilience_models.v2_hazard_conditioned_functional_ugnn.constants import (
    AGGREGATION_MEAN,
    CANONICAL_AGGREGATION_TYPES,
    V2_0_IMPLEMENTED_AGGREGATION_TYPES,
)
from urban_resilience_models.v2_hazard_conditioned_functional_ugnn.functional_message_passing import (
    aggregators as aggregators_module,
)
from urban_resilience_models.v2_hazard_conditioned_functional_ugnn.functional_message_passing.aggregators import (
    AGGREGATOR_SCHEMA_VERSION,
    Aggregator,
    MessageAggregator,
)


NODE_COUNT = 6
HIDDEN_DIM = 3
EDGE_COUNT = 8


# =============================================================================
# Controlled upstream contracts
# =============================================================================


class FakeFunctionalMessagePassingConfig:
    def __init__(
        self,
        *,
        aggregation_type: str = AGGREGATION_MEAN,
        enabled: bool = True,
        validation_error: Exception | None = None,
        implementation_error: Exception | None = None,
    ) -> None:
        self.aggregation_type = aggregation_type
        self.enabled = enabled
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


class FakeInputs:
    def __init__(
        self,
        *,
        num_nodes: int,
        hidden_dim: int,
        num_edges: int,
        target_index: torch.Tensor,
        device: torch.device,
        dtype: torch.dtype,
    ) -> None:
        self.num_nodes = num_nodes
        self.hidden_dim = hidden_dim
        self.num_edges = num_edges
        self.target_index = target_index
        self.device = device
        self.dtype = dtype


class FakeEdgeMessageOutput:
    def __init__(
        self,
        *,
        edge_messages: torch.Tensor,
        source_inputs: FakeInputs,
    ) -> None:
        self.edge_messages = edge_messages
        self.source_inputs = source_inputs


@dataclass
class FakeAggregationOutput:
    node_aggregate: torch.Tensor
    incoming_edge_count: torch.Tensor
    source_messages: object
    aggregation_mode: str
    encoder_architecture_fingerprint: str


@pytest.fixture(autouse=True)
def _patch_upstream_types(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        aggregators_module,
        "FunctionalMessagePassingConfig",
        FakeFunctionalMessagePassingConfig,
    )
    monkeypatch.setattr(
        aggregators_module,
        "EdgeMessageOutput",
        FakeEdgeMessageOutput,
    )
    monkeypatch.setattr(
        aggregators_module,
        "AggregationOutput",
        FakeAggregationOutput,
    )


# =============================================================================
# Helpers
# =============================================================================


def _target_index(
    *,
    device: torch.device | str = "cpu",
) -> torch.Tensor:
    # Node 1 receives two parallel logical messages.
    # Node 2 receives a self-loop message.
    # Node 3 receives three messages.
    # Node 4 receives two messages.
    # Nodes 0 and 5 are isolated targets.
    return torch.tensor(
        [1, 1, 2, 3, 3, 3, 4, 4],
        dtype=torch.long,
        device=device,
    )


def _edge_messages(
    *,
    dtype: torch.dtype = torch.float32,
    device: torch.device | str = "cpu",
    requires_grad: bool = False,
) -> torch.Tensor:
    values = torch.tensor(
        [
            [1.0, 2.0, 3.0],
            [3.0, 4.0, 5.0],
            [2.0, 5.0, 8.0],
            [2.0, 4.0, 6.0],
            [4.0, 6.0, 8.0],
            [6.0, 8.0, 10.0],
            [10.0, 20.0, 30.0],
            [14.0, 24.0, 34.0],
        ],
        dtype=dtype,
        device=device,
    )
    values.requires_grad_(requires_grad)
    return values


def _messages(
    *,
    edge_messages: torch.Tensor | None = None,
    target_index: torch.Tensor | None = None,
    num_nodes: int = NODE_COUNT,
    hidden_dim: int = HIDDEN_DIM,
    num_edges: int | None = None,
    device: torch.device | None = None,
    dtype: torch.dtype | None = None,
) -> FakeEdgeMessageOutput:
    resolved_messages = (
        _edge_messages()
        if edge_messages is None
        else edge_messages
    )
    resolved_target = (
        _target_index(
            device=resolved_messages.device
        )
        if target_index is None
        else target_index
    )
    resolved_num_edges = (
        int(resolved_messages.shape[0])
        if num_edges is None
        else num_edges
    )
    inputs = FakeInputs(
        num_nodes=num_nodes,
        hidden_dim=hidden_dim,
        num_edges=resolved_num_edges,
        target_index=resolved_target,
        device=(
            resolved_messages.device
            if device is None
            else device
        ),
        dtype=(
            resolved_messages.dtype
            if dtype is None
            else dtype
        ),
    )
    return FakeEdgeMessageOutput(
        edge_messages=resolved_messages,
        source_inputs=inputs,
    )


def _expected_counts(
    *,
    device: torch.device | str = "cpu",
) -> torch.Tensor:
    return torch.tensor(
        [0, 2, 1, 3, 2, 0],
        dtype=torch.long,
        device=device,
    )


def _expected_sum(
    *,
    dtype: torch.dtype = torch.float32,
    device: torch.device | str = "cpu",
) -> torch.Tensor:
    return torch.tensor(
        [
            [0.0, 0.0, 0.0],
            [4.0, 6.0, 8.0],
            [2.0, 5.0, 8.0],
            [12.0, 18.0, 24.0],
            [24.0, 44.0, 64.0],
            [0.0, 0.0, 0.0],
        ],
        dtype=dtype,
        device=device,
    )


def _expected_mean(
    *,
    dtype: torch.dtype = torch.float32,
    device: torch.device | str = "cpu",
) -> torch.Tensor:
    return torch.tensor(
        [
            [0.0, 0.0, 0.0],
            [2.0, 3.0, 4.0],
            [2.0, 5.0, 8.0],
            [4.0, 6.0, 8.0],
            [12.0, 22.0, 32.0],
            [0.0, 0.0, 0.0],
        ],
        dtype=dtype,
        device=device,
    )


def _canonical_unimplemented_mode() -> str | None:
    for mode in CANONICAL_AGGREGATION_TYPES:
        if mode not in V2_0_IMPLEMENTED_AGGREGATION_TYPES:
            return mode
    return None


# =============================================================================
# Public identity and constructor
# =============================================================================


def test_schema_version_is_nonempty() -> None:
    assert isinstance(
        AGGREGATOR_SCHEMA_VERSION,
        str,
    )
    assert AGGREGATOR_SCHEMA_VERSION.strip()


def test_public_alias_points_to_message_aggregator() -> None:
    assert Aggregator is MessageAggregator


def test_class_is_torch_module() -> None:
    assert issubclass(
        MessageAggregator,
        nn.Module,
    )


def test_default_constructor_selects_mean() -> None:
    aggregator = MessageAggregator()

    assert aggregator.mode == AGGREGATION_MEAN
    assert aggregator.is_mean


def test_constructor_accepts_mean() -> None:
    aggregator = MessageAggregator(
        mode=AGGREGATION_MEAN
    )

    assert aggregator.mode == AGGREGATION_MEAN


def test_constructor_strips_whitespace() -> None:
    aggregator = MessageAggregator(
        mode=f"  {AGGREGATION_MEAN}  "
    )

    assert aggregator.mode == AGGREGATION_MEAN


@pytest.mark.parametrize(
    "mode",
    (
        "",
        " ",
        "\t",
    ),
)
def test_constructor_rejects_blank_mode(
    mode: str,
) -> None:
    with pytest.raises(
        ValueError,
        match="non-empty string",
    ):
        MessageAggregator(mode=mode)


@pytest.mark.parametrize(
    "mode",
    (
        None,
        1,
        True,
        object(),
    ),
)
def test_constructor_rejects_nonstring_mode(
    mode: Any,
) -> None:
    with pytest.raises(
        TypeError,
        match="must be a string",
    ):
        MessageAggregator(mode=mode)


def test_constructor_rejects_unknown_mode() -> None:
    with pytest.raises(
        ValueError,
        match="Unknown aggregation mode",
    ):
        MessageAggregator(
            mode="unknown_aggregation"
        )


def test_constructor_rejects_canonical_unimplemented_mode() -> None:
    mode = _canonical_unimplemented_mode()

    if mode is None:
        pytest.skip(
            "No canonical unimplemented aggregation mode exists."
        )

    with pytest.raises(
        NotImplementedError,
        match="canonical but not implemented",
    ):
        MessageAggregator(mode=mode)


def test_implemented_modes_are_canonical() -> None:
    assert set(
        V2_0_IMPLEMENTED_AGGREGATION_TYPES
    ).issubset(
        set(CANONICAL_AGGREGATION_TYPES)
    )
    assert AGGREGATION_MEAN in (
        V2_0_IMPLEMENTED_AGGREGATION_TYPES
    )


# =============================================================================
# Construction from config
# =============================================================================


def test_from_config_builds_mean_aggregator() -> None:
    config = FakeFunctionalMessagePassingConfig(
        aggregation_type=AGGREGATION_MEAN,
        enabled=True,
    )

    aggregator = MessageAggregator.from_config(
        config=config
    )

    assert config.validate_calls == 1
    assert config.assert_implemented_calls == 1
    assert aggregator.mode == AGGREGATION_MEAN


def test_from_disabled_config_does_not_assert_implemented() -> None:
    config = FakeFunctionalMessagePassingConfig(
        aggregation_type=AGGREGATION_MEAN,
        enabled=False,
    )

    aggregator = MessageAggregator.from_config(
        config=config
    )

    assert config.validate_calls == 1
    assert config.assert_implemented_calls == 0
    assert aggregator.is_mean


def test_from_config_rejects_wrong_type() -> None:
    with pytest.raises(
        TypeError,
        match="FunctionalMessagePassingConfig",
    ):
        MessageAggregator.from_config(
            config=object()  # type: ignore[arg-type]
        )


def test_from_config_propagates_validation_error() -> None:
    config = FakeFunctionalMessagePassingConfig(
        validation_error=RuntimeError(
            "invalid config"
        )
    )

    with pytest.raises(
        RuntimeError,
        match="invalid config",
    ):
        MessageAggregator.from_config(
            config=config
        )


def test_from_config_propagates_implementation_error() -> None:
    config = FakeFunctionalMessagePassingConfig(
        implementation_error=(
            NotImplementedError(
                "aggregation unavailable"
            )
        )
    )

    with pytest.raises(
        NotImplementedError,
        match="aggregation unavailable",
    ):
        MessageAggregator.from_config(
            config=config
        )


# =============================================================================
# Parameter-free identity and fingerprints
# =============================================================================


def test_aggregator_is_parameter_free() -> None:
    aggregator = MessageAggregator()

    assert aggregator.parameter_count == 0
    assert aggregator.trainable_parameter_count == 0
    assert tuple(aggregator.parameters()) == ()
    assert aggregator.state_dict() == {}


def test_architecture_dict_is_exact() -> None:
    aggregator = MessageAggregator()

    assert aggregator.architecture_dict() == {
        "schema_version": AGGREGATOR_SCHEMA_VERSION,
        "mode": AGGREGATION_MEAN,
        "implemented_formula": (
            "mean_messages_by_target_node"
        ),
        "parameter_count": 0,
        "grouping_axis": (
            "source_inputs.target_index"
        ),
        "num_segments": (
            "source_inputs.num_nodes"
        ),
        "denominator": (
            "retained_incoming_edge_count"
        ),
        "isolated_node_policy": (
            "exact_zero"
        ),
        "relation_agnostic": True,
        "relation_family_agnostic": True,
        "hazard_agnostic": True,
        "graph_agnostic_after_edge_validation": True,
        "attention_group_agnostic": True,
        "operation_order": [
            "validate_edge_message_output",
            "count_edges_by_target_node",
            "sum_messages_by_target_node",
            "divide_nonempty_nodes_by_incoming_edge_count",
            "preserve_exact_zero_for_isolated_nodes",
            "construct_metadata_preserving_output",
        ],
        "output_schema": "AggregationOutput",
    }


def test_architecture_fingerprint_is_stable() -> None:
    first = MessageAggregator()
    second = MessageAggregator()

    assert first.architecture_dict() == (
        second.architecture_dict()
    )
    assert first.architecture_fingerprint() == (
        second.architecture_fingerprint()
    )


def test_parameter_fingerprint_is_stable() -> None:
    first = MessageAggregator()
    second = MessageAggregator()

    assert first.parameter_fingerprint() == (
        second.parameter_fingerprint()
    )


def test_architecture_and_parameter_fingerprints_are_distinct() -> None:
    aggregator = MessageAggregator()

    assert (
        aggregator.architecture_fingerprint()
        != aggregator.parameter_fingerprint()
    )


def test_assert_finite_parameters_is_noop() -> None:
    MessageAggregator().assert_finite_parameters()


def test_assert_finite_parameters_rejects_nonzero_parameter_contract() -> None:
    class InvalidAggregator(
        MessageAggregator
    ):
        @property
        def parameter_count(self) -> int:
            return 1

    with pytest.raises(
        RuntimeError,
        match="must remain parameter-free",
    ):
        InvalidAggregator().assert_finite_parameters()


# =============================================================================
# Incoming counts
# =============================================================================


def test_compute_incoming_edge_count_exact_values() -> None:
    messages = _messages()

    counts = (
        MessageAggregator()
        .compute_incoming_edge_count(
            messages
        )
    )

    assert torch.equal(
        counts,
        _expected_counts(),
    )


def test_incoming_counts_have_exact_contract() -> None:
    messages = _messages()
    counts = (
        MessageAggregator()
        .compute_incoming_edge_count(
            messages
        )
    )

    assert counts.shape == (NODE_COUNT,)
    assert counts.dtype == torch.long
    assert counts.device == (
        messages.source_inputs.device
    )
    assert int(counts.sum().item()) == (
        EDGE_COUNT
    )


def test_parallel_messages_are_counted_individually() -> None:
    counts = (
        MessageAggregator()
        .compute_incoming_edge_count(
            _messages()
        )
    )

    assert counts[1].item() == 2


def test_single_self_loop_message_counts_once() -> None:
    counts = (
        MessageAggregator()
        .compute_incoming_edge_count(
            _messages()
        )
    )

    assert counts[2].item() == 1


def test_isolated_nodes_have_zero_incoming_count() -> None:
    counts = (
        MessageAggregator()
        .compute_incoming_edge_count(
            _messages()
        )
    )

    assert counts[0].item() == 0
    assert counts[5].item() == 0


# =============================================================================
# Target-node sums
# =============================================================================


def test_compute_node_sum_exact_values() -> None:
    observed = (
        MessageAggregator()
        .compute_node_sum(
            _messages()
        )
    )

    assert torch.equal(
        observed,
        _expected_sum(),
    )


@pytest.mark.parametrize(
    "dtype",
    (
        torch.float32,
        torch.float64,
    ),
)
def test_node_sum_preserves_dtype_and_device(
    dtype: torch.dtype,
) -> None:
    values = _edge_messages(dtype=dtype)
    messages = _messages(
        edge_messages=values,
        dtype=dtype,
    )

    observed = (
        MessageAggregator()
        .compute_node_sum(messages)
    )

    assert observed.dtype == dtype
    assert observed.device == values.device
    assert torch.equal(
        observed,
        _expected_sum(dtype=dtype),
    )


def test_node_sum_is_exact_zero_for_isolated_nodes() -> None:
    observed = (
        MessageAggregator()
        .compute_node_sum(
            _messages()
        )
    )

    assert torch.equal(
        observed[
            torch.tensor(
                [True, False, False, False, False, True]
            )
        ],
        torch.zeros(2, HIDDEN_DIM),
    )


# =============================================================================
# Target-node means
# =============================================================================


def test_compute_node_mean_exact_values() -> None:
    observed = (
        MessageAggregator()
        .compute_node_mean(
            _messages()
        )
    )

    assert torch.equal(
        observed,
        _expected_mean(),
    )


def test_mean_denominator_is_retained_incoming_edge_count() -> None:
    messages = _messages()
    aggregator = MessageAggregator()

    node_sum = aggregator.compute_node_sum(
        messages
    )
    counts = (
        aggregator.compute_incoming_edge_count(
            messages
        )
    )
    node_mean = aggregator.compute_node_mean(
        messages
    )

    present = counts > 0
    expected = (
        node_sum[present]
        / counts[present]
        .to(node_sum.dtype)
        .unsqueeze(-1)
    )

    assert torch.equal(
        node_mean[present],
        expected,
    )


def test_mean_is_not_global_edge_mean() -> None:
    messages = _messages()
    observed = (
        MessageAggregator()
        .compute_node_mean(messages)
    )
    global_mean = (
        messages.edge_messages.mean(dim=0)
    )

    assert not torch.equal(
        observed[1],
        global_mean,
    )
    assert not torch.equal(
        observed[3],
        global_mean,
    )


def test_single_incoming_message_is_unchanged() -> None:
    messages = _messages()
    observed = (
        MessageAggregator()
        .compute_node_mean(messages)
    )

    # Only edge index 2 targets node 2.
    assert torch.equal(
        observed[2],
        messages.edge_messages[2],
    )


def test_isolated_nodes_have_exact_zero_mean() -> None:
    observed = (
        MessageAggregator()
        .compute_node_mean(
            _messages()
        )
    )

    assert torch.equal(
        observed[0],
        torch.zeros(HIDDEN_DIM),
    )
    assert torch.equal(
        observed[5],
        torch.zeros(HIDDEN_DIM),
    )


@pytest.mark.parametrize(
    "dtype",
    (
        torch.float32,
        torch.float64,
    ),
)
def test_node_mean_preserves_dtype(
    dtype: torch.dtype,
) -> None:
    observed = (
        MessageAggregator()
        .compute_node_mean(
            _messages(
                edge_messages=(
                    _edge_messages(
                        dtype=dtype
                    )
                ),
                dtype=dtype,
            )
        )
    )

    assert observed.dtype == dtype
    assert torch.equal(
        observed,
        _expected_mean(dtype=dtype),
    )


# =============================================================================
# Grouping semantics and invariances
# =============================================================================


def test_aggregation_is_invariant_to_edge_order() -> None:
    messages = _messages()
    permutation = torch.tensor(
        [7, 2, 0, 6, 1, 5, 3, 4],
        dtype=torch.long,
    )
    permuted = _messages(
        edge_messages=(
            messages.edge_messages[
                permutation
            ]
        ),
        target_index=(
            messages
            .source_inputs
            .target_index[
                permutation
            ]
        ),
    )

    aggregator = MessageAggregator()
    first, first_counts = (
        aggregator.aggregate_tensor(
            messages
        )
    )
    second, second_counts = (
        aggregator.aggregate_tensor(
            permuted
        )
    )

    assert torch.equal(
        first_counts,
        second_counts,
    )
    assert torch.allclose(
        first,
        second,
        atol=1e-6,
        rtol=1e-6,
    )


def test_aggregation_depends_only_on_target_index_and_messages() -> None:
    messages = _messages()
    aggregator = MessageAggregator()

    first = aggregator.compute_node_mean(
        messages
    )

    inputs = messages.source_inputs
    inputs.edge_relation_index = torch.tensor(
        [2, 1, 0, 2, 1, 0, 2, 1],
        dtype=torch.long,
    )
    inputs.edge_relation_family_index = torch.tensor(
        [1, 1, 0, 1, 0, 0, 1, 1],
        dtype=torch.long,
    )
    inputs.node_batch_index = torch.tensor(
        [0, 0, 0, 1, 1, 1],
        dtype=torch.long,
    )
    inputs.hazard_query = torch.randn(
        NODE_COUNT,
        5,
    )
    inputs.attention_group_id = torch.arange(
        EDGE_COUNT,
        dtype=torch.long,
    )
    inputs.control_edge_mask = torch.tensor(
        [False, False, True, False, False, True, False, False]
    )

    second = aggregator.compute_node_mean(
        messages
    )

    assert torch.equal(first, second)


def test_equal_target_groups_ignore_relation_boundaries() -> None:
    values = torch.tensor(
        [
            [0.0, 0.0, 0.0],
            [10.0, 20.0, 30.0],
        ]
    )
    target = torch.tensor(
        [3, 3],
        dtype=torch.long,
    )
    messages = _messages(
        edge_messages=values,
        target_index=target,
        num_edges=2,
    )
    messages.source_inputs.edge_relation_index = torch.tensor(
        [0, 2],
        dtype=torch.long,
    )

    observed = (
        MessageAggregator()
        .compute_node_mean(messages)
    )

    assert torch.equal(
        observed[3],
        torch.tensor(
            [5.0, 10.0, 15.0]
        ),
    )


def test_message_scaling_scales_sum_and_mean() -> None:
    messages = _messages()
    scale = 3.5
    scaled = _messages(
        edge_messages=(
            messages.edge_messages
            * scale
        ),
        target_index=(
            messages
            .source_inputs
            .target_index
        ),
    )
    aggregator = MessageAggregator()

    assert torch.allclose(
        aggregator.compute_node_sum(
            scaled
        ),
        aggregator.compute_node_sum(
            messages
        )
        * scale,
    )
    assert torch.allclose(
        aggregator.compute_node_mean(
            scaled
        ),
        aggregator.compute_node_mean(
            messages
        )
        * scale,
    )


# =============================================================================
# Aggregate tuple and forward output
# =============================================================================


def test_aggregate_tensor_returns_mean_and_counts() -> None:
    aggregate, counts = (
        MessageAggregator()
        .aggregate_tensor(
            _messages()
        )
    )

    assert torch.equal(
        aggregate,
        _expected_mean(),
    )
    assert torch.equal(
        counts,
        _expected_counts(),
    )


def test_forward_constructs_complete_output() -> None:
    aggregator = MessageAggregator()
    messages = _messages()

    output = aggregator(messages)

    assert isinstance(
        output,
        FakeAggregationOutput,
    )
    assert output.source_messages is messages
    assert output.aggregation_mode == (
        AGGREGATION_MEAN
    )
    assert torch.equal(
        output.node_aggregate,
        _expected_mean(),
    )
    assert torch.equal(
        output.incoming_edge_count,
        _expected_counts(),
    )
    assert (
        output.encoder_architecture_fingerprint
        == aggregator.architecture_fingerprint()
    )


def test_forward_is_deterministic() -> None:
    aggregator = MessageAggregator()
    messages = _messages()

    first = aggregator(messages)
    second = aggregator(messages)

    assert torch.equal(
        first.node_aggregate,
        second.node_aggregate,
    )
    assert torch.equal(
        first.incoming_edge_count,
        second.incoming_edge_count,
    )
    assert (
        first.encoder_architecture_fingerprint
        == second.encoder_architecture_fingerprint
    )


# =============================================================================
# Zero-edge behavior
# =============================================================================


def test_zero_edge_graph_returns_zero_counts() -> None:
    messages = _messages(
        edge_messages=torch.empty(
            0,
            HIDDEN_DIM,
        ),
        target_index=torch.empty(
            0,
            dtype=torch.long,
        ),
        num_edges=0,
    )

    counts = (
        MessageAggregator()
        .compute_incoming_edge_count(
            messages
        )
    )

    assert torch.equal(
        counts,
        torch.zeros(
            NODE_COUNT,
            dtype=torch.long,
        ),
    )


def test_zero_edge_graph_returns_zero_sum() -> None:
    messages = _messages(
        edge_messages=torch.empty(
            0,
            HIDDEN_DIM,
        ),
        target_index=torch.empty(
            0,
            dtype=torch.long,
        ),
        num_edges=0,
    )

    observed = (
        MessageAggregator()
        .compute_node_sum(messages)
    )

    assert torch.equal(
        observed,
        torch.zeros(
            NODE_COUNT,
            HIDDEN_DIM,
        ),
    )


def test_zero_edge_graph_returns_zero_mean() -> None:
    messages = _messages(
        edge_messages=torch.empty(
            0,
            HIDDEN_DIM,
        ),
        target_index=torch.empty(
            0,
            dtype=torch.long,
        ),
        num_edges=0,
    )

    observed = (
        MessageAggregator()
        .compute_node_mean(messages)
    )

    assert torch.equal(
        observed,
        torch.zeros(
            NODE_COUNT,
            HIDDEN_DIM,
        ),
    )


def test_zero_edge_forward_output_contract() -> None:
    messages = _messages(
        edge_messages=torch.empty(
            0,
            HIDDEN_DIM,
            dtype=torch.float64,
        ),
        target_index=torch.empty(
            0,
            dtype=torch.long,
        ),
        num_edges=0,
        dtype=torch.float64,
    )

    output = MessageAggregator()(messages)

    assert output.node_aggregate.shape == (
        NODE_COUNT,
        HIDDEN_DIM,
    )
    assert output.node_aggregate.dtype == (
        torch.float64
    )
    assert torch.equal(
        output.node_aggregate,
        torch.zeros(
            NODE_COUNT,
            HIDDEN_DIM,
            dtype=torch.float64,
        ),
    )
    assert torch.equal(
        output.incoming_edge_count,
        torch.zeros(
            NODE_COUNT,
            dtype=torch.long,
        ),
    )


# =============================================================================
# Input validation failures
# =============================================================================


def test_rejects_wrong_message_type() -> None:
    with pytest.raises(
        TypeError,
        match="EdgeMessageOutput",
    ):
        MessageAggregator().aggregate_tensor(
            object()  # type: ignore[arg-type]
        )


def test_rejects_non_tensor_edge_messages() -> None:
    messages = _messages()
    messages.edge_messages = [  # type: ignore[assignment]
        [1.0, 2.0, 3.0]
    ]

    with pytest.raises(
        TypeError,
        match="edge_messages must be a tensor",
    ):
        MessageAggregator().aggregate_tensor(
            messages
        )


@pytest.mark.parametrize(
    "edge_messages",
    (
        torch.zeros(HIDDEN_DIM),
        torch.zeros(
            2,
            HIDDEN_DIM,
            1,
        ),
    ),
)
def test_rejects_invalid_edge_message_rank(
    edge_messages: torch.Tensor,
) -> None:
    messages = _messages(
        edge_messages=edge_messages,
        target_index=torch.zeros(
            int(edge_messages.shape[0]),
            dtype=torch.long,
        ),
        num_edges=int(
            edge_messages.shape[0]
        ),
    )

    with pytest.raises(
        ValueError,
        match=r"shape \[E, H\]",
    ):
        MessageAggregator().aggregate_tensor(
            messages
        )


def test_rejects_edge_message_row_mismatch() -> None:
    values = torch.zeros(
        EDGE_COUNT - 1,
        HIDDEN_DIM,
    )
    messages = _messages(
        edge_messages=values,
        target_index=_target_index(),
        num_edges=EDGE_COUNT,
    )

    with pytest.raises(
        ValueError,
        match="shape does not match",
    ):
        MessageAggregator().aggregate_tensor(
            messages
        )


def test_rejects_edge_message_width_mismatch() -> None:
    values = torch.zeros(
        EDGE_COUNT,
        HIDDEN_DIM + 1,
    )
    messages = _messages(
        edge_messages=values,
        hidden_dim=HIDDEN_DIM,
    )

    with pytest.raises(
        ValueError,
        match="shape does not match",
    ):
        MessageAggregator().aggregate_tensor(
            messages
        )


@pytest.mark.parametrize(
    "dtype",
    (
        torch.long,
        torch.int32,
        torch.bool,
    ),
)
def test_rejects_nonfloating_edge_messages(
    dtype: torch.dtype,
) -> None:
    values = torch.zeros(
        EDGE_COUNT,
        HIDDEN_DIM,
        dtype=dtype,
    )
    messages = _messages(
        edge_messages=values,
        dtype=dtype,
    )

    with pytest.raises(
        ValueError,
        match="floating-point dtype",
    ):
        MessageAggregator().aggregate_tensor(
            messages
        )


@pytest.mark.parametrize(
    "bad_value",
    (
        float("nan"),
        float("inf"),
        float("-inf"),
    ),
)
def test_rejects_nonfinite_edge_messages(
    bad_value: float,
) -> None:
    values = _edge_messages()
    values[0, 0] = bad_value

    with pytest.raises(
        ValueError,
        match="finite values",
    ):
        MessageAggregator().aggregate_tensor(
            _messages(
                edge_messages=values
            )
        )


@pytest.mark.parametrize(
    "num_nodes",
    (
        0,
        -1,
    ),
)
def test_rejects_nonpositive_node_count(
    num_nodes: int,
) -> None:
    messages = _messages(
        edge_messages=torch.empty(
            0,
            HIDDEN_DIM,
        ),
        target_index=torch.empty(
            0,
            dtype=torch.long,
        ),
        num_nodes=num_nodes,
        num_edges=0,
    )

    with pytest.raises(
        ValueError,
        match="at least one node",
    ):
        MessageAggregator().aggregate_tensor(
            messages
        )


def test_rejects_edge_message_dtype_mismatch_with_inputs() -> None:
    messages = _messages(
        edge_messages=_edge_messages(
            dtype=torch.float64
        ),
        dtype=torch.float32,
    )

    with pytest.raises(
        ValueError,
        match="share one dtype",
    ):
        MessageAggregator().aggregate_tensor(
            messages
        )


def test_rejects_non_tensor_target_index() -> None:
    messages = _messages()
    messages.source_inputs.target_index = [  # type: ignore[assignment]
        1,
        1,
    ]

    with pytest.raises(
        TypeError,
        match="target_index must be a tensor",
    ):
        MessageAggregator().aggregate_tensor(
            messages
        )


def test_rejects_invalid_target_index_rank() -> None:
    messages = _messages()
    messages.source_inputs.target_index = (
        _target_index().unsqueeze(0)
    )

    with pytest.raises(
        ValueError,
        match=r"shape \[E\]",
    ):
        MessageAggregator().aggregate_tensor(
            messages
        )


@pytest.mark.parametrize(
    "dtype",
    (
        torch.int32,
        torch.float32,
        torch.bool,
    ),
)
def test_rejects_invalid_target_index_dtype(
    dtype: torch.dtype,
) -> None:
    messages = _messages()
    messages.source_inputs.target_index = (
        _target_index().to(dtype=dtype)
    )

    with pytest.raises(
        ValueError,
        match="torch.long",
    ):
        MessageAggregator().aggregate_tensor(
            messages
        )


def test_rejects_target_index_length_mismatch() -> None:
    messages = _messages()
    messages.source_inputs.target_index = (
        _target_index()[:-1]
    )

    with pytest.raises(
        ValueError,
        match="length must equal the edge count",
    ):
        MessageAggregator().aggregate_tensor(
            messages
        )


@pytest.mark.parametrize(
    "bad_index",
    (
        -1,
        NODE_COUNT,
    ),
)
def test_rejects_target_index_out_of_range(
    bad_index: int,
) -> None:
    target = _target_index()
    target[0] = bad_index

    with pytest.raises(
        ValueError,
        match="out-of-range",
    ):
        MessageAggregator().aggregate_tensor(
            _messages(
                target_index=target
            )
        )


# =============================================================================
# Internal incoming-count guards
# =============================================================================


def test_rejects_wrong_count_shape(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_counts(
        segment_ids: torch.Tensor,
        *,
        num_segments: int,
    ) -> torch.Tensor:
        return torch.zeros(
            num_segments + 1,
            dtype=torch.long,
            device=segment_ids.device,
        )

    monkeypatch.setattr(
        aggregators_module,
        "segment_counts",
        fake_counts,
    )

    with pytest.raises(
        RuntimeError,
        match="returned shape",
    ):
        (
            MessageAggregator()
            .compute_incoming_edge_count(
                _messages()
            )
        )


def test_rejects_wrong_count_dtype(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_counts(
        segment_ids: torch.Tensor,
        *,
        num_segments: int,
    ) -> torch.Tensor:
        return torch.zeros(
            num_segments,
            dtype=torch.float32,
            device=segment_ids.device,
        )

    monkeypatch.setattr(
        aggregators_module,
        "segment_counts",
        fake_counts,
    )

    with pytest.raises(
        RuntimeError,
        match="must use torch.long",
    ):
        (
            MessageAggregator()
            .compute_incoming_edge_count(
                _messages()
            )
        )


def test_rejects_negative_count(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_counts(
        segment_ids: torch.Tensor,
        *,
        num_segments: int,
    ) -> torch.Tensor:
        result = torch.zeros(
            num_segments,
            dtype=torch.long,
            device=segment_ids.device,
        )
        result[0] = -1
        result[1] = EDGE_COUNT + 1
        return result

    monkeypatch.setattr(
        aggregators_module,
        "segment_counts",
        fake_counts,
    )

    with pytest.raises(
        RuntimeError,
        match="must be nonnegative",
    ):
        (
            MessageAggregator()
            .compute_incoming_edge_count(
                _messages()
            )
        )


def test_rejects_count_sum_mismatch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_counts(
        segment_ids: torch.Tensor,
        *,
        num_segments: int,
    ) -> torch.Tensor:
        return torch.zeros(
            num_segments,
            dtype=torch.long,
            device=segment_ids.device,
        )

    monkeypatch.setattr(
        aggregators_module,
        "segment_counts",
        fake_counts,
    )

    with pytest.raises(
        RuntimeError,
        match="do not sum",
    ):
        (
            MessageAggregator()
            .compute_incoming_edge_count(
                _messages()
            )
        )


# =============================================================================
# Internal aggregate guards
# =============================================================================


@pytest.mark.parametrize(
    "operation_name",
    (
        "segment_sum",
        "segment_mean",
    ),
)
def test_rejects_wrong_aggregate_shape(
    monkeypatch: pytest.MonkeyPatch,
    operation_name: str,
) -> None:
    def fake_operation(
        values: torch.Tensor,
        segment_ids: torch.Tensor,
        *,
        num_segments: int,
    ) -> torch.Tensor:
        return torch.zeros(
            num_segments + 1,
            values.shape[1],
            dtype=values.dtype,
            device=values.device,
        )

    monkeypatch.setattr(
        aggregators_module,
        operation_name,
        fake_operation,
    )

    aggregator = MessageAggregator()

    with pytest.raises(
        RuntimeError,
        match="has shape",
    ):
        if operation_name == "segment_sum":
            aggregator.compute_node_sum(
                _messages()
            )
        else:
            aggregator.compute_node_mean(
                _messages()
            )


@pytest.mark.parametrize(
    "operation_name",
    (
        "segment_sum",
        "segment_mean",
    ),
)
def test_rejects_wrong_aggregate_dtype(
    monkeypatch: pytest.MonkeyPatch,
    operation_name: str,
) -> None:
    def fake_operation(
        values: torch.Tensor,
        segment_ids: torch.Tensor,
        *,
        num_segments: int,
    ) -> torch.Tensor:
        return torch.zeros(
            num_segments,
            values.shape[1],
            dtype=torch.float64,
            device=values.device,
        )

    monkeypatch.setattr(
        aggregators_module,
        operation_name,
        fake_operation,
    )

    aggregator = MessageAggregator()

    with pytest.raises(
        RuntimeError,
        match="changed dtype",
    ):
        if operation_name == "segment_sum":
            aggregator.compute_node_sum(
                _messages()
            )
        else:
            aggregator.compute_node_mean(
                _messages()
            )


@pytest.mark.parametrize(
    "operation_name",
    (
        "segment_sum",
        "segment_mean",
    ),
)
def test_rejects_nonfinite_aggregate(
    monkeypatch: pytest.MonkeyPatch,
    operation_name: str,
) -> None:
    def fake_operation(
        values: torch.Tensor,
        segment_ids: torch.Tensor,
        *,
        num_segments: int,
    ) -> torch.Tensor:
        result = torch.zeros(
            num_segments,
            values.shape[1],
            dtype=values.dtype,
            device=values.device,
        )
        result[0, 0] = float("nan")
        return result

    monkeypatch.setattr(
        aggregators_module,
        operation_name,
        fake_operation,
    )

    aggregator = MessageAggregator()

    with pytest.raises(
        FloatingPointError,
        match="NaN or infinity",
    ):
        if operation_name == "segment_sum":
            aggregator.compute_node_sum(
                _messages()
            )
        else:
            aggregator.compute_node_mean(
                _messages()
            )


def test_rejects_nonzero_isolated_mean(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_mean(
        values: torch.Tensor,
        segment_ids: torch.Tensor,
        *,
        num_segments: int,
    ) -> torch.Tensor:
        result = torch.zeros(
            num_segments,
            values.shape[1],
            dtype=values.dtype,
            device=values.device,
        )
        result[0] = 1.0
        return result

    monkeypatch.setattr(
        aggregators_module,
        "segment_mean",
        fake_mean,
    )

    with pytest.raises(
        RuntimeError,
        match="Isolated nodes must receive exact zero",
    ):
        MessageAggregator().compute_node_mean(
            _messages()
        )


def test_validate_aggregate_tensor_rejects_nontensor() -> None:
    aggregator = MessageAggregator()

    with pytest.raises(
        RuntimeError,
        match="must be a tensor",
    ):
        aggregator._validate_aggregate_tensor(
            name="node_sum",
            value=[1.0],  # type: ignore[arg-type]
            messages=_messages(),
        )


def test_corrupted_mode_reaches_defensive_dispatch_guard() -> None:
    aggregator = MessageAggregator()
    aggregator.mode = "corrupted"

    with pytest.raises(
        RuntimeError,
        match="unsupported mode",
    ):
        aggregator.aggregate_tensor(
            _messages()
        )


# =============================================================================
# Autograd
# =============================================================================


def test_node_sum_backward_gives_unit_gradient_per_edge() -> None:
    values = _edge_messages(
        requires_grad=True
    )
    messages = _messages(
        edge_messages=values
    )

    (
        MessageAggregator()
        .compute_node_sum(messages)
        .sum()
        .backward()
    )

    assert values.grad is not None
    assert torch.equal(
        values.grad,
        torch.ones_like(values),
    )


def test_node_mean_backward_distributes_by_target_count() -> None:
    values = _edge_messages(
        requires_grad=True
    )
    messages = _messages(
        edge_messages=values
    )

    (
        MessageAggregator()
        .compute_node_mean(messages)
        .sum()
        .backward()
    )

    assert values.grad is not None
    counts = _expected_counts()
    expected_per_edge = (
        1.0
        / counts[
            messages
            .source_inputs
            .target_index
        ].to(values.dtype)
    )
    expected = (
        expected_per_edge
        .unsqueeze(-1)
        .expand_as(values)
    )

    assert torch.equal(
        values.grad,
        expected,
    )


def test_forward_backward_is_finite() -> None:
    values = _edge_messages(
        requires_grad=True
    )
    output = MessageAggregator()(
        _messages(
            edge_messages=values
        )
    )

    output.node_aggregate.square().mean().backward()

    assert values.grad is not None
    assert bool(
        torch.isfinite(
            values.grad
        ).all().item()
    )


def test_mean_gradcheck() -> None:
    target = _target_index()
    aggregator = MessageAggregator()
    values = torch.randn(
        EDGE_COUNT,
        HIDDEN_DIM,
        dtype=torch.float64,
        requires_grad=True,
    )

    assert torch.autograd.gradcheck(
        lambda tensor: (
            aggregator.compute_node_mean(
                _messages(
                    edge_messages=tensor,
                    target_index=target,
                    dtype=torch.float64,
                )
            )
        ),
        (values,),
        eps=1e-6,
        atol=1e-5,
        rtol=1e-3,
    )


# =============================================================================
# Representation
# =============================================================================


def test_extra_repr_contains_contract_identity() -> None:
    representation = (
        MessageAggregator()
        .extra_repr()
    )

    assert AGGREGATION_MEAN in representation
    assert "grouping=target_node" in (
        representation
    )
    assert "parameter_count=0" in (
        representation
    )
    assert (
        "isolated_node_policy='exact_zero'"
        in representation
    )


# =============================================================================
# Optional CUDA
# =============================================================================


@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA is unavailable.",
)
def test_rejects_edge_message_device_mismatch() -> None:
    values = _edge_messages(
        device="cpu"
    )
    messages = _messages(
        edge_messages=values,
        target_index=_target_index(
            device="cuda"
        ),
        device=torch.device("cuda"),
    )

    with pytest.raises(
        ValueError,
        match="Edge messages and source inputs must share one device",
    ):
        MessageAggregator().aggregate_tensor(
            messages
        )


@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA is unavailable.",
)
def test_rejects_target_index_device_mismatch() -> None:
    values = _edge_messages(
        device="cuda"
    )
    messages = _messages(
        edge_messages=values,
        target_index=_target_index(
            device="cpu"
        ),
        device=torch.device("cuda"),
    )

    with pytest.raises(
        ValueError,
        match="Target indices and source inputs must share one device",
    ):
        MessageAggregator().aggregate_tensor(
            messages
        )


@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA is unavailable.",
)
def test_cuda_results_match_cpu() -> None:
    cpu_messages = _messages()
    cuda_messages = _messages(
        edge_messages=_edge_messages(
            device="cuda"
        ),
        target_index=_target_index(
            device="cuda"
        ),
        device=torch.device("cuda"),
    )
    aggregator = MessageAggregator()

    cpu_output = aggregator(cpu_messages)
    cuda_output = aggregator(cuda_messages)

    assert torch.equal(
        cpu_output.incoming_edge_count,
        cuda_output.incoming_edge_count.cpu(),
    )
    assert torch.allclose(
        cpu_output.node_aggregate,
        cuda_output.node_aggregate.cpu(),
        atol=1e-6,
        rtol=1e-6,
    )


@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA is unavailable.",
)
def test_cuda_backward_is_finite() -> None:
    values = _edge_messages(
        device="cuda",
        requires_grad=True,
    )
    messages = _messages(
        edge_messages=values,
        target_index=_target_index(
            device="cuda"
        ),
        device=torch.device("cuda"),
    )

    (
        MessageAggregator()(messages)
        .node_aggregate
        .square()
        .mean()
        .backward()
    )

    assert values.grad is not None
    assert bool(
        torch.isfinite(
            values.grad
        ).all().item()
    )


@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA is unavailable.",
)
def test_rejects_internal_count_device_change(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_counts(
        segment_ids: torch.Tensor,
        *,
        num_segments: int,
    ) -> torch.Tensor:
        return torch.zeros(
            num_segments,
            dtype=torch.long,
            device="cpu",
        )

    monkeypatch.setattr(
        aggregators_module,
        "segment_counts",
        fake_counts,
    )

    messages = _messages(
        edge_messages=_edge_messages(
            device="cuda"
        ),
        target_index=_target_index(
            device="cuda"
        ),
        device=torch.device("cuda"),
    )

    with pytest.raises(
        RuntimeError,
        match="changed device",
    ):
        (
            MessageAggregator()
            .compute_incoming_edge_count(
                messages
            )
        )


@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA is unavailable.",
)
@pytest.mark.parametrize(
    "operation_name",
    (
        "segment_sum",
        "segment_mean",
    ),
)
def test_rejects_internal_aggregate_device_change(
    monkeypatch: pytest.MonkeyPatch,
    operation_name: str,
) -> None:
    def fake_operation(
        values: torch.Tensor,
        segment_ids: torch.Tensor,
        *,
        num_segments: int,
    ) -> torch.Tensor:
        return torch.zeros(
            num_segments,
            values.shape[1],
            dtype=values.dtype,
            device="cpu",
        )

    monkeypatch.setattr(
        aggregators_module,
        operation_name,
        fake_operation,
    )

    messages = _messages(
        edge_messages=_edge_messages(
            device="cuda"
        ),
        target_index=_target_index(
            device="cuda"
        ),
        device=torch.device("cuda"),
    )
    aggregator = MessageAggregator()

    with pytest.raises(
        RuntimeError,
        match="changed device",
    ):
        if operation_name == "segment_sum":
            aggregator.compute_node_sum(
                messages
            )
        else:
            aggregator.compute_node_mean(
                messages
            )
