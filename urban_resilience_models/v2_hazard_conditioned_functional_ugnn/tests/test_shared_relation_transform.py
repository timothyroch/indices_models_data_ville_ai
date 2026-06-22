"""
Contract tests for the shared relation transform.

Target repository path:
    urban_resilience_models/
        v2_hazard_conditioned_functional_ugnn/
            tests/
                test_shared_relation_transform.py

Implementation under test:
    urban_resilience_models/
        v2_hazard_conditioned_functional_ugnn/
            functional_message_passing/
                relation_transforms/
                    shared_transform.py

This suite freezes the bounded shared-transform contract independently from
relation dispatch, per-relation transforms, structural normalization, gates,
attention, message construction, aggregation, and layer updates.

Covered behavior:

- constructor and public identity;
- exact source-state gathering;
- one shared linear map across all relation identities;
- empty-edge and empty-node/empty-edge handling;
- strict shape, dtype, device, range, and finiteness validation;
- explicit absence of activation, normalization, and dropout;
- deterministic evaluation behavior;
- finite forward and backward passes;
- architecture and parameter fingerprint separation;
- state-dict round trips and stable parameter names;
- finite-parameter corruption detection;
- optional CUDA execution.
"""

from __future__ import annotations

from typing import Any

import pytest
import torch
from torch import nn

from urban_resilience_models.v2_hazard_conditioned_functional_ugnn.constants import (
    RELATION_TRANSFORM_SHARED,
)
from urban_resilience_models.v2_hazard_conditioned_functional_ugnn.functional_message_passing.relation_transforms.shared_transform import (
    SHARED_RELATION_TRANSFORM_SCHEMA_VERSION,
    SharedRelationTransform,
)


HIDDEN_DIM = 4
NODE_COUNT = 5
EDGE_COUNT = 6


# =============================================================================
# Helpers
# =============================================================================


def _module(
    *,
    hidden_dim: int = HIDDEN_DIM,
    bias: bool = True,
) -> SharedRelationTransform:
    return SharedRelationTransform(
        hidden_dim=hidden_dim,
        bias=bias,
    )


def _node_state(
    *,
    node_count: int = NODE_COUNT,
    hidden_dim: int = HIDDEN_DIM,
    dtype: torch.dtype = torch.float32,
    device: torch.device | str = "cpu",
    requires_grad: bool = False,
    offset: float = 0.0,
) -> torch.Tensor:
    values = (
        torch.arange(
            node_count * hidden_dim,
            dtype=dtype,
            device=device,
        )
        .reshape(node_count, hidden_dim)
        / 10.0
        + offset
    )
    values.requires_grad_(requires_grad)
    return values


def _source_index(
    *,
    device: torch.device | str = "cpu",
) -> torch.Tensor:
    return torch.tensor(
        [0, 2, 2, 4, 1, 3],
        dtype=torch.long,
        device=device,
    )


# =============================================================================
# Published identity and constructor contract
# =============================================================================


def test_schema_version_is_nonempty() -> None:
    assert isinstance(
        SHARED_RELATION_TRANSFORM_SCHEMA_VERSION,
        str,
    )
    assert (
        SHARED_RELATION_TRANSFORM_SCHEMA_VERSION
        .strip()
    )


def test_constructor_preserves_contract() -> None:
    module = _module(
        hidden_dim=HIDDEN_DIM,
        bias=True,
    )

    assert module.hidden_dim == HIDDEN_DIM
    assert module.bias is True
    assert module.input_dim == HIDDEN_DIM
    assert module.output_dim == HIDDEN_DIM
    assert module.transform_mode == (
        RELATION_TRANSFORM_SHARED
    )


def test_constructor_without_bias() -> None:
    module = _module(
        bias=False,
    )

    assert module.bias is False
    assert module.linear.bias is None


def test_constructor_builds_exact_linear_layer() -> None:
    module = _module()

    assert isinstance(
        module.linear,
        nn.Linear,
    )
    assert module.linear.in_features == (
        HIDDEN_DIM
    )
    assert module.linear.out_features == (
        HIDDEN_DIM
    )


def test_constructor_has_no_hidden_activation_normalization_or_dropout() -> None:
    module = _module()

    child_names = tuple(
        dict(
            module.named_children()
        )
    )

    assert child_names == ("linear",)
    assert not any(
        isinstance(
            child,
            (
                nn.Dropout,
                nn.LayerNorm,
                nn.BatchNorm1d,
                nn.GELU,
                nn.ReLU,
            ),
        )
        for child in module.modules()
        if child is not module
        and child is not module.linear
    )


@pytest.mark.parametrize(
    "hidden_dim",
    (
        0,
        -1,
        True,
        1.5,
    ),
)
def test_constructor_rejects_invalid_hidden_dim(
    hidden_dim: Any,
) -> None:
    with pytest.raises(
        ValueError,
        match="positive integer",
    ):
        _module(
            hidden_dim=hidden_dim,
        )


@pytest.mark.parametrize(
    "bias",
    (
        0,
        1,
        "true",
        None,
    ),
)
def test_constructor_rejects_nonboolean_bias(
    bias: Any,
) -> None:
    with pytest.raises(
        TypeError,
        match="Boolean",
    ):
        _module(
            bias=bias,
        )


def test_device_and_dtype_properties_follow_parameters() -> None:
    module = _module()

    assert module.device == torch.device(
        "cpu"
    )
    assert module.dtype == torch.float32

    module = module.double()

    assert module.dtype == torch.float64


def test_parameter_counts_are_correct_with_bias() -> None:
    module = _module(
        bias=True,
    )

    assert module.parameter_count == (
        HIDDEN_DIM * HIDDEN_DIM
        + HIDDEN_DIM
    )
    assert (
        module.trainable_parameter_count
        == module.parameter_count
    )


def test_parameter_counts_are_correct_without_bias() -> None:
    module = _module(
        bias=False,
    )

    assert module.parameter_count == (
        HIDDEN_DIM * HIDDEN_DIM
    )


# =============================================================================
# Source-state gathering
# =============================================================================


def test_gather_source_state_matches_direct_indexing() -> None:
    module = _module()
    node_state = _node_state()
    source_index = _source_index()

    gathered = module.gather_source_state(
        node_state,
        source_index,
    )

    assert torch.equal(
        gathered,
        node_state[source_index],
    )


def test_gather_source_state_preserves_duplicate_edges() -> None:
    module = _module()
    node_state = _node_state()
    source_index = torch.tensor(
        [2, 2, 2],
        dtype=torch.long,
    )

    gathered = module.gather_source_state(
        node_state,
        source_index,
    )

    assert torch.equal(
        gathered,
        node_state[2]
        .unsqueeze(0)
        .repeat(3, 1),
    )


def test_gather_source_state_preserves_edge_order() -> None:
    module = _module()
    node_state = _node_state()
    source_index = torch.tensor(
        [4, 0, 3, 1],
        dtype=torch.long,
    )

    gathered = module.gather_source_state(
        node_state,
        source_index,
    )

    assert torch.equal(
        gathered[0],
        node_state[4],
    )
    assert torch.equal(
        gathered[1],
        node_state[0],
    )
    assert torch.equal(
        gathered[2],
        node_state[3],
    )
    assert torch.equal(
        gathered[3],
        node_state[1],
    )


def test_gather_source_state_supports_zero_edges() -> None:
    module = _module()
    node_state = _node_state()
    source_index = torch.empty(
        0,
        dtype=torch.long,
    )

    gathered = module.gather_source_state(
        node_state,
        source_index,
    )

    assert gathered.shape == (
        0,
        HIDDEN_DIM,
    )
    assert gathered.dtype == (
        node_state.dtype
    )
    assert gathered.device == (
        node_state.device
    )


def test_gather_source_state_supports_zero_nodes_when_zero_edges() -> None:
    module = _module()
    node_state = _node_state(
        node_count=0,
    )
    source_index = torch.empty(
        0,
        dtype=torch.long,
    )

    gathered = module.gather_source_state(
        node_state,
        source_index,
    )

    assert gathered.shape == (
        0,
        HIDDEN_DIM,
    )


# =============================================================================
# Forward mathematics
# =============================================================================


def test_forward_matches_manual_shared_linear_map() -> None:
    module = _module()
    module.eval()

    node_state = _node_state()
    source_index = _source_index()

    observed = module(
        node_state,
        source_index,
    )
    expected = torch.nn.functional.linear(
        node_state[source_index],
        module.linear.weight,
        module.linear.bias,
    )

    assert torch.equal(
        observed,
        expected,
    )


def test_transform_source_state_matches_linear_module() -> None:
    module = _module()
    source_state = _node_state(
        node_count=EDGE_COUNT,
    )

    observed = module.transform_source_state(
        source_state
    )
    expected = module.linear(
        source_state
    )

    assert torch.equal(
        observed,
        expected,
    )


def test_forward_equals_gather_then_transform() -> None:
    module = _module()
    node_state = _node_state()
    source_index = _source_index()

    observed = module(
        node_state,
        source_index,
    )
    expected = module.transform_source_state(
        module.gather_source_state(
            node_state,
            source_index,
        )
    )

    assert torch.equal(
        observed,
        expected,
    )


def test_same_source_state_receives_same_transform_across_edges() -> None:
    module = _module()
    node_state = _node_state()
    source_index = torch.tensor(
        [2, 2, 2, 2],
        dtype=torch.long,
    )

    output = module(
        node_state,
        source_index,
    )

    assert torch.equal(
        output,
        output[0]
        .unsqueeze(0)
        .repeat(4, 1),
    )


def test_shared_transform_is_relation_identity_agnostic() -> None:
    module = _module()
    node_state = _node_state()

    first_order = torch.tensor(
        [0, 2, 4],
        dtype=torch.long,
    )
    second_order = torch.tensor(
        [4, 0, 2],
        dtype=torch.long,
    )

    first = module(
        node_state,
        first_order,
    )
    second = module(
        node_state,
        second_order,
    )

    assert torch.equal(
        first[
            torch.tensor(
                [2, 0, 1]
            )
        ],
        second,
    )


def test_forward_output_shape_and_finiteness() -> None:
    module = _module()
    output = module(
        _node_state(),
        _source_index(),
    )

    assert output.shape == (
        EDGE_COUNT,
        HIDDEN_DIM,
    )
    assert bool(
        torch.isfinite(output)
        .all()
        .item()
    )


def test_forward_supports_zero_edges() -> None:
    module = _module()

    output = module(
        _node_state(),
        torch.empty(
            0,
            dtype=torch.long,
        ),
    )

    assert output.shape == (
        0,
        HIDDEN_DIM,
    )


def test_forward_supports_zero_nodes_and_zero_edges() -> None:
    module = _module()

    output = module(
        _node_state(
            node_count=0,
        ),
        torch.empty(
            0,
            dtype=torch.long,
        ),
    )

    assert output.shape == (
        0,
        HIDDEN_DIM,
    )


def test_forward_is_deterministic_in_eval_mode() -> None:
    module = _module()
    module.eval()
    node_state = _node_state()
    source_index = _source_index()

    first = module(
        node_state,
        source_index,
    )
    second = module(
        node_state,
        source_index,
    )

    assert torch.equal(
        first,
        second,
    )


def test_forward_is_deterministic_in_train_mode() -> None:
    module = _module()
    module.train()
    node_state = _node_state()
    source_index = _source_index()

    first = module(
        node_state,
        source_index,
    )
    second = module(
        node_state,
        source_index,
    )

    assert torch.equal(
        first,
        second,
    )


def test_without_bias_zero_input_maps_to_zero() -> None:
    module = _module(
        bias=False,
    )
    node_state = torch.zeros(
        NODE_COUNT,
        HIDDEN_DIM,
    )

    output = module(
        node_state,
        _source_index(),
    )

    assert torch.equal(
        output,
        torch.zeros_like(output),
    )


def test_with_bias_zero_input_maps_to_repeated_bias() -> None:
    module = _module(
        bias=True,
    )
    node_state = torch.zeros(
        NODE_COUNT,
        HIDDEN_DIM,
    )

    output = module(
        node_state,
        _source_index(),
    )

    expected = (
        module.linear.bias
        .unsqueeze(0)
        .expand_as(output)
    )

    assert torch.equal(
        output,
        expected,
    )


def test_identity_weight_recovers_gathered_source_state() -> None:
    module = _module(
        bias=False,
    )
    with torch.no_grad():
        module.linear.weight.copy_(
            torch.eye(HIDDEN_DIM)
        )

    node_state = _node_state()
    source_index = _source_index()
    output = module(
        node_state,
        source_index,
    )

    assert torch.equal(
        output,
        node_state[source_index],
    )


# =============================================================================
# Forward validation failures
# =============================================================================


def test_forward_rejects_non_tensor_node_state() -> None:
    module = _module()

    with pytest.raises(
        TypeError,
        match="node_state must be a tensor",
    ):
        module(  # type: ignore[arg-type]
            [[0.0] * HIDDEN_DIM],
            torch.tensor(
                [0],
                dtype=torch.long,
            ),
        )


@pytest.mark.parametrize(
    "node_state",
    (
        torch.zeros(HIDDEN_DIM),
        torch.zeros(
            2,
            HIDDEN_DIM,
            1,
        ),
    ),
)
def test_forward_rejects_invalid_node_state_rank(
    node_state: torch.Tensor,
) -> None:
    module = _module()

    with pytest.raises(
        ValueError,
        match=r"shape \[N, H\]",
    ):
        module(
            node_state,
            torch.tensor(
                [0],
                dtype=torch.long,
            ),
        )


def test_forward_rejects_hidden_width_mismatch() -> None:
    module = _module()

    with pytest.raises(
        ValueError,
        match="feature width",
    ):
        module(
            torch.zeros(
                NODE_COUNT,
                HIDDEN_DIM + 1,
            ),
            _source_index(),
        )


def test_forward_rejects_nonfloating_node_state() -> None:
    module = _module()

    with pytest.raises(
        ValueError,
        match="floating-point",
    ):
        module(
            torch.zeros(
                NODE_COUNT,
                HIDDEN_DIM,
                dtype=torch.long,
            ),
            _source_index(),
        )


@pytest.mark.parametrize(
    "bad_value",
    (
        float("nan"),
        float("inf"),
        float("-inf"),
    ),
)
def test_forward_rejects_nonfinite_node_state(
    bad_value: float,
) -> None:
    module = _module()
    node_state = _node_state()
    node_state[0, 0] = bad_value

    with pytest.raises(
        ValueError,
        match="finite",
    ):
        module(
            node_state,
            _source_index(),
        )


def test_forward_rejects_non_tensor_source_index() -> None:
    module = _module()

    with pytest.raises(
        TypeError,
        match="source_index must be a tensor",
    ):
        module(  # type: ignore[arg-type]
            _node_state(),
            [0, 1],
        )


def test_forward_rejects_invalid_source_index_rank() -> None:
    module = _module()

    with pytest.raises(
        ValueError,
        match=r"shape \[E\]",
    ):
        module(
            _node_state(),
            torch.tensor(
                [[0, 1]],
                dtype=torch.long,
            ),
        )


@pytest.mark.parametrize(
    "dtype",
    (
        torch.int32,
        torch.float32,
        torch.bool,
    ),
)
def test_forward_rejects_invalid_source_index_dtype(
    dtype: torch.dtype,
) -> None:
    module = _module()

    with pytest.raises(
        ValueError,
        match="torch.long",
    ):
        module(
            _node_state(),
            torch.tensor(
                [0, 1],
                dtype=dtype,
            ),
        )


def test_forward_rejects_negative_source_index() -> None:
    module = _module()

    with pytest.raises(
        ValueError,
        match="out-of-range",
    ):
        module(
            _node_state(),
            torch.tensor(
                [0, -1],
                dtype=torch.long,
            ),
        )


def test_forward_rejects_source_index_equal_to_node_count() -> None:
    module = _module()

    with pytest.raises(
        ValueError,
        match="out-of-range",
    ):
        module(
            _node_state(),
            torch.tensor(
                [NODE_COUNT],
                dtype=torch.long,
            ),
        )


def test_forward_rejects_nonempty_edges_for_zero_nodes() -> None:
    module = _module()

    with pytest.raises(
        ValueError,
        match="cannot be nonempty",
    ):
        module(
            _node_state(
                node_count=0,
            ),
            torch.tensor(
                [0],
                dtype=torch.long,
            ),
        )


def test_transform_source_state_rejects_wrong_width() -> None:
    module = _module()

    with pytest.raises(
        ValueError,
        match="feature width",
    ):
        module.transform_source_state(
            torch.zeros(
                EDGE_COUNT,
                HIDDEN_DIM + 1,
            )
        )


def test_forward_rejects_input_parameter_dtype_mismatch() -> None:
    module = _module()
    node_state = _node_state(
        dtype=torch.float64,
    )

    with pytest.raises(
        ValueError,
        match="dtype must match",
    ):
        module(
            node_state,
            _source_index(),
        )


def test_double_module_accepts_float64() -> None:
    module = _module().double()

    output = module(
        _node_state(
            dtype=torch.float64,
        ),
        _source_index(),
    )

    assert output.dtype == torch.float64


# =============================================================================
# Output and parameter corruption
# =============================================================================


@pytest.mark.parametrize(
    "bad_value",
    (
        float("nan"),
        float("inf"),
        float("-inf"),
    ),
)
def test_forward_rejects_nonfinite_output_from_corrupted_weight(
    bad_value: float,
) -> None:
    module = _module()

    with torch.no_grad():
        module.linear.weight[
            0,
            0,
        ] = bad_value

    with pytest.raises(
        FloatingPointError,
        match="produced NaN or infinity",
    ):
        module(
            _node_state(),
            _source_index(),
        )


def test_assert_finite_parameters_passes_for_valid_module() -> None:
    module = _module()
    module.assert_finite_parameters()


@pytest.mark.parametrize(
    "bad_value",
    (
        float("nan"),
        float("inf"),
        float("-inf"),
    ),
)
def test_assert_finite_parameters_detects_corruption(
    bad_value: float,
) -> None:
    module = _module()

    with torch.no_grad():
        module.linear.weight[
            0,
            0,
        ] = bad_value

    with pytest.raises(
        ValueError,
        match="NaN or infinity",
    ):
        module.assert_finite_parameters()


# =============================================================================
# Autograd
# =============================================================================


def test_backward_reaches_node_state() -> None:
    module = _module()
    node_state = _node_state(
        requires_grad=True,
    )

    output = module(
        node_state,
        _source_index(),
    )
    output.square().mean().backward()

    assert node_state.grad is not None
    assert node_state.grad.shape == (
        node_state.shape
    )
    assert bool(
        torch.isfinite(
            node_state.grad
        ).all().item()
    )


def test_backward_reaches_all_parameters() -> None:
    module = _module()
    node_state = _node_state(
        requires_grad=True,
    )

    module(
        node_state,
        _source_index(),
    ).sum().backward()

    for parameter in module.parameters():
        assert parameter.grad is not None
        assert bool(
            torch.isfinite(
                parameter.grad
            ).all().item()
        )


def test_duplicate_source_edges_accumulate_input_gradient() -> None:
    module = _module(
        bias=False,
    )
    with torch.no_grad():
        module.linear.weight.copy_(
            torch.eye(HIDDEN_DIM)
        )

    node_state = _node_state(
        requires_grad=True,
    )
    source_index = torch.tensor(
        [2, 2, 2],
        dtype=torch.long,
    )

    module(
        node_state,
        source_index,
    ).sum().backward()

    assert torch.equal(
        node_state.grad[2],
        torch.full(
            (HIDDEN_DIM,),
            3.0,
        ),
    )
    other = torch.ones(
        NODE_COUNT,
        dtype=torch.bool,
    )
    other[2] = False
    assert torch.equal(
        node_state.grad[other],
        torch.zeros_like(
            node_state.grad[other]
        ),
    )


def test_zero_edge_backward_is_valid() -> None:
    module = _module()
    node_state = _node_state(
        requires_grad=True,
    )

    output = module(
        node_state,
        torch.empty(
            0,
            dtype=torch.long,
        ),
    )
    loss = (
        output.sum()
        + node_state.sum() * 0.0
    )
    loss.backward()

    assert node_state.grad is not None
    assert torch.equal(
        node_state.grad,
        torch.zeros_like(
            node_state.grad
        ),
    )


def test_forward_gradcheck() -> None:
    module = _module(
        bias=True,
    ).double()

    node_state = torch.randn(
        NODE_COUNT,
        HIDDEN_DIM,
        dtype=torch.float64,
        requires_grad=True,
    )
    source_index = _source_index()

    assert torch.autograd.gradcheck(
        lambda state: module(
            state,
            source_index,
        ),
        (node_state,),
        eps=1e-6,
        atol=1e-5,
        rtol=1e-3,
    )


# =============================================================================
# Architecture and parameter identity
# =============================================================================


def test_architecture_dict_is_exact() -> None:
    module = _module(
        bias=True,
    )

    assert module.architecture_dict() == {
        "schema_version": (
            SHARED_RELATION_TRANSFORM_SCHEMA_VERSION
        ),
        "transform_mode": (
            RELATION_TRANSFORM_SHARED
        ),
        "hidden_dim": HIDDEN_DIM,
        "bias": True,
        "parameter_sharing": (
            "one_linear_map_for_all_relations"
        ),
        "operation_order": [
            "gather_source_node_state",
            "shared_linear_transform",
        ],
        "activation": None,
        "normalization": None,
        "dropout": 0.0,
    }


def test_architecture_fingerprint_is_stable() -> None:
    first = _module()
    second = _module()

    assert first.architecture_dict() == (
        second.architecture_dict()
    )
    assert first.architecture_fingerprint() == (
        second.architecture_fingerprint()
    )


def test_architecture_fingerprint_changes_with_hidden_dim() -> None:
    first = _module(
        hidden_dim=4,
    )
    second = _module(
        hidden_dim=5,
    )

    assert first.architecture_fingerprint() != (
        second.architecture_fingerprint()
    )


def test_architecture_fingerprint_changes_with_bias_policy() -> None:
    first = _module(
        bias=True,
    )
    second = _module(
        bias=False,
    )

    assert first.architecture_fingerprint() != (
        second.architecture_fingerprint()
    )


def test_parameter_fingerprint_is_reproducible_under_seed() -> None:
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(123)
        first = _module()

    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(123)
        second = _module()

    assert first.parameter_fingerprint() == (
        second.parameter_fingerprint()
    )


def test_parameter_fingerprint_changes_after_weight_mutation() -> None:
    module = _module()
    architecture = (
        module.architecture_fingerprint()
    )
    before = module.parameter_fingerprint()

    with torch.no_grad():
        module.linear.weight[
            0,
            0,
        ] += 1.0

    assert module.parameter_fingerprint() != (
        before
    )
    assert module.architecture_fingerprint() == (
        architecture
    )


def test_parameter_fingerprint_changes_after_bias_mutation() -> None:
    module = _module(
        bias=True,
    )
    before = module.parameter_fingerprint()

    with torch.no_grad():
        module.linear.bias[0] += 1.0

    assert module.parameter_fingerprint() != (
        before
    )


def test_state_dict_keys_are_stable_with_bias() -> None:
    module = _module(
        bias=True,
    )

    assert tuple(
        module.state_dict()
    ) == (
        "linear.weight",
        "linear.bias",
    )


def test_state_dict_keys_are_stable_without_bias() -> None:
    module = _module(
        bias=False,
    )

    assert tuple(
        module.state_dict()
    ) == (
        "linear.weight",
    )


def test_state_dict_round_trip_preserves_outputs() -> None:
    source = _module()
    target = _module()

    target.load_state_dict(
        source.state_dict(),
        strict=True,
    )

    source.eval()
    target.eval()
    node_state = _node_state()
    source_index = _source_index()

    assert torch.equal(
        source(
            node_state,
            source_index,
        ),
        target(
            node_state,
            source_index,
        ),
    )
    assert source.parameter_fingerprint() == (
        target.parameter_fingerprint()
    )


def test_extra_repr_contains_contract_identity() -> None:
    module = _module(
        bias=False,
    )
    representation = module.extra_repr()

    assert f"hidden_dim={HIDDEN_DIM}" in (
        representation
    )
    assert "bias=False" in representation
    assert RELATION_TRANSFORM_SHARED in (
        representation
    )


# =============================================================================
# Optional CUDA
# =============================================================================


@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA is unavailable.",
)
def test_forward_rejects_node_state_source_index_device_mismatch() -> None:
    module = _module().cuda()

    with pytest.raises(
        ValueError,
        match="must share one device",
    ):
        module(
            _node_state(
                device="cuda",
            ),
            _source_index(
                device="cpu",
            ),
        )


@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA is unavailable.",
)
def test_forward_rejects_input_parameter_device_mismatch() -> None:
    module = _module().cuda()

    with pytest.raises(
        ValueError,
        match="parameters must share one device",
    ):
        module(
            _node_state(
                device="cpu",
            ),
            _source_index(
                device="cpu",
            ),
        )


@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA is unavailable.",
)
def test_cuda_forward_matches_cpu_after_state_copy() -> None:
    cpu_module = _module()
    cuda_module = _module().cuda()
    cuda_module.load_state_dict(
        cpu_module.state_dict(),
        strict=True,
    )

    node_state = _node_state()
    source_index = _source_index()

    cpu_output = cpu_module(
        node_state,
        source_index,
    )
    cuda_output = cuda_module(
        node_state.cuda(),
        source_index.cuda(),
    ).cpu()

    assert torch.allclose(
        cpu_output,
        cuda_output,
        atol=1e-6,
        rtol=1e-6,
    )


@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA is unavailable.",
)
def test_cuda_backward_is_finite() -> None:
    module = _module().cuda()
    node_state = _node_state(
        device="cuda",
        requires_grad=True,
    )
    source_index = _source_index(
        device="cuda",
    )

    module(
        node_state,
        source_index,
    ).square().mean().backward()

    assert node_state.grad is not None
    assert bool(
        torch.isfinite(
            node_state.grad
        ).all().item()
    )

    for parameter in module.parameters():
        assert parameter.grad is not None
        assert bool(
            torch.isfinite(
                parameter.grad
            ).all().item()
        )
