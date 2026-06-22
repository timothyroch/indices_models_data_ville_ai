"""
Contract tests for independently parameterized relation transforms.

Target repository path:
    urban_resilience_models/
        v2_hazard_conditioned_functional_ugnn/
            tests/
                test_per_relation_transform.py

Implementation under test:
    urban_resilience_models/
        v2_hazard_conditioned_functional_ugnn/
            functional_message_passing/
                relation_transforms/
                    per_relation_transform.py

This suite freezes the bounded per-relation transform contract independently
from relation-mode dispatch, structural normalization, gates, attention,
message construction, aggregation, and layer updates.

Covered behavior:

- deterministic compiled relation ordering and stable ontology metadata;
- independent ``Linear(H, H)`` modules for every active relation;
- control/placebo relation metadata without special mathematical treatment;
- exact source-state gathering and dense relation-index selection;
- zero-edge batches and zero-edge relation groups;
- explicit distinction between stable IDs and dense tensor indices;
- relation-specific architecture and parameter fingerprints;
- global architecture and parameter fingerprints;
- stable semantic state-dict keys and round trips;
- finite forward and backward behavior;
- gradient isolation and zero gradients for absent relation groups;
- strict constructor, shape, dtype, device, range, and finiteness validation;
- optional construction from a compiled registry;
- optional CUDA parity and gradients.
"""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Any

import pytest
import torch
from torch import nn

from urban_resilience_models.v2_hazard_conditioned_functional_ugnn.constants import (
    RELATION_TRANSFORM_PER_RELATION,
)
from urban_resilience_models.v2_hazard_conditioned_functional_ugnn.functional_message_passing.relation_transforms import (
    per_relation_transform as per_relation_module,
)
from urban_resilience_models.v2_hazard_conditioned_functional_ugnn.functional_message_passing.relation_transforms.per_relation_transform import (
    PER_RELATION_TRANSFORM_SCHEMA_VERSION,
    PerRelationTransform,
)


HIDDEN_DIM = 4
NODE_COUNT = 6
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
CONTROL_RELATION_MASK = (
    False,
    False,
    True,
)
RELATION_COUNT = len(
    RELATION_NAMES
)


# =============================================================================
# Controlled compiled-registry contract
# =============================================================================


@dataclass
class FakeSpecification:
    is_control: bool


@dataclass
class FakeCompiledEntry:
    name: str
    relation_id: int
    specification: FakeSpecification


class FakeCompiledRelationRegistry:
    def __init__(
        self,
        *,
        names: tuple[str, ...] = (
            RELATION_NAMES
        ),
        stable_ids: tuple[int, ...] = (
            STABLE_RELATION_IDS
        ),
        controls: tuple[bool, ...] = (
            CONTROL_RELATION_MASK
        ),
    ) -> None:
        self.relation_names = names
        self.stable_relation_ids = (
            stable_ids
        )
        self.entries = tuple(
            FakeCompiledEntry(
                name=name,
                relation_id=relation_id,
                specification=FakeSpecification(
                    is_control=is_control
                ),
            )
            for name, relation_id, is_control in zip(
                names,
                stable_ids,
                controls,
                strict=True,
            )
        )
        self.validated = False

    def validate(self) -> None:
        self.validated = True


@pytest.fixture(autouse=True)
def _patch_compiled_registry_type(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        per_relation_module,
        "CompiledRelationRegistry",
        FakeCompiledRelationRegistry,
    )


# =============================================================================
# Helpers
# =============================================================================


def _module(
    *,
    hidden_dim: int = HIDDEN_DIM,
    relation_names: tuple[str, ...] = (
        RELATION_NAMES
    ),
    stable_relation_ids: tuple[int, ...] = (
        STABLE_RELATION_IDS
    ),
    control_relation_mask: (
        tuple[bool, ...] | None
    ) = CONTROL_RELATION_MASK,
    bias: bool = True,
) -> PerRelationTransform:
    return PerRelationTransform(
        hidden_dim=hidden_dim,
        relation_names=relation_names,
        stable_relation_ids=(
            stable_relation_ids
        ),
        control_relation_mask=(
            control_relation_mask
        ),
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
    state = (
        torch.arange(
            node_count * hidden_dim,
            dtype=dtype,
            device=device,
        )
        .reshape(
            node_count,
            hidden_dim,
        )
        / 10.0
        + offset
    )
    state.requires_grad_(requires_grad)
    return state


def _source_index(
    *,
    device: torch.device | str = "cpu",
) -> torch.Tensor:
    return torch.tensor(
        [0, 2, 2, 4, 1, 5, 3],
        dtype=torch.long,
        device=device,
    )


def _edge_relation_index(
    *,
    device: torch.device | str = "cpu",
) -> torch.Tensor:
    return torch.tensor(
        [0, 1, 2, 1, 0, 2, 1],
        dtype=torch.long,
        device=device,
    )


def _configure_distinct_maps(
    module: PerRelationTransform,
) -> None:
    with torch.no_grad():
        for relation_index in range(
            module.relation_count
        ):
            linear = (
                module
                .module_for_relation_index(
                    relation_index
                )
            )
            linear.weight.copy_(
                torch.eye(
                    module.hidden_dim,
                    dtype=linear.weight.dtype,
                    device=linear.weight.device,
                )
                * float(
                    relation_index + 1
                )
            )
            if linear.bias is not None:
                linear.bias.fill_(
                    float(relation_index)
                )


def _manual_transform(
    module: PerRelationTransform,
    node_state: torch.Tensor,
    source_index: torch.Tensor,
    edge_relation_index: torch.Tensor,
) -> torch.Tensor:
    source = node_state[
        source_index
    ]
    rows: list[torch.Tensor] = []

    for edge_index in range(
        int(source.shape[0])
    ):
        relation_index = int(
            edge_relation_index[
                edge_index
            ].item()
        )
        linear = (
            module
            .module_for_relation_index(
                relation_index
            )
        )
        rows.append(
            torch.nn.functional.linear(
                source[edge_index],
                linear.weight,
                linear.bias,
            )
        )

    if not rows:
        return node_state.new_empty(
            0,
            module.hidden_dim,
        )

    return torch.stack(rows)


# =============================================================================
# Published identity and constructor
# =============================================================================


def test_schema_version_is_nonempty() -> None:
    assert isinstance(
        PER_RELATION_TRANSFORM_SCHEMA_VERSION,
        str,
    )
    assert (
        PER_RELATION_TRANSFORM_SCHEMA_VERSION
        .strip()
    )


def test_constructor_preserves_relation_metadata() -> None:
    module = _module()

    assert module.hidden_dim == (
        HIDDEN_DIM
    )
    assert module.bias is True
    assert module.relation_names == (
        RELATION_NAMES
    )
    assert module.stable_relation_ids == (
        STABLE_RELATION_IDS
    )
    assert module.control_relation_mask == (
        CONTROL_RELATION_MASK
    )
    assert module.relation_count == (
        RELATION_COUNT
    )
    assert module.input_dim == HIDDEN_DIM
    assert module.output_dim == HIDDEN_DIM
    assert module.transform_mode == (
        RELATION_TRANSFORM_PER_RELATION
    )


def test_constructor_defaults_control_mask_to_false() -> None:
    module = _module(
        control_relation_mask=None,
    )

    assert module.control_relation_mask == (
        False,
        False,
        False,
    )


def test_constructor_without_bias() -> None:
    module = _module(
        bias=False,
    )

    assert module.bias is False
    for linear in (
        module.relation_transforms.values()
    ):
        assert isinstance(
            linear,
            nn.Linear,
        )
        assert linear.bias is None


def test_constructor_builds_one_linear_per_relation() -> None:
    module = _module()

    assert isinstance(
        module.relation_transforms,
        nn.ModuleDict,
    )
    assert len(
        module.relation_transforms
    ) == RELATION_COUNT
    assert tuple(
        module.relation_transforms
    ) == module.relation_module_keys

    for relation_index in range(
        RELATION_COUNT
    ):
        linear = (
            module
            .module_for_relation_index(
                relation_index
            )
        )
        assert isinstance(
            linear,
            nn.Linear,
        )
        assert linear.in_features == (
            HIDDEN_DIM
        )
        assert linear.out_features == (
            HIDDEN_DIM
        )


def test_relation_module_keys_are_deterministic() -> None:
    module = _module()

    assert module.relation_module_keys == (
        "relation_0000_id_100",
        "relation_0001_id_200",
        "relation_0002_id_900",
    )


def test_constructor_has_no_hidden_activation_normalization_or_dropout() -> None:
    module = _module()

    assert tuple(
        dict(module.named_children())
    ) == ("relation_transforms",)

    for child in module.modules():
        if (
            child is module
            or child
            is module.relation_transforms
            or isinstance(child, nn.Linear)
        ):
            continue

        assert not isinstance(
            child,
            (
                nn.Dropout,
                nn.LayerNorm,
                nn.BatchNorm1d,
                nn.GELU,
                nn.ReLU,
            ),
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
    "relation_names",
    (
        (),
        ("spatial", "spatial"),
        ("spatial", ""),
        ("spatial", " "),
    ),
)
def test_constructor_rejects_invalid_relation_names(
    relation_names: tuple[str, ...],
) -> None:
    stable_ids = tuple(
        range(len(relation_names))
    )

    with pytest.raises(ValueError):
        _module(
            relation_names=relation_names,
            stable_relation_ids=(
                stable_ids
            ),
            control_relation_mask=tuple(
                False
                for _ in relation_names
            ),
        )


@pytest.mark.parametrize(
    "relation_names",
    (
        "spatial",
        b"spatial",
        123,
        None,
    ),
)
def test_constructor_rejects_nonsequence_relation_names(
    relation_names: Any,
) -> None:
    with pytest.raises(
        TypeError,
        match="sequence of strings",
    ):
        PerRelationTransform(
            hidden_dim=HIDDEN_DIM,
            relation_names=relation_names,
            stable_relation_ids=(100,),
        )


def test_constructor_rejects_nonstr_relation_name() -> None:
    with pytest.raises(
        ValueError,
        match="non-empty string",
    ):
        PerRelationTransform(
            hidden_dim=HIDDEN_DIM,
            relation_names=(
                "spatial",
                3,  # type: ignore[arg-type]
            ),
            stable_relation_ids=(
                100,
                200,
            ),
        )


def test_constructor_rejects_stable_id_length_mismatch() -> None:
    with pytest.raises(
        ValueError,
        match="align exactly",
    ):
        _module(
            stable_relation_ids=(
                100,
                200,
            ),
        )


@pytest.mark.parametrize(
    "stable_relation_ids",
    (
        (100, True, 900),
        (100, 2.5, 900),
        (100, "200", 900),
    ),
)
def test_constructor_rejects_noninteger_stable_ids(
    stable_relation_ids: Any,
) -> None:
    with pytest.raises(
        TypeError,
        match="must be an integer",
    ):
        _module(
            stable_relation_ids=(
                stable_relation_ids
            ),
        )


def test_constructor_rejects_negative_stable_id() -> None:
    with pytest.raises(
        ValueError,
        match="nonnegative",
    ):
        _module(
            stable_relation_ids=(
                100,
                -1,
                900,
            ),
        )


def test_constructor_rejects_duplicate_stable_ids() -> None:
    with pytest.raises(
        ValueError,
        match="must be unique",
    ):
        _module(
            stable_relation_ids=(
                100,
                100,
                900,
            ),
        )


def test_constructor_rejects_nonsequence_stable_ids() -> None:
    with pytest.raises(
        TypeError,
        match="sequence of integers",
    ):
        PerRelationTransform(
            hidden_dim=HIDDEN_DIM,
            relation_names=(
                "spatial",
            ),
            stable_relation_ids=100,  # type: ignore[arg-type]
        )


def test_constructor_rejects_control_mask_length_mismatch() -> None:
    with pytest.raises(
        ValueError,
        match="align exactly",
    ):
        _module(
            control_relation_mask=(
                False,
                True,
            ),
        )


def test_constructor_rejects_nonboolean_control_value() -> None:
    with pytest.raises(
        TypeError,
        match="must be a Boolean",
    ):
        _module(
            control_relation_mask=(
                False,
                1,  # type: ignore[arg-type]
                True,
            ),
        )


def test_constructor_rejects_nonsequence_control_mask() -> None:
    with pytest.raises(
        TypeError,
        match="sequence of Booleans",
    ):
        PerRelationTransform(
            hidden_dim=HIDDEN_DIM,
            relation_names=(
                "spatial",
            ),
            stable_relation_ids=(100,),
            control_relation_mask=True,  # type: ignore[arg-type]
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
    per_relation = (
        HIDDEN_DIM * HIDDEN_DIM
        + HIDDEN_DIM
    )

    assert module.parameter_count == (
        RELATION_COUNT * per_relation
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
        RELATION_COUNT
        * HIDDEN_DIM
        * HIDDEN_DIM
    )


# =============================================================================
# Construction from compiled registry
# =============================================================================


def test_from_compiled_registry_preserves_order_and_controls() -> None:
    registry = (
        FakeCompiledRelationRegistry()
    )

    module = (
        PerRelationTransform
        .from_compiled_registry(
            hidden_dim=HIDDEN_DIM,
            compiled_relation_registry=(
                registry
            ),
            bias=False,
        )
    )

    assert registry.validated
    assert module.relation_names == (
        RELATION_NAMES
    )
    assert module.stable_relation_ids == (
        STABLE_RELATION_IDS
    )
    assert module.control_relation_mask == (
        CONTROL_RELATION_MASK
    )
    assert module.bias is False


def test_from_compiled_registry_rejects_wrong_type() -> None:
    with pytest.raises(
        TypeError,
        match="CompiledRelationRegistry",
    ):
        (
            PerRelationTransform
            .from_compiled_registry(
                hidden_dim=HIDDEN_DIM,
                compiled_relation_registry=object(),  # type: ignore[arg-type]
            )
        )


def test_from_compiled_registry_propagates_validation_failure() -> None:
    class InvalidRegistry(
        FakeCompiledRelationRegistry
    ):
        def validate(self) -> None:
            raise RuntimeError(
                "invalid registry"
            )

    with pytest.raises(
        RuntimeError,
        match="invalid registry",
    ):
        (
            PerRelationTransform
            .from_compiled_registry(
                hidden_dim=HIDDEN_DIM,
                compiled_relation_registry=(
                    InvalidRegistry()
                ),
            )
        )


# =============================================================================
# Relation identity lookup
# =============================================================================


def test_relation_index_lookup_by_name() -> None:
    module = _module()

    assert module.relation_index(
        "spatial_adjacency"
    ) == 0
    assert module.relation_index(
        "temporal_lag"
    ) == 1
    assert module.relation_index(
        "random_placebo"
    ) == 2


def test_relation_index_lookup_by_stable_id() -> None:
    module = _module()

    assert module.relation_index(
        100
    ) == 0
    assert module.relation_index(
        200
    ) == 1
    assert module.relation_index(
        900
    ) == 2


def test_integer_relation_lookup_is_stable_id_not_dense_index() -> None:
    module = _module()

    with pytest.raises(
        KeyError,
        match="Unknown stable relation ID 0",
    ):
        module.relation_index(0)

    assert (
        module.module_for_relation_index(
            0
        )
        is module.module_for_relation(
            100
        )
    )


def test_relation_lookup_maps_are_read_only() -> None:
    module = _module()

    assert isinstance(
        module.relation_index_by_name,
        MappingProxyType,
    )
    assert isinstance(
        module.relation_index_by_stable_id,
        MappingProxyType,
    )

    with pytest.raises(TypeError):
        module.relation_index_by_name[
            "new"
        ] = 3  # type: ignore[index]

    with pytest.raises(TypeError):
        module.relation_index_by_stable_id[
            999
        ] = 3  # type: ignore[index]


def test_relation_index_rejects_unknown_name() -> None:
    with pytest.raises(
        KeyError,
        match="Unknown relation name",
    ):
        _module().relation_index(
            "unknown"
        )


def test_relation_index_rejects_unknown_stable_id() -> None:
    with pytest.raises(
        KeyError,
        match="Unknown stable relation ID",
    ):
        _module().relation_index(
            999
        )


@pytest.mark.parametrize(
    "relation",
    (
        True,
        1.5,
        None,
        object(),
    ),
)
def test_relation_index_rejects_invalid_type(
    relation: Any,
) -> None:
    with pytest.raises(
        TypeError,
        match="canonical name or stable integer ID",
    ):
        _module().relation_index(
            relation
        )


def test_module_for_relation_index_returns_exact_module() -> None:
    module = _module()

    for index, key in enumerate(
        module.relation_module_keys
    ):
        assert (
            module
            .module_for_relation_index(
                index
            )
            is module.relation_transforms[
                key
            ]
        )


@pytest.mark.parametrize(
    "relation_index",
    (
        -1,
        RELATION_COUNT,
    ),
)
def test_module_for_relation_index_rejects_out_of_range(
    relation_index: int,
) -> None:
    with pytest.raises(
        IndexError,
        match="outside the compiled relation range",
    ):
        (
            _module()
            .module_for_relation_index(
                relation_index
            )
        )


@pytest.mark.parametrize(
    "relation_index",
    (
        True,
        1.5,
        "0",
        None,
    ),
)
def test_module_for_relation_index_rejects_invalid_type(
    relation_index: Any,
) -> None:
    with pytest.raises(
        TypeError,
        match="must be an integer",
    ):
        (
            _module()
            .module_for_relation_index(
                relation_index
            )
        )


# =============================================================================
# Stacked parameters
# =============================================================================


def test_stacked_weight_has_relation_axis() -> None:
    module = _module()

    stacked = module.stacked_weight()

    assert stacked.shape == (
        RELATION_COUNT,
        HIDDEN_DIM,
        HIDDEN_DIM,
    )

    for relation_index in range(
        RELATION_COUNT
    ):
        assert torch.equal(
            stacked[relation_index],
            module
            .module_for_relation_index(
                relation_index
            )
            .weight,
        )


def test_stacked_bias_has_relation_axis() -> None:
    module = _module(
        bias=True,
    )

    stacked = module.stacked_bias()

    assert stacked is not None
    assert stacked.shape == (
        RELATION_COUNT,
        HIDDEN_DIM,
    )

    for relation_index in range(
        RELATION_COUNT
    ):
        assert torch.equal(
            stacked[relation_index],
            module
            .module_for_relation_index(
                relation_index
            )
            .bias,
        )


def test_stacked_bias_is_none_without_bias() -> None:
    assert (
        _module(
            bias=False
        ).stacked_bias()
        is None
    )


def test_stacked_weight_remains_connected_to_parameters() -> None:
    module = _module()
    module.stacked_weight().sum().backward()

    for relation_index in range(
        RELATION_COUNT
    ):
        weight_grad = (
            module
            .module_for_relation_index(
                relation_index
            )
            .weight
            .grad
        )
        assert weight_grad is not None
        assert torch.equal(
            weight_grad,
            torch.ones_like(
                weight_grad
            ),
        )


# =============================================================================
# Source gathering
# =============================================================================


def test_gather_source_state_matches_direct_indexing() -> None:
    module = _module()
    node_state = _node_state()
    source_index = _source_index()
    relation_index = (
        _edge_relation_index()
    )

    gathered = (
        module.gather_source_state(
            node_state,
            source_index,
            relation_index,
        )
    )

    assert torch.equal(
        gathered,
        node_state[source_index],
    )


def test_gather_source_state_preserves_duplicate_sources() -> None:
    module = _module()
    node_state = _node_state()
    source_index = torch.tensor(
        [2, 2, 2],
        dtype=torch.long,
    )
    relation_index = torch.tensor(
        [0, 1, 2],
        dtype=torch.long,
    )

    gathered = (
        module.gather_source_state(
            node_state,
            source_index,
            relation_index,
        )
    )

    assert torch.equal(
        gathered,
        node_state[2]
        .unsqueeze(0)
        .repeat(3, 1),
    )


def test_gather_source_state_supports_zero_edges() -> None:
    module = _module()
    node_state = _node_state()

    gathered = (
        module.gather_source_state(
            node_state,
            torch.empty(
                0,
                dtype=torch.long,
            ),
            torch.empty(
                0,
                dtype=torch.long,
            ),
        )
    )

    assert gathered.shape == (
        0,
        HIDDEN_DIM,
    )


def test_gather_source_state_supports_zero_nodes_and_edges() -> None:
    module = _module()

    gathered = (
        module.gather_source_state(
            _node_state(
                node_count=0
            ),
            torch.empty(
                0,
                dtype=torch.long,
            ),
            torch.empty(
                0,
                dtype=torch.long,
            ),
        )
    )

    assert gathered.shape == (
        0,
        HIDDEN_DIM,
    )


# =============================================================================
# Forward mathematics
# =============================================================================


def test_forward_matches_manual_relation_specific_maps() -> None:
    module = _module()
    _configure_distinct_maps(module)

    node_state = _node_state()
    source_index = _source_index()
    relation_index = (
        _edge_relation_index()
    )

    observed = module(
        node_state,
        source_index,
        relation_index,
    )
    expected = _manual_transform(
        module,
        node_state,
        source_index,
        relation_index,
    )

    assert torch.equal(
        observed,
        expected,
    )


def test_transform_source_state_matches_forward_after_gather() -> None:
    module = _module()
    node_state = _node_state()
    source_index = _source_index()
    relation_index = (
        _edge_relation_index()
    )

    source_state = (
        module.gather_source_state(
            node_state,
            source_index,
            relation_index,
        )
    )

    assert torch.equal(
        module(
            node_state,
            source_index,
            relation_index,
        ),
        module.transform_source_state(
            source_state,
            relation_index,
        ),
    )


def test_same_source_state_differs_across_relation_maps() -> None:
    module = _module(
        bias=False,
    )
    _configure_distinct_maps(module)
    node_state = _node_state()
    source_index = torch.tensor(
        [2, 2, 2],
        dtype=torch.long,
    )
    relation_index = torch.tensor(
        [0, 1, 2],
        dtype=torch.long,
    )

    output = module(
        node_state,
        source_index,
        relation_index,
    )

    assert torch.equal(
        output[0],
        node_state[2],
    )
    assert torch.equal(
        output[1],
        2.0 * node_state[2],
    )
    assert torch.equal(
        output[2],
        3.0 * node_state[2],
    )


def test_same_relation_applies_same_map_to_all_edges() -> None:
    module = _module()
    _configure_distinct_maps(module)
    node_state = _node_state()
    source_index = torch.tensor(
        [0, 1, 2],
        dtype=torch.long,
    )
    relation_index = torch.tensor(
        [1, 1, 1],
        dtype=torch.long,
    )

    output = module(
        node_state,
        source_index,
        relation_index,
    )
    linear = (
        module
        .module_for_relation_index(1)
    )
    expected = linear(
        node_state[source_index]
    )

    assert torch.equal(
        output,
        expected,
    )


def test_control_relation_uses_ordinary_independent_parameters() -> None:
    module = _module(
        bias=False,
    )
    _configure_distinct_maps(module)
    node_state = _node_state()
    source_index = torch.tensor(
        [0],
        dtype=torch.long,
    )
    relation_index = torch.tensor(
        [2],
        dtype=torch.long,
    )

    output = module(
        node_state,
        source_index,
        relation_index,
    )

    assert module.control_relation_mask[2]
    assert torch.equal(
        output[0],
        3.0 * node_state[0],
    )


def test_forward_output_shape_and_finiteness() -> None:
    module = _module()

    output = module(
        _node_state(),
        _source_index(),
        _edge_relation_index(),
    )

    assert output.shape == (
        int(_source_index().shape[0]),
        HIDDEN_DIM,
    )
    assert bool(
        torch.isfinite(output)
        .all()
        .item()
    )


def test_forward_supports_relation_with_zero_edges() -> None:
    module = _module()
    node_state = _node_state()
    source_index = torch.tensor(
        [0, 1, 2],
        dtype=torch.long,
    )
    relation_index = torch.tensor(
        [0, 0, 2],
        dtype=torch.long,
    )

    output = module(
        node_state,
        source_index,
        relation_index,
    )

    assert output.shape == (
        3,
        HIDDEN_DIM,
    )


def test_forward_supports_zero_edges() -> None:
    module = _module()

    output = module(
        _node_state(),
        torch.empty(
            0,
            dtype=torch.long,
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


def test_forward_supports_zero_nodes_and_zero_edges() -> None:
    module = _module()

    output = module(
        _node_state(
            node_count=0
        ),
        torch.empty(
            0,
            dtype=torch.long,
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


def test_forward_is_deterministic_in_train_and_eval() -> None:
    module = _module()
    node_state = _node_state()
    source_index = _source_index()
    relation_index = (
        _edge_relation_index()
    )

    module.train()
    train_first = module(
        node_state,
        source_index,
        relation_index,
    )
    train_second = module(
        node_state,
        source_index,
        relation_index,
    )

    module.eval()
    eval_first = module(
        node_state,
        source_index,
        relation_index,
    )
    eval_second = module(
        node_state,
        source_index,
        relation_index,
    )

    assert torch.equal(
        train_first,
        train_second,
    )
    assert torch.equal(
        eval_first,
        eval_second,
    )
    assert torch.equal(
        train_first,
        eval_first,
    )


def test_identity_weights_recover_gathered_state_for_every_relation() -> None:
    module = _module(
        bias=False,
    )

    with torch.no_grad():
        for linear in (
            module
            .relation_transforms
            .values()
        ):
            linear.weight.copy_(
                torch.eye(HIDDEN_DIM)
            )

    node_state = _node_state()
    source_index = _source_index()
    relation_index = (
        _edge_relation_index()
    )

    output = module(
        node_state,
        source_index,
        relation_index,
    )

    assert torch.equal(
        output,
        node_state[source_index],
    )


def test_zero_input_maps_to_relation_biases() -> None:
    module = _module(
        bias=True,
    )
    _configure_distinct_maps(module)
    node_state = torch.zeros(
        NODE_COUNT,
        HIDDEN_DIM,
    )
    source_index = torch.tensor(
        [0, 1, 2],
        dtype=torch.long,
    )
    relation_index = torch.tensor(
        [0, 1, 2],
        dtype=torch.long,
    )

    output = module(
        node_state,
        source_index,
        relation_index,
    )

    assert torch.equal(
        output,
        torch.stack(
            [
                torch.zeros(HIDDEN_DIM),
                torch.ones(HIDDEN_DIM),
                torch.full(
                    (HIDDEN_DIM,),
                    2.0,
                ),
            ]
        ),
    )


# =============================================================================
# Forward validation failures
# =============================================================================


def test_forward_rejects_non_tensor_node_state() -> None:
    with pytest.raises(
        TypeError,
        match="node_state must be a tensor",
    ):
        _module()(  # type: ignore[arg-type]
            [[0.0] * HIDDEN_DIM],
            torch.tensor(
                [0],
                dtype=torch.long,
            ),
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
    with pytest.raises(
        ValueError,
        match=r"shape \[M, H\]",
    ):
        _module()(
            node_state,
            torch.tensor(
                [0],
                dtype=torch.long,
            ),
            torch.tensor(
                [0],
                dtype=torch.long,
            ),
        )


def test_forward_rejects_node_state_width_mismatch() -> None:
    with pytest.raises(
        ValueError,
        match="feature width",
    ):
        _module()(
            torch.zeros(
                NODE_COUNT,
                HIDDEN_DIM + 1,
            ),
            _source_index(),
            _edge_relation_index(),
        )


def test_forward_rejects_nonfloating_node_state() -> None:
    with pytest.raises(
        ValueError,
        match="floating-point",
    ):
        _module()(
            torch.zeros(
                NODE_COUNT,
                HIDDEN_DIM,
                dtype=torch.long,
            ),
            _source_index(),
            _edge_relation_index(),
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
    node_state = _node_state()
    node_state[0, 0] = bad_value

    with pytest.raises(
        ValueError,
        match="finite",
    ):
        _module()(
            node_state,
            _source_index(),
            _edge_relation_index(),
        )


def test_forward_rejects_non_tensor_source_index() -> None:
    with pytest.raises(
        TypeError,
        match="source_index must be a tensor",
    ):
        _module()(  # type: ignore[arg-type]
            _node_state(),
            [0],
            torch.tensor(
                [0],
                dtype=torch.long,
            ),
        )


def test_forward_rejects_invalid_source_index_rank() -> None:
    with pytest.raises(
        ValueError,
        match=r"shape \[E\]",
    ):
        _module()(
            _node_state(),
            torch.tensor(
                [[0]],
                dtype=torch.long,
            ),
            torch.tensor(
                [0],
                dtype=torch.long,
            ),
        )


def test_forward_rejects_invalid_source_index_dtype() -> None:
    with pytest.raises(
        ValueError,
        match="torch.long",
    ):
        _module()(
            _node_state(),
            torch.tensor(
                [0],
                dtype=torch.int32,
            ),
            torch.tensor(
                [0],
                dtype=torch.long,
            ),
        )


@pytest.mark.parametrize(
    "source_index",
    (
        torch.tensor(
            [-1],
            dtype=torch.long,
        ),
        torch.tensor(
            [NODE_COUNT],
            dtype=torch.long,
        ),
    ),
)
def test_forward_rejects_source_index_out_of_range(
    source_index: torch.Tensor,
) -> None:
    with pytest.raises(
        ValueError,
        match="out-of-range",
    ):
        _module()(
            _node_state(),
            source_index,
            torch.tensor(
                [0],
                dtype=torch.long,
            ),
        )


def test_forward_rejects_non_tensor_relation_index() -> None:
    with pytest.raises(
        TypeError,
        match="edge_relation_index must be a tensor",
    ):
        _module()(  # type: ignore[arg-type]
            _node_state(),
            torch.tensor(
                [0],
                dtype=torch.long,
            ),
            [0],
        )


def test_forward_rejects_invalid_relation_index_rank() -> None:
    with pytest.raises(
        ValueError,
        match=r"shape \[E\]",
    ):
        _module()(
            _node_state(),
            torch.tensor(
                [0],
                dtype=torch.long,
            ),
            torch.tensor(
                [[0]],
                dtype=torch.long,
            ),
        )


def test_forward_rejects_invalid_relation_index_dtype() -> None:
    with pytest.raises(
        ValueError,
        match="torch.long",
    ):
        _module()(
            _node_state(),
            torch.tensor(
                [0],
                dtype=torch.long,
            ),
            torch.tensor(
                [0],
                dtype=torch.int32,
            ),
        )


@pytest.mark.parametrize(
    "relation_index",
    (
        torch.tensor(
            [-1],
            dtype=torch.long,
        ),
        torch.tensor(
            [RELATION_COUNT],
            dtype=torch.long,
        ),
    ),
)
def test_forward_rejects_relation_index_out_of_range(
    relation_index: torch.Tensor,
) -> None:
    with pytest.raises(
        ValueError,
        match="out-of-range",
    ):
        _module()(
            _node_state(),
            torch.tensor(
                [0],
                dtype=torch.long,
            ),
            relation_index,
        )


def test_forward_rejects_edge_vector_length_mismatch() -> None:
    with pytest.raises(
        ValueError,
        match="length must equal",
    ):
        _module()(
            _node_state(),
            torch.tensor(
                [0, 1],
                dtype=torch.long,
            ),
            torch.tensor(
                [0],
                dtype=torch.long,
            ),
        )


def test_transform_source_state_rejects_width_mismatch() -> None:
    with pytest.raises(
        ValueError,
        match="feature width",
    ):
        _module().transform_source_state(
            torch.zeros(
                2,
                HIDDEN_DIM + 1,
            ),
            torch.tensor(
                [0, 1],
                dtype=torch.long,
            ),
        )


def test_forward_rejects_input_parameter_dtype_mismatch() -> None:
    module = _module()

    with pytest.raises(
        ValueError,
        match="dtype must match",
    ):
        module(
            _node_state(
                dtype=torch.float64
            ),
            _source_index(),
            _edge_relation_index(),
        )


def test_double_module_accepts_float64() -> None:
    module = _module().double()

    output = module(
        _node_state(
            dtype=torch.float64
        ),
        _source_index(),
        _edge_relation_index(),
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
        module.module_for_relation_index(
            0
        ).weight[0, 0] = bad_value

    with pytest.raises(
        FloatingPointError,
        match="produced NaN or infinity",
    ):
        module(
            _node_state(),
            _source_index(),
            _edge_relation_index(),
        )


def test_assert_finite_parameters_passes() -> None:
    _module().assert_finite_parameters()


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
        module.module_for_relation_index(
            2
        ).weight[0, 0] = bad_value

    with pytest.raises(
        ValueError,
        match="NaN or infinity",
    ):
        module.assert_finite_parameters()


# =============================================================================
# Autograd and relation isolation
# =============================================================================


def test_backward_reaches_node_state() -> None:
    module = _module()
    node_state = _node_state(
        requires_grad=True,
    )

    module(
        node_state,
        _source_index(),
        _edge_relation_index(),
    ).square().mean().backward()

    assert node_state.grad is not None
    assert node_state.grad.shape == (
        node_state.shape
    )
    assert bool(
        torch.isfinite(
            node_state.grad
        ).all().item()
    )


def test_backward_reaches_all_relation_parameters_when_all_relations_present() -> None:
    module = _module()
    node_state = _node_state(
        requires_grad=True,
    )

    module(
        node_state,
        _source_index(),
        _edge_relation_index(),
    ).sum().backward()

    for relation_index in range(
        RELATION_COUNT
    ):
        linear = (
            module
            .module_for_relation_index(
                relation_index
            )
        )
        assert linear.weight.grad is not None
        assert bool(
            torch.isfinite(
                linear.weight.grad
            ).all().item()
        )
        assert linear.bias is not None
        assert linear.bias.grad is not None


def test_absent_relation_receives_explicit_zero_gradient() -> None:
    module = _module()
    node_state = _node_state(
        requires_grad=True,
    )
    source_index = torch.tensor(
        [0, 1, 2],
        dtype=torch.long,
    )
    relation_index = torch.tensor(
        [0, 0, 2],
        dtype=torch.long,
    )

    module(
        node_state,
        source_index,
        relation_index,
    ).sum().backward()

    absent = (
        module
        .module_for_relation_index(1)
    )
    assert absent.weight.grad is not None
    assert torch.equal(
        absent.weight.grad,
        torch.zeros_like(
            absent.weight.grad
        ),
    )
    assert absent.bias is not None
    assert absent.bias.grad is not None
    assert torch.equal(
        absent.bias.grad,
        torch.zeros_like(
            absent.bias.grad
        ),
    )


def test_mutating_one_relation_changes_only_its_edges() -> None:
    module = _module(
        bias=False,
    )
    node_state = _node_state()
    source_index = _source_index()
    relation_index = (
        _edge_relation_index()
    )

    before = module(
        node_state,
        source_index,
        relation_index,
    )

    with torch.no_grad():
        module.module_for_relation_index(
            1
        ).weight += torch.eye(
            HIDDEN_DIM
        )

    after = module(
        node_state,
        source_index,
        relation_index,
    )
    relation_one = (
        relation_index == 1
    )
    other = ~relation_one

    assert not torch.equal(
        before[relation_one],
        after[relation_one],
    )
    assert torch.equal(
        before[other],
        after[other],
    )


def test_zero_edge_backward_connects_all_parameters() -> None:
    module = _module()
    node_state = _node_state(
        node_count=0,
        requires_grad=True,
    )

    output = module(
        node_state,
        torch.empty(
            0,
            dtype=torch.long,
        ),
        torch.empty(
            0,
            dtype=torch.long,
        ),
    )
    assert output.requires_grad
    output.sum().backward()

    # The explicit empty-edge branch preserves parameter connectivity.
    # No source row was gathered, so the empty node-state tensor is not
    # required to receive a gradient object.
    assert node_state.grad is None

    for parameter in module.parameters():
        assert parameter.grad is not None
        assert torch.equal(
            parameter.grad,
            torch.zeros_like(
                parameter.grad
            ),
        )


def test_forward_gradcheck() -> None:
    module = _module().double()
    node_state = torch.randn(
        NODE_COUNT,
        HIDDEN_DIM,
        dtype=torch.float64,
        requires_grad=True,
    )
    source_index = _source_index()
    relation_index = (
        _edge_relation_index()
    )

    assert torch.autograd.gradcheck(
        lambda state: module(
            state,
            source_index,
            relation_index,
        ),
        (node_state,),
        eps=1e-6,
        atol=1e-5,
        rtol=1e-3,
    )


# =============================================================================
# Architecture and parameter fingerprints
# =============================================================================


def test_relation_architecture_dict_is_exact() -> None:
    module = _module()

    assert (
        module
        .relation_architecture_dict(2)
        == {
            "schema_version": (
                PER_RELATION_TRANSFORM_SCHEMA_VERSION
            ),
            "transform_mode": (
                RELATION_TRANSFORM_PER_RELATION
            ),
            "relation_index": 2,
            "relation_name": (
                "random_placebo"
            ),
            "stable_relation_id": 900,
            "is_control": True,
            "module_key": (
                "relation_0002_id_900"
            ),
            "hidden_dim": HIDDEN_DIM,
            "bias": True,
            "operation": (
                "relation_specific_linear_transform"
            ),
            "activation": None,
            "normalization": None,
            "dropout": 0.0,
        }
    )


def test_relation_architecture_fingerprints_are_read_only() -> None:
    fingerprints = (
        _module()
        .relation_architecture_fingerprints()
    )

    assert isinstance(
        fingerprints,
        MappingProxyType,
    )
    assert tuple(fingerprints) == (
        RELATION_NAMES
    )

    with pytest.raises(TypeError):
        fingerprints[
            "new"
        ] = "value"  # type: ignore[index]


def test_relation_parameter_fingerprints_are_read_only() -> None:
    fingerprints = (
        _module()
        .relation_parameter_fingerprints()
    )

    assert isinstance(
        fingerprints,
        MappingProxyType,
    )
    assert tuple(fingerprints) == (
        RELATION_NAMES
    )

    with pytest.raises(TypeError):
        fingerprints[
            "new"
        ] = "value"  # type: ignore[index]


def test_relation_fingerprints_are_stable_under_seed() -> None:
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(123)
        first = _module()

    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(123)
        second = _module()

    assert dict(
        first
        .relation_architecture_fingerprints()
    ) == dict(
        second
        .relation_architecture_fingerprints()
    )
    assert dict(
        first
        .relation_parameter_fingerprints()
    ) == dict(
        second
        .relation_parameter_fingerprints()
    )


def test_mutating_one_relation_changes_only_its_parameter_fingerprint() -> None:
    module = _module()
    before = dict(
        module
        .relation_parameter_fingerprints()
    )

    with torch.no_grad():
        module.module_for_relation(
            "temporal_lag"
        ).weight[0, 0] += 1.0

    after = dict(
        module
        .relation_parameter_fingerprints()
    )

    assert before[
        "temporal_lag"
    ] != after[
        "temporal_lag"
    ]
    assert before[
        "spatial_adjacency"
    ] == after[
        "spatial_adjacency"
    ]
    assert before[
        "random_placebo"
    ] == after[
        "random_placebo"
    ]


def test_architecture_dict_is_complete() -> None:
    module = _module()
    architecture = (
        module.architecture_dict()
    )

    assert architecture[
        "schema_version"
    ] == PER_RELATION_TRANSFORM_SCHEMA_VERSION
    assert architecture[
        "transform_mode"
    ] == RELATION_TRANSFORM_PER_RELATION
    assert architecture[
        "hidden_dim"
    ] == HIDDEN_DIM
    assert architecture[
        "bias"
    ] is True
    assert architecture[
        "relation_count"
    ] == RELATION_COUNT
    assert architecture[
        "relation_names"
    ] == list(
        RELATION_NAMES
    )
    assert architecture[
        "stable_relation_ids"
    ] == list(
        STABLE_RELATION_IDS
    )
    assert architecture[
        "control_relation_mask"
    ] == list(
        CONTROL_RELATION_MASK
    )
    assert architecture[
        "relation_module_keys"
    ] == list(
        module.relation_module_keys
    )
    assert architecture[
        "parameter_sharing"
    ] == "independent_linear_map_per_relation"
    assert architecture[
        "activation"
    ] is None
    assert architecture[
        "normalization"
    ] is None
    assert architecture[
        "dropout"
    ] == 0.0


def test_architecture_fingerprint_is_stable() -> None:
    first = _module()
    second = _module()

    assert first.architecture_dict() == (
        second.architecture_dict()
    )
    assert first.architecture_fingerprint() == (
        second.architecture_fingerprint()
    )


@pytest.mark.parametrize(
    "builder",
    (
        lambda: _module(
            hidden_dim=5
        ),
        lambda: _module(
            bias=False
        ),
        lambda: _module(
            relation_names=(
                "temporal_lag",
                "spatial_adjacency",
                "random_placebo",
            ),
            stable_relation_ids=(
                200,
                100,
                900,
            ),
            control_relation_mask=(
                False,
                False,
                True,
            ),
        ),
        lambda: _module(
            control_relation_mask=(
                False,
                True,
                True,
            )
        ),
    ),
)
def test_architecture_fingerprint_changes_with_contract(
    builder: Any,
) -> None:
    assert (
        _module()
        .architecture_fingerprint()
        != builder()
        .architecture_fingerprint()
    )


def test_parameter_fingerprint_is_reproducible_under_seed() -> None:
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(321)
        first = _module()

    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(321)
        second = _module()

    assert first.parameter_fingerprint() == (
        second.parameter_fingerprint()
    )


def test_parameter_fingerprint_changes_after_mutation() -> None:
    module = _module()
    architecture = (
        module.architecture_fingerprint()
    )
    before = (
        module.parameter_fingerprint()
    )

    with torch.no_grad():
        module.module_for_relation_index(
            0
        ).weight[0, 0] += 1.0

    assert module.parameter_fingerprint() != (
        before
    )
    assert module.architecture_fingerprint() == (
        architecture
    )


# =============================================================================
# State dict and representation
# =============================================================================


def test_state_dict_keys_are_semantic_and_stable_with_bias() -> None:
    module = _module(
        bias=True,
    )

    assert tuple(
        module.state_dict()
    ) == (
        "relation_transforms.relation_0000_id_100.weight",
        "relation_transforms.relation_0000_id_100.bias",
        "relation_transforms.relation_0001_id_200.weight",
        "relation_transforms.relation_0001_id_200.bias",
        "relation_transforms.relation_0002_id_900.weight",
        "relation_transforms.relation_0002_id_900.bias",
    )


def test_state_dict_keys_are_semantic_and_stable_without_bias() -> None:
    module = _module(
        bias=False,
    )

    assert tuple(
        module.state_dict()
    ) == (
        "relation_transforms.relation_0000_id_100.weight",
        "relation_transforms.relation_0001_id_200.weight",
        "relation_transforms.relation_0002_id_900.weight",
    )


def test_state_dict_round_trip_preserves_outputs_and_fingerprints() -> None:
    source = _module()
    target = _module()

    target.load_state_dict(
        source.state_dict(),
        strict=True,
    )

    node_state = _node_state()
    source_index = _source_index()
    relation_index = (
        _edge_relation_index()
    )

    assert torch.equal(
        source(
            node_state,
            source_index,
            relation_index,
        ),
        target(
            node_state,
            source_index,
            relation_index,
        ),
    )
    assert source.parameter_fingerprint() == (
        target.parameter_fingerprint()
    )
    assert dict(
        source
        .relation_parameter_fingerprints()
    ) == dict(
        target
        .relation_parameter_fingerprints()
    )


def test_state_dict_rejects_different_relation_order_under_strict_load() -> None:
    source = _module()
    target = _module(
        relation_names=(
            "temporal_lag",
            "spatial_adjacency",
            "random_placebo",
        ),
        stable_relation_ids=(
            200,
            100,
            900,
        ),
        control_relation_mask=(
            False,
            False,
            True,
        ),
    )

    with pytest.raises(
        RuntimeError,
    ):
        target.load_state_dict(
            source.state_dict(),
            strict=True,
        )


def test_extra_repr_contains_contract_identity() -> None:
    module = _module(
        bias=False,
    )
    representation = (
        module.extra_repr()
    )

    assert (
        f"hidden_dim={HIDDEN_DIM}"
        in representation
    )
    assert (
        f"relation_count={RELATION_COUNT}"
        in representation
    )
    assert (
        "control_relation_count=1"
        in representation
    )
    assert "bias=False" in representation
    assert (
        RELATION_TRANSFORM_PER_RELATION
        in representation
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
                device="cuda"
            ),
            _source_index(
                device="cpu"
            ),
            _edge_relation_index(
                device="cuda"
            ),
        )


@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA is unavailable.",
)
def test_forward_rejects_relation_index_device_mismatch() -> None:
    module = _module().cuda()

    with pytest.raises(
        ValueError,
        match="edge_relation_index must share one device",
    ):
        module(
            _node_state(
                device="cuda"
            ),
            _source_index(
                device="cuda"
            ),
            _edge_relation_index(
                device="cpu"
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
                device="cpu"
            ),
            _source_index(
                device="cpu"
            ),
            _edge_relation_index(
                device="cpu"
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
    relation_index = (
        _edge_relation_index()
    )

    cpu_output = cpu_module(
        node_state,
        source_index,
        relation_index,
    )
    cuda_output = cuda_module(
        node_state.cuda(),
        source_index.cuda(),
        relation_index.cuda(),
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

    module(
        node_state,
        _source_index(
            device="cuda"
        ),
        _edge_relation_index(
            device="cuda"
        ),
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
