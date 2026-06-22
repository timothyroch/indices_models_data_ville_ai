"""
Contract tests for structural edge normalization.

Target repository path:
    urban_resilience_models/
        v2_hazard_conditioned_functional_ugnn/
            tests/
                test_edge_normalization.py

Implementation under test:
    urban_resilience_models/
        v2_hazard_conditioned_functional_ugnn/
            functional_message_passing/
                edge_normalization.py

This suite freezes the bounded V2.0 normalization contract independently from
relation transforms, semantic edge weights, hazard-conditioned gates, learned
attention, message construction, and target-node aggregation.

Covered behavior:

- canonical versus implemented normalization modes;
- construction from functional-message-passing configuration;
- exact multiplicative identity coefficients for ``none``;
- directed source- and target-degree diagnostics;
- parallel edges, self-loops, isolated nodes, and zero-edge graphs;
- strict input type, rank, dtype, device, length, and range validation;
- output shape, dtype, device, finiteness, nonnegativity, and identity guards;
- parameter-free state, architecture, and fingerprints;
- metadata-preserving output construction;
- optional CUDA execution.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pytest
import torch
from torch import nn

from urban_resilience_models.v2_hazard_conditioned_functional_ugnn.constants import (
    CANONICAL_EDGE_NORMALIZATION_TYPES,
    EDGE_NORMALIZATION_NONE,
    V2_0_IMPLEMENTED_EDGE_NORMALIZATION_TYPES,
)
from urban_resilience_models.v2_hazard_conditioned_functional_ugnn.functional_message_passing import (
    edge_normalization as edge_normalization_module,
)
from urban_resilience_models.v2_hazard_conditioned_functional_ugnn.functional_message_passing.edge_normalization import (
    EDGE_NORMALIZATION_SCHEMA_VERSION,
    EdgeNormalization,
)


NODE_COUNT = 6
EDGE_COUNT = 8


# =============================================================================
# Controlled upstream contracts
# =============================================================================


class FakeFunctionalMessagePassingConfig:
    def __init__(
        self,
        *,
        edge_normalization_type: str = EDGE_NORMALIZATION_NONE,
        enabled: bool = True,
        validation_error: Exception | None = None,
        implementation_error: Exception | None = None,
    ) -> None:
        self.edge_normalization_type = edge_normalization_type
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


class FakeFunctionalMessagePassingInputs:
    def __init__(
        self,
        *,
        num_nodes: int,
        source_index: torch.Tensor,
        target_index: torch.Tensor,
        dtype: torch.dtype = torch.float32,
        device: torch.device | None = None,
        num_edges: int | None = None,
    ) -> None:
        self.num_nodes = num_nodes
        self.source_index = source_index
        self.target_index = target_index
        self.dtype = dtype
        self.device = (
            source_index.device
            if device is None
            else device
        )
        self.num_edges = (
            int(source_index.shape[0])
            if num_edges is None
            else num_edges
        )


@dataclass
class FakeStructuralEdgeNormalizationOutput:
    coefficients: torch.Tensor
    source_inputs: object
    normalization_mode: str
    encoder_architecture_fingerprint: str
    source_degree: torch.Tensor | None = None
    target_degree: torch.Tensor | None = None


@pytest.fixture(autouse=True)
def _patch_upstream_types(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        edge_normalization_module,
        "FunctionalMessagePassingConfig",
        FakeFunctionalMessagePassingConfig,
    )
    monkeypatch.setattr(
        edge_normalization_module,
        "FunctionalMessagePassingInputs",
        FakeFunctionalMessagePassingInputs,
    )
    monkeypatch.setattr(
        edge_normalization_module,
        "StructuralEdgeNormalizationOutput",
        FakeStructuralEdgeNormalizationOutput,
    )


# =============================================================================
# Helpers
# =============================================================================


def _source_index(
    *,
    device: torch.device | str = "cpu",
) -> torch.Tensor:
    # Parallel 0→1 edges, a self-loop 2→2, and node 5 isolated.
    return torch.tensor(
        [0, 0, 2, 2, 3, 4, 4, 1],
        dtype=torch.long,
        device=device,
    )


def _target_index(
    *,
    device: torch.device | str = "cpu",
) -> torch.Tensor:
    return torch.tensor(
        [1, 1, 2, 3, 4, 0, 3, 4],
        dtype=torch.long,
        device=device,
    )


def _inputs(
    *,
    num_nodes: int = NODE_COUNT,
    source_index: torch.Tensor | None = None,
    target_index: torch.Tensor | None = None,
    dtype: torch.dtype = torch.float32,
    device: torch.device | None = None,
    num_edges: int | None = None,
) -> FakeFunctionalMessagePassingInputs:
    resolved_source = (
        _source_index()
        if source_index is None
        else source_index
    )
    resolved_target = (
        _target_index(
            device=resolved_source.device
        )
        if target_index is None
        else target_index
    )

    return FakeFunctionalMessagePassingInputs(
        num_nodes=num_nodes,
        source_index=resolved_source,
        target_index=resolved_target,
        dtype=dtype,
        device=device,
        num_edges=num_edges,
    )


def _canonical_unimplemented_mode() -> str | None:
    for mode in (
        CANONICAL_EDGE_NORMALIZATION_TYPES
    ):
        if mode not in (
            V2_0_IMPLEMENTED_EDGE_NORMALIZATION_TYPES
        ):
            return mode
    return None


# =============================================================================
# Public identity and constructor
# =============================================================================


def test_schema_version_is_nonempty() -> None:
    assert isinstance(
        EDGE_NORMALIZATION_SCHEMA_VERSION,
        str,
    )
    assert EDGE_NORMALIZATION_SCHEMA_VERSION.strip()


def test_class_is_torch_module() -> None:
    assert issubclass(
        EdgeNormalization,
        nn.Module,
    )


def test_default_constructor_selects_none() -> None:
    normalization = EdgeNormalization()

    assert normalization.mode == (
        EDGE_NORMALIZATION_NONE
    )
    assert normalization.is_identity


def test_constructor_accepts_none() -> None:
    normalization = EdgeNormalization(
        mode=EDGE_NORMALIZATION_NONE
    )

    assert normalization.mode == (
        EDGE_NORMALIZATION_NONE
    )


def test_constructor_strips_whitespace() -> None:
    normalization = EdgeNormalization(
        mode=f"  {EDGE_NORMALIZATION_NONE}  "
    )

    assert normalization.mode == (
        EDGE_NORMALIZATION_NONE
    )


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
        EdgeNormalization(mode=mode)


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
        EdgeNormalization(mode=mode)


def test_constructor_rejects_unknown_mode() -> None:
    with pytest.raises(
        ValueError,
        match="Unknown edge normalization mode",
    ):
        EdgeNormalization(
            mode="unknown_normalization"
        )


def test_constructor_rejects_canonical_unimplemented_mode() -> None:
    mode = _canonical_unimplemented_mode()

    if mode is None:
        pytest.skip(
            "No canonical unimplemented normalization mode exists."
        )

    with pytest.raises(
        NotImplementedError,
        match="canonical but not implemented",
    ):
        EdgeNormalization(mode=mode)


def test_implemented_modes_are_canonical() -> None:
    assert set(
        V2_0_IMPLEMENTED_EDGE_NORMALIZATION_TYPES
    ).issubset(
        set(CANONICAL_EDGE_NORMALIZATION_TYPES)
    )
    assert EDGE_NORMALIZATION_NONE in (
        V2_0_IMPLEMENTED_EDGE_NORMALIZATION_TYPES
    )


# =============================================================================
# Construction from config
# =============================================================================


def test_from_config_builds_normalization() -> None:
    config = FakeFunctionalMessagePassingConfig(
        edge_normalization_type=(
            EDGE_NORMALIZATION_NONE
        ),
        enabled=True,
    )

    normalization = EdgeNormalization.from_config(
        config=config
    )

    assert config.validate_calls == 1
    assert (
        config.assert_implemented_calls
        == 1
    )
    assert normalization.mode == (
        EDGE_NORMALIZATION_NONE
    )


def test_from_disabled_config_does_not_assert_implemented() -> None:
    config = FakeFunctionalMessagePassingConfig(
        edge_normalization_type=(
            EDGE_NORMALIZATION_NONE
        ),
        enabled=False,
    )

    normalization = EdgeNormalization.from_config(
        config=config
    )

    assert config.validate_calls == 1
    assert (
        config.assert_implemented_calls
        == 0
    )
    assert normalization.is_identity


def test_from_config_rejects_wrong_type() -> None:
    with pytest.raises(
        TypeError,
        match="FunctionalMessagePassingConfig",
    ):
        EdgeNormalization.from_config(
            config=object()  # type: ignore[arg-type]
        )


def test_from_config_propagates_validation_error() -> None:
    config = FakeFunctionalMessagePassingConfig(
        validation_error=RuntimeError(
            "config invalid"
        )
    )

    with pytest.raises(
        RuntimeError,
        match="config invalid",
    ):
        EdgeNormalization.from_config(
            config=config
        )


def test_from_config_propagates_implementation_error() -> None:
    config = FakeFunctionalMessagePassingConfig(
        implementation_error=(
            NotImplementedError(
                "normalization unavailable"
            )
        )
    )

    with pytest.raises(
        NotImplementedError,
        match="normalization unavailable",
    ):
        EdgeNormalization.from_config(
            config=config
        )


# =============================================================================
# Parameter-free identity and fingerprints
# =============================================================================


def test_module_is_parameter_free() -> None:
    normalization = EdgeNormalization()

    assert normalization.parameter_count == 0
    assert (
        normalization.trainable_parameter_count
        == 0
    )
    assert tuple(
        normalization.parameters()
    ) == ()
    assert normalization.state_dict() == {}


def test_architecture_dict_is_exact() -> None:
    normalization = EdgeNormalization()

    assert normalization.architecture_dict() == {
        "schema_version": (
            EDGE_NORMALIZATION_SCHEMA_VERSION
        ),
        "mode": EDGE_NORMALIZATION_NONE,
        "implemented_formula": "n_e = 1",
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


def test_architecture_fingerprint_is_stable() -> None:
    first = EdgeNormalization()
    second = EdgeNormalization()

    assert first.architecture_dict() == (
        second.architecture_dict()
    )
    assert first.architecture_fingerprint() == (
        second.architecture_fingerprint()
    )


def test_parameter_fingerprint_is_stable() -> None:
    first = EdgeNormalization()
    second = EdgeNormalization()

    assert first.parameter_fingerprint() == (
        second.parameter_fingerprint()
    )


def test_architecture_and_parameter_fingerprints_are_distinct() -> None:
    normalization = EdgeNormalization()

    assert (
        normalization.architecture_fingerprint()
        != normalization.parameter_fingerprint()
    )


def test_assert_finite_parameters_is_noop_for_baseline() -> None:
    EdgeNormalization().assert_finite_parameters()


def test_assert_finite_parameters_rejects_nonzero_parameter_contract() -> None:
    class InvalidNormalization(
        EdgeNormalization
    ):
        @property
        def parameter_count(self) -> int:
            return 1

    with pytest.raises(
        RuntimeError,
        match="must remain parameter-free",
    ):
        InvalidNormalization().assert_finite_parameters()


# =============================================================================
# Directed degree semantics
# =============================================================================


def test_compute_degrees_exact_values() -> None:
    normalization = EdgeNormalization()
    inputs = _inputs()

    source_degree, target_degree = (
        normalization.compute_degrees(
            inputs
        )
    )

    assert torch.equal(
        source_degree,
        torch.tensor(
            [2, 1, 2, 1, 2, 0],
            dtype=torch.long,
        ),
    )
    assert torch.equal(
        target_degree,
        torch.tensor(
            [1, 2, 1, 2, 2, 0],
            dtype=torch.long,
        ),
    )


def test_degree_vectors_have_expected_contract() -> None:
    inputs = _inputs()
    source_degree, target_degree = (
        EdgeNormalization()
        .compute_degrees(inputs)
    )

    assert source_degree.shape == (
        NODE_COUNT,
    )
    assert target_degree.shape == (
        NODE_COUNT,
    )
    assert source_degree.dtype == (
        torch.long
    )
    assert target_degree.dtype == (
        torch.long
    )
    assert source_degree.device == (
        inputs.device
    )
    assert target_degree.device == (
        inputs.device
    )


def test_degree_sums_equal_edge_count() -> None:
    inputs = _inputs()
    source_degree, target_degree = (
        EdgeNormalization()
        .compute_degrees(inputs)
    )

    assert int(
        source_degree.sum().item()
    ) == inputs.num_edges
    assert int(
        target_degree.sum().item()
    ) == inputs.num_edges


def test_parallel_edges_are_counted_individually() -> None:
    source_degree, target_degree = (
        EdgeNormalization()
        .compute_degrees(_inputs())
    )

    assert source_degree[0].item() == 2
    assert target_degree[1].item() == 2


def test_self_loop_contributes_to_both_directed_degrees() -> None:
    source_degree, target_degree = (
        EdgeNormalization()
        .compute_degrees(_inputs())
    )

    # Node 2 owns edges 2→2 and 2→3; it receives the 2→2 self-loop.
    assert source_degree[2].item() == 2
    assert target_degree[2].item() == 1


def test_isolated_node_has_zero_source_and_target_degree() -> None:
    source_degree, target_degree = (
        EdgeNormalization()
        .compute_degrees(_inputs())
    )

    assert source_degree[5].item() == 0
    assert target_degree[5].item() == 0


def test_edge_order_does_not_change_degrees() -> None:
    inputs = _inputs()
    permutation = torch.tensor(
        [7, 2, 0, 6, 1, 5, 3, 4],
        dtype=torch.long,
    )
    permuted = _inputs(
        source_index=(
            inputs.source_index[
                permutation
            ]
        ),
        target_index=(
            inputs.target_index[
                permutation
            ]
        ),
    )

    first = EdgeNormalization().compute_degrees(
        inputs
    )
    second = EdgeNormalization().compute_degrees(
        permuted
    )

    assert torch.equal(
        first[0],
        second[0],
    )
    assert torch.equal(
        first[1],
        second[1],
    )


def test_zero_edge_graph_returns_zero_degrees() -> None:
    inputs = _inputs(
        source_index=torch.empty(
            0,
            dtype=torch.long,
        ),
        target_index=torch.empty(
            0,
            dtype=torch.long,
        ),
        num_edges=0,
    )

    source_degree, target_degree = (
        EdgeNormalization()
        .compute_degrees(inputs)
    )

    assert torch.equal(
        source_degree,
        torch.zeros(
            NODE_COUNT,
            dtype=torch.long,
        ),
    )
    assert torch.equal(
        target_degree,
        torch.zeros(
            NODE_COUNT,
            dtype=torch.long,
        ),
    )


# =============================================================================
# Identity coefficients
# =============================================================================


@pytest.mark.parametrize(
    "dtype",
    (
        torch.float32,
        torch.float64,
        torch.bfloat16,
    ),
)
def test_compute_coefficients_returns_exact_ones(
    dtype: torch.dtype,
) -> None:
    inputs = _inputs(dtype=dtype)

    coefficients = (
        EdgeNormalization()
        .compute_coefficients(inputs)
    )

    assert coefficients.shape == (
        EDGE_COUNT,
    )
    assert coefficients.dtype == dtype
    assert coefficients.device == (
        inputs.device
    )
    assert torch.equal(
        coefficients,
        torch.ones(
            EDGE_COUNT,
            dtype=dtype,
        ),
    )


def test_identity_coefficients_are_relation_agnostic() -> None:
    inputs = _inputs()
    first = (
        EdgeNormalization()
        .compute_coefficients(inputs)
    )

    # Structural normalization has no access to edge relation identity.
    inputs.edge_relation_index = torch.tensor(
        [2, 0, 1, 2, 2, 1, 0, 0],
        dtype=torch.long,
    )
    second = (
        EdgeNormalization()
        .compute_coefficients(inputs)
    )

    assert torch.equal(first, second)


def test_identity_coefficients_are_independent_of_semantic_metadata() -> None:
    inputs = _inputs()
    first = (
        EdgeNormalization()
        .compute_coefficients(inputs)
    )

    inputs.semantic_edge_weight = torch.linspace(
        0.1,
        0.8,
        steps=EDGE_COUNT,
    )
    inputs.hazard_query = torch.randn(
        NODE_COUNT,
        3,
    )
    second = (
        EdgeNormalization()
        .compute_coefficients(inputs)
    )

    assert torch.equal(first, second)


def test_zero_edge_graph_returns_empty_coefficients() -> None:
    inputs = _inputs(
        source_index=torch.empty(
            0,
            dtype=torch.long,
        ),
        target_index=torch.empty(
            0,
            dtype=torch.long,
        ),
        num_edges=0,
        dtype=torch.float64,
    )

    coefficients = (
        EdgeNormalization()
        .compute_coefficients(inputs)
    )

    assert coefficients.shape == (0,)
    assert coefficients.dtype == (
        torch.float64
    )


# =============================================================================
# Metadata-preserving forward output
# =============================================================================


def test_forward_constructs_complete_output() -> None:
    normalization = EdgeNormalization()
    inputs = _inputs()

    output = normalization(inputs)

    assert isinstance(
        output,
        FakeStructuralEdgeNormalizationOutput,
    )
    assert output.source_inputs is inputs
    assert output.normalization_mode == (
        EDGE_NORMALIZATION_NONE
    )
    assert torch.equal(
        output.coefficients,
        torch.ones(
            EDGE_COUNT,
            dtype=inputs.dtype,
        ),
    )
    assert torch.equal(
        output.source_degree,
        torch.tensor(
            [2, 1, 2, 1, 2, 0],
            dtype=torch.long,
        ),
    )
    assert torch.equal(
        output.target_degree,
        torch.tensor(
            [1, 2, 1, 2, 2, 0],
            dtype=torch.long,
        ),
    )
    assert (
        output.encoder_architecture_fingerprint
        == normalization.architecture_fingerprint()
    )


def test_forward_zero_edge_output() -> None:
    normalization = EdgeNormalization()
    inputs = _inputs(
        source_index=torch.empty(
            0,
            dtype=torch.long,
        ),
        target_index=torch.empty(
            0,
            dtype=torch.long,
        ),
        num_edges=0,
    )

    output = normalization(inputs)

    assert output.coefficients.shape == (
        0,
    )
    assert torch.equal(
        output.source_degree,
        torch.zeros(
            NODE_COUNT,
            dtype=torch.long,
        ),
    )
    assert torch.equal(
        output.target_degree,
        torch.zeros(
            NODE_COUNT,
            dtype=torch.long,
        ),
    )


def test_forward_is_deterministic() -> None:
    normalization = EdgeNormalization()
    inputs = _inputs()

    first = normalization(inputs)
    second = normalization(inputs)

    assert torch.equal(
        first.coefficients,
        second.coefficients,
    )
    assert torch.equal(
        first.source_degree,
        second.source_degree,
    )
    assert torch.equal(
        first.target_degree,
        second.target_degree,
    )
    assert (
        first.encoder_architecture_fingerprint
        == second.encoder_architecture_fingerprint
    )


# =============================================================================
# Input validation failures
# =============================================================================


def test_rejects_wrong_input_type() -> None:
    with pytest.raises(
        TypeError,
        match="FunctionalMessagePassingInputs",
    ):
        EdgeNormalization().compute_degrees(
            object()  # type: ignore[arg-type]
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
    inputs = _inputs(
        num_nodes=num_nodes,
        source_index=torch.empty(
            0,
            dtype=torch.long,
        ),
        target_index=torch.empty(
            0,
            dtype=torch.long,
        ),
        num_edges=0,
    )

    with pytest.raises(
        ValueError,
        match="at least one node",
    ):
        EdgeNormalization().compute_degrees(
            inputs
        )


@pytest.mark.parametrize(
    "dtype",
    (
        torch.long,
        torch.int32,
        torch.bool,
    ),
)
def test_rejects_nonfloating_input_dtype(
    dtype: torch.dtype,
) -> None:
    with pytest.raises(
        ValueError,
        match="floating-point dtype",
    ):
        EdgeNormalization().compute_coefficients(
            _inputs(dtype=dtype)
        )


@pytest.mark.parametrize(
    "field",
    (
        "source_index",
        "target_index",
    ),
)
def test_rejects_non_tensor_edge_index(
    field: str,
) -> None:
    inputs = _inputs()
    setattr(
        inputs,
        field,
        [0, 1],  # type: ignore[arg-type]
    )

    with pytest.raises(
        TypeError,
        match=f"{field} must be a tensor",
    ):
        EdgeNormalization().compute_degrees(
            inputs
        )


@pytest.mark.parametrize(
    "field",
    (
        "source_index",
        "target_index",
    ),
)
def test_rejects_invalid_edge_index_rank(
    field: str,
) -> None:
    inputs = _inputs()
    setattr(
        inputs,
        field,
        torch.tensor(
            [[0, 1]],
            dtype=torch.long,
        ),
    )

    with pytest.raises(
        ValueError,
        match=r"shape \[E\]",
    ):
        EdgeNormalization().compute_degrees(
            inputs
        )


@pytest.mark.parametrize(
    "field",
    (
        "source_index",
        "target_index",
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
def test_rejects_invalid_edge_index_dtype(
    field: str,
    dtype: torch.dtype,
) -> None:
    inputs = _inputs()
    original = getattr(
        inputs,
        field,
    )
    setattr(
        inputs,
        field,
        original.to(dtype=dtype),
    )

    with pytest.raises(
        ValueError,
        match="torch.long",
    ):
        EdgeNormalization().compute_degrees(
            inputs
        )


@pytest.mark.parametrize(
    "field",
    (
        "source_index",
        "target_index",
    ),
)
def test_rejects_edge_index_length_mismatch(
    field: str,
) -> None:
    inputs = _inputs()
    setattr(
        inputs,
        field,
        getattr(inputs, field)[:-1],
    )

    with pytest.raises(
        ValueError,
        match="length must equal the edge count",
    ):
        EdgeNormalization().compute_degrees(
            inputs
        )


@pytest.mark.parametrize(
    "field",
    (
        "source_index",
        "target_index",
    ),
)
@pytest.mark.parametrize(
    "bad_index",
    (
        -1,
        NODE_COUNT,
    ),
)
def test_rejects_out_of_range_edge_index(
    field: str,
    bad_index: int,
) -> None:
    inputs = _inputs()
    index = getattr(
        inputs,
        field,
    ).clone()
    index[0] = bad_index
    setattr(
        inputs,
        field,
        index,
    )

    with pytest.raises(
        ValueError,
        match="out-of-range",
    ):
        EdgeNormalization().compute_degrees(
            inputs
        )


def test_rejects_declared_edge_count_mismatch() -> None:
    inputs = _inputs(
        num_edges=EDGE_COUNT + 1
    )

    with pytest.raises(
        ValueError,
        match="length must equal the edge count",
    ):
        EdgeNormalization().compute_degrees(
            inputs
        )


# =============================================================================
# Internal degree guards
# =============================================================================


def test_rejects_wrong_source_degree_shape(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original = torch.bincount
    call_count = 0

    def fake_bincount(
        values: torch.Tensor,
        *,
        minlength: int,
    ) -> torch.Tensor:
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            return torch.zeros(
                minlength + 1,
                dtype=torch.long,
                device=values.device,
            )
        return original(
            values,
            minlength=minlength,
        )

    monkeypatch.setattr(
        edge_normalization_module.torch,
        "bincount",
        fake_bincount,
    )

    with pytest.raises(
        RuntimeError,
        match="Source-degree computation returned an invalid shape",
    ):
        EdgeNormalization().compute_degrees(
            _inputs()
        )


def test_rejects_wrong_target_degree_shape(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original = torch.bincount
    call_count = 0

    def fake_bincount(
        values: torch.Tensor,
        *,
        minlength: int,
    ) -> torch.Tensor:
        nonlocal call_count
        call_count += 1
        if call_count == 2:
            return torch.zeros(
                minlength + 1,
                dtype=torch.long,
                device=values.device,
            )
        return original(
            values,
            minlength=minlength,
        )

    monkeypatch.setattr(
        edge_normalization_module.torch,
        "bincount",
        fake_bincount,
    )

    with pytest.raises(
        RuntimeError,
        match="Target-degree computation returned an invalid shape",
    ):
        EdgeNormalization().compute_degrees(
            _inputs()
        )


def test_rejects_wrong_degree_dtype(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_bincount(
        values: torch.Tensor,
        *,
        minlength: int,
    ) -> torch.Tensor:
        return torch.zeros(
            minlength,
            dtype=torch.float32,
            device=values.device,
        )

    monkeypatch.setattr(
        edge_normalization_module.torch,
        "bincount",
        fake_bincount,
    )

    with pytest.raises(
        RuntimeError,
        match="must return torch.long",
    ):
        EdgeNormalization().compute_degrees(
            _inputs()
        )


def test_rejects_source_degree_sum_mismatch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original = torch.bincount
    call_count = 0

    def fake_bincount(
        values: torch.Tensor,
        *,
        minlength: int,
    ) -> torch.Tensor:
        nonlocal call_count
        call_count += 1
        result = original(
            values,
            minlength=minlength,
        )
        if call_count == 1:
            result = result.clone()
            result[0] += 1
        return result

    monkeypatch.setattr(
        edge_normalization_module.torch,
        "bincount",
        fake_bincount,
    )

    with pytest.raises(
        RuntimeError,
        match="Source degrees do not sum",
    ):
        EdgeNormalization().compute_degrees(
            _inputs()
        )


def test_rejects_target_degree_sum_mismatch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original = torch.bincount
    call_count = 0

    def fake_bincount(
        values: torch.Tensor,
        *,
        minlength: int,
    ) -> torch.Tensor:
        nonlocal call_count
        call_count += 1
        result = original(
            values,
            minlength=minlength,
        )
        if call_count == 2:
            result = result.clone()
            result[0] += 1
        return result

    monkeypatch.setattr(
        edge_normalization_module.torch,
        "bincount",
        fake_bincount,
    )

    with pytest.raises(
        RuntimeError,
        match="Target degrees do not sum",
    ):
        EdgeNormalization().compute_degrees(
            _inputs()
        )


# =============================================================================
# Internal coefficient guards
# =============================================================================


def test_rejects_wrong_coefficient_shape(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_ones(
        size: Any,
        *,
        dtype: torch.dtype,
        device: torch.device,
    ) -> torch.Tensor:
        return torch.ones(
            EDGE_COUNT + 1,
            dtype=dtype,
            device=device,
        )

    original_ones = torch.ones
    monkeypatch.setattr(
        edge_normalization_module.torch,
        "ones",
        fake_ones,
    )

    # Avoid recursion in fake_ones.
    def corrected_fake_ones(
        size: Any,
        *,
        dtype: torch.dtype,
        device: torch.device,
    ) -> torch.Tensor:
        return original_ones(
            EDGE_COUNT + 1,
            dtype=dtype,
            device=device,
        )

    monkeypatch.setattr(
        edge_normalization_module.torch,
        "ones",
        corrected_fake_ones,
    )

    with pytest.raises(
        RuntimeError,
        match="returned shape",
    ):
        EdgeNormalization().compute_coefficients(
            _inputs()
        )


def test_rejects_wrong_coefficient_dtype(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original_ones = torch.ones

    def fake_ones(
        size: Any,
        *,
        dtype: torch.dtype,
        device: torch.device,
    ) -> torch.Tensor:
        return original_ones(
            size,
            dtype=torch.float64,
            device=device,
        )

    monkeypatch.setattr(
        edge_normalization_module.torch,
        "ones",
        fake_ones,
    )

    with pytest.raises(
        RuntimeError,
        match="changed dtype",
    ):
        EdgeNormalization().compute_coefficients(
            _inputs(dtype=torch.float32)
        )


def test_rejects_nonfinite_coefficients(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original_ones = torch.ones

    def fake_ones(
        size: Any,
        *,
        dtype: torch.dtype,
        device: torch.device,
    ) -> torch.Tensor:
        result = original_ones(
            size,
            dtype=dtype,
            device=device,
        )
        result[0] = float("nan")
        return result

    monkeypatch.setattr(
        edge_normalization_module.torch,
        "ones",
        fake_ones,
    )

    with pytest.raises(
        FloatingPointError,
        match="NaN or infinity",
    ):
        EdgeNormalization().compute_coefficients(
            _inputs()
        )


def test_rejects_negative_coefficients(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original_ones = torch.ones

    def fake_ones(
        size: Any,
        *,
        dtype: torch.dtype,
        device: torch.device,
    ) -> torch.Tensor:
        result = original_ones(
            size,
            dtype=dtype,
            device=device,
        )
        result[0] = -1.0
        return result

    monkeypatch.setattr(
        edge_normalization_module.torch,
        "ones",
        fake_ones,
    )

    with pytest.raises(
        FloatingPointError,
        match="must be nonnegative",
    ):
        EdgeNormalization().compute_coefficients(
            _inputs()
        )


def test_none_rejects_nonidentity_coefficients(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original_ones = torch.ones

    def fake_ones(
        size: Any,
        *,
        dtype: torch.dtype,
        device: torch.device,
    ) -> torch.Tensor:
        return original_ones(
            size,
            dtype=dtype,
            device=device,
        ) * 0.5

    monkeypatch.setattr(
        edge_normalization_module.torch,
        "ones",
        fake_ones,
    )

    with pytest.raises(
        RuntimeError,
        match="must return exact identity",
    ):
        EdgeNormalization().compute_coefficients(
            _inputs()
        )


def test_corrupted_mode_reaches_defensive_dispatch_guard() -> None:
    normalization = EdgeNormalization()
    normalization.mode = "corrupted"

    with pytest.raises(
        RuntimeError,
        match="unsupported mode",
    ):
        normalization.compute_coefficients(
            _inputs()
        )


# =============================================================================
# Representation
# =============================================================================


def test_extra_repr_contains_contract_identity() -> None:
    representation = (
        EdgeNormalization()
        .extra_repr()
    )

    assert EDGE_NORMALIZATION_NONE in (
        representation
    )
    assert "parameter_count=0" in (
        representation
    )
    assert "degree_diagnostics=True" in (
        representation
    )


# =============================================================================
# Optional CUDA
# =============================================================================


@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA is unavailable.",
)
@pytest.mark.parametrize(
    "field",
    (
        "source_index",
        "target_index",
    ),
)
def test_rejects_edge_index_device_mismatch(
    field: str,
) -> None:
    inputs = _inputs(
        source_index=_source_index(
            device="cuda"
        ),
        target_index=_target_index(
            device="cuda"
        ),
        device=torch.device("cuda"),
    )
    setattr(
        inputs,
        field,
        getattr(inputs, field).cpu(),
    )

    with pytest.raises(
        ValueError,
        match="must share one device",
    ):
        EdgeNormalization().compute_degrees(
            inputs
        )


@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA is unavailable.",
)
def test_cuda_results_match_cpu() -> None:
    cpu_inputs = _inputs()
    cuda_inputs = _inputs(
        source_index=_source_index(
            device="cuda"
        ),
        target_index=_target_index(
            device="cuda"
        ),
        device=torch.device("cuda"),
    )
    normalization = EdgeNormalization()

    cpu_output = normalization(
        cpu_inputs
    )
    cuda_output = normalization(
        cuda_inputs
    )

    assert torch.equal(
        cpu_output.coefficients,
        cuda_output.coefficients.cpu(),
    )
    assert torch.equal(
        cpu_output.source_degree,
        cuda_output.source_degree.cpu(),
    )
    assert torch.equal(
        cpu_output.target_degree,
        cuda_output.target_degree.cpu(),
    )


@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA is unavailable.",
)
def test_rejects_degree_device_change(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original = torch.bincount

    def fake_bincount(
        values: torch.Tensor,
        *,
        minlength: int,
    ) -> torch.Tensor:
        return original(
            values.cpu(),
            minlength=minlength,
        )

    monkeypatch.setattr(
        edge_normalization_module.torch,
        "bincount",
        fake_bincount,
    )

    inputs = _inputs(
        source_index=_source_index(
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
        EdgeNormalization().compute_degrees(
            inputs
        )


@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA is unavailable.",
)
def test_rejects_coefficient_device_change(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original_ones = torch.ones

    def fake_ones(
        size: Any,
        *,
        dtype: torch.dtype,
        device: torch.device,
    ) -> torch.Tensor:
        return original_ones(
            size,
            dtype=dtype,
            device="cpu",
        )

    monkeypatch.setattr(
        edge_normalization_module.torch,
        "ones",
        fake_ones,
    )

    inputs = _inputs(
        source_index=_source_index(
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
        EdgeNormalization().compute_coefficients(
            inputs
        )
