"""
Contract tests for the concat-projection fusion baseline.

Target repository path:
    urban_resilience_models/
        v2_hazard_conditioned_functional_ugnn/
            tests/
                test_concat_projection.py

Implementation under test:
    urban_resilience_models/
        v2_hazard_conditioned_functional_ugnn/
            fusion/
                concat_projection.py

This suite isolates the mathematical baseline from schema extraction and
orchestration. It freezes:

- canonical component identities and ordering;
- strict ordered-mapping semantics;
- independent per-component projections;
- deterministic concatenation;
- final fusion-network structure;
- optional retention of the concatenated state;
- optional exact input-value fingerprints;
- immutable algorithm outputs;
- architecture and parameter identities;
- finite forward and backward behavior;
- empty-batch and mixed floating-dtype handling;
- state-dict round trips and parameter-corruption detection.
"""

from __future__ import annotations

from collections import OrderedDict
from types import MappingProxyType
from typing import Any

import pytest
import torch
from torch import nn

from urban_resilience_models.v2_hazard_conditioned_functional_ugnn.fusion.component_projection import (
    ComponentProjection,
)
from urban_resilience_models.v2_hazard_conditioned_functional_ugnn.fusion.concat_projection import (
    CANONICAL_FUSION_COMPONENT_ORDER,
    CONCAT_PROJECTION_FUSION_SCHEMA_VERSION,
    CONCAT_PROJECTION_OUTPUT_SCHEMA_VERSION,
    ConcatProjectionFusion,
    ConcatProjectionFusionOutput,
    FUSION_COMPONENT_HAZARD_CONTEXT,
    FUSION_COMPONENT_HAZARD_MEMORY_STATE,
    FUSION_COMPONENT_MEMORY_STATE,
    FUSION_COMPONENT_NODE_TYPE_EMBEDDING,
    FUSION_COMPONENT_STATIC_STATE,
    canonical_component_order,
)


STATIC_DIM = 5
MEMORY_DIM = 7
HAZARD_MEMORY_DIM = 6
HAZARD_CONTEXT_DIM = 8
NODE_TYPE_DIM = 4
OUTPUT_DIM = 11


# =============================================================================
# Helpers
# =============================================================================


def _canonical_dims(
    *,
    include_static: bool = True,
    include_memory: bool = True,
    include_hazard_memory: bool = False,
    include_hazard_context: bool = True,
    include_node_type: bool = False,
) -> OrderedDict[str, int]:
    dims: dict[str, int] = {}

    if include_static:
        dims[FUSION_COMPONENT_STATIC_STATE] = STATIC_DIM

    if include_memory:
        dims[FUSION_COMPONENT_MEMORY_STATE] = MEMORY_DIM

    if include_hazard_memory:
        dims[FUSION_COMPONENT_HAZARD_MEMORY_STATE] = (
            HAZARD_MEMORY_DIM
        )

    if include_hazard_context:
        dims[FUSION_COMPONENT_HAZARD_CONTEXT] = (
            HAZARD_CONTEXT_DIM
        )

    if include_node_type:
        dims[FUSION_COMPONENT_NODE_TYPE_EMBEDDING] = (
            NODE_TYPE_DIM
        )

    return OrderedDict(
        (
            name,
            dims[name],
        )
        for name in CANONICAL_FUSION_COMPONENT_ORDER
        if name in dims
    )


def _fusion(
    *,
    component_input_dims: OrderedDict[str, int] | None = None,
    component_order: tuple[str, ...] | None = None,
    output_dim: int = OUTPUT_DIM,
    dropout: float = 0.0,
    layer_norm: bool = True,
    retain_concatenated_state: bool = False,
    record_input_fingerprint: bool = False,
) -> ConcatProjectionFusion:
    dims = (
        _canonical_dims()
        if component_input_dims is None
        else component_input_dims
    )

    return ConcatProjectionFusion(
        component_input_dims=dims,
        output_dim=output_dim,
        component_order=component_order,
        dropout=dropout,
        layer_norm=layer_norm,
        retain_concatenated_state=(
            retain_concatenated_state
        ),
        record_input_fingerprint=(
            record_input_fingerprint
        ),
    )


def _components(
    rows: int = 4,
    *,
    dims: OrderedDict[str, int] | None = None,
    dtype: torch.dtype = torch.float32,
    offset: float = 0.0,
    requires_grad: bool = False,
) -> OrderedDict[str, torch.Tensor]:
    resolved = (
        _canonical_dims()
        if dims is None
        else dims
    )

    components: OrderedDict[
        str,
        torch.Tensor,
    ] = OrderedDict()

    for index, (name, width) in enumerate(
        resolved.items()
    ):
        values = (
            torch.arange(
                rows * width,
                dtype=dtype,
            )
            .reshape(rows, width)
            / 10.0
            + offset
            + float(index)
        )
        values.requires_grad_(requires_grad)
        components[name] = values

    return components


def _output_contract(
    *,
    rows: int = 3,
    component_order: tuple[str, ...] = (
        FUSION_COMPONENT_STATIC_STATE,
        FUSION_COMPONENT_MEMORY_STATE,
    ),
    output_dim: int = OUTPUT_DIM,
    retain_concat: bool = True,
    input_value_fingerprint: str | None = (
        "input-values"
    ),
) -> ConcatProjectionFusionOutput:
    projected = OrderedDict(
        (
            name,
            torch.full(
                (rows, output_dim),
                float(index + 1),
            ),
        )
        for index, name in enumerate(
            component_order
        )
    )

    concatenated = (
        torch.cat(
            list(projected.values()),
            dim=-1,
        )
        if retain_concat
        else None
    )

    return ConcatProjectionFusionOutput(
        fused_state=torch.zeros(
            rows,
            output_dim,
        ),
        projected_components=projected,
        component_order=component_order,
        architecture_fingerprint="architecture",
        concatenated_state=concatenated,
        input_value_fingerprint=(
            input_value_fingerprint
        ),
    )


# =============================================================================
# Published identity and component vocabulary
# =============================================================================


def test_schema_versions_are_nonempty() -> None:
    for value in (
        CONCAT_PROJECTION_FUSION_SCHEMA_VERSION,
        CONCAT_PROJECTION_OUTPUT_SCHEMA_VERSION,
    ):
        assert isinstance(value, str)
        assert value.strip()


def test_canonical_component_order_is_exact() -> None:
    assert CANONICAL_FUSION_COMPONENT_ORDER == (
        "static_state",
        "memory_state",
        "hazard_memory_state",
        "hazard_context",
        "node_type_embedding",
    )


def test_component_constants_match_canonical_order() -> None:
    assert (
        FUSION_COMPONENT_STATIC_STATE,
        FUSION_COMPONENT_MEMORY_STATE,
        FUSION_COMPONENT_HAZARD_MEMORY_STATE,
        FUSION_COMPONENT_HAZARD_CONTEXT,
        FUSION_COMPONENT_NODE_TYPE_EMBEDDING,
    ) == CANONICAL_FUSION_COMPONENT_ORDER


def test_canonical_component_order_filters_known_names() -> None:
    resolved = canonical_component_order(
        (
            FUSION_COMPONENT_HAZARD_CONTEXT,
            FUSION_COMPONENT_STATIC_STATE,
            FUSION_COMPONENT_MEMORY_STATE,
        )
    )

    assert resolved == (
        FUSION_COMPONENT_STATIC_STATE,
        FUSION_COMPONENT_MEMORY_STATE,
        FUSION_COMPONENT_HAZARD_CONTEXT,
    )


def test_canonical_component_order_rejects_unknown_name() -> None:
    with pytest.raises(
        ValueError,
        match="undefined",
    ):
        canonical_component_order(
            (
                FUSION_COMPONENT_STATIC_STATE,
                "experimental_component",
            )
        )


def test_canonical_component_order_rejects_duplicates() -> None:
    with pytest.raises(
        ValueError,
        match="duplicates",
    ):
        canonical_component_order(
            (
                FUSION_COMPONENT_STATIC_STATE,
                FUSION_COMPONENT_STATIC_STATE,
            )
        )


def test_canonical_component_order_rejects_blank_name() -> None:
    with pytest.raises(
        ValueError,
        match="non-empty",
    ):
        canonical_component_order(
            (" ",)
        )


# =============================================================================
# Constructor and architecture assembly
# =============================================================================


def test_constructor_derives_canonical_order() -> None:
    dims = OrderedDict(
        (
            (
                FUSION_COMPONENT_HAZARD_CONTEXT,
                HAZARD_CONTEXT_DIM,
            ),
            (
                FUSION_COMPONENT_STATIC_STATE,
                STATIC_DIM,
            ),
            (
                FUSION_COMPONENT_MEMORY_STATE,
                MEMORY_DIM,
            ),
        )
    )

    fusion = _fusion(
        component_input_dims=dims,
    )

    assert fusion.component_order == (
        FUSION_COMPONENT_STATIC_STATE,
        FUSION_COMPONENT_MEMORY_STATE,
        FUSION_COMPONENT_HAZARD_CONTEXT,
    )


def test_constructor_preserves_explicit_experimental_order() -> None:
    dims = OrderedDict(
        (
            ("experimental_a", 3),
            ("experimental_b", 4),
        )
    )
    fusion = _fusion(
        component_input_dims=dims,
        component_order=(
            "experimental_b",
            "experimental_a",
        ),
    )

    assert fusion.component_order == (
        "experimental_b",
        "experimental_a",
    )
    assert tuple(
        fusion.component_input_dims
    ) == fusion.component_order
    assert fusion.component_input_dims[
        "experimental_b"
    ] == 4
    assert fusion.component_input_dims[
        "experimental_a"
    ] == 3


def test_constructor_freezes_component_dimensions() -> None:
    dims = _canonical_dims()
    fusion = _fusion(
        component_input_dims=dims,
    )

    assert isinstance(
        fusion.component_input_dims,
        MappingProxyType,
    )

    with pytest.raises(TypeError):
        fusion.component_input_dims[
            FUSION_COMPONENT_STATIC_STATE
        ] = 99  # type: ignore[index]


def test_constructor_builds_one_projection_per_component() -> None:
    fusion = _fusion()

    assert isinstance(
        fusion.component_projections,
        nn.ModuleDict,
    )
    assert tuple(
        fusion.component_projections
    ) == fusion.component_order

    for name in fusion.component_order:
        projection = (
            fusion.component_projections[name]
        )
        assert isinstance(
            projection,
            ComponentProjection,
        )
        assert projection.component_name == name
        assert projection.input_dim == (
            fusion.component_input_dims[name]
        )
        assert projection.output_dim == OUTPUT_DIM


def test_constructor_builds_explicit_fusion_network() -> None:
    fusion = _fusion(
        dropout=0.25,
        layer_norm=True,
    )

    assert isinstance(
        fusion.fusion_network,
        nn.Sequential,
    )
    assert tuple(
        fusion.fusion_network._modules
    ) == (
        "linear_in",
        "activation",
        "dropout",
        "linear_out",
        "normalization",
    )
    assert isinstance(
        fusion.fusion_network.linear_in,
        nn.Linear,
    )
    assert isinstance(
        fusion.fusion_network.activation,
        nn.GELU,
    )
    assert isinstance(
        fusion.fusion_network.dropout,
        nn.Dropout,
    )
    assert isinstance(
        fusion.fusion_network.linear_out,
        nn.Linear,
    )
    assert isinstance(
        fusion.fusion_network.normalization,
        nn.LayerNorm,
    )
    assert fusion.fusion_network.dropout.p == 0.25


def test_constructor_without_layer_norm_uses_identity() -> None:
    fusion = _fusion(
        layer_norm=False,
    )

    assert isinstance(
        fusion.fusion_network.normalization,
        nn.Identity,
    )

    for projection in (
        fusion.component_projections.values()
    ):
        assert isinstance(
            projection.normalization_layer,
            nn.Identity,
        )


def test_public_dimension_properties() -> None:
    fusion = _fusion()

    assert fusion.component_count == len(
        fusion.component_order
    )
    assert fusion.fusion_input_dim == (
        fusion.component_count
        * OUTPUT_DIM
    )
    assert fusion.output_dim == OUTPUT_DIM
    assert fusion.device == torch.device("cpu")
    assert fusion.dtype == torch.float32


@pytest.mark.parametrize(
    "component_input_dims",
    (
        None,
        [],
    ),
)
def test_constructor_rejects_non_mapping_dimensions(
    component_input_dims: Any,
) -> None:
    with pytest.raises(TypeError, match="mapping"):
        ConcatProjectionFusion(
            component_input_dims=component_input_dims,  # type: ignore[arg-type]
            output_dim=OUTPUT_DIM,
        )


def test_constructor_rejects_empty_dimensions() -> None:
    with pytest.raises(
        ValueError,
        match="At least one",
    ):
        ConcatProjectionFusion(
            component_input_dims={},
            output_dim=OUTPUT_DIM,
        )


@pytest.mark.parametrize(
    "width",
    (
        0,
        -1,
        True,
        1.5,
    ),
)
def test_constructor_rejects_invalid_component_width(
    width: Any,
) -> None:
    with pytest.raises(
        ValueError,
        match="positive integer",
    ):
        ConcatProjectionFusion(
            component_input_dims={
                FUSION_COMPONENT_STATIC_STATE: width,
            },
            output_dim=OUTPUT_DIM,
        )


@pytest.mark.parametrize(
    "output_dim",
    (
        0,
        -1,
        True,
        1.5,
    ),
)
def test_constructor_rejects_invalid_output_dim(
    output_dim: Any,
) -> None:
    with pytest.raises(
        ValueError,
        match="positive integer",
    ):
        _fusion(
            output_dim=output_dim,
        )


@pytest.mark.parametrize(
    "dropout",
    (
        -0.1,
        1.0,
        1.1,
        float("nan"),
        float("inf"),
    ),
)
def test_constructor_rejects_invalid_dropout(
    dropout: float,
) -> None:
    with pytest.raises(ValueError):
        _fusion(dropout=dropout)


def test_constructor_rejects_boolean_dropout() -> None:
    with pytest.raises(TypeError, match="numeric"):
        _fusion(dropout=True)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "field",
    (
        "layer_norm",
        "retain_concatenated_state",
        "record_input_fingerprint",
    ),
)
def test_constructor_rejects_non_boolean_flags(
    field: str,
) -> None:
    kwargs: dict[str, Any] = {
        "component_input_dims": _canonical_dims(),
        "output_dim": OUTPUT_DIM,
        field: 1,
    }

    with pytest.raises(TypeError, match="Boolean"):
        ConcatProjectionFusion(**kwargs)


def test_constructor_rejects_order_name_mismatch() -> None:
    dims = OrderedDict(
        (
            ("a", 3),
            ("b", 4),
        )
    )

    with pytest.raises(
        ValueError,
        match="exactly the same",
    ):
        _fusion(
            component_input_dims=dims,
            component_order=("a",),
        )


def test_constructor_rejects_duplicate_explicit_order() -> None:
    dims = OrderedDict(
        (
            ("a", 3),
            ("b", 4),
        )
    )

    with pytest.raises(
        ValueError,
        match="duplicates",
    ):
        _fusion(
            component_input_dims=dims,
            component_order=("a", "a"),
        )


# =============================================================================
# Forward path
# =============================================================================


def test_forward_shape_order_and_finiteness() -> None:
    fusion = _fusion()
    fusion.eval()
    components = _components(rows=4)

    output = fusion(components)

    assert isinstance(
        output,
        ConcatProjectionFusionOutput,
    )
    assert output.fused_state.shape == (
        4,
        OUTPUT_DIM,
    )
    assert output.item_count == 4
    assert output.output_dim == OUTPUT_DIM
    assert output.component_count == len(
        fusion.component_order
    )
    assert output.component_order == (
        fusion.component_order
    )
    assert tuple(
        output.projected_components
    ) == fusion.component_order
    assert bool(
        torch.isfinite(
            output.fused_state
        ).all().item()
    )

    for state in (
        output.projected_components.values()
    ):
        assert state.shape == (
            4,
            OUTPUT_DIM,
        )
        assert bool(
            torch.isfinite(state).all().item()
        )


def test_forward_does_not_retain_concat_by_default() -> None:
    output = _fusion()(
        _components(rows=3)
    )

    assert output.concatenated_state is None


def test_forward_retains_exact_concat_when_enabled() -> None:
    fusion = _fusion(
        retain_concatenated_state=True,
    )
    fusion.eval()

    output = fusion(
        _components(rows=3)
    )

    assert output.concatenated_state is not None
    expected = torch.cat(
        [
            output.projected_components[name]
            for name in output.component_order
        ],
        dim=-1,
    )
    assert torch.equal(
        output.concatenated_state,
        expected,
    )
    assert output.concatenated_state.shape == (
        3,
        fusion.fusion_input_dim,
    )


def test_forward_records_input_fingerprint_when_enabled() -> None:
    fusion = _fusion(
        record_input_fingerprint=True,
    )
    fusion.eval()
    components = _components(rows=3)

    first = fusion(components)
    second = fusion(components)

    assert first.input_value_fingerprint is not None
    assert first.input_value_fingerprint == (
        second.input_value_fingerprint
    )


def test_input_fingerprint_changes_with_values() -> None:
    fusion = _fusion(
        record_input_fingerprint=True,
    )
    fusion.eval()

    first = fusion(
        _components(rows=3, offset=0.0)
    )
    second = fusion(
        _components(rows=3, offset=1.0)
    )

    assert first.input_value_fingerprint != (
        second.input_value_fingerprint
    )


def test_forward_does_not_record_input_fingerprint_by_default() -> None:
    output = _fusion()(
        _components(rows=2)
    )

    assert output.input_value_fingerprint is None


def test_forward_supports_empty_batches() -> None:
    fusion = _fusion()
    fusion.eval()

    output = fusion(
        _components(rows=0)
    )

    assert output.fused_state.shape == (
        0,
        OUTPUT_DIM,
    )

    for state in (
        output.projected_components.values()
    ):
        assert state.shape == (
            0,
            OUTPUT_DIM,
        )


def test_forward_casts_component_inputs_to_module_dtype() -> None:
    fusion = _fusion()
    fusion.eval()
    components = _components(
        rows=3,
        dtype=torch.float64,
    )

    output = fusion(components)

    assert output.fused_state.dtype == torch.float32

    for state in (
        output.projected_components.values()
    ):
        assert state.dtype == torch.float32


def test_double_module_produces_float64_outputs() -> None:
    fusion = _fusion().double()
    fusion.eval()
    components = _components(
        rows=3,
        dtype=torch.float64,
    )

    output = fusion(components)

    assert fusion.dtype == torch.float64
    assert output.fused_state.dtype == torch.float64


def test_eval_mode_is_deterministic_with_dropout() -> None:
    fusion = _fusion(
        dropout=0.75,
    )
    fusion.eval()
    components = _components(rows=4)

    first = fusion(components)
    second = fusion(components)

    assert torch.equal(
        first.fused_state,
        second.fused_state,
    )

    for name in fusion.component_order:
        assert torch.equal(
            first.projected_components[name],
            second.projected_components[name],
        )


def test_output_architecture_fingerprint_matches_module() -> None:
    fusion = _fusion()
    output = fusion(
        _components(rows=2)
    )

    assert output.architecture_fingerprint == (
        fusion.architecture_fingerprint()
    )


def test_forward_accepts_explicit_experimental_components() -> None:
    dims = OrderedDict(
        (
            ("expert_context", 3),
            ("operational_context", 4),
        )
    )
    order = (
        "operational_context",
        "expert_context",
    )
    fusion = _fusion(
        component_input_dims=dims,
        component_order=order,
    )
    components = OrderedDict(
        (
            (
                "operational_context",
                torch.ones(2, 4),
            ),
            (
                "expert_context",
                torch.ones(2, 3),
            ),
        )
    )

    output = fusion(components)

    assert output.component_order == order
    assert output.fused_state.shape == (
        2,
        OUTPUT_DIM,
    )


# =============================================================================
# Forward input-policy failures
# =============================================================================


def test_forward_rejects_non_mapping_components() -> None:
    fusion = _fusion()

    with pytest.raises(TypeError, match="mapping"):
        fusion(  # type: ignore[arg-type]
            [torch.zeros(2, STATIC_DIM)]
        )


def test_forward_rejects_missing_component() -> None:
    fusion = _fusion()
    components = _components(rows=2)
    components.pop(
        FUSION_COMPONENT_MEMORY_STATE
    )

    with pytest.raises(
        ValueError,
        match="Missing",
    ):
        fusion(components)


def test_forward_rejects_unexpected_component() -> None:
    fusion = _fusion()
    components = _components(rows=2)
    components["unexpected"] = torch.zeros(
        2,
        3,
    )

    with pytest.raises(
        ValueError,
        match="unexpected",
    ):
        fusion(components)


def test_forward_rejects_correct_names_in_wrong_order() -> None:
    fusion = _fusion()
    components = _components(rows=2)
    reversed_components = OrderedDict(
        reversed(
            tuple(components.items())
        )
    )

    with pytest.raises(
        ValueError,
        match="mapping order",
    ):
        fusion(reversed_components)


def test_forward_rejects_non_tensor_component() -> None:
    fusion = _fusion()
    components: OrderedDict[
        str,
        Any,
    ] = _components(rows=2)
    components[
        FUSION_COMPONENT_STATIC_STATE
    ] = [[0.0] * STATIC_DIM]

    with pytest.raises(TypeError, match="tensor"):
        fusion(components)


@pytest.mark.parametrize(
    "bad_tensor",
    (
        torch.zeros(STATIC_DIM),
        torch.zeros(2, STATIC_DIM, 1),
    ),
)
def test_forward_rejects_invalid_component_rank(
    bad_tensor: torch.Tensor,
) -> None:
    fusion = _fusion()
    components = _components(rows=2)
    components[
        FUSION_COMPONENT_STATIC_STATE
    ] = bad_tensor

    with pytest.raises(ValueError, match="shape"):
        fusion(components)


def test_forward_rejects_component_width_mismatch() -> None:
    fusion = _fusion()
    components = _components(rows=2)
    components[
        FUSION_COMPONENT_STATIC_STATE
    ] = torch.zeros(
        2,
        STATIC_DIM + 1,
    )

    with pytest.raises(ValueError, match="width"):
        fusion(components)


def test_forward_rejects_nonfloating_component() -> None:
    fusion = _fusion()
    components = _components(rows=2)
    components[
        FUSION_COMPONENT_STATIC_STATE
    ] = torch.zeros(
        2,
        STATIC_DIM,
        dtype=torch.long,
    )

    with pytest.raises(
        ValueError,
        match="floating-point",
    ):
        fusion(components)


def test_forward_rejects_row_count_mismatch() -> None:
    fusion = _fusion()
    components = _components(rows=2)
    components[
        FUSION_COMPONENT_MEMORY_STATE
    ] = torch.zeros(
        3,
        MEMORY_DIM,
    )

    with pytest.raises(
        ValueError,
        match="same item count",
    ):
        fusion(components)


@pytest.mark.parametrize(
    "bad_value",
    (
        float("nan"),
        float("inf"),
        float("-inf"),
    ),
)
def test_forward_rejects_nonfinite_component(
    bad_value: float,
) -> None:
    fusion = _fusion()
    components = _components(rows=2)
    components[
        FUSION_COMPONENT_STATIC_STATE
    ][0, 0] = bad_value

    with pytest.raises(ValueError, match="finite"):
        fusion(components)


def test_forward_rejects_nonfinite_fusion_output() -> None:
    fusion = _fusion(
        layer_norm=False,
    )
    components = _components(rows=2)

    with torch.no_grad():
        fusion.fusion_network.linear_out.weight[
            0,
            0,
        ] = float("nan")

    with pytest.raises(ValueError, match="finite"):
        fusion(components)


# =============================================================================
# Output schema
# =============================================================================


def test_output_preserves_projected_mapping_immutably() -> None:
    output = _output_contract()

    assert isinstance(
        output.projected_components,
        MappingProxyType,
    )

    with pytest.raises(TypeError):
        output.projected_components[
            "new"
        ] = torch.zeros(3, OUTPUT_DIM)  # type: ignore[index]


def test_output_copies_mapping_structure() -> None:
    projected = OrderedDict(
        (
            (
                FUSION_COMPONENT_STATIC_STATE,
                torch.ones(2, OUTPUT_DIM),
            ),
        )
    )
    output = ConcatProjectionFusionOutput(
        fused_state=torch.zeros(
            2,
            OUTPUT_DIM,
        ),
        projected_components=projected,
        component_order=(
            FUSION_COMPONENT_STATIC_STATE,
        ),
        architecture_fingerprint="architecture",
    )

    projected[
        FUSION_COMPONENT_MEMORY_STATE
    ] = torch.ones(2, OUTPUT_DIM)

    assert tuple(
        output.projected_components
    ) == (
        FUSION_COMPONENT_STATIC_STATE,
    )


def test_output_rejects_non_tensor_fused_state() -> None:
    with pytest.raises(TypeError, match="tensor"):
        ConcatProjectionFusionOutput(
            fused_state=[[0.0]],  # type: ignore[arg-type]
            projected_components={
                FUSION_COMPONENT_STATIC_STATE: (
                    torch.zeros(1, OUTPUT_DIM)
                )
            },
            component_order=(
                FUSION_COMPONENT_STATIC_STATE,
            ),
            architecture_fingerprint="architecture",
        )


@pytest.mark.parametrize(
    "fused",
    (
        torch.zeros(OUTPUT_DIM),
        torch.zeros(2, OUTPUT_DIM, 1),
    ),
)
def test_output_rejects_invalid_fused_rank(
    fused: torch.Tensor,
) -> None:
    with pytest.raises(ValueError, match="shape"):
        ConcatProjectionFusionOutput(
            fused_state=fused,
            projected_components={
                FUSION_COMPONENT_STATIC_STATE: (
                    torch.zeros(2, OUTPUT_DIM)
                )
            },
            component_order=(
                FUSION_COMPONENT_STATIC_STATE,
            ),
            architecture_fingerprint="architecture",
        )


def test_output_rejects_nonfloating_fused_state() -> None:
    with pytest.raises(
        ValueError,
        match="floating-point",
    ):
        ConcatProjectionFusionOutput(
            fused_state=torch.zeros(
                2,
                OUTPUT_DIM,
                dtype=torch.long,
            ),
            projected_components={
                FUSION_COMPONENT_STATIC_STATE: (
                    torch.zeros(2, OUTPUT_DIM)
                )
            },
            component_order=(
                FUSION_COMPONENT_STATIC_STATE,
            ),
            architecture_fingerprint="architecture",
        )


def test_output_rejects_nonfinite_fused_state() -> None:
    fused = torch.zeros(
        2,
        OUTPUT_DIM,
    )
    fused[0, 0] = float("nan")

    with pytest.raises(ValueError, match="finite"):
        ConcatProjectionFusionOutput(
            fused_state=fused,
            projected_components={
                FUSION_COMPONENT_STATIC_STATE: (
                    torch.zeros(2, OUTPUT_DIM)
                )
            },
            component_order=(
                FUSION_COMPONENT_STATIC_STATE,
            ),
            architecture_fingerprint="architecture",
        )


def test_output_rejects_empty_component_order() -> None:
    with pytest.raises(
        ValueError,
        match="cannot be empty",
    ):
        ConcatProjectionFusionOutput(
            fused_state=torch.zeros(
                2,
                OUTPUT_DIM,
            ),
            projected_components={},
            component_order=(),
            architecture_fingerprint="architecture",
        )


def test_output_rejects_duplicate_component_order() -> None:
    with pytest.raises(
        ValueError,
        match="duplicates",
    ):
        ConcatProjectionFusionOutput(
            fused_state=torch.zeros(
                2,
                OUTPUT_DIM,
            ),
            projected_components={
                FUSION_COMPONENT_STATIC_STATE: (
                    torch.zeros(2, OUTPUT_DIM)
                )
            },
            component_order=(
                FUSION_COMPONENT_STATIC_STATE,
                FUSION_COMPONENT_STATIC_STATE,
            ),
            architecture_fingerprint="architecture",
        )


def test_output_rejects_mapping_order_mismatch() -> None:
    projected = OrderedDict(
        (
            (
                FUSION_COMPONENT_MEMORY_STATE,
                torch.zeros(2, OUTPUT_DIM),
            ),
            (
                FUSION_COMPONENT_STATIC_STATE,
                torch.zeros(2, OUTPUT_DIM),
            ),
        )
    )

    with pytest.raises(
        ValueError,
        match="order differs",
    ):
        ConcatProjectionFusionOutput(
            fused_state=torch.zeros(
                2,
                OUTPUT_DIM,
            ),
            projected_components=projected,
            component_order=(
                FUSION_COMPONENT_STATIC_STATE,
                FUSION_COMPONENT_MEMORY_STATE,
            ),
            architecture_fingerprint="architecture",
        )


def test_output_rejects_projected_shape_mismatch() -> None:
    with pytest.raises(
        ValueError,
        match="shape differs",
    ):
        ConcatProjectionFusionOutput(
            fused_state=torch.zeros(
                2,
                OUTPUT_DIM,
            ),
            projected_components={
                FUSION_COMPONENT_STATIC_STATE: (
                    torch.zeros(
                        2,
                        OUTPUT_DIM + 1,
                    )
                )
            },
            component_order=(
                FUSION_COMPONENT_STATIC_STATE,
            ),
            architecture_fingerprint="architecture",
        )


def test_output_rejects_invalid_concatenated_shape() -> None:
    with pytest.raises(
        ValueError,
        match="concatenation contract",
    ):
        ConcatProjectionFusionOutput(
            fused_state=torch.zeros(
                2,
                OUTPUT_DIM,
            ),
            projected_components={
                FUSION_COMPONENT_STATIC_STATE: (
                    torch.zeros(2, OUTPUT_DIM)
                )
            },
            component_order=(
                FUSION_COMPONENT_STATIC_STATE,
            ),
            architecture_fingerprint="architecture",
            concatenated_state=torch.zeros(
                2,
                OUTPUT_DIM + 1,
            ),
        )


def test_output_rejects_blank_identity_fields() -> None:
    with pytest.raises(
        ValueError,
        match="non-empty",
    ):
        ConcatProjectionFusionOutput(
            fused_state=torch.zeros(
                2,
                OUTPUT_DIM,
            ),
            projected_components={
                FUSION_COMPONENT_STATIC_STATE: (
                    torch.zeros(2, OUTPUT_DIM)
                )
            },
            component_order=(
                FUSION_COMPONENT_STATIC_STATE,
            ),
            architecture_fingerprint="",
        )


# =============================================================================
# Backward contract
# =============================================================================


def test_backward_produces_finite_component_gradients() -> None:
    fusion = _fusion()
    components = _components(
        rows=4,
        requires_grad=True,
    )

    output = fusion(components)
    output.fused_state.square().mean().backward()

    for values in components.values():
        assert values.grad is not None
        assert values.grad.shape == values.shape
        assert bool(
            torch.isfinite(
                values.grad
            ).all().item()
        )


def test_backward_produces_finite_parameter_gradients() -> None:
    fusion = _fusion()
    components = _components(
        rows=4,
        requires_grad=True,
    )

    fusion(
        components
    ).fused_state.sum().backward()

    for parameter in fusion.parameters():
        assert parameter.grad is not None
        assert bool(
            torch.isfinite(
                parameter.grad
            ).all().item()
        )


def test_retained_concatenated_state_remains_in_graph() -> None:
    fusion = _fusion(
        retain_concatenated_state=True,
    )
    components = _components(
        rows=3,
        requires_grad=True,
    )
    output = fusion(components)

    assert output.concatenated_state is not None
    loss = (
        output.fused_state.mean()
        + output.concatenated_state.mean()
    )
    loss.backward()

    for values in components.values():
        assert values.grad is not None


# =============================================================================
# Architecture and parameter identity
# =============================================================================


def test_architecture_dict_is_complete() -> None:
    fusion = _fusion(
        dropout=0.2,
        layer_norm=True,
        retain_concatenated_state=True,
        record_input_fingerprint=True,
    )
    architecture = fusion.architecture_dict()

    assert architecture[
        "schema_version"
    ] == CONCAT_PROJECTION_FUSION_SCHEMA_VERSION
    assert architecture["algorithm"] == "concat_projection"
    assert architecture["component_order"] == list(
        fusion.component_order
    )
    assert architecture["component_count"] == (
        fusion.component_count
    )
    assert architecture["output_dim"] == OUTPUT_DIM
    assert architecture["fusion_input_dim"] == (
        fusion.fusion_input_dim
    )
    assert architecture["dropout"] == 0.2
    assert architecture["layer_norm"] is True
    assert architecture[
        "retain_concatenated_state"
    ] is True
    assert architecture[
        "record_input_fingerprint"
    ] is True
    assert tuple(
        architecture["component_projectors"]
    ) == fusion.component_order
    assert architecture["fusion_network"][
        "operation_order"
    ] == [
        "linear_in",
        "gelu",
        "dropout",
        "linear_out",
        "layer_norm",
    ]


def test_architecture_fingerprint_is_stable() -> None:
    first = _fusion()
    second = _fusion()

    assert first.architecture_dict() == (
        second.architecture_dict()
    )
    assert first.architecture_fingerprint() == (
        second.architecture_fingerprint()
    )


@pytest.mark.parametrize(
    "builder",
    (
        lambda: _fusion(
            output_dim=OUTPUT_DIM + 1,
        ),
        lambda: _fusion(
            dropout=0.25,
        ),
        lambda: _fusion(
            layer_norm=False,
        ),
        lambda: _fusion(
            retain_concatenated_state=True,
        ),
        lambda: _fusion(
            record_input_fingerprint=True,
        ),
        lambda: _fusion(
            component_input_dims=_canonical_dims(
                include_hazard_context=False,
            ),
        ),
    ),
)
def test_architecture_fingerprint_changes_with_contract(
    builder: Any,
) -> None:
    baseline = _fusion()
    changed = builder()

    assert baseline.architecture_fingerprint() != (
        changed.architecture_fingerprint()
    )


def test_parameter_fingerprint_is_reproducible_under_seed() -> None:
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(123)
        first = _fusion()

    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(123)
        second = _fusion()

    assert first.parameter_fingerprint() == (
        second.parameter_fingerprint()
    )


def test_parameter_fingerprint_changes_after_mutation() -> None:
    fusion = _fusion()
    before = fusion.parameter_fingerprint()
    architecture = (
        fusion.architecture_fingerprint()
    )

    with torch.no_grad():
        fusion.fusion_network.linear_in.weight[
            0,
            0,
        ] += 1.0

    assert fusion.parameter_fingerprint() != before
    assert fusion.architecture_fingerprint() == (
        architecture
    )


def test_component_architecture_fingerprints_are_immutable() -> None:
    fusion = _fusion()
    fingerprints = (
        fusion.component_architecture_fingerprints()
    )

    assert isinstance(
        fingerprints,
        MappingProxyType,
    )
    assert tuple(fingerprints) == (
        fusion.component_order
    )

    with pytest.raises(TypeError):
        fingerprints[
            FUSION_COMPONENT_STATIC_STATE
        ] = "changed"  # type: ignore[index]


def test_component_parameter_fingerprints_change_locally() -> None:
    fusion = _fusion()
    before = dict(
        fusion.component_parameter_fingerprints()
    )

    with torch.no_grad():
        (
            fusion
            .component_projections[
                FUSION_COMPONENT_STATIC_STATE
            ]
            .linear
            .weight[0, 0]
        ) += 1.0

    after = dict(
        fusion.component_parameter_fingerprints()
    )

    assert before[
        FUSION_COMPONENT_STATIC_STATE
    ] != after[
        FUSION_COMPONENT_STATIC_STATE
    ]

    for name in fusion.component_order:
        if name != FUSION_COMPONENT_STATIC_STATE:
            assert before[name] == after[name]


def test_state_dict_round_trip_preserves_outputs() -> None:
    source = _fusion(
        dropout=0.0,
    )
    target = _fusion(
        dropout=0.0,
    )
    target.load_state_dict(
        source.state_dict(),
        strict=True,
    )

    source.eval()
    target.eval()
    components = _components(rows=4)

    source_output = source(components)
    target_output = target(components)

    assert torch.equal(
        source_output.fused_state,
        target_output.fused_state,
    )

    for name in source.component_order:
        assert torch.equal(
            source_output.projected_components[name],
            target_output.projected_components[name],
        )


def test_state_dict_keys_use_semantic_names() -> None:
    fusion = _fusion()
    keys = tuple(
        fusion.state_dict()
    )

    assert any(
        key.startswith(
            "component_projections.static_state.linear."
        )
        for key in keys
    )
    assert any(
        key.startswith(
            "component_projections.memory_state.linear."
        )
        for key in keys
    )
    assert "fusion_network.linear_in.weight" in keys
    assert "fusion_network.linear_out.weight" in keys


def test_finite_parameter_check_passes() -> None:
    fusion = _fusion()
    fusion.assert_finite_parameters()


@pytest.mark.parametrize(
    "bad_value",
    (
        float("nan"),
        float("inf"),
        float("-inf"),
    ),
)
def test_finite_parameter_check_detects_corruption(
    bad_value: float,
) -> None:
    fusion = _fusion()

    with torch.no_grad():
        fusion.fusion_network.linear_in.weight[
            0,
            0,
        ] = bad_value

    with pytest.raises(
        ValueError,
        match="NaN|infinity",
    ):
        fusion.assert_finite_parameters()


def test_extra_repr_contains_architecture_identity() -> None:
    fusion = _fusion(
        dropout=0.2,
        retain_concatenated_state=True,
        record_input_fingerprint=True,
    )
    representation = fusion.extra_repr()

    assert "component_order" in representation
    assert f"output_dim={OUTPUT_DIM}" in representation
    assert "dropout=0.2" in representation
    assert "retain_concatenated_state=True" in representation
    assert "record_input_fingerprint=True" in representation


# =============================================================================
# Optional device contract
# =============================================================================


@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA is unavailable.",
)
def test_forward_rejects_module_input_device_mismatch() -> None:
    fusion = _fusion().cuda()
    components = _components(rows=2)

    with pytest.raises(
        ValueError,
        match="share one device",
    ):
        fusion(components)


@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA is unavailable.",
)
def test_cuda_forward_and_backward_are_finite() -> None:
    fusion = _fusion().cuda()
    components = OrderedDict(
        (
            name,
            values.cuda(),
        )
        for name, values in _components(
            rows=3,
            requires_grad=True,
        ).items()
    )

    for values in components.values():
        values.retain_grad()

    output = fusion(components)
    output.fused_state.square().mean().backward()

    assert output.fused_state.device.type == "cuda"
    assert bool(
        torch.isfinite(
            output.fused_state
        ).all().item()
    )

    for values in components.values():
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
def test_output_rejects_projected_device_mismatch() -> None:
    with pytest.raises(
        ValueError,
        match="share one device",
    ):
        ConcatProjectionFusionOutput(
            fused_state=torch.zeros(
                2,
                OUTPUT_DIM,
            ),
            projected_components={
                FUSION_COMPONENT_STATIC_STATE: (
                    torch.zeros(
                        2,
                        OUTPUT_DIM,
                        device="cuda",
                    )
                )
            },
            component_order=(
                FUSION_COMPONENT_STATIC_STATE,
            ),
            architecture_fingerprint="architecture",
        )
