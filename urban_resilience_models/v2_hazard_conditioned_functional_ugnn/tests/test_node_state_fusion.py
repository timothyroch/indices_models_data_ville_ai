"""
Contract tests for the split node-state fusion orchestrator.

Target repository path:
    urban_resilience_models/
        v2_hazard_conditioned_functional_ugnn/
            tests/
                test_node_state_fusion.py

Implementation under test:
    urban_resilience_models/
        v2_hazard_conditioned_functional_ugnn/
            fusion/
                node_state_fusion.py

The schema layer, component projector, and concat-projection algorithm have
their own focused suites. This file freezes the orchestration boundary:

- canonical versus implemented fusion modes;
- construction from ``ModelConfig``;
- enabled and disabled component policies;
- typed tensor extraction in canonical order;
- node-type embedding lookup and range checks;
- dispatch to ``ConcatProjectionFusion``;
- preservation of complete source inputs in the final output;
- architecture, parameter, component, and lineage identities;
- finite forward and backward behavior;
- migration of checkpoints from the original monolithic implementation.

The small low-level factories for ``LagMemoryEncoding`` and
``HazardQueryEncoding`` intentionally bypass their constructors. Those
upstream constructors are covered by their own suites; this suite needs only
valid metadata-bearing objects with the exact fields consumed by the fusion
orchestrator.
"""

from __future__ import annotations

from collections import OrderedDict
from types import MappingProxyType
from typing import Any, Mapping

import pytest
import torch
from torch import nn

from urban_resilience_models.v2_hazard_conditioned_functional_ugnn.config import (
    CANONICAL_NODE_FUSION_MODES,
    HAZARD_CONDITIONING_EMBEDDING,
    HAZARD_CONDITIONING_QUERIED_MEMORY,
    MEMORY_ENCODER_LAG,
    MEMORY_QUERY_HAZARD_ATTENTION,
    MEMORY_QUERY_NONE,
    NODE_FUSION_CONCAT_PROJECTION,
    NODE_FUSION_FILM,
    NODE_FUSION_GATED,
    NODE_FUSION_PROJECTED_SUM,
    HazardConfig,
    MemoryConfig,
    ModelConfig,
    NodeStateFusionConfig,
)
from urban_resilience_models.v2_hazard_conditioned_functional_ugnn.fusion.component_projection import (
    ComponentProjection,
)
from urban_resilience_models.v2_hazard_conditioned_functional_ugnn.fusion.concat_projection import (
    FUSION_COMPONENT_HAZARD_CONTEXT,
    FUSION_COMPONENT_HAZARD_MEMORY_STATE,
    FUSION_COMPONENT_MEMORY_STATE,
    FUSION_COMPONENT_NODE_TYPE_EMBEDDING,
    FUSION_COMPONENT_STATIC_STATE,
    ConcatProjectionFusion,
)
from urban_resilience_models.v2_hazard_conditioned_functional_ugnn.fusion.node_state_fusion import (
    CANONICAL_NODE_STATE_FUSION_MODES,
    IMPLEMENTED_NODE_STATE_FUSION_MODES,
    NODE_STATE_FUSION_ENCODER_SCHEMA_VERSION,
    NODE_STATE_FUSION_LEGACY_STATE_DICT_SCHEMA_VERSION,
    NodeStateFusion,
    NodeStateFusionMode,
)
from urban_resilience_models.v2_hazard_conditioned_functional_ugnn.fusion.schemas import (
    NodeAlignment,
    NodeStateComponent,
    NodeStateFusionInputs,
    NodeStateFusionOutput,
)
from urban_resilience_models.v2_hazard_conditioned_functional_ugnn.hazard.hazard_query_encoder import (
    HazardQueryEncoding,
)
from urban_resilience_models.v2_hazard_conditioned_functional_ugnn.memory.lag_memory_encoder import (
    LagMemoryEncoding,
)


STATIC_DIM = 5
MEMORY_DIM = 7
HAZARD_MEMORY_DIM = 6
HAZARD_CONTEXT_DIM = 8
OUTPUT_DIM = 11
NODE_TYPE_COUNT = 3
NODE_TYPE_EMBEDDING_DIM = 4


# =============================================================================
# Helpers
# =============================================================================


def _alignment(
    item_count: int,
) -> NodeAlignment:
    return NodeAlignment(
        item_count=item_count,
        item_ids=tuple(
            f"item-{index}"
            for index in range(item_count)
        ),
        source_fingerprint="alignment-source",
    )


def _component(
    item_count: int,
    feature_dim: int,
    *,
    name: str,
    offset: float = 0.0,
    requires_grad: bool = False,
    alignment: NodeAlignment | None = None,
) -> NodeStateComponent:
    values = (
        torch.arange(
            item_count * feature_dim,
            dtype=torch.float32,
        )
        .reshape(item_count, feature_dim)
        / 10.0
        + offset
    )
    values.requires_grad_(requires_grad)

    return NodeStateComponent(
        values=values,
        component_name=name,
        source_fingerprint=f"{name}-source",
        alignment_fingerprint=(
            alignment.fingerprint()
            if alignment is not None
            else None
        ),
    )


def _memory_encoding(
    item_count: int,
    *,
    hidden_dim: int = MEMORY_DIM,
    offset: float = 0.0,
    requires_grad: bool = False,
) -> LagMemoryEncoding:
    """
    Build the minimal valid object consumed by the fusion schema/orchestrator.

    The full LagMemoryEncoding constructor is tested in
    test_lag_memory_encoder.py.
    """

    state = (
        torch.arange(
            item_count * hidden_dim,
            dtype=torch.float32,
        )
        .reshape(item_count, hidden_dim)
        / 10.0
        + offset
    )
    state.requires_grad_(requires_grad)

    encoding = object.__new__(
        LagMemoryEncoding
    )
    object.__setattr__(
        encoding,
        "memory_state",
        state,
    )
    object.__setattr__(
        encoding,
        "source_batch",
        None,
    )
    object.__setattr__(
        encoding,
        "lag_feature_states",
        None,
    )
    object.__setattr__(
        encoding,
        "lag_weights",
        None,
    )
    object.__setattr__(
        encoding,
        "encoder_architecture_fingerprint",
        "memory-architecture",
    )
    object.__setattr__(
        encoding,
        "lineage_fingerprint",
        f"memory-lineage-{offset}",
    )
    object.__setattr__(
        encoding,
        "schema_version",
        "test-memory-schema",
    )
    return encoding


def _hazard_query_encoding(
    item_count: int,
    *,
    query_dim: int = HAZARD_CONTEXT_DIM,
    offset: float = 0.0,
    requires_grad: bool = False,
) -> HazardQueryEncoding:
    """
    Build the minimal valid object consumed by fusion orchestration.

    The full HazardQueryEncoding constructor is tested in the hazard-query
    suite.
    """

    query = (
        torch.arange(
            item_count * query_dim,
            dtype=torch.float32,
        )
        .reshape(item_count, query_dim)
        / 10.0
        + offset
    )
    query.requires_grad_(requires_grad)

    encoding = object.__new__(
        HazardQueryEncoding
    )
    object.__setattr__(
        encoding,
        "query",
        query,
    )
    object.__setattr__(
        encoding,
        "source_embedding",
        object(),
    )
    object.__setattr__(
        encoding,
        "projected_hazard_embedding",
        query,
    )
    object.__setattr__(
        encoding,
        "hazard_feature_state",
        None,
    )
    object.__setattr__(
        encoding,
        "scenario_state",
        None,
    )
    object.__setattr__(
        encoding,
        "month_state",
        None,
    )
    object.__setattr__(
        encoding,
        "forecast_horizon_state",
        None,
    )
    object.__setattr__(
        encoding,
        "weather_state",
        None,
    )
    object.__setattr__(
        encoding,
        "event_state",
        None,
    )
    object.__setattr__(
        encoding,
        "query_encoder_architecture_fingerprint",
        "hazard-query-architecture",
    )
    object.__setattr__(
        encoding,
        "lineage_fingerprint",
        f"hazard-query-lineage-{offset}",
    )
    object.__setattr__(
        encoding,
        "schema_version",
        "test-hazard-query-schema",
    )
    return encoding


def _static_only_fusion(
    *,
    output_dim: int = OUTPUT_DIM,
    dropout: float = 0.0,
    layer_norm: bool = True,
) -> NodeStateFusion:
    return NodeStateFusion(
        mode=NODE_FUSION_CONCAT_PROJECTION,
        output_dim=output_dim,
        include_static_state=True,
        static_input_dim=STATIC_DIM,
        include_memory_state=False,
        memory_input_dim=None,
        include_hazard_memory_state=False,
        hazard_memory_input_dim=None,
        include_hazard_context=False,
        hazard_context_input_dim=None,
        include_node_type_embedding=False,
        node_type_count=None,
        node_type_embedding_dim=NODE_TYPE_EMBEDDING_DIM,
        dropout=dropout,
        layer_norm=layer_norm,
    )


def _all_component_fusion() -> NodeStateFusion:
    return NodeStateFusion(
        mode=NodeStateFusionMode.CONCAT_PROJECTION,
        output_dim=OUTPUT_DIM,
        include_static_state=True,
        static_input_dim=STATIC_DIM,
        include_memory_state=True,
        memory_input_dim=MEMORY_DIM,
        include_hazard_memory_state=True,
        hazard_memory_input_dim=HAZARD_MEMORY_DIM,
        include_hazard_context=True,
        hazard_context_input_dim=HAZARD_CONTEXT_DIM,
        include_node_type_embedding=True,
        node_type_count=NODE_TYPE_COUNT,
        node_type_embedding_dim=NODE_TYPE_EMBEDDING_DIM,
        dropout=0.0,
        layer_norm=True,
    )


def _static_inputs(
    item_count: int = 4,
    *,
    requires_grad: bool = False,
) -> NodeStateFusionInputs:
    alignment = _alignment(item_count)
    return NodeStateFusionInputs(
        alignment=alignment,
        static_state=_component(
            item_count,
            STATIC_DIM,
            name=FUSION_COMPONENT_STATIC_STATE,
            requires_grad=requires_grad,
            alignment=alignment,
        ),
        source_fingerprint="fusion-input-source",
    )


def _all_inputs(
    item_count: int = 6,
    *,
    requires_grad: bool = False,
) -> NodeStateFusionInputs:
    alignment = _alignment(item_count)

    return NodeStateFusionInputs(
        alignment=alignment,
        static_state=_component(
            item_count,
            STATIC_DIM,
            name=FUSION_COMPONENT_STATIC_STATE,
            requires_grad=requires_grad,
            alignment=alignment,
        ),
        memory_state=_memory_encoding(
            item_count,
            requires_grad=requires_grad,
        ),
        hazard_memory_state=_component(
            item_count,
            HAZARD_MEMORY_DIM,
            name=FUSION_COMPONENT_HAZARD_MEMORY_STATE,
            requires_grad=requires_grad,
            alignment=alignment,
        ),
        hazard_context=_hazard_query_encoding(
            item_count,
            requires_grad=requires_grad,
        ),
        node_type_ids=torch.tensor(
            [
                index % NODE_TYPE_COUNT
                for index in range(item_count)
            ],
            dtype=torch.long,
        ),
        source_fingerprint="all-components-source",
    )


def _legacy_state_dict(
    fusion: NodeStateFusion,
) -> OrderedDict[str, torch.Tensor]:
    """
    Convert the current split state dict to the original monolithic key layout.
    """

    legacy: OrderedDict[
        str,
        torch.Tensor,
    ] = OrderedDict()

    for key, value in fusion.state_dict().items():
        if key.startswith(
            "node_type_embedding."
        ):
            legacy[key] = value.clone()
            continue

        prefix = (
            "fusion_algorithm."
            "component_projections."
        )
        if key.startswith(prefix):
            remainder = key[len(prefix):]
            component_name, suffix = remainder.split(
                ".",
                maxsplit=1,
            )

            if suffix.startswith("linear."):
                legacy[
                    "component_projections."
                    f"{component_name}.network.0."
                    f"{suffix[len('linear.'):]}"
                ] = value.clone()
                continue

            if suffix.startswith(
                "normalization_layer."
            ):
                legacy[
                    "component_projections."
                    f"{component_name}.network.2."
                    f"{suffix[len('normalization_layer.'):]}"
                ] = value.clone()
                continue

        network_prefix = (
            "fusion_algorithm.fusion_network."
        )
        if key.startswith(network_prefix):
            suffix = key[len(network_prefix):]

            mapping = (
                ("linear_in.", "fusion_network.0."),
                ("linear_out.", "fusion_network.3."),
                ("normalization.", "fusion_network.4."),
            )

            for new_prefix, old_prefix in mapping:
                if suffix.startswith(new_prefix):
                    legacy[
                        old_prefix
                        + suffix[len(new_prefix):]
                    ] = value.clone()
                    break
            else:
                raise AssertionError(
                    f"Unhandled fusion-network key {key!r}."
                )
            continue

        raise AssertionError(
            f"Unhandled split state-dict key {key!r}."
        )

    return legacy


# =============================================================================
# Published identity and mode capability
# =============================================================================


def test_schema_versions_are_nonempty() -> None:
    for value in (
        NODE_STATE_FUSION_ENCODER_SCHEMA_VERSION,
        NODE_STATE_FUSION_LEGACY_STATE_DICT_SCHEMA_VERSION,
    ):
        assert isinstance(value, str)
        assert value.strip()


def test_local_modes_match_config_vocabulary() -> None:
    assert tuple(
        mode.value
        for mode in CANONICAL_NODE_STATE_FUSION_MODES
    ) == tuple(CANONICAL_NODE_FUSION_MODES)

    assert tuple(
        mode.value
        for mode in NodeStateFusionMode
    ) == (
        NODE_FUSION_CONCAT_PROJECTION,
        NODE_FUSION_PROJECTED_SUM,
        NODE_FUSION_GATED,
        NODE_FUSION_FILM,
    )


def test_only_concat_projection_is_implemented() -> None:
    assert IMPLEMENTED_NODE_STATE_FUSION_MODES == (
        NodeStateFusionMode.CONCAT_PROJECTION,
    )


@pytest.mark.parametrize(
    "mode",
    (
        NodeStateFusionMode.PROJECTED_SUM,
        NodeStateFusionMode.GATED_FUSION,
        NodeStateFusionMode.FILM_CONDITIONING,
        NODE_FUSION_PROJECTED_SUM,
        NODE_FUSION_GATED,
        NODE_FUSION_FILM,
    ),
)
def test_constructor_distinguishes_known_unimplemented_modes(
    mode: NodeStateFusionMode | str,
) -> None:
    with pytest.raises(
        NotImplementedError,
        match="canonical but not implemented",
    ):
        NodeStateFusion(
            mode=mode,
            output_dim=OUTPUT_DIM,
            include_static_state=True,
            static_input_dim=STATIC_DIM,
            include_memory_state=False,
            memory_input_dim=None,
            include_hazard_memory_state=False,
            hazard_memory_input_dim=None,
            include_hazard_context=False,
            hazard_context_input_dim=None,
            include_node_type_embedding=False,
            node_type_count=None,
        )


def test_constructor_rejects_unknown_mode() -> None:
    with pytest.raises(
        ValueError,
        match="Unknown node-state fusion mode",
    ):
        NodeStateFusion(
            mode="not_a_real_mode",
            output_dim=OUTPUT_DIM,
            include_static_state=True,
            static_input_dim=STATIC_DIM,
            include_memory_state=False,
            memory_input_dim=None,
            include_hazard_memory_state=False,
            hazard_memory_input_dim=None,
            include_hazard_context=False,
            hazard_context_input_dim=None,
            include_node_type_embedding=False,
            node_type_count=None,
        )


# =============================================================================
# Constructor contract
# =============================================================================


def test_static_only_constructor_contract() -> None:
    fusion = _static_only_fusion(
        dropout=0.2,
        layer_norm=True,
    )

    assert fusion.mode is (
        NodeStateFusionMode.CONCAT_PROJECTION
    )
    assert fusion.output_dim == OUTPUT_DIM
    assert fusion.include_static_state
    assert not fusion.include_memory_state
    assert fusion.static_input_dim == STATIC_DIM
    assert fusion.memory_input_dim is None
    assert fusion.component_order == (
        FUSION_COMPONENT_STATIC_STATE,
    )
    assert fusion.component_count == 1
    assert fusion.dropout == 0.2
    assert fusion.layer_norm is True


def test_constructor_builds_split_algorithm() -> None:
    fusion = _all_component_fusion()

    assert isinstance(
        fusion.fusion_algorithm,
        ConcatProjectionFusion,
    )
    assert fusion.component_projections is (
        fusion.fusion_algorithm
        .component_projections
    )
    assert fusion.fusion_network is (
        fusion.fusion_algorithm
        .fusion_network
    )


def test_all_component_order_is_canonical() -> None:
    fusion = _all_component_fusion()

    assert fusion.component_order == (
        FUSION_COMPONENT_STATIC_STATE,
        FUSION_COMPONENT_MEMORY_STATE,
        FUSION_COMPONENT_HAZARD_MEMORY_STATE,
        FUSION_COMPONENT_HAZARD_CONTEXT,
        FUSION_COMPONENT_NODE_TYPE_EMBEDDING,
    )

    assert tuple(
        fusion.component_input_dims
    ) == fusion.component_order
    assert dict(
        fusion.component_input_dims
    ) == {
        FUSION_COMPONENT_STATIC_STATE: STATIC_DIM,
        FUSION_COMPONENT_MEMORY_STATE: MEMORY_DIM,
        FUSION_COMPONENT_HAZARD_MEMORY_STATE: (
            HAZARD_MEMORY_DIM
        ),
        FUSION_COMPONENT_HAZARD_CONTEXT: (
            HAZARD_CONTEXT_DIM
        ),
        FUSION_COMPONENT_NODE_TYPE_EMBEDDING: (
            NODE_TYPE_EMBEDDING_DIM
        ),
    }


def test_node_type_embedding_is_constructed_only_when_enabled() -> None:
    static = _static_only_fusion()
    full = _all_component_fusion()

    assert static.node_type_embedding is None
    assert isinstance(
        full.node_type_embedding,
        nn.Embedding,
    )
    assert full.node_type_embedding.num_embeddings == (
        NODE_TYPE_COUNT
    )
    assert full.node_type_embedding.embedding_dim == (
        NODE_TYPE_EMBEDDING_DIM
    )


def test_constructor_rejects_no_components() -> None:
    with pytest.raises(
        ValueError,
        match="at least one component",
    ):
        NodeStateFusion(
            mode=NODE_FUSION_CONCAT_PROJECTION,
            output_dim=OUTPUT_DIM,
            include_static_state=False,
            static_input_dim=None,
            include_memory_state=False,
            memory_input_dim=None,
            include_hazard_memory_state=False,
            hazard_memory_input_dim=None,
            include_hazard_context=False,
            hazard_context_input_dim=None,
            include_node_type_embedding=False,
            node_type_count=None,
        )


@pytest.mark.parametrize(
    "field",
    (
        "include_static_state",
        "include_memory_state",
        "include_hazard_memory_state",
        "include_hazard_context",
        "include_node_type_embedding",
        "layer_norm",
    ),
)
def test_constructor_rejects_non_boolean_flags(
    field: str,
) -> None:
    kwargs: dict[str, Any] = {
        "mode": NODE_FUSION_CONCAT_PROJECTION,
        "output_dim": OUTPUT_DIM,
        "include_static_state": True,
        "static_input_dim": STATIC_DIM,
        "include_memory_state": False,
        "memory_input_dim": None,
        "include_hazard_memory_state": False,
        "hazard_memory_input_dim": None,
        "include_hazard_context": False,
        "hazard_context_input_dim": None,
        "include_node_type_embedding": False,
        "node_type_count": None,
        field: 1,
    }

    with pytest.raises(TypeError, match="Boolean"):
        NodeStateFusion(**kwargs)


@pytest.mark.parametrize(
    ("include_field", "dimension_field"),
    (
        (
            "include_static_state",
            "static_input_dim",
        ),
        (
            "include_memory_state",
            "memory_input_dim",
        ),
        (
            "include_hazard_memory_state",
            "hazard_memory_input_dim",
        ),
        (
            "include_hazard_context",
            "hazard_context_input_dim",
        ),
    ),
)
def test_enabled_component_requires_dimension(
    include_field: str,
    dimension_field: str,
) -> None:
    kwargs: dict[str, Any] = {
        "mode": NODE_FUSION_CONCAT_PROJECTION,
        "output_dim": OUTPUT_DIM,
        "include_static_state": False,
        "static_input_dim": None,
        "include_memory_state": False,
        "memory_input_dim": None,
        "include_hazard_memory_state": False,
        "hazard_memory_input_dim": None,
        "include_hazard_context": False,
        "hazard_context_input_dim": None,
        "include_node_type_embedding": True,
        "node_type_count": NODE_TYPE_COUNT,
        "node_type_embedding_dim": (
            NODE_TYPE_EMBEDDING_DIM
        ),
    }
    kwargs[include_field] = True
    kwargs[dimension_field] = None

    with pytest.raises(
        ValueError,
        match=dimension_field,
    ):
        NodeStateFusion(**kwargs)


@pytest.mark.parametrize(
    ("include_field", "dimension_field", "dimension"),
    (
        (
            "include_static_state",
            "static_input_dim",
            STATIC_DIM,
        ),
        (
            "include_memory_state",
            "memory_input_dim",
            MEMORY_DIM,
        ),
        (
            "include_hazard_memory_state",
            "hazard_memory_input_dim",
            HAZARD_MEMORY_DIM,
        ),
        (
            "include_hazard_context",
            "hazard_context_input_dim",
            HAZARD_CONTEXT_DIM,
        ),
    ),
)
def test_disabled_component_rejects_dimension(
    include_field: str,
    dimension_field: str,
    dimension: int,
) -> None:
    kwargs: dict[str, Any] = {
        "mode": NODE_FUSION_CONCAT_PROJECTION,
        "output_dim": OUTPUT_DIM,
        "include_static_state": True,
        "static_input_dim": STATIC_DIM,
        "include_memory_state": False,
        "memory_input_dim": None,
        "include_hazard_memory_state": False,
        "hazard_memory_input_dim": None,
        "include_hazard_context": False,
        "hazard_context_input_dim": None,
        "include_node_type_embedding": False,
        "node_type_count": None,
    }
    kwargs[include_field] = False
    kwargs[dimension_field] = dimension

    # Keep one valid component enabled when testing static itself.
    if include_field == "include_static_state":
        kwargs["include_node_type_embedding"] = True
        kwargs["node_type_count"] = NODE_TYPE_COUNT

    with pytest.raises(
        ValueError,
        match="must be None",
    ):
        NodeStateFusion(**kwargs)


def test_node_type_embedding_requires_count() -> None:
    with pytest.raises(
        ValueError,
        match="node_type_count",
    ):
        NodeStateFusion(
            mode=NODE_FUSION_CONCAT_PROJECTION,
            output_dim=OUTPUT_DIM,
            include_static_state=False,
            static_input_dim=None,
            include_memory_state=False,
            memory_input_dim=None,
            include_hazard_memory_state=False,
            hazard_memory_input_dim=None,
            include_hazard_context=False,
            hazard_context_input_dim=None,
            include_node_type_embedding=True,
            node_type_count=None,
        )


def test_disabled_node_type_embedding_rejects_count() -> None:
    with pytest.raises(
        ValueError,
        match="must be None",
    ):
        NodeStateFusion(
            mode=NODE_FUSION_CONCAT_PROJECTION,
            output_dim=OUTPUT_DIM,
            include_static_state=True,
            static_input_dim=STATIC_DIM,
            include_memory_state=False,
            memory_input_dim=None,
            include_hazard_memory_state=False,
            hazard_memory_input_dim=None,
            include_hazard_context=False,
            hazard_context_input_dim=None,
            include_node_type_embedding=False,
            node_type_count=NODE_TYPE_COUNT,
        )


@pytest.mark.parametrize(
    "value",
    (
        0,
        -1,
        True,
        1.5,
    ),
)
def test_constructor_rejects_invalid_node_type_embedding_dim(
    value: Any,
) -> None:
    with pytest.raises(
        ValueError,
        match="positive integer",
    ):
        NodeStateFusion(
            mode=NODE_FUSION_CONCAT_PROJECTION,
            output_dim=OUTPUT_DIM,
            include_static_state=True,
            static_input_dim=STATIC_DIM,
            include_memory_state=False,
            memory_input_dim=None,
            include_hazard_memory_state=False,
            hazard_memory_input_dim=None,
            include_hazard_context=False,
            hazard_context_input_dim=None,
            include_node_type_embedding=False,
            node_type_count=None,
            node_type_embedding_dim=value,
        )


# =============================================================================
# Construction from ModelConfig
# =============================================================================


def test_from_config_builds_static_only_fusion() -> None:
    model = ModelConfig(
        static_input_dim=STATIC_DIM,
        hidden_dim=OUTPUT_DIM,
        fusion=NodeStateFusionConfig(
            mode=NODE_FUSION_CONCAT_PROJECTION,
            output_dim=OUTPUT_DIM,
            include_static_state=True,
        ),
    )

    fusion = NodeStateFusion.from_config(
        model
    )

    assert fusion.component_order == (
        FUSION_COMPONENT_STATIC_STATE,
    )
    assert fusion.static_input_dim == STATIC_DIM
    assert fusion.output_dim == OUTPUT_DIM


def test_from_config_requires_model_config() -> None:
    with pytest.raises(
        TypeError,
        match="ModelConfig",
    ):
        NodeStateFusion.from_config(  # type: ignore[arg-type]
            object()
        )


def test_from_config_rejects_unresolved_static_dimension() -> None:
    model = ModelConfig(
        static_input_dim=None,
        hidden_dim=OUTPUT_DIM,
        fusion=NodeStateFusionConfig(
            output_dim=OUTPUT_DIM,
            include_static_state=True,
        ),
    )

    with pytest.raises(
        ValueError,
        match="static_input_dim",
    ):
        NodeStateFusion.from_config(model)


def test_from_config_resolves_memory_dimension() -> None:
    model = ModelConfig(
        static_input_dim=STATIC_DIM,
        hidden_dim=OUTPUT_DIM,
        memory=MemoryConfig(
            encoder_type=MEMORY_ENCODER_LAG,
            query_type=MEMORY_QUERY_NONE,
            hidden_dim=MEMORY_DIM,
            lag_feature_names=("lag_1",),
        ),
        fusion=NodeStateFusionConfig(
            output_dim=OUTPUT_DIM,
            include_static_state=True,
            include_memory_state=True,
        ),
    )

    fusion = NodeStateFusion.from_config(
        model
    )

    assert fusion.memory_input_dim == MEMORY_DIM
    assert fusion.component_order == (
        FUSION_COMPONENT_STATIC_STATE,
        FUSION_COMPONENT_MEMORY_STATE,
    )


def test_from_config_resolves_hazard_context_dimension() -> None:
    model = ModelConfig(
        static_input_dim=STATIC_DIM,
        hidden_dim=OUTPUT_DIM,
        hazard=HazardConfig(
            conditioning_mode=(
                HAZARD_CONDITIONING_EMBEDDING
            ),
            output_dim=HAZARD_CONTEXT_DIM,
        ),
        fusion=NodeStateFusionConfig(
            output_dim=OUTPUT_DIM,
            include_static_state=True,
            include_hazard_context=True,
        ),
    )

    fusion = NodeStateFusion.from_config(
        model
    )

    assert fusion.hazard_context_input_dim == (
        HAZARD_CONTEXT_DIM
    )
    assert fusion.component_order == (
        FUSION_COMPONENT_STATIC_STATE,
        FUSION_COMPONENT_HAZARD_CONTEXT,
    )


def test_from_config_resolves_hazard_memory_dimension() -> None:
    model = ModelConfig(
        static_input_dim=STATIC_DIM,
        hidden_dim=OUTPUT_DIM,
        memory=MemoryConfig(
            encoder_type=MEMORY_ENCODER_LAG,
            query_type=(
                MEMORY_QUERY_HAZARD_ATTENTION
            ),
            hidden_dim=HAZARD_MEMORY_DIM,
            return_temporal_states=True,
            lag_feature_names=("lag_1",),
        ),
        hazard=HazardConfig(
            conditioning_mode=(
                HAZARD_CONDITIONING_QUERIED_MEMORY
            ),
            output_dim=HAZARD_CONTEXT_DIM,
        ),
        fusion=NodeStateFusionConfig(
            output_dim=OUTPUT_DIM,
            include_static_state=True,
            include_hazard_memory_state=True,
        ),
    )

    fusion = NodeStateFusion.from_config(
        model
    )

    assert fusion.hazard_memory_input_dim == (
        HAZARD_MEMORY_DIM
    )
    assert fusion.component_order == (
        FUSION_COMPONENT_STATIC_STATE,
        FUSION_COMPONENT_HAZARD_MEMORY_STATE,
    )


def test_from_config_resolves_node_type_count() -> None:
    model = ModelConfig(
        static_input_dim=STATIC_DIM,
        hidden_dim=OUTPUT_DIM,
        node_type_count=NODE_TYPE_COUNT,
        fusion=NodeStateFusionConfig(
            output_dim=OUTPUT_DIM,
            include_static_state=True,
            include_node_type_embedding=True,
            node_type_embedding_dim=(
                NODE_TYPE_EMBEDDING_DIM
            ),
        ),
    )

    fusion = NodeStateFusion.from_config(
        model
    )

    assert fusion.node_type_count == (
        NODE_TYPE_COUNT
    )
    assert fusion.node_type_embedding_dim == (
        NODE_TYPE_EMBEDDING_DIM
    )


@pytest.mark.parametrize(
    "mode",
    (
        NODE_FUSION_PROJECTED_SUM,
        NODE_FUSION_GATED,
        NODE_FUSION_FILM,
    ),
)
def test_from_config_rejects_known_unimplemented_modes(
    mode: str,
) -> None:
    model = ModelConfig(
        static_input_dim=STATIC_DIM,
        hidden_dim=OUTPUT_DIM,
        fusion=NodeStateFusionConfig(
            mode=mode,
            output_dim=OUTPUT_DIM,
            include_static_state=True,
        ),
    )

    with pytest.raises(
        NotImplementedError,
        match="canonical but not implemented",
    ):
        NodeStateFusion.from_config(model)


# =============================================================================
# Component extraction
# =============================================================================


def test_extract_static_component() -> None:
    fusion = _static_only_fusion()
    inputs = _static_inputs(3)

    extracted = fusion.extract_component_tensors(
        inputs
    )

    assert isinstance(
        extracted,
        MappingProxyType,
    )
    assert tuple(extracted) == (
        FUSION_COMPONENT_STATIC_STATE,
    )
    assert extracted[
        FUSION_COMPONENT_STATIC_STATE
    ] is inputs.static_state.values


def test_extract_all_components_in_canonical_order() -> None:
    fusion = _all_component_fusion()
    inputs = _all_inputs(6)

    extracted = fusion.extract_component_tensors(
        inputs
    )

    assert tuple(extracted) == (
        fusion.component_order
    )
    assert extracted[
        FUSION_COMPONENT_STATIC_STATE
    ] is inputs.static_state.values
    assert extracted[
        FUSION_COMPONENT_MEMORY_STATE
    ] is inputs.memory_state.memory_state
    assert extracted[
        FUSION_COMPONENT_HAZARD_MEMORY_STATE
    ] is inputs.hazard_memory_state.values
    assert extracted[
        FUSION_COMPONENT_HAZARD_CONTEXT
    ] is inputs.hazard_context.query

    node_type_state = extracted[
        FUSION_COMPONENT_NODE_TYPE_EMBEDDING
    ]
    assert node_type_state.shape == (
        6,
        NODE_TYPE_EMBEDDING_DIM,
    )


def test_extracted_mapping_is_read_only() -> None:
    fusion = _static_only_fusion()
    extracted = fusion.extract_component_tensors(
        _static_inputs(2)
    )

    with pytest.raises(TypeError):
        extracted[
            "new"
        ] = torch.zeros(2, 3)  # type: ignore[index]


def test_extract_rejects_non_input_contract() -> None:
    fusion = _static_only_fusion()

    with pytest.raises(
        TypeError,
        match="NodeStateFusionInputs",
    ):
        fusion.extract_component_tensors(  # type: ignore[arg-type]
            torch.zeros(2, STATIC_DIM)
        )


def test_extract_requires_enabled_component() -> None:
    fusion = _static_only_fusion()
    inputs = NodeStateFusionInputs(
        alignment=_alignment(2),
    )

    with pytest.raises(
        ValueError,
        match="static_state is required",
    ):
        fusion.extract_component_tensors(
            inputs
        )


def test_extract_rejects_disabled_component() -> None:
    fusion = _static_only_fusion()
    alignment = _alignment(2)
    inputs = NodeStateFusionInputs(
        alignment=alignment,
        static_state=_component(
            2,
            STATIC_DIM,
            name=FUSION_COMPONENT_STATIC_STATE,
            alignment=alignment,
        ),
        memory_state=_memory_encoding(2),
    )

    with pytest.raises(
        ValueError,
        match="disabled",
    ):
        fusion.extract_component_tensors(
            inputs
        )


def test_extract_rejects_node_type_below_range() -> None:
    fusion = NodeStateFusion(
        mode=NODE_FUSION_CONCAT_PROJECTION,
        output_dim=OUTPUT_DIM,
        include_static_state=False,
        static_input_dim=None,
        include_memory_state=False,
        memory_input_dim=None,
        include_hazard_memory_state=False,
        hazard_memory_input_dim=None,
        include_hazard_context=False,
        hazard_context_input_dim=None,
        include_node_type_embedding=True,
        node_type_count=NODE_TYPE_COUNT,
        node_type_embedding_dim=(
            NODE_TYPE_EMBEDDING_DIM
        ),
    )
    inputs = NodeStateFusionInputs(
        alignment=_alignment(2),
        node_type_ids=torch.tensor(
            [-1, 0],
            dtype=torch.long,
        ),
    )

    with pytest.raises(
        IndexError,
        match="outside",
    ):
        fusion.extract_component_tensors(
            inputs
        )


def test_extract_rejects_node_type_above_range() -> None:
    fusion = NodeStateFusion(
        mode=NODE_FUSION_CONCAT_PROJECTION,
        output_dim=OUTPUT_DIM,
        include_static_state=False,
        static_input_dim=None,
        include_memory_state=False,
        memory_input_dim=None,
        include_hazard_memory_state=False,
        hazard_memory_input_dim=None,
        include_hazard_context=False,
        hazard_context_input_dim=None,
        include_node_type_embedding=True,
        node_type_count=NODE_TYPE_COUNT,
        node_type_embedding_dim=(
            NODE_TYPE_EMBEDDING_DIM
        ),
    )
    inputs = NodeStateFusionInputs(
        alignment=_alignment(2),
        node_type_ids=torch.tensor(
            [0, NODE_TYPE_COUNT],
            dtype=torch.long,
        ),
    )

    with pytest.raises(
        IndexError,
        match="outside",
    ):
        fusion.extract_component_tensors(
            inputs
        )


def test_extract_supports_empty_node_type_batch() -> None:
    fusion = NodeStateFusion(
        mode=NODE_FUSION_CONCAT_PROJECTION,
        output_dim=OUTPUT_DIM,
        include_static_state=False,
        static_input_dim=None,
        include_memory_state=False,
        memory_input_dim=None,
        include_hazard_memory_state=False,
        hazard_memory_input_dim=None,
        include_hazard_context=False,
        hazard_context_input_dim=None,
        include_node_type_embedding=True,
        node_type_count=NODE_TYPE_COUNT,
        node_type_embedding_dim=(
            NODE_TYPE_EMBEDDING_DIM
        ),
    )
    inputs = NodeStateFusionInputs(
        alignment=NodeAlignment(
            item_count=0
        ),
        node_type_ids=torch.empty(
            0,
            dtype=torch.long,
        ),
    )

    extracted = fusion.extract_component_tensors(
        inputs
    )

    assert extracted[
        FUSION_COMPONENT_NODE_TYPE_EMBEDDING
    ].shape == (
        0,
        NODE_TYPE_EMBEDDING_DIM,
    )


# =============================================================================
# Forward path
# =============================================================================


def test_forward_rejects_bare_tensor() -> None:
    fusion = _static_only_fusion()

    with pytest.raises(
        TypeError,
        match="bare tensors",
    ):
        fusion(  # type: ignore[arg-type]
            torch.zeros(2, STATIC_DIM)
        )


def test_forward_rejects_component_width_mismatch() -> None:
    fusion = _static_only_fusion()
    alignment = _alignment(2)
    inputs = NodeStateFusionInputs(
        alignment=alignment,
        static_state=_component(
            2,
            STATIC_DIM + 1,
            name=FUSION_COMPONENT_STATIC_STATE,
            alignment=alignment,
        ),
    )

    with pytest.raises(
        ValueError,
        match="width",
    ):
        fusion(inputs)


def test_static_forward_shape_metadata_and_finiteness() -> None:
    fusion = _static_only_fusion()
    fusion.eval()
    inputs = _static_inputs(4)

    output = fusion(inputs)

    assert isinstance(
        output,
        NodeStateFusionOutput,
    )
    assert output.fused_state.shape == (
        4,
        OUTPUT_DIM,
    )
    assert output.source_inputs is inputs
    assert output.alignment is inputs.alignment
    assert output.fusion_mode is (
        NodeStateFusionMode.CONCAT_PROJECTION
    )
    assert tuple(
        output.projected_components
    ) == (
        FUSION_COMPONENT_STATIC_STATE,
    )
    assert bool(
        torch.isfinite(
            output.fused_state
        ).all().item()
    )


def test_all_component_forward_preserves_sources() -> None:
    fusion = _all_component_fusion()
    fusion.eval()
    inputs = _all_inputs(6)

    output = fusion(inputs)

    assert output.source_inputs is inputs
    assert output.source_inputs.static_state is (
        inputs.static_state
    )
    assert output.source_inputs.memory_state is (
        inputs.memory_state
    )
    assert (
        output
        .source_inputs
        .hazard_memory_state
        is inputs.hazard_memory_state
    )
    assert output.source_inputs.hazard_context is (
        inputs.hazard_context
    )
    assert output.fused_state.shape == (
        6,
        OUTPUT_DIM,
    )
    assert tuple(
        output.projected_components
    ) == fusion.component_order


def test_forward_supports_empty_static_batch() -> None:
    fusion = _static_only_fusion()
    inputs = NodeStateFusionInputs(
        alignment=NodeAlignment(
            item_count=0,
        ),
        static_state=NodeStateComponent(
            values=torch.empty(
                0,
                STATIC_DIM,
            ),
            component_name=(
                FUSION_COMPONENT_STATIC_STATE
            ),
        ),
    )

    output = fusion(inputs)

    assert output.fused_state.shape == (
        0,
        OUTPUT_DIM,
    )


def test_eval_mode_is_deterministic_with_dropout() -> None:
    fusion = _static_only_fusion(
        dropout=0.75,
    )
    fusion.eval()
    inputs = _static_inputs(4)

    first = fusion(inputs)
    second = fusion(inputs)

    assert torch.equal(
        first.fused_state,
        second.fused_state,
    )


def test_output_lineage_matches_public_method() -> None:
    fusion = _static_only_fusion()
    inputs = _static_inputs(3)
    output = fusion(inputs)

    assert output.lineage_fingerprint == (
        fusion.lineage_fingerprint(inputs)
    )
    assert output.encoder_architecture_fingerprint == (
        fusion.architecture_fingerprint()
    )


# =============================================================================
# Forward and backward gradients
# =============================================================================


def test_static_backward_is_finite() -> None:
    fusion = _static_only_fusion()
    inputs = _static_inputs(
        4,
        requires_grad=True,
    )

    output = fusion(inputs)
    output.fused_state.square().mean().backward()

    assert inputs.static_state.values.grad is not None
    assert bool(
        torch.isfinite(
            inputs.static_state.values.grad
        ).all().item()
    )

    for parameter in fusion.parameters():
        assert parameter.grad is not None
        assert bool(
            torch.isfinite(
                parameter.grad
            ).all().item()
        )


def test_all_component_backward_reaches_dense_sources() -> None:
    fusion = _all_component_fusion()
    inputs = _all_inputs(
        6,
        requires_grad=True,
    )

    fusion(
        inputs
    ).fused_state.square().mean().backward()

    dense_sources = (
        inputs.static_state.values,
        inputs.memory_state.memory_state,
        inputs.hazard_memory_state.values,
        inputs.hazard_context.query,
    )

    for state in dense_sources:
        assert state.grad is not None
        assert bool(
            torch.isfinite(
                state.grad
            ).all().item()
        )

    assert (
        fusion
        .node_type_embedding
        .weight
        .grad
        is not None
    )


# =============================================================================
# Architecture, parameter, component, and lineage identity
# =============================================================================


def test_architecture_dict_is_complete() -> None:
    fusion = _all_component_fusion()
    architecture = fusion.architecture_dict()

    assert architecture[
        "schema_version"
    ] == NODE_STATE_FUSION_ENCODER_SCHEMA_VERSION
    assert architecture["mode"] == (
        NODE_FUSION_CONCAT_PROJECTION
    )
    assert architecture["component_order"] == list(
        fusion.component_order
    )
    assert architecture["output_dim"] == OUTPUT_DIM
    assert architecture[
        "include_node_type_embedding"
    ] is True
    assert architecture[
        "node_type_count"
    ] == NODE_TYPE_COUNT
    assert architecture[
        "fusion_algorithm"
    ] == fusion.fusion_algorithm.architecture_dict()


def test_architecture_fingerprint_is_stable() -> None:
    first = _static_only_fusion()
    second = _static_only_fusion()

    assert first.architecture_dict() == (
        second.architecture_dict()
    )
    assert first.architecture_fingerprint() == (
        second.architecture_fingerprint()
    )


@pytest.mark.parametrize(
    "builder",
    (
        lambda: _static_only_fusion(
            output_dim=OUTPUT_DIM + 1,
        ),
        lambda: _static_only_fusion(
            dropout=0.25,
        ),
        lambda: _static_only_fusion(
            layer_norm=False,
        ),
        _all_component_fusion,
    ),
)
def test_architecture_fingerprint_changes_with_contract(
    builder: Any,
) -> None:
    baseline = _static_only_fusion()
    changed = builder()

    assert baseline.architecture_fingerprint() != (
        changed.architecture_fingerprint()
    )


def test_parameter_fingerprint_is_reproducible_under_seed() -> None:
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(123)
        first = _all_component_fusion()

    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(123)
        second = _all_component_fusion()

    assert first.parameter_fingerprint() == (
        second.parameter_fingerprint()
    )


def test_parameter_fingerprint_changes_after_mutation() -> None:
    fusion = _static_only_fusion()
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


def test_component_fingerprint_views_are_immutable() -> None:
    fusion = _all_component_fusion()

    architecture = (
        fusion.component_architecture_fingerprints()
    )
    parameters = (
        fusion.component_parameter_fingerprints()
    )

    assert isinstance(
        architecture,
        MappingProxyType,
    )
    assert isinstance(
        parameters,
        MappingProxyType,
    )
    assert tuple(architecture) == (
        fusion.component_order
    )
    assert tuple(parameters) == (
        fusion.component_order
    )


def test_input_lineage_changes_output_lineage() -> None:
    fusion = _static_only_fusion()

    first = _static_inputs(2)
    second_alignment = _alignment(2)
    second = NodeStateFusionInputs(
        alignment=second_alignment,
        static_state=_component(
            2,
            STATIC_DIM,
            name=FUSION_COMPONENT_STATIC_STATE,
            offset=1.0,
            alignment=second_alignment,
        ),
        source_fingerprint="fusion-input-source",
    )

    assert fusion.lineage_fingerprint(
        first
    ) != fusion.lineage_fingerprint(
        second
    )


def test_lineage_rejects_non_input_contract() -> None:
    fusion = _static_only_fusion()

    with pytest.raises(
        TypeError,
        match="NodeStateFusionInputs",
    ):
        fusion.lineage_fingerprint(  # type: ignore[arg-type]
            object()
        )


def test_state_dict_keys_show_split_architecture() -> None:
    fusion = _all_component_fusion()
    keys = tuple(
        fusion.state_dict()
    )

    assert (
        "node_type_embedding.weight"
        in keys
    )
    assert any(
        key.startswith(
            "fusion_algorithm."
            "component_projections."
            "static_state.linear."
        )
        for key in keys
    )
    assert (
        "fusion_algorithm."
        "fusion_network.linear_in.weight"
        in keys
    )


def test_finite_parameter_check_passes() -> None:
    fusion = _all_component_fusion()
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
    fusion = _static_only_fusion()

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


def test_extra_repr_contains_public_contract() -> None:
    fusion = _all_component_fusion()
    representation = fusion.extra_repr()

    assert "concat_projection" in representation
    assert f"output_dim={OUTPUT_DIM}" in representation
    assert "component_order" in representation


# =============================================================================
# Legacy checkpoint migration
# =============================================================================


def test_upgrade_legacy_state_dict_maps_all_keys() -> None:
    fusion = _all_component_fusion()
    legacy = _legacy_state_dict(fusion)

    upgraded = fusion.upgrade_legacy_state_dict(
        legacy
    )

    assert tuple(upgraded) == tuple(
        fusion.state_dict()
    )

    for key, tensor in fusion.state_dict().items():
        assert torch.equal(
            upgraded[key],
            tensor,
        )


def test_load_legacy_state_dict_preserves_outputs() -> None:
    source = _all_component_fusion()
    target = _all_component_fusion()
    legacy = _legacy_state_dict(source)

    target.load_legacy_state_dict(
        legacy,
        strict=True,
    )

    source.eval()
    target.eval()
    inputs = _all_inputs(6)

    source_output = source(inputs)
    target_output = target(inputs)

    assert torch.equal(
        source_output.fused_state,
        target_output.fused_state,
    )

    for name in source.component_order:
        assert torch.equal(
            source_output.projected_components[name],
            target_output.projected_components[name],
        )


def test_upgrade_retains_already_split_keys() -> None:
    fusion = _static_only_fusion()
    current = OrderedDict(
        (
            key,
            value.clone(),
        )
        for key, value in fusion.state_dict().items()
    )

    upgraded = fusion.upgrade_legacy_state_dict(
        current
    )

    assert tuple(upgraded) == tuple(current)
    for key in current:
        assert torch.equal(
            upgraded[key],
            current[key],
        )


def test_upgrade_retains_unknown_key_for_loader_diagnostics() -> None:
    fusion = _static_only_fusion()
    upgraded = fusion.upgrade_legacy_state_dict(
        {
            "unknown.weight": torch.ones(1),
        }
    )

    assert tuple(upgraded) == (
        "unknown.weight",
    )


def test_upgrade_rejects_non_mapping() -> None:
    fusion = _static_only_fusion()

    with pytest.raises(
        TypeError,
        match="mapping",
    ):
        fusion.upgrade_legacy_state_dict(  # type: ignore[arg-type]
            []
        )


def test_upgrade_rejects_non_string_key() -> None:
    fusion = _static_only_fusion()

    with pytest.raises(
        TypeError,
        match="keys must be strings",
    ):
        fusion.upgrade_legacy_state_dict(
            {
                1: torch.ones(1),  # type: ignore[dict-item]
            }
        )


def test_upgrade_rejects_non_tensor_value() -> None:
    fusion = _static_only_fusion()

    with pytest.raises(
        TypeError,
        match="must be a tensor",
    ):
        fusion.upgrade_legacy_state_dict(
            {
                "weight": [1.0],  # type: ignore[dict-item]
            }
        )


def test_upgrade_detects_key_collision() -> None:
    fusion = _static_only_fusion()
    current_key = (
        "fusion_algorithm."
        "fusion_network.linear_in.weight"
    )
    legacy_key = "fusion_network.0.weight"
    tensor = fusion.state_dict()[
        current_key
    ]

    with pytest.raises(
        ValueError,
        match="collision",
    ):
        fusion.upgrade_legacy_state_dict(
            OrderedDict(
                (
                    (current_key, tensor),
                    (legacy_key, tensor),
                )
            )
        )


# =============================================================================
# Optional device contract
# =============================================================================


@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA is unavailable.",
)
def test_extract_rejects_module_input_device_mismatch() -> None:
    fusion = _static_only_fusion().cuda()
    inputs = _static_inputs(2)

    with pytest.raises(
        ValueError,
        match="share one device",
    ):
        fusion.extract_component_tensors(
            inputs
        )


@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA is unavailable.",
)
def test_cuda_forward_and_backward_are_finite() -> None:
    fusion = _static_only_fusion().cuda()
    alignment = _alignment(3)
    values = (
        torch.arange(
            3 * STATIC_DIM,
            dtype=torch.float32,
            device="cuda",
        )
        .reshape(3, STATIC_DIM)
        / 10.0
    )
    values.requires_grad_(True)
    inputs = NodeStateFusionInputs(
        alignment=alignment,
        static_state=NodeStateComponent(
            values=values,
            component_name=(
                FUSION_COMPONENT_STATIC_STATE
            ),
        ),
    )

    output = fusion(inputs)
    output.fused_state.square().mean().backward()

    assert output.fused_state.device.type == "cuda"
    assert values.grad is not None
    assert bool(
        torch.isfinite(
            values.grad
        ).all().item()
    )
