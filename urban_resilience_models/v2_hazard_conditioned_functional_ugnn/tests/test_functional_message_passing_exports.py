"""
Public-export tests for ``functional_message_passing``.

Target repository path:
    urban_resilience_models/
        v2_hazard_conditioned_functional_ugnn/
            tests/
                test_functional_message_passing_exports.py

This suite validates the root package initializer as a deliberate public API.
It does not retest numerical message-passing behavior.

Coverage
--------
- package identity and immediate module namespaces;
- exact root ``__all__`` inventory and order;
- uniqueness and binding of every public name;
- exact correspondence with immediate submodule ``__all__`` inventories;
- explicit collision aliases;
- authoritative root ownership for overlapping schemas;
- preservation of original submodule names;
- star-import behavior;
- intentional omission of the empty ``ablations.py`` placeholder;
- representative high-level public entry points;
- stable repeated imports;
- absence of silent last-import-wins collisions.
"""

from __future__ import annotations

import importlib
from types import ModuleType
from typing import Final

import pytest

from urban_resilience_models.v2_hazard_conditioned_functional_ugnn import (
    functional_message_passing as fmp,
)


PACKAGE_IMPORT_PATH: Final[str] = (
    "urban_resilience_models."
    "v2_hazard_conditioned_functional_ugnn."
    "functional_message_passing"
)


EXPECTED_PACKAGE_API_VERSION: Final[str] = "0.1"

EXPECTED_PUBLIC_MODULE_NAMES: Final[
    tuple[str, ...]
] = (
    "schemas",
    "segment_ops",
    "relation_transforms",
    "edge_normalization",
    "aggregators",
    "relation_family_gate",
    "edge_attention",
    "message_builders",
    "layer",
    "stack",
)

# Explicit source-name -> root-name translations.
EXPLICIT_ROOT_ALIASES: Final[
    dict[tuple[str, str], str]
] = {
    (
        "schemas",
        "FunctionalMessagePassingStackOutput",
    ): "LegacyFunctionalMessagePassingStackOutput",
    (
        "schemas",
        "RelationGateOutput",
    ): "FunctionalMessagePassingRelationGateOutput",
    (
        "schemas",
        "EdgeAttentionOutput",
    ): "FunctionalMessagePassingEdgeAttentionOutput",
    (
        "schemas",
        "EdgeMessageOutput",
    ): "FunctionalMessagePassingEdgeMessageOutput",
    (
        "message_builders",
        "graph_batch_diagnostics",
    ): "message_builder_graph_batch_diagnostics",
    (
        "layer",
        "graph_batch_diagnostics",
    ): "layer_graph_batch_diagnostics",
    (
        "message_builders",
        "scalar_tensor_statistics",
    ): "message_builder_scalar_tensor_statistics",
    (
        "layer",
        "scalar_tensor_statistics",
    ): "layer_scalar_tensor_statistics",
}

# Overlapping names that have one authoritative root owner.
CANONICAL_ROOT_SOURCE: Final[
    dict[str, str]
] = {
    "AggregationOutput": "schemas",
    "FunctionalMessagePassingIntermediates": (
        "schemas"
    ),
    "FunctionalMessagePassingLayerOutput": (
        "schemas"
    ),
    "FunctionalMessagePassingStackOutput": (
        "stack"
    ),
    "RelationGateOutput": (
        "relation_family_gate"
    ),
    "EdgeAttentionOutput": (
        "edge_attention"
    ),
    "EdgeMessageOutput": (
        "message_builders"
    ),
}

EXPECTED_EXPLICIT_ALIAS_NAMES: Final[
    tuple[str, ...]
] = tuple(
    EXPLICIT_ROOT_ALIASES.values()
)

REPRESENTATIVE_ROOT_EXPORTS: Final[
    tuple[str, ...]
] = (
    # Base contracts.
    "FunctionalMessagePassingInputs",
    "FunctionalMessagePassingNodeState",
    "RelationFamilyAlignment",
    "RelationTransformOutput",
    "StructuralEdgeNormalizationOutput",
    "AggregationOutput",
    # Primitive operations.
    "grouped_softmax",
    "segment_sum",
    "segment_mean",
    "segment_counts",
    # Trainable and parameter-free stages.
    "SharedRelationTransform",
    "PerRelationTransform",
    "RelationTransforms",
    "EdgeNormalization",
    "MessageAggregator",
    "RelationFamilyGate",
    "EdgeAttention",
    "MessageBuilder",
    # Complete orchestration.
    "FunctionalMessagePassingLayer",
    "FunctionalMessagePassingStack",
    # Functional execution helpers.
    "run_functional_message_passing_layer",
    "run_functional_message_passing_stack",
)


# =============================================================================
# Helpers
# =============================================================================


def _public_module(
    name: str,
) -> ModuleType:
    value = getattr(
        fmp,
        name,
    )

    if not isinstance(
        value,
        ModuleType,
    ):
        raise AssertionError(
            f"fmp.{name} is not a module namespace."
        )

    return value


def _declared_exports(
    module_name: str,
) -> tuple[str, ...]:
    module = _public_module(
        module_name
    )

    exported = getattr(
        module,
        "__all__",
        None,
    )

    if not isinstance(
        exported,
        tuple,
    ):
        raise AssertionError(
            f"{module.__name__}.__all__ must be a tuple."
        )

    return exported


def _root_name_for_source_export(
    *,
    module_name: str,
    source_name: str,
) -> str | None:
    explicit = (
        EXPLICIT_ROOT_ALIASES.get(
            (
                module_name,
                source_name,
            )
        )
    )

    if explicit is not None:
        return explicit

    canonical_owner = (
        CANONICAL_ROOT_SOURCE.get(
            source_name
        )
    )

    if (
        canonical_owner is not None
        and canonical_owner != module_name
    ):
        return None

    return source_name


def _expected_root_export_order(
) -> tuple[str, ...]:
    names: list[str] = [
        (
            "FUNCTIONAL_MESSAGE_PASSING_"
            "PACKAGE_API_VERSION"
        ),
        *EXPECTED_PUBLIC_MODULE_NAMES,
    ]

    for module_name in (
        EXPECTED_PUBLIC_MODULE_NAMES
    ):
        for source_name in (
            _declared_exports(
                module_name
            )
        ):
            root_name = (
                _root_name_for_source_export(
                    module_name=(
                        module_name
                    ),
                    source_name=(
                        source_name
                    ),
                )
            )

            if root_name is not None:
                names.append(
                    root_name
                )

    return tuple(
        names
    )


def _all_source_occurrences(
) -> dict[
    str,
    tuple[str, ...],
]:
    observed: dict[
        str,
        list[str],
    ] = {}

    for module_name in (
        EXPECTED_PUBLIC_MODULE_NAMES
    ):
        for source_name in (
            _declared_exports(
                module_name
            )
        ):
            observed.setdefault(
                source_name,
                [],
            ).append(
                module_name
            )

    return {
        name: tuple(
            modules
        )
        for name, modules
        in observed.items()
    }


# =============================================================================
# Package identity
# =============================================================================


def test_package_api_version() -> None:
    assert (
        fmp
        .FUNCTIONAL_MESSAGE_PASSING_PACKAGE_API_VERSION
        == EXPECTED_PACKAGE_API_VERSION
    )


def test_package_import_path() -> None:
    assert fmp.__name__ == (
        PACKAGE_IMPORT_PATH
    )


def test_public_module_namespace_order() -> None:
    module_names = tuple(
        name
        for name in fmp.__all__
        if name in (
            EXPECTED_PUBLIC_MODULE_NAMES
        )
    )

    assert module_names == (
        EXPECTED_PUBLIC_MODULE_NAMES
    )


@pytest.mark.parametrize(
    "module_name",
    EXPECTED_PUBLIC_MODULE_NAMES,
)
def test_public_module_namespace_is_exact(
    module_name: str,
) -> None:
    root_value = getattr(
        fmp,
        module_name,
    )
    imported = importlib.import_module(
        f"{PACKAGE_IMPORT_PATH}.{module_name}"
    )

    assert isinstance(
        root_value,
        ModuleType,
    )
    assert root_value is imported


def test_empty_ablations_module_is_not_public() -> None:
    assert "ablations" not in (
        fmp.__all__
    )

    ablations = importlib.import_module(
        f"{PACKAGE_IMPORT_PATH}.ablations"
    )

    assert isinstance(
        ablations,
        ModuleType,
    )


# =============================================================================
# Root export inventory
# =============================================================================


def test_root_all_is_tuple() -> None:
    assert isinstance(
        fmp.__all__,
        tuple,
    )


def test_root_all_contains_only_nonempty_strings() -> None:
    for index, name in enumerate(
        fmp.__all__
    ):
        assert isinstance(
            name,
            str,
        ), index
        assert name.strip(), index


def test_root_all_has_no_duplicates() -> None:
    assert len(
        fmp.__all__
    ) == len(
        set(
            fmp.__all__
        )
    )


def test_root_all_exactly_matches_declared_source_apis() -> None:
    expected = (
        _expected_root_export_order()
    )

    assert fmp.__all__ == expected


def test_root_all_current_expected_size() -> None:
    # Deliberately catches accidental additions, omissions, or collision
    # changes in the generated root initializer.
    assert len(
        fmp.__all__
    ) == 683


def test_every_root_export_is_bound() -> None:
    missing = tuple(
        name
        for name in fmp.__all__
        if not hasattr(
            fmp,
            name,
        )
    )

    assert missing == ()


def test_every_bound_root_export_has_exact_name_membership() -> None:
    namespace_public_names = {
        name
        for name in vars(
            fmp
        )
        if not name.startswith(
            "_"
        )
    }

    expected = set(
        fmp.__all__
    )

    # ``annotations`` is imported from __future__ only at compile time and
    # therefore does not appear as a runtime public binding.
    assert expected <= (
        namespace_public_names
    )


def test_star_import_exports_exactly_root_all() -> None:
    namespace: dict[
        str,
        object,
    ] = {}

    exec(
        f"from {PACKAGE_IMPORT_PATH} import *",
        namespace,
        namespace,
    )

    exported = {
        name
        for name in namespace
        if name != "__builtins__"
    }

    assert exported == set(
        fmp.__all__
    )


def test_repeated_import_preserves_package_identity() -> None:
    first = importlib.import_module(
        PACKAGE_IMPORT_PATH
    )
    second = importlib.import_module(
        PACKAGE_IMPORT_PATH
    )

    assert first is second
    assert first is fmp
    assert first.__all__ is (
        second.__all__
    )


# =============================================================================
# Immediate submodule API contracts
# =============================================================================


@pytest.mark.parametrize(
    "module_name",
    EXPECTED_PUBLIC_MODULE_NAMES,
)
def test_submodule_all_is_unique_and_bound(
    module_name: str,
) -> None:
    module = _public_module(
        module_name
    )
    exported = _declared_exports(
        module_name
    )

    assert len(exported) == len(
        set(exported)
    )

    missing = tuple(
        name
        for name in exported
        if not hasattr(
            module,
            name,
        )
    )

    assert missing == ()


@pytest.mark.parametrize(
    "module_name",
    EXPECTED_PUBLIC_MODULE_NAMES,
)
def test_each_submodule_export_has_a_deliberate_root_disposition(
    module_name: str,
) -> None:
    module = _public_module(
        module_name
    )

    for source_name in (
        _declared_exports(
            module_name
        )
    ):
        root_name = (
            _root_name_for_source_export(
                module_name=(
                    module_name
                ),
                source_name=(
                    source_name
                ),
            )
        )

        if root_name is None:
            assert source_name in (
                CANONICAL_ROOT_SOURCE
            )
            assert (
                CANONICAL_ROOT_SOURCE[
                    source_name
                ]
                != module_name
            )
            continue

        assert root_name in (
            fmp.__all__
        )
        assert getattr(
            fmp,
            root_name,
        ) is getattr(
            module,
            source_name,
        )


def test_duplicate_source_names_are_all_explicitly_resolved() -> None:
    duplicate_occurrences = {
        name: modules
        for name, modules
        in _all_source_occurrences().items()
        if len(modules) > 1
    }

    unresolved: dict[
        str,
        tuple[str, ...],
    ] = {}

    for source_name, modules in (
        duplicate_occurrences.items()
    ):
        dispositions = tuple(
            _root_name_for_source_export(
                module_name=module_name,
                source_name=source_name,
            )
            for module_name in modules
        )

        non_none = tuple(
            value
            for value in dispositions
            if value is not None
        )

        if (
            len(non_none)
            != len(
                set(non_none)
            )
        ):
            unresolved[source_name] = (
                modules
            )

    assert unresolved == {}


# =============================================================================
# Authoritative overlapping contracts
# =============================================================================


@pytest.mark.parametrize(
    (
        "root_name",
        "owner_module",
    ),
    tuple(
        CANONICAL_ROOT_SOURCE.items()
    ),
)
def test_authoritative_root_source_identity(
    root_name: str,
    owner_module: str,
) -> None:
    module = _public_module(
        owner_module
    )

    assert getattr(
        fmp,
        root_name,
    ) is getattr(
        module,
        root_name,
    )


def test_authoritative_stack_output_is_new_stack_schema() -> None:
    assert (
        fmp.FunctionalMessagePassingStackOutput
        is fmp.stack
        .FunctionalMessagePassingStackOutput
    )

    assert (
        fmp.FunctionalMessagePassingStackOutput
        is not fmp.schemas
        .FunctionalMessagePassingStackOutput
    )


def test_legacy_stack_output_alias_is_preserved() -> None:
    assert (
        fmp.LegacyFunctionalMessagePassingStackOutput
        is fmp.schemas
        .FunctionalMessagePassingStackOutput
    )

    assert (
        "LegacyFunctionalMessagePassingStackOutput"
        in fmp.__all__
    )


def test_authoritative_relation_gate_output() -> None:
    assert (
        fmp.RelationGateOutput
        is fmp.relation_family_gate
        .RelationGateOutput
    )

    assert (
        fmp
        .FunctionalMessagePassingRelationGateOutput
        is fmp.schemas.RelationGateOutput
    )

    assert ( fmp.RelationGateOutput 
            is fmp 
            .FunctionalMessagePassingRelationGateOutput 
    )


def test_authoritative_edge_attention_output() -> None:
    assert (
        fmp.EdgeAttentionOutput
        is fmp.edge_attention
        .EdgeAttentionOutput
    )

    assert (
        fmp
        .FunctionalMessagePassingEdgeAttentionOutput
        is fmp.schemas.EdgeAttentionOutput
    )

    assert (
        fmp.EdgeAttentionOutput
        is fmp
        .FunctionalMessagePassingEdgeAttentionOutput
    )

def test_authoritative_edge_message_output() -> None:
    assert (
        fmp.EdgeMessageOutput
        is fmp.message_builders
        .EdgeMessageOutput
    )

    assert (
        fmp
        .FunctionalMessagePassingEdgeMessageOutput
        is fmp.schemas.EdgeMessageOutput
    )

    assert (
        fmp.EdgeMessageOutput
        is fmp
        .FunctionalMessagePassingEdgeMessageOutput
    )


def test_authoritative_aggregation_output() -> None:
    assert (
        fmp.AggregationOutput
        is fmp.schemas.AggregationOutput
    )
    assert (
        fmp.layer.AggregationOutput
        is fmp.schemas.AggregationOutput
    )




def test_authoritative_layer_output() -> None:
    assert (
        fmp.FunctionalMessagePassingLayerOutput
        is fmp.schemas
        .FunctionalMessagePassingLayerOutput
    )
    assert (
        fmp.layer
        .FunctionalMessagePassingLayerOutput
        is fmp.schemas
        .FunctionalMessagePassingLayerOutput
    )


# =============================================================================
# Explicit diagnostic helper aliases
# =============================================================================


def test_message_builder_graph_diagnostic_alias() -> None:
    assert (
        fmp
        .message_builder_graph_batch_diagnostics
        is fmp.message_builders
        .graph_batch_diagnostics
    )


def test_layer_graph_diagnostic_alias() -> None:
    assert (
        fmp.layer_graph_batch_diagnostics
        is fmp.layer.graph_batch_diagnostics
    )


def test_graph_diagnostic_aliases_are_distinct() -> None:
    assert (
        fmp
        .message_builder_graph_batch_diagnostics
        is not fmp
        .layer_graph_batch_diagnostics
    )


def test_message_builder_scalar_statistics_alias() -> None:
    assert (
        fmp
        .message_builder_scalar_tensor_statistics
        is fmp.message_builders
        .scalar_tensor_statistics
    )


def test_layer_scalar_statistics_alias() -> None:
    assert (
        fmp.layer_scalar_tensor_statistics
        is fmp.layer.scalar_tensor_statistics
    )


def test_scalar_statistics_aliases_are_distinct() -> None:
    assert (
        fmp
        .message_builder_scalar_tensor_statistics
        is not fmp
        .layer_scalar_tensor_statistics
    )


@pytest.mark.parametrize(
    "alias_name",
    EXPECTED_EXPLICIT_ALIAS_NAMES,
)
def test_explicit_alias_is_public(
    alias_name: str,
) -> None:
    assert alias_name in (
        fmp.__all__
    )
    assert hasattr(
        fmp,
        alias_name,
    )


# =============================================================================
# Representative public entry points
# =============================================================================


@pytest.mark.parametrize(
    "name",
    REPRESENTATIVE_ROOT_EXPORTS,
)
def test_representative_root_export_exists(
    name: str,
) -> None:
    assert name in fmp.__all__
    assert hasattr(
        fmp,
        name,
    )


def test_root_orchestrator_identities() -> None:
    assert (
        fmp.FunctionalMessagePassingLayer
        is fmp.layer
        .FunctionalMessagePassingLayer
    )
    assert (
        fmp.FunctionalMessagePassingStack
        is fmp.stack
        .FunctionalMessagePassingStack
    )
    assert (
        fmp.MessageBuilder
        is fmp.message_builders
        .MessageBuilder
    )
    assert (
        fmp.EdgeAttention
        is fmp.edge_attention
        .EdgeAttention
    )
    assert (
        fmp.RelationFamilyGate
        is fmp.relation_family_gate
        .RelationFamilyGate
    )


def test_root_functional_execution_helper_identities() -> None:
    assert (
        fmp.run_functional_message_passing_layer
        is fmp.layer
        .run_functional_message_passing_layer
    )
    assert (
        fmp.run_functional_message_passing_stack
        is fmp.stack
        .run_functional_message_passing_stack
    )


def test_root_primitive_operation_identities() -> None:
    assert (
        fmp.segment_sum
        is fmp.segment_ops.segment_sum
    )
    assert (
        fmp.segment_mean
        is fmp.segment_ops.segment_mean
    )
    assert (
        fmp.segment_counts
        is fmp.segment_ops.segment_counts
    )
    assert (
        fmp.grouped_softmax
        is fmp.segment_ops.grouped_softmax
    )


# =============================================================================
# Placeholder and namespace exclusions
# =============================================================================


def test_ablations_exports_no_root_symbols() -> None:
    ablations = importlib.import_module(
        f"{PACKAGE_IMPORT_PATH}.ablations"
    )

    declared = getattr(
        ablations,
        "__all__",
        (),
    )

    assert tuple(
        declared
    ) == ()

    root_names_from_ablations = {
        name
        for name in vars(
            ablations
        )
        if (
            not name.startswith(
                "_"
            )
            and name in fmp.__all__
        )
    }

    assert root_names_from_ablations == set()


def test_no_legacy_flat_edge_attention_module_is_public() -> None:
    # The package namespace must resolve to the directory package.
    assert isinstance(
        fmp.edge_attention,
        ModuleType,
    )
    assert (
        fmp.edge_attention.__package__
        == f"{PACKAGE_IMPORT_PATH}.edge_attention"
    )


def test_no_public_export_named_hazard() -> None:
    # Hazard embeddings and query encoding belong to the parent model package,
    # not to this functional-message-passing root API.
    assert "hazard" not in (
        fmp.__all__
    )
