"""
Tests for the canonical V2 hazard registry.

Target path:
    urban_resilience_models/
        v2_hazard_conditioned_functional_ugnn/
            tests/
                test_hazard_registry.py

Implementation under test:
    urban_resilience_models/
        v2_hazard_conditioned_functional_ugnn/
            hazard/
                hazard_registry.py

Python requirement:
    Python >= 3.11

These tests intentionally freeze:

- canonical hazard names;
- stable sparse IDs;
- ontology roles;
- flood and winter-weather hierarchies;
- queryability versus operational support;
- fallback and unknown-hazard semantics;
- canonical ordering;
- fingerprint layers;
- serialization and compatibility behavior;
- immutability contracts.

Run from the repository root:

    pytest -q urban_resilience_models/v2_hazard_conditioned_functional_ugnn/tests/test_hazard_registry.py
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import FrozenInstanceError, replace
import json
from typing import Any

import pytest

from urban_resilience_models.v2_hazard_conditioned_functional_ugnn.hazard.hazard_registry import (
    ALL_HAZARDS,
    ALL_PRIOR_HAZARDS,
    CANONICAL_HAZARD_DISPLAY_NAMES,
    CANONICAL_HAZARD_ONTOLOGY_ROLES,
    CANONICAL_HAZARD_PARENTS,
    CANONICAL_HAZARD_QUERY_ALLOWED,
    CANONICAL_HAZARD_STABLE_IDS,
    DEFAULT_HAZARD_REGISTRY,
    DEFAULT_HAZARD_REGISTRY_IDENTITY,
    DEFAULT_HAZARD_REGISTRY_NAME,
    DISRUPTION_HAZARDS,
    EVENT_FAMILY_HAZARDS,
    FALLBACK_HAZARD,
    FALLBACK_PRIOR_HAZARD,
    HAZARD_ID_ALL_HAZARD,
    HAZARD_ID_CIVIL_SECURITY_EVENT,
    HAZARD_ID_FLOOD,
    HAZARD_ID_FREEZING_RAIN,
    HAZARD_ID_HEAT,
    HAZARD_ID_OUTAGE,
    HAZARD_ID_PLUVIAL_FLOOD,
    HAZARD_ID_RIVERINE_FLOOD,
    HAZARD_ID_ROAD_DISRUPTION,
    HAZARD_ID_SNOWSTORM,
    HAZARD_ID_UNKNOWN,
    HAZARD_ID_WINTER_STORM,
    HAZARD_REGISTRY_SCHEMA_VERSION,
    HAZARD_REGISTRY_VERSION,
    HazardKind,
    HazardOntologyRole,
    HazardRegistry,
    HazardRegistryEntry,
    HazardRegistryIdentity,
    HazardSupportState,
    MECHANISM_FAMILY_HAZARDS,
    MECHANISM_HAZARDS,
    QUERYABLE_HAZARDS,
    RUNTIME_HAZARDS,
    RUNTIME_PRIOR_HAZARDS,
    UNKNOWN_HAZARD_NAME,
    build_default_hazard_registry,
    get_default_hazard_registry,
    get_default_hazard_registry_identity,
    normalize_hazard_kind,
    normalize_hazard_name,
)


# =============================================================================
# Fixtures and helpers
# =============================================================================


@pytest.fixture()
def registry() -> HazardRegistry:
    """Return the canonical default V2 hazard registry."""

    return DEFAULT_HAZARD_REGISTRY


def _entry(
    registry: HazardRegistry,
    hazard: HazardKind,
) -> HazardRegistryEntry:
    """Return one canonical registry entry."""

    return registry.get_by_name(hazard)


def _replace_registry_entry(
    registry: HazardRegistry,
    hazard: HazardKind,
    **changes: Any,
) -> HazardRegistry:
    """
    Return a registry with one entry replaced.

    Registry construction revalidates and recanonicalizes the complete
    ontology.
    """

    replacement = replace(
        registry.get_by_name(hazard),
        **changes,
    )

    entries = tuple(
        replacement if entry.name == hazard else entry
        for entry in registry.entries
    )

    return replace(
        registry,
        entries=entries,
    )


# =============================================================================
# Default registry identity
# =============================================================================


def test_default_registry_constructs() -> None:
    registry = build_default_hazard_registry()

    assert isinstance(registry, HazardRegistry)
    registry.assert_current_compatibility()


def test_default_getters_return_singleton_registry(
    registry: HazardRegistry,
) -> None:
    assert get_default_hazard_registry() is registry
    assert get_default_hazard_registry_identity() is registry
    assert DEFAULT_HAZARD_REGISTRY_IDENTITY is registry


def test_canonical_class_name_and_compatibility_alias() -> None:
    assert HazardRegistryIdentity is HazardRegistry


def test_default_registry_versions(
    registry: HazardRegistry,
) -> None:
    assert registry.registry_name == DEFAULT_HAZARD_REGISTRY_NAME
    assert registry.registry_version == HAZARD_REGISTRY_VERSION
    assert registry.schema_version == HAZARD_REGISTRY_SCHEMA_VERSION


def test_exact_canonical_hazard_order(
    registry: HazardRegistry,
) -> None:
    assert registry.hazard_kinds == (
        HazardKind.FLOOD,
        HazardKind.RIVERINE_FLOOD,
        HazardKind.PLUVIAL_FLOOD,
        HazardKind.HEAT,
        HazardKind.OUTAGE,
        HazardKind.ROAD_DISRUPTION,
        HazardKind.CIVIL_SECURITY_EVENT,
        HazardKind.WINTER_STORM,
        HazardKind.SNOWSTORM,
        HazardKind.FREEZING_RAIN,
        HazardKind.ALL_HAZARD,
    )


def test_entries_are_ordered_by_stable_id(
    registry: HazardRegistry,
) -> None:
    assert registry.stable_hazard_ids == (
        HAZARD_ID_FLOOD,
        HAZARD_ID_RIVERINE_FLOOD,
        HAZARD_ID_PLUVIAL_FLOOD,
        HAZARD_ID_HEAT,
        HAZARD_ID_OUTAGE,
        HAZARD_ID_ROAD_DISRUPTION,
        HAZARD_ID_CIVIL_SECURITY_EVENT,
        HAZARD_ID_WINTER_STORM,
        HAZARD_ID_SNOWSTORM,
        HAZARD_ID_FREEZING_RAIN,
        HAZARD_ID_ALL_HAZARD,
    )

    assert registry.stable_hazard_ids == tuple(
        sorted(registry.stable_hazard_ids)
    )


def test_stable_ids_are_unique(
    registry: HazardRegistry,
) -> None:
    assert len(set(registry.stable_hazard_ids)) == len(
        registry.stable_hazard_ids
    )


@pytest.mark.parametrize(
    ("hazard", "expected_id"),
    (
        (HazardKind.FLOOD, HAZARD_ID_FLOOD),
        (
            HazardKind.RIVERINE_FLOOD,
            HAZARD_ID_RIVERINE_FLOOD,
        ),
        (
            HazardKind.PLUVIAL_FLOOD,
            HAZARD_ID_PLUVIAL_FLOOD,
        ),
        (HazardKind.HEAT, HAZARD_ID_HEAT),
        (HazardKind.OUTAGE, HAZARD_ID_OUTAGE),
        (
            HazardKind.ROAD_DISRUPTION,
            HAZARD_ID_ROAD_DISRUPTION,
        ),
        (
            HazardKind.CIVIL_SECURITY_EVENT,
            HAZARD_ID_CIVIL_SECURITY_EVENT,
        ),
        (
            HazardKind.WINTER_STORM,
            HAZARD_ID_WINTER_STORM,
        ),
        (
            HazardKind.SNOWSTORM,
            HAZARD_ID_SNOWSTORM,
        ),
        (
            HazardKind.FREEZING_RAIN,
            HAZARD_ID_FREEZING_RAIN,
        ),
        (
            HazardKind.ALL_HAZARD,
            HAZARD_ID_ALL_HAZARD,
        ),
    ),
)
def test_canonical_stable_ids(
    registry: HazardRegistry,
    hazard: HazardKind,
    expected_id: int,
) -> None:
    entry = registry.get_by_name(hazard)

    assert entry.stable_hazard_id == expected_id
    assert (
        CANONICAL_HAZARD_STABLE_IDS[hazard.value]
        == expected_id
    )


@pytest.mark.parametrize("hazard", tuple(HazardKind))
def test_canonical_display_names(
    registry: HazardRegistry,
    hazard: HazardKind,
) -> None:
    assert (
        registry.get_by_name(hazard).display_name
        == CANONICAL_HAZARD_DISPLAY_NAMES[hazard.value]
    )


def test_unknown_hazard_is_reserved_but_not_semantic(
    registry: HazardRegistry,
) -> None:
    assert UNKNOWN_HAZARD_NAME not in registry
    assert HAZARD_ID_UNKNOWN not in registry
    assert UNKNOWN_HAZARD_NAME not in registry.hazard_names
    assert HAZARD_ID_UNKNOWN not in registry.stable_hazard_ids


# =============================================================================
# Ontology roles and hierarchy
# =============================================================================


@pytest.mark.parametrize(
    ("hazard", "role"),
    (
        (
            HazardKind.FLOOD,
            HazardOntologyRole.MECHANISM_FAMILY,
        ),
        (
            HazardKind.RIVERINE_FLOOD,
            HazardOntologyRole.MECHANISM,
        ),
        (
            HazardKind.PLUVIAL_FLOOD,
            HazardOntologyRole.MECHANISM,
        ),
        (
            HazardKind.HEAT,
            HazardOntologyRole.MECHANISM,
        ),
        (
            HazardKind.OUTAGE,
            HazardOntologyRole.DISRUPTION,
        ),
        (
            HazardKind.ROAD_DISRUPTION,
            HazardOntologyRole.DISRUPTION,
        ),
        (
            HazardKind.CIVIL_SECURITY_EVENT,
            HazardOntologyRole.EVENT_FAMILY,
        ),
        (
            HazardKind.WINTER_STORM,
            HazardOntologyRole.MECHANISM_FAMILY,
        ),
        (
            HazardKind.SNOWSTORM,
            HazardOntologyRole.MECHANISM,
        ),
        (
            HazardKind.FREEZING_RAIN,
            HazardOntologyRole.MECHANISM,
        ),
        (
            HazardKind.ALL_HAZARD,
            HazardOntologyRole.FALLBACK,
        ),
    ),
)
def test_canonical_ontology_roles(
    registry: HazardRegistry,
    hazard: HazardKind,
    role: HazardOntologyRole,
) -> None:
    entry = registry.get_by_name(hazard)

    assert entry.ontology_role == role
    assert CANONICAL_HAZARD_ONTOLOGY_ROLES[hazard.value] == role


def test_role_group_constants() -> None:
    assert MECHANISM_FAMILY_HAZARDS == (
        HazardKind.FLOOD,
        HazardKind.WINTER_STORM,
    )

    assert MECHANISM_HAZARDS == (
        HazardKind.RIVERINE_FLOOD,
        HazardKind.PLUVIAL_FLOOD,
        HazardKind.HEAT,
        HazardKind.SNOWSTORM,
        HazardKind.FREEZING_RAIN,
    )

    assert DISRUPTION_HAZARDS == (
        HazardKind.OUTAGE,
        HazardKind.ROAD_DISRUPTION,
    )

    assert EVENT_FAMILY_HAZARDS == (
        HazardKind.CIVIL_SECURITY_EVENT,
    )


@pytest.mark.parametrize(
    ("hazard", "expected_parent"),
    (
        (HazardKind.FLOOD, None),
        (
            HazardKind.RIVERINE_FLOOD,
            HazardKind.FLOOD,
        ),
        (
            HazardKind.PLUVIAL_FLOOD,
            HazardKind.FLOOD,
        ),
        (HazardKind.HEAT, None),
        (HazardKind.OUTAGE, None),
        (HazardKind.ROAD_DISRUPTION, None),
        (HazardKind.CIVIL_SECURITY_EVENT, None),
        (HazardKind.WINTER_STORM, None),
        (
            HazardKind.SNOWSTORM,
            HazardKind.WINTER_STORM,
        ),
        (
            HazardKind.FREEZING_RAIN,
            HazardKind.WINTER_STORM,
        ),
        (HazardKind.ALL_HAZARD, None),
    ),
)
def test_canonical_parent_contracts(
    registry: HazardRegistry,
    hazard: HazardKind,
    expected_parent: HazardKind | None,
) -> None:
    entry = registry.get_by_name(hazard)

    assert entry.parent_hazard == expected_parent
    assert (
        CANONICAL_HAZARD_PARENTS[hazard.value]
        == (
            expected_parent.value
            if expected_parent is not None
            else None
        )
    )


def test_flood_children(
    registry: HazardRegistry,
) -> None:
    assert tuple(
        child.name
        for child in registry.children_of(HazardKind.FLOOD)
    ) == (
        HazardKind.RIVERINE_FLOOD,
        HazardKind.PLUVIAL_FLOOD,
    )


def test_winter_storm_children(
    registry: HazardRegistry,
) -> None:
    assert tuple(
        child.name
        for child in registry.children_of(
            HazardKind.WINTER_STORM
        )
    ) == (
        HazardKind.SNOWSTORM,
        HazardKind.FREEZING_RAIN,
    )


@pytest.mark.parametrize(
    ("hazard", "expected_ancestors"),
    (
        (
            HazardKind.RIVERINE_FLOOD,
            (HazardKind.FLOOD,),
        ),
        (
            HazardKind.PLUVIAL_FLOOD,
            (HazardKind.FLOOD,),
        ),
        (
            HazardKind.SNOWSTORM,
            (HazardKind.WINTER_STORM,),
        ),
        (
            HazardKind.FREEZING_RAIN,
            (HazardKind.WINTER_STORM,),
        ),
        (HazardKind.HEAT, ()),
        (HazardKind.OUTAGE, ()),
    ),
)
def test_ancestors(
    registry: HazardRegistry,
    hazard: HazardKind,
    expected_ancestors: tuple[HazardKind, ...],
) -> None:
    assert tuple(
        ancestor.name
        for ancestor in registry.ancestors_of(hazard)
    ) == expected_ancestors


def test_descendants(
    registry: HazardRegistry,
) -> None:
    assert tuple(
        descendant.name
        for descendant in registry.descendants_of(
            HazardKind.FLOOD
        )
    ) == (
        HazardKind.RIVERINE_FLOOD,
        HazardKind.PLUVIAL_FLOOD,
    )

    assert tuple(
        descendant.name
        for descendant in registry.descendants_of(
            HazardKind.WINTER_STORM
        )
    ) == (
        HazardKind.SNOWSTORM,
        HazardKind.FREEZING_RAIN,
    )


@pytest.mark.parametrize(
    ("hazard", "expected_root"),
    (
        (
            HazardKind.RIVERINE_FLOOD,
            HazardKind.FLOOD,
        ),
        (
            HazardKind.PLUVIAL_FLOOD,
            HazardKind.FLOOD,
        ),
        (
            HazardKind.SNOWSTORM,
            HazardKind.WINTER_STORM,
        ),
        (
            HazardKind.FREEZING_RAIN,
            HazardKind.WINTER_STORM,
        ),
        (
            HazardKind.CIVIL_SECURITY_EVENT,
            HazardKind.CIVIL_SECURITY_EVENT,
        ),
    ),
)
def test_root_of(
    registry: HazardRegistry,
    hazard: HazardKind,
    expected_root: HazardKind,
) -> None:
    assert registry.root_of(hazard).name == expected_root


def test_root_entries(
    registry: HazardRegistry,
) -> None:
    root_names = {
        entry.name
        for entry in registry.root_entries
    }

    assert HazardKind.FLOOD in root_names
    assert HazardKind.WINTER_STORM in root_names
    assert HazardKind.HEAT in root_names
    assert HazardKind.OUTAGE in root_names
    assert HazardKind.CIVIL_SECURITY_EVENT in root_names

    assert HazardKind.PLUVIAL_FLOOD not in root_names
    assert HazardKind.SNOWSTORM not in root_names


# =============================================================================
# Queryability and support
# =============================================================================


def test_queryable_hazards_match_constant(
    registry: HazardRegistry,
) -> None:
    assert registry.queryable_hazard_kinds == QUERYABLE_HAZARDS
    assert registry.runtime_hazard_kinds == QUERYABLE_HAZARDS


def test_canonical_queryability_mapping(
    registry: HazardRegistry,
) -> None:
    for hazard in HazardKind:
        entry = registry.get_by_name(hazard)

        assert (
            entry.query_allowed
            == CANONICAL_HAZARD_QUERY_ALLOWED[
                hazard.value
            ]
        )


def test_compatibility_hazard_collections() -> None:
    assert RUNTIME_HAZARDS == QUERYABLE_HAZARDS
    assert RUNTIME_PRIOR_HAZARDS == QUERYABLE_HAZARDS
    assert ALL_PRIOR_HAZARDS == ALL_HAZARDS
    assert FALLBACK_PRIOR_HAZARD == FALLBACK_HAZARD


@pytest.mark.parametrize("hazard", QUERYABLE_HAZARDS)
def test_all_queryable_hazards_are_allowed(
    registry: HazardRegistry,
    hazard: HazardKind,
) -> None:
    entry = registry.assert_queryable_hazard(hazard)

    assert entry.name == hazard
    assert entry.query_allowed
    assert not entry.fallback_only


def test_all_hazard_is_fallback_only(
    registry: HazardRegistry,
) -> None:
    entry = registry.get_by_name(
        HazardKind.ALL_HAZARD
    )

    assert entry.fallback_only
    assert not entry.query_allowed
    assert (
        entry.support_state
        == HazardSupportState.FALLBACK_ONLY
    )

    with pytest.raises(
        ValueError,
        match="fallback-only",
    ):
        registry.assert_queryable_hazard(
            HazardKind.ALL_HAZARD
        )

    assert (
        registry.assert_queryable_hazard(
            HazardKind.ALL_HAZARD,
            allow_fallback=True,
        )
        is entry
    )


def test_project_support_summary(
    registry: HazardRegistry,
) -> None:
    assert (
        _entry(
            registry,
            HazardKind.FLOOD,
        ).support_state
        == HazardSupportState.PARTIALLY_DATA_BACKED
    )

    assert (
        _entry(
            registry,
            HazardKind.PLUVIAL_FLOOD,
        ).support_state
        == HazardSupportState.PARTIALLY_DATA_BACKED
    )

    assert (
        _entry(
            registry,
            HazardKind.RIVERINE_FLOOD,
        ).support_state
        == HazardSupportState.PLANNED
    )

    assert (
        _entry(
            registry,
            HazardKind.CIVIL_SECURITY_EVENT,
        ).support_state
        == HazardSupportState.PARTIALLY_DATA_BACKED
    )

    assert (
        _entry(
            registry,
            HazardKind.WINTER_STORM,
        ).support_state
        == HazardSupportState.PLANNED
    )

    assert (
        _entry(
            registry,
            HazardKind.SNOWSTORM,
        ).support_state
        == HazardSupportState.PLANNED
    )

    assert (
        _entry(
            registry,
            HazardKind.FREEZING_RAIN,
        ).support_state
        == HazardSupportState.PLANNED
    )


def test_partially_backed_hazard_requires_explicit_training_override(
    registry: HazardRegistry,
) -> None:
    with pytest.raises(
        ValueError,
        match="not approved",
    ):
        registry.assert_training_supported_hazard(
            HazardKind.PLUVIAL_FLOOD
        )

    entry = registry.assert_training_supported_hazard(
        HazardKind.PLUVIAL_FLOOD,
        allow_partially_data_backed=True,
    )

    assert entry.name == HazardKind.PLUVIAL_FLOOD


def test_planned_hazard_is_not_training_supported(
    registry: HazardRegistry,
) -> None:
    with pytest.raises(
        ValueError,
        match="not approved",
    ):
        registry.assert_training_supported_hazard(
            HazardKind.FREEZING_RAIN,
            allow_partially_data_backed=True,
        )


def test_fallback_hazard_is_never_training_supported(
    registry: HazardRegistry,
) -> None:
    with pytest.raises(ValueError):
        registry.assert_training_supported_hazard(
            HazardKind.ALL_HAZARD,
            allow_partially_data_backed=True,
        )


# =============================================================================
# Lookup behavior
# =============================================================================


@pytest.mark.parametrize("hazard", tuple(HazardKind))
def test_lookup_by_enum_and_string(
    registry: HazardRegistry,
    hazard: HazardKind,
) -> None:
    by_enum = registry.get_by_name(hazard)
    by_string = registry.get_by_name(hazard.value)

    assert by_enum is by_string


@pytest.mark.parametrize("hazard", tuple(HazardKind))
def test_stable_id_round_trip(
    registry: HazardRegistry,
    hazard: HazardKind,
) -> None:
    stable_id = registry.stable_id_for(hazard)

    assert (
        registry.name_for_stable_id(stable_id)
        == hazard.value
    )
    assert (
        registry.get_by_stable_id(stable_id).name
        == hazard
    )


def test_unknown_name_raises(
    registry: HazardRegistry,
) -> None:
    with pytest.raises(ValueError):
        registry.get_by_name("not_a_hazard")


def test_unknown_stable_id_raises(
    registry: HazardRegistry,
) -> None:
    with pytest.raises(
        KeyError,
        match="Unknown canonical stable hazard ID",
    ):
        registry.get_by_stable_id(123_456)


def test_contains_supports_enum_string_and_id(
    registry: HazardRegistry,
) -> None:
    assert HazardKind.HEAT in registry
    assert HazardKind.HEAT.value in registry
    assert HAZARD_ID_HEAT in registry

    assert "not_a_hazard" not in registry
    assert 123_456 not in registry
    assert True not in registry


def test_normalization_helpers() -> None:
    assert (
        normalize_hazard_kind("pluvial_flood")
        == HazardKind.PLUVIAL_FLOOD
    )
    assert (
        normalize_hazard_kind(
            HazardKind.FREEZING_RAIN
        )
        == HazardKind.FREEZING_RAIN
    )
    assert (
        normalize_hazard_name(
            HazardKind.WINTER_STORM
        )
        == "winter_storm"
    )

    with pytest.raises(ValueError):
        normalize_hazard_name("not_a_hazard")


# =============================================================================
# Entry validation
# =============================================================================


def test_query_allowed_must_be_boolean(
    registry: HazardRegistry,
) -> None:
    flood = registry.get_by_name(HazardKind.FLOOD)

    with pytest.raises(
        TypeError,
        match="query_allowed must be a Boolean",
    ):
        replace(
            flood,
            query_allowed=1,  # type: ignore[arg-type]
        )


def test_tags_are_normalized_to_tuple(
    registry: HazardRegistry,
) -> None:
    flood = registry.get_by_name(HazardKind.FLOOD)

    reconstructed = HazardRegistryEntry(
        stable_hazard_id=flood.stable_hazard_id,
        name=flood.name,
        display_name=flood.display_name,
        description=flood.description,
        ontology_role=flood.ontology_role,
        parent_hazard=flood.parent_hazard,
        query_allowed=flood.query_allowed,
        support_state=flood.support_state,
        tags=["one", "two"],  # type: ignore[arg-type]
    )

    assert reconstructed.tags == ("one", "two")
    assert isinstance(reconstructed.tags, tuple)


def test_duplicate_tags_are_rejected(
    registry: HazardRegistry,
) -> None:
    flood = registry.get_by_name(HazardKind.FLOOD)

    with pytest.raises(
        ValueError,
        match="duplicate",
    ):
        replace(
            flood,
            tags=("duplicate", "duplicate"),
        )


def test_wrong_stable_id_is_rejected(
    registry: HazardRegistry,
) -> None:
    flood = registry.get_by_name(HazardKind.FLOOD)

    with pytest.raises(
        ValueError,
        match="must use stable ID",
    ):
        replace(
            flood,
            stable_hazard_id=999,
        )


def test_wrong_display_name_is_rejected(
    registry: HazardRegistry,
) -> None:
    flood = registry.get_by_name(HazardKind.FLOOD)

    with pytest.raises(
        ValueError,
        match="must use display name",
    ):
        replace(
            flood,
            display_name="Incorrect",
        )


def test_wrong_ontology_role_is_rejected(
    registry: HazardRegistry,
) -> None:
    flood = registry.get_by_name(HazardKind.FLOOD)

    with pytest.raises(
        ValueError,
        match="must use ontology role",
    ):
        replace(
            flood,
            ontology_role=HazardOntologyRole.MECHANISM,
        )


def test_wrong_parent_is_rejected(
    registry: HazardRegistry,
) -> None:
    pluvial = registry.get_by_name(
        HazardKind.PLUVIAL_FLOOD
    )

    with pytest.raises(
        ValueError,
        match="must use parent",
    ):
        replace(
            pluvial,
            parent_hazard=HazardKind.WINTER_STORM,
        )


def test_self_parent_is_rejected(
    registry: HazardRegistry,
) -> None:
    pluvial = registry.get_by_name(
        HazardKind.PLUVIAL_FLOOD
    )

    with pytest.raises(ValueError):
        replace(
            pluvial,
            parent_hazard=HazardKind.PLUVIAL_FLOOD,
        )


def test_fallback_cannot_be_queryable(
    registry: HazardRegistry,
) -> None:
    fallback = registry.get_by_name(
        HazardKind.ALL_HAZARD
    )

    with pytest.raises(ValueError):
        replace(
            fallback,
            query_allowed=True,
        )


def test_fallback_requires_fallback_support_state(
    registry: HazardRegistry,
) -> None:
    fallback = registry.get_by_name(
        HazardKind.ALL_HAZARD
    )

    with pytest.raises(
        ValueError,
        match="Fallback hazards must use support state",
    ):
        replace(
            fallback,
            support_state=HazardSupportState.PLANNED,
        )


def test_nonfallback_cannot_use_fallback_support_state(
    registry: HazardRegistry,
) -> None:
    heat = registry.get_by_name(HazardKind.HEAT)

    with pytest.raises(
        ValueError,
        match="Only fallback hazards",
    ):
        replace(
            heat,
            support_state=(
                HazardSupportState.FALLBACK_ONLY
            ),
        )


def test_deprecated_hazard_cannot_remain_queryable(
    registry: HazardRegistry,
) -> None:
    heat = registry.get_by_name(HazardKind.HEAT)

    with pytest.raises(
        ValueError,
        match="Deprecated hazards cannot remain queryable",
    ):
        replace(
            heat,
            support_state=HazardSupportState.DEPRECATED,
        )


# =============================================================================
# Registry structural validation
# =============================================================================


def test_registry_entries_sequence_is_normalized_to_tuple(
    registry: HazardRegistry,
) -> None:
    reconstructed = HazardRegistry(
        entries=list(registry.entries),  # type: ignore[arg-type]
    )

    assert isinstance(reconstructed.entries, tuple)
    assert reconstructed.hazard_names == registry.hazard_names


def test_duplicate_entry_is_rejected(
    registry: HazardRegistry,
) -> None:
    duplicate_entries = (
        *registry.entries,
        registry.entries[0],
    )

    with pytest.raises(ValueError):
        HazardRegistry(entries=duplicate_entries)


def test_missing_hazard_is_rejected(
    registry: HazardRegistry,
) -> None:
    incomplete = tuple(
        entry
        for entry in registry.entries
        if entry.name != HazardKind.SNOWSTORM
    )

    with pytest.raises(
        ValueError,
        match="must cover the complete current vocabulary",
    ):
        HazardRegistry(entries=incomplete)


def test_registry_contains_exactly_one_fallback(
    registry: HazardRegistry,
) -> None:
    fallback_entries = tuple(
        entry
        for entry in registry.entries
        if entry.fallback_only
    )

    assert len(fallback_entries) == 1
    assert (
        fallback_entries[0].name
        == HazardKind.ALL_HAZARD
    )


# =============================================================================
# Fingerprint layering
# =============================================================================


def test_support_change_preserves_semantic_fingerprints(
    registry: HazardRegistry,
) -> None:
    modified = _replace_registry_entry(
        registry,
        HazardKind.FLOOD,
        support_state=HazardSupportState.DATA_BACKED,
    )

    assert (
        modified.semantic_content_fingerprint()
        == registry.semantic_content_fingerprint()
    )
    assert (
        modified.versioned_semantic_fingerprint()
        == registry.versioned_semantic_fingerprint()
    )

    assert (
        modified.operational_fingerprint()
        != registry.operational_fingerprint()
    )
    assert (
        modified.snapshot_fingerprint()
        != registry.snapshot_fingerprint()
    )


def test_description_and_tag_changes_only_affect_snapshot(
    registry: HazardRegistry,
) -> None:
    flood = registry.get_by_name(HazardKind.FLOOD)

    modified = _replace_registry_entry(
        registry,
        HazardKind.FLOOD,
        description=(
            f"{flood.description} Editorial clarification."
        ),
        tags=(*flood.tags, "editorial_test"),
    )

    assert (
        modified.semantic_content_fingerprint()
        == registry.semantic_content_fingerprint()
    )
    assert (
        modified.versioned_semantic_fingerprint()
        == registry.versioned_semantic_fingerprint()
    )
    assert (
        modified.operational_fingerprint()
        == registry.operational_fingerprint()
    )
    assert (
        modified.snapshot_fingerprint()
        != registry.snapshot_fingerprint()
    )


def test_version_change_preserves_content_but_changes_compatibility(
    registry: HazardRegistry,
) -> None:
    modified = replace(
        registry,
        registry_version="0.2",
        schema_version="0.2",
    )

    assert (
        modified.semantic_content_fingerprint()
        == registry.semantic_content_fingerprint()
    )
    assert (
        modified.versioned_semantic_fingerprint()
        != registry.versioned_semantic_fingerprint()
    )
    assert (
        modified.compatibility_fingerprint()
        != registry.compatibility_fingerprint()
    )


def test_registry_name_change_alters_semantic_content(
    registry: HazardRegistry,
) -> None:
    modified = replace(
        registry,
        registry_name="different_registry_name",
    )

    assert (
        modified.semantic_content_fingerprint()
        != registry.semantic_content_fingerprint()
    )


def test_default_fingerprint_is_strict_compatibility_fingerprint(
    registry: HazardRegistry,
) -> None:
    assert (
        registry.fingerprint()
        == registry.compatibility_fingerprint()
    )
    assert (
        registry.semantic_fingerprint()
        == registry.semantic_content_fingerprint()
    )


# =============================================================================
# Serialization
# =============================================================================


def test_json_serialization_round_trip(
    registry: HazardRegistry,
) -> None:
    payload = json.loads(
        json.dumps(registry.to_dict())
    )

    reconstructed = HazardRegistry.from_dict(
        payload,
        require_current_version=True,
    )

    assert reconstructed == registry
    assert (
        reconstructed.semantic_content_fingerprint()
        == registry.semantic_content_fingerprint()
    )
    assert (
        reconstructed.versioned_semantic_fingerprint()
        == registry.versioned_semantic_fingerprint()
    )
    assert (
        reconstructed.operational_fingerprint()
        == registry.operational_fingerprint()
    )
    assert (
        reconstructed.snapshot_fingerprint()
        == registry.snapshot_fingerprint()
    )


def test_unordered_serialized_entries_are_canonicalized(
    registry: HazardRegistry,
) -> None:
    payload = deepcopy(registry.to_dict())
    payload["entries"] = list(
        reversed(payload["entries"])
    )

    reconstructed = HazardRegistry.from_dict(
        payload,
        require_current_version=True,
    )

    assert reconstructed.hazard_names == registry.hazard_names
    assert (
        reconstructed.fingerprint()
        == registry.fingerprint()
    )


@pytest.mark.parametrize(
    "fingerprint_field",
    (
        "semantic_content_fingerprint",
        "versioned_semantic_fingerprint",
        "operational_fingerprint",
        "snapshot_fingerprint",
    ),
)
def test_tampered_serialized_fingerprint_is_rejected(
    registry: HazardRegistry,
    fingerprint_field: str,
) -> None:
    payload = deepcopy(registry.to_dict())
    payload[fingerprint_field] = "tampered"

    with pytest.raises(
        ValueError,
        match="fingerprint",
    ):
        HazardRegistry.from_dict(payload)


def test_unknown_registry_field_is_rejected(
    registry: HazardRegistry,
) -> None:
    payload = deepcopy(registry.to_dict())
    payload["unexpected_field"] = "unexpected"

    with pytest.raises(
        ValueError,
        match="Unknown HazardRegistry fields",
    ):
        HazardRegistry.from_dict(payload)


def test_unknown_entry_field_is_rejected(
    registry: HazardRegistry,
) -> None:
    payload = deepcopy(registry.to_dict())
    payload["entries"][0]["unexpected_field"] = "unexpected"

    with pytest.raises(
        ValueError,
        match="Unknown fields for HazardRegistryEntry",
    ):
        HazardRegistry.from_dict(payload)


def test_missing_serialized_entry_is_rejected(
    registry: HazardRegistry,
) -> None:
    payload = deepcopy(registry.to_dict())
    payload["entries"] = payload["entries"][:-1]

    # Remove fingerprints so validation reaches the structural error first.
    payload.pop("semantic_content_fingerprint")
    payload.pop("versioned_semantic_fingerprint")
    payload.pop("operational_fingerprint")
    payload.pop("snapshot_fingerprint")

    with pytest.raises(
        ValueError,
        match="must cover the complete current vocabulary",
    ):
        HazardRegistry.from_dict(payload)


def test_legacy_entry_field_names_are_supported(
    registry: HazardRegistry,
) -> None:
    flood = registry.get_by_name(HazardKind.FLOOD)
    payload = flood.to_dict()

    payload["runtime_allowed"] = payload.pop(
        "query_allowed"
    )
    payload["fallback_only"] = flood.fallback_only
    payload["parent_hazard_name"] = payload.pop(
        "parent_hazard"
    )

    reconstructed = HazardRegistryEntry.from_dict(
        payload
    )

    assert reconstructed == flood


def test_legacy_fallback_flag_mismatch_is_rejected(
    registry: HazardRegistry,
) -> None:
    flood = registry.get_by_name(HazardKind.FLOOD)
    payload = flood.to_dict()
    payload["fallback_only"] = True

    with pytest.raises(
        ValueError,
        match="fallback_only disagrees",
    ):
        HazardRegistryEntry.from_dict(payload)


def test_legacy_semantic_fingerprint_is_accepted(
    registry: HazardRegistry,
) -> None:
    payload = deepcopy(registry.to_dict())
    payload["semantic_fingerprint"] = (
        registry.semantic_fingerprint()
    )

    reconstructed = HazardRegistry.from_dict(
        payload,
        require_current_version=True,
    )

    assert reconstructed == registry


# =============================================================================
# Version compatibility
# =============================================================================


def test_older_version_labels_load_when_ontology_is_compatible(
    registry: HazardRegistry,
) -> None:
    older = replace(
        registry,
        registry_version="0.2",
        schema_version="0.2",
    )

    payload = older.to_dict()

    reconstructed = HazardRegistry.from_dict(
        payload,
        require_current_version=False,
    )

    assert reconstructed.registry_version == "0.2"
    assert reconstructed.schema_version == "0.2"
    assert (
        reconstructed.semantic_content_fingerprint()
        == registry.semantic_content_fingerprint()
    )


def test_older_version_labels_fail_current_compatibility(
    registry: HazardRegistry,
) -> None:
    older = replace(
        registry,
        registry_version="0.2",
        schema_version="0.2",
    )

    with pytest.raises(
        ValueError,
        match="version is not current",
    ):
        older.assert_current_compatibility()

    with pytest.raises(
        ValueError,
        match="version is not current",
    ):
        HazardRegistry.from_dict(
            older.to_dict(),
            require_current_version=True,
        )


def test_genuinely_different_historical_vocabulary_is_not_loaded(
    registry: HazardRegistry,
) -> None:
    payload = deepcopy(registry.to_dict())
    payload["registry_version"] = "0.1"
    payload["schema_version"] = "0.1"
    payload["entries"] = [
        entry
        for entry in payload["entries"]
        if entry["name"] != HazardKind.FREEZING_RAIN.value
    ]

    for field_name in (
        "semantic_content_fingerprint",
        "versioned_semantic_fingerprint",
        "operational_fingerprint",
        "snapshot_fingerprint",
    ):
        payload.pop(field_name)

    with pytest.raises(
        ValueError,
        match="must cover the complete current vocabulary",
    ):
        HazardRegistry.from_dict(
            payload,
            require_current_version=False,
        )


# =============================================================================
# Immutability
# =============================================================================


def test_entry_is_frozen(
    registry: HazardRegistry,
) -> None:
    entry = registry.get_by_name(HazardKind.FLOOD)

    with pytest.raises(FrozenInstanceError):
        entry.description = "mutated"  # type: ignore[misc]


def test_registry_is_frozen(
    registry: HazardRegistry,
) -> None:
    with pytest.raises(FrozenInstanceError):
        registry.registry_name = "mutated"  # type: ignore[misc]


def test_lookup_mappings_are_read_only(
    registry: HazardRegistry,
) -> None:
    with pytest.raises(TypeError):
        registry.by_name["new"] = registry.entries[0]  # type: ignore[index]

    with pytest.raises(TypeError):
        registry.by_stable_id[999] = registry.entries[0]  # type: ignore[index]


def test_children_results_are_immutable_tuples(
    registry: HazardRegistry,
) -> None:
    children = registry.children_of(
        HazardKind.FLOOD
    )

    assert isinstance(children, tuple)

    with pytest.raises(AttributeError):
        children.append(  # type: ignore[attr-defined]
            registry.get_by_name(HazardKind.HEAT)
        )