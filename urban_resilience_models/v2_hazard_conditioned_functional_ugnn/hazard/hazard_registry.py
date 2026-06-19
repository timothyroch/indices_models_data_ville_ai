"""
Canonical hazard ontology for the V2 functional UGNN.

Target path:
    urban_resilience_models/
        v2_hazard_conditioned_functional_ugnn/
            hazard/
                hazard_registry.py

Python requirement:
    Python >= 3.11

This module is the single authority for:

- canonical hazard names;
- stable, sparse hazard IDs;
- display names;
- hazard ontology roles;
- parent-child hazard relationships;
- queryability by the model architecture;
- current project-level data-support summaries;
- semantic, compatibility, operational, and snapshot fingerprints;
- immutable registry serialization and validation.

It does not own:

- hazard embeddings;
- hazard-relation priors;
- task-specific target definitions;
- task-specific data-support approval;
- dynamic urban state;
- antecedent weather-history encoding;
- scenario-feature encoding;
- model-training policy.

Dependency direction
--------------------
Both hazard embeddings and hazard-relation priors depend on this neutral
registry:

    hazard_registry.py
        ↑               ↑
        │               │
    hazard_embeddings.py
    hazard_relation_priors.py

Neither dependent module may redefine hazard names, stable IDs, ontology roles,
display names, hierarchy, or fallback semantics.

Hazard hierarchy
----------------
The current ontology includes mechanism families and their subtypes:

    flood
    ├── riverine_flood
    └── pluvial_flood

    winter_storm
    ├── snowstorm
    └── freezing_rain

``pluvial_flood`` refers to flooding caused by intense rainfall, surface
runoff, insufficient drainage capacity, and ponding in low-lying areas or
topographic depressions. Water may travel downslope before accumulating; the
hazard is not defined as water remaining on the slope itself.

Overlapping ontology roles
--------------------------
Not every category is a mutually exclusive physical mechanism.

For example:

- ``freezing_rain`` is a physical hazard mechanism;
- ``outage`` is a resulting infrastructure disruption;
- ``road_disruption`` is a transport disruption;
- ``civil_security_event`` is a broad event family that may include several
  mechanisms and disruptions simultaneously.

A future multi-label hazard query may therefore represent an event as:

    mechanism:
        freezing_rain

    disruptions:
        outage
        road_disruption

    event family:
        civil_security_event

V2 may still select one primary hazard, but the ontology does not falsely claim
that all hazard categories are mutually exclusive.

Stable IDs versus dense neural indices
--------------------------------------
``stable_hazard_id`` is a durable semantic identity. Stable IDs are sparse and
must never be used directly as neural embedding-table indices.

Dense indices are created later by ``hazard_embeddings.py``:

    hazard_index = 0, 1, ..., H - 1

Persisted artifacts should preserve:

    hazard_index -> stable_hazard_id -> canonical hazard name

Queryability versus data support
--------------------------------
Two distinct questions are represented:

1. Can the architecture represent and query this hazard?
2. Is the present project data pipeline sufficiently mature to train or
   evaluate it as an ordinary supervised task?

A planned hazard may be queryable for:

- architecture tests;
- ontology experiments;
- explicitly controlled counterfactuals.

That does not make it approved for publication-grade supervised training.

The support state in this registry is only a project-level summary. The
long-term design should place definitive support claims in a scoped manifest
conditioned on:

- hazard;
- target;
- dataset;
- region;
- geographic scale;
- forecast horizon;
- validation protocol.

ALL_HAZARD and unknown hazards
------------------------------
``all_hazard`` is an intentional fallback category. It is not a default model
query.

The synthetic unknown-hazard name and stable ID are reserved here so all
modules use the same values. They are excluded from the semantic registry:

    unknown input != canonical hazard
    unknown input != intentional all-hazard query

Historical loading
------------------
Non-current version strings may be loaded when their contents remain
structurally compatible with the present ontology.

This module does not claim to load arbitrary historical ontologies containing
renamed hazards, retired IDs, or different category sets. Such snapshots
require explicit migration logic.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field, fields
from enum import StrEnum
from hashlib import sha256
import json
from types import MappingProxyType
from typing import Any, Final, Iterator, Mapping, Sequence


# =============================================================================
# Registry identity
# =============================================================================


HAZARD_REGISTRY_SCHEMA_VERSION: Final[str] = "0.3"
HAZARD_REGISTRY_VERSION: Final[str] = "0.3"

DEFAULT_HAZARD_REGISTRY_NAME: Final[str] = (
    "v2_canonical_hazard_registry"
)


# =============================================================================
# Canonical vocabulary
# =============================================================================


class HazardKind(StrEnum):
    """Canonical hazard names used throughout the V2 package."""

    # Hydrological mechanism family and subtypes.
    FLOOD = "flood"
    RIVERINE_FLOOD = "riverine_flood"
    PLUVIAL_FLOOD = "pluvial_flood"

    # Thermal mechanism.
    HEAT = "heat"

    # System disruptions.
    OUTAGE = "outage"
    ROAD_DISRUPTION = "road_disruption"

    # Broad administrative/event family.
    CIVIL_SECURITY_EVENT = "civil_security_event"

    # Winter-weather mechanism family and subtypes.
    WINTER_STORM = "winter_storm"
    SNOWSTORM = "snowstorm"
    FREEZING_RAIN = "freezing_rain"

    # Semantic fallback, not a default runtime query.
    ALL_HAZARD = "all_hazard"


class HazardOntologyRole(StrEnum):
    """
    Ontological role played by a hazard category.

    MECHANISM_FAMILY
        Parent category grouping related physical mechanisms.

    MECHANISM
        Physical or environmental forcing mechanism.

    DISRUPTION
        Resulting loss or degradation of an urban system.

    EVENT_FAMILY
        Broad event or administrative category that may overlap with several
        mechanisms and disruptions.

    FALLBACK
        Generic semantic fallback used only when explicitly requested.
    """

    MECHANISM_FAMILY = "mechanism_family"
    MECHANISM = "mechanism"
    DISRUPTION = "disruption"
    EVENT_FAMILY = "event_family"
    FALLBACK = "fallback"


class HazardSupportState(StrEnum):
    """
    Current project-level operational support.

    DATA_BACKED
        A sufficiently frozen target, dataset, horizon, geography, and
        validation pipeline currently exists.

    PARTIALLY_DATA_BACKED
        Relevant data and proxy targets exist, but the complete direct hazard
        task or validation interpretation is not yet fully frozen.

    PLANNED
        The category belongs to the intended ontology but does not currently
        have a complete operational supervised pipeline.

    FALLBACK_ONLY
        The category exists only for semantic fallback or controlled generic
        experiments.

    DEPRECATED
        The category remains loadable for compatible historical artifacts but
        must not be used in new queries.
    """

    DATA_BACKED = "data_backed"
    PARTIALLY_DATA_BACKED = "partially_data_backed"
    PLANNED = "planned"
    FALLBACK_ONLY = "fallback_only"
    DEPRECATED = "deprecated"


CANONICAL_HAZARD_SUPPORT_STATES: Final[tuple[str, ...]] = tuple(
    state.value
    for state in HazardSupportState
)


# =============================================================================
# Stable hazard IDs
#
# IDs are sparse and grouped by family.
# Never reuse a retired or deprecated ID.
# =============================================================================


HAZARD_ID_FLOOD: Final[int] = 100
HAZARD_ID_RIVERINE_FLOOD: Final[int] = 110
HAZARD_ID_PLUVIAL_FLOOD: Final[int] = 120

HAZARD_ID_HEAT: Final[int] = 200

HAZARD_ID_OUTAGE: Final[int] = 300

HAZARD_ID_ROAD_DISRUPTION: Final[int] = 400

HAZARD_ID_CIVIL_SECURITY_EVENT: Final[int] = 500

HAZARD_ID_WINTER_STORM: Final[int] = 600
HAZARD_ID_SNOWSTORM: Final[int] = 610
HAZARD_ID_FREEZING_RAIN: Final[int] = 620

HAZARD_ID_ALL_HAZARD: Final[int] = 900


# Synthetic non-semantic identity for optional unknown-input handling.
UNKNOWN_HAZARD_NAME: Final[str] = "__unknown_hazard__"
UNKNOWN_HAZARD_DISPLAY_NAME: Final[str] = "Unknown hazard"
HAZARD_ID_UNKNOWN: Final[int] = 999_999


# =============================================================================
# Canonical semantic contracts
#
# These mappings define immutable semantic properties. Operational support
# states are intentionally not included here.
# =============================================================================


CANONICAL_HAZARD_STABLE_IDS: Final[Mapping[str, int]] = MappingProxyType(
    {
        HazardKind.FLOOD.value: HAZARD_ID_FLOOD,
        HazardKind.RIVERINE_FLOOD.value: HAZARD_ID_RIVERINE_FLOOD,
        HazardKind.PLUVIAL_FLOOD.value: HAZARD_ID_PLUVIAL_FLOOD,
        HazardKind.HEAT.value: HAZARD_ID_HEAT,
        HazardKind.OUTAGE.value: HAZARD_ID_OUTAGE,
        HazardKind.ROAD_DISRUPTION.value: HAZARD_ID_ROAD_DISRUPTION,
        HazardKind.CIVIL_SECURITY_EVENT.value: (
            HAZARD_ID_CIVIL_SECURITY_EVENT
        ),
        HazardKind.WINTER_STORM.value: HAZARD_ID_WINTER_STORM,
        HazardKind.SNOWSTORM.value: HAZARD_ID_SNOWSTORM,
        HazardKind.FREEZING_RAIN.value: HAZARD_ID_FREEZING_RAIN,
        HazardKind.ALL_HAZARD.value: HAZARD_ID_ALL_HAZARD,
    }
)


CANONICAL_HAZARD_DISPLAY_NAMES: Final[Mapping[str, str]] = (
    MappingProxyType(
        {
            HazardKind.FLOOD.value: "Flood",
            HazardKind.RIVERINE_FLOOD.value: "Riverine flood",
            HazardKind.PLUVIAL_FLOOD.value: "Pluvial flood",
            HazardKind.HEAT.value: "Heat",
            HazardKind.OUTAGE.value: "Outage",
            HazardKind.ROAD_DISRUPTION.value: "Road disruption",
            HazardKind.CIVIL_SECURITY_EVENT.value: (
                "Civil-security event"
            ),
            HazardKind.WINTER_STORM.value: "Winter storm",
            HazardKind.SNOWSTORM.value: "Snowstorm",
            HazardKind.FREEZING_RAIN.value: (
                "Freezing rain / ice storm"
            ),
            HazardKind.ALL_HAZARD.value: "All hazard",
        }
    )
)


CANONICAL_HAZARD_ONTOLOGY_ROLES: Final[
    Mapping[str, HazardOntologyRole]
] = MappingProxyType(
    {
        HazardKind.FLOOD.value: (
            HazardOntologyRole.MECHANISM_FAMILY
        ),
        HazardKind.RIVERINE_FLOOD.value: (
            HazardOntologyRole.MECHANISM
        ),
        HazardKind.PLUVIAL_FLOOD.value: (
            HazardOntologyRole.MECHANISM
        ),
        HazardKind.HEAT.value: (
            HazardOntologyRole.MECHANISM
        ),
        HazardKind.OUTAGE.value: (
            HazardOntologyRole.DISRUPTION
        ),
        HazardKind.ROAD_DISRUPTION.value: (
            HazardOntologyRole.DISRUPTION
        ),
        HazardKind.CIVIL_SECURITY_EVENT.value: (
            HazardOntologyRole.EVENT_FAMILY
        ),
        HazardKind.WINTER_STORM.value: (
            HazardOntologyRole.MECHANISM_FAMILY
        ),
        HazardKind.SNOWSTORM.value: (
            HazardOntologyRole.MECHANISM
        ),
        HazardKind.FREEZING_RAIN.value: (
            HazardOntologyRole.MECHANISM
        ),
        HazardKind.ALL_HAZARD.value: (
            HazardOntologyRole.FALLBACK
        ),
    }
)


CANONICAL_HAZARD_PARENTS: Final[
    Mapping[str, str | None]
] = MappingProxyType(
    {
        HazardKind.FLOOD.value: None,
        HazardKind.RIVERINE_FLOOD.value: HazardKind.FLOOD.value,
        HazardKind.PLUVIAL_FLOOD.value: HazardKind.FLOOD.value,
        HazardKind.HEAT.value: None,
        HazardKind.OUTAGE.value: None,
        HazardKind.ROAD_DISRUPTION.value: None,
        HazardKind.CIVIL_SECURITY_EVENT.value: None,
        HazardKind.WINTER_STORM.value: None,
        HazardKind.SNOWSTORM.value: HazardKind.WINTER_STORM.value,
        HazardKind.FREEZING_RAIN.value: HazardKind.WINTER_STORM.value,
        HazardKind.ALL_HAZARD.value: None,
    }
)


CANONICAL_HAZARD_QUERY_ALLOWED: Final[Mapping[str, bool]] = (
    MappingProxyType(
        {
            HazardKind.FLOOD.value: True,
            HazardKind.RIVERINE_FLOOD.value: True,
            HazardKind.PLUVIAL_FLOOD.value: True,
            HazardKind.HEAT.value: True,
            HazardKind.OUTAGE.value: True,
            HazardKind.ROAD_DISRUPTION.value: True,
            HazardKind.CIVIL_SECURITY_EVENT.value: True,
            HazardKind.WINTER_STORM.value: True,
            HazardKind.SNOWSTORM.value: True,
            HazardKind.FREEZING_RAIN.value: True,
            HazardKind.ALL_HAZARD.value: False,
        }
    )
)


QUERYABLE_HAZARDS: Final[tuple[HazardKind, ...]] = (
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
)

FALLBACK_HAZARD: Final[HazardKind] = HazardKind.ALL_HAZARD

ALL_HAZARDS: Final[tuple[HazardKind, ...]] = (
    *QUERYABLE_HAZARDS,
    FALLBACK_HAZARD,
)

MECHANISM_FAMILY_HAZARDS: Final[tuple[HazardKind, ...]] = (
    HazardKind.FLOOD,
    HazardKind.WINTER_STORM,
)

MECHANISM_HAZARDS: Final[tuple[HazardKind, ...]] = (
    HazardKind.RIVERINE_FLOOD,
    HazardKind.PLUVIAL_FLOOD,
    HazardKind.HEAT,
    HazardKind.SNOWSTORM,
    HazardKind.FREEZING_RAIN,
)

DISRUPTION_HAZARDS: Final[tuple[HazardKind, ...]] = (
    HazardKind.OUTAGE,
    HazardKind.ROAD_DISRUPTION,
)

EVENT_FAMILY_HAZARDS: Final[tuple[HazardKind, ...]] = (
    HazardKind.CIVIL_SECURITY_EVENT,
)


# Transitional compatibility aliases. New code should use the neutral names.
RUNTIME_HAZARDS: Final[tuple[HazardKind, ...]] = QUERYABLE_HAZARDS
RUNTIME_PRIOR_HAZARDS: Final[tuple[HazardKind, ...]] = QUERYABLE_HAZARDS
FALLBACK_PRIOR_HAZARD: Final[HazardKind] = FALLBACK_HAZARD
ALL_PRIOR_HAZARDS: Final[tuple[HazardKind, ...]] = ALL_HAZARDS


# =============================================================================
# Generic helpers
# =============================================================================


def _require_nonempty_string(name: str, value: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(
            f"{name} must be a non-empty string."
        )


def _require_nonnegative_int(name: str, value: int) -> None:
    if (
        isinstance(value, bool)
        or not isinstance(value, int)
        or value < 0
    ):
        raise ValueError(
            f"{name} must be a nonnegative integer."
        )


def _require_boolean(name: str, value: bool) -> None:
    if not isinstance(value, bool):
        raise TypeError(
            f"{name} must be a Boolean."
        )


def _require_unique_strings(
    name: str,
    values: Sequence[str],
) -> None:
    for index, value in enumerate(values):
        _require_nonempty_string(
            f"{name}[{index}]",
            value,
        )

    duplicates = sorted(
        value
        for value, count in Counter(values).items()
        if count > 1
    )

    if duplicates:
        raise ValueError(
            f"{name} contains duplicate values: {duplicates}."
        )


def _require_mapping(
    name: str,
    value: Any,
) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(
            f"{name} must be a mapping."
        )

    return value


def _reject_unknown_fields(
    object_type: type[Any],
    payload: Mapping[str, Any],
) -> None:
    allowed = {
        definition.name
        for definition in fields(object_type)
        if definition.init
    }
    unknown = sorted(
        set(payload) - allowed
    )

    if unknown:
        raise ValueError(
            f"Unknown fields for {object_type.__name__}: {unknown}."
        )


def _as_tuple(
    name: str,
    value: Any,
) -> tuple[Any, ...]:
    if isinstance(value, tuple):
        return value

    if isinstance(value, list):
        return tuple(value)

    raise TypeError(
        f"{name} must be a list or tuple."
    )


def _canonical_json(
    payload: Mapping[str, Any],
) -> str:
    return json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )


def _fingerprint(
    payload: Mapping[str, Any],
) -> str:
    return sha256(
        _canonical_json(payload).encode("utf-8")
    ).hexdigest()


def normalize_hazard_kind(
    value: HazardKind | str,
) -> HazardKind:
    """Return a canonical ``HazardKind``."""

    if isinstance(value, HazardKind):
        return value

    return HazardKind(value)


def normalize_hazard_name(
    value: HazardKind | str,
) -> str:
    """Return a canonical hazard-name string."""

    return normalize_hazard_kind(value).value


# =============================================================================
# Registry entry
# =============================================================================


@dataclass(slots=True, frozen=True)
class HazardRegistryEntry:
    """One stable semantic hazard identity."""

    stable_hazard_id: int

    name: HazardKind
    display_name: str
    description: str

    ontology_role: HazardOntologyRole
    parent_hazard: HazardKind | None

    query_allowed: bool
    support_state: HazardSupportState

    tags: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.tags, tuple):
            if (
                isinstance(self.tags, Sequence)
                and not isinstance(self.tags, (str, bytes))
            ):
                object.__setattr__(
                    self,
                    "tags",
                    tuple(self.tags),
                )
            else:
                raise TypeError(
                    "tags must be a tuple or non-string sequence."
                )

        self.validate()

    @property
    def fallback_only(self) -> bool:
        """Compatibility property for dependent modules."""

        return self.ontology_role == HazardOntologyRole.FALLBACK

    @property
    def runtime_allowed(self) -> bool:
        """
        Compatibility alias.

        New code should use ``query_allowed`` because data support is tracked
        separately.
        """

        return self.query_allowed

    @property
    def parent_hazard_name(self) -> str | None:
        if self.parent_hazard is None:
            return None

        return self.parent_hazard.value

    @property
    def is_data_backed(self) -> bool:
        return self.support_state == HazardSupportState.DATA_BACKED

    @property
    def is_partially_data_backed(self) -> bool:
        return (
            self.support_state
            == HazardSupportState.PARTIALLY_DATA_BACKED
        )

    @property
    def is_planned(self) -> bool:
        return self.support_state == HazardSupportState.PLANNED

    @property
    def is_deprecated(self) -> bool:
        return self.support_state == HazardSupportState.DEPRECATED

    @property
    def usable_for_new_query(self) -> bool:
        return (
            self.query_allowed
            and not self.fallback_only
            and not self.is_deprecated
        )

    def validate(self) -> None:
        _require_nonnegative_int(
            "stable_hazard_id",
            self.stable_hazard_id,
        )

        if not isinstance(self.name, HazardKind):
            raise TypeError(
                "name must be a HazardKind."
            )

        _require_nonempty_string(
            "display_name",
            self.display_name,
        )
        _require_nonempty_string(
            "description",
            self.description,
        )
        _require_boolean(
            "query_allowed",
            self.query_allowed,
        )
        _require_unique_strings(
            "hazard tags",
            self.tags,
        )

        if not isinstance(
            self.ontology_role,
            HazardOntologyRole,
        ):
            raise TypeError(
                "ontology_role must be a HazardOntologyRole."
            )

        if (
            self.parent_hazard is not None
            and not isinstance(
                self.parent_hazard,
                HazardKind,
            )
        ):
            raise TypeError(
                "parent_hazard must be absent or a HazardKind."
            )

        if not isinstance(
            self.support_state,
            HazardSupportState,
        ):
            raise TypeError(
                "support_state must be a HazardSupportState."
            )

        canonical_name = self.name.value

        expected_id = CANONICAL_HAZARD_STABLE_IDS[
            canonical_name
        ]

        if self.stable_hazard_id != expected_id:
            raise ValueError(
                f"Hazard {canonical_name!r} must use stable ID "
                f"{expected_id}; observed {self.stable_hazard_id}."
            )

        expected_display_name = (
            CANONICAL_HAZARD_DISPLAY_NAMES[
                canonical_name
            ]
        )

        if self.display_name != expected_display_name:
            raise ValueError(
                f"Hazard {canonical_name!r} must use display name "
                f"{expected_display_name!r}; observed "
                f"{self.display_name!r}."
            )

        expected_role = CANONICAL_HAZARD_ONTOLOGY_ROLES[
            canonical_name
        ]

        if self.ontology_role != expected_role:
            raise ValueError(
                f"Hazard {canonical_name!r} must use ontology role "
                f"{expected_role.value!r}; observed "
                f"{self.ontology_role.value!r}."
            )

        expected_parent_name = CANONICAL_HAZARD_PARENTS[
            canonical_name
        ]
        observed_parent_name = self.parent_hazard_name

        if observed_parent_name != expected_parent_name:
            raise ValueError(
                f"Hazard {canonical_name!r} must use parent "
                f"{expected_parent_name!r}; observed "
                f"{observed_parent_name!r}."
            )

        expected_query_allowed = (
            CANONICAL_HAZARD_QUERY_ALLOWED[
                canonical_name
            ]
        )

        if self.query_allowed != expected_query_allowed:
            raise ValueError(
                f"Hazard {canonical_name!r} has query_allowed="
                f"{self.query_allowed}; expected "
                f"{expected_query_allowed}."
            )

        if self.parent_hazard == self.name:
            raise ValueError(
                "A hazard cannot be its own parent."
            )

        if self.fallback_only:
            if self.name != FALLBACK_HAZARD:
                raise ValueError(
                    "Only all_hazard may use the fallback role."
                )

            if self.query_allowed:
                raise ValueError(
                    "Fallback hazards cannot be normal query hazards."
                )

            if (
                self.support_state
                != HazardSupportState.FALLBACK_ONLY
            ):
                raise ValueError(
                    "Fallback hazards must use support state "
                    "'fallback_only'."
                )

        elif (
            self.support_state
            == HazardSupportState.FALLBACK_ONLY
        ):
            raise ValueError(
                "Only fallback hazards may use support state "
                "'fallback_only'."
            )

        if self.is_deprecated and self.query_allowed:
            raise ValueError(
                "Deprecated hazards cannot remain queryable."
            )

        if (
            self.ontology_role
            in {
                HazardOntologyRole.DISRUPTION,
                HazardOntologyRole.EVENT_FAMILY,
                HazardOntologyRole.FALLBACK,
            }
            and self.parent_hazard is not None
        ):
            raise ValueError(
                f"Hazard role {self.ontology_role.value!r} cannot have "
                "a mechanism-family parent in the current ontology."
            )

    def semantic_content_dict(self) -> dict[str, Any]:
        """
        Pure semantic content.

        Registry and schema version numbers are deliberately excluded.
        """

        return {
            "stable_hazard_id": self.stable_hazard_id,
            "name": self.name.value,
            "display_name": self.display_name,
            "ontology_role": self.ontology_role.value,
            "parent_hazard": self.parent_hazard_name,
            "query_allowed": self.query_allowed,
        }

    def operational_dict(self) -> dict[str, Any]:
        """Semantic content plus current support status."""

        return {
            **self.semantic_content_dict(),
            "support_state": self.support_state.value,
        }

    def to_dict(self) -> dict[str, Any]:
        """Complete human-readable entry snapshot."""

        return {
            **self.operational_dict(),
            "description": self.description,
            "tags": list(self.tags),
        }

    @classmethod
    def from_dict(
        cls,
        payload: Mapping[str, Any],
    ) -> HazardRegistryEntry:
        mapping = dict(
            _require_mapping(
                "HazardRegistryEntry",
                payload,
            )
        )

        # Transitional migration from the previous field names.
        if (
            "runtime_allowed" in mapping
            and "query_allowed" not in mapping
        ):
            mapping["query_allowed"] = mapping.pop(
                "runtime_allowed"
            )

        serialized_fallback_only = mapping.pop(
            "fallback_only",
            None,
        )

        if (
            "parent_hazard_name" in mapping
            and "parent_hazard" not in mapping
        ):
            mapping["parent_hazard"] = mapping.pop(
                "parent_hazard_name"
            )

        _reject_unknown_fields(cls, mapping)

        if "name" in mapping:
            mapping["name"] = HazardKind(
                mapping["name"]
            )

        if "ontology_role" in mapping:
            mapping["ontology_role"] = (
                HazardOntologyRole(
                    mapping["ontology_role"]
                )
            )

        if mapping.get("parent_hazard") is not None:
            mapping["parent_hazard"] = HazardKind(
                mapping["parent_hazard"]
            )

        if "support_state" in mapping:
            mapping["support_state"] = (
                HazardSupportState(
                    mapping["support_state"]
                )
            )

        if "tags" in mapping:
            mapping["tags"] = _as_tuple(
                "tags",
                mapping["tags"],
            )

        entry = cls(**mapping)

        if (
            serialized_fallback_only is not None
            and not isinstance(
                serialized_fallback_only,
                bool,
            )
        ):
            raise TypeError(
                "Serialized fallback_only must be a Boolean."
            )

        if (
            serialized_fallback_only is not None
            and serialized_fallback_only
            != entry.fallback_only
        ):
            raise ValueError(
                "Serialized fallback_only disagrees with the "
                "entry ontology role."
            )

        return entry


# =============================================================================
# Canonical registry
# =============================================================================


@dataclass(slots=True, frozen=True)
class HazardRegistry:
    """Immutable canonical hazard registry."""

    entries: tuple[HazardRegistryEntry, ...]

    registry_name: str = DEFAULT_HAZARD_REGISTRY_NAME
    registry_version: str = HAZARD_REGISTRY_VERSION
    schema_version: str = HAZARD_REGISTRY_SCHEMA_VERSION

    description: str = (
        "Canonical hazard identities, ontology roles, hierarchy, query "
        "classification, and current project-level support summaries for "
        "the V2 functional UGNN."
    )

    _by_name: Mapping[
        str,
        HazardRegistryEntry,
    ] = field(
        init=False,
        repr=False,
        compare=False,
    )
    _by_stable_id: Mapping[
        int,
        HazardRegistryEntry,
    ] = field(
        init=False,
        repr=False,
        compare=False,
    )
    _children_by_name: Mapping[
        str,
        tuple[HazardRegistryEntry, ...],
    ] = field(
        init=False,
        repr=False,
        compare=False,
    )

    def __post_init__(self) -> None:
        if not isinstance(self.entries, tuple):
            if (
                isinstance(self.entries, Sequence)
                and not isinstance(self.entries, (str, bytes))
            ):
                object.__setattr__(
                    self,
                    "entries",
                    tuple(self.entries),
                )
            else:
                raise TypeError(
                    "entries must be a tuple or non-string sequence."
                )

        canonical_entries = tuple(
            sorted(
                self.entries,
                key=lambda entry: (
                    entry.stable_hazard_id
                    if isinstance(
                        entry,
                        HazardRegistryEntry,
                    )
                    else -1
                ),
            )
        )

        object.__setattr__(
            self,
            "entries",
            canonical_entries,
        )

        self.validate(
            require_current_version=False,
        )

        by_name = {
            entry.name.value: entry
            for entry in canonical_entries
        }
        by_stable_id = {
            entry.stable_hazard_id: entry
            for entry in canonical_entries
        }

        children: dict[
            str,
            list[HazardRegistryEntry],
        ] = {
            entry.name.value: []
            for entry in canonical_entries
        }

        for entry in canonical_entries:
            if entry.parent_hazard is not None:
                children[
                    entry.parent_hazard.value
                ].append(entry)

        children_by_name = {
            parent_name: tuple(
                sorted(
                    child_entries,
                    key=lambda entry: (
                        entry.stable_hazard_id
                    ),
                )
            )
            for parent_name, child_entries
            in children.items()
        }

        object.__setattr__(
            self,
            "_by_name",
            MappingProxyType(by_name),
        )
        object.__setattr__(
            self,
            "_by_stable_id",
            MappingProxyType(by_stable_id),
        )
        object.__setattr__(
            self,
            "_children_by_name",
            MappingProxyType(children_by_name),
        )

    def validate(
        self,
        *,
        require_current_version: bool,
    ) -> None:
        _require_nonempty_string(
            "registry_name",
            self.registry_name,
        )
        _require_nonempty_string(
            "registry_version",
            self.registry_version,
        )
        _require_nonempty_string(
            "schema_version",
            self.schema_version,
        )
        _require_nonempty_string(
            "description",
            self.description,
        )

        if require_current_version:
            if (
                self.registry_version
                != HAZARD_REGISTRY_VERSION
            ):
                raise ValueError(
                    "Hazard-registry version is not current. "
                    f"Observed {self.registry_version!r}; expected "
                    f"{HAZARD_REGISTRY_VERSION!r}."
                )

            if (
                self.schema_version
                != HAZARD_REGISTRY_SCHEMA_VERSION
            ):
                raise ValueError(
                    "Hazard-registry schema version is not current. "
                    f"Observed {self.schema_version!r}; expected "
                    f"{HAZARD_REGISTRY_SCHEMA_VERSION!r}."
                )

        if not self.entries:
            raise ValueError(
                "A hazard registry cannot be empty."
            )

        for index, entry in enumerate(self.entries):
            if not isinstance(
                entry,
                HazardRegistryEntry,
            ):
                raise TypeError(
                    f"entries[{index}] must be a HazardRegistryEntry."
                )

            entry.validate()

        names = tuple(
            entry.name.value
            for entry in self.entries
        )
        stable_ids = tuple(
            entry.stable_hazard_id
            for entry in self.entries
        )

        _require_unique_strings(
            "hazard registry names",
            names,
        )

        duplicate_ids = sorted(
            value
            for value, count in Counter(
                stable_ids
            ).items()
            if count > 1
        )

        if duplicate_ids:
            raise ValueError(
                "Hazard registry contains duplicate stable IDs: "
                f"{duplicate_ids}."
            )

        observed_hazards = {
            entry.name
            for entry in self.entries
        }
        expected_hazards = set(ALL_HAZARDS)

        missing = sorted(
            hazard.value
            for hazard in (
                expected_hazards - observed_hazards
            )
        )
        unexpected = sorted(
            hazard.value
            for hazard in (
                observed_hazards - expected_hazards
            )
        )

        if missing or unexpected:
            raise ValueError(
                "Hazard registry must cover the complete current "
                f"vocabulary. Missing={missing}; unexpected={unexpected}."
            )

        if (
            UNKNOWN_HAZARD_NAME in names
            or HAZARD_ID_UNKNOWN in stable_ids
        ):
            raise ValueError(
                "The synthetic unknown hazard must not appear in the "
                "semantic registry."
            )

        by_name = {
            entry.name.value: entry
            for entry in self.entries
        }

        fallback_entries = tuple(
            entry
            for entry in self.entries
            if entry.fallback_only
        )

        if len(fallback_entries) != 1:
            raise ValueError(
                "The registry must contain exactly one fallback hazard."
            )

        if fallback_entries[0].name != FALLBACK_HAZARD:
            raise ValueError(
                "The fallback hazard must be all_hazard."
            )

        observed_queryable = {
            entry.name
            for entry in self.entries
            if entry.query_allowed
        }

        if observed_queryable != set(QUERYABLE_HAZARDS):
            raise ValueError(
                "Queryable entries do not match QUERYABLE_HAZARDS."
            )

        self._validate_hierarchy(by_name)

    def _validate_hierarchy(
        self,
        by_name: Mapping[str, HazardRegistryEntry],
    ) -> None:
        for entry in self.entries:
            if entry.parent_hazard is None:
                continue

            parent_name = entry.parent_hazard.value

            if parent_name not in by_name:
                raise ValueError(
                    f"Hazard {entry.name.value!r} references missing "
                    f"parent {parent_name!r}."
                )

            parent = by_name[parent_name]

            if (
                parent.ontology_role
                != HazardOntologyRole.MECHANISM_FAMILY
            ):
                raise ValueError(
                    f"Parent hazard {parent.name.value!r} must use role "
                    "'mechanism_family'."
                )

            if (
                entry.ontology_role
                not in {
                    HazardOntologyRole.MECHANISM,
                    HazardOntologyRole.MECHANISM_FAMILY,
                }
            ):
                raise ValueError(
                    f"Hazard {entry.name.value!r} has a parent but role "
                    f"{entry.ontology_role.value!r} is not hierarchical."
                )

        visitation: dict[str, int] = {
            entry.name.value: 0
            for entry in self.entries
        }

        def visit(name: str) -> None:
            state = visitation[name]

            if state == 1:
                raise ValueError(
                    "Hazard hierarchy contains a cycle involving "
                    f"{name!r}."
                )

            if state == 2:
                return

            visitation[name] = 1
            parent = by_name[name].parent_hazard

            if parent is not None:
                visit(parent.value)

            visitation[name] = 2

        for entry in self.entries:
            visit(entry.name.value)

    def assert_current_compatibility(self) -> None:
        self.validate(
            require_current_version=True,
        )

    def __len__(self) -> int:
        return len(self.entries)

    def __iter__(self) -> Iterator[HazardRegistryEntry]:
        return iter(self.entries)

    def __contains__(self, value: object) -> bool:
        if isinstance(value, HazardKind):
            return value.value in self._by_name

        if isinstance(value, str):
            return value in self._by_name

        if isinstance(value, int) and not isinstance(value, bool):
            return value in self._by_stable_id

        return False

    @property
    def by_name(
        self,
    ) -> Mapping[str, HazardRegistryEntry]:
        return self._by_name

    @property
    def by_stable_id(
        self,
    ) -> Mapping[int, HazardRegistryEntry]:
        return self._by_stable_id

    @property
    def hazard_names(self) -> tuple[str, ...]:
        return tuple(
            entry.name.value
            for entry in self.entries
        )

    @property
    def hazard_kinds(self) -> tuple[HazardKind, ...]:
        return tuple(
            entry.name
            for entry in self.entries
        )

    @property
    def stable_hazard_ids(self) -> tuple[int, ...]:
        return tuple(
            entry.stable_hazard_id
            for entry in self.entries
        )

    @property
    def queryable_hazard_names(self) -> tuple[str, ...]:
        return tuple(
            entry.name.value
            for entry in self.entries
            if entry.query_allowed
        )

    @property
    def queryable_hazard_kinds(
        self,
    ) -> tuple[HazardKind, ...]:
        return tuple(
            entry.name
            for entry in self.entries
            if entry.query_allowed
        )

    @property
    def runtime_hazard_names(self) -> tuple[str, ...]:
        """Compatibility alias for queryable hazard names."""

        return self.queryable_hazard_names

    @property
    def runtime_hazard_kinds(
        self,
    ) -> tuple[HazardKind, ...]:
        """Compatibility alias for queryable hazard kinds."""

        return self.queryable_hazard_kinds

    @property
    def fallback_entry(self) -> HazardRegistryEntry:
        return self.get_by_name(FALLBACK_HAZARD)

    @property
    def fallback_hazard_name(self) -> str:
        return self.fallback_entry.name.value

    @property
    def fallback_hazard_kind(self) -> HazardKind:
        return self.fallback_entry.name

    @property
    def support_map(
        self,
    ) -> Mapping[str, HazardSupportState]:
        return MappingProxyType(
            {
                entry.name.value: entry.support_state
                for entry in self.entries
            }
        )

    @property
    def data_backed_hazard_names(
        self,
    ) -> tuple[str, ...]:
        return tuple(
            entry.name.value
            for entry in self.entries
            if entry.is_data_backed
        )

    @property
    def partially_data_backed_hazard_names(
        self,
    ) -> tuple[str, ...]:
        return tuple(
            entry.name.value
            for entry in self.entries
            if entry.is_partially_data_backed
        )

    @property
    def planned_hazard_names(
        self,
    ) -> tuple[str, ...]:
        return tuple(
            entry.name.value
            for entry in self.entries
            if entry.is_planned
        )

    @property
    def root_entries(
        self,
    ) -> tuple[HazardRegistryEntry, ...]:
        return tuple(
            entry
            for entry in self.entries
            if entry.parent_hazard is None
        )

    def get_by_name(
        self,
        hazard: HazardKind | str,
    ) -> HazardRegistryEntry:
        name = normalize_hazard_name(hazard)

        try:
            return self._by_name[name]
        except KeyError as exc:
            raise KeyError(
                f"Unknown canonical hazard name {name!r}."
            ) from exc

    def get_by_stable_id(
        self,
        stable_hazard_id: int,
    ) -> HazardRegistryEntry:
        try:
            return self._by_stable_id[
                stable_hazard_id
            ]
        except KeyError as exc:
            raise KeyError(
                "Unknown canonical stable hazard ID "
                f"{stable_hazard_id}."
            ) from exc

    def stable_id_for(
        self,
        hazard: HazardKind | str,
    ) -> int:
        return self.get_by_name(
            hazard
        ).stable_hazard_id

    def name_for_stable_id(
        self,
        stable_hazard_id: int,
    ) -> str:
        return self.get_by_stable_id(
            stable_hazard_id
        ).name.value

    def children_of(
        self,
        hazard: HazardKind | str,
    ) -> tuple[HazardRegistryEntry, ...]:
        name = normalize_hazard_name(hazard)
        self.get_by_name(name)

        return self._children_by_name[name]

    def ancestors_of(
        self,
        hazard: HazardKind | str,
    ) -> tuple[HazardRegistryEntry, ...]:
        entry = self.get_by_name(hazard)
        ancestors: list[HazardRegistryEntry] = []

        while entry.parent_hazard is not None:
            entry = self.get_by_name(
                entry.parent_hazard
            )
            ancestors.append(entry)

        return tuple(ancestors)

    def descendants_of(
        self,
        hazard: HazardKind | str,
    ) -> tuple[HazardRegistryEntry, ...]:
        root = self.get_by_name(hazard)
        descendants: list[HazardRegistryEntry] = []

        def collect(
            entry: HazardRegistryEntry,
        ) -> None:
            for child in self.children_of(entry.name):
                descendants.append(child)
                collect(child)

        collect(root)

        return tuple(descendants)

    def root_of(
        self,
        hazard: HazardKind | str,
    ) -> HazardRegistryEntry:
        entry = self.get_by_name(hazard)

        while entry.parent_hazard is not None:
            entry = self.get_by_name(
                entry.parent_hazard
            )

        return entry

    def assert_queryable_hazard(
        self,
        hazard: HazardKind | str,
        *,
        allow_fallback: bool = False,
    ) -> HazardRegistryEntry:
        """
        Verify that the architecture may represent this hazard.

        This does not assert that a supervised training dataset exists.
        """

        entry = self.get_by_name(hazard)

        if entry.is_deprecated:
            raise ValueError(
                f"Hazard {entry.name.value!r} is deprecated."
            )

        if entry.fallback_only:
            if not allow_fallback:
                raise ValueError(
                    f"Hazard {entry.name.value!r} is fallback-only."
                )

            return entry

        if not entry.query_allowed:
            raise ValueError(
                f"Hazard {entry.name.value!r} is not queryable."
            )

        return entry

    def assert_training_supported_hazard(
        self,
        hazard: HazardKind | str,
        *,
        allow_partially_data_backed: bool = False,
    ) -> HazardRegistryEntry:
        """
        Verify project-level eligibility for ordinary supervised training.

        Final publication-grade approval should eventually use a scoped
        hazard-support manifest rather than this global summary alone.
        """

        entry = self.assert_queryable_hazard(
            hazard,
            allow_fallback=False,
        )

        if entry.is_data_backed:
            return entry

        if (
            allow_partially_data_backed
            and entry.is_partially_data_backed
        ):
            return entry

        raise ValueError(
            f"Hazard {entry.name.value!r} is not approved by the "
            "current project-level support summary for ordinary "
            "supervised training. Current support state: "
            f"{entry.support_state.value!r}."
        )

    # ------------------------------------------------------------------
    # Fingerprint layers
    # ------------------------------------------------------------------

    def semantic_content_dict(self) -> dict[str, Any]:
        """
        Pure ontology content.

        Version numbers, support states, descriptions, and tags are excluded.
        """

        return {
            "registry_name": self.registry_name,
            "entries": [
                entry.semantic_content_dict()
                for entry in self.entries
            ],
        }

    def versioned_semantic_dict(self) -> dict[str, Any]:
        """
        Semantic content plus registry and schema versions.
        """

        return {
            **self.semantic_content_dict(),
            "registry_version": self.registry_version,
            "schema_version": self.schema_version,
        }

    def operational_dict(self) -> dict[str, Any]:
        """
        Versioned semantic identity plus current support states.
        """

        return {
            **self.versioned_semantic_dict(),
            "entries": [
                entry.operational_dict()
                for entry in self.entries
            ],
        }

    def snapshot_dict(self) -> dict[str, Any]:
        """
        Complete human-readable registry snapshot.
        """

        return {
            **self.operational_dict(),
            "description": self.description,
            "entries": [
                entry.to_dict()
                for entry in self.entries
            ],
        }

    def semantic_content_fingerprint(self) -> str:
        return _fingerprint(
            self.semantic_content_dict()
        )

    def semantic_fingerprint(self) -> str:
        """
        Alias for the pure semantic-content fingerprint.
        """

        return self.semantic_content_fingerprint()

    def versioned_semantic_fingerprint(self) -> str:
        return _fingerprint(
            self.versioned_semantic_dict()
        )

    def compatibility_fingerprint(self) -> str:
        """
        Alias for strict versioned semantic compatibility.
        """

        return self.versioned_semantic_fingerprint()

    def operational_fingerprint(self) -> str:
        return _fingerprint(
            self.operational_dict()
        )

    def snapshot_fingerprint(self) -> str:
        return _fingerprint(
            self.snapshot_dict()
        )

    def fingerprint(self) -> str:
        """
        Default strict artifact fingerprint.

        New semantic comparisons should call ``semantic_fingerprint()``
        explicitly.
        """

        return self.compatibility_fingerprint()

    def to_dict(self) -> dict[str, Any]:
        return {
            **self.snapshot_dict(),
            "semantic_content_fingerprint": (
                self.semantic_content_fingerprint()
            ),
            "versioned_semantic_fingerprint": (
                self.versioned_semantic_fingerprint()
            ),
            "operational_fingerprint": (
                self.operational_fingerprint()
            ),
            "snapshot_fingerprint": (
                self.snapshot_fingerprint()
            ),
        }

    @classmethod
    def from_dict(
        cls,
        payload: Mapping[str, Any],
        *,
        require_current_version: bool = False,
        verify_serialized_fingerprints: bool = True,
    ) -> HazardRegistry:
        mapping = dict(
            _require_mapping(
                "HazardRegistry",
                payload,
            )
        )

        serialized_semantic_content_fingerprint = mapping.pop(
            "semantic_content_fingerprint",
            None,
        )
        serialized_versioned_semantic_fingerprint = mapping.pop(
            "versioned_semantic_fingerprint",
            None,
        )
        serialized_operational_fingerprint = mapping.pop(
            "operational_fingerprint",
            None,
        )
        serialized_snapshot_fingerprint = mapping.pop(
            "snapshot_fingerprint",
            None,
        )

        # Compatibility with the earlier single semantic fingerprint.
        legacy_semantic_fingerprint = mapping.pop(
            "semantic_fingerprint",
            None,
        )

        raw_entries = mapping.pop(
            "entries",
            None,
        )

        allowed = {
            "registry_name",
            "registry_version",
            "schema_version",
            "description",
        }
        unknown = sorted(
            set(mapping) - allowed
        )

        if unknown:
            raise ValueError(
                f"Unknown HazardRegistry fields: {unknown}."
            )

        if not isinstance(raw_entries, list):
            raise TypeError(
                "HazardRegistry.entries must be a list."
            )

        if not raw_entries:
            raise ValueError(
                "HazardRegistry.entries cannot be empty."
            )

        registry = cls(
            entries=tuple(
                HazardRegistryEntry.from_dict(
                    _require_mapping(
                        f"entries[{index}]",
                        value,
                    )
                )
                for index, value in enumerate(
                    raw_entries
                )
            ),
            **mapping,
        )

        registry.validate(
            require_current_version=require_current_version,
        )

        if verify_serialized_fingerprints:
            if (
                serialized_semantic_content_fingerprint
                is not None
                and serialized_semantic_content_fingerprint
                != registry.semantic_content_fingerprint()
            ):
                raise ValueError(
                    "Serialized semantic-content fingerprint does not "
                    "match the reconstructed registry."
                )

            if (
                serialized_versioned_semantic_fingerprint
                is not None
                and serialized_versioned_semantic_fingerprint
                != registry.versioned_semantic_fingerprint()
            ):
                raise ValueError(
                    "Serialized versioned semantic fingerprint does not "
                    "match the reconstructed registry."
                )

            if (
                serialized_operational_fingerprint
                is not None
                and serialized_operational_fingerprint
                != registry.operational_fingerprint()
            ):
                raise ValueError(
                    "Serialized operational fingerprint does not match "
                    "the reconstructed registry."
                )

            if (
                serialized_snapshot_fingerprint
                is not None
                and serialized_snapshot_fingerprint
                != registry.snapshot_fingerprint()
            ):
                raise ValueError(
                    "Serialized snapshot fingerprint does not match the "
                    "reconstructed registry."
                )

            if (
                legacy_semantic_fingerprint is not None
                and legacy_semantic_fingerprint
                not in {
                    registry.semantic_fingerprint(),
                    registry.versioned_semantic_fingerprint(),
                }
            ):
                raise ValueError(
                    "Legacy semantic fingerprint does not match either "
                    "supported semantic fingerprint interpretation."
                )

        return registry


# Historical compatibility alias.
HazardRegistryIdentity = HazardRegistry


# =============================================================================
# Default registry construction
# =============================================================================


def build_default_hazard_registry() -> HazardRegistry:
    registry = HazardRegistry(
        entries=(
            HazardRegistryEntry(
                stable_hazard_id=HAZARD_ID_FLOOD,
                name=HazardKind.FLOOD,
                display_name=(
                    CANONICAL_HAZARD_DISPLAY_NAMES[
                        HazardKind.FLOOD.value
                    ]
                ),
                description=(
                    "Parent hydrological hazard family covering flood "
                    "mechanisms whose exact source may be unknown or "
                    "combined."
                ),
                ontology_role=(
                    HazardOntologyRole.MECHANISM_FAMILY
                ),
                parent_hazard=None,
                query_allowed=True,
                support_state=(
                    HazardSupportState
                    .PARTIALLY_DATA_BACKED
                ),
                tags=(
                    "environmental",
                    "hydrological",
                    "mechanism_family",
                    "proxy_targets_available",
                ),
            ),
            HazardRegistryEntry(
                stable_hazard_id=(
                    HAZARD_ID_RIVERINE_FLOOD
                ),
                name=HazardKind.RIVERINE_FLOOD,
                display_name=(
                    CANONICAL_HAZARD_DISPLAY_NAMES[
                        HazardKind.RIVERINE_FLOOD.value
                    ]
                ),
                description=(
                    "Flooding caused by a river or stream exceeding its "
                    "channel capacity or inundating its floodplain."
                ),
                ontology_role=HazardOntologyRole.MECHANISM,
                parent_hazard=HazardKind.FLOOD,
                query_allowed=True,
                support_state=HazardSupportState.PLANNED,
                tags=(
                    "environmental",
                    "hydrological",
                    "river",
                    "floodplain",
                    "planned",
                ),
            ),
            HazardRegistryEntry(
                stable_hazard_id=(
                    HAZARD_ID_PLUVIAL_FLOOD
                ),
                name=HazardKind.PLUVIAL_FLOOD,
                display_name=(
                    CANONICAL_HAZARD_DISPLAY_NAMES[
                        HazardKind.PLUVIAL_FLOOD.value
                    ]
                ),
                description=(
                    "Flooding caused by intense rainfall, surface runoff, "
                    "drainage-capacity exceedance, and ponding in "
                    "low-lying areas or topographic depressions, "
                    "independently of river overflow."
                ),
                ontology_role=HazardOntologyRole.MECHANISM,
                parent_hazard=HazardKind.FLOOD,
                query_allowed=True,
                support_state=(
                    HazardSupportState
                    .PARTIALLY_DATA_BACKED
                ),
                tags=(
                    "environmental",
                    "hydrological",
                    "rainfall",
                    "surface_runoff",
                    "urban_drainage",
                    "ponding",
                    "proxy_targets_available",
                ),
            ),
            HazardRegistryEntry(
                stable_hazard_id=HAZARD_ID_HEAT,
                name=HazardKind.HEAT,
                display_name=(
                    CANONICAL_HAZARD_DISPLAY_NAMES[
                        HazardKind.HEAT.value
                    ]
                ),
                description=(
                    "Extreme heat, heat-wave exposure, urban heat-island "
                    "effects, and associated service or health burden."
                ),
                ontology_role=HazardOntologyRole.MECHANISM,
                parent_hazard=None,
                query_allowed=True,
                support_state=HazardSupportState.PLANNED,
                tags=(
                    "environmental",
                    "thermal",
                    "heat_wave",
                    "planned",
                ),
            ),
            HazardRegistryEntry(
                stable_hazard_id=HAZARD_ID_OUTAGE,
                name=HazardKind.OUTAGE,
                display_name=(
                    CANONICAL_HAZARD_DISPLAY_NAMES[
                        HazardKind.OUTAGE.value
                    ]
                ),
                description=(
                    "Loss or degradation of power, communications, "
                    "utilities, or another critical urban service."
                ),
                ontology_role=HazardOntologyRole.DISRUPTION,
                parent_hazard=None,
                query_allowed=True,
                support_state=HazardSupportState.PLANNED,
                tags=(
                    "infrastructure",
                    "service_loss",
                    "cascading_disruption",
                    "planned",
                ),
            ),
            HazardRegistryEntry(
                stable_hazard_id=(
                    HAZARD_ID_ROAD_DISRUPTION
                ),
                name=HazardKind.ROAD_DISRUPTION,
                display_name=(
                    CANONICAL_HAZARD_DISPLAY_NAMES[
                        HazardKind.ROAD_DISRUPTION.value
                    ]
                ),
                description=(
                    "Road closure, transport-network degradation, or "
                    "loss of physical accessibility resulting from one "
                    "or more underlying hazard mechanisms."
                ),
                ontology_role=HazardOntologyRole.DISRUPTION,
                parent_hazard=None,
                query_allowed=True,
                support_state=HazardSupportState.PLANNED,
                tags=(
                    "transport",
                    "accessibility",
                    "system_disruption",
                    "planned",
                ),
            ),
            HazardRegistryEntry(
                stable_hazard_id=(
                    HAZARD_ID_CIVIL_SECURITY_EVENT
                ),
                name=HazardKind.CIVIL_SECURITY_EVENT,
                display_name=(
                    CANONICAL_HAZARD_DISPLAY_NAMES[
                        HazardKind.CIVIL_SECURITY_EVENT.value
                    ]
                ),
                description=(
                    "Broad civil-security or emergency-event family that "
                    "may include several physical mechanisms and system "
                    "disruptions simultaneously."
                ),
                ontology_role=HazardOntologyRole.EVENT_FAMILY,
                parent_hazard=None,
                query_allowed=True,
                support_state=(
                    HazardSupportState
                    .PARTIALLY_DATA_BACKED
                ),
                tags=(
                    "civil_security",
                    "event_family",
                    "multi_label_compatible",
                    "partially_data_backed",
                ),
            ),
            HazardRegistryEntry(
                stable_hazard_id=(
                    HAZARD_ID_WINTER_STORM
                ),
                name=HazardKind.WINTER_STORM,
                display_name=(
                    CANONICAL_HAZARD_DISPLAY_NAMES[
                        HazardKind.WINTER_STORM.value
                    ]
                ),
                description=(
                    "Parent winter-weather hazard family covering snow, "
                    "freezing precipitation, ice accumulation, and "
                    "related cold-season disruption."
                ),
                ontology_role=(
                    HazardOntologyRole.MECHANISM_FAMILY
                ),
                parent_hazard=None,
                query_allowed=True,
                support_state=HazardSupportState.PLANNED,
                tags=(
                    "environmental",
                    "winter_weather",
                    "mechanism_family",
                    "planned",
                ),
            ),
            HazardRegistryEntry(
                stable_hazard_id=HAZARD_ID_SNOWSTORM,
                name=HazardKind.SNOWSTORM,
                display_name=(
                    CANONICAL_HAZARD_DISPLAY_NAMES[
                        HazardKind.SNOWSTORM.value
                    ]
                ),
                description=(
                    "Heavy snowfall, blowing snow, or snow accumulation "
                    "that degrades transport, accessibility, and urban "
                    "service operations."
                ),
                ontology_role=HazardOntologyRole.MECHANISM,
                parent_hazard=HazardKind.WINTER_STORM,
                query_allowed=True,
                support_state=HazardSupportState.PLANNED,
                tags=(
                    "environmental",
                    "winter_weather",
                    "snow",
                    "transport",
                    "planned",
                ),
            ),
            HazardRegistryEntry(
                stable_hazard_id=(
                    HAZARD_ID_FREEZING_RAIN
                ),
                name=HazardKind.FREEZING_RAIN,
                display_name=(
                    CANONICAL_HAZARD_DISPLAY_NAMES[
                        HazardKind.FREEZING_RAIN.value
                    ]
                ),
                description=(
                    "Freezing precipitation that forms ice on roads, "
                    "vegetation, power lines, and structures; substantial "
                    "accumulation may constitute an ice storm."
                ),
                ontology_role=HazardOntologyRole.MECHANISM,
                parent_hazard=HazardKind.WINTER_STORM,
                query_allowed=True,
                support_state=HazardSupportState.PLANNED,
                tags=(
                    "environmental",
                    "winter_weather",
                    "ice",
                    "freezing_precipitation",
                    "outage_pathway",
                    "road_disruption_pathway",
                    "planned",
                ),
            ),
            HazardRegistryEntry(
                stable_hazard_id=(
                    HAZARD_ID_ALL_HAZARD
                ),
                name=HazardKind.ALL_HAZARD,
                display_name=(
                    CANONICAL_HAZARD_DISPLAY_NAMES[
                        HazardKind.ALL_HAZARD.value
                    ]
                ),
                description=(
                    "Fallback semantic category used for generic prior "
                    "resolution or explicitly controlled all-hazard "
                    "experiments."
                ),
                ontology_role=HazardOntologyRole.FALLBACK,
                parent_hazard=None,
                query_allowed=False,
                support_state=(
                    HazardSupportState.FALLBACK_ONLY
                ),
                tags=(
                    "fallback",
                    "not_default_query",
                ),
            ),
        ),
    )

    registry.assert_current_compatibility()
    return registry


DEFAULT_HAZARD_REGISTRY: Final[HazardRegistry] = (
    build_default_hazard_registry()
)

# Compatibility alias expected by earlier dependent drafts.
DEFAULT_HAZARD_REGISTRY_IDENTITY: Final[HazardRegistry] = (
    DEFAULT_HAZARD_REGISTRY
)


def get_default_hazard_registry() -> HazardRegistry:
    """Return the immutable canonical V2 hazard registry."""

    return DEFAULT_HAZARD_REGISTRY


def get_default_hazard_registry_identity() -> HazardRegistry:
    """Compatibility alias for earlier dependent modules."""

    return DEFAULT_HAZARD_REGISTRY


__all__ = (
    "ALL_HAZARDS",
    "ALL_PRIOR_HAZARDS",
    "CANONICAL_HAZARD_DISPLAY_NAMES",
    "CANONICAL_HAZARD_ONTOLOGY_ROLES",
    "CANONICAL_HAZARD_PARENTS",
    "CANONICAL_HAZARD_QUERY_ALLOWED",
    "CANONICAL_HAZARD_STABLE_IDS",
    "CANONICAL_HAZARD_SUPPORT_STATES",
    "DEFAULT_HAZARD_REGISTRY",
    "DEFAULT_HAZARD_REGISTRY_IDENTITY",
    "DEFAULT_HAZARD_REGISTRY_NAME",
    "DISRUPTION_HAZARDS",
    "EVENT_FAMILY_HAZARDS",
    "FALLBACK_HAZARD",
    "FALLBACK_PRIOR_HAZARD",
    "HAZARD_ID_ALL_HAZARD",
    "HAZARD_ID_CIVIL_SECURITY_EVENT",
    "HAZARD_ID_FLOOD",
    "HAZARD_ID_FREEZING_RAIN",
    "HAZARD_ID_HEAT",
    "HAZARD_ID_OUTAGE",
    "HAZARD_ID_PLUVIAL_FLOOD",
    "HAZARD_ID_RIVERINE_FLOOD",
    "HAZARD_ID_ROAD_DISRUPTION",
    "HAZARD_ID_SNOWSTORM",
    "HAZARD_ID_UNKNOWN",
    "HAZARD_ID_WINTER_STORM",
    "HAZARD_REGISTRY_SCHEMA_VERSION",
    "HAZARD_REGISTRY_VERSION",
    "HazardKind",
    "HazardOntologyRole",
    "HazardRegistry",
    "HazardRegistryEntry",
    "HazardRegistryIdentity",
    "HazardSupportState",
    "MECHANISM_FAMILY_HAZARDS",
    "MECHANISM_HAZARDS",
    "QUERYABLE_HAZARDS",
    "RUNTIME_HAZARDS",
    "RUNTIME_PRIOR_HAZARDS",
    "UNKNOWN_HAZARD_DISPLAY_NAME",
    "UNKNOWN_HAZARD_NAME",
    "build_default_hazard_registry",
    "get_default_hazard_registry",
    "get_default_hazard_registry_identity",
    "normalize_hazard_kind",
    "normalize_hazard_name",
)