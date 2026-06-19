"""
Scoped hazard-relation priors for the V2 functional UGNN.

Target path:
    urban_resilience_models/
        v2_hazard_conditioned_functional_ugnn/
            relations/
                hazard_relation_priors.py

Python requirement:
    Python >= 3.11

This module owns only prior-specific concepts:

- qualitative prior strength and confidence;
- task and dataset applicability scope;
- evidence and empirical-estimation provenance;
- hazard/relation fallback-resolution policy;
- confidence attenuation under ontology inheritance;
- optional confidence-adjusted gate initialization;
- deterministic compilation against a relation registry;
- prior artifact serialization and fingerprints.

It does not define hazard semantics. Canonical hazard names, stable IDs,
queryability, support states, ontology roles, parent-child relationships, and
fallback identity are imported from ``hazard/hazard_registry.py``.

Interpretation
-------------------------
A high prior means:

    "Before fitting the model, this relation is provisionally expected to be
    comparatively relevant for this hazard and task scope."

It does not mean:

- the relation must be active;
- the relation causes the target;
- the relation increases risk;
- other relations are forbidden.

Protective relations may receive high relevance priors without implying a
positive risk effect.

Default registry status
-----------------------
The default numerical values are ontology-derived, provisional, and not
externally calibrated. They may be used for diagnostics or weak sensitivity
initialization, but not for substantive training regularization.

Two-dimensional inheritance
---------------------------
Sparse custom registries may inherit priors along both ontologies:

- hazard hierarchy, such as ``pluvial_flood -> flood``;
- relation hierarchy, such as a specialized relation -> relation family.

The resolved cell always preserves:

- the requested hazard and relation;
- the source hazard and relation;
- separate hazard and relation inheritance distances.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field, fields
from datetime import date
from enum import StrEnum
from hashlib import sha256
import json
import math
from types import MappingProxyType
from typing import Any, Final, Iterator, Mapping, Sequence

from .. import constants as C
from ..hazard.hazard_registry import (
    ALL_HAZARDS,
    DEFAULT_HAZARD_REGISTRY,
    FALLBACK_HAZARD,
    HazardKind,
    HazardRegistry,
)
from .relation_registry import (
    CompiledRelationRegistry,
    DEFAULT_RELATION_REGISTRY,
    RelationRegistry,
)


# =============================================================================
# Schema identity
# =============================================================================


PRIOR_APPLICABILITY_SCHEMA_VERSION: Final[str] = "0.2"
EMPIRICAL_PRIOR_PROVENANCE_SCHEMA_VERSION: Final[str] = "0.2"
HAZARD_RELATION_PRIOR_SCHEMA_VERSION: Final[str] = "0.3"
HAZARD_RELATION_PRIOR_REGISTRY_VERSION: Final[str] = "0.3"
COMPILED_HAZARD_PRIOR_SCHEMA_VERSION: Final[str] = "0.3"

DEFAULT_HAZARD_RELATION_PRIOR_REGISTRY_NAME: Final[str] = (
    "v2_provisional_scoped_hazard_relation_priors"
)


# =============================================================================
# Prior vocabulary
# =============================================================================


class PriorStrength(StrEnum):
    """Qualitative expected relation relevance."""

    VERY_LOW = "very_low"
    LOW = "low"
    LOW_MEDIUM = "low_medium"
    MEDIUM = "medium"
    MEDIUM_HIGH = "medium_high"
    HIGH = "high"
    VERY_HIGH = "very_high"


PRIOR_MEAN_BY_STRENGTH: Final[Mapping[PriorStrength, float]] = (
    MappingProxyType(
        {
            PriorStrength.VERY_LOW: 0.10,
            PriorStrength.LOW: 0.20,
            PriorStrength.LOW_MEDIUM: 0.35,
            PriorStrength.MEDIUM: 0.50,
            PriorStrength.MEDIUM_HIGH: 0.65,
            PriorStrength.HIGH: 0.80,
            PriorStrength.VERY_HIGH: 0.90,
        }
    )
)


class PriorEvidenceType(StrEnum):
    """Primary evidentiary basis for one prior cell."""

    PROVISIONAL_ONTOLOGY = "provisional_ontology"
    ONTOLOGY = "ontology"
    LITERATURE = "literature"
    EXPERT = "expert"
    EMPIRICAL = "empirical"
    MIXED = "mixed"
    CONTROL = "control"


class PriorResolutionMode(StrEnum):
    """How a requested hazard-relation cell was resolved."""

    EXPLICIT = "explicit"
    RELATION_ANCESTOR = "relation_ancestor"
    HAZARD_ANCESTOR = "hazard_ancestor"
    HAZARD_RELATION_ANCESTOR = "hazard_relation_ancestor"
    ALL_HAZARD = "all_hazard"
    ALL_HAZARD_RELATION_ANCESTOR = "all_hazard_relation_ancestor"
    NEUTRAL_DEFAULT = "neutral_default"


class PriorResolutionPolicy(StrEnum):
    """Fallback precedence when an explicit cell is missing."""

    HAZARD_FIRST = "hazard_first"
    RELATION_FIRST = "relation_first"
    EXPLICIT_ONLY = "explicit_only"


class PriorRegistryStatus(StrEnum):
    """Scientific maturity of a prior registry."""

    PROVISIONAL = "provisional"
    REVIEWED = "reviewed"
    CALIBRATED = "calibrated"


class GateInitializationActivation(StrEnum):
    """Gate activation supported by prior-to-bias conversion."""

    SIGMOID = "sigmoid"


# =============================================================================
# Generic helpers
# =============================================================================


def _require_nonempty_string(name: str, value: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string.")


def _require_finite_number(name: str, value: int | float) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be numeric.")

    converted = float(value)
    if not math.isfinite(converted):
        raise ValueError(f"{name} must be finite.")
    return converted


def _require_probability(
    name: str,
    value: float,
    *,
    include_endpoints: bool = True,
) -> None:
    converted = _require_finite_number(name, value)
    valid = (
        0.0 <= converted <= 1.0
        if include_endpoints
        else 0.0 < converted < 1.0
    )
    if not valid:
        interval = "[0, 1]" if include_endpoints else "(0, 1)"
        raise ValueError(f"{name} must lie in {interval}.")


def _require_unique_strings(name: str, values: Sequence[str]) -> None:
    for index, value in enumerate(values):
        _require_nonempty_string(f"{name}[{index}]", value)

    duplicates = sorted(
        value
        for value, count in Counter(values).items()
        if count > 1
    )
    if duplicates:
        raise ValueError(f"{name} contains duplicates: {duplicates}.")


def _require_mapping(name: str, value: Any) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must be a mapping.")
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
    unknown = sorted(set(payload) - allowed)
    if unknown:
        raise ValueError(
            f"Unknown fields for {object_type.__name__}: {unknown}."
        )


def _as_tuple(name: str, value: Any) -> tuple[Any, ...]:
    if isinstance(value, tuple):
        return value
    if isinstance(value, list):
        return tuple(value)
    raise TypeError(f"{name} must be a list or tuple.")


def _canonical_json(payload: Mapping[str, Any]) -> str:
    return json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )


def _fingerprint(payload: Mapping[str, Any]) -> str:
    return sha256(_canonical_json(payload).encode("utf-8")).hexdigest()


def _hazard(value: HazardKind | str) -> HazardKind:
    return value if isinstance(value, HazardKind) else HazardKind(value)


def _validate_iso_date(name: str, value: str | None) -> None:
    if value is None:
        return
    _require_nonempty_string(name, value)
    try:
        date.fromisoformat(value)
    except ValueError as exc:
        raise ValueError(
            f"{name} must use ISO date format YYYY-MM-DD."
        ) from exc


# =============================================================================
# Applicability scope
# =============================================================================


@dataclass(slots=True, frozen=True)
class PriorApplicationContext:
    """Concrete task context requesting use of a prior registry."""

    target_family: str
    target_name: str
    forecast_horizon: str
    geography_level: str
    study_region: str

    dataset_fingerprint: str | None = None
    study_id: str | None = None

    def __post_init__(self) -> None:
        for name, value in (
            ("target_family", self.target_family),
            ("target_name", self.target_name),
            ("forecast_horizon", self.forecast_horizon),
            ("geography_level", self.geography_level),
            ("study_region", self.study_region),
        ):
            _require_nonempty_string(name, value)

        if self.dataset_fingerprint is not None:
            _require_nonempty_string(
                "dataset_fingerprint", self.dataset_fingerprint
            )
        if self.study_id is not None:
            _require_nonempty_string("study_id", self.study_id)

    def to_dict(self) -> dict[str, Any]:
        return {
            "target_family": self.target_family,
            "target_name": self.target_name,
            "forecast_horizon": self.forecast_horizon,
            "geography_level": self.geography_level,
            "study_region": self.study_region,
            "dataset_fingerprint": self.dataset_fingerprint,
            "study_id": self.study_id,
        }

    @classmethod
    def from_dict(
        cls,
        payload: Mapping[str, Any],
    ) -> PriorApplicationContext:
        mapping = dict(_require_mapping("PriorApplicationContext", payload))
        _reject_unknown_fields(cls, mapping)
        return cls(**mapping)


@dataclass(slots=True, frozen=True)
class PriorApplicabilityScope:
    """
    Task scope under which a prior registry may be used.

    Empty tuple constraints mean that the corresponding dimension is
    unrestricted.
    """

    target_family: str | None = None
    target_names: tuple[str, ...] = ()
    forecast_horizons: tuple[str, ...] = ()
    geography_levels: tuple[str, ...] = ()
    study_regions: tuple[str, ...] = ()
    dataset_fingerprints: tuple[str, ...] = ()
    study_ids: tuple[str, ...] = ()
    schema_version: str = PRIOR_APPLICABILITY_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.target_family is not None:
            _require_nonempty_string("target_family", self.target_family)

        for name, values in (
            ("target_names", self.target_names),
            ("forecast_horizons", self.forecast_horizons),
            ("geography_levels", self.geography_levels),
            ("study_regions", self.study_regions),
            ("dataset_fingerprints", self.dataset_fingerprints),
            ("study_ids", self.study_ids),
        ):
            _require_unique_strings(name, values)

        _require_nonempty_string("schema_version", self.schema_version)

    def matches(self, context: PriorApplicationContext) -> bool:
        if not isinstance(context, PriorApplicationContext):
            raise TypeError("context must be a PriorApplicationContext.")

        if (
            self.target_family is not None
            and context.target_family != self.target_family
        ):
            return False

        checks = (
            (self.target_names, context.target_name),
            (self.forecast_horizons, context.forecast_horizon),
            (self.geography_levels, context.geography_level),
            (self.study_regions, context.study_region),
            (self.dataset_fingerprints, context.dataset_fingerprint),
            (self.study_ids, context.study_id),
        )
        return all(
            not allowed_values or observed in allowed_values
            for allowed_values, observed in checks
        )

    def assert_applicable(self, context: PriorApplicationContext) -> None:
        if not self.matches(context):
            raise ValueError(
                "The hazard-relation prior registry is not applicable to "
                "the supplied prediction context."
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "target_family": self.target_family,
            "target_names": list(self.target_names),
            "forecast_horizons": list(self.forecast_horizons),
            "geography_levels": list(self.geography_levels),
            "study_regions": list(self.study_regions),
            "dataset_fingerprints": list(self.dataset_fingerprints),
            "study_ids": list(self.study_ids),
            "schema_version": self.schema_version,
        }

    def fingerprint(self) -> str:
        return _fingerprint(self.to_dict())

    @classmethod
    def from_dict(
        cls,
        payload: Mapping[str, Any],
    ) -> PriorApplicabilityScope:
        mapping = dict(_require_mapping("PriorApplicabilityScope", payload))
        _reject_unknown_fields(cls, mapping)
        for field_name in (
            "target_names",
            "forecast_horizons",
            "geography_levels",
            "study_regions",
            "dataset_fingerprints",
            "study_ids",
        ):
            if field_name in mapping:
                mapping[field_name] = _as_tuple(field_name, mapping[field_name])
        return cls(**mapping)


DEFAULT_PRIOR_APPLICABILITY_SCOPE: Final[PriorApplicabilityScope] = (
    PriorApplicabilityScope(
        target_family="municipal_service_disruption_burden",
        target_names=("water_drainage_count",),
        forecast_horizons=("next_month",),
        geography_levels=("census_tract",),
        study_regions=("montreal",),
    )
)

DEFAULT_PRIOR_APPLICATION_CONTEXT: Final[PriorApplicationContext] = (
    PriorApplicationContext(
        target_family="municipal_service_disruption_burden",
        target_name="water_drainage_count",
        forecast_horizon="next_month",
        geography_level="census_tract",
        study_region="montreal",
    )
)


# =============================================================================
# Evidence provenance
# =============================================================================


@dataclass(slots=True, frozen=True)
class EmpiricalPriorProvenance:
    """Leakage-sensitive provenance for an empirically estimated prior."""

    dataset_fingerprint: str
    split_fingerprint: str
    source_artifact_fingerprint: str
    estimation_cutoff: str
    estimator_name: str
    estimator_version: str
    held_out_estimation: bool
    random_seed: int
    schema_version: str = EMPIRICAL_PRIOR_PROVENANCE_SCHEMA_VERSION

    def __post_init__(self) -> None:
        for name, value in (
            ("dataset_fingerprint", self.dataset_fingerprint),
            ("split_fingerprint", self.split_fingerprint),
            ("source_artifact_fingerprint", self.source_artifact_fingerprint),
            ("estimation_cutoff", self.estimation_cutoff),
            ("estimator_name", self.estimator_name),
            ("estimator_version", self.estimator_version),
            ("schema_version", self.schema_version),
        ):
            _require_nonempty_string(name, value)

        if not isinstance(self.held_out_estimation, bool):
            raise TypeError("held_out_estimation must be a Boolean.")
        if (
            isinstance(self.random_seed, bool)
            or not isinstance(self.random_seed, int)
            or self.random_seed < 0
        ):
            raise ValueError("random_seed must be a nonnegative integer.")

    def to_dict(self) -> dict[str, Any]:
        return {
            "dataset_fingerprint": self.dataset_fingerprint,
            "split_fingerprint": self.split_fingerprint,
            "source_artifact_fingerprint": self.source_artifact_fingerprint,
            "estimation_cutoff": self.estimation_cutoff,
            "estimator_name": self.estimator_name,
            "estimator_version": self.estimator_version,
            "held_out_estimation": self.held_out_estimation,
            "random_seed": self.random_seed,
            "schema_version": self.schema_version,
        }

    @classmethod
    def from_dict(
        cls,
        payload: Mapping[str, Any],
    ) -> EmpiricalPriorProvenance:
        mapping = dict(_require_mapping("EmpiricalPriorProvenance", payload))
        _reject_unknown_fields(cls, mapping)
        return cls(**mapping)


# =============================================================================
# Per-cell prior definition
# =============================================================================


@dataclass(slots=True, frozen=True)
class PriorCellDefinition:
    """Scientific metadata for one hazard-relation prior cell."""

    strength: PriorStrength
    confidence: float
    evidence_type: PriorEvidenceType
    rationale: str
    caveat: str | None = None
    initialization_allowed: bool = True
    regularization_allowed: bool = False
    evidence_reference_ids: tuple[str, ...] = ()
    expert_source_ids: tuple[str, ...] = ()
    reviewed_by: tuple[str, ...] = ()
    review_date: str | None = None
    empirical_provenance: EmpiricalPriorProvenance | None = None

    def __post_init__(self) -> None:
        self.validate()

    @property
    def prior_mean(self) -> float:
        return PRIOR_MEAN_BY_STRENGTH[self.strength]

    @property
    def is_neutral(self) -> bool:
        return self.strength == PriorStrength.MEDIUM

    def validate(self) -> None:
        if not isinstance(self.strength, PriorStrength):
            raise TypeError("strength must be a PriorStrength.")
        if not isinstance(self.evidence_type, PriorEvidenceType):
            raise TypeError("evidence_type must be a PriorEvidenceType.")
        if not isinstance(self.initialization_allowed, bool):
            raise TypeError("initialization_allowed must be a Boolean.")
        if not isinstance(self.regularization_allowed, bool):
            raise TypeError("regularization_allowed must be a Boolean.")

        _require_probability("confidence", self.confidence)
        _require_probability(
            "prior_mean", self.prior_mean, include_endpoints=False
        )
        _require_nonempty_string("rationale", self.rationale)
        if self.caveat is not None:
            _require_nonempty_string("caveat", self.caveat)

        for name, values in (
            ("evidence_reference_ids", self.evidence_reference_ids),
            ("expert_source_ids", self.expert_source_ids),
            ("reviewed_by", self.reviewed_by),
        ):
            _require_unique_strings(name, values)
        _validate_iso_date("review_date", self.review_date)

        if self.confidence == 0.0 and (
            self.initialization_allowed or self.regularization_allowed
        ):
            raise ValueError(
                "Zero-confidence cells cannot initialize or regularize."
            )

        if self.evidence_type == PriorEvidenceType.CONTROL:
            if not self.is_neutral or self.confidence != 0.0:
                raise ValueError("Control priors must be neutral and zero-confidence.")
            if self.initialization_allowed or self.regularization_allowed:
                raise ValueError("Control priors cannot initialize or regularize.")
        elif self.evidence_type == PriorEvidenceType.LITERATURE:
            if not self.evidence_reference_ids:
                raise ValueError("Literature priors require reference IDs.")
        elif self.evidence_type == PriorEvidenceType.EXPERT:
            if not self.expert_source_ids:
                raise ValueError("Expert priors require expert source IDs.")
        elif self.evidence_type == PriorEvidenceType.EMPIRICAL:
            if self.empirical_provenance is None:
                raise ValueError("Empirical priors require provenance.")
        elif self.evidence_type == PriorEvidenceType.MIXED:
            if not (
                self.evidence_reference_ids
                or self.expert_source_ids
                or self.empirical_provenance is not None
            ):
                raise ValueError("Mixed priors require auditable evidence.")

        if (
            self.evidence_type
            not in {PriorEvidenceType.EMPIRICAL, PriorEvidenceType.MIXED}
            and self.empirical_provenance is not None
        ):
            raise ValueError(
                "empirical_provenance is only valid for empirical or mixed "
                "priors."
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "strength": self.strength.value,
            "prior_mean": self.prior_mean,
            "confidence": self.confidence,
            "evidence_type": self.evidence_type.value,
            "rationale": self.rationale,
            "caveat": self.caveat,
            "initialization_allowed": self.initialization_allowed,
            "regularization_allowed": self.regularization_allowed,
            "evidence_reference_ids": list(self.evidence_reference_ids),
            "expert_source_ids": list(self.expert_source_ids),
            "reviewed_by": list(self.reviewed_by),
            "review_date": self.review_date,
            "empirical_provenance": (
                self.empirical_provenance.to_dict()
                if self.empirical_provenance is not None
                else None
            ),
        }

    @classmethod
    def from_dict(
        cls,
        payload: Mapping[str, Any],
    ) -> PriorCellDefinition:
        mapping = dict(_require_mapping("PriorCellDefinition", payload))
        serialized_mean = mapping.pop("prior_mean", None)
        _reject_unknown_fields(cls, mapping)
        mapping["strength"] = PriorStrength(mapping["strength"])
        mapping["evidence_type"] = PriorEvidenceType(mapping["evidence_type"])
        for field_name in (
            "evidence_reference_ids",
            "expert_source_ids",
            "reviewed_by",
        ):
            if field_name in mapping:
                mapping[field_name] = _as_tuple(field_name, mapping[field_name])
        if mapping.get("empirical_provenance") is not None:
            mapping["empirical_provenance"] = EmpiricalPriorProvenance.from_dict(
                _require_mapping(
                    "empirical_provenance", mapping["empirical_provenance"]
                )
            )
        definition = cls(**mapping)
        if (
            serialized_mean is not None
            and abs(float(serialized_mean) - definition.prior_mean) > 1e-12
        ):
            raise ValueError("Serialized prior_mean does not match strength.")
        return definition


@dataclass(slots=True, frozen=True)
class HazardRelationPrior:
    """One explicit hazard-relation prior cell."""

    hazard: HazardKind
    relation_name: str
    definition: PriorCellDefinition
    schema_version: str = HAZARD_RELATION_PRIOR_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if not isinstance(self.hazard, HazardKind):
            raise TypeError("hazard must be a HazardKind.")
        _require_nonempty_string("relation_name", self.relation_name)
        if not isinstance(self.definition, PriorCellDefinition):
            raise TypeError("definition must be a PriorCellDefinition.")
        _require_nonempty_string("schema_version", self.schema_version)

    @property
    def prior_mean(self) -> float:
        return self.definition.prior_mean

    @property
    def confidence(self) -> float:
        return self.definition.confidence

    def to_dict(self) -> dict[str, Any]:
        return {
            "hazard": self.hazard.value,
            "relation_name": self.relation_name,
            "definition": self.definition.to_dict(),
            "schema_version": self.schema_version,
        }

    @classmethod
    def from_dict(
        cls,
        payload: Mapping[str, Any],
    ) -> HazardRelationPrior:
        mapping = dict(_require_mapping("HazardRelationPrior", payload))
        _reject_unknown_fields(cls, mapping)
        mapping["hazard"] = HazardKind(mapping["hazard"])
        mapping["definition"] = PriorCellDefinition.from_dict(
            _require_mapping("definition", mapping["definition"])
        )
        return cls(**mapping)


@dataclass(slots=True, frozen=True)
class RelationPriorProfile:
    """Compact relation profile with independent hazard-cell overrides."""

    relation_name: str
    default_definition: PriorCellDefinition
    overrides: tuple[tuple[HazardKind, PriorCellDefinition], ...] = ()

    def __post_init__(self) -> None:
        _require_nonempty_string("relation_name", self.relation_name)
        if not isinstance(self.default_definition, PriorCellDefinition):
            raise TypeError(
                "default_definition must be a PriorCellDefinition."
            )

        hazards: list[HazardKind] = []
        for index, pair in enumerate(self.overrides):
            if not isinstance(pair, tuple) or len(pair) != 2:
                raise TypeError(f"overrides[{index}] must be a two-item tuple.")
            hazard, definition = pair
            if not isinstance(hazard, HazardKind):
                raise TypeError("Override hazards must be HazardKind values.")
            if hazard not in ALL_HAZARDS:
                raise ValueError(f"Unsupported override hazard {hazard!r}.")
            if not isinstance(definition, PriorCellDefinition):
                raise TypeError(
                    "Override definitions must be PriorCellDefinition objects."
                )
            hazards.append(hazard)

        duplicates = sorted(
            hazard.value
            for hazard, count in Counter(hazards).items()
            if count > 1
        )
        if duplicates:
            raise ValueError(f"Duplicate hazard overrides: {duplicates}.")

    @property
    def override_map(self) -> Mapping[HazardKind, PriorCellDefinition]:
        return MappingProxyType(dict(self.overrides))

    def build_priors(
        self,
        *,
        hazard_registry: HazardRegistry,
    ) -> tuple[HazardRelationPrior, ...]:
        overrides = self.override_map
        return tuple(
            HazardRelationPrior(
                hazard=entry.name,
                relation_name=self.relation_name,
                definition=overrides.get(entry.name, self.default_definition),
            )
            for entry in hazard_registry.entries
        )


# =============================================================================
# Resolved prior
# =============================================================================


@dataclass(slots=True, frozen=True)
class ResolvedHazardRelationPrior:
    """One requested prior cell after explicit fallback resolution."""

    hazard: HazardKind
    relation_name: str
    prior_mean: float
    confidence: float
    initialization_allowed: bool
    regularization_allowed: bool
    evidence_type: PriorEvidenceType
    rationale: str
    caveat: str | None
    resolution_mode: PriorResolutionMode
    source_hazard: HazardKind | None
    source_relation_name: str | None
    hazard_inheritance_distance: int = 0
    relation_inheritance_distance: int = 0

    def __post_init__(self) -> None:
        if not isinstance(self.hazard, HazardKind):
            raise TypeError("hazard must be a HazardKind.")
        _require_nonempty_string("relation_name", self.relation_name)
        _require_probability("prior_mean", self.prior_mean, include_endpoints=False)
        _require_probability("confidence", self.confidence)
        if not isinstance(self.evidence_type, PriorEvidenceType):
            raise TypeError("evidence_type must be a PriorEvidenceType.")
        if not isinstance(self.resolution_mode, PriorResolutionMode):
            raise TypeError("resolution_mode must be a PriorResolutionMode.")
        _require_nonempty_string("rationale", self.rationale)
        if self.caveat is not None:
            _require_nonempty_string("caveat", self.caveat)

        for name, distance in (
            ("hazard_inheritance_distance", self.hazard_inheritance_distance),
            ("relation_inheritance_distance", self.relation_inheritance_distance),
        ):
            if (
                isinstance(distance, bool)
                or not isinstance(distance, int)
                or distance < 0
            ):
                raise ValueError(f"{name} must be a nonnegative integer.")

        if self.confidence == 0.0 and (
            self.initialization_allowed or self.regularization_allowed
        ):
            raise ValueError(
                "Zero-confidence resolved cells cannot initialize or regularize."
            )

        if self.resolution_mode == PriorResolutionMode.NEUTRAL_DEFAULT:
            if (
                self.source_hazard is not None
                or self.source_relation_name is not None
                or self.hazard_inheritance_distance != 0
                or self.relation_inheritance_distance != 0
            ):
                raise ValueError("Neutral defaults cannot identify a source prior.")
        elif self.source_hazard is None or self.source_relation_name is None:
            raise ValueError(
                "Non-neutral resolutions must identify source hazard and relation."
            )

    def effective_initialization_mean(self, *, neutral_prior_mean: float) -> float:
        _require_probability(
            "neutral_prior_mean", neutral_prior_mean, include_endpoints=False
        )
        if not self.initialization_allowed:
            return neutral_prior_mean
        return neutral_prior_mean + self.confidence * (
            self.prior_mean - neutral_prior_mean
        )


# =============================================================================
# Prior registry
# =============================================================================


@dataclass(slots=True, frozen=True)
class HazardRelationPriorRegistry:
    """Immutable, scoped and versioned hazard-relation prior registry."""

    priors: tuple[HazardRelationPrior, ...]
    applicability_scope: PriorApplicabilityScope
    source_hazard_registry: HazardRegistry

    source_relation_registry_name: str
    source_relation_registry_version: str
    source_relation_semantic_fingerprint: str
    source_relation_names: tuple[str, ...]
    source_stable_relation_ids: tuple[int, ...]

    registry_name: str = DEFAULT_HAZARD_RELATION_PRIOR_REGISTRY_NAME
    registry_version: str = HAZARD_RELATION_PRIOR_REGISTRY_VERSION
    schema_version: str = HAZARD_RELATION_PRIOR_SCHEMA_VERSION
    registry_status: PriorRegistryStatus = PriorRegistryStatus.PROVISIONAL
    regularization_approved: bool = False
    provisional_notice: str = (
        "Ontology-derived provisional priors; not externally calibrated and "
        "not approved for substantive regularization."
    )
    neutral_prior_mean: float = 0.50

    _by_key: Mapping[
        tuple[HazardKind, str],
        HazardRelationPrior,
    ] = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        self._validate_nonordering_fields()

        hazard_position = {
            entry.name: index
            for index, entry in enumerate(self.source_hazard_registry.entries)
        }
        relation_position = {
            relation_name: index
            for index, relation_name in enumerate(self.source_relation_names)
        }

        unknown_relations = sorted(
            {prior.relation_name for prior in self.priors}
            - set(self.source_relation_names)
        )
        if unknown_relations:
            raise ValueError(
                "Prior registry contains relations absent from its source "
                f"registry: {unknown_relations}."
            )

        canonical_priors = tuple(
            sorted(
                self.priors,
                key=lambda prior: (
                    hazard_position[prior.hazard],
                    relation_position[prior.relation_name],
                ),
            )
        )
        object.__setattr__(self, "priors", canonical_priors)
        self._validate_prior_cells()
        object.__setattr__(
            self,
            "_by_key",
            MappingProxyType(
                {
                    (prior.hazard, prior.relation_name): prior
                    for prior in canonical_priors
                }
            ),
        )

    def _validate_nonordering_fields(self) -> None:
        if not isinstance(self.applicability_scope, PriorApplicabilityScope):
            raise TypeError(
                "applicability_scope must be a PriorApplicabilityScope."
            )
        if not isinstance(self.source_hazard_registry, HazardRegistry):
            raise TypeError("source_hazard_registry must be a HazardRegistry.")
        if not isinstance(self.registry_status, PriorRegistryStatus):
            raise TypeError("registry_status must be a PriorRegistryStatus.")
        if not isinstance(self.regularization_approved, bool):
            raise TypeError("regularization_approved must be a Boolean.")

        for name, value in (
            ("source_relation_registry_name", self.source_relation_registry_name),
            (
                "source_relation_registry_version",
                self.source_relation_registry_version,
            ),
            (
                "source_relation_semantic_fingerprint",
                self.source_relation_semantic_fingerprint,
            ),
            ("registry_name", self.registry_name),
            ("registry_version", self.registry_version),
            ("schema_version", self.schema_version),
            ("provisional_notice", self.provisional_notice),
        ):
            _require_nonempty_string(name, value)

        _require_unique_strings("source_relation_names", self.source_relation_names)
        if len(self.source_relation_names) != len(self.source_stable_relation_ids):
            raise ValueError(
                "source_relation_names and source_stable_relation_ids must align."
            )
        if len(set(self.source_stable_relation_ids)) != len(
            self.source_stable_relation_ids
        ):
            raise ValueError("source_stable_relation_ids must be unique.")
        for relation_id in self.source_stable_relation_ids:
            if (
                isinstance(relation_id, bool)
                or not isinstance(relation_id, int)
                or relation_id < 0
            ):
                raise ValueError(
                    "Stable relation IDs must be nonnegative integers."
                )

        _require_probability(
            "neutral_prior_mean", self.neutral_prior_mean, include_endpoints=False
        )
        if (
            self.registry_status == PriorRegistryStatus.PROVISIONAL
            and self.regularization_approved
        ):
            raise ValueError(
                "A provisional registry cannot be approved for regularization."
            )

    def _validate_prior_cells(self) -> None:
        if not self.priors:
            raise ValueError("A hazard-relation prior registry cannot be empty.")

        keys = tuple(
            (prior.hazard, prior.relation_name)
            for prior in self.priors
        )
        duplicate_keys = sorted(
            (hazard.value, relation_name)
            for (hazard, relation_name), count in Counter(keys).items()
            if count > 1
        )
        if duplicate_keys:
            raise ValueError(f"Duplicate prior cells: {duplicate_keys}.")

        allowed_hazards = set(self.source_hazard_registry.hazard_kinds)
        for prior in self.priors:
            if prior.hazard not in allowed_hazards:
                raise ValueError(
                    f"Prior hazard {prior.hazard.value!r} is absent from the "
                    "source hazard registry."
                )
            if (
                not self.regularization_approved
                and prior.definition.regularization_allowed
            ):
                raise ValueError(
                    "Cells cannot permit regularization while the registry is "
                    "not approved for it."
                )

    def validate_against_sources(
        self,
        *,
        hazard_registry: HazardRegistry,
        relation_registry: RelationRegistry,
        require_complete: bool = True,
        require_current_versions: bool = True,
    ) -> None:
        if require_current_versions:
            hazard_registry.assert_current_compatibility()
            relation_registry.assert_current_compatibility()
            if self.registry_version != HAZARD_RELATION_PRIOR_REGISTRY_VERSION:
                raise ValueError("Hazard-prior registry version is not current.")
            if self.schema_version != HAZARD_RELATION_PRIOR_SCHEMA_VERSION:
                raise ValueError("Hazard-prior schema version is not current.")

        if (
            hazard_registry.registry_name
            != self.source_hazard_registry.registry_name
            or hazard_registry.registry_version
            != self.source_hazard_registry.registry_version
            or hazard_registry.compatibility_fingerprint()
            != self.source_hazard_registry.compatibility_fingerprint()
        ):
            raise ValueError(
                "Prior registry references a different hazard registry."
            )

        if (
            relation_registry.registry_name
            != self.source_relation_registry_name
            or relation_registry.registry_version
            != self.source_relation_registry_version
            or relation_registry.semantic_fingerprint()
            != self.source_relation_semantic_fingerprint
            or relation_registry.relation_names != self.source_relation_names
            or relation_registry.relation_ids != self.source_stable_relation_ids
        ):
            raise ValueError(
                "Prior registry references a different relation registry."
            )

        for prior in self.priors:
            entry = relation_registry.get_entry_by_name(prior.relation_name)
            definition = prior.definition
            if entry.specification.is_control:
                if (
                    definition.evidence_type != PriorEvidenceType.CONTROL
                    or not definition.is_neutral
                    or definition.confidence != 0.0
                    or definition.initialization_allowed
                    or definition.regularization_allowed
                ):
                    raise ValueError(
                        f"Control relation {prior.relation_name!r} must have "
                        "neutral, zero-confidence, non-operative priors."
                    )
            elif definition.evidence_type == PriorEvidenceType.CONTROL:
                raise ValueError(
                    f"Non-control relation {prior.relation_name!r} cannot use "
                    "control evidence."
                )

        if require_complete:
            expected = {
                (hazard, relation_name)
                for hazard in hazard_registry.hazard_kinds
                for relation_name in relation_registry.relation_names
            }
            observed = {
                (prior.hazard, prior.relation_name)
                for prior in self.priors
            }
            missing = sorted(
                (hazard.value, relation_name)
                for hazard, relation_name in expected - observed
            )
            if missing:
                raise ValueError(f"Prior registry is incomplete: {missing}.")

    def __len__(self) -> int:
        return len(self.priors)

    def __iter__(self) -> Iterator[HazardRelationPrior]:
        return iter(self.priors)

    def get_explicit(
        self,
        hazard: HazardKind | str,
        relation_name: str,
    ) -> HazardRelationPrior | None:
        return self._by_key.get((_hazard(hazard), relation_name))

    def require_explicit(
        self,
        hazard: HazardKind | str,
        relation_name: str,
    ) -> HazardRelationPrior:
        prior = self.get_explicit(hazard, relation_name)
        if prior is None:
            raise KeyError(
                "No explicit prior for "
                f"{_hazard(hazard).value!r} and {relation_name!r}."
            )
        return prior

    def resolve(
        self,
        hazard: HazardKind | str,
        relation_name: str,
        *,
        hazard_registry: HazardRegistry,
        relation_registry: RelationRegistry,
        resolution_policy: PriorResolutionPolicy = PriorResolutionPolicy.HAZARD_FIRST,
        hazard_inheritance_confidence_decay: float = 0.80,
        relation_inheritance_confidence_decay: float = 0.75,
        allow_all_hazard_fallback: bool = True,
        require_explicit: bool = False,
    ) -> ResolvedHazardRelationPrior:
        requested_hazard = _hazard(hazard)
        hazard_registry.get_by_name(requested_hazard)
        relation_registry.get_entry_by_name(relation_name)

        if not isinstance(resolution_policy, PriorResolutionPolicy):
            raise TypeError(
                "resolution_policy must be a PriorResolutionPolicy."
            )

        hazard_decay = _require_finite_number(
            "hazard_inheritance_confidence_decay",
            hazard_inheritance_confidence_decay,
        )
        relation_decay = _require_finite_number(
            "relation_inheritance_confidence_decay",
            relation_inheritance_confidence_decay,
        )
        if not 0.0 <= hazard_decay <= 1.0:
            raise ValueError(
                "hazard_inheritance_confidence_decay must lie in [0, 1]."
            )
        if not 0.0 <= relation_decay <= 1.0:
            raise ValueError(
                "relation_inheritance_confidence_decay must lie in [0, 1]."
            )

        explicit = self.get_explicit(requested_hazard, relation_name)
        if explicit is not None:
            return _resolved_from_prior(
                explicit,
                requested_hazard=requested_hazard,
                requested_relation_name=relation_name,
                resolution_mode=PriorResolutionMode.EXPLICIT,
                hazard_inheritance_distance=0,
                relation_inheritance_distance=0,
                confidence_scale=1.0,
            )

        if require_explicit or resolution_policy == PriorResolutionPolicy.EXPLICIT_ONLY:
            if require_explicit:
                raise KeyError(
                    "Explicit prior required but missing for "
                    f"{requested_hazard.value!r} and {relation_name!r}."
                )
            return self._neutral_default(requested_hazard, relation_name)

        hazard_ancestors = tuple(
            enumerate(
                hazard_registry.ancestors_of(requested_hazard),
                start=1,
            )
        )
        relation_ancestors = tuple(
            enumerate(
                relation_registry.ancestors_of(relation_name),
                start=1,
            )
        )

        exact_hazard_relation_ancestors = tuple(
            (
                requested_hazard,
                ancestor.name,
                0,
                distance,
                PriorResolutionMode.RELATION_ANCESTOR,
            )
            for distance, ancestor in relation_ancestors
        )
        hazard_ancestor_exact_relation = tuple(
            (
                ancestor.name,
                relation_name,
                distance,
                0,
                PriorResolutionMode.HAZARD_ANCESTOR,
            )
            for distance, ancestor in hazard_ancestors
        )
        combined_ancestors = tuple(
            (
                hazard_ancestor.name,
                relation_ancestor.name,
                hazard_distance,
                relation_distance,
                PriorResolutionMode.HAZARD_RELATION_ANCESTOR,
            )
            for hazard_distance, hazard_ancestor in hazard_ancestors
            for relation_distance, relation_ancestor in relation_ancestors
        )

        fallback_candidates: tuple[
            tuple[HazardKind, str, int, int, PriorResolutionMode], ...
        ] = ()
        if (
            allow_all_hazard_fallback
            and requested_hazard != FALLBACK_HAZARD
        ):
            fallback_candidates = (
                (
                    FALLBACK_HAZARD,
                    relation_name,
                    0,
                    0,
                    PriorResolutionMode.ALL_HAZARD,
                ),
                *tuple(
                    (
                        FALLBACK_HAZARD,
                        ancestor.name,
                        0,
                        distance,
                        PriorResolutionMode.ALL_HAZARD_RELATION_ANCESTOR,
                    )
                    for distance, ancestor in relation_ancestors
                ),
            )

        if resolution_policy == PriorResolutionPolicy.HAZARD_FIRST:
            candidates = (
                *exact_hazard_relation_ancestors,
                *hazard_ancestor_exact_relation,
                *combined_ancestors,
                *fallback_candidates,
            )
        else:
            candidates = (
                *hazard_ancestor_exact_relation,
                *exact_hazard_relation_ancestors,
                *combined_ancestors,
                *fallback_candidates,
            )

        for (
            source_hazard,
            source_relation_name,
            hazard_distance,
            relation_distance,
            resolution_mode,
        ) in candidates:
            source_prior = self.get_explicit(
                source_hazard,
                source_relation_name,
            )
            if source_prior is None:
                continue

            confidence_scale = (
                hazard_decay ** hazard_distance
            ) * (
                relation_decay ** relation_distance
            )
            return _resolved_from_prior(
                source_prior,
                requested_hazard=requested_hazard,
                requested_relation_name=relation_name,
                resolution_mode=resolution_mode,
                hazard_inheritance_distance=hazard_distance,
                relation_inheritance_distance=relation_distance,
                confidence_scale=confidence_scale,
            )

        return self._neutral_default(requested_hazard, relation_name)

    def _neutral_default(
        self,
        requested_hazard: HazardKind,
        relation_name: str,
    ) -> ResolvedHazardRelationPrior:
        return ResolvedHazardRelationPrior(
            hazard=requested_hazard,
            relation_name=relation_name,
            prior_mean=self.neutral_prior_mean,
            confidence=0.0,
            initialization_allowed=False,
            regularization_allowed=False,
            evidence_type=PriorEvidenceType.PROVISIONAL_ONTOLOGY,
            rationale=(
                "No explicit or inherited prior was available; a neutral "
                "non-operative default was used."
            ),
            caveat=None,
            resolution_mode=PriorResolutionMode.NEUTRAL_DEFAULT,
            source_hazard=None,
            source_relation_name=None,
            hazard_inheritance_distance=0,
            relation_inheritance_distance=0,
        )

    def compile(
        self,
        compiled_relation_registry: CompiledRelationRegistry,
        *,
        source_hazard_registry: HazardRegistry,
        source_relation_registry: RelationRegistry,
        application_context: PriorApplicationContext,
        hazards: Sequence[HazardKind | str],
        resolution_policy: PriorResolutionPolicy = PriorResolutionPolicy.HAZARD_FIRST,
        hazard_inheritance_confidence_decay: float = 0.80,
        relation_inheritance_confidence_decay: float = 0.75,
        allow_all_hazard_fallback: bool = True,
        require_explicit: bool = False,
        require_queryable_hazards: bool = True,
        require_training_supported_hazards: bool = False,
        allow_partially_data_backed: bool = False,
        allow_fallback_hazard: bool = False,
    ) -> CompiledHazardRelationPriors:
        if not isinstance(compiled_relation_registry, CompiledRelationRegistry):
            raise TypeError(
                "compiled_relation_registry must be a CompiledRelationRegistry."
            )

        self.applicability_scope.assert_applicable(application_context)
        self.validate_against_sources(
            hazard_registry=source_hazard_registry,
            relation_registry=source_relation_registry,
            require_complete=False,
            require_current_versions=True,
        )
        compiled_relation_registry.assert_matches_source_registry(
            source_relation_registry,
            require_operational_match=False,
        )

        requested_hazards = tuple(_hazard(value) for value in hazards)
        if not requested_hazards:
            raise ValueError("At least one hazard must be explicitly requested.")

        duplicate_hazards = sorted(
            hazard.value
            for hazard, count in Counter(requested_hazards).items()
            if count > 1
        )
        if duplicate_hazards:
            raise ValueError(
                f"Duplicate requested hazards: {duplicate_hazards}."
            )

        for hazard_value in requested_hazards:
            if require_training_supported_hazards:
                source_hazard_registry.assert_training_supported_hazard(
                    hazard_value,
                    allow_partially_data_backed=allow_partially_data_backed,
                )
            elif require_queryable_hazards:
                source_hazard_registry.assert_queryable_hazard(
                    hazard_value,
                    allow_fallback=allow_fallback_hazard,
                )
            else:
                source_hazard_registry.get_by_name(hazard_value)

        resolved_rows = tuple(
            tuple(
                self.resolve(
                    hazard_value,
                    entry.name,
                    hazard_registry=source_hazard_registry,
                    relation_registry=source_relation_registry,
                    resolution_policy=resolution_policy,
                    hazard_inheritance_confidence_decay=(
                        hazard_inheritance_confidence_decay
                    ),
                    relation_inheritance_confidence_decay=(
                        relation_inheritance_confidence_decay
                    ),
                    allow_all_hazard_fallback=allow_all_hazard_fallback,
                    require_explicit=require_explicit,
                )
                for entry in compiled_relation_registry.entries
            )
            for hazard_value in requested_hazards
        )

        effective_rows = tuple(
            tuple(
                cell.effective_initialization_mean(
                    neutral_prior_mean=self.neutral_prior_mean
                )
                for cell in row
            )
            for row in resolved_rows
        )
        regularization_mask = tuple(
            tuple(
                self.regularization_approved
                and cell.regularization_allowed
                and cell.confidence > 0.0
                for cell in row
            )
            for row in resolved_rows
        )

        return CompiledHazardRelationPriors(
            hazard_names=tuple(hazard.value for hazard in requested_hazards),
            stable_hazard_ids=tuple(
                source_hazard_registry.stable_id_for(hazard)
                for hazard in requested_hazards
            ),
            relation_names=compiled_relation_registry.relation_names,
            stable_relation_ids=compiled_relation_registry.stable_relation_ids,
            prior_mean_matrix=tuple(
                tuple(cell.prior_mean for cell in row)
                for row in resolved_rows
            ),
            confidence_matrix=tuple(
                tuple(cell.confidence for cell in row)
                for row in resolved_rows
            ),
            effective_initialization_mean_matrix=effective_rows,
            initialization_mask=tuple(
                tuple(
                    cell.initialization_allowed and cell.confidence > 0.0
                    for cell in row
                )
                for row in resolved_rows
            ),
            regularization_mask=regularization_mask,
            resolution_mode_matrix=tuple(
                tuple(cell.resolution_mode.value for cell in row)
                for row in resolved_rows
            ),
            source_hazard_matrix=tuple(
                tuple(
                    cell.source_hazard.value
                    if cell.source_hazard is not None
                    else None
                    for cell in row
                )
                for row in resolved_rows
            ),
            source_relation_matrix=tuple(
                tuple(cell.source_relation_name for cell in row)
                for row in resolved_rows
            ),
            hazard_inheritance_distance_matrix=tuple(
                tuple(cell.hazard_inheritance_distance for cell in row)
                for row in resolved_rows
            ),
            relation_inheritance_distance_matrix=tuple(
                tuple(cell.relation_inheritance_distance for cell in row)
                for row in resolved_rows
            ),
            neutral_prior_mean=self.neutral_prior_mean,
            resolution_policy=resolution_policy,
            hazard_inheritance_confidence_decay=(
                hazard_inheritance_confidence_decay
            ),
            relation_inheritance_confidence_decay=(
                relation_inheritance_confidence_decay
            ),
            allow_all_hazard_fallback=allow_all_hazard_fallback,
            application_context=application_context,
            source_applicability_scope_fingerprint=(
                self.applicability_scope.fingerprint()
            ),
            source_hazard_registry_name=source_hazard_registry.registry_name,
            source_hazard_registry_version=source_hazard_registry.registry_version,
            source_hazard_semantic_fingerprint=(
                source_hazard_registry.semantic_fingerprint()
            ),
            source_hazard_compatibility_fingerprint=(
                source_hazard_registry.compatibility_fingerprint()
            ),
            source_hazard_operational_fingerprint=(
                source_hazard_registry.operational_fingerprint()
            ),
            source_prior_registry_name=self.registry_name,
            source_prior_registry_version=self.registry_version,
            source_prior_registry_fingerprint=self.fingerprint(),
            source_prior_registry_status=self.registry_status,
            source_relation_registry_name=(
                source_relation_registry.registry_name
            ),
            source_relation_registry_version=(
                source_relation_registry.registry_version
            ),
            source_relation_semantic_fingerprint=(
                source_relation_registry.semantic_fingerprint()
            ),
            source_compiled_relation_fingerprint=(
                compiled_relation_registry.fingerprint()
            ),
        )

    def canonical_dict(self) -> dict[str, Any]:
        return {
            "registry_name": self.registry_name,
            "registry_version": self.registry_version,
            "schema_version": self.schema_version,
            "registry_status": self.registry_status.value,
            "regularization_approved": self.regularization_approved,
            "provisional_notice": self.provisional_notice,
            "neutral_prior_mean": self.neutral_prior_mean,
            "applicability_scope": self.applicability_scope.to_dict(),
            "source_hazard_registry": self.source_hazard_registry.to_dict(),
            "source_relation_registry_name": self.source_relation_registry_name,
            "source_relation_registry_version": self.source_relation_registry_version,
            "source_relation_semantic_fingerprint": (
                self.source_relation_semantic_fingerprint
            ),
            "source_relation_names": list(self.source_relation_names),
            "source_stable_relation_ids": list(self.source_stable_relation_ids),
            "priors": [prior.to_dict() for prior in self.priors],
        }

    def fingerprint(self) -> str:
        return _fingerprint(self.canonical_dict())

    def to_dict(self) -> dict[str, Any]:
        return {**self.canonical_dict(), "fingerprint": self.fingerprint()}

    @classmethod
    def from_dict(
        cls,
        payload: Mapping[str, Any],
        *,
        hazard_registry: HazardRegistry | None = None,
        relation_registry: RelationRegistry | None = None,
        require_current_versions: bool = False,
        require_complete: bool = False,
    ) -> HazardRelationPriorRegistry:
        mapping = dict(
            _require_mapping("HazardRelationPriorRegistry", payload)
        )
        serialized_fingerprint = mapping.pop("fingerprint", None)
        _reject_unknown_fields(cls, mapping)

        raw_priors = mapping["priors"]
        if not isinstance(raw_priors, list):
            raise TypeError("priors must be a list.")
        mapping["priors"] = tuple(
            HazardRelationPrior.from_dict(
                _require_mapping(f"priors[{index}]", value)
            )
            for index, value in enumerate(raw_priors)
        )
        mapping["applicability_scope"] = PriorApplicabilityScope.from_dict(
            _require_mapping(
                "applicability_scope", mapping["applicability_scope"]
            )
        )
        mapping["source_hazard_registry"] = HazardRegistry.from_dict(
            _require_mapping(
                "source_hazard_registry", mapping["source_hazard_registry"]
            )
        )
        mapping["source_relation_names"] = _as_tuple(
            "source_relation_names", mapping["source_relation_names"]
        )
        mapping["source_stable_relation_ids"] = _as_tuple(
            "source_stable_relation_ids",
            mapping["source_stable_relation_ids"],
        )
        mapping["registry_status"] = PriorRegistryStatus(
            mapping["registry_status"]
        )

        registry = cls(**mapping)
        if (
            serialized_fingerprint is not None
            and serialized_fingerprint != registry.fingerprint()
        ):
            raise ValueError(
                "Serialized prior-registry fingerprint does not match the "
                "reconstructed registry."
            )

        if hazard_registry is not None and relation_registry is not None:
            registry.validate_against_sources(
                hazard_registry=hazard_registry,
                relation_registry=relation_registry,
                require_complete=require_complete,
                require_current_versions=require_current_versions,
            )
        return registry


# =============================================================================
# Dense compiled priors
# =============================================================================


@dataclass(slots=True, frozen=True)
class CompiledHazardRelationPriors:
    """Dense immutable prior matrices aligned to one compiled relation registry."""

    hazard_names: tuple[str, ...]
    stable_hazard_ids: tuple[int, ...]
    relation_names: tuple[str, ...]
    stable_relation_ids: tuple[int, ...]

    prior_mean_matrix: tuple[tuple[float, ...], ...]
    confidence_matrix: tuple[tuple[float, ...], ...]
    effective_initialization_mean_matrix: tuple[tuple[float, ...], ...]
    initialization_mask: tuple[tuple[bool, ...], ...]
    regularization_mask: tuple[tuple[bool, ...], ...]
    resolution_mode_matrix: tuple[tuple[str, ...], ...]
    source_hazard_matrix: tuple[tuple[str | None, ...], ...]
    source_relation_matrix: tuple[tuple[str | None, ...], ...]
    hazard_inheritance_distance_matrix: tuple[tuple[int, ...], ...]
    relation_inheritance_distance_matrix: tuple[tuple[int, ...], ...]

    neutral_prior_mean: float
    resolution_policy: PriorResolutionPolicy
    hazard_inheritance_confidence_decay: float
    relation_inheritance_confidence_decay: float
    allow_all_hazard_fallback: bool

    application_context: PriorApplicationContext
    source_applicability_scope_fingerprint: str

    source_hazard_registry_name: str
    source_hazard_registry_version: str
    source_hazard_semantic_fingerprint: str
    source_hazard_compatibility_fingerprint: str
    source_hazard_operational_fingerprint: str

    source_prior_registry_name: str
    source_prior_registry_version: str
    source_prior_registry_fingerprint: str
    source_prior_registry_status: PriorRegistryStatus

    source_relation_registry_name: str
    source_relation_registry_version: str
    source_relation_semantic_fingerprint: str
    source_compiled_relation_fingerprint: str

    schema_version: str = COMPILED_HAZARD_PRIOR_SCHEMA_VERSION

    def __post_init__(self) -> None:
        self.validate()

    @property
    def num_hazards(self) -> int:
        return len(self.hazard_names)

    @property
    def num_relations(self) -> int:
        return len(self.relation_names)

    def validate(self) -> None:
        if not self.hazard_names or not self.relation_names:
            raise ValueError(
                "Compiled priors require at least one hazard and relation."
            )
        _require_unique_strings("hazard_names", self.hazard_names)
        _require_unique_strings("relation_names", self.relation_names)
        for hazard_name in self.hazard_names:
            HazardKind(hazard_name)

        if len(self.stable_hazard_ids) != self.num_hazards:
            raise ValueError("stable_hazard_ids must align with hazard_names.")
        if len(set(self.stable_hazard_ids)) != self.num_hazards:
            raise ValueError("stable_hazard_ids must be unique.")
        if len(self.stable_relation_ids) != self.num_relations:
            raise ValueError(
                "stable_relation_ids must align with relation_names."
            )
        if len(set(self.stable_relation_ids)) != self.num_relations:
            raise ValueError("stable_relation_ids must be unique.")

        matrices: Mapping[str, Sequence[Sequence[Any]]] = {
            "prior_mean_matrix": self.prior_mean_matrix,
            "confidence_matrix": self.confidence_matrix,
            "effective_initialization_mean_matrix": (
                self.effective_initialization_mean_matrix
            ),
            "initialization_mask": self.initialization_mask,
            "regularization_mask": self.regularization_mask,
            "resolution_mode_matrix": self.resolution_mode_matrix,
            "source_hazard_matrix": self.source_hazard_matrix,
            "source_relation_matrix": self.source_relation_matrix,
            "hazard_inheritance_distance_matrix": (
                self.hazard_inheritance_distance_matrix
            ),
            "relation_inheritance_distance_matrix": (
                self.relation_inheritance_distance_matrix
            ),
        }
        for matrix_name, matrix in matrices.items():
            if len(matrix) != self.num_hazards:
                raise ValueError(
                    f"{matrix_name} must have {self.num_hazards} rows."
                )
            for row_index, row in enumerate(matrix):
                if len(row) != self.num_relations:
                    raise ValueError(
                        f"{matrix_name}[{row_index}] must have "
                        f"{self.num_relations} columns."
                    )

        _require_probability(
            "neutral_prior_mean", self.neutral_prior_mean, include_endpoints=False
        )
        for name, decay in (
            (
                "hazard_inheritance_confidence_decay",
                self.hazard_inheritance_confidence_decay,
            ),
            (
                "relation_inheritance_confidence_decay",
                self.relation_inheritance_confidence_decay,
            ),
        ):
            converted = _require_finite_number(name, decay)
            if not 0.0 <= converted <= 1.0:
                raise ValueError(f"{name} must lie in [0, 1].")

        if not isinstance(self.resolution_policy, PriorResolutionPolicy):
            raise TypeError(
                "resolution_policy must be a PriorResolutionPolicy."
            )
        if not isinstance(self.application_context, PriorApplicationContext):
            raise TypeError(
                "application_context must be a PriorApplicationContext."
            )
        if not isinstance(self.source_prior_registry_status, PriorRegistryStatus):
            raise TypeError(
                "source_prior_registry_status must be a PriorRegistryStatus."
            )

        for row_index in range(self.num_hazards):
            for column_index in range(self.num_relations):
                prior_mean = self.prior_mean_matrix[row_index][column_index]
                confidence = self.confidence_matrix[row_index][column_index]
                effective_mean = self.effective_initialization_mean_matrix[
                    row_index
                ][column_index]
                _require_probability(
                    "compiled prior mean", prior_mean, include_endpoints=False
                )
                _require_probability("compiled confidence", confidence)
                _require_probability(
                    "effective initialization mean",
                    effective_mean,
                    include_endpoints=False,
                )
                PriorResolutionMode(
                    self.resolution_mode_matrix[row_index][column_index]
                )
                source_hazard = self.source_hazard_matrix[row_index][column_index]
                if source_hazard is not None:
                    HazardKind(source_hazard)

                if (
                    self.regularization_mask[row_index][column_index]
                    and (
                        confidence == 0.0
                        or self.source_prior_registry_status
                        == PriorRegistryStatus.PROVISIONAL
                    )
                ):
                    raise ValueError(
                        "Provisional or zero-confidence cells cannot regularize."
                    )
                if not self.initialization_mask[row_index][column_index]:
                    if abs(effective_mean - self.neutral_prior_mean) > 1e-12:
                        raise ValueError(
                            "Initialization-disabled cells must retain the "
                            "neutral effective mean."
                        )

        for name, value in (
            (
                "source_applicability_scope_fingerprint",
                self.source_applicability_scope_fingerprint,
            ),
            ("source_hazard_registry_name", self.source_hazard_registry_name),
            (
                "source_hazard_registry_version",
                self.source_hazard_registry_version,
            ),
            (
                "source_hazard_semantic_fingerprint",
                self.source_hazard_semantic_fingerprint,
            ),
            (
                "source_hazard_compatibility_fingerprint",
                self.source_hazard_compatibility_fingerprint,
            ),
            (
                "source_hazard_operational_fingerprint",
                self.source_hazard_operational_fingerprint,
            ),
            ("source_prior_registry_name", self.source_prior_registry_name),
            (
                "source_prior_registry_version",
                self.source_prior_registry_version,
            ),
            (
                "source_prior_registry_fingerprint",
                self.source_prior_registry_fingerprint,
            ),
            (
                "source_relation_registry_name",
                self.source_relation_registry_name,
            ),
            (
                "source_relation_registry_version",
                self.source_relation_registry_version,
            ),
            (
                "source_relation_semantic_fingerprint",
                self.source_relation_semantic_fingerprint,
            ),
            (
                "source_compiled_relation_fingerprint",
                self.source_compiled_relation_fingerprint,
            ),
            ("schema_version", self.schema_version),
        ):
            _require_nonempty_string(name, value)

    def gate_bias_logit_matrix(
        self,
        *,
        activation: GateInitializationActivation = (
            GateInitializationActivation.SIGMOID
        ),
        epsilon: float = 1e-4,
    ) -> tuple[tuple[float, ...], ...]:
        if activation != GateInitializationActivation.SIGMOID:
            raise ValueError(
                "Compiled prior logits currently support sigmoid gates only."
            )
        epsilon_value = _require_finite_number("epsilon", epsilon)
        if not 0.0 < epsilon_value < 0.5:
            raise ValueError("epsilon must lie strictly between zero and 0.5.")

        rows: list[tuple[float, ...]] = []
        for hazard_index in range(self.num_hazards):
            row: list[float] = []
            for relation_index in range(self.num_relations):
                if not self.initialization_mask[hazard_index][relation_index]:
                    row.append(0.0)
                    continue
                mean = self.effective_initialization_mean_matrix[hazard_index][
                    relation_index
                ]
                clipped = min(1.0 - epsilon_value, max(epsilon_value, mean))
                row.append(math.log(clipped / (1.0 - clipped)))
            rows.append(tuple(row))
        return tuple(rows)

    def regularization_weight_matrix(self) -> tuple[tuple[float, ...], ...]:
        return tuple(
            tuple(
                self.confidence_matrix[hazard_index][relation_index]
                if self.regularization_mask[hazard_index][relation_index]
                else 0.0
                for relation_index in range(self.num_relations)
            )
            for hazard_index in range(self.num_hazards)
        )

    def assert_matches_sources(
        self,
        *,
        prior_registry: HazardRelationPriorRegistry,
        hazard_registry: HazardRegistry,
        relation_registry: RelationRegistry,
        compiled_relation_registry: CompiledRelationRegistry,
    ) -> None:
        """Verify every source identity claimed by this compiled artifact."""

        prior_registry.validate_against_sources(
            hazard_registry=hazard_registry,
            relation_registry=relation_registry,
            require_complete=False,
            require_current_versions=False,
        )
        prior_registry.applicability_scope.assert_applicable(
            self.application_context
        )
        compiled_relation_registry.assert_matches_source_registry(
            relation_registry,
            require_operational_match=False,
        )

        mismatches: list[str] = []

        if prior_registry.fingerprint() != self.source_prior_registry_fingerprint:
            mismatches.append("prior registry fingerprint")
        if (
            prior_registry.applicability_scope.fingerprint()
            != self.source_applicability_scope_fingerprint
        ):
            mismatches.append("applicability scope fingerprint")
        if (
            hazard_registry.semantic_fingerprint()
            != self.source_hazard_semantic_fingerprint
        ):
            mismatches.append("hazard semantic fingerprint")
        if (
            hazard_registry.compatibility_fingerprint()
            != self.source_hazard_compatibility_fingerprint
        ):
            mismatches.append("hazard compatibility fingerprint")
        if (
            hazard_registry.operational_fingerprint()
            != self.source_hazard_operational_fingerprint
        ):
            mismatches.append("hazard operational fingerprint")
        if (
            relation_registry.semantic_fingerprint()
            != self.source_relation_semantic_fingerprint
        ):
            mismatches.append("relation semantic fingerprint")
        if (
            compiled_relation_registry.fingerprint()
            != self.source_compiled_relation_fingerprint
        ):
            mismatches.append("compiled relation fingerprint")
        if compiled_relation_registry.relation_names != self.relation_names:
            mismatches.append("relation ordering")
        if (
            compiled_relation_registry.stable_relation_ids
            != self.stable_relation_ids
        ):
            mismatches.append("stable relation IDs")
        if tuple(
            hazard_registry.stable_id_for(name)
            for name in self.hazard_names
        ) != self.stable_hazard_ids:
            mismatches.append("stable hazard IDs")

        if mismatches:
            raise ValueError(
                "Compiled hazard priors do not match their claimed sources: "
                f"{mismatches}."
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "hazard_names": list(self.hazard_names),
            "stable_hazard_ids": list(self.stable_hazard_ids),
            "relation_names": list(self.relation_names),
            "stable_relation_ids": list(self.stable_relation_ids),
            "prior_mean_matrix": [list(row) for row in self.prior_mean_matrix],
            "confidence_matrix": [list(row) for row in self.confidence_matrix],
            "effective_initialization_mean_matrix": [
                list(row)
                for row in self.effective_initialization_mean_matrix
            ],
            "initialization_mask": [
                list(row) for row in self.initialization_mask
            ],
            "regularization_mask": [
                list(row) for row in self.regularization_mask
            ],
            "resolution_mode_matrix": [
                list(row) for row in self.resolution_mode_matrix
            ],
            "source_hazard_matrix": [
                list(row) for row in self.source_hazard_matrix
            ],
            "source_relation_matrix": [
                list(row) for row in self.source_relation_matrix
            ],
            "hazard_inheritance_distance_matrix": [
                list(row)
                for row in self.hazard_inheritance_distance_matrix
            ],
            "relation_inheritance_distance_matrix": [
                list(row)
                for row in self.relation_inheritance_distance_matrix
            ],
            "neutral_prior_mean": self.neutral_prior_mean,
            "resolution_policy": self.resolution_policy.value,
            "hazard_inheritance_confidence_decay": (
                self.hazard_inheritance_confidence_decay
            ),
            "relation_inheritance_confidence_decay": (
                self.relation_inheritance_confidence_decay
            ),
            "allow_all_hazard_fallback": self.allow_all_hazard_fallback,
            "application_context": self.application_context.to_dict(),
            "source_applicability_scope_fingerprint": (
                self.source_applicability_scope_fingerprint
            ),
            "source_hazard_registry_name": self.source_hazard_registry_name,
            "source_hazard_registry_version": self.source_hazard_registry_version,
            "source_hazard_semantic_fingerprint": (
                self.source_hazard_semantic_fingerprint
            ),
            "source_hazard_compatibility_fingerprint": (
                self.source_hazard_compatibility_fingerprint
            ),
            "source_hazard_operational_fingerprint": (
                self.source_hazard_operational_fingerprint
            ),
            "source_prior_registry_name": self.source_prior_registry_name,
            "source_prior_registry_version": self.source_prior_registry_version,
            "source_prior_registry_fingerprint": (
                self.source_prior_registry_fingerprint
            ),
            "source_prior_registry_status": (
                self.source_prior_registry_status.value
            ),
            "source_relation_registry_name": self.source_relation_registry_name,
            "source_relation_registry_version": (
                self.source_relation_registry_version
            ),
            "source_relation_semantic_fingerprint": (
                self.source_relation_semantic_fingerprint
            ),
            "source_compiled_relation_fingerprint": (
                self.source_compiled_relation_fingerprint
            ),
        }

    def fingerprint(self) -> str:
        return _fingerprint(self.to_dict())

    @classmethod
    def from_dict(
        cls,
        payload: Mapping[str, Any],
    ) -> CompiledHazardRelationPriors:
        mapping = dict(
            _require_mapping("CompiledHazardRelationPriors", payload)
        )
        _reject_unknown_fields(cls, mapping)

        for field_name in (
            "hazard_names",
            "stable_hazard_ids",
            "relation_names",
            "stable_relation_ids",
        ):
            mapping[field_name] = _as_tuple(field_name, mapping[field_name])

        for field_name in (
            "prior_mean_matrix",
            "confidence_matrix",
            "effective_initialization_mean_matrix",
            "initialization_mask",
            "regularization_mask",
            "resolution_mode_matrix",
            "source_hazard_matrix",
            "source_relation_matrix",
            "hazard_inheritance_distance_matrix",
            "relation_inheritance_distance_matrix",
        ):
            raw_matrix = _as_tuple(field_name, mapping[field_name])
            mapping[field_name] = tuple(
                _as_tuple(f"{field_name}[{row_index}]", row)
                for row_index, row in enumerate(raw_matrix)
            )

        mapping["resolution_policy"] = PriorResolutionPolicy(
            mapping["resolution_policy"]
        )
        mapping["source_prior_registry_status"] = PriorRegistryStatus(
            mapping["source_prior_registry_status"]
        )
        mapping["application_context"] = PriorApplicationContext.from_dict(
            _require_mapping(
                "application_context", mapping["application_context"]
            )
        )
        return cls(**mapping)


# =============================================================================
# Default provisional profiles
# =============================================================================


def _provisional_cell(
    strength: PriorStrength,
    confidence: float,
    rationale: str,
    *,
    caveat: str | None = None,
    initialization_allowed: bool = True,
) -> PriorCellDefinition:
    return PriorCellDefinition(
        strength=strength,
        confidence=confidence,
        evidence_type=PriorEvidenceType.PROVISIONAL_ONTOLOGY,
        rationale=rationale,
        caveat=caveat,
        initialization_allowed=initialization_allowed,
        regularization_allowed=False,
    )


def _control_cell(rationale: str) -> PriorCellDefinition:
    return PriorCellDefinition(
        strength=PriorStrength.MEDIUM,
        confidence=0.0,
        evidence_type=PriorEvidenceType.CONTROL,
        rationale=rationale,
        caveat=(
            "This is a control topology and must not receive mechanistic "
            "hazard preference."
        ),
        initialization_allowed=False,
        regularization_allowed=False,
    )


def _profile(
    relation_name: str,
    *,
    default: PriorCellDefinition,
    overrides: Mapping[HazardKind, PriorCellDefinition] | None = None,
) -> RelationPriorProfile:
    return RelationPriorProfile(
        relation_name=relation_name,
        default_definition=default,
        overrides=tuple(
            (hazard, definition)
            for hazard, definition in (
                overrides.items() if overrides is not None else ()
            )
        ),
    )


DEFAULT_RELATION_PRIOR_PROFILES: Final[
    tuple[RelationPriorProfile, ...]
] = (
    _profile(
        C.REL_RANDOM_PLACEBO,
        default=_control_cell(
            "Random edges are topology controls rather than hazard pathways."
        ),
    ),
    _profile(
        C.REL_CENTROID_KNN,
        default=_control_cell(
            "Generic nearest-neighbor edges are geometric controls rather "
            "than hazard mechanisms."
        ),
    ),
    _profile(
        C.REL_SPATIAL_ADJACENCY,
        default=_provisional_cell(
            PriorStrength.MEDIUM,
            0.20,
            "Spatial contiguity may capture local spillover or shared urban "
            "conditions, but it is less mechanistic than functional edges.",
        ),
        overrides={
            HazardKind.FLOOD: _provisional_cell(
                PriorStrength.MEDIUM_HIGH,
                0.35,
                "Flood burden may cluster across contiguous areas through "
                "shared terrain, drainage, and hydrological conditions.",
            ),
            HazardKind.RIVERINE_FLOOD: _provisional_cell(
                PriorStrength.MEDIUM_HIGH,
                0.35,
                "Riverine inundation may affect adjacent floodplain areas.",
            ),
            HazardKind.PLUVIAL_FLOOD: _provisional_cell(
                PriorStrength.HIGH,
                0.45,
                "Pluvial runoff and ponding may propagate across contiguous "
                "urban terrain and drainage catchments.",
            ),
            HazardKind.ROAD_DISRUPTION: _provisional_cell(
                PriorStrength.MEDIUM_HIGH,
                0.30,
                "Local transport disruptions may affect neighboring tracts.",
            ),
            HazardKind.WINTER_STORM: _provisional_cell(
                PriorStrength.MEDIUM_HIGH,
                0.30,
                "Winter-weather burden often clusters spatially.",
            ),
            HazardKind.SNOWSTORM: _provisional_cell(
                PriorStrength.MEDIUM_HIGH,
                0.30,
                "Snow accumulation and accessibility impacts may cluster "
                "across neighboring tracts.",
            ),
            HazardKind.FREEZING_RAIN: _provisional_cell(
                PriorStrength.MEDIUM_HIGH,
                0.30,
                "Ice accumulation and local accessibility impacts may be "
                "spatially clustered.",
            ),
            HazardKind.CIVIL_SECURITY_EVENT: _provisional_cell(
                PriorStrength.MEDIUM_HIGH,
                0.30,
                "Broad civil-security impacts may exhibit local clustering.",
            ),
        },
    ),
    _profile(
        C.REL_TEMPORAL_MEMORY,
        default=_provisional_cell(
            PriorStrength.HIGH,
            0.40,
            "Historical persistence, antecedent conditions, and seasonality "
            "may matter across urban hazards.",
        ),
        overrides={
            HazardKind.FLOOD: _provisional_cell(
                PriorStrength.HIGH,
                0.55,
                "Flood and drainage burden may exhibit recurrence, antecedent "
                "wetness, and seasonal persistence.",
            ),
            HazardKind.PLUVIAL_FLOOD: _provisional_cell(
                PriorStrength.VERY_HIGH,
                0.60,
                "Antecedent rainfall, drainage saturation, and repeated local "
                "ponding are expected to matter strongly.",
            ),
            HazardKind.HEAT: _provisional_cell(
                PriorStrength.VERY_HIGH,
                0.60,
                "Heat consequences depend on accumulated thermal load, hot "
                "nights, and incomplete recovery from preceding days.",
            ),
            HazardKind.WINTER_STORM: _provisional_cell(
                PriorStrength.HIGH,
                0.50,
                "Snow accumulation and freeze-thaw history create strong "
                "antecedent-state dependence.",
            ),
            HazardKind.SNOWSTORM: _provisional_cell(
                PriorStrength.HIGH,
                0.50,
                "Recent snow accumulation and incomplete clearing may shape "
                "current disruption.",
            ),
            HazardKind.FREEZING_RAIN: _provisional_cell(
                PriorStrength.VERY_HIGH,
                0.55,
                "Preceding temperature, melt, precipitation, and refreezing "
                "history may strongly influence ice burden.",
            ),
            HazardKind.CIVIL_SECURITY_EVENT: _provisional_cell(
                PriorStrength.HIGH,
                0.45,
                "Recent event history may inform broad civil-security burden.",
            ),
        },
    ),
    _profile(
        C.REL_HISTORICAL_EVENT_PROPAGATION,
        default=_provisional_cell(
            PriorStrength.LOW_MEDIUM,
            0.15,
            "Historical event propagation may matter but is highly task- and "
            "construction-dependent.",
            caveat=(
                "This relation is leakage-sensitive and may partly encode "
                "reporting persistence."
            ),
        ),
        overrides={
            HazardKind.FLOOD: _provisional_cell(
                PriorStrength.HIGH,
                0.40,
                "Repeated flood and drainage events may identify persistent "
                "local pathways.",
            ),
            HazardKind.PLUVIAL_FLOOD: _provisional_cell(
                PriorStrength.HIGH,
                0.45,
                "Repeated drainage and ponding events may identify persistent "
                "urban runoff pathways.",
            ),
            HazardKind.CIVIL_SECURITY_EVENT: _provisional_cell(
                PriorStrength.HIGH,
                0.35,
                "Recent event concentrations may identify broad event "
                "propagation patterns.",
            ),
            HazardKind.WINTER_STORM: _provisional_cell(
                PriorStrength.MEDIUM_HIGH,
                0.25,
                "Recent winter disruptions may reveal persistent operational "
                "stress and incomplete recovery.",
            ),
        },
    ),
    _profile(
        C.REL_HYDROLOGICAL_EXPOSURE,
        default=_provisional_cell(
            PriorStrength.VERY_LOW,
            0.10,
            "Hydrological exposure is not expected to dominate hazards "
            "unrelated to water.",
        ),
        overrides={
            HazardKind.FLOOD: _provisional_cell(
                PriorStrength.VERY_HIGH,
                0.65,
                "Hydrological exposure is a direct flood pathway.",
            ),
            HazardKind.RIVERINE_FLOOD: _provisional_cell(
                PriorStrength.VERY_HIGH,
                0.75,
                "Water-body and river connectivity are direct riverine-flood "
                "pathways.",
            ),
            HazardKind.PLUVIAL_FLOOD: _provisional_cell(
                PriorStrength.MEDIUM_HIGH,
                0.35,
                "Hydrological context may matter, though pluvial flooding is "
                "primarily driven by rainfall, runoff, and drainage capacity.",
            ),
            HazardKind.ROAD_DISRUPTION: _provisional_cell(
                PriorStrength.MEDIUM,
                0.25,
                "Hydrological exposure may contribute to road closure.",
            ),
        },
    ),
    _profile(
        C.REL_FLOOD_ZONE_EXPOSURE,
        default=_provisional_cell(
            PriorStrength.VERY_LOW,
            0.10,
            "Mapped flood-zone overlap has narrow relevance outside flooding.",
        ),
        overrides={
            HazardKind.FLOOD: _provisional_cell(
                PriorStrength.HIGH,
                0.55,
                "Flood-zone overlap is a direct general flood-exposure signal.",
            ),
            HazardKind.RIVERINE_FLOOD: _provisional_cell(
                PriorStrength.VERY_HIGH,
                0.75,
                "Floodplain overlap is a direct riverine-flood signal.",
            ),
            HazardKind.PLUVIAL_FLOOD: _provisional_cell(
                PriorStrength.MEDIUM,
                0.20,
                "Pluvial flooding may occur outside mapped river flood zones.",
                caveat=(
                    "River-oriented flood maps may miss rainfall-driven urban "
                    "ponding."
                ),
            ),
        },
    ),
    _profile(
        C.REL_LOW_ELEVATION_EXPOSURE,
        default=_provisional_cell(
            PriorStrength.VERY_LOW,
            0.10,
            "Low terrain position has limited relevance outside water hazards.",
        ),
        overrides={
            HazardKind.FLOOD: _provisional_cell(
                PriorStrength.HIGH,
                0.55,
                "Low terrain may increase susceptibility to water accumulation.",
            ),
            HazardKind.RIVERINE_FLOOD: _provisional_cell(
                PriorStrength.HIGH,
                0.50,
                "Low floodplain terrain may increase riverine exposure.",
            ),
            HazardKind.PLUVIAL_FLOOD: _provisional_cell(
                PriorStrength.VERY_HIGH,
                0.65,
                "Topographic depressions and low points are direct ponding "
                "pathways for pluvial flooding.",
            ),
        },
    ),
    _profile(
        C.REL_HEAT_EXPOSURE,
        default=_provisional_cell(
            PriorStrength.VERY_LOW,
            0.10,
            "Thermal exposure has narrow relevance outside heat hazards.",
        ),
        overrides={
            HazardKind.HEAT: _provisional_cell(
                PriorStrength.VERY_HIGH,
                0.70,
                "Heat-island exposure is a direct heat-hazard pathway.",
            ),
            HazardKind.CIVIL_SECURITY_EVENT: _provisional_cell(
                PriorStrength.MEDIUM,
                0.20,
                "Extreme heat may contribute to broad civil-security burden.",
            ),
        },
    ),
    _profile(
        C.REL_CANOPY_PROTECTION,
        default=_provisional_cell(
            PriorStrength.VERY_LOW,
            0.10,
            "Canopy protection has limited relevance for most non-heat hazards.",
            caveat=(
                "High relevance indicates a protective pathway, not a positive "
                "risk effect."
            ),
        ),
        overrides={
            HazardKind.HEAT: _provisional_cell(
                PriorStrength.VERY_HIGH,
                0.65,
                "Canopy and green space are expected to provide thermal "
                "protection.",
            ),
            HazardKind.PLUVIAL_FLOOD: _provisional_cell(
                PriorStrength.MEDIUM,
                0.20,
                "Vegetation may provide secondary runoff-retention benefits.",
            ),
        },
    ),
    _profile(
        C.REL_SERVICE_ACCESS,
        default=_provisional_cell(
            PriorStrength.MEDIUM_HIGH,
            0.35,
            "Service accessibility may influence adaptive capacity and response "
            "across hazards.",
        ),
        overrides={
            HazardKind.HEAT: _provisional_cell(
                PriorStrength.HIGH,
                0.50,
                "Access to health and cooling services may reduce heat burden.",
            ),
            HazardKind.ROAD_DISRUPTION: _provisional_cell(
                PriorStrength.HIGH,
                0.50,
                "Service reachability is directly affected by transport "
                "disruption.",
            ),
            HazardKind.CIVIL_SECURITY_EVENT: _provisional_cell(
                PriorStrength.HIGH,
                0.50,
                "Access to emergency services may matter strongly.",
            ),
            HazardKind.WINTER_STORM: _provisional_cell(
                PriorStrength.HIGH,
                0.45,
                "Winter weather may degrade access to emergency and essential "
                "services.",
            ),
            HazardKind.SNOWSTORM: _provisional_cell(
                PriorStrength.HIGH,
                0.50,
                "Snow accumulation may reduce service accessibility.",
            ),
            HazardKind.FREEZING_RAIN: _provisional_cell(
                PriorStrength.HIGH,
                0.50,
                "Ice conditions may reduce access to emergency and essential "
                "services.",
            ),
        },
    ),
    _profile(
        C.REL_COOLING_ACCESS,
        default=_provisional_cell(
            PriorStrength.VERY_LOW,
            0.10,
            "Cooling access has narrow relevance outside heat response.",
        ),
        overrides={
            HazardKind.HEAT: _provisional_cell(
                PriorStrength.HIGH,
                0.55,
                "Travel access to cooling resources is an adaptive-capacity "
                "pathway during heat events.",
            ),
            HazardKind.ROAD_DISRUPTION: _provisional_cell(
                PriorStrength.MEDIUM,
                0.25,
                "Transport disruption may limit access to cooling resources.",
            ),
        },
    ),
    _profile(
        C.REL_ROAD_ACCESS,
        default=_provisional_cell(
            PriorStrength.LOW_MEDIUM,
            0.20,
            "Road connectivity may influence access and response across hazards.",
        ),
        overrides={
            HazardKind.FLOOD: _provisional_cell(
                PriorStrength.MEDIUM_HIGH,
                0.30,
                "Flooding may disrupt road accessibility.",
            ),
            HazardKind.PLUVIAL_FLOOD: _provisional_cell(
                PriorStrength.HIGH,
                0.45,
                "Urban ponding and runoff may directly disrupt streets and "
                "underpasses.",
            ),
            HazardKind.OUTAGE: _provisional_cell(
                PriorStrength.MEDIUM_HIGH,
                0.35,
                "Transport access may affect outage response and repair.",
            ),
            HazardKind.ROAD_DISRUPTION: _provisional_cell(
                PriorStrength.VERY_HIGH,
                0.65,
                "Road connectivity is a direct transport-disruption pathway.",
            ),
            HazardKind.CIVIL_SECURITY_EVENT: _provisional_cell(
                PriorStrength.HIGH,
                0.50,
                "Road access may be critical for evacuation and response.",
            ),
            HazardKind.WINTER_STORM: _provisional_cell(
                PriorStrength.HIGH,
                0.55,
                "Winter weather directly affects road accessibility.",
            ),
            HazardKind.SNOWSTORM: _provisional_cell(
                PriorStrength.VERY_HIGH,
                0.65,
                "Snow accumulation and clearing directly affect road access.",
            ),
            HazardKind.FREEZING_RAIN: _provisional_cell(
                PriorStrength.VERY_HIGH,
                0.65,
                "Ice accumulation directly affects road safety and access.",
            ),
        },
    ),
    _profile(
        C.REL_INFRASTRUCTURE_DEPENDENCY,
        default=_provisional_cell(
            PriorStrength.MEDIUM,
            0.25,
            "Infrastructure dependence may create cascading service effects.",
        ),
        overrides={
            HazardKind.OUTAGE: _provisional_cell(
                PriorStrength.VERY_HIGH,
                0.65,
                "Infrastructure dependencies are direct outage and cascading-"
                "failure pathways.",
            ),
            HazardKind.CIVIL_SECURITY_EVENT: _provisional_cell(
                PriorStrength.HIGH,
                0.50,
                "Critical dependencies may drive broad service loss.",
            ),
            HazardKind.WINTER_STORM: _provisional_cell(
                PriorStrength.HIGH,
                0.45,
                "Winter weather may create coupled transport, power, and service "
                "disruptions.",
            ),
            HazardKind.FREEZING_RAIN: _provisional_cell(
                PriorStrength.VERY_HIGH,
                0.60,
                "Ice loading may trigger power and infrastructure cascades.",
            ),
        },
    ),
    _profile(
        C.REL_DRAINAGE_DEPENDENCY,
        default=_provisional_cell(
            PriorStrength.VERY_LOW,
            0.10,
            "Drainage dependency has narrow relevance outside water hazards.",
            caveat=(
                "Its utility depends strongly on drainage-asset data quality."
            ),
        ),
        overrides={
            HazardKind.FLOOD: _provisional_cell(
                PriorStrength.HIGH,
                0.55,
                "Urban flood burden may depend strongly on drainage capacity.",
            ),
            HazardKind.RIVERINE_FLOOD: _provisional_cell(
                PriorStrength.LOW_MEDIUM,
                0.20,
                "Drainage may influence local consequences but is not the main "
                "riverine mechanism.",
            ),
            HazardKind.PLUVIAL_FLOOD: _provisional_cell(
                PriorStrength.VERY_HIGH,
                0.75,
                "Drainage and sewer capacity are direct pluvial-flood pathways.",
            ),
            HazardKind.ROAD_DISRUPTION: _provisional_cell(
                PriorStrength.MEDIUM,
                0.20,
                "Drainage failure may contribute to flooded roads.",
            ),
        },
    ),
    _profile(
        C.REL_CRITICAL_FACILITY_DEPENDENCY,
        default=_provisional_cell(
            PriorStrength.MEDIUM,
            0.25,
            "Essential services may depend on critical facilities across hazards.",
        ),
        overrides={
            HazardKind.OUTAGE: _provisional_cell(
                PriorStrength.HIGH,
                0.55,
                "Hospitals and essential services may be strongly affected by "
                "outages.",
            ),
            HazardKind.ROAD_DISRUPTION: _provisional_cell(
                PriorStrength.HIGH,
                0.45,
                "Critical facilities may become inaccessible.",
            ),
            HazardKind.CIVIL_SECURITY_EVENT: _provisional_cell(
                PriorStrength.HIGH,
                0.50,
                "Critical-facility dependencies may amplify emergency burden.",
            ),
            HazardKind.WINTER_STORM: _provisional_cell(
                PriorStrength.HIGH,
                0.45,
                "Winter weather may disrupt access to or operation of critical "
                "facilities.",
            ),
            HazardKind.FREEZING_RAIN: _provisional_cell(
                PriorStrength.HIGH,
                0.50,
                "Ice-related outages and access loss may affect critical "
                "facilities.",
            ),
        },
    ),
    _profile(
        C.REL_REPORTING_SIMILARITY,
        default=_provisional_cell(
            PriorStrength.MEDIUM,
            0.15,
            "Reporting similarity may help predict observed complaint burden but "
            "may encode reporting propensity rather than physical disruption.",
            caveat=(
                "Interpret learned relevance cautiously and enforce strict "
                "temporal construction."
            ),
        ),
    ),
    _profile(
        C.REL_SOCIOECONOMIC_SIMILARITY,
        default=_provisional_cell(
            PriorStrength.MEDIUM,
            0.20,
            "Similar vulnerability profiles may share sensitivity and adaptive-"
            "capacity patterns.",
            caveat="Similarity is not geographic or causal propagation.",
        ),
        overrides={
            HazardKind.HEAT: _provisional_cell(
                PriorStrength.HIGH,
                0.40,
                "Socioeconomic vulnerability may shape heat sensitivity and "
                "adaptive capacity.",
            ),
            HazardKind.CIVIL_SECURITY_EVENT: _provisional_cell(
                PriorStrength.HIGH,
                0.40,
                "Broad civil-security burden may vary with vulnerability.",
            ),
            HazardKind.WINTER_STORM: _provisional_cell(
                PriorStrength.MEDIUM_HIGH,
                0.30,
                "Winter disruption and recovery may vary with adaptive capacity.",
            ),
        },
    ),
    _profile(
        C.REL_CROSS_SCALE_PARENT,
        default=_provisional_cell(
            PriorStrength.MEDIUM_HIGH,
            0.25,
            "Higher-scale context may provide event and administrative "
            "information for local prediction.",
        ),
        overrides={
            HazardKind.CIVIL_SECURITY_EVENT: _provisional_cell(
                PriorStrength.HIGH,
                0.35,
                "Broad civil-security events may require higher-scale context.",
            ),
            HazardKind.WINTER_STORM: _provisional_cell(
                PriorStrength.HIGH,
                0.35,
                "Winter storms often have regional forcing and citywide response.",
            ),
        },
    ),
    _profile(
        C.REL_CROSS_SCALE_CHILD,
        default=_provisional_cell(
            PriorStrength.MEDIUM_HIGH,
            0.25,
            "Higher-scale state may inform lower-scale prediction through "
            "downscaled context.",
        ),
        overrides={
            HazardKind.CIVIL_SECURITY_EVENT: _provisional_cell(
                PriorStrength.HIGH,
                0.35,
                "Broad event context may propagate to child geographies.",
            ),
            HazardKind.WINTER_STORM: _provisional_cell(
                PriorStrength.HIGH,
                0.35,
                "Regional winter forcing may propagate to tract-level effects.",
            ),
        },
    ),
    _profile(
        C.REL_ADMINISTRATIVE_MEMBERSHIP,
        default=_provisional_cell(
            PriorStrength.MEDIUM,
            0.20,
            "Administrative membership supports multi-scale context but is not "
            "itself a physical pathway.",
        ),
        overrides={
            HazardKind.CIVIL_SECURITY_EVENT: _provisional_cell(
                PriorStrength.HIGH,
                0.35,
                "Administrative hierarchy may matter for broad emergency events.",
            ),
        },
    ),
)


# =============================================================================
# Default registry construction
# =============================================================================


def build_default_hazard_relation_prior_registry(
    *,
    hazard_registry: HazardRegistry = DEFAULT_HAZARD_REGISTRY,
    relation_registry: RelationRegistry = DEFAULT_RELATION_REGISTRY,
    applicability_scope: PriorApplicabilityScope = (
        DEFAULT_PRIOR_APPLICABILITY_SCOPE
    ),
) -> HazardRelationPriorRegistry:
    """Build the complete provisional scoped prior registry."""

    profile_names = tuple(
        profile.relation_name
        for profile in DEFAULT_RELATION_PRIOR_PROFILES
    )
    _require_unique_strings("default prior profile names", profile_names)

    expected = set(relation_registry.relation_names)
    observed = set(profile_names)
    missing = sorted(expected - observed)
    unexpected = sorted(observed - expected)
    if missing or unexpected:
        raise RuntimeError(
            "Default prior profiles do not match the current relation "
            f"registry. Missing={missing}; unexpected={unexpected}."
        )

    priors = tuple(
        prior
        for profile in DEFAULT_RELATION_PRIOR_PROFILES
        for prior in profile.build_priors(
            hazard_registry=hazard_registry
        )
    )

    registry = HazardRelationPriorRegistry(
        priors=priors,
        applicability_scope=applicability_scope,
        source_hazard_registry=hazard_registry,
        source_relation_registry_name=relation_registry.registry_name,
        source_relation_registry_version=relation_registry.registry_version,
        source_relation_semantic_fingerprint=(
            relation_registry.semantic_fingerprint()
        ),
        source_relation_names=relation_registry.relation_names,
        source_stable_relation_ids=relation_registry.relation_ids,
        registry_status=PriorRegistryStatus.PROVISIONAL,
        regularization_approved=False,
    )
    registry.validate_against_sources(
        hazard_registry=hazard_registry,
        relation_registry=relation_registry,
        require_complete=True,
        require_current_versions=True,
    )
    return registry


DEFAULT_HAZARD_RELATION_PRIOR_REGISTRY: Final[
    HazardRelationPriorRegistry
] = build_default_hazard_relation_prior_registry()


def get_default_hazard_relation_prior_registry() -> HazardRelationPriorRegistry:
    return DEFAULT_HAZARD_RELATION_PRIOR_REGISTRY


def compile_default_hazard_relation_priors(
    compiled_relation_registry: CompiledRelationRegistry,
    *,
    application_context: PriorApplicationContext,
    hazards: Sequence[HazardKind | str],
    source_hazard_registry: HazardRegistry = DEFAULT_HAZARD_REGISTRY,
    source_relation_registry: RelationRegistry = DEFAULT_RELATION_REGISTRY,
    resolution_policy: PriorResolutionPolicy = PriorResolutionPolicy.HAZARD_FIRST,
    hazard_inheritance_confidence_decay: float = 0.80,
    relation_inheritance_confidence_decay: float = 0.75,
    require_training_supported_hazards: bool = False,
    allow_partially_data_backed: bool = False,
) -> CompiledHazardRelationPriors:
    return DEFAULT_HAZARD_RELATION_PRIOR_REGISTRY.compile(
        compiled_relation_registry,
        source_hazard_registry=source_hazard_registry,
        source_relation_registry=source_relation_registry,
        application_context=application_context,
        hazards=hazards,
        resolution_policy=resolution_policy,
        hazard_inheritance_confidence_decay=(
            hazard_inheritance_confidence_decay
        ),
        relation_inheritance_confidence_decay=(
            relation_inheritance_confidence_decay
        ),
        allow_all_hazard_fallback=True,
        require_explicit=False,
        require_queryable_hazards=True,
        require_training_supported_hazards=(
            require_training_supported_hazards
        ),
        allow_partially_data_backed=allow_partially_data_backed,
        allow_fallback_hazard=False,
    )


# =============================================================================
# Resolution helper
# =============================================================================


def _resolved_from_prior(
    source_prior: HazardRelationPrior,
    *,
    requested_hazard: HazardKind,
    requested_relation_name: str,
    resolution_mode: PriorResolutionMode,
    hazard_inheritance_distance: int,
    relation_inheritance_distance: int,
    confidence_scale: float,
) -> ResolvedHazardRelationPrior:
    scale = _require_finite_number("confidence_scale", confidence_scale)
    if not 0.0 <= scale <= 1.0:
        raise ValueError("confidence_scale must lie in [0, 1].")

    definition = source_prior.definition
    inherited_confidence = definition.confidence * scale
    return ResolvedHazardRelationPrior(
        hazard=requested_hazard,
        relation_name=requested_relation_name,
        prior_mean=definition.prior_mean,
        confidence=inherited_confidence,
        initialization_allowed=(
            definition.initialization_allowed
            and inherited_confidence > 0.0
        ),
        regularization_allowed=(
            definition.regularization_allowed
            and inherited_confidence > 0.0
        ),
        evidence_type=definition.evidence_type,
        rationale=definition.rationale,
        caveat=definition.caveat,
        resolution_mode=resolution_mode,
        source_hazard=source_prior.hazard,
        source_relation_name=source_prior.relation_name,
        hazard_inheritance_distance=hazard_inheritance_distance,
        relation_inheritance_distance=relation_inheritance_distance,
    )


__all__ = (
    "COMPILED_HAZARD_PRIOR_SCHEMA_VERSION",
    "CompiledHazardRelationPriors",
    "DEFAULT_HAZARD_RELATION_PRIOR_REGISTRY",
    "DEFAULT_HAZARD_RELATION_PRIOR_REGISTRY_NAME",
    "DEFAULT_PRIOR_APPLICATION_CONTEXT",
    "DEFAULT_PRIOR_APPLICABILITY_SCOPE",
    "DEFAULT_RELATION_PRIOR_PROFILES",
    "EMPIRICAL_PRIOR_PROVENANCE_SCHEMA_VERSION",
    "EmpiricalPriorProvenance",
    "GateInitializationActivation",
    "HAZARD_RELATION_PRIOR_REGISTRY_VERSION",
    "HAZARD_RELATION_PRIOR_SCHEMA_VERSION",
    "HazardRelationPrior",
    "HazardRelationPriorRegistry",
    "PRIOR_APPLICABILITY_SCHEMA_VERSION",
    "PRIOR_MEAN_BY_STRENGTH",
    "PriorApplicationContext",
    "PriorApplicabilityScope",
    "PriorCellDefinition",
    "PriorEvidenceType",
    "PriorRegistryStatus",
    "PriorResolutionMode",
    "PriorResolutionPolicy",
    "PriorStrength",
    "RelationPriorProfile",
    "ResolvedHazardRelationPrior",
    "build_default_hazard_relation_prior_registry",
    "compile_default_hazard_relation_priors",
    "get_default_hazard_relation_prior_registry",
)
