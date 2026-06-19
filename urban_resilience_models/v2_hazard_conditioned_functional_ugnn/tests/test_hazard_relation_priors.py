"""
Tests for scoped hazard-relation priors in the V2 functional UGNN.

Target path:
    urban_resilience_models/
        v2_hazard_conditioned_functional_ugnn/
            tests/
                test_hazard_relation_priors.py

Implementations under test:
    urban_resilience_models/
        v2_hazard_conditioned_functional_ugnn/
            hazard/
                hazard_registry.py
            relations/
                relation_registry.py
                hazard_relation_priors.py

Python requirement:
    Python >= 3.11

The suite freezes the integration boundary among the canonical hazard
registry, canonical relation registry, and scoped prior registry. It tests:

- neutral ownership of hazard semantics;
- complete default prior coverage;
- preservation of reviewed V2 prior values;
- conservative provisional values for newly introduced hazards;
- task-scope applicability;
- control-relation safeguards;
- hazard and relation hierarchy inheritance;
- all-hazard and neutral fallback behavior;
- confidence-adjusted gate initialization;
- prohibition of provisional regularization;
- dense compilation, stable IDs, and lookup helpers;
- source fingerprints and artifact matching;
- serialization, tamper detection, and immutability.

Run from the repository root:

    python -m pytest -q \
        urban_resilience_models/v2_hazard_conditioned_functional_ugnn/tests/test_hazard_relation_priors.py
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import FrozenInstanceError, replace
import json
import math
from types import MappingProxyType
from typing import Any

import pytest

from urban_resilience_models.v2_hazard_conditioned_functional_ugnn import (
    constants as C,
)
from urban_resilience_models.v2_hazard_conditioned_functional_ugnn.hazard.hazard_registry import (
    DEFAULT_HAZARD_REGISTRY,
    HazardKind,
    HazardRegistry,
)
from urban_resilience_models.v2_hazard_conditioned_functional_ugnn.relations import (
    hazard_relation_priors as prior_module,
)
from urban_resilience_models.v2_hazard_conditioned_functional_ugnn.relations.hazard_relation_priors import (
    COMPILED_HAZARD_PRIOR_SCHEMA_VERSION,
    DEFAULT_HAZARD_RELATION_PRIOR_REGISTRY,
    DEFAULT_PRIOR_APPLICATION_CONTEXT,
    DEFAULT_PRIOR_APPLICABILITY_SCOPE,
    DEFAULT_RELATION_PRIOR_PROFILES,
    HAZARD_RELATION_PRIOR_REGISTRY_VERSION,
    HAZARD_RELATION_PRIOR_SCHEMA_VERSION,
    CompiledHazardRelationPriors,
    EmpiricalPriorProvenance,
    GateInitializationActivation,
    HazardRelationPrior,
    HazardRelationPriorRegistry,
    PriorApplicationContext,
    PriorApplicabilityScope,
    PriorCellDefinition,
    PriorEvidenceType,
    PriorRegistryStatus,
    PriorResolutionMode,
    PriorResolutionPolicy,
    PriorStrength,
    RelationPriorProfile,
    ResolvedHazardRelationPrior,
    build_default_hazard_relation_prior_registry,
    compile_default_hazard_relation_priors,
    get_default_hazard_relation_prior_registry,
)
from urban_resilience_models.v2_hazard_conditioned_functional_ugnn.relations.relation_registry import (
    DEFAULT_RELATION_REGISTRY,
    CompiledRelationRegistry,
    HierarchyCompilationPolicy,
    RelationRegistry,
)


# =============================================================================
# Fixtures and helpers
# =============================================================================


@pytest.fixture(scope="module")
def hazard_registry() -> HazardRegistry:
    """Return the tested canonical hazard registry."""

    return DEFAULT_HAZARD_REGISTRY


@pytest.fixture(scope="module")
def relation_registry() -> RelationRegistry:
    """Return the tested canonical relation registry."""

    return DEFAULT_RELATION_REGISTRY


@pytest.fixture(scope="module")
def prior_registry() -> HazardRelationPriorRegistry:
    """Return the complete default scoped prior registry."""

    return DEFAULT_HAZARD_RELATION_PRIOR_REGISTRY


@pytest.fixture(scope="module")
def compiled_relation_registry(
    relation_registry: RelationRegistry,
) -> CompiledRelationRegistry:
    """
    Compile the complete relation ontology for metadata tests.

    Implementation and message-passing requirements are intentionally disabled
    because this test targets ontology alignment rather than current runtime
    availability. Parent-child overlap is explicitly allowed so every canonical
    relation receives one prior column.
    """

    return relation_registry.compile(
        relation_registry.relation_names,
        require_implemented=False,
        require_message_passing=False,
        require_training=False,
        allow_control_relations=True,
        hierarchy_policy=HierarchyCompilationPolicy.ALLOW_OVERLAP,
    )


@pytest.fixture(scope="module")
def compiled_priors(
    prior_registry: HazardRelationPriorRegistry,
    hazard_registry: HazardRegistry,
    relation_registry: RelationRegistry,
    compiled_relation_registry: CompiledRelationRegistry,
) -> CompiledHazardRelationPriors:
    """Compile representative hazard rows against every relation."""

    return prior_registry.compile(
        compiled_relation_registry,
        source_hazard_registry=hazard_registry,
        source_relation_registry=relation_registry,
        application_context=DEFAULT_PRIOR_APPLICATION_CONTEXT,
        hazards=(
            HazardKind.FLOOD,
            HazardKind.HEAT,
            HazardKind.PLUVIAL_FLOOD,
        ),
    )


def _definition(
    *,
    strength: PriorStrength = PriorStrength.HIGH,
    confidence: float = 0.50,
    rationale: str = "Synthetic test prior.",
    initialization_allowed: bool = True,
    regularization_allowed: bool = False,
    evidence_type: PriorEvidenceType = (
        PriorEvidenceType.PROVISIONAL_ONTOLOGY
    ),
) -> PriorCellDefinition:
    """Build a compact valid prior definition for resolution tests."""

    return PriorCellDefinition(
        strength=strength,
        confidence=confidence,
        evidence_type=evidence_type,
        rationale=rationale,
        initialization_allowed=initialization_allowed,
        regularization_allowed=regularization_allowed,
    )


def _prior(
    hazard: HazardKind,
    relation_name: str,
    *,
    strength: PriorStrength = PriorStrength.HIGH,
    confidence: float = 0.50,
    rationale: str = "Synthetic test prior.",
) -> HazardRelationPrior:
    """Build one explicit synthetic prior cell."""

    return HazardRelationPrior(
        hazard=hazard,
        relation_name=relation_name,
        definition=_definition(
            strength=strength,
            confidence=confidence,
            rationale=rationale,
        ),
    )


def _sparse_registry(
    *priors: HazardRelationPrior,
) -> HazardRelationPriorRegistry:
    """Reuse the default source identities with a deliberately sparse table."""

    return replace(
        DEFAULT_HAZARD_RELATION_PRIOR_REGISTRY,
        priors=tuple(priors),
    )


def _replace_prior_cell(
    registry: HazardRelationPriorRegistry,
    hazard: HazardKind,
    relation_name: str,
    *,
    definition: PriorCellDefinition,
) -> HazardRelationPriorRegistry:
    """Replace one explicit cell and rebuild the immutable registry."""

    replaced = False
    updated: list[HazardRelationPrior] = []

    for prior in registry.priors:
        if prior.hazard == hazard and prior.relation_name == relation_name:
            updated.append(
                replace(
                    prior,
                    definition=definition,
                )
            )
            replaced = True
        else:
            updated.append(prior)

    if not replaced:
        raise AssertionError(
            f"Missing test cell {(hazard.value, relation_name)!r}."
        )

    return replace(
        registry,
        priors=tuple(updated),
    )


def _child_relation_pair(
    relation_registry: RelationRegistry,
) -> tuple[str, str]:
    """Return one tested child relation and its direct parent."""

    for entry in relation_registry.entries:
        parent_name = entry.specification.parent_relation_name
        if parent_name is not None:
            return entry.name, parent_name

    raise AssertionError(
        "The canonical relation registry contains no hierarchy edge."
    )


# =============================================================================
# Ownership and default integration
# =============================================================================


def test_prior_module_imports_canonical_hazard_kind() -> None:
    """The prior module consumes—rather than redefines—hazard semantics."""

    assert prior_module.HazardKind is HazardKind

    for semantic_name in (
        "HazardKind",
        "HazardSupportState",
        "HazardOntologyRole",
        "HAZARD_ID_FLOOD",
        "CANONICAL_HAZARD_STABLE_IDS",
        "QUERYABLE_HAZARDS",
    ):
        assert semantic_name not in prior_module.__all__


def test_default_registry_constructs_against_current_sources(
    hazard_registry: HazardRegistry,
    relation_registry: RelationRegistry,
) -> None:
    registry = build_default_hazard_relation_prior_registry(
        hazard_registry=hazard_registry,
        relation_registry=relation_registry,
    )

    registry.validate_against_sources(
        hazard_registry=hazard_registry,
        relation_registry=relation_registry,
        require_complete=True,
        require_current_versions=True,
    )


def test_default_getter_returns_singleton(
    prior_registry: HazardRelationPriorRegistry,
) -> None:
    assert (
        get_default_hazard_relation_prior_registry()
        is prior_registry
    )


def test_default_registry_versions(
    prior_registry: HazardRelationPriorRegistry,
) -> None:
    assert (
        prior_registry.registry_version
        == HAZARD_RELATION_PRIOR_REGISTRY_VERSION
    )
    assert prior_registry.schema_version == HAZARD_RELATION_PRIOR_SCHEMA_VERSION
    assert prior_registry.registry_status == PriorRegistryStatus.PROVISIONAL
    assert not prior_registry.regularization_approved


def test_default_registry_references_current_sources(
    prior_registry: HazardRelationPriorRegistry,
    hazard_registry: HazardRegistry,
    relation_registry: RelationRegistry,
) -> None:
    assert prior_registry.source_hazard_registry is hazard_registry
    assert (
        prior_registry.source_hazard_registry.compatibility_fingerprint()
        == hazard_registry.compatibility_fingerprint()
    )
    assert (
        prior_registry.source_relation_registry_name
        == relation_registry.registry_name
    )
    assert (
        prior_registry.source_relation_registry_version
        == relation_registry.registry_version
    )
    assert (
        prior_registry.source_relation_semantic_fingerprint
        == relation_registry.semantic_fingerprint()
    )
    assert prior_registry.source_relation_names == relation_registry.relation_names
    assert (
        prior_registry.source_stable_relation_ids
        == relation_registry.relation_ids
    )


def test_default_profiles_cover_exact_relation_registry(
    relation_registry: RelationRegistry,
) -> None:
    profile_names = tuple(
        profile.relation_name
        for profile in DEFAULT_RELATION_PRIOR_PROFILES
    )

    assert len(profile_names) == len(relation_registry.relation_names)
    assert len(set(profile_names)) == len(profile_names)
    assert set(profile_names) == set(relation_registry.relation_names)
    assert len(set(profile_names)) == len(profile_names)


def test_default_prior_table_is_complete_and_canonically_ordered(
    prior_registry: HazardRelationPriorRegistry,
    hazard_registry: HazardRegistry,
    relation_registry: RelationRegistry,
) -> None:
    expected_keys = tuple(
        (hazard, relation_name)
        for hazard in hazard_registry.hazard_kinds
        for relation_name in relation_registry.relation_names
    )
    observed_keys = tuple(
        (prior.hazard, prior.relation_name)
        for prior in prior_registry.priors
    )

    assert len(prior_registry) == (
        len(hazard_registry) * len(relation_registry)
    )
    assert observed_keys == expected_keys


def test_default_registry_rejects_different_hazard_source(
    prior_registry: HazardRelationPriorRegistry,
    hazard_registry: HazardRegistry,
    relation_registry: RelationRegistry,
) -> None:
    mismatched = replace(
        hazard_registry,
        registry_name="different_hazard_registry",
    )

    with pytest.raises(
        ValueError,
        match="different hazard registry",
    ):
        prior_registry.validate_against_sources(
            hazard_registry=mismatched,
            relation_registry=relation_registry,
        )


def test_default_registry_rejects_different_relation_source(
    prior_registry: HazardRelationPriorRegistry,
    hazard_registry: HazardRegistry,
    relation_registry: RelationRegistry,
) -> None:
    mismatched = replace(
        relation_registry,
        registry_name="different_relation_registry",
    )

    with pytest.raises(
        ValueError,
        match="different relation registry",
    ):
        prior_registry.validate_against_sources(
            hazard_registry=hazard_registry,
            relation_registry=mismatched,
        )


# =============================================================================
# Reviewed numerical prior contracts
# =============================================================================


def test_prior_mean_vocabulary() -> None:
    assert _definition(strength=PriorStrength.VERY_LOW).prior_mean == 0.10
    assert _definition(strength=PriorStrength.LOW).prior_mean == 0.20
    assert _definition(strength=PriorStrength.LOW_MEDIUM).prior_mean == 0.35
    assert _definition(strength=PriorStrength.MEDIUM).prior_mean == 0.50
    assert _definition(strength=PriorStrength.MEDIUM_HIGH).prior_mean == 0.65
    assert _definition(strength=PriorStrength.HIGH).prior_mean == 0.80
    assert _definition(strength=PriorStrength.VERY_HIGH).prior_mean == 0.90


def test_preserved_temporal_memory_values_for_existing_hazards(
    prior_registry: HazardRelationPriorRegistry,
) -> None:
    flood = prior_registry.require_explicit(
        HazardKind.FLOOD,
        C.REL_TEMPORAL_MEMORY,
    )
    heat = prior_registry.require_explicit(
        HazardKind.HEAT,
        C.REL_TEMPORAL_MEMORY,
    )
    outage = prior_registry.require_explicit(
        HazardKind.OUTAGE,
        C.REL_TEMPORAL_MEMORY,
    )

    assert flood.definition.strength == PriorStrength.HIGH
    assert flood.confidence == pytest.approx(0.55)

    assert heat.definition.strength == PriorStrength.HIGH
    assert heat.confidence == pytest.approx(0.45)

    # OUTAGE uses the preserved profile default.
    assert outage.definition.strength == PriorStrength.MEDIUM_HIGH
    assert outage.confidence == pytest.approx(0.35)


def test_new_hazard_temporal_priors_are_conservative_and_provisional(
    prior_registry: HazardRelationPriorRegistry,
) -> None:
    pluvial = prior_registry.require_explicit(
        HazardKind.PLUVIAL_FLOOD,
        C.REL_TEMPORAL_MEMORY,
    )
    freezing_rain = prior_registry.require_explicit(
        HazardKind.FREEZING_RAIN,
        C.REL_TEMPORAL_MEMORY,
    )

    assert pluvial.definition.strength == PriorStrength.HIGH
    assert pluvial.confidence == pytest.approx(0.40)
    assert (
        pluvial.definition.evidence_type
        == PriorEvidenceType.PROVISIONAL_ONTOLOGY
    )
    assert not pluvial.definition.regularization_allowed

    assert freezing_rain.definition.strength == PriorStrength.HIGH
    assert freezing_rain.confidence == pytest.approx(0.35)
    assert (
        freezing_rain.definition.evidence_type
        == PriorEvidenceType.PROVISIONAL_ONTOLOGY
    )
    assert not freezing_rain.definition.regularization_allowed


def test_pluvial_drainage_prior_is_explicit_but_nonregularizing(
    prior_registry: HazardRelationPriorRegistry,
) -> None:
    cell = prior_registry.require_explicit(
        HazardKind.PLUVIAL_FLOOD,
        C.REL_DRAINAGE_DEPENDENCY,
    )

    assert cell.definition.strength == PriorStrength.VERY_HIGH
    assert cell.confidence == pytest.approx(0.70)
    assert cell.definition.initialization_allowed
    assert not cell.definition.regularization_allowed


def test_all_default_cells_are_nonregularizing(
    prior_registry: HazardRelationPriorRegistry,
) -> None:
    assert all(
        not prior.definition.regularization_allowed
        for prior in prior_registry
    )


# =============================================================================
# Control-relation safeguards
# =============================================================================


def test_control_relations_are_neutral_zero_confidence_and_nonoperative(
    prior_registry: HazardRelationPriorRegistry,
    hazard_registry: HazardRegistry,
    relation_registry: RelationRegistry,
) -> None:
    control_names = tuple(
        entry.name
        for entry in relation_registry.entries
        if entry.specification.is_control
    )

    assert control_names

    for hazard in hazard_registry.hazard_kinds:
        for relation_name in control_names:
            definition = prior_registry.require_explicit(
                hazard,
                relation_name,
            ).definition

            assert definition.evidence_type == PriorEvidenceType.CONTROL
            assert definition.strength == PriorStrength.MEDIUM
            assert definition.prior_mean == pytest.approx(0.50)
            assert definition.confidence == pytest.approx(0.0)
            assert not definition.initialization_allowed
            assert not definition.regularization_allowed


def test_control_relation_with_ordinary_prior_is_rejected_by_source_validation(
    prior_registry: HazardRelationPriorRegistry,
    hazard_registry: HazardRegistry,
    relation_registry: RelationRegistry,
) -> None:
    control_name = next(
        entry.name
        for entry in relation_registry.entries
        if entry.specification.is_control
    )

    invalid = _replace_prior_cell(
        prior_registry,
        HazardKind.FLOOD,
        control_name,
        definition=_definition(),
    )

    with pytest.raises(
        ValueError,
        match="Control relation",
    ):
        invalid.validate_against_sources(
            hazard_registry=hazard_registry,
            relation_registry=relation_registry,
            require_complete=True,
        )


def test_noncontrol_relation_cannot_use_control_evidence(
    prior_registry: HazardRelationPriorRegistry,
    hazard_registry: HazardRegistry,
    relation_registry: RelationRegistry,
) -> None:
    control_definition = PriorCellDefinition(
        strength=PriorStrength.MEDIUM,
        confidence=0.0,
        evidence_type=PriorEvidenceType.CONTROL,
        rationale="Synthetic control definition.",
        initialization_allowed=False,
        regularization_allowed=False,
    )

    invalid = _replace_prior_cell(
        prior_registry,
        HazardKind.FLOOD,
        C.REL_TEMPORAL_MEMORY,
        definition=control_definition,
    )

    with pytest.raises(
        ValueError,
        match="Non-control relation",
    ):
        invalid.validate_against_sources(
            hazard_registry=hazard_registry,
            relation_registry=relation_registry,
            require_complete=True,
        )


# =============================================================================
# Applicability scope
# =============================================================================


def test_default_scope_matches_default_context() -> None:
    assert DEFAULT_PRIOR_APPLICABILITY_SCOPE.matches(
        DEFAULT_PRIOR_APPLICATION_CONTEXT
    )
    DEFAULT_PRIOR_APPLICABILITY_SCOPE.assert_applicable(
        DEFAULT_PRIOR_APPLICATION_CONTEXT
    )


def test_default_scope_rejects_wrong_target() -> None:
    wrong_context = replace(
        DEFAULT_PRIOR_APPLICATION_CONTEXT,
        target_name="different_target",
    )

    assert not DEFAULT_PRIOR_APPLICABILITY_SCOPE.matches(wrong_context)

    with pytest.raises(
        ValueError,
        match="not applicable",
    ):
        DEFAULT_PRIOR_APPLICABILITY_SCOPE.assert_applicable(
            wrong_context
        )


def test_scope_serialization_round_trip() -> None:
    reconstructed = PriorApplicabilityScope.from_dict(
        json.loads(
            json.dumps(DEFAULT_PRIOR_APPLICABILITY_SCOPE.to_dict())
        )
    )

    assert reconstructed == DEFAULT_PRIOR_APPLICABILITY_SCOPE
    assert isinstance(reconstructed.target_names, tuple)
    assert (
        reconstructed.fingerprint()
        == DEFAULT_PRIOR_APPLICABILITY_SCOPE.fingerprint()
    )


def test_context_serialization_round_trip() -> None:
    reconstructed = PriorApplicationContext.from_dict(
        json.loads(
            json.dumps(DEFAULT_PRIOR_APPLICATION_CONTEXT.to_dict())
        )
    )

    assert reconstructed == DEFAULT_PRIOR_APPLICATION_CONTEXT


# =============================================================================
# Prior-cell validation and provenance
# =============================================================================


def test_initialization_and_regularization_flags_require_booleans() -> None:
    with pytest.raises(
        TypeError,
        match="initialization_allowed must be a Boolean",
    ):
        replace(
            _definition(),
            initialization_allowed=1,  # type: ignore[arg-type]
        )

    with pytest.raises(
        TypeError,
        match="regularization_allowed must be a Boolean",
    ):
        replace(
            _definition(),
            regularization_allowed=1,  # type: ignore[arg-type]
        )


def test_zero_confidence_cell_cannot_initialize_or_regularize() -> None:
    with pytest.raises(
        ValueError,
        match="Zero-confidence cells",
    ):
        _definition(confidence=0.0)


def test_literature_prior_requires_reference_ids() -> None:
    with pytest.raises(
        ValueError,
        match="Literature priors require reference IDs",
    ):
        PriorCellDefinition(
            strength=PriorStrength.HIGH,
            confidence=0.5,
            evidence_type=PriorEvidenceType.LITERATURE,
            rationale="Literature prior without citation metadata.",
        )


def test_expert_prior_requires_expert_source_ids() -> None:
    with pytest.raises(
        ValueError,
        match="Expert priors require expert source IDs",
    ):
        PriorCellDefinition(
            strength=PriorStrength.HIGH,
            confidence=0.5,
            evidence_type=PriorEvidenceType.EXPERT,
            rationale="Expert prior without source identity.",
        )


def test_empirical_prior_requires_provenance() -> None:
    with pytest.raises(
        ValueError,
        match="Empirical priors require provenance",
    ):
        PriorCellDefinition(
            strength=PriorStrength.HIGH,
            confidence=0.5,
            evidence_type=PriorEvidenceType.EMPIRICAL,
            rationale="Empirical prior without provenance.",
        )


def test_empirical_provenance_round_trip() -> None:
    provenance = EmpiricalPriorProvenance(
        dataset_fingerprint="dataset-fingerprint",
        split_fingerprint="split-fingerprint",
        source_artifact_fingerprint="source-artifact-fingerprint",
        estimation_cutoff="2026-06-01",
        estimator_name="held_out_rank_estimator",
        estimator_version="0.1",
        held_out_estimation=True,
        random_seed=17,
    )

    reconstructed = EmpiricalPriorProvenance.from_dict(
        json.loads(json.dumps(provenance.to_dict()))
    )

    assert reconstructed == provenance


def test_prior_cell_round_trip_and_mean_verification() -> None:
    definition = PriorCellDefinition(
        strength=PriorStrength.HIGH,
        confidence=0.45,
        evidence_type=PriorEvidenceType.LITERATURE,
        rationale="Auditable literature-backed test prior.",
        evidence_reference_ids=("reference-1",),
        reviewed_by=("reviewer-1",),
        review_date="2026-06-19",
    )

    payload = json.loads(json.dumps(definition.to_dict()))
    reconstructed = PriorCellDefinition.from_dict(payload)

    assert reconstructed == definition

    payload["prior_mean"] = 0.123
    with pytest.raises(
        ValueError,
        match="prior_mean does not match strength",
    ):
        PriorCellDefinition.from_dict(payload)


def test_invalid_review_date_is_rejected() -> None:
    with pytest.raises(
        ValueError,
        match="ISO date format",
    ):
        replace(
            _definition(),
            review_date="June 19, 2026",
        )


# =============================================================================
# Provisional regularization safeguards
# =============================================================================


def test_provisional_registry_cannot_be_regularization_approved(
    prior_registry: HazardRelationPriorRegistry,
) -> None:
    with pytest.raises(
        ValueError,
        match="provisional registry cannot be approved",
    ):
        replace(
            prior_registry,
            regularization_approved=True,
        )


def test_unapproved_registry_rejects_regularizing_cell(
    prior_registry: HazardRelationPriorRegistry,
) -> None:
    source = prior_registry.require_explicit(
        HazardKind.FLOOD,
        C.REL_TEMPORAL_MEMORY,
    )
    regularizing_definition = replace(
        source.definition,
        regularization_allowed=True,
    )

    with pytest.raises(
        ValueError,
        match="not approved",
    ):
        _replace_prior_cell(
            prior_registry,
            HazardKind.FLOOD,
            C.REL_TEMPORAL_MEMORY,
            definition=regularizing_definition,
        )


# =============================================================================
# Explicit lookup and resolution
# =============================================================================


def test_explicit_lookup_and_requirement(
    prior_registry: HazardRelationPriorRegistry,
) -> None:
    explicit = prior_registry.get_explicit(
        HazardKind.FLOOD,
        C.REL_TEMPORAL_MEMORY,
    )

    assert explicit is not None
    assert (
        prior_registry.require_explicit(
            HazardKind.FLOOD,
            C.REL_TEMPORAL_MEMORY,
        )
        is explicit
    )

    sparse = _sparse_registry(explicit)
    assert sparse.get_explicit(HazardKind.HEAT, C.REL_TEMPORAL_MEMORY) is None

    with pytest.raises(
        KeyError,
        match="No explicit prior",
    ):
        sparse.require_explicit(
            HazardKind.HEAT,
            C.REL_TEMPORAL_MEMORY,
        )


def test_explicit_resolution_preserves_requested_identity(
    hazard_registry: HazardRegistry,
    relation_registry: RelationRegistry,
) -> None:
    source = _prior(
        HazardKind.PLUVIAL_FLOOD,
        C.REL_TEMPORAL_MEMORY,
        confidence=0.60,
    )
    sparse = _sparse_registry(source)

    resolved = sparse.resolve(
        HazardKind.PLUVIAL_FLOOD,
        C.REL_TEMPORAL_MEMORY,
        hazard_registry=hazard_registry,
        relation_registry=relation_registry,
    )

    assert resolved.hazard == HazardKind.PLUVIAL_FLOOD
    assert resolved.relation_name == C.REL_TEMPORAL_MEMORY
    assert resolved.resolution_mode == PriorResolutionMode.EXPLICIT
    assert resolved.source_hazard == HazardKind.PLUVIAL_FLOOD
    assert resolved.source_relation_name == C.REL_TEMPORAL_MEMORY
    assert resolved.hazard_inheritance_distance == 0
    assert resolved.relation_inheritance_distance == 0
    assert resolved.confidence == pytest.approx(0.60)


def test_hazard_ancestor_resolution_decays_confidence(
    hazard_registry: HazardRegistry,
    relation_registry: RelationRegistry,
) -> None:
    source = _prior(
        HazardKind.FLOOD,
        C.REL_TEMPORAL_MEMORY,
        confidence=0.50,
    )
    sparse = _sparse_registry(source)

    resolved = sparse.resolve(
        HazardKind.PLUVIAL_FLOOD,
        C.REL_TEMPORAL_MEMORY,
        hazard_registry=hazard_registry,
        relation_registry=relation_registry,
        hazard_inheritance_confidence_decay=0.80,
    )

    assert resolved.hazard == HazardKind.PLUVIAL_FLOOD
    assert resolved.source_hazard == HazardKind.FLOOD
    assert resolved.source_relation_name == C.REL_TEMPORAL_MEMORY
    assert resolved.resolution_mode == PriorResolutionMode.HAZARD_ANCESTOR
    assert resolved.hazard_inheritance_distance == 1
    assert resolved.relation_inheritance_distance == 0
    assert resolved.confidence == pytest.approx(0.50 * 0.80)


def test_relation_ancestor_resolution_decays_confidence(
    hazard_registry: HazardRegistry,
    relation_registry: RelationRegistry,
) -> None:
    child_name, parent_name = _child_relation_pair(relation_registry)
    source = _prior(
        HazardKind.FLOOD,
        parent_name,
        confidence=0.60,
    )
    sparse = _sparse_registry(source)

    resolved = sparse.resolve(
        HazardKind.FLOOD,
        child_name,
        hazard_registry=hazard_registry,
        relation_registry=relation_registry,
        relation_inheritance_confidence_decay=0.75,
    )

    assert resolved.hazard == HazardKind.FLOOD
    assert resolved.relation_name == child_name
    assert resolved.source_hazard == HazardKind.FLOOD
    assert resolved.source_relation_name == parent_name
    assert resolved.resolution_mode == PriorResolutionMode.RELATION_ANCESTOR
    assert resolved.hazard_inheritance_distance == 0
    assert resolved.relation_inheritance_distance == 1
    assert resolved.confidence == pytest.approx(0.60 * 0.75)


def test_combined_hazard_relation_inheritance_multiplies_decays(
    hazard_registry: HazardRegistry,
    relation_registry: RelationRegistry,
) -> None:
    child_name, parent_name = _child_relation_pair(relation_registry)
    source = _prior(
        HazardKind.FLOOD,
        parent_name,
        confidence=0.50,
    )
    sparse = _sparse_registry(source)

    resolved = sparse.resolve(
        HazardKind.PLUVIAL_FLOOD,
        child_name,
        hazard_registry=hazard_registry,
        relation_registry=relation_registry,
        hazard_inheritance_confidence_decay=0.80,
        relation_inheritance_confidence_decay=0.75,
    )

    assert resolved.hazard == HazardKind.PLUVIAL_FLOOD
    assert resolved.relation_name == child_name
    assert resolved.source_hazard == HazardKind.FLOOD
    assert resolved.source_relation_name == parent_name
    assert (
        resolved.resolution_mode
        == PriorResolutionMode.HAZARD_RELATION_ANCESTOR
    )
    assert resolved.hazard_inheritance_distance == 1
    assert resolved.relation_inheritance_distance == 1
    assert resolved.confidence == pytest.approx(0.50 * 0.80 * 0.75)


def test_resolution_policy_controls_same_hazard_vs_same_relation_precedence(
    hazard_registry: HazardRegistry,
    relation_registry: RelationRegistry,
) -> None:
    child_name, parent_name = _child_relation_pair(relation_registry)

    same_hazard_parent_relation = _prior(
        HazardKind.PLUVIAL_FLOOD,
        parent_name,
        strength=PriorStrength.HIGH,
        confidence=0.40,
        rationale="Same hazard, ancestor relation.",
    )
    parent_hazard_same_relation = _prior(
        HazardKind.FLOOD,
        child_name,
        strength=PriorStrength.LOW,
        confidence=0.70,
        rationale="Ancestor hazard, same relation.",
    )
    sparse = _sparse_registry(
        same_hazard_parent_relation,
        parent_hazard_same_relation,
    )

    hazard_first = sparse.resolve(
        HazardKind.PLUVIAL_FLOOD,
        child_name,
        hazard_registry=hazard_registry,
        relation_registry=relation_registry,
        resolution_policy=PriorResolutionPolicy.HAZARD_FIRST,
    )
    relation_first = sparse.resolve(
        HazardKind.PLUVIAL_FLOOD,
        child_name,
        hazard_registry=hazard_registry,
        relation_registry=relation_registry,
        resolution_policy=PriorResolutionPolicy.RELATION_FIRST,
    )

    assert hazard_first.resolution_mode == PriorResolutionMode.RELATION_ANCESTOR
    assert hazard_first.source_hazard == HazardKind.PLUVIAL_FLOOD
    assert hazard_first.source_relation_name == parent_name

    assert relation_first.resolution_mode == PriorResolutionMode.HAZARD_ANCESTOR
    assert relation_first.source_hazard == HazardKind.FLOOD
    assert relation_first.source_relation_name == child_name


def test_all_hazard_fallback_preserves_requested_hazard(
    hazard_registry: HazardRegistry,
    relation_registry: RelationRegistry,
) -> None:
    source = _prior(
        HazardKind.ALL_HAZARD,
        C.REL_TEMPORAL_MEMORY,
        confidence=0.40,
    )
    sparse = _sparse_registry(source)

    resolved = sparse.resolve(
        HazardKind.HEAT,
        C.REL_TEMPORAL_MEMORY,
        hazard_registry=hazard_registry,
        relation_registry=relation_registry,
    )

    assert resolved.hazard == HazardKind.HEAT
    assert resolved.relation_name == C.REL_TEMPORAL_MEMORY
    assert resolved.source_hazard == HazardKind.ALL_HAZARD
    assert resolved.source_relation_name == C.REL_TEMPORAL_MEMORY
    assert resolved.resolution_mode == PriorResolutionMode.ALL_HAZARD
    assert resolved.hazard_inheritance_distance == 0
    assert resolved.relation_inheritance_distance == 0
    assert resolved.confidence == pytest.approx(0.40)


def test_neutral_default_is_nonoperative(
    hazard_registry: HazardRegistry,
    relation_registry: RelationRegistry,
) -> None:
    sparse = _sparse_registry(
        _prior(
            HazardKind.FLOOD,
            C.REL_TEMPORAL_MEMORY,
        )
    )

    resolved = sparse.resolve(
        HazardKind.HEAT,
        C.REL_ROAD_ACCESS,
        hazard_registry=hazard_registry,
        relation_registry=relation_registry,
        allow_all_hazard_fallback=False,
    )

    assert resolved.hazard == HazardKind.HEAT
    assert resolved.relation_name == C.REL_ROAD_ACCESS
    assert resolved.prior_mean == pytest.approx(sparse.neutral_prior_mean)
    assert resolved.confidence == pytest.approx(0.0)
    assert not resolved.initialization_allowed
    assert not resolved.regularization_allowed
    assert resolved.resolution_mode == PriorResolutionMode.NEUTRAL_DEFAULT
    assert resolved.source_hazard is None
    assert resolved.source_relation_name is None


def test_explicit_only_policy_returns_neutral_when_not_strict(
    hazard_registry: HazardRegistry,
    relation_registry: RelationRegistry,
) -> None:
    sparse = _sparse_registry(
        _prior(HazardKind.FLOOD, C.REL_TEMPORAL_MEMORY)
    )

    resolved = sparse.resolve(
        HazardKind.HEAT,
        C.REL_TEMPORAL_MEMORY,
        hazard_registry=hazard_registry,
        relation_registry=relation_registry,
        resolution_policy=PriorResolutionPolicy.EXPLICIT_ONLY,
        require_explicit=False,
    )

    assert resolved.resolution_mode == PriorResolutionMode.NEUTRAL_DEFAULT


def test_require_explicit_raises_for_missing_cell(
    hazard_registry: HazardRegistry,
    relation_registry: RelationRegistry,
) -> None:
    sparse = _sparse_registry(
        _prior(HazardKind.FLOOD, C.REL_TEMPORAL_MEMORY)
    )

    with pytest.raises(
        KeyError,
        match="Explicit prior required",
    ):
        sparse.resolve(
            HazardKind.HEAT,
            C.REL_TEMPORAL_MEMORY,
            hazard_registry=hazard_registry,
            relation_registry=relation_registry,
            require_explicit=True,
        )


def test_resolution_decay_parameters_are_bounded(
    hazard_registry: HazardRegistry,
    relation_registry: RelationRegistry,
) -> None:
    sparse = _sparse_registry(
        _prior(HazardKind.FLOOD, C.REL_TEMPORAL_MEMORY)
    )

    with pytest.raises(
        ValueError,
        match="hazard_inheritance_confidence_decay",
    ):
        sparse.resolve(
            HazardKind.PLUVIAL_FLOOD,
            C.REL_TEMPORAL_MEMORY,
            hazard_registry=hazard_registry,
            relation_registry=relation_registry,
            hazard_inheritance_confidence_decay=1.1,
        )

    with pytest.raises(
        ValueError,
        match="relation_inheritance_confidence_decay",
    ):
        sparse.resolve(
            HazardKind.PLUVIAL_FLOOD,
            C.REL_TEMPORAL_MEMORY,
            hazard_registry=hazard_registry,
            relation_registry=relation_registry,
            relation_inheritance_confidence_decay=-0.1,
        )


def test_effective_initialization_mean_uses_confidence_shrinkage() -> None:
    resolved = ResolvedHazardRelationPrior(
        hazard=HazardKind.FLOOD,
        relation_name=C.REL_TEMPORAL_MEMORY,
        prior_mean=0.80,
        confidence=0.50,
        initialization_allowed=True,
        regularization_allowed=False,
        evidence_type=PriorEvidenceType.PROVISIONAL_ONTOLOGY,
        rationale="Synthetic resolved cell.",
        caveat=None,
        resolution_mode=PriorResolutionMode.EXPLICIT,
        source_hazard=HazardKind.FLOOD,
        source_relation_name=C.REL_TEMPORAL_MEMORY,
    )

    assert resolved.effective_initialization_mean(
        neutral_prior_mean=0.50
    ) == pytest.approx(0.65)

    disabled = replace(
        resolved,
        initialization_allowed=False,
    )
    assert disabled.effective_initialization_mean(
        neutral_prior_mean=0.50
    ) == pytest.approx(0.50)


# =============================================================================
# Dense compilation and lookup contracts
# =============================================================================


def test_compile_default_convenience_function(
    compiled_relation_registry: CompiledRelationRegistry,
) -> None:
    compiled = compile_default_hazard_relation_priors(
        compiled_relation_registry,
        application_context=DEFAULT_PRIOR_APPLICATION_CONTEXT,
        hazards=(HazardKind.FLOOD, HazardKind.HEAT),
    )

    assert compiled.hazard_names == ("flood", "heat")
    assert compiled.relation_names == compiled_relation_registry.relation_names


def test_compiled_dimensions_and_stable_ids(
    compiled_priors: CompiledHazardRelationPriors,
    hazard_registry: HazardRegistry,
    compiled_relation_registry: CompiledRelationRegistry,
) -> None:
    assert compiled_priors.schema_version == COMPILED_HAZARD_PRIOR_SCHEMA_VERSION
    assert compiled_priors.num_hazards == 3
    assert compiled_priors.num_relations == len(compiled_relation_registry)

    assert compiled_priors.hazard_names == (
        HazardKind.FLOOD.value,
        HazardKind.HEAT.value,
        HazardKind.PLUVIAL_FLOOD.value,
    )
    assert compiled_priors.stable_hazard_ids == tuple(
        hazard_registry.stable_id_for(name)
        for name in compiled_priors.hazard_names
    )
    assert compiled_priors.relation_names == (
        compiled_relation_registry.relation_names
    )
    assert compiled_priors.stable_relation_ids == (
        compiled_relation_registry.stable_relation_ids
    )


def test_compiled_matrix_shapes(
    compiled_priors: CompiledHazardRelationPriors,
) -> None:
    matrices = (
        compiled_priors.prior_mean_matrix,
        compiled_priors.confidence_matrix,
        compiled_priors.effective_initialization_mean_matrix,
        compiled_priors.initialization_mask,
        compiled_priors.regularization_mask,
        compiled_priors.resolution_mode_matrix,
        compiled_priors.source_hazard_matrix,
        compiled_priors.source_relation_matrix,
        compiled_priors.hazard_inheritance_distance_matrix,
        compiled_priors.relation_inheritance_distance_matrix,
    )

    for matrix in matrices:
        assert len(matrix) == compiled_priors.num_hazards
        assert all(
            len(row) == compiled_priors.num_relations
            for row in matrix
        )


def test_compiled_lookup_helpers_are_restored_and_read_only(
    compiled_priors: CompiledHazardRelationPriors,
) -> None:
    assert compiled_priors.hazard_index(HazardKind.FLOOD) == 0
    assert compiled_priors.hazard_index("heat") == 1

    temporal_index = compiled_priors.relation_index(
        C.REL_TEMPORAL_MEMORY
    )
    assert (
        compiled_priors.relation_names[temporal_index]
        == C.REL_TEMPORAL_MEMORY
    )

    assert isinstance(compiled_priors.hazard_index_by_name, MappingProxyType)
    assert isinstance(compiled_priors.relation_index_by_name, MappingProxyType)

    with pytest.raises(TypeError):
        compiled_priors.hazard_index_by_name["new"] = 99  # type: ignore[index]

    with pytest.raises(TypeError):
        compiled_priors.relation_index_by_name["new"] = 99  # type: ignore[index]

    with pytest.raises(KeyError, match="is not compiled"):
        compiled_priors.hazard_index(HazardKind.FREEZING_RAIN)

    with pytest.raises(KeyError, match="is not compiled"):
        compiled_priors.relation_index("not_a_relation")


def test_complete_default_compilation_resolves_every_cell_explicitly(
    compiled_priors: CompiledHazardRelationPriors,
) -> None:
    assert all(
        mode == PriorResolutionMode.EXPLICIT.value
        for row in compiled_priors.resolution_mode_matrix
        for mode in row
    )
    assert all(
        distance == 0
        for row in compiled_priors.hazard_inheritance_distance_matrix
        for distance in row
    )
    assert all(
        distance == 0
        for row in compiled_priors.relation_inheritance_distance_matrix
        for distance in row
    )


def test_compiled_effective_mean_matches_confidence_adjustment(
    compiled_priors: CompiledHazardRelationPriors,
) -> None:
    hazard_index = compiled_priors.hazard_index(HazardKind.FLOOD)
    relation_index = compiled_priors.relation_index(C.REL_TEMPORAL_MEMORY)

    prior_mean = compiled_priors.prior_mean_matrix[
        hazard_index
    ][relation_index]
    confidence = compiled_priors.confidence_matrix[
        hazard_index
    ][relation_index]
    effective = compiled_priors.effective_initialization_mean_matrix[
        hazard_index
    ][relation_index]

    expected = compiled_priors.neutral_prior_mean + confidence * (
        prior_mean - compiled_priors.neutral_prior_mean
    )

    assert prior_mean == pytest.approx(0.80)
    assert confidence == pytest.approx(0.55)
    assert effective == pytest.approx(expected)


def test_control_columns_have_zero_gate_bias(
    compiled_priors: CompiledHazardRelationPriors,
    relation_registry: RelationRegistry,
) -> None:
    logits = compiled_priors.gate_bias_logit_matrix()
    control_names = tuple(
        entry.name
        for entry in relation_registry.entries
        if entry.specification.is_control
        and entry.name in compiled_priors.relation_names
    )

    assert control_names

    for relation_name in control_names:
        relation_index = compiled_priors.relation_index(relation_name)
        for hazard_index in range(compiled_priors.num_hazards):
            assert not compiled_priors.initialization_mask[
                hazard_index
            ][relation_index]
            assert logits[hazard_index][relation_index] == pytest.approx(0.0)


def test_noncontrol_informative_cell_has_expected_sigmoid_logit(
    compiled_priors: CompiledHazardRelationPriors,
) -> None:
    hazard_index = compiled_priors.hazard_index(HazardKind.FLOOD)
    relation_index = compiled_priors.relation_index(C.REL_TEMPORAL_MEMORY)
    logits = compiled_priors.gate_bias_logit_matrix(
        activation=GateInitializationActivation.SIGMOID,
    )

    effective = compiled_priors.effective_initialization_mean_matrix[
        hazard_index
    ][relation_index]
    expected_logit = math.log(effective / (1.0 - effective))

    assert compiled_priors.initialization_mask[
        hazard_index
    ][relation_index]
    assert logits[hazard_index][relation_index] == pytest.approx(
        expected_logit
    )


def test_gate_logit_epsilon_is_validated(
    compiled_priors: CompiledHazardRelationPriors,
) -> None:
    with pytest.raises(
        ValueError,
        match="epsilon",
    ):
        compiled_priors.gate_bias_logit_matrix(epsilon=0.0)

    with pytest.raises(
        ValueError,
        match="epsilon",
    ):
        compiled_priors.gate_bias_logit_matrix(epsilon=0.5)


def test_default_regularization_weights_are_all_zero(
    compiled_priors: CompiledHazardRelationPriors,
) -> None:
    assert all(
        not enabled
        for row in compiled_priors.regularization_mask
        for enabled in row
    )
    assert all(
        weight == pytest.approx(0.0)
        for row in compiled_priors.regularization_weight_matrix()
        for weight in row
    )


def test_compilation_records_all_source_fingerprints(
    compiled_priors: CompiledHazardRelationPriors,
    prior_registry: HazardRelationPriorRegistry,
    hazard_registry: HazardRegistry,
    relation_registry: RelationRegistry,
    compiled_relation_registry: CompiledRelationRegistry,
) -> None:
    assert (
        compiled_priors.source_prior_registry_fingerprint
        == prior_registry.fingerprint()
    )
    assert (
        compiled_priors.source_applicability_scope_fingerprint
        == prior_registry.applicability_scope.fingerprint()
    )
    assert (
        compiled_priors.source_hazard_semantic_fingerprint
        == hazard_registry.semantic_fingerprint()
    )
    assert (
        compiled_priors.source_hazard_compatibility_fingerprint
        == hazard_registry.compatibility_fingerprint()
    )
    assert (
        compiled_priors.source_hazard_operational_fingerprint
        == hazard_registry.operational_fingerprint()
    )
    assert (
        compiled_priors.source_relation_semantic_fingerprint
        == relation_registry.semantic_fingerprint()
    )
    assert (
        compiled_priors.source_compiled_relation_fingerprint
        == compiled_relation_registry.fingerprint()
    )


def test_compiled_artifact_matches_claimed_sources(
    compiled_priors: CompiledHazardRelationPriors,
    prior_registry: HazardRelationPriorRegistry,
    hazard_registry: HazardRegistry,
    relation_registry: RelationRegistry,
    compiled_relation_registry: CompiledRelationRegistry,
) -> None:
    compiled_priors.assert_matches_sources(
        prior_registry=prior_registry,
        hazard_registry=hazard_registry,
        relation_registry=relation_registry,
        compiled_relation_registry=compiled_relation_registry,
    )


def test_compiled_artifact_rejects_mismatched_source(
    compiled_priors: CompiledHazardRelationPriors,
    prior_registry: HazardRelationPriorRegistry,
    hazard_registry: HazardRegistry,
    relation_registry: RelationRegistry,
    compiled_relation_registry: CompiledRelationRegistry,
) -> None:
    mismatched_hazard_registry = replace(
        hazard_registry,
        registry_name="different_hazard_registry",
    )

    with pytest.raises(ValueError):
        compiled_priors.assert_matches_sources(
            prior_registry=prior_registry,
            hazard_registry=mismatched_hazard_registry,
            relation_registry=relation_registry,
            compiled_relation_registry=compiled_relation_registry,
        )


def test_compilation_rejects_duplicate_hazards(
    prior_registry: HazardRelationPriorRegistry,
    hazard_registry: HazardRegistry,
    relation_registry: RelationRegistry,
    compiled_relation_registry: CompiledRelationRegistry,
) -> None:
    with pytest.raises(
        ValueError,
        match="Duplicate requested hazards",
    ):
        prior_registry.compile(
            compiled_relation_registry,
            source_hazard_registry=hazard_registry,
            source_relation_registry=relation_registry,
            application_context=DEFAULT_PRIOR_APPLICATION_CONTEXT,
            hazards=(HazardKind.FLOOD, HazardKind.FLOOD),
        )


def test_compilation_rejects_out_of_scope_context(
    prior_registry: HazardRelationPriorRegistry,
    hazard_registry: HazardRegistry,
    relation_registry: RelationRegistry,
    compiled_relation_registry: CompiledRelationRegistry,
) -> None:
    wrong_context = replace(
        DEFAULT_PRIOR_APPLICATION_CONTEXT,
        study_region="different_region",
    )

    with pytest.raises(
        ValueError,
        match="not applicable",
    ):
        prior_registry.compile(
            compiled_relation_registry,
            source_hazard_registry=hazard_registry,
            source_relation_registry=relation_registry,
            application_context=wrong_context,
            hazards=(HazardKind.FLOOD,),
        )


def test_training_support_requires_explicit_partial_override(
    prior_registry: HazardRelationPriorRegistry,
    hazard_registry: HazardRegistry,
    relation_registry: RelationRegistry,
    compiled_relation_registry: CompiledRelationRegistry,
) -> None:
    with pytest.raises(
        ValueError,
        match="not approved",
    ):
        prior_registry.compile(
            compiled_relation_registry,
            source_hazard_registry=hazard_registry,
            source_relation_registry=relation_registry,
            application_context=DEFAULT_PRIOR_APPLICATION_CONTEXT,
            hazards=(HazardKind.PLUVIAL_FLOOD,),
            require_training_supported_hazards=True,
            allow_partially_data_backed=False,
        )

    compiled = prior_registry.compile(
        compiled_relation_registry,
        source_hazard_registry=hazard_registry,
        source_relation_registry=relation_registry,
        application_context=DEFAULT_PRIOR_APPLICATION_CONTEXT,
        hazards=(HazardKind.PLUVIAL_FLOOD,),
        require_training_supported_hazards=True,
        allow_partially_data_backed=True,
    )

    assert compiled.hazard_names == (HazardKind.PLUVIAL_FLOOD.value,)


def test_planned_hazard_remains_training_unsupported(
    prior_registry: HazardRelationPriorRegistry,
    hazard_registry: HazardRegistry,
    relation_registry: RelationRegistry,
    compiled_relation_registry: CompiledRelationRegistry,
) -> None:
    with pytest.raises(
        ValueError,
        match="not approved",
    ):
        prior_registry.compile(
            compiled_relation_registry,
            source_hazard_registry=hazard_registry,
            source_relation_registry=relation_registry,
            application_context=DEFAULT_PRIOR_APPLICATION_CONTEXT,
            hazards=(HazardKind.HEAT,),
            require_training_supported_hazards=True,
            allow_partially_data_backed=True,
        )


def test_fallback_hazard_requires_explicit_compile_permission(
    prior_registry: HazardRelationPriorRegistry,
    hazard_registry: HazardRegistry,
    relation_registry: RelationRegistry,
    compiled_relation_registry: CompiledRelationRegistry,
) -> None:
    with pytest.raises(
        ValueError,
        match="fallback-only",
    ):
        prior_registry.compile(
            compiled_relation_registry,
            source_hazard_registry=hazard_registry,
            source_relation_registry=relation_registry,
            application_context=DEFAULT_PRIOR_APPLICATION_CONTEXT,
            hazards=(HazardKind.ALL_HAZARD,),
        )

    compiled = prior_registry.compile(
        compiled_relation_registry,
        source_hazard_registry=hazard_registry,
        source_relation_registry=relation_registry,
        application_context=DEFAULT_PRIOR_APPLICATION_CONTEXT,
        hazards=(HazardKind.ALL_HAZARD,),
        allow_fallback_hazard=True,
    )

    assert compiled.hazard_names == (HazardKind.ALL_HAZARD.value,)


# =============================================================================
# Serialization and tamper detection
# =============================================================================


def test_prior_registry_json_round_trip(
    prior_registry: HazardRelationPriorRegistry,
    hazard_registry: HazardRegistry,
    relation_registry: RelationRegistry,
) -> None:
    payload = json.loads(
        json.dumps(prior_registry.to_dict())
    )

    reconstructed = HazardRelationPriorRegistry.from_dict(
        payload,
        hazard_registry=hazard_registry,
        relation_registry=relation_registry,
        require_current_versions=True,
        require_complete=True,
    )

    assert reconstructed == prior_registry
    assert reconstructed.fingerprint() == prior_registry.fingerprint()


def test_prior_registry_reorders_serialized_cells_canonically(
    prior_registry: HazardRelationPriorRegistry,
    hazard_registry: HazardRegistry,
    relation_registry: RelationRegistry,
) -> None:
    payload = deepcopy(prior_registry.to_dict())
    payload["priors"] = list(reversed(payload["priors"]))

    reconstructed = HazardRelationPriorRegistry.from_dict(
        payload,
        hazard_registry=hazard_registry,
        relation_registry=relation_registry,
        require_current_versions=True,
        require_complete=True,
    )

    assert reconstructed.priors == prior_registry.priors
    assert reconstructed.fingerprint() == prior_registry.fingerprint()


def test_tampered_prior_registry_fingerprint_is_rejected(
    prior_registry: HazardRelationPriorRegistry,
) -> None:
    payload = deepcopy(prior_registry.to_dict())
    payload["fingerprint"] = "tampered"

    with pytest.raises(
        ValueError,
        match="fingerprint",
    ):
        HazardRelationPriorRegistry.from_dict(payload)


def test_unknown_prior_registry_field_is_rejected(
    prior_registry: HazardRelationPriorRegistry,
) -> None:
    payload = deepcopy(prior_registry.to_dict())
    payload["unexpected_field"] = "unexpected"

    with pytest.raises(
        ValueError,
        match="Unknown fields",
    ):
        HazardRelationPriorRegistry.from_dict(payload)


def test_compiled_prior_json_round_trip(
    compiled_priors: CompiledHazardRelationPriors,
) -> None:
    reconstructed = CompiledHazardRelationPriors.from_dict(
        json.loads(
            json.dumps(compiled_priors.to_dict())
        )
    )

    assert reconstructed == compiled_priors
    assert reconstructed.fingerprint() == compiled_priors.fingerprint()
    assert isinstance(reconstructed.hazard_names, tuple)
    assert isinstance(reconstructed.prior_mean_matrix, tuple)
    assert isinstance(reconstructed.prior_mean_matrix[0], tuple)


def test_compiled_prior_rejects_unknown_field(
    compiled_priors: CompiledHazardRelationPriors,
) -> None:
    payload = deepcopy(compiled_priors.to_dict())
    payload["unexpected_field"] = "unexpected"

    with pytest.raises(
        ValueError,
        match="Unknown fields",
    ):
        CompiledHazardRelationPriors.from_dict(payload)


def test_compiled_prior_rejects_wrong_matrix_shape(
    compiled_priors: CompiledHazardRelationPriors,
) -> None:
    with pytest.raises(
        ValueError,
        match="prior_mean_matrix",
    ):
        replace(
            compiled_priors,
            prior_mean_matrix=(
                compiled_priors.prior_mean_matrix[:-1]
            ),
        )


# =============================================================================
# Immutability
# =============================================================================


def test_prior_cell_is_frozen(
    prior_registry: HazardRelationPriorRegistry,
) -> None:
    cell = prior_registry.priors[0]

    with pytest.raises(FrozenInstanceError):
        cell.relation_name = "mutated"  # type: ignore[misc]


def test_prior_registry_is_frozen(
    prior_registry: HazardRelationPriorRegistry,
) -> None:
    with pytest.raises(FrozenInstanceError):
        prior_registry.registry_name = "mutated"  # type: ignore[misc]


def test_compiled_prior_is_frozen(
    compiled_priors: CompiledHazardRelationPriors,
) -> None:
    with pytest.raises(FrozenInstanceError):
        compiled_priors.neutral_prior_mean = 0.4  # type: ignore[misc]


def test_profile_override_map_is_read_only() -> None:
    profile = RelationPriorProfile(
        relation_name=C.REL_TEMPORAL_MEMORY,
        default_definition=_definition(),
        overrides=(
            (
                HazardKind.FLOOD,
                _definition(confidence=0.60),
            ),
        ),
    )

    assert isinstance(profile.override_map, MappingProxyType)

    with pytest.raises(TypeError):
        profile.override_map[HazardKind.HEAT] = _definition()  # type: ignore[index]
