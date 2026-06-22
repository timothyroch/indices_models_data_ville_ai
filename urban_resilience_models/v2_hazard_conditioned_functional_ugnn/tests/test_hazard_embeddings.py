"""
Contract tests for V2 hazard embeddings.

Target repository path:
    urban_resilience_models/
        v2_hazard_conditioned_functional_ugnn/
            tests/
                test_hazard_embeddings.py

Implementation under test:
    urban_resilience_models/
        v2_hazard_conditioned_functional_ugnn/
            hazard/
                hazard_embeddings.py

These tests intentionally freeze the corrected hazard-embedding contract:

- hazard semantics come only from hazard_registry.py;
- sparse stable IDs are never used as dense tensor indices;
- configuration vocabulary exactly matches implementation vocabulary;
- learned, fixed, and fixed-plus-residual modes are reproducible;
- explicit unknown handling follows both input and row policies;
- a fixed-zero unknown row remains zero during optimization;
- encoded metadata must agree with dense indices;
- fixed artifacts preserve dtype, identity, and provenance;
- graph-to-node broadcasting preserves semantic metadata;
- outputs and exported artifacts remain finite and auditable.

The suite targets the corrected implementation, not the older draft that imported
hazard identity from relations/hazard_relation_priors.py.
"""

from __future__ import annotations

from dataclasses import replace
import json
from types import MappingProxyType
from typing import Any

import pytest
import torch

from urban_resilience_models.v2_hazard_conditioned_functional_ugnn.config import (
    CANONICAL_HAZARD_EMBEDDING_INITIALIZATIONS,
    CANONICAL_HAZARD_EMBEDDING_MODES,
    HAZARD_EMBEDDING_INIT_NORMAL,
    HAZARD_EMBEDDING_INIT_XAVIER_UNIFORM,
    HAZARD_EMBEDDING_INIT_ZERO,
    HAZARD_EMBEDDING_MODE_FIXED,
    HAZARD_EMBEDDING_MODE_FIXED_PLUS_RESIDUAL,
    HAZARD_EMBEDDING_MODE_LEARNED,
    HazardEmbeddingConfig,
    UNKNOWN_EMBEDDING_POLICY_ERROR,
    UNKNOWN_EMBEDDING_POLICY_LEARNED,
    UNKNOWN_EMBEDDING_POLICY_ZERO_FIXED,
    UNKNOWN_ID_POLICY_ERROR,
    UNKNOWN_ID_POLICY_USE_UNKNOWN_EMBEDDING,
)
from urban_resilience_models.v2_hazard_conditioned_functional_ugnn.hazard.hazard_embeddings import (
    DEFAULT_RUNTIME_HAZARD_EMBEDDING_VOCABULARY,
    FixedHazardEmbeddingArtifact,
    HazardEmbeddingInitialization,
    HazardEmbeddingLayer,
    HazardEmbeddingLookup,
    HazardEmbeddingMode,
    HazardEmbeddingVocabulary,
    HazardIndexBatch,
    NodeAlignedHazardEmbeddingLookup,
    UnknownHazardPolicy,
    broadcast_graph_embeddings_to_nodes,
    build_hazard_embedding_vocabulary,
)
from urban_resilience_models.v2_hazard_conditioned_functional_ugnn.hazard.hazard_registry import (
    DEFAULT_HAZARD_REGISTRY,
    HAZARD_ID_UNKNOWN,
    UNKNOWN_HAZARD_NAME,
    HazardKind,
    HazardRegistry,
)


EMBEDDING_DIM = 4

EXPECTED_DEFAULT_HAZARDS = (
    HazardKind.FLOOD.value,
    HazardKind.RIVERINE_FLOOD.value,
    HazardKind.PLUVIAL_FLOOD.value,
    HazardKind.HEAT.value,
    HazardKind.OUTAGE.value,
    HazardKind.ROAD_DISRUPTION.value,
    HazardKind.CIVIL_SECURITY_EVENT.value,
    HazardKind.WINTER_STORM.value,
    HazardKind.SNOWSTORM.value,
    HazardKind.FREEZING_RAIN.value,
)


# =============================================================================
# Helpers
# =============================================================================


@pytest.fixture()
def hazard_registry() -> HazardRegistry:
    return DEFAULT_HAZARD_REGISTRY


@pytest.fixture()
def runtime_vocabulary(
    hazard_registry: HazardRegistry,
) -> HazardEmbeddingVocabulary:
    return build_hazard_embedding_vocabulary(
        hazard_registry=hazard_registry,
    )


@pytest.fixture()
def unknown_vocabulary(
    hazard_registry: HazardRegistry,
) -> HazardEmbeddingVocabulary:
    return build_hazard_embedding_vocabulary(
        hazard_registry=hazard_registry,
        include_unknown=True,
    )


def _matrix_for(
    vocabulary: HazardEmbeddingVocabulary,
    *,
    dtype: torch.dtype = torch.float32,
    embedding_dim: int = EMBEDDING_DIM,
) -> torch.Tensor:
    values = torch.arange(
        len(vocabulary) * embedding_dim,
        dtype=torch.float64,
    ).reshape(len(vocabulary), embedding_dim)
    return values.to(dtype=dtype) / 10.0


def _artifact(
    vocabulary: HazardEmbeddingVocabulary,
    *,
    matrix: torch.Tensor | None = None,
    dtype: torch.dtype = torch.float32,
    source_id: str = "unit-test-fixed-hazards",
    source_version: str = "1",
    source_fingerprint: str = "unit-test-source-fingerprint",
) -> FixedHazardEmbeddingArtifact:
    if matrix is None:
        matrix = _matrix_for(vocabulary, dtype=dtype)

    return FixedHazardEmbeddingArtifact(
        embedding_matrix=matrix,
        hazard_names=vocabulary.hazard_names,
        stable_hazard_ids=vocabulary.stable_hazard_ids,
        vocabulary_fingerprint=vocabulary.fingerprint(),
        source_id=source_id,
        source_version=source_version,
        source_fingerprint=source_fingerprint,
    )


def _learned_config(
    **changes: Any,
) -> HazardEmbeddingConfig:
    base = HazardEmbeddingConfig(
        embedding_dim=EMBEDDING_DIM,
        mode=HAZARD_EMBEDDING_MODE_LEARNED,
        initialization=HAZARD_EMBEDDING_INIT_NORMAL,
        initialization_seed=17,
        initialization_std=0.05,
    )
    return base.replace(**changes)


def _unknown_config(
    *,
    unknown_embedding_policy: str,
    **changes: Any,
) -> HazardEmbeddingConfig:
    base = HazardEmbeddingConfig(
        embedding_dim=EMBEDDING_DIM,
        mode=HAZARD_EMBEDDING_MODE_LEARNED,
        unknown_input_policy=(
            UNKNOWN_ID_POLICY_USE_UNKNOWN_EMBEDDING
        ),
        unknown_embedding_policy=unknown_embedding_policy,
        include_unknown_row=True,
        initialization=HAZARD_EMBEDDING_INIT_NORMAL,
        initialization_seed=23,
        initialization_std=0.05,
    )
    return base.replace(**changes)


def _fixed_config(
    artifact: FixedHazardEmbeddingArtifact,
    *,
    mode: str = HAZARD_EMBEDDING_MODE_FIXED,
    initialization: str = HAZARD_EMBEDDING_INIT_ZERO,
    fixed_scale: float = 1.0,
    residual_scale: float = 1.0,
    **changes: Any,
) -> HazardEmbeddingConfig:
    base = HazardEmbeddingConfig(
        embedding_dim=artifact.embedding_dim,
        mode=mode,
        initialization=initialization,
        initialization_seed=29,
        initialization_std=0.0,
        fixed_artifact_reference="memory://unit-test-fixed-hazards",
        fixed_artifact_fingerprint=artifact.embedding_fingerprint,
        fixed_scale=fixed_scale,
        residual_scale=residual_scale,
    )
    return base.replace(**changes)


def _layer(
    config: HazardEmbeddingConfig,
    *,
    hazard_registry: HazardRegistry = DEFAULT_HAZARD_REGISTRY,
    fixed_artifact: FixedHazardEmbeddingArtifact | None = None,
) -> HazardEmbeddingLayer:
    """
    Exercise the required config-to-module construction boundary.

    ``from_config`` is the authoritative integration point. It prevents model
    builders from manually translating configuration fields and accidentally
    omitting them from experiment identity.
    """

    return HazardEmbeddingLayer.from_config(
        config,
        hazard_registry=hazard_registry,
        fixed_artifact=fixed_artifact,
    )


# =============================================================================
# Configuration and implementation vocabulary alignment
# =============================================================================


def test_embedding_mode_values_match_config() -> None:
    assert tuple(mode.value for mode in HazardEmbeddingMode) == (
        CANONICAL_HAZARD_EMBEDDING_MODES
    )
    assert HAZARD_EMBEDDING_MODE_LEARNED == "learned"
    assert HAZARD_EMBEDDING_MODE_FIXED == "fixed"
    assert (
        HAZARD_EMBEDDING_MODE_FIXED_PLUS_RESIDUAL
        == "fixed_plus_residual"
    )


def test_initialization_values_match_config() -> None:
    assert tuple(
        initialization.value
        for initialization in HazardEmbeddingInitialization
    ) == CANONICAL_HAZARD_EMBEDDING_INITIALIZATIONS

    assert HAZARD_EMBEDDING_INIT_NORMAL == "normal"
    assert HAZARD_EMBEDDING_INIT_ZERO == "zero"
    assert HAZARD_EMBEDDING_INIT_XAVIER_UNIFORM == "xavier_uniform"


def test_unknown_input_policy_values_match_config() -> None:
    assert {
        policy.value
        for policy in UnknownHazardPolicy
    } == {
        UNKNOWN_ID_POLICY_ERROR,
        UNKNOWN_ID_POLICY_USE_UNKNOWN_EMBEDDING,
    }


def test_config_fields_are_preserved_by_layer(
    hazard_registry: HazardRegistry,
) -> None:
    config = _unknown_config(
        unknown_embedding_policy=UNKNOWN_EMBEDDING_POLICY_LEARNED,
        include_fallback_hazard=True,
        fixed_scale=0.75,
        residual_scale=0.25,
        require_training_supported_hazards=False,
        allow_partially_data_backed_for_training=True,
        require_queryable_hazards_for_inference=True,
        allow_fallback_hazard_for_inference=True,
        allow_planned_hazard_counterfactuals=True,
    )
    layer = _layer(config, hazard_registry=hazard_registry)

    assert layer.embedding_dim == config.embedding_dim
    assert layer.mode.value == config.mode
    assert layer.unknown_policy.value == config.unknown_input_policy
    assert (
        layer.unknown_embedding_policy
        == config.unknown_embedding_policy
    )
    assert layer.fixed_scale == pytest.approx(config.fixed_scale)
    assert layer.residual_scale == pytest.approx(config.residual_scale)

    assert layer.require_training_supported_hazards is False
    assert layer.allow_partially_data_backed_for_training is True
    assert layer.require_queryable_hazards_for_inference is True
    assert layer.allow_fallback_hazard_for_inference is True
    assert layer.allow_planned_hazard_counterfactuals is True


# =============================================================================
# Vocabulary construction and neutral-registry ownership
# =============================================================================


def test_default_vocabulary_uses_exact_queryable_hazard_order(
    runtime_vocabulary: HazardEmbeddingVocabulary,
) -> None:
    assert runtime_vocabulary.hazard_names == EXPECTED_DEFAULT_HAZARDS
    assert UNKNOWN_HAZARD_NAME not in runtime_vocabulary.hazard_names
    assert HazardKind.ALL_HAZARD.value not in (
        runtime_vocabulary.hazard_names
    )


def test_default_vocabulary_singleton_matches_builder(
    runtime_vocabulary: HazardEmbeddingVocabulary,
) -> None:
    assert (
        DEFAULT_RUNTIME_HAZARD_EMBEDDING_VOCABULARY
        == runtime_vocabulary
    )
    assert (
        DEFAULT_RUNTIME_HAZARD_EMBEDDING_VOCABULARY.fingerprint()
        == runtime_vocabulary.fingerprint()
    )


def test_vocabulary_copies_identity_from_hazard_registry(
    hazard_registry: HazardRegistry,
    runtime_vocabulary: HazardEmbeddingVocabulary,
) -> None:
    assert runtime_vocabulary.stable_hazard_ids == tuple(
        hazard_registry.stable_id_for(name)
        for name in runtime_vocabulary.hazard_names
    )

    for vocabulary_entry in runtime_vocabulary:
        registry_entry = hazard_registry.get_by_name(
            vocabulary_entry.name
        )
        assert (
            vocabulary_entry.stable_hazard_id
            == registry_entry.stable_hazard_id
        )
        assert (
            vocabulary_entry.display_name
            == registry_entry.display_name
        )
        assert (
            vocabulary_entry.support_state
            == registry_entry.support_state
        )


def test_vocabulary_dense_indices_are_contiguous_and_read_only(
    runtime_vocabulary: HazardEmbeddingVocabulary,
) -> None:
    assert tuple(
        runtime_vocabulary.dense_index_by_name[name]
        for name in runtime_vocabulary.hazard_names
    ) == tuple(range(len(runtime_vocabulary)))

    assert isinstance(
        runtime_vocabulary.dense_index_by_name,
        MappingProxyType,
    )
    assert isinstance(
        runtime_vocabulary.dense_index_by_stable_id,
        MappingProxyType,
    )

    with pytest.raises(TypeError):
        runtime_vocabulary.dense_index_by_name["new"] = 99  # type: ignore[index]


def test_fallback_and_unknown_rows_are_explicit_opt_ins(
    hazard_registry: HazardRegistry,
) -> None:
    vocabulary = build_hazard_embedding_vocabulary(
        hazard_registry=hazard_registry,
        include_fallback_hazard=True,
        include_unknown=True,
    )

    assert vocabulary.hazard_names[:-2] == EXPECTED_DEFAULT_HAZARDS
    assert vocabulary.hazard_names[-2:] == (
        HazardKind.ALL_HAZARD.value,
        UNKNOWN_HAZARD_NAME,
    )
    assert vocabulary.stable_hazard_ids[-1] == HAZARD_ID_UNKNOWN
    assert vocabulary.unknown_index == len(vocabulary) - 1


def test_vocabulary_round_trip_and_fingerprint_validation(
    hazard_registry: HazardRegistry,
    runtime_vocabulary: HazardEmbeddingVocabulary,
) -> None:
    payload = json.loads(
        json.dumps(runtime_vocabulary.to_dict())
    )

    reconstructed = HazardEmbeddingVocabulary.from_dict(
        payload,
        hazard_registry=hazard_registry,
    )

    assert reconstructed == runtime_vocabulary
    assert reconstructed.fingerprint() == (
        runtime_vocabulary.fingerprint()
    )

    payload["entries"][0]["display_name"] = "Tampered"
    with pytest.raises(ValueError, match="fingerprint"):
        HazardEmbeddingVocabulary.from_dict(
            payload,
            hazard_registry=hazard_registry,
        )


# =============================================================================
# Name/stable-ID encoding and unknown-input contracts
# =============================================================================


def test_name_and_stable_id_encoding_round_trip(
    runtime_vocabulary: HazardEmbeddingVocabulary,
) -> None:
    names = (
        HazardKind.FLOOD,
        HazardKind.HEAT,
        HazardKind.FREEZING_RAIN,
    )
    encoded = runtime_vocabulary.encode_names(
        names,
        unknown_policy=UNKNOWN_ID_POLICY_ERROR,
    )

    assert encoded.hazard_names == tuple(
        hazard.value for hazard in names
    )
    assert runtime_vocabulary.decode_indices(
        encoded.dense_indices.tolist()
    ) == encoded.hazard_names

    reencoded = runtime_vocabulary.encode_stable_ids(
        encoded.stable_hazard_ids.tolist(),
        unknown_policy=UNKNOWN_ID_POLICY_ERROR,
    )
    assert torch.equal(
        reencoded.dense_indices,
        encoded.dense_indices,
    )
    assert reencoded.hazard_names == encoded.hazard_names


def test_unrecognized_inputs_are_rejected_under_error_policy(
    runtime_vocabulary: HazardEmbeddingVocabulary,
) -> None:
    with pytest.raises(ValueError, match="Unknown hazard"):
        runtime_vocabulary.encode_names(
            ("not_a_hazard",),
            unknown_policy=UNKNOWN_ID_POLICY_ERROR,
        )

    with pytest.raises(ValueError, match="Unknown stable hazard"):
        runtime_vocabulary.encode_stable_ids(
            (123_456_789,),
            unknown_policy=UNKNOWN_ID_POLICY_ERROR,
        )


def test_explicit_unknown_is_rejected_under_error_policy(
    unknown_vocabulary: HazardEmbeddingVocabulary,
) -> None:
    with pytest.raises(ValueError, match="Unknown hazard"):
        unknown_vocabulary.encode_names(
            (UNKNOWN_HAZARD_NAME,),
            unknown_policy=UNKNOWN_ID_POLICY_ERROR,
        )

    with pytest.raises(ValueError, match="Unknown stable hazard"):
        unknown_vocabulary.encode_stable_ids(
            (HAZARD_ID_UNKNOWN,),
            unknown_policy=UNKNOWN_ID_POLICY_ERROR,
        )


def test_unknown_inputs_map_to_one_explicit_row(
    unknown_vocabulary: HazardEmbeddingVocabulary,
) -> None:
    encoded = unknown_vocabulary.encode_names(
        ("not_a_hazard", UNKNOWN_HAZARD_NAME),
        unknown_policy=(
            UNKNOWN_ID_POLICY_USE_UNKNOWN_EMBEDDING
        ),
    )

    assert encoded.hazard_names == (
        UNKNOWN_HAZARD_NAME,
        UNKNOWN_HAZARD_NAME,
    )
    assert encoded.unknown_mask.tolist() == [True, True]
    assert encoded.dense_indices.tolist() == [
        unknown_vocabulary.unknown_index,
        unknown_vocabulary.unknown_index,
    ]
    assert encoded.stable_hazard_ids.tolist() == [
        HAZARD_ID_UNKNOWN,
        HAZARD_ID_UNKNOWN,
    ]


def test_index_batch_detects_semantic_identity_mismatch(
    runtime_vocabulary: HazardEmbeddingVocabulary,
) -> None:
    valid = runtime_vocabulary.encode_names(
        (HazardKind.FLOOD,),
        unknown_policy=UNKNOWN_ID_POLICY_ERROR,
    )

    wrong_name = HazardIndexBatch(
        dense_indices=valid.dense_indices,
        stable_hazard_ids=valid.stable_hazard_ids,
        hazard_names=(HazardKind.HEAT.value,),
        unknown_mask=valid.unknown_mask,
        vocabulary_fingerprint=valid.vocabulary_fingerprint,
    )
    with pytest.raises(ValueError, match="hazard name"):
        wrong_name.assert_matches_vocabulary(runtime_vocabulary)

    wrong_stable_id = HazardIndexBatch(
        dense_indices=valid.dense_indices,
        stable_hazard_ids=torch.tensor(
            [runtime_vocabulary.stable_hazard_ids[1]],
            dtype=torch.long,
        ),
        hazard_names=valid.hazard_names,
        unknown_mask=valid.unknown_mask,
        vocabulary_fingerprint=valid.vocabulary_fingerprint,
    )
    with pytest.raises(ValueError, match="stable hazard"):
        wrong_stable_id.assert_matches_vocabulary(runtime_vocabulary)

    wrong_unknown_mask = HazardIndexBatch(
        dense_indices=valid.dense_indices,
        stable_hazard_ids=valid.stable_hazard_ids,
        hazard_names=valid.hazard_names,
        unknown_mask=torch.tensor([True]),
        vocabulary_fingerprint=valid.vocabulary_fingerprint,
    )
    with pytest.raises(ValueError, match="unknown"):
        wrong_unknown_mask.assert_matches_vocabulary(runtime_vocabulary)


# =============================================================================
# Learned embeddings and deterministic initialization
# =============================================================================


@pytest.mark.parametrize(
    "initialization",
    (
        HAZARD_EMBEDDING_INIT_NORMAL,
        HAZARD_EMBEDDING_INIT_XAVIER_UNIFORM,
        HAZARD_EMBEDDING_INIT_ZERO,
    ),
)
def test_seeded_learned_initialization_is_reproducible(
    initialization: str,
    hazard_registry: HazardRegistry,
) -> None:
    config = _learned_config(
        initialization=initialization,
        initialization_std=(
            0.05
            if initialization == HAZARD_EMBEDDING_INIT_NORMAL
            else 0.0
        ),
    )

    first = _layer(config, hazard_registry=hazard_registry)
    second = _layer(config, hazard_registry=hazard_registry)

    assert first.learned_embeddings is not None
    assert second.learned_embeddings is not None
    assert torch.equal(
        first.learned_embeddings.weight,
        second.learned_embeddings.weight,
    )


def test_different_initialization_seed_changes_learned_table(
    hazard_registry: HazardRegistry,
) -> None:
    first = _layer(
        _learned_config(initialization_seed=1),
        hazard_registry=hazard_registry,
    )
    second = _layer(
        _learned_config(initialization_seed=2),
        hazard_registry=hazard_registry,
    )

    assert first.learned_embeddings is not None
    assert second.learned_embeddings is not None
    assert not torch.equal(
        first.learned_embeddings.weight,
        second.learned_embeddings.weight,
    )


def test_learned_lookup_shape_identity_and_finiteness(
    hazard_registry: HazardRegistry,
) -> None:
    layer = _layer(
        _learned_config(),
        hazard_registry=hazard_registry,
    )
    lookup = layer.lookup_names(
        (
            HazardKind.FLOOD,
            HazardKind.HEAT,
            HazardKind.OUTAGE,
        )
    )

    assert isinstance(lookup, HazardEmbeddingLookup)
    assert lookup.embeddings.shape == (3, EMBEDDING_DIM)
    assert lookup.indices.hazard_names == (
        HazardKind.FLOOD.value,
        HazardKind.HEAT.value,
        HazardKind.OUTAGE.value,
    )
    assert bool(torch.isfinite(lookup.embeddings).all().item())


def test_freeze_learned_embeddings_disables_gradients(
    hazard_registry: HazardRegistry,
) -> None:
    layer = _layer(
        _learned_config(freeze_learned_embeddings=True),
        hazard_registry=hazard_registry,
    )

    assert layer.learned_embeddings is not None
    assert not layer.learned_embeddings.weight.requires_grad
    assert not layer.trainable


# =============================================================================
# Unknown embedding-row behavior
# =============================================================================


def test_zero_fixed_unknown_row_uses_padding_idx(
    hazard_registry: HazardRegistry,
) -> None:
    layer = _layer(
        _unknown_config(
            unknown_embedding_policy=(
                UNKNOWN_EMBEDDING_POLICY_ZERO_FIXED
            )
        ),
        hazard_registry=hazard_registry,
    )

    assert layer.learned_embeddings is not None
    unknown_index = layer.vocabulary.unknown_index
    assert unknown_index is not None
    assert layer.learned_embeddings.padding_idx == unknown_index
    assert torch.equal(
        layer.learned_embeddings.weight[unknown_index],
        torch.zeros(EMBEDDING_DIM),
    )


def test_zero_fixed_unknown_row_remains_zero_after_optimizer_step(
    hazard_registry: HazardRegistry,
) -> None:
    layer = _layer(
        _unknown_config(
            unknown_embedding_policy=(
                UNKNOWN_EMBEDDING_POLICY_ZERO_FIXED
            )
        ),
        hazard_registry=hazard_registry,
    )

    assert layer.learned_embeddings is not None
    unknown_index = layer.vocabulary.unknown_index
    assert unknown_index is not None

    optimizer = torch.optim.SGD(layer.parameters(), lr=0.5)
    optimizer.zero_grad()

    lookup = layer.lookup_names(("unrecognized",))
    lookup.embeddings.sum().backward()
    optimizer.step()

    assert torch.equal(
        layer.learned_embeddings.weight[unknown_index],
        torch.zeros(EMBEDDING_DIM),
    )


def test_learned_unknown_row_can_receive_updates(
    hazard_registry: HazardRegistry,
) -> None:
    layer = _layer(
        _unknown_config(
            unknown_embedding_policy=(
                UNKNOWN_EMBEDDING_POLICY_LEARNED
            )
        ),
        hazard_registry=hazard_registry,
    )

    assert layer.learned_embeddings is not None
    unknown_index = layer.vocabulary.unknown_index
    assert unknown_index is not None
    assert layer.learned_embeddings.padding_idx is None

    before = (
        layer.learned_embeddings.weight[unknown_index]
        .detach()
        .clone()
    )

    optimizer = torch.optim.SGD(layer.parameters(), lr=0.5)
    optimizer.zero_grad()
    layer.lookup_names(("unrecognized",)).embeddings.sum().backward()
    optimizer.step()

    after = (
        layer.learned_embeddings.weight[unknown_index]
        .detach()
        .clone()
    )
    assert not torch.equal(before, after)


# =============================================================================
# Fixed artifacts
# =============================================================================


def test_fixed_artifact_clones_input_and_preserves_dtype(
    runtime_vocabulary: HazardEmbeddingVocabulary,
) -> None:
    source = _matrix_for(
        runtime_vocabulary,
        dtype=torch.float64,
    )
    expected = source.clone()

    artifact = _artifact(
        runtime_vocabulary,
        matrix=source,
    )
    source.zero_()

    assert artifact.embedding_matrix.dtype == torch.float64
    assert torch.equal(
        artifact.embedding_matrix,
        expected,
    )
    assert artifact.embedding_matrix.device.type == "cpu"
    assert not artifact.embedding_matrix.requires_grad


def test_fixed_artifact_rejects_nonfinite_values(
    runtime_vocabulary: HazardEmbeddingVocabulary,
) -> None:
    matrix = _matrix_for(runtime_vocabulary)
    matrix[0, 0] = float("nan")

    with pytest.raises(ValueError, match="finite|NaN"):
        _artifact(runtime_vocabulary, matrix=matrix)


def test_fixed_artifact_round_trip_preserves_dtype_and_fingerprint(
    runtime_vocabulary: HazardEmbeddingVocabulary,
) -> None:
    artifact = _artifact(
        runtime_vocabulary,
        dtype=torch.float64,
    )
    payload = json.loads(
        json.dumps(artifact.to_dict(include_values=True))
    )

    assert payload["dtype"] == "torch.float64"

    reconstructed = FixedHazardEmbeddingArtifact.from_dict(
        payload
    )

    assert reconstructed.embedding_matrix.dtype == torch.float64
    assert torch.equal(
        reconstructed.embedding_matrix,
        artifact.embedding_matrix,
    )
    assert reconstructed.embedding_fingerprint == (
        artifact.embedding_fingerprint
    )


def test_fixed_artifact_rejects_vocabulary_mismatch(
    hazard_registry: HazardRegistry,
    runtime_vocabulary: HazardEmbeddingVocabulary,
) -> None:
    artifact = _artifact(runtime_vocabulary)
    expanded = build_hazard_embedding_vocabulary(
        hazard_registry=hazard_registry,
        include_fallback_hazard=True,
    )

    with pytest.raises(ValueError, match="vocabulary|ordering|stable"):
        artifact.assert_matches_vocabulary(expanded)


def test_config_fingerprint_must_match_supplied_fixed_artifact(
    hazard_registry: HazardRegistry,
    runtime_vocabulary: HazardEmbeddingVocabulary,
) -> None:
    artifact = _artifact(runtime_vocabulary)
    config = _fixed_config(artifact).replace(
        fixed_artifact_fingerprint="wrong-fingerprint"
    )

    with pytest.raises(ValueError, match="fingerprint"):
        _layer(
            config,
            hazard_registry=hazard_registry,
            fixed_artifact=artifact,
        )


# =============================================================================
# Fixed and residual mode behavior
# =============================================================================


def test_fixed_mode_returns_scaled_artifact_exactly(
    hazard_registry: HazardRegistry,
    runtime_vocabulary: HazardEmbeddingVocabulary,
) -> None:
    artifact = _artifact(runtime_vocabulary)
    config = _fixed_config(
        artifact,
        fixed_scale=0.25,
    )
    layer = _layer(
        config,
        hazard_registry=hazard_registry,
        fixed_artifact=artifact,
    )

    table = layer.full_embedding_table().embeddings
    assert torch.equal(
        table,
        artifact.embedding_matrix * 0.25,
    )
    assert layer.learned_embeddings is None
    assert not layer.trainable


def test_zero_initialized_residual_starts_at_scaled_fixed_table(
    hazard_registry: HazardRegistry,
    runtime_vocabulary: HazardEmbeddingVocabulary,
) -> None:
    artifact = _artifact(runtime_vocabulary)
    config = _fixed_config(
        artifact,
        mode=HAZARD_EMBEDDING_MODE_FIXED_PLUS_RESIDUAL,
        initialization=HAZARD_EMBEDDING_INIT_ZERO,
        fixed_scale=0.5,
        residual_scale=0.75,
    )
    layer = _layer(
        config,
        hazard_registry=hazard_registry,
        fixed_artifact=artifact,
    )

    assert layer.learned_embeddings is not None
    assert torch.equal(
        layer.learned_embeddings.weight,
        torch.zeros_like(layer.learned_embeddings.weight),
    )
    assert torch.equal(
        layer.full_embedding_table().embeddings,
        artifact.embedding_matrix * 0.5,
    )


def test_residual_scale_controls_effective_table(
    hazard_registry: HazardRegistry,
    runtime_vocabulary: HazardEmbeddingVocabulary,
) -> None:
    artifact = _artifact(runtime_vocabulary)
    config = _fixed_config(
        artifact,
        mode=HAZARD_EMBEDDING_MODE_FIXED_PLUS_RESIDUAL,
        initialization=HAZARD_EMBEDDING_INIT_ZERO,
        fixed_scale=0.5,
        residual_scale=0.25,
    )
    layer = _layer(
        config,
        hazard_registry=hazard_registry,
        fixed_artifact=artifact,
    )

    assert layer.learned_embeddings is not None
    with torch.no_grad():
        layer.learned_embeddings.weight.fill_(2.0)

    expected = (
        artifact.embedding_matrix * 0.5
        + torch.full_like(artifact.embedding_matrix, 0.5)
    )
    assert torch.equal(
        layer.full_embedding_table().embeddings,
        expected,
    )


# =============================================================================
# Batch identity and output validation
# =============================================================================


def test_lookup_index_batch_revalidates_all_semantic_metadata(
    hazard_registry: HazardRegistry,
) -> None:
    layer = _layer(
        _learned_config(),
        hazard_registry=hazard_registry,
    )
    valid = layer.vocabulary.encode_names(
        (HazardKind.FLOOD,),
        unknown_policy=UNKNOWN_ID_POLICY_ERROR,
    )

    forged = HazardIndexBatch(
        dense_indices=valid.dense_indices,
        stable_hazard_ids=valid.stable_hazard_ids,
        hazard_names=(HazardKind.HEAT.value,),
        unknown_mask=valid.unknown_mask,
        vocabulary_fingerprint=valid.vocabulary_fingerprint,
    )

    with pytest.raises(ValueError, match="hazard name"):
        layer.lookup_index_batch(forged)


def test_lookup_rejects_batch_from_different_vocabulary(
    hazard_registry: HazardRegistry,
) -> None:
    layer = _layer(
        _learned_config(),
        hazard_registry=hazard_registry,
    )
    different_vocabulary = build_hazard_embedding_vocabulary(
        hazard_registry=hazard_registry,
        include_fallback_hazard=True,
    )
    batch = different_vocabulary.encode_names(
        (HazardKind.FLOOD,),
        unknown_policy=UNKNOWN_ID_POLICY_ERROR,
    )

    with pytest.raises(ValueError, match="different vocabulary"):
        layer.lookup_index_batch(batch)


def test_lookup_rejects_nonfinite_effective_embeddings(
    hazard_registry: HazardRegistry,
) -> None:
    layer = _layer(
        _learned_config(),
        hazard_registry=hazard_registry,
    )
    assert layer.learned_embeddings is not None

    with torch.no_grad():
        layer.learned_embeddings.weight[0, 0] = float("inf")

    with pytest.raises(ValueError, match="finite|infinity"):
        layer.lookup_names((HazardKind.FLOOD,))


# =============================================================================
# Packed graph to node alignment
# =============================================================================


def test_raw_graph_broadcast(
) -> None:
    graph_embeddings = torch.tensor(
        [
            [1.0, 2.0],
            [3.0, 4.0],
            [5.0, 6.0],
        ]
    )
    node_batch_index = torch.tensor(
        [0, 0, 1, 2, 2],
        dtype=torch.long,
    )

    node_embeddings = broadcast_graph_embeddings_to_nodes(
        graph_embeddings,
        node_batch_index,
    )

    assert torch.equal(
        node_embeddings,
        graph_embeddings[node_batch_index],
    )


@pytest.mark.parametrize(
    "node_batch_index",
    (
        torch.tensor([0, 2], dtype=torch.long),
        torch.tensor([-1, 0], dtype=torch.long),
        torch.tensor([0, 3], dtype=torch.long),
    ),
)
def test_raw_graph_broadcast_rejects_invalid_membership(
    node_batch_index: torch.Tensor,
) -> None:
    graph_embeddings = torch.zeros(3, 2)

    with pytest.raises((ValueError, IndexError)):
        broadcast_graph_embeddings_to_nodes(
            graph_embeddings,
            node_batch_index,
        )


def test_graph_lookup_preserves_graph_identity_metadata(
    hazard_registry: HazardRegistry,
) -> None:
    layer = _layer(
        _learned_config(),
        hazard_registry=hazard_registry,
    )
    node_batch_index = torch.tensor(
        [0, 0, 1, 1, 1],
        dtype=torch.long,
    )

    result = layer.lookup_graph_hazards_for_nodes(
        (
            HazardKind.FLOOD,
            HazardKind.HEAT,
        ),
        node_batch_index,
    )

    assert isinstance(result, NodeAlignedHazardEmbeddingLookup)
    assert result.node_embeddings.shape == (
        len(node_batch_index),
        EMBEDDING_DIM,
    )
    assert result.graph_lookup.indices.hazard_names == (
        HazardKind.FLOOD.value,
        HazardKind.HEAT.value,
    )
    assert torch.equal(
        result.node_batch_index,
        node_batch_index,
    )
    assert torch.equal(
        result.node_embeddings,
        result.graph_lookup.embeddings[node_batch_index],
    )


# =============================================================================
# Fingerprints and fixed-artifact export
# =============================================================================


def test_architecture_and_parameter_fingerprints_are_separate(
    hazard_registry: HazardRegistry,
) -> None:
    first = _layer(
        _learned_config(initialization_seed=11),
        hazard_registry=hazard_registry,
    )
    second = _layer(
        _learned_config(initialization_seed=11),
        hazard_registry=hazard_registry,
    )

    assert first.architecture_fingerprint() == (
        second.architecture_fingerprint()
    )
    assert first.parameter_fingerprint() == (
        second.parameter_fingerprint()
    )

    assert second.learned_embeddings is not None
    with torch.no_grad():
        second.learned_embeddings.weight[0, 0] += 1.0

    assert first.architecture_fingerprint() == (
        second.architecture_fingerprint()
    )
    assert first.parameter_fingerprint() != (
        second.parameter_fingerprint()
    )


def test_exported_fixed_artifact_matches_effective_table(
    hazard_registry: HazardRegistry,
) -> None:
    layer = _layer(
        _learned_config(),
        hazard_registry=hazard_registry,
    )

    artifact = layer.export_as_fixed_artifact(
        source_id="trained-hazard-embedding",
        source_version="checkpoint-7",
    )

    effective = (
        layer.full_embedding_table()
        .embeddings
        .detach()
        .cpu()
    )

    assert torch.equal(
        artifact.embedding_matrix,
        effective,
    )
    assert artifact.hazard_names == layer.vocabulary.hazard_names
    assert artifact.stable_hazard_ids == (
        layer.vocabulary.stable_hazard_ids
    )
    assert artifact.vocabulary_fingerprint == (
        layer.vocabulary.fingerprint()
    )
    assert artifact.embedding_fingerprint == (
        layer.effective_table_fingerprint()
    )


def test_export_identity_changes_when_effective_scaling_changes(
    hazard_registry: HazardRegistry,
    runtime_vocabulary: HazardEmbeddingVocabulary,
) -> None:
    artifact = _artifact(runtime_vocabulary)

    first = _layer(
        _fixed_config(artifact, fixed_scale=1.0),
        hazard_registry=hazard_registry,
        fixed_artifact=artifact,
    )
    second = _layer(
        _fixed_config(artifact, fixed_scale=0.5),
        hazard_registry=hazard_registry,
        fixed_artifact=artifact,
    )

    first_export = first.export_as_fixed_artifact(
        source_id="scaled-export",
        source_version="1",
    )
    second_export = second.export_as_fixed_artifact(
        source_id="scaled-export",
        source_version="1",
    )

    assert first_export.embedding_fingerprint != (
        second_export.embedding_fingerprint
    )
    assert first_export.source_fingerprint != (
        second_export.source_fingerprint
    )


# =============================================================================
# Direct validation at the implementation boundary
# =============================================================================


def test_layer_rejects_negative_scales_even_without_config(
    runtime_vocabulary: HazardEmbeddingVocabulary,
) -> None:
    with pytest.raises(ValueError, match="fixed_scale"):
        HazardEmbeddingLayer(
            vocabulary=runtime_vocabulary,
            embedding_dim=EMBEDDING_DIM,
            mode=HAZARD_EMBEDDING_MODE_LEARNED,
            unknown_policy=UNKNOWN_ID_POLICY_ERROR,
            unknown_embedding_policy=UNKNOWN_EMBEDDING_POLICY_ERROR,
            initialization=HAZARD_EMBEDDING_INIT_NORMAL,
            initialization_seed=1,
            initialization_std=0.05,
            fixed_scale=-1.0,
        )

    with pytest.raises(ValueError, match="residual_scale"):
        HazardEmbeddingLayer(
            vocabulary=runtime_vocabulary,
            embedding_dim=EMBEDDING_DIM,
            mode=HAZARD_EMBEDDING_MODE_LEARNED,
            unknown_policy=UNKNOWN_ID_POLICY_ERROR,
            unknown_embedding_policy=UNKNOWN_EMBEDDING_POLICY_ERROR,
            initialization=HAZARD_EMBEDDING_INIT_NORMAL,
            initialization_seed=1,
            initialization_std=0.05,
            residual_scale=-1.0,
        )


def test_learned_mode_rejects_fixed_artifact(
    runtime_vocabulary: HazardEmbeddingVocabulary,
) -> None:
    artifact = _artifact(runtime_vocabulary)

    with pytest.raises(ValueError, match="Learned|fixed artifact"):
        HazardEmbeddingLayer(
            vocabulary=runtime_vocabulary,
            embedding_dim=EMBEDDING_DIM,
            mode=HAZARD_EMBEDDING_MODE_LEARNED,
            unknown_policy=UNKNOWN_ID_POLICY_ERROR,
            unknown_embedding_policy=UNKNOWN_EMBEDDING_POLICY_ERROR,
            initialization=HAZARD_EMBEDDING_INIT_NORMAL,
            initialization_seed=1,
            initialization_std=0.05,
            fixed_artifact=artifact,
        )


def test_fixed_modes_require_artifact(
    runtime_vocabulary: HazardEmbeddingVocabulary,
) -> None:
    for mode in (
        HAZARD_EMBEDDING_MODE_FIXED,
        HAZARD_EMBEDDING_MODE_FIXED_PLUS_RESIDUAL,
    ):
        with pytest.raises(ValueError, match="requires.*artifact"):
            HazardEmbeddingLayer(
                vocabulary=runtime_vocabulary,
                embedding_dim=EMBEDDING_DIM,
                mode=mode,
                unknown_policy=UNKNOWN_ID_POLICY_ERROR,
                unknown_embedding_policy=UNKNOWN_EMBEDDING_POLICY_ERROR,
                initialization=HAZARD_EMBEDDING_INIT_ZERO,
                initialization_seed=1,
                initialization_std=0.0,
            )
