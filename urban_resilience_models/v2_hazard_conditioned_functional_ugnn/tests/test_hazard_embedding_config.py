"""Focused tests for nested hazard-embedding configuration."""

import pytest

from urban_resilience_models.v2_hazard_conditioned_functional_ugnn.config import (
    ExperimentConfig,
    HAZARD_CONDITIONING_EMBEDDING,
    HAZARD_EMBEDDING_MODE_FIXED,
    HAZARD_EMBEDDING_MODE_FIXED_PLUS_RESIDUAL,
    HazardConfig,
    HazardEmbeddingConfig,
    UNKNOWN_EMBEDDING_POLICY_LEARNED,
    UNKNOWN_EMBEDDING_POLICY_ZERO_FIXED,
    UNKNOWN_ID_POLICY_USE_UNKNOWN_EMBEDDING,
)


def test_nested_embedding_round_trip_and_hash() -> None:
    config = ExperimentConfig.minimal_skeleton()
    payload = config.to_dict()

    assert payload["model"]["hazard"]["embedding"]["embedding_dim"] == 32
    assert ExperimentConfig.from_dict(payload) == config

    changed_embedding = config.model.hazard.embedding.replace(
        initialization_seed=43
    )
    changed = config.replace(
        model=config.model.replace(
            hazard=config.model.hazard.replace(
                embedding=changed_embedding
            )
        )
    )
    assert changed.scientific_config_hash() != config.scientific_config_hash()


def test_legacy_flat_hazard_embedding_shape_migrates() -> None:
    payload = ExperimentConfig.minimal_skeleton().to_dict()
    hazard_payload = payload["model"]["hazard"]
    hazard_payload.pop("embedding")
    hazard_payload["embedding_dim"] = 48
    hazard_payload["unknown_hazard_policy"] = (
        UNKNOWN_ID_POLICY_USE_UNKNOWN_EMBEDDING
    )

    migrated = ExperimentConfig.from_dict(payload)
    embedding = migrated.model.hazard.embedding

    assert embedding.embedding_dim == 48
    assert embedding.unknown_input_policy == (
        UNKNOWN_ID_POLICY_USE_UNKNOWN_EMBEDDING
    )
    assert embedding.unknown_embedding_policy == (
        UNKNOWN_EMBEDDING_POLICY_LEARNED
    )
    assert embedding.include_unknown_row


def test_legacy_and_nested_shapes_cannot_be_mixed() -> None:
    payload = ExperimentConfig.minimal_skeleton().to_dict()
    payload["model"]["hazard"]["embedding_dim"] = 32

    with pytest.raises(ValueError, match="cannot mix"):
        ExperimentConfig.from_dict(payload)


def test_unknown_input_and_unknown_row_contract() -> None:
    with pytest.raises(TypeError, match="Boolean"):
        HazardEmbeddingConfig(include_unknown_row=1)  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="include_unknown_row=False"):
        HazardEmbeddingConfig(include_unknown_row=True)

    with pytest.raises(ValueError, match="include_unknown_row=True"):
        HazardEmbeddingConfig(
            unknown_input_policy=UNKNOWN_ID_POLICY_USE_UNKNOWN_EMBEDDING,
            unknown_embedding_policy=UNKNOWN_EMBEDDING_POLICY_ZERO_FIXED,
        )

    zero_fixed = HazardEmbeddingConfig(
        unknown_input_policy=UNKNOWN_ID_POLICY_USE_UNKNOWN_EMBEDDING,
        unknown_embedding_policy=UNKNOWN_EMBEDDING_POLICY_ZERO_FIXED,
        include_unknown_row=True,
    )
    learned = zero_fixed.replace(
        unknown_embedding_policy=UNKNOWN_EMBEDDING_POLICY_LEARNED
    )
    assert zero_fixed.include_unknown_row
    assert learned.include_unknown_row


def test_fixed_artifact_validation_and_readiness() -> None:
    with pytest.raises(ValueError, match="fixed_artifact_reference"):
        HazardEmbeddingConfig(mode=HAZARD_EMBEDDING_MODE_FIXED)

    fixed = HazardEmbeddingConfig(
        mode=HAZARD_EMBEDDING_MODE_FIXED,
        fixed_artifact_reference="artifacts/hazards/fixed.pt",
        initialization_std=0.0,
    )
    with pytest.raises(ValueError, match="fixed_artifact_fingerprint"):
        fixed.assert_construction_ready()

    ready = fixed.replace(fixed_artifact_fingerprint="sha256:abc123")
    ready.assert_construction_ready()

    ready.replace(
        mode=HAZARD_EMBEDDING_MODE_FIXED_PLUS_RESIDUAL,
        initialization_std=0.02,
    ).assert_construction_ready()


def test_disabled_hazard_skips_embedding_readiness() -> None:
    fixed = HazardEmbeddingConfig(
        mode=HAZARD_EMBEDDING_MODE_FIXED,
        fixed_artifact_reference="artifact-key",
        initialization_std=0.0,
    )
    HazardConfig(embedding=fixed).assert_construction_ready()

    enabled = HazardConfig(
        conditioning_mode=HAZARD_CONDITIONING_EMBEDDING,
        embedding=fixed,
    )
    with pytest.raises(ValueError, match="fixed_artifact_fingerprint"):
        enabled.assert_construction_ready()


def test_artifact_reference_is_not_scientific_identity() -> None:
    base = ExperimentConfig.minimal_skeleton()
    first_embedding = HazardEmbeddingConfig(
        mode=HAZARD_EMBEDDING_MODE_FIXED,
        fixed_artifact_reference="/mnt/lab-a/fixed.pt",
        fixed_artifact_fingerprint="sha256:same-content",
        initialization_std=0.0,
    )
    second_embedding = first_embedding.replace(
        fixed_artifact_reference="/home/tim/fixed.pt"
    )
    changed_content = first_embedding.replace(
        fixed_artifact_fingerprint="sha256:different-content"
    )

    def with_embedding(embedding: HazardEmbeddingConfig) -> ExperimentConfig:
        return base.replace(
            model=base.model.replace(
                hazard=base.model.hazard.replace(embedding=embedding)
            )
        )

    first = with_embedding(first_embedding)
    second = with_embedding(second_embedding)
    third = with_embedding(changed_content)

    assert first.full_config_hash() != second.full_config_hash()
    assert first.scientific_config_hash() == second.scientific_config_hash()
    assert first.scientific_config_hash() != third.scientific_config_hash()


def test_north_star_preset_support_policy() -> None:
    embedding = (
        ExperimentConfig.v2_0_north_star_target().model.hazard.embedding
    )
    assert embedding.require_training_supported_hazards
    assert embedding.allow_partially_data_backed_for_training
    assert embedding.require_queryable_hazards_for_inference
    assert not embedding.allow_fallback_hazard_for_inference
    assert embedding.allow_planned_hazard_counterfactuals
