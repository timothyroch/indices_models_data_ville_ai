"""Contract tests for the metadata-preserving V2 lag-memory encoder."""

from __future__ import annotations

from types import MappingProxyType

import pytest
import torch

from urban_resilience_models.v2_hazard_conditioned_functional_ugnn.config import (
    MemoryConfig,
)
from urban_resilience_models.v2_hazard_conditioned_functional_ugnn.constants import (
    MEMORY_ENCODER_LAG,
    MEMORY_ENCODER_NONE,
    MEMORY_QUERY_NONE,
)
from urban_resilience_models.v2_hazard_conditioned_functional_ugnn.memory.lag_memory_encoder import (
    LAG_MEMORY_BATCH_SCHEMA_VERSION,
    LAG_MEMORY_ENCODER_SCHEMA_VERSION,
    LAG_MEMORY_ENCODING_SCHEMA_VERSION,
    LagMemoryBatch,
    LagMemoryEncoder,
    LagMemoryEncoding,
)


LAG_FEATURE_NAMES = (
    "target_lag_1",
    "target_lag_3",
    "reporting_lag_1",
)
HIDDEN_DIM = 8


@pytest.fixture()
def lag_values() -> torch.Tensor:
    return torch.tensor(
        [
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
        ],
        dtype=torch.float32,
    )


@pytest.fixture()
def feature_mask() -> torch.Tensor:
    return torch.tensor(
        [
            [True, False, True],
            [True, True, False],
        ],
        dtype=torch.bool,
    )


@pytest.fixture()
def batch(
    lag_values: torch.Tensor,
    feature_mask: torch.Tensor,
) -> LagMemoryBatch:
    return LagMemoryBatch(
        lag_values=lag_values,
        lag_feature_names=LAG_FEATURE_NAMES,
        feature_mask=feature_mask,
        source_history_length=4,
        history_mask=torch.tensor(
            [
                [True, True, True, True],
                [False, True, True, True],
            ],
            dtype=torch.bool,
        ),
        history_time_points=torch.tensor(
            [1, 2, 3, 4],
            dtype=torch.long,
        ),
        source_fingerprint="lag-source-v1",
    )


@pytest.fixture()
def encoder() -> LagMemoryEncoder:
    return LagMemoryEncoder(
        lag_feature_names=LAG_FEATURE_NAMES,
        hidden_dim=HIDDEN_DIM,
        dropout=0.0,
        return_lag_states=True,
        return_lag_weights=True,
    )


# =============================================================================
# Schema and constructor contracts
# =============================================================================


def test_schema_versions_are_nonempty() -> None:
    assert LAG_MEMORY_BATCH_SCHEMA_VERSION
    assert LAG_MEMORY_ENCODER_SCHEMA_VERSION
    assert LAG_MEMORY_ENCODING_SCHEMA_VERSION


@pytest.mark.parametrize(
    "names",
    (
        (),
        ("lag_1", "lag_1"),
        ("lag_1", ""),
    ),
)
def test_encoder_rejects_invalid_lag_feature_names(
    names: tuple[str, ...],
) -> None:
    with pytest.raises(ValueError):
        LagMemoryEncoder(
            lag_feature_names=names,
            hidden_dim=HIDDEN_DIM,
        )


@pytest.mark.parametrize("hidden_dim", (0, -1, True))
def test_encoder_rejects_invalid_hidden_dim(
    hidden_dim: int,
) -> None:
    with pytest.raises(ValueError, match="hidden_dim"):
        LagMemoryEncoder(
            lag_feature_names=LAG_FEATURE_NAMES,
            hidden_dim=hidden_dim,
        )


@pytest.mark.parametrize("dropout", (-0.1, 1.0, 1.5))
def test_encoder_rejects_invalid_dropout(
    dropout: float,
) -> None:
    with pytest.raises((TypeError, ValueError), match="dropout"):
        LagMemoryEncoder(
            lag_feature_names=LAG_FEATURE_NAMES,
            hidden_dim=HIDDEN_DIM,
            dropout=dropout,
        )


def test_encoder_rejects_non_boolean_export_flags() -> None:
    with pytest.raises(TypeError, match="return_lag_states"):
        LagMemoryEncoder(
            lag_feature_names=LAG_FEATURE_NAMES,
            hidden_dim=HIDDEN_DIM,
            return_lag_states=1,  # type: ignore[arg-type]
        )

    with pytest.raises(TypeError, match="return_lag_weights"):
        LagMemoryEncoder(
            lag_feature_names=LAG_FEATURE_NAMES,
            hidden_dim=HIDDEN_DIM,
            return_lag_weights=1,  # type: ignore[arg-type]
        )


def test_encoder_architecture_fields_are_resolved(
    encoder: LagMemoryEncoder,
) -> None:
    assert encoder.lag_feature_names == LAG_FEATURE_NAMES
    assert encoder.num_lag_features == len(LAG_FEATURE_NAMES)
    assert encoder.hidden_dim == HIDDEN_DIM
    assert encoder.return_lag_states
    assert encoder.return_lag_weights
    assert encoder.device.type == "cpu"
    assert encoder.dtype == torch.float32


# =============================================================================
# LagMemoryBatch validation and metadata
# =============================================================================


def test_batch_rejects_bare_non_tensor_values() -> None:
    with pytest.raises(TypeError, match="lag_values"):
        LagMemoryBatch(
            lag_values=[[1.0]],  # type: ignore[arg-type]
            lag_feature_names=("lag_1",),
        )


def test_batch_requires_two_dimensional_floating_values() -> None:
    with pytest.raises(ValueError, match="shape"):
        LagMemoryBatch(
            lag_values=torch.tensor([1.0]),
            lag_feature_names=("lag_1",),
        )

    with pytest.raises(ValueError, match="floating"):
        LagMemoryBatch(
            lag_values=torch.tensor([[1]], dtype=torch.long),
            lag_feature_names=("lag_1",),
        )


def test_batch_rejects_nonfinite_values() -> None:
    with pytest.raises(ValueError, match="finite"):
        LagMemoryBatch(
            lag_values=torch.tensor([[float("nan")]]),
            lag_feature_names=("lag_1",),
        )

    with pytest.raises(ValueError, match="finite"):
        LagMemoryBatch(
            lag_values=torch.tensor([[float("inf")]]),
            lag_feature_names=("lag_1",),
        )


def test_batch_names_must_match_value_columns(
    lag_values: torch.Tensor,
) -> None:
    with pytest.raises(ValueError, match="align"):
        LagMemoryBatch(
            lag_values=lag_values,
            lag_feature_names=("lag_1", "lag_2"),
        )

    with pytest.raises(ValueError, match="duplicates"):
        LagMemoryBatch(
            lag_values=lag_values,
            lag_feature_names=("lag_1", "lag_1", "lag_3"),
        )


def test_feature_mask_contract(
    lag_values: torch.Tensor,
) -> None:
    with pytest.raises(ValueError, match="torch.bool"):
        LagMemoryBatch(
            lag_values=lag_values,
            lag_feature_names=LAG_FEATURE_NAMES,
            feature_mask=torch.ones_like(lag_values),
        )

    with pytest.raises(ValueError, match="same shape"):
        LagMemoryBatch(
            lag_values=lag_values,
            lag_feature_names=LAG_FEATURE_NAMES,
            feature_mask=torch.ones(
                2,
                2,
                dtype=torch.bool,
            ),
        )

    with pytest.raises(ValueError, match="at least one"):
        LagMemoryBatch(
            lag_values=lag_values,
            lag_feature_names=LAG_FEATURE_NAMES,
            feature_mask=torch.tensor(
                [
                    [True, False, False],
                    [False, False, False],
                ],
                dtype=torch.bool,
            ),
        )


def test_effective_feature_mask_defaults_to_all_available(
    lag_values: torch.Tensor,
) -> None:
    value = LagMemoryBatch(
        lag_values=lag_values,
        lag_feature_names=LAG_FEATURE_NAMES,
    )

    assert value.feature_mask is None
    assert torch.equal(
        value.effective_feature_mask,
        torch.ones_like(
            lag_values,
            dtype=torch.bool,
        ),
    )


def test_source_history_length_must_be_positive(
    lag_values: torch.Tensor,
) -> None:
    with pytest.raises(ValueError, match="source_history_length"):
        LagMemoryBatch(
            lag_values=lag_values,
            lag_feature_names=LAG_FEATURE_NAMES,
            source_history_length=0,
        )


def test_history_mask_contract(
    lag_values: torch.Tensor,
) -> None:
    with pytest.raises(ValueError, match="shape"):
        LagMemoryBatch(
            lag_values=lag_values,
            lag_feature_names=LAG_FEATURE_NAMES,
            history_mask=torch.ones(4, dtype=torch.bool),
        )

    with pytest.raises(ValueError, match="torch.bool"):
        LagMemoryBatch(
            lag_values=lag_values,
            lag_feature_names=LAG_FEATURE_NAMES,
            history_mask=torch.ones(2, 4),
        )

    with pytest.raises(ValueError, match="rows"):
        LagMemoryBatch(
            lag_values=lag_values,
            lag_feature_names=LAG_FEATURE_NAMES,
            history_mask=torch.ones(
                3,
                4,
                dtype=torch.bool,
            ),
        )


def test_history_time_points_contract(
    lag_values: torch.Tensor,
) -> None:
    with pytest.raises(ValueError, match="shape"):
        LagMemoryBatch(
            lag_values=lag_values,
            lag_feature_names=LAG_FEATURE_NAMES,
            history_time_points=torch.ones(2, 3, 1),
        )

    with pytest.raises(ValueError, match="numeric"):
        LagMemoryBatch(
            lag_values=lag_values,
            lag_feature_names=LAG_FEATURE_NAMES,
            history_time_points=torch.tensor(
                [True, False],
                dtype=torch.bool,
            ),
        )

    with pytest.raises(ValueError, match="finite"):
        LagMemoryBatch(
            lag_values=lag_values,
            lag_feature_names=LAG_FEATURE_NAMES,
            history_time_points=torch.tensor(
                [1.0, float("nan")],
            ),
        )

    with pytest.raises(ValueError, match="rows"):
        LagMemoryBatch(
            lag_values=lag_values,
            lag_feature_names=LAG_FEATURE_NAMES,
            history_time_points=torch.ones(3, 4),
        )


def test_history_lengths_must_agree(
    lag_values: torch.Tensor,
) -> None:
    with pytest.raises(ValueError, match="same history length"):
        LagMemoryBatch(
            lag_values=lag_values,
            lag_feature_names=LAG_FEATURE_NAMES,
            history_mask=torch.ones(
                2,
                4,
                dtype=torch.bool,
            ),
            history_time_points=torch.arange(3),
        )

    with pytest.raises(ValueError, match="source_history_length"):
        LagMemoryBatch(
            lag_values=lag_values,
            lag_feature_names=LAG_FEATURE_NAMES,
            source_history_length=5,
            history_mask=torch.ones(
                2,
                4,
                dtype=torch.bool,
            ),
        )


def test_history_time_points_must_be_nondecreasing(
    lag_values: torch.Tensor,
) -> None:
    with pytest.raises(ValueError, match="nondecreasing"):
        LagMemoryBatch(
            lag_values=lag_values,
            lag_feature_names=LAG_FEATURE_NAMES,
            history_time_points=torch.tensor([1, 3, 2, 4]),
        )

    with pytest.raises(ValueError, match="nondecreasing"):
        LagMemoryBatch(
            lag_values=lag_values,
            lag_feature_names=LAG_FEATURE_NAMES,
            history_mask=torch.ones(
                2,
                4,
                dtype=torch.bool,
            ),
            history_time_points=torch.tensor(
                [
                    [1, 2, 3, 4],
                    [1, 3, 2, 4],
                ],
            ),
        )


def test_masked_invalid_history_pairs_do_not_create_false_order_errors(
    lag_values: torch.Tensor,
) -> None:
    value = LagMemoryBatch(
        lag_values=lag_values,
        lag_feature_names=LAG_FEATURE_NAMES,
        history_mask=torch.tensor(
            [
                [True, False, True, True],
                [True, True, True, True],
            ],
            dtype=torch.bool,
        ),
        history_time_points=torch.tensor(
            [
                [3, 1, 2, 4],
                [1, 2, 3, 4],
            ],
        ),
    )

    assert value.resolved_history_length == 4


def test_shared_history_time_points_expand_to_items(
    batch: LagMemoryBatch,
) -> None:
    expanded = batch.expanded_history_time_points()

    assert expanded is not None
    assert expanded.shape == (2, 4)
    assert torch.equal(
        expanded[0],
        torch.tensor([1, 2, 3, 4]),
    )
    assert torch.equal(expanded[0], expanded[1])


def test_batch_metadata_and_value_fingerprints_are_separate(
    batch: LagMemoryBatch,
) -> None:
    same_values_new_source = LagMemoryBatch(
        lag_values=batch.lag_values.clone(),
        lag_feature_names=batch.lag_feature_names,
        feature_mask=batch.feature_mask.clone(),
        source_history_length=batch.source_history_length,
        history_mask=batch.history_mask.clone(),
        history_time_points=batch.history_time_points.clone(),
        source_fingerprint="lag-source-v2",
    )
    changed_values = LagMemoryBatch(
        lag_values=batch.lag_values + 1.0,
        lag_feature_names=batch.lag_feature_names,
        feature_mask=batch.feature_mask,
        source_history_length=batch.source_history_length,
        history_mask=batch.history_mask,
        history_time_points=batch.history_time_points,
        source_fingerprint=batch.source_fingerprint,
    )

    assert batch.metadata_fingerprint() != (
        same_values_new_source.metadata_fingerprint()
    )
    assert batch.value_fingerprint() == (
        same_values_new_source.value_fingerprint()
    )
    assert batch.value_fingerprint() != (
        changed_values.value_fingerprint()
    )


def test_batch_to_preserves_metadata(
    batch: LagMemoryBatch,
) -> None:
    moved = batch.to("cpu")

    assert moved is not batch
    assert moved.lag_feature_names == batch.lag_feature_names
    assert moved.source_history_length == batch.source_history_length
    assert moved.source_fingerprint == batch.source_fingerprint
    assert moved.schema_version == batch.schema_version
    assert torch.equal(moved.lag_values, batch.lag_values)
    assert torch.equal(moved.feature_mask, batch.feature_mask)
    assert torch.equal(moved.history_mask, batch.history_mask)
    assert torch.equal(
        moved.history_time_points,
        batch.history_time_points,
    )


# =============================================================================
# Configuration construction
# =============================================================================


def test_from_config_constructs_lag_encoder() -> None:
    config = MemoryConfig(
        encoder_type=MEMORY_ENCODER_LAG,
        query_type=MEMORY_QUERY_NONE,
        input_dim=len(LAG_FEATURE_NAMES),
        hidden_dim=HIDDEN_DIM,
        dropout=0.1,
        return_temporal_states=True,
        return_temporal_attention=True,
        lag_feature_names=LAG_FEATURE_NAMES,
    )

    encoder = LagMemoryEncoder.from_config(config)

    assert encoder.lag_feature_names == LAG_FEATURE_NAMES
    assert encoder.hidden_dim == HIDDEN_DIM
    assert encoder.dropout == pytest.approx(0.1)
    assert encoder.return_lag_states
    assert encoder.return_lag_weights


def test_from_config_accepts_unresolved_lag_input_dim() -> None:
    config = MemoryConfig(
        encoder_type=MEMORY_ENCODER_LAG,
        query_type=MEMORY_QUERY_NONE,
        input_dim=None,
        hidden_dim=HIDDEN_DIM,
        lag_feature_names=LAG_FEATURE_NAMES,
    )

    encoder = LagMemoryEncoder.from_config(config)
    assert encoder.num_lag_features == len(LAG_FEATURE_NAMES)


def test_from_config_rejects_wrong_object() -> None:
    with pytest.raises(TypeError, match="MemoryConfig"):
        LagMemoryEncoder.from_config(object())  # type: ignore[arg-type]


def test_from_config_rejects_non_lag_encoder() -> None:
    config = MemoryConfig(
        encoder_type=MEMORY_ENCODER_NONE,
        query_type=MEMORY_QUERY_NONE,
    )

    with pytest.raises(ValueError, match="encoder_type='lag'"):
        LagMemoryEncoder.from_config(config)


def test_from_config_rejects_input_dim_mismatch() -> None:
    config = MemoryConfig(
        encoder_type=MEMORY_ENCODER_LAG,
        query_type=MEMORY_QUERY_NONE,
        input_dim=2,
        hidden_dim=HIDDEN_DIM,
        lag_feature_names=LAG_FEATURE_NAMES,
    )

    with pytest.raises(ValueError, match="input_dim"):
        LagMemoryEncoder.from_config(config)


# =============================================================================
# Forward encoding and metadata preservation
# =============================================================================


def test_forward_rejects_bare_tensor(
    encoder: LagMemoryEncoder,
    lag_values: torch.Tensor,
) -> None:
    with pytest.raises(TypeError, match="bare tensors"):
        encoder(lag_values)  # type: ignore[arg-type]


def test_forward_requires_exact_feature_names_and_order(
    encoder: LagMemoryEncoder,
    lag_values: torch.Tensor,
) -> None:
    wrong_order = LagMemoryBatch(
        lag_values=lag_values,
        lag_feature_names=tuple(reversed(LAG_FEATURE_NAMES)),
    )

    with pytest.raises(ValueError, match="names or ordering"):
        encoder(wrong_order)


def test_forward_returns_metadata_preserving_encoding(
    encoder: LagMemoryEncoder,
    batch: LagMemoryBatch,
) -> None:
    result = encoder(batch)

    assert isinstance(result, LagMemoryEncoding)
    assert result.source_batch is batch
    assert result.memory_state.shape == (2, HIDDEN_DIM)
    assert result.lag_feature_states is not None
    assert result.lag_feature_states.shape == (
        2,
        len(LAG_FEATURE_NAMES),
        HIDDEN_DIM,
    )
    assert result.lag_weights is not None
    assert result.lag_weights.shape == (
        2,
        len(LAG_FEATURE_NAMES),
    )
    assert result.lag_feature_names == LAG_FEATURE_NAMES
    assert result.hidden_dim == HIDDEN_DIM
    assert result.item_count == 2
    assert result.encoder_architecture_fingerprint
    assert result.lineage_fingerprint
    assert bool(torch.isfinite(result.memory_state).all().item())


def test_masked_mean_weights_are_exact(
    encoder: LagMemoryEncoder,
    batch: LagMemoryBatch,
) -> None:
    result = encoder(batch)
    assert result.lag_weights is not None

    expected = torch.tensor(
        [
            [0.5, 0.0, 0.5],
            [0.5, 0.5, 0.0],
        ],
        dtype=result.lag_weights.dtype,
    )

    assert torch.equal(result.lag_weights, expected)


def test_masked_values_do_not_change_memory_state(
    encoder: LagMemoryEncoder,
    batch: LagMemoryBatch,
) -> None:
    encoder.eval()
    changed = batch.lag_values.clone()
    changed[0, 1] = 10_000.0
    changed[1, 2] = -10_000.0

    changed_batch = LagMemoryBatch(
        lag_values=changed,
        lag_feature_names=batch.lag_feature_names,
        feature_mask=batch.feature_mask,
        source_history_length=batch.source_history_length,
        history_mask=batch.history_mask,
        history_time_points=batch.history_time_points,
        source_fingerprint=batch.source_fingerprint,
    )

    first = encoder(batch).memory_state
    second = encoder(changed_batch).memory_state

    assert torch.equal(first, second)


def test_output_export_flags_control_optional_states(
    batch: LagMemoryBatch,
) -> None:
    encoder = LagMemoryEncoder(
        lag_feature_names=LAG_FEATURE_NAMES,
        hidden_dim=HIDDEN_DIM,
        return_lag_states=False,
        return_lag_weights=False,
    )

    result = encoder(batch)

    assert result.lag_feature_states is None
    assert result.lag_weights is None
    assert set(result.component_states) == {"memory_state"}


def test_component_states_mapping_is_read_only(
    encoder: LagMemoryEncoder,
    batch: LagMemoryBatch,
) -> None:
    result = encoder(batch)
    components = result.component_states

    assert isinstance(components, MappingProxyType)
    assert set(components) == {
        "memory_state",
        "lag_feature_states",
        "lag_weights",
    }

    with pytest.raises(TypeError):
        components["new"] = result.memory_state  # type: ignore[index]


def test_encoder_respects_module_dtype(
    encoder: LagMemoryEncoder,
    batch: LagMemoryBatch,
) -> None:
    encoder = encoder.double()
    result = encoder(batch)

    assert result.memory_state.dtype == torch.float64
    assert result.lag_feature_states.dtype == torch.float64
    assert result.lag_weights.dtype == torch.float64


def test_empty_batch_is_supported(
    encoder: LagMemoryEncoder,
) -> None:
    empty = LagMemoryBatch(
        lag_values=torch.empty(
            0,
            len(LAG_FEATURE_NAMES),
        ),
        lag_feature_names=LAG_FEATURE_NAMES,
    )

    result = encoder(empty)

    assert result.memory_state.shape == (0, HIDDEN_DIM)
    assert result.lag_feature_states.shape == (
        0,
        len(LAG_FEATURE_NAMES),
        HIDDEN_DIM,
    )
    assert result.lag_weights.shape == (
        0,
        len(LAG_FEATURE_NAMES),
    )


def test_backward_pass_is_finite_and_masked_values_get_zero_gradient(
    feature_mask: torch.Tensor,
) -> None:
    values = torch.tensor(
        [
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
        ],
        requires_grad=True,
    )
    batch = LagMemoryBatch(
        lag_values=values,
        lag_feature_names=LAG_FEATURE_NAMES,
        feature_mask=feature_mask,
    )
    encoder = LagMemoryEncoder(
        lag_feature_names=LAG_FEATURE_NAMES,
        hidden_dim=HIDDEN_DIM,
    )

    loss = encoder(batch).memory_state.square().sum()
    loss.backward()

    assert values.grad is not None
    assert bool(torch.isfinite(values.grad).all().item())
    assert torch.equal(
        values.grad.masked_select(~feature_mask),
        torch.zeros_like(
            values.grad.masked_select(~feature_mask)
        ),
    )

    parameter_gradients = [
        parameter.grad
        for parameter in encoder.parameters()
        if parameter.grad is not None
    ]
    assert parameter_gradients
    assert all(
        bool(torch.isfinite(gradient).all().item())
        for gradient in parameter_gradients
    )


# =============================================================================
# Fingerprints and diagnostics
# =============================================================================


def test_architecture_fingerprint_is_parameter_independent(
    batch: LagMemoryBatch,
) -> None:
    torch.manual_seed(7)
    first = LagMemoryEncoder(
        lag_feature_names=LAG_FEATURE_NAMES,
        hidden_dim=HIDDEN_DIM,
    )
    torch.manual_seed(7)
    second = LagMemoryEncoder(
        lag_feature_names=LAG_FEATURE_NAMES,
        hidden_dim=HIDDEN_DIM,
    )

    assert first.architecture_fingerprint() == (
        second.architecture_fingerprint()
    )
    assert first.parameter_fingerprint() == (
        second.parameter_fingerprint()
    )

    with torch.no_grad():
        second.value_projection.weight[0, 0] += 1.0

    assert first.architecture_fingerprint() == (
        second.architecture_fingerprint()
    )
    assert first.parameter_fingerprint() != (
        second.parameter_fingerprint()
    )

    assert first.lineage_fingerprint(batch) == (
        first.lineage_fingerprint(batch)
    )


def test_architecture_fingerprint_changes_with_contract() -> None:
    first = LagMemoryEncoder(
        lag_feature_names=LAG_FEATURE_NAMES,
        hidden_dim=HIDDEN_DIM,
    )
    second = LagMemoryEncoder(
        lag_feature_names=tuple(
            reversed(LAG_FEATURE_NAMES)
        ),
        hidden_dim=HIDDEN_DIM,
    )
    third = LagMemoryEncoder(
        lag_feature_names=LAG_FEATURE_NAMES,
        hidden_dim=HIDDEN_DIM + 1,
    )

    assert first.architecture_fingerprint() != (
        second.architecture_fingerprint()
    )
    assert first.architecture_fingerprint() != (
        third.architecture_fingerprint()
    )


def test_lineage_fingerprint_changes_with_values_and_source(
    encoder: LagMemoryEncoder,
    batch: LagMemoryBatch,
) -> None:
    changed_values = LagMemoryBatch(
        lag_values=batch.lag_values + 1.0,
        lag_feature_names=batch.lag_feature_names,
        feature_mask=batch.feature_mask,
        source_history_length=batch.source_history_length,
        history_mask=batch.history_mask,
        history_time_points=batch.history_time_points,
        source_fingerprint=batch.source_fingerprint,
    )
    changed_source = LagMemoryBatch(
        lag_values=batch.lag_values,
        lag_feature_names=batch.lag_feature_names,
        feature_mask=batch.feature_mask,
        source_history_length=batch.source_history_length,
        history_mask=batch.history_mask,
        history_time_points=batch.history_time_points,
        source_fingerprint="different-source",
    )

    assert encoder.lineage_fingerprint(batch) != (
        encoder.lineage_fingerprint(changed_values)
    )
    assert encoder.lineage_fingerprint(batch) != (
        encoder.lineage_fingerprint(changed_source)
    )


def test_assert_finite_parameters_detects_corruption(
    encoder: LagMemoryEncoder,
) -> None:
    encoder.assert_finite_parameters()

    with torch.no_grad():
        encoder.value_projection.weight[0, 0] = float("inf")

    with pytest.raises(ValueError, match="infinity"):
        encoder.assert_finite_parameters()


# =============================================================================
# LagMemoryEncoding validation
# =============================================================================


def test_encoding_rejects_misaligned_memory_rows(
    batch: LagMemoryBatch,
) -> None:
    with pytest.raises(ValueError, match="rows"):
        LagMemoryEncoding(
            memory_state=torch.zeros(3, HIDDEN_DIM),
            source_batch=batch,
            lag_feature_states=None,
            lag_weights=None,
            encoder_architecture_fingerprint="architecture",
            lineage_fingerprint="lineage",
        )


def test_encoding_rejects_invalid_lag_weights(
    batch: LagMemoryBatch,
) -> None:
    with pytest.raises(ValueError, match="sum to one"):
        LagMemoryEncoding(
            memory_state=torch.zeros(2, HIDDEN_DIM),
            source_batch=batch,
            lag_feature_states=None,
            lag_weights=torch.full(
                (2, len(LAG_FEATURE_NAMES)),
                0.2,
            ),
            encoder_architecture_fingerprint="architecture",
            lineage_fingerprint="lineage",
        )

    unavailable_nonzero = torch.tensor(
        [
            [0.4, 0.2, 0.4],
            [0.4, 0.4, 0.2],
        ],
    )
    with pytest.raises(ValueError, match="Unavailable"):
        LagMemoryEncoding(
            memory_state=torch.zeros(2, HIDDEN_DIM),
            source_batch=batch,
            lag_feature_states=None,
            lag_weights=unavailable_nonzero,
            encoder_architecture_fingerprint="architecture",
            lineage_fingerprint="lineage",
        )


@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA is unavailable.",
)
def test_encoder_rejects_batch_on_different_device(
    batch: LagMemoryBatch,
) -> None:
    encoder = LagMemoryEncoder(
        lag_feature_names=LAG_FEATURE_NAMES,
        hidden_dim=HIDDEN_DIM,
    ).cuda()

    with pytest.raises(ValueError, match="same device"):
        encoder(batch)
