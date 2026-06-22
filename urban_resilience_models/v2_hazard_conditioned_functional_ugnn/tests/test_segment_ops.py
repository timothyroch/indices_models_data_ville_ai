"""
Contract tests for generic grouped tensor operations.

Target repository path:
    urban_resilience_models/
        v2_hazard_conditioned_functional_ugnn/
            tests/
                test_segment_ops.py

Implementation under test:
    urban_resilience_models/
        v2_hazard_conditioned_functional_ugnn/
            functional_message_passing/
                segment_ops.py

This suite freezes the low-level grouped-operation contract independently from
attention, aggregation, relation semantics, graph packing, and model modules.

Covered behavior:

- dense zero-based segment-ID validation;
- counts and presence masks;
- sum, mean, and maximum over arbitrary trailing dimensions;
- explicit absent-segment semantics;
- numerically stable grouped softmax;
- normalization diagnostics;
- permutation and additive-shift invariance;
- empty inputs and zero-segment cases;
- exact dtype, device, shape, range, and finiteness checks;
- finite autograd and double-precision gradcheck;
- optional CUDA execution.
"""

from __future__ import annotations

from typing import Any

import pytest
import torch

from urban_resilience_models.v2_hazard_conditioned_functional_ugnn.functional_message_passing.segment_ops import (
    SEGMENT_OPS_SCHEMA_VERSION,
    SegmentMaxResult,
    assert_grouped_softmax_normalized,
    grouped_softmax,
    grouped_softmax_sums,
    segment_counts,
    segment_max,
    segment_mean,
    segment_presence,
    segment_sum,
    validate_segment_ids,
)


# =============================================================================
# Helpers
# =============================================================================


def _ids(
    *,
    device: torch.device | str = "cpu",
) -> torch.Tensor:
    return torch.tensor(
        [0, 0, 2, 2, 2, 4],
        dtype=torch.long,
        device=device,
    )


def _values_1d(
    *,
    dtype: torch.dtype = torch.float32,
    device: torch.device | str = "cpu",
    requires_grad: bool = False,
) -> torch.Tensor:
    values = torch.tensor(
        [1.0, 3.0, -2.0, 4.0, 6.0, 5.0],
        dtype=dtype,
        device=device,
    )
    values.requires_grad_(requires_grad)
    return values


def _values_2d(
    *,
    dtype: torch.dtype = torch.float32,
    device: torch.device | str = "cpu",
    requires_grad: bool = False,
) -> torch.Tensor:
    values = torch.tensor(
        [
            [1.0, 10.0],
            [3.0, 20.0],
            [-2.0, 30.0],
            [4.0, 40.0],
            [6.0, 50.0],
            [5.0, 60.0],
        ],
        dtype=dtype,
        device=device,
    )
    values.requires_grad_(requires_grad)
    return values


def _values_3d(
    *,
    dtype: torch.dtype = torch.float32,
    device: torch.device | str = "cpu",
    requires_grad: bool = False,
) -> torch.Tensor:
    values = torch.arange(
        6 * 2 * 3,
        dtype=dtype,
        device=device,
    ).reshape(6, 2, 3)
    values = values / 10.0
    values.requires_grad_(requires_grad)
    return values


def _reference_sum(
    values: torch.Tensor,
    segment_ids: torch.Tensor,
    num_segments: int,
) -> torch.Tensor:
    result = torch.zeros(
        (
            num_segments,
            *tuple(values.shape[1:]),
        ),
        dtype=values.dtype,
        device=values.device,
    )
    for item_index in range(
        int(values.shape[0])
    ):
        result[
            int(segment_ids[item_index].item())
        ] += values[item_index]
    return result


def _reference_mean(
    values: torch.Tensor,
    segment_ids: torch.Tensor,
    num_segments: int,
) -> torch.Tensor:
    sums = _reference_sum(
        values,
        segment_ids,
        num_segments,
    )
    counts = torch.bincount(
        segment_ids,
        minlength=num_segments,
    )
    shape = (
        num_segments,
        *([1] * (values.ndim - 1)),
    )
    return sums / (
        counts
        .clamp_min(1)
        .to(values.dtype)
        .view(shape)
    )


def _reference_max(
    values: torch.Tensor,
    segment_ids: torch.Tensor,
    num_segments: int,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    counts = torch.bincount(
        segment_ids,
        minlength=num_segments,
    )
    present = counts > 0
    result = torch.zeros(
        (
            num_segments,
            *tuple(values.shape[1:]),
        ),
        dtype=values.dtype,
        device=values.device,
    )

    for segment in range(num_segments):
        mask = segment_ids == segment
        if bool(mask.any().item()):
            result[segment] = values[mask].amax(
                dim=0
            )

    return result, present, counts


def _reference_grouped_softmax(
    logits: torch.Tensor,
    segment_ids: torch.Tensor,
    num_segments: int,
) -> torch.Tensor:
    result = torch.empty_like(logits)

    for segment in range(num_segments):
        mask = segment_ids == segment
        if bool(mask.any().item()):
            result[mask] = torch.softmax(
                logits[mask],
                dim=0,
            )

    return result


# =============================================================================
# Public identity
# =============================================================================


def test_schema_version_is_nonempty() -> None:
    assert isinstance(
        SEGMENT_OPS_SCHEMA_VERSION,
        str,
    )
    assert SEGMENT_OPS_SCHEMA_VERSION.strip()


# =============================================================================
# validate_segment_ids
# =============================================================================


def test_validate_segment_ids_accepts_dense_ids_with_empty_segments() -> None:
    validate_segment_ids(
        _ids(),
        num_segments=6,
    )


def test_validate_segment_ids_accepts_empty_ids_with_zero_segments() -> None:
    validate_segment_ids(
        torch.empty(
            0,
            dtype=torch.long,
        ),
        num_segments=0,
    )


def test_validate_segment_ids_accepts_empty_ids_with_declared_segments() -> None:
    validate_segment_ids(
        torch.empty(
            0,
            dtype=torch.long,
        ),
        num_segments=4,
    )


def test_validate_segment_ids_accepts_explicit_matching_item_count() -> None:
    ids = _ids()
    validate_segment_ids(
        ids,
        num_segments=6,
        item_count=int(ids.shape[0]),
    )


def test_validate_segment_ids_rejects_item_count_mismatch() -> None:
    with pytest.raises(
        ValueError,
        match="length must match",
    ):
        validate_segment_ids(
            _ids(),
            num_segments=6,
            item_count=5,
        )


@pytest.mark.parametrize(
    "item_count",
    (
        -1,
        True,
        1.5,
    ),
)
def test_validate_segment_ids_rejects_invalid_item_count(
    item_count: Any,
) -> None:
    with pytest.raises(
        ValueError,
        match="nonnegative integer",
    ):
        validate_segment_ids(
            _ids(),
            num_segments=6,
            item_count=item_count,
        )


@pytest.mark.parametrize(
    "num_segments",
    (
        -1,
        True,
        1.5,
    ),
)
def test_validate_segment_ids_rejects_invalid_num_segments(
    num_segments: Any,
) -> None:
    with pytest.raises(
        ValueError,
        match="nonnegative integer",
    ):
        validate_segment_ids(
            _ids(),
            num_segments=num_segments,
        )


def test_validate_segment_ids_rejects_zero_segments_for_nonempty_ids() -> None:
    with pytest.raises(
        ValueError,
        match="cannot be zero",
    ):
        validate_segment_ids(
            torch.tensor(
                [0],
                dtype=torch.long,
            ),
            num_segments=0,
        )


def test_validate_segment_ids_rejects_non_tensor() -> None:
    with pytest.raises(
        TypeError,
        match="must be a tensor",
    ):
        validate_segment_ids(  # type: ignore[arg-type]
            [0, 1],
            num_segments=2,
        )


def test_validate_segment_ids_rejects_wrong_rank() -> None:
    with pytest.raises(
        ValueError,
        match=r"shape \[M\]",
    ):
        validate_segment_ids(
            torch.tensor(
                [[0, 1]],
                dtype=torch.long,
            ),
            num_segments=2,
        )


@pytest.mark.parametrize(
    "dtype",
    (
        torch.int32,
        torch.float32,
        torch.bool,
    ),
)
def test_validate_segment_ids_rejects_wrong_dtype(
    dtype: torch.dtype,
) -> None:
    with pytest.raises(
        ValueError,
        match="torch.long",
    ):
        validate_segment_ids(
            torch.tensor(
                [0, 1],
                dtype=dtype,
            ),
            num_segments=2,
        )


def test_validate_segment_ids_rejects_negative_id() -> None:
    with pytest.raises(
        ValueError,
        match="out-of-range",
    ):
        validate_segment_ids(
            torch.tensor(
                [0, -1],
                dtype=torch.long,
            ),
            num_segments=2,
        )


def test_validate_segment_ids_rejects_id_equal_to_num_segments() -> None:
    with pytest.raises(
        ValueError,
        match="out-of-range",
    ):
        validate_segment_ids(
            torch.tensor(
                [0, 2],
                dtype=torch.long,
            ),
            num_segments=2,
        )


def test_validate_segment_ids_accepts_sparse_presence_with_dense_range() -> None:
    validate_segment_ids(
        torch.tensor(
            [0, 4, 4],
            dtype=torch.long,
        ),
        num_segments=5,
    )


# =============================================================================
# segment_counts and segment_presence
# =============================================================================


def test_segment_counts_exact_values() -> None:
    counts = segment_counts(
        _ids(),
        num_segments=6,
    )

    assert counts.dtype == torch.long
    assert torch.equal(
        counts,
        torch.tensor(
            [2, 0, 3, 0, 1, 0],
            dtype=torch.long,
        ),
    )


def test_segment_counts_empty_zero_segments() -> None:
    counts = segment_counts(
        torch.empty(
            0,
            dtype=torch.long,
        ),
        num_segments=0,
    )

    assert counts.shape == (0,)
    assert counts.dtype == torch.long


def test_segment_counts_empty_declared_segments() -> None:
    counts = segment_counts(
        torch.empty(
            0,
            dtype=torch.long,
        ),
        num_segments=4,
    )

    assert torch.equal(
        counts,
        torch.zeros(
            4,
            dtype=torch.long,
        ),
    )


def test_segment_presence_exact_values() -> None:
    presence = segment_presence(
        _ids(),
        num_segments=6,
    )

    assert presence.dtype == torch.bool
    assert torch.equal(
        presence,
        torch.tensor(
            [True, False, True, False, True, False],
            dtype=torch.bool,
        ),
    )


def test_segment_presence_matches_counts_positive() -> None:
    ids = _ids()
    assert torch.equal(
        segment_presence(
            ids,
            num_segments=6,
        ),
        segment_counts(
            ids,
            num_segments=6,
        )
        > 0,
    )


# =============================================================================
# segment_sum
# =============================================================================


@pytest.mark.parametrize(
    "values_factory",
    (
        _values_1d,
        _values_2d,
        _values_3d,
    ),
)
def test_segment_sum_matches_reference(
    values_factory: Any,
) -> None:
    values = values_factory()
    ids = _ids()

    observed = segment_sum(
        values,
        ids,
        num_segments=6,
    )
    expected = _reference_sum(
        values,
        ids,
        6,
    )

    assert torch.equal(
        observed,
        expected,
    )


def test_segment_sum_1d_exact_values() -> None:
    observed = segment_sum(
        _values_1d(),
        _ids(),
        num_segments=6,
    )

    assert torch.equal(
        observed,
        torch.tensor(
            [4.0, 0.0, 8.0, 0.0, 5.0, 0.0],
        ),
    )


def test_segment_sum_2d_exact_values() -> None:
    observed = segment_sum(
        _values_2d(),
        _ids(),
        num_segments=6,
    )

    expected = torch.tensor(
        [
            [4.0, 30.0],
            [0.0, 0.0],
            [8.0, 120.0],
            [0.0, 0.0],
            [5.0, 60.0],
            [0.0, 0.0],
        ]
    )

    assert torch.equal(
        observed,
        expected,
    )


def test_segment_sum_preserves_dtype_and_device() -> None:
    values = _values_2d(
        dtype=torch.float64,
    )
    observed = segment_sum(
        values,
        _ids(),
        num_segments=6,
    )

    assert observed.dtype == torch.float64
    assert observed.device == values.device


def test_segment_sum_supports_empty_values_with_zero_segments() -> None:
    values = torch.empty(
        0,
        2,
        3,
    )
    ids = torch.empty(
        0,
        dtype=torch.long,
    )

    observed = segment_sum(
        values,
        ids,
        num_segments=0,
    )

    assert observed.shape == (
        0,
        2,
        3,
    )


def test_segment_sum_supports_empty_values_with_declared_segments() -> None:
    values = torch.empty(
        0,
        2,
        3,
    )
    ids = torch.empty(
        0,
        dtype=torch.long,
    )

    observed = segment_sum(
        values,
        ids,
        num_segments=4,
    )

    assert observed.shape == (
        4,
        2,
        3,
    )
    assert torch.equal(
        observed,
        torch.zeros_like(observed),
    )


def test_segment_sum_rejects_scalar_values() -> None:
    with pytest.raises(
        ValueError,
        match="rank at least 1",
    ):
        segment_sum(
            torch.tensor(1.0),
            torch.empty(
                0,
                dtype=torch.long,
            ),
            num_segments=0,
        )


def test_segment_sum_rejects_nonfloating_values() -> None:
    with pytest.raises(
        ValueError,
        match="floating-point",
    ):
        segment_sum(
            torch.tensor(
                [1, 2],
                dtype=torch.long,
            ),
            torch.tensor(
                [0, 0],
                dtype=torch.long,
            ),
            num_segments=1,
        )


@pytest.mark.parametrize(
    "bad_value",
    (
        float("nan"),
        float("inf"),
        float("-inf"),
    ),
)
def test_segment_sum_rejects_nonfinite_values(
    bad_value: float,
) -> None:
    values = _values_1d()
    values[0] = bad_value

    with pytest.raises(
        ValueError,
        match="finite",
    ):
        segment_sum(
            values,
            _ids(),
            num_segments=6,
        )


def test_segment_sum_rejects_length_mismatch() -> None:
    with pytest.raises(
        ValueError,
        match="length must match",
    ):
        segment_sum(
            _values_1d(),
            torch.tensor(
                [0, 0],
                dtype=torch.long,
            ),
            num_segments=2,
        )


# =============================================================================
# segment_mean
# =============================================================================


@pytest.mark.parametrize(
    "values_factory",
    (
        _values_1d,
        _values_2d,
        _values_3d,
    ),
)
def test_segment_mean_matches_reference(
    values_factory: Any,
) -> None:
    values = values_factory()
    ids = _ids()

    observed = segment_mean(
        values,
        ids,
        num_segments=6,
    )
    expected = _reference_mean(
        values,
        ids,
        6,
    )

    assert torch.equal(
        observed,
        expected,
    )


def test_segment_mean_1d_exact_values() -> None:
    observed = segment_mean(
        _values_1d(),
        _ids(),
        num_segments=6,
    )

    assert torch.equal(
        observed,
        torch.tensor(
            [
                2.0,
                0.0,
                8.0 / 3.0,
                0.0,
                5.0,
                0.0,
            ]
        ),
    )


def test_segment_mean_empty_segments_are_exact_zero() -> None:
    observed = segment_mean(
        _values_2d(),
        _ids(),
        num_segments=6,
    )

    assert torch.equal(
        observed[
            torch.tensor(
                [False, True, False, True, False, True]
            )
        ],
        torch.zeros(
            3,
            2,
        ),
    )


def test_segment_mean_supports_empty_values() -> None:
    observed = segment_mean(
        torch.empty(
            0,
            3,
        ),
        torch.empty(
            0,
            dtype=torch.long,
        ),
        num_segments=4,
    )

    assert observed.shape == (
        4,
        3,
    )
    assert torch.equal(
        observed,
        torch.zeros_like(observed),
    )


def test_segment_mean_preserves_float64() -> None:
    observed = segment_mean(
        _values_2d(
            dtype=torch.float64,
        ),
        _ids(),
        num_segments=6,
    )

    assert observed.dtype == torch.float64


# =============================================================================
# SegmentMaxResult and segment_max
# =============================================================================


@pytest.mark.parametrize(
    "values_factory",
    (
        _values_1d,
        _values_2d,
        _values_3d,
    ),
)
def test_segment_max_matches_reference(
    values_factory: Any,
) -> None:
    values = values_factory()
    ids = _ids()

    observed = segment_max(
        values,
        ids,
        num_segments=6,
    )
    expected_values, expected_presence, expected_counts = (
        _reference_max(
            values,
            ids,
            6,
        )
    )

    assert torch.equal(
        observed.values,
        expected_values,
    )
    assert torch.equal(
        observed.present_mask,
        expected_presence,
    )
    assert torch.equal(
        observed.counts,
        expected_counts,
    )


def test_segment_max_1d_exact_values() -> None:
    observed = segment_max(
        _values_1d(),
        _ids(),
        num_segments=6,
    )

    assert torch.equal(
        observed.values,
        torch.tensor(
            [3.0, 0.0, 6.0, 0.0, 5.0, 0.0],
        ),
    )
    assert torch.equal(
        observed.present_mask,
        torch.tensor(
            [True, False, True, False, True, False],
        ),
    )
    assert torch.equal(
        observed.counts,
        torch.tensor(
            [2, 0, 3, 0, 1, 0],
            dtype=torch.long,
        ),
    )
    assert observed.num_segments == 6


def test_segment_max_absent_segments_are_exact_zero() -> None:
    observed = segment_max(
        _values_3d(),
        _ids(),
        num_segments=6,
    )
    absent = ~observed.present_mask

    assert torch.equal(
        observed.values[absent],
        torch.zeros_like(
            observed.values[absent]
        ),
    )


def test_segment_max_supports_empty_zero_segments() -> None:
    observed = segment_max(
        torch.empty(
            0,
            2,
        ),
        torch.empty(
            0,
            dtype=torch.long,
        ),
        num_segments=0,
    )

    assert observed.values.shape == (
        0,
        2,
    )
    assert observed.present_mask.shape == (
        0,
    )
    assert observed.counts.shape == (
        0,
    )


def test_segment_max_supports_empty_declared_segments() -> None:
    observed = segment_max(
        torch.empty(
            0,
            2,
        ),
        torch.empty(
            0,
            dtype=torch.long,
        ),
        num_segments=3,
    )

    assert observed.values.shape == (
        3,
        2,
    )
    assert torch.equal(
        observed.values,
        torch.zeros_like(
            observed.values
        ),
    )
    assert torch.equal(
        observed.present_mask,
        torch.zeros(
            3,
            dtype=torch.bool,
        ),
    )
    assert torch.equal(
        observed.counts,
        torch.zeros(
            3,
            dtype=torch.long,
        ),
    )


def test_segment_max_result_valid_manual_contract() -> None:
    result = SegmentMaxResult(
        values=torch.tensor(
            [
                [1.0, 2.0],
                [0.0, 0.0],
            ]
        ),
        present_mask=torch.tensor(
            [True, False],
            dtype=torch.bool,
        ),
        counts=torch.tensor(
            [2, 0],
            dtype=torch.long,
        ),
    )

    assert result.num_segments == 2


def test_segment_max_result_rejects_presence_mismatch() -> None:
    with pytest.raises(
        ValueError,
        match="counts > 0",
    ):
        SegmentMaxResult(
            values=torch.tensor(
                [1.0, 0.0]
            ),
            present_mask=torch.tensor(
                [False, False],
                dtype=torch.bool,
            ),
            counts=torch.tensor(
                [1, 0],
                dtype=torch.long,
            ),
        )


def test_segment_max_result_rejects_nonzero_absent_values() -> None:
    with pytest.raises(
        ValueError,
        match="exact zeros",
    ):
        SegmentMaxResult(
            values=torch.tensor(
                [1.0, 2.0]
            ),
            present_mask=torch.tensor(
                [True, False],
                dtype=torch.bool,
            ),
            counts=torch.tensor(
                [1, 0],
                dtype=torch.long,
            ),
        )


@pytest.mark.parametrize(
    "present_mask",
    (
        torch.tensor(
            [1, 0],
            dtype=torch.long,
        ),
        torch.tensor(
            [[True, False]],
            dtype=torch.bool,
        ),
    ),
)
def test_segment_max_result_rejects_invalid_presence(
    present_mask: torch.Tensor,
) -> None:
    with pytest.raises(ValueError):
        SegmentMaxResult(
            values=torch.tensor(
                [1.0, 0.0]
            ),
            present_mask=present_mask,
            counts=torch.tensor(
                [1, 0],
                dtype=torch.long,
            ),
        )


@pytest.mark.parametrize(
    "counts",
    (
        torch.tensor(
            [1.0, 0.0]
        ),
        torch.tensor(
            [[1, 0]],
            dtype=torch.long,
        ),
        torch.tensor(
            [1, -1],
            dtype=torch.long,
        ),
    ),
)
def test_segment_max_result_rejects_invalid_counts(
    counts: torch.Tensor,
) -> None:
    with pytest.raises(ValueError):
        SegmentMaxResult(
            values=torch.tensor(
                [1.0, 0.0]
            ),
            present_mask=torch.tensor(
                [True, False],
                dtype=torch.bool,
            ),
            counts=counts,
        )


# =============================================================================
# grouped_softmax
# =============================================================================


@pytest.mark.parametrize(
    "values_factory",
    (
        _values_1d,
        _values_2d,
        _values_3d,
    ),
)
def test_grouped_softmax_matches_reference(
    values_factory: Any,
) -> None:
    logits = values_factory()
    ids = _ids()

    observed = grouped_softmax(
        logits,
        ids,
        num_segments=6,
    )
    expected = _reference_grouped_softmax(
        logits,
        ids,
        6,
    )

    assert torch.allclose(
        observed,
        expected,
        atol=1e-6,
        rtol=1e-6,
    )


def test_grouped_softmax_preserves_shape_dtype_and_device() -> None:
    logits = _values_3d(
        dtype=torch.float64,
    )
    observed = grouped_softmax(
        logits,
        _ids(),
        num_segments=6,
    )

    assert observed.shape == logits.shape
    assert observed.dtype == logits.dtype
    assert observed.device == logits.device


def test_grouped_softmax_nonempty_groups_sum_to_one() -> None:
    logits = _values_2d()
    ids = _ids()

    weights = grouped_softmax(
        logits,
        ids,
        num_segments=6,
    )
    sums = grouped_softmax_sums(
        weights,
        ids,
        num_segments=6,
    )
    counts = segment_counts(
        ids,
        num_segments=6,
    )
    present = counts > 0

    assert torch.allclose(
        sums[present],
        torch.ones_like(
            sums[present]
        ),
    )
    assert torch.equal(
        sums[~present],
        torch.zeros_like(
            sums[~present]
        ),
    )


def test_grouped_softmax_single_item_segment_is_exact_one() -> None:
    ids = _ids()
    weights = grouped_softmax(
        _values_2d(),
        ids,
        num_segments=6,
    )
    counts = segment_counts(
        ids,
        num_segments=6,
    )
    single_edges = (
        counts[ids] == 1
    )

    assert torch.equal(
        weights[single_edges],
        torch.ones_like(
            weights[single_edges]
        ),
    )


def test_grouped_softmax_is_stable_for_extreme_logits() -> None:
    ids = torch.tensor(
        [0, 0, 0, 1, 1],
        dtype=torch.long,
    )
    logits = torch.tensor(
        [
            [10000.0, -10000.0],
            [9999.0, -9999.0],
            [9998.0, -9998.0],
            [-10000.0, 10000.0],
            [-9999.0, 9999.0],
        ]
    )

    weights = grouped_softmax(
        logits,
        ids,
        num_segments=3,
    )

    assert bool(
        torch.isfinite(weights)
        .all()
        .item()
    )
    assert_grouped_softmax_normalized(
        weights,
        ids,
        num_segments=3,
    )


def test_grouped_softmax_is_invariant_to_groupwise_shift() -> None:
    ids = _ids()
    logits = _values_2d()
    shifts = torch.tensor(
        [
            [1000.0, -1000.0],
            [2.0, 3.0],
            [-50.0, 25.0],
            [0.0, 0.0],
            [10.0, -10.0],
            [7.0, 8.0],
        ]
    )
    shifted = (
        logits
        + shifts[ids]
    )

    first = grouped_softmax(
        logits,
        ids,
        num_segments=6,
    )
    second = grouped_softmax(
        shifted,
        ids,
        num_segments=6,
    )

    assert torch.allclose(
        first,
        second,
        atol=1e-6,
        rtol=1e-6,
    )


def test_grouped_softmax_is_permutation_equivariant() -> None:
    ids = _ids()
    logits = _values_2d()
    permutation = torch.tensor(
        [5, 2, 0, 4, 1, 3],
        dtype=torch.long,
    )
    inverse = torch.empty_like(
        permutation
    )
    inverse[permutation] = torch.arange(
        int(permutation.shape[0])
    )

    original = grouped_softmax(
        logits,
        ids,
        num_segments=6,
    )
    permuted = grouped_softmax(
        logits[permutation],
        ids[permutation],
        num_segments=6,
    )

    assert torch.allclose(
        permuted[inverse],
        original,
        atol=1e-6,
        rtol=1e-6,
    )


def test_grouped_softmax_supports_empty_values_zero_segments() -> None:
    logits = torch.empty(
        0,
        2,
    )
    ids = torch.empty(
        0,
        dtype=torch.long,
    )

    weights = grouped_softmax(
        logits,
        ids,
        num_segments=0,
    )

    assert weights.shape == (
        0,
        2,
    )


def test_grouped_softmax_supports_empty_values_declared_segments() -> None:
    logits = torch.empty(
        0,
        2,
    )
    ids = torch.empty(
        0,
        dtype=torch.long,
    )

    weights = grouped_softmax(
        logits,
        ids,
        num_segments=4,
    )

    assert weights.shape == (
        0,
        2,
    )


def test_grouped_softmax_rejects_nonfinite_logits() -> None:
    logits = _values_1d()
    logits[0] = float("inf")

    with pytest.raises(
        ValueError,
        match="finite",
    ):
        grouped_softmax(
            logits,
            _ids(),
            num_segments=6,
        )


# =============================================================================
# grouped_softmax_sums and assertion helper
# =============================================================================


def test_grouped_softmax_sums_matches_segment_sum() -> None:
    ids = _ids()
    weights = grouped_softmax(
        _values_2d(),
        ids,
        num_segments=6,
    )

    assert torch.equal(
        grouped_softmax_sums(
            weights,
            ids,
            num_segments=6,
        ),
        segment_sum(
            weights,
            ids,
            num_segments=6,
        ),
    )


def test_assert_grouped_softmax_normalized_accepts_valid_weights() -> None:
    ids = _ids()
    weights = grouped_softmax(
        _values_3d(),
        ids,
        num_segments=6,
    )

    assert_grouped_softmax_normalized(
        weights,
        ids,
        num_segments=6,
    )


def test_assert_grouped_softmax_normalized_accepts_empty_weights() -> None:
    assert_grouped_softmax_normalized(
        torch.empty(
            0,
            2,
        ),
        torch.empty(
            0,
            dtype=torch.long,
        ),
        num_segments=4,
    )


def test_assert_grouped_softmax_normalized_rejects_negative_weights() -> None:
    ids = torch.tensor(
        [0, 0],
        dtype=torch.long,
    )

    with pytest.raises(
        ValueError,
        match="nonnegative",
    ):
        assert_grouped_softmax_normalized(
            torch.tensor(
                [1.1, -0.1]
            ),
            ids,
            num_segments=1,
        )


def test_assert_grouped_softmax_normalized_rejects_wrong_sum() -> None:
    ids = torch.tensor(
        [0, 0, 1],
        dtype=torch.long,
    )

    with pytest.raises(
        ValueError,
        match="must sum to one",
    ):
        assert_grouped_softmax_normalized(
            torch.tensor(
                [0.25, 0.25, 1.0]
            ),
            ids,
            num_segments=2,
        )


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("atol", -1.0),
        ("rtol", -1.0),
        ("atol", float("nan")),
        ("rtol", float("inf")),
        ("atol", True),
        ("rtol", True),
        ("atol", "bad"),
        ("rtol", "bad"),
    ),
)
def test_assert_grouped_softmax_normalized_rejects_invalid_tolerance(
    field: str,
    value: Any,
) -> None:
    ids = torch.tensor(
        [0],
        dtype=torch.long,
    )
    kwargs: dict[str, Any] = {
        "num_segments": 1,
        field: value,
    }

    with pytest.raises(
        ValueError,
        match="finite nonnegative",
    ):
        assert_grouped_softmax_normalized(
            torch.tensor([1.0]),
            ids,
            **kwargs,
        )


def test_assert_grouped_softmax_normalized_respects_custom_tolerance() -> None:
    ids = torch.tensor(
        [0, 0],
        dtype=torch.long,
    )
    weights = torch.tensor(
        [0.5005, 0.5005]
    )

    with pytest.raises(ValueError):
        assert_grouped_softmax_normalized(
            weights,
            ids,
            num_segments=1,
            atol=1e-5,
            rtol=1e-5,
        )

    assert_grouped_softmax_normalized(
        weights,
        ids,
        num_segments=1,
        atol=2e-3,
        rtol=0.0,
    )


# =============================================================================
# Autograd
# =============================================================================


@pytest.mark.parametrize(
    "operation",
    (
        segment_sum,
        segment_mean,
    ),
)
def test_sum_and_mean_backward_are_finite(
    operation: Any,
) -> None:
    values = _values_2d(
        requires_grad=True,
    )
    output = operation(
        values,
        _ids(),
        num_segments=6,
    )
    output.square().sum().backward()

    assert values.grad is not None
    assert values.grad.shape == (
        values.shape
    )
    assert bool(
        torch.isfinite(values.grad)
        .all()
        .item()
    )


def test_segment_max_backward_is_finite_without_ties() -> None:
    values = _values_2d(
        requires_grad=True,
    )
    result = segment_max(
        values,
        _ids(),
        num_segments=6,
    )
    result.values.square().sum().backward()

    assert values.grad is not None
    assert bool(
        torch.isfinite(values.grad)
        .all()
        .item()
    )


def test_grouped_softmax_backward_is_finite() -> None:
    logits = _values_3d(
        requires_grad=True,
    )
    weights = grouped_softmax(
        logits,
        _ids(),
        num_segments=6,
    )
    loss = (
        weights.square().sum()
    )
    loss.backward()

    assert logits.grad is not None
    assert logits.grad.shape == (
        logits.shape
    )
    assert bool(
        torch.isfinite(logits.grad)
        .all()
        .item()
    )


def test_grouped_softmax_group_gradient_sums_are_zero() -> None:
    logits = _values_2d(
        dtype=torch.float64,
        requires_grad=True,
    )
    weights = grouped_softmax(
        logits,
        _ids(),
        num_segments=6,
    )
    coefficients = torch.arange(
        weights.numel(),
        dtype=torch.float64,
    ).reshape_as(weights)
    (
        weights * coefficients
    ).sum().backward()

    gradient_sums = segment_sum(
        logits.grad,
        _ids(),
        num_segments=6,
    )
    present = segment_presence(
        _ids(),
        num_segments=6,
    )

    assert torch.allclose(
        gradient_sums[present],
        torch.zeros_like(
            gradient_sums[present]
        ),
        atol=1e-10,
        rtol=1e-10,
    )


def test_segment_sum_gradcheck() -> None:
    values = torch.randn(
        5,
        3,
        dtype=torch.float64,
        requires_grad=True,
    )
    ids = torch.tensor(
        [0, 0, 1, 2, 2],
        dtype=torch.long,
    )

    assert torch.autograd.gradcheck(
        lambda tensor: segment_sum(
            tensor,
            ids,
            num_segments=4,
        ),
        (values,),
        eps=1e-6,
        atol=1e-5,
        rtol=1e-3,
    )


def test_segment_mean_gradcheck() -> None:
    values = torch.randn(
        5,
        3,
        dtype=torch.float64,
        requires_grad=True,
    )
    ids = torch.tensor(
        [0, 0, 1, 2, 2],
        dtype=torch.long,
    )

    assert torch.autograd.gradcheck(
        lambda tensor: segment_mean(
            tensor,
            ids,
            num_segments=4,
        ),
        (values,),
        eps=1e-6,
        atol=1e-5,
        rtol=1e-3,
    )


def test_grouped_softmax_gradcheck() -> None:
    logits = torch.randn(
        5,
        2,
        dtype=torch.float64,
        requires_grad=True,
    )
    ids = torch.tensor(
        [0, 0, 1, 2, 2],
        dtype=torch.long,
    )

    assert torch.autograd.gradcheck(
        lambda tensor: grouped_softmax(
            tensor,
            ids,
            num_segments=4,
        ),
        (logits,),
        eps=1e-6,
        atol=1e-5,
        rtol=1e-3,
    )


# =============================================================================
# Algebraic properties
# =============================================================================


def test_segment_sum_is_permutation_invariant() -> None:
    ids = _ids()
    values = _values_3d()
    permutation = torch.tensor(
        [5, 2, 0, 4, 1, 3],
        dtype=torch.long,
    )

    first = segment_sum(
        values,
        ids,
        num_segments=6,
    )
    second = segment_sum(
        values[permutation],
        ids[permutation],
        num_segments=6,
    )

    assert torch.allclose(
        first,
        second,
        atol=1e-6,
        rtol=1e-6,
    )


def test_segment_mean_is_permutation_invariant() -> None:
    ids = _ids()
    values = _values_2d()
    permutation = torch.tensor(
        [5, 2, 0, 4, 1, 3],
        dtype=torch.long,
    )

    first = segment_mean(
        values,
        ids,
        num_segments=6,
    )
    second = segment_mean(
        values[permutation],
        ids[permutation],
        num_segments=6,
    )

    assert torch.allclose(
        first,
        second,
        atol=1e-6,
        rtol=1e-6,
    )


def test_segment_max_is_permutation_invariant() -> None:
    ids = _ids()
    values = _values_2d()
    permutation = torch.tensor(
        [5, 2, 0, 4, 1, 3],
        dtype=torch.long,
    )

    first = segment_max(
        values,
        ids,
        num_segments=6,
    )
    second = segment_max(
        values[permutation],
        ids[permutation],
        num_segments=6,
    )

    assert torch.equal(
        first.values,
        second.values,
    )
    assert torch.equal(
        first.present_mask,
        second.present_mask,
    )
    assert torch.equal(
        first.counts,
        second.counts,
    )


def test_segment_sum_is_linear() -> None:
    ids = _ids()
    first = _values_2d()
    second = torch.flip(
        _values_2d(),
        dims=(0,),
    )
    scalar = 2.5

    left = segment_sum(
        first + scalar * second,
        ids,
        num_segments=6,
    )
    right = (
        segment_sum(
            first,
            ids,
            num_segments=6,
        )
        + scalar
        * segment_sum(
            second,
            ids,
            num_segments=6,
        )
    )

    assert torch.allclose(
        left,
        right,
    )


def test_segment_mean_of_constant_values_is_constant_on_present_segments() -> None:
    ids = _ids()
    values = torch.full(
        (
            int(ids.shape[0]),
            3,
        ),
        7.25,
    )
    means = segment_mean(
        values,
        ids,
        num_segments=6,
    )
    present = segment_presence(
        ids,
        num_segments=6,
    )

    assert torch.equal(
        means[present],
        torch.full(
            (
                int(present.sum().item()),
                3,
            ),
            7.25,
        ),
    )
    assert torch.equal(
        means[~present],
        torch.zeros_like(
            means[~present]
        ),
    )


# =============================================================================
# Optional CUDA
# =============================================================================


@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA is unavailable.",
)
def test_values_and_ids_device_mismatch_is_rejected() -> None:
    values = _values_2d(
        device="cuda",
    )
    ids = _ids(
        device="cpu",
    )

    with pytest.raises(
        ValueError,
        match="share one device",
    ):
        segment_sum(
            values,
            ids,
            num_segments=6,
        )


@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA is unavailable.",
)
def test_cuda_operations_match_cpu() -> None:
    cpu_ids = _ids()
    cpu_values = _values_2d()
    cuda_ids = cpu_ids.cuda()
    cuda_values = cpu_values.cuda()

    assert torch.equal(
        segment_counts(
            cuda_ids,
            num_segments=6,
        ).cpu(),
        segment_counts(
            cpu_ids,
            num_segments=6,
        ),
    )
    assert torch.allclose(
        segment_sum(
            cuda_values,
            cuda_ids,
            num_segments=6,
        ).cpu(),
        segment_sum(
            cpu_values,
            cpu_ids,
            num_segments=6,
        ),
    )
    assert torch.allclose(
        segment_mean(
            cuda_values,
            cuda_ids,
            num_segments=6,
        ).cpu(),
        segment_mean(
            cpu_values,
            cpu_ids,
            num_segments=6,
        ),
    )
    assert torch.allclose(
        segment_max(
            cuda_values,
            cuda_ids,
            num_segments=6,
        ).values.cpu(),
        segment_max(
            cpu_values,
            cpu_ids,
            num_segments=6,
        ).values,
    )
    assert torch.allclose(
        grouped_softmax(
            cuda_values,
            cuda_ids,
            num_segments=6,
        ).cpu(),
        grouped_softmax(
            cpu_values,
            cpu_ids,
            num_segments=6,
        ),
    )


@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA is unavailable.",
)
def test_cuda_grouped_softmax_backward_is_finite() -> None:
    logits = _values_3d(
        device="cuda",
        requires_grad=True,
    )
    ids = _ids(
        device="cuda",
    )

    grouped_softmax(
        logits,
        ids,
        num_segments=6,
    ).square().sum().backward()

    assert logits.grad is not None
    assert bool(
        torch.isfinite(logits.grad)
        .all()
        .item()
    )
