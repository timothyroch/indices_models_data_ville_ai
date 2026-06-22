"""
Validated grouped tensor operations for functional message passing.

Target path:
    urban_resilience_models/
        v2_hazard_conditioned_functional_ugnn/
            functional_message_passing/
                segment_ops.py

This module owns generic first-axis grouped operations:

- segment counts;
- segment-presence masks;
- segment sum;
- segment mean;
- segment maximum with explicit empty-segment metadata;
- numerically stable grouped softmax.

It does not own:

- node, edge, relation, family, graph, or hazard semantics;
- attention group construction;
- target-node aggregation policy;
- relation registries;
- trainable parameters;
- model configuration;
- metadata-bearing FMP outputs.

All grouping is expressed by:

    values:       [M, ...]
    segment_ids:  [M]
    num_segments: S

The first axis is grouped. Every trailing coordinate is reduced independently.

Contract
--------
- ``segment_ids`` uses ``torch.long``.
- ``values`` and ``segment_ids`` share one device.
- neural values are floating-point and finite.
- segment IDs lie in ``[0, num_segments - 1]``.
- empty inputs and absent segments are valid.
- absent sums and means are exact zeros.
- absent maxima are exact zeros but are never ambiguous because
  ``segment_max`` returns an explicit ``present_mask`` and ``counts``.
- grouped softmax normalizes every nonempty segment independently over the
  first axis for every trailing coordinate.
- no hidden device movement or dtype conversion occurs.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

import torch


# =============================================================================
# Public identity
# =============================================================================


SEGMENT_OPS_SCHEMA_VERSION: Final[str] = "0.1"


# =============================================================================
# Validation helpers
# =============================================================================


def _require_nonnegative_int(
    name: str,
    value: int,
) -> None:
    if (
        isinstance(value, bool)
        or not isinstance(value, int)
        or value < 0
    ):
        raise ValueError(
            f"{name} must be a nonnegative integer."
        )


def _require_tensor(
    name: str,
    value: torch.Tensor,
) -> None:
    if not isinstance(value, torch.Tensor):
        raise TypeError(
            f"{name} must be a tensor."
        )


def _require_values(
    values: torch.Tensor,
) -> None:
    _require_tensor("values", values)

    if values.ndim < 1:
        raise ValueError(
            "values must have rank at least 1 with the grouped "
            "items on axis 0."
        )

    if not values.dtype.is_floating_point:
        raise ValueError(
            "values must use a floating-point dtype."
        )

    if not bool(
        torch.isfinite(values)
        .all()
        .item()
    ):
        raise ValueError(
            "values must contain only finite values."
        )


def _require_segment_ids(
    segment_ids: torch.Tensor,
) -> None:
    _require_tensor(
        "segment_ids",
        segment_ids,
    )

    if segment_ids.ndim != 1:
        raise ValueError(
            "segment_ids must have shape [M]."
        )

    if segment_ids.dtype != torch.long:
        raise ValueError(
            "segment_ids must use torch.long."
        )


def _validate_grouping(
    *,
    item_count: int,
    segment_ids: torch.Tensor,
    num_segments: int,
    values_device: torch.device | None = None,
) -> None:
    _require_nonnegative_int(
        "item_count",
        item_count,
    )
    _require_nonnegative_int(
        "num_segments",
        num_segments,
    )
    _require_segment_ids(segment_ids)

    if int(segment_ids.shape[0]) != item_count:
        raise ValueError(
            "segment_ids length must match the first dimension "
            f"of values. Observed {int(segment_ids.shape[0])} "
            f"and {item_count}."
        )

    if (
        values_device is not None
        and segment_ids.device != values_device
    ):
        raise ValueError(
            "values and segment_ids must share one device. "
            f"Observed {values_device} and {segment_ids.device}."
        )

    if item_count == 0:
        return

    if num_segments == 0:
        raise ValueError(
            "num_segments cannot be zero when segment_ids is nonempty."
        )

    minimum = int(
        segment_ids.min().item()
    )
    maximum = int(
        segment_ids.max().item()
    )

    if minimum < 0 or maximum >= num_segments:
        raise ValueError(
            "segment_ids contains out-of-range values. "
            f"Observed range [{minimum}, {maximum}]; "
            f"valid range is [0, {num_segments - 1}]."
        )


def _output_shape(
    values: torch.Tensor,
    num_segments: int,
) -> tuple[int, ...]:
    return (
        num_segments,
        *tuple(values.shape[1:]),
    )


def _expanded_segment_ids(
    segment_ids: torch.Tensor,
    values: torch.Tensor,
) -> torch.Tensor:
    if values.ndim == 1:
        return segment_ids

    view_shape = (
        int(segment_ids.shape[0]),
        *([1] * (values.ndim - 1)),
    )

    return (
        segment_ids
        .view(view_shape)
        .expand_as(values)
    )


def _expanded_counts(
    counts: torch.Tensor,
    values_ndim: int,
) -> torch.Tensor:
    if values_ndim == 1:
        return counts

    return counts.view(
        counts.shape[0],
        *([1] * (values_ndim - 1)),
    )


def _assert_finite_output(
    name: str,
    value: torch.Tensor,
) -> None:
    if not bool(
        torch.isfinite(value)
        .all()
        .item()
    ):
        raise FloatingPointError(
            f"{name} produced NaN or infinity from finite inputs."
        )


# =============================================================================
# Public validation
# =============================================================================


def validate_segment_ids(
    segment_ids: torch.Tensor,
    *,
    num_segments: int,
    item_count: int | None = None,
    device: torch.device | None = None,
) -> None:
    """
    Validate a dense zero-based segment-ID vector.

    Parameters
    ----------
    segment_ids:
        Dense group IDs with shape ``[M]`` and dtype ``torch.long``.

    num_segments:
        Total number of possible segments. Empty segments are allowed.

    item_count:
        Expected ``M``. Defaults to ``len(segment_ids)``.

    device:
        Optional device that ``segment_ids`` must share.
    """

    _require_segment_ids(segment_ids)

    resolved_item_count = (
        int(segment_ids.shape[0])
        if item_count is None
        else item_count
    )

    _validate_grouping(
        item_count=resolved_item_count,
        segment_ids=segment_ids,
        num_segments=num_segments,
        values_device=device,
    )


# =============================================================================
# Counts and presence
# =============================================================================


def segment_counts(
    segment_ids: torch.Tensor,
    *,
    num_segments: int,
) -> torch.Tensor:
    """
    Count items in every segment.

    Returns
    -------
    torch.Tensor
        ``torch.long`` tensor with shape ``[num_segments]``.
        Empty segments contain zero.
    """

    validate_segment_ids(
        segment_ids,
        num_segments=num_segments,
    )

    counts = torch.bincount(
        segment_ids,
        minlength=num_segments,
    )

    if int(counts.shape[0]) != num_segments:
        counts = counts[:num_segments]

    return counts


def segment_presence(
    segment_ids: torch.Tensor,
    *,
    num_segments: int,
) -> torch.Tensor:
    """
    Return a Boolean mask identifying nonempty segments.
    """

    return (
        segment_counts(
            segment_ids,
            num_segments=num_segments,
        )
        > 0
    )


# =============================================================================
# Sum and mean
# =============================================================================


def segment_sum(
    values: torch.Tensor,
    segment_ids: torch.Tensor,
    *,
    num_segments: int,
) -> torch.Tensor:
    """
    Sum ``values`` independently within every segment.

    ``values`` may have shape ``[M]`` or ``[M, ...]``. The output has shape
    ``[num_segments, ...]``. Empty segments are exact zeros.
    """

    _require_values(values)
    _validate_grouping(
        item_count=int(values.shape[0]),
        segment_ids=segment_ids,
        num_segments=num_segments,
        values_device=values.device,
    )

    result = torch.zeros(
        _output_shape(
            values,
            num_segments,
        ),
        dtype=values.dtype,
        device=values.device,
    )

    if int(values.shape[0]) > 0:
        result.index_add_(
            0,
            segment_ids,
            values,
        )

    _assert_finite_output(
        "segment_sum",
        result,
    )
    return result


def segment_mean(
    values: torch.Tensor,
    segment_ids: torch.Tensor,
    *,
    num_segments: int,
) -> torch.Tensor:
    """
    Compute the arithmetic mean independently within every segment.

    Empty segments are exact zeros. No division by zero occurs.
    """

    sums = segment_sum(
        values,
        segment_ids,
        num_segments=num_segments,
    )
    counts = segment_counts(
        segment_ids,
        num_segments=num_segments,
    )

    denominator = (
        _expanded_counts(
            counts,
            values.ndim,
        )
        .clamp_min(1)
        .to(dtype=values.dtype)
    )

    result = sums / denominator

    _assert_finite_output(
        "segment_mean",
        result,
    )
    return result


# =============================================================================
# Maximum
# =============================================================================


@dataclass(slots=True, frozen=True)
class SegmentMaxResult:
    """
    Maximum values plus explicit empty-segment metadata.

    ``values`` uses exact zeros for absent segments. Consumers must use
    ``present_mask`` when the difference between a true zero maximum and an
    absent segment matters.
    """

    values: torch.Tensor
    present_mask: torch.Tensor
    counts: torch.Tensor

    schema_version: str = (
        SEGMENT_OPS_SCHEMA_VERSION
    )

    def __post_init__(self) -> None:
        _require_values(self.values)
        _require_tensor(
            "present_mask",
            self.present_mask,
        )
        _require_tensor(
            "counts",
            self.counts,
        )

        if self.present_mask.ndim != 1:
            raise ValueError(
                "present_mask must have shape [num_segments]."
            )

        if self.present_mask.dtype != torch.bool:
            raise ValueError(
                "present_mask must use torch.bool."
            )

        if self.counts.ndim != 1:
            raise ValueError(
                "counts must have shape [num_segments]."
            )

        if self.counts.dtype != torch.long:
            raise ValueError(
                "counts must use torch.long."
            )

        num_segments = int(
            self.values.shape[0]
        )

        if tuple(
            self.present_mask.shape
        ) != (num_segments,):
            raise ValueError(
                "present_mask must align with values axis 0."
            )

        if tuple(
            self.counts.shape
        ) != (num_segments,):
            raise ValueError(
                "counts must align with values axis 0."
            )

        if not (
            self.values.device
            == self.present_mask.device
            == self.counts.device
        ):
            raise ValueError(
                "SegmentMaxResult tensors must share one device."
            )

        if bool(
            (self.counts < 0)
            .any()
            .item()
        ):
            raise ValueError(
                "counts must be nonnegative."
            )

        expected_presence = self.counts > 0

        if not torch.equal(
            self.present_mask,
            expected_presence,
        ):
            raise ValueError(
                "present_mask must equal counts > 0."
            )

        absent = ~self.present_mask

        if bool(absent.any().item()):
            expanded_absent = _expanded_counts(
                absent,
                self.values.ndim,
            )

            if not torch.equal(
                self.values.masked_select(
                    expanded_absent.expand_as(
                        self.values
                    )
                ),
                torch.zeros_like(
                    self.values.masked_select(
                        expanded_absent.expand_as(
                            self.values
                        )
                    )
                ),
            ):
                raise ValueError(
                    "Absent segment maxima must be exact zeros."
                )

    @property
    def num_segments(self) -> int:
        return int(self.values.shape[0])


def segment_max(
    values: torch.Tensor,
    segment_ids: torch.Tensor,
    *,
    num_segments: int,
) -> SegmentMaxResult:
    """
    Compute per-segment maxima with explicit presence metadata.

    Internally, absent segments use ``-inf`` only as a reduction sentinel.
    The public result replaces that sentinel with exact zeros and exposes
    ``present_mask`` so absence is never mistaken for a valid maximum.
    """

    _require_values(values)
    _validate_grouping(
        item_count=int(values.shape[0]),
        segment_ids=segment_ids,
        num_segments=num_segments,
        values_device=values.device,
    )

    counts = segment_counts(
        segment_ids,
        num_segments=num_segments,
    )
    present = counts > 0
    output_shape = _output_shape(
        values,
        num_segments,
    )

    if int(values.shape[0]) == 0:
        maxima = torch.zeros(
            output_shape,
            dtype=values.dtype,
            device=values.device,
        )
        return SegmentMaxResult(
            values=maxima,
            present_mask=present,
            counts=counts,
        )

    maxima = torch.full(
        output_shape,
        float("-inf"),
        dtype=values.dtype,
        device=values.device,
    )

    expanded_ids = _expanded_segment_ids(
        segment_ids,
        values,
    )

    maxima.scatter_reduce_(
        0,
        expanded_ids,
        values,
        reduce="amax",
        include_self=True,
    )

    expanded_presence = (
        _expanded_counts(
            present,
            values.ndim,
        )
        .expand_as(maxima)
    )

    maxima = torch.where(
        expanded_presence,
        maxima,
        torch.zeros_like(maxima),
    )

    _assert_finite_output(
        "segment_max",
        maxima,
    )

    return SegmentMaxResult(
        values=maxima,
        present_mask=present,
        counts=counts,
    )


# =============================================================================
# Grouped softmax
# =============================================================================


def grouped_softmax(
    logits: torch.Tensor,
    segment_ids: torch.Tensor,
    *,
    num_segments: int,
) -> torch.Tensor:
    """
    Numerically stable softmax within each nonempty segment.

    Parameters
    ----------
    logits:
        Floating finite tensor with shape ``[M]`` or ``[M, ...]``.

    segment_ids:
        Dense zero-based group IDs with shape ``[M]``.

    num_segments:
        Total number of possible groups. Empty groups are valid.

    Returns
    -------
    torch.Tensor
        Tensor with the same shape, dtype, and device as ``logits``.

        For every nonempty segment and every trailing coordinate, weights sum
        to one within floating-point tolerance. A one-item segment receives
        exact weight one.
    """

    _require_values(logits)
    _validate_grouping(
        item_count=int(logits.shape[0]),
        segment_ids=segment_ids,
        num_segments=num_segments,
        values_device=logits.device,
    )

    if int(logits.shape[0]) == 0:
        return torch.empty_like(logits)

    maxima = segment_max(
        logits,
        segment_ids,
        num_segments=num_segments,
    ).values

    centered = (
        logits
        - maxima[segment_ids]
    )
    exponentials = torch.exp(centered)

    denominators = segment_sum(
        exponentials,
        segment_ids,
        num_segments=num_segments,
    )

    gathered_denominators = (
        denominators[segment_ids]
    )

    if bool(
        (gathered_denominators <= 0)
        .any()
        .item()
    ):
        raise FloatingPointError(
            "grouped_softmax encountered a nonpositive denominator."
        )

    weights = (
        exponentials
        / gathered_denominators
    )

    _assert_finite_output(
        "grouped_softmax",
        weights,
    )

    return weights


def grouped_softmax_sums(
    weights: torch.Tensor,
    segment_ids: torch.Tensor,
    *,
    num_segments: int,
) -> torch.Tensor:
    """
    Sum candidate grouped-softmax weights by segment.

    This diagnostic helper performs no normalization. It is useful for
    asserting that every nonempty segment sums to one.
    """

    return segment_sum(
        weights,
        segment_ids,
        num_segments=num_segments,
    )


def assert_grouped_softmax_normalized(
    weights: torch.Tensor,
    segment_ids: torch.Tensor,
    *,
    num_segments: int,
    atol: float | None = None,
    rtol: float | None = None,
) -> None:
    """
    Raise ``ValueError`` unless nonempty segments sum to one.

    Empty segments are expected to sum to zero.
    """

    _require_values(weights)
    _validate_grouping(
        item_count=int(weights.shape[0]),
        segment_ids=segment_ids,
        num_segments=num_segments,
        values_device=weights.device,
    )

    if bool(
        (weights < 0)
        .any()
        .item()
    ):
        raise ValueError(
            "Grouped-softmax weights must be nonnegative."
        )

    sums = grouped_softmax_sums(
        weights,
        segment_ids,
        num_segments=num_segments,
    )
    counts = segment_counts(
        segment_ids,
        num_segments=num_segments,
    )
    present = counts > 0

    if atol is None or rtol is None:
        if weights.dtype in (
            torch.float16,
            torch.bfloat16,
        ):
            default_atol = 5e-3
            default_rtol = 5e-3
        elif weights.dtype == torch.float64:
            default_atol = 1e-10
            default_rtol = 1e-10
        else:
            default_atol = 1e-5
            default_rtol = 1e-5

        resolved_atol = (
            default_atol
            if atol is None
            else atol
        )
        resolved_rtol = (
            default_rtol
            if rtol is None
            else rtol
        )
    else:
        resolved_atol = atol
        resolved_rtol = rtol

    for name, tolerance in (
        ("atol", resolved_atol),
        ("rtol", resolved_rtol),
    ):
        if (
            isinstance(tolerance, bool)
            or not isinstance(
                tolerance,
                (int, float),
            )
            or not torch.isfinite(
                torch.tensor(
                    float(tolerance)
                )
            ).item()
            or float(tolerance) < 0
        ):
            raise ValueError(
                f"{name} must be a finite nonnegative number."
            )

    expanded_presence = (
        _expanded_counts(
            present,
            weights.ndim,
        )
        .expand_as(sums)
    )

    if bool(present.any().item()):
        observed = sums.masked_select(
            expanded_presence
        )
        expected = torch.ones_like(
            observed
        )

        if not torch.allclose(
            observed,
            expected,
            atol=float(resolved_atol),
            rtol=float(resolved_rtol),
        ):
            raise ValueError(
                "Every nonempty segment must sum to one."
            )

    absent = ~present

    if bool(absent.any().item()):
        expanded_absent = (
            _expanded_counts(
                absent,
                weights.ndim,
            )
            .expand_as(sums)
        )
        absent_sums = sums.masked_select(
            expanded_absent
        )

        if not torch.equal(
            absent_sums,
            torch.zeros_like(absent_sums),
        ):
            raise ValueError(
                "Empty segments must sum to exact zero."
            )


__all__ = (
    "SEGMENT_OPS_SCHEMA_VERSION",
    "SegmentMaxResult",
    "assert_grouped_softmax_normalized",
    "grouped_softmax",
    "grouped_softmax_sums",
    "segment_counts",
    "segment_max",
    "segment_mean",
    "segment_presence",
    "segment_sum",
    "validate_segment_ids",
)
