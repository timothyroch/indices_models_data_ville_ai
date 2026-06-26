"""
Typed temporal-coordinate contracts for shared urban-memory histories.

Target path:
    urban_resilience_models/
        v2_hazard_conditioned_functional_ugnn/
            memory/
                schemas/
                    temporal_coordinates.py

This module owns:

- absolute temporal coordinates with shape ``[N, T]``;
- relative temporal offsets with shape ``[N, T]``;
- explicit temporal layout semantics;
- explicit chronology and duplicate-time policies;
- padding-direction vocabulary used by historical inputs;
- absolute reference-time semantics;
- relative anchor semantics;
- regular-grid spacing validation;
- deterministic semantic, value, and temporal-axis fingerprints;
- device-preserving reconstruction;
- alignment validation against a history timestep mask.

It does not own:

- historical feature values;
- timestep or feature-observation masks;
- node or feature axes;
- missing-value policy;
- recurrent, Transformer, pooling, or retrieval behavior;
- hazard-query semantics;
- conversion from raw datetime objects.

Coordinate convention
---------------------
All model-facing coordinates use shape ``[N, T]`` and are ordered from oldest
to newest over valid temporal positions.

Two layouts are recognized:

``event_sequence``
    ``T`` indexes declared temporal events. Unequal coordinate gaps are valid.

``regular_grid``
    ``T`` indexes a regular temporal grid. A positive ``regular_step`` is
    required and consecutive valid coordinates must differ by that step.

Absolute coordinates use ``torch.long``. Their interpretation is frozen by an
explicit unit, calendar, and timezone convention.

Relative coordinates use a floating dtype. By default they represent history
relative to a prediction or reference origin:

    reference origin = 0
    historical observations <= 0

The latest valid observation is not forced to equal zero unless the explicit
``latest_observation`` anchor is selected. This preserves observation
staleness across nodes.

Mask relationship
-----------------
The coordinate object does not own the history timestep mask. Call
``validate_against_mask(...)`` or ``validate_temporal_coordinates(...)`` at the
history-input boundary.

Padding coordinates must equal the canonical value zero. This value is
interpreted only through the timestep mask and therefore does not become a
valid timestamp or temporal observation by itself.

Scientific interpretation
--------------------------
Temporal coordinates encode model-facing order and spacing. They do not prove
causal ordering, feature availability, absence of leakage, or correctness of
the upstream calendar transformation. Those properties require the complete
history-input and data-pipeline contracts.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from hashlib import sha256
import json
import math
from typing import Any, Final, Mapping, TypeAlias

import torch


# =============================================================================
# Schema identity
# =============================================================================


ABSOLUTE_TEMPORAL_COORDINATES_SCHEMA_VERSION: Final[str] = "0.1"
RELATIVE_TEMPORAL_COORDINATES_SCHEMA_VERSION: Final[str] = "0.1"

TEMPORAL_COORDINATE_CANONICAL_PADDING_VALUE: Final[int | float] = 0

TEMPORAL_COORDINATE_SCIENTIFIC_INTERPRETATION: Final[str] = (
    "model_facing_temporal_order_and_spacing_not_causal_provenance"
)


# =============================================================================
# Controlled temporal vocabularies
# =============================================================================


class TemporalCoordinateKind(StrEnum):
    """Discriminator for one temporal-coordinate representation."""

    ABSOLUTE = "absolute"
    RELATIVE = "relative"


class TemporalLayout(StrEnum):
    """Meaning of the packed temporal axis."""

    EVENT_SEQUENCE = "event_sequence"
    REGULAR_GRID = "regular_grid"


class TemporalChronologicalOrder(StrEnum):
    """Canonical ordering of valid temporal positions."""

    OLDEST_TO_NEWEST = "oldest_to_newest"


class TemporalPaddingDirection(StrEnum):
    """Location of batch padding relative to valid temporal positions."""

    LEFT = "left"
    RIGHT = "right"
    NONE = "none"


class TemporalDuplicatePolicy(StrEnum):
    """Whether equal consecutive valid coordinates are permitted."""

    ALLOW_EQUAL = "allow_equal"
    ERROR = "error"


class AbsoluteTemporalReferenceKind(StrEnum):
    """Meaning of optional absolute reference-time values."""

    NONE = "none"
    PREDICTION_ORIGIN = "prediction_origin"
    EXPLICIT_REFERENCE_TIME = "explicit_reference_time"


class RelativeTemporalAnchor(StrEnum):
    """Anchor used to interpret relative temporal offsets."""

    PREDICTION_ORIGIN = "prediction_origin"
    EXPLICIT_REFERENCE_TIME = "explicit_reference_time"
    LATEST_OBSERVATION = "latest_observation"


CANONICAL_TEMPORAL_COORDINATE_KINDS: Final[tuple[str, ...]] = tuple(
    value.value
    for value in TemporalCoordinateKind
)

CANONICAL_TEMPORAL_LAYOUTS: Final[tuple[str, ...]] = tuple(
    value.value
    for value in TemporalLayout
)

CANONICAL_TEMPORAL_CHRONOLOGICAL_ORDERS: Final[tuple[str, ...]] = tuple(
    value.value
    for value in TemporalChronologicalOrder
)

CANONICAL_TEMPORAL_PADDING_DIRECTIONS: Final[tuple[str, ...]] = tuple(
    value.value
    for value in TemporalPaddingDirection
)

CANONICAL_TEMPORAL_DUPLICATE_POLICIES: Final[tuple[str, ...]] = tuple(
    value.value
    for value in TemporalDuplicatePolicy
)

CANONICAL_ABSOLUTE_TEMPORAL_REFERENCE_KINDS: Final[
    tuple[str, ...]
] = tuple(
    value.value
    for value in AbsoluteTemporalReferenceKind
)

CANONICAL_RELATIVE_TEMPORAL_ANCHORS: Final[tuple[str, ...]] = tuple(
    value.value
    for value in RelativeTemporalAnchor
)


# =============================================================================
# Generic helpers
# =============================================================================


def _require_nonempty_string(
    name: str,
    value: str,
) -> None:
    if not isinstance(value, str):
        raise TypeError(
            f"{name} must be a string."
        )

    if not value.strip():
        raise ValueError(
            f"{name} must be a non-empty string."
        )


def _require_boolean(
    name: str,
    value: bool,
) -> None:
    if not isinstance(value, bool):
        raise TypeError(
            f"{name} must be a Boolean."
        )


def _require_positive_int(
    name: str,
    value: int,
) -> None:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(
            f"{name} must be an integer."
        )

    if value <= 0:
        raise ValueError(
            f"{name} must be strictly positive."
        )


def _require_positive_finite_number(
    name: str,
    value: int | float,
) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(
            f"{name} must be numeric."
        )

    converted = float(value)

    if not math.isfinite(converted):
        raise ValueError(
            f"{name} must be finite."
        )

    if converted <= 0.0:
        raise ValueError(
            f"{name} must be strictly positive."
        )

    return converted


def _normalize_temporal_layout(
    value: TemporalLayout | str,
) -> TemporalLayout:
    if isinstance(value, TemporalLayout):
        return value

    return TemporalLayout(value)


def _normalize_chronological_order(
    value: TemporalChronologicalOrder | str,
) -> TemporalChronologicalOrder:
    if isinstance(value, TemporalChronologicalOrder):
        return value

    return TemporalChronologicalOrder(value)


def _normalize_padding_direction(
    value: TemporalPaddingDirection | str,
) -> TemporalPaddingDirection:
    if isinstance(value, TemporalPaddingDirection):
        return value

    return TemporalPaddingDirection(value)


def _normalize_duplicate_policy(
    value: TemporalDuplicatePolicy | str,
) -> TemporalDuplicatePolicy:
    if isinstance(value, TemporalDuplicatePolicy):
        return value

    return TemporalDuplicatePolicy(value)


def _normalize_absolute_reference_kind(
    value: AbsoluteTemporalReferenceKind | str,
) -> AbsoluteTemporalReferenceKind:
    if isinstance(value, AbsoluteTemporalReferenceKind):
        return value

    return AbsoluteTemporalReferenceKind(value)


def _normalize_relative_anchor(
    value: RelativeTemporalAnchor | str,
) -> RelativeTemporalAnchor:
    if isinstance(value, RelativeTemporalAnchor):
        return value

    return RelativeTemporalAnchor(value)


def _require_rank_two_tensor(
    name: str,
    value: torch.Tensor,
) -> None:
    if not isinstance(value, torch.Tensor):
        raise TypeError(
            f"{name} must be a tensor."
        )

    if value.ndim != 2:
        raise ValueError(
            f"{name} must have shape [N, T]; "
            f"observed {tuple(value.shape)}."
        )

    node_count = int(value.shape[0])
    sequence_length = int(value.shape[1])

    _require_positive_int(
        f"{name} node dimension",
        node_count,
    )
    _require_positive_int(
        f"{name} temporal dimension",
        sequence_length,
    )


def _require_long_matrix(
    name: str,
    value: torch.Tensor,
) -> None:
    _require_rank_two_tensor(
        name,
        value,
    )

    if value.dtype != torch.long:
        raise ValueError(
            f"{name} must use torch.long."
        )


def _require_float_matrix(
    name: str,
    value: torch.Tensor,
) -> None:
    _require_rank_two_tensor(
        name,
        value,
    )

    if not value.dtype.is_floating_point:
        raise ValueError(
            f"{name} must use a floating-point dtype."
        )

    if not bool(
        torch.isfinite(value).all().item()
    ):
        raise ValueError(
            f"{name} must contain only finite values."
        )


def _require_bool_mask(
    name: str,
    value: torch.Tensor,
    *,
    shape: tuple[int, int],
    device: torch.device,
) -> None:
    if not isinstance(value, torch.Tensor):
        raise TypeError(
            f"{name} must be a tensor."
        )

    if value.dtype != torch.bool:
        raise ValueError(
            f"{name} must use torch.bool."
        )

    if tuple(value.shape) != shape:
        raise ValueError(
            f"{name} must have shape {shape}; "
            f"observed {tuple(value.shape)}."
        )

    if value.device != device:
        raise ValueError(
            f"{name} and temporal coordinates must share one device."
        )


def _require_reference_vector(
    name: str,
    value: torch.Tensor,
    *,
    node_count: int,
    device: torch.device,
) -> None:
    if not isinstance(value, torch.Tensor):
        raise TypeError(
            f"{name} must be a tensor."
        )

    if value.ndim != 1:
        raise ValueError(
            f"{name} must have shape [{node_count}]; "
            f"observed {tuple(value.shape)}."
        )

    if value.dtype != torch.long:
        raise ValueError(
            f"{name} must use torch.long."
        )

    if int(value.shape[0]) != node_count:
        raise ValueError(
            f"{name} must contain one value per node."
        )

    if value.device != device:
        raise ValueError(
            f"{name} and temporal coordinates must share one device."
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


def _tensor_fingerprint(
    tensors: Mapping[str, torch.Tensor],
) -> str:
    digest = sha256()

    for name in sorted(tensors):
        tensor = (
            tensors[name]
            .detach()
            .cpu()
            .contiguous()
        )

        digest.update(
            name.encode("utf-8")
        )
        digest.update(
            str(tensor.dtype).encode("utf-8")
        )
        digest.update(
            json.dumps(
                list(tensor.shape),
                separators=(",", ":"),
            ).encode("utf-8")
        )
        digest.update(
            tensor.view(torch.uint8)
            .numpy()
            .tobytes()
        )

    return digest.hexdigest()


def _validate_layout_step(
    *,
    layout: TemporalLayout,
    regular_step: int | float | None,
    require_integer: bool,
) -> int | float | None:
    if layout == TemporalLayout.EVENT_SEQUENCE:
        if regular_step is not None:
            raise ValueError(
                "regular_step must be None for event_sequence layout."
            )

        return None

    if regular_step is None:
        raise ValueError(
            "regular_step is required for regular_grid layout."
        )

    if require_integer:
        if isinstance(regular_step, bool) or not isinstance(regular_step, int):
            raise TypeError(
                "Absolute regular_step must be an integer."
            )

        if regular_step <= 0:
            raise ValueError(
                "Absolute regular_step must be strictly positive."
            )

        return regular_step

    return _require_positive_finite_number(
        "regular_step",
        regular_step,
    )


def _validate_padding_contiguity(
    timestep_mask: torch.Tensor,
    *,
    padding_direction: TemporalPaddingDirection,
) -> None:
    sequence_length = int(
        timestep_mask.shape[1]
    )

    for row_index in range(
        int(timestep_mask.shape[0])
    ):
        row = timestep_mask[row_index]
        valid_indices = torch.nonzero(
            row,
            as_tuple=False,
        ).flatten()

        if valid_indices.numel() == 0:
            continue

        first = int(
            valid_indices[0].item()
        )
        last = int(
            valid_indices[-1].item()
        )
        valid_count = int(
            valid_indices.numel()
        )

        if last - first + 1 != valid_count:
            raise ValueError(
                "timestep_mask valid positions must be contiguous; "
                f"row {row_index} contains internal padding."
            )

        if padding_direction == TemporalPaddingDirection.LEFT:
            if last != sequence_length - 1:
                raise ValueError(
                    "Left-padded histories must place valid positions "
                    f"at the end of the sequence; row {row_index} does not."
                )

        elif padding_direction == TemporalPaddingDirection.RIGHT:
            if first != 0:
                raise ValueError(
                    "Right-padded histories must place valid positions "
                    f"at the beginning of the sequence; row {row_index} does not."
                )

        else:
            if valid_count != sequence_length:
                raise ValueError(
                    "padding_direction='none' requires every temporal "
                    f"position to be valid; row {row_index} contains padding."
                )


def _validate_canonical_padding_values(
    values: torch.Tensor,
    timestep_mask: torch.Tensor,
) -> None:
    padding_values = values[
        ~timestep_mask
    ]

    if padding_values.numel() == 0:
        return

    expected = torch.zeros_like(
        padding_values
    )

    if not torch.equal(
        padding_values,
        expected,
    ):
        raise ValueError(
            "Temporal coordinates at padded positions must equal "
            "the canonical padding value zero."
        )


def _validate_monotonicity_and_duplicates(
    values: torch.Tensor,
    timestep_mask: torch.Tensor,
    *,
    duplicate_policy: TemporalDuplicatePolicy,
) -> None:
    for row_index in range(
        int(values.shape[0])
    ):
        valid = values[row_index][
            timestep_mask[row_index]
        ]

        if valid.numel() <= 1:
            continue

        differences = valid[1:] - valid[:-1]

        if duplicate_policy == TemporalDuplicatePolicy.ALLOW_EQUAL:
            invalid = differences < 0
            message = "nondecreasing"
        else:
            invalid = differences <= 0
            message = "strictly increasing"

        if bool(
            invalid.any().item()
        ):
            raise ValueError(
                "Valid temporal coordinates must be "
                f"{message} from oldest to newest; "
                f"row {row_index} violates this contract."
            )


def _validate_absolute_regular_grid(
    values: torch.Tensor,
    timestep_mask: torch.Tensor,
    *,
    regular_step: int,
) -> None:
    for row_index in range(
        int(values.shape[0])
    ):
        valid = values[row_index][
            timestep_mask[row_index]
        ]

        if valid.numel() <= 1:
            continue

        differences = valid[1:] - valid[:-1]
        expected = torch.full_like(
            differences,
            regular_step,
        )

        if not torch.equal(
            differences,
            expected,
        ):
            raise ValueError(
                "regular_grid absolute coordinates must use exactly "
                f"regular_step={regular_step}; row {row_index} does not."
            )


def _validate_relative_regular_grid(
    values: torch.Tensor,
    timestep_mask: torch.Tensor,
    *,
    regular_step: float,
    absolute_tolerance: float,
    relative_tolerance: float,
) -> None:
    for row_index in range(
        int(values.shape[0])
    ):
        valid = values[row_index][
            timestep_mask[row_index]
        ]

        if valid.numel() <= 1:
            continue

        differences = valid[1:] - valid[:-1]
        expected = torch.full_like(
            differences,
            regular_step,
        )

        if not bool(
            torch.isclose(
                differences,
                expected,
                atol=absolute_tolerance,
                rtol=relative_tolerance,
            ).all().item()
        ):
            raise ValueError(
                "regular_grid relative coordinates must use "
                f"regular_step={regular_step}; row {row_index} does not."
            )


# =============================================================================
# Absolute temporal coordinates
# =============================================================================


@dataclass(
    slots=True,
    frozen=True,
)
class AbsoluteTemporalCoordinates:
    """
    Integer absolute coordinates aligned with ``[N, T]`` histories.

    Parameters
    ----------
    values:
        ``torch.long`` tensor with shape ``[N, T]``.

    unit:
        Explicit integer-time unit, for example ``unix_seconds``,
        ``unix_days``, or a project-defined integer calendar unit.

    calendar:
        Calendar convention used before conversion to integer coordinates.

    timezone:
        Timezone convention. ``UTC`` is the default model-facing convention.

    reference_kind:
        Meaning of ``reference_time_values``.

    reference_time_values:
        Optional ``torch.long`` tensor ``[N]``. When present, every valid
        historical coordinate must be less than or equal to its node reference
        time.

    layout:
        ``event_sequence`` or ``regular_grid``.

    regular_step:
        Required positive integer for ``regular_grid`` and forbidden for
        ``event_sequence``.
    """

    values: torch.Tensor
    unit: str

    calendar: str = "proleptic_gregorian"
    timezone: str = "UTC"

    reference_kind: (
        AbsoluteTemporalReferenceKind
        | str
    ) = AbsoluteTemporalReferenceKind.NONE
    reference_time_values: (
        torch.Tensor
        | None
    ) = None

    layout: TemporalLayout | str = (
        TemporalLayout.EVENT_SEQUENCE
    )
    chronological_order: (
        TemporalChronologicalOrder
        | str
    ) = TemporalChronologicalOrder.OLDEST_TO_NEWEST
    duplicate_policy: (
        TemporalDuplicatePolicy
        | str
    ) = TemporalDuplicatePolicy.ALLOW_EQUAL

    regular_step: int | None = None

    schema_version: str = (
        ABSOLUTE_TEMPORAL_COORDINATES_SCHEMA_VERSION
    )

    def __post_init__(
        self,
    ) -> None:
        _require_long_matrix(
            "values",
            self.values,
        )
        _require_nonempty_string(
            "unit",
            self.unit,
        )
        _require_nonempty_string(
            "calendar",
            self.calendar,
        )
        _require_nonempty_string(
            "timezone",
            self.timezone,
        )

        reference_kind = (
            _normalize_absolute_reference_kind(
                self.reference_kind
            )
        )
        layout = _normalize_temporal_layout(
            self.layout
        )
        chronological_order = (
            _normalize_chronological_order(
                self.chronological_order
            )
        )
        duplicate_policy = (
            _normalize_duplicate_policy(
                self.duplicate_policy
            )
        )

        object.__setattr__(
            self,
            "reference_kind",
            reference_kind,
        )
        object.__setattr__(
            self,
            "layout",
            layout,
        )
        object.__setattr__(
            self,
            "chronological_order",
            chronological_order,
        )
        object.__setattr__(
            self,
            "duplicate_policy",
            duplicate_policy,
        )

        if (
            chronological_order
            != TemporalChronologicalOrder.OLDEST_TO_NEWEST
        ):
            raise ValueError(
                "Only oldest_to_newest temporal order is supported."
            )

        if reference_kind == AbsoluteTemporalReferenceKind.NONE:
            if self.reference_time_values is not None:
                raise ValueError(
                    "reference_time_values must be None when "
                    "reference_kind='none'."
                )
        else:
            if self.reference_time_values is None:
                raise ValueError(
                    "reference_time_values are required when "
                    "reference_kind is not 'none'."
                )

            _require_reference_vector(
                "reference_time_values",
                self.reference_time_values,
                node_count=self.node_count,
                device=self.device,
            )

        normalized_step = _validate_layout_step(
            layout=layout,
            regular_step=self.regular_step,
            require_integer=True,
        )
        object.__setattr__(
            self,
            "regular_step",
            normalized_step,
        )

        _require_nonempty_string(
            "schema_version",
            self.schema_version,
        )

    @property
    def coordinate_kind(
        self,
    ) -> TemporalCoordinateKind:
        return TemporalCoordinateKind.ABSOLUTE

    @property
    def node_count(
        self,
    ) -> int:
        return int(
            self.values.shape[0]
        )

    @property
    def sequence_length(
        self,
    ) -> int:
        return int(
            self.values.shape[1]
        )

    @property
    def shape(
        self,
    ) -> tuple[int, int]:
        return (
            self.node_count,
            self.sequence_length,
        )

    @property
    def device(
        self,
    ) -> torch.device:
        return self.values.device

    @property
    def dtype(
        self,
    ) -> torch.dtype:
        return self.values.dtype

    def validate_against_mask(
        self,
        timestep_mask: torch.Tensor,
        *,
        padding_direction: (
            TemporalPaddingDirection
            | str
        ),
    ) -> None:
        padding = _normalize_padding_direction(
            padding_direction
        )

        _require_bool_mask(
            "timestep_mask",
            timestep_mask,
            shape=self.shape,
            device=self.device,
        )
        _validate_padding_contiguity(
            timestep_mask,
            padding_direction=padding,
        )
        _validate_canonical_padding_values(
            self.values,
            timestep_mask,
        )
        _validate_monotonicity_and_duplicates(
            self.values,
            timestep_mask,
            duplicate_policy=self.duplicate_policy,
        )

        if self.layout == TemporalLayout.REGULAR_GRID:
            assert isinstance(
                self.regular_step,
                int,
            )
            _validate_absolute_regular_grid(
                self.values,
                timestep_mask,
                regular_step=self.regular_step,
            )

        if self.reference_time_values is not None:
            for row_index in range(
                self.node_count
            ):
                valid = self.values[row_index][
                    timestep_mask[row_index]
                ]

                if valid.numel() == 0:
                    continue

                reference = self.reference_time_values[
                    row_index
                ]

                if bool(
                    (valid > reference).any().item()
                ):
                    raise ValueError(
                        "Historical absolute coordinates cannot exceed "
                        "their node reference time; "
                        f"row {row_index} violates this contract."
                    )

    def semantic_dict(
        self,
    ) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "coordinate_kind": self.coordinate_kind.value,
            "unit": self.unit,
            "calendar": self.calendar,
            "timezone": self.timezone,
            "reference_kind": self.reference_kind.value,
            "has_reference_time_values": (
                self.reference_time_values
                is not None
            ),
            "layout": self.layout.value,
            "chronological_order": (
                self.chronological_order.value
            ),
            "duplicate_policy": (
                self.duplicate_policy.value
            ),
            "regular_step": self.regular_step,
            "node_count": self.node_count,
            "sequence_length": self.sequence_length,
            "canonical_padding_value": (
                TEMPORAL_COORDINATE_CANONICAL_PADDING_VALUE
            ),
        }

    def semantic_fingerprint(
        self,
    ) -> str:
        return _fingerprint(
            self.semantic_dict()
        )

    def value_fingerprint(
        self,
    ) -> str:
        tensors = {
            "values": self.values,
        }

        if self.reference_time_values is not None:
            tensors["reference_time_values"] = (
                self.reference_time_values
            )

        return _tensor_fingerprint(
            tensors
        )

    def temporal_axis_fingerprint(
        self,
    ) -> str:
        return _fingerprint(
            {
                "semantic_fingerprint": (
                    self.semantic_fingerprint()
                ),
                "value_fingerprint": (
                    self.value_fingerprint()
                ),
            }
        )

    def fingerprint(
        self,
    ) -> str:
        return self.temporal_axis_fingerprint()

    def to(
        self,
        device: torch.device | str,
    ) -> "AbsoluteTemporalCoordinates":
        return AbsoluteTemporalCoordinates(
            values=self.values.to(
                device=device
            ),
            unit=self.unit,
            calendar=self.calendar,
            timezone=self.timezone,
            reference_kind=self.reference_kind,
            reference_time_values=(
                self.reference_time_values.to(
                    device=device
                )
                if self.reference_time_values is not None
                else None
            ),
            layout=self.layout,
            chronological_order=(
                self.chronological_order
            ),
            duplicate_policy=self.duplicate_policy,
            regular_step=self.regular_step,
            schema_version=self.schema_version,
        )


# =============================================================================
# Relative temporal coordinates
# =============================================================================


@dataclass(
    slots=True,
    frozen=True,
)
class RelativeTemporalCoordinates:
    """
    Floating temporal offsets aligned with ``[N, T]`` histories.

    By default, coordinates are anchored at prediction origin:

        prediction/reference origin = 0
        historical observations <= 0

    ``latest_observation`` is supported explicitly for compatibility or
    specialized models. When selected, every nonempty row must have latest
    valid offset exactly zero within the configured tolerance.
    """

    values: torch.Tensor
    unit: str

    anchor: RelativeTemporalAnchor | str = (
        RelativeTemporalAnchor.PREDICTION_ORIGIN
    )
    anchor_source_fingerprint: str | None = None

    layout: TemporalLayout | str = (
        TemporalLayout.EVENT_SEQUENCE
    )
    chronological_order: (
        TemporalChronologicalOrder
        | str
    ) = TemporalChronologicalOrder.OLDEST_TO_NEWEST
    duplicate_policy: (
        TemporalDuplicatePolicy
        | str
    ) = TemporalDuplicatePolicy.ALLOW_EQUAL

    regular_step: float | None = None

    require_nonpositive_history: bool = True
    absolute_tolerance: float = 1e-7
    relative_tolerance: float = 1e-6

    schema_version: str = (
        RELATIVE_TEMPORAL_COORDINATES_SCHEMA_VERSION
    )

    def __post_init__(
        self,
    ) -> None:
        _require_float_matrix(
            "values",
            self.values,
        )
        _require_nonempty_string(
            "unit",
            self.unit,
        )
        _require_boolean(
            "require_nonpositive_history",
            self.require_nonpositive_history,
        )

        anchor = _normalize_relative_anchor(
            self.anchor
        )
        layout = _normalize_temporal_layout(
            self.layout
        )
        chronological_order = (
            _normalize_chronological_order(
                self.chronological_order
            )
        )
        duplicate_policy = (
            _normalize_duplicate_policy(
                self.duplicate_policy
            )
        )

        object.__setattr__(
            self,
            "anchor",
            anchor,
        )
        object.__setattr__(
            self,
            "layout",
            layout,
        )
        object.__setattr__(
            self,
            "chronological_order",
            chronological_order,
        )
        object.__setattr__(
            self,
            "duplicate_policy",
            duplicate_policy,
        )

        if (
            chronological_order
            != TemporalChronologicalOrder.OLDEST_TO_NEWEST
        ):
            raise ValueError(
                "Only oldest_to_newest temporal order is supported."
            )

        if self.anchor_source_fingerprint is not None:
            _require_nonempty_string(
                "anchor_source_fingerprint",
                self.anchor_source_fingerprint,
            )

        normalized_step = _validate_layout_step(
            layout=layout,
            regular_step=self.regular_step,
            require_integer=False,
        )
        object.__setattr__(
            self,
            "regular_step",
            normalized_step,
        )

        absolute_tolerance = (
            _require_positive_finite_number(
                "absolute_tolerance",
                self.absolute_tolerance,
            )
        )
        relative_tolerance = (
            _require_positive_finite_number(
                "relative_tolerance",
                self.relative_tolerance,
            )
        )
        object.__setattr__(
            self,
            "absolute_tolerance",
            absolute_tolerance,
        )
        object.__setattr__(
            self,
            "relative_tolerance",
            relative_tolerance,
        )

        if self.require_nonpositive_history:
            tolerance = self.absolute_tolerance

            if bool(
                (
                    self.values
                    > tolerance
                )
                .any()
                .item()
            ):
                raise ValueError(
                    "Relative historical offsets must be nonpositive "
                    "when require_nonpositive_history=True."
                )

        _require_nonempty_string(
            "schema_version",
            self.schema_version,
        )

    @property
    def coordinate_kind(
        self,
    ) -> TemporalCoordinateKind:
        return TemporalCoordinateKind.RELATIVE

    @property
    def node_count(
        self,
    ) -> int:
        return int(
            self.values.shape[0]
        )

    @property
    def sequence_length(
        self,
    ) -> int:
        return int(
            self.values.shape[1]
        )

    @property
    def shape(
        self,
    ) -> tuple[int, int]:
        return (
            self.node_count,
            self.sequence_length,
        )

    @property
    def device(
        self,
    ) -> torch.device:
        return self.values.device

    @property
    def dtype(
        self,
    ) -> torch.dtype:
        return self.values.dtype

    def validate_against_mask(
        self,
        timestep_mask: torch.Tensor,
        *,
        padding_direction: (
            TemporalPaddingDirection
            | str
        ),
    ) -> None:
        padding = _normalize_padding_direction(
            padding_direction
        )

        _require_bool_mask(
            "timestep_mask",
            timestep_mask,
            shape=self.shape,
            device=self.device,
        )
        _validate_padding_contiguity(
            timestep_mask,
            padding_direction=padding,
        )
        _validate_canonical_padding_values(
            self.values,
            timestep_mask,
        )
        _validate_monotonicity_and_duplicates(
            self.values,
            timestep_mask,
            duplicate_policy=self.duplicate_policy,
        )

        if self.layout == TemporalLayout.REGULAR_GRID:
            assert isinstance(
                self.regular_step,
                float,
            )
            _validate_relative_regular_grid(
                self.values,
                timestep_mask,
                regular_step=self.regular_step,
                absolute_tolerance=(
                    self.absolute_tolerance
                ),
                relative_tolerance=(
                    self.relative_tolerance
                ),
            )

        if self.anchor == RelativeTemporalAnchor.LATEST_OBSERVATION:
            for row_index in range(
                self.node_count
            ):
                valid = self.values[row_index][
                    timestep_mask[row_index]
                ]

                if valid.numel() == 0:
                    continue

                latest = valid[-1]
                zero = torch.zeros_like(
                    latest
                )

                if not bool(
                    torch.isclose(
                        latest,
                        zero,
                        atol=self.absolute_tolerance,
                        rtol=self.relative_tolerance,
                    ).item()
                ):
                    raise ValueError(
                        "latest_observation anchoring requires the "
                        "latest valid offset to equal zero; "
                        f"row {row_index} violates this contract."
                    )

    def semantic_dict(
        self,
    ) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "coordinate_kind": self.coordinate_kind.value,
            "unit": self.unit,
            "anchor": self.anchor.value,
            "anchor_source_fingerprint": (
                self.anchor_source_fingerprint
            ),
            "layout": self.layout.value,
            "chronological_order": (
                self.chronological_order.value
            ),
            "duplicate_policy": (
                self.duplicate_policy.value
            ),
            "regular_step": self.regular_step,
            "require_nonpositive_history": (
                self.require_nonpositive_history
            ),
            "absolute_tolerance": (
                self.absolute_tolerance
            ),
            "relative_tolerance": (
                self.relative_tolerance
            ),
            "node_count": self.node_count,
            "sequence_length": self.sequence_length,
            "canonical_padding_value": (
                TEMPORAL_COORDINATE_CANONICAL_PADDING_VALUE
            ),
        }

    def semantic_fingerprint(
        self,
    ) -> str:
        return _fingerprint(
            self.semantic_dict()
        )

    def value_fingerprint(
        self,
    ) -> str:
        return _tensor_fingerprint(
            {
                "values": self.values,
            }
        )

    def temporal_axis_fingerprint(
        self,
    ) -> str:
        return _fingerprint(
            {
                "semantic_fingerprint": (
                    self.semantic_fingerprint()
                ),
                "value_fingerprint": (
                    self.value_fingerprint()
                ),
            }
        )

    def fingerprint(
        self,
    ) -> str:
        return self.temporal_axis_fingerprint()

    def to(
        self,
        device: torch.device | str,
    ) -> "RelativeTemporalCoordinates":
        return RelativeTemporalCoordinates(
            values=self.values.to(
                device=device
            ),
            unit=self.unit,
            anchor=self.anchor,
            anchor_source_fingerprint=(
                self.anchor_source_fingerprint
            ),
            layout=self.layout,
            chronological_order=(
                self.chronological_order
            ),
            duplicate_policy=self.duplicate_policy,
            regular_step=self.regular_step,
            require_nonpositive_history=(
                self.require_nonpositive_history
            ),
            absolute_tolerance=(
                self.absolute_tolerance
            ),
            relative_tolerance=(
                self.relative_tolerance
            ),
            schema_version=self.schema_version,
        )


# =============================================================================
# Shared union and validators
# =============================================================================


TemporalCoordinates: TypeAlias = (
    AbsoluteTemporalCoordinates
    | RelativeTemporalCoordinates
)


def validate_temporal_coordinates(
    coordinates: TemporalCoordinates,
    timestep_mask: torch.Tensor,
    *,
    padding_direction: (
        TemporalPaddingDirection
        | str
    ),
) -> None:
    """
    Validate one coordinate object against a history timestep mask.

    This function is the preferred neutral boundary for
    ``HistoricalSequenceInputs``.
    """

    if not isinstance(
        coordinates,
        (
            AbsoluteTemporalCoordinates,
            RelativeTemporalCoordinates,
        ),
    ):
        raise TypeError(
            "coordinates must be AbsoluteTemporalCoordinates or "
            "RelativeTemporalCoordinates."
        )

    coordinates.validate_against_mask(
        timestep_mask,
        padding_direction=padding_direction,
    )


def temporal_coordinates_fingerprint(
    coordinates: TemporalCoordinates,
) -> str:
    """Return the deterministic temporal-axis fingerprint."""

    if not isinstance(
        coordinates,
        (
            AbsoluteTemporalCoordinates,
            RelativeTemporalCoordinates,
        ),
    ):
        raise TypeError(
            "coordinates must be AbsoluteTemporalCoordinates or "
            "RelativeTemporalCoordinates."
        )

    return (
        coordinates
        .temporal_axis_fingerprint()
    )


# =============================================================================
# Public API
# =============================================================================


__all__ = (
    # Schema identity.
    "ABSOLUTE_TEMPORAL_COORDINATES_SCHEMA_VERSION",
    "RELATIVE_TEMPORAL_COORDINATES_SCHEMA_VERSION",
    "TEMPORAL_COORDINATE_CANONICAL_PADDING_VALUE",
    "TEMPORAL_COORDINATE_SCIENTIFIC_INTERPRETATION",

    # Controlled vocabularies.
    "TemporalCoordinateKind",
    "TemporalLayout",
    "TemporalChronologicalOrder",
    "TemporalPaddingDirection",
    "TemporalDuplicatePolicy",
    "AbsoluteTemporalReferenceKind",
    "RelativeTemporalAnchor",
    "CANONICAL_TEMPORAL_COORDINATE_KINDS",
    "CANONICAL_TEMPORAL_LAYOUTS",
    "CANONICAL_TEMPORAL_CHRONOLOGICAL_ORDERS",
    "CANONICAL_TEMPORAL_PADDING_DIRECTIONS",
    "CANONICAL_TEMPORAL_DUPLICATE_POLICIES",
    "CANONICAL_ABSOLUTE_TEMPORAL_REFERENCE_KINDS",
    "CANONICAL_RELATIVE_TEMPORAL_ANCHORS",

    # Coordinate contracts.
    "AbsoluteTemporalCoordinates",
    "RelativeTemporalCoordinates",
    "TemporalCoordinates",

    # Neutral helpers.
    "validate_temporal_coordinates",
    "temporal_coordinates_fingerprint",
)
