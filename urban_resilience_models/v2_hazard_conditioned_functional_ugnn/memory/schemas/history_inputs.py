"""
Metadata-preserving historical-sequence inputs for shared temporal memory.

Target path:
    urban_resilience_models/
        v2_hazard_conditioned_functional_ugnn/
            memory/
                schemas/
                    history_inputs.py

This module freezes the model-facing historical input boundary used by
baseline, recurrent, Transformer, pooling, and hazard-query memory modules.

The authoritative tensor contracts are:

    history:                  [N, T, D], floating and finite
    timestep_mask:            [N, T], Boolean
    feature_observed_mask:    [N, T, D], Boolean or absent

where:

- ``N`` is the node/item axis;
- ``T`` is the packed temporal axis;
- ``D`` is the ordered historical feature axis.

Mask semantics
--------------
``timestep_mask[n, t]`` means that temporal position ``t`` belongs to node
``n``'s declared history. It does not mean that at least one feature was
observed.

``feature_observed_mask[n, t, d]`` means that feature ``d`` was observed at
that declared temporal position before any model-facing imputation or finite
placeholder policy.

A real timestep may therefore satisfy:

    timestep_mask[n, t] = True
    feature_observed_mask[n, t, :] = False

This preserves the fact that time passed even when every feature was missing.
The timestep mask must never be derived from
``feature_observed_mask.any(dim=-1)``.

Value policy
------------
All model-facing history values must be finite. Raw NaN and infinite values
must be handled upstream. Missingness is represented by explicit policy and,
when available or required, ``feature_observed_mask``.

Padding is canonical:

    history at padded positions = exactly zero
    feature_observed_mask at padded positions = False
    temporal coordinates at padded positions = exactly zero

Canonical padding prevents hidden information from being stored in masked
positions and gives downstream tests a deterministic leakage boundary.

Alignment
---------
The schema preserves complete neutral identities for:

- node IDs and packed graph membership;
- ordered feature names;
- absolute timestamps or relative temporal offsets;
- source-data and preprocessing provenance.

The memory schema does not import ``fusion.schemas.NodeAlignment``. Fusion
currently depends on memory contracts, so importing fusion here would risk:

    memory -> fusion -> memory

Zero-history policy
-------------------
Rows with no valid temporal positions are rejected by default. They may be
retained only through the explicit ``allow_zero_history`` policy and only with
left or right padding. Downstream pooling and retrieval modules must then
apply an explicit cold-start policy; they may not pretend that an all-masked
row has normalized temporal weights.

Interpretation
--------------------------
These contracts preserve model-facing data identity, temporal ordering,
missingness metadata, and software lineage. They do not prove that the data
are causally valid, leakage-free, correctly imputed, or representative.
"""

from __future__ import annotations

from dataclasses import dataclass
from dataclasses import replace as dataclass_replace
from enum import StrEnum
from hashlib import sha256
import json
from typing import Any, Final, Mapping, Self

import torch

from .provenance import (
    MemorySourceProvenance,
    TemporalFeatureAxis,
    TemporalNodeAxis,
)
from .temporal_coordinates import (
    AbsoluteTemporalCoordinates,
    RelativeTemporalCoordinates,
    TemporalCoordinates,
    TemporalPaddingDirection,
    temporal_coordinates_fingerprint,
    validate_temporal_coordinates,
)


# =============================================================================
# Schema identity and constants
# =============================================================================


HISTORICAL_SEQUENCE_INPUTS_SCHEMA_VERSION: Final[str] = "0.1"

HISTORY_VALUE_SEMANTICS: Final[str] = (
    "finite_model_ready_values"
)

TIMESTEP_MASK_SEMANTICS: Final[str] = (
    "declared_temporal_position_not_padding"
)

FEATURE_OBSERVED_MASK_SEMANTICS: Final[str] = (
    "feature_observed_before_model_facing_imputation_or_placeholder"
)

HISTORY_CANONICAL_PADDING_VALUE: Final[int | float] = 0

HISTORY_INPUT_SCIENTIFIC_INTERPRETATION: Final[str] = (
    "model_facing_history_contract_not_causal_or_leakage_proof"
)


# =============================================================================
# Controlled input-policy vocabularies
# =============================================================================


class HistoryMissingValuePolicy(StrEnum):
    """
    Model-facing missing-value policy for historical features.

    ``complete``
        Every feature at every valid timestep is declared observed.
        ``feature_observed_mask`` may be absent or all-true over valid
        timesteps.

    ``upstream_imputed``
        Values are finite and model-ready after an upstream imputation
        procedure. ``feature_observed_mask`` is optional. When present, it
        preserves pre-imputation observation status.

    ``finite_placeholder_with_mask``
        Missing values use finite placeholders and
        ``feature_observed_mask`` is required to distinguish observed from
        missing features.
    """

    COMPLETE = "complete"
    UPSTREAM_IMPUTED = "upstream_imputed"
    FINITE_PLACEHOLDER_WITH_MASK = (
        "finite_placeholder_with_mask"
    )


class HistoryZeroLengthPolicy(StrEnum):
    """Whether nodes with no valid historical timestep are accepted."""

    ERROR = "error"
    ALLOW_ZERO_HISTORY = "allow_zero_history"


CANONICAL_HISTORY_MISSING_VALUE_POLICIES: Final[
    tuple[str, ...]
] = tuple(
    value.value
    for value in HistoryMissingValuePolicy
)

CANONICAL_HISTORY_ZERO_LENGTH_POLICIES: Final[
    tuple[str, ...]
] = tuple(
    value.value
    for value in HistoryZeroLengthPolicy
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


def _normalize_missing_value_policy(
    value: HistoryMissingValuePolicy | str,
) -> HistoryMissingValuePolicy:
    if isinstance(
        value,
        HistoryMissingValuePolicy,
    ):
        return value

    return HistoryMissingValuePolicy(
        value
    )


def _normalize_zero_length_policy(
    value: HistoryZeroLengthPolicy | str,
) -> HistoryZeroLengthPolicy:
    if isinstance(
        value,
        HistoryZeroLengthPolicy,
    ):
        return value

    return HistoryZeroLengthPolicy(
        value
    )


def _normalize_padding_direction(
    value: TemporalPaddingDirection | str,
) -> TemporalPaddingDirection:
    if isinstance(
        value,
        TemporalPaddingDirection,
    ):
        return value

    return TemporalPaddingDirection(
        value
    )


def _require_history_tensor(
    value: torch.Tensor,
) -> None:
    if not isinstance(
        value,
        torch.Tensor,
    ):
        raise TypeError(
            "history must be a tensor."
        )

    if value.ndim != 3:
        raise ValueError(
            "history must have shape [N, T, D]; "
            f"observed {tuple(value.shape)}."
        )

    node_count = int(
        value.shape[0]
    )
    sequence_length = int(
        value.shape[1]
    )
    feature_dim = int(
        value.shape[2]
    )

    if (
        node_count <= 0
        or sequence_length <= 0
        or feature_dim <= 0
    ):
        raise ValueError(
            "history dimensions N, T, and D must all be "
            "strictly positive."
        )

    if not value.dtype.is_floating_point:
        raise ValueError(
            "history must use a floating-point dtype."
        )

    if not bool(
        torch.isfinite(
            value
        ).all().item()
    ):
        raise ValueError(
            "history must contain only finite model-facing values."
        )


def _require_timestep_mask(
    value: torch.Tensor,
    *,
    shape: tuple[int, int],
    device: torch.device,
) -> None:
    if not isinstance(
        value,
        torch.Tensor,
    ):
        raise TypeError(
            "timestep_mask must be a tensor."
        )

    if value.dtype != torch.bool:
        raise ValueError(
            "timestep_mask must use torch.bool."
        )

    if tuple(
        value.shape
    ) != shape:
        raise ValueError(
            "timestep_mask must have shape "
            f"{shape}; observed {tuple(value.shape)}."
        )

    if value.device != device:
        raise ValueError(
            "history and timestep_mask must share one device."
        )


def _require_feature_observed_mask(
    value: torch.Tensor,
    *,
    shape: tuple[int, int, int],
    device: torch.device,
) -> None:
    if not isinstance(
        value,
        torch.Tensor,
    ):
        raise TypeError(
            "feature_observed_mask must be a tensor."
        )

    if value.dtype != torch.bool:
        raise ValueError(
            "feature_observed_mask must use torch.bool."
        )

    if tuple(
        value.shape
    ) != shape:
        raise ValueError(
            "feature_observed_mask must have shape "
            f"{shape}; observed {tuple(value.shape)}."
        )

    if value.device != device:
        raise ValueError(
            "history and feature_observed_mask must share one device."
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
        _canonical_json(
            payload
        ).encode(
            "utf-8"
        )
    ).hexdigest()


def _tensor_fingerprint(
    tensors: Mapping[
        str,
        torch.Tensor,
    ],
) -> str:
    digest = sha256()

    for name in sorted(
        tensors
    ):
        tensor = (
            tensors[name]
            .detach()
            .cpu()
            .contiguous()
        )

        digest.update(
            name.encode(
                "utf-8"
            )
        )
        digest.update(
            str(
                tensor.dtype
            ).encode(
                "utf-8"
            )
        )
        digest.update(
            json.dumps(
                list(
                    tensor.shape
                ),
                separators=(
                    ",",
                    ":",
                ),
            ).encode(
                "utf-8"
            )
        )
        digest.update(
            tensor.view(
                torch.uint8
            )
            .numpy()
            .tobytes()
        )

    return digest.hexdigest()


def _validate_canonical_history_padding(
    history: torch.Tensor,
    timestep_mask: torch.Tensor,
) -> None:
    padding_mask = (
        ~timestep_mask
    ).unsqueeze(
        -1
    ).expand_as(
        history
    )

    padded_values = history[
        padding_mask
    ]

    if padded_values.numel() == 0:
        return

    if not torch.equal(
        padded_values,
        torch.zeros_like(
            padded_values
        ),
    ):
        raise ValueError(
            "history values at padded timesteps must equal the "
            "canonical padding value zero."
        )


def _validate_feature_mask_padding(
    feature_observed_mask: torch.Tensor,
    timestep_mask: torch.Tensor,
) -> None:
    padding_mask = (
        ~timestep_mask
    ).unsqueeze(
        -1
    ).expand_as(
        feature_observed_mask
    )

    if bool(
        feature_observed_mask[
            padding_mask
        ].any().item()
    ):
        raise ValueError(
            "feature_observed_mask must be False at padded timesteps."
        )


def _validate_missingness_policy(
    *,
    missing_value_policy: HistoryMissingValuePolicy,
    timestep_mask: torch.Tensor,
    feature_observed_mask: torch.Tensor | None,
) -> None:
    if (
        missing_value_policy
        == HistoryMissingValuePolicy.FINITE_PLACEHOLDER_WITH_MASK
        and feature_observed_mask is None
    ):
        raise ValueError(
            "finite_placeholder_with_mask requires "
            "feature_observed_mask."
        )

    if (
        missing_value_policy
        == HistoryMissingValuePolicy.COMPLETE
        and feature_observed_mask is not None
    ):
        valid_features = timestep_mask.unsqueeze(
            -1
        ).expand_as(
            feature_observed_mask
        )

        if not bool(
            feature_observed_mask[
                valid_features
            ].all().item()
        ):
            raise ValueError(
                "missing_value_policy='complete' requires every "
                "feature at every valid timestep to be observed."
            )


def _validate_zero_history_policy(
    *,
    timestep_mask: torch.Tensor,
    zero_length_policy: HistoryZeroLengthPolicy,
    padding_direction: TemporalPaddingDirection,
) -> None:
    valid_lengths = timestep_mask.sum(
        dim=1,
        dtype=torch.long,
    )
    zero_rows = valid_lengths == 0

    if not bool(
        zero_rows.any().item()
    ):
        return

    if (
        zero_length_policy
        == HistoryZeroLengthPolicy.ERROR
    ):
        indices = torch.nonzero(
            zero_rows,
            as_tuple=False,
        ).flatten().detach().cpu().tolist()

        raise ValueError(
            "Every node must contain at least one valid historical "
            "timestep unless allow_zero_history is selected. "
            f"Zero-history rows: {indices}."
        )

    if (
        padding_direction
        == TemporalPaddingDirection.NONE
    ):
        raise ValueError(
            "allow_zero_history is incompatible with "
            "padding_direction='none' because an all-masked row is "
            "entirely padded."
        )


def _validate_shared_device(
    *,
    history: torch.Tensor,
    node_axis: TemporalNodeAxis,
    temporal_coordinates: TemporalCoordinates,
) -> None:
    if node_axis.device != history.device:
        raise ValueError(
            "history and node_axis must share one device."
        )

    if temporal_coordinates.device != history.device:
        raise ValueError(
            "history and temporal_coordinates must share one device."
        )


def _validate_axis_alignment(
    *,
    history_shape: tuple[int, int, int],
    node_axis: TemporalNodeAxis,
    feature_axis: TemporalFeatureAxis,
    temporal_coordinates: TemporalCoordinates,
) -> None:
    node_count, sequence_length, feature_dim = (
        history_shape
    )

    if node_axis.node_count != node_count:
        raise ValueError(
            "node_axis must align with history dimension N."
        )

    if feature_axis.feature_dim != feature_dim:
        raise ValueError(
            "feature_axis must align with history dimension D."
        )

    if temporal_coordinates.shape != (
        node_count,
        sequence_length,
    ):
        raise ValueError(
            "temporal_coordinates must align with history "
            "dimensions [N, T]."
        )


def _validate_dtype_request(
    dtype: torch.dtype | None,
) -> None:
    if dtype is None:
        return

    if not isinstance(
        dtype,
        torch.dtype,
    ):
        raise TypeError(
            "dtype must be a torch.dtype or None."
        )

    if not dtype.is_floating_point:
        raise ValueError(
            "history dtype must remain floating-point."
        )


# =============================================================================
# Historical input contract
# =============================================================================


@dataclass(
    slots=True,
    frozen=True,
)
class HistoricalSequenceInputs:
    """
    Complete metadata-preserving historical input for temporal memory.

    Parameters
    ----------
    history:
        Finite floating tensor with shape ``[N, T, D]``.

    timestep_mask:
        Boolean tensor ``[N, T]``. ``True`` means that a temporal slot belongs
        to the declared sequence. It does not imply that any feature was
        observed.

    node_axis:
        Stable node IDs and packed-graph membership aligned with ``N``.

    feature_axis:
        Ordered historical feature identities aligned with ``D``.

    temporal_coordinates:
        Absolute timestamps or relative temporal offsets aligned with
        ``[N, T]``.

    source_provenance:
        Identity of the upstream data and preprocessing pipeline.

    feature_observed_mask:
        Optional Boolean tensor ``[N, T, D]`` preserving pre-imputation
        feature observation status.

    padding_direction:
        Left, right, or no padding. Valid timestep positions must be
        contiguous according to this declaration.

    missing_value_policy:
        Explicit model-facing missingness contract.

    zero_length_policy:
        Whether all-masked node histories are rejected or preserved for an
        explicit downstream cold-start policy.
    """

    history: torch.Tensor
    timestep_mask: torch.Tensor

    node_axis: TemporalNodeAxis
    feature_axis: TemporalFeatureAxis
    temporal_coordinates: TemporalCoordinates
    source_provenance: MemorySourceProvenance

    feature_observed_mask: (
        torch.Tensor
        | None
    ) = None

    padding_direction: (
        TemporalPaddingDirection
        | str
    ) = TemporalPaddingDirection.RIGHT

    missing_value_policy: (
        HistoryMissingValuePolicy
        | str
    ) = HistoryMissingValuePolicy.UPSTREAM_IMPUTED

    zero_length_policy: (
        HistoryZeroLengthPolicy
        | str
    ) = HistoryZeroLengthPolicy.ERROR

    history_name: str = (
        "historical_sequence_inputs"
    )

    schema_version: str = (
        HISTORICAL_SEQUENCE_INPUTS_SCHEMA_VERSION
    )

    def __post_init__(
        self,
    ) -> None:
        _require_history_tensor(
            self.history
        )

        shape = (
            self.node_count,
            self.sequence_length,
        )

        _require_timestep_mask(
            self.timestep_mask,
            shape=shape,
            device=self.device,
        )

        if not isinstance(
            self.node_axis,
            TemporalNodeAxis,
        ):
            raise TypeError(
                "node_axis must be a TemporalNodeAxis."
            )

        if not isinstance(
            self.feature_axis,
            TemporalFeatureAxis,
        ):
            raise TypeError(
                "feature_axis must be a TemporalFeatureAxis."
            )

        if not isinstance(
            self.temporal_coordinates,
            (
                AbsoluteTemporalCoordinates,
                RelativeTemporalCoordinates,
            ),
        ):
            raise TypeError(
                "temporal_coordinates must be "
                "AbsoluteTemporalCoordinates or "
                "RelativeTemporalCoordinates."
            )

        if not isinstance(
            self.source_provenance,
            MemorySourceProvenance,
        ):
            raise TypeError(
                "source_provenance must be a "
                "MemorySourceProvenance."
            )

        padding_direction = (
            _normalize_padding_direction(
                self.padding_direction
            )
        )
        missing_value_policy = (
            _normalize_missing_value_policy(
                self.missing_value_policy
            )
        )
        zero_length_policy = (
            _normalize_zero_length_policy(
                self.zero_length_policy
            )
        )

        object.__setattr__(
            self,
            "padding_direction",
            padding_direction,
        )
        object.__setattr__(
            self,
            "missing_value_policy",
            missing_value_policy,
        )
        object.__setattr__(
            self,
            "zero_length_policy",
            zero_length_policy,
        )

        if self.feature_observed_mask is not None:
            _require_feature_observed_mask(
                self.feature_observed_mask,
                shape=self.history_shape,
                device=self.device,
            )
            _validate_feature_mask_padding(
                self.feature_observed_mask,
                self.timestep_mask,
            )

        _validate_axis_alignment(
            history_shape=self.history_shape,
            node_axis=self.node_axis,
            feature_axis=self.feature_axis,
            temporal_coordinates=(
                self.temporal_coordinates
            ),
        )
        _validate_shared_device(
            history=self.history,
            node_axis=self.node_axis,
            temporal_coordinates=(
                self.temporal_coordinates
            ),
        )

        validate_temporal_coordinates(
            self.temporal_coordinates,
            self.timestep_mask,
            padding_direction=(
                padding_direction
            ),
        )

        _validate_canonical_history_padding(
            self.history,
            self.timestep_mask,
        )
        _validate_missingness_policy(
            missing_value_policy=(
                missing_value_policy
            ),
            timestep_mask=(
                self.timestep_mask
            ),
            feature_observed_mask=(
                self.feature_observed_mask
            ),
        )
        _validate_zero_history_policy(
            timestep_mask=(
                self.timestep_mask
            ),
            zero_length_policy=(
                zero_length_policy
            ),
            padding_direction=(
                padding_direction
            ),
        )

        _require_nonempty_string(
            "history_name",
            self.history_name,
        )
        _require_nonempty_string(
            "schema_version",
            self.schema_version,
        )

    # -------------------------------------------------------------------------
    # Structural properties
    # -------------------------------------------------------------------------

    @property
    def node_count(
        self,
    ) -> int:
        return int(
            self.history.shape[0]
        )

    @property
    def item_count(
        self,
    ) -> int:
        return self.node_count

    @property
    def sequence_length(
        self,
    ) -> int:
        return int(
            self.history.shape[1]
        )

    @property
    def feature_dim(
        self,
    ) -> int:
        return int(
            self.history.shape[2]
        )

    @property
    def history_shape(
        self,
    ) -> tuple[int, int, int]:
        return (
            self.node_count,
            self.sequence_length,
            self.feature_dim,
        )

    @property
    def device(
        self,
    ) -> torch.device:
        return self.history.device

    @property
    def dtype(
        self,
    ) -> torch.dtype:
        return self.history.dtype

    @property
    def node_ids(
        self,
    ) -> tuple[str, ...]:
        return self.node_axis.node_ids

    @property
    def node_batch_index(
        self,
    ) -> torch.Tensor:
        return self.node_axis.node_batch_index

    @property
    def graph_count(
        self,
    ) -> int:
        return self.node_axis.graph_count

    @property
    def graph_ids(
        self,
    ) -> tuple[str, ...]:
        return self.node_axis.graph_ids

    @property
    def feature_names(
        self,
    ) -> tuple[str, ...]:
        return self.feature_axis.feature_names

    @property
    def source_fingerprint(
        self,
    ) -> str:
        return (
            self
            .source_provenance
            .source_fingerprint
        )

    @property
    def has_feature_observation_mask(
        self,
    ) -> bool:
        return (
            self.feature_observed_mask
            is not None
        )

    @property
    def has_zero_history(
        self,
    ) -> bool:
        return bool(
            (
                self.valid_lengths
                == 0
            ).any().item()
        )

    @property
    def valid_lengths(
        self,
    ) -> torch.Tensor:
        return self.timestep_mask.sum(
            dim=1,
            dtype=torch.long,
        )

    @property
    def first_valid_indices(
        self,
    ) -> torch.Tensor:
        lengths = self.valid_lengths
        candidate = (
            self
            .timestep_mask
            .to(
                dtype=torch.long
            )
            .argmax(
                dim=1
            )
        )
        missing = torch.full_like(
            candidate,
            -1,
        )

        return torch.where(
            lengths > 0,
            candidate,
            missing,
        )

    @property
    def last_valid_indices(
        self,
    ) -> torch.Tensor:
        lengths = self.valid_lengths
        reversed_candidate = (
            torch.flip(
                self.timestep_mask,
                dims=(1,),
            )
            .to(
                dtype=torch.long
            )
            .argmax(
                dim=1
            )
        )
        candidate = (
            self.sequence_length
            - 1
            - reversed_candidate
        )
        missing = torch.full_like(
            candidate,
            -1,
        )

        return torch.where(
            lengths > 0,
            candidate,
            missing,
        )

    # -------------------------------------------------------------------------
    # Deterministic identities
    # -------------------------------------------------------------------------

    def timestep_mask_fingerprint(
        self,
    ) -> str:
        return _tensor_fingerprint(
            {
                "timestep_mask": (
                    self.timestep_mask
                ),
            }
        )

    def feature_observation_fingerprint(
        self,
    ) -> str | None:
        if (
            self.feature_observed_mask
            is None
        ):
            return None

        return _tensor_fingerprint(
            {
                "feature_observed_mask": (
                    self
                    .feature_observed_mask
                ),
            }
        )

    def temporal_alignment_fingerprint(
        self,
    ) -> str:
        return _fingerprint(
            {
                "temporal_coordinates_fingerprint": (
                    temporal_coordinates_fingerprint(
                        self
                        .temporal_coordinates
                    )
                ),
                "timestep_mask_fingerprint": (
                    self
                    .timestep_mask_fingerprint()
                ),
                "padding_direction": (
                    self
                    .padding_direction
                    .value
                ),
                "zero_length_policy": (
                    self
                    .zero_length_policy
                    .value
                ),
            }
        )

    def alignment_fingerprint(
        self,
    ) -> str:
        return _fingerprint(
            {
                "node_axis_fingerprint": (
                    self
                    .node_axis
                    .fingerprint()
                ),
                "temporal_alignment_fingerprint": (
                    self
                    .temporal_alignment_fingerprint()
                ),
                "feature_axis_fingerprint": (
                    self
                    .feature_axis
                    .fingerprint()
                ),
            }
        )

    def semantic_dict(
        self,
    ) -> dict[str, Any]:
        return {
            "schema_version": (
                self.schema_version
            ),
            "history_name": (
                self.history_name
            ),
            "history_shape": list(
                self.history_shape
            ),
            "history_value_semantics": (
                HISTORY_VALUE_SEMANTICS
            ),
            "timestep_mask_semantics": (
                TIMESTEP_MASK_SEMANTICS
            ),
            "feature_observed_mask_semantics": (
                FEATURE_OBSERVED_MASK_SEMANTICS
            ),
            "has_feature_observation_mask": (
                self
                .has_feature_observation_mask
            ),
            "padding_direction": (
                self
                .padding_direction
                .value
            ),
            "missing_value_policy": (
                self
                .missing_value_policy
                .value
            ),
            "zero_length_policy": (
                self
                .zero_length_policy
                .value
            ),
            "canonical_padding_value": (
                HISTORY_CANONICAL_PADDING_VALUE
            ),
            "node_axis_fingerprint": (
                self
                .node_axis
                .fingerprint()
            ),
            "feature_axis_fingerprint": (
                self
                .feature_axis
                .fingerprint()
            ),
            "temporal_coordinates_fingerprint": (
                temporal_coordinates_fingerprint(
                    self
                    .temporal_coordinates
                )
            ),
            "source_provenance_fingerprint": (
                self
                .source_provenance
                .fingerprint()
            ),
            "scientific_interpretation": (
                HISTORY_INPUT_SCIENTIFIC_INTERPRETATION
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
        tensors: dict[
            str,
            torch.Tensor,
        ] = {
            "history": (
                self.history
            ),
            "timestep_mask": (
                self.timestep_mask
            ),
        }

        if (
            self.feature_observed_mask
            is not None
        ):
            tensors[
                "feature_observed_mask"
            ] = self.feature_observed_mask

        return _tensor_fingerprint(
            tensors
        )

    def lineage_dict(
        self,
    ) -> dict[str, Any]:
        return {
            "schema_version": (
                self.schema_version
            ),
            "source_provenance_fingerprint": (
                self
                .source_provenance
                .fingerprint()
            ),
            "alignment_fingerprint": (
                self
                .alignment_fingerprint()
            ),
            "semantic_fingerprint": (
                self
                .semantic_fingerprint()
            ),
            "value_fingerprint": (
                self
                .value_fingerprint()
            ),
            "feature_observation_fingerprint": (
                self
                .feature_observation_fingerprint()
            ),
        }

    def lineage_fingerprint(
        self,
    ) -> str:
        return _fingerprint(
            self.lineage_dict()
        )

    def fingerprint(
        self,
    ) -> str:
        return self.lineage_fingerprint()

    # -------------------------------------------------------------------------
    # Validated reconstruction
    # -------------------------------------------------------------------------

    def to(
        self,
        device: torch.device | str,
        *,
        dtype: torch.dtype | None = None,
        non_blocking: bool = False,
    ) -> Self:
        _validate_dtype_request(
            dtype
        )

        target_dtype = (
            self.dtype
            if dtype is None
            else dtype
        )

        return type(self)(
            history=self.history.to(
                device=device,
                dtype=target_dtype,
                non_blocking=non_blocking,
            ),
            timestep_mask=(
                self
                .timestep_mask
                .to(
                    device=device,
                    non_blocking=(
                        non_blocking
                    ),
                )
            ),
            node_axis=(
                self
                .node_axis
                .to(
                    device
                )
            ),
            feature_axis=(
                self.feature_axis
            ),
            temporal_coordinates=(
                self
                .temporal_coordinates
                .to(
                    device
                )
            ),
            source_provenance=(
                self
                .source_provenance
            ),
            feature_observed_mask=(
                self
                .feature_observed_mask
                .to(
                    device=device,
                    non_blocking=(
                        non_blocking
                    ),
                )
                if (
                    self
                    .feature_observed_mask
                    is not None
                )
                else None
            ),
            padding_direction=(
                self.padding_direction
            ),
            missing_value_policy=(
                self
                .missing_value_policy
            ),
            zero_length_policy=(
                self
                .zero_length_policy
            ),
            history_name=(
                self.history_name
            ),
            schema_version=(
                self.schema_version
            ),
        )

    def replace(
        self,
        **changes: Any,
    ) -> Self:
        """
        Reconstruct this schema with validated field changes.

        Direct field mutation is prohibited by the frozen dataclass.
        """

        return dataclass_replace(
            self,
            **changes,
        )


# =============================================================================
# Compact aliases
# =============================================================================


TemporalHistoryInputs = HistoricalSequenceInputs
HistoryInputs = HistoricalSequenceInputs


# =============================================================================
# Public API
# =============================================================================


__all__ = (
    # Schema identity and fixed semantics.
    "HISTORICAL_SEQUENCE_INPUTS_SCHEMA_VERSION",
    "HISTORY_VALUE_SEMANTICS",
    "TIMESTEP_MASK_SEMANTICS",
    "FEATURE_OBSERVED_MASK_SEMANTICS",
    "HISTORY_CANONICAL_PADDING_VALUE",
    "HISTORY_INPUT_SCIENTIFIC_INTERPRETATION",

    # Controlled input policies.
    "HistoryMissingValuePolicy",
    "HistoryZeroLengthPolicy",
    "CANONICAL_HISTORY_MISSING_VALUE_POLICIES",
    "CANONICAL_HISTORY_ZERO_LENGTH_POLICIES",

    # Main contract and compact aliases.
    "HistoricalSequenceInputs",
    "TemporalHistoryInputs",
    "HistoryInputs",
)
