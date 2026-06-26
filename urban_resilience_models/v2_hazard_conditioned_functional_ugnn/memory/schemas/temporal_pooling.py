"""
Shared temporal-pooling output contract for urban memory.

Target path:
    urban_resilience_models/
        v2_hazard_conditioned_functional_ugnn/
            memory/
                schemas/
                    temporal_pooling.py

This module freezes the public output contract for hazard-independent temporal
pooling over a ``TemporalSequenceEncoding``.

The authoritative tensors are:

    pooled_memory:       [N, P]
    pooling_weights:     [N, A, T]

where:

- ``N`` is the preserved node/item axis;
- ``P`` is the pooled output dimension;
- ``A`` is the number of temporal pooling heads;
- ``T`` is the preserved temporal axis.

The exact ``TemporalSequenceEncoding`` object is retained. Pooling therefore
cannot silently detach the pooled state from:

- the encoded sequence ``[N, T, H]``;
- the timestep mask ``[N, T]``;
- node IDs and packed-graph membership;
- temporal coordinates;
- source-history provenance;
- encoder architecture and lineage.

Pooling semantics
-----------------
Generic temporal pooling is hazard-independent. It may still be node-specific
or learned. Hazard-conditioned temporal retrieval belongs in the separate
query-retrieval and hazard-queried-memory contracts.

Every pooling head must assign:

- exactly zero mass to padded timesteps;
- nonnegative finite mass to valid timesteps;
- total mass one for every node with nonempty history;
- total mass zero for every zero-history node.

The schema uses configurable numerical tolerances for normalized weight checks.
The stored weights remain the exact model outputs; the schema does not
renormalize or clamp them.

Zero-history behavior
---------------------
A zero-history node may occur only when the source history explicitly permits
it. The pooling output must declare one policy:

``error``
    Reject zero-history rows.

``zero``
    Keep all temporal weights and the pooled memory row exactly zero.

``learned_fallback``
    Keep temporal weights exactly zero but permit a finite learned or otherwise
    declared fallback vector in ``pooled_memory``.

Head reduction
--------------
``pooling_weights`` always preserves the per-head temporal weights. The final
``pooled_memory`` is two-dimensional regardless of the number of heads.

The declared head-reduction policy records how head-specific summaries were
combined:

- ``single_head``;
- ``mean``;
- ``weighted_mean``;
- ``concat_projection``;
- ``other``.

For ``weighted_mean``, normalized nonnegative ``head_reduction_weights [A]``
are required. They describe reduction across pooling heads, not temporal
weights.

Interpretability safeguard
---------------------------
Pooling weights may be described as:

- temporal attention mass;
- pooling weight;
- model-assigned temporal relevance.

They do not by themselves establish causal historical importance, real-world
causal effects, or mechanistic explanation.
"""

from __future__ import annotations

from dataclasses import dataclass
from dataclasses import replace as dataclass_replace
from enum import StrEnum
from hashlib import sha256
import json
import math
from typing import Any, Final, Mapping, Self

import torch

from .provenance import (
    MemoryComputationProvenance,
)
from .sequence_encoding import (
    TemporalSequenceEncoding,
)


# =============================================================================
# Schema identity and fixed interpretation
# =============================================================================


TEMPORAL_POOLING_OUTPUT_SCHEMA_VERSION: Final[str] = "0.1"

TEMPORAL_POOLING_WEIGHT_SEMANTICS: Final[str] = (
    "model_assigned_temporal_relevance_not_causal_importance"
)

TEMPORAL_POOLING_PADDING_POLICY: Final[str] = (
    "exact_zero_mass_at_padded_positions"
)

TEMPORAL_POOLING_NORMALIZATION_POLICY: Final[str] = (
    "unit_mass_per_nonempty_node_and_head_zero_mass_for_zero_history"
)

TEMPORAL_POOLING_SCIENTIFIC_INTERPRETATION: Final[str] = (
    "hazard_independent_model_pooling_not_causal_explanation"
)


# =============================================================================
# Controlled pooling vocabularies
# =============================================================================


class TemporalPoolingKind(StrEnum):
    """Semantic family of one hazard-independent temporal pooling operation."""

    LAST_VALID = "last_valid"
    MASKED_MEAN = "masked_mean"
    MASKED_MAX = "masked_max"
    LEARNED_ATTENTION = "learned_attention"
    MULTIHEAD_ATTENTION = "multihead_attention"
    OTHER = "other"


class TemporalPoolingHeadReduction(StrEnum):
    """How head-specific temporal summaries form ``pooled_memory``."""

    SINGLE_HEAD = "single_head"
    MEAN = "mean"
    WEIGHTED_MEAN = "weighted_mean"
    CONCAT_PROJECTION = "concat_projection"
    OTHER = "other"


class TemporalPoolingZeroHistoryPolicy(StrEnum):
    """How pooling handles a node with no valid historical timestep."""

    ERROR = "error"
    ZERO = "zero"
    LEARNED_FALLBACK = "learned_fallback"


CANONICAL_TEMPORAL_POOLING_KINDS: Final[tuple[str, ...]] = tuple(
    value.value
    for value in TemporalPoolingKind
)

CANONICAL_TEMPORAL_POOLING_HEAD_REDUCTIONS: Final[
    tuple[str, ...]
] = tuple(
    value.value
    for value in TemporalPoolingHeadReduction
)

CANONICAL_TEMPORAL_POOLING_ZERO_HISTORY_POLICIES: Final[
    tuple[str, ...]
] = tuple(
    value.value
    for value in TemporalPoolingZeroHistoryPolicy
)


# =============================================================================
# Generic helpers
# =============================================================================


def _require_nonempty_string(
    name: str,
    value: str,
) -> None:
    if not isinstance(
        value,
        str,
    ):
        raise TypeError(
            f"{name} must be a string."
        )

    if not value.strip():
        raise ValueError(
            f"{name} must be a non-empty string."
        )


def _require_positive_finite_number(
    name: str,
    value: int | float,
) -> float:
    if (
        isinstance(
            value,
            bool,
        )
        or not isinstance(
            value,
            (
                int,
                float,
            ),
        )
    ):
        raise TypeError(
            f"{name} must be numeric."
        )

    converted = float(
        value
    )

    if not math.isfinite(
        converted
    ):
        raise ValueError(
            f"{name} must be finite."
        )

    if converted <= 0.0:
        raise ValueError(
            f"{name} must be strictly positive."
        )

    return converted


def _normalize_pooling_kind(
    value: TemporalPoolingKind | str,
) -> TemporalPoolingKind:
    if isinstance(
        value,
        TemporalPoolingKind,
    ):
        return value

    return TemporalPoolingKind(
        value
    )


def _normalize_head_reduction(
    value: TemporalPoolingHeadReduction | str,
) -> TemporalPoolingHeadReduction:
    if isinstance(
        value,
        TemporalPoolingHeadReduction,
    ):
        return value

    return TemporalPoolingHeadReduction(
        value
    )


def _normalize_zero_history_policy(
    value: TemporalPoolingZeroHistoryPolicy | str,
) -> TemporalPoolingZeroHistoryPolicy:
    if isinstance(
        value,
        TemporalPoolingZeroHistoryPolicy,
    ):
        return value

    return TemporalPoolingZeroHistoryPolicy(
        value
    )


def _require_pooled_memory(
    value: torch.Tensor,
) -> None:
    if not isinstance(
        value,
        torch.Tensor,
    ):
        raise TypeError(
            "pooled_memory must be a tensor."
        )

    if value.ndim != 2:
        raise ValueError(
            "pooled_memory must have shape [N, P]; "
            f"observed {tuple(value.shape)}."
        )

    node_count = int(
        value.shape[0]
    )
    output_dim = int(
        value.shape[1]
    )

    if (
        node_count <= 0
        or output_dim <= 0
    ):
        raise ValueError(
            "pooled_memory dimensions N and P must both be "
            "strictly positive."
        )

    if not value.dtype.is_floating_point:
        raise ValueError(
            "pooled_memory must use a floating-point dtype."
        )

    if not bool(
        torch.isfinite(
            value
        ).all().item()
    ):
        raise ValueError(
            "pooled_memory must contain only finite values."
        )


def _require_pooling_weights(
    value: torch.Tensor,
) -> None:
    if not isinstance(
        value,
        torch.Tensor,
    ):
        raise TypeError(
            "pooling_weights must be a tensor."
        )

    if value.ndim != 3:
        raise ValueError(
            "pooling_weights must have shape [N, A, T]; "
            f"observed {tuple(value.shape)}."
        )

    node_count = int(
        value.shape[0]
    )
    num_heads = int(
        value.shape[1]
    )
    sequence_length = int(
        value.shape[2]
    )

    if (
        node_count <= 0
        or num_heads <= 0
        or sequence_length <= 0
    ):
        raise ValueError(
            "pooling_weights dimensions N, A, and T must all be "
            "strictly positive."
        )

    if not value.dtype.is_floating_point:
        raise ValueError(
            "pooling_weights must use a floating-point dtype."
        )

    if not bool(
        torch.isfinite(
            value
        ).all().item()
    ):
        raise ValueError(
            "pooling_weights must contain only finite values."
        )

    if bool(
        (
            value
            < 0
        ).any().item()
    ):
        raise ValueError(
            "pooling_weights must be nonnegative."
        )


def _require_head_reduction_weights(
    value: torch.Tensor,
    *,
    num_heads: int,
    device: torch.device,
    dtype: torch.dtype,
    absolute_tolerance: float,
    relative_tolerance: float,
) -> None:
    if not isinstance(
        value,
        torch.Tensor,
    ):
        raise TypeError(
            "head_reduction_weights must be a tensor."
        )

    if value.ndim != 1:
        raise ValueError(
            "head_reduction_weights must have shape [A]."
        )

    if int(
        value.shape[0]
    ) != num_heads:
        raise ValueError(
            "head_reduction_weights must contain one value per "
            "pooling head."
        )

    if value.device != device:
        raise ValueError(
            "head_reduction_weights and pooled tensors must share "
            "one device."
        )

    if value.dtype != dtype:
        raise ValueError(
            "head_reduction_weights must use the same dtype as "
            "pooled_memory and pooling_weights."
        )

    if not bool(
        torch.isfinite(
            value
        ).all().item()
    ):
        raise ValueError(
            "head_reduction_weights must contain only finite values."
        )

    if bool(
        (
            value
            < 0
        ).any().item()
    ):
        raise ValueError(
            "head_reduction_weights must be nonnegative."
        )

    total = value.sum()
    one = torch.ones_like(
        total
    )

    if not bool(
        torch.isclose(
            total,
            one,
            atol=absolute_tolerance,
            rtol=relative_tolerance,
        ).item()
    ):
        raise ValueError(
            "head_reduction_weights must sum to one."
        )


def _validate_source_alignment(
    pooled_memory: torch.Tensor,
    pooling_weights: torch.Tensor,
    source_encoding: TemporalSequenceEncoding,
) -> None:
    if int(
        pooled_memory.shape[0]
    ) != source_encoding.node_count:
        raise ValueError(
            "pooled_memory dimension N must match source_encoding."
        )

    if int(
        pooling_weights.shape[0]
    ) != source_encoding.node_count:
        raise ValueError(
            "pooling_weights dimension N must match source_encoding."
        )

    if int(
        pooling_weights.shape[2]
    ) != source_encoding.sequence_length:
        raise ValueError(
            "pooling_weights dimension T must match source_encoding."
        )

    if pooled_memory.device != source_encoding.device:
        raise ValueError(
            "pooled_memory and source_encoding must share one device."
        )

    if pooling_weights.device != source_encoding.device:
        raise ValueError(
            "pooling_weights and source_encoding must share one device."
        )

    if pooled_memory.dtype != source_encoding.dtype:
        raise ValueError(
            "pooled_memory and source_encoding must use one dtype."
        )

    if pooling_weights.dtype != source_encoding.dtype:
        raise ValueError(
            "pooling_weights and source_encoding must use one dtype."
        )


def _validate_padding_mass(
    pooling_weights: torch.Tensor,
    timestep_mask: torch.Tensor,
) -> None:
    expanded_mask = (
        timestep_mask
        .unsqueeze(
            1
        )
        .expand_as(
            pooling_weights
        )
    )

    padded_weights = pooling_weights[
        ~expanded_mask
    ]

    if padded_weights.numel() == 0:
        return

    if not torch.equal(
        padded_weights,
        torch.zeros_like(
            padded_weights
        ),
    ):
        raise ValueError(
            "pooling_weights at padded timesteps must be exactly zero."
        )


def _validate_weight_normalization(
    pooling_weights: torch.Tensor,
    timestep_mask: torch.Tensor,
    *,
    absolute_tolerance: float,
    relative_tolerance: float,
) -> None:
    mass = pooling_weights.sum(
        dim=-1
    )
    has_history = timestep_mask.any(
        dim=-1
    )

    nonempty_mass = mass[
        has_history
    ]

    if nonempty_mass.numel() > 0:
        expected_one = torch.ones_like(
            nonempty_mass
        )

        if not bool(
            torch.isclose(
                nonempty_mass,
                expected_one,
                atol=absolute_tolerance,
                rtol=relative_tolerance,
            ).all().item()
        ):
            raise ValueError(
                "Each pooling head must assign total mass one to "
                "every nonempty history."
            )

    zero_history_mass = mass[
        ~has_history
    ]

    if zero_history_mass.numel() > 0:
        if not torch.equal(
            zero_history_mass,
            torch.zeros_like(
                zero_history_mass
            ),
        ):
            raise ValueError(
                "Every pooling head must assign total mass zero to "
                "zero-history rows."
            )


def _validate_zero_history_behavior(
    pooled_memory: torch.Tensor,
    source_encoding: TemporalSequenceEncoding,
    *,
    zero_history_policy: TemporalPoolingZeroHistoryPolicy,
) -> None:
    has_history = (
        source_encoding
        .timestep_mask
        .any(
            dim=-1
        )
    )
    zero_rows = ~has_history

    if not bool(
        zero_rows.any().item()
    ):
        return

    if (
        zero_history_policy
        == TemporalPoolingZeroHistoryPolicy.ERROR
    ):
        row_indices = (
            torch.nonzero(
                zero_rows,
                as_tuple=False,
            )
            .flatten()
            .detach()
            .cpu()
            .tolist()
        )

        raise ValueError(
            "Zero-history rows are incompatible with "
            "zero_history_policy='error'. "
            f"Rows: {row_indices}."
        )

    if (
        zero_history_policy
        == TemporalPoolingZeroHistoryPolicy.ZERO
    ):
        zero_memory = pooled_memory[
            zero_rows
        ]

        if not torch.equal(
            zero_memory,
            torch.zeros_like(
                zero_memory
            ),
        ):
            raise ValueError(
                "zero_history_policy='zero' requires pooled_memory "
                "to be exactly zero for zero-history rows."
            )


def _validate_head_reduction_contract(
    *,
    head_reduction: TemporalPoolingHeadReduction,
    num_heads: int,
    head_reduction_weights: torch.Tensor | None,
    device: torch.device,
    dtype: torch.dtype,
    absolute_tolerance: float,
    relative_tolerance: float,
) -> None:
    if (
        head_reduction
        == TemporalPoolingHeadReduction.SINGLE_HEAD
    ):
        if num_heads != 1:
            raise ValueError(
                "head_reduction='single_head' requires exactly one "
                "pooling head."
            )

        if head_reduction_weights is not None:
            raise ValueError(
                "head_reduction_weights must be None for "
                "head_reduction='single_head'."
            )

        return

    if num_heads <= 1:
        raise ValueError(
            f"head_reduction={head_reduction.value!r} requires more "
            "than one pooling head."
        )

    if (
        head_reduction
        == TemporalPoolingHeadReduction.WEIGHTED_MEAN
    ):
        if head_reduction_weights is None:
            raise ValueError(
                "head_reduction='weighted_mean' requires "
                "head_reduction_weights."
            )

        _require_head_reduction_weights(
            head_reduction_weights,
            num_heads=num_heads,
            device=device,
            dtype=dtype,
            absolute_tolerance=absolute_tolerance,
            relative_tolerance=relative_tolerance,
        )
        return

    if head_reduction_weights is not None:
        raise ValueError(
            "head_reduction_weights are only valid for "
            "head_reduction='weighted_mean'."
        )


def _validate_computation_alignment(
    computation_provenance: MemoryComputationProvenance,
    source_encoding: TemporalSequenceEncoding,
) -> None:
    lineage = computation_provenance.lineage

    source_lineage = (
        source_encoding
        .lineage_fingerprint()
    )

    if source_lineage not in (
        lineage
        .source_lineage_fingerprints
    ):
        raise ValueError(
            "Pooling computation lineage must include the exact "
            "source-encoding lineage fingerprint."
        )

    expected_node_axis = (
        source_encoding
        .node_axis
        .fingerprint()
    )
    expected_temporal_axis = (
        source_encoding
        .temporal_alignment_fingerprint()
    )
    expected_feature_axis = (
        source_encoding
        .feature_axis
        .fingerprint()
    )

    if (
        lineage.node_axis_fingerprint
        is not None
        and lineage.node_axis_fingerprint
        != expected_node_axis
    ):
        raise ValueError(
            "Pooling lineage node_axis_fingerprint must match the "
            "source encoding."
        )

    if (
        lineage.temporal_axis_fingerprint
        is not None
        and lineage.temporal_axis_fingerprint
        != expected_temporal_axis
    ):
        raise ValueError(
            "Pooling lineage temporal_axis_fingerprint must match the "
            "source encoding."
        )

    if (
        lineage.feature_axis_fingerprint
        is not None
        and lineage.feature_axis_fingerprint
        != expected_feature_axis
    ):
        raise ValueError(
            "Pooling lineage feature_axis_fingerprint must match the "
            "source encoding."
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


# =============================================================================
# Shared temporal-pooling output
# =============================================================================


@dataclass(
    slots=True,
    frozen=True,
)
class TemporalPoolingOutput:
    """
    Hazard-independent pooled memory with preserved temporal weights.

    Parameters
    ----------
    pooled_memory:
        Finite floating tensor ``[N, P]``.

    pooling_weights:
        Nonnegative finite tensor ``[N, A, T]``. Padding receives exactly zero
        mass. Each head sums to one over valid timesteps for nonempty histories.

    source_encoding:
        The exact ``TemporalSequenceEncoding`` object consumed by pooling.

    pooling_kind:
        Semantic pooling family.

    head_reduction:
        Declared reduction of head-specific summaries into ``pooled_memory``.

    zero_history_policy:
        Explicit behavior for all-masked source histories.

    computation_provenance:
        Pooling architecture, optional parameter snapshot, and lineage.

    head_reduction_weights:
        Optional normalized weights ``[A]`` required only for
        ``weighted_mean`` head reduction.
    """

    pooled_memory: torch.Tensor
    pooling_weights: torch.Tensor
    source_encoding: TemporalSequenceEncoding

    pooling_kind: TemporalPoolingKind | str
    head_reduction: (
        TemporalPoolingHeadReduction
        | str
    )
    zero_history_policy: (
        TemporalPoolingZeroHistoryPolicy
        | str
    )

    computation_provenance: (
        MemoryComputationProvenance
    )

    head_reduction_weights: (
        torch.Tensor
        | None
    ) = None

    absolute_tolerance: float = 1e-7
    relative_tolerance: float = 1e-6

    pooling_name: str = (
        "temporal_pooling_output"
    )

    schema_version: str = (
        TEMPORAL_POOLING_OUTPUT_SCHEMA_VERSION
    )

    def __post_init__(
        self,
    ) -> None:
        _require_pooled_memory(
            self.pooled_memory
        )
        _require_pooling_weights(
            self.pooling_weights
        )

        if not isinstance(
            self.source_encoding,
            TemporalSequenceEncoding,
        ):
            raise TypeError(
                "source_encoding must be a "
                "TemporalSequenceEncoding."
            )

        pooling_kind = (
            _normalize_pooling_kind(
                self.pooling_kind
            )
        )
        head_reduction = (
            _normalize_head_reduction(
                self.head_reduction
            )
        )
        zero_history_policy = (
            _normalize_zero_history_policy(
                self.zero_history_policy
            )
        )

        object.__setattr__(
            self,
            "pooling_kind",
            pooling_kind,
        )
        object.__setattr__(
            self,
            "head_reduction",
            head_reduction,
        )
        object.__setattr__(
            self,
            "zero_history_policy",
            zero_history_policy,
        )

        if not isinstance(
            self.computation_provenance,
            MemoryComputationProvenance,
        ):
            raise TypeError(
                "computation_provenance must be a "
                "MemoryComputationProvenance."
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

        _validate_source_alignment(
            self.pooled_memory,
            self.pooling_weights,
            self.source_encoding,
        )
        _validate_padding_mass(
            self.pooling_weights,
            self.timestep_mask,
        )
        _validate_weight_normalization(
            self.pooling_weights,
            self.timestep_mask,
            absolute_tolerance=(
                absolute_tolerance
            ),
            relative_tolerance=(
                relative_tolerance
            ),
        )
        _validate_zero_history_behavior(
            self.pooled_memory,
            self.source_encoding,
            zero_history_policy=(
                zero_history_policy
            ),
        )
        _validate_head_reduction_contract(
            head_reduction=head_reduction,
            num_heads=self.num_heads,
            head_reduction_weights=(
                self.head_reduction_weights
            ),
            device=self.device,
            dtype=self.dtype,
            absolute_tolerance=(
                absolute_tolerance
            ),
            relative_tolerance=(
                relative_tolerance
            ),
        )
        _validate_computation_alignment(
            self.computation_provenance,
            self.source_encoding,
        )

        _require_nonempty_string(
            "pooling_name",
            self.pooling_name,
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
            self
            .pooled_memory
            .shape[0]
        )

    @property
    def item_count(
        self,
    ) -> int:
        return self.node_count

    @property
    def pooled_dim(
        self,
    ) -> int:
        return int(
            self
            .pooled_memory
            .shape[1]
        )

    @property
    def output_dim(
        self,
    ) -> int:
        return self.pooled_dim

    @property
    def num_heads(
        self,
    ) -> int:
        return int(
            self
            .pooling_weights
            .shape[1]
        )

    @property
    def sequence_length(
        self,
    ) -> int:
        return int(
            self
            .pooling_weights
            .shape[2]
        )

    @property
    def pooled_shape(
        self,
    ) -> tuple[int, int]:
        return (
            self.node_count,
            self.pooled_dim,
        )

    @property
    def weight_shape(
        self,
    ) -> tuple[int, int, int]:
        return (
            self.node_count,
            self.num_heads,
            self.sequence_length,
        )

    @property
    def device(
        self,
    ) -> torch.device:
        return (
            self
            .pooled_memory
            .device
        )

    @property
    def dtype(
        self,
    ) -> torch.dtype:
        return (
            self
            .pooled_memory
            .dtype
        )

    # -------------------------------------------------------------------------
    # Exact source-object preservation
    # -------------------------------------------------------------------------

    @property
    def source_sequence_encoding(
        self,
    ) -> TemporalSequenceEncoding:
        return self.source_encoding

    @property
    def encoded_sequence(
        self,
    ) -> torch.Tensor:
        return (
            self
            .source_encoding
            .encoded_sequence
        )

    @property
    def source_history(
        self,
    ):
        return (
            self
            .source_encoding
            .source_history
        )

    @property
    def timestep_mask(
        self,
    ) -> torch.Tensor:
        return (
            self
            .source_encoding
            .timestep_mask
        )

    @property
    def valid_lengths(
        self,
    ) -> torch.Tensor:
        return (
            self
            .source_encoding
            .valid_lengths
        )

    @property
    def node_axis(
        self,
    ):
        return (
            self
            .source_encoding
            .node_axis
        )

    @property
    def feature_axis(
        self,
    ):
        return (
            self
            .source_encoding
            .feature_axis
        )

    @property
    def temporal_coordinates(
        self,
    ):
        return (
            self
            .source_encoding
            .temporal_coordinates
        )

    @property
    def node_ids(
        self,
    ) -> tuple[str, ...]:
        return (
            self
            .source_encoding
            .node_ids
        )

    @property
    def node_batch_index(
        self,
    ) -> torch.Tensor:
        return (
            self
            .source_encoding
            .node_batch_index
        )

    @property
    def graph_count(
        self,
    ) -> int:
        return (
            self
            .source_encoding
            .graph_count
        )

    @property
    def graph_ids(
        self,
    ) -> tuple[str, ...]:
        return (
            self
            .source_encoding
            .graph_ids
        )

    @property
    def has_zero_history(
        self,
    ) -> bool:
        return (
            self
            .source_encoding
            .has_zero_history
        )

    # -------------------------------------------------------------------------
    # Provenance convenience
    # -------------------------------------------------------------------------

    @property
    def architecture_fingerprint(
        self,
    ) -> str:
        return (
            self
            .computation_provenance
            .architecture_fingerprint
        )

    @property
    def parameter_snapshot_fingerprint(
        self,
    ) -> str | None:
        return (
            self
            .computation_provenance
            .parameter_snapshot_fingerprint
        )

    @property
    def computation_lineage_fingerprint(
        self,
    ) -> str:
        return (
            self
            .computation_provenance
            .lineage_fingerprint
        )

    # -------------------------------------------------------------------------
    # Deterministic identities
    # -------------------------------------------------------------------------

    def alignment_fingerprint(
        self,
    ) -> str:
        return (
            self
            .source_encoding
            .alignment_fingerprint()
        )

    def temporal_alignment_fingerprint(
        self,
    ) -> str:
        return (
            self
            .source_encoding
            .temporal_alignment_fingerprint()
        )

    def semantic_dict(
        self,
    ) -> dict[str, Any]:
        return {
            "schema_version": (
                self.schema_version
            ),
            "pooling_name": (
                self.pooling_name
            ),
            "pooling_kind": (
                self.pooling_kind.value
            ),
            "head_reduction": (
                self.head_reduction.value
            ),
            "zero_history_policy": (
                self.zero_history_policy.value
            ),
            "pooled_shape": list(
                self.pooled_shape
            ),
            "weight_shape": list(
                self.weight_shape
            ),
            "has_head_reduction_weights": (
                self.head_reduction_weights
                is not None
            ),
            "weight_semantics": (
                TEMPORAL_POOLING_WEIGHT_SEMANTICS
            ),
            "padding_policy": (
                TEMPORAL_POOLING_PADDING_POLICY
            ),
            "normalization_policy": (
                TEMPORAL_POOLING_NORMALIZATION_POLICY
            ),
            "absolute_tolerance": (
                self.absolute_tolerance
            ),
            "relative_tolerance": (
                self.relative_tolerance
            ),
            "source_encoding_lineage_fingerprint": (
                self
                .source_encoding
                .lineage_fingerprint()
            ),
            "source_alignment_fingerprint": (
                self
                .source_encoding
                .alignment_fingerprint()
            ),
            "architecture_fingerprint": (
                self
                .architecture_fingerprint
            ),
            "parameter_snapshot_fingerprint": (
                self
                .parameter_snapshot_fingerprint
            ),
            "computation_lineage_fingerprint": (
                self
                .computation_lineage_fingerprint
            ),
            "scientific_interpretation": (
                TEMPORAL_POOLING_SCIENTIFIC_INTERPRETATION
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
            "pooled_memory": (
                self.pooled_memory
            ),
            "pooling_weights": (
                self.pooling_weights
            ),
        }

        if (
            self.head_reduction_weights
            is not None
        ):
            tensors[
                "head_reduction_weights"
            ] = self.head_reduction_weights

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
            "source_encoding_lineage_fingerprint": (
                self
                .source_encoding
                .lineage_fingerprint()
            ),
            "computation_provenance_fingerprint": (
                self
                .computation_provenance
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
        non_blocking: bool = False,
    ) -> Self:
        """
        Move the complete pooling contract to one device.

        Dtype conversion is intentionally not exposed here. The exact source
        encoding and its computation lineage are dtype-sensitive software
        artifacts. Construct a new encoding and matching provenance when a
        different numerical dtype is required.
        """

        moved_source = (
            self
            .source_encoding
            .to(
                device,
                dtype=self.dtype,
                non_blocking=non_blocking,
            )
        )

        return type(self)(
            pooled_memory=(
                self
                .pooled_memory
                .to(
                    device=device,
                    non_blocking=(
                        non_blocking
                    ),
                )
            ),
            pooling_weights=(
                self
                .pooling_weights
                .to(
                    device=device,
                    non_blocking=(
                        non_blocking
                    ),
                )
            ),
            source_encoding=moved_source,
            pooling_kind=(
                self.pooling_kind
            ),
            head_reduction=(
                self.head_reduction
            ),
            zero_history_policy=(
                self.zero_history_policy
            ),
            computation_provenance=(
                self
                .computation_provenance
            ),
            head_reduction_weights=(
                self
                .head_reduction_weights
                .to(
                    device=device,
                    non_blocking=(
                        non_blocking
                    ),
                )
                if (
                    self.head_reduction_weights
                    is not None
                )
                else None
            ),
            absolute_tolerance=(
                self.absolute_tolerance
            ),
            relative_tolerance=(
                self.relative_tolerance
            ),
            pooling_name=(
                self.pooling_name
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


TemporalMemoryPoolingOutput = TemporalPoolingOutput
PoolingOutput = TemporalPoolingOutput


# =============================================================================
# Public API
# =============================================================================


__all__ = (
    # Schema identity and interpretation.
    "TEMPORAL_POOLING_OUTPUT_SCHEMA_VERSION",
    "TEMPORAL_POOLING_WEIGHT_SEMANTICS",
    "TEMPORAL_POOLING_PADDING_POLICY",
    "TEMPORAL_POOLING_NORMALIZATION_POLICY",
    "TEMPORAL_POOLING_SCIENTIFIC_INTERPRETATION",

    # Controlled vocabularies.
    "TemporalPoolingKind",
    "TemporalPoolingHeadReduction",
    "TemporalPoolingZeroHistoryPolicy",
    "CANONICAL_TEMPORAL_POOLING_KINDS",
    "CANONICAL_TEMPORAL_POOLING_HEAD_REDUCTIONS",
    "CANONICAL_TEMPORAL_POOLING_ZERO_HISTORY_POLICIES",

    # Main contract and aliases.
    "TemporalPoolingOutput",
    "TemporalMemoryPoolingOutput",
    "PoolingOutput",
)
