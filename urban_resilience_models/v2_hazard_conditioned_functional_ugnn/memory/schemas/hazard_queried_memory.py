"""
Hazard-conditioned temporal-retrieval and fused-memory contracts.

Target path:
    urban_resilience_models/
        v2_hazard_conditioned_functional_ugnn/
            memory/
                schemas/
                    hazard_queried_memory.py

This module owns two related but distinct public contracts:

``TemporalQueryRetrievalOutput``
    A query-neutral result of reading an unpooled temporal sequence with one
    aligned query. It preserves retrieved context, per-head temporal retrieval
    weights, the exact source sequence, query alignment identity, architecture
    provenance, and execution lineage.

``HazardQueriedMemory``
    A hazard-specific orchestration result that preserves the complete
    ``UrbanMemory``, the complete ``HazardQueryEncoding``, the explicit
    node-aligned hazard query, the exact retrieval result, the final fused
    memory, declared fusion policy, optional intermediate fusion components,
    architecture provenance, and execution lineage.

The two principal representations must remain separate:

``retrieved_context``
    Information read from the temporal sequence using the aligned query.

``fused_memory``
    Representation produced after combining retrieved context with declared
    components such as generic urban memory, hazard query state, current node
    state, or projected/gated intermediates.

They are not interchangeable aliases.

Dependency boundary
-------------------
``TemporalQueryRetrievalOutput`` is hazard-neutral and imports no hazard
implementation.

``HazardQueriedMemory`` imports only stable metadata-preserving hazard result
contracts:

- ``HazardQueryEncoding``;
- ``HazardEmbeddingLookup``;
- ``NodeAlignedHazardEmbeddingLookup``.

It does not import cross-attention implementations. This preserves the clean
dependency direction:

    memory schemas
        -> hazard cross-attention implementation
        -> hazard-queried-memory orchestration

and prevents a shared schema from depending on a trainable retrieval module.

Hazard-query alignment
----------------------
A ``HazardQueryEncoding`` may be:

``NodeAlignedHazardEmbeddingLookup``
    One query row per node. Its ``node_batch_index`` must equal the urban
    memory node membership exactly.

``HazardEmbeddingLookup``
    One query row per packed graph. Node alignment is performed explicitly:

        node_hazard_query = graph_query[node_batch_index]

Ordinary PyTorch broadcasting is forbidden. The exact original hazard query
remains attached for canonical hazard names, stable IDs, vocabulary identity,
unknown-mask semantics, and query-encoder lineage.

Retrieval weights
-----------------
Retrieval weights use shape ``[N, A, T]`` and must assign:

- exact zero mass to padding;
- nonnegative finite mass to valid timesteps;
- total mass one per head for nonempty histories;
- total mass zero per head for zero-history rows.

These weights may be described as retrieval mass or model-assigned temporal
relevance. They do not by themselves prove causal historical importance,
mechanistic explanation, or real-world causal effects.

Zero-history behavior
---------------------
Retrieval supports explicit policies:

``error``
    Reject zero-history rows.

``zero``
    Require zero retrieval weights and zero retrieved context.

``learned_fallback``
    Require zero retrieval weights but permit a finite learned fallback
    context.

Interpretation
--------------------------
Hazard-queried memory is a model representation and reproducibility contract.
It does not establish causal hazard effects, causal historical attribution,
calibrated risk, or faithful mechanistic explanation.
"""

from __future__ import annotations

from dataclasses import dataclass
from dataclasses import field
from dataclasses import replace as dataclass_replace
from enum import StrEnum
from hashlib import sha256
import json
import math
from types import MappingProxyType
from typing import Any, Final, Mapping, Self

import torch

from ...hazard.hazard_embeddings import (
    HazardEmbeddingLookup,
    NodeAlignedHazardEmbeddingLookup,
)
from ...hazard.hazard_query_encoder import (
    HazardQueryEncoding,
)
from .provenance import (
    MemoryComputationProvenance,
)
from .sequence_encoding import (
    TemporalSequenceEncoding,
)
from .urban_memory import (
    UrbanMemory,
)


# =============================================================================
# Schema identity and interpretation
# =============================================================================


TEMPORAL_QUERY_RETRIEVAL_OUTPUT_SCHEMA_VERSION: Final[str] = "0.1"
HAZARD_QUERIED_MEMORY_SCHEMA_VERSION: Final[str] = "0.1"

TEMPORAL_QUERY_RETRIEVAL_WEIGHT_SEMANTICS: Final[str] = (
    "model_assigned_temporal_relevance_not_causal_importance"
)

TEMPORAL_QUERY_RETRIEVAL_PADDING_POLICY: Final[str] = (
    "exact_zero_mass_at_padded_positions"
)

TEMPORAL_QUERY_RETRIEVAL_NORMALIZATION_POLICY: Final[str] = (
    "unit_mass_per_nonempty_node_and_head_zero_mass_for_zero_history"
)

HAZARD_QUERY_ALIGNMENT_POLICY: Final[str] = (
    "explicit_node_alignment_no_implicit_broadcasting"
)

HAZARD_QUERIED_MEMORY_SCIENTIFIC_INTERPRETATION: Final[str] = (
    "hazard_conditioned_model_memory_not_causal_effect_or_explanation"
)


# =============================================================================
# Controlled vocabularies
# =============================================================================


class TemporalQueryRetrievalKind(StrEnum):
    """Semantic family of one sequence-query retrieval operation."""

    DOT_PRODUCT_ATTENTION = "dot_product_attention"
    SCALED_DOT_PRODUCT_ATTENTION = (
        "scaled_dot_product_attention"
    )
    ADDITIVE_ATTENTION = "additive_attention"
    MULTIHEAD_CROSS_ATTENTION = (
        "multihead_cross_attention"
    )
    OTHER = "other"


class TemporalQueryRetrievalHeadReduction(StrEnum):
    """How per-head retrieved summaries form ``retrieved_context``."""

    SINGLE_HEAD = "single_head"
    MEAN = "mean"
    WEIGHTED_MEAN = "weighted_mean"
    CONCAT_PROJECTION = "concat_projection"
    OTHER = "other"


class TemporalQueryRetrievalZeroHistoryPolicy(StrEnum):
    """How retrieval handles a node with no valid history."""

    ERROR = "error"
    ZERO = "zero"
    LEARNED_FALLBACK = "learned_fallback"


class HazardQueryAlignmentScope(StrEnum):
    """Original scope of one hazard query before node alignment."""

    NODE = "node"
    GRAPH_BROADCAST = "graph_broadcast"


class HazardMemoryFusionPolicy(StrEnum):
    """Declared operation producing final hazard-conditioned memory."""

    RETRIEVED_ONLY = "retrieved_only"
    CONCAT_PROJECTION = "concat_projection"
    PROJECTED_SUM = "projected_sum"
    GATED_FUSION = "gated_fusion"
    FILM_CONDITIONING = "film_conditioning"
    OTHER = "other"


CANONICAL_TEMPORAL_QUERY_RETRIEVAL_KINDS: Final[
    tuple[str, ...]
] = tuple(
    value.value
    for value in TemporalQueryRetrievalKind
)

CANONICAL_TEMPORAL_QUERY_RETRIEVAL_HEAD_REDUCTIONS: Final[
    tuple[str, ...]
] = tuple(
    value.value
    for value in TemporalQueryRetrievalHeadReduction
)

CANONICAL_TEMPORAL_QUERY_RETRIEVAL_ZERO_HISTORY_POLICIES: Final[
    tuple[str, ...]
] = tuple(
    value.value
    for value in TemporalQueryRetrievalZeroHistoryPolicy
)

CANONICAL_HAZARD_QUERY_ALIGNMENT_SCOPES: Final[
    tuple[str, ...]
] = tuple(
    value.value
    for value in HazardQueryAlignmentScope
)

CANONICAL_HAZARD_MEMORY_FUSION_POLICIES: Final[
    tuple[str, ...]
] = tuple(
    value.value
    for value in HazardMemoryFusionPolicy
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


def _normalize_retrieval_kind(
    value: TemporalQueryRetrievalKind | str,
) -> TemporalQueryRetrievalKind:
    if isinstance(
        value,
        TemporalQueryRetrievalKind,
    ):
        return value

    return TemporalQueryRetrievalKind(
        value
    )


def _normalize_retrieval_head_reduction(
    value: (
        TemporalQueryRetrievalHeadReduction
        | str
    ),
) -> TemporalQueryRetrievalHeadReduction:
    if isinstance(
        value,
        TemporalQueryRetrievalHeadReduction,
    ):
        return value

    return TemporalQueryRetrievalHeadReduction(
        value
    )


def _normalize_retrieval_zero_history_policy(
    value: (
        TemporalQueryRetrievalZeroHistoryPolicy
        | str
    ),
) -> TemporalQueryRetrievalZeroHistoryPolicy:
    if isinstance(
        value,
        TemporalQueryRetrievalZeroHistoryPolicy,
    ):
        return value

    return TemporalQueryRetrievalZeroHistoryPolicy(
        value
    )


def _normalize_fusion_policy(
    value: HazardMemoryFusionPolicy | str,
) -> HazardMemoryFusionPolicy:
    if isinstance(
        value,
        HazardMemoryFusionPolicy,
    ):
        return value

    return HazardMemoryFusionPolicy(
        value
    )


def _require_float_matrix(
    name: str,
    value: torch.Tensor,
) -> None:
    if not isinstance(
        value,
        torch.Tensor,
    ):
        raise TypeError(
            f"{name} must be a tensor."
        )

    if value.ndim != 2:
        raise ValueError(
            f"{name} must have shape [N, D]; "
            f"observed {tuple(value.shape)}."
        )

    if (
        int(
            value.shape[0]
        )
        <= 0
        or int(
            value.shape[1]
        )
        <= 0
    ):
        raise ValueError(
            f"{name} dimensions must be strictly positive."
        )

    if not value.dtype.is_floating_point:
        raise ValueError(
            f"{name} must use a floating-point dtype."
        )

    if not bool(
        torch.isfinite(
            value
        ).all().item()
    ):
        raise ValueError(
            f"{name} must contain only finite values."
        )


def _require_attention_weights(
    value: torch.Tensor,
) -> None:
    if not isinstance(
        value,
        torch.Tensor,
    ):
        raise TypeError(
            "attention_weights must be a tensor."
        )

    if value.ndim != 3:
        raise ValueError(
            "attention_weights must have shape [N, A, T]; "
            f"observed {tuple(value.shape)}."
        )

    if any(
        int(
            dimension
        )
        <= 0
        for dimension in value.shape
    ):
        raise ValueError(
            "attention_weights dimensions N, A, and T must all be "
            "strictly positive."
        )

    if not value.dtype.is_floating_point:
        raise ValueError(
            "attention_weights must use a floating-point dtype."
        )

    if not bool(
        torch.isfinite(
            value
        ).all().item()
    ):
        raise ValueError(
            "attention_weights must contain only finite values."
        )

    if bool(
        (
            value
            < 0
        ).any().item()
    ):
        raise ValueError(
            "attention_weights must be nonnegative."
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
            "retrieval head."
        )

    if value.device != device:
        raise ValueError(
            "head_reduction_weights and retrieval tensors must share "
            "one device."
        )

    if value.dtype != dtype:
        raise ValueError(
            "head_reduction_weights must use the retrieval dtype."
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


def _validate_retrieval_source_alignment(
    retrieved_context: torch.Tensor,
    attention_weights: torch.Tensor,
    source_sequence: TemporalSequenceEncoding,
) -> None:
    if int(
        retrieved_context.shape[0]
    ) != source_sequence.node_count:
        raise ValueError(
            "retrieved_context dimension N must match source_sequence."
        )

    if int(
        attention_weights.shape[0]
    ) != source_sequence.node_count:
        raise ValueError(
            "attention_weights dimension N must match source_sequence."
        )

    if int(
        attention_weights.shape[2]
    ) != source_sequence.sequence_length:
        raise ValueError(
            "attention_weights dimension T must match source_sequence."
        )

    if retrieved_context.device != source_sequence.device:
        raise ValueError(
            "retrieved_context and source_sequence must share one "
            "device."
        )

    if attention_weights.device != source_sequence.device:
        raise ValueError(
            "attention_weights and source_sequence must share one "
            "device."
        )

    if retrieved_context.dtype != source_sequence.dtype:
        raise ValueError(
            "retrieved_context and source_sequence must use one dtype."
        )

    if attention_weights.dtype != source_sequence.dtype:
        raise ValueError(
            "attention_weights and source_sequence must use one dtype."
        )


def _validate_padding_mass(
    attention_weights: torch.Tensor,
    timestep_mask: torch.Tensor,
) -> None:
    expanded_mask = (
        timestep_mask
        .unsqueeze(
            1
        )
        .expand_as(
            attention_weights
        )
    )

    padded_weights = attention_weights[
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
            "attention_weights at padded timesteps must be exactly "
            "zero."
        )


def _validate_attention_normalization(
    attention_weights: torch.Tensor,
    timestep_mask: torch.Tensor,
    *,
    absolute_tolerance: float,
    relative_tolerance: float,
) -> None:
    mass = attention_weights.sum(
        dim=-1
    )
    has_history = timestep_mask.any(
        dim=-1
    )

    nonempty_mass = mass[
        has_history
    ]

    if nonempty_mass.numel() > 0:
        expected = torch.ones_like(
            nonempty_mass
        )

        if not bool(
            torch.isclose(
                nonempty_mass,
                expected,
                atol=absolute_tolerance,
                rtol=relative_tolerance,
            ).all().item()
        ):
            raise ValueError(
                "Each retrieval head must assign total mass one to "
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
                "Each retrieval head must assign total mass zero to "
                "zero-history rows."
            )


def _validate_retrieval_zero_history_behavior(
    retrieved_context: torch.Tensor,
    source_sequence: TemporalSequenceEncoding,
    *,
    zero_history_policy: (
        TemporalQueryRetrievalZeroHistoryPolicy
    ),
) -> None:
    zero_rows = ~(
        source_sequence
        .timestep_mask
        .any(
            dim=-1
        )
    )

    if not bool(
        zero_rows.any().item()
    ):
        return

    if (
        zero_history_policy
        == TemporalQueryRetrievalZeroHistoryPolicy.ERROR
    ):
        indices = (
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
            "Zero-history rows are incompatible with retrieval "
            "zero_history_policy='error'. "
            f"Rows: {indices}."
        )

    if (
        zero_history_policy
        == TemporalQueryRetrievalZeroHistoryPolicy.ZERO
    ):
        rows = retrieved_context[
            zero_rows
        ]

        if not torch.equal(
            rows,
            torch.zeros_like(
                rows
            ),
        ):
            raise ValueError(
                "retrieval zero_history_policy='zero' requires "
                "retrieved_context to be exactly zero for zero-history "
                "rows."
            )


def _validate_retrieval_head_reduction(
    *,
    head_reduction: (
        TemporalQueryRetrievalHeadReduction
    ),
    num_heads: int,
    head_reduction_weights: torch.Tensor | None,
    device: torch.device,
    dtype: torch.dtype,
    absolute_tolerance: float,
    relative_tolerance: float,
) -> None:
    if (
        head_reduction
        == TemporalQueryRetrievalHeadReduction.SINGLE_HEAD
    ):
        if num_heads != 1:
            raise ValueError(
                "retrieval head_reduction='single_head' requires "
                "exactly one head."
            )

        if head_reduction_weights is not None:
            raise ValueError(
                "head_reduction_weights must be None for single-head "
                "retrieval."
            )

        return

    if num_heads <= 1:
        raise ValueError(
            f"retrieval head_reduction={head_reduction.value!r} "
            "requires more than one head."
        )

    if (
        head_reduction
        == TemporalQueryRetrievalHeadReduction.WEIGHTED_MEAN
    ):
        if head_reduction_weights is None:
            raise ValueError(
                "weighted-mean retrieval requires "
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
            "head_reduction_weights are valid only for "
            "head_reduction='weighted_mean'."
        )


def _validate_retrieval_computation_alignment(
    computation_provenance: MemoryComputationProvenance,
    source_sequence: TemporalSequenceEncoding,
    *,
    query_lineage_fingerprint: str,
) -> None:
    lineage = computation_provenance.lineage
    required_sources = {
        source_sequence.lineage_fingerprint(),
        query_lineage_fingerprint,
    }
    observed_sources = set(
        lineage.source_lineage_fingerprints
    )
    missing = sorted(
        required_sources
        - observed_sources
    )

    if missing:
        raise ValueError(
            "Retrieval computation lineage must include the exact "
            "source-sequence and query lineage fingerprints. Missing: "
            f"{missing}."
        )

    expected_node_axis = (
        source_sequence
        .node_axis
        .fingerprint()
    )
    expected_temporal_axis = (
        source_sequence
        .temporal_alignment_fingerprint()
    )
    expected_feature_axis = (
        source_sequence
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
            "Retrieval lineage node_axis_fingerprint must match the "
            "source sequence."
        )

    if (
        lineage.temporal_axis_fingerprint
        is not None
        and lineage.temporal_axis_fingerprint
        != expected_temporal_axis
    ):
        raise ValueError(
            "Retrieval lineage temporal_axis_fingerprint must match "
            "the source sequence."
        )

    if (
        lineage.feature_axis_fingerprint
        is not None
        and lineage.feature_axis_fingerprint
        != expected_feature_axis
    ):
        raise ValueError(
            "Retrieval lineage feature_axis_fingerprint must match "
            "the source sequence."
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


def _immutable_tensor_mapping(
    name: str,
    values: Mapping[
        str,
        torch.Tensor,
    ],
    *,
    node_count: int,
    device: torch.device,
    dtype: torch.dtype,
) -> Mapping[
    str,
    torch.Tensor,
]:
    if not isinstance(
        values,
        Mapping,
    ):
        raise TypeError(
            f"{name} must be a mapping."
        )

    copied: dict[
        str,
        torch.Tensor,
    ] = {}

    for component_name, tensor in (
        values.items()
    ):
        _require_nonempty_string(
            f"{name} key",
            component_name,
        )
        _require_float_matrix(
            f"{name}[{component_name!r}]",
            tensor,
        )

        if int(
            tensor.shape[0]
        ) != node_count:
            raise ValueError(
                f"{name}[{component_name!r}] must contain one row "
                "per node."
            )

        if tensor.device != device:
            raise ValueError(
                f"{name}[{component_name!r}] must share the fused "
                "memory device."
            )

        if tensor.dtype != dtype:
            raise ValueError(
                f"{name}[{component_name!r}] must share the fused "
                "memory dtype."
            )

        copied[
            component_name
        ] = tensor

    return MappingProxyType(
        copied
    )


# =============================================================================
# Query-neutral temporal retrieval output
# =============================================================================


@dataclass(
    slots=True,
    frozen=True,
)
class TemporalQueryRetrievalOutput:
    """
    Query-neutral temporal retrieval over one preserved sequence.

    ``query_alignment_fingerprint`` identifies the exact query after alignment
    to the source sequence node axis.

    ``query_lineage_fingerprint`` identifies the metadata-preserving upstream
    query object.
    """

    retrieved_context: torch.Tensor
    attention_weights: torch.Tensor
    source_sequence: TemporalSequenceEncoding

    query_alignment_fingerprint: str
    query_lineage_fingerprint: str

    retrieval_kind: (
        TemporalQueryRetrievalKind
        | str
    )
    head_reduction: (
        TemporalQueryRetrievalHeadReduction
        | str
    )
    zero_history_policy: (
        TemporalQueryRetrievalZeroHistoryPolicy
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

    retrieval_name: str = (
        "temporal_query_retrieval"
    )

    schema_version: str = (
        TEMPORAL_QUERY_RETRIEVAL_OUTPUT_SCHEMA_VERSION
    )

    def __post_init__(
        self,
    ) -> None:
        _require_float_matrix(
            "retrieved_context",
            self.retrieved_context,
        )
        _require_attention_weights(
            self.attention_weights
        )

        if not isinstance(
            self.source_sequence,
            TemporalSequenceEncoding,
        ):
            raise TypeError(
                "source_sequence must be a "
                "TemporalSequenceEncoding."
            )

        _require_nonempty_string(
            "query_alignment_fingerprint",
            self.query_alignment_fingerprint,
        )
        _require_nonempty_string(
            "query_lineage_fingerprint",
            self.query_lineage_fingerprint,
        )

        retrieval_kind = (
            _normalize_retrieval_kind(
                self.retrieval_kind
            )
        )
        head_reduction = (
            _normalize_retrieval_head_reduction(
                self.head_reduction
            )
        )
        zero_history_policy = (
            _normalize_retrieval_zero_history_policy(
                self.zero_history_policy
            )
        )

        object.__setattr__(
            self,
            "retrieval_kind",
            retrieval_kind,
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

        _validate_retrieval_source_alignment(
            self.retrieved_context,
            self.attention_weights,
            self.source_sequence,
        )
        _validate_padding_mass(
            self.attention_weights,
            self.timestep_mask,
        )
        _validate_attention_normalization(
            self.attention_weights,
            self.timestep_mask,
            absolute_tolerance=(
                absolute_tolerance
            ),
            relative_tolerance=(
                relative_tolerance
            ),
        )
        _validate_retrieval_zero_history_behavior(
            self.retrieved_context,
            self.source_sequence,
            zero_history_policy=(
                zero_history_policy
            ),
        )
        _validate_retrieval_head_reduction(
            head_reduction=(
                head_reduction
            ),
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
        _validate_retrieval_computation_alignment(
            self.computation_provenance,
            self.source_sequence,
            query_lineage_fingerprint=(
                self.query_lineage_fingerprint
            ),
        )

        _require_nonempty_string(
            "retrieval_name",
            self.retrieval_name,
        )
        _require_nonempty_string(
            "schema_version",
            self.schema_version,
        )

    @property
    def node_count(
        self,
    ) -> int:
        return int(
            self
            .retrieved_context
            .shape[0]
        )

    @property
    def item_count(
        self,
    ) -> int:
        return self.node_count

    @property
    def retrieval_dim(
        self,
    ) -> int:
        return int(
            self
            .retrieved_context
            .shape[1]
        )

    @property
    def output_dim(
        self,
    ) -> int:
        return self.retrieval_dim

    @property
    def num_heads(
        self,
    ) -> int:
        return int(
            self
            .attention_weights
            .shape[1]
        )

    @property
    def sequence_length(
        self,
    ) -> int:
        return int(
            self
            .attention_weights
            .shape[2]
        )

    @property
    def context_shape(
        self,
    ) -> tuple[int, int]:
        return (
            self.node_count,
            self.retrieval_dim,
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
            .retrieved_context
            .device
        )

    @property
    def dtype(
        self,
    ) -> torch.dtype:
        return (
            self
            .retrieved_context
            .dtype
        )

    @property
    def source_sequence_encoding(
        self,
    ) -> TemporalSequenceEncoding:
        return self.source_sequence

    @property
    def encoded_sequence(
        self,
    ) -> torch.Tensor:
        return (
            self
            .source_sequence
            .encoded_sequence
        )

    @property
    def timestep_mask(
        self,
    ) -> torch.Tensor:
        return (
            self
            .source_sequence
            .timestep_mask
        )

    @property
    def node_axis(
        self,
    ):
        return (
            self
            .source_sequence
            .node_axis
        )

    @property
    def temporal_coordinates(
        self,
    ):
        return (
            self
            .source_sequence
            .temporal_coordinates
        )

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

    def alignment_fingerprint(
        self,
    ) -> str:
        return (
            self
            .source_sequence
            .alignment_fingerprint()
        )

    def temporal_alignment_fingerprint(
        self,
    ) -> str:
        return (
            self
            .source_sequence
            .temporal_alignment_fingerprint()
        )

    def semantic_dict(
        self,
    ) -> dict[str, Any]:
        return {
            "schema_version": (
                self.schema_version
            ),
            "retrieval_name": (
                self.retrieval_name
            ),
            "retrieval_kind": (
                self.retrieval_kind.value
            ),
            "head_reduction": (
                self.head_reduction.value
            ),
            "zero_history_policy": (
                self.zero_history_policy.value
            ),
            "context_shape": list(
                self.context_shape
            ),
            "weight_shape": list(
                self.weight_shape
            ),
            "has_head_reduction_weights": (
                self.head_reduction_weights
                is not None
            ),
            "query_alignment_fingerprint": (
                self.query_alignment_fingerprint
            ),
            "query_lineage_fingerprint": (
                self.query_lineage_fingerprint
            ),
            "weight_semantics": (
                TEMPORAL_QUERY_RETRIEVAL_WEIGHT_SEMANTICS
            ),
            "padding_policy": (
                TEMPORAL_QUERY_RETRIEVAL_PADDING_POLICY
            ),
            "normalization_policy": (
                TEMPORAL_QUERY_RETRIEVAL_NORMALIZATION_POLICY
            ),
            "absolute_tolerance": (
                self.absolute_tolerance
            ),
            "relative_tolerance": (
                self.relative_tolerance
            ),
            "source_sequence_lineage_fingerprint": (
                self
                .source_sequence
                .lineage_fingerprint()
            ),
            "source_alignment_fingerprint": (
                self
                .source_sequence
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
                HAZARD_QUERIED_MEMORY_SCIENTIFIC_INTERPRETATION
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
            "retrieved_context": (
                self.retrieved_context
            ),
            "attention_weights": (
                self.attention_weights
            ),
        }

        if self.head_reduction_weights is not None:
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
            "source_sequence_lineage_fingerprint": (
                self
                .source_sequence
                .lineage_fingerprint()
            ),
            "query_lineage_fingerprint": (
                self.query_lineage_fingerprint
            ),
            "query_alignment_fingerprint": (
                self.query_alignment_fingerprint
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

    def to(
        self,
        device: torch.device | str,
        *,
        source_sequence: (
            TemporalSequenceEncoding
            | None
        ) = None,
        non_blocking: bool = False,
    ) -> Self:
        """
        Move the retrieval contract while optionally injecting an already
        moved source sequence.

        ``source_sequence`` is used by ``HazardQueriedMemory.to`` to preserve
        exact identity with the moved ``UrbanMemory.sequence_encoding``.
        """

        moved_source = (
            self
            .source_sequence
            .to(
                device,
                dtype=self.dtype,
                non_blocking=non_blocking,
            )
            if source_sequence is None
            else source_sequence
        )

        if not isinstance(
            moved_source,
            TemporalSequenceEncoding,
        ):
            raise TypeError(
                "source_sequence must be a "
                "TemporalSequenceEncoding or None."
            )

        if (
            moved_source
            .lineage_fingerprint()
            != self
            .source_sequence
            .lineage_fingerprint()
        ):
            raise ValueError(
                "Injected source_sequence must preserve the original "
                "sequence lineage."
            )

        return type(self)(
            retrieved_context=(
                self
                .retrieved_context
                .to(
                    device=device,
                    non_blocking=(
                        non_blocking
                    ),
                )
            ),
            attention_weights=(
                self
                .attention_weights
                .to(
                    device=device,
                    non_blocking=(
                        non_blocking
                    ),
                )
            ),
            source_sequence=moved_source,
            query_alignment_fingerprint=(
                self.query_alignment_fingerprint
            ),
            query_lineage_fingerprint=(
                self.query_lineage_fingerprint
            ),
            retrieval_kind=(
                self.retrieval_kind
            ),
            head_reduction=(
                self.head_reduction
            ),
            zero_history_policy=(
                self.zero_history_policy
            ),
            computation_provenance=(
                self.computation_provenance
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
                if self.head_reduction_weights is not None
                else None
            ),
            absolute_tolerance=(
                self.absolute_tolerance
            ),
            relative_tolerance=(
                self.relative_tolerance
            ),
            retrieval_name=(
                self.retrieval_name
            ),
            schema_version=(
                self.schema_version
            ),
        )

    def replace(
        self,
        **changes: Any,
    ) -> Self:
        return dataclass_replace(
            self,
            **changes,
        )


# =============================================================================
# Hazard-query alignment
# =============================================================================


def _validate_hazard_query_type(
    source_hazard_query: HazardQueryEncoding,
) -> None:
    if not isinstance(
        source_hazard_query,
        HazardQueryEncoding,
    ):
        raise TypeError(
            "source_hazard_query must be a HazardQueryEncoding."
        )

    _require_nonempty_string(
        "source_hazard_query.lineage_fingerprint",
        source_hazard_query.lineage_fingerprint,
    )
    _require_nonempty_string(
        "source_hazard_query.query_encoder_architecture_fingerprint",
        source_hazard_query.query_encoder_architecture_fingerprint,
    )


def _resolve_hazard_query_scope(
    source_hazard_query: HazardQueryEncoding,
) -> HazardQueryAlignmentScope:
    source = source_hazard_query.source_embedding

    if isinstance(
        source,
        NodeAlignedHazardEmbeddingLookup,
    ):
        return HazardQueryAlignmentScope.NODE

    if isinstance(
        source,
        HazardEmbeddingLookup,
    ):
        return HazardQueryAlignmentScope.GRAPH_BROADCAST

    raise TypeError(
        "HazardQueryEncoding.source_embedding must be a "
        "HazardEmbeddingLookup or "
        "NodeAlignedHazardEmbeddingLookup."
    )


def _validate_node_hazard_query(
    *,
    source_urban_memory: UrbanMemory,
    source_hazard_query: HazardQueryEncoding,
    node_hazard_query: torch.Tensor,
) -> HazardQueryAlignmentScope:
    _require_float_matrix(
        "node_hazard_query",
        node_hazard_query,
    )

    if int(
        node_hazard_query.shape[0]
    ) != source_urban_memory.node_count:
        raise ValueError(
            "node_hazard_query must contain one row per urban-memory "
            "node."
        )

    if node_hazard_query.device != source_urban_memory.device:
        raise ValueError(
            "node_hazard_query and source_urban_memory must share one "
            "device."
        )

    if node_hazard_query.dtype != source_urban_memory.dtype:
        raise ValueError(
            "node_hazard_query and source_urban_memory must use one "
            "floating dtype."
        )

    source_query = source_hazard_query.query

    if source_query.device != source_urban_memory.device:
        raise ValueError(
            "source_hazard_query and source_urban_memory must share "
            "one device."
        )

    if source_query.dtype != source_urban_memory.dtype:
        raise ValueError(
            "source_hazard_query and source_urban_memory must use one "
            "floating dtype."
        )

    scope = _resolve_hazard_query_scope(
        source_hazard_query
    )
    source_embedding = (
        source_hazard_query
        .source_embedding
    )

    if scope == HazardQueryAlignmentScope.NODE:
        assert isinstance(
            source_embedding,
            NodeAlignedHazardEmbeddingLookup,
        )

        if source_hazard_query.item_count != source_urban_memory.node_count:
            raise ValueError(
                "Node-aligned hazard query rows must match the urban "
                "memory node count."
            )

        node_batch_index = (
            source_embedding
            .node_batch_index
        )

        if not isinstance(
            node_batch_index,
            torch.Tensor,
        ):
            raise TypeError(
                "Node-aligned hazard metadata must preserve "
                "node_batch_index."
            )

        if node_batch_index.dtype != torch.long:
            raise ValueError(
                "Hazard node_batch_index must use torch.long."
            )

        if tuple(
            node_batch_index.shape
        ) != (
            source_urban_memory.node_count,
        ):
            raise ValueError(
                "Hazard node_batch_index must have shape [N]."
            )

        if node_batch_index.device != source_urban_memory.device:
            raise ValueError(
                "Hazard node membership and urban memory must share "
                "one device."
            )

        if not torch.equal(
            node_batch_index,
            source_urban_memory.node_batch_index,
        ):
            raise ValueError(
                "Node-aligned hazard graph membership differs from "
                "urban-memory node_batch_index."
            )

        expected = source_query
    else:
        assert isinstance(
            source_embedding,
            HazardEmbeddingLookup,
        )

        if source_hazard_query.item_count != source_urban_memory.graph_count:
            raise ValueError(
                "Graph-scoped hazard query rows must match the packed "
                "urban-memory graph count."
            )

        expected = source_query[
            source_urban_memory
            .node_batch_index
        ]

    if not torch.equal(
        node_hazard_query,
        expected,
    ):
        raise ValueError(
            "node_hazard_query must equal the explicitly aligned "
            "HazardQueryEncoding query values."
        )

    return scope


def hazard_query_alignment_fingerprint(
    *,
    source_urban_memory: UrbanMemory,
    source_hazard_query: HazardQueryEncoding,
    node_hazard_query: torch.Tensor,
) -> str:
    """
    Return the exact node-aligned hazard-query identity.

    Validation is performed before fingerprinting.
    """

    _validate_hazard_query_type(
        source_hazard_query
    )
    scope = _validate_node_hazard_query(
        source_urban_memory=source_urban_memory,
        source_hazard_query=source_hazard_query,
        node_hazard_query=node_hazard_query,
    )

    return _fingerprint(
        {
            "alignment_policy": (
                HAZARD_QUERY_ALIGNMENT_POLICY
            ),
            "alignment_scope": (
                scope.value
            ),
            "source_hazard_query_lineage_fingerprint": (
                source_hazard_query
                .lineage_fingerprint
            ),
            "source_hazard_query_architecture_fingerprint": (
                source_hazard_query
                .query_encoder_architecture_fingerprint
            ),
            "urban_memory_node_axis_fingerprint": (
                source_urban_memory
                .node_axis
                .fingerprint()
            ),
            "urban_memory_graph_count": (
                source_urban_memory
                .graph_count
            ),
            "node_batch_index_value_fingerprint": (
                _tensor_fingerprint(
                    {
                        "node_batch_index": (
                            source_urban_memory
                            .node_batch_index
                        ),
                    }
                )
            ),
            "node_hazard_query_value_fingerprint": (
                _tensor_fingerprint(
                    {
                        "node_hazard_query": (
                            node_hazard_query
                        ),
                    }
                )
            ),
        }
    )


def _validate_hazard_memory_computation_alignment(
    *,
    computation_provenance: MemoryComputationProvenance,
    source_urban_memory: UrbanMemory,
    source_hazard_query: HazardQueryEncoding,
    retrieval: TemporalQueryRetrievalOutput,
) -> None:
    lineage = computation_provenance.lineage
    required_sources = {
        source_urban_memory.lineage_fingerprint(),
        source_hazard_query.lineage_fingerprint,
        retrieval.lineage_fingerprint(),
    }
    observed_sources = set(
        lineage.source_lineage_fingerprints
    )
    missing = sorted(
        required_sources
        - observed_sources
    )

    if missing:
        raise ValueError(
            "Hazard-memory fusion lineage must include urban memory, "
            "hazard query, and retrieval lineage. Missing: "
            f"{missing}."
        )

    expected_node_axis = (
        source_urban_memory
        .node_axis
        .fingerprint()
    )
    expected_temporal_axis = (
        source_urban_memory
        .temporal_alignment_fingerprint()
    )
    expected_feature_axis = (
        source_urban_memory
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
            "Hazard-memory lineage node_axis_fingerprint must match "
            "urban memory."
        )

    if (
        lineage.temporal_axis_fingerprint
        is not None
        and lineage.temporal_axis_fingerprint
        != expected_temporal_axis
    ):
        raise ValueError(
            "Hazard-memory lineage temporal_axis_fingerprint must "
            "match urban memory."
        )

    if (
        lineage.feature_axis_fingerprint
        is not None
        and lineage.feature_axis_fingerprint
        != expected_feature_axis
    ):
        raise ValueError(
            "Hazard-memory lineage feature_axis_fingerprint must "
            "match urban memory."
        )


# =============================================================================
# Hazard-queried fused memory
# =============================================================================


@dataclass(
    slots=True,
    frozen=True,
)
class HazardQueriedMemory:
    """
    Hazard-conditioned memory with preserved retrieval and fusion stages.

    The exact ``UrbanMemory``, ``HazardQueryEncoding``, and
    ``TemporalQueryRetrievalOutput`` objects remain attached.

    ``node_hazard_query`` records the explicit node-aligned query used by
    retrieval and fusion.

    ``fusion_components`` may preserve projected generic memory, projected
    query state, gates, residual states, or other named node-aligned
    intermediates. It is immutable and may be empty.
    """

    source_urban_memory: UrbanMemory
    source_hazard_query: HazardQueryEncoding
    node_hazard_query: torch.Tensor

    retrieval: TemporalQueryRetrievalOutput
    fused_memory: torch.Tensor

    fusion_policy: HazardMemoryFusionPolicy | str
    computation_provenance: (
        MemoryComputationProvenance
    )

    fusion_components: Mapping[
        str,
        torch.Tensor,
    ] = field(
        default_factory=dict
    )

    memory_name: str = (
        "hazard_queried_memory"
    )

    schema_version: str = (
        HAZARD_QUERIED_MEMORY_SCHEMA_VERSION
    )

    _alignment_scope: HazardQueryAlignmentScope = field(
        init=False,
        repr=False,
        compare=False,
    )

    def __post_init__(
        self,
    ) -> None:
        if not isinstance(
            self.source_urban_memory,
            UrbanMemory,
        ):
            raise TypeError(
                "source_urban_memory must be an UrbanMemory."
            )

        _validate_hazard_query_type(
            self.source_hazard_query
        )

        scope = _validate_node_hazard_query(
            source_urban_memory=(
                self.source_urban_memory
            ),
            source_hazard_query=(
                self.source_hazard_query
            ),
            node_hazard_query=(
                self.node_hazard_query
            ),
        )
        object.__setattr__(
            self,
            "_alignment_scope",
            scope,
        )

        if not isinstance(
            self.retrieval,
            TemporalQueryRetrievalOutput,
        ):
            raise TypeError(
                "retrieval must be a "
                "TemporalQueryRetrievalOutput."
            )

        if (
            self.retrieval.source_sequence
            is not self
            .source_urban_memory
            .sequence_encoding
        ):
            raise ValueError(
                "retrieval.source_sequence must be the exact same "
                "TemporalSequenceEncoding object preserved by "
                "source_urban_memory."
            )

        expected_query_alignment = (
            hazard_query_alignment_fingerprint(
                source_urban_memory=(
                    self.source_urban_memory
                ),
                source_hazard_query=(
                    self.source_hazard_query
                ),
                node_hazard_query=(
                    self.node_hazard_query
                ),
            )
        )

        if (
            self.retrieval
            .query_alignment_fingerprint
            != expected_query_alignment
        ):
            raise ValueError(
                "retrieval.query_alignment_fingerprint does not match "
                "the supplied node-aligned hazard query."
            )

        if (
            self.retrieval
            .query_lineage_fingerprint
            != self
            .source_hazard_query
            .lineage_fingerprint
        ):
            raise ValueError(
                "retrieval.query_lineage_fingerprint must match the "
                "complete source hazard query."
            )

        _require_float_matrix(
            "fused_memory",
            self.fused_memory,
        )

        if int(
            self.fused_memory.shape[0]
        ) != self.node_count:
            raise ValueError(
                "fused_memory must contain one row per urban-memory "
                "node."
            )

        if self.fused_memory.device != self.device:
            raise ValueError(
                "fused_memory and all source memory/query tensors must "
                "share one device."
            )

        if self.fused_memory.dtype != self.dtype:
            raise ValueError(
                "fused_memory and all source memory/query tensors must "
                "use one dtype."
            )

        if self.retrieval.device != self.device:
            raise ValueError(
                "retrieval and source_urban_memory must share one "
                "device."
            )

        if self.retrieval.dtype != self.dtype:
            raise ValueError(
                "retrieval and source_urban_memory must use one dtype."
            )

        fusion_policy = _normalize_fusion_policy(
            self.fusion_policy
        )
        object.__setattr__(
            self,
            "fusion_policy",
            fusion_policy,
        )

        if (
            fusion_policy
            == HazardMemoryFusionPolicy.RETRIEVED_ONLY
        ):
            if (
                self.fused_memory.shape
                != self
                .retrieval
                .retrieved_context
                .shape
                or not torch.equal(
                    self.fused_memory,
                    self
                    .retrieval
                    .retrieved_context,
                )
            ):
                raise ValueError(
                    "fusion_policy='retrieved_only' requires "
                    "fused_memory to equal retrieved_context exactly."
                )

        if not isinstance(
            self.computation_provenance,
            MemoryComputationProvenance,
        ):
            raise TypeError(
                "computation_provenance must be a "
                "MemoryComputationProvenance."
            )

        frozen_components = (
            _immutable_tensor_mapping(
                "fusion_components",
                self.fusion_components,
                node_count=self.node_count,
                device=self.device,
                dtype=self.dtype,
            )
        )
        object.__setattr__(
            self,
            "fusion_components",
            frozen_components,
        )

        _validate_hazard_memory_computation_alignment(
            computation_provenance=(
                self.computation_provenance
            ),
            source_urban_memory=(
                self.source_urban_memory
            ),
            source_hazard_query=(
                self.source_hazard_query
            ),
            retrieval=(
                self.retrieval
            ),
        )

        _require_nonempty_string(
            "memory_name",
            self.memory_name,
        )
        _require_nonempty_string(
            "schema_version",
            self.schema_version,
        )

    @property
    def alignment_scope(
        self,
    ) -> HazardQueryAlignmentScope:
        return self._alignment_scope

    @property
    def node_count(
        self,
    ) -> int:
        return (
            self
            .source_urban_memory
            .node_count
        )

    @property
    def item_count(
        self,
    ) -> int:
        return self.node_count

    @property
    def query_dim(
        self,
    ) -> int:
        return int(
            self
            .node_hazard_query
            .shape[1]
        )

    @property
    def retrieval_dim(
        self,
    ) -> int:
        return (
            self
            .retrieval
            .retrieval_dim
        )

    @property
    def fused_dim(
        self,
    ) -> int:
        return int(
            self
            .fused_memory
            .shape[1]
        )

    @property
    def output_dim(
        self,
    ) -> int:
        return self.fused_dim

    @property
    def device(
        self,
    ) -> torch.device:
        return (
            self
            .source_urban_memory
            .device
        )

    @property
    def dtype(
        self,
    ) -> torch.dtype:
        return (
            self
            .source_urban_memory
            .dtype
        )

    @property
    def sequence_encoding(
        self,
    ) -> TemporalSequenceEncoding:
        return (
            self
            .source_urban_memory
            .sequence_encoding
        )

    @property
    def encoded_sequence(
        self,
    ) -> torch.Tensor:
        return (
            self
            .source_urban_memory
            .encoded_sequence
        )

    @property
    def generic_pooled_memory(
        self,
    ) -> torch.Tensor | None:
        return (
            self
            .source_urban_memory
            .generic_pooled_memory
        )

    @property
    def retrieved_context(
        self,
    ) -> torch.Tensor:
        return (
            self
            .retrieval
            .retrieved_context
        )

    @property
    def temporal_retrieval_weights(
        self,
    ) -> torch.Tensor:
        return (
            self
            .retrieval
            .attention_weights
        )

    @property
    def timestep_mask(
        self,
    ) -> torch.Tensor:
        return (
            self
            .source_urban_memory
            .timestep_mask
        )

    @property
    def node_axis(
        self,
    ):
        return (
            self
            .source_urban_memory
            .node_axis
        )

    @property
    def feature_axis(
        self,
    ):
        return (
            self
            .source_urban_memory
            .feature_axis
        )

    @property
    def temporal_coordinates(
        self,
    ):
        return (
            self
            .source_urban_memory
            .temporal_coordinates
        )

    @property
    def node_batch_index(
        self,
    ) -> torch.Tensor:
        return (
            self
            .source_urban_memory
            .node_batch_index
        )

    @property
    def graph_count(
        self,
    ) -> int:
        return (
            self
            .source_urban_memory
            .graph_count
        )

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

    def query_alignment_fingerprint(
        self,
    ) -> str:
        return hazard_query_alignment_fingerprint(
            source_urban_memory=(
                self.source_urban_memory
            ),
            source_hazard_query=(
                self.source_hazard_query
            ),
            node_hazard_query=(
                self.node_hazard_query
            ),
        )

    def alignment_fingerprint(
        self,
    ) -> str:
        return (
            self
            .source_urban_memory
            .alignment_fingerprint()
        )

    def temporal_alignment_fingerprint(
        self,
    ) -> str:
        return (
            self
            .source_urban_memory
            .temporal_alignment_fingerprint()
        )

    def semantic_dict(
        self,
    ) -> dict[str, Any]:
        return {
            "schema_version": (
                self.schema_version
            ),
            "memory_name": (
                self.memory_name
            ),
            "fusion_policy": (
                self.fusion_policy.value
            ),
            "alignment_scope": (
                self.alignment_scope.value
            ),
            "query_dim": (
                self.query_dim
            ),
            "retrieval_dim": (
                self.retrieval_dim
            ),
            "fused_dim": (
                self.fused_dim
            ),
            "fusion_component_names": sorted(
                self.fusion_components
            ),
            "query_alignment_policy": (
                HAZARD_QUERY_ALIGNMENT_POLICY
            ),
            "query_alignment_fingerprint": (
                self
                .query_alignment_fingerprint()
            ),
            "source_urban_memory_lineage_fingerprint": (
                self
                .source_urban_memory
                .lineage_fingerprint()
            ),
            "source_hazard_query_lineage_fingerprint": (
                self
                .source_hazard_query
                .lineage_fingerprint
            ),
            "retrieval_lineage_fingerprint": (
                self
                .retrieval
                .lineage_fingerprint()
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
                HAZARD_QUERIED_MEMORY_SCIENTIFIC_INTERPRETATION
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
            "node_hazard_query": (
                self.node_hazard_query
            ),
            "fused_memory": (
                self.fused_memory
            ),
        }

        for name, tensor in (
            self.fusion_components.items()
        ):
            tensors[
                f"fusion_component::{name}"
            ] = tensor

        return _fingerprint(
            {
                "urban_memory_value_fingerprint": (
                    self
                    .source_urban_memory
                    .value_fingerprint()
                ),
                "retrieval_value_fingerprint": (
                    self
                    .retrieval
                    .value_fingerprint()
                ),
                "local_tensor_fingerprint": (
                    _tensor_fingerprint(
                        tensors
                    )
                ),
            }
        )

    def lineage_dict(
        self,
    ) -> dict[str, Any]:
        return {
            "schema_version": (
                self.schema_version
            ),
            "source_urban_memory_lineage_fingerprint": (
                self
                .source_urban_memory
                .lineage_fingerprint()
            ),
            "source_hazard_query_lineage_fingerprint": (
                self
                .source_hazard_query
                .lineage_fingerprint
            ),
            "retrieval_lineage_fingerprint": (
                self
                .retrieval
                .lineage_fingerprint()
            ),
            "computation_provenance_fingerprint": (
                self
                .computation_provenance
                .fingerprint()
            ),
            "query_alignment_fingerprint": (
                self
                .query_alignment_fingerprint()
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

    def to(
        self,
        device: torch.device | str,
        *,
        non_blocking: bool = False,
    ) -> Self:
        """
        Move the complete hazard-queried-memory contract.

        The metadata-preserving hazard query must already be on the requested
        device. This schema does not reconstruct hazard embedding vocabulary
        objects or query intermediates owned by the hazard subsystem.
        """

        target = torch.device(
            device
        )

        if (
            self
            .source_hazard_query
            .query
            .device
            != target
        ):
            raise ValueError(
                "Move the metadata-preserving hazard embedding/query "
                "upstream before moving HazardQueriedMemory."
            )

        moved_urban_memory = (
            self
            .source_urban_memory
            .to(
                target,
                non_blocking=non_blocking,
            )
        )

        moved_retrieval = (
            self
            .retrieval
            .to(
                target,
                source_sequence=(
                    moved_urban_memory
                    .sequence_encoding
                ),
                non_blocking=non_blocking,
            )
        )

        moved_components = {
            name: tensor.to(
                device=target,
                non_blocking=non_blocking,
            )
            for name, tensor in (
                self.fusion_components.items()
            )
        }

        return type(self)(
            source_urban_memory=(
                moved_urban_memory
            ),
            source_hazard_query=(
                self.source_hazard_query
            ),
            node_hazard_query=(
                self
                .node_hazard_query
                .to(
                    device=target,
                    non_blocking=non_blocking,
                )
            ),
            retrieval=moved_retrieval,
            fused_memory=(
                self
                .fused_memory
                .to(
                    device=target,
                    non_blocking=non_blocking,
                )
            ),
            fusion_policy=(
                self.fusion_policy
            ),
            computation_provenance=(
                self.computation_provenance
            ),
            fusion_components=(
                moved_components
            ),
            memory_name=(
                self.memory_name
            ),
            schema_version=(
                self.schema_version
            ),
        )

    def replace(
        self,
        **changes: Any,
    ) -> Self:
        return dataclass_replace(
            self,
            **changes,
        )


# =============================================================================
# Compact aliases
# =============================================================================


TemporalRetrievalOutput = TemporalQueryRetrievalOutput
QueryRetrievalOutput = TemporalQueryRetrievalOutput
HazardConditionedMemory = HazardQueriedMemory


# =============================================================================
# Public API
# =============================================================================


__all__ = (
    # Schema identity and interpretation.
    "TEMPORAL_QUERY_RETRIEVAL_OUTPUT_SCHEMA_VERSION",
    "HAZARD_QUERIED_MEMORY_SCHEMA_VERSION",
    "TEMPORAL_QUERY_RETRIEVAL_WEIGHT_SEMANTICS",
    "TEMPORAL_QUERY_RETRIEVAL_PADDING_POLICY",
    "TEMPORAL_QUERY_RETRIEVAL_NORMALIZATION_POLICY",
    "HAZARD_QUERY_ALIGNMENT_POLICY",
    "HAZARD_QUERIED_MEMORY_SCIENTIFIC_INTERPRETATION",

    # Controlled vocabularies.
    "TemporalQueryRetrievalKind",
    "TemporalQueryRetrievalHeadReduction",
    "TemporalQueryRetrievalZeroHistoryPolicy",
    "HazardQueryAlignmentScope",
    "HazardMemoryFusionPolicy",
    "CANONICAL_TEMPORAL_QUERY_RETRIEVAL_KINDS",
    "CANONICAL_TEMPORAL_QUERY_RETRIEVAL_HEAD_REDUCTIONS",
    "CANONICAL_TEMPORAL_QUERY_RETRIEVAL_ZERO_HISTORY_POLICIES",
    "CANONICAL_HAZARD_QUERY_ALIGNMENT_SCOPES",
    "CANONICAL_HAZARD_MEMORY_FUSION_POLICIES",

    # Query-neutral retrieval.
    "TemporalQueryRetrievalOutput",
    "TemporalRetrievalOutput",
    "QueryRetrievalOutput",

    # Hazard-specific alignment and fused memory.
    "hazard_query_alignment_fingerprint",
    "HazardQueriedMemory",
    "HazardConditionedMemory",
)
