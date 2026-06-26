"""
Shared sequence-encoding contract for temporal memory.

Target path:
    urban_resilience_models/
        v2_hazard_conditioned_functional_ugnn/
            memory/
                schemas/
                    sequence_encoding.py

This module freezes the common public output contract shared by temporal
sequence encoders such as:

- identity or pass-through sequence baselines;
- GRU encoders;
- LSTM encoders;
- Transformer encoders;
- future sequence-preserving temporal encoders.

The authoritative encoded tensor is:

    encoded_sequence: [N, T, H]

where:

- ``N`` is the preserved node/item axis;
- ``T`` is the preserved temporal axis;
- ``H`` is the encoder output dimension.

The exact ``HistoricalSequenceInputs`` object is retained. The timestep mask,
node axis, feature axis, temporal coordinates, and source-data provenance are
therefore not copied into detached parallel fields. They remain available
through read-only convenience properties.

Shared-contract boundary
------------------------
This schema intentionally does not define a generic final recurrent state.

A GRU hidden state, an LSTM hidden/cell state, a bidirectional recurrent
summary, and a Transformer last-token representation are not equivalent. 
Encoder-specific final states may be retained by local diagnostics
or implementation-specific contracts, but downstream shared modules must use:

- ``encoded_sequence`` for sequence-aware processing; or
- an explicit ``TemporalPoolingOutput`` for a generic ``[N, H]`` summary.

Padding policy
--------------
Encoded values at padded temporal positions must be exactly zero. This gives
pooling and retrieval modules a deterministic leakage boundary and prevents
masked positions from carrying hidden information.

Zero-history rows are permitted only when the source history explicitly allows
them. Their entire encoded row must therefore be zero.

Provenance
----------
The output preserves:

- the exact source-history object;
- node, temporal, and source-feature alignment;
- architecture provenance;
- optional parameter-snapshot provenance;
- execution lineage.

Parameter-snapshot provenance remains optional during ordinary training-time
forwards.

Interpretation
--------------------------
An encoded sequence is a learned model representation. It does not by itself
establish causal temporal influence, calibrated uncertainty, mechanistic
interpretability, or real-world historical importance.
"""

from __future__ import annotations

from dataclasses import dataclass
from dataclasses import replace as dataclass_replace
from enum import StrEnum
from hashlib import sha256
import json
from typing import Any, Final, Mapping, Self

import torch

from .history_inputs import (
    HistoricalSequenceInputs,
)
from .provenance import (
    MemoryComputationProvenance,
)


# =============================================================================
# Schema identity and fixed interpretation
# =============================================================================


TEMPORAL_SEQUENCE_ENCODING_SCHEMA_VERSION: Final[str] = "0.1"

TEMPORAL_SEQUENCE_ENCODING_VALUE_SEMANTICS: Final[str] = (
    "learned_or_deterministic_sequence_representation"
)

TEMPORAL_SEQUENCE_ENCODING_PADDING_POLICY: Final[str] = (
    "exact_zero_at_padded_positions"
)

TEMPORAL_SEQUENCE_ENCODING_SCIENTIFIC_INTERPRETATION: Final[str] = (
    "model_representation_not_causal_or_mechanistic_explanation"
)


# =============================================================================
# Controlled encoder-kind vocabulary
# =============================================================================


class TemporalSequenceEncoderKind(StrEnum):
    """
    Semantic family of the sequence-preserving encoder.

    This vocabulary describes representation semantics. It does not claim that
    a corresponding implementation is currently available.
    """

    IDENTITY_SEQUENCE = "identity_sequence"
    GRU = "gru"
    LSTM = "lstm"
    TRANSFORMER = "transformer"
    TEMPORAL_MLP = "temporal_mlp"
    OTHER = "other"


CANONICAL_TEMPORAL_SEQUENCE_ENCODER_KINDS: Final[
    tuple[str, ...]
] = tuple(
    value.value
    for value in TemporalSequenceEncoderKind
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


def _normalize_encoder_kind(
    value: TemporalSequenceEncoderKind | str,
) -> TemporalSequenceEncoderKind:
    if isinstance(
        value,
        TemporalSequenceEncoderKind,
    ):
        return value

    return TemporalSequenceEncoderKind(
        value
    )


def _require_encoded_sequence(
    value: torch.Tensor,
) -> None:
    if not isinstance(
        value,
        torch.Tensor,
    ):
        raise TypeError(
            "encoded_sequence must be a tensor."
        )

    if value.ndim != 3:
        raise ValueError(
            "encoded_sequence must have shape [N, T, H]; "
            f"observed {tuple(value.shape)}."
        )

    node_count = int(
        value.shape[0]
    )
    sequence_length = int(
        value.shape[1]
    )
    hidden_dim = int(
        value.shape[2]
    )

    if (
        node_count <= 0
        or sequence_length <= 0
        or hidden_dim <= 0
    ):
        raise ValueError(
            "encoded_sequence dimensions N, T, and H must all be "
            "strictly positive."
        )

    if not value.dtype.is_floating_point:
        raise ValueError(
            "encoded_sequence must use a floating-point dtype."
        )

    if not bool(
        torch.isfinite(
            value
        ).all().item()
    ):
        raise ValueError(
            "encoded_sequence must contain only finite values."
        )


def _validate_encoded_padding(
    encoded_sequence: torch.Tensor,
    timestep_mask: torch.Tensor,
) -> None:
    padding_mask = (
        ~timestep_mask
    ).unsqueeze(
        -1
    ).expand_as(
        encoded_sequence
    )

    padded_values = encoded_sequence[
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
            "encoded_sequence values at padded timesteps must be "
            "exactly zero."
        )


def _validate_source_alignment(
    encoded_sequence: torch.Tensor,
    source_history: HistoricalSequenceInputs,
) -> None:
    if int(
        encoded_sequence.shape[0]
    ) != source_history.node_count:
        raise ValueError(
            "encoded_sequence dimension N must match source_history."
        )

    if int(
        encoded_sequence.shape[1]
    ) != source_history.sequence_length:
        raise ValueError(
            "encoded_sequence dimension T must match source_history."
        )

    if encoded_sequence.device != source_history.device:
        raise ValueError(
            "encoded_sequence and source_history must share one device."
        )


def _validate_computation_alignment(
    computation_provenance: MemoryComputationProvenance,
    source_history: HistoricalSequenceInputs,
) -> None:
    lineage = computation_provenance.lineage

    expected_node_axis = (
        source_history
        .node_axis
        .fingerprint()
    )
    expected_temporal_axis = (
        source_history
        .temporal_alignment_fingerprint()
    )
    expected_feature_axis = (
        source_history
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
            "Computation lineage node_axis_fingerprint must match "
            "the source history node axis."
        )

    if (
        lineage.temporal_axis_fingerprint
        is not None
        and lineage.temporal_axis_fingerprint
        != expected_temporal_axis
    ):
        raise ValueError(
            "Computation lineage temporal_axis_fingerprint must match "
            "the source history temporal alignment."
        )

    if (
        lineage.feature_axis_fingerprint
        is not None
        and lineage.feature_axis_fingerprint
        != expected_feature_axis
    ):
        raise ValueError(
            "Computation lineage feature_axis_fingerprint must match "
            "the source history feature axis."
        )

    source_lineage = (
        source_history
        .lineage_fingerprint()
    )

    if source_lineage not in (
        lineage
        .source_lineage_fingerprints
    ):
        raise ValueError(
            "Computation lineage must include the exact source-history "
            "lineage fingerprint."
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
            "encoded_sequence dtype must remain floating-point."
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
# Shared sequence-encoding contract
# =============================================================================


@dataclass(
    slots=True,
    frozen=True,
)
class TemporalSequenceEncoding:
    """
    Shared GRU/LSTM/Transformer-neutral temporal encoding.

    Parameters
    ----------
    encoded_sequence:
        Finite floating tensor ``[N, T, H]``.

    source_history:
        The exact ``HistoricalSequenceInputs`` object consumed by the encoder.

    encoder_kind:
        Semantic encoder family.

    computation_provenance:
        Architecture, optional parameter snapshot, and execution lineage.

    encoding_name:
        Human-readable identity of this representation.

    Notes
    -----
    The public contract intentionally contains no generic final hidden state.
    A downstream ``[N, H]`` summary must be produced by an explicit pooling
    policy.
    """

    encoded_sequence: torch.Tensor
    source_history: HistoricalSequenceInputs

    encoder_kind: (
        TemporalSequenceEncoderKind
        | str
    )
    computation_provenance: (
        MemoryComputationProvenance
    )

    encoding_name: str = (
        "temporal_sequence_encoding"
    )

    schema_version: str = (
        TEMPORAL_SEQUENCE_ENCODING_SCHEMA_VERSION
    )

    def __post_init__(
        self,
    ) -> None:
        _require_encoded_sequence(
            self.encoded_sequence
        )

        if not isinstance(
            self.source_history,
            HistoricalSequenceInputs,
        ):
            raise TypeError(
                "source_history must be a HistoricalSequenceInputs."
            )

        encoder_kind = (
            _normalize_encoder_kind(
                self.encoder_kind
            )
        )
        object.__setattr__(
            self,
            "encoder_kind",
            encoder_kind,
        )

        if not isinstance(
            self.computation_provenance,
            MemoryComputationProvenance,
        ):
            raise TypeError(
                "computation_provenance must be a "
                "MemoryComputationProvenance."
            )

        _validate_source_alignment(
            self.encoded_sequence,
            self.source_history,
        )
        _validate_encoded_padding(
            self.encoded_sequence,
            self.timestep_mask,
        )
        _validate_computation_alignment(
            self.computation_provenance,
            self.source_history,
        )

        _require_nonempty_string(
            "encoding_name",
            self.encoding_name,
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
            .encoded_sequence
            .shape[0]
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
            self
            .encoded_sequence
            .shape[1]
        )

    @property
    def hidden_dim(
        self,
    ) -> int:
        return int(
            self
            .encoded_sequence
            .shape[2]
        )

    @property
    def encoded_shape(
        self,
    ) -> tuple[int, int, int]:
        return (
            self.node_count,
            self.sequence_length,
            self.hidden_dim,
        )

    @property
    def device(
        self,
    ) -> torch.device:
        return (
            self
            .encoded_sequence
            .device
        )

    @property
    def dtype(
        self,
    ) -> torch.dtype:
        return (
            self
            .encoded_sequence
            .dtype
        )

    # -------------------------------------------------------------------------
    # Exact source-contract preservation
    # -------------------------------------------------------------------------

    @property
    def timestep_mask(
        self,
    ) -> torch.Tensor:
        return (
            self
            .source_history
            .timestep_mask
        )

    @property
    def valid_lengths(
        self,
    ) -> torch.Tensor:
        return (
            self
            .source_history
            .valid_lengths
        )

    @property
    def node_axis(
        self,
    ):
        return (
            self
            .source_history
            .node_axis
        )

    @property
    def feature_axis(
        self,
    ):
        return (
            self
            .source_history
            .feature_axis
        )

    @property
    def temporal_coordinates(
        self,
    ):
        return (
            self
            .source_history
            .temporal_coordinates
        )

    @property
    def source_provenance(
        self,
    ):
        return (
            self
            .source_history
            .source_provenance
        )

    @property
    def node_ids(
        self,
    ) -> tuple[str, ...]:
        return (
            self
            .source_history
            .node_ids
        )

    @property
    def node_batch_index(
        self,
    ) -> torch.Tensor:
        return (
            self
            .source_history
            .node_batch_index
        )

    @property
    def graph_count(
        self,
    ) -> int:
        return (
            self
            .source_history
            .graph_count
        )

    @property
    def graph_ids(
        self,
    ) -> tuple[str, ...]:
        return (
            self
            .source_history
            .graph_ids
        )

    @property
    def has_zero_history(
        self,
    ) -> bool:
        return (
            self
            .source_history
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
            .source_history
            .alignment_fingerprint()
        )

    def temporal_alignment_fingerprint(
        self,
    ) -> str:
        return (
            self
            .source_history
            .temporal_alignment_fingerprint()
        )

    def semantic_dict(
        self,
    ) -> dict[str, Any]:
        return {
            "schema_version": (
                self.schema_version
            ),
            "encoding_name": (
                self.encoding_name
            ),
            "encoder_kind": (
                self.encoder_kind.value
            ),
            "encoded_shape": list(
                self.encoded_shape
            ),
            "value_semantics": (
                TEMPORAL_SEQUENCE_ENCODING_VALUE_SEMANTICS
            ),
            "padding_policy": (
                TEMPORAL_SEQUENCE_ENCODING_PADDING_POLICY
            ),
            "source_history_lineage_fingerprint": (
                self
                .source_history
                .lineage_fingerprint()
            ),
            "source_alignment_fingerprint": (
                self
                .source_history
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
                TEMPORAL_SEQUENCE_ENCODING_SCIENTIFIC_INTERPRETATION
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
                "encoded_sequence": (
                    self
                    .encoded_sequence
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
            "source_history_lineage_fingerprint": (
                self
                .source_history
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
        dtype: torch.dtype | None = None,
        non_blocking: bool = False,
    ) -> Self:
        """
        Move the complete encoding contract to one device.

        When ``dtype`` is supplied, both the encoded sequence and the
        source-history values are converted to that floating dtype. Integer
        temporal coordinates and masks retain their required dtypes.
        """

        _validate_dtype_request(
            dtype
        )

        target_dtype = (
            self.dtype
            if dtype is None
            else dtype
        )

        moved_source = (
            self
            .source_history
            .to(
                device,
                dtype=target_dtype,
                non_blocking=non_blocking,
            )
        )

        moved_encoded = (
            self
            .encoded_sequence
            .to(
                device=device,
                dtype=target_dtype,
                non_blocking=non_blocking,
            )
        )

        return type(self)(
            encoded_sequence=moved_encoded,
            source_history=moved_source,
            encoder_kind=self.encoder_kind,
            computation_provenance=(
                self.computation_provenance
            ),
            encoding_name=self.encoding_name,
            schema_version=self.schema_version,
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


SequenceEncoding = TemporalSequenceEncoding
SharedTemporalSequenceEncoding = TemporalSequenceEncoding


# =============================================================================
# Public API
# =============================================================================


__all__ = (
    # Schema identity and interpretation.
    "TEMPORAL_SEQUENCE_ENCODING_SCHEMA_VERSION",
    "TEMPORAL_SEQUENCE_ENCODING_VALUE_SEMANTICS",
    "TEMPORAL_SEQUENCE_ENCODING_PADDING_POLICY",
    "TEMPORAL_SEQUENCE_ENCODING_SCIENTIFIC_INTERPRETATION",

    # Encoder-kind vocabulary.
    "TemporalSequenceEncoderKind",
    "CANONICAL_TEMPORAL_SEQUENCE_ENCODER_KINDS",

    # Main contract and aliases.
    "TemporalSequenceEncoding",
    "SequenceEncoding",
    "SharedTemporalSequenceEncoding",
)
