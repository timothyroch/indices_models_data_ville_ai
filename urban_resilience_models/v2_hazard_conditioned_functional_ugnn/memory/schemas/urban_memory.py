"""
Shared urban-memory orchestration contract.

Target path:
    urban_resilience_models/
        v2_hazard_conditioned_functional_ugnn/
            memory/
                schemas/
                    urban_memory.py

This module freezes the public contract that combines:

- one preserved ``TemporalSequenceEncoding``;
- optional hazard-independent ``TemporalPoolingOutput``;
- explicit urban-memory assembly semantics;
- architecture, optional parameter-snapshot, and execution provenance.

The core rule is:

    the encoded sequence is always preserved

even when a generic pooled memory exists.

An ``UrbanMemory`` object therefore never replaces ``[N, T, H]`` with a
detached ``[N, P]`` tensor. Sequence-aware hazard retrieval, diagnostics,
counterfactual analyses, and future temporal modules retain access to the
complete encoded history.

Runtime object identity
-----------------------
When temporal pooling is present:

    urban_memory.temporal_pooling.source_encoding
        is urban_memory.sequence_encoding

This exact identity requirement prevents an orchestration object from
combining a pooled state with a different, merely shape-compatible sequence.

After serialization and deserialization, Python object identity is not a
portable guarantee. Artifact compatibility should then be checked through
semantic, value, alignment, and lineage fingerprints.

Assembly policies
-----------------
``sequence_only``
    Preserve the encoded sequence without a generic pooled memory.

``sequence_with_generic_pooling``
    Preserve the encoded sequence and one hazard-independent temporal pooling
    result.

No generic ``final_state`` is introduced. The optional pooled representation
is available explicitly through ``temporal_pooling.pooled_memory``.

Hazard boundary
---------------
This schema is hazard-neutral. Hazard-conditioned temporal retrieval belongs
in ``hazard_queried_memory.py`` and must consume the preserved unpooled
sequence rather than reinterpret the generic pooled state as the full history.

Interpretation
--------------------------
Urban memory is a model representation and software lineage object. It does
not by itself establish causal historical influence, faithful explanation,
data sufficiency, or real-world mechanistic memory.
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
    MemoryComputationProvenance,
)
from .sequence_encoding import (
    TemporalSequenceEncoding,
)
from .temporal_pooling import (
    TemporalPoolingOutput,
)


# =============================================================================
# Schema identity and fixed interpretation
# =============================================================================


URBAN_MEMORY_SCHEMA_VERSION: Final[str] = "0.1"

URBAN_MEMORY_SEQUENCE_PRESERVATION_POLICY: Final[str] = (
    "always_preserve_exact_temporal_sequence_encoding"
)

URBAN_MEMORY_POOLING_SCOPE: Final[str] = (
    "hazard_independent_generic_temporal_pooling"
)

URBAN_MEMORY_SCIENTIFIC_INTERPRETATION: Final[str] = (
    "model_memory_representation_not_causal_or_mechanistic_memory"
)


# =============================================================================
# Controlled orchestration vocabulary
# =============================================================================


class UrbanMemoryAssemblyPolicy(StrEnum):
    """Declared components retained by one urban-memory object."""

    SEQUENCE_ONLY = "sequence_only"
    SEQUENCE_WITH_GENERIC_POOLING = (
        "sequence_with_generic_pooling"
    )


CANONICAL_URBAN_MEMORY_ASSEMBLY_POLICIES: Final[
    tuple[str, ...]
] = tuple(
    value.value
    for value in UrbanMemoryAssemblyPolicy
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


def _normalize_assembly_policy(
    value: UrbanMemoryAssemblyPolicy | str,
) -> UrbanMemoryAssemblyPolicy:
    if isinstance(
        value,
        UrbanMemoryAssemblyPolicy,
    ):
        return value

    return UrbanMemoryAssemblyPolicy(
        value
    )


def _validate_assembly_contract(
    *,
    assembly_policy: UrbanMemoryAssemblyPolicy,
    temporal_pooling: TemporalPoolingOutput | None,
) -> None:
    if (
        assembly_policy
        == UrbanMemoryAssemblyPolicy.SEQUENCE_ONLY
    ):
        if temporal_pooling is not None:
            raise ValueError(
                "assembly_policy='sequence_only' requires "
                "temporal_pooling to be None."
            )

        return

    if temporal_pooling is None:
        raise ValueError(
            "assembly_policy='sequence_with_generic_pooling' requires "
            "a TemporalPoolingOutput."
        )


def _validate_exact_pooling_source(
    *,
    sequence_encoding: TemporalSequenceEncoding,
    temporal_pooling: TemporalPoolingOutput | None,
) -> None:
    if temporal_pooling is None:
        return

    if (
        temporal_pooling.source_encoding
        is not sequence_encoding
    ):
        raise ValueError(
            "temporal_pooling.source_encoding must be the exact same "
            "TemporalSequenceEncoding object as sequence_encoding."
        )


def _validate_computation_alignment(
    *,
    computation_provenance: MemoryComputationProvenance,
    sequence_encoding: TemporalSequenceEncoding,
    temporal_pooling: TemporalPoolingOutput | None,
) -> None:
    lineage = computation_provenance.lineage

    required_sources = {
        sequence_encoding.lineage_fingerprint(),
    }

    if temporal_pooling is not None:
        required_sources.add(
            temporal_pooling.lineage_fingerprint()
        )

    observed_sources = set(
        lineage.source_lineage_fingerprints
    )
    missing_sources = sorted(
        required_sources
        - observed_sources
    )

    if missing_sources:
        raise ValueError(
            "Urban-memory computation lineage must include every "
            "preserved source lineage. Missing: "
            f"{missing_sources}."
        )

    expected_node_axis = (
        sequence_encoding
        .node_axis
        .fingerprint()
    )
    expected_temporal_axis = (
        sequence_encoding
        .temporal_alignment_fingerprint()
    )
    expected_feature_axis = (
        sequence_encoding
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
            "Urban-memory lineage node_axis_fingerprint must match "
            "the preserved sequence."
        )

    if (
        lineage.temporal_axis_fingerprint
        is not None
        and lineage.temporal_axis_fingerprint
        != expected_temporal_axis
    ):
        raise ValueError(
            "Urban-memory lineage temporal_axis_fingerprint must "
            "match the preserved sequence."
        )

    if (
        lineage.feature_axis_fingerprint
        is not None
        and lineage.feature_axis_fingerprint
        != expected_feature_axis
    ):
        raise ValueError(
            "Urban-memory lineage feature_axis_fingerprint must match "
            "the preserved sequence."
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


# =============================================================================
# Shared urban-memory contract
# =============================================================================


@dataclass(
    slots=True,
    frozen=True,
)
class UrbanMemory:
    """
    Preserved temporal sequence with optional generic temporal pooling.

    Parameters
    ----------
    sequence_encoding:
        The authoritative ``TemporalSequenceEncoding``. This object is always
        retained regardless of the assembly policy.

    assembly_policy:
        Whether the memory preserves only the sequence or both the sequence
        and generic hazard-independent pooling.

    computation_provenance:
        Urban-memory orchestration architecture, optional parameter snapshot,
        and execution lineage.

    temporal_pooling:
        Optional ``TemporalPoolingOutput``. When supplied, its
        ``source_encoding`` must be the exact same object as
        ``sequence_encoding``.
    """

    sequence_encoding: TemporalSequenceEncoding
    assembly_policy: (
        UrbanMemoryAssemblyPolicy
        | str
    )
    computation_provenance: (
        MemoryComputationProvenance
    )

    temporal_pooling: (
        TemporalPoolingOutput
        | None
    ) = None

    memory_name: str = (
        "urban_memory"
    )

    schema_version: str = (
        URBAN_MEMORY_SCHEMA_VERSION
    )

    def __post_init__(
        self,
    ) -> None:
        if not isinstance(
            self.sequence_encoding,
            TemporalSequenceEncoding,
        ):
            raise TypeError(
                "sequence_encoding must be a "
                "TemporalSequenceEncoding."
            )

        assembly_policy = (
            _normalize_assembly_policy(
                self.assembly_policy
            )
        )
        object.__setattr__(
            self,
            "assembly_policy",
            assembly_policy,
        )

        if (
            self.temporal_pooling
            is not None
            and not isinstance(
                self.temporal_pooling,
                TemporalPoolingOutput,
            )
        ):
            raise TypeError(
                "temporal_pooling must be a TemporalPoolingOutput "
                "or None."
            )

        if not isinstance(
            self.computation_provenance,
            MemoryComputationProvenance,
        ):
            raise TypeError(
                "computation_provenance must be a "
                "MemoryComputationProvenance."
            )

        _validate_assembly_contract(
            assembly_policy=assembly_policy,
            temporal_pooling=(
                self.temporal_pooling
            ),
        )
        _validate_exact_pooling_source(
            sequence_encoding=(
                self.sequence_encoding
            ),
            temporal_pooling=(
                self.temporal_pooling
            ),
        )
        _validate_computation_alignment(
            computation_provenance=(
                self.computation_provenance
            ),
            sequence_encoding=(
                self.sequence_encoding
            ),
            temporal_pooling=(
                self.temporal_pooling
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

    # -------------------------------------------------------------------------
    # Structural properties
    # -------------------------------------------------------------------------

    @property
    def node_count(
        self,
    ) -> int:
        return (
            self
            .sequence_encoding
            .node_count
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
        return (
            self
            .sequence_encoding
            .sequence_length
        )

    @property
    def sequence_hidden_dim(
        self,
    ) -> int:
        return (
            self
            .sequence_encoding
            .hidden_dim
        )

    @property
    def encoded_shape(
        self,
    ) -> tuple[int, int, int]:
        return (
            self
            .sequence_encoding
            .encoded_shape
        )

    @property
    def pooled_dim(
        self,
    ) -> int | None:
        if self.temporal_pooling is None:
            return None

        return (
            self
            .temporal_pooling
            .pooled_dim
        )

    @property
    def device(
        self,
    ) -> torch.device:
        return (
            self
            .sequence_encoding
            .device
        )

    @property
    def dtype(
        self,
    ) -> torch.dtype:
        return (
            self
            .sequence_encoding
            .dtype
        )

    @property
    def has_temporal_pooling(
        self,
    ) -> bool:
        return (
            self.temporal_pooling
            is not None
        )

    @property
    def has_generic_pooled_memory(
        self,
    ) -> bool:
        return self.has_temporal_pooling

    @property
    def has_zero_history(
        self,
    ) -> bool:
        return (
            self
            .sequence_encoding
            .has_zero_history
        )

    # -------------------------------------------------------------------------
    # Exact preserved source access
    # -------------------------------------------------------------------------

    @property
    def encoded_sequence(
        self,
    ) -> torch.Tensor:
        return (
            self
            .sequence_encoding
            .encoded_sequence
        )

    @property
    def source_history(
        self,
    ):
        return (
            self
            .sequence_encoding
            .source_history
        )

    @property
    def timestep_mask(
        self,
    ) -> torch.Tensor:
        return (
            self
            .sequence_encoding
            .timestep_mask
        )

    @property
    def valid_lengths(
        self,
    ) -> torch.Tensor:
        return (
            self
            .sequence_encoding
            .valid_lengths
        )

    @property
    def node_axis(
        self,
    ):
        return (
            self
            .sequence_encoding
            .node_axis
        )

    @property
    def feature_axis(
        self,
    ):
        return (
            self
            .sequence_encoding
            .feature_axis
        )

    @property
    def temporal_coordinates(
        self,
    ):
        return (
            self
            .sequence_encoding
            .temporal_coordinates
        )

    @property
    def node_ids(
        self,
    ) -> tuple[str, ...]:
        return (
            self
            .sequence_encoding
            .node_ids
        )

    @property
    def node_batch_index(
        self,
    ) -> torch.Tensor:
        return (
            self
            .sequence_encoding
            .node_batch_index
        )

    @property
    def graph_count(
        self,
    ) -> int:
        return (
            self
            .sequence_encoding
            .graph_count
        )

    @property
    def graph_ids(
        self,
    ) -> tuple[str, ...]:
        return (
            self
            .sequence_encoding
            .graph_ids
        )

    # -------------------------------------------------------------------------
    # Explicit pooled-memory access
    # -------------------------------------------------------------------------

    @property
    def generic_pooled_memory(
        self,
    ) -> torch.Tensor | None:
        if self.temporal_pooling is None:
            return None

        return (
            self
            .temporal_pooling
            .pooled_memory
        )

    @property
    def temporal_pooling_weights(
        self,
    ) -> torch.Tensor | None:
        if self.temporal_pooling is None:
            return None

        return (
            self
            .temporal_pooling
            .pooling_weights
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
            .sequence_encoding
            .alignment_fingerprint()
        )

    def temporal_alignment_fingerprint(
        self,
    ) -> str:
        return (
            self
            .sequence_encoding
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
            "assembly_policy": (
                self.assembly_policy.value
            ),
            "encoded_shape": list(
                self.encoded_shape
            ),
            "has_temporal_pooling": (
                self.has_temporal_pooling
            ),
            "pooled_dim": (
                self.pooled_dim
            ),
            "sequence_preservation_policy": (
                URBAN_MEMORY_SEQUENCE_PRESERVATION_POLICY
            ),
            "pooling_scope": (
                URBAN_MEMORY_POOLING_SCOPE
            ),
            "sequence_encoding_lineage_fingerprint": (
                self
                .sequence_encoding
                .lineage_fingerprint()
            ),
            "temporal_pooling_lineage_fingerprint": (
                self
                .temporal_pooling
                .lineage_fingerprint()
                if (
                    self.temporal_pooling
                    is not None
                )
                else None
            ),
            "source_alignment_fingerprint": (
                self
                .sequence_encoding
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
                URBAN_MEMORY_SCIENTIFIC_INTERPRETATION
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
        return _fingerprint(
            {
                "sequence_encoding_value_fingerprint": (
                    self
                    .sequence_encoding
                    .value_fingerprint()
                ),
                "temporal_pooling_value_fingerprint": (
                    self
                    .temporal_pooling
                    .value_fingerprint()
                    if (
                        self.temporal_pooling
                        is not None
                    )
                    else None
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
            "sequence_encoding_lineage_fingerprint": (
                self
                .sequence_encoding
                .lineage_fingerprint()
            ),
            "temporal_pooling_lineage_fingerprint": (
                self
                .temporal_pooling
                .lineage_fingerprint()
                if (
                    self.temporal_pooling
                    is not None
                )
                else None
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
        Move the complete urban-memory contract to one device.

        The sequence is moved exactly once. When pooling exists, the pooling
        object is reconstructed around that same moved sequence so the runtime
        identity invariant remains true.
        """

        moved_sequence = (
            self
            .sequence_encoding
            .to(
                device,
                dtype=self.dtype,
                non_blocking=non_blocking,
            )
        )

        moved_pooling: (
            TemporalPoolingOutput
            | None
        ) = None

        if self.temporal_pooling is not None:
            source_pooling = (
                self.temporal_pooling
            )

            moved_pooling = type(
                source_pooling
            )(
                pooled_memory=(
                    source_pooling
                    .pooled_memory
                    .to(
                        device=device,
                        non_blocking=(
                            non_blocking
                        ),
                    )
                ),
                pooling_weights=(
                    source_pooling
                    .pooling_weights
                    .to(
                        device=device,
                        non_blocking=(
                            non_blocking
                        ),
                    )
                ),
                source_encoding=(
                    moved_sequence
                ),
                pooling_kind=(
                    source_pooling
                    .pooling_kind
                ),
                head_reduction=(
                    source_pooling
                    .head_reduction
                ),
                zero_history_policy=(
                    source_pooling
                    .zero_history_policy
                ),
                computation_provenance=(
                    source_pooling
                    .computation_provenance
                ),
                head_reduction_weights=(
                    source_pooling
                    .head_reduction_weights
                    .to(
                        device=device,
                        non_blocking=(
                            non_blocking
                        ),
                    )
                    if (
                        source_pooling
                        .head_reduction_weights
                        is not None
                    )
                    else None
                ),
                absolute_tolerance=(
                    source_pooling
                    .absolute_tolerance
                ),
                relative_tolerance=(
                    source_pooling
                    .relative_tolerance
                ),
                pooling_name=(
                    source_pooling
                    .pooling_name
                ),
                schema_version=(
                    source_pooling
                    .schema_version
                ),
            )

        return type(self)(
            sequence_encoding=(
                moved_sequence
            ),
            assembly_policy=(
                self.assembly_policy
            ),
            computation_provenance=(
                self.computation_provenance
            ),
            temporal_pooling=(
                moved_pooling
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
        """
        Reconstruct this schema with validated field changes.

        Replacing ``sequence_encoding`` without replacing a present pooling
        object will fail unless the pooling object points to that exact new
        sequence.
        """

        return dataclass_replace(
            self,
            **changes,
        )


# =============================================================================
# Compact aliases
# =============================================================================


UrbanTemporalMemory = UrbanMemory
SharedUrbanMemory = UrbanMemory


# =============================================================================
# Public API
# =============================================================================


__all__ = (
    # Schema identity and interpretation.
    "URBAN_MEMORY_SCHEMA_VERSION",
    "URBAN_MEMORY_SEQUENCE_PRESERVATION_POLICY",
    "URBAN_MEMORY_POOLING_SCOPE",
    "URBAN_MEMORY_SCIENTIFIC_INTERPRETATION",

    # Assembly vocabulary.
    "UrbanMemoryAssemblyPolicy",
    "CANONICAL_URBAN_MEMORY_ASSEMBLY_POLICIES",

    # Main contract and aliases.
    "UrbanMemory",
    "UrbanTemporalMemory",
    "SharedUrbanMemory",
)
