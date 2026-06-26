"""
Canonicalization, packing, unpacking, and restoration for Phase 6 recurrence.

Target path:
    urban_resilience_models/
        v2_hazard_conditioned_functional_ugnn/
            memory/
                recurrent_memory_encoder/
                    sequence_packing.py

This module owns the execution plumbing shared by GRU and LSTM encoders:

    source history
        -> gather valid timesteps in chronological order
        -> canonical right-padded nonempty-node batch
        -> reorder into actual execution order
        -> create ``PackedSequence`` for packed execution
        -> unpack recurrent outputs with ``total_length=T``
        -> restore ascending nonempty-node order
        -> restore the source temporal layout
        -> scatter sequence outputs and recurrent states to the full node axis

It does not derive history lengths, choose the sorting plan, adapt feature
channels, create recurrent initial states, execute a GRU/LSTM kernel, or build
provenance.

Frozen Phase 6 semantics
------------------------
Canonical temporal layout
    Every nonempty source row is represented as a contiguous valid prefix in a
    right-padded ``_CanonicalRecurrentBatch``. Source left/right/no-padding
    layout is retained only as metadata.

Packed execution order
    ``RecurrentExecutionMetadata.sorted_to_nonempty`` maps sorted positions to
    ascending nonempty-node positions. The inverse
    ``nonempty_to_sorted`` restores ascending nonempty-node order.

Packing boundary
    Phase 6 performs its own stable sort and always calls
    ``pack_padded_sequence(..., enforce_sorted=True)``. The temporary CPU copy
    of lengths exists only at this PyTorch API boundary.

Reference execution
    Reference metadata uses identity permutations. The same order-restoration
    functions therefore work without a second implementation.

Temporal restoration
    Valid recurrent outputs are placed at the exact source
    ``timestep_mask`` positions. All source padding remains exactly zero.

Zero-history nodes
    Zero-history rows are absent from recurrent execution and are introduced
    only by full-axis scattering into exact-zero placeholders.

State layout
    Recurrent kernels return flat states ``[L * directions, M, H]`` in actual
    execution order. This module restores nonempty-node order, converts to
    canonical ``[L, directions, M, H]``, and scatters to
    ``[L, directions, N, H]``.

Autograd
    Gathering, ordering, unpacking, restoration, and scattering preserve
    gradients. Source padded positions are never gathered and therefore receive
    exactly zero gradient.
"""

from __future__ import annotations

from typing import Final
from typing import NamedTuple

import torch
from torch.nn.utils.rnn import (
    PackedSequence,
    pack_padded_sequence,
    pad_packed_sequence,
)

from ..schemas.history_inputs import (
    HistoricalSequenceInputs,
)
from ..schemas.temporal_coordinates import (
    TemporalPaddingDirection,
)
from .schemas import (
    RecurrentExecutionMetadata,
    RecurrentExecutionPath,
    RecurrentStateLayout,
    _CanonicalRecurrentBatch,
)


# =============================================================================
# Component identity and frozen execution policies
# =============================================================================


RECURRENT_SEQUENCE_PACKING_IMPLEMENTATION_VERSION: Final[str] = "0.1"

RECURRENT_SEQUENCE_PACKING_COMPONENT_NAME: Final[str] = (
    "recurrent_sequence_packing"
)

RECURRENT_SEQUENCE_PACKING_COMPONENT_KIND: Final[str] = (
    "canonicalize_pack_restore_recurrent_sequences"
)

RECURRENT_SEQUENCE_PACKING_CANONICALIZATION_VERSION: Final[str] = (
    "valid_timesteps_chronological_to_right_padded_v1"
)

RECURRENT_SEQUENCE_PACKING_SORT_POLICY: Final[str] = (
    "metadata_defined_stable_descending_length_order_v1"
)

RECURRENT_SEQUENCE_PACKING_PACK_POLICY: Final[str] = (
    "explicit_sort_then_pack_enforce_sorted_true_v1"
)

RECURRENT_SEQUENCE_PACKING_RESTORE_POLICY: Final[str] = (
    "inverse_order_then_source_mask_positions_then_full_node_scatter_v1"
)

RECURRENT_SEQUENCE_PACKING_STATE_POLICY: Final[str] = (
    "inverse_batch_order_then_canonical_layout_then_full_node_scatter_v1"
)

RECURRENT_SEQUENCE_PACKING_PADDING_VALUE: Final[float] = 0.0

RECURRENT_SEQUENCE_PACKING_SCIENTIFIC_INTERPRETATION: Final[str] = (
    "execution_layout_transformation_not_temporal_importance_or_causality"
)


# =============================================================================
# Lightweight package-private execution-order view
# =============================================================================


class _ExecutionOrderedRecurrentBatch(NamedTuple):
    """
    Canonical values reordered into actual recurrent execution order.

    Fields
    ------
    values:
        ``[M, T, F]`` canonical right-padded values in execution order.

    timestep_mask:
        ``[M, T]`` canonical valid-prefix mask in execution order.

    lengths:
        ``[M]`` positive lengths in execution order.

    source_node_indices:
        ``[M]`` original source-node indices in execution order.

    value_stage:
        Human-readable stage copied from the canonical input batch.
    """

    values: torch.Tensor
    timestep_mask: torch.Tensor
    lengths: torch.Tensor
    source_node_indices: torch.Tensor
    value_stage: str


# =============================================================================
# Generic validation
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


def _require_nonnegative_int(
    name: str,
    value: int,
) -> None:
    if (
        isinstance(
            value,
            bool,
        )
        or not isinstance(
            value,
            int,
        )
    ):
        raise TypeError(
            f"{name} must be an integer."
        )

    if value < 0:
        raise ValueError(
            f"{name} must be nonnegative."
        )


def _require_positive_int(
    name: str,
    value: int,
) -> None:
    _require_nonnegative_int(
        name,
        value,
    )

    if value == 0:
        raise ValueError(
            f"{name} must be strictly positive."
        )


def _validate_source_and_metadata(
    source_history: HistoricalSequenceInputs,
    metadata: RecurrentExecutionMetadata,
) -> None:
    if not isinstance(
        source_history,
        HistoricalSequenceInputs,
    ):
        raise TypeError(
            "source_history must be a HistoricalSequenceInputs."
        )

    if not isinstance(
        metadata,
        RecurrentExecutionMetadata,
    ):
        raise TypeError(
            "metadata must be a RecurrentExecutionMetadata."
        )

    if metadata.source_node_count != source_history.node_count:
        raise ValueError(
            "metadata source-node count must match source_history."
        )

    if metadata.device != source_history.device:
        raise ValueError(
            "metadata and source_history must share one device."
        )

    if not torch.equal(
        metadata.history_lengths,
        source_history.valid_lengths,
    ):
        raise ValueError(
            "metadata history_lengths must exactly equal "
            "source_history.timestep_mask.sum(dim=1)."
        )

    if (
        metadata.original_padding_direction
        != source_history.padding_direction
    ):
        raise ValueError(
            "metadata original padding direction must match source_history."
        )


def _validate_batch_and_metadata(
    batch: _CanonicalRecurrentBatch,
    metadata: RecurrentExecutionMetadata,
) -> None:
    if not isinstance(
        batch,
        _CanonicalRecurrentBatch,
    ):
        raise TypeError(
            "batch must be a _CanonicalRecurrentBatch."
        )

    if not isinstance(
        metadata,
        RecurrentExecutionMetadata,
    ):
        raise TypeError(
            "metadata must be a RecurrentExecutionMetadata."
        )

    if batch.source_node_count != metadata.source_node_count:
        raise ValueError(
            "Canonical batch source-node count must match metadata."
        )

    if batch.device != metadata.device:
        raise ValueError(
            "Canonical batch and metadata must share one device."
        )

    if batch.nonempty_node_count != metadata.nonempty_node_count:
        raise ValueError(
            "Canonical batch nonempty-node count must match metadata."
        )

    if not torch.equal(
        batch.nonempty_node_indices,
        metadata.nonempty_node_indices,
    ):
        raise ValueError(
            "Canonical batch nonempty_node_indices must exactly match "
            "metadata nonempty-node order."
        )

    if not torch.equal(
        batch.lengths,
        metadata.nonempty_history_lengths,
    ):
        raise ValueError(
            "Canonical batch lengths must equal metadata nonempty lengths."
        )

    if (
        batch.original_padding_direction
        != metadata.original_padding_direction
    ):
        raise ValueError(
            "Canonical batch original padding direction must match metadata."
        )


def _validate_sequence_tensor(
    sequence: torch.Tensor,
    *,
    name: str,
    expected_node_count: int,
    expected_sequence_length: int,
    expected_device: torch.device,
) -> None:
    if not isinstance(
        sequence,
        torch.Tensor,
    ):
        raise TypeError(
            f"{name} must be a tensor."
        )

    if sequence.ndim != 3:
        raise ValueError(
            f"{name} must have shape [M, T, H]."
        )

    if not sequence.dtype.is_floating_point:
        raise ValueError(
            f"{name} must use a floating-point dtype."
        )

    if sequence.layout != torch.strided:
        raise ValueError(
            f"{name} must use strided tensor layout."
        )

    if sequence.is_meta:
        raise ValueError(
            f"{name} cannot reside on the meta device."
        )

    if int(
        sequence.shape[0]
    ) != expected_node_count:
        raise ValueError(
            f"{name} node dimension must equal {expected_node_count}."
        )

    if int(
        sequence.shape[1]
    ) != expected_sequence_length:
        raise ValueError(
            f"{name} temporal dimension must equal "
            f"{expected_sequence_length}."
        )

    if int(
        sequence.shape[2]
    ) <= 0:
        raise ValueError(
            f"{name} feature dimension must be strictly positive."
        )

    if sequence.device != expected_device:
        raise ValueError(
            f"{name} device must match execution metadata."
        )

    if not bool(
        torch.isfinite(
            sequence
        ).all().item()
    ):
        raise ValueError(
            f"{name} must contain only finite values."
        )


def _validate_exact_zero_padding(
    values: torch.Tensor,
    timestep_mask: torch.Tensor,
    *,
    name: str,
) -> None:
    padded = values.masked_select(
        (
            ~timestep_mask
        ).unsqueeze(
            -1
        ).expand_as(
            values
        )
    )

    if padded.numel() == 0:
        return

    if bool(
        torch.any(
            padded != 0
        ).item()
    ):
        raise ValueError(
            f"{name} must be exactly zero at padded timesteps."
        )


def _validate_execution_ordered_batch(
    ordered: _ExecutionOrderedRecurrentBatch,
    metadata: RecurrentExecutionMetadata,
    *,
    sequence_length: int,
) -> None:
    if not isinstance(
        ordered,
        _ExecutionOrderedRecurrentBatch,
    ):
        raise TypeError(
            "ordered must be an _ExecutionOrderedRecurrentBatch."
        )

    _validate_sequence_tensor(
        ordered.values,
        name="ordered.values",
        expected_node_count=(
            metadata.nonempty_node_count
        ),
        expected_sequence_length=(
            sequence_length
        ),
        expected_device=metadata.device,
    )

    if not isinstance(
        ordered.timestep_mask,
        torch.Tensor,
    ):
        raise TypeError(
            "ordered.timestep_mask must be a tensor."
        )

    if ordered.timestep_mask.dtype != torch.bool:
        raise ValueError(
            "ordered.timestep_mask must use torch.bool."
        )

    if tuple(
        ordered.timestep_mask.shape
    ) != (
        metadata.nonempty_node_count,
        sequence_length,
    ):
        raise ValueError(
            "ordered.timestep_mask must have shape [M, T]."
        )

    if ordered.timestep_mask.device != metadata.device:
        raise ValueError(
            "ordered.timestep_mask device must match metadata."
        )

    for name, tensor in (
        (
            "ordered.lengths",
            ordered.lengths,
        ),
        (
            "ordered.source_node_indices",
            ordered.source_node_indices,
        ),
    ):
        if not isinstance(
            tensor,
            torch.Tensor,
        ):
            raise TypeError(
                f"{name} must be a tensor."
            )

        if tensor.ndim != 1:
            raise ValueError(
                f"{name} must be one-dimensional."
            )

        if tensor.dtype != torch.long:
            raise ValueError(
                f"{name} must use torch.int64."
            )

        if tensor.device != metadata.device:
            raise ValueError(
                f"{name} device must match metadata."
            )

        if tensor.numel() != metadata.nonempty_node_count:
            raise ValueError(
                f"{name} must contain M entries."
            )

    if not torch.equal(
        ordered.lengths,
        metadata.sorted_history_lengths,
    ):
        raise ValueError(
            "ordered.lengths must equal metadata sorted lengths."
        )

    if not torch.equal(
        ordered.source_node_indices,
        metadata.sorted_node_indices,
    ):
        raise ValueError(
            "ordered.source_node_indices must equal metadata sorted nodes."
        )

    expected_mask = (
        torch.arange(
            sequence_length,
            dtype=torch.long,
            device=metadata.device,
        ).unsqueeze(
            0
        )
        < ordered.lengths.unsqueeze(
            1
        )
    )

    if not torch.equal(
        ordered.timestep_mask,
        expected_mask,
    ):
        raise ValueError(
            "ordered.timestep_mask must be a right-padded valid-prefix mask."
        )

    _validate_exact_zero_padding(
        ordered.values,
        ordered.timestep_mask,
        name="ordered.values",
    )
    _require_nonempty_string(
        "ordered.value_stage",
        ordered.value_stage,
    )


# =============================================================================
# Source-to-canonical transformation
# =============================================================================


def canonicalize_recurrent_history(
    source_history: HistoricalSequenceInputs,
    metadata: RecurrentExecutionMetadata,
    *,
    value_stage: str = "canonical_raw_history",
) -> _CanonicalRecurrentBatch:
    """
    Gather valid source timesteps into a canonical right-padded batch.

    Valid values are gathered in source temporal order. Because source masks are
    required to represent left, right, or no padding, positional order is also
    chronological order.
    """

    _validate_source_and_metadata(
        source_history,
        metadata,
    )
    _require_nonempty_string(
        "value_stage",
        value_stage,
    )

    nonempty_indices = (
        metadata.nonempty_node_indices
    )
    nonempty_count = (
        metadata.nonempty_node_count
    )
    sequence_length = (
        source_history.sequence_length
    )
    feature_dim = (
        source_history.feature_dim
    )

    source_values = source_history.history.index_select(
        0,
        nonempty_indices,
    )
    source_mask = (
        source_history
        .timestep_mask
        .index_select(
            0,
            nonempty_indices,
        )
    )
    lengths = metadata.nonempty_history_lengths

    canonical_mask = (
        torch.arange(
            sequence_length,
            dtype=torch.long,
            device=source_history.device,
        ).unsqueeze(
            0
        )
        < lengths.unsqueeze(
            1
        )
    )

    canonical_values = source_values.new_zeros(
        (
            nonempty_count,
            sequence_length,
            feature_dim,
        )
    )

    if nonempty_count > 0:
        valid_source_values = source_values[
            source_mask
        ]

        if int(
            valid_source_values.shape[0]
        ) != int(
            lengths.sum().item()
        ):
            raise RuntimeError(
                "Source valid-value count does not match metadata lengths."
            )

        canonical_values[
            canonical_mask
        ] = valid_source_values

    return _CanonicalRecurrentBatch(
        values=canonical_values,
        timestep_mask=canonical_mask,
        lengths=lengths,
        nonempty_node_indices=(
            nonempty_indices
        ),
        source_node_count=(
            source_history.node_count
        ),
        original_padding_direction=(
            source_history.padding_direction
        ),
        value_stage=value_stage,
    )


def gather_single_node_valid_sequence(
    source_history: HistoricalSequenceInputs,
    node_index: int,
) -> torch.Tensor:
    """
    Gather one source node as ``[1, L, D]`` in chronological order.

    A zero-history node returns ``[1, 0, D]`` and must not be passed to a
    recurrent kernel.
    """

    if not isinstance(
        source_history,
        HistoricalSequenceInputs,
    ):
        raise TypeError(
            "source_history must be a HistoricalSequenceInputs."
        )

    _require_nonnegative_int(
        "node_index",
        node_index,
    )

    if node_index >= source_history.node_count:
        raise IndexError(
            "node_index is outside the source node axis."
        )

    valid = source_history.timestep_mask[
        node_index
    ]
    values = source_history.history[
        node_index
    ][
        valid
    ]

    return values.unsqueeze(
        0
    )


def gather_canonical_node_sequence(
    batch: _CanonicalRecurrentBatch,
    nonempty_position: int,
) -> torch.Tensor:
    """
    Gather one canonical nonempty row as ``[1, L, F]``.

    This is the intended reference-execution boundary after shared input
    adaptation.
    """

    if not isinstance(
        batch,
        _CanonicalRecurrentBatch,
    ):
        raise TypeError(
            "batch must be a _CanonicalRecurrentBatch."
        )

    _require_nonnegative_int(
        "nonempty_position",
        nonempty_position,
    )

    if nonempty_position >= batch.nonempty_node_count:
        raise IndexError(
            "nonempty_position is outside the canonical nonempty-node axis."
        )

    length = int(
        batch.lengths[
            nonempty_position
        ].item()
    )

    return batch.values[
        nonempty_position : nonempty_position + 1,
        :length,
        :,
    ]


# =============================================================================
# Execution-order transformation and packing
# =============================================================================


def order_canonical_recurrent_batch_for_execution(
    batch: _CanonicalRecurrentBatch,
    metadata: RecurrentExecutionMetadata,
) -> _ExecutionOrderedRecurrentBatch:
    """
    Reorder a canonical batch by metadata-defined actual execution order.

    For reference execution the permutation is identity.
    """

    _validate_batch_and_metadata(
        batch,
        metadata,
    )

    permutation = (
        metadata.sorted_to_nonempty
    )

    ordered = _ExecutionOrderedRecurrentBatch(
        values=batch.values.index_select(
            0,
            permutation,
        ),
        timestep_mask=(
            batch.timestep_mask.index_select(
                0,
                permutation,
            )
        ),
        lengths=batch.lengths.index_select(
            0,
            permutation,
        ),
        source_node_indices=(
            batch
            .nonempty_node_indices
            .index_select(
                0,
                permutation,
            )
        ),
        value_stage=batch.value_stage,
    )

    _validate_execution_ordered_batch(
        ordered,
        metadata,
        sequence_length=(
            batch.sequence_length
        ),
    )

    return ordered


def sort_canonical_recurrent_batch(
    batch: _CanonicalRecurrentBatch,
    metadata: RecurrentExecutionMetadata,
) -> _ExecutionOrderedRecurrentBatch:
    """
    Packed-path wrapper requiring metadata.execution_path == ``packed``.
    """

    if not isinstance(
        metadata,
        RecurrentExecutionMetadata,
    ):
        raise TypeError(
            "metadata must be a RecurrentExecutionMetadata."
        )

    if (
        metadata.execution_path
        != RecurrentExecutionPath.PACKED
    ):
        raise ValueError(
            "sort_canonical_recurrent_batch requires packed metadata."
        )

    return order_canonical_recurrent_batch_for_execution(
        batch,
        metadata,
    )


def pack_canonical_recurrent_batch(
    batch: _CanonicalRecurrentBatch,
    metadata: RecurrentExecutionMetadata,
) -> PackedSequence:
    """
    Create a PyTorch ``PackedSequence`` after explicit metadata-defined sorting.

    ``lengths.cpu()`` is intentionally the only CPU metadata copy.
    """

    ordered = sort_canonical_recurrent_batch(
        batch,
        metadata,
    )

    if metadata.nonempty_node_count == 0:
        raise ValueError(
            "Cannot create a PackedSequence for an all-zero-history batch."
        )

    if not bool(
        torch.all(
            ordered.lengths[:-1]
            >= ordered.lengths[1:]
        ).item()
    ):
        raise RuntimeError(
            "Packed execution lengths must be nonincreasing."
        )

    packed = pack_padded_sequence(
        ordered.values,
        ordered.lengths.detach().cpu(),
        batch_first=True,
        enforce_sorted=True,
    )

    if not bool(
        torch.isfinite(
            packed.data
        ).all().item()
    ):
        raise RuntimeError(
            "Packed recurrent input contains nonfinite values."
        )

    return packed


def unpack_recurrent_sequence(
    packed_output: PackedSequence,
    metadata: RecurrentExecutionMetadata,
    *,
    total_length: int,
) -> torch.Tensor:
    """
    Unpack recurrent output to ``[M, T, Hout]`` in packed execution order.
    """

    if not isinstance(
        packed_output,
        PackedSequence,
    ):
        raise TypeError(
            "packed_output must be a PackedSequence."
        )

    if not isinstance(
        metadata,
        RecurrentExecutionMetadata,
    ):
        raise TypeError(
            "metadata must be a RecurrentExecutionMetadata."
        )

    if (
        metadata.execution_path
        != RecurrentExecutionPath.PACKED
    ):
        raise ValueError(
            "unpack_recurrent_sequence requires packed metadata."
        )

    _require_positive_int(
        "total_length",
        total_length,
    )

    if metadata.nonempty_node_count == 0:
        raise ValueError(
            "Cannot unpack a PackedSequence for an all-zero-history batch."
        )

    if not bool(
        torch.isfinite(
            packed_output.data
        ).all().item()
    ):
        raise ValueError(
            "Raw packed recurrent output must contain only finite values."
        )

    unpacked, returned_lengths = pad_packed_sequence(
        packed_output,
        batch_first=True,
        padding_value=(
            RECURRENT_SEQUENCE_PACKING_PADDING_VALUE
        ),
        total_length=total_length,
    )

    if tuple(
        unpacked.shape[:2]
    ) != (
        metadata.nonempty_node_count,
        total_length,
    ):
        raise RuntimeError(
            "Unpacked recurrent output has an unexpected node/temporal shape."
        )

    if int(
        unpacked.shape[2]
    ) <= 0:
        raise RuntimeError(
            "Unpacked recurrent output width must be strictly positive."
        )

    if not bool(
        torch.isfinite(
            unpacked
        ).all().item()
    ):
        raise ValueError(
            "Raw unpacked recurrent output must contain only finite values."
        )

    if not torch.equal(
        returned_lengths.to(
            device=metadata.device,
            dtype=torch.long,
        ),
        metadata.sorted_history_lengths,
    ):
        raise RuntimeError(
            "pad_packed_sequence returned lengths inconsistent with metadata."
        )

    ordered_mask = (
        torch.arange(
            total_length,
            dtype=torch.long,
            device=unpacked.device,
        ).unsqueeze(
            0
        )
        < metadata.sorted_history_lengths.unsqueeze(
            1
        )
    )

    _validate_exact_zero_padding(
        unpacked,
        ordered_mask,
        name="unpacked recurrent output",
    )

    return unpacked


# =============================================================================
# Sequence-order and temporal-layout restoration
# =============================================================================


def restore_nonempty_sequence_order(
    execution_sequence: torch.Tensor,
    metadata: RecurrentExecutionMetadata,
) -> torch.Tensor:
    """
    Restore ascending original nonempty-node order from actual execution order.
    """

    if not isinstance(
        metadata,
        RecurrentExecutionMetadata,
    ):
        raise TypeError(
            "metadata must be a RecurrentExecutionMetadata."
        )

    _validate_sequence_tensor(
        execution_sequence,
        name="execution_sequence",
        expected_node_count=(
            metadata.nonempty_node_count
        ),
        expected_sequence_length=(
            int(
                execution_sequence.shape[1]
            )
            if (
                isinstance(
                    execution_sequence,
                    torch.Tensor,
                )
                and execution_sequence.ndim == 3
            )
            else 0
        ),
        expected_device=metadata.device,
    )

    return execution_sequence.index_select(
        0,
        metadata.nonempty_to_sorted,
    )


def restore_source_temporal_layout(
    canonical_nonempty_sequence: torch.Tensor,
    source_history: HistoricalSequenceInputs,
    metadata: RecurrentExecutionMetadata,
) -> torch.Tensor:
    """
    Place canonical valid prefixes at exact source temporal-mask positions.

    Returns ``[M, T, Hout]`` in ascending nonempty-node order.
    """

    _validate_source_and_metadata(
        source_history,
        metadata,
    )
    _validate_sequence_tensor(
        canonical_nonempty_sequence,
        name="canonical_nonempty_sequence",
        expected_node_count=(
            metadata.nonempty_node_count
        ),
        expected_sequence_length=(
            source_history.sequence_length
        ),
        expected_device=source_history.device,
    )

    canonical_mask = (
        torch.arange(
            source_history.sequence_length,
            dtype=torch.long,
            device=source_history.device,
        ).unsqueeze(
            0
        )
        < metadata.nonempty_history_lengths.unsqueeze(
            1
        )
    )

    _validate_exact_zero_padding(
        canonical_nonempty_sequence,
        canonical_mask,
        name="canonical_nonempty_sequence",
    )

    source_mask = (
        source_history
        .timestep_mask
        .index_select(
            0,
            metadata.nonempty_node_indices,
        )
    )

    restored = canonical_nonempty_sequence.new_zeros(
        canonical_nonempty_sequence.shape
    )

    if metadata.nonempty_node_count > 0:
        valid_values = canonical_nonempty_sequence[
            canonical_mask
        ]

        if int(
            valid_values.shape[0]
        ) != int(
            source_mask.sum().item()
        ):
            raise RuntimeError(
                "Canonical/source valid-position counts do not match."
            )

        restored[
            source_mask
        ] = valid_values

    _validate_exact_zero_padding(
        restored,
        source_mask,
        name="source-layout nonempty sequence",
    )

    return restored


def scatter_nonempty_sequence_to_source(
    nonempty_source_sequence: torch.Tensor,
    metadata: RecurrentExecutionMetadata,
) -> torch.Tensor:
    """
    Scatter ``[M, T, Hout]`` onto the complete source node axis ``[N,T,Hout]``.
    """

    if not isinstance(
        metadata,
        RecurrentExecutionMetadata,
    ):
        raise TypeError(
            "metadata must be a RecurrentExecutionMetadata."
        )

    _validate_sequence_tensor(
        nonempty_source_sequence,
        name="nonempty_source_sequence",
        expected_node_count=(
            metadata.nonempty_node_count
        ),
        expected_sequence_length=(
            int(
                nonempty_source_sequence.shape[1]
            )
            if (
                isinstance(
                    nonempty_source_sequence,
                    torch.Tensor,
                )
                and nonempty_source_sequence.ndim == 3
            )
            else 0
        ),
        expected_device=metadata.device,
    )

    full = nonempty_source_sequence.new_zeros(
        (
            metadata.source_node_count,
            int(
                nonempty_source_sequence.shape[1]
            ),
            int(
                nonempty_source_sequence.shape[2]
            ),
        )
    )

    return full.index_copy(
        0,
        metadata.nonempty_node_indices,
        nonempty_source_sequence,
    )


def restore_recurrent_sequence_to_source(
    execution_sequence: torch.Tensor,
    source_history: HistoricalSequenceInputs,
    metadata: RecurrentExecutionMetadata,
) -> torch.Tensor:
    """
    Restore one packed/reference execution sequence to full source layout.
    """

    _validate_source_and_metadata(
        source_history,
        metadata,
    )
    _validate_sequence_tensor(
        execution_sequence,
        name="execution_sequence",
        expected_node_count=(
            metadata.nonempty_node_count
        ),
        expected_sequence_length=(
            source_history.sequence_length
        ),
        expected_device=source_history.device,
    )

    nonempty_order = restore_nonempty_sequence_order(
        execution_sequence,
        metadata,
    )
    source_temporal = restore_source_temporal_layout(
        nonempty_order,
        source_history,
        metadata,
    )
    full = scatter_nonempty_sequence_to_source(
        source_temporal,
        metadata,
    )

    _validate_exact_zero_padding(
        full,
        source_history.timestep_mask,
        name="full restored recurrent sequence",
    )

    if metadata.zero_history_count > 0:
        zero_rows = full.index_select(
            0,
            metadata.zero_history_node_indices,
        )

        if bool(
            torch.any(
                zero_rows != 0
            ).item()
        ):
            raise RuntimeError(
                "Zero-history sequence rows must remain exactly zero."
            )

    return full


def restore_single_node_sequence(
    valid_sequence: torch.Tensor,
    source_history: HistoricalSequenceInputs,
    node_index: int,
) -> torch.Tensor:
    """
    Restore one ``[1, L, Hout]`` reference result to ``[1, T, Hout]``.
    """

    if not isinstance(
        source_history,
        HistoricalSequenceInputs,
    ):
        raise TypeError(
            "source_history must be a HistoricalSequenceInputs."
        )

    _require_nonnegative_int(
        "node_index",
        node_index,
    )

    if node_index >= source_history.node_count:
        raise IndexError(
            "node_index is outside the source node axis."
        )

    if not isinstance(
        valid_sequence,
        torch.Tensor,
    ):
        raise TypeError(
            "valid_sequence must be a tensor."
        )

    if valid_sequence.ndim != 3:
        raise ValueError(
            "valid_sequence must have shape [1, L, Hout]."
        )

    if int(
        valid_sequence.shape[0]
    ) != 1:
        raise ValueError(
            "valid_sequence must contain exactly one node."
        )

    if int(
        valid_sequence.shape[2]
    ) <= 0:
        raise ValueError(
            "valid_sequence output width must be strictly positive."
        )

    if not valid_sequence.dtype.is_floating_point:
        raise ValueError(
            "valid_sequence must use a floating-point dtype."
        )

    if valid_sequence.device != source_history.device:
        raise ValueError(
            "valid_sequence device must match source_history."
        )

    if valid_sequence.dtype != source_history.dtype:
        raise ValueError(
            "valid_sequence dtype must match source_history."
        )

    if not bool(
        torch.isfinite(
            valid_sequence
        ).all().item()
    ):
        raise ValueError(
            "valid_sequence must contain only finite values."
        )

    expected_length = int(
        source_history.valid_lengths[
            node_index
        ].item()
    )

    if int(
        valid_sequence.shape[1]
    ) != expected_length:
        raise ValueError(
            "valid_sequence temporal length must equal the node's "
            "source history length."
        )

    restored = valid_sequence.new_zeros(
        (
            1,
            source_history.sequence_length,
            int(
                valid_sequence.shape[2]
            ),
        )
    )
    node_mask = source_history.timestep_mask[
        node_index : node_index + 1
    ]

    if expected_length > 0:
        restored[
            node_mask
        ] = valid_sequence.reshape(
            expected_length,
            int(
                valid_sequence.shape[2]
            ),
        )

    _validate_exact_zero_padding(
        restored,
        node_mask,
        name="restored single-node sequence",
    )

    return restored


# =============================================================================
# Recurrent-state order restoration and full-axis scattering
# =============================================================================


def restore_nonempty_flat_state_order(
    execution_flat_state: torch.Tensor,
    metadata: RecurrentExecutionMetadata,
    layout: RecurrentStateLayout,
    *,
    name: str = "execution_flat_state",
) -> torch.Tensor:
    """
    Restore flat state batch axis from execution order to nonempty-node order.
    """

    if not isinstance(
        metadata,
        RecurrentExecutionMetadata,
    ):
        raise TypeError(
            "metadata must be a RecurrentExecutionMetadata."
        )

    if not isinstance(
        layout,
        RecurrentStateLayout,
    ):
        raise TypeError(
            "layout must be a RecurrentStateLayout."
        )

    layout.validate_flat_state(
        execution_flat_state,
        name=name,
        node_count=(
            metadata.nonempty_node_count
        ),
    )

    if execution_flat_state.device != metadata.device:
        raise ValueError(
            f"{name} device must match execution metadata."
        )

    return execution_flat_state.index_select(
        1,
        metadata.nonempty_to_sorted,
    )


def scatter_nonempty_canonical_state_to_source(
    nonempty_canonical_state: torch.Tensor,
    metadata: RecurrentExecutionMetadata,
    layout: RecurrentStateLayout,
    *,
    name: str = "nonempty_canonical_state",
) -> torch.Tensor:
    """
    Scatter ``[L,D,M,H]`` onto complete ``[L,D,N,H]`` node layout.
    """

    if not isinstance(
        metadata,
        RecurrentExecutionMetadata,
    ):
        raise TypeError(
            "metadata must be a RecurrentExecutionMetadata."
        )

    if not isinstance(
        layout,
        RecurrentStateLayout,
    ):
        raise TypeError(
            "layout must be a RecurrentStateLayout."
        )

    layout.validate_canonical_state(
        nonempty_canonical_state,
        name=name,
        node_count=(
            metadata.nonempty_node_count
        ),
    )

    if nonempty_canonical_state.device != metadata.device:
        raise ValueError(
            f"{name} device must match execution metadata."
        )

    full = nonempty_canonical_state.new_zeros(
        (
            layout.num_layers,
            layout.num_directions,
            metadata.source_node_count,
            layout.hidden_dim,
        )
    )

    full = full.index_copy(
        2,
        metadata.nonempty_node_indices,
        nonempty_canonical_state,
    )

    if metadata.zero_history_count > 0:
        zero_state = full.index_select(
            2,
            metadata.zero_history_node_indices,
        )

        if bool(
            torch.any(
                zero_state != 0
            ).item()
        ):
            raise RuntimeError(
                "Zero-history recurrent states must remain exactly zero."
            )

    return full


def restore_and_scatter_recurrent_state(
    execution_flat_state: torch.Tensor,
    metadata: RecurrentExecutionMetadata,
    layout: RecurrentStateLayout,
    *,
    name: str = "execution_flat_state",
) -> torch.Tensor:
    """
    Restore one flat hidden/cell state to canonical full-node layout.
    """

    nonempty_flat = restore_nonempty_flat_state_order(
        execution_flat_state,
        metadata,
        layout,
        name=name,
    )
    nonempty_canonical = layout.unflatten_state(
        nonempty_flat,
        name=f"{name}_nonempty_order",
    )

    return scatter_nonempty_canonical_state_to_source(
        nonempty_canonical,
        metadata,
        layout,
        name=f"{name}_canonical",
    )


def restore_and_scatter_recurrent_states(
    execution_hidden_state: torch.Tensor,
    execution_cell_state: torch.Tensor | None,
    metadata: RecurrentExecutionMetadata,
    layout: RecurrentStateLayout,
) -> tuple[
    torch.Tensor,
    torch.Tensor | None,
]:
    """
    Restore hidden and optional LSTM cell states independently.
    """

    hidden = restore_and_scatter_recurrent_state(
        execution_hidden_state,
        metadata,
        layout,
        name="execution_hidden_state",
    )

    if execution_cell_state is None:
        return (
            hidden,
            None,
        )

    cell = restore_and_scatter_recurrent_state(
        execution_cell_state,
        metadata,
        layout,
        name="execution_cell_state",
    )

    return (
        hidden,
        cell,
    )


def restore_and_scatter_lstm_states(
    execution_hidden_state: torch.Tensor,
    execution_cell_state: torch.Tensor,
    metadata: RecurrentExecutionMetadata,
    layout: RecurrentStateLayout,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
]:
    """LSTM-specific wrapper requiring both hidden and cell state tensors."""

    if execution_cell_state is None:
        raise TypeError(
            "execution_cell_state must be a tensor for LSTM restoration."
        )

    hidden, cell = restore_and_scatter_recurrent_states(
        execution_hidden_state,
        execution_cell_state,
        metadata,
        layout,
    )

    assert cell is not None

    return (
        hidden,
        cell,
    )


# =============================================================================
# Compact aliases
# =============================================================================


canonicalize_history = canonicalize_recurrent_history
order_batch_for_execution = (
    order_canonical_recurrent_batch_for_execution
)
pack_recurrent_batch = pack_canonical_recurrent_batch
unpack_recurrent_output = unpack_recurrent_sequence
restore_sequence_to_source = restore_recurrent_sequence_to_source
restore_state_to_source = restore_and_scatter_recurrent_state


# =============================================================================
# Module API
# =============================================================================


__all__ = (
    # Component identity and frozen policies.
    "RECURRENT_SEQUENCE_PACKING_IMPLEMENTATION_VERSION",
    "RECURRENT_SEQUENCE_PACKING_COMPONENT_NAME",
    "RECURRENT_SEQUENCE_PACKING_COMPONENT_KIND",
    "RECURRENT_SEQUENCE_PACKING_CANONICALIZATION_VERSION",
    "RECURRENT_SEQUENCE_PACKING_SORT_POLICY",
    "RECURRENT_SEQUENCE_PACKING_PACK_POLICY",
    "RECURRENT_SEQUENCE_PACKING_RESTORE_POLICY",
    "RECURRENT_SEQUENCE_PACKING_STATE_POLICY",
    "RECURRENT_SEQUENCE_PACKING_PADDING_VALUE",
    "RECURRENT_SEQUENCE_PACKING_SCIENTIFIC_INTERPRETATION",

    # Source-to-canonical transformation.
    "canonicalize_recurrent_history",
    "gather_single_node_valid_sequence",
    "gather_canonical_node_sequence",

    # Execution ordering and packing.
    "order_canonical_recurrent_batch_for_execution",
    "sort_canonical_recurrent_batch",
    "pack_canonical_recurrent_batch",
    "unpack_recurrent_sequence",

    # Sequence restoration.
    "restore_nonempty_sequence_order",
    "restore_source_temporal_layout",
    "scatter_nonempty_sequence_to_source",
    "restore_recurrent_sequence_to_source",
    "restore_single_node_sequence",

    # State restoration.
    "restore_nonempty_flat_state_order",
    "scatter_nonempty_canonical_state_to_source",
    "restore_and_scatter_recurrent_state",
    "restore_and_scatter_recurrent_states",
    "restore_and_scatter_lstm_states",

    # Compact aliases.
    "canonicalize_history",
    "order_batch_for_execution",
    "pack_recurrent_batch",
    "unpack_recurrent_output",
    "restore_sequence_to_source",
    "restore_state_to_source",
)
