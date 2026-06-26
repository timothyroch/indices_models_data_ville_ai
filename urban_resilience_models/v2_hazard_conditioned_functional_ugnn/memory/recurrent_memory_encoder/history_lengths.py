"""
History-length analysis and node partitioning for recurrent memory encoders.

Target path:
    urban_resilience_models/
        v2_hazard_conditioned_functional_ugnn/
            memory/
                recurrent_memory_encoder/
                    history_lengths.py

This module owns the non-neural preprocessing needed to describe which nodes
can enter recurrent execution:

- derive exact valid history lengths from ``timestep_mask``;
- partition the original node axis into nonempty and zero-history nodes;
- enforce the source-level zero-history policy;
- construct deterministic packed-path sorting permutations;
- construct identity reference-path permutations;
- assemble validated ``RecurrentExecutionMetadata``.

It does not gather temporal values, canonicalize left padding, create
``PackedSequence`` objects, initialize recurrent states, or execute GRU/LSTM
modules.

Frozen sorting semantics
------------------------
``enforce_sorted_lengths=False``
    Packed execution accepts arbitrary nonempty-node order and applies a stable
    descending-length sort internally. Equal-length ties retain ascending
    original node order.

``enforce_sorted_lengths=True``
    Packed execution requires nonempty-node lengths to already be
    nonincreasing. No sort is performed and both permutations are identity.

Reference execution never sorts. Its permutations are always identity, and no
descending-length requirement applies.

Zero-history semantics
----------------------
The source ``HistoricalSequenceInputs.zero_length_policy`` is the only policy
authority.

``ERROR``
    Any zero-history node is rejected.

``ALLOW_ZERO_HISTORY``
    Zero-history nodes are preserved in metadata and excluded from recurrent
    execution. Later execution stages must leave their sequence outputs and
    recurrent states exactly zero.

No fake length-one observation is introduced.
"""

from __future__ import annotations

from typing import Final

import torch

from ..config import (
    RecurrentSequenceEncoderConfig,
)
from ..schemas.history_inputs import (
    HistoricalSequenceInputs,
    HistoryZeroLengthPolicy,
)
from ..schemas.temporal_coordinates import (
    TemporalPaddingDirection,
)
from .schemas import (
    RecurrentExecutionMetadata,
    RecurrentExecutionPath,
)


# =============================================================================
# Module identity and frozen policies
# =============================================================================


RECURRENT_HISTORY_LENGTHS_IMPLEMENTATION_VERSION: Final[str] = "0.1"

RECURRENT_HISTORY_LENGTHS_COMPONENT_NAME: Final[str] = (
    "recurrent_history_lengths"
)

RECURRENT_HISTORY_LENGTHS_OPERATION_NAME: Final[str] = (
    "derive_partition_and_execution_order"
)

RECURRENT_HISTORY_LENGTHS_SORT_POLICY: Final[str] = (
    "stable_descending_length_original_nonempty_order_ties"
)

RECURRENT_HISTORY_LENGTHS_ZERO_POLICY_SOURCE: Final[str] = (
    "historical_sequence_inputs_zero_length_policy"
)

RECURRENT_HISTORY_LENGTHS_SCIENTIFIC_INTERPRETATION: Final[str] = (
    "sequence_existence_and_execution_order_metadata_not_temporal_importance"
)


# =============================================================================
# Generic validation
# =============================================================================


def _require_boolean(
    name: str,
    value: bool,
) -> None:
    if not isinstance(
        value,
        bool,
    ):
        raise TypeError(
            f"{name} must be a Boolean."
        )


def _require_positive_int(
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

    if value <= 0:
        raise ValueError(
            f"{name} must be strictly positive."
        )


def _normalize_execution_path(
    value: RecurrentExecutionPath | str,
) -> RecurrentExecutionPath:
    if isinstance(
        value,
        RecurrentExecutionPath,
    ):
        return value

    try:
        return RecurrentExecutionPath(
            value
        )
    except (
        TypeError,
        ValueError,
    ) as error:
        raise ValueError(
            f"Unsupported recurrent execution path {value!r}."
        ) from error


def _normalize_zero_length_policy(
    value: HistoryZeroLengthPolicy | str,
) -> HistoryZeroLengthPolicy:
    if isinstance(
        value,
        HistoryZeroLengthPolicy,
    ):
        return value

    try:
        return HistoryZeroLengthPolicy(
            value
        )
    except (
        TypeError,
        ValueError,
    ) as error:
        raise ValueError(
            f"Unsupported history zero-length policy {value!r}."
        ) from error


def _normalize_padding_direction(
    value: TemporalPaddingDirection | str,
) -> TemporalPaddingDirection:
    if isinstance(
        value,
        TemporalPaddingDirection,
    ):
        return value

    try:
        return TemporalPaddingDirection(
            value
        )
    except (
        TypeError,
        ValueError,
    ) as error:
        raise ValueError(
            f"Unsupported temporal padding direction {value!r}."
        ) from error


def _validate_lengths_tensor(
    history_lengths: torch.Tensor,
    *,
    source_sequence_length: int | None = None,
    allow_empty_node_axis: bool = False,
) -> torch.Tensor:
    """
    Validate and detach-clone one source-device history-length vector.
    """

    if not isinstance(
        history_lengths,
        torch.Tensor,
    ):
        raise TypeError(
            "history_lengths must be a tensor."
        )

    if history_lengths.ndim != 1:
        raise ValueError(
            "history_lengths must be one-dimensional."
        )

    if history_lengths.dtype != torch.long:
        raise ValueError(
            "history_lengths must use torch.int64."
        )

    if history_lengths.layout != torch.strided:
        raise ValueError(
            "history_lengths must use strided tensor layout."
        )

    if history_lengths.is_meta:
        raise ValueError(
            "history_lengths cannot reside on the meta device."
        )

    if history_lengths.requires_grad:
        raise ValueError(
            "history_lengths must not require gradients."
        )

    if (
        not allow_empty_node_axis
        and history_lengths.numel() == 0
    ):
        raise ValueError(
            "history_lengths must describe at least one source node."
        )

    if bool(
        torch.any(
            history_lengths < 0
        ).item()
    ):
        raise ValueError(
            "history_lengths must be nonnegative."
        )

    if source_sequence_length is not None:
        _require_positive_int(
            "source_sequence_length",
            source_sequence_length,
        )

        if bool(
            torch.any(
                history_lengths
                > source_sequence_length
            ).item()
        ):
            raise ValueError(
                "history_lengths cannot exceed source_sequence_length."
            )

    return (
        history_lengths
        .detach()
        .clone()
    )


def _validate_permutation(
    permutation: torch.Tensor,
    *,
    size: int,
    name: str,
) -> None:
    if not isinstance(
        permutation,
        torch.Tensor,
    ):
        raise TypeError(
            f"{name} must be a tensor."
        )

    if permutation.ndim != 1:
        raise ValueError(
            f"{name} must be one-dimensional."
        )

    if permutation.dtype != torch.long:
        raise ValueError(
            f"{name} must use torch.int64."
        )

    if permutation.numel() != size:
        raise ValueError(
            f"{name} must contain exactly {size} positions."
        )

    expected = torch.arange(
        size,
        dtype=torch.long,
        device=permutation.device,
    )

    if not torch.equal(
        torch.sort(
            permutation
        ).values,
        expected,
    ):
        raise ValueError(
            f"{name} must be a permutation of range({size})."
        )


# =============================================================================
# Length derivation and source-policy validation
# =============================================================================


def derive_recurrent_history_lengths(
    source_history: HistoricalSequenceInputs,
) -> torch.Tensor:
    """
    Derive detached source-device lengths from ``timestep_mask``.

    The returned tensor has shape ``[N]`` and dtype ``torch.int64``.
    """

    if not isinstance(
        source_history,
        HistoricalSequenceInputs,
    ):
        raise TypeError(
            "source_history must be a HistoricalSequenceInputs."
        )

    lengths = source_history.timestep_mask.sum(
        dim=1,
        dtype=torch.long,
    )

    return _validate_lengths_tensor(
        lengths,
        source_sequence_length=(
            source_history.sequence_length
        ),
    )


def validate_recurrent_history_lengths(
    history_lengths: torch.Tensor,
    *,
    source_node_count: int | None = None,
    source_sequence_length: int | None = None,
) -> torch.Tensor:
    """
    Validate an externally supplied history-length vector.

    A detached clone is returned so callers cannot mutate validated metadata
    through an aliased tensor.
    """

    validated = _validate_lengths_tensor(
        history_lengths,
        source_sequence_length=(
            source_sequence_length
        ),
    )

    if source_node_count is not None:
        _require_positive_int(
            "source_node_count",
            source_node_count,
        )

        if validated.numel() != source_node_count:
            raise ValueError(
                "history_lengths size must equal source_node_count."
            )

    return validated


def validate_zero_history_policy(
    history_lengths: torch.Tensor,
    *,
    zero_length_policy: (
        HistoryZeroLengthPolicy
        | str
    ),
) -> None:
    """
    Enforce the source-level zero-history policy.

    This function deliberately accepts no recurrent-specific fallback policy.
    """

    lengths = _validate_lengths_tensor(
        history_lengths,
    )
    policy = _normalize_zero_length_policy(
        zero_length_policy
    )

    zero_history_count = int(
        torch.count_nonzero(
            lengths == 0
        ).item()
    )

    if (
        zero_history_count > 0
        and policy
        == HistoryZeroLengthPolicy.ERROR
    ):
        raise ValueError(
            "Zero-history nodes are not allowed because "
            "zero_length_policy='error'."
        )


def validate_source_zero_history_policy(
    source_history: HistoricalSequenceInputs,
    *,
    history_lengths: torch.Tensor | None = None,
) -> torch.Tensor:
    """
    Validate the source history's declared zero-history policy.

    Returns the validated length vector used for the check.
    """

    if not isinstance(
        source_history,
        HistoricalSequenceInputs,
    ):
        raise TypeError(
            "source_history must be a HistoricalSequenceInputs."
        )

    if history_lengths is None:
        lengths = derive_recurrent_history_lengths(
            source_history
        )
    else:
        lengths = validate_recurrent_history_lengths(
            history_lengths,
            source_node_count=(
                source_history.node_count
            ),
            source_sequence_length=(
                source_history.sequence_length
            ),
        )

        expected = source_history.valid_lengths

        if not torch.equal(
            lengths,
            expected,
        ):
            raise ValueError(
                "history_lengths must exactly equal "
                "source_history.timestep_mask.sum(dim=1)."
            )

    validate_zero_history_policy(
        lengths,
        zero_length_policy=(
            source_history.zero_length_policy
        ),
    )

    return lengths


# =============================================================================
# Node partitioning
# =============================================================================


def partition_recurrent_history_nodes(
    history_lengths: torch.Tensor,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
]:
    """
    Partition nodes into strictly increasing original-node index vectors.

    Returns
    -------
    nonempty_node_indices:
        Original node indices with positive history length.

    zero_history_node_indices:
        Original node indices with zero history length.
    """

    lengths = _validate_lengths_tensor(
        history_lengths,
    )

    nonempty = torch.nonzero(
        lengths > 0,
        as_tuple=False,
    ).flatten()

    zero_history = torch.nonzero(
        lengths == 0,
        as_tuple=False,
    ).flatten()

    return (
        nonempty.to(
            dtype=torch.long
        ),
        zero_history.to(
            dtype=torch.long
        ),
    )


def gather_nonempty_history_lengths(
    history_lengths: torch.Tensor,
    nonempty_node_indices: torch.Tensor,
) -> torch.Tensor:
    """Return lengths in ascending original nonempty-node order."""

    lengths = _validate_lengths_tensor(
        history_lengths,
    )

    if not isinstance(
        nonempty_node_indices,
        torch.Tensor,
    ):
        raise TypeError(
            "nonempty_node_indices must be a tensor."
        )

    if nonempty_node_indices.ndim != 1:
        raise ValueError(
            "nonempty_node_indices must be one-dimensional."
        )

    if nonempty_node_indices.dtype != torch.long:
        raise ValueError(
            "nonempty_node_indices must use torch.int64."
        )

    if (
        nonempty_node_indices.device
        != lengths.device
    ):
        raise ValueError(
            "nonempty_node_indices device must match history_lengths."
        )

    if nonempty_node_indices.numel() == 0:
        return lengths.new_empty(
            (
                0,
            )
        )

    if bool(
        torch.any(
            nonempty_node_indices < 0
        ).item()
    ) or bool(
        torch.any(
            nonempty_node_indices
            >= lengths.numel()
        ).item()
    ):
        raise ValueError(
            "nonempty_node_indices must lie in range(N)."
        )

    if nonempty_node_indices.numel() > 1:
        if not bool(
            torch.all(
                nonempty_node_indices[1:]
                > nonempty_node_indices[:-1]
            ).item()
        ):
            raise ValueError(
                "nonempty_node_indices must be strictly increasing."
            )

    selected = lengths.index_select(
        0,
        nonempty_node_indices,
    )

    if bool(
        torch.any(
            selected <= 0
        ).item()
    ):
        raise ValueError(
            "nonempty_node_indices may reference only positive lengths."
        )

    return selected


# =============================================================================
# Stable packed-order construction
# =============================================================================


def lengths_are_nonincreasing(
    lengths: torch.Tensor,
) -> bool:
    """Return whether a length vector is nonincreasing."""

    validated = _validate_lengths_tensor(
        lengths,
        allow_empty_node_axis=True,
    )

    if validated.numel() <= 1:
        return True

    return bool(
        torch.all(
            validated[:-1]
            >= validated[1:]
        ).item()
    )


def stable_descending_length_permutation(
    nonempty_history_lengths: torch.Tensor,
) -> torch.Tensor:
    """
    Return stable descending sort positions for positive nonempty lengths.

    Equal-length ties retain their incoming order, which is defined elsewhere
    as ascending original node order.
    """

    lengths = _validate_lengths_tensor(
        nonempty_history_lengths,
        allow_empty_node_axis=True,
    )

    if bool(
        torch.any(
            lengths <= 0
        ).item()
    ):
        raise ValueError(
            "nonempty_history_lengths must be strictly positive."
        )

    if lengths.numel() == 0:
        return lengths.new_empty(
            (
                0,
            )
        )

    return torch.argsort(
        lengths,
        dim=0,
        descending=True,
        stable=True,
    ).to(
        dtype=torch.long
    )


def inverse_permutation(
    permutation: torch.Tensor,
) -> torch.Tensor:
    """
    Return the exact inverse of a one-dimensional permutation.

    If ``permutation[s] = m``, the result satisfies ``inverse[m] = s``.
    """

    if not isinstance(
        permutation,
        torch.Tensor,
    ):
        raise TypeError(
            "permutation must be a tensor."
        )

    if permutation.ndim != 1:
        raise ValueError(
            "permutation must be one-dimensional."
        )

    if permutation.dtype != torch.long:
        raise ValueError(
            "permutation must use torch.int64."
        )

    size = int(
        permutation.numel()
    )

    _validate_permutation(
        permutation,
        size=size,
        name="permutation",
    )

    inverse = torch.empty_like(
        permutation
    )

    inverse.scatter_(
        0,
        permutation,
        torch.arange(
            size,
            dtype=torch.long,
            device=permutation.device,
        ),
    )

    return inverse


def build_recurrent_sort_permutations(
    nonempty_history_lengths: torch.Tensor,
    *,
    execution_path: (
        RecurrentExecutionPath
        | str
    ),
    enforce_sorted_lengths: bool,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    bool,
]:
    """
    Construct actual execution-order permutations.

    Returns
    -------
    sorted_to_nonempty:
        Sorted/execution position to nonempty-order position.

    nonempty_to_sorted:
        Exact inverse permutation.

    sort_was_applied:
        Whether the execution order differs from nonempty-node order.
    """

    lengths = _validate_lengths_tensor(
        nonempty_history_lengths,
        allow_empty_node_axis=True,
    )

    if bool(
        torch.any(
            lengths <= 0
        ).item()
    ):
        raise ValueError(
            "nonempty_history_lengths must be strictly positive."
        )

    path = _normalize_execution_path(
        execution_path
    )
    _require_boolean(
        "enforce_sorted_lengths",
        enforce_sorted_lengths,
    )

    size = int(
        lengths.numel()
    )
    identity = torch.arange(
        size,
        dtype=torch.long,
        device=lengths.device,
    )

    if path == RecurrentExecutionPath.REFERENCE:
        if enforce_sorted_lengths:
            raise ValueError(
                "Reference execution cannot enforce packed sorted lengths."
            )

        return (
            identity,
            identity.clone(),
            False,
        )

    if enforce_sorted_lengths:
        if not lengths_are_nonincreasing(
            lengths
        ):
            raise ValueError(
                "enforce_sorted_lengths=True requires nonempty-node "
                "history lengths to already be nonincreasing."
            )

        return (
            identity,
            identity.clone(),
            False,
        )

    sorted_to_nonempty = (
        stable_descending_length_permutation(
            lengths
        )
    )
    nonempty_to_sorted = inverse_permutation(
        sorted_to_nonempty
    )
    sort_was_applied = not torch.equal(
        sorted_to_nonempty,
        identity,
    )

    return (
        sorted_to_nonempty,
        nonempty_to_sorted,
        sort_was_applied,
    )


# =============================================================================
# Metadata assembly
# =============================================================================


def build_recurrent_execution_metadata_from_lengths(
    history_lengths: torch.Tensor,
    *,
    source_sequence_length: int,
    zero_length_policy: (
        HistoryZeroLengthPolicy
        | str
    ),
    original_padding_direction: (
        TemporalPaddingDirection
        | str
    ),
    execution_path: (
        RecurrentExecutionPath
        | str
    ),
    enforce_sorted_lengths: bool = False,
) -> RecurrentExecutionMetadata:
    """
    Build complete validated execution metadata from a length vector.

    This lower-level constructor is useful for isolated tests and preprocessing
    utilities. Normal encoder code should generally call
    ``build_recurrent_execution_metadata`` with the exact source history and
    recurrent configuration.
    """

    lengths = _validate_lengths_tensor(
        history_lengths,
        source_sequence_length=(
            source_sequence_length
        ),
    )
    policy = _normalize_zero_length_policy(
        zero_length_policy
    )
    padding_direction = _normalize_padding_direction(
        original_padding_direction
    )
    path = _normalize_execution_path(
        execution_path
    )
    _require_boolean(
        "enforce_sorted_lengths",
        enforce_sorted_lengths,
    )

    if (
        path == RecurrentExecutionPath.REFERENCE
        and enforce_sorted_lengths
    ):
        raise ValueError(
            "Reference execution cannot enforce packed sorted lengths."
        )

    validate_zero_history_policy(
        lengths,
        zero_length_policy=policy,
    )

    (
        nonempty_node_indices,
        zero_history_node_indices,
    ) = partition_recurrent_history_nodes(
        lengths
    )

    nonempty_lengths = (
        gather_nonempty_history_lengths(
            lengths,
            nonempty_node_indices,
        )
    )

    (
        sorted_to_nonempty,
        nonempty_to_sorted,
        sort_was_applied,
    ) = build_recurrent_sort_permutations(
        nonempty_lengths,
        execution_path=path,
        enforce_sorted_lengths=(
            enforce_sorted_lengths
        ),
    )

    return RecurrentExecutionMetadata(
        history_lengths=lengths,
        nonempty_node_indices=(
            nonempty_node_indices
        ),
        zero_history_node_indices=(
            zero_history_node_indices
        ),
        sorted_to_nonempty=(
            sorted_to_nonempty
        ),
        nonempty_to_sorted=(
            nonempty_to_sorted
        ),
        execution_path=path,
        sort_was_applied=(
            sort_was_applied
        ),
        original_padding_direction=(
            padding_direction
        ),
    )


def build_recurrent_execution_metadata(
    source_history: HistoricalSequenceInputs,
    config: RecurrentSequenceEncoderConfig,
) -> RecurrentExecutionMetadata:
    """
    Build the exact Phase 6 execution metadata for one source/config pair.
    """

    if not isinstance(
        source_history,
        HistoricalSequenceInputs,
    ):
        raise TypeError(
            "source_history must be a HistoricalSequenceInputs."
        )

    if not isinstance(
        config,
        RecurrentSequenceEncoderConfig,
    ):
        raise TypeError(
            "config must be a RecurrentSequenceEncoderConfig."
        )

    lengths = validate_source_zero_history_policy(
        source_history
    )

    execution_path = (
        RecurrentExecutionPath.PACKED
        if config.pack_sequences
        else RecurrentExecutionPath.REFERENCE
    )

    return build_recurrent_execution_metadata_from_lengths(
        lengths,
        source_sequence_length=(
            source_history.sequence_length
        ),
        zero_length_policy=(
            source_history.zero_length_policy
        ),
        original_padding_direction=(
            source_history.padding_direction
        ),
        execution_path=(
            execution_path
        ),
        enforce_sorted_lengths=(
            config.enforce_sorted_lengths
        ),
    )


# =============================================================================
# Compact aliases
# =============================================================================


derive_history_lengths = derive_recurrent_history_lengths
partition_history_nodes = partition_recurrent_history_nodes
stable_length_sort = stable_descending_length_permutation
build_execution_metadata = build_recurrent_execution_metadata


# =============================================================================
# Public API
# =============================================================================


__all__ = (
    # Module identity and policies.
    "RECURRENT_HISTORY_LENGTHS_IMPLEMENTATION_VERSION",
    "RECURRENT_HISTORY_LENGTHS_COMPONENT_NAME",
    "RECURRENT_HISTORY_LENGTHS_OPERATION_NAME",
    "RECURRENT_HISTORY_LENGTHS_SORT_POLICY",
    "RECURRENT_HISTORY_LENGTHS_ZERO_POLICY_SOURCE",
    "RECURRENT_HISTORY_LENGTHS_SCIENTIFIC_INTERPRETATION",

    # Length derivation and policy validation.
    "derive_recurrent_history_lengths",
    "validate_recurrent_history_lengths",
    "validate_zero_history_policy",
    "validate_source_zero_history_policy",

    # Node partitioning.
    "partition_recurrent_history_nodes",
    "gather_nonempty_history_lengths",

    # Sorting and permutations.
    "lengths_are_nonincreasing",
    "stable_descending_length_permutation",
    "inverse_permutation",
    "build_recurrent_sort_permutations",

    # Metadata assembly.
    "build_recurrent_execution_metadata_from_lengths",
    "build_recurrent_execution_metadata",

    # Compact aliases.
    "derive_history_lengths",
    "partition_history_nodes",
    "stable_length_sort",
    "build_execution_metadata",
)
