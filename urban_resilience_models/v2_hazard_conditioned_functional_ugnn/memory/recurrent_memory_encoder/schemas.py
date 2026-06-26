"""
Immutable runtime schemas for Phase 6 recurrent temporal-memory encoders.

Target path:
    urban_resilience_models/
        v2_hazard_conditioned_functional_ugnn/
            memory/
                recurrent_memory_encoder/
                    schemas.py

The public recurrent contract is deliberately split into three objects:

``RecurrentExecutionMetadata``
    Auditable node partitioning and execution-order metadata.

``RecurrentStateLayout``
    The semantic layer/direction layout of GRU and LSTM states.

``RecurrentSequenceEncoderRun``
    The shared ``TemporalSequenceEncoding`` together with recurrent-specific
    final hidden and optional cell states.

The shared public sequence output remains architecture-neutral:

    TemporalSequenceEncoding.encoded_sequence: [N, T, Hout]

Recurrent states use a research-facing canonical layout:

    [num_layers, num_directions, N, hidden_dim]

rather than PyTorch's flattened kernel layout:

    [num_layers * num_directions, N, hidden_dim]

Execution-order semantics
-------------------------
``nonempty_node_indices`` defines nonempty-node order as ascending original node
index.

For packed execution:

    sorted_to_nonempty[s]
        = nonempty-order position used at sorted position ``s``

    nonempty_to_sorted[m]
        = sorted position occupied by nonempty-order position ``m``

Therefore:

    sorted_values = nonempty_values[sorted_to_nonempty]
    nonempty_values = sorted_values[nonempty_to_sorted]

For reference execution, no sort is performed. Both permutations must be the
identity, and no descending-length invariant is imposed.

Zero-history semantics
----------------------
When the source history permits zero-history nodes, those nodes execute zero
recurrent steps. Their encoded rows, hidden states, and LSTM cell states must
remain exactly zero. No fake length-one observation is represented.

Mutation boundary
-----------------
Integer execution metadata tensors are detached and cloned during schema
construction. Recurrent value/state tensors are not cloned because they may
participate in autograd.

Interpretation
--------------------------
These schemas describe representation and execution structure. They do not
establish causal temporal effects, mechanistic interpretation, calibrated
uncertainty, or real-world feature importance.
"""

from __future__ import annotations

from dataclasses import dataclass
from dataclasses import replace as dataclass_replace
from enum import StrEnum
from hashlib import sha256
import json
from typing import Any, Final, Self

import torch

from ..schemas.sequence_encoding import (
    TemporalSequenceEncoderKind,
    TemporalSequenceEncoding,
)
from ..schemas.temporal_coordinates import (
    TemporalPaddingDirection,
)


# =============================================================================
# Schema identity and frozen semantic policies
# =============================================================================


RECURRENT_SCHEMAS_VERSION: Final[str] = "0.1"

RECURRENT_EXECUTION_METADATA_SCHEMA_VERSION: Final[str] = "0.1"

RECURRENT_STATE_LAYOUT_SCHEMA_VERSION: Final[str] = "0.1"

RECURRENT_SEQUENCE_ENCODER_RUN_SCHEMA_VERSION: Final[str] = "0.1"

RECURRENT_CANONICAL_BATCH_SCHEMA_VERSION: Final[str] = "0.1"

RECURRENT_CANONICAL_PADDING_DIRECTION: Final[
    TemporalPaddingDirection
] = TemporalPaddingDirection.RIGHT

RECURRENT_STATE_AXIS_ORDER: Final[str] = (
    "layer_direction_node_hidden"
)

RECURRENT_FLAT_STATE_AXIS_ORDER: Final[str] = (
    "layer_major_direction_minor"
)

RECURRENT_DIRECTION_FEATURE_ORDER: Final[str] = (
    "forward_then_backward"
)

RECURRENT_ZERO_HISTORY_STATE_POLICY: Final[str] = (
    "exact_zero_after_zero_recurrent_steps"
)

RECURRENT_EXECUTION_METADATA_SCIENTIFIC_INTERPRETATION: Final[str] = (
    "execution_order_and_partition_metadata_not_model_architecture"
)

RECURRENT_STATE_LAYOUT_SCIENTIFIC_INTERPRETATION: Final[str] = (
    "recurrent_state_axis_contract_not_generic_temporal_memory_summary"
)

RECURRENT_SEQUENCE_ENCODER_RUN_SCIENTIFIC_INTERPRETATION: Final[str] = (
    "recurrent_model_representation_and_states_not_causal_explanation"
)


# =============================================================================
# Controlled execution-path vocabulary
# =============================================================================


class RecurrentExecutionPath(StrEnum):
    """Execution strategy used for one recurrent sequence-encoding run."""

    PACKED = "packed"
    REFERENCE = "reference"


CANONICAL_RECURRENT_EXECUTION_PATHS: Final[
    tuple[str, ...]
] = tuple(
    member.value
    for member in RecurrentExecutionPath
)


# =============================================================================
# Generic validation and fingerprint helpers
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
            "Unknown recurrent execution path "
            f"{value!r}. Expected one of "
            f"{CANONICAL_RECURRENT_EXECUTION_PATHS}."
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
        allowed = tuple(
            member.value
            for member in TemporalPaddingDirection
        )
        raise ValueError(
            "Unknown temporal padding direction "
            f"{value!r}. Expected one of {allowed}."
        ) from error


def _canonical_json(
    value: Any,
) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(
            ",",
            ":",
        ),
        ensure_ascii=False,
        allow_nan=False,
    )


def _fingerprint(
    value: Any,
) -> str:
    return sha256(
        _canonical_json(
            value
        ).encode(
            "utf-8"
        )
    ).hexdigest()


def _tensor_fingerprint(
    tensors: dict[str, torch.Tensor],
) -> str:
    digest = sha256()

    for name in sorted(
        tensors
    ):
        tensor = tensors[
            name
        ]

        if not isinstance(
            tensor,
            torch.Tensor,
        ):
            raise TypeError(
                f"{name} must be a tensor."
            )

        if tensor.is_meta:
            raise ValueError(
                "Cannot fingerprint tensors on the meta device."
            )

        if tensor.layout != torch.strided:
            raise ValueError(
                "Only strided tensors are supported for fingerprinting."
            )

        canonical = (
            tensor
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
                canonical.dtype
            ).encode(
                "utf-8"
            )
        )
        digest.update(
            _canonical_json(
                list(
                    canonical.shape
                )
            ).encode(
                "utf-8"
            )
        )
        digest.update(
            canonical
            .view(
                torch.uint8
            )
            .numpy()
            .tobytes()
        )

    return digest.hexdigest()


def _clone_index_tensor(
    name: str,
    value: torch.Tensor,
) -> torch.Tensor:
    if not isinstance(
        value,
        torch.Tensor,
    ):
        raise TypeError(
            f"{name} must be a tensor."
        )

    if value.ndim != 1:
        raise ValueError(
            f"{name} must be one-dimensional; "
            f"observed shape {tuple(value.shape)}."
        )

    if value.dtype != torch.long:
        raise ValueError(
            f"{name} must use torch.int64."
        )

    if value.layout != torch.strided:
        raise ValueError(
            f"{name} must use strided tensor layout."
        )

    if value.is_meta:
        raise ValueError(
            f"{name} cannot reside on the meta device."
        )

    if value.requires_grad:
        raise ValueError(
            f"{name} must not require gradients."
        )

    return (
        value
        .detach()
        .clone()
    )


def _require_shared_device(
    tensors: dict[str, torch.Tensor],
) -> torch.device:
    if not tensors:
        raise ValueError(
            "At least one tensor is required."
        )

    iterator = iter(
        tensors.items()
    )
    first_name, first_tensor = next(
        iterator
    )
    device = first_tensor.device

    for name, tensor in iterator:
        if tensor.device != device:
            raise ValueError(
                f"{name} device {tensor.device} does not match "
                f"{first_name} device {device}."
            )

    return device


def _is_identity_permutation(
    permutation: torch.Tensor,
) -> bool:
    expected = torch.arange(
        permutation.numel(),
        dtype=torch.long,
        device=permutation.device,
    )

    return bool(
        torch.equal(
            permutation,
            expected,
        )
    )


def _validate_permutation(
    name: str,
    permutation: torch.Tensor,
    *,
    size: int,
) -> None:
    if permutation.numel() != size:
        raise ValueError(
            f"{name} must contain exactly {size} entries."
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


def _validate_strictly_increasing(
    name: str,
    values: torch.Tensor,
) -> None:
    if values.numel() <= 1:
        return

    if not bool(
        torch.all(
            values[1:]
            > values[:-1]
        ).item()
    ):
        raise ValueError(
            f"{name} must be strictly increasing."
        )


def _require_floating_state_tensor(
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

    if value.ndim != 4:
        raise ValueError(
            f"{name} must have canonical shape [L, directions, N, H]; "
            f"observed {tuple(value.shape)}."
        )

    if not value.dtype.is_floating_point:
        raise ValueError(
            f"{name} must use a floating-point dtype."
        )

    if value.layout != torch.strided:
        raise ValueError(
            f"{name} must use strided tensor layout."
        )

    if value.is_meta:
        raise ValueError(
            f"{name} cannot reside on the meta device."
        )

    if not bool(
        torch.isfinite(
            value
        ).all().item()
    ):
        raise ValueError(
            f"{name} must contain only finite values."
        )


def _require_exact_zero(
    name: str,
    value: torch.Tensor,
) -> None:
    if value.numel() == 0:
        return

    if bool(
        torch.any(
            value != 0
        ).item()
    ):
        raise ValueError(
            f"{name} must be exactly zero."
        )


# =============================================================================
# Recurrent execution metadata
# =============================================================================


@dataclass(
    slots=True,
    frozen=True,
)
class RecurrentExecutionMetadata:
    """
    Immutable node partition and execution-order metadata.

    ``nonempty_node_indices`` and ``zero_history_node_indices`` are strictly
    increasing original node indices.

    Packed-path permutation semantics
    ---------------------------------
    ``sorted_to_nonempty[s]`` is the nonempty-order position used at sorted
    position ``s``.

    ``nonempty_to_sorted[m]`` is the sorted position occupied by nonempty-order
    position ``m``.

    Reference-path semantics
    ------------------------
    Both permutations are identity because no sorting was executed.
    """

    history_lengths: torch.Tensor

    nonempty_node_indices: torch.Tensor
    zero_history_node_indices: torch.Tensor

    sorted_to_nonempty: torch.Tensor
    nonempty_to_sorted: torch.Tensor

    execution_path: (
        RecurrentExecutionPath
        | str
    )

    sort_was_applied: bool

    original_padding_direction: (
        TemporalPaddingDirection
        | str
    )

    schema_version: str = (
        RECURRENT_EXECUTION_METADATA_SCHEMA_VERSION
    )

    def __post_init__(
        self,
    ) -> None:
        history_lengths = _clone_index_tensor(
            "history_lengths",
            self.history_lengths,
        )
        nonempty_node_indices = _clone_index_tensor(
            "nonempty_node_indices",
            self.nonempty_node_indices,
        )
        zero_history_node_indices = _clone_index_tensor(
            "zero_history_node_indices",
            self.zero_history_node_indices,
        )
        sorted_to_nonempty = _clone_index_tensor(
            "sorted_to_nonempty",
            self.sorted_to_nonempty,
        )
        nonempty_to_sorted = _clone_index_tensor(
            "nonempty_to_sorted",
            self.nonempty_to_sorted,
        )

        _require_shared_device(
            {
                "history_lengths": history_lengths,
                "nonempty_node_indices": (
                    nonempty_node_indices
                ),
                "zero_history_node_indices": (
                    zero_history_node_indices
                ),
                "sorted_to_nonempty": (
                    sorted_to_nonempty
                ),
                "nonempty_to_sorted": (
                    nonempty_to_sorted
                ),
            }
        )

        if history_lengths.numel() <= 0:
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

        execution_path = _normalize_execution_path(
            self.execution_path
        )
        padding_direction = _normalize_padding_direction(
            self.original_padding_direction
        )
        _require_boolean(
            "sort_was_applied",
            self.sort_was_applied,
        )
        _require_nonempty_string(
            "schema_version",
            self.schema_version,
        )

        source_node_count = int(
            history_lengths.numel()
        )
        nonempty_count = int(
            nonempty_node_indices.numel()
        )
        zero_count = int(
            zero_history_node_indices.numel()
        )

        if (
            nonempty_count
            + zero_count
            != source_node_count
        ):
            raise ValueError(
                "Nonempty and zero-history node counts must partition "
                "the complete source node axis."
            )

        _validate_strictly_increasing(
            "nonempty_node_indices",
            nonempty_node_indices,
        )
        _validate_strictly_increasing(
            "zero_history_node_indices",
            zero_history_node_indices,
        )

        complete_partition = torch.cat(
            (
                nonempty_node_indices,
                zero_history_node_indices,
            ),
            dim=0,
        )

        expected_nodes = torch.arange(
            source_node_count,
            dtype=torch.long,
            device=history_lengths.device,
        )

        if not torch.equal(
            torch.sort(
                complete_partition
            ).values,
            expected_nodes,
        ):
            raise ValueError(
                "nonempty_node_indices and zero_history_node_indices "
                "must be unique, disjoint, and jointly equal range(N)."
            )

        if nonempty_count > 0:
            nonempty_lengths = history_lengths.index_select(
                0,
                nonempty_node_indices,
            )

            if bool(
                torch.any(
                    nonempty_lengths <= 0
                ).item()
            ):
                raise ValueError(
                    "Every nonempty node must have positive history length."
                )
        else:
            nonempty_lengths = history_lengths.new_empty(
                (
                    0,
                )
            )

        if zero_count > 0:
            zero_lengths = history_lengths.index_select(
                0,
                zero_history_node_indices,
            )

            if bool(
                torch.any(
                    zero_lengths != 0
                ).item()
            ):
                raise ValueError(
                    "Every zero-history node must have history length zero."
                )

        _validate_permutation(
            "sorted_to_nonempty",
            sorted_to_nonempty,
            size=nonempty_count,
        )
        _validate_permutation(
            "nonempty_to_sorted",
            nonempty_to_sorted,
            size=nonempty_count,
        )

        expected_positions = torch.arange(
            nonempty_count,
            dtype=torch.long,
            device=history_lengths.device,
        )

        if not torch.equal(
            sorted_to_nonempty.index_select(
                0,
                nonempty_to_sorted,
            ),
            expected_positions,
        ):
            raise ValueError(
                "sorted_to_nonempty and nonempty_to_sorted must be "
                "exact inverse permutations."
            )

        if not torch.equal(
            nonempty_to_sorted.index_select(
                0,
                sorted_to_nonempty,
            ),
            expected_positions,
        ):
            raise ValueError(
                "nonempty_to_sorted and sorted_to_nonempty must be "
                "exact inverse permutations."
            )

        identity_sort = _is_identity_permutation(
            sorted_to_nonempty
        )

        if execution_path == RecurrentExecutionPath.REFERENCE:
            if not identity_sort:
                raise ValueError(
                    "Reference execution must use identity permutations."
                )

            if not _is_identity_permutation(
                nonempty_to_sorted
            ):
                raise ValueError(
                    "Reference execution must use identity permutations."
                )

            if self.sort_was_applied:
                raise ValueError(
                    "Reference execution cannot report sorting."
                )

        else:
            sorted_lengths = nonempty_lengths.index_select(
                0,
                sorted_to_nonempty,
            )

            if sorted_lengths.numel() > 1:
                if not bool(
                    torch.all(
                        sorted_lengths[:-1]
                        >= sorted_lengths[1:]
                    ).item()
                ):
                    raise ValueError(
                        "Packed execution lengths must be nonincreasing "
                        "in sorted order."
                    )

                equal_adjacent = (
                    sorted_lengths[:-1]
                    == sorted_lengths[1:]
                )

                if bool(
                    equal_adjacent.any().item()
                ):
                    left_positions = (
                        sorted_to_nonempty[:-1][
                            equal_adjacent
                        ]
                    )
                    right_positions = (
                        sorted_to_nonempty[1:][
                            equal_adjacent
                        ]
                    )

                    if not bool(
                        torch.all(
                            left_positions
                            < right_positions
                        ).item()
                    ):
                        raise ValueError(
                            "Equal-length packed nodes must preserve stable "
                            "nonempty-node order."
                        )

            expected_sort_was_applied = (
                not identity_sort
            )

            if (
                self.sort_was_applied
                != expected_sort_was_applied
            ):
                raise ValueError(
                    "sort_was_applied must exactly match whether the "
                    "packed permutation is nonidentity."
                )

        object.__setattr__(
            self,
            "history_lengths",
            history_lengths,
        )
        object.__setattr__(
            self,
            "nonempty_node_indices",
            nonempty_node_indices,
        )
        object.__setattr__(
            self,
            "zero_history_node_indices",
            zero_history_node_indices,
        )
        object.__setattr__(
            self,
            "sorted_to_nonempty",
            sorted_to_nonempty,
        )
        object.__setattr__(
            self,
            "nonempty_to_sorted",
            nonempty_to_sorted,
        )
        object.__setattr__(
            self,
            "execution_path",
            execution_path,
        )
        object.__setattr__(
            self,
            "original_padding_direction",
            padding_direction,
        )

    # -------------------------------------------------------------------------
    # Structural properties
    # -------------------------------------------------------------------------

    @property
    def source_node_count(
        self,
    ) -> int:
        return int(
            self.history_lengths.numel()
        )

    @property
    def node_count(
        self,
    ) -> int:
        return self.source_node_count

    @property
    def nonempty_node_count(
        self,
    ) -> int:
        return int(
            self.nonempty_node_indices.numel()
        )

    @property
    def zero_history_count(
        self,
    ) -> int:
        return int(
            self.zero_history_node_indices.numel()
        )

    @property
    def has_zero_history(
        self,
    ) -> bool:
        return self.zero_history_count > 0

    @property
    def all_zero_history(
        self,
    ) -> bool:
        return self.nonempty_node_count == 0

    @property
    def device(
        self,
    ) -> torch.device:
        return self.history_lengths.device

    @property
    def canonical_padding_direction(
        self,
    ) -> TemporalPaddingDirection:
        return RECURRENT_CANONICAL_PADDING_DIRECTION

    @property
    def nonempty_history_lengths(
        self,
    ) -> torch.Tensor:
        return self.history_lengths.index_select(
            0,
            self.nonempty_node_indices,
        )

    @property
    def sorted_history_lengths(
        self,
    ) -> torch.Tensor:
        return self.nonempty_history_lengths.index_select(
            0,
            self.sorted_to_nonempty,
        )

    @property
    def sorted_node_indices(
        self,
    ) -> torch.Tensor:
        return self.nonempty_node_indices.index_select(
            0,
            self.sorted_to_nonempty,
        )

    @property
    def identity_permutation(
        self,
    ) -> bool:
        return _is_identity_permutation(
            self.sorted_to_nonempty
        )

    # -------------------------------------------------------------------------
    # Deterministic identity
    # -------------------------------------------------------------------------

    def semantic_dict(
        self,
    ) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "execution_path": self.execution_path.value,
            "sort_was_applied": self.sort_was_applied,
            "original_padding_direction": (
                self.original_padding_direction.value
            ),
            "canonical_padding_direction": (
                self.canonical_padding_direction.value
            ),
            "source_node_count": self.source_node_count,
            "nonempty_node_count": self.nonempty_node_count,
            "zero_history_count": self.zero_history_count,
            "scientific_interpretation": (
                RECURRENT_EXECUTION_METADATA_SCIENTIFIC_INTERPRETATION
            ),
        }

    def value_fingerprint(
        self,
    ) -> str:
        return _tensor_fingerprint(
            {
                "history_lengths": self.history_lengths,
                "nonempty_node_indices": (
                    self.nonempty_node_indices
                ),
                "zero_history_node_indices": (
                    self.zero_history_node_indices
                ),
                "sorted_to_nonempty": (
                    self.sorted_to_nonempty
                ),
                "nonempty_to_sorted": (
                    self.nonempty_to_sorted
                ),
            }
        )

    def fingerprint(
        self,
    ) -> str:
        return _fingerprint(
            {
                "semantic": self.semantic_dict(),
                "value_fingerprint": self.value_fingerprint(),
            }
        )

    # -------------------------------------------------------------------------
    # Validated reconstruction
    # -------------------------------------------------------------------------

    def to(
        self,
        device: torch.device | str,
        *,
        non_blocking: bool = False,
    ) -> Self:
        return type(self)(
            history_lengths=self.history_lengths.to(
                device=device,
                non_blocking=non_blocking,
            ),
            nonempty_node_indices=(
                self.nonempty_node_indices.to(
                    device=device,
                    non_blocking=non_blocking,
                )
            ),
            zero_history_node_indices=(
                self.zero_history_node_indices.to(
                    device=device,
                    non_blocking=non_blocking,
                )
            ),
            sorted_to_nonempty=(
                self.sorted_to_nonempty.to(
                    device=device,
                    non_blocking=non_blocking,
                )
            ),
            nonempty_to_sorted=(
                self.nonempty_to_sorted.to(
                    device=device,
                    non_blocking=non_blocking,
                )
            ),
            execution_path=self.execution_path,
            sort_was_applied=self.sort_was_applied,
            original_padding_direction=(
                self.original_padding_direction
            ),
            schema_version=self.schema_version,
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
# Recurrent state-axis contract
# =============================================================================


@dataclass(
    slots=True,
    frozen=True,
)
class RecurrentStateLayout:
    """
    Semantic layout of recurrent hidden and optional LSTM cell states.

    Canonical public state shape:

        [num_layers, num_directions, N, hidden_dim]

    PyTorch flat state shape:

        [num_layers * num_directions, N, hidden_dim]
    """

    num_layers: int
    num_directions: int
    hidden_dim: int

    schema_version: str = (
        RECURRENT_STATE_LAYOUT_SCHEMA_VERSION
    )

    def __post_init__(
        self,
    ) -> None:
        _require_positive_int(
            "num_layers",
            self.num_layers,
        )
        _require_positive_int(
            "num_directions",
            self.num_directions,
        )
        _require_positive_int(
            "hidden_dim",
            self.hidden_dim,
        )
        _require_nonempty_string(
            "schema_version",
            self.schema_version,
        )

        if self.num_directions not in (
            1,
            2,
        ):
            raise ValueError(
                "num_directions must be 1 or 2."
            )

    @property
    def direction_order(
        self,
    ) -> tuple[str, ...]:
        if self.num_directions == 1:
            return (
                "forward",
            )

        return (
            "forward",
            "backward",
        )

    @property
    def state_axis_order(
        self,
    ) -> str:
        return RECURRENT_STATE_AXIS_ORDER

    @property
    def flat_axis_order(
        self,
    ) -> str:
        return RECURRENT_FLAT_STATE_AXIS_ORDER

    @property
    def flat_layer_direction_size(
        self,
    ) -> int:
        return (
            self.num_layers
            * self.num_directions
        )

    @property
    def output_dim(
        self,
    ) -> int:
        return (
            self.hidden_dim
            * self.num_directions
        )

    @property
    def is_bidirectional(
        self,
    ) -> bool:
        return self.num_directions == 2

    def flat_index(
        self,
        *,
        layer_index: int,
        direction_index: int,
    ) -> int:
        _require_nonnegative_int(
            "layer_index",
            layer_index,
        )
        _require_nonnegative_int(
            "direction_index",
            direction_index,
        )

        if layer_index >= self.num_layers:
            raise IndexError(
                "layer_index is outside the recurrent layer axis."
            )

        if direction_index >= self.num_directions:
            raise IndexError(
                "direction_index is outside the recurrent direction axis."
            )

        return (
            layer_index
            * self.num_directions
            + direction_index
        )

    def validate_canonical_state(
        self,
        state: torch.Tensor,
        *,
        name: str = "state",
        node_count: int | None = None,
    ) -> None:
        _require_floating_state_tensor(
            name,
            state,
        )

        if (
            int(
                state.shape[0]
            )
            != self.num_layers
        ):
            raise ValueError(
                f"{name} layer dimension must equal {self.num_layers}."
            )

        if (
            int(
                state.shape[1]
            )
            != self.num_directions
        ):
            raise ValueError(
                f"{name} direction dimension must equal "
                f"{self.num_directions}."
            )

        if (
            int(
                state.shape[3]
            )
            != self.hidden_dim
        ):
            raise ValueError(
                f"{name} hidden dimension must equal {self.hidden_dim}."
            )

        if node_count is not None:
            _require_nonnegative_int(
                "node_count",
                node_count,
            )

            if (
                int(
                    state.shape[2]
                )
                != node_count
            ):
                raise ValueError(
                    f"{name} node dimension must equal {node_count}."
                )

    def validate_flat_state(
        self,
        state: torch.Tensor,
        *,
        name: str = "flat_state",
        node_count: int | None = None,
    ) -> None:
        if not isinstance(
            state,
            torch.Tensor,
        ):
            raise TypeError(
                f"{name} must be a tensor."
            )

        if state.ndim != 3:
            raise ValueError(
                f"{name} must have flat shape "
                "[L * directions, N, H]."
            )

        if not state.dtype.is_floating_point:
            raise ValueError(
                f"{name} must use a floating-point dtype."
            )

        if state.is_meta:
            raise ValueError(
                f"{name} cannot reside on the meta device."
            )

        if state.layout != torch.strided:
            raise ValueError(
                f"{name} must use strided tensor layout."
            )

        if not bool(
            torch.isfinite(
                state
            ).all().item()
        ):
            raise ValueError(
                f"{name} must contain only finite values."
            )

        if (
            int(
                state.shape[0]
            )
            != self.flat_layer_direction_size
        ):
            raise ValueError(
                f"{name} leading dimension must equal "
                f"{self.flat_layer_direction_size}."
            )

        if (
            int(
                state.shape[2]
            )
            != self.hidden_dim
        ):
            raise ValueError(
                f"{name} hidden dimension must equal {self.hidden_dim}."
            )

        if node_count is not None:
            _require_nonnegative_int(
                "node_count",
                node_count,
            )

            if int(
                state.shape[1]
            ) != node_count:
                raise ValueError(
                    f"{name} node dimension must equal {node_count}."
                )

    def flatten_state(
        self,
        state: torch.Tensor,
        *,
        name: str = "state",
    ) -> torch.Tensor:
        self.validate_canonical_state(
            state,
            name=name,
        )

        return state.reshape(
            self.flat_layer_direction_size,
            int(
                state.shape[2]
            ),
            self.hidden_dim,
        )

    def unflatten_state(
        self,
        state: torch.Tensor,
        *,
        name: str = "flat_state",
    ) -> torch.Tensor:
        self.validate_flat_state(
            state,
            name=name,
        )

        return state.reshape(
            self.num_layers,
            self.num_directions,
            int(
                state.shape[1]
            ),
            self.hidden_dim,
        )

    def semantic_dict(
        self,
    ) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "num_layers": self.num_layers,
            "num_directions": self.num_directions,
            "hidden_dim": self.hidden_dim,
            "output_dim": self.output_dim,
            "direction_order": list(
                self.direction_order
            ),
            "state_axis_order": self.state_axis_order,
            "flat_axis_order": self.flat_axis_order,
            "scientific_interpretation": (
                RECURRENT_STATE_LAYOUT_SCIENTIFIC_INTERPRETATION
            ),
        }

    def fingerprint(
        self,
    ) -> str:
        return _fingerprint(
            self.semantic_dict()
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
# Recurrent-specific execution result
# =============================================================================


@dataclass(
    slots=True,
    frozen=True,
)
class RecurrentSequenceEncoderRun:
    """
    Recurrent-specific result while preserving the shared public contract.

    ``public_output`` remains the object consumed by generic downstream memory
    components. Hidden and optional cell states are retained only for recurrent
    diagnostics, audits, and explicitly recurrent-aware consumers.
    """

    public_output: TemporalSequenceEncoding

    final_hidden_state: torch.Tensor
    final_cell_state: torch.Tensor | None

    state_layout: RecurrentStateLayout
    execution_metadata: RecurrentExecutionMetadata

    run_name: str = (
        "recurrent_sequence_encoder_run"
    )

    schema_version: str = (
        RECURRENT_SEQUENCE_ENCODER_RUN_SCHEMA_VERSION
    )

    def __post_init__(
        self,
    ) -> None:
        if not isinstance(
            self.public_output,
            TemporalSequenceEncoding,
        ):
            raise TypeError(
                "public_output must be a TemporalSequenceEncoding."
            )

        if not isinstance(
            self.state_layout,
            RecurrentStateLayout,
        ):
            raise TypeError(
                "state_layout must be a RecurrentStateLayout."
            )

        if not isinstance(
            self.execution_metadata,
            RecurrentExecutionMetadata,
        ):
            raise TypeError(
                "execution_metadata must be a "
                "RecurrentExecutionMetadata."
            )

        _require_nonempty_string(
            "run_name",
            self.run_name,
        )
        _require_nonempty_string(
            "schema_version",
            self.schema_version,
        )

        encoder_kind = (
            self
            .public_output
            .encoder_kind
        )

        if encoder_kind not in (
            TemporalSequenceEncoderKind.GRU,
            TemporalSequenceEncoderKind.LSTM,
        ):
            raise ValueError(
                "RecurrentSequenceEncoderRun requires a GRU or LSTM "
                "public output."
            )

        node_count = (
            self
            .public_output
            .node_count
        )

        self.state_layout.validate_canonical_state(
            self.final_hidden_state,
            name="final_hidden_state",
            node_count=node_count,
        )

        if (
            self.public_output.hidden_dim
            != self.state_layout.output_dim
        ):
            raise ValueError(
                "public_output hidden width must equal "
                "hidden_dim * num_directions."
            )

        if (
            self.final_hidden_state.device
            != self.public_output.device
        ):
            raise ValueError(
                "final_hidden_state device must match public_output."
            )

        if (
            self.final_hidden_state.dtype
            != self.public_output.dtype
        ):
            raise ValueError(
                "final_hidden_state dtype must match public_output."
            )

        if (
            self.public_output.device
            != self.public_output.source_history.device
        ):
            raise ValueError(
                "public_output and source history devices must match."
            )

        if (
            self.public_output.dtype
            != self.public_output.source_history.dtype
        ):
            raise ValueError(
                "public_output and source history dtypes must match."
            )

        if encoder_kind == TemporalSequenceEncoderKind.GRU:
            if self.final_cell_state is not None:
                raise ValueError(
                    "GRU runs must not contain final_cell_state."
                )

        else:
            if self.final_cell_state is None:
                raise ValueError(
                    "LSTM runs require final_cell_state."
                )

            self.state_layout.validate_canonical_state(
                self.final_cell_state,
                name="final_cell_state",
                node_count=node_count,
            )

            if (
                self.final_cell_state.device
                != self.final_hidden_state.device
            ):
                raise ValueError(
                    "final_cell_state device must match hidden state."
                )

            if (
                self.final_cell_state.dtype
                != self.final_hidden_state.dtype
            ):
                raise ValueError(
                    "final_cell_state dtype must match hidden state."
                )

        if (
            self.execution_metadata.source_node_count
            != node_count
        ):
            raise ValueError(
                "execution metadata node count must match public output."
            )

        if (
            self.execution_metadata.device
            != self.public_output.device
        ):
            raise ValueError(
                "execution metadata device must match public output."
            )

        expected_lengths = (
            self
            .public_output
            .source_history
            .valid_lengths
        )

        if not torch.equal(
            self.execution_metadata.history_lengths,
            expected_lengths,
        ):
            raise ValueError(
                "execution metadata history_lengths must exactly equal "
                "source_history.timestep_mask.sum(dim=1)."
            )

        if bool(
            torch.any(
                self.execution_metadata.history_lengths
                > self.public_output.sequence_length
            ).item()
        ):
            raise ValueError(
                "history lengths cannot exceed the public sequence length."
            )

        if (
            self.execution_metadata.original_padding_direction
            != self.public_output.source_history.padding_direction
        ):
            raise ValueError(
                "execution metadata original padding direction must "
                "match source history."
            )

        zero_indices = (
            self
            .execution_metadata
            .zero_history_node_indices
        )

        if zero_indices.numel() > 0:
            zero_hidden = (
                self
                .final_hidden_state
                .index_select(
                    2,
                    zero_indices,
                )
            )
            _require_exact_zero(
                "zero-history hidden states",
                zero_hidden,
            )

            if self.final_cell_state is not None:
                zero_cell = (
                    self
                    .final_cell_state
                    .index_select(
                        2,
                        zero_indices,
                    )
                )
                _require_exact_zero(
                    "zero-history cell states",
                    zero_cell,
                )

            zero_encoded = (
                self
                .public_output
                .encoded_sequence
                .index_select(
                    0,
                    zero_indices,
                )
            )
            _require_exact_zero(
                "zero-history encoded sequences",
                zero_encoded,
            )

    # -------------------------------------------------------------------------
    # Structural properties
    # -------------------------------------------------------------------------

    @property
    def source_history(
        self,
    ):
        return self.public_output.source_history

    @property
    def encoder_kind(
        self,
    ) -> TemporalSequenceEncoderKind:
        return self.public_output.encoder_kind

    @property
    def node_count(
        self,
    ) -> int:
        return self.public_output.node_count

    @property
    def sequence_length(
        self,
    ) -> int:
        return self.public_output.sequence_length

    @property
    def hidden_dim(
        self,
    ) -> int:
        return self.state_layout.hidden_dim

    @property
    def output_dim(
        self,
    ) -> int:
        return self.public_output.hidden_dim

    @property
    def num_layers(
        self,
    ) -> int:
        return self.state_layout.num_layers

    @property
    def num_directions(
        self,
    ) -> int:
        return self.state_layout.num_directions

    @property
    def is_bidirectional(
        self,
    ) -> bool:
        return self.state_layout.is_bidirectional

    @property
    def device(
        self,
    ) -> torch.device:
        return self.public_output.device

    @property
    def dtype(
        self,
    ) -> torch.dtype:
        return self.public_output.dtype

    @property
    def final_hidden_state_flat(
        self,
    ) -> torch.Tensor:
        return self.state_layout.flatten_state(
            self.final_hidden_state,
            name="final_hidden_state",
        )

    @property
    def final_cell_state_flat(
        self,
    ) -> torch.Tensor | None:
        if self.final_cell_state is None:
            return None

        return self.state_layout.flatten_state(
            self.final_cell_state,
            name="final_cell_state",
        )

    @property
    def has_cell_state(
        self,
    ) -> bool:
        return self.final_cell_state is not None

    @property
    def architecture_fingerprint(
        self,
    ) -> str:
        return (
            self
            .public_output
            .architecture_fingerprint
        )

    @property
    def computation_lineage_fingerprint(
        self,
    ) -> str:
        return (
            self
            .public_output
            .computation_lineage_fingerprint
        )

    # -------------------------------------------------------------------------
    # Deterministic identity
    # -------------------------------------------------------------------------

    def semantic_dict(
        self,
    ) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "run_name": self.run_name,
            "encoder_kind": self.encoder_kind.value,
            "node_count": self.node_count,
            "sequence_length": self.sequence_length,
            "hidden_dim": self.hidden_dim,
            "output_dim": self.output_dim,
            "num_layers": self.num_layers,
            "num_directions": self.num_directions,
            "has_cell_state": self.has_cell_state,
            "state_layout_fingerprint": (
                self.state_layout.fingerprint()
            ),
            "execution_metadata_fingerprint": (
                self.execution_metadata.fingerprint()
            ),
            "public_output_lineage_fingerprint": (
                self.public_output.lineage_fingerprint()
            ),
            "zero_history_state_policy": (
                RECURRENT_ZERO_HISTORY_STATE_POLICY
            ),
            "scientific_interpretation": (
                RECURRENT_SEQUENCE_ENCODER_RUN_SCIENTIFIC_INTERPRETATION
            ),
        }

    def value_fingerprint(
        self,
    ) -> str:
        tensors = {
            "final_hidden_state": (
                self.final_hidden_state
            ),
        }

        if self.final_cell_state is not None:
            tensors[
                "final_cell_state"
            ] = self.final_cell_state

        return _tensor_fingerprint(
            tensors
        )

    def lineage_dict(
        self,
    ) -> dict[str, Any]:
        return {
            "semantic": self.semantic_dict(),
            "value_fingerprint": self.value_fingerprint(),
            "architecture_fingerprint": (
                self.architecture_fingerprint
            ),
            "computation_lineage_fingerprint": (
                self.computation_lineage_fingerprint
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

    def replace(
        self,
        **changes: Any,
    ) -> Self:
        return dataclass_replace(
            self,
            **changes,
        )


# =============================================================================
# Private canonical recurrent batch
# =============================================================================


@dataclass(
    slots=True,
    frozen=True,
)
class _CanonicalRecurrentBatch:
    """
    Package-private canonical right-padded nonempty-node batch.

    ``values`` may contain raw model-facing features or input-adapted features.
    It preserves autograd and is therefore not cloned.

    This object contains only nonempty nodes. ``M`` may be zero for an
    all-zero-history short-circuit preparation, but every represented row must
    otherwise have positive length.
    """

    values: torch.Tensor
    timestep_mask: torch.Tensor
    lengths: torch.Tensor
    nonempty_node_indices: torch.Tensor

    source_node_count: int

    original_padding_direction: (
        TemporalPaddingDirection
        | str
    )

    value_stage: str = "canonical_raw_history"

    schema_version: str = (
        RECURRENT_CANONICAL_BATCH_SCHEMA_VERSION
    )

    def __post_init__(
        self,
    ) -> None:
        if not isinstance(
            self.values,
            torch.Tensor,
        ):
            raise TypeError(
                "values must be a tensor."
            )

        if self.values.ndim != 3:
            raise ValueError(
                "values must have shape [M, T, F]."
            )

        if not self.values.dtype.is_floating_point:
            raise ValueError(
                "values must use a floating-point dtype."
            )

        if self.values.is_meta:
            raise ValueError(
                "values cannot reside on the meta device."
            )

        if self.values.layout != torch.strided:
            raise ValueError(
                "values must use strided tensor layout."
            )

        if not bool(
            torch.isfinite(
                self.values
            ).all().item()
        ):
            raise ValueError(
                "values must contain only finite values."
            )

        nonempty_count = int(
            self.values.shape[0]
        )
        sequence_length = int(
            self.values.shape[1]
        )
        feature_dim = int(
            self.values.shape[2]
        )

        if sequence_length <= 0 or feature_dim <= 0:
            raise ValueError(
                "Canonical sequence length and feature width must be "
                "strictly positive."
            )

        if not isinstance(
            self.timestep_mask,
            torch.Tensor,
        ):
            raise TypeError(
                "timestep_mask must be a tensor."
            )

        if self.timestep_mask.dtype != torch.bool:
            raise ValueError(
                "timestep_mask must use torch.bool."
            )

        if tuple(
            self.timestep_mask.shape
        ) != (
            nonempty_count,
            sequence_length,
        ):
            raise ValueError(
                "timestep_mask must have shape [M, T] aligned with values."
            )

        if (
            self.timestep_mask.device
            != self.values.device
        ):
            raise ValueError(
                "timestep_mask device must match values."
            )

        lengths = _clone_index_tensor(
            "lengths",
            self.lengths,
        )
        nonempty_node_indices = _clone_index_tensor(
            "nonempty_node_indices",
            self.nonempty_node_indices,
        )

        if lengths.device != self.values.device:
            raise ValueError(
                "lengths device must match values."
            )

        if (
            nonempty_node_indices.device
            != self.values.device
        ):
            raise ValueError(
                "nonempty_node_indices device must match values."
            )

        if lengths.numel() != nonempty_count:
            raise ValueError(
                "lengths must contain one entry per canonical row."
            )

        if (
            nonempty_node_indices.numel()
            != nonempty_count
        ):
            raise ValueError(
                "nonempty_node_indices must contain one entry per "
                "canonical row."
            )

        _require_positive_int(
            "source_node_count",
            self.source_node_count,
        )

        if nonempty_count > self.source_node_count:
            raise ValueError(
                "Canonical nonempty-node count cannot exceed source count."
            )

        _validate_strictly_increasing(
            "nonempty_node_indices",
            nonempty_node_indices,
        )

        if nonempty_count > 0:
            if bool(
                torch.any(
                    lengths <= 0
                ).item()
            ):
                raise ValueError(
                    "Every canonical row must have positive length."
                )

            if bool(
                torch.any(
                    lengths > sequence_length
                ).item()
            ):
                raise ValueError(
                    "Canonical lengths cannot exceed T."
                )

            if bool(
                torch.any(
                    nonempty_node_indices < 0
                ).item()
            ) or bool(
                torch.any(
                    nonempty_node_indices
                    >= self.source_node_count
                ).item()
            ):
                raise ValueError(
                    "nonempty_node_indices must lie in range(source_node_count)."
                )

        expected_lengths = self.timestep_mask.sum(
            dim=1,
            dtype=torch.long,
        )

        if not torch.equal(
            lengths,
            expected_lengths,
        ):
            raise ValueError(
                "lengths must exactly equal timestep_mask.sum(dim=1)."
            )

        positions = torch.arange(
            sequence_length,
            dtype=torch.long,
            device=self.values.device,
        ).unsqueeze(
            0
        )

        expected_mask = (
            positions
            < lengths.unsqueeze(
                1
            )
        )

        if not torch.equal(
            self.timestep_mask,
            expected_mask,
        ):
            raise ValueError(
                "Canonical timestep_mask must be right-padded with a "
                "contiguous valid prefix."
            )

        padded_values = self.values.masked_select(
            (
                ~self.timestep_mask
            ).unsqueeze(
                -1
            ).expand_as(
                self.values
            )
        )

        _require_exact_zero(
            "canonical padded values",
            padded_values,
        )

        padding_direction = _normalize_padding_direction(
            self.original_padding_direction
        )
        _require_nonempty_string(
            "value_stage",
            self.value_stage,
        )
        _require_nonempty_string(
            "schema_version",
            self.schema_version,
        )

        object.__setattr__(
            self,
            "lengths",
            lengths,
        )
        object.__setattr__(
            self,
            "nonempty_node_indices",
            nonempty_node_indices,
        )
        object.__setattr__(
            self,
            "original_padding_direction",
            padding_direction,
        )

    @property
    def nonempty_node_count(
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
    def feature_dim(
        self,
    ) -> int:
        return int(
            self.values.shape[2]
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

    @property
    def canonical_padding_direction(
        self,
    ) -> TemporalPaddingDirection:
        return RECURRENT_CANONICAL_PADDING_DIRECTION


# =============================================================================
# Compact aliases
# =============================================================================


RecurrentEncoderRun = RecurrentSequenceEncoderRun
RecurrentRun = RecurrentSequenceEncoderRun
RecurrentStateAxisLayout = RecurrentStateLayout


# =============================================================================
# Public API
# =============================================================================


__all__ = (
    # Schema identity and fixed interpretation.
    "RECURRENT_SCHEMAS_VERSION",
    "RECURRENT_EXECUTION_METADATA_SCHEMA_VERSION",
    "RECURRENT_STATE_LAYOUT_SCHEMA_VERSION",
    "RECURRENT_SEQUENCE_ENCODER_RUN_SCHEMA_VERSION",
    "RECURRENT_CANONICAL_PADDING_DIRECTION",
    "RECURRENT_STATE_AXIS_ORDER",
    "RECURRENT_FLAT_STATE_AXIS_ORDER",
    "RECURRENT_DIRECTION_FEATURE_ORDER",
    "RECURRENT_ZERO_HISTORY_STATE_POLICY",
    "RECURRENT_EXECUTION_METADATA_SCIENTIFIC_INTERPRETATION",
    "RECURRENT_STATE_LAYOUT_SCIENTIFIC_INTERPRETATION",
    "RECURRENT_SEQUENCE_ENCODER_RUN_SCIENTIFIC_INTERPRETATION",

    # Execution-path vocabulary.
    "RecurrentExecutionPath",
    "CANONICAL_RECURRENT_EXECUTION_PATHS",

    # Public recurrent schemas.
    "RecurrentExecutionMetadata",
    "RecurrentStateLayout",
    "RecurrentSequenceEncoderRun",

    # Compact aliases.
    "RecurrentEncoderRun",
    "RecurrentRun",
    "RecurrentStateAxisLayout",
)
