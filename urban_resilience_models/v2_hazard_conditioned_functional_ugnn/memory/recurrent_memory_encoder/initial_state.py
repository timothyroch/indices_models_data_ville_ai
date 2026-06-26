"""
Exact-zero initial-state construction for Phase 6 recurrent memory encoders.

Target path:
    urban_resilience_models/
        v2_hazard_conditioned_functional_ugnn/
            memory/
                recurrent_memory_encoder/
                    initial_state.py

Phase 6 supports one initialization policy only:

    exact-zero hidden state
    exact-zero cell state for LSTM

Caller-supplied, learned, stateful, streaming, and truncated-backpropagation
initial states are deliberately postponed.

Kernel-facing states use PyTorch's flat layout:

    [num_layers * num_directions, batch_size, hidden_dim]

Research-facing recurrent run states use the canonical layout:

    [num_layers, num_directions, node_count, hidden_dim]

This module constructs both forms and validates that a configured ``nn.GRU`` or
``nn.LSTM`` exactly matches the frozen recurrent configuration before any state
is allocated.

Zero-history semantics
----------------------
A zero-history node executes zero recurrent transitions. Under the Phase 6
zero-initialization policy, its final hidden and optional cell states therefore
remain exactly zero.

The all-zero-history short circuit may use the canonical full-node helpers in
this module without executing the recurrent kernel. Source/module dtype-device
compatibility and explicit parameter snapshots must still be validated by the
complete encoder before that short circuit.

Dtype and device
----------------
Initial states are allocated from the recurrent kernel's floating parameters,
not from caller-provided dtype/device strings. This guarantees exact alignment
with the actual GRU/LSTM kernel.
"""

from __future__ import annotations

import math
from typing import Final
from typing import TypeAlias

import torch
from torch import nn

from ..config import (
    RecurrentCellKind,
    RecurrentSequenceEncoderConfig,
)
from .schemas import (
    RecurrentStateLayout,
)


# =============================================================================
# Component identity and frozen policy
# =============================================================================


RECURRENT_INITIAL_STATE_IMPLEMENTATION_VERSION: Final[str] = "0.1"

RECURRENT_INITIAL_STATE_COMPONENT_NAME: Final[str] = (
    "recurrent_initial_state"
)

RECURRENT_INITIAL_STATE_COMPONENT_KIND: Final[str] = (
    "exact_zero_recurrent_state_factory"
)

RECURRENT_INITIAL_STATE_OPERATION_NAME: Final[str] = (
    "construct_exact_zero_recurrent_state"
)

RECURRENT_INITIAL_STATE_POLICY: Final[str] = "exact_zero_v1"

RECURRENT_INITIAL_STATE_FLAT_AXIS_ORDER: Final[str] = (
    "layer_major_direction_minor_batch_hidden"
)

RECURRENT_INITIAL_STATE_CANONICAL_AXIS_ORDER: Final[str] = (
    "layer_direction_node_hidden"
)

RECURRENT_INITIAL_STATE_CALLER_SUPPLIED_SUPPORTED: Final[bool] = False

RECURRENT_INITIAL_STATE_LEARNED_SUPPORTED: Final[bool] = False

RECURRENT_INITIAL_STATE_STATEFUL_SUPPORTED: Final[bool] = False

RECURRENT_INITIAL_STATE_SCIENTIFIC_INTERPRETATION: Final[str] = (
    "deterministic_zero_initialization_not_learned_cold_start_memory"
)


GRUInitialState: TypeAlias = torch.Tensor

LSTMInitialState: TypeAlias = tuple[
    torch.Tensor,
    torch.Tensor,
]

RecurrentInitialState: TypeAlias = (
    GRUInitialState
    | LSTMInitialState
)

CanonicalRecurrentFinalStates: TypeAlias = tuple[
    torch.Tensor,
    torch.Tensor | None,
]


# =============================================================================
# Generic validation
# =============================================================================


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


def _validate_recurrent_config(
    config: RecurrentSequenceEncoderConfig,
) -> None:
    if not isinstance(
        config,
        RecurrentSequenceEncoderConfig,
    ):
        raise TypeError(
            "config must be a RecurrentSequenceEncoderConfig."
        )


def _expected_kernel_type(
    config: RecurrentSequenceEncoderConfig,
) -> type[nn.GRU] | type[nn.LSTM]:
    if config.cell_kind == RecurrentCellKind.GRU:
        return nn.GRU

    return nn.LSTM


def _kernel_kind_name(
    kernel: nn.Module,
) -> str:
    if isinstance(
        kernel,
        nn.GRU,
    ):
        return "gru"

    if isinstance(
        kernel,
        nn.LSTM,
    ):
        return "lstm"

    return type(
        kernel
    ).__name__


def _effective_input_dim(
    config: RecurrentSequenceEncoderConfig,
) -> int:
    if config.input_projection_dim is not None:
        return config.input_projection_dim

    return config.input_dim


def _num_directions(
    config: RecurrentSequenceEncoderConfig,
) -> int:
    return (
        2
        if config.bidirectional
        else 1
    )


def _dropout_matches(
    actual: float,
    expected: float,
) -> bool:
    return math.isclose(
        float(
            actual
        ),
        float(
            expected
        ),
        rel_tol=0.0,
        abs_tol=0.0,
    )


# =============================================================================
# Kernel compatibility
# =============================================================================


def validate_recurrent_kernel_configuration(
    kernel: nn.Module,
    config: RecurrentSequenceEncoderConfig,
) -> None:
    """
    Validate one configured ``nn.GRU`` or ``nn.LSTM`` against Phase 6 config.

    The check is intentionally strict. A kernel that differs in input size,
    hidden size, depth, bias, directionality, dropout, or batch layout is a
    different executable architecture and must not receive states constructed
    for this config.
    """

    if not isinstance(
        kernel,
        nn.Module,
    ):
        raise TypeError(
            "kernel must be a torch.nn.Module."
        )

    _validate_recurrent_config(
        config
    )

    expected_type = _expected_kernel_type(
        config
    )

    if not isinstance(
        kernel,
        expected_type,
    ):
        raise TypeError(
            "Recurrent kernel type does not match config.cell_kind: "
            f"expected {expected_type.__name__}, observed "
            f"{type(kernel).__name__}."
        )

    if int(
        kernel.input_size
    ) != _effective_input_dim(
        config
    ):
        raise ValueError(
            "Recurrent kernel input_size does not match the effective "
            "configured input dimension."
        )

    if int(
        kernel.hidden_size
    ) != config.hidden_dim:
        raise ValueError(
            "Recurrent kernel hidden_size does not match config.hidden_dim."
        )

    if int(
        kernel.num_layers
    ) != config.num_layers:
        raise ValueError(
            "Recurrent kernel num_layers does not match config.num_layers."
        )

    if bool(
        kernel.bias
    ) != config.use_bias:
        raise ValueError(
            "Recurrent kernel bias policy does not match config.use_bias."
        )

    if bool(
        kernel.batch_first
    ) is not True:
        raise ValueError(
            "Phase 6 recurrent kernels must use batch_first=True."
        )

    if bool(
        kernel.bidirectional
    ) != config.bidirectional:
        raise ValueError(
            "Recurrent kernel bidirectionality does not match config."
        )

    if not _dropout_matches(
        kernel.dropout,
        config.dropout,
    ):
        raise ValueError(
            "Recurrent kernel dropout does not match config.dropout."
        )

    if isinstance(
        kernel,
        nn.LSTM,
    ):
        if int(
            kernel.proj_size
        ) != 0:
            raise ValueError(
                "Phase 6 does not support PyTorch LSTM proj_size."
            )


def infer_recurrent_kernel_device_dtype(
    kernel: nn.Module,
) -> tuple[
    torch.device,
    torch.dtype,
]:
    """
    Infer one uniform floating device/dtype from recurrent kernel state.

    All floating parameters and buffers must agree. At least one floating
    parameter is required.
    """

    if not isinstance(
        kernel,
        nn.Module,
    ):
        raise TypeError(
            "kernel must be a torch.nn.Module."
        )

    observed_device: torch.device | None = None
    observed_dtype: torch.dtype | None = None
    floating_parameter_count = 0

    for name, parameter in kernel.named_parameters():
        if parameter.is_meta:
            raise ValueError(
                f"Kernel parameter {name!r} cannot reside on the meta device."
            )

        if not parameter.dtype.is_floating_point:
            continue

        floating_parameter_count += 1

        if observed_device is None:
            observed_device = parameter.device
            observed_dtype = parameter.dtype
            continue

        if parameter.device != observed_device:
            raise ValueError(
                "All floating recurrent parameters must share one device."
            )

        if parameter.dtype != observed_dtype:
            raise ValueError(
                "All floating recurrent parameters must share one dtype."
            )

    if floating_parameter_count == 0:
        raise RuntimeError(
            "The recurrent kernel must own at least one floating parameter."
        )

    for name, buffer in kernel.named_buffers():
        if buffer.is_meta:
            raise ValueError(
                f"Kernel buffer {name!r} cannot reside on the meta device."
            )

        if not buffer.dtype.is_floating_point:
            continue

        if buffer.device != observed_device:
            raise ValueError(
                "Floating recurrent buffers must match parameter device."
            )

        if buffer.dtype != observed_dtype:
            raise ValueError(
                "Floating recurrent buffers must match parameter dtype."
            )

    assert observed_device is not None
    assert observed_dtype is not None

    return (
        observed_device,
        observed_dtype,
    )


def validate_recurrent_kernel_runtime(
    kernel: nn.Module,
    config: RecurrentSequenceEncoderConfig,
) -> tuple[
    torch.device,
    torch.dtype,
]:
    """
    Validate architecture compatibility and return kernel device/dtype.
    """

    validate_recurrent_kernel_configuration(
        kernel,
        config,
    )

    return infer_recurrent_kernel_device_dtype(
        kernel
    )


# =============================================================================
# Layout construction
# =============================================================================


def build_recurrent_state_layout(
    config: RecurrentSequenceEncoderConfig,
) -> RecurrentStateLayout:
    """Build the canonical recurrent state-axis contract for one config."""

    _validate_recurrent_config(
        config
    )

    return RecurrentStateLayout(
        num_layers=config.num_layers,
        num_directions=_num_directions(
            config
        ),
        hidden_dim=config.hidden_dim,
    )


def recurrent_flat_state_shape(
    config: RecurrentSequenceEncoderConfig,
    *,
    batch_size: int,
) -> tuple[
    int,
    int,
    int,
]:
    """Return PyTorch kernel-facing state shape ``[L*D, B, H]``."""

    _validate_recurrent_config(
        config
    )
    _require_nonnegative_int(
        "batch_size",
        batch_size,
    )

    return (
        config.num_layers
        * _num_directions(
            config
        ),
        batch_size,
        config.hidden_dim,
    )


def recurrent_canonical_state_shape(
    config: RecurrentSequenceEncoderConfig,
    *,
    node_count: int,
) -> tuple[
    int,
    int,
    int,
    int,
]:
    """Return research-facing state shape ``[L, D, N, H]``."""

    _validate_recurrent_config(
        config
    )
    _require_nonnegative_int(
        "node_count",
        node_count,
    )

    return (
        config.num_layers,
        _num_directions(
            config
        ),
        node_count,
        config.hidden_dim,
    )


# =============================================================================
# Exact-zero state allocation
# =============================================================================


def _kernel_reference_parameter(
    kernel: nn.Module,
) -> torch.Tensor:
    device, dtype = infer_recurrent_kernel_device_dtype(
        kernel
    )

    for parameter in kernel.parameters():
        if (
            parameter.dtype == dtype
            and parameter.device == device
        ):
            return parameter

    raise RuntimeError(
        "Could not locate the recurrent kernel reference parameter."
    )


def build_zero_flat_hidden_state(
    kernel: nn.Module,
    config: RecurrentSequenceEncoderConfig,
    *,
    batch_size: int,
) -> torch.Tensor:
    """
    Build exact-zero hidden state in PyTorch flat layout ``[L*D, B, H]``.
    """

    _require_nonnegative_int(
        "batch_size",
        batch_size,
    )
    validate_recurrent_kernel_runtime(
        kernel,
        config,
    )
    reference = _kernel_reference_parameter(
        kernel
    )

    return reference.new_zeros(
        recurrent_flat_state_shape(
            config,
            batch_size=batch_size,
        )
    )


def build_zero_flat_cell_state(
    kernel: nn.Module,
    config: RecurrentSequenceEncoderConfig,
    *,
    batch_size: int,
) -> torch.Tensor:
    """
    Build exact-zero LSTM cell state in flat layout ``[L*D, B, H]``.
    """

    _require_nonnegative_int(
        "batch_size",
        batch_size,
    )
    _validate_recurrent_config(
        config
    )

    if config.cell_kind != RecurrentCellKind.LSTM:
        raise ValueError(
            "Cell-state construction requires config.cell_kind='lstm'."
        )

    if not isinstance(
        kernel,
        nn.LSTM,
    ):
        raise TypeError(
            "Cell-state construction requires an nn.LSTM kernel."
        )

    validate_recurrent_kernel_runtime(
        kernel,
        config,
    )
    reference = _kernel_reference_parameter(
        kernel
    )

    return reference.new_zeros(
        recurrent_flat_state_shape(
            config,
            batch_size=batch_size,
        )
    )


def build_zero_gru_initial_state(
    kernel: nn.Module,
    config: RecurrentSequenceEncoderConfig,
    *,
    batch_size: int,
) -> GRUInitialState:
    """Build one exact-zero GRU hidden state."""

    _validate_recurrent_config(
        config
    )

    if config.cell_kind != RecurrentCellKind.GRU:
        raise ValueError(
            "GRU initial-state construction requires cell_kind='gru'."
        )

    if not isinstance(
        kernel,
        nn.GRU,
    ):
        raise TypeError(
            "GRU initial-state construction requires an nn.GRU kernel."
        )

    return build_zero_flat_hidden_state(
        kernel,
        config,
        batch_size=batch_size,
    )


def build_zero_lstm_initial_state(
    kernel: nn.Module,
    config: RecurrentSequenceEncoderConfig,
    *,
    batch_size: int,
) -> LSTMInitialState:
    """Build exact-zero LSTM hidden and cell states."""

    _validate_recurrent_config(
        config
    )

    if config.cell_kind != RecurrentCellKind.LSTM:
        raise ValueError(
            "LSTM initial-state construction requires cell_kind='lstm'."
        )

    if not isinstance(
        kernel,
        nn.LSTM,
    ):
        raise TypeError(
            "LSTM initial-state construction requires an nn.LSTM kernel."
        )

    hidden = build_zero_flat_hidden_state(
        kernel,
        config,
        batch_size=batch_size,
    )
    cell = build_zero_flat_cell_state(
        kernel,
        config,
        batch_size=batch_size,
    )

    return (
        hidden,
        cell,
    )


def build_zero_recurrent_initial_state(
    kernel: nn.Module,
    config: RecurrentSequenceEncoderConfig,
    *,
    batch_size: int,
) -> RecurrentInitialState:
    """Dispatch exact-zero initialization by configured recurrent cell kind."""

    _validate_recurrent_config(
        config
    )

    if config.cell_kind == RecurrentCellKind.GRU:
        return build_zero_gru_initial_state(
            kernel,
            config,
            batch_size=batch_size,
        )

    return build_zero_lstm_initial_state(
        kernel,
        config,
        batch_size=batch_size,
    )


def build_zero_canonical_hidden_state(
    kernel: nn.Module,
    config: RecurrentSequenceEncoderConfig,
    *,
    node_count: int,
) -> torch.Tensor:
    """
    Build exact-zero hidden state in canonical ``[L, D, N, H]`` layout.
    """

    _require_nonnegative_int(
        "node_count",
        node_count,
    )
    flat = build_zero_flat_hidden_state(
        kernel,
        config,
        batch_size=node_count,
    )
    layout = build_recurrent_state_layout(
        config
    )

    canonical = layout.unflatten_state(
        flat,
        name="zero_flat_hidden_state",
    )

    validate_exact_zero_recurrent_state(
        canonical,
        layout=layout,
        node_count=node_count,
        name="zero_canonical_hidden_state",
    )

    return canonical


def build_zero_canonical_cell_state(
    kernel: nn.Module,
    config: RecurrentSequenceEncoderConfig,
    *,
    node_count: int,
) -> torch.Tensor:
    """
    Build exact-zero LSTM cell state in canonical ``[L, D, N, H]`` layout.
    """

    _require_nonnegative_int(
        "node_count",
        node_count,
    )
    flat = build_zero_flat_cell_state(
        kernel,
        config,
        batch_size=node_count,
    )
    layout = build_recurrent_state_layout(
        config
    )

    canonical = layout.unflatten_state(
        flat,
        name="zero_flat_cell_state",
    )

    validate_exact_zero_recurrent_state(
        canonical,
        layout=layout,
        node_count=node_count,
        name="zero_canonical_cell_state",
    )

    return canonical


def build_zero_canonical_final_states(
    kernel: nn.Module,
    config: RecurrentSequenceEncoderConfig,
    *,
    node_count: int,
) -> CanonicalRecurrentFinalStates:
    """
    Build full-node exact-zero final-state placeholders.

    These placeholders are useful before scattering nonempty recurrent states
    and for the all-zero-history short circuit.
    """

    hidden = build_zero_canonical_hidden_state(
        kernel,
        config,
        node_count=node_count,
    )

    if config.cell_kind == RecurrentCellKind.GRU:
        return (
            hidden,
            None,
        )

    cell = build_zero_canonical_cell_state(
        kernel,
        config,
        node_count=node_count,
    )

    return (
        hidden,
        cell,
    )


# =============================================================================
# State-layout conversion and validation
# =============================================================================


def flatten_recurrent_state(
    state: torch.Tensor,
    *,
    layout: RecurrentStateLayout,
    name: str = "canonical_recurrent_state",
) -> torch.Tensor:
    """Convert canonical ``[L,D,N,H]`` to flat ``[L*D,N,H]``."""

    if not isinstance(
        layout,
        RecurrentStateLayout,
    ):
        raise TypeError(
            "layout must be a RecurrentStateLayout."
        )

    return layout.flatten_state(
        state,
        name=name,
    )


def unflatten_recurrent_state(
    state: torch.Tensor,
    *,
    layout: RecurrentStateLayout,
    name: str = "flat_recurrent_state",
) -> torch.Tensor:
    """Convert flat ``[L*D,N,H]`` to canonical ``[L,D,N,H]``."""

    if not isinstance(
        layout,
        RecurrentStateLayout,
    ):
        raise TypeError(
            "layout must be a RecurrentStateLayout."
        )

    return layout.unflatten_state(
        state,
        name=name,
    )


def validate_exact_zero_recurrent_state(
    state: torch.Tensor,
    *,
    layout: RecurrentStateLayout,
    node_count: int,
    name: str = "recurrent_state",
) -> None:
    """Validate canonical shape, finiteness, and exact-zero values."""

    if not isinstance(
        layout,
        RecurrentStateLayout,
    ):
        raise TypeError(
            "layout must be a RecurrentStateLayout."
        )

    _require_nonnegative_int(
        "node_count",
        node_count,
    )

    layout.validate_canonical_state(
        state,
        name=name,
        node_count=node_count,
    )

    if state.numel() == 0:
        return

    if bool(
        torch.any(
            state != 0
        ).item()
    ):
        raise ValueError(
            f"{name} must contain exact zeros."
        )


def validate_flat_initial_state(
    state: torch.Tensor,
    *,
    kernel: nn.Module,
    config: RecurrentSequenceEncoderConfig,
    batch_size: int,
    name: str = "initial_state",
) -> None:
    """
    Validate one kernel-facing flat state against config and kernel runtime.
    """

    _require_nonnegative_int(
        "batch_size",
        batch_size,
    )
    device, dtype = validate_recurrent_kernel_runtime(
        kernel,
        config,
    )
    layout = build_recurrent_state_layout(
        config
    )

    layout.validate_flat_state(
        state,
        name=name,
        node_count=batch_size,
    )

    if state.device != device:
        raise ValueError(
            f"{name} device must match recurrent kernel device."
        )

    if state.dtype != dtype:
        raise ValueError(
            f"{name} dtype must match recurrent kernel dtype."
        )


def validate_zero_recurrent_initial_state(
    initial_state: RecurrentInitialState,
    *,
    kernel: nn.Module,
    config: RecurrentSequenceEncoderConfig,
    batch_size: int,
) -> None:
    """
    Validate the complete Phase 6 zero-initial-state structure.
    """

    _validate_recurrent_config(
        config
    )

    if config.cell_kind == RecurrentCellKind.GRU:
        if not isinstance(
            initial_state,
            torch.Tensor,
        ):
            raise TypeError(
                "GRU initial_state must be one hidden-state tensor."
            )

        validate_flat_initial_state(
            initial_state,
            kernel=kernel,
            config=config,
            batch_size=batch_size,
            name="gru_initial_hidden_state",
        )

        if initial_state.numel() > 0 and bool(
            torch.any(
                initial_state != 0
            ).item()
        ):
            raise ValueError(
                "GRU initial hidden state must be exactly zero."
            )

        return

    if (
        not isinstance(
            initial_state,
            tuple,
        )
        or len(
            initial_state
        )
        != 2
    ):
        raise TypeError(
            "LSTM initial_state must be a (hidden, cell) tuple."
        )

    hidden, cell = initial_state

    validate_flat_initial_state(
        hidden,
        kernel=kernel,
        config=config,
        batch_size=batch_size,
        name="lstm_initial_hidden_state",
    )
    validate_flat_initial_state(
        cell,
        kernel=kernel,
        config=config,
        batch_size=batch_size,
        name="lstm_initial_cell_state",
    )

    if hidden.numel() > 0 and bool(
        torch.any(
            hidden != 0
        ).item()
    ):
        raise ValueError(
            "LSTM initial hidden state must be exactly zero."
        )

    if cell.numel() > 0 and bool(
        torch.any(
            cell != 0
        ).item()
    ):
        raise ValueError(
            "LSTM initial cell state must be exactly zero."
        )


# =============================================================================
# Compact aliases
# =============================================================================


build_state_layout = build_recurrent_state_layout
build_zero_initial_state = build_zero_recurrent_initial_state
build_zero_final_states = build_zero_canonical_final_states
flatten_state = flatten_recurrent_state
unflatten_state = unflatten_recurrent_state


# =============================================================================
# Public API
# =============================================================================


__all__ = (
    # Component identity and policy.
    "RECURRENT_INITIAL_STATE_IMPLEMENTATION_VERSION",
    "RECURRENT_INITIAL_STATE_COMPONENT_NAME",
    "RECURRENT_INITIAL_STATE_COMPONENT_KIND",
    "RECURRENT_INITIAL_STATE_OPERATION_NAME",
    "RECURRENT_INITIAL_STATE_POLICY",
    "RECURRENT_INITIAL_STATE_FLAT_AXIS_ORDER",
    "RECURRENT_INITIAL_STATE_CANONICAL_AXIS_ORDER",
    "RECURRENT_INITIAL_STATE_CALLER_SUPPLIED_SUPPORTED",
    "RECURRENT_INITIAL_STATE_LEARNED_SUPPORTED",
    "RECURRENT_INITIAL_STATE_STATEFUL_SUPPORTED",
    "RECURRENT_INITIAL_STATE_SCIENTIFIC_INTERPRETATION",

    # Type aliases.
    "GRUInitialState",
    "LSTMInitialState",
    "RecurrentInitialState",
    "CanonicalRecurrentFinalStates",

    # Kernel compatibility.
    "validate_recurrent_kernel_configuration",
    "infer_recurrent_kernel_device_dtype",
    "validate_recurrent_kernel_runtime",

    # Layout and shapes.
    "build_recurrent_state_layout",
    "recurrent_flat_state_shape",
    "recurrent_canonical_state_shape",

    # Exact-zero state construction.
    "build_zero_flat_hidden_state",
    "build_zero_flat_cell_state",
    "build_zero_gru_initial_state",
    "build_zero_lstm_initial_state",
    "build_zero_recurrent_initial_state",
    "build_zero_canonical_hidden_state",
    "build_zero_canonical_cell_state",
    "build_zero_canonical_final_states",

    # Conversion and validation.
    "flatten_recurrent_state",
    "unflatten_recurrent_state",
    "validate_exact_zero_recurrent_state",
    "validate_flat_initial_state",
    "validate_zero_recurrent_initial_state",

    # Compact aliases.
    "build_state_layout",
    "build_zero_initial_state",
    "build_zero_final_states",
    "flatten_state",
    "unflatten_state",
)
