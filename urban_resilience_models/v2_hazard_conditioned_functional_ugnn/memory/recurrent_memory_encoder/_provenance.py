"""
Private provenance utilities for GRU/LSTM temporal-memory encoders.

Target path:
    urban_resilience_models/
        v2_hazard_conditioned_functional_ugnn/
            memory/
                recurrent_memory_encoder/
                    _provenance.py

This module centralizes mechanical provenance construction for Phase 6
recurrent sequence encoders. It does not define recurrent runtime schemas and
does not implement sequence canonicalization, packing, sorting, restoration,
or recurrent execution.

The central scientific requirement is a strict separation between:

architecture identity
    The mathematical recurrent model: GRU versus LSTM, feature dimensions,
    recurrent depth, directionality, bias policy, input projection,
    LayerNorm, dropout, and zero-initialization policy.

execution identity
    How that same model was executed for one batch: packed versus per-node
    reference execution, sorting behavior, original padding layout,
    train/evaluation mode, and all-zero-history short-circuit behavior.

Consequently, these configuration fields are deliberately excluded from the
architecture fingerprint:

- ``pack_sequences``;
- ``enforce_sorted_lengths``.

Two otherwise identical recurrent encoders that differ only in those fields
must have the same architecture fingerprint. Their execution lineage may
differ.

Parameter snapshots are optional and explicit. They are never generated during
an ordinary forward call. When a snapshot is supplied, callers should validate
it before any all-zero-history early return so stale or incompatible model
state cannot be silently accepted.
"""

from __future__ import annotations

from dataclasses import asdict
from dataclasses import is_dataclass
from enum import Enum
from hashlib import sha256
import json
import math
from pathlib import Path
from typing import Any, Final, Mapping, Sequence

import torch
from torch import nn

from ..config import (
    RecurrentCellKind,
    RecurrentSequenceEncoderConfig,
)
from ..schemas.history_inputs import (
    HistoricalSequenceInputs,
)
from ..schemas.provenance import (
    MemoryArchitectureProvenance,
    MemoryComputationProvenance,
    MemoryExecutionLineage,
    MemoryParameterSnapshotProvenance,
)


# =============================================================================
# Private utility identity and frozen Phase 6 policies
# =============================================================================


RECURRENT_PROVENANCE_IMPLEMENTATION_VERSION: Final[str] = "0.1"

RECURRENT_PROVENANCE_SCOPE: Final[str] = (
    "recurrent_architecture_and_execution_provenance_construction"
)

RECURRENT_ARCHITECTURE_PAYLOAD_VERSION: Final[str] = "0.1"

RECURRENT_EXECUTION_PAYLOAD_VERSION: Final[str] = "0.1"

RECURRENT_CANONICALIZATION_VERSION: Final[str] = (
    "valid_timesteps_chronological_to_right_padded_v1"
)

RECURRENT_STABLE_SORT_POLICY_VERSION: Final[str] = (
    "stable_descending_length_original_nonempty_order_ties_v1"
)

RECURRENT_INITIAL_STATE_POLICY: Final[str] = "exact_zero_v1"

RECURRENT_INPUT_PROJECTION_BIAS_POLICY: Final[str] = (
    "use_bias_controls_projection_and_recurrent_bias"
)

RECURRENT_LAYER_NORM_AFFINE_POLICY: Final[str] = (
    "elementwise_affine_true"
)

RECURRENT_DIRECTION_FEATURE_ORDER: Final[str] = (
    "forward_then_backward"
)

RECURRENT_STATE_LAYOUT_POLICY: Final[str] = (
    "layer_direction_node_hidden"
)

RECURRENT_TEMPORAL_INTERACTION: Final[bool] = True

RECURRENT_FEATURE_OBSERVATION_MASK_CONSUMED: Final[bool] = False

RECURRENT_TEMPORAL_COORDINATES_CONSUMED: Final[bool] = False

RECURRENT_HAZARD_CONDITIONED: Final[bool] = False

RECURRENT_PARAMETER_SNAPSHOT_FORMAT_VERSION: Final[str] = (
    "recurrent_named_parameters_v1"
)

RECURRENT_EXECUTION_PATH_PACKED: Final[str] = "packed"

RECURRENT_EXECUTION_PATH_REFERENCE: Final[str] = "reference"

RECURRENT_EXECUTION_PATHS: Final[tuple[str, ...]] = (
    RECURRENT_EXECUTION_PATH_PACKED,
    RECURRENT_EXECUTION_PATH_REFERENCE,
)


# =============================================================================
# Canonical component identity
# =============================================================================


def canonical_recurrent_component_name(
    cell_kind: RecurrentCellKind | str,
) -> str:
    """Return the architecture-level component name for one recurrent cell."""

    kind = _normalize_cell_kind(
        cell_kind
    )

    if kind == RecurrentCellKind.GRU:
        return "gru_sequence_encoder"

    return "lstm_sequence_encoder"


def canonical_recurrent_component_kind(
    cell_kind: RecurrentCellKind | str,
) -> str:
    """Return the canonical architecture kind: ``gru`` or ``lstm``."""

    return _normalize_cell_kind(
        cell_kind
    ).value


def canonical_recurrent_operation_name(
    cell_kind: RecurrentCellKind | str,
) -> str:
    """Return the canonical sequence-encoding operation name."""

    kind = _normalize_cell_kind(
        cell_kind
    )

    if kind == RecurrentCellKind.GRU:
        return "encode_history_with_gru"

    return "encode_history_with_lstm"


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


def _require_optional_nonempty_string(
    name: str,
    value: str | None,
) -> None:
    if value is None:
        return

    _require_nonempty_string(
        name,
        value,
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


def _require_optional_nonnegative_int(
    name: str,
    value: int | None,
) -> None:
    if value is None:
        return

    _require_nonnegative_int(
        name,
        value,
    )


def _normalize_cell_kind(
    value: RecurrentCellKind | str,
) -> RecurrentCellKind:
    if isinstance(
        value,
        RecurrentCellKind,
    ):
        return value

    if not isinstance(
        value,
        str,
    ):
        raise TypeError(
            "cell_kind must be a RecurrentCellKind or string."
        )

    try:
        return RecurrentCellKind(
            value
        )
    except ValueError as error:
        raise ValueError(
            f"Unsupported recurrent cell kind {value!r}."
        ) from error


def _normalize_execution_path(
    value: str | Enum,
) -> str:
    if isinstance(
        value,
        Enum,
    ):
        value = value.value

    if not isinstance(
        value,
        str,
    ):
        raise TypeError(
            "execution_path must be a string or string-valued Enum."
        )

    if value not in RECURRENT_EXECUTION_PATHS:
        raise ValueError(
            "execution_path must be one of "
            f"{RECURRENT_EXECUTION_PATHS}; observed {value!r}."
        )

    return value


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


def _validate_counts(
    *,
    nonempty_node_count: int,
    zero_history_count: int,
    source_node_count: int,
) -> None:
    _require_nonnegative_int(
        "nonempty_node_count",
        nonempty_node_count,
    )
    _require_nonnegative_int(
        "zero_history_count",
        zero_history_count,
    )
    _require_nonnegative_int(
        "source_node_count",
        source_node_count,
    )

    if (
        nonempty_node_count
        + zero_history_count
        != source_node_count
    ):
        raise ValueError(
            "nonempty_node_count + zero_history_count must equal "
            "the source node count."
        )


# =============================================================================
# Canonical JSON and deterministic fingerprints
# =============================================================================


def _to_plain_json_value(
    value: Any,
) -> Any:
    """
    Convert supported metadata to deterministic plain JSON data.

    Tensor values are deliberately unsupported. Runtime tensor identities belong
    in recurrent schemas, while current parameter values belong in an explicit
    parameter snapshot.
    """

    if value is None:
        return None

    if isinstance(
        value,
        (
            str,
            bool,
            int,
        ),
    ):
        return value

    if isinstance(
        value,
        float,
    ):
        if not math.isfinite(
            value
        ):
            raise ValueError(
                "Provenance metadata cannot contain NaN or infinity."
            )
        return value

    if isinstance(
        value,
        Enum,
    ):
        return _to_plain_json_value(
            value.value
        )

    if isinstance(
        value,
        Path,
    ):
        return str(
            value
        )

    if isinstance(
        value,
        torch.dtype,
    ):
        return str(
            value
        )

    if isinstance(
        value,
        torch.device,
    ):
        return str(
            value
        )

    if is_dataclass(
        value
    ) and not isinstance(
        value,
        type,
    ):
        return _to_plain_json_value(
            asdict(
                value
            )
        )

    if isinstance(
        value,
        Mapping,
    ):
        converted: dict[str, Any] = {}

        for key, item in value.items():
            if not isinstance(
                key,
                str,
            ):
                raise TypeError(
                    "Provenance metadata mapping keys must be strings."
                )

            _require_nonempty_string(
                "provenance metadata key",
                key,
            )
            converted[
                key
            ] = _to_plain_json_value(
                item
            )

        return converted

    if isinstance(
        value,
        (
            tuple,
            list,
        ),
    ):
        return [
            _to_plain_json_value(
                item
            )
            for item in value
        ]

    raise TypeError(
        "Unsupported provenance metadata value of type "
        f"{type(value).__name__!r}."
    )


def _canonical_json(
    payload: Mapping[str, Any],
) -> str:
    if not isinstance(
        payload,
        Mapping,
    ):
        raise TypeError(
            "Canonical provenance payload must be a mapping."
        )

    plain = _to_plain_json_value(
        payload
    )

    if not isinstance(
        plain,
        dict,
    ):
        raise TypeError(
            "Canonical provenance payload must convert to a mapping."
        )

    return json.dumps(
        plain,
        sort_keys=True,
        separators=(
            ",",
            ":",
        ),
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
# Explicit architecture allowlist
# =============================================================================


def recurrent_architecture_metadata(
    config: RecurrentSequenceEncoderConfig,
) -> dict[str, Any]:
    """
    Return the frozen Phase 6 architecture allowlist.

    This function must not be replaced with ``asdict(config)`` or
    ``config.config_hash()`` because the configuration also contains execution
    fields that must not alter architecture identity.
    """

    _validate_recurrent_config(
        config
    )

    num_directions = (
        2
        if config.bidirectional
        else 1
    )
    projection_enabled = (
        config.input_projection_dim
        is not None
    )
    effective_input_dim = (
        config.input_projection_dim
        if projection_enabled
        else config.input_dim
    )

    return {
        "architecture_payload_version": (
            RECURRENT_ARCHITECTURE_PAYLOAD_VERSION
        ),
        "cell_kind": config.cell_kind.value,
        "schema_encoder_kind": (
            config.schema_encoder_kind.value
        ),
        "input_dim": config.input_dim,
        "effective_recurrent_input_dim": (
            effective_input_dim
        ),
        "hidden_dim": config.hidden_dim,
        "output_dim": config.output_dim,
        "num_layers": config.num_layers,
        "num_directions": num_directions,
        "dropout": config.dropout,
        "bidirectional": config.bidirectional,
        "use_bias": config.use_bias,
        "input_projection_enabled": (
            projection_enabled
        ),
        "input_projection_dim": (
            config.input_projection_dim
        ),
        "input_projection_bias_enabled": (
            config.use_bias
            if projection_enabled
            else False
        ),
        "input_projection_bias_policy": (
            RECURRENT_INPUT_PROJECTION_BIAS_POLICY
        ),
        "layer_normalization": (
            config.layer_normalization
        ),
        "layer_norm_elementwise_affine": True,
        "layer_norm_affine_policy": (
            RECURRENT_LAYER_NORM_AFFINE_POLICY
        ),
        "initial_state_policy": (
            RECURRENT_INITIAL_STATE_POLICY
        ),
        "direction_feature_order": (
            RECURRENT_DIRECTION_FEATURE_ORDER
        ),
        "state_layout_policy": (
            RECURRENT_STATE_LAYOUT_POLICY
        ),
        "temporal_interaction": (
            RECURRENT_TEMPORAL_INTERACTION
        ),
        "feature_observation_mask_consumed": (
            RECURRENT_FEATURE_OBSERVATION_MASK_CONSUMED
        ),
        "temporal_coordinates_consumed": (
            RECURRENT_TEMPORAL_COORDINATES_CONSUMED
        ),
        "hazard_conditioned": (
            RECURRENT_HAZARD_CONDITIONED
        ),
    }


def recurrent_execution_configuration_metadata(
    config: RecurrentSequenceEncoderConfig,
) -> dict[str, Any]:
    """
    Return configuration fields that select execution strategy.

    This metadata is intentionally separate from architecture identity.
    """

    _validate_recurrent_config(
        config
    )

    return {
        "execution_payload_version": (
            RECURRENT_EXECUTION_PAYLOAD_VERSION
        ),
        "pack_sequences": (
            config.pack_sequences
        ),
        "enforce_sorted_lengths": (
            config.enforce_sorted_lengths
        ),
        "selected_execution_path": (
            RECURRENT_EXECUTION_PATH_PACKED
            if config.pack_sequences
            else RECURRENT_EXECUTION_PATH_REFERENCE
        ),
    }


def recurrent_architecture_configuration_fingerprint(
    config: RecurrentSequenceEncoderConfig,
) -> str:
    """
    Fingerprint only mathematical architecture configuration.

    The result is invariant to ``pack_sequences`` and
    ``enforce_sorted_lengths``.
    """

    return _fingerprint(
        {
            "scope": (
                "recurrent_architecture_configuration"
            ),
            "architecture": (
                recurrent_architecture_metadata(
                    config
                )
            ),
        }
    )


def recurrent_execution_configuration_fingerprint(
    config: RecurrentSequenceEncoderConfig,
) -> str:
    """
    Fingerprint architecture plus selected execution configuration.

    This fingerprint is suitable for execution-lineage metadata, not for
    architecture identity.
    """

    return _fingerprint(
        {
            "scope": (
                "recurrent_execution_configuration"
            ),
            "architecture_configuration_fingerprint": (
                recurrent_architecture_configuration_fingerprint(
                    config
                )
            ),
            "execution": (
                recurrent_execution_configuration_metadata(
                    config
                )
            ),
        }
    )


def _merge_metadata(
    *,
    base: Mapping[str, Any],
    extra: Mapping[str, Any] | None,
    name: str,
) -> dict[str, Any]:
    """
    Merge caller metadata without allowing frozen fields to be overridden.
    """

    result = dict(
        base
    )

    if extra is None:
        return result

    if not isinstance(
        extra,
        Mapping,
    ):
        raise TypeError(
            f"{name} must be a mapping or None."
        )

    collisions = sorted(
        set(
            result
        )
        & set(
            extra
        )
    )

    if collisions:
        raise ValueError(
            f"{name} cannot override frozen keys: {collisions}."
        )

    for key, value in extra.items():
        if not isinstance(
            key,
            str,
        ):
            raise TypeError(
                f"{name} keys must be strings."
            )
        _require_nonempty_string(
            f"{name} key",
            key,
        )
        result[
            key
        ] = value

    # Validate JSON compatibility now rather than after schema construction.
    _to_plain_json_value(
        result
    )

    return result


def recurrent_architecture_fingerprint(
    config: RecurrentSequenceEncoderConfig,
    *,
    extra_architecture_metadata: (
        Mapping[str, Any]
        | None
    ) = None,
    implementation_version: str = (
        RECURRENT_PROVENANCE_IMPLEMENTATION_VERSION
    ),
) -> str:
    """
    Return a dispatcher-independent recurrent architecture fingerprint.
    """

    _validate_recurrent_config(
        config
    )
    _require_nonempty_string(
        "implementation_version",
        implementation_version,
    )

    metadata = _merge_metadata(
        base=recurrent_architecture_metadata(
            config
        ),
        extra=extra_architecture_metadata,
        name="extra_architecture_metadata",
    )

    return _fingerprint(
        {
            "provenance_scope": (
                RECURRENT_PROVENANCE_SCOPE
            ),
            "implementation_version": (
                implementation_version
            ),
            "component_kind": (
                canonical_recurrent_component_kind(
                    config.cell_kind
                )
            ),
            "architecture_configuration_fingerprint": (
                recurrent_architecture_configuration_fingerprint(
                    config
                )
            ),
            "architecture_metadata": (
                metadata
            ),
        }
    )


# =============================================================================
# Parameter counting and explicit snapshot identity
# =============================================================================


def count_module_parameters(
    module: nn.Module,
) -> tuple[int, int]:
    """Return ``(parameter_count, trainable_parameter_count)``."""

    if not isinstance(
        module,
        nn.Module,
    ):
        raise TypeError(
            "module must be a torch.nn.Module."
        )

    parameter_count = 0
    trainable_count = 0

    for parameter in module.parameters():
        count = int(
            parameter.numel()
        )
        parameter_count += count

        if parameter.requires_grad:
            trainable_count += count

    return (
        parameter_count,
        trainable_count,
    )


def module_parameter_snapshot_fingerprint(
    module: nn.Module,
) -> str:
    """
    Fingerprint all named parameters independently of device placement.

    The snapshot includes parameters owned by the input adapter and recurrent
    kernel because both are children of the encoder module. Persistent buffers
    are deliberately excluded; this is parameter identity rather than a full
    state-dict or optimizer snapshot.
    """

    if not isinstance(
        module,
        nn.Module,
    ):
        raise TypeError(
            "module must be a torch.nn.Module."
        )

    digest = sha256()
    digest.update(
        RECURRENT_PARAMETER_SNAPSHOT_FORMAT_VERSION.encode(
            "utf-8"
        )
    )

    observed_names: list[str] = []

    for name, parameter in module.named_parameters():
        _require_nonempty_string(
            "parameter name",
            name,
        )
        observed_names.append(
            name
        )

        if parameter.is_meta:
            raise ValueError(
                "Cannot fingerprint parameters on the meta device."
            )

        if parameter.layout != torch.strided:
            raise ValueError(
                "Only strided parameters are supported for snapshot "
                f"fingerprinting; observed {parameter.layout} for {name!r}."
            )

        tensor = (
            parameter
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
            b"1"
            if parameter.requires_grad
            else b"0"
        )
        digest.update(
            tensor.view(
                torch.uint8
            )
            .numpy()
            .tobytes()
        )

    if len(
        observed_names
    ) != len(
        set(
            observed_names
        )
    ):
        raise ValueError(
            "Module parameter names must be unique."
        )

    return digest.hexdigest()


def build_parameter_snapshot_provenance(
    module: nn.Module,
    *,
    checkpoint_id: str | None = None,
    checkpoint_fingerprint: str | None = None,
    training_step: int | None = None,
) -> MemoryParameterSnapshotProvenance:
    """
    Explicitly construct one recurrent parameter-snapshot provenance object.

    This helper must not be called automatically by ``forward`` or
    ``encode_with_state``.
    """

    _require_optional_nonempty_string(
        "checkpoint_id",
        checkpoint_id,
    )
    _require_optional_nonempty_string(
        "checkpoint_fingerprint",
        checkpoint_fingerprint,
    )
    _require_optional_nonnegative_int(
        "training_step",
        training_step,
    )

    parameter_count, trainable_count = (
        count_module_parameters(
            module
        )
    )

    return MemoryParameterSnapshotProvenance(
        parameter_snapshot_fingerprint=(
            module_parameter_snapshot_fingerprint(
                module
            )
        ),
        checkpoint_id=checkpoint_id,
        checkpoint_fingerprint=(
            checkpoint_fingerprint
        ),
        training_step=training_step,
        parameter_count=parameter_count,
        trainable_parameter_count=(
            trainable_count
        ),
    )


def validate_parameter_snapshot_provenance(
    module: nn.Module,
    parameter_snapshot: (
        MemoryParameterSnapshotProvenance
        | None
    ),
) -> None:
    """
    Validate one optional snapshot against the module's current parameters.

    Call this before an all-zero-history short circuit.
    """

    if parameter_snapshot is None:
        return

    if not isinstance(
        parameter_snapshot,
        MemoryParameterSnapshotProvenance,
    ):
        raise TypeError(
            "parameter_snapshot must be a "
            "MemoryParameterSnapshotProvenance or None."
        )

    parameter_count, trainable_count = (
        count_module_parameters(
            module
        )
    )

    if (
        parameter_snapshot.parameter_count
        is not None
        and parameter_snapshot.parameter_count
        != parameter_count
    ):
        raise ValueError(
            "parameter_snapshot.parameter_count does not match the "
            "current recurrent module."
        )

    if (
        parameter_snapshot.trainable_parameter_count
        is not None
        and parameter_snapshot.trainable_parameter_count
        != trainable_count
    ):
        raise ValueError(
            "parameter_snapshot.trainable_parameter_count does not "
            "match the current recurrent module."
        )

    current_fingerprint = (
        module_parameter_snapshot_fingerprint(
            module
        )
    )

    if (
        parameter_snapshot
        .parameter_snapshot_fingerprint
        != current_fingerprint
    ):
        raise ValueError(
            "parameter_snapshot is stale or belongs to a different "
            "recurrent module."
        )


# =============================================================================
# Architecture provenance
# =============================================================================


def build_recurrent_architecture_provenance(
    config: RecurrentSequenceEncoderConfig,
    *,
    extra_architecture_metadata: (
        Mapping[str, Any]
        | None
    ) = None,
    implementation_version: str = (
        RECURRENT_PROVENANCE_IMPLEMENTATION_VERSION
    ),
) -> MemoryArchitectureProvenance:
    """
    Build architecture provenance from the explicit Phase 6 allowlist.

    The component name is canonicalized from ``cell_kind`` so direct and
    dispatcher-selected encoders share the same architecture identity.
    """

    _validate_recurrent_config(
        config
    )
    _require_nonempty_string(
        "implementation_version",
        implementation_version,
    )

    metadata = _merge_metadata(
        base=recurrent_architecture_metadata(
            config
        ),
        extra=extra_architecture_metadata,
        name="extra_architecture_metadata",
    )

    architecture_configuration_fingerprint = (
        recurrent_architecture_configuration_fingerprint(
            config
        )
    )

    return MemoryArchitectureProvenance(
        component_name=(
            canonical_recurrent_component_name(
                config.cell_kind
            )
        ),
        component_kind=(
            canonical_recurrent_component_kind(
                config.cell_kind
            )
        ),
        architecture_fingerprint=(
            recurrent_architecture_fingerprint(
                config,
                extra_architecture_metadata=(
                    extra_architecture_metadata
                ),
                implementation_version=(
                    implementation_version
                ),
            )
        ),
        configuration_fingerprint=(
            architecture_configuration_fingerprint
        ),
        implementation_version=(
            implementation_version
        ),
        architecture_metadata={
            **metadata,
            "provenance_scope": (
                RECURRENT_PROVENANCE_SCOPE
            ),
        },
    )


# =============================================================================
# Execution lineage and computation provenance
# =============================================================================


def recurrent_execution_lineage_metadata(
    config: RecurrentSequenceEncoderConfig,
    *,
    execution_path: str | Enum,
    sort_was_applied: bool,
    source_padding_direction: str | Enum,
    module_training: bool,
    nonempty_node_count: int,
    zero_history_count: int,
    source_node_count: int,
    adapter_executed: bool,
    recurrent_kernel_executed: bool,
    all_zero_history_short_circuit: bool,
    extra_lineage_metadata: (
        Mapping[str, Any]
        | None
    ) = None,
) -> dict[str, Any]:
    """Build the frozen execution-lineage payload for one recurrent run."""

    _validate_recurrent_config(
        config
    )
    path = _normalize_execution_path(
        execution_path
    )
    _require_boolean(
        "sort_was_applied",
        sort_was_applied,
    )
    _require_boolean(
        "module_training",
        module_training,
    )
    _require_boolean(
        "adapter_executed",
        adapter_executed,
    )
    _require_boolean(
        "recurrent_kernel_executed",
        recurrent_kernel_executed,
    )
    _require_boolean(
        "all_zero_history_short_circuit",
        all_zero_history_short_circuit,
    )
    _validate_counts(
        nonempty_node_count=nonempty_node_count,
        zero_history_count=zero_history_count,
        source_node_count=source_node_count,
    )

    if isinstance(
        source_padding_direction,
        Enum,
    ):
        source_padding_direction = (
            source_padding_direction.value
        )

    _require_nonempty_string(
        "source_padding_direction",
        source_padding_direction,
    )

    expected_path = (
        RECURRENT_EXECUTION_PATH_PACKED
        if config.pack_sequences
        else RECURRENT_EXECUTION_PATH_REFERENCE
    )

    if path != expected_path:
        raise ValueError(
            "execution_path does not match config.pack_sequences: "
            f"expected {expected_path!r}, observed {path!r}."
        )

    if (
        path == RECURRENT_EXECUTION_PATH_REFERENCE
        and sort_was_applied
    ):
        raise ValueError(
            "Reference recurrent execution cannot report sorting."
        )

    if all_zero_history_short_circuit:
        if nonempty_node_count != 0:
            raise ValueError(
                "all_zero_history_short_circuit requires "
                "nonempty_node_count=0."
            )
        if adapter_executed:
            raise ValueError(
                "The all-zero-history short circuit must not execute "
                "the input adapter."
            )
        if recurrent_kernel_executed:
            raise ValueError(
                "The all-zero-history short circuit must not execute "
                "the recurrent kernel."
            )
        if sort_was_applied:
            raise ValueError(
                "The all-zero-history short circuit cannot report sorting."
            )
    else:
        if nonempty_node_count == 0:
            raise ValueError(
                "A run with no nonempty nodes must report the "
                "all-zero-history short circuit."
            )
        if not adapter_executed:
            raise ValueError(
                "A nonempty recurrent run must execute the input adapter."
            )
        if not recurrent_kernel_executed:
            raise ValueError(
                "A nonempty recurrent run must execute the recurrent kernel."
            )

    dropout_active = (
        module_training
        and config.num_layers > 1
        and config.dropout > 0.0
    )

    base = {
        "execution_payload_version": (
            RECURRENT_EXECUTION_PAYLOAD_VERSION
        ),
        "execution_path": path,
        "selected_pack_sequences": (
            config.pack_sequences
        ),
        "enforce_sorted_lengths": (
            config.enforce_sorted_lengths
        ),
        "sort_was_applied": (
            sort_was_applied
        ),
        "stable_sort_policy_version": (
            RECURRENT_STABLE_SORT_POLICY_VERSION
        ),
        "canonicalization_version": (
            RECURRENT_CANONICALIZATION_VERSION
        ),
        "canonical_padding_direction": "right",
        "original_padding_direction": (
            source_padding_direction
        ),
        "module_training": (
            module_training
        ),
        "dropout_probability": (
            config.dropout
        ),
        "dropout_active": (
            dropout_active
        ),
        "nonempty_node_count": (
            nonempty_node_count
        ),
        "zero_history_count": (
            zero_history_count
        ),
        "source_node_count": (
            source_node_count
        ),
        "adapter_executed": (
            adapter_executed
        ),
        "recurrent_kernel_executed": (
            recurrent_kernel_executed
        ),
        "all_zero_history_short_circuit": (
            all_zero_history_short_circuit
        ),
        "feature_observation_mask_consumed": (
            RECURRENT_FEATURE_OBSERVATION_MASK_CONSUMED
        ),
        "temporal_coordinates_consumed": (
            RECURRENT_TEMPORAL_COORDINATES_CONSUMED
        ),
        "hazard_conditioned": (
            RECURRENT_HAZARD_CONDITIONED
        ),
        "execution_configuration_fingerprint": (
            recurrent_execution_configuration_fingerprint(
                config
            )
        ),
    }

    return _merge_metadata(
        base=base,
        extra=extra_lineage_metadata,
        name="extra_lineage_metadata",
    )


def build_recurrent_sequence_computation_provenance(
    *,
    source_history: HistoricalSequenceInputs,
    config: RecurrentSequenceEncoderConfig,
    execution_path: str | Enum,
    sort_was_applied: bool,
    module_training: bool,
    nonempty_node_count: int,
    zero_history_count: int,
    adapter_executed: bool,
    recurrent_kernel_executed: bool,
    all_zero_history_short_circuit: bool,
    parameter_snapshot: (
        MemoryParameterSnapshotProvenance
        | None
    ) = None,
    extra_architecture_metadata: (
        Mapping[str, Any]
        | None
    ) = None,
    extra_lineage_metadata: (
        Mapping[str, Any]
        | None
    ) = None,
    operation_name: str | None = None,
    implementation_version: str = (
        RECURRENT_PROVENANCE_IMPLEMENTATION_VERSION
    ),
) -> MemoryComputationProvenance:
    """
    Build complete provenance for one GRU/LSTM sequence-encoding run.

    Snapshot freshness is expected to be validated against the module before
    this function is called. This function verifies only the snapshot schema
    and links its fingerprint into execution lineage.
    """

    if not isinstance(
        source_history,
        HistoricalSequenceInputs,
    ):
        raise TypeError(
            "source_history must be a HistoricalSequenceInputs."
        )

    _validate_recurrent_config(
        config
    )

    if (
        parameter_snapshot is not None
        and not isinstance(
            parameter_snapshot,
            MemoryParameterSnapshotProvenance,
        )
    ):
        raise TypeError(
            "parameter_snapshot must be a "
            "MemoryParameterSnapshotProvenance or None."
        )

    if operation_name is None:
        operation_name = (
            canonical_recurrent_operation_name(
                config.cell_kind
            )
        )

    _require_nonempty_string(
        "operation_name",
        operation_name,
    )

    architecture = (
        build_recurrent_architecture_provenance(
            config,
            extra_architecture_metadata=(
                extra_architecture_metadata
            ),
            implementation_version=(
                implementation_version
            ),
        )
    )

    lineage_metadata = (
        recurrent_execution_lineage_metadata(
            config,
            execution_path=execution_path,
            sort_was_applied=(
                sort_was_applied
            ),
            source_padding_direction=(
                source_history
                .padding_direction
            ),
            module_training=(
                module_training
            ),
            nonempty_node_count=(
                nonempty_node_count
            ),
            zero_history_count=(
                zero_history_count
            ),
            source_node_count=(
                source_history.node_count
            ),
            adapter_executed=(
                adapter_executed
            ),
            recurrent_kernel_executed=(
                recurrent_kernel_executed
            ),
            all_zero_history_short_circuit=(
                all_zero_history_short_circuit
            ),
            extra_lineage_metadata=(
                extra_lineage_metadata
            ),
        )
    )

    snapshot_fingerprint = (
        parameter_snapshot
        .parameter_snapshot_fingerprint
        if parameter_snapshot is not None
        else None
    )

    lineage = MemoryExecutionLineage(
        operation_name=operation_name,
        source_lineage_fingerprints=(
            source_history
            .lineage_fingerprint(),
        ),
        architecture_fingerprint=(
            architecture
            .architecture_fingerprint
        ),
        parameter_snapshot_fingerprint=(
            snapshot_fingerprint
        ),
        configuration_fingerprint=(
            architecture
            .configuration_fingerprint
        ),
        node_axis_fingerprint=(
            source_history
            .node_axis
            .fingerprint()
        ),
        temporal_axis_fingerprint=(
            source_history
            .temporal_alignment_fingerprint()
        ),
        feature_axis_fingerprint=(
            source_history
            .feature_axis
            .fingerprint()
        ),
        lineage_metadata={
            **lineage_metadata,
            "recurrent_component_kind": (
                canonical_recurrent_component_kind(
                    config.cell_kind
                )
            ),
            "provenance_scope": (
                RECURRENT_PROVENANCE_SCOPE
            ),
        },
    )

    return MemoryComputationProvenance(
        architecture=architecture,
        lineage=lineage,
        parameter_snapshot=parameter_snapshot,
    )


# =============================================================================
# Import-boundary declaration
# =============================================================================


# Private module: no package-level public export surface.
__all__: tuple[str, ...] = ()
