"""
Private provenance utilities for simple temporal baselines.

Target path:
    urban_resilience_models/
        v2_hazard_conditioned_functional_ugnn/
            memory/
                baseline_encoders/
                    _provenance.py

This module centralizes only the mechanical construction of Phase 4 provenance
objects for Phase 5 baseline components.

It does not define new public runtime schemas. Baseline implementations still
return the canonical contracts:

- ``TemporalSequenceEncoding``;
- ``TemporalPoolingOutput``;
- ``UrbanMemory`` when assembled elsewhere.

Each caller remains responsible for supplying explicit scientific semantics,
including:

- the concrete component kind;
- dimensions;
- bias and activation choices;
- temporal-interaction policy;
- masking policy;
- zero-history behavior;
- pooling semantics.

The helper standardizes:

- canonical architecture fingerprints;
- optional configuration fingerprints;
- optional parameter-snapshot fingerprints;
- exact source-lineage linkage;
- node, temporal, and feature-axis fingerprints;
- construction of ``MemoryComputationProvenance``.

Parameter snapshots are never computed implicitly during a forward pass.
Callers must request them explicitly, typically for evaluation, export,
checkpoint auditing, or reproducibility artifacts.
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

from ..schemas.history_inputs import (
    HistoricalSequenceInputs,
)
from ..schemas.provenance import (
    MemoryArchitectureProvenance,
    MemoryComputationProvenance,
    MemoryExecutionLineage,
    MemoryParameterSnapshotProvenance,
)
from ..schemas.sequence_encoding import (
    TemporalSequenceEncoding,
)


# =============================================================================
# Private utility identity
# =============================================================================


BASELINE_PROVENANCE_IMPLEMENTATION_VERSION: Final[str] = "0.1"

BASELINE_PROVENANCE_SCOPE: Final[str] = (
    "mechanical_provenance_construction_only"
)


# =============================================================================
# Canonical JSON and fingerprints
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


def _require_optional_nonnegative_int(
    name: str,
    value: int | None,
) -> None:
    if value is None:
        return

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
            f"{name} must be an integer or None."
        )

    if value < 0:
        raise ValueError(
            f"{name} must be nonnegative."
        )


def _to_plain_json_value(
    value: Any,
) -> Any:
    """
    Convert one supported value to deterministic plain JSON data.

    Architecture metadata should remain compact. Tensor values and module
    parameters are deliberately excluded; parameter identity belongs in the
    optional parameter-snapshot contract.
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
        converted: dict[
            str,
            Any,
        ] = {}

        for key, item in (
            value.items()
        ):
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
    plain = _to_plain_json_value(
        payload
    )

    if not isinstance(
        plain,
        dict,
    ):
        raise TypeError(
            "Canonical provenance payload must be a mapping."
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


def baseline_configuration_fingerprint(
    configuration: Any,
) -> str:
    """
    Return a deterministic configuration fingerprint.

    Preferred configuration objects expose ``config_hash()``. Dataclasses,
    mappings, and plain JSON-compatible structures are also accepted.
    """

    config_hash = getattr(
        configuration,
        "config_hash",
        None,
    )

    if callable(
        config_hash
    ):
        value = config_hash()
        _require_nonempty_string(
            "configuration config_hash",
            value,
        )
        return value

    if is_dataclass(
        configuration
    ) and not isinstance(
        configuration,
        type,
    ):
        payload = asdict(
            configuration
        )
    elif isinstance(
        configuration,
        Mapping,
    ):
        payload = dict(
            configuration
        )
    else:
        payload = {
            "configuration": (
                _to_plain_json_value(
                    configuration
                )
            ),
        }

    if not isinstance(
        payload,
        Mapping,
    ):
        payload = {
            "configuration": payload,
        }

    return _fingerprint(
        payload
    )


def baseline_architecture_fingerprint(
    *,
    component_name: str,
    component_kind: str,
    architecture_metadata: Mapping[str, Any],
    configuration_fingerprint: str | None = None,
    implementation_version: str = (
        BASELINE_PROVENANCE_IMPLEMENTATION_VERSION
    ),
) -> str:
    """
    Fingerprint architecture semantics without current parameter values.
    """

    _require_nonempty_string(
        "component_name",
        component_name,
    )
    _require_nonempty_string(
        "component_kind",
        component_kind,
    )
    _require_nonempty_string(
        "implementation_version",
        implementation_version,
    )
    _require_optional_nonempty_string(
        "configuration_fingerprint",
        configuration_fingerprint,
    )

    if not isinstance(
        architecture_metadata,
        Mapping,
    ):
        raise TypeError(
            "architecture_metadata must be a mapping."
        )

    return _fingerprint(
        {
            "component_name": (
                component_name
            ),
            "component_kind": (
                component_kind
            ),
            "implementation_version": (
                implementation_version
            ),
            "configuration_fingerprint": (
                configuration_fingerprint
            ),
            "architecture_metadata": (
                architecture_metadata
            ),
            "provenance_scope": (
                BASELINE_PROVENANCE_SCOPE
            ),
        }
    )


# =============================================================================
# Parameter counting and explicit snapshot identity
# =============================================================================


def count_module_parameters(
    module: nn.Module,
) -> tuple[int, int]:
    """
    Return ``(parameter_count, trainable_parameter_count)``.
    """

    if not isinstance(
        module,
        nn.Module,
    ):
        raise TypeError(
            "module must be a torch.nn.Module."
        )

    parameter_count = 0
    trainable_count = 0

    for parameter in (
        module.parameters()
    ):
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
    Fingerprint named parameters independently of device placement.

    Buffers are excluded. This function represents trainable/model parameter
    identity, not complete module state or optimizer state.
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
        b"baseline_parameter_snapshot_v1"
    )

    names: list[str] = []

    for name, parameter in (
        module.named_parameters()
    ):
        _require_nonempty_string(
            "parameter name",
            name,
        )
        names.append(
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
        names
    ) != len(
        set(
            names
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
    Explicitly construct one parameter-snapshot provenance object.

    This helper must not be called automatically by baseline ``forward``.
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


# =============================================================================
# Architecture and computation-provenance construction
# =============================================================================


def build_baseline_architecture_provenance(
    *,
    component_name: str,
    component_kind: str,
    architecture_metadata: Mapping[str, Any],
    configuration_fingerprint: str | None = None,
    implementation_version: str = (
        BASELINE_PROVENANCE_IMPLEMENTATION_VERSION
    ),
) -> MemoryArchitectureProvenance:
    """
    Build stable architecture provenance for one baseline component.
    """

    fingerprint = baseline_architecture_fingerprint(
        component_name=component_name,
        component_kind=component_kind,
        architecture_metadata=(
            architecture_metadata
        ),
        configuration_fingerprint=(
            configuration_fingerprint
        ),
        implementation_version=(
            implementation_version
        ),
    )

    return MemoryArchitectureProvenance(
        component_name=component_name,
        component_kind=component_kind,
        architecture_fingerprint=fingerprint,
        configuration_fingerprint=(
            configuration_fingerprint
        ),
        implementation_version=(
            implementation_version
        ),
        architecture_metadata={
            **dict(
                architecture_metadata
            ),
            "provenance_scope": (
                BASELINE_PROVENANCE_SCOPE
            ),
        },
    )


def _build_computation_provenance(
    *,
    source_lineage_fingerprints: Sequence[str],
    node_axis_fingerprint: str,
    temporal_axis_fingerprint: str,
    feature_axis_fingerprint: str,
    operation_name: str,
    component_name: str,
    component_kind: str,
    architecture_metadata: Mapping[str, Any],
    configuration_fingerprint: str | None,
    parameter_snapshot: (
        MemoryParameterSnapshotProvenance
        | None
    ),
    lineage_metadata: Mapping[str, Any],
    implementation_version: str,
) -> MemoryComputationProvenance:
    _require_nonempty_string(
        "operation_name",
        operation_name,
    )

    source_fingerprints = tuple(
        source_lineage_fingerprints
    )

    if not source_fingerprints:
        raise ValueError(
            "At least one source lineage fingerprint is required."
        )

    for index, value in enumerate(
        source_fingerprints
    ):
        _require_nonempty_string(
            f"source_lineage_fingerprints[{index}]",
            value,
        )

    if len(
        set(
            source_fingerprints
        )
    ) != len(
        source_fingerprints
    ):
        raise ValueError(
            "source_lineage_fingerprints must be unique."
        )

    for name, value in (
        (
            "node_axis_fingerprint",
            node_axis_fingerprint,
        ),
        (
            "temporal_axis_fingerprint",
            temporal_axis_fingerprint,
        ),
        (
            "feature_axis_fingerprint",
            feature_axis_fingerprint,
        ),
    ):
        _require_nonempty_string(
            name,
            value,
        )

    if (
        parameter_snapshot
        is not None
        and not isinstance(
            parameter_snapshot,
            MemoryParameterSnapshotProvenance,
        )
    ):
        raise TypeError(
            "parameter_snapshot must be a "
            "MemoryParameterSnapshotProvenance or None."
        )

    if not isinstance(
        lineage_metadata,
        Mapping,
    ):
        raise TypeError(
            "lineage_metadata must be a mapping."
        )

    architecture = (
        build_baseline_architecture_provenance(
            component_name=component_name,
            component_kind=component_kind,
            architecture_metadata=(
                architecture_metadata
            ),
            configuration_fingerprint=(
                configuration_fingerprint
            ),
            implementation_version=(
                implementation_version
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
            source_fingerprints
        ),
        architecture_fingerprint=(
            architecture
            .architecture_fingerprint
        ),
        parameter_snapshot_fingerprint=(
            snapshot_fingerprint
        ),
        configuration_fingerprint=(
            configuration_fingerprint
        ),
        node_axis_fingerprint=(
            node_axis_fingerprint
        ),
        temporal_axis_fingerprint=(
            temporal_axis_fingerprint
        ),
        feature_axis_fingerprint=(
            feature_axis_fingerprint
        ),
        lineage_metadata={
            **dict(
                lineage_metadata
            ),
            "baseline_component_kind": (
                component_kind
            ),
            "provenance_scope": (
                BASELINE_PROVENANCE_SCOPE
            ),
        },
    )

    return MemoryComputationProvenance(
        architecture=architecture,
        lineage=lineage,
        parameter_snapshot=parameter_snapshot,
    )


def build_sequence_encoder_computation_provenance(
    *,
    source_history: HistoricalSequenceInputs,
    operation_name: str,
    component_name: str,
    component_kind: str,
    architecture_metadata: Mapping[str, Any],
    configuration_fingerprint: str | None = None,
    parameter_snapshot: (
        MemoryParameterSnapshotProvenance
        | None
    ) = None,
    lineage_metadata: Mapping[str, Any] | None = None,
    implementation_version: str = (
        BASELINE_PROVENANCE_IMPLEMENTATION_VERSION
    ),
) -> MemoryComputationProvenance:
    """
    Build provenance for a baseline sequence encoder.

    The exact ``HistoricalSequenceInputs`` lineage and axes are retained.
    """

    if not isinstance(
        source_history,
        HistoricalSequenceInputs,
    ):
        raise TypeError(
            "source_history must be a HistoricalSequenceInputs."
        )

    return _build_computation_provenance(
        source_lineage_fingerprints=(
            source_history
            .lineage_fingerprint(),
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
        operation_name=operation_name,
        component_name=component_name,
        component_kind=component_kind,
        architecture_metadata=(
            architecture_metadata
        ),
        configuration_fingerprint=(
            configuration_fingerprint
        ),
        parameter_snapshot=parameter_snapshot,
        lineage_metadata=(
            {}
            if lineage_metadata is None
            else lineage_metadata
        ),
        implementation_version=(
            implementation_version
        ),
    )


def build_temporal_pooling_computation_provenance(
    *,
    source_encoding: TemporalSequenceEncoding,
    operation_name: str,
    component_name: str,
    component_kind: str,
    architecture_metadata: Mapping[str, Any],
    configuration_fingerprint: str | None = None,
    parameter_snapshot: (
        MemoryParameterSnapshotProvenance
        | None
    ) = None,
    lineage_metadata: Mapping[str, Any] | None = None,
    implementation_version: str = (
        BASELINE_PROVENANCE_IMPLEMENTATION_VERSION
    ),
) -> MemoryComputationProvenance:
    """
    Build provenance for deterministic or learned baseline temporal pooling.

    The exact ``TemporalSequenceEncoding`` lineage and source axes are retained.
    """

    if not isinstance(
        source_encoding,
        TemporalSequenceEncoding,
    ):
        raise TypeError(
            "source_encoding must be a TemporalSequenceEncoding."
        )

    return _build_computation_provenance(
        source_lineage_fingerprints=(
            source_encoding
            .lineage_fingerprint(),
        ),
        node_axis_fingerprint=(
            source_encoding
            .node_axis
            .fingerprint()
        ),
        temporal_axis_fingerprint=(
            source_encoding
            .temporal_alignment_fingerprint()
        ),
        feature_axis_fingerprint=(
            source_encoding
            .feature_axis
            .fingerprint()
        ),
        operation_name=operation_name,
        component_name=component_name,
        component_kind=component_kind,
        architecture_metadata=(
            architecture_metadata
        ),
        configuration_fingerprint=(
            configuration_fingerprint
        ),
        parameter_snapshot=parameter_snapshot,
        lineage_metadata=(
            {}
            if lineage_metadata is None
            else lineage_metadata
        ),
        implementation_version=(
            implementation_version
        ),
    )


# Private module: no package-level public export surface.
__all__: tuple[str, ...] = ()
