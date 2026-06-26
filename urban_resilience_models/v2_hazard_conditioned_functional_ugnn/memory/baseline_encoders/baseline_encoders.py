"""
Construction and execution dispatch for Phase 5 temporal baselines.

Target path:
    urban_resilience_models/
        v2_hazard_conditioned_functional_ugnn/
            memory/
                baseline_encoders/
                    baseline_encoders.py

This module keeps sequence representation and temporal reduction explicit.

Sequence builders construct exactly one of:

- ``IdentitySequenceEncoder``;
- ``LinearProjectionSequenceEncoder``;
- ``PointwiseMLPSequenceEncoder``.

Temporal-pooling builders construct exactly one of:

- ``MaskedMeanTemporalPooler``;
- ``LastValidTemporalPooler``.

The module deliberately does not define a combined ``BaselineEncoder`` class.
Sequence encoding and temporal pooling have different scientific meanings and
different canonical Phase 4 outputs:

    HistoricalSequenceInputs
        -> TemporalSequenceEncoding
        -> TemporalPoolingOutput

Keeping the stages separate supports controlled comparisons such as:

- identity versus width-matched projection;
- linear versus nonlinear pointwise feature transformation;
- masked mean versus last valid;
- generic pooling versus later hazard-conditioned retrieval.

Recognized but unimplemented pooling kinds are rejected with
``NotImplementedError``. They are never silently replaced by another pooling
operation.

The thin execution helpers call modules through ``nn.Module.__call__`` so
PyTorch hooks and standard module behavior remain intact. They add only type
and output-contract validation.
"""

from __future__ import annotations

from typing import Final, TypeAlias

from torch import nn

from ..config import (
    BaselineSequenceEncoderConfig,
    BaselineSequenceEncoderKind,
    TemporalPoolingConfig,
    TemporalSequenceEncoderConfig,
)
from ..schemas.history_inputs import (
    HistoricalSequenceInputs,
)
from ..schemas.provenance import (
    MemoryParameterSnapshotProvenance,
)
from ..schemas.sequence_encoding import (
    TemporalSequenceEncoding,
)
from ..schemas.temporal_pooling import (
    TemporalPoolingKind,
    TemporalPoolingOutput,
)
from .identity_sequence_encoder import (
    IdentitySequenceEncoder,
)
from .last_valid_pooling import (
    LastValidTemporalPooler,
)
from .linear_projection_sequence_encoder import (
    LinearProjectionSequenceEncoder,
)
from .masked_mean_pooling import (
    MaskedMeanTemporalPooler,
)
from .pointwise_mlp_sequence_encoder import (
    PointwiseMLPSequenceEncoder,
)


# =============================================================================
# Dispatcher identity and capability vocabulary
# =============================================================================


BASELINE_ENCODER_DISPATCHER_IMPLEMENTATION_VERSION: Final[str] = "0.1"

BASELINE_ENCODER_DISPATCHER_SCIENTIFIC_INTERPRETATION: Final[str] = (
    "explicit_sequence_representation_and_temporal_reduction_dispatch"
)

IMPLEMENTED_BASELINE_SEQUENCE_ENCODER_KINDS: Final[
    tuple[str, ...]
] = (
    BaselineSequenceEncoderKind.IDENTITY.value,
    BaselineSequenceEncoderKind.LINEAR_PROJECTION.value,
    BaselineSequenceEncoderKind.TEMPORAL_MLP.value,
)

IMPLEMENTED_BASELINE_TEMPORAL_POOLING_KINDS: Final[
    tuple[str, ...]
] = (
    TemporalPoolingKind.MASKED_MEAN.value,
    TemporalPoolingKind.LAST_VALID.value,
)

RECOGNIZED_UNIMPLEMENTED_BASELINE_TEMPORAL_POOLING_KINDS: Final[
    tuple[str, ...]
] = (
    TemporalPoolingKind.MASKED_MAX.value,
    TemporalPoolingKind.LEARNED_ATTENTION.value,
    TemporalPoolingKind.MULTIHEAD_ATTENTION.value,
    TemporalPoolingKind.OTHER.value,
)


# =============================================================================
# Supported module types
# =============================================================================


BaselineSequenceEncoderModule: TypeAlias = (
    IdentitySequenceEncoder
    | LinearProjectionSequenceEncoder
    | PointwiseMLPSequenceEncoder
)

BaselineTemporalPoolerModule: TypeAlias = (
    MaskedMeanTemporalPooler
    | LastValidTemporalPooler
)

BASELINE_SEQUENCE_ENCODER_TYPES: Final[
    tuple[type[nn.Module], ...]
] = (
    IdentitySequenceEncoder,
    LinearProjectionSequenceEncoder,
    PointwiseMLPSequenceEncoder,
)

BASELINE_TEMPORAL_POOLER_TYPES: Final[
    tuple[type[nn.Module], ...]
] = (
    MaskedMeanTemporalPooler,
    LastValidTemporalPooler,
)


# =============================================================================
# Type predicates
# =============================================================================


def is_baseline_sequence_encoder(
    module: object,
) -> bool:
    """Return whether ``module`` is one implemented Phase 5 sequence encoder."""

    return isinstance(
        module,
        BASELINE_SEQUENCE_ENCODER_TYPES,
    )


def is_baseline_temporal_pooler(
    module: object,
) -> bool:
    """Return whether ``module`` is one implemented Phase 5 temporal pooler."""

    return isinstance(
        module,
        BASELINE_TEMPORAL_POOLER_TYPES,
    )


# =============================================================================
# Configuration extraction
# =============================================================================


def _extract_baseline_sequence_config(
    config: (
        BaselineSequenceEncoderConfig
        | TemporalSequenceEncoderConfig
    ),
) -> BaselineSequenceEncoderConfig:
    """
    Resolve a direct baseline config or the baseline branch of dispatch config.
    """

    if isinstance(
        config,
        BaselineSequenceEncoderConfig,
    ):
        return config

    if not isinstance(
        config,
        TemporalSequenceEncoderConfig,
    ):
        raise TypeError(
            "config must be a BaselineSequenceEncoderConfig or "
            "TemporalSequenceEncoderConfig."
        )

    if config.baseline is None:
        raise NotImplementedError(
            "The Phase 5 baseline dispatcher cannot construct "
            f"encoder_kind={config.encoder_kind.value!r}. Supply a "
            "TemporalSequenceEncoderConfig whose active branch is "
            "'baseline', or use the recurrent/Transformer dispatcher "
            "introduced in its corresponding phase."
        )

    if (
        config.recurrent is not None
        or config.transformer is not None
    ):
        raise ValueError(
            "A baseline TemporalSequenceEncoderConfig must not also "
            "contain recurrent or Transformer configuration."
        )

    return config.baseline


# =============================================================================
# Builders
# =============================================================================


def build_baseline_sequence_encoder(
    config: (
        BaselineSequenceEncoderConfig
        | TemporalSequenceEncoderConfig
    ),
) -> BaselineSequenceEncoderModule:
    """
    Construct one implemented no-temporal-interaction sequence encoder.

    Parameters
    ----------
    config:
        Either a direct ``BaselineSequenceEncoderConfig`` or a
        ``TemporalSequenceEncoderConfig`` whose active branch is ``baseline``.

    Returns
    -------
    BaselineSequenceEncoderModule
        A newly constructed module. No parameters are shared with previously
        built modules.
    """

    baseline_config = (
        _extract_baseline_sequence_config(
            config
        )
    )

    if (
        baseline_config.kind
        == BaselineSequenceEncoderKind.IDENTITY
    ):
        return IdentitySequenceEncoder(
            baseline_config
        )

    if (
        baseline_config.kind
        == BaselineSequenceEncoderKind.LINEAR_PROJECTION
    ):
        return LinearProjectionSequenceEncoder(
            baseline_config
        )

    if (
        baseline_config.kind
        == BaselineSequenceEncoderKind.TEMPORAL_MLP
    ):
        return PointwiseMLPSequenceEncoder(
            baseline_config
        )

    # ``BaselineSequenceEncoderKind`` currently contains only implemented
    # values. Keep an explicit branch so future recognized values fail loudly.
    raise NotImplementedError(
        "Recognized baseline sequence encoder kind "
        f"{baseline_config.kind.value!r} is not implemented. "
        f"Implemented kinds: {IMPLEMENTED_BASELINE_SEQUENCE_ENCODER_KINDS}."
    )


def build_baseline_temporal_pooler(
    config: TemporalPoolingConfig,
) -> BaselineTemporalPoolerModule:
    """
    Construct one implemented deterministic temporal pooler.

    Recognized learned or otherwise deferred pooling modes raise
    ``NotImplementedError`` rather than being silently substituted.
    """

    if not isinstance(
        config,
        TemporalPoolingConfig,
    ):
        raise TypeError(
            "config must be a TemporalPoolingConfig."
        )

    if config.kind == TemporalPoolingKind.MASKED_MEAN:
        return MaskedMeanTemporalPooler(
            config
        )

    if config.kind == TemporalPoolingKind.LAST_VALID:
        return LastValidTemporalPooler(
            config
        )

    if (
        config.kind.value
        in RECOGNIZED_UNIMPLEMENTED_BASELINE_TEMPORAL_POOLING_KINDS
    ):
        raise NotImplementedError(
            "Temporal pooling kind "
            f"{config.kind.value!r} is recognized but is not part of "
            "the Phase 5 deterministic baseline implementation. "
            "Implemented kinds: "
            f"{IMPLEMENTED_BASELINE_TEMPORAL_POOLING_KINDS}."
        )

    raise NotImplementedError(
        "Temporal pooling kind "
        f"{config.kind.value!r} is not implemented by the Phase 5 "
        "baseline dispatcher."
    )


# =============================================================================
# Thin execution helpers
# =============================================================================


def encode_baseline_history(
    encoder: BaselineSequenceEncoderModule,
    source_history: HistoricalSequenceInputs,
    *,
    parameter_snapshot: (
        MemoryParameterSnapshotProvenance
        | None
    ) = None,
) -> TemporalSequenceEncoding:
    """
    Execute one supported baseline sequence encoder.

    The helper does not change train/eval mode, move devices, cast dtypes,
    generate parameter snapshots, or alter the source history.
    """

    if not is_baseline_sequence_encoder(
        encoder
    ):
        raise TypeError(
            "encoder must be an implemented Phase 5 baseline sequence "
            f"encoder; observed {type(encoder).__name__!r}."
        )

    if not isinstance(
        source_history,
        HistoricalSequenceInputs,
    ):
        raise TypeError(
            "source_history must be a HistoricalSequenceInputs."
        )

    output = encoder(
        source_history,
        parameter_snapshot=parameter_snapshot,
    )

    if not isinstance(
        output,
        TemporalSequenceEncoding,
    ):
        raise RuntimeError(
            "A baseline sequence encoder returned an invalid runtime "
            f"contract {type(output).__name__!r}; expected "
            "TemporalSequenceEncoding."
        )

    if output.source_history is not source_history:
        raise RuntimeError(
            "A baseline sequence encoder must preserve the exact "
            "source_history object."
        )

    return output


def pool_baseline_sequence(
    pooler: BaselineTemporalPoolerModule,
    source_encoding: TemporalSequenceEncoding,
    *,
    parameter_snapshot: (
        MemoryParameterSnapshotProvenance
        | None
    ) = None,
) -> TemporalPoolingOutput:
    """
    Execute one supported deterministic baseline temporal pooler.

    The helper does not construct an ``UrbanMemory`` object; assembly remains
    an explicit downstream concern.
    """

    if not is_baseline_temporal_pooler(
        pooler
    ):
        raise TypeError(
            "pooler must be an implemented Phase 5 baseline temporal "
            f"pooler; observed {type(pooler).__name__!r}."
        )

    if not isinstance(
        source_encoding,
        TemporalSequenceEncoding,
    ):
        raise TypeError(
            "source_encoding must be a TemporalSequenceEncoding."
        )

    output = pooler(
        source_encoding,
        parameter_snapshot=parameter_snapshot,
    )

    if not isinstance(
        output,
        TemporalPoolingOutput,
    ):
        raise RuntimeError(
            "A baseline temporal pooler returned an invalid runtime "
            f"contract {type(output).__name__!r}; expected "
            "TemporalPoolingOutput."
        )

    if output.source_encoding is not source_encoding:
        raise RuntimeError(
            "A baseline temporal pooler must preserve the exact "
            "source_encoding object."
        )

    return output


def run_baseline_pipeline(
    sequence_encoder: BaselineSequenceEncoderModule,
    source_history: HistoricalSequenceInputs,
    *,
    temporal_pooler: BaselineTemporalPoolerModule | None = None,
    sequence_parameter_snapshot: (
        MemoryParameterSnapshotProvenance
        | None
    ) = None,
    pooling_parameter_snapshot: (
        MemoryParameterSnapshotProvenance
        | None
    ) = None,
) -> tuple[
    TemporalSequenceEncoding,
    TemporalPoolingOutput | None,
]:
    """
    Execute an explicit baseline sequence stage and optional pooling stage.

    This convenience function is deliberately stateless and returns the two
    canonical outputs directly. It does not create a combined model class or a
    baseline-specific wrapper schema.
    """

    if (
        temporal_pooler is None
        and pooling_parameter_snapshot is not None
    ):
        raise ValueError(
            "pooling_parameter_snapshot requires temporal_pooler."
        )

    sequence_output = encode_baseline_history(
        sequence_encoder,
        source_history,
        parameter_snapshot=(
            sequence_parameter_snapshot
        ),
    )

    if temporal_pooler is None:
        return (
            sequence_output,
            None,
        )

    pooling_output = pool_baseline_sequence(
        temporal_pooler,
        sequence_output,
        parameter_snapshot=(
            pooling_parameter_snapshot
        ),
    )

    return (
        sequence_output,
        pooling_output,
    )


# =============================================================================
# Compact aliases
# =============================================================================


build_sequence_encoder = build_baseline_sequence_encoder
build_temporal_pooler = build_baseline_temporal_pooler
encode_history = encode_baseline_history
pool_sequence = pool_baseline_sequence
run_pipeline = run_baseline_pipeline


# =============================================================================
# Public API
# =============================================================================


__all__ = (
    # Identity and capabilities.
    "BASELINE_ENCODER_DISPATCHER_IMPLEMENTATION_VERSION",
    "BASELINE_ENCODER_DISPATCHER_SCIENTIFIC_INTERPRETATION",
    "IMPLEMENTED_BASELINE_SEQUENCE_ENCODER_KINDS",
    "IMPLEMENTED_BASELINE_TEMPORAL_POOLING_KINDS",
    "RECOGNIZED_UNIMPLEMENTED_BASELINE_TEMPORAL_POOLING_KINDS",

    # Supported module types.
    "BaselineSequenceEncoderModule",
    "BaselineTemporalPoolerModule",
    "BASELINE_SEQUENCE_ENCODER_TYPES",
    "BASELINE_TEMPORAL_POOLER_TYPES",

    # Predicates.
    "is_baseline_sequence_encoder",
    "is_baseline_temporal_pooler",

    # Builders.
    "build_baseline_sequence_encoder",
    "build_baseline_temporal_pooler",

    # Execution.
    "encode_baseline_history",
    "pool_baseline_sequence",
    "run_baseline_pipeline",

    # Compact aliases.
    "build_sequence_encoder",
    "build_temporal_pooler",
    "encode_history",
    "pool_sequence",
    "run_pipeline",
)
