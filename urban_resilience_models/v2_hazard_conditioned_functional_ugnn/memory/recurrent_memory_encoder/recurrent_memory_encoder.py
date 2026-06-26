"""
Construction, dispatch, and orchestration for Phase 6 recurrent memory encoders.

Target path:
    urban_resilience_models/
        v2_hazard_conditioned_functional_ugnn/
            memory/
                recurrent_memory_encoder/
                    recurrent_memory_encoder.py

This module is the thin public orchestration layer for Phase 6 recurrence. It
selects exactly one cell-specific encoder and delegates all numerical execution
to that implementation:

    RecurrentSequenceEncoderConfig
        -> GRUSequenceEncoder | LSTMSequenceEncoder
        -> TemporalSequenceEncoding | RecurrentSequenceEncoderRun

It does not implement a third recurrent kernel and does not duplicate history
canonicalization, packing, reference execution, state restoration, parameter
snapshot validation, or provenance construction.

Two configuration entry points are accepted:

- a direct ``RecurrentSequenceEncoderConfig``;
- a top-level ``TemporalSequenceEncoderConfig`` whose active branch is
  ``recurrent``.

``RecurrentMemoryEncoder`` is an ``nn.Module`` facade. Device/dtype movement,
training mode, hooks, parameters, and state dictionaries remain standard
PyTorch behavior because the selected cell-specific encoder is a registered
child module.

Diagnostics remain explicit. Ordinary ``forward`` and ``encode_with_state`` do
not compute diagnostic reports. ``encode_with_diagnostics`` and
``diagnose_run`` invoke the detached diagnostics facade only when requested.

Architecture identity
---------------------
The wrapper delegates architecture provenance to the selected cell-specific
encoder. Consequently, direct and orchestrated construction of the same GRU or
LSTM architecture share the same architecture fingerprint. Packed/reference
execution flags remain execution-lineage choices rather than architecture
identity.

Parameter snapshots
-------------------
Snapshot construction delegates to the selected cell-specific encoder. This
preserves the same parameter names and snapshot fingerprint as direct use of
that encoder. Snapshots are explicit and are never created automatically.
"""

from __future__ import annotations

from typing import Final
from typing import TypeAlias

from torch import nn

from ..config import (
    RecurrentCellKind,
    RecurrentSequenceEncoderConfig,
    TemporalSequenceEncoderConfig,
)
from ..schemas.history_inputs import (
    HistoricalSequenceInputs,
)
from ..schemas.provenance import (
    MemoryArchitectureProvenance,
    MemoryParameterSnapshotProvenance,
)
from ..schemas.sequence_encoding import (
    TemporalSequenceEncoding,
    TemporalSequenceEncoderKind,
)
from .diagnostics import (
    PackedReferenceComparisonDiagnostics,
    RecurrentDiagnostics,
    RecurrentRunDiagnostics,
)
from .gru_encoder import (
    GRUSequenceEncoder,
)
from .lstm_encoder import (
    LSTMSequenceEncoder,
)
from .schemas import (
    RecurrentSequenceEncoderRun,
    RecurrentStateLayout,
)


# =============================================================================
# Dispatcher identity and frozen capability vocabulary
# =============================================================================


RECURRENT_MEMORY_ENCODER_IMPLEMENTATION_VERSION: Final[str] = "0.1"

RECURRENT_MEMORY_ENCODER_COMPONENT_NAME: Final[str] = (
    "recurrent_memory_encoder"
)

RECURRENT_MEMORY_ENCODER_COMPONENT_KIND: Final[str] = (
    "recurrent_sequence_encoder_dispatcher_and_facade"
)

RECURRENT_MEMORY_ENCODER_OPERATION_NAME: Final[str] = (
    "dispatch_and_encode_recurrent_temporal_history"
)

RECURRENT_MEMORY_ENCODER_DIAGNOSTICS_POLICY: Final[str] = (
    "explicit_detached_diagnostics_only_v1"
)

RECURRENT_MEMORY_ENCODER_PARAMETER_SNAPSHOT_POLICY: Final[str] = (
    "explicit_delegate_to_selected_cell_encoder_v1"
)

RECURRENT_MEMORY_ENCODER_ARCHITECTURE_IDENTITY_POLICY: Final[str] = (
    "delegate_cell_specific_architecture_identity_v1"
)

RECURRENT_MEMORY_ENCODER_SCIENTIFIC_INTERPRETATION: Final[str] = (
    "cell_family_dispatch_and_execution_orchestration_not_new_model_family"
)

IMPLEMENTED_RECURRENT_CELL_KINDS: Final[tuple[str, ...]] = (
    RecurrentCellKind.GRU.value,
    RecurrentCellKind.LSTM.value,
)

IMPLEMENTED_RECURRENT_SEQUENCE_ENCODER_KINDS: Final[
    tuple[str, ...]
] = (
    TemporalSequenceEncoderKind.GRU.value,
    TemporalSequenceEncoderKind.LSTM.value,
)


# =============================================================================
# Supported direct module types
# =============================================================================


RecurrentSequenceEncoderModule: TypeAlias = (
    GRUSequenceEncoder
    | LSTMSequenceEncoder
)

RECURRENT_SEQUENCE_ENCODER_TYPES: Final[
    tuple[type[nn.Module], ...]
] = (
    GRUSequenceEncoder,
    LSTMSequenceEncoder,
)


# =============================================================================
# Configuration extraction
# =============================================================================


def extract_recurrent_sequence_config(
    config: (
        RecurrentSequenceEncoderConfig
        | TemporalSequenceEncoderConfig
    ),
) -> RecurrentSequenceEncoderConfig:
    """
    Resolve a direct recurrent config or the recurrent branch of dispatch config.

    Baseline and Transformer top-level configurations are rejected explicitly;
    no fallback encoder is selected.
    """

    if isinstance(
        config,
        RecurrentSequenceEncoderConfig,
    ):
        return config

    if not isinstance(
        config,
        TemporalSequenceEncoderConfig,
    ):
        raise TypeError(
            "config must be a RecurrentSequenceEncoderConfig or "
            "TemporalSequenceEncoderConfig."
        )

    if config.recurrent is None:
        raise NotImplementedError(
            "The Phase 6 recurrent dispatcher cannot construct "
            f"encoder_kind={config.encoder_kind.value!r}. Supply a "
            "TemporalSequenceEncoderConfig whose active branch is "
            "'recurrent', or use the dispatcher belonging to the active "
            "baseline/Transformer phase."
        )

    if (
        config.baseline is not None
        or config.transformer is not None
    ):
        raise ValueError(
            "A recurrent TemporalSequenceEncoderConfig must not also "
            "contain baseline or Transformer configuration."
        )

    expected_kind = (
        config.recurrent
        .schema_encoder_kind
    )

    if config.encoder_kind != expected_kind:
        raise ValueError(
            "Top-level encoder_kind does not match the active recurrent "
            f"configuration; expected {expected_kind.value!r}."
        )

    return config.recurrent


# =============================================================================
# Direct builders and predicates
# =============================================================================


def is_recurrent_sequence_encoder(
    module: object,
) -> bool:
    """Return whether ``module`` is one direct Phase 6 GRU/LSTM encoder."""

    return isinstance(
        module,
        RECURRENT_SEQUENCE_ENCODER_TYPES,
    )


def build_recurrent_sequence_encoder(
    config: (
        RecurrentSequenceEncoderConfig
        | TemporalSequenceEncoderConfig
    ),
) -> RecurrentSequenceEncoderModule:
    """
    Construct exactly one cell-specific recurrent sequence encoder.

    No parameters are shared with previously built modules.
    """

    recurrent_config = extract_recurrent_sequence_config(
        config
    )

    if recurrent_config.cell_kind == RecurrentCellKind.GRU:
        return GRUSequenceEncoder(
            recurrent_config
        )

    if recurrent_config.cell_kind == RecurrentCellKind.LSTM:
        return LSTMSequenceEncoder(
            recurrent_config
        )

    raise NotImplementedError(
        "Recognized recurrent cell kind "
        f"{recurrent_config.cell_kind.value!r} is not implemented. "
        f"Implemented kinds: {IMPLEMENTED_RECURRENT_CELL_KINDS}."
    )


# =============================================================================
# Thin direct execution helpers
# =============================================================================


def encode_recurrent_history(
    encoder: RecurrentSequenceEncoderModule,
    source_history: HistoricalSequenceInputs,
    *,
    parameter_snapshot: (
        MemoryParameterSnapshotProvenance
        | None
    ) = None,
) -> TemporalSequenceEncoding:
    """
    Execute one direct GRU/LSTM encoder and return the public sequence contract.

    The helper calls the module through ``nn.Module.__call__`` so hooks and
    ordinary PyTorch module behavior remain active.
    """

    if not is_recurrent_sequence_encoder(
        encoder
    ):
        raise TypeError(
            "encoder must be a GRUSequenceEncoder or LSTMSequenceEncoder; "
            f"observed {type(encoder).__name__!r}."
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
            "A recurrent sequence encoder returned an invalid public "
            f"contract {type(output).__name__!r}; expected "
            "TemporalSequenceEncoding."
        )

    if output.source_history is not source_history:
        raise RuntimeError(
            "A recurrent sequence encoder must preserve the exact "
            "source_history object."
        )

    if output.encoder_kind != encoder.config.schema_encoder_kind:
        raise RuntimeError(
            "A recurrent sequence encoder returned an unexpected "
            "encoder_kind."
        )

    return output


def encode_recurrent_history_with_state(
    encoder: RecurrentSequenceEncoderModule,
    source_history: HistoricalSequenceInputs,
    *,
    parameter_snapshot: (
        MemoryParameterSnapshotProvenance
        | None
    ) = None,
) -> RecurrentSequenceEncoderRun:
    """Execute one direct GRU/LSTM encoder and retain final recurrent state."""

    if not is_recurrent_sequence_encoder(
        encoder
    ):
        raise TypeError(
            "encoder must be a GRUSequenceEncoder or LSTMSequenceEncoder; "
            f"observed {type(encoder).__name__!r}."
        )

    if not isinstance(
        source_history,
        HistoricalSequenceInputs,
    ):
        raise TypeError(
            "source_history must be a HistoricalSequenceInputs."
        )

    run = encoder.encode_with_state(
        source_history,
        parameter_snapshot=parameter_snapshot,
    )

    if not isinstance(
        run,
        RecurrentSequenceEncoderRun,
    ):
        raise RuntimeError(
            "A recurrent sequence encoder returned an invalid stateful "
            f"contract {type(run).__name__!r}; expected "
            "RecurrentSequenceEncoderRun."
        )

    if run.source_history is not source_history:
        raise RuntimeError(
            "A recurrent run must preserve the exact source_history object."
        )

    if run.encoder_kind != encoder.config.schema_encoder_kind:
        raise RuntimeError(
            "A recurrent run returned an unexpected encoder_kind."
        )

    return run


# =============================================================================
# RecurrentMemoryEncoder facade
# =============================================================================


class RecurrentMemoryEncoder(nn.Module):
    """
    Thin GRU/LSTM construction, execution, and diagnostics facade.

    Parameters
    ----------
    config:
        Direct recurrent configuration or a top-level sequence configuration
        whose active branch is recurrent.

    diagnostics:
        Optional detached diagnostics policy. ``None`` creates an enabled
        default facade. Diagnostics are still computed only by explicit
        diagnostic methods.
    """

    config: RecurrentSequenceEncoderConfig
    requested_config: (
        RecurrentSequenceEncoderConfig
        | TemporalSequenceEncoderConfig
    )
    encoder: RecurrentSequenceEncoderModule
    diagnostics: RecurrentDiagnostics

    def __init__(
        self,
        config: (
            RecurrentSequenceEncoderConfig
            | TemporalSequenceEncoderConfig
        ),
        *,
        diagnostics: RecurrentDiagnostics | None = None,
    ) -> None:
        super().__init__()

        recurrent_config = extract_recurrent_sequence_config(
            config
        )

        if (
            diagnostics is not None
            and not isinstance(
                diagnostics,
                RecurrentDiagnostics,
            )
        ):
            raise TypeError(
                "diagnostics must be a RecurrentDiagnostics or None."
            )

        self.requested_config = config
        self.config = recurrent_config
        self.encoder = build_recurrent_sequence_encoder(
            recurrent_config
        )
        self.diagnostics = (
            diagnostics
            if diagnostics is not None
            else RecurrentDiagnostics()
        )

    # -------------------------------------------------------------------------
    # Selected implementation and architecture properties
    # -------------------------------------------------------------------------

    @property
    def core_encoder(
        self,
    ) -> RecurrentSequenceEncoderModule:
        return self.encoder

    @property
    def cell_kind(
        self,
    ) -> RecurrentCellKind:
        return self.config.cell_kind

    @property
    def encoder_kind(
        self,
    ) -> TemporalSequenceEncoderKind:
        return self.config.schema_encoder_kind

    @property
    def input_dim(
        self,
    ) -> int:
        return self.encoder.input_dim

    @property
    def recurrent_input_dim(
        self,
    ) -> int:
        return self.encoder.recurrent_input_dim

    @property
    def hidden_dim(
        self,
    ) -> int:
        return self.encoder.hidden_dim

    @property
    def output_dim(
        self,
    ) -> int:
        return self.encoder.output_dim

    @property
    def num_layers(
        self,
    ) -> int:
        return self.encoder.num_layers

    @property
    def num_directions(
        self,
    ) -> int:
        return self.encoder.num_directions

    @property
    def is_bidirectional(
        self,
    ) -> bool:
        return self.encoder.is_bidirectional

    @property
    def kernel(
        self,
    ) -> nn.GRU | nn.LSTM:
        return self.encoder.kernel

    @property
    def input_adapter(
        self,
    ) -> nn.Module:
        return self.encoder.input_adapter

    @property
    def state_layout(
        self,
    ) -> RecurrentStateLayout:
        return self.encoder.state_layout

    @property
    def parameter_count(
        self,
    ) -> int:
        return self.encoder.parameter_count

    @property
    def trainable_parameter_count(
        self,
    ) -> int:
        return self.encoder.trainable_parameter_count

    @property
    def dropout_active(
        self,
    ) -> bool:
        return self.encoder.dropout_active

    @property
    def diagnostics_enabled(
        self,
    ) -> bool:
        return self.diagnostics.enabled

    # -------------------------------------------------------------------------
    # Architecture and parameter provenance
    # -------------------------------------------------------------------------

    def architecture_provenance(
        self,
    ) -> MemoryArchitectureProvenance:
        """
        Return the selected cell encoder's canonical architecture provenance.
        """

        return self.encoder.architecture_provenance()

    def build_parameter_snapshot(
        self,
        *,
        checkpoint_id: str | None = None,
        checkpoint_fingerprint: str | None = None,
        training_step: int | None = None,
    ) -> MemoryParameterSnapshotProvenance:
        """
        Explicitly snapshot the selected cell encoder's current parameters.
        """

        return self.encoder.build_parameter_snapshot(
            checkpoint_id=checkpoint_id,
            checkpoint_fingerprint=(
                checkpoint_fingerprint
            ),
            training_step=training_step,
        )

    # -------------------------------------------------------------------------
    # Public execution
    # -------------------------------------------------------------------------

    def forward(
        self,
        source_history: HistoricalSequenceInputs,
        *,
        parameter_snapshot: (
            MemoryParameterSnapshotProvenance
            | None
        ) = None,
    ) -> TemporalSequenceEncoding:
        """Delegate public sequence encoding to the selected cell encoder."""

        return encode_recurrent_history(
            self.encoder,
            source_history,
            parameter_snapshot=parameter_snapshot,
        )

    def encode_with_state(
        self,
        source_history: HistoricalSequenceInputs,
        *,
        parameter_snapshot: (
            MemoryParameterSnapshotProvenance
            | None
        ) = None,
    ) -> RecurrentSequenceEncoderRun:
        """Delegate stateful execution to the selected cell encoder."""

        return encode_recurrent_history_with_state(
            self.encoder,
            source_history,
            parameter_snapshot=parameter_snapshot,
        )

    # -------------------------------------------------------------------------
    # Explicit diagnostics
    # -------------------------------------------------------------------------

    def diagnose_run(
        self,
        run: RecurrentSequenceEncoderRun,
        *,
        component_name: str | None = None,
    ) -> RecurrentRunDiagnostics | None:
        """
        Diagnose one run produced by this architecture.

        Architecture compatibility is checked by the diagnostics layer.
        """

        if not isinstance(
            run,
            RecurrentSequenceEncoderRun,
        ):
            raise TypeError(
                "run must be a RecurrentSequenceEncoderRun."
            )

        return self.diagnostics.diagnose(
            self,
            run,
            component_name=component_name,
        )

    def encode_with_diagnostics(
        self,
        source_history: HistoricalSequenceInputs,
        *,
        parameter_snapshot: (
            MemoryParameterSnapshotProvenance
            | None
        ) = None,
        component_name: str | None = None,
    ) -> tuple[
        RecurrentSequenceEncoderRun,
        RecurrentRunDiagnostics | None,
    ]:
        """
        Execute once with state, then build an optional detached report.
        """

        run = self.encode_with_state(
            source_history,
            parameter_snapshot=parameter_snapshot,
        )
        report = self.diagnose_run(
            run,
            component_name=component_name,
        )

        return (
            run,
            report,
        )

    def compare_execution_runs(
        self,
        packed_run: RecurrentSequenceEncoderRun,
        reference_run: RecurrentSequenceEncoderRun,
        *,
        rtol: float | None = None,
        atol: float | None = None,
    ) -> PackedReferenceComparisonDiagnostics | None:
        """Compare explicit packed and reference runs through diagnostics."""

        return self.diagnostics.compare(
            packed_run,
            reference_run,
            rtol=rtol,
            atol=atol,
        )


# =============================================================================
# Facade builders, predicates, and helpers
# =============================================================================


RecurrentEncoderLike: TypeAlias = (
    RecurrentSequenceEncoderModule
    | RecurrentMemoryEncoder
)

RECURRENT_ENCODER_LIKE_TYPES: Final[
    tuple[type[nn.Module], ...]
] = (
    GRUSequenceEncoder,
    LSTMSequenceEncoder,
    RecurrentMemoryEncoder,
)


def is_recurrent_encoder_like(
    module: object,
) -> bool:
    """Return whether ``module`` is a direct encoder or recurrent facade."""

    return isinstance(
        module,
        RECURRENT_ENCODER_LIKE_TYPES,
    )


def build_recurrent_memory_encoder(
    config: (
        RecurrentSequenceEncoderConfig
        | TemporalSequenceEncoderConfig
    ),
    *,
    diagnostics: RecurrentDiagnostics | None = None,
) -> RecurrentMemoryEncoder:
    """Construct the public Phase 6 recurrent encoder facade."""

    return RecurrentMemoryEncoder(
        config,
        diagnostics=diagnostics,
    )


def encode_recurrent_memory(
    encoder: RecurrentEncoderLike,
    source_history: HistoricalSequenceInputs,
    *,
    parameter_snapshot: (
        MemoryParameterSnapshotProvenance
        | None
    ) = None,
) -> TemporalSequenceEncoding:
    """Execute a direct recurrent encoder or the recurrent facade."""

    if isinstance(
        encoder,
        RecurrentMemoryEncoder,
    ):
        return encoder(
            source_history,
            parameter_snapshot=parameter_snapshot,
        )

    return encode_recurrent_history(
        encoder,
        source_history,
        parameter_snapshot=parameter_snapshot,
    )


def encode_recurrent_memory_with_state(
    encoder: RecurrentEncoderLike,
    source_history: HistoricalSequenceInputs,
    *,
    parameter_snapshot: (
        MemoryParameterSnapshotProvenance
        | None
    ) = None,
) -> RecurrentSequenceEncoderRun:
    """Execute stateful recurrence through a direct encoder or facade."""

    if isinstance(
        encoder,
        RecurrentMemoryEncoder,
    ):
        return encoder.encode_with_state(
            source_history,
            parameter_snapshot=parameter_snapshot,
        )

    return encode_recurrent_history_with_state(
        encoder,
        source_history,
        parameter_snapshot=parameter_snapshot,
    )


# =============================================================================
# Compact aliases
# =============================================================================


RecurrentEncoder = RecurrentMemoryEncoder
RecurrentSequenceEncoder = RecurrentMemoryEncoder

extract_recurrent_config = extract_recurrent_sequence_config

build_recurrent_encoder = build_recurrent_memory_encoder
build_sequence_encoder = build_recurrent_sequence_encoder

encode_history = encode_recurrent_memory
encode_history_with_state = encode_recurrent_memory_with_state


# =============================================================================
# Public API
# =============================================================================


__all__ = (
    # Identity and capabilities.
    "RECURRENT_MEMORY_ENCODER_IMPLEMENTATION_VERSION",
    "RECURRENT_MEMORY_ENCODER_COMPONENT_NAME",
    "RECURRENT_MEMORY_ENCODER_COMPONENT_KIND",
    "RECURRENT_MEMORY_ENCODER_OPERATION_NAME",
    "RECURRENT_MEMORY_ENCODER_DIAGNOSTICS_POLICY",
    "RECURRENT_MEMORY_ENCODER_PARAMETER_SNAPSHOT_POLICY",
    "RECURRENT_MEMORY_ENCODER_ARCHITECTURE_IDENTITY_POLICY",
    "RECURRENT_MEMORY_ENCODER_SCIENTIFIC_INTERPRETATION",
    "IMPLEMENTED_RECURRENT_CELL_KINDS",
    "IMPLEMENTED_RECURRENT_SEQUENCE_ENCODER_KINDS",

    # Direct module types and predicates.
    "RecurrentSequenceEncoderModule",
    "RECURRENT_SEQUENCE_ENCODER_TYPES",
    "is_recurrent_sequence_encoder",

    # Configuration and direct construction.
    "extract_recurrent_sequence_config",
    "build_recurrent_sequence_encoder",

    # Direct execution.
    "encode_recurrent_history",
    "encode_recurrent_history_with_state",

    # Public facade.
    "RecurrentMemoryEncoder",
    "RecurrentEncoderLike",
    "RECURRENT_ENCODER_LIKE_TYPES",
    "is_recurrent_encoder_like",
    "build_recurrent_memory_encoder",
    "encode_recurrent_memory",
    "encode_recurrent_memory_with_state",

    # Compact aliases.
    "RecurrentEncoder",
    "RecurrentSequenceEncoder",
    "extract_recurrent_config",
    "build_recurrent_encoder",
    "build_sequence_encoder",
    "encode_history",
    "encode_history_with_state",
)
