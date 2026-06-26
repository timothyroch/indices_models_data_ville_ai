"""
GRU temporal sequence encoder for Phase 6 urban memory.

Target path:
    urban_resilience_models/
        v2_hazard_conditioned_functional_ugnn/
            memory/
                recurrent_memory_encoder/
                    gru_encoder.py

The encoder preserves the shared public sequence contract while exposing final
GRU hidden states through ``RecurrentSequenceEncoderRun``.

Public API
----------
``forward(history, *, parameter_snapshot=None)``
    Return only ``TemporalSequenceEncoding``.

``encode_with_state(history, *, parameter_snapshot=None)``
    Return ``RecurrentSequenceEncoderRun`` containing the same public output,
    canonical final hidden state, state layout, and execution metadata.

Both methods delegate to one private ``_execute`` implementation.

Execution strategies
--------------------
Packed path
    1. Validate source/module dtype, device, feature width, and exact padding.
    2. Validate the optional explicit parameter snapshot.
    3. Build history-length and execution-order metadata.
    4. Canonicalize source left/right/no padding to right-padded valid prefixes.
    5. Apply the shared pointwise input adapter.
    6. Pack using the explicit stable metadata-defined order.
    7. Execute one ``nn.GRU`` with exact-zero initial state.
    8. Unpack, restore node order, restore source temporal layout, and scatter
       zero-history rows.

Reference path
    Each nonempty node is executed independently on its exact valid adapted
    subsequence. This is a correctness oracle, not naive dense padded execution.

All-zero history
    Compatibility and snapshot validation still run. Input adaptation, packing,
    and the GRU kernel are skipped. Sequence output and final hidden state are
    allocated as exact zeros without an artificial autograd anchor.

Frozen boundaries
----------------------------
- ``feature_observed_mask`` is not consumed as a model input.
- temporal coordinate values are not concatenated automatically.
- hazard information is not consumed.
- valid all-feature-missing timesteps remain real recurrent transitions.
- bidirectional execution is not an online-causal representation.
- caller-supplied, learned, streaming, and TBPTT initial states are postponed.
"""

from __future__ import annotations

from typing import Final

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
    MemoryParameterSnapshotProvenance,
)
from ..schemas.sequence_encoding import (
    TemporalSequenceEncoding,
)
from ._provenance import (
    build_parameter_snapshot_provenance,
    build_recurrent_architecture_provenance,
    build_recurrent_sequence_computation_provenance,
    count_module_parameters,
    validate_parameter_snapshot_provenance,
)
from .history_lengths import (
    build_recurrent_execution_metadata,
)
from .initial_state import (
    build_recurrent_state_layout,
    build_zero_canonical_final_states,
    build_zero_gru_initial_state,
    validate_recurrent_kernel_runtime,
)
from .input_adapter import (
    RecurrentInputAdapter,
    validate_recurrent_module_source_compatibility,
)
from .schemas import (
    RecurrentExecutionMetadata,
    RecurrentExecutionPath,
    RecurrentSequenceEncoderRun,
    RecurrentStateLayout,
)
from .sequence_packing import (
    canonicalize_recurrent_history,
    gather_canonical_node_sequence,
    pack_canonical_recurrent_batch,
    restore_and_scatter_recurrent_state,
    restore_recurrent_sequence_to_source,
    restore_single_node_sequence,
    scatter_nonempty_sequence_to_source,
    unpack_recurrent_sequence,
)


# =============================================================================
# Component identity and frozen policies
# =============================================================================


GRU_SEQUENCE_ENCODER_IMPLEMENTATION_VERSION: Final[str] = "0.1"

GRU_SEQUENCE_ENCODER_COMPONENT_NAME: Final[str] = (
    "gru_sequence_encoder"
)

GRU_SEQUENCE_ENCODER_COMPONENT_KIND: Final[str] = (
    "recurrent_temporal_sequence_encoder"
)

GRU_SEQUENCE_ENCODER_OPERATION_NAME: Final[str] = (
    "encode_temporal_history_with_gru"
)

GRU_SEQUENCE_ENCODER_ENCODING_NAME: Final[str] = (
    "gru_temporal_sequence_encoding"
)

GRU_SEQUENCE_ENCODER_RUN_NAME: Final[str] = (
    "gru_sequence_encoder_run"
)

GRU_SEQUENCE_ENCODER_INITIAL_STATE_POLICY: Final[str] = (
    "exact_zero_v1"
)

GRU_SEQUENCE_ENCODER_REFERENCE_POLICY: Final[str] = (
    "independent_exact_valid_subsequence_per_nonempty_node_v1"
)

GRU_SEQUENCE_ENCODER_ALL_ZERO_POLICY: Final[str] = (
    "validate_then_skip_adapter_packing_and_kernel_v1"
)

GRU_SEQUENCE_ENCODER_OUTPUT_DIRECTION_ORDER: Final[tuple[str, ...]] = (
    "forward",
    "backward",
)

GRU_SEQUENCE_ENCODER_FEATURE_OBSERVATION_MASK_CONSUMED: Final[bool] = False

GRU_SEQUENCE_ENCODER_TEMPORAL_COORDINATES_CONSUMED: Final[bool] = False

GRU_SEQUENCE_ENCODER_HAZARD_CONDITIONED: Final[bool] = False

GRU_SEQUENCE_ENCODER_SCIENTIFIC_INTERPRETATION: Final[str] = (
    "learned_recurrent_temporal_representation_not_causal_effect_estimate"
)


# =============================================================================
# Local validation
# =============================================================================


def _validate_gru_config(
    config: RecurrentSequenceEncoderConfig,
) -> None:
    if not isinstance(
        config,
        RecurrentSequenceEncoderConfig,
    ):
        raise TypeError(
            "config must be a RecurrentSequenceEncoderConfig."
        )

    if config.cell_kind != RecurrentCellKind.GRU:
        raise ValueError(
            "GRUSequenceEncoder requires config.cell_kind='gru'."
        )


def _validate_source_history(
    source_history: HistoricalSequenceInputs,
) -> None:
    if not isinstance(
        source_history,
        HistoricalSequenceInputs,
    ):
        raise TypeError(
            "source_history must be a HistoricalSequenceInputs."
        )


def _validate_raw_sequence(
    sequence: torch.Tensor,
    *,
    expected_node_count: int,
    expected_sequence_length: int,
    expected_output_dim: int,
    expected_device: torch.device,
    expected_dtype: torch.dtype,
    name: str,
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
            f"{name} must have shape [N, T, Hout]."
        )

    expected_shape = (
        expected_node_count,
        expected_sequence_length,
        expected_output_dim,
    )

    if tuple(
        sequence.shape
    ) != expected_shape:
        raise ValueError(
            f"{name} shape must equal {expected_shape}; observed "
            f"{tuple(sequence.shape)}."
        )

    if sequence.device != expected_device:
        raise ValueError(
            f"{name} device must match source history."
        )

    if sequence.dtype != expected_dtype:
        raise ValueError(
            f"{name} dtype must match source history."
        )

    if not bool(
        torch.isfinite(
            sequence
        ).all().item()
    ):
        raise ValueError(
            f"{name} must contain only finite values."
        )


def _validate_raw_hidden_state(
    hidden_state: torch.Tensor,
    *,
    layout: RecurrentStateLayout,
    expected_node_count: int,
    expected_device: torch.device,
    expected_dtype: torch.dtype,
    name: str,
) -> None:
    layout.validate_flat_state(
        hidden_state,
        name=name,
        node_count=expected_node_count,
    )

    if hidden_state.device != expected_device:
        raise ValueError(
            f"{name} device must match source history."
        )

    if hidden_state.dtype != expected_dtype:
        raise ValueError(
            f"{name} dtype must match source history."
        )


# =============================================================================
# GRU sequence encoder
# =============================================================================


class GRUSequenceEncoder(nn.Module):
    """
    Phase 6 GRU sequence encoder with packed and exact-reference execution.

    Parameters
    ----------
    config:
        Frozen recurrent encoder configuration with ``cell_kind='gru'``.

    Notes
    -----
    ``pack_sequences`` and ``enforce_sorted_lengths`` change execution lineage
    but not architecture identity. The architecture fingerprint is therefore
    stable across packed and reference instances with otherwise identical
    model configuration.
    """

    config: RecurrentSequenceEncoderConfig
    input_adapter: RecurrentInputAdapter
    gru: nn.GRU

    def __init__(
        self,
        config: RecurrentSequenceEncoderConfig,
    ) -> None:
        super().__init__()

        _validate_gru_config(
            config
        )

        self.config = config
        self.input_adapter = RecurrentInputAdapter(
            config
        )
        self.gru = nn.GRU(
            input_size=(
                self.input_adapter.effective_input_dim
            ),
            hidden_size=config.hidden_dim,
            num_layers=config.num_layers,
            bias=config.use_bias,
            batch_first=True,
            dropout=config.dropout,
            bidirectional=config.bidirectional,
        )

        validate_recurrent_kernel_runtime(
            self.gru,
            self.config,
        )

    # -------------------------------------------------------------------------
    # Structural properties
    # -------------------------------------------------------------------------

    @property
    def kernel(
        self,
    ) -> nn.GRU:
        return self.gru

    @property
    def cell_kind(
        self,
    ) -> RecurrentCellKind:
        return RecurrentCellKind.GRU

    @property
    def input_dim(
        self,
    ) -> int:
        return self.config.input_dim

    @property
    def recurrent_input_dim(
        self,
    ) -> int:
        return self.input_adapter.effective_input_dim

    @property
    def hidden_dim(
        self,
    ) -> int:
        return self.config.hidden_dim

    @property
    def output_dim(
        self,
    ) -> int:
        return self.config.output_dim

    @property
    def num_layers(
        self,
    ) -> int:
        return self.config.num_layers

    @property
    def num_directions(
        self,
    ) -> int:
        return (
            2
            if self.config.bidirectional
            else 1
        )

    @property
    def is_bidirectional(
        self,
    ) -> bool:
        return self.config.bidirectional

    @property
    def state_layout(
        self,
    ) -> RecurrentStateLayout:
        return build_recurrent_state_layout(
            self.config
        )

    @property
    def parameter_count(
        self,
    ) -> int:
        return count_module_parameters(
            self
        )[0]

    @property
    def trainable_parameter_count(
        self,
    ) -> int:
        return count_module_parameters(
            self
        )[1]

    @property
    def dropout_active(
        self,
    ) -> bool:
        return (
            self.training
            and self.config.num_layers > 1
            and self.config.dropout > 0.0
        )

    # -------------------------------------------------------------------------
    # Explicit architecture and parameter provenance
    # -------------------------------------------------------------------------

    def architecture_provenance(
        self,
    ) -> MemoryArchitectureProvenance:
        """Return architecture provenance without executing the encoder."""

        return build_recurrent_architecture_provenance(
            self.config
        )

    def build_parameter_snapshot(
        self,
        *,
        checkpoint_id: str | None = None,
        checkpoint_fingerprint: str | None = None,
        training_step: int | None = None,
    ) -> MemoryParameterSnapshotProvenance:
        """
        Explicitly capture the current adapter and GRU parameter state.

        Snapshots are never created automatically by ``forward``.
        """

        return build_parameter_snapshot_provenance(
            self,
            checkpoint_id=checkpoint_id,
            checkpoint_fingerprint=(
                checkpoint_fingerprint
            ),
            training_step=training_step,
        )

    # -------------------------------------------------------------------------
    # Public execution API
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
        """Encode history and return the shared public sequence contract."""

        return self._execute(
            source_history,
            parameter_snapshot=parameter_snapshot,
        ).public_output

    def encode_with_state(
        self,
        source_history: HistoricalSequenceInputs,
        *,
        parameter_snapshot: (
            MemoryParameterSnapshotProvenance
            | None
        ) = None,
    ) -> RecurrentSequenceEncoderRun:
        """Encode history and retain canonical final GRU hidden state."""

        return self._execute(
            source_history,
            parameter_snapshot=parameter_snapshot,
        )

    # -------------------------------------------------------------------------
    # One authoritative execution implementation
    # -------------------------------------------------------------------------

    def _execute(
        self,
        source_history: HistoricalSequenceInputs,
        *,
        parameter_snapshot: (
            MemoryParameterSnapshotProvenance
            | None
        ),
    ) -> RecurrentSequenceEncoderRun:
        _validate_source_history(
            source_history
        )

        # This full-module preflight is deliberately before the all-zero check.
        validate_recurrent_module_source_compatibility(
            self,
            source_history,
            expected_input_dim=self.config.input_dim,
            require_floating_parameter=True,
        )
        validate_recurrent_kernel_runtime(
            self.gru,
            self.config,
        )
        validate_parameter_snapshot_provenance(
            self,
            parameter_snapshot,
        )

        execution_metadata = (
            build_recurrent_execution_metadata(
                source_history,
                self.config,
            )
        )
        state_layout = self.state_layout

        if execution_metadata.all_zero_history:
            return self._execute_all_zero_history(
                source_history=source_history,
                execution_metadata=(
                    execution_metadata
                ),
                state_layout=state_layout,
                parameter_snapshot=(
                    parameter_snapshot
                ),
            )

        if (
            execution_metadata.execution_path
            == RecurrentExecutionPath.PACKED
        ):
            (
                encoded_sequence,
                final_hidden_state,
            ) = self._execute_packed(
                source_history=source_history,
                execution_metadata=(
                    execution_metadata
                ),
                state_layout=state_layout,
            )
        else:
            (
                encoded_sequence,
                final_hidden_state,
            ) = self._execute_reference(
                source_history=source_history,
                execution_metadata=(
                    execution_metadata
                ),
                state_layout=state_layout,
            )

        return self._build_run(
            source_history=source_history,
            execution_metadata=(
                execution_metadata
            ),
            state_layout=state_layout,
            encoded_sequence=encoded_sequence,
            final_hidden_state=(
                final_hidden_state
            ),
            parameter_snapshot=(
                parameter_snapshot
            ),
            adapter_executed=True,
            recurrent_kernel_executed=True,
            all_zero_history_short_circuit=False,
        )

    # -------------------------------------------------------------------------
    # Packed path
    # -------------------------------------------------------------------------

    def _execute_packed(
        self,
        *,
        source_history: HistoricalSequenceInputs,
        execution_metadata: RecurrentExecutionMetadata,
        state_layout: RecurrentStateLayout,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
    ]:
        canonical = canonicalize_recurrent_history(
            source_history,
            execution_metadata,
            value_stage="canonical_raw_history",
        )
        adapted = (
            self
            .input_adapter
            .adapt_canonical_batch(
                canonical
            )
        )
        packed_input = pack_canonical_recurrent_batch(
            adapted,
            execution_metadata,
        )
        initial_hidden = build_zero_gru_initial_state(
            self.gru,
            self.config,
            batch_size=(
                execution_metadata
                .nonempty_node_count
            ),
        )

        packed_output, execution_hidden = self.gru(
            packed_input,
            initial_hidden,
        )

        _validate_raw_hidden_state(
            execution_hidden,
            layout=state_layout,
            expected_node_count=(
                execution_metadata
                .nonempty_node_count
            ),
            expected_device=source_history.device,
            expected_dtype=source_history.dtype,
            name="packed_gru_final_hidden_state",
        )

        execution_sequence = unpack_recurrent_sequence(
            packed_output,
            execution_metadata,
            total_length=(
                source_history
                .sequence_length
            ),
        )
        _validate_raw_sequence(
            execution_sequence,
            expected_node_count=(
                execution_metadata
                .nonempty_node_count
            ),
            expected_sequence_length=(
                source_history
                .sequence_length
            ),
            expected_output_dim=(
                self.config.output_dim
            ),
            expected_device=source_history.device,
            expected_dtype=source_history.dtype,
            name="packed_gru_sequence_output",
        )

        encoded_sequence = (
            restore_recurrent_sequence_to_source(
                execution_sequence,
                source_history,
                execution_metadata,
            )
        )
        final_hidden_state = (
            restore_and_scatter_recurrent_state(
                execution_hidden,
                execution_metadata,
                state_layout,
                name=(
                    "packed_gru_final_hidden_state"
                ),
            )
        )

        return (
            encoded_sequence,
            final_hidden_state,
        )

    # -------------------------------------------------------------------------
    # Exact per-node reference path
    # -------------------------------------------------------------------------

    def _execute_reference(
        self,
        *,
        source_history: HistoricalSequenceInputs,
        execution_metadata: RecurrentExecutionMetadata,
        state_layout: RecurrentStateLayout,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
    ]:
        canonical = canonicalize_recurrent_history(
            source_history,
            execution_metadata,
            value_stage="canonical_raw_history",
        )
        adapted = (
            self
            .input_adapter
            .adapt_canonical_batch(
                canonical
            )
        )

        restored_rows: list[torch.Tensor] = []
        hidden_rows: list[torch.Tensor] = []

        for (
            nonempty_position,
            source_node_index,
        ) in enumerate(
            execution_metadata
            .nonempty_node_indices
            .tolist()
        ):
            node_sequence = (
                gather_canonical_node_sequence(
                    adapted,
                    nonempty_position,
                )
            )
            node_length = int(
                node_sequence.shape[1]
            )
            initial_hidden = (
                build_zero_gru_initial_state(
                    self.gru,
                    self.config,
                    batch_size=1,
                )
            )
            node_output, node_hidden = self.gru(
                node_sequence,
                initial_hidden,
            )

            _validate_raw_sequence(
                node_output,
                expected_node_count=1,
                expected_sequence_length=(
                    node_length
                ),
                expected_output_dim=(
                    self.config.output_dim
                ),
                expected_device=(
                    source_history.device
                ),
                expected_dtype=(
                    source_history.dtype
                ),
                name=(
                    "reference_gru_node_output"
                ),
            )
            _validate_raw_hidden_state(
                node_hidden,
                layout=state_layout,
                expected_node_count=1,
                expected_device=(
                    source_history.device
                ),
                expected_dtype=(
                    source_history.dtype
                ),
                name=(
                    "reference_gru_node_hidden_state"
                ),
            )

            restored_rows.append(
                restore_single_node_sequence(
                    node_output,
                    source_history,
                    source_node_index,
                )
            )
            hidden_rows.append(
                node_hidden
            )

        nonempty_source_sequence = torch.cat(
            restored_rows,
            dim=0,
        )
        execution_hidden = torch.cat(
            hidden_rows,
            dim=1,
        )

        _validate_raw_sequence(
            nonempty_source_sequence,
            expected_node_count=(
                execution_metadata
                .nonempty_node_count
            ),
            expected_sequence_length=(
                source_history
                .sequence_length
            ),
            expected_output_dim=(
                self.config.output_dim
            ),
            expected_device=source_history.device,
            expected_dtype=source_history.dtype,
            name=(
                "reference_gru_nonempty_source_sequence"
            ),
        )
        _validate_raw_hidden_state(
            execution_hidden,
            layout=state_layout,
            expected_node_count=(
                execution_metadata
                .nonempty_node_count
            ),
            expected_device=source_history.device,
            expected_dtype=source_history.dtype,
            name=(
                "reference_gru_nonempty_hidden_state"
            ),
        )

        encoded_sequence = (
            scatter_nonempty_sequence_to_source(
                nonempty_source_sequence,
                execution_metadata,
            )
        )
        final_hidden_state = (
            restore_and_scatter_recurrent_state(
                execution_hidden,
                execution_metadata,
                state_layout,
                name=(
                    "reference_gru_final_hidden_state"
                ),
            )
        )

        return (
            encoded_sequence,
            final_hidden_state,
        )

    # -------------------------------------------------------------------------
    # All-zero short circuit
    # -------------------------------------------------------------------------

    def _execute_all_zero_history(
        self,
        *,
        source_history: HistoricalSequenceInputs,
        execution_metadata: RecurrentExecutionMetadata,
        state_layout: RecurrentStateLayout,
        parameter_snapshot: (
            MemoryParameterSnapshotProvenance
            | None
        ),
    ) -> RecurrentSequenceEncoderRun:
        encoded_sequence = (
            source_history
            .history
            .new_zeros(
                (
                    source_history.node_count,
                    source_history.sequence_length,
                    self.config.output_dim,
                )
            )
        )
        (
            final_hidden_state,
            final_cell_state,
        ) = build_zero_canonical_final_states(
            self.gru,
            self.config,
            node_count=(
                source_history.node_count
            ),
        )

        if final_cell_state is not None:
            raise RuntimeError(
                "GRU all-zero state construction must not produce a cell state."
            )

        state_layout.validate_canonical_state(
            final_hidden_state,
            name=(
                "all_zero_gru_final_hidden_state"
            ),
            node_count=(
                source_history.node_count
            ),
        )

        return self._build_run(
            source_history=source_history,
            execution_metadata=(
                execution_metadata
            ),
            state_layout=state_layout,
            encoded_sequence=encoded_sequence,
            final_hidden_state=(
                final_hidden_state
            ),
            parameter_snapshot=(
                parameter_snapshot
            ),
            adapter_executed=False,
            recurrent_kernel_executed=False,
            all_zero_history_short_circuit=True,
        )

    # -------------------------------------------------------------------------
    # Shared public schema and provenance assembly
    # -------------------------------------------------------------------------

    def _build_run(
        self,
        *,
        source_history: HistoricalSequenceInputs,
        execution_metadata: RecurrentExecutionMetadata,
        state_layout: RecurrentStateLayout,
        encoded_sequence: torch.Tensor,
        final_hidden_state: torch.Tensor,
        parameter_snapshot: (
            MemoryParameterSnapshotProvenance
            | None
        ),
        adapter_executed: bool,
        recurrent_kernel_executed: bool,
        all_zero_history_short_circuit: bool,
    ) -> RecurrentSequenceEncoderRun:
        _validate_raw_sequence(
            encoded_sequence,
            expected_node_count=(
                source_history.node_count
            ),
            expected_sequence_length=(
                source_history.sequence_length
            ),
            expected_output_dim=(
                self.config.output_dim
            ),
            expected_device=source_history.device,
            expected_dtype=source_history.dtype,
            name="gru_encoded_sequence",
        )
        state_layout.validate_canonical_state(
            final_hidden_state,
            name="gru_final_hidden_state",
            node_count=(
                source_history.node_count
            ),
        )

        if final_hidden_state.device != source_history.device:
            raise ValueError(
                "GRU final hidden-state device must match source history."
            )

        if final_hidden_state.dtype != source_history.dtype:
            raise ValueError(
                "GRU final hidden-state dtype must match source history."
            )

        computation_provenance = (
            build_recurrent_sequence_computation_provenance(
                source_history=source_history,
                config=self.config,
                execution_path=(
                    execution_metadata
                    .execution_path
                ),
                sort_was_applied=(
                    execution_metadata
                    .sort_was_applied
                ),
                module_training=self.training,
                nonempty_node_count=(
                    execution_metadata
                    .nonempty_node_count
                ),
                zero_history_count=(
                    execution_metadata
                    .zero_history_count
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
                parameter_snapshot=(
                    parameter_snapshot
                ),
                extra_lineage_metadata={
                    "encoder_implementation_version": (
                        GRU_SEQUENCE_ENCODER_IMPLEMENTATION_VERSION
                    ),
                    "initial_state_policy": (
                        GRU_SEQUENCE_ENCODER_INITIAL_STATE_POLICY
                    ),
                    "reference_execution_policy": (
                        GRU_SEQUENCE_ENCODER_REFERENCE_POLICY
                    ),
                    "all_zero_history_policy": (
                        GRU_SEQUENCE_ENCODER_ALL_ZERO_POLICY
                    ),
                },
                operation_name=(
                    GRU_SEQUENCE_ENCODER_OPERATION_NAME
                ),
            )
        )

        public_output = TemporalSequenceEncoding(
            encoded_sequence=(
                encoded_sequence
            ),
            source_history=source_history,
            encoder_kind=(
                self.config.schema_encoder_kind
            ),
            computation_provenance=(
                computation_provenance
            ),
            encoding_name=(
                GRU_SEQUENCE_ENCODER_ENCODING_NAME
            ),
        )

        return RecurrentSequenceEncoderRun(
            public_output=public_output,
            final_hidden_state=(
                final_hidden_state
            ),
            final_cell_state=None,
            state_layout=state_layout,
            execution_metadata=(
                execution_metadata
            ),
            run_name=(
                GRU_SEQUENCE_ENCODER_RUN_NAME
            ),
        )


# =============================================================================
# Builders and compact aliases
# =============================================================================


def build_gru_sequence_encoder(
    config: RecurrentSequenceEncoderConfig,
) -> GRUSequenceEncoder:
    """Construct a Phase 6 GRU sequence encoder."""

    return GRUSequenceEncoder(
        config
    )


def build_gru_parameter_snapshot(
    encoder: GRUSequenceEncoder,
    *,
    checkpoint_id: str | None = None,
    checkpoint_fingerprint: str | None = None,
    training_step: int | None = None,
) -> MemoryParameterSnapshotProvenance:
    """Explicit functional wrapper for GRU parameter snapshot creation."""

    if not isinstance(
        encoder,
        GRUSequenceEncoder,
    ):
        raise TypeError(
            "encoder must be a GRUSequenceEncoder."
        )

    return encoder.build_parameter_snapshot(
        checkpoint_id=checkpoint_id,
        checkpoint_fingerprint=(
            checkpoint_fingerprint
        ),
        training_step=training_step,
    )


GRUEncoder = GRUSequenceEncoder
build_gru_encoder = build_gru_sequence_encoder
snapshot_gru_parameters = build_gru_parameter_snapshot


# =============================================================================
# Public API
# =============================================================================


__all__ = (
    # Component identity and frozen policies.
    "GRU_SEQUENCE_ENCODER_IMPLEMENTATION_VERSION",
    "GRU_SEQUENCE_ENCODER_COMPONENT_NAME",
    "GRU_SEQUENCE_ENCODER_COMPONENT_KIND",
    "GRU_SEQUENCE_ENCODER_OPERATION_NAME",
    "GRU_SEQUENCE_ENCODER_ENCODING_NAME",
    "GRU_SEQUENCE_ENCODER_RUN_NAME",
    "GRU_SEQUENCE_ENCODER_INITIAL_STATE_POLICY",
    "GRU_SEQUENCE_ENCODER_REFERENCE_POLICY",
    "GRU_SEQUENCE_ENCODER_ALL_ZERO_POLICY",
    "GRU_SEQUENCE_ENCODER_OUTPUT_DIRECTION_ORDER",
    "GRU_SEQUENCE_ENCODER_FEATURE_OBSERVATION_MASK_CONSUMED",
    "GRU_SEQUENCE_ENCODER_TEMPORAL_COORDINATES_CONSUMED",
    "GRU_SEQUENCE_ENCODER_HAZARD_CONDITIONED",
    "GRU_SEQUENCE_ENCODER_SCIENTIFIC_INTERPRETATION",

    # Encoder and builders.
    "GRUSequenceEncoder",
    "build_gru_sequence_encoder",
    "build_gru_parameter_snapshot",

    # Compact aliases.
    "GRUEncoder",
    "build_gru_encoder",
    "snapshot_gru_parameters",
)
