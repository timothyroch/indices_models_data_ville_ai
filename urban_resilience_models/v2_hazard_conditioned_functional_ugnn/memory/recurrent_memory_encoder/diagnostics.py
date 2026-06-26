"""
Detached diagnostics for Phase 6 recurrent temporal encoders.

Target path:
    urban_resilience_models/
        v2_hazard_conditioned_functional_ugnn/
            memory/
                recurrent_memory_encoder/
                    diagnostics.py

The diagnostics in this module are descriptive audit artifacts. They do not
alter recurrent execution, gradients, parameters, masks, provenance, or model
state. They make no causal, calibrated-risk, feature-importance, or mechanistic
interpretation claims.

The module supports two complementary artifacts:

``RecurrentRunDiagnostics``
    Numerical, structural, execution, missingness, and provenance diagnostics
    for one validated ``RecurrentSequenceEncoderRun``.

``PackedReferenceComparisonDiagnostics``
    Detached comparison of a packed run and an exact per-node reference run.

Reported run diagnostics include:

- parameter counts and module mode;
- packed/reference execution path and sorting behavior;
- history-length and zero-history statistics;
- valid all-feature-missing timestep counts when observation metadata exists;
- output, hidden-state, and optional cell-state finite counts;
- exact-zero padded-output and zero-history leakage checks;
- valid output and final-state norm summaries;
- descriptive hidden-state near-boundary counts;
- architecture, execution, source, state-layout, and parameter-snapshot
  fingerprints.

Packed/reference comparisons include:

- architecture, source, state-layout, and snapshot compatibility;
- output, hidden-state, and optional cell-state error summaries;
- valid/padded/zero-history error summaries;
- dtype-aware allclose decisions;
- an explicit flag recording whether dropout makes equality an inappropriate
  expectation.

Parameter values are not fingerprinted automatically by diagnostics. Parameter
snapshot hashing remains an explicit operation in ``_provenance.py``.
"""

from __future__ import annotations

from dataclasses import asdict
from dataclasses import dataclass
from hashlib import sha256
import json
import math
from typing import Any, Final, Self

import torch
from torch import nn

from ..schemas.sequence_encoding import (
    TemporalSequenceEncoderKind,
)
from ._provenance import (
    count_module_parameters,
)
from .schemas import (
    RecurrentExecutionPath,
    RecurrentSequenceEncoderRun,
)


# =============================================================================
# Diagnostic identity and frozen interpretation
# =============================================================================


RECURRENT_DIAGNOSTICS_SCHEMA_VERSION: Final[str] = "0.1"

RECURRENT_DIAGNOSTICS_COMPONENT_NAME: Final[str] = (
    "recurrent_encoder_diagnostics"
)

RECURRENT_DIAGNOSTICS_COMPONENT_KIND: Final[str] = (
    "detached_recurrent_execution_audit"
)

RECURRENT_DIAGNOSTICS_OPERATION_NAME: Final[str] = (
    "diagnose_recurrent_sequence_encoder_run"
)

RECURRENT_PACKED_REFERENCE_COMPARISON_OPERATION_NAME: Final[str] = (
    "compare_packed_and_reference_recurrent_runs"
)

RECURRENT_DIAGNOSTICS_DEFAULT_HIDDEN_BOUNDARY_THRESHOLD: Final[float] = 0.99

RECURRENT_DIAGNOSTICS_SCIENTIFIC_INTERPRETATION: Final[str] = (
    "descriptive_numerical_and_structural_audit_not_causal_explanation"
)

RECURRENT_PACKED_REFERENCE_COMPARISON_INTERPRETATION: Final[str] = (
    "numerical_execution_path_comparison_not_model_validity_claim"
)


# =============================================================================
# Scalar and tensor helpers
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
            f"{name} must be a boolean."
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


def _require_finite_float(
    name: str,
    value: float,
) -> None:
    if (
        isinstance(
            value,
            bool,
        )
        or not isinstance(
            value,
            (
                int,
                float,
            ),
        )
    ):
        raise TypeError(
            f"{name} must be numeric."
        )

    if not math.isfinite(
        float(
            value
        )
    ):
        raise ValueError(
            f"{name} must be finite."
        )


def _require_nonnegative_finite_float(
    name: str,
    value: float,
) -> None:
    _require_finite_float(
        name,
        value,
    )

    if float(
        value
    ) < 0.0:
        raise ValueError(
            f"{name} must be nonnegative."
        )


def _require_optional_nonnegative_finite_float(
    name: str,
    value: float | None,
) -> None:
    if value is None:
        return

    _require_nonnegative_finite_float(
        name,
        value,
    )


def _require_fraction(
    name: str,
    value: float,
) -> None:
    _require_nonnegative_finite_float(
        name,
        value,
    )

    if float(
        value
    ) > 1.0:
        raise ValueError(
            f"{name} cannot exceed one."
        )


def _require_optional_fraction(
    name: str,
    value: float | None,
) -> None:
    if value is None:
        return

    _require_fraction(
        name,
        value,
    )


def _require_positive_tolerance(
    name: str,
    value: float,
    *,
    allow_zero: bool,
) -> None:
    _require_nonnegative_finite_float(
        name,
        value,
    )

    if (
        not allow_zero
        and float(
            value
        )
        == 0.0
    ):
        raise ValueError(
            f"{name} must be strictly positive."
        )


def _tensor_nonfinite_count(
    value: torch.Tensor,
) -> int:
    return int(
        (
            ~torch.isfinite(
                value
            )
        )
        .sum()
        .item()
    )


def _maximum_absolute(
    value: torch.Tensor,
) -> float:
    if value.numel() == 0:
        return 0.0

    finite = value[
        torch.isfinite(
            value
        )
    ]

    if finite.numel() == 0:
        return 0.0

    return float(
        finite
        .abs()
        .max()
        .item()
    )


def _mean_absolute(
    value: torch.Tensor,
) -> float:
    if value.numel() == 0:
        return 0.0

    finite = value[
        torch.isfinite(
            value
        )
    ]

    if finite.numel() == 0:
        return 0.0

    return float(
        finite
        .abs()
        .mean()
        .item()
    )


def _root_mean_square(
    value: torch.Tensor,
) -> float:
    if value.numel() == 0:
        return 0.0

    finite = value[
        torch.isfinite(
            value
        )
    ]

    if finite.numel() == 0:
        return 0.0

    return float(
        torch.sqrt(
            finite
            .to(
                dtype=torch.float64
            )
            .square()
            .mean()
        )
        .item()
    )


def _mean_and_max_vector_norm(
    value: torch.Tensor,
) -> tuple[
    float,
    float,
]:
    if value.numel() == 0:
        return (
            0.0,
            0.0,
        )

    if value.ndim < 1:
        raise ValueError(
            "Vector-norm diagnostics require at least one dimension."
        )

    finite_rows = torch.isfinite(
        value
    ).all(
        dim=-1
    )
    finite_value = value[
        finite_rows
    ]

    if finite_value.numel() == 0:
        return (
            0.0,
            0.0,
        )

    norms = torch.linalg.vector_norm(
        finite_value,
        ord=2,
        dim=-1,
    )

    return (
        float(
            norms
            .to(
                dtype=torch.float64
            )
            .mean()
            .item()
        ),
        float(
            norms
            .max()
            .item()
        ),
    )


def _difference_metrics(
    first: torch.Tensor,
    second: torch.Tensor,
) -> tuple[
    int,
    float,
    float,
    float,
]:
    if first.shape != second.shape:
        raise ValueError(
            "Comparison tensors must have identical shapes."
        )

    difference = (
        first.detach()
        - second.detach()
    )
    nonfinite_count = _tensor_nonfinite_count(
        difference
    )

    return (
        nonfinite_count,
        _maximum_absolute(
            difference
        ),
        _mean_absolute(
            difference
        ),
        _root_mean_square(
            difference
        ),
    )


def _default_tolerances(
    dtype: torch.dtype,
) -> tuple[
    float,
    float,
]:
    if dtype == torch.float64:
        return (
            1e-7,
            1e-9,
        )

    if dtype == torch.float32:
        return (
            1e-5,
            1e-6,
        )

    return (
        1e-3,
        1e-4,
    )


def _lineage_metadata(
    run: RecurrentSequenceEncoderRun,
) -> dict[str, Any]:
    metadata = (
        run
        .public_output
        .computation_provenance
        .lineage
        .lineage_metadata
    )

    if not isinstance(
        metadata,
        dict,
    ):
        return dict(
            metadata
        )

    return dict(
        metadata
    )


def _lineage_boolean(
    metadata: dict[str, Any],
    name: str,
    *,
    default: bool,
) -> bool:
    value = metadata.get(
        name,
        default,
    )

    if not isinstance(
        value,
        bool,
    ):
        raise ValueError(
            f"Execution lineage field {name!r} must be boolean."
        )

    return value


def _lineage_float(
    metadata: dict[str, Any],
    name: str,
    *,
    default: float,
) -> float:
    value = metadata.get(
        name,
        default,
    )

    if (
        isinstance(
            value,
            bool,
        )
        or not isinstance(
            value,
            (
                int,
                float,
            ),
        )
    ):
        raise ValueError(
            f"Execution lineage field {name!r} must be numeric."
        )

    converted = float(
        value
    )

    if not math.isfinite(
        converted
    ):
        raise ValueError(
            f"Execution lineage field {name!r} must be finite."
        )

    return converted


# =============================================================================
# Deterministic serialization
# =============================================================================


class _DiagnosticsMixin:
    """Tensor-free deterministic serialization and fingerprinting."""

    def to_dict(
        self,
    ) -> dict[str, Any]:
        return asdict(
            self
        )

    def to_json(
        self,
        *,
        indent: int | None = 2,
    ) -> str:
        return json.dumps(
            self.to_dict(),
            indent=indent,
            sort_keys=True,
            ensure_ascii=False,
            allow_nan=False,
        )

    def fingerprint(
        self,
    ) -> str:
        canonical = json.dumps(
            self.to_dict(),
            sort_keys=True,
            separators=(
                ",",
                ":",
            ),
            ensure_ascii=False,
            allow_nan=False,
        )

        return sha256(
            canonical.encode(
                "utf-8"
            )
        ).hexdigest()


# =============================================================================
# Run diagnostics schema
# =============================================================================


@dataclass(
    frozen=True,
    slots=True,
)
class RecurrentRunDiagnostics(
    _DiagnosticsMixin
):
    """Immutable diagnostics for one recurrent encoder run."""

    component_name: str
    encoder_kind: str
    execution_path: str

    node_count: int
    sequence_length: int
    input_dim: int
    output_dim: int
    hidden_dim: int
    num_layers: int
    num_directions: int
    bidirectional: bool
    has_cell_state: bool

    parameter_count: int
    trainable_parameter_count: int
    module_training: bool
    dropout_probability: float
    dropout_active: bool

    original_padding_direction: str
    canonical_padding_direction: str
    sort_was_applied: bool
    identity_permutation: bool
    all_zero_history_short_circuit: bool
    adapter_executed: bool
    recurrent_kernel_executed: bool

    zero_history_count: int
    nonempty_history_count: int
    valid_timestep_count: int
    padded_timestep_count: int

    valid_length_min: int
    valid_length_mean: float
    valid_length_max: int

    feature_observed_mask_available: bool
    all_features_missing_valid_timestep_count: int | None
    all_features_missing_valid_timestep_fraction: float | None

    nonfinite_output_count: int
    nonfinite_hidden_state_count: int
    nonfinite_cell_state_count: int | None

    max_abs_padded_output: float
    max_abs_zero_history_output: float
    max_abs_zero_history_hidden_state: float
    max_abs_zero_history_cell_state: float | None

    mean_valid_output_vector_l2_norm: float
    max_valid_output_vector_l2_norm: float
    mean_final_hidden_vector_l2_norm: float
    max_final_hidden_vector_l2_norm: float
    mean_final_cell_vector_l2_norm: float | None
    max_final_cell_vector_l2_norm: float | None

    hidden_boundary_abs_threshold: float
    hidden_abs_ge_boundary_count: int
    hidden_abs_ge_boundary_fraction: float

    exact_zero_padding: bool
    exact_zero_history_output: bool
    exact_zero_history_hidden_state: bool
    exact_zero_history_cell_state: bool | None

    output_requires_grad: bool
    hidden_state_requires_grad: bool
    cell_state_requires_grad: bool | None

    architecture_fingerprint: str
    computation_lineage_fingerprint: str
    source_lineage_fingerprint: str
    run_lineage_fingerprint: str
    state_layout_fingerprint: str
    execution_metadata_fingerprint: str
    parameter_snapshot_fingerprint: str | None

    schema_version: str = RECURRENT_DIAGNOSTICS_SCHEMA_VERSION

    def __post_init__(
        self,
    ) -> None:
        for name, value in (
            (
                "component_name",
                self.component_name,
            ),
            (
                "encoder_kind",
                self.encoder_kind,
            ),
            (
                "execution_path",
                self.execution_path,
            ),
            (
                "original_padding_direction",
                self.original_padding_direction,
            ),
            (
                "canonical_padding_direction",
                self.canonical_padding_direction,
            ),
            (
                "architecture_fingerprint",
                self.architecture_fingerprint,
            ),
            (
                "computation_lineage_fingerprint",
                self.computation_lineage_fingerprint,
            ),
            (
                "source_lineage_fingerprint",
                self.source_lineage_fingerprint,
            ),
            (
                "run_lineage_fingerprint",
                self.run_lineage_fingerprint,
            ),
            (
                "state_layout_fingerprint",
                self.state_layout_fingerprint,
            ),
            (
                "execution_metadata_fingerprint",
                self.execution_metadata_fingerprint,
            ),
            (
                "schema_version",
                self.schema_version,
            ),
        ):
            _require_nonempty_string(
                name,
                value,
            )

        if self.encoder_kind not in (
            TemporalSequenceEncoderKind.GRU.value,
            TemporalSequenceEncoderKind.LSTM.value,
        ):
            raise ValueError(
                "encoder_kind must be 'gru' or 'lstm'."
            )

        if self.execution_path not in (
            RecurrentExecutionPath.PACKED.value,
            RecurrentExecutionPath.REFERENCE.value,
        ):
            raise ValueError(
                "execution_path must be 'packed' or 'reference'."
            )

        for name, value in (
            (
                "node_count",
                self.node_count,
            ),
            (
                "sequence_length",
                self.sequence_length,
            ),
            (
                "input_dim",
                self.input_dim,
            ),
            (
                "output_dim",
                self.output_dim,
            ),
            (
                "hidden_dim",
                self.hidden_dim,
            ),
            (
                "num_layers",
                self.num_layers,
            ),
            (
                "num_directions",
                self.num_directions,
            ),
            (
                "parameter_count",
                self.parameter_count,
            ),
            (
                "trainable_parameter_count",
                self.trainable_parameter_count,
            ),
            (
                "zero_history_count",
                self.zero_history_count,
            ),
            (
                "nonempty_history_count",
                self.nonempty_history_count,
            ),
            (
                "valid_timestep_count",
                self.valid_timestep_count,
            ),
            (
                "padded_timestep_count",
                self.padded_timestep_count,
            ),
            (
                "valid_length_min",
                self.valid_length_min,
            ),
            (
                "valid_length_max",
                self.valid_length_max,
            ),
            (
                "nonfinite_output_count",
                self.nonfinite_output_count,
            ),
            (
                "nonfinite_hidden_state_count",
                self.nonfinite_hidden_state_count,
            ),
            (
                "hidden_abs_ge_boundary_count",
                self.hidden_abs_ge_boundary_count,
            ),
        ):
            _require_nonnegative_int(
                name,
                value,
            )

        _require_optional_nonnegative_int(
            "all_features_missing_valid_timestep_count",
            self.all_features_missing_valid_timestep_count,
        )
        _require_optional_nonnegative_int(
            "nonfinite_cell_state_count",
            self.nonfinite_cell_state_count,
        )

        for name, value in (
            (
                "bidirectional",
                self.bidirectional,
            ),
            (
                "has_cell_state",
                self.has_cell_state,
            ),
            (
                "module_training",
                self.module_training,
            ),
            (
                "dropout_active",
                self.dropout_active,
            ),
            (
                "sort_was_applied",
                self.sort_was_applied,
            ),
            (
                "identity_permutation",
                self.identity_permutation,
            ),
            (
                "all_zero_history_short_circuit",
                self.all_zero_history_short_circuit,
            ),
            (
                "adapter_executed",
                self.adapter_executed,
            ),
            (
                "recurrent_kernel_executed",
                self.recurrent_kernel_executed,
            ),
            (
                "feature_observed_mask_available",
                self.feature_observed_mask_available,
            ),
            (
                "exact_zero_padding",
                self.exact_zero_padding,
            ),
            (
                "exact_zero_history_output",
                self.exact_zero_history_output,
            ),
            (
                "exact_zero_history_hidden_state",
                self.exact_zero_history_hidden_state,
            ),
            (
                "output_requires_grad",
                self.output_requires_grad,
            ),
            (
                "hidden_state_requires_grad",
                self.hidden_state_requires_grad,
            ),
        ):
            _require_boolean(
                name,
                value,
            )

        if self.exact_zero_history_cell_state is not None:
            _require_boolean(
                "exact_zero_history_cell_state",
                self.exact_zero_history_cell_state,
            )

        if self.cell_state_requires_grad is not None:
            _require_boolean(
                "cell_state_requires_grad",
                self.cell_state_requires_grad,
            )

        for name, value in (
            (
                "dropout_probability",
                self.dropout_probability,
            ),
            (
                "valid_length_mean",
                self.valid_length_mean,
            ),
            (
                "max_abs_padded_output",
                self.max_abs_padded_output,
            ),
            (
                "max_abs_zero_history_output",
                self.max_abs_zero_history_output,
            ),
            (
                "max_abs_zero_history_hidden_state",
                self.max_abs_zero_history_hidden_state,
            ),
            (
                "mean_valid_output_vector_l2_norm",
                self.mean_valid_output_vector_l2_norm,
            ),
            (
                "max_valid_output_vector_l2_norm",
                self.max_valid_output_vector_l2_norm,
            ),
            (
                "mean_final_hidden_vector_l2_norm",
                self.mean_final_hidden_vector_l2_norm,
            ),
            (
                "max_final_hidden_vector_l2_norm",
                self.max_final_hidden_vector_l2_norm,
            ),
            (
                "hidden_boundary_abs_threshold",
                self.hidden_boundary_abs_threshold,
            ),
        ):
            _require_nonnegative_finite_float(
                name,
                value,
            )

        _require_optional_nonnegative_finite_float(
            "all_features_missing_valid_timestep_fraction",
            self.all_features_missing_valid_timestep_fraction,
        )
        _require_optional_nonnegative_finite_float(
            "max_abs_zero_history_cell_state",
            self.max_abs_zero_history_cell_state,
        )
        _require_optional_nonnegative_finite_float(
            "mean_final_cell_vector_l2_norm",
            self.mean_final_cell_vector_l2_norm,
        )
        _require_optional_nonnegative_finite_float(
            "max_final_cell_vector_l2_norm",
            self.max_final_cell_vector_l2_norm,
        )
        _require_fraction(
            "hidden_abs_ge_boundary_fraction",
            self.hidden_abs_ge_boundary_fraction,
        )
        _require_optional_fraction(
            "all_features_missing_valid_timestep_fraction",
            self.all_features_missing_valid_timestep_fraction,
        )

        if self.node_count <= 0:
            raise ValueError(
                "node_count must be strictly positive."
            )

        if (
            self.sequence_length <= 0
            or self.input_dim <= 0
            or self.output_dim <= 0
            or self.hidden_dim <= 0
            or self.num_layers <= 0
            or self.num_directions <= 0
        ):
            raise ValueError(
                "Sequence and recurrent dimensions must be strictly positive."
            )

        if self.trainable_parameter_count > self.parameter_count:
            raise ValueError(
                "trainable_parameter_count cannot exceed parameter_count."
            )

        if (
            self.zero_history_count
            + self.nonempty_history_count
            != self.node_count
        ):
            raise ValueError(
                "zero_history_count + nonempty_history_count must equal "
                "node_count."
            )

        if (
            self.valid_timestep_count
            + self.padded_timestep_count
            != self.node_count
            * self.sequence_length
        ):
            raise ValueError(
                "Valid and padded timestep counts must cover [N,T]."
            )

        expected_directions = (
            2
            if self.bidirectional
            else 1
        )

        if self.num_directions != expected_directions:
            raise ValueError(
                "num_directions must agree with bidirectional."
            )

        if self.output_dim != (
            self.hidden_dim
            * self.num_directions
        ):
            raise ValueError(
                "output_dim must equal hidden_dim * num_directions."
            )

        if self.has_cell_state:
            if self.encoder_kind != TemporalSequenceEncoderKind.LSTM.value:
                raise ValueError(
                    "Only LSTM diagnostics may report a cell state."
                )

            required_cell_values = (
                self.nonfinite_cell_state_count,
                self.max_abs_zero_history_cell_state,
                self.mean_final_cell_vector_l2_norm,
                self.max_final_cell_vector_l2_norm,
                self.exact_zero_history_cell_state,
                self.cell_state_requires_grad,
            )

            if any(
                value is None
                for value in required_cell_values
            ):
                raise ValueError(
                    "LSTM diagnostics require all cell-state fields."
                )
        else:
            if self.encoder_kind != TemporalSequenceEncoderKind.GRU.value:
                raise ValueError(
                    "LSTM diagnostics must report a cell state."
                )

            optional_cell_values = (
                self.nonfinite_cell_state_count,
                self.max_abs_zero_history_cell_state,
                self.mean_final_cell_vector_l2_norm,
                self.max_final_cell_vector_l2_norm,
                self.exact_zero_history_cell_state,
                self.cell_state_requires_grad,
            )

            if any(
                value is not None
                for value in optional_cell_values
            ):
                raise ValueError(
                    "GRU diagnostics must use None for cell-state fields."
                )

        if self.feature_observed_mask_available:
            if (
                self.all_features_missing_valid_timestep_count is None
                or self.all_features_missing_valid_timestep_fraction is None
            ):
                raise ValueError(
                    "Available observation diagnostics require count and "
                    "fraction."
                )
        elif (
            self.all_features_missing_valid_timestep_count is not None
            or self.all_features_missing_valid_timestep_fraction is not None
        ):
            raise ValueError(
                "Unavailable observation diagnostics must use None values."
            )

        if self.execution_path == RecurrentExecutionPath.REFERENCE.value:
            if self.sort_was_applied:
                raise ValueError(
                    "Reference execution cannot report sorting."
                )

            if not self.identity_permutation:
                raise ValueError(
                    "Reference execution must use identity permutations."
                )

        if self.all_zero_history_short_circuit:
            if self.nonempty_history_count != 0:
                raise ValueError(
                    "All-zero short circuit requires no nonempty histories."
                )

            if self.adapter_executed or self.recurrent_kernel_executed:
                raise ValueError(
                    "All-zero short circuit must skip adapter and kernel."
                )

        if self.hidden_boundary_abs_threshold <= 0.0:
            raise ValueError(
                "hidden_boundary_abs_threshold must be strictly positive."
            )

        if self.parameter_snapshot_fingerprint is not None:
            _require_nonempty_string(
                "parameter_snapshot_fingerprint",
                self.parameter_snapshot_fingerprint,
            )

    @property
    def is_numerically_clean(
        self,
    ) -> bool:
        return (
            self.nonfinite_output_count == 0
            and self.nonfinite_hidden_state_count == 0
            and (
                self.nonfinite_cell_state_count
                in (
                    None,
                    0,
                )
            )
            and self.exact_zero_padding
            and self.exact_zero_history_output
            and self.exact_zero_history_hidden_state
            and (
                self.exact_zero_history_cell_state
                in (
                    None,
                    True,
                )
            )
        )

    @property
    def execution_is_structurally_consistent(
        self,
    ) -> bool:
        if self.all_zero_history_short_circuit:
            return (
                not self.adapter_executed
                and not self.recurrent_kernel_executed
            )

        return (
            self.adapter_executed
            and self.recurrent_kernel_executed
        )


# =============================================================================
# Packed/reference comparison schema
# =============================================================================


@dataclass(
    frozen=True,
    slots=True,
)
class PackedReferenceComparisonDiagnostics(
    _DiagnosticsMixin
):
    """Detached comparison of packed and exact-reference recurrent runs."""

    encoder_kind: str

    node_count: int
    sequence_length: int
    output_dim: int
    hidden_dim: int
    num_layers: int
    num_directions: int
    has_cell_state: bool

    rtol: float
    atol: float

    packed_execution_path: str
    reference_execution_path: str

    architecture_match: bool
    source_object_identity: bool
    source_alignment_match: bool
    source_values_match: bool
    source_masks_match: bool
    source_lineage_match: bool
    state_layout_match: bool

    parameter_snapshot_both_absent: bool
    parameter_snapshot_match: bool
    parameter_identity_verified: bool

    dropout_active_in_either: bool
    equality_expectation_valid: bool

    output_difference_nonfinite_count: int
    output_max_abs_error: float
    output_mean_abs_error: float
    output_rmse: float
    output_allclose: bool

    valid_output_max_abs_error: float
    padded_output_max_abs_error: float
    zero_history_output_max_abs_error: float

    hidden_difference_nonfinite_count: int
    hidden_max_abs_error: float
    hidden_mean_abs_error: float
    hidden_rmse: float
    hidden_allclose: bool
    zero_history_hidden_max_abs_error: float

    cell_comparison_available: bool
    cell_difference_nonfinite_count: int | None
    cell_max_abs_error: float | None
    cell_mean_abs_error: float | None
    cell_rmse: float | None
    cell_allclose: bool | None
    zero_history_cell_max_abs_error: float | None

    packed_run_lineage_fingerprint: str
    reference_run_lineage_fingerprint: str

    schema_version: str = RECURRENT_DIAGNOSTICS_SCHEMA_VERSION

    def __post_init__(
        self,
    ) -> None:
        for name, value in (
            (
                "encoder_kind",
                self.encoder_kind,
            ),
            (
                "packed_execution_path",
                self.packed_execution_path,
            ),
            (
                "reference_execution_path",
                self.reference_execution_path,
            ),
            (
                "packed_run_lineage_fingerprint",
                self.packed_run_lineage_fingerprint,
            ),
            (
                "reference_run_lineage_fingerprint",
                self.reference_run_lineage_fingerprint,
            ),
            (
                "schema_version",
                self.schema_version,
            ),
        ):
            _require_nonempty_string(
                name,
                value,
            )

        if self.encoder_kind not in (
            TemporalSequenceEncoderKind.GRU.value,
            TemporalSequenceEncoderKind.LSTM.value,
        ):
            raise ValueError(
                "encoder_kind must be 'gru' or 'lstm'."
            )

        if self.packed_execution_path != RecurrentExecutionPath.PACKED.value:
            raise ValueError(
                "packed_execution_path must be 'packed'."
            )

        if (
            self.reference_execution_path
            != RecurrentExecutionPath.REFERENCE.value
        ):
            raise ValueError(
                "reference_execution_path must be 'reference'."
            )

        for name, value in (
            (
                "node_count",
                self.node_count,
            ),
            (
                "sequence_length",
                self.sequence_length,
            ),
            (
                "output_dim",
                self.output_dim,
            ),
            (
                "hidden_dim",
                self.hidden_dim,
            ),
            (
                "num_layers",
                self.num_layers,
            ),
            (
                "num_directions",
                self.num_directions,
            ),
            (
                "output_difference_nonfinite_count",
                self.output_difference_nonfinite_count,
            ),
            (
                "hidden_difference_nonfinite_count",
                self.hidden_difference_nonfinite_count,
            ),
        ):
            _require_nonnegative_int(
                name,
                value,
            )

        _require_optional_nonnegative_int(
            "cell_difference_nonfinite_count",
            self.cell_difference_nonfinite_count,
        )

        for name, value in (
            (
                "has_cell_state",
                self.has_cell_state,
            ),
            (
                "architecture_match",
                self.architecture_match,
            ),
            (
                "source_object_identity",
                self.source_object_identity,
            ),
            (
                "source_alignment_match",
                self.source_alignment_match,
            ),
            (
                "source_values_match",
                self.source_values_match,
            ),
            (
                "source_masks_match",
                self.source_masks_match,
            ),
            (
                "source_lineage_match",
                self.source_lineage_match,
            ),
            (
                "state_layout_match",
                self.state_layout_match,
            ),
            (
                "parameter_snapshot_both_absent",
                self.parameter_snapshot_both_absent,
            ),
            (
                "parameter_snapshot_match",
                self.parameter_snapshot_match,
            ),
            (
                "parameter_identity_verified",
                self.parameter_identity_verified,
            ),
            (
                "dropout_active_in_either",
                self.dropout_active_in_either,
            ),
            (
                "equality_expectation_valid",
                self.equality_expectation_valid,
            ),
            (
                "output_allclose",
                self.output_allclose,
            ),
            (
                "hidden_allclose",
                self.hidden_allclose,
            ),
            (
                "cell_comparison_available",
                self.cell_comparison_available,
            ),
        ):
            _require_boolean(
                name,
                value,
            )

        if self.cell_allclose is not None:
            _require_boolean(
                "cell_allclose",
                self.cell_allclose,
            )

        _require_positive_tolerance(
            "rtol",
            self.rtol,
            allow_zero=True,
        )
        _require_positive_tolerance(
            "atol",
            self.atol,
            allow_zero=True,
        )

        for name, value in (
            (
                "output_max_abs_error",
                self.output_max_abs_error,
            ),
            (
                "output_mean_abs_error",
                self.output_mean_abs_error,
            ),
            (
                "output_rmse",
                self.output_rmse,
            ),
            (
                "valid_output_max_abs_error",
                self.valid_output_max_abs_error,
            ),
            (
                "padded_output_max_abs_error",
                self.padded_output_max_abs_error,
            ),
            (
                "zero_history_output_max_abs_error",
                self.zero_history_output_max_abs_error,
            ),
            (
                "hidden_max_abs_error",
                self.hidden_max_abs_error,
            ),
            (
                "hidden_mean_abs_error",
                self.hidden_mean_abs_error,
            ),
            (
                "hidden_rmse",
                self.hidden_rmse,
            ),
            (
                "zero_history_hidden_max_abs_error",
                self.zero_history_hidden_max_abs_error,
            ),
        ):
            _require_nonnegative_finite_float(
                name,
                value,
            )

        for name, value in (
            (
                "cell_max_abs_error",
                self.cell_max_abs_error,
            ),
            (
                "cell_mean_abs_error",
                self.cell_mean_abs_error,
            ),
            (
                "cell_rmse",
                self.cell_rmse,
            ),
            (
                "zero_history_cell_max_abs_error",
                self.zero_history_cell_max_abs_error,
            ),
        ):
            _require_optional_nonnegative_finite_float(
                name,
                value,
            )

        if (
            self.node_count <= 0
            or self.sequence_length <= 0
            or self.output_dim <= 0
            or self.hidden_dim <= 0
            or self.num_layers <= 0
            or self.num_directions <= 0
        ):
            raise ValueError(
                "Comparison dimensions must be strictly positive."
            )

        if self.has_cell_state:
            if self.encoder_kind != TemporalSequenceEncoderKind.LSTM.value:
                raise ValueError(
                    "Only LSTM comparisons may include cell state."
                )

            cell_values = (
                self.cell_difference_nonfinite_count,
                self.cell_max_abs_error,
                self.cell_mean_abs_error,
                self.cell_rmse,
                self.cell_allclose,
                self.zero_history_cell_max_abs_error,
            )

            if any(
                value is None
                for value in cell_values
            ):
                raise ValueError(
                    "LSTM comparisons require all cell-state fields."
                )
        else:
            if self.encoder_kind != TemporalSequenceEncoderKind.GRU.value:
                raise ValueError(
                    "LSTM comparisons require cell-state diagnostics."
                )

            cell_values = (
                self.cell_difference_nonfinite_count,
                self.cell_max_abs_error,
                self.cell_mean_abs_error,
                self.cell_rmse,
                self.cell_allclose,
                self.zero_history_cell_max_abs_error,
            )

            if any(
                value is not None
                for value in cell_values
            ):
                raise ValueError(
                    "GRU comparisons must use None cell-state fields."
                )

        expected_verified = (
            not self.parameter_snapshot_both_absent
            and self.parameter_snapshot_match
        )

        if self.parameter_identity_verified != expected_verified:
            raise ValueError(
                "parameter_identity_verified must reflect matched explicit "
                "snapshots."
            )

        expected_equality_valid = (
            self.architecture_match
            and self.source_alignment_match
            and self.source_values_match
            and self.source_masks_match
            and self.state_layout_match
            and not self.dropout_active_in_either
        )

        if self.equality_expectation_valid != expected_equality_valid:
            raise ValueError(
                "equality_expectation_valid is inconsistent with comparison "
                "compatibility fields."
            )

    @property
    def inputs_are_compatible(
        self,
    ) -> bool:
        return (
            self.architecture_match
            and self.source_alignment_match
            and self.source_values_match
            and self.source_masks_match
            and self.state_layout_match
        )

    @property
    def is_numerically_equivalent(
        self,
    ) -> bool:
        return (
            self.output_difference_nonfinite_count == 0
            and self.hidden_difference_nonfinite_count == 0
            and (
                self.cell_difference_nonfinite_count
                in (
                    None,
                    0,
                )
            )
            and self.output_allclose
            and self.hidden_allclose
            and (
                self.cell_allclose
                in (
                    None,
                    True,
                )
            )
        )

    @property
    def is_controlled_equivalence(
        self,
    ) -> bool:
        return (
            self.equality_expectation_valid
            and self.parameter_identity_verified
            and self.is_numerically_equivalent
        )


# =============================================================================
# Observation and norm diagnostics
# =============================================================================


def _all_features_missing_diagnostics(
    run: RecurrentSequenceEncoderRun,
) -> tuple[
    bool,
    int | None,
    float | None,
]:
    source = run.source_history
    feature_mask = source.feature_observed_mask

    if feature_mask is None:
        return (
            False,
            None,
            None,
        )

    valid = source.timestep_mask
    all_missing_valid = (
        valid
        & (
            ~feature_mask.any(
                dim=-1
            )
        )
    )
    count = int(
        all_missing_valid
        .sum()
        .item()
    )
    valid_count = int(
        valid.sum().item()
    )
    fraction = (
        float(
            count
        )
        / float(
            valid_count
        )
        if valid_count > 0
        else 0.0
    )

    return (
        True,
        count,
        fraction,
    )


def _hidden_boundary_diagnostics(
    hidden_state: torch.Tensor,
    *,
    threshold: float,
) -> tuple[
    int,
    float,
]:
    absolute = hidden_state.detach().abs()
    count = int(
        (
            absolute
            >= threshold
        )
        .sum()
        .item()
    )
    total = int(
        absolute.numel()
    )

    return (
        count,
        (
            float(
                count
            )
            / float(
                total
            )
            if total > 0
            else 0.0
        ),
    )


# =============================================================================
# Public run diagnostic function
# =============================================================================


def diagnose_recurrent_run(
    module: nn.Module,
    run: RecurrentSequenceEncoderRun,
    *,
    component_name: str | None = None,
    hidden_boundary_abs_threshold: float = (
        RECURRENT_DIAGNOSTICS_DEFAULT_HIDDEN_BOUNDARY_THRESHOLD
    ),
) -> RecurrentRunDiagnostics:
    """
    Compute detached diagnostics for one validated recurrent run.

    The returned object contains only Python scalars and strings.
    """

    if not isinstance(
        module,
        nn.Module,
    ):
        raise TypeError(
            "module must be a torch.nn.Module."
        )

    if not isinstance(
        run,
        RecurrentSequenceEncoderRun,
    ):
        raise TypeError(
            "run must be a RecurrentSequenceEncoderRun."
        )

    _require_nonnegative_finite_float(
        "hidden_boundary_abs_threshold",
        hidden_boundary_abs_threshold,
    )

    if hidden_boundary_abs_threshold <= 0.0:
        raise ValueError(
            "hidden_boundary_abs_threshold must be strictly positive."
        )

    if component_name is None:
        component_name = module.__class__.__name__

    _require_nonempty_string(
        "component_name",
        component_name,
    )

    architecture_method = getattr(
        module,
        "architecture_provenance",
        None,
    )

    if callable(
        architecture_method
    ):
        module_architecture = architecture_method()

        if (
            module_architecture.architecture_fingerprint
            != run.architecture_fingerprint
        ):
            raise ValueError(
                "module architecture does not match recurrent run."
            )

    parameter_count, trainable_count = count_module_parameters(
        module
    )

    output = run.public_output.encoded_sequence.detach()
    hidden = run.final_hidden_state.detach()
    cell = (
        run.final_cell_state.detach()
        if run.final_cell_state is not None
        else None
    )
    mask = run.source_history.timestep_mask.detach()
    lengths = run.execution_metadata.history_lengths.detach()
    zero_indices = (
        run
        .execution_metadata
        .zero_history_node_indices
    )
    nonempty_indices = (
        run
        .execution_metadata
        .nonempty_node_indices
    )

    padded_output = output[
        (
            ~mask
        ).unsqueeze(
            -1
        ).expand_as(
            output
        )
    ]
    valid_output = output[
        mask
    ]

    if zero_indices.numel() > 0:
        zero_output = output.index_select(
            0,
            zero_indices,
        )
        zero_hidden = hidden.index_select(
            2,
            zero_indices,
        )
        zero_cell = (
            cell.index_select(
                2,
                zero_indices,
            )
            if cell is not None
            else None
        )
    else:
        zero_output = output.new_empty(
            (
                0,
                run.sequence_length,
                run.output_dim,
            )
        )
        zero_hidden = hidden.new_empty(
            (
                run.num_layers,
                run.num_directions,
                0,
                run.hidden_dim,
            )
        )
        zero_cell = (
            cell.new_empty(
                (
                    run.num_layers,
                    run.num_directions,
                    0,
                    run.hidden_dim,
                )
            )
            if cell is not None
            else None
        )

    (
        mean_valid_output_norm,
        max_valid_output_norm,
    ) = _mean_and_max_vector_norm(
        valid_output
    )
    (
        mean_hidden_norm,
        max_hidden_norm,
    ) = _mean_and_max_vector_norm(
        hidden
    )

    if cell is not None:
        (
            mean_cell_norm,
            max_cell_norm,
        ) = _mean_and_max_vector_norm(
            cell
        )
    else:
        mean_cell_norm = None
        max_cell_norm = None

    (
        hidden_boundary_count,
        hidden_boundary_fraction,
    ) = _hidden_boundary_diagnostics(
        hidden,
        threshold=(
            hidden_boundary_abs_threshold
        ),
    )
    (
        observed_mask_available,
        all_missing_count,
        all_missing_fraction,
    ) = _all_features_missing_diagnostics(
        run
    )

    lineage = _lineage_metadata(
        run
    )
    dropout_probability = _lineage_float(
        lineage,
        "dropout_probability",
        default=float(
            run
            .public_output
            .computation_provenance
            .architecture
            .architecture_metadata
            .get(
                "dropout",
                0.0,
            )
        ),
    )
    dropout_active = _lineage_boolean(
        lineage,
        "dropout_active",
        default=False,
    )
    all_zero_short_circuit = _lineage_boolean(
        lineage,
        "all_zero_history_short_circuit",
        default=run.execution_metadata.all_zero_history,
    )
    adapter_executed = _lineage_boolean(
        lineage,
        "adapter_executed",
        default=(
            not run.execution_metadata.all_zero_history
        ),
    )
    kernel_executed = _lineage_boolean(
        lineage,
        "recurrent_kernel_executed",
        default=(
            not run.execution_metadata.all_zero_history
        ),
    )

    valid_timestep_count = int(
        mask.sum().item()
    )
    max_abs_padded_output = _maximum_absolute(
        padded_output
    )
    max_abs_zero_output = _maximum_absolute(
        zero_output
    )
    max_abs_zero_hidden = _maximum_absolute(
        zero_hidden
    )
    max_abs_zero_cell = (
        _maximum_absolute(
            zero_cell
        )
        if zero_cell is not None
        else None
    )

    return RecurrentRunDiagnostics(
        component_name=component_name,
        encoder_kind=run.encoder_kind.value,
        execution_path=(
            run
            .execution_metadata
            .execution_path
            .value
        ),
        node_count=run.node_count,
        sequence_length=run.sequence_length,
        input_dim=run.source_history.feature_dim,
        output_dim=run.output_dim,
        hidden_dim=run.hidden_dim,
        num_layers=run.num_layers,
        num_directions=run.num_directions,
        bidirectional=run.is_bidirectional,
        has_cell_state=run.has_cell_state,
        parameter_count=parameter_count,
        trainable_parameter_count=trainable_count,
        module_training=module.training,
        dropout_probability=dropout_probability,
        dropout_active=dropout_active,
        original_padding_direction=(
            run
            .execution_metadata
            .original_padding_direction
            .value
        ),
        canonical_padding_direction=(
            run
            .execution_metadata
            .canonical_padding_direction
            .value
        ),
        sort_was_applied=(
            run
            .execution_metadata
            .sort_was_applied
        ),
        identity_permutation=(
            run
            .execution_metadata
            .identity_permutation
        ),
        all_zero_history_short_circuit=(
            all_zero_short_circuit
        ),
        adapter_executed=adapter_executed,
        recurrent_kernel_executed=(
            kernel_executed
        ),
        zero_history_count=(
            run
            .execution_metadata
            .zero_history_count
        ),
        nonempty_history_count=(
            run
            .execution_metadata
            .nonempty_node_count
        ),
        valid_timestep_count=(
            valid_timestep_count
        ),
        padded_timestep_count=(
            run.node_count
            * run.sequence_length
            - valid_timestep_count
        ),
        valid_length_min=int(
            lengths.min().item()
        ),
        valid_length_mean=float(
            lengths
            .to(
                dtype=torch.float64
            )
            .mean()
            .item()
        ),
        valid_length_max=int(
            lengths.max().item()
        ),
        feature_observed_mask_available=(
            observed_mask_available
        ),
        all_features_missing_valid_timestep_count=(
            all_missing_count
        ),
        all_features_missing_valid_timestep_fraction=(
            all_missing_fraction
        ),
        nonfinite_output_count=(
            _tensor_nonfinite_count(
                output
            )
        ),
        nonfinite_hidden_state_count=(
            _tensor_nonfinite_count(
                hidden
            )
        ),
        nonfinite_cell_state_count=(
            _tensor_nonfinite_count(
                cell
            )
            if cell is not None
            else None
        ),
        max_abs_padded_output=(
            max_abs_padded_output
        ),
        max_abs_zero_history_output=(
            max_abs_zero_output
        ),
        max_abs_zero_history_hidden_state=(
            max_abs_zero_hidden
        ),
        max_abs_zero_history_cell_state=(
            max_abs_zero_cell
        ),
        mean_valid_output_vector_l2_norm=(
            mean_valid_output_norm
        ),
        max_valid_output_vector_l2_norm=(
            max_valid_output_norm
        ),
        mean_final_hidden_vector_l2_norm=(
            mean_hidden_norm
        ),
        max_final_hidden_vector_l2_norm=(
            max_hidden_norm
        ),
        mean_final_cell_vector_l2_norm=(
            mean_cell_norm
        ),
        max_final_cell_vector_l2_norm=(
            max_cell_norm
        ),
        hidden_boundary_abs_threshold=float(
            hidden_boundary_abs_threshold
        ),
        hidden_abs_ge_boundary_count=(
            hidden_boundary_count
        ),
        hidden_abs_ge_boundary_fraction=(
            hidden_boundary_fraction
        ),
        exact_zero_padding=(
            max_abs_padded_output == 0.0
        ),
        exact_zero_history_output=(
            max_abs_zero_output == 0.0
        ),
        exact_zero_history_hidden_state=(
            max_abs_zero_hidden == 0.0
        ),
        exact_zero_history_cell_state=(
            (
                max_abs_zero_cell == 0.0
            )
            if max_abs_zero_cell is not None
            else None
        ),
        output_requires_grad=(
            run
            .public_output
            .encoded_sequence
            .requires_grad
        ),
        hidden_state_requires_grad=(
            run
            .final_hidden_state
            .requires_grad
        ),
        cell_state_requires_grad=(
            run
            .final_cell_state
            .requires_grad
            if run.final_cell_state is not None
            else None
        ),
        architecture_fingerprint=(
            run.architecture_fingerprint
        ),
        computation_lineage_fingerprint=(
            run.computation_lineage_fingerprint
        ),
        source_lineage_fingerprint=(
            run
            .source_history
            .lineage_fingerprint()
        ),
        run_lineage_fingerprint=(
            run.lineage_fingerprint()
        ),
        state_layout_fingerprint=(
            run
            .state_layout
            .fingerprint()
        ),
        execution_metadata_fingerprint=(
            run
            .execution_metadata
            .fingerprint()
        ),
        parameter_snapshot_fingerprint=(
            run
            .public_output
            .parameter_snapshot_fingerprint
        ),
    )


# =============================================================================
# Packed/reference comparison
# =============================================================================


def _validate_comparison_shapes(
    packed_run: RecurrentSequenceEncoderRun,
    reference_run: RecurrentSequenceEncoderRun,
) -> None:
    if packed_run.encoder_kind != reference_run.encoder_kind:
        raise ValueError(
            "Packed and reference runs must use the same recurrent cell kind."
        )

    if packed_run.node_count != reference_run.node_count:
        raise ValueError(
            "Packed and reference runs must have the same node count."
        )

    if packed_run.sequence_length != reference_run.sequence_length:
        raise ValueError(
            "Packed and reference runs must have the same sequence length."
        )

    if packed_run.output_dim != reference_run.output_dim:
        raise ValueError(
            "Packed and reference runs must have the same output width."
        )

    if packed_run.hidden_dim != reference_run.hidden_dim:
        raise ValueError(
            "Packed and reference runs must have the same state width."
        )

    if packed_run.num_layers != reference_run.num_layers:
        raise ValueError(
            "Packed and reference runs must have the same layer count."
        )

    if packed_run.num_directions != reference_run.num_directions:
        raise ValueError(
            "Packed and reference runs must have the same direction count."
        )

    if packed_run.has_cell_state != reference_run.has_cell_state:
        raise ValueError(
            "Packed and reference runs must agree on cell-state presence."
        )

    if packed_run.dtype != reference_run.dtype:
        raise ValueError(
            "Packed and reference runs must use the same dtype."
        )

    if packed_run.device != reference_run.device:
        raise ValueError(
            "Packed and reference runs must use the same device."
        )


def compare_packed_reference_runs(
    packed_run: RecurrentSequenceEncoderRun,
    reference_run: RecurrentSequenceEncoderRun,
    *,
    rtol: float | None = None,
    atol: float | None = None,
) -> PackedReferenceComparisonDiagnostics:
    """
    Compare packed and exact-reference runs without modifying either run.

    Equality is considered an appropriate expectation only when architecture,
    source alignment/values/masks, and state layout match and neither execution
    lineage reports active recurrent dropout.
    """

    if not isinstance(
        packed_run,
        RecurrentSequenceEncoderRun,
    ):
        raise TypeError(
            "packed_run must be a RecurrentSequenceEncoderRun."
        )

    if not isinstance(
        reference_run,
        RecurrentSequenceEncoderRun,
    ):
        raise TypeError(
            "reference_run must be a RecurrentSequenceEncoderRun."
        )

    if (
        packed_run.execution_metadata.execution_path
        != RecurrentExecutionPath.PACKED
    ):
        raise ValueError(
            "packed_run must use packed execution."
        )

    if (
        reference_run.execution_metadata.execution_path
        != RecurrentExecutionPath.REFERENCE
    ):
        raise ValueError(
            "reference_run must use reference execution."
        )

    _validate_comparison_shapes(
        packed_run,
        reference_run,
    )

    default_rtol, default_atol = _default_tolerances(
        packed_run.dtype
    )

    if rtol is None:
        rtol = default_rtol

    if atol is None:
        atol = default_atol

    _require_positive_tolerance(
        "rtol",
        rtol,
        allow_zero=True,
    )
    _require_positive_tolerance(
        "atol",
        atol,
        allow_zero=True,
    )

    packed_output = (
        packed_run
        .public_output
        .encoded_sequence
        .detach()
    )
    reference_output = (
        reference_run
        .public_output
        .encoded_sequence
        .detach()
    )
    packed_hidden = (
        packed_run
        .final_hidden_state
        .detach()
    )
    reference_hidden = (
        reference_run
        .final_hidden_state
        .detach()
    )

    (
        output_nonfinite,
        output_max,
        output_mean,
        output_rmse,
    ) = _difference_metrics(
        packed_output,
        reference_output,
    )
    (
        hidden_nonfinite,
        hidden_max,
        hidden_mean,
        hidden_rmse,
    ) = _difference_metrics(
        packed_hidden,
        reference_hidden,
    )

    packed_mask = (
        packed_run
        .source_history
        .timestep_mask
    )
    reference_mask = (
        reference_run
        .source_history
        .timestep_mask
    )
    source_masks_match = torch.equal(
        packed_mask,
        reference_mask,
    )
    source_values_match = torch.equal(
        packed_run.source_history.history,
        reference_run.source_history.history,
    )
    source_alignment_match = (
        packed_run
        .public_output
        .alignment_fingerprint()
        == reference_run
        .public_output
        .alignment_fingerprint()
    )
    source_lineage_match = (
        packed_run
        .source_history
        .lineage_fingerprint()
        == reference_run
        .source_history
        .lineage_fingerprint()
    )

    if source_masks_match:
        valid_mask = packed_mask
        padded_mask = ~packed_mask

        valid_difference = (
            packed_output[
                valid_mask
            ]
            - reference_output[
                valid_mask
            ]
        )
        padded_difference = (
            packed_output[
                padded_mask
            ]
            - reference_output[
                padded_mask
            ]
        )
        valid_output_max = _maximum_absolute(
            valid_difference
        )
        padded_output_max = _maximum_absolute(
            padded_difference
        )
    else:
        valid_output_max = output_max
        padded_output_max = output_max

    zero_indices_match = torch.equal(
        packed_run
        .execution_metadata
        .zero_history_node_indices,
        reference_run
        .execution_metadata
        .zero_history_node_indices,
    )

    if zero_indices_match:
        zero_indices = (
            packed_run
            .execution_metadata
            .zero_history_node_indices
        )
    else:
        zero_indices = torch.empty(
            0,
            dtype=torch.long,
            device=packed_run.device,
        )

    if zero_indices.numel() > 0:
        zero_output_difference = (
            packed_output.index_select(
                0,
                zero_indices,
            )
            - reference_output.index_select(
                0,
                zero_indices,
            )
        )
        zero_hidden_difference = (
            packed_hidden.index_select(
                2,
                zero_indices,
            )
            - reference_hidden.index_select(
                2,
                zero_indices,
            )
        )
        zero_output_max = _maximum_absolute(
            zero_output_difference
        )
        zero_hidden_max = _maximum_absolute(
            zero_hidden_difference
        )
    else:
        zero_output_max = 0.0
        zero_hidden_max = 0.0

    if packed_run.has_cell_state:
        assert packed_run.final_cell_state is not None
        assert reference_run.final_cell_state is not None

        packed_cell = packed_run.final_cell_state.detach()
        reference_cell = reference_run.final_cell_state.detach()
        (
            cell_nonfinite,
            cell_max,
            cell_mean,
            cell_rmse,
        ) = _difference_metrics(
            packed_cell,
            reference_cell,
        )
        cell_allclose = bool(
            torch.allclose(
                packed_cell,
                reference_cell,
                rtol=float(
                    rtol
                ),
                atol=float(
                    atol
                ),
                equal_nan=False,
            )
        )

        if zero_indices.numel() > 0:
            zero_cell_max = _maximum_absolute(
                packed_cell.index_select(
                    2,
                    zero_indices,
                )
                - reference_cell.index_select(
                    2,
                    zero_indices,
                )
            )
        else:
            zero_cell_max = 0.0
    else:
        cell_nonfinite = None
        cell_max = None
        cell_mean = None
        cell_rmse = None
        cell_allclose = None
        zero_cell_max = None

    packed_snapshot = (
        packed_run
        .public_output
        .parameter_snapshot_fingerprint
    )
    reference_snapshot = (
        reference_run
        .public_output
        .parameter_snapshot_fingerprint
    )
    snapshots_both_absent = (
        packed_snapshot is None
        and reference_snapshot is None
    )
    snapshot_match = (
        packed_snapshot
        == reference_snapshot
    )
    parameter_identity_verified = (
        not snapshots_both_absent
        and snapshot_match
    )

    packed_lineage = _lineage_metadata(
        packed_run
    )
    reference_lineage = _lineage_metadata(
        reference_run
    )
    dropout_active = (
        _lineage_boolean(
            packed_lineage,
            "dropout_active",
            default=False,
        )
        or _lineage_boolean(
            reference_lineage,
            "dropout_active",
            default=False,
        )
    )

    architecture_match = (
        packed_run.architecture_fingerprint
        == reference_run.architecture_fingerprint
    )
    state_layout_match = (
        packed_run.state_layout.fingerprint()
        == reference_run.state_layout.fingerprint()
    )
    equality_expectation_valid = (
        architecture_match
        and source_alignment_match
        and source_values_match
        and source_masks_match
        and state_layout_match
        and not dropout_active
    )

    return PackedReferenceComparisonDiagnostics(
        encoder_kind=packed_run.encoder_kind.value,
        node_count=packed_run.node_count,
        sequence_length=packed_run.sequence_length,
        output_dim=packed_run.output_dim,
        hidden_dim=packed_run.hidden_dim,
        num_layers=packed_run.num_layers,
        num_directions=packed_run.num_directions,
        has_cell_state=packed_run.has_cell_state,
        rtol=float(
            rtol
        ),
        atol=float(
            atol
        ),
        packed_execution_path=(
            packed_run
            .execution_metadata
            .execution_path
            .value
        ),
        reference_execution_path=(
            reference_run
            .execution_metadata
            .execution_path
            .value
        ),
        architecture_match=(
            architecture_match
        ),
        source_object_identity=(
            packed_run.source_history
            is reference_run.source_history
        ),
        source_alignment_match=(
            source_alignment_match
        ),
        source_values_match=(
            source_values_match
        ),
        source_masks_match=(
            source_masks_match
        ),
        source_lineage_match=(
            source_lineage_match
        ),
        state_layout_match=(
            state_layout_match
        ),
        parameter_snapshot_both_absent=(
            snapshots_both_absent
        ),
        parameter_snapshot_match=(
            snapshot_match
        ),
        parameter_identity_verified=(
            parameter_identity_verified
        ),
        dropout_active_in_either=(
            dropout_active
        ),
        equality_expectation_valid=(
            equality_expectation_valid
        ),
        output_difference_nonfinite_count=(
            output_nonfinite
        ),
        output_max_abs_error=(
            output_max
        ),
        output_mean_abs_error=(
            output_mean
        ),
        output_rmse=(
            output_rmse
        ),
        output_allclose=bool(
            torch.allclose(
                packed_output,
                reference_output,
                rtol=float(
                    rtol
                ),
                atol=float(
                    atol
                ),
                equal_nan=False,
            )
        ),
        valid_output_max_abs_error=(
            valid_output_max
        ),
        padded_output_max_abs_error=(
            padded_output_max
        ),
        zero_history_output_max_abs_error=(
            zero_output_max
        ),
        hidden_difference_nonfinite_count=(
            hidden_nonfinite
        ),
        hidden_max_abs_error=(
            hidden_max
        ),
        hidden_mean_abs_error=(
            hidden_mean
        ),
        hidden_rmse=(
            hidden_rmse
        ),
        hidden_allclose=bool(
            torch.allclose(
                packed_hidden,
                reference_hidden,
                rtol=float(
                    rtol
                ),
                atol=float(
                    atol
                ),
                equal_nan=False,
            )
        ),
        zero_history_hidden_max_abs_error=(
            zero_hidden_max
        ),
        cell_comparison_available=(
            packed_run.has_cell_state
        ),
        cell_difference_nonfinite_count=(
            cell_nonfinite
        ),
        cell_max_abs_error=(
            cell_max
        ),
        cell_mean_abs_error=(
            cell_mean
        ),
        cell_rmse=(
            cell_rmse
        ),
        cell_allclose=(
            cell_allclose
        ),
        zero_history_cell_max_abs_error=(
            zero_cell_max
        ),
        packed_run_lineage_fingerprint=(
            packed_run.lineage_fingerprint()
        ),
        reference_run_lineage_fingerprint=(
            reference_run.lineage_fingerprint()
        ),
    )


# =============================================================================
# Diagnostic facade
# =============================================================================


@dataclass(
    frozen=True,
    slots=True,
)
class RecurrentDiagnostics:
    """
    Stateless facade for recurrent run diagnostics and path comparison.

    ``enabled`` exists so a future orchestrator may carry a diagnostics policy
    object without changing ordinary forward execution. Ordinary encoder
    ``forward`` methods do not invoke this object.
    """

    enabled: bool = True
    hidden_boundary_abs_threshold: float = (
        RECURRENT_DIAGNOSTICS_DEFAULT_HIDDEN_BOUNDARY_THRESHOLD
    )
    schema_version: str = RECURRENT_DIAGNOSTICS_SCHEMA_VERSION

    def __post_init__(
        self,
    ) -> None:
        _require_boolean(
            "enabled",
            self.enabled,
        )
        _require_nonnegative_finite_float(
            "hidden_boundary_abs_threshold",
            self.hidden_boundary_abs_threshold,
        )
        _require_nonempty_string(
            "schema_version",
            self.schema_version,
        )

        if self.hidden_boundary_abs_threshold <= 0.0:
            raise ValueError(
                "hidden_boundary_abs_threshold must be strictly positive."
            )

    def diagnose(
        self,
        module: nn.Module,
        run: RecurrentSequenceEncoderRun,
        *,
        component_name: str | None = None,
    ) -> RecurrentRunDiagnostics | None:
        if not self.enabled:
            return None

        return diagnose_recurrent_run(
            module,
            run,
            component_name=component_name,
            hidden_boundary_abs_threshold=(
                self.hidden_boundary_abs_threshold
            ),
        )

    def report(
        self,
        module: nn.Module,
        run: RecurrentSequenceEncoderRun,
        *,
        component_name: str | None = None,
    ) -> RecurrentRunDiagnostics | None:
        return self.diagnose(
            module,
            run,
            component_name=component_name,
        )

    def compare(
        self,
        packed_run: RecurrentSequenceEncoderRun,
        reference_run: RecurrentSequenceEncoderRun,
        *,
        rtol: float | None = None,
        atol: float | None = None,
    ) -> PackedReferenceComparisonDiagnostics | None:
        if not self.enabled:
            return None

        return compare_packed_reference_runs(
            packed_run,
            reference_run,
            rtol=rtol,
            atol=atol,
        )

    def replace(
        self,
        **changes: Any,
    ) -> Self:
        from dataclasses import replace as dataclass_replace

        return dataclass_replace(
            self,
            **changes,
        )


# =============================================================================
# Compact aliases
# =============================================================================


RecurrentEncoderDiagnostics = RecurrentDiagnostics
RecurrentDiagnosticReport = RecurrentRunDiagnostics
PackedReferenceDiagnostics = PackedReferenceComparisonDiagnostics

diagnose_recurrent_encoder_run = diagnose_recurrent_run
build_recurrent_diagnostic_report = diagnose_recurrent_run
compare_recurrent_execution_paths = compare_packed_reference_runs
compare_packed_and_reference_runs = compare_packed_reference_runs


# =============================================================================
# Public API
# =============================================================================


__all__ = (
    # Identity and interpretation.
    "RECURRENT_DIAGNOSTICS_SCHEMA_VERSION",
    "RECURRENT_DIAGNOSTICS_COMPONENT_NAME",
    "RECURRENT_DIAGNOSTICS_COMPONENT_KIND",
    "RECURRENT_DIAGNOSTICS_OPERATION_NAME",
    "RECURRENT_PACKED_REFERENCE_COMPARISON_OPERATION_NAME",
    "RECURRENT_DIAGNOSTICS_DEFAULT_HIDDEN_BOUNDARY_THRESHOLD",
    "RECURRENT_DIAGNOSTICS_SCIENTIFIC_INTERPRETATION",
    "RECURRENT_PACKED_REFERENCE_COMPARISON_INTERPRETATION",

    # Diagnostic schemas.
    "RecurrentRunDiagnostics",
    "PackedReferenceComparisonDiagnostics",
    "RecurrentDiagnostics",

    # Diagnostic functions.
    "diagnose_recurrent_run",
    "compare_packed_reference_runs",

    # Compact aliases.
    "RecurrentEncoderDiagnostics",
    "RecurrentDiagnosticReport",
    "PackedReferenceDiagnostics",
    "diagnose_recurrent_encoder_run",
    "build_recurrent_diagnostic_report",
    "compare_recurrent_execution_paths",
    "compare_packed_and_reference_runs",
)
