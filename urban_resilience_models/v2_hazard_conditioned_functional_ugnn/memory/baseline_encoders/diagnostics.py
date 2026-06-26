"""
Detached diagnostics for Phase 5 temporal baselines.

Target path:
    urban_resilience_models/
        v2_hazard_conditioned_functional_ugnn/
            memory/
                baseline_encoders/
                    diagnostics.py

The diagnostics in this module are descriptive audit artifacts. They do not
alter forward results, gradients, parameters, masks, provenance, or model
state. They make no causal, calibrated-risk, or feature-importance claims.

The module reports high-value numerical and structural checks for:

- sequence-preserving baseline encoders;
- deterministic temporal poolers;
- an optional composed baseline pipeline.

Sequence diagnostics include:

- total and trainable parameter counts;
- valid-history length statistics;
- zero-history counts;
- nonfinite output counts;
- optional nonfinite pre-mask counts;
- maximum absolute padded output;
- valid-vector norm summaries.

Pooling diagnostics include:

- zero-history counts;
- nonfinite and negative weight counts;
- maximum normalization error;
- maximum padded pooling mass;
- zero-history output leakage;
- reconstruction error from the declared weights;
- deviation from the exact deterministic baseline weights;
- pooled-vector norm summaries;
- last-valid selection mismatch;
- last-valid all-feature-missing selection statistics.

Parameter values are not fingerprinted automatically. Parameter counting is
cheap and deterministic; full parameter snapshot hashing remains an explicit
operation in ``_provenance.py``.
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
    TemporalSequenceEncoding,
)
from ..schemas.temporal_pooling import (
    TemporalPoolingKind,
    TemporalPoolingOutput,
)
from ._provenance import (
    count_module_parameters,
)


# =============================================================================
# Diagnostic identity
# =============================================================================


BASELINE_DIAGNOSTICS_SCHEMA_VERSION: Final[str] = "0.1"

BASELINE_DIAGNOSTICS_SCIENTIFIC_INTERPRETATION: Final[str] = (
    "descriptive_numerical_and_structural_audit_not_causal_explanation"
)


# =============================================================================
# Scalar helpers
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


def _require_finite_nonnegative_float(
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

    converted = float(
        value
    )

    if not math.isfinite(
        converted
    ):
        raise ValueError(
            f"{name} must be finite."
        )

    if converted < 0.0:
        raise ValueError(
            f"{name} must be nonnegative."
        )


def _require_optional_finite_nonnegative_float(
    name: str,
    value: float | None,
) -> None:
    if value is None:
        return

    _require_finite_nonnegative_float(
        name,
        value,
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


def _tensor_negative_count(
    value: torch.Tensor,
) -> int:
    finite = torch.isfinite(
        value
    )

    return int(
        (
            finite
            & (
                value
                < 0
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


def _mean_and_max_vector_norm(
    value: torch.Tensor,
) -> tuple[float, float]:
    if value.numel() == 0:
        return (
            0.0,
            0.0,
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
            norms.mean().item()
        ),
        float(
            norms.max().item()
        ),
    )


def _maximum_absolute_difference(
    first: torch.Tensor,
    second: torch.Tensor,
) -> float:
    if first.shape != second.shape:
        raise ValueError(
            "Diagnostic tensors must have the same shape."
        )

    difference = (
        first
        - second
    )

    finite = torch.isfinite(
        difference
    )

    if not bool(
        finite.all().item()
    ):
        return float(
            "inf"
        )

    return _maximum_absolute(
        difference
    )


# =============================================================================
# Serialization mixin
# =============================================================================


class _DiagnosticsMixin:
    """Deterministic serialization and fingerprinting for audit artifacts."""

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
# Sequence diagnostics
# =============================================================================


@dataclass(
    frozen=True,
    slots=True,
)
class BaselineSequenceDiagnostics(
    _DiagnosticsMixin
):
    """Immutable diagnostics for one ``TemporalSequenceEncoding``."""

    component_name: str

    node_count: int
    sequence_length: int
    hidden_dim: int

    parameter_count: int
    trainable_parameter_count: int
    module_training: bool

    zero_history_count: int
    nonempty_history_count: int
    valid_timestep_count: int
    padded_timestep_count: int

    valid_length_min: int
    valid_length_mean: float
    valid_length_max: int

    pre_mask_output_supplied: bool
    nonfinite_pre_mask_count: int | None
    nonfinite_output_count: int

    max_abs_padded_output: float
    mean_valid_vector_l2_norm: float
    max_valid_vector_l2_norm: float

    exact_zero_padding: bool
    output_requires_grad: bool

    architecture_fingerprint: str
    source_lineage_fingerprint: str
    output_lineage_fingerprint: str

    schema_version: str = BASELINE_DIAGNOSTICS_SCHEMA_VERSION

    def __post_init__(
        self,
    ) -> None:
        for name, value in (
            (
                "component_name",
                self.component_name,
            ),
            (
                "architecture_fingerprint",
                self.architecture_fingerprint,
            ),
            (
                "source_lineage_fingerprint",
                self.source_lineage_fingerprint,
            ),
            (
                "output_lineage_fingerprint",
                self.output_lineage_fingerprint,
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
                "hidden_dim",
                self.hidden_dim,
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
        ):
            _require_nonnegative_int(
                name,
                value,
            )

        _require_optional_nonnegative_int(
            "nonfinite_pre_mask_count",
            self.nonfinite_pre_mask_count,
        )

        for name, value in (
            (
                "valid_length_mean",
                self.valid_length_mean,
            ),
            (
                "max_abs_padded_output",
                self.max_abs_padded_output,
            ),
            (
                "mean_valid_vector_l2_norm",
                self.mean_valid_vector_l2_norm,
            ),
            (
                "max_valid_vector_l2_norm",
                self.max_valid_vector_l2_norm,
            ),
        ):
            _require_finite_nonnegative_float(
                name,
                value,
            )

        if self.trainable_parameter_count > self.parameter_count:
            raise ValueError(
                "trainable_parameter_count cannot exceed "
                "parameter_count."
            )

        if (
            self.zero_history_count
            + self.nonempty_history_count
            != self.node_count
        ):
            raise ValueError(
                "zero_history_count + nonempty_history_count must "
                "equal node_count."
            )

        if (
            self.valid_timestep_count
            + self.padded_timestep_count
            != self.node_count
            * self.sequence_length
        ):
            raise ValueError(
                "valid and padded timestep counts must cover [N, T]."
            )

        if self.pre_mask_output_supplied:
            if self.nonfinite_pre_mask_count is None:
                raise ValueError(
                    "nonfinite_pre_mask_count is required when a "
                    "pre-mask tensor was supplied."
                )
        elif self.nonfinite_pre_mask_count is not None:
            raise ValueError(
                "nonfinite_pre_mask_count must be None when no "
                "pre-mask tensor was supplied."
            )

    @property
    def is_numerically_clean(
        self,
    ) -> bool:
        return (
            self.nonfinite_output_count == 0
            and (
                self.nonfinite_pre_mask_count
                in (
                    None,
                    0,
                )
            )
            and self.exact_zero_padding
        )


# =============================================================================
# Pooling diagnostics
# =============================================================================


@dataclass(
    frozen=True,
    slots=True,
)
class BaselinePoolingDiagnostics(
    _DiagnosticsMixin
):
    """Immutable diagnostics for one ``TemporalPoolingOutput``."""

    component_name: str
    pooling_kind: str

    node_count: int
    sequence_length: int
    output_dim: int
    num_heads: int

    parameter_count: int
    trainable_parameter_count: int
    module_training: bool

    zero_history_count: int
    nonempty_history_count: int

    nonfinite_weight_count: int
    negative_weight_count: int
    nonfinite_pooled_output_count: int

    max_normalization_error: float
    max_padded_pooling_mass: float
    max_zero_history_weight_mass: float
    max_abs_zero_history_pooled_memory: float

    deterministic_weight_max_error: float | None
    pooled_reconstruction_max_error: float | None

    mean_pooled_vector_l2_norm: float
    max_pooled_vector_l2_norm: float

    last_valid_selection_mismatch_count: int | None
    selected_all_features_missing_status_available: bool
    selected_all_features_missing_count: int | None
    selected_all_features_missing_fraction: float | None

    exact_source_encoding_identity: bool
    pooled_output_requires_grad: bool

    architecture_fingerprint: str
    source_lineage_fingerprint: str
    output_lineage_fingerprint: str

    schema_version: str = BASELINE_DIAGNOSTICS_SCHEMA_VERSION

    def __post_init__(
        self,
    ) -> None:
        for name, value in (
            (
                "component_name",
                self.component_name,
            ),
            (
                "pooling_kind",
                self.pooling_kind,
            ),
            (
                "architecture_fingerprint",
                self.architecture_fingerprint,
            ),
            (
                "source_lineage_fingerprint",
                self.source_lineage_fingerprint,
            ),
            (
                "output_lineage_fingerprint",
                self.output_lineage_fingerprint,
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
                "num_heads",
                self.num_heads,
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
                "nonfinite_weight_count",
                self.nonfinite_weight_count,
            ),
            (
                "negative_weight_count",
                self.negative_weight_count,
            ),
            (
                "nonfinite_pooled_output_count",
                self.nonfinite_pooled_output_count,
            ),
        ):
            _require_nonnegative_int(
                name,
                value,
            )

        _require_optional_nonnegative_int(
            "last_valid_selection_mismatch_count",
            self.last_valid_selection_mismatch_count,
        )
        _require_optional_nonnegative_int(
            "selected_all_features_missing_count",
            self.selected_all_features_missing_count,
        )

        for name, value in (
            (
                "max_normalization_error",
                self.max_normalization_error,
            ),
            (
                "max_padded_pooling_mass",
                self.max_padded_pooling_mass,
            ),
            (
                "max_zero_history_weight_mass",
                self.max_zero_history_weight_mass,
            ),
            (
                "max_abs_zero_history_pooled_memory",
                self.max_abs_zero_history_pooled_memory,
            ),
            (
                "mean_pooled_vector_l2_norm",
                self.mean_pooled_vector_l2_norm,
            ),
            (
                "max_pooled_vector_l2_norm",
                self.max_pooled_vector_l2_norm,
            ),
        ):
            _require_finite_nonnegative_float(
                name,
                value,
            )

        _require_optional_finite_nonnegative_float(
            "deterministic_weight_max_error",
            self.deterministic_weight_max_error,
        )
        _require_optional_finite_nonnegative_float(
            "pooled_reconstruction_max_error",
            self.pooled_reconstruction_max_error,
        )
        _require_optional_finite_nonnegative_float(
            "selected_all_features_missing_fraction",
            self.selected_all_features_missing_fraction,
        )

        if self.trainable_parameter_count > self.parameter_count:
            raise ValueError(
                "trainable_parameter_count cannot exceed "
                "parameter_count."
            )

        if (
            self.zero_history_count
            + self.nonempty_history_count
            != self.node_count
        ):
            raise ValueError(
                "zero_history_count + nonempty_history_count must "
                "equal node_count."
            )

        if self.selected_all_features_missing_status_available:
            if (
                self.selected_all_features_missing_count is None
                or self.selected_all_features_missing_fraction is None
            ):
                raise ValueError(
                    "Available all-missing selection diagnostics "
                    "require count and fraction."
                )
        elif (
            self.selected_all_features_missing_count is not None
            or self.selected_all_features_missing_fraction is not None
        ):
            raise ValueError(
                "Unavailable all-missing selection diagnostics must "
                "use None for count and fraction."
            )

        if (
            self.selected_all_features_missing_fraction is not None
            and self.selected_all_features_missing_fraction > 1.0
        ):
            raise ValueError(
                "selected_all_features_missing_fraction cannot "
                "exceed one."
            )

    @property
    def is_numerically_clean(
        self,
    ) -> bool:
        return (
            self.nonfinite_weight_count == 0
            and self.negative_weight_count == 0
            and self.nonfinite_pooled_output_count == 0
            and self.max_normalization_error == 0.0
            and self.max_padded_pooling_mass == 0.0
            and self.max_zero_history_weight_mass == 0.0
            and self.max_abs_zero_history_pooled_memory == 0.0
            and (
                self.deterministic_weight_max_error
                in (
                    None,
                    0.0,
                )
            )
            and (
                self.pooled_reconstruction_max_error
                in (
                    None,
                    0.0,
                )
            )
            and (
                self.last_valid_selection_mismatch_count
                in (
                    None,
                    0,
                )
            )
        )


# =============================================================================
# Pipeline diagnostics
# =============================================================================


@dataclass(
    frozen=True,
    slots=True,
)
class BaselineDiagnostics(
    _DiagnosticsMixin
):
    """Combined diagnostics for a sequence baseline and optional pooler."""

    sequence: BaselineSequenceDiagnostics
    pooling: BaselinePoolingDiagnostics | None

    pooling_present: bool
    pooling_source_is_sequence: bool | None

    schema_version: str = BASELINE_DIAGNOSTICS_SCHEMA_VERSION

    def __post_init__(
        self,
    ) -> None:
        if not isinstance(
            self.sequence,
            BaselineSequenceDiagnostics,
        ):
            raise TypeError(
                "sequence must be BaselineSequenceDiagnostics."
            )

        if (
            self.pooling is not None
            and not isinstance(
                self.pooling,
                BaselinePoolingDiagnostics,
            )
        ):
            raise TypeError(
                "pooling must be BaselinePoolingDiagnostics or None."
            )

        _require_nonempty_string(
            "schema_version",
            self.schema_version,
        )

        if self.pooling_present != (
            self.pooling is not None
        ):
            raise ValueError(
                "pooling_present must match pooling availability."
            )

        if self.pooling_present:
            if self.pooling_source_is_sequence is None:
                raise ValueError(
                    "pooling_source_is_sequence is required when "
                    "pooling is present."
                )
        elif self.pooling_source_is_sequence is not None:
            raise ValueError(
                "pooling_source_is_sequence must be None when "
                "pooling is absent."
            )

    @property
    def is_numerically_clean(
        self,
    ) -> bool:
        return (
            self.sequence.is_numerically_clean
            and (
                self.pooling is None
                or self.pooling.is_numerically_clean
            )
            and (
                self.pooling_source_is_sequence
                in (
                    None,
                    True,
                )
            )
        )


# =============================================================================
# Expected deterministic pooling weights
# =============================================================================


def _expected_masked_mean_weights(
    output: TemporalPoolingOutput,
) -> torch.Tensor:
    mask = output.timestep_mask
    lengths = mask.sum(
        dim=-1,
        keepdim=True,
    )
    safe_lengths = lengths.clamp_min(
        1
    )

    weights = (
        mask.to(
            dtype=output.dtype
        )
        / safe_lengths.to(
            dtype=output.dtype
        )
    )

    return weights.unsqueeze(
        1
    )


def _expected_last_valid_weights(
    output: TemporalPoolingOutput,
) -> torch.Tensor:
    mask = output.timestep_mask
    node_count = output.node_count
    sequence_length = output.sequence_length

    indices = torch.arange(
        sequence_length,
        dtype=torch.long,
        device=mask.device,
    ).unsqueeze(
        0
    ).expand_as(
        mask
    )

    last_indices = torch.where(
        mask,
        indices,
        torch.full_like(
            indices,
            -1,
        ),
    ).max(
        dim=-1
    ).values

    weights = torch.zeros(
        (
            node_count,
            sequence_length,
        ),
        dtype=output.dtype,
        device=output.device,
    )

    nonempty = (
        last_indices
        >= 0
    )

    if bool(
        nonempty.any().item()
    ):
        rows = torch.nonzero(
            nonempty,
            as_tuple=False,
        ).flatten()
        weights[
            rows,
            last_indices[
                rows
            ],
        ] = 1.0

    return weights.unsqueeze(
        1
    )


def _expected_deterministic_weights(
    output: TemporalPoolingOutput,
) -> torch.Tensor | None:
    if output.pooling_kind == TemporalPoolingKind.MASKED_MEAN:
        return _expected_masked_mean_weights(
            output
        )

    if output.pooling_kind == TemporalPoolingKind.LAST_VALID:
        return _expected_last_valid_weights(
            output
        )

    return None


# =============================================================================
# Last-valid missingness diagnostics
# =============================================================================


def _last_valid_selection_diagnostics(
    output: TemporalPoolingOutput,
) -> tuple[
    int | None,
    bool,
    int | None,
    float | None,
]:
    if output.pooling_kind != TemporalPoolingKind.LAST_VALID:
        return (
            None,
            False,
            None,
            None,
        )

    expected = _expected_last_valid_weights(
        output
    )
    actual_indices = (
        output
        .pooling_weights
        .squeeze(
            1
        )
        .argmax(
            dim=-1
        )
    )
    expected_indices = (
        expected
        .squeeze(
            1
        )
        .argmax(
            dim=-1
        )
    )

    nonempty = (
        output.valid_lengths
        > 0
    )

    mismatch_count = int(
        (
            nonempty
            & (
                actual_indices
                != expected_indices
            )
        )
        .sum()
        .item()
    )

    feature_observed_mask = (
        output
        .source_history
        .feature_observed_mask
    )

    if feature_observed_mask is None:
        return (
            mismatch_count,
            False,
            None,
            None,
        )

    nonempty_count = int(
        nonempty.sum().item()
    )

    if nonempty_count == 0:
        return (
            mismatch_count,
            True,
            0,
            0.0,
        )

    rows = torch.nonzero(
        nonempty,
        as_tuple=False,
    ).flatten()

    selected = feature_observed_mask[
        rows,
        actual_indices[
            rows
        ],
    ]

    selected_all_missing = (
        ~selected.any(
            dim=-1
        )
    )
    count = int(
        selected_all_missing
        .sum()
        .item()
    )

    return (
        mismatch_count,
        True,
        count,
        float(
            count
        )
        / float(
            nonempty_count
        ),
    )


# =============================================================================
# Public diagnostic functions
# =============================================================================


def diagnose_sequence_baseline(
    module: nn.Module,
    output: TemporalSequenceEncoding,
    *,
    pre_mask_output: torch.Tensor | None = None,
    component_name: str | None = None,
) -> BaselineSequenceDiagnostics:
    """
    Compute detached diagnostics for one sequence-baseline output.

    ``pre_mask_output`` is optional because the current forward contracts do
    not expose internal pre-mask tensors. Supplying it is useful in explicit
    debug or evaluation runs.
    """

    if not isinstance(
        module,
        nn.Module,
    ):
        raise TypeError(
            "module must be a torch.nn.Module."
        )

    if not isinstance(
        output,
        TemporalSequenceEncoding,
    ):
        raise TypeError(
            "output must be a TemporalSequenceEncoding."
        )

    if component_name is None:
        component_name = (
            module
            .__class__
            .__name__
        )

    _require_nonempty_string(
        "component_name",
        component_name,
    )

    parameter_count, trainable_count = (
        count_module_parameters(
            module
        )
    )

    encoded = (
        output
        .encoded_sequence
        .detach()
    )
    mask = (
        output
        .timestep_mask
        .detach()
    )
    valid_lengths = (
        output
        .valid_lengths
        .detach()
    )

    if pre_mask_output is not None:
        if not isinstance(
            pre_mask_output,
            torch.Tensor,
        ):
            raise TypeError(
                "pre_mask_output must be a torch.Tensor or None."
            )

        if pre_mask_output.shape != encoded.shape:
            raise ValueError(
                "pre_mask_output must have the same shape as "
                "output.encoded_sequence."
            )

        nonfinite_pre_mask_count = (
            _tensor_nonfinite_count(
                pre_mask_output.detach()
            )
        )
    else:
        nonfinite_pre_mask_count = None

    padded = (
        ~mask
    ).unsqueeze(
        -1
    ).expand_as(
        encoded
    )
    padded_values = encoded[
        padded
    ]
    valid_vectors = encoded[
        mask
    ]

    (
        mean_valid_norm,
        max_valid_norm,
    ) = _mean_and_max_vector_norm(
        valid_vectors
    )

    zero_history_count = int(
        (
            valid_lengths
            == 0
        )
        .sum()
        .item()
    )

    valid_timestep_count = int(
        mask.sum().item()
    )

    max_abs_padded_output = (
        _maximum_absolute(
            padded_values
        )
    )

    return BaselineSequenceDiagnostics(
        component_name=component_name,
        node_count=output.node_count,
        sequence_length=output.sequence_length,
        hidden_dim=output.hidden_dim,
        parameter_count=parameter_count,
        trainable_parameter_count=trainable_count,
        module_training=module.training,
        zero_history_count=zero_history_count,
        nonempty_history_count=(
            output.node_count
            - zero_history_count
        ),
        valid_timestep_count=valid_timestep_count,
        padded_timestep_count=(
            output.node_count
            * output.sequence_length
            - valid_timestep_count
        ),
        valid_length_min=int(
            valid_lengths.min().item()
        )
        if valid_lengths.numel() > 0
        else 0,
        valid_length_mean=float(
            valid_lengths
            .to(
                dtype=torch.float64
            )
            .mean()
            .item()
        )
        if valid_lengths.numel() > 0
        else 0.0,
        valid_length_max=int(
            valid_lengths.max().item()
        )
        if valid_lengths.numel() > 0
        else 0,
        pre_mask_output_supplied=(
            pre_mask_output is not None
        ),
        nonfinite_pre_mask_count=(
            nonfinite_pre_mask_count
        ),
        nonfinite_output_count=(
            _tensor_nonfinite_count(
                encoded
            )
        ),
        max_abs_padded_output=(
            max_abs_padded_output
        ),
        mean_valid_vector_l2_norm=(
            mean_valid_norm
        ),
        max_valid_vector_l2_norm=(
            max_valid_norm
        ),
        exact_zero_padding=(
            max_abs_padded_output
            == 0.0
        ),
        output_requires_grad=(
            output
            .encoded_sequence
            .requires_grad
        ),
        architecture_fingerprint=(
            output.architecture_fingerprint
        ),
        source_lineage_fingerprint=(
            output
            .source_history
            .lineage_fingerprint()
        ),
        output_lineage_fingerprint=(
            output.lineage_fingerprint()
        ),
    )


def diagnose_pooling_baseline(
    module: nn.Module,
    output: TemporalPoolingOutput,
    *,
    component_name: str | None = None,
) -> BaselinePoolingDiagnostics:
    """Compute detached diagnostics for one temporal-pooling output."""

    if not isinstance(
        module,
        nn.Module,
    ):
        raise TypeError(
            "module must be a torch.nn.Module."
        )

    if not isinstance(
        output,
        TemporalPoolingOutput,
    ):
        raise TypeError(
            "output must be a TemporalPoolingOutput."
        )

    if component_name is None:
        component_name = (
            module
            .__class__
            .__name__
        )

    _require_nonempty_string(
        "component_name",
        component_name,
    )

    parameter_count, trainable_count = (
        count_module_parameters(
            module
        )
    )

    weights = (
        output
        .pooling_weights
        .detach()
    )
    pooled = (
        output
        .pooled_memory
        .detach()
    )
    encoded = (
        output
        .source_encoding
        .encoded_sequence
        .detach()
    )
    mask = (
        output
        .timestep_mask
        .detach()
    )
    valid_lengths = (
        output
        .valid_lengths
        .detach()
    )
    zero_history = (
        valid_lengths
        == 0
    )
    nonempty = (
        ~zero_history
    )

    weight_mass = weights.sum(
        dim=-1
    )
    expected_mass = (
        nonempty
        .to(
            dtype=weights.dtype
        )
        .unsqueeze(
            -1
        )
        .expand_as(
            weight_mass
        )
    )

    normalization_error = (
        weight_mass
        - expected_mass
    ).abs()

    padded_mass = (
        weights
        .abs()
        * (
            ~mask
        )
        .unsqueeze(
            1
        )
        .to(
            dtype=weights.dtype
        )
    ).sum(
        dim=-1
    )

    if bool(
        zero_history.any().item()
    ):
        zero_history_weight_mass = (
            weights[
                zero_history
            ]
            .abs()
            .sum(
                dim=-1
            )
        )
        zero_history_pooled = pooled[
            zero_history
        ]
    else:
        zero_history_weight_mass = torch.empty(
            0,
            dtype=weights.dtype,
            device=weights.device,
        )
        zero_history_pooled = torch.empty(
            (
                0,
                output.output_dim,
            ),
            dtype=pooled.dtype,
            device=pooled.device,
        )

    expected_weights = (
        _expected_deterministic_weights(
            output
        )
    )

    deterministic_weight_max_error = (
        _maximum_absolute_difference(
            weights,
            expected_weights,
        )
        if expected_weights is not None
        else None
    )

    if output.num_heads == 1:
        reconstructed = torch.bmm(
            weights,
            encoded,
        ).squeeze(
            1
        )
        pooled_reconstruction_max_error = (
            _maximum_absolute_difference(
                pooled,
                reconstructed,
            )
        )
    else:
        pooled_reconstruction_max_error = None

    (
        mean_pooled_norm,
        max_pooled_norm,
    ) = _mean_and_max_vector_norm(
        pooled
    )

    (
        last_valid_selection_mismatch_count,
        selected_status_available,
        selected_all_missing_count,
        selected_all_missing_fraction,
    ) = _last_valid_selection_diagnostics(
        output
    )

    zero_history_count = int(
        zero_history.sum().item()
    )

    return BaselinePoolingDiagnostics(
        component_name=component_name,
        pooling_kind=output.pooling_kind.value,
        node_count=output.node_count,
        sequence_length=output.sequence_length,
        output_dim=output.output_dim,
        num_heads=output.num_heads,
        parameter_count=parameter_count,
        trainable_parameter_count=trainable_count,
        module_training=module.training,
        zero_history_count=zero_history_count,
        nonempty_history_count=(
            output.node_count
            - zero_history_count
        ),
        nonfinite_weight_count=(
            _tensor_nonfinite_count(
                weights
            )
        ),
        negative_weight_count=(
            _tensor_negative_count(
                weights
            )
        ),
        nonfinite_pooled_output_count=(
            _tensor_nonfinite_count(
                pooled
            )
        ),
        max_normalization_error=(
            _maximum_absolute(
                normalization_error
            )
        ),
        max_padded_pooling_mass=(
            _maximum_absolute(
                padded_mass
            )
        ),
        max_zero_history_weight_mass=(
            _maximum_absolute(
                zero_history_weight_mass
            )
        ),
        max_abs_zero_history_pooled_memory=(
            _maximum_absolute(
                zero_history_pooled
            )
        ),
        deterministic_weight_max_error=(
            deterministic_weight_max_error
        ),
        pooled_reconstruction_max_error=(
            pooled_reconstruction_max_error
        ),
        mean_pooled_vector_l2_norm=(
            mean_pooled_norm
        ),
        max_pooled_vector_l2_norm=(
            max_pooled_norm
        ),
        last_valid_selection_mismatch_count=(
            last_valid_selection_mismatch_count
        ),
        selected_all_features_missing_status_available=(
            selected_status_available
        ),
        selected_all_features_missing_count=(
            selected_all_missing_count
        ),
        selected_all_features_missing_fraction=(
            selected_all_missing_fraction
        ),
        exact_source_encoding_identity=(
            output.source_encoding
            is output.source_sequence_encoding
        ),
        pooled_output_requires_grad=(
            output.pooled_memory.requires_grad
        ),
        architecture_fingerprint=(
            output.architecture_fingerprint
        ),
        source_lineage_fingerprint=(
            output
            .source_encoding
            .lineage_fingerprint()
        ),
        output_lineage_fingerprint=(
            output.lineage_fingerprint()
        ),
    )


def diagnose_baseline_pipeline(
    sequence_module: nn.Module,
    sequence_output: TemporalSequenceEncoding,
    *,
    pooling_module: nn.Module | None = None,
    pooling_output: TemporalPoolingOutput | None = None,
    pre_mask_output: torch.Tensor | None = None,
) -> BaselineDiagnostics:
    """
    Diagnose a sequence baseline and its optional deterministic pooler.

    ``pooling_module`` and ``pooling_output`` must either both be supplied or
    both be omitted.
    """

    if (
        pooling_module is None
    ) != (
        pooling_output is None
    ):
        raise ValueError(
            "pooling_module and pooling_output must either both be "
            "supplied or both be None."
        )

    sequence_diagnostics = (
        diagnose_sequence_baseline(
            sequence_module,
            sequence_output,
            pre_mask_output=(
                pre_mask_output
            ),
        )
    )

    if (
        pooling_module is None
        or pooling_output is None
    ):
        return BaselineDiagnostics(
            sequence=sequence_diagnostics,
            pooling=None,
            pooling_present=False,
            pooling_source_is_sequence=None,
        )

    pooling_diagnostics = (
        diagnose_pooling_baseline(
            pooling_module,
            pooling_output,
        )
    )

    return BaselineDiagnostics(
        sequence=sequence_diagnostics,
        pooling=pooling_diagnostics,
        pooling_present=True,
        pooling_source_is_sequence=(
            pooling_output.source_encoding
            is sequence_output
        ),
    )


# Compact function aliases.
collect_sequence_baseline_diagnostics = diagnose_sequence_baseline
collect_pooling_baseline_diagnostics = diagnose_pooling_baseline
collect_baseline_diagnostics = diagnose_baseline_pipeline


# =============================================================================
# Public API
# =============================================================================


__all__ = (
    "BASELINE_DIAGNOSTICS_SCHEMA_VERSION",
    "BASELINE_DIAGNOSTICS_SCIENTIFIC_INTERPRETATION",
    "BaselineSequenceDiagnostics",
    "BaselinePoolingDiagnostics",
    "BaselineDiagnostics",
    "diagnose_sequence_baseline",
    "diagnose_pooling_baseline",
    "diagnose_baseline_pipeline",
    "collect_sequence_baseline_diagnostics",
    "collect_pooling_baseline_diagnostics",
    "collect_baseline_diagnostics",
)
