"""
Explicit tensor-free diagnostics for functional message-passing stacks.

Target path:
    urban_resilience_models/
        v2_hazard_conditioned_functional_ugnn/
            functional_message_passing/
                stack/
                    diagnostics.py

This module builds descriptive, tensor-free reports from validated
``FunctionalMessagePassingStackRun`` objects.

It owns:

- stack-level state-magnitude summaries;
- per-depth summaries when the required layer output is available;
- update-to-state ratios;
- descriptive regularization inventories;
- retention and audit-coverage summaries;
- sharing-policy consistency observations;
- lineage and fingerprint inventories;
- configurable collapse, explosion, and large-update alerts;
- deterministic diagnostic-report fingerprints;
- explicit claim boundaries.

It does not own:

- stack numerical execution;
- parameter sharing;
- state rebinding;
- output retention;
- layer trace construction;
- loss reduction;
- causal attribution;
- explanation faithfulness;
- uncertainty calibration.

Availability
------------
Per-depth tensor statistics require a layer output. They are available when:

- a complete audit trace exists; or
- that depth's public layer output was retained.

When a retention policy discards a depth and no audit trace exists, the report
marks that depth unavailable rather than reconstructing or fabricating its
statistics.

The initial and final stack states are always available, so total stack drift
statistics are always reported.

Memory and gradient discipline
------------------------------
Diagnostics are explicit. Tensor summaries are computed from detached CPU
``float64`` copies and returned as ordinary Python numbers. No diagnostic
report contains tensors, modules, or autograd references.

Scientific limits
-----------------
The report is descriptive. A large update, gate penalty, state-norm change,
attention pattern, or retained pathway is not by itself evidence of:

- causal influence;
- explanation faithfulness;
- calibrated confidence;
- mechanistic identifiability;
- counterfactual validity;
- robustness outside the evaluated data distribution.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from hashlib import sha256
import json
import math
from types import MappingProxyType
from typing import Any, Final, Mapping, Sequence

import torch

from ...constants import (
    STACK_RETENTION_ALL_LAYERS,
    STACK_RETENTION_FINAL_LAYER,
    STACK_RETENTION_NONE,
    STACK_SHARING_FULLY_SHARED,
    STACK_SHARING_INDEPENDENT,
)
from .schemas import (
    FunctionalMessagePassingStackDepthRecord,
    FunctionalMessagePassingStackRun,
    FunctionalMessagePassingStackRunWithDiagnostics,
    validate_functional_message_passing_stack_run,
)


# =============================================================================
# Public identity
# =============================================================================


STACK_DIAGNOSTIC_THRESHOLDS_SCHEMA_VERSION: Final[str] = "0.1"
STACK_DEPTH_DIAGNOSTIC_SCHEMA_VERSION: Final[str] = "0.1"
STACK_DIAGNOSTIC_REPORT_SCHEMA_VERSION: Final[str] = "0.1"
STACK_DIAGNOSTICS_SCHEMA_VERSION: Final[str] = "0.1"

STACK_DIAGNOSTICS_AFFECT_NUMERICAL_RESULTS: Final[bool] = False
STACK_DIAGNOSTICS_RETAIN_AUTOGRAD_REFERENCES: Final[bool] = False
STACK_DIAGNOSTICS_RETURN_TENSORS: Final[bool] = False
STACK_DIAGNOSTICS_ESTABLISH_CAUSALITY: Final[bool] = False
STACK_DIAGNOSTICS_ESTABLISH_FAITHFULNESS: Final[bool] = False
STACK_DIAGNOSTICS_ESTABLISH_CALIBRATION: Final[bool] = False
STACK_DIAGNOSTICS_ESTABLISH_IDENTIFIABILITY: Final[bool] = False

STACK_DIAGNOSTIC_DEPTH_SOURCE_AUDIT_TRACE: Final[str] = "audit_trace"
STACK_DIAGNOSTIC_DEPTH_SOURCE_RETAINED_OUTPUT: Final[str] = (
    "retained_output"
)
STACK_DIAGNOSTIC_DEPTH_SOURCE_UNAVAILABLE: Final[str] = "unavailable"

CANONICAL_STACK_DIAGNOSTIC_DEPTH_SOURCES: Final[
    tuple[str, ...]
] = (
    STACK_DIAGNOSTIC_DEPTH_SOURCE_AUDIT_TRACE,
    STACK_DIAGNOSTIC_DEPTH_SOURCE_RETAINED_OUTPUT,
    STACK_DIAGNOSTIC_DEPTH_SOURCE_UNAVAILABLE,
)

STACK_DIAGNOSTIC_ALERT_TOTAL_STATE_COLLAPSE: Final[str] = (
    "total_state_norm_collapse"
)
STACK_DIAGNOSTIC_ALERT_TOTAL_STATE_EXPLOSION: Final[str] = (
    "total_state_norm_explosion"
)
STACK_DIAGNOSTIC_ALERT_TOTAL_UPDATE_LARGE: Final[str] = (
    "total_update_ratio_large"
)
STACK_DIAGNOSTIC_ALERT_DEPTH_STATE_COLLAPSE: Final[str] = (
    "depth_state_norm_collapse"
)
STACK_DIAGNOSTIC_ALERT_DEPTH_STATE_EXPLOSION: Final[str] = (
    "depth_state_norm_explosion"
)
STACK_DIAGNOSTIC_ALERT_DEPTH_UPDATE_LARGE: Final[str] = (
    "depth_update_ratio_large"
)
STACK_DIAGNOSTIC_ALERT_DEPTH_UPDATE_TINY: Final[str] = (
    "depth_update_ratio_tiny"
)
STACK_DIAGNOSTIC_ALERT_DEPTH_DIRECTION_REVERSAL: Final[str] = (
    "depth_state_direction_reversal"
)
STACK_DIAGNOSTIC_ALERT_INCOMPLETE_DEPTH_COVERAGE: Final[str] = (
    "incomplete_depth_diagnostic_coverage"
)
STACK_DIAGNOSTIC_ALERT_SHARED_PARAMETER_DRIFT: Final[str] = (
    "fully_shared_parameter_fingerprint_drift"
)
STACK_DIAGNOSTIC_ALERT_SHARED_ARCHITECTURE_DRIFT: Final[str] = (
    "fully_shared_architecture_fingerprint_drift"
)

STACK_DIAGNOSTIC_SCIENTIFIC_CLAIMS: Final[
    Mapping[str, bool]
] = MappingProxyType(
    {
        "descriptive_state_statistics": True,
        "descriptive_update_statistics": True,
        "descriptive_regularization_inventory": True,
        "descriptive_retention_coverage": True,
        "causal_attribution": False,
        "faithful_explanation": False,
        "calibrated_uncertainty": False,
        "counterfactual_validity": False,
        "mechanistic_identifiability": False,
        "out_of_distribution_robustness": False,
    }
)


# =============================================================================
# Generic helpers
# =============================================================================


def _to_plain_json_value(
    value: Any,
) -> Any:
    if isinstance(
        value,
        Mapping,
    ):
        return {
            str(key): _to_plain_json_value(
                child
            )
            for key, child in value.items()
        }

    if isinstance(
        value,
        (
            tuple,
            list,
        ),
    ):
        return [
            _to_plain_json_value(
                child
            )
            for child in value
        ]

    return value


def _canonical_json(
    payload: Mapping[str, Any],
) -> str:
    return json.dumps(
        _to_plain_json_value(
            payload
        ),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )



def _fingerprint(
    payload: Mapping[str, Any],
) -> str:
    return sha256(
        _canonical_json(payload).encode(
            "utf-8"
        )
    ).hexdigest()


def _require_nonempty_string(
    name: str,
    value: str,
) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(
            f"{name} must be a non-empty string."
        )


def _require_nonnegative_int(
    name: str,
    value: int,
) -> None:
    if isinstance(value, bool) or not isinstance(
        value,
        int,
    ):
        raise TypeError(
            f"{name} must be an integer."
        )

    if value < 0:
        raise ValueError(
            f"{name} must be nonnegative."
        )


def _require_positive_int(
    name: str,
    value: int,
) -> None:
    if isinstance(value, bool) or not isinstance(
        value,
        int,
    ):
        raise TypeError(
            f"{name} must be an integer."
        )

    if value <= 0:
        raise ValueError(
            f"{name} must be strictly positive."
        )


def _require_boolean(
    name: str,
    value: bool,
) -> None:
    if not isinstance(value, bool):
        raise TypeError(
            f"{name} must be Boolean."
        )


def _require_finite_float(
    name: str,
    value: float,
    *,
    strictly_positive: bool = False,
    nonnegative: bool = False,
) -> None:
    if isinstance(value, bool) or not isinstance(
        value,
        (
            int,
            float,
        ),
    ):
        raise TypeError(
            f"{name} must be numeric."
        )

    numeric = float(value)

    if not math.isfinite(
        numeric
    ):
        raise ValueError(
            f"{name} must be finite."
        )

    if strictly_positive and numeric <= 0.0:
        raise ValueError(
            f"{name} must be strictly positive."
        )

    if nonnegative and numeric < 0.0:
        raise ValueError(
            f"{name} must be nonnegative."
        )


def _require_optional_finite_float(
    name: str,
    value: float | None,
) -> None:
    if value is not None:
        _require_finite_float(
            name,
            value,
        )


def _require_string_tuple(
    name: str,
    values: tuple[str, ...],
) -> None:
    if not isinstance(
        values,
        tuple,
    ):
        raise TypeError(
            f"{name} must be a tuple."
        )

    for index, value in enumerate(
        values
    ):
        _require_nonempty_string(
            f"{name}[{index}]",
            value,
        )


def _require_fingerprint_tuple(
    name: str,
    values: tuple[str, ...],
    *,
    expected_length: int,
) -> None:
    if not isinstance(
        values,
        tuple,
    ):
        raise TypeError(
            f"{name} must be a tuple."
        )

    if len(values) != expected_length:
        raise ValueError(
            f"{name} must contain {expected_length} values."
        )

    for index, value in enumerate(
        values
    ):
        _require_nonempty_string(
            f"{name}[{index}]",
            value,
        )


def _assert_tensor_free(
    value: Any,
    *,
    path: str,
) -> None:
    if isinstance(
        value,
        torch.Tensor,
    ):
        raise TypeError(
            f"{path} must not contain tensors."
        )

    if isinstance(
        value,
        torch.nn.Module,
    ):
        raise TypeError(
            f"{path} must not contain modules."
        )

    if value is None or isinstance(
        value,
        (
            str,
            bool,
            int,
            float,
        ),
    ):
        return

    if isinstance(
        value,
        Mapping,
    ):
        for key, child in value.items():
            if not isinstance(
                key,
                str,
            ):
                raise TypeError(
                    f"{path} mapping keys must be strings."
                )
            _assert_tensor_free(
                child,
                path=f"{path}.{key}",
            )
        return

    if isinstance(
        value,
        (
            tuple,
            list,
        ),
    ):
        for index, child in enumerate(
            value
        ):
            _assert_tensor_free(
                child,
                path=f"{path}[{index}]",
            )
        return

    raise TypeError(
        f"{path} contains unsupported value type "
        f"{type(value).__name__!r}."
    )


def _freeze_tensor_free(
    value: Any,
    *,
    path: str,
) -> Any:
    _assert_tensor_free(
        value,
        path=path,
    )

    if isinstance(
        value,
        Mapping,
    ):
        return MappingProxyType(
            {
                key: _freeze_tensor_free(
                    child,
                    path=f"{path}.{key}",
                )
                for key, child in value.items()
            }
        )

    if isinstance(
        value,
        (
            tuple,
            list,
        ),
    ):
        return tuple(
            _freeze_tensor_free(
                child,
                path=f"{path}[{index}]",
            )
            for index, child in enumerate(
                value
            )
        )

    return value


def _tensor_as_cpu_float64(
    value: torch.Tensor,
    *,
    name: str,
) -> torch.Tensor:
    if not isinstance(
        value,
        torch.Tensor,
    ):
        raise TypeError(
            f"{name} must be a tensor."
        )

    if not value.dtype.is_floating_point:
        raise ValueError(
            f"{name} must use a floating-point dtype."
        )

    detached = (
        value
        .detach()
        .to(
            device="cpu",
            dtype=torch.float64,
        )
        .contiguous()
    )

    if not bool(
        torch.isfinite(
            detached
        ).all().item()
    ):
        raise FloatingPointError(
            f"{name} must contain only finite values."
        )

    return detached


def _l2_norm(
    value: torch.Tensor,
    *,
    name: str,
) -> float:
    tensor = _tensor_as_cpu_float64(
        value,
        name=name,
    )
    return float(
        torch.linalg.vector_norm(
            tensor.reshape(-1),
            ord=2,
        ).item()
    )


def _root_mean_square(
    value: torch.Tensor,
    *,
    name: str,
) -> float:
    tensor = _tensor_as_cpu_float64(
        value,
        name=name,
    )

    if tensor.numel() == 0:
        return 0.0

    return float(
        torch.sqrt(
            torch.mean(
                tensor.square()
            )
        ).item()
    )


def _mean_absolute(
    value: torch.Tensor,
    *,
    name: str,
) -> float:
    tensor = _tensor_as_cpu_float64(
        value,
        name=name,
    )

    if tensor.numel() == 0:
        return 0.0

    return float(
        tensor.abs().mean().item()
    )


def _maximum_absolute(
    value: torch.Tensor,
    *,
    name: str,
) -> float:
    tensor = _tensor_as_cpu_float64(
        value,
        name=name,
    )

    if tensor.numel() == 0:
        return 0.0

    return float(
        tensor.abs().max().item()
    )


def _safe_ratio(
    numerator: float,
    denominator: float,
    *,
    epsilon: float,
) -> float:
    _require_finite_float(
        "numerator",
        numerator,
        nonnegative=True,
    )
    _require_finite_float(
        "denominator",
        denominator,
        nonnegative=True,
    )
    _require_finite_float(
        "epsilon",
        epsilon,
        strictly_positive=True,
    )

    return float(
        numerator
        / max(
            denominator,
            epsilon,
        )
    )


def _cosine_similarity(
    left: torch.Tensor,
    right: torch.Tensor,
    *,
    epsilon: float,
    name: str,
) -> float:
    left_tensor = (
        _tensor_as_cpu_float64(
            left,
            name=f"{name}.left",
        ).reshape(-1)
    )
    right_tensor = (
        _tensor_as_cpu_float64(
            right,
            name=f"{name}.right",
        ).reshape(-1)
    )

    if left_tensor.shape != (
        right_tensor.shape
    ):
        raise ValueError(
            f"{name} tensors must have the same shape."
        )

    left_norm = float(
        torch.linalg.vector_norm(
            left_tensor,
            ord=2,
        ).item()
    )
    right_norm = float(
        torch.linalg.vector_norm(
            right_tensor,
            ord=2,
        ).item()
    )

    denominator = max(
        left_norm * right_norm,
        epsilon,
    )

    return float(
        torch.dot(
            left_tensor,
            right_tensor,
        ).item()
        / denominator
    )


def _scalar_tensor_mapping_to_floats(
    values: Mapping[str, torch.Tensor],
    *,
    name: str,
) -> Mapping[str, float]:
    if not isinstance(
        values,
        Mapping,
    ):
        raise TypeError(
            f"{name} must be a mapping."
        )

    converted: dict[str, float] = {}

    for key, value in values.items():
        _require_nonempty_string(
            f"{name} key",
            key,
        )

        if not isinstance(
            value,
            torch.Tensor,
        ):
            raise TypeError(
                f"{name}[{key!r}] must be a tensor."
            )

        if not value.dtype.is_floating_point:
            raise ValueError(
                f"{name}[{key!r}] must use a floating-point dtype."
            )

        if value.numel() != 1:
            raise ValueError(
                f"{name}[{key!r}] must contain exactly one scalar."
            )

        detached = (
            value
            .detach()
            .to(
                device="cpu",
                dtype=torch.float64,
            )
        )
        scalar = float(
            detached.reshape(-1)[0].item()
        )

        if not math.isfinite(
            scalar
        ):
            raise FloatingPointError(
                f"{name}[{key!r}] must be finite."
            )

        converted[key] = scalar

    return MappingProxyType(
        converted
    )


def _descriptive_regularization_summary(
    depth_records: tuple[
        FunctionalMessagePassingStackDepthRecord,
        ...,
    ],
) -> Mapping[str, Any]:
    per_depth: list[
        Mapping[str, Any]
    ] = []
    descriptive_sum_by_name: dict[
        str,
        float,
    ] = {}
    occurrence_count_by_name: dict[
        str,
        int,
    ] = {}

    for record in depth_records:
        values = (
            _scalar_tensor_mapping_to_floats(
                record.regularization_terms,
                name=(
                    "depth_records"
                    f"[{record.layer_index}]"
                    ".regularization_terms"
                ),
            )
        )

        per_depth.append(
            MappingProxyType(
                {
                    "layer_index": (
                        record.layer_index
                    ),
                    "values": values,
                }
            )
        )

        for name, scalar in values.items():
            descriptive_sum_by_name[
                name
            ] = (
                descriptive_sum_by_name
                .get(
                    name,
                    0.0,
                )
                + scalar
            )
            occurrence_count_by_name[
                name
            ] = (
                occurrence_count_by_name
                .get(
                    name,
                    0,
                )
                + 1
            )

    return _freeze_tensor_free(
        {
            "per_depth": tuple(
                per_depth
            ),
            "unique_names": tuple(
                sorted(
                    descriptive_sum_by_name
                )
            ),
            "descriptive_sum_by_name": dict(
                sorted(
                    descriptive_sum_by_name
                    .items()
                )
            ),
            "occurrence_count_by_name": dict(
                sorted(
                    occurrence_count_by_name
                    .items()
                )
            ),
            "training_reduction_applied": False,
            "shared_parameter_deduplication_applied": False,
        },
        path="regularization_summary",
    )


# =============================================================================
# Diagnostic thresholds
# =============================================================================


@dataclass(slots=True, frozen=True)
class StackDiagnosticThresholds:
    """
    Numerical thresholds used only to emit descriptive warning labels.
    """

    norm_epsilon: float = 1.0e-12

    collapse_output_to_input_ratio: float = 1.0e-3
    explosion_output_to_input_ratio: float = 1.0e2

    tiny_update_to_input_ratio: float = 1.0e-8
    large_update_to_input_ratio: float = 1.0e1

    direction_reversal_cosine_threshold: float = -0.25

    schema_version: str = (
        STACK_DIAGNOSTIC_THRESHOLDS_SCHEMA_VERSION
    )

    def __post_init__(self) -> None:
        _require_finite_float(
            "norm_epsilon",
            self.norm_epsilon,
            strictly_positive=True,
        )
        _require_finite_float(
            "collapse_output_to_input_ratio",
            self.collapse_output_to_input_ratio,
            strictly_positive=True,
        )
        _require_finite_float(
            "explosion_output_to_input_ratio",
            self.explosion_output_to_input_ratio,
            strictly_positive=True,
        )
        _require_finite_float(
            "tiny_update_to_input_ratio",
            self.tiny_update_to_input_ratio,
            nonnegative=True,
        )
        _require_finite_float(
            "large_update_to_input_ratio",
            self.large_update_to_input_ratio,
            strictly_positive=True,
        )
        _require_finite_float(
            "direction_reversal_cosine_threshold",
            self.direction_reversal_cosine_threshold,
        )

        if (
            self.collapse_output_to_input_ratio
            >= self.explosion_output_to_input_ratio
        ):
            raise ValueError(
                "collapse_output_to_input_ratio must be smaller than "
                "explosion_output_to_input_ratio."
            )

        if (
            self.tiny_update_to_input_ratio
            >= self.large_update_to_input_ratio
        ):
            raise ValueError(
                "tiny_update_to_input_ratio must be smaller than "
                "large_update_to_input_ratio."
            )

        if not (
            -1.0
            <= self.direction_reversal_cosine_threshold
            <= 1.0
        ):
            raise ValueError(
                "direction_reversal_cosine_threshold must lie in [-1, 1]."
            )

        _require_nonempty_string(
            "schema_version",
            self.schema_version,
        )

    def as_dict(
        self,
    ) -> dict[str, Any]:
        return {
            "schema_version": (
                self.schema_version
            ),
            "norm_epsilon": (
                float(
                    self.norm_epsilon
                )
            ),
            "collapse_output_to_input_ratio": (
                float(
                    self
                    .collapse_output_to_input_ratio
                )
            ),
            "explosion_output_to_input_ratio": (
                float(
                    self
                    .explosion_output_to_input_ratio
                )
            ),
            "tiny_update_to_input_ratio": (
                float(
                    self
                    .tiny_update_to_input_ratio
                )
            ),
            "large_update_to_input_ratio": (
                float(
                    self
                    .large_update_to_input_ratio
                )
            ),
            "direction_reversal_cosine_threshold": (
                float(
                    self
                    .direction_reversal_cosine_threshold
                )
            ),
        }

    def fingerprint(
        self,
    ) -> str:
        return _fingerprint(
            self.as_dict()
        )


# =============================================================================
# Per-depth diagnostic summary
# =============================================================================


@dataclass(slots=True, frozen=True)
class StackDepthDiagnostic:
    """
    Tensor-free descriptive summary for one executed stack depth.
    """

    layer_index: int
    available: bool
    source: str
    retained_output: bool

    source_state_l2_norm: float | None
    output_state_l2_norm: float | None
    update_l2_norm: float | None
    aggregate_l2_norm: float | None

    output_to_source_norm_ratio: float | None
    update_to_source_norm_ratio: float | None
    aggregate_to_source_norm_ratio: float | None

    source_state_rms: float | None
    output_state_rms: float | None
    source_state_mean_abs: float | None
    output_state_mean_abs: float | None
    output_state_max_abs: float | None
    source_output_cosine_similarity: float | None

    architecture_fingerprint: str
    parameter_fingerprint: str
    lineage_fingerprint: str

    regularization_values: Mapping[str, float] = field(
        default_factory=dict
    )
    alerts: tuple[str, ...] = ()

    schema_version: str = (
        STACK_DEPTH_DIAGNOSTIC_SCHEMA_VERSION
    )

    def __post_init__(self) -> None:
        _require_nonnegative_int(
            "layer_index",
            self.layer_index,
        )
        _require_boolean(
            "available",
            self.available,
        )
        _require_nonempty_string(
            "source",
            self.source,
        )

        if self.source not in (
            CANONICAL_STACK_DIAGNOSTIC_DEPTH_SOURCES
        ):
            raise ValueError(
                "source must be a recognized stack diagnostic depth source."
            )

        _require_boolean(
            "retained_output",
            self.retained_output,
        )

        numeric_fields = {
            "source_state_l2_norm": (
                self.source_state_l2_norm
            ),
            "output_state_l2_norm": (
                self.output_state_l2_norm
            ),
            "update_l2_norm": (
                self.update_l2_norm
            ),
            "aggregate_l2_norm": (
                self.aggregate_l2_norm
            ),
            "output_to_source_norm_ratio": (
                self
                .output_to_source_norm_ratio
            ),
            "update_to_source_norm_ratio": (
                self
                .update_to_source_norm_ratio
            ),
            "aggregate_to_source_norm_ratio": (
                self
                .aggregate_to_source_norm_ratio
            ),
            "source_state_rms": (
                self.source_state_rms
            ),
            "output_state_rms": (
                self.output_state_rms
            ),
            "source_state_mean_abs": (
                self.source_state_mean_abs
            ),
            "output_state_mean_abs": (
                self.output_state_mean_abs
            ),
            "output_state_max_abs": (
                self.output_state_max_abs
            ),
            "source_output_cosine_similarity": (
                self
                .source_output_cosine_similarity
            ),
        }

        for name, value in (
            numeric_fields.items()
        ):
            _require_optional_finite_float(
                name,
                value,
            )

        required_when_available = (
            self.source_state_l2_norm,
            self.output_state_l2_norm,
            self.update_l2_norm,
            self.aggregate_l2_norm,
            self.output_to_source_norm_ratio,
            self.update_to_source_norm_ratio,
            self.aggregate_to_source_norm_ratio,
            self.source_state_rms,
            self.output_state_rms,
            self.source_state_mean_abs,
            self.output_state_mean_abs,
            self.output_state_max_abs,
            self.source_output_cosine_similarity,
        )

        if self.available:
            if self.source == (
                STACK_DIAGNOSTIC_DEPTH_SOURCE_UNAVAILABLE
            ):
                raise ValueError(
                    "An available depth diagnostic cannot use source "
                    "'unavailable'."
                )

            if any(
                value is None
                for value in (
                    required_when_available
                )
            ):
                raise ValueError(
                    "Available depth diagnostics require all tensor summary "
                    "values."
                )
        else:
            if self.source != (
                STACK_DIAGNOSTIC_DEPTH_SOURCE_UNAVAILABLE
            ):
                raise ValueError(
                    "Unavailable depth diagnostics must use source "
                    "'unavailable'."
                )

            if any(
                value is not None
                for value in (
                    required_when_available
                )
            ):
                raise ValueError(
                    "Unavailable depth diagnostics must not contain tensor "
                    "summary values."
                )

        _require_nonempty_string(
            "architecture_fingerprint",
            self.architecture_fingerprint,
        )
        _require_nonempty_string(
            "parameter_fingerprint",
            self.parameter_fingerprint,
        )
        _require_nonempty_string(
            "lineage_fingerprint",
            self.lineage_fingerprint,
        )

        frozen_regularization = (
            _freeze_tensor_free(
                dict(
                    self.regularization_values
                ),
                path="regularization_values",
            )
        )
        object.__setattr__(
            self,
            "regularization_values",
            frozen_regularization,
        )

        _require_string_tuple(
            "alerts",
            self.alerts,
        )
        _require_nonempty_string(
            "schema_version",
            self.schema_version,
        )

    def as_dict(
        self,
    ) -> dict[str, Any]:
        return {
            "schema_version": (
                self.schema_version
            ),
            "layer_index": (
                self.layer_index
            ),
            "available": (
                self.available
            ),
            "source": (
                self.source
            ),
            "retained_output": (
                self.retained_output
            ),
            "source_state_l2_norm": (
                self.source_state_l2_norm
            ),
            "output_state_l2_norm": (
                self.output_state_l2_norm
            ),
            "update_l2_norm": (
                self.update_l2_norm
            ),
            "aggregate_l2_norm": (
                self.aggregate_l2_norm
            ),
            "output_to_source_norm_ratio": (
                self
                .output_to_source_norm_ratio
            ),
            "update_to_source_norm_ratio": (
                self
                .update_to_source_norm_ratio
            ),
            "aggregate_to_source_norm_ratio": (
                self
                .aggregate_to_source_norm_ratio
            ),
            "source_state_rms": (
                self.source_state_rms
            ),
            "output_state_rms": (
                self.output_state_rms
            ),
            "source_state_mean_abs": (
                self.source_state_mean_abs
            ),
            "output_state_mean_abs": (
                self.output_state_mean_abs
            ),
            "output_state_max_abs": (
                self.output_state_max_abs
            ),
            "source_output_cosine_similarity": (
                self
                .source_output_cosine_similarity
            ),
            "architecture_fingerprint": (
                self.architecture_fingerprint
            ),
            "parameter_fingerprint": (
                self.parameter_fingerprint
            ),
            "lineage_fingerprint": (
                self.lineage_fingerprint
            ),
            "regularization_values": dict(
                self.regularization_values
            ),
            "alerts": list(
                self.alerts
            ),
        }

    def fingerprint(
        self,
    ) -> str:
        return _fingerprint(
            self.as_dict()
        )


# =============================================================================
# Complete tensor-free report
# =============================================================================


@dataclass(slots=True, frozen=True)
class FunctionalMessagePassingStackDiagnosticReport:
    """
    Complete immutable tensor-free stack diagnostic report.
    """

    num_layers: int
    num_nodes: int
    hidden_dim: int

    dtype: str
    device: str

    sharing_policy: str
    retention_policy: str
    layer_trace_mode: str
    audit_mode: bool
    training: bool

    retained_layer_indices: tuple[int, ...]
    audit_trace_available: bool
    available_depth_indices: tuple[int, ...]

    initial_state_l2_norm: float
    final_state_l2_norm: float
    total_update_l2_norm: float

    final_to_initial_norm_ratio: float
    total_update_to_initial_norm_ratio: float

    initial_state_rms: float
    final_state_rms: float
    initial_final_cosine_similarity: float

    depth_summaries: tuple[
        StackDepthDiagnostic,
        ...,
    ]

    layer_architecture_fingerprints: tuple[str, ...]
    layer_parameter_fingerprints: tuple[str, ...]
    layer_lineage_fingerprints: tuple[str, ...]

    regularization_summary: Mapping[str, Any]

    stack_architecture_fingerprint: str
    stack_parameter_fingerprint: str
    execution_contract_fingerprint: str
    lineage_fingerprint: str

    thresholds: StackDiagnosticThresholds
    alerts: tuple[str, ...]

    scientific_claims: Mapping[str, bool] = field(
        default_factory=lambda: (
            STACK_DIAGNOSTIC_SCIENTIFIC_CLAIMS
        )
    )

    schema_version: str = (
        STACK_DIAGNOSTIC_REPORT_SCHEMA_VERSION
    )

    def __post_init__(self) -> None:
        _require_positive_int(
            "num_layers",
            self.num_layers,
        )
        _require_positive_int(
            "num_nodes",
            self.num_nodes,
        )
        _require_positive_int(
            "hidden_dim",
            self.hidden_dim,
        )
        _require_nonempty_string(
            "dtype",
            self.dtype,
        )
        _require_nonempty_string(
            "device",
            self.device,
        )
        _require_nonempty_string(
            "sharing_policy",
            self.sharing_policy,
        )
        _require_nonempty_string(
            "retention_policy",
            self.retention_policy,
        )
        _require_nonempty_string(
            "layer_trace_mode",
            self.layer_trace_mode,
        )
        _require_boolean(
            "audit_mode",
            self.audit_mode,
        )
        _require_boolean(
            "training",
            self.training,
        )
        _require_boolean(
            "audit_trace_available",
            self.audit_trace_available,
        )

        if not isinstance(
            self.retained_layer_indices,
            tuple,
        ):
            raise TypeError(
                "retained_layer_indices must be a tuple."
            )

        if not isinstance(
            self.available_depth_indices,
            tuple,
        ):
            raise TypeError(
                "available_depth_indices must be a tuple."
            )

        for name, values in (
            (
                "retained_layer_indices",
                self.retained_layer_indices,
            ),
            (
                "available_depth_indices",
                self.available_depth_indices,
            ),
        ):
            previous = -1
            for position, value in enumerate(
                values
            ):
                _require_nonnegative_int(
                    f"{name}[{position}]",
                    value,
                )
                if value >= self.num_layers:
                    raise ValueError(
                        f"{name}[{position}] lies outside the stack."
                    )
                if value <= previous:
                    raise ValueError(
                        f"{name} must be strictly increasing."
                    )
                previous = value

        nonnegative_metrics = {
            "initial_state_l2_norm": (
                self.initial_state_l2_norm
            ),
            "final_state_l2_norm": (
                self.final_state_l2_norm
            ),
            "total_update_l2_norm": (
                self.total_update_l2_norm
            ),
            "final_to_initial_norm_ratio": (
                self.final_to_initial_norm_ratio
            ),
            "total_update_to_initial_norm_ratio": (
                self
                .total_update_to_initial_norm_ratio
            ),
            "initial_state_rms": (
                self.initial_state_rms
            ),
            "final_state_rms": (
                self.final_state_rms
            ),
        }

        for name, value in (
            nonnegative_metrics.items()
        ):
            _require_finite_float(
                name,
                value,
                nonnegative=True,
            )

        _require_finite_float(
            "initial_final_cosine_similarity",
            self.initial_final_cosine_similarity,
        )

        if not isinstance(
            self.depth_summaries,
            tuple,
        ):
            raise TypeError(
                "depth_summaries must be a tuple."
            )

        if len(self.depth_summaries) != (
            self.num_layers
        ):
            raise ValueError(
                "depth_summaries must contain one entry per layer."
            )

        observed_available: list[int] = []

        for expected_index, summary in enumerate(
            self.depth_summaries
        ):
            if not isinstance(
                summary,
                StackDepthDiagnostic,
            ):
                raise TypeError(
                    f"depth_summaries[{expected_index}] must be a "
                    "StackDepthDiagnostic."
                )

            if summary.layer_index != (
                expected_index
            ):
                raise ValueError(
                    "depth_summaries must use contiguous zero-based indices."
                )

            if summary.available:
                observed_available.append(
                    expected_index
                )

        if tuple(observed_available) != (
            self.available_depth_indices
        ):
            raise ValueError(
                "available_depth_indices differ from depth_summaries."
            )

        _require_fingerprint_tuple(
            "layer_architecture_fingerprints",
            self.layer_architecture_fingerprints,
            expected_length=self.num_layers,
        )
        _require_fingerprint_tuple(
            "layer_parameter_fingerprints",
            self.layer_parameter_fingerprints,
            expected_length=self.num_layers,
        )
        _require_fingerprint_tuple(
            "layer_lineage_fingerprints",
            self.layer_lineage_fingerprints,
            expected_length=self.num_layers,
        )

        object.__setattr__(
            self,
            "regularization_summary",
            _freeze_tensor_free(
                self.regularization_summary,
                path="regularization_summary",
            ),
        )

        for name, value in (
            (
                "stack_architecture_fingerprint",
                self.stack_architecture_fingerprint,
            ),
            (
                "stack_parameter_fingerprint",
                self.stack_parameter_fingerprint,
            ),
            (
                "execution_contract_fingerprint",
                self.execution_contract_fingerprint,
            ),
            (
                "lineage_fingerprint",
                self.lineage_fingerprint,
            ),
        ):
            _require_nonempty_string(
                name,
                value,
            )

        if not isinstance(
            self.thresholds,
            StackDiagnosticThresholds,
        ):
            raise TypeError(
                "thresholds must be a StackDiagnosticThresholds."
            )

        _require_string_tuple(
            "alerts",
            self.alerts,
        )

        frozen_claims = (
            _freeze_tensor_free(
                self.scientific_claims,
                path="scientific_claims",
            )
        )
        object.__setattr__(
            self,
            "scientific_claims",
            frozen_claims,
        )

        _require_nonempty_string(
            "schema_version",
            self.schema_version,
        )

        _assert_tensor_free(
            self.as_dict(
                include_report_fingerprint=False
            ),
            path="diagnostic_report",
        )

    @property
    def num_retained_layers(self) -> int:
        return len(
            self.retained_layer_indices
        )

    @property
    def num_available_depths(self) -> int:
        return len(
            self.available_depth_indices
        )

    @property
    def complete_depth_coverage(self) -> bool:
        return (
            self.num_available_depths
            == self.num_layers
        )

    @property
    def has_alerts(self) -> bool:
        return bool(
            self.alerts
        )

    def as_dict(
        self,
        *,
        include_report_fingerprint: bool = True,
    ) -> dict[str, Any]:
        values: dict[str, Any] = {
            "schema_version": (
                self.schema_version
            ),
            "num_layers": (
                self.num_layers
            ),
            "num_nodes": (
                self.num_nodes
            ),
            "hidden_dim": (
                self.hidden_dim
            ),
            "dtype": (
                self.dtype
            ),
            "device": (
                self.device
            ),
            "sharing_policy": (
                self.sharing_policy
            ),
            "retention_policy": (
                self.retention_policy
            ),
            "layer_trace_mode": (
                self.layer_trace_mode
            ),
            "audit_mode": (
                self.audit_mode
            ),
            "training": (
                self.training
            ),
            "retained_layer_indices": list(
                self.retained_layer_indices
            ),
            "num_retained_layers": (
                self.num_retained_layers
            ),
            "audit_trace_available": (
                self.audit_trace_available
            ),
            "available_depth_indices": list(
                self.available_depth_indices
            ),
            "num_available_depths": (
                self.num_available_depths
            ),
            "complete_depth_coverage": (
                self.complete_depth_coverage
            ),
            "initial_state_l2_norm": (
                self.initial_state_l2_norm
            ),
            "final_state_l2_norm": (
                self.final_state_l2_norm
            ),
            "total_update_l2_norm": (
                self.total_update_l2_norm
            ),
            "final_to_initial_norm_ratio": (
                self.final_to_initial_norm_ratio
            ),
            "total_update_to_initial_norm_ratio": (
                self
                .total_update_to_initial_norm_ratio
            ),
            "initial_state_rms": (
                self.initial_state_rms
            ),
            "final_state_rms": (
                self.final_state_rms
            ),
            "initial_final_cosine_similarity": (
                self.initial_final_cosine_similarity
            ),
            "depth_summaries": [
                summary.as_dict()
                for summary
                in self.depth_summaries
            ],
            "layer_architecture_fingerprints": list(
                self
                .layer_architecture_fingerprints
            ),
            "layer_parameter_fingerprints": list(
                self
                .layer_parameter_fingerprints
            ),
            "layer_lineage_fingerprints": list(
                self
                .layer_lineage_fingerprints
            ),
            "regularization_summary": dict(
                self.regularization_summary
            ),
            "stack_architecture_fingerprint": (
                self.stack_architecture_fingerprint
            ),
            "stack_parameter_fingerprint": (
                self.stack_parameter_fingerprint
            ),
            "execution_contract_fingerprint": (
                self.execution_contract_fingerprint
            ),
            "lineage_fingerprint": (
                self.lineage_fingerprint
            ),
            "thresholds": (
                self.thresholds.as_dict()
            ),
            "alerts": list(
                self.alerts
            ),
            "has_alerts": (
                self.has_alerts
            ),
            "scientific_claims": dict(
                self.scientific_claims
            ),
        }

        if include_report_fingerprint:
            values[
                "report_fingerprint"
            ] = self.report_fingerprint()

        return values

    def report_fingerprint(
        self,
    ) -> str:
        return _fingerprint(
            self.as_dict(
                include_report_fingerprint=False
            )
        )

    def public_report(
        self,
    ) -> Mapping[str, Any]:
        return _freeze_tensor_free(
            self.as_dict(),
            path="diagnostic_report",
        )


# =============================================================================
# Depth-summary construction
# =============================================================================


@dataclass(slots=True, frozen=True)
class _DepthTensorView:
    layer_index: int
    source: str
    retained_output: bool

    source_state: torch.Tensor
    output_state: torch.Tensor
    node_aggregate: torch.Tensor

    architecture_fingerprint: str
    parameter_fingerprint: str
    lineage_fingerprint: str

    regularization_terms: Mapping[
        str,
        torch.Tensor,
    ]


def _depth_tensor_views(
    run: FunctionalMessagePassingStackRun,
) -> Mapping[int, _DepthTensorView]:
    internal = run.internal_output
    audit_trace = internal.audit_trace

    views: dict[
        int,
        _DepthTensorView,
    ] = {}

    retained_indices = set(
        run.public_output
        .retained_layer_indices
    )

    if audit_trace is not None:
        for layer_run in (
            audit_trace.layer_runs
        ):
            index = layer_run.layer_index
            views[index] = _DepthTensorView(
                layer_index=index,
                source=(
                    STACK_DIAGNOSTIC_DEPTH_SOURCE_AUDIT_TRACE
                ),
                retained_output=(
                    index in retained_indices
                ),
                source_state=(
                    layer_run
                    .source_inputs
                    .node_state
                    .fused_state
                ),
                output_state=(
                    layer_run
                    .public_output
                    .updated_node_state
                ),
                node_aggregate=(
                    layer_run
                    .public_output
                    .node_aggregate
                ),
                architecture_fingerprint=(
                    layer_run
                    .public_output
                    .encoder_architecture_fingerprint
                ),
                parameter_fingerprint=(
                    layer_run
                    .internal_output
                    .layer_parameter_fingerprint
                ),
                lineage_fingerprint=(
                    layer_run
                    .public_output
                    .lineage_fingerprint
                ),
                regularization_terms=(
                    layer_run
                    .public_output
                    .regularization_terms
                ),
            )

        return MappingProxyType(
            views
        )

    for output in (
        run.public_output
        .retained_layer_outputs
    ):
        index = output.layer_index
        record = (
            internal.depth_records[index]
        )

        views[index] = _DepthTensorView(
            layer_index=index,
            source=(
                STACK_DIAGNOSTIC_DEPTH_SOURCE_RETAINED_OUTPUT
            ),
            retained_output=True,
            source_state=(
                output
                .source_inputs
                .node_state
                .fused_state
            ),
            output_state=(
                output.updated_node_state
            ),
            node_aggregate=(
                output.node_aggregate
            ),
            architecture_fingerprint=(
                record
                .output_architecture_fingerprint
            ),
            parameter_fingerprint=(
                record
                .output_parameter_fingerprint
            ),
            lineage_fingerprint=(
                record
                .output_lineage_fingerprint
            ),
            regularization_terms=(
                output.regularization_terms
            ),
        )

    return MappingProxyType(
        views
    )


def _depth_alerts(
    *,
    source_state_l2_norm: float,
    output_state_l2_norm: float,
    update_to_source_norm_ratio: float,
    source_output_cosine_similarity: float,
    thresholds: StackDiagnosticThresholds,
) -> tuple[str, ...]:
    output_ratio = _safe_ratio(
        output_state_l2_norm,
        source_state_l2_norm,
        epsilon=(
            thresholds.norm_epsilon
        ),
    )

    alerts: list[str] = []

    if output_ratio <= (
        thresholds
        .collapse_output_to_input_ratio
    ):
        alerts.append(
            STACK_DIAGNOSTIC_ALERT_DEPTH_STATE_COLLAPSE
        )

    if output_ratio >= (
        thresholds
        .explosion_output_to_input_ratio
    ):
        alerts.append(
            STACK_DIAGNOSTIC_ALERT_DEPTH_STATE_EXPLOSION
        )

    if update_to_source_norm_ratio >= (
        thresholds
        .large_update_to_input_ratio
    ):
        alerts.append(
            STACK_DIAGNOSTIC_ALERT_DEPTH_UPDATE_LARGE
        )

    if update_to_source_norm_ratio <= (
        thresholds
        .tiny_update_to_input_ratio
    ):
        alerts.append(
            STACK_DIAGNOSTIC_ALERT_DEPTH_UPDATE_TINY
        )

    if source_output_cosine_similarity <= (
        thresholds
        .direction_reversal_cosine_threshold
    ):
        alerts.append(
            STACK_DIAGNOSTIC_ALERT_DEPTH_DIRECTION_REVERSAL
        )

    return tuple(
        alerts
    )


def _available_depth_summary(
    view: _DepthTensorView,
    *,
    thresholds: StackDiagnosticThresholds,
) -> StackDepthDiagnostic:
    source_state = view.source_state
    output_state = view.output_state
    node_aggregate = view.node_aggregate
    update = (
        output_state
        - source_state
    )

    source_norm = _l2_norm(
        source_state,
        name=(
            f"depth_{view.layer_index}.source_state"
        ),
    )
    output_norm = _l2_norm(
        output_state,
        name=(
            f"depth_{view.layer_index}.output_state"
        ),
    )
    update_norm = _l2_norm(
        update,
        name=(
            f"depth_{view.layer_index}.update"
        ),
    )
    aggregate_norm = _l2_norm(
        node_aggregate,
        name=(
            f"depth_{view.layer_index}.node_aggregate"
        ),
    )

    output_ratio = _safe_ratio(
        output_norm,
        source_norm,
        epsilon=(
            thresholds.norm_epsilon
        ),
    )
    update_ratio = _safe_ratio(
        update_norm,
        source_norm,
        epsilon=(
            thresholds.norm_epsilon
        ),
    )
    aggregate_ratio = _safe_ratio(
        aggregate_norm,
        source_norm,
        epsilon=(
            thresholds.norm_epsilon
        ),
    )
    cosine = _cosine_similarity(
        source_state,
        output_state,
        epsilon=(
            thresholds.norm_epsilon
        ),
        name=(
            f"depth_{view.layer_index}.source_output"
        ),
    )

    alerts = _depth_alerts(
        source_state_l2_norm=(
            source_norm
        ),
        output_state_l2_norm=(
            output_norm
        ),
        update_to_source_norm_ratio=(
            update_ratio
        ),
        source_output_cosine_similarity=(
            cosine
        ),
        thresholds=thresholds,
    )

    return StackDepthDiagnostic(
        layer_index=(
            view.layer_index
        ),
        available=True,
        source=view.source,
        retained_output=(
            view.retained_output
        ),
        source_state_l2_norm=(
            source_norm
        ),
        output_state_l2_norm=(
            output_norm
        ),
        update_l2_norm=(
            update_norm
        ),
        aggregate_l2_norm=(
            aggregate_norm
        ),
        output_to_source_norm_ratio=(
            output_ratio
        ),
        update_to_source_norm_ratio=(
            update_ratio
        ),
        aggregate_to_source_norm_ratio=(
            aggregate_ratio
        ),
        source_state_rms=(
            _root_mean_square(
                source_state,
                name=(
                    f"depth_{view.layer_index}.source_state"
                ),
            )
        ),
        output_state_rms=(
            _root_mean_square(
                output_state,
                name=(
                    f"depth_{view.layer_index}.output_state"
                ),
            )
        ),
        source_state_mean_abs=(
            _mean_absolute(
                source_state,
                name=(
                    f"depth_{view.layer_index}.source_state"
                ),
            )
        ),
        output_state_mean_abs=(
            _mean_absolute(
                output_state,
                name=(
                    f"depth_{view.layer_index}.output_state"
                ),
            )
        ),
        output_state_max_abs=(
            _maximum_absolute(
                output_state,
                name=(
                    f"depth_{view.layer_index}.output_state"
                ),
            )
        ),
        source_output_cosine_similarity=(
            cosine
        ),
        architecture_fingerprint=(
            view.architecture_fingerprint
        ),
        parameter_fingerprint=(
            view.parameter_fingerprint
        ),
        lineage_fingerprint=(
            view.lineage_fingerprint
        ),
        regularization_values=(
            _scalar_tensor_mapping_to_floats(
                view.regularization_terms,
                name=(
                    f"depth_{view.layer_index}"
                    ".regularization_terms"
                ),
            )
        ),
        alerts=alerts,
    )


def _unavailable_depth_summary(
    record: FunctionalMessagePassingStackDepthRecord,
) -> StackDepthDiagnostic:
    return StackDepthDiagnostic(
        layer_index=(
            record.layer_index
        ),
        available=False,
        source=(
            STACK_DIAGNOSTIC_DEPTH_SOURCE_UNAVAILABLE
        ),
        retained_output=False,
        source_state_l2_norm=None,
        output_state_l2_norm=None,
        update_l2_norm=None,
        aggregate_l2_norm=None,
        output_to_source_norm_ratio=None,
        update_to_source_norm_ratio=None,
        aggregate_to_source_norm_ratio=None,
        source_state_rms=None,
        output_state_rms=None,
        source_state_mean_abs=None,
        output_state_mean_abs=None,
        output_state_max_abs=None,
        source_output_cosine_similarity=None,
        architecture_fingerprint=(
            record
            .output_architecture_fingerprint
        ),
        parameter_fingerprint=(
            record
            .output_parameter_fingerprint
        ),
        lineage_fingerprint=(
            record
            .output_lineage_fingerprint
        ),
        regularization_values=(
            _scalar_tensor_mapping_to_floats(
                record.regularization_terms,
                name=(
                    "depth_records"
                    f"[{record.layer_index}]"
                    ".regularization_terms"
                ),
            )
        ),
        alerts=(),
    )


# =============================================================================
# Report construction
# =============================================================================


def _stack_level_alerts(
    *,
    initial_norm: float,
    final_norm: float,
    update_ratio: float,
    complete_depth_coverage: bool,
    sharing_policy: str,
    layer_architecture_fingerprints: tuple[str, ...],
    layer_parameter_fingerprints: tuple[str, ...],
    thresholds: StackDiagnosticThresholds,
) -> tuple[str, ...]:
    final_ratio = _safe_ratio(
        final_norm,
        initial_norm,
        epsilon=(
            thresholds.norm_epsilon
        ),
    )

    alerts: list[str] = []

    if final_ratio <= (
        thresholds
        .collapse_output_to_input_ratio
    ):
        alerts.append(
            STACK_DIAGNOSTIC_ALERT_TOTAL_STATE_COLLAPSE
        )

    if final_ratio >= (
        thresholds
        .explosion_output_to_input_ratio
    ):
        alerts.append(
            STACK_DIAGNOSTIC_ALERT_TOTAL_STATE_EXPLOSION
        )

    if update_ratio >= (
        thresholds
        .large_update_to_input_ratio
    ):
        alerts.append(
            STACK_DIAGNOSTIC_ALERT_TOTAL_UPDATE_LARGE
        )

    if not complete_depth_coverage:
        alerts.append(
            STACK_DIAGNOSTIC_ALERT_INCOMPLETE_DEPTH_COVERAGE
        )

    if sharing_policy == (
        STACK_SHARING_FULLY_SHARED
    ):
        if len(
            set(
                layer_architecture_fingerprints
            )
        ) != 1:
            alerts.append(
                STACK_DIAGNOSTIC_ALERT_SHARED_ARCHITECTURE_DRIFT
            )

        if len(
            set(
                layer_parameter_fingerprints
            )
        ) != 1:
            alerts.append(
                STACK_DIAGNOSTIC_ALERT_SHARED_PARAMETER_DRIFT
            )

    return tuple(
        alerts
    )


def build_functional_message_passing_stack_diagnostic_report(
    run: FunctionalMessagePassingStackRun,
    *,
    thresholds: StackDiagnosticThresholds | None = None,
) -> FunctionalMessagePassingStackDiagnosticReport:
    """
    Build one validated tensor-free diagnostic report.
    """

    validate_functional_message_passing_stack_run(
        run
    )

    resolved_thresholds = (
        StackDiagnosticThresholds()
        if thresholds is None
        else thresholds
    )

    if not isinstance(
        resolved_thresholds,
        StackDiagnosticThresholds,
    ):
        raise TypeError(
            "thresholds must be a StackDiagnosticThresholds or None."
        )

    stack_inputs = run.stack_inputs
    internal = run.internal_output
    public = run.public_output

    initial_state = (
        run
        .source_inputs
        .node_state
        .fused_state
    )
    final_state = (
        run.final_node_state
    )
    total_update = (
        final_state
        - initial_state
    )

    initial_norm = _l2_norm(
        initial_state,
        name="initial_state",
    )
    final_norm = _l2_norm(
        final_state,
        name="final_state",
    )
    update_norm = _l2_norm(
        total_update,
        name="total_update",
    )

    final_ratio = _safe_ratio(
        final_norm,
        initial_norm,
        epsilon=(
            resolved_thresholds
            .norm_epsilon
        ),
    )
    update_ratio = _safe_ratio(
        update_norm,
        initial_norm,
        epsilon=(
            resolved_thresholds
            .norm_epsilon
        ),
    )

    tensor_views = _depth_tensor_views(
        run
    )

    depth_summaries: list[
        StackDepthDiagnostic
    ] = []

    for record in (
        internal.depth_records
    ):
        view = tensor_views.get(
            record.layer_index
        )

        if view is None:
            summary = (
                _unavailable_depth_summary(
                    record
                )
            )
        else:
            summary = (
                _available_depth_summary(
                    view,
                    thresholds=(
                        resolved_thresholds
                    ),
                )
            )

        depth_summaries.append(
            summary
        )

    summaries = tuple(
        depth_summaries
    )
    available_indices = tuple(
        summary.layer_index
        for summary in summaries
        if summary.available
    )
    complete_coverage = (
        len(available_indices)
        == stack_inputs.num_layers
    )

    layer_architecture_fingerprints = (
        internal
        .layer_architecture_fingerprints
    )
    layer_parameter_fingerprints = (
        internal
        .layer_parameter_fingerprints
    )
    layer_lineage_fingerprints = (
        internal
        .layer_lineage_fingerprints
    )

    stack_alerts = list(
        _stack_level_alerts(
            initial_norm=initial_norm,
            final_norm=final_norm,
            update_ratio=update_ratio,
            complete_depth_coverage=(
                complete_coverage
            ),
            sharing_policy=(
                stack_inputs.sharing_policy
            ),
            layer_architecture_fingerprints=(
                layer_architecture_fingerprints
            ),
            layer_parameter_fingerprints=(
                layer_parameter_fingerprints
            ),
            thresholds=(
                resolved_thresholds
            ),
        )
    )

    for summary in summaries:
        stack_alerts.extend(
            f"layer_{summary.layer_index}:{alert}"
            for alert in summary.alerts
        )

    return FunctionalMessagePassingStackDiagnosticReport(
        num_layers=(
            stack_inputs.num_layers
        ),
        num_nodes=(
            stack_inputs.num_nodes
        ),
        hidden_dim=(
            stack_inputs.hidden_dim
        ),
        dtype=str(
            stack_inputs.dtype
        ),
        device=str(
            stack_inputs.device
        ),
        sharing_policy=(
            stack_inputs.sharing_policy
        ),
        retention_policy=(
            stack_inputs.retention_policy
        ),
        layer_trace_mode=(
            stack_inputs.layer_trace_mode
        ),
        audit_mode=(
            stack_inputs.audit_mode
        ),
        training=(
            stack_inputs.training
        ),
        retained_layer_indices=(
            public.retained_layer_indices
        ),
        audit_trace_available=(
            internal.audit_trace
            is not None
        ),
        available_depth_indices=(
            available_indices
        ),
        initial_state_l2_norm=(
            initial_norm
        ),
        final_state_l2_norm=(
            final_norm
        ),
        total_update_l2_norm=(
            update_norm
        ),
        final_to_initial_norm_ratio=(
            final_ratio
        ),
        total_update_to_initial_norm_ratio=(
            update_ratio
        ),
        initial_state_rms=(
            _root_mean_square(
                initial_state,
                name="initial_state",
            )
        ),
        final_state_rms=(
            _root_mean_square(
                final_state,
                name="final_state",
            )
        ),
        initial_final_cosine_similarity=(
            _cosine_similarity(
                initial_state,
                final_state,
                epsilon=(
                    resolved_thresholds
                    .norm_epsilon
                ),
                name=(
                    "initial_final_state"
                ),
            )
        ),
        depth_summaries=summaries,
        layer_architecture_fingerprints=(
            layer_architecture_fingerprints
        ),
        layer_parameter_fingerprints=(
            layer_parameter_fingerprints
        ),
        layer_lineage_fingerprints=(
            layer_lineage_fingerprints
        ),
        regularization_summary=(
            _descriptive_regularization_summary(
                internal.depth_records
            )
        ),
        stack_architecture_fingerprint=(
            internal
            .stack_architecture_fingerprint
        ),
        stack_parameter_fingerprint=(
            internal
            .stack_parameter_fingerprint
        ),
        execution_contract_fingerprint=(
            internal
            .execution_contract_fingerprint
        ),
        lineage_fingerprint=(
            internal.lineage_fingerprint
        ),
        thresholds=(
            resolved_thresholds
        ),
        alerts=tuple(
            stack_alerts
        ),
    )


def build_stack_diagnostic_report(
    run: FunctionalMessagePassingStackRun,
    *,
    thresholds: StackDiagnosticThresholds | None = None,
) -> FunctionalMessagePassingStackDiagnosticReport:
    return (
        build_functional_message_passing_stack_diagnostic_report(
            run,
            thresholds=thresholds,
        )
    )


def validate_functional_message_passing_stack_diagnostic_report(
    report: FunctionalMessagePassingStackDiagnosticReport,
) -> None:
    if not isinstance(
        report,
        FunctionalMessagePassingStackDiagnosticReport,
    ):
        raise TypeError(
            "report must be a "
            "FunctionalMessagePassingStackDiagnosticReport."
        )

    _assert_tensor_free(
        report.as_dict(),
        path="diagnostic_report",
    )

    expected = _fingerprint(
        report.as_dict(
            include_report_fingerprint=False
        )
    )

    if report.report_fingerprint() != (
        expected
    ):
        raise ValueError(
            "Diagnostic report fingerprint is inconsistent."
        )


def validate_stack_diagnostic_report(
    report: FunctionalMessagePassingStackDiagnosticReport,
) -> None:
    validate_functional_message_passing_stack_diagnostic_report(
        report
    )


# =============================================================================
# Diagnostic orchestrator
# =============================================================================


@dataclass(slots=True, frozen=True)
class FunctionalMessagePassingStackDiagnostics:
    """
    Explicit diagnostics orchestrator with immutable alert thresholds.
    """

    thresholds: StackDiagnosticThresholds = field(
        default_factory=StackDiagnosticThresholds
    )
    enabled: bool = True

    schema_version: str = (
        STACK_DIAGNOSTICS_SCHEMA_VERSION
    )

    def __post_init__(self) -> None:
        if not isinstance(
            self.thresholds,
            StackDiagnosticThresholds,
        ):
            raise TypeError(
                "thresholds must be a StackDiagnosticThresholds."
            )

        _require_boolean(
            "enabled",
            self.enabled,
        )
        _require_nonempty_string(
            "schema_version",
            self.schema_version,
        )

    def architecture_dict(
        self,
    ) -> dict[str, Any]:
        return {
            "schema_version": (
                self.schema_version
            ),
            "enabled": (
                self.enabled
            ),
            "thresholds": (
                self.thresholds.as_dict()
            ),
            "affects_numerical_results": (
                STACK_DIAGNOSTICS_AFFECT_NUMERICAL_RESULTS
            ),
            "returns_tensors": (
                STACK_DIAGNOSTICS_RETURN_TENSORS
            ),
            "retains_autograd_references": (
                STACK_DIAGNOSTICS_RETAIN_AUTOGRAD_REFERENCES
            ),
        }

    def architecture_fingerprint(
        self,
    ) -> str:
        return _fingerprint(
            self.architecture_dict()
        )

    def report(
        self,
        run: FunctionalMessagePassingStackRun,
    ) -> FunctionalMessagePassingStackDiagnosticReport:
        if not self.enabled:
            raise RuntimeError(
                "Stack diagnostics are disabled."
            )

        return (
            build_functional_message_passing_stack_diagnostic_report(
                run,
                thresholds=(
                    self.thresholds
                ),
            )
        )

    def public_report(
        self,
        run: FunctionalMessagePassingStackRun,
    ) -> Mapping[str, Any]:
        return self.report(
            run
        ).public_report()

    def run_with_diagnostics(
        self,
        run: FunctionalMessagePassingStackRun,
    ) -> FunctionalMessagePassingStackRunWithDiagnostics:
        return (
            FunctionalMessagePassingStackRunWithDiagnostics(
                run=run,
                diagnostic_report=(
                    self.public_report(
                        run
                    )
                ),
            )
        )


# =============================================================================
# Compact aliases
# =============================================================================


StackDiagnostics = FunctionalMessagePassingStackDiagnostics
StackDiagnosticReport = (
    FunctionalMessagePassingStackDiagnosticReport
)
DepthDiagnostic = StackDepthDiagnostic
DiagnosticThresholds = StackDiagnosticThresholds

build_diagnostic_report = (
    build_functional_message_passing_stack_diagnostic_report
)
validate_diagnostic_report = (
    validate_functional_message_passing_stack_diagnostic_report
)


__all__ = (
    # Public identity.
    "STACK_DIAGNOSTIC_THRESHOLDS_SCHEMA_VERSION",
    "STACK_DEPTH_DIAGNOSTIC_SCHEMA_VERSION",
    "STACK_DIAGNOSTIC_REPORT_SCHEMA_VERSION",
    "STACK_DIAGNOSTICS_SCHEMA_VERSION",
    "STACK_DIAGNOSTICS_AFFECT_NUMERICAL_RESULTS",
    "STACK_DIAGNOSTICS_RETAIN_AUTOGRAD_REFERENCES",
    "STACK_DIAGNOSTICS_RETURN_TENSORS",
    "STACK_DIAGNOSTICS_ESTABLISH_CAUSALITY",
    "STACK_DIAGNOSTICS_ESTABLISH_FAITHFULNESS",
    "STACK_DIAGNOSTICS_ESTABLISH_CALIBRATION",
    "STACK_DIAGNOSTICS_ESTABLISH_IDENTIFIABILITY",
    "STACK_DIAGNOSTIC_DEPTH_SOURCE_AUDIT_TRACE",
    "STACK_DIAGNOSTIC_DEPTH_SOURCE_RETAINED_OUTPUT",
    "STACK_DIAGNOSTIC_DEPTH_SOURCE_UNAVAILABLE",
    "CANONICAL_STACK_DIAGNOSTIC_DEPTH_SOURCES",
    "STACK_DIAGNOSTIC_ALERT_TOTAL_STATE_COLLAPSE",
    "STACK_DIAGNOSTIC_ALERT_TOTAL_STATE_EXPLOSION",
    "STACK_DIAGNOSTIC_ALERT_TOTAL_UPDATE_LARGE",
    "STACK_DIAGNOSTIC_ALERT_DEPTH_STATE_COLLAPSE",
    "STACK_DIAGNOSTIC_ALERT_DEPTH_STATE_EXPLOSION",
    "STACK_DIAGNOSTIC_ALERT_DEPTH_UPDATE_LARGE",
    "STACK_DIAGNOSTIC_ALERT_DEPTH_UPDATE_TINY",
    "STACK_DIAGNOSTIC_ALERT_DEPTH_DIRECTION_REVERSAL",
    "STACK_DIAGNOSTIC_ALERT_INCOMPLETE_DEPTH_COVERAGE",
    "STACK_DIAGNOSTIC_ALERT_SHARED_PARAMETER_DRIFT",
    "STACK_DIAGNOSTIC_ALERT_SHARED_ARCHITECTURE_DRIFT",
    "STACK_DIAGNOSTIC_SCIENTIFIC_CLAIMS",
    # Thresholds and typed reports.
    "StackDiagnosticThresholds",
    "DiagnosticThresholds",
    "StackDepthDiagnostic",
    "DepthDiagnostic",
    "FunctionalMessagePassingStackDiagnosticReport",
    "StackDiagnosticReport",
    # Construction and validation.
    "build_functional_message_passing_stack_diagnostic_report",
    "build_stack_diagnostic_report",
    "build_diagnostic_report",
    "validate_functional_message_passing_stack_diagnostic_report",
    "validate_stack_diagnostic_report",
    "validate_diagnostic_report",
    # Orchestration.
    "FunctionalMessagePassingStackDiagnostics",
    "StackDiagnostics",
)
