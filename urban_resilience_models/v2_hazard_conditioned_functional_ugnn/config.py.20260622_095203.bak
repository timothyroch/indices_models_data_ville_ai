"""
Typed configuration system for the V2 hazard-conditioned functional UGNN.

Target path:
    urban_resilience_models/
        v2_hazard_conditioned_functional_ugnn/
            config.py

This module is the single source of truth for:

- dataset and artifact identity;
- data-pipeline expectations;
- model construction;
- node-state fusion;
- training and optimization;
- uncertainty and reporting extensions;
- explanation capture;
- runtime execution;
- experiment reproducibility.

Stable names and canonical vocabularies belong in ``constants.py``.
Tensor contracts and model outputs belong in ``schemas.py``.
Neural implementations belong in their respective subpackages.

Configuration objects are immutable. Modify them with ``replace()`` so every
new configuration is reconstructed and validated.

Validation levels
-----------------
``validate()`` checks conceptual and cross-module validity.

``assert_implemented()`` checks whether the current repository implements the
selected canonical capabilities.

``assert_construction_ready()`` additionally checks that dimensions and
registry-derived values required to instantiate the model have been resolved.

Loading older configurations
----------------------------
Configuration construction permits older version metadata for inspection and
migration. Pass ``require_current_versions=True`` when loading or validating a
configuration intended for current execution.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import (
    asdict,
    dataclass,
    field,
    fields,
    is_dataclass,
    replace as dataclass_replace,
)
import json
import math
from hashlib import sha256
from pathlib import Path
from typing import Any, Mapping, Self, Sequence, TypeVar

from .constants import (
    AGGREGATION_MEAN,
    ATTENTION_HEAD_REDUCTION_MEAN,
    ATTENTION_MODE_HAZARD_CONDITIONED,
    ATTENTION_MODE_MULTIHEAD_HAZARD_CONDITIONED,
    ATTENTION_MODE_UNIFORM,
    ATTENTION_NORMALIZATION_TARGET_NODE_RELATION,
    CANONICAL_AGGREGATION_TYPES,
    CANONICAL_ATTENTION_HEAD_REDUCTIONS,
    CANONICAL_ATTENTION_MODES,
    CANONICAL_ATTENTION_NORMALIZATION_MODES,
    CANONICAL_EDGE_NORMALIZATION_TYPES,
    CANONICAL_HAZARD_CONDITIONING_MODES,
    CANONICAL_MEMORY_ENCODER_TYPES,
    CANONICAL_MEMORY_QUERY_TYPES,
    CANONICAL_PREDICTION_HEAD_TYPES,
    CANONICAL_RELATION_GATE_ACTIVATIONS,
    CANONICAL_RELATION_GATE_SCOPES,
    CANONICAL_RELATION_NAMES,
    CANONICAL_RELATION_TRANSFORM_TYPES,
    CANONICAL_REPORTING_BIAS_TYPES,
    CANONICAL_SCOPES,
    CANONICAL_UNCERTAINTY_HEAD_TYPES,
    CANONICAL_UNCERTAINTY_METHOD_TYPES,
    CONTROL_RELATION_NAMES,
    EDGE_NORMALIZATION_NONE,
    HAZARD_CONDITIONING_EMBEDDING,
    HAZARD_CONDITIONING_EMBEDDING_SCENARIO,
    HAZARD_CONDITIONING_FULL,
    HAZARD_CONDITIONING_GATE_ATTENTION,
    HAZARD_CONDITIONING_NONE,
    HAZARD_CONDITIONING_QUERIED_MEMORY,
    HAZARD_CONDITIONING_RELATION_GATE,
    MEMORY_ENCODER_GRU,
    MEMORY_ENCODER_LAG,
    MEMORY_ENCODER_LSTM,
    MEMORY_ENCODER_NONE,
    MEMORY_ENCODER_TRANSFORMER,
    MEMORY_QUERY_HAZARD_ATTENTION,
    MEMORY_QUERY_NONE,
    MEMORY_QUERY_SCENARIO_ATTENTION,
    MODEL_FAMILY_ID,
    PREDICTION_HEAD_MULTI_HORIZON,
    PREDICTION_HEAD_NEGATIVE_BINOMIAL,
    PREDICTION_HEAD_NONNEGATIVE_REGRESSION,
    PREDICTION_HEAD_POISSON_RATE,
    PREDICTION_HEAD_RANKING,
    PREDICTION_HEAD_REGRESSION,
    RELATION_GATE_ACTIVATION_SIGMOID,
    RELATION_GATE_SCOPE_TARGET_NODE,
    RELATION_TRANSFORM_PER_RELATION,
    RELATION_TRANSFORM_SHARED,
    REPORTING_BIAS_NONE,
    SCOPE_GRAPH,
    UNCERTAINTY_HEAD_NONE,
    UNCERTAINTY_HEAD_QUANTILE,
    UNCERTAINTY_METHOD_ENSEMBLE,
    UNCERTAINTY_METHOD_MC_DROPOUT,
    UNCERTAINTY_METHOD_NONE,
    V2_0_CANDIDATE_RELATION_NAMES,
    V2_0_IMPLEMENTED_AGGREGATION_TYPES,
    V2_0_IMPLEMENTED_ATTENTION_HEAD_REDUCTIONS,
    V2_0_IMPLEMENTED_ATTENTION_MODES,
    V2_0_IMPLEMENTED_ATTENTION_NORMALIZATION_MODES,
    V2_0_IMPLEMENTED_EDGE_NORMALIZATION_TYPES,
    V2_0_IMPLEMENTED_HAZARD_CONDITIONING_MODES,
    V2_0_IMPLEMENTED_MEMORY_ENCODER_TYPES,
    V2_0_IMPLEMENTED_MEMORY_QUERY_TYPES,
    V2_0_IMPLEMENTED_PREDICTION_HEAD_TYPES,
    V2_0_IMPLEMENTED_RELATION_GATE_ACTIVATIONS,
    V2_0_IMPLEMENTED_RELATION_GATE_SCOPES,
    V2_0_IMPLEMENTED_RELATION_NAMES,
    V2_0_IMPLEMENTED_RELATION_TRANSFORM_TYPES,
    V2_0_IMPLEMENTED_REPORTING_BIAS_TYPES,
    V2_0_IMPLEMENTED_UNCERTAINTY_HEAD_TYPES,
    V2_0_IMPLEMENTED_UNCERTAINTY_METHOD_TYPES,
)
from .schemas import ContractVersions


# =============================================================================
# Local runtime vocabularies
# =============================================================================

OPTIMIZER_ADAM: str = "adam"
OPTIMIZER_ADAMW: str = "adamw"
OPTIMIZER_SGD: str = "sgd"

CANONICAL_OPTIMIZERS: tuple[str, ...] = (
    OPTIMIZER_ADAM,
    OPTIMIZER_ADAMW,
    OPTIMIZER_SGD,
)

SCHEDULER_NONE: str = "none"
SCHEDULER_REDUCE_ON_PLATEAU: str = "reduce_on_plateau"
SCHEDULER_COSINE: str = "cosine"
SCHEDULER_STEP: str = "step"

CANONICAL_SCHEDULERS: tuple[str, ...] = (
    SCHEDULER_NONE,
    SCHEDULER_REDUCE_ON_PLATEAU,
    SCHEDULER_COSINE,
    SCHEDULER_STEP,
)

LOSS_MSE: str = "mse"
LOSS_MAE: str = "mae"
LOSS_HUBER: str = "huber"
LOSS_POISSON_NLL: str = "poisson_nll"
LOSS_NEGATIVE_BINOMIAL_NLL: str = "negative_binomial_nll"
LOSS_QUANTILE: str = "quantile"
LOSS_RANKING: str = "ranking"
LOSS_MULTITASK: str = "multitask"

CANONICAL_LOSSES: tuple[str, ...] = (
    LOSS_MSE,
    LOSS_MAE,
    LOSS_HUBER,
    LOSS_POISSON_NLL,
    LOSS_NEGATIVE_BINOMIAL_NLL,
    LOSS_QUANTILE,
    LOSS_RANKING,
    LOSS_MULTITASK,
)

MONITOR_MAE: str = "validation_mae"
MONITOR_RMSE: str = "validation_rmse"
MONITOR_POISSON_DEVIANCE: str = "validation_poisson_deviance"
MONITOR_SPEARMAN: str = "validation_spearman"
MONITOR_NDCG: str = "validation_ndcg"
MONITOR_TOP_K_OVERLAP: str = "validation_top_k_overlap"
MONITOR_TOTAL_LOSS: str = "validation_loss"

CANONICAL_MONITOR_METRICS: tuple[str, ...] = (
    MONITOR_MAE,
    MONITOR_RMSE,
    MONITOR_POISSON_DEVIANCE,
    MONITOR_SPEARMAN,
    MONITOR_NDCG,
    MONITOR_TOP_K_OVERLAP,
    MONITOR_TOTAL_LOSS,
)

MONITOR_MODE_MIN: str = "min"
MONITOR_MODE_MAX: str = "max"
MONITOR_MODE_AUTO: str = "auto"

CANONICAL_MONITOR_MODES: tuple[str, ...] = (
    MONITOR_MODE_MIN,
    MONITOR_MODE_MAX,
    MONITOR_MODE_AUTO,
)

UNKNOWN_ID_POLICY_ERROR: str = "error"
UNKNOWN_ID_POLICY_USE_UNKNOWN_EMBEDDING: str = "use_unknown_embedding"

CANONICAL_UNKNOWN_ID_POLICIES: tuple[str, ...] = (
    UNKNOWN_ID_POLICY_ERROR,
    UNKNOWN_ID_POLICY_USE_UNKNOWN_EMBEDDING,
)

NODE_FUSION_CONCAT_PROJECTION: str = "concat_projection"
NODE_FUSION_PROJECTED_SUM: str = "projected_sum"
NODE_FUSION_GATED: str = "gated_fusion"
NODE_FUSION_FILM: str = "film_conditioning"

CANONICAL_NODE_FUSION_MODES: tuple[str, ...] = (
    NODE_FUSION_CONCAT_PROJECTION,
    NODE_FUSION_PROJECTED_SUM,
    NODE_FUSION_GATED,
    NODE_FUSION_FILM,
)

# Update when the corresponding fusion modules are implemented.
V2_0_IMPLEMENTED_NODE_FUSION_MODES: tuple[str, ...] = ()

PRECISION_FLOAT32: str = "float32"
PRECISION_FLOAT64: str = "float64"
PRECISION_BFLOAT16: str = "bfloat16"
PRECISION_FLOAT16: str = "float16"

CANONICAL_PRECISIONS: tuple[str, ...] = (
    PRECISION_FLOAT32,
    PRECISION_FLOAT64,
    PRECISION_BFLOAT16,
    PRECISION_FLOAT16,
)


# =============================================================================
# Generic validation helpers
# =============================================================================


def _require_nonempty_string(name: str, value: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string.")


def _require_choice(
    name: str,
    value: str,
    allowed: Sequence[str],
) -> None:
    if value not in allowed:
        raise ValueError(
            f"Unknown {name} {value!r}. Expected one of {tuple(allowed)}."
        )


def _require_finite_number(
    name: str,
    value: int | float,
) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be numeric.")

    converted = float(value)

    if not math.isfinite(converted):
        raise ValueError(f"{name} must be finite.")

    return converted


def _require_positive_int(name: str, value: int) -> None:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer.")

    if value <= 0:
        raise ValueError(f"{name} must be strictly positive.")


def _require_nonnegative_int(name: str, value: int) -> None:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer.")

    if value < 0:
        raise ValueError(f"{name} must be nonnegative.")


def _require_positive_float(name: str, value: float) -> None:
    converted = _require_finite_number(name, value)

    if converted <= 0.0:
        raise ValueError(f"{name} must be strictly positive.")


def _require_nonnegative_float(name: str, value: float) -> None:
    converted = _require_finite_number(name, value)

    if converted < 0.0:
        raise ValueError(f"{name} must be nonnegative.")


def _require_probability(
    name: str,
    value: float,
    *,
    include_one: bool = False,
) -> None:
    converted = _require_finite_number(name, value)
    upper_bound = 1.0 if include_one else 1.0

    upper_valid = (
        converted <= upper_bound
        if include_one
        else converted < upper_bound
    )

    if converted < 0.0 or not upper_valid:
        interval = "[0, 1]" if include_one else "[0, 1)"
        raise ValueError(f"{name} must lie in {interval}.")


def _require_open_probability(name: str, value: float) -> None:
    converted = _require_finite_number(name, value)

    if not 0.0 < converted < 1.0:
        raise ValueError(
            f"{name} must lie strictly between 0 and 1."
        )


def _require_unique_strings(
    name: str,
    values: Sequence[str],
) -> None:
    for index, value in enumerate(values):
        _require_nonempty_string(f"{name}[{index}]", value)

    duplicates = sorted(
        value
        for value, count in Counter(values).items()
        if count > 1
    )

    if duplicates:
        raise ValueError(
            f"{name} contains duplicate values: {duplicates}."
        )


def _require_subset(
    name: str,
    selected: Sequence[str],
    available: Sequence[str],
) -> None:
    missing = sorted(set(selected) - set(available))

    if missing:
        raise ValueError(
            f"{name} contains unavailable values: {missing}."
        )


def _validate_quantile_levels(
    quantile_levels: Sequence[float],
) -> None:
    if not quantile_levels:
        raise ValueError(
            "quantile_levels must not be empty."
        )

    converted = tuple(
        _require_finite_number(
            f"quantile_levels[{index}]",
            level,
        )
        for index, level in enumerate(quantile_levels)
    )

    if any(not 0.0 < level < 1.0 for level in converted):
        raise ValueError(
            "Every quantile level must lie strictly between 0 and 1."
        )

    if tuple(sorted(converted)) != converted:
        raise ValueError(
            "quantile_levels must be strictly increasing."
        )

    if len(set(converted)) != len(converted):
        raise ValueError(
            "quantile_levels must not contain duplicates."
        )


def _assert_implemented(
    capability_name: str,
    selected_value: str,
    implemented_values: Sequence[str],
) -> None:
    if selected_value not in implemented_values:
        raise NotImplementedError(
            f"{capability_name}={selected_value!r} is canonical but is "
            "not currently implemented. Implemented values: "
            f"{tuple(implemented_values)}."
        )


def _assert_all_implemented(
    capability_name: str,
    selected_values: Sequence[str],
    implemented_values: Sequence[str],
) -> None:
    missing = sorted(set(selected_values) - set(implemented_values))

    if missing:
        raise NotImplementedError(
            f"{capability_name} contains unimplemented values: {missing}. "
            f"Implemented values: {tuple(implemented_values)}."
        )


# =============================================================================
# Strict canonical serialization
# =============================================================================


def _canonicalize(value: Any) -> Any:
    if is_dataclass(value):
        return {
            item.name: _canonicalize(getattr(value, item.name))
            for item in fields(value)
        }

    if isinstance(value, Mapping):
        result: dict[str, Any] = {}

        for key, item in value.items():
            if not isinstance(key, str):
                raise TypeError(
                    "Configuration mappings must use string keys."
                )
            result[key] = _canonicalize(item)

        return result

    if isinstance(value, tuple):
        return [_canonicalize(item) for item in value]

    if isinstance(value, list):
        return [_canonicalize(item) for item in value]

    if value is None or isinstance(value, (str, bool, int)):
        return value

    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(
                "Configuration serialization does not permit NaN or "
                "infinite values."
            )
        return value

    raise TypeError(
        "Unsupported configuration value type: "
        f"{type(value).__name__}."
    )


def _canonical_json(payload: Mapping[str, Any]) -> str:
    return json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )


class ConfigMixin:
    """Shared immutable configuration helpers."""

    def replace(self, **changes: Any) -> Self:
        return dataclass_replace(self, **changes)

    def to_dict(self) -> dict[str, Any]:
        payload = _canonicalize(self)

        if not isinstance(payload, dict):
            raise TypeError(
                "Top-level configuration serialization must be a mapping."
            )

        return payload

    def to_json(self, *, indent: int | None = 2) -> str:
        return json.dumps(
            self.to_dict(),
            indent=indent,
            sort_keys=True,
            ensure_ascii=False,
            allow_nan=False,
        )


# =============================================================================
# Dataset and data-pipeline configuration
# =============================================================================


@dataclass(slots=True, frozen=True)
class DatasetConfig(ConfigMixin):
    """Scientifically identifies the dataset and graph artifacts."""

    adapter_name: str = "unconfigured"
    dataset_version: str = "unversioned"
    graph_version: str = "unversioned"
    panel_version: str = "unversioned"

    artifact_root: str = "."
    geography_level: str = "unspecified"
    feature_contract_name: str = "unversioned"
    split_definition: str = "unspecified"
    target_source: str = "unspecified"

    target_names: tuple[str, ...] = ("future_burden",)

    time_start: str | None = None
    time_end: str | None = None

    def __post_init__(self) -> None:
        self.validate()

    def validate(self) -> None:
        for name, value in (
            ("adapter_name", self.adapter_name),
            ("dataset_version", self.dataset_version),
            ("graph_version", self.graph_version),
            ("panel_version", self.panel_version),
            ("artifact_root", self.artifact_root),
            ("geography_level", self.geography_level),
            ("feature_contract_name", self.feature_contract_name),
            ("split_definition", self.split_definition),
            ("target_source", self.target_source),
        ):
            _require_nonempty_string(name, value)

        _require_unique_strings(
            "dataset target_names",
            self.target_names,
        )

        if (self.time_start is None) != (self.time_end is None):
            raise ValueError(
                "time_start and time_end must either both be provided or "
                "both be absent."
            )

        if self.time_start is not None:
            _require_nonempty_string("time_start", self.time_start)
            _require_nonempty_string("time_end", self.time_end)

            if self.time_start > self.time_end:
                raise ValueError(
                    "time_start must not be later than time_end."
                )


@dataclass(slots=True, frozen=True)
class DataConfig(ConfigMixin):
    """
    Describes what the data pipeline emits and how it is batched.

    Essential schema, graph-membership, and temporal-leakage checks are never
    disabled. Diagnostic flags only control expensive additional audits.
    """

    history_length: int | None = 12
    emits_history_sequences: bool = True
    emits_history_time_points: bool = True
    available_lag_feature_names: tuple[str, ...] = ()

    full_batch: bool = True
    batch_size: int = 1
    num_workers: int = 0
    pin_memory: bool = False

    hazard_scope: str = SCOPE_GRAPH
    scenario_scope: str = SCOPE_GRAPH

    allow_cross_graph_edges: bool = False
    use_graph_ptr: bool = True

    audit_all_finite_values: bool = True
    audit_duplicate_identifiers: bool = True
    audit_attention_normalization: bool = False
    audit_full_registry_semantics: bool = True

    def __post_init__(self) -> None:
        self.validate()

    def validate(self) -> None:
        if self.history_length is not None:
            _require_positive_int(
                "history_length",
                self.history_length,
            )

        _require_unique_strings(
            "available_lag_feature_names",
            self.available_lag_feature_names,
        )

        _require_positive_int("batch_size", self.batch_size)
        _require_nonnegative_int("num_workers", self.num_workers)

        _require_choice(
            "hazard_scope",
            self.hazard_scope,
            CANONICAL_SCOPES,
        )
        _require_choice(
            "scenario_scope",
            self.scenario_scope,
            CANONICAL_SCOPES,
        )

        if self.full_batch and self.batch_size != 1:
            raise ValueError(
                "full_batch=True requires batch_size=1. One loader item "
                "may still contain a packed multi-graph scenario batch."
            )

        if self.emits_history_time_points and not (
            self.emits_history_sequences
        ):
            raise ValueError(
                "History time points cannot be emitted without history "
                "sequences."
            )


# =============================================================================
# Memory, hazard, and fusion configuration
# =============================================================================


@dataclass(slots=True, frozen=True)
class MemoryConfig(ConfigMixin):
    """Controls urban-memory encoding and hazard-conditioned retrieval."""

    encoder_type: str = MEMORY_ENCODER_NONE
    query_type: str = MEMORY_QUERY_NONE

    input_dim: int | None = None
    hidden_dim: int = 64
    num_layers: int = 1
    dropout: float = 0.0

    bidirectional: bool = False

    transformer_heads: int = 4
    causal_attention: bool = True

    return_temporal_states: bool = False
    return_temporal_attention: bool = False

    lag_feature_names: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        self.validate()

    @property
    def enabled(self) -> bool:
        return self.encoder_type != MEMORY_ENCODER_NONE

    @property
    def recurrent(self) -> bool:
        return self.encoder_type in (
            MEMORY_ENCODER_GRU,
            MEMORY_ENCODER_LSTM,
        )

    @property
    def hazard_queried(self) -> bool:
        return self.query_type in (
            MEMORY_QUERY_HAZARD_ATTENTION,
            MEMORY_QUERY_SCENARIO_ATTENTION,
        )

    def validate(self) -> None:
        _require_choice(
            "memory encoder type",
            self.encoder_type,
            CANONICAL_MEMORY_ENCODER_TYPES,
        )
        _require_choice(
            "memory query type",
            self.query_type,
            CANONICAL_MEMORY_QUERY_TYPES,
        )

        if self.input_dim is not None:
            _require_positive_int(
                "memory input_dim",
                self.input_dim,
            )

        _require_positive_int(
            "memory hidden_dim",
            self.hidden_dim,
        )
        _require_positive_int(
            "memory num_layers",
            self.num_layers,
        )
        _require_probability(
            "memory dropout",
            self.dropout,
        )

        _require_unique_strings(
            "lag_feature_names",
            self.lag_feature_names,
        )

        if self.encoder_type == MEMORY_ENCODER_NONE:
            if self.query_type != MEMORY_QUERY_NONE:
                raise ValueError(
                    "Memory queries require an enabled memory encoder."
                )

            if (
                self.return_temporal_states
                or self.return_temporal_attention
            ):
                raise ValueError(
                    "Temporal memory outputs require an enabled encoder."
                )

            if self.bidirectional:
                raise ValueError(
                    "bidirectional is not meaningful when memory is "
                    "disabled."
                )

            if self.lag_feature_names:
                raise ValueError(
                    "lag_feature_names require the lag memory encoder."
                )

        if self.encoder_type == MEMORY_ENCODER_LAG:
            if not self.lag_feature_names:
                raise ValueError(
                    "The lag memory encoder requires lag_feature_names."
                )

            if self.bidirectional:
                raise ValueError(
                    "bidirectional only applies to GRU or LSTM encoders."
                )

        elif self.lag_feature_names:
            raise ValueError(
                "lag_feature_names may only be used with the lag encoder."
            )

        if self.recurrent:
            if self.transformer_heads != 4:
                raise ValueError(
                    "transformer_heads is inactive for recurrent encoders "
                    "and must remain at its default value."
                )

        elif self.bidirectional:
            raise ValueError(
                "bidirectional only applies to GRU or LSTM encoders."
            )

        if self.encoder_type == MEMORY_ENCODER_TRANSFORMER:
            _require_positive_int(
                "transformer_heads",
                self.transformer_heads,
            )

            if self.hidden_dim % self.transformer_heads != 0:
                raise ValueError(
                    "Transformer memory hidden_dim must be divisible by "
                    "transformer_heads."
                )

        if self.hazard_queried and not self.return_temporal_states:
            raise ValueError(
                "Hazard-queried memory requires "
                "return_temporal_states=True."
            )

        if (
            self.return_temporal_attention
            and not self.return_temporal_states
        ):
            raise ValueError(
                "Temporal-attention export requires temporal states."
            )

    def assert_implemented(self) -> None:
        _assert_implemented(
            "memory encoder type",
            self.encoder_type,
            V2_0_IMPLEMENTED_MEMORY_ENCODER_TYPES,
        )
        _assert_implemented(
            "memory query type",
            self.query_type,
            V2_0_IMPLEMENTED_MEMORY_QUERY_TYPES,
        )


@dataclass(slots=True, frozen=True)
class HazardConfig(ConfigMixin):
    """Controls hazard identity, scenario context, and conditioning mode."""

    conditioning_mode: str = HAZARD_CONDITIONING_NONE

    hazard_scope: str = SCOPE_GRAPH
    scenario_scope: str = SCOPE_GRAPH

    embedding_dim: int = 32
    output_dim: int = 64

    use_hazard_features: bool = False
    hazard_feature_dim: int | None = None

    use_scenario_features: bool = False
    scenario_feature_dim: int | None = None

    unknown_hazard_policy: str = UNKNOWN_ID_POLICY_ERROR

    def __post_init__(self) -> None:
        self.validate()

    @property
    def enabled(self) -> bool:
        return self.conditioning_mode != HAZARD_CONDITIONING_NONE

    def validate(self) -> None:
        _require_choice(
            "hazard conditioning mode",
            self.conditioning_mode,
            CANONICAL_HAZARD_CONDITIONING_MODES,
        )
        _require_choice(
            "hazard_scope",
            self.hazard_scope,
            CANONICAL_SCOPES,
        )
        _require_choice(
            "scenario_scope",
            self.scenario_scope,
            CANONICAL_SCOPES,
        )
        _require_choice(
            "unknown_hazard_policy",
            self.unknown_hazard_policy,
            CANONICAL_UNKNOWN_ID_POLICIES,
        )

        _require_positive_int(
            "hazard embedding_dim",
            self.embedding_dim,
        )
        _require_positive_int(
            "hazard output_dim",
            self.output_dim,
        )

        if self.use_hazard_features:
            if self.hazard_feature_dim is None:
                raise ValueError(
                    "hazard_feature_dim is required when hazard features "
                    "are enabled."
                )
            _require_positive_int(
                "hazard_feature_dim",
                self.hazard_feature_dim,
            )
        elif self.hazard_feature_dim is not None:
            raise ValueError(
                "hazard_feature_dim must be None when hazard features "
                "are disabled."
            )

        if self.use_scenario_features:
            if self.scenario_feature_dim is None:
                raise ValueError(
                    "scenario_feature_dim is required when scenario "
                    "features are enabled."
                )
            _require_positive_int(
                "scenario_feature_dim",
                self.scenario_feature_dim,
            )
        elif self.scenario_feature_dim is not None:
            raise ValueError(
                "scenario_feature_dim must be None when scenario "
                "features are disabled."
            )

        if (
            self.conditioning_mode
            == HAZARD_CONDITIONING_EMBEDDING_SCENARIO
            and not self.use_scenario_features
        ):
            raise ValueError(
                "hazard_and_scenario conditioning requires scenario "
                "features."
            )

        if (
            self.conditioning_mode == HAZARD_CONDITIONING_NONE
            and (
                self.use_hazard_features
                or self.use_scenario_features
            )
        ):
            raise ValueError(
                "Hazard/scenario features cannot be enabled while hazard "
                "conditioning is disabled."
            )

    def assert_implemented(self) -> None:
        _assert_implemented(
            "hazard conditioning mode",
            self.conditioning_mode,
            V2_0_IMPLEMENTED_HAZARD_CONDITIONING_MODES,
        )


@dataclass(slots=True, frozen=True)
class NodeStateFusionConfig(ConfigMixin):
    """Controls how model components form the initial node state."""

    mode: str = NODE_FUSION_CONCAT_PROJECTION
    output_dim: int = 64

    include_static_state: bool = True
    include_memory_state: bool = False
    include_hazard_memory_state: bool = False
    include_hazard_context: bool = False
    include_node_type_embedding: bool = False

    node_type_embedding_dim: int = 16

    dropout: float = 0.0
    layer_norm: bool = True

    def __post_init__(self) -> None:
        self.validate()

    def validate(self) -> None:
        _require_choice(
            "node-state fusion mode",
            self.mode,
            CANONICAL_NODE_FUSION_MODES,
        )
        _require_positive_int(
            "fusion output_dim",
            self.output_dim,
        )
        _require_positive_int(
            "node_type_embedding_dim",
            self.node_type_embedding_dim,
        )
        _require_probability(
            "fusion dropout",
            self.dropout,
        )

        if not any(
            (
                self.include_static_state,
                self.include_memory_state,
                self.include_hazard_memory_state,
                self.include_hazard_context,
                self.include_node_type_embedding,
            )
        ):
            raise ValueError(
                "Node-state fusion must include at least one component."
            )

    def assert_implemented(self) -> None:
        _assert_implemented(
            "node-state fusion mode",
            self.mode,
            V2_0_IMPLEMENTED_NODE_FUSION_MODES,
        )


# =============================================================================
# Relation and functional message-passing configuration
# =============================================================================


@dataclass(slots=True, frozen=True)
class RelationConfig(ConfigMixin):
    """Controls active relation families and relation-family gating."""

    active_relation_names: tuple[str, ...] = ()

    gate_enabled: bool = False
    gate_scope: str = RELATION_GATE_SCOPE_TARGET_NODE
    gate_activation: str = RELATION_GATE_ACTIVATION_SIGMOID
    gate_hidden_dim: int = 64

    use_relation_priors: bool = False
    relation_prior_strength: float = 0.0

    allow_control_relations: bool = True
    allow_control_relations_in_explanations: bool = False

    def __post_init__(self) -> None:
        self.validate()

    @property
    def active_control_relations(self) -> frozenset[str]:
        return frozenset(
            set(self.active_relation_names)
            & set(CONTROL_RELATION_NAMES)
        )

    def validate(self) -> None:
        _require_unique_strings(
            "active_relation_names",
            self.active_relation_names,
        )

        unknown = sorted(
            set(self.active_relation_names)
            - set(CANONICAL_RELATION_NAMES)
        )

        if unknown:
            raise ValueError(
                f"Unknown active relation names: {unknown}."
            )

        _require_choice(
            "relation gate scope",
            self.gate_scope,
            CANONICAL_RELATION_GATE_SCOPES,
        )
        _require_choice(
            "relation gate activation",
            self.gate_activation,
            CANONICAL_RELATION_GATE_ACTIVATIONS,
        )
        _require_positive_int(
            "gate_hidden_dim",
            self.gate_hidden_dim,
        )
        _require_nonnegative_float(
            "relation_prior_strength",
            self.relation_prior_strength,
        )

        if (
            self.active_control_relations
            and not self.allow_control_relations
        ):
            raise ValueError(
                "Control relations are active but "
                "allow_control_relations=False. Active controls: "
                f"{sorted(self.active_control_relations)}."
            )

        if (
            self.allow_control_relations_in_explanations
            and not self.allow_control_relations
        ):
            raise ValueError(
                "Control relations cannot be permitted in explanations "
                "when they are forbidden from the model."
            )

        if self.use_relation_priors and not self.gate_enabled:
            raise ValueError(
                "Relation priors require gate_enabled=True."
            )

        if (
            not self.use_relation_priors
            and self.relation_prior_strength != 0.0
        ):
            raise ValueError(
                "relation_prior_strength must be zero when priors are "
                "disabled."
            )

    def assert_implemented(self) -> None:
        _assert_all_implemented(
            "active_relation_names",
            self.active_relation_names,
            V2_0_IMPLEMENTED_RELATION_NAMES,
        )

        if self.gate_enabled:
            _assert_implemented(
                "relation gate scope",
                self.gate_scope,
                V2_0_IMPLEMENTED_RELATION_GATE_SCOPES,
            )
            _assert_implemented(
                "relation gate activation",
                self.gate_activation,
                V2_0_IMPLEMENTED_RELATION_GATE_ACTIVATIONS,
            )


@dataclass(slots=True, frozen=True)
class FunctionalMessagePassingConfig(ConfigMixin):
    """Controls the custom graph-layer stack."""

    enabled: bool = False
    num_layers: int = 1

    relation_transform_type: str = RELATION_TRANSFORM_SHARED
    aggregation_type: str = AGGREGATION_MEAN
    edge_normalization_type: str = EDGE_NORMALIZATION_NONE

    attention_enabled: bool = False
    attention_mode: str = ATTENTION_MODE_UNIFORM
    attention_normalization: str = (
        ATTENTION_NORMALIZATION_TARGET_NODE_RELATION
    )
    attention_heads: int = 1
    attention_head_reduction: str = ATTENTION_HEAD_REDUCTION_MEAN

    residual: bool = True
    layer_norm: bool = True
    dropout: float = 0.1

    capture_intermediate_messages: bool = False

    def __post_init__(self) -> None:
        self.validate()

    @property
    def hazard_conditioned_attention(self) -> bool:
        return self.attention_mode in (
            ATTENTION_MODE_HAZARD_CONDITIONED,
            ATTENTION_MODE_MULTIHEAD_HAZARD_CONDITIONED,
        )

    def validate(self) -> None:
        _require_positive_int(
            "message-passing num_layers",
            self.num_layers,
        )
        _require_choice(
            "relation transform type",
            self.relation_transform_type,
            CANONICAL_RELATION_TRANSFORM_TYPES,
        )
        _require_choice(
            "aggregation type",
            self.aggregation_type,
            CANONICAL_AGGREGATION_TYPES,
        )
        _require_choice(
            "edge normalization type",
            self.edge_normalization_type,
            CANONICAL_EDGE_NORMALIZATION_TYPES,
        )
        _require_choice(
            "attention mode",
            self.attention_mode,
            CANONICAL_ATTENTION_MODES,
        )
        _require_choice(
            "attention normalization",
            self.attention_normalization,
            CANONICAL_ATTENTION_NORMALIZATION_MODES,
        )
        _require_choice(
            "attention head reduction",
            self.attention_head_reduction,
            CANONICAL_ATTENTION_HEAD_REDUCTIONS,
        )

        _require_positive_int(
            "attention_heads",
            self.attention_heads,
        )
        _require_probability(
            "message-passing dropout",
            self.dropout,
        )

        if not self.enabled:
            if self.attention_enabled:
                raise ValueError(
                    "attention_enabled requires message passing."
                )

            if self.capture_intermediate_messages:
                raise ValueError(
                    "Intermediate edge messages require message passing."
                )

        if not self.attention_enabled:
            if self.attention_mode != ATTENTION_MODE_UNIFORM:
                raise ValueError(
                    "Non-uniform attention modes require "
                    "attention_enabled=True."
                )

        if (
            self.attention_mode
            == ATTENTION_MODE_MULTIHEAD_HAZARD_CONDITIONED
        ):
            if self.attention_heads < 2:
                raise ValueError(
                    "Multi-head attention requires at least two heads."
                )
        elif self.attention_heads != 1:
            raise ValueError(
                "attention_heads must equal 1 unless multi-head "
                "attention is selected."
            )

    def assert_implemented(self) -> None:
        if not self.enabled:
            return

        _assert_implemented(
            "relation transform type",
            self.relation_transform_type,
            V2_0_IMPLEMENTED_RELATION_TRANSFORM_TYPES,
        )
        _assert_implemented(
            "aggregation type",
            self.aggregation_type,
            V2_0_IMPLEMENTED_AGGREGATION_TYPES,
        )
        _assert_implemented(
            "edge normalization type",
            self.edge_normalization_type,
            V2_0_IMPLEMENTED_EDGE_NORMALIZATION_TYPES,
        )
        _assert_implemented(
            "attention mode",
            self.attention_mode,
            V2_0_IMPLEMENTED_ATTENTION_MODES,
        )
        _assert_implemented(
            "attention normalization",
            self.attention_normalization,
            V2_0_IMPLEMENTED_ATTENTION_NORMALIZATION_MODES,
        )
        _assert_implemented(
            "attention head reduction",
            self.attention_head_reduction,
            V2_0_IMPLEMENTED_ATTENTION_HEAD_REDUCTIONS,
        )


# =============================================================================
# Output-head configuration
# =============================================================================


@dataclass(slots=True, frozen=True)
class PredictionHeadConfig(ConfigMixin):
    """Controls node-level prediction outputs."""

    head_type: str = PREDICTION_HEAD_REGRESSION

    num_targets: int = 1
    target_names: tuple[str, ...] = ("future_burden",)
    target_horizons: tuple[str, ...] = ("next_1_month",)

    hidden_dims: tuple[int, ...] = (64,)
    dropout: float = 0.0

    output_bias: bool = True
    higher_is_riskier: bool = True

    def __post_init__(self) -> None:
        self.validate()

    def validate(self) -> None:
        _require_choice(
            "prediction head type",
            self.head_type,
            CANONICAL_PREDICTION_HEAD_TYPES,
        )
        _require_positive_int(
            "num_targets",
            self.num_targets,
        )
        _require_probability(
            "prediction dropout",
            self.dropout,
        )

        _require_unique_strings(
            "target_names",
            self.target_names,
        )

        for index, horizon in enumerate(self.target_horizons):
            _require_nonempty_string(
                f"target_horizons[{index}]",
                horizon,
            )

        if len(self.target_names) != self.num_targets:
            raise ValueError(
                "target_names length must equal num_targets."
            )

        if len(self.target_horizons) != self.num_targets:
            raise ValueError(
                "target_horizons length must equal num_targets."
            )

        for index, hidden_dim in enumerate(self.hidden_dims):
            _require_positive_int(
                f"hidden_dims[{index}]",
                hidden_dim,
            )

        if (
            self.head_type == PREDICTION_HEAD_MULTI_HORIZON
            and self.num_targets < 2
        ):
            raise ValueError(
                "The multi-horizon head requires at least two outputs."
            )

    def assert_implemented(self) -> None:
        _assert_implemented(
            "prediction head type",
            self.head_type,
            V2_0_IMPLEMENTED_PREDICTION_HEAD_TYPES,
        )


@dataclass(slots=True, frozen=True)
class UncertaintyConfig(ConfigMixin):
    """Controls uncertainty heads and inference/calibration procedures."""

    head_type: str = UNCERTAINTY_HEAD_NONE
    method_type: str = UNCERTAINTY_METHOD_NONE

    quantile_levels: tuple[float, ...] = ()

    mc_dropout_samples: int = 30
    ensemble_size: int = 5
    conformal_alpha: float = 0.1

    def __post_init__(self) -> None:
        self.validate()

    @property
    def enabled(self) -> bool:
        return (
            self.head_type != UNCERTAINTY_HEAD_NONE
            or self.method_type != UNCERTAINTY_METHOD_NONE
        )

    def validate(self) -> None:
        _require_choice(
            "uncertainty head type",
            self.head_type,
            CANONICAL_UNCERTAINTY_HEAD_TYPES,
        )
        _require_choice(
            "uncertainty method type",
            self.method_type,
            CANONICAL_UNCERTAINTY_METHOD_TYPES,
        )

        if self.head_type == UNCERTAINTY_HEAD_QUANTILE:
            _validate_quantile_levels(self.quantile_levels)
        elif self.quantile_levels:
            raise ValueError(
                "quantile_levels may only be provided for a quantile "
                "uncertainty head."
            )

        _require_positive_int(
            "mc_dropout_samples",
            self.mc_dropout_samples,
        )
        _require_positive_int(
            "ensemble_size",
            self.ensemble_size,
        )
        _require_open_probability(
            "conformal_alpha",
            self.conformal_alpha,
        )

        if (
            self.method_type == UNCERTAINTY_METHOD_MC_DROPOUT
            and self.mc_dropout_samples < 2
        ):
            raise ValueError(
                "Monte Carlo dropout requires at least two samples."
            )

        if (
            self.method_type == UNCERTAINTY_METHOD_ENSEMBLE
            and self.ensemble_size < 2
        ):
            raise ValueError(
                "Ensemble uncertainty requires at least two models."
            )

    def assert_implemented(self) -> None:
        _assert_implemented(
            "uncertainty head type",
            self.head_type,
            V2_0_IMPLEMENTED_UNCERTAINTY_HEAD_TYPES,
        )
        _assert_implemented(
            "uncertainty method type",
            self.method_type,
            V2_0_IMPLEMENTED_UNCERTAINTY_METHOD_TYPES,
        )


@dataclass(slots=True, frozen=True)
class ReportingBiasConfig(ConfigMixin):
    """Controls optional reporting-propensity modeling."""

    mode: str = REPORTING_BIAS_NONE

    hidden_dim: int = 64
    dropout: float = 0.0

    reporting_feature_names: tuple[str, ...] = ()
    detach_latent_branch: bool = False

    def __post_init__(self) -> None:
        self.validate()

    @property
    def enabled(self) -> bool:
        return self.mode != REPORTING_BIAS_NONE

    def validate(self) -> None:
        _require_choice(
            "reporting-bias mode",
            self.mode,
            CANONICAL_REPORTING_BIAS_TYPES,
        )
        _require_positive_int(
            "reporting hidden_dim",
            self.hidden_dim,
        )
        _require_probability(
            "reporting dropout",
            self.dropout,
        )
        _require_unique_strings(
            "reporting_feature_names",
            self.reporting_feature_names,
        )

        if self.enabled and not self.reporting_feature_names:
            raise ValueError(
                "An enabled reporting-bias module requires reporting "
                "features."
            )

    def assert_implemented(self) -> None:
        _assert_implemented(
            "reporting-bias mode",
            self.mode,
            V2_0_IMPLEMENTED_REPORTING_BIAS_TYPES,
        )


# =============================================================================
# Explanation configuration
# =============================================================================


@dataclass(slots=True, frozen=True)
class ExplanationConfig(ConfigMixin):
    """Controls retained explanation traces and exported summaries."""

    enabled: bool = False

    return_relation_gates: bool = False
    return_edge_attention: bool = False
    return_temporal_attention: bool = False
    return_pathway_scores: bool = False
    return_intermediate_states: bool = False

    top_k_relations: int = 5
    top_k_edges: int = 10
    top_k_history_periods: int = 5

    include_control_relations: bool = False
    export_per_attention_head: bool = False

    def __post_init__(self) -> None:
        self.validate()

    def validate(self) -> None:
        _require_nonnegative_int(
            "top_k_relations",
            self.top_k_relations,
        )
        _require_nonnegative_int(
            "top_k_edges",
            self.top_k_edges,
        )
        _require_nonnegative_int(
            "top_k_history_periods",
            self.top_k_history_periods,
        )

        requested = (
            self.return_relation_gates,
            self.return_edge_attention,
            self.return_temporal_attention,
            self.return_pathway_scores,
            self.return_intermediate_states,
        )

        if not self.enabled and any(requested):
            raise ValueError(
                "Explanation outputs cannot be requested while "
                "enabled=False."
            )


# =============================================================================
# Structured optimization, scheduling, and loss configuration
# =============================================================================


@dataclass(slots=True, frozen=True)
class OptimizerConfig(ConfigMixin):
    name: str = OPTIMIZER_ADAMW

    learning_rate: float = 1e-3
    weight_decay: float = 1e-4

    betas: tuple[float, float] = (0.9, 0.999)
    epsilon: float = 1e-8

    momentum: float = 0.0
    nesterov: bool = False
    amsgrad: bool = False

    def __post_init__(self) -> None:
        self.validate()

    def validate(self) -> None:
        _require_choice(
            "optimizer",
            self.name,
            CANONICAL_OPTIMIZERS,
        )
        _require_positive_float(
            "learning_rate",
            self.learning_rate,
        )
        _require_nonnegative_float(
            "weight_decay",
            self.weight_decay,
        )
        _require_positive_float(
            "optimizer epsilon",
            self.epsilon,
        )

        if len(self.betas) != 2:
            raise ValueError("betas must contain exactly two values.")

        for index, beta in enumerate(self.betas):
            _require_open_probability(
                f"betas[{index}]",
                beta,
            )

        _require_probability(
            "momentum",
            self.momentum,
            include_one=True,
        )

        if self.name == OPTIMIZER_SGD:
            if self.amsgrad:
                raise ValueError(
                    "AMSGrad is not available for SGD."
                )

            if self.nesterov and self.momentum <= 0.0:
                raise ValueError(
                    "Nesterov SGD requires positive momentum."
                )
        elif self.nesterov:
            raise ValueError(
                "nesterov only applies to SGD."
            )


@dataclass(slots=True, frozen=True)
class SchedulerConfig(ConfigMixin):
    name: str = SCHEDULER_NONE

    factor: float = 0.5
    patience: int = 10
    minimum_learning_rate: float = 1e-6
    threshold: float = 1e-4

    step_size: int = 25
    gamma: float = 0.5

    cosine_period: int = 100

    def __post_init__(self) -> None:
        self.validate()

    def validate(self) -> None:
        _require_choice(
            "scheduler",
            self.name,
            CANONICAL_SCHEDULERS,
        )
        _require_open_probability(
            "scheduler factor",
            self.factor,
        )
        _require_nonnegative_int(
            "scheduler patience",
            self.patience,
        )
        _require_nonnegative_float(
            "minimum_learning_rate",
            self.minimum_learning_rate,
        )
        _require_nonnegative_float(
            "scheduler threshold",
            self.threshold,
        )
        _require_positive_int(
            "step_size",
            self.step_size,
        )
        _require_open_probability(
            "gamma",
            self.gamma,
        )
        _require_positive_int(
            "cosine_period",
            self.cosine_period,
        )


@dataclass(slots=True, frozen=True)
class LossConfig(ConfigMixin):
    name: str = LOSS_MSE

    huber_delta: float = 1.0

    poisson_log_input: bool = False
    poisson_epsilon: float = 1e-8

    ranking_margin: float = 1.0
    ranking_cutoff: int | None = None

    quantile_levels: tuple[float, ...] = ()

    task_weights: tuple[float, ...] = ()

    def __post_init__(self) -> None:
        self.validate()

    def validate(self) -> None:
        _require_choice(
            "loss",
            self.name,
            CANONICAL_LOSSES,
        )
        _require_positive_float(
            "huber_delta",
            self.huber_delta,
        )
        _require_positive_float(
            "poisson_epsilon",
            self.poisson_epsilon,
        )
        _require_positive_float(
            "ranking_margin",
            self.ranking_margin,
        )

        if self.ranking_cutoff is not None:
            _require_positive_int(
                "ranking_cutoff",
                self.ranking_cutoff,
            )

        if self.name == LOSS_QUANTILE:
            _validate_quantile_levels(self.quantile_levels)
        elif self.quantile_levels:
            raise ValueError(
                "quantile_levels may only be set for quantile loss."
            )

        for index, weight in enumerate(self.task_weights):
            _require_nonnegative_float(
                f"task_weights[{index}]",
                weight,
            )

        if self.task_weights and not any(
            weight > 0.0
            for weight in self.task_weights
        ):
            raise ValueError(
                "At least one task weight must be positive."
            )


@dataclass(slots=True, frozen=True)
class TrainingConfig(ConfigMixin):
    """Controls optimization, validation, and checkpoint selection."""

    optimizer: OptimizerConfig = field(
        default_factory=OptimizerConfig
    )
    scheduler: SchedulerConfig = field(
        default_factory=SchedulerConfig
    )
    loss: LossConfig = field(default_factory=LossConfig)

    max_epochs: int = 200
    patience: int = 25
    validation_interval: int = 1

    gradient_clip_norm: float | None = 1.0

    monitor_metric: str = MONITOR_MAE
    monitor_mode: str = MONITOR_MODE_AUTO

    random_seed: int = 42
    deterministic: bool = True

    save_best_checkpoint: bool = True
    save_last_checkpoint: bool = True

    relation_gate_regularization_weight: float = 0.0
    attention_regularization_weight: float = 0.0
    uncertainty_loss_weight: float = 0.0
    reporting_loss_weight: float = 0.0

    def __post_init__(self) -> None:
        self.validate()

    def validate(self) -> None:
        self.optimizer.validate()
        self.scheduler.validate()
        self.loss.validate()

        _require_positive_int(
            "max_epochs",
            self.max_epochs,
        )
        _require_nonnegative_int(
            "patience",
            self.patience,
        )
        _require_positive_int(
            "validation_interval",
            self.validation_interval,
        )
        _require_nonnegative_int(
            "random_seed",
            self.random_seed,
        )

        if self.gradient_clip_norm is not None:
            _require_positive_float(
                "gradient_clip_norm",
                self.gradient_clip_norm,
            )

        _require_choice(
            "monitor_metric",
            self.monitor_metric,
            CANONICAL_MONITOR_METRICS,
        )
        _require_choice(
            "monitor_mode",
            self.monitor_mode,
            CANONICAL_MONITOR_MODES,
        )

        for name, value in (
            (
                "relation_gate_regularization_weight",
                self.relation_gate_regularization_weight,
            ),
            (
                "attention_regularization_weight",
                self.attention_regularization_weight,
            ),
            (
                "uncertainty_loss_weight",
                self.uncertainty_loss_weight,
            ),
            (
                "reporting_loss_weight",
                self.reporting_loss_weight,
            ),
        ):
            _require_nonnegative_float(name, value)

        if self.patience >= self.max_epochs:
            raise ValueError(
                "patience must be smaller than max_epochs."
            )

    def resolved_monitor_mode(self) -> str:
        if self.monitor_mode != MONITOR_MODE_AUTO:
            return self.monitor_mode

        maximizing = {
            MONITOR_SPEARMAN,
            MONITOR_NDCG,
            MONITOR_TOP_K_OVERLAP,
        }

        return (
            MONITOR_MODE_MAX
            if self.monitor_metric in maximizing
            else MONITOR_MODE_MIN
        )


# =============================================================================
# Runtime configuration
# =============================================================================


@dataclass(slots=True, frozen=True)
class RuntimeConfig(ConfigMixin):
    device: str = "auto"
    precision: str = PRECISION_FLOAT32

    use_amp: bool = False
    compile_model: bool = False
    gradient_accumulation_steps: int = 1

    def __post_init__(self) -> None:
        self.validate()

    def validate(self) -> None:
        _require_nonempty_string("device", self.device)
        _require_choice(
            "precision",
            self.precision,
            CANONICAL_PRECISIONS,
        )
        _require_positive_int(
            "gradient_accumulation_steps",
            self.gradient_accumulation_steps,
        )

        if (
            self.use_amp
            and self.precision not in (
                PRECISION_FLOAT16,
                PRECISION_BFLOAT16,
            )
        ):
            raise ValueError(
                "AMP requires float16 or bfloat16 precision."
            )


# =============================================================================
# Model configuration
# =============================================================================


@dataclass(slots=True, frozen=True)
class ModelConfig(ConfigMixin):
    """Complete architecture configuration for one V2 model instance."""

    model_family_id: str = MODEL_FAMILY_ID

    static_input_dim: int | None = None
    hidden_dim: int = 64
    node_type_count: int | None = None
    input_dropout: float = 0.0

    memory: MemoryConfig = field(default_factory=MemoryConfig)
    hazard: HazardConfig = field(default_factory=HazardConfig)
    fusion: NodeStateFusionConfig = field(
        default_factory=NodeStateFusionConfig
    )
    relations: RelationConfig = field(default_factory=RelationConfig)
    message_passing: FunctionalMessagePassingConfig = field(
        default_factory=FunctionalMessagePassingConfig
    )
    prediction: PredictionHeadConfig = field(
        default_factory=PredictionHeadConfig
    )
    uncertainty: UncertaintyConfig = field(
        default_factory=UncertaintyConfig
    )
    reporting_bias: ReportingBiasConfig = field(
        default_factory=ReportingBiasConfig
    )

    def __post_init__(self) -> None:
        self.validate()

    def validate(self) -> None:
        if self.model_family_id != MODEL_FAMILY_ID:
            raise ValueError(
                "ModelConfig belongs to an incompatible model family. "
                f"Observed {self.model_family_id!r}, "
                f"expected {MODEL_FAMILY_ID!r}."
            )

        if self.static_input_dim is not None:
            _require_positive_int(
                "static_input_dim",
                self.static_input_dim,
            )

        if self.node_type_count is not None:
            _require_positive_int(
                "node_type_count",
                self.node_type_count,
            )

        _require_positive_int(
            "hidden_dim",
            self.hidden_dim,
        )
        _require_probability(
            "input_dropout",
            self.input_dropout,
        )

        self.memory.validate()
        self.hazard.validate()
        self.fusion.validate()
        self.relations.validate()
        self.message_passing.validate()
        self.prediction.validate()
        self.uncertainty.validate()
        self.reporting_bias.validate()

        if self.fusion.output_dim != self.hidden_dim:
            raise ValueError(
                "fusion.output_dim must equal ModelConfig.hidden_dim."
            )

        self._validate_fusion_dependencies()
        self._validate_conditioning_identity()

    def _validate_fusion_dependencies(self) -> None:
        if (
            self.fusion.include_memory_state
            and not self.memory.enabled
        ):
            raise ValueError(
                "Fusion cannot include memory_state when memory is "
                "disabled."
            )

        if (
            self.fusion.include_hazard_memory_state
            and not self.memory.hazard_queried
        ):
            raise ValueError(
                "Hazard-memory fusion requires hazard-queried memory."
            )

        if (
            self.fusion.include_hazard_context
            and not self.hazard.enabled
        ):
            raise ValueError(
                "Hazard-context fusion requires hazard conditioning."
            )

        if (
            self.fusion.include_node_type_embedding
            and self.node_type_count is not None
            and self.node_type_count <= 0
        ):
            raise ValueError(
                "Node-type fusion requires a valid node_type_count."
            )

    def _validate_conditioning_identity(self) -> None:
        mode = self.hazard.conditioning_mode
        gate = self.relations.gate_enabled
        hazard_attention = (
            self.message_passing.hazard_conditioned_attention
        )
        queried_memory = self.memory.hazard_queried

        if (
            self.message_passing.enabled
            and not self.relations.active_relation_names
        ):
            raise ValueError(
                "Enabled message passing requires active relations."
            )

        if gate and not self.message_passing.enabled:
            raise ValueError(
                "Relation-family gating requires message passing."
            )

        if mode == HAZARD_CONDITIONING_NONE:
            if gate or hazard_attention or queried_memory:
                raise ValueError(
                    "Hazard-blind conditioning cannot enable gates, "
                    "hazard-conditioned attention, or queried memory."
                )

            if self.fusion.include_hazard_context:
                raise ValueError(
                    "Hazard-blind conditioning cannot fuse hazard context."
                )

        elif mode == HAZARD_CONDITIONING_EMBEDDING:
            if (
                gate
                or hazard_attention
                or queried_memory
                or self.hazard.use_scenario_features
            ):
                raise ValueError(
                    "hazard_embedding is an embedding-only ablation."
                )

            if not self.fusion.include_hazard_context:
                raise ValueError(
                    "hazard_embedding must expose hazard context through "
                    "node-state fusion."
                )

        elif mode == HAZARD_CONDITIONING_EMBEDDING_SCENARIO:
            if not self.hazard.use_scenario_features:
                raise ValueError(
                    "hazard_and_scenario requires scenario features."
                )

            if gate or hazard_attention or queried_memory:
                raise ValueError(
                    "hazard_and_scenario is an encoder-only ablation."
                )

            if not self.fusion.include_hazard_context:
                raise ValueError(
                    "hazard_and_scenario must fuse hazard context."
                )

        elif mode == HAZARD_CONDITIONING_RELATION_GATE:
            if not gate:
                raise ValueError(
                    "relation_gate conditioning requires an enabled gate."
                )

            if hazard_attention or queried_memory:
                raise ValueError(
                    "relation_gate conditioning excludes hazard edge "
                    "attention and queried memory."
                )

        elif mode == HAZARD_CONDITIONING_GATE_ATTENTION:
            if not gate:
                raise ValueError(
                    "relation_gate_and_attention requires a relation gate."
                )

            if not self.message_passing.attention_enabled:
                raise ValueError(
                    "relation_gate_and_attention requires edge attention."
                )

            if not hazard_attention:
                raise ValueError(
                    "relation_gate_and_attention requires a "
                    "hazard-conditioned attention mode."
                )

            if queried_memory:
                raise ValueError(
                    "relation_gate_and_attention excludes queried memory."
                )

        elif mode == HAZARD_CONDITIONING_QUERIED_MEMORY:
            if not queried_memory:
                raise ValueError(
                    "hazard_queried_memory requires a hazard/scenario "
                    "memory query."
                )

            if gate or hazard_attention:
                raise ValueError(
                    "hazard_queried_memory is an isolated memory ablation "
                    "and excludes gates and hazard edge attention."
                )

            if not self.fusion.include_hazard_memory_state:
                raise ValueError(
                    "hazard_queried_memory must fuse hazard memory."
                )

        elif mode == HAZARD_CONDITIONING_FULL:
            if not gate:
                raise ValueError(
                    "Full conditioning requires relation gates."
                )

            if not hazard_attention:
                raise ValueError(
                    "Full conditioning requires hazard-conditioned edge "
                    "attention."
                )

            if not queried_memory:
                raise ValueError(
                    "Full conditioning requires hazard-queried memory."
                )

            if not self.fusion.include_hazard_memory_state:
                raise ValueError(
                    "Full conditioning must fuse hazard memory."
                )

        if (
            self.memory.query_type == MEMORY_QUERY_SCENARIO_ATTENTION
            and not self.hazard.use_scenario_features
        ):
            raise ValueError(
                "Scenario-attention memory requires scenario features."
            )

    def assert_implemented(self) -> None:
        self.memory.assert_implemented()
        self.hazard.assert_implemented()
        self.fusion.assert_implemented()
        self.relations.assert_implemented()
        self.message_passing.assert_implemented()
        self.prediction.assert_implemented()
        self.uncertainty.assert_implemented()
        self.reporting_bias.assert_implemented()

    def assert_dimensions_resolved(self) -> None:
        if (
            self.fusion.include_static_state
            and self.static_input_dim is None
        ):
            raise ValueError(
                "static_input_dim must be resolved before model "
                "construction."
            )

        if (
            self.memory.enabled
            and self.memory.encoder_type
            in (
                MEMORY_ENCODER_GRU,
                MEMORY_ENCODER_LSTM,
                MEMORY_ENCODER_TRANSFORMER,
            )
            and self.memory.input_dim is None
        ):
            raise ValueError(
                "memory.input_dim must be resolved before recurrent or "
                "transformer construction."
            )

        if (
            self.fusion.include_node_type_embedding
            and self.node_type_count is None
        ):
            raise ValueError(
                "node_type_count must be resolved before node-type "
                "embedding construction."
            )


# =============================================================================
# Complete experiment configuration
# =============================================================================


@dataclass(slots=True, frozen=True)
class ExperimentConfig(ConfigMixin):
    """Reproducible configuration for one named experiment."""

    experiment_name: str = "unnamed_experiment"
    experiment_family: str = "v2_development"

    contract_versions: ContractVersions = field(
        default_factory=ContractVersions.current
    )

    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(
        default_factory=TrainingConfig
    )
    explanations: ExplanationConfig = field(
        default_factory=ExplanationConfig
    )
    runtime: RuntimeConfig = field(
        default_factory=RuntimeConfig
    )

    output_directory: str = "artifacts/v2_runs"
    tags: tuple[str, ...] = ()
    notes: str = ""

    def __post_init__(self) -> None:
        self.validate(require_current_versions=False)

    def validate(
        self,
        *,
        require_current_versions: bool = False,
    ) -> None:
        _require_nonempty_string(
            "experiment_name",
            self.experiment_name,
        )
        _require_nonempty_string(
            "experiment_family",
            self.experiment_family,
        )
        _require_nonempty_string(
            "output_directory",
            self.output_directory,
        )
        _require_unique_strings("tags", self.tags)

        self.contract_versions.validate(
            require_current=require_current_versions
        )

        self.dataset.validate()
        self.data.validate()
        self.model.validate()
        self.training.validate()
        self.explanations.validate()
        self.runtime.validate()

        self._validate_dataset_and_targets()
        self._validate_data_model_contract()
        self._validate_explanation_dependencies()
        self._validate_training_dependencies()
        self._validate_loss_compatibility()

    def _validate_dataset_and_targets(self) -> None:
        if (
            self.dataset.target_names
            != self.model.prediction.target_names
        ):
            raise ValueError(
                "Dataset target_names must match prediction target_names."
            )

    def _validate_data_model_contract(self) -> None:
        if (
            self.data.hazard_scope
            != self.model.hazard.hazard_scope
        ):
            raise ValueError(
                "Data and hazard encoder hazard scopes differ."
            )

        if (
            self.data.scenario_scope
            != self.model.hazard.scenario_scope
        ):
            raise ValueError(
                "Data and hazard encoder scenario scopes differ."
            )

        memory = self.model.memory

        if memory.enabled and self.data.history_length is None:
            raise ValueError(
                "Enabled memory requires a declared history_length."
            )

        if (
            memory.encoder_type
            in (
                MEMORY_ENCODER_GRU,
                MEMORY_ENCODER_LSTM,
                MEMORY_ENCODER_TRANSFORMER,
            )
            and not self.data.emits_history_sequences
        ):
            raise ValueError(
                "Recurrent and transformer memory require historical "
                "sequence tensors."
            )

        if (
            memory.return_temporal_attention
            and not self.data.emits_history_time_points
        ):
            raise ValueError(
                "Temporal-attention explanations require real history "
                "time points."
            )

        if memory.encoder_type == MEMORY_ENCODER_LAG:
            _require_subset(
                "memory lag_feature_names",
                memory.lag_feature_names,
                self.data.available_lag_feature_names,
            )

    def _validate_explanation_dependencies(self) -> None:
        explanation = self.explanations
        model = self.model

        if not explanation.enabled:
            return

        if (
            explanation.return_relation_gates
            and not model.relations.gate_enabled
        ):
            raise ValueError(
                "Relation-gate explanations require an enabled gate."
            )

        if (
            explanation.return_edge_attention
            and not model.message_passing.attention_enabled
        ):
            raise ValueError(
                "Edge-attention explanations require enabled attention."
            )

        if (
            explanation.return_temporal_attention
            and not model.memory.return_temporal_attention
        ):
            raise ValueError(
                "Temporal-attention explanations require temporal "
                "attention from the memory module."
            )

        if (
            explanation.return_pathway_scores
            and not (
                explanation.return_relation_gates
                or explanation.return_edge_attention
            )
        ):
            raise ValueError(
                "Pathway scores require relation-gate or edge-attention "
                "outputs."
            )

        if (
            explanation.export_per_attention_head
            and model.message_passing.attention_heads < 2
        ):
            raise ValueError(
                "Per-head export requires multi-head attention."
            )

        if (
            explanation.include_control_relations
            and not model.relations
            .allow_control_relations_in_explanations
        ):
            raise ValueError(
                "ExplanationConfig requests control relations, but "
                "RelationConfig forbids them in explanations."
            )

    def _validate_training_dependencies(self) -> None:
        training = self.training
        model = self.model

        if (
            training.relation_gate_regularization_weight > 0.0
            and not model.relations.gate_enabled
        ):
            raise ValueError(
                "Relation-gate regularization requires an enabled gate."
            )

        if (
            training.attention_regularization_weight > 0.0
            and not model.message_passing.attention_enabled
        ):
            raise ValueError(
                "Attention regularization requires enabled attention."
            )

        if (
            training.uncertainty_loss_weight > 0.0
            and not model.uncertainty.enabled
        ):
            raise ValueError(
                "A positive uncertainty loss weight requires an enabled "
                "uncertainty module."
            )

        if (
            training.reporting_loss_weight > 0.0
            and not model.reporting_bias.enabled
        ):
            raise ValueError(
                "A positive reporting loss weight requires an enabled "
                "reporting-bias module."
            )

    def _validate_loss_compatibility(self) -> None:
        loss = self.training.loss
        prediction = self.model.prediction
        uncertainty = self.model.uncertainty

        regression_heads = {
            PREDICTION_HEAD_REGRESSION,
            PREDICTION_HEAD_NONNEGATIVE_REGRESSION,
            PREDICTION_HEAD_MULTI_HORIZON,
        }

        if loss.name in (
            LOSS_MSE,
            LOSS_MAE,
            LOSS_HUBER,
        ):
            if prediction.head_type not in regression_heads:
                raise ValueError(
                    f"{loss.name} is incompatible with prediction head "
                    f"{prediction.head_type!r}."
                )

        elif loss.name == LOSS_POISSON_NLL:
            if prediction.head_type != PREDICTION_HEAD_POISSON_RATE:
                raise ValueError(
                    "Poisson NLL requires the poisson_rate prediction "
                    "head."
                )

        elif loss.name == LOSS_NEGATIVE_BINOMIAL_NLL:
            if (
                prediction.head_type
                != PREDICTION_HEAD_NEGATIVE_BINOMIAL
            ):
                raise ValueError(
                    "Negative-binomial NLL requires the "
                    "negative_binomial prediction head."
                )

        elif loss.name == LOSS_RANKING:
            if prediction.head_type != PREDICTION_HEAD_RANKING:
                raise ValueError(
                    "Ranking loss requires the ranking prediction head."
                )

        elif loss.name == LOSS_QUANTILE:
            if uncertainty.head_type != UNCERTAINTY_HEAD_QUANTILE:
                raise ValueError(
                    "Quantile loss requires a quantile uncertainty head."
                )

            if (
                tuple(loss.quantile_levels)
                != tuple(uncertainty.quantile_levels)
            ):
                raise ValueError(
                    "Loss and uncertainty quantile levels must match."
                )

        elif loss.name == LOSS_MULTITASK:
            task_count = (
                prediction.num_targets
                + int(uncertainty.enabled)
                + int(self.model.reporting_bias.enabled)
            )

            if task_count < 2:
                raise ValueError(
                    "Multitask loss requires at least two output tasks."
                )

            if (
                loss.task_weights
                and len(loss.task_weights) != task_count
            ):
                raise ValueError(
                    "task_weights must match the number of output tasks."
                )

    def assert_implemented(self) -> None:
        self.model.assert_implemented()

    def assert_construction_ready(self) -> None:
        self.validate(require_current_versions=True)
        self.assert_implemented()
        self.model.assert_dimensions_resolved()

        if self.dataset.adapter_name == "unconfigured":
            raise ValueError(
                "dataset.adapter_name must be configured before "
                "construction."
            )

    # -------------------------------------------------------------------------
    # Hashing
    # -------------------------------------------------------------------------

    def full_config_hash(self) -> str:
        return sha256(
            _canonical_json(self.to_dict()).encode("utf-8")
        ).hexdigest()

    def scientific_config_dict(self) -> dict[str, Any]:
        payload = self.to_dict()

        for key in (
            "experiment_name",
            "experiment_family",
            "output_directory",
            "tags",
            "notes",
        ):
            payload.pop(key, None)

        payload["dataset"].pop("artifact_root", None)

        payload["data"].pop("num_workers", None)
        payload["data"].pop("pin_memory", None)

        payload["runtime"].pop("device", None)
        payload["runtime"].pop("compile_model", None)

        payload["training"].pop(
            "save_best_checkpoint",
            None,
        )
        payload["training"].pop(
            "save_last_checkpoint",
            None,
        )

        return payload

    def scientific_config_hash(self) -> str:
        return sha256(
            _canonical_json(
                self.scientific_config_dict()
            ).encode("utf-8")
        ).hexdigest()

    def config_hash(self) -> str:
        """
        Backward-compatible alias for the scientific configuration hash.

        Run metadata should normally use this value.
        """

        return self.scientific_config_hash()

    # -------------------------------------------------------------------------
    # Strict persistence
    # -------------------------------------------------------------------------

    @classmethod
    def from_dict(
        cls,
        payload: Mapping[str, Any],
        *,
        require_current_versions: bool = False,
    ) -> ExperimentConfig:
        config = _build_experiment_config(payload)
        config.validate(
            require_current_versions=require_current_versions
        )
        return config

    @classmethod
    def from_json(
        cls,
        value: str,
        *,
        require_current_versions: bool = False,
    ) -> ExperimentConfig:
        parsed = json.loads(value)

        if not isinstance(parsed, Mapping):
            raise TypeError(
                "Serialized experiment configuration must be a JSON "
                "object."
            )

        return cls.from_dict(
            parsed,
            require_current_versions=require_current_versions,
        )

    @classmethod
    def load(
        cls,
        path: str | Path,
        *,
        require_current_versions: bool = False,
    ) -> ExperimentConfig:
        source = Path(path)

        try:
            text = source.read_text(encoding="utf-8")
        except OSError as exc:
            raise OSError(
                f"Could not read configuration file {source}."
            ) from exc

        return cls.from_json(
            text,
            require_current_versions=require_current_versions,
        )

    def save(
        self,
        path: str | Path,
        *,
        indent: int = 2,
    ) -> None:
        destination = Path(path)
        destination.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        try:
            destination.write_text(
                self.to_json(indent=indent) + "\n",
                encoding="utf-8",
            )
        except OSError as exc:
            raise OSError(
                f"Could not write configuration file {destination}."
            ) from exc

    # -------------------------------------------------------------------------
    # Presets
    # -------------------------------------------------------------------------

    @classmethod
    def minimal_skeleton(cls) -> ExperimentConfig:
        """
        Contract-only smoke configuration.

        This preset is conceptually valid but does not claim that a full model
        can currently be instantiated.
        """

        return cls(
            experiment_name="v2_minimal_skeleton",
            experiment_family="contract_smoke",
            dataset=DatasetConfig(
                adapter_name="schema_only",
                dataset_version="schema-only",
                graph_version="schema-only",
                panel_version="schema-only",
                artifact_root=".",
                geography_level="synthetic",
                feature_contract_name="schema-only",
                split_definition="none",
                target_source="synthetic",
                target_names=("future_burden",),
            ),
            data=DataConfig(
                history_length=None,
                emits_history_sequences=False,
                emits_history_time_points=False,
                full_batch=True,
                batch_size=1,
            ),
            model=ModelConfig(
                static_input_dim=None,
                hidden_dim=64,
                memory=MemoryConfig(
                    encoder_type=MEMORY_ENCODER_NONE,
                    query_type=MEMORY_QUERY_NONE,
                ),
                hazard=HazardConfig(
                    conditioning_mode=HAZARD_CONDITIONING_NONE,
                ),
                fusion=NodeStateFusionConfig(
                    mode=NODE_FUSION_CONCAT_PROJECTION,
                    output_dim=64,
                    include_static_state=True,
                ),
                relations=RelationConfig(
                    active_relation_names=(),
                    gate_enabled=False,
                ),
                message_passing=FunctionalMessagePassingConfig(
                    enabled=False,
                ),
                prediction=PredictionHeadConfig(
                    head_type=PREDICTION_HEAD_REGRESSION,
                    num_targets=1,
                    target_names=("future_burden",),
                    target_horizons=("next_1_month",),
                ),
                uncertainty=UncertaintyConfig(),
                reporting_bias=ReportingBiasConfig(),
            ),
            training=TrainingConfig(
                loss=LossConfig(name=LOSS_MSE),
            ),
            explanations=ExplanationConfig(enabled=False),
            tags=("skeleton", "contract_smoke"),
        )

    @classmethod
    def v2_0_runnable_baseline(cls) -> ExperimentConfig:
        """
        Return a currently executable baseline.

        Until the required prediction and fusion modules are marked as
        implemented, this method raises instead of returning a misleading
        configuration.
        """

        required_conditions = (
            PREDICTION_HEAD_REGRESSION
            in V2_0_IMPLEMENTED_PREDICTION_HEAD_TYPES
            and MEMORY_ENCODER_NONE
            in V2_0_IMPLEMENTED_MEMORY_ENCODER_TYPES
            and MEMORY_QUERY_NONE
            in V2_0_IMPLEMENTED_MEMORY_QUERY_TYPES
            and HAZARD_CONDITIONING_NONE
            in V2_0_IMPLEMENTED_HAZARD_CONDITIONING_MODES
            and NODE_FUSION_CONCAT_PROJECTION
            in V2_0_IMPLEMENTED_NODE_FUSION_MODES
        )

        if not required_conditions:
            raise NotImplementedError(
                "No honest runnable V2 baseline is available yet. "
                "Implement and register the static fusion and regression "
                "paths before enabling this preset."
            )

        config = cls.minimal_skeleton().replace(
            experiment_name="v2_0_runnable_baseline",
            experiment_family="runnable_baseline",
            model=cls.minimal_skeleton().model.replace(
                static_input_dim=1,
            ),
        )

        config.assert_construction_ready()
        return config

    @classmethod
    def v2_0_north_star_target(cls) -> ExperimentConfig:
        """
        Intended first research-grade V2 architecture.

        This preset may select canonical capabilities that are not yet
        implemented. ``assert_implemented()`` reports the remaining gaps.
        """

        lag_features = (
            "lag_1",
            "rolling_3",
            "rolling_6",
            "rolling_12",
            "seasonal_historical_mean",
        )

        return cls(
            experiment_name="v2_0_north_star_target",
            experiment_family=(
                "hazard_conditioned_functional_message_passing"
            ),
            dataset=DatasetConfig(
                adapter_name="benchmark_adapter",
                dataset_version="to_be_resolved",
                graph_version="to_be_resolved",
                panel_version="to_be_resolved",
                artifact_root="urban_graph_benchmark/artifacts",
                geography_level="tract_or_cd_month",
                feature_contract_name="v2_feature_parity",
                split_definition="chronological_train_val_test",
                target_source="future_urban_burden",
                target_names=("future_burden",),
            ),
            data=DataConfig(
                history_length=12,
                emits_history_sequences=True,
                emits_history_time_points=True,
                available_lag_feature_names=lag_features,
                full_batch=True,
                batch_size=1,
                hazard_scope=SCOPE_GRAPH,
                scenario_scope=SCOPE_GRAPH,
                audit_attention_normalization=True,
            ),
            model=ModelConfig(
                static_input_dim=None,
                hidden_dim=64,
                memory=MemoryConfig(
                    encoder_type=MEMORY_ENCODER_LAG,
                    query_type=MEMORY_QUERY_NONE,
                    hidden_dim=64,
                    lag_feature_names=lag_features,
                ),
                hazard=HazardConfig(
                    conditioning_mode=(
                        HAZARD_CONDITIONING_GATE_ATTENTION
                    ),
                    hazard_scope=SCOPE_GRAPH,
                    scenario_scope=SCOPE_GRAPH,
                    embedding_dim=32,
                    output_dim=64,
                ),
                fusion=NodeStateFusionConfig(
                    mode=NODE_FUSION_CONCAT_PROJECTION,
                    output_dim=64,
                    include_static_state=True,
                    include_memory_state=True,
                    include_hazard_context=False,
                    dropout=0.1,
                    layer_norm=True,
                ),
                relations=RelationConfig(
                    active_relation_names=(
                        V2_0_CANDIDATE_RELATION_NAMES
                    ),
                    gate_enabled=True,
                    gate_scope=RELATION_GATE_SCOPE_TARGET_NODE,
                    gate_activation=(
                        RELATION_GATE_ACTIVATION_SIGMOID
                    ),
                    gate_hidden_dim=64,
                    use_relation_priors=False,
                    allow_control_relations=True,
                    allow_control_relations_in_explanations=False,
                ),
                message_passing=FunctionalMessagePassingConfig(
                    enabled=True,
                    num_layers=1,
                    relation_transform_type=(
                        RELATION_TRANSFORM_PER_RELATION
                    ),
                    aggregation_type=AGGREGATION_MEAN,
                    edge_normalization_type=EDGE_NORMALIZATION_NONE,
                    attention_enabled=True,
                    attention_mode=(
                        ATTENTION_MODE_HAZARD_CONDITIONED
                    ),
                    attention_normalization=(
                        ATTENTION_NORMALIZATION_TARGET_NODE_RELATION
                    ),
                    attention_heads=1,
                    attention_head_reduction=(
                        ATTENTION_HEAD_REDUCTION_MEAN
                    ),
                    residual=True,
                    layer_norm=True,
                    dropout=0.1,
                ),
                prediction=PredictionHeadConfig(
                    head_type=(
                        PREDICTION_HEAD_NONNEGATIVE_REGRESSION
                    ),
                    num_targets=1,
                    target_names=("future_burden",),
                    target_horizons=("next_1_month",),
                    hidden_dims=(64,),
                    dropout=0.1,
                ),
                uncertainty=UncertaintyConfig(),
                reporting_bias=ReportingBiasConfig(),
            ),
            training=TrainingConfig(
                optimizer=OptimizerConfig(
                    name=OPTIMIZER_ADAMW,
                    learning_rate=1e-3,
                    weight_decay=1e-4,
                ),
                scheduler=SchedulerConfig(
                    name=SCHEDULER_REDUCE_ON_PLATEAU,
                    factor=0.5,
                    patience=10,
                    minimum_learning_rate=1e-6,
                ),
                loss=LossConfig(
                    name=LOSS_HUBER,
                    huber_delta=1.0,
                ),
                max_epochs=200,
                patience=25,
                monitor_metric=MONITOR_MAE,
                monitor_mode=MONITOR_MODE_MIN,
                random_seed=42,
            ),
            explanations=ExplanationConfig(
                enabled=True,
                return_relation_gates=True,
                return_edge_attention=True,
                return_pathway_scores=True,
                top_k_relations=5,
                top_k_edges=10,
                include_control_relations=False,
            ),
            runtime=RuntimeConfig(
                device="auto",
                precision=PRECISION_FLOAT32,
                use_amp=False,
            ),
            tags=(
                "v2",
                "north_star",
                "hazard_conditioned",
                "functional_message_passing",
            ),
        )


# =============================================================================
# Strict deserialization helpers
# =============================================================================


ConfigType = TypeVar("ConfigType")


def _require_mapping(
    name: str,
    payload: Any,
) -> Mapping[str, Any]:
    if not isinstance(payload, Mapping):
        raise TypeError(f"{name} must be a mapping.")

    return payload


def _reject_unknown_fields(
    config_type: type[Any],
    payload: Mapping[str, Any],
) -> None:
    allowed = {
        item.name
        for item in fields(config_type)
    }
    unknown = sorted(set(payload) - allowed)

    if unknown:
        raise ValueError(
            f"Unknown fields for {config_type.__name__}: {unknown}."
        )


def _as_tuple(
    name: str,
    value: Any,
) -> tuple[Any, ...]:
    if isinstance(value, tuple):
        return value

    if isinstance(value, list):
        return tuple(value)

    raise TypeError(f"{name} must be a list or tuple.")


def _build_simple_config(
    config_type: type[ConfigType],
    payload: Mapping[str, Any],
    *,
    tuple_fields: Sequence[str] = (),
) -> ConfigType:
    mapping = _require_mapping(
        config_type.__name__,
        payload,
    )
    _reject_unknown_fields(config_type, mapping)

    kwargs = dict(mapping)

    for field_name in tuple_fields:
        if field_name in kwargs:
            kwargs[field_name] = _as_tuple(
                field_name,
                kwargs[field_name],
            )

    return config_type(**kwargs)


def _build_contract_versions(
    payload: Mapping[str, Any],
) -> ContractVersions:
    return _build_simple_config(
        ContractVersions,
        payload,
    )


def _build_dataset_config(
    payload: Mapping[str, Any],
) -> DatasetConfig:
    return _build_simple_config(
        DatasetConfig,
        payload,
        tuple_fields=("target_names",),
    )


def _build_data_config(
    payload: Mapping[str, Any],
) -> DataConfig:
    return _build_simple_config(
        DataConfig,
        payload,
        tuple_fields=("available_lag_feature_names",),
    )


def _build_memory_config(
    payload: Mapping[str, Any],
) -> MemoryConfig:
    return _build_simple_config(
        MemoryConfig,
        payload,
        tuple_fields=("lag_feature_names",),
    )


def _build_hazard_config(
    payload: Mapping[str, Any],
) -> HazardConfig:
    return _build_simple_config(
        HazardConfig,
        payload,
    )


def _build_fusion_config(
    payload: Mapping[str, Any],
) -> NodeStateFusionConfig:
    return _build_simple_config(
        NodeStateFusionConfig,
        payload,
    )


def _build_relation_config(
    payload: Mapping[str, Any],
) -> RelationConfig:
    return _build_simple_config(
        RelationConfig,
        payload,
        tuple_fields=("active_relation_names",),
    )


def _build_message_passing_config(
    payload: Mapping[str, Any],
) -> FunctionalMessagePassingConfig:
    return _build_simple_config(
        FunctionalMessagePassingConfig,
        payload,
    )


def _build_prediction_config(
    payload: Mapping[str, Any],
) -> PredictionHeadConfig:
    return _build_simple_config(
        PredictionHeadConfig,
        payload,
        tuple_fields=(
            "target_names",
            "target_horizons",
            "hidden_dims",
        ),
    )


def _build_uncertainty_config(
    payload: Mapping[str, Any],
) -> UncertaintyConfig:
    return _build_simple_config(
        UncertaintyConfig,
        payload,
        tuple_fields=("quantile_levels",),
    )


def _build_reporting_config(
    payload: Mapping[str, Any],
) -> ReportingBiasConfig:
    return _build_simple_config(
        ReportingBiasConfig,
        payload,
        tuple_fields=("reporting_feature_names",),
    )


def _build_explanation_config(
    payload: Mapping[str, Any],
) -> ExplanationConfig:
    return _build_simple_config(
        ExplanationConfig,
        payload,
    )


def _build_optimizer_config(
    payload: Mapping[str, Any],
) -> OptimizerConfig:
    return _build_simple_config(
        OptimizerConfig,
        payload,
        tuple_fields=("betas",),
    )


def _build_scheduler_config(
    payload: Mapping[str, Any],
) -> SchedulerConfig:
    return _build_simple_config(
        SchedulerConfig,
        payload,
    )


def _build_loss_config(
    payload: Mapping[str, Any],
) -> LossConfig:
    return _build_simple_config(
        LossConfig,
        payload,
        tuple_fields=(
            "quantile_levels",
            "task_weights",
        ),
    )


def _build_training_config(
    payload: Mapping[str, Any],
) -> TrainingConfig:
    mapping = dict(
        _require_mapping("TrainingConfig", payload)
    )
    _reject_unknown_fields(TrainingConfig, mapping)

    if "optimizer" in mapping:
        mapping["optimizer"] = _build_optimizer_config(
            _require_mapping(
                "TrainingConfig.optimizer",
                mapping["optimizer"],
            )
        )

    if "scheduler" in mapping:
        mapping["scheduler"] = _build_scheduler_config(
            _require_mapping(
                "TrainingConfig.scheduler",
                mapping["scheduler"],
            )
        )

    if "loss" in mapping:
        mapping["loss"] = _build_loss_config(
            _require_mapping(
                "TrainingConfig.loss",
                mapping["loss"],
            )
        )

    return TrainingConfig(**mapping)


def _build_runtime_config(
    payload: Mapping[str, Any],
) -> RuntimeConfig:
    return _build_simple_config(
        RuntimeConfig,
        payload,
    )


def _build_model_config(
    payload: Mapping[str, Any],
) -> ModelConfig:
    mapping = dict(
        _require_mapping("ModelConfig", payload)
    )
    _reject_unknown_fields(ModelConfig, mapping)

    nested_builders = {
        "memory": _build_memory_config,
        "hazard": _build_hazard_config,
        "fusion": _build_fusion_config,
        "relations": _build_relation_config,
        "message_passing": _build_message_passing_config,
        "prediction": _build_prediction_config,
        "uncertainty": _build_uncertainty_config,
        "reporting_bias": _build_reporting_config,
    }

    for field_name, builder in nested_builders.items():
        if field_name in mapping:
            mapping[field_name] = builder(
                _require_mapping(
                    f"ModelConfig.{field_name}",
                    mapping[field_name],
                )
            )

    return ModelConfig(**mapping)


def _build_experiment_config(
    payload: Mapping[str, Any],
) -> ExperimentConfig:
    mapping = dict(
        _require_mapping("ExperimentConfig", payload)
    )
    _reject_unknown_fields(ExperimentConfig, mapping)

    nested_builders = {
        "contract_versions": _build_contract_versions,
        "dataset": _build_dataset_config,
        "data": _build_data_config,
        "model": _build_model_config,
        "training": _build_training_config,
        "explanations": _build_explanation_config,
        "runtime": _build_runtime_config,
    }

    for field_name, builder in nested_builders.items():
        if field_name in mapping:
            mapping[field_name] = builder(
                _require_mapping(
                    f"ExperimentConfig.{field_name}",
                    mapping[field_name],
                )
            )

    if "tags" in mapping:
        mapping["tags"] = _as_tuple(
            "tags",
            mapping["tags"],
        )

    return ExperimentConfig(**mapping)


__all__ = (
    "CANONICAL_LOSSES",
    "CANONICAL_MONITOR_METRICS",
    "CANONICAL_MONITOR_MODES",
    "CANONICAL_NODE_FUSION_MODES",
    "CANONICAL_OPTIMIZERS",
    "CANONICAL_PRECISIONS",
    "CANONICAL_SCHEDULERS",
    "DataConfig",
    "DatasetConfig",
    "ExperimentConfig",
    "ExplanationConfig",
    "FunctionalMessagePassingConfig",
    "HazardConfig",
    "LossConfig",
    "MemoryConfig",
    "ModelConfig",
    "NodeStateFusionConfig",
    "OptimizerConfig",
    "PredictionHeadConfig",
    "RelationConfig",
    "ReportingBiasConfig",
    "RuntimeConfig",
    "SchedulerConfig",
    "TrainingConfig",
    "UncertaintyConfig",
)