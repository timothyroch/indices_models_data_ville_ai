"""
Hazard-query construction for the V2 hazard-conditioned functional UGNN.

Target path:
    urban_resilience_models/
        v2_hazard_conditioned_functional_ugnn/
            hazard/
                hazard_query_encoder.py

This module owns:

- projection of metadata-preserving hazard embeddings;
- optional hazard-feature encoding;
- optional scenario-feature encoding;
- cyclical month/season context;
- continuous forecast-horizon encoding;
- weather-metadata encoding;
- event-metadata encoding;
- final multimodal hazard-query fusion;
- query-encoder architecture and parameter fingerprints.

It does not own:

- hazard identity or support semantics;
- hazard embedding vocabulary construction;
- hazard embedding lookup;
- graph relation gating;
- edge attention;
- temporal-memory retrieval;
- prediction heads.

Input boundary
--------------
The encoder consumes either:

- ``HazardEmbeddingLookup`` for item/graph-aligned hazard embeddings; or
- ``NodeAlignedHazardEmbeddingLookup`` for node-aligned embeddings that retain
  the originating graph-level hazard metadata.

It deliberately does not accept a bare embedding tensor. This prevents hazard
vectors from becoming detached from their canonical names, stable IDs,
unknown mask, vocabulary fingerprint, and embedding architecture lineage.

Context scopes
--------------
Optional contextual tensors use an explicit scope:

``item``
    One row per output query item.

``graph``
    One row per packed graph. For node-aligned embedding inputs, graph-scoped
    context is broadcast with the preserved ``node_batch_index``. For ordinary
    graph/item-level embedding inputs, graph and item scope are equivalent.

Month values are integers in ``[1, 12]``. Forecast horizons are positive,
finite numeric steps in the experiment's declared unit, such as months.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from hashlib import sha256
import json
import math
from types import MappingProxyType
from typing import Any, Final, Mapping

import torch
from torch import nn

from .hazard_embeddings import (
    HazardEmbeddingLookup,
    NodeAlignedHazardEmbeddingLookup,
)


# =============================================================================
# Schema identity
# =============================================================================


HAZARD_QUERY_ENCODER_SCHEMA_VERSION: Final[str] = "0.1"
HAZARD_QUERY_ENCODING_SCHEMA_VERSION: Final[str] = "0.1"


# =============================================================================
# Generic helpers
# =============================================================================


def _require_nonempty_string(name: str, value: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string.")


def _require_positive_int(name: str, value: int) -> None:
    if (
        isinstance(value, bool)
        or not isinstance(value, int)
        or value <= 0
    ):
        raise ValueError(f"{name} must be a positive integer.")


def _require_nonnegative_float(
    name: str,
    value: int | float,
) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
    ):
        raise TypeError(f"{name} must be numeric.")

    converted = float(value)

    if not math.isfinite(converted):
        raise ValueError(f"{name} must be finite.")

    if converted < 0.0:
        raise ValueError(f"{name} must be nonnegative.")

    return converted


def _canonical_json(payload: Mapping[str, Any]) -> str:
    return json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )


def _fingerprint(payload: Mapping[str, Any]) -> str:
    return sha256(
        _canonical_json(payload).encode("utf-8")
    ).hexdigest()


def _tensor_fingerprint(
    tensors: Mapping[str, torch.Tensor],
) -> str:
    digest = sha256()

    for name in sorted(tensors):
        tensor = (
            tensors[name]
            .detach()
            .cpu()
            .contiguous()
        )

        digest.update(name.encode("utf-8"))
        digest.update(str(tensor.dtype).encode("utf-8"))
        digest.update(
            json.dumps(
                list(tensor.shape),
                separators=(",", ":"),
            ).encode("utf-8")
        )
        digest.update(
            tensor.view(torch.uint8).numpy().tobytes()
        )

    return digest.hexdigest()


def _assert_finite_tensor(
    name: str,
    value: torch.Tensor,
) -> None:
    if value.dtype.is_floating_point and not bool(
        torch.isfinite(value).all().item()
    ):
        raise ValueError(
            f"{name} must contain only finite values."
        )


# =============================================================================
# Scoped contextual tensors
# =============================================================================


class HazardQueryTensorScope(StrEnum):
    """Alignment scope of one hazard-query context tensor."""

    ITEM = "item"
    GRAPH = "graph"


@dataclass(slots=True, frozen=True)
class ScopedHazardQueryTensor:
    """
    A context tensor with an explicit alignment scope.

    ``values`` must have at least one dimension. Its first dimension is the
    item or graph axis.
    """

    values: torch.Tensor
    scope: HazardQueryTensorScope | str
    name: str

    def __post_init__(self) -> None:
        if not isinstance(self.values, torch.Tensor):
            raise TypeError("values must be a tensor.")

        if self.values.ndim < 1:
            raise ValueError(
                "values must have at least one dimension."
            )

        scope = (
            self.scope
            if isinstance(self.scope, HazardQueryTensorScope)
            else HazardQueryTensorScope(self.scope)
        )
        object.__setattr__(self, "scope", scope)

        _require_nonempty_string("name", self.name)
        _assert_finite_tensor(self.name, self.values)

    @property
    def rows(self) -> int:
        return int(self.values.shape[0])


@dataclass(slots=True, frozen=True)
class HazardQueryContext:
    """Optional contextual inputs used to construct a hazard query."""

    hazard_features: ScopedHazardQueryTensor | None = None
    scenario_features: ScopedHazardQueryTensor | None = None
    month_indices: ScopedHazardQueryTensor | None = None
    forecast_horizon_steps: ScopedHazardQueryTensor | None = None
    weather_features: ScopedHazardQueryTensor | None = None
    event_features: ScopedHazardQueryTensor | None = None

    def __post_init__(self) -> None:
        for name, value in (
            ("hazard_features", self.hazard_features),
            ("scenario_features", self.scenario_features),
            ("month_indices", self.month_indices),
            (
                "forecast_horizon_steps",
                self.forecast_horizon_steps,
            ),
            ("weather_features", self.weather_features),
            ("event_features", self.event_features),
        ):
            if (
                value is not None
                and not isinstance(
                    value,
                    ScopedHazardQueryTensor,
                )
            ):
                raise TypeError(
                    f"{name} must be a ScopedHazardQueryTensor "
                    "or None."
                )


HazardEmbeddingResult = (
    HazardEmbeddingLookup
    | NodeAlignedHazardEmbeddingLookup
)


def _source_graph_lookup(
    source: HazardEmbeddingResult,
) -> HazardEmbeddingLookup:
    if isinstance(source, HazardEmbeddingLookup):
        return source

    if isinstance(
        source,
        NodeAlignedHazardEmbeddingLookup,
    ):
        return source.graph_lookup

    raise TypeError(
        "source must be a HazardEmbeddingLookup or "
        "NodeAlignedHazardEmbeddingLookup."
    )


def _source_item_embeddings(
    source: HazardEmbeddingResult,
) -> torch.Tensor:
    if isinstance(source, HazardEmbeddingLookup):
        return source.embeddings

    if isinstance(
        source,
        NodeAlignedHazardEmbeddingLookup,
    ):
        return source.node_embeddings

    raise TypeError(
        "source must be a HazardEmbeddingLookup or "
        "NodeAlignedHazardEmbeddingLookup."
    )


def _source_item_count(
    source: HazardEmbeddingResult,
) -> int:
    return int(
        _source_item_embeddings(source).shape[0]
    )


def _source_graph_count(
    source: HazardEmbeddingResult,
) -> int:
    return len(_source_graph_lookup(source).indices)


def _align_scoped_tensor(
    scoped: ScopedHazardQueryTensor,
    source: HazardEmbeddingResult,
) -> torch.Tensor:
    """
    Align a scoped tensor to the encoder output-item axis.

    No hidden device transfer is performed.
    """

    source_embeddings = _source_item_embeddings(source)

    if scoped.values.device != source_embeddings.device:
        raise ValueError(
            f"{scoped.name} and the hazard embeddings must share "
            "one device."
        )

    if scoped.scope == HazardQueryTensorScope.ITEM:
        expected_rows = _source_item_count(source)

        if scoped.rows != expected_rows:
            raise ValueError(
                f"{scoped.name} has {scoped.rows} item rows; "
                f"expected {expected_rows}."
            )

        return scoped.values

    expected_graph_rows = _source_graph_count(source)

    if scoped.rows != expected_graph_rows:
        raise ValueError(
            f"{scoped.name} has {scoped.rows} graph rows; "
            f"expected {expected_graph_rows}."
        )

    if isinstance(source, HazardEmbeddingLookup):
        # A graph-level lookup already has one item per graph.
        return scoped.values

    return scoped.values[source.node_batch_index]


# =============================================================================
# Query output
# =============================================================================


@dataclass(slots=True, frozen=True)
class HazardQueryEncoding:
    """
    Final query vectors plus every metadata-bearing intermediate component.

    ``source_embedding`` preserves canonical names, stable IDs, unknown masks,
    graph membership, vocabulary identity, and embedding architecture lineage.
    """

    query: torch.Tensor
    source_embedding: HazardEmbeddingResult

    projected_hazard_embedding: torch.Tensor

    hazard_feature_state: torch.Tensor | None
    scenario_state: torch.Tensor | None
    month_state: torch.Tensor | None
    forecast_horizon_state: torch.Tensor | None
    weather_state: torch.Tensor | None
    event_state: torch.Tensor | None

    query_encoder_architecture_fingerprint: str
    lineage_fingerprint: str
    schema_version: str = HAZARD_QUERY_ENCODING_SCHEMA_VERSION

    def __post_init__(self) -> None:
        source_values = _source_item_embeddings(
            self.source_embedding
        )

        if not isinstance(self.query, torch.Tensor):
            raise TypeError("query must be a tensor.")

        if self.query.ndim != 2:
            raise ValueError(
                "query must have shape [items, query_dim]."
            )

        if not self.query.dtype.is_floating_point:
            raise ValueError(
                "query must use a floating-point dtype."
            )

        if int(self.query.shape[0]) != int(
            source_values.shape[0]
        ):
            raise ValueError(
                "Query rows must align with source hazard embeddings."
            )

        if self.query.device != source_values.device:
            raise ValueError(
                "Query values and source hazard embeddings must "
                "share one device."
            )

        _assert_finite_tensor("query", self.query)

        states = (
            (
                "projected_hazard_embedding",
                self.projected_hazard_embedding,
            ),
            ("hazard_feature_state", self.hazard_feature_state),
            ("scenario_state", self.scenario_state),
            ("month_state", self.month_state),
            (
                "forecast_horizon_state",
                self.forecast_horizon_state,
            ),
            ("weather_state", self.weather_state),
            ("event_state", self.event_state),
        )

        for name, state in states:
            if state is None:
                continue

            if not isinstance(state, torch.Tensor):
                raise TypeError(f"{name} must be a tensor or None.")

            if state.ndim != 2:
                raise ValueError(
                    f"{name} must have shape [items, component_dim]."
                )

            if int(state.shape[0]) != int(self.query.shape[0]):
                raise ValueError(
                    f"{name} rows must align with query rows."
                )

            if state.device != self.query.device:
                raise ValueError(
                    f"{name} and query must share one device."
                )

            if not state.dtype.is_floating_point:
                raise ValueError(
                    f"{name} must use a floating-point dtype."
                )

            _assert_finite_tensor(name, state)

        for name, value in (
            (
                "query_encoder_architecture_fingerprint",
                self.query_encoder_architecture_fingerprint,
            ),
            ("lineage_fingerprint", self.lineage_fingerprint),
            ("schema_version", self.schema_version),
        ):
            _require_nonempty_string(name, value)

    @property
    def query_dim(self) -> int:
        return int(self.query.shape[1])

    @property
    def item_count(self) -> int:
        return int(self.query.shape[0])

    @property
    def graph_lookup(self) -> HazardEmbeddingLookup:
        return _source_graph_lookup(
            self.source_embedding
        )

    @property
    def component_states(
        self,
    ) -> Mapping[str, torch.Tensor]:
        values: dict[str, torch.Tensor] = {
            "hazard_embedding": (
                self.projected_hazard_embedding
            ),
        }

        optional = (
            ("hazard_features", self.hazard_feature_state),
            ("scenario", self.scenario_state),
            ("month", self.month_state),
            (
                "forecast_horizon",
                self.forecast_horizon_state,
            ),
            ("weather", self.weather_state),
            ("event", self.event_state),
        )

        for name, value in optional:
            if value is not None:
                values[name] = value

        return MappingProxyType(values)


# =============================================================================
# Component encoders
# =============================================================================


class _ContinuousFeatureEncoder(nn.Module):
    """Small projection block for dense continuous metadata."""

    def __init__(
        self,
        *,
        input_dim: int,
        output_dim: int,
        dropout: float,
    ) -> None:
        super().__init__()

        _require_positive_int("input_dim", input_dim)
        _require_positive_int("output_dim", output_dim)
        dropout_value = _require_nonnegative_float(
            "dropout",
            dropout,
        )

        if dropout_value >= 1.0:
            raise ValueError(
                "dropout must be strictly smaller than 1."
            )

        self.input_dim = input_dim
        self.output_dim = output_dim

        self.network = nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.GELU(),
            nn.LayerNorm(output_dim),
            nn.Dropout(dropout_value),
        )

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        if values.ndim != 2:
            raise ValueError(
                "Continuous feature inputs must have shape "
                "[items, feature_dim]."
            )

        if int(values.shape[1]) != self.input_dim:
            raise ValueError(
                "Continuous feature width does not match the "
                f"configured input_dim={self.input_dim}."
            )

        if not values.dtype.is_floating_point:
            raise ValueError(
                "Continuous feature inputs must use a floating-point "
                "dtype."
            )

        _assert_finite_tensor(
            "continuous feature inputs",
            values,
        )

        output = self.network(values)
        _assert_finite_tensor(
            "continuous feature state",
            output,
        )
        return output


class CyclicalMonthEncoder(nn.Module):
    """
    Encode calendar month using a cyclical sine/cosine representation.

    Months are one-indexed integers: January=1, ..., December=12.
    """

    def __init__(
        self,
        *,
        output_dim: int,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        _require_positive_int("output_dim", output_dim)

        self.output_dim = output_dim
        self.encoder = _ContinuousFeatureEncoder(
            input_dim=2,
            output_dim=output_dim,
            dropout=dropout,
        )

    def forward(
        self,
        month_indices: torch.Tensor,
    ) -> torch.Tensor:
        if month_indices.ndim == 2:
            if int(month_indices.shape[1]) != 1:
                raise ValueError(
                    "month_indices must have shape [items] or "
                    "[items, 1]."
                )
            month_indices = month_indices[:, 0]

        if month_indices.ndim != 1:
            raise ValueError(
                "month_indices must have shape [items] or [items, 1]."
            )

        if month_indices.dtype.is_floating_point:
            _assert_finite_tensor(
                "month_indices",
                month_indices,
            )
            if not torch.equal(
                month_indices,
                month_indices.round(),
            ):
                raise ValueError(
                    "month_indices must contain integer-valued months."
                )
        elif month_indices.dtype not in (
            torch.int8,
            torch.int16,
            torch.int32,
            torch.int64,
            torch.uint8,
        ):
            raise ValueError(
                "month_indices must use an integer or floating-point "
                "numeric dtype."
            )

        months = month_indices.to(
            dtype=self.encoder.network[0].weight.dtype
        )

        if months.numel() > 0:
            minimum = float(months.min().item())
            maximum = float(months.max().item())

            if minimum < 1.0 or maximum > 12.0:
                raise ValueError(
                    "month_indices must lie in [1, 12]."
                )

        angle = (
            (months - 1.0)
            * (2.0 * math.pi / 12.0)
        )
        features = torch.stack(
            (
                torch.sin(angle),
                torch.cos(angle),
            ),
            dim=-1,
        )
        return self.encoder(features)


class ForecastHorizonEncoder(nn.Module):
    """
    Encode a positive continuous forecast horizon with log-sinusoidal features.

    This accepts arbitrary positive finite horizon values and therefore avoids
    coupling the module to one hard-coded maximum horizon vocabulary.
    """

    def __init__(
        self,
        *,
        output_dim: int,
        num_frequencies: int = 4,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()

        _require_positive_int("output_dim", output_dim)
        _require_positive_int(
            "num_frequencies",
            num_frequencies,
        )

        self.output_dim = output_dim
        self.num_frequencies = num_frequencies

        frequencies = torch.pow(
            torch.tensor(2.0),
            torch.arange(
                num_frequencies,
                dtype=torch.float32,
            ),
        )
        self.register_buffer(
            "frequencies",
            frequencies,
            persistent=True,
        )

        self.encoder = _ContinuousFeatureEncoder(
            input_dim=1 + 2 * num_frequencies,
            output_dim=output_dim,
            dropout=dropout,
        )

    def forward(
        self,
        horizon_steps: torch.Tensor,
    ) -> torch.Tensor:
        if horizon_steps.ndim == 2:
            if int(horizon_steps.shape[1]) != 1:
                raise ValueError(
                    "forecast_horizon_steps must have shape [items] "
                    "or [items, 1]."
                )
            horizon_steps = horizon_steps[:, 0]

        if horizon_steps.ndim != 1:
            raise ValueError(
                "forecast_horizon_steps must have shape [items] "
                "or [items, 1]."
            )

        if not (
            horizon_steps.dtype.is_floating_point
            or horizon_steps.dtype
            in (
                torch.int8,
                torch.int16,
                torch.int32,
                torch.int64,
                torch.uint8,
            )
        ):
            raise ValueError(
                "forecast_horizon_steps must use a numeric dtype."
            )

        values = horizon_steps.to(
            dtype=self.encoder.network[0].weight.dtype
        )
        _assert_finite_tensor(
            "forecast_horizon_steps",
            values,
        )

        if values.numel() > 0 and float(
            values.min().item()
        ) <= 0.0:
            raise ValueError(
                "forecast_horizon_steps must be strictly positive."
            )

        logged = torch.log1p(values)
        frequencies = self.frequencies.to(
            device=logged.device,
            dtype=logged.dtype,
        )
        angles = logged.unsqueeze(-1) / frequencies

        features = torch.cat(
            (
                logged.unsqueeze(-1),
                torch.sin(angles),
                torch.cos(angles),
            ),
            dim=-1,
        )
        return self.encoder(features)


# =============================================================================
# Hazard query encoder
# =============================================================================


class HazardQueryEncoder(nn.Module):
    """
    Fuse a metadata-preserving hazard embedding with optional context.

    Every enabled component is projected to ``component_dim``. The component
    states are concatenated and mapped to one final ``output_dim`` query.

    The module never accepts a bare hazard embedding tensor. Use
    ``HazardEmbeddingLayer.lookup_*`` or
    ``HazardEmbeddingLayer.lookup_graph_hazards_for_nodes`` first.
    """

    def __init__(
        self,
        *,
        hazard_embedding_dim: int,
        output_dim: int,
        component_dim: int | None = None,
        hazard_feature_dim: int | None = None,
        scenario_feature_dim: int | None = None,
        use_month_context: bool = False,
        use_forecast_horizon: bool = False,
        forecast_horizon_frequencies: int = 4,
        weather_feature_dim: int | None = None,
        event_feature_dim: int | None = None,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()

        _require_positive_int(
            "hazard_embedding_dim",
            hazard_embedding_dim,
        )
        _require_positive_int("output_dim", output_dim)

        resolved_component_dim = (
            output_dim
            if component_dim is None
            else component_dim
        )
        _require_positive_int(
            "component_dim",
            resolved_component_dim,
        )

        dropout_value = _require_nonnegative_float(
            "dropout",
            dropout,
        )
        if dropout_value >= 1.0:
            raise ValueError(
                "dropout must be strictly smaller than 1."
            )

        for name, value in (
            ("hazard_feature_dim", hazard_feature_dim),
            ("scenario_feature_dim", scenario_feature_dim),
            ("weather_feature_dim", weather_feature_dim),
            ("event_feature_dim", event_feature_dim),
        ):
            if value is not None:
                _require_positive_int(name, value)

        if not isinstance(use_month_context, bool):
            raise TypeError(
                "use_month_context must be a Boolean."
            )
        if not isinstance(use_forecast_horizon, bool):
            raise TypeError(
                "use_forecast_horizon must be a Boolean."
            )

        self.hazard_embedding_dim = hazard_embedding_dim
        self.output_dim = output_dim
        self.component_dim = resolved_component_dim

        self.hazard_feature_dim = hazard_feature_dim
        self.scenario_feature_dim = scenario_feature_dim
        self.weather_feature_dim = weather_feature_dim
        self.event_feature_dim = event_feature_dim

        self.use_month_context = use_month_context
        self.use_forecast_horizon = use_forecast_horizon
        self.forecast_horizon_frequencies = (
            forecast_horizon_frequencies
        )
        self.dropout = dropout_value

        self.hazard_projection = _ContinuousFeatureEncoder(
            input_dim=hazard_embedding_dim,
            output_dim=resolved_component_dim,
            dropout=dropout_value,
        )

        self.hazard_feature_encoder = (
            _ContinuousFeatureEncoder(
                input_dim=hazard_feature_dim,
                output_dim=resolved_component_dim,
                dropout=dropout_value,
            )
            if hazard_feature_dim is not None
            else None
        )

        self.scenario_encoder = (
            _ContinuousFeatureEncoder(
                input_dim=scenario_feature_dim,
                output_dim=resolved_component_dim,
                dropout=dropout_value,
            )
            if scenario_feature_dim is not None
            else None
        )

        self.month_encoder = (
            CyclicalMonthEncoder(
                output_dim=resolved_component_dim,
                dropout=dropout_value,
            )
            if use_month_context
            else None
        )

        self.forecast_horizon_encoder = (
            ForecastHorizonEncoder(
                output_dim=resolved_component_dim,
                num_frequencies=(
                    forecast_horizon_frequencies
                ),
                dropout=dropout_value,
            )
            if use_forecast_horizon
            else None
        )

        self.weather_encoder = (
            _ContinuousFeatureEncoder(
                input_dim=weather_feature_dim,
                output_dim=resolved_component_dim,
                dropout=dropout_value,
            )
            if weather_feature_dim is not None
            else None
        )

        self.event_encoder = (
            _ContinuousFeatureEncoder(
                input_dim=event_feature_dim,
                output_dim=resolved_component_dim,
                dropout=dropout_value,
            )
            if event_feature_dim is not None
            else None
        )

        component_count = 1 + sum(
            (
                hazard_feature_dim is not None,
                scenario_feature_dim is not None,
                use_month_context,
                use_forecast_horizon,
                weather_feature_dim is not None,
                event_feature_dim is not None,
            )
        )
        self.component_count = int(component_count)

        fusion_input_dim = (
            self.component_count
            * resolved_component_dim
        )

        self.fusion = nn.Sequential(
            nn.LayerNorm(fusion_input_dim),
            nn.Linear(fusion_input_dim, output_dim),
            nn.GELU(),
            nn.Dropout(dropout_value),
            nn.Linear(output_dim, output_dim),
            nn.LayerNorm(output_dim),
        )

    # ------------------------------------------------------------------
    # Configuration construction
    # ------------------------------------------------------------------

    @classmethod
    def from_config(
        cls,
        config: "HazardConfig",
        *,
        component_dim: int | None = None,
        use_month_context: bool = False,
        use_forecast_horizon: bool = False,
        forecast_horizon_frequencies: int = 4,
        weather_feature_dim: int | None = None,
        event_feature_dim: int | None = None,
        dropout: float = 0.0,
    ) -> "HazardQueryEncoder":
        """
        Construct from the frozen hazard configuration contract.

        Month, horizon, weather, and event dimensions are explicit constructor
        options because the current ``HazardConfig`` does not yet claim those
        data-pipeline fields. They remain part of this module's architecture
        fingerprint and therefore cannot be silently changed.
        """

        from ..config import HazardConfig

        if not isinstance(config, HazardConfig):
            raise TypeError(
                "config must be a HazardConfig."
            )

        config.validate()

        if not config.enabled:
            raise ValueError(
                "HazardQueryEncoder requires enabled hazard "
                "conditioning."
            )

        return cls(
            hazard_embedding_dim=(
                config.embedding.embedding_dim
            ),
            output_dim=config.output_dim,
            component_dim=component_dim,
            hazard_feature_dim=(
                config.hazard_feature_dim
                if config.use_hazard_features
                else None
            ),
            scenario_feature_dim=(
                config.scenario_feature_dim
                if config.use_scenario_features
                else None
            ),
            use_month_context=use_month_context,
            use_forecast_horizon=(
                use_forecast_horizon
            ),
            forecast_horizon_frequencies=(
                forecast_horizon_frequencies
            ),
            weather_feature_dim=weather_feature_dim,
            event_feature_dim=event_feature_dim,
            dropout=dropout,
        )

    # ------------------------------------------------------------------
    # Forward path
    # ------------------------------------------------------------------

    @property
    def device(self) -> torch.device:
        return self.hazard_projection.network[0].weight.device

    @property
    def dtype(self) -> torch.dtype:
        return self.hazard_projection.network[0].weight.dtype

    def _require_context_contract(
        self,
        context: HazardQueryContext,
    ) -> None:
        contracts = (
            (
                "hazard_features",
                self.hazard_feature_encoder,
                context.hazard_features,
            ),
            (
                "scenario_features",
                self.scenario_encoder,
                context.scenario_features,
            ),
            (
                "month_indices",
                self.month_encoder,
                context.month_indices,
            ),
            (
                "forecast_horizon_steps",
                self.forecast_horizon_encoder,
                context.forecast_horizon_steps,
            ),
            (
                "weather_features",
                self.weather_encoder,
                context.weather_features,
            ),
            (
                "event_features",
                self.event_encoder,
                context.event_features,
            ),
        )

        for name, encoder, value in contracts:
            if encoder is None and value is not None:
                raise ValueError(
                    f"{name} was supplied, but its encoder is "
                    "disabled."
                )

            if encoder is not None and value is None:
                raise ValueError(
                    f"{name} is required because its encoder is "
                    "enabled."
                )

    def _align_continuous_features(
        self,
        *,
        scoped: ScopedHazardQueryTensor,
        source: HazardEmbeddingResult,
        expected_dim: int,
    ) -> torch.Tensor:
        values = _align_scoped_tensor(
            scoped,
            source,
        )

        if values.ndim != 2:
            raise ValueError(
                f"{scoped.name} must have shape "
                "[rows, feature_dim]."
            )

        if int(values.shape[1]) != expected_dim:
            raise ValueError(
                f"{scoped.name} has feature width "
                f"{int(values.shape[1])}; expected {expected_dim}."
            )

        if not values.dtype.is_floating_point:
            raise ValueError(
                f"{scoped.name} must use a floating-point dtype."
            )

        return values.to(dtype=self.dtype)

    def forward(
        self,
        source_embedding: HazardEmbeddingResult,
        context: HazardQueryContext | None = None,
    ) -> HazardQueryEncoding:
        if not isinstance(
            source_embedding,
            (
                HazardEmbeddingLookup,
                NodeAlignedHazardEmbeddingLookup,
            ),
        ):
            raise TypeError(
                "source_embedding must preserve hazard metadata; "
                "bare tensors are not accepted."
            )

        if context is None:
            context = HazardQueryContext()

        if not isinstance(context, HazardQueryContext):
            raise TypeError(
                "context must be a HazardQueryContext."
            )

        source_values = _source_item_embeddings(
            source_embedding
        )

        if source_values.device != self.device:
            raise ValueError(
                "Hazard embeddings and HazardQueryEncoder must "
                "share one device."
            )

        if int(source_values.shape[1]) != (
            self.hazard_embedding_dim
        ):
            raise ValueError(
                "Hazard embedding width does not match "
                f"hazard_embedding_dim={self.hazard_embedding_dim}."
            )

        self._require_context_contract(context)

        hazard_state = self.hazard_projection(
            source_values.to(dtype=self.dtype)
        )

        hazard_feature_state: torch.Tensor | None = None
        scenario_state: torch.Tensor | None = None
        month_state: torch.Tensor | None = None
        horizon_state: torch.Tensor | None = None
        weather_state: torch.Tensor | None = None
        event_state: torch.Tensor | None = None

        components = [hazard_state]

        if self.hazard_feature_encoder is not None:
            assert context.hazard_features is not None
            values = self._align_continuous_features(
                scoped=context.hazard_features,
                source=source_embedding,
                expected_dim=self.hazard_feature_dim,
            )
            hazard_feature_state = (
                self.hazard_feature_encoder(values)
            )
            components.append(hazard_feature_state)

        if self.scenario_encoder is not None:
            assert context.scenario_features is not None
            values = self._align_continuous_features(
                scoped=context.scenario_features,
                source=source_embedding,
                expected_dim=self.scenario_feature_dim,
            )
            scenario_state = self.scenario_encoder(
                values
            )
            components.append(scenario_state)

        if self.month_encoder is not None:
            assert context.month_indices is not None
            values = _align_scoped_tensor(
                context.month_indices,
                source_embedding,
            )
            month_state = self.month_encoder(values)
            components.append(month_state)

        if self.forecast_horizon_encoder is not None:
            assert (
                context.forecast_horizon_steps is not None
            )
            values = _align_scoped_tensor(
                context.forecast_horizon_steps,
                source_embedding,
            )
            horizon_state = (
                self.forecast_horizon_encoder(values)
            )
            components.append(horizon_state)

        if self.weather_encoder is not None:
            assert context.weather_features is not None
            values = self._align_continuous_features(
                scoped=context.weather_features,
                source=source_embedding,
                expected_dim=self.weather_feature_dim,
            )
            weather_state = self.weather_encoder(
                values
            )
            components.append(weather_state)

        if self.event_encoder is not None:
            assert context.event_features is not None
            values = self._align_continuous_features(
                scoped=context.event_features,
                source=source_embedding,
                expected_dim=self.event_feature_dim,
            )
            event_state = self.event_encoder(
                values
            )
            components.append(event_state)

        concatenated = torch.cat(
            components,
            dim=-1,
        )

        expected_width = (
            self.component_count
            * self.component_dim
        )
        if int(concatenated.shape[1]) != expected_width:
            raise RuntimeError(
                "Hazard-query component width differs from the "
                "constructed fusion contract."
            )

        query = self.fusion(concatenated)
        _assert_finite_tensor(
            "hazard query",
            query,
        )

        architecture_fingerprint = (
            self.architecture_fingerprint()
        )
        lineage_fingerprint = (
            self.lineage_fingerprint(
                source_embedding
            )
        )

        return HazardQueryEncoding(
            query=query,
            source_embedding=source_embedding,
            projected_hazard_embedding=hazard_state,
            hazard_feature_state=hazard_feature_state,
            scenario_state=scenario_state,
            month_state=month_state,
            forecast_horizon_state=horizon_state,
            weather_state=weather_state,
            event_state=event_state,
            query_encoder_architecture_fingerprint=(
                architecture_fingerprint
            ),
            lineage_fingerprint=lineage_fingerprint,
        )

    # ------------------------------------------------------------------
    # Identity and diagnostics
    # ------------------------------------------------------------------

    def architecture_dict(self) -> dict[str, Any]:
        return {
            "schema_version": (
                HAZARD_QUERY_ENCODER_SCHEMA_VERSION
            ),
            "hazard_embedding_dim": (
                self.hazard_embedding_dim
            ),
            "output_dim": self.output_dim,
            "component_dim": self.component_dim,
            "component_count": self.component_count,
            "hazard_feature_dim": self.hazard_feature_dim,
            "scenario_feature_dim": (
                self.scenario_feature_dim
            ),
            "use_month_context": self.use_month_context,
            "use_forecast_horizon": (
                self.use_forecast_horizon
            ),
            "forecast_horizon_frequencies": (
                self.forecast_horizon_frequencies
            ),
            "weather_feature_dim": (
                self.weather_feature_dim
            ),
            "event_feature_dim": self.event_feature_dim,
            "dropout": self.dropout,
            "month_encoding": "cyclical_sine_cosine",
            "forecast_horizon_encoding": (
                "continuous_log_sinusoidal"
            ),
            "fusion": "concatenate_projected_components_mlp",
        }

    def architecture_fingerprint(self) -> str:
        return _fingerprint(
            self.architecture_dict()
        )

    def parameter_fingerprint(self) -> str:
        return _tensor_fingerprint(
            {
                name: tensor
                for name, tensor
                in self.state_dict().items()
            }
        )

    def lineage_fingerprint(
        self,
        source_embedding: HazardEmbeddingResult,
    ) -> str:
        graph_lookup = _source_graph_lookup(
            source_embedding
        )

        return _fingerprint(
            {
                "query_encoder_architecture_fingerprint": (
                    self.architecture_fingerprint()
                ),
                "embedding_architecture_fingerprint": (
                    graph_lookup.architecture_fingerprint
                ),
                "embedding_vocabulary_fingerprint": (
                    graph_lookup.indices.vocabulary_fingerprint
                ),
                "embedding_mode": (
                    graph_lookup.embedding_mode.value
                ),
                "node_aligned": isinstance(
                    source_embedding,
                    NodeAlignedHazardEmbeddingLookup,
                ),
            }
        )

    def assert_finite_parameters(self) -> None:
        for name, tensor in self.state_dict().items():
            if (
                tensor.dtype.is_floating_point
                and not bool(
                    torch.isfinite(tensor).all().item()
                )
            ):
                raise ValueError(
                    f"Hazard-query tensor {name!r} contains NaN "
                    "or infinity."
                )

    def extra_repr(self) -> str:
        return (
            f"hazard_embedding_dim={self.hazard_embedding_dim}, "
            f"output_dim={self.output_dim}, "
            f"component_dim={self.component_dim}, "
            f"components={self.component_count}"
        )


__all__ = (
    "CyclicalMonthEncoder",
    "ForecastHorizonEncoder",
    "HAZARD_QUERY_ENCODER_SCHEMA_VERSION",
    "HAZARD_QUERY_ENCODING_SCHEMA_VERSION",
    "HazardEmbeddingResult",
    "HazardQueryContext",
    "HazardQueryEncoder",
    "HazardQueryEncoding",
    "HazardQueryTensorScope",
    "ScopedHazardQueryTensor",
)
