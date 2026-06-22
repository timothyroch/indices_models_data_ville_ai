"""
Metadata-preserving lag-memory encoder for the V2 functional UGNN.

Target path:
    urban_resilience_models/
        v2_hazard_conditioned_functional_ugnn/
            memory/
                lag_memory_encoder.py

This module owns:

- validation of named lag-feature batches;
- explicit feature and history masks;
- preservation of source history/time metadata;
- learned encoding of scalar lag values and lag-feature identity;
- masked aggregation into one node/item memory state;
- optional per-lag states and deterministic lag weights;
- architecture, parameter, and lineage fingerprints.

It does not own:

- extraction of lag features from raw panels;
- recurrent or transformer history encoding;
- hazard-conditioned memory retrieval;
- node-state fusion;
- graph message passing;
- prediction heads.

Input boundary
--------------
The encoder consumes ``LagMemoryBatch`` rather than a bare tensor. This keeps
the lag-feature names, feature mask, source history length, history mask,
history time points, and derivation fingerprint attached to the values used to
construct memory.

The configured ``lag_feature_names`` are ordered semantic identities. The
incoming batch must use exactly the same names in exactly the same order.

Architecture
------------
Each scalar lag feature is encoded as:

    value_projection(lag_value) + learned_feature_identity

The resulting per-lag states are transformed, masked, and aggregated with a
transparent masked mean. This first implementation intentionally avoids a
learned temporal-attention mechanism; hazard-conditioned retrieval belongs in
a later memory-query module.

The normalized masked-mean weights may be returned as deterministic temporal
weights for auditing and explanations.
"""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
import math
from types import MappingProxyType
from typing import Any, Final, Mapping, Sequence

import torch
from torch import nn


# =============================================================================
# Schema identity
# =============================================================================


LAG_MEMORY_BATCH_SCHEMA_VERSION: Final[str] = "0.1"
LAG_MEMORY_ENCODER_SCHEMA_VERSION: Final[str] = "0.1"
LAG_MEMORY_ENCODING_SCHEMA_VERSION: Final[str] = "0.1"


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


def _require_unique_strings(
    name: str,
    values: Sequence[str],
) -> None:
    observed: set[str] = set()
    duplicates: set[str] = set()

    for index, value in enumerate(values):
        _require_nonempty_string(
            f"{name}[{index}]",
            value,
        )

        if value in observed:
            duplicates.add(value)

        observed.add(value)

    if duplicates:
        raise ValueError(
            f"{name} contains duplicates: {sorted(duplicates)}."
        )


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
    tensor: torch.Tensor,
) -> None:
    if tensor.dtype.is_floating_point and not bool(
        torch.isfinite(tensor).all().item()
    ):
        raise ValueError(
            f"{name} must contain only finite values."
        )


def _is_numeric_tensor(tensor: torch.Tensor) -> bool:
    return (
        tensor.dtype.is_floating_point
        or tensor.dtype
        in (
            torch.int8,
            torch.int16,
            torch.int32,
            torch.int64,
            torch.uint8,
        )
    )


# =============================================================================
# Metadata-preserving lag-memory input
# =============================================================================


@dataclass(slots=True, frozen=True)
class LagMemoryBatch:
    """
    Named lag features and the metadata from which they were derived.

    Parameters
    ----------
    lag_values:
        Floating tensor with shape ``[items, num_lag_features]``.

    lag_feature_names:
        Ordered names aligned with the second axis of ``lag_values``.

    feature_mask:
        Optional Boolean tensor with the same shape as ``lag_values``.
        ``True`` marks an available lag value. Missing values must still use a
        finite placeholder in ``lag_values``; the mask controls participation.

    source_history_length:
        Optional declared number of historical periods used to derive the lag
        features.

    history_mask:
        Optional Boolean tensor ``[items, history_length]`` describing which
        raw historical periods were available.

    history_time_points:
        Optional numeric tensor ``[history_length]`` shared by all items or
        ``[items, history_length]`` for item-specific histories. Values must be
        nondecreasing over valid periods.

    source_fingerprint:
        Optional upstream artifact or transformation fingerprint.
    """

    lag_values: torch.Tensor
    lag_feature_names: tuple[str, ...]

    feature_mask: torch.Tensor | None = None

    source_history_length: int | None = None
    history_mask: torch.Tensor | None = None
    history_time_points: torch.Tensor | None = None

    source_fingerprint: str | None = None
    schema_version: str = LAG_MEMORY_BATCH_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if not isinstance(self.lag_values, torch.Tensor):
            raise TypeError("lag_values must be a tensor.")

        if self.lag_values.ndim != 2:
            raise ValueError(
                "lag_values must have shape "
                "[items, num_lag_features]."
            )

        if not self.lag_values.dtype.is_floating_point:
            raise ValueError(
                "lag_values must use a floating-point dtype."
            )

        _assert_finite_tensor(
            "lag_values",
            self.lag_values,
        )

        _require_unique_strings(
            "lag_feature_names",
            self.lag_feature_names,
        )

        if int(self.lag_values.shape[1]) != len(
            self.lag_feature_names
        ):
            raise ValueError(
                "lag_feature_names must align with lag_values columns."
            )

        if not self.lag_feature_names:
            raise ValueError(
                "At least one lag feature is required."
            )

        item_count = int(self.lag_values.shape[0])
        lag_count = int(self.lag_values.shape[1])

        if self.feature_mask is not None:
            if not isinstance(
                self.feature_mask,
                torch.Tensor,
            ):
                raise TypeError(
                    "feature_mask must be a tensor or None."
                )

            if self.feature_mask.dtype != torch.bool:
                raise ValueError(
                    "feature_mask must use torch.bool."
                )

            if tuple(self.feature_mask.shape) != (
                item_count,
                lag_count,
            ):
                raise ValueError(
                    "feature_mask must have the same shape as "
                    "lag_values."
                )

            if self.feature_mask.device != (
                self.lag_values.device
            ):
                raise ValueError(
                    "feature_mask and lag_values must share one device."
                )

            if item_count > 0 and bool(
                (~self.feature_mask.any(dim=1)).any().item()
            ):
                raise ValueError(
                    "Every item must have at least one available lag "
                    "feature."
                )

        if self.source_history_length is not None:
            _require_positive_int(
                "source_history_length",
                self.source_history_length,
            )

        self._validate_history_metadata(
            item_count=item_count,
        )

        if self.source_fingerprint is not None:
            _require_nonempty_string(
                "source_fingerprint",
                self.source_fingerprint,
            )

        _require_nonempty_string(
            "schema_version",
            self.schema_version,
        )

    def _validate_history_metadata(
        self,
        *,
        item_count: int,
    ) -> None:
        observed_history_length: int | None = None

        if self.history_mask is not None:
            if not isinstance(
                self.history_mask,
                torch.Tensor,
            ):
                raise TypeError(
                    "history_mask must be a tensor or None."
                )

            if self.history_mask.ndim != 2:
                raise ValueError(
                    "history_mask must have shape "
                    "[items, history_length]."
                )

            if self.history_mask.dtype != torch.bool:
                raise ValueError(
                    "history_mask must use torch.bool."
                )

            if int(self.history_mask.shape[0]) != item_count:
                raise ValueError(
                    "history_mask rows must align with lag_values."
                )

            if self.history_mask.device != (
                self.lag_values.device
            ):
                raise ValueError(
                    "history_mask and lag_values must share one device."
                )

            observed_history_length = int(
                self.history_mask.shape[1]
            )

        if self.history_time_points is not None:
            if not isinstance(
                self.history_time_points,
                torch.Tensor,
            ):
                raise TypeError(
                    "history_time_points must be a tensor or None."
                )

            if self.history_time_points.ndim not in (1, 2):
                raise ValueError(
                    "history_time_points must have shape "
                    "[history_length] or [items, history_length]."
                )

            if not _is_numeric_tensor(
                self.history_time_points
            ):
                raise ValueError(
                    "history_time_points must use a numeric dtype."
                )

            _assert_finite_tensor(
                "history_time_points",
                self.history_time_points,
            )

            if self.history_time_points.device != (
                self.lag_values.device
            ):
                raise ValueError(
                    "history_time_points and lag_values must share "
                    "one device."
                )

            if self.history_time_points.ndim == 1:
                time_length = int(
                    self.history_time_points.shape[0]
                )
            else:
                if int(
                    self.history_time_points.shape[0]
                ) != item_count:
                    raise ValueError(
                        "history_time_points rows must align with "
                        "lag_values."
                    )

                time_length = int(
                    self.history_time_points.shape[1]
                )

            if (
                observed_history_length is not None
                and time_length != observed_history_length
            ):
                raise ValueError(
                    "history_time_points and history_mask must use "
                    "the same history length."
                )

            observed_history_length = time_length
            self._validate_time_order()

        if (
            self.source_history_length is not None
            and observed_history_length is not None
            and self.source_history_length
            != observed_history_length
        ):
            raise ValueError(
                "source_history_length does not match the supplied "
                "history metadata."
            )

    def _validate_time_order(self) -> None:
        assert self.history_time_points is not None

        times = self.history_time_points

        if times.ndim == 1:
            if times.numel() > 1 and bool(
                (times[1:] < times[:-1]).any().item()
            ):
                raise ValueError(
                    "history_time_points must be nondecreasing."
                )

            return

        if int(times.shape[1]) <= 1:
            return

        if self.history_mask is None:
            invalid = times[:, 1:] < times[:, :-1]
        else:
            adjacent_valid = (
                self.history_mask[:, 1:]
                & self.history_mask[:, :-1]
            )
            invalid = (
                (times[:, 1:] < times[:, :-1])
                & adjacent_valid
            )

        if bool(invalid.any().item()):
            raise ValueError(
                "history_time_points must be nondecreasing over "
                "valid periods."
            )

    def __len__(self) -> int:
        return int(self.lag_values.shape[0])

    @property
    def item_count(self) -> int:
        return len(self)

    @property
    def num_lag_features(self) -> int:
        return int(self.lag_values.shape[1])

    @property
    def device(self) -> torch.device:
        return self.lag_values.device

    @property
    def effective_feature_mask(self) -> torch.Tensor:
        if self.feature_mask is not None:
            return self.feature_mask

        return torch.ones(
            self.lag_values.shape,
            dtype=torch.bool,
            device=self.lag_values.device,
        )

    @property
    def resolved_history_length(self) -> int | None:
        if self.source_history_length is not None:
            return self.source_history_length

        if self.history_mask is not None:
            return int(self.history_mask.shape[1])

        if self.history_time_points is None:
            return None

        return int(self.history_time_points.shape[-1])

    def expanded_history_time_points(
        self,
    ) -> torch.Tensor | None:
        if self.history_time_points is None:
            return None

        if self.history_time_points.ndim == 2:
            return self.history_time_points

        return self.history_time_points.unsqueeze(0).expand(
            self.item_count,
            -1,
        )

    def metadata_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "lag_feature_names": list(
                self.lag_feature_names
            ),
            "item_count": self.item_count,
            "num_lag_features": self.num_lag_features,
            "source_history_length": (
                self.resolved_history_length
            ),
            "has_feature_mask": (
                self.feature_mask is not None
            ),
            "has_history_mask": (
                self.history_mask is not None
            ),
            "has_history_time_points": (
                self.history_time_points is not None
            ),
            "history_time_points_shared": (
                self.history_time_points is not None
                and self.history_time_points.ndim == 1
            ),
            "source_fingerprint": (
                self.source_fingerprint
            ),
        }

    def metadata_fingerprint(self) -> str:
        return _fingerprint(
            self.metadata_dict()
        )

    def value_fingerprint(self) -> str:
        tensors: dict[str, torch.Tensor] = {
            "lag_values": self.lag_values,
            "feature_mask": (
                self.effective_feature_mask
            ),
        }

        if self.history_mask is not None:
            tensors["history_mask"] = (
                self.history_mask
            )

        if self.history_time_points is not None:
            tensors["history_time_points"] = (
                self.history_time_points
            )

        return _tensor_fingerprint(tensors)

    def to(
        self,
        device: torch.device | str,
    ) -> "LagMemoryBatch":
        return LagMemoryBatch(
            lag_values=self.lag_values.to(
                device=device
            ),
            lag_feature_names=self.lag_feature_names,
            feature_mask=(
                self.feature_mask.to(device=device)
                if self.feature_mask is not None
                else None
            ),
            source_history_length=(
                self.source_history_length
            ),
            history_mask=(
                self.history_mask.to(device=device)
                if self.history_mask is not None
                else None
            ),
            history_time_points=(
                self.history_time_points.to(
                    device=device
                )
                if self.history_time_points is not None
                else None
            ),
            source_fingerprint=self.source_fingerprint,
            schema_version=self.schema_version,
        )


# =============================================================================
# Metadata-preserving lag-memory output
# =============================================================================


@dataclass(slots=True, frozen=True)
class LagMemoryEncoding:
    """
    Encoded memory state with aligned lag/history metadata.

    ``source_batch`` is retained intact so downstream fusion and explanation
    code can trace each memory state back to its ordered lag features and
    source history metadata.
    """

    memory_state: torch.Tensor
    source_batch: LagMemoryBatch

    lag_feature_states: torch.Tensor | None
    lag_weights: torch.Tensor | None

    encoder_architecture_fingerprint: str
    lineage_fingerprint: str

    schema_version: str = (
        LAG_MEMORY_ENCODING_SCHEMA_VERSION
    )

    def __post_init__(self) -> None:
        if not isinstance(
            self.memory_state,
            torch.Tensor,
        ):
            raise TypeError(
                "memory_state must be a tensor."
            )

        if self.memory_state.ndim != 2:
            raise ValueError(
                "memory_state must have shape "
                "[items, hidden_dim]."
            )

        if not self.memory_state.dtype.is_floating_point:
            raise ValueError(
                "memory_state must use a floating-point dtype."
            )

        if int(self.memory_state.shape[0]) != (
            self.source_batch.item_count
        ):
            raise ValueError(
                "memory_state rows must align with source_batch."
            )

        if self.memory_state.device != (
            self.source_batch.device
        ):
            raise ValueError(
                "memory_state and source_batch must share one "
                "device."
            )

        _assert_finite_tensor(
            "memory_state",
            self.memory_state,
        )

        if self.lag_feature_states is not None:
            if not isinstance(
                self.lag_feature_states,
                torch.Tensor,
            ):
                raise TypeError(
                    "lag_feature_states must be a tensor or None."
                )

            expected_prefix = (
                self.source_batch.item_count,
                self.source_batch.num_lag_features,
            )

            if (
                self.lag_feature_states.ndim != 3
                or tuple(
                    self.lag_feature_states.shape[:2]
                ) != expected_prefix
            ):
                raise ValueError(
                    "lag_feature_states must have shape "
                    "[items, num_lag_features, hidden_dim]."
                )

            if self.lag_feature_states.device != (
                self.memory_state.device
            ):
                raise ValueError(
                    "lag_feature_states and memory_state must share "
                    "one device."
                )

            if not (
                self.lag_feature_states
                .dtype
                .is_floating_point
            ):
                raise ValueError(
                    "lag_feature_states must use a floating-point "
                    "dtype."
                )

            _assert_finite_tensor(
                "lag_feature_states",
                self.lag_feature_states,
            )

        if self.lag_weights is not None:
            if not isinstance(
                self.lag_weights,
                torch.Tensor,
            ):
                raise TypeError(
                    "lag_weights must be a tensor or None."
                )

            expected_shape = (
                self.source_batch.item_count,
                self.source_batch.num_lag_features,
            )

            if tuple(self.lag_weights.shape) != (
                expected_shape
            ):
                raise ValueError(
                    "lag_weights must have shape "
                    "[items, num_lag_features]."
                )

            if self.lag_weights.device != (
                self.memory_state.device
            ):
                raise ValueError(
                    "lag_weights and memory_state must share one "
                    "device."
                )

            if not self.lag_weights.dtype.is_floating_point:
                raise ValueError(
                    "lag_weights must use a floating-point dtype."
                )

            _assert_finite_tensor(
                "lag_weights",
                self.lag_weights,
            )

            if self.source_batch.item_count > 0:
                row_sums = self.lag_weights.sum(dim=1)

                if not torch.allclose(
                    row_sums,
                    torch.ones_like(row_sums),
                    atol=1e-6,
                    rtol=1e-6,
                ):
                    raise ValueError(
                        "lag_weights must sum to one for every "
                        "item."
                    )

                unavailable_weight = (
                    self.lag_weights.masked_select(
                        ~self.source_batch
                        .effective_feature_mask
                    )
                )

                if (
                    unavailable_weight.numel() > 0
                    and not torch.equal(
                        unavailable_weight,
                        torch.zeros_like(
                            unavailable_weight
                        ),
                    )
                ):
                    raise ValueError(
                        "Unavailable lag features must receive zero "
                        "weight."
                    )

        for name, value in (
            (
                "encoder_architecture_fingerprint",
                self.encoder_architecture_fingerprint,
            ),
            (
                "lineage_fingerprint",
                self.lineage_fingerprint,
            ),
            ("schema_version", self.schema_version),
        ):
            _require_nonempty_string(name, value)

    @property
    def item_count(self) -> int:
        return int(self.memory_state.shape[0])

    @property
    def hidden_dim(self) -> int:
        return int(self.memory_state.shape[1])

    @property
    def lag_feature_names(self) -> tuple[str, ...]:
        return self.source_batch.lag_feature_names

    @property
    def component_states(
        self,
    ) -> Mapping[str, torch.Tensor]:
        values: dict[str, torch.Tensor] = {
            "memory_state": self.memory_state,
        }

        if self.lag_feature_states is not None:
            values["lag_feature_states"] = (
                self.lag_feature_states
            )

        if self.lag_weights is not None:
            values["lag_weights"] = self.lag_weights

        return MappingProxyType(values)


# =============================================================================
# Lag-memory encoder
# =============================================================================


class LagMemoryEncoder(nn.Module):
    """
    Encode named lag features into one memory state per item.

    The feature-identity embedding ensures that equal numeric values in
    different lag columns remain semantically distinguishable.

    Aggregation is a masked mean. The resulting normalized weights are
    deterministic and interpretable; they are not learned temporal attention.
    """

    def __init__(
        self,
        *,
        lag_feature_names: Sequence[str],
        hidden_dim: int,
        dropout: float = 0.0,
        return_lag_states: bool = False,
        return_lag_weights: bool = False,
    ) -> None:
        super().__init__()

        names = tuple(lag_feature_names)

        _require_unique_strings(
            "lag_feature_names",
            names,
        )

        if not names:
            raise ValueError(
                "LagMemoryEncoder requires at least one lag "
                "feature."
            )

        _require_positive_int(
            "hidden_dim",
            hidden_dim,
        )

        dropout_value = _require_nonnegative_float(
            "dropout",
            dropout,
        )

        if dropout_value >= 1.0:
            raise ValueError(
                "dropout must be strictly smaller than 1."
            )

        if not isinstance(return_lag_states, bool):
            raise TypeError(
                "return_lag_states must be a Boolean."
            )

        if not isinstance(return_lag_weights, bool):
            raise TypeError(
                "return_lag_weights must be a Boolean."
            )

        self.lag_feature_names = names
        self.hidden_dim = hidden_dim
        self.dropout = dropout_value
        self.return_lag_states = return_lag_states
        self.return_lag_weights = return_lag_weights

        self.value_projection = nn.Linear(
            1,
            hidden_dim,
        )
        self.feature_identity = nn.Embedding(
            len(names),
            hidden_dim,
        )

        self.feature_encoder = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout_value),
            nn.LayerNorm(hidden_dim),
        )

        self.output_projection = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout_value),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
        )

        feature_indices = torch.arange(
            len(names),
            dtype=torch.long,
        )
        self.register_buffer(
            "feature_indices",
            feature_indices,
            persistent=True,
        )

    # ------------------------------------------------------------------
    # Configuration construction
    # ------------------------------------------------------------------

    @classmethod
    def from_config(
        cls,
        config: "MemoryConfig",
    ) -> "LagMemoryEncoder":
        """
        Construct from the frozen ``MemoryConfig`` contract.

        The lag encoder infers its input width from
        ``len(config.lag_feature_names)``. If ``input_dim`` is supplied
        explicitly, it must agree with that width.
        """

        from ..config import MemoryConfig
        from ..constants import MEMORY_ENCODER_LAG

        if not isinstance(config, MemoryConfig):
            raise TypeError(
                "config must be a MemoryConfig."
            )

        config.validate()

        if config.encoder_type != MEMORY_ENCODER_LAG:
            raise ValueError(
                "LagMemoryEncoder.from_config requires "
                "encoder_type='lag'."
            )

        if (
            config.input_dim is not None
            and config.input_dim
            != len(config.lag_feature_names)
        ):
            raise ValueError(
                "memory.input_dim must equal the number of "
                "lag_feature_names when provided for the lag "
                "encoder."
            )

        return cls(
            lag_feature_names=(
                config.lag_feature_names
            ),
            hidden_dim=config.hidden_dim,
            dropout=config.dropout,
            return_lag_states=(
                config.return_temporal_states
            ),
            return_lag_weights=(
                config.return_temporal_attention
            ),
        )

    # ------------------------------------------------------------------
    # Forward path
    # ------------------------------------------------------------------

    @property
    def device(self) -> torch.device:
        return self.value_projection.weight.device

    @property
    def dtype(self) -> torch.dtype:
        return self.value_projection.weight.dtype

    @property
    def num_lag_features(self) -> int:
        return len(self.lag_feature_names)

    def forward(
        self,
        batch: LagMemoryBatch,
    ) -> LagMemoryEncoding:
        if not isinstance(batch, LagMemoryBatch):
            raise TypeError(
                "LagMemoryEncoder requires a LagMemoryBatch; "
                "bare tensors are not accepted."
            )

        if batch.lag_feature_names != (
            self.lag_feature_names
        ):
            raise ValueError(
                "LagMemoryBatch feature names or ordering differ "
                "from the encoder configuration."
            )

        if batch.device != self.device:
            raise ValueError(
                "LagMemoryBatch and LagMemoryEncoder must be on the same device."
                "one device."
            )

        values = batch.lag_values.to(
            dtype=self.dtype
        )
        mask = batch.effective_feature_mask

        if tuple(values.shape) != (
            batch.item_count,
            self.num_lag_features,
        ):
            raise RuntimeError(
                "Lag-memory value shape differs from the encoder "
                "contract."
            )

        value_states = self.value_projection(
            values.unsqueeze(-1)
        )

        identity_states = self.feature_identity(
            self.feature_indices
        ).unsqueeze(0)

        feature_states = self.feature_encoder(
            value_states + identity_states
        )

        mask_values = mask.to(
            dtype=feature_states.dtype
        )
        denominators = mask_values.sum(
            dim=1,
            keepdim=True,
        )

        if batch.item_count > 0 and bool(
            (denominators <= 0.0).any().item()
        ):
            raise ValueError(
                "Every item must have at least one available lag "
                "feature."
            )

        lag_weights = (
            mask_values
            / denominators.clamp_min(1.0)
        )

        pooled = (
            feature_states
            * lag_weights.unsqueeze(-1)
        ).sum(dim=1)

        memory_state = self.output_projection(
            pooled
        )

        _assert_finite_tensor(
            "lag feature states",
            feature_states,
        )
        _assert_finite_tensor(
            "lag weights",
            lag_weights,
        )
        _assert_finite_tensor(
            "memory_state",
            memory_state,
        )

        architecture_fingerprint = (
            self.architecture_fingerprint()
        )

        return LagMemoryEncoding(
            memory_state=memory_state,
            source_batch=batch,
            lag_feature_states=(
                feature_states
                if self.return_lag_states
                else None
            ),
            lag_weights=(
                lag_weights
                if self.return_lag_weights
                else None
            ),
            encoder_architecture_fingerprint=(
                architecture_fingerprint
            ),
            lineage_fingerprint=(
                self.lineage_fingerprint(batch)
            ),
        )

    # ------------------------------------------------------------------
    # Identity and diagnostics
    # ------------------------------------------------------------------

    def architecture_dict(self) -> dict[str, Any]:
        return {
            "schema_version": (
                LAG_MEMORY_ENCODER_SCHEMA_VERSION
            ),
            "lag_feature_names": list(
                self.lag_feature_names
            ),
            "num_lag_features": (
                self.num_lag_features
            ),
            "hidden_dim": self.hidden_dim,
            "dropout": self.dropout,
            "return_lag_states": (
                self.return_lag_states
            ),
            "return_lag_weights": (
                self.return_lag_weights
            ),
            "value_encoding": (
                "scalar_linear_plus_feature_identity"
            ),
            "aggregation": "masked_mean",
            "output_projection": (
                "two_layer_gelu_layernorm"
            ),
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
        batch: LagMemoryBatch,
    ) -> str:
        if not isinstance(batch, LagMemoryBatch):
            raise TypeError(
                "batch must be a LagMemoryBatch."
            )

        return _fingerprint(
            {
                "encoder_architecture_fingerprint": (
                    self.architecture_fingerprint()
                ),
                "batch_metadata_fingerprint": (
                    batch.metadata_fingerprint()
                ),
                "batch_value_fingerprint": (
                    batch.value_fingerprint()
                ),
                "source_fingerprint": (
                    batch.source_fingerprint
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
                    f"Lag-memory tensor {name!r} contains NaN "
                    "or infinity."
                )

    def extra_repr(self) -> str:
        return (
            f"num_lag_features={self.num_lag_features}, "
            f"hidden_dim={self.hidden_dim}, "
            f"return_lag_states={self.return_lag_states}, "
            f"return_lag_weights={self.return_lag_weights}"
        )


__all__ = (
    "LAG_MEMORY_BATCH_SCHEMA_VERSION",
    "LAG_MEMORY_ENCODER_SCHEMA_VERSION",
    "LAG_MEMORY_ENCODING_SCHEMA_VERSION",
    "LagMemoryBatch",
    "LagMemoryEncoder",
    "LagMemoryEncoding",
)
