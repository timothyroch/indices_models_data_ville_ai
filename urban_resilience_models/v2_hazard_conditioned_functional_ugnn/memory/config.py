"""
Typed local configuration for shared urban temporal memory.

Target path:
    urban_resilience_models/
        v2_hazard_conditioned_functional_ugnn/
            memory/
                config.py

This module is the single source of truth for memory-specific runtime and
model-construction choices.

It configures:

- historical-input handling;
- baseline sequence encoders;
- GRU and LSTM sequence encoders;
- Transformer sequence encoders;
- hazard-independent temporal pooling;
- urban-memory assembly;
- hazard-query cross-attention retrieval;
- hazard-memory fusion;
- the complete memory subsystem.

Stable tensor and output semantics belong in ``memory.schemas``.
Trainable behavior belongs in the corresponding implementation folders.
The top-level V2 ``config.py`` should later import or adapt these local
contracts rather than maintain a second independent set of memory fields.

Configuration objects are frozen. Every instance is validated at construction,
and changes must use ``dataclasses.replace`` or the provided ``replace`` helper.
This prevents a logged configuration hash from silently becoming stale after
in-place mutation.

Design boundary
---------------
The local configuration describes conceptually valid memory architectures. It
does not claim that every recognized architecture is currently implemented.
Repository-wide implementation capability checks remain a top-level concern.

All generic pooling is hazard-independent. Hazard-conditioned temporal reading
is configured separately through ``TemporalCrossAttentionConfig`` and always
consumes the preserved ``[N, T, H]`` sequence.

No configuration option introduces a generic recurrent ``final_state``.
Shared downstream behavior uses either:

- the preserved temporal sequence; or
- an explicit temporal pooling output.
"""

from __future__ import annotations

from dataclasses import asdict
from dataclasses import dataclass
from dataclasses import field
from dataclasses import replace as dataclass_replace
from enum import StrEnum
from hashlib import sha256
import json
import math
from typing import Any, Final, Self, Sequence

from .schemas.hazard_queried_memory import (
    HazardMemoryFusionPolicy,
    TemporalQueryRetrievalHeadReduction,
    TemporalQueryRetrievalKind,
    TemporalQueryRetrievalZeroHistoryPolicy,
)
from .schemas.history_inputs import (
    HistoryMissingValuePolicy,
    HistoryZeroLengthPolicy,
)
from .schemas.sequence_encoding import (
    TemporalSequenceEncoderKind,
)
from .schemas.temporal_coordinates import (
    TemporalPaddingDirection,
)
from .schemas.temporal_pooling import (
    TemporalPoolingHeadReduction,
    TemporalPoolingKind,
    TemporalPoolingZeroHistoryPolicy,
)
from .schemas.urban_memory import (
    UrbanMemoryAssemblyPolicy,
)


# =============================================================================
# Configuration identity
# =============================================================================


MEMORY_CONFIG_SCHEMA_VERSION: Final[str] = "0.1"

MEMORY_CONFIG_SCIENTIFIC_INTERPRETATION: Final[str] = (
    "runtime_architecture_selection_not_implementation_or_causal_claim"
)


# =============================================================================
# Local runtime vocabularies
# =============================================================================


class BaselineSequenceEncoderKind(StrEnum):
    """Simple sequence-preserving baseline encoder."""

    IDENTITY = "identity"
    LINEAR_PROJECTION = "linear_projection"
    TEMPORAL_MLP = "temporal_mlp"


class RecurrentCellKind(StrEnum):
    """Recurrent cell used for temporal sequence encoding."""

    GRU = "gru"
    LSTM = "lstm"


class TransformerPositionalEncodingKind(StrEnum):
    """Position/time representation added before Transformer blocks."""

    NONE = "none"
    SINUSOIDAL = "sinusoidal"
    LEARNED = "learned"
    RELATIVE_TIME_MLP = "relative_time_mlp"


class MemoryActivation(StrEnum):
    """Activation used by local memory projections and feed-forward layers."""

    RELU = "relu"
    GELU = "gelu"
    SILU = "silu"
    TANH = "tanh"


class TemporalAttentionScoreKind(StrEnum):
    """Score function for generic pooling or query retrieval."""

    DOT_PRODUCT = "dot_product"
    SCALED_DOT_PRODUCT = "scaled_dot_product"
    ADDITIVE = "additive"
    BILINEAR = "bilinear"


class FusionGateActivation(StrEnum):
    """Activation used for learned hazard-memory fusion gates."""

    SIGMOID = "sigmoid"
    TANH = "tanh"


CANONICAL_BASELINE_SEQUENCE_ENCODER_KINDS: Final[
    tuple[str, ...]
] = tuple(
    value.value
    for value in BaselineSequenceEncoderKind
)

CANONICAL_RECURRENT_CELL_KINDS: Final[tuple[str, ...]] = tuple(
    value.value
    for value in RecurrentCellKind
)

CANONICAL_TRANSFORMER_POSITIONAL_ENCODING_KINDS: Final[
    tuple[str, ...]
] = tuple(
    value.value
    for value in TransformerPositionalEncodingKind
)

CANONICAL_MEMORY_ACTIVATIONS: Final[tuple[str, ...]] = tuple(
    value.value
    for value in MemoryActivation
)

CANONICAL_TEMPORAL_ATTENTION_SCORE_KINDS: Final[
    tuple[str, ...]
] = tuple(
    value.value
    for value in TemporalAttentionScoreKind
)

CANONICAL_FUSION_GATE_ACTIVATIONS: Final[tuple[str, ...]] = tuple(
    value.value
    for value in FusionGateActivation
)


# =============================================================================
# Validation helpers
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
            f"{name} must be a Boolean."
        )


def _require_positive_int(
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

    if value <= 0:
        raise ValueError(
            f"{name} must be strictly positive."
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


def _require_optional_positive_int(
    name: str,
    value: int | None,
) -> None:
    if value is None:
        return

    _require_positive_int(
        name,
        value,
    )


def _require_finite_number(
    name: str,
    value: int | float,
) -> float:
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

    return converted


def _require_positive_number(
    name: str,
    value: int | float,
) -> float:
    converted = _require_finite_number(
        name,
        value,
    )

    if converted <= 0.0:
        raise ValueError(
            f"{name} must be strictly positive."
        )

    return converted


def _require_probability(
    name: str,
    value: int | float,
    *,
    include_one: bool = False,
) -> float:
    converted = _require_finite_number(
        name,
        value,
    )

    upper_valid = (
        converted <= 1.0
        if include_one
        else converted < 1.0
    )

    if converted < 0.0 or not upper_valid:
        interval = (
            "[0, 1]"
            if include_one
            else "[0, 1)"
        )
        raise ValueError(
            f"{name} must lie in {interval}."
        )

    return converted


def _require_unique_nonempty_strings(
    name: str,
    values: Sequence[str],
) -> tuple[str, ...]:
    normalized = tuple(
        values
    )

    for index, value in enumerate(
        normalized
    ):
        _require_nonempty_string(
            f"{name}[{index}]",
            value,
        )

    if len(
        set(
            normalized
        )
    ) != len(
        normalized
    ):
        raise ValueError(
            f"{name} must not contain duplicates."
        )

    return normalized


def _normalize_enum(
    enum_type: type[StrEnum],
    value: StrEnum | str,
    *,
    name: str,
) -> StrEnum:
    if isinstance(
        value,
        enum_type,
    ):
        return value

    try:
        return enum_type(
            value
        )
    except (
        TypeError,
        ValueError,
    ) as error:
        allowed = tuple(
            member.value
            for member in enum_type
        )
        raise ValueError(
            f"Unknown {name} {value!r}. Expected one of {allowed}."
        ) from error


def _validate_head_reduction_weights(
    name: str,
    weights: tuple[float, ...] | None,
    *,
    num_heads: int,
    required: bool,
) -> tuple[float, ...] | None:
    if weights is None:
        if required:
            raise ValueError(
                f"{name} are required."
            )

        return None

    normalized = tuple(
        float(
            _require_finite_number(
                f"{name}[{index}]",
                value,
            )
        )
        for index, value in enumerate(
            weights
        )
    )

    if len(
        normalized
    ) != num_heads:
        raise ValueError(
            f"{name} must contain one value per head."
        )

    if any(
        value < 0.0
        for value in normalized
    ):
        raise ValueError(
            f"{name} must be nonnegative."
        )

    if not math.isclose(
        sum(
            normalized
        ),
        1.0,
        rel_tol=1e-6,
        abs_tol=1e-7,
    ):
        raise ValueError(
            f"{name} must sum to one."
        )

    return normalized


def _encoder_schema_kind(
    config: "TemporalSequenceEncoderConfig",
) -> TemporalSequenceEncoderKind:
    return config.encoder_kind


# =============================================================================
# Serialization mixin
# =============================================================================


class MemoryConfigMixin:
    """Immutable serialization, hashing, and validated replacement helpers."""

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

    def config_hash(
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

    def replace(
        self,
        **changes: Any,
    ) -> Self:
        return dataclass_replace(
            self,
            **changes,
        )


# =============================================================================
# Historical-input configuration
# =============================================================================


@dataclass(
    slots=True,
    frozen=True,
)
class HistoryInputConfig(
    MemoryConfigMixin
):
    """Controls model-facing history padding, missingness, and cold starts."""

    padding_direction: (
        TemporalPaddingDirection
        | str
    ) = TemporalPaddingDirection.RIGHT

    missing_value_policy: (
        HistoryMissingValuePolicy
        | str
    ) = HistoryMissingValuePolicy.UPSTREAM_IMPUTED

    zero_length_policy: (
        HistoryZeroLengthPolicy
        | str
    ) = HistoryZeroLengthPolicy.ERROR

    require_finite_values: bool = True
    require_canonical_zero_padding: bool = True

    def __post_init__(
        self,
    ) -> None:
        object.__setattr__(
            self,
            "padding_direction",
            _normalize_enum(
                TemporalPaddingDirection,
                self.padding_direction,
                name="padding_direction",
            ),
        )
        object.__setattr__(
            self,
            "missing_value_policy",
            _normalize_enum(
                HistoryMissingValuePolicy,
                self.missing_value_policy,
                name="missing_value_policy",
            ),
        )
        object.__setattr__(
            self,
            "zero_length_policy",
            _normalize_enum(
                HistoryZeroLengthPolicy,
                self.zero_length_policy,
                name="zero_length_policy",
            ),
        )

        _require_boolean(
            "require_finite_values",
            self.require_finite_values,
        )
        _require_boolean(
            "require_canonical_zero_padding",
            self.require_canonical_zero_padding,
        )

        if not self.require_finite_values:
            raise ValueError(
                "Shared memory schemas require finite model-facing "
                "history values."
            )

        if not self.require_canonical_zero_padding:
            raise ValueError(
                "Shared memory schemas require canonical zero padding."
            )

        if (
            self.zero_length_policy
            == HistoryZeroLengthPolicy.ALLOW_ZERO_HISTORY
            and self.padding_direction
            == TemporalPaddingDirection.NONE
        ):
            raise ValueError(
                "allow_zero_history is incompatible with "
                "padding_direction='none'."
            )


# =============================================================================
# Baseline sequence encoder
# =============================================================================


@dataclass(
    slots=True,
    frozen=True,
)
class BaselineSequenceEncoderConfig(
    MemoryConfigMixin
):
    """
    Simple sequence-preserving baseline encoder.

    ``identity`` preserves the feature dimension and performs no projection.

    ``linear_projection`` applies one trainable projection independently at
    each valid timestep.

    ``temporal_mlp`` applies a shared per-timestep MLP. Temporal aggregation
    remains the responsibility of ``TemporalPoolingConfig``.
    """

    kind: BaselineSequenceEncoderKind | str = (
        BaselineSequenceEncoderKind.IDENTITY
    )

    input_dim: int = 1
    output_dim: int = 1

    hidden_dim: int | None = None
    num_hidden_layers: int = 1

    activation: MemoryActivation | str = (
        MemoryActivation.GELU
    )
    dropout: float = 0.0
    layer_normalization: bool = False
    use_bias: bool = True

    def __post_init__(
        self,
    ) -> None:
        object.__setattr__(
            self,
            "kind",
            _normalize_enum(
                BaselineSequenceEncoderKind,
                self.kind,
                name="baseline encoder kind",
            ),
        )
        object.__setattr__(
            self,
            "activation",
            _normalize_enum(
                MemoryActivation,
                self.activation,
                name="baseline activation",
            ),
        )

        _require_positive_int(
            "baseline input_dim",
            self.input_dim,
        )
        _require_positive_int(
            "baseline output_dim",
            self.output_dim,
        )
        _require_optional_positive_int(
            "baseline hidden_dim",
            self.hidden_dim,
        )
        _require_positive_int(
            "baseline num_hidden_layers",
            self.num_hidden_layers,
        )
        object.__setattr__(
            self,
            "dropout",
            _require_probability(
                "baseline dropout",
                self.dropout,
            ),
        )
        _require_boolean(
            "baseline layer_normalization",
            self.layer_normalization,
        )
        _require_boolean(
            "baseline use_bias",
            self.use_bias,
        )

        if self.kind == BaselineSequenceEncoderKind.IDENTITY:
            if self.output_dim != self.input_dim:
                raise ValueError(
                    "identity baseline requires output_dim == input_dim."
                )

            if self.hidden_dim is not None:
                raise ValueError(
                    "identity baseline requires hidden_dim=None."
                )

            if self.num_hidden_layers != 1:
                raise ValueError(
                    "identity baseline requires num_hidden_layers=1."
                )

            if self.dropout != 0.0:
                raise ValueError(
                    "identity baseline requires dropout=0."
                )

        elif (
            self.kind
            == BaselineSequenceEncoderKind.LINEAR_PROJECTION
        ):
            if self.hidden_dim is not None:
                raise ValueError(
                    "linear_projection baseline requires hidden_dim=None."
                )

            if self.num_hidden_layers != 1:
                raise ValueError(
                    "linear_projection baseline requires "
                    "num_hidden_layers=1."
                )

        else:
            if self.hidden_dim is None:
                raise ValueError(
                    "temporal_mlp baseline requires hidden_dim."
                )

    @property
    def schema_encoder_kind(
        self,
    ) -> TemporalSequenceEncoderKind:
        if self.kind == BaselineSequenceEncoderKind.TEMPORAL_MLP:
            return TemporalSequenceEncoderKind.TEMPORAL_MLP

        return TemporalSequenceEncoderKind.IDENTITY_SEQUENCE


# =============================================================================
# Recurrent sequence encoder
# =============================================================================


@dataclass(
    slots=True,
    frozen=True,
)
class RecurrentSequenceEncoderConfig(
    MemoryConfigMixin
):
    """GRU or LSTM sequence encoder configuration."""

    cell_kind: RecurrentCellKind | str = (
        RecurrentCellKind.GRU
    )

    input_dim: int = 1
    hidden_dim: int = 64
    num_layers: int = 1

    dropout: float = 0.0
    bidirectional: bool = False
    use_bias: bool = True

    input_projection_dim: int | None = None
    layer_normalization: bool = False

    pack_sequences: bool = True
    enforce_sorted_lengths: bool = False

    def __post_init__(
        self,
    ) -> None:
        object.__setattr__(
            self,
            "cell_kind",
            _normalize_enum(
                RecurrentCellKind,
                self.cell_kind,
                name="recurrent cell kind",
            ),
        )

        _require_positive_int(
            "recurrent input_dim",
            self.input_dim,
        )
        _require_positive_int(
            "recurrent hidden_dim",
            self.hidden_dim,
        )
        _require_positive_int(
            "recurrent num_layers",
            self.num_layers,
        )
        object.__setattr__(
            self,
            "dropout",
            _require_probability(
                "recurrent dropout",
                self.dropout,
            ),
        )
        _require_boolean(
            "recurrent bidirectional",
            self.bidirectional,
        )
        _require_boolean(
            "recurrent use_bias",
            self.use_bias,
        )
        _require_optional_positive_int(
            "recurrent input_projection_dim",
            self.input_projection_dim,
        )
        _require_boolean(
            "recurrent layer_normalization",
            self.layer_normalization,
        )
        _require_boolean(
            "recurrent pack_sequences",
            self.pack_sequences,
        )
        _require_boolean(
            "recurrent enforce_sorted_lengths",
            self.enforce_sorted_lengths,
        )

        if self.num_layers == 1 and self.dropout != 0.0:
            raise ValueError(
                "PyTorch recurrent dropout is applied only between "
                "stacked layers; num_layers=1 requires dropout=0."
            )

        if (
            self.enforce_sorted_lengths
            and not self.pack_sequences
        ):
            raise ValueError(
                "enforce_sorted_lengths requires pack_sequences=True."
            )

    @property
    def output_dim(
        self,
    ) -> int:
        return (
            self.hidden_dim
            * (
                2
                if self.bidirectional
                else 1
            )
        )

    @property
    def schema_encoder_kind(
        self,
    ) -> TemporalSequenceEncoderKind:
        if self.cell_kind == RecurrentCellKind.GRU:
            return TemporalSequenceEncoderKind.GRU

        return TemporalSequenceEncoderKind.LSTM


# =============================================================================
# Transformer sequence encoder
# =============================================================================


@dataclass(
    slots=True,
    frozen=True,
)
class TransformerSequenceEncoderConfig(
    MemoryConfigMixin
):
    """Transformer sequence encoder configuration."""

    input_dim: int = 1
    model_dim: int = 64

    num_layers: int = 2
    num_heads: int = 4
    feed_forward_dim: int = 256

    dropout: float = 0.1
    attention_dropout: float = 0.0

    activation: MemoryActivation | str = (
        MemoryActivation.GELU
    )
    positional_encoding: (
        TransformerPositionalEncodingKind
        | str
    ) = TransformerPositionalEncodingKind.RELATIVE_TIME_MLP

    causal_attention: bool = True
    norm_first: bool = True
    use_bias: bool = True

    max_sequence_length: int | None = None

    def __post_init__(
        self,
    ) -> None:
        object.__setattr__(
            self,
            "activation",
            _normalize_enum(
                MemoryActivation,
                self.activation,
                name="Transformer activation",
            ),
        )
        object.__setattr__(
            self,
            "positional_encoding",
            _normalize_enum(
                TransformerPositionalEncodingKind,
                self.positional_encoding,
                name="Transformer positional encoding",
            ),
        )

        _require_positive_int(
            "Transformer input_dim",
            self.input_dim,
        )
        _require_positive_int(
            "Transformer model_dim",
            self.model_dim,
        )
        _require_positive_int(
            "Transformer num_layers",
            self.num_layers,
        )
        _require_positive_int(
            "Transformer num_heads",
            self.num_heads,
        )
        _require_positive_int(
            "Transformer feed_forward_dim",
            self.feed_forward_dim,
        )
        object.__setattr__(
            self,
            "dropout",
            _require_probability(
                "Transformer dropout",
                self.dropout,
            ),
        )
        object.__setattr__(
            self,
            "attention_dropout",
            _require_probability(
                "Transformer attention_dropout",
                self.attention_dropout,
            ),
        )
        _require_boolean(
            "Transformer causal_attention",
            self.causal_attention,
        )
        _require_boolean(
            "Transformer norm_first",
            self.norm_first,
        )
        _require_boolean(
            "Transformer use_bias",
            self.use_bias,
        )
        _require_optional_positive_int(
            "Transformer max_sequence_length",
            self.max_sequence_length,
        )

        if self.model_dim % self.num_heads != 0:
            raise ValueError(
                "Transformer model_dim must be divisible by num_heads."
            )

        if self.feed_forward_dim < self.model_dim:
            raise ValueError(
                "Transformer feed_forward_dim must be at least model_dim."
            )

        if (
            self.positional_encoding
            == TransformerPositionalEncodingKind.LEARNED
            and self.max_sequence_length is None
        ):
            raise ValueError(
                "Learned positional encoding requires "
                "max_sequence_length."
            )

    @property
    def output_dim(
        self,
    ) -> int:
        return self.model_dim

    @property
    def schema_encoder_kind(
        self,
    ) -> TemporalSequenceEncoderKind:
        return TemporalSequenceEncoderKind.TRANSFORMER


# =============================================================================
# Sequence encoder dispatch configuration
# =============================================================================


@dataclass(
    slots=True,
    frozen=True,
)
class TemporalSequenceEncoderConfig(
    MemoryConfigMixin
):
    """Select exactly one sequence-preserving encoder configuration."""

    encoder_kind: (
        TemporalSequenceEncoderKind
        | str
    ) = TemporalSequenceEncoderKind.IDENTITY_SEQUENCE

    baseline: BaselineSequenceEncoderConfig | None = field(
        default_factory=BaselineSequenceEncoderConfig
    )
    recurrent: RecurrentSequenceEncoderConfig | None = None
    transformer: TransformerSequenceEncoderConfig | None = None

    def __post_init__(
        self,
    ) -> None:
        object.__setattr__(
            self,
            "encoder_kind",
            _normalize_enum(
                TemporalSequenceEncoderKind,
                self.encoder_kind,
                name="sequence encoder kind",
            ),
        )

        active = tuple(
            name
            for name, config in (
                (
                    "baseline",
                    self.baseline,
                ),
                (
                    "recurrent",
                    self.recurrent,
                ),
                (
                    "transformer",
                    self.transformer,
                ),
            )
            if config is not None
        )

        if len(
            active
        ) != 1:
            raise ValueError(
                "Exactly one of baseline, recurrent, or transformer "
                f"must be configured; observed {active}."
            )

        if self.baseline is not None:
            if not isinstance(
                self.baseline,
                BaselineSequenceEncoderConfig,
            ):
                raise TypeError(
                    "baseline must be a "
                    "BaselineSequenceEncoderConfig or None."
                )

            expected = (
                self.baseline
                .schema_encoder_kind
            )
            if self.encoder_kind != expected:
                raise ValueError(
                    "encoder_kind does not match the active baseline "
                    f"configuration; expected {expected.value!r}."
                )

        if self.recurrent is not None:
            if not isinstance(
                self.recurrent,
                RecurrentSequenceEncoderConfig,
            ):
                raise TypeError(
                    "recurrent must be a "
                    "RecurrentSequenceEncoderConfig or None."
                )

            expected = (
                self.recurrent
                .schema_encoder_kind
            )
            if self.encoder_kind != expected:
                raise ValueError(
                    "encoder_kind does not match the active recurrent "
                    f"configuration; expected {expected.value!r}."
                )

        if self.transformer is not None:
            if not isinstance(
                self.transformer,
                TransformerSequenceEncoderConfig,
            ):
                raise TypeError(
                    "transformer must be a "
                    "TransformerSequenceEncoderConfig or None."
                )

            if (
                self.encoder_kind
                != TemporalSequenceEncoderKind.TRANSFORMER
            ):
                raise ValueError(
                    "Transformer configuration requires "
                    "encoder_kind='transformer'."
                )

    @classmethod
    def identity(
        cls,
        *,
        input_dim: int,
    ) -> "TemporalSequenceEncoderConfig":
        return cls(
            encoder_kind=(
                TemporalSequenceEncoderKind.IDENTITY_SEQUENCE
            ),
            baseline=BaselineSequenceEncoderConfig(
                kind=BaselineSequenceEncoderKind.IDENTITY,
                input_dim=input_dim,
                output_dim=input_dim,
            ),
        )

    @classmethod
    def temporal_mlp(
        cls,
        *,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        num_hidden_layers: int = 1,
        dropout: float = 0.0,
    ) -> "TemporalSequenceEncoderConfig":
        return cls(
            encoder_kind=(
                TemporalSequenceEncoderKind.TEMPORAL_MLP
            ),
            baseline=BaselineSequenceEncoderConfig(
                kind=BaselineSequenceEncoderKind.TEMPORAL_MLP,
                input_dim=input_dim,
                hidden_dim=hidden_dim,
                output_dim=output_dim,
                num_hidden_layers=num_hidden_layers,
                dropout=dropout,
            ),
        )

    @classmethod
    def recurrent_encoder(
        cls,
        config: RecurrentSequenceEncoderConfig,
    ) -> "TemporalSequenceEncoderConfig":
        if not isinstance(
            config,
            RecurrentSequenceEncoderConfig,
        ):
            raise TypeError(
                "config must be a RecurrentSequenceEncoderConfig."
            )

        return cls(
            encoder_kind=(
                config.schema_encoder_kind
            ),
            baseline=None,
            recurrent=config,
        )

    @classmethod
    def transformer_encoder(
        cls,
        config: TransformerSequenceEncoderConfig,
    ) -> "TemporalSequenceEncoderConfig":
        if not isinstance(
            config,
            TransformerSequenceEncoderConfig,
        ):
            raise TypeError(
                "config must be a "
                "TransformerSequenceEncoderConfig."
            )

        return cls(
            encoder_kind=(
                TemporalSequenceEncoderKind.TRANSFORMER
            ),
            baseline=None,
            transformer=config,
        )

    @property
    def input_dim(
        self,
    ) -> int:
        if self.baseline is not None:
            return self.baseline.input_dim

        if self.recurrent is not None:
            return self.recurrent.input_dim

        assert self.transformer is not None
        return self.transformer.input_dim

    @property
    def output_dim(
        self,
    ) -> int:
        if self.baseline is not None:
            return self.baseline.output_dim

        if self.recurrent is not None:
            return self.recurrent.output_dim

        assert self.transformer is not None
        return self.transformer.output_dim


# =============================================================================
# Hazard-independent temporal pooling
# =============================================================================


@dataclass(
    slots=True,
    frozen=True,
)
class TemporalPoolingConfig(
    MemoryConfigMixin
):
    """Configuration for generic hazard-independent temporal pooling."""

    kind: TemporalPoolingKind | str = (
        TemporalPoolingKind.MASKED_MEAN
    )

    output_dim: int = 64
    project_output: bool = False

    num_heads: int = 1
    head_reduction: (
        TemporalPoolingHeadReduction
        | str
    ) = TemporalPoolingHeadReduction.SINGLE_HEAD

    head_reduction_weights: tuple[float, ...] | None = None

    score_kind: TemporalAttentionScoreKind | str = (
        TemporalAttentionScoreKind.ADDITIVE
    )
    score_hidden_dim: int | None = None

    dropout: float = 0.0

    zero_history_policy: (
        TemporalPoolingZeroHistoryPolicy
        | str
    ) = TemporalPoolingZeroHistoryPolicy.ERROR

    def __post_init__(
        self,
    ) -> None:
        object.__setattr__(
            self,
            "kind",
            _normalize_enum(
                TemporalPoolingKind,
                self.kind,
                name="temporal pooling kind",
            ),
        )
        object.__setattr__(
            self,
            "head_reduction",
            _normalize_enum(
                TemporalPoolingHeadReduction,
                self.head_reduction,
                name="pooling head reduction",
            ),
        )
        object.__setattr__(
            self,
            "score_kind",
            _normalize_enum(
                TemporalAttentionScoreKind,
                self.score_kind,
                name="pooling score kind",
            ),
        )
        object.__setattr__(
            self,
            "zero_history_policy",
            _normalize_enum(
                TemporalPoolingZeroHistoryPolicy,
                self.zero_history_policy,
                name="pooling zero-history policy",
            ),
        )

        _require_positive_int(
            "pooling output_dim",
            self.output_dim,
        )
        _require_boolean(
            "pooling project_output",
            self.project_output,
        )
        _require_positive_int(
            "pooling num_heads",
            self.num_heads,
        )
        _require_optional_positive_int(
            "pooling score_hidden_dim",
            self.score_hidden_dim,
        )
        object.__setattr__(
            self,
            "dropout",
            _require_probability(
                "pooling dropout",
                self.dropout,
            ),
        )

        deterministic_single_head = {
            TemporalPoolingKind.LAST_VALID,
            TemporalPoolingKind.MASKED_MEAN,
            TemporalPoolingKind.MASKED_MAX,
            TemporalPoolingKind.LEARNED_ATTENTION,
        }

        if self.kind in deterministic_single_head:
            if self.num_heads != 1:
                raise ValueError(
                    f"pooling kind {self.kind.value!r} requires "
                    "num_heads=1."
                )

            if (
                self.head_reduction
                != TemporalPoolingHeadReduction.SINGLE_HEAD
            ):
                raise ValueError(
                    f"pooling kind {self.kind.value!r} requires "
                    "head_reduction='single_head'."
                )

        if self.kind == TemporalPoolingKind.MULTIHEAD_ATTENTION:
            if self.num_heads <= 1:
                raise ValueError(
                    "multihead_attention pooling requires "
                    "num_heads > 1."
                )

            if (
                self.head_reduction
                == TemporalPoolingHeadReduction.SINGLE_HEAD
            ):
                raise ValueError(
                    "multihead_attention pooling requires a multihead "
                    "reduction policy."
                )

        if (
            self.head_reduction
            == TemporalPoolingHeadReduction.SINGLE_HEAD
            and self.num_heads != 1
        ):
            raise ValueError(
                "single_head pooling reduction requires num_heads=1."
            )

        if (
            self.head_reduction
            != TemporalPoolingHeadReduction.SINGLE_HEAD
            and self.num_heads <= 1
        ):
            raise ValueError(
                "Multihead pooling reduction requires num_heads > 1."
            )

        weighted = (
            self.head_reduction
            == TemporalPoolingHeadReduction.WEIGHTED_MEAN
        )
        normalized_weights = (
            _validate_head_reduction_weights(
                "pooling head_reduction_weights",
                self.head_reduction_weights,
                num_heads=self.num_heads,
                required=weighted,
            )
        )
        object.__setattr__(
            self,
            "head_reduction_weights",
            normalized_weights,
        )

        if (
            not weighted
            and normalized_weights is not None
        ):
            raise ValueError(
                "pooling head_reduction_weights are valid only for "
                "weighted_mean."
            )

        attention_kinds = {
            TemporalPoolingKind.LEARNED_ATTENTION,
            TemporalPoolingKind.MULTIHEAD_ATTENTION,
        }

        if self.kind in attention_kinds:
            if self.score_hidden_dim is None:
                raise ValueError(
                    "Attention pooling requires score_hidden_dim."
                )
        else:
            if self.score_hidden_dim is not None:
                raise ValueError(
                    "score_hidden_dim is valid only for learned "
                    "attention pooling."
                )

            if self.dropout != 0.0:
                raise ValueError(
                    "Pooling dropout is valid only for learned "
                    "attention pooling."
                )

    def validate_against_sequence_dim(
        self,
        sequence_dim: int,
    ) -> None:
        _require_positive_int(
            "sequence_dim",
            sequence_dim,
        )

        if (
            not self.project_output
            and self.output_dim != sequence_dim
        ):
            raise ValueError(
                "Pooling without output projection requires "
                "output_dim == sequence encoder output_dim."
            )


# =============================================================================
# Urban-memory assembly
# =============================================================================


@dataclass(
    slots=True,
    frozen=True,
)
class UrbanMemoryConfig(
    MemoryConfigMixin
):
    """Complete sequence encoding and generic-pooling configuration."""

    history_inputs: HistoryInputConfig = field(
        default_factory=HistoryInputConfig
    )

    sequence_encoder: TemporalSequenceEncoderConfig = field(
        default_factory=lambda: (
            TemporalSequenceEncoderConfig.identity(
                input_dim=1
            )
        )
    )

    assembly_policy: (
        UrbanMemoryAssemblyPolicy
        | str
    ) = UrbanMemoryAssemblyPolicy.SEQUENCE_ONLY

    temporal_pooling: TemporalPoolingConfig | None = None

    def __post_init__(
        self,
    ) -> None:
        if not isinstance(
            self.history_inputs,
            HistoryInputConfig,
        ):
            raise TypeError(
                "history_inputs must be a HistoryInputConfig."
            )

        if not isinstance(
            self.sequence_encoder,
            TemporalSequenceEncoderConfig,
        ):
            raise TypeError(
                "sequence_encoder must be a "
                "TemporalSequenceEncoderConfig."
            )

        object.__setattr__(
            self,
            "assembly_policy",
            _normalize_enum(
                UrbanMemoryAssemblyPolicy,
                self.assembly_policy,
                name="urban-memory assembly policy",
            ),
        )

        if (
            self.temporal_pooling is not None
            and not isinstance(
                self.temporal_pooling,
                TemporalPoolingConfig,
            )
        ):
            raise TypeError(
                "temporal_pooling must be a TemporalPoolingConfig "
                "or None."
            )

        if (
            self.assembly_policy
            == UrbanMemoryAssemblyPolicy.SEQUENCE_ONLY
        ):
            if self.temporal_pooling is not None:
                raise ValueError(
                    "sequence_only assembly requires "
                    "temporal_pooling=None."
                )

        else:
            if self.temporal_pooling is None:
                raise ValueError(
                    "sequence_with_generic_pooling requires a "
                    "TemporalPoolingConfig."
                )

            self.temporal_pooling.validate_against_sequence_dim(
                self.sequence_encoder.output_dim
            )

            if (
                self.history_inputs.zero_length_policy
                == HistoryZeroLengthPolicy.ALLOW_ZERO_HISTORY
                and self.temporal_pooling.zero_history_policy
                == TemporalPoolingZeroHistoryPolicy.ERROR
            ):
                raise ValueError(
                    "Urban memory allows zero-history inputs but the "
                    "pooling policy rejects them."
                )

    @property
    def sequence_dim(
        self,
    ) -> int:
        return self.sequence_encoder.output_dim

    @property
    def generic_memory_dim(
        self,
    ) -> int | None:
        if self.temporal_pooling is None:
            return None

        return self.temporal_pooling.output_dim

    @property
    def has_generic_pooling(
        self,
    ) -> bool:
        return self.temporal_pooling is not None


# =============================================================================
# Hazard-query temporal cross-attention
# =============================================================================


@dataclass(
    slots=True,
    frozen=True,
)
class TemporalCrossAttentionConfig(
    MemoryConfigMixin
):
    """Hazard-query retrieval over the preserved temporal sequence."""

    retrieval_kind: (
        TemporalQueryRetrievalKind
        | str
    ) = TemporalQueryRetrievalKind.MULTIHEAD_CROSS_ATTENTION

    query_dim: int = 64
    output_dim: int = 64

    num_heads: int = 4
    head_reduction: (
        TemporalQueryRetrievalHeadReduction
        | str
    ) = TemporalQueryRetrievalHeadReduction.CONCAT_PROJECTION

    head_reduction_weights: tuple[float, ...] | None = None

    score_kind: TemporalAttentionScoreKind | str = (
        TemporalAttentionScoreKind.SCALED_DOT_PRODUCT
    )
    key_dim: int | None = None
    value_dim: int | None = None

    attention_dropout: float = 0.0
    output_dropout: float = 0.0
    use_bias: bool = True

    zero_history_policy: (
        TemporalQueryRetrievalZeroHistoryPolicy
        | str
    ) = TemporalQueryRetrievalZeroHistoryPolicy.ERROR

    def __post_init__(
        self,
    ) -> None:
        object.__setattr__(
            self,
            "retrieval_kind",
            _normalize_enum(
                TemporalQueryRetrievalKind,
                self.retrieval_kind,
                name="retrieval kind",
            ),
        )
        object.__setattr__(
            self,
            "head_reduction",
            _normalize_enum(
                TemporalQueryRetrievalHeadReduction,
                self.head_reduction,
                name="retrieval head reduction",
            ),
        )
        object.__setattr__(
            self,
            "score_kind",
            _normalize_enum(
                TemporalAttentionScoreKind,
                self.score_kind,
                name="retrieval score kind",
            ),
        )
        object.__setattr__(
            self,
            "zero_history_policy",
            _normalize_enum(
                TemporalQueryRetrievalZeroHistoryPolicy,
                self.zero_history_policy,
                name="retrieval zero-history policy",
            ),
        )

        _require_positive_int(
            "retrieval query_dim",
            self.query_dim,
        )
        _require_positive_int(
            "retrieval output_dim",
            self.output_dim,
        )
        _require_positive_int(
            "retrieval num_heads",
            self.num_heads,
        )
        _require_optional_positive_int(
            "retrieval key_dim",
            self.key_dim,
        )
        _require_optional_positive_int(
            "retrieval value_dim",
            self.value_dim,
        )
        object.__setattr__(
            self,
            "attention_dropout",
            _require_probability(
                "retrieval attention_dropout",
                self.attention_dropout,
            ),
        )
        object.__setattr__(
            self,
            "output_dropout",
            _require_probability(
                "retrieval output_dropout",
                self.output_dropout,
            ),
        )
        _require_boolean(
            "retrieval use_bias",
            self.use_bias,
        )

        if (
            self.head_reduction
            == TemporalQueryRetrievalHeadReduction.SINGLE_HEAD
        ):
            if self.num_heads != 1:
                raise ValueError(
                    "single_head retrieval requires num_heads=1."
                )
        elif self.num_heads <= 1:
            raise ValueError(
                "Multihead retrieval reduction requires num_heads > 1."
            )

        if (
            self.retrieval_kind
            != TemporalQueryRetrievalKind.MULTIHEAD_CROSS_ATTENTION
            and self.num_heads != 1
        ):
            raise ValueError(
                "Only multihead_cross_attention supports num_heads > 1."
            )

        weighted = (
            self.head_reduction
            == TemporalQueryRetrievalHeadReduction.WEIGHTED_MEAN
        )
        normalized_weights = (
            _validate_head_reduction_weights(
                "retrieval head_reduction_weights",
                self.head_reduction_weights,
                num_heads=self.num_heads,
                required=weighted,
            )
        )
        object.__setattr__(
            self,
            "head_reduction_weights",
            normalized_weights,
        )

        if (
            not weighted
            and normalized_weights is not None
        ):
            raise ValueError(
                "retrieval head_reduction_weights are valid only for "
                "weighted_mean."
            )

        if (
            self.head_reduction
            == TemporalQueryRetrievalHeadReduction.CONCAT_PROJECTION
            and self.output_dim % self.num_heads != 0
        ):
            raise ValueError(
                "concat_projection retrieval requires output_dim "
                "divisible by num_heads."
            )

    def validate_against_sequence_dim(
        self,
        sequence_dim: int,
    ) -> None:
        _require_positive_int(
            "sequence_dim",
            sequence_dim,
        )

        resolved_key_dim = (
            sequence_dim
            if self.key_dim is None
            else self.key_dim
        )
        resolved_value_dim = (
            sequence_dim
            if self.value_dim is None
            else self.value_dim
        )

        if (
            self.score_kind
            in {
                TemporalAttentionScoreKind.DOT_PRODUCT,
                TemporalAttentionScoreKind.SCALED_DOT_PRODUCT,
            }
            and resolved_key_dim % self.num_heads != 0
        ):
            raise ValueError(
                "Dot-product retrieval key dimension must be "
                "divisible by num_heads."
            )

        if (
            self.head_reduction
            == TemporalQueryRetrievalHeadReduction.CONCAT_PROJECTION
            and resolved_value_dim % self.num_heads != 0
        ):
            raise ValueError(
                "Concatenated multihead retrieval value dimension "
                "must be divisible by num_heads."
            )


# =============================================================================
# Hazard-memory fusion
# =============================================================================


@dataclass(
    slots=True,
    frozen=True,
)
class HazardMemoryFusionConfig(
    MemoryConfigMixin
):
    """Configuration for combining retrieval with declared memory components."""

    policy: HazardMemoryFusionPolicy | str = (
        HazardMemoryFusionPolicy.CONCAT_PROJECTION
    )

    output_dim: int = 64

    include_retrieved_context: bool = True
    include_generic_pooled_memory: bool = True
    include_hazard_query: bool = True
    include_current_node_state: bool = False

    current_node_dim: int | None = None
    projection_hidden_dim: int | None = None

    dropout: float = 0.0
    use_bias: bool = True

    gate_activation: FusionGateActivation | str = (
        FusionGateActivation.SIGMOID
    )

    preserve_components_for_diagnostics: bool = True

    def __post_init__(
        self,
    ) -> None:
        object.__setattr__(
            self,
            "policy",
            _normalize_enum(
                HazardMemoryFusionPolicy,
                self.policy,
                name="hazard-memory fusion policy",
            ),
        )
        object.__setattr__(
            self,
            "gate_activation",
            _normalize_enum(
                FusionGateActivation,
                self.gate_activation,
                name="fusion gate activation",
            ),
        )

        _require_positive_int(
            "fusion output_dim",
            self.output_dim,
        )
        _require_boolean(
            "include_retrieved_context",
            self.include_retrieved_context,
        )
        _require_boolean(
            "include_generic_pooled_memory",
            self.include_generic_pooled_memory,
        )
        _require_boolean(
            "include_hazard_query",
            self.include_hazard_query,
        )
        _require_boolean(
            "include_current_node_state",
            self.include_current_node_state,
        )
        _require_optional_positive_int(
            "fusion current_node_dim",
            self.current_node_dim,
        )
        _require_optional_positive_int(
            "fusion projection_hidden_dim",
            self.projection_hidden_dim,
        )
        object.__setattr__(
            self,
            "dropout",
            _require_probability(
                "fusion dropout",
                self.dropout,
            ),
        )
        _require_boolean(
            "fusion use_bias",
            self.use_bias,
        )
        _require_boolean(
            "preserve_components_for_diagnostics",
            self.preserve_components_for_diagnostics,
        )

        if not self.include_retrieved_context:
            raise ValueError(
                "Hazard-memory fusion must include retrieved context."
            )

        if self.include_current_node_state:
            if self.current_node_dim is None:
                raise ValueError(
                    "current_node_dim is required when current node "
                    "state is included."
                )
        elif self.current_node_dim is not None:
            raise ValueError(
                "current_node_dim must be None when current node "
                "state is excluded."
            )

        if self.policy == HazardMemoryFusionPolicy.RETRIEVED_ONLY:
            if (
                self.include_generic_pooled_memory
                or self.include_hazard_query
                or self.include_current_node_state
            ):
                raise ValueError(
                    "retrieved_only fusion forbids all additional "
                    "components."
                )

            if self.projection_hidden_dim is not None:
                raise ValueError(
                    "retrieved_only fusion requires "
                    "projection_hidden_dim=None."
                )

            if self.dropout != 0.0:
                raise ValueError(
                    "retrieved_only fusion requires dropout=0."
                )

        elif (
            self.policy
            == HazardMemoryFusionPolicy.GATED_FUSION
        ):
            component_count = sum(
                (
                    self.include_retrieved_context,
                    self.include_generic_pooled_memory,
                    self.include_hazard_query,
                    self.include_current_node_state,
                )
            )
            if component_count < 2:
                raise ValueError(
                    "gated_fusion requires at least two components."
                )

        elif (
            self.policy
            == HazardMemoryFusionPolicy.FILM_CONDITIONING
            and not self.include_hazard_query
        ):
            raise ValueError(
                "FiLM conditioning requires the hazard query."
            )

    def validate_against_dimensions(
        self,
        *,
        retrieval_dim: int,
        query_dim: int,
        generic_memory_dim: int | None,
    ) -> None:
        _require_positive_int(
            "retrieval_dim",
            retrieval_dim,
        )
        _require_positive_int(
            "query_dim",
            query_dim,
        )

        if (
            self.include_generic_pooled_memory
            and generic_memory_dim is None
        ):
            raise ValueError(
                "Fusion requests generic pooled memory, but urban "
                "memory has no temporal pooling."
            )

        if generic_memory_dim is not None:
            _require_positive_int(
                "generic_memory_dim",
                generic_memory_dim,
            )

        if (
            self.policy
            == HazardMemoryFusionPolicy.RETRIEVED_ONLY
            and self.output_dim != retrieval_dim
        ):
            raise ValueError(
                "retrieved_only fusion requires output_dim == "
                "retrieval output_dim."
            )

        if (
            self.policy
            == HazardMemoryFusionPolicy.PROJECTED_SUM
        ):
            if self.output_dim <= 0:
                raise ValueError(
                    "projected_sum requires a positive output_dim."
                )

        if (
            self.policy
            == HazardMemoryFusionPolicy.FILM_CONDITIONING
            and self.output_dim != retrieval_dim
        ):
            raise ValueError(
                "FiLM conditioning preserves the retrieved-context "
                "dimension; output_dim must equal retrieval_dim."
            )


# =============================================================================
# Complete memory subsystem
# =============================================================================


@dataclass(
    slots=True,
    frozen=True,
)
class MemoryConfig(
    MemoryConfigMixin
):
    """Complete local configuration for urban and hazard-queried memory."""

    urban_memory: UrbanMemoryConfig = field(
        default_factory=UrbanMemoryConfig
    )

    temporal_retrieval: TemporalCrossAttentionConfig | None = None
    hazard_fusion: HazardMemoryFusionConfig | None = None

    config_name: str = "memory_config"
    schema_version: str = MEMORY_CONFIG_SCHEMA_VERSION

    def __post_init__(
        self,
    ) -> None:
        if not isinstance(
            self.urban_memory,
            UrbanMemoryConfig,
        ):
            raise TypeError(
                "urban_memory must be an UrbanMemoryConfig."
            )

        if (
            self.temporal_retrieval is not None
            and not isinstance(
                self.temporal_retrieval,
                TemporalCrossAttentionConfig,
            )
        ):
            raise TypeError(
                "temporal_retrieval must be a "
                "TemporalCrossAttentionConfig or None."
            )

        if (
            self.hazard_fusion is not None
            and not isinstance(
                self.hazard_fusion,
                HazardMemoryFusionConfig,
            )
        ):
            raise TypeError(
                "hazard_fusion must be a "
                "HazardMemoryFusionConfig or None."
            )

        _require_nonempty_string(
            "config_name",
            self.config_name,
        )
        _require_nonempty_string(
            "schema_version",
            self.schema_version,
        )

        retrieval_enabled = (
            self.temporal_retrieval
            is not None
        )
        fusion_enabled = (
            self.hazard_fusion
            is not None
        )

        if retrieval_enabled != fusion_enabled:
            raise ValueError(
                "temporal_retrieval and hazard_fusion must either both "
                "be configured or both be None."
            )

        if self.temporal_retrieval is not None:
            self.temporal_retrieval.validate_against_sequence_dim(
                self.urban_memory.sequence_dim
            )

            assert self.hazard_fusion is not None
            self.hazard_fusion.validate_against_dimensions(
                retrieval_dim=(
                    self
                    .temporal_retrieval
                    .output_dim
                ),
                query_dim=(
                    self
                    .temporal_retrieval
                    .query_dim
                ),
                generic_memory_dim=(
                    self
                    .urban_memory
                    .generic_memory_dim
                ),
            )

            if (
                self
                .urban_memory
                .history_inputs
                .zero_length_policy
                == HistoryZeroLengthPolicy.ALLOW_ZERO_HISTORY
                and self
                .temporal_retrieval
                .zero_history_policy
                == TemporalQueryRetrievalZeroHistoryPolicy.ERROR
            ):
                raise ValueError(
                    "History inputs allow zero-history nodes, but "
                    "temporal retrieval rejects them."
                )

    @property
    def hazard_queried(
        self,
    ) -> bool:
        return self.temporal_retrieval is not None

    @property
    def sequence_dim(
        self,
    ) -> int:
        return self.urban_memory.sequence_dim

    @property
    def generic_memory_dim(
        self,
    ) -> int | None:
        return self.urban_memory.generic_memory_dim

    @property
    def final_memory_dim(
        self,
    ) -> int:
        if self.hazard_fusion is not None:
            return self.hazard_fusion.output_dim

        if self.generic_memory_dim is not None:
            return self.generic_memory_dim

        return self.sequence_dim

    @classmethod
    def baseline_sequence_only(
        cls,
        *,
        history_feature_dim: int,
    ) -> "MemoryConfig":
        return cls(
            urban_memory=UrbanMemoryConfig(
                sequence_encoder=(
                    TemporalSequenceEncoderConfig.identity(
                        input_dim=history_feature_dim
                    )
                ),
                assembly_policy=(
                    UrbanMemoryAssemblyPolicy.SEQUENCE_ONLY
                ),
            )
        )

    @classmethod
    def recurrent_with_pooling(
        cls,
        *,
        history_feature_dim: int,
        hidden_dim: int = 64,
        cell_kind: RecurrentCellKind | str = RecurrentCellKind.GRU,
        pooling_kind: TemporalPoolingKind | str = (
            TemporalPoolingKind.MASKED_MEAN
        ),
    ) -> "MemoryConfig":
        recurrent = RecurrentSequenceEncoderConfig(
            cell_kind=cell_kind,
            input_dim=history_feature_dim,
            hidden_dim=hidden_dim,
        )
        encoder = (
            TemporalSequenceEncoderConfig.recurrent_encoder(
                recurrent
            )
        )
        pooling = TemporalPoolingConfig(
            kind=pooling_kind,
            output_dim=encoder.output_dim,
        )
        return cls(
            urban_memory=UrbanMemoryConfig(
                sequence_encoder=encoder,
                assembly_policy=(
                    UrbanMemoryAssemblyPolicy.SEQUENCE_WITH_GENERIC_POOLING
                ),
                temporal_pooling=pooling,
            )
        )

    @classmethod
    def hazard_queried_recurrent(
        cls,
        *,
        history_feature_dim: int,
        hazard_query_dim: int,
        hidden_dim: int = 64,
        retrieval_heads: int = 4,
        output_dim: int = 64,
    ) -> "MemoryConfig":
        recurrent = RecurrentSequenceEncoderConfig(
            cell_kind=RecurrentCellKind.GRU,
            input_dim=history_feature_dim,
            hidden_dim=hidden_dim,
        )
        encoder = (
            TemporalSequenceEncoderConfig.recurrent_encoder(
                recurrent
            )
        )
        pooling = TemporalPoolingConfig(
            kind=TemporalPoolingKind.MASKED_MEAN,
            output_dim=encoder.output_dim,
        )
        retrieval = TemporalCrossAttentionConfig(
            query_dim=hazard_query_dim,
            output_dim=output_dim,
            num_heads=retrieval_heads,
            head_reduction=(
                TemporalQueryRetrievalHeadReduction.CONCAT_PROJECTION
            ),
        )
        fusion = HazardMemoryFusionConfig(
            policy=HazardMemoryFusionPolicy.CONCAT_PROJECTION,
            output_dim=output_dim,
            include_retrieved_context=True,
            include_generic_pooled_memory=True,
            include_hazard_query=True,
        )
        return cls(
            urban_memory=UrbanMemoryConfig(
                sequence_encoder=encoder,
                assembly_policy=(
                    UrbanMemoryAssemblyPolicy.SEQUENCE_WITH_GENERIC_POOLING
                ),
                temporal_pooling=pooling,
            ),
            temporal_retrieval=retrieval,
            hazard_fusion=fusion,
        )


# =============================================================================
# Compact aliases
# =============================================================================


BaselineEncoderConfig = BaselineSequenceEncoderConfig
RecurrentEncoderConfig = RecurrentSequenceEncoderConfig
TransformerEncoderConfig = TransformerSequenceEncoderConfig
UrbanMemoryEncoderConfig = UrbanMemoryConfig
CrossAttentionRetrievalConfig = TemporalCrossAttentionConfig
MemoryFusionConfig = HazardMemoryFusionConfig


# =============================================================================
# Public API
# =============================================================================


__all__ = (
    # Identity.
    "MEMORY_CONFIG_SCHEMA_VERSION",
    "MEMORY_CONFIG_SCIENTIFIC_INTERPRETATION",

    # Runtime vocabularies.
    "BaselineSequenceEncoderKind",
    "RecurrentCellKind",
    "TransformerPositionalEncodingKind",
    "MemoryActivation",
    "TemporalAttentionScoreKind",
    "FusionGateActivation",
    "CANONICAL_BASELINE_SEQUENCE_ENCODER_KINDS",
    "CANONICAL_RECURRENT_CELL_KINDS",
    "CANONICAL_TRANSFORMER_POSITIONAL_ENCODING_KINDS",
    "CANONICAL_MEMORY_ACTIVATIONS",
    "CANONICAL_TEMPORAL_ATTENTION_SCORE_KINDS",
    "CANONICAL_FUSION_GATE_ACTIVATIONS",

    # Configuration contracts.
    "MemoryConfigMixin",
    "HistoryInputConfig",
    "BaselineSequenceEncoderConfig",
    "RecurrentSequenceEncoderConfig",
    "TransformerSequenceEncoderConfig",
    "TemporalSequenceEncoderConfig",
    "TemporalPoolingConfig",
    "UrbanMemoryConfig",
    "TemporalCrossAttentionConfig",
    "HazardMemoryFusionConfig",
    "MemoryConfig",

    # Compact aliases.
    "BaselineEncoderConfig",
    "RecurrentEncoderConfig",
    "TransformerEncoderConfig",
    "UrbanMemoryEncoderConfig",
    "CrossAttentionRetrievalConfig",
    "MemoryFusionConfig",
)
