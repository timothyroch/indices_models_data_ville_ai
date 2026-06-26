"""
Shared input adaptation for Phase 6 GRU and LSTM sequence encoders.

Target path:
    urban_resilience_models/
        v2_hazard_conditioned_functional_ugnn/
            memory/
                recurrent_memory_encoder/
                    input_adapter.py

This module gives GRU and LSTM encoders one identical model-facing input path:

    canonical history
        -> optional Linear(input_dim, input_projection_dim)
        -> optional LayerNorm(effective_input_dim)
        -> validate the complete pre-mask result is finite
        -> restore exact-zero canonical padding with torch.where

The adapter is pointwise over ``(node, timestep)``. It mixes feature channels
but never communicates across the temporal axis.

Frozen Phase 6 policies
-----------------------
Projection
    ``input_projection_dim`` defines the recurrent kernel's effective
    ``input_size`` when configured.

Bias
    ``RecurrentSequenceEncoderConfig.use_bias`` controls both the optional
    projection bias and GRU/LSTM biases.

Layer normalization
    LayerNorm is applied after the optional projection and uses
    ``elementwise_affine=True``.

Padding
    Projection bias and LayerNorm affine parameters may turn padded zero inputs
    into nonzero values. The full adapted tensor is first checked for
    finiteness, then padded positions are replaced by exact zeros with
    ``torch.where``. Multiplication is forbidden because ``0 * NaN`` is NaN.

Missingness and coordinates
    The adapter consumes only the model-ready history values and
    ``timestep_mask``. It does not concatenate ``feature_observed_mask``,
    temporal-coordinate values, hazard embeddings, or missingness indicators.

All-zero histories
    The complete recurrent encoder must validate source/module compatibility
    and any explicit parameter snapshot before taking an all-zero-history early
    return. The adapter itself should then be skipped. This module exposes a
    shared compatibility validator for that preflight check.

Autograd
    Adapted values preserve autograd. Integer canonical-batch metadata is
    reconstructed through the private validated schema.
"""

from __future__ import annotations

from typing import Final

import torch
from torch import nn

from ..config import (
    RecurrentSequenceEncoderConfig,
)
from ..schemas.history_inputs import (
    HistoricalSequenceInputs,
)
from .schemas import (
    _CanonicalRecurrentBatch,
)


# =============================================================================
# Component identity and frozen policies
# =============================================================================


RECURRENT_INPUT_ADAPTER_IMPLEMENTATION_VERSION: Final[str] = "0.1"

RECURRENT_INPUT_ADAPTER_COMPONENT_NAME: Final[str] = (
    "recurrent_input_adapter"
)

RECURRENT_INPUT_ADAPTER_COMPONENT_KIND: Final[str] = (
    "pointwise_recurrent_input_adapter"
)

RECURRENT_INPUT_ADAPTER_OPERATION_NAME: Final[str] = (
    "adapt_recurrent_input_features"
)

RECURRENT_INPUT_ADAPTER_OUTPUT_STAGE: Final[str] = (
    "recurrent_input_adapter_output"
)

RECURRENT_INPUT_ADAPTER_PROJECTION_POLICY: Final[str] = (
    "optional_shared_linear_projection_per_timestep"
)

RECURRENT_INPUT_ADAPTER_BIAS_POLICY: Final[str] = (
    "use_bias_controls_projection_and_recurrent_bias"
)

RECURRENT_INPUT_ADAPTER_NORMALIZATION_POLICY: Final[str] = (
    "optional_layer_norm_after_projection_elementwise_affine_true"
)

RECURRENT_INPUT_ADAPTER_PADDING_POLICY: Final[str] = (
    "validate_finite_pre_mask_then_exact_zero_with_torch_where"
)

RECURRENT_INPUT_ADAPTER_FEATURE_OBSERVATION_POLICY: Final[str] = (
    "feature_observed_mask_not_consumed"
)

RECURRENT_INPUT_ADAPTER_TEMPORAL_COORDINATE_POLICY: Final[str] = (
    "temporal_coordinates_not_consumed"
)

RECURRENT_INPUT_ADAPTER_HAZARD_POLICY: Final[str] = (
    "hazard_information_not_consumed"
)

RECURRENT_INPUT_ADAPTER_TEMPORAL_INTERACTION: Final[bool] = False

RECURRENT_INPUT_ADAPTER_LAYER_NORM_ELEMENTWISE_AFFINE: Final[bool] = True

RECURRENT_INPUT_ADAPTER_SCIENTIFIC_INTERPRETATION: Final[str] = (
    "shared_feature_adaptation_not_temporal_reasoning_or_causal_effect"
)


# =============================================================================
# Generic validation
# =============================================================================


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


def _validate_recurrent_config(
    config: RecurrentSequenceEncoderConfig,
) -> None:
    if not isinstance(
        config,
        RecurrentSequenceEncoderConfig,
    ):
        raise TypeError(
            "config must be a RecurrentSequenceEncoderConfig."
        )


def _validate_input_values(
    values: torch.Tensor,
    *,
    expected_input_dim: int,
) -> None:
    if not isinstance(
        values,
        torch.Tensor,
    ):
        raise TypeError(
            "values must be a tensor."
        )

    if values.ndim != 3:
        raise ValueError(
            "values must have shape [M, T, D]."
        )

    if not values.dtype.is_floating_point:
        raise ValueError(
            "values must use a floating-point dtype."
        )

    if values.is_meta:
        raise ValueError(
            "values cannot reside on the meta device."
        )

    if values.layout != torch.strided:
        raise ValueError(
            "values must use strided tensor layout."
        )

    _require_positive_int(
        "expected_input_dim",
        expected_input_dim,
    )

    if int(
        values.shape[-1]
    ) != expected_input_dim:
        raise ValueError(
            "values feature width does not match the recurrent input "
            f"configuration: expected {expected_input_dim}, observed "
            f"{int(values.shape[-1])}."
        )

    if int(
        values.shape[1]
    ) <= 0:
        raise ValueError(
            "values must have a strictly positive temporal dimension."
        )

    if not bool(
        torch.isfinite(
            values
        ).all().item()
    ):
        nonfinite_count = int(
            (
                ~torch.isfinite(
                    values
                )
            )
            .sum()
            .item()
        )
        raise ValueError(
            "Recurrent input values must be finite before adaptation. "
            f"Nonfinite count: {nonfinite_count}."
        )


def _validate_timestep_mask(
    timestep_mask: torch.Tensor,
    *,
    values: torch.Tensor,
) -> None:
    if not isinstance(
        timestep_mask,
        torch.Tensor,
    ):
        raise TypeError(
            "timestep_mask must be a tensor."
        )

    if timestep_mask.ndim != 2:
        raise ValueError(
            "timestep_mask must have shape [M, T]."
        )

    if timestep_mask.dtype != torch.bool:
        raise ValueError(
            "timestep_mask must use torch.bool."
        )

    if timestep_mask.is_meta:
        raise ValueError(
            "timestep_mask cannot reside on the meta device."
        )

    if timestep_mask.layout != torch.strided:
        raise ValueError(
            "timestep_mask must use strided tensor layout."
        )

    expected_shape = (
        int(
            values.shape[0]
        ),
        int(
            values.shape[1]
        ),
    )

    if tuple(
        timestep_mask.shape
    ) != expected_shape:
        raise ValueError(
            "timestep_mask shape must align with the node and temporal "
            f"axes of values: expected {expected_shape}, observed "
            f"{tuple(timestep_mask.shape)}."
        )

    if timestep_mask.device != values.device:
        raise ValueError(
            "timestep_mask and values must share one device."
        )


def _require_finite_adapter_output(
    value: torch.Tensor,
) -> None:
    if not bool(
        torch.isfinite(
            value
        ).all().item()
    ):
        nonfinite_count = int(
            (
                ~torch.isfinite(
                    value
                )
            )
            .sum()
            .item()
        )
        raise ValueError(
            "RecurrentInputAdapter produced nonfinite pre-mask values. "
            f"Nonfinite count: {nonfinite_count}."
        )


def exact_zero_recurrent_padding(
    value: torch.Tensor,
    timestep_mask: torch.Tensor,
) -> torch.Tensor:
    """
    Replace every masked temporal position by exact zeros.

    ``torch.where`` is required so NaN or infinity cannot survive through
    multiplication by zero.
    """

    if not isinstance(
        value,
        torch.Tensor,
    ):
        raise TypeError(
            "value must be a tensor."
        )

    if value.ndim != 3:
        raise ValueError(
            "value must have shape [M, T, F]."
        )

    _validate_timestep_mask(
        timestep_mask,
        values=value,
    )

    return torch.where(
        timestep_mask.unsqueeze(
            -1
        ),
        value,
        torch.zeros_like(
            value
        ),
    )


# =============================================================================
# Module/source compatibility preflight
# =============================================================================


def _iter_module_parameters_and_buffers(
    module: nn.Module,
):
    for name, parameter in module.named_parameters():
        yield (
            f"parameter:{name}",
            parameter,
            True,
        )

    for name, buffer in module.named_buffers():
        yield (
            f"buffer:{name}",
            buffer,
            False,
        )


def validate_recurrent_module_tensor_compatibility(
    module: nn.Module,
    values: torch.Tensor,
    *,
    expected_input_dim: int,
    require_floating_parameter: bool = False,
) -> None:
    """
    Validate module device/dtype compatibility without executing the module.

    This helper is suitable for the all-zero-history preflight path. It checks
    every parameter and buffer owned by the complete recurrent encoder,
    including the input adapter and GRU/LSTM kernel.
    """

    if not isinstance(
        module,
        nn.Module,
    ):
        raise TypeError(
            "module must be a torch.nn.Module."
        )

    _require_boolean(
        "require_floating_parameter",
        require_floating_parameter,
    )
    _validate_input_values(
        values,
        expected_input_dim=(
            expected_input_dim
        ),
    )

    floating_parameter_count = 0

    for name, tensor, is_parameter in (
        _iter_module_parameters_and_buffers(
            module
        )
    ):
        if tensor.is_meta:
            raise ValueError(
                f"{name} cannot reside on the meta device."
            )

        if tensor.device != values.device:
            raise ValueError(
                f"{name} device {tensor.device} does not match recurrent "
                f"input device {values.device}."
            )

        if tensor.dtype.is_floating_point:
            if tensor.dtype != values.dtype:
                raise ValueError(
                    f"{name} dtype {tensor.dtype} does not match recurrent "
                    f"input dtype {values.dtype}."
                )

            if is_parameter:
                floating_parameter_count += 1

    if (
        require_floating_parameter
        and floating_parameter_count == 0
    ):
        raise RuntimeError(
            "The recurrent module must own at least one floating-point "
            "parameter."
        )


def validate_recurrent_module_source_compatibility(
    module: nn.Module,
    source_history: HistoricalSequenceInputs,
    *,
    expected_input_dim: int,
    require_floating_parameter: bool = True,
) -> None:
    """
    Validate source shape, runtime finiteness, and module device/dtype.

    Recurrent encoders should call this before checking whether every source
    node has zero history and before validating an explicit parameter snapshot.
    """

    if not isinstance(
        source_history,
        HistoricalSequenceInputs,
    ):
        raise TypeError(
            "source_history must be a HistoricalSequenceInputs."
        )

    if source_history.feature_dim != expected_input_dim:
        raise ValueError(
            "source_history feature width does not match the recurrent "
            f"configuration: expected {expected_input_dim}, observed "
            f"{source_history.feature_dim}."
        )

    validate_recurrent_module_tensor_compatibility(
        module,
        source_history.history,
        expected_input_dim=(
            expected_input_dim
        ),
        require_floating_parameter=(
            require_floating_parameter
        ),
    )

    padding_mask = (
        ~source_history.timestep_mask
    ).unsqueeze(
        -1
    ).expand_as(
        source_history.history
    )

    padded_values = source_history.history[
        padding_mask
    ]

    if padded_values.numel() > 0:
        if not torch.equal(
            padded_values,
            torch.zeros_like(
                padded_values
            ),
        ):
            raise ValueError(
                "source_history padding must remain exactly zero at "
                "execution time."
            )


# =============================================================================
# Recurrent input adapter
# =============================================================================


class RecurrentInputAdapter(nn.Module):
    """
    Shared optional projection and LayerNorm for GRU/LSTM inputs.

    Parameters
    ----------
    config:
        Frozen recurrent sequence-encoder configuration.

    Notes
    -----
    The adapter does not apply dropout. Recurrent dropout remains owned by the
    GRU/LSTM kernel and is active only between stacked recurrent layers.
    """

    config: RecurrentSequenceEncoderConfig
    projection: nn.Module
    normalization: nn.Module

    def __init__(
        self,
        config: RecurrentSequenceEncoderConfig,
    ) -> None:
        super().__init__()

        _validate_recurrent_config(
            config
        )

        self.config = config

        if config.input_projection_dim is None:
            self.projection = nn.Identity()
        else:
            self.projection = nn.Linear(
                in_features=config.input_dim,
                out_features=(
                    config.input_projection_dim
                ),
                bias=config.use_bias,
            )

        if config.layer_normalization:
            self.normalization = nn.LayerNorm(
                self.effective_input_dim,
                elementwise_affine=(
                    RECURRENT_INPUT_ADAPTER_LAYER_NORM_ELEMENTWISE_AFFINE
                ),
            )
        else:
            self.normalization = nn.Identity()

    # -------------------------------------------------------------------------
    # Structural properties
    # -------------------------------------------------------------------------

    @property
    def input_dim(
        self,
    ) -> int:
        return self.config.input_dim

    @property
    def effective_input_dim(
        self,
    ) -> int:
        if self.config.input_projection_dim is not None:
            return self.config.input_projection_dim

        return self.config.input_dim

    @property
    def output_dim(
        self,
    ) -> int:
        return self.effective_input_dim

    @property
    def projection_enabled(
        self,
    ) -> bool:
        return self.config.input_projection_dim is not None

    @property
    def layer_normalization_enabled(
        self,
    ) -> bool:
        return self.config.layer_normalization

    @property
    def projection_bias_enabled(
        self,
    ) -> bool:
        return (
            self.projection_enabled
            and self.config.use_bias
        )

    @property
    def has_temporal_interaction(
        self,
    ) -> bool:
        return RECURRENT_INPUT_ADAPTER_TEMPORAL_INTERACTION

    @property
    def parameter_count(
        self,
    ) -> int:
        return sum(
            int(
                parameter.numel()
            )
            for parameter in self.parameters()
        )

    @property
    def trainable_parameter_count(
        self,
    ) -> int:
        return sum(
            int(
                parameter.numel()
            )
            for parameter in self.parameters()
            if parameter.requires_grad
        )

    # -------------------------------------------------------------------------
    # Architecture description
    # -------------------------------------------------------------------------

    def architecture_metadata(
        self,
    ) -> dict[str, object]:
        return {
            "component_name": (
                RECURRENT_INPUT_ADAPTER_COMPONENT_NAME
            ),
            "component_kind": (
                RECURRENT_INPUT_ADAPTER_COMPONENT_KIND
            ),
            "implementation_version": (
                RECURRENT_INPUT_ADAPTER_IMPLEMENTATION_VERSION
            ),
            "input_dim": self.input_dim,
            "effective_input_dim": (
                self.effective_input_dim
            ),
            "projection_enabled": (
                self.projection_enabled
            ),
            "projection_dim": (
                self.config.input_projection_dim
            ),
            "projection_bias_enabled": (
                self.projection_bias_enabled
            ),
            "bias_policy": (
                RECURRENT_INPUT_ADAPTER_BIAS_POLICY
            ),
            "layer_normalization_enabled": (
                self.layer_normalization_enabled
            ),
            "layer_norm_elementwise_affine": (
                RECURRENT_INPUT_ADAPTER_LAYER_NORM_ELEMENTWISE_AFFINE
            ),
            "normalization_policy": (
                RECURRENT_INPUT_ADAPTER_NORMALIZATION_POLICY
            ),
            "padding_policy": (
                RECURRENT_INPUT_ADAPTER_PADDING_POLICY
            ),
            "feature_observation_policy": (
                RECURRENT_INPUT_ADAPTER_FEATURE_OBSERVATION_POLICY
            ),
            "temporal_coordinate_policy": (
                RECURRENT_INPUT_ADAPTER_TEMPORAL_COORDINATE_POLICY
            ),
            "hazard_policy": (
                RECURRENT_INPUT_ADAPTER_HAZARD_POLICY
            ),
            "temporal_interaction": False,
            "parameter_count": self.parameter_count,
            "trainable_parameter_count": (
                self.trainable_parameter_count
            ),
            "scientific_interpretation": (
                RECURRENT_INPUT_ADAPTER_SCIENTIFIC_INTERPRETATION
            ),
        }

    # -------------------------------------------------------------------------
    # Compatibility
    # -------------------------------------------------------------------------

    def validate_values_compatibility(
        self,
        values: torch.Tensor,
    ) -> None:
        validate_recurrent_module_tensor_compatibility(
            self,
            values,
            expected_input_dim=self.input_dim,
            require_floating_parameter=False,
        )

    # -------------------------------------------------------------------------
    # Forward and canonical-batch adaptation
    # -------------------------------------------------------------------------

    def forward(
        self,
        values: torch.Tensor,
        timestep_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Adapt ``[M, T, D]`` values to ``[M, T, F]``.

        The complete pre-mask adapted tensor is validated for finiteness before
        exact-zero padding is restored.
        """

        _validate_input_values(
            values,
            expected_input_dim=self.input_dim,
        )
        _validate_timestep_mask(
            timestep_mask,
            values=values,
        )
        self.validate_values_compatibility(
            values
        )

        projected = self.projection(
            values
        )
        adapted = self.normalization(
            projected
        )

        expected_shape = (
            int(
                values.shape[0]
            ),
            int(
                values.shape[1]
            ),
            self.effective_input_dim,
        )

        if tuple(
            adapted.shape
        ) != expected_shape:
            raise RuntimeError(
                "RecurrentInputAdapter produced an unexpected shape: "
                f"expected {expected_shape}, observed "
                f"{tuple(adapted.shape)}."
            )

        if adapted.device != values.device:
            raise RuntimeError(
                "RecurrentInputAdapter changed the tensor device."
            )

        if adapted.dtype != values.dtype:
            raise RuntimeError(
                "RecurrentInputAdapter changed the floating dtype."
            )

        _require_finite_adapter_output(
            adapted
        )

        zero_padded = exact_zero_recurrent_padding(
            adapted,
            timestep_mask,
        )

        if not bool(
            torch.isfinite(
                zero_padded
            ).all().item()
        ):
            raise RuntimeError(
                "Exact-zero padding unexpectedly produced nonfinite values."
            )

        return zero_padded

    def adapt_canonical_batch(
        self,
        batch: _CanonicalRecurrentBatch,
    ) -> _CanonicalRecurrentBatch:
        """
        Adapt one validated canonical right-padded nonempty-node batch.
        """

        if not isinstance(
            batch,
            _CanonicalRecurrentBatch,
        ):
            raise TypeError(
                "batch must be a _CanonicalRecurrentBatch."
            )

        if batch.feature_dim != self.input_dim:
            raise ValueError(
                "Canonical batch feature width does not match the input "
                f"adapter: expected {self.input_dim}, observed "
                f"{batch.feature_dim}."
            )

        adapted = self(
            batch.values,
            batch.timestep_mask,
        )

        return _CanonicalRecurrentBatch(
            values=adapted,
            timestep_mask=batch.timestep_mask,
            lengths=batch.lengths,
            nonempty_node_indices=(
                batch.nonempty_node_indices
            ),
            source_node_count=(
                batch.source_node_count
            ),
            original_padding_direction=(
                batch.original_padding_direction
            ),
            value_stage=(
                RECURRENT_INPUT_ADAPTER_OUTPUT_STAGE
            ),
        )


# =============================================================================
# Builders and compact functional wrappers
# =============================================================================


def build_recurrent_input_adapter(
    config: RecurrentSequenceEncoderConfig,
) -> RecurrentInputAdapter:
    """Construct the shared Phase 6 recurrent input adapter."""

    return RecurrentInputAdapter(
        config
    )


def adapt_recurrent_input_values(
    adapter: RecurrentInputAdapter,
    values: torch.Tensor,
    timestep_mask: torch.Tensor,
) -> torch.Tensor:
    """Functional wrapper around ``RecurrentInputAdapter.forward``."""

    if not isinstance(
        adapter,
        RecurrentInputAdapter,
    ):
        raise TypeError(
            "adapter must be a RecurrentInputAdapter."
        )

    return adapter(
        values,
        timestep_mask,
    )


def adapt_canonical_recurrent_batch(
    adapter: RecurrentInputAdapter,
    batch: _CanonicalRecurrentBatch,
) -> _CanonicalRecurrentBatch:
    """Functional wrapper around ``adapt_canonical_batch``."""

    if not isinstance(
        adapter,
        RecurrentInputAdapter,
    ):
        raise TypeError(
            "adapter must be a RecurrentInputAdapter."
        )

    return adapter.adapt_canonical_batch(
        batch
    )


# Compact internal aliases.
InputAdapter = RecurrentInputAdapter
build_input_adapter = build_recurrent_input_adapter
adapt_input_values = adapt_recurrent_input_values


# =============================================================================
# Module API
# =============================================================================


__all__ = (
    # Component identity and frozen policies.
    "RECURRENT_INPUT_ADAPTER_IMPLEMENTATION_VERSION",
    "RECURRENT_INPUT_ADAPTER_COMPONENT_NAME",
    "RECURRENT_INPUT_ADAPTER_COMPONENT_KIND",
    "RECURRENT_INPUT_ADAPTER_OPERATION_NAME",
    "RECURRENT_INPUT_ADAPTER_OUTPUT_STAGE",
    "RECURRENT_INPUT_ADAPTER_PROJECTION_POLICY",
    "RECURRENT_INPUT_ADAPTER_BIAS_POLICY",
    "RECURRENT_INPUT_ADAPTER_NORMALIZATION_POLICY",
    "RECURRENT_INPUT_ADAPTER_PADDING_POLICY",
    "RECURRENT_INPUT_ADAPTER_FEATURE_OBSERVATION_POLICY",
    "RECURRENT_INPUT_ADAPTER_TEMPORAL_COORDINATE_POLICY",
    "RECURRENT_INPUT_ADAPTER_HAZARD_POLICY",
    "RECURRENT_INPUT_ADAPTER_TEMPORAL_INTERACTION",
    "RECURRENT_INPUT_ADAPTER_LAYER_NORM_ELEMENTWISE_AFFINE",
    "RECURRENT_INPUT_ADAPTER_SCIENTIFIC_INTERPRETATION",

    # Compatibility and exact-zero utilities.
    "validate_recurrent_module_tensor_compatibility",
    "validate_recurrent_module_source_compatibility",
    "exact_zero_recurrent_padding",

    # Adapter.
    "RecurrentInputAdapter",
    "build_recurrent_input_adapter",
    "adapt_recurrent_input_values",
    "adapt_canonical_recurrent_batch",

    # Compact aliases.
    "InputAdapter",
    "build_input_adapter",
    "adapt_input_values",
)
