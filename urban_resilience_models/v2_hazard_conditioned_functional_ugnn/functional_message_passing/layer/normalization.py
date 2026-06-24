"""
Explicit post-residual normalization for one functional message-passing layer.

Target path:
    urban_resilience_models/
        v2_hazard_conditioned_functional_ugnn/
            functional_message_passing/
                layer/
                    normalization.py

This module owns exactly one node-level operation after the residual-update
stage:

    updated_node_state = normalization(post_residual_state)

Two bounded modes are implemented:

``none``
    Return the exact ``post_residual_state`` tensor object. No clone, detach,
    cast, copy, device move, parameter, or buffer is introduced.

``layer_norm``
    Apply feature-wise layer normalization independently to every node over
    the final hidden axis.

The bounded V2.0 placement is explicitly post-residual:

    node aggregate
        -> dropout
        -> optional additive residual
        -> optional layer normalization
        -> updated node state

The module does not own relation transforms, edge normalization, relation
gates, edge attention, message construction, target-node aggregation, dropout,
residual addition, multi-layer iteration, or prediction.

Shape contract
--------------
Input and output both have shape ``[N, H]``. Layer normalization operates only
over the final hidden dimension ``H`` and therefore preserves:

- node order;
- graph membership;
- node count;
- hidden width;
- dtype;
- device;
- autograd connectivity.

Empty node batches ``[0, H]`` are supported.

Affine parameters
-----------------
When layer normalization is enabled, elementwise affine scale and optional bias
parameters may be learned:

    y = ((x - mean_H(x)) / sqrt(var_H(x) + epsilon)) * weight + bias

where mean and variance are computed independently for each node over the
hidden-feature axis.

The implementation owns these parameters directly rather than wrapping an
opaque nested ``nn.LayerNorm`` module. This makes parameter provenance,
initialization, fingerprinting, and finite-value validation explicit.

Initialization is canonical:

    weight = 1
    bias   = 0

With these initial values, enabled layer normalization begins as ordinary
non-affine layer normalization.

Architecture versus parameter identity
---------------------------------------
The architecture fingerprint includes:

- normalization mode;
- post-residual placement;
- hidden width;
- epsilon;
- affine enablement;
- bias enablement.

The parameter fingerprint includes the exact current values, shapes, and dtypes
of learned scale and bias tensors. Train/eval mode does not affect either
fingerprint because layer normalization has identical train and evaluation
numerics.

Interpretation
-------------------------
Normalized values and their descriptive statistics are model-state traces.
They are not automatically causal importance, explanation faithfulness,
uncertainty, counterfactual effect, or mechanistic-identifiability evidence.
"""

from __future__ import annotations

from hashlib import sha256
import json
import math
from typing import Any, Final, Mapping

import torch
from torch import nn
from torch.nn import functional as F

from .schemas import (
    CANONICAL_LAYER_NORMALIZATION_MODES,
    CANONICAL_LAYER_NORMALIZATION_POSITIONS,
    LAYER_NORMALIZATION_LAYER_NORM,
    LAYER_NORMALIZATION_NONE,
    LAYER_NORMALIZATION_POST_RESIDUAL,
    LAYER_POST_NORMALIZATION_FORMULA,
    V2_0_IMPLEMENTED_LAYER_NORMALIZATION_MODES,
    V2_0_IMPLEMENTED_LAYER_NORMALIZATION_POSITIONS,
    LayerNormalizationOutput,
    LayerResidualUpdateOutput,
)


# =============================================================================
# Public identity
# =============================================================================


LAYER_NORMALIZER_SCHEMA_VERSION: Final[str] = "0.1"

LAYER_NORMALIZER_OPERATION: Final[str] = (
    "optional_post_residual_feature_layer_normalization"
)

LAYER_NORMALIZER_OPERATION_ORDER: Final[
    tuple[str, ...]
] = (
    "validate_exact_residual_update_lineage",
    "bind_input_state_to_exact_post_residual_state",
    "validate_normalization_mode_and_position",
    "validate_optional_affine_parameters",
    "apply_or_bypass_feature_layer_normalization",
    "validate_output_shape_dtype_device_and_finiteness",
    "construct_layer_normalization_output",
)

LAYER_NORMALIZER_NORMALIZED_AXIS: Final[int] = -1
LAYER_NORMALIZER_STATISTIC_SCOPE: Final[str] = (
    "independent_per_node_over_hidden_feature_axis"
)
LAYER_NORMALIZER_VARIANCE_ESTIMATOR: Final[str] = (
    "biased_population_variance"
)

LAYER_NORMALIZER_AGGREGATION_OWNED_HERE: Final[bool] = False
LAYER_NORMALIZER_DROPOUT_OWNED_HERE: Final[bool] = False
LAYER_NORMALIZER_RESIDUAL_OWNED_HERE: Final[bool] = False
LAYER_NORMALIZER_MULTI_LAYER_ITERATION_OWNED_HERE: Final[bool] = False

LAYER_DISABLED_NORMALIZATION_IDENTITY_POLICY: Final[str] = (
    "exact_input_tensor_identity"
)

LAYER_NORMALIZER_DEFAULT_EPSILON: Final[float] = 1e-5
LAYER_NORMALIZER_DEFAULT_ELEMENTWISE_AFFINE: Final[bool] = True
LAYER_NORMALIZER_DEFAULT_BIAS_ENABLED: Final[bool] = True


# =============================================================================
# Generic helpers
# =============================================================================


def _canonical_json(
    payload: Mapping[str, Any],
) -> str:
    return json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )


def _fingerprint(
    payload: Mapping[str, Any],
) -> str:
    return sha256(
        _canonical_json(payload)
        .encode("utf-8")
    ).hexdigest()


def _tensor_fingerprint(
    tensors: Mapping[str, torch.Tensor],
) -> str:
    """
    Fingerprint exact tensor names, shapes, dtypes, and values.

    Tensors are detached and copied to CPU only for hashing. Model tensors are
    never modified.
    """

    digest = sha256()

    for name in sorted(tensors):
        tensor = (
            tensors[name]
            .detach()
            .cpu()
            .contiguous()
        )

        digest.update(
            name.encode("utf-8")
        )
        digest.update(
            str(tensor.dtype).encode(
                "utf-8"
            )
        )
        digest.update(
            json.dumps(
                list(tensor.shape),
                separators=(",", ":"),
            ).encode("utf-8")
        )
        digest.update(
            tensor
            .view(torch.uint8)
            .numpy()
            .tobytes()
        )

    return digest.hexdigest()


def _require_nonempty_string(
    name: str,
    value: str,
) -> None:
    if not isinstance(
        value,
        str,
    ) or not value.strip():
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
            f"{name} must be Boolean."
        )


def _require_positive_int(
    name: str,
    value: int,
) -> int:
    if isinstance(
        value,
        bool,
    ) or not isinstance(
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

    return value


def _require_positive_float(
    name: str,
    value: float,
) -> float:
    if isinstance(
        value,
        bool,
    ) or not isinstance(
        value,
        (int, float),
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

    if numeric <= 0.0:
        raise ValueError(
            f"{name} must be strictly positive."
        )

    return numeric


def _require_choice(
    name: str,
    value: str,
    choices: tuple[str, ...],
) -> None:
    _require_nonempty_string(
        name,
        value,
    )

    if value not in choices:
        raise ValueError(
            f"{name} must be one of "
            f"{choices!r}; observed {value!r}."
        )


def _require_normalization_mode(
    normalization_mode: str,
) -> None:
    _require_choice(
        "normalization_mode",
        normalization_mode,
        CANONICAL_LAYER_NORMALIZATION_MODES,
    )

    if normalization_mode not in (
        V2_0_IMPLEMENTED_LAYER_NORMALIZATION_MODES
    ):
        raise NotImplementedError(
            f"Normalization mode {normalization_mode!r} is canonical "
            "but not implemented in bounded V2.0."
        )


def _require_normalization_position(
    normalization_position: str,
) -> None:
    _require_choice(
        "normalization_position",
        normalization_position,
        CANONICAL_LAYER_NORMALIZATION_POSITIONS,
    )

    if normalization_position not in (
        V2_0_IMPLEMENTED_LAYER_NORMALIZATION_POSITIONS
    ):
        raise NotImplementedError(
            f"Normalization position {normalization_position!r} is "
            "canonical but not implemented in bounded V2.0."
        )


def _require_residual_update(
    residual_update: LayerResidualUpdateOutput,
) -> None:
    if not isinstance(
        residual_update,
        LayerResidualUpdateOutput,
    ):
        raise TypeError(
            "residual_update must be a "
            "LayerResidualUpdateOutput."
        )


def _require_float_matrix(
    name: str,
    value: torch.Tensor,
) -> None:
    if not isinstance(
        value,
        torch.Tensor,
    ):
        raise TypeError(
            f"{name} must be a tensor."
        )

    if value.ndim != 2:
        raise ValueError(
            f"{name} must have rank 2 and "
            "shape [N, H]; observed "
            f"{tuple(value.shape)}."
        )

    if not value.dtype.is_floating_point:
        raise ValueError(
            f"{name} must use a floating-point dtype."
        )

    if not bool(
        torch.isfinite(value)
        .all()
        .item()
    ):
        raise FloatingPointError(
            f"{name} must contain only finite values."
        )


def _require_vector_parameter(
    name: str,
    value: torch.Tensor,
    *,
    hidden_dim: int,
    reference: torch.Tensor,
) -> None:
    if not isinstance(
        value,
        torch.Tensor,
    ):
        raise TypeError(
            f"{name} must be a tensor."
        )

    if value.ndim != 1:
        raise ValueError(
            f"{name} must have rank 1 and shape [H]."
        )

    if tuple(
        value.shape
    ) != (
        hidden_dim,
    ):
        raise ValueError(
            f"{name} must have shape ({hidden_dim},); "
            f"observed {tuple(value.shape)}."
        )

    if not value.dtype.is_floating_point:
        raise ValueError(
            f"{name} must use a floating-point dtype."
        )

    if value.dtype != (
        reference.dtype
    ):
        raise ValueError(
            f"{name} must share the input-state dtype."
        )

    if value.device != (
        reference.device
    ):
        raise ValueError(
            f"{name} must share the input-state device."
        )

    if not bool(
        torch.isfinite(value)
        .all()
        .item()
    ):
        raise FloatingPointError(
            f"{name} must contain only finite values."
        )


def _require_same_matrix_contract(
    *,
    input_state: torch.Tensor,
    output_state: torch.Tensor,
) -> None:
    _require_float_matrix(
        "input_state",
        input_state,
    )
    _require_float_matrix(
        "output_state",
        output_state,
    )

    if tuple(
        output_state.shape
    ) != tuple(
        input_state.shape
    ):
        raise ValueError(
            "Layer normalization must preserve "
            "the node-state shape."
        )

    if output_state.dtype != (
        input_state.dtype
    ):
        raise ValueError(
            "Layer normalization must preserve dtype."
        )

    if output_state.device != (
        input_state.device
    ):
        raise ValueError(
            "Layer normalization must preserve device."
        )


def _resolve_affine_contract(
    *,
    normalization_mode: str,
    weight: torch.Tensor | None,
    bias: torch.Tensor | None,
) -> tuple[bool, bool]:
    if normalization_mode == (
        LAYER_NORMALIZATION_NONE
    ):
        if weight is not None:
            raise ValueError(
                "Disabled normalization must not receive a weight tensor."
            )

        if bias is not None:
            raise ValueError(
                "Disabled normalization must not receive a bias tensor."
            )

        return False, False

    if bias is not None and weight is None:
        raise ValueError(
            "Layer-normalization bias requires an affine weight tensor."
        )

    return (
        weight is not None,
        bias is not None,
    )


def _parameter_state(
    *,
    weight: torch.Tensor | None,
    bias: torch.Tensor | None,
) -> dict[str, torch.Tensor]:
    state: dict[str, torch.Tensor] = {}

    if weight is not None:
        state["weight"] = weight

    if bias is not None:
        state["bias"] = bias

    return state


def _parameter_fingerprint_from_tensors(
    *,
    normalization_mode: str,
    weight: torch.Tensor | None,
    bias: torch.Tensor | None,
) -> str | None:
    if normalization_mode == (
        LAYER_NORMALIZATION_NONE
    ):
        return None

    return _tensor_fingerprint(
        _parameter_state(
            weight=weight,
            bias=bias,
        )
    )


def _default_tolerances(
    dtype: torch.dtype,
) -> tuple[float, float]:
    if dtype in (
        torch.float16,
        torch.bfloat16,
    ):
        return 2e-3, 2e-3

    if dtype == torch.float64:
        return 1e-10, 1e-9

    return 1e-6, 1e-5


# =============================================================================
# Mode conversion
# =============================================================================


def normalization_mode_from_enabled(
    layer_norm_enabled: bool,
) -> str:
    """
    Convert the historical Boolean configuration into an explicit mode.
    """

    _require_boolean(
        "layer_norm_enabled",
        layer_norm_enabled,
    )

    return (
        LAYER_NORMALIZATION_LAYER_NORM
        if layer_norm_enabled
        else LAYER_NORMALIZATION_NONE
    )


def normalization_enabled_from_mode(
    normalization_mode: str,
) -> bool:
    """
    Convert the explicit mode into the historical Boolean flag.
    """

    _require_normalization_mode(
        normalization_mode
    )

    return normalization_mode != (
        LAYER_NORMALIZATION_NONE
    )


# =============================================================================
# Architecture identity
# =============================================================================


def layer_normalizer_architecture_dict(
    *,
    normalization_mode: str,
    normalization_position: str,
    hidden_dim: int,
    epsilon: float,
    elementwise_affine: bool,
    bias_enabled: bool,
) -> dict[str, Any]:
    """
    Return the numerical normalization architecture.
    """

    _require_normalization_mode(
        normalization_mode
    )
    _require_normalization_position(
        normalization_position
    )
    hidden_dim = _require_positive_int(
        "hidden_dim",
        hidden_dim,
    )
    epsilon = _require_positive_float(
        "epsilon",
        epsilon,
    )
    _require_boolean(
        "elementwise_affine",
        elementwise_affine,
    )
    _require_boolean(
        "bias_enabled",
        bias_enabled,
    )

    if normalization_position != (
        LAYER_NORMALIZATION_POST_RESIDUAL
    ):
        raise ValueError(
            "The bounded layer normalizer requires "
            "post_residual placement."
        )

    if normalization_mode == (
        LAYER_NORMALIZATION_NONE
    ):
        if elementwise_affine:
            raise ValueError(
                "Disabled normalization cannot enable "
                "elementwise affine parameters."
            )

        if bias_enabled:
            raise ValueError(
                "Disabled normalization cannot enable bias."
            )

    if (
        normalization_mode
        == LAYER_NORMALIZATION_LAYER_NORM
        and bias_enabled
        and not elementwise_affine
    ):
        raise ValueError(
            "Layer-normalization bias requires "
            "elementwise_affine=True."
        )

    parameter_count = 0

    if (
        normalization_mode
        == LAYER_NORMALIZATION_LAYER_NORM
        and elementwise_affine
    ):
        parameter_count += hidden_dim

        if bias_enabled:
            parameter_count += hidden_dim

    return {
        "schema_version": (
            LAYER_NORMALIZER_SCHEMA_VERSION
        ),
        "operation": (
            LAYER_NORMALIZER_OPERATION
        ),
        "operation_order": list(
            LAYER_NORMALIZER_OPERATION_ORDER
        ),
        "normalization_mode": (
            normalization_mode
        ),
        "normalization_enabled": (
            normalization_enabled_from_mode(
                normalization_mode
            )
        ),
        "normalization_position": (
            normalization_position
        ),
        "hidden_dim": hidden_dim,
        "normalized_shape": [
            hidden_dim
        ],
        "normalized_axis": (
            LAYER_NORMALIZER_NORMALIZED_AXIS
        ),
        "statistic_scope": (
            LAYER_NORMALIZER_STATISTIC_SCOPE
        ),
        "variance_estimator": (
            LAYER_NORMALIZER_VARIANCE_ESTIMATOR
        ),
        "epsilon": epsilon,
        "elementwise_affine": (
            elementwise_affine
        ),
        "bias_enabled": (
            bias_enabled
        ),
        "parameter_count": (
            parameter_count
        ),
        "disabled_identity_policy": (
            LAYER_DISABLED_NORMALIZATION_IDENTITY_POLICY
        ),
        "formula": (
            LAYER_POST_NORMALIZATION_FORMULA
        ),
        "aggregation_owned_here": (
            LAYER_NORMALIZER_AGGREGATION_OWNED_HERE
        ),
        "dropout_owned_here": (
            LAYER_NORMALIZER_DROPOUT_OWNED_HERE
        ),
        "residual_owned_here": (
            LAYER_NORMALIZER_RESIDUAL_OWNED_HERE
        ),
        "multi_layer_iteration_owned_here": (
            LAYER_NORMALIZER_MULTI_LAYER_ITERATION_OWNED_HERE
        ),
        "claims_causal_importance": False,
        "claims_explanation_faithfulness": False,
    }


def layer_normalizer_architecture_fingerprint(
    *,
    normalization_mode: str,
    normalization_position: str,
    hidden_dim: int,
    epsilon: float,
    elementwise_affine: bool,
    bias_enabled: bool,
) -> str:
    return _fingerprint(
        layer_normalizer_architecture_dict(
            normalization_mode=(
                normalization_mode
            ),
            normalization_position=(
                normalization_position
            ),
            hidden_dim=hidden_dim,
            epsilon=epsilon,
            elementwise_affine=(
                elementwise_affine
            ),
            bias_enabled=bias_enabled,
        )
    )


# =============================================================================
# Low-level numerical operation
# =============================================================================


def apply_layer_normalization(
    input_state: torch.Tensor,
    *,
    normalization_mode: str,
    epsilon: float,
    weight: torch.Tensor | None = None,
    bias: torch.Tensor | None = None,
) -> torch.Tensor:
    """
    Apply or bypass feature-wise layer normalization.

    Disabled normalization returns the exact input tensor object.
    """

    _require_float_matrix(
        "input_state",
        input_state,
    )
    _require_normalization_mode(
        normalization_mode
    )
    epsilon = _require_positive_float(
        "epsilon",
        epsilon,
    )

    (
        elementwise_affine,
        _bias_enabled,
    ) = _resolve_affine_contract(
        normalization_mode=(
            normalization_mode
        ),
        weight=weight,
        bias=bias,
    )

    hidden_dim = int(
        input_state.shape[-1]
    )

    if hidden_dim <= 0:
        raise ValueError(
            "input_state hidden dimension must be strictly positive."
        )

    if weight is not None:
        _require_vector_parameter(
            "weight",
            weight,
            hidden_dim=hidden_dim,
            reference=input_state,
        )

    if bias is not None:
        _require_vector_parameter(
            "bias",
            bias,
            hidden_dim=hidden_dim,
            reference=input_state,
        )

    if normalization_mode == (
        LAYER_NORMALIZATION_NONE
    ):
        return input_state

    output_state = F.layer_norm(
        input_state,
        normalized_shape=(
            hidden_dim,
        ),
        weight=(
            weight
            if elementwise_affine
            else None
        ),
        bias=bias,
        eps=epsilon,
    )

    _require_same_matrix_contract(
        input_state=input_state,
        output_state=output_state,
    )

    if not bool(
        torch.isfinite(output_state)
        .all()
        .item()
    ):
        raise FloatingPointError(
            "Layer normalization produced non-finite values."
        )

    return output_state


# =============================================================================
# Complete functional normalization stage
# =============================================================================


def build_layer_normalization_output(
    *,
    residual_update: LayerResidualUpdateOutput,
    normalization_mode: str,
    normalization_position: str = (
        LAYER_NORMALIZATION_POST_RESIDUAL
    ),
    epsilon: float = (
        LAYER_NORMALIZER_DEFAULT_EPSILON
    ),
    weight: torch.Tensor | None = None,
    bias: torch.Tensor | None = None,
    normalizer_architecture_fingerprint: (
        str | None
    ) = None,
    normalizer_parameter_fingerprint: (
        str | None
    ) = None,
) -> LayerNormalizationOutput:
    """
    Execute the complete normalization stage and construct its immutable output.
    """

    _require_residual_update(
        residual_update
    )
    _require_normalization_mode(
        normalization_mode
    )
    _require_normalization_position(
        normalization_position
    )
    epsilon = _require_positive_float(
        "epsilon",
        epsilon,
    )

    if normalization_position != (
        LAYER_NORMALIZATION_POST_RESIDUAL
    ):
        raise ValueError(
            "The current layer schema requires post_residual "
            "normalization."
        )

    input_state = (
        residual_update
        .post_residual_state
    )
    hidden_dim = int(
        input_state.shape[-1]
    )

    (
        elementwise_affine,
        bias_enabled,
    ) = _resolve_affine_contract(
        normalization_mode=(
            normalization_mode
        ),
        weight=weight,
        bias=bias,
    )

    output_state = apply_layer_normalization(
        input_state,
        normalization_mode=(
            normalization_mode
        ),
        epsilon=epsilon,
        weight=weight,
        bias=bias,
    )

    expected_architecture_fingerprint = (
        layer_normalizer_architecture_fingerprint(
            normalization_mode=(
                normalization_mode
            ),
            normalization_position=(
                normalization_position
            ),
            hidden_dim=hidden_dim,
            epsilon=epsilon,
            elementwise_affine=(
                elementwise_affine
            ),
            bias_enabled=(
                bias_enabled
            ),
        )
    )

    if normalizer_architecture_fingerprint is None:
        normalizer_architecture_fingerprint = (
            expected_architecture_fingerprint
        )

    _require_nonempty_string(
        "normalizer_architecture_fingerprint",
        normalizer_architecture_fingerprint,
    )

    if (
        normalizer_architecture_fingerprint
        != expected_architecture_fingerprint
    ):
        raise ValueError(
            "normalizer_architecture_fingerprint does not match the "
            "declared normalization architecture."
        )

    expected_parameter_fingerprint = (
        _parameter_fingerprint_from_tensors(
            normalization_mode=(
                normalization_mode
            ),
            weight=weight,
            bias=bias,
        )
    )

    if normalizer_parameter_fingerprint is None:
        normalizer_parameter_fingerprint = (
            expected_parameter_fingerprint
        )

    if normalization_mode == (
        LAYER_NORMALIZATION_NONE
    ):
        if normalizer_parameter_fingerprint is not None:
            raise ValueError(
                "Disabled normalization must not expose a parameter "
                "fingerprint."
            )
    else:
        _require_nonempty_string(
            "normalizer_parameter_fingerprint",
            normalizer_parameter_fingerprint,
        )

        if (
            normalizer_parameter_fingerprint
            != expected_parameter_fingerprint
        ):
            raise ValueError(
                "normalizer_parameter_fingerprint does not match the "
                "supplied normalization parameters."
            )

    output = LayerNormalizationOutput(
        input_state=input_state,
        output_state=output_state,
        residual_update=(
            residual_update
        ),
        normalization_mode=(
            normalization_mode
        ),
        normalization_position=(
            normalization_position
        ),
        epsilon=epsilon,
        normalizer_architecture_fingerprint=(
            normalizer_architecture_fingerprint
        ),
        normalizer_parameter_fingerprint=(
            normalizer_parameter_fingerprint
        ),
    )

    validate_layer_normalization_output(
        output=output,
        residual_update=(
            residual_update
        ),
        normalization_mode=(
            normalization_mode
        ),
        normalization_position=(
            normalization_position
        ),
        epsilon=epsilon,
        weight=weight,
        bias=bias,
        normalizer_architecture_fingerprint=(
            normalizer_architecture_fingerprint
        ),
        normalizer_parameter_fingerprint=(
            normalizer_parameter_fingerprint
        ),
    )

    return output


# =============================================================================
# Output validation
# =============================================================================


def validate_layer_normalization_output(
    *,
    output: LayerNormalizationOutput,
    residual_update: (
        LayerResidualUpdateOutput
        | None
    ) = None,
    normalization_mode: str | None = None,
    normalization_position: (
        str | None
    ) = None,
    epsilon: float | None = None,
    weight: torch.Tensor | None = None,
    bias: torch.Tensor | None = None,
    normalizer_architecture_fingerprint: (
        str | None
    ) = None,
    normalizer_parameter_fingerprint: (
        str | None
    ) = None,
) -> None:
    """
    Validate one normalization output and optional exact expectations.

    When weight and bias are supplied, the numerical layer-normalization
    equation is recomputed and checked.
    """

    if not isinstance(
        output,
        LayerNormalizationOutput,
    ):
        raise TypeError(
            "output must be a "
            "LayerNormalizationOutput."
        )

    _require_residual_update(
        output.residual_update
    )
    _require_same_matrix_contract(
        input_state=output.input_state,
        output_state=output.output_state,
    )

    if output.input_state is not (
        output
        .residual_update
        .post_residual_state
    ):
        raise ValueError(
            "output.input_state must preserve the exact residual-update "
            "output tensor object."
        )

    if residual_update is not None:
        _require_residual_update(
            residual_update
        )

        if output.residual_update is not (
            residual_update
        ):
            raise ValueError(
                "output must preserve the exact expected residual_update "
                "object."
            )

    if normalization_mode is not None:
        _require_normalization_mode(
            normalization_mode
        )

        if output.normalization_mode != (
            normalization_mode
        ):
            raise ValueError(
                "output normalization_mode differs from the expected "
                "mode."
            )

    if normalization_position is not None:
        _require_normalization_position(
            normalization_position
        )

        if output.normalization_position != (
            normalization_position
        ):
            raise ValueError(
                "output normalization_position differs from the expected "
                "position."
            )

    if epsilon is not None:
        expected_epsilon = (
            _require_positive_float(
                "epsilon",
                epsilon,
            )
        )

        if output.epsilon != (
            expected_epsilon
        ):
            raise ValueError(
                "output epsilon differs from the expected value."
            )

    if output.normalization_position != (
        LAYER_NORMALIZATION_POST_RESIDUAL
    ):
        raise ValueError(
            "Bounded V2.0 normalization output must use post_residual "
            "placement."
        )

    hidden_dim = int(
        output.input_state.shape[-1]
    )

    # ``LayerNormalizationOutput`` intentionally stores fingerprints rather
    # than parameter tensors. A generic consumer such as diagnostics can
    # therefore validate lineage and architecture provenance without owning
    # the originating ``LayerNormalizer``. Exact parameter-value and numerical
    # equation validation is performed only when an explicit parameter
    # contract is supplied by the caller.
    explicit_parameter_contract = bool(
        weight is not None
        or bias is not None
        or normalizer_parameter_fingerprint
        is not None
    )

    if weight is not None:
        _require_vector_parameter(
            "weight",
            weight,
            hidden_dim=hidden_dim,
            reference=output.input_state,
        )

    if bias is not None:
        _require_vector_parameter(
            "bias",
            bias,
            hidden_dim=hidden_dim,
            reference=output.input_state,
        )

    if output.normalization_mode == (
        LAYER_NORMALIZATION_NONE
    ):
        valid_architecture_fingerprints = {
            layer_normalizer_architecture_fingerprint(
                normalization_mode=(
                    output.normalization_mode
                ),
                normalization_position=(
                    output.normalization_position
                ),
                hidden_dim=hidden_dim,
                epsilon=output.epsilon,
                elementwise_affine=False,
                bias_enabled=False,
            )
        }
    elif explicit_parameter_contract:
        (
            elementwise_affine,
            bias_enabled,
        ) = _resolve_affine_contract(
            normalization_mode=(
                output.normalization_mode
            ),
            weight=weight,
            bias=bias,
        )
        valid_architecture_fingerprints = {
            layer_normalizer_architecture_fingerprint(
                normalization_mode=(
                    output.normalization_mode
                ),
                normalization_position=(
                    output.normalization_position
                ),
                hidden_dim=hidden_dim,
                epsilon=output.epsilon,
                elementwise_affine=(
                    elementwise_affine
                ),
                bias_enabled=(
                    bias_enabled
                ),
            )
        }
    else:
        # Without the originating parameters, the output schema cannot
        # distinguish non-affine LayerNorm from affine LayerNorm with or
        # without bias. Validate that the declared fingerprint matches one of
        # the bounded V2.0 contracts.
        valid_architecture_fingerprints = {
            layer_normalizer_architecture_fingerprint(
                normalization_mode=(
                    output.normalization_mode
                ),
                normalization_position=(
                    output.normalization_position
                ),
                hidden_dim=hidden_dim,
                epsilon=output.epsilon,
                elementwise_affine=(
                    elementwise_affine
                ),
                bias_enabled=(
                    bias_enabled
                ),
            )
            for (
                elementwise_affine,
                bias_enabled,
            ) in (
                (False, False),
                (True, False),
                (True, True),
            )
        }

    if (
        output
        .normalizer_architecture_fingerprint
        not in valid_architecture_fingerprints
    ):
        raise ValueError(
            "output normalizer architecture fingerprint does not match "
            "any valid declared normalization contract."
        )

    if (
        normalizer_architecture_fingerprint
        is not None
    ):
        _require_nonempty_string(
            "normalizer_architecture_fingerprint",
            normalizer_architecture_fingerprint,
        )

        if (
            output
            .normalizer_architecture_fingerprint
            != normalizer_architecture_fingerprint
        ):
            raise ValueError(
                "output normalizer architecture fingerprint differs from "
                "the expected fingerprint."
            )

    expected_parameter_fingerprint = (
        _parameter_fingerprint_from_tensors(
            normalization_mode=(
                output.normalization_mode
            ),
            weight=weight,
            bias=bias,
        )
        if explicit_parameter_contract
        else None
    )

    if output.normalization_mode == (
        LAYER_NORMALIZATION_NONE
    ):
        if output.output_state is not (
            output.input_state
        ):
            raise ValueError(
                "Disabled normalization must preserve exact input tensor "
                "identity."
            )

        if (
            output
            .normalizer_parameter_fingerprint
            is not None
        ):
            raise ValueError(
                "Disabled normalization must not expose a parameter "
                "fingerprint."
            )
    else:
        _require_nonempty_string(
            "output.normalizer_parameter_fingerprint",
            output.normalizer_parameter_fingerprint,
        )

        if explicit_parameter_contract:
            if (
                output
                .normalizer_parameter_fingerprint
                != expected_parameter_fingerprint
            ):
                raise ValueError(
                    "output normalizer parameter fingerprint does not match "
                    "the supplied parameters."
                )

            expected_output = (
                apply_layer_normalization(
                    output.input_state,
                    normalization_mode=(
                        output.normalization_mode
                    ),
                    epsilon=output.epsilon,
                    weight=weight,
                    bias=bias,
                )
            )
            atol, rtol = (
                _default_tolerances(
                    output.output_state.dtype
                )
            )

            if not torch.allclose(
                output.output_state,
                expected_output,
                atol=atol,
                rtol=rtol,
            ):
                raise ValueError(
                    "output_state differs from the declared "
                    "layer-normalization equation."
                )

    if (
        normalizer_parameter_fingerprint
        is not None
    ):
        _require_nonempty_string(
            "normalizer_parameter_fingerprint",
            normalizer_parameter_fingerprint,
        )

        if (
            output
            .normalizer_parameter_fingerprint
            != normalizer_parameter_fingerprint
        ):
            raise ValueError(
                "output normalizer parameter fingerprint differs from "
                "the expected fingerprint."
            )


# =============================================================================
# Descriptive diagnostics
# =============================================================================


def _matrix_statistics(
    value: torch.Tensor,
) -> dict[str, Any]:
    _require_float_matrix(
        "value",
        value,
    )

    detached = value.detach()
    element_count = int(
        detached.numel()
    )
    node_count = int(
        detached.shape[0]
    )
    hidden_dim = int(
        detached.shape[1]
    )

    if element_count == 0:
        return {
            "element_count": 0,
            "node_count": node_count,
            "hidden_dim": hidden_dim,
            "minimum": None,
            "maximum": None,
            "mean": None,
            "standard_deviation": None,
            "mean_absolute_value": None,
            "l2_norm": 0.0,
            "zero_count": 0,
            "finite": True,
        }

    return {
        "element_count": (
            element_count
        ),
        "node_count": node_count,
        "hidden_dim": hidden_dim,
        "minimum": float(
            detached.min().item()
        ),
        "maximum": float(
            detached.max().item()
        ),
        "mean": float(
            detached.mean().item()
        ),
        "standard_deviation": float(
            detached.std(
                unbiased=False
            ).item()
        ),
        "mean_absolute_value": float(
            detached.abs().mean().item()
        ),
        "l2_norm": float(
            torch.linalg.vector_norm(
                detached
            ).item()
        ),
        "zero_count": int(
            (detached == 0)
            .sum()
            .item()
        ),
        "finite": True,
    }


def _per_node_feature_statistics(
    value: torch.Tensor,
) -> dict[str, Any]:
    _require_float_matrix(
        "value",
        value,
    )

    node_count = int(
        value.shape[0]
    )

    if node_count == 0:
        return {
            "node_count": 0,
            "mean_of_node_means": None,
            "maximum_absolute_node_mean": None,
            "mean_of_node_variances": None,
            "minimum_node_variance": None,
            "maximum_node_variance": None,
        }

    detached = value.detach()
    node_means = detached.mean(
        dim=-1
    )
    node_variances = detached.var(
        dim=-1,
        unbiased=False,
    )

    return {
        "node_count": node_count,
        "mean_of_node_means": float(
            node_means.mean().item()
        ),
        "maximum_absolute_node_mean": float(
            node_means.abs().max().item()
        ),
        "mean_of_node_variances": float(
            node_variances.mean().item()
        ),
        "minimum_node_variance": float(
            node_variances.min().item()
        ),
        "maximum_node_variance": float(
            node_variances.max().item()
        ),
    }


def layer_normalization_diagnostic_summary(
    output: LayerNormalizationOutput,
) -> dict[str, Any]:
    """
    Return detached descriptive statistics for one normalization output.

    Numerical summaries do not imply causality or explanation faithfulness.
    """

    if not isinstance(
        output,
        LayerNormalizationOutput,
    ):
        raise TypeError(
            "output must be a "
            "LayerNormalizationOutput."
        )

    _require_same_matrix_contract(
        input_state=output.input_state,
        output_state=output.output_state,
    )

    return {
        "schema_version": (
            LAYER_NORMALIZER_SCHEMA_VERSION
        ),
        "operation": (
            LAYER_NORMALIZER_OPERATION
        ),
        "layer_index": (
            output
            .layer_inputs
            .layer_index
        ),
        "normalization_mode": (
            output.normalization_mode
        ),
        "normalization_enabled": (
            output.normalization_enabled
        ),
        "normalization_position": (
            output.normalization_position
        ),
        "epsilon": float(
            output.epsilon
        ),
        "num_nodes": output.num_nodes,
        "hidden_dim": output.hidden_dim,
        "dtype": str(output.dtype),
        "device": str(output.device),
        "input_state": (
            _matrix_statistics(
                output.input_state
            )
        ),
        "output_state": (
            _matrix_statistics(
                output.output_state
            )
        ),
        "input_per_node_features": (
            _per_node_feature_statistics(
                output.input_state
            )
        ),
        "output_per_node_features": (
            _per_node_feature_statistics(
                output.output_state
            )
        ),
        "exact_identity": {
            "input_is_exact_post_residual_state": (
                output.input_state
                is output
                .residual_update
                .post_residual_state
            ),
            "output_is_input": (
                output.output_state
                is output.input_state
            ),
        },
        "architecture_fingerprint": (
            output
            .normalizer_architecture_fingerprint
        ),
        "parameter_fingerprint": (
            output
            .normalizer_parameter_fingerprint
        ),
        "aggregation_performed_here": False,
        "dropout_performed_here": False,
        "residual_performed_here": False,
        "causal_importance_claim": False,
        "explanation_faithfulness_claim": False,
    }


# =============================================================================
# Trainable module wrapper
# =============================================================================


class LayerNormalizer(nn.Module):
    """
    Optional post-residual feature-wise layer normalization.

    Parameters
    ----------
    hidden_dim:
        Stable node-state width ``H``.
    normalization_mode:
        ``"none"`` or ``"layer_norm"``.
    normalization_position:
        Must be ``"post_residual"`` in bounded V2.0.
    epsilon:
        Strictly positive numerical stabilizer.
    elementwise_affine:
        Learn one scale per hidden feature when normalization is enabled.
    bias_enabled:
        Learn one bias per hidden feature. Requires affine scale.
    device, dtype:
        Optional parameter-construction placement.
    """

    hidden_dim: int
    normalization_mode: str
    normalization_position: str
    epsilon: float
    elementwise_affine: bool
    bias_enabled: bool

    weight: nn.Parameter | None
    bias: nn.Parameter | None

    def __init__(
        self,
        hidden_dim: int,
        *,
        normalization_mode: str = (
            LAYER_NORMALIZATION_LAYER_NORM
        ),
        normalization_position: str = (
            LAYER_NORMALIZATION_POST_RESIDUAL
        ),
        epsilon: float = (
            LAYER_NORMALIZER_DEFAULT_EPSILON
        ),
        elementwise_affine: bool = (
            LAYER_NORMALIZER_DEFAULT_ELEMENTWISE_AFFINE
        ),
        bias_enabled: bool = (
            LAYER_NORMALIZER_DEFAULT_BIAS_ENABLED
        ),
        device: (
            torch.device | str | None
        ) = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()

        hidden_dim = _require_positive_int(
            "hidden_dim",
            hidden_dim,
        )
        _require_normalization_mode(
            normalization_mode
        )
        _require_normalization_position(
            normalization_position
        )
        epsilon = _require_positive_float(
            "epsilon",
            epsilon,
        )
        _require_boolean(
            "elementwise_affine",
            elementwise_affine,
        )
        _require_boolean(
            "bias_enabled",
            bias_enabled,
        )

        # This validates cross-field architecture invariants before parameters
        # are allocated.
        layer_normalizer_architecture_dict(
            normalization_mode=(
                normalization_mode
            ),
            normalization_position=(
                normalization_position
            ),
            hidden_dim=hidden_dim,
            epsilon=epsilon,
            elementwise_affine=(
                elementwise_affine
            ),
            bias_enabled=(
                bias_enabled
            ),
        )

        self.hidden_dim = hidden_dim
        self.normalization_mode = (
            normalization_mode
        )
        self.normalization_position = (
            normalization_position
        )
        self.epsilon = epsilon
        self.elementwise_affine = (
            elementwise_affine
        )
        self.bias_enabled = (
            bias_enabled
        )

        factory_kwargs = {
            "device": device,
            "dtype": dtype,
        }

        if (
            self.normalization_enabled
            and self.elementwise_affine
        ):
            self.weight = nn.Parameter(
                torch.empty(
                    self.hidden_dim,
                    **factory_kwargs,
                )
            )

            if self.bias_enabled:
                self.bias = nn.Parameter(
                    torch.empty(
                        self.hidden_dim,
                        **factory_kwargs,
                    )
                )
            else:
                self.register_parameter(
                    "bias",
                    None,
                )
        else:
            self.register_parameter(
                "weight",
                None,
            )
            self.register_parameter(
                "bias",
                None,
            )

        self.reset_parameters()
        self.assert_finite_parameters()

    @classmethod
    def from_flag(
        cls,
        hidden_dim: int,
        *,
        layer_norm_enabled: bool,
        epsilon: float = (
            LAYER_NORMALIZER_DEFAULT_EPSILON
        ),
        elementwise_affine: bool = (
            LAYER_NORMALIZER_DEFAULT_ELEMENTWISE_AFFINE
        ),
        bias_enabled: bool = (
            LAYER_NORMALIZER_DEFAULT_BIAS_ENABLED
        ),
        device: (
            torch.device | str | None
        ) = None,
        dtype: torch.dtype | None = None,
    ) -> "LayerNormalizer":
        mode = normalization_mode_from_enabled(
            layer_norm_enabled
        )

        if not layer_norm_enabled:
            elementwise_affine = False
            bias_enabled = False

        return cls(
            hidden_dim,
            normalization_mode=mode,
            normalization_position=(
                LAYER_NORMALIZATION_POST_RESIDUAL
            ),
            epsilon=epsilon,
            elementwise_affine=(
                elementwise_affine
            ),
            bias_enabled=(
                bias_enabled
            ),
            device=device,
            dtype=dtype,
        )

    @property
    def normalization_enabled(
        self,
    ) -> bool:
        return normalization_enabled_from_mode(
            self.normalization_mode
        )

    @property
    def parameter_count(self) -> int:
        return sum(
            int(parameter.numel())
            for parameter in self.parameters()
        )

    @property
    def trainable_parameter_count(
        self,
    ) -> int:
        return sum(
            int(parameter.numel())
            for parameter in self.parameters()
            if parameter.requires_grad
        )

    @property
    def buffer_count(self) -> int:
        return sum(
            int(buffer.numel())
            for buffer in self.buffers()
        )

    @property
    def expected_parameter_count(
        self,
    ) -> int:
        return int(
            self.architecture_dict()[
                "parameter_count"
            ]
        )

    def reset_parameters(
        self,
    ) -> None:
        if self.weight is not None:
            nn.init.ones_(
                self.weight
            )

        if self.bias is not None:
            nn.init.zeros_(
                self.bias
            )

    def architecture_dict(
        self,
    ) -> dict[str, Any]:
        return (
            layer_normalizer_architecture_dict(
                normalization_mode=(
                    self.normalization_mode
                ),
                normalization_position=(
                    self.normalization_position
                ),
                hidden_dim=self.hidden_dim,
                epsilon=self.epsilon,
                elementwise_affine=(
                    self.elementwise_affine
                ),
                bias_enabled=(
                    self.bias_enabled
                ),
            )
        )

    def architecture_fingerprint(
        self,
    ) -> str:
        return (
            layer_normalizer_architecture_fingerprint(
                normalization_mode=(
                    self.normalization_mode
                ),
                normalization_position=(
                    self.normalization_position
                ),
                hidden_dim=self.hidden_dim,
                epsilon=self.epsilon,
                elementwise_affine=(
                    self.elementwise_affine
                ),
                bias_enabled=(
                    self.bias_enabled
                ),
            )
        )

    def parameter_fingerprint(
        self,
    ) -> str | None:
        return (
            _parameter_fingerprint_from_tensors(
                normalization_mode=(
                    self.normalization_mode
                ),
                weight=self.weight,
                bias=self.bias,
            )
        )

    def runtime_dict(
        self,
    ) -> dict[str, Any]:
        return {
            "architecture": (
                self.architecture_dict()
            ),
            "architecture_fingerprint": (
                self.architecture_fingerprint()
            ),
            "parameter_fingerprint": (
                self.parameter_fingerprint()
            ),
            "training": self.training,
            "parameter_count": (
                self.parameter_count
            ),
            "trainable_parameter_count": (
                self.trainable_parameter_count
            ),
            "buffer_count": (
                self.buffer_count
            ),
        }

    def assert_finite_parameters(
        self,
    ) -> None:
        for name, parameter in (
            self.named_parameters()
        ):
            if not bool(
                torch.isfinite(
                    parameter
                )
                .all()
                .item()
            ):
                raise FloatingPointError(
                    f"LayerNormalizer parameter "
                    f"{name!r} contains non-finite values."
                )

    def assert_parameter_contract(
        self,
    ) -> None:
        if self.buffer_count != 0:
            raise RuntimeError(
                "LayerNormalizer must remain buffer-free."
            )

        if self.parameter_count != (
            self.expected_parameter_count
        ):
            raise RuntimeError(
                "LayerNormalizer parameter count differs from its "
                "declared architecture."
            )

        if (
            self.normalization_mode
            == LAYER_NORMALIZATION_NONE
        ):
            if self.weight is not None:
                raise RuntimeError(
                    "Disabled normalization must not own weight."
                )

            if self.bias is not None:
                raise RuntimeError(
                    "Disabled normalization must not own bias."
                )

            if self.state_dict():
                raise RuntimeError(
                    "Disabled normalization must have an empty state_dict."
                )
        else:
            if self.elementwise_affine:
                if self.weight is None:
                    raise RuntimeError(
                        "Affine layer normalization requires weight."
                    )
            elif self.weight is not None:
                raise RuntimeError(
                    "Non-affine layer normalization must not own weight."
                )

            if self.bias_enabled:
                if self.bias is None:
                    raise RuntimeError(
                        "Bias-enabled layer normalization requires bias."
                    )
            elif self.bias is not None:
                raise RuntimeError(
                    "Bias-disabled layer normalization must not own bias."
                )

        self.assert_finite_parameters()

    def normalize(
        self,
        residual_update: LayerResidualUpdateOutput,
    ) -> LayerNormalizationOutput:
        """
        Normalize one exact residual-update output.
        """

        self.assert_parameter_contract()
        _require_residual_update(
            residual_update
        )

        input_state = (
            residual_update
            .post_residual_state
        )

        if int(
            input_state.shape[-1]
        ) != self.hidden_dim:
            raise ValueError(
                "Residual-update hidden dimension differs from "
                f"LayerNormalizer hidden_dim={self.hidden_dim}."
            )

        return build_layer_normalization_output(
            residual_update=(
                residual_update
            ),
            normalization_mode=(
                self.normalization_mode
            ),
            normalization_position=(
                self.normalization_position
            ),
            epsilon=self.epsilon,
            weight=self.weight,
            bias=self.bias,
            normalizer_architecture_fingerprint=(
                self.architecture_fingerprint()
            ),
            normalizer_parameter_fingerprint=(
                self.parameter_fingerprint()
            ),
        )

    def diagnostic_summary(
        self,
        output: LayerNormalizationOutput,
    ) -> dict[str, Any]:
        """
        Validate module provenance and return detached diagnostics.
        """

        self.assert_parameter_contract()

        validate_layer_normalization_output(
            output=output,
            normalization_mode=(
                self.normalization_mode
            ),
            normalization_position=(
                self.normalization_position
            ),
            epsilon=self.epsilon,
            weight=self.weight,
            bias=self.bias,
            normalizer_architecture_fingerprint=(
                self.architecture_fingerprint()
            ),
            normalizer_parameter_fingerprint=(
                self.parameter_fingerprint()
            ),
        )

        summary = (
            layer_normalization_diagnostic_summary(
                output
            )
        )
        summary["parameter_count"] = (
            self.parameter_count
        )
        summary[
            "trainable_parameter_count"
        ] = self.trainable_parameter_count
        summary["buffer_count"] = (
            self.buffer_count
        )
        summary["elementwise_affine"] = (
            self.elementwise_affine
        )
        summary["bias_enabled"] = (
            self.bias_enabled
        )

        return summary

    def forward(
        self,
        residual_update: LayerResidualUpdateOutput,
    ) -> LayerNormalizationOutput:
        return self.normalize(
            residual_update
        )

    def extra_repr(self) -> str:
        return (
            f"hidden_dim={self.hidden_dim}, "
            f"normalization_mode={self.normalization_mode!r}, "
            f"normalization_position={self.normalization_position!r}, "
            f"epsilon={self.epsilon}, "
            f"elementwise_affine={self.elementwise_affine}, "
            f"bias_enabled={self.bias_enabled}, "
            f"parameter_count={self.parameter_count}"
        )


# =============================================================================
# Builders and aliases
# =============================================================================


def build_layer_normalizer(
    hidden_dim: int,
    *,
    normalization_mode: str = (
        LAYER_NORMALIZATION_LAYER_NORM
    ),
    normalization_position: str = (
        LAYER_NORMALIZATION_POST_RESIDUAL
    ),
    epsilon: float = (
        LAYER_NORMALIZER_DEFAULT_EPSILON
    ),
    elementwise_affine: bool = (
        LAYER_NORMALIZER_DEFAULT_ELEMENTWISE_AFFINE
    ),
    bias_enabled: bool = (
        LAYER_NORMALIZER_DEFAULT_BIAS_ENABLED
    ),
    device: (
        torch.device | str | None
    ) = None,
    dtype: torch.dtype | None = None,
) -> LayerNormalizer:
    """
    Construct the optional post-residual normalizer.
    """

    return LayerNormalizer(
        hidden_dim,
        normalization_mode=(
            normalization_mode
        ),
        normalization_position=(
            normalization_position
        ),
        epsilon=epsilon,
        elementwise_affine=(
            elementwise_affine
        ),
        bias_enabled=bias_enabled,
        device=device,
        dtype=dtype,
    )


def build_layer_normalizer_from_flag(
    hidden_dim: int,
    *,
    layer_norm_enabled: bool,
    epsilon: float = (
        LAYER_NORMALIZER_DEFAULT_EPSILON
    ),
    elementwise_affine: bool = (
        LAYER_NORMALIZER_DEFAULT_ELEMENTWISE_AFFINE
    ),
    bias_enabled: bool = (
        LAYER_NORMALIZER_DEFAULT_BIAS_ENABLED
    ),
    device: (
        torch.device | str | None
    ) = None,
    dtype: torch.dtype | None = None,
) -> LayerNormalizer:
    """
    Construct from the historical Boolean layer-normalization flag.
    """

    return LayerNormalizer.from_flag(
        hidden_dim,
        layer_norm_enabled=(
            layer_norm_enabled
        ),
        epsilon=epsilon,
        elementwise_affine=(
            elementwise_affine
        ),
        bias_enabled=bias_enabled,
        device=device,
        dtype=dtype,
    )


def normalize_layer_state(
    *,
    residual_update: LayerResidualUpdateOutput,
    normalization_mode: str,
    epsilon: float = (
        LAYER_NORMALIZER_DEFAULT_EPSILON
    ),
    weight: torch.Tensor | None = None,
    bias: torch.Tensor | None = None,
) -> LayerNormalizationOutput:
    """
    Functional spelling for post-residual normalization.
    """

    return build_layer_normalization_output(
        residual_update=(
            residual_update
        ),
        normalization_mode=(
            normalization_mode
        ),
        normalization_position=(
            LAYER_NORMALIZATION_POST_RESIDUAL
        ),
        epsilon=epsilon,
        weight=weight,
        bias=bias,
    )


FunctionalLayerNormalizer = LayerNormalizer
MessagePassingLayerNormalizer = (
    LayerNormalizer
)
PostResidualLayerNormalizer = (
    LayerNormalizer
)

build_normalizer = build_layer_normalizer
build_normalizer_from_flag = (
    build_layer_normalizer_from_flag
)

apply_normalization = (
    apply_layer_normalization
)
resolve_layer_normalization = (
    normalize_layer_state
)


__all__ = (
    # Public identity.
    "LAYER_NORMALIZER_SCHEMA_VERSION",
    "LAYER_NORMALIZER_OPERATION",
    "LAYER_NORMALIZER_OPERATION_ORDER",
    "LAYER_NORMALIZER_NORMALIZED_AXIS",
    "LAYER_NORMALIZER_STATISTIC_SCOPE",
    "LAYER_NORMALIZER_VARIANCE_ESTIMATOR",
    "LAYER_NORMALIZER_AGGREGATION_OWNED_HERE",
    "LAYER_NORMALIZER_DROPOUT_OWNED_HERE",
    "LAYER_NORMALIZER_RESIDUAL_OWNED_HERE",
    "LAYER_NORMALIZER_MULTI_LAYER_ITERATION_OWNED_HERE",
    "LAYER_DISABLED_NORMALIZATION_IDENTITY_POLICY",
    "LAYER_NORMALIZER_DEFAULT_EPSILON",
    "LAYER_NORMALIZER_DEFAULT_ELEMENTWISE_AFFINE",
    "LAYER_NORMALIZER_DEFAULT_BIAS_ENABLED",
    # Mode conversion.
    "normalization_mode_from_enabled",
    "normalization_enabled_from_mode",
    # Architecture.
    "layer_normalizer_architecture_dict",
    "layer_normalizer_architecture_fingerprint",
    # Low-level operation.
    "apply_layer_normalization",
    "apply_normalization",
    # Complete functional stage.
    "build_layer_normalization_output",
    "normalize_layer_state",
    "resolve_layer_normalization",
    "validate_layer_normalization_output",
    "layer_normalization_diagnostic_summary",
    # Module API.
    "LayerNormalizer",
    "FunctionalLayerNormalizer",
    "MessagePassingLayerNormalizer",
    "PostResidualLayerNormalizer",
    # Builders.
    "build_layer_normalizer",
    "build_normalizer",
    "build_layer_normalizer_from_flag",
    "build_normalizer_from_flag",
)
