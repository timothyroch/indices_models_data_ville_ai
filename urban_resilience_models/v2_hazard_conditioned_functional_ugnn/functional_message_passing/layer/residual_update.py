"""
Parameter-free residual update for one functional message-passing layer.

Target path:
    urban_resilience_models/
        v2_hazard_conditioned_functional_ugnn/
            functional_message_passing/
                layer/
                    residual_update.py

This module owns exactly two node-level operations after target-node message
aggregation:

    post_dropout_update = dropout(node_aggregate)

    post_residual_state =
        source_node_state + post_dropout_update
            when additive residuals are enabled

        post_dropout_update
            when residuals are disabled

The module does not own relation transforms, edge normalization, relation
gates, edge attention, edge-message construction, target-node aggregation,
layer normalization, multi-layer iteration, or prediction.

Exact contracts
---------------
``AggregationOutput.node_aggregate`` is already node aligned ``[N, H]``.
The residual updater therefore performs no scatter, gather, indexing, pooling,
or hidden projection.

The exact source node state is:

    layer_inputs.source_inputs.node_state.fused_state

and must be preserved as the ``residual_source_state`` object in the immutable
``LayerResidualUpdateOutput``.

Dropout identity
----------------
When either:

- ``dropout_probability == 0``; or
- the updater is in evaluation mode,

the post-dropout update is the exact same tensor object as the aggregation
output. No clone, detach, cast, copy, or device move is introduced.

When training with nonzero dropout, ordinary inverted dropout is applied. The
realized tensor preserves shape, dtype, device, and autograd connectivity.
Non-finite results, including numerical overflow caused by dropout rescaling,
are rejected.

Residual identity
-----------------
When residuals are disabled, ``post_residual_state`` is the exact same tensor
object as ``post_dropout_update``.

When additive residuals are enabled, a new differentiable tensor is produced
by exact elementwise addition with the source node state.

Training contract
-----------------
``FunctionalMessagePassingLayerInputs.training`` must agree with the module's
current ``training`` flag. This prevents traces from falsely reporting that
dropout was enabled or disabled under a different execution mode.

Interpretation
-------------------------
The residual update is a state-transition mechanism. Magnitudes in the source
state, update branch, or residual output are descriptive model values and are
not automatically causal importance, explanation faithfulness, uncertainty,
or mechanistic-identifiability evidence.

The updater is parameter-free and buffer-free.
"""

from __future__ import annotations

from hashlib import sha256
import json
import math
from typing import Any, Final, Mapping

import torch
from torch import nn
from torch.nn import functional as F

from ..schemas import (
    AggregationOutput,
)
from .schemas import (
    CANONICAL_LAYER_RESIDUAL_MODES,
    LAYER_ADDITIVE_RESIDUAL_FORMULA,
    LAYER_DISABLED_RESIDUAL_FORMULA,
    LAYER_RESIDUAL_ADDITIVE,
    LAYER_RESIDUAL_DISABLED,
    LAYER_UPDATE_BRANCH_FORMULA,
    V2_0_IMPLEMENTED_LAYER_RESIDUAL_MODES,
    FunctionalMessagePassingLayerInputs,
    LayerResidualUpdateOutput,
)


# =============================================================================
# Public identity
# =============================================================================


LAYER_RESIDUAL_UPDATER_SCHEMA_VERSION: Final[str] = "0.1"

LAYER_RESIDUAL_UPDATER_OPERATION: Final[str] = (
    "dropout_then_optional_additive_residual_update"
)

LAYER_RESIDUAL_UPDATER_OPERATION_ORDER: Final[
    tuple[str, ...]
] = (
    "validate_exact_aggregation_and_layer_input_lineage",
    "bind_pre_dropout_update_to_exact_node_aggregate",
    "apply_or_bypass_inverted_dropout",
    "bind_pre_residual_state_to_realized_update_branch",
    "bind_residual_source_to_exact_input_node_state",
    "apply_or_bypass_additive_residual",
    "validate_finite_post_residual_state",
    "construct_layer_residual_update_output",
)

LAYER_RESIDUAL_UPDATER_PARAMETER_FREE: Final[bool] = True
LAYER_RESIDUAL_UPDATER_BUFFER_FREE: Final[bool] = True
LAYER_RESIDUAL_UPDATER_PROJECTION_OWNED_HERE: Final[bool] = False
LAYER_RESIDUAL_UPDATER_AGGREGATION_OWNED_HERE: Final[bool] = False
LAYER_RESIDUAL_UPDATER_NORMALIZATION_OWNED_HERE: Final[bool] = False

LAYER_DROPOUT_SEMANTICS: Final[str] = (
    "elementwise_inverted_dropout_on_node_aggregate"
)

LAYER_DISABLED_DROPOUT_IDENTITY_POLICY: Final[str] = (
    "exact_input_tensor_identity"
)

LAYER_DISABLED_RESIDUAL_IDENTITY_POLICY: Final[str] = (
    "exact_post_dropout_tensor_identity"
)


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


def _require_probability(
    name: str,
    value: float,
) -> float:
    if isinstance(value, bool) or not isinstance(
        value,
        (int, float),
    ):
        raise TypeError(
            f"{name} must be numeric."
        )

    numeric = float(value)

    if not math.isfinite(numeric):
        raise ValueError(
            f"{name} must be finite."
        )

    if not 0.0 <= numeric < 1.0:
        raise ValueError(
            f"{name} must lie in [0, 1)."
        )

    return numeric


def _require_boolean(
    name: str,
    value: bool,
) -> None:
    if not isinstance(value, bool):
        raise TypeError(
            f"{name} must be Boolean."
        )


def _require_nonempty_string(
    name: str,
    value: str,
) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(
            f"{name} must be a non-empty string."
        )


def _require_residual_mode(
    residual_mode: str,
) -> None:
    _require_nonempty_string(
        "residual_mode",
        residual_mode,
    )

    if residual_mode not in (
        CANONICAL_LAYER_RESIDUAL_MODES
    ):
        raise ValueError(
            "residual_mode must be one of "
            f"{CANONICAL_LAYER_RESIDUAL_MODES!r}; "
            f"observed {residual_mode!r}."
        )

    if residual_mode not in (
        V2_0_IMPLEMENTED_LAYER_RESIDUAL_MODES
    ):
        raise NotImplementedError(
            f"Residual mode {residual_mode!r} is canonical but not "
            "implemented in bounded V2.0."
        )


def _require_layer_inputs(
    layer_inputs: FunctionalMessagePassingLayerInputs,
) -> None:
    if not isinstance(
        layer_inputs,
        FunctionalMessagePassingLayerInputs,
    ):
        raise TypeError(
            "layer_inputs must be a "
            "FunctionalMessagePassingLayerInputs."
        )


def _require_aggregation(
    aggregation: AggregationOutput,
) -> None:
    if not isinstance(
        aggregation,
        AggregationOutput,
    ):
        raise TypeError(
            "aggregation must be an AggregationOutput."
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
            f"{name} must have rank 2 and shape [N, H]; "
            f"observed {tuple(value.shape)}."
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


def _require_same_tensor_contract(
    *,
    left_name: str,
    left: torch.Tensor,
    right_name: str,
    right: torch.Tensor,
) -> None:
    _require_float_matrix(
        left_name,
        left,
    )
    _require_float_matrix(
        right_name,
        right,
    )

    if tuple(left.shape) != tuple(right.shape):
        raise ValueError(
            f"{left_name} and {right_name} must share one shape; "
            f"observed {tuple(left.shape)} and {tuple(right.shape)}."
        )

    if left.dtype != right.dtype:
        raise ValueError(
            f"{left_name} and {right_name} must share one dtype; "
            f"observed {left.dtype} and {right.dtype}."
        )

    if left.device != right.device:
        raise ValueError(
            f"{left_name} and {right_name} must share one device; "
            f"observed {left.device} and {right.device}."
        )


def _require_exact_layer_lineage(
    *,
    aggregation: AggregationOutput,
    layer_inputs: FunctionalMessagePassingLayerInputs,
) -> None:
    _require_aggregation(
        aggregation
    )
    _require_layer_inputs(
        layer_inputs
    )

    aggregation_inputs = (
        aggregation
        .source_messages
        .source_inputs
    )

    if aggregation_inputs is not (
        layer_inputs.source_inputs
    ):
        raise ValueError(
            "aggregation and layer_inputs must share the exact same "
            "FunctionalMessagePassingInputs object."
        )

    _require_same_tensor_contract(
        left_name="aggregation.node_aggregate",
        left=aggregation.node_aggregate,
        right_name="layer_inputs.input_node_state",
        right=layer_inputs.input_node_state,
    )

    expected_shape = (
        layer_inputs.num_nodes,
        layer_inputs.hidden_dim,
    )

    if tuple(
        aggregation.node_aggregate.shape
    ) != expected_shape:
        raise ValueError(
            "aggregation.node_aggregate must have shape "
            f"{expected_shape}; observed "
            f"{tuple(aggregation.node_aggregate.shape)}."
        )


# =============================================================================
# Residual-mode conversion
# =============================================================================


def residual_mode_from_enabled(
    residual_enabled: bool,
) -> str:
    """
    Convert the historical Boolean residual flag into the explicit mode.
    """

    _require_boolean(
        "residual_enabled",
        residual_enabled,
    )

    return (
        LAYER_RESIDUAL_ADDITIVE
        if residual_enabled
        else LAYER_RESIDUAL_DISABLED
    )


def residual_enabled_from_mode(
    residual_mode: str,
) -> bool:
    """
    Convert the explicit residual mode into the public Boolean flag.
    """

    _require_residual_mode(
        residual_mode
    )

    return residual_mode == (
        LAYER_RESIDUAL_ADDITIVE
    )


# =============================================================================
# Static architecture identity
# =============================================================================


def layer_residual_updater_architecture_dict(
    *,
    residual_mode: str,
    dropout_probability: float,
) -> dict[str, Any]:
    """
    Return the numerical residual-update architecture.

    Training/evaluation mode is deliberately excluded because it is runtime
    state, not model architecture.
    """

    _require_residual_mode(
        residual_mode
    )
    dropout_probability = (
        _require_probability(
            "dropout_probability",
            dropout_probability,
        )
    )

    return {
        "schema_version": (
            LAYER_RESIDUAL_UPDATER_SCHEMA_VERSION
        ),
        "operation": (
            LAYER_RESIDUAL_UPDATER_OPERATION
        ),
        "operation_order": list(
            LAYER_RESIDUAL_UPDATER_OPERATION_ORDER
        ),
        "residual_mode": residual_mode,
        "residual_enabled": (
            residual_enabled_from_mode(
                residual_mode
            )
        ),
        "dropout_probability": (
            dropout_probability
        ),
        "dropout_semantics": (
            LAYER_DROPOUT_SEMANTICS
        ),
        "dropout_identity_policy": (
            LAYER_DISABLED_DROPOUT_IDENTITY_POLICY
        ),
        "residual_identity_policy": (
            LAYER_DISABLED_RESIDUAL_IDENTITY_POLICY
        ),
        "update_branch_formula": (
            LAYER_UPDATE_BRANCH_FORMULA
        ),
        "residual_formula": (
            LAYER_ADDITIVE_RESIDUAL_FORMULA
            if residual_mode
            == LAYER_RESIDUAL_ADDITIVE
            else LAYER_DISABLED_RESIDUAL_FORMULA
        ),
        "parameter_free": (
            LAYER_RESIDUAL_UPDATER_PARAMETER_FREE
        ),
        "buffer_free": (
            LAYER_RESIDUAL_UPDATER_BUFFER_FREE
        ),
        "projection_owned_here": (
            LAYER_RESIDUAL_UPDATER_PROJECTION_OWNED_HERE
        ),
        "aggregation_owned_here": (
            LAYER_RESIDUAL_UPDATER_AGGREGATION_OWNED_HERE
        ),
        "normalization_owned_here": (
            LAYER_RESIDUAL_UPDATER_NORMALIZATION_OWNED_HERE
        ),
        "claims_causal_importance": False,
        "claims_explanation_faithfulness": False,
    }


def layer_residual_updater_architecture_fingerprint(
    *,
    residual_mode: str,
    dropout_probability: float,
) -> str:
    return _fingerprint(
        layer_residual_updater_architecture_dict(
            residual_mode=residual_mode,
            dropout_probability=(
                dropout_probability
            ),
        )
    )


# =============================================================================
# Low-level numerical operations
# =============================================================================


def apply_layer_update_dropout(
    pre_dropout_update: torch.Tensor,
    *,
    dropout_probability: float,
    training: bool,
) -> torch.Tensor:
    """
    Apply inverted dropout to one node-aligned update tensor.

    Disabled or evaluation-mode dropout returns the exact input tensor object.
    """

    _require_float_matrix(
        "pre_dropout_update",
        pre_dropout_update,
    )
    dropout_probability = (
        _require_probability(
            "dropout_probability",
            dropout_probability,
        )
    )
    _require_boolean(
        "training",
        training,
    )

    if (
        dropout_probability == 0.0
        or not training
    ):
        return pre_dropout_update

    post_dropout_update = F.dropout(
        pre_dropout_update,
        p=dropout_probability,
        training=True,
        inplace=False,
    )

    if tuple(
        post_dropout_update.shape
    ) != tuple(
        pre_dropout_update.shape
    ):
        raise RuntimeError(
            "Dropout changed the node-update shape unexpectedly."
        )

    if post_dropout_update.dtype != (
        pre_dropout_update.dtype
    ):
        raise RuntimeError(
            "Dropout changed the node-update dtype unexpectedly."
        )

    if post_dropout_update.device != (
        pre_dropout_update.device
    ):
        raise RuntimeError(
            "Dropout changed the node-update device unexpectedly."
        )

    if not bool(
        torch.isfinite(
            post_dropout_update
        )
        .all()
        .item()
    ):
        raise FloatingPointError(
            "Layer update dropout produced non-finite values. This may "
            "indicate overflow during inverted-dropout rescaling."
        )

    return post_dropout_update


def apply_layer_residual(
    *,
    residual_source_state: torch.Tensor,
    post_dropout_update: torch.Tensor,
    residual_mode: str,
) -> torch.Tensor:
    """
    Apply the explicit residual mode.

    Disabled residuals return ``post_dropout_update`` by exact identity.
    """

    _require_residual_mode(
        residual_mode
    )
    _require_same_tensor_contract(
        left_name="residual_source_state",
        left=residual_source_state,
        right_name="post_dropout_update",
        right=post_dropout_update,
    )

    if residual_mode == (
        LAYER_RESIDUAL_DISABLED
    ):
        return post_dropout_update

    post_residual_state = (
        residual_source_state
        + post_dropout_update
    )

    if tuple(
        post_residual_state.shape
    ) != tuple(
        residual_source_state.shape
    ):
        raise RuntimeError(
            "Residual addition changed shape unexpectedly."
        )

    if post_residual_state.dtype != (
        residual_source_state.dtype
    ):
        raise RuntimeError(
            "Residual addition changed dtype unexpectedly."
        )

    if post_residual_state.device != (
        residual_source_state.device
    ):
        raise RuntimeError(
            "Residual addition changed device unexpectedly."
        )

    if not bool(
        torch.isfinite(
            post_residual_state
        )
        .all()
        .item()
    ):
        raise FloatingPointError(
            "Residual addition produced non-finite values. This may "
            "indicate overflow in the source state or update branch."
        )

    return post_residual_state


# =============================================================================
# Complete functional residual update
# =============================================================================


def build_layer_residual_update_output(
    *,
    aggregation: AggregationOutput,
    layer_inputs: FunctionalMessagePassingLayerInputs,
    residual_mode: str,
    dropout_probability: float,
    training: bool,
    updater_architecture_fingerprint: (
        str | None
    ) = None,
) -> LayerResidualUpdateOutput:
    """
    Execute the complete parameter-free residual-update stage.
    """

    _require_exact_layer_lineage(
        aggregation=aggregation,
        layer_inputs=layer_inputs,
    )
    _require_residual_mode(
        residual_mode
    )
    dropout_probability = (
        _require_probability(
            "dropout_probability",
            dropout_probability,
        )
    )
    _require_boolean(
        "training",
        training,
    )

    if training != layer_inputs.training:
        raise ValueError(
            "Residual-update training mode must match "
            "layer_inputs.training."
        )

    pre_dropout_update = (
        aggregation.node_aggregate
    )
    post_dropout_update = (
        apply_layer_update_dropout(
            pre_dropout_update,
            dropout_probability=(
                dropout_probability
            ),
            training=training,
        )
    )

    pre_residual_state = (
        post_dropout_update
    )
    residual_source_state = (
        layer_inputs.input_node_state
    )
    post_residual_state = (
        apply_layer_residual(
            residual_source_state=(
                residual_source_state
            ),
            post_dropout_update=(
                post_dropout_update
            ),
            residual_mode=residual_mode,
        )
    )

    if (
        updater_architecture_fingerprint
        is None
    ):
        updater_architecture_fingerprint = (
            layer_residual_updater_architecture_fingerprint(
                residual_mode=residual_mode,
                dropout_probability=(
                    dropout_probability
                ),
            )
        )

    _require_nonempty_string(
        "updater_architecture_fingerprint",
        updater_architecture_fingerprint,
    )

    output = LayerResidualUpdateOutput(
        pre_dropout_update=(
            pre_dropout_update
        ),
        post_dropout_update=(
            post_dropout_update
        ),
        pre_residual_state=(
            pre_residual_state
        ),
        post_residual_state=(
            post_residual_state
        ),
        residual_source_state=(
            residual_source_state
        ),
        aggregation=aggregation,
        layer_inputs=layer_inputs,
        residual_mode=residual_mode,
        dropout_probability=(
            dropout_probability
        ),
        training=training,
        updater_architecture_fingerprint=(
            updater_architecture_fingerprint
        ),
        updater_parameter_fingerprint=None,
    )

    validate_layer_residual_update_output(
        output=output,
        aggregation=aggregation,
        layer_inputs=layer_inputs,
        residual_mode=residual_mode,
        dropout_probability=(
            dropout_probability
        ),
        training=training,
        updater_architecture_fingerprint=(
            updater_architecture_fingerprint
        ),
    )

    return output


# =============================================================================
# Output validation
# =============================================================================


def validate_layer_residual_update_output(
    *,
    output: LayerResidualUpdateOutput,
    aggregation: AggregationOutput | None = None,
    layer_inputs: (
        FunctionalMessagePassingLayerInputs
        | None
    ) = None,
    residual_mode: str | None = None,
    dropout_probability: float | None = None,
    training: bool | None = None,
    updater_architecture_fingerprint: (
        str | None
    ) = None,
) -> None:
    """
    Validate one residual-update output and optional exact expectations.
    """

    if not isinstance(
        output,
        LayerResidualUpdateOutput,
    ):
        raise TypeError(
            "output must be a "
            "LayerResidualUpdateOutput."
        )

    _require_exact_layer_lineage(
        aggregation=output.aggregation,
        layer_inputs=output.layer_inputs,
    )

    if aggregation is not None:
        _require_aggregation(
            aggregation
        )

        if output.aggregation is not (
            aggregation
        ):
            raise ValueError(
                "output must preserve the exact expected aggregation "
                "object."
            )

    if layer_inputs is not None:
        _require_layer_inputs(
            layer_inputs
        )

        if output.layer_inputs is not (
            layer_inputs
        ):
            raise ValueError(
                "output must preserve the exact expected layer_inputs "
                "object."
            )

    if residual_mode is not None:
        _require_residual_mode(
            residual_mode
        )

        if output.residual_mode != (
            residual_mode
        ):
            raise ValueError(
                "output residual_mode differs from the expected mode."
            )

    if dropout_probability is not None:
        expected_probability = (
            _require_probability(
                "dropout_probability",
                dropout_probability,
            )
        )

        if output.dropout_probability != (
            expected_probability
        ):
            raise ValueError(
                "output dropout_probability differs from the expected "
                "probability."
            )

    if training is not None:
        _require_boolean(
            "training",
            training,
        )

        if output.training is not training:
            raise ValueError(
                "output training flag differs from the expected mode."
            )

    if (
        updater_architecture_fingerprint
        is not None
    ):
        _require_nonempty_string(
            "updater_architecture_fingerprint",
            updater_architecture_fingerprint,
        )

        if (
            output
            .updater_architecture_fingerprint
            != updater_architecture_fingerprint
        ):
            raise ValueError(
                "output updater architecture fingerprint differs from "
                "the expected fingerprint."
            )

    if output.pre_dropout_update is not (
        output
        .aggregation
        .node_aggregate
    ):
        raise ValueError(
            "pre_dropout_update must preserve the exact aggregation "
            "tensor object."
        )

    if output.pre_residual_state is not (
        output.post_dropout_update
    ):
        raise ValueError(
            "pre_residual_state must preserve the exact realized "
            "post_dropout_update tensor object."
        )

    if output.residual_source_state is not (
        output.layer_inputs.input_node_state
    ):
        raise ValueError(
            "residual_source_state must preserve the exact source node-state "
            "tensor object."
        )

    _require_same_tensor_contract(
        left_name="residual_source_state",
        left=output.residual_source_state,
        right_name="post_dropout_update",
        right=output.post_dropout_update,
    )
    _require_same_tensor_contract(
        left_name="post_residual_state",
        left=output.post_residual_state,
        right_name="post_dropout_update",
        right=output.post_dropout_update,
    )

    if (
        output.dropout_probability == 0.0
        or not output.training
    ):
        if output.post_dropout_update is not (
            output.pre_dropout_update
        ):
            raise ValueError(
                "Disabled or evaluation-mode dropout must preserve exact "
                "input tensor identity."
            )

    if output.residual_mode == (
        LAYER_RESIDUAL_DISABLED
    ):
        if output.post_residual_state is not (
            output.post_dropout_update
        ):
            raise ValueError(
                "Disabled residual update must preserve exact "
                "post_dropout_update tensor identity."
            )
    else:
        expected = (
            output.residual_source_state
            + output.post_dropout_update
        )
        atol, rtol = (
            _default_tolerances(
                output.post_residual_state.dtype
            )
        )

        if not torch.allclose(
            output.post_residual_state,
            expected,
            atol=atol,
            rtol=rtol,
        ):
            raise ValueError(
                "Additive residual output differs from source state plus "
                "post-dropout update."
            )

    expected_fingerprint = (
        layer_residual_updater_architecture_fingerprint(
            residual_mode=(
                output.residual_mode
            ),
            dropout_probability=(
                output.dropout_probability
            ),
        )
    )

    if (
        output
        .updater_architecture_fingerprint
        != expected_fingerprint
    ):
        raise ValueError(
            "output updater architecture fingerprint does not match the "
            "declared residual mode and dropout probability."
        )

    if (
        output.updater_parameter_fingerprint
        is not None
    ):
        raise ValueError(
            "The parameter-free residual updater must not expose a "
            "parameter fingerprint."
        )


def _default_tolerances(
    dtype: torch.dtype,
) -> tuple[float, float]:
    if dtype in (
        torch.float16,
        torch.bfloat16,
    ):
        return 1e-3, 1e-3

    if dtype == torch.float64:
        return 1e-10, 1e-9

    return 1e-6, 1e-5


# =============================================================================
# Descriptive diagnostics
# =============================================================================


def _tensor_statistics(
    value: torch.Tensor,
) -> dict[str, Any]:
    _require_float_matrix(
        "value",
        value,
    )

    detached = value.detach()
    count = int(
        detached.numel()
    )

    if count == 0:
        return {
            "element_count": 0,
            "minimum": None,
            "maximum": None,
            "mean": None,
            "mean_absolute_value": None,
            "l2_norm": 0.0,
            "zero_count": 0,
            "finite": True,
        }

    return {
        "element_count": count,
        "minimum": float(
            detached.min().item()
        ),
        "maximum": float(
            detached.max().item()
        ),
        "mean": float(
            detached.mean().item()
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


def layer_residual_update_diagnostic_summary(
    output: LayerResidualUpdateOutput,
) -> dict[str, Any]:
    """
    Return detached descriptive statistics for one residual update.

    Values are not causal or explanation-faithfulness claims.
    """

    validate_layer_residual_update_output(
        output=output
    )

    post_dropout_is_input = (
        output.post_dropout_update
        is output.pre_dropout_update
    )
    post_residual_is_update = (
        output.post_residual_state
        is output.post_dropout_update
    )

    return {
        "schema_version": (
            LAYER_RESIDUAL_UPDATER_SCHEMA_VERSION
        ),
        "operation": (
            LAYER_RESIDUAL_UPDATER_OPERATION
        ),
        "layer_index": (
            output.layer_inputs.layer_index
        ),
        "residual_mode": (
            output.residual_mode
        ),
        "residual_enabled": (
            output.residual_enabled
        ),
        "dropout_probability": float(
            output.dropout_probability
        ),
        "training": output.training,
        "dropout_active": bool(
            output.training
            and output.dropout_probability
            > 0.0
        ),
        "num_nodes": output.num_nodes,
        "hidden_dim": output.hidden_dim,
        "dtype": str(output.dtype),
        "device": str(output.device),
        "pre_dropout_update": (
            _tensor_statistics(
                output.pre_dropout_update
            )
        ),
        "post_dropout_update": (
            _tensor_statistics(
                output.post_dropout_update
            )
        ),
        "residual_source_state": (
            _tensor_statistics(
                output.residual_source_state
            )
        ),
        "post_residual_state": (
            _tensor_statistics(
                output.post_residual_state
            )
        ),
        "exact_identity": {
            "pre_dropout_is_exact_aggregate": (
                output.pre_dropout_update
                is output
                .aggregation
                .node_aggregate
            ),
            "post_dropout_is_pre_dropout": (
                post_dropout_is_input
            ),
            "pre_residual_is_post_dropout": (
                output.pre_residual_state
                is output.post_dropout_update
            ),
            "residual_source_is_exact_input_state": (
                output.residual_source_state
                is output
                .layer_inputs
                .input_node_state
            ),
            "post_residual_is_post_dropout": (
                post_residual_is_update
            ),
        },
        "parameter_free": True,
        "buffer_free": True,
        "aggregation_performed_here": False,
        "normalization_performed_here": False,
        "causal_importance_claim": False,
        "explanation_faithfulness_claim": False,
    }


# =============================================================================
# Parameter-free module wrapper
# =============================================================================


class LayerResidualUpdater(nn.Module):
    """
    Parameter-free dropout and optional additive residual stage.

    Parameters
    ----------
    residual_mode:
        ``"additive"`` or ``"disabled"``.
    dropout_probability:
        Elementwise inverted-dropout probability applied to the node aggregate
        before residual addition.
    """

    residual_mode: str
    dropout_probability: float

    def __init__(
        self,
        *,
        residual_mode: str = (
            LAYER_RESIDUAL_ADDITIVE
        ),
        dropout_probability: float = 0.0,
    ) -> None:
        super().__init__()

        _require_residual_mode(
            residual_mode
        )
        dropout_probability = (
            _require_probability(
                "dropout_probability",
                dropout_probability,
            )
        )

        self.residual_mode = residual_mode
        self.dropout_probability = (
            dropout_probability
        )

        self.assert_parameter_free()

    @classmethod
    def from_flags(
        cls,
        *,
        residual_enabled: bool,
        dropout_probability: float,
    ) -> "LayerResidualUpdater":
        return cls(
            residual_mode=(
                residual_mode_from_enabled(
                    residual_enabled
                )
            ),
            dropout_probability=(
                dropout_probability
            ),
        )

    @property
    def residual_enabled(self) -> bool:
        return residual_enabled_from_mode(
            self.residual_mode
        )

    @property
    def dropout_enabled(self) -> bool:
        return self.dropout_probability > 0.0

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
    def parameter_fingerprint(
        self,
    ) -> None:
        return None

    def architecture_dict(
        self,
    ) -> dict[str, Any]:
        return (
            layer_residual_updater_architecture_dict(
                residual_mode=(
                    self.residual_mode
                ),
                dropout_probability=(
                    self.dropout_probability
                ),
            )
        )

    def architecture_fingerprint(
        self,
    ) -> str:
        return (
            layer_residual_updater_architecture_fingerprint(
                residual_mode=(
                    self.residual_mode
                ),
                dropout_probability=(
                    self.dropout_probability
                ),
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
            "training": self.training,
            "dropout_active": bool(
                self.training
                and self.dropout_probability
                > 0.0
            ),
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

    def assert_parameter_free(
        self,
    ) -> None:
        parameters = tuple(
            self.named_parameters()
        )
        buffers = tuple(
            self.named_buffers()
        )

        if parameters:
            raise RuntimeError(
                "LayerResidualUpdater must remain parameter-free. "
                f"Observed parameters: "
                f"{tuple(name for name, _ in parameters)}."
            )

        if buffers:
            raise RuntimeError(
                "LayerResidualUpdater must remain buffer-free. "
                f"Observed buffers: "
                f"{tuple(name for name, _ in buffers)}."
            )

        if self.state_dict():
            raise RuntimeError(
                "LayerResidualUpdater must have an empty state_dict."
            )

        if self.parameter_count != 0:
            raise RuntimeError(
                "LayerResidualUpdater parameter_count must be zero."
            )

        if self.trainable_parameter_count != 0:
            raise RuntimeError(
                "LayerResidualUpdater trainable_parameter_count must be "
                "zero."
            )

        if self.buffer_count != 0:
            raise RuntimeError(
                "LayerResidualUpdater buffer_count must be zero."
            )

    def update(
        self,
        *,
        aggregation: AggregationOutput,
        layer_inputs: (
            FunctionalMessagePassingLayerInputs
        ),
    ) -> LayerResidualUpdateOutput:
        """
        Execute one residual-update stage under the module's runtime mode.
        """

        self.assert_parameter_free()

        if layer_inputs.training is not (
            self.training
        ):
            raise ValueError(
                "layer_inputs.training must match "
                "LayerResidualUpdater.training. Construct layer_inputs after "
                "setting train/eval mode."
            )

        return build_layer_residual_update_output(
            aggregation=aggregation,
            layer_inputs=layer_inputs,
            residual_mode=self.residual_mode,
            dropout_probability=(
                self.dropout_probability
            ),
            training=self.training,
            updater_architecture_fingerprint=(
                self.architecture_fingerprint()
            ),
        )

    def diagnostic_summary(
        self,
        output: LayerResidualUpdateOutput,
    ) -> dict[str, Any]:
        """
        Validate module provenance and return detached diagnostics.
        """

        self.assert_parameter_free()

        validate_layer_residual_update_output(
            output=output,
            residual_mode=(
                self.residual_mode
            ),
            dropout_probability=(
                self.dropout_probability
            ),
            training=output.training,
            updater_architecture_fingerprint=(
                self.architecture_fingerprint()
            ),
        )

        return (
            layer_residual_update_diagnostic_summary(
                output
            )
        )

    def forward(
        self,
        *,
        aggregation: AggregationOutput,
        layer_inputs: (
            FunctionalMessagePassingLayerInputs
        ),
    ) -> LayerResidualUpdateOutput:
        return self.update(
            aggregation=aggregation,
            layer_inputs=layer_inputs,
        )

    def extra_repr(self) -> str:
        return (
            f"residual_mode={self.residual_mode!r}, "
            f"dropout_probability={self.dropout_probability}, "
            "projection_owned_here=False, "
            "parameter_free=True"
        )


# =============================================================================
# Builders and aliases
# =============================================================================


def build_layer_residual_updater(
    *,
    residual_mode: str = (
        LAYER_RESIDUAL_ADDITIVE
    ),
    dropout_probability: float = 0.0,
) -> LayerResidualUpdater:
    """
    Construct the parameter-free residual updater.
    """

    return LayerResidualUpdater(
        residual_mode=residual_mode,
        dropout_probability=(
            dropout_probability
        ),
    )


def build_layer_residual_updater_from_flags(
    *,
    residual_enabled: bool,
    dropout_probability: float,
) -> LayerResidualUpdater:
    """
    Construct from the historical Boolean residual configuration.
    """

    return LayerResidualUpdater.from_flags(
        residual_enabled=(
            residual_enabled
        ),
        dropout_probability=(
            dropout_probability
        ),
    )


def apply_layer_residual_update(
    *,
    aggregation: AggregationOutput,
    layer_inputs: FunctionalMessagePassingLayerInputs,
    residual_mode: str,
    dropout_probability: float,
    training: bool,
) -> LayerResidualUpdateOutput:
    """
    Functional spelling for the complete residual-update stage.
    """

    return build_layer_residual_update_output(
        aggregation=aggregation,
        layer_inputs=layer_inputs,
        residual_mode=residual_mode,
        dropout_probability=(
            dropout_probability
        ),
        training=training,
    )


ResidualUpdater = LayerResidualUpdater
FunctionalResidualUpdater = (
    LayerResidualUpdater
)
MessagePassingResidualUpdater = (
    LayerResidualUpdater
)

build_residual_updater = (
    build_layer_residual_updater
)
build_residual_updater_from_flags = (
    build_layer_residual_updater_from_flags
)

apply_update_dropout = (
    apply_layer_update_dropout
)
apply_residual = apply_layer_residual
resolve_layer_residual_update = (
    apply_layer_residual_update
)


__all__ = (
    # Public identity.
    "LAYER_RESIDUAL_UPDATER_SCHEMA_VERSION",
    "LAYER_RESIDUAL_UPDATER_OPERATION",
    "LAYER_RESIDUAL_UPDATER_OPERATION_ORDER",
    "LAYER_RESIDUAL_UPDATER_PARAMETER_FREE",
    "LAYER_RESIDUAL_UPDATER_BUFFER_FREE",
    "LAYER_RESIDUAL_UPDATER_PROJECTION_OWNED_HERE",
    "LAYER_RESIDUAL_UPDATER_AGGREGATION_OWNED_HERE",
    "LAYER_RESIDUAL_UPDATER_NORMALIZATION_OWNED_HERE",
    "LAYER_DROPOUT_SEMANTICS",
    "LAYER_DISABLED_DROPOUT_IDENTITY_POLICY",
    "LAYER_DISABLED_RESIDUAL_IDENTITY_POLICY",
    # Residual-mode conversion.
    "residual_mode_from_enabled",
    "residual_enabled_from_mode",
    # Architecture.
    "layer_residual_updater_architecture_dict",
    "layer_residual_updater_architecture_fingerprint",
    # Low-level numerical operations.
    "apply_layer_update_dropout",
    "apply_update_dropout",
    "apply_layer_residual",
    "apply_residual",
    # Complete functional stage.
    "build_layer_residual_update_output",
    "apply_layer_residual_update",
    "resolve_layer_residual_update",
    "validate_layer_residual_update_output",
    "layer_residual_update_diagnostic_summary",
    # Module API.
    "LayerResidualUpdater",
    "ResidualUpdater",
    "FunctionalResidualUpdater",
    "MessagePassingResidualUpdater",
    # Builders.
    "build_layer_residual_updater",
    "build_residual_updater",
    "build_layer_residual_updater_from_flags",
    "build_residual_updater_from_flags",
)
