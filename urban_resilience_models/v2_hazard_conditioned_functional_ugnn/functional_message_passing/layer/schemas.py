"""
Immutable schemas for one functional message-passing layer.

Target path:
    urban_resilience_models/
        v2_hazard_conditioned_functional_ugnn/
            functional_message_passing/
                layer/
                    schemas.py

This module freezes the contracts around one complete functional
message-passing layer without implementing the neural or scatter mathematics.

A bounded layer consumes one exact ``FunctionalMessagePassingInputs`` object
and coordinates already separated subsystems:

    relation transforms
        -> structural edge normalization
        -> optional exact-relation gating
        -> optional edge attention
        -> edge-message construction
        -> target-node aggregation
        -> residual update
        -> layer normalization
        -> updated node state

Stable shape contract
---------------------
The layer preserves hidden width:

    source node state       [N, H]
    target-node aggregate   [N, H]
    residual update         [N, H]
    normalized output       [N, H]

The layer does not own multi-layer iteration or prediction heads.

Trace policy
------------
Intermediate traces may be expensive because full tracing retains edge-level
objects and tensors. Three explicit policies are defined:

``none``
    Retain no optional layer trace.

``node``
    Retain node-level aggregation, residual-update, and normalization stages,
    but not edge-level transform, gate, attention, or message-builder stages.

``full``
    Retain the complete exact stage chain, including edge-level objects.

The current Boolean configuration field ``capture_intermediate_messages`` maps
to ``none`` or ``full``. The ``node`` policy is preserved as an explicit
forward-compatible contract for diagnostics and stack-level trace control.

Residual update
---------------
The residual stage receives the target-node aggregate and the exact source
node state. In the bounded additive form:

    update_branch = dropout(node_aggregate)

    post_residual_state =
        source_node_state + update_branch     when residual is enabled
        update_branch                         when residual is disabled

The schema records both pre-dropout and post-dropout update tensors so training
stochasticity remains auditable. It does not attempt to infer a dropout mask.

Normalization
-------------
Normalization is explicit:

``none``
    The output is the exact input tensor object.

``layer_norm``
    A layer-normalization module produces the final node state.

Normalization placement is also explicit. The bounded V2.0 layer uses
post-residual normalization, while pre-residual normalization remains a
canonical future extension.

Lineage
-------
Every retained object must refer to the exact same
``FunctionalMessagePassingInputs`` instance. Equivalent values from distinct
objects are rejected. Final tensors preserve dtype, device, node order, graph
membership, and hidden width.

Public compatibility
--------------------
The existing top-level public contracts are re-exported:

- ``AggregationOutput``
- ``FunctionalMessagePassingIntermediates``
- ``FunctionalMessagePassingLayerOutput``

This module adds richer internal residual, normalization, trace, and complete
layer-stage schemas, plus validators for constructing the existing public
layer output without silently changing its meaning.

Limits
-----------------
Retained gates, attention weights, coefficients, aggregates, and node-state
magnitudes are diagnostic model traces. They are not automatically:

- causal importance scores;
- faithful explanations;
- calibrated uncertainty;
- counterfactual effects;
- mechanistic identifiability evidence.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from hashlib import sha256
import json
import math
from types import MappingProxyType
from typing import Any, Final, Mapping, NamedTuple

import torch

from ..message_builders import (
    MessageBuilderRun,
)
from ..schemas import (
    AggregationOutput,
    EdgeAttentionOutput,
    EdgeMessageOutput,
    FunctionalMessagePassingInputs,
    FunctionalMessagePassingIntermediates,
    FunctionalMessagePassingLayerOutput,
    RelationGateOutput,
    RelationTransformOutput,
    StructuralEdgeNormalizationOutput,
)


# =============================================================================
# Schema versions
# =============================================================================


LAYER_INPUTS_SCHEMA_VERSION: Final[str] = "0.1"
LAYER_TRACE_POLICY_SCHEMA_VERSION: Final[str] = "0.1"
LAYER_RESIDUAL_UPDATE_SCHEMA_VERSION: Final[str] = "0.1"
LAYER_NORMALIZATION_OUTPUT_SCHEMA_VERSION: Final[str] = "0.1"
LAYER_INTERMEDIATE_TRACE_SCHEMA_VERSION: Final[str] = "0.1"
LAYER_COMPUTATION_OUTPUT_SCHEMA_VERSION: Final[str] = "0.1"


# =============================================================================
# Trace-policy vocabulary
# =============================================================================


LAYER_TRACE_NONE: Final[str] = "none"
LAYER_TRACE_NODE: Final[str] = "node"
LAYER_TRACE_FULL: Final[str] = "full"

CANONICAL_LAYER_TRACE_MODES: Final[
    tuple[str, ...]
] = (
    LAYER_TRACE_NONE,
    LAYER_TRACE_NODE,
    LAYER_TRACE_FULL,
)

V2_0_IMPLEMENTED_LAYER_TRACE_MODES: Final[
    tuple[str, ...]
] = (
    LAYER_TRACE_NONE,
    LAYER_TRACE_NODE,
    LAYER_TRACE_FULL,
)


# =============================================================================
# Residual vocabulary
# =============================================================================


LAYER_RESIDUAL_DISABLED: Final[str] = "disabled"
LAYER_RESIDUAL_ADDITIVE: Final[str] = "additive"

CANONICAL_LAYER_RESIDUAL_MODES: Final[
    tuple[str, ...]
] = (
    LAYER_RESIDUAL_DISABLED,
    LAYER_RESIDUAL_ADDITIVE,
)

V2_0_IMPLEMENTED_LAYER_RESIDUAL_MODES: Final[
    tuple[str, ...]
] = (
    LAYER_RESIDUAL_DISABLED,
    LAYER_RESIDUAL_ADDITIVE,
)


# =============================================================================
# Normalization vocabulary
# =============================================================================


LAYER_NORMALIZATION_NONE: Final[str] = "none"
LAYER_NORMALIZATION_LAYER_NORM: Final[str] = "layer_norm"

CANONICAL_LAYER_NORMALIZATION_MODES: Final[
    tuple[str, ...]
] = (
    LAYER_NORMALIZATION_NONE,
    LAYER_NORMALIZATION_LAYER_NORM,
)

V2_0_IMPLEMENTED_LAYER_NORMALIZATION_MODES: Final[
    tuple[str, ...]
] = (
    LAYER_NORMALIZATION_NONE,
    LAYER_NORMALIZATION_LAYER_NORM,
)

LAYER_NORMALIZATION_PRE_RESIDUAL: Final[str] = "pre_residual"
LAYER_NORMALIZATION_POST_RESIDUAL: Final[str] = "post_residual"

CANONICAL_LAYER_NORMALIZATION_POSITIONS: Final[
    tuple[str, ...]
] = (
    LAYER_NORMALIZATION_PRE_RESIDUAL,
    LAYER_NORMALIZATION_POST_RESIDUAL,
)

V2_0_IMPLEMENTED_LAYER_NORMALIZATION_POSITIONS: Final[
    tuple[str, ...]
] = (
    LAYER_NORMALIZATION_POST_RESIDUAL,
)


# =============================================================================
# Frozen equations and interpretations
# =============================================================================


LAYER_UPDATE_BRANCH_FORMULA: Final[str] = (
    "post_dropout_update = dropout(pre_dropout_update)"
)

LAYER_ADDITIVE_RESIDUAL_FORMULA: Final[str] = (
    "post_residual_state = residual_source_state + post_dropout_update"
)

LAYER_DISABLED_RESIDUAL_FORMULA: Final[str] = (
    "post_residual_state = post_dropout_update"
)

LAYER_POST_NORMALIZATION_FORMULA: Final[str] = (
    "updated_node_state = normalization(post_residual_state)"
)

LAYER_INPUT_LAYOUT: Final[str] = "node_state_[N,H]"
LAYER_AGGREGATE_LAYOUT: Final[str] = "target_node_aggregate_[N,H]"
LAYER_OUTPUT_LAYOUT: Final[str] = "updated_node_state_[N,H]"

LAYER_SCIENTIFIC_INTERPRETATION: Final[str] = (
    "one_functional_message_passing_state_update"
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
        _canonical_json(payload).encode(
            "utf-8"
        )
    ).hexdigest()


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


def _require_boolean(
    name: str,
    value: bool,
) -> None:
    if not isinstance(value, bool):
        raise TypeError(
            f"{name} must be Boolean."
        )


def _require_probability(
    name: str,
    value: float,
) -> None:
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


def _require_positive_float(
    name: str,
    value: float,
) -> None:
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

    if numeric <= 0.0:
        raise ValueError(
            f"{name} must be strictly positive."
        )


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
            f"{name} must be one of {choices!r}; "
            f"observed {value!r}."
        )


def _require_source_inputs(
    source_inputs: FunctionalMessagePassingInputs,
) -> None:
    if not isinstance(
        source_inputs,
        FunctionalMessagePassingInputs,
    ):
        raise TypeError(
            "source_inputs must be a "
            "FunctionalMessagePassingInputs."
        )


def _require_float_matrix(
    name: str,
    value: torch.Tensor,
    *,
    source_inputs: FunctionalMessagePassingInputs,
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

    expected = (
        source_inputs.num_nodes,
        source_inputs.hidden_dim,
    )
    observed = tuple(
        int(size)
        for size in value.shape
    )

    if observed != expected:
        raise ValueError(
            f"{name} must have shape {expected}; "
            f"observed {observed}."
        )

    if not value.dtype.is_floating_point:
        raise ValueError(
            f"{name} must use a floating-point dtype."
        )

    if value.dtype != source_inputs.dtype:
        raise ValueError(
            f"{name} and source_inputs must share one dtype."
        )

    if value.device != source_inputs.device:
        raise ValueError(
            f"{name} and source_inputs must share one device."
        )

    if not bool(
        torch.isfinite(value).all().item()
    ):
        raise FloatingPointError(
            f"{name} must contain only finite values."
        )


def _require_long_vector(
    name: str,
    value: torch.Tensor,
    *,
    length: int,
    device: torch.device,
) -> None:
    if not isinstance(
        value,
        torch.Tensor,
    ):
        raise TypeError(
            f"{name} must be a tensor."
        )

    if value.dtype != torch.long:
        raise ValueError(
            f"{name} must use torch.long dtype."
        )

    if value.ndim != 1:
        raise ValueError(
            f"{name} must have rank 1."
        )

    if tuple(value.shape) != (
        length,
    ):
        raise ValueError(
            f"{name} must have shape ({length},); "
            f"observed {tuple(value.shape)}."
        )

    if value.device != device:
        raise ValueError(
            f"{name} must share the source-input device."
        )


def _require_exact_tensor_identity(
    *,
    name: str,
    value: torch.Tensor,
    expected: torch.Tensor,
) -> None:
    if value is not expected:
        raise ValueError(
            f"{name} must preserve the exact expected tensor object."
        )


def _immutable_scalar_tensor_mapping(
    name: str,
    values: Mapping[
        str,
        torch.Tensor,
    ],
    *,
    device: torch.device,
) -> Mapping[str, torch.Tensor]:
    if not isinstance(
        values,
        Mapping,
    ):
        raise TypeError(
            f"{name} must be a mapping."
        )

    copied: dict[str, torch.Tensor] = {}

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

        if value.ndim != 0:
            raise ValueError(
                f"{name}[{key!r}] must be a scalar tensor."
            )

        if not value.dtype.is_floating_point:
            raise ValueError(
                f"{name}[{key!r}] must use a floating-point dtype."
            )

        if value.device != device:
            raise ValueError(
                f"{name}[{key!r}] must share the layer-output device."
            )

        if not bool(
            torch.isfinite(value).item()
        ):
            raise FloatingPointError(
                f"{name}[{key!r}] must be finite."
            )

        copied[key] = value

    return MappingProxyType(
        copied
    )


# =============================================================================
# Trace policy
# =============================================================================


@dataclass(slots=True, frozen=True)
class LayerTracePolicy:
    """
    Explicit retention policy for one layer.

    ``none`` retains no optional trace.
    ``node`` retains only node-level stages.
    ``full`` retains the complete edge- and node-level stage chain.
    """

    mode: str = LAYER_TRACE_NONE
    schema_version: str = (
        LAYER_TRACE_POLICY_SCHEMA_VERSION
    )

    def __post_init__(self) -> None:
        _require_choice(
            "mode",
            self.mode,
            CANONICAL_LAYER_TRACE_MODES,
        )
        _require_nonempty_string(
            "schema_version",
            self.schema_version,
        )

    @classmethod
    def from_capture_intermediate_messages(
        cls,
        capture: bool,
    ) -> "LayerTracePolicy":
        _require_boolean(
            "capture",
            capture,
        )

        return cls(
            mode=(
                LAYER_TRACE_FULL
                if capture
                else LAYER_TRACE_NONE
            )
        )

    def assert_implemented(self) -> None:
        if self.mode not in (
            V2_0_IMPLEMENTED_LAYER_TRACE_MODES
        ):
            raise NotImplementedError(
                f"Layer trace mode {self.mode!r} is canonical but not "
                "implemented in bounded V2.0."
            )

    @property
    def enabled(self) -> bool:
        return self.mode != LAYER_TRACE_NONE

    @property
    def retain_node_stages(self) -> bool:
        return self.mode in (
            LAYER_TRACE_NODE,
            LAYER_TRACE_FULL,
        )

    @property
    def retain_edge_stages(self) -> bool:
        return self.mode == LAYER_TRACE_FULL

    @property
    def capture_intermediate_messages(
        self,
    ) -> bool:
        return self.mode == LAYER_TRACE_FULL

    def architecture_dict(
        self,
    ) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "mode": self.mode,
            "retain_node_stages": (
                self.retain_node_stages
            ),
            "retain_edge_stages": (
                self.retain_edge_stages
            ),
        }

    def architecture_fingerprint(
        self,
    ) -> str:
        return _fingerprint(
            self.architecture_dict()
        )


# =============================================================================
# Layer inputs
# =============================================================================


@dataclass(slots=True, frozen=True)
class FunctionalMessagePassingLayerInputs:
    """
    Immutable one-layer execution identity.

    The actual graph, node-state, hazard, and relation metadata remain owned by
    ``source_inputs``. This wrapper adds only layer-local execution identity.
    """

    source_inputs: FunctionalMessagePassingInputs
    layer_index: int
    trace_policy: LayerTracePolicy = field(
        default_factory=LayerTracePolicy
    )
    training: bool = True
    source_stack_fingerprint: str | None = None
    schema_version: str = (
        LAYER_INPUTS_SCHEMA_VERSION
    )

    def __post_init__(self) -> None:
        _require_source_inputs(
            self.source_inputs
        )
        _require_nonnegative_int(
            "layer_index",
            self.layer_index,
        )

        if not isinstance(
            self.trace_policy,
            LayerTracePolicy,
        ):
            raise TypeError(
                "trace_policy must be a LayerTracePolicy."
            )

        self.trace_policy.assert_implemented()

        _require_boolean(
            "training",
            self.training,
        )

        if self.source_stack_fingerprint is not None:
            _require_nonempty_string(
                "source_stack_fingerprint",
                self.source_stack_fingerprint,
            )

        _require_nonempty_string(
            "schema_version",
            self.schema_version,
        )

    @property
    def num_nodes(self) -> int:
        return self.source_inputs.num_nodes

    @property
    def num_edges(self) -> int:
        return self.source_inputs.num_edges

    @property
    def hidden_dim(self) -> int:
        return self.source_inputs.hidden_dim

    @property
    def dtype(self) -> torch.dtype:
        return self.source_inputs.dtype

    @property
    def device(self) -> torch.device:
        return self.source_inputs.device

    @property
    def input_node_state(
        self,
    ) -> torch.Tensor:
        return (
            self
            .source_inputs
            .node_state
            .fused_state
        )

    def lineage_dict(
        self,
    ) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "layer_index": self.layer_index,
            "training": self.training,
            "trace_policy": (
                self.trace_policy
                .architecture_dict()
            ),
            "source_inputs_lineage_fingerprint": (
                self.source_inputs
                .lineage_fingerprint()
            ),
            "source_stack_fingerprint": (
                self.source_stack_fingerprint
            ),
        }

    def lineage_fingerprint(
        self,
    ) -> str:
        return _fingerprint(
            self.lineage_dict()
        )


# =============================================================================
# Residual update output
# =============================================================================


@dataclass(slots=True, frozen=True)
class LayerResidualUpdateOutput:
    """
    Complete node-level residual-update trace.

    ``pre_dropout_update`` is the deterministic update branch entering
    dropout. ``post_dropout_update`` is the realized update branch after
    dropout. ``pre_residual_state`` is an explicit alias of the realized
    update branch. ``post_residual_state`` is the output of the residual
    operation.
    """

    pre_dropout_update: torch.Tensor
    post_dropout_update: torch.Tensor
    pre_residual_state: torch.Tensor
    post_residual_state: torch.Tensor
    residual_source_state: torch.Tensor

    aggregation: AggregationOutput
    layer_inputs: FunctionalMessagePassingLayerInputs

    residual_mode: str
    dropout_probability: float
    training: bool

    updater_architecture_fingerprint: str
    updater_parameter_fingerprint: str | None = None

    schema_version: str = (
        LAYER_RESIDUAL_UPDATE_SCHEMA_VERSION
    )

    def __post_init__(self) -> None:
        if not isinstance(
            self.aggregation,
            AggregationOutput,
        ):
            raise TypeError(
                "aggregation must be an AggregationOutput."
            )

        if not isinstance(
            self.layer_inputs,
            FunctionalMessagePassingLayerInputs,
        ):
            raise TypeError(
                "layer_inputs must be a "
                "FunctionalMessagePassingLayerInputs."
            )

        source_inputs = (
            self.layer_inputs.source_inputs
        )

        if (
            self
            .aggregation
            .source_messages
            .source_inputs
            is not source_inputs
        ):
            raise ValueError(
                "aggregation and layer_inputs must share the exact same "
                "FunctionalMessagePassingInputs object."
            )

        for name, state in (
            (
                "pre_dropout_update",
                self.pre_dropout_update,
            ),
            (
                "post_dropout_update",
                self.post_dropout_update,
            ),
            (
                "pre_residual_state",
                self.pre_residual_state,
            ),
            (
                "post_residual_state",
                self.post_residual_state,
            ),
            (
                "residual_source_state",
                self.residual_source_state,
            ),
        ):
            _require_float_matrix(
                name,
                state,
                source_inputs=source_inputs,
            )

        _require_exact_tensor_identity(
            name="pre_dropout_update",
            value=self.pre_dropout_update,
            expected=(
                self.aggregation.node_aggregate
            ),
        )
        _require_exact_tensor_identity(
            name="pre_residual_state",
            value=self.pre_residual_state,
            expected=self.post_dropout_update,
        )
        _require_exact_tensor_identity(
            name="residual_source_state",
            value=self.residual_source_state,
            expected=(
                self.layer_inputs
                .input_node_state
            ),
        )

        _require_choice(
            "residual_mode",
            self.residual_mode,
            CANONICAL_LAYER_RESIDUAL_MODES,
        )
        _require_probability(
            "dropout_probability",
            self.dropout_probability,
        )
        _require_boolean(
            "training",
            self.training,
        )

        if self.training != (
            self.layer_inputs.training
        ):
            raise ValueError(
                "Residual-update training flag differs from layer_inputs."
            )

        atol, rtol = _default_tolerances(
            self.post_residual_state.dtype
        )

        if self.residual_mode == (
            LAYER_RESIDUAL_ADDITIVE
        ):
            expected = (
                self.residual_source_state
                + self.post_dropout_update
            )
        elif self.residual_mode == (
            LAYER_RESIDUAL_DISABLED
        ):
            expected = self.post_dropout_update
        else:
            raise RuntimeError(
                "Unreachable residual-mode branch."
            )

        if not torch.allclose(
            self.post_residual_state,
            expected,
            atol=atol,
            rtol=rtol,
        ):
            raise ValueError(
                "post_residual_state does not match the declared residual "
                "equation."
            )

        if (
            self.dropout_probability == 0.0
            or not self.training
        ):
            _require_exact_tensor_identity(
                name="post_dropout_update",
                value=self.post_dropout_update,
                expected=self.pre_dropout_update,
            )

        _require_nonempty_string(
            "updater_architecture_fingerprint",
            self.updater_architecture_fingerprint,
        )

        if (
            self.updater_parameter_fingerprint
            is not None
        ):
            _require_nonempty_string(
                "updater_parameter_fingerprint",
                self.updater_parameter_fingerprint,
            )

        _require_nonempty_string(
            "schema_version",
            self.schema_version,
        )

    @property
    def source_inputs(
        self,
    ) -> FunctionalMessagePassingInputs:
        return self.layer_inputs.source_inputs

    @property
    def residual_enabled(self) -> bool:
        return self.residual_mode == (
            LAYER_RESIDUAL_ADDITIVE
        )

    @property
    def num_nodes(self) -> int:
        return self.source_inputs.num_nodes

    @property
    def hidden_dim(self) -> int:
        return self.source_inputs.hidden_dim

    @property
    def dtype(self) -> torch.dtype:
        return self.post_residual_state.dtype

    @property
    def device(self) -> torch.device:
        return self.post_residual_state.device

    def architecture_dict(
        self,
    ) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "residual_mode": self.residual_mode,
            "dropout_probability": float(
                self.dropout_probability
            ),
            "training": self.training,
            "update_branch_formula": (
                LAYER_UPDATE_BRANCH_FORMULA
            ),
            "residual_formula": (
                LAYER_ADDITIVE_RESIDUAL_FORMULA
                if self.residual_enabled
                else LAYER_DISABLED_RESIDUAL_FORMULA
            ),
            "updater_architecture_fingerprint": (
                self.updater_architecture_fingerprint
            ),
            "updater_parameter_fingerprint": (
                self.updater_parameter_fingerprint
            ),
        }

    def lineage_fingerprint(
        self,
    ) -> str:
        return _fingerprint(
            {
                "layer_inputs": (
                    self.layer_inputs
                    .lineage_fingerprint()
                ),
                "aggregation_architecture_fingerprint": (
                    self
                    .aggregation
                    .encoder_architecture_fingerprint
                ),
                "architecture": (
                    self.architecture_dict()
                ),
            }
        )


# =============================================================================
# Normalization output
# =============================================================================


@dataclass(slots=True, frozen=True)
class LayerNormalizationOutput:
    """
    Explicit normalization stage following the residual update.
    """

    input_state: torch.Tensor
    output_state: torch.Tensor

    residual_update: LayerResidualUpdateOutput

    normalization_mode: str
    normalization_position: str
    epsilon: float

    normalizer_architecture_fingerprint: str
    normalizer_parameter_fingerprint: str | None = None

    schema_version: str = (
        LAYER_NORMALIZATION_OUTPUT_SCHEMA_VERSION
    )

    def __post_init__(self) -> None:
        if not isinstance(
            self.residual_update,
            LayerResidualUpdateOutput,
        ):
            raise TypeError(
                "residual_update must be a "
                "LayerResidualUpdateOutput."
            )

        source_inputs = (
            self.residual_update.source_inputs
        )

        _require_float_matrix(
            "input_state",
            self.input_state,
            source_inputs=source_inputs,
        )
        _require_float_matrix(
            "output_state",
            self.output_state,
            source_inputs=source_inputs,
        )

        _require_exact_tensor_identity(
            name="input_state",
            value=self.input_state,
            expected=(
                self
                .residual_update
                .post_residual_state
            ),
        )

        _require_choice(
            "normalization_mode",
            self.normalization_mode,
            CANONICAL_LAYER_NORMALIZATION_MODES,
        )
        _require_choice(
            "normalization_position",
            self.normalization_position,
            CANONICAL_LAYER_NORMALIZATION_POSITIONS,
        )

        if self.normalization_position != (
            LAYER_NORMALIZATION_POST_RESIDUAL
        ):
            raise ValueError(
                "The current layer output schema consumes the post-residual "
                "state and therefore requires post_residual normalization."
            )

        _require_positive_float(
            "epsilon",
            self.epsilon,
        )

        if self.normalization_mode == (
            LAYER_NORMALIZATION_NONE
        ):
            _require_exact_tensor_identity(
                name="output_state",
                value=self.output_state,
                expected=self.input_state,
            )

            if (
                self.normalizer_parameter_fingerprint
                is not None
            ):
                raise ValueError(
                    "Disabled normalization must not retain a parameter "
                    "fingerprint."
                )

        _require_nonempty_string(
            "normalizer_architecture_fingerprint",
            self.normalizer_architecture_fingerprint,
        )

        if (
            self.normalizer_parameter_fingerprint
            is not None
        ):
            _require_nonempty_string(
                "normalizer_parameter_fingerprint",
                self.normalizer_parameter_fingerprint,
            )

        _require_nonempty_string(
            "schema_version",
            self.schema_version,
        )

    @property
    def source_inputs(
        self,
    ) -> FunctionalMessagePassingInputs:
        return (
            self
            .residual_update
            .source_inputs
        )

    @property
    def layer_inputs(
        self,
    ) -> FunctionalMessagePassingLayerInputs:
        return (
            self
            .residual_update
            .layer_inputs
        )

    @property
    def normalization_enabled(
        self,
    ) -> bool:
        return self.normalization_mode != (
            LAYER_NORMALIZATION_NONE
        )

    @property
    def num_nodes(self) -> int:
        return self.source_inputs.num_nodes

    @property
    def hidden_dim(self) -> int:
        return self.source_inputs.hidden_dim

    @property
    def dtype(self) -> torch.dtype:
        return self.output_state.dtype

    @property
    def device(self) -> torch.device:
        return self.output_state.device

    def architecture_dict(
        self,
    ) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "normalization_mode": (
                self.normalization_mode
            ),
            "normalization_position": (
                self.normalization_position
            ),
            "epsilon": float(self.epsilon),
            "formula": (
                LAYER_POST_NORMALIZATION_FORMULA
            ),
            "normalizer_architecture_fingerprint": (
                self.normalizer_architecture_fingerprint
            ),
            "normalizer_parameter_fingerprint": (
                self.normalizer_parameter_fingerprint
            ),
        }

    def lineage_fingerprint(
        self,
    ) -> str:
        return _fingerprint(
            {
                "residual_update": (
                    self
                    .residual_update
                    .lineage_fingerprint()
                ),
                "architecture": (
                    self.architecture_dict()
                ),
            }
        )


# =============================================================================
# Optional retained trace
# =============================================================================


@dataclass(slots=True, frozen=True)
class FunctionalMessagePassingLayerTrace:
    """
    Optional exact one-layer trace controlled by ``LayerTracePolicy``.

    Node-level fields are always required whenever a trace object exists.
    Edge-level fields are required only for ``full`` tracing and forbidden for
    ``node`` tracing.
    """

    layer_inputs: FunctionalMessagePassingLayerInputs

    aggregation: AggregationOutput
    residual_update: LayerResidualUpdateOutput
    normalization: LayerNormalizationOutput

    relation_transform: RelationTransformOutput | None = None
    edge_normalization: (
        StructuralEdgeNormalizationOutput
        | None
    ) = None
    relation_gate: RelationGateOutput | None = None
    edge_attention: EdgeAttentionOutput | None = None
    edge_messages: EdgeMessageOutput | None = None
    message_builder_run: MessageBuilderRun | None = None

    schema_version: str = (
        LAYER_INTERMEDIATE_TRACE_SCHEMA_VERSION
    )

    def __post_init__(self) -> None:
        if not isinstance(
            self.layer_inputs,
            FunctionalMessagePassingLayerInputs,
        ):
            raise TypeError(
                "layer_inputs must be a "
                "FunctionalMessagePassingLayerInputs."
            )

        policy = self.layer_inputs.trace_policy

        if not policy.retain_node_stages:
            raise ValueError(
                "A layer trace object is forbidden when trace policy is "
                "'none'."
            )

        if not isinstance(
            self.aggregation,
            AggregationOutput,
        ):
            raise TypeError(
                "aggregation must be an AggregationOutput."
            )

        if not isinstance(
            self.residual_update,
            LayerResidualUpdateOutput,
        ):
            raise TypeError(
                "residual_update must be a "
                "LayerResidualUpdateOutput."
            )

        if not isinstance(
            self.normalization,
            LayerNormalizationOutput,
        ):
            raise TypeError(
                "normalization must be a "
                "LayerNormalizationOutput."
            )

        source_inputs = (
            self.layer_inputs.source_inputs
        )

        if (
            self
            .aggregation
            .source_messages
            .source_inputs
            is not source_inputs
        ):
            raise ValueError(
                "aggregation uses different source inputs."
            )

        if (
            self.residual_update.layer_inputs
            is not self.layer_inputs
        ):
            raise ValueError(
                "residual_update must preserve the exact layer_inputs "
                "object."
            )

        if (
            self.residual_update.aggregation
            is not self.aggregation
        ):
            raise ValueError(
                "residual_update must consume the exact retained "
                "aggregation object."
            )

        if (
            self.normalization.residual_update
            is not self.residual_update
        ):
            raise ValueError(
                "normalization must consume the exact retained "
                "residual_update object."
            )

        edge_values = (
            self.relation_transform,
            self.edge_normalization,
            self.edge_messages,
            self.message_builder_run,
        )

        if policy.mode == LAYER_TRACE_NODE:
            if any(
                value is not None
                for value in edge_values
            ):
                raise ValueError(
                    "Node-only layer traces must not retain edge-level "
                    "transform, normalization, message, or builder objects."
                )

            if self.relation_gate is not None:
                raise ValueError(
                    "Node-only layer traces must not retain relation gates."
                )

            if self.edge_attention is not None:
                raise ValueError(
                    "Node-only layer traces must not retain edge attention."
                )

        elif policy.mode == LAYER_TRACE_FULL:
            if not isinstance(
                self.relation_transform,
                RelationTransformOutput,
            ):
                raise TypeError(
                    "Full traces require relation_transform."
                )

            if not isinstance(
                self.edge_normalization,
                StructuralEdgeNormalizationOutput,
            ):
                raise TypeError(
                    "Full traces require edge_normalization."
                )

            if not isinstance(
                self.edge_messages,
                EdgeMessageOutput,
            ):
                raise TypeError(
                    "Full traces require edge_messages."
                )

            if not isinstance(
                self.message_builder_run,
                MessageBuilderRun,
            ):
                raise TypeError(
                    "Full traces require message_builder_run."
                )

            if (
                self.relation_transform.source_inputs
                is not source_inputs
            ):
                raise ValueError(
                    "relation_transform uses different source inputs."
                )

            if (
                self.edge_normalization.source_inputs
                is not source_inputs
            ):
                raise ValueError(
                    "edge_normalization uses different source inputs."
                )

            if (
                self.edge_messages.source_inputs
                is not source_inputs
            ):
                raise ValueError(
                    "edge_messages uses different source inputs."
                )

            if (
                self.aggregation.source_messages
                is not self.edge_messages
            ):
                raise ValueError(
                    "aggregation must consume the exact retained "
                    "edge_messages object."
                )

            if (
                self.message_builder_run.public_output
                is not self.edge_messages
            ):
                raise ValueError(
                    "message_builder_run must preserve the exact retained "
                    "edge_messages public output."
                )

            if (
                self.message_builder_run
                .composition_output
                .relation_transform
                is not self.relation_transform
            ):
                raise ValueError(
                    "message_builder_run and trace must preserve the exact "
                    "relation_transform object."
                )

            if (
                self.message_builder_run
                .resolved_coefficients
                .edge_normalization
                is not self.edge_normalization
            ):
                raise ValueError(
                    "message_builder_run and trace must preserve the exact "
                    "edge_normalization object."
                )

            expected_gate = (
                self
                .message_builder_run
                .resolved_coefficients
                .relation_gate
            )
            expected_attention = (
                self
                .message_builder_run
                .resolved_coefficients
                .edge_attention
            )

            if self.relation_gate is not (
                expected_gate
            ):
                raise ValueError(
                    "Trace relation_gate differs from the exact "
                    "message-builder source."
                )

            if self.edge_attention is not (
                expected_attention
            ):
                raise ValueError(
                    "Trace edge_attention differs from the exact "
                    "message-builder source."
                )

        else:
            raise RuntimeError(
                "Unreachable trace-policy branch."
            )

        _require_nonempty_string(
            "schema_version",
            self.schema_version,
        )

    @property
    def source_inputs(
        self,
    ) -> FunctionalMessagePassingInputs:
        return self.layer_inputs.source_inputs

    @property
    def trace_mode(self) -> str:
        return self.layer_inputs.trace_policy.mode

    @property
    def updated_node_state(
        self,
    ) -> torch.Tensor:
        return self.normalization.output_state

    def lineage_fingerprint(
        self,
    ) -> str:
        payload: dict[str, Any] = {
            "schema_version": self.schema_version,
            "layer_inputs": (
                self.layer_inputs
                .lineage_fingerprint()
            ),
            "trace_mode": self.trace_mode,
            "aggregation_architecture_fingerprint": (
                self
                .aggregation
                .encoder_architecture_fingerprint
            ),
            "residual_update": (
                self
                .residual_update
                .lineage_fingerprint()
            ),
            "normalization": (
                self
                .normalization
                .lineage_fingerprint()
            ),
        }

        if self.message_builder_run is not None:
            payload["message_builder"] = {
                "architecture_fingerprint": (
                    self
                    .message_builder_run
                    .public_output
                    .encoder_architecture_fingerprint
                ),
                "source_inputs_lineage_fingerprint": (
                    self
                    .message_builder_run
                    .public_output
                    .source_inputs
                    .lineage_fingerprint()
                ),
            }

        return _fingerprint(
            payload
        )


# =============================================================================
# Complete internal layer output
# =============================================================================


@dataclass(slots=True, frozen=True)
class LayerComputationOutput:
    """
    Complete immutable internal result of one functional message-passing layer.
    """

    updated_node_state: torch.Tensor

    layer_inputs: FunctionalMessagePassingLayerInputs
    aggregation: AggregationOutput
    residual_update: LayerResidualUpdateOutput
    normalization: LayerNormalizationOutput

    layer_architecture_fingerprint: str
    layer_parameter_fingerprint: str | None
    lineage_fingerprint: str

    trace: (
        FunctionalMessagePassingLayerTrace
        | None
    ) = None

    regularization_terms: Mapping[
        str,
        torch.Tensor,
    ] = field(default_factory=dict)

    schema_version: str = (
        LAYER_COMPUTATION_OUTPUT_SCHEMA_VERSION
    )

    def __post_init__(self) -> None:
        if not isinstance(
            self.layer_inputs,
            FunctionalMessagePassingLayerInputs,
        ):
            raise TypeError(
                "layer_inputs must be a "
                "FunctionalMessagePassingLayerInputs."
            )

        if not isinstance(
            self.aggregation,
            AggregationOutput,
        ):
            raise TypeError(
                "aggregation must be an AggregationOutput."
            )

        if not isinstance(
            self.residual_update,
            LayerResidualUpdateOutput,
        ):
            raise TypeError(
                "residual_update must be a "
                "LayerResidualUpdateOutput."
            )

        if not isinstance(
            self.normalization,
            LayerNormalizationOutput,
        ):
            raise TypeError(
                "normalization must be a "
                "LayerNormalizationOutput."
            )

        source_inputs = (
            self.layer_inputs.source_inputs
        )

        _require_float_matrix(
            "updated_node_state",
            self.updated_node_state,
            source_inputs=source_inputs,
        )

        _require_exact_tensor_identity(
            name="updated_node_state",
            value=self.updated_node_state,
            expected=self.normalization.output_state,
        )

        if (
            self
            .aggregation
            .source_messages
            .source_inputs
            is not source_inputs
        ):
            raise ValueError(
                "aggregation and layer_inputs use different source inputs."
            )

        if (
            self.residual_update.layer_inputs
            is not self.layer_inputs
        ):
            raise ValueError(
                "residual_update must preserve the exact layer_inputs "
                "object."
            )

        if (
            self.residual_update.aggregation
            is not self.aggregation
        ):
            raise ValueError(
                "residual_update must consume the exact aggregation object."
            )

        if (
            self.normalization.residual_update
            is not self.residual_update
        ):
            raise ValueError(
                "normalization must consume the exact residual_update "
                "object."
            )

        policy = self.layer_inputs.trace_policy

        if policy.mode == LAYER_TRACE_NONE:
            if self.trace is not None:
                raise ValueError(
                    "Trace policy 'none' requires trace=None."
                )
        else:
            if not isinstance(
                self.trace,
                FunctionalMessagePassingLayerTrace,
            ):
                raise TypeError(
                    "Enabled tracing requires a "
                    "FunctionalMessagePassingLayerTrace."
                )

            if self.trace.layer_inputs is not (
                self.layer_inputs
            ):
                raise ValueError(
                    "trace must preserve the exact layer_inputs object."
                )

            if self.trace.aggregation is not (
                self.aggregation
            ):
                raise ValueError(
                    "trace must preserve the exact aggregation object."
                )

            if self.trace.residual_update is not (
                self.residual_update
            ):
                raise ValueError(
                    "trace must preserve the exact residual_update object."
                )

            if self.trace.normalization is not (
                self.normalization
            ):
                raise ValueError(
                    "trace must preserve the exact normalization object."
                )

        _require_nonempty_string(
            "layer_architecture_fingerprint",
            self.layer_architecture_fingerprint,
        )

        if self.layer_parameter_fingerprint is not None:
            _require_nonempty_string(
                "layer_parameter_fingerprint",
                self.layer_parameter_fingerprint,
            )

        _require_nonempty_string(
            "lineage_fingerprint",
            self.lineage_fingerprint,
        )
        _require_nonempty_string(
            "schema_version",
            self.schema_version,
        )

        object.__setattr__(
            self,
            "regularization_terms",
            _immutable_scalar_tensor_mapping(
                "regularization_terms",
                self.regularization_terms,
                device=self.updated_node_state.device,
            ),
        )

    @property
    def source_inputs(
        self,
    ) -> FunctionalMessagePassingInputs:
        return self.layer_inputs.source_inputs

    @property
    def layer_index(self) -> int:
        return self.layer_inputs.layer_index

    @property
    def trace_policy(
        self,
    ) -> LayerTracePolicy:
        return self.layer_inputs.trace_policy

    @property
    def node_aggregate(
        self,
    ) -> torch.Tensor:
        return self.aggregation.node_aggregate

    @property
    def incoming_edge_count(
        self,
    ) -> torch.Tensor:
        return self.aggregation.incoming_edge_count

    @property
    def residual_enabled(self) -> bool:
        return self.residual_update.residual_enabled

    @property
    def layer_norm_enabled(self) -> bool:
        return (
            self.normalization
            .normalization_enabled
        )

    @property
    def num_nodes(self) -> int:
        return self.source_inputs.num_nodes

    @property
    def hidden_dim(self) -> int:
        return self.source_inputs.hidden_dim

    @property
    def dtype(self) -> torch.dtype:
        return self.updated_node_state.dtype

    @property
    def device(self) -> torch.device:
        return self.updated_node_state.device

    def architecture_dict(
        self,
    ) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "scientific_interpretation": (
                LAYER_SCIENTIFIC_INTERPRETATION
            ),
            "layer_index": self.layer_index,
            "trace_policy": (
                self.trace_policy
                .architecture_dict()
            ),
            "aggregation_mode": (
                self.aggregation.aggregation_mode
            ),
            "residual_update": (
                self.residual_update
                .architecture_dict()
            ),
            "normalization": (
                self.normalization
                .architecture_dict()
            ),
            "layer_architecture_fingerprint": (
                self.layer_architecture_fingerprint
            ),
            "layer_parameter_fingerprint": (
                self.layer_parameter_fingerprint
            ),
            "input_layout": LAYER_INPUT_LAYOUT,
            "aggregate_layout": (
                LAYER_AGGREGATE_LAYOUT
            ),
            "output_layout": LAYER_OUTPUT_LAYOUT,
        }

    def value_lineage_dict(
        self,
    ) -> dict[str, Any]:
        return {
            "source_inputs_lineage_fingerprint": (
                self.source_inputs
                .lineage_fingerprint()
            ),
            "layer_inputs_lineage_fingerprint": (
                self.layer_inputs
                .lineage_fingerprint()
            ),
            "residual_update_lineage_fingerprint": (
                self.residual_update
                .lineage_fingerprint()
            ),
            "normalization_lineage_fingerprint": (
                self.normalization
                .lineage_fingerprint()
            ),
            "trace_lineage_fingerprint": (
                self.trace.lineage_fingerprint()
                if self.trace is not None
                else None
            ),
            "declared_lineage_fingerprint": (
                self.lineage_fingerprint
            ),
        }


# =============================================================================
# Layer stage tuple
# =============================================================================


class FunctionalMessagePassingLayerStages(
    NamedTuple
):
    """
    Exact node-level stage chain before public output assembly.
    """

    aggregation: AggregationOutput
    residual_update: LayerResidualUpdateOutput
    normalization: LayerNormalizationOutput
    computation_output: LayerComputationOutput


# =============================================================================
# Stage validators
# =============================================================================


def validate_layer_stage_chain(
    *,
    layer_inputs: FunctionalMessagePassingLayerInputs,
    aggregation: AggregationOutput,
    residual_update: LayerResidualUpdateOutput,
    normalization: LayerNormalizationOutput,
    computation_output: LayerComputationOutput | None = None,
) -> None:
    """
    Validate exact lineage across aggregation, residual, and normalization.
    """

    if not isinstance(
        layer_inputs,
        FunctionalMessagePassingLayerInputs,
    ):
        raise TypeError(
            "layer_inputs must be a "
            "FunctionalMessagePassingLayerInputs."
        )

    if not isinstance(
        aggregation,
        AggregationOutput,
    ):
        raise TypeError(
            "aggregation must be an AggregationOutput."
        )

    if not isinstance(
        residual_update,
        LayerResidualUpdateOutput,
    ):
        raise TypeError(
            "residual_update must be a "
            "LayerResidualUpdateOutput."
        )

    if not isinstance(
        normalization,
        LayerNormalizationOutput,
    ):
        raise TypeError(
            "normalization must be a "
            "LayerNormalizationOutput."
        )

    source_inputs = layer_inputs.source_inputs

    if (
        aggregation
        .source_messages
        .source_inputs
        is not source_inputs
    ):
        raise ValueError(
            "aggregation and layer_inputs must share exact source inputs."
        )

    if residual_update.layer_inputs is not (
        layer_inputs
    ):
        raise ValueError(
            "residual_update must preserve exact layer_inputs."
        )

    if residual_update.aggregation is not (
        aggregation
    ):
        raise ValueError(
            "residual_update must consume exact aggregation."
        )

    if normalization.residual_update is not (
        residual_update
    ):
        raise ValueError(
            "normalization must consume exact residual_update."
        )

    if computation_output is not None:
        if not isinstance(
            computation_output,
            LayerComputationOutput,
        ):
            raise TypeError(
                "computation_output must be a "
                "LayerComputationOutput."
            )

        if computation_output.layer_inputs is not (
            layer_inputs
        ):
            raise ValueError(
                "computation_output must preserve exact layer_inputs."
            )

        if computation_output.aggregation is not (
            aggregation
        ):
            raise ValueError(
                "computation_output must preserve exact aggregation."
            )

        if computation_output.residual_update is not (
            residual_update
        ):
            raise ValueError(
                "computation_output must preserve exact residual_update."
            )

        if computation_output.normalization is not (
            normalization
        ):
            raise ValueError(
                "computation_output must preserve exact normalization."
            )


def build_public_layer_intermediates(
    trace: FunctionalMessagePassingLayerTrace,
) -> FunctionalMessagePassingIntermediates:
    """
    Convert a full internal trace into the existing public intermediate schema.

    Node-only traces cannot be represented by the historical public schema
    because that schema requires all edge-level objects.
    """

    if not isinstance(
        trace,
        FunctionalMessagePassingLayerTrace,
    ):
        raise TypeError(
            "trace must be a "
            "FunctionalMessagePassingLayerTrace."
        )

    if trace.trace_mode != LAYER_TRACE_FULL:
        raise ValueError(
            "Only full traces can be converted to "
            "FunctionalMessagePassingIntermediates."
        )

    assert trace.relation_transform is not None
    assert trace.edge_normalization is not None
    assert trace.edge_messages is not None

    return FunctionalMessagePassingIntermediates(
        relation_transform=(
            trace.relation_transform
        ),
        edge_normalization=(
            trace.edge_normalization
        ),
        relation_gate=trace.relation_gate,
        edge_attention=trace.edge_attention,
        edge_messages=trace.edge_messages,
        aggregation=trace.aggregation,
        pre_residual_state=(
            trace
            .residual_update
            .pre_residual_state
        ),
        post_residual_state=(
            trace
            .residual_update
            .post_residual_state
        ),
    )


def validate_public_layer_output(
    *,
    public_output: FunctionalMessagePassingLayerOutput,
    internal_output: LayerComputationOutput,
) -> None:
    """
    Validate exact compatibility with the existing public layer schema.
    """

    if not isinstance(
        public_output,
        FunctionalMessagePassingLayerOutput,
    ):
        raise TypeError(
            "public_output must be a "
            "FunctionalMessagePassingLayerOutput."
        )

    if not isinstance(
        internal_output,
        LayerComputationOutput,
    ):
        raise TypeError(
            "internal_output must be a "
            "LayerComputationOutput."
        )

    _require_exact_tensor_identity(
        name="public_output.updated_node_state",
        value=public_output.updated_node_state,
        expected=(
            internal_output
            .updated_node_state
        ),
    )
    _require_exact_tensor_identity(
        name="public_output.node_aggregate",
        value=public_output.node_aggregate,
        expected=(
            internal_output.node_aggregate
        ),
    )
    _require_exact_tensor_identity(
        name="public_output.incoming_edge_count",
        value=public_output.incoming_edge_count,
        expected=(
            internal_output
            .incoming_edge_count
        ),
    )

    if public_output.source_inputs is not (
        internal_output.source_inputs
    ):
        raise ValueError(
            "public_output must preserve exact source_inputs."
        )

    if public_output.layer_index != (
        internal_output.layer_index
    ):
        raise ValueError(
            "public_output layer_index differs from internal output."
        )

    if public_output.residual_enabled is not (
        internal_output.residual_enabled
    ):
        raise ValueError(
            "public_output residual flag differs from internal output."
        )

    if public_output.layer_norm_enabled is not (
        internal_output.layer_norm_enabled
    ):
        raise ValueError(
            "public_output layer-norm flag differs from internal output."
        )

    if (
        public_output
        .encoder_architecture_fingerprint
        != internal_output
        .layer_architecture_fingerprint
    ):
        raise ValueError(
            "public_output architecture fingerprint differs from internal "
            "output."
        )

    if public_output.lineage_fingerprint != (
        internal_output.lineage_fingerprint
    ):
        raise ValueError(
            "public_output lineage fingerprint differs from internal output."
        )

    if dict(
        public_output.regularization_terms
    ) != dict(
        internal_output.regularization_terms
    ):
        raise ValueError(
            "public_output regularization terms differ from internal output."
        )

    if internal_output.trace_policy.mode == (
        LAYER_TRACE_FULL
    ):
        if public_output.intermediates is None:
            raise ValueError(
                "Full internal tracing requires public intermediates."
            )

        assert internal_output.trace is not None

        if (
            public_output
            .intermediates
            .relation_transform
            is not internal_output
            .trace
            .relation_transform
        ):
            raise ValueError(
                "Public intermediates lost exact relation-transform lineage."
            )

        if (
            public_output
            .intermediates
            .aggregation
            is not internal_output
            .aggregation
        ):
            raise ValueError(
                "Public intermediates lost exact aggregation lineage."
            )
    else:
        if public_output.intermediates is not None:
            raise ValueError(
                "Non-full traces cannot populate the historical public "
                "intermediates field."
            )


# =============================================================================
# Architecture helpers
# =============================================================================


def layer_schema_architecture_dict() -> dict[str, Any]:
    """
    Return static schema-level boundaries.
    """

    return {
        "layer_inputs_schema_version": (
            LAYER_INPUTS_SCHEMA_VERSION
        ),
        "trace_policy_schema_version": (
            LAYER_TRACE_POLICY_SCHEMA_VERSION
        ),
        "residual_update_schema_version": (
            LAYER_RESIDUAL_UPDATE_SCHEMA_VERSION
        ),
        "normalization_output_schema_version": (
            LAYER_NORMALIZATION_OUTPUT_SCHEMA_VERSION
        ),
        "intermediate_trace_schema_version": (
            LAYER_INTERMEDIATE_TRACE_SCHEMA_VERSION
        ),
        "computation_output_schema_version": (
            LAYER_COMPUTATION_OUTPUT_SCHEMA_VERSION
        ),
        "scientific_interpretation": (
            LAYER_SCIENTIFIC_INTERPRETATION
        ),
        "input_layout": LAYER_INPUT_LAYOUT,
        "aggregate_layout": (
            LAYER_AGGREGATE_LAYOUT
        ),
        "output_layout": LAYER_OUTPUT_LAYOUT,
        "canonical_trace_modes": list(
            CANONICAL_LAYER_TRACE_MODES
        ),
        "implemented_trace_modes": list(
            V2_0_IMPLEMENTED_LAYER_TRACE_MODES
        ),
        "canonical_residual_modes": list(
            CANONICAL_LAYER_RESIDUAL_MODES
        ),
        "implemented_residual_modes": list(
            V2_0_IMPLEMENTED_LAYER_RESIDUAL_MODES
        ),
        "canonical_normalization_modes": list(
            CANONICAL_LAYER_NORMALIZATION_MODES
        ),
        "implemented_normalization_modes": list(
            V2_0_IMPLEMENTED_LAYER_NORMALIZATION_MODES
        ),
        "canonical_normalization_positions": list(
            CANONICAL_LAYER_NORMALIZATION_POSITIONS
        ),
        "implemented_normalization_positions": list(
            V2_0_IMPLEMENTED_LAYER_NORMALIZATION_POSITIONS
        ),
        "update_branch_formula": (
            LAYER_UPDATE_BRANCH_FORMULA
        ),
        "additive_residual_formula": (
            LAYER_ADDITIVE_RESIDUAL_FORMULA
        ),
        "disabled_residual_formula": (
            LAYER_DISABLED_RESIDUAL_FORMULA
        ),
        "post_normalization_formula": (
            LAYER_POST_NORMALIZATION_FORMULA
        ),
        "multi_layer_iteration_owned_here": False,
        "prediction_owned_here": False,
        "claims_causal_importance": False,
        "claims_explanation_faithfulness": False,
    }


def layer_schema_architecture_fingerprint() -> str:
    return _fingerprint(
        layer_schema_architecture_dict()
    )


# =============================================================================
# Aliases
# =============================================================================


LayerInputs = FunctionalMessagePassingLayerInputs
ResidualUpdateOutput = LayerResidualUpdateOutput
NormalizationOutput = LayerNormalizationOutput
LayerTrace = FunctionalMessagePassingLayerTrace
FunctionalMessagePassingLayerComputation = (
    LayerComputationOutput
)
LayerStages = FunctionalMessagePassingLayerStages


__all__ = (
    # Schema versions.
    "LAYER_INPUTS_SCHEMA_VERSION",
    "LAYER_TRACE_POLICY_SCHEMA_VERSION",
    "LAYER_RESIDUAL_UPDATE_SCHEMA_VERSION",
    "LAYER_NORMALIZATION_OUTPUT_SCHEMA_VERSION",
    "LAYER_INTERMEDIATE_TRACE_SCHEMA_VERSION",
    "LAYER_COMPUTATION_OUTPUT_SCHEMA_VERSION",
    # Trace policy.
    "LAYER_TRACE_NONE",
    "LAYER_TRACE_NODE",
    "LAYER_TRACE_FULL",
    "CANONICAL_LAYER_TRACE_MODES",
    "V2_0_IMPLEMENTED_LAYER_TRACE_MODES",
    "LayerTracePolicy",
    # Residual vocabulary.
    "LAYER_RESIDUAL_DISABLED",
    "LAYER_RESIDUAL_ADDITIVE",
    "CANONICAL_LAYER_RESIDUAL_MODES",
    "V2_0_IMPLEMENTED_LAYER_RESIDUAL_MODES",
    # Normalization vocabulary.
    "LAYER_NORMALIZATION_NONE",
    "LAYER_NORMALIZATION_LAYER_NORM",
    "CANONICAL_LAYER_NORMALIZATION_MODES",
    "V2_0_IMPLEMENTED_LAYER_NORMALIZATION_MODES",
    "LAYER_NORMALIZATION_PRE_RESIDUAL",
    "LAYER_NORMALIZATION_POST_RESIDUAL",
    "CANONICAL_LAYER_NORMALIZATION_POSITIONS",
    "V2_0_IMPLEMENTED_LAYER_NORMALIZATION_POSITIONS",
    # Equations and layouts.
    "LAYER_UPDATE_BRANCH_FORMULA",
    "LAYER_ADDITIVE_RESIDUAL_FORMULA",
    "LAYER_DISABLED_RESIDUAL_FORMULA",
    "LAYER_POST_NORMALIZATION_FORMULA",
    "LAYER_INPUT_LAYOUT",
    "LAYER_AGGREGATE_LAYOUT",
    "LAYER_OUTPUT_LAYOUT",
    "LAYER_SCIENTIFIC_INTERPRETATION",
    # Layer inputs.
    "FunctionalMessagePassingLayerInputs",
    "LayerInputs",
    # Residual output.
    "LayerResidualUpdateOutput",
    "ResidualUpdateOutput",
    # Normalization output.
    "LayerNormalizationOutput",
    "NormalizationOutput",
    # Trace.
    "FunctionalMessagePassingLayerTrace",
    "LayerTrace",
    # Internal output.
    "LayerComputationOutput",
    "FunctionalMessagePassingLayerComputation",
    # Stage tuple.
    "FunctionalMessagePassingLayerStages",
    "LayerStages",
    # Validators and public compatibility.
    "validate_layer_stage_chain",
    "build_public_layer_intermediates",
    "validate_public_layer_output",
    # Architecture.
    "layer_schema_architecture_dict",
    "layer_schema_architecture_fingerprint",
    # Existing public contracts.
    "AggregationOutput",
    "FunctionalMessagePassingIntermediates",
    "FunctionalMessagePassingLayerOutput",
)
