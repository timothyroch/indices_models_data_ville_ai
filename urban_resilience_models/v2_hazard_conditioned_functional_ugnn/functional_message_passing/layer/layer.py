"""
Complete orchestration for one functional message-passing layer.

Target path:
    urban_resilience_models/
        v2_hazard_conditioned_functional_ugnn/
            functional_message_passing/
                layer/
                    layer.py

This module coordinates one bounded graph-neural state transition:

    FunctionalMessagePassingInputs
        -> relation-specific source-state transformation
        -> structural edge normalization
        -> optional exact-relation gate
        -> optional exact-relation edge attention
        -> multiplicative edge-message construction
        -> existing target-node mean aggregation
        -> dropout and optional additive residual update
        -> optional post-residual layer normalization
        -> FunctionalMessagePassingLayerOutput

Ownership boundary
------------------
This file owns orchestration, exact stage ordering, cross-component lineage,
runtime trace selection, public-output assembly, layer-level architecture and
parameter provenance, and optional explicit diagnostics.

It does not reimplement:

- relation-transform mathematics;
- structural edge-normalization mathematics;
- relation-gate mathematics;
- edge-attention scoring or grouped softmax;
- message-factor multiplication;
- target-node aggregation mathematics;
- residual-update mathematics;
- layer-normalization mathematics;
- multi-layer stacking;
- prediction or readout.

Disabled mechanisms
-------------------
Disabled relation gating is represented by:

    relation_gate module = None
    relation_gate output = None
    multiplicative gate factor = exact one in the message builder

Disabled edge attention is represented by:

    edge_attention module = None
    edge_attention output = None
    multiplicative attention factor = exact one in the message builder

This is intentionally distinct from enabled uniform attention, which computes
zero logits followed by exact target-node/relation grouped softmax.

Trace policy
------------
Trace retention is runtime metadata, not numerical architecture.

``none``
    Return no optional internal trace.

``node``
    Retain aggregation, residual update, and normalization objects.

``full``
    Retain the complete edge- and node-level stage chain and expose the
    historical public ``FunctionalMessagePassingIntermediates`` object.

Ordinary ``forward`` returns only the public layer output. ``run_complete``
returns a complete audit object. Diagnostics are generated only through
explicit diagnostic methods and are excluded from the numerical architecture
fingerprint.

Layer index
-----------
The layer index is supplied at execution time rather than frozen in the module.
This is required so the same layer instance can be reused by a future fully
shared stack while still emitting the correct zero-based runtime layer index.

Regularization
--------------
The layer does not invent losses. Scalar regularization terms emitted by the
optional relation gate are preserved under the ``relation_gate.`` namespace.
Training code may combine these terms later.

Limits
-----------------
Gate values, attention values, message coefficients, aggregate magnitudes,
state changes, and retained traces are descriptive model quantities. They are
not automatically:

- causal importance scores;
- faithful explanations;
- calibrated uncertainty;
- counterfactual effects;
- mechanistic-identifiability evidence.
"""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
import math
from types import MappingProxyType
from typing import Any, Final, Mapping, NamedTuple

import torch
from torch import nn

from ...config import (
    FunctionalMessagePassingConfig,
    RelationConfig,
)
from ..aggregators import (
    MessageAggregator,
)
from ..edge_attention import (
    EdgeAttention,
)
from ..edge_normalization import (
    EdgeNormalization,
)
from ..message_builders import (
    MESSAGE_SEMANTIC_EDGE_POLICY_IGNORE,
    EdgeMessageBuilder,
    MessageBuilderRun,
    build_edge_message_builder,
)
from ..relation_family_gate import (
    RelationFamilyGate,
)
from ..relation_transforms import (
    RelationTransforms,
)
from ..schemas import (
    AggregationOutput,
    EdgeAttentionOutput,
    FunctionalMessagePassingInputs,
    FunctionalMessagePassingLayerOutput,
    RelationGateOutput,
    RelationTransformOutput,
    StructuralEdgeNormalizationOutput,
)
from .diagnostics import (
    DEFAULT_LAYER_DIAGNOSTIC_THRESHOLDS,
    LayerDiagnosticThresholds,
    LayerDiagnostics,
    build_layer_diagnostics,
)
from .normalization import (
    LAYER_NORMALIZATION_POST_RESIDUAL,
    LayerNormalizer,
    build_layer_normalizer_from_flag,
    validate_layer_normalization_output,
)
from .residual_update import (
    LayerResidualUpdater,
    build_layer_residual_updater_from_flags,
    validate_layer_residual_update_output,
)
from .schemas import (
    LAYER_TRACE_FULL,
    LAYER_TRACE_NONE,
    FunctionalMessagePassingLayerInputs,
    FunctionalMessagePassingLayerStages,
    FunctionalMessagePassingLayerTrace,
    LayerComputationOutput,
    LayerNormalizationOutput,
    LayerResidualUpdateOutput,
    LayerTracePolicy,
    build_public_layer_intermediates,
    validate_layer_stage_chain,
    validate_public_layer_output,
)


# =============================================================================
# Public identity
# =============================================================================


FUNCTIONAL_MESSAGE_PASSING_LAYER_SCHEMA_VERSION: Final[str] = "0.1"

FUNCTIONAL_MESSAGE_PASSING_LAYER_OPERATION_ORDER: Final[
    tuple[str, ...]
] = (
    "validate_component_and_runtime_contracts",
    "construct_runtime_layer_inputs",
    "transform_edge_aligned_source_states",
    "compute_structural_edge_normalization",
    "compute_optional_exact_relation_gates",
    "compute_optional_exact_relation_edge_attention",
    "construct_final_edge_messages",
    "aggregate_messages_by_target_node",
    "apply_dropout_and_optional_additive_residual",
    "apply_optional_post_residual_layer_normalization",
    "collect_scalar_regularization_terms",
    "construct_optional_trace",
    "construct_internal_layer_output",
    "assemble_public_layer_output",
    "validate_exact_complete_run_lineage",
)

FUNCTIONAL_MESSAGE_PASSING_LAYER_SCIENTIFIC_INTERPRETATION: Final[
    str
] = "one_hazard_conditioned_functional_graph_state_transition"

FUNCTIONAL_MESSAGE_PASSING_LAYER_OUTPUT_SCHEMA: Final[str] = (
    "FunctionalMessagePassingLayerOutput"
)

FUNCTIONAL_MESSAGE_PASSING_LAYER_AGGREGATION_ORCHESTRATED_HERE: Final[
    bool
] = True
FUNCTIONAL_MESSAGE_PASSING_LAYER_AGGREGATION_MATH_OWNED_HERE: Final[
    bool
] = False
FUNCTIONAL_MESSAGE_PASSING_LAYER_STACKING_OWNED_HERE: Final[bool] = False
FUNCTIONAL_MESSAGE_PASSING_LAYER_PREDICTION_OWNED_HERE: Final[bool] = False

FUNCTIONAL_MESSAGE_PASSING_LAYER_DISABLED_GATE_REPRESENTATION: Final[
    str
] = "None_with_message_builder_multiplicative_identity_one"
FUNCTIONAL_MESSAGE_PASSING_LAYER_DISABLED_ATTENTION_REPRESENTATION: Final[
    str
] = "None_with_message_builder_multiplicative_identity_one"
FUNCTIONAL_MESSAGE_PASSING_LAYER_UNIFORM_ATTENTION_REPRESENTATION: Final[
    str
] = "enabled_zero_logits_then_target_relation_grouped_softmax"

FUNCTIONAL_MESSAGE_PASSING_LAYER_TRACE_AFFECTS_NUMERICS: Final[bool] = False
FUNCTIONAL_MESSAGE_PASSING_LAYER_DIAGNOSTICS_AFFECT_NUMERICS: Final[
    bool
] = False


# =============================================================================
# Complete execution contracts
# =============================================================================


class FunctionalMessagePassingLayerEdgeStages(
    NamedTuple
):
    """
    Exact edge-level stage chain for one layer execution.
    """

    relation_transform: RelationTransformOutput
    edge_normalization: StructuralEdgeNormalizationOutput
    relation_gate: RelationGateOutput | None
    edge_attention: EdgeAttentionOutput | None
    message_builder_run: MessageBuilderRun


class FunctionalMessagePassingLayerNodeStages(
    NamedTuple
):
    """
    Exact node-level stages before internal/public output assembly.
    """

    aggregation: AggregationOutput
    residual_update: LayerResidualUpdateOutput
    normalization: LayerNormalizationOutput


@dataclass(slots=True, frozen=True)
class FunctionalMessagePassingLayerRun:
    """
    Complete exact execution record for one layer.

    Ordinary model execution may discard this object and retain only
    ``public_output``. The complete run is useful for tests, audits, and
    explicit diagnostics.
    """

    layer_inputs: FunctionalMessagePassingLayerInputs
    edge_stages: FunctionalMessagePassingLayerEdgeStages
    node_stages: FunctionalMessagePassingLayerNodeStages
    internal_output: LayerComputationOutput
    public_output: FunctionalMessagePassingLayerOutput

    schema_version: str = (
        FUNCTIONAL_MESSAGE_PASSING_LAYER_SCHEMA_VERSION
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
            self.edge_stages,
            FunctionalMessagePassingLayerEdgeStages,
        ):
            raise TypeError(
                "edge_stages must be a "
                "FunctionalMessagePassingLayerEdgeStages."
            )

        if not isinstance(
            self.node_stages,
            FunctionalMessagePassingLayerNodeStages,
        ):
            raise TypeError(
                "node_stages must be a "
                "FunctionalMessagePassingLayerNodeStages."
            )

        if not isinstance(
            self.internal_output,
            LayerComputationOutput,
        ):
            raise TypeError(
                "internal_output must be a LayerComputationOutput."
            )

        if not isinstance(
            self.public_output,
            FunctionalMessagePassingLayerOutput,
        ):
            raise TypeError(
                "public_output must be a "
                "FunctionalMessagePassingLayerOutput."
            )

        source_inputs = self.layer_inputs.source_inputs
        edge = self.edge_stages
        node = self.node_stages

        if edge.relation_transform.source_inputs is not source_inputs:
            raise ValueError(
                "relation_transform must preserve exact layer source inputs."
            )

        if edge.edge_normalization.source_inputs is not source_inputs:
            raise ValueError(
                "edge_normalization must preserve exact layer source inputs."
            )

        if (
            edge.relation_gate is not None
            and edge.relation_gate.source_inputs is not source_inputs
        ):
            raise ValueError(
                "relation_gate must preserve exact layer source inputs."
            )

        if (
            edge.edge_attention is not None
            and edge.edge_attention.source_inputs is not source_inputs
        ):
            raise ValueError(
                "edge_attention must preserve exact layer source inputs."
            )

        if (
            edge.message_builder_run.public_output.source_inputs
            is not source_inputs
        ):
            raise ValueError(
                "message_builder_run must preserve exact layer source inputs."
            )

        if (
            node.aggregation.source_messages
            is not edge.message_builder_run.public_output
        ):
            raise ValueError(
                "aggregation must consume the exact message-builder public "
                "output."
            )

        if node.residual_update.layer_inputs is not self.layer_inputs:
            raise ValueError(
                "residual_update must preserve exact layer_inputs."
            )

        if node.residual_update.aggregation is not node.aggregation:
            raise ValueError(
                "residual_update must consume exact aggregation."
            )

        if node.normalization.residual_update is not node.residual_update:
            raise ValueError(
                "normalization must consume exact residual_update."
            )

        if self.internal_output.layer_inputs is not self.layer_inputs:
            raise ValueError(
                "internal_output must preserve exact layer_inputs."
            )

        if self.internal_output.aggregation is not node.aggregation:
            raise ValueError(
                "internal_output must preserve exact aggregation."
            )

        if self.internal_output.residual_update is not node.residual_update:
            raise ValueError(
                "internal_output must preserve exact residual_update."
            )

        if self.internal_output.normalization is not node.normalization:
            raise ValueError(
                "internal_output must preserve exact normalization."
            )

        validate_public_layer_output(
            public_output=self.public_output,
            internal_output=self.internal_output,
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
    def layer_index(self) -> int:
        return self.layer_inputs.layer_index

    @property
    def updated_node_state(
        self,
    ) -> torch.Tensor:
        return self.public_output.updated_node_state

    @property
    def trace(
        self,
    ) -> FunctionalMessagePassingLayerTrace | None:
        return self.internal_output.trace


@dataclass(slots=True, frozen=True)
class FunctionalMessagePassingLayerRunWithDiagnostics:
    """
    Complete layer run plus an explicit tensor-free diagnostic report.
    """

    run: FunctionalMessagePassingLayerRun
    diagnostic_report: Mapping[str, Any]

    def __post_init__(self) -> None:
        if not isinstance(
            self.run,
            FunctionalMessagePassingLayerRun,
        ):
            raise TypeError(
                "run must be a FunctionalMessagePassingLayerRun."
            )

        if not isinstance(
            self.diagnostic_report,
            Mapping,
        ):
            raise TypeError(
                "diagnostic_report must be a mapping."
            )

        _assert_tensor_free_mapping(
            self.diagnostic_report
        )

        object.__setattr__(
            self,
            "diagnostic_report",
            MappingProxyType(
                dict(self.diagnostic_report)
            ),
        )

    @property
    def public_output(
        self,
    ) -> FunctionalMessagePassingLayerOutput:
        return self.run.public_output

    @property
    def internal_output(
        self,
    ) -> LayerComputationOutput:
        return self.run.internal_output


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


def _require_nonempty_string(
    name: str,
    value: str,
) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(
            f"{name} must be a non-empty string."
        )


def _require_optional_nonempty_string(
    name: str,
    value: str | None,
) -> None:
    if value is not None:
        _require_nonempty_string(
            name,
            value,
        )


def _require_boolean(
    name: str,
    value: bool,
) -> None:
    if not isinstance(value, bool):
        raise TypeError(
            f"{name} must be Boolean."
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


def _resolve_method_or_property(
    module: nn.Module,
    name: str,
) -> Any:
    value = getattr(
        module,
        name,
        None,
    )

    if callable(value):
        return value()

    return value


def _require_module_fingerprint(
    *,
    module_name: str,
    module: nn.Module,
    attribute: str,
    allow_none: bool,
) -> str | None:
    value = _resolve_method_or_property(
        module,
        attribute,
    )

    if value is None:
        if allow_none:
            return None

        raise RuntimeError(
            f"{module_name}.{attribute} must return a non-empty string."
        )

    _require_nonempty_string(
        f"{module_name}.{attribute}",
        value,
    )

    return value


def _module_architecture_dict(
    *,
    module_name: str,
    module: nn.Module,
) -> dict[str, Any]:
    method = getattr(
        module,
        "architecture_dict",
        None,
    )

    if not callable(method):
        raise RuntimeError(
            f"{module_name} must expose architecture_dict()."
        )

    value = method()

    if not isinstance(value, Mapping):
        raise RuntimeError(
            f"{module_name}.architecture_dict() must return a mapping."
        )

    return dict(value)


def _assert_finite_module_parameters(
    *,
    module_name: str,
    module: nn.Module,
) -> None:
    method = getattr(
        module,
        "assert_finite_parameters",
        None,
    )

    if callable(method):
        method()

    for parameter_name, parameter in module.named_parameters():
        if not bool(
            torch.isfinite(parameter)
            .all()
            .item()
        ):
            raise FloatingPointError(
                f"{module_name} parameter {parameter_name!r} contains "
                "non-finite values."
            )


def _assert_tensor_free_mapping(
    value: Mapping[str, Any],
) -> None:
    def visit(
        item: Any,
        *,
        path: str,
    ) -> None:
        if isinstance(item, torch.Tensor):
            raise TypeError(
                f"{path} must not retain tensors."
            )

        if isinstance(item, nn.Module):
            raise TypeError(
                f"{path} must not retain modules."
            )

        if isinstance(item, Mapping):
            for key, nested in item.items():
                visit(
                    nested,
                    path=f"{path}.{key}",
                )
            return

        if isinstance(item, (list, tuple)):
            for index, nested in enumerate(item):
                visit(
                    nested,
                    path=f"{path}[{index}]",
                )
            return

        if (
            item is None
            or isinstance(
                item,
                (
                    str,
                    bool,
                    int,
                    float,
                ),
            )
        ):
            return

        raise TypeError(
            f"{path} contains unsupported value type "
            f"{type(item).__name__}."
        )

    visit(
        value,
        path="diagnostic_report",
    )


def _resolve_trace_policy(
    *,
    default_policy: LayerTracePolicy,
    trace_policy: LayerTracePolicy | str | None,
    capture_intermediate_messages: bool | None,
) -> LayerTracePolicy:
    if not isinstance(
        default_policy,
        LayerTracePolicy,
    ):
        raise TypeError(
            "default_policy must be a LayerTracePolicy."
        )

    if (
        trace_policy is not None
        and capture_intermediate_messages is not None
    ):
        raise ValueError(
            "Specify trace_policy or capture_intermediate_messages, not both."
        )

    if capture_intermediate_messages is not None:
        _require_boolean(
            "capture_intermediate_messages",
            capture_intermediate_messages,
        )
        resolved = (
            LayerTracePolicy
            .from_capture_intermediate_messages(
                capture_intermediate_messages
            )
        )
    elif trace_policy is None:
        resolved = default_policy
    elif isinstance(
        trace_policy,
        LayerTracePolicy,
    ):
        resolved = trace_policy
    elif isinstance(trace_policy, str):
        resolved = LayerTracePolicy(
            mode=trace_policy
        )
    else:
        raise TypeError(
            "trace_policy must be a LayerTracePolicy, string, or None."
        )

    resolved.assert_implemented()
    return resolved


def _merge_regularization_terms(
    relation_gate: RelationGateOutput | None,
) -> Mapping[str, torch.Tensor]:
    terms: dict[str, torch.Tensor] = {}

    if relation_gate is None:
        return MappingProxyType(
            terms
        )

    for name, value in relation_gate.regularization_terms.items():
        _require_nonempty_string(
            "relation-gate regularization term name",
            name,
        )

        if not isinstance(value, torch.Tensor):
            raise TypeError(
                f"Relation-gate regularization term {name!r} must be a "
                "tensor."
            )

        if value.ndim != 0:
            raise ValueError(
                f"Relation-gate regularization term {name!r} must be scalar."
            )

        if not value.dtype.is_floating_point:
            raise ValueError(
                f"Relation-gate regularization term {name!r} must use a "
                "floating-point dtype."
            )

        if value.device != relation_gate.source_inputs.device:
            raise ValueError(
                f"Relation-gate regularization term {name!r} must share the "
                "source-input device."
            )

        if not bool(
            torch.isfinite(value).item()
        ):
            raise FloatingPointError(
                f"Relation-gate regularization term {name!r} must be finite."
            )

        namespaced = f"relation_gate.{name}"

        if namespaced in terms:
            raise ValueError(
                f"Duplicate layer regularization term {namespaced!r}."
            )

        terms[namespaced] = value

    return MappingProxyType(
        terms
    )


# =============================================================================
# Public assembly and run validation
# =============================================================================


def assemble_functional_message_passing_layer_output(
    *,
    internal_output: LayerComputationOutput,
) -> FunctionalMessagePassingLayerOutput:
    """
    Assemble the existing public layer schema without recomputing tensors.
    """

    if not isinstance(
        internal_output,
        LayerComputationOutput,
    ):
        raise TypeError(
            "internal_output must be a LayerComputationOutput."
        )

    intermediates = None

    if internal_output.trace_policy.mode == LAYER_TRACE_FULL:
        if internal_output.trace is None:
            raise ValueError(
                "Full trace policy requires an internal trace."
            )

        intermediates = build_public_layer_intermediates(
            internal_output.trace
        )

    public_output = FunctionalMessagePassingLayerOutput(
        updated_node_state=(
            internal_output.updated_node_state
        ),
        node_aggregate=(
            internal_output.node_aggregate
        ),
        incoming_edge_count=(
            internal_output.incoming_edge_count
        ),
        source_inputs=(
            internal_output.source_inputs
        ),
        layer_index=(
            internal_output.layer_index
        ),
        residual_enabled=(
            internal_output.residual_enabled
        ),
        layer_norm_enabled=(
            internal_output.layer_norm_enabled
        ),
        encoder_architecture_fingerprint=(
            internal_output
            .layer_architecture_fingerprint
        ),
        lineage_fingerprint=(
            internal_output.lineage_fingerprint
        ),
        intermediates=intermediates,
        regularization_terms=(
            internal_output.regularization_terms
        ),
    )

    validate_public_layer_output(
        public_output=public_output,
        internal_output=internal_output,
    )

    return public_output


def validate_functional_message_passing_layer_run(
    run: FunctionalMessagePassingLayerRun,
) -> None:
    """
    Validate exact edge-to-node-to-public lineage for one complete run.
    """

    if not isinstance(
        run,
        FunctionalMessagePassingLayerRun,
    ):
        raise TypeError(
            "run must be a FunctionalMessagePassingLayerRun."
        )

    edge = run.edge_stages
    node = run.node_stages

    validate_layer_stage_chain(
        layer_inputs=run.layer_inputs,
        aggregation=node.aggregation,
        residual_update=node.residual_update,
        normalization=node.normalization,
        computation_output=run.internal_output,
    )

    validate_layer_residual_update_output(
        output=node.residual_update,
        aggregation=node.aggregation,
        layer_inputs=run.layer_inputs,
    )

    if (
        edge.message_builder_run.public_output
        is not node.aggregation.source_messages
    ):
        raise ValueError(
            "Complete run lost exact message-builder-to-aggregation lineage."
        )

    validate_public_layer_output(
        public_output=run.public_output,
        internal_output=run.internal_output,
    )


# =============================================================================
# Layer orchestrator
# =============================================================================


class FunctionalMessagePassingLayer(nn.Module):
    """
    Coordinate one complete functional message-passing state transition.

    Parameters
    ----------
    relation_transforms:
        Existing relation-transform dispatcher.

    edge_normalization:
        Existing graph-structural edge normalizer.

    relation_gate:
        Optional exact-relation hazard-conditioned gate. ``None`` represents
        disabled gating.

    edge_attention:
        Optional exact-relation edge-attention orchestrator. ``None``
        represents disabled attention.

    message_builder:
        Existing parameter-free multiplicative edge-message builder.

    aggregator:
        Existing target-node message aggregator.

    residual_updater:
        Dropout and optional additive residual stage.

    normalizer:
        Optional post-residual layer-normalization stage.

    default_trace_policy:
        Runtime retention default. It does not affect numerical architecture.

    diagnostics:
        Optional explicit tensor-free diagnostics module. It is never executed
        by ordinary ``forward`` and does not affect numerical architecture.
    """

    relation_transforms: RelationTransforms
    edge_normalization: EdgeNormalization
    relation_gate: RelationFamilyGate | None
    edge_attention: EdgeAttention | None
    message_builder: EdgeMessageBuilder
    aggregator: MessageAggregator
    residual_updater: LayerResidualUpdater
    normalizer: LayerNormalizer
    default_trace_policy: LayerTracePolicy
    diagnostics: LayerDiagnostics | None

    def __init__(
        self,
        *,
        relation_transforms: RelationTransforms,
        edge_normalization: EdgeNormalization,
        relation_gate: RelationFamilyGate | None,
        edge_attention: EdgeAttention | None,
        message_builder: EdgeMessageBuilder,
        aggregator: MessageAggregator,
        residual_updater: LayerResidualUpdater,
        normalizer: LayerNormalizer,
        default_trace_policy: LayerTracePolicy | None = None,
        diagnostics: LayerDiagnostics | None = None,
    ) -> None:
        super().__init__()

        _require_layer_components(
            relation_transforms=relation_transforms,
            edge_normalization=edge_normalization,
            relation_gate=relation_gate,
            edge_attention=edge_attention,
            message_builder=message_builder,
            aggregator=aggregator,
            residual_updater=residual_updater,
            normalizer=normalizer,
            diagnostics=diagnostics,
        )

        self.relation_transforms = relation_transforms
        self.edge_normalization = edge_normalization
        self.relation_gate = relation_gate
        self.edge_attention = edge_attention
        self.message_builder = message_builder
        self.aggregator = aggregator
        self.residual_updater = residual_updater
        self.normalizer = normalizer
        self.default_trace_policy = (
            LayerTracePolicy()
            if default_trace_policy is None
            else default_trace_policy
        )
        self.diagnostics = diagnostics

        if not isinstance(
            self.default_trace_policy,
            LayerTracePolicy,
        ):
            raise TypeError(
                "default_trace_policy must be a LayerTracePolicy."
            )

        self.default_trace_policy.assert_implemented()
        self._assert_static_component_contract()

    # ------------------------------------------------------------------
    # Construction from configuration
    # ------------------------------------------------------------------

    @classmethod
    def from_config(
        cls,
        *,
        config: FunctionalMessagePassingConfig,
        relation_config: RelationConfig,
        source_inputs: FunctionalMessagePassingInputs,
        semantic_edge_policy: str = (
            MESSAGE_SEMANTIC_EDGE_POLICY_IGNORE
        ),
        relation_transform_bias: bool = True,
        relation_gate_use_node_state: bool = True,
        relation_gate_use_hazard_query: bool = True,
        relation_gate_layer_norm: bool = True,
        relation_gate_bias: bool = True,
        relation_prior_epsilon: float = 1e-4,
        layer_norm_epsilon: float = 1e-5,
        layer_norm_elementwise_affine: bool = True,
        layer_norm_bias_enabled: bool = True,
        diagnostics_enabled: bool = False,
        diagnostics_include_per_graph: bool = True,
        diagnostics_include_edge_report: bool = True,
        diagnostic_thresholds: LayerDiagnosticThresholds = (
            DEFAULT_LAYER_DIAGNOSTIC_THRESHOLDS
        ),
    ) -> "FunctionalMessagePassingLayer":
        """
        Build one enabled layer from canonical configuration and source schema.

        Construction is aligned to the source input's hidden width, device,
        dtype, exact relation ordering, and compiled registry.
        """

        if not isinstance(
            config,
            FunctionalMessagePassingConfig,
        ):
            raise TypeError(
                "config must be a FunctionalMessagePassingConfig."
            )

        if not isinstance(
            relation_config,
            RelationConfig,
        ):
            raise TypeError(
                "relation_config must be a RelationConfig."
            )

        _require_source_inputs(
            source_inputs
        )
        _require_boolean(
            "relation_transform_bias",
            relation_transform_bias,
        )
        _require_boolean(
            "relation_gate_use_node_state",
            relation_gate_use_node_state,
        )
        _require_boolean(
            "relation_gate_use_hazard_query",
            relation_gate_use_hazard_query,
        )
        _require_boolean(
            "relation_gate_layer_norm",
            relation_gate_layer_norm,
        )
        _require_boolean(
            "relation_gate_bias",
            relation_gate_bias,
        )
        _require_boolean(
            "layer_norm_elementwise_affine",
            layer_norm_elementwise_affine,
        )
        _require_boolean(
            "layer_norm_bias_enabled",
            layer_norm_bias_enabled,
        )
        _require_boolean(
            "diagnostics_enabled",
            diagnostics_enabled,
        )
        _require_boolean(
            "diagnostics_include_per_graph",
            diagnostics_include_per_graph,
        )
        _require_boolean(
            "diagnostics_include_edge_report",
            diagnostics_include_edge_report,
        )
        _require_probability(
            "layer_norm_epsilon proxy",
            min(float(layer_norm_epsilon), 0.999999),
        )

        if float(layer_norm_epsilon) <= 0.0 or not math.isfinite(
            float(layer_norm_epsilon)
        ):
            raise ValueError(
                "layer_norm_epsilon must be finite and strictly positive."
            )

        if float(relation_prior_epsilon) <= 0.0 or not math.isfinite(
            float(relation_prior_epsilon)
        ):
            raise ValueError(
                "relation_prior_epsilon must be finite and strictly positive."
            )

        config.validate()
        relation_config.validate()

        if not config.enabled:
            raise ValueError(
                "FunctionalMessagePassingLayer.from_config requires "
                "config.enabled=True."
            )

        config.assert_implemented()

        relation_transforms = RelationTransforms.from_config(
            config=config,
            hidden_dim=source_inputs.hidden_dim,
            compiled_relation_registry=(
                source_inputs.compiled_relation_registry
            ),
            bias=relation_transform_bias,
        ).to(
            device=source_inputs.device,
            dtype=source_inputs.dtype,
        )

        edge_normalization = EdgeNormalization.from_config(
            config=config
        )

        relation_gate = (
            RelationFamilyGate.from_config(
                config=relation_config,
                source_inputs=source_inputs,
                use_node_state=relation_gate_use_node_state,
                use_hazard_query=relation_gate_use_hazard_query,
                layer_norm=relation_gate_layer_norm,
                relation_bias=relation_gate_bias,
                prior_epsilon=relation_prior_epsilon,
            )
            if relation_config.gate_enabled
            else None
        )

        edge_attention = (
            EdgeAttention.from_config(
                config=config,
                source_inputs=source_inputs,
            )
            if config.attention_enabled
            else None
        )

        message_builder = build_edge_message_builder(
            semantic_edge_policy=semantic_edge_policy,
            diagnostics_enabled=False,
        )

        aggregator = MessageAggregator.from_config(
            config=config
        )

        residual_updater = (
            build_layer_residual_updater_from_flags(
                residual_enabled=config.residual,
                dropout_probability=config.dropout,
            )
        )

        normalizer = build_layer_normalizer_from_flag(
            source_inputs.hidden_dim,
            layer_norm_enabled=config.layer_norm,
            epsilon=float(layer_norm_epsilon),
            elementwise_affine=(
                layer_norm_elementwise_affine
            ),
            bias_enabled=(
                layer_norm_bias_enabled
            ),
            device=source_inputs.device,
            dtype=source_inputs.dtype,
        )

        default_trace_policy = (
            LayerTracePolicy
            .from_capture_intermediate_messages(
                config.capture_intermediate_messages
            )
        )

        diagnostics = (
            build_layer_diagnostics(
                include_per_graph=(
                    diagnostics_include_per_graph
                ),
                include_edge_report=(
                    diagnostics_include_edge_report
                ),
                thresholds=(
                    diagnostic_thresholds
                ),
            )
            if diagnostics_enabled
            else None
        )

        return cls(
            relation_transforms=relation_transforms,
            edge_normalization=edge_normalization,
            relation_gate=relation_gate,
            edge_attention=edge_attention,
            message_builder=message_builder,
            aggregator=aggregator,
            residual_updater=residual_updater,
            normalizer=normalizer,
            default_trace_policy=(
                default_trace_policy
            ),
            diagnostics=diagnostics,
        )

    # ------------------------------------------------------------------
    # Public component identity
    # ------------------------------------------------------------------

    @property
    def hidden_dim(self) -> int:
        return self.relation_transforms.hidden_dim

    @property
    def relation_names(
        self,
    ) -> tuple[str, ...]:
        return self.relation_transforms.relation_names

    @property
    def stable_relation_ids(
        self,
    ) -> tuple[int, ...]:
        return self.relation_transforms.stable_relation_ids

    @property
    def num_relations(self) -> int:
        return self.relation_transforms.relation_count

    @property
    def gate_enabled(self) -> bool:
        return self.relation_gate is not None

    @property
    def attention_enabled(self) -> bool:
        return self.edge_attention is not None

    @property
    def residual_enabled(self) -> bool:
        return self.residual_updater.residual_enabled

    @property
    def layer_norm_enabled(self) -> bool:
        return self.normalizer.normalization_enabled

    @property
    def diagnostics_enabled(self) -> bool:
        return self.diagnostics is not None

    @property
    def semantic_edge_policy(self) -> str:
        return self.message_builder.semantic_edge_policy

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

    # ------------------------------------------------------------------
    # Architecture and parameter provenance
    # ------------------------------------------------------------------

    def numerical_architecture_dict(
        self,
    ) -> dict[str, Any]:
        """
        Return numerical architecture only.

        Runtime layer index, training mode, trace policy, and diagnostic
        settings are intentionally excluded.
        """

        return {
            "schema_version": (
                FUNCTIONAL_MESSAGE_PASSING_LAYER_SCHEMA_VERSION
            ),
            "scientific_interpretation": (
                FUNCTIONAL_MESSAGE_PASSING_LAYER_SCIENTIFIC_INTERPRETATION
            ),
            "operation_order": list(
                FUNCTIONAL_MESSAGE_PASSING_LAYER_OPERATION_ORDER
            ),
            "hidden_dim": self.hidden_dim,
            "num_relations": self.num_relations,
            "relation_names": list(
                self.relation_names
            ),
            "stable_relation_ids": list(
                self.stable_relation_ids
            ),
            "gate_enabled": (
                self.gate_enabled
            ),
            "attention_enabled": (
                self.attention_enabled
            ),
            "residual_enabled": (
                self.residual_enabled
            ),
            "layer_norm_enabled": (
                self.layer_norm_enabled
            ),
            "semantic_edge_policy": (
                self.semantic_edge_policy
            ),
            "relation_transforms": (
                _module_architecture_dict(
                    module_name="relation_transforms",
                    module=self.relation_transforms,
                )
            ),
            "edge_normalization": (
                _module_architecture_dict(
                    module_name="edge_normalization",
                    module=self.edge_normalization,
                )
            ),
            "relation_gate": (
                _module_architecture_dict(
                    module_name="relation_gate",
                    module=self.relation_gate,
                )
                if self.relation_gate is not None
                else None
            ),
            "edge_attention": (
                _module_architecture_dict(
                    module_name="edge_attention",
                    module=self.edge_attention,
                )
                if self.edge_attention is not None
                else None
            ),
            "message_builder": (
                self.message_builder
                .numerical_architecture_dict()
            ),
            "aggregator": (
                _module_architecture_dict(
                    module_name="aggregator",
                    module=self.aggregator,
                )
            ),
            "residual_updater": (
                self.residual_updater
                .architecture_dict()
            ),
            "normalizer": (
                self.normalizer
                .architecture_dict()
            ),
            "disabled_gate_representation": (
                FUNCTIONAL_MESSAGE_PASSING_LAYER_DISABLED_GATE_REPRESENTATION
            ),
            "disabled_attention_representation": (
                FUNCTIONAL_MESSAGE_PASSING_LAYER_DISABLED_ATTENTION_REPRESENTATION
            ),
            "uniform_attention_representation": (
                FUNCTIONAL_MESSAGE_PASSING_LAYER_UNIFORM_ATTENTION_REPRESENTATION
            ),
            "output_schema": (
                FUNCTIONAL_MESSAGE_PASSING_LAYER_OUTPUT_SCHEMA
            ),
            "aggregation_orchestrated_here": (
                FUNCTIONAL_MESSAGE_PASSING_LAYER_AGGREGATION_ORCHESTRATED_HERE
            ),
            "aggregation_math_owned_here": (
                FUNCTIONAL_MESSAGE_PASSING_LAYER_AGGREGATION_MATH_OWNED_HERE
            ),
            "stacking_owned_here": (
                FUNCTIONAL_MESSAGE_PASSING_LAYER_STACKING_OWNED_HERE
            ),
            "prediction_owned_here": (
                FUNCTIONAL_MESSAGE_PASSING_LAYER_PREDICTION_OWNED_HERE
            ),
            "trace_affects_numerics": (
                FUNCTIONAL_MESSAGE_PASSING_LAYER_TRACE_AFFECTS_NUMERICS
            ),
            "diagnostics_affect_numerics": (
                FUNCTIONAL_MESSAGE_PASSING_LAYER_DIAGNOSTICS_AFFECT_NUMERICS
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
            "claims_causal_importance": False,
            "claims_explanation_faithfulness": False,
        }

    def architecture_dict(
        self,
    ) -> dict[str, Any]:
        return self.numerical_architecture_dict()

    def architecture_fingerprint(
        self,
    ) -> str:
        return _fingerprint(
            self.numerical_architecture_dict()
        )

    def parameter_fingerprint(
        self,
    ) -> str:
        """
        Compose exact component parameter fingerprints.

        Parameter-free stages may report ``None``. The layer fingerprint still
        remains a non-empty string because it identifies the complete ordered
        parameter-bearing architecture.
        """

        payload = {
            "schema_version": (
                FUNCTIONAL_MESSAGE_PASSING_LAYER_SCHEMA_VERSION
            ),
            "relation_transforms": (
                _require_module_fingerprint(
                    module_name="relation_transforms",
                    module=self.relation_transforms,
                    attribute="parameter_fingerprint",
                    allow_none=False,
                )
            ),
            "edge_normalization": (
                _require_module_fingerprint(
                    module_name="edge_normalization",
                    module=self.edge_normalization,
                    attribute="parameter_fingerprint",
                    allow_none=False,
                )
            ),
            "relation_gate": (
                _require_module_fingerprint(
                    module_name="relation_gate",
                    module=self.relation_gate,
                    attribute="parameter_fingerprint",
                    allow_none=False,
                )
                if self.relation_gate is not None
                else None
            ),
            "edge_attention": (
                _require_module_fingerprint(
                    module_name="edge_attention",
                    module=self.edge_attention,
                    attribute="parameter_fingerprint",
                    allow_none=True,
                )
                if self.edge_attention is not None
                else None
            ),
            "message_builder": (
                self.message_builder
                .parameter_fingerprint
            ),
            "aggregator": (
                _require_module_fingerprint(
                    module_name="aggregator",
                    module=self.aggregator,
                    attribute="parameter_fingerprint",
                    allow_none=False,
                )
            ),
            "residual_updater": (
                self.residual_updater
                .parameter_fingerprint
            ),
            "normalizer": (
                self.normalizer
                .parameter_fingerprint()
            ),
            "ordered_state_dict_keys": list(
                self.state_dict()
            ),
            "parameter_count": (
                self.parameter_count
            ),
        }

        return _fingerprint(
            payload
        )

    def diagnostics_architecture_dict(
        self,
    ) -> dict[str, Any] | None:
        if self.diagnostics is None:
            return None

        return self.diagnostics.architecture_dict()

    def runtime_dict(
        self,
    ) -> dict[str, Any]:
        return {
            "numerical_architecture": (
                self.numerical_architecture_dict()
            ),
            "architecture_fingerprint": (
                self.architecture_fingerprint()
            ),
            "parameter_fingerprint": (
                self.parameter_fingerprint()
            ),
            "training": self.training,
            "default_trace_policy": (
                self.default_trace_policy
                .architecture_dict()
            ),
            "diagnostics_enabled": (
                self.diagnostics_enabled
            ),
            "diagnostics_architecture": (
                self.diagnostics_architecture_dict()
            ),
        }

    # ------------------------------------------------------------------
    # Component validation
    # ------------------------------------------------------------------

    def _assert_static_component_contract(
        self,
    ) -> None:
        if self.relation_transforms.hidden_dim != (
            self.normalizer.hidden_dim
        ):
            raise ValueError(
                "RelationTransforms hidden width and LayerNormalizer "
                "hidden width must match."
            )

        if self.normalizer.normalization_position != (
            LAYER_NORMALIZATION_POST_RESIDUAL
        ):
            raise ValueError(
                "The bounded layer requires post-residual normalization."
            )

        if self.relation_gate is not None:
            if self.relation_gate.relation_names != (
                self.relation_names
            ):
                raise ValueError(
                    "Relation gate relation ordering differs from relation "
                    "transforms."
                )

            if self.relation_gate.stable_relation_ids != (
                self.stable_relation_ids
            ):
                raise ValueError(
                    "Relation gate stable IDs differ from relation transforms."
                )

            if self.relation_gate.num_relations != self.num_relations:
                raise ValueError(
                    "Relation gate relation count differs from relation "
                    "transforms."
                )

        if self.edge_attention is not None:
            if self.edge_attention.relation_names != (
                self.relation_names
            ):
                raise ValueError(
                    "Edge attention relation ordering differs from relation "
                    "transforms."
                )

            if self.edge_attention.stable_relation_ids != (
                self.stable_relation_ids
            ):
                raise ValueError(
                    "Edge attention stable IDs differ from relation "
                    "transforms."
                )

            if self.edge_attention.num_relations != self.num_relations:
                raise ValueError(
                    "Edge attention relation count differs from relation "
                    "transforms."
                )

        self.residual_updater.assert_parameter_free()
        self.normalizer.assert_parameter_contract()
        self.message_builder.assert_parameter_free()

        if self.diagnostics is not None:
            self.diagnostics.assert_parameter_free()

        self.assert_finite_parameters()

    def _validate_source_inputs(
        self,
        source_inputs: FunctionalMessagePassingInputs,
    ) -> None:
        _require_source_inputs(
            source_inputs
        )

        if source_inputs.hidden_dim != self.hidden_dim:
            raise ValueError(
                "source_inputs hidden width differs from the layer."
            )

        if source_inputs.relation_names != self.relation_names:
            raise ValueError(
                "source_inputs relation ordering differs from the layer."
            )

        if source_inputs.stable_relation_ids != self.stable_relation_ids:
            raise ValueError(
                "source_inputs stable relation IDs differ from the layer."
            )

        if source_inputs.num_relations != self.num_relations:
            raise ValueError(
                "source_inputs relation count differs from the layer."
            )

        transform_registry_fingerprint = (
            self.relation_transforms
            .compiled_relation_registry_fingerprint
        )
        input_registry_fingerprint = (
            source_inputs
            .compiled_relation_registry
            .fingerprint()
        )

        if input_registry_fingerprint != transform_registry_fingerprint:
            raise ValueError(
                "source_inputs reference a different compiled relation "
                "registry from the layer."
            )

        if self.residual_updater.training is not self.training:
            raise RuntimeError(
                "Residual updater train/eval mode differs from the layer."
            )

        self.assert_finite_parameters()

    def assert_finite_parameters(
        self,
    ) -> None:
        for name, module in (
            (
                "relation_transforms",
                self.relation_transforms,
            ),
            (
                "edge_normalization",
                self.edge_normalization,
            ),
            (
                "message_builder",
                self.message_builder,
            ),
            (
                "aggregator",
                self.aggregator,
            ),
            (
                "residual_updater",
                self.residual_updater,
            ),
            (
                "normalizer",
                self.normalizer,
            ),
        ):
            _assert_finite_module_parameters(
                module_name=name,
                module=module,
            )

        if self.relation_gate is not None:
            _assert_finite_module_parameters(
                module_name="relation_gate",
                module=self.relation_gate,
            )

        if self.edge_attention is not None:
            _assert_finite_module_parameters(
                module_name="edge_attention",
                module=self.edge_attention,
            )

        if self.diagnostics is not None:
            self.diagnostics.assert_parameter_free()

        observed_parameter_count = sum(
            int(parameter.numel())
            for parameter in self.parameters()
        )

        if observed_parameter_count != self.parameter_count:
            raise RuntimeError(
                "FunctionalMessagePassingLayer parameter counting is "
                "inconsistent."
            )

    # ------------------------------------------------------------------
    # Runtime trace and layer-input construction
    # ------------------------------------------------------------------

    def resolve_trace_policy(
        self,
        *,
        trace_policy: LayerTracePolicy | str | None = None,
        capture_intermediate_messages: bool | None = None,
    ) -> LayerTracePolicy:
        return _resolve_trace_policy(
            default_policy=(
                self.default_trace_policy
            ),
            trace_policy=trace_policy,
            capture_intermediate_messages=(
                capture_intermediate_messages
            ),
        )

    def build_layer_inputs(
        self,
        source_inputs: FunctionalMessagePassingInputs,
        *,
        layer_index: int,
        trace_policy: LayerTracePolicy | str | None = None,
        capture_intermediate_messages: bool | None = None,
        source_stack_fingerprint: str | None = None,
    ) -> FunctionalMessagePassingLayerInputs:
        self._validate_source_inputs(
            source_inputs
        )
        _require_nonnegative_int(
            "layer_index",
            layer_index,
        )
        _require_optional_nonempty_string(
            "source_stack_fingerprint",
            source_stack_fingerprint,
        )

        resolved_trace_policy = (
            self.resolve_trace_policy(
                trace_policy=trace_policy,
                capture_intermediate_messages=(
                    capture_intermediate_messages
                ),
            )
        )

        return FunctionalMessagePassingLayerInputs(
            source_inputs=source_inputs,
            layer_index=layer_index,
            trace_policy=(
                resolved_trace_policy
            ),
            training=self.training,
            source_stack_fingerprint=(
                source_stack_fingerprint
            ),
        )

    # ------------------------------------------------------------------
    # Edge-level execution
    # ------------------------------------------------------------------

    def compute_edge_stages(
        self,
        source_inputs: FunctionalMessagePassingInputs,
    ) -> FunctionalMessagePassingLayerEdgeStages:
        """
        Execute relation transform through final edge-message construction.
        """

        self._validate_source_inputs(
            source_inputs
        )

        relation_transform = (
            self.relation_transforms(
                source_inputs
            )
        )
        edge_normalization = (
            self.edge_normalization(
                source_inputs
            )
        )
        relation_gate = (
            self.relation_gate(
                source_inputs
            )
            if self.relation_gate is not None
            else None
        )
        edge_attention = (
            self.edge_attention(
                source_inputs
            )
            if self.edge_attention is not None
            else None
        )
        message_builder_run = (
            self.message_builder.run_complete(
                relation_transform=(
                    relation_transform
                ),
                edge_normalization=(
                    edge_normalization
                ),
                relation_gate=(
                    relation_gate
                ),
                edge_attention=(
                    edge_attention
                ),
                source_inputs=(
                    source_inputs
                ),
            )
        )

        stages = (
            FunctionalMessagePassingLayerEdgeStages(
                relation_transform=(
                    relation_transform
                ),
                edge_normalization=(
                    edge_normalization
                ),
                relation_gate=(
                    relation_gate
                ),
                edge_attention=(
                    edge_attention
                ),
                message_builder_run=(
                    message_builder_run
                ),
            )
        )

        self._validate_edge_stages(
            stages,
            source_inputs=source_inputs,
        )

        return stages

    def _validate_edge_stages(
        self,
        stages: FunctionalMessagePassingLayerEdgeStages,
        *,
        source_inputs: FunctionalMessagePassingInputs,
    ) -> None:
        if not isinstance(
            stages,
            FunctionalMessagePassingLayerEdgeStages,
        ):
            raise TypeError(
                "stages must be a "
                "FunctionalMessagePassingLayerEdgeStages."
            )

        if stages.relation_transform.source_inputs is not source_inputs:
            raise ValueError(
                "Relation-transform stage lost exact source-input identity."
            )

        if stages.edge_normalization.source_inputs is not source_inputs:
            raise ValueError(
                "Edge-normalization stage lost exact source-input identity."
            )

        if self.gate_enabled != (
            stages.relation_gate is not None
        ):
            raise ValueError(
                "Relation-gate output presence differs from layer "
                "configuration."
            )

        if (
            stages.relation_gate is not None
            and stages.relation_gate.source_inputs is not source_inputs
        ):
            raise ValueError(
                "Relation-gate stage lost exact source-input identity."
            )

        if self.attention_enabled != (
            stages.edge_attention is not None
        ):
            raise ValueError(
                "Edge-attention output presence differs from layer "
                "configuration."
            )

        if (
            stages.edge_attention is not None
            and stages.edge_attention.source_inputs is not source_inputs
        ):
            raise ValueError(
                "Edge-attention stage lost exact source-input identity."
            )

        if (
            stages.message_builder_run
            .composition_output
            .relation_transform
            is not stages.relation_transform
        ):
            raise ValueError(
                "Message builder lost exact relation-transform lineage."
            )

        if (
            stages.message_builder_run
            .resolved_coefficients
            .edge_normalization
            is not stages.edge_normalization
        ):
            raise ValueError(
                "Message builder lost exact edge-normalization lineage."
            )

        if (
            stages.message_builder_run
            .resolved_coefficients
            .relation_gate
            is not stages.relation_gate
        ):
            raise ValueError(
                "Message builder lost exact relation-gate lineage."
            )

        if (
            stages.message_builder_run
            .resolved_coefficients
            .edge_attention
            is not stages.edge_attention
        ):
            raise ValueError(
                "Message builder lost exact edge-attention lineage."
            )

        if (
            stages.message_builder_run
            .public_output
            .source_inputs
            is not source_inputs
        ):
            raise ValueError(
                "Message-builder output lost exact source-input identity."
            )

    # ------------------------------------------------------------------
    # Node-level execution
    # ------------------------------------------------------------------

    def compute_node_stages(
        self,
        *,
        edge_stages: FunctionalMessagePassingLayerEdgeStages,
        layer_inputs: FunctionalMessagePassingLayerInputs,
    ) -> FunctionalMessagePassingLayerNodeStages:
        """
        Execute existing aggregation, residual update, and normalization.
        """

        if not isinstance(
            edge_stages,
            FunctionalMessagePassingLayerEdgeStages,
        ):
            raise TypeError(
                "edge_stages must be a "
                "FunctionalMessagePassingLayerEdgeStages."
            )

        if not isinstance(
            layer_inputs,
            FunctionalMessagePassingLayerInputs,
        ):
            raise TypeError(
                "layer_inputs must be a "
                "FunctionalMessagePassingLayerInputs."
            )

        self._validate_edge_stages(
            edge_stages,
            source_inputs=(
                layer_inputs.source_inputs
            ),
        )

        if layer_inputs.training is not self.training:
            raise ValueError(
                "layer_inputs.training must match the layer runtime mode."
            )

        aggregation = self.aggregator(
            edge_stages
            .message_builder_run
            .public_output
        )
        residual_update = self.residual_updater(
            aggregation=aggregation,
            layer_inputs=layer_inputs,
        )
        normalization = self.normalizer(
            residual_update
        )

        stages = (
            FunctionalMessagePassingLayerNodeStages(
                aggregation=aggregation,
                residual_update=(
                    residual_update
                ),
                normalization=(
                    normalization
                ),
            )
        )

        self._validate_node_stages(
            stages,
            edge_stages=edge_stages,
            layer_inputs=layer_inputs,
        )

        return stages

    def _validate_node_stages(
        self,
        stages: FunctionalMessagePassingLayerNodeStages,
        *,
        edge_stages: FunctionalMessagePassingLayerEdgeStages,
        layer_inputs: FunctionalMessagePassingLayerInputs,
    ) -> None:
        if not isinstance(
            stages,
            FunctionalMessagePassingLayerNodeStages,
        ):
            raise TypeError(
                "stages must be a "
                "FunctionalMessagePassingLayerNodeStages."
            )

        if (
            stages.aggregation.source_messages
            is not edge_stages
            .message_builder_run
            .public_output
        ):
            raise ValueError(
                "Aggregation must consume the exact edge-message output."
            )

        if stages.residual_update.aggregation is not stages.aggregation:
            raise ValueError(
                "Residual updater must consume the exact aggregation output."
            )

        if stages.residual_update.layer_inputs is not layer_inputs:
            raise ValueError(
                "Residual updater must preserve the exact layer_inputs."
            )

        if (
            stages.normalization.residual_update
            is not stages.residual_update
        ):
            raise ValueError(
                "Normalizer must consume the exact residual-update output."
            )

        validate_layer_residual_update_output(
            output=stages.residual_update,
            aggregation=stages.aggregation,
            layer_inputs=layer_inputs,
            residual_mode=(
                self.residual_updater
                .residual_mode
            ),
            dropout_probability=(
                self.residual_updater
                .dropout_probability
            ),
            training=self.training,
            updater_architecture_fingerprint=(
                self.residual_updater
                .architecture_fingerprint()
            ),
        )

        validate_layer_normalization_output(
            output=stages.normalization,
            residual_update=(
                stages.residual_update
            ),
            normalization_mode=(
                self.normalizer
                .normalization_mode
            ),
            normalization_position=(
                self.normalizer
                .normalization_position
            ),
            epsilon=self.normalizer.epsilon,
            weight=self.normalizer.weight,
            bias=self.normalizer.bias,
            normalizer_architecture_fingerprint=(
                self.normalizer
                .architecture_fingerprint()
            ),
            normalizer_parameter_fingerprint=(
                self.normalizer
                .parameter_fingerprint()
            ),
        )

    # ------------------------------------------------------------------
    # Trace, internal output, and lineage assembly
    # ------------------------------------------------------------------

    def build_trace(
        self,
        *,
        layer_inputs: FunctionalMessagePassingLayerInputs,
        edge_stages: FunctionalMessagePassingLayerEdgeStages,
        node_stages: FunctionalMessagePassingLayerNodeStages,
    ) -> FunctionalMessagePassingLayerTrace | None:
        policy = layer_inputs.trace_policy

        if policy.mode == LAYER_TRACE_NONE:
            return None

        if policy.mode == LAYER_TRACE_FULL:
            return FunctionalMessagePassingLayerTrace(
                layer_inputs=layer_inputs,
                aggregation=(
                    node_stages.aggregation
                ),
                residual_update=(
                    node_stages.residual_update
                ),
                normalization=(
                    node_stages.normalization
                ),
                relation_transform=(
                    edge_stages.relation_transform
                ),
                edge_normalization=(
                    edge_stages.edge_normalization
                ),
                relation_gate=(
                    edge_stages.relation_gate
                ),
                edge_attention=(
                    edge_stages.edge_attention
                ),
                edge_messages=(
                    edge_stages
                    .message_builder_run
                    .public_output
                ),
                message_builder_run=(
                    edge_stages
                    .message_builder_run
                ),
            )

        return FunctionalMessagePassingLayerTrace(
            layer_inputs=layer_inputs,
            aggregation=(
                node_stages.aggregation
            ),
            residual_update=(
                node_stages.residual_update
            ),
            normalization=(
                node_stages.normalization
            ),
        )

    def _lineage_fingerprint(
        self,
        *,
        layer_inputs: FunctionalMessagePassingLayerInputs,
        edge_stages: FunctionalMessagePassingLayerEdgeStages,
        node_stages: FunctionalMessagePassingLayerNodeStages,
        trace: FunctionalMessagePassingLayerTrace | None,
        parameter_fingerprint: str,
    ) -> str:
        payload = {
            "schema_version": (
                FUNCTIONAL_MESSAGE_PASSING_LAYER_SCHEMA_VERSION
            ),
            "layer_inputs_lineage_fingerprint": (
                layer_inputs
                .lineage_fingerprint()
            ),
            "source_inputs_lineage_fingerprint": (
                layer_inputs
                .source_inputs
                .lineage_fingerprint()
            ),
            "layer_architecture_fingerprint": (
                self.architecture_fingerprint()
            ),
            "layer_parameter_fingerprint": (
                parameter_fingerprint
            ),
            "relation_transform": {
                "schema_version": (
                    edge_stages
                    .relation_transform
                    .schema_version
                ),
                "architecture_fingerprint": (
                    edge_stages
                    .relation_transform
                    .encoder_architecture_fingerprint
                ),
                "parameter_fingerprint": (
                    edge_stages
                    .relation_transform
                    .parameter_fingerprint
                ),
            },
            "edge_normalization": {
                "schema_version": (
                    edge_stages
                    .edge_normalization
                    .schema_version
                ),
                "architecture_fingerprint": (
                    edge_stages
                    .edge_normalization
                    .encoder_architecture_fingerprint
                ),
            },
            "relation_gate": (
                {
                    "schema_version": (
                        edge_stages
                        .relation_gate
                        .schema_version
                    ),
                    "architecture_fingerprint": (
                        edge_stages
                        .relation_gate
                        .encoder_architecture_fingerprint
                    ),
                    "parameter_fingerprint": (
                        edge_stages
                        .relation_gate
                        .parameter_fingerprint
                    ),
                }
                if edge_stages.relation_gate is not None
                else None
            ),
            "edge_attention": (
                {
                    "schema_version": (
                        edge_stages
                        .edge_attention
                        .schema_version
                    ),
                    "architecture_fingerprint": (
                        edge_stages
                        .edge_attention
                        .encoder_architecture_fingerprint
                    ),
                    "parameter_fingerprint": (
                        edge_stages
                        .edge_attention
                        .parameter_fingerprint
                    ),
                }
                if edge_stages.edge_attention is not None
                else None
            ),
            "edge_messages": {
                "schema_version": (
                    edge_stages
                    .message_builder_run
                    .public_output
                    .schema_version
                ),
                "architecture_fingerprint": (
                    edge_stages
                    .message_builder_run
                    .public_output
                    .encoder_architecture_fingerprint
                ),
            },
            "aggregation": {
                "schema_version": (
                    node_stages
                    .aggregation
                    .schema_version
                ),
                "architecture_fingerprint": (
                    node_stages
                    .aggregation
                    .encoder_architecture_fingerprint
                ),
            },
            "residual_update_lineage_fingerprint": (
                node_stages
                .residual_update
                .lineage_fingerprint()
            ),
            "normalization_lineage_fingerprint": (
                node_stages
                .normalization
                .lineage_fingerprint()
            ),
            "trace_lineage_fingerprint": (
                trace.lineage_fingerprint()
                if trace is not None
                else None
            ),
        }

        return _fingerprint(
            payload
        )

    def build_internal_output(
        self,
        *,
        layer_inputs: FunctionalMessagePassingLayerInputs,
        edge_stages: FunctionalMessagePassingLayerEdgeStages,
        node_stages: FunctionalMessagePassingLayerNodeStages,
    ) -> LayerComputationOutput:
        trace = self.build_trace(
            layer_inputs=layer_inputs,
            edge_stages=edge_stages,
            node_stages=node_stages,
        )
        parameter_fingerprint = (
            self.parameter_fingerprint()
        )
        lineage_fingerprint = (
            self._lineage_fingerprint(
                layer_inputs=layer_inputs,
                edge_stages=edge_stages,
                node_stages=node_stages,
                trace=trace,
                parameter_fingerprint=(
                    parameter_fingerprint
                ),
            )
        )
        regularization_terms = (
            _merge_regularization_terms(
                edge_stages.relation_gate
            )
        )

        output = LayerComputationOutput(
            updated_node_state=(
                node_stages
                .normalization
                .output_state
            ),
            layer_inputs=layer_inputs,
            aggregation=(
                node_stages.aggregation
            ),
            residual_update=(
                node_stages.residual_update
            ),
            normalization=(
                node_stages.normalization
            ),
            layer_architecture_fingerprint=(
                self.architecture_fingerprint()
            ),
            layer_parameter_fingerprint=(
                parameter_fingerprint
            ),
            lineage_fingerprint=(
                lineage_fingerprint
            ),
            trace=trace,
            regularization_terms=(
                regularization_terms
            ),
        )

        validate_layer_stage_chain(
            layer_inputs=layer_inputs,
            aggregation=(
                node_stages.aggregation
            ),
            residual_update=(
                node_stages.residual_update
            ),
            normalization=(
                node_stages.normalization
            ),
            computation_output=output,
        )

        return output

    # ------------------------------------------------------------------
    # Complete execution
    # ------------------------------------------------------------------

    def run_from_layer_inputs(
        self,
        layer_inputs: FunctionalMessagePassingLayerInputs,
    ) -> FunctionalMessagePassingLayerRun:
        """
        Execute one complete layer using a preconstructed runtime input object.
        """

        if not isinstance(
            layer_inputs,
            FunctionalMessagePassingLayerInputs,
        ):
            raise TypeError(
                "layer_inputs must be a "
                "FunctionalMessagePassingLayerInputs."
            )

        self._validate_source_inputs(
            layer_inputs.source_inputs
        )

        if layer_inputs.training is not self.training:
            raise ValueError(
                "layer_inputs.training must match the layer runtime mode."
            )

        edge_stages = self.compute_edge_stages(
            layer_inputs.source_inputs
        )
        node_stages = self.compute_node_stages(
            edge_stages=edge_stages,
            layer_inputs=layer_inputs,
        )
        internal_output = self.build_internal_output(
            layer_inputs=layer_inputs,
            edge_stages=edge_stages,
            node_stages=node_stages,
        )
        public_output = (
            assemble_functional_message_passing_layer_output(
                internal_output=(
                    internal_output
                )
            )
        )

        run = FunctionalMessagePassingLayerRun(
            layer_inputs=layer_inputs,
            edge_stages=edge_stages,
            node_stages=node_stages,
            internal_output=internal_output,
            public_output=public_output,
        )

        self._validate_owned_run(
            run
        )

        return run

    def run_complete(
        self,
        source_inputs: FunctionalMessagePassingInputs,
        *,
        layer_index: int = 0,
        trace_policy: LayerTracePolicy | str | None = None,
        capture_intermediate_messages: bool | None = None,
        source_stack_fingerprint: str | None = None,
    ) -> FunctionalMessagePassingLayerRun:
        """
        Execute one complete layer and retain the complete audit run.
        """

        layer_inputs = self.build_layer_inputs(
            source_inputs,
            layer_index=layer_index,
            trace_policy=trace_policy,
            capture_intermediate_messages=(
                capture_intermediate_messages
            ),
            source_stack_fingerprint=(
                source_stack_fingerprint
            ),
        )

        return self.run_from_layer_inputs(
            layer_inputs
        )

    def _validate_owned_run(
        self,
        run: FunctionalMessagePassingLayerRun,
    ) -> None:
        validate_functional_message_passing_layer_run(
            run
        )

        if (
            run.internal_output
            .layer_architecture_fingerprint
            != self.architecture_fingerprint()
        ):
            raise ValueError(
                "Internal output architecture fingerprint differs from the "
                "owning layer."
            )

        if (
            run.internal_output
            .layer_parameter_fingerprint
            != self.parameter_fingerprint()
        ):
            raise ValueError(
                "Internal output parameter fingerprint differs from the "
                "owning layer."
            )

        self._validate_edge_stages(
            run.edge_stages,
            source_inputs=run.source_inputs,
        )
        self._validate_node_stages(
            run.node_stages,
            edge_stages=run.edge_stages,
            layer_inputs=run.layer_inputs,
        )

    def forward(
        self,
        source_inputs: FunctionalMessagePassingInputs,
        *,
        layer_index: int = 0,
        trace_policy: LayerTracePolicy | str | None = None,
        capture_intermediate_messages: bool | None = None,
        source_stack_fingerprint: str | None = None,
    ) -> FunctionalMessagePassingLayerOutput:
        """
        Return only the public output for ordinary model execution.
        """

        return self.run_complete(
            source_inputs,
            layer_index=layer_index,
            trace_policy=trace_policy,
            capture_intermediate_messages=(
                capture_intermediate_messages
            ),
            source_stack_fingerprint=(
                source_stack_fingerprint
            ),
        ).public_output

    # ------------------------------------------------------------------
    # Explicit diagnostics
    # ------------------------------------------------------------------

    def diagnostic_report(
        self,
        *,
        run: FunctionalMessagePassingLayerRun,
    ) -> dict[str, Any]:
        """
        Produce a tensor-free report for an already completed exact run.
        """

        if not isinstance(
            run,
            FunctionalMessagePassingLayerRun,
        ):
            raise TypeError(
                "run must be a FunctionalMessagePassingLayerRun."
            )

        if self.diagnostics is None:
            raise RuntimeError(
                "Diagnostics are not configured for this "
                "FunctionalMessagePassingLayer."
            )

        self._validate_owned_run(
            run
        )

        return self.diagnostics.public_report(
            public_output=(
                run.public_output
            ),
            internal_output=(
                run.internal_output
            ),
        )

    def forward_with_diagnostics(
        self,
        source_inputs: FunctionalMessagePassingInputs,
        *,
        layer_index: int = 0,
        trace_policy: LayerTracePolicy | str | None = None,
        capture_intermediate_messages: bool | None = None,
        source_stack_fingerprint: str | None = None,
    ) -> FunctionalMessagePassingLayerRunWithDiagnostics:
        """
        Execute one complete layer and explicitly generate diagnostics.
        """

        run = self.run_complete(
            source_inputs,
            layer_index=layer_index,
            trace_policy=trace_policy,
            capture_intermediate_messages=(
                capture_intermediate_messages
            ),
            source_stack_fingerprint=(
                source_stack_fingerprint
            ),
        )
        report = self.diagnostic_report(
            run=run
        )

        return FunctionalMessagePassingLayerRunWithDiagnostics(
            run=run,
            diagnostic_report=report,
        )

    def extra_repr(self) -> str:
        return (
            f"hidden_dim={self.hidden_dim}, "
            f"num_relations={self.num_relations}, "
            f"gate_enabled={self.gate_enabled}, "
            f"attention_enabled={self.attention_enabled}, "
            f"residual_enabled={self.residual_enabled}, "
            f"layer_norm_enabled={self.layer_norm_enabled}, "
            f"semantic_edge_policy={self.semantic_edge_policy!r}, "
            f"default_trace_mode={self.default_trace_policy.mode!r}, "
            f"diagnostics_enabled={self.diagnostics_enabled}, "
            "layer_index_runtime_supplied=True, "
            "stacking_owned_here=False"
        )


# =============================================================================
# Component validation helper
# =============================================================================


def _require_layer_components(
    *,
    relation_transforms: RelationTransforms,
    edge_normalization: EdgeNormalization,
    relation_gate: RelationFamilyGate | None,
    edge_attention: EdgeAttention | None,
    message_builder: EdgeMessageBuilder,
    aggregator: MessageAggregator,
    residual_updater: LayerResidualUpdater,
    normalizer: LayerNormalizer,
    diagnostics: LayerDiagnostics | None,
) -> None:
    if not isinstance(
        relation_transforms,
        RelationTransforms,
    ):
        raise TypeError(
            "relation_transforms must be a RelationTransforms."
        )

    if not isinstance(
        edge_normalization,
        EdgeNormalization,
    ):
        raise TypeError(
            "edge_normalization must be an EdgeNormalization."
        )

    if (
        relation_gate is not None
        and not isinstance(
            relation_gate,
            RelationFamilyGate,
        )
    ):
        raise TypeError(
            "relation_gate must be a RelationFamilyGate or None."
        )

    if (
        edge_attention is not None
        and not isinstance(
            edge_attention,
            EdgeAttention,
        )
    ):
        raise TypeError(
            "edge_attention must be an EdgeAttention or None."
        )

    if not isinstance(
        message_builder,
        EdgeMessageBuilder,
    ):
        raise TypeError(
            "message_builder must be an EdgeMessageBuilder."
        )

    if not isinstance(
        aggregator,
        MessageAggregator,
    ):
        raise TypeError(
            "aggregator must be a MessageAggregator."
        )

    if not isinstance(
        residual_updater,
        LayerResidualUpdater,
    ):
        raise TypeError(
            "residual_updater must be a LayerResidualUpdater."
        )

    if not isinstance(
        normalizer,
        LayerNormalizer,
    ):
        raise TypeError(
            "normalizer must be a LayerNormalizer."
        )

    if (
        diagnostics is not None
        and not isinstance(
            diagnostics,
            LayerDiagnostics,
        )
    ):
        raise TypeError(
            "diagnostics must be a LayerDiagnostics or None."
        )


# =============================================================================
# Functional builders and execution helpers
# =============================================================================


def build_functional_message_passing_layer(
    *,
    relation_transforms: RelationTransforms,
    edge_normalization: EdgeNormalization,
    relation_gate: RelationFamilyGate | None,
    edge_attention: EdgeAttention | None,
    message_builder: EdgeMessageBuilder,
    aggregator: MessageAggregator,
    residual_updater: LayerResidualUpdater,
    normalizer: LayerNormalizer,
    default_trace_policy: LayerTracePolicy | None = None,
    diagnostics: LayerDiagnostics | None = None,
) -> FunctionalMessagePassingLayer:
    """
    Construct one layer from explicit existing components.
    """

    return FunctionalMessagePassingLayer(
        relation_transforms=relation_transforms,
        edge_normalization=edge_normalization,
        relation_gate=relation_gate,
        edge_attention=edge_attention,
        message_builder=message_builder,
        aggregator=aggregator,
        residual_updater=residual_updater,
        normalizer=normalizer,
        default_trace_policy=(
            default_trace_policy
        ),
        diagnostics=diagnostics,
    )


def build_functional_message_passing_layer_from_config(
    *,
    config: FunctionalMessagePassingConfig,
    relation_config: RelationConfig,
    source_inputs: FunctionalMessagePassingInputs,
    semantic_edge_policy: str = (
        MESSAGE_SEMANTIC_EDGE_POLICY_IGNORE
    ),
    diagnostics_enabled: bool = False,
    **kwargs: Any,
) -> FunctionalMessagePassingLayer:
    """
    Functional spelling for ``FunctionalMessagePassingLayer.from_config``.
    """

    return FunctionalMessagePassingLayer.from_config(
        config=config,
        relation_config=relation_config,
        source_inputs=source_inputs,
        semantic_edge_policy=(
            semantic_edge_policy
        ),
        diagnostics_enabled=(
            diagnostics_enabled
        ),
        **kwargs,
    )


def run_functional_message_passing_layer(
    layer: FunctionalMessagePassingLayer,
    source_inputs: FunctionalMessagePassingInputs,
    *,
    layer_index: int = 0,
    trace_policy: LayerTracePolicy | str | None = None,
    capture_intermediate_messages: bool | None = None,
    source_stack_fingerprint: str | None = None,
) -> FunctionalMessagePassingLayerOutput:
    """
    Functional spelling for the ordinary public forward pass.
    """

    if not isinstance(
        layer,
        FunctionalMessagePassingLayer,
    ):
        raise TypeError(
            "layer must be a FunctionalMessagePassingLayer."
        )

    return layer(
        source_inputs,
        layer_index=layer_index,
        trace_policy=trace_policy,
        capture_intermediate_messages=(
            capture_intermediate_messages
        ),
        source_stack_fingerprint=(
            source_stack_fingerprint
        ),
    )


def run_functional_message_passing_layer_complete(
    layer: FunctionalMessagePassingLayer,
    source_inputs: FunctionalMessagePassingInputs,
    *,
    layer_index: int = 0,
    trace_policy: LayerTracePolicy | str | None = None,
    capture_intermediate_messages: bool | None = None,
    source_stack_fingerprint: str | None = None,
) -> FunctionalMessagePassingLayerRun:
    """
    Functional spelling for a complete auditable layer run.
    """

    if not isinstance(
        layer,
        FunctionalMessagePassingLayer,
    ):
        raise TypeError(
            "layer must be a FunctionalMessagePassingLayer."
        )

    return layer.run_complete(
        source_inputs,
        layer_index=layer_index,
        trace_policy=trace_policy,
        capture_intermediate_messages=(
            capture_intermediate_messages
        ),
        source_stack_fingerprint=(
            source_stack_fingerprint
        ),
    )


# =============================================================================
# Compact aliases
# =============================================================================


HazardConditionedFunctionalMessagePassingLayer = (
    FunctionalMessagePassingLayer
)
FunctionalLayer = FunctionalMessagePassingLayer
MessagePassingLayer = FunctionalMessagePassingLayer

LayerEdgeStages = FunctionalMessagePassingLayerEdgeStages
LayerNodeStages = FunctionalMessagePassingLayerNodeStages
LayerRun = FunctionalMessagePassingLayerRun
LayerRunWithDiagnostics = (
    FunctionalMessagePassingLayerRunWithDiagnostics
)

build_layer = (
    build_functional_message_passing_layer
)
build_layer_from_config = (
    build_functional_message_passing_layer_from_config
)
run_layer = (
    run_functional_message_passing_layer
)
run_layer_complete = (
    run_functional_message_passing_layer_complete
)


__all__ = (
    # Public identity.
    "FUNCTIONAL_MESSAGE_PASSING_LAYER_SCHEMA_VERSION",
    "FUNCTIONAL_MESSAGE_PASSING_LAYER_OPERATION_ORDER",
    "FUNCTIONAL_MESSAGE_PASSING_LAYER_SCIENTIFIC_INTERPRETATION",
    "FUNCTIONAL_MESSAGE_PASSING_LAYER_OUTPUT_SCHEMA",
    "FUNCTIONAL_MESSAGE_PASSING_LAYER_AGGREGATION_ORCHESTRATED_HERE",
    "FUNCTIONAL_MESSAGE_PASSING_LAYER_AGGREGATION_MATH_OWNED_HERE",
    "FUNCTIONAL_MESSAGE_PASSING_LAYER_STACKING_OWNED_HERE",
    "FUNCTIONAL_MESSAGE_PASSING_LAYER_PREDICTION_OWNED_HERE",
    "FUNCTIONAL_MESSAGE_PASSING_LAYER_DISABLED_GATE_REPRESENTATION",
    "FUNCTIONAL_MESSAGE_PASSING_LAYER_DISABLED_ATTENTION_REPRESENTATION",
    "FUNCTIONAL_MESSAGE_PASSING_LAYER_UNIFORM_ATTENTION_REPRESENTATION",
    "FUNCTIONAL_MESSAGE_PASSING_LAYER_TRACE_AFFECTS_NUMERICS",
    "FUNCTIONAL_MESSAGE_PASSING_LAYER_DIAGNOSTICS_AFFECT_NUMERICS",
    # Complete execution contracts.
    "FunctionalMessagePassingLayerEdgeStages",
    "LayerEdgeStages",
    "FunctionalMessagePassingLayerNodeStages",
    "LayerNodeStages",
    "FunctionalMessagePassingLayerRun",
    "LayerRun",
    "FunctionalMessagePassingLayerRunWithDiagnostics",
    "LayerRunWithDiagnostics",
    # Public assembly and validation.
    "assemble_functional_message_passing_layer_output",
    "validate_functional_message_passing_layer_run",
    # Main orchestrator.
    "FunctionalMessagePassingLayer",
    "HazardConditionedFunctionalMessagePassingLayer",
    "FunctionalLayer",
    "MessagePassingLayer",
    # Builders.
    "build_functional_message_passing_layer",
    "build_layer",
    "build_functional_message_passing_layer_from_config",
    "build_layer_from_config",
    # Functional execution.
    "run_functional_message_passing_layer",
    "run_layer",
    "run_functional_message_passing_layer_complete",
    "run_layer_complete",
)
