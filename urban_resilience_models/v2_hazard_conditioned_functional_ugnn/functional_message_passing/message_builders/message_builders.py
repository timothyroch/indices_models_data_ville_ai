"""
Complete orchestration of functional edge-message construction.

Target path:
    urban_resilience_models/
        v2_hazard_conditioned_functional_ugnn/
            functional_message_passing/
                message_builders/
                    message_builders.py

This module coordinates the already separated message-builder stages:

    RelationTransformOutput
        + StructuralEdgeNormalizationOutput
        + optional RelationGateOutput
        + optional EdgeAttentionOutput
        + explicit semantic-edge policy
            ↓
    ResolvedMessageCoefficients
            ↓
    MessageCompositionOutput
            ↓
    EdgeMessageOutput

No new equation is introduced here. The orchestrator validates exact
lineage, invokes the parameter-free coefficient resolver and message composer,
and assembles the existing public ``EdgeMessageOutput`` contract.

Frozen message equation
-----------------------
For every stored directed edge ``e``:

    u_e
        = relation_transform.transformed_source_state[e]

    n_e
        = edge_normalization.coefficients[e]

    g_e
        = relation_gate.edge_gate_values[e]
          or exact one when relation gating is disabled

    alpha_e
        = edge_attention.edge_weights[e]
          or exact one when edge attention is disabled

    w_e
        = source_graph.semantic_edge_weight[e]
          when explicitly consumed
          or exact one otherwise

    c_e
        = n_e * g_e * alpha_e * w_e

    m_e
        = u_e * c_e

The scalar coefficient is broadcast only across the hidden-feature axis.

Separation
---------------------
This orchestrator preserves the distinct meanings of:

- exact-relation source transformation;
- graph-structural edge normalization;
- target-node exact-relation gating;
- within-relation concrete-edge attention;
- optional external semantic edge weighting.

It does not reinterpret any factor as another, renormalize the final product,
or collapse factor provenance.

Disabled mechanisms
--------------------
Disabled relation gating and disabled edge attention are represented by
``None`` at the public boundary and contribute exact multiplicative identity
one internally.

Therefore:

    disabled attention != enabled uniform attention

Enabled uniform attention is still a normalized routing mechanism. Disabled
attention contributes no routing coefficient beyond identity one.

Semantic-edge policy
--------------------
The semantic-edge policy is frozen in ``MessageCoefficientResolver``:

``ignore``
    Do not consume the graph semantic-edge tensor. The public output stores
    ``semantic_edge_weight=None``.

``use_source_graph``
    Consume and preserve the exact source-graph semantic-edge tensor.

The policy cannot be changed per forward call. It is part of the numerical
architecture fingerprint.

Diagnostics
-----------
An optional ``MessageBuilderDiagnostics`` component may be attached. It is
never executed by the ordinary ``forward`` method and is deliberately excluded
from the numerical architecture fingerprint. Enabling or disabling reporting
must not change the identity of the numerical model.

Use ``forward_with_diagnostics`` or ``diagnostic_report`` explicitly when a
tensor-free research report is required.

Scope exclusions
----------------
This module does not own:

- relation-transform computation;
- structural-normalization computation;
- relation-gate prediction;
- edge-attention scoring or normalization;
- semantic-edge-weight estimation;
- target-node aggregation;
- residual updates;
- dropout;
- layer normalization;
- multi-layer execution;
- causal, counterfactual, or explanation-faithfulness claims.

The complete message-builder subsystem is parameter-free and buffer-free.
Trainable parameter provenance remains attached to upstream relation-transform,
relation-gate, and edge-attention outputs.
"""

from __future__ import annotations

from hashlib import sha256
import json
from typing import Any, Final, Mapping, NamedTuple

from torch import nn

from ..schemas import (
    EdgeAttentionOutput,
    EdgeMessageOutput,
    FunctionalMessagePassingInputs,
    RelationGateOutput,
    RelationTransformOutput,
    StructuralEdgeNormalizationOutput,
)
from .coefficient_resolution import (
    MessageCoefficientResolver,
    build_message_coefficient_resolver,
    validate_resolved_message_coefficients,
)
from .diagnostics import (
    MessageBuilderDiagnostics,
    MessageBuilderDiagnosticThresholds,
    DEFAULT_MESSAGE_BUILDER_DIAGNOSTIC_THRESHOLDS,
    build_message_builder_diagnostics,
)
from .message_composition import (
    EdgeMessageComposer,
    build_edge_message_composer,
    validate_message_composition_output,
)
from .relation_state_gather import (
    RelationStateGather,
)
from .schemas import (
    MESSAGE_COMBINED_COEFFICIENT_FORMULA,
    MESSAGE_COMPOSITION_FORMULA,
    MESSAGE_DISABLED_FACTOR_POLICY,
    MESSAGE_FACTOR_ORDER,
    MESSAGE_SEMANTIC_EDGE_POLICY_IGNORE,
    MessageBuilderStages,
    MessageCompositionOutput,
    ResolvedMessageCoefficients,
    validate_message_builder_stage_chain,
    validate_public_edge_message_output,
)


# =============================================================================
# Public identity
# =============================================================================


MESSAGE_BUILDERS_ORCHESTRATOR_SCHEMA_VERSION: Final[str] = "0.1"

MESSAGE_BUILDERS_OPERATION_ORDER: Final[
    tuple[str, ...]
] = (
    "validate_exact_upstream_source_input_lineage",
    "resolve_explicit_scalar_edge_message_coefficients",
    "validate_resolved_coefficient_stage",
    "compose_edge_aligned_message_vectors",
    "validate_message_composition_stage",
    "assemble_public_edge_message_output",
    "validate_exact_public_output_lineage",
)

MESSAGE_BUILDERS_SCIENTIFIC_INTERPRETATION: Final[str] = (
    "hierarchical_functional_edge_message_construction"
)

MESSAGE_BUILDERS_DISABLED_GATE_REPRESENTATION: Final[str] = (
    "None_publicly_and_exact_identity_one_internally"
)

MESSAGE_BUILDERS_DISABLED_ATTENTION_REPRESENTATION: Final[str] = (
    "None_publicly_and_exact_identity_one_internally"
)

MESSAGE_BUILDERS_PARAMETER_FREE: Final[bool] = True
MESSAGE_BUILDERS_BUFFER_FREE: Final[bool] = True
MESSAGE_BUILDERS_AGGREGATION_OWNED_HERE: Final[bool] = False

MESSAGE_BUILDERS_OUTPUT_SCHEMA: Final[str] = "EdgeMessageOutput"


# =============================================================================
# Small run contracts
# =============================================================================


class MessageBuilderRun(NamedTuple):
    """
    Exact complete internal and public stage chain.

    The tuple preserves source-object identity and introduces no copied tensors.
    """

    resolved_coefficients: ResolvedMessageCoefficients
    composition_output: MessageCompositionOutput
    public_output: EdgeMessageOutput


class MessageBuilderRunWithDiagnostics(NamedTuple):
    """
    Complete message-builder run plus an explicitly requested tensor-free report.
    """

    public_output: EdgeMessageOutput
    diagnostic_report: dict[str, Any]


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


def _require_nonempty_string(
    name: str,
    value: str,
) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(
            f"{name} must be a non-empty string."
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


def _require_relation_transform(
    relation_transform: RelationTransformOutput,
) -> None:
    if not isinstance(
        relation_transform,
        RelationTransformOutput,
    ):
        raise TypeError(
            "relation_transform must be a "
            "RelationTransformOutput."
        )


def _require_edge_normalization(
    edge_normalization: StructuralEdgeNormalizationOutput,
) -> None:
    if not isinstance(
        edge_normalization,
        StructuralEdgeNormalizationOutput,
    ):
        raise TypeError(
            "edge_normalization must be a "
            "StructuralEdgeNormalizationOutput."
        )


def _require_optional_relation_gate(
    relation_gate: RelationGateOutput | None,
) -> None:
    if (
        relation_gate is not None
        and not isinstance(
            relation_gate,
            RelationGateOutput,
        )
    ):
        raise TypeError(
            "relation_gate must be a RelationGateOutput or None."
        )


def _require_optional_edge_attention(
    edge_attention: EdgeAttentionOutput | None,
) -> None:
    if (
        edge_attention is not None
        and not isinstance(
            edge_attention,
            EdgeAttentionOutput,
        )
    ):
        raise TypeError(
            "edge_attention must be an EdgeAttentionOutput or None."
        )


def _require_coefficient_resolver(
    coefficient_resolver: MessageCoefficientResolver,
) -> None:
    if not isinstance(
        coefficient_resolver,
        MessageCoefficientResolver,
    ):
        raise TypeError(
            "coefficient_resolver must be a "
            "MessageCoefficientResolver."
        )


def _require_message_composer(
    message_composer: EdgeMessageComposer,
) -> None:
    if not isinstance(
        message_composer,
        EdgeMessageComposer,
    ):
        raise TypeError(
            "message_composer must be an EdgeMessageComposer."
        )


def _require_optional_diagnostics(
    diagnostics: MessageBuilderDiagnostics | None,
) -> None:
    if (
        diagnostics is not None
        and not isinstance(
            diagnostics,
            MessageBuilderDiagnostics,
        )
    ):
        raise TypeError(
            "diagnostics must be a MessageBuilderDiagnostics or None."
        )


def _resolve_source_inputs(
    *,
    relation_transform: RelationTransformOutput,
    edge_normalization: StructuralEdgeNormalizationOutput,
    relation_gate: RelationGateOutput | None,
    edge_attention: EdgeAttentionOutput | None,
    source_inputs: FunctionalMessagePassingInputs | None,
) -> FunctionalMessagePassingInputs:
    """
    Resolve and validate one exact upstream input object.
    """

    _require_relation_transform(
        relation_transform
    )
    _require_edge_normalization(
        edge_normalization
    )
    _require_optional_relation_gate(
        relation_gate
    )
    _require_optional_edge_attention(
        edge_attention
    )

    transform_inputs = (
        relation_transform.source_inputs
    )
    normalization_inputs = (
        edge_normalization.source_inputs
    )

    _require_source_inputs(
        transform_inputs
    )
    _require_source_inputs(
        normalization_inputs
    )

    if transform_inputs is not normalization_inputs:
        raise ValueError(
            "relation_transform and edge_normalization must share the "
            "exact same FunctionalMessagePassingInputs object."
        )

    if (
        relation_gate is not None
        and relation_gate.source_inputs
        is not transform_inputs
    ):
        raise ValueError(
            "relation_gate must share the exact same "
            "FunctionalMessagePassingInputs object as relation_transform."
        )

    if (
        edge_attention is not None
        and edge_attention.source_inputs
        is not transform_inputs
    ):
        raise ValueError(
            "edge_attention must share the exact same "
            "FunctionalMessagePassingInputs object as relation_transform."
        )

    if source_inputs is not None:
        _require_source_inputs(
            source_inputs
        )

        if transform_inputs is not source_inputs:
            raise ValueError(
                "All message-builder inputs must reference the exact "
                "supplied source_inputs object."
            )

    return transform_inputs


# =============================================================================
# Static public-output assembly
# =============================================================================


def assemble_edge_message_output(
    *,
    composition_output: MessageCompositionOutput,
    encoder_architecture_fingerprint: str,
) -> EdgeMessageOutput:
    """
    Assemble the existing public ``EdgeMessageOutput`` contract.

    No tensors are recomputed. Every source object and tensor is preserved by
    exact identity.
    """

    if not isinstance(
        composition_output,
        MessageCompositionOutput,
    ):
        raise TypeError(
            "composition_output must be a "
            "MessageCompositionOutput."
        )

    _require_nonempty_string(
        "encoder_architecture_fingerprint",
        encoder_architecture_fingerprint,
    )

    resolved = (
        composition_output
        .resolved_coefficients
    )

    validate_message_builder_stage_chain(
        resolved_coefficients=resolved,
        composition_output=composition_output,
    )
    validate_resolved_message_coefficients(
        output=resolved,
        source_inputs=(
            composition_output
            .source_inputs
        ),
    )
    validate_message_composition_output(
        output=composition_output,
        relation_transform=(
            composition_output
            .relation_transform
        ),
        resolved_coefficients=resolved,
        source_inputs=(
            composition_output
            .source_inputs
        ),
    )

    output = EdgeMessageOutput(
        edge_messages=(
            composition_output
            .edge_messages
        ),
        relation_transform=(
            composition_output
            .relation_transform
        ),
        edge_normalization=(
            resolved.edge_normalization
        ),
        relation_gate=(
            resolved.relation_gate
        ),
        edge_attention=(
            resolved.edge_attention
        ),
        semantic_edge_weight=(
            resolved.semantic_edge_weight
        ),
        encoder_architecture_fingerprint=(
            encoder_architecture_fingerprint
        ),
    )

    validate_public_edge_message_output(
        public_output=output,
        composition_output=composition_output,
    )

    return output


def validate_complete_message_builder_run(
    *,
    resolved_coefficients: ResolvedMessageCoefficients,
    composition_output: MessageCompositionOutput,
    public_output: EdgeMessageOutput,
    source_inputs: FunctionalMessagePassingInputs | None = None,
    encoder_architecture_fingerprint: str | None = None,
) -> None:
    """
    Validate exact identity across all internal and public message stages.
    """

    if not isinstance(
        resolved_coefficients,
        ResolvedMessageCoefficients,
    ):
        raise TypeError(
            "resolved_coefficients must be a "
            "ResolvedMessageCoefficients."
        )

    if not isinstance(
        composition_output,
        MessageCompositionOutput,
    ):
        raise TypeError(
            "composition_output must be a "
            "MessageCompositionOutput."
        )

    if not isinstance(
        public_output,
        EdgeMessageOutput,
    ):
        raise TypeError(
            "public_output must be an EdgeMessageOutput."
        )

    validate_message_builder_stage_chain(
        resolved_coefficients=(
            resolved_coefficients
        ),
        composition_output=(
            composition_output
        ),
    )
    validate_public_edge_message_output(
        public_output=public_output,
        composition_output=composition_output,
    )

    resolved_inputs = (
        composition_output.source_inputs
    )

    if source_inputs is not None:
        _require_source_inputs(
            source_inputs
        )

        if resolved_inputs is not source_inputs:
            raise ValueError(
                "The complete message-builder run must preserve the "
                "exact expected source_inputs object."
            )

    if (
        resolved_coefficients.source_inputs
        is not resolved_inputs
    ):
        raise ValueError(
            "resolved_coefficients lost exact source-input lineage."
        )

    if public_output.source_inputs is not (
        resolved_inputs
    ):
        raise ValueError(
            "public_output lost exact source-input lineage."
        )

    if (
        public_output.edge_messages
        is not composition_output.edge_messages
    ):
        raise ValueError(
            "public_output must preserve the exact composed "
            "edge_messages tensor object."
        )

    if encoder_architecture_fingerprint is not None:
        _require_nonempty_string(
            "encoder_architecture_fingerprint",
            encoder_architecture_fingerprint,
        )

        if (
            public_output
            .encoder_architecture_fingerprint
            != encoder_architecture_fingerprint
        ):
            raise ValueError(
                "public_output architecture fingerprint differs from "
                "the expected message-builder architecture."
            )


# =============================================================================
# Complete orchestrator
# =============================================================================


class EdgeMessageBuilder(nn.Module):
    """
    Coordinate coefficient resolution, message composition, and public output.

    Parameters
    ----------
    coefficient_resolver:
        Parameter-free scalar-factor resolver with one frozen semantic-edge
        policy.
    message_composer:
        Parameter-free edge-vector composer. Its embedded
        ``RelationStateGather`` owns the zero-copy transformed-state boundary.
    diagnostics:
        Optional parameter-free report generator. It is excluded from the
        numerical architecture fingerprint and is never run implicitly.
    """

    coefficient_resolver: MessageCoefficientResolver
    message_composer: EdgeMessageComposer
    diagnostics: MessageBuilderDiagnostics | None

    def __init__(
        self,
        *,
        coefficient_resolver: MessageCoefficientResolver,
        message_composer: EdgeMessageComposer,
        diagnostics: MessageBuilderDiagnostics | None = None,
    ) -> None:
        super().__init__()

        _require_coefficient_resolver(
            coefficient_resolver
        )
        _require_message_composer(
            message_composer
        )
        _require_optional_diagnostics(
            diagnostics
        )

        coefficient_resolver.assert_parameter_free()
        message_composer.assert_parameter_free()

        if diagnostics is not None:
            diagnostics.assert_parameter_free()

        self.coefficient_resolver = (
            coefficient_resolver
        )
        self.message_composer = (
            message_composer
        )
        self.diagnostics = diagnostics

        self._assert_component_contract()

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    @classmethod
    def from_policy(
        cls,
        *,
        semantic_edge_policy: str = (
            MESSAGE_SEMANTIC_EDGE_POLICY_IGNORE
        ),
        diagnostics_enabled: bool = False,
        diagnostics_include_per_relation: bool = True,
        diagnostics_include_per_graph: bool = True,
        diagnostic_thresholds: MessageBuilderDiagnosticThresholds = (
            DEFAULT_MESSAGE_BUILDER_DIAGNOSTIC_THRESHOLDS
        ),
    ) -> "EdgeMessageBuilder":
        """
        Construct the complete bounded message-builder subsystem.
        """

        if not isinstance(
            diagnostics_enabled,
            bool,
        ):
            raise TypeError(
                "diagnostics_enabled must be boolean."
            )

        coefficient_resolver = (
            build_message_coefficient_resolver(
                semantic_edge_policy=(
                    semantic_edge_policy
                )
            )
        )
        message_composer = (
            build_edge_message_composer()
        )
        diagnostics = (
            build_message_builder_diagnostics(
                include_per_relation=(
                    diagnostics_include_per_relation
                ),
                include_per_graph=(
                    diagnostics_include_per_graph
                ),
                thresholds=(
                    diagnostic_thresholds
                ),
            )
            if diagnostics_enabled
            else None
        )

        return cls(
            coefficient_resolver=(
                coefficient_resolver
            ),
            message_composer=(
                message_composer
            ),
            diagnostics=diagnostics,
        )

    # ------------------------------------------------------------------
    # Public component identity
    # ------------------------------------------------------------------

    @property
    def semantic_edge_policy(
        self,
    ) -> str:
        return (
            self
            .coefficient_resolver
            .semantic_edge_policy
        )

    @property
    def relation_state_gather(
        self,
    ) -> RelationStateGather:
        return (
            self
            .message_composer
            .relation_state_gather
        )

    @property
    def diagnostics_enabled(
        self,
    ) -> bool:
        return self.diagnostics is not None

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
        """
        The message-builder subsystem is parameter-free.

        Trainable parameter provenance remains attached to upstream stage
        outputs.
        """

        return None

    # ------------------------------------------------------------------
    # Architecture and provenance
    # ------------------------------------------------------------------

    def numerical_architecture_dict(
        self,
    ) -> dict[str, Any]:
        """
        Return only the numerical model architecture.

        Diagnostics configuration is intentionally excluded so report settings
        cannot change model identity.
        """

        return {
            "schema_version": (
                MESSAGE_BUILDERS_ORCHESTRATOR_SCHEMA_VERSION
            ),
            "scientific_interpretation": (
                MESSAGE_BUILDERS_SCIENTIFIC_INTERPRETATION
            ),
            "operation_order": list(
                MESSAGE_BUILDERS_OPERATION_ORDER
            ),
            "factor_order": list(
                MESSAGE_FACTOR_ORDER
            ),
            "combined_coefficient_formula": (
                MESSAGE_COMBINED_COEFFICIENT_FORMULA
            ),
            "message_composition_formula": (
                MESSAGE_COMPOSITION_FORMULA
            ),
            "semantic_edge_policy": (
                self.semantic_edge_policy
            ),
            "disabled_gate_representation": (
                MESSAGE_BUILDERS_DISABLED_GATE_REPRESENTATION
            ),
            "disabled_attention_representation": (
                MESSAGE_BUILDERS_DISABLED_ATTENTION_REPRESENTATION
            ),
            "coefficient_resolver": (
                self
                .coefficient_resolver
                .architecture_dict()
            ),
            "message_composer": (
                self
                .message_composer
                .architecture_dict()
            ),
            "output_schema": (
                MESSAGE_BUILDERS_OUTPUT_SCHEMA
            ),
            "parameter_free": (
                MESSAGE_BUILDERS_PARAMETER_FREE
            ),
            "buffer_free": (
                MESSAGE_BUILDERS_BUFFER_FREE
            ),
            "aggregation_owned_here": (
                MESSAGE_BUILDERS_AGGREGATION_OWNED_HERE
            ),
            "residual_update_owned_here": False,
            "dropout_owned_here": False,
            "layer_normalization_owned_here": False,
            "claims_causal_importance": False,
            "claims_explanation_faithfulness": False,
        }

    def architecture_dict(
        self,
    ) -> dict[str, Any]:
        """
        Alias for the numerical architecture used in public output identity.
        """

        return self.numerical_architecture_dict()

    def architecture_fingerprint(
        self,
    ) -> str:
        return _fingerprint(
            self.numerical_architecture_dict()
        )

    def diagnostics_architecture_dict(
        self,
    ) -> dict[str, Any] | None:
        if self.diagnostics is None:
            return None

        return self.diagnostics.architecture_dict()

    def full_runtime_dict(
        self,
    ) -> dict[str, Any]:
        """
        Return numerical architecture plus non-numerical reporting settings.
        """

        return {
            "numerical_architecture": (
                self.numerical_architecture_dict()
            ),
            "numerical_architecture_fingerprint": (
                self.architecture_fingerprint()
            ),
            "diagnostics_enabled": (
                self.diagnostics_enabled
            ),
            "diagnostics_architecture": (
                self.diagnostics_architecture_dict()
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

    # ------------------------------------------------------------------
    # Component validation
    # ------------------------------------------------------------------

    def _assert_component_contract(
        self,
    ) -> None:
        self.coefficient_resolver.assert_parameter_free()
        self.message_composer.assert_parameter_free()

        if self.diagnostics is not None:
            self.diagnostics.assert_parameter_free()

        if not isinstance(
            self.relation_state_gather,
            RelationStateGather,
        ):
            raise TypeError(
                "message_composer must own a RelationStateGather."
            )

        if (
            self
            .coefficient_resolver
            .semantic_edge_policy
            != self.semantic_edge_policy
        ):
            raise RuntimeError(
                "Message-builder semantic-edge policy is internally "
                "inconsistent."
            )

        self.assert_parameter_free()

    def assert_parameter_free(
        self,
    ) -> None:
        self.coefficient_resolver.assert_parameter_free()
        self.message_composer.assert_parameter_free()

        if self.diagnostics is not None:
            self.diagnostics.assert_parameter_free()

        parameters = tuple(
            self.named_parameters()
        )
        buffers = tuple(
            self.named_buffers()
        )
        state = self.state_dict()

        if parameters:
            raise RuntimeError(
                "EdgeMessageBuilder must remain parameter-free. "
                f"Observed parameters: "
                f"{tuple(name for name, _ in parameters)}."
            )

        if buffers:
            raise RuntimeError(
                "EdgeMessageBuilder must remain buffer-free. "
                f"Observed buffers: "
                f"{tuple(name for name, _ in buffers)}."
            )

        if state:
            raise RuntimeError(
                "EdgeMessageBuilder must have an empty state_dict."
            )

        if self.parameter_count != 0:
            raise RuntimeError(
                "EdgeMessageBuilder parameter_count must be zero."
            )

        if self.trainable_parameter_count != 0:
            raise RuntimeError(
                "EdgeMessageBuilder trainable_parameter_count must be "
                "zero."
            )

        if self.buffer_count != 0:
            raise RuntimeError(
                "EdgeMessageBuilder buffer_count must be zero."
            )

    # ------------------------------------------------------------------
    # Internal and public execution
    # ------------------------------------------------------------------

    def run_stages(
        self,
        *,
        relation_transform: RelationTransformOutput,
        edge_normalization: StructuralEdgeNormalizationOutput,
        relation_gate: RelationGateOutput | None = None,
        edge_attention: EdgeAttentionOutput | None = None,
        source_inputs: FunctionalMessagePassingInputs | None = None,
    ) -> MessageBuilderStages:
        """
        Execute and return the two immutable internal message-builder stages.
        """

        self._assert_component_contract()

        resolved_inputs = _resolve_source_inputs(
            relation_transform=(
                relation_transform
            ),
            edge_normalization=(
                edge_normalization
            ),
            relation_gate=relation_gate,
            edge_attention=edge_attention,
            source_inputs=source_inputs,
        )

        resolved_coefficients = (
            self.coefficient_resolver(
                edge_normalization=(
                    edge_normalization
                ),
                relation_gate=relation_gate,
                edge_attention=edge_attention,
                source_inputs=resolved_inputs,
            )
        )

        validate_resolved_message_coefficients(
            output=resolved_coefficients,
            source_inputs=resolved_inputs,
            edge_normalization=(
                edge_normalization
            ),
            relation_gate=relation_gate,
            edge_attention=edge_attention,
            semantic_edge_policy=(
                self.semantic_edge_policy
            ),
        )

        composition_output = (
            self.message_composer(
                relation_transform=(
                    relation_transform
                ),
                resolved_coefficients=(
                    resolved_coefficients
                ),
                source_inputs=resolved_inputs,
            )
        )

        validate_message_composition_output(
            output=composition_output,
            relation_transform=(
                relation_transform
            ),
            resolved_coefficients=(
                resolved_coefficients
            ),
            source_inputs=resolved_inputs,
            composer_architecture_fingerprint=(
                self
                .message_composer
                .architecture_fingerprint()
            ),
        )
        validate_message_builder_stage_chain(
            resolved_coefficients=(
                resolved_coefficients
            ),
            composition_output=(
                composition_output
            ),
        )

        return (
            resolved_coefficients,
            composition_output,
        )

    def run_complete(
        self,
        *,
        relation_transform: RelationTransformOutput,
        edge_normalization: StructuralEdgeNormalizationOutput,
        relation_gate: RelationGateOutput | None = None,
        edge_attention: EdgeAttentionOutput | None = None,
        source_inputs: FunctionalMessagePassingInputs | None = None,
    ) -> MessageBuilderRun:
        """
        Execute internal stages and assemble the public output.
        """

        (
            resolved_coefficients,
            composition_output,
        ) = self.run_stages(
            relation_transform=(
                relation_transform
            ),
            edge_normalization=(
                edge_normalization
            ),
            relation_gate=relation_gate,
            edge_attention=edge_attention,
            source_inputs=source_inputs,
        )

        public_output = (
            assemble_edge_message_output(
                composition_output=(
                    composition_output
                ),
                encoder_architecture_fingerprint=(
                    self.architecture_fingerprint()
                ),
            )
        )

        validate_complete_message_builder_run(
            resolved_coefficients=(
                resolved_coefficients
            ),
            composition_output=(
                composition_output
            ),
            public_output=public_output,
            source_inputs=(
                composition_output
                .source_inputs
            ),
            encoder_architecture_fingerprint=(
                self.architecture_fingerprint()
            ),
        )

        return MessageBuilderRun(
            resolved_coefficients=(
                resolved_coefficients
            ),
            composition_output=(
                composition_output
            ),
            public_output=public_output,
        )

    def forward(
        self,
        *,
        relation_transform: RelationTransformOutput,
        edge_normalization: StructuralEdgeNormalizationOutput,
        relation_gate: RelationGateOutput | None = None,
        edge_attention: EdgeAttentionOutput | None = None,
        source_inputs: FunctionalMessagePassingInputs | None = None,
    ) -> EdgeMessageOutput:
        """
        Construct final edge messages under the frozen semantic-edge policy.
        """

        return self.run_complete(
            relation_transform=(
                relation_transform
            ),
            edge_normalization=(
                edge_normalization
            ),
            relation_gate=relation_gate,
            edge_attention=edge_attention,
            source_inputs=source_inputs,
        ).public_output

    # ------------------------------------------------------------------
    # Explicit diagnostics
    # ------------------------------------------------------------------

    def diagnostic_report(
        self,
        *,
        run: MessageBuilderRun,
    ) -> dict[str, Any]:
        """
        Produce a report for an already completed exact run.
        """

        if not isinstance(
            run,
            MessageBuilderRun,
        ):
            raise TypeError(
                "run must be a MessageBuilderRun."
            )

        if self.diagnostics is None:
            raise RuntimeError(
                "Diagnostics are not configured for this "
                "EdgeMessageBuilder."
            )

        self._assert_component_contract()

        validate_complete_message_builder_run(
            resolved_coefficients=(
                run.resolved_coefficients
            ),
            composition_output=(
                run.composition_output
            ),
            public_output=(
                run.public_output
            ),
            source_inputs=(
                run.composition_output
                .source_inputs
            ),
            encoder_architecture_fingerprint=(
                self.architecture_fingerprint()
            ),
        )

        return self.diagnostics.public_report(
            public_output=(
                run.public_output
            ),
            composition_output=(
                run.composition_output
            ),
        )

    def forward_with_diagnostics(
        self,
        *,
        relation_transform: RelationTransformOutput,
        edge_normalization: StructuralEdgeNormalizationOutput,
        relation_gate: RelationGateOutput | None = None,
        edge_attention: EdgeAttentionOutput | None = None,
        source_inputs: FunctionalMessagePassingInputs | None = None,
    ) -> MessageBuilderRunWithDiagnostics:
        """
        Execute one complete run and explicitly generate its report.
        """

        run = self.run_complete(
            relation_transform=(
                relation_transform
            ),
            edge_normalization=(
                edge_normalization
            ),
            relation_gate=relation_gate,
            edge_attention=edge_attention,
            source_inputs=source_inputs,
        )

        report = self.diagnostic_report(
            run=run
        )

        return MessageBuilderRunWithDiagnostics(
            public_output=(
                run.public_output
            ),
            diagnostic_report=report,
        )

    def extra_repr(self) -> str:
        return (
            f"semantic_edge_policy={self.semantic_edge_policy!r}, "
            f"diagnostics_enabled={self.diagnostics_enabled}, "
            f"factor_order={MESSAGE_FACTOR_ORDER!r}, "
            "aggregation_owned_here=False, "
            "parameter_free=True"
        )


# =============================================================================
# Functional builders and execution helpers
# =============================================================================


def build_edge_message_builder(
    *,
    semantic_edge_policy: str = (
        MESSAGE_SEMANTIC_EDGE_POLICY_IGNORE
    ),
    diagnostics_enabled: bool = False,
    diagnostics_include_per_relation: bool = True,
    diagnostics_include_per_graph: bool = True,
    diagnostic_thresholds: MessageBuilderDiagnosticThresholds = (
        DEFAULT_MESSAGE_BUILDER_DIAGNOSTIC_THRESHOLDS
    ),
) -> EdgeMessageBuilder:
    """
    Build the complete parameter-free message-builder subsystem.
    """

    return EdgeMessageBuilder.from_policy(
        semantic_edge_policy=(
            semantic_edge_policy
        ),
        diagnostics_enabled=(
            diagnostics_enabled
        ),
        diagnostics_include_per_relation=(
            diagnostics_include_per_relation
        ),
        diagnostics_include_per_graph=(
            diagnostics_include_per_graph
        ),
        diagnostic_thresholds=(
            diagnostic_thresholds
        ),
    )


def run_edge_message_builder(
    builder: EdgeMessageBuilder,
    *,
    relation_transform: RelationTransformOutput,
    edge_normalization: StructuralEdgeNormalizationOutput,
    relation_gate: RelationGateOutput | None = None,
    edge_attention: EdgeAttentionOutput | None = None,
    source_inputs: FunctionalMessagePassingInputs | None = None,
) -> EdgeMessageOutput:
    """
    Functional spelling for ``EdgeMessageBuilder.forward``.
    """

    if not isinstance(
        builder,
        EdgeMessageBuilder,
    ):
        raise TypeError(
            "builder must be an EdgeMessageBuilder."
        )

    return builder(
        relation_transform=(
            relation_transform
        ),
        edge_normalization=(
            edge_normalization
        ),
        relation_gate=relation_gate,
        edge_attention=edge_attention,
        source_inputs=source_inputs,
    )


def run_edge_message_builder_stages(
    builder: EdgeMessageBuilder,
    *,
    relation_transform: RelationTransformOutput,
    edge_normalization: StructuralEdgeNormalizationOutput,
    relation_gate: RelationGateOutput | None = None,
    edge_attention: EdgeAttentionOutput | None = None,
    source_inputs: FunctionalMessagePassingInputs | None = None,
) -> MessageBuilderStages:
    """
    Functional spelling for the exact internal stage chain.
    """

    if not isinstance(
        builder,
        EdgeMessageBuilder,
    ):
        raise TypeError(
            "builder must be an EdgeMessageBuilder."
        )

    return builder.run_stages(
        relation_transform=(
            relation_transform
        ),
        edge_normalization=(
            edge_normalization
        ),
        relation_gate=relation_gate,
        edge_attention=edge_attention,
        source_inputs=source_inputs,
    )


# =============================================================================
# Compact aliases
# =============================================================================


MessageBuilder = EdgeMessageBuilder
FunctionalMessageBuilder = EdgeMessageBuilder
FunctionalEdgeMessageBuilder = EdgeMessageBuilder

build_message_builder = build_edge_message_builder
build_message_builders = build_edge_message_builder

run_message_builder = run_edge_message_builder
run_message_builder_stages = (
    run_edge_message_builder_stages
)


__all__ = (
    # Public identity.
    "MESSAGE_BUILDERS_ORCHESTRATOR_SCHEMA_VERSION",
    "MESSAGE_BUILDERS_OPERATION_ORDER",
    "MESSAGE_BUILDERS_SCIENTIFIC_INTERPRETATION",
    "MESSAGE_BUILDERS_DISABLED_GATE_REPRESENTATION",
    "MESSAGE_BUILDERS_DISABLED_ATTENTION_REPRESENTATION",
    "MESSAGE_BUILDERS_PARAMETER_FREE",
    "MESSAGE_BUILDERS_BUFFER_FREE",
    "MESSAGE_BUILDERS_AGGREGATION_OWNED_HERE",
    "MESSAGE_BUILDERS_OUTPUT_SCHEMA",
    # Complete run contracts.
    "MessageBuilderRun",
    "MessageBuilderRunWithDiagnostics",
    # Public-output assembly and validation.
    "assemble_edge_message_output",
    "validate_complete_message_builder_run",
    # Orchestrator.
    "EdgeMessageBuilder",
    "MessageBuilder",
    "FunctionalMessageBuilder",
    "FunctionalEdgeMessageBuilder",
    # Builders.
    "build_edge_message_builder",
    "build_message_builder",
    "build_message_builders",
    # Functional execution.
    "run_edge_message_builder",
    "run_message_builder",
    "run_edge_message_builder_stages",
    "run_message_builder_stages",
)
