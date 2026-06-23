"""
Complete exact-relation edge-attention orchestration.

Target path:
    urban_resilience_models/
        v2_hazard_conditioned_functional_ugnn/
            functional_message_passing/
                edge_attention/
                    edge_attention.py

This module composes the three independently tested edge-attention stages:

1. edge-level score prediction;
2. exact target-node/relation grouped normalization;
3. attention-head reduction.

It then constructs the public ``EdgeAttentionOutput`` consumed by later
message construction.

Role
---------------
For each stored directed edge ``e = (s_e -> t_e)`` with exact compiled
relation index ``r_e``:

    raw_score[e, a]
        = score_function(
            source_state[s_e],
            target_state[t_e],
            optional target_hazard_query[t_e],
            exact_relation_identity[r_e],
        )

    attention_weight[e, a]
        = softmax(
            raw_score[:, a]
            within group (t_e, r_e)
        )

    edge_attention[e]
        = mean_a attention_weight[e, a]

The bounded implementation therefore produces a routing coefficient that
answers:

    Within one exact relation mechanism arriving at one target node, how
    should routing mass be distributed across the concrete incoming edges?

It does not answer:

    How active is the relation mechanism itself?

That second question belongs to the independent relation gate. The eventual
message coefficient may combine:

    relation_gate[target, relation]
    * edge_attention[edge]
    * structural_edge_normalization[edge]
    * optional_semantic_edge_weight[edge]

but this module owns only the edge-attention factor.

Hierarchical routing contract
-----------------------------
Attention normalization is performed separately for every exact
target-node/relation group:

    group_id[e]
        = target_index[e] * num_relations
          + edge_relation_index[e]

For every nonempty group and every head:

    sum_{e in group} attention_weight[e, head] = 1

The arithmetic mean across independently normalized heads preserves this
group normalization. Relations do not compete inside edge attention.
Independent sigmoid relation gates may activate multiple relation mechanisms
simultaneously.

Bounded score modes
-------------------
``uniform``
    Exact zero logits, followed by target-relation grouped softmax. The result
    is reciprocal group-size routing. This is enabled uniform attention, not
    disabled attention.

``hazard_blind``
    Learned source-target/exact-relation compatibility without hazard input.
    This is a critical ablation for separating generic learned neighbor
    selection from hazard-conditioned routing.

``hazard_conditioned``
    Learned single-head compatibility using a node-aligned target hazard
    query.

``multihead_hazard_conditioned``
    Independently normalized learned heads followed by equal-weight arithmetic
    mean reduction.

The canonical ``semantic_weight`` mode is not implemented here. Semantic edge
weights are data coefficients with a separate provenance and should not be
silently reinterpreted as attention logits.

Disabled attention
------------------
This module represents *enabled* edge attention only.

When attention is disabled, the higher-level message-passing layer should
store ``edge_attention=None`` and use the multiplicative identity one.
Constructing this module from a configuration with
``attention_enabled=False`` is rejected. This prevents a scientifically
important ambiguity:

    disabled attention != enabled uniform attention

Scope exclusions
----------------
This orchestrator does not own:

- relation-gate computation;
- edge-attribute encoders;
- semantic edge weights;
- structural edge normalization;
- relation transforms;
- edge masking;
- message construction;
- aggregation;
- residual connections;
- layer normalization;
- dropout;
- explanation faithfulness;
- causal interpretation.

Attention values are routing coefficients. They may support descriptive
diagnostics, but they are not automatically causal importance scores.

Auditability
------------
Every stage returns an immutable metadata-bearing schema:

``EdgeAttentionScoreOutput``
    raw finite logits and scorer identity;

``AttentionNormalizationOutput``
    normalized head-level weights, exact group IDs, and dense group counts;

``AttentionHeadReductionOutput``
    final edge-aligned attention weights;

``EdgeAttentionOutput``
    public compact output used by message construction.

The orchestrator verifies exact object lineage between stages before exposing
the final public output. Architecture fingerprints cover all three component
architectures and the stage order. Parameter fingerprints cover the trainable
score function; the normalization and head-reduction stages are required to
remain parameter-free.

Scale considerations
--------------------
Province-scale graph execution should not retain unnecessary intermediate
diagnostic tensors. Ordinary ``forward`` returns only the public output.
``compute_stages`` is available for controlled tests and research audits.
Compact scalar diagnostics are computed only when explicitly requested.

Capability manifests
--------------------
``from_config`` validates canonical configuration but deliberately does not
call ``config.assert_implemented()``. The global implemented-capability
constants should be promoted only after package exports, the complete
orchestrator, and focused integration tests agree.
"""

from __future__ import annotations

from hashlib import sha256
import json
from typing import Any, Final, Mapping, TypeAlias

import torch
from torch import nn

from ...config import (
    FunctionalMessagePassingConfig,
)
from ...constants import (
    ATTENTION_HEAD_REDUCTION_MEAN,
    ATTENTION_NORMALIZATION_TARGET_NODE_RELATION,
)
from ..schemas import (
    EdgeAttentionOutput,
    FunctionalMessagePassingInputs,
)
from .attention_normalization import (
    AttentionNormalizer,
    TargetNodeRelationAttentionNormalization,
    build_attention_normalizer,
    maximum_attention_normalization_error,
)
from .multihead import (
    AttentionHeadReducer,
    MeanAttentionHeadReduction,
    build_attention_head_reducer,
    head_disagreement_summary,
    maximum_reduced_attention_normalization_error,
)
from .schemas import (
    AttentionHeadReductionOutput,
    AttentionNormalizationOutput,
    EdgeAttentionScoreOutput,
)
from .score_functions import (
    DEFAULT_EDGE_ATTENTION_HIDDEN_DIM,
    AdditiveEdgeAttentionScoreFunction,
    EdgeAttentionScoreFunction,
    UniformEdgeAttentionScoreFunction,
    build_edge_attention_score_function,
)


# =============================================================================
# Public identity
# =============================================================================


EDGE_ATTENTION_ORCHESTRATOR_SCHEMA_VERSION: Final[str] = "0.1"

EDGE_ATTENTION_OPERATION_ORDER: Final[tuple[str, ...]] = (
    "validate_functional_message_passing_inputs",
    "predict_edge_attention_logits",
    "construct_exact_target_node_relation_groups",
    "normalize_logits_independently_per_head",
    "reduce_attention_heads_by_arithmetic_mean",
    "validate_exact_stage_lineage",
    "construct_public_edge_attention_output",
)

EDGE_ATTENTION_GROUP_SEMANTICS: Final[str] = (
    "target_node_exact_compiled_relation"
)

EDGE_ATTENTION_SCIENTIFIC_INTERPRETATION: Final[str] = (
    "within_relation_incoming_edge_routing"
)


# =============================================================================
# Public stage-chain type
# =============================================================================


EdgeAttentionStages: TypeAlias = tuple[
    EdgeAttentionScoreOutput,
    AttentionNormalizationOutput,
    AttentionHeadReductionOutput,
]


# =============================================================================
# Validation and fingerprint helpers
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
        _canonical_json(payload).encode("utf-8")
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


def _require_inputs(
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

    if source_inputs.num_relations <= 0:
        raise ValueError(
            "Edge attention requires at least one exact compiled "
            "relation."
        )

    if not source_inputs.dtype.is_floating_point:
        raise ValueError(
            "Edge attention requires a floating-point node-state dtype."
        )


def _require_score_function(
    score_function: EdgeAttentionScoreFunction,
) -> None:
    if not isinstance(
        score_function,
        (
            UniformEdgeAttentionScoreFunction,
            AdditiveEdgeAttentionScoreFunction,
        ),
    ):
        raise TypeError(
            "score_function must be a "
            "UniformEdgeAttentionScoreFunction or "
            "AdditiveEdgeAttentionScoreFunction."
        )


def _require_normalizer(
    normalizer: AttentionNormalizer,
) -> None:
    if not isinstance(
        normalizer,
        TargetNodeRelationAttentionNormalization,
    ):
        raise TypeError(
            "normalizer must be a "
            "TargetNodeRelationAttentionNormalization."
        )


def _require_head_reducer(
    head_reducer: AttentionHeadReducer,
) -> None:
    if not isinstance(
        head_reducer,
        MeanAttentionHeadReduction,
    ):
        raise TypeError(
            "head_reducer must be a "
            "MeanAttentionHeadReduction."
        )


def _require_score_output(
    score_output: EdgeAttentionScoreOutput,
) -> None:
    if not isinstance(
        score_output,
        EdgeAttentionScoreOutput,
    ):
        raise TypeError(
            "score_output must be an "
            "EdgeAttentionScoreOutput."
        )


def _require_normalization_output(
    normalization_output: AttentionNormalizationOutput,
) -> None:
    if not isinstance(
        normalization_output,
        AttentionNormalizationOutput,
    ):
        raise TypeError(
            "normalization_output must be an "
            "AttentionNormalizationOutput."
        )


def _require_reduction_output(
    reduction_output: AttentionHeadReductionOutput,
) -> None:
    if not isinstance(
        reduction_output,
        AttentionHeadReductionOutput,
    ):
        raise TypeError(
            "reduction_output must be an "
            "AttentionHeadReductionOutput."
        )


def _validate_stage_chain(
    score_output: EdgeAttentionScoreOutput,
    normalization_output: AttentionNormalizationOutput,
    reduction_output: AttentionHeadReductionOutput,
) -> None:
    """
    Validate exact stage identity and metadata continuity.

    Exact object identity is intentional. A final attention output must not
    splice logits from one graph, normalized weights from another, or reduced
    weights from a separately reconstructed stage chain.
    """

    _require_score_output(
        score_output
    )
    _require_normalization_output(
        normalization_output
    )
    _require_reduction_output(
        reduction_output
    )

    if (
        normalization_output
        .source_score_output
        is not score_output
    ):
        raise ValueError(
            "normalization_output must reference the exact supplied "
            "score_output object."
        )

    if (
        reduction_output
        .source_normalization_output
        is not normalization_output
    ):
        raise ValueError(
            "reduction_output must reference the exact supplied "
            "normalization_output object."
        )

    source_inputs = score_output.source_inputs

    if (
        normalization_output.source_inputs
        is not source_inputs
    ):
        raise ValueError(
            "Score and normalization stages must share the exact same "
            "FunctionalMessagePassingInputs object."
        )

    if (
        reduction_output.source_inputs
        is not source_inputs
    ):
        raise ValueError(
            "Score and head-reduction stages must share the exact same "
            "FunctionalMessagePassingInputs object."
        )

    if (
        score_output.num_heads
        != normalization_output.num_heads
    ):
        raise ValueError(
            "Score and normalization stages disagree on attention head "
            "count."
        )

    if (
        score_output.num_heads
        != reduction_output.num_heads
    ):
        raise ValueError(
            "Score and head-reduction stages disagree on attention head "
            "count."
        )

    if (
        normalization_output.attention_mode
        != score_output.attention_mode
    ):
        raise ValueError(
            "Normalization attention mode differs from the score stage."
        )

    if (
        reduction_output.attention_mode
        != score_output.attention_mode
    ):
        raise ValueError(
            "Head-reduction attention mode differs from the score stage."
        )

    if (
        reduction_output.normalization_mode
        != normalization_output.normalization_mode
    ):
        raise ValueError(
            "Head-reduction normalization mode differs from the "
            "normalization stage."
        )

    if not torch.equal(
        normalization_output.group_ids,
        reduction_output.group_ids,
    ):
        raise ValueError(
            "Normalization and head-reduction stages disagree on group "
            "IDs."
        )

    if not torch.equal(
        normalization_output.group_counts,
        reduction_output.group_counts,
    ):
        raise ValueError(
            "Normalization and head-reduction stages disagree on group "
            "counts."
        )

    if (
        reduction_output.raw_scores_by_head
        is not score_output.raw_scores_by_head
    ):
        raise ValueError(
            "Head-reduction lineage must preserve the exact raw-score "
            "tensor object."
        )

    if (
        reduction_output.normalized_weights_by_head
        is not normalization_output.normalized_weights_by_head
    ):
        raise ValueError(
            "Head-reduction lineage must preserve the exact normalized "
            "weight tensor object."
        )


# =============================================================================
# Public final-output assembly
# =============================================================================


def assemble_edge_attention_output(
    *,
    score_output: EdgeAttentionScoreOutput,
    normalization_output: AttentionNormalizationOutput,
    reduction_output: AttentionHeadReductionOutput,
    encoder_architecture_fingerprint: str,
    parameter_fingerprint: str | None = None,
) -> EdgeAttentionOutput:
    """
    Assemble the public edge-attention output from one exact stage chain.

    The helper performs no numerical recomputation. It validates lineage and
    exposes the exact tensors produced by the three owning stages.
    """

    _validate_stage_chain(
        score_output,
        normalization_output,
        reduction_output,
    )
    _require_nonempty_string(
        "encoder_architecture_fingerprint",
        encoder_architecture_fingerprint,
    )
    _require_optional_nonempty_string(
        "parameter_fingerprint",
        parameter_fingerprint,
    )

    return EdgeAttentionOutput(
        raw_scores_by_head=(
            score_output.raw_scores_by_head
        ),
        normalized_weights_by_head=(
            normalization_output
            .normalized_weights_by_head
        ),
        edge_weights=(
            reduction_output.edge_weights
        ),
        group_ids=(
            normalization_output.group_ids
        ),
        group_counts=(
            normalization_output.group_counts
        ),
        source_inputs=(
            score_output.source_inputs
        ),
        attention_mode=(
            score_output.attention_mode
        ),
        normalization_mode=(
            normalization_output
            .normalization_mode
        ),
        head_reduction=(
            reduction_output.head_reduction
        ),
        encoder_architecture_fingerprint=(
            encoder_architecture_fingerprint
        ),
        parameter_fingerprint=(
            parameter_fingerprint
        ),
    )


# =============================================================================
# Complete edge-attention orchestrator
# =============================================================================


class EdgeAttention(nn.Module):
    """
    Coordinate scoring, grouped normalization, and head reduction.

    Parameters
    ----------
    score_function:
        Uniform or learned exact-relation edge-logit predictor.
    normalizer:
        Parameter-free target-node/exact-relation grouped softmax.
    head_reducer:
        Parameter-free arithmetic-mean reduction with a frozen expected head
        count.
    """

    score_function: EdgeAttentionScoreFunction
    normalizer: AttentionNormalizer
    head_reducer: AttentionHeadReducer

    def __init__(
        self,
        *,
        score_function: EdgeAttentionScoreFunction,
        normalizer: AttentionNormalizer,
        head_reducer: AttentionHeadReducer,
    ) -> None:
        super().__init__()

        _require_score_function(
            score_function
        )
        _require_normalizer(
            normalizer
        )
        _require_head_reducer(
            head_reducer
        )

        if normalizer.normalization_mode != (
            ATTENTION_NORMALIZATION_TARGET_NODE_RELATION
        ):
            raise ValueError(
                "The bounded EdgeAttention orchestrator requires exact "
                "target-node/relation normalization."
            )

        if head_reducer.head_reduction != (
            ATTENTION_HEAD_REDUCTION_MEAN
        ):
            raise ValueError(
                "The bounded EdgeAttention orchestrator requires mean "
                "attention-head reduction."
            )

        if (
            head_reducer.num_heads
            != score_function.num_heads
        ):
            raise ValueError(
                "score_function and head_reducer must agree on the exact "
                "attention head count. Observed "
                f"{score_function.num_heads} and "
                f"{head_reducer.num_heads}."
            )

        self.score_function = score_function
        self.normalizer = normalizer
        self.head_reducer = head_reducer

        self._assert_component_contract()

    # ------------------------------------------------------------------
    # Construction from configuration
    # ------------------------------------------------------------------

    @classmethod
    def from_config(
        cls,
        *,
        config: FunctionalMessagePassingConfig,
        source_inputs: FunctionalMessagePassingInputs,
        score_hidden_dim: int = (
            DEFAULT_EDGE_ATTENTION_HIDDEN_DIM
        ),
    ) -> "EdgeAttention":
        """
        Build the complete enabled edge-attention stack.

        The constructor rejects ``attention_enabled=False`` rather than
        constructing uniform attention as a substitute for the disabled
        multiplicative identity.
        """

        if not isinstance(
            config,
            FunctionalMessagePassingConfig,
        ):
            raise TypeError(
                "config must be a "
                "FunctionalMessagePassingConfig."
            )

        config.validate()
        _require_inputs(
            source_inputs
        )

        if not config.enabled:
            raise ValueError(
                "EdgeAttention.from_config requires functional message "
                "passing to be enabled."
            )

        if not config.attention_enabled:
            raise ValueError(
                "EdgeAttention.from_config represents enabled attention "
                "and requires attention_enabled=True. Disabled attention "
                "must be represented by edge_attention=None in the "
                "higher-level message-passing layer."
            )

        score_function = (
            build_edge_attention_score_function(
                config=config,
                source_inputs=source_inputs,
                hidden_dim=score_hidden_dim,
            )
        )
        normalizer = build_attention_normalizer(
            config=config
        )
        head_reducer = (
            build_attention_head_reducer(
                config=config
            )
        )

        return cls(
            score_function=score_function,
            normalizer=normalizer,
            head_reducer=head_reducer,
        )

    # ------------------------------------------------------------------
    # Public component identity
    # ------------------------------------------------------------------

    @property
    def attention_mode(self) -> str:
        return self.score_function.mode

    @property
    def normalization_mode(self) -> str:
        return self.normalizer.normalization_mode

    @property
    def head_reduction(self) -> str:
        return self.head_reducer.head_reduction

    @property
    def num_heads(self) -> int:
        return self.score_function.num_heads

    @property
    def relation_names(
        self,
    ) -> tuple[str, ...]:
        return self.score_function.relation_names

    @property
    def stable_relation_ids(
        self,
    ) -> tuple[int, ...]:
        return self.score_function.stable_relation_ids

    @property
    def num_relations(self) -> int:
        return self.score_function.num_relations

    @property
    def uses_hazard_query(self) -> bool:
        return bool(
            getattr(
                self.score_function,
                "uses_hazard_query",
                False,
            )
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

    # ------------------------------------------------------------------
    # Architecture and parameter provenance
    # ------------------------------------------------------------------

    def architecture_dict(
        self,
    ) -> dict[str, Any]:
        return {
            "schema_version": (
                EDGE_ATTENTION_ORCHESTRATOR_SCHEMA_VERSION
            ),
            "attention_mode": (
                self.attention_mode
            ),
            "normalization_mode": (
                self.normalization_mode
            ),
            "head_reduction": (
                self.head_reduction
            ),
            "num_heads": self.num_heads,
            "num_relations": (
                self.num_relations
            ),
            "relation_names": list(
                self.relation_names
            ),
            "stable_relation_ids": list(
                self.stable_relation_ids
            ),
            "group_semantics": (
                EDGE_ATTENTION_GROUP_SEMANTICS
            ),
            "scientific_interpretation": (
                EDGE_ATTENTION_SCIENTIFIC_INTERPRETATION
            ),
            "score_function": (
                self.score_function
                .architecture_dict()
            ),
            "normalization_stage": (
                self.normalizer
                .architecture_dict()
            ),
            "head_reduction_stage": (
                self.head_reducer
                .architecture_dict()
            ),
            "attention_enabled_semantics": (
                "module_exists_only_for_enabled_attention"
            ),
            "disabled_attention_representation": (
                "None_at_higher_level_with_multiplicative_identity_one"
            ),
            "uniform_attention_representation": (
                "exact_zero_logits_then_grouped_softmax"
            ),
            "relation_gate_owned_here": False,
            "edge_attributes_used": False,
            "semantic_edge_weight_owned_here": False,
            "structural_normalization_owned_here": False,
            "message_construction_owned_here": False,
            "aggregation_owned_here": False,
            "claims_causal_importance": False,
            "claims_head_specialization": False,
            "parameter_count": (
                self.parameter_count
            ),
            "trainable_parameter_count": (
                self.trainable_parameter_count
            ),
            "operation_order": list(
                EDGE_ATTENTION_OPERATION_ORDER
            ),
            "output_schema": (
                "EdgeAttentionOutput"
            ),
        }

    def architecture_fingerprint(
        self,
    ) -> str:
        return _fingerprint(
            self.architecture_dict()
        )

    def parameter_fingerprint(
        self,
    ) -> str | None:
        """
        Fingerprint the complete trainable attention state.

        Normalization and head reduction are parameter-free. They remain in
        the payload so a future incompatible parameterization cannot silently
        masquerade as the same orchestration contract.
        """

        score_parameter_fingerprint = (
            self.score_function
            .parameter_fingerprint()
        )
        normalization_parameter_fingerprint = (
            self.normalizer
            .parameter_fingerprint()
        )
        reduction_parameter_fingerprint = (
            self.head_reducer
            .parameter_fingerprint()
        )

        if self.parameter_count == 0:
            if (
                score_parameter_fingerprint
                is not None
                or normalization_parameter_fingerprint
                is not None
                or reduction_parameter_fingerprint
                is not None
            ):
                raise RuntimeError(
                    "A parameter-free edge-attention stack reported a "
                    "non-null component parameter fingerprint."
                )

            return None

        if score_parameter_fingerprint is None:
            raise RuntimeError(
                "Trainable edge attention requires a score-function "
                "parameter fingerprint."
            )

        return _fingerprint(
            {
                "schema_version": (
                    EDGE_ATTENTION_ORCHESTRATOR_SCHEMA_VERSION
                ),
                "module": type(self).__name__,
                "attention_mode": (
                    self.attention_mode
                ),
                "score_function_parameter_fingerprint": (
                    score_parameter_fingerprint
                ),
                "normalization_parameter_fingerprint": (
                    normalization_parameter_fingerprint
                ),
                "head_reduction_parameter_fingerprint": (
                    reduction_parameter_fingerprint
                ),
                "parameter_count": (
                    self.parameter_count
                ),
                "trainable_parameter_count": (
                    self.trainable_parameter_count
                ),
            }
        )

    # ------------------------------------------------------------------
    # Component consistency and parameter checks
    # ------------------------------------------------------------------

    def _assert_component_contract(
        self,
    ) -> None:
        if (
            self.head_reducer.num_heads
            != self.score_function.num_heads
        ):
            raise RuntimeError(
                "Edge-attention component head counts are inconsistent."
            )

        if self.normalization_mode != (
            ATTENTION_NORMALIZATION_TARGET_NODE_RELATION
        ):
            raise RuntimeError(
                "Edge-attention normalization contract changed after "
                "construction."
            )

        if self.head_reduction != (
            ATTENTION_HEAD_REDUCTION_MEAN
        ):
            raise RuntimeError(
                "Edge-attention head-reduction contract changed after "
                "construction."
            )

        self.normalizer.assert_parameter_free()
        self.head_reducer.assert_parameter_free()

        component_parameter_count = (
            self.score_function.parameter_count
            + self.normalizer.parameter_count
            + self.head_reducer.parameter_count
        )

        if component_parameter_count != (
            self.parameter_count
        ):
            raise RuntimeError(
                "Edge-attention parameter counting is inconsistent "
                "across components."
            )

        component_trainable_count = (
            self.score_function
            .trainable_parameter_count
            + self.normalizer
            .trainable_parameter_count
            + self.head_reducer
            .trainable_parameter_count
        )

        if component_trainable_count != (
            self.trainable_parameter_count
        ):
            raise RuntimeError(
                "Edge-attention trainable-parameter counting is "
                "inconsistent across components."
            )

    def assert_finite_parameters(
        self,
    ) -> None:
        self.score_function.assert_finite_parameters()
        self.normalizer.assert_parameter_free()
        self.head_reducer.assert_parameter_free()
        self._assert_component_contract()

    # ------------------------------------------------------------------
    # Runtime input validation
    # ------------------------------------------------------------------

    def _validate_source_inputs(
        self,
        source_inputs: FunctionalMessagePassingInputs,
    ) -> None:
        _require_inputs(
            source_inputs
        )

        if source_inputs.relation_names != (
            self.relation_names
        ):
            raise ValueError(
                "Edge-attention relation ordering differs from source "
                "inputs."
            )

        if source_inputs.stable_relation_ids != (
            self.stable_relation_ids
        ):
            raise ValueError(
                "Edge-attention stable relation IDs differ from source "
                "inputs."
            )

        if source_inputs.num_relations != (
            self.num_relations
        ):
            raise ValueError(
                "Edge-attention relation count differs from source "
                "inputs."
            )

        if self.uses_hazard_query:
            hazard_query = (
                source_inputs.node_hazard_query
            )

            if hazard_query is None:
                raise ValueError(
                    "The configured edge-attention scorer requires a "
                    "node-aligned hazard query."
                )

            if hazard_query.ndim != 2:
                raise ValueError(
                    "source_inputs.node_hazard_query must have shape "
                    "[N, Q]."
                )

    # ------------------------------------------------------------------
    # Individually auditable stages
    # ------------------------------------------------------------------

    def score_edges(
        self,
        source_inputs: FunctionalMessagePassingInputs,
    ) -> EdgeAttentionScoreOutput:
        """
        Predict raw edge-attention logits.
        """

        self._validate_source_inputs(
            source_inputs
        )
        self.assert_finite_parameters()

        score_output = self.score_function(
            source_inputs
        )

        _require_score_output(
            score_output
        )

        if score_output.source_inputs is not (
            source_inputs
        ):
            raise RuntimeError(
                "Score function must preserve the exact source-input "
                "object."
            )

        if score_output.attention_mode != (
            self.attention_mode
        ):
            raise RuntimeError(
                "Score output attention mode differs from the "
                "orchestrator."
            )

        if score_output.num_heads != (
            self.num_heads
        ):
            raise RuntimeError(
                "Score output head count differs from the orchestrator."
            )

        return score_output

    def normalize_scores(
        self,
        score_output: EdgeAttentionScoreOutput,
    ) -> AttentionNormalizationOutput:
        """
        Apply exact target-node/relation grouped softmax independently per head.
        """

        _require_score_output(
            score_output
        )

        if score_output.attention_mode != (
            self.attention_mode
        ):
            raise ValueError(
                "score_output attention mode differs from the "
                "orchestrator."
            )

        if score_output.num_heads != (
            self.num_heads
        ):
            raise ValueError(
                "score_output head count differs from the orchestrator."
            )

        normalization_output = (
            self.normalizer(
                score_output
            )
        )

        _require_normalization_output(
            normalization_output
        )

        if (
            normalization_output
            .source_score_output
            is not score_output
        ):
            raise RuntimeError(
                "Normalizer must preserve the exact score-output object."
            )

        if (
            normalization_output
            .normalization_mode
            != self.normalization_mode
        ):
            raise RuntimeError(
                "Normalization output mode differs from the "
                "orchestrator."
            )

        return normalization_output

    def reduce_heads(
        self,
        normalization_output: AttentionNormalizationOutput,
    ) -> AttentionHeadReductionOutput:
        """
        Reduce independently normalized heads by arithmetic mean.
        """

        _require_normalization_output(
            normalization_output
        )

        if normalization_output.num_heads != (
            self.num_heads
        ):
            raise ValueError(
                "normalization_output head count differs from the "
                "orchestrator."
            )

        if (
            normalization_output
            .normalization_mode
            != self.normalization_mode
        ):
            raise ValueError(
                "normalization_output mode differs from the "
                "orchestrator."
            )

        reduction_output = (
            self.head_reducer(
                normalization_output
            )
        )

        _require_reduction_output(
            reduction_output
        )

        if (
            reduction_output
            .source_normalization_output
            is not normalization_output
        ):
            raise RuntimeError(
                "Head reducer must preserve the exact normalization "
                "output object."
            )

        if reduction_output.head_reduction != (
            self.head_reduction
        ):
            raise RuntimeError(
                "Head-reduction output policy differs from the "
                "orchestrator."
            )

        return reduction_output

    def compute_stages(
        self,
        source_inputs: FunctionalMessagePassingInputs,
    ) -> EdgeAttentionStages:
        """
        Execute and return all three immutable stage outputs.

        This method is intended for focused tests, numerical audits, and
        controlled research diagnostics. Ordinary model execution should use
        ``forward`` to avoid retaining extra Python references to intermediate
        metadata objects.
        """

        score_output = self.score_edges(
            source_inputs
        )
        normalization_output = (
            self.normalize_scores(
                score_output
            )
        )
        reduction_output = self.reduce_heads(
            normalization_output
        )

        _validate_stage_chain(
            score_output,
            normalization_output,
            reduction_output,
        )

        return (
            score_output,
            normalization_output,
            reduction_output,
        )

    # ------------------------------------------------------------------
    # Final-output assembly and forward
    # ------------------------------------------------------------------

    def assemble_output(
        self,
        score_output: EdgeAttentionScoreOutput,
        normalization_output: AttentionNormalizationOutput,
        reduction_output: AttentionHeadReductionOutput,
    ) -> EdgeAttentionOutput:
        """
        Construct the public output from one valid internal stage chain.
        """

        _validate_stage_chain(
            score_output,
            normalization_output,
            reduction_output,
        )

        if score_output.attention_mode != (
            self.attention_mode
        ):
            raise ValueError(
                "Stage-chain attention mode differs from the "
                "orchestrator."
            )

        if (
            normalization_output
            .normalization_mode
            != self.normalization_mode
        ):
            raise ValueError(
                "Stage-chain normalization mode differs from the "
                "orchestrator."
            )

        if reduction_output.head_reduction != (
            self.head_reduction
        ):
            raise ValueError(
                "Stage-chain head reduction differs from the "
                "orchestrator."
            )

        if score_output.num_heads != (
            self.num_heads
        ):
            raise ValueError(
                "Stage-chain head count differs from the orchestrator."
            )

        return assemble_edge_attention_output(
            score_output=score_output,
            normalization_output=(
                normalization_output
            ),
            reduction_output=(
                reduction_output
            ),
            encoder_architecture_fingerprint=(
                self.architecture_fingerprint()
            ),
            parameter_fingerprint=(
                self.parameter_fingerprint()
            ),
        )

    def forward(
        self,
        source_inputs: FunctionalMessagePassingInputs,
    ) -> EdgeAttentionOutput:
        """
        Produce complete enabled edge attention.
        """

        self._validate_source_inputs(
            source_inputs
        )
        self.assert_finite_parameters()

        (
            score_output,
            normalization_output,
            reduction_output,
        ) = self.compute_stages(
            source_inputs
        )

        return self.assemble_output(
            score_output,
            normalization_output,
            reduction_output,
        )

    # ------------------------------------------------------------------
    # Explicit diagnostics
    # ------------------------------------------------------------------

    def diagnostic_summary(
        self,
        reduction_output: AttentionHeadReductionOutput,
    ) -> dict[str, Any]:
        """
        Return compact descriptive diagnostics for one completed stage chain.

        No quantity in this summary should be interpreted as causal importance,
        calibrated uncertainty, or proof of head specialization.
        """

        _require_reduction_output(
            reduction_output
        )

        normalization_output = (
            reduction_output
            .source_normalization_output
        )
        score_output = (
            normalization_output
            .source_score_output
        )

        if score_output.attention_mode != (
            self.attention_mode
        ):
            raise ValueError(
                "Diagnostic stage chain differs from the orchestrator "
                "attention mode."
            )

        if score_output.num_heads != (
            self.num_heads
        ):
            raise ValueError(
                "Diagnostic stage chain differs from the orchestrator "
                "head count."
            )

        nonempty_group_count = int(
            (
                normalization_output
                .group_counts
                > 0
            )
            .sum()
            .item()
        )

        return {
            "schema_version": (
                EDGE_ATTENTION_ORCHESTRATOR_SCHEMA_VERSION
            ),
            "attention_mode": (
                self.attention_mode
            ),
            "normalization_mode": (
                self.normalization_mode
            ),
            "head_reduction": (
                self.head_reduction
            ),
            "num_edges": (
                score_output
                .source_inputs
                .num_edges
            ),
            "num_heads": self.num_heads,
            "num_groups": (
                normalization_output
                .num_groups
            ),
            "num_nonempty_groups": (
                nonempty_group_count
            ),
            "maximum_head_level_normalization_error": (
                maximum_attention_normalization_error(
                    normalization_output
                    .normalized_weights_by_head,
                    normalization_output
                    .group_ids,
                    num_groups=(
                        normalization_output
                        .num_groups
                    ),
                )
            ),
            "maximum_reduced_normalization_error": (
                maximum_reduced_attention_normalization_error(
                    reduction_output
                    .edge_weights,
                    normalization_output,
                )
            ),
            "head_disagreement": (
                head_disagreement_summary(
                    normalization_output
                    .normalized_weights_by_head
                )
            ),
            "interpretation": (
                EDGE_ATTENTION_SCIENTIFIC_INTERPRETATION
            ),
            "causal_importance_claim": False,
            "uncertainty_calibration_claim": False,
            "head_specialization_claim": False,
        }

    # ------------------------------------------------------------------
    # Representation
    # ------------------------------------------------------------------

    def extra_repr(self) -> str:
        return (
            f"attention_mode={self.attention_mode!r}, "
            f"normalization_mode={self.normalization_mode!r}, "
            f"head_reduction={self.head_reduction!r}, "
            f"num_heads={self.num_heads}, "
            f"num_relations={self.num_relations}, "
            f"uses_hazard_query={self.uses_hazard_query}, "
            f"parameter_count={self.parameter_count}, "
            "attention_enabled=True"
        )


# =============================================================================
# Public construction dispatcher
# =============================================================================


def build_edge_attention(
    *,
    config: FunctionalMessagePassingConfig,
    source_inputs: FunctionalMessagePassingInputs,
    score_hidden_dim: int = (
        DEFAULT_EDGE_ATTENTION_HIDDEN_DIM
    ),
) -> EdgeAttention:
    """
    Construct the complete enabled edge-attention stack.
    """

    return EdgeAttention.from_config(
        config=config,
        source_inputs=source_inputs,
        score_hidden_dim=score_hidden_dim,
    )


# Compact aliases for package exports and call sites.
EdgeAttentionModule = EdgeAttention
FunctionalEdgeAttention = EdgeAttention
HazardConditionedEdgeAttention = EdgeAttention
build_edge_attention_module = build_edge_attention
run_edge_attention = EdgeAttention.forward


__all__ = (
    "EDGE_ATTENTION_GROUP_SEMANTICS",
    "EDGE_ATTENTION_OPERATION_ORDER",
    "EDGE_ATTENTION_ORCHESTRATOR_SCHEMA_VERSION",
    "EDGE_ATTENTION_SCIENTIFIC_INTERPRETATION",
    "EdgeAttention",
    "EdgeAttentionModule",
    "EdgeAttentionStages",
    "FunctionalEdgeAttention",
    "HazardConditionedEdgeAttention",
    "assemble_edge_attention_output",
    "build_edge_attention",
    "build_edge_attention_module",
    "run_edge_attention",
)
