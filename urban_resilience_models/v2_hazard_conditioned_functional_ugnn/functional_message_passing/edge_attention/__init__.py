"""
Public API for exact-relation edge attention.

Target path:
    urban_resilience_models/
        v2_hazard_conditioned_functional_ugnn/
            functional_message_passing/
                edge_attention/
                    __init__.py

This package implements the complete enabled edge-attention subsystem for the
bounded V2.0 hazard-conditioned functional UGNN.

Scientific decomposition
------------------------
The package preserves three distinct operations:

1. **Edge-score prediction**

   For every stored directed edge ``e = (s_e -> t_e)`` with exact compiled
   relation index ``r_e``, a score function predicts one finite logit per
   attention head.

2. **Exact target-node/relation normalization**

   Logits are normalized independently inside the group

       (target node t_e, exact compiled relation r_e)

   using the deterministic dense identifier

       target_index[e] * num_relations + edge_relation_index[e]

   and independently for every head.

3. **Attention-head reduction**

   Independently normalized heads are reduced by an equal-weight arithmetic
   mean, producing one final edge-aligned routing coefficient.

The complete orchestrator exposes:

    score -> normalize -> reduce -> EdgeAttentionOutput

Scientific separation from relation gating
------------------------------------------
Edge attention and relation gating are intentionally different mechanisms.

``relation gate``
    Determines how strongly an exact relation mechanism contributes at a
    target node.

``edge attention``
    Distributes routing mass among the concrete incoming edges belonging to
    one already identified target-node/relation group.

A later message-construction stage may combine:

    relation_gate[target, relation]
    * edge_attention[edge]
    * structural_edge_normalization[edge]
    * optional_semantic_edge_weight[edge]

This package owns only the edge-attention factor.

Enabled uniform attention versus disabled attention
----------------------------------------------------
Uniform attention is implemented as exact zero logits followed by grouped
softmax, producing reciprocal group-size weights.

Disabled attention is not represented by this package. The higher-level
message-passing layer should represent disabled attention with
``edge_attention=None`` and use the multiplicative identity one.

Therefore:

    disabled attention != enabled uniform attention

Bounded score modes
-------------------
The public score-function API supports:

``uniform``
    Parameter-free exact-zero logits.

``hazard_blind``
    Learned source-target/exact-relation compatibility without hazard input.

``hazard_conditioned``
    Learned single-head compatibility using a node-aligned target hazard
    query.

``multihead_hazard_conditioned``
    Independently parameterized learned heads, each normalized separately,
    followed by arithmetic-mean reduction.

The canonical ``semantic_weight`` mode is not reinterpreted as attention.
Semantic edge weights remain separate data coefficients with their own
provenance.

Public contracts
----------------
The package exports immutable stage schemas:

``EdgeAttentionScoreOutput``
    Raw score-stage tensor and scorer provenance.

``AttentionNormalizationOutput``
    Per-head normalized weights, exact group IDs, and dense group counts.

``AttentionHeadReductionOutput``
    Final reduced edge weights and complete stage lineage.

``EdgeAttentionOutput``
    Compact public output consumed by later message construction.

Audit and diagnostic helpers
----------------------------
The public API also exposes focused diagnostics for:

- exact group construction and group counts;
- head-level group sums;
- normalization error;
- reduced normalization error;
- per-edge head variance;
- per-edge head standard deviation;
- per-edge head range;
- per-edge mean absolute deviation;
- compact head-disagreement summaries.

These diagnostics are descriptive. Attention values are routing coefficients,
not automatically causal importance, calibrated uncertainty, or proof of head
specialization.

Import behavior
---------------
Importing this package:

- does not construct neural modules;
- does not move tensors between devices;
- does not mutate relation registries;
- does not modify capability manifests;
- does not execute attention;
- does not allocate data-dependent buffers.

The historical flat module
``functional_message_passing/edge_attention.py`` must not coexist with this
package directory. It should be removed or renamed before this package is
installed, otherwise Python import resolution may be ambiguous across tools
and environments.
"""

# =============================================================================
# Immutable schemas and public score metadata
# =============================================================================

from .schemas import (
    ATTENTION_HEAD_REDUCTION_OUTPUT_SCHEMA_VERSION,
    ATTENTION_NORMALIZATION_OUTPUT_SCHEMA_VERSION,
    EDGE_ATTENTION_INPUT_EXACT_RELATION_EMBEDDING,
    EDGE_ATTENTION_INPUT_SOURCE_NODE_STATE,
    EDGE_ATTENTION_INPUT_TARGET_HAZARD_QUERY,
    EDGE_ATTENTION_INPUT_TARGET_NODE_STATE,
    EDGE_ATTENTION_SCHEMA_MODES,
    EDGE_ATTENTION_SCORE_FUNCTION_ADDITIVE,
    EDGE_ATTENTION_SCORE_FUNCTION_UNIFORM,
    EDGE_ATTENTION_SCORE_OUTPUT_SCHEMA_VERSION,
    AttentionHeadReductionOutput,
    AttentionNormalizationOutput,
    EdgeAttentionOutput,
    EdgeAttentionScoreOutput,
)

# =============================================================================
# Edge-score prediction
# =============================================================================

from .score_functions import (
    DEFAULT_EDGE_ATTENTION_HIDDEN_DIM,
    EDGE_ATTENTION_SCORE_FUNCTIONS_SCHEMA_VERSION,
    EDGE_ATTENTION_SCORE_MODES,
    LEARNED_EDGE_ATTENTION_MODES,
    AdditiveAttentionScoreFunction,
    AdditiveEdgeAttentionScoreFunction,
    EdgeAttentionScoreFunction,
    UniformAttentionScoreFunction,
    UniformEdgeAttentionScoreFunction,
    build_attention_score_function,
    build_edge_attention_score_function,
)

# =============================================================================
# Exact target-node/relation normalization
# =============================================================================

from .attention_normalization import (
    ATTENTION_GROUP_ID_FORMULA,
    ATTENTION_GROUP_KEY_TARGET_NODE_EXACT_RELATION,
    ATTENTION_NORMALIZATION_SCHEMA_VERSION,
    IMPLEMENTED_ATTENTION_NORMALIZATION_MODES,
    AttentionNormalization,
    AttentionNormalizer,
    EdgeAttentionNormalizer,
    TargetNodeRelationAttentionNormalization,
    apply_attention_normalization,
    assert_attention_normalized,
    attention_group_sums,
    build_attention_normalizer,
    build_edge_attention_normalizer,
    build_target_node_relation_group_counts,
    build_target_node_relation_group_ids,
    maximum_attention_normalization_error,
    normalize_attention_logits,
    normalize_edge_attention_scores,
)

# =============================================================================
# Attention-head reduction and disagreement diagnostics
# =============================================================================

from .multihead import (
    ATTENTION_HEAD_MEAN_FORMULA,
    ATTENTION_HEAD_REDUCTION_INTERPRETATION,
    ATTENTION_MULTIHEAD_SCHEMA_VERSION,
    IMPLEMENTED_ATTENTION_HEAD_REDUCTIONS,
    AttentionHeadReducer,
    AttentionHeadReduction,
    EdgeAttentionHeadReducer,
    MeanAttentionHeadReduction,
    MultiheadAttentionReduction,
    apply_attention_head_reduction,
    assert_reduced_attention_normalized,
    attention_head_mean,
    attention_head_mean_absolute_deviation,
    attention_head_range,
    attention_head_standard_deviation,
    attention_head_variance,
    build_attention_head_reducer,
    build_edge_attention_head_reducer,
    head_disagreement_summary,
    maximum_attention_head_range,
    maximum_reduced_attention_normalization_error,
    mean_attention_head_standard_deviation,
    mean_reduce_attention_heads,
    reduce_attention_heads,
    reduce_normalized_attention_heads,
)

# =============================================================================
# Complete orchestrator
# =============================================================================

from .edge_attention import (
    EDGE_ATTENTION_GROUP_SEMANTICS,
    EDGE_ATTENTION_OPERATION_ORDER,
    EDGE_ATTENTION_ORCHESTRATOR_SCHEMA_VERSION,
    EDGE_ATTENTION_SCIENTIFIC_INTERPRETATION,
    EdgeAttention,
    EdgeAttentionModule,
    EdgeAttentionStages,
    FunctionalEdgeAttention,
    HazardConditionedEdgeAttention,
    assemble_edge_attention_output,
    build_edge_attention,
    build_edge_attention_module,
    run_edge_attention,
)


__all__ = (
    # ------------------------------------------------------------------
    # Schema versions and immutable metadata constants
    # ------------------------------------------------------------------
    "ATTENTION_HEAD_REDUCTION_OUTPUT_SCHEMA_VERSION",
    "ATTENTION_NORMALIZATION_OUTPUT_SCHEMA_VERSION",
    "EDGE_ATTENTION_SCORE_OUTPUT_SCHEMA_VERSION",
    "EDGE_ATTENTION_SCORE_FUNCTIONS_SCHEMA_VERSION",
    "ATTENTION_NORMALIZATION_SCHEMA_VERSION",
    "ATTENTION_MULTIHEAD_SCHEMA_VERSION",
    "EDGE_ATTENTION_ORCHESTRATOR_SCHEMA_VERSION",
    # ------------------------------------------------------------------
    # Scientific vocabularies and formulas
    # ------------------------------------------------------------------
    "EDGE_ATTENTION_SCHEMA_MODES",
    "EDGE_ATTENTION_SCORE_MODES",
    "LEARNED_EDGE_ATTENTION_MODES",
    "IMPLEMENTED_ATTENTION_NORMALIZATION_MODES",
    "IMPLEMENTED_ATTENTION_HEAD_REDUCTIONS",
    "EDGE_ATTENTION_SCORE_FUNCTION_UNIFORM",
    "EDGE_ATTENTION_SCORE_FUNCTION_ADDITIVE",
    "EDGE_ATTENTION_INPUT_SOURCE_NODE_STATE",
    "EDGE_ATTENTION_INPUT_TARGET_NODE_STATE",
    "EDGE_ATTENTION_INPUT_TARGET_HAZARD_QUERY",
    "EDGE_ATTENTION_INPUT_EXACT_RELATION_EMBEDDING",
    "ATTENTION_GROUP_KEY_TARGET_NODE_EXACT_RELATION",
    "ATTENTION_GROUP_ID_FORMULA",
    "ATTENTION_HEAD_MEAN_FORMULA",
    "ATTENTION_HEAD_REDUCTION_INTERPRETATION",
    "EDGE_ATTENTION_GROUP_SEMANTICS",
    "EDGE_ATTENTION_OPERATION_ORDER",
    "EDGE_ATTENTION_SCIENTIFIC_INTERPRETATION",
    "DEFAULT_EDGE_ATTENTION_HIDDEN_DIM",
    # ------------------------------------------------------------------
    # Immutable stage and final-output contracts
    # ------------------------------------------------------------------
    "EdgeAttentionScoreOutput",
    "AttentionNormalizationOutput",
    "AttentionHeadReductionOutput",
    "EdgeAttentionOutput",
    "EdgeAttentionStages",
    # ------------------------------------------------------------------
    # Score-function modules and builders
    # ------------------------------------------------------------------
    "EdgeAttentionScoreFunction",
    "UniformEdgeAttentionScoreFunction",
    "UniformAttentionScoreFunction",
    "AdditiveEdgeAttentionScoreFunction",
    "AdditiveAttentionScoreFunction",
    "build_edge_attention_score_function",
    "build_attention_score_function",
    # ------------------------------------------------------------------
    # Exact target-node/relation normalization
    # ------------------------------------------------------------------
    "AttentionNormalizer",
    "TargetNodeRelationAttentionNormalization",
    "AttentionNormalization",
    "EdgeAttentionNormalizer",
    "build_target_node_relation_group_ids",
    "build_target_node_relation_group_counts",
    "normalize_attention_logits",
    "normalize_edge_attention_scores",
    "apply_attention_normalization",
    "build_attention_normalizer",
    "build_edge_attention_normalizer",
    "attention_group_sums",
    "assert_attention_normalized",
    "maximum_attention_normalization_error",
    # ------------------------------------------------------------------
    # Attention-head reduction
    # ------------------------------------------------------------------
    "AttentionHeadReducer",
    "MeanAttentionHeadReduction",
    "AttentionHeadReduction",
    "MultiheadAttentionReduction",
    "EdgeAttentionHeadReducer",
    "mean_reduce_attention_heads",
    "reduce_attention_heads",
    "reduce_normalized_attention_heads",
    "apply_attention_head_reduction",
    "build_attention_head_reducer",
    "build_edge_attention_head_reducer",
    "assert_reduced_attention_normalized",
    "maximum_reduced_attention_normalization_error",
    # ------------------------------------------------------------------
    # Descriptive head-disagreement diagnostics
    # ------------------------------------------------------------------
    "attention_head_mean",
    "attention_head_variance",
    "attention_head_standard_deviation",
    "attention_head_range",
    "attention_head_mean_absolute_deviation",
    "maximum_attention_head_range",
    "mean_attention_head_standard_deviation",
    "head_disagreement_summary",
    # ------------------------------------------------------------------
    # Complete enabled edge-attention orchestrator
    # ------------------------------------------------------------------
    "EdgeAttention",
    "EdgeAttentionModule",
    "FunctionalEdgeAttention",
    "HazardConditionedEdgeAttention",
    "assemble_edge_attention_output",
    "build_edge_attention",
    "build_edge_attention_module",
    "run_edge_attention",
)
