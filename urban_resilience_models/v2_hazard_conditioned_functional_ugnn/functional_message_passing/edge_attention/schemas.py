"""
Typed contracts for exact-relation edge attention.

Target path:
    urban_resilience_models/
        v2_hazard_conditioned_functional_ugnn/
            functional_message_passing/
                edge_attention/
                    schemas.py

This module owns immutable, metadata-preserving contracts for the three
internal stages of the bounded V2.0 edge-attention subsystem:

1. edge-level score production;
2. grouped normalization;
3. reduction across attention heads.

The final public ``EdgeAttentionOutput`` remains defined in
``functional_message_passing.schemas`` and is assembled by the later
``edge_attention.py`` orchestrator.

Responsibility
-------------------------
Relation gates and edge attention have deliberately different meanings.

For a target node ``t`` and exact compiled relation ``r``:

    relation gate g[t, r]
        controls how strongly mechanism r contributes at target t;

    edge attention alpha[e]
        distributes routing weight among the concrete incoming edges that
        share target t and relation r.

The bounded normalization group is therefore:

    group(e) = (target_index[e], edge_relation_index[e])

encoded densely as:

    group_id[e] =
        target_index[e] * num_relations
        + edge_relation_index[e]

For every nonempty group and every attention head:

    sum(alpha[e, head] for e in group) = 1

This is hierarchical functional routing, not a probabilistic mixture over
relations: sigmoid relation gates do not sum to one across relation channels.

Formal conditioning requirement
-------------------------------
Grouped softmax is invariant to a scalar added to every logit in one group:

    softmax(logit[e] + constant[target, relation])
        == softmax(logit[e])

Consequently, hazard, target, or relation context is behaviorally meaningful
only when it can change pairwise logit differences between competing edges.
The bounded learned scorer is identified as exact-relation additive
compatibility:

    score[e, head] =
        v_head^T tanh(
            W_source_head h_source[e]
            + W_target_head h_target[e]
            + optional W_hazard_head q_target[e]
            + relation_embedding[relation[e], head]
        )

The nonlinearity allows target, hazard, and relation context to change how
source-varying information is evaluated. A standalone group-constant hazard
or relation bias is intentionally not part of this contract.

Bounded V2.0 decisions
----------------------
- score tensors always have shape ``[E, A]``;
- the validated baseline is single-head;
- multihead tensors are represented without claiming head specialization;
- exact compiled relation identities, not pooled semantic families, index
  relation embeddings and attention groups;
- uniform attention uses exact zero logits followed by grouped softmax;
- hazard-blind additive attention is retained as an ablation;
- hazard-conditioned additive attention consumes the target-node hazard query;
- edge attributes, semantic edge weights, relation-gate values, structural
  normalization, relation transforms, message construction, and aggregation
  are not consumed by the scorer;
- mean is the bounded head-reduction policy;
- attention is a learned routing trace, not causal importance.

This module does not own:
- trainable score functions;
- grouped-softmax computation;
- attention-head reduction mathematics;
- relation-gate computation;
- relation transforms;
- edge masking;
- edge attributes or semantic edge-weight modulation;
- message construction or aggregation;
- hazard shuffling, counterfactual interventions, sensitivity metrics, or
  explanation-faithfulness experiments.

The contracts intentionally preserve lineage and exact relation order while
avoiding routine storage of large scorer preactivations or gradient-based
diagnostics in every forward pass.
"""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
from typing import Any, Final, Mapping, Sequence

import torch

from ...constants import (
    ATTENTION_HEAD_REDUCTION_MEAN,
    ATTENTION_MODE_HAZARD_BLIND,
    ATTENTION_MODE_HAZARD_CONDITIONED,
    ATTENTION_MODE_MULTIHEAD_HAZARD_CONDITIONED,
    ATTENTION_MODE_UNIFORM,
    ATTENTION_NORMALIZATION_TARGET_NODE_RELATION,
)
from ..schemas import (
    EdgeAttentionOutput,
    FunctionalMessagePassingInputs,
)
from ..segment_ops import (
    assert_grouped_softmax_normalized,
    segment_counts,
)


# =============================================================================
# Public schema and formula identity
# =============================================================================


EDGE_ATTENTION_SCORE_OUTPUT_SCHEMA_VERSION: Final[str] = "0.1"
ATTENTION_NORMALIZATION_OUTPUT_SCHEMA_VERSION: Final[str] = "0.1"
ATTENTION_HEAD_REDUCTION_OUTPUT_SCHEMA_VERSION: Final[str] = "0.1"


EDGE_ATTENTION_SCORE_FUNCTION_UNIFORM: Final[str] = (
    "uniform_zero_logits"
)
EDGE_ATTENTION_SCORE_FUNCTION_ADDITIVE: Final[str] = (
    "exact_relation_additive_compatibility"
)


EDGE_ATTENTION_INPUT_SOURCE_NODE_STATE: Final[str] = (
    "source_node_state"
)
EDGE_ATTENTION_INPUT_TARGET_NODE_STATE: Final[str] = (
    "target_node_state"
)
EDGE_ATTENTION_INPUT_TARGET_HAZARD_QUERY: Final[str] = (
    "target_node_hazard_query"
)
EDGE_ATTENTION_INPUT_EXACT_RELATION_EMBEDDING: Final[str] = (
    "exact_relation_embedding"
)


EDGE_ATTENTION_SCHEMA_MODES: Final[tuple[str, ...]] = (
    ATTENTION_MODE_UNIFORM,
    ATTENTION_MODE_HAZARD_BLIND,
    ATTENTION_MODE_HAZARD_CONDITIONED,
    ATTENTION_MODE_MULTIHEAD_HAZARD_CONDITIONED,
)


# =============================================================================
# Generic validation and fingerprint helpers
# =============================================================================


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


def _require_positive_int(
    name: str,
    value: int,
) -> None:
    if (
        isinstance(value, bool)
        or not isinstance(value, int)
        or value <= 0
    ):
        raise ValueError(
            f"{name} must be a positive integer."
        )


def _require_unique_strings(
    name: str,
    values: Sequence[str],
) -> None:
    seen: set[str] = set()
    duplicates: set[str] = set()

    for index, value in enumerate(values):
        _require_nonempty_string(
            f"{name}[{index}]",
            value,
        )

        if value in seen:
            duplicates.add(value)

        seen.add(value)

    if duplicates:
        raise ValueError(
            f"{name} contains duplicates: "
            f"{sorted(duplicates)}."
        )


def _require_unique_nonnegative_ints(
    name: str,
    values: Sequence[int],
) -> None:
    seen: set[int] = set()
    duplicates: set[int] = set()

    for index, value in enumerate(values):
        if (
            isinstance(value, bool)
            or not isinstance(value, int)
        ):
            raise TypeError(
                f"{name}[{index}] must be an integer."
            )

        if value < 0:
            raise ValueError(
                f"{name}[{index}] must be nonnegative."
            )

        if value in seen:
            duplicates.add(value)

        seen.add(value)

    if duplicates:
        raise ValueError(
            f"{name} contains duplicates: "
            f"{sorted(duplicates)}."
        )


def _require_tensor(
    name: str,
    value: torch.Tensor,
    *,
    ndim: int,
) -> None:
    if not isinstance(
        value,
        torch.Tensor,
    ):
        raise TypeError(
            f"{name} must be a tensor."
        )

    if value.ndim != ndim:
        raise ValueError(
            f"{name} must have rank {ndim}; "
            f"observed shape {tuple(value.shape)}."
        )


def _require_float_tensor(
    name: str,
    value: torch.Tensor,
    *,
    ndim: int,
) -> None:
    _require_tensor(
        name,
        value,
        ndim=ndim,
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
        raise ValueError(
            f"{name} must contain only finite values."
        )


def _require_long_tensor(
    name: str,
    value: torch.Tensor,
    *,
    ndim: int,
) -> None:
    _require_tensor(
        name,
        value,
        ndim=ndim,
    )

    if value.dtype != torch.long:
        raise ValueError(
            f"{name} must use torch.long."
        )


def _require_shape(
    name: str,
    value: torch.Tensor,
    expected: tuple[int, ...],
) -> None:
    if tuple(value.shape) != expected:
        raise ValueError(
            f"{name} must have shape {expected}; "
            f"observed {tuple(value.shape)}."
        )


def _require_index_range(
    name: str,
    value: torch.Tensor,
    *,
    upper_bound: int,
) -> None:
    if value.numel() == 0:
        return

    if upper_bound <= 0:
        raise ValueError(
            f"{name} cannot be nonempty when upper_bound is "
            f"{upper_bound}."
        )

    minimum = int(
        value.min().item()
    )
    maximum = int(
        value.max().item()
    )

    if minimum < 0 or maximum >= upper_bound:
        raise ValueError(
            f"{name} contains out-of-range indices. "
            f"Observed range [{minimum}, {maximum}]; "
            f"valid range is [0, {upper_bound - 1}]."
        )


def _devices_match(
    first: torch.device | str,
    second: torch.device | str,
) -> bool:
    first_device = torch.device(first)
    second_device = torch.device(second)

    if first_device.type != second_device.type:
        return False

    if first_device.type != "cuda":
        return first_device == second_device

    first_index = (
        torch.cuda.current_device()
        if first_device.index is None
        else first_device.index
    )
    second_index = (
        torch.cuda.current_device()
        if second_device.index is None
        else second_device.index
    )

    return first_index == second_index


def _require_device(
    name: str,
    value: torch.Tensor,
    expected: torch.device | str,
) -> None:
    if not _devices_match(
        value.device,
        expected,
    ):
        raise ValueError(
            f"{name} must be on device {torch.device(expected)}; "
            f"observed {value.device}."
        )


def _require_dtype(
    name: str,
    value: torch.Tensor,
    expected: torch.dtype,
) -> None:
    if value.dtype != expected:
        raise ValueError(
            f"{name} must use dtype {expected}; "
            f"observed {value.dtype}."
        )


def _default_tolerances(
    dtype: torch.dtype,
) -> tuple[float, float]:
    if dtype in (
        torch.float16,
        torch.bfloat16,
    ):
        return 5e-3, 5e-3

    if dtype == torch.float64:
        return 1e-10, 1e-10

    return 1e-5, 1e-5


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


def _tensor_fingerprint(
    tensors: Mapping[str, torch.Tensor],
) -> str:
    digest = sha256()

    for name in sorted(tensors):
        tensor = tensors[name]

        if not isinstance(
            tensor,
            torch.Tensor,
        ):
            raise TypeError(
                f"{name} must be a tensor."
            )

        detached = (
            tensor
            .detach()
            .contiguous()
            .cpu()
        )
        raw = (
            detached
            .view(torch.uint8)
            .numpy()
            .tobytes()
        )

        digest.update(
            name.encode("utf-8")
        )
        digest.update(
            str(tuple(detached.shape)).encode(
                "utf-8"
            )
        )
        digest.update(
            str(detached.dtype).encode(
                "utf-8"
            )
        )
        digest.update(raw)

    return digest.hexdigest()


def _normalize_attention_mode(
    attention_mode: str,
) -> str:
    if not isinstance(
        attention_mode,
        str,
    ):
        raise TypeError(
            "attention_mode must be a string."
        )

    normalized = attention_mode.strip()

    if not normalized:
        raise ValueError(
            "attention_mode must be a non-empty string."
        )

    if normalized not in EDGE_ATTENTION_SCHEMA_MODES:
        raise ValueError(
            "The bounded edge-attention score schema does not support "
            f"attention mode {normalized!r}. Expected one of "
            f"{EDGE_ATTENTION_SCHEMA_MODES!r}."
        )

    return normalized


def _expected_score_function(
    attention_mode: str,
) -> str:
    if attention_mode == ATTENTION_MODE_UNIFORM:
        return EDGE_ATTENTION_SCORE_FUNCTION_UNIFORM

    return EDGE_ATTENTION_SCORE_FUNCTION_ADDITIVE


def _expected_input_feature_names(
    attention_mode: str,
) -> tuple[str, ...]:
    if attention_mode == ATTENTION_MODE_UNIFORM:
        return ()

    common = (
        EDGE_ATTENTION_INPUT_SOURCE_NODE_STATE,
        EDGE_ATTENTION_INPUT_TARGET_NODE_STATE,
        EDGE_ATTENTION_INPUT_EXACT_RELATION_EMBEDDING,
    )

    if attention_mode == ATTENTION_MODE_HAZARD_BLIND:
        return common

    return (
        EDGE_ATTENTION_INPUT_SOURCE_NODE_STATE,
        EDGE_ATTENTION_INPUT_TARGET_NODE_STATE,
        EDGE_ATTENTION_INPUT_TARGET_HAZARD_QUERY,
        EDGE_ATTENTION_INPUT_EXACT_RELATION_EMBEDDING,
    )


def _validate_mode_and_head_count(
    *,
    attention_mode: str,
    num_heads: int,
) -> None:
    _require_positive_int(
        "num_heads",
        num_heads,
    )

    if (
        attention_mode
        == ATTENTION_MODE_MULTIHEAD_HAZARD_CONDITIONED
    ):
        if num_heads < 2:
            raise ValueError(
                "multihead_hazard_conditioned attention requires at "
                "least two attention heads."
            )
        return

    if num_heads != 1:
        raise ValueError(
            f"Attention mode {attention_mode!r} requires exactly one "
            "attention head in the bounded V2.0 contract."
        )


def _validate_relation_identity(
    *,
    relation_names: tuple[str, ...],
    stable_relation_ids: tuple[int, ...],
    compiled_relation_registry_fingerprint: str,
    source_inputs: FunctionalMessagePassingInputs,
) -> None:
    _require_unique_strings(
        "relation_names",
        relation_names,
    )
    _require_unique_nonnegative_ints(
        "stable_relation_ids",
        stable_relation_ids,
    )

    if not relation_names:
        raise ValueError(
            "At least one compiled relation is required."
        )

    if len(relation_names) != len(
        stable_relation_ids
    ):
        raise ValueError(
            "relation_names and stable_relation_ids must align."
        )

    if relation_names != (
        source_inputs.relation_names
    ):
        raise ValueError(
            "relation_names must exactly match the compiled relation "
            "order in source_inputs."
        )

    if stable_relation_ids != (
        source_inputs.stable_relation_ids
    ):
        raise ValueError(
            "stable_relation_ids must exactly match the compiled "
            "relation order in source_inputs."
        )

    _require_nonempty_string(
        "compiled_relation_registry_fingerprint",
        compiled_relation_registry_fingerprint,
    )

    expected_fingerprint = (
        source_inputs
        .compiled_relation_registry
        .fingerprint()
    )

    if (
        compiled_relation_registry_fingerprint
        != expected_fingerprint
    ):
        raise ValueError(
            "compiled_relation_registry_fingerprint references a "
            "different compiled relation registry."
        )


def _validate_edge_head_tensor(
    name: str,
    value: torch.Tensor,
    *,
    source_inputs: FunctionalMessagePassingInputs,
    num_heads: int | None = None,
) -> None:
    _require_float_tensor(
        name,
        value,
        ndim=2,
    )

    expected_heads = (
        int(value.shape[1])
        if num_heads is None
        else num_heads
    )

    _require_positive_int(
        f"{name} head count",
        expected_heads,
    )
    _require_shape(
        name,
        value,
        (
            source_inputs.num_edges,
            expected_heads,
        ),
    )
    _require_device(
        name,
        value,
        source_inputs.device,
    )
    _require_dtype(
        name,
        value,
        source_inputs.dtype,
    )


# =============================================================================
# Raw edge-score contract
# =============================================================================


@dataclass(slots=True, frozen=True)
class EdgeAttentionScoreOutput:
    """
    Raw edge logits before grouped normalization.

    ``raw_scores_by_head[e, a]`` is the finite unnormalized compatibility
    score for stored directed edge ``e`` and attention head ``a``.

    The object records the exact compiled relation order used by the scorer.
    This is necessary because learned relation embeddings are indexed by dense
    compiled relation index, while stable ontology IDs are sparse metadata and
    must never be used as parameter-table rows.

    The output deliberately carries no relation-gate values. Gate and
    attention remain independently identifiable routing factors.
    """

    raw_scores_by_head: torch.Tensor
    source_inputs: FunctionalMessagePassingInputs

    relation_names: tuple[str, ...]
    stable_relation_ids: tuple[int, ...]
    compiled_relation_registry_fingerprint: str

    attention_mode: str
    score_function: str
    input_feature_names: tuple[str, ...]

    encoder_architecture_fingerprint: str
    parameter_fingerprint: str | None = None

    schema_version: str = (
        EDGE_ATTENTION_SCORE_OUTPUT_SCHEMA_VERSION
    )

    def __post_init__(self) -> None:
        if not isinstance(
            self.source_inputs,
            FunctionalMessagePassingInputs,
        ):
            raise TypeError(
                "source_inputs must be a "
                "FunctionalMessagePassingInputs."
            )

        object.__setattr__(
            self,
            "relation_names",
            tuple(self.relation_names),
        )
        object.__setattr__(
            self,
            "stable_relation_ids",
            tuple(self.stable_relation_ids),
        )
        object.__setattr__(
            self,
            "input_feature_names",
            tuple(self.input_feature_names),
        )

        _validate_relation_identity(
            relation_names=self.relation_names,
            stable_relation_ids=(
                self.stable_relation_ids
            ),
            compiled_relation_registry_fingerprint=(
                self
                .compiled_relation_registry_fingerprint
            ),
            source_inputs=self.source_inputs,
        )

        normalized_mode = (
            _normalize_attention_mode(
                self.attention_mode
            )
        )

        if normalized_mode != self.attention_mode:
            raise ValueError(
                "attention_mode must use its canonical spelling without "
                "leading or trailing whitespace."
            )

        _validate_edge_head_tensor(
            "raw_scores_by_head",
            self.raw_scores_by_head,
            source_inputs=self.source_inputs,
        )

        _validate_mode_and_head_count(
            attention_mode=self.attention_mode,
            num_heads=self.num_heads,
        )

        _require_nonempty_string(
            "score_function",
            self.score_function,
        )

        expected_score_function = (
            _expected_score_function(
                self.attention_mode
            )
        )

        if self.score_function != (
            expected_score_function
        ):
            raise ValueError(
                f"Attention mode {self.attention_mode!r} requires "
                f"score_function={expected_score_function!r}; "
                f"observed {self.score_function!r}."
            )

        _require_unique_strings(
            "input_feature_names",
            self.input_feature_names,
        )

        expected_features = (
            _expected_input_feature_names(
                self.attention_mode
            )
        )

        if self.input_feature_names != expected_features:
            raise ValueError(
                f"Attention mode {self.attention_mode!r} requires "
                f"input_feature_names={expected_features!r}; "
                f"observed {self.input_feature_names!r}."
            )

        if self.uses_hazard_query:
            if (
                self
                .source_inputs
                .node_hazard_query
                is None
            ):
                raise ValueError(
                    "Hazard-conditioned edge attention requires a "
                    "node-aligned hazard query."
                )

        if self.attention_mode == (
            ATTENTION_MODE_UNIFORM
        ):
            if not torch.equal(
                self.raw_scores_by_head,
                torch.zeros_like(
                    self.raw_scores_by_head
                ),
            ):
                raise ValueError(
                    "Uniform attention requires exact zero raw scores."
                )

        _require_nonempty_string(
            "encoder_architecture_fingerprint",
            self.encoder_architecture_fingerprint,
        )
        _require_optional_nonempty_string(
            "parameter_fingerprint",
            self.parameter_fingerprint,
        )
        _require_nonempty_string(
            "schema_version",
            self.schema_version,
        )

    @classmethod
    def relation_identity_from_inputs(
        cls,
        *,
        source_inputs: FunctionalMessagePassingInputs,
    ) -> dict[str, object]:
        """
        Return canonical constructor fields for exact relation identity.

        The helper performs no tensor computation and does not create a score
        output. It exists to prevent score functions from reconstructing or
        reordering relation metadata independently.
        """

        if not isinstance(
            source_inputs,
            FunctionalMessagePassingInputs,
        ):
            raise TypeError(
                "source_inputs must be a "
                "FunctionalMessagePassingInputs."
            )

        return {
            "relation_names": (
                source_inputs.relation_names
            ),
            "stable_relation_ids": (
                source_inputs.stable_relation_ids
            ),
            "compiled_relation_registry_fingerprint": (
                source_inputs
                .compiled_relation_registry
                .fingerprint()
            ),
        }

    @property
    def num_edges(self) -> int:
        return self.source_inputs.num_edges

    @property
    def num_heads(self) -> int:
        return int(
            self.raw_scores_by_head.shape[1]
        )

    @property
    def num_relations(self) -> int:
        return len(self.relation_names)

    @property
    def device(self) -> torch.device:
        return self.raw_scores_by_head.device

    @property
    def dtype(self) -> torch.dtype:
        return self.raw_scores_by_head.dtype

    @property
    def uses_source_node_state(self) -> bool:
        return (
            EDGE_ATTENTION_INPUT_SOURCE_NODE_STATE
            in self.input_feature_names
        )

    @property
    def uses_target_node_state(self) -> bool:
        return (
            EDGE_ATTENTION_INPUT_TARGET_NODE_STATE
            in self.input_feature_names
        )

    @property
    def uses_hazard_query(self) -> bool:
        return (
            EDGE_ATTENTION_INPUT_TARGET_HAZARD_QUERY
            in self.input_feature_names
        )

    @property
    def uses_relation_embedding(self) -> bool:
        return (
            EDGE_ATTENTION_INPUT_EXACT_RELATION_EMBEDDING
            in self.input_feature_names
        )

    @property
    def uses_edge_attributes(self) -> bool:
        """
        Whether this bounded score contract consumes edge attributes.

        V2.0 intentionally returns ``False``. Edge attributes remain preserved
        in ``source_inputs`` for future explicitly configured score functions.
        """

        return False

    @property
    def relation_axis_dict(
        self,
    ) -> dict[str, Any]:
        return {
            "relation_names": list(
                self.relation_names
            ),
            "stable_relation_ids": list(
                self.stable_relation_ids
            ),
            "compiled_relation_registry_fingerprint": (
                self
                .compiled_relation_registry_fingerprint
            ),
        }

    def relation_axis_fingerprint(
        self,
    ) -> str:
        return _fingerprint(
            self.relation_axis_dict
        )

    def lineage_dict(
        self,
    ) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "attention_mode": self.attention_mode,
            "score_function": self.score_function,
            "num_heads": self.num_heads,
            "input_feature_names": list(
                self.input_feature_names
            ),
            "uses_hazard_query": (
                self.uses_hazard_query
            ),
            "uses_edge_attributes": (
                self.uses_edge_attributes
            ),
            "relation_axis_fingerprint": (
                self.relation_axis_fingerprint()
            ),
            "source_input_lineage_fingerprint": (
                self
                .source_inputs
                .lineage_fingerprint()
            ),
            "encoder_architecture_fingerprint": (
                self
                .encoder_architecture_fingerprint
            ),
            "parameter_fingerprint": (
                self.parameter_fingerprint
            ),
        }

    def lineage_fingerprint(
        self,
    ) -> str:
        return _fingerprint(
            self.lineage_dict()
        )

    def value_fingerprint(
        self,
    ) -> str:
        return _tensor_fingerprint(
            {
                "raw_scores_by_head": (
                    self.raw_scores_by_head
                ),
            }
        )

    def fingerprint(
        self,
    ) -> str:
        return _fingerprint(
            {
                "lineage_fingerprint": (
                    self.lineage_fingerprint()
                ),
                "value_fingerprint": (
                    self.value_fingerprint()
                ),
            }
        )


# =============================================================================
# Grouped-normalization contract
# =============================================================================


@dataclass(slots=True, frozen=True)
class AttentionNormalizationOutput:
    """
    Head-level attention weights normalized within exact target-relation groups.

    The schema validates grouping identity, group counts, nonnegativity, group
    normalization, singleton exactness, and the uniform-attention identity.

    It intentionally does not recompute grouped softmax from the raw logits in
    every constructor call. Recomputing a second differentiable grouped
    softmax during each training forward would duplicate an expensive
    edge-level operation. Exact logit-to-weight equivalence belongs in focused
    tests for ``attention_normalization.py``; this runtime contract validates
    the complete mathematical invariants of the resulting distribution.
    """

    normalized_weights_by_head: torch.Tensor
    group_ids: torch.Tensor
    group_counts: torch.Tensor

    source_score_output: EdgeAttentionScoreOutput

    normalization_mode: str

    encoder_architecture_fingerprint: str
    parameter_fingerprint: str | None = None

    schema_version: str = (
        ATTENTION_NORMALIZATION_OUTPUT_SCHEMA_VERSION
    )

    def __post_init__(self) -> None:
        if not isinstance(
            self.source_score_output,
            EdgeAttentionScoreOutput,
        ):
            raise TypeError(
                "source_score_output must be an "
                "EdgeAttentionScoreOutput."
            )

        _validate_edge_head_tensor(
            "normalized_weights_by_head",
            self.normalized_weights_by_head,
            source_inputs=self.source_inputs,
            num_heads=self.num_heads,
        )

        _require_long_tensor(
            "group_ids",
            self.group_ids,
            ndim=1,
        )
        _require_shape(
            "group_ids",
            self.group_ids,
            (self.source_inputs.num_edges,),
        )
        _require_device(
            "group_ids",
            self.group_ids,
            self.source_inputs.device,
        )
        _require_index_range(
            "group_ids",
            self.group_ids,
            upper_bound=self.num_groups,
        )

        _require_long_tensor(
            "group_counts",
            self.group_counts,
            ndim=1,
        )
        _require_shape(
            "group_counts",
            self.group_counts,
            (self.num_groups,),
        )
        _require_device(
            "group_counts",
            self.group_counts,
            self.source_inputs.device,
        )

        if bool(
            (self.group_counts < 0)
            .any()
            .item()
        ):
            raise ValueError(
                "group_counts must be nonnegative."
            )

        _require_nonempty_string(
            "normalization_mode",
            self.normalization_mode,
        )

        if self.normalization_mode != (
            ATTENTION_NORMALIZATION_TARGET_NODE_RELATION
        ):
            raise ValueError(
                "The bounded V2.0 attention-normalization schema "
                "supports only exact target-node + relation grouping."
            )

        expected_group_ids = (
            self
            .source_inputs
            .attention_group_id
        )

        if not torch.equal(
            self.group_ids,
            expected_group_ids,
        ):
            raise ValueError(
                "group_ids must encode target node + exact dense "
                "relation index."
            )

        expected_counts = segment_counts(
            self.group_ids,
            num_segments=self.num_groups,
        )

        if not torch.equal(
            self.group_counts,
            expected_counts,
        ):
            raise ValueError(
                "group_counts must equal the counts implied by "
                "group_ids."
            )

        if int(
            self.group_counts.sum().item()
        ) != self.source_inputs.num_edges:
            raise ValueError(
                "group_counts must sum to the stored edge count."
            )

        assert_grouped_softmax_normalized(
            self.normalized_weights_by_head,
            self.group_ids,
            num_segments=self.num_groups,
        )

        singleton_edges = (
            self.group_counts[
                self.group_ids
            ]
            == 1
        )

        if bool(
            singleton_edges.any().item()
        ):
            observed = (
                self
                .normalized_weights_by_head[
                    singleton_edges
                ]
            )
            expected = torch.ones_like(
                observed
            )

            if not torch.equal(
                observed,
                expected,
            ):
                raise ValueError(
                    "Every one-edge attention group must receive exact "
                    "weight one for every head."
                )

        if self.attention_mode == (
            ATTENTION_MODE_UNIFORM
        ):
            if self.source_inputs.num_edges > 0:
                denominators = (
                    self
                    .group_counts[
                        self.group_ids
                    ]
                    .to(
                        dtype=(
                            self
                            .normalized_weights_by_head
                            .dtype
                        )
                    )
                    .unsqueeze(1)
                )
                expected_uniform = (
                    torch.ones_like(
                        self
                        .normalized_weights_by_head
                    )
                    / denominators
                )
            else:
                expected_uniform = (
                    torch.empty_like(
                        self
                        .normalized_weights_by_head
                    )
                )

            atol, rtol = _default_tolerances(
                self
                .normalized_weights_by_head
                .dtype
            )

            if not torch.allclose(
                self.normalized_weights_by_head,
                expected_uniform,
                atol=atol,
                rtol=rtol,
            ):
                raise ValueError(
                    "Uniform attention must assign reciprocal group-size "
                    "weight to every edge independently for each head."
                )

        _require_nonempty_string(
            "encoder_architecture_fingerprint",
            self.encoder_architecture_fingerprint,
        )
        _require_optional_nonempty_string(
            "parameter_fingerprint",
            self.parameter_fingerprint,
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
            .source_score_output
            .source_inputs
        )

    @property
    def raw_scores_by_head(
        self,
    ) -> torch.Tensor:
        return (
            self
            .source_score_output
            .raw_scores_by_head
        )

    @property
    def attention_mode(self) -> str:
        return (
            self
            .source_score_output
            .attention_mode
        )

    @property
    def score_function(self) -> str:
        return (
            self
            .source_score_output
            .score_function
        )

    @property
    def num_heads(self) -> int:
        return (
            self
            .source_score_output
            .num_heads
        )

    @property
    def num_groups(self) -> int:
        return (
            self
            .source_inputs
            .attention_num_groups
        )

    @property
    def group_presence(self) -> torch.Tensor:
        return self.group_counts > 0

    @property
    def num_nonempty_groups(self) -> int:
        return int(
            self.group_presence.sum().item()
        )

    @property
    def device(self) -> torch.device:
        return (
            self
            .normalized_weights_by_head
            .device
        )

    @property
    def dtype(self) -> torch.dtype:
        return (
            self
            .normalized_weights_by_head
            .dtype
        )

    def lineage_dict(
        self,
    ) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "normalization_mode": (
                self.normalization_mode
            ),
            "group_key": (
                "target_node_exact_relation"
            ),
            "num_groups": self.num_groups,
            "num_nonempty_groups": (
                self.num_nonempty_groups
            ),
            "source_score_fingerprint": (
                self
                .source_score_output
                .fingerprint()
            ),
            "encoder_architecture_fingerprint": (
                self
                .encoder_architecture_fingerprint
            ),
            "parameter_fingerprint": (
                self.parameter_fingerprint
            ),
        }

    def lineage_fingerprint(
        self,
    ) -> str:
        return _fingerprint(
            self.lineage_dict()
        )

    def value_fingerprint(
        self,
    ) -> str:
        return _tensor_fingerprint(
            {
                "normalized_weights_by_head": (
                    self.normalized_weights_by_head
                ),
                "group_ids": self.group_ids,
                "group_counts": self.group_counts,
            }
        )

    def fingerprint(
        self,
    ) -> str:
        return _fingerprint(
            {
                "lineage_fingerprint": (
                    self.lineage_fingerprint()
                ),
                "value_fingerprint": (
                    self.value_fingerprint()
                ),
            }
        )


# =============================================================================
# Head-reduction contract
# =============================================================================


@dataclass(slots=True, frozen=True)
class AttentionHeadReductionOutput:
    """
    One edge-aligned routing coefficient after attention-head reduction.

    The bounded policy is the arithmetic mean:

        edge_weights[e] =
            mean(
                normalized_weights_by_head[e, :]
            )

    Because every head is independently normalized within each exact
    target-relation group, the arithmetic mean is also group-normalized.

    This tensor is still an attention routing coefficient. It is not a
    relation gate, semantic edge weight, structural normalizer, aggregation
    denominator, or causal importance score.
    """

    edge_weights: torch.Tensor

    source_normalization_output: (
        AttentionNormalizationOutput
    )

    head_reduction: str

    encoder_architecture_fingerprint: str
    parameter_fingerprint: str | None = None

    schema_version: str = (
        ATTENTION_HEAD_REDUCTION_OUTPUT_SCHEMA_VERSION
    )

    def __post_init__(self) -> None:
        if not isinstance(
            self.source_normalization_output,
            AttentionNormalizationOutput,
        ):
            raise TypeError(
                "source_normalization_output must be an "
                "AttentionNormalizationOutput."
            )

        _require_float_tensor(
            "edge_weights",
            self.edge_weights,
            ndim=1,
        )
        _require_shape(
            "edge_weights",
            self.edge_weights,
            (self.source_inputs.num_edges,),
        )
        _require_device(
            "edge_weights",
            self.edge_weights,
            self.source_inputs.device,
        )
        _require_dtype(
            "edge_weights",
            self.edge_weights,
            self.source_inputs.dtype,
        )

        _require_nonempty_string(
            "head_reduction",
            self.head_reduction,
        )

        if self.head_reduction != (
            ATTENTION_HEAD_REDUCTION_MEAN
        ):
            raise ValueError(
                "The bounded V2.0 head-reduction schema supports only "
                f"{ATTENTION_HEAD_REDUCTION_MEAN!r}."
            )

        expected = (
            self
            .normalized_weights_by_head
            .mean(dim=1)
        )
        atol, rtol = _default_tolerances(
            self.edge_weights.dtype
        )

        if not torch.allclose(
            self.edge_weights,
            expected,
            atol=atol,
            rtol=rtol,
        ):
            raise ValueError(
                "Mean head reduction must equal the arithmetic mean of "
                "normalized head-level weights."
            )

        assert_grouped_softmax_normalized(
            self.edge_weights,
            self.group_ids,
            num_segments=self.num_groups,
        )

        singleton_edges = (
            self.group_counts[
                self.group_ids
            ]
            == 1
        )

        if bool(
            singleton_edges.any().item()
        ):
            if not torch.equal(
                self.edge_weights[
                    singleton_edges
                ],
                torch.ones_like(
                    self.edge_weights[
                        singleton_edges
                    ]
                ),
            ):
                raise ValueError(
                    "Every one-edge attention group must retain exact "
                    "reduced weight one."
                )

        _require_nonempty_string(
            "encoder_architecture_fingerprint",
            self.encoder_architecture_fingerprint,
        )
        _require_optional_nonempty_string(
            "parameter_fingerprint",
            self.parameter_fingerprint,
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
            .source_normalization_output
            .source_inputs
        )

    @property
    def source_score_output(
        self,
    ) -> EdgeAttentionScoreOutput:
        return (
            self
            .source_normalization_output
            .source_score_output
        )

    @property
    def raw_scores_by_head(
        self,
    ) -> torch.Tensor:
        return (
            self
            .source_score_output
            .raw_scores_by_head
        )

    @property
    def normalized_weights_by_head(
        self,
    ) -> torch.Tensor:
        return (
            self
            .source_normalization_output
            .normalized_weights_by_head
        )

    @property
    def group_ids(self) -> torch.Tensor:
        return (
            self
            .source_normalization_output
            .group_ids
        )

    @property
    def group_counts(self) -> torch.Tensor:
        return (
            self
            .source_normalization_output
            .group_counts
        )

    @property
    def attention_mode(self) -> str:
        return (
            self
            .source_score_output
            .attention_mode
        )

    @property
    def normalization_mode(self) -> str:
        return (
            self
            .source_normalization_output
            .normalization_mode
        )

    @property
    def num_heads(self) -> int:
        return (
            self
            .source_score_output
            .num_heads
        )

    @property
    def num_groups(self) -> int:
        return (
            self
            .source_normalization_output
            .num_groups
        )

    @property
    def device(self) -> torch.device:
        return self.edge_weights.device

    @property
    def dtype(self) -> torch.dtype:
        return self.edge_weights.dtype

    def lineage_dict(
        self,
    ) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "head_reduction": self.head_reduction,
            "num_heads": self.num_heads,
            "source_normalization_fingerprint": (
                self
                .source_normalization_output
                .fingerprint()
            ),
            "encoder_architecture_fingerprint": (
                self
                .encoder_architecture_fingerprint
            ),
            "parameter_fingerprint": (
                self.parameter_fingerprint
            ),
        }

    def lineage_fingerprint(
        self,
    ) -> str:
        return _fingerprint(
            self.lineage_dict()
        )

    def value_fingerprint(
        self,
    ) -> str:
        return _tensor_fingerprint(
            {
                "edge_weights": self.edge_weights,
            }
        )

    def fingerprint(
        self,
    ) -> str:
        return _fingerprint(
            {
                "lineage_fingerprint": (
                    self.lineage_fingerprint()
                ),
                "value_fingerprint": (
                    self.value_fingerprint()
                ),
            }
        )


__all__ = (
    "ATTENTION_HEAD_REDUCTION_OUTPUT_SCHEMA_VERSION",
    "ATTENTION_NORMALIZATION_OUTPUT_SCHEMA_VERSION",
    "EDGE_ATTENTION_INPUT_EXACT_RELATION_EMBEDDING",
    "EDGE_ATTENTION_INPUT_SOURCE_NODE_STATE",
    "EDGE_ATTENTION_INPUT_TARGET_HAZARD_QUERY",
    "EDGE_ATTENTION_INPUT_TARGET_NODE_STATE",
    "EDGE_ATTENTION_SCHEMA_MODES",
    "EDGE_ATTENTION_SCORE_FUNCTION_ADDITIVE",
    "EDGE_ATTENTION_SCORE_FUNCTION_UNIFORM",
    "EDGE_ATTENTION_SCORE_OUTPUT_SCHEMA_VERSION",
    "AttentionHeadReductionOutput",
    "AttentionNormalizationOutput",
    "EdgeAttentionOutput",
    "EdgeAttentionScoreOutput",
)
