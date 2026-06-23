"""
Edge-attention score functions for hazard-conditioned functional message passing.

Target path:
    urban_resilience_models/
        v2_hazard_conditioned_functional_ugnn/
            functional_message_passing/
                edge_attention/
                    score_functions.py

This module owns edge-level *logit prediction only*. It does not normalize
logits, reduce attention heads, apply relation gates, transform messages,
consume semantic edge weights, or aggregate messages.

Role
---------------
For a stored directed edge ``e = (s_e -> t_e)`` with exact compiled relation
index ``r_e``, the learned bounded scorer asks:

    Under the current target-node context, hazard query, and relation
    identity, how compatible is source node s_e with target node t_e?

The bounded additive compatibility function is:

    z[e, a] =
        W_source[a] h_source[e]
        + W_target[a] h_target[e]
        + optional W_hazard[a] q_target[e]
        + relation_embedding[r_e, a]

    score[e, a] =
        v[a]^T tanh(z[e, a])

where:

- ``a`` indexes attention heads;
- ``h_source`` and ``h_target`` are rows of the shared fused node state;
- ``q_target`` is the explicitly node-aligned hazard query;
- relation embeddings are indexed by the dense exact compiled relation axis;
- stable ontology IDs remain metadata and are never parameter-table indices.

Why additive interaction is necessary
-------------------------------------
Attention is normalized later within each fixed target-node/exact-relation
group. Any purely additive scalar depending only on target, relation, or hazard
would be constant across the competing edges in that group and would disappear
under softmax. The nonlinear compatibility state allows contextual terms to
change how source-varying information is evaluated and therefore permits
hazard-dependent pairwise logit differences.

The implementation deliberately contains no standalone relation bias or
hazard bias. Such terms would be group-constant and behaviorally ineffective
after grouped softmax.

Implemented score modes
-----------------------
``uniform``
    Parameter-free exact-zero logits with shape ``[E, 1]``. The later grouped
    softmax produces reciprocal group-size weights. This is enabled uniform
    attention, not the disabled-attention multiplicative identity.

``hazard_blind``
    Learned single-head additive compatibility using source state, target
    state, and exact relation identity. This is the key scientific ablation
    for separating generic learned neighbor selection from hazard use.

``hazard_conditioned``
    Learned single-head additive compatibility adding the target-node hazard
    query.

``multihead_hazard_conditioned``
    The same formula with ``A >= 2`` independently parameterized heads. The
    tensor contract is implemented for controlled experiments, but downstream
    capability manifests should not claim head specialization without
    empirical evidence.

Bounded exclusions
------------------
The scorer does not consume:

- relation-gate values;
- transformed source messages;
- edge attributes;
- semantic edge weights;
- structural edge normalization;
- attention masks;
- aggregation statistics.

Those quantities have distinct scientific meanings and are composed later by
their owning modules. Edge attributes remain preserved in
``FunctionalMessagePassingInputs`` for a future explicitly configured scorer;
the first implementation does not infer an architecture width from whichever
batch happens to arrive.

Initialization
--------------
Source, target, hazard, relation, and head-vector parameters use Xavier
initialization. Context-producing parameters are additionally scaled at
initialization by ``1 / sqrt(number_of_context_components)``. This preserves
the mathematical score formula while reducing the risk that the initial sum
immediately drives ``tanh`` into saturation.

The module exposes preactivation construction for controlled diagnostics, but
does not attach large preactivation tensors or gradient-sensitivity metrics to
every ordinary output.

Contract
--------
- input is a validated ``FunctionalMessagePassingInputs``;
- raw scores have shape ``[E, A]``;
- output dtype and device exactly match the fused node state;
- relation order must exactly match the architecture's compiled relation axis;
- conditioned modes require a node-aligned finite hazard query;
- no hidden device movement or dtype conversion occurs during forward;
- empty edge sets return finite ``[0, A]`` tensors;
- parameters and outputs must remain finite;
- outputs preserve complete score-stage metadata through
  ``EdgeAttentionScoreOutput``.
"""

from __future__ import annotations

from hashlib import sha256
import json
import math
from typing import Any, Final, Mapping, Sequence, TypeAlias

import torch
from torch import nn

from ...config import (
    FunctionalMessagePassingConfig,
)
from ...constants import (
    ATTENTION_MODE_HAZARD_BLIND,
    ATTENTION_MODE_HAZARD_CONDITIONED,
    ATTENTION_MODE_MULTIHEAD_HAZARD_CONDITIONED,
    ATTENTION_MODE_SEMANTIC_WEIGHT,
    ATTENTION_MODE_UNIFORM,
    CANONICAL_ATTENTION_MODES,
)
from ..schemas import (
    FunctionalMessagePassingInputs,
)
from .schemas import (
    EDGE_ATTENTION_INPUT_EXACT_RELATION_EMBEDDING,
    EDGE_ATTENTION_INPUT_SOURCE_NODE_STATE,
    EDGE_ATTENTION_INPUT_TARGET_HAZARD_QUERY,
    EDGE_ATTENTION_INPUT_TARGET_NODE_STATE,
    EDGE_ATTENTION_SCORE_FUNCTION_ADDITIVE,
    EDGE_ATTENTION_SCORE_FUNCTION_UNIFORM,
    EdgeAttentionScoreOutput,
)


# =============================================================================
# Public identity
# =============================================================================


EDGE_ATTENTION_SCORE_FUNCTIONS_SCHEMA_VERSION: Final[str] = "0.1"

DEFAULT_EDGE_ATTENTION_HIDDEN_DIM: Final[int] = 64

LEARNED_EDGE_ATTENTION_MODES: Final[tuple[str, ...]] = (
    ATTENTION_MODE_HAZARD_BLIND,
    ATTENTION_MODE_HAZARD_CONDITIONED,
    ATTENTION_MODE_MULTIHEAD_HAZARD_CONDITIONED,
)

EDGE_ATTENTION_SCORE_MODES: Final[tuple[str, ...]] = (
    ATTENTION_MODE_UNIFORM,
    *LEARNED_EDGE_ATTENTION_MODES,
)


# =============================================================================
# Validation and fingerprint helpers
# =============================================================================


def _require_positive_int(
    name: str,
    value: int,
) -> int:
    if (
        isinstance(value, bool)
        or not isinstance(value, int)
    ):
        raise TypeError(
            f"{name} must be an integer."
        )

    if value <= 0:
        raise ValueError(
            f"{name} must be positive."
        )

    return value


def _require_nonempty_string(
    name: str,
    value: str,
) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(
            f"{name} must be a non-empty string."
        )


def _require_unique_strings(
    name: str,
    values: Sequence[str],
) -> tuple[str, ...]:
    normalized = tuple(values)
    seen: set[str] = set()
    duplicates: set[str] = set()

    for index, value in enumerate(
        normalized
    ):
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

    return normalized


def _require_unique_nonnegative_ints(
    name: str,
    values: Sequence[int],
) -> tuple[int, ...]:
    normalized = tuple(values)
    seen: set[int] = set()
    duplicates: set[int] = set()

    for index, value in enumerate(
        normalized
    ):
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

    return normalized


def _normalize_mode(
    mode: str,
) -> str:
    if not isinstance(mode, str):
        raise TypeError(
            "attention mode must be a string."
        )

    normalized = mode.strip()

    if not normalized:
        raise ValueError(
            "attention mode must be a non-empty string."
        )

    if normalized not in CANONICAL_ATTENTION_MODES:
        raise ValueError(
            f"Unknown attention mode {normalized!r}. Expected one of "
            f"{tuple(CANONICAL_ATTENTION_MODES)!r}."
        )

    if normalized == ATTENTION_MODE_SEMANTIC_WEIGHT:
        raise NotImplementedError(
            "semantic_weight is a data-coefficient mode, not a learned "
            "or uniform edge-score function in the bounded V2.0 "
            "edge-attention package."
        )

    if normalized not in EDGE_ATTENTION_SCORE_MODES:
        raise NotImplementedError(
            f"Attention mode {normalized!r} does not have a bounded "
            "V2.0 score-function implementation."
        )

    return normalized


def _validate_mode_and_head_count(
    *,
    mode: str,
    num_heads: int,
) -> None:
    _require_positive_int(
        "num_heads",
        num_heads,
    )

    if mode == (
        ATTENTION_MODE_MULTIHEAD_HAZARD_CONDITIONED
    ):
        if num_heads < 2:
            raise ValueError(
                "multihead_hazard_conditioned attention requires at "
                "least two heads."
            )
        return

    if num_heads != 1:
        raise ValueError(
            f"Attention mode {mode!r} requires exactly one head in the "
            "bounded V2.0 score contract."
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


def _require_float_matrix(
    name: str,
    value: torch.Tensor,
    *,
    shape: tuple[int, int],
    dtype: torch.dtype,
    device: torch.device,
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
            f"{name} must have rank 2; observed shape "
            f"{tuple(value.shape)}."
        )

    if tuple(value.shape) != shape:
        raise ValueError(
            f"{name} must have shape {shape}; observed "
            f"{tuple(value.shape)}."
        )

    if not value.dtype.is_floating_point:
        raise ValueError(
            f"{name} must use a floating-point dtype."
        )

    if value.dtype != dtype:
        raise ValueError(
            f"{name} must use dtype {dtype}; observed {value.dtype}."
        )

    if not _devices_match(
        value.device,
        device,
    ):
        raise ValueError(
            f"{name} must be on device {device}; observed "
            f"{value.device}."
        )

    if not bool(
        torch.isfinite(value)
        .all()
        .item()
    ):
        raise ValueError(
            f"{name} must contain only finite values."
        )


def _require_float_tensor_3d(
    name: str,
    value: torch.Tensor,
    *,
    shape: tuple[int, int, int],
    dtype: torch.dtype,
    device: torch.device,
) -> None:
    if not isinstance(
        value,
        torch.Tensor,
    ):
        raise TypeError(
            f"{name} must be a tensor."
        )

    if value.ndim != 3:
        raise ValueError(
            f"{name} must have rank 3; observed shape "
            f"{tuple(value.shape)}."
        )

    if tuple(value.shape) != shape:
        raise ValueError(
            f"{name} must have shape {shape}; observed "
            f"{tuple(value.shape)}."
        )

    if not value.dtype.is_floating_point:
        raise ValueError(
            f"{name} must use a floating-point dtype."
        )

    if value.dtype != dtype:
        raise ValueError(
            f"{name} must use dtype {dtype}; observed {value.dtype}."
        )

    if not _devices_match(
        value.device,
        device,
    ):
        raise ValueError(
            f"{name} must be on device {device}; observed "
            f"{value.device}."
        )

    if not bool(
        torch.isfinite(value)
        .all()
        .item()
    ):
        raise ValueError(
            f"{name} must contain only finite values."
        )


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


def _relation_axis_fingerprint(
    *,
    relation_names: Sequence[str],
    stable_relation_ids: Sequence[int],
) -> str:
    return _fingerprint(
        {
            "relation_names": list(
                relation_names
            ),
            "stable_relation_ids": list(
                stable_relation_ids
            ),
        }
    )


# =============================================================================
# Common exact-relation score-function base
# =============================================================================


class _ExactRelationScoreFunctionBase(nn.Module):
    """
    Shared architecture identity and runtime alignment validation.

    The base class intentionally does not define score mathematics.
    """

    mode: str
    relation_names: tuple[str, ...]
    stable_relation_ids: tuple[int, ...]
    num_heads: int

    def __init__(
        self,
        *,
        mode: str,
        relation_names: Sequence[str],
        stable_relation_ids: Sequence[int],
        num_heads: int,
    ) -> None:
        super().__init__()

        self.mode = _normalize_mode(
            mode
        )
        self.relation_names = (
            _require_unique_strings(
                "relation_names",
                relation_names,
            )
        )
        self.stable_relation_ids = (
            _require_unique_nonnegative_ints(
                "stable_relation_ids",
                stable_relation_ids,
            )
        )

        if not self.relation_names:
            raise ValueError(
                "At least one exact compiled relation is required."
            )

        if len(self.relation_names) != len(
            self.stable_relation_ids
        ):
            raise ValueError(
                "relation_names and stable_relation_ids must align."
            )

        self.num_heads = (
            _require_positive_int(
                "num_heads",
                num_heads,
            )
        )
        _validate_mode_and_head_count(
            mode=self.mode,
            num_heads=self.num_heads,
        )

    @property
    def num_relations(self) -> int:
        return len(self.relation_names)

    @property
    def relation_axis_fingerprint(
        self,
    ) -> str:
        return _relation_axis_fingerprint(
            relation_names=(
                self.relation_names
            ),
            stable_relation_ids=(
                self.stable_relation_ids
            ),
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

    def _validate_relation_alignment(
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
                "Runtime relation_names differ from the exact compiled "
                "relation order used to construct the edge-attention "
                "score function."
            )

        if source_inputs.stable_relation_ids != (
            self.stable_relation_ids
        ):
            raise ValueError(
                "Runtime stable_relation_ids differ from the exact "
                "compiled relation order used to construct the "
                "edge-attention score function."
            )

        if source_inputs.num_relations != (
            self.num_relations
        ):
            raise ValueError(
                "Runtime relation count differs from the score-function "
                "architecture."
            )

    def architecture_dict(
        self,
    ) -> dict[str, Any]:
        raise NotImplementedError

    def architecture_fingerprint(
        self,
    ) -> str:
        return _fingerprint(
            self.architecture_dict()
        )

    def parameter_fingerprint(
        self,
    ) -> str | None:
        state = {
            key: value
            for key, value
            in self.state_dict().items()
        }

        if not state:
            return None

        return _tensor_fingerprint(
            state
        )

    def assert_finite_parameters(
        self,
    ) -> None:
        for name, parameter in (
            self.named_parameters()
        ):
            if not bool(
                torch.isfinite(parameter)
                .all()
                .item()
            ):
                raise FloatingPointError(
                    "Edge-attention score-function parameter "
                    f"{name!r} contains NaN or infinity."
                )


# =============================================================================
# Uniform score function
# =============================================================================


class UniformEdgeAttentionScoreFunction(
    _ExactRelationScoreFunctionBase
):
    """
    Parameter-free exact-zero edge logits.

    Later grouped normalization converts these logits into uniform weights
    within every nonempty target-node/exact-relation group.
    """

    def __init__(
        self,
        *,
        relation_names: Sequence[str],
        stable_relation_ids: Sequence[int],
    ) -> None:
        super().__init__(
            mode=ATTENTION_MODE_UNIFORM,
            relation_names=relation_names,
            stable_relation_ids=(
                stable_relation_ids
            ),
            num_heads=1,
        )

    @classmethod
    def from_config(
        cls,
        *,
        config: FunctionalMessagePassingConfig,
        source_inputs: FunctionalMessagePassingInputs,
    ) -> "UniformEdgeAttentionScoreFunction":
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

        if config.attention_mode != (
            ATTENTION_MODE_UNIFORM
        ):
            raise ValueError(
                "UniformEdgeAttentionScoreFunction.from_config requires "
                "attention_mode='uniform'."
            )

        module = cls(
            relation_names=(
                source_inputs.relation_names
            ),
            stable_relation_ids=(
                source_inputs.stable_relation_ids
            ),
        )

        # Parameter-free, but move the module for consistent buffer behavior
        # if buffers are introduced in a compatible future schema version.
        return module.to(
            device=source_inputs.device,
            dtype=source_inputs.dtype,
        )

    @property
    def input_feature_names(
        self,
    ) -> tuple[str, ...]:
        return ()

    def architecture_dict(
        self,
    ) -> dict[str, Any]:
        return {
            "schema_version": (
                EDGE_ATTENTION_SCORE_FUNCTIONS_SCHEMA_VERSION
            ),
            "mode": self.mode,
            "score_function": (
                EDGE_ATTENTION_SCORE_FUNCTION_UNIFORM
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
            "relation_axis_fingerprint": (
                self.relation_axis_fingerprint
            ),
            "input_feature_names": [],
            "uses_node_state": False,
            "uses_hazard_query": False,
            "uses_relation_embedding": False,
            "uses_edge_attributes": False,
            "uses_relation_gate": False,
            "raw_score_identity": (
                "exact_zero_logits"
            ),
            "parameter_count": (
                self.parameter_count
            ),
            "output_schema": (
                "EdgeAttentionScoreOutput"
            ),
        }

    def score_tensor(
        self,
        source_inputs: FunctionalMessagePassingInputs,
    ) -> torch.Tensor:
        self._validate_relation_alignment(
            source_inputs
        )

        return torch.zeros(
            (
                source_inputs.num_edges,
                1,
            ),
            dtype=source_inputs.dtype,
            device=source_inputs.device,
        )

    def forward(
        self,
        source_inputs: FunctionalMessagePassingInputs,
    ) -> EdgeAttentionScoreOutput:
        self.assert_finite_parameters()
        raw_scores = self.score_tensor(
            source_inputs
        )

        return EdgeAttentionScoreOutput(
            raw_scores_by_head=raw_scores,
            source_inputs=source_inputs,
            relation_names=(
                self.relation_names
            ),
            stable_relation_ids=(
                self.stable_relation_ids
            ),
            compiled_relation_registry_fingerprint=(
                source_inputs
                .compiled_relation_registry
                .fingerprint()
            ),
            attention_mode=self.mode,
            score_function=(
                EDGE_ATTENTION_SCORE_FUNCTION_UNIFORM
            ),
            input_feature_names=(
                self.input_feature_names
            ),
            encoder_architecture_fingerprint=(
                self.architecture_fingerprint()
            ),
            parameter_fingerprint=None,
        )

    def extra_repr(self) -> str:
        return (
            f"num_relations={self.num_relations}, "
            f"mode={self.mode!r}, "
            "num_heads=1, parameter_free=True"
        )


# =============================================================================
# Learned additive compatibility score function
# =============================================================================


class AdditiveEdgeAttentionScoreFunction(
    _ExactRelationScoreFunctionBase
):
    """
    Exact-relation-aware additive compatibility scorer.

    Parameters
    ----------
    node_state_dim:
        Width of the fused shared node state.
    hazard_query_dim:
        Width of the node-aligned hazard query for conditioned modes.
        Must be ``None`` for ``hazard_blind``.
    relation_names:
        Exact compiled relation order used by the learned relation embedding
        table.
    stable_relation_ids:
        Stable ontology IDs aligned one-to-one with ``relation_names``.
    hidden_dim:
        Width of each head-specific additive compatibility state.
    mode:
        ``hazard_blind``, ``hazard_conditioned``, or
        ``multihead_hazard_conditioned``.
    num_heads:
        One for bounded single-head modes; at least two for the explicit
        multihead mode.
    """

    node_state_dim: int
    hazard_query_dim: int | None
    hidden_dim: int

    def __init__(
        self,
        *,
        node_state_dim: int,
        hazard_query_dim: int | None,
        relation_names: Sequence[str],
        stable_relation_ids: Sequence[int],
        hidden_dim: int = (
            DEFAULT_EDGE_ATTENTION_HIDDEN_DIM
        ),
        mode: str = (
            ATTENTION_MODE_HAZARD_CONDITIONED
        ),
        num_heads: int = 1,
    ) -> None:
        normalized_mode = _normalize_mode(
            mode
        )

        if normalized_mode not in (
            LEARNED_EDGE_ATTENTION_MODES
        ):
            raise ValueError(
                "AdditiveEdgeAttentionScoreFunction requires a learned "
                "attention mode."
            )

        super().__init__(
            mode=normalized_mode,
            relation_names=relation_names,
            stable_relation_ids=(
                stable_relation_ids
            ),
            num_heads=num_heads,
        )

        self.node_state_dim = (
            _require_positive_int(
                "node_state_dim",
                node_state_dim,
            )
        )
        self.hidden_dim = (
            _require_positive_int(
                "hidden_dim",
                hidden_dim,
            )
        )

        if self.uses_hazard_query:
            if hazard_query_dim is None:
                raise ValueError(
                    "Hazard-conditioned additive attention requires a "
                    "positive hazard_query_dim."
                )

            self.hazard_query_dim = (
                _require_positive_int(
                    "hazard_query_dim",
                    hazard_query_dim,
                )
            )
        else:
            if hazard_query_dim is not None:
                raise ValueError(
                    "hazard_query_dim must be None for hazard_blind "
                    "additive attention."
                )

            self.hazard_query_dim = None

        projection_width = (
            self.num_heads
            * self.hidden_dim
        )

        # Biases are intentionally omitted. Standalone context biases add no
        # useful identity and may create misleading group-constant terms.
        self.source_projection = nn.Linear(
            self.node_state_dim,
            projection_width,
            bias=False,
        )
        self.target_projection = nn.Linear(
            self.node_state_dim,
            projection_width,
            bias=False,
        )

        if self.uses_hazard_query:
            assert self.hazard_query_dim is not None
            self.hazard_projection: nn.Linear | None = (
                nn.Linear(
                    self.hazard_query_dim,
                    projection_width,
                    bias=False,
                )
            )
        else:
            self.hazard_projection = None

        self.relation_embeddings = nn.Parameter(
            torch.empty(
                self.num_relations,
                self.num_heads,
                self.hidden_dim,
            )
        )
        self.score_vectors = nn.Parameter(
            torch.empty(
                self.num_heads,
                self.hidden_dim,
            )
        )

        self.reset_parameters()

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    @classmethod
    def from_config(
        cls,
        *,
        config: FunctionalMessagePassingConfig,
        source_inputs: FunctionalMessagePassingInputs,
        hidden_dim: int = (
            DEFAULT_EDGE_ATTENTION_HIDDEN_DIM
        ),
    ) -> "AdditiveEdgeAttentionScoreFunction":
        """
        Build an additive scorer aligned to one FMP input contract.

        ``hidden_dim`` is explicit because the current
        ``FunctionalMessagePassingConfig`` does not yet own an attention
        compatibility width. The value is recorded in the architecture
        fingerprint and should become a configuration field before broad
        experiment sweeps.
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

        mode = _normalize_mode(
            config.attention_mode
        )

        if mode not in LEARNED_EDGE_ATTENTION_MODES:
            raise ValueError(
                "AdditiveEdgeAttentionScoreFunction.from_config "
                "requires a learned attention mode."
            )

        node_state = (
            source_inputs
            .node_state
            .fused_state
        )
        _require_float_matrix(
            "source_inputs.node_state.fused_state",
            node_state,
            shape=(
                source_inputs.num_nodes,
                source_inputs.hidden_dim,
            ),
            dtype=source_inputs.dtype,
            device=source_inputs.device,
        )

        if mode == ATTENTION_MODE_HAZARD_BLIND:
            hazard_query_dim = None
        else:
            hazard_query = (
                source_inputs
                .node_hazard_query
            )

            if hazard_query is None:
                raise ValueError(
                    "Hazard-conditioned additive attention requires "
                    "source_inputs.node_hazard_query."
                )

            if (
                not isinstance(
                    hazard_query,
                    torch.Tensor,
                )
                or hazard_query.ndim != 2
            ):
                raise ValueError(
                    "source_inputs.node_hazard_query must have shape "
                    "[N, Q]."
                )

            if int(hazard_query.shape[1]) <= 0:
                raise ValueError(
                    "source_inputs.node_hazard_query must have a "
                    "positive feature width."
                )

            hazard_query_dim = int(
                hazard_query.shape[1]
            )

        module = cls(
            node_state_dim=(
                source_inputs.hidden_dim
            ),
            hazard_query_dim=(
                hazard_query_dim
            ),
            relation_names=(
                source_inputs.relation_names
            ),
            stable_relation_ids=(
                source_inputs.stable_relation_ids
            ),
            hidden_dim=hidden_dim,
            mode=mode,
            num_heads=(
                config.attention_heads
            ),
        )

        return module.to(
            device=source_inputs.device,
            dtype=source_inputs.dtype,
        )

    # ------------------------------------------------------------------
    # Public identity
    # ------------------------------------------------------------------

    @property
    def uses_hazard_query(self) -> bool:
        return self.mode in (
            ATTENTION_MODE_HAZARD_CONDITIONED,
            ATTENTION_MODE_MULTIHEAD_HAZARD_CONDITIONED,
        )

    @property
    def input_feature_names(
        self,
    ) -> tuple[str, ...]:
        common = (
            EDGE_ATTENTION_INPUT_SOURCE_NODE_STATE,
            EDGE_ATTENTION_INPUT_TARGET_NODE_STATE,
        )

        if self.uses_hazard_query:
            return (
                *common,
                EDGE_ATTENTION_INPUT_TARGET_HAZARD_QUERY,
                EDGE_ATTENTION_INPUT_EXACT_RELATION_EMBEDDING,
            )

        return (
            *common,
            EDGE_ATTENTION_INPUT_EXACT_RELATION_EMBEDDING,
        )

    @property
    def context_component_count(
        self,
    ) -> int:
        # source + target + exact relation + optional hazard
        return 4 if self.uses_hazard_query else 3

    @property
    def context_initialization_scale(
        self,
    ) -> float:
        return 1.0 / math.sqrt(
            float(
                self.context_component_count
            )
        )

    @property
    def projection_width(self) -> int:
        return (
            self.num_heads
            * self.hidden_dim
        )

    def architecture_dict(
        self,
    ) -> dict[str, Any]:
        return {
            "schema_version": (
                EDGE_ATTENTION_SCORE_FUNCTIONS_SCHEMA_VERSION
            ),
            "mode": self.mode,
            "score_function": (
                EDGE_ATTENTION_SCORE_FUNCTION_ADDITIVE
            ),
            "score_formula": (
                "v_head^T tanh("
                "W_source_head h_source + "
                "W_target_head h_target + "
                "optional W_hazard_head q_target + "
                "exact_relation_embedding)"
            ),
            "node_state_dim": (
                self.node_state_dim
            ),
            "hazard_query_dim": (
                self.hazard_query_dim
            ),
            "hidden_dim": self.hidden_dim,
            "num_heads": self.num_heads,
            "projection_width": (
                self.projection_width
            ),
            "num_relations": (
                self.num_relations
            ),
            "relation_names": list(
                self.relation_names
            ),
            "stable_relation_ids": list(
                self.stable_relation_ids
            ),
            "relation_axis_fingerprint": (
                self.relation_axis_fingerprint
            ),
            "input_feature_names": list(
                self.input_feature_names
            ),
            "uses_node_state": True,
            "uses_hazard_query": (
                self.uses_hazard_query
            ),
            "uses_relation_embedding": True,
            "uses_edge_attributes": False,
            "uses_relation_gate": False,
            "compatibility_activation": (
                "tanh"
            ),
            "projection_bias": False,
            "standalone_relation_bias": False,
            "standalone_hazard_bias": False,
            "relation_identity_parameterization": (
                "learned_exact_relation_embeddings"
            ),
            "head_parameterization": (
                "independent_projection_slices_and_score_vectors"
            ),
            "context_component_count": (
                self.context_component_count
            ),
            "context_initialization_scale": (
                self.context_initialization_scale
            ),
            "parameter_count": (
                self.parameter_count
            ),
            "output_schema": (
                "EdgeAttentionScoreOutput"
            ),
        }

    # ------------------------------------------------------------------
    # Initialization and parameter checks
    # ------------------------------------------------------------------

    def reset_parameters(
        self,
    ) -> None:
        scale = (
            self.context_initialization_scale
        )

        nn.init.xavier_uniform_(
            self.source_projection.weight
        )
        nn.init.xavier_uniform_(
            self.target_projection.weight
        )

        with torch.no_grad():
            self.source_projection.weight.mul_(
                scale
            )
            self.target_projection.weight.mul_(
                scale
            )

        if self.hazard_projection is not None:
            nn.init.xavier_uniform_(
                self.hazard_projection.weight
            )
            with torch.no_grad():
                (
                    self
                    .hazard_projection
                    .weight
                    .mul_(scale)
                )

        relation_matrix = (
            self.relation_embeddings.view(
                self.num_relations,
                self.projection_width,
            )
        )
        nn.init.xavier_uniform_(
            relation_matrix
        )
        with torch.no_grad():
            self.relation_embeddings.mul_(
                scale
            )

        nn.init.xavier_uniform_(
            self.score_vectors
        )

    def _parameter_device_dtype(
        self,
    ) -> tuple[
        torch.device,
        torch.dtype,
    ]:
        parameters = tuple(
            self.parameters()
        )

        if not parameters:
            raise RuntimeError(
                "Additive edge-attention scorer unexpectedly has no "
                "parameters."
            )

        device = parameters[0].device
        dtype = parameters[0].dtype

        if not dtype.is_floating_point:
            raise RuntimeError(
                "Edge-attention scorer parameters must use a "
                "floating-point dtype."
            )

        for parameter in parameters[1:]:
            if not _devices_match(
                parameter.device,
                device,
            ):
                raise RuntimeError(
                    "Edge-attention scorer parameters must share one "
                    "device."
                )

            if parameter.dtype != dtype:
                raise RuntimeError(
                    "Edge-attention scorer parameters must share one "
                    "floating-point dtype."
                )

        return device, dtype

    # ------------------------------------------------------------------
    # Runtime validation and tensor construction
    # ------------------------------------------------------------------

    def _validate_runtime_inputs(
        self,
        source_inputs: FunctionalMessagePassingInputs,
    ) -> None:
        self._validate_relation_alignment(
            source_inputs
        )

        parameter_device, parameter_dtype = (
            self._parameter_device_dtype()
        )

        if not _devices_match(
            parameter_device,
            source_inputs.device,
        ):
            raise ValueError(
                "Edge-attention scorer parameters and source_inputs "
                "must share one device. Observed "
                f"{parameter_device} and {source_inputs.device}."
            )

        if parameter_dtype != source_inputs.dtype:
            raise ValueError(
                "Edge-attention scorer parameters and source_inputs "
                "must share one floating-point dtype. Observed "
                f"{parameter_dtype} and {source_inputs.dtype}."
            )

        node_state = (
            source_inputs
            .node_state
            .fused_state
        )
        _require_float_matrix(
            "source_inputs.node_state.fused_state",
            node_state,
            shape=(
                source_inputs.num_nodes,
                self.node_state_dim,
            ),
            dtype=source_inputs.dtype,
            device=source_inputs.device,
        )

        if self.uses_hazard_query:
            hazard_query = (
                source_inputs
                .node_hazard_query
            )

            if hazard_query is None:
                raise ValueError(
                    "Hazard-conditioned edge attention requires "
                    "source_inputs.node_hazard_query."
                )

            assert self.hazard_query_dim is not None
            _require_float_matrix(
                "source_inputs.node_hazard_query",
                hazard_query,
                shape=(
                    source_inputs.num_nodes,
                    self.hazard_query_dim,
                ),
                dtype=source_inputs.dtype,
                device=source_inputs.device,
            )

    def _reshape_projection(
        self,
        value: torch.Tensor,
        *,
        edge_count: int,
    ) -> torch.Tensor:
        return value.reshape(
            edge_count,
            self.num_heads,
            self.hidden_dim,
        )

    def projected_source_state(
        self,
        source_inputs: FunctionalMessagePassingInputs,
    ) -> torch.Tensor:
        """
        Return ``W_source h_source`` with shape ``[E, A, D]``.
        """

        self._validate_runtime_inputs(
            source_inputs
        )
        source_state = (
            source_inputs
            .node_state
            .fused_state[
                source_inputs.source_index
            ]
        )
        projected = (
            self.source_projection(
                source_state
            )
        )
        output = self._reshape_projection(
            projected,
            edge_count=(
                source_inputs.num_edges
            ),
        )

        _require_float_tensor_3d(
            "projected_source_state",
            output,
            shape=(
                source_inputs.num_edges,
                self.num_heads,
                self.hidden_dim,
            ),
            dtype=source_inputs.dtype,
            device=source_inputs.device,
        )
        return output

    def projected_target_state(
        self,
        source_inputs: FunctionalMessagePassingInputs,
    ) -> torch.Tensor:
        """
        Return ``W_target h_target`` with shape ``[E, A, D]``.
        """

        self._validate_runtime_inputs(
            source_inputs
        )
        target_state = (
            source_inputs
            .node_state
            .fused_state[
                source_inputs.target_index
            ]
        )
        projected = (
            self.target_projection(
                target_state
            )
        )
        output = self._reshape_projection(
            projected,
            edge_count=(
                source_inputs.num_edges
            ),
        )

        _require_float_tensor_3d(
            "projected_target_state",
            output,
            shape=(
                source_inputs.num_edges,
                self.num_heads,
                self.hidden_dim,
            ),
            dtype=source_inputs.dtype,
            device=source_inputs.device,
        )
        return output

    def projected_target_hazard_query(
        self,
        source_inputs: FunctionalMessagePassingInputs,
    ) -> torch.Tensor | None:
        """
        Return ``W_hazard q_target`` with shape ``[E, A, D]``.

        Hazard-blind attention returns ``None`` and does not read the runtime
        hazard-query tensor even if one is preserved in ``source_inputs``.
        """

        self._validate_runtime_inputs(
            source_inputs
        )

        if not self.uses_hazard_query:
            return None

        if self.hazard_projection is None:
            raise RuntimeError(
                "Hazard-conditioned scorer is missing its hazard "
                "projection."
            )

        hazard_query = (
            source_inputs
            .node_hazard_query
        )

        if hazard_query is None:
            raise RuntimeError(
                "Validated hazard query unexpectedly disappeared."
            )

        target_query = hazard_query[
            source_inputs.target_index
        ]
        projected = self.hazard_projection(
            target_query
        )
        output = self._reshape_projection(
            projected,
            edge_count=(
                source_inputs.num_edges
            ),
        )

        _require_float_tensor_3d(
            "projected_target_hazard_query",
            output,
            shape=(
                source_inputs.num_edges,
                self.num_heads,
                self.hidden_dim,
            ),
            dtype=source_inputs.dtype,
            device=source_inputs.device,
        )
        return output

    def edge_relation_embeddings(
        self,
        source_inputs: FunctionalMessagePassingInputs,
    ) -> torch.Tensor:
        """
        Gather exact-relation embeddings with shape ``[E, A, D]``.
        """

        self._validate_runtime_inputs(
            source_inputs
        )
        output = self.relation_embeddings[
            source_inputs.edge_relation_index
        ]

        _require_float_tensor_3d(
            "edge_relation_embeddings",
            output,
            shape=(
                source_inputs.num_edges,
                self.num_heads,
                self.hidden_dim,
            ),
            dtype=source_inputs.dtype,
            device=source_inputs.device,
        )
        return output

    def compatibility_preactivations(
        self,
        source_inputs: FunctionalMessagePassingInputs,
    ) -> torch.Tensor:
        """
        Build the additive compatibility state ``z[e, a, :]``.

        This method is intentionally exposed for controlled diagnostics of
        near-linear and saturated ``tanh`` regimes. The returned tensor is not
        retained automatically in ordinary score outputs.
        """

        self._validate_runtime_inputs(
            source_inputs
        )

        source_term = (
            self.projected_source_state(
                source_inputs
            )
        )
        target_term = (
            self.projected_target_state(
                source_inputs
            )
        )
        relation_term = (
            self.edge_relation_embeddings(
                source_inputs
            )
        )

        preactivations = (
            source_term
            + target_term
            + relation_term
        )

        hazard_term = (
            self.projected_target_hazard_query(
                source_inputs
            )
        )
        if hazard_term is not None:
            preactivations = (
                preactivations
                + hazard_term
            )

        _require_float_tensor_3d(
            "compatibility_preactivations",
            preactivations,
            shape=(
                source_inputs.num_edges,
                self.num_heads,
                self.hidden_dim,
            ),
            dtype=source_inputs.dtype,
            device=source_inputs.device,
        )
        return preactivations

    def compatibility_state(
        self,
        source_inputs: FunctionalMessagePassingInputs,
    ) -> torch.Tensor:
        """
        Apply ``tanh`` to compatibility preactivations.
        """

        state = torch.tanh(
            self.compatibility_preactivations(
                source_inputs
            )
        )

        _require_float_tensor_3d(
            "compatibility_state",
            state,
            shape=(
                source_inputs.num_edges,
                self.num_heads,
                self.hidden_dim,
            ),
            dtype=source_inputs.dtype,
            device=source_inputs.device,
        )
        return state

    def score_tensor(
        self,
        source_inputs: FunctionalMessagePassingInputs,
    ) -> torch.Tensor:
        """
        Return raw additive logits with shape ``[E, A]``.
        """

        state = self.compatibility_state(
            source_inputs
        )
        scores = torch.einsum(
            "eah,ah->ea",
            state,
            self.score_vectors,
        )

        _require_float_matrix(
            "raw_edge_attention_scores",
            scores,
            shape=(
                source_inputs.num_edges,
                self.num_heads,
            ),
            dtype=source_inputs.dtype,
            device=source_inputs.device,
        )
        return scores

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self,
        source_inputs: FunctionalMessagePassingInputs,
    ) -> EdgeAttentionScoreOutput:
        self.assert_finite_parameters()
        self._validate_runtime_inputs(
            source_inputs
        )

        raw_scores = self.score_tensor(
            source_inputs
        )

        return EdgeAttentionScoreOutput(
            raw_scores_by_head=raw_scores,
            source_inputs=source_inputs,
            relation_names=(
                self.relation_names
            ),
            stable_relation_ids=(
                self.stable_relation_ids
            ),
            compiled_relation_registry_fingerprint=(
                source_inputs
                .compiled_relation_registry
                .fingerprint()
            ),
            attention_mode=self.mode,
            score_function=(
                EDGE_ATTENTION_SCORE_FUNCTION_ADDITIVE
            ),
            input_feature_names=(
                self.input_feature_names
            ),
            encoder_architecture_fingerprint=(
                self.architecture_fingerprint()
            ),
            parameter_fingerprint=(
                self.parameter_fingerprint()
            ),
        )

    # ------------------------------------------------------------------
    # Representation
    # ------------------------------------------------------------------

    def extra_repr(self) -> str:
        return (
            f"node_state_dim={self.node_state_dim}, "
            f"hazard_query_dim={self.hazard_query_dim}, "
            f"hidden_dim={self.hidden_dim}, "
            f"num_relations={self.num_relations}, "
            f"num_heads={self.num_heads}, "
            f"mode={self.mode!r}, "
            f"uses_hazard_query={self.uses_hazard_query}, "
            "uses_edge_attributes=False"
        )


# =============================================================================
# Public construction dispatcher
# =============================================================================


EdgeAttentionScoreFunction: TypeAlias = (
    UniformEdgeAttentionScoreFunction
    | AdditiveEdgeAttentionScoreFunction
)


def build_edge_attention_score_function(
    *,
    config: FunctionalMessagePassingConfig,
    source_inputs: FunctionalMessagePassingInputs,
    hidden_dim: int = (
        DEFAULT_EDGE_ATTENTION_HIDDEN_DIM
    ),
) -> EdgeAttentionScoreFunction:
    """
    Construct the score-function implementation selected by configuration.

    This dispatcher validates canonical configuration but does not call
    ``config.assert_implemented()``. Score-function implementation is only one
    part of the complete attention subsystem; capability manifests should be
    updated after score, normalization, head reduction, orchestration, and
    focused tests all agree.
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

    mode = _normalize_mode(
        config.attention_mode
    )

    if mode == ATTENTION_MODE_UNIFORM:
        return (
            UniformEdgeAttentionScoreFunction
            .from_config(
                config=config,
                source_inputs=source_inputs,
            )
        )

    return (
        AdditiveEdgeAttentionScoreFunction
        .from_config(
            config=config,
            source_inputs=source_inputs,
            hidden_dim=hidden_dim,
        )
    )


# Compact aliases for call sites and future package exports.
UniformAttentionScoreFunction = (
    UniformEdgeAttentionScoreFunction
)
AdditiveAttentionScoreFunction = (
    AdditiveEdgeAttentionScoreFunction
)
build_attention_score_function = (
    build_edge_attention_score_function
)


__all__ = (
    "DEFAULT_EDGE_ATTENTION_HIDDEN_DIM",
    "EDGE_ATTENTION_SCORE_FUNCTIONS_SCHEMA_VERSION",
    "EDGE_ATTENTION_SCORE_MODES",
    "LEARNED_EDGE_ATTENTION_MODES",
    "AdditiveAttentionScoreFunction",
    "AdditiveEdgeAttentionScoreFunction",
    "EdgeAttentionScoreFunction",
    "UniformAttentionScoreFunction",
    "UniformEdgeAttentionScoreFunction",
    "build_attention_score_function",
    "build_edge_attention_score_function",
)
