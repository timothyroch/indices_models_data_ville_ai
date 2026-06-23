"""
Exact target-node/relation normalization for edge-attention logits.

Target path:
    urban_resilience_models/
        v2_hazard_conditioned_functional_ugnn/
            functional_message_passing/
                edge_attention/
                    attention_normalization.py

This module owns one bounded operation:

    convert finite edge logits ``[E, A]`` into nonnegative attention weights
    ``[E, A]`` by applying a numerically stable softmax independently within
    every target-node/exact-relation group and every attention head.

It does not own:

- edge-score prediction;
- relation-gate computation;
- attention-head reduction;
- edge masking;
- relation transforms;
- structural edge normalization;
- semantic edge weights;
- message construction;
- target-node aggregation;
- causal or faithfulness interpretation.

Separation
----------------------
For each stored directed edge ``e = (s_e -> t_e)`` with dense exact compiled
relation index ``r_e``, the bounded group is:

    gamma(e) = (t_e, r_e)

and the deterministic dense group ID is:

    group_id[e] =
        target_index[e] * num_relations
        + edge_relation_index[e]

with:

    0 <= group_id[e] < num_nodes * num_relations

For every nonempty group ``g`` and every head ``a``:

    alpha[e, a]
        = exp(logit[e, a] - max_{j in g} logit[j, a])
          -------------------------------------------------
          sum_{j in g} exp(logit[j, a] - max_{k in g} logit[k, a])

and:

    sum_{e in g} alpha[e, a] = 1

The maximum subtraction is required for numerical stability and is delegated
to the validated generic ``segment_ops.grouped_softmax`` implementation.

Why exact target-relation grouping matters
------------------------------------------
The relation gate and edge attention intentionally answer different questions:

    relation gate:
        How strongly should relation mechanism r contribute at target t?

    edge attention:
        Among the concrete incoming edges of relation r at target t, which
        sources should receive routing mass?

Normalizing attention across exact target-relation groups preserves this
separation. Relations do not compete inside attention; they are controlled
later by independent sigmoid relation gates.

This is hierarchical functional routing, not a probabilistic mixture across
relations. Attention sums to one only inside each nonempty target-relation
group. Relation gates need not sum to one.

Group-constant invariance
-------------------------
For any scalar constant shared by every edge in one group:

    softmax(logit[e] + constant[group(e)])
        == softmax(logit[e])

This invariance is mathematically correct and is one reason the learned score
function must let hazard/target/relation context modify source-dependent
pairwise logit differences rather than merely add a group-constant bias.

Disabled versus uniform attention
---------------------------------
This module represents *enabled* attention normalization.

    attention disabled:
        no softmax is performed;
        downstream multiplicative coefficient is 1.

    uniform attention enabled:
        score function emits exact zero logits;
        grouped softmax produces 1 / group_size.

Those behaviors are not interchangeable.

Boundary behavior
-----------------
- ``E = 0`` is valid and returns ``[0, A]`` weights.
- A compiled relation may have zero edges in a batch.
- Absent groups retain count zero and produce no synthetic entries.
- A singleton group receives exact weight one for every head.
- Packed graphs remain isolated because node indices are globally packed and
  upstream FMP validation rejects cross-graph edges.
- Control/placebo relations use the same normalization mathematics; their
  substantive interpretation is an exporter policy.
- No hidden device transfer or floating-point cast occurs.
- Gradients flow through grouped softmax to raw logits.
- The module is parameter-free.

Runtime and audit philosophy
----------------------------
``AttentionNormalizationOutput`` performs mandatory schema-level validation of
group identity, counts, nonnegativity, singleton exactness, and per-head group
sums. This module additionally validates the direct grouped-softmax result
before constructing that metadata-bearing output. The duplicate check is
intentional at this critical numerical boundary: one validation protects the
operation itself, while the schema protects independently constructed or
deserialized outputs.

The complete attention subsystem should update capability manifests only after
score prediction, normalization, head reduction, orchestration, and their
focused tests all agree. Therefore ``from_config`` validates canonical
configuration but does not call ``config.assert_implemented()``.
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
    ATTENTION_NORMALIZATION_TARGET_NODE_RELATION,
    CANONICAL_ATTENTION_NORMALIZATION_MODES,
)
from ..schemas import (
    FunctionalMessagePassingInputs,
)
from ..segment_ops import (
    SEGMENT_OPS_SCHEMA_VERSION,
    assert_grouped_softmax_normalized,
    grouped_softmax,
    grouped_softmax_sums,
    segment_counts,
    validate_segment_ids,
)
from .schemas import (
    AttentionNormalizationOutput,
    EdgeAttentionScoreOutput,
)


# =============================================================================
# Public identity
# =============================================================================


ATTENTION_NORMALIZATION_SCHEMA_VERSION: Final[str] = "0.1"

ATTENTION_GROUP_KEY_TARGET_NODE_EXACT_RELATION: Final[str] = (
    "target_node_exact_relation"
)

ATTENTION_GROUP_ID_FORMULA: Final[str] = (
    "target_index * num_relations + edge_relation_index"
)

IMPLEMENTED_ATTENTION_NORMALIZATION_MODES: Final[tuple[str, ...]] = (
    ATTENTION_NORMALIZATION_TARGET_NODE_RELATION,
)


# =============================================================================
# Validation and fingerprint helpers
# =============================================================================


def _require_nonnegative_int(
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

    if value < 0:
        raise ValueError(
            f"{name} must be nonnegative."
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


def _normalize_normalization_mode(
    normalization_mode: str,
) -> str:
    if not isinstance(
        normalization_mode,
        str,
    ):
        raise TypeError(
            "attention normalization mode must be a string."
        )

    normalized = normalization_mode.strip()

    if not normalized:
        raise ValueError(
            "attention normalization mode must be a non-empty string."
        )

    if normalized not in (
        CANONICAL_ATTENTION_NORMALIZATION_MODES
    ):
        raise ValueError(
            "Unknown attention normalization mode "
            f"{normalized!r}. Expected one of "
            f"{tuple(CANONICAL_ATTENTION_NORMALIZATION_MODES)!r}."
        )

    if normalized not in (
        IMPLEMENTED_ATTENTION_NORMALIZATION_MODES
    ):
        raise NotImplementedError(
            "Attention normalization mode "
            f"{normalized!r} is canonical but not implemented in the "
            "bounded V2.0 edge-attention package."
        )

    return normalized


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


def _require_score_output(
    source_score_output: EdgeAttentionScoreOutput,
) -> None:
    if not isinstance(
        source_score_output,
        EdgeAttentionScoreOutput,
    ):
        raise TypeError(
            "source_score_output must be an "
            "EdgeAttentionScoreOutput."
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


def _require_finite_float_head_matrix(
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
            f"{name} must have shape [E, A]; "
            f"observed {tuple(value.shape)}."
        )

    if int(value.shape[1]) <= 0:
        raise ValueError(
            f"{name} must contain at least one attention head."
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


def _require_group_ids(
    group_ids: torch.Tensor,
    *,
    edge_count: int,
    num_groups: int,
    device: torch.device,
) -> None:
    if not isinstance(
        group_ids,
        torch.Tensor,
    ):
        raise TypeError(
            "group_ids must be a tensor."
        )

    if group_ids.ndim != 1:
        raise ValueError(
            "group_ids must have shape [E]."
        )

    if tuple(group_ids.shape) != (
        edge_count,
    ):
        raise ValueError(
            "group_ids must align with the edge axis. "
            f"Expected {(edge_count,)}; observed "
            f"{tuple(group_ids.shape)}."
        )

    if group_ids.dtype != torch.long:
        raise ValueError(
            "group_ids must use torch.long."
        )

    if not _devices_match(
        group_ids.device,
        device,
    ):
        raise ValueError(
            "group_ids and attention logits must share one device. "
            f"Observed {group_ids.device} and {device}."
        )

    validate_segment_ids(
        group_ids,
        num_segments=num_groups,
        item_count=edge_count,
        device=device,
    )


def _require_group_counts(
    group_counts: torch.Tensor,
    *,
    num_groups: int,
    edge_count: int,
    device: torch.device,
) -> None:
    if not isinstance(
        group_counts,
        torch.Tensor,
    ):
        raise TypeError(
            "group_counts must be a tensor."
        )

    if group_counts.ndim != 1:
        raise ValueError(
            "group_counts must have shape [num_groups]."
        )

    if tuple(group_counts.shape) != (
        num_groups,
    ):
        raise ValueError(
            "group_counts must align with the complete dense group "
            f"axis. Expected {(num_groups,)}; observed "
            f"{tuple(group_counts.shape)}."
        )

    if group_counts.dtype != torch.long:
        raise ValueError(
            "group_counts must use torch.long."
        )

    if not _devices_match(
        group_counts.device,
        device,
    ):
        raise ValueError(
            "group_counts and attention logits must share one device. "
            f"Observed {group_counts.device} and {device}."
        )

    if bool(
        (group_counts < 0)
        .any()
        .item()
    ):
        raise ValueError(
            "group_counts must be nonnegative."
        )

    observed_total = int(
        group_counts.sum().item()
    )

    if observed_total != edge_count:
        raise ValueError(
            "group_counts must sum to the edge count. "
            f"Observed {observed_total}; expected {edge_count}."
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


# =============================================================================
# Deterministic exact target-relation grouping
# =============================================================================


def build_target_node_relation_group_ids(
    source_inputs: FunctionalMessagePassingInputs,
) -> torch.Tensor:
    """
    Build deterministic dense IDs for target-node/exact-relation groups.

    Returns
    -------
    torch.Tensor
        ``torch.long`` tensor with shape ``[E]`` on the source-input device.

        The implementation uses:

            target_index * num_relations + edge_relation_index

        Dense relation indices come from the exact compiled registry. Stable
        ontology IDs are metadata and never participate in this arithmetic.
    """

    _require_source_inputs(
        source_inputs
    )

    group_ids = (
        source_inputs.target_index
        * source_inputs.num_relations
        + source_inputs.edge_relation_index
    )

    _require_group_ids(
        group_ids,
        edge_count=source_inputs.num_edges,
        num_groups=(
            source_inputs.attention_num_groups
        ),
        device=source_inputs.device,
    )

    expected = (
        source_inputs.attention_group_id
    )

    if not torch.equal(
        group_ids,
        expected,
    ):
        raise RuntimeError(
            "FunctionalMessagePassingInputs.attention_group_id no "
            "longer matches the canonical target-node/exact-relation "
            "grouping formula."
        )

    return group_ids


def build_target_node_relation_group_counts(
    source_inputs: FunctionalMessagePassingInputs,
    *,
    group_ids: torch.Tensor | None = None,
) -> torch.Tensor:
    """
    Count edges on the complete dense ``N * R`` attention-group axis.

    Absent groups remain explicit zeros. No compact remapping is performed,
    preserving deterministic target/relation identity for diagnostics.
    """

    _require_source_inputs(
        source_inputs
    )

    resolved_group_ids = (
        build_target_node_relation_group_ids(
            source_inputs
        )
        if group_ids is None
        else group_ids
    )

    _require_group_ids(
        resolved_group_ids,
        edge_count=source_inputs.num_edges,
        num_groups=(
            source_inputs.attention_num_groups
        ),
        device=source_inputs.device,
    )

    expected_ids = (
        source_inputs.attention_group_id
    )

    if not torch.equal(
        resolved_group_ids,
        expected_ids,
    ):
        raise ValueError(
            "group_ids must encode target node + exact dense relation "
            "index."
        )

    counts = segment_counts(
        resolved_group_ids,
        num_segments=(
            source_inputs.attention_num_groups
        ),
    )

    _require_group_counts(
        counts,
        num_groups=(
            source_inputs.attention_num_groups
        ),
        edge_count=source_inputs.num_edges,
        device=source_inputs.device,
    )

    return counts


# =============================================================================
# Low-level grouped normalization
# =============================================================================


def normalize_attention_logits(
    logits_by_head: torch.Tensor,
    group_ids: torch.Tensor,
    *,
    num_groups: int,
) -> torch.Tensor:
    """
    Apply numerically stable grouped softmax independently per head.

    Parameters
    ----------
    logits_by_head:
        Finite floating-point tensor ``[E, A]``.
    group_ids:
        Dense zero-based segment IDs ``[E]``.
    num_groups:
        Total number of possible groups, including absent groups.

    Returns
    -------
    torch.Tensor
        Nonnegative tensor ``[E, A]`` with the same shape, dtype, device, and
        autograd connectivity as ``logits_by_head``.
    """

    _require_finite_float_head_matrix(
        "logits_by_head",
        logits_by_head,
    )
    resolved_num_groups = (
        _require_nonnegative_int(
            "num_groups",
            num_groups,
        )
    )

    edge_count = int(
        logits_by_head.shape[0]
    )

    _require_group_ids(
        group_ids,
        edge_count=edge_count,
        num_groups=resolved_num_groups,
        device=logits_by_head.device,
    )

    weights = grouped_softmax(
        logits_by_head,
        group_ids,
        num_segments=resolved_num_groups,
    )

    if not isinstance(
        weights,
        torch.Tensor,
    ):
        raise RuntimeError(
            "grouped_softmax must return a tensor."
        )

    if weights.shape != (
        logits_by_head.shape
    ):
        raise RuntimeError(
            "Grouped attention normalization changed shape. "
            f"Observed {tuple(weights.shape)}; expected "
            f"{tuple(logits_by_head.shape)}."
        )

    if weights.dtype != (
        logits_by_head.dtype
    ):
        raise RuntimeError(
            "Grouped attention normalization changed dtype."
        )

    if not _devices_match(
        weights.device,
        logits_by_head.device,
    ):
        raise RuntimeError(
            "Grouped attention normalization changed device."
        )

    if not bool(
        torch.isfinite(weights)
        .all()
        .item()
    ):
        raise FloatingPointError(
            "Grouped attention normalization produced NaN or infinity."
        )

    if bool(
        (weights < 0)
        .any()
        .item()
    ):
        raise FloatingPointError(
            "Grouped attention normalization produced negative weights."
        )

    assert_grouped_softmax_normalized(
        weights,
        group_ids,
        num_segments=resolved_num_groups,
    )

    return weights


# =============================================================================
# Diagnostics
# =============================================================================


def attention_group_sums(
    normalized_weights_by_head: torch.Tensor,
    group_ids: torch.Tensor,
    *,
    num_groups: int,
) -> torch.Tensor:
    """
    Sum candidate attention weights on the dense group axis.

    This helper does not normalize. It is intended for diagnostics, tests, and
    research audits. The result has shape ``[num_groups, A]``.
    """

    _require_finite_float_head_matrix(
        "normalized_weights_by_head",
        normalized_weights_by_head,
    )
    resolved_num_groups = (
        _require_nonnegative_int(
            "num_groups",
            num_groups,
        )
    )

    _require_group_ids(
        group_ids,
        edge_count=int(
            normalized_weights_by_head.shape[0]
        ),
        num_groups=resolved_num_groups,
        device=(
            normalized_weights_by_head.device
        ),
    )

    sums = grouped_softmax_sums(
        normalized_weights_by_head,
        group_ids,
        num_segments=resolved_num_groups,
    )

    expected_shape = (
        resolved_num_groups,
        int(
            normalized_weights_by_head
            .shape[1]
        ),
    )

    if tuple(sums.shape) != expected_shape:
        raise RuntimeError(
            "Attention group-sum diagnostics returned the wrong shape. "
            f"Observed {tuple(sums.shape)}; expected {expected_shape}."
        )

    return sums


def maximum_attention_normalization_error(
    normalized_weights_by_head: torch.Tensor,
    group_ids: torch.Tensor,
    *,
    num_groups: int,
) -> float:
    """
    Return the largest absolute group-sum error over nonempty groups.

    Empty groups are expected to sum to exact zero and do not contribute to the
    returned error. When every group is empty, the error is ``0.0``.

    This diagnostic detaches the scalar result and should not be used as a
    training objective.
    """

    sums = attention_group_sums(
        normalized_weights_by_head,
        group_ids,
        num_groups=num_groups,
    )
    counts = segment_counts(
        group_ids,
        num_segments=num_groups,
    )
    present = counts > 0

    if not bool(
        present.any().item()
    ):
        return 0.0

    error = (
        sums[present]
        - torch.ones_like(
            sums[present]
        )
    ).abs().max()

    return float(
        error.detach().item()
    )


def assert_attention_normalized(
    normalized_weights_by_head: torch.Tensor,
    group_ids: torch.Tensor,
    *,
    num_groups: int,
    atol: float | None = None,
    rtol: float | None = None,
) -> None:
    """
    Public attention-specific wrapper around the generic segment assertion.
    """

    _require_finite_float_head_matrix(
        "normalized_weights_by_head",
        normalized_weights_by_head,
    )
    resolved_num_groups = (
        _require_nonnegative_int(
            "num_groups",
            num_groups,
        )
    )
    _require_group_ids(
        group_ids,
        edge_count=int(
            normalized_weights_by_head.shape[0]
        ),
        num_groups=resolved_num_groups,
        device=(
            normalized_weights_by_head.device
        ),
    )

    assert_grouped_softmax_normalized(
        normalized_weights_by_head,
        group_ids,
        num_segments=resolved_num_groups,
        atol=atol,
        rtol=rtol,
    )


# =============================================================================
# Metadata-preserving functional normalization
# =============================================================================


def normalize_edge_attention_scores(
    source_score_output: EdgeAttentionScoreOutput,
    *,
    normalization_mode: str = (
        ATTENTION_NORMALIZATION_TARGET_NODE_RELATION
    ),
    encoder_architecture_fingerprint: str | None = None,
) -> AttentionNormalizationOutput:
    """
    Normalize one metadata-bearing edge-score output.

    The function is useful for stateless call sites. The class wrapper below is
    preferred when architecture identity should be constructed once and reused.
    """

    _require_score_output(
        source_score_output
    )

    normalized_mode = (
        _normalize_normalization_mode(
            normalization_mode
        )
    )
    source_inputs = (
        source_score_output.source_inputs
    )

    group_ids = (
        build_target_node_relation_group_ids(
            source_inputs
        )
    )
    group_counts = (
        build_target_node_relation_group_counts(
            source_inputs,
            group_ids=group_ids,
        )
    )
    normalized_weights = (
        normalize_attention_logits(
            source_score_output
            .raw_scores_by_head,
            group_ids,
            num_groups=(
                source_inputs
                .attention_num_groups
            ),
        )
    )

    if encoder_architecture_fingerprint is None:
        architecture_fingerprint = (
            _fingerprint(
                {
                    "schema_version": (
                        ATTENTION_NORMALIZATION_SCHEMA_VERSION
                    ),
                    "module": (
                        "functional_normalize_edge_attention_scores"
                    ),
                    "normalization_mode": (
                        normalized_mode
                    ),
                    "group_key": (
                        ATTENTION_GROUP_KEY_TARGET_NODE_EXACT_RELATION
                    ),
                    "group_id_formula": (
                        ATTENTION_GROUP_ID_FORMULA
                    ),
                    "softmax_implementation": (
                        "segment_ops.grouped_softmax"
                    ),
                    "segment_ops_schema_version": (
                        SEGMENT_OPS_SCHEMA_VERSION
                    ),
                    "head_policy": (
                        "normalize_each_head_independently"
                    ),
                    "parameter_count": 0,
                }
            )
        )
    else:
        _require_nonempty_string(
            "encoder_architecture_fingerprint",
            encoder_architecture_fingerprint,
        )
        architecture_fingerprint = (
            encoder_architecture_fingerprint
        )

    return AttentionNormalizationOutput(
        normalized_weights_by_head=(
            normalized_weights
        ),
        group_ids=group_ids,
        group_counts=group_counts,
        source_score_output=(
            source_score_output
        ),
        normalization_mode=(
            normalized_mode
        ),
        encoder_architecture_fingerprint=(
            architecture_fingerprint
        ),
        parameter_fingerprint=None,
    )


# =============================================================================
# Parameter-free normalization module
# =============================================================================


class TargetNodeRelationAttentionNormalization(
    nn.Module
):
    """
    Parameter-free exact target-node/relation grouped softmax.

    Parameters
    ----------
    normalization_mode:
        Canonical normalization name. Bounded V2.0 implements only
        ``target_node_and_relation``.
    """

    normalization_mode: str

    def __init__(
        self,
        *,
        normalization_mode: str = (
            ATTENTION_NORMALIZATION_TARGET_NODE_RELATION
        ),
    ) -> None:
        super().__init__()

        self.normalization_mode = (
            _normalize_normalization_mode(
                normalization_mode
            )
        )

        if self.normalization_mode != (
            ATTENTION_NORMALIZATION_TARGET_NODE_RELATION
        ):
            raise RuntimeError(
                "Internal attention-normalization dispatch is incomplete "
                f"for mode {self.normalization_mode!r}."
            )

    # ------------------------------------------------------------------
    # Construction from configuration
    # ------------------------------------------------------------------

    @classmethod
    def from_config(
        cls,
        *,
        config: FunctionalMessagePassingConfig,
    ) -> "TargetNodeRelationAttentionNormalization":
        """
        Build the normalization stage from FMP configuration.

        This method deliberately does not call ``config.assert_implemented()``.
        The full capability manifest should be promoted only after every
        attention stage and its tests are complete.
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

        return cls(
            normalization_mode=(
                config.attention_normalization
            )
        )

    # ------------------------------------------------------------------
    # Public identity
    # ------------------------------------------------------------------

    @property
    def group_key(self) -> str:
        return (
            ATTENTION_GROUP_KEY_TARGET_NODE_EXACT_RELATION
        )

    @property
    def group_id_formula(self) -> str:
        return ATTENTION_GROUP_ID_FORMULA

    @property
    def parameter_count(self) -> int:
        return 0

    @property
    def trainable_parameter_count(
        self,
    ) -> int:
        return 0

    def architecture_dict(
        self,
    ) -> dict[str, Any]:
        return {
            "schema_version": (
                ATTENTION_NORMALIZATION_SCHEMA_VERSION
            ),
            "normalization_mode": (
                self.normalization_mode
            ),
            "group_key": self.group_key,
            "group_id_formula": (
                self.group_id_formula
            ),
            "group_axis": (
                "dense_num_nodes_times_num_exact_relations"
            ),
            "relation_identity": (
                "exact_compiled_relation_index"
            ),
            "stable_relation_ids_used_as_indices": False,
            "softmax_implementation": (
                "segment_ops.grouped_softmax"
            ),
            "segment_ops_schema_version": (
                SEGMENT_OPS_SCHEMA_VERSION
            ),
            "numerical_stabilization": (
                "subtract_group_maximum_before_exponentiation"
            ),
            "heads_normalized_independently": True,
            "nonempty_group_sum": 1.0,
            "absent_group_count": 0,
            "absent_group_synthetic_edges": False,
            "singleton_group_weight": 1.0,
            "empty_edge_sets_supported": True,
            "control_relations_use_same_math": True,
            "attention_disabled_handled_here": False,
            "uniform_attention_policy": (
                "zero_logits_then_grouped_softmax"
            ),
            "edge_masking_owned_here": False,
            "relation_gate_owned_here": False,
            "head_reduction_owned_here": False,
            "aggregation_owned_here": False,
            "parameter_count": 0,
            "output_schema": (
                "AttentionNormalizationOutput"
            ),
            "operation_order": [
                "validate_score_output",
                "construct_target_node_exact_relation_group_ids",
                "count_edges_on_dense_group_axis",
                "apply_stable_grouped_softmax_per_head",
                "assert_group_normalization",
                "construct_attention_normalization_output",
            ],
        }

    def architecture_fingerprint(
        self,
    ) -> str:
        return _fingerprint(
            self.architecture_dict()
        )

    def parameter_fingerprint(
        self,
    ) -> None:
        return None

    def assert_parameter_free(
        self,
    ) -> None:
        parameters = tuple(
            self.parameters()
        )
        buffers = tuple(
            self.buffers()
        )

        if parameters:
            raise RuntimeError(
                "Attention normalization must remain parameter-free."
            )

        if buffers:
            raise RuntimeError(
                "Bounded attention normalization must not retain "
                "data-dependent buffers."
            )

        if self.state_dict():
            raise RuntimeError(
                "Bounded attention normalization must have an empty "
                "state_dict."
            )

    # ------------------------------------------------------------------
    # Group construction and diagnostics
    # ------------------------------------------------------------------

    def group_ids(
        self,
        source_inputs: FunctionalMessagePassingInputs,
    ) -> torch.Tensor:
        return (
            build_target_node_relation_group_ids(
                source_inputs
            )
        )

    def group_counts(
        self,
        source_inputs: FunctionalMessagePassingInputs,
        *,
        group_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return (
            build_target_node_relation_group_counts(
                source_inputs,
                group_ids=group_ids,
            )
        )

    def normalize_logits(
        self,
        logits_by_head: torch.Tensor,
        group_ids: torch.Tensor,
        *,
        num_groups: int,
    ) -> torch.Tensor:
        return normalize_attention_logits(
            logits_by_head,
            group_ids,
            num_groups=num_groups,
        )

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self,
        source_score_output: EdgeAttentionScoreOutput,
    ) -> AttentionNormalizationOutput:
        self.assert_parameter_free()
        _require_score_output(
            source_score_output
        )

        source_inputs = (
            source_score_output
            .source_inputs
        )
        group_ids = self.group_ids(
            source_inputs
        )
        group_counts = self.group_counts(
            source_inputs,
            group_ids=group_ids,
        )
        normalized_weights = (
            self.normalize_logits(
                source_score_output
                .raw_scores_by_head,
                group_ids,
                num_groups=(
                    source_inputs
                    .attention_num_groups
                ),
            )
        )

        return AttentionNormalizationOutput(
            normalized_weights_by_head=(
                normalized_weights
            ),
            group_ids=group_ids,
            group_counts=group_counts,
            source_score_output=(
                source_score_output
            ),
            normalization_mode=(
                self.normalization_mode
            ),
            encoder_architecture_fingerprint=(
                self.architecture_fingerprint()
            ),
            parameter_fingerprint=None,
        )

    def extra_repr(self) -> str:
        return (
            f"normalization_mode={self.normalization_mode!r}, "
            f"group_key={self.group_key!r}, "
            "heads_normalized_independently=True, "
            "parameter_free=True"
        )


# =============================================================================
# Public construction dispatcher
# =============================================================================


AttentionNormalizer: TypeAlias = (
    TargetNodeRelationAttentionNormalization
)


def build_attention_normalizer(
    *,
    config: FunctionalMessagePassingConfig,
) -> AttentionNormalizer:
    """
    Construct the bounded attention-normalization implementation.
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
    normalized_mode = (
        _normalize_normalization_mode(
            config.attention_normalization
        )
    )

    if normalized_mode == (
        ATTENTION_NORMALIZATION_TARGET_NODE_RELATION
    ):
        return (
            TargetNodeRelationAttentionNormalization(
                normalization_mode=(
                    normalized_mode
                )
            )
        )

    raise RuntimeError(
        "Internal attention-normalization dispatch is incomplete for "
        f"mode {normalized_mode!r}."
    )


# Compact aliases for package exports and call sites.
AttentionNormalization = (
    TargetNodeRelationAttentionNormalization
)
EdgeAttentionNormalizer = (
    TargetNodeRelationAttentionNormalization
)
apply_attention_normalization = (
    normalize_edge_attention_scores
)
build_edge_attention_normalizer = (
    build_attention_normalizer
)


__all__ = (
    "ATTENTION_GROUP_ID_FORMULA",
    "ATTENTION_GROUP_KEY_TARGET_NODE_EXACT_RELATION",
    "ATTENTION_NORMALIZATION_SCHEMA_VERSION",
    "IMPLEMENTED_ATTENTION_NORMALIZATION_MODES",
    "AttentionNormalization",
    "AttentionNormalizer",
    "EdgeAttentionNormalizer",
    "TargetNodeRelationAttentionNormalization",
    "apply_attention_normalization",
    "assert_attention_normalized",
    "attention_group_sums",
    "build_attention_normalizer",
    "build_edge_attention_normalizer",
    "build_target_node_relation_group_counts",
    "build_target_node_relation_group_ids",
    "maximum_attention_normalization_error",
    "normalize_attention_logits",
    "normalize_edge_attention_scores",
)
