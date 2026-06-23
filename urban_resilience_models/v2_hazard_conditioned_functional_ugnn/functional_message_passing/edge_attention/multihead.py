"""
Attention-head reduction for exact-relation edge attention.

Target path:
    urban_resilience_models/
        v2_hazard_conditioned_functional_ugnn/
            functional_message_passing/
                edge_attention/
                    multihead.py

This module owns the final operation that converts independently normalized
head-level edge-attention weights ``[E, A]`` into one edge-aligned routing
coefficient ``[E]``.

Bounded V2.0 policy
-------------------
The only implemented head-reduction policy is the arithmetic mean:

    alpha_bar[e] =
        (1 / A) * sum_{a=1}^{A} alpha[e, a]

where each head has already been normalized independently inside the exact
target-node/relation group:

    sum_{e in E_{t,r}} alpha[e, a] = 1

Therefore:

    sum_{e in E_{t,r}} alpha_bar[e]
        = (1 / A) * sum_a 1
        = 1

The arithmetic mean preserves group normalization exactly in real arithmetic
and to numerical tolerance in floating-point arithmetic.

Interpretation
-------------------------
Multiple attention heads provide multiple learned routing distributions over
the same target-node/exact-relation edge group. Mean reduction forms an
equal-weight ensemble of those distributions.

This implementation does *not* claim that:

- different heads specialize;
- head disagreement is epistemic uncertainty;
- one head is more scientifically valid than another;
- mean reduction is universally optimal;
- attention is causal importance.

Those claims require explicit empirical evidence. The diagnostic helpers in
this module quantify head agreement, but they do not convert agreement or
disagreement into causal, confidence, or uncertainty semantics.

Why mean is the bounded default
-------------------------------
Mean reduction is deliberately conservative:

- it introduces no learned head-selection parameters;
- it preserves nonnegativity;
- it preserves exact-relation group normalization;
- it is invariant to permutations of the head axis;
- it reduces a single head to the identity;
- it does not privilege an arbitrary head;
- it keeps the final edge coefficient directly auditable.

Canonical but unimplemented alternatives such as weighted mean, maximum, and
no reduction are rejected explicitly. They have different
numerical consequences:

``weighted_mean``
    requires a defined source of head weights, normalization of those weights,
    and a decision about whether weights are global, relation-specific,
    hazard-conditioned, or edge-specific;

``max``
    generally destroys target-relation group normalization and changes the
    meaning of the coefficient;

``none``
    leaves a head axis that downstream message construction must represent
    explicitly rather than silently broadcast or collapse.

Scope boundaries
----------------
This module does not own:

- edge-score prediction;
- target-node/relation grouped softmax;
- relation gates;
- relation transforms;
- edge attributes;
- semantic edge weights;
- structural edge normalization;
- message construction;
- target-node aggregation;
- edge masking;
- explanation faithfulness or causal interpretation.

Input contract
--------------
The metadata-bearing input is ``AttentionNormalizationOutput``. It already
guarantees:

- normalized weights have shape ``[E, A]``;
- weights are finite and nonnegative;
- group IDs encode target node + exact dense relation;
- every nonempty group sums to one independently per head;
- singleton groups have exact weight one;
- group counts cover the complete dense ``N * R`` group axis.

This module validates the reduction result again at the stage boundary and
constructs ``AttentionHeadReductionOutput``.

Boundary behavior
-----------------
- ``A = 1`` returns the sole head exactly.
- ``A >= 2`` returns the arithmetic mean.
- ``E = 0`` returns a finite empty tensor ``[0]``.
- absent groups remain absent;
- singleton groups remain exact one;
- edge order is preserved;
- head-order permutations do not change the result;
- dtype, device, and autograd connectivity are preserved;
- no hidden cast or device transfer occurs;
- the module is parameter-free and buffer-free.

Diagnostics
-----------
Head disagreement can reveal whether a multihead model is using genuinely
different routing patterns, but low or high disagreement is not automatically
good. This module exposes:

- per-edge head variance;
- per-edge head standard deviation;
- per-edge head range;
- per-edge mean absolute deviation from the head mean;
- maximum and mean aggregate disagreement summaries.

These diagnostics detach only when returning Python scalars. Tensor-valued
diagnostics preserve autograd and can be used in controlled research analyses.
They are not included in every ordinary forward output, avoiding unnecessary
storage at province-scale graph sizes.

Capability manifests
--------------------
``from_config`` validates canonical configuration but does not call
``config.assert_implemented()``. The complete attention capability should be
promoted only after score functions, normalization, head reduction,
orchestration, and their focused tests agree.
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
    ATTENTION_HEAD_REDUCTION_MAX,
    ATTENTION_HEAD_REDUCTION_MEAN,
    ATTENTION_HEAD_REDUCTION_NONE,
    ATTENTION_HEAD_REDUCTION_WEIGHTED_MEAN,
    CANONICAL_ATTENTION_HEAD_REDUCTIONS,
)
from ..segment_ops import (
    SEGMENT_OPS_SCHEMA_VERSION,
    assert_grouped_softmax_normalized,
)
from .schemas import (
    AttentionHeadReductionOutput,
    AttentionNormalizationOutput,
)


# =============================================================================
# Public identity
# =============================================================================


ATTENTION_MULTIHEAD_SCHEMA_VERSION: Final[str] = "0.1"

IMPLEMENTED_ATTENTION_HEAD_REDUCTIONS: Final[tuple[str, ...]] = (
    ATTENTION_HEAD_REDUCTION_MEAN,
)

ATTENTION_HEAD_MEAN_FORMULA: Final[str] = (
    "edge_weights = normalized_weights_by_head.mean(dim=1)"
)

ATTENTION_HEAD_REDUCTION_INTERPRETATION: Final[str] = (
    "equal_weight_ensemble_of_independently_normalized_attention_heads"
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


def _normalize_head_reduction(
    head_reduction: str,
) -> str:
    if not isinstance(
        head_reduction,
        str,
    ):
        raise TypeError(
            "attention head reduction must be a string."
        )

    normalized = head_reduction.strip()

    if not normalized:
        raise ValueError(
            "attention head reduction must be a non-empty string."
        )

    if normalized not in (
        CANONICAL_ATTENTION_HEAD_REDUCTIONS
    ):
        raise ValueError(
            "Unknown attention head reduction "
            f"{normalized!r}. Expected one of "
            f"{tuple(CANONICAL_ATTENTION_HEAD_REDUCTIONS)!r}."
        )

    if normalized not in (
        IMPLEMENTED_ATTENTION_HEAD_REDUCTIONS
    ):
        if normalized == (
            ATTENTION_HEAD_REDUCTION_WEIGHTED_MEAN
        ):
            reason = (
                "weighted_mean requires an explicit, validated source "
                "and normalization contract for head weights"
            )
        elif normalized == (
            ATTENTION_HEAD_REDUCTION_MAX
        ):
            reason = (
                "max does not generally preserve target-relation group "
                "normalization"
            )
        elif normalized == (
            ATTENTION_HEAD_REDUCTION_NONE
        ):
            reason = (
                "none requires downstream message tensors to retain an "
                "explicit head axis"
            )
        else:
            reason = (
                "the policy has no bounded V2.0 implementation"
            )

        raise NotImplementedError(
            "Attention head reduction "
            f"{normalized!r} is canonical but not implemented: "
            f"{reason}."
        )

    return normalized


def _require_normalization_output(
    source_normalization_output: (
        AttentionNormalizationOutput
    ),
) -> None:
    if not isinstance(
        source_normalization_output,
        AttentionNormalizationOutput,
    ):
        raise TypeError(
            "source_normalization_output must be an "
            "AttentionNormalizationOutput."
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


def _require_head_matrix(
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

    if bool(
        (value < 0)
        .any()
        .item()
    ):
        raise ValueError(
            f"{name} must be nonnegative."
        )


def _require_edge_vector(
    name: str,
    value: torch.Tensor,
    *,
    edge_count: int,
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

    if value.ndim != 1:
        raise ValueError(
            f"{name} must have shape [E]; "
            f"observed {tuple(value.shape)}."
        )

    if tuple(value.shape) != (
        edge_count,
    ):
        raise ValueError(
            f"{name} must have shape {(edge_count,)}; "
            f"observed {tuple(value.shape)}."
        )

    if not value.dtype.is_floating_point:
        raise ValueError(
            f"{name} must use a floating-point dtype."
        )

    if value.dtype != dtype:
        raise ValueError(
            f"{name} must use dtype {dtype}; "
            f"observed {value.dtype}."
        )

    if not _devices_match(
        value.device,
        device,
    ):
        raise ValueError(
            f"{name} must be on device {device}; "
            f"observed {value.device}."
        )

    if not bool(
        torch.isfinite(value)
        .all()
        .item()
    ):
        raise ValueError(
            f"{name} must contain only finite values."
        )

    if bool(
        (value < 0)
        .any()
        .item()
    ):
        raise ValueError(
            f"{name} must be nonnegative."
        )


def _require_same_shape_dtype_device(
    first_name: str,
    first: torch.Tensor,
    second_name: str,
    second: torch.Tensor,
) -> None:
    if tuple(first.shape) != tuple(
        second.shape
    ):
        raise ValueError(
            f"{first_name} and {second_name} must have the same shape. "
            f"Observed {tuple(first.shape)} and {tuple(second.shape)}."
        )

    if first.dtype != second.dtype:
        raise ValueError(
            f"{first_name} and {second_name} must have the same dtype. "
            f"Observed {first.dtype} and {second.dtype}."
        )

    if not _devices_match(
        first.device,
        second.device,
    ):
        raise ValueError(
            f"{first_name} and {second_name} must share one device. "
            f"Observed {first.device} and {second.device}."
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


# =============================================================================
# Low-level arithmetic-mean reduction
# =============================================================================


def mean_reduce_attention_heads(
    normalized_weights_by_head: torch.Tensor,
) -> torch.Tensor:
    """
    Reduce head-level attention weights by arithmetic mean.

    Parameters
    ----------
    normalized_weights_by_head:
        Finite nonnegative floating tensor ``[E, A]``.

    Returns
    -------
    torch.Tensor
        Edge-aligned tensor ``[E]`` with the same dtype and device.

    Notes
    -----
    This low-level helper validates tensor-local invariants but cannot prove
    target-relation group normalization without group metadata. The
    metadata-bearing wrapper and module validate that property.
    """

    _require_head_matrix(
        "normalized_weights_by_head",
        normalized_weights_by_head,
    )

    num_heads = int(
        normalized_weights_by_head.shape[1]
    )

    if num_heads == 1:
        # ``select`` preserves an exact identity for the single-head case and
        # avoids a floating reduction that is mathematically unnecessary.
        edge_weights = (
            normalized_weights_by_head[
                :,
                0,
            ]
        )
    else:
        edge_weights = (
            normalized_weights_by_head
            .mean(dim=1)
        )

    _require_edge_vector(
        "edge_weights",
        edge_weights,
        edge_count=int(
            normalized_weights_by_head.shape[0]
        ),
        dtype=(
            normalized_weights_by_head.dtype
        ),
        device=(
            normalized_weights_by_head.device
        ),
    )

    return edge_weights


def reduce_attention_heads(
    normalized_weights_by_head: torch.Tensor,
    *,
    head_reduction: str = (
        ATTENTION_HEAD_REDUCTION_MEAN
    ),
) -> torch.Tensor:
    """
    Dispatch a low-level attention-head reduction.

    Bounded V2.0 implements only arithmetic mean.
    """

    normalized_reduction = (
        _normalize_head_reduction(
            head_reduction
        )
    )

    if normalized_reduction == (
        ATTENTION_HEAD_REDUCTION_MEAN
    ):
        return mean_reduce_attention_heads(
            normalized_weights_by_head
        )

    raise RuntimeError(
        "Internal attention-head reduction dispatch is incomplete for "
        f"{normalized_reduction!r}."
    )


# =============================================================================
# Group-normalization preservation
# =============================================================================


def assert_reduced_attention_normalized(
    edge_weights: torch.Tensor,
    source_normalization_output: (
        AttentionNormalizationOutput
    ),
    *,
    atol: float | None = None,
    rtol: float | None = None,
) -> None:
    """
    Assert that reduced edge weights remain normalized on the source groups.
    """

    _require_normalization_output(
        source_normalization_output
    )
    _require_edge_vector(
        "edge_weights",
        edge_weights,
        edge_count=(
            source_normalization_output
            .source_inputs
            .num_edges
        ),
        dtype=(
            source_normalization_output
            .dtype
        ),
        device=(
            source_normalization_output
            .device
        ),
    )

    assert_grouped_softmax_normalized(
        edge_weights,
        source_normalization_output.group_ids,
        num_segments=(
            source_normalization_output
            .num_groups
        ),
        atol=atol,
        rtol=rtol,
    )


def maximum_reduced_attention_normalization_error(
    edge_weights: torch.Tensor,
    source_normalization_output: (
        AttentionNormalizationOutput
    ),
) -> float:
    """
    Return the maximum absolute group-sum error on nonempty groups.

    The returned Python scalar is detached and is intended for audits and
    diagnostics, not as a training objective.
    """

    _require_normalization_output(
        source_normalization_output
    )
    _require_edge_vector(
        "edge_weights",
        edge_weights,
        edge_count=(
            source_normalization_output
            .source_inputs
            .num_edges
        ),
        dtype=(
            source_normalization_output
            .dtype
        ),
        device=(
            source_normalization_output
            .device
        ),
    )

    num_groups = (
        source_normalization_output
        .num_groups
    )
    sums = torch.zeros(
        num_groups,
        dtype=edge_weights.dtype,
        device=edge_weights.device,
    )

    if edge_weights.numel() > 0:
        sums.index_add_(
            0,
            source_normalization_output
            .group_ids,
            edge_weights,
        )

    present = (
        source_normalization_output
        .group_counts
        > 0
    )

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


# =============================================================================
# Head-disagreement diagnostics
# =============================================================================


def attention_head_mean(
    normalized_weights_by_head: torch.Tensor,
) -> torch.Tensor:
    """
    Return the arithmetic head mean ``[E]``.

    This is identical to the bounded reduction policy.
    """

    return mean_reduce_attention_heads(
        normalized_weights_by_head
    )


def attention_head_variance(
    normalized_weights_by_head: torch.Tensor,
) -> torch.Tensor:
    """
    Return population variance across heads for every edge.

    The population definition (``unbiased=False``) returns exact zero for a
    single head and avoids undefined sample variance at ``A = 1``.
    """

    _require_head_matrix(
        "normalized_weights_by_head",
        normalized_weights_by_head,
    )

    if int(
        normalized_weights_by_head.shape[0]
    ) == 0:
        # Return an empty edge-aligned view without invoking torch.var.
        #
        # Selecting an existing head preserves:
        # - shape [E];
        # - dtype;
        # - device;
        # - autograd connectivity.
        #
        # _require_head_matrix guarantees that A >= 1.
        return normalized_weights_by_head[
            :,
            0,
        ]

    return normalized_weights_by_head.var(
        dim=1,
        unbiased=False,
    )


def attention_head_standard_deviation(
    normalized_weights_by_head: torch.Tensor,
) -> torch.Tensor:
    """
    Return population standard deviation across heads for every edge.
    """

    variance = attention_head_variance(
        normalized_weights_by_head
    )
    return torch.sqrt(
        variance
    )


def attention_head_range(
    normalized_weights_by_head: torch.Tensor,
) -> torch.Tensor:
    """
    Return ``max_head - min_head`` for every edge.
    """

    _require_head_matrix(
        "normalized_weights_by_head",
        normalized_weights_by_head,
    )

    maximum = (
        normalized_weights_by_head
        .max(dim=1)
        .values
    )
    minimum = (
        normalized_weights_by_head
        .min(dim=1)
        .values
    )

    return maximum - minimum


def attention_head_mean_absolute_deviation(
    normalized_weights_by_head: torch.Tensor,
) -> torch.Tensor:
    """
    Return per-edge mean absolute deviation from the head mean.
    """

    _require_head_matrix(
        "normalized_weights_by_head",
        normalized_weights_by_head,
    )

    mean = (
        normalized_weights_by_head
        .mean(
            dim=1,
            keepdim=True,
        )
    )

    return (
        normalized_weights_by_head
        - mean
    ).abs().mean(dim=1)


def maximum_attention_head_range(
    normalized_weights_by_head: torch.Tensor,
) -> float:
    """
    Return the largest per-edge head range as a detached Python scalar.
    """

    ranges = attention_head_range(
        normalized_weights_by_head
    )

    if ranges.numel() == 0:
        return 0.0

    return float(
        ranges.max().detach().item()
    )


def mean_attention_head_standard_deviation(
    normalized_weights_by_head: torch.Tensor,
) -> float:
    """
    Return mean per-edge head standard deviation as a detached scalar.
    """

    standard_deviation = (
        attention_head_standard_deviation(
            normalized_weights_by_head
        )
    )

    if standard_deviation.numel() == 0:
        return 0.0

    return float(
        standard_deviation
        .mean()
        .detach()
        .item()
    )


def head_disagreement_summary(
    normalized_weights_by_head: torch.Tensor,
) -> dict[str, float | int]:
    """
    Return compact descriptive head-disagreement statistics.

    These quantities are descriptive only. They are not calibrated
    uncertainty, causal evidence, or proof of head specialization.
    """

    _require_head_matrix(
        "normalized_weights_by_head",
        normalized_weights_by_head,
    )

    edge_count = int(
        normalized_weights_by_head.shape[0]
    )
    num_heads = int(
        normalized_weights_by_head.shape[1]
    )

    if edge_count == 0:
        return {
            "edge_count": 0,
            "num_heads": num_heads,
            "maximum_head_range": 0.0,
            "mean_head_range": 0.0,
            "mean_head_standard_deviation": 0.0,
            "mean_head_absolute_deviation": 0.0,
        }

    ranges = attention_head_range(
        normalized_weights_by_head
    )
    standard_deviation = (
        attention_head_standard_deviation(
            normalized_weights_by_head
        )
    )
    mean_absolute_deviation = (
        attention_head_mean_absolute_deviation(
            normalized_weights_by_head
        )
    )

    return {
        "edge_count": edge_count,
        "num_heads": num_heads,
        "maximum_head_range": float(
            ranges.max().detach().item()
        ),
        "mean_head_range": float(
            ranges.mean().detach().item()
        ),
        "mean_head_standard_deviation": float(
            standard_deviation
            .mean()
            .detach()
            .item()
        ),
        "mean_head_absolute_deviation": float(
            mean_absolute_deviation
            .mean()
            .detach()
            .item()
        ),
    }


# =============================================================================
# Metadata-preserving functional reduction
# =============================================================================


def reduce_normalized_attention_heads(
    source_normalization_output: (
        AttentionNormalizationOutput
    ),
    *,
    head_reduction: str = (
        ATTENTION_HEAD_REDUCTION_MEAN
    ),
    encoder_architecture_fingerprint: str | None = None,
) -> AttentionHeadReductionOutput:
    """
    Reduce one metadata-bearing normalization output.

    This stateless wrapper infers the head count from the source output. The
    class below is preferred when an expected head count must be frozen in the
    model architecture.
    """

    _require_normalization_output(
        source_normalization_output
    )
    normalized_reduction = (
        _normalize_head_reduction(
            head_reduction
        )
    )

    edge_weights = reduce_attention_heads(
        source_normalization_output
        .normalized_weights_by_head,
        head_reduction=normalized_reduction,
    )

    assert_reduced_attention_normalized(
        edge_weights,
        source_normalization_output,
    )

    if encoder_architecture_fingerprint is None:
        architecture_fingerprint = _fingerprint(
            {
                "schema_version": (
                    ATTENTION_MULTIHEAD_SCHEMA_VERSION
                ),
                "module": (
                    "functional_reduce_normalized_attention_heads"
                ),
                "head_reduction": (
                    normalized_reduction
                ),
                "reduction_formula": (
                    ATTENTION_HEAD_MEAN_FORMULA
                ),
                "interpretation": (
                    ATTENTION_HEAD_REDUCTION_INTERPRETATION
                ),
                "num_heads": (
                    source_normalization_output
                    .num_heads
                ),
                "single_head_identity": True,
                "head_permutation_invariant": True,
                "preserves_group_normalization": True,
                "parameter_count": 0,
                "segment_ops_schema_version": (
                    SEGMENT_OPS_SCHEMA_VERSION
                ),
            }
        )
    else:
        _require_nonempty_string(
            "encoder_architecture_fingerprint",
            encoder_architecture_fingerprint,
        )
        architecture_fingerprint = (
            encoder_architecture_fingerprint
        )

    return AttentionHeadReductionOutput(
        edge_weights=edge_weights,
        source_normalization_output=(
            source_normalization_output
        ),
        head_reduction=(
            normalized_reduction
        ),
        encoder_architecture_fingerprint=(
            architecture_fingerprint
        ),
        parameter_fingerprint=None,
    )


# =============================================================================
# Parameter-free mean-reduction module
# =============================================================================


class MeanAttentionHeadReduction(nn.Module):
    """
    Parameter-free arithmetic-mean reduction across attention heads.

    Parameters
    ----------
    num_heads:
        Exact number of heads expected at runtime. Recording this in the module
        architecture prevents silently applying a checkpoint configured for a
        different head count.
    head_reduction:
        Canonical reduction policy. Bounded V2.0 accepts only ``mean``.
    """

    num_heads: int
    head_reduction: str

    def __init__(
        self,
        *,
        num_heads: int,
        head_reduction: str = (
            ATTENTION_HEAD_REDUCTION_MEAN
        ),
    ) -> None:
        super().__init__()

        self.num_heads = _require_positive_int(
            "num_heads",
            num_heads,
        )
        self.head_reduction = (
            _normalize_head_reduction(
                head_reduction
            )
        )

        if self.head_reduction != (
            ATTENTION_HEAD_REDUCTION_MEAN
        ):
            raise RuntimeError(
                "Internal attention-head reduction dispatch is incomplete "
                f"for {self.head_reduction!r}."
            )

    # ------------------------------------------------------------------
    # Construction from configuration
    # ------------------------------------------------------------------

    @classmethod
    def from_config(
        cls,
        *,
        config: FunctionalMessagePassingConfig,
    ) -> "MeanAttentionHeadReduction":
        """
        Build the reduction stage from FMP configuration.
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
            num_heads=(
                config.attention_heads
            ),
            head_reduction=(
                config.attention_head_reduction
            ),
        )

    # ------------------------------------------------------------------
    # Public identity
    # ------------------------------------------------------------------

    @property
    def parameter_count(self) -> int:
        return 0

    @property
    def trainable_parameter_count(
        self,
    ) -> int:
        return 0

    @property
    def single_head_identity(self) -> bool:
        return self.num_heads == 1

    def architecture_dict(
        self,
    ) -> dict[str, Any]:
        return {
            "schema_version": (
                ATTENTION_MULTIHEAD_SCHEMA_VERSION
            ),
            "head_reduction": (
                self.head_reduction
            ),
            "num_heads": self.num_heads,
            "input_shape": "[E, A]",
            "output_shape": "[E]",
            "reduction_axis": 1,
            "reduction_formula": (
                ATTENTION_HEAD_MEAN_FORMULA
            ),
            "interpretation": (
                ATTENTION_HEAD_REDUCTION_INTERPRETATION
            ),
            "head_mixture_weights": (
                "equal_1_over_num_heads"
            ),
            "learned_head_weights": False,
            "hazard_conditioned_head_weights": False,
            "relation_conditioned_head_weights": False,
            "edge_conditioned_head_weights": False,
            "single_head_identity": (
                self.single_head_identity
            ),
            "head_permutation_invariant": True,
            "preserves_nonnegativity": True,
            "preserves_target_relation_group_normalization": True,
            "preserves_edge_order": True,
            "preserves_dtype": True,
            "preserves_device": True,
            "preserves_autograd": True,
            "empty_edge_sets_supported": True,
            "claims_head_specialization": False,
            "claims_uncertainty_calibration": False,
            "claims_causal_importance": False,
            "relation_gate_owned_here": False,
            "normalization_owned_here": False,
            "message_construction_owned_here": False,
            "aggregation_owned_here": False,
            "parameter_count": 0,
            "segment_ops_schema_version": (
                SEGMENT_OPS_SCHEMA_VERSION
            ),
            "output_schema": (
                "AttentionHeadReductionOutput"
            ),
            "operation_order": [
                "validate_normalization_output",
                "validate_runtime_head_count",
                "mean_reduce_head_axis",
                "assert_target_relation_group_normalization",
                "construct_attention_head_reduction_output",
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
                "Attention-head reduction must remain parameter-free."
            )

        if buffers:
            raise RuntimeError(
                "Bounded attention-head reduction must not retain "
                "data-dependent buffers."
            )

        if self.state_dict():
            raise RuntimeError(
                "Bounded attention-head reduction must have an empty "
                "state_dict."
            )

    # ------------------------------------------------------------------
    # Runtime validation and diagnostics
    # ------------------------------------------------------------------

    def _validate_runtime_head_count(
        self,
        source_normalization_output: (
            AttentionNormalizationOutput
        ),
    ) -> None:
        observed_heads = (
            source_normalization_output
            .num_heads
        )

        if observed_heads != self.num_heads:
            raise ValueError(
                "Runtime attention head count differs from the "
                "head-reduction architecture. "
                f"Observed {observed_heads}; expected {self.num_heads}."
            )

    def reduce_tensor(
        self,
        normalized_weights_by_head: torch.Tensor,
    ) -> torch.Tensor:
        _require_head_matrix(
            "normalized_weights_by_head",
            normalized_weights_by_head,
        )

        observed_heads = int(
            normalized_weights_by_head.shape[1]
        )

        if observed_heads != self.num_heads:
            raise ValueError(
                "normalized_weights_by_head has the wrong head count. "
                f"Observed {observed_heads}; expected {self.num_heads}."
            )

        return reduce_attention_heads(
            normalized_weights_by_head,
            head_reduction=(
                self.head_reduction
            ),
        )

    def disagreement_summary(
        self,
        source_normalization_output: (
            AttentionNormalizationOutput
        ),
    ) -> dict[str, float | int]:
        _require_normalization_output(
            source_normalization_output
        )
        self._validate_runtime_head_count(
            source_normalization_output
        )

        return head_disagreement_summary(
            source_normalization_output
            .normalized_weights_by_head
        )

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self,
        source_normalization_output: (
            AttentionNormalizationOutput
        ),
    ) -> AttentionHeadReductionOutput:
        self.assert_parameter_free()
        _require_normalization_output(
            source_normalization_output
        )
        self._validate_runtime_head_count(
            source_normalization_output
        )

        edge_weights = self.reduce_tensor(
            source_normalization_output
            .normalized_weights_by_head
        )

        assert_reduced_attention_normalized(
            edge_weights,
            source_normalization_output,
        )

        return AttentionHeadReductionOutput(
            edge_weights=edge_weights,
            source_normalization_output=(
                source_normalization_output
            ),
            head_reduction=(
                self.head_reduction
            ),
            encoder_architecture_fingerprint=(
                self.architecture_fingerprint()
            ),
            parameter_fingerprint=None,
        )

    def extra_repr(self) -> str:
        return (
            f"num_heads={self.num_heads}, "
            f"head_reduction={self.head_reduction!r}, "
            f"single_head_identity={self.single_head_identity}, "
            "head_permutation_invariant=True, "
            "parameter_free=True"
        )


# =============================================================================
# Public construction dispatcher
# =============================================================================


AttentionHeadReducer: TypeAlias = (
    MeanAttentionHeadReduction
)


def build_attention_head_reducer(
    *,
    config: FunctionalMessagePassingConfig,
) -> AttentionHeadReducer:
    """
    Construct the bounded attention-head reduction implementation.
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
    normalized_reduction = (
        _normalize_head_reduction(
            config.attention_head_reduction
        )
    )

    if normalized_reduction == (
        ATTENTION_HEAD_REDUCTION_MEAN
    ):
        return MeanAttentionHeadReduction(
            num_heads=(
                config.attention_heads
            ),
            head_reduction=(
                normalized_reduction
            ),
        )

    raise RuntimeError(
        "Internal attention-head reduction dispatch is incomplete for "
        f"{normalized_reduction!r}."
    )


# Compact aliases for package exports and call sites.
AttentionHeadReduction = (
    MeanAttentionHeadReduction
)
MultiheadAttentionReduction = (
    MeanAttentionHeadReduction
)
EdgeAttentionHeadReducer = (
    MeanAttentionHeadReduction
)
apply_attention_head_reduction = (
    reduce_normalized_attention_heads
)
build_edge_attention_head_reducer = (
    build_attention_head_reducer
)


__all__ = (
    "ATTENTION_HEAD_MEAN_FORMULA",
    "ATTENTION_HEAD_REDUCTION_INTERPRETATION",
    "ATTENTION_MULTIHEAD_SCHEMA_VERSION",
    "IMPLEMENTED_ATTENTION_HEAD_REDUCTIONS",
    "AttentionHeadReducer",
    "AttentionHeadReduction",
    "EdgeAttentionHeadReducer",
    "MeanAttentionHeadReduction",
    "MultiheadAttentionReduction",
    "apply_attention_head_reduction",
    "assert_reduced_attention_normalized",
    "attention_head_mean",
    "attention_head_mean_absolute_deviation",
    "attention_head_range",
    "attention_head_standard_deviation",
    "attention_head_variance",
    "build_attention_head_reducer",
    "build_edge_attention_head_reducer",
    "head_disagreement_summary",
    "maximum_attention_head_range",
    "maximum_reduced_attention_normalization_error",
    "mean_attention_head_standard_deviation",
    "mean_reduce_attention_heads",
    "reduce_attention_heads",
    "reduce_normalized_attention_heads",
)
