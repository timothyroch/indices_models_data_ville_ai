"""
Immutable contracts for functional edge-message construction.

Target path:
    urban_resilience_models/
        v2_hazard_conditioned_functional_ugnn/
            functional_message_passing/
                message_builders/
                    schemas.py

This module freezes the internal contracts used to construct final edge
messages from already validated functional-message-passing components.

The public final ``EdgeMessageOutput`` remains defined in
``functional_message_passing.schemas``.  The message-builder package adds two
independently auditable internal stages:

1. ``ResolvedMessageCoefficients``

   Resolves every scalar edge coefficient, including exact multiplicative
   identities for disabled mechanisms.

2. ``MessageCompositionOutput``

   Applies the combined scalar coefficient to the already edge-aligned
   relation-transformed source state.

The later ``message_builders.py`` orchestrator can assemble the public
``EdgeMessageOutput`` without recomputing any numerical stage.

Equation
-------------------
For every stored directed edge ``e``:

    u_e
        = relation_transform.transformed_source_state[e]

    n_e
        = structural edge-normalization coefficient

    g_e
        = edge-aligned relation-gate coefficient, or exact one when the gate
          is disabled

    alpha_e
        = final reduced edge-attention coefficient, or exact one when
          attention is disabled

    w_e
        = optional data-provided semantic edge coefficient, or exact one when
          semantic edge weighting is not consumed

The final message is:

    m_e = u_e * n_e * g_e * alpha_e * w_e

Scalar coefficients are broadcast only across the final hidden-feature axis.

Hierarchical interpretation
---------------------------
The factors have distinct meanings:

``relation transform``
    Produces a mechanism-specific source representation.

``structural normalization``
    Encodes a graph-structural scaling rule.

``relation gate``
    Controls how strongly one exact relation mechanism contributes at the
    target node.

``edge attention``
    Routes mass among concrete incoming edges within one fixed
    target-node/exact-relation group.

``semantic edge weight``
    Preserves an externally supplied data coefficient.  It is not silently
    reinterpreted as attention, relation gating, or degree normalization.

The schema does not collapse these factors into one unexplained tensor.  It
retains every resolved factor and its source contract for auditability.

Disabled mechanisms
-------------------
Disabled relation gating and disabled edge attention are represented by
``None`` at their source-output boundary and by exact all-one tensors in the
resolved coefficient stage.

Therefore:

    disabled attention != enabled uniform attention

Enabled uniform attention may assign reciprocal group-size weights.  Disabled
attention contributes one.

Semantic-edge policy
--------------------
``UrbanGraphBatch.semantic_edge_weight`` may be present even when an
experiment does not consume it.  To prevent silent scale changes, the message
builder must choose an explicit policy:

``ignore``
    Preserve the graph field upstream but resolve the message factor to exact
    one.

``use_source_graph``
    Consume the exact source-graph semantic-edge tensor as a multiplicative
    factor.

No untracked override tensor is accepted by this bounded contract.

Relation-transform layout
-------------------------
The existing public ``RelationTransformOutput`` already stores
``transformed_source_state`` in edge-aligned shape ``[E, H]``.  Consequently,
the bounded message builder does not need a second relation-state gather
schema.  Introducing another gather stage would duplicate relation lookup and
create an avoidable opportunity for relation-axis misalignment.

Scope exclusions
----------------
This module does not own:

- relation-transform mathematics;
- structural-normalization mathematics;
- relation-gate prediction or activation;
- edge-attention scoring or normalization;
- semantic-edge-weight estimation;
- target-node aggregation;
- residual updates;
- layer normalization;
- dropout;
- multi-layer execution;
- causal or explanatory claims.

All contracts are immutable, preserve exact source-object lineage, validate
shape/dtype/device consistency, and support empty edge sets.
"""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
from types import MappingProxyType
from typing import Any, Final, Mapping, TypeAlias

import torch

from ..schemas import (
    EdgeAttentionOutput,
    EdgeMessageOutput,
    FunctionalMessagePassingInputs,
    RelationGateOutput,
    RelationTransformOutput,
    StructuralEdgeNormalizationOutput,
)


# =============================================================================
# Public schema and equation identity
# =============================================================================


RESOLVED_MESSAGE_COEFFICIENTS_SCHEMA_VERSION: Final[str] = "0.1"
MESSAGE_COMPOSITION_OUTPUT_SCHEMA_VERSION: Final[str] = "0.1"


MESSAGE_SEMANTIC_EDGE_POLICY_IGNORE: Final[str] = "ignore"
MESSAGE_SEMANTIC_EDGE_POLICY_USE_SOURCE_GRAPH: Final[str] = (
    "use_source_graph"
)

CANONICAL_MESSAGE_SEMANTIC_EDGE_POLICIES: Final[
    tuple[str, ...]
] = (
    MESSAGE_SEMANTIC_EDGE_POLICY_IGNORE,
    MESSAGE_SEMANTIC_EDGE_POLICY_USE_SOURCE_GRAPH,
)

IMPLEMENTED_MESSAGE_SEMANTIC_EDGE_POLICIES: Final[
    tuple[str, ...]
] = CANONICAL_MESSAGE_SEMANTIC_EDGE_POLICIES


MESSAGE_FACTOR_STRUCTURAL_NORMALIZATION: Final[str] = (
    "structural_normalization"
)
MESSAGE_FACTOR_RELATION_GATE: Final[str] = "relation_gate"
MESSAGE_FACTOR_EDGE_ATTENTION: Final[str] = "edge_attention"
MESSAGE_FACTOR_SEMANTIC_EDGE_WEIGHT: Final[str] = (
    "semantic_edge_weight"
)

MESSAGE_FACTOR_ORDER: Final[tuple[str, ...]] = (
    MESSAGE_FACTOR_STRUCTURAL_NORMALIZATION,
    MESSAGE_FACTOR_RELATION_GATE,
    MESSAGE_FACTOR_EDGE_ATTENTION,
    MESSAGE_FACTOR_SEMANTIC_EDGE_WEIGHT,
)

MESSAGE_DISABLED_FACTOR_POLICY: Final[str] = (
    "exact_multiplicative_identity_one"
)

MESSAGE_TRANSFORM_INPUT_LAYOUT: Final[str] = (
    "edge_aligned_transformed_source_state_[E,H]"
)

MESSAGE_COMBINED_COEFFICIENT_FORMULA: Final[str] = (
    "combined_coefficient = "
    "structural_normalization_factor "
    "* relation_gate_factor "
    "* edge_attention_factor "
    "* semantic_edge_factor"
)

MESSAGE_COMPOSITION_FORMULA: Final[str] = (
    "edge_messages = transformed_source_state "
    "* combined_coefficient.unsqueeze(-1)"
)


# =============================================================================
# Generic validation helpers
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


def _require_shape(
    name: str,
    value: torch.Tensor,
    expected: tuple[int, ...],
) -> None:
    observed = tuple(
        int(size)
        for size in value.shape
    )

    if observed != expected:
        raise ValueError(
            f"{name} must have shape {expected}; "
            f"observed {observed}."
        )


def _require_same_device(
    tensors: Mapping[
        str,
        torch.Tensor | None,
    ],
) -> None:
    observed = {
        name: tensor.device
        for name, tensor in tensors.items()
        if tensor is not None
    }

    if not observed:
        return

    devices = set(
        observed.values()
    )

    if len(devices) != 1:
        raise ValueError(
            "All message-builder tensors must share one device. "
            f"Observed {observed}."
        )


def _require_same_float_dtype(
    tensors: Mapping[
        str,
        torch.Tensor | None,
    ],
) -> None:
    observed = {
        name: tensor.dtype
        for name, tensor in tensors.items()
        if tensor is not None
    }

    if not observed:
        return

    dtypes = set(
        observed.values()
    )

    if len(dtypes) != 1:
        raise ValueError(
            "All floating message-builder tensors must share one dtype. "
            f"Observed {observed}."
        )


def _require_exact_identity_ones(
    name: str,
    value: torch.Tensor,
) -> None:
    expected = torch.ones_like(
        value
    )

    if not torch.equal(
        value,
        expected,
    ):
        raise ValueError(
            f"{name} must be the exact multiplicative identity one "
            "when its mechanism is disabled."
        )


def _require_nonnegative(
    name: str,
    value: torch.Tensor,
) -> None:
    if bool(
        (value < 0)
        .any()
        .item()
    ):
        raise ValueError(
            f"{name} must be nonnegative."
        )


def _default_tolerances(
    dtype: torch.dtype,
) -> tuple[float, float]:
    if dtype in (
        torch.float16,
        torch.bfloat16,
    ):
        return 1e-3, 1e-3

    if dtype == torch.float32:
        return 1e-6, 1e-5

    if dtype == torch.float64:
        return 1e-10, 1e-9

    return 1e-6, 1e-5


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


def _tensor_value_fingerprint(
    tensors: Mapping[
        str,
        torch.Tensor,
    ],
) -> str:
    digest = sha256()

    for name in sorted(tensors):
        tensor = tensors[name]

        _require_nonempty_string(
            "tensor fingerprint name",
            name,
        )

        detached = (
            tensor
            .detach()
            .contiguous()
            .cpu()
        )

        digest.update(
            name.encode("utf-8")
        )
        digest.update(
            str(detached.dtype)
            .encode("utf-8")
        )
        digest.update(
            repr(
                tuple(
                    int(size)
                    for size in detached.shape
                )
            ).encode("utf-8")
        )

        if detached.numel() > 0:
            digest.update(
                detached.numpy().tobytes()
            )

    return digest.hexdigest()


def _normalize_semantic_edge_policy(
    value: str,
) -> str:
    if not isinstance(value, str):
        raise TypeError(
            "semantic_edge_policy must be a string."
        )

    normalized = value.strip()

    if not normalized:
        raise ValueError(
            "semantic_edge_policy must be a non-empty string."
        )

    if normalized not in (
        CANONICAL_MESSAGE_SEMANTIC_EDGE_POLICIES
    ):
        raise ValueError(
            "Unknown semantic-edge policy "
            f"{normalized!r}. Expected one of "
            f"{CANONICAL_MESSAGE_SEMANTIC_EDGE_POLICIES!r}."
        )

    if normalized not in (
        IMPLEMENTED_MESSAGE_SEMANTIC_EDGE_POLICIES
    ):
        raise NotImplementedError(
            "Semantic-edge policy "
            f"{normalized!r} is canonical but not implemented."
        )

    return normalized


def _require_source_inputs(
    value: FunctionalMessagePassingInputs,
) -> None:
    if not isinstance(
        value,
        FunctionalMessagePassingInputs,
    ):
        raise TypeError(
            "source_inputs must be a "
            "FunctionalMessagePassingInputs."
        )


def _require_relation_transform(
    value: RelationTransformOutput,
) -> None:
    if not isinstance(
        value,
        RelationTransformOutput,
    ):
        raise TypeError(
            "relation_transform must be a "
            "RelationTransformOutput."
        )


def _require_edge_normalization(
    value: StructuralEdgeNormalizationOutput,
) -> None:
    if not isinstance(
        value,
        StructuralEdgeNormalizationOutput,
    ):
        raise TypeError(
            "edge_normalization must be a "
            "StructuralEdgeNormalizationOutput."
        )


def _require_optional_relation_gate(
    value: RelationGateOutput | None,
) -> None:
    if (
        value is not None
        and not isinstance(
            value,
            RelationGateOutput,
        )
    ):
        raise TypeError(
            "relation_gate must be a RelationGateOutput or None."
        )


def _require_optional_edge_attention(
    value: EdgeAttentionOutput | None,
) -> None:
    if (
        value is not None
        and not isinstance(
            value,
            EdgeAttentionOutput,
        )
    ):
        raise TypeError(
            "edge_attention must be an EdgeAttentionOutput or None."
        )


# =============================================================================
# Resolved scalar message coefficients
# =============================================================================


@dataclass(slots=True, frozen=True)
class ResolvedMessageCoefficients:
    """
    Every scalar edge factor after explicit identity resolution.

    All resolved factor tensors have shape ``[E]``.  Optional source mechanisms
    remain optional in metadata, while their resolved tensors are always
    present so message composition contains no hidden conditionals.

    ``combined_coefficient`` must equal, in the documented operation order:

        structural_normalization_factor
        * relation_gate_factor
        * edge_attention_factor
        * semantic_edge_factor

    The structural, gate, and attention factors are nonnegative.  The bounded
    schema does not force semantic edge weights to be nonnegative because their
    source-data semantics must be established by the graph contract rather
    than silently changed here.
    """

    structural_normalization_factor: torch.Tensor
    relation_gate_factor: torch.Tensor
    edge_attention_factor: torch.Tensor
    semantic_edge_factor: torch.Tensor
    combined_coefficient: torch.Tensor

    source_inputs: FunctionalMessagePassingInputs
    edge_normalization: StructuralEdgeNormalizationOutput

    relation_gate: RelationGateOutput | None
    edge_attention: EdgeAttentionOutput | None
    semantic_edge_weight: torch.Tensor | None

    semantic_edge_policy: str
    resolver_architecture_fingerprint: str

    schema_version: str = (
        RESOLVED_MESSAGE_COEFFICIENTS_SCHEMA_VERSION
    )

    def __post_init__(self) -> None:
        _require_source_inputs(
            self.source_inputs
        )
        _require_edge_normalization(
            self.edge_normalization
        )
        _require_optional_relation_gate(
            self.relation_gate
        )
        _require_optional_edge_attention(
            self.edge_attention
        )

        semantic_edge_policy = (
            _normalize_semantic_edge_policy(
                self.semantic_edge_policy
            )
        )
        object.__setattr__(
            self,
            "semantic_edge_policy",
            semantic_edge_policy,
        )

        if (
            self.edge_normalization.source_inputs
            is not self.source_inputs
        ):
            raise ValueError(
                "edge_normalization must reference the exact supplied "
                "source_inputs object."
            )

        if (
            self.relation_gate is not None
            and self.relation_gate.source_inputs
            is not self.source_inputs
        ):
            raise ValueError(
                "relation_gate and edge_normalization must share the "
                "exact same FunctionalMessagePassingInputs object."
            )

        if (
            self.edge_attention is not None
            and self.edge_attention.source_inputs
            is not self.source_inputs
        ):
            raise ValueError(
                "edge_attention and edge_normalization must share the "
                "exact same FunctionalMessagePassingInputs object."
            )

        expected_shape = (
            self.source_inputs.num_edges,
        )

        for name, tensor in (
            (
                "structural_normalization_factor",
                self.structural_normalization_factor,
            ),
            (
                "relation_gate_factor",
                self.relation_gate_factor,
            ),
            (
                "edge_attention_factor",
                self.edge_attention_factor,
            ),
            (
                "semantic_edge_factor",
                self.semantic_edge_factor,
            ),
            (
                "combined_coefficient",
                self.combined_coefficient,
            ),
        ):
            _require_float_tensor(
                name,
                tensor,
                ndim=1,
            )
            _require_shape(
                name,
                tensor,
                expected_shape,
            )

        if self.semantic_edge_weight is not None:
            _require_float_tensor(
                "semantic_edge_weight",
                self.semantic_edge_weight,
                ndim=1,
            )
            _require_shape(
                "semantic_edge_weight",
                self.semantic_edge_weight,
                expected_shape,
            )

        _require_same_device(
            {
                "node_state": (
                    self
                    .source_inputs
                    .node_state
                    .fused_state
                ),
                "structural_normalization_factor": (
                    self.structural_normalization_factor
                ),
                "relation_gate_factor": (
                    self.relation_gate_factor
                ),
                "edge_attention_factor": (
                    self.edge_attention_factor
                ),
                "semantic_edge_factor": (
                    self.semantic_edge_factor
                ),
                "combined_coefficient": (
                    self.combined_coefficient
                ),
                "semantic_edge_weight": (
                    self.semantic_edge_weight
                ),
            }
        )
        _require_same_float_dtype(
            {
                "node_state": (
                    self
                    .source_inputs
                    .node_state
                    .fused_state
                ),
                "structural_normalization_factor": (
                    self.structural_normalization_factor
                ),
                "relation_gate_factor": (
                    self.relation_gate_factor
                ),
                "edge_attention_factor": (
                    self.edge_attention_factor
                ),
                "semantic_edge_factor": (
                    self.semantic_edge_factor
                ),
                "combined_coefficient": (
                    self.combined_coefficient
                ),
                "semantic_edge_weight": (
                    self.semantic_edge_weight
                ),
            }
        )

        if (
            self.structural_normalization_factor
            is not self.edge_normalization.coefficients
        ):
            raise ValueError(
                "structural_normalization_factor must be the exact "
                "edge_normalization.coefficients tensor object."
            )

        _require_nonnegative(
            "structural_normalization_factor",
            self.structural_normalization_factor,
        )

        if self.relation_gate is None:
            _require_exact_identity_ones(
                "relation_gate_factor",
                self.relation_gate_factor,
            )
        else:
            if (
                self.relation_gate_factor
                is not self.relation_gate.edge_gate_values
            ):
                raise ValueError(
                    "Enabled relation_gate_factor must be the exact "
                    "relation_gate.edge_gate_values tensor object."
                )

            _require_nonnegative(
                "relation_gate_factor",
                self.relation_gate_factor,
            )

        if self.edge_attention is None:
            _require_exact_identity_ones(
                "edge_attention_factor",
                self.edge_attention_factor,
            )
        else:
            if (
                self.edge_attention_factor
                is not self.edge_attention.edge_weights
            ):
                raise ValueError(
                    "Enabled edge_attention_factor must be the exact "
                    "edge_attention.edge_weights tensor object."
                )

            _require_nonnegative(
                "edge_attention_factor",
                self.edge_attention_factor,
            )

        graph_semantic_edge_weight = (
            self
            .source_inputs
            .source_graph
            .semantic_edge_weight
        )

        if semantic_edge_policy == (
            MESSAGE_SEMANTIC_EDGE_POLICY_IGNORE
        ):
            if self.semantic_edge_weight is not None:
                raise ValueError(
                    "semantic_edge_weight must be None when "
                    "semantic_edge_policy='ignore'."
                )

            _require_exact_identity_ones(
                "semantic_edge_factor",
                self.semantic_edge_factor,
            )
        elif semantic_edge_policy == (
            MESSAGE_SEMANTIC_EDGE_POLICY_USE_SOURCE_GRAPH
        ):
            if graph_semantic_edge_weight is None:
                raise ValueError(
                    "semantic_edge_policy='use_source_graph' requires "
                    "source_graph.semantic_edge_weight."
                )

            if (
                self.semantic_edge_weight
                is not graph_semantic_edge_weight
            ):
                raise ValueError(
                    "semantic_edge_weight must be the exact "
                    "source_graph.semantic_edge_weight tensor object."
                )

            if (
                self.semantic_edge_factor
                is not self.semantic_edge_weight
            ):
                raise ValueError(
                    "Enabled semantic_edge_factor must be the exact "
                    "semantic_edge_weight tensor object."
                )
        else:
            raise RuntimeError(
                "Unreachable semantic-edge policy branch."
            )

        expected_combined = (
            self.structural_normalization_factor
            * self.relation_gate_factor
        )
        expected_combined = (
            expected_combined
            * self.edge_attention_factor
        )
        expected_combined = (
            expected_combined
            * self.semantic_edge_factor
        )

        atol, rtol = _default_tolerances(
            self.combined_coefficient.dtype
        )

        if not torch.allclose(
            self.combined_coefficient,
            expected_combined,
            atol=atol,
            rtol=rtol,
        ):
            raise ValueError(
                "combined_coefficient does not equal the explicit "
                "product of the four resolved message factors."
            )

        _require_nonempty_string(
            "resolver_architecture_fingerprint",
            self.resolver_architecture_fingerprint,
        )
        _require_nonempty_string(
            "schema_version",
            self.schema_version,
        )

    @property
    def num_edges(self) -> int:
        return int(
            self.combined_coefficient.shape[0]
        )

    @property
    def dtype(self) -> torch.dtype:
        return self.combined_coefficient.dtype

    @property
    def device(self) -> torch.device:
        return self.combined_coefficient.device

    @property
    def relation_gate_enabled(self) -> bool:
        return self.relation_gate is not None

    @property
    def edge_attention_enabled(self) -> bool:
        return self.edge_attention is not None

    @property
    def semantic_edge_weight_enabled(
        self,
    ) -> bool:
        return self.semantic_edge_weight is not None

    @property
    def factor_mapping(
        self,
    ) -> Mapping[str, torch.Tensor]:
        return MappingProxyType(
            {
                MESSAGE_FACTOR_STRUCTURAL_NORMALIZATION: (
                    self.structural_normalization_factor
                ),
                MESSAGE_FACTOR_RELATION_GATE: (
                    self.relation_gate_factor
                ),
                MESSAGE_FACTOR_EDGE_ATTENTION: (
                    self.edge_attention_factor
                ),
                MESSAGE_FACTOR_SEMANTIC_EDGE_WEIGHT: (
                    self.semantic_edge_factor
                ),
            }
        )

    @property
    def active_factor_names(
        self,
    ) -> tuple[str, ...]:
        names = [
            MESSAGE_FACTOR_STRUCTURAL_NORMALIZATION
        ]

        if self.relation_gate_enabled:
            names.append(
                MESSAGE_FACTOR_RELATION_GATE
            )

        if self.edge_attention_enabled:
            names.append(
                MESSAGE_FACTOR_EDGE_ATTENTION
            )

        if self.semantic_edge_weight_enabled:
            names.append(
                MESSAGE_FACTOR_SEMANTIC_EDGE_WEIGHT
            )

        return tuple(names)

    @property
    def disabled_factor_names(
        self,
    ) -> tuple[str, ...]:
        return tuple(
            name
            for name in MESSAGE_FACTOR_ORDER
            if name not in self.active_factor_names
        )

    @property
    def parameter_fingerprint(
        self,
    ) -> None:
        """
        Coefficient resolution is parameter-free.

        Trainable parameter provenance remains attached to relation-gate and
        edge-attention source outputs.
        """

        return None

    def architecture_dict(
        self,
    ) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "module_contract": (
                "ResolvedMessageCoefficients"
            ),
            "factor_order": list(
                MESSAGE_FACTOR_ORDER
            ),
            "combined_coefficient_formula": (
                MESSAGE_COMBINED_COEFFICIENT_FORMULA
            ),
            "disabled_factor_policy": (
                MESSAGE_DISABLED_FACTOR_POLICY
            ),
            "semantic_edge_policy": (
                self.semantic_edge_policy
            ),
            "relation_gate_enabled": (
                self.relation_gate_enabled
            ),
            "edge_attention_enabled": (
                self.edge_attention_enabled
            ),
            "semantic_edge_weight_enabled": (
                self.semantic_edge_weight_enabled
            ),
            "structural_normalization_mode": (
                self
                .edge_normalization
                .normalization_mode
            ),
            "resolver_architecture_fingerprint": (
                self.resolver_architecture_fingerprint
            ),
            "parameter_free": True,
        }

    def architecture_fingerprint(
        self,
    ) -> str:
        return _fingerprint(
            self.architecture_dict()
        )

    def lineage_dict(
        self,
    ) -> dict[str, Any]:
        values: dict[str, Any] = {
            "source_inputs_lineage_fingerprint": (
                self
                .source_inputs
                .lineage_fingerprint()
            ),
            "edge_normalization_architecture_fingerprint": (
                self
                .edge_normalization
                .encoder_architecture_fingerprint
            ),
            "relation_gate_architecture_fingerprint": (
                self
                .relation_gate
                .encoder_architecture_fingerprint
                if self.relation_gate is not None
                else None
            ),
            "relation_gate_parameter_fingerprint": (
                self
                .relation_gate
                .parameter_fingerprint
                if self.relation_gate is not None
                else None
            ),
            "edge_attention_architecture_fingerprint": (
                self
                .edge_attention
                .encoder_architecture_fingerprint
                if self.edge_attention is not None
                else None
            ),
            "edge_attention_parameter_fingerprint": (
                self
                .edge_attention
                .parameter_fingerprint
                if self.edge_attention is not None
                else None
            ),
            "semantic_edge_policy": (
                self.semantic_edge_policy
            ),
            "resolver_architecture_fingerprint": (
                self.resolver_architecture_fingerprint
            ),
        }

        return values

    def lineage_fingerprint(
        self,
    ) -> str:
        return _fingerprint(
            self.lineage_dict()
        )

    def value_fingerprint(
        self,
    ) -> str:
        return _tensor_value_fingerprint(
            {
                "structural_normalization_factor": (
                    self.structural_normalization_factor
                ),
                "relation_gate_factor": (
                    self.relation_gate_factor
                ),
                "edge_attention_factor": (
                    self.edge_attention_factor
                ),
                "semantic_edge_factor": (
                    self.semantic_edge_factor
                ),
                "combined_coefficient": (
                    self.combined_coefficient
                ),
            }
        )


# =============================================================================
# Parameter-free message composition output
# =============================================================================


@dataclass(slots=True, frozen=True)
class MessageCompositionOutput:
    """
    Edge messages after multiplying transformed states by resolved factors.

    This stage performs no aggregation and introduces no trainable parameters.
    The relation-transformed source state is already edge aligned ``[E, H]``.
    """

    edge_messages: torch.Tensor

    relation_transform: RelationTransformOutput
    resolved_coefficients: ResolvedMessageCoefficients

    composer_architecture_fingerprint: str

    schema_version: str = (
        MESSAGE_COMPOSITION_OUTPUT_SCHEMA_VERSION
    )

    def __post_init__(self) -> None:
        _require_relation_transform(
            self.relation_transform
        )

        if not isinstance(
            self.resolved_coefficients,
            ResolvedMessageCoefficients,
        ):
            raise TypeError(
                "resolved_coefficients must be a "
                "ResolvedMessageCoefficients."
            )

        source_inputs = (
            self.relation_transform.source_inputs
        )

        if (
            self.resolved_coefficients.source_inputs
            is not source_inputs
        ):
            raise ValueError(
                "relation_transform and resolved_coefficients must "
                "share the exact same FunctionalMessagePassingInputs "
                "object."
            )

        _require_float_tensor(
            "edge_messages",
            self.edge_messages,
            ndim=2,
        )
        _require_shape(
            "edge_messages",
            self.edge_messages,
            (
                source_inputs.num_edges,
                source_inputs.hidden_dim,
            ),
        )

        _require_same_device(
            {
                "edge_messages": (
                    self.edge_messages
                ),
                "transformed_source_state": (
                    self
                    .relation_transform
                    .transformed_source_state
                ),
                "combined_coefficient": (
                    self
                    .resolved_coefficients
                    .combined_coefficient
                ),
            }
        )
        _require_same_float_dtype(
            {
                "edge_messages": (
                    self.edge_messages
                ),
                "transformed_source_state": (
                    self
                    .relation_transform
                    .transformed_source_state
                ),
                "combined_coefficient": (
                    self
                    .resolved_coefficients
                    .combined_coefficient
                ),
            }
        )

        expected = (
            self
            .relation_transform
            .transformed_source_state
            * self
            .resolved_coefficients
            .combined_coefficient
            .unsqueeze(-1)
        )

        atol, rtol = _default_tolerances(
            self.edge_messages.dtype
        )

        if not torch.allclose(
            self.edge_messages,
            expected,
            atol=atol,
            rtol=rtol,
        ):
            raise ValueError(
                "edge_messages do not equal transformed_source_state "
                "multiplied by the resolved combined coefficient."
            )

        _require_nonempty_string(
            "composer_architecture_fingerprint",
            self.composer_architecture_fingerprint,
        )
        _require_nonempty_string(
            "schema_version",
            self.schema_version,
        )

    @property
    def source_inputs(
        self,
    ) -> FunctionalMessagePassingInputs:
        return self.relation_transform.source_inputs

    @property
    def num_edges(self) -> int:
        return int(
            self.edge_messages.shape[0]
        )

    @property
    def hidden_dim(self) -> int:
        return int(
            self.edge_messages.shape[1]
        )

    @property
    def dtype(self) -> torch.dtype:
        return self.edge_messages.dtype

    @property
    def device(self) -> torch.device:
        return self.edge_messages.device

    @property
    def combined_coefficient(
        self,
    ) -> torch.Tensor:
        return (
            self
            .resolved_coefficients
            .combined_coefficient
        )

    @property
    def parameter_fingerprint(
        self,
    ) -> None:
        """
        Message composition is parameter-free.

        Trainable parameter provenance remains attached to the relation
        transform, relation gate, and edge attention.
        """

        return None

    def architecture_dict(
        self,
    ) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "module_contract": (
                "MessageCompositionOutput"
            ),
            "transform_input_layout": (
                MESSAGE_TRANSFORM_INPUT_LAYOUT
            ),
            "composition_formula": (
                MESSAGE_COMPOSITION_FORMULA
            ),
            "factor_order": list(
                MESSAGE_FACTOR_ORDER
            ),
            "composer_architecture_fingerprint": (
                self.composer_architecture_fingerprint
            ),
            "parameter_free": True,
            "aggregation_owned_here": False,
            "residual_update_owned_here": False,
            "layer_normalization_owned_here": False,
        }

    def architecture_fingerprint(
        self,
    ) -> str:
        return _fingerprint(
            self.architecture_dict()
        )

    def lineage_dict(
        self,
    ) -> dict[str, Any]:
        return {
            "source_inputs_lineage_fingerprint": (
                self
                .source_inputs
                .lineage_fingerprint()
            ),
            "relation_transform_architecture_fingerprint": (
                self
                .relation_transform
                .encoder_architecture_fingerprint
            ),
            "relation_transform_parameter_fingerprint": (
                self
                .relation_transform
                .parameter_fingerprint
            ),
            "resolved_coefficients_lineage_fingerprint": (
                self
                .resolved_coefficients
                .lineage_fingerprint()
            ),
            "composer_architecture_fingerprint": (
                self.composer_architecture_fingerprint
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
        return _tensor_value_fingerprint(
            {
                "edge_messages": (
                    self.edge_messages
                ),
                "combined_coefficient": (
                    self.combined_coefficient
                ),
            }
        )


# =============================================================================
# Exact message-builder stage chain
# =============================================================================


MessageBuilderStages: TypeAlias = tuple[
    ResolvedMessageCoefficients,
    MessageCompositionOutput,
]


def validate_message_builder_stage_chain(
    *,
    resolved_coefficients: ResolvedMessageCoefficients,
    composition_output: MessageCompositionOutput,
) -> None:
    """
    Validate exact object lineage between message-builder stages.
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

    if (
        composition_output.resolved_coefficients
        is not resolved_coefficients
    ):
        raise ValueError(
            "composition_output must reference the exact supplied "
            "resolved_coefficients object."
        )

    if (
        composition_output.source_inputs
        is not resolved_coefficients.source_inputs
    ):
        raise ValueError(
            "Message-builder stages must share the exact same "
            "FunctionalMessagePassingInputs object."
        )


# =============================================================================
# Public final-output compatibility validation
# =============================================================================


def validate_public_edge_message_output(
    *,
    public_output: EdgeMessageOutput,
    composition_output: MessageCompositionOutput,
) -> None:
    """
    Verify that a public ``EdgeMessageOutput`` exactly reflects one internal
    message-builder stage chain.

    The function performs validation only.  Final output assembly belongs to
    the later orchestrator.
    """

    if not isinstance(
        public_output,
        EdgeMessageOutput,
    ):
        raise TypeError(
            "public_output must be an EdgeMessageOutput."
        )

    if not isinstance(
        composition_output,
        MessageCompositionOutput,
    ):
        raise TypeError(
            "composition_output must be a "
            "MessageCompositionOutput."
        )

    resolved = (
        composition_output
        .resolved_coefficients
    )

    if (
        public_output.source_inputs
        is not composition_output.source_inputs
    ):
        raise ValueError(
            "public_output and composition_output must share the exact "
            "same FunctionalMessagePassingInputs object."
        )

    if (
        public_output.relation_transform
        is not composition_output.relation_transform
    ):
        raise ValueError(
            "public_output must preserve the exact relation_transform "
            "object from composition_output."
        )

    if (
        public_output.edge_normalization
        is not resolved.edge_normalization
    ):
        raise ValueError(
            "public_output must preserve the exact edge_normalization "
            "object from resolved_coefficients."
        )

    if (
        public_output.relation_gate
        is not resolved.relation_gate
    ):
        raise ValueError(
            "public_output must preserve the exact relation_gate object "
            "from resolved_coefficients."
        )

    if (
        public_output.edge_attention
        is not resolved.edge_attention
    ):
        raise ValueError(
            "public_output must preserve the exact edge_attention object "
            "from resolved_coefficients."
        )

    if (
        public_output.semantic_edge_weight
        is not resolved.semantic_edge_weight
    ):
        raise ValueError(
            "public_output must preserve the exact semantic_edge_weight "
            "object from resolved_coefficients."
        )

    if (
        public_output.edge_messages
        is not composition_output.edge_messages
    ):
        raise ValueError(
            "public_output must preserve the exact edge_messages tensor "
            "from composition_output."
        )


# =============================================================================
# Compact aliases
# =============================================================================


MessageCoefficients = ResolvedMessageCoefficients
EdgeMessageCompositionOutput = MessageCompositionOutput


__all__ = (
    # Schema versions.
    "RESOLVED_MESSAGE_COEFFICIENTS_SCHEMA_VERSION",
    "MESSAGE_COMPOSITION_OUTPUT_SCHEMA_VERSION",
    # Semantic-edge policies.
    "MESSAGE_SEMANTIC_EDGE_POLICY_IGNORE",
    "MESSAGE_SEMANTIC_EDGE_POLICY_USE_SOURCE_GRAPH",
    "CANONICAL_MESSAGE_SEMANTIC_EDGE_POLICIES",
    "IMPLEMENTED_MESSAGE_SEMANTIC_EDGE_POLICIES",
    # Factor identity and equations.
    "MESSAGE_FACTOR_STRUCTURAL_NORMALIZATION",
    "MESSAGE_FACTOR_RELATION_GATE",
    "MESSAGE_FACTOR_EDGE_ATTENTION",
    "MESSAGE_FACTOR_SEMANTIC_EDGE_WEIGHT",
    "MESSAGE_FACTOR_ORDER",
    "MESSAGE_DISABLED_FACTOR_POLICY",
    "MESSAGE_TRANSFORM_INPUT_LAYOUT",
    "MESSAGE_COMBINED_COEFFICIENT_FORMULA",
    "MESSAGE_COMPOSITION_FORMULA",
    # Internal contracts.
    "ResolvedMessageCoefficients",
    "MessageCoefficients",
    "MessageCompositionOutput",
    "EdgeMessageCompositionOutput",
    "MessageBuilderStages",
    # Final public contract re-export.
    "EdgeMessageOutput",
    # Validators.
    "validate_message_builder_stage_chain",
    "validate_public_edge_message_output",
)
