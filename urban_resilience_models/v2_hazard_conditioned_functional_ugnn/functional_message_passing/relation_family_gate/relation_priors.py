"""
Compiled hazard-relation prior integration for relation gating.

Target path:
    urban_resilience_models/
        v2_hazard_conditioned_functional_ugnn/
            functional_message_passing/
                relation_family_gate/
                    relation_priors.py

This module adapts the immutable, ontology-aware
``CompiledHazardRelationPriors`` artifact to the node-aligned exact relation
axis consumed by functional message passing.

It owns:

- validation of compiled prior and exact relation-axis alignment;
- recovery of graph hazard identities from ``HazardQueryEncoding`` lineage;
- explicit graph-to-node prior-row expansion through ``node_batch_index``;
- tensorization of compiled prior means, confidence, masks, and logits;
- confidence-adjusted sigmoid bias lookup through
  ``CompiledHazardRelationPriors.gate_bias_logit_matrix``;
- configurable scaling of prior logits;
- node-aligned regularization weights and resolution diagnostics;
- metadata-preserving ``RelationPriorContribution`` construction.

It does not own:

- prior values, evidence, ontology inheritance, or applicability scope;
- hazard embedding or hazard-query construction;
- neural gate prediction;
- gate activation;
- edge-aligned relation lookup;
- message construction, attention, or aggregation.

Mathematical contract
---------------------
Let the compiled prior artifact contain one row for each compiled hazard and
one column for each exact compiled relation identity.

For node ``n`` belonging to graph ``g(n)`` with hazard row ``h(g(n))``:

    prior_mean[n, r]
        = compiled.prior_mean_matrix[h(g(n)), r]

    confidence[n, r]
        = compiled.confidence_matrix[h(g(n)), r]

    initialization_mask[n, r]
        = compiled.initialization_mask[h(g(n)), r]

    regularization_mask[n, r]
        = compiled.regularization_mask[h(g(n)), r]

    base_logit[n, r]
        = compiled.gate_bias_logit_matrix(...)[h(g(n)), r]

    logit_contribution[n, r]
        = relation_prior_strength * base_logit[n, r]

The compiled artifact already applies its own confidence-adjusted effective
initialization means and neutral behavior. This module does not reinterpret,
pool, or override them.

The trainable gate axis remains the exact compiled relation axis ``R``.
Semantic relation families are never used to pool prior columns.
"""

from __future__ import annotations

from collections import Counter
from hashlib import sha256
import json
import math
from typing import Any, Final, Mapping

import torch
from torch import nn

from ...config import RelationConfig
from ...hazard.hazard_embeddings import (
    HazardEmbeddingLookup,
    NodeAlignedHazardEmbeddingLookup,
)
from ...relations.hazard_relation_priors import (
    CompiledHazardRelationPriors,
    GateInitializationActivation,
)
from ..schemas import (
    FunctionalMessagePassingInputs,
)
from .schemas import (
    RelationGateAxis,
    RelationPriorContribution,
)


# =============================================================================
# Public identity
# =============================================================================


RELATION_PRIOR_INTEGRATION_SCHEMA_VERSION: Final[str] = "0.1"


# =============================================================================
# Generic helpers
# =============================================================================


def _require_nonnegative_finite_float(
    name: str,
    value: float,
) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
    ):
        raise TypeError(
            f"{name} must be numeric."
        )

    converted = float(value)

    if not math.isfinite(converted):
        raise ValueError(
            f"{name} must be finite."
        )

    if converted < 0.0:
        raise ValueError(
            f"{name} must be nonnegative."
        )

    return converted


def _require_probability_epsilon(
    epsilon: float,
) -> float:
    converted = _require_nonnegative_finite_float(
        "epsilon",
        epsilon,
    )

    if not 0.0 < converted < 0.5:
        raise ValueError(
            "epsilon must lie strictly between zero and 0.5."
        )

    return converted


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


def _devices_match(
    first: torch.device | str,
    second: torch.device | str,
) -> bool:
    first_device = torch.device(first)
    second_device = torch.device(second)

    if first_device.type != (
        second_device.type
    ):
        return False

    if first_device.type != "cuda":
        return first_device == (
            second_device
        )

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

    if source_inputs.num_nodes <= 0:
        raise ValueError(
            "Relation-prior integration requires at least one node."
        )

    if source_inputs.num_graphs <= 0:
        raise ValueError(
            "Relation-prior integration requires at least one graph."
        )

    if source_inputs.num_relations <= 0:
        raise ValueError(
            "Relation-prior integration requires at least one relation."
        )

    if not source_inputs.dtype.is_floating_point:
        raise ValueError(
            "Relation-prior integration requires a floating-point "
            "node-state dtype."
        )


def _require_compiled_priors(
    source_inputs: FunctionalMessagePassingInputs,
) -> CompiledHazardRelationPriors:
    compiled = (
        source_inputs.compiled_relation_priors
    )

    if compiled is None:
        raise ValueError(
            "Relation-prior integration requires "
            "source_inputs.compiled_relation_priors."
        )

    if not isinstance(
        compiled,
        CompiledHazardRelationPriors,
    ):
        raise TypeError(
            "source_inputs.compiled_relation_priors must be a "
            "CompiledHazardRelationPriors."
        )

    compiled.validate()

    if compiled.relation_names != (
        source_inputs.relation_names
    ):
        raise ValueError(
            "Compiled hazard-prior relation ordering differs from the "
            "functional message-passing relation axis."
        )

    if compiled.stable_relation_ids != (
        source_inputs.stable_relation_ids
    ):
        raise ValueError(
            "Compiled hazard-prior stable relation IDs differ from the "
            "functional message-passing relation axis."
        )

    if (
        compiled.source_compiled_relation_fingerprint
        != source_inputs
        .compiled_relation_registry
        .fingerprint()
    ):
        raise ValueError(
            "Compiled hazard priors reference a different compiled "
            "relation registry."
        )

    if compiled.num_relations != (
        source_inputs.num_relations
    ):
        raise ValueError(
            "Compiled hazard-prior relation count differs from the "
            "functional message-passing relation count."
        )

    return compiled


def _graph_lookup_and_node_membership(
    source_inputs: FunctionalMessagePassingInputs,
) -> tuple[
    HazardEmbeddingLookup,
    torch.Tensor,
]:
    encoding = source_inputs.hazard_query

    if encoding is None:
        raise ValueError(
            "Relation-prior integration requires "
            "source_inputs.hazard_query so runtime hazard identity is "
            "preserved."
        )

    source_embedding = (
        encoding.source_embedding
    )

    if isinstance(
        source_embedding,
        HazardEmbeddingLookup,
    ):
        graph_lookup = source_embedding
        node_batch_index = (
            source_inputs.node_batch_index
        )
    elif isinstance(
        source_embedding,
        NodeAlignedHazardEmbeddingLookup,
    ):
        graph_lookup = (
            source_embedding.graph_lookup
        )
        node_batch_index = (
            source_embedding.node_batch_index
        )

        if not _devices_match(
            node_batch_index.device,
            source_inputs.node_batch_index.device,
        ):
            raise ValueError(
                "Node-aligned hazard metadata and functional "
                "message-passing inputs must share one device."
            )

        if not torch.equal(
            node_batch_index,
            source_inputs.node_batch_index,
        ):
            raise ValueError(
                "Node-aligned hazard graph membership differs from "
                "source_inputs.node_batch_index."
            )
    else:
        raise TypeError(
            "HazardQueryEncoding.source_embedding must be a "
            "HazardEmbeddingLookup or "
            "NodeAlignedHazardEmbeddingLookup."
        )

    if not _devices_match(
        graph_lookup.embeddings.device,
        source_inputs.device,
    ):
        raise ValueError(
            "Hazard lookup embeddings and functional message-passing "
            "inputs must share one device."
        )

    if not _devices_match(
        graph_lookup.indices.device,
        source_inputs.device,
    ):
        raise ValueError(
            "Hazard identity metadata and functional message-passing "
            "inputs must share one device."
        )

    if not _devices_match(
        node_batch_index.device,
        source_inputs.device,
    ):
        raise ValueError(
            "Node graph membership and functional message-passing "
            "inputs must share one device."
        )

    if len(graph_lookup.indices) != (
        source_inputs.num_graphs
    ):
        raise ValueError(
            "Graph-level hazard identity rows must match the packed "
            f"graph count {source_inputs.num_graphs}; observed "
            f"{len(graph_lookup.indices)}."
        )

    if tuple(
        node_batch_index.shape
    ) != (
        source_inputs.num_nodes,
    ):
        raise ValueError(
            "node_batch_index must have shape [N] for relation-prior "
            "alignment."
        )

    if node_batch_index.dtype != torch.long:
        raise ValueError(
            "node_batch_index must use torch.long for relation-prior "
            "alignment."
        )

    if node_batch_index.numel() > 0:
        minimum = int(
            node_batch_index.min().item()
        )
        maximum = int(
            node_batch_index.max().item()
        )

        if (
            minimum < 0
            or maximum >= source_inputs.num_graphs
        ):
            raise ValueError(
                "node_batch_index contains out-of-range graph indices. "
                f"Observed range [{minimum}, {maximum}]; valid range is "
                f"[0, {source_inputs.num_graphs - 1}]."
            )

    if bool(
        graph_lookup
        .indices
        .unknown_mask
        .any()
        .item()
    ):
        unknown_positions = [
            index
            for index in range(
                len(graph_lookup.indices)
            )
            if bool(
                graph_lookup
                .indices
                .unknown_mask[index]
                .item()
            )
        ]
        raise ValueError(
            "Compiled hazard-relation priors cannot be aligned to "
            "unknown runtime hazards. Unknown graph positions: "
            f"{unknown_positions}."
        )

    return (
        graph_lookup,
        node_batch_index,
    )


def _hazard_row_maps(
    compiled: CompiledHazardRelationPriors,
) -> tuple[
    Mapping[str, int],
    Mapping[int, int],
]:
    if len(compiled.hazard_names) != (
        compiled.num_hazards
    ):
        raise ValueError(
            "Compiled prior hazard names do not align with the hazard "
            "axis."
        )

    if len(compiled.stable_hazard_ids) != (
        compiled.num_hazards
    ):
        raise ValueError(
            "Compiled prior stable hazard IDs do not align with the "
            "hazard axis."
        )

    by_name = {
        name: index
        for index, name
        in enumerate(compiled.hazard_names)
    }
    by_stable_id = {
        stable_id: index
        for index, stable_id
        in enumerate(compiled.stable_hazard_ids)
    }

    if len(by_name) != compiled.num_hazards:
        raise ValueError(
            "Compiled prior hazard names must be unique."
        )

    if len(by_stable_id) != (
        compiled.num_hazards
    ):
        raise ValueError(
            "Compiled prior stable hazard IDs must be unique."
        )

    return by_name, by_stable_id


def _graph_prior_rows(
    *,
    graph_lookup: HazardEmbeddingLookup,
    compiled: CompiledHazardRelationPriors,
    device: torch.device | str,
) -> tuple[
    torch.Tensor,
    tuple[int, ...],
]:
    by_name, by_stable_id = (
        _hazard_row_maps(compiled)
    )

    row_indices: list[int] = []

    for graph_index, hazard_name in enumerate(
        graph_lookup.indices.hazard_names
    ):
        stable_hazard_id = int(
            graph_lookup
            .indices
            .stable_hazard_ids[graph_index]
            .item()
        )

        row_by_name = by_name.get(
            hazard_name
        )
        row_by_stable_id = by_stable_id.get(
            stable_hazard_id
        )

        if row_by_name is None:
            raise ValueError(
                "Runtime hazard "
                f"{hazard_name!r} at graph position {graph_index} is "
                "absent from the compiled prior hazard axis."
            )

        if row_by_stable_id is None:
            raise ValueError(
                "Runtime stable hazard ID "
                f"{stable_hazard_id} at graph position {graph_index} is "
                "absent from the compiled prior hazard axis."
            )

        if row_by_name != row_by_stable_id:
            raise ValueError(
                "Runtime hazard name and stable hazard ID resolve to "
                "different compiled prior rows at graph position "
                f"{graph_index}."
            )

        expected_name = (
            compiled.hazard_names[
                row_by_name
            ]
        )
        expected_stable_id = (
            compiled.stable_hazard_ids[
                row_by_name
            ]
        )

        if (
            expected_name != hazard_name
            or expected_stable_id
            != stable_hazard_id
        ):
            raise ValueError(
                "Runtime hazard identity differs from the compiled prior "
                "hazard ordering."
            )

        row_indices.append(
            row_by_name
        )

    tensor = torch.tensor(
        row_indices,
        dtype=torch.long,
        device=device,
    )

    return tensor, tuple(row_indices)


def _require_node_relation_tensor(
    name: str,
    value: torch.Tensor,
    *,
    source_inputs: FunctionalMessagePassingInputs,
    dtype: torch.dtype | None = None,
) -> None:
    if not isinstance(
        value,
        torch.Tensor,
    ):
        raise RuntimeError(
            f"{name} must be a tensor."
        )

    expected_shape = (
        source_inputs.num_nodes,
        source_inputs.num_relations,
    )

    if tuple(value.shape) != (
        expected_shape
    ):
        raise RuntimeError(
            f"{name} has shape {tuple(value.shape)}; expected "
            f"{expected_shape}."
        )

    if not _devices_match(
        value.device,
        source_inputs.device,
    ):
        raise RuntimeError(
            f"{name} changed device."
        )

    if (
        dtype is not None
        and value.dtype != dtype
    ):
        raise RuntimeError(
            f"{name} changed dtype."
        )

    if value.dtype.is_floating_point:
        if not bool(
            torch.isfinite(value)
            .all()
            .item()
        ):
            raise FloatingPointError(
                f"{name} contains NaN or infinity."
            )


def _matrix_tensor(
    values: Any,
    *,
    dtype: torch.dtype,
    device: torch.device | str,
) -> torch.Tensor:
    return torch.tensor(
        values,
        dtype=dtype,
        device=device,
    )


# =============================================================================
# Compiled prior integration
# =============================================================================


class RelationPriorContributionBuilder(
    nn.Module
):
    """
    Build node-aligned relation-gate prior contributions.

    Parameters
    ----------
    strength:
        Nonnegative scalar multiplying compiled sigmoid gate-bias logits.
    epsilon:
        Clipping epsilon passed to
        ``CompiledHazardRelationPriors.gate_bias_logit_matrix``.
    """

    strength: float
    epsilon: float

    def __init__(
        self,
        *,
        strength: float = 0.0,
        epsilon: float = 1e-4,
    ) -> None:
        super().__init__()

        self.strength = (
            _require_nonnegative_finite_float(
                "strength",
                strength,
            )
        )
        self.epsilon = (
            _require_probability_epsilon(
                epsilon
            )
        )

    # ------------------------------------------------------------------
    # Construction from configuration
    # ------------------------------------------------------------------

    @classmethod
    def from_config(
        cls,
        *,
        config: RelationConfig,
        epsilon: float = 1e-4,
    ) -> "RelationPriorContributionBuilder":
        if not isinstance(
            config,
            RelationConfig,
        ):
            raise TypeError(
                "config must be a RelationConfig."
            )

        config.validate()

        if config.gate_enabled:
            config.assert_implemented()

        return cls(
            strength=(
                config.relation_prior_strength
            ),
            epsilon=epsilon,
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
    def is_zero_strength(self) -> bool:
        return self.strength == 0.0

    def architecture_dict(
        self,
    ) -> dict[str, Any]:
        return {
            "schema_version": (
                RELATION_PRIOR_INTEGRATION_SCHEMA_VERSION
            ),
            "strength": self.strength,
            "epsilon": self.epsilon,
            "parameter_count": 0,
            "gate_axis": (
                "exact_compiled_relation_axis"
            ),
            "hazard_alignment": (
                "graph_hazard_identity_then_node_batch_expansion"
            ),
            "base_logit_source": (
                "CompiledHazardRelationPriors."
                "gate_bias_logit_matrix"
            ),
            "regularization_weight_source": (
                "CompiledHazardRelationPriors."
                "regularization_weight_matrix"
            ),
            "logit_formula": (
                "logit_contribution = "
                "relation_prior_strength * compiled_gate_bias_logit"
            ),
            "family_pooling": False,
            "unknown_hazard_policy": "error",
            "parameter_free": True,
            "operation_order": [
                "validate_functional_message_passing_inputs",
                "validate_compiled_prior_relation_axis",
                "recover_graph_hazard_identities",
                "resolve_compiled_prior_hazard_rows",
                "expand_graph_rows_to_nodes",
                "tensorize_compiled_prior_matrices",
                "scale_compiled_gate_bias_logits",
                "construct_relation_prior_contribution",
            ],
            "output_schema": (
                "RelationPriorContribution"
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
    ) -> str:
        return _fingerprint(
            {
                "schema_version": (
                    RELATION_PRIOR_INTEGRATION_SCHEMA_VERSION
                ),
                "module": type(self).__name__,
                "parameter_count": 0,
                "state_dict_keys": list(
                    self.state_dict()
                ),
            }
        )

    def assert_finite_parameters(
        self,
    ) -> None:
        if self.parameter_count != 0:
            raise RuntimeError(
                "Relation-prior integration must remain parameter-free."
            )

    # ------------------------------------------------------------------
    # Hazard-row alignment
    # ------------------------------------------------------------------

    def resolve_node_hazard_rows(
        self,
        source_inputs: FunctionalMessagePassingInputs,
    ) -> torch.Tensor:
        """
        Return the compiled prior hazard row selected by each node.

        The result has shape ``[N]`` and dtype ``torch.long``.
        """

        _require_inputs(
            source_inputs
        )
        compiled = _require_compiled_priors(
            source_inputs
        )
        graph_lookup, node_batch_index = (
            _graph_lookup_and_node_membership(
                source_inputs
            )
        )
        graph_rows, _ = _graph_prior_rows(
            graph_lookup=graph_lookup,
            compiled=compiled,
            device=source_inputs.device,
        )

        if tuple(graph_rows.shape) != (
            source_inputs.num_graphs,
        ):
            raise RuntimeError(
                "Compiled graph hazard-row lookup returned an invalid "
                "shape."
            )

        if graph_rows.dtype != torch.long:
            raise RuntimeError(
                "Compiled graph hazard-row lookup must use torch.long."
            )

        if not _devices_match(
            graph_rows.device,
            source_inputs.device,
        ):
            raise RuntimeError(
                "Compiled graph hazard-row lookup changed device."
            )

        node_rows = graph_rows[
            node_batch_index
        ]

        if tuple(node_rows.shape) != (
            source_inputs.num_nodes,
        ):
            raise RuntimeError(
                "Node-aligned compiled hazard-row lookup returned an "
                "invalid shape."
            )

        if node_rows.dtype != torch.long:
            raise RuntimeError(
                "Node-aligned compiled hazard-row lookup must use "
                "torch.long."
            )

        if not _devices_match(
            node_rows.device,
            source_inputs.device,
        ):
            raise RuntimeError(
                "Node-aligned compiled hazard-row lookup changed device."
            )

        return node_rows

    # ------------------------------------------------------------------
    # Matrix resolution
    # ------------------------------------------------------------------

    def _resolved_tensors(
        self,
        source_inputs: FunctionalMessagePassingInputs,
    ) -> dict[str, torch.Tensor]:
        _require_inputs(
            source_inputs
        )
        compiled = _require_compiled_priors(
            source_inputs
        )
        node_rows = (
            self.resolve_node_hazard_rows(
                source_inputs
            )
        )

        prior_mean_table = _matrix_tensor(
            compiled.prior_mean_matrix,
            dtype=source_inputs.dtype,
            device=source_inputs.device,
        )
        confidence_table = _matrix_tensor(
            compiled.confidence_matrix,
            dtype=source_inputs.dtype,
            device=source_inputs.device,
        )
        initialization_mask_table = (
            _matrix_tensor(
                compiled.initialization_mask,
                dtype=torch.bool,
                device=source_inputs.device,
            )
        )
        regularization_mask_table = (
            _matrix_tensor(
                compiled.regularization_mask,
                dtype=torch.bool,
                device=source_inputs.device,
            )
        )
        base_logit_table = _matrix_tensor(
            compiled.gate_bias_logit_matrix(
                activation=(
                    GateInitializationActivation
                    .SIGMOID
                ),
                epsilon=self.epsilon,
            ),
            dtype=source_inputs.dtype,
            device=source_inputs.device,
        )
        regularization_weight_table = (
            _matrix_tensor(
                compiled
                .regularization_weight_matrix(),
                dtype=source_inputs.dtype,
                device=source_inputs.device,
            )
        )

        table_shape = (
            compiled.num_hazards,
            compiled.num_relations,
        )

        for name, table, dtype in (
            (
                "prior_mean_table",
                prior_mean_table,
                source_inputs.dtype,
            ),
            (
                "confidence_table",
                confidence_table,
                source_inputs.dtype,
            ),
            (
                "initialization_mask_table",
                initialization_mask_table,
                torch.bool,
            ),
            (
                "regularization_mask_table",
                regularization_mask_table,
                torch.bool,
            ),
            (
                "base_logit_table",
                base_logit_table,
                source_inputs.dtype,
            ),
            (
                "regularization_weight_table",
                regularization_weight_table,
                source_inputs.dtype,
            ),
        ):
            if tuple(table.shape) != (
                table_shape
            ):
                raise RuntimeError(
                    f"{name} has shape {tuple(table.shape)}; expected "
                    f"{table_shape}."
                )

            if table.dtype != dtype:
                raise RuntimeError(
                    f"{name} changed dtype."
                )

            if not _devices_match(
                table.device,
                source_inputs.device,
            ):
                raise RuntimeError(
                    f"{name} changed device."
                )

            if table.dtype.is_floating_point:
                if not bool(
                    torch.isfinite(table)
                    .all()
                    .item()
                ):
                    raise FloatingPointError(
                        f"{name} contains NaN or infinity."
                    )

        prior_mean = prior_mean_table[
            node_rows
        ]
        confidence = confidence_table[
            node_rows
        ]
        initialization_mask = (
            initialization_mask_table[
                node_rows
            ]
        )
        regularization_mask = (
            regularization_mask_table[
                node_rows
            ]
        )
        base_logits = base_logit_table[
            node_rows
        ]
        regularization_weights = (
            regularization_weight_table[
                node_rows
            ]
        )
        logit_contribution = (
            base_logits * self.strength
        )

        for name, value, dtype in (
            (
                "prior_mean",
                prior_mean,
                source_inputs.dtype,
            ),
            (
                "confidence",
                confidence,
                source_inputs.dtype,
            ),
            (
                "initialization_mask",
                initialization_mask,
                torch.bool,
            ),
            (
                "regularization_mask",
                regularization_mask,
                torch.bool,
            ),
            (
                "base_logits",
                base_logits,
                source_inputs.dtype,
            ),
            (
                "regularization_weights",
                regularization_weights,
                source_inputs.dtype,
            ),
            (
                "logit_contribution",
                logit_contribution,
                source_inputs.dtype,
            ),
        ):
            _require_node_relation_tensor(
                name,
                value,
                source_inputs=source_inputs,
                dtype=dtype,
            )

        if bool(
            (
                (prior_mean <= 0)
                | (prior_mean >= 1)
            )
            .any()
            .item()
        ):
            raise RuntimeError(
                "Resolved prior means must lie strictly inside (0, 1)."
            )

        if bool(
            (
                (confidence < 0)
                | (confidence > 1)
            )
            .any()
            .item()
        ):
            raise RuntimeError(
                "Resolved prior confidence must lie in [0, 1]."
            )

        if bool(
            (
                regularization_weights < 0
            )
            .any()
            .item()
        ):
            raise RuntimeError(
                "Resolved regularization weights must be nonnegative."
            )

        if bool(
            (
                regularization_weights[
                    ~regularization_mask
                ]
                != 0
            )
            .any()
            .item()
        ):
            raise RuntimeError(
                "Regularization-disabled cells must have exact zero "
                "regularization weight."
            )

        if self.is_zero_strength:
            if not torch.equal(
                logit_contribution,
                torch.zeros_like(
                    logit_contribution
                ),
            ):
                raise RuntimeError(
                    "Zero relation-prior strength must produce exact zero "
                    "logit contribution."
                )

        return {
            "node_hazard_rows": node_rows,
            "prior_mean": prior_mean,
            "confidence": confidence,
            "initialization_mask": (
                initialization_mask
            ),
            "regularization_mask": (
                regularization_mask
            ),
            "base_logits": base_logits,
            "regularization_weights": (
                regularization_weights
            ),
            "logit_contribution": (
                logit_contribution
            ),
        }

    def regularization_weights(
        self,
        source_inputs: FunctionalMessagePassingInputs,
    ) -> torch.Tensor:
        """
        Return node-aligned confidence-weighted regularization weights.

        Shape is ``[N, R]``. Cells outside the compiled regularization mask
        are exact zero.
        """

        return self._resolved_tensors(
            source_inputs
        )["regularization_weights"]

    def _resolution_summary(
        self,
        source_inputs: FunctionalMessagePassingInputs,
    ) -> dict[str, int]:
        compiled = _require_compiled_priors(
            source_inputs
        )
        graph_lookup, node_batch_index = (
            _graph_lookup_and_node_membership(
                source_inputs
            )
        )
        _, graph_row_indices = (
            _graph_prior_rows(
                graph_lookup=graph_lookup,
                compiled=compiled,
                device=source_inputs.device,
            )
        )

        node_counts = torch.bincount(
            node_batch_index,
            minlength=source_inputs.num_graphs,
        )

        if tuple(node_counts.shape) != (
            source_inputs.num_graphs,
        ):
            raise RuntimeError(
                "Graph node counting returned an invalid shape."
            )

        if node_counts.dtype != torch.long:
            raise RuntimeError(
                "Graph node counting must use torch.long."
            )

        if not _devices_match(
            node_counts.device,
            source_inputs.device,
        ):
            raise RuntimeError(
                "Graph node counting changed device."
            )

        if int(
            node_counts.sum().item()
        ) != source_inputs.num_nodes:
            raise RuntimeError(
                "Graph node counts do not sum to the node count."
            )

        counter: Counter[str] = Counter()

        for graph_index, prior_row in enumerate(
            graph_row_indices
        ):
            graph_node_count = int(
                node_counts[
                    graph_index
                ].item()
            )

            for mode in (
                compiled
                .resolution_mode_matrix[
                    prior_row
                ]
            ):
                counter[mode] += (
                    graph_node_count
                )

        expected_cells = (
            source_inputs.num_nodes
            * source_inputs.num_relations
        )

        if sum(counter.values()) != (
            expected_cells
        ):
            raise RuntimeError(
                "Prior-resolution diagnostics do not cover every "
                "node-relation cell."
            )

        return dict(
            sorted(counter.items())
        )

    # ------------------------------------------------------------------
    # Metadata-preserving output
    # ------------------------------------------------------------------

    def forward(
        self,
        source_inputs: FunctionalMessagePassingInputs,
        *,
        axis: RelationGateAxis | None = None,
    ) -> RelationPriorContribution:
        """
        Build one node-aligned prior contribution over the exact relation axis.
        """

        _require_inputs(
            source_inputs
        )

        resolved_axis = (
            RelationGateAxis.from_inputs(
                source_inputs=source_inputs
            )
            if axis is None
            else axis
        )

        if not isinstance(
            resolved_axis,
            RelationGateAxis,
        ):
            raise TypeError(
                "axis must be a RelationGateAxis or None."
            )

        resolved_axis.assert_matches_inputs(
            source_inputs
        )

        tensors = self._resolved_tensors(
            source_inputs
        )
        compiled = _require_compiled_priors(
            source_inputs
        )

        return RelationPriorContribution(
            logit_contribution=(
                tensors[
                    "logit_contribution"
                ]
            ),
            source_inputs=source_inputs,
            axis=resolved_axis,
            strength=self.strength,
            source_compiled_prior_fingerprint=(
                compiled.fingerprint()
            ),
            prior_mean=tensors[
                "prior_mean"
            ],
            confidence=tensors[
                "confidence"
            ],
            initialization_mask=tensors[
                "initialization_mask"
            ],
            regularization_mask=tensors[
                "regularization_mask"
            ],
            resolution_summary=(
                self._resolution_summary(
                    source_inputs
                )
            ),
        )

    # ------------------------------------------------------------------
    # Representation
    # ------------------------------------------------------------------

    def extra_repr(self) -> str:
        return (
            f"strength={self.strength}, "
            f"epsilon={self.epsilon}, "
            "parameter_count=0, "
            "family_pooling=False"
        )


# Compact aliases for call sites that prefer a shorter stage name.
RelationPriorBuilder = (
    RelationPriorContributionBuilder
)
RelationPriorIntegration = (
    RelationPriorContributionBuilder
)


__all__ = (
    "RELATION_PRIOR_INTEGRATION_SCHEMA_VERSION",
    "RelationPriorBuilder",
    "RelationPriorContributionBuilder",
    "RelationPriorIntegration",
)
