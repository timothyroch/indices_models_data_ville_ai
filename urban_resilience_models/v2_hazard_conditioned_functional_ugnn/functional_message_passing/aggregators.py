"""
Target-node message aggregation for functional message passing.

Target path:
    urban_resilience_models/
        v2_hazard_conditioned_functional_ugnn/
            functional_message_passing/
                aggregators.py

This module reduces final edge messages by target node. It is intentionally
downstream from every message factor:

    relation transform
    × structural normalization
    × optional relation gate
    × optional edge attention
    × optional semantic edge weight
    → final edge message
    → target-node aggregation

Bounded V2.0 baseline
---------------------
The implemented aggregation mode is ``mean``. For target node ``i``:

    I(i) = {e : target_index[e] = i}
    d_i  = |I(i)|

    aggregate[i] =
        sum(message[e] for e in I(i)) / d_i,  when d_i > 0
        0,                                      when d_i = 0

The denominator is the number of retained incoming edges. It is not:

- the sum of attention weights;
- the number of relation identities;
- the number of semantic relation families;
- graph size;
- source degree;
- target degree computed before masking.

Aggregation groups only by ``target_index``. It does not regroup by relation,
family, hazard, graph, control status, or attention group.

This module owns:

- incoming-edge counts by target node;
- target-node sums;
- target-node mean reduction;
- exact zero output for isolated nodes;
- metadata-preserving ``AggregationOutput`` construction;
- parameter-free architecture and fingerprint identity.

It does not own:

- edge-message construction;
- structural normalization;
- relation gates;
- attention scoring or normalization;
- semantic edge weights;
- residual connections;
- dropout;
- layer normalization;
- graph batching or edge masking.

Contract
--------
- input is a validated ``EdgeMessageOutput``;
- edge messages have shape ``[E, H]``;
- target indices have shape ``[E]`` and dtype ``torch.long``;
- output aggregate has shape ``[N, H]``;
- incoming counts have shape ``[N]`` and dtype ``torch.long``;
- all tensors remain on the source input device;
- aggregate dtype equals edge-message and node-state dtype;
- zero-edge graphs return exact zero aggregates and counts;
- isolated nodes receive exact zero aggregates;
- no trainable parameters, hidden casting, device movement, fallback, or
  additional normalization occurs.
"""

from __future__ import annotations

from hashlib import sha256
import json
from typing import Any, Final, Mapping

import torch
from torch import nn

from ..config import (
    FunctionalMessagePassingConfig,
)
from ..constants import (
    AGGREGATION_MEAN,
    CANONICAL_AGGREGATION_TYPES,
    V2_0_IMPLEMENTED_AGGREGATION_TYPES,
)
from .schemas import (
    AggregationOutput,
    EdgeMessageOutput,
)
from .segment_ops import (
    segment_counts,
    segment_mean,
    segment_sum,
)


# =============================================================================
# Public identity
# =============================================================================


AGGREGATOR_SCHEMA_VERSION: Final[str] = "0.1"


# =============================================================================
# Validation and fingerprint helpers
# =============================================================================


def _normalize_mode(
    mode: str,
) -> str:
    if not isinstance(mode, str):
        raise TypeError(
            "aggregation mode must be a string."
        )

    normalized = mode.strip()

    if not normalized:
        raise ValueError(
            "aggregation mode must be a non-empty string."
        )

    if normalized not in (
        CANONICAL_AGGREGATION_TYPES
    ):
        raise ValueError(
            "Unknown aggregation mode "
            f"{normalized!r}. Expected one of "
            f"{tuple(CANONICAL_AGGREGATION_TYPES)!r}."
        )

    if normalized not in (
        V2_0_IMPLEMENTED_AGGREGATION_TYPES
    ):
        raise NotImplementedError(
            "Aggregation mode "
            f"{normalized!r} is canonical but not implemented in V2.0."
        )

    return normalized


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


def _require_edge_messages(
    messages: EdgeMessageOutput,
) -> None:
    if not isinstance(
        messages,
        EdgeMessageOutput,
    ):
        raise TypeError(
            "messages must be an EdgeMessageOutput."
        )

    inputs = messages.source_inputs
    edge_messages = messages.edge_messages

    if not isinstance(
        edge_messages,
        torch.Tensor,
    ):
        raise TypeError(
            "messages.edge_messages must be a tensor."
        )

    if edge_messages.ndim != 2:
        raise ValueError(
            "messages.edge_messages must have shape [E, H]; "
            f"observed {tuple(edge_messages.shape)}."
        )

    expected_shape = (
        inputs.num_edges,
        inputs.hidden_dim,
    )

    if tuple(edge_messages.shape) != (
        expected_shape
    ):
        raise ValueError(
            "messages.edge_messages shape does not match source inputs. "
            f"Observed {tuple(edge_messages.shape)}; expected "
            f"{expected_shape}."
        )

    if not edge_messages.dtype.is_floating_point:
        raise ValueError(
            "messages.edge_messages must use a floating-point dtype."
        )

    if not bool(
        torch.isfinite(edge_messages)
        .all()
        .item()
    ):
        raise ValueError(
            "messages.edge_messages must contain only finite values."
        )

    if inputs.num_nodes <= 0:
        raise ValueError(
            "Target-node aggregation requires at least one node."
        )

    if not _devices_match(
        edge_messages.device,
        inputs.device,
    ):
        raise ValueError(
            "Edge messages and source inputs must share one device. "
            f"Observed {edge_messages.device} and {inputs.device}."
        )

    if edge_messages.dtype != (
        inputs.dtype
    ):
        raise ValueError(
            "Edge messages and source node state must share one dtype. "
            f"Observed {edge_messages.dtype} and {inputs.dtype}."
        )

    target_index = inputs.target_index

    if not isinstance(
        target_index,
        torch.Tensor,
    ):
        raise TypeError(
            "source_inputs.target_index must be a tensor."
        )

    if target_index.ndim != 1:
        raise ValueError(
            "source_inputs.target_index must have shape [E]; "
            f"observed {tuple(target_index.shape)}."
        )

    if target_index.dtype != torch.long:
        raise ValueError(
            "source_inputs.target_index must use torch.long."
        )

    if int(target_index.shape[0]) != (
        inputs.num_edges
    ):
        raise ValueError(
            "source_inputs.target_index length must equal the edge count "
            f"{inputs.num_edges}; observed "
            f"{int(target_index.shape[0])}."
        )

    if not _devices_match(
        target_index.device,
        inputs.device,
    ):
        raise ValueError(
            "Target indices and source inputs must share one device. "
            f"Observed {target_index.device} and {inputs.device}."
        )

    if target_index.numel() > 0:
        minimum = int(
            target_index.min().item()
        )
        maximum = int(
            target_index.max().item()
        )

        if (
            minimum < 0
            or maximum >= inputs.num_nodes
        ):
            raise ValueError(
                "source_inputs.target_index contains out-of-range node "
                f"indices. Observed range [{minimum}, {maximum}]; "
                f"valid range is [0, {inputs.num_nodes - 1}]."
            )


# =============================================================================
# Target-node aggregator
# =============================================================================


class MessageAggregator(nn.Module):
    """
    Aggregate final edge messages by target node.

    Parameters
    ----------
    mode:
        Canonical aggregation mode. Bounded V2.0 supports only ``mean``.
    """

    mode: str

    def __init__(
        self,
        *,
        mode: str = AGGREGATION_MEAN,
    ) -> None:
        super().__init__()

        self.mode = _normalize_mode(
            mode
        )

        if self.mode != AGGREGATION_MEAN:
            # Defensive exhaustiveness check. _normalize_mode has already
            # rejected canonical unimplemented modes.
            raise RuntimeError(
                "Internal aggregation dispatch is incomplete for mode "
                f"{self.mode!r}."
            )

    # ------------------------------------------------------------------
    # Construction from configuration
    # ------------------------------------------------------------------

    @classmethod
    def from_config(
        cls,
        *,
        config: FunctionalMessagePassingConfig,
    ) -> "MessageAggregator":
        """
        Build the aggregator from functional-message-passing configuration.
        """

        if not isinstance(
            config,
            FunctionalMessagePassingConfig,
        ):
            raise TypeError(
                "config must be a FunctionalMessagePassingConfig."
            )

        config.validate()

        if config.enabled:
            config.assert_implemented()

        return cls(
            mode=config.aggregation_type
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
    def is_mean(self) -> bool:
        return self.mode == AGGREGATION_MEAN

    def architecture_dict(
        self,
    ) -> dict[str, Any]:
        return {
            "schema_version": (
                AGGREGATOR_SCHEMA_VERSION
            ),
            "mode": self.mode,
            "implemented_formula": (
                "mean_messages_by_target_node"
                if self.is_mean
                else None
            ),
            "parameter_count": 0,
            "grouping_axis": (
                "source_inputs.target_index"
            ),
            "num_segments": (
                "source_inputs.num_nodes"
            ),
            "denominator": (
                "retained_incoming_edge_count"
            ),
            "isolated_node_policy": (
                "exact_zero"
            ),
            "relation_agnostic": True,
            "relation_family_agnostic": True,
            "hazard_agnostic": True,
            "graph_agnostic_after_edge_validation": True,
            "attention_group_agnostic": True,
            "operation_order": [
                "validate_edge_message_output",
                "count_edges_by_target_node",
                "sum_messages_by_target_node",
                "divide_nonempty_nodes_by_incoming_edge_count",
                "preserve_exact_zero_for_isolated_nodes",
                "construct_metadata_preserving_output",
            ],
            "output_schema": "AggregationOutput",
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
        """
        Deterministic identity for a parameter-free aggregator.
        """

        return _fingerprint(
            {
                "schema_version": (
                    AGGREGATOR_SCHEMA_VERSION
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
        """
        Parameter-free compatibility hook.
        """

        if self.parameter_count != 0:
            raise RuntimeError(
                "The bounded message aggregator must remain "
                "parameter-free."
            )

    # ------------------------------------------------------------------
    # Grouped reductions
    # ------------------------------------------------------------------

    def compute_incoming_edge_count(
        self,
        messages: EdgeMessageOutput,
    ) -> torch.Tensor:
        """
        Count retained incoming edges for every target node.

        Returns
        -------
        torch.Tensor
            ``torch.long`` tensor with shape ``[N]``.
        """

        _require_edge_messages(messages)
        inputs = messages.source_inputs

        counts = segment_counts(
            inputs.target_index,
            num_segments=inputs.num_nodes,
        )

        expected_shape = (
            inputs.num_nodes,
        )

        if tuple(counts.shape) != (
            expected_shape
        ):
            raise RuntimeError(
                "Incoming-edge counting returned shape "
                f"{tuple(counts.shape)}; expected {expected_shape}."
            )

        if counts.dtype != torch.long:
            raise RuntimeError(
                "Incoming-edge counts must use torch.long."
            )

        if not _devices_match(
            counts.device,
            inputs.device,
        ):
            raise RuntimeError(
                "Incoming-edge counting changed device."
            )

        if bool(
            (counts < 0)
            .any()
            .item()
        ):
            raise RuntimeError(
                "Incoming-edge counts must be nonnegative."
            )

        if int(counts.sum().item()) != (
            inputs.num_edges
        ):
            raise RuntimeError(
                "Incoming-edge counts do not sum to the retained edge "
                "count."
            )

        return counts

    def compute_node_sum(
        self,
        messages: EdgeMessageOutput,
    ) -> torch.Tensor:
        """
        Sum edge messages independently for every target node.

        Returns
        -------
        torch.Tensor
            Tensor ``[N, H]``. Isolated nodes contain exact zeros.
        """

        _require_edge_messages(messages)
        inputs = messages.source_inputs

        node_sum = segment_sum(
            messages.edge_messages,
            inputs.target_index,
            num_segments=inputs.num_nodes,
        )

        self._validate_aggregate_tensor(
            name="node_sum",
            value=node_sum,
            messages=messages,
        )

        return node_sum

    def compute_node_mean(
        self,
        messages: EdgeMessageOutput,
    ) -> torch.Tensor:
        """
        Mean edge messages independently for every target node.

        Isolated nodes receive exact zero.
        """

        _require_edge_messages(messages)
        inputs = messages.source_inputs

        node_mean = segment_mean(
            messages.edge_messages,
            inputs.target_index,
            num_segments=inputs.num_nodes,
        )

        self._validate_aggregate_tensor(
            name="node_mean",
            value=node_mean,
            messages=messages,
        )

        counts = (
            self.compute_incoming_edge_count(
                messages
            )
        )
        isolated = counts == 0

        if bool(isolated.any().item()):
            if not torch.equal(
                node_mean[isolated],
                torch.zeros_like(
                    node_mean[isolated]
                ),
            ):
                raise RuntimeError(
                    "Isolated nodes must receive exact zero mean "
                    "aggregates."
                )

        return node_mean

    def aggregate_tensor(
        self,
        messages: EdgeMessageOutput,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Return ``(node_aggregate, incoming_edge_count)``.

        This lower-level method avoids schema construction while preserving
        the complete message-input contract.
        """

        _require_edge_messages(messages)

        if self.mode == AGGREGATION_MEAN:
            counts = (
                self.compute_incoming_edge_count(
                    messages
                )
            )
            aggregate = (
                self.compute_node_mean(
                    messages
                )
            )
        else:
            raise RuntimeError(
                "Internal aggregation dispatch reached an unsupported "
                f"mode {self.mode!r}."
            )

        return aggregate, counts

    def _validate_aggregate_tensor(
        self,
        *,
        name: str,
        value: torch.Tensor,
        messages: EdgeMessageOutput,
    ) -> None:
        inputs = messages.source_inputs
        expected_shape = (
            inputs.num_nodes,
            inputs.hidden_dim,
        )

        if not isinstance(
            value,
            torch.Tensor,
        ):
            raise RuntimeError(
                f"{name} must be a tensor."
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
            inputs.device,
        ):
            raise RuntimeError(
                f"{name} changed device."
            )

        if value.dtype != inputs.dtype:
            raise RuntimeError(
                f"{name} changed dtype."
            )

        if not bool(
            torch.isfinite(value)
            .all()
            .item()
        ):
            raise FloatingPointError(
                f"{name} contains NaN or infinity."
            )

    # ------------------------------------------------------------------
    # Metadata-preserving public output
    # ------------------------------------------------------------------

    def forward(
        self,
        messages: EdgeMessageOutput,
    ) -> AggregationOutput:
        """
        Aggregate final edge messages and retain their source object.
        """

        aggregate, counts = (
            self.aggregate_tensor(messages)
        )

        return AggregationOutput(
            node_aggregate=aggregate,
            incoming_edge_count=counts,
            source_messages=messages,
            aggregation_mode=self.mode,
            encoder_architecture_fingerprint=(
                self.architecture_fingerprint()
            ),
        )

    # ------------------------------------------------------------------
    # Representation
    # ------------------------------------------------------------------

    def extra_repr(self) -> str:
        return (
            f"mode={self.mode!r}, "
            "grouping=target_node, "
            "parameter_count=0, "
            "isolated_node_policy='exact_zero'"
        )


# A compact compatibility alias for call sites that prefer the generic name.
Aggregator = MessageAggregator


__all__ = (
    "AGGREGATOR_SCHEMA_VERSION",
    "Aggregator",
    "MessageAggregator",
)
