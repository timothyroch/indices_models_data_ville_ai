"""
Neural prediction of target-node relation-gate logits.

Target path:
    urban_resilience_models/
        v2_hazard_conditioned_functional_ugnn/
            functional_message_passing/
                relation_family_gate/
                    gate_network.py

The bounded V2.0 gate network predicts one independent logit for every target
node and every exact compiled relation identity.

The trainable gate axis is the compiled relation axis ``R``. Semantic
relation-family metadata is preserved by ``RelationGateAxis`` for diagnostics
and future hierarchical models, but is never used to pool, collapse, or
reorder trainable relation channels.

Permitted inputs
----------------
The network can consume either or both of:

- the fused target-node state ``source_inputs.node_state.fused_state [N, D]``;
- the node-aligned hazard query ``source_inputs.node_hazard_query [N, Q]``.

At least one input source must be enabled. Hazard-query input is enabled by
default because this subsystem is hazard-conditioned. Node-state input is also
enabled by default so relation relevance may vary spatially within one packed
graph.

Relation identity is represented explicitly through one learned embedding per
compiled relation column.

Mathematical contract
---------------------
Let ``x_n`` be the concatenation of enabled node-aligned inputs:

    x_n = concat(node_state_n, hazard_query_n)

The context encoder is:

    h_n = LayerNorm(
        GELU(
            W_2 GELU(W_1 x_n + b_1) + b_2
        )
    )

when layer normalization is enabled, and the same expression without
``LayerNorm`` otherwise.

For learned relation embedding ``e_r`` and optional relation bias ``b_r``:

    logit[n, r] =
        <h_n, e_r> / sqrt(H) + b_r

where ``H`` is ``hidden_dim``.

The output is ``GateNetworkOutput`` with logits ``[N, R]``. No prior,
activation, or edge lookup is applied in this module.

This module owns:

- exact relation-axis parameter alignment;
- node-state and hazard-query feature selection;
- context encoding;
- learned relation-identity scoring;
- parameter initialization and finite-parameter checks;
- architecture and parameter fingerprints;
- metadata-preserving ``GateNetworkOutput`` construction.

It does not own:

- compiled hazard-relation priors;
- sigmoid activation;
- target-node/relation edge lookup;
- relation-family pooling;
- attention, message construction, or aggregation.
"""

from __future__ import annotations

from hashlib import sha256
import json
import math
from typing import Any, Final, Mapping, Sequence

import torch
from torch import nn
from torch.nn import functional as F

from ...config import RelationConfig
from ...constants import (
    CANONICAL_RELATION_GATE_SCOPES,
    RELATION_GATE_SCOPE_TARGET_NODE,
    V2_0_IMPLEMENTED_RELATION_GATE_SCOPES,
)
from ..schemas import (
    FunctionalMessagePassingInputs,
)
from .schemas import (
    GateNetworkOutput,
    RelationGateAxis,
)


# =============================================================================
# Public identity
# =============================================================================


RELATION_GATE_NETWORK_SCHEMA_VERSION: Final[str] = "0.1"


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


def _require_bool(
    name: str,
    value: bool,
) -> bool:
    if not isinstance(value, bool):
        raise TypeError(
            f"{name} must be a bool."
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


def _normalize_scope(
    scope: str,
) -> str:
    if not isinstance(scope, str):
        raise TypeError(
            "relation-gate scope must be a string."
        )

    normalized = scope.strip()

    if not normalized:
        raise ValueError(
            "relation-gate scope must be a non-empty string."
        )

    if normalized not in (
        CANONICAL_RELATION_GATE_SCOPES
    ):
        raise ValueError(
            "Unknown relation-gate scope "
            f"{normalized!r}. Expected one of "
            f"{tuple(CANONICAL_RELATION_GATE_SCOPES)!r}."
        )

    if normalized not in (
        V2_0_IMPLEMENTED_RELATION_GATE_SCOPES
    ):
        raise NotImplementedError(
            "Relation-gate scope "
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
        _canonical_json(payload).encode(
            "utf-8"
        )
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
            "Relation-gate prediction requires at least one node."
        )

    if source_inputs.num_relations <= 0:
        raise ValueError(
            "Relation-gate prediction requires at least one relation."
        )

    if not source_inputs.dtype.is_floating_point:
        raise ValueError(
            "Relation-gate prediction requires a floating-point "
            "node-state dtype."
        )


def _require_feature_matrix(
    name: str,
    value: torch.Tensor,
    *,
    num_nodes: int,
    feature_dim: int,
    dtype: torch.dtype,
    device: torch.device | str,
) -> None:
    if not isinstance(
        value,
        torch.Tensor,
    ):
        raise TypeError(
            f"{name} must be a tensor."
        )

    expected_shape = (
        num_nodes,
        feature_dim,
    )

    if tuple(value.shape) != (
        expected_shape
    ):
        raise ValueError(
            f"{name} must have shape {expected_shape}; "
            f"observed {tuple(value.shape)}."
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
            f"{name} must share the functional message-passing input "
            f"device. Observed {value.device} and {torch.device(device)}."
        )

    if not bool(
        torch.isfinite(value)
        .all()
        .item()
    ):
        raise ValueError(
            f"{name} must contain only finite values."
        )


def _require_output_matrix(
    name: str,
    value: torch.Tensor,
    *,
    expected_shape: tuple[int, int],
    dtype: torch.dtype,
    device: torch.device | str,
) -> None:
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

    if value.dtype != dtype:
        raise RuntimeError(
            f"{name} changed dtype."
        )

    if not _devices_match(
        value.device,
        device,
    ):
        raise RuntimeError(
            f"{name} changed device."
        )

    if not bool(
        torch.isfinite(value)
        .all()
        .item()
    ):
        raise FloatingPointError(
            f"{name} contains NaN or infinity."
        )


# =============================================================================
# Relation gate network
# =============================================================================


class RelationGateNetwork(nn.Module):
    """
    Predict target-node logits over the exact compiled relation axis.

    Parameters
    ----------
    node_state_dim:
        Width of ``source_inputs.node_state.fused_state``.
    hazard_query_dim:
        Width of ``source_inputs.node_hazard_query``. Must be positive even
        when hazard-query input is disabled so the architecture remains
        explicit and serializable.
    relation_names:
        Exact compiled relation order used by trainable relation embeddings.
    stable_relation_ids:
        Stable relation IDs aligned one-to-one with ``relation_names``.
    hidden_dim:
        Context and relation-embedding width.
    scope:
        Canonical relation-gate scope. V2.0 supports only target-node scope.
    use_node_state:
        Whether fused target-node state participates in context prediction.
    use_hazard_query:
        Whether node-aligned hazard query participates in context prediction.
    layer_norm:
        Whether to normalize the hidden context before relation scoring.
    relation_bias:
        Whether to include one learned scalar bias per exact relation.
    """

    node_state_dim: int
    hazard_query_dim: int
    relation_names: tuple[str, ...]
    stable_relation_ids: tuple[int, ...]
    hidden_dim: int
    scope: str
    use_node_state: bool
    use_hazard_query: bool
    layer_norm_enabled: bool
    relation_bias_enabled: bool

    def __init__(
        self,
        *,
        node_state_dim: int,
        hazard_query_dim: int,
        relation_names: Sequence[str],
        stable_relation_ids: Sequence[int],
        hidden_dim: int = 64,
        scope: str = (
            RELATION_GATE_SCOPE_TARGET_NODE
        ),
        use_node_state: bool = True,
        use_hazard_query: bool = True,
        layer_norm: bool = True,
        relation_bias: bool = True,
    ) -> None:
        super().__init__()

        self.node_state_dim = (
            _require_positive_int(
                "node_state_dim",
                node_state_dim,
            )
        )
        self.hazard_query_dim = (
            _require_positive_int(
                "hazard_query_dim",
                hazard_query_dim,
            )
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
                "At least one exact relation is required."
            )

        if len(self.relation_names) != len(
            self.stable_relation_ids
        ):
            raise ValueError(
                "relation_names and stable_relation_ids must align."
            )

        self.hidden_dim = (
            _require_positive_int(
                "hidden_dim",
                hidden_dim,
            )
        )
        self.scope = _normalize_scope(
            scope
        )
        self.use_node_state = _require_bool(
            "use_node_state",
            use_node_state,
        )
        self.use_hazard_query = (
            _require_bool(
                "use_hazard_query",
                use_hazard_query,
            )
        )
        self.layer_norm_enabled = (
            _require_bool(
                "layer_norm",
                layer_norm,
            )
        )
        self.relation_bias_enabled = (
            _require_bool(
                "relation_bias",
                relation_bias,
            )
        )

        if not (
            self.use_node_state
            or self.use_hazard_query
        ):
            raise ValueError(
                "At least one of use_node_state or use_hazard_query "
                "must be enabled."
            )

        if self.scope != (
            RELATION_GATE_SCOPE_TARGET_NODE
        ):
            raise RuntimeError(
                "Internal relation-gate scope dispatch is incomplete for "
                f"scope {self.scope!r}."
            )

        self.input_projection = nn.Linear(
            self.input_dim,
            self.hidden_dim,
            bias=True,
        )
        self.hidden_projection = nn.Linear(
            self.hidden_dim,
            self.hidden_dim,
            bias=True,
        )
        self.context_norm: nn.Module = (
            nn.LayerNorm(
                self.hidden_dim
            )
            if self.layer_norm_enabled
            else nn.Identity()
        )

        self.relation_embeddings = (
            nn.Parameter(
                torch.empty(
                    self.num_relations,
                    self.hidden_dim,
                )
            )
        )

        if self.relation_bias_enabled:
            self.relation_bias = (
                nn.Parameter(
                    torch.empty(
                        self.num_relations
                    )
                )
            )
        else:
            self.register_parameter(
                "relation_bias",
                None,
            )

        self.reset_parameters()

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    @classmethod
    def from_config(
        cls,
        *,
        config: RelationConfig,
        source_inputs: FunctionalMessagePassingInputs,
        use_node_state: bool = True,
        use_hazard_query: bool = True,
        layer_norm: bool = True,
        relation_bias: bool = True,
    ) -> "RelationGateNetwork":
        """
        Build a gate network aligned to one FMP input contract.

        The returned module is moved to the source-input device and dtype.
        """

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

        _require_inputs(
            source_inputs
        )

        node_state = (
            source_inputs
            .node_state
            .fused_state
        )

        if not isinstance(
            node_state,
            torch.Tensor,
        ) or node_state.ndim != 2:
            raise ValueError(
                "source_inputs.node_state.fused_state must have shape "
                "[N, D]."
            )

        hazard_query = (
            source_inputs
            .node_hazard_query
        )

        if use_hazard_query:
            if hazard_query is None:
                raise ValueError(
                    "use_hazard_query=True requires "
                    "source_inputs.node_hazard_query."
                )

            if not isinstance(
                hazard_query,
                torch.Tensor,
            ) or hazard_query.ndim != 2:
                raise ValueError(
                    "source_inputs.node_hazard_query must have shape "
                    "[N, Q]."
                )

            hazard_query_dim = int(
                hazard_query.shape[1]
            )
        else:
            # The explicit constructor still requires a positive serialized
            # width. Reuse the available query width when present; otherwise
            # use one as a non-operative architecture placeholder.
            hazard_query_dim = (
                int(hazard_query.shape[1])
                if isinstance(
                    hazard_query,
                    torch.Tensor,
                )
                and hazard_query.ndim == 2
                and int(
                    hazard_query.shape[1]
                ) > 0
                else 1
            )

        module = cls(
            node_state_dim=int(
                node_state.shape[1]
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
            hidden_dim=(
                config.gate_hidden_dim
            ),
            scope=config.gate_scope,
            use_node_state=use_node_state,
            use_hazard_query=(
                use_hazard_query
            ),
            layer_norm=layer_norm,
            relation_bias=relation_bias,
        )

        return module.to(
            device=source_inputs.device,
            dtype=source_inputs.dtype,
        )

    # ------------------------------------------------------------------
    # Public identity
    # ------------------------------------------------------------------

    @property
    def num_relations(self) -> int:
        return len(self.relation_names)

    @property
    def input_dim(self) -> int:
        value = 0

        if self.use_node_state:
            value += self.node_state_dim

        if self.use_hazard_query:
            value += self.hazard_query_dim

        return value

    @property
    def input_feature_names(
        self,
    ) -> tuple[str, ...]:
        names: list[str] = []

        if self.use_node_state:
            names.append("node_state")

        if self.use_hazard_query:
            names.append("hazard_query")

        return tuple(names)

    @property
    def parameter_count(self) -> int:
        return sum(
            int(parameter.numel())
            for parameter
            in self.parameters()
        )

    @property
    def trainable_parameter_count(
        self,
    ) -> int:
        return sum(
            int(parameter.numel())
            for parameter
            in self.parameters()
            if parameter.requires_grad
        )

    @property
    def relation_score_scale(
        self,
    ) -> float:
        return 1.0 / math.sqrt(
            float(self.hidden_dim)
        )

    def architecture_dict(
        self,
    ) -> dict[str, Any]:
        return {
            "schema_version": (
                RELATION_GATE_NETWORK_SCHEMA_VERSION
            ),
            "scope": self.scope,
            "node_state_dim": (
                self.node_state_dim
            ),
            "hazard_query_dim": (
                self.hazard_query_dim
            ),
            "input_dim": self.input_dim,
            "hidden_dim": self.hidden_dim,
            "num_relations": (
                self.num_relations
            ),
            "relation_names": list(
                self.relation_names
            ),
            "stable_relation_ids": list(
                self.stable_relation_ids
            ),
            "input_feature_names": list(
                self.input_feature_names
            ),
            "use_node_state": (
                self.use_node_state
            ),
            "use_hazard_query": (
                self.use_hazard_query
            ),
            "layer_norm": (
                self.layer_norm_enabled
            ),
            "relation_bias": (
                self.relation_bias_enabled
            ),
            "context_activation": "gelu",
            "context_depth": 2,
            "relation_identity_parameterization": (
                "learned_exact_relation_embeddings"
            ),
            "relation_score_formula": (
                "dot(context, relation_embedding) / sqrt(hidden_dim) "
                "+ optional_relation_bias"
            ),
            "relation_channels_compete": False,
            "family_pooling": False,
            "parameter_count": (
                self.parameter_count
            ),
            "output_schema": (
                "GateNetworkOutput"
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
        return _tensor_fingerprint(
            {
                key: value
                for key, value
                in self.state_dict().items()
            }
        )

    # ------------------------------------------------------------------
    # Initialization and parameter checks
    # ------------------------------------------------------------------

    def reset_parameters(
        self,
    ) -> None:
        nn.init.xavier_uniform_(
            self.input_projection.weight
        )
        nn.init.zeros_(
            self.input_projection.bias
        )

        nn.init.xavier_uniform_(
            self.hidden_projection.weight
        )
        nn.init.zeros_(
            self.hidden_projection.bias
        )

        if isinstance(
            self.context_norm,
            nn.LayerNorm,
        ):
            self.context_norm.reset_parameters()

        nn.init.xavier_uniform_(
            self.relation_embeddings
        )

        if self.relation_bias is not None:
            nn.init.zeros_(
                self.relation_bias
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
                    "Relation-gate network parameter "
                    f"{name!r} contains NaN or infinity."
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
                "Relation-gate network unexpectedly has no parameters."
            )

        device = parameters[0].device
        dtype = parameters[0].dtype

        if not dtype.is_floating_point:
            raise RuntimeError(
                "Relation-gate network parameters must use a "
                "floating-point dtype."
            )

        for parameter in parameters[1:]:
            if not _devices_match(
                parameter.device,
                device,
            ):
                raise RuntimeError(
                    "Relation-gate network parameters span multiple "
                    "devices."
                )

            if parameter.dtype != dtype:
                raise RuntimeError(
                    "Relation-gate network parameters span multiple "
                    "dtypes."
                )

        return device, dtype

    # ------------------------------------------------------------------
    # Input assembly
    # ------------------------------------------------------------------

    def _validate_relation_axis(
        self,
        source_inputs: FunctionalMessagePassingInputs,
        axis: RelationGateAxis,
    ) -> None:
        if not isinstance(
            axis,
            RelationGateAxis,
        ):
            raise TypeError(
                "axis must be a RelationGateAxis."
            )

        axis.assert_matches_inputs(
            source_inputs
        )

        if axis.relation_names != (
            self.relation_names
        ):
            raise ValueError(
                "Relation-gate network relation ordering differs from "
                "the runtime exact relation axis."
            )

        if axis.stable_relation_ids != (
            self.stable_relation_ids
        ):
            raise ValueError(
                "Relation-gate network stable relation IDs differ from "
                "the runtime exact relation axis."
            )

        if axis.num_relations != (
            self.num_relations
        ):
            raise ValueError(
                "Relation-gate network relation count differs from the "
                "runtime exact relation axis."
            )

    def build_context(
        self,
        source_inputs: FunctionalMessagePassingInputs,
    ) -> torch.Tensor:
        """
        Concatenate enabled node-aligned gate-network inputs.

        The result has shape ``[N, input_dim]`` and is never silently cast or
        moved.
        """

        _require_inputs(
            source_inputs
        )

        features: list[torch.Tensor] = []

        if self.use_node_state:
            node_state = (
                source_inputs
                .node_state
                .fused_state
            )
            _require_feature_matrix(
                "source_inputs.node_state.fused_state",
                node_state,
                num_nodes=(
                    source_inputs.num_nodes
                ),
                feature_dim=(
                    self.node_state_dim
                ),
                dtype=source_inputs.dtype,
                device=source_inputs.device,
            )
            features.append(node_state)

        if self.use_hazard_query:
            hazard_query = (
                source_inputs
                .node_hazard_query
            )

            if hazard_query is None:
                raise ValueError(
                    "use_hazard_query=True requires "
                    "source_inputs.node_hazard_query."
                )

            _require_feature_matrix(
                "source_inputs.node_hazard_query",
                hazard_query,
                num_nodes=(
                    source_inputs.num_nodes
                ),
                feature_dim=(
                    self.hazard_query_dim
                ),
                dtype=source_inputs.dtype,
                device=source_inputs.device,
            )
            features.append(
                hazard_query
            )

        if not features:
            raise RuntimeError(
                "Relation-gate network has no enabled input features."
            )

        context = (
            features[0]
            if len(features) == 1
            else torch.cat(
                features,
                dim=-1,
            )
        )

        _require_output_matrix(
            "relation_gate_context",
            context,
            expected_shape=(
                source_inputs.num_nodes,
                self.input_dim,
            ),
            dtype=source_inputs.dtype,
            device=source_inputs.device,
        )

        return context

    # ------------------------------------------------------------------
    # Neural prediction
    # ------------------------------------------------------------------

    def encode_context(
        self,
        context: torch.Tensor,
    ) -> torch.Tensor:
        """
        Encode one ``[N, input_dim]`` context matrix to ``[N, hidden_dim]``.
        """

        if not isinstance(
            context,
            torch.Tensor,
        ):
            raise TypeError(
                "context must be a tensor."
            )

        if context.ndim != 2:
            raise ValueError(
                "context must have shape [N, input_dim]."
            )

        if int(
            context.shape[1]
        ) != self.input_dim:
            raise ValueError(
                "context width differs from the configured input_dim "
                f"{self.input_dim}; observed {int(context.shape[1])}."
            )

        if not context.dtype.is_floating_point:
            raise ValueError(
                "context must use a floating-point dtype."
            )

        if not bool(
            torch.isfinite(context)
            .all()
            .item()
        ):
            raise ValueError(
                "context must contain only finite values."
            )

        parameter_device, parameter_dtype = (
            self._parameter_device_dtype()
        )

        if not _devices_match(
            context.device,
            parameter_device,
        ):
            raise ValueError(
                "Context and relation-gate network parameters must share "
                f"one device. Observed {context.device} and "
                f"{parameter_device}."
            )

        if context.dtype != parameter_dtype:
            raise ValueError(
                "Context and relation-gate network parameters must use "
                f"one dtype. Observed {context.dtype} and "
                f"{parameter_dtype}."
            )

        hidden = self.input_projection(
            context
        )
        hidden = F.gelu(hidden)
        hidden = self.hidden_projection(
            hidden
        )
        hidden = F.gelu(hidden)
        hidden = self.context_norm(
            hidden
        )

        _require_output_matrix(
            "encoded_gate_context",
            hidden,
            expected_shape=(
                int(context.shape[0]),
                self.hidden_dim,
            ),
            dtype=context.dtype,
            device=context.device,
        )

        return hidden

    def score_relations(
        self,
        encoded_context: torch.Tensor,
    ) -> torch.Tensor:
        """
        Score exact relation identities from encoded node context.

        Returns ``[N, R]`` logits.
        """

        if not isinstance(
            encoded_context,
            torch.Tensor,
        ):
            raise TypeError(
                "encoded_context must be a tensor."
            )

        if encoded_context.ndim != 2:
            raise ValueError(
                "encoded_context must have shape [N, hidden_dim]."
            )

        if int(
            encoded_context.shape[1]
        ) != self.hidden_dim:
            raise ValueError(
                "encoded_context width differs from hidden_dim "
                f"{self.hidden_dim}; observed "
                f"{int(encoded_context.shape[1])}."
            )

        if not encoded_context.dtype.is_floating_point:
            raise ValueError(
                "encoded_context must use a floating-point dtype."
            )

        if not bool(
            torch.isfinite(encoded_context)
            .all()
            .item()
        ):
            raise ValueError(
                "encoded_context must contain only finite values."
            )

        parameter_device, parameter_dtype = (
            self._parameter_device_dtype()
        )

        if not _devices_match(
            encoded_context.device,
            parameter_device,
        ):
            raise ValueError(
                "Encoded context and relation-gate network parameters "
                "must share one device."
            )

        if encoded_context.dtype != (
            parameter_dtype
        ):
            raise ValueError(
                "Encoded context and relation-gate network parameters "
                "must use one dtype."
            )

        logits = torch.matmul(
            encoded_context,
            self.relation_embeddings.transpose(
                0,
                1,
            ),
        )
        logits = (
            logits
            * self.relation_score_scale
        )

        if self.relation_bias is not None:
            logits = (
                logits
                + self.relation_bias
            )

        _require_output_matrix(
            "relation_gate_logits",
            logits,
            expected_shape=(
                int(
                    encoded_context.shape[0]
                ),
                self.num_relations,
            ),
            dtype=encoded_context.dtype,
            device=encoded_context.device,
        )

        return logits

    def forward(
        self,
        source_inputs: FunctionalMessagePassingInputs,
        *,
        axis: RelationGateAxis | None = None,
    ) -> GateNetworkOutput:
        """
        Predict metadata-preserving target-node relation logits.
        """

        _require_inputs(
            source_inputs
        )
        self.assert_finite_parameters()

        resolved_axis = (
            RelationGateAxis.from_inputs(
                source_inputs=source_inputs
            )
            if axis is None
            else axis
        )

        self._validate_relation_axis(
            source_inputs,
            resolved_axis,
        )

        context = self.build_context(
            source_inputs
        )
        hidden = self.encode_context(
            context
        )
        logits = self.score_relations(
            hidden
        )

        _require_output_matrix(
            "relation_gate_logits",
            logits,
            expected_shape=(
                source_inputs.num_nodes,
                source_inputs.num_relations,
            ),
            dtype=source_inputs.dtype,
            device=source_inputs.device,
        )

        return GateNetworkOutput(
            logits=logits,
            source_inputs=source_inputs,
            axis=resolved_axis,
            scope=self.scope,
            encoder_architecture_fingerprint=(
                self.architecture_fingerprint()
            ),
            parameter_fingerprint=(
                self.parameter_fingerprint()
            ),
            input_feature_names=(
                self.input_feature_names
            ),
        )

    # ------------------------------------------------------------------
    # Representation
    # ------------------------------------------------------------------

    def extra_repr(self) -> str:
        return (
            f"node_state_dim={self.node_state_dim}, "
            f"hazard_query_dim={self.hazard_query_dim}, "
            f"input_dim={self.input_dim}, "
            f"hidden_dim={self.hidden_dim}, "
            f"num_relations={self.num_relations}, "
            f"scope={self.scope!r}, "
            f"use_node_state={self.use_node_state}, "
            f"use_hazard_query={self.use_hazard_query}, "
            f"layer_norm={self.layer_norm_enabled}, "
            f"relation_bias={self.relation_bias_enabled}"
        )


# Compact alias for call sites that prefer the shorter stage name.
GateNetwork = RelationGateNetwork


__all__ = (
    "RELATION_GATE_NETWORK_SCHEMA_VERSION",
    "GateNetwork",
    "RelationGateNetwork",
)
