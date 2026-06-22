"""
Shared linear relation transform for functional message passing.

Target path:
    urban_resilience_models/
        v2_hazard_conditioned_functional_ugnn/
            functional_message_passing/
                relation_transforms/
                    shared_transform.py

The shared transform is the simplest relation-transform ablation:

    h_source[e] = node_state[source_index[e]]
    transformed[e] = W_shared h_source[e] + b_shared

One parameter set is used for every edge, regardless of exact relation
identity, semantic family, graph membership, control status, hazard, gate, or
attention value.

This module owns only the shared source-state transformation. It does not own:

- relation-mode dispatch;
- per-relation parameters;
- relation registries;
- structural normalization;
- relation gates or hazard priors;
- edge attention;
- message-factor multiplication;
- target-node aggregation;
- residual or layer-normalization updates;
- metadata-bearing ``RelationTransformOutput`` construction.

The public dispatcher in ``relation_transforms.py`` is responsible for
wrapping this tensor result in the subsystem schema.

Contract
--------
- ``node_state`` has shape ``[N, H]`` and floating dtype.
- ``source_index`` has shape ``[E]`` and dtype ``torch.long``.
- ``H`` equals the configured hidden dimension.
- source indices lie in ``[0, N - 1]``.
- tensors and module parameters share one device.
- node-state dtype equals module parameter dtype.
- finite input and finite output are required.
- ``E = 0`` and ``N = 0, E = 0`` are valid.
- no hidden casting, device movement, relation lookup, or fallback occurs.
"""

from __future__ import annotations

from hashlib import sha256
import json
from typing import Any, Final, Mapping

import torch
from torch import nn

from ...constants import RELATION_TRANSFORM_SHARED


# =============================================================================
# Public identity
# =============================================================================


SHARED_RELATION_TRANSFORM_SCHEMA_VERSION: Final[str] = "0.1"


# =============================================================================
# Validation and fingerprint helpers
# =============================================================================


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


def _require_bool(
    name: str,
    value: bool,
) -> None:
    if not isinstance(value, bool):
        raise TypeError(
            f"{name} must be a Boolean."
        )


def _require_nonempty_string(
    name: str,
    value: str,
) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(
            f"{name} must be a non-empty string."
        )


def _require_node_state(
    node_state: torch.Tensor,
    *,
    hidden_dim: int,
) -> None:
    if not isinstance(
        node_state,
        torch.Tensor,
    ):
        raise TypeError(
            "node_state must be a tensor."
        )

    if node_state.ndim != 2:
        raise ValueError(
            "node_state must have shape [N, H]; "
            f"observed {tuple(node_state.shape)}."
        )

    if int(node_state.shape[1]) != hidden_dim:
        raise ValueError(
            "node_state feature width does not match the shared "
            f"transform hidden_dim. Observed {int(node_state.shape[1])}; "
            f"expected {hidden_dim}."
        )

    if not node_state.dtype.is_floating_point:
        raise ValueError(
            "node_state must use a floating-point dtype."
        )

    if not bool(
        torch.isfinite(node_state)
        .all()
        .item()
    ):
        raise ValueError(
            "node_state must contain only finite values."
        )


def _require_source_index(
    source_index: torch.Tensor,
    *,
    num_nodes: int,
) -> None:
    if not isinstance(
        source_index,
        torch.Tensor,
    ):
        raise TypeError(
            "source_index must be a tensor."
        )

    if source_index.ndim != 1:
        raise ValueError(
            "source_index must have shape [E]; "
            f"observed {tuple(source_index.shape)}."
        )

    if source_index.dtype != torch.long:
        raise ValueError(
            "source_index must use torch.long."
        )

    if source_index.numel() == 0:
        return

    if num_nodes == 0:
        raise ValueError(
            "source_index cannot be nonempty when node_state has "
            "zero rows."
        )

    minimum = int(
        source_index.min().item()
    )
    maximum = int(
        source_index.max().item()
    )

    if minimum < 0 or maximum >= num_nodes:
        raise ValueError(
            "source_index contains out-of-range node indices. "
            f"Observed range [{minimum}, {maximum}]; "
            f"valid range is [0, {num_nodes - 1}]."
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


def _state_dict_fingerprint(
    state_dict: Mapping[str, torch.Tensor],
) -> str:
    digest = sha256()

    for name in sorted(state_dict):
        tensor = (
            state_dict[name]
            .detach()
            .cpu()
            .contiguous()
        )

        digest.update(
            name.encode("utf-8")
        )
        digest.update(
            str(tensor.dtype).encode("utf-8")
        )
        digest.update(
            json.dumps(
                list(tensor.shape),
                separators=(",", ":"),
            ).encode("utf-8")
        )
        digest.update(
            tensor.view(torch.uint8)
            .numpy()
            .tobytes()
        )

    return digest.hexdigest()


# =============================================================================
# Shared transform
# =============================================================================


class SharedRelationTransform(nn.Module):
    """
    Apply one shared linear transformation to every edge source state.

    Parameters
    ----------
    hidden_dim:
        Input and output width ``H``.

    bias:
        Whether the shared linear map has an additive bias.

    Notes
    -----
    This class intentionally contains no activation, normalization, or
    dropout. Those operations would change the baseline equation and belong
    to explicitly configured later stages rather than being hidden inside the
    relation transform.
    """

    hidden_dim: int
    bias: bool

    def __init__(
        self,
        *,
        hidden_dim: int,
        bias: bool = True,
    ) -> None:
        super().__init__()

        _require_positive_int(
            "hidden_dim",
            hidden_dim,
        )
        _require_bool(
            "bias",
            bias,
        )

        self.hidden_dim = hidden_dim
        self.bias = bias

        self.linear = nn.Linear(
            hidden_dim,
            hidden_dim,
            bias=bias,
        )

    # ------------------------------------------------------------------
    # Public identity
    # ------------------------------------------------------------------

    @property
    def transform_mode(self) -> str:
        return RELATION_TRANSFORM_SHARED

    @property
    def input_dim(self) -> int:
        return self.hidden_dim

    @property
    def output_dim(self) -> int:
        return self.hidden_dim

    @property
    def device(self) -> torch.device:
        return self.linear.weight.device

    @property
    def dtype(self) -> torch.dtype:
        return self.linear.weight.dtype

    @property
    def parameter_count(self) -> int:
        return sum(
            parameter.numel()
            for parameter in self.parameters()
        )

    @property
    def trainable_parameter_count(
        self,
    ) -> int:
        return sum(
            parameter.numel()
            for parameter in self.parameters()
            if parameter.requires_grad
        )

    def architecture_dict(
        self,
    ) -> dict[str, Any]:
        return {
            "schema_version": (
                SHARED_RELATION_TRANSFORM_SCHEMA_VERSION
            ),
            "transform_mode": (
                RELATION_TRANSFORM_SHARED
            ),
            "hidden_dim": self.hidden_dim,
            "bias": self.bias,
            "parameter_sharing": (
                "one_linear_map_for_all_relations"
            ),
            "operation_order": [
                "gather_source_node_state",
                "shared_linear_transform",
            ],
            "activation": None,
            "normalization": None,
            "dropout": 0.0,
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
        return _state_dict_fingerprint(
            self.state_dict()
        )

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

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
                raise ValueError(
                    "Shared relation-transform parameter "
                    f"{name!r} contains NaN or infinity."
                )

    def _validate_forward_inputs(
        self,
        node_state: torch.Tensor,
        source_index: torch.Tensor,
    ) -> None:
        _require_node_state(
            node_state,
            hidden_dim=self.hidden_dim,
        )
        _require_source_index(
            source_index,
            num_nodes=int(
                node_state.shape[0]
            ),
        )

        if (
            node_state.device
            != source_index.device
        ):
            raise ValueError(
                "node_state and source_index must share one device. "
                f"Observed {node_state.device} and "
                f"{source_index.device}."
            )

        if node_state.device != self.device:
            raise ValueError(
                "node_state and SharedRelationTransform parameters "
                "must share one device. "
                f"Observed {node_state.device} and {self.device}."
            )

        if node_state.dtype != self.dtype:
            raise ValueError(
                "node_state dtype must match "
                "SharedRelationTransform parameter dtype. "
                f"Observed {node_state.dtype} and {self.dtype}."
            )

    # ------------------------------------------------------------------
    # Mathematical operations
    # ------------------------------------------------------------------

    def gather_source_state(
        self,
        node_state: torch.Tensor,
        source_index: torch.Tensor,
    ) -> torch.Tensor:
        """
        Return edge-aligned source states with shape ``[E, H]``.

        This method performs the same strict validation as ``forward`` and is
        exposed for diagnostics and focused tests. It does not apply trainable
        parameters.
        """

        self._validate_forward_inputs(
            node_state,
            source_index,
        )

        gathered = node_state.index_select(
            0,
            source_index,
        )

        expected_shape = (
            int(source_index.shape[0]),
            self.hidden_dim,
        )

        if tuple(gathered.shape) != (
            expected_shape
        ):
            raise RuntimeError(
                "Internal source-state gathering produced an invalid "
                f"shape {tuple(gathered.shape)}; expected "
                f"{expected_shape}."
            )

        return gathered

    def transform_source_state(
        self,
        source_state: torch.Tensor,
    ) -> torch.Tensor:
        """
        Apply the shared linear map to an already edge-aligned state.

        ``source_state`` must have shape ``[E, H]``. This lower-level method
        is useful for dispatcher tests and avoids re-gathering when a caller
        has already validated source alignment.
        """

        _require_node_state(
            source_state,
            hidden_dim=self.hidden_dim,
        )

        if source_state.device != self.device:
            raise ValueError(
                "source_state and SharedRelationTransform parameters "
                "must share one device. "
                f"Observed {source_state.device} and {self.device}."
            )

        if source_state.dtype != self.dtype:
            raise ValueError(
                "source_state dtype must match "
                "SharedRelationTransform parameter dtype. "
                f"Observed {source_state.dtype} and {self.dtype}."
            )

        transformed = self.linear(
            source_state
        )

        expected_shape = (
            int(source_state.shape[0]),
            self.hidden_dim,
        )

        if tuple(transformed.shape) != (
            expected_shape
        ):
            raise RuntimeError(
                "Shared linear transformation produced an invalid "
                f"shape {tuple(transformed.shape)}; expected "
                f"{expected_shape}."
            )

        if not bool(
            torch.isfinite(transformed)
            .all()
            .item()
        ):
            raise FloatingPointError(
                "Shared relation transformation produced NaN or "
                "infinity from finite inputs."
            )

        return transformed

    def forward(
        self,
        node_state: torch.Tensor,
        source_index: torch.Tensor,
    ) -> torch.Tensor:
        """
        Gather source-node states and apply the shared linear transform.

        Returns
        -------
        torch.Tensor
            Edge-aligned transformed source states with shape ``[E, H]``.
        """

        source_state = self.gather_source_state(
            node_state,
            source_index,
        )
        return self.transform_source_state(
            source_state
        )

    # ------------------------------------------------------------------
    # State and representation
    # ------------------------------------------------------------------

    def extra_repr(self) -> str:
        return (
            f"hidden_dim={self.hidden_dim}, "
            f"bias={self.bias}, "
            f"transform_mode={self.transform_mode!r}"
        )


__all__ = (
    "SHARED_RELATION_TRANSFORM_SCHEMA_VERSION",
    "SharedRelationTransform",
)
