"""
Reusable component projection primitive for V2 node-state fusion.

Target path:
    urban_resilience_models/
        v2_hazard_conditioned_functional_ugnn/
            fusion/
                component_projection.py

This module owns:

- validation of one dense fusion-component tensor;
- projection from a component-specific width to the common fusion width;
- the baseline activation/normalization/dropout sequence;
- projection architecture and parameter fingerprints;
- finite-parameter and finite-output diagnostics.

It does not own:

- fusion schemas;
- component extraction from typed upstream results;
- component ordering;
- concatenation or final fusion;
- fusion-mode dispatch;
- attribution or uncertainty propagation.

Baseline projection
-------------------
The current concat-projection baseline uses:

    input
      -> Linear(input_dim, output_dim)
      -> GELU
      -> optional LayerNorm(output_dim)
      -> Dropout

The module recognizes only the strategies implemented here. Future projection
families should be added explicitly rather than silently changing this
baseline.
"""

from __future__ import annotations

from enum import StrEnum
from hashlib import sha256
import json
import math
from typing import Any, Final, Mapping

import torch
from torch import nn


# =============================================================================
# Schema identity
# =============================================================================


COMPONENT_PROJECTION_SCHEMA_VERSION: Final[str] = "0.1"


# =============================================================================
# Controlled vocabularies
# =============================================================================


class ComponentProjectionActivation(StrEnum):
    """Implemented nonlinearities for component projection."""

    GELU = "gelu"


class ComponentProjectionNormalization(StrEnum):
    """Implemented normalization strategies for component projection."""

    NONE = "none"
    LAYER_NORM = "layer_norm"


# =============================================================================
# Generic helpers
# =============================================================================


def _require_nonempty_string(
    name: str,
    value: str,
) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(
            f"{name} must be a non-empty string."
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


def _require_probability(
    name: str,
    value: int | float,
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

    if not 0.0 <= converted < 1.0:
        raise ValueError(
            f"{name} must lie in [0, 1)."
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
        _canonical_json(payload).encode("utf-8")
    ).hexdigest()


def _tensor_fingerprint(
    tensors: Mapping[str, torch.Tensor],
) -> str:
    digest = sha256()

    for name in sorted(tensors):
        tensor = (
            tensors[name]
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


def _assert_finite_tensor(
    name: str,
    tensor: torch.Tensor,
) -> None:
    if (
        tensor.dtype.is_floating_point
        and not bool(
            torch.isfinite(tensor)
            .all()
            .item()
        )
    ):
        raise ValueError(
            f"{name} must contain only finite values."
        )


def _normalize_activation(
    value: ComponentProjectionActivation | str,
) -> ComponentProjectionActivation:
    if isinstance(
        value,
        ComponentProjectionActivation,
    ):
        return value

    return ComponentProjectionActivation(value)


def _normalize_normalization(
    value: ComponentProjectionNormalization | str,
) -> ComponentProjectionNormalization:
    if isinstance(
        value,
        ComponentProjectionNormalization,
    ):
        return value

    return ComponentProjectionNormalization(value)


# =============================================================================
# Projection primitive
# =============================================================================


class ComponentProjection(nn.Module):
    """
    Project one dense fusion component to a common hidden width.

    Parameters
    ----------
    input_dim:
        Width of the incoming component.

    output_dim:
        Common fusion width.

    component_name:
        Stable semantic label used in diagnostics and fingerprints.

    activation:
        Implemented value: ``gelu``.

    normalization:
        ``layer_norm`` or ``none``.

    dropout:
        Dropout probability in ``[0, 1)``.
    """

    def __init__(
        self,
        *,
        input_dim: int,
        output_dim: int,
        component_name: str,
        activation: (
            ComponentProjectionActivation
            | str
        ) = ComponentProjectionActivation.GELU,
        normalization: (
            ComponentProjectionNormalization
            | str
        ) = ComponentProjectionNormalization.LAYER_NORM,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()

        _require_positive_int(
            "input_dim",
            input_dim,
        )
        _require_positive_int(
            "output_dim",
            output_dim,
        )
        _require_nonempty_string(
            "component_name",
            component_name,
        )

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.component_name = component_name
        self.activation = _normalize_activation(
            activation
        )
        self.normalization = (
            _normalize_normalization(
                normalization
            )
        )
        self.dropout = _require_probability(
            "dropout",
            dropout,
        )

        self.linear = nn.Linear(
            input_dim,
            output_dim,
        )

        if (
            self.activation
            == ComponentProjectionActivation.GELU
        ):
            self.activation_layer: nn.Module = nn.GELU()
        else:
            raise NotImplementedError(
                f"Activation {self.activation.value!r} "
                "is canonical but not implemented."
            )

        if (
            self.normalization
            == ComponentProjectionNormalization.LAYER_NORM
        ):
            self.normalization_layer: nn.Module = nn.LayerNorm(
                output_dim
            )
        elif (
            self.normalization
            == ComponentProjectionNormalization.NONE
        ):
            self.normalization_layer = nn.Identity()
        else:
            raise NotImplementedError(
                f"Normalization {self.normalization.value!r} "
                "is canonical but not implemented."
            )

        self.dropout_layer = nn.Dropout(
            self.dropout
        )

    @classmethod
    def baseline(
        cls,
        *,
        input_dim: int,
        output_dim: int,
        component_name: str,
        dropout: float = 0.0,
        layer_norm: bool = True,
    ) -> "ComponentProjection":
        """
        Construct the exact projection used by the concat baseline.
        """

        if not isinstance(
            layer_norm,
            bool,
        ):
            raise TypeError(
                "layer_norm must be a Boolean."
            )

        return cls(
            input_dim=input_dim,
            output_dim=output_dim,
            component_name=component_name,
            activation=(
                ComponentProjectionActivation.GELU
            ),
            normalization=(
                ComponentProjectionNormalization.LAYER_NORM
                if layer_norm
                else ComponentProjectionNormalization.NONE
            ),
            dropout=dropout,
        )

    # ------------------------------------------------------------------
    # Forward path
    # ------------------------------------------------------------------

    @property
    def device(self) -> torch.device:
        return self.linear.weight.device

    @property
    def dtype(self) -> torch.dtype:
        return self.linear.weight.dtype

    def forward(
        self,
        values: torch.Tensor,
    ) -> torch.Tensor:
        if not isinstance(
            values,
            torch.Tensor,
        ):
            raise TypeError(
                f"{self.component_name} values must be a tensor."
            )

        if values.ndim != 2:
            raise ValueError(
                f"{self.component_name} values must have shape "
                "[items, input_dim]."
            )

        if int(
            values.shape[1]
        ) != self.input_dim:
            raise ValueError(
                f"{self.component_name} feature width "
                f"{int(values.shape[1])} does not match "
                f"input_dim={self.input_dim}."
            )

        if not values.dtype.is_floating_point:
            raise ValueError(
                f"{self.component_name} values must use a "
                "floating-point dtype."
            )

        if values.device != self.device:
            raise ValueError(
                f"{self.component_name} values and "
                "ComponentProjection must share one device."
            )

        _assert_finite_tensor(
            f"{self.component_name} values",
            values,
        )

        projected = self.linear(
            values.to(
                dtype=self.dtype
            )
        )
        projected = self.activation_layer(
            projected
        )
        projected = self.normalization_layer(
            projected
        )
        projected = self.dropout_layer(
            projected
        )

        if projected.ndim != 2:
            raise RuntimeError(
                "ComponentProjection returned an invalid rank."
            )

        if tuple(
            projected.shape
        ) != (
            int(values.shape[0]),
            self.output_dim,
        ):
            raise RuntimeError(
                "ComponentProjection returned a shape that differs "
                "from its architecture contract."
            )

        _assert_finite_tensor(
            f"projected {self.component_name}",
            projected,
        )

        return projected

    # ------------------------------------------------------------------
    # Identity and diagnostics
    # ------------------------------------------------------------------

    def architecture_dict(
        self,
    ) -> dict[str, Any]:
        return {
            "schema_version": (
                COMPONENT_PROJECTION_SCHEMA_VERSION
            ),
            "component_name": (
                self.component_name
            ),
            "input_dim": self.input_dim,
            "output_dim": self.output_dim,
            "activation": (
                self.activation.value
            ),
            "normalization": (
                self.normalization.value
            ),
            "dropout": self.dropout,
            "operation_order": [
                "linear",
                "activation",
                "normalization",
                "dropout",
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
    ) -> str:
        return _tensor_fingerprint(
            {
                name: tensor
                for name, tensor
                in self.state_dict().items()
            }
        )

    def assert_finite_parameters(
        self,
    ) -> None:
        for name, tensor in (
            self.state_dict().items()
        ):
            if (
                tensor.dtype.is_floating_point
                and not bool(
                    torch.isfinite(tensor)
                    .all()
                    .item()
                )
            ):
                raise ValueError(
                    f"ComponentProjection tensor {name!r} "
                    "contains NaN or infinity."
                )

    def extra_repr(
        self,
    ) -> str:
        return (
            f"component_name={self.component_name!r}, "
            f"input_dim={self.input_dim}, "
            f"output_dim={self.output_dim}, "
            f"activation={self.activation.value!r}, "
            f"normalization={self.normalization.value!r}, "
            f"dropout={self.dropout}"
        )


__all__ = (
    "COMPONENT_PROJECTION_SCHEMA_VERSION",
    "ComponentProjection",
    "ComponentProjectionActivation",
    "ComponentProjectionNormalization",
)
