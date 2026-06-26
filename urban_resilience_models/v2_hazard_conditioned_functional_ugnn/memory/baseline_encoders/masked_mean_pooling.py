"""
Deterministic masked-mean temporal pooling.

Target path:
    urban_resilience_models/
        v2_hazard_conditioned_functional_ugnn/
            memory/
                baseline_encoders/
                    masked_mean_pooling.py

This module pools a preserved temporal sequence with exact uniform mass over
valid timesteps:

    weight[n, 0, t] =
        1 / valid_length[n]    when timestep_mask[n, t] is True
        0                      otherwise

    pooled[n, :] =
        sum_t weight[n, 0, t] * encoded_sequence[n, t, :]

The operation consumes ``TemporalSequenceEncoding`` and returns the canonical
``TemporalPoolingOutput``. The exact source-encoding object is preserved.

Scientific role
---------------
Masked mean is a hazard-independent deterministic reduction. It introduces:

- no learned temporal weighting;
- no order-sensitive temporal interaction;
- no trainable fallback;
- no causal or feature-importance interpretation.

It intentionally discards temporal order after the sequence encoder. Any
temporal interaction already present in a GRU, LSTM, or Transformer encoding
remains embedded in the timestep representations, but the pooling operation
itself is a uniform reduction.

Projection policy
-----------------
Phase 5 keeps this baseline parameter-free. ``project_output=True`` is
recognized by the shared configuration but deliberately rejected here. A
projected pooled representation is a separate learned component and should be
introduced explicitly rather than hidden inside the deterministic baseline.

Consequently:

    config.output_dim == source_encoding.hidden_dim

is required at execution.

Zero-history policy
-------------------
Supported policies:

``error``
    Reject a batch containing any node with zero valid timesteps.

``zero``
    Emit exactly zero pooling weights and exactly zero pooled memory for every
    zero-history node.

``learned_fallback``
    Rejected. A learned cold-start representation is not part of this
    deterministic baseline.

Numerical policy
----------------
The complete source sequence is revalidated for finiteness and exact zero
padding at execution time. This protects against in-place mutation after the
frozen Phase 4 schema was constructed.

Pooling weights use the same floating dtype and device as the source sequence.
"""

from __future__ import annotations

from typing import Final

import torch
from torch import nn

from ..config import (
    TemporalPoolingConfig,
)
from ..schemas.provenance import (
    MemoryParameterSnapshotProvenance,
)
from ..schemas.sequence_encoding import (
    TemporalSequenceEncoding,
)
from ..schemas.temporal_pooling import (
    TemporalPoolingHeadReduction,
    TemporalPoolingKind,
    TemporalPoolingOutput,
    TemporalPoolingZeroHistoryPolicy,
)
from ._provenance import (
    BASELINE_PROVENANCE_IMPLEMENTATION_VERSION,
    baseline_configuration_fingerprint,
    build_temporal_pooling_computation_provenance,
    module_parameter_snapshot_fingerprint,
)


# =============================================================================
# Component identity
# =============================================================================


MASKED_MEAN_POOLING_COMPONENT_NAME: Final[str] = (
    "masked_mean_temporal_pooler"
)

MASKED_MEAN_POOLING_COMPONENT_KIND: Final[str] = "masked_mean"

MASKED_MEAN_POOLING_OPERATION_NAME: Final[str] = (
    "pool_masked_mean_sequence"
)

MASKED_MEAN_POOLING_IMPLEMENTATION_VERSION: Final[str] = (
    BASELINE_PROVENANCE_IMPLEMENTATION_VERSION
)

MASKED_MEAN_POOLING_TEMPORAL_INTERACTION: Final[bool] = False

MASKED_MEAN_POOLING_HAZARD_CONDITIONED: Final[bool] = False

MASKED_MEAN_POOLING_WEIGHT_POLICY: Final[str] = (
    "uniform_unit_mass_over_valid_timesteps"
)

MASKED_MEAN_POOLING_PADDING_POLICY: Final[str] = (
    "exact_zero_mass_at_padded_timesteps"
)

MASKED_MEAN_POOLING_PROJECTION_POLICY: Final[str] = (
    "phase_five_parameter_free_no_output_projection"
)

MASKED_MEAN_POOLING_ZERO_HISTORY_POLICIES: Final[
    tuple[str, ...]
] = (
    TemporalPoolingZeroHistoryPolicy.ERROR.value,
    TemporalPoolingZeroHistoryPolicy.ZERO.value,
)


# =============================================================================
# Validation helpers
# =============================================================================


def _validate_masked_mean_config(
    config: TemporalPoolingConfig,
) -> None:
    if not isinstance(
        config,
        TemporalPoolingConfig,
    ):
        raise TypeError(
            "config must be a TemporalPoolingConfig."
        )

    if config.kind != TemporalPoolingKind.MASKED_MEAN:
        raise ValueError(
            "MaskedMeanTemporalPooler requires "
            "config.kind='masked_mean'."
        )

    if config.num_heads != 1:
        raise ValueError(
            "MaskedMeanTemporalPooler requires num_heads=1."
        )

    if (
        config.head_reduction
        != TemporalPoolingHeadReduction.SINGLE_HEAD
    ):
        raise ValueError(
            "MaskedMeanTemporalPooler requires "
            "head_reduction='single_head'."
        )

    if config.head_reduction_weights is not None:
        raise ValueError(
            "MaskedMeanTemporalPooler does not accept "
            "head_reduction_weights."
        )

    if config.project_output:
        raise NotImplementedError(
            "MaskedMeanTemporalPooler is parameter-free in Phase 5. "
            "project_output=True is recognized but not implemented."
        )

    if config.score_hidden_dim is not None:
        raise ValueError(
            "MaskedMeanTemporalPooler does not use score_hidden_dim."
        )

    if config.dropout != 0.0:
        raise ValueError(
            "MaskedMeanTemporalPooler requires dropout=0."
        )

    if (
        config.zero_history_policy
        == TemporalPoolingZeroHistoryPolicy.LEARNED_FALLBACK
    ):
        raise NotImplementedError(
            "MaskedMeanTemporalPooler does not implement "
            "zero_history_policy='learned_fallback'."
        )

    if (
        config.zero_history_policy.value
        not in MASKED_MEAN_POOLING_ZERO_HISTORY_POLICIES
    ):
        raise ValueError(
            "Unsupported masked-mean zero-history policy "
            f"{config.zero_history_policy.value!r}."
        )


def _validate_source_encoding(
    source_encoding: TemporalSequenceEncoding,
    *,
    expected_output_dim: int,
) -> None:
    if not isinstance(
        source_encoding,
        TemporalSequenceEncoding,
    ):
        raise TypeError(
            "source_encoding must be a TemporalSequenceEncoding."
        )

    if source_encoding.hidden_dim != expected_output_dim:
        raise ValueError(
            "Parameter-free masked mean preserves the encoded feature "
            "width. config.output_dim must equal "
            "source_encoding.hidden_dim: expected "
            f"{expected_output_dim}, observed "
            f"{source_encoding.hidden_dim}."
        )

    finite = torch.isfinite(
        source_encoding.encoded_sequence
    )

    if not bool(
        finite.all().item()
    ):
        nonfinite_count = int(
            (
                ~finite
            )
            .sum()
            .item()
        )
        raise ValueError(
            "source_encoding.encoded_sequence must remain finite at "
            "execution time. Nonfinite count: "
            f"{nonfinite_count}."
        )

    padded = (
        ~source_encoding
        .timestep_mask
    ).unsqueeze(
        -1
    ).expand_as(
        source_encoding
        .encoded_sequence
    )

    padded_values = (
        source_encoding
        .encoded_sequence[
            padded
        ]
    )

    if (
        padded_values.numel() > 0
        and not torch.equal(
            padded_values,
            torch.zeros_like(
                padded_values
            ),
        )
    ):
        raise ValueError(
            "source_encoding.encoded_sequence must retain exact zero "
            "padding at execution time."
        )


def _validate_zero_history_policy(
    source_encoding: TemporalSequenceEncoding,
    *,
    policy: TemporalPoolingZeroHistoryPolicy,
) -> None:
    zero_history = (
        source_encoding
        .valid_lengths
        == 0
    )

    if (
        policy
        == TemporalPoolingZeroHistoryPolicy.ERROR
        and bool(
            zero_history.any().item()
        )
    ):
        indices = (
            torch.nonzero(
                zero_history,
                as_tuple=False,
            )
            .flatten()
            .detach()
            .cpu()
            .tolist()
        )
        raise ValueError(
            "MaskedMeanTemporalPooler received zero-history nodes "
            "under zero_history_policy='error'. Node indices: "
            f"{indices}."
        )


def _validate_parameter_snapshot(
    module: nn.Module,
    parameter_snapshot: (
        MemoryParameterSnapshotProvenance
        | None
    ),
) -> None:
    if parameter_snapshot is None:
        return

    if not isinstance(
        parameter_snapshot,
        MemoryParameterSnapshotProvenance,
    ):
        raise TypeError(
            "parameter_snapshot must be a "
            "MemoryParameterSnapshotProvenance or None."
        )

    if (
        parameter_snapshot.parameter_count
        not in (
            None,
            0,
        )
    ):
        raise ValueError(
            "MaskedMeanTemporalPooler parameter snapshots must report "
            "parameter_count=0 when supplied."
        )

    if (
        parameter_snapshot.trainable_parameter_count
        not in (
            None,
            0,
        )
    ):
        raise ValueError(
            "MaskedMeanTemporalPooler parameter snapshots must report "
            "trainable_parameter_count=0 when supplied."
        )

    expected = module_parameter_snapshot_fingerprint(
        module
    )

    if (
        parameter_snapshot
        .parameter_snapshot_fingerprint
        != expected
    ):
        raise ValueError(
            "parameter_snapshot does not identify this exact "
            "parameter-free MaskedMeanTemporalPooler."
        )


def _masked_mean_weights(
    source_encoding: TemporalSequenceEncoding,
) -> torch.Tensor:
    """
    Return exact deterministic weights with shape ``[N, 1, T]``.
    """

    mask = source_encoding.timestep_mask
    lengths = mask.sum(
        dim=-1,
        keepdim=True,
    )

    safe_lengths = lengths.clamp_min(
        1
    )

    weights = (
        mask.to(
            dtype=source_encoding.dtype
        )
        / safe_lengths.to(
            dtype=source_encoding.dtype
        )
    )

    return weights.unsqueeze(
        1
    )


def _masked_mean_memory(
    source_encoding: TemporalSequenceEncoding,
    pooling_weights: torch.Tensor,
) -> torch.Tensor:
    pooled = torch.bmm(
        pooling_weights,
        source_encoding.encoded_sequence,
    ).squeeze(
        1
    )

    if not bool(
        torch.isfinite(
            pooled
        ).all().item()
    ):
        raise ValueError(
            "MaskedMeanTemporalPooler produced nonfinite pooled "
            "memory."
        )

    zero_history = (
        source_encoding
        .valid_lengths
        == 0
    )

    return torch.where(
        zero_history.unsqueeze(
            -1
        ),
        torch.zeros_like(
            pooled
        ),
        pooled,
    )


# =============================================================================
# Masked-mean pooler
# =============================================================================


class MaskedMeanTemporalPooler(nn.Module):
    """
    Uniformly average valid temporal representations.

    Parameters
    ----------
    config:
        ``TemporalPoolingConfig`` with ``kind='masked_mean'``,
        ``num_heads=1``, ``head_reduction='single_head'``, and no output
        projection.

    Returns
    -------
    TemporalPoolingOutput
        Canonical Phase 4 pooling contract with exact source identity and
        deterministic weights ``[N, 1, T]``.
    """

    config: TemporalPoolingConfig

    def __init__(
        self,
        config: TemporalPoolingConfig,
    ) -> None:
        super().__init__()

        _validate_masked_mean_config(
            config
        )

        self.config = config
        self._configuration_fingerprint = (
            baseline_configuration_fingerprint(
                config
            )
        )

    # -------------------------------------------------------------------------
    # Structural properties
    # -------------------------------------------------------------------------

    @property
    def output_dim(
        self,
    ) -> int:
        return self.config.output_dim

    @property
    def num_heads(
        self,
    ) -> int:
        return 1

    @property
    def pooling_kind(
        self,
    ) -> TemporalPoolingKind:
        return TemporalPoolingKind.MASKED_MEAN

    @property
    def head_reduction(
        self,
    ) -> TemporalPoolingHeadReduction:
        return TemporalPoolingHeadReduction.SINGLE_HEAD

    @property
    def zero_history_policy(
        self,
    ) -> TemporalPoolingZeroHistoryPolicy:
        return self.config.zero_history_policy

    @property
    def configuration_fingerprint(
        self,
    ) -> str:
        return self._configuration_fingerprint

    @property
    def parameter_count(
        self,
    ) -> int:
        return 0

    @property
    def trainable_parameter_count(
        self,
    ) -> int:
        return 0

    @property
    def has_temporal_interaction(
        self,
    ) -> bool:
        return MASKED_MEAN_POOLING_TEMPORAL_INTERACTION

    @property
    def is_hazard_conditioned(
        self,
    ) -> bool:
        return MASKED_MEAN_POOLING_HAZARD_CONDITIONED

    # -------------------------------------------------------------------------
    # Provenance semantics
    # -------------------------------------------------------------------------

    def architecture_metadata(
        self,
    ) -> dict[str, object]:
        return {
            "pooling_kind": self.pooling_kind.value,
            "output_dim": self.output_dim,
            "num_heads": self.num_heads,
            "head_reduction": self.head_reduction.value,
            "zero_history_policy": (
                self.zero_history_policy.value
            ),
            "weight_policy": (
                MASKED_MEAN_POOLING_WEIGHT_POLICY
            ),
            "padding_policy": (
                MASKED_MEAN_POOLING_PADDING_POLICY
            ),
            "projection_policy": (
                MASKED_MEAN_POOLING_PROJECTION_POLICY
            ),
            "project_output": False,
            "parameter_count": 0,
            "trainable_parameter_count": 0,
            "trainable_temporal_weights": False,
            "temporal_interaction": False,
            "hazard_conditioned": False,
            "order_sensitive_pooling": False,
            "source_sequence_preserved": True,
            "implementation_version": (
                MASKED_MEAN_POOLING_IMPLEMENTATION_VERSION
            ),
        }

    # -------------------------------------------------------------------------
    # Forward
    # -------------------------------------------------------------------------

    def forward(
        self,
        source_encoding: TemporalSequenceEncoding,
        *,
        parameter_snapshot: (
            MemoryParameterSnapshotProvenance
            | None
        ) = None,
    ) -> TemporalPoolingOutput:
        """
        Pool one encoded temporal batch with exact uniform valid-time weights.
        """

        _validate_source_encoding(
            source_encoding,
            expected_output_dim=self.output_dim,
        )
        _validate_zero_history_policy(
            source_encoding,
            policy=self.zero_history_policy,
        )
        _validate_parameter_snapshot(
            self,
            parameter_snapshot,
        )

        pooling_weights = _masked_mean_weights(
            source_encoding
        )
        pooled_memory = _masked_mean_memory(
            source_encoding,
            pooling_weights,
        )

        computation_provenance = (
            build_temporal_pooling_computation_provenance(
                source_encoding=source_encoding,
                operation_name=(
                    MASKED_MEAN_POOLING_OPERATION_NAME
                ),
                component_name=(
                    MASKED_MEAN_POOLING_COMPONENT_NAME
                ),
                component_kind=(
                    MASKED_MEAN_POOLING_COMPONENT_KIND
                ),
                architecture_metadata=(
                    self.architecture_metadata()
                ),
                configuration_fingerprint=(
                    self.configuration_fingerprint
                ),
                parameter_snapshot=(
                    parameter_snapshot
                ),
                lineage_metadata={
                    "source_encoding_identity_preserved": True,
                    "uniform_valid_timestep_weights": True,
                    "temporal_order_used": False,
                    "temporal_coordinates_consumed": False,
                    "hazard_query_consumed": False,
                    "zero_history_rows_zeroed": (
                        self.zero_history_policy
                        == TemporalPoolingZeroHistoryPolicy.ZERO
                    ),
                },
                implementation_version=(
                    MASKED_MEAN_POOLING_IMPLEMENTATION_VERSION
                ),
            )
        )

        return TemporalPoolingOutput(
            pooled_memory=pooled_memory,
            pooling_weights=pooling_weights,
            source_encoding=source_encoding,
            pooling_kind=self.pooling_kind,
            head_reduction=self.head_reduction,
            zero_history_policy=(
                self.zero_history_policy
            ),
            computation_provenance=(
                computation_provenance
            ),
            pooling_name=(
                "masked_mean_temporal_pooling"
            ),
        )

    # -------------------------------------------------------------------------
    # Representation
    # -------------------------------------------------------------------------

    def extra_repr(
        self,
    ) -> str:
        return (
            f"output_dim={self.output_dim}, "
            "num_heads=1, "
            "head_reduction=single_head, "
            f"zero_history_policy={self.zero_history_policy.value}, "
            "parameters=0, "
            "temporal_interaction=False"
        )


# =============================================================================
# Compact aliases
# =============================================================================


MaskedMeanPooling = MaskedMeanTemporalPooler
MaskedMeanTemporalPooling = MaskedMeanTemporalPooler


# =============================================================================
# Public API
# =============================================================================


__all__ = (
    "MASKED_MEAN_POOLING_COMPONENT_NAME",
    "MASKED_MEAN_POOLING_COMPONENT_KIND",
    "MASKED_MEAN_POOLING_OPERATION_NAME",
    "MASKED_MEAN_POOLING_IMPLEMENTATION_VERSION",
    "MASKED_MEAN_POOLING_TEMPORAL_INTERACTION",
    "MASKED_MEAN_POOLING_HAZARD_CONDITIONED",
    "MASKED_MEAN_POOLING_WEIGHT_POLICY",
    "MASKED_MEAN_POOLING_PADDING_POLICY",
    "MASKED_MEAN_POOLING_PROJECTION_POLICY",
    "MASKED_MEAN_POOLING_ZERO_HISTORY_POLICIES",
    "MaskedMeanTemporalPooler",
    "MaskedMeanPooling",
    "MaskedMeanTemporalPooling",
)
