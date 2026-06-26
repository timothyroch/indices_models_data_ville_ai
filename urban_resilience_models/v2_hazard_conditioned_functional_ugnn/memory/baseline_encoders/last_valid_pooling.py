"""
Deterministic last-valid temporal pooling.

Target path:
    urban_resilience_models/
        v2_hazard_conditioned_functional_ugnn/
            memory/
                baseline_encoders/
                    last_valid_pooling.py

This module selects the chronologically latest valid temporal representation
for every node.

For each node ``n`` with at least one valid timestep:

    last_index[n] =
        max { t : timestep_mask[n, t] is True }

    weight[n, 0, last_index[n]] = 1
    weight[n, 0, t] = 0 otherwise

    pooled[n, :] =
        encoded_sequence[n, last_index[n], :]

The operation consumes ``TemporalSequenceEncoding`` and returns the canonical
``TemporalPoolingOutput``. The exact source-encoding object is preserved.

Role
---------------
Last-valid pooling is a deterministic, hazard-independent, order-sensitive
reduction. It introduces:

- no learned temporal weighting;
- no trainable fallback;
- no direct use of temporal-coordinate values;
- no causal or feature-importance interpretation.

The source contract already establishes oldest-to-newest ordering on valid
positions. Therefore, the greatest valid storage index is the latest represented
temporal slot for both left- and right-padded histories.

Terminology
-----------
The implementation is called ``last_valid`` rather than ``last_observation``.
A real temporal slot remains valid even when all original features at that slot
were missing before upstream imputation. Such a slot is selected when it is
chronologically latest.

The implementation records, in execution-lineage metadata:

- the number of nonempty nodes whose selected timestep had no observed feature;
- the corresponding fraction among nonempty nodes.

These are descriptive audit signals only. The pooler does not silently skip
all-feature-missing valid timesteps.

Projection policy
-----------------
Phase 5 keeps this baseline parameter-free. ``project_output=True`` is
recognized by the shared configuration but deliberately rejected.

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
padding at execution time. Safe temporary gather indices are used for
zero-history rows, and those rows are then explicitly zeroed. The
implementation never gathers index ``-1``.
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


LAST_VALID_POOLING_COMPONENT_NAME: Final[str] = (
    "last_valid_temporal_pooler"
)

LAST_VALID_POOLING_COMPONENT_KIND: Final[str] = "last_valid"

LAST_VALID_POOLING_OPERATION_NAME: Final[str] = (
    "pool_last_valid_sequence"
)

LAST_VALID_POOLING_IMPLEMENTATION_VERSION: Final[str] = (
    BASELINE_PROVENANCE_IMPLEMENTATION_VERSION
)

LAST_VALID_POOLING_TEMPORAL_INTERACTION: Final[bool] = False

LAST_VALID_POOLING_HAZARD_CONDITIONED: Final[bool] = False

LAST_VALID_POOLING_WEIGHT_POLICY: Final[str] = (
    "unit_mass_at_greatest_valid_temporal_index"
)

LAST_VALID_POOLING_PADDING_POLICY: Final[str] = (
    "mask_derived_padding_direction_independent_selection"
)

LAST_VALID_POOLING_PROJECTION_POLICY: Final[str] = (
    "phase_five_parameter_free_no_output_projection"
)

LAST_VALID_POOLING_MISSINGNESS_POLICY: Final[str] = (
    "timestep_validity_controls_selection_feature_missingness_is_diagnostic"
)

LAST_VALID_POOLING_ZERO_HISTORY_POLICIES: Final[
    tuple[str, ...]
] = (
    TemporalPoolingZeroHistoryPolicy.ERROR.value,
    TemporalPoolingZeroHistoryPolicy.ZERO.value,
)


# =============================================================================
# Validation helpers
# =============================================================================


def _validate_last_valid_config(
    config: TemporalPoolingConfig,
) -> None:
    if not isinstance(
        config,
        TemporalPoolingConfig,
    ):
        raise TypeError(
            "config must be a TemporalPoolingConfig."
        )

    if config.kind != TemporalPoolingKind.LAST_VALID:
        raise ValueError(
            "LastValidTemporalPooler requires "
            "config.kind='last_valid'."
        )

    if config.num_heads != 1:
        raise ValueError(
            "LastValidTemporalPooler requires num_heads=1."
        )

    if (
        config.head_reduction
        != TemporalPoolingHeadReduction.SINGLE_HEAD
    ):
        raise ValueError(
            "LastValidTemporalPooler requires "
            "head_reduction='single_head'."
        )

    if config.head_reduction_weights is not None:
        raise ValueError(
            "LastValidTemporalPooler does not accept "
            "head_reduction_weights."
        )

    if config.project_output:
        raise NotImplementedError(
            "LastValidTemporalPooler is parameter-free in Phase 5. "
            "project_output=True is recognized but not implemented."
        )

    if config.score_hidden_dim is not None:
        raise ValueError(
            "LastValidTemporalPooler does not use score_hidden_dim."
        )

    if config.dropout != 0.0:
        raise ValueError(
            "LastValidTemporalPooler requires dropout=0."
        )

    if (
        config.zero_history_policy
        == TemporalPoolingZeroHistoryPolicy.LEARNED_FALLBACK
    ):
        raise NotImplementedError(
            "LastValidTemporalPooler does not implement "
            "zero_history_policy='learned_fallback'."
        )

    if (
        config.zero_history_policy.value
        not in LAST_VALID_POOLING_ZERO_HISTORY_POLICIES
    ):
        raise ValueError(
            "Unsupported last-valid zero-history policy "
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
            "Parameter-free last-valid pooling preserves the encoded "
            "feature width. config.output_dim must equal "
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


def _zero_history_mask(
    source_encoding: TemporalSequenceEncoding,
) -> torch.Tensor:
    return (
        source_encoding
        .valid_lengths
        == 0
    )


def _validate_zero_history_policy(
    source_encoding: TemporalSequenceEncoding,
    *,
    policy: TemporalPoolingZeroHistoryPolicy,
) -> None:
    zero_history = _zero_history_mask(
        source_encoding
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
            "LastValidTemporalPooler received zero-history nodes "
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
            "LastValidTemporalPooler parameter snapshots must report "
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
            "LastValidTemporalPooler parameter snapshots must report "
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
            "parameter_snapshot does not identify the current empty "
            "parameter state of LastValidTemporalPooler."
        )


# =============================================================================
# Deterministic index, weights, and gathered memory
# =============================================================================


def _last_valid_indices(
    source_encoding: TemporalSequenceEncoding,
) -> torch.Tensor:
    """
    Return greatest valid temporal indices with ``-1`` for empty histories.

    The result has shape ``[N]`` and dtype ``torch.long``.
    """

    mask = source_encoding.timestep_mask
    sequence_length = int(
        mask.shape[1]
    )

    indices = torch.arange(
        sequence_length,
        dtype=torch.long,
        device=mask.device,
    ).unsqueeze(
        0
    ).expand_as(
        mask
    )

    invalid_value = torch.full_like(
        indices,
        -1,
    )

    return torch.where(
        mask,
        indices,
        invalid_value,
    ).max(
        dim=-1
    ).values


def _safe_last_valid_indices(
    last_indices: torch.Tensor,
) -> torch.Tensor:
    """
    Replace empty-history ``-1`` indices by safe temporary gather index zero.
    """

    return last_indices.clamp_min(
        0
    )


def _last_valid_weights(
    source_encoding: TemporalSequenceEncoding,
    last_indices: torch.Tensor,
) -> torch.Tensor:
    """
    Return exact deterministic one-hot weights with shape ``[N, 1, T]``.
    """

    node_count = source_encoding.node_count
    sequence_length = source_encoding.sequence_length

    weights = torch.zeros(
        (
            node_count,
            sequence_length,
        ),
        dtype=source_encoding.dtype,
        device=source_encoding.device,
    )

    nonempty = (
        last_indices
        >= 0
    )

    if bool(
        nonempty.any().item()
    ):
        rows = torch.nonzero(
            nonempty,
            as_tuple=False,
        ).flatten()

        weights[
            rows,
            last_indices[
                rows
            ],
        ] = 1.0

    return weights.unsqueeze(
        1
    )


def _last_valid_memory(
    source_encoding: TemporalSequenceEncoding,
    last_indices: torch.Tensor,
) -> torch.Tensor:
    """
    Gather latest valid vectors without ever indexing with ``-1``.
    """

    safe_indices = _safe_last_valid_indices(
        last_indices
    )

    gather_indices = (
        safe_indices
        .view(
            -1,
            1,
            1,
        )
        .expand(
            -1,
            1,
            source_encoding.hidden_dim,
        )
    )

    pooled = (
        source_encoding
        .encoded_sequence
        .gather(
            dim=1,
            index=gather_indices,
        )
        .squeeze(
            1
        )
    )

    zero_history = (
        last_indices
        < 0
    )

    pooled = torch.where(
        zero_history.unsqueeze(
            -1
        ),
        torch.zeros_like(
            pooled
        ),
        pooled,
    )

    if not bool(
        torch.isfinite(
            pooled
        ).all().item()
    ):
        raise ValueError(
            "LastValidTemporalPooler produced nonfinite pooled "
            "memory."
        )

    return pooled


# =============================================================================
# Missingness audit metadata
# =============================================================================


def _selected_all_features_missing_statistics(
    source_encoding: TemporalSequenceEncoding,
    last_indices: torch.Tensor,
) -> tuple[int, int, float]:
    """
    Return ``(count, nonempty_count, fraction)`` for all-missing selections.

    When ``feature_observed_mask`` is absent, the count is zero and the
    fraction is zero because original observation status is unavailable.
    """

    nonempty = (
        last_indices
        >= 0
    )
    nonempty_count = int(
        nonempty.sum().item()
    )

    feature_observed_mask = (
        source_encoding
        .source_history
        .feature_observed_mask
    )

    if (
        feature_observed_mask is None
        or nonempty_count == 0
    ):
        return (
            0,
            nonempty_count,
            0.0,
        )

    rows = torch.nonzero(
        nonempty,
        as_tuple=False,
    ).flatten()

    selected_observation_mask = (
        feature_observed_mask[
            rows,
            last_indices[
                rows
            ],
        ]
    )

    selected_has_any_observed_feature = (
        selected_observation_mask
        .any(
            dim=-1
        )
    )

    all_missing_count = int(
        (
            ~selected_has_any_observed_feature
        )
        .sum()
        .item()
    )

    fraction = (
        float(
            all_missing_count
        )
        / float(
            nonempty_count
        )
    )

    return (
        all_missing_count,
        nonempty_count,
        fraction,
    )


# =============================================================================
# Last-valid pooler
# =============================================================================


class LastValidTemporalPooler(nn.Module):
    """
    Select the greatest valid temporal index for each node.

    Parameters
    ----------
    config:
        ``TemporalPoolingConfig`` with ``kind='last_valid'``,
        ``num_heads=1``, ``head_reduction='single_head'``, and no output
        projection.

    Returns
    -------
    TemporalPoolingOutput
        Canonical Phase 4 pooling contract with exact source identity and
        deterministic one-hot weights ``[N, 1, T]``.
    """

    config: TemporalPoolingConfig

    def __init__(
        self,
        config: TemporalPoolingConfig,
    ) -> None:
        super().__init__()

        _validate_last_valid_config(
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
        return TemporalPoolingKind.LAST_VALID

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
        return LAST_VALID_POOLING_TEMPORAL_INTERACTION

    @property
    def is_hazard_conditioned(
        self,
    ) -> bool:
        return LAST_VALID_POOLING_HAZARD_CONDITIONED

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
                LAST_VALID_POOLING_WEIGHT_POLICY
            ),
            "padding_policy": (
                LAST_VALID_POOLING_PADDING_POLICY
            ),
            "projection_policy": (
                LAST_VALID_POOLING_PROJECTION_POLICY
            ),
            "missingness_policy": (
                LAST_VALID_POOLING_MISSINGNESS_POLICY
            ),
            "project_output": False,
            "parameter_count": 0,
            "trainable_parameter_count": 0,
            "trainable_temporal_weights": False,
            "temporal_interaction": False,
            "hazard_conditioned": False,
            "order_sensitive_pooling": True,
            "selection_uses_timestep_mask": True,
            "feature_observed_mask_controls_selection": False,
            "source_sequence_preserved": True,
            "implementation_version": (
                LAST_VALID_POOLING_IMPLEMENTATION_VERSION
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
        Select the latest valid temporal representation for every node.
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

        last_indices = _last_valid_indices(
            source_encoding
        )
        pooling_weights = _last_valid_weights(
            source_encoding,
            last_indices,
        )
        pooled_memory = _last_valid_memory(
            source_encoding,
            last_indices,
        )

        (
            all_missing_selected_count,
            nonempty_node_count,
            all_missing_selected_fraction,
        ) = _selected_all_features_missing_statistics(
            source_encoding,
            last_indices,
        )

        computation_provenance = (
            build_temporal_pooling_computation_provenance(
                source_encoding=source_encoding,
                operation_name=(
                    LAST_VALID_POOLING_OPERATION_NAME
                ),
                component_name=(
                    LAST_VALID_POOLING_COMPONENT_NAME
                ),
                component_kind=(
                    LAST_VALID_POOLING_COMPONENT_KIND
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
                    "greatest_valid_temporal_index_selected": True,
                    "temporal_order_used": True,
                    "temporal_coordinates_consumed": False,
                    "hazard_query_consumed": False,
                    "feature_observed_mask_controls_selection": False,
                    "selected_all_features_missing_count": (
                        all_missing_selected_count
                    ),
                    "nonempty_node_count": (
                        nonempty_node_count
                    ),
                    "selected_all_features_missing_fraction": (
                        all_missing_selected_fraction
                    ),
                    "zero_history_rows_zeroed": (
                        self.zero_history_policy
                        == TemporalPoolingZeroHistoryPolicy.ZERO
                    ),
                },
                implementation_version=(
                    LAST_VALID_POOLING_IMPLEMENTATION_VERSION
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
                "last_valid_temporal_pooling"
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
            "order_sensitive=True, "
            "temporal_interaction=False"
        )


# =============================================================================
# Compact aliases
# =============================================================================


LastValidPooling = LastValidTemporalPooler
LastValidTemporalPooling = LastValidTemporalPooler
LastObservationTemporalPooler = LastValidTemporalPooler


# =============================================================================
# Public API
# =============================================================================


__all__ = (
    "LAST_VALID_POOLING_COMPONENT_NAME",
    "LAST_VALID_POOLING_COMPONENT_KIND",
    "LAST_VALID_POOLING_OPERATION_NAME",
    "LAST_VALID_POOLING_IMPLEMENTATION_VERSION",
    "LAST_VALID_POOLING_TEMPORAL_INTERACTION",
    "LAST_VALID_POOLING_HAZARD_CONDITIONED",
    "LAST_VALID_POOLING_WEIGHT_POLICY",
    "LAST_VALID_POOLING_PADDING_POLICY",
    "LAST_VALID_POOLING_PROJECTION_POLICY",
    "LAST_VALID_POOLING_MISSINGNESS_POLICY",
    "LAST_VALID_POOLING_ZERO_HISTORY_POLICIES",
    "LastValidTemporalPooler",
    "LastValidPooling",
    "LastValidTemporalPooling",
    "LastObservationTemporalPooler",
)
