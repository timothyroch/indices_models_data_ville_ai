"""
Unit tests for the V2 schema layer.

Target path:
    urban_resilience_models/
        v2_hazard_conditioned_functional_ugnn/
            tests/
                test_schemas.py

The tests focus on contract behavior rather than model behavior:

- graph and node membership;
- temporal causality;
- optional supervision;
- heterogeneous feature blocks;
- registry-membership hooks;
- relation-gate alignment;
- attention normalization;
- prediction and explanation alignment;
- uncertainty validation;
- controlled device transfer and replacement.

These tests assume that the project root is available on PYTHONPATH.
"""

from __future__ import annotations

from datetime import datetime, timedelta

import pytest
import torch
from torch import Tensor

from urban_resilience_models.v2_hazard_conditioned_functional_ugnn.constants import (
    ATTENTION_HEAD_REDUCTION_NONE,
    ATTENTION_NORMALIZATION_TARGET_NODE,
    ATTENTION_NORMALIZATION_TARGET_NODE_RELATION,
    ATTENTION_NORMALIZATION_UNNORMALIZED_SIGMOID,
    MODEL_FAMILY_ID,
    RELATION_GATE_SCOPE_GRAPH,
    RELATION_GATE_SCOPE_SOURCE_TARGET,
    RELATION_GATE_SCOPE_TARGET_NODE,
    RELATION_REGISTRY_VERSION,
    SCOPE_GRAPH,
)
from urban_resilience_models.v2_hazard_conditioned_functional_ugnn.schemas import (
    BatchMetadata,
    ContractVersions,
    EdgeAttentionOutput,
    ExplanationTrace,
    IntermediateStates,
    ModelOutput,
    PredictionAlignment,
    PredictionOutput,
    RelationGateOutput,
    RunMetadata,
    TemporalMetadata,
    TypedNodeFeatureBlock,
    UncertaintyOutput,
    UrbanGraphBatch,
)


# =============================================================================
# Test factories
# =============================================================================


def _target_names(num_targets: int) -> tuple[str, ...]:
    return tuple(
        f"future_burden_{index + 1}"
        for index in range(num_targets)
    )


def _target_horizons(num_targets: int) -> tuple[str, ...]:
    return tuple(
        f"next_{index + 1}_month"
        for index in range(num_targets)
    )


def _temporal_metadata(
    *,
    batch_size: int,
    num_targets: int,
    history_length: int = 3,
    supervised: bool,
    origin_shift_days: int = 0,
) -> TemporalMetadata:
    origins = tuple(
        datetime(2025, 4, 1)
        + timedelta(days=origin_shift_days + graph_index)
        for graph_index in range(batch_size)
    )

    history_starts = tuple(
        origin - timedelta(days=90)
        for origin in origins
    )
    history_ends = tuple(
        origin - timedelta(days=1)
        for origin in origins
    )
    availability_cutoffs = origins

    history_time_points = tuple(
        tuple(
            history_starts[graph_index]
            + timedelta(days=30 * time_index)
            for time_index in range(history_length)
        )
        for graph_index in range(batch_size)
    )

    target_starts = None
    target_ends = None

    if supervised:
        target_starts = tuple(
            tuple(
                origins[graph_index]
                + timedelta(days=30 * (target_index + 1))
                for target_index in range(num_targets)
            )
            for graph_index in range(batch_size)
        )

        target_ends = tuple(
            tuple(
                target_starts[graph_index][target_index]
                + timedelta(days=29)
                for target_index in range(num_targets)
            )
            for graph_index in range(batch_size)
        )

    return TemporalMetadata(
        origin_time=origins,
        history_start_time=history_starts,
        history_end_time=history_ends,
        feature_availability_cutoff=availability_cutoffs,
        history_time_points_by_graph=history_time_points,
        target_start_time_by_graph_target=target_starts,
        target_end_time_by_graph_target=target_ends,
    )


def _batch_metadata(
    *,
    batch_size: int,
    supervised: bool,
) -> BatchMetadata:
    split = "train" if supervised else "inference"

    return BatchMetadata(
        dataset_version="test-dataset-v1",
        graph_version="test-graph-v1",
        split_by_graph=(split,) * batch_size,
        experiment_name="schema_unit_test",
    )


def _single_graph_batch_kwargs(
    *,
    supervised: bool = False,
    num_targets: int | None = None,
) -> dict[str, object]:
    if num_targets is None:
        num_targets = 2 if supervised else 1

    num_nodes = 3
    history_length = 3

    kwargs: dict[str, object] = {
        "external_node_ids": (
            "node-001",
            "node-002",
            "node-003",
        ),
        "node_batch_index": torch.zeros(
            num_nodes,
            dtype=torch.long,
        ),
        "hazard_ids": torch.tensor(
            [0],
            dtype=torch.long,
        ),
        "hazard_scope": SCOPE_GRAPH,
        "edge_index": torch.tensor(
            [
                [0, 1],
                [1, 2],
            ],
            dtype=torch.long,
        ),
        "edge_relation_type": torch.tensor(
            [0, 0],
            dtype=torch.long,
        ),
        "temporal": _temporal_metadata(
            batch_size=1,
            num_targets=num_targets,
            history_length=history_length,
            supervised=supervised,
        ),
        "contract_versions": ContractVersions.current(),
        "metadata": _batch_metadata(
            batch_size=1,
            supervised=supervised,
        ),
        "target_names": _target_names(num_targets),
        "target_horizons": _target_horizons(num_targets),
        "node_features": torch.tensor(
            [
                [1.0, 10.0],
                [2.0, 20.0],
                [3.0, 30.0],
            ],
            dtype=torch.float32,
        ),
        "history_sequences": torch.arange(
            num_nodes * history_length * 2,
            dtype=torch.float32,
        ).reshape(num_nodes, history_length, 2),
        "history_mask": torch.ones(
            (num_nodes, history_length),
            dtype=torch.bool,
        ),
        "feature_names": (
            "population_density",
            "vulnerability_score",
        ),
        "history_feature_names": (
            "past_burden",
            "past_event_count",
        ),
    }

    if supervised:
        kwargs["targets"] = torch.tensor(
            [
                [1.0, 2.0],
                [2.0, 3.0],
                [3.0, 4.0],
            ],
            dtype=torch.float32,
        )[:, :num_targets]

        kwargs["target_mask"] = torch.ones(
            (num_nodes, num_targets),
            dtype=torch.bool,
        )

    return kwargs


def _two_graph_batch_kwargs(
    *,
    cross_graph_edge: bool = False,
) -> dict[str, object]:
    num_nodes = 4
    num_targets = 1
    history_length = 3

    edge_index = (
        torch.tensor(
            [
                [0],
                [2],
            ],
            dtype=torch.long,
        )
        if cross_graph_edge
        else torch.tensor(
            [
                [0, 2],
                [1, 3],
            ],
            dtype=torch.long,
        )
    )

    num_edges = edge_index.shape[1]

    return {
        "external_node_ids": (
            "graph-0-node-0",
            "graph-0-node-1",
            "graph-1-node-0",
            "graph-1-node-1",
        ),
        "node_batch_index": torch.tensor(
            [0, 0, 1, 1],
            dtype=torch.long,
        ),
        "hazard_ids": torch.tensor(
            [0, 1],
            dtype=torch.long,
        ),
        "hazard_scope": SCOPE_GRAPH,
        "edge_index": edge_index,
        "edge_relation_type": torch.zeros(
            num_edges,
            dtype=torch.long,
        ),
        "temporal": _temporal_metadata(
            batch_size=2,
            num_targets=num_targets,
            history_length=history_length,
            supervised=False,
        ),
        "contract_versions": ContractVersions.current(),
        "metadata": _batch_metadata(
            batch_size=2,
            supervised=False,
        ),
        "target_names": _target_names(num_targets),
        "target_horizons": _target_horizons(num_targets),
        "node_features": torch.ones(
            (num_nodes, 2),
            dtype=torch.float32,
        ),
        "history_sequences": torch.ones(
            (num_nodes, history_length, 2),
            dtype=torch.float32,
        ),
        "history_mask": torch.ones(
            (num_nodes, history_length),
            dtype=torch.bool,
        ),
    }


def _run_metadata(
    *,
    run_id: str = "run-001",
) -> RunMetadata:
    return RunMetadata(
        checkpoint_id="checkpoint-001",
        run_id=run_id,
        experiment_name="schema_unit_test",
        random_seed=42,
        dataset_version="test-dataset-v1",
        graph_version="test-graph-v1",
        config_hash="abc123",
        model_family_id=MODEL_FAMILY_ID,
    )


def _prediction_alignment(
    *,
    num_nodes: int = 3,
    num_targets: int = 2,
    device: torch.device | str = "cpu",
    run_id: str = "run-001",
    hazard_id: int = 0,
) -> PredictionAlignment:
    origin = datetime(2025, 4, 1)

    target_starts = (
        tuple(
            origin + timedelta(days=30 * (index + 1))
            for index in range(num_targets)
        ),
    )
    target_ends = (
        tuple(
            target_starts[0][index] + timedelta(days=29)
            for index in range(num_targets)
        ),
    )

    return PredictionAlignment(
        external_node_ids=tuple(
            f"node-{index:03d}"
            for index in range(num_nodes)
        ),
        node_batch_index=torch.zeros(
            num_nodes,
            dtype=torch.long,
            device=device,
        ),
        hazard_ids=torch.tensor(
            [hazard_id],
            dtype=torch.long,
            device=device,
        ),
        hazard_scope=SCOPE_GRAPH,
        origin_time=(origin,),
        target_names=_target_names(num_targets),
        target_horizons=_target_horizons(num_targets),
        run_metadata=_run_metadata(run_id=run_id),
        contract_versions=ContractVersions.current(),
        target_start_time_by_graph_target=target_starts,
        target_end_time_by_graph_target=target_ends,
    )


# =============================================================================
# UrbanGraphBatch construction
# =============================================================================


class TestUrbanGraphBatch:
    def test_valid_inference_batch(self) -> None:
        batch = UrbanGraphBatch(
            **_single_graph_batch_kwargs(supervised=False)
        )

        assert batch.num_nodes == 3
        assert batch.num_edges == 2
        assert batch.batch_size == 1
        assert batch.num_targets == 1
        assert batch.is_supervised is False
        assert batch.targets is None
        assert batch.target_mask is None

    def test_valid_supervised_batch(self) -> None:
        batch = UrbanGraphBatch(
            **_single_graph_batch_kwargs(supervised=True)
        )

        assert batch.is_supervised is True
        assert batch.targets is not None
        assert batch.target_mask is not None
        assert batch.targets.shape == (3, 2)
        assert batch.target_mask.shape == (3, 2)
        assert batch.num_targets == 2

    def test_inference_batch_does_not_require_targets(self) -> None:
        kwargs = _single_graph_batch_kwargs(supervised=False)

        kwargs["targets"] = None
        kwargs["target_mask"] = None

        batch = UrbanGraphBatch(**kwargs)

        assert batch.targets is None
        assert batch.target_mask is None

    def test_target_mask_without_targets_is_rejected(self) -> None:
        kwargs = _single_graph_batch_kwargs(supervised=False)
        kwargs["target_mask"] = torch.ones(
            (3, 1),
            dtype=torch.bool,
        )

        with pytest.raises(
            ValueError,
            match="target_mask cannot be provided without targets",
        ):
            UrbanGraphBatch(**kwargs)

    def test_targets_require_target_names(self) -> None:
        kwargs = _single_graph_batch_kwargs(supervised=True)
        kwargs["target_names"] = ()
        kwargs["target_horizons"] = ()

        with pytest.raises(
            ValueError,
            match="target_names are required",
        ):
            UrbanGraphBatch(**kwargs)

    def test_duplicate_target_names_are_rejected(self) -> None:
        kwargs = _single_graph_batch_kwargs(supervised=True)
        kwargs["target_names"] = (
            "future_burden",
            "future_burden",
        )

        with pytest.raises(
            ValueError,
            match="duplicate values",
        ):
            UrbanGraphBatch(**kwargs)

    def test_invalid_edge_index_rank_is_rejected_descriptively(
        self,
    ) -> None:
        kwargs = _single_graph_batch_kwargs(supervised=False)
        kwargs["edge_index"] = torch.tensor(
            [0, 1],
            dtype=torch.long,
        )
        kwargs["edge_relation_type"] = torch.tensor(
            [0],
            dtype=torch.long,
        )

        with pytest.raises(
            ValueError,
            match="edge_index must have rank 2",
        ):
            UrbanGraphBatch(**kwargs)

    def test_invalid_edge_index_first_dimension_is_rejected(
        self,
    ) -> None:
        kwargs = _single_graph_batch_kwargs(supervised=False)
        kwargs["edge_index"] = torch.tensor(
            [
                [0, 1],
                [1, 2],
                [2, 0],
            ],
            dtype=torch.long,
        )

        with pytest.raises(
            ValueError,
            match=r"edge_index must have shape \[2, E\]",
        ):
            UrbanGraphBatch(**kwargs)

    def test_out_of_range_edge_endpoint_is_rejected(self) -> None:
        kwargs = _single_graph_batch_kwargs(supervised=False)
        kwargs["edge_index"] = torch.tensor(
            [
                [0, 1],
                [1, 99],
            ],
            dtype=torch.long,
        )

        with pytest.raises(
            ValueError,
            match="invalid node index",
        ):
            UrbanGraphBatch(**kwargs)

    def test_cross_graph_edge_is_rejected_by_default(self) -> None:
        with pytest.raises(
            ValueError,
            match="Cross-graph edge detected",
        ):
            UrbanGraphBatch(
                **_two_graph_batch_kwargs(
                    cross_graph_edge=True
                )
            )

    def test_valid_multi_graph_batch(self) -> None:
        batch = UrbanGraphBatch(
            **_two_graph_batch_kwargs(
                cross_graph_edge=False
            )
        )

        assert batch.batch_size == 2
        assert batch.num_nodes == 4
        assert batch.num_edges == 2
        assert torch.equal(
            batch.derived_edge_batch_index(),
            torch.tensor([0, 1], dtype=torch.long),
        )

    def test_graph_ptr_must_match_node_batch_index(self) -> None:
        kwargs = _two_graph_batch_kwargs()
        kwargs["graph_ptr"] = torch.tensor(
            [0, 3, 4],
            dtype=torch.long,
        )

        with pytest.raises(
            ValueError,
            match="graph_ptr is inconsistent",
        ):
            UrbanGraphBatch(**kwargs)

    def test_valid_graph_ptr(self) -> None:
        kwargs = _two_graph_batch_kwargs()
        kwargs["graph_ptr"] = torch.tensor(
            [0, 2, 4],
            dtype=torch.long,
        )

        batch = UrbanGraphBatch(**kwargs)

        assert torch.equal(
            batch.graph_ptr,
            torch.tensor([0, 2, 4], dtype=torch.long),
        )

    def test_zero_edge_graph_is_valid(self) -> None:
        kwargs = _single_graph_batch_kwargs(supervised=False)
        kwargs["edge_index"] = torch.empty(
            (2, 0),
            dtype=torch.long,
        )
        kwargs["edge_relation_type"] = torch.empty(
            (0,),
            dtype=torch.long,
        )

        batch = UrbanGraphBatch(**kwargs)

        assert batch.num_edges == 0
        assert batch.source_index.numel() == 0
        assert batch.target_index.numel() == 0

    def test_duplicate_external_edge_ids_are_rejected(self) -> None:
        kwargs = _single_graph_batch_kwargs(supervised=False)
        kwargs["external_edge_ids"] = (
            "edge-001",
            "edge-001",
        )

        with pytest.raises(
            ValueError,
            match="Duplicate external edge ID",
        ):
            UrbanGraphBatch(**kwargs)

    def test_duplicate_feature_names_are_rejected(self) -> None:
        kwargs = _single_graph_batch_kwargs(supervised=False)
        kwargs["feature_names"] = (
            "population",
            "population",
        )

        with pytest.raises(
            ValueError,
            match="duplicate values",
        ):
            UrbanGraphBatch(**kwargs)


# =============================================================================
# Temporal causality
# =============================================================================


class TestTemporalMetadata:
    def test_future_history_is_rejected(self) -> None:
        kwargs = _single_graph_batch_kwargs(supervised=False)

        temporal = _temporal_metadata(
            batch_size=1,
            num_targets=1,
            history_length=3,
            supervised=False,
        )

        temporal.history_end_time = (
            datetime(2025, 4, 2),
        )
        kwargs["temporal"] = temporal

        with pytest.raises(
            ValueError,
            match="history_end_time must not exceed origin_time",
        ):
            UrbanGraphBatch(**kwargs)

    def test_future_feature_cutoff_is_rejected(self) -> None:
        kwargs = _single_graph_batch_kwargs(supervised=False)

        temporal = _temporal_metadata(
            batch_size=1,
            num_targets=1,
            history_length=3,
            supervised=False,
        )
        temporal.feature_availability_cutoff = (
            datetime(2025, 4, 2),
        )
        kwargs["temporal"] = temporal

        with pytest.raises(
            ValueError,
            match="feature_availability_cutoff must not exceed",
        ):
            UrbanGraphBatch(**kwargs)

    def test_target_start_at_origin_is_rejected(self) -> None:
        kwargs = _single_graph_batch_kwargs(supervised=True)

        temporal = _temporal_metadata(
            batch_size=1,
            num_targets=2,
            history_length=3,
            supervised=True,
        )

        temporal.target_start_time_by_graph_target = (
            (
                datetime(2025, 4, 1),
                datetime(2025, 6, 1),
            ),
        )
        kwargs["temporal"] = temporal

        with pytest.raises(
            ValueError,
            match="target_start_time must be strictly after",
        ):
            UrbanGraphBatch(**kwargs)

    def test_multi_horizon_target_windows_are_valid(self) -> None:
        batch = UrbanGraphBatch(
            **_single_graph_batch_kwargs(
                supervised=True,
                num_targets=2,
            )
        )

        starts = (
            batch.temporal.target_start_time_by_graph_target
        )
        ends = (
            batch.temporal.target_end_time_by_graph_target
        )

        assert starts is not None
        assert ends is not None
        assert len(starts) == 1
        assert len(starts[0]) == 2
        assert len(ends[0]) == 2

    def test_wrong_target_window_width_is_rejected(self) -> None:
        kwargs = _single_graph_batch_kwargs(supervised=True)

        temporal = _temporal_metadata(
            batch_size=1,
            num_targets=2,
            history_length=3,
            supervised=True,
        )
        temporal.target_start_time_by_graph_target = (
            (datetime(2025, 5, 1),),
        )
        kwargs["temporal"] = temporal

        with pytest.raises(
            ValueError,
            match="must contain 2 target values",
        ):
            UrbanGraphBatch(**kwargs)

    def test_edge_observed_after_availability_cutoff_is_rejected(
        self,
    ) -> None:
        kwargs = _single_graph_batch_kwargs(supervised=False)

        temporal = _temporal_metadata(
            batch_size=1,
            num_targets=1,
            history_length=3,
            supervised=False,
        )
        temporal.edge_observation_time = (
            datetime(2025, 4, 2),
            datetime(2025, 3, 1),
        )
        kwargs["temporal"] = temporal

        with pytest.raises(
            ValueError,
            match="after the feature-availability cutoff",
        ):
            UrbanGraphBatch(**kwargs)

    def test_invalid_edge_validity_interval_is_rejected(
        self,
    ) -> None:
        kwargs = _single_graph_batch_kwargs(supervised=False)

        temporal = _temporal_metadata(
            batch_size=1,
            num_targets=1,
            history_length=3,
            supervised=False,
        )
        temporal.edge_valid_from = (
            datetime(2025, 3, 15),
            datetime(2025, 3, 1),
        )
        temporal.edge_valid_to = (
            datetime(2025, 3, 1),
            datetime(2025, 5, 1),
        )
        kwargs["temporal"] = temporal

        with pytest.raises(
            ValueError,
            match="edge_valid_from must not exceed edge_valid_to",
        ):
            UrbanGraphBatch(**kwargs)


# =============================================================================
# Heterogeneous feature blocks
# =============================================================================


class TestTypedNodeFeatureBlocks:
    def test_valid_typed_feature_blocks(self) -> None:
        kwargs = _single_graph_batch_kwargs(supervised=False)

        node_type = torch.tensor(
            [0, 0, 1],
            dtype=torch.long,
        )

        tract_block = TypedNodeFeatureBlock(
            node_type_name="census_tract",
            node_type_id=0,
            internal_node_index=torch.tensor(
                [0, 1],
                dtype=torch.long,
            ),
            features=torch.tensor(
                [
                    [1.0, 2.0],
                    [3.0, 4.0],
                ],
                dtype=torch.float32,
            ),
            feature_names=(
                "population",
                "density",
            ),
        )

        drainage_block = TypedNodeFeatureBlock(
            node_type_name="drainage_asset",
            node_type_id=1,
            internal_node_index=torch.tensor(
                [2],
                dtype=torch.long,
            ),
            features=torch.tensor(
                [[5.0, 6.0, 7.0]],
                dtype=torch.float32,
            ),
            feature_names=(
                "capacity",
                "age",
                "condition",
            ),
        )

        kwargs["node_features"] = None
        kwargs["node_type"] = node_type
        kwargs["node_feature_blocks"] = (
            tract_block,
            drainage_block,
        )
        kwargs["feature_names"] = ()

        batch = UrbanGraphBatch(**kwargs)

        assert batch.node_features is None
        assert len(batch.node_feature_blocks) == 2
        assert torch.equal(batch.node_type, node_type)

    def test_feature_block_must_match_packed_node_type(
        self,
    ) -> None:
        kwargs = _single_graph_batch_kwargs(supervised=False)

        kwargs["node_features"] = None
        kwargs["feature_names"] = ()
        kwargs["node_type"] = torch.tensor(
            [0, 0, 1],
            dtype=torch.long,
        )
        kwargs["node_feature_blocks"] = (
            TypedNodeFeatureBlock(
                node_type_name="drainage_asset",
                node_type_id=1,
                internal_node_index=torch.tensor(
                    [0, 1, 2],
                    dtype=torch.long,
                ),
                features=torch.ones(
                    (3, 2),
                    dtype=torch.float32,
                ),
            ),
        )

        with pytest.raises(
            ValueError,
            match="does not match the packed node_type tensor",
        ):
            UrbanGraphBatch(**kwargs)

    def test_feature_blocks_must_cover_every_node_once(
        self,
    ) -> None:
        kwargs = _single_graph_batch_kwargs(supervised=False)

        kwargs["node_features"] = None
        kwargs["feature_names"] = ()
        kwargs["node_type"] = torch.tensor(
            [0, 0, 1],
            dtype=torch.long,
        )
        kwargs["node_feature_blocks"] = (
            TypedNodeFeatureBlock(
                node_type_name="census_tract",
                node_type_id=0,
                internal_node_index=torch.tensor(
                    [0, 1],
                    dtype=torch.long,
                ),
                features=torch.ones(
                    (2, 2),
                    dtype=torch.float32,
                ),
            ),
        )

        with pytest.raises(
            ValueError,
            match="collectively cover all nodes",
        ):
            UrbanGraphBatch(**kwargs)


# =============================================================================
# Registry membership boundary
# =============================================================================


class TestRegistryMembership:
    def test_valid_registry_membership(self) -> None:
        batch = UrbanGraphBatch(
            **_single_graph_batch_kwargs(supervised=False)
        )

        batch.validate_registry_membership(
            valid_hazard_ids={0, 1},
            valid_relation_ids={0, 1, 2},
        )

    def test_unknown_hazard_id_is_rejected(self) -> None:
        batch = UrbanGraphBatch(
            **_single_graph_batch_kwargs(supervised=False)
        )

        with pytest.raises(
            ValueError,
            match="Unknown hazard IDs",
        ):
            batch.validate_registry_membership(
                valid_hazard_ids={1, 2},
                valid_relation_ids={0},
            )

    def test_unknown_relation_id_is_rejected(self) -> None:
        batch = UrbanGraphBatch(
            **_single_graph_batch_kwargs(supervised=False)
        )

        with pytest.raises(
            ValueError,
            match="Unknown relation IDs",
        ):
            batch.validate_registry_membership(
                valid_hazard_ids={0},
                valid_relation_ids={1, 2},
            )

    def test_unknown_node_type_id_is_rejected(self) -> None:
        kwargs = _single_graph_batch_kwargs(supervised=False)
        kwargs["node_type"] = torch.tensor(
            [0, 0, 5],
            dtype=torch.long,
        )

        batch = UrbanGraphBatch(**kwargs)

        with pytest.raises(
            ValueError,
            match="Unknown node-type IDs",
        ):
            batch.validate_registry_membership(
                valid_hazard_ids={0},
                valid_relation_ids={0},
                valid_node_type_ids={0, 1},
            )


# =============================================================================
# Relation gates
# =============================================================================


class TestRelationGateOutput:
    @pytest.mark.parametrize(
        ("scope", "shape"),
        (
            (RELATION_GATE_SCOPE_GRAPH, (1, 3)),
            (RELATION_GATE_SCOPE_TARGET_NODE, (4, 3)),
            (RELATION_GATE_SCOPE_SOURCE_TARGET, (5, 3)),
        ),
    )
    def test_scope_appropriate_gate_shape(
        self,
        scope: str,
        shape: tuple[int, int],
    ) -> None:
        output = RelationGateOutput(
            gate_values=torch.ones(
                shape,
                dtype=torch.float32,
            ),
            scope=scope,
            relation_registry_version=RELATION_REGISTRY_VERSION,
        )

        output.validate_alignment(
            num_graphs=1,
            num_nodes=4,
            num_edges=5,
            num_relations=3,
        )

    def test_incorrect_gate_shape_is_rejected(self) -> None:
        output = RelationGateOutput(
            gate_values=torch.ones(
                (4, 3),
                dtype=torch.float32,
            ),
            scope=RELATION_GATE_SCOPE_GRAPH,
            relation_registry_version=RELATION_REGISTRY_VERSION,
        )

        with pytest.raises(
            ValueError,
            match="requires shape",
        ):
            output.validate_alignment(
                num_graphs=1,
                num_nodes=4,
                num_edges=5,
                num_relations=3,
            )

    def test_negative_gate_values_are_rejected(self) -> None:
        with pytest.raises(
            ValueError,
            match="nonnegative",
        ):
            RelationGateOutput(
                gate_values=torch.tensor(
                    [[0.5, -0.1]],
                    dtype=torch.float32,
                ),
                scope=RELATION_GATE_SCOPE_GRAPH,
                relation_registry_version=(
                    RELATION_REGISTRY_VERSION
                ),
            )


# =============================================================================
# Attention outputs
# =============================================================================


class TestEdgeAttentionOutput:
    def test_target_node_normalization(self) -> None:
        edge_index = torch.tensor(
            [
                [0, 2, 1],
                [1, 1, 2],
            ],
            dtype=torch.long,
        )
        edge_relation_type = torch.tensor(
            [0, 0, 0],
            dtype=torch.long,
        )

        output = EdgeAttentionOutput(
            attention_weight=torch.tensor(
                [0.25, 0.75, 1.0],
                dtype=torch.float32,
            ),
            normalization_scope=(
                ATTENTION_NORMALIZATION_TARGET_NODE
            ),
            head_reduction_policy=(
                ATTENTION_HEAD_REDUCTION_NONE
            ),
        )

        output.validate_normalization(
            edge_index=edge_index,
            edge_relation_type=edge_relation_type,
        )

    def test_invalid_target_node_normalization_is_rejected(
        self,
    ) -> None:
        edge_index = torch.tensor(
            [
                [0, 2, 1],
                [1, 1, 2],
            ],
            dtype=torch.long,
        )
        edge_relation_type = torch.tensor(
            [0, 0, 0],
            dtype=torch.long,
        )

        output = EdgeAttentionOutput(
            attention_weight=torch.tensor(
                [0.20, 0.20, 1.0],
                dtype=torch.float32,
            ),
            normalization_scope=(
                ATTENTION_NORMALIZATION_TARGET_NODE
            ),
            head_reduction_policy=(
                ATTENTION_HEAD_REDUCTION_NONE
            ),
        )

        with pytest.raises(
            ValueError,
            match="do not sum to one",
        ):
            output.validate_normalization(
                edge_index=edge_index,
                edge_relation_type=edge_relation_type,
            )

    def test_target_node_relation_normalization(self) -> None:
        edge_index = torch.tensor(
            [
                [0, 2, 3, 4],
                [1, 1, 1, 1],
            ],
            dtype=torch.long,
        )
        edge_relation_type = torch.tensor(
            [0, 0, 1, 1],
            dtype=torch.long,
        )

        output = EdgeAttentionOutput(
            attention_weight=torch.tensor(
                [0.4, 0.6, 0.3, 0.7],
                dtype=torch.float32,
            ),
            normalization_scope=(
                ATTENTION_NORMALIZATION_TARGET_NODE_RELATION
            ),
            head_reduction_policy=(
                ATTENTION_HEAD_REDUCTION_NONE
            ),
        )

        output.validate_normalization(
            edge_index=edge_index,
            edge_relation_type=edge_relation_type,
        )

    def test_unnormalized_sigmoid_attention_is_exempt(
        self,
    ) -> None:
        output = EdgeAttentionOutput(
            attention_weight=torch.tensor(
                [0.2, 0.4, 0.7],
                dtype=torch.float32,
            ),
            normalization_scope=(
                ATTENTION_NORMALIZATION_UNNORMALIZED_SIGMOID
            ),
            head_reduction_policy=(
                ATTENTION_HEAD_REDUCTION_NONE
            ),
        )

        output.validate_normalization(
            edge_index=torch.tensor(
                [
                    [0, 1, 2],
                    [1, 2, 0],
                ],
                dtype=torch.long,
            ),
            edge_relation_type=torch.tensor(
                [0, 0, 0],
                dtype=torch.long,
            ),
        )


# =============================================================================
# Prediction alignment
# =============================================================================


class TestPredictionAlignment:
    def test_equivalent_alignments_pass(self) -> None:
        first = _prediction_alignment()
        second = _prediction_alignment()

        first.assert_equivalent(second)

    def test_different_hazards_are_not_equivalent(self) -> None:
        first = _prediction_alignment(hazard_id=0)
        second = _prediction_alignment(hazard_id=1)

        with pytest.raises(
            ValueError,
            match="hazard_ids",
        ):
            first.assert_equivalent(second)

    def test_different_run_ids_are_not_equivalent(self) -> None:
        first = _prediction_alignment(run_id="run-001")
        second = _prediction_alignment(run_id="run-002")

        with pytest.raises(
            ValueError,
            match="run_metadata",
        ):
            first.assert_equivalent(second)

    def test_incompatible_model_family_is_rejected(self) -> None:
        metadata = RunMetadata(
            checkpoint_id="checkpoint-001",
            run_id="run-001",
            experiment_name="schema_unit_test",
            random_seed=42,
            dataset_version="test-dataset-v1",
            graph_version="test-graph-v1",
            config_hash="abc123",
            model_family_id="another_model_family",
        )

        with pytest.raises(
            ValueError,
            match="incompatible model family",
        ):
            metadata.validate()


# =============================================================================
# Explanation traces
# =============================================================================


class TestExplanationTrace:
    def test_temporal_attention_requires_time_points(
        self,
    ) -> None:
        alignment = _prediction_alignment(
            num_nodes=3,
            num_targets=1,
        )

        trace = ExplanationTrace(
            alignment=alignment,
            temporal_attention=torch.full(
                (3, 3),
                1.0 / 3.0,
                dtype=torch.float32,
            ),
        )

        with pytest.raises(
            ValueError,
            match="history_time_points_by_graph is required",
        ):
            trace.validate(
                num_nodes=3,
                num_targets=1,
            )

    def test_valid_temporal_attention_trace(self) -> None:
        alignment = _prediction_alignment(
            num_nodes=3,
            num_targets=1,
        )

        trace = ExplanationTrace(
            alignment=alignment,
            temporal_attention=torch.full(
                (3, 3),
                1.0 / 3.0,
                dtype=torch.float32,
            ),
            history_time_points_by_graph=(
                (
                    datetime(2025, 1, 1),
                    datetime(2025, 2, 1),
                    datetime(2025, 3, 1),
                ),
            ),
        )

        trace.validate(
            num_nodes=3,
            num_targets=1,
        )

    def test_edge_attention_requires_relation_types(
        self,
    ) -> None:
        alignment = _prediction_alignment(
            num_nodes=3,
            num_targets=1,
        )

        attention = EdgeAttentionOutput(
            attention_weight=torch.tensor(
                [1.0],
                dtype=torch.float32,
            ),
            normalization_scope=(
                ATTENTION_NORMALIZATION_TARGET_NODE
            ),
            head_reduction_policy=(
                ATTENTION_HEAD_REDUCTION_NONE
            ),
        )

        trace = ExplanationTrace(
            alignment=alignment,
            edge_attention_output=attention,
            edge_index=torch.tensor(
                [
                    [0],
                    [1],
                ],
                dtype=torch.long,
            ),
        )

        with pytest.raises(
            ValueError,
            match="edge_relation_type is required",
        ):
            trace.validate(
                num_nodes=3,
                num_targets=1,
            )


# =============================================================================
# Uncertainty validation
# =============================================================================


class TestUncertaintyOutput:
    def test_valid_quantiles(self) -> None:
        quantiles = torch.tensor(
            [
                [[1.0, 2.0, 3.0]],
                [[2.0, 3.0, 4.0]],
                [[3.0, 4.0, 5.0]],
            ],
            dtype=torch.float32,
        )

        output = UncertaintyOutput(
            method="quantile",
            quantiles=quantiles,
            quantile_levels=(0.1, 0.5, 0.9),
        )

        output.validate(
            num_nodes=3,
            num_targets=1,
        )

    @pytest.mark.parametrize(
        "levels",
        (
            (0.0, 0.5, 0.9),
            (0.1, 0.9, 1.0),
            (-0.1, 0.5, 0.9),
        ),
    )
    def test_invalid_quantile_levels_are_rejected(
        self,
        levels: tuple[float, ...],
    ) -> None:
        output = UncertaintyOutput(
            method="quantile",
            quantiles=torch.ones(
                (3, 1, 3),
                dtype=torch.float32,
            ),
            quantile_levels=levels,
        )

        with pytest.raises(
            ValueError,
            match="strictly between 0 and 1",
        ):
            output.validate(
                num_nodes=3,
                num_targets=1,
            )

    def test_unsorted_quantile_levels_are_rejected(self) -> None:
        output = UncertaintyOutput(
            method="quantile",
            quantiles=torch.ones(
                (3, 1, 3),
                dtype=torch.float32,
            ),
            quantile_levels=(0.5, 0.1, 0.9),
        )

        with pytest.raises(
            ValueError,
            match="strictly increasing",
        ):
            output.validate(
                num_nodes=3,
                num_targets=1,
            )

    def test_crossing_quantile_predictions_are_rejected(
        self,
    ) -> None:
        output = UncertaintyOutput(
            method="quantile",
            quantiles=torch.tensor(
                [
                    [[1.0, 0.5, 2.0]],
                    [[1.0, 2.0, 3.0]],
                    [[2.0, 3.0, 4.0]],
                ],
                dtype=torch.float32,
            ),
            quantile_levels=(0.1, 0.5, 0.9),
        )

        with pytest.raises(
            ValueError,
            match="nondecreasing",
        ):
            output.validate(
                num_nodes=3,
                num_targets=1,
            )

    def test_invalid_interval_order_is_rejected(self) -> None:
        output = UncertaintyOutput(
            method="interval",
            lower_bound=torch.tensor(
                [[2.0], [1.0], [1.0]],
                dtype=torch.float32,
            ),
            upper_bound=torch.tensor(
                [[1.0], [2.0], [2.0]],
                dtype=torch.float32,
            ),
        )

        with pytest.raises(
            ValueError,
            match="lower_bound cannot exceed upper_bound",
        ):
            output.validate(
                num_nodes=3,
                num_targets=1,
            )


# =============================================================================
# Intermediate and full model outputs
# =============================================================================


class TestModelOutput:
    def test_valid_typed_intermediate_states(self) -> None:
        alignment = _prediction_alignment(
            num_nodes=3,
            num_targets=1,
        )
        prediction = PredictionOutput(
            prediction_mean=torch.ones(
                (3, 1),
                dtype=torch.float32,
            ),
            alignment=alignment,
        )

        states = IntermediateStates(
            node_states={
                "layer_0": torch.ones(
                    (3, 8),
                    dtype=torch.float32,
                )
            },
            edge_states={
                "messages": torch.ones(
                    (2, 8),
                    dtype=torch.float32,
                )
            },
            graph_states={
                "hazard_context": torch.ones(
                    (1, 8),
                    dtype=torch.float32,
                )
            },
            temporal_states={
                "memory": torch.ones(
                    (3, 3, 8),
                    dtype=torch.float32,
                )
            },
            auxiliary_states={
                "scalar_summary": torch.tensor(
                    [1.0],
                    dtype=torch.float32,
                )
            },
        )

        output = ModelOutput(
            predictions=prediction,
            intermediate_states=states,
            num_edges=2,
        )

        assert output.num_edges == 2
        assert "layer_0" in output.intermediate_states.node_states

    def test_wrong_edge_intermediate_alignment_is_rejected(
        self,
    ) -> None:
        alignment = _prediction_alignment(
            num_nodes=3,
            num_targets=1,
        )
        prediction = PredictionOutput(
            prediction_mean=torch.ones(
                (3, 1),
                dtype=torch.float32,
            ),
            alignment=alignment,
        )

        states = IntermediateStates(
            edge_states={
                "messages": torch.ones(
                    (3, 8),
                    dtype=torch.float32,
                )
            },
        )

        with pytest.raises(
            ValueError,
            match="first dimension 2",
        ):
            ModelOutput(
                predictions=prediction,
                intermediate_states=states,
                num_edges=2,
            )


@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA is required to construct a real device mismatch.",
)
def test_model_output_rejects_cross_child_device_mismatch() -> None:
    alignment = _prediction_alignment(
        num_nodes=3,
        num_targets=1,
        device="cuda",
    )

    prediction = PredictionOutput(
        prediction_mean=torch.ones(
            (3, 1),
            dtype=torch.float32,
            device="cuda",
        ),
        alignment=alignment,
    )

    uncertainty = UncertaintyOutput(
        method="heteroscedastic_variance",
        variance=torch.ones(
            (3, 1),
            dtype=torch.float32,
            device="cpu",
        ),
    )

    with pytest.raises(
        ValueError,
        match="same device",
    ):
        ModelOutput(
            predictions=prediction,
            uncertainty=uncertainty,
        )


# =============================================================================
# Controlled reconstruction and device movement
# =============================================================================


class TestBatchUtilities:
    def test_batch_to_returns_revalidated_copy(self) -> None:
        batch = UrbanGraphBatch(
            **_single_graph_batch_kwargs(supervised=True)
        )

        moved = batch.to("cpu")

        assert moved is not batch
        assert moved.node_batch_index.device.type == "cpu"
        assert moved.edge_index.device.type == "cpu"
        assert moved.targets is not None
        assert moved.targets.device.type == "cpu"
        assert torch.equal(
            moved.node_batch_index,
            batch.node_batch_index,
        )

    def test_batch_replace_returns_valid_reconstructed_copy(
        self,
    ) -> None:
        batch = UrbanGraphBatch(
            **_single_graph_batch_kwargs(supervised=False)
        )

        replacement_metadata = BatchMetadata(
            dataset_version="test-dataset-v2",
            graph_version="test-graph-v2",
            split_by_graph=("inference",),
            experiment_name="replacement_test",
        )

        replaced = batch.replace(
            metadata=replacement_metadata
        )

        assert replaced is not batch
        assert replaced.metadata.dataset_version == (
            "test-dataset-v2"
        )
        assert batch.metadata.dataset_version == (
            "test-dataset-v1"
        )

    def test_batch_replace_revalidates_changes(self) -> None:
        batch = UrbanGraphBatch(
            **_single_graph_batch_kwargs(supervised=False)
        )

        with pytest.raises(
            ValueError,
            match="edge_index must have rank 2",
        ):
            batch.replace(
                edge_index=torch.tensor(
                    [0, 1],
                    dtype=torch.long,
                )
            )