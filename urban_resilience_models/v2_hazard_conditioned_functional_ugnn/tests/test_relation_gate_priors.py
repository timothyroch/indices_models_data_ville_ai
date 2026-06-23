"""
Contract tests for compiled hazard-relation prior integration.

Target repository path:
    urban_resilience_models/
        v2_hazard_conditioned_functional_ugnn/
            tests/
                test_relation_gate_priors.py

Implementation under test:
    functional_message_passing/
        relation_family_gate/
            relation_priors.py

The bounded V2.0 prior-integration contract is:

- recover one exact runtime hazard identity per packed graph;
- resolve that identity against the compiled hazard-prior axis;
- expand graph prior rows to nodes through ``node_batch_index``;
- retain the exact compiled relation axis without family pooling;
- tensorize prior means, confidence, masks, gate-bias logits, and
  regularization weights;
- scale only the compiled gate-bias logits by
  ``relation_prior_strength``;
- return a metadata-preserving ``RelationPriorContribution``.

This suite covers constructor and configuration behavior, artifact alignment,
graph and node hazard lookup modes, exact tensor values, zero-strength
behavior, resolution diagnostics, fingerprints, validation failures, internal
result guards, semantic device comparison, and optional CUDA execution.
"""

from __future__ import annotations

from enum import Enum
from types import SimpleNamespace
from typing import Any
from unittest.mock import patch

import pytest
import torch
from torch import nn

from urban_resilience_models.v2_hazard_conditioned_functional_ugnn.functional_message_passing.relation_family_gate import (
    relation_priors as relation_priors_module,
)
from urban_resilience_models.v2_hazard_conditioned_functional_ugnn.functional_message_passing.relation_family_gate import (
    schemas as schemas_module,
)
from urban_resilience_models.v2_hazard_conditioned_functional_ugnn.functional_message_passing.relation_family_gate.relation_priors import (
    RELATION_PRIOR_INTEGRATION_SCHEMA_VERSION,
    RelationPriorBuilder,
    RelationPriorContributionBuilder,
    RelationPriorIntegration,
)
from urban_resilience_models.v2_hazard_conditioned_functional_ugnn.functional_message_passing.relation_family_gate.schemas import (
    RelationGateAxis,
    RelationPriorContribution,
)


NODE_COUNT = 5
GRAPH_COUNT = 2
RELATION_COUNT = 3
HAZARD_COUNT = 3

RELATION_NAMES = (
    "spatial_adjacency",
    "temporal_lag",
    "random_placebo",
)
STABLE_RELATION_IDS = (
    100,
    200,
    900,
)
CONTROL_MASK_VALUES = (
    False,
    False,
    True,
)

FAMILY_NAMES = (
    "physical",
    "control",
)
STABLE_FAMILY_IDS = (
    10,
    90,
)
RELATION_FAMILY_INDICES = (
    0,
    0,
    1,
)

COMPILED_REGISTRY_FINGERPRINT = (
    "compiled-relation-registry"
)
SOURCE_REGISTRY_FINGERPRINT = (
    "source-relation-registry"
)
COMPILED_PRIOR_FINGERPRINT = (
    "compiled-hazard-relation-priors"
)

COMPILED_HAZARD_NAMES = (
    "flood",
    "heat",
    "outage",
)
COMPILED_STABLE_HAZARD_IDS = (
    10,
    20,
    30,
)

GRAPH_HAZARD_NAMES = (
    "flood",
    "heat",
)
GRAPH_STABLE_HAZARD_IDS = (
    10,
    20,
)

NODE_BATCH_INDEX = (
    0,
    0,
    1,
    1,
    1,
)

PRIOR_MEAN_MATRIX = (
    (0.80, 0.60, 0.50),
    (0.70, 0.40, 0.50),
    (0.65, 0.55, 0.50),
)
CONFIDENCE_MATRIX = (
    (0.90, 0.80, 0.00),
    (0.70, 0.60, 0.00),
    (0.50, 0.40, 0.00),
)
INITIALIZATION_MASK = (
    (True, True, False),
    (True, True, False),
    (True, True, False),
)
REGULARIZATION_MASK = (
    (True, False, False),
    (False, True, False),
    (True, True, False),
)
BASE_LOGIT_MATRIX = (
    (1.00, 0.50, 0.00),
    (0.80, -0.40, 0.00),
    (0.60, 0.20, 0.00),
)
REGULARIZATION_WEIGHT_MATRIX = (
    (0.90, 0.00, 0.00),
    (0.00, 0.60, 0.00),
    (0.50, 0.40, 0.00),
)
RESOLUTION_MODE_MATRIX = (
    (
        "explicit",
        "explicit",
        "neutral_default",
    ),
    (
        "explicit",
        "relation_ancestor",
        "neutral_default",
    ),
    (
        "hazard_ancestor",
        "relation_ancestor",
        "neutral_default",
    ),
)


# =============================================================================
# Controlled upstream contracts
# =============================================================================


class FakeRelationConfig:
    def __init__(
        self,
        *,
        relation_prior_strength: float = 0.5,
        gate_enabled: bool = True,
        validation_error: Exception | None = None,
        implementation_error: Exception | None = None,
    ) -> None:
        self.relation_prior_strength = (
            relation_prior_strength
        )
        self.gate_enabled = gate_enabled
        self.validation_error = validation_error
        self.implementation_error = (
            implementation_error
        )
        self.validate_calls = 0
        self.assert_implemented_calls = 0

    def validate(self) -> None:
        self.validate_calls += 1
        if self.validation_error is not None:
            raise self.validation_error

    def assert_implemented(self) -> None:
        self.assert_implemented_calls += 1
        if self.implementation_error is not None:
            raise self.implementation_error


class FakeGateInitializationActivation(
    Enum
):
    SIGMOID = "sigmoid"


class FakeCompiledRegistry:
    def __init__(
        self,
        fingerprint: str = (
            COMPILED_REGISTRY_FINGERPRINT
        ),
    ) -> None:
        self._fingerprint = fingerprint

    def fingerprint(self) -> str:
        return self._fingerprint


class FakeHazardIndices:
    def __init__(
        self,
        *,
        hazard_names: tuple[str, ...] = (
            GRAPH_HAZARD_NAMES
        ),
        stable_hazard_ids: (
            torch.Tensor | None
        ) = None,
        unknown_mask: torch.Tensor | None = None,
        device: torch.device | str = "cpu",
    ) -> None:
        resolved_device = torch.device(
            device
        )
        self.hazard_names = tuple(
            hazard_names
        )
        self.stable_hazard_ids = (
            torch.tensor(
                GRAPH_STABLE_HAZARD_IDS,
                dtype=torch.long,
                device=resolved_device,
            )
            if stable_hazard_ids is None
            else stable_hazard_ids
        )
        self.unknown_mask = (
            torch.zeros(
                len(self.hazard_names),
                dtype=torch.bool,
                device=resolved_device,
            )
            if unknown_mask is None
            else unknown_mask
        )
        self.device = resolved_device

    def __len__(self) -> int:
        return len(self.hazard_names)


class FakeHazardEmbeddingLookup:
    def __init__(
        self,
        *,
        embeddings: torch.Tensor | None = None,
        indices: FakeHazardIndices | None = None,
        device: torch.device | str = "cpu",
    ) -> None:
        resolved_device = torch.device(
            device
        )
        self.embeddings = (
            torch.zeros(
                GRAPH_COUNT,
                4,
                dtype=torch.float32,
                device=resolved_device,
            )
            if embeddings is None
            else embeddings
        )
        self.indices = (
            FakeHazardIndices(
                device=resolved_device
            )
            if indices is None
            else indices
        )


class FakeNodeAlignedHazardEmbeddingLookup:
    def __init__(
        self,
        *,
        graph_lookup: (
            FakeHazardEmbeddingLookup | None
        ) = None,
        node_batch_index: (
            torch.Tensor | None
        ) = None,
        device: torch.device | str = "cpu",
    ) -> None:
        resolved_device = torch.device(
            device
        )
        self.graph_lookup = (
            FakeHazardEmbeddingLookup(
                device=resolved_device
            )
            if graph_lookup is None
            else graph_lookup
        )
        self.node_batch_index = (
            torch.tensor(
                NODE_BATCH_INDEX,
                dtype=torch.long,
                device=resolved_device,
            )
            if node_batch_index is None
            else node_batch_index
        )


class FakeCompiledHazardRelationPriors:
    def __init__(
        self,
        *,
        hazard_names: tuple[str, ...] = (
            COMPILED_HAZARD_NAMES
        ),
        stable_hazard_ids: tuple[int, ...] = (
            COMPILED_STABLE_HAZARD_IDS
        ),
        relation_names: tuple[str, ...] = (
            RELATION_NAMES
        ),
        stable_relation_ids: tuple[int, ...] = (
            STABLE_RELATION_IDS
        ),
        prior_mean_matrix: Any = (
            PRIOR_MEAN_MATRIX
        ),
        confidence_matrix: Any = (
            CONFIDENCE_MATRIX
        ),
        initialization_mask: Any = (
            INITIALIZATION_MASK
        ),
        regularization_mask: Any = (
            REGULARIZATION_MASK
        ),
        resolution_mode_matrix: Any = (
            RESOLUTION_MODE_MATRIX
        ),
        base_logit_matrix: Any = (
            BASE_LOGIT_MATRIX
        ),
        regularization_weight_matrix: Any = (
            REGULARIZATION_WEIGHT_MATRIX
        ),
        source_compiled_relation_fingerprint: str = (
            COMPILED_REGISTRY_FINGERPRINT
        ),
        fingerprint: str = (
            COMPILED_PRIOR_FINGERPRINT
        ),
        num_hazards: int | None = None,
        num_relations: int | None = None,
        validation_error: Exception | None = None,
    ) -> None:
        self.hazard_names = tuple(
            hazard_names
        )
        self.stable_hazard_ids = tuple(
            stable_hazard_ids
        )
        self.relation_names = tuple(
            relation_names
        )
        self.stable_relation_ids = tuple(
            stable_relation_ids
        )
        self.prior_mean_matrix = (
            prior_mean_matrix
        )
        self.confidence_matrix = (
            confidence_matrix
        )
        self.initialization_mask = (
            initialization_mask
        )
        self.regularization_mask = (
            regularization_mask
        )
        self.resolution_mode_matrix = (
            resolution_mode_matrix
        )
        self._base_logit_matrix = (
            base_logit_matrix
        )
        self._regularization_weight_matrix = (
            regularization_weight_matrix
        )
        self.source_compiled_relation_fingerprint = (
            source_compiled_relation_fingerprint
        )
        self.num_hazards = (
            len(self.hazard_names)
            if num_hazards is None
            else num_hazards
        )
        self.num_relations = (
            len(self.relation_names)
            if num_relations is None
            else num_relations
        )
        self._fingerprint = fingerprint
        self.validation_error = (
            validation_error
        )
        self.validate_calls = 0
        self.gate_bias_calls: list[
            tuple[Any, float]
        ] = []
        self.regularization_weight_calls = 0

    def validate(self) -> None:
        self.validate_calls += 1
        if self.validation_error is not None:
            raise self.validation_error

    def gate_bias_logit_matrix(
        self,
        *,
        activation: Any,
        epsilon: float,
    ) -> Any:
        self.gate_bias_calls.append(
            (activation, epsilon)
        )
        return self._base_logit_matrix

    def regularization_weight_matrix(
        self,
    ) -> Any:
        self.regularization_weight_calls += 1
        return self._regularization_weight_matrix

    def fingerprint(self) -> str:
        return self._fingerprint


class FakeFunctionalMessagePassingInputs:
    def __init__(
        self,
        *,
        num_nodes: int = NODE_COUNT,
        num_graphs: int = GRAPH_COUNT,
        relation_names: tuple[str, ...] = (
            RELATION_NAMES
        ),
        stable_relation_ids: tuple[int, ...] = (
            STABLE_RELATION_IDS
        ),
        dtype: torch.dtype = torch.float32,
        device: torch.device | str = "cpu",
        node_batch_index: torch.Tensor | None = None,
        compiled_relation_priors: object | None = ...,
        hazard_query: object | None = ...,
        compiled_registry_fingerprint: str = (
            COMPILED_REGISTRY_FINGERPRINT
        ),
        lineage_fingerprint: str = (
            "functional-message-passing-inputs"
        ),
    ) -> None:
        resolved_device = torch.device(
            device
        )
        self.num_nodes = num_nodes
        self.num_graphs = num_graphs
        self.relation_names = tuple(
            relation_names
        )
        self.stable_relation_ids = tuple(
            stable_relation_ids
        )
        self.num_relations = len(
            self.relation_names
        )
        self.dtype = dtype
        self.device = resolved_device
        self.node_batch_index = (
            torch.tensor(
                NODE_BATCH_INDEX,
                dtype=torch.long,
                device=resolved_device,
            )
            if node_batch_index is None
            else node_batch_index
        )
        self.control_relation_mask = (
            torch.tensor(
                CONTROL_MASK_VALUES,
                dtype=torch.bool,
                device=resolved_device,
            )
        )
        self.compiled_relation_registry = (
            FakeCompiledRegistry(
                compiled_registry_fingerprint
            )
        )
        self.compiled_relation_priors = (
            FakeCompiledHazardRelationPriors()
            if compiled_relation_priors is ...
            else compiled_relation_priors
        )
        self.hazard_query = (
            SimpleNamespace(
                source_embedding=(
                    FakeHazardEmbeddingLookup(
                        device=resolved_device
                    )
                )
            )
            if hazard_query is ...
            else hazard_query
        )
        self.relation_families = (
            SimpleNamespace(
                family_names=FAMILY_NAMES,
                stable_family_ids=(
                    STABLE_FAMILY_IDS
                ),
                relation_family_index_by_relation=torch.tensor(
                    RELATION_FAMILY_INDICES,
                    dtype=torch.long,
                    device=resolved_device,
                ),
                source_relation_registry_fingerprint=(
                    SOURCE_REGISTRY_FINGERPRINT
                ),
            )
        )
        self._lineage_fingerprint = (
            lineage_fingerprint
        )

    def lineage_fingerprint(self) -> str:
        return self._lineage_fingerprint


@pytest.fixture(autouse=True)
def _patch_upstream_types(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        relation_priors_module,
        "RelationConfig",
        FakeRelationConfig,
    )
    monkeypatch.setattr(
        relation_priors_module,
        "HazardEmbeddingLookup",
        FakeHazardEmbeddingLookup,
    )
    monkeypatch.setattr(
        relation_priors_module,
        "NodeAlignedHazardEmbeddingLookup",
        FakeNodeAlignedHazardEmbeddingLookup,
    )
    monkeypatch.setattr(
        relation_priors_module,
        "CompiledHazardRelationPriors",
        FakeCompiledHazardRelationPriors,
    )
    monkeypatch.setattr(
        relation_priors_module,
        "GateInitializationActivation",
        FakeGateInitializationActivation,
    )
    monkeypatch.setattr(
        relation_priors_module,
        "FunctionalMessagePassingInputs",
        FakeFunctionalMessagePassingInputs,
    )
    monkeypatch.setattr(
        schemas_module,
        "FunctionalMessagePassingInputs",
        FakeFunctionalMessagePassingInputs,
    )


# =============================================================================
# Helpers
# =============================================================================


def _inputs(
    **kwargs: Any,
) -> FakeFunctionalMessagePassingInputs:
    return FakeFunctionalMessagePassingInputs(
        **kwargs
    )


def _axis(
    *,
    inputs: (
        FakeFunctionalMessagePassingInputs
        | None
    ) = None,
) -> RelationGateAxis:
    resolved_inputs = (
        _inputs()
        if inputs is None
        else inputs
    )
    return RelationGateAxis.from_inputs(
        source_inputs=resolved_inputs
    )


def _builder(
    *,
    strength: float = 0.5,
    epsilon: float = 1e-4,
) -> RelationPriorContributionBuilder:
    return RelationPriorContributionBuilder(
        strength=strength,
        epsilon=epsilon,
    )


def _expected_rows(
    *,
    device: torch.device | str = "cpu",
) -> torch.Tensor:
    return torch.tensor(
        [0, 0, 1, 1, 1],
        dtype=torch.long,
        device=device,
    )


def _expected_prior_mean(
    *,
    dtype: torch.dtype = torch.float32,
    device: torch.device | str = "cpu",
) -> torch.Tensor:
    return torch.tensor(
        [
            PRIOR_MEAN_MATRIX[0],
            PRIOR_MEAN_MATRIX[0],
            PRIOR_MEAN_MATRIX[1],
            PRIOR_MEAN_MATRIX[1],
            PRIOR_MEAN_MATRIX[1],
        ],
        dtype=dtype,
        device=device,
    )


def _expected_confidence(
    *,
    dtype: torch.dtype = torch.float32,
    device: torch.device | str = "cpu",
) -> torch.Tensor:
    return torch.tensor(
        [
            CONFIDENCE_MATRIX[0],
            CONFIDENCE_MATRIX[0],
            CONFIDENCE_MATRIX[1],
            CONFIDENCE_MATRIX[1],
            CONFIDENCE_MATRIX[1],
        ],
        dtype=dtype,
        device=device,
    )


def _expected_initialization_mask(
    *,
    device: torch.device | str = "cpu",
) -> torch.Tensor:
    return torch.tensor(
        [
            INITIALIZATION_MASK[0],
            INITIALIZATION_MASK[0],
            INITIALIZATION_MASK[1],
            INITIALIZATION_MASK[1],
            INITIALIZATION_MASK[1],
        ],
        dtype=torch.bool,
        device=device,
    )


def _expected_regularization_mask(
    *,
    device: torch.device | str = "cpu",
) -> torch.Tensor:
    return torch.tensor(
        [
            REGULARIZATION_MASK[0],
            REGULARIZATION_MASK[0],
            REGULARIZATION_MASK[1],
            REGULARIZATION_MASK[1],
            REGULARIZATION_MASK[1],
        ],
        dtype=torch.bool,
        device=device,
    )


def _expected_base_logits(
    *,
    dtype: torch.dtype = torch.float32,
    device: torch.device | str = "cpu",
) -> torch.Tensor:
    return torch.tensor(
        [
            BASE_LOGIT_MATRIX[0],
            BASE_LOGIT_MATRIX[0],
            BASE_LOGIT_MATRIX[1],
            BASE_LOGIT_MATRIX[1],
            BASE_LOGIT_MATRIX[1],
        ],
        dtype=dtype,
        device=device,
    )


def _expected_regularization_weights(
    *,
    dtype: torch.dtype = torch.float32,
    device: torch.device | str = "cpu",
) -> torch.Tensor:
    return torch.tensor(
        [
            REGULARIZATION_WEIGHT_MATRIX[0],
            REGULARIZATION_WEIGHT_MATRIX[0],
            REGULARIZATION_WEIGHT_MATRIX[1],
            REGULARIZATION_WEIGHT_MATRIX[1],
            REGULARIZATION_WEIGHT_MATRIX[1],
        ],
        dtype=dtype,
        device=device,
    )


def _expected_resolution_summary() -> dict[
    str,
    int,
]:
    return {
        "explicit": 7,
        "neutral_default": 5,
        "relation_ancestor": 3,
    }


# =============================================================================
# Public identity and constructor
# =============================================================================


def test_schema_version_is_nonempty() -> None:
    assert isinstance(
        RELATION_PRIOR_INTEGRATION_SCHEMA_VERSION,
        str,
    )
    assert (
        RELATION_PRIOR_INTEGRATION_SCHEMA_VERSION
        .strip()
    )


def test_aliases_point_to_builder() -> None:
    assert RelationPriorBuilder is (
        RelationPriorContributionBuilder
    )
    assert RelationPriorIntegration is (
        RelationPriorContributionBuilder
    )


def test_builder_is_torch_module() -> None:
    assert issubclass(
        RelationPriorContributionBuilder,
        nn.Module,
    )


def test_default_constructor_contract() -> None:
    builder = (
        RelationPriorContributionBuilder()
    )

    assert builder.strength == 0.0
    assert builder.epsilon == 1e-4
    assert builder.is_zero_strength


def test_constructor_accepts_positive_strength() -> None:
    builder = _builder(
        strength=0.75,
        epsilon=1e-3,
    )

    assert builder.strength == 0.75
    assert builder.epsilon == 1e-3
    assert not builder.is_zero_strength


@pytest.mark.parametrize(
    "strength",
    (
        True,
        "0.5",
        None,
    ),
)
def test_constructor_rejects_nonnumeric_strength(
    strength: Any,
) -> None:
    with pytest.raises(
        TypeError,
        match="must be numeric",
    ):
        RelationPriorContributionBuilder(
            strength=strength
        )


@pytest.mark.parametrize(
    "strength",
    (
        -0.1,
        float("nan"),
        float("inf"),
        float("-inf"),
    ),
)
def test_constructor_rejects_invalid_strength(
    strength: float,
) -> None:
    with pytest.raises(ValueError):
        RelationPriorContributionBuilder(
            strength=strength
        )


@pytest.mark.parametrize(
    "epsilon",
    (
        True,
        "0.01",
        None,
    ),
)
def test_constructor_rejects_nonnumeric_epsilon(
    epsilon: Any,
) -> None:
    with pytest.raises(
        TypeError,
        match="must be numeric",
    ):
        RelationPriorContributionBuilder(
            epsilon=epsilon
        )


@pytest.mark.parametrize(
    "epsilon",
    (
        -0.1,
        0.0,
        0.5,
        1.0,
        float("nan"),
        float("inf"),
    ),
)
def test_constructor_rejects_invalid_epsilon(
    epsilon: float,
) -> None:
    with pytest.raises(ValueError):
        RelationPriorContributionBuilder(
            epsilon=epsilon
        )


# =============================================================================
# Construction from config
# =============================================================================


def test_from_config_builds_builder() -> None:
    config = FakeRelationConfig(
        relation_prior_strength=0.7,
        gate_enabled=True,
    )

    builder = (
        RelationPriorContributionBuilder
        .from_config(
            config=config,
            epsilon=1e-3,
        )
    )

    assert config.validate_calls == 1
    assert (
        config.assert_implemented_calls
        == 1
    )
    assert builder.strength == 0.7
    assert builder.epsilon == 1e-3


def test_from_disabled_config_skips_assert_implemented() -> None:
    config = FakeRelationConfig(
        relation_prior_strength=0.2,
        gate_enabled=False,
    )

    builder = (
        RelationPriorContributionBuilder
        .from_config(
            config=config
        )
    )

    assert config.validate_calls == 1
    assert (
        config.assert_implemented_calls
        == 0
    )
    assert builder.strength == 0.2


def test_from_config_rejects_wrong_type() -> None:
    with pytest.raises(
        TypeError,
        match="RelationConfig",
    ):
        RelationPriorContributionBuilder.from_config(
            config=object()  # type: ignore[arg-type]
        )


def test_from_config_propagates_validation_error() -> None:
    config = FakeRelationConfig(
        validation_error=RuntimeError(
            "invalid relation config"
        )
    )

    with pytest.raises(
        RuntimeError,
        match="invalid relation config",
    ):
        RelationPriorContributionBuilder.from_config(
            config=config
        )


def test_from_config_propagates_implementation_error() -> None:
    config = FakeRelationConfig(
        implementation_error=(
            NotImplementedError(
                "prior integration unavailable"
            )
        )
    )

    with pytest.raises(
        NotImplementedError,
        match="prior integration unavailable",
    ):
        RelationPriorContributionBuilder.from_config(
            config=config
        )


# =============================================================================
# Parameter-free identity and architecture
# =============================================================================


def test_builder_is_parameter_free() -> None:
    builder = _builder()

    assert builder.parameter_count == 0
    assert (
        builder.trainable_parameter_count
        == 0
    )
    assert tuple(builder.parameters()) == ()
    assert builder.state_dict() == {}


def test_architecture_dict_is_exact() -> None:
    builder = _builder(
        strength=0.5,
        epsilon=1e-4,
    )

    assert builder.architecture_dict() == {
        "schema_version": (
            RELATION_PRIOR_INTEGRATION_SCHEMA_VERSION
        ),
        "strength": 0.5,
        "epsilon": 1e-4,
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


def test_architecture_fingerprint_is_stable() -> None:
    first = _builder()
    second = _builder()

    assert first.architecture_dict() == (
        second.architecture_dict()
    )
    assert first.architecture_fingerprint() == (
        second.architecture_fingerprint()
    )


def test_strength_changes_architecture_fingerprint() -> None:
    assert (
        _builder(
            strength=0.5
        ).architecture_fingerprint()
        != _builder(
            strength=0.6
        ).architecture_fingerprint()
    )


def test_epsilon_changes_architecture_fingerprint() -> None:
    assert (
        _builder(
            epsilon=1e-4
        ).architecture_fingerprint()
        != _builder(
            epsilon=1e-3
        ).architecture_fingerprint()
    )


def test_parameter_fingerprint_is_stable() -> None:
    assert (
        _builder().parameter_fingerprint()
        == _builder().parameter_fingerprint()
    )


def test_assert_finite_parameters_is_noop() -> None:
    _builder().assert_finite_parameters()


def test_assert_finite_parameters_rejects_nonzero_contract() -> None:
    class InvalidBuilder(
        RelationPriorContributionBuilder
    ):
        @property
        def parameter_count(self) -> int:
            return 1

    with pytest.raises(
        RuntimeError,
        match="must remain parameter-free",
    ):
        InvalidBuilder().assert_finite_parameters()


# =============================================================================
# Semantic device helper
# =============================================================================


def test_devices_match_cpu() -> None:
    assert relation_priors_module._devices_match(
        "cpu",
        torch.device("cpu"),
    )


def test_devices_match_rejects_cpu_cuda() -> None:
    assert not relation_priors_module._devices_match(
        "cpu",
        "cuda:0",
    )


def test_devices_match_explicit_cuda_indices() -> None:
    assert relation_priors_module._devices_match(
        "cuda:0",
        "cuda:0",
    )
    assert not relation_priors_module._devices_match(
        "cuda:0",
        "cuda:1",
    )


def test_devices_match_resolves_implicit_cuda_index() -> None:
    with patch.object(
        torch.cuda,
        "current_device",
        return_value=0,
    ):
        assert (
            relation_priors_module
            ._devices_match(
                "cuda",
                "cuda:0",
            )
        )
        assert (
            relation_priors_module
            ._devices_match(
                "cuda:0",
                "cuda",
            )
        )
        assert not (
            relation_priors_module
            ._devices_match(
                "cuda",
                "cuda:1",
            )
        )


# =============================================================================
# Input and compiled-artifact validation
# =============================================================================


def test_rejects_wrong_input_type() -> None:
    with pytest.raises(
        TypeError,
        match="FunctionalMessagePassingInputs",
    ):
        _builder().resolve_node_hazard_rows(
            object()  # type: ignore[arg-type]
        )


@pytest.mark.parametrize(
    ("field_name", "value", "message"),
    (
        (
            "num_nodes",
            0,
            "at least one node",
        ),
        (
            "num_graphs",
            0,
            "at least one graph",
        ),
    ),
)
def test_rejects_nonpositive_input_counts(
    field_name: str,
    value: int,
    message: str,
) -> None:
    inputs = _inputs()
    setattr(
        inputs,
        field_name,
        value,
    )

    with pytest.raises(
        ValueError,
        match=message,
    ):
        _builder().resolve_node_hazard_rows(
            inputs
        )


def test_rejects_zero_relation_count() -> None:
    inputs = _inputs(
        relation_names=(),
        stable_relation_ids=(),
    )

    with pytest.raises(
        ValueError,
        match="at least one relation",
    ):
        _builder().resolve_node_hazard_rows(
            inputs
        )


@pytest.mark.parametrize(
    "dtype",
    (
        torch.long,
        torch.int32,
        torch.bool,
    ),
)
def test_rejects_nonfloating_input_dtype(
    dtype: torch.dtype,
) -> None:
    inputs = _inputs(
        dtype=dtype
    )

    with pytest.raises(
        ValueError,
        match="floating-point",
    ):
        _builder().resolve_node_hazard_rows(
            inputs
        )


def test_requires_compiled_prior_artifact() -> None:
    inputs = _inputs(
        compiled_relation_priors=None
    )

    with pytest.raises(
        ValueError,
        match="compiled_relation_priors",
    ):
        _builder().resolve_node_hazard_rows(
            inputs
        )


def test_rejects_wrong_compiled_prior_type() -> None:
    inputs = _inputs(
        compiled_relation_priors=object()
    )

    with pytest.raises(
        TypeError,
        match="CompiledHazardRelationPriors",
    ):
        _builder().resolve_node_hazard_rows(
            inputs
        )


def test_compiled_prior_validate_is_called() -> None:
    compiled = (
        FakeCompiledHazardRelationPriors()
    )
    inputs = _inputs(
        compiled_relation_priors=compiled
    )

    _builder().resolve_node_hazard_rows(
        inputs
    )

    assert compiled.validate_calls >= 1


def test_compiled_prior_validation_error_propagates() -> None:
    compiled = (
        FakeCompiledHazardRelationPriors(
            validation_error=RuntimeError(
                "invalid compiled priors"
            )
        )
    )
    inputs = _inputs(
        compiled_relation_priors=compiled
    )

    with pytest.raises(
        RuntimeError,
        match="invalid compiled priors",
    ):
        _builder().resolve_node_hazard_rows(
            inputs
        )


def test_rejects_compiled_relation_order_mismatch() -> None:
    compiled = (
        FakeCompiledHazardRelationPriors(
            relation_names=(
                "temporal_lag",
                "spatial_adjacency",
                "random_placebo",
            )
        )
    )
    inputs = _inputs(
        compiled_relation_priors=compiled
    )

    with pytest.raises(
        ValueError,
        match="relation ordering differs",
    ):
        _builder().resolve_node_hazard_rows(
            inputs
        )


def test_rejects_compiled_stable_relation_id_mismatch() -> None:
    compiled = (
        FakeCompiledHazardRelationPriors(
            stable_relation_ids=(
                100,
                201,
                900,
            )
        )
    )
    inputs = _inputs(
        compiled_relation_priors=compiled
    )

    with pytest.raises(
        ValueError,
        match="stable relation IDs differ",
    ):
        _builder().resolve_node_hazard_rows(
            inputs
        )


def test_rejects_compiled_registry_fingerprint_mismatch() -> None:
    compiled = (
        FakeCompiledHazardRelationPriors(
            source_compiled_relation_fingerprint=(
                "different-registry"
            )
        )
    )
    inputs = _inputs(
        compiled_relation_priors=compiled
    )

    with pytest.raises(
        ValueError,
        match="different compiled relation registry",
    ):
        _builder().resolve_node_hazard_rows(
            inputs
        )


def test_rejects_compiled_relation_count_mismatch() -> None:
    compiled = (
        FakeCompiledHazardRelationPriors(
            num_relations=RELATION_COUNT + 1
        )
    )
    inputs = _inputs(
        compiled_relation_priors=compiled
    )

    with pytest.raises(
        ValueError,
        match="relation count differs",
    ):
        _builder().resolve_node_hazard_rows(
            inputs
        )


# =============================================================================
# Graph hazard lookup and node membership
# =============================================================================


def test_graph_lookup_mode_uses_input_node_batch_index() -> None:
    inputs = _inputs()
    graph_lookup, node_batch_index = (
        relation_priors_module
        ._graph_lookup_and_node_membership(
            inputs
        )
    )

    assert graph_lookup is (
        inputs
        .hazard_query
        .source_embedding
    )
    assert node_batch_index is (
        inputs.node_batch_index
    )


def test_node_aligned_lookup_mode() -> None:
    graph_lookup = (
        FakeHazardEmbeddingLookup()
    )
    node_membership = torch.tensor(
        NODE_BATCH_INDEX,
        dtype=torch.long,
    )
    inputs = _inputs(
        hazard_query=SimpleNamespace(
            source_embedding=(
                FakeNodeAlignedHazardEmbeddingLookup(
                    graph_lookup=graph_lookup,
                    node_batch_index=(
                        node_membership
                    ),
                )
            )
        )
    )

    observed_lookup, observed_membership = (
        relation_priors_module
        ._graph_lookup_and_node_membership(
            inputs
        )
    )

    assert observed_lookup is graph_lookup
    assert observed_membership is (
        node_membership
    )


def test_requires_hazard_query() -> None:
    inputs = _inputs(
        hazard_query=None
    )

    with pytest.raises(
        ValueError,
        match="requires source_inputs.hazard_query",
    ):
        _builder().resolve_node_hazard_rows(
            inputs
        )


def test_rejects_unknown_source_embedding_type() -> None:
    inputs = _inputs(
        hazard_query=SimpleNamespace(
            source_embedding=object()
        )
    )

    with pytest.raises(
        TypeError,
        match="HazardEmbeddingLookup",
    ):
        _builder().resolve_node_hazard_rows(
            inputs
        )


def test_node_aligned_lookup_rejects_membership_mismatch() -> None:
    inputs = _inputs(
        hazard_query=SimpleNamespace(
            source_embedding=(
                FakeNodeAlignedHazardEmbeddingLookup(
                    node_batch_index=torch.tensor(
                        [0, 1, 1, 1, 1],
                        dtype=torch.long,
                    )
                )
            )
        )
    )

    with pytest.raises(
        ValueError,
        match="graph membership differs",
    ):
        _builder().resolve_node_hazard_rows(
            inputs
        )


def test_rejects_graph_hazard_count_mismatch() -> None:
    lookup = FakeHazardEmbeddingLookup(
        indices=FakeHazardIndices(
            hazard_names=("flood",),
            stable_hazard_ids=torch.tensor(
                [10],
                dtype=torch.long,
            ),
        ),
        embeddings=torch.zeros(
            1,
            4,
        ),
    )
    inputs = _inputs(
        hazard_query=SimpleNamespace(
            source_embedding=lookup
        )
    )

    with pytest.raises(
        ValueError,
        match="packed graph count",
    ):
        _builder().resolve_node_hazard_rows(
            inputs
        )


def test_rejects_node_batch_shape_mismatch() -> None:
    inputs = _inputs(
        node_batch_index=torch.tensor(
            [0, 0, 1],
            dtype=torch.long,
        )
    )

    with pytest.raises(
        ValueError,
        match=r"shape \[N\]",
    ):
        _builder().resolve_node_hazard_rows(
            inputs
        )


def test_rejects_node_batch_dtype_mismatch() -> None:
    inputs = _inputs(
        node_batch_index=torch.tensor(
            NODE_BATCH_INDEX,
            dtype=torch.int32,
        )
    )

    with pytest.raises(
        ValueError,
        match="torch.long",
    ):
        _builder().resolve_node_hazard_rows(
            inputs
        )


@pytest.mark.parametrize(
    "bad_index",
    (
        -1,
        GRAPH_COUNT,
    ),
)
def test_rejects_out_of_range_node_batch_index(
    bad_index: int,
) -> None:
    node_batch = torch.tensor(
        NODE_BATCH_INDEX,
        dtype=torch.long,
    )
    node_batch[0] = bad_index
    inputs = _inputs(
        node_batch_index=node_batch
    )

    with pytest.raises(
        ValueError,
        match="out-of-range graph indices",
    ):
        _builder().resolve_node_hazard_rows(
            inputs
        )


def test_rejects_unknown_runtime_hazard() -> None:
    lookup = FakeHazardEmbeddingLookup(
        indices=FakeHazardIndices(
            unknown_mask=torch.tensor(
                [False, True],
                dtype=torch.bool,
            )
        )
    )
    inputs = _inputs(
        hazard_query=SimpleNamespace(
            source_embedding=lookup
        )
    )

    with pytest.raises(
        ValueError,
        match="unknown runtime hazards",
    ):
        _builder().resolve_node_hazard_rows(
            inputs
        )


# =============================================================================
# Hazard-row resolution
# =============================================================================


def test_resolve_node_hazard_rows_exact_values() -> None:
    observed = (
        _builder()
        .resolve_node_hazard_rows(
            _inputs()
        )
    )

    assert torch.equal(
        observed,
        _expected_rows(),
    )


def test_hazard_row_maps_exact_values() -> None:
    compiled = (
        FakeCompiledHazardRelationPriors()
    )

    by_name, by_id = (
        relation_priors_module
        ._hazard_row_maps(compiled)
    )

    assert by_name == {
        "flood": 0,
        "heat": 1,
        "outage": 2,
    }
    assert by_id == {
        10: 0,
        20: 1,
        30: 2,
    }


def test_hazard_row_maps_rejects_name_axis_length_mismatch() -> None:
    compiled = (
        FakeCompiledHazardRelationPriors(
            num_hazards=HAZARD_COUNT + 1
        )
    )

    with pytest.raises(
        ValueError,
        match="hazard names do not align",
    ):
        relation_priors_module._hazard_row_maps(
            compiled
        )


def test_hazard_row_maps_rejects_duplicate_names() -> None:
    compiled = (
        FakeCompiledHazardRelationPriors(
            hazard_names=(
                "flood",
                "flood",
                "outage",
            )
        )
    )

    with pytest.raises(
        ValueError,
        match="hazard names must be unique",
    ):
        relation_priors_module._hazard_row_maps(
            compiled
        )


def test_hazard_row_maps_rejects_duplicate_ids() -> None:
    compiled = (
        FakeCompiledHazardRelationPriors(
            stable_hazard_ids=(
                10,
                10,
                30,
            )
        )
    )

    with pytest.raises(
        ValueError,
        match="stable hazard IDs must be unique",
    ):
        relation_priors_module._hazard_row_maps(
            compiled
        )


def test_graph_prior_rows_rejects_unknown_name() -> None:
    lookup = FakeHazardEmbeddingLookup(
        indices=FakeHazardIndices(
            hazard_names=(
                "flood",
                "wildfire",
            ),
            stable_hazard_ids=torch.tensor(
                [10, 20],
                dtype=torch.long,
            ),
        )
    )

    with pytest.raises(
        ValueError,
        match="absent from the compiled prior hazard axis",
    ):
        relation_priors_module._graph_prior_rows(
            graph_lookup=lookup,
            compiled=(
                FakeCompiledHazardRelationPriors()
            ),
            device="cpu",
        )


def test_graph_prior_rows_rejects_unknown_stable_id() -> None:
    lookup = FakeHazardEmbeddingLookup(
        indices=FakeHazardIndices(
            stable_hazard_ids=torch.tensor(
                [10, 999],
                dtype=torch.long,
            )
        )
    )

    with pytest.raises(
        ValueError,
        match="stable hazard ID",
    ):
        relation_priors_module._graph_prior_rows(
            graph_lookup=lookup,
            compiled=(
                FakeCompiledHazardRelationPriors()
            ),
            device="cpu",
        )


def test_graph_prior_rows_rejects_name_id_disagreement() -> None:
    lookup = FakeHazardEmbeddingLookup(
        indices=FakeHazardIndices(
            hazard_names=(
                "flood",
                "heat",
            ),
            stable_hazard_ids=torch.tensor(
                [20, 10],
                dtype=torch.long,
            ),
        )
    )

    with pytest.raises(
        ValueError,
        match="different compiled prior rows",
    ):
        relation_priors_module._graph_prior_rows(
            graph_lookup=lookup,
            compiled=(
                FakeCompiledHazardRelationPriors()
            ),
            device="cpu",
        )


# =============================================================================
# Tensor resolution and regularization
# =============================================================================


def test_resolved_tensors_exact_values() -> None:
    inputs = _inputs()
    builder = _builder(
        strength=0.5
    )

    tensors = builder._resolved_tensors(
        inputs
    )

    assert torch.equal(
        tensors["node_hazard_rows"],
        _expected_rows(),
    )
    assert torch.equal(
        tensors["prior_mean"],
        _expected_prior_mean(),
    )
    assert torch.equal(
        tensors["confidence"],
        _expected_confidence(),
    )
    assert torch.equal(
        tensors["initialization_mask"],
        _expected_initialization_mask(),
    )
    assert torch.equal(
        tensors["regularization_mask"],
        _expected_regularization_mask(),
    )
    assert torch.equal(
        tensors["base_logits"],
        _expected_base_logits(),
    )
    assert torch.equal(
        tensors["regularization_weights"],
        _expected_regularization_weights(),
    )
    assert torch.equal(
        tensors["logit_contribution"],
        _expected_base_logits() * 0.5,
    )


@pytest.mark.parametrize(
    "dtype",
    (
        torch.float32,
        torch.float64,
    ),
)
def test_resolved_float_tensors_preserve_dtype(
    dtype: torch.dtype,
) -> None:
    inputs = _inputs(
        dtype=dtype
    )
    tensors = _builder()._resolved_tensors(
        inputs
    )

    for name in (
        "prior_mean",
        "confidence",
        "base_logits",
        "regularization_weights",
        "logit_contribution",
    ):
        assert tensors[name].dtype == dtype


def test_compiled_gate_bias_receives_sigmoid_and_epsilon() -> None:
    compiled = (
        FakeCompiledHazardRelationPriors()
    )
    inputs = _inputs(
        compiled_relation_priors=compiled
    )
    builder = _builder(
        epsilon=0.0125
    )

    builder._resolved_tensors(
        inputs
    )

    assert compiled.gate_bias_calls
    activation, epsilon = (
        compiled.gate_bias_calls[-1]
    )
    assert activation is (
        FakeGateInitializationActivation
        .SIGMOID
    )
    assert epsilon == 0.0125


def test_regularization_weight_matrix_is_called() -> None:
    compiled = (
        FakeCompiledHazardRelationPriors()
    )
    inputs = _inputs(
        compiled_relation_priors=compiled
    )

    _builder().regularization_weights(
        inputs
    )

    assert (
        compiled.regularization_weight_calls
        >= 1
    )


def test_regularization_weights_exact_values() -> None:
    observed = (
        _builder()
        .regularization_weights(
            _inputs()
        )
    )

    assert torch.equal(
        observed,
        _expected_regularization_weights(),
    )


def test_zero_strength_produces_exact_zero_contribution() -> None:
    tensors = (
        _builder(
            strength=0.0
        )
        ._resolved_tensors(
            _inputs()
        )
    )

    assert torch.equal(
        tensors["logit_contribution"],
        torch.zeros(
            NODE_COUNT,
            RELATION_COUNT,
        ),
    )


def test_strength_scales_only_logit_contribution() -> None:
    inputs = _inputs()
    first = _builder(
        strength=0.5
    )._resolved_tensors(inputs)
    second = _builder(
        strength=2.0
    )._resolved_tensors(inputs)

    for name in (
        "prior_mean",
        "confidence",
        "initialization_mask",
        "regularization_mask",
        "base_logits",
        "regularization_weights",
    ):
        assert torch.equal(
            first[name],
            second[name],
        )

    assert torch.equal(
        second["logit_contribution"],
        first["logit_contribution"]
        * 4.0,
    )


@pytest.mark.parametrize(
    ("field_name", "bad_matrix", "message"),
    (
        (
            "prior_mean_matrix",
            (
                (0.0, 0.6, 0.5),
                PRIOR_MEAN_MATRIX[1],
                PRIOR_MEAN_MATRIX[2],
            ),
            "strictly inside",
        ),
        (
            "confidence_matrix",
            (
                (1.1, 0.8, 0.0),
                CONFIDENCE_MATRIX[1],
                CONFIDENCE_MATRIX[2],
            ),
            r"\[0, 1\]",
        ),
        (
            "regularization_weight_matrix",
            (
                (-0.1, 0.0, 0.0),
                REGULARIZATION_WEIGHT_MATRIX[1],
                REGULARIZATION_WEIGHT_MATRIX[2],
            ),
            "nonnegative",
        ),
    ),
)
def test_rejects_invalid_resolved_values(
    field_name: str,
    bad_matrix: Any,
    message: str,
) -> None:
    kwargs: dict[str, Any] = {}
    if field_name == (
        "regularization_weight_matrix"
    ):
        kwargs[
            "regularization_weight_matrix"
        ] = bad_matrix
    else:
        kwargs[field_name] = bad_matrix

    compiled = (
        FakeCompiledHazardRelationPriors(
            **kwargs
        )
    )
    inputs = _inputs(
        compiled_relation_priors=compiled
    )

    with pytest.raises(
        RuntimeError,
        match=message,
    ):
        _builder()._resolved_tensors(
            inputs
        )


def test_rejects_nonzero_weight_outside_regularization_mask() -> None:
    compiled = (
        FakeCompiledHazardRelationPriors(
            regularization_weight_matrix=(
                (0.9, 0.1, 0.0),
                REGULARIZATION_WEIGHT_MATRIX[1],
                REGULARIZATION_WEIGHT_MATRIX[2],
            )
        )
    )
    inputs = _inputs(
        compiled_relation_priors=compiled
    )

    with pytest.raises(
        RuntimeError,
        match="Regularization-disabled cells",
    ):
        _builder()._resolved_tensors(
            inputs
        )


def test_rejects_nonfinite_compiled_float_table() -> None:
    compiled = (
        FakeCompiledHazardRelationPriors(
            prior_mean_matrix=(
                (
                    float("nan"),
                    0.6,
                    0.5,
                ),
                PRIOR_MEAN_MATRIX[1],
                PRIOR_MEAN_MATRIX[2],
            )
        )
    )
    inputs = _inputs(
        compiled_relation_priors=compiled
    )

    with pytest.raises(
        FloatingPointError,
        match="NaN or infinity",
    ):
        _builder()._resolved_tensors(
            inputs
        )


def test_rejects_compiled_table_shape_mismatch() -> None:
    compiled = (
        FakeCompiledHazardRelationPriors(
            confidence_matrix=(
                CONFIDENCE_MATRIX[0],
                CONFIDENCE_MATRIX[1],
            )
        )
    )
    inputs = _inputs(
        compiled_relation_priors=compiled
    )

    with pytest.raises(
        RuntimeError,
        match="confidence_table has shape",
    ):
        _builder()._resolved_tensors(
            inputs
        )


# =============================================================================
# Resolution summary
# =============================================================================


def test_resolution_summary_exact_values() -> None:
    observed = (
        _builder()
        ._resolution_summary(
            _inputs()
        )
    )

    assert observed == (
        _expected_resolution_summary()
    )


def test_resolution_summary_is_sorted() -> None:
    summary = (
        _builder()
        ._resolution_summary(
            _inputs()
        )
    )

    assert list(summary) == sorted(summary)


def test_resolution_summary_counts_all_node_relation_cells() -> None:
    summary = (
        _builder()
        ._resolution_summary(
            _inputs()
        )
    )

    assert sum(summary.values()) == (
        NODE_COUNT * RELATION_COUNT
    )


def test_resolution_summary_rejects_incomplete_mode_matrix() -> None:
    compiled = (
        FakeCompiledHazardRelationPriors(
            resolution_mode_matrix=(
                (
                    "explicit",
                    "explicit",
                ),
                RESOLUTION_MODE_MATRIX[1],
                RESOLUTION_MODE_MATRIX[2],
            )
        )
    )
    inputs = _inputs(
        compiled_relation_priors=compiled
    )

    with pytest.raises(
        RuntimeError,
        match="do not cover every",
    ):
        _builder()._resolution_summary(
            inputs
        )


# =============================================================================
# Forward output
# =============================================================================


def test_forward_constructs_complete_output() -> None:
    inputs = _inputs()
    axis = _axis(inputs=inputs)
    builder = _builder(
        strength=0.5
    )

    output = builder(
        inputs,
        axis=axis,
    )

    assert isinstance(
        output,
        RelationPriorContribution,
    )
    assert output.source_inputs is inputs
    assert output.axis is axis
    assert output.strength == 0.5
    assert (
        output
        .source_compiled_prior_fingerprint
        == COMPILED_PRIOR_FINGERPRINT
    )
    assert torch.equal(
        output.logit_contribution,
        _expected_base_logits() * 0.5,
    )
    assert torch.equal(
        output.prior_mean,
        _expected_prior_mean(),
    )
    assert torch.equal(
        output.confidence,
        _expected_confidence(),
    )
    assert torch.equal(
        output.initialization_mask,
        _expected_initialization_mask(),
    )
    assert torch.equal(
        output.regularization_mask,
        _expected_regularization_mask(),
    )
    assert dict(
        output.resolution_summary
    ) == _expected_resolution_summary()


def test_forward_builds_axis_when_absent() -> None:
    inputs = _inputs()

    output = _builder()(inputs)

    assert isinstance(
        output.axis,
        RelationGateAxis,
    )
    output.axis.assert_matches_inputs(
        inputs
    )


def test_forward_rejects_wrong_axis_type() -> None:
    with pytest.raises(
        TypeError,
        match="RelationGateAxis or None",
    ):
        _builder()(
            _inputs(),
            axis=object(),  # type: ignore[arg-type]
        )


def test_forward_rejects_axis_mismatch() -> None:
    inputs = _inputs()
    axis = _axis(inputs=inputs)
    object.__setattr__(
        axis,
        "compiled_relation_registry_fingerprint",
        "different-registry",
    )

    with pytest.raises(
        ValueError,
        match="different compiled relation registry",
    ):
        _builder()(
            inputs,
            axis=axis,
        )


def test_forward_is_deterministic() -> None:
    inputs = _inputs()
    builder = _builder()

    first = builder(inputs)
    second = builder(inputs)

    assert torch.equal(
        first.logit_contribution,
        second.logit_contribution,
    )
    assert torch.equal(
        first.prior_mean,
        second.prior_mean,
    )
    assert dict(
        first.resolution_summary
    ) == dict(
        second.resolution_summary
    )
    assert first.fingerprint() == (
        second.fingerprint()
    )


# =============================================================================
# Internal result guards
# =============================================================================


def test_require_node_relation_tensor_rejects_nontensor() -> None:
    with pytest.raises(
        RuntimeError,
        match="must be a tensor",
    ):
        (
            relation_priors_module
            ._require_node_relation_tensor(
                "value",
                [1.0],  # type: ignore[arg-type]
                source_inputs=_inputs(),
            )
        )


def test_require_node_relation_tensor_rejects_shape() -> None:
    with pytest.raises(
        RuntimeError,
        match="has shape",
    ):
        (
            relation_priors_module
            ._require_node_relation_tensor(
                "value",
                torch.zeros(
                    NODE_COUNT,
                    RELATION_COUNT + 1,
                ),
                source_inputs=_inputs(),
            )
        )


def test_require_node_relation_tensor_rejects_dtype() -> None:
    with pytest.raises(
        RuntimeError,
        match="changed dtype",
    ):
        (
            relation_priors_module
            ._require_node_relation_tensor(
                "value",
                torch.zeros(
                    NODE_COUNT,
                    RELATION_COUNT,
                    dtype=torch.float64,
                ),
                source_inputs=_inputs(),
                dtype=torch.float32,
            )
        )


def test_require_node_relation_tensor_rejects_nonfinite() -> None:
    value = torch.zeros(
        NODE_COUNT,
        RELATION_COUNT,
    )
    value[0, 0] = float("nan")

    with pytest.raises(
        FloatingPointError,
        match="NaN or infinity",
    ):
        (
            relation_priors_module
            ._require_node_relation_tensor(
                "value",
                value,
                source_inputs=_inputs(),
                dtype=torch.float32,
            )
        )


def test_require_node_relation_tensor_rejects_device_change(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        relation_priors_module,
        "_devices_match",
        lambda first, second: False,
    )

    with pytest.raises(
        RuntimeError,
        match="changed device",
    ):
        (
            relation_priors_module
            ._require_node_relation_tensor(
                "value",
                torch.zeros(
                    NODE_COUNT,
                    RELATION_COUNT,
                ),
                source_inputs=_inputs(),
                dtype=torch.float32,
            )
        )


# =============================================================================
# Representation
# =============================================================================


def test_extra_repr_contains_contract_identity() -> None:
    representation = _builder(
        strength=0.5,
        epsilon=1e-4,
    ).extra_repr()

    assert "strength=0.5" in representation
    assert "epsilon=0.0001" in (
        representation
    )
    assert "parameter_count=0" in (
        representation
    )
    assert "family_pooling=False" in (
        representation
    )


# =============================================================================
# Optional CUDA
# =============================================================================


@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA is unavailable.",
)
def test_cuda_forward_accepts_implicit_device_metadata() -> None:
    inputs = _inputs(
        device=torch.device("cuda")
    )
    output = _builder()(inputs)

    assert output.device.type == "cuda"
    assert (
        output.logit_contribution.device.type
        == "cuda"
    )
    assert output.prior_mean.device.type == (
        "cuda"
    )
    assert output.confidence.device.type == (
        "cuda"
    )


@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA is unavailable.",
)
def test_cuda_regularization_weights_match_cpu() -> None:
    cpu = _builder().regularization_weights(
        _inputs()
    )
    cuda = _builder().regularization_weights(
        _inputs(
            device=torch.device("cuda")
        )
    )

    assert torch.equal(
        cpu,
        cuda.cpu(),
    )


@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA is unavailable.",
)
def test_cuda_rejects_cpu_hazard_lookup_embeddings() -> None:
    inputs = _inputs(
        device=torch.device("cuda"),
        hazard_query=SimpleNamespace(
            source_embedding=(
                FakeHazardEmbeddingLookup(
                    device="cpu"
                )
            )
        ),
    )

    with pytest.raises(
        ValueError,
        match="Hazard lookup embeddings",
    ):
        _builder().resolve_node_hazard_rows(
            inputs
        )


@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA is unavailable.",
)
def test_cuda_rejects_cpu_node_aligned_membership() -> None:
    graph_lookup = (
        FakeHazardEmbeddingLookup(
            device="cuda"
        )
    )
    inputs = _inputs(
        device=torch.device("cuda"),
        hazard_query=SimpleNamespace(
            source_embedding=(
                FakeNodeAlignedHazardEmbeddingLookup(
                    graph_lookup=graph_lookup,
                    node_batch_index=torch.tensor(
                        NODE_BATCH_INDEX,
                        dtype=torch.long,
                        device="cpu",
                    ),
                )
            )
        ),
    )

    with pytest.raises(
        ValueError,
        match="must share one device",
    ):
        _builder().resolve_node_hazard_rows(
            inputs
        )
