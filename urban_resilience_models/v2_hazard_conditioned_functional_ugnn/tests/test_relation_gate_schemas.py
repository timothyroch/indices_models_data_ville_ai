"""
Contract tests for relation-gate schema objects.

Target repository path:
    urban_resilience_models/
        v2_hazard_conditioned_functional_ugnn/
            tests/
                test_relation_gate_schemas.py

Implementation under test:
    functional_message_passing/
        relation_family_gate/
            schemas.py

The bounded V2.0 trainable gate axis is the exact compiled relation axis
``R``. Semantic relation-family metadata is diagnostic and must never replace,
collapse, or reorder exact relation identities.

This suite tests:

- exact relation-axis metadata and fingerprints;
- optional relation-family alignment;
- compatibility checks against functional-message-passing inputs;
- neural target-node relation logits;
- node-aligned prior logit contributions and provenance traces;
- sigmoid activation outputs;
- shape, dtype, device, range, finiteness, and lineage validation;
- immutable metadata and deterministic fingerprints;
- autograd preservation;
- semantic CUDA-device comparison.
"""

from __future__ import annotations

from dataclasses import FrozenInstanceError
from types import MappingProxyType, SimpleNamespace
from typing import Any
from unittest.mock import patch

import pytest
import torch

from urban_resilience_models.v2_hazard_conditioned_functional_ugnn.constants import (
    RELATION_GATE_ACTIVATION_SIGMOID,
    RELATION_GATE_SCOPE_TARGET_NODE,
)
from urban_resilience_models.v2_hazard_conditioned_functional_ugnn.functional_message_passing.relation_family_gate import (
    schemas as schemas_module,
)
from urban_resilience_models.v2_hazard_conditioned_functional_ugnn.functional_message_passing.relation_family_gate.schemas import (
    GATE_ACTIVATION_OUTPUT_SCHEMA_VERSION,
    GATE_NETWORK_OUTPUT_SCHEMA_VERSION,
    RELATION_GATE_AXIS_SCHEMA_VERSION,
    RELATION_PRIOR_CONTRIBUTION_SCHEMA_VERSION,
    GateActivationOutput,
    GateNetworkOutput,
    RelationGateAxis,
    RelationGateOutput,
    RelationPriorContribution,
)


NODE_COUNT = 4
RELATION_COUNT = 3
FAMILY_COUNT = 2

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
    "compiled-relation-priors"
)


# =============================================================================
# Controlled upstream contracts
# =============================================================================


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


class FakeCompiledPriors:
    def __init__(
        self,
        fingerprint: str = (
            COMPILED_PRIOR_FINGERPRINT
        ),
    ) -> None:
        self._fingerprint = fingerprint

    def fingerprint(self) -> str:
        return self._fingerprint


class FakeFunctionalMessagePassingInputs:
    def __init__(
        self,
        *,
        num_nodes: int = NODE_COUNT,
        relation_names: tuple[str, ...] = (
            RELATION_NAMES
        ),
        stable_relation_ids: tuple[int, ...] = (
            STABLE_RELATION_IDS
        ),
        control_relation_mask: (
            torch.Tensor | None
        ) = None,
        compiled_registry_fingerprint: str = (
            COMPILED_REGISTRY_FINGERPRINT
        ),
        relation_families: object | None = ...,
        compiled_relation_priors: object | None = ...,
        dtype: torch.dtype = torch.float32,
        device: torch.device | str = "cpu",
        lineage_fingerprint: str = (
            "functional-message-passing-inputs"
        ),
    ) -> None:
        resolved_device = torch.device(
            device
        )

        self.num_nodes = num_nodes
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

        self.control_relation_mask = (
            torch.tensor(
                CONTROL_MASK_VALUES,
                dtype=torch.bool,
                device=resolved_device,
            )
            if control_relation_mask is None
            else control_relation_mask
        )

        self.compiled_relation_registry = (
            FakeCompiledRegistry(
                compiled_registry_fingerprint
            )
        )

        if relation_families is ...:
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
        else:
            self.relation_families = (
                relation_families
            )

        if compiled_relation_priors is ...:
            self.compiled_relation_priors = (
                FakeCompiledPriors()
            )
        else:
            self.compiled_relation_priors = (
                compiled_relation_priors
            )

        self._lineage_fingerprint = (
            lineage_fingerprint
        )

    def lineage_fingerprint(self) -> str:
        return self._lineage_fingerprint


@pytest.fixture(autouse=True)
def _patch_input_contract(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
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
    relation_names: tuple[str, ...] = (
        RELATION_NAMES
    ),
    stable_relation_ids: tuple[int, ...] = (
        STABLE_RELATION_IDS
    ),
    control_relation_mask: (
        torch.Tensor | None
    ) = None,
    compiled_relation_registry_fingerprint: str = (
        COMPILED_REGISTRY_FINGERPRINT
    ),
    family_names: tuple[str, ...] = (
        FAMILY_NAMES
    ),
    stable_family_ids: tuple[int, ...] = (
        STABLE_FAMILY_IDS
    ),
    relation_family_index_by_relation: (
        torch.Tensor | None
    ) = None,
    source_relation_registry_fingerprint: (
        str | None
    ) = SOURCE_REGISTRY_FINGERPRINT,
    schema_version: str = (
        RELATION_GATE_AXIS_SCHEMA_VERSION
    ),
) -> RelationGateAxis:
    resolved_inputs = (
        _inputs()
        if inputs is None
        else inputs
    )
    resolved_device = resolved_inputs.device

    return RelationGateAxis(
        relation_names=relation_names,
        stable_relation_ids=(
            stable_relation_ids
        ),
        control_relation_mask=(
            torch.tensor(
                CONTROL_MASK_VALUES,
                dtype=torch.bool,
                device=resolved_device,
            )
            if control_relation_mask is None
            else control_relation_mask
        ),
        compiled_relation_registry_fingerprint=(
            compiled_relation_registry_fingerprint
        ),
        family_names=family_names,
        stable_family_ids=(
            stable_family_ids
        ),
        relation_family_index_by_relation=(
            torch.tensor(
                RELATION_FAMILY_INDICES,
                dtype=torch.long,
                device=resolved_device,
            )
            if (
                relation_family_index_by_relation
                is None
                and family_names
            )
            else relation_family_index_by_relation
        ),
        source_relation_registry_fingerprint=(
            source_relation_registry_fingerprint
        ),
        schema_version=schema_version,
    )


def _axis_without_families(
    *,
    inputs: (
        FakeFunctionalMessagePassingInputs
        | None
    ) = None,
) -> RelationGateAxis:
    resolved_inputs = (
        _inputs(
            relation_families=None
        )
        if inputs is None
        else inputs
    )

    return RelationGateAxis(
        relation_names=RELATION_NAMES,
        stable_relation_ids=(
            STABLE_RELATION_IDS
        ),
        control_relation_mask=torch.tensor(
            CONTROL_MASK_VALUES,
            dtype=torch.bool,
            device=resolved_inputs.device,
        ),
        compiled_relation_registry_fingerprint=(
            COMPILED_REGISTRY_FINGERPRINT
        ),
    )


def _logits(
    *,
    dtype: torch.dtype = torch.float32,
    device: torch.device | str = "cpu",
    requires_grad: bool = False,
) -> torch.Tensor:
    value = torch.tensor(
        [
            [0.0, 0.5, -0.5],
            [0.2, -0.2, 0.1],
            [1.0, -1.0, 0.0],
            [0.3, 0.4, 0.5],
        ],
        dtype=dtype,
        device=device,
    )
    value.requires_grad_(
        requires_grad
    )
    return value


def _network_output(
    *,
    inputs: (
        FakeFunctionalMessagePassingInputs
        | None
    ) = None,
    axis: RelationGateAxis | None = None,
    logits: torch.Tensor | None = None,
    scope: str = (
        RELATION_GATE_SCOPE_TARGET_NODE
    ),
    encoder_architecture_fingerprint: str = (
        "gate-network-architecture"
    ),
    parameter_fingerprint: str | None = (
        "gate-network-parameters"
    ),
    input_feature_names: tuple[str, ...] = (
        "node_state",
        "hazard_query",
    ),
    schema_version: str = (
        GATE_NETWORK_OUTPUT_SCHEMA_VERSION
    ),
) -> GateNetworkOutput:
    resolved_inputs = (
        _inputs()
        if inputs is None
        else inputs
    )
    resolved_axis = (
        _axis(
            inputs=resolved_inputs
        )
        if axis is None
        else axis
    )
    resolved_logits = (
        _logits(
            dtype=resolved_inputs.dtype,
            device=resolved_inputs.device,
        )
        if logits is None
        else logits
    )

    return GateNetworkOutput(
        logits=resolved_logits,
        source_inputs=resolved_inputs,
        axis=resolved_axis,
        scope=scope,
        encoder_architecture_fingerprint=(
            encoder_architecture_fingerprint
        ),
        parameter_fingerprint=(
            parameter_fingerprint
        ),
        input_feature_names=(
            input_feature_names
        ),
        schema_version=schema_version,
    )


def _prior_contribution(
    *,
    inputs: (
        FakeFunctionalMessagePassingInputs
        | None
    ) = None,
    axis: RelationGateAxis | None = None,
    logit_contribution: (
        torch.Tensor | None
    ) = None,
    strength: float = 0.25,
    source_compiled_prior_fingerprint: str = (
        COMPILED_PRIOR_FINGERPRINT
    ),
    prior_mean: torch.Tensor | None = None,
    confidence: torch.Tensor | None = None,
    initialization_mask: (
        torch.Tensor | None
    ) = None,
    regularization_mask: (
        torch.Tensor | None
    ) = None,
    resolution_summary: (
        dict[str, int] | None
    ) = None,
    schema_version: str = (
        RELATION_PRIOR_CONTRIBUTION_SCHEMA_VERSION
    ),
) -> RelationPriorContribution:
    resolved_inputs = (
        _inputs()
        if inputs is None
        else inputs
    )
    resolved_axis = (
        _axis(
            inputs=resolved_inputs
        )
        if axis is None
        else axis
    )
    shape = (
        resolved_inputs.num_nodes,
        resolved_inputs.num_relations,
    )
    device = resolved_inputs.device
    dtype = resolved_inputs.dtype

    return RelationPriorContribution(
        logit_contribution=(
            torch.full(
                shape,
                0.1,
                dtype=dtype,
                device=device,
            )
            if logit_contribution is None
            else logit_contribution
        ),
        source_inputs=resolved_inputs,
        axis=resolved_axis,
        strength=strength,
        source_compiled_prior_fingerprint=(
            source_compiled_prior_fingerprint
        ),
        prior_mean=(
            torch.full(
                shape,
                0.6,
                dtype=dtype,
                device=device,
            )
            if prior_mean is None
            else prior_mean
        ),
        confidence=(
            torch.full(
                shape,
                0.8,
                dtype=dtype,
                device=device,
            )
            if confidence is None
            else confidence
        ),
        initialization_mask=(
            torch.ones(
                shape,
                dtype=torch.bool,
                device=device,
            )
            if initialization_mask
            is None
            else initialization_mask
        ),
        regularization_mask=(
            torch.zeros(
                shape,
                dtype=torch.bool,
                device=device,
            )
            if regularization_mask
            is None
            else regularization_mask
        ),
        resolution_summary=(
            {
                "explicit": (
                    resolved_inputs.num_nodes
                    * resolved_inputs.num_relations
                ),
            }
            if resolution_summary is None
            else resolution_summary
        ),
        schema_version=schema_version,
    )


def _activation_output(
    *,
    network: GateNetworkOutput | None = None,
    prior: (
        RelationPriorContribution | None
    ) = ...,
    gate_logits: torch.Tensor | None = None,
    gate_values: torch.Tensor | None = None,
    activation: str = (
        RELATION_GATE_ACTIVATION_SIGMOID
    ),
    encoder_architecture_fingerprint: str = (
        "gate-activation-architecture"
    ),
    parameter_fingerprint: str | None = None,
    schema_version: str = (
        GATE_ACTIVATION_OUTPUT_SCHEMA_VERSION
    ),
) -> GateActivationOutput:
    resolved_network = (
        _network_output()
        if network is None
        else network
    )

    if prior is ...:
        resolved_prior = _prior_contribution(
            inputs=(
                resolved_network
                .source_inputs
            ),
            axis=resolved_network.axis,
        )
    else:
        resolved_prior = prior

    expected_logits = (
        resolved_network.logits
        if resolved_prior is None
        else (
            resolved_network.logits
            + resolved_prior
            .logit_contribution
        )
    )
    resolved_logits = (
        expected_logits
        if gate_logits is None
        else gate_logits
    )
    resolved_values = (
        torch.sigmoid(
            resolved_logits
        )
        if gate_values is None
        else gate_values
    )

    return GateActivationOutput(
        gate_logits=resolved_logits,
        gate_values=resolved_values,
        source_network_output=(
            resolved_network
        ),
        prior_contribution=resolved_prior,
        activation=activation,
        encoder_architecture_fingerprint=(
            encoder_architecture_fingerprint
        ),
        parameter_fingerprint=(
            parameter_fingerprint
        ),
        schema_version=schema_version,
    )


# =============================================================================
# Public identity
# =============================================================================


@pytest.mark.parametrize(
    "version",
    (
        RELATION_GATE_AXIS_SCHEMA_VERSION,
        GATE_NETWORK_OUTPUT_SCHEMA_VERSION,
        RELATION_PRIOR_CONTRIBUTION_SCHEMA_VERSION,
        GATE_ACTIVATION_OUTPUT_SCHEMA_VERSION,
    ),
)
def test_schema_versions_are_nonempty(
    version: str,
) -> None:
    assert isinstance(version, str)
    assert version.strip()


def test_relation_gate_output_is_reexported() -> None:
    assert (
        RelationGateOutput
        is schemas_module.RelationGateOutput
    )


@pytest.mark.parametrize(
    "schema_type",
    (
        RelationGateAxis,
        GateNetworkOutput,
        RelationPriorContribution,
        GateActivationOutput,
    ),
)
def test_schema_types_are_frozen_dataclasses(
    schema_type: type[Any],
) -> None:
    assert hasattr(
        schema_type,
        "__dataclass_fields__",
    )
    assert schema_type.__dataclass_params__.frozen


# =============================================================================
# Semantic device helper
# =============================================================================


def test_devices_match_cpu() -> None:
    assert schemas_module._devices_match(
        "cpu",
        torch.device("cpu"),
    )


def test_devices_match_rejects_different_types() -> None:
    assert not schemas_module._devices_match(
        "cpu",
        "cuda:0",
    )


def test_devices_match_explicit_cuda_indices() -> None:
    assert schemas_module._devices_match(
        "cuda:0",
        "cuda:0",
    )
    assert not schemas_module._devices_match(
        "cuda:0",
        "cuda:1",
    )


def test_devices_match_resolves_implicit_cuda_index() -> None:
    with patch.object(
        torch.cuda,
        "current_device",
        return_value=0,
    ):
        assert schemas_module._devices_match(
            "cuda",
            "cuda:0",
        )
        assert schemas_module._devices_match(
            "cuda:0",
            "cuda",
        )
        assert not schemas_module._devices_match(
            "cuda",
            "cuda:1",
        )


# =============================================================================
# RelationGateAxis — valid construction and properties
# =============================================================================


def test_axis_valid_contract() -> None:
    axis = _axis()

    assert axis.relation_names == (
        RELATION_NAMES
    )
    assert axis.stable_relation_ids == (
        STABLE_RELATION_IDS
    )
    assert axis.num_relations == (
        RELATION_COUNT
    )
    assert axis.family_names == (
        FAMILY_NAMES
    )
    assert axis.stable_family_ids == (
        STABLE_FAMILY_IDS
    )
    assert axis.num_families == FAMILY_COUNT
    assert axis.has_family_metadata
    assert axis.device == torch.device(
        "cpu"
    )
    assert axis.control_relation_names == (
        "random_placebo",
    )


def test_axis_from_inputs_preserves_exact_metadata() -> None:
    inputs = _inputs()
    axis = RelationGateAxis.from_inputs(
        source_inputs=inputs
    )

    assert axis.relation_names == (
        inputs.relation_names
    )
    assert axis.stable_relation_ids == (
        inputs.stable_relation_ids
    )
    assert (
        axis.control_relation_mask
        is inputs.control_relation_mask
    )
    assert (
        axis
        .relation_family_index_by_relation
        is inputs
        .relation_families
        .relation_family_index_by_relation
    )
    assert (
        axis
        .compiled_relation_registry_fingerprint
        == inputs
        .compiled_relation_registry
        .fingerprint()
    )


def test_axis_without_family_metadata() -> None:
    inputs = _inputs(
        relation_families=None
    )
    axis = RelationGateAxis.from_inputs(
        source_inputs=inputs
    )

    assert not axis.has_family_metadata
    assert axis.family_names == ()
    assert axis.stable_family_ids == ()
    assert (
        axis
        .relation_family_index_by_relation
        is None
    )
    assert (
        axis
        .source_relation_registry_fingerprint
        is None
    )


def test_axis_converts_sequence_metadata_to_tuples() -> None:
    axis = RelationGateAxis(
        relation_names=list(  # type: ignore[arg-type]
            RELATION_NAMES
        ),
        stable_relation_ids=list(  # type: ignore[arg-type]
            STABLE_RELATION_IDS
        ),
        control_relation_mask=torch.tensor(
            CONTROL_MASK_VALUES,
            dtype=torch.bool,
        ),
        compiled_relation_registry_fingerprint=(
            COMPILED_REGISTRY_FINGERPRINT
        ),
        family_names=list(  # type: ignore[arg-type]
            FAMILY_NAMES
        ),
        stable_family_ids=list(  # type: ignore[arg-type]
            STABLE_FAMILY_IDS
        ),
        relation_family_index_by_relation=torch.tensor(
            RELATION_FAMILY_INDICES,
            dtype=torch.long,
        ),
        source_relation_registry_fingerprint=(
            SOURCE_REGISTRY_FINGERPRINT
        ),
    )

    assert isinstance(
        axis.relation_names,
        tuple,
    )
    assert isinstance(
        axis.stable_relation_ids,
        tuple,
    )
    assert isinstance(
        axis.family_names,
        tuple,
    )
    assert isinstance(
        axis.stable_family_ids,
        tuple,
    )


def test_axis_is_frozen() -> None:
    axis = _axis()

    with pytest.raises(
        FrozenInstanceError
    ):
        axis.relation_names = ()  # type: ignore[misc]


def test_axis_from_inputs_rejects_wrong_type() -> None:
    with pytest.raises(
        TypeError,
        match="FunctionalMessagePassingInputs",
    ):
        RelationGateAxis.from_inputs(
            source_inputs=object()  # type: ignore[arg-type]
        )


# =============================================================================
# RelationGateAxis — constructor validation
# =============================================================================


@pytest.mark.parametrize(
    "relation_names",
    (
        (),
        ("",),
        (" ",),
        ("a", "a"),
    ),
)
def test_axis_rejects_invalid_relation_names(
    relation_names: tuple[str, ...],
) -> None:
    ids = tuple(
        range(len(relation_names))
    )
    mask = torch.zeros(
        len(relation_names),
        dtype=torch.bool,
    )

    with pytest.raises(ValueError):
        RelationGateAxis(
            relation_names=relation_names,
            stable_relation_ids=ids,
            control_relation_mask=mask,
            compiled_relation_registry_fingerprint=(
                COMPILED_REGISTRY_FINGERPRINT
            ),
        )


@pytest.mark.parametrize(
    "stable_relation_ids",
    (
        (100, -1, 900),
        (100, 100, 900),
    ),
)
def test_axis_rejects_invalid_stable_relation_ids(
    stable_relation_ids: tuple[int, ...],
) -> None:
    with pytest.raises(ValueError):
        _axis(
            stable_relation_ids=(
                stable_relation_ids
            )
        )


@pytest.mark.parametrize(
    "stable_relation_ids",
    (
        (100, True, 900),
        (100, "200", 900),
    ),
)
def test_axis_rejects_noninteger_stable_relation_ids(
    stable_relation_ids: Any,
) -> None:
    with pytest.raises(
        TypeError,
        match="must be an integer",
    ):
        _axis(
            stable_relation_ids=(
                stable_relation_ids
            )
        )


def test_axis_rejects_relation_name_id_length_mismatch() -> None:
    with pytest.raises(
        ValueError,
        match="must align",
    ):
        _axis(
            stable_relation_ids=(
                100,
                200,
            )
        )


@pytest.mark.parametrize(
    "mask",
    (
        torch.tensor(
            [0, 0, 1],
            dtype=torch.long,
        ),
        torch.tensor(
            [[False, False, True]],
            dtype=torch.bool,
        ),
        torch.tensor(
            [False, True],
            dtype=torch.bool,
        ),
    ),
)
def test_axis_rejects_invalid_control_mask(
    mask: torch.Tensor,
) -> None:
    with pytest.raises(ValueError):
        _axis(
            control_relation_mask=mask
        )


@pytest.mark.parametrize(
    "fingerprint",
    (
        "",
        " ",
    ),
)
def test_axis_rejects_blank_compiled_registry_fingerprint(
    fingerprint: str,
) -> None:
    with pytest.raises(
        ValueError,
        match="non-empty string",
    ):
        _axis(
            compiled_relation_registry_fingerprint=(
                fingerprint
            )
        )


def test_axis_rejects_partial_family_names_only() -> None:
    with pytest.raises(ValueError):
        RelationGateAxis(
            relation_names=RELATION_NAMES,
            stable_relation_ids=(
                STABLE_RELATION_IDS
            ),
            control_relation_mask=torch.tensor(
                CONTROL_MASK_VALUES,
                dtype=torch.bool,
            ),
            compiled_relation_registry_fingerprint=(
                COMPILED_REGISTRY_FINGERPRINT
            ),
            family_names=FAMILY_NAMES,
        )


def test_axis_rejects_partial_family_ids_only() -> None:
    with pytest.raises(ValueError):
        RelationGateAxis(
            relation_names=RELATION_NAMES,
            stable_relation_ids=(
                STABLE_RELATION_IDS
            ),
            control_relation_mask=torch.tensor(
                CONTROL_MASK_VALUES,
                dtype=torch.bool,
            ),
            compiled_relation_registry_fingerprint=(
                COMPILED_REGISTRY_FINGERPRINT
            ),
            stable_family_ids=(
                STABLE_FAMILY_IDS
            ),
        )


def test_axis_rejects_missing_relation_family_index() -> None:
    with pytest.raises(
        ValueError,
        match="relation_family_index_by_relation",
    ):
        RelationGateAxis(
            relation_names=RELATION_NAMES,
            stable_relation_ids=(
                STABLE_RELATION_IDS
            ),
            control_relation_mask=torch.tensor(
                CONTROL_MASK_VALUES,
                dtype=torch.bool,
            ),
            compiled_relation_registry_fingerprint=(
                COMPILED_REGISTRY_FINGERPRINT
            ),
            family_names=FAMILY_NAMES,
            stable_family_ids=(
                STABLE_FAMILY_IDS
            ),
            source_relation_registry_fingerprint=(
                SOURCE_REGISTRY_FINGERPRINT
            ),
        )


@pytest.mark.parametrize(
    "family_names",
    (
        ("physical", "physical"),
        ("physical", ""),
    ),
)
def test_axis_rejects_invalid_family_names(
    family_names: tuple[str, ...],
) -> None:
    with pytest.raises(ValueError):
        _axis(
            family_names=family_names
        )


@pytest.mark.parametrize(
    "stable_family_ids",
    (
        (10, -1),
        (10, 10),
    ),
)
def test_axis_rejects_invalid_stable_family_ids(
    stable_family_ids: tuple[int, ...],
) -> None:
    with pytest.raises(ValueError):
        _axis(
            stable_family_ids=(
                stable_family_ids
            )
        )


def test_axis_rejects_family_name_id_length_mismatch() -> None:
    with pytest.raises(
        ValueError,
        match="must align",
    ):
        _axis(
            stable_family_ids=(10,)
        )


@pytest.mark.parametrize(
    "indices",
    (
        torch.tensor(
            RELATION_FAMILY_INDICES,
            dtype=torch.int32,
        ),
        torch.tensor(
            [RELATION_FAMILY_INDICES],
            dtype=torch.long,
        ),
        torch.tensor(
            [0, 1],
            dtype=torch.long,
        ),
    ),
)
def test_axis_rejects_invalid_relation_family_index_tensor(
    indices: torch.Tensor,
) -> None:
    with pytest.raises(ValueError):
        _axis(
            relation_family_index_by_relation=(
                indices
            )
        )


@pytest.mark.parametrize(
    "indices",
    (
        torch.tensor(
            [-1, 0, 1],
            dtype=torch.long,
        ),
        torch.tensor(
            [0, 0, FAMILY_COUNT],
            dtype=torch.long,
        ),
    ),
)
def test_axis_rejects_out_of_range_family_indices(
    indices: torch.Tensor,
) -> None:
    with pytest.raises(
        ValueError,
        match="out-of-range",
    ):
        _axis(
            relation_family_index_by_relation=(
                indices
            )
        )


def test_axis_requires_every_family_to_be_represented() -> None:
    with pytest.raises(
        ValueError,
        match="Every declared relation family",
    ):
        _axis(
            relation_family_index_by_relation=torch.tensor(
                [0, 0, 0],
                dtype=torch.long,
            )
        )


def test_axis_rejects_missing_source_registry_fingerprint() -> None:
    with pytest.raises(
        ValueError,
        match="source_relation_registry_fingerprint",
    ):
        _axis(
            source_relation_registry_fingerprint=None
        )


@pytest.mark.parametrize(
    "schema_version",
    (
        "",
        " ",
    ),
)
def test_axis_rejects_blank_schema_version(
    schema_version: str,
) -> None:
    with pytest.raises(
        ValueError,
        match="non-empty string",
    ):
        _axis(
            schema_version=schema_version
        )


# =============================================================================
# RelationGateAxis — input compatibility
# =============================================================================


def test_axis_matches_inputs() -> None:
    inputs = _inputs()
    _axis(
        inputs=inputs
    ).assert_matches_inputs(inputs)


def test_axis_rejects_wrong_input_type() -> None:
    with pytest.raises(
        TypeError,
        match="FunctionalMessagePassingInputs",
    ):
        _axis().assert_matches_inputs(
            object()  # type: ignore[arg-type]
        )


def test_axis_detects_relation_order_mismatch() -> None:
    inputs = _inputs(
        relation_names=(
            "temporal_lag",
            "spatial_adjacency",
            "random_placebo",
        )
    )

    with pytest.raises(
        ValueError,
        match="axis ordering differs",
    ):
        _axis().assert_matches_inputs(
            inputs
        )


def test_axis_detects_stable_relation_id_mismatch() -> None:
    inputs = _inputs(
        stable_relation_ids=(
            100,
            201,
            900,
        )
    )

    with pytest.raises(
        ValueError,
        match="stable relation IDs differ",
    ):
        _axis().assert_matches_inputs(
            inputs
        )


def test_axis_detects_compiled_registry_mismatch() -> None:
    inputs = _inputs(
        compiled_registry_fingerprint=(
            "different-registry"
        )
    )

    with pytest.raises(
        ValueError,
        match="different compiled relation registry",
    ):
        _axis().assert_matches_inputs(
            inputs
        )


def test_axis_detects_control_mask_mismatch() -> None:
    inputs = _inputs(
        control_relation_mask=torch.tensor(
            [False, True, True],
            dtype=torch.bool,
        )
    )

    with pytest.raises(
        ValueError,
        match="control mask differs",
    ):
        _axis().assert_matches_inputs(
            inputs
        )


def test_axis_with_family_metadata_rejects_inputs_without_it() -> None:
    inputs = _inputs(
        relation_families=None
    )

    with pytest.raises(
        ValueError,
        match="axis contains family metadata",
    ):
        _axis().assert_matches_inputs(
            inputs
        )


def test_axis_without_family_metadata_rejects_inputs_with_it() -> None:
    inputs = _inputs()
    axis = _axis_without_families(
        inputs=_inputs(
            relation_families=None
        )
    )

    with pytest.raises(
        ValueError,
        match="Source inputs contain relation-family metadata",
    ):
        axis.assert_matches_inputs(
            inputs
        )


def test_axis_detects_family_order_mismatch() -> None:
    inputs = _inputs()
    inputs.relation_families.family_names = (
        "control",
        "physical",
    )

    with pytest.raises(
        ValueError,
        match="family ordering differs",
    ):
        _axis().assert_matches_inputs(
            inputs
        )


def test_axis_detects_stable_family_id_mismatch() -> None:
    inputs = _inputs()
    inputs.relation_families.stable_family_ids = (
        10,
        91,
    )

    with pytest.raises(
        ValueError,
        match="stable family IDs differ",
    ):
        _axis().assert_matches_inputs(
            inputs
        )


def test_axis_detects_relation_family_alignment_mismatch() -> None:
    inputs = _inputs()
    inputs.relation_families.relation_family_index_by_relation = (
        torch.tensor(
            [0, 1, 1],
            dtype=torch.long,
        )
    )

    with pytest.raises(
        ValueError,
        match="alignment differs",
    ):
        _axis().assert_matches_inputs(
            inputs
        )


def test_axis_detects_source_registry_fingerprint_mismatch() -> None:
    inputs = _inputs()
    inputs.relation_families.source_relation_registry_fingerprint = (
        "different-source-registry"
    )

    with pytest.raises(
        ValueError,
        match="different source relation registry",
    ):
        _axis().assert_matches_inputs(
            inputs
        )


# =============================================================================
# RelationGateAxis — fingerprints
# =============================================================================


def test_axis_semantic_dict_is_stable() -> None:
    axis = _axis()

    assert axis.semantic_dict() == {
        "schema_version": (
            RELATION_GATE_AXIS_SCHEMA_VERSION
        ),
        "relation_names": list(
            RELATION_NAMES
        ),
        "stable_relation_ids": list(
            STABLE_RELATION_IDS
        ),
        "compiled_relation_registry_fingerprint": (
            COMPILED_REGISTRY_FINGERPRINT
        ),
        "family_names": list(
            FAMILY_NAMES
        ),
        "stable_family_ids": list(
            STABLE_FAMILY_IDS
        ),
        "source_relation_registry_fingerprint": (
            SOURCE_REGISTRY_FINGERPRINT
        ),
    }


def test_axis_fingerprints_are_deterministic() -> None:
    first = _axis()
    second = _axis()

    assert first.semantic_fingerprint() == (
        second.semantic_fingerprint()
    )
    assert first.value_fingerprint() == (
        second.value_fingerprint()
    )
    assert first.fingerprint() == (
        second.fingerprint()
    )


def test_axis_semantic_change_changes_semantic_fingerprint() -> None:
    first = _axis()
    second = _axis(
        compiled_relation_registry_fingerprint=(
            "different-registry"
        )
    )

    assert first.semantic_fingerprint() != (
        second.semantic_fingerprint()
    )


def test_axis_control_mask_change_changes_value_fingerprint() -> None:
    first = _axis()
    second = _axis(
        control_relation_mask=torch.tensor(
            [False, True, True],
            dtype=torch.bool,
        )
    )

    assert first.value_fingerprint() != (
        second.value_fingerprint()
    )


def test_axis_family_alignment_change_changes_value_fingerprint() -> None:
    first = _axis()
    second = _axis(
        relation_family_index_by_relation=torch.tensor(
            [0, 1, 1],
            dtype=torch.long,
        )
    )

    assert first.value_fingerprint() != (
        second.value_fingerprint()
    )


# =============================================================================
# GateNetworkOutput — valid contract
# =============================================================================


def test_network_output_valid_contract() -> None:
    output = _network_output()

    assert output.logits.shape == (
        NODE_COUNT,
        RELATION_COUNT,
    )
    assert output.num_nodes == NODE_COUNT
    assert output.num_relations == (
        RELATION_COUNT
    )
    assert output.device == torch.device(
        "cpu"
    )
    assert output.dtype == torch.float32
    assert (
        output.control_relation_mask
        is output.axis.control_relation_mask
    )
    assert output.input_feature_names == (
        "node_state",
        "hazard_query",
    )


def test_network_output_converts_feature_names_to_tuple() -> None:
    output = _network_output(
        input_feature_names=[  # type: ignore[arg-type]
            "node_state",
            "hazard_query",
        ]
    )

    assert isinstance(
        output.input_feature_names,
        tuple,
    )


def test_network_output_is_frozen() -> None:
    output = _network_output()

    with pytest.raises(
        FrozenInstanceError
    ):
        output.scope = "x"  # type: ignore[misc]


def test_network_output_lineage_and_value_fingerprints() -> None:
    output = _network_output()

    assert output.lineage_dict() == {
        "schema_version": (
            GATE_NETWORK_OUTPUT_SCHEMA_VERSION
        ),
        "scope": (
            RELATION_GATE_SCOPE_TARGET_NODE
        ),
        "axis_fingerprint": (
            output.axis.fingerprint()
        ),
        "source_input_lineage_fingerprint": (
            output
            .source_inputs
            .lineage_fingerprint()
        ),
        "encoder_architecture_fingerprint": (
            "gate-network-architecture"
        ),
        "parameter_fingerprint": (
            "gate-network-parameters"
        ),
        "input_feature_names": [
            "node_state",
            "hazard_query",
        ],
    }
    assert output.lineage_fingerprint()
    assert output.value_fingerprint()
    assert output.fingerprint()


def test_network_value_change_changes_only_value_fingerprint() -> None:
    first = _network_output()
    changed_logits = first.logits.clone()
    changed_logits[0, 0] += 1.0
    second = _network_output(
        inputs=first.source_inputs,
        axis=first.axis,
        logits=changed_logits,
    )

    assert first.lineage_fingerprint() == (
        second.lineage_fingerprint()
    )
    assert first.value_fingerprint() != (
        second.value_fingerprint()
    )
    assert first.fingerprint() != (
        second.fingerprint()
    )


# =============================================================================
# GateNetworkOutput — validation
# =============================================================================


def test_network_output_rejects_wrong_source_input_type() -> None:
    with pytest.raises(
        TypeError,
        match="FunctionalMessagePassingInputs",
    ):
        GateNetworkOutput(
            logits=_logits(),
            source_inputs=object(),  # type: ignore[arg-type]
            axis=_axis(),
            scope=(
                RELATION_GATE_SCOPE_TARGET_NODE
            ),
            encoder_architecture_fingerprint="x",
        )


def test_network_output_rejects_wrong_axis_type() -> None:
    with pytest.raises(
        TypeError,
        match="RelationGateAxis",
    ):
        GateNetworkOutput(
            logits=_logits(),
            source_inputs=_inputs(),
            axis=object(),  # type: ignore[arg-type]
            scope=(
                RELATION_GATE_SCOPE_TARGET_NODE
            ),
            encoder_architecture_fingerprint="x",
        )


@pytest.mark.parametrize(
    "logits",
    (
        torch.zeros(
            NODE_COUNT,
            RELATION_COUNT,
            1,
        ),
        torch.zeros(
            NODE_COUNT,
            RELATION_COUNT + 1,
        ),
        torch.zeros(
            NODE_COUNT - 1,
            RELATION_COUNT,
        ),
        torch.zeros(
            NODE_COUNT,
            RELATION_COUNT,
            dtype=torch.long,
        ),
    ),
)
def test_network_output_rejects_invalid_logits(
    logits: torch.Tensor,
) -> None:
    with pytest.raises(ValueError):
        _network_output(
            logits=logits
        )


@pytest.mark.parametrize(
    "bad_value",
    (
        float("nan"),
        float("inf"),
        float("-inf"),
    ),
)
def test_network_output_rejects_nonfinite_logits(
    bad_value: float,
) -> None:
    logits = _logits()
    logits[0, 0] = bad_value

    with pytest.raises(
        ValueError,
        match="finite values",
    ):
        _network_output(
            logits=logits
        )


def test_network_output_rejects_dtype_mismatch() -> None:
    with pytest.raises(
        ValueError,
        match="must use dtype",
    ):
        _network_output(
            logits=_logits(
                dtype=torch.float64
            )
        )


@pytest.mark.parametrize(
    "scope",
    (
        "",
        " ",
        "graph",
        "source_node",
    ),
)
def test_network_output_rejects_invalid_scope(
    scope: str,
) -> None:
    with pytest.raises(ValueError):
        _network_output(
            scope=scope
        )


@pytest.mark.parametrize(
    "fingerprint",
    (
        "",
        " ",
    ),
)
def test_network_output_rejects_blank_architecture_fingerprint(
    fingerprint: str,
) -> None:
    with pytest.raises(ValueError):
        _network_output(
            encoder_architecture_fingerprint=(
                fingerprint
            )
        )


@pytest.mark.parametrize(
    "parameter_fingerprint",
    (
        "",
        " ",
    ),
)
def test_network_output_rejects_blank_optional_parameter_fingerprint(
    parameter_fingerprint: str,
) -> None:
    with pytest.raises(ValueError):
        _network_output(
            parameter_fingerprint=(
                parameter_fingerprint
            )
        )


@pytest.mark.parametrize(
    "feature_names",
    (
        ("node_state", "node_state"),
        ("node_state", ""),
    ),
)
def test_network_output_rejects_invalid_feature_names(
    feature_names: tuple[str, ...],
) -> None:
    with pytest.raises(ValueError):
        _network_output(
            input_feature_names=feature_names
        )


# =============================================================================
# RelationPriorContribution — valid contract
# =============================================================================


def test_prior_contribution_valid_contract() -> None:
    contribution = _prior_contribution()

    assert contribution.logit_contribution.shape == (
        NODE_COUNT,
        RELATION_COUNT,
    )
    assert contribution.device == torch.device(
        "cpu"
    )
    assert contribution.dtype == torch.float32
    assert contribution.strength == 0.25
    assert isinstance(
        contribution.resolution_summary,
        MappingProxyType,
    )
    assert dict(
        contribution.resolution_summary
    ) == {
        "explicit": (
            NODE_COUNT
            * RELATION_COUNT
        ),
    }


def test_prior_resolution_summary_is_immutable_copy() -> None:
    source = {
        "explicit": 12,
    }
    contribution = _prior_contribution(
        resolution_summary=source
    )
    source["explicit"] = 99

    assert dict(
        contribution.resolution_summary
    ) == {
        "explicit": 12,
    }

    with pytest.raises(TypeError):
        contribution.resolution_summary[
            "x"
        ] = 1  # type: ignore[index]


def test_prior_contribution_is_frozen() -> None:
    contribution = _prior_contribution()

    with pytest.raises(
        FrozenInstanceError
    ):
        contribution.strength = 1.0  # type: ignore[misc]


def test_prior_lineage_and_value_fingerprints() -> None:
    contribution = _prior_contribution()

    assert contribution.lineage_dict() == {
        "schema_version": (
            RELATION_PRIOR_CONTRIBUTION_SCHEMA_VERSION
        ),
        "axis_fingerprint": (
            contribution.axis.fingerprint()
        ),
        "source_input_lineage_fingerprint": (
            contribution
            .source_inputs
            .lineage_fingerprint()
        ),
        "source_compiled_prior_fingerprint": (
            COMPILED_PRIOR_FINGERPRINT
        ),
        "strength": 0.25,
        "resolution_summary": {
            "explicit": 12,
        },
    }
    assert contribution.lineage_fingerprint()
    assert contribution.value_fingerprint()
    assert contribution.fingerprint()


def test_zero_strength_accepts_exact_zero_contribution() -> None:
    inputs = _inputs()
    contribution = _prior_contribution(
        inputs=inputs,
        strength=0.0,
        logit_contribution=torch.zeros(
            NODE_COUNT,
            RELATION_COUNT,
            dtype=inputs.dtype,
            device=inputs.device,
        ),
    )

    assert contribution.strength == 0.0


# =============================================================================
# RelationPriorContribution — validation
# =============================================================================


def test_prior_rejects_wrong_source_input_type() -> None:
    with pytest.raises(
        TypeError,
        match="FunctionalMessagePassingInputs",
    ):
        RelationPriorContribution(
            logit_contribution=torch.zeros(
                NODE_COUNT,
                RELATION_COUNT,
            ),
            source_inputs=object(),  # type: ignore[arg-type]
            axis=_axis(),
            strength=0.5,
            source_compiled_prior_fingerprint=(
                COMPILED_PRIOR_FINGERPRINT
            ),
        )


def test_prior_rejects_wrong_axis_type() -> None:
    with pytest.raises(
        TypeError,
        match="RelationGateAxis",
    ):
        RelationPriorContribution(
            logit_contribution=torch.zeros(
                NODE_COUNT,
                RELATION_COUNT,
            ),
            source_inputs=_inputs(),
            axis=object(),  # type: ignore[arg-type]
            strength=0.5,
            source_compiled_prior_fingerprint=(
                COMPILED_PRIOR_FINGERPRINT
            ),
        )


@pytest.mark.parametrize(
    "contribution",
    (
        torch.zeros(
            NODE_COUNT,
            RELATION_COUNT,
            1,
        ),
        torch.zeros(
            NODE_COUNT,
            RELATION_COUNT + 1,
        ),
        torch.zeros(
            NODE_COUNT,
            RELATION_COUNT,
            dtype=torch.long,
        ),
    ),
)
def test_prior_rejects_invalid_logit_contribution(
    contribution: torch.Tensor,
) -> None:
    with pytest.raises(ValueError):
        _prior_contribution(
            logit_contribution=(
                contribution
            )
        )


@pytest.mark.parametrize(
    "bad_value",
    (
        float("nan"),
        float("inf"),
        float("-inf"),
    ),
)
def test_prior_rejects_nonfinite_logit_contribution(
    bad_value: float,
) -> None:
    value = torch.zeros(
        NODE_COUNT,
        RELATION_COUNT,
    )
    value[0, 0] = bad_value

    with pytest.raises(
        ValueError,
        match="finite values",
    ):
        _prior_contribution(
            logit_contribution=value
        )


@pytest.mark.parametrize(
    "strength",
    (
        True,
        "0.5",
        None,
    ),
)
def test_prior_rejects_nonnumeric_strength(
    strength: Any,
) -> None:
    with pytest.raises(
        TypeError,
        match="must be numeric",
    ):
        _prior_contribution(
            strength=strength
        )


@pytest.mark.parametrize(
    "strength",
    (
        -0.1,
        float("nan"),
        float("inf"),
    ),
)
def test_prior_rejects_invalid_strength(
    strength: float,
) -> None:
    with pytest.raises(ValueError):
        _prior_contribution(
            strength=strength
        )


def test_prior_rejects_missing_compiled_prior_artifact() -> None:
    inputs = _inputs(
        compiled_relation_priors=None
    )

    with pytest.raises(
        ValueError,
        match="compiled_relation_priors",
    ):
        _prior_contribution(
            inputs=inputs,
            axis=_axis(
                inputs=inputs
            ),
        )


def test_prior_rejects_compiled_prior_fingerprint_mismatch() -> None:
    with pytest.raises(
        ValueError,
        match="different compiled prior artifact",
    ):
        _prior_contribution(
            source_compiled_prior_fingerprint=(
                "different-prior-artifact"
            )
        )


@pytest.mark.parametrize(
    "field_name",
    (
        "prior_mean",
        "confidence",
    ),
)
@pytest.mark.parametrize(
    "bad_value",
    (
        -0.01,
        1.01,
    ),
)
def test_prior_rejects_probability_trace_out_of_range(
    field_name: str,
    bad_value: float,
) -> None:
    value = torch.full(
        (
            NODE_COUNT,
            RELATION_COUNT,
        ),
        0.5,
    )
    value[0, 0] = bad_value

    kwargs = {
        field_name: value,
    }

    with pytest.raises(
        ValueError,
        match=r"\[0, 1\]",
    ):
        _prior_contribution(
            **kwargs
        )


@pytest.mark.parametrize(
    "field_name",
    (
        "initialization_mask",
        "regularization_mask",
    ),
)
def test_prior_rejects_nonboolean_masks(
    field_name: str,
) -> None:
    kwargs = {
        field_name: torch.zeros(
            NODE_COUNT,
            RELATION_COUNT,
            dtype=torch.long,
        ),
    }

    with pytest.raises(
        ValueError,
        match="torch.bool",
    ):
        _prior_contribution(
            **kwargs
        )


def test_prior_zero_strength_rejects_nonzero_contribution() -> None:
    with pytest.raises(
        ValueError,
        match="exact zero",
    ):
        _prior_contribution(
            strength=0.0
        )


def test_prior_rejects_nonmapping_resolution_summary() -> None:
    with pytest.raises(
        TypeError,
        match="must be a mapping",
    ):
        _prior_contribution(
            resolution_summary=[  # type: ignore[arg-type]
                ("explicit", 12)
            ]
        )


@pytest.mark.parametrize(
    "summary",
    (
        {"": 1},
        {"explicit": -1},
        {"explicit": True},
        {"explicit": 1.5},
    ),
)
def test_prior_rejects_invalid_resolution_summary(
    summary: dict[str, Any],
) -> None:
    with pytest.raises(
        (TypeError, ValueError)
    ):
        _prior_contribution(
            resolution_summary=summary
        )


# =============================================================================
# GateActivationOutput — valid contract
# =============================================================================


def test_activation_output_with_prior() -> None:
    output = _activation_output()

    expected_logits = (
        output
        .source_network_output
        .logits
        + output
        .prior_contribution
        .logit_contribution
    )

    assert torch.equal(
        output.gate_logits,
        expected_logits,
    )
    assert torch.equal(
        output.gate_values,
        torch.sigmoid(expected_logits),
    )
    assert (
        output.source_inputs
        is output
        .source_network_output
        .source_inputs
    )
    assert (
        output.axis
        is output
        .source_network_output
        .axis
    )
    assert output.scope == (
        RELATION_GATE_SCOPE_TARGET_NODE
    )
    assert output.device == torch.device(
        "cpu"
    )
    assert output.dtype == torch.float32
    assert (
        output.control_relation_mask
        is output.axis.control_relation_mask
    )


def test_activation_output_without_prior() -> None:
    network = _network_output(
        inputs=_inputs(
            compiled_relation_priors=None
        )
    )
    output = _activation_output(
        network=network,
        prior=None,
    )

    assert output.prior_contribution is None
    assert torch.equal(
        output.gate_logits,
        network.logits,
    )
    assert torch.equal(
        output.gate_values,
        torch.sigmoid(network.logits),
    )


def test_activation_output_is_frozen() -> None:
    output = _activation_output()

    with pytest.raises(
        FrozenInstanceError
    ):
        output.activation = "x"  # type: ignore[misc]


def test_activation_lineage_and_value_fingerprints() -> None:
    output = _activation_output()

    assert output.lineage_dict() == {
        "schema_version": (
            GATE_ACTIVATION_OUTPUT_SCHEMA_VERSION
        ),
        "activation": (
            RELATION_GATE_ACTIVATION_SIGMOID
        ),
        "scope": (
            RELATION_GATE_SCOPE_TARGET_NODE
        ),
        "source_network_fingerprint": (
            output
            .source_network_output
            .fingerprint()
        ),
        "prior_contribution_fingerprint": (
            output
            .prior_contribution
            .fingerprint()
        ),
        "encoder_architecture_fingerprint": (
            "gate-activation-architecture"
        ),
        "parameter_fingerprint": None,
    }
    assert output.lineage_fingerprint()
    assert output.value_fingerprint()
    assert output.fingerprint()


def test_activation_values_remain_differentiable() -> None:
    inputs = _inputs()
    logits = _logits(
        requires_grad=True
    )
    network = _network_output(
        inputs=inputs,
        axis=_axis(
            inputs=inputs
        ),
        logits=logits,
    )
    output = _activation_output(
        network=network,
        prior=None,
    )

    output.gate_values.sum().backward()

    assert logits.grad is not None
    assert bool(
        torch.isfinite(
            logits.grad
        ).all()
        .item()
    )


# =============================================================================
# GateActivationOutput — validation
# =============================================================================


def test_activation_rejects_wrong_network_output_type() -> None:
    with pytest.raises(
        TypeError,
        match="GateNetworkOutput",
    ):
        GateActivationOutput(
            gate_logits=_logits(),
            gate_values=torch.sigmoid(
                _logits()
            ),
            source_network_output=object(),  # type: ignore[arg-type]
            prior_contribution=None,
            activation=(
                RELATION_GATE_ACTIVATION_SIGMOID
            ),
            encoder_architecture_fingerprint="x",
        )


@pytest.mark.parametrize(
    "field_name",
    (
        "gate_logits",
        "gate_values",
    ),
)
def test_activation_rejects_invalid_shape(
    field_name: str,
) -> None:
    network = _network_output()
    kwargs = {
        field_name: torch.zeros(
            NODE_COUNT,
            RELATION_COUNT + 1,
        ),
    }

    with pytest.raises(ValueError):
        _activation_output(
            network=network,
            **kwargs
        )


@pytest.mark.parametrize(
    "activation",
    (
        "",
        " ",
        "softmax",
        "relu",
    ),
)
def test_activation_rejects_invalid_activation(
    activation: str,
) -> None:
    with pytest.raises(ValueError):
        _activation_output(
            activation=activation
        )


def test_activation_rejects_wrong_prior_type() -> None:
    network = _network_output()
    logits = network.logits

    with pytest.raises(
        TypeError,
        match="RelationPriorContribution",
    ):
        GateActivationOutput(
            gate_logits=logits,
            gate_values=torch.sigmoid(
                logits
            ),
            source_network_output=network,
            prior_contribution=object(),  # type: ignore[arg-type]
            activation=(
                RELATION_GATE_ACTIVATION_SIGMOID
            ),
            encoder_architecture_fingerprint="x",
        )


def test_activation_requires_exact_same_source_inputs_object() -> None:
    network = _network_output()
    other_inputs = _inputs()
    other_axis = _axis(
        inputs=other_inputs
    )
    prior = _prior_contribution(
        inputs=other_inputs,
        axis=other_axis,
    )
    combined = (
        network.logits
        + prior.logit_contribution
    )

    with pytest.raises(
        ValueError,
        match="exact same source_inputs object",
    ):
        GateActivationOutput(
            gate_logits=combined,
            gate_values=torch.sigmoid(
                combined
            ),
            source_network_output=network,
            prior_contribution=prior,
            activation=(
                RELATION_GATE_ACTIVATION_SIGMOID
            ),
            encoder_architecture_fingerprint="x",
        )


def test_activation_rejects_relation_axis_mismatch() -> None:
    inputs = _inputs()
    network_axis = _axis(
        inputs=inputs
    )
    network = _network_output(
        inputs=inputs,
        axis=network_axis,
    )
    prior = _prior_contribution(
        inputs=inputs,
        axis=network_axis,
    )

    changed_axis = RelationGateAxis(
        relation_names=RELATION_NAMES,
        stable_relation_ids=(
            STABLE_RELATION_IDS
        ),
        control_relation_mask=torch.tensor(
            [False, True, True],
            dtype=torch.bool,
        ),
        compiled_relation_registry_fingerprint=(
            COMPILED_REGISTRY_FINGERPRINT
        ),
        family_names=FAMILY_NAMES,
        stable_family_ids=(
            STABLE_FAMILY_IDS
        ),
        relation_family_index_by_relation=torch.tensor(
            RELATION_FAMILY_INDICES,
            dtype=torch.long,
        ),
        source_relation_registry_fingerprint=(
            SOURCE_REGISTRY_FINGERPRINT
        ),
    )

    # The prior constructor already validates its original axis against the
    # source inputs. Mutate only this frozen test instance afterward so the
    # activation schema's own cross-source axis check is exercised directly.
    object.__setattr__(
        prior,
        "axis",
        changed_axis,
    )

    combined = (
        network.logits
        + prior.logit_contribution
    )

    with pytest.raises(
        ValueError,
        match="same relation-gate axis",
    ):
        GateActivationOutput(
            gate_logits=combined,
            gate_values=torch.sigmoid(
                combined
            ),
            source_network_output=network,
            prior_contribution=prior,
            activation=(
                RELATION_GATE_ACTIVATION_SIGMOID
            ),
            encoder_architecture_fingerprint="x",
        )


def test_activation_rejects_incorrect_combined_logits() -> None:
    network = _network_output()
    prior = _prior_contribution(
        inputs=network.source_inputs,
        axis=network.axis,
    )

    with pytest.raises(
        ValueError,
        match="neural logits plus",
    ):
        _activation_output(
            network=network,
            prior=prior,
            gate_logits=network.logits,
        )


def test_activation_rejects_incorrect_sigmoid_values() -> None:
    network = _network_output()

    with pytest.raises(
        ValueError,
        match=r"sigmoid\(gate_logits\)",
    ):
        _activation_output(
            network=network,
            prior=None,
            gate_values=torch.zeros_like(
                network.logits
            ),
        )


@pytest.mark.parametrize(
    "fingerprint",
    (
        "",
        " ",
    ),
)
def test_activation_rejects_blank_architecture_fingerprint(
    fingerprint: str,
) -> None:
    with pytest.raises(ValueError):
        _activation_output(
            encoder_architecture_fingerprint=(
                fingerprint
            )
        )


@pytest.mark.parametrize(
    "parameter_fingerprint",
    (
        "",
        " ",
    ),
)
def test_activation_rejects_blank_parameter_fingerprint(
    parameter_fingerprint: str,
) -> None:
    with pytest.raises(ValueError):
        _activation_output(
            parameter_fingerprint=(
                parameter_fingerprint
            )
        )


# =============================================================================
# Optional CUDA
# =============================================================================


@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA is unavailable.",
)
def test_cuda_axis_and_outputs_accept_implicit_device_metadata() -> None:
    inputs = _inputs(
        device=torch.device("cuda")
    )
    axis = RelationGateAxis.from_inputs(
        source_inputs=inputs
    )
    network = _network_output(
        inputs=inputs,
        axis=axis,
        logits=_logits(
            device="cuda"
        ),
    )
    prior = _prior_contribution(
        inputs=inputs,
        axis=axis,
    )
    activation = _activation_output(
        network=network,
        prior=prior,
    )

    assert axis.device.type == "cuda"
    assert network.device.type == "cuda"
    assert prior.device.type == "cuda"
    assert activation.device.type == "cuda"


@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA is unavailable.",
)
def test_cuda_rejects_cpu_logits_against_cuda_inputs() -> None:
    inputs = _inputs(
        device=torch.device("cuda")
    )
    axis = RelationGateAxis.from_inputs(
        source_inputs=inputs
    )

    with pytest.raises(
        ValueError,
        match="must be on device",
    ):
        _network_output(
            inputs=inputs,
            axis=axis,
            logits=_logits(
                device="cpu"
            ),
        )
