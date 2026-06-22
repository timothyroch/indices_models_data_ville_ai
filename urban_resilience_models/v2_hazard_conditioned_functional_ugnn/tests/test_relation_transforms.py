"""
Contract tests for the public relation-transform dispatcher and package exports.

Target repository path:
    urban_resilience_models/
        v2_hazard_conditioned_functional_ugnn/
            tests/
                test_relation_transforms.py

Implementations under test:
    functional_message_passing/
        relation_transforms/
            __init__.py
            shared_transform.py
            per_relation_transform.py
            relation_transforms.py

The shared and per-relation mathematical modules have focused suites of their
own. This suite tests the orchestration boundary:

- public package exports;
- canonical versus implemented mode handling;
- construction from configuration;
- exact compiled-registry compatibility;
- shared/per-relation dispatch;
- metadata-preserving output construction;
- relation-specific and global fingerprints;
- stable dispatcher state-dict structure;
- empty-edge behavior;
- input width, registry, control-mask, device, dtype, and finiteness failures;
- implementation output shape, dtype, device, and finiteness guards;
- finite autograd and optional CUDA behavior.
"""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType, MethodType
from typing import Any

import pytest
import torch
from torch import nn

from urban_resilience_models.v2_hazard_conditioned_functional_ugnn.constants import (
    CANONICAL_RELATION_TRANSFORM_TYPES,
    RELATION_TRANSFORM_PER_RELATION,
    RELATION_TRANSFORM_SHARED,
    V2_0_IMPLEMENTED_RELATION_TRANSFORM_TYPES,
)
from urban_resilience_models.v2_hazard_conditioned_functional_ugnn.functional_message_passing.relation_transforms import (
    PER_RELATION_TRANSFORM_SCHEMA_VERSION,
    RELATION_TRANSFORMS_DISPATCHER_SCHEMA_VERSION,
    SHARED_RELATION_TRANSFORM_SCHEMA_VERSION,
    PerRelationTransform,
    RelationTransforms,
    SharedRelationTransform,
)
from urban_resilience_models.v2_hazard_conditioned_functional_ugnn.functional_message_passing.relation_transforms import (
    relation_transforms as dispatcher_module,
)


HIDDEN_DIM = 4
NODE_COUNT = 6
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
CONTROL_MASK = (
    False,
    False,
    True,
)
RELATION_COUNT = len(RELATION_NAMES)


# =============================================================================
# Controlled upstream contracts
# =============================================================================


@dataclass
class FakeSpecification:
    is_control: bool


@dataclass
class FakeCompiledEntry:
    name: str
    relation_id: int
    specification: FakeSpecification


class FakeCompiledRelationRegistry:
    def __init__(
        self,
        *,
        relation_names: tuple[str, ...] = RELATION_NAMES,
        stable_relation_ids: tuple[int, ...] = STABLE_RELATION_IDS,
        control_mask: tuple[bool, ...] = CONTROL_MASK,
        fingerprint: str = "compiled-registry",
        length_override: int | None = None,
        validation_error: Exception | None = None,
    ) -> None:
        self.relation_names = relation_names
        self.stable_relation_ids = stable_relation_ids
        self.entries = tuple(
            FakeCompiledEntry(
                name=name,
                relation_id=relation_id,
                specification=FakeSpecification(
                    is_control=is_control
                ),
            )
            for name, relation_id, is_control in zip(
                relation_names,
                stable_relation_ids,
                control_mask,
            )
        )
        self._fingerprint = fingerprint
        self.length_override = length_override
        self.validation_error = validation_error
        self.validated = False

    def __len__(self) -> int:
        if self.length_override is not None:
            return self.length_override
        return len(self.entries)

    def validate(self) -> None:
        self.validated = True
        if self.validation_error is not None:
            raise self.validation_error

    def fingerprint(self) -> str:
        return self._fingerprint


class FakeFunctionalMessagePassingConfig:
    def __init__(
        self,
        *,
        relation_transform_type: str = RELATION_TRANSFORM_SHARED,
        enabled: bool = True,
        validation_error: Exception | None = None,
        implementation_error: Exception | None = None,
    ) -> None:
        self.relation_transform_type = relation_transform_type
        self.enabled = enabled
        self.validation_error = validation_error
        self.implementation_error = implementation_error
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


@dataclass
class FakeNodeState:
    fused_state: torch.Tensor


class FakeFunctionalMessagePassingInputs:
    def __init__(
        self,
        *,
        node_state: torch.Tensor,
        source_index: torch.Tensor,
        edge_relation_index: torch.Tensor,
        compiled_relation_registry: FakeCompiledRelationRegistry,
        relation_names: tuple[str, ...] | None = None,
        stable_relation_ids: tuple[int, ...] | None = None,
        control_relation_mask: torch.Tensor | None = None,
        hidden_dim: int | None = None,
        num_edges: int | None = None,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        self.node_state = FakeNodeState(
            fused_state=node_state
        )
        self.source_index = source_index
        self.edge_relation_index = edge_relation_index
        self.compiled_relation_registry = (
            compiled_relation_registry
        )
        self.relation_names = (
            compiled_relation_registry.relation_names
            if relation_names is None
            else relation_names
        )
        self.stable_relation_ids = (
            compiled_relation_registry.stable_relation_ids
            if stable_relation_ids is None
            else stable_relation_ids
        )
        self.control_relation_mask = (
            torch.tensor(
                CONTROL_MASK,
                dtype=torch.bool,
                device=node_state.device,
            )
            if control_relation_mask is None
            else control_relation_mask
        )
        self.hidden_dim = (
            int(node_state.shape[1])
            if hidden_dim is None
            else hidden_dim
        )
        self.num_edges = (
            int(source_index.shape[0])
            if num_edges is None
            else num_edges
        )
        self.device = (
            node_state.device
            if device is None
            else device
        )
        self.dtype = (
            node_state.dtype
            if dtype is None
            else dtype
        )


class FakeRelationTransformOutput:
    def __init__(
        self,
        *,
        transformed_source_state: torch.Tensor,
        source_inputs: object,
        transform_mode: str,
        encoder_architecture_fingerprint: str,
        parameter_fingerprint: str | None = None,
        relation_parameter_fingerprints: Any = None,
    ) -> None:
        self.transformed_source_state = (
            transformed_source_state
        )
        self.source_inputs = source_inputs
        self.transform_mode = transform_mode
        self.encoder_architecture_fingerprint = (
            encoder_architecture_fingerprint
        )
        self.parameter_fingerprint = (
            parameter_fingerprint
        )
        self.relation_parameter_fingerprints = (
            relation_parameter_fingerprints
            if relation_parameter_fingerprints is not None
            else {}
        )


@pytest.fixture(autouse=True)
def _patch_dispatcher_upstream_types(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        dispatcher_module,
        "CompiledRelationRegistry",
        FakeCompiledRelationRegistry,
    )
    monkeypatch.setattr(
        dispatcher_module,
        "FunctionalMessagePassingConfig",
        FakeFunctionalMessagePassingConfig,
    )
    monkeypatch.setattr(
        dispatcher_module,
        "FunctionalMessagePassingInputs",
        FakeFunctionalMessagePassingInputs,
    )
    monkeypatch.setattr(
        dispatcher_module,
        "RelationTransformOutput",
        FakeRelationTransformOutput,
    )


# =============================================================================
# Helpers
# =============================================================================


def _registry(
    *,
    fingerprint: str = "compiled-registry",
    **kwargs: Any,
) -> FakeCompiledRelationRegistry:
    return FakeCompiledRelationRegistry(
        fingerprint=fingerprint,
        **kwargs,
    )


def _dispatcher(
    *,
    mode: str = RELATION_TRANSFORM_SHARED,
    hidden_dim: int = HIDDEN_DIM,
    registry: FakeCompiledRelationRegistry | None = None,
    bias: bool = True,
) -> RelationTransforms:
    return RelationTransforms(
        mode=mode,
        hidden_dim=hidden_dim,
        compiled_relation_registry=(
            _registry()
            if registry is None
            else registry
        ),
        bias=bias,
    )


def _node_state(
    *,
    node_count: int = NODE_COUNT,
    hidden_dim: int = HIDDEN_DIM,
    dtype: torch.dtype = torch.float32,
    device: torch.device | str = "cpu",
    requires_grad: bool = False,
    offset: float = 0.0,
) -> torch.Tensor:
    values = (
        torch.arange(
            node_count * hidden_dim,
            dtype=dtype,
            device=device,
        )
        .reshape(node_count, hidden_dim)
        / 10.0
        + offset
    )
    values.requires_grad_(requires_grad)
    return values


def _source_index(
    *,
    device: torch.device | str = "cpu",
) -> torch.Tensor:
    return torch.tensor(
        [0, 2, 2, 4, 1, 5, 3],
        dtype=torch.long,
        device=device,
    )


def _edge_relation_index(
    *,
    device: torch.device | str = "cpu",
) -> torch.Tensor:
    return torch.tensor(
        [0, 1, 2, 1, 0, 2, 1],
        dtype=torch.long,
        device=device,
    )


def _inputs(
    *,
    registry: FakeCompiledRelationRegistry | None = None,
    node_state: torch.Tensor | None = None,
    source_index: torch.Tensor | None = None,
    edge_relation_index: torch.Tensor | None = None,
    relation_names: tuple[str, ...] | None = None,
    stable_relation_ids: tuple[int, ...] | None = None,
    control_relation_mask: torch.Tensor | None = None,
    hidden_dim: int | None = None,
    num_edges: int | None = None,
    device: torch.device | None = None,
    dtype: torch.dtype | None = None,
) -> FakeFunctionalMessagePassingInputs:
    resolved_registry = (
        _registry()
        if registry is None
        else registry
    )
    resolved_state = (
        _node_state()
        if node_state is None
        else node_state
    )

    return FakeFunctionalMessagePassingInputs(
        node_state=resolved_state,
        source_index=(
            _source_index(
                device=resolved_state.device
            )
            if source_index is None
            else source_index
        ),
        edge_relation_index=(
            _edge_relation_index(
                device=resolved_state.device
            )
            if edge_relation_index is None
            else edge_relation_index
        ),
        compiled_relation_registry=(
            resolved_registry
        ),
        relation_names=relation_names,
        stable_relation_ids=(
            stable_relation_ids
        ),
        control_relation_mask=(
            control_relation_mask
        ),
        hidden_dim=hidden_dim,
        num_edges=num_edges,
        device=device,
        dtype=dtype,
    )


def _configure_shared_identity(
    dispatcher: RelationTransforms,
) -> None:
    implementation = dispatcher.implementation
    assert isinstance(
        implementation,
        SharedRelationTransform,
    )

    with torch.no_grad():
        implementation.linear.weight.copy_(
            torch.eye(
                dispatcher.hidden_dim,
                dtype=implementation.dtype,
                device=implementation.device,
            )
        )
        if implementation.linear.bias is not None:
            implementation.linear.bias.zero_()


def _configure_per_relation_maps(
    dispatcher: RelationTransforms,
) -> None:
    implementation = dispatcher.implementation
    assert isinstance(
        implementation,
        PerRelationTransform,
    )

    with torch.no_grad():
        for relation_index in range(
            implementation.relation_count
        ):
            linear = (
                implementation
                .module_for_relation_index(
                    relation_index
                )
            )
            linear.weight.copy_(
                torch.eye(
                    dispatcher.hidden_dim,
                    dtype=linear.weight.dtype,
                    device=linear.weight.device,
                )
                * float(
                    relation_index + 1
                )
            )
            if linear.bias is not None:
                linear.bias.fill_(
                    float(relation_index)
                )


def _canonical_unimplemented_mode() -> str | None:
    for mode in (
        CANONICAL_RELATION_TRANSFORM_TYPES
    ):
        if mode not in (
            V2_0_IMPLEMENTED_RELATION_TRANSFORM_TYPES
        ):
            return mode
    return None


# =============================================================================
# Package exports and schema identities
# =============================================================================


def test_package_exports_expected_classes() -> None:
    assert (
        SharedRelationTransform.__name__
        == "SharedRelationTransform"
    )
    assert (
        PerRelationTransform.__name__
        == "PerRelationTransform"
    )
    assert (
        RelationTransforms.__name__
        == "RelationTransforms"
    )


@pytest.mark.parametrize(
    "version",
    (
        SHARED_RELATION_TRANSFORM_SCHEMA_VERSION,
        PER_RELATION_TRANSFORM_SCHEMA_VERSION,
        RELATION_TRANSFORMS_DISPATCHER_SCHEMA_VERSION,
    ),
)
def test_schema_versions_are_nonempty(
    version: str,
) -> None:
    assert isinstance(version, str)
    assert version.strip()


def test_package_classes_are_real_module_types() -> None:
    assert issubclass(
        SharedRelationTransform,
        nn.Module,
    )
    assert issubclass(
        PerRelationTransform,
        nn.Module,
    )
    assert issubclass(
        RelationTransforms,
        nn.Module,
    )


# =============================================================================
# Mode validation
# =============================================================================


@pytest.mark.parametrize(
    "mode",
    (
        RELATION_TRANSFORM_SHARED,
        RELATION_TRANSFORM_PER_RELATION,
    ),
)
def test_constructor_accepts_implemented_modes(
    mode: str,
) -> None:
    dispatcher = _dispatcher(
        mode=mode
    )

    assert dispatcher.mode == mode


def test_constructor_strips_mode_whitespace() -> None:
    dispatcher = _dispatcher(
        mode=f"  {RELATION_TRANSFORM_SHARED}  "
    )

    assert dispatcher.mode == (
        RELATION_TRANSFORM_SHARED
    )


@pytest.mark.parametrize(
    "mode",
    (
        "",
        " ",
        "\t",
    ),
)
def test_constructor_rejects_blank_mode(
    mode: str,
) -> None:
    with pytest.raises(
        ValueError,
        match="non-empty string",
    ):
        _dispatcher(mode=mode)


@pytest.mark.parametrize(
    "mode",
    (
        None,
        1,
        True,
        object(),
    ),
)
def test_constructor_rejects_nonstring_mode(
    mode: Any,
) -> None:
    with pytest.raises(
        TypeError,
        match="must be a string",
    ):
        _dispatcher(mode=mode)


def test_constructor_rejects_unknown_mode() -> None:
    with pytest.raises(
        ValueError,
        match="Unknown relation transform mode",
    ):
        _dispatcher(
            mode="unknown_transform"
        )


def test_constructor_rejects_canonical_unimplemented_mode() -> None:
    mode = _canonical_unimplemented_mode()

    if mode is None:
        pytest.skip(
            "No canonical unimplemented relation-transform mode exists."
        )

    with pytest.raises(
        NotImplementedError,
        match="canonical but not implemented",
    ):
        _dispatcher(mode=mode)


def test_canonical_vocabulary_contains_implemented_modes() -> None:
    assert set(
        V2_0_IMPLEMENTED_RELATION_TRANSFORM_TYPES
    ).issubset(
        set(CANONICAL_RELATION_TRANSFORM_TYPES)
    )
    assert RELATION_TRANSFORM_SHARED in (
        V2_0_IMPLEMENTED_RELATION_TRANSFORM_TYPES
    )
    assert RELATION_TRANSFORM_PER_RELATION in (
        V2_0_IMPLEMENTED_RELATION_TRANSFORM_TYPES
    )


# =============================================================================
# Constructor and registry contract
# =============================================================================


def test_shared_constructor_contract() -> None:
    registry = _registry()
    dispatcher = _dispatcher(
        mode=RELATION_TRANSFORM_SHARED,
        registry=registry,
        bias=True,
    )

    assert registry.validated
    assert dispatcher.hidden_dim == HIDDEN_DIM
    assert dispatcher.bias is True
    assert dispatcher.input_dim == HIDDEN_DIM
    assert dispatcher.output_dim == HIDDEN_DIM
    assert dispatcher.relation_count == RELATION_COUNT
    assert dispatcher.relation_names == RELATION_NAMES
    assert (
        dispatcher.stable_relation_ids
        == STABLE_RELATION_IDS
    )
    assert (
        dispatcher.control_relation_mask
        == CONTROL_MASK
    )
    assert (
        dispatcher
        .compiled_relation_registry_fingerprint
        == registry.fingerprint()
    )
    assert isinstance(
        dispatcher.implementation,
        SharedRelationTransform,
    )
    assert dispatcher.is_shared
    assert not dispatcher.is_per_relation


def test_per_relation_constructor_contract() -> None:
    registry = _registry()
    dispatcher = _dispatcher(
        mode=RELATION_TRANSFORM_PER_RELATION,
        registry=registry,
        bias=False,
    )

    assert registry.validated
    assert dispatcher.bias is False
    assert isinstance(
        dispatcher.implementation,
        PerRelationTransform,
    )
    assert dispatcher.is_per_relation
    assert not dispatcher.is_shared
    assert (
        dispatcher
        .implementation
        .relation_names
        == RELATION_NAMES
    )
    assert (
        dispatcher
        .implementation
        .stable_relation_ids
        == STABLE_RELATION_IDS
    )
    assert (
        dispatcher
        .implementation
        .control_relation_mask
        == CONTROL_MASK
    )


def test_implementation_is_registered_under_stable_name() -> None:
    dispatcher = _dispatcher()

    assert tuple(
        dict(dispatcher.named_children())
    ) == ("implementation",)
    assert (
        dispatcher._modules[
            "implementation"
        ]
        is dispatcher.implementation
    )


@pytest.mark.parametrize(
    "hidden_dim",
    (
        0,
        -1,
        True,
        1.5,
    ),
)
def test_constructor_rejects_invalid_hidden_dim(
    hidden_dim: Any,
) -> None:
    with pytest.raises(
        ValueError,
        match="positive integer",
    ):
        _dispatcher(
            hidden_dim=hidden_dim
        )


@pytest.mark.parametrize(
    "bias",
    (
        0,
        1,
        "true",
        None,
    ),
)
def test_constructor_rejects_nonboolean_bias(
    bias: Any,
) -> None:
    with pytest.raises(
        TypeError,
        match="Boolean",
    ):
        _dispatcher(bias=bias)


def test_constructor_rejects_wrong_registry_type() -> None:
    with pytest.raises(
        TypeError,
        match="CompiledRelationRegistry",
    ):
        RelationTransforms(
            mode=RELATION_TRANSFORM_SHARED,
            hidden_dim=HIDDEN_DIM,
            compiled_relation_registry=object(),  # type: ignore[arg-type]
        )


def test_constructor_propagates_registry_validation_error() -> None:
    registry = _registry(
        validation_error=RuntimeError(
            "registry invalid"
        )
    )

    with pytest.raises(
        RuntimeError,
        match="registry invalid",
    ):
        _dispatcher(registry=registry)


def test_constructor_rejects_empty_registry() -> None:
    registry = _registry(
        relation_names=(),
        stable_relation_ids=(),
        control_mask=(),
        length_override=0,
    )

    with pytest.raises(
        ValueError,
        match="At least one compiled relation",
    ):
        _dispatcher(registry=registry)


def test_constructor_rejects_relation_name_id_length_mismatch() -> None:
    registry = _registry(
        relation_names=(
            "spatial",
            "temporal",
        ),
        stable_relation_ids=(100,),
        control_mask=(False,),
        length_override=1,
    )

    with pytest.raises(
        ValueError,
        match="names and stable IDs",
    ):
        _dispatcher(registry=registry)


def test_constructor_rejects_metadata_entry_length_mismatch() -> None:
    registry = _registry(
        relation_names=(
            "spatial",
            "temporal",
        ),
        stable_relation_ids=(
            100,
            200,
        ),
        control_mask=(False,),
        length_override=2,
    )

    with pytest.raises(
        ValueError,
        match="metadata must align",
    ):
        _dispatcher(registry=registry)


@pytest.mark.parametrize(
    "relation_names",
    (
        ("spatial", ""),
        ("spatial", " "),
        ("spatial", "spatial"),
    ),
)
def test_constructor_rejects_invalid_relation_names(
    relation_names: tuple[str, ...],
) -> None:
    registry = _registry(
        relation_names=relation_names,
        stable_relation_ids=tuple(
            range(
                100,
                100 + len(
                    relation_names
                ),
            )
        ),
        control_mask=tuple(
            False
            for _ in relation_names
        ),
    )

    with pytest.raises(ValueError):
        _dispatcher(registry=registry)


def test_constructor_rejects_noninteger_stable_id() -> None:
    registry = _registry(
        stable_relation_ids=(
            100,
            "200",  # type: ignore[arg-type]
            900,
        )
    )

    with pytest.raises(
        TypeError,
        match="must be an integer",
    ):
        _dispatcher(registry=registry)


def test_constructor_rejects_negative_stable_id() -> None:
    registry = _registry(
        stable_relation_ids=(
            100,
            -1,
            900,
        )
    )

    with pytest.raises(
        ValueError,
        match="nonnegative",
    ):
        _dispatcher(registry=registry)


def test_constructor_rejects_duplicate_stable_ids() -> None:
    registry = _registry(
        stable_relation_ids=(
            100,
            100,
            900,
        )
    )

    with pytest.raises(
        ValueError,
        match="must be unique",
    ):
        _dispatcher(registry=registry)


def test_constructor_rejects_control_metadata_length_mismatch() -> None:
    registry = _registry(
        relation_names=(
            "spatial",
            "temporal",
            "control",
        ),
        stable_relation_ids=(
            100,
            200,
            900,
        ),
        control_mask=(
            False,
            True,
        ),
        length_override=2,
    )

    with pytest.raises(
        ValueError,
        match="metadata must align",
    ):
        _dispatcher(registry=registry)


def test_device_dtype_and_parameter_counts_follow_implementation() -> None:
    shared = _dispatcher(
        mode=RELATION_TRANSFORM_SHARED
    )
    per_relation = _dispatcher(
        mode=RELATION_TRANSFORM_PER_RELATION
    )

    assert shared.device == torch.device(
        "cpu"
    )
    assert shared.dtype == torch.float32
    assert shared.parameter_count == (
        shared
        .implementation
        .parameter_count
    )
    assert (
        shared.trainable_parameter_count
        == shared.parameter_count
    )

    assert per_relation.parameter_count == (
        per_relation
        .implementation
        .parameter_count
    )
    assert (
        per_relation.parameter_count
        > shared.parameter_count
    )

    shared = shared.double()
    assert shared.dtype == torch.float64


# =============================================================================
# Construction from configuration
# =============================================================================


@pytest.mark.parametrize(
    "mode",
    (
        RELATION_TRANSFORM_SHARED,
        RELATION_TRANSFORM_PER_RELATION,
    ),
)
def test_from_config_builds_selected_mode(
    mode: str,
) -> None:
    config = FakeFunctionalMessagePassingConfig(
        relation_transform_type=mode,
        enabled=True,
    )
    registry = _registry()

    dispatcher = (
        RelationTransforms.from_config(
            config=config,
            hidden_dim=HIDDEN_DIM,
            compiled_relation_registry=(
                registry
            ),
            bias=False,
        )
    )

    assert config.validate_calls == 1
    assert (
        config.assert_implemented_calls
        == 1
    )
    assert dispatcher.mode == mode
    assert dispatcher.bias is False


def test_from_disabled_config_does_not_assert_implemented() -> None:
    config = FakeFunctionalMessagePassingConfig(
        relation_transform_type=(
            RELATION_TRANSFORM_SHARED
        ),
        enabled=False,
    )

    dispatcher = (
        RelationTransforms.from_config(
            config=config,
            hidden_dim=HIDDEN_DIM,
            compiled_relation_registry=(
                _registry()
            ),
        )
    )

    assert config.validate_calls == 1
    assert (
        config.assert_implemented_calls
        == 0
    )
    assert dispatcher.mode == (
        RELATION_TRANSFORM_SHARED
    )


def test_from_config_rejects_wrong_type() -> None:
    with pytest.raises(
        TypeError,
        match="FunctionalMessagePassingConfig",
    ):
        RelationTransforms.from_config(
            config=object(),  # type: ignore[arg-type]
            hidden_dim=HIDDEN_DIM,
            compiled_relation_registry=(
                _registry()
            ),
        )


def test_from_config_propagates_validation_error() -> None:
    config = FakeFunctionalMessagePassingConfig(
        validation_error=RuntimeError(
            "config invalid"
        )
    )

    with pytest.raises(
        RuntimeError,
        match="config invalid",
    ):
        RelationTransforms.from_config(
            config=config,
            hidden_dim=HIDDEN_DIM,
            compiled_relation_registry=(
                _registry()
            ),
        )


def test_from_config_propagates_implementation_error() -> None:
    config = FakeFunctionalMessagePassingConfig(
        implementation_error=(
            NotImplementedError(
                "config mode unavailable"
            )
        )
    )

    with pytest.raises(
        NotImplementedError,
        match="config mode unavailable",
    ):
        RelationTransforms.from_config(
            config=config,
            hidden_dim=HIDDEN_DIM,
            compiled_relation_registry=(
                _registry()
            ),
        )


# =============================================================================
# Input compatibility validation
# =============================================================================


def test_transform_rejects_wrong_input_type() -> None:
    with pytest.raises(
        TypeError,
        match="FunctionalMessagePassingInputs",
    ):
        _dispatcher().transform_tensor(
            object()  # type: ignore[arg-type]
        )


def test_transform_rejects_hidden_width_mismatch() -> None:
    dispatcher = _dispatcher()

    with pytest.raises(
        ValueError,
        match="node-state width differs",
    ):
        dispatcher.transform_tensor(
            _inputs(
                hidden_dim=HIDDEN_DIM + 1
            )
        )


def test_transform_rejects_relation_order_mismatch() -> None:
    dispatcher = _dispatcher()

    with pytest.raises(
        ValueError,
        match="relation ordering differs",
    ):
        dispatcher.transform_tensor(
            _inputs(
                relation_names=(
                    "temporal_lag",
                    "spatial_adjacency",
                    "random_placebo",
                )
            )
        )


def test_transform_rejects_stable_id_mismatch() -> None:
    dispatcher = _dispatcher()

    with pytest.raises(
        ValueError,
        match="stable relation IDs differ",
    ):
        dispatcher.transform_tensor(
            _inputs(
                stable_relation_ids=(
                    100,
                    201,
                    900,
                )
            )
        )


def test_transform_rejects_registry_fingerprint_mismatch() -> None:
    dispatcher = _dispatcher(
        registry=_registry(
            fingerprint="registry-a"
        )
    )

    with pytest.raises(
        ValueError,
        match="different compiled relation registry",
    ):
        dispatcher.transform_tensor(
            _inputs(
                registry=_registry(
                    fingerprint="registry-b"
                )
            )
        )


def test_transform_rejects_control_mask_mismatch() -> None:
    dispatcher = _dispatcher()

    with pytest.raises(
        ValueError,
        match="control-relation metadata differs",
    ):
        dispatcher.transform_tensor(
            _inputs(
                control_relation_mask=torch.tensor(
                    [False, True, True],
                    dtype=torch.bool,
                )
            )
        )


def test_transform_rejects_input_parameter_dtype_mismatch() -> None:
    dispatcher = _dispatcher()
    state = _node_state(
        dtype=torch.float64
    )

    with pytest.raises(
        ValueError,
        match="dtype must match",
    ):
        dispatcher.transform_tensor(
            _inputs(
                node_state=state,
                dtype=torch.float64,
            )
        )


def test_double_dispatcher_accepts_float64() -> None:
    dispatcher = _dispatcher().double()
    state = _node_state(
        dtype=torch.float64
    )

    output = dispatcher.transform_tensor(
        _inputs(
            node_state=state,
            dtype=torch.float64,
        )
    )

    assert output.dtype == torch.float64


# =============================================================================
# Shared dispatch
# =============================================================================


def test_shared_transform_tensor_matches_implementation() -> None:
    dispatcher = _dispatcher(
        mode=RELATION_TRANSFORM_SHARED,
        bias=False,
    )
    _configure_shared_identity(
        dispatcher
    )
    inputs = _inputs()

    observed = dispatcher.transform_tensor(
        inputs
    )
    expected = (
        inputs
        .node_state
        .fused_state[
            inputs.source_index
        ]
    )

    assert torch.equal(
        observed,
        expected,
    )


def test_shared_forward_constructs_metadata_output() -> None:
    dispatcher = _dispatcher(
        mode=RELATION_TRANSFORM_SHARED
    )
    inputs = _inputs()

    output = dispatcher(inputs)

    assert isinstance(
        output,
        FakeRelationTransformOutput,
    )
    assert output.source_inputs is inputs
    assert output.transform_mode == (
        RELATION_TRANSFORM_SHARED
    )
    assert output.transformed_source_state.shape == (
        inputs.num_edges,
        HIDDEN_DIM,
    )
    assert (
        output.encoder_architecture_fingerprint
        == dispatcher.architecture_fingerprint()
    )
    assert (
        output.parameter_fingerprint
        == dispatcher.parameter_fingerprint()
    )
    assert isinstance(
        output.relation_parameter_fingerprints,
        MappingProxyType,
    )
    assert dict(
        output.relation_parameter_fingerprints
    ) == {}


def test_shared_relation_fingerprint_maps_are_empty_and_read_only() -> None:
    dispatcher = _dispatcher(
        mode=RELATION_TRANSFORM_SHARED
    )

    architecture = (
        dispatcher
        .relation_architecture_fingerprints()
    )
    parameters = (
        dispatcher
        .relation_parameter_fingerprints()
    )

    assert isinstance(
        architecture,
        MappingProxyType,
    )
    assert isinstance(
        parameters,
        MappingProxyType,
    )
    assert dict(architecture) == {}
    assert dict(parameters) == {}

    with pytest.raises(TypeError):
        architecture[
            "x"
        ] = "y"  # type: ignore[index]


# =============================================================================
# Per-relation dispatch
# =============================================================================


def test_per_relation_transform_tensor_matches_distinct_maps() -> None:
    dispatcher = _dispatcher(
        mode=(
            RELATION_TRANSFORM_PER_RELATION
        ),
        bias=False,
    )
    _configure_per_relation_maps(
        dispatcher
    )
    inputs = _inputs()

    output = dispatcher.transform_tensor(
        inputs
    )
    gathered = (
        inputs
        .node_state
        .fused_state[
            inputs.source_index
        ]
    )
    expected = torch.stack(
        [
            (
                float(
                    relation_index.item()
                    + 1
                )
                * gathered[edge_index]
            )
            for edge_index, relation_index
            in enumerate(
                inputs.edge_relation_index
            )
        ]
    )

    assert torch.equal(
        output,
        expected,
    )


def test_per_relation_forward_constructs_relation_fingerprints() -> None:
    dispatcher = _dispatcher(
        mode=(
            RELATION_TRANSFORM_PER_RELATION
        )
    )
    inputs = _inputs()

    output = dispatcher(inputs)

    assert output.source_inputs is inputs
    assert output.transform_mode == (
        RELATION_TRANSFORM_PER_RELATION
    )
    assert tuple(
        output.relation_parameter_fingerprints
    ) == RELATION_NAMES
    assert dict(
        output.relation_parameter_fingerprints
    ) == dict(
        dispatcher
        .relation_parameter_fingerprints()
    )


def test_per_relation_fingerprint_maps_are_read_only() -> None:
    dispatcher = _dispatcher(
        mode=(
            RELATION_TRANSFORM_PER_RELATION
        )
    )

    architecture = (
        dispatcher
        .relation_architecture_fingerprints()
    )
    parameters = (
        dispatcher
        .relation_parameter_fingerprints()
    )

    assert isinstance(
        architecture,
        MappingProxyType,
    )
    assert isinstance(
        parameters,
        MappingProxyType,
    )
    assert tuple(architecture) == (
        RELATION_NAMES
    )
    assert tuple(parameters) == (
        RELATION_NAMES
    )

    with pytest.raises(TypeError):
        parameters[
            "x"
        ] = "y"  # type: ignore[index]


def test_mutating_one_relation_changes_only_its_reported_fingerprint() -> None:
    dispatcher = _dispatcher(
        mode=(
            RELATION_TRANSFORM_PER_RELATION
        )
    )
    implementation = dispatcher.implementation
    assert isinstance(
        implementation,
        PerRelationTransform,
    )

    before = dict(
        dispatcher
        .relation_parameter_fingerprints()
    )

    with torch.no_grad():
        implementation.module_for_relation(
            "temporal_lag"
        ).weight[0, 0] += 1.0

    after = dict(
        dispatcher
        .relation_parameter_fingerprints()
    )

    assert (
        before["temporal_lag"]
        != after["temporal_lag"]
    )
    assert (
        before["spatial_adjacency"]
        == after["spatial_adjacency"]
    )
    assert (
        before["random_placebo"]
        == after["random_placebo"]
    )


# =============================================================================
# Empty-edge behavior
# =============================================================================


@pytest.mark.parametrize(
    "mode",
    (
        RELATION_TRANSFORM_SHARED,
        RELATION_TRANSFORM_PER_RELATION,
    ),
)
def test_transform_supports_zero_edges(
    mode: str,
) -> None:
    dispatcher = _dispatcher(
        mode=mode
    )
    state = _node_state(
        node_count=0
    )
    inputs = _inputs(
        node_state=state,
        source_index=torch.empty(
            0,
            dtype=torch.long,
        ),
        edge_relation_index=torch.empty(
            0,
            dtype=torch.long,
        ),
        num_edges=0,
    )

    output = dispatcher(inputs)

    assert output.transformed_source_state.shape == (
        0,
        HIDDEN_DIM,
    )
    assert output.transformed_source_state.dtype == (
        dispatcher.dtype
    )
    assert output.transformed_source_state.device == (
        dispatcher.device
    )


# =============================================================================
# Implementation output guards
# =============================================================================


class WrongShapeSharedTransform(
    SharedRelationTransform
):
    def forward(
        self,
        node_state: torch.Tensor,
        source_index: torch.Tensor,
    ) -> torch.Tensor:
        return node_state.new_zeros(
            (
                int(source_index.shape[0]),
                self.hidden_dim + 1,
            )
        )


class WrongDtypeSharedTransform(
    SharedRelationTransform
):
    def forward(
        self,
        node_state: torch.Tensor,
        source_index: torch.Tensor,
    ) -> torch.Tensor:
        return node_state.new_zeros(
            (
                int(source_index.shape[0]),
                self.hidden_dim,
            ),
            dtype=torch.float64,
        )


class NonfiniteSharedTransform(
    SharedRelationTransform
):
    def forward(
        self,
        node_state: torch.Tensor,
        source_index: torch.Tensor,
    ) -> torch.Tensor:
        result = node_state.new_zeros(
            (
                int(source_index.shape[0]),
                self.hidden_dim,
            )
        )
        if result.numel() > 0:
            result[0, 0] = float("nan")
        return result


def test_dispatcher_rejects_wrong_implementation_shape() -> None:
    dispatcher = _dispatcher(
        mode=RELATION_TRANSFORM_SHARED
    )
    dispatcher._modules[
        "implementation"
    ] = WrongShapeSharedTransform(
        hidden_dim=HIDDEN_DIM
    )

    with pytest.raises(
        RuntimeError,
        match="returned shape",
    ):
        dispatcher.transform_tensor(
            _inputs()
        )


def test_dispatcher_rejects_wrong_implementation_dtype() -> None:
    dispatcher = _dispatcher(
        mode=RELATION_TRANSFORM_SHARED
    )
    dispatcher._modules[
        "implementation"
    ] = WrongDtypeSharedTransform(
        hidden_dim=HIDDEN_DIM
    )

    with pytest.raises(
        RuntimeError,
        match="changed dtype",
    ):
        dispatcher.transform_tensor(
            _inputs()
        )


def test_dispatcher_rejects_nonfinite_implementation_output() -> None:
    dispatcher = _dispatcher(
        mode=RELATION_TRANSFORM_SHARED
    )
    dispatcher._modules[
        "implementation"
    ] = NonfiniteSharedTransform(
        hidden_dim=HIDDEN_DIM
    )

    with pytest.raises(
        FloatingPointError,
        match="NaN or infinity",
    ):
        dispatcher.transform_tensor(
            _inputs()
        )


def test_implementation_property_rejects_invalid_registered_type() -> None:
    dispatcher = _dispatcher()
    dispatcher._modules[
        "implementation"
    ] = nn.Identity()

    with pytest.raises(
        RuntimeError,
        match="invalid type",
    ):
        _ = dispatcher.implementation


def test_shared_mode_rejects_per_relation_implementation() -> None:
    dispatcher = _dispatcher(
        mode=RELATION_TRANSFORM_SHARED
    )
    dispatcher._modules[
        "implementation"
    ] = PerRelationTransform(
        hidden_dim=HIDDEN_DIM,
        relation_names=RELATION_NAMES,
        stable_relation_ids=(
            STABLE_RELATION_IDS
        ),
        control_relation_mask=(
            CONTROL_MASK
        ),
    )

    with pytest.raises(
        RuntimeError,
        match="Shared mode has an invalid",
    ):
        dispatcher.transform_tensor(
            _inputs()
        )


def test_per_relation_mode_rejects_shared_implementation() -> None:
    dispatcher = _dispatcher(
        mode=(
            RELATION_TRANSFORM_PER_RELATION
        )
    )
    dispatcher._modules[
        "implementation"
    ] = SharedRelationTransform(
        hidden_dim=HIDDEN_DIM
    )

    with pytest.raises(
        RuntimeError,
        match="Per-relation mode has an invalid",
    ):
        dispatcher.transform_tensor(
            _inputs()
        )


# =============================================================================
# Architecture and parameter identity
# =============================================================================


@pytest.mark.parametrize(
    "mode",
    (
        RELATION_TRANSFORM_SHARED,
        RELATION_TRANSFORM_PER_RELATION,
    ),
)
def test_implementation_architecture_dict_matches_implementation(
    mode: str,
) -> None:
    dispatcher = _dispatcher(
        mode=mode
    )

    assert (
        dispatcher
        .implementation_architecture_dict()
        == dispatcher
        .implementation
        .architecture_dict()
    )


@pytest.mark.parametrize(
    "mode",
    (
        RELATION_TRANSFORM_SHARED,
        RELATION_TRANSFORM_PER_RELATION,
    ),
)
def test_architecture_dict_contains_dispatch_contract(
    mode: str,
) -> None:
    dispatcher = _dispatcher(
        mode=mode
    )
    architecture = (
        dispatcher.architecture_dict()
    )

    assert architecture[
        "schema_version"
    ] == (
        RELATION_TRANSFORMS_DISPATCHER_SCHEMA_VERSION
    )
    assert architecture["mode"] == mode
    assert architecture[
        "hidden_dim"
    ] == HIDDEN_DIM
    assert architecture["bias"] is True
    assert architecture[
        "relation_count"
    ] == RELATION_COUNT
    assert architecture[
        "relation_names"
    ] == list(
        RELATION_NAMES
    )
    assert architecture[
        "stable_relation_ids"
    ] == list(
        STABLE_RELATION_IDS
    )
    assert architecture[
        "control_relation_mask"
    ] == list(
        CONTROL_MASK
    )
    assert architecture[
        "compiled_relation_registry_fingerprint"
    ] == "compiled-registry"
    assert architecture[
        "implementation_type"
    ] == type(
        dispatcher.implementation
    ).__name__
    assert architecture[
        "implementation_architecture"
    ] == (
        dispatcher
        .implementation
        .architecture_dict()
    )
    assert architecture[
        "output_schema"
    ] == "RelationTransformOutput"


@pytest.mark.parametrize(
    "mode",
    (
        RELATION_TRANSFORM_SHARED,
        RELATION_TRANSFORM_PER_RELATION,
    ),
)
def test_architecture_fingerprint_is_stable(
    mode: str,
) -> None:
    first = _dispatcher(mode=mode)
    second = _dispatcher(mode=mode)

    assert first.architecture_dict() == (
        second.architecture_dict()
    )
    assert first.architecture_fingerprint() == (
        second.architecture_fingerprint()
    )


def test_architecture_fingerprint_differs_by_mode() -> None:
    assert (
        _dispatcher(
            mode=RELATION_TRANSFORM_SHARED
        ).architecture_fingerprint()
        != _dispatcher(
            mode=RELATION_TRANSFORM_PER_RELATION
        ).architecture_fingerprint()
    )


@pytest.mark.parametrize(
    "builder",
    (
        lambda: _dispatcher(
            hidden_dim=5
        ),
        lambda: _dispatcher(
            bias=False
        ),
        lambda: _dispatcher(
            registry=_registry(
                fingerprint="other"
            )
        ),
        lambda: _dispatcher(
            registry=_registry(
                control_mask=(
                    False,
                    True,
                    True,
                )
            )
        ),
    ),
)
def test_architecture_fingerprint_changes_with_contract(
    builder: Any,
) -> None:
    assert (
        _dispatcher()
        .architecture_fingerprint()
        != builder()
        .architecture_fingerprint()
    )


@pytest.mark.parametrize(
    "mode",
    (
        RELATION_TRANSFORM_SHARED,
        RELATION_TRANSFORM_PER_RELATION,
    ),
)
def test_parameter_fingerprint_is_reproducible_under_seed(
    mode: str,
) -> None:
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(123)
        first = _dispatcher(mode=mode)

    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(123)
        second = _dispatcher(mode=mode)

    assert first.parameter_fingerprint() == (
        second.parameter_fingerprint()
    )


@pytest.mark.parametrize(
    "mode",
    (
        RELATION_TRANSFORM_SHARED,
        RELATION_TRANSFORM_PER_RELATION,
    ),
)
def test_parameter_mutation_changes_parameter_not_architecture_fingerprint(
    mode: str,
) -> None:
    dispatcher = _dispatcher(mode=mode)
    architecture = (
        dispatcher.architecture_fingerprint()
    )
    before = (
        dispatcher.parameter_fingerprint()
    )

    with torch.no_grad():
        next(
            dispatcher.parameters()
        )[0, 0] += 1.0

    assert dispatcher.parameter_fingerprint() != (
        before
    )
    assert dispatcher.architecture_fingerprint() == (
        architecture
    )


# =============================================================================
# State dict and finite parameter checks
# =============================================================================


def test_shared_state_dict_keys_are_stable() -> None:
    dispatcher = _dispatcher(
        mode=RELATION_TRANSFORM_SHARED,
        bias=True,
    )

    assert tuple(
        dispatcher.state_dict()
    ) == (
        "implementation.linear.weight",
        "implementation.linear.bias",
    )


def test_per_relation_state_dict_keys_are_stable() -> None:
    dispatcher = _dispatcher(
        mode=RELATION_TRANSFORM_PER_RELATION,
        bias=True,
    )

    assert tuple(
        dispatcher.state_dict()
    ) == (
        "implementation.relation_transforms.relation_0000_id_100.weight",
        "implementation.relation_transforms.relation_0000_id_100.bias",
        "implementation.relation_transforms.relation_0001_id_200.weight",
        "implementation.relation_transforms.relation_0001_id_200.bias",
        "implementation.relation_transforms.relation_0002_id_900.weight",
        "implementation.relation_transforms.relation_0002_id_900.bias",
    )


@pytest.mark.parametrize(
    "mode",
    (
        RELATION_TRANSFORM_SHARED,
        RELATION_TRANSFORM_PER_RELATION,
    ),
)
def test_state_dict_round_trip_preserves_output(
    mode: str,
) -> None:
    source = _dispatcher(mode=mode)
    target = _dispatcher(mode=mode)
    target.load_state_dict(
        source.state_dict(),
        strict=True,
    )
    inputs = _inputs()

    assert torch.equal(
        source.transform_tensor(inputs),
        target.transform_tensor(inputs),
    )
    assert source.parameter_fingerprint() == (
        target.parameter_fingerprint()
    )


def test_strict_state_dict_load_rejects_different_mode() -> None:
    shared = _dispatcher(
        mode=RELATION_TRANSFORM_SHARED
    )
    per_relation = _dispatcher(
        mode=RELATION_TRANSFORM_PER_RELATION
    )

    with pytest.raises(RuntimeError):
        per_relation.load_state_dict(
            shared.state_dict(),
            strict=True,
        )


@pytest.mark.parametrize(
    "mode",
    (
        RELATION_TRANSFORM_SHARED,
        RELATION_TRANSFORM_PER_RELATION,
    ),
)
def test_assert_finite_parameters_passes(
    mode: str,
) -> None:
    _dispatcher(
        mode=mode
    ).assert_finite_parameters()


@pytest.mark.parametrize(
    "mode",
    (
        RELATION_TRANSFORM_SHARED,
        RELATION_TRANSFORM_PER_RELATION,
    ),
)
@pytest.mark.parametrize(
    "bad_value",
    (
        float("nan"),
        float("inf"),
        float("-inf"),
    ),
)
def test_assert_finite_parameters_detects_corruption(
    mode: str,
    bad_value: float,
) -> None:
    dispatcher = _dispatcher(mode=mode)

    with torch.no_grad():
        next(
            dispatcher.parameters()
        )[0, 0] = bad_value

    with pytest.raises(
        ValueError,
        match="NaN or infinity",
    ):
        dispatcher.assert_finite_parameters()


# =============================================================================
# Autograd
# =============================================================================


@pytest.mark.parametrize(
    "mode",
    (
        RELATION_TRANSFORM_SHARED,
        RELATION_TRANSFORM_PER_RELATION,
    ),
)
def test_backward_reaches_node_state_and_parameters(
    mode: str,
) -> None:
    dispatcher = _dispatcher(mode=mode)
    state = _node_state(
        requires_grad=True
    )
    inputs = _inputs(
        node_state=state
    )

    dispatcher(
        inputs
    ).transformed_source_state.square().mean().backward()

    assert state.grad is not None
    assert bool(
        torch.isfinite(
            state.grad
        ).all().item()
    )

    for parameter in (
        dispatcher.parameters()
    ):
        assert parameter.grad is not None
        assert bool(
            torch.isfinite(
                parameter.grad
            ).all().item()
        )


@pytest.mark.parametrize(
    "mode",
    (
        RELATION_TRANSFORM_SHARED,
        RELATION_TRANSFORM_PER_RELATION,
    ),
)
def test_transform_gradcheck(
    mode: str,
) -> None:
    dispatcher = _dispatcher(
        mode=mode
    ).double()
    state = torch.randn(
        NODE_COUNT,
        HIDDEN_DIM,
        dtype=torch.float64,
        requires_grad=True,
    )
    inputs = _inputs(
        node_state=state,
        dtype=torch.float64,
    )

    assert torch.autograd.gradcheck(
        lambda tensor: dispatcher.transform_tensor(
            _inputs(
                node_state=tensor,
                registry=inputs.compiled_relation_registry,
                source_index=inputs.source_index,
                edge_relation_index=(
                    inputs.edge_relation_index
                ),
                dtype=torch.float64,
            )
        ),
        (state,),
        eps=1e-6,
        atol=1e-5,
        rtol=1e-3,
    )


# =============================================================================
# Representation
# =============================================================================


@pytest.mark.parametrize(
    "mode",
    (
        RELATION_TRANSFORM_SHARED,
        RELATION_TRANSFORM_PER_RELATION,
    ),
)
def test_extra_repr_contains_contract_identity(
    mode: str,
) -> None:
    dispatcher = _dispatcher(
        mode=mode,
        bias=False,
    )
    representation = (
        dispatcher.extra_repr()
    )

    assert mode in representation
    assert (
        f"hidden_dim={HIDDEN_DIM}"
        in representation
    )
    assert (
        f"relation_count={RELATION_COUNT}"
        in representation
    )
    assert (
        "control_relation_count=1"
        in representation
    )
    assert "bias=False" in representation


# =============================================================================
# Optional CUDA
# =============================================================================


@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA is unavailable.",
)
def test_input_parameter_device_mismatch_is_rejected() -> None:
    dispatcher = _dispatcher().cuda()
    inputs = _inputs(
        node_state=_node_state(
            device="cpu"
        ),
        source_index=_source_index(
            device="cpu"
        ),
        edge_relation_index=(
            _edge_relation_index(
                device="cpu"
            )
        ),
    )

    with pytest.raises(
        ValueError,
        match="must share one device",
    ):
        dispatcher.transform_tensor(
            inputs
        )


@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA is unavailable.",
)
@pytest.mark.parametrize(
    "mode",
    (
        RELATION_TRANSFORM_SHARED,
        RELATION_TRANSFORM_PER_RELATION,
    ),
)
def test_cuda_forward_matches_cpu_after_state_copy(
    mode: str,
) -> None:
    cpu_dispatcher = _dispatcher(
        mode=mode
    )
    cuda_dispatcher = _dispatcher(
        mode=mode
    ).cuda()
    cuda_dispatcher.load_state_dict(
        cpu_dispatcher.state_dict(),
        strict=True,
    )

    cpu_state = _node_state()
    cpu_inputs = _inputs(
        node_state=cpu_state
    )
    cuda_state = cpu_state.cuda()
    cuda_inputs = _inputs(
        node_state=cuda_state,
        source_index=_source_index(
            device="cuda"
        ),
        edge_relation_index=(
            _edge_relation_index(
                device="cuda"
            )
        ),
        control_relation_mask=torch.tensor(
            CONTROL_MASK,
            dtype=torch.bool,
            device="cuda",
        ),
    )

    cpu_output = (
        cpu_dispatcher
        .transform_tensor(
            cpu_inputs
        )
    )
    cuda_output = (
        cuda_dispatcher
        .transform_tensor(
            cuda_inputs
        )
        .cpu()
    )

    assert torch.allclose(
        cpu_output,
        cuda_output,
        atol=1e-6,
        rtol=1e-6,
    )


@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA is unavailable.",
)
@pytest.mark.parametrize(
    "mode",
    (
        RELATION_TRANSFORM_SHARED,
        RELATION_TRANSFORM_PER_RELATION,
    ),
)
def test_cuda_backward_is_finite(
    mode: str,
) -> None:
    dispatcher = _dispatcher(
        mode=mode
    ).cuda()
    state = _node_state(
        device="cuda",
        requires_grad=True,
    )
    inputs = _inputs(
        node_state=state,
        source_index=_source_index(
            device="cuda"
        ),
        edge_relation_index=(
            _edge_relation_index(
                device="cuda"
            )
        ),
        control_relation_mask=torch.tensor(
            CONTROL_MASK,
            dtype=torch.bool,
            device="cuda",
        ),
    )

    dispatcher(
        inputs
    ).transformed_source_state.square().mean().backward()

    assert state.grad is not None
    assert bool(
        torch.isfinite(
            state.grad
        ).all().item()
    )

    for parameter in (
        dispatcher.parameters()
    ):
        assert parameter.grad is not None
        assert bool(
            torch.isfinite(
                parameter.grad
            ).all().item()
        )
