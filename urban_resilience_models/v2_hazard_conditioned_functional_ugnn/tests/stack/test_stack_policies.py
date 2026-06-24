"""
Contract tests for functional-message-passing stack policies.

Target repository path:
    urban_resilience_models/
        v2_hazard_conditioned_functional_ugnn/
            tests/
                stack/
                    test_stack_policies.py

Implementations under test:
    functional_message_passing/
        stack/
            sharing_policy.py
            trace_policy.py

This suite freezes the bounded Phase 3 policy contracts independently from
stack numerical execution, state rebinding, diagnostics, and public stack
assembly.

Covered behavior
----------------
Sharing policy:

- canonical and implemented vocabulary;
- independent and fully shared policy metadata;
- exact module ownership by depth;
- rejection of equal-but-distinct modules under fully shared execution;
- rejection of repeated modules under independent execution;
- rejection of cross-depth exact Parameter aliases;
- rejection of cross-depth shared parameter storage;
- constant hidden width, relation ordering, stable relation IDs, and compiled
  registry lineage;
- factory invocation counts;
- deterministic architecture and parameter fingerprints;
- state-dict registration-prefix metadata;
- post-construction ownership revalidation;
- public aliases and exports.

Trace and retention policy:

- ``none``, ``final_layer``, and ``all_layers`` retention;
- orthogonality between stack retention and one-layer trace detail;
- exact zero-based retained-index contracts;
- per-depth decisions;
- audit mode;
- deterministic execution-contract fingerprints;
- numerical-equivalence declarations;
- conflict detection during policy resolution;
- decision validation;
- public aliases and exports.

The layer objects used by the sharing tests are controlled ``nn.Module``
doubles. They expose exactly the metadata and parameter surfaces consumed by
``sharing_policy.py``; no one-layer numerical behavior is exercised here.
"""

from __future__ import annotations

from dataclasses import FrozenInstanceError
from hashlib import sha256
import json
from typing import Any, Final

import pytest
import torch
from torch import nn

from urban_resilience_models.v2_hazard_conditioned_functional_ugnn.constants import (
    CANONICAL_STACK_RETENTION_POLICIES,
    CANONICAL_STACK_SHARING_POLICIES,
    STACK_RETENTION_ALL_LAYERS,
    STACK_RETENTION_FINAL_LAYER,
    STACK_RETENTION_NONE,
    STACK_SHARING_FULLY_SHARED,
    STACK_SHARING_INDEPENDENT,
    V2_0_IMPLEMENTED_STACK_RETENTION_POLICIES,
    V2_0_IMPLEMENTED_STACK_SHARING_POLICIES,
)
from urban_resilience_models.v2_hazard_conditioned_functional_ugnn.functional_message_passing.layer.schemas import (
    LAYER_TRACE_FULL,
    LAYER_TRACE_NODE,
    LAYER_TRACE_NONE,
    LayerTracePolicy,
)
from urban_resilience_models.v2_hazard_conditioned_functional_ugnn.functional_message_passing.stack import (
    sharing_policy as sharing_module,
)
from urban_resilience_models.v2_hazard_conditioned_functional_ugnn.functional_message_passing.stack import (
    trace_policy as trace_module,
)
from urban_resilience_models.v2_hazard_conditioned_functional_ugnn.functional_message_passing.stack.sharing_policy import (
    FULLY_SHARED_LAYER_STATE_DICT_PREFIX,
    INDEPENDENT_LAYER_STATE_DICT_PREFIX,
    STACK_LAYER_SHARING_PLAN_SCHEMA_VERSION,
    STACK_SHARING_FULLY_SHARED_REQUIRES_EXACT_MODULE_REUSE,
    STACK_SHARING_INDEPENDENT_REQUIRES_DISTINCT_MODULES,
    STACK_SHARING_INDEPENDENT_REQUIRES_DISTINCT_PARAMETERS,
    STACK_SHARING_PARTIAL_SUPPORTED,
    STACK_SHARING_POLICY_AFFECTS_DIAGNOSTICS,
    STACK_SHARING_POLICY_AFFECTS_NUMERICAL_ARCHITECTURE,
    STACK_SHARING_POLICY_AFFECTS_OUTPUT_RETENTION,
    STACK_SHARING_POLICY_AFFECTS_PARAMETER_OWNERSHIP,
    STACK_SHARING_POLICY_AFFECTS_TRACE_DETAIL,
    STACK_SHARING_POLICY_SCHEMA_VERSION,
    STACK_SHARING_SCIENTIFIC_INTERPRETATION,
    LayerSharingPlan,
    SharingPolicy,
    StackLayerSharingPlan,
    StackSharingPolicy,
    assert_stack_sharing_policy_implemented,
    build_fully_shared_stack_layer_sharing_plan,
    build_independent_stack_layer_sharing_plan,
    build_layer_sharing_plan,
    build_layer_sharing_plan_from_factory,
    build_stack_layer_sharing_plan,
    build_stack_layer_sharing_plan_from_factory,
    is_fully_shared_stack_sharing_policy,
    is_independent_stack_sharing_policy,
    layer_for_stack_depth,
    normalize_sharing_policy,
    normalize_stack_sharing_policy,
    resolve_sharing_policy,
    resolve_stack_sharing_policy,
    validate_layer_sharing_plan,
    validate_stack_layer_sharing_plan,
)
from urban_resilience_models.v2_hazard_conditioned_functional_ugnn.functional_message_passing.stack.trace_policy import (
    STACK_AUDIT_MODE_AFFECTS_NUMERICAL_RESULTS,
    STACK_LAYER_TRACE_AFFECTS_NUMERICAL_RESULTS,
    STACK_LAYER_TRACE_POLICY_IS_OUTPUT_CONTRACT,
    STACK_RETENTION_AFFECTS_NUMERICAL_RESULTS,
    STACK_RETENTION_ALL_LAYERS_RETAINS_FINAL_LAYER,
    STACK_RETENTION_FINAL_LAYER_RETAINS_FINAL_LAYER,
    STACK_RETENTION_NONE_RETAINS_FINAL_LAYER,
    STACK_RETENTION_POLICY_IS_OUTPUT_CONTRACT,
    STACK_RETENTION_POLICY_SCHEMA_VERSION,
    STACK_TRACE_DECISION_SCHEMA_VERSION,
    STACK_TRACE_POLICY_IS_NUMERICAL_ARCHITECTURE,
    STACK_TRACE_POLICY_SCHEMA_VERSION,
    STACK_TRACE_POLICY_SCIENTIFIC_INTERPRETATION,
    RetentionPolicy,
    StackRetentionPolicy,
    StackTraceDecision,
    StackTracePolicy,
    TraceDecision,
    TracePolicy,
    assert_stack_retention_policy_implemented,
    assert_stack_trace_policy_matches,
    build_stack_trace_decisions,
    build_stack_trace_policy,
    build_trace_decisions,
    build_trace_policy,
    is_all_layers_stack_retention,
    is_final_layer_stack_retention,
    is_no_stack_retention,
    normalize_retention_policy,
    normalize_stack_retention_policy,
    resolve_retention_policy,
    resolve_stack_retention_policy,
    resolve_stack_trace_policy,
    resolve_trace_policy,
    stack_trace_policies_are_numerically_equivalent,
    validate_retained_layer_indices,
    validate_stack_trace_decisions,
    validate_trace_decisions,
)


HIDDEN_DIM: Final[int] = 4
NUM_LAYERS: Final[int] = 3
RELATION_NAMES: Final[tuple[str, ...]] = (
    "spatial_adjacency",
    "temporal_lag",
    "random_placebo",
)
STABLE_RELATION_IDS: Final[tuple[int, ...]] = (
    100,
    200,
    900,
)
COMPILED_REGISTRY_FINGERPRINT: Final[str] = (
    "compiled-registry"
)


# =============================================================================
# Controlled layer contract
# =============================================================================


def _fingerprint_payload(
    payload: dict[str, Any],
) -> str:
    return sha256(
        json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


class ControlledFunctionalMessagePassingLayer(
    nn.Module
):
    """
    Minimal metadata-bearing layer double for sharing-policy tests.
    """

    def __init__(
        self,
        *,
        tag: str,
        hidden_dim: int = HIDDEN_DIM,
        relation_names: tuple[str, ...] = (
            RELATION_NAMES
        ),
        stable_relation_ids: tuple[int, ...] = (
            STABLE_RELATION_IDS
        ),
        registry_fingerprint: str = (
            COMPILED_REGISTRY_FINGERPRINT
        ),
        architecture_fingerprint: str | None = None,
        parameter_fingerprint_override: str | None = None,
        weight: nn.Parameter | None = None,
        bias: nn.Parameter | None = None,
        fill_value: float = 0.0,
    ) -> None:
        super().__init__()

        self.tag = tag
        self.hidden_dim = hidden_dim
        self.relation_names = relation_names
        self.stable_relation_ids = (
            stable_relation_ids
        )
        self.compiled_relation_registry_fingerprint = (
            registry_fingerprint
        )

        self._architecture_fingerprint = (
            architecture_fingerprint
            if architecture_fingerprint
            is not None
            else _fingerprint_payload(
                {
                    "hidden_dim": hidden_dim,
                    "relation_names": list(
                        relation_names
                    ),
                    "stable_relation_ids": list(
                        stable_relation_ids
                    ),
                    "registry_fingerprint": (
                        registry_fingerprint
                    ),
                }
            )
        )
        self._parameter_fingerprint_override = (
            parameter_fingerprint_override
        )

        resolved_weight = (
            nn.Parameter(
                torch.full(
                    (
                        hidden_dim,
                        hidden_dim,
                    ),
                    fill_value,
                    dtype=torch.float32,
                )
            )
            if weight is None
            else weight
        )
        resolved_bias = (
            nn.Parameter(
                torch.full(
                    (hidden_dim,),
                    fill_value + 0.25,
                    dtype=torch.float32,
                )
            )
            if bias is None
            else bias
        )

        self.weight = resolved_weight
        self.bias = resolved_bias

    def architecture_fingerprint(
        self,
    ) -> str:
        return self._architecture_fingerprint

    def parameter_fingerprint(
        self,
    ) -> str:
        if (
            self._parameter_fingerprint_override
            is not None
        ):
            return (
                self
                ._parameter_fingerprint_override
            )

        values: list[dict[str, Any]] = []

        for name, parameter in (
            self.named_parameters()
        ):
            detached = (
                parameter
                .detach()
                .cpu()
                .to(dtype=torch.float64)
            )
            values.append(
                {
                    "name": name,
                    "shape": list(
                        detached.shape
                    ),
                    "sum": float(
                        detached.sum().item()
                    ),
                    "squared_sum": float(
                        detached.square()
                        .sum()
                        .item()
                    ),
                }
            )

        return _fingerprint_payload(
            {
                "tag": self.tag,
                "parameters": values,
            }
        )


class PropertyFingerprintLayer(
    ControlledFunctionalMessagePassingLayer
):
    """
    Confirms that sharing policy accepts fingerprint properties as well.
    """

    @property
    def architecture_fingerprint(
        self,
    ) -> str:
        return self._architecture_fingerprint

    @property
    def parameter_fingerprint(
        self,
    ) -> str:
        values = tuple(
            float(
                parameter
                .detach()
                .sum()
                .item()
            )
            for parameter
            in self.parameters()
        )
        return _fingerprint_payload(
            {
                "tag": self.tag,
                "parameter_sums": values,
            }
        )


@pytest.fixture(autouse=True)
def _patch_layer_contract(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        sharing_module,
        "FunctionalMessagePassingLayer",
        ControlledFunctionalMessagePassingLayer,
    )


def _layer(
    index: int,
    *,
    hidden_dim: int = HIDDEN_DIM,
    relation_names: tuple[str, ...] = (
        RELATION_NAMES
    ),
    stable_relation_ids: tuple[int, ...] = (
        STABLE_RELATION_IDS
    ),
    registry_fingerprint: str = (
        COMPILED_REGISTRY_FINGERPRINT
    ),
    architecture_fingerprint: str | None = None,
    parameter_fingerprint_override: str | None = None,
    weight: nn.Parameter | None = None,
    bias: nn.Parameter | None = None,
    fill_value: float | None = None,
) -> ControlledFunctionalMessagePassingLayer:
    return ControlledFunctionalMessagePassingLayer(
        tag=f"layer-{index}",
        hidden_dim=hidden_dim,
        relation_names=relation_names,
        stable_relation_ids=(
            stable_relation_ids
        ),
        registry_fingerprint=(
            registry_fingerprint
        ),
        architecture_fingerprint=(
            architecture_fingerprint
        ),
        parameter_fingerprint_override=(
            parameter_fingerprint_override
        ),
        weight=weight,
        bias=bias,
        fill_value=(
            float(index + 1)
            if fill_value is None
            else fill_value
        ),
    )


def _independent_layers(
    count: int = NUM_LAYERS,
) -> tuple[
    ControlledFunctionalMessagePassingLayer,
    ...,
]:
    return tuple(
        _layer(index)
        for index in range(count)
    )


def _independent_plan(
    count: int = NUM_LAYERS,
    *,
    require_uniform_training_mode: bool = True,
) -> StackLayerSharingPlan:
    return build_independent_stack_layer_sharing_plan(
        _independent_layers(
            count
        ),
        require_uniform_training_mode=(
            require_uniform_training_mode
        ),
    )


def _shared_plan(
    count: int = NUM_LAYERS,
    *,
    layer: (
        ControlledFunctionalMessagePassingLayer
        | None
    ) = None,
) -> StackLayerSharingPlan:
    shared = (
        _layer(0)
        if layer is None
        else layer
    )
    return (
        build_fully_shared_stack_layer_sharing_plan(
            shared,
            num_layers=count,
        )
    )


# =============================================================================
# Published identity and vocabulary
# =============================================================================


def test_stack_policy_schema_versions_are_nonempty() -> None:
    versions = (
        STACK_SHARING_POLICY_SCHEMA_VERSION,
        STACK_LAYER_SHARING_PLAN_SCHEMA_VERSION,
        STACK_RETENTION_POLICY_SCHEMA_VERSION,
        STACK_TRACE_POLICY_SCHEMA_VERSION,
        STACK_TRACE_DECISION_SCHEMA_VERSION,
    )

    for version in versions:
        assert isinstance(
            version,
            str,
        )
        assert version.strip()


def test_stack_vocabulary_matches_constants() -> None:
    assert tuple(
        CANONICAL_STACK_SHARING_POLICIES
    ) == (
        STACK_SHARING_INDEPENDENT,
        STACK_SHARING_FULLY_SHARED,
    )
    assert tuple(
        V2_0_IMPLEMENTED_STACK_SHARING_POLICIES
    ) == (
        STACK_SHARING_INDEPENDENT,
        STACK_SHARING_FULLY_SHARED,
    )

    assert tuple(
        CANONICAL_STACK_RETENTION_POLICIES
    ) == (
        STACK_RETENTION_NONE,
        STACK_RETENTION_FINAL_LAYER,
        STACK_RETENTION_ALL_LAYERS,
    )
    assert tuple(
        V2_0_IMPLEMENTED_STACK_RETENTION_POLICIES
    ) == (
        STACK_RETENTION_NONE,
        STACK_RETENTION_FINAL_LAYER,
        STACK_RETENTION_ALL_LAYERS,
    )


def test_sharing_policy_scientific_flags() -> None:
    assert (
        STACK_SHARING_POLICY_AFFECTS_NUMERICAL_ARCHITECTURE
        is True
    )
    assert (
        STACK_SHARING_POLICY_AFFECTS_PARAMETER_OWNERSHIP
        is True
    )
    assert (
        STACK_SHARING_POLICY_AFFECTS_TRACE_DETAIL
        is False
    )
    assert (
        STACK_SHARING_POLICY_AFFECTS_OUTPUT_RETENTION
        is False
    )
    assert (
        STACK_SHARING_POLICY_AFFECTS_DIAGNOSTICS
        is False
    )
    assert (
        STACK_SHARING_INDEPENDENT_REQUIRES_DISTINCT_MODULES
        is True
    )
    assert (
        STACK_SHARING_INDEPENDENT_REQUIRES_DISTINCT_PARAMETERS
        is True
    )
    assert (
        STACK_SHARING_FULLY_SHARED_REQUIRES_EXACT_MODULE_REUSE
        is True
    )
    assert STACK_SHARING_PARTIAL_SUPPORTED is False
    assert isinstance(
        STACK_SHARING_SCIENTIFIC_INTERPRETATION,
        str,
    )
    assert (
        STACK_SHARING_SCIENTIFIC_INTERPRETATION
    )


def test_trace_policy_scientific_flags() -> None:
    assert (
        STACK_RETENTION_AFFECTS_NUMERICAL_RESULTS
        is False
    )
    assert (
        STACK_LAYER_TRACE_AFFECTS_NUMERICAL_RESULTS
        is False
    )
    assert (
        STACK_AUDIT_MODE_AFFECTS_NUMERICAL_RESULTS
        is False
    )
    assert (
        STACK_RETENTION_NONE_RETAINS_FINAL_LAYER
        is False
    )
    assert (
        STACK_RETENTION_FINAL_LAYER_RETAINS_FINAL_LAYER
        is True
    )
    assert (
        STACK_RETENTION_ALL_LAYERS_RETAINS_FINAL_LAYER
        is True
    )
    assert (
        STACK_RETENTION_POLICY_IS_OUTPUT_CONTRACT
        is True
    )
    assert (
        STACK_LAYER_TRACE_POLICY_IS_OUTPUT_CONTRACT
        is True
    )
    assert (
        STACK_TRACE_POLICY_IS_NUMERICAL_ARCHITECTURE
        is False
    )
    assert isinstance(
        STACK_TRACE_POLICY_SCIENTIFIC_INTERPRETATION,
        str,
    )
    assert (
        STACK_TRACE_POLICY_SCIENTIFIC_INTERPRETATION
    )


# =============================================================================
# Sharing-policy normalization
# =============================================================================


@pytest.mark.parametrize(
    "name",
    (
        STACK_SHARING_INDEPENDENT,
        STACK_SHARING_FULLY_SHARED,
    ),
)
def test_normalize_stack_sharing_policy_accepts_implemented_names(
    name: str,
) -> None:
    assert (
        normalize_stack_sharing_policy(
            name
        )
        == name
    )
    assert (
        normalize_stack_sharing_policy(
            f"  {name}\n"
        )
        == name
    )
    assert (
        normalize_sharing_policy(
            name
        )
        == name
    )
    assert_stack_sharing_policy_implemented(
        name
    )


@pytest.mark.parametrize(
    "value",
    (
        None,
        1,
        True,
        (),
        object(),
    ),
)
def test_normalize_stack_sharing_policy_rejects_non_strings(
    value: Any,
) -> None:
    with pytest.raises(
        TypeError,
        match="must be a string",
    ):
        normalize_stack_sharing_policy(
            value  # type: ignore[arg-type]
        )


@pytest.mark.parametrize(
    "value",
    (
        "",
        " ",
        "\n\t",
    ),
)
def test_normalize_stack_sharing_policy_rejects_blank_strings(
    value: str,
) -> None:
    with pytest.raises(
        ValueError,
        match="non-empty",
    ):
        normalize_stack_sharing_policy(
            value
        )


def test_normalize_stack_sharing_policy_rejects_unknown_name() -> None:
    with pytest.raises(
        ValueError,
        match="Unknown stack sharing policy",
    ):
        normalize_stack_sharing_policy(
            "partial"
        )


def test_normalize_stack_sharing_policy_rejects_canonical_unimplemented(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        sharing_module,
        "CANONICAL_STACK_SHARING_POLICIES",
        (
            STACK_SHARING_INDEPENDENT,
            STACK_SHARING_FULLY_SHARED,
            "partial",
        ),
    )
    monkeypatch.setattr(
        sharing_module,
        "V2_0_IMPLEMENTED_STACK_SHARING_POLICIES",
        (
            STACK_SHARING_INDEPENDENT,
            STACK_SHARING_FULLY_SHARED,
        ),
    )

    with pytest.raises(
        NotImplementedError,
        match="canonical but not implemented",
    ):
        normalize_stack_sharing_policy(
            "partial"
        )


# =============================================================================
# StackSharingPolicy
# =============================================================================


def test_stack_sharing_policy_defaults_to_independent() -> None:
    policy = StackSharingPolicy()

    assert policy.name == (
        STACK_SHARING_INDEPENDENT
    )
    assert policy.is_independent
    assert not policy.is_fully_shared
    assert (
        policy.requires_distinct_layer_objects
    )
    assert not (
        policy.requires_exact_layer_reuse
    )


def test_stack_sharing_policy_named_constructors() -> None:
    independent = (
        StackSharingPolicy.independent()
    )
    shared = (
        StackSharingPolicy.fully_shared()
    )

    assert independent.name == (
        STACK_SHARING_INDEPENDENT
    )
    assert shared.name == (
        STACK_SHARING_FULLY_SHARED
    )
    assert not independent.is_fully_shared
    assert shared.is_fully_shared
    assert (
        shared.requires_exact_layer_reuse
    )


@pytest.mark.parametrize(
    (
        "policy",
        "unique_count",
        "owner_mapping",
        "prefixes",
    ),
    (
        (
            StackSharingPolicy.independent(),
            NUM_LAYERS,
            (0, 1, 2),
            (
                "layers.0",
                "layers.1",
                "layers.2",
            ),
        ),
        (
            StackSharingPolicy.fully_shared(),
            1,
            (0, 0, 0),
            ("shared_layer",),
        ),
    ),
)
def test_stack_sharing_policy_expected_ownership(
    policy: StackSharingPolicy,
    unique_count: int,
    owner_mapping: tuple[int, ...],
    prefixes: tuple[str, ...],
) -> None:
    assert (
        policy.expected_unique_layer_count(
            num_layers=NUM_LAYERS
        )
        == unique_count
    )
    assert (
        policy.expected_depth_to_owner_index(
            num_layers=NUM_LAYERS
        )
        == owner_mapping
    )
    assert (
        policy.registration_prefixes(
            num_layers=NUM_LAYERS
        )
        == prefixes
    )


@pytest.mark.parametrize(
    "num_layers",
    (
        0,
        -1,
    ),
)
def test_stack_sharing_policy_rejects_nonpositive_depth(
    num_layers: int,
) -> None:
    policy = StackSharingPolicy()

    with pytest.raises(
        ValueError,
        match="strictly positive",
    ):
        policy.expected_unique_layer_count(
            num_layers=num_layers
        )


@pytest.mark.parametrize(
    "num_layers",
    (
        True,
        1.5,
        "3",
    ),
)
def test_stack_sharing_policy_rejects_noninteger_depth(
    num_layers: Any,
) -> None:
    policy = StackSharingPolicy()

    with pytest.raises(
        TypeError,
        match="must be an integer",
    ):
        policy.expected_depth_to_owner_index(
            num_layers=num_layers
        )


def test_stack_sharing_policy_fingerprints_are_deterministic() -> None:
    first = (
        StackSharingPolicy.independent()
    )
    second = (
        StackSharingPolicy.independent()
    )

    assert first.architecture_dict() == (
        second.architecture_dict()
    )
    assert (
        first.architecture_fingerprint()
        == second.architecture_fingerprint()
    )


def test_stack_sharing_policy_fingerprint_changes_with_policy() -> None:
    independent = (
        StackSharingPolicy.independent()
    )
    shared = (
        StackSharingPolicy.fully_shared()
    )

    assert independent.architecture_dict() != (
        shared.architecture_dict()
    )
    assert (
        independent.architecture_fingerprint()
        != shared.architecture_fingerprint()
    )


def test_stack_sharing_policy_is_frozen() -> None:
    policy = StackSharingPolicy()

    with pytest.raises(
        FrozenInstanceError,
    ):
        policy.name = (  # type: ignore[misc]
            STACK_SHARING_FULLY_SHARED
        )


def test_stack_sharing_policy_rejects_blank_schema_version() -> None:
    with pytest.raises(
        ValueError,
        match="schema_version",
    ):
        StackSharingPolicy(
            schema_version=""
        )


def test_resolve_stack_sharing_policy_preserves_policy_object() -> None:
    policy = (
        StackSharingPolicy.fully_shared()
    )

    assert (
        resolve_stack_sharing_policy(
            policy
        )
        is policy
    )
    assert (
        resolve_sharing_policy(
            policy
        )
        is policy
    )


def test_resolve_stack_sharing_policy_constructs_from_string() -> None:
    policy = (
        resolve_stack_sharing_policy(
            STACK_SHARING_FULLY_SHARED
        )
    )

    assert isinstance(
        policy,
        StackSharingPolicy,
    )
    assert policy.is_fully_shared


@pytest.mark.parametrize(
    "value",
    (
        None,
        1,
        object(),
    ),
)
def test_resolve_stack_sharing_policy_rejects_invalid_type(
    value: Any,
) -> None:
    with pytest.raises(
        TypeError,
        match="string or StackSharingPolicy",
    ):
        resolve_stack_sharing_policy(
            value  # type: ignore[arg-type]
        )


def test_stack_sharing_policy_predicates() -> None:
    assert (
        is_independent_stack_sharing_policy(
            STACK_SHARING_INDEPENDENT
        )
    )
    assert not (
        is_independent_stack_sharing_policy(
            STACK_SHARING_FULLY_SHARED
        )
    )
    assert (
        is_fully_shared_stack_sharing_policy(
            StackSharingPolicy.fully_shared()
        )
    )


def test_sharing_aliases_are_exact() -> None:
    assert SharingPolicy is (
        StackSharingPolicy
    )
    assert LayerSharingPlan is (
        StackLayerSharingPlan
    )
    assert normalize_sharing_policy is (
        normalize_stack_sharing_policy
    )
    assert resolve_sharing_policy is (
        resolve_stack_sharing_policy
    )
    assert build_layer_sharing_plan is (
        build_stack_layer_sharing_plan
    )
    assert (
        build_layer_sharing_plan_from_factory
        is build_stack_layer_sharing_plan_from_factory
    )
    assert validate_layer_sharing_plan is (
        validate_stack_layer_sharing_plan
    )


# =============================================================================
# Independent sharing plans
# =============================================================================


def test_independent_plan_valid_contract() -> None:
    layers = _independent_layers()
    plan = (
        build_independent_stack_layer_sharing_plan(
            layers
        )
    )

    assert plan.policy == (
        StackSharingPolicy.independent()
    )
    assert plan.sharing_policy == (
        STACK_SHARING_INDEPENDENT
    )
    assert plan.num_layers == NUM_LAYERS
    assert plan.layers_by_depth == layers

    for expected, observed in zip(
        layers,
        plan.layers_by_depth,
        strict=True,
    ):
        assert observed is expected

    assert plan.unique_layers == layers
    assert plan.num_unique_layers == (
        NUM_LAYERS
    )
    assert (
        plan.depth_to_unique_layer_index
        == (0, 1, 2)
    )
    assert (
        plan.unique_layer_registration_prefixes
        == (
            "layers.0",
            "layers.1",
            "layers.2",
        )
    )
    assert plan.hidden_dim == HIDDEN_DIM
    assert plan.relation_names == (
        RELATION_NAMES
    )
    assert plan.stable_relation_ids == (
        STABLE_RELATION_IDS
    )
    assert (
        plan.compiled_relation_registry_fingerprint
        == COMPILED_REGISTRY_FINGERPRINT
    )
    assert plan.training is True


def test_independent_plan_depth_resolution() -> None:
    layers = _independent_layers()
    plan = (
        build_independent_stack_layer_sharing_plan(
            layers
        )
    )

    for depth in range(NUM_LAYERS):
        assert (
            plan.layer_for_depth(
                depth
            )
            is layers[depth]
        )
        assert (
            layer_for_stack_depth(
                plan,
                depth,
            )
            is layers[depth]
        )
        assert (
            plan.unique_layer_for_owner_index(
                depth
            )
            is layers[depth]
        )


@pytest.mark.parametrize(
    "depth",
    (
        NUM_LAYERS,
        NUM_LAYERS + 1,
    ),
)
def test_layer_for_depth_rejects_out_of_range(
    depth: int,
) -> None:
    plan = _independent_plan()

    with pytest.raises(
        IndexError,
        match="outside",
    ):
        plan.layer_for_depth(
            depth
        )


@pytest.mark.parametrize(
    "depth",
    (
        -1,
        True,
        1.5,
    ),
)
def test_layer_for_depth_rejects_invalid_depth(
    depth: Any,
) -> None:
    plan = _independent_plan()

    expected = (
        ValueError
        if depth == -1
        else TypeError
    )

    with pytest.raises(expected):
        plan.layer_for_depth(
            depth
        )


def test_independent_plan_architecture_fingerprint_is_deterministic() -> None:
    first = _independent_plan()
    second = _independent_plan()

    assert (
        first.numerical_architecture_dict()
        == second.numerical_architecture_dict()
    )
    assert (
        first.architecture_fingerprint()
        == second.architecture_fingerprint()
    )


def test_plan_architecture_fingerprint_changes_with_sharing_policy() -> None:
    independent = _independent_plan()
    shared = _shared_plan()

    assert (
        independent.architecture_fingerprint()
        != shared.architecture_fingerprint()
    )


def test_independent_plan_parameter_fingerprint_is_deterministic() -> None:
    first = _independent_plan()
    second = _independent_plan()

    assert (
        first.parameter_ownership_dict()
        == second.parameter_ownership_dict()
    )
    assert (
        first.parameter_fingerprint()
        == second.parameter_fingerprint()
    )


def test_plan_parameter_fingerprint_changes_after_parameter_update() -> None:
    plan = _independent_plan()
    before = plan.parameter_fingerprint()

    with torch.no_grad():
        plan.layers_by_depth[1].weight.add_(
            2.0
        )

    after = plan.parameter_fingerprint()

    assert after != before


def test_independent_plan_accepts_fingerprint_properties() -> None:
    layers = tuple(
        PropertyFingerprintLayer(
            tag=f"property-{index}",
            fill_value=float(index + 1),
        )
        for index in range(NUM_LAYERS)
    )

    plan = (
        build_independent_stack_layer_sharing_plan(
            layers
        )
    )

    assert len(
        plan.layer_architecture_fingerprints_by_depth()
    ) == NUM_LAYERS
    assert len(
        plan.layer_parameter_fingerprints_by_depth()
    ) == NUM_LAYERS


def test_independent_plan_rejects_empty_sequence() -> None:
    with pytest.raises(
        ValueError,
        match="at least one",
    ):
        build_independent_stack_layer_sharing_plan(
            ()
        )


def test_independent_plan_rejects_nonsequence() -> None:
    with pytest.raises(
        TypeError,
        match="sequence",
    ):
        build_independent_stack_layer_sharing_plan(
            _layer(0)  # type: ignore[arg-type]
        )


def test_independent_plan_rejects_nonlayer_member() -> None:
    with pytest.raises(
        TypeError,
        match="FunctionalMessagePassingLayer",
    ):
        build_independent_stack_layer_sharing_plan(
            (
                _layer(0),
                object(),  # type: ignore[arg-type]
            )
        )


def test_independent_plan_rejects_repeated_layer_object() -> None:
    first = _layer(0)

    with pytest.raises(
        ValueError,
        match="distinct exact",
    ):
        build_independent_stack_layer_sharing_plan(
            (
                first,
                first,
            )
        )


def test_independent_plan_rejects_exact_parameter_alias() -> None:
    first = _layer(0)
    second = _layer(
        1,
        weight=first.weight,
    )

    with pytest.raises(
        ValueError,
        match="share exact Parameter",
    ):
        build_independent_stack_layer_sharing_plan(
            (
                first,
                second,
            )
        )


def test_independent_plan_rejects_shared_parameter_storage() -> None:
    first = _layer(0)

    shared_storage_parameter = nn.Parameter(
        first.weight.detach()
    )
    assert (
        shared_storage_parameter
        is not first.weight
    )
    assert (
        shared_storage_parameter
        .untyped_storage()
        .data_ptr()
        == first.weight
        .untyped_storage()
        .data_ptr()
    )

    second = _layer(
        1,
        weight=shared_storage_parameter,
    )

    with pytest.raises(
        ValueError,
        match="share nonempty parameter storage",
    ):
        build_independent_stack_layer_sharing_plan(
            (
                first,
                second,
            )
        )


def test_independent_plan_accepts_equal_values_in_distinct_storage() -> None:
    first = _layer(
        0,
        fill_value=1.0,
    )
    second = _layer(
        1,
        fill_value=1.0,
    )

    assert torch.equal(
        first.weight,
        second.weight,
    )
    assert first.weight is not (
        second.weight
    )
    assert (
        first.weight
        .untyped_storage()
        .data_ptr()
        != second.weight
        .untyped_storage()
        .data_ptr()
    )

    plan = (
        build_independent_stack_layer_sharing_plan(
            (
                first,
                second,
            )
        )
    )

    assert plan.num_unique_layers == 2


def test_independent_plan_rejects_mixed_training_modes_by_default() -> None:
    first = _layer(0)
    second = _layer(1)
    second.eval()

    with pytest.raises(
        ValueError,
        match="train/eval mode",
    ):
        build_independent_stack_layer_sharing_plan(
            (
                first,
                second,
            )
        )


def test_independent_plan_can_allow_mixed_training_modes() -> None:
    first = _layer(0)
    second = _layer(1)
    second.eval()

    plan = (
        build_independent_stack_layer_sharing_plan(
            (
                first,
                second,
            ),
            require_uniform_training_mode=False,
        )
    )

    assert plan.layers_by_depth == (
        first,
        second,
    )


def test_independent_plan_rejects_hidden_width_mismatch() -> None:
    with pytest.raises(
        ValueError,
        match="constant hidden width",
    ):
        build_independent_stack_layer_sharing_plan(
            (
                _layer(0),
                _layer(
                    1,
                    hidden_dim=(
                        HIDDEN_DIM + 1
                    ),
                ),
            )
        )


def test_independent_plan_rejects_relation_order_mismatch() -> None:
    with pytest.raises(
        ValueError,
        match="same exact relation ordering",
    ):
        build_independent_stack_layer_sharing_plan(
            (
                _layer(0),
                _layer(
                    1,
                    relation_names=(
                        "temporal_lag",
                        "spatial_adjacency",
                        "random_placebo",
                    ),
                ),
            )
        )


def test_independent_plan_rejects_stable_relation_id_mismatch() -> None:
    with pytest.raises(
        ValueError,
        match="same stable relation IDs",
    ):
        build_independent_stack_layer_sharing_plan(
            (
                _layer(0),
                _layer(
                    1,
                    stable_relation_ids=(
                        100,
                        201,
                        900,
                    ),
                ),
            )
        )


def test_independent_plan_rejects_registry_fingerprint_mismatch() -> None:
    with pytest.raises(
        ValueError,
        match="same compiled relation registry",
    ):
        build_independent_stack_layer_sharing_plan(
            (
                _layer(0),
                _layer(
                    1,
                    registry_fingerprint=(
                        "different-registry"
                    ),
                ),
            )
        )


def test_plan_rejects_blank_registry_fingerprint() -> None:
    with pytest.raises(
        ValueError,
        match="compiled_relation_registry_fingerprint",
    ):
        build_independent_stack_layer_sharing_plan(
            (
                _layer(
                    0,
                    registry_fingerprint="",
                ),
            )
        )


def test_plan_rejects_blank_architecture_fingerprint() -> None:
    with pytest.raises(
        ValueError,
        match="architecture_fingerprint",
    ):
        build_independent_stack_layer_sharing_plan(
            (
                _layer(
                    0,
                    architecture_fingerprint="",
                ),
            )
        )


def test_plan_rejects_blank_parameter_fingerprint() -> None:
    with pytest.raises(
        ValueError,
        match="parameter_fingerprint",
    ):
        build_independent_stack_layer_sharing_plan(
            (
                _layer(
                    0,
                    parameter_fingerprint_override="",
                ),
            )
        )


def test_plan_is_frozen() -> None:
    plan = _independent_plan()

    with pytest.raises(
        FrozenInstanceError,
    ):
        plan.num_layers = 4  # type: ignore[misc]


def test_validate_independent_plan() -> None:
    plan = _independent_plan()

    validate_stack_layer_sharing_plan(
        plan,
        expected_num_layers=NUM_LAYERS,
        expected_sharing_policy=(
            STACK_SHARING_INDEPENDENT
        ),
    )
    validate_layer_sharing_plan(
        plan
    )


def test_validate_plan_rejects_wrong_type() -> None:
    with pytest.raises(
        TypeError,
        match="StackLayerSharingPlan",
    ):
        validate_stack_layer_sharing_plan(
            object()  # type: ignore[arg-type]
        )


def test_validate_plan_rejects_wrong_expected_depth() -> None:
    plan = _independent_plan()

    with pytest.raises(
        ValueError,
        match="depth differs",
    ):
        validate_stack_layer_sharing_plan(
            plan,
            expected_num_layers=(
                NUM_LAYERS + 1
            ),
        )


def test_validate_plan_rejects_wrong_expected_policy() -> None:
    plan = _independent_plan()

    with pytest.raises(
        ValueError,
        match="policy differs",
    ):
        validate_stack_layer_sharing_plan(
            plan,
            expected_sharing_policy=(
                STACK_SHARING_FULLY_SHARED
            ),
        )


def test_plan_revalidation_detects_postconstruction_parameter_alias() -> None:
    plan = _independent_plan()

    plan.layers_by_depth[1].weight = (
        plan.layers_by_depth[0].weight
    )

    with pytest.raises(
        ValueError,
        match="share exact Parameter",
    ):
        plan.validate_current_ownership()


def test_plan_revalidation_detects_postconstruction_training_mismatch() -> None:
    plan = _independent_plan()

    plan.layers_by_depth[1].eval()

    with pytest.raises(
        ValueError,
        match="train/eval mode",
    ):
        plan.validate_current_ownership()


def test_plan_revalidation_can_ignore_training_mismatch() -> None:
    plan = _independent_plan()

    plan.layers_by_depth[1].eval()

    plan.validate_current_ownership(
        require_uniform_training_mode=False,
    )


# =============================================================================
# Fully shared plans
# =============================================================================


def test_fully_shared_plan_valid_contract() -> None:
    shared = _layer(0)
    plan = (
        build_fully_shared_stack_layer_sharing_plan(
            shared,
            num_layers=NUM_LAYERS,
        )
    )

    assert plan.policy == (
        StackSharingPolicy.fully_shared()
    )
    assert plan.sharing_policy == (
        STACK_SHARING_FULLY_SHARED
    )
    assert plan.num_unique_layers == 1
    assert plan.unique_layers == (
        shared,
    )
    assert (
        plan.depth_to_unique_layer_index
        == (0, 0, 0)
    )
    assert (
        plan.unique_layer_registration_prefixes
        == (
            FULLY_SHARED_LAYER_STATE_DICT_PREFIX,
        )
    )

    for layer in plan.layers_by_depth:
        assert layer is shared


def test_fully_shared_plan_resolves_every_depth_to_same_object() -> None:
    shared = _layer(0)
    plan = _shared_plan(
        layer=shared
    )

    for depth in range(NUM_LAYERS):
        assert (
            plan.layer_for_depth(
                depth
            )
            is shared
        )


def test_fully_shared_plan_rejects_nonlayer() -> None:
    with pytest.raises(
        TypeError,
        match="FunctionalMessagePassingLayer",
    ):
        build_fully_shared_stack_layer_sharing_plan(
            object(),  # type: ignore[arg-type]
            num_layers=2,
        )


@pytest.mark.parametrize(
    "num_layers",
    (
        0,
        -1,
    ),
)
def test_fully_shared_plan_rejects_nonpositive_depth(
    num_layers: int,
) -> None:
    with pytest.raises(
        ValueError,
        match="strictly positive",
    ):
        build_fully_shared_stack_layer_sharing_plan(
            _layer(0),
            num_layers=num_layers,
        )


def test_general_builder_accepts_shared_single_object() -> None:
    shared = _layer(0)

    plan = build_stack_layer_sharing_plan(
        shared,
        num_layers=NUM_LAYERS,
        sharing_policy=(
            STACK_SHARING_FULLY_SHARED
        ),
    )

    assert plan.num_unique_layers == 1
    assert all(
        layer is shared
        for layer in plan.layers_by_depth
    )


def test_general_builder_accepts_shared_one_element_sequence() -> None:
    shared = _layer(0)

    plan = build_stack_layer_sharing_plan(
        (shared,),
        num_layers=NUM_LAYERS,
        sharing_policy=(
            StackSharingPolicy
            .fully_shared()
        ),
    )

    assert plan.layers_by_depth == (
        shared,
        shared,
        shared,
    )


def test_general_builder_accepts_shared_repeated_sequence() -> None:
    shared = _layer(0)

    plan = build_stack_layer_sharing_plan(
        (
            shared,
            shared,
            shared,
        ),
        num_layers=NUM_LAYERS,
        sharing_policy=(
            STACK_SHARING_FULLY_SHARED
        ),
    )

    assert plan.unique_layers == (
        shared,
    )


def test_general_builder_rejects_equal_but_distinct_shared_layers() -> None:
    first = _layer(
        0,
        fill_value=1.0,
    )
    second = _layer(
        0,
        fill_value=1.0,
    )

    assert first is not second
    assert torch.equal(
        first.weight,
        second.weight,
    )

    with pytest.raises(
        ValueError,
        match="same exact layer object",
    ):
        build_stack_layer_sharing_plan(
            (
                first,
                second,
            ),
            num_layers=2,
            sharing_policy=(
                STACK_SHARING_FULLY_SHARED
            ),
        )


def test_general_builder_rejects_invalid_shared_sequence_length() -> None:
    with pytest.raises(
        ValueError,
        match="one layer object",
    ):
        build_stack_layer_sharing_plan(
            (
                _layer(0),
                _layer(1),
            ),
            num_layers=3,
            sharing_policy=(
                STACK_SHARING_FULLY_SHARED
            ),
        )


def test_general_builder_requires_sequence_for_independent() -> None:
    with pytest.raises(
        TypeError,
        match="requires a sequence",
    ):
        build_stack_layer_sharing_plan(
            _layer(0),
            num_layers=1,
            sharing_policy=(
                STACK_SHARING_INDEPENDENT
            ),
        )


def test_general_builder_requires_exact_independent_depth() -> None:
    with pytest.raises(
        ValueError,
        match="exactly num_layers",
    ):
        build_stack_layer_sharing_plan(
            (
                _layer(0),
                _layer(1),
            ),
            num_layers=3,
            sharing_policy=(
                STACK_SHARING_INDEPENDENT
            ),
        )


# =============================================================================
# Factory-based sharing plans
# =============================================================================


def test_independent_factory_called_once_per_depth() -> None:
    calls: list[int] = []

    def factory(
        depth: int,
    ) -> ControlledFunctionalMessagePassingLayer:
        calls.append(
            depth
        )
        return _layer(
            depth
        )

    plan = (
        build_stack_layer_sharing_plan_from_factory(
            factory,
            num_layers=NUM_LAYERS,
            sharing_policy=(
                STACK_SHARING_INDEPENDENT
            ),
        )
    )

    assert calls == [
        0,
        1,
        2,
    ]
    assert plan.num_unique_layers == (
        NUM_LAYERS
    )


def test_fully_shared_factory_called_exactly_once() -> None:
    calls: list[int] = []

    def factory(
        depth: int,
    ) -> ControlledFunctionalMessagePassingLayer:
        calls.append(
            depth
        )
        return _layer(
            depth
        )

    plan = (
        build_layer_sharing_plan_from_factory(
            factory,
            num_layers=NUM_LAYERS,
            sharing_policy=(
                STACK_SHARING_FULLY_SHARED
            ),
        )
    )

    assert calls == [0]
    assert plan.num_unique_layers == 1
    assert all(
        layer is plan.layers_by_depth[0]
        for layer in plan.layers_by_depth
    )


def test_factory_builder_rejects_noncallable() -> None:
    with pytest.raises(
        TypeError,
        match="callable",
    ):
        build_stack_layer_sharing_plan_from_factory(
            object(),  # type: ignore[arg-type]
            num_layers=2,
        )


def test_factory_builder_rejects_invalid_return_type() -> None:
    def factory(
        depth: int,
    ) -> object:
        return object()

    with pytest.raises(
        TypeError,
        match="FunctionalMessagePassingLayer",
    ):
        build_stack_layer_sharing_plan_from_factory(
            factory,  # type: ignore[arg-type]
            num_layers=2,
            sharing_policy=(
                STACK_SHARING_INDEPENDENT
            ),
        )


def test_independent_factory_rejects_same_object_for_every_depth() -> None:
    shared = _layer(0)

    def factory(
        depth: int,
    ) -> ControlledFunctionalMessagePassingLayer:
        return shared

    with pytest.raises(
        ValueError,
        match="distinct exact",
    ):
        build_stack_layer_sharing_plan_from_factory(
            factory,
            num_layers=NUM_LAYERS,
            sharing_policy=(
                STACK_SHARING_INDEPENDENT
            ),
        )


# =============================================================================
# Retention-policy normalization
# =============================================================================


@pytest.mark.parametrize(
    "name",
    (
        STACK_RETENTION_NONE,
        STACK_RETENTION_FINAL_LAYER,
        STACK_RETENTION_ALL_LAYERS,
    ),
)
def test_normalize_stack_retention_policy_accepts_implemented_names(
    name: str,
) -> None:
    assert (
        normalize_stack_retention_policy(
            name
        )
        == name
    )
    assert (
        normalize_stack_retention_policy(
            f"\t{name} "
        )
        == name
    )
    assert (
        normalize_retention_policy(
            name
        )
        == name
    )
    assert_stack_retention_policy_implemented(
        name
    )


@pytest.mark.parametrize(
    "value",
    (
        None,
        1,
        True,
        (),
        object(),
    ),
)
def test_normalize_stack_retention_policy_rejects_non_strings(
    value: Any,
) -> None:
    with pytest.raises(
        TypeError,
        match="must be a string",
    ):
        normalize_stack_retention_policy(
            value  # type: ignore[arg-type]
        )


@pytest.mark.parametrize(
    "value",
    (
        "",
        " ",
        "\n\t",
    ),
)
def test_normalize_stack_retention_policy_rejects_blank_strings(
    value: str,
) -> None:
    with pytest.raises(
        ValueError,
        match="non-empty",
    ):
        normalize_stack_retention_policy(
            value
        )


def test_normalize_stack_retention_policy_rejects_unknown_name() -> None:
    with pytest.raises(
        ValueError,
        match="Unknown stack retention policy",
    ):
        normalize_stack_retention_policy(
            "selected_layers"
        )


def test_normalize_stack_retention_policy_rejects_canonical_unimplemented(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        trace_module,
        "CANONICAL_STACK_RETENTION_POLICIES",
        (
            STACK_RETENTION_NONE,
            STACK_RETENTION_FINAL_LAYER,
            STACK_RETENTION_ALL_LAYERS,
            "selected_layers",
        ),
    )
    monkeypatch.setattr(
        trace_module,
        "V2_0_IMPLEMENTED_STACK_RETENTION_POLICIES",
        (
            STACK_RETENTION_NONE,
            STACK_RETENTION_FINAL_LAYER,
            STACK_RETENTION_ALL_LAYERS,
        ),
    )

    with pytest.raises(
        NotImplementedError,
        match="canonical but not implemented",
    ):
        normalize_stack_retention_policy(
            "selected_layers"
        )


# =============================================================================
# StackRetentionPolicy
# =============================================================================


def test_stack_retention_policy_defaults_to_none() -> None:
    policy = StackRetentionPolicy()

    assert policy.name == (
        STACK_RETENTION_NONE
    )
    assert policy.retains_none
    assert not policy.retains_any_layer
    assert not policy.retains_final_layer


def test_stack_retention_named_constructors() -> None:
    none = StackRetentionPolicy.none()
    final = (
        StackRetentionPolicy.final_layer()
    )
    all_layers = (
        StackRetentionPolicy.all_layers()
    )

    assert none.retains_none
    assert (
        final.retains_final_layer_only
    )
    assert final.retains_final_layer
    assert all_layers.retains_all_layers
    assert all_layers.retains_final_layer


@pytest.mark.parametrize(
    (
        "policy",
        "expected",
    ),
    (
        (
            StackRetentionPolicy.none(),
            (),
        ),
        (
            StackRetentionPolicy.final_layer(),
            (NUM_LAYERS - 1,),
        ),
        (
            StackRetentionPolicy.all_layers(),
            (0, 1, 2),
        ),
    ),
)
def test_retention_policy_expected_indices(
    policy: StackRetentionPolicy,
    expected: tuple[int, ...],
) -> None:
    assert policy.expected_indices(
        num_layers=NUM_LAYERS
    ) == expected
    assert policy.expected_count(
        num_layers=NUM_LAYERS
    ) == len(expected)


@pytest.mark.parametrize(
    (
        "policy",
        "expected",
    ),
    (
        (
            STACK_RETENTION_NONE,
            (False, False, False),
        ),
        (
            STACK_RETENTION_FINAL_LAYER,
            (False, False, True),
        ),
        (
            STACK_RETENTION_ALL_LAYERS,
            (True, True, True),
        ),
    ),
)
def test_retention_policy_should_retain_table(
    policy: str,
    expected: tuple[bool, ...],
) -> None:
    resolved = (
        StackRetentionPolicy(
            name=policy
        )
    )

    observed = tuple(
        resolved.should_retain(
            layer_index=index,
            num_layers=NUM_LAYERS,
        )
        for index in range(
            NUM_LAYERS
        )
    )

    assert observed == expected


def test_retention_policy_rejects_out_of_range_layer_index() -> None:
    with pytest.raises(
        IndexError,
        match="outside",
    ):
        StackRetentionPolicy.all_layers().should_retain(
            layer_index=NUM_LAYERS,
            num_layers=NUM_LAYERS,
        )


@pytest.mark.parametrize(
    "layer_index",
    (
        -1,
        True,
        1.5,
    ),
)
def test_retention_policy_rejects_invalid_layer_index(
    layer_index: Any,
) -> None:
    expected = (
        ValueError
        if layer_index == -1
        else TypeError
    )

    with pytest.raises(expected):
        StackRetentionPolicy.none().should_retain(
            layer_index=layer_index,
            num_layers=NUM_LAYERS,
        )


def test_retention_policy_validates_exact_indices() -> None:
    StackRetentionPolicy.none().validate_retained_indices(
        (),
        num_layers=NUM_LAYERS,
    )
    StackRetentionPolicy.final_layer().validate_retained_indices(
        (2,),
        num_layers=NUM_LAYERS,
    )
    StackRetentionPolicy.all_layers().validate_retained_indices(
        (0, 1, 2),
        num_layers=NUM_LAYERS,
    )


def test_retention_policy_rejects_wrong_indices() -> None:
    with pytest.raises(
        ValueError,
        match="do not match",
    ):
        StackRetentionPolicy.final_layer().validate_retained_indices(
            (0,),
            num_layers=NUM_LAYERS,
        )


@pytest.mark.parametrize(
    "values",
    (
        "0,1,2",
        (0, True),
        (0, -1),
        (0, 1.5),
    ),
)
def test_retention_policy_rejects_invalid_index_sequence(
    values: Any,
) -> None:
    with pytest.raises(
        (
            TypeError,
            ValueError,
        ),
    ):
        StackRetentionPolicy.all_layers().validate_retained_indices(
            values,  # type: ignore[arg-type]
            num_layers=NUM_LAYERS,
        )


def test_retention_policy_fingerprints_are_deterministic() -> None:
    first = (
        StackRetentionPolicy.final_layer()
    )
    second = (
        StackRetentionPolicy.final_layer()
    )

    assert (
        first.execution_contract_dict()
        == second.execution_contract_dict()
    )
    assert (
        first.execution_contract_fingerprint()
        == second.execution_contract_fingerprint()
    )


def test_retention_policy_fingerprint_changes_with_mode() -> None:
    assert (
        StackRetentionPolicy.none()
        .execution_contract_fingerprint()
        != StackRetentionPolicy.all_layers()
        .execution_contract_fingerprint()
    )


def test_retention_policy_is_frozen() -> None:
    policy = StackRetentionPolicy()

    with pytest.raises(
        FrozenInstanceError,
    ):
        policy.name = (  # type: ignore[misc]
            STACK_RETENTION_ALL_LAYERS
        )


def test_resolve_stack_retention_policy() -> None:
    policy = (
        StackRetentionPolicy.final_layer()
    )

    assert (
        resolve_stack_retention_policy(
            policy
        )
        is policy
    )
    assert (
        resolve_retention_policy(
            STACK_RETENTION_FINAL_LAYER
        )
        == policy
    )


@pytest.mark.parametrize(
    "value",
    (
        None,
        1,
        object(),
    ),
)
def test_resolve_stack_retention_policy_rejects_invalid_type(
    value: Any,
) -> None:
    with pytest.raises(
        TypeError,
        match="string or StackRetentionPolicy",
    ):
        resolve_stack_retention_policy(
            value  # type: ignore[arg-type]
        )


def test_stack_retention_predicates() -> None:
    assert is_no_stack_retention(
        STACK_RETENTION_NONE
    )
    assert (
        is_final_layer_stack_retention(
            STACK_RETENTION_FINAL_LAYER
        )
    )
    assert (
        is_all_layers_stack_retention(
            StackRetentionPolicy.all_layers()
        )
    )


# =============================================================================
# StackTraceDecision
# =============================================================================


@pytest.mark.parametrize(
    (
        "mode",
        "captures_layer",
        "captures_node",
        "captures_full",
    ),
    (
        (
            LAYER_TRACE_NONE,
            False,
            False,
            False,
        ),
        (
            LAYER_TRACE_NODE,
            True,
            True,
            False,
        ),
        (
            LAYER_TRACE_FULL,
            True,
            True,
            True,
        ),
    ),
)
def test_stack_trace_decision_trace_detail(
    mode: str,
    captures_layer: bool,
    captures_node: bool,
    captures_full: bool,
) -> None:
    decision = StackTraceDecision(
        layer_index=1,
        num_layers=NUM_LAYERS,
        retain_public_output=False,
        is_final_layer=False,
        layer_trace_policy=mode,
        audit_mode=True,
    )

    assert decision.layer_trace_mode == (
        mode
    )
    assert (
        decision.captures_layer_trace
        is captures_layer
    )
    assert (
        decision.captures_node_trace
        is captures_node
    )
    assert (
        decision.captures_full_trace
        is captures_full
    )
    assert (
        decision.retains_or_audits_layer
        is True
    )


def test_stack_trace_decision_retains_or_audits() -> None:
    neither = StackTraceDecision(
        layer_index=0,
        num_layers=2,
        retain_public_output=False,
        is_final_layer=False,
        layer_trace_policy=(
            LAYER_TRACE_NONE
        ),
        audit_mode=False,
    )
    retained = StackTraceDecision(
        layer_index=1,
        num_layers=2,
        retain_public_output=True,
        is_final_layer=True,
        layer_trace_policy=(
            LAYER_TRACE_NONE
        ),
        audit_mode=False,
    )

    assert not (
        neither.retains_or_audits_layer
    )
    assert retained.retains_or_audits_layer


def test_stack_trace_decision_fingerprint_is_deterministic() -> None:
    first = StackTraceDecision(
        layer_index=2,
        num_layers=3,
        retain_public_output=True,
        is_final_layer=True,
        layer_trace_policy=(
            LAYER_TRACE_FULL
        ),
        audit_mode=True,
    )
    second = StackTraceDecision(
        layer_index=2,
        num_layers=3,
        retain_public_output=True,
        is_final_layer=True,
        layer_trace_policy=(
            LayerTracePolicy(
                mode=LAYER_TRACE_FULL
            )
        ),
        audit_mode=True,
    )

    assert (
        first.execution_contract_dict()
        == second.execution_contract_dict()
    )
    assert (
        first.execution_contract_fingerprint()
        == second.execution_contract_fingerprint()
    )


def test_stack_trace_decision_rejects_wrong_final_flag() -> None:
    with pytest.raises(
        ValueError,
        match="is_final_layer",
    ):
        StackTraceDecision(
            layer_index=2,
            num_layers=3,
            retain_public_output=True,
            is_final_layer=False,
            layer_trace_policy=(
                LAYER_TRACE_NONE
            ),
        )


def test_stack_trace_decision_rejects_out_of_range_layer() -> None:
    with pytest.raises(
        ValueError,
        match="smaller than num_layers",
    ):
        StackTraceDecision(
            layer_index=3,
            num_layers=3,
            retain_public_output=False,
            is_final_layer=False,
            layer_trace_policy=(
                LAYER_TRACE_NONE
            ),
        )


@pytest.mark.parametrize(
    (
        "field",
        "value",
    ),
    (
        (
            "retain_public_output",
            1,
        ),
        (
            "is_final_layer",
            0,
        ),
        (
            "audit_mode",
            "yes",
        ),
    ),
)
def test_stack_trace_decision_rejects_nonboolean_flags(
    field: str,
    value: Any,
) -> None:
    kwargs: dict[str, Any] = {
        "layer_index": 0,
        "num_layers": 2,
        "retain_public_output": False,
        "is_final_layer": False,
        "layer_trace_policy": (
            LAYER_TRACE_NONE
        ),
        "audit_mode": False,
    }
    kwargs[field] = value

    with pytest.raises(
        TypeError,
        match="Boolean",
    ):
        StackTraceDecision(**kwargs)


# =============================================================================
# StackTracePolicy
# =============================================================================


def test_stack_trace_policy_defaults_to_minimal() -> None:
    policy = StackTracePolicy()

    assert policy.retention_name == (
        STACK_RETENTION_NONE
    )
    assert policy.layer_trace_mode == (
        LAYER_TRACE_NONE
    )
    assert policy.audit_mode is False
    assert not policy.retains_any_layer
    assert not policy.captures_any_layer_trace


def test_stack_trace_policy_named_constructors() -> None:
    minimal = StackTracePolicy.minimal()
    final = (
        StackTracePolicy.retain_final_layer(
            layer_trace_policy=(
                LAYER_TRACE_NODE
            )
        )
    )
    all_layers = (
        StackTracePolicy.retain_all_layers(
            layer_trace_policy=(
                LAYER_TRACE_FULL
            )
        )
    )
    audit = (
        StackTracePolicy.full_audit()
    )

    assert minimal == StackTracePolicy()
    assert final.retention_name == (
        STACK_RETENTION_FINAL_LAYER
    )
    assert final.layer_trace_mode == (
        LAYER_TRACE_NODE
    )
    assert all_layers.retention_name == (
        STACK_RETENTION_ALL_LAYERS
    )
    assert (
        all_layers.captures_full_layer_trace
    )
    assert audit.audit_mode
    assert audit.retains_final_layer
    assert (
        audit.layer_trace_mode
        == LAYER_TRACE_FULL
    )


@pytest.mark.parametrize(
    (
        "retention",
        "expected",
    ),
    (
        (
            STACK_RETENTION_NONE,
            (),
        ),
        (
            STACK_RETENTION_FINAL_LAYER,
            (2,),
        ),
        (
            STACK_RETENTION_ALL_LAYERS,
            (0, 1, 2),
        ),
    ),
)
def test_stack_trace_policy_expected_retention(
    retention: str,
    expected: tuple[int, ...],
) -> None:
    policy = StackTracePolicy(
        retention_policy=retention,
        layer_trace_policy=(
            LAYER_TRACE_FULL
        ),
    )

    assert (
        policy.expected_retained_indices(
            num_layers=NUM_LAYERS
        )
        == expected
    )
    assert (
        policy.expected_retained_count(
            num_layers=NUM_LAYERS
        )
        == len(expected)
    )


@pytest.mark.parametrize(
    (
        "retention",
        "expected",
    ),
    (
        (
            STACK_RETENTION_NONE,
            (False, False, False),
        ),
        (
            STACK_RETENTION_FINAL_LAYER,
            (False, False, True),
        ),
        (
            STACK_RETENTION_ALL_LAYERS,
            (True, True, True),
        ),
    ),
)
def test_stack_trace_policy_decisions_match_retention(
    retention: str,
    expected: tuple[bool, ...],
) -> None:
    policy = StackTracePolicy(
        retention_policy=retention,
        layer_trace_policy=(
            LAYER_TRACE_NODE
        ),
        audit_mode=True,
    )
    decisions = policy.decisions(
        num_layers=NUM_LAYERS
    )

    assert tuple(
        decision.retain_public_output
        for decision in decisions
    ) == expected
    assert tuple(
        decision.layer_index
        for decision in decisions
    ) == (0, 1, 2)
    assert tuple(
        decision.is_final_layer
        for decision in decisions
    ) == (
        False,
        False,
        True,
    )
    assert all(
        decision.layer_trace_mode
        == LAYER_TRACE_NODE
        for decision in decisions
    )
    assert all(
        decision.audit_mode
        for decision in decisions
    )


def test_audit_mode_does_not_force_public_retention() -> None:
    policy = StackTracePolicy(
        retention_policy=(
            STACK_RETENTION_NONE
        ),
        layer_trace_policy=(
            LAYER_TRACE_FULL
        ),
        audit_mode=True,
    )

    decisions = policy.decisions(
        num_layers=NUM_LAYERS
    )

    assert all(
        not decision.retain_public_output
        for decision in decisions
    )
    assert all(
        decision.audit_mode
        for decision in decisions
    )


def test_stack_trace_policy_execution_fingerprint_is_deterministic() -> None:
    first = StackTracePolicy(
        retention_policy=(
            STACK_RETENTION_FINAL_LAYER
        ),
        layer_trace_policy=(
            LAYER_TRACE_NODE
        ),
        audit_mode=True,
    )
    second = StackTracePolicy(
        retention_policy=(
            StackRetentionPolicy
            .final_layer()
        ),
        layer_trace_policy=(
            LayerTracePolicy(
                mode=LAYER_TRACE_NODE
            )
        ),
        audit_mode=True,
    )

    assert (
        first.execution_contract_dict()
        == second.execution_contract_dict()
    )
    assert (
        first.execution_contract_fingerprint()
        == second.execution_contract_fingerprint()
    )


def test_stack_trace_execution_fingerprint_changes_with_observability() -> None:
    minimal = StackTracePolicy.minimal()
    verbose = StackTracePolicy.full_audit()

    assert (
        minimal.execution_contract_fingerprint()
        != verbose.execution_contract_fingerprint()
    )


def test_stack_trace_numerical_fingerprint_ignores_observability() -> None:
    minimal = StackTracePolicy.minimal()
    verbose = StackTracePolicy.full_audit()

    assert (
        minimal.numerical_equivalence_dict()
        == verbose.numerical_equivalence_dict()
    )
    assert (
        minimal.numerical_equivalence_fingerprint()
        == verbose.numerical_equivalence_fingerprint()
    )


def test_stack_trace_policy_is_frozen() -> None:
    policy = StackTracePolicy()

    with pytest.raises(
        FrozenInstanceError,
    ):
        policy.audit_mode = True  # type: ignore[misc]


def test_stack_trace_policy_rejects_invalid_layer_trace_type() -> None:
    with pytest.raises(
        TypeError,
        match="LayerTracePolicy",
    ):
        StackTracePolicy(
            layer_trace_policy=object(),  # type: ignore[arg-type]
        )


# =============================================================================
# Trace-policy resolution and construction
# =============================================================================


def test_resolve_stack_trace_policy_defaults() -> None:
    policy = resolve_stack_trace_policy()

    assert policy == (
        StackTracePolicy.minimal()
    )


def test_resolve_stack_trace_policy_constructs_from_overrides() -> None:
    policy = resolve_stack_trace_policy(
        retention_policy=(
            STACK_RETENTION_ALL_LAYERS
        ),
        layer_trace_policy=(
            LAYER_TRACE_FULL
        ),
        audit_mode=True,
    )

    assert policy == StackTracePolicy(
        retention_policy=(
            STACK_RETENTION_ALL_LAYERS
        ),
        layer_trace_policy=(
            LAYER_TRACE_FULL
        ),
        audit_mode=True,
    )


def test_resolve_stack_trace_policy_preserves_existing_object() -> None:
    policy = (
        StackTracePolicy.retain_final_layer(
            layer_trace_policy=(
                LAYER_TRACE_NODE
            )
        )
    )

    assert (
        resolve_stack_trace_policy(
            policy
        )
        is policy
    )
    assert (
        resolve_trace_policy(
            policy
        )
        is policy
    )


def test_resolve_stack_trace_policy_accepts_matching_overrides() -> None:
    policy = StackTracePolicy(
        retention_policy=(
            STACK_RETENTION_FINAL_LAYER
        ),
        layer_trace_policy=(
            LAYER_TRACE_NODE
        ),
        audit_mode=True,
    )

    assert (
        resolve_stack_trace_policy(
            policy,
            retention_policy=(
                STACK_RETENTION_FINAL_LAYER
            ),
            layer_trace_policy=(
                LAYER_TRACE_NODE
            ),
            audit_mode=True,
        )
        is policy
    )


@pytest.mark.parametrize(
    (
        "override_name",
        "override_value",
    ),
    (
        (
            "retention_policy",
            STACK_RETENTION_ALL_LAYERS,
        ),
        (
            "layer_trace_policy",
            LAYER_TRACE_FULL,
        ),
        (
            "audit_mode",
            False,
        ),
    ),
)
def test_resolve_stack_trace_policy_rejects_conflicting_overrides(
    override_name: str,
    override_value: Any,
) -> None:
    policy = StackTracePolicy(
        retention_policy=(
            STACK_RETENTION_FINAL_LAYER
        ),
        layer_trace_policy=(
            LAYER_TRACE_NODE
        ),
        audit_mode=True,
    )

    kwargs = {
        override_name: override_value,
    }

    with pytest.raises(
        ValueError,
        match="conflicts",
    ):
        resolve_stack_trace_policy(
            policy,
            **kwargs,
        )


def test_resolve_stack_trace_policy_rejects_wrong_value_type() -> None:
    with pytest.raises(
        TypeError,
        match="StackTracePolicy or None",
    ):
        resolve_stack_trace_policy(
            object()  # type: ignore[arg-type]
        )


def test_build_stack_trace_policy_and_alias() -> None:
    first = build_stack_trace_policy(
        retention_policy=(
            STACK_RETENTION_FINAL_LAYER
        ),
        layer_trace_policy=(
            LAYER_TRACE_NODE
        ),
        audit_mode=True,
    )
    second = build_trace_policy(
        retention_policy=(
            STACK_RETENTION_FINAL_LAYER
        ),
        layer_trace_policy=(
            LAYER_TRACE_NODE
        ),
        audit_mode=True,
    )

    assert first == second


def test_build_stack_trace_decisions_and_alias() -> None:
    first = build_stack_trace_decisions(
        num_layers=NUM_LAYERS,
        retention_policy=(
            STACK_RETENTION_ALL_LAYERS
        ),
        layer_trace_policy=(
            LAYER_TRACE_NODE
        ),
        audit_mode=False,
    )
    second = build_trace_decisions(
        num_layers=NUM_LAYERS,
        retention_policy=(
            STACK_RETENTION_ALL_LAYERS
        ),
        layer_trace_policy=(
            LAYER_TRACE_NODE
        ),
        audit_mode=False,
    )

    assert first == second
    assert len(first) == NUM_LAYERS


# =============================================================================
# Trace-decision validation
# =============================================================================


def test_validate_stack_trace_decisions_accepts_policy_generated_values() -> None:
    policy = StackTracePolicy(
        retention_policy=(
            STACK_RETENTION_FINAL_LAYER
        ),
        layer_trace_policy=(
            LAYER_TRACE_FULL
        ),
        audit_mode=True,
    )
    decisions = policy.decisions(
        num_layers=NUM_LAYERS
    )

    validate_stack_trace_decisions(
        decisions,
        policy=policy,
        num_layers=NUM_LAYERS,
    )
    validate_trace_decisions(
        decisions,
        policy=policy,
        num_layers=NUM_LAYERS,
    )
    policy.validate_decisions(
        decisions,
        num_layers=NUM_LAYERS,
    )


def test_validate_stack_trace_decisions_rejects_nonsequence() -> None:
    with pytest.raises(
        TypeError,
        match="sequence",
    ):
        validate_stack_trace_decisions(
            object(),  # type: ignore[arg-type]
            policy=StackTracePolicy(),
            num_layers=1,
        )


def test_validate_stack_trace_decisions_rejects_wrong_policy_type() -> None:
    with pytest.raises(
        TypeError,
        match="StackTracePolicy",
    ):
        validate_stack_trace_decisions(
            (),
            policy=object(),  # type: ignore[arg-type]
            num_layers=1,
        )


def test_validate_stack_trace_decisions_rejects_wrong_length() -> None:
    policy = StackTracePolicy()

    with pytest.raises(
        ValueError,
        match="exactly one decision per layer",
    ):
        validate_stack_trace_decisions(
            policy.decisions(
                num_layers=2
            ),
            policy=policy,
            num_layers=3,
        )


def test_validate_stack_trace_decisions_rejects_wrong_member_type() -> None:
    with pytest.raises(
        TypeError,
        match="StackTraceDecision",
    ):
        validate_stack_trace_decisions(
            (
                object(),  # type: ignore[arg-type]
            ),
            policy=StackTracePolicy(),
            num_layers=1,
        )


def test_validate_stack_trace_decisions_rejects_noncontiguous_indices() -> None:
    policy = StackTracePolicy(
        retention_policy=(
            STACK_RETENTION_ALL_LAYERS
        ),
    )
    decisions = (
        StackTraceDecision(
            layer_index=1,
            num_layers=2,
            retain_public_output=True,
            is_final_layer=True,
            layer_trace_policy=(
                LAYER_TRACE_NONE
            ),
        ),
        StackTraceDecision(
            layer_index=0,
            num_layers=2,
            retain_public_output=True,
            is_final_layer=False,
            layer_trace_policy=(
                LAYER_TRACE_NONE
            ),
        ),
    )

    with pytest.raises(
        ValueError,
        match="contiguous zero-based",
    ):
        validate_stack_trace_decisions(
            decisions,
            policy=policy,
            num_layers=2,
        )


def test_validate_stack_trace_decisions_rejects_num_layers_mismatch() -> None:
    policy = StackTracePolicy()
    decisions = (
        StackTraceDecision(
            layer_index=0,
            num_layers=2,
            retain_public_output=False,
            is_final_layer=False,
            layer_trace_policy=(
                LAYER_TRACE_NONE
            ),
        ),
        StackTraceDecision(
            layer_index=1,
            num_layers=3,
            retain_public_output=False,
            is_final_layer=False,
            layer_trace_policy=(
                LAYER_TRACE_NONE
            ),
        ),
    )

    with pytest.raises(
        ValueError,
        match="same num_layers",
    ):
        validate_stack_trace_decisions(
            decisions,
            policy=policy,
            num_layers=2,
        )


def test_validate_stack_trace_decisions_rejects_layer_trace_mismatch() -> None:
    policy = StackTracePolicy(
        layer_trace_policy=(
            LAYER_TRACE_NONE
        )
    )
    decisions = (
        StackTraceDecision(
            layer_index=0,
            num_layers=1,
            retain_public_output=False,
            is_final_layer=True,
            layer_trace_policy=(
                LAYER_TRACE_FULL
            ),
        ),
    )

    with pytest.raises(
        ValueError,
        match="layer policy differs",
    ):
        validate_stack_trace_decisions(
            decisions,
            policy=policy,
            num_layers=1,
        )


def test_validate_stack_trace_decisions_rejects_audit_mismatch() -> None:
    policy = StackTracePolicy(
        audit_mode=False
    )
    decisions = (
        StackTraceDecision(
            layer_index=0,
            num_layers=1,
            retain_public_output=False,
            is_final_layer=True,
            layer_trace_policy=(
                LAYER_TRACE_NONE
            ),
            audit_mode=True,
        ),
    )

    with pytest.raises(
        ValueError,
        match="audit_mode differs",
    ):
        validate_stack_trace_decisions(
            decisions,
            policy=policy,
            num_layers=1,
        )


def test_validate_stack_trace_decisions_rejects_retention_mismatch() -> None:
    policy = StackTracePolicy(
        retention_policy=(
            STACK_RETENTION_FINAL_LAYER
        )
    )
    decisions = (
        StackTraceDecision(
            layer_index=0,
            num_layers=1,
            retain_public_output=False,
            is_final_layer=True,
            layer_trace_policy=(
                LAYER_TRACE_NONE
            ),
        ),
    )

    with pytest.raises(
        ValueError,
        match="retention flag differs",
    ):
        validate_stack_trace_decisions(
            decisions,
            policy=policy,
            num_layers=1,
        )


def test_validate_retained_layer_indices_helper() -> None:
    validate_retained_layer_indices(
        (2,),
        retention_policy=(
            STACK_RETENTION_FINAL_LAYER
        ),
        num_layers=3,
    )

    with pytest.raises(
        ValueError,
        match="do not match",
    ):
        validate_retained_layer_indices(
            (0,),
            retention_policy=(
                STACK_RETENTION_FINAL_LAYER
            ),
            num_layers=3,
        )


# =============================================================================
# Numerical equivalence and exact matching
# =============================================================================


@pytest.mark.parametrize(
    "left",
    (
        StackTracePolicy.minimal(),
        StackTracePolicy.retain_final_layer(
            layer_trace_policy=(
                LAYER_TRACE_NODE
            )
        ),
        StackTracePolicy.full_audit(),
    ),
)
@pytest.mark.parametrize(
    "right",
    (
        StackTracePolicy.minimal(),
        StackTracePolicy.retain_all_layers(
            layer_trace_policy=(
                LAYER_TRACE_FULL
            )
        ),
        StackTracePolicy(
            retention_policy=(
                STACK_RETENTION_NONE
            ),
            layer_trace_policy=(
                LAYER_TRACE_FULL
            ),
            audit_mode=True,
        ),
    ),
)
def test_all_bounded_trace_policies_are_numerically_equivalent(
    left: StackTracePolicy,
    right: StackTracePolicy,
) -> None:
    assert (
        stack_trace_policies_are_numerically_equivalent(
            left,
            right,
        )
    )


@pytest.mark.parametrize(
    (
        "left",
        "right",
    ),
    (
        (
            object(),
            StackTracePolicy(),
        ),
        (
            StackTracePolicy(),
            object(),
        ),
    ),
)
def test_trace_numerical_equivalence_rejects_wrong_types(
    left: Any,
    right: Any,
) -> None:
    with pytest.raises(
        TypeError,
    ):
        stack_trace_policies_are_numerically_equivalent(
            left,  # type: ignore[arg-type]
            right,  # type: ignore[arg-type]
        )


def test_assert_stack_trace_policy_matches() -> None:
    policy = StackTracePolicy(
        retention_policy=(
            STACK_RETENTION_FINAL_LAYER
        ),
        layer_trace_policy=(
            LAYER_TRACE_NODE
        ),
        audit_mode=True,
    )

    assert_stack_trace_policy_matches(
        policy,
        retention_policy=(
            STACK_RETENTION_FINAL_LAYER
        ),
        layer_trace_policy=(
            LAYER_TRACE_NODE
        ),
        audit_mode=True,
    )


def test_assert_stack_trace_policy_matches_rejects_difference() -> None:
    policy = StackTracePolicy.minimal()

    with pytest.raises(
        ValueError,
        match="differs",
    ):
        assert_stack_trace_policy_matches(
            policy,
            retention_policy=(
                STACK_RETENTION_ALL_LAYERS
            ),
            layer_trace_policy=(
                LAYER_TRACE_FULL
            ),
            audit_mode=True,
        )


def test_assert_stack_trace_policy_matches_rejects_wrong_type() -> None:
    with pytest.raises(
        TypeError,
        match="StackTracePolicy",
    ):
        assert_stack_trace_policy_matches(
            object(),  # type: ignore[arg-type]
            retention_policy=(
                STACK_RETENTION_NONE
            ),
            layer_trace_policy=(
                LAYER_TRACE_NONE
            ),
            audit_mode=False,
        )


# =============================================================================
# Policy orthogonality
# =============================================================================


def test_sharing_and_trace_policy_fingerprints_are_separate() -> None:
    independent = (
        StackSharingPolicy.independent()
    )
    shared = (
        StackSharingPolicy.fully_shared()
    )
    minimal = StackTracePolicy.minimal()
    audited = StackTracePolicy.full_audit()

    assert (
        independent.architecture_fingerprint()
        != shared.architecture_fingerprint()
    )
    assert (
        minimal.execution_contract_fingerprint()
        != audited.execution_contract_fingerprint()
    )
    assert (
        minimal.numerical_equivalence_fingerprint()
        == audited.numerical_equivalence_fingerprint()
    )


def test_retention_is_independent_of_layer_trace_detail() -> None:
    no_trace = StackTracePolicy(
        retention_policy=(
            STACK_RETENTION_ALL_LAYERS
        ),
        layer_trace_policy=(
            LAYER_TRACE_NONE
        ),
    )
    full_trace = StackTracePolicy(
        retention_policy=(
            STACK_RETENTION_ALL_LAYERS
        ),
        layer_trace_policy=(
            LAYER_TRACE_FULL
        ),
    )

    assert (
        no_trace.expected_retained_indices(
            num_layers=NUM_LAYERS
        )
        == full_trace.expected_retained_indices(
            num_layers=NUM_LAYERS
        )
        == (0, 1, 2)
    )


def test_layer_trace_detail_is_independent_of_retention() -> None:
    no_retention = StackTracePolicy(
        retention_policy=(
            STACK_RETENTION_NONE
        ),
        layer_trace_policy=(
            LAYER_TRACE_NODE
        ),
    )
    all_retention = StackTracePolicy(
        retention_policy=(
            STACK_RETENTION_ALL_LAYERS
        ),
        layer_trace_policy=(
            LAYER_TRACE_NODE
        ),
    )

    assert (
        no_retention.layer_trace_policy
        == all_retention.layer_trace_policy
    )
    assert (
        no_retention.expected_retained_indices(
            num_layers=NUM_LAYERS
        )
        == ()
    )
    assert (
        all_retention.expected_retained_indices(
            num_layers=NUM_LAYERS
        )
        == (0, 1, 2)
    )


# =============================================================================
# Public aliases and export surfaces
# =============================================================================


def test_trace_aliases_are_exact() -> None:
    assert RetentionPolicy is (
        StackRetentionPolicy
    )
    assert TracePolicy is (
        StackTracePolicy
    )
    assert TraceDecision is (
        StackTraceDecision
    )
    assert normalize_retention_policy is (
        normalize_stack_retention_policy
    )
    assert resolve_retention_policy is (
        resolve_stack_retention_policy
    )
    assert resolve_trace_policy is (
        resolve_stack_trace_policy
    )
    assert build_trace_policy is (
        build_stack_trace_policy
    )
    assert build_trace_decisions is (
        build_stack_trace_decisions
    )
    assert validate_trace_decisions is (
        validate_stack_trace_decisions
    )


def test_sharing_policy_all_is_unique_and_bound() -> None:
    exported = tuple(
        sharing_module.__all__
    )

    assert len(exported) == len(
        set(exported)
    )

    for name in exported:
        assert hasattr(
            sharing_module,
            name,
        ), name


def test_trace_policy_all_is_unique_and_bound() -> None:
    exported = tuple(
        trace_module.__all__
    )

    assert len(exported) == len(
        set(exported)
    )

    for name in exported:
        assert hasattr(
            trace_module,
            name,
        ), name


def test_sharing_policy_expected_public_exports() -> None:
    expected = {
        "STACK_SHARING_POLICY_SCHEMA_VERSION",
        "STACK_LAYER_SHARING_PLAN_SCHEMA_VERSION",
        "STACK_SHARING_POLICY_AFFECTS_NUMERICAL_ARCHITECTURE",
        "STACK_SHARING_POLICY_AFFECTS_PARAMETER_OWNERSHIP",
        "STACK_SHARING_POLICY_AFFECTS_TRACE_DETAIL",
        "STACK_SHARING_POLICY_AFFECTS_OUTPUT_RETENTION",
        "STACK_SHARING_POLICY_AFFECTS_DIAGNOSTICS",
        "STACK_SHARING_INDEPENDENT_REQUIRES_DISTINCT_MODULES",
        "STACK_SHARING_INDEPENDENT_REQUIRES_DISTINCT_PARAMETERS",
        "STACK_SHARING_FULLY_SHARED_REQUIRES_EXACT_MODULE_REUSE",
        "STACK_SHARING_PARTIAL_SUPPORTED",
        "STACK_SHARING_SCIENTIFIC_INTERPRETATION",
        "INDEPENDENT_LAYER_STATE_DICT_PREFIX",
        "FULLY_SHARED_LAYER_STATE_DICT_PREFIX",
        "normalize_stack_sharing_policy",
        "normalize_sharing_policy",
        "assert_stack_sharing_policy_implemented",
        "is_independent_stack_sharing_policy",
        "is_fully_shared_stack_sharing_policy",
        "StackSharingPolicy",
        "SharingPolicy",
        "resolve_stack_sharing_policy",
        "resolve_sharing_policy",
        "StackLayerSharingPlan",
        "LayerSharingPlan",
        "build_independent_stack_layer_sharing_plan",
        "build_fully_shared_stack_layer_sharing_plan",
        "build_stack_layer_sharing_plan",
        "build_layer_sharing_plan",
        "build_stack_layer_sharing_plan_from_factory",
        "build_layer_sharing_plan_from_factory",
        "validate_stack_layer_sharing_plan",
        "validate_layer_sharing_plan",
        "layer_for_stack_depth",
    }

    assert set(
        sharing_module.__all__
    ) == expected


def test_trace_policy_expected_public_exports() -> None:
    expected = {
        "STACK_RETENTION_POLICY_SCHEMA_VERSION",
        "STACK_TRACE_POLICY_SCHEMA_VERSION",
        "STACK_TRACE_DECISION_SCHEMA_VERSION",
        "STACK_RETENTION_AFFECTS_NUMERICAL_RESULTS",
        "STACK_LAYER_TRACE_AFFECTS_NUMERICAL_RESULTS",
        "STACK_AUDIT_MODE_AFFECTS_NUMERICAL_RESULTS",
        "STACK_RETENTION_NONE_RETAINS_FINAL_LAYER",
        "STACK_RETENTION_FINAL_LAYER_RETAINS_FINAL_LAYER",
        "STACK_RETENTION_ALL_LAYERS_RETAINS_FINAL_LAYER",
        "STACK_RETENTION_POLICY_IS_OUTPUT_CONTRACT",
        "STACK_LAYER_TRACE_POLICY_IS_OUTPUT_CONTRACT",
        "STACK_TRACE_POLICY_IS_NUMERICAL_ARCHITECTURE",
        "STACK_TRACE_POLICY_SCIENTIFIC_INTERPRETATION",
        "normalize_stack_retention_policy",
        "normalize_retention_policy",
        "assert_stack_retention_policy_implemented",
        "resolve_stack_retention_policy",
        "resolve_retention_policy",
        "is_no_stack_retention",
        "is_final_layer_stack_retention",
        "is_all_layers_stack_retention",
        "StackRetentionPolicy",
        "RetentionPolicy",
        "StackTraceDecision",
        "TraceDecision",
        "StackTracePolicy",
        "TracePolicy",
        "resolve_stack_trace_policy",
        "resolve_trace_policy",
        "build_stack_trace_policy",
        "build_trace_policy",
        "build_stack_trace_decisions",
        "build_trace_decisions",
        "validate_stack_trace_decisions",
        "validate_trace_decisions",
        "validate_retained_layer_indices",
        "assert_stack_trace_policy_matches",
        "stack_trace_policies_are_numerically_equivalent",
    }

    assert set(
        trace_module.__all__
    ) == expected


def test_registration_prefix_constants() -> None:
    assert (
        INDEPENDENT_LAYER_STATE_DICT_PREFIX
        == "layers"
    )
    assert (
        FULLY_SHARED_LAYER_STATE_DICT_PREFIX
        == "shared_layer"
    )
