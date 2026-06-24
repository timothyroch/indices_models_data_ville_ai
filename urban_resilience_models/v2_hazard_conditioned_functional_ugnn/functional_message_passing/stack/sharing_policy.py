"""
Parameter-ownership policy for multi-layer functional message passing.

Target path:
    urban_resilience_models/
        v2_hazard_conditioned_functional_ugnn/
            functional_message_passing/
                stack/
                    sharing_policy.py

This module freezes how complete ``FunctionalMessagePassingLayer`` modules are
owned and reused across stack depth.

It owns:

- canonical and implemented sharing-policy validation;
- immutable sharing-policy metadata;
- exact layer ownership plans;
- independent-layer object and parameter-alias checks;
- fully-shared exact-module-reuse checks;
- deterministic depth-to-owner mappings;
- architecture and parameter fingerprint composition;
- factory-based construction of independent or shared layer plans;
- stable state-dict registration-name recommendations.

It does not own:

- construction of one functional message-passing layer from model config;
- layer mathematics;
- stack iteration;
- state rebinding;
- stack output retention;
- layer trace detail;
- diagnostics;
- optimization or regularization reduction;
- checkpoint migration.

Bounded V2.0 policies
---------------------
``independent``
    Every stack depth owns a distinct ``FunctionalMessagePassingLayer``
    instance. Distinct layers must not share exact ``Parameter`` objects or
    nonempty parameter storage.

``fully_shared``
    Every stack depth applies the same exact
    ``FunctionalMessagePassingLayer`` object. Equal values in separately
    instantiated layers are not considered sharing.

Partial sharing is not implemented. A future policy must define exactly which
submodules are shared, how they are registered, how parameter fingerprints are
composed, and how parameter-only regularization is reduced before it can be
added to the implemented vocabulary.

Scientific interpretation
--------------------------
Sharing policy changes the numerical hypothesis class:

- independent layers may learn depth-specific transformations;
- a fully shared layer defines repeated application of one transition
  operator.

The policy therefore belongs to the numerical architecture fingerprint. It is
not a trace, retention, diagnostics, or presentation setting.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from hashlib import sha256
import json
from typing import Any, Final, TypeAlias

import torch
from torch import nn

from ...constants import (
    CANONICAL_STACK_SHARING_POLICIES,
    STACK_SHARING_FULLY_SHARED,
    STACK_SHARING_INDEPENDENT,
    V2_0_IMPLEMENTED_STACK_SHARING_POLICIES,
)
from ..layer.layer import FunctionalMessagePassingLayer


# =============================================================================
# Public identity
# =============================================================================


STACK_SHARING_POLICY_SCHEMA_VERSION: Final[str] = "0.1"
STACK_LAYER_SHARING_PLAN_SCHEMA_VERSION: Final[str] = "0.1"

STACK_SHARING_POLICY_AFFECTS_NUMERICAL_ARCHITECTURE: Final[bool] = True
STACK_SHARING_POLICY_AFFECTS_PARAMETER_OWNERSHIP: Final[bool] = True
STACK_SHARING_POLICY_AFFECTS_TRACE_DETAIL: Final[bool] = False
STACK_SHARING_POLICY_AFFECTS_OUTPUT_RETENTION: Final[bool] = False
STACK_SHARING_POLICY_AFFECTS_DIAGNOSTICS: Final[bool] = False

STACK_SHARING_INDEPENDENT_REQUIRES_DISTINCT_MODULES: Final[bool] = True
STACK_SHARING_INDEPENDENT_REQUIRES_DISTINCT_PARAMETERS: Final[bool] = True
STACK_SHARING_FULLY_SHARED_REQUIRES_EXACT_MODULE_REUSE: Final[bool] = True
STACK_SHARING_PARTIAL_SUPPORTED: Final[bool] = False

STACK_SHARING_SCIENTIFIC_INTERPRETATION: Final[str] = (
    "depth_specific_operators_or_repeated_shared_transition_operator"
)

INDEPENDENT_LAYER_STATE_DICT_PREFIX: Final[str] = "layers"
FULLY_SHARED_LAYER_STATE_DICT_PREFIX: Final[str] = "shared_layer"


StackLayerFactory: TypeAlias = Callable[
    [int],
    FunctionalMessagePassingLayer,
]

LayerOrLayerSequence: TypeAlias = (
    FunctionalMessagePassingLayer
    | Sequence[FunctionalMessagePassingLayer]
)


# =============================================================================
# Generic helpers
# =============================================================================


def _canonical_json(
    payload: dict[str, Any],
) -> str:
    return json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )


def _fingerprint(
    payload: dict[str, Any],
) -> str:
    return sha256(
        _canonical_json(payload).encode(
            "utf-8"
        )
    ).hexdigest()


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
    if isinstance(value, bool) or not isinstance(
        value,
        int,
    ):
        raise TypeError(
            f"{name} must be an integer."
        )

    if value <= 0:
        raise ValueError(
            f"{name} must be strictly positive."
        )


def _require_nonnegative_int(
    name: str,
    value: int,
) -> None:
    if isinstance(value, bool) or not isinstance(
        value,
        int,
    ):
        raise TypeError(
            f"{name} must be an integer."
        )

    if value < 0:
        raise ValueError(
            f"{name} must be nonnegative."
        )


def _require_boolean(
    name: str,
    value: bool,
) -> None:
    if not isinstance(value, bool):
        raise TypeError(
            f"{name} must be Boolean."
        )


def _require_layer(
    value: FunctionalMessagePassingLayer,
    *,
    name: str,
) -> None:
    if not isinstance(
        value,
        FunctionalMessagePassingLayer,
    ):
        raise TypeError(
            f"{name} must be a FunctionalMessagePassingLayer."
        )


def _resolve_method_or_property(
    value: object,
    name: str,
) -> Any:
    resolved = getattr(
        value,
        name,
        None,
    )

    if callable(resolved):
        resolved = resolved()

    return resolved


def _resolve_layer_architecture_fingerprint(
    layer: FunctionalMessagePassingLayer,
    *,
    name: str,
) -> str:
    _require_layer(
        layer,
        name=name,
    )

    fingerprint = (
        _resolve_method_or_property(
            layer,
            "architecture_fingerprint",
        )
    )

    _require_nonempty_string(
        f"{name}.architecture_fingerprint",
        fingerprint,
    )

    return fingerprint


def _resolve_layer_parameter_fingerprint(
    layer: FunctionalMessagePassingLayer,
    *,
    name: str,
) -> str:
    _require_layer(
        layer,
        name=name,
    )

    fingerprint = (
        _resolve_method_or_property(
            layer,
            "parameter_fingerprint",
        )
    )

    _require_nonempty_string(
        f"{name}.parameter_fingerprint",
        fingerprint,
    )

    return fingerprint


def _normalize_layer_sequence(
    values: Sequence[
        FunctionalMessagePassingLayer
    ],
    *,
    name: str,
) -> tuple[
    FunctionalMessagePassingLayer,
    ...,
]:
    if isinstance(
        values,
        (
            str,
            bytes,
        ),
    ) or not isinstance(
        values,
        Sequence,
    ):
        raise TypeError(
            f"{name} must be a sequence of "
            "FunctionalMessagePassingLayer objects."
        )

    resolved = tuple(values)

    for index, layer in enumerate(
        resolved
    ):
        _require_layer(
            layer,
            name=f"{name}[{index}]",
        )

    return resolved


def _unique_objects_by_identity(
    values: Sequence[object],
) -> tuple[object, ...]:
    unique: list[object] = []
    observed: set[int] = set()

    for value in values:
        identity = id(value)

        if identity in observed:
            continue

        observed.add(identity)
        unique.append(value)

    return tuple(unique)


def _depth_to_unique_object_index(
    values: Sequence[object],
) -> tuple[int, ...]:
    identity_to_index: dict[int, int] = {}
    mapping: list[int] = []

    for value in values:
        identity = id(value)

        if identity not in (
            identity_to_index
        ):
            identity_to_index[
                identity
            ] = len(
                identity_to_index
            )

        mapping.append(
            identity_to_index[
                identity
            ]
        )

    return tuple(mapping)


def _named_parameters_without_deduplication(
    module: nn.Module,
) -> tuple[
    tuple[str, nn.Parameter],
    ...,
]:
    try:
        values = module.named_parameters(
            recurse=True,
            remove_duplicate=False,
        )
    except TypeError:
        values = module.named_parameters(
            recurse=True,
        )

    return tuple(values)


def _parameter_storage_identity(
    parameter: nn.Parameter,
) -> tuple[Any, ...]:
    if not isinstance(
        parameter,
        nn.Parameter,
    ):
        raise TypeError(
            "parameter must be an nn.Parameter."
        )

    if parameter.numel() == 0:
        return (
            "empty_parameter_object",
            id(parameter),
        )

    try:
        storage = (
            parameter
            .detach()
            .untyped_storage()
        )
        data_pointer = int(
            storage.data_ptr()
        )
    except (
        AttributeError,
        RuntimeError,
    ):
        return (
            "parameter_object",
            id(parameter),
        )

    return (
        "storage",
        parameter.device.type,
        parameter.device.index,
        data_pointer,
    )


def _collect_layer_parameters(
    layer: FunctionalMessagePassingLayer,
    *,
    depth: int,
) -> tuple[
    tuple[str, nn.Parameter],
    ...,
]:
    _require_nonnegative_int(
        "depth",
        depth,
    )
    _require_layer(
        layer,
        name=f"layers_by_depth[{depth}]",
    )

    return tuple(
        (
            f"depth_{depth}.{name}",
            parameter,
        )
        for name, parameter in (
            _named_parameters_without_deduplication(
                layer
            )
        )
    )


def _assert_independent_parameter_ownership(
    layers: tuple[
        FunctionalMessagePassingLayer,
        ...,
    ],
) -> None:
    parameter_owner_by_object: dict[
        int,
        str,
    ] = {}
    parameter_owner_by_storage: dict[
        tuple[Any, ...],
        str,
    ] = {}

    for depth, layer in enumerate(
        layers
    ):
        for qualified_name, parameter in (
            _collect_layer_parameters(
                layer,
                depth=depth,
            )
        ):
            object_identity = id(
                parameter
            )

            previous_object_owner = (
                parameter_owner_by_object.get(
                    object_identity
                )
            )

            if previous_object_owner is not None:
                previous_depth = (
                    previous_object_owner
                    .split(
                        ".",
                        maxsplit=1,
                    )[0]
                )
                current_depth = (
                    qualified_name
                    .split(
                        ".",
                        maxsplit=1,
                    )[0]
                )

                if previous_depth != current_depth:
                    raise ValueError(
                        "Independent stack layers must not share exact "
                        "Parameter objects. "
                        f"Observed alias between {previous_object_owner!r} "
                        f"and {qualified_name!r}."
                    )

            parameter_owner_by_object[
                object_identity
            ] = qualified_name

            storage_identity = (
                _parameter_storage_identity(
                    parameter
                )
            )
            previous_storage_owner = (
                parameter_owner_by_storage.get(
                    storage_identity
                )
            )

            if previous_storage_owner is not None:
                previous_depth = (
                    previous_storage_owner
                    .split(
                        ".",
                        maxsplit=1,
                    )[0]
                )
                current_depth = (
                    qualified_name
                    .split(
                        ".",
                        maxsplit=1,
                    )[0]
                )

                if previous_depth != current_depth:
                    raise ValueError(
                        "Independent stack layers must not share nonempty "
                        "parameter storage. "
                        f"Observed storage alias between "
                        f"{previous_storage_owner!r} and "
                        f"{qualified_name!r}."
                    )

            parameter_owner_by_storage[
                storage_identity
            ] = qualified_name


def _assert_uniform_training_mode(
    layers: tuple[
        FunctionalMessagePassingLayer,
        ...,
    ],
) -> None:
    observed = tuple(
        bool(layer.training)
        for layer in layers
    )

    if len(set(observed)) > 1:
        raise ValueError(
            "All stack layers must share one train/eval mode before "
            "execution."
        )


# =============================================================================
# Sharing-policy normalization
# =============================================================================


def normalize_stack_sharing_policy(
    value: str,
) -> str:
    """
    Validate and normalize one stack-sharing policy name.

    Whitespace is stripped. Unknown names raise ``ValueError``.
    Canonical-but-unimplemented names raise ``NotImplementedError``.
    """

    if not isinstance(
        value,
        str,
    ):
        raise TypeError(
            "stack sharing policy must be a string."
        )

    normalized = value.strip()

    if not normalized:
        raise ValueError(
            "stack sharing policy must be a non-empty string."
        )

    if normalized not in (
        CANONICAL_STACK_SHARING_POLICIES
    ):
        raise ValueError(
            "Unknown stack sharing policy "
            f"{normalized!r}. Expected one of "
            f"{tuple(CANONICAL_STACK_SHARING_POLICIES)!r}."
        )

    if normalized not in (
        V2_0_IMPLEMENTED_STACK_SHARING_POLICIES
    ):
        raise NotImplementedError(
            "Stack sharing policy "
            f"{normalized!r} is canonical but not implemented in "
            "bounded V2.0."
        )

    return normalized


def assert_stack_sharing_policy_implemented(
    value: str,
) -> None:
    normalize_stack_sharing_policy(
        value
    )


def is_independent_stack_sharing_policy(
    value: str | "StackSharingPolicy",
) -> bool:
    return (
        resolve_stack_sharing_policy(
            value
        ).name
        == STACK_SHARING_INDEPENDENT
    )


def is_fully_shared_stack_sharing_policy(
    value: str | "StackSharingPolicy",
) -> bool:
    return (
        resolve_stack_sharing_policy(
            value
        ).name
        == STACK_SHARING_FULLY_SHARED
    )


# =============================================================================
# Immutable sharing policy
# =============================================================================


@dataclass(slots=True, frozen=True)
class StackSharingPolicy:
    """
    Immutable sharing-policy metadata independent of one particular depth.
    """

    name: str = (
        STACK_SHARING_INDEPENDENT
    )
    schema_version: str = (
        STACK_SHARING_POLICY_SCHEMA_VERSION
    )

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "name",
            normalize_stack_sharing_policy(
                self.name
            ),
        )

        _require_nonempty_string(
            "schema_version",
            self.schema_version,
        )

    @classmethod
    def independent(
        cls,
    ) -> "StackSharingPolicy":
        return cls(
            name=(
                STACK_SHARING_INDEPENDENT
            )
        )

    @classmethod
    def fully_shared(
        cls,
    ) -> "StackSharingPolicy":
        return cls(
            name=(
                STACK_SHARING_FULLY_SHARED
            )
        )

    @property
    def is_independent(self) -> bool:
        return (
            self.name
            == STACK_SHARING_INDEPENDENT
        )

    @property
    def is_fully_shared(self) -> bool:
        return (
            self.name
            == STACK_SHARING_FULLY_SHARED
        )

    @property
    def requires_distinct_layer_objects(
        self,
    ) -> bool:
        return self.is_independent

    @property
    def requires_exact_layer_reuse(
        self,
    ) -> bool:
        return self.is_fully_shared

    def expected_unique_layer_count(
        self,
        *,
        num_layers: int,
    ) -> int:
        _require_positive_int(
            "num_layers",
            num_layers,
        )

        if self.is_independent:
            return num_layers

        if self.is_fully_shared:
            return 1

        raise RuntimeError(
            "Unreachable stack-sharing branch."
        )

    def expected_depth_to_owner_index(
        self,
        *,
        num_layers: int,
    ) -> tuple[int, ...]:
        _require_positive_int(
            "num_layers",
            num_layers,
        )

        if self.is_independent:
            return tuple(
                range(num_layers)
            )

        if self.is_fully_shared:
            return tuple(
                0
                for _ in range(
                    num_layers
                )
            )

        raise RuntimeError(
            "Unreachable stack-sharing branch."
        )

    def registration_prefixes(
        self,
        *,
        num_layers: int,
    ) -> tuple[str, ...]:
        _require_positive_int(
            "num_layers",
            num_layers,
        )

        if self.is_independent:
            return tuple(
                f"{INDEPENDENT_LAYER_STATE_DICT_PREFIX}.{index}"
                for index in range(
                    num_layers
                )
            )

        if self.is_fully_shared:
            return (
                FULLY_SHARED_LAYER_STATE_DICT_PREFIX,
            )

        raise RuntimeError(
            "Unreachable stack-sharing branch."
        )

    def architecture_dict(
        self,
    ) -> dict[str, Any]:
        return {
            "schema_version": (
                self.schema_version
            ),
            "name": (
                self.name
            ),
            "requires_distinct_layer_objects": (
                self.requires_distinct_layer_objects
            ),
            "requires_exact_layer_reuse": (
                self.requires_exact_layer_reuse
            ),
            "partial_sharing_supported": (
                STACK_SHARING_PARTIAL_SUPPORTED
            ),
        }

    def architecture_fingerprint(
        self,
    ) -> str:
        return _fingerprint(
            self.architecture_dict()
        )


def resolve_stack_sharing_policy(
    value: str | StackSharingPolicy,
) -> StackSharingPolicy:
    if isinstance(
        value,
        StackSharingPolicy,
    ):
        return value

    if isinstance(
        value,
        str,
    ):
        return StackSharingPolicy(
            name=value
        )

    raise TypeError(
        "sharing_policy must be a string or StackSharingPolicy."
    )


# =============================================================================
# Immutable layer-ownership plan
# =============================================================================


@dataclass(slots=True, frozen=True)
class StackLayerSharingPlan:
    """
    Exact depth-to-layer ownership plan for one stack.

    ``layers_by_depth`` always contains one resolved layer object per executed
    depth. Under ``fully_shared`` every tuple entry is the same exact object.
    Under ``independent`` every tuple entry is a distinct object with no
    cross-depth parameter aliasing.
    """

    policy: StackSharingPolicy | str
    num_layers: int

    layers_by_depth: tuple[
        FunctionalMessagePassingLayer,
        ...,
    ]

    require_uniform_training_mode: bool = True

    schema_version: str = (
        STACK_LAYER_SHARING_PLAN_SCHEMA_VERSION
    )

    def __post_init__(self) -> None:
        policy = (
            resolve_stack_sharing_policy(
                self.policy
            )
        )
        object.__setattr__(
            self,
            "policy",
            policy,
        )

        _require_positive_int(
            "num_layers",
            self.num_layers,
        )
        _require_boolean(
            "require_uniform_training_mode",
            self.require_uniform_training_mode,
        )

        layers = (
            _normalize_layer_sequence(
                self.layers_by_depth,
                name="layers_by_depth",
            )
        )
        object.__setattr__(
            self,
            "layers_by_depth",
            layers,
        )

        if len(layers) != (
            self.num_layers
        ):
            raise ValueError(
                "layers_by_depth must contain exactly one resolved layer "
                "object per stack depth."
            )

        unique_layers = tuple(
            _unique_objects_by_identity(
                layers
            )
        )
        expected_unique_count = (
            policy.expected_unique_layer_count(
                num_layers=(
                    self.num_layers
                )
            )
        )

        if len(unique_layers) != (
            expected_unique_count
        ):
            if policy.is_independent:
                raise ValueError(
                    "Independent sharing requires one distinct exact "
                    "FunctionalMessagePassingLayer object per depth."
                )

            raise ValueError(
                "Fully shared execution requires one exact "
                "FunctionalMessagePassingLayer object reused at every "
                "depth."
            )

        observed_mapping = (
            _depth_to_unique_object_index(
                layers
            )
        )
        expected_mapping = (
            policy
            .expected_depth_to_owner_index(
                num_layers=(
                    self.num_layers
                )
            )
        )

        if observed_mapping != (
            expected_mapping
        ):
            raise ValueError(
                "Depth-to-layer ownership does not match the selected "
                "stack sharing policy."
            )

        if policy.is_independent:
            _assert_independent_parameter_ownership(
                layers
            )

        if self.require_uniform_training_mode:
            _assert_uniform_training_mode(
                layers
            )

        hidden_dims = tuple(
            int(layer.hidden_dim)
            for layer in layers
        )

        if len(set(hidden_dims)) != 1:
            raise ValueError(
                "All stack layers must use one constant hidden width."
            )

        relation_names = tuple(
            tuple(layer.relation_names)
            for layer in layers
        )

        if len(set(relation_names)) != 1:
            raise ValueError(
                "All stack layers must use the same exact relation ordering."
            )

        stable_relation_ids = tuple(
            tuple(
                layer.stable_relation_ids
            )
            for layer in layers
        )

        if len(
            set(
                stable_relation_ids
            )
        ) != 1:
            raise ValueError(
                "All stack layers must use the same stable relation IDs."
            )

        registry_fingerprints = tuple(
            str(
                layer
                .compiled_relation_registry_fingerprint
            )
            for layer in layers
        )

        if len(
            set(
                registry_fingerprints
            )
        ) != 1:
            raise ValueError(
                "All stack layers must reference the same compiled "
                "relation registry fingerprint."
            )

        for index, fingerprint in enumerate(
            registry_fingerprints
        ):
            _require_nonempty_string(
                (
                    "layers_by_depth"
                    f"[{index}]."
                    "compiled_relation_registry_fingerprint"
                ),
                fingerprint,
            )

        for index, layer in enumerate(
            layers
        ):
            _resolve_layer_architecture_fingerprint(
                layer,
                name=(
                    f"layers_by_depth[{index}]"
                ),
            )
            _resolve_layer_parameter_fingerprint(
                layer,
                name=(
                    f"layers_by_depth[{index}]"
                ),
            )

        _require_nonempty_string(
            "schema_version",
            self.schema_version,
        )

    @property
    def sharing_policy(self) -> str:
        return self.policy.name

    @property
    def hidden_dim(self) -> int:
        return int(
            self.layers_by_depth[0]
            .hidden_dim
        )

    @property
    def relation_names(
        self,
    ) -> tuple[str, ...]:
        return tuple(
            self.layers_by_depth[0]
            .relation_names
        )

    @property
    def stable_relation_ids(
        self,
    ) -> tuple[int, ...]:
        return tuple(
            self.layers_by_depth[0]
            .stable_relation_ids
        )

    @property
    def compiled_relation_registry_fingerprint(
        self,
    ) -> str:
        return str(
            self.layers_by_depth[0]
            .compiled_relation_registry_fingerprint
        )

    @property
    def unique_layers(
        self,
    ) -> tuple[
        FunctionalMessagePassingLayer,
        ...,
    ]:
        return tuple(
            _unique_objects_by_identity(
                self.layers_by_depth
            )
        )

    @property
    def num_unique_layers(self) -> int:
        return len(
            self.unique_layers
        )

    @property
    def depth_to_unique_layer_index(
        self,
    ) -> tuple[int, ...]:
        return (
            _depth_to_unique_object_index(
                self.layers_by_depth
            )
        )

    @property
    def unique_layer_registration_prefixes(
        self,
    ) -> tuple[str, ...]:
        return (
            self.policy
            .registration_prefixes(
                num_layers=(
                    self.num_layers
                )
            )
        )

    @property
    def training(self) -> bool:
        return bool(
            self.layers_by_depth[0]
            .training
        )

    def layer_for_depth(
        self,
        depth: int,
    ) -> FunctionalMessagePassingLayer:
        _require_nonnegative_int(
            "depth",
            depth,
        )

        if depth >= self.num_layers:
            raise IndexError(
                "depth lies outside the configured stack."
            )

        return self.layers_by_depth[
            depth
        ]

    def unique_layer_for_owner_index(
        self,
        owner_index: int,
    ) -> FunctionalMessagePassingLayer:
        _require_nonnegative_int(
            "owner_index",
            owner_index,
        )

        if owner_index >= (
            self.num_unique_layers
        ):
            raise IndexError(
                "owner_index lies outside the unique-layer ownership plan."
            )

        return self.unique_layers[
            owner_index
        ]

    def layer_architecture_fingerprints_by_depth(
        self,
    ) -> tuple[str, ...]:
        return tuple(
            _resolve_layer_architecture_fingerprint(
                layer,
                name=f"layers_by_depth[{index}]",
            )
            for index, layer in enumerate(
                self.layers_by_depth
            )
        )

    def unique_layer_architecture_fingerprints(
        self,
    ) -> tuple[str, ...]:
        return tuple(
            _resolve_layer_architecture_fingerprint(
                layer,
                name=f"unique_layers[{index}]",
            )
            for index, layer in enumerate(
                self.unique_layers
            )
        )

    def layer_parameter_fingerprints_by_depth(
        self,
    ) -> tuple[str, ...]:
        return tuple(
            _resolve_layer_parameter_fingerprint(
                layer,
                name=f"layers_by_depth[{index}]",
            )
            for index, layer in enumerate(
                self.layers_by_depth
            )
        )

    def unique_layer_parameter_fingerprints(
        self,
    ) -> tuple[str, ...]:
        return tuple(
            _resolve_layer_parameter_fingerprint(
                layer,
                name=f"unique_layers[{index}]",
            )
            for index, layer in enumerate(
                self.unique_layers
            )
        )

    def numerical_architecture_dict(
        self,
    ) -> dict[str, Any]:
        return {
            "schema_version": (
                self.schema_version
            ),
            "policy": (
                self.policy
                .architecture_dict()
            ),
            "num_layers": (
                self.num_layers
            ),
            "num_unique_layers": (
                self.num_unique_layers
            ),
            "depth_to_unique_layer_index": list(
                self.depth_to_unique_layer_index
            ),
            "hidden_dim": (
                self.hidden_dim
            ),
            "relation_names": list(
                self.relation_names
            ),
            "stable_relation_ids": list(
                self.stable_relation_ids
            ),
            "compiled_relation_registry_fingerprint": (
                self
                .compiled_relation_registry_fingerprint
            ),
            "unique_layer_architecture_fingerprints": list(
                self
                .unique_layer_architecture_fingerprints()
            ),
        }

    def architecture_fingerprint(
        self,
    ) -> str:
        return _fingerprint(
            self.numerical_architecture_dict()
        )

    def parameter_ownership_dict(
        self,
    ) -> dict[str, Any]:
        return {
            "schema_version": (
                self.schema_version
            ),
            "sharing_policy": (
                self.sharing_policy
            ),
            "num_layers": (
                self.num_layers
            ),
            "num_unique_layers": (
                self.num_unique_layers
            ),
            "depth_to_unique_layer_index": list(
                self.depth_to_unique_layer_index
            ),
            "unique_layer_registration_prefixes": list(
                self
                .unique_layer_registration_prefixes
            ),
            "unique_layer_parameter_fingerprints": list(
                self
                .unique_layer_parameter_fingerprints()
            ),
        }

    def parameter_fingerprint(
        self,
    ) -> str:
        return _fingerprint(
            self.parameter_ownership_dict()
        )

    def validate_current_ownership(
        self,
        *,
        require_uniform_training_mode: bool | None = None,
    ) -> None:
        """
        Revalidate module identity, parameter ownership, and architecture.

        This catches post-construction module replacement or accidental
        cross-depth parameter tying. Parameter values may legitimately change
        during optimization; only ownership and finite component-level
        fingerprint availability are checked.
        """

        if require_uniform_training_mode is None:
            require_uniform = (
                self.require_uniform_training_mode
            )
        else:
            _require_boolean(
                "require_uniform_training_mode",
                require_uniform_training_mode,
            )
            require_uniform = (
                require_uniform_training_mode
            )

        rebuilt = StackLayerSharingPlan(
            policy=self.policy,
            num_layers=self.num_layers,
            layers_by_depth=(
                self.layers_by_depth
            ),
            require_uniform_training_mode=(
                require_uniform
            ),
            schema_version=(
                self.schema_version
            ),
        )

        if (
            rebuilt.depth_to_unique_layer_index
            != self.depth_to_unique_layer_index
        ):
            raise ValueError(
                "Current depth-to-owner mapping differs from the frozen "
                "sharing plan."
            )


# =============================================================================
# Plan construction
# =============================================================================


def build_independent_stack_layer_sharing_plan(
    layers: Sequence[
        FunctionalMessagePassingLayer
    ],
    *,
    require_uniform_training_mode: bool = True,
) -> StackLayerSharingPlan:
    resolved = _normalize_layer_sequence(
        layers,
        name="layers",
    )

    if not resolved:
        raise ValueError(
            "Independent stack construction requires at least one layer."
        )

    return StackLayerSharingPlan(
        policy=(
            StackSharingPolicy
            .independent()
        ),
        num_layers=len(
            resolved
        ),
        layers_by_depth=resolved,
        require_uniform_training_mode=(
            require_uniform_training_mode
        ),
    )


def build_fully_shared_stack_layer_sharing_plan(
    layer: FunctionalMessagePassingLayer,
    *,
    num_layers: int,
    require_uniform_training_mode: bool = True,
) -> StackLayerSharingPlan:
    _require_layer(
        layer,
        name="layer",
    )
    _require_positive_int(
        "num_layers",
        num_layers,
    )

    return StackLayerSharingPlan(
        policy=(
            StackSharingPolicy
            .fully_shared()
        ),
        num_layers=num_layers,
        layers_by_depth=tuple(
            layer
            for _ in range(
                num_layers
            )
        ),
        require_uniform_training_mode=(
            require_uniform_training_mode
        ),
    )


def build_stack_layer_sharing_plan(
    layers: LayerOrLayerSequence,
    *,
    num_layers: int,
    sharing_policy: (
        str
        | StackSharingPolicy
    ) = STACK_SHARING_INDEPENDENT,
    require_uniform_training_mode: bool = True,
) -> StackLayerSharingPlan:
    """
    Build one exact layer-ownership plan from explicit layer objects.

    For ``independent``, ``layers`` must be a sequence of length
    ``num_layers``.

    For ``fully_shared``, ``layers`` may be:

    - one layer object;
    - a one-element sequence;
    - a sequence of length ``num_layers`` whose every element is the same
      exact layer object.
    """

    policy = (
        resolve_stack_sharing_policy(
            sharing_policy
        )
    )
    _require_positive_int(
        "num_layers",
        num_layers,
    )
    _require_boolean(
        "require_uniform_training_mode",
        require_uniform_training_mode,
    )

    if policy.is_independent:
        if isinstance(
            layers,
            FunctionalMessagePassingLayer,
        ):
            raise TypeError(
                "Independent sharing requires a sequence containing one "
                "distinct layer object per depth."
            )

        resolved_layers = (
            _normalize_layer_sequence(
                layers,
                name="layers",
            )
        )

        if len(resolved_layers) != (
            num_layers
        ):
            raise ValueError(
                "Independent sharing requires exactly num_layers layer "
                "objects."
            )

        return StackLayerSharingPlan(
            policy=policy,
            num_layers=num_layers,
            layers_by_depth=(
                resolved_layers
            ),
            require_uniform_training_mode=(
                require_uniform_training_mode
            ),
        )

    if policy.is_fully_shared:
        if isinstance(
            layers,
            FunctionalMessagePassingLayer,
        ):
            shared_layer = layers
            resolved_layers = tuple(
                shared_layer
                for _ in range(
                    num_layers
                )
            )
        else:
            supplied = (
                _normalize_layer_sequence(
                    layers,
                    name="layers",
                )
            )

            if len(supplied) == 1:
                shared_layer = (
                    supplied[0]
                )
                resolved_layers = tuple(
                    shared_layer
                    for _ in range(
                        num_layers
                    )
                )
            elif len(supplied) == (
                num_layers
            ):
                shared_layer = (
                    supplied[0]
                )

                if any(
                    layer is not shared_layer
                    for layer in supplied[1:]
                ):
                    raise ValueError(
                        "Fully shared execution requires the same exact "
                        "layer object at every depth."
                    )

                resolved_layers = supplied
            else:
                raise ValueError(
                    "Fully shared construction requires one layer object, "
                    "a one-element sequence, or a num_layers-length "
                    "sequence containing one repeated exact object."
                )

        return StackLayerSharingPlan(
            policy=policy,
            num_layers=num_layers,
            layers_by_depth=(
                resolved_layers
            ),
            require_uniform_training_mode=(
                require_uniform_training_mode
            ),
        )

    raise RuntimeError(
        "Unreachable stack-sharing branch."
    )


def build_stack_layer_sharing_plan_from_factory(
    layer_factory: StackLayerFactory,
    *,
    num_layers: int,
    sharing_policy: (
        str
        | StackSharingPolicy
    ) = STACK_SHARING_INDEPENDENT,
    require_uniform_training_mode: bool = True,
) -> StackLayerSharingPlan:
    """
    Construct a sharing plan from an explicit depth-aware layer factory.

    The factory is called once per depth under ``independent`` and exactly once
    under ``fully_shared``. The function never uses ``copy`` or ``deepcopy``;
    callers retain full control over layer construction and initialization.
    """

    if not callable(
        layer_factory
    ):
        raise TypeError(
            "layer_factory must be callable."
        )

    _require_positive_int(
        "num_layers",
        num_layers,
    )
    policy = (
        resolve_stack_sharing_policy(
            sharing_policy
        )
    )

    if policy.is_independent:
        layers = tuple(
            layer_factory(depth)
            for depth in range(
                num_layers
            )
        )

        for depth, layer in enumerate(
            layers
        ):
            _require_layer(
                layer,
                name=(
                    f"layer_factory({depth})"
                ),
            )

        return StackLayerSharingPlan(
            policy=policy,
            num_layers=num_layers,
            layers_by_depth=layers,
            require_uniform_training_mode=(
                require_uniform_training_mode
            ),
        )

    if policy.is_fully_shared:
        layer = layer_factory(0)
        _require_layer(
            layer,
            name="layer_factory(0)",
        )

        return (
            build_fully_shared_stack_layer_sharing_plan(
                layer,
                num_layers=num_layers,
                require_uniform_training_mode=(
                    require_uniform_training_mode
                ),
            )
        )

    raise RuntimeError(
        "Unreachable stack-sharing branch."
    )


# =============================================================================
# Validation and public aliases
# =============================================================================


def validate_stack_layer_sharing_plan(
    plan: StackLayerSharingPlan,
    *,
    expected_num_layers: int | None = None,
    expected_sharing_policy: (
        str
        | StackSharingPolicy
        | None
    ) = None,
    require_uniform_training_mode: bool | None = None,
) -> None:
    if not isinstance(
        plan,
        StackLayerSharingPlan,
    ):
        raise TypeError(
            "plan must be a StackLayerSharingPlan."
        )

    plan.validate_current_ownership(
        require_uniform_training_mode=(
            require_uniform_training_mode
        ),
    )

    if expected_num_layers is not None:
        _require_positive_int(
            "expected_num_layers",
            expected_num_layers,
        )

        if plan.num_layers != (
            expected_num_layers
        ):
            raise ValueError(
                "Stack layer-sharing plan depth differs from the expected "
                "num_layers."
            )

    if expected_sharing_policy is not None:
        expected_policy = (
            resolve_stack_sharing_policy(
                expected_sharing_policy
            )
        )

        if plan.policy != (
            expected_policy
        ):
            raise ValueError(
                "Stack layer-sharing plan policy differs from the expected "
                "sharing policy."
            )


def layer_for_stack_depth(
    plan: StackLayerSharingPlan,
    depth: int,
) -> FunctionalMessagePassingLayer:
    validate_stack_layer_sharing_plan(
        plan,
        require_uniform_training_mode=False,
    )
    return plan.layer_for_depth(
        depth
    )


SharingPolicy = StackSharingPolicy
LayerSharingPlan = StackLayerSharingPlan

normalize_sharing_policy = (
    normalize_stack_sharing_policy
)
resolve_sharing_policy = (
    resolve_stack_sharing_policy
)
build_layer_sharing_plan = (
    build_stack_layer_sharing_plan
)
build_layer_sharing_plan_from_factory = (
    build_stack_layer_sharing_plan_from_factory
)
validate_layer_sharing_plan = (
    validate_stack_layer_sharing_plan
)


__all__ = (
    # Public identity.
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
    # Policy normalization.
    "normalize_stack_sharing_policy",
    "normalize_sharing_policy",
    "assert_stack_sharing_policy_implemented",
    "is_independent_stack_sharing_policy",
    "is_fully_shared_stack_sharing_policy",
    # Immutable policy.
    "StackSharingPolicy",
    "SharingPolicy",
    "resolve_stack_sharing_policy",
    "resolve_sharing_policy",
    # Layer ownership.
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
)
