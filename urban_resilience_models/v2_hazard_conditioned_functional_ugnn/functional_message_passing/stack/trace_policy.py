"""
Trace-detail and output-retention policy for functional message-passing stacks.

Target path:
    urban_resilience_models/
        v2_hazard_conditioned_functional_ugnn/
            functional_message_passing/
                stack/
                    trace_policy.py

This module separates two independent runtime decisions:

1. Stack output retention
   Which public layer outputs remain available after the stack finishes.

2. Layer trace detail
   How much internal detail each executed layer output may retain.

The distinction is deliberate:

``StackRetentionPolicy``
    ``none``
        retain no public layer outputs;

    ``final_layer``
        retain only the final public layer output;

    ``all_layers``
        retain every public layer output.

``LayerTracePolicy``
    imported from ``functional_message_passing.layer.schemas``;

    ``none``
        retain no optional layer trace;

    ``node``
        retain node-level aggregation, residual, and normalization stages;

    ``full``
        retain the complete edge- and node-level layer trace.

These policies are orthogonal. For example:

    retention = all_layers
    layer trace = none

retains every lightweight public layer output without edge-level trace
objects, while:

    retention = final_layer
    layer trace = full

retains only the final public output but permits that output to carry its full
one-layer trace.

This module owns:

- stack-retention normalization and validation;
- immutable retention metadata;
- immutable combined stack trace-policy metadata;
- exact retained-index calculation;
- per-depth retention decisions;
- execution-contract dictionaries and fingerprints;
- compatibility helpers for string and typed-policy call sites.

It does not own:

- numerical layer execution;
- stack parameter sharing;
- state rebinding;
- diagnostic generation;
- complete audit-trace construction;
- checkpoint serialization;
- training-loss reduction.

Bounded V2.0 rules
------------------
- Stack depth is strictly positive.
- Retained indices are zero-based and strictly policy-derived.
- Trace and retention policies never alter numerical results.
- Audit mode is explicit and does not silently force public output retention.
- Future canonical policies raise ``NotImplementedError`` until implemented.
"""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
from typing import Any, Final, Iterable, Sequence

from ...constants import (
    CANONICAL_STACK_RETENTION_POLICIES,
    STACK_RETENTION_ALL_LAYERS,
    STACK_RETENTION_FINAL_LAYER,
    STACK_RETENTION_NONE,
    V2_0_IMPLEMENTED_STACK_RETENTION_POLICIES,
)
from ..layer.schemas import (
    LAYER_TRACE_FULL,
    LAYER_TRACE_NODE,
    LAYER_TRACE_NONE,
    LayerTracePolicy,
)
from .schemas import expected_retained_layer_indices


# =============================================================================
# Public identity
# =============================================================================


STACK_RETENTION_POLICY_SCHEMA_VERSION: Final[str] = "0.1"
STACK_TRACE_POLICY_SCHEMA_VERSION: Final[str] = "0.1"
STACK_TRACE_DECISION_SCHEMA_VERSION: Final[str] = "0.1"

STACK_RETENTION_AFFECTS_NUMERICAL_RESULTS: Final[bool] = False
STACK_LAYER_TRACE_AFFECTS_NUMERICAL_RESULTS: Final[bool] = False
STACK_AUDIT_MODE_AFFECTS_NUMERICAL_RESULTS: Final[bool] = False

STACK_RETENTION_NONE_RETAINS_FINAL_LAYER: Final[bool] = False
STACK_RETENTION_FINAL_LAYER_RETAINS_FINAL_LAYER: Final[bool] = True
STACK_RETENTION_ALL_LAYERS_RETAINS_FINAL_LAYER: Final[bool] = True

STACK_RETENTION_POLICY_IS_OUTPUT_CONTRACT: Final[bool] = True
STACK_LAYER_TRACE_POLICY_IS_OUTPUT_CONTRACT: Final[bool] = True
STACK_TRACE_POLICY_IS_NUMERICAL_ARCHITECTURE: Final[bool] = False

STACK_TRACE_POLICY_SCIENTIFIC_INTERPRETATION: Final[str] = (
    "runtime_observability_contract_without_numerical_model_change"
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


def _require_integer_sequence(
    name: str,
    values: Sequence[int],
) -> tuple[int, ...]:
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
            f"{name} must be a sequence of integers."
        )

    resolved = tuple(values)

    for index, value in enumerate(
        resolved
    ):
        if isinstance(value, bool) or not isinstance(
            value,
            int,
        ):
            raise TypeError(
                f"{name}[{index}] must be an integer."
            )

        if value < 0:
            raise ValueError(
                f"{name}[{index}] must be nonnegative."
            )

    return resolved


def _resolve_layer_trace_policy(
    value: str | LayerTracePolicy,
) -> LayerTracePolicy:
    if isinstance(
        value,
        LayerTracePolicy,
    ):
        policy = value
    elif isinstance(
        value,
        str,
    ):
        policy = LayerTracePolicy(
            mode=value
        )
    else:
        raise TypeError(
            "layer_trace_policy must be a string or LayerTracePolicy."
        )

    policy.assert_implemented()
    return policy


# =============================================================================
# Stack-retention normalization
# =============================================================================


def normalize_stack_retention_policy(
    value: str,
) -> str:
    """
    Validate and normalize one stack-retention policy name.

    Leading and trailing whitespace are removed. Unknown names raise
    ``ValueError``. Canonical-but-unimplemented names raise
    ``NotImplementedError``.
    """

    if not isinstance(
        value,
        str,
    ):
        raise TypeError(
            "stack retention policy must be a string."
        )

    normalized = value.strip()

    if not normalized:
        raise ValueError(
            "stack retention policy must be a non-empty string."
        )

    if normalized not in (
        CANONICAL_STACK_RETENTION_POLICIES
    ):
        raise ValueError(
            "Unknown stack retention policy "
            f"{normalized!r}. Expected one of "
            f"{tuple(CANONICAL_STACK_RETENTION_POLICIES)!r}."
        )

    if normalized not in (
        V2_0_IMPLEMENTED_STACK_RETENTION_POLICIES
    ):
        raise NotImplementedError(
            "Stack retention policy "
            f"{normalized!r} is canonical but not implemented in "
            "bounded V2.0."
        )

    return normalized


def assert_stack_retention_policy_implemented(
    value: str,
) -> None:
    normalize_stack_retention_policy(
        value
    )


def resolve_stack_retention_policy(
    value: str | "StackRetentionPolicy",
) -> "StackRetentionPolicy":
    if isinstance(
        value,
        StackRetentionPolicy,
    ):
        return value

    if isinstance(
        value,
        str,
    ):
        return StackRetentionPolicy(
            name=value
        )

    raise TypeError(
        "retention_policy must be a string or StackRetentionPolicy."
    )


def is_no_stack_retention(
    value: str | "StackRetentionPolicy",
) -> bool:
    return (
        resolve_stack_retention_policy(
            value
        ).name
        == STACK_RETENTION_NONE
    )


def is_final_layer_stack_retention(
    value: str | "StackRetentionPolicy",
) -> bool:
    return (
        resolve_stack_retention_policy(
            value
        ).name
        == STACK_RETENTION_FINAL_LAYER
    )


def is_all_layers_stack_retention(
    value: str | "StackRetentionPolicy",
) -> bool:
    return (
        resolve_stack_retention_policy(
            value
        ).name
        == STACK_RETENTION_ALL_LAYERS
    )


# =============================================================================
# Immutable stack-retention policy
# =============================================================================


@dataclass(slots=True, frozen=True)
class StackRetentionPolicy:
    """
    Immutable policy describing which public layer outputs survive execution.
    """

    name: str = (
        STACK_RETENTION_NONE
    )
    schema_version: str = (
        STACK_RETENTION_POLICY_SCHEMA_VERSION
    )

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "name",
            normalize_stack_retention_policy(
                self.name
            ),
        )

        _require_nonempty_string(
            "schema_version",
            self.schema_version,
        )

    @classmethod
    def none(
        cls,
    ) -> "StackRetentionPolicy":
        return cls(
            name=(
                STACK_RETENTION_NONE
            )
        )

    @classmethod
    def final_layer(
        cls,
    ) -> "StackRetentionPolicy":
        return cls(
            name=(
                STACK_RETENTION_FINAL_LAYER
            )
        )

    @classmethod
    def all_layers(
        cls,
    ) -> "StackRetentionPolicy":
        return cls(
            name=(
                STACK_RETENTION_ALL_LAYERS
            )
        )

    @property
    def retains_none(self) -> bool:
        return (
            self.name
            == STACK_RETENTION_NONE
        )

    @property
    def retains_final_layer_only(
        self,
    ) -> bool:
        return (
            self.name
            == STACK_RETENTION_FINAL_LAYER
        )

    @property
    def retains_all_layers(
        self,
    ) -> bool:
        return (
            self.name
            == STACK_RETENTION_ALL_LAYERS
        )

    @property
    def retains_any_layer(self) -> bool:
        return not self.retains_none

    @property
    def retains_final_layer(self) -> bool:
        return (
            self.retains_final_layer_only
            or self.retains_all_layers
        )

    def expected_indices(
        self,
        *,
        num_layers: int,
    ) -> tuple[int, ...]:
        return expected_retained_layer_indices(
            retention_policy=(
                self.name
            ),
            num_layers=num_layers,
        )

    def expected_count(
        self,
        *,
        num_layers: int,
    ) -> int:
        return len(
            self.expected_indices(
                num_layers=num_layers
            )
        )

    def should_retain(
        self,
        *,
        layer_index: int,
        num_layers: int,
    ) -> bool:
        _require_nonnegative_int(
            "layer_index",
            layer_index,
        )
        _require_positive_int(
            "num_layers",
            num_layers,
        )

        if layer_index >= num_layers:
            raise IndexError(
                "layer_index lies outside the configured stack."
            )

        if self.retains_none:
            return False

        if self.retains_final_layer_only:
            return (
                layer_index
                == num_layers - 1
            )

        if self.retains_all_layers:
            return True

        raise RuntimeError(
            "Unreachable stack-retention branch."
        )

    def validate_retained_indices(
        self,
        values: Sequence[int],
        *,
        num_layers: int,
    ) -> None:
        observed = _require_integer_sequence(
            "retained_layer_indices",
            values,
        )
        expected = self.expected_indices(
            num_layers=num_layers
        )

        if observed != expected:
            raise ValueError(
                "retained_layer_indices do not match the selected stack "
                f"retention policy. Expected {expected!r}; "
                f"observed {observed!r}."
            )

    def execution_contract_dict(
        self,
    ) -> dict[str, Any]:
        return {
            "schema_version": (
                self.schema_version
            ),
            "name": (
                self.name
            ),
            "retains_none": (
                self.retains_none
            ),
            "retains_final_layer_only": (
                self.retains_final_layer_only
            ),
            "retains_all_layers": (
                self.retains_all_layers
            ),
            "affects_numerical_results": (
                STACK_RETENTION_AFFECTS_NUMERICAL_RESULTS
            ),
        }

    def execution_contract_fingerprint(
        self,
    ) -> str:
        return _fingerprint(
            self.execution_contract_dict()
        )


# =============================================================================
# Per-depth trace and retention decision
# =============================================================================


@dataclass(slots=True, frozen=True)
class StackTraceDecision:
    """
    Exact trace and output-retention decision for one stack depth.
    """

    layer_index: int
    num_layers: int

    retain_public_output: bool
    is_final_layer: bool

    layer_trace_policy: (
        LayerTracePolicy | str
    )

    audit_mode: bool = False

    schema_version: str = (
        STACK_TRACE_DECISION_SCHEMA_VERSION
    )

    def __post_init__(self) -> None:
        _require_nonnegative_int(
            "layer_index",
            self.layer_index,
        )
        _require_positive_int(
            "num_layers",
            self.num_layers,
        )

        if self.layer_index >= (
            self.num_layers
        ):
            raise ValueError(
                "layer_index must be smaller than num_layers."
            )

        _require_boolean(
            "retain_public_output",
            self.retain_public_output,
        )
        _require_boolean(
            "is_final_layer",
            self.is_final_layer,
        )
        _require_boolean(
            "audit_mode",
            self.audit_mode,
        )

        expected_final = (
            self.layer_index
            == self.num_layers - 1
        )

        if self.is_final_layer != (
            expected_final
        ):
            raise ValueError(
                "is_final_layer does not match layer_index and num_layers."
            )

        object.__setattr__(
            self,
            "layer_trace_policy",
            _resolve_layer_trace_policy(
                self.layer_trace_policy
            ),
        )

        _require_nonempty_string(
            "schema_version",
            self.schema_version,
        )

    @property
    def layer_trace_mode(self) -> str:
        return self.layer_trace_policy.mode

    @property
    def captures_layer_trace(self) -> bool:
        return (
            self.layer_trace_mode
            != LAYER_TRACE_NONE
        )

    @property
    def captures_node_trace(self) -> bool:
        return self.layer_trace_mode in (
            LAYER_TRACE_NODE,
            LAYER_TRACE_FULL,
        )

    @property
    def captures_full_trace(self) -> bool:
        return (
            self.layer_trace_mode
            == LAYER_TRACE_FULL
        )

    @property
    def retains_or_audits_layer(
        self,
    ) -> bool:
        return (
            self.retain_public_output
            or self.audit_mode
        )

    def execution_contract_dict(
        self,
    ) -> dict[str, Any]:
        return {
            "schema_version": (
                self.schema_version
            ),
            "layer_index": (
                self.layer_index
            ),
            "num_layers": (
                self.num_layers
            ),
            "retain_public_output": (
                self.retain_public_output
            ),
            "is_final_layer": (
                self.is_final_layer
            ),
            "layer_trace_policy": (
                self
                .layer_trace_policy
                .architecture_dict()
            ),
            "audit_mode": (
                self.audit_mode
            ),
        }

    def execution_contract_fingerprint(
        self,
    ) -> str:
        return _fingerprint(
            self.execution_contract_dict()
        )


# =============================================================================
# Combined stack trace policy
# =============================================================================


@dataclass(slots=True, frozen=True)
class StackTracePolicy:
    """
    Immutable combined output-retention and layer-trace execution contract.
    """

    retention_policy: (
        StackRetentionPolicy | str
    ) = STACK_RETENTION_NONE

    layer_trace_policy: (
        LayerTracePolicy | str
    ) = LAYER_TRACE_NONE

    audit_mode: bool = False

    schema_version: str = (
        STACK_TRACE_POLICY_SCHEMA_VERSION
    )

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "retention_policy",
            resolve_stack_retention_policy(
                self.retention_policy
            ),
        )
        object.__setattr__(
            self,
            "layer_trace_policy",
            _resolve_layer_trace_policy(
                self.layer_trace_policy
            ),
        )

        _require_boolean(
            "audit_mode",
            self.audit_mode,
        )
        _require_nonempty_string(
            "schema_version",
            self.schema_version,
        )

    @classmethod
    def minimal(
        cls,
    ) -> "StackTracePolicy":
        return cls(
            retention_policy=(
                STACK_RETENTION_NONE
            ),
            layer_trace_policy=(
                LAYER_TRACE_NONE
            ),
            audit_mode=False,
        )

    @classmethod
    def retain_final_layer(
        cls,
        *,
        layer_trace_policy: (
            LayerTracePolicy | str
        ) = LAYER_TRACE_NONE,
    ) -> "StackTracePolicy":
        return cls(
            retention_policy=(
                STACK_RETENTION_FINAL_LAYER
            ),
            layer_trace_policy=(
                layer_trace_policy
            ),
            audit_mode=False,
        )

    @classmethod
    def retain_all_layers(
        cls,
        *,
        layer_trace_policy: (
            LayerTracePolicy | str
        ) = LAYER_TRACE_NONE,
    ) -> "StackTracePolicy":
        return cls(
            retention_policy=(
                STACK_RETENTION_ALL_LAYERS
            ),
            layer_trace_policy=(
                layer_trace_policy
            ),
            audit_mode=False,
        )

    @classmethod
    def full_audit(
        cls,
        *,
        retention_policy: (
            StackRetentionPolicy | str
        ) = STACK_RETENTION_ALL_LAYERS,
        layer_trace_policy: (
            LayerTracePolicy | str
        ) = LAYER_TRACE_FULL,
    ) -> "StackTracePolicy":
        return cls(
            retention_policy=(
                retention_policy
            ),
            layer_trace_policy=(
                layer_trace_policy
            ),
            audit_mode=True,
        )

    @property
    def retention_name(self) -> str:
        return self.retention_policy.name

    @property
    def layer_trace_mode(self) -> str:
        return self.layer_trace_policy.mode

    @property
    def retains_any_layer(self) -> bool:
        return (
            self.retention_policy
            .retains_any_layer
        )

    @property
    def retains_final_layer(self) -> bool:
        return (
            self.retention_policy
            .retains_final_layer
        )

    @property
    def captures_any_layer_trace(
        self,
    ) -> bool:
        return (
            self.layer_trace_mode
            != LAYER_TRACE_NONE
        )

    @property
    def captures_full_layer_trace(
        self,
    ) -> bool:
        return (
            self.layer_trace_mode
            == LAYER_TRACE_FULL
        )

    def expected_retained_indices(
        self,
        *,
        num_layers: int,
    ) -> tuple[int, ...]:
        return (
            self.retention_policy
            .expected_indices(
                num_layers=num_layers
            )
        )

    def expected_retained_count(
        self,
        *,
        num_layers: int,
    ) -> int:
        return (
            self.retention_policy
            .expected_count(
                num_layers=num_layers
            )
        )

    def should_retain(
        self,
        *,
        layer_index: int,
        num_layers: int,
    ) -> bool:
        return (
            self.retention_policy
            .should_retain(
                layer_index=layer_index,
                num_layers=num_layers,
            )
        )

    def decision_for_depth(
        self,
        *,
        layer_index: int,
        num_layers: int,
    ) -> StackTraceDecision:
        return StackTraceDecision(
            layer_index=layer_index,
            num_layers=num_layers,
            retain_public_output=(
                self.should_retain(
                    layer_index=layer_index,
                    num_layers=num_layers,
                )
            ),
            is_final_layer=(
                layer_index
                == num_layers - 1
            ),
            layer_trace_policy=(
                self.layer_trace_policy
            ),
            audit_mode=(
                self.audit_mode
            ),
        )

    def decisions(
        self,
        *,
        num_layers: int,
    ) -> tuple[
        StackTraceDecision,
        ...,
    ]:
        _require_positive_int(
            "num_layers",
            num_layers,
        )

        return tuple(
            self.decision_for_depth(
                layer_index=layer_index,
                num_layers=num_layers,
            )
            for layer_index in range(
                num_layers
            )
        )

    def validate_decisions(
        self,
        values: Sequence[
            StackTraceDecision
        ],
        *,
        num_layers: int,
    ) -> None:
        validate_stack_trace_decisions(
            values,
            policy=self,
            num_layers=num_layers,
        )

    def execution_contract_dict(
        self,
    ) -> dict[str, Any]:
        return {
            "schema_version": (
                self.schema_version
            ),
            "retention_policy": (
                self
                .retention_policy
                .execution_contract_dict()
            ),
            "layer_trace_policy": (
                self
                .layer_trace_policy
                .architecture_dict()
            ),
            "audit_mode": (
                self.audit_mode
            ),
            "retention_affects_numerical_results": (
                STACK_RETENTION_AFFECTS_NUMERICAL_RESULTS
            ),
            "layer_trace_affects_numerical_results": (
                STACK_LAYER_TRACE_AFFECTS_NUMERICAL_RESULTS
            ),
            "audit_mode_affects_numerical_results": (
                STACK_AUDIT_MODE_AFFECTS_NUMERICAL_RESULTS
            ),
        }

    def execution_contract_fingerprint(
        self,
    ) -> str:
        return _fingerprint(
            self.execution_contract_dict()
        )

    def numerical_equivalence_dict(
        self,
    ) -> dict[str, Any]:
        """
        Return the policy-independent numerical statement.

        The dictionary intentionally excludes retention, trace detail, and
        audit mode because none of them may change stack numerics.
        """

        return {
            "retention_affects_numerical_results": (
                STACK_RETENTION_AFFECTS_NUMERICAL_RESULTS
            ),
            "layer_trace_affects_numerical_results": (
                STACK_LAYER_TRACE_AFFECTS_NUMERICAL_RESULTS
            ),
            "audit_mode_affects_numerical_results": (
                STACK_AUDIT_MODE_AFFECTS_NUMERICAL_RESULTS
            ),
        }

    def numerical_equivalence_fingerprint(
        self,
    ) -> str:
        return _fingerprint(
            self.numerical_equivalence_dict()
        )


# =============================================================================
# Combined-policy resolution and construction
# =============================================================================


def resolve_stack_trace_policy(
    value: StackTracePolicy | None = None,
    *,
    retention_policy: (
        StackRetentionPolicy | str | None
    ) = None,
    layer_trace_policy: (
        LayerTracePolicy | str | None
    ) = None,
    audit_mode: bool | None = None,
) -> StackTracePolicy:
    """
    Resolve one combined trace policy without ambiguous overrides.

    Passing an existing ``StackTracePolicy`` together with a conflicting
    explicit override raises ``ValueError``.
    """

    if value is None:
        resolved_retention: (
            StackRetentionPolicy | str
        ) = (
            STACK_RETENTION_NONE
            if retention_policy is None
            else retention_policy
        )
        resolved_layer_trace: (
            LayerTracePolicy | str
        ) = (
            LAYER_TRACE_NONE
            if layer_trace_policy is None
            else layer_trace_policy
        )
        resolved_audit = (
            False
            if audit_mode is None
            else audit_mode
        )

        return StackTracePolicy(
            retention_policy=(
                resolved_retention
            ),
            layer_trace_policy=(
                resolved_layer_trace
            ),
            audit_mode=(
                resolved_audit
            ),
        )

    if not isinstance(
        value,
        StackTracePolicy,
    ):
        raise TypeError(
            "value must be a StackTracePolicy or None."
        )

    if retention_policy is not None:
        requested_retention = (
            resolve_stack_retention_policy(
                retention_policy
            )
        )

        if requested_retention != (
            value.retention_policy
        ):
            raise ValueError(
                "retention_policy conflicts with the supplied "
                "StackTracePolicy."
            )

    if layer_trace_policy is not None:
        requested_layer_trace = (
            _resolve_layer_trace_policy(
                layer_trace_policy
            )
        )

        if requested_layer_trace != (
            value.layer_trace_policy
        ):
            raise ValueError(
                "layer_trace_policy conflicts with the supplied "
                "StackTracePolicy."
            )

    if audit_mode is not None:
        _require_boolean(
            "audit_mode",
            audit_mode,
        )

        if audit_mode != value.audit_mode:
            raise ValueError(
                "audit_mode conflicts with the supplied StackTracePolicy."
            )

    return value


def build_stack_trace_policy(
    *,
    retention_policy: (
        StackRetentionPolicy | str
    ) = STACK_RETENTION_NONE,
    layer_trace_policy: (
        LayerTracePolicy | str
    ) = LAYER_TRACE_NONE,
    audit_mode: bool = False,
) -> StackTracePolicy:
    return StackTracePolicy(
        retention_policy=(
            retention_policy
        ),
        layer_trace_policy=(
            layer_trace_policy
        ),
        audit_mode=audit_mode,
    )


def build_stack_trace_decisions(
    *,
    num_layers: int,
    policy: StackTracePolicy | None = None,
    retention_policy: (
        StackRetentionPolicy | str | None
    ) = None,
    layer_trace_policy: (
        LayerTracePolicy | str | None
    ) = None,
    audit_mode: bool | None = None,
) -> tuple[
    StackTraceDecision,
    ...,
]:
    resolved = resolve_stack_trace_policy(
        policy,
        retention_policy=(
            retention_policy
        ),
        layer_trace_policy=(
            layer_trace_policy
        ),
        audit_mode=(
            audit_mode
        ),
    )

    return resolved.decisions(
        num_layers=num_layers
    )


# =============================================================================
# Decision validation
# =============================================================================


def validate_stack_trace_decisions(
    values: Sequence[
        StackTraceDecision
    ],
    *,
    policy: StackTracePolicy,
    num_layers: int,
) -> None:
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
            "values must be a sequence of StackTraceDecision objects."
        )

    if not isinstance(
        policy,
        StackTracePolicy,
    ):
        raise TypeError(
            "policy must be a StackTracePolicy."
        )

    _require_positive_int(
        "num_layers",
        num_layers,
    )

    resolved = tuple(values)

    if len(resolved) != num_layers:
        raise ValueError(
            "Trace decisions must contain exactly one decision per layer."
        )

    observed_retained: list[int] = []

    for expected_index, decision in enumerate(
        resolved
    ):
        if not isinstance(
            decision,
            StackTraceDecision,
        ):
            raise TypeError(
                f"values[{expected_index}] must be a StackTraceDecision."
            )

        if decision.layer_index != (
            expected_index
        ):
            raise ValueError(
                "Trace decisions must use contiguous zero-based layer "
                "indices."
            )

        if decision.num_layers != (
            num_layers
        ):
            raise ValueError(
                "Every trace decision must carry the same num_layers."
            )

        if (
            decision.layer_trace_policy
            != policy.layer_trace_policy
        ):
            raise ValueError(
                "Trace decision layer policy differs from the combined "
                "StackTracePolicy."
            )

        if decision.audit_mode != (
            policy.audit_mode
        ):
            raise ValueError(
                "Trace decision audit_mode differs from the combined "
                "StackTracePolicy."
            )

        expected_retain = (
            policy.should_retain(
                layer_index=expected_index,
                num_layers=num_layers,
            )
        )

        if decision.retain_public_output != (
            expected_retain
        ):
            raise ValueError(
                "Trace decision output-retention flag differs from policy."
            )

        if decision.retain_public_output:
            observed_retained.append(
                expected_index
            )

    expected_retained = (
        policy.expected_retained_indices(
            num_layers=num_layers
        )
    )

    if tuple(observed_retained) != (
        expected_retained
    ):
        raise ValueError(
            "Trace decisions retain the wrong layer indices."
        )


def validate_retained_layer_indices(
    values: Sequence[int],
    *,
    retention_policy: (
        StackRetentionPolicy | str
    ),
    num_layers: int,
) -> None:
    policy = (
        resolve_stack_retention_policy(
            retention_policy
        )
    )
    policy.validate_retained_indices(
        values,
        num_layers=num_layers,
    )


# =============================================================================
# Policy compatibility and equivalence
# =============================================================================


def stack_trace_policies_are_numerically_equivalent(
    left: StackTracePolicy,
    right: StackTracePolicy,
) -> bool:
    """
    Return whether two valid trace policies are required to preserve numerics.

    Every bounded V2.0 trace-policy pair is numerically equivalent because
    retention, layer trace detail, and audit mode are observational settings.
    """

    if not isinstance(
        left,
        StackTracePolicy,
    ):
        raise TypeError(
            "left must be a StackTracePolicy."
        )

    if not isinstance(
        right,
        StackTracePolicy,
    ):
        raise TypeError(
            "right must be a StackTracePolicy."
        )

    return (
        left.numerical_equivalence_fingerprint()
        == right.numerical_equivalence_fingerprint()
    )


def assert_stack_trace_policy_matches(
    policy: StackTracePolicy,
    *,
    retention_policy: (
        StackRetentionPolicy | str
    ),
    layer_trace_policy: (
        LayerTracePolicy | str
    ),
    audit_mode: bool,
) -> None:
    if not isinstance(
        policy,
        StackTracePolicy,
    ):
        raise TypeError(
            "policy must be a StackTracePolicy."
        )

    expected = StackTracePolicy(
        retention_policy=(
            retention_policy
        ),
        layer_trace_policy=(
            layer_trace_policy
        ),
        audit_mode=audit_mode,
    )

    if policy != expected:
        raise ValueError(
            "StackTracePolicy differs from the expected runtime contract."
        )


# =============================================================================
# Compact aliases
# =============================================================================


RetentionPolicy = StackRetentionPolicy
TracePolicy = StackTracePolicy
TraceDecision = StackTraceDecision

normalize_retention_policy = (
    normalize_stack_retention_policy
)
resolve_retention_policy = (
    resolve_stack_retention_policy
)
resolve_trace_policy = (
    resolve_stack_trace_policy
)
build_trace_policy = (
    build_stack_trace_policy
)
build_trace_decisions = (
    build_stack_trace_decisions
)
validate_trace_decisions = (
    validate_stack_trace_decisions
)


__all__ = (
    # Public identity.
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
    # Retention-policy normalization.
    "normalize_stack_retention_policy",
    "normalize_retention_policy",
    "assert_stack_retention_policy_implemented",
    "resolve_stack_retention_policy",
    "resolve_retention_policy",
    "is_no_stack_retention",
    "is_final_layer_stack_retention",
    "is_all_layers_stack_retention",
    # Immutable policy contracts.
    "StackRetentionPolicy",
    "RetentionPolicy",
    "StackTraceDecision",
    "TraceDecision",
    "StackTracePolicy",
    "TracePolicy",
    # Combined construction and resolution.
    "resolve_stack_trace_policy",
    "resolve_trace_policy",
    "build_stack_trace_policy",
    "build_trace_policy",
    "build_stack_trace_decisions",
    "build_trace_decisions",
    # Validation.
    "validate_stack_trace_decisions",
    "validate_trace_decisions",
    "validate_retained_layer_indices",
    "assert_stack_trace_policy_matches",
    # Equivalence.
    "stack_trace_policies_are_numerically_equivalent",
)
