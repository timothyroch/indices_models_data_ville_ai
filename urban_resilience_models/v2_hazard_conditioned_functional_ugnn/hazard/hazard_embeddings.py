"""
Hazard embedding vocabulary and lookup layers for the V2 functional UGNN.

This module depends on ``hazard_registry.py`` for every semantic hazard
identity. It owns only dense neural lookup indices, embedding parameters,
fixed embedding artifacts, and graph-to-node alignment.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field, fields
from enum import StrEnum
from hashlib import sha256
import json
import math
from types import MappingProxyType
from typing import Any, Final, Iterator, Mapping, Sequence, TYPE_CHECKING

import torch
from torch import nn
from torch.nn import functional as F

from .hazard_registry import (
    CANONICAL_HAZARD_DISPLAY_NAMES,
    CANONICAL_HAZARD_STABLE_IDS,
    DEFAULT_HAZARD_REGISTRY,
    DEFAULT_HAZARD_REGISTRY_IDENTITY,
    HAZARD_ID_ALL_HAZARD,
    HAZARD_ID_CIVIL_SECURITY_EVENT,
    HAZARD_ID_FLOOD,
    HAZARD_ID_FREEZING_RAIN,
    HAZARD_ID_HEAT,
    HAZARD_ID_OUTAGE,
    HAZARD_ID_PLUVIAL_FLOOD,
    HAZARD_ID_RIVERINE_FLOOD,
    HAZARD_ID_ROAD_DISRUPTION,
    HAZARD_ID_SNOWSTORM,
    HAZARD_ID_UNKNOWN,
    HAZARD_ID_WINTER_STORM,
    UNKNOWN_HAZARD_DISPLAY_NAME,
    UNKNOWN_HAZARD_NAME,
    HazardKind,
    HazardRegistry,
    HazardRegistryIdentity,
    HazardSupportState,
)

if TYPE_CHECKING:
    from ..config import HazardEmbeddingConfig


# =============================================================================
# Schema and vocabulary identity
# =============================================================================


HAZARD_EMBEDDING_VOCABULARY_SCHEMA_VERSION: Final[str] = "0.2"
FIXED_HAZARD_EMBEDDING_ARTIFACT_SCHEMA_VERSION: Final[str] = "0.2"
HAZARD_EMBEDDING_ARCHITECTURE_SCHEMA_VERSION: Final[str] = "0.2"

DEFAULT_HAZARD_EMBEDDING_VOCABULARY_NAME: Final[str] = (
    "v2_runtime_hazard_embedding_vocabulary"
)


# =============================================================================
# Runtime vocabulary
# =============================================================================


class UnknownHazardPolicy(StrEnum):
    """Policy for names or stable IDs absent from the active vocabulary."""

    ERROR = "error"
    USE_UNKNOWN_EMBEDDING = "use_unknown_embedding"


class HazardEmbeddingMode(StrEnum):
    """Source of the effective hazard embedding."""

    LEARNED = "learned"
    FIXED = "fixed"
    FIXED_PLUS_RESIDUAL = "fixed_plus_residual"

    # Compatibility alias for the pre-config-refactor spelling.
    FIXED_PLUS_LEARNED_RESIDUAL = "fixed_plus_residual"


class HazardEmbeddingInitialization(StrEnum):
    """Initialization policy for learned embedding parameters."""

    NORMAL = "normal"
    ZERO = "zero"
    XAVIER_UNIFORM = "xavier_uniform"


UNKNOWN_EMBEDDING_POLICY_ERROR: Final[str] = "error"
UNKNOWN_EMBEDDING_POLICY_ZERO_FIXED: Final[str] = "zero_fixed"
UNKNOWN_EMBEDDING_POLICY_LEARNED: Final[str] = "learned"


# =============================================================================
# Generic helpers
# =============================================================================


def _require_nonempty_string(name: str, value: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string.")


def _require_boolean(name: str, value: bool) -> None:
    if not isinstance(value, bool):
        raise TypeError(f"{name} must be a Boolean.")


def _require_positive_int(name: str, value: int) -> None:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer.")
    if value <= 0:
        raise ValueError(f"{name} must be strictly positive.")


def _require_nonnegative_int(name: str, value: int) -> None:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer.")
    if value < 0:
        raise ValueError(f"{name} must be nonnegative.")


def _require_finite_number(name: str, value: int | float) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be numeric.")
    converted = float(value)
    if not math.isfinite(converted):
        raise ValueError(f"{name} must be finite.")
    return converted


def _require_nonnegative_float(name: str, value: int | float) -> float:
    converted = _require_finite_number(name, value)
    if converted < 0.0:
        raise ValueError(f"{name} must be nonnegative.")
    return converted


def _require_mapping(name: str, value: Any) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must be a mapping.")
    return value


def _reject_unknown_fields(
    object_type: type[Any],
    payload: Mapping[str, Any],
) -> None:
    allowed = {
        definition.name
        for definition in fields(object_type)
        if definition.init
    }
    unknown = sorted(set(payload) - allowed)
    if unknown:
        raise ValueError(
            f"Unknown fields for {object_type.__name__}: {unknown}."
        )


def _as_tuple(name: str, value: Any) -> tuple[Any, ...]:
    if isinstance(value, tuple):
        return value
    if isinstance(value, list):
        return tuple(value)
    raise TypeError(f"{name} must be a list or tuple.")


def _require_unique_strings(name: str, values: Sequence[str]) -> None:
    for index, value in enumerate(values):
        _require_nonempty_string(f"{name}[{index}]", value)
    duplicates = sorted(
        value
        for value, count in Counter(values).items()
        if count > 1
    )
    if duplicates:
        raise ValueError(f"{name} contains duplicate values: {duplicates}.")


def _canonical_json(payload: Mapping[str, Any]) -> str:
    return json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )


def _fingerprint(payload: Mapping[str, Any]) -> str:
    return sha256(_canonical_json(payload).encode("utf-8")).hexdigest()


def _tensor_fingerprint(tensors: Mapping[str, torch.Tensor]) -> str:
    digest = sha256()
    for name in sorted(tensors):
        tensor = tensors[name].detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(tensor.dtype).encode("utf-8"))
        digest.update(
            json.dumps(
                list(tensor.shape),
                separators=(",", ":"),
            ).encode("utf-8")
        )
        digest.update(tensor.view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def _normalize_unknown_policy(
    value: UnknownHazardPolicy | str,
) -> UnknownHazardPolicy:
    if isinstance(value, UnknownHazardPolicy):
        return value
    return UnknownHazardPolicy(value)


def _normalize_mode(
    value: HazardEmbeddingMode | str,
) -> HazardEmbeddingMode:
    if isinstance(value, HazardEmbeddingMode):
        return value
    if value == "fixed_plus_learned_residual":
        value = HazardEmbeddingMode.FIXED_PLUS_RESIDUAL.value
    return HazardEmbeddingMode(value)


def _normalize_initialization(
    value: HazardEmbeddingInitialization | str,
) -> HazardEmbeddingInitialization:
    if isinstance(value, HazardEmbeddingInitialization):
        return value
    return HazardEmbeddingInitialization(value)


def _normalize_hazard_name(value: HazardKind | str) -> str:
    if isinstance(value, HazardKind):
        return value.value
    _require_nonempty_string("hazard name", value)
    return value


_DTYPE_BY_NAME: Final[Mapping[str, torch.dtype]] = MappingProxyType(
    {
        "torch.float16": torch.float16,
        "torch.bfloat16": torch.bfloat16,
        "torch.float32": torch.float32,
        "torch.float64": torch.float64,
    }
)


def _dtype_from_name(value: str) -> torch.dtype:
    try:
        return _DTYPE_BY_NAME[value]
    except KeyError as exc:
        raise ValueError(f"Unsupported serialized tensor dtype {value!r}.") from exc


def _seeded_generator(
    *,
    device: torch.device,
    seed: int,
) -> torch.Generator:
    """Create a generator local to the parameter device and RNG stream."""

    generator = torch.Generator(device=device)
    generator.manual_seed(seed)
    return generator


# =============================================================================
# Vocabulary entries
# =============================================================================


@dataclass(slots=True, frozen=True)
class HazardVocabularyEntry:
    """One stable hazard identity exposed to the embedding table."""

    stable_hazard_id: int
    name: str
    display_name: str
    runtime_allowed: bool
    fallback_only: bool
    support_state: HazardSupportState | None
    is_unknown: bool = False

    def __post_init__(self) -> None:
        _require_nonnegative_int("stable_hazard_id", self.stable_hazard_id)
        _require_nonempty_string("hazard name", self.name)
        _require_nonempty_string("hazard display_name", self.display_name)
        _require_boolean("runtime_allowed", self.runtime_allowed)
        _require_boolean("fallback_only", self.fallback_only)
        _require_boolean("is_unknown", self.is_unknown)

        if self.support_state is not None and not isinstance(
            self.support_state,
            HazardSupportState,
        ):
            raise TypeError(
                "support_state must be absent or a HazardSupportState."
            )

        if self.is_unknown:
            if self.name != UNKNOWN_HAZARD_NAME:
                raise ValueError(
                    "Unknown entries must use UNKNOWN_HAZARD_NAME."
                )
            if self.stable_hazard_id != HAZARD_ID_UNKNOWN:
                raise ValueError(
                    "Unknown entries must use HAZARD_ID_UNKNOWN."
                )
            if (
                self.runtime_allowed
                or self.fallback_only
                or self.support_state is not None
            ):
                raise ValueError(
                    "Unknown entries cannot claim queryability, fallback "
                    "semantics, or data support."
                )
        elif self.support_state is None:
            raise ValueError(
                "Canonical hazard entries require a support state."
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "stable_hazard_id": self.stable_hazard_id,
            "name": self.name,
            "display_name": self.display_name,
            "runtime_allowed": self.runtime_allowed,
            "fallback_only": self.fallback_only,
            "support_state": (
                self.support_state.value
                if self.support_state is not None
                else None
            ),
            "is_unknown": self.is_unknown,
        }

    @classmethod
    def from_dict(
        cls,
        payload: Mapping[str, Any],
    ) -> "HazardVocabularyEntry":
        mapping = dict(_require_mapping("HazardVocabularyEntry", payload))
        _reject_unknown_fields(cls, mapping)
        if mapping.get("support_state") is not None:
            mapping["support_state"] = HazardSupportState(
                mapping["support_state"]
            )
        return cls(**mapping)


# =============================================================================
# Immutable dense vocabulary
# =============================================================================


@dataclass(slots=True, frozen=True)
class HazardEmbeddingVocabulary:
    """Immutable stable-hazard to dense-row mapping."""

    entries: tuple[HazardVocabularyEntry, ...]
    source_hazard_registry_name: str
    source_hazard_registry_version: str
    source_hazard_registry_fingerprint: str
    vocabulary_name: str = DEFAULT_HAZARD_EMBEDDING_VOCABULARY_NAME
    schema_version: str = HAZARD_EMBEDDING_VOCABULARY_SCHEMA_VERSION

    _by_name: Mapping[str, HazardVocabularyEntry] = field(
        init=False,
        repr=False,
        compare=False,
    )
    _by_stable_id: Mapping[int, HazardVocabularyEntry] = field(
        init=False,
        repr=False,
        compare=False,
    )
    _dense_index_by_name: Mapping[str, int] = field(
        init=False,
        repr=False,
        compare=False,
    )
    _dense_index_by_stable_id: Mapping[int, int] = field(
        init=False,
        repr=False,
        compare=False,
    )

    def __post_init__(self) -> None:
        if not self.entries:
            raise ValueError(
                "A hazard embedding vocabulary cannot be empty."
            )
        entries = tuple(self.entries)
        for index, entry in enumerate(entries):
            if not isinstance(entry, HazardVocabularyEntry):
                raise TypeError(
                    f"entries[{index}] must be a HazardVocabularyEntry."
                )
        entries = tuple(
            sorted(entries, key=lambda entry: entry.stable_hazard_id)
        )
        object.__setattr__(self, "entries", entries)

        for name, value in (
            ("source_hazard_registry_name", self.source_hazard_registry_name),
            (
                "source_hazard_registry_version",
                self.source_hazard_registry_version,
            ),
            (
                "source_hazard_registry_fingerprint",
                self.source_hazard_registry_fingerprint,
            ),
            ("vocabulary_name", self.vocabulary_name),
            ("schema_version", self.schema_version),
        ):
            _require_nonempty_string(name, value)

        names = tuple(entry.name for entry in entries)
        stable_ids = tuple(entry.stable_hazard_id for entry in entries)
        _require_unique_strings("hazard vocabulary names", names)
        if len(set(stable_ids)) != len(stable_ids):
            raise ValueError(
                "Hazard vocabulary contains duplicate stable IDs."
            )
        if sum(entry.is_unknown for entry in entries) > 1:
            raise ValueError(
                "A hazard vocabulary may contain at most one unknown row."
            )

        by_name = {entry.name: entry for entry in entries}
        by_stable_id = {
            entry.stable_hazard_id: entry for entry in entries
        }
        dense_by_name = {
            entry.name: index for index, entry in enumerate(entries)
        }
        dense_by_stable = {
            entry.stable_hazard_id: index
            for index, entry in enumerate(entries)
        }
        object.__setattr__(self, "_by_name", MappingProxyType(by_name))
        object.__setattr__(
            self,
            "_by_stable_id",
            MappingProxyType(by_stable_id),
        )
        object.__setattr__(
            self,
            "_dense_index_by_name",
            MappingProxyType(dense_by_name),
        )
        object.__setattr__(
            self,
            "_dense_index_by_stable_id",
            MappingProxyType(dense_by_stable),
        )

    def __len__(self) -> int:
        return len(self.entries)

    def __iter__(self) -> Iterator[HazardVocabularyEntry]:
        return iter(self.entries)

    def __contains__(self, value: object) -> bool:
        if isinstance(value, str):
            return value in self._by_name
        if isinstance(value, int) and not isinstance(value, bool):
            return value in self._by_stable_id
        return False

    @property
    def hazard_names(self) -> tuple[str, ...]:
        return tuple(entry.name for entry in self.entries)

    @property
    def stable_hazard_ids(self) -> tuple[int, ...]:
        return tuple(entry.stable_hazard_id for entry in self.entries)

    @property
    def runtime_hazard_names(self) -> tuple[str, ...]:
        return tuple(
            entry.name
            for entry in self.entries
            if entry.runtime_allowed
        )

    @property
    def fallback_hazard_names(self) -> tuple[str, ...]:
        return tuple(
            entry.name
            for entry in self.entries
            if entry.fallback_only
        )

    @property
    def includes_unknown(self) -> bool:
        return UNKNOWN_HAZARD_NAME in self._by_name

    @property
    def includes_fallback_hazard(self) -> bool:
        return any(entry.fallback_only for entry in self.entries)

    @property
    def unknown_index(self) -> int | None:
        return self._dense_index_by_name.get(UNKNOWN_HAZARD_NAME)

    @property
    def dense_index_by_name(self) -> Mapping[str, int]:
        return self._dense_index_by_name

    @property
    def dense_index_by_stable_id(self) -> Mapping[int, int]:
        return self._dense_index_by_stable_id

    def entry_for_name(self, hazard_name: str) -> HazardVocabularyEntry:
        try:
            return self._by_name[hazard_name]
        except KeyError as exc:
            raise KeyError(
                f"Hazard {hazard_name!r} is absent from the active "
                "embedding vocabulary."
            ) from exc

    def entry_for_stable_id(
        self,
        stable_hazard_id: int,
    ) -> HazardVocabularyEntry:
        try:
            return self._by_stable_id[stable_hazard_id]
        except KeyError as exc:
            raise KeyError(
                f"Stable hazard ID {stable_hazard_id} is absent from "
                "the active embedding vocabulary."
            ) from exc

    def entry_for_index(self, hazard_index: int) -> HazardVocabularyEntry:
        if isinstance(hazard_index, bool) or not isinstance(hazard_index, int):
            raise TypeError("hazard_index must be an integer.")
        if not 0 <= hazard_index < len(self):
            raise IndexError(
                f"hazard_index {hazard_index} is outside "
                f"[0, {len(self) - 1}]."
            )
        return self.entries[hazard_index]

    def index_for_name(self, hazard_name: str) -> int:
        try:
            return self._dense_index_by_name[hazard_name]
        except KeyError as exc:
            raise KeyError(
                f"Hazard {hazard_name!r} is absent from the active "
                "embedding vocabulary."
            ) from exc

    def index_for_stable_id(self, stable_hazard_id: int) -> int:
        try:
            return self._dense_index_by_stable_id[stable_hazard_id]
        except KeyError as exc:
            raise KeyError(
                f"Stable hazard ID {stable_hazard_id} is absent from "
                "the active embedding vocabulary."
            ) from exc

    def encode_names(
        self,
        hazard_names: Sequence[HazardKind | str],
        *,
        unknown_policy: UnknownHazardPolicy | str,
        device: torch.device | str | None = None,
    ) -> "HazardIndexBatch":
        policy = _normalize_unknown_policy(unknown_policy)
        if not hazard_names:
            raise ValueError("At least one hazard name is required.")

        dense_indices: list[int] = []
        stable_ids: list[int] = []
        normalized_names: list[str] = []
        unknown_mask: list[bool] = []
        rejected: list[str] = []

        for value in hazard_names:
            name = _normalize_hazard_name(value)
            entry = self._by_name.get(name)

            explicit_unknown = (
                entry is not None and entry.is_unknown
            ) or name == UNKNOWN_HAZARD_NAME

            if entry is None or explicit_unknown:
                if policy == UnknownHazardPolicy.ERROR:
                    rejected.append(name)
                    continue
                if self.unknown_index is None:
                    raise ValueError(
                        "Unknown-hazard fallback was requested, but the "
                        "vocabulary has no unknown embedding row."
                    )
                entry = self.entry_for_name(UNKNOWN_HAZARD_NAME)
                unknown = True
            else:
                unknown = False

            dense_indices.append(self.index_for_name(entry.name))
            stable_ids.append(entry.stable_hazard_id)
            normalized_names.append(entry.name)
            unknown_mask.append(unknown)

        if rejected:
            raise ValueError(
                f"Unknown hazard names: {sorted(set(rejected))}."
            )

        return HazardIndexBatch(
            dense_indices=torch.tensor(
                dense_indices,
                dtype=torch.long,
                device=device,
            ),
            stable_hazard_ids=torch.tensor(
                stable_ids,
                dtype=torch.long,
                device=device,
            ),
            hazard_names=tuple(normalized_names),
            unknown_mask=torch.tensor(
                unknown_mask,
                dtype=torch.bool,
                device=device,
            ),
            vocabulary_fingerprint=self.fingerprint(),
        )

    def encode_stable_ids(
        self,
        stable_hazard_ids: Sequence[int],
        *,
        unknown_policy: UnknownHazardPolicy | str,
        device: torch.device | str | None = None,
    ) -> "HazardIndexBatch":
        policy = _normalize_unknown_policy(unknown_policy)
        if not stable_hazard_ids:
            raise ValueError("At least one stable hazard ID is required.")

        names: list[str] = []
        rejected: list[int] = []
        for stable_id in stable_hazard_ids:
            if isinstance(stable_id, bool) or not isinstance(stable_id, int):
                raise TypeError("Stable hazard IDs must be integers.")
            entry = self._by_stable_id.get(stable_id)
            explicit_unknown = (
                entry is not None and entry.is_unknown
            ) or stable_id == HAZARD_ID_UNKNOWN
            if entry is None or explicit_unknown:
                if policy == UnknownHazardPolicy.ERROR:
                    rejected.append(stable_id)
                    continue
                names.append(UNKNOWN_HAZARD_NAME)
            else:
                names.append(entry.name)

        if rejected:
            raise ValueError(
                f"Unknown stable hazard IDs: {sorted(set(rejected))}."
            )
        return self.encode_names(
            names,
            unknown_policy=policy,
            device=device,
        )

    def decode_indices(
        self,
        dense_indices: Sequence[int],
    ) -> tuple[str, ...]:
        return tuple(
            self.entry_for_index(index).name for index in dense_indices
        )

    def assert_matches_hazard_registry(
        self,
        hazard_registry: HazardRegistry,
    ) -> None:
        if not isinstance(hazard_registry, HazardRegistry):
            raise TypeError(
                "hazard_registry must be a HazardRegistry."
            )
        if hazard_registry.registry_name != self.source_hazard_registry_name:
            raise ValueError(
                "Embedding vocabulary references a different hazard-"
                "registry name."
            )
        if (
            hazard_registry.registry_version
            != self.source_hazard_registry_version
        ):
            raise ValueError(
                "Embedding vocabulary references a different hazard-"
                "registry version."
            )
        if (
            hazard_registry.compatibility_fingerprint()
            != self.source_hazard_registry_fingerprint
        ):
            raise ValueError(
                "Embedding vocabulary references a different compatible "
                "hazard registry."
            )

        for entry in self.entries:
            if entry.is_unknown:
                continue
            registry_entry = hazard_registry.get_by_name(entry.name)
            if entry.stable_hazard_id != registry_entry.stable_hazard_id:
                raise ValueError(
                    f"Hazard {entry.name!r} stable identity differs from "
                    "the source registry."
                )
            if entry.display_name != registry_entry.display_name:
                raise ValueError(
                    f"Hazard {entry.name!r} display name differs from "
                    "the source registry."
                )
            if entry.runtime_allowed != registry_entry.query_allowed:
                raise ValueError(
                    f"Hazard {entry.name!r} queryability differs from "
                    "the source registry."
                )
            if entry.fallback_only != registry_entry.fallback_only:
                raise ValueError(
                    f"Hazard {entry.name!r} fallback semantics differ from "
                    "the source registry."
                )
            if entry.support_state != registry_entry.support_state:
                raise ValueError(
                    f"Hazard {entry.name!r} support status differs from "
                    "the source registry."
                )

    def semantic_dict(self) -> dict[str, Any]:
        return {
            "vocabulary_name": self.vocabulary_name,
            "schema_version": self.schema_version,
            "source_hazard_registry_name": self.source_hazard_registry_name,
            "source_hazard_registry_version": (
                self.source_hazard_registry_version
            ),
            "source_hazard_registry_fingerprint": (
                self.source_hazard_registry_fingerprint
            ),
            "entries": [
                {"hazard_index": index, **entry.to_dict()}
                for index, entry in enumerate(self.entries)
            ],
        }

    def fingerprint(self) -> str:
        return _fingerprint(self.semantic_dict())

    def to_dict(self) -> dict[str, Any]:
        return {**self.semantic_dict(), "fingerprint": self.fingerprint()}

    @classmethod
    def from_dict(
        cls,
        payload: Mapping[str, Any],
        *,
        hazard_registry: HazardRegistry | None = None,
    ) -> "HazardEmbeddingVocabulary":
        mapping = dict(
            _require_mapping("HazardEmbeddingVocabulary", payload)
        )
        serialized_fingerprint = mapping.pop("fingerprint", None)
        raw_entries = mapping.pop("entries", None)
        allowed = {
            "vocabulary_name",
            "schema_version",
            "source_hazard_registry_name",
            "source_hazard_registry_version",
            "source_hazard_registry_fingerprint",
        }
        unknown = sorted(set(mapping) - allowed)
        if unknown:
            raise ValueError(
                "Unknown HazardEmbeddingVocabulary fields: "
                f"{unknown}."
            )
        if not isinstance(raw_entries, list):
            raise TypeError(
                "HazardEmbeddingVocabulary.entries must be a list."
            )

        entries: list[HazardVocabularyEntry] = []
        for expected_index, raw_entry in enumerate(raw_entries):
            entry_mapping = dict(
                _require_mapping(
                    f"entries[{expected_index}]",
                    raw_entry,
                )
            )
            observed_index = entry_mapping.pop("hazard_index", None)
            if observed_index != expected_index:
                raise ValueError(
                    "Serialized hazard indices must be contiguous and "
                    "ordered from zero."
                )
            entries.append(HazardVocabularyEntry.from_dict(entry_mapping))

        vocabulary = cls(entries=tuple(entries), **mapping)
        if (
            serialized_fingerprint is not None
            and serialized_fingerprint != vocabulary.fingerprint()
        ):
            raise ValueError(
                "Serialized hazard-vocabulary fingerprint does not match "
                "the reconstructed vocabulary."
            )
        if hazard_registry is not None:
            vocabulary.assert_matches_hazard_registry(hazard_registry)
        return vocabulary


# =============================================================================
# Encoded hazard-index batch
# =============================================================================


@dataclass(slots=True, frozen=True)
class HazardIndexBatch:
    """Dense hazard indices with preserved semantic identities."""

    dense_indices: torch.Tensor
    stable_hazard_ids: torch.Tensor
    hazard_names: tuple[str, ...]
    unknown_mask: torch.Tensor
    vocabulary_fingerprint: str

    def __post_init__(self) -> None:
        for name, tensor in (
            ("dense_indices", self.dense_indices),
            ("stable_hazard_ids", self.stable_hazard_ids),
            ("unknown_mask", self.unknown_mask),
        ):
            if not isinstance(tensor, torch.Tensor):
                raise TypeError(f"{name} must be a tensor.")
            if tensor.ndim != 1:
                raise ValueError(f"{name} must be one-dimensional.")
        if self.dense_indices.dtype != torch.long:
            raise ValueError("dense_indices must use torch.long.")
        if self.stable_hazard_ids.dtype != torch.long:
            raise ValueError("stable_hazard_ids must use torch.long.")
        if self.unknown_mask.dtype != torch.bool:
            raise ValueError("unknown_mask must use torch.bool.")

        length = int(self.dense_indices.shape[0])
        if (
            int(self.stable_hazard_ids.shape[0]) != length
            or int(self.unknown_mask.shape[0]) != length
            or len(self.hazard_names) != length
        ):
            raise ValueError(
                "HazardIndexBatch fields must have equal length."
            )
        if (
            self.dense_indices.device != self.stable_hazard_ids.device
            or self.dense_indices.device != self.unknown_mask.device
        ):
            raise ValueError(
                "HazardIndexBatch tensors must share one device."
            )
        _require_nonempty_string(
            "vocabulary_fingerprint",
            self.vocabulary_fingerprint,
        )
        for name in self.hazard_names:
            _require_nonempty_string("encoded hazard name", name)

    def __len__(self) -> int:
        return int(self.dense_indices.shape[0])

    @property
    def device(self) -> torch.device:
        return self.dense_indices.device

    def to(self, device: torch.device | str) -> "HazardIndexBatch":
        return HazardIndexBatch(
            dense_indices=self.dense_indices.to(device=device),
            stable_hazard_ids=self.stable_hazard_ids.to(device=device),
            hazard_names=self.hazard_names,
            unknown_mask=self.unknown_mask.to(device=device),
            vocabulary_fingerprint=self.vocabulary_fingerprint,
        )

    def assert_matches_vocabulary(
        self,
        vocabulary: HazardEmbeddingVocabulary,
    ) -> None:
        if not isinstance(vocabulary, HazardEmbeddingVocabulary):
            raise TypeError(
                "vocabulary must be a HazardEmbeddingVocabulary."
            )
        if self.vocabulary_fingerprint != vocabulary.fingerprint():
            raise ValueError(
                "Hazard indices were encoded under a different vocabulary."
            )

        for position in range(len(self)):
            dense_index = int(self.dense_indices[position].item())
            try:
                expected = vocabulary.entry_for_index(dense_index)
            except IndexError as exc:
                raise ValueError(
                    f"dense hazard index at position {position} is invalid."
                ) from exc

            observed_name = self.hazard_names[position]
            if observed_name != expected.name:
                raise ValueError(
                    "HazardIndexBatch hazard name does not match its dense "
                    f"index at position {position}."
                )

            observed_stable_id = int(
                self.stable_hazard_ids[position].item()
            )
            if observed_stable_id != expected.stable_hazard_id:
                raise ValueError(
                    "HazardIndexBatch stable hazard ID does not match its "
                    f"dense index at position {position}."
                )

            observed_unknown = bool(self.unknown_mask[position].item())
            if observed_unknown != expected.is_unknown:
                raise ValueError(
                    "HazardIndexBatch unknown mask does not match its dense "
                    f"index at position {position}."
                )


# =============================================================================
# Fixed embedding artifacts
# =============================================================================


@dataclass(slots=True, frozen=True)
class FixedHazardEmbeddingArtifact:
    """Auditable fixed hazard embedding table."""

    embedding_matrix: torch.Tensor
    hazard_names: tuple[str, ...]
    stable_hazard_ids: tuple[int, ...]
    vocabulary_fingerprint: str
    source_id: str
    source_version: str
    source_fingerprint: str
    architecture_lineage_fingerprint: str | None = None
    schema_version: str = FIXED_HAZARD_EMBEDDING_ARTIFACT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if not isinstance(self.embedding_matrix, torch.Tensor):
            raise TypeError("embedding_matrix must be a tensor.")
        if self.embedding_matrix.ndim != 2:
            raise ValueError(
                "embedding_matrix must have shape "
                "[num_hazards, embedding_dim]."
            )
        if not self.embedding_matrix.dtype.is_floating_point:
            raise ValueError(
                "embedding_matrix must use a floating-point dtype."
            )
        normalized = (
            self.embedding_matrix.detach().cpu().contiguous().clone()
        )
        if not bool(torch.isfinite(normalized).all().item()):
            raise ValueError(
                "embedding_matrix must contain only finite values; NaN or "
                "infinity is not allowed."
            )
        object.__setattr__(self, "embedding_matrix", normalized)
        object.__setattr__(self, "hazard_names", tuple(self.hazard_names))
        object.__setattr__(
            self,
            "stable_hazard_ids",
            tuple(self.stable_hazard_ids),
        )

        rows = int(normalized.shape[0])
        if (
            len(self.hazard_names) != rows
            or len(self.stable_hazard_ids) != rows
        ):
            raise ValueError(
                "Fixed embedding identities must align with matrix rows."
            )
        _require_unique_strings("fixed hazard names", self.hazard_names)
        if len(set(self.stable_hazard_ids)) != len(self.stable_hazard_ids):
            raise ValueError("Fixed stable hazard IDs must be unique.")
        for name, value in (
            ("vocabulary_fingerprint", self.vocabulary_fingerprint),
            ("source_id", self.source_id),
            ("source_version", self.source_version),
            ("source_fingerprint", self.source_fingerprint),
            ("schema_version", self.schema_version),
        ):
            _require_nonempty_string(name, value)
        if self.architecture_lineage_fingerprint is not None:
            _require_nonempty_string(
                "architecture_lineage_fingerprint",
                self.architecture_lineage_fingerprint,
            )

    @property
    def embedding_dim(self) -> int:
        return int(self.embedding_matrix.shape[1])

    @property
    def num_embeddings(self) -> int:
        return int(self.embedding_matrix.shape[0])

    @property
    def embedding_fingerprint(self) -> str:
        return _tensor_fingerprint(
            {"embedding_matrix": self.embedding_matrix}
        )

    def assert_matches_vocabulary(
        self,
        vocabulary: HazardEmbeddingVocabulary,
    ) -> None:
        if self.vocabulary_fingerprint != vocabulary.fingerprint():
            raise ValueError(
                "Fixed embeddings were created for a different vocabulary."
            )
        if self.hazard_names != vocabulary.hazard_names:
            raise ValueError(
                "Fixed embedding hazard ordering differs from the active "
                "vocabulary."
            )
        if self.stable_hazard_ids != vocabulary.stable_hazard_ids:
            raise ValueError(
                "Fixed embedding stable IDs differ from the active "
                "vocabulary."
            )

    def metadata_dict(self) -> dict[str, Any]:
        return {
            "hazard_names": list(self.hazard_names),
            "stable_hazard_ids": list(self.stable_hazard_ids),
            "embedding_dim": self.embedding_dim,
            "num_embeddings": self.num_embeddings,
            "dtype": str(self.embedding_matrix.dtype),
            "embedding_fingerprint": self.embedding_fingerprint,
            "vocabulary_fingerprint": self.vocabulary_fingerprint,
            "source_id": self.source_id,
            "source_version": self.source_version,
            "source_fingerprint": self.source_fingerprint,
            "architecture_lineage_fingerprint": (
                self.architecture_lineage_fingerprint
            ),
            "schema_version": self.schema_version,
        }

    def to_dict(self, *, include_values: bool = True) -> dict[str, Any]:
        payload = self.metadata_dict()
        if include_values:
            payload["embedding_matrix"] = self.embedding_matrix.tolist()
        return payload

    @classmethod
    def from_dict(
        cls,
        payload: Mapping[str, Any],
        *,
        dtype: torch.dtype | None = None,
    ) -> "FixedHazardEmbeddingArtifact":
        mapping = dict(
            _require_mapping("FixedHazardEmbeddingArtifact", payload)
        )
        serialized_embedding_fingerprint = mapping.pop(
            "embedding_fingerprint",
            None,
        )
        serialized_embedding_dim = mapping.pop("embedding_dim", None)
        serialized_num_embeddings = mapping.pop("num_embeddings", None)
        serialized_dtype = mapping.pop("dtype", None)

        allowed = {
            "embedding_matrix",
            "hazard_names",
            "stable_hazard_ids",
            "vocabulary_fingerprint",
            "source_id",
            "source_version",
            "source_fingerprint",
            "architecture_lineage_fingerprint",
            "schema_version",
        }
        unknown = sorted(set(mapping) - allowed)
        if unknown:
            raise ValueError(
                "Unknown FixedHazardEmbeddingArtifact fields: "
                f"{unknown}."
            )
        if "embedding_matrix" not in mapping:
            raise ValueError(
                "Serialized fixed embeddings require embedding_matrix."
            )
        if dtype is None:
            dtype = (
                _dtype_from_name(serialized_dtype)
                if serialized_dtype is not None
                else torch.float32
            )
        elif (
            serialized_dtype is not None
            and str(dtype) != serialized_dtype
        ):
            raise ValueError(
                "Requested dtype disagrees with serialized fixed-artifact "
                "dtype."
            )

        mapping["embedding_matrix"] = torch.tensor(
            mapping["embedding_matrix"],
            dtype=dtype,
        )
        mapping["hazard_names"] = _as_tuple(
            "hazard_names",
            mapping["hazard_names"],
        )
        mapping["stable_hazard_ids"] = _as_tuple(
            "stable_hazard_ids",
            mapping["stable_hazard_ids"],
        )
        artifact = cls(**mapping)

        if (
            serialized_embedding_dim is not None
            and serialized_embedding_dim != artifact.embedding_dim
        ):
            raise ValueError(
                "Serialized embedding_dim does not match the matrix."
            )
        if (
            serialized_num_embeddings is not None
            and serialized_num_embeddings != artifact.num_embeddings
        ):
            raise ValueError(
                "Serialized num_embeddings does not match the matrix."
            )
        if (
            serialized_embedding_fingerprint is not None
            and serialized_embedding_fingerprint
            != artifact.embedding_fingerprint
        ):
            raise ValueError(
                "Serialized fixed-embedding fingerprint does not match "
                "the matrix values."
            )
        return artifact


# =============================================================================
# Lookup outputs
# =============================================================================


@dataclass(slots=True, frozen=True)
class HazardEmbeddingLookup:
    """Hazard embeddings with their aligned semantic identities."""

    embeddings: torch.Tensor
    indices: HazardIndexBatch
    embedding_mode: HazardEmbeddingMode
    architecture_fingerprint: str

    def __post_init__(self) -> None:
        if not isinstance(self.embeddings, torch.Tensor):
            raise TypeError("embeddings must be a tensor.")
        if self.embeddings.ndim != 2:
            raise ValueError(
                "embeddings must have shape [items, embedding_dim]."
            )
        if not self.embeddings.dtype.is_floating_point:
            raise ValueError(
                "embeddings must use a floating-point dtype."
            )
        if int(self.embeddings.shape[0]) != len(self.indices):
            raise ValueError(
                "Embedding rows must align with encoded hazard items."
            )
        if self.embeddings.device != self.indices.device:
            raise ValueError(
                "Embedding values and hazard-index metadata must share "
                "one device."
            )
        if not bool(torch.isfinite(self.embeddings).all().item()):
            raise ValueError(
                "Hazard embedding lookup must contain only finite values; "
                "NaN or infinity was observed."
            )
        if not isinstance(self.embedding_mode, HazardEmbeddingMode):
            raise TypeError(
                "embedding_mode must be a HazardEmbeddingMode."
            )
        _require_nonempty_string(
            "architecture_fingerprint",
            self.architecture_fingerprint,
        )

    @property
    def embedding_dim(self) -> int:
        return int(self.embeddings.shape[1])


@dataclass(slots=True, frozen=True)
class NodeAlignedHazardEmbeddingLookup:
    """Node-aligned values plus the graph-level semantic lookup."""

    node_embeddings: torch.Tensor
    graph_lookup: HazardEmbeddingLookup
    node_batch_index: torch.Tensor

    def __post_init__(self) -> None:
        if not isinstance(self.node_embeddings, torch.Tensor):
            raise TypeError("node_embeddings must be a tensor.")
        if self.node_embeddings.ndim != 2:
            raise ValueError(
                "node_embeddings must have shape [num_nodes, embedding_dim]."
            )
        if not isinstance(self.graph_lookup, HazardEmbeddingLookup):
            raise TypeError(
                "graph_lookup must be a HazardEmbeddingLookup."
            )
        if not isinstance(self.node_batch_index, torch.Tensor):
            raise TypeError("node_batch_index must be a tensor.")
        if (
            self.node_batch_index.ndim != 1
            or self.node_batch_index.dtype != torch.long
        ):
            raise ValueError(
                "node_batch_index must be a one-dimensional torch.long "
                "tensor."
            )
        if int(self.node_embeddings.shape[0]) != int(
            self.node_batch_index.shape[0]
        ):
            raise ValueError(
                "Node-aligned embeddings and node_batch_index must align."
            )
        if int(self.node_embeddings.shape[1]) != (
            self.graph_lookup.embedding_dim
        ):
            raise ValueError(
                "Node-aligned and graph-level embedding widths differ."
            )
        if (
            self.node_embeddings.device
            != self.graph_lookup.embeddings.device
            or self.node_embeddings.device
            != self.node_batch_index.device
        ):
            raise ValueError(
                "Node-aligned values and graph metadata must share one "
                "device."
            )
        if not bool(torch.isfinite(self.node_embeddings).all().item()):
            raise ValueError(
                "Node-aligned hazard embeddings must be finite."
            )


# =============================================================================
# Hazard embedding module
# =============================================================================


class HazardEmbeddingLayer(nn.Module):
    """Learned, fixed, or fixed-plus-residual hazard embedding table."""

    def __init__(
        self,
        *,
        vocabulary: HazardEmbeddingVocabulary,
        embedding_dim: int,
        mode: HazardEmbeddingMode | str = HazardEmbeddingMode.LEARNED,
        unknown_policy: UnknownHazardPolicy | str = UnknownHazardPolicy.ERROR,
        unknown_embedding_policy: str = UNKNOWN_EMBEDDING_POLICY_ERROR,
        initialization: HazardEmbeddingInitialization | str = (
            HazardEmbeddingInitialization.NORMAL
        ),
        initialization_seed: int = 42,
        initialization_std: float = 0.02,
        fixed_artifact: FixedHazardEmbeddingArtifact | None = None,
        fixed_artifact_fingerprint: str | None = None,
        fixed_scale: float = 1.0,
        residual_scale: float = 1.0,
        freeze_learned_embeddings: bool = False,
        hazard_registry: HazardRegistry = DEFAULT_HAZARD_REGISTRY,
        require_training_supported_hazards: bool = True,
        allow_partially_data_backed_for_training: bool = False,
        require_queryable_hazards_for_inference: bool = True,
        allow_fallback_hazard_for_inference: bool = False,
        allow_planned_hazard_counterfactuals: bool = False,
        # Compatibility aliases from the earlier draft.
        learned_initialization: HazardEmbeddingInitialization | str | None = None,
        residual_initialization: HazardEmbeddingInitialization | str | None = None,
        learned_residual_scale: float | None = None,
        zero_unknown_row: bool | None = None,
    ) -> None:
        super().__init__()
        if not isinstance(vocabulary, HazardEmbeddingVocabulary):
            raise TypeError(
                "vocabulary must be a HazardEmbeddingVocabulary."
            )
        if not isinstance(hazard_registry, HazardRegistry):
            raise TypeError("hazard_registry must be a HazardRegistry.")

        _require_positive_int("embedding_dim", embedding_dim)
        _require_nonnegative_int("initialization_seed", initialization_seed)
        self.initialization_std = _require_nonnegative_float(
            "initialization_std",
            initialization_std,
        )
        self.fixed_scale = _require_nonnegative_float(
            "fixed_scale",
            fixed_scale,
        )
        if learned_residual_scale is not None:
            residual_scale = learned_residual_scale
        self.residual_scale = _require_nonnegative_float(
            "residual_scale",
            residual_scale,
        )

        for name, value in (
            ("freeze_learned_embeddings", freeze_learned_embeddings),
            (
                "require_training_supported_hazards",
                require_training_supported_hazards,
            ),
            (
                "allow_partially_data_backed_for_training",
                allow_partially_data_backed_for_training,
            ),
            (
                "require_queryable_hazards_for_inference",
                require_queryable_hazards_for_inference,
            ),
            (
                "allow_fallback_hazard_for_inference",
                allow_fallback_hazard_for_inference,
            ),
            (
                "allow_planned_hazard_counterfactuals",
                allow_planned_hazard_counterfactuals,
            ),
        ):
            _require_boolean(name, value)

        self.vocabulary = vocabulary
        self.embedding_dim = embedding_dim
        self.mode = _normalize_mode(mode)
        self.unknown_policy = _normalize_unknown_policy(unknown_policy)
        self.unknown_embedding_policy = unknown_embedding_policy
        self.initialization_seed = initialization_seed
        self.hazard_registry = hazard_registry
        self.require_training_supported_hazards = (
            require_training_supported_hazards
        )
        self.allow_partially_data_backed_for_training = (
            allow_partially_data_backed_for_training
        )
        self.require_queryable_hazards_for_inference = (
            require_queryable_hazards_for_inference
        )
        self.allow_fallback_hazard_for_inference = (
            allow_fallback_hazard_for_inference
        )
        self.allow_planned_hazard_counterfactuals = (
            allow_planned_hazard_counterfactuals
        )

        if learned_initialization is not None:
            initialization = learned_initialization
        if residual_initialization is not None and (
            self.mode == HazardEmbeddingMode.FIXED_PLUS_RESIDUAL
        ):
            initialization = residual_initialization
        self.initialization = _normalize_initialization(initialization)

        if zero_unknown_row is not None:
            _require_boolean("zero_unknown_row", zero_unknown_row)
            translated = (
                UNKNOWN_EMBEDDING_POLICY_ZERO_FIXED
                if zero_unknown_row
                else UNKNOWN_EMBEDDING_POLICY_LEARNED
            )
            if (
                unknown_embedding_policy != UNKNOWN_EMBEDDING_POLICY_ERROR
                and unknown_embedding_policy != translated
            ):
                raise ValueError(
                    "zero_unknown_row conflicts with "
                    "unknown_embedding_policy."
                )
            if self.unknown_policy == UnknownHazardPolicy.USE_UNKNOWN_EMBEDDING:
                self.unknown_embedding_policy = translated

        if self.unknown_embedding_policy not in {
            UNKNOWN_EMBEDDING_POLICY_ERROR,
            UNKNOWN_EMBEDDING_POLICY_ZERO_FIXED,
            UNKNOWN_EMBEDDING_POLICY_LEARNED,
        }:
            raise ValueError(
                "unknown_embedding_policy must be 'error', 'zero_fixed', "
                "or 'learned'."
            )
        if self.unknown_policy == UnknownHazardPolicy.ERROR:
            if self.unknown_embedding_policy != UNKNOWN_EMBEDDING_POLICY_ERROR:
                raise ValueError(
                    "Unknown input policy 'error' requires unknown "
                    "embedding policy 'error'."
                )
        else:
            if not self.vocabulary.includes_unknown:
                raise ValueError(
                    "USE_UNKNOWN_EMBEDDING requires an explicit unknown "
                    "vocabulary row."
                )
            if self.unknown_embedding_policy not in {
                UNKNOWN_EMBEDDING_POLICY_ZERO_FIXED,
                UNKNOWN_EMBEDDING_POLICY_LEARNED,
            }:
                raise ValueError(
                    "Unknown-row lookup requires a zero-fixed or learned "
                    "unknown embedding policy."
                )

        if (
            self.initialization == HazardEmbeddingInitialization.NORMAL
            and self.mode != HazardEmbeddingMode.FIXED
            and self.initialization_std <= 0.0
        ):
            raise ValueError(
                "initialization_std must be strictly positive for normal "
                "learned initialization."
            )

        self._fixed_source_id: str | None = None
        self._fixed_source_version: str | None = None
        self._fixed_source_fingerprint: str | None = None
        self._fixed_embedding_fingerprint: str | None = None
        self._fixed_architecture_lineage_fingerprint: str | None = None

        padding_idx = (
            vocabulary.unknown_index
            if self.unknown_embedding_policy
            == UNKNOWN_EMBEDDING_POLICY_ZERO_FIXED
            else None
        )

        if self.mode == HazardEmbeddingMode.LEARNED:
            if fixed_artifact is not None:
                raise ValueError(
                    "Learned mode cannot receive a fixed artifact."
                )
            if fixed_artifact_fingerprint is not None:
                raise ValueError(
                    "Learned mode cannot receive a fixed artifact "
                    "fingerprint."
                )
            self.learned_embeddings: nn.Embedding | None = nn.Embedding(
                num_embeddings=len(vocabulary),
                embedding_dim=embedding_dim,
                padding_idx=padding_idx,
            )
            self.register_buffer(
                "fixed_embeddings",
                torch.empty(0, embedding_dim),
                persistent=False,
            )
        else:
            if fixed_artifact is None:
                raise ValueError(
                    f"Embedding mode {self.mode.value!r} requires a fixed "
                    "artifact."
                )
            fixed_artifact.assert_matches_vocabulary(vocabulary)
            if fixed_artifact.embedding_dim != embedding_dim:
                raise ValueError(
                    "Fixed artifact embedding width does not match "
                    "embedding_dim."
                )
            if (
                fixed_artifact_fingerprint is not None
                and fixed_artifact_fingerprint
                != fixed_artifact.embedding_fingerprint
            ):
                raise ValueError(
                    "Configured fixed-artifact fingerprint does not match "
                    "the supplied artifact."
                )
            self._fixed_source_id = fixed_artifact.source_id
            self._fixed_source_version = fixed_artifact.source_version
            self._fixed_source_fingerprint = (
                fixed_artifact.source_fingerprint
            )
            self._fixed_embedding_fingerprint = (
                fixed_artifact.embedding_fingerprint
            )
            self._fixed_architecture_lineage_fingerprint = (
                fixed_artifact.architecture_lineage_fingerprint
            )
            self.register_buffer(
                "fixed_embeddings",
                fixed_artifact.embedding_matrix.clone(),
                persistent=True,
            )

            if self.mode == HazardEmbeddingMode.FIXED_PLUS_RESIDUAL:
                self.learned_embeddings = nn.Embedding(
                    num_embeddings=len(vocabulary),
                    embedding_dim=embedding_dim,
                    padding_idx=padding_idx,
                ).to(dtype=self.fixed_embeddings.dtype)
            else:
                if (
                    self.unknown_embedding_policy
                    == UNKNOWN_EMBEDDING_POLICY_LEARNED
                ):
                    raise ValueError(
                        "Fixed-only mode cannot use a learned unknown row."
                    )
                self.learned_embeddings = None
                if padding_idx is not None and not torch.equal(
                    self.fixed_embeddings[padding_idx],
                    torch.zeros_like(self.fixed_embeddings[padding_idx]),
                ):
                    raise ValueError(
                        "A zero-fixed unknown policy requires a zero unknown "
                        "row in the fixed artifact."
                    )

        self.reset_parameters()
        if freeze_learned_embeddings:
            self.freeze_learned_embeddings()

    @classmethod
    def from_config(
        cls,
        config: "HazardEmbeddingConfig",
        *,
        hazard_registry: HazardRegistry = DEFAULT_HAZARD_REGISTRY,
        fixed_artifact: FixedHazardEmbeddingArtifact | None = None,
    ) -> "HazardEmbeddingLayer":
        from ..config import HazardEmbeddingConfig

        if not isinstance(config, HazardEmbeddingConfig):
            raise TypeError(
                "config must be a HazardEmbeddingConfig."
            )
        config.validate()
        vocabulary = build_hazard_embedding_vocabulary(
            hazard_registry=hazard_registry,
            include_fallback_hazard=config.include_fallback_hazard,
            include_unknown=config.include_unknown_row,
        )
        return cls(
            vocabulary=vocabulary,
            embedding_dim=config.embedding_dim,
            mode=config.mode,
            unknown_policy=config.unknown_input_policy,
            unknown_embedding_policy=config.unknown_embedding_policy,
            initialization=config.initialization,
            initialization_seed=config.initialization_seed,
            initialization_std=config.initialization_std,
            fixed_artifact=fixed_artifact,
            fixed_artifact_fingerprint=config.fixed_artifact_fingerprint,
            fixed_scale=config.fixed_scale,
            residual_scale=config.residual_scale,
            freeze_learned_embeddings=config.freeze_learned_embeddings,
            hazard_registry=hazard_registry,
            require_training_supported_hazards=(
                config.require_training_supported_hazards
            ),
            allow_partially_data_backed_for_training=(
                config.allow_partially_data_backed_for_training
            ),
            require_queryable_hazards_for_inference=(
                config.require_queryable_hazards_for_inference
            ),
            allow_fallback_hazard_for_inference=(
                config.allow_fallback_hazard_for_inference
            ),
            allow_planned_hazard_counterfactuals=(
                config.allow_planned_hazard_counterfactuals
            ),
        )

    @classmethod
    def from_hazard_registry(
        cls,
        *,
        embedding_dim: int,
        hazard_registry: HazardRegistry = DEFAULT_HAZARD_REGISTRY,
        mode: HazardEmbeddingMode | str = HazardEmbeddingMode.LEARNED,
        unknown_policy: UnknownHazardPolicy | str = UnknownHazardPolicy.ERROR,
        unknown_embedding_policy: str = UNKNOWN_EMBEDDING_POLICY_ERROR,
        include_fallback_hazard: bool = False,
        include_unknown: bool | None = None,
        fixed_artifact: FixedHazardEmbeddingArtifact | None = None,
        initialization: HazardEmbeddingInitialization | str = (
            HazardEmbeddingInitialization.NORMAL
        ),
        initialization_std: float = 0.02,
        initialization_seed: int = 42,
        fixed_scale: float = 1.0,
        residual_scale: float = 1.0,
        **kwargs: Any,
    ) -> "HazardEmbeddingLayer":
        policy = _normalize_unknown_policy(unknown_policy)
        if include_unknown is None:
            include_unknown = (
                policy == UnknownHazardPolicy.USE_UNKNOWN_EMBEDDING
            )
        vocabulary = build_hazard_embedding_vocabulary(
            hazard_registry=hazard_registry,
            include_fallback_hazard=include_fallback_hazard,
            include_unknown=include_unknown,
        )
        return cls(
            vocabulary=vocabulary,
            embedding_dim=embedding_dim,
            mode=mode,
            unknown_policy=policy,
            unknown_embedding_policy=unknown_embedding_policy,
            fixed_artifact=fixed_artifact,
            initialization=initialization,
            initialization_std=initialization_std,
            initialization_seed=initialization_seed,
            fixed_scale=fixed_scale,
            residual_scale=residual_scale,
            hazard_registry=hazard_registry,
            **kwargs,
        )

    def reset_parameters(self) -> None:
        if self.learned_embeddings is None:
            return
        weight = self.learned_embeddings.weight
        generator = _seeded_generator(
            device=weight.device,
            seed=self.initialization_seed,
        )
        if self.initialization == HazardEmbeddingInitialization.NORMAL:
            nn.init.normal_(
                weight,
                mean=0.0,
                std=self.initialization_std,
                generator=generator,
            )
        elif self.initialization == HazardEmbeddingInitialization.ZERO:
            nn.init.zeros_(weight)
        elif (
            self.initialization
            == HazardEmbeddingInitialization.XAVIER_UNIFORM
        ):
            nn.init.xavier_uniform_(weight, generator=generator)
        else:
            raise RuntimeError(
                f"Unhandled initialization {self.initialization!r}."
            )

        if self.learned_embeddings.padding_idx is not None:
            with torch.no_grad():
                weight[self.learned_embeddings.padding_idx].zero_()

    def forward(self, hazard_indices: torch.Tensor) -> torch.Tensor:
        self._validate_dense_indices(hazard_indices)
        if self.mode == HazardEmbeddingMode.LEARNED:
            if self.learned_embeddings is None:
                raise RuntimeError("Learned embedding table is unavailable.")
            return self.learned_embeddings(hazard_indices)

        fixed = F.embedding(hazard_indices, self.fixed_embeddings)
        if self.mode == HazardEmbeddingMode.FIXED:
            return fixed * self.fixed_scale
        if self.learned_embeddings is None:
            raise RuntimeError("Learned residual table is unavailable.")
        residual = self.learned_embeddings(hazard_indices)
        return (
            fixed * self.fixed_scale
            + residual * self.residual_scale
        )

    def _assert_operation_supported(
        self,
        index_batch: HazardIndexBatch,
        *,
        for_training: bool,
        counterfactual: bool,
        require_data_backed: bool,
        require_runtime_allowed: bool,
    ) -> None:
        for name, is_unknown in zip(
            index_batch.hazard_names,
            index_batch.unknown_mask.tolist(),
            strict=True,
        ):
            if is_unknown:
                continue
            if for_training or require_data_backed:
                if self.require_training_supported_hazards or require_data_backed:
                    self.hazard_registry.assert_training_supported_hazard(
                        name,
                        allow_partially_data_backed=(
                            self.allow_partially_data_backed_for_training
                            if not require_data_backed
                            else False
                        ),
                    )
                continue

            if require_runtime_allowed or (
                self.require_queryable_hazards_for_inference
            ):
                entry = self.hazard_registry.assert_queryable_hazard(
                    name,
                    allow_fallback=(
                        self.allow_fallback_hazard_for_inference
                    ),
                )
                if (
                    counterfactual
                    and entry.support_state == HazardSupportState.PLANNED
                    and not self.allow_planned_hazard_counterfactuals
                ):
                    raise ValueError(
                        f"Planned hazard {name!r} is not enabled for "
                        "counterfactual lookup."
                    )

    def lookup_index_batch(
        self,
        index_batch: HazardIndexBatch,
        *,
        for_training: bool = False,
        counterfactual: bool = False,
        require_runtime_allowed: bool = True,
        require_data_backed: bool = False,
    ) -> HazardEmbeddingLookup:
        if not isinstance(index_batch, HazardIndexBatch):
            raise TypeError("index_batch must be a HazardIndexBatch.")
        index_batch.assert_matches_vocabulary(self.vocabulary)
        if index_batch.device != self.device:
            index_batch = index_batch.to(self.device)
        self._assert_operation_supported(
            index_batch,
            for_training=for_training,
            counterfactual=counterfactual,
            require_data_backed=require_data_backed,
            require_runtime_allowed=require_runtime_allowed,
        )
        embeddings = self(index_batch.dense_indices)
        if embeddings.ndim != 2:
            raise RuntimeError(
                "HazardIndexBatch lookup must return a two-dimensional "
                "embedding matrix."
            )
        return HazardEmbeddingLookup(
            embeddings=embeddings,
            indices=index_batch,
            embedding_mode=self.mode,
            architecture_fingerprint=self.architecture_fingerprint(),
        )

    def lookup_names(
        self,
        hazard_names: Sequence[HazardKind | str],
        *,
        for_training: bool = False,
        counterfactual: bool = False,
        require_runtime_allowed: bool = True,
        require_data_backed: bool = False,
    ) -> HazardEmbeddingLookup:
        batch = self.vocabulary.encode_names(
            hazard_names,
            unknown_policy=self.unknown_policy,
            device=self.device,
        )
        return self.lookup_index_batch(
            batch,
            for_training=for_training,
            counterfactual=counterfactual,
            require_runtime_allowed=require_runtime_allowed,
            require_data_backed=require_data_backed,
        )

    def lookup_stable_ids(
        self,
        stable_hazard_ids: Sequence[int],
        *,
        for_training: bool = False,
        counterfactual: bool = False,
        require_runtime_allowed: bool = True,
        require_data_backed: bool = False,
    ) -> HazardEmbeddingLookup:
        batch = self.vocabulary.encode_stable_ids(
            stable_hazard_ids,
            unknown_policy=self.unknown_policy,
            device=self.device,
        )
        return self.lookup_index_batch(
            batch,
            for_training=for_training,
            counterfactual=counterfactual,
            require_runtime_allowed=require_runtime_allowed,
            require_data_backed=require_data_backed,
        )

    def runtime_embedding_table(
        self,
        *,
        require_data_backed: bool = False,
        counterfactual: bool = False,
    ) -> HazardEmbeddingLookup:
        return self.lookup_names(
            self.vocabulary.runtime_hazard_names,
            require_runtime_allowed=True,
            require_data_backed=require_data_backed,
            counterfactual=counterfactual,
        )

    def full_embedding_table(self) -> HazardEmbeddingLookup:
        dense_indices = torch.arange(
            len(self.vocabulary),
            dtype=torch.long,
            device=self.device,
        )
        batch = HazardIndexBatch(
            dense_indices=dense_indices,
            stable_hazard_ids=torch.tensor(
                self.vocabulary.stable_hazard_ids,
                dtype=torch.long,
                device=self.device,
            ),
            hazard_names=self.vocabulary.hazard_names,
            unknown_mask=torch.tensor(
                [entry.is_unknown for entry in self.vocabulary],
                dtype=torch.bool,
                device=self.device,
            ),
            vocabulary_fingerprint=self.vocabulary.fingerprint(),
        )
        # Full-table inspection is structural; it must not reject fallback or
        # planned rows that were explicitly included in the vocabulary.
        batch.assert_matches_vocabulary(self.vocabulary)
        embeddings = self(batch.dense_indices)
        return HazardEmbeddingLookup(
            embeddings=embeddings,
            indices=batch,
            embedding_mode=self.mode,
            architecture_fingerprint=self.architecture_fingerprint(),
        )

    def lookup_graph_hazards_for_nodes(
        self,
        graph_hazard_names: Sequence[HazardKind | str],
        node_batch_index: torch.Tensor,
        *,
        require_data_backed: bool = False,
        for_training: bool = False,
        counterfactual: bool = False,
    ) -> NodeAlignedHazardEmbeddingLookup:
        graph_lookup = self.lookup_names(
            graph_hazard_names,
            require_runtime_allowed=True,
            require_data_backed=require_data_backed,
            for_training=for_training,
            counterfactual=counterfactual,
        )
        node_batch_index = node_batch_index.to(
            device=graph_lookup.embeddings.device
        )
        node_embeddings = broadcast_graph_embeddings_to_nodes(
            graph_lookup.embeddings,
            node_batch_index,
        )
        return NodeAlignedHazardEmbeddingLookup(
            node_embeddings=node_embeddings,
            graph_lookup=graph_lookup,
            node_batch_index=node_batch_index,
        )

    def counterfactual_runtime_grid(
        self,
        *,
        num_graphs: int,
        require_data_backed: bool = False,
    ) -> tuple[HazardEmbeddingLookup, torch.Tensor]:
        _require_positive_int("num_graphs", num_graphs)
        lookup = self.runtime_embedding_table(
            require_data_backed=require_data_backed,
            counterfactual=True,
        )
        grid = lookup.embeddings.unsqueeze(0).expand(num_graphs, -1, -1)
        return lookup, grid

    @property
    def device(self) -> torch.device:
        if self.learned_embeddings is not None:
            return self.learned_embeddings.weight.device
        return self.fixed_embeddings.device

    @property
    def dtype(self) -> torch.dtype:
        if self.learned_embeddings is not None:
            return self.learned_embeddings.weight.dtype
        return self.fixed_embeddings.dtype

    @property
    def num_embeddings(self) -> int:
        return len(self.vocabulary)

    @property
    def trainable(self) -> bool:
        return (
            self.learned_embeddings is not None
            and self.learned_embeddings.weight.requires_grad
        )

    def freeze_learned_embeddings(self) -> None:
        if self.learned_embeddings is not None:
            self.learned_embeddings.weight.requires_grad_(False)

    def unfreeze_learned_embeddings(self) -> None:
        if self.learned_embeddings is None:
            raise ValueError(
                "This hazard embedding mode has no learned parameters."
            )
        self.learned_embeddings.weight.requires_grad_(True)

    def assert_matches_hazard_registry(
        self,
        hazard_registry: HazardRegistry,
    ) -> None:
        self.vocabulary.assert_matches_hazard_registry(hazard_registry)

    def assert_finite_parameters(self) -> None:
        for name, tensor in self.state_dict().items():
            if tensor.dtype.is_floating_point and not bool(
                torch.isfinite(tensor).all().item()
            ):
                raise ValueError(
                    f"Hazard embedding tensor {name!r} contains NaN or "
                    "infinity."
                )

    def architecture_dict(self) -> dict[str, Any]:
        return {
            "schema_version": HAZARD_EMBEDDING_ARCHITECTURE_SCHEMA_VERSION,
            "vocabulary_fingerprint": self.vocabulary.fingerprint(),
            "source_hazard_registry_fingerprint": (
                self.vocabulary.source_hazard_registry_fingerprint
            ),
            "vocabulary_name": self.vocabulary.vocabulary_name,
            "hazard_names": list(self.vocabulary.hazard_names),
            "stable_hazard_ids": list(
                self.vocabulary.stable_hazard_ids
            ),
            "include_fallback_hazard": (
                self.vocabulary.includes_fallback_hazard
            ),
            "include_unknown_row": self.vocabulary.includes_unknown,
            "embedding_dim": self.embedding_dim,
            "num_embeddings": self.num_embeddings,
            "mode": self.mode.value,
            "unknown_policy": self.unknown_policy.value,
            "unknown_embedding_policy": self.unknown_embedding_policy,
            "initialization": self.initialization.value,
            "initialization_std": self.initialization_std,
            "initialization_seed": self.initialization_seed,
            "fixed_scale": self.fixed_scale,
            "residual_scale": self.residual_scale,
            "require_training_supported_hazards": (
                self.require_training_supported_hazards
            ),
            "allow_partially_data_backed_for_training": (
                self.allow_partially_data_backed_for_training
            ),
            "require_queryable_hazards_for_inference": (
                self.require_queryable_hazards_for_inference
            ),
            "allow_fallback_hazard_for_inference": (
                self.allow_fallback_hazard_for_inference
            ),
            "allow_planned_hazard_counterfactuals": (
                self.allow_planned_hazard_counterfactuals
            ),
            "fixed_source_id": self._fixed_source_id,
            "fixed_source_version": self._fixed_source_version,
            "fixed_source_fingerprint": self._fixed_source_fingerprint,
            "fixed_embedding_fingerprint": (
                self._fixed_embedding_fingerprint
            ),
            "fixed_architecture_lineage_fingerprint": (
                self._fixed_architecture_lineage_fingerprint
            ),
        }

    def architecture_fingerprint(self) -> str:
        return _fingerprint(self.architecture_dict())

    def parameter_fingerprint(self) -> str:
        return _tensor_fingerprint(
            {
                name: tensor
                for name, tensor in self.state_dict().items()
            }
        )

    def effective_table_fingerprint(self) -> str:
        table = self.full_embedding_table().embeddings.detach().cpu()
        return _tensor_fingerprint({"embedding_matrix": table})

    def export_as_fixed_artifact(
        self,
        *,
        source_id: str,
        source_version: str,
    ) -> FixedHazardEmbeddingArtifact:
        _require_nonempty_string("source_id", source_id)
        _require_nonempty_string("source_version", source_version)
        effective_table = (
            self.full_embedding_table().embeddings.detach().cpu()
        )
        effective_fingerprint = _tensor_fingerprint(
            {"embedding_matrix": effective_table}
        )
        architecture_fingerprint = self.architecture_fingerprint()
        lineage_fingerprint = _fingerprint(
            {
                "source_id": source_id,
                "source_version": source_version,
                "architecture_fingerprint": architecture_fingerprint,
                "effective_table_fingerprint": effective_fingerprint,
            }
        )
        return FixedHazardEmbeddingArtifact(
            embedding_matrix=effective_table,
            hazard_names=self.vocabulary.hazard_names,
            stable_hazard_ids=self.vocabulary.stable_hazard_ids,
            vocabulary_fingerprint=self.vocabulary.fingerprint(),
            source_id=source_id,
            source_version=source_version,
            source_fingerprint=lineage_fingerprint,
            architecture_lineage_fingerprint=architecture_fingerprint,
        )

    def extra_repr(self) -> str:
        return (
            f"num_embeddings={self.num_embeddings}, "
            f"embedding_dim={self.embedding_dim}, "
            f"mode={self.mode.value!r}, "
            f"unknown_policy={self.unknown_policy.value!r}, "
            f"trainable={self.trainable}"
        )

    def _validate_dense_indices(self, hazard_indices: torch.Tensor) -> None:
        if not isinstance(hazard_indices, torch.Tensor):
            raise TypeError("hazard_indices must be a tensor.")
        if hazard_indices.dtype != torch.long:
            raise ValueError("hazard_indices must use torch.long.")
        if hazard_indices.numel() == 0:
            return
        minimum = int(hazard_indices.min().item())
        maximum = int(hazard_indices.max().item())
        if minimum < 0 or maximum >= len(self.vocabulary):
            raise IndexError(
                "hazard_indices contains values outside the active "
                f"vocabulary range [0, {len(self.vocabulary) - 1}]."
            )


# =============================================================================
# Vocabulary construction
# =============================================================================


def build_hazard_embedding_vocabulary(
    *,
    hazard_registry: HazardRegistry = DEFAULT_HAZARD_REGISTRY,
    include_fallback_hazard: bool = False,
    include_unknown: bool = False,
    vocabulary_name: str = DEFAULT_HAZARD_EMBEDDING_VOCABULARY_NAME,
) -> HazardEmbeddingVocabulary:
    if not isinstance(hazard_registry, HazardRegistry):
        raise TypeError("hazard_registry must be a HazardRegistry.")
    _require_boolean("include_fallback_hazard", include_fallback_hazard)
    _require_boolean("include_unknown", include_unknown)
    _require_nonempty_string("vocabulary_name", vocabulary_name)

    selected_entries = [
        hazard_registry.get_by_name(hazard)
        for hazard in hazard_registry.queryable_hazard_kinds
    ]
    if include_fallback_hazard:
        selected_entries.append(hazard_registry.fallback_entry)

    entries = [
        HazardVocabularyEntry(
            stable_hazard_id=entry.stable_hazard_id,
            name=entry.name.value,
            display_name=entry.display_name,
            runtime_allowed=entry.query_allowed,
            fallback_only=entry.fallback_only,
            support_state=entry.support_state,
            is_unknown=False,
        )
        for entry in selected_entries
    ]
    if include_unknown:
        entries.append(
            HazardVocabularyEntry(
                stable_hazard_id=HAZARD_ID_UNKNOWN,
                name=UNKNOWN_HAZARD_NAME,
                display_name=UNKNOWN_HAZARD_DISPLAY_NAME,
                runtime_allowed=False,
                fallback_only=False,
                support_state=None,
                is_unknown=True,
            )
        )

    vocabulary = HazardEmbeddingVocabulary(
        entries=tuple(entries),
        source_hazard_registry_name=hazard_registry.registry_name,
        source_hazard_registry_version=hazard_registry.registry_version,
        source_hazard_registry_fingerprint=(
            hazard_registry.compatibility_fingerprint()
        ),
        vocabulary_name=vocabulary_name,
    )
    vocabulary.assert_matches_hazard_registry(hazard_registry)
    return vocabulary


DEFAULT_RUNTIME_HAZARD_EMBEDDING_VOCABULARY: Final[
    HazardEmbeddingVocabulary
] = build_hazard_embedding_vocabulary()


# =============================================================================
# Packed-graph broadcasting
# =============================================================================


def broadcast_graph_embeddings_to_nodes(
    graph_embeddings: torch.Tensor,
    node_batch_index: torch.Tensor,
) -> torch.Tensor:
    if not isinstance(graph_embeddings, torch.Tensor):
        raise TypeError("graph_embeddings must be a tensor.")
    if graph_embeddings.ndim != 2:
        raise ValueError(
            "graph_embeddings must have shape "
            "[num_graphs, embedding_dim]."
        )
    if not graph_embeddings.dtype.is_floating_point:
        raise ValueError(
            "graph_embeddings must use a floating-point dtype."
        )
    if not bool(torch.isfinite(graph_embeddings).all().item()):
        raise ValueError("graph_embeddings must contain finite values.")
    if not isinstance(node_batch_index, torch.Tensor):
        raise TypeError("node_batch_index must be a tensor.")
    if (
        node_batch_index.ndim != 1
        or node_batch_index.dtype != torch.long
    ):
        raise ValueError(
            "node_batch_index must be a one-dimensional torch.long "
            "tensor."
        )
    if graph_embeddings.device != node_batch_index.device:
        raise ValueError(
            "graph_embeddings and node_batch_index must share one device."
        )

    num_graphs = int(graph_embeddings.shape[0])
    if node_batch_index.numel() == 0:
        if num_graphs != 0:
            raise ValueError(
                "A nonempty graph embedding table cannot be broadcast to "
                "an empty packed-node batch."
            )
        return graph_embeddings.new_empty(
            (0, int(graph_embeddings.shape[1]))
        )

    minimum = int(node_batch_index.min().item())
    maximum = int(node_batch_index.max().item())
    if minimum < 0 or maximum >= num_graphs:
        raise IndexError(
            "node_batch_index contains graph IDs outside the graph "
            f"embedding range [0, {num_graphs - 1}]."
        )
    observed = torch.unique(node_batch_index, sorted=True)
    expected = torch.arange(
        maximum + 1,
        device=node_batch_index.device,
        dtype=torch.long,
    )
    if not torch.equal(observed, expected):
        raise ValueError(
            "node_batch_index graph IDs must be contiguous from zero."
        )
    if maximum + 1 != num_graphs:
        raise ValueError(
            "graph_embeddings must contain exactly one row for every "
            "graph represented by node_batch_index."
        )
    return graph_embeddings[node_batch_index]


__all__ = (
    "CANONICAL_HAZARD_DISPLAY_NAMES",
    "CANONICAL_HAZARD_STABLE_IDS",
    "DEFAULT_HAZARD_EMBEDDING_VOCABULARY_NAME",
    "DEFAULT_RUNTIME_HAZARD_EMBEDDING_VOCABULARY",
    "FIXED_HAZARD_EMBEDDING_ARTIFACT_SCHEMA_VERSION",
    "FixedHazardEmbeddingArtifact",
    "HAZARD_EMBEDDING_ARCHITECTURE_SCHEMA_VERSION",
    "HAZARD_EMBEDDING_VOCABULARY_SCHEMA_VERSION",
    "HAZARD_ID_ALL_HAZARD",
    "HAZARD_ID_CIVIL_SECURITY_EVENT",
    "HAZARD_ID_FLOOD",
    "HAZARD_ID_FREEZING_RAIN",
    "HAZARD_ID_HEAT",
    "HAZARD_ID_OUTAGE",
    "HAZARD_ID_PLUVIAL_FLOOD",
    "HAZARD_ID_RIVERINE_FLOOD",
    "HAZARD_ID_ROAD_DISRUPTION",
    "HAZARD_ID_SNOWSTORM",
    "HAZARD_ID_UNKNOWN",
    "HAZARD_ID_WINTER_STORM",
    "HazardEmbeddingInitialization",
    "HazardEmbeddingLayer",
    "HazardEmbeddingLookup",
    "HazardEmbeddingMode",
    "HazardEmbeddingVocabulary",
    "HazardIndexBatch",
    "HazardVocabularyEntry",
    "NodeAlignedHazardEmbeddingLookup",
    "UNKNOWN_HAZARD_NAME",
    "UnknownHazardPolicy",
    "broadcast_graph_embeddings_to_nodes",
    "build_hazard_embedding_vocabulary",
)
