"""Registry for index implementations."""

from __future__ import annotations

from typing import Type

from ville_indices.core.base import CompositeIndex

_REGISTRY: dict[str, Type[CompositeIndex]] = {}


def register_index(name: str, index_class: Type[CompositeIndex], *aliases: str) -> None:
    for key in (name, *aliases):
        _REGISTRY[key] = index_class


def get_index_class(name: str) -> Type[CompositeIndex]:
    try:
        return _REGISTRY[name]
    except KeyError as exc:
        available = ", ".join(sorted(_REGISTRY)) or "none"
        raise KeyError(f"No index class registered for '{name}'. Available: {available}") from exc


def create_index(name: str, *args, **kwargs) -> CompositeIndex:
    return get_index_class(name)(*args, **kwargs)


def registered_indices() -> list[str]:
    return sorted(_REGISTRY)
