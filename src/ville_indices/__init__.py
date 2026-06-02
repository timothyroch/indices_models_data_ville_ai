"""VILLE_IA composite-index benchmark framework."""

from ville_indices.core.base import CompositeIndex
from ville_indices.core.recipe import Recipe, load_recipe

__all__ = ["CompositeIndex", "Recipe", "load_recipe", "run_index"]

__version__ = "0.1.0"


def __getattr__(name: str):
    if name == "run_index":
        from ville_indices.run import run_index

        return run_index
    raise AttributeError(name)
