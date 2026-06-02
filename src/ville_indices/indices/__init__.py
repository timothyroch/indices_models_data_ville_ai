"""Available index implementations."""

from ville_indices.core.registry import register_index
from ville_indices.indices.dummy_index import DummyAdditiveIndex
from ville_indices.indices.sovi_like import SoviLikeIndex
from ville_indices.indices.svi_like import SviLikeIndex

register_index("dummy_additive_index", DummyAdditiveIndex)
register_index("sovi_like", SoviLikeIndex)
register_index("svi_like", SviLikeIndex)

__all__ = ["DummyAdditiveIndex", "SoviLikeIndex", "SviLikeIndex"]
