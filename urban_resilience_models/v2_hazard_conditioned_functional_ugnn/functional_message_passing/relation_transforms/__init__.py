"""
Public exports for functional message-passing relation transforms.

Target path:
    urban_resilience_models/
        v2_hazard_conditioned_functional_ugnn/
            functional_message_passing/
                relation_transforms/
                    __init__.py

This package exposes:

- the shared relation-transform implementation;
- the independently parameterized per-relation implementation;
- the public metadata-preserving dispatcher.

Importing this package performs no model construction, registry compilation,
device movement, or other runtime side effect.
"""

from .shared_transform import (
    SHARED_RELATION_TRANSFORM_SCHEMA_VERSION,
    SharedRelationTransform,
)
from .per_relation_transform import (
    PER_RELATION_TRANSFORM_SCHEMA_VERSION,
    PerRelationTransform,
)
from .relation_transforms import (
    RELATION_TRANSFORMS_DISPATCHER_SCHEMA_VERSION,
    RelationTransforms,
)


__all__ = (
    "PER_RELATION_TRANSFORM_SCHEMA_VERSION",
    "RELATION_TRANSFORMS_DISPATCHER_SCHEMA_VERSION",
    "SHARED_RELATION_TRANSFORM_SCHEMA_VERSION",
    "PerRelationTransform",
    "RelationTransforms",
    "SharedRelationTransform",
)
