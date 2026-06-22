"""
Public API for the V2 node-state fusion subsystem.

The package is organized around one stable orchestration boundary:

    NodeStateFusion

Current implemented algorithm:

    concat_projection

Public contracts:

- ``NodeAlignment``;
- ``NodeStateComponent``;
- ``NodeStateFusionInputs``;
- ``NodeStateFusionOutput``;
- ``ComponentProjection``;
- ``ConcatProjectionFusion``;
- ``NodeStateFusion``.

Future research modules intentionally exist as separate files but are not
imported here until they contain tested implementations:

- ``component_registry.py``;
- ``diagnostics.py``;
- ``gated_fusion.py``;
- ``film_conditioning.py``;
- ``hazard_conditioned_fusion.py``;
- ``node_type_experts.py``;
- ``component_attribution.py``;
- ``uncertainty_fusion.py``.

Keeping unfinished modules out of the public namespace prevents empty
placeholders from being mistaken for implemented capabilities.
"""

from __future__ import annotations

from .component_projection import (
    COMPONENT_PROJECTION_SCHEMA_VERSION,
    ComponentProjection,
    ComponentProjectionActivation,
    ComponentProjectionNormalization,
)
from .concat_projection import (
    CANONICAL_FUSION_COMPONENT_ORDER,
    CONCAT_PROJECTION_FUSION_SCHEMA_VERSION,
    CONCAT_PROJECTION_OUTPUT_SCHEMA_VERSION,
    ConcatProjectionFusion,
    ConcatProjectionFusionOutput,
    FUSION_COMPONENT_HAZARD_CONTEXT,
    FUSION_COMPONENT_HAZARD_MEMORY_STATE,
    FUSION_COMPONENT_MEMORY_STATE,
    FUSION_COMPONENT_NODE_TYPE_EMBEDDING,
    FUSION_COMPONENT_STATIC_STATE,
    canonical_component_order,
)
from .node_state_fusion import (
    CANONICAL_NODE_STATE_FUSION_MODES,
    IMPLEMENTED_NODE_STATE_FUSION_MODES,
    NODE_STATE_FUSION_ENCODER_SCHEMA_VERSION,
    NODE_STATE_FUSION_LEGACY_STATE_DICT_SCHEMA_VERSION,
    NodeStateFusion,
    NodeStateFusionMode,
)
from .schemas import (
    NODE_ALIGNMENT_SCHEMA_VERSION,
    NODE_STATE_COMPONENT_SCHEMA_VERSION,
    NODE_STATE_FUSION_INPUT_SCHEMA_VERSION,
    NODE_STATE_FUSION_OUTPUT_SCHEMA_VERSION,
    NodeAlignment,
    NodeStateComponent,
    NodeStateFusionInputs,
    NodeStateFusionOutput,
)


__all__ = (
    # Schema versions.
    "COMPONENT_PROJECTION_SCHEMA_VERSION",
    "CONCAT_PROJECTION_FUSION_SCHEMA_VERSION",
    "CONCAT_PROJECTION_OUTPUT_SCHEMA_VERSION",
    "NODE_ALIGNMENT_SCHEMA_VERSION",
    "NODE_STATE_COMPONENT_SCHEMA_VERSION",
    "NODE_STATE_FUSION_ENCODER_SCHEMA_VERSION",
    "NODE_STATE_FUSION_INPUT_SCHEMA_VERSION",
    "NODE_STATE_FUSION_LEGACY_STATE_DICT_SCHEMA_VERSION",
    "NODE_STATE_FUSION_OUTPUT_SCHEMA_VERSION",

    # Canonical component identity.
    "CANONICAL_FUSION_COMPONENT_ORDER",
    "FUSION_COMPONENT_STATIC_STATE",
    "FUSION_COMPONENT_MEMORY_STATE",
    "FUSION_COMPONENT_HAZARD_MEMORY_STATE",
    "FUSION_COMPONENT_HAZARD_CONTEXT",
    "FUSION_COMPONENT_NODE_TYPE_EMBEDDING",
    "canonical_component_order",

    # Fusion-mode capability identity.
    "CANONICAL_NODE_STATE_FUSION_MODES",
    "IMPLEMENTED_NODE_STATE_FUSION_MODES",
    "NodeStateFusionMode",

    # Typed schemas.
    "NodeAlignment",
    "NodeStateComponent",
    "NodeStateFusionInputs",
    "NodeStateFusionOutput",

    # Projection primitive.
    "ComponentProjection",
    "ComponentProjectionActivation",
    "ComponentProjectionNormalization",

    # Baseline algorithm.
    "ConcatProjectionFusion",
    "ConcatProjectionFusionOutput",

    # Stable orchestration boundary.
    "NodeStateFusion",
)
