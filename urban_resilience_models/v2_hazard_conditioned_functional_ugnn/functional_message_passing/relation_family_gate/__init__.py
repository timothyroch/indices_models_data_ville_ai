"""
Public API for hazard-conditioned relation gating.

Target path:
    urban_resilience_models/
        v2_hazard_conditioned_functional_ugnn/
            functional_message_passing/
                relation_family_gate/
                    __init__.py

The historical package name ``relation_family_gate`` is retained for research
continuity. In the bounded V2.0 implementation, however, trainable gates operate
over the exact compiled relation axis rather than a pooled semantic-family
axis. Semantic family metadata remains available through ``RelationGateAxis``
for diagnostics, explanations, and future hierarchical extensions.

This package exposes:

- immutable relation-gate schemas;
- neural exact-relation logit prediction;
- optional compiled hazard-relation prior integration;
- independent sigmoid activation;
- complete relation-gate orchestration;
- compact aliases for common call sites.

Importing this package does not construct model components or mutate registries.
"""

from .activations import (
    RELATION_GATE_ACTIVATIONS_SCHEMA_VERSION,
    GateActivation,
    RelationGateActivation,
    apply_relation_gate_activation,
    sigmoid_gate_activation,
)
from .gate_network import (
    RELATION_GATE_NETWORK_SCHEMA_VERSION,
    GateNetwork,
    RelationGateNetwork,
)
from .relation_family_gate import (
    RELATION_FAMILY_GATE_SCHEMA_VERSION,
    RelationFamilyGate,
    RelationGate,
)
from .relation_priors import (
    RELATION_PRIOR_INTEGRATION_SCHEMA_VERSION,
    RelationPriorBuilder,
    RelationPriorContributionBuilder,
    RelationPriorIntegration,
)
from .schemas import (
    GATE_ACTIVATION_OUTPUT_SCHEMA_VERSION,
    GATE_NETWORK_OUTPUT_SCHEMA_VERSION,
    RELATION_GATE_AXIS_SCHEMA_VERSION,
    RELATION_PRIOR_CONTRIBUTION_SCHEMA_VERSION,
    GateActivationOutput,
    GateNetworkOutput,
    RelationGateAxis,
    RelationGateOutput,
    RelationPriorContribution,
)


__all__ = (
    # Schema versions
    "GATE_ACTIVATION_OUTPUT_SCHEMA_VERSION",
    "GATE_NETWORK_OUTPUT_SCHEMA_VERSION",
    "RELATION_FAMILY_GATE_SCHEMA_VERSION",
    "RELATION_GATE_ACTIVATIONS_SCHEMA_VERSION",
    "RELATION_GATE_AXIS_SCHEMA_VERSION",
    "RELATION_GATE_NETWORK_SCHEMA_VERSION",
    "RELATION_PRIOR_CONTRIBUTION_SCHEMA_VERSION",
    "RELATION_PRIOR_INTEGRATION_SCHEMA_VERSION",
    # Immutable contracts
    "GateActivationOutput",
    "GateNetworkOutput",
    "RelationGateAxis",
    "RelationGateOutput",
    "RelationPriorContribution",
    # Neural gate network
    "GateNetwork",
    "RelationGateNetwork",
    # Prior integration
    "RelationPriorBuilder",
    "RelationPriorContributionBuilder",
    "RelationPriorIntegration",
    # Activation
    "GateActivation",
    "RelationGateActivation",
    "apply_relation_gate_activation",
    "sigmoid_gate_activation",
    # Complete orchestrator
    "RelationFamilyGate",
    "RelationGate",
)
