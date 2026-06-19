# Model Family Manifest

This manifest lists the model families that live under:

```text
urban_resilience_models/
```

The purpose of this file is to make the model-family layer explicit, versioned, and understandable to future collaborators.

This folder is not the benchmark layer.

It is the home for reusable urban resilience model architectures that may eventually support research experiments, inference services, and UI/dashboard integration.

---

## 1. Manifest purpose

The manifest answers five questions:

```text
Which model families exist?
What scientific role does each family play?
What is the current implementation status?
What benchmark or prior work justified the family?
What should collaborators read first?
```

It should stay concise.

Detailed architecture notes belong in each model family’s own documentation folder.

---

## 2. Active model families

| Model family ID                         | Name                                                     | Status                                   | Role                                                                        |
| --------------------------------------- | -------------------------------------------------------- | ---------------------------------------- | --------------------------------------------------------------------------- |
| `v2_hazard_conditioned_functional_ugnn` | Hazard-Conditioned Functional Urban Graph Neural Network | design / skeleton / early implementation | First independent custom research model after the controlled benchmark work |

---

## 3. Model family: `v2_hazard_conditioned_functional_ugnn`

### 3.1 Short name

```text
V2 Hazard-Conditioned Functional UGNN
```

### 3.2 Full name

```text
Hazard-Conditioned Functional Urban Graph Neural Network
```

### 3.3 Compact research framing

```text
Hazard-Conditioned Functional Message Passing
```

### 3.4 Location

```text
urban_resilience_models/v2_hazard_conditioned_functional_ugnn/
```

### 3.5 Status

```text
design / skeleton / early implementation
```

This model family is not yet production-ready.

It is currently in the architecture-contract phase.

The first implementation priority is:

```text
identity documents
lineage
model-family manifest
V2 README
architecture north star
relation ontology
module interfaces
ablation ladder
constants
schemas
configuration objects
minimal forward pass
shape tests
```

---

## 4. Scientific role

`v2_hazard_conditioned_functional_ugnn` is the first independent model family after the controlled graph benchmark work.

The benchmark established that a serious urban graph model must be compared against:

```text
static vulnerability indices
history-only baselines
calibrated index predictors
tabular feature-parity ML
no-edge neural controls
random/placebo graph controls
kNN graph controls
real graph topology
```

V2 exists because the next model should not simply be another generic graph neural network.

It should model:

```text
urban memory
hazard queries
functional relation families
hazard-conditioned relation gates
edge-level attention
prediction and uncertainty
pathway explanations
```

The central research question is:

> Can an urban graph model learn different risk pathways for different hazards, while remaining predictive, interpretable, and controlled against non-graph and placebo-graph baselines?

---

## 5. Core architecture summary

The V2 architecture is built around seven core components:

```text
1. Urban Memory Encoder
2. Hazard Query Encoder
3. Relation-Family Gate
4. Edge-Level Attention
5. Functional Message Passing
6. Prediction + Uncertainty Head
7. Explanation / Pathway Export
```

The high-level flow is:

```text
static urban features
+ temporal history
+ hazard/scenario context
+ typed urban relations
        |
        v
urban memory encoding
        |
        v
hazard-conditioned memory retrieval
        |
        v
relation-family gating
        |
        v
edge-level attention
        |
        v
functional message passing
        |
        v
prediction / uncertainty / reporting heads
        |
        v
pathway explanation exports
```

The model should eventually be able to answer:

```text
What is the predicted future burden?
How uncertain is the prediction?
Which hazard was queried?
Which relation families were activated?
Which edges or neighboring entities mattered?
Which past time windows mattered?
How would the prediction change under another hazard?
```

---

## 6. Intended first implementation scope

The first working V2.0 implementation should be disciplined.

It should start with:

```text
lag/rolling urban memory
simple learned hazard embeddings
explicit relation-family IDs
hazard-conditioned relation gates
simple edge attention
relation-specific transforms
mean or sum aggregation
single prediction head
basic relation-gate and attention exports
```

It should not start with:

```text
full transformer memory
full heterogeneous graph
complete UI deployment
reporting-bias head as a required component
complex counterfactual engine
large-scale uncertainty calibration
all hazard types at once
```

Those belong in later V2.x or V3 stages.

---

## 7. Expected evolution

The expected evolution is:

```text
V2.0:
  minimal hazard-conditioned functional message passing
  lag/rolling memory
  simple hazard embeddings
  relation gates
  simple edge attention
  prediction head
  basic explanation export

V2.1:
  recurrent or transformer urban memory
  hazard-queried memory
  richer relation priors
  stronger ablations

V2.2:
  uncertainty heads
  counterfactual hazard swaps
  reporting-bias-aware prediction head

V3:
  fuller heterogeneous functional urban graph
  infrastructure/service nodes
  flood-zone nodes
  heat-island/canopy nodes
  road/service/access/dependency pathways
```

The numbering may evolve, but the principle should remain:

```text
controlled benchmark first
custom functional architecture second
full heterogeneous urban graph later
```

---

## 8. Required controls and ablations

Every new mechanism should have a control.

The minimum ablation ladder should include:

```text
B3-style tabular feature parity
no-edge neural
random-edge graph
kNN graph
real adjacency graph
real graph + hazard embedding
real graph + relation gates
real graph + edge attention
real graph + hazard-queried memory
real graph + uncertainty/reporting extensions
```

For V2, a module is not justified because it is elegant.

It is justified if it improves at least one of:

```text
prediction
top-k prioritization
hazard-specific behavior
interpretability
stability
uncertainty calibration
scientific understanding
```

---

## 9. Primary success criteria

The strongest V2 result would show:

```text
lower MAE or better count prediction
+ better top-k municipal prioritization
+ real topology beating placebo topology
+ hazard-specific relation gates shifting in plausible ways
+ meaningful memory retrieval under different hazards
+ stable pathway explanations
+ useful uncertainty estimates
```

A weaker but still useful result would show that only some of these conditions hold.

A negative result is also scientifically useful if it clarifies that:

```text
features dominate topology
history dominates graph structure
relation ontology needs refinement
hazard conditioning does not yet add value
```

---

## 10. Documentation entry points

Recommended reading order:

```text
urban_resilience_models/README.md
urban_resilience_models/lineage.md
urban_resilience_models/model_family_manifest.md
urban_resilience_models/v2_hazard_conditioned_functional_ugnn/README.md
urban_resilience_models/v2_hazard_conditioned_functional_ugnn/docs/architecture_north_star.md
urban_resilience_models/v2_hazard_conditioned_functional_ugnn/docs/relation_family_ontology.md
urban_resilience_models/v2_hazard_conditioned_functional_ugnn/docs/module_interfaces.md
urban_resilience_models/v2_hazard_conditioned_functional_ugnn/docs/ablation_ladder.md
```

---

## 11. Current manifest status

| Field                        | Value                                   |
| ---------------------------- | --------------------------------------- |
| Manifest version             | `0.1`                                   |
| Active model families        | `1`                                     |
| Primary active family        | `v2_hazard_conditioned_functional_ugnn` |
| Production-ready models      | `0`                                     |
| Research-ready models        | `0`                                     |
| Skeleton/design-stage models | `1`                                     |

---

## 12. North-star reminder

This model-family layer exists because the project is moving:

```text
from static vulnerability indices
to controlled graph benchmarks
to hazard-conditioned functional urban reasoning
```

The benchmark made the project credible.

The independent model family should make it original.

Everything in this manifest should preserve that distinction.
