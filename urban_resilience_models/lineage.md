# Model Lineage

This document explains why `urban_resilience_models/` starts directly with a Version 2 model family, and how the controlled benchmark work leads into the independent model architecture.

---

## 1. Why lineage matters

The model family in this repository should not appear out of nowhere.

It is not simply:

```text
a new GNN architecture
```

It is the next step after a controlled empirical benchmark.

The benchmark established the need for a model that goes beyond:

```text
static vulnerability indices
history-only baselines
calibrated index predictors
tabular feature-parity ML
generic graph baselines
no-edge neural controls
random/placebo graph controls
kNN graph controls
```

The V2 model family exists because the benchmark made one thing clear:

```text
Static indices and generic graph models are not enough.
The next model must combine urban memory, hazard context, functional relation pathways, and interpretability.
```

---

## 2. Versioning logic

The independent model package starts at Version 2 because Version 1 already exists conceptually in the benchmark work.

```text
Version 1:
  controlled graph benchmarking inside urban_graph_benchmark/

Version 2:
  first independent custom research model inside urban_resilience_models/
```

This avoids duplicating the controlled benchmark models inside the independent model package.

The benchmark remains the evaluation laboratory.

The model package becomes the home of reusable research and deployment-oriented model families.

---

## 3. Version 1 — Controlled benchmark foundation

Version 1 refers to the controlled benchmark work that tested whether graph structure could add value beyond strong non-graph and placebo controls.

This work includes:

```text
G1 / G1.5:
  controlled graph proof-of-concept models

B1:
  direct static SoVI validation

B0:
  history-only baselines

B2:
  calibrated SoVI predictors

B3:
  tabular feature-parity ML

B4:
  no-edge neural control
  random/placebo graph control
  kNN graph control
  real adjacency graph control
```

The scientific role of Version 1 was not to create the final model.

Its role was to answer:

```text
What must a serious urban graph model beat?
```

Version 1 showed that a graph claim is weak if it only beats a static index.

A stronger graph claim must compare against:

```text
history
calibrated indices
tabular ML with the same features
no-edge neural models
random graph controls
generic spatial graph controls
real graph topology
```

This benchmark foundation is the reason Version 2 can now be designed with more ambition.

---

## 4. Lessons from Version 1

The benchmark work produced several guiding lessons.

### 4.1 Static vulnerability is useful but incomplete

SVI and SoVI provide meaningful vulnerability signals, but they are not enough to explain operational disruption burden on their own.

They are better understood as:

```text
baseline vulnerability context
```

rather than complete predictive models.

---

### 4.2 History matters

History and recent stress signals are often strong predictors.

This justifies the need for an urban memory component.

A future model should not ignore historical stress because the benchmark already showed that temporal patterns matter.

This motivates:

```text
Urban Memory Encoder
Hazard-Queried Memory
temporal attention
lag/rolling memory controls
recurrent or transformer memory variants
```

---

### 4.3 Graph value must be controlled

A graph model can look strong for the wrong reason.

It may win because of:

```text
better features
temporal history
neural capacity
generic spatial smoothing
random topology effects
```

Therefore, the model family must preserve strong ablations and controls.

This motivates:

```text
no-edge variants
random-edge variants
kNN variants
real-adjacency variants
hazard-blind variants
no-gate variants
no-attention variants
```

---

### 4.4 Generic message passing is not enough

The benchmark graph models helped establish a controlled starting point, but they did not yet represent the full urban mechanism.

The next step should not be another generic GraphSAGE-style model.

The next step should be:

```text
hazard-conditioned functional message passing
```

where different hazards activate different urban relation families.

---

### 4.5 Interpretability should be designed early

Urban resilience models are not only prediction systems.

They should help answer:

```text
Why is this area at risk?
Which relation families mattered?
Which neighboring entities mattered?
Which past events mattered?
How would the prediction change under another hazard?
```

This motivates:

```text
relation-family gates
edge attention
temporal attention
pathway exports
counterfactual hazard swaps
uncertainty summaries
UI-ready explanation payloads
```

---

## 5. Version 2 — First independent custom research model

Version 2 is the first model family that lives independently inside:

```text
urban_resilience_models/
```

The active V2 folder is:

```text
urban_resilience_models/v2_hazard_conditioned_functional_ugnn/
```

Full name:

```text
Hazard-Conditioned Functional Urban Graph Neural Network
```

Compact name:

```text
Hazard-Conditioned Functional Message Passing
```

V2 is not intended to be another benchmark baseline.

It is intended to be the first custom research model whose architecture is defined by:

```text
urban memory
hazard queries
functional relation families
hazard-conditioned relation gates
edge-level attention
functional message passing
prediction and uncertainty heads
pathway explanation exports
```

---

## 6. Conceptual shift from Version 1 to Version 2

Version 1 asked:

```text
Does graph structure help under controlled conditions?
```

Version 2 asks:

```text
Can an urban graph model learn hazard-specific functional pathways of disruption?
```

The shift is important.

Version 1 was mostly about controlled empirical comparison.

Version 2 is about building an interpretable model of urban risk mechanisms.

---

## 7. Version 1 versus Version 2

| Dimension            | Version 1 benchmark work           | Version 2 model family                         |
| -------------------- | ---------------------------------- | ---------------------------------------------- |
| Main location        | `urban_graph_benchmark/`           | `urban_resilience_models/`                     |
| Main role            | Controlled evaluation              | Independent model architecture                 |
| Main question        | Does graph structure help?         | Which hazard-specific pathways drive risk?     |
| Graph type           | Controlled spatial graph variants  | Functional urban relation graph                |
| Temporal information | Lag/rolling/history features       | Urban memory encoder and hazard-queried memory |
| Hazard handling      | Mostly target-specific or implicit | Explicit hazard query conditioning             |
| Message passing      | Generic graph message passing      | Hazard-conditioned functional message passing  |
| Interpretability     | Metrics and ablations              | Gates, attention, pathways, counterfactuals    |
| Deployment relevance | Research benchmark                 | Future reusable model/API layer                |

---

## 8. Dependency direction

The model package should remain independent from benchmark internals.

Correct dependency direction:

```text
urban_resilience_models/common
        ↓
urban_resilience_models/v2_hazard_conditioned_functional_ugnn
        ↓
benchmark adapters / experiments / training scripts
        ↓
UI or deployment wrappers
```

The model may provide adapters for benchmark artifacts.

For example:

```text
data/benchmark_adapters.py
```

may know how to read benchmark outputs.

However, the core model should not depend on benchmark-specific file paths, benchmark report names, or one-off experiment scripts.

Allowed:

```text
benchmark_adapters.py reads cd_month_panel.parquet
benchmark_adapters.py maps benchmark columns into V2 schemas
```

Not allowed:

```text
model.py hard-codes benchmark output paths
functional_message_passing/layer.py knows benchmark column names
prediction_heads.py writes benchmark reports
```

The model family should be reusable beyond the original benchmark.

---

## 9. Why Version 2 is not production yet

Version 2 is a research model family.

Its first goal is not immediate UI deployment.

Its first goal is to establish stable contracts and a controlled architecture:

```text
schemas
configuration objects
relation ontology
module interfaces
ablation ladder
minimal model forward pass
shape tests
evaluation hooks
explanation payload contracts
```

Production deployment can come later.

The model should be designed so that future ML engineers can eventually use it, but it should not prematurely optimize for production before the scientific core is validated.

---

## 10. Expected evolution

The expected evolution is:

```text
V1:
  controlled graph benchmark

V2.0:
  minimal hazard-conditioned functional message passing
  lag/rolling urban memory
  simple hazard embeddings
  relation-family gates
  simple edge attention
  prediction head
  basic explanation exports

V2.1:
  recurrent or transformer urban memory
  hazard-queried memory
  stronger relation priors
  richer ablations

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

The exact numbering may evolve, but the principle should remain:

```text
controlled benchmark first
custom functional architecture second
full heterogeneous urban graph later
```

---

## 11. What must be preserved from the benchmark

Even though V2 is independent, it should preserve the benchmark discipline.

Every new mechanism should have a control.

Examples:

```text
Urban Memory Encoder
  compared against lag/rolling history

Hazard Query Encoder
  compared against hazard-blind model

Relation-Family Gate
  compared against uniform gates or no gates

Edge Attention
  compared against uniform attention

Functional Message Passing
  compared against generic message passing

Real topology
  compared against no-edge, random-edge, and kNN controls

Uncertainty Head
  evaluated with calibration or interval metrics

Reporting-Bias Head
  compared against direct observed-report prediction
```

The strongest V2 result is not merely:

```text
lower MAE
```

The strongest result is:

```text
better prediction
better top-k prioritization
real topology beating placebo topology
hazard-specific gate shifts
stable pathway explanations
useful uncertainty estimates
```

---

## 12. North-star transition

The project’s intellectual transition is:

```text
from static vulnerability indices
to controlled graph benchmarks
to hazard-conditioned functional urban reasoning
```

Version 1 made the project credible.

Version 2 should make it original.

The V2 model should eventually allow a future reader to say:

> This model remembers past urban stress, receives a hazard query, activates functional urban relation pathways, passes information through the relevant parts of the city, predicts future burden, and exports interpretable pathway evidence.

That is the lineage.

Everything in `urban_resilience_models/` should serve that progression.
