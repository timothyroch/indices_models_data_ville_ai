# V2 Hazard-Conditioned Functional UGNN

**Model family:** `v2_hazard_conditioned_functional_ugnn`
**Full name:** Hazard-Conditioned Functional Urban Graph Neural Network
**Compact framing:** Hazard-Conditioned Functional Message Passing
**Status:** not production-ready

---

## 1. What this model is

`v2_hazard_conditioned_functional_ugnn` is the first independent custom research model family after the controlled benchmark work.

It is not meant to be just another graph neural network applied to city data.

It is designed as:

```text
hazard-conditioned functional message passing
over urban units with temporal memory
and interpretable relation pathways
```

The core idea is:

```text
The current hazard decides which urban relations matter,
the city’s past stress history informs node state,
and message passing becomes a functional explanation of risk pathways.
```

In practical terms, the model should eventually answer:

```text
What is the predicted future urban burden?
Which hazard is being queried?
Which past stress signals mattered?
Which relation families were activated?
Which edges or neighboring urban entities mattered?
How uncertain is the prediction?
How would the explanation change under another hazard?
```

---

## 2. Why this model exists

The controlled benchmark work showed that a serious urban graph model cannot simply be compared against a static vulnerability index.

A credible graph model must be compared against:

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

That benchmark foundation justified moving beyond generic graph baselines.

The purpose of V2 is to build the first model whose main contribution is not merely graph message passing, but:

```text
hazard-conditioned routing of information
through functional urban relation families
with explicit urban memory
and interpretable pathway exports
```

---

## 3. Version lineage

This model starts at Version 2 because Version 1 exists conceptually inside the benchmark work.

```text
Version 1:
  controlled graph benchmarking
  G1 / G1.5 proof-of-concept graph models
  B1 direct SoVI validation
  B0 history-only baselines
  B2 calibrated SoVI baselines
  B3 tabular feature-parity ML
  B4 no-edge / random-edge / kNN / real graph controls

Version 2:
  first independent custom research model
  hazard-conditioned functional message passing
  urban memory
  relation-family gates
  edge attention
  prediction and uncertainty heads
  pathway explanation exports
```

Version 1 asked:

```text
Does graph structure help under controlled conditions?
```

Version 2 asks:

```text
Can an urban graph model learn hazard-specific functional pathways of disruption?
```

---

## 4. Core research question

The central research question is:

> Can an urban graph model learn different risk pathways for different hazards, while remaining predictive, interpretable, and controlled against non-graph and placebo-graph baselines?

This expands into five operational questions:

```text
1. Urban memory:
   Which past events, complaints, seasons, or stress patterns matter?

2. Hazard conditioning:
   Does the model behave differently for flood, heat, outage, road disruption, or civil-security queries?

3. Functional relations:
   Which relation families are activated under each hazard?

4. Spatial/message-passing value:
   Does real topology help beyond no-edge, tabular, random-edge, and kNN controls?

5. Interpretability:
   Can the model export relation gates, edge attention, pathway summaries, uncertainty, and counterfactual hazard explanations?
```

---

## 5. Core modules

The model family is organized around seven core modules.

```text
1. Urban Memory Encoder
2. Hazard Query Encoder
3. Relation-Family Gate
4. Edge-Level Attention
5. Functional Message Passing
6. Prediction + Uncertainty Head
7. Explanation / Pathway Export
```

### 5.1 Urban Memory Encoder

Purpose:

```text
Encode each urban node’s past stress trajectory.
```

Possible inputs:

```text
past 311 counts
past hazard-specific event counts
lag features
rolling-window features
seasonality
past event indicators
history masks
```

Possible implementations:

```text
lag/rolling memory
GRU
LSTM
temporal attention
small temporal transformer
hazard-queried memory
```

Scientific role:

```text
History matters, so each urban node should carry memory.
```

---

### 5.2 Hazard Query Encoder

Purpose:

```text
Encode the current hazard or scenario being queried.
```

Possible inputs:

```text
hazard type
hazard family
month or season
weather/hydro indicators
event severity
forecast horizon
scenario metadata
```

Example hazards:

```text
flood
heat
outage
road disruption
civil-security event
all-hazard
```

Scientific role:

```text
The same city should activate different mechanisms under different hazards.
```

---

### 5.3 Relation-Family Gate

Purpose:

```text
Given a hazard query, decide which relation families should be active.
```

Example behavior:

```text
Flood:
  hydrological exposure: high
  drainage dependency: high
  canopy protection: low

Heat:
  heat exposure: high
  canopy protection: high
  drainage dependency: low
```

Scientific role:

```text
Relation gates are the first major explanation object.
They answer: which urban mechanism mattered?
```

---

### 5.4 Edge-Level Attention

Purpose:

```text
Within active relation families, decide which specific edges or neighboring entities matter.
```

Relation gates answer:

```text
Which mechanism type mattered?
```

Edge attention answers:

```text
Which specific connection mattered?
```

Examples:

```text
Which neighboring tract mattered?
Which flood-zone edge mattered?
Which service-access relation mattered?
Which past temporal edge mattered?
```

---

### 5.5 Functional Message Passing

Purpose:

```text
Route information through the urban graph using hazard-conditioned gates and edge attention.
```

Conceptual message:

```text
message =
  relation-specific transform(source node state)
  × hazard-conditioned relation gate
  × edge-level attention
  × optional edge normalization
```

Scientific role:

```text
This is the core custom model contribution.
```

The model should not pass the same messages for every hazard.

---

### 5.6 Prediction + Uncertainty Head

Purpose:

```text
Convert learned node states into predictions and uncertainty estimates.
```

Possible outputs:

```text
future burden
risk score
expected count
top-k prioritization score
prediction interval
uncertainty score
latent disruption score
reporting propensity score
```

The first implementation may only use a simple prediction head.

Uncertainty and reporting-bias heads can be added in later V2.x stages.

---

### 5.7 Explanation / Pathway Export

Purpose:

```text
Convert internal model objects into structured explanation artifacts.
```

Possible exports:

```text
top relation families
relation gate values
top edges
edge attention summaries
temporal attention summaries
top history windows
pathway scores
counterfactual hazard deltas
uncertainty summaries
UI-ready explanation payloads
```

Interpretability should be designed into the model, not added as an afterthought.

---

## 6. Expected package structure

```text
v2_hazard_conditioned_functional_ugnn/
├── README.md
├── config.py
├── schemas.py
├── model.py
├── constants.py
│
├── data/
│   ├── __init__.py
│   ├── datasets.py
│   ├── batch_collators.py
│   ├── benchmark_adapters.py
│   ├── graph_loaders.py
│   └── feature_builders.py
│
├── memory/
│   ├── __init__.py
│   ├── urban_memory_encoder.py
│   ├── lag_memory_encoder.py
│   ├── recurrent_memory_encoder.py
│   ├── temporal_attention.py
│   ├── transformer_encoder.py
│   └── hazard_queried_memory.py
│
├── hazard/
│   ├── __init__.py
│   ├── hazard_query_encoder.py
│   ├── hazard_embeddings.py
│   ├── cross_attention.py
│   └── scenario_encoder.py
│
├── relations/
│   ├── __init__.py
│   ├── relation_types.py
│   ├── relation_registry.py
│   ├── hazard_relation_priors.py
│   └── relation_validation.py
│
├── functional_message_passing/
│   ├── __init__.py
│   ├── layer.py
│   ├── relation_family_gate.py
│   ├── edge_attention.py
│   ├── relation_transforms.py
│   ├── message_builders.py
│   ├── aggregators.py
│   ├── edge_normalization.py
│   └── ablations.py
│
├── heads/
│   ├── __init__.py
│   ├── prediction_heads.py
│   ├── uncertainty_heads.py
│   └── reporting_bias_heads.py
│
├── explanations/
│   ├── __init__.py
│   ├── pathway_exporter.py
│   ├── relation_gate_exporter.py
│   ├── attention_exporter.py
│   ├── counterfactuals.py
│   └── explanation_schemas.py
│
├── training/
│   ├── __init__.py
│   ├── losses.py
│   ├── trainer.py
│   ├── evaluation.py
│   └── callbacks.py
│
├── inference/
│   ├── __init__.py
│   ├── predictor.py
│   ├── explanation_api.py
│   └── ui_payloads.py
│
├── experiments/
│   ├── __init__.py
│   ├── ablations.py
│   ├── configs.py
│   └── registry.py
│
├── tests/
│   ├── test_schemas.py
│   ├── test_memory_encoder.py
│   ├── test_hazard_encoder.py
│   ├── test_relation_family_gate.py
│   ├── test_functional_message_passing.py
│   └── test_prediction_shapes.py
│
└── docs/
    ├── architecture_north_star.md
    ├── module_interfaces.md
    ├── ablation_ladder.md
    ├── relation_family_ontology.md
    └── ui_integration_contract.md
```

---

## 7. Folder responsibilities

### 7.1 `data/`

Owns the bridge between raw data, benchmark artifacts, and V2 model contracts.

Should handle:

```text
dataset objects
batch collation
benchmark output adapters
graph loading
feature tensor construction
```

Should not contain:

```text
model layers
message passing
training loops
metric reports
```

---

### 7.2 `memory/`

Owns urban memory.

Should handle:

```text
lag/rolling memory
recurrent memory
temporal attention
transformer memory
hazard-queried memory
```

The first implementation should start with lag/rolling memory before learned memory.

---

### 7.3 `hazard/`

Owns hazard and scenario encoding.

Should handle:

```text
hazard embeddings
hazard query vectors
scenario context
cross-attention primitives
```

Should not own relation gates or message passing.

---

### 7.4 `relations/`

Owns relation ontology and validation.

Should define:

```text
relation family IDs
relation registry
hazard-relation priors
relation validation utilities
```

This folder protects the model from relation-name drift.

---

### 7.5 `functional_message_passing/`

Owns the core custom message-passing contribution.

Should handle:

```text
relation-family gates
edge attention
relation-specific transforms
message construction
aggregation
edge normalization
message-passing ablations
```

This folder is the intellectual center of V2.

---

### 7.6 `heads/`

Owns model outputs.

Should handle:

```text
prediction heads
uncertainty heads
reporting-bias heads
```

The first implementation should start with a simple prediction head.

---

### 7.7 `explanations/`

Owns structured explanation exports.

Should handle:

```text
relation gate export
attention export
pathway summaries
counterfactuals
explanation schemas
```

This folder should eventually produce stable payloads for research analysis and UI integration.

---

### 7.8 `training/`

Owns optimization and evaluation loops.

Should handle:

```text
losses
trainer
evaluation
callbacks
checkpointing
early stopping
```

Should not define the architecture.

---

### 7.9 `inference/`

Owns model serving and UI-facing prediction logic.

Should handle:

```text
predictor wrappers
explanation API
UI payload conversion
```

The UI should not need to understand training code or benchmark reports.

---

### 7.10 `experiments/`

Owns named experimental variants and ablation configurations.

Should handle:

```text
experiment registry
named ablation configs
reproducible configuration presets
```

---

### 7.11 `tests/`

Owns shape and contract tests.

Initial tests should focus on:

```text
schema validity
memory encoder output shapes
hazard encoder output shapes
relation gate dimensions
functional message-passing output shapes
prediction head output shapes
explanation schema validity
```

---

### 7.12 `docs/`

Owns conceptual and interface documentation.

Core docs:

```text
architecture_north_star.md
module_interfaces.md
ablation_ladder.md
relation_family_ontology.md
ui_integration_contract.md
```

---

## 8. Minimal V2.0 scope

The first working V2.0 implementation should be disciplined.

It should include:

```text
nodes:
  tract-month or CD-month nodes

features:
  same feature parity as the benchmark where possible

memory:
  lag/rolling urban memory encoder

hazard:
  simple learned hazard embedding

relations:
  spatial adjacency
  kNN
  random/placebo
  temporal memory or history features
  simple exposure/protection relations if available

message passing:
  relation-family gate
  simple edge attention or uniform attention
  relation-specific transforms
  mean or sum aggregation

heads:
  count/regression prediction head

explanations:
  relation gate export
  edge attention export
```

It should not start with:

```text
full transformer memory
full heterogeneous graph
reporting-bias head as a required component
large uncertainty system
complex counterfactual engine
production UI deployment
all possible hazard types
```

Those belong in later V2.x stages.

---

## 9. Required ablation ladder

Every V2 version should preserve an ablation ladder.

Minimum ladder:

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

Each ambitious module must earn its complexity.

A module is useful if it improves at least one of:

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

## 10. What success looks like

The strongest success case is:

```text
lower MAE
+ better top-k ranking
+ real topology beats placebo topology
+ relation gates shift by hazard in plausible ways
+ memory retrieval differs by hazard
+ explanation exports identify coherent pathways
+ uncertainty estimates are useful
```

Possible useful partial successes:

```text
the model improves top-k ranking but not MAE
the model gives stable hazard-specific relation gates
hazard-queried memory retrieves meaningful history
explanations reveal relation-family differences
uncertainty identifies sparse or rare hazard cases
```

A negative result is also useful if it clarifies that:

```text
features dominate topology
history dominates graph structure
hazard conditioning does not yet add value
relation ontology needs refinement
```

---

## 11. Non-goals for V2.0

V2.0 should not try to solve every urban resilience problem at once.

Non-goals:

```text
full Québec-wide heterogeneous graph from day one
all possible hazard types
perfect infrastructure modeling
complete causal inference
production UI deployment before validation
massive transformer architecture before simple controls
custom low-level GPU kernels
custom optimizer
```

The model should grow from a controlled scientific core.

---

## 12. Relationship to UI integration

This package is not a UI package, but it should be designed so that UI integration is possible later.

Eventually, inference should expose stable outputs such as:

```text
prediction:
  risk_score
  expected_burden
  target_horizon
  uncertainty

explanation:
  top_relation_families
  top_edges
  top_neighbor_nodes
  top_history_windows
  pathway_scores
  counterfactual_hazard_comparison
```

The UI should consume structured outputs.

It should not need to know:

```text
training loop internals
benchmark report generation
ablation registry details
tensor-shape implementation details
```

---

## 13. Recommended reading order

For new collaborators, read:

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
