# Urban Resilience Models

This folder contains the independent model families developed after the controlled benchmark work.

The benchmark folder answers:

```text
Which baselines matter?
Do static vulnerability indices, history, tabular ML, and graph controls justify richer models?
```

This folder answers:

```text
What reusable research and deployment-ready model families are we building next?
```

The main principle is simple:

```text
The benchmark justifies the model.
The model should not be structurally trapped inside the benchmark.
```

---

## 1. Purpose

`urban_resilience_models/` is the home for urban resilience model families that may eventually be reused by researchers, ML engineers, and dashboard/UI systems.

It is intentionally separated from:

```text
urban_graph_benchmark/
```

because benchmark code and model-family code have different roles.

```text
urban_graph_benchmark/
  controlled experiments, baselines, ablations, metrics, reports

urban_resilience_models/
  reusable model architectures, interfaces, inference APIs, explanation payloads
```

The benchmark remains the evaluation laboratory.
This folder is where the actual model families live.

---

## 2. Scientific lineage

The current model family starts at Version 2 because Version 1 already exists conceptually in the benchmark work.

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
  pathway explanations
```

Version 1 proved that a graph model must be compared against serious controls.

Version 2 begins the actual custom research architecture.

---

## 3. Current model family

The current active model family is:

```text
v2_hazard_conditioned_functional_ugnn/
```

Full name:

```text
Hazard-Conditioned Functional Urban Graph Neural Network
```

Compact research framing:

```text
Hazard-Conditioned Functional Message Passing
```

This model family is built around the idea that:

```text
The current hazard decides which urban relations matter,
the city’s past stress history informs node state,
and message passing becomes a functional explanation of risk pathways.
```

It is not meant to be just another generic GNN applied to city data.

It is meant to become an interpretable urban reasoning system that combines:

```text
urban memory
hazard queries
functional relation families
hazard-conditioned gates
edge-level attention
prediction heads
uncertainty
pathway explanations
```

---

## 4. Repository structure

Expected high-level structure:

```text
urban_resilience_models/
├── README.md
├── lineage.md
├── model_family_manifest.md
│
├── common/
│   ├── __init__.py
│   ├── schemas.py
│   ├── graph_contracts.py
│   ├── feature_contracts.py
│   ├── prediction_contracts.py
│   ├── explanation_contracts.py
│   └── io.py
│
├── v2_hazard_conditioned_functional_ugnn/
│   ├── README.md
│   ├── config.py
│   ├── schemas.py
│   ├── model.py
│   ├── constants.py
│   ├── data/
│   ├── memory/
│   ├── hazard/
│   ├── relations/
│   ├── functional_message_passing/
│   ├── heads/
│   ├── explanations/
│   ├── training/
│   ├── inference/
│   ├── experiments/
│   ├── tests/
│   └── docs/
│
├── model_cards/
│   └── v2_hazard_conditioned_functional_ugnn.md
│
└── artifacts/
    └── README.md
```

---

## 5. Design philosophy

### 5.1 Separate models from benchmarks

Model code should not be deeply coupled to benchmark-specific paths, reports, or experiment scripts.

Allowed:

```text
data/benchmark_adapters.py can know how to read benchmark outputs.
```

Not allowed:

```text
model.py should know benchmark output paths.
functional message-passing layers should know benchmark column names.
prediction heads should know report file names.
```

The clean dependency direction is:

```text
urban_resilience_models/common
        ↓
urban_resilience_models/v2_hazard_conditioned_functional_ugnn
        ↓
benchmark adapters / training scripts / experiments
        ↓
UI or deployment wrappers
```

---

### 5.2 Contracts before complexity

This package should be interface-first.

Before building complex layers, the model family should define stable contracts for:

```text
node features
temporal history
hazard context
relation families
edge attributes
prediction outputs
explanation outputs
UI payloads
```

The exact implementation can evolve.

The contracts should remain stable.

---

### 5.3 Controlled before complex

Every ambitious module needs an ablation.

A module is not justified because it is elegant or advanced.

A module is justified if it improves at least one of:

```text
predictive performance
top-k prioritization
hazard-specific behavior
interpretability
stability
uncertainty calibration
scientific understanding
```

The model family should avoid architecture soup.

---

### 5.4 Hazard-conditioned, not hazard-blind

The model should not pass messages the same way for every hazard.

A flood query and a heat query should activate different relation pathways.

Examples:

```text
Flood:
  hydrological exposure
  drainage dependency
  low elevation
  water/drainage memory
  spatial propagation

Heat:
  heat island exposure
  canopy/protection
  impervious surface
  vulnerable population
  summer memory
```

---

### 5.5 Functional, not merely spatial

The graph should not only represent geographic closeness.

It should eventually represent urban mechanisms such as:

```text
exposure
protection
access
dependency
cascade
memory
reporting similarity
administrative adjacency
```

The long-term goal is a functional urban graph, not only a spatial adjacency graph.

---

### 5.6 Interpretable by construction

Interpretability should not be added only after training.

The model should be designed to export:

```text
relation-family gates
edge attention weights
temporal attention weights
pathway scores
counterfactual hazard comparisons
uncertainty summaries
```

The model should not only say:

```text
this area is high risk
```

It should eventually say:

```text
this area is high risk under flood because drainage-related relations,
recent water complaints, and hydrological exposure pathways were activated.
```

---

## 6. Relationship to future UI and ML engineering

This folder is designed so that future ML engineers can reuse the model family without treating it as a benchmark artifact.

The eventual UI-facing layer should be able to consume stable outputs such as:

```text
prediction:
  risk_score
  expected_burden
  target_horizon
  uncertainty

explanation:
  top_relation_families
  top_edges
  top_history_windows
  pathway_scores
  counterfactual_hazard_comparison
```

The UI should not need to understand training loops, benchmark reports, or internal ablation logic.

The model package should expose clean inference and explanation APIs.

---

## 7. Active development status

Current status:

```text
skeleton / architecture design / early implementation
```

This package is not yet production-ready.

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
```

Only after those are stable should the project move into full PyTorch implementation.

---

## 8. Immediate next files

Recommended first files to implement:

```text
urban_resilience_models/README.md
urban_resilience_models/lineage.md
urban_resilience_models/model_family_manifest.md
urban_resilience_models/v2_hazard_conditioned_functional_ugnn/README.md
urban_resilience_models/v2_hazard_conditioned_functional_ugnn/docs/architecture_north_star.md
urban_resilience_models/v2_hazard_conditioned_functional_ugnn/docs/relation_family_ontology.md
urban_resilience_models/v2_hazard_conditioned_functional_ugnn/docs/module_interfaces.md
urban_resilience_models/v2_hazard_conditioned_functional_ugnn/docs/ablation_ladder.md
urban_resilience_models/v2_hazard_conditioned_functional_ugnn/constants.py
urban_resilience_models/v2_hazard_conditioned_functional_ugnn/schemas.py
```

The implementation order should be:

```text
identity → lineage → ontology → interfaces → constants → schemas → config → modules
```

---

## 9. North-star statement

This package exists so that a future reader can say:

> This is not just a graph neural network applied to city data. It is a hazard-conditioned urban reasoning system. It remembers past stress, receives a hazard query, activates functional relation pathways, passes messages through the relevant parts of the city, predicts future burden, and exports interpretable pathway evidence.

Everything in this package should serve that statement.

If a file does not help with one of these responsibilities, it probably belongs elsewhere.
