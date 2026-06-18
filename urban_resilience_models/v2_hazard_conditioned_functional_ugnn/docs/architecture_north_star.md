# Architecture North Star — V2 Hazard-Conditioned Functional UGNN

**Model family:** `v2_hazard_conditioned_functional_ugnn`  
**Intended location:** `urban_resilience_models/v2_hazard_conditioned_functional_ugnn/docs/architecture_north_star.md`  
**Status:** North-star architecture document, not a frozen implementation specification  
**Scope:** This document explains why each core file/folder exists, what it should own, what it must not own, and how the model family should evolve without losing its scientific identity.

---

## 1. One-sentence identity

`v2_hazard_conditioned_functional_ugnn` is an urban resilience graph model family where **urban memory**, **hazard queries**, and **functional relation pathways** jointly determine how information moves through an urban graph.

The model is not just “a graph neural network for city data.”

It is specifically:

```text
hazard-conditioned functional message passing
over urban units with temporal memory
and interpretable relation pathways
```

The central idea is:

```text
The current hazard decides which urban relations matter,
the city’s past stress history informs node state,
and message passing becomes a functional explanation of risk pathways.
```

---

## 2. Scientific lineage

This model starts at Version 2 because Version 1 already exists conceptually in the benchmark work.

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
  pathway export
```

The benchmark demonstrated that a credible graph claim requires comparison against static vulnerability indices, history baselines, calibrated indices, tabular ML with the same features, no-edge neural controls, random/placebo graph controls, kNN graph controls, and real graph topology.

This V2 architecture is the next step after that benchmark foundation. It should not repeat G1/G1.5 as another generic tract GraphSAGE. It should implement the first model whose novelty is the **hazard-conditioned routing of urban information through functional relation families**.

---

## 3. Core research question

The model family should answer:

> Can an urban graph model learn different risk pathways for different hazards, while remaining predictive, interpretable, and controlled against non-graph and placebo-graph baselines?

This expands into five operational questions:

1. **Urban memory:** Which past events, complaints, seasons, or stress patterns matter for the current prediction?
2. **Hazard conditioning:** Does the model behave differently for flood, heat, outage, road disruption, or civil-security event queries?
3. **Functional relations:** Which relation families are activated under each hazard?
4. **Spatial/message-passing value:** Does real topology help beyond no-edge, tabular, random-edge, and generic kNN controls?
5. **Interpretability:** Can the model export relation gates, edge attention, pathway summaries, and counterfactual hazard explanations?

---

## 4. Design principles

### 4.1 Controlled before complex

Every new module must have a matching ablation. A module is not justified by elegance alone. It is justified if it improves predictive performance, explanation quality, stability, or scientific interpretability over a simpler control.

### 4.2 Hazard-conditioned, not hazard-blind

The model must not pass the same messages for every hazard. A flood query and a heat query should activate different relation families.

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

### 4.3 Functional, not merely spatial

The graph should not only represent geographic closeness. It should represent urban mechanisms:

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

### 4.4 Interpretable by construction

Interpretability should not be added as an afterthought. Relation gates, edge attention, pathway scores, and counterfactual exports should be designed into the forward pass and output contracts.

### 4.5 Implementation can evolve, contracts must remain stable

The exact neural layer may change. GRU can become LSTM, LSTM can become transformer, mean aggregation can become attention, and single-hazard conditioning can become multi-hazard scenario encoding. But the conceptual contracts should remain stable:

```text
node state
history sequence
hazard context
edge index
edge type / relation family
edge attributes
relation gate outputs
attention outputs
prediction outputs
explanation outputs
```

---

## 5. High-level architecture

The north-star flow is:

```text
raw benchmark/model batch
        |
        v
data adapters and schema validation
        |
        v
static node features + temporal history + graph edges + hazard/scenario input
        |
        v
Urban Memory Encoder
        |
        v
Hazard Query Encoder
        |
        v
Hazard-Queried Memory
        |
        v
Relation-Family Gate
        |
        v
Edge Attention
        |
        v
Functional Message Passing
        |
        v
Prediction / uncertainty / reporting heads
        |
        v
Pathway explanation export
```

A concise computational form:

```text
node representation =
    static urban features
  + encoded urban memory
  + hazard-queried historical context

message along edge =
    relation-specific transform(node state)
  × hazard-conditioned relation gate
  × edge-level attention
  × optional edge normalization

prediction =
    future burden / risk / ranking score
  + optional uncertainty
  + optional reporting-propensity decomposition
```

This equation is conceptual. It should guide the implementation, not freeze the exact tensor operations.

---

## 6. Package structure

The model family should live under:

```text
urban_resilience_models/
├── common/
├── v2_hazard_conditioned_functional_ugnn/
├── model_cards/
└── artifacts/
```

The V2 package should contain:

```text
v2_hazard_conditioned_functional_ugnn/
├── README.md
├── config.py
├── schemas.py
├── model.py
├── constants.py
├── data/
├── memory/
├── hazard/
├── relations/
├── functional_message_passing/
├── heads/
├── explanations/
├── training/
├── inference/
├── experiments/
├── tests/
└── docs/
```

The rest of this document explains what each core file is for.

---

# 7. Top-level V2 files

## 7.1 `README.md`

### Purpose

Explain the model family to a new researcher or collaborator in five minutes.

### Should contain

- Model identity.
- Relation to Version 1 benchmark.
- Main modules.
- Minimal usage example.
- Current implementation status.
- List of supported hazards.
- List of supported relation families.
- Pointers to `architecture_north_star.md`, `module_interfaces.md`, and `ablation_ladder.md`.

### Should not contain

- Long mathematical derivations.
- Full experiment results.
- Deep implementation notes that belong in module docs.

---

## 7.2 `config.py`

### Purpose

Define typed configuration objects that control model construction, data loading, training, inference, and ablations.

This file is the single source of truth for model-family options.

### Should contain

Configuration dataclasses or equivalent typed structures for:

```text
ModelConfig
MemoryConfig
HazardConfig
RelationConfig
FunctionalMessagePassingConfig
PredictionHeadConfig
UncertaintyConfig
ReportingBiasConfig
TrainingConfig
ExperimentConfig
```

### Important design rule

`config.py` should describe what is configurable, not implement the model.

For example, it may define:

```text
memory_encoder_type = lag | gru | lstm | transformer
relation_gate_type = hazard_mlp | prior_regularized | none
message_passing_layers = 1, 2, 3, ...
```

but it should not contain the GRU or relation gate implementation.

### What it must protect against

Configuration drift. If every script starts inventing its own argument names, the model family becomes impossible to reproduce. This file prevents that.

---

## 7.3 `schemas.py`

### Purpose

Define V2-specific data schemas and batch objects.

This is where the model’s internal input/output structure becomes explicit.

### Should define

Conceptual batch objects such as:

```text
UrbanGraphBatch
NodeFeatureBatch
TemporalHistoryBatch
HazardContextBatch
RelationEdgeBatch
PredictionBatch
ExplanationBatch
```

### Expected fields

A V2 batch should be able to represent:

```text
node_ids
node_features
history_sequences
history_masks
hazard_ids
hazard_features
scenario_features
edge_index
edge_relation_type
edge_attributes
edge_time_index
targets
target_masks
split labels
metadata
```

### Should not contain

- Heavy data loading.
- Model layers.
- Training loops.

### Why this matters

The V2 model will be modular. Without explicit schemas, different modules will quietly disagree about tensor shapes and meanings.

---

## 7.4 `constants.py`

### Purpose

Centralize stable names and controlled vocabularies.

### Should contain

```text
hazard IDs
relation family IDs
default feature names
default target names
canonical split names
known prediction head names
known explanation field names
```

### Examples

```text
HAZARD_FLOOD
HAZARD_HEAT
REL_SPATIAL_ADJACENCY
REL_HYDROLOGICAL_EXPOSURE
REL_DRAINAGE_DEPENDENCY
REL_HEAT_EXPOSURE
REL_CANOPY_PROTECTION
REL_SERVICE_ACCESS
REL_TEMPORAL_MEMORY
```

### Should not contain

Runtime configuration or experiment-specific parameters.

---

## 7.5 `model.py`

### Purpose

Define the top-level V2 model class.

This file orchestrates the full forward pass.

### It should own

The high-level wiring:

```text
input projection
memory encoder call
hazard encoder call
hazard-queried memory call
functional message-passing stack
prediction head call
optional uncertainty/reporting heads
optional explanation payload assembly
```

### It should not own

The detailed logic of:

```text
GRU/LSTM/transformer memory
hazard embeddings
relation-family gates
edge attention
message aggregation
prediction losses
explanation export formatting
```

Those belong in submodules.

### Ideal conceptual forward pass

```text
1. Validate batch contract.
2. Encode static node features.
3. Encode urban memory from temporal histories.
4. Encode hazard/scenario context.
5. Query memory with hazard context.
6. Build initial node state.
7. For each message-passing layer:
     a. compute relation-family gates
     b. compute edge attention
     c. build relation-specific messages
     d. aggregate messages
     e. update node states
     f. store explanation hooks if requested
8. Predict future burden/risk.
9. Optionally predict uncertainty.
10. Optionally decompose reporting propensity.
11. Return predictions and structured explanation traces.
```

### North-star principle

`model.py` should read like an architecture diagram in code. If it becomes 1,000 lines, the package has lost modular discipline.

---

# 8. `data/` — data and benchmark adapters

The `data/` folder bridges existing benchmark artifacts and the V2 model contracts.

It is core because V2 must consume the outputs of the benchmark suite without polluting model modules with file-specific logic.

## 8.1 `data/datasets.py`

### Purpose

Define dataset classes that expose model-ready batches.

### Should own

- Loading examples.
- Indexing samples.
- Returning graph/time/hazard batches.
- Supporting train/validation/test splits.
- Optional filtering by hazard, target, geography, or time period.

### Should not own

- Neural model logic.
- Metric calculation.
- Relation ontology definitions.

### Important future requirement

The dataset should eventually support multiple resolutions:

```text
Montréal tract-month
Québec CD-month
heterogeneous urban graph snapshot
event-centered scenario batch
```

But it should start with the simplest controlled version.

---

## 8.2 `data/batch_collators.py`

### Purpose

Convert dataset examples into batched tensors.

### Should own

- Padding temporal histories.
- Combining graph edges.
- Aligning node indices.
- Building masks.
- Creating mini-batches or full-batch graph inputs.
- Handling variable-length histories.

### Why it matters

Temporal graph batches are fragile. This file prevents batching logic from leaking into the trainer or model.

---

## 8.3 `data/benchmark_adapters.py`

### Purpose

Convert `urban_graph_benchmark` outputs into V2 data contracts.

### Should own

Adapters for files such as:

```text
cd_month_panel.parquet
cd_graph_nodes.parquet
cd_graph_edges_adjacency.parquet
cd_graph_edges_knn.parquet
cd_graph_edges_random_placebo.parquet
Montréal tract-month benchmark panel
G1/G1.5 graph artifacts
```

### Should answer

How does a benchmark table become a V2 `UrbanGraphBatch`?

### Should not own

Model-specific logic. It should only map data formats.

### Why this file is essential

The benchmark and the model package have different roles:

```text
urban_graph_benchmark:
  controlled experiments and baselines

urban_resilience_models:
  reusable research/deployable model families
```

This adapter is the bridge.

---

## 8.4 `data/graph_loaders.py`

### Purpose

Load node and edge tables into graph objects expected by the model.

### Should own

- Loading node files.
- Loading edge files.
- Validating node IDs.
- Validating edge endpoints.
- Mapping external IDs to integer indices.
- Filtering edge families.
- Adding/removing self-loops if configured.
- Constructing edge attributes.

### Should not own

Relation semantics. Those belong in `relations/`.

---

## 8.5 `data/feature_builders.py`

### Purpose

Build model-ready feature tensors from raw/static/temporal feature columns.

### Should own

- Feature selection.
- Normalization/scaling hooks.
- History windows.
- Hazard-specific feature groups.
- Missing-value handling interfaces.

### Should not own

The memory encoder itself. This file prepares inputs; `memory/` learns from them.

---

# 9. `memory/` — urban memory

The `memory/` folder is one of the core research modules.

Its central question:

> How should a node remember past urban stress, and how should the current hazard retrieve the relevant part of that memory?

Urban memory is justified because the benchmarks showed that history is a major signal. However, V2 should not assume learned memory is always better than lag/rolling features. It must test that.

---

## 9.1 `memory/urban_memory_encoder.py`

### Purpose

Define the abstract or coordinating memory encoder interface.

### Should own

- Common memory encoder contract.
- Dispatch to lag, recurrent, transformer, or attention-based encoders.
- Output shape and metadata standardization.

### Should expose

A stable interface like:

```text
encode(history_sequence, history_mask, optional_context) -> memory_state
```

### Should not own

All implementations. It should coordinate them.

### North-star output

The memory encoder should produce a node-level vector representing past urban stress:

```text
urban_memory_state[node, hidden_dim]
```

This vector should be usable by message passing and prediction heads.

---

## 9.2 `memory/lag_memory_encoder.py`

### Purpose

Represent handcrafted historical memory.

This is the simplest memory encoder and an essential ablation.

### Inputs

```text
lag_1
rolling_3
rolling_6
rolling_12
seasonal historical mean
hazard-specific rolling features
```

### Why it exists

It anchors learned memory against a transparent baseline. If the GRU/LSTM/transformer cannot beat this, the learned memory module is not justified.

### Should not become

A dump for all feature engineering. It should only encode memory features.

---

## 9.3 `memory/recurrent_memory_encoder.py`

### Purpose

Implement sequence-based learned memory with GRU/LSTM-style encoders.

### Inputs

```text
history_sequence[node, time, features]
history_mask[node, time]
```

### Outputs

```text
memory_state[node, hidden_dim]
optional temporal_hidden_states[node, time, hidden_dim]
```

### Implementation freedom

This file can use:

```text
GRU
LSTM
bidirectional recurrent encoders
small recurrent stacks
```

The north star does not freeze which one wins.

### Ablation role

The key experiment is:

```text
Does recurrent learned memory improve over lag/rolling memory?
```

---

## 9.4 `memory/temporal_attention.py`

### Purpose

Implement temporal attention over past months.

### Role

This module answers:

```text
Which past months matter for the current prediction?
```

It can be used with recurrent outputs or directly on projected history features.

### Should export

Temporal attention weights that can later feed explanation payloads.

### Important caution

Temporal attention is not a complete explanation by itself. It is one interpretability signal among several.

---

## 9.5 `memory/transformer_encoder.py`

### Purpose

Provide a transformer-style alternative to recurrent memory.

### Role

This is for richer temporal patterns when history windows become longer or more complex.

### Should support

- Positional/month encodings.
- Temporal masks.
- Optional causal constraints.
- Compact transformer blocks.

### Should not be used just because it sounds advanced

It must be tested against lag and recurrent memory.

---

## 9.6 `memory/hazard_queried_memory.py`

### Purpose

Retrieve hazard-relevant memory from a node’s past.

This is one of the most original modules.

### Central idea

A node should not have one generic memory for every hazard. A flood query should retrieve different historical signals than a heat query.

Conceptually:

```text
hazard query attends over temporal memory
```

Examples:

```text
Flood query:
  attends to water/drainage complaints
  spring months
  past flood events
  drainage stress

Heat query:
  attends to summer periods
  heat exposure
  canopy/protection context
  vulnerable population stress
```

### Outputs

```text
hazard_conditioned_memory[node, hidden_dim]
temporal_attention_weights[node, time]
optional memory_explanation_payload
```

### Why it matters

This module turns the model from:

```text
LSTM node encoder
```

into:

```text
hazard-queried urban memory
```

That is much more defensible as a custom research contribution.

---

# 10. `hazard/` — hazard and scenario encoding

The `hazard/` folder defines what the model knows about the current hazard or scenario.

Its central question:

> What is the model being asked to forecast risk under?

---

## 10.1 `hazard/hazard_query_encoder.py`

### Purpose

Produce a hazard context vector that conditions memory and message passing.

### Inputs may include

```text
hazard ID
hazard type embedding
scenario attributes
month/season
event metadata
weather/hydro indicators
```

### Outputs

```text
hazard_context[batch_or_node, hidden_dim]
```

### Should not own

Relation gates or edge attention. It only encodes the hazard query.

---

## 10.2 `hazard/hazard_embeddings.py`

### Purpose

Define and manage learned or fixed hazard embeddings.

### Examples

```text
flood
heat
outage
road disruption
civil-security event
all-hazard
```

### Implementation freedom

Embeddings can start as learned lookup vectors. Later they can incorporate textual descriptions, scenario features, or hazard ontology priors.

---

## 10.3 `hazard/cross_attention.py`

### Purpose

Provide reusable cross-attention primitives when a hazard query attends over another sequence or set.

### Possible uses

```text
hazard query attends over temporal memory
hazard query attends over relation families
hazard query attends over scenario tokens
```

### Design caution

This module should remain generic. If the attention is specifically about memory, the higher-level logic belongs in `memory/hazard_queried_memory.py`.

---

## 10.4 `hazard/scenario_encoder.py`

### Purpose

Encode dynamic scenario context beyond hazard type.

### Examples

```text
month/season
precipitation
temperature anomaly
river level
snowmelt proxy
civil-security event severity
return period
forecast horizon
```

### Why it matters

Hazard type alone is often too coarse. A flood during heavy rain and high river level is different from a generic flood query.

---

# 11. `relations/` — relation ontology and validation

The `relations/` folder defines the semantic relation system used by the graph.

This is core because the model is not simply spatial. It is functional.

---

## 11.1 `relations/relation_types.py`

### Purpose

Define relation family identifiers.

### Examples

```text
spatial_adjacency
centroid_knn
random_placebo
temporal_memory
hydrological_exposure
drainage_dependency
heat_exposure
canopy_protection
service_access
infrastructure_dependency
reporting_similarity
socioeconomic_similarity
```

### Should include

- Stable IDs.
- Human-readable names.
- Short descriptions.
- Whether relation is directed/undirected.
- Whether relation is real, derived, or placebo/control.

---

## 11.2 `relations/relation_registry.py`

### Purpose

Central registry of all relation families.

### Should answer

For each relation:

```text
What does it mean?
What hazards is it relevant to?
What edge attributes are expected?
Is it allowed in training?
Is it allowed in explanation?
Is it a control relation?
```

### Why it matters

Without a registry, relation IDs will drift across data builders, message-passing layers, and explanation exports.

---

## 11.3 `relations/hazard_relation_priors.py`

### Purpose

Define optional prior expectations about which relations should matter for which hazards.

### Examples

```text
flood:
  high prior for hydrological_exposure
  high prior for drainage_dependency
  medium prior for spatial_adjacency
  low prior for canopy_protection

heat:
  high prior for heat_exposure
  high prior for canopy_protection
  medium prior for social_vulnerability
  low prior for drainage_dependency
```

### Important rule

These priors should guide or regularize, not hard-code, the model. The model must still be able to learn surprising relationships.

---

## 11.4 `relations/relation_validation.py`

### Purpose

Validate edge tables against relation contracts.

### Should check

```text
valid relation family IDs
valid source/target node IDs
required edge attributes
directedness convention
no missing endpoints
no invalid placeholders
control graph labels
```

### Why it matters

A graph model can silently become invalid if edge types are mislabelled.

---

# 12. `functional_message_passing/` — core model contribution

This is the intellectual center of V2.

Its central question:

> Given node states, a hazard context, and typed urban relations, how should the model route messages through the city?

This folder deserves its own subpackage because the contribution is not a single layer. It is a family of interacting mechanisms.

---

## 12.1 `functional_message_passing/layer.py`

### Purpose

Define the main functional message-passing layer.

This file coordinates:

```text
relation-family gate
edge attention
relation-specific transforms
message construction
aggregation
node update
explanation trace collection
```

### Should own

The high-level layer forward pass.

### Should not own

The internal implementation of each subcomponent. Those belong in sibling files.

### Conceptual forward pass

```text
1. Receive node_state, edge_index, edge_type, edge_attr, hazard_context.
2. Compute relation-family gate values.
3. Compute edge attention weights.
4. Transform source node states through relation-specific transforms.
5. Build messages.
6. Normalize or weight messages.
7. Aggregate messages at target nodes.
8. Update node representations.
9. Return updated node states and optional explanation traces.
```

### What should be configurable

```text
number of relation families
hidden dimension
aggregation type
gate type
attention type
residual connections
normalization
dropout
whether to return explanations
```

### What should remain flexible

The exact mathematical form of the layer should remain open. The north star specifies the responsibilities, not the final equation.

---

## 12.2 `functional_message_passing/relation_family_gate.py`

### Purpose

Compute hazard-conditioned weights for relation families.

### Central question

```text
Which types of urban mechanisms should be active under this hazard?
```

### Inputs

```text
hazard_context
optional node_state
optional scenario_context
optional relation priors
```

### Outputs

```text
relation_gate[hazard or node, relation_family]
```

### Example behavior

```text
Flood:
  hydrological_exposure = high
  drainage_dependency = high
  heat_exposure = low

Heat:
  heat_exposure = high
  canopy_protection = high
  drainage_dependency = low
```

### Explanation role

These gates are one of the primary explanation objects.

They answer:

```text
Which relation families drove the prediction?
```

### Must not do

It should not compute edge-specific attention. That belongs to `edge_attention.py`.

---

## 12.3 `functional_message_passing/edge_attention.py`

### Purpose

Compute attention over specific edges within active relation families.

### Central question

```text
Which neighboring nodes matter under this hazard and relation family?
```

### Inputs

```text
source node state
target node state
edge attributes
relation family
hazard context
relation gate value
```

### Outputs

```text
edge_attention[edge]
optional attention grouped by relation family
```

### Explanation role

This module provides edge-level interpretability.

It answers:

```text
Which exact neighboring tract, infrastructure node, exposure polygon, or service node mattered?
```

### Important distinction

Relation gates answer:

```text
Which mechanism type mattered?
```

Edge attention answers:

```text
Which specific connection mattered?
```

Both are needed.

---

## 12.4 `functional_message_passing/relation_transforms.py`

### Purpose

Define relation-specific transformations of source node states.

### Central idea

Messages through different relation families should not all use the same transformation.

For example, hydrological exposure messages, service access messages, temporal memory messages, and canopy protection messages may require different learned transformations.

### Implementation options

This file may eventually support:

```text
one linear transform per relation family
basis decomposition
shared transform plus relation embedding
low-rank relation transforms
typed MLPs
```

The north star does not freeze which version is best.

---

## 12.5 `functional_message_passing/message_builders.py`

### Purpose

Construct edge-level messages from transformed node states, edge attributes, gates, and attention.

### Conceptual message

```text
message =
  relation_transform(source_state)
  × relation_gate
  × edge_attention
  × edge_weight_or_normalization
```

### Should own

- Combining source state with edge attributes.
- Applying relation gates.
- Applying attention weights.
- Optional edge feature modulation.

### Should not own

Aggregation into target nodes. That belongs to `aggregators.py`.

---

## 12.6 `functional_message_passing/aggregators.py`

### Purpose

Aggregate incoming messages at target nodes.

### Options

```text
sum
mean
degree-normalized sum
attention-weighted sum
relation-wise aggregation
multi-relation concatenation
```

### Important role

Different aggregation choices are ablations. The first implementation can be simple, but the file should allow future expansion.

---

## 12.7 `functional_message_passing/edge_normalization.py`

### Purpose

Handle graph normalization and edge weighting.

### Should own

- Degree normalization.
- Edge-weight clipping.
- Distance-based weighting.
- Relation-wise normalization.
- Optional self-loop conventions.

### Why it matters

Some graph gains can come from smoothing/normalization rather than meaningful topology. This file makes that explicit.

---

## 12.8 `functional_message_passing/ablations.py`

### Purpose

Define controlled variants of the message-passing layer.

### Examples

```text
no relation gates
uniform relation gates
hazard-blind gates
no edge attention
uniform edge attention
shared relation transform
random relation labels
no graph / identity edges
```

### Why it matters

This architecture has many moving parts. Ablations prevent the model from becoming untestable.

---

# 13. `heads/` — prediction, uncertainty, reporting bias

The `heads/` folder converts learned node states into decision outputs.

---

## 13.1 `heads/prediction_heads.py`

### Purpose

Predict future burden, risk score, count intensity, or ranking score.

### Possible outputs

```text
future 311 count
future event burden
hazard-specific risk score
top-k prioritization score
multi-horizon predictions
```

### Should support

- Regression/count outputs.
- Optional multi-task heads.
- Optional horizon-specific heads.

### Should not own

Uncertainty intervals or reporting decomposition unless those are explicitly part of a specialized head.

---

## 13.2 `heads/uncertainty_heads.py`

### Purpose

Estimate uncertainty around predictions.

### Possible approaches

```text
quantile outputs
variance head
ensemble-compatible outputs
Monte Carlo dropout-compatible hooks
conformal interval wrappers
```

### North-star role

For municipal decision support, it is not enough to say high risk. The model should eventually say high risk with high confidence, or moderate risk with high uncertainty because this hazard-node combination is rare.

---

## 13.3 `heads/reporting_bias_heads.py`

### Purpose

Model observed reports as a mixture of latent disruption and reporting propensity.

### Motivation

311 calls are not pure physical disruption. They are observed reports.

A conceptual decomposition is:

```text
observed burden = latent disruption × reporting propensity
```

### Potential outputs

```text
latent disruption score
reporting propensity score
observed report prediction
```

### Important warning

This is an advanced module. It should not be forced into the first V2 implementation unless the data supports it. But the architecture should leave room for it.

---

# 14. `explanations/` — pathway exports

The `explanations/` folder converts model internals into structured, interpretable outputs.

This is core enough to design early, but the first implementation can be minimal.

---

## 14.1 `explanations/pathway_exporter.py`

### Purpose

Aggregate relation gates and edge attention into human-readable pathway summaries.

### Should answer

```text
Which relation families drove this prediction?
Which edges mattered most?
Which past time periods mattered?
Which hazard-specific pathway was activated?
```

### Output examples

```text
top_relation_families
top_edges
top_neighbor_nodes
top_history_months
pathway_scores
```

---

## 14.2 `explanations/relation_gate_exporter.py`

### Purpose

Export relation gate values in a stable schema.

### Why separate from `pathway_exporter.py`

Relation gates are core scientific evidence. They deserve a direct export independent of higher-level formatting.

---

## 14.3 `explanations/attention_exporter.py`

### Purpose

Export edge and temporal attention weights.

### Warning

Attention weights must be interpreted carefully. They are useful diagnostic signals, not complete causal explanations.

---

## 14.4 `explanations/counterfactuals.py`

### Purpose

Support counterfactual explanation experiments.

### Examples

```text
same node, flood query → heat query
remove drainage relations
remove canopy relations
replace real edges with random edges
remove past history
```

### Why it matters

Counterfactual and ablation-based explanations are more convincing than raw attention weights.

---

## 14.5 `explanations/explanation_schemas.py`

### Purpose

Define stable explanation output contracts.

### Should include

```text
prediction ID
node ID
hazard ID
target horizon
relation gates
edge attention summary
temporal attention summary
counterfactual deltas
uncertainty summary
```

---

# 15. `training/` — training and evaluation

Training is important, but it should not define the architecture.

## 15.1 `training/losses.py`

### Purpose

Define losses used by V2.

Examples:

```text
MSE
Huber
Poisson negative log likelihood
quantile loss
ranking loss
multi-task loss
regularization for relation gates
```

Relation-gate regularization may eventually encourage sparse or interpretable hazard-specific pathway activation.

## 15.2 `training/trainer.py`

### Purpose

Own the training loop.

Should handle:

```text
forward pass
loss computation
backpropagation
early stopping
checkpointing
logging
validation
```

It should not contain model architecture logic.

## 15.3 `training/evaluation.py`

### Purpose

Evaluate predictions using the same metric families as the benchmark.

Metrics should include:

```text
MAE
RMSE
Poisson deviance
Spearman
Kendall
NDCG@k
top-k overlap
calibration/uncertainty metrics when available
```

## 15.4 `training/callbacks.py`

### Purpose

Optional training hooks:

```text
early stopping
checkpoint saving
metric logging
explanation snapshots
learning-rate scheduling
```

---

# 16. `inference/` — prediction and UI integration

Inference can stay relatively thin at first.

## 16.1 `inference/predictor.py`

Load a trained model and produce predictions.

## 16.2 `inference/explanation_api.py`

Expose explanations in a stable API.

## 16.3 `inference/ui_payloads.py`

Convert predictions and explanations into payloads usable by a dashboard.

This should not drive the research model. It should translate outputs.

---

# 17. `experiments/` — ablations and reproducibility

## 17.1 `experiments/ablations.py`

Define named ablation variants.

Examples:

```text
no_memory
lag_memory_only
recurrent_memory
no_hazard_conditioning
hazard_embedding_only
no_relation_gates
no_edge_attention
random_edges
knn_edges
real_adjacency
```

## 17.2 `experiments/configs.py`

Store reusable experiment configurations.

## 17.3 `experiments/registry.py`

Map experiment names to config objects and expected outputs.

The registry should make experiments reproducible by name.

---

# 18. `tests/` and `docs/`

These are important but should remain lightweight at first.

## 18.1 `tests/`

Tests should focus on shape contracts and schema validity:

```text
batch construction
memory encoder output shape
hazard encoder output shape
relation gate dimensions
message-passing output shape
prediction head output shape
explanation schema validity
```

## 18.2 `docs/`

Docs should explain architecture and contracts:

```text
architecture_north_star.md
module_interfaces.md
ablation_ladder.md
relation_family_ontology.md
ui_integration_contract.md
```

This file is the north star. It should not be the only documentation forever.

---

# 19. Minimum viable V2 implementation

The first implementation should not include everything.

A disciplined V2.0 should be:

```text
nodes:
  tract-month or CD-month nodes

features:
  same feature parity as B3 where possible

memory:
  lag/rolling memory encoder first

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
  mean/sum aggregation

heads:
  count/regression prediction head

explanations:
  relation gate export
  edge attention export
```

Do not start with:

```text
full transformer memory
full heterogeneous graph
reporting-bias head
uncertainty head
complex counterfactual engine
```

Those belong in later V2.x or V3 phases.

---

# 20. Required ablation ladder

Every version should preserve this ladder as much as possible:

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

The central test is not whether the final model is fancy. The central test is whether each added mechanism earns its complexity.

---

# 21. What success looks like

The V2 model is successful if it can show at least one of the following:

1. It improves prediction over B3, no-edge, random-edge, and generic graph controls.
2. It shows hazard-specific relation activation that is stable and interpretable.
3. It produces better top-k municipal prioritization.
4. It retrieves different urban memories for different hazards.
5. It generates meaningful counterfactual pathway shifts.
6. It identifies uncertainty in sparse or rare hazard scenarios.
7. It separates observed reporting from latent disruption more convincingly than raw 311 models.

The strongest success case is not merely lower MAE. The strongest success case is:

```text
lower MAE
+ better top-k ranking
+ real topology beats placebo topology
+ relation gates shift by hazard in plausible ways
+ explanation exports identify coherent urban pathways
```

---

# 22. What would count as failure

The model family should be considered weak if:

```text
real graph does not beat no-edge neural
real graph does not beat random-edge graph
hazard gates do not differ by hazard
learned memory does not beat lag/rolling memory
attention/pathway exports are unstable or nonsensical
B3 tabular ML remains stronger across most metrics
```

Failure is still scientifically useful. It would tell us that the data supports features/history more than topology or that the relation ontology needs refinement.

---

# 23. Non-goals for V2

V2 should not try to solve every urban resilience problem at once.

Non-goals:

```text
full Québec-wide heterogeneous graph from day one
every possible hazard type
perfect infrastructure modeling
complete causal inference
production UI deployment before model validation
massive transformer architecture before simple controls
```

The model should grow from a controlled scientific core.

---

# 24. Final north-star statement

The V2 model should be built so that a future reader can say:

> This is not just a graph neural network applied to city data. It is a hazard-conditioned urban reasoning system. It remembers past stress, receives a hazard query, activates functional relation pathways, passes messages through the relevant parts of the city, predicts future burden, and exports interpretable pathway evidence.

Everything in this package should serve that statement.

If a file does not help with one of these responsibilities, it probably belongs elsewhere.
