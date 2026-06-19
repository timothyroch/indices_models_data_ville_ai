# Module Interfaces

**Model family:** `v2_hazard_conditioned_functional_ugnn`
**File:** `urban_resilience_models/v2_hazard_conditioned_functional_ugnn/docs/module_interfaces.md`
**Status:** official interface-contract draft
**Scope:** stable conceptual contracts, recommended V2.0 defaults, and extensibility rules for the V2 model family

---

## 1. Purpose

The V2 architecture contains several interacting mechanisms:

```text
urban memory
hazard and scenario encoding
hazard-queried memory
relation-family gating
edge-level attention
relation-specific transformations
functional message passing
prediction
uncertainty
reporting-bias decomposition
pathway explanations
counterfactual diagnostics
```

Without explicit interfaces, these modules can silently disagree about:

```text
tensor shapes
graph membership
node ordering
hazard scope
relation semantics
temporal availability
history masks
edge direction
gate scope
prediction alignment
explanation formats
schema versions
```

This document defines the contracts that should remain stable while internal implementations evolve.

For example:

```text
GRU may become LSTM.
LSTM may become a temporal transformer.
Mean aggregation may become relation-wise attention.
Sigmoid gates may become sparse gates.
A single-horizon head may become a multi-horizon head.
```

Those changes are allowed.

The stable meanings of inputs, outputs, identifiers, relation IDs, prediction alignment, and explanation records must not drift silently.

---

# 2. Normative language

This document uses three levels of obligation.

## 2.1 MUST

A required invariant or stable contract.

A violation is an interface error.

Example:

```text
Every edge MUST reference valid internal node indices.
```

## 2.2 SHOULD

A recommended V2.0 default.

It may change with explicit justification, configuration, documentation, and tests.

Example:

```text
V2.0 SHOULD use sigmoid relation gates because multiple relation families may be active simultaneously.
```

## 2.3 MAY

An optional or future-supported implementation.

Example:

```text
Future implementations MAY use sparsemax, entmax, or top-k relation routing.
```

This distinction prevents recommended mathematical choices from becoming accidental permanent contracts.

---

# 3. Core design rules

Each module MUST have:

```text
one clear scientific responsibility
one explicit input contract
one explicit output contract
one ownership boundary
one matching ablation or control
```

A module MUST NOT silently perform work that belongs to another module.

Examples:

```text
The Hazard Query Encoder encodes hazard context.
It does not compute relation-family gates.

The Relation-Family Gate scores mechanism families.
It does not normalize individual edges.

The Edge-Level Attention module scores specific connections.
It does not aggregate messages.

The Prediction Head predicts outcomes.
It does not format UI payloads.

The Benchmark Adapter maps external artifacts into V2 schemas.
It does not implement neural layers.
```

---

# 4. Global dimension symbols

| Symbol       | Meaning                                                 |
| ------------ | ------------------------------------------------------- |
| `B`          | number of graph/scenario instances in a batch           |
| `N`          | total number of packed nodes across all graph instances |
| `E`          | total number of packed directed edges                   |
| `T`          | maximum temporal-history length                         |
| `R`          | number of registered relation families                  |
| `A`          | number of attention heads                               |
| `K`          | number of prediction targets or horizons                |
| `Q`          | number of uncertainty quantiles                         |
| `F_static`   | width of homogeneous static/current node features       |
| `F_history`  | features per historical time step                       |
| `F_edge`     | edge-attribute width                                    |
| `F_hazard`   | hazard-feature width                                    |
| `F_scenario` | scenario-feature width                                  |
| `H`          | shared hidden-state width                               |
| `H_memory`   | memory-representation width                             |
| `H_hazard`   | hazard-context width                                    |
| `H_scenario` | scenario-context width                                  |

`N` and `E` refer to the packed batch, not necessarily one graph.

---

# 5. Versioning contract

Every saved checkpoint, exported prediction bundle, and explanation artifact MUST record compatible contract versions.

Required version identifiers:

```text
model_family_version
model_config_version
batch_schema_version
relation_registry_version
hazard_registry_version
feature_contract_version
prediction_schema_version
explanation_schema_version
```

Example:

```json
{
  "model_family_version": "v2.0.0-dev",
  "model_config_version": "0.1",
  "batch_schema_version": "0.2",
  "relation_registry_version": "0.1",
  "hazard_registry_version": "0.1",
  "feature_contract_version": "0.1",
  "prediction_schema_version": "0.1",
  "explanation_schema_version": "0.1"
}
```

A checkpoint MUST NOT silently load against an incompatible relation or hazard registry.

For example, if relation ID `4` represented `heat_exposure` during training, it must not represent `service_access` during inference.

Incompatible changes MUST require at least one of:

```text
a schema-version increment
a registry-version increment
a migration function
an explicit compatibility override
```

---

# 6. Identifier conventions

## 6.1 Internal node index

Model-facing graph tensors MUST use contiguous integer indices:

```text
0, 1, 2, ..., N - 1
```

The canonical name is:

```text
internal_node_index
```

This index is normally implicit in tensor row order.

## 6.2 External node ID

Domain-facing geographic or infrastructure identifiers MUST be preserved separately:

```text
external_node_ids: Sequence[str]
```

Examples:

```text
census tract DGUID
census division ID
road segment ID
hospital ID
drainage asset ID
flood-zone polygon ID
```

The model MUST NOT assume that an external ID is a valid tensor index.

## 6.3 Mapping invariant

A batch MUST preserve an unambiguous mapping:

```text
internal_node_index ↔ external_node_id
```

Predictions and explanations MUST remain aligned with external node IDs.

## 6.4 Edge identities

An edge is identified internally by its row position:

```text
internal_edge_index = 0, 1, ..., E - 1
```

If an external edge ID exists, it MAY be preserved as:

```text
external_edge_ids: Sequence[str] | None
```

---

# 7. Graph batching contract

## 7.1 Required graph membership

Every packed batch MUST include:

```text
node_batch_index: [N]
```

where:

```text
node_batch_index[i] = graph/scenario instance containing node i
```

The values MUST satisfy:

```text
0 <= node_batch_index[i] < B
```

Even a single-graph batch SHOULD provide:

```text
node_batch_index = zeros([N])
```

This avoids separate code paths for single-graph and multi-graph execution.

## 7.2 Optional graph pointer

A packed batch MAY include:

```text
graph_ptr: [B + 1]
```

where:

```text
graph_ptr[b]
graph_ptr[b + 1]
```

define the contiguous node range for graph `b`.

If present, it MUST agree with `node_batch_index`.

## 7.3 Edge batch membership

A packed batch MAY include:

```text
edge_batch_index: [E]
```

It can usually be derived as:

```python
source_index = edge_index[0]
target_index = edge_index[1]

source_batch = node_batch_index[source_index]
target_batch = node_batch_index[target_index]
```

For ordinary within-graph message passing, the invariant MUST be:

```text
source_batch == target_batch
```

and:

```python
edge_batch_index = source_batch
```

Cross-graph edges MUST be rejected unless a future explicitly registered relation permits them.

## 7.4 Hazard broadcasting

For graph-level hazard context:

```text
graph_hazard_context: [B, H_hazard]
```

node-aligned context MUST be created explicitly:

```python
node_hazard_context = graph_hazard_context[node_batch_index]
```

Implicit or shape-dependent broadcasting SHOULD be avoided.

---

# 8. Edge-index convention

The canonical edge-index shape is:

```python
edge_index.shape == [2, E]
```

with:

```python
edge_index[0]  # source internal-node indices
edge_index[1]  # target internal-node indices
```

Every index MUST satisfy:

```text
0 <= edge_index < N
```

Each edge is directed.

Undirected graph relations SHOULD be stored as two directed edges:

```text
A → B
B → A
```

Directedness semantics MUST be defined in the relation registry.

---

# 9. Node-type and heterogeneous-feature contract

## 9.1 V2.0 homogeneous case

For a tract-only or CD-only graph, raw features MAY be supplied as:

```text
node_features: [N, F_static]
```

## 9.2 Heterogeneous case

When multiple node types exist, their raw feature spaces may differ.

Examples:

```text
tracts
roads
water bodies
hospitals
drainage assets
heat-island polygons
flood-zone polygons
```

The stable model-facing contract is:

```text
Raw features may be heterogeneous.
Each node type receives an appropriate input projector.
Functional message passing receives one packed shared hidden state.
```

Required packed tensors:

```text
node_state: [N, H]
node_type:  [N]
```

Possible preprocessing:

```text
tract raw features
    → tract input projector
    → [N_tract, H]

road raw features
    → road input projector
    → [N_road, H]

drainage raw features
    → drainage input projector
    → [N_drainage, H]
```

Then:

```text
all projected states
    → packed node_state [N, H]
```

## 9.3 Typed raw-feature mappings

Data adapters MAY expose:

```text
node_features_by_type
```

before projection.

Core functional message-passing modules SHOULD NOT depend on typed raw-feature dictionaries. They SHOULD receive the shared hidden-space representation.

## 9.4 Node-type requirement

`node_type: [N]`:

```text
MAY be omitted for a declared single-node-type experiment.
MUST be present when more than one node type is packed.
```

---

# 10. Weight terminology

The following concepts MUST remain distinct.

## 10.1 Semantic edge weight

A data-provided domain quantity:

```text
semantic_edge_weight: [E]
```

Examples:

```text
shared boundary length
distance-decay value
area overlap
service capacity
flow estimate
accessibility score
```

## 10.2 Normalization weight

A mathematically derived graph-normalization term:

```text
normalization_weight: [E]
```

Examples:

```text
degree normalization
relation-wise degree normalization
symmetric normalization
```

## 10.3 Attention weight

A learned edge-specific score:

```text
attention_weight: [E]
```

or:

```text
attention_weight_by_head: [E, A]
```

## 10.4 Relation-gate weight

A learned hazard-conditioned mechanism-family score:

```text
relation_gate_weight: [B, R]
```

or:

```text
relation_gate_weight: [N, R]
```

These values MUST NOT be treated as interchangeable.

---

# 11. Floating-point, index, and mask conventions

Unless justified otherwise:

```text
node features: float32
history sequences: float32
edge attributes: float32
hazard features: float32
scenario features: float32
model parameters: float32
```

Index tensors MUST normally use:

```text
torch.long
```

Boolean masks MUST use:

```text
torch.bool
```

For every mask:

```text
True  = valid / available / included
False = missing / padded / excluded
```

This applies to:

```text
history masks
target masks
node masks
edge masks
feature-availability masks
```

---

# 12. Missing-value contract

Uncontrolled `NaN` or infinite values MUST NOT reach neural modules.

Missingness MUST be handled through one or more of:

```text
training-fitted imputation
explicit missingness indicators
feature masks
history masks
relation-specific absence indicators
```

A module SHOULD validate finite tensors during research development.

A future optimized inference mode MAY reduce validation overhead after contracts are stable.

---

# 13. Temporal causality and feature-availability contract

This is a scientific invariant, not merely metadata.

## 13.1 Required temporal fields

Each graph/scenario sample MUST preserve:

```text
origin_time
history_start_time
history_end_time
feature_availability_cutoff
forecast_horizon
```

A supervised sample MUST additionally preserve:

```text
target_start_time
target_end_time
```

These may be represented per graph instance:

```text
[B]
```

or per node when truly node-specific:

```text
[N]
```

The scope MUST be explicit.

## 13.2 Causality invariants

For every supervised sample:

```text
history_start_time <= history_end_time
history_end_time <= origin_time
feature_availability_cutoff <= origin_time
target_start_time > origin_time
target_end_time >= target_start_time
```

A feature MUST NOT be used if it was unavailable at prediction origin time.

## 13.3 Forecast-horizon semantics

`forecast_horizon` MUST identify what prediction means.

Examples:

```text
next_1_month
next_3_months
next_6_months
event_occurrence_within_30_days
```

Prediction records MUST preserve the horizon.

## 13.4 Dynamic edge availability

If edges can change over time, the schema MAY include:

```text
edge_valid_from:       [E]
edge_valid_to:         [E]
edge_observation_time: [E]
```

When present, the active-edge invariant MUST be enforced relative to `origin_time`.

A model MUST NOT use a future-observed edge at an earlier prediction origin.

## 13.5 Validation responsibility

Temporal invariants MUST be checked in at least:

```text
data adapters
dataset construction
schema validation
pre-training audits
```

Tensor-shape validity alone is insufficient.

---

# 14. Canonical V2 batch contract

The canonical model input is provisionally named:

```text
UrbanGraphBatch
```

The exact Python representation MAY be:

```text
a frozen dataclass
a validated dataclass
a typed mapping
a dedicated tensor container
```

## 14.1 Required fields

A minimal inference-ready batch MUST contain:

```text
external_node_ids
node_batch_index
node_features or preprojected node_state
history_sequences or explicit no-history declaration
history_mask when histories are padded
hazard_ids
edge_index
edge_relation_type
origin_time
history_start_time
history_end_time
feature_availability_cutoff
forecast_horizon
contract_versions
```

## 14.2 Conditionally required fields

```text
node_type:
  required for heterogeneous-node batches

scenario_features:
  required when scenario-conditioned configuration is active

edge_attributes:
  required when the active relation contracts require them

targets and target_mask:
  required for supervised training/evaluation
  optional for inference and counterfactual analysis
```

## 14.3 Optional fields

```text
graph_ptr
edge_batch_index
external_edge_ids
hazard_features
semantic_edge_weight
edge_valid_from
edge_valid_to
edge_observation_time
split_labels
geographic_metadata
feature_names
history_feature_names
edge_attribute_names
typed_metadata
```

## 14.4 Canonical shapes

| Field                  |                                           Shape | Type  |
| ---------------------- | ----------------------------------------------: | ----- |
| `node_features`        |                                 `[N, F_static]` | float |
| `node_state`           |                               optional `[N, H]` | float |
| `node_type`            |                      optional/conditional `[N]` | long  |
| `node_batch_index`     |                                           `[N]` | long  |
| `graph_ptr`            |                                optional `[B+1]` | long  |
| `history_sequences`    |                             `[N, T, F_history]` | float |
| `history_mask`         |                                        `[N, T]` | bool  |
| `hazard_ids`           |                                  `[B]` or `[N]` | long  |
| `hazard_features`      |     optional `[B, F_hazard]` or `[N, F_hazard]` | float |
| `scenario_features`    | optional `[B, F_scenario]` or `[N, F_scenario]` | float |
| `edge_index`           |                                        `[2, E]` | long  |
| `edge_relation_type`   |                                           `[E]` | long  |
| `edge_attributes`      |                          optional `[E, F_edge]` | float |
| `semantic_edge_weight` |                                  optional `[E]` | float |
| `edge_batch_index`     |                                  optional `[E]` | long  |
| `targets`              |                               optional `[N, K]` | float |
| `target_mask`          |                               optional `[N, K]` | bool  |

## 14.5 Supervision invariant

For supervised training or evaluation:

```text
targets is not None
target_mask is not None
targets.shape == target_mask.shape
```

For inference:

```text
targets MAY be None
target_mask MAY be None
```

A predictor MUST NOT fabricate targets.

---

# 15. Typed metadata contract

An unrestricted required dictionary SHOULD be avoided.

Metadata SHOULD use typed structures such as:

```text
BatchMetadata
TemporalMetadata
GeographicMetadata
ContractVersionMetadata
RunMetadata
```

Optional arbitrary extension fields MAY exist under a clearly marked namespace:

```text
extra_metadata
```

Core model logic MUST NOT depend on undocumented arbitrary keys.

---

# 16. Hazard scope

Hazard context may be graph-level or node-level.

## 16.1 Graph-level hazard

```text
hazard_ids: [B]
```

Each graph/scenario instance has one hazard query.

Node alignment:

```python
node_hazard_ids = hazard_ids[node_batch_index]
```

This SHOULD be the default V2.0 mode.

## 16.2 Node-level hazard

```text
hazard_ids: [N]
```

Different nodes may receive different hazard queries.

This MAY be supported in later experiments.

## 16.3 Scope declaration

The batch or configuration MUST declare:

```text
hazard_scope = graph | node
```

The model MUST NOT infer hazard scope only from tensor length when `B == N` could be ambiguous.

---

# 17. Top-level model interface

## 17.1 Module

```text
model.py
```

## 17.2 Suggested class

```python
HazardConditionedFunctionalUGNN
```

## 17.3 Responsibility

The top-level model orchestrates:

```text
batch validation
typed input projection
urban memory encoding
hazard/scenario encoding
hazard-queried memory
initial node-state construction
functional message-passing stack
prediction heads
optional uncertainty/reporting heads
structured explanation collection
```

It MUST NOT contain detailed implementations of every submodule.

## 17.4 Suggested forward interface

```python
def forward(
    self,
    batch: UrbanGraphBatch,
    *,
    return_explanations: bool = False,
    return_intermediate_states: bool = False,
) -> ModelOutput:
    ...
```

## 17.5 High-level sequence

```text
1. Validate schema and registry versions.
2. Validate temporal causality and feature availability.
3. Validate graph membership and edge endpoints.
4. Project raw node features into shared hidden space.
5. Encode urban memory.
6. Encode hazard and scenario context.
7. Retrieve hazard-relevant memory.
8. Build the initial node state.
9. Apply functional message-passing layers.
10. Predict future burden or risk.
11. Optionally estimate uncertainty.
12. Optionally decompose reporting propensity.
13. Assemble explanation traces.
14. Return node-aligned structured outputs.
```

## 17.6 Suggested output

```text
ModelOutput
```

Required:

```text
predictions
prediction_alignment
contract_versions
```

Optional:

```text
uncertainty
reporting_bias
explanations
intermediate_states
regularization_terms
```

## 17.7 Prediction alignment

Every output MUST preserve:

```text
external_node_ids
node_batch_index
target name
forecast horizon
hazard ID
origin time
checkpoint or run ID
```

Predictions MUST NOT be returned as anonymous rows detached from domain identities.

---

# 18. Input projection interface

## 18.1 Responsibility

Map raw node-type-specific features into shared hidden space.

Suggested abstraction:

```python
class NodeInputProjector(nn.Module):
    def forward(
        self,
        node_features: Tensor | Mapping[int, Tensor],
        *,
        node_type: Tensor | None = None,
    ) -> Tensor:
        ...
```

Output:

```text
node_state: [N, H]
```

## 18.2 Homogeneous V2.0 case

V2.0 MAY use one shared projector:

```text
[N, F_static] → [N, H]
```

## 18.3 Heterogeneous case

Future versions SHOULD support one projector per node type.

The projector MUST preserve node ordering.

---

# 19. Urban Memory Encoder interface

## 19.1 Package

```text
memory/
```

## 19.2 Coordinating module

```text
memory/urban_memory_encoder.py
```

## 19.3 Scientific responsibility

```text
What does each urban node remember about its past stress?
```

## 19.4 Suggested interface

```python
class UrbanMemoryEncoder(nn.Module):
    def forward(
        self,
        history_sequence: Tensor,
        history_mask: Tensor | None = None,
        *,
        return_temporal_states: bool = False,
        return_attention: bool = False,
    ) -> MemoryEncoderOutput:
        ...
```

The generic `context` argument is intentionally omitted from the base interface. Hazard-dependent retrieval belongs in `HazardQueriedMemory`.

## 19.5 Inputs

```text
history_sequence: [N, T, F_history]
history_mask:     optional [N, T]
```

## 19.6 Outputs

```text
memory_state:       [N, H_memory]
temporal_states:    optional [N, T, H_memory]
temporal_attention: optional [N, T]
```

The output SHOULD use a structured object:

```text
MemoryEncoderOutput
```

## 19.7 Ownership

The encoder owns:

```text
history-mask handling
sequence encoding
memory-state extraction
standardized outputs across encoder types
```

It does not own:

```text
hazard embedding
hazard-memory cross-attention
relation gating
graph propagation
prediction
```

## 19.8 Required errors

It MUST reject:

```text
non-rank-3 history input
mask shape not equal to [N, T]
unexpected non-finite values
feature width inconsistent with configuration
unsupported all-masked histories without fallback
```

## 19.9 Matching ablations

```text
no memory
lag/rolling memory
GRU memory
LSTM memory
transformer memory
```

---

# 20. Lag Memory Encoder interface

## 20.1 Module

```text
memory/lag_memory_encoder.py
```

## 20.2 Responsibility

Encode handcrafted historical summaries.

Inputs may include:

```text
lag_1
rolling_3
rolling_6
rolling_12
seasonal historical mean
hazard-specific rolling features
```

Suggested interface:

```python
class LagMemoryEncoder(nn.Module):
    def forward(
        self,
        memory_features: Tensor,
    ) -> MemoryEncoderOutput:
        ...
```

Shapes:

```text
memory_features: [N, F_memory]
memory_state:    [N, H_memory]
```

This is the transparent memory baseline and SHOULD be implemented before recurrent or transformer memory.

---

# 21. Recurrent Memory Encoder interface

## 21.1 Module

```text
memory/recurrent_memory_encoder.py
```

Suggested configuration:

```text
cell_type = gru | lstm
hidden_dim
num_layers
dropout
bidirectional
```

Input:

```text
history_sequence: [N, T, F_history]
history_mask:     [N, T]
```

Output:

```text
memory_state:    [N, H_memory]
temporal_states: [N, T, H_memory]
```

The recurrent implementation MUST respect valid history lengths.

---

# 22. Transformer Memory Encoder interface

## 22.1 Module

```text
memory/transformer_encoder.py
```

Responsibilities:

```text
history feature projection
temporal/positional encoding
padding-mask support
optional causal masking
temporal-state export
```

Input:

```text
history_sequence: [N, T, F_history]
history_mask:     [N, T]
```

Output:

```text
memory_state:    [N, H_memory]
temporal_states: [N, T, H_memory]
```

V2.0 MAY leave this module as an explicit `NotImplementedError` placeholder.

---

# 23. Temporal Attention interface

## 23.1 Module

```text
memory/temporal_attention.py
```

## 23.2 Scientific responsibility

```text
Which past periods matter for the current representation?
```

Suggested interface:

```python
class TemporalAttention(nn.Module):
    def forward(
        self,
        temporal_states: Tensor,
        *,
        query: Tensor | None = None,
        history_mask: Tensor | None = None,
    ) -> TemporalAttentionOutput:
        ...
```

Inputs:

```text
temporal_states: [N, T, H_memory]
query:           optional [N, H_query]
history_mask:    optional [N, T]
```

Outputs:

```text
context_state:     [N, H_memory]
attention_weights: [N, T]
```

For nodes with valid history:

```text
attention over valid periods SHOULD sum approximately to 1
```

Masked periods MUST receive zero weight.

Attention MUST be described as a model attention signal, not causal importance.

---

# 24. Hazard Embedding interface

## 24.1 Module

```text
hazard/hazard_embeddings.py
```

Suggested interface:

```python
class HazardEmbedding(nn.Module):
    def forward(self, hazard_ids: Tensor) -> Tensor:
        ...
```

Input:

```text
hazard_ids: [B] or [N]
```

Output:

```text
hazard_embedding: [..., H_hazard]
```

Unknown hazard IDs MUST be rejected.

---

# 25. Scenario Encoder interface

## 25.1 Module

```text
hazard/scenario_encoder.py
```

Encodes dynamic context such as:

```text
month
season
precipitation
temperature anomaly
river level
snowmelt proxy
event severity
forecast horizon
```

Suggested interface:

```python
class ScenarioEncoder(nn.Module):
    def forward(
        self,
        scenario_features: Tensor,
        *,
        scenario_mask: Tensor | None = None,
    ) -> Tensor:
        ...
```

Input:

```text
[B, F_scenario] or [N, F_scenario]
```

Output:

```text
[B, H_scenario] or [N, H_scenario]
```

Its scope MUST match the declared scenario scope.

---

# 26. Hazard Query Encoder interface

## 26.1 Module

```text
hazard/hazard_query_encoder.py
```

## 26.2 Scientific responsibility

```text
Under what hazard and scenario is the model reasoning?
```

Suggested interface:

```python
class HazardQueryEncoder(nn.Module):
    def forward(
        self,
        hazard_ids: Tensor,
        *,
        hazard_features: Tensor | None = None,
        scenario_features: Tensor | None = None,
        node_batch_index: Tensor | None = None,
        scope: str = "graph",
    ) -> HazardEncoderOutput:
        ...
```

## 26.3 Graph-level behavior

For:

```text
hazard_ids: [B]
```

the encoder SHOULD produce:

```text
graph_hazard_context: [B, H_hazard]
node_hazard_context:  [N, H_hazard]
```

using:

```python
node_hazard_context = graph_hazard_context[node_batch_index]
```

## 26.4 Node-level behavior

For:

```text
hazard_ids: [N]
```

the output is already node-aligned.

## 26.5 Ownership

The encoder owns:

```text
hazard embedding
scenario projection
hazard-scenario fusion
explicit graph-to-node broadcasting
```

It does not own:

```text
temporal retrieval
relation gating
edge attention
prediction
```

---

# 27. Generic Cross-Attention interface

## 27.1 Module

```text
hazard/cross_attention.py
```

Possible uses:

```text
hazard query → temporal states
hazard query → relation tokens
hazard query → scenario tokens
```

Suggested interface:

```python
class CrossAttention(nn.Module):
    def forward(
        self,
        query: Tensor,
        key_value: Tensor,
        *,
        key_value_mask: Tensor | None = None,
        return_attention: bool = False,
    ) -> CrossAttentionOutput:
        ...
```

Typical shapes:

```text
query:             [N, 1, H]
key_value:         [N, T, H]
key_value_mask:    [N, T]
context:           [N, H]
attention_by_head: optional [N, A, T]
```

The module MUST document how multiple heads are reduced for exported explanations.

Recommended exporter default:

```text
mean attention across heads
```

while preserving per-head values for diagnostics when requested.

---

# 28. Hazard-Queried Memory interface

## 28.1 Module

```text
memory/hazard_queried_memory.py
```

## 28.2 Scientific responsibility

```text
Which parts of this node’s history matter under the current hazard?
```

Suggested interface:

```python
class HazardQueriedMemory(nn.Module):
    def forward(
        self,
        temporal_states: Tensor,
        hazard_context: Tensor,
        *,
        history_mask: Tensor | None = None,
        base_memory_state: Tensor | None = None,
        return_attention: bool = False,
    ) -> HazardQueriedMemoryOutput:
        ...
```

Inputs:

```text
temporal_states:   [N, T, H_memory]
hazard_context:    [N, H_hazard]
history_mask:      optional [N, T]
base_memory_state: optional [N, H_memory]
```

Outputs:

```text
hazard_memory_state: [N, H_memory]
temporal_attention:  optional [N, T] or [N, A, T]
```

This module owns hazard-to-memory retrieval.

It does not own hazard embedding, relation gating, or graph aggregation.

Ablations:

```text
generic memory only
hazard embedding concatenated after memory
hazard-queried memory
hazard-queried memory with temporal-attention export
```

---

# 29. Relation registry interface

## 29.1 Modules

```text
relations/relation_types.py
relations/relation_registry.py
relations/hazard_relation_priors.py
relations/relation_validation.py
```

## 29.2 Relation specification

Suggested object:

```python
@dataclass(frozen=True)
class RelationSpec:
    relation_id: int
    name: str
    display_name: str
    semantic_role: str
    source_node_types: tuple[str, ...]
    target_node_types: tuple[str, ...]
    directed: bool
    is_control_relation: bool
    is_real_relation: bool
    explanation_allowed: bool
    required_edge_attributes: tuple[str, ...]
    optional_edge_attributes: tuple[str, ...]
    registry_version: str
```

## 29.3 Registry functions

```python
def get_relation_spec(relation_id: int) -> RelationSpec:
    ...

def get_relation_id(name: str) -> int:
    ...

def get_relation_name(relation_id: int) -> str:
    ...

def list_relations() -> tuple[RelationSpec, ...]:
    ...
```

Unknown relation IDs MUST be rejected.

## 29.4 Prior contract

Hazard-relation priors MUST include:

```text
prior version
hazard-registry version
relation-registry version
scope
construction method
```

They MUST NOT be derived from validation or test outcomes unless explicitly being analyzed after evaluation rather than used for training.

---

# 30. Relation-Family Gate interface

## 30.1 Module

```text
functional_message_passing/relation_family_gate.py
```

## 30.2 Scientific responsibility

```text
Which categories of urban mechanisms are active under this hazard?
```

Suggested interface:

```python
class RelationFamilyGate(nn.Module):
    def forward(
        self,
        hazard_context: Tensor,
        *,
        node_state: Tensor | None = None,
        scenario_context: Tensor | None = None,
        relation_priors: Tensor | None = None,
        relation_mask: Tensor | None = None,
        scope: str = "target_node",
    ) -> RelationGateOutput:
        ...
```

## 30.3 Supported scopes

```text
graph
target_node
source_node
source_target
```

### Graph scope

```text
gate_values: [B, R]
```

### Target-node scope

```text
gate_values: [N, R]
```

Canonical V2.0 interpretation:

> A target-node gate specifies which incoming relation families the target node activates under the current hazard.

For edge `source → target`:

```python
edge_gate = gate_values[
    target_index,
    edge_relation_type,
]
```

### Source-node scope

```python
edge_gate = gate_values[
    source_index,
    edge_relation_type,
]
```

### Source-target scope

A learned function MAY combine source and target states.

## 30.4 V2.0 default

V2.0 SHOULD use:

```text
scope = target_node
```

unless an experiment explicitly tests another scope.

## 30.5 Output contract

```text
gate_values
gate_logits
scope
relation_mask
regularization_terms
registry_version
```

Gate values MUST expose one value per registered active relation family.

V2.0 SHOULD use:

```text
sigmoid gate values in [0, 1]
```

Future versions MAY use:

```text
sparsemax
entmax
top-k routing
hard-concrete gates
```

## 30.6 Priors

Priors MAY enter as:

```text
initial biases
regularization targets
input features
```

They MUST NOT silently prohibit unexpected relations unless hard constraints are explicitly configured.

## 30.7 Ablations

```text
no gate
uniform gate
hazard-blind learned gate
hazard-conditioned graph gate
hazard-conditioned target-node gate
prior-regularized gate
```

---

# 31. Edge-Level Attention interface

## 31.1 Module

```text
functional_message_passing/edge_attention.py
```

## 31.2 Scientific responsibility

```text
Which specific connection matters under this hazard?
```

Suggested interface:

```python
class EdgeAttention(nn.Module):
    def forward(
        self,
        node_state: Tensor,
        edge_index: Tensor,
        edge_relation_type: Tensor,
        hazard_context: Tensor,
        *,
        edge_attributes: Tensor | None = None,
        relation_gate_weight: Tensor | None = None,
        edge_mask: Tensor | None = None,
    ) -> EdgeAttentionOutput:
        ...
```

Inputs:

```text
node_state:         [N, H]
edge_index:         [2, E]
edge_relation_type: [E]
hazard_context:     [N, H_hazard]
edge_attributes:    optional [E, F_edge]
edge_mask:          optional [E]
```

Outputs:

```text
attention_logits:         [E] or [E, A]
attention_weight:         [E]
attention_weight_by_head: optional [E, A]
normalization_scope
head_reduction_policy
```

## 31.3 Normalization scope

The scope MUST be explicit.

Supported modes MAY include:

```text
target_node
target_node_and_relation
global_relation
unnormalized_sigmoid
```

V2.0 SHOULD begin with:

```text
target_node_and_relation
```

or:

```text
target_node
```

and record the choice in experiment metadata.

## 31.4 Gate-attention separation

```text
relation gate:
  importance of a mechanism family

edge attention:
  importance of a specific connection
```

Final learned routing weight MAY be:

```text
relation_gate_weight
× attention_weight
```

but both components MUST remain separately exportable.

## 31.5 Explanation policy

Attention records MUST include:

```text
source external ID
target external ID
relation ID
hazard ID
attention value
normalization scope
layer index
head index or reduction policy
checkpoint/run ID
registry versions
```

Attention MUST NOT be labeled as causal importance.

---

# 32. Relation Transform interface

## 32.1 Module

```text
functional_message_passing/relation_transforms.py
```

Scientific idea:

```text
Information should not travel through drainage, canopy, service-access, and adjacency relations through identical transformations.
```

Suggested interface:

```python
class RelationTransforms(nn.Module):
    def forward(
        self,
        source_state: Tensor,
        edge_relation_type: Tensor,
        *,
        edge_attributes: Tensor | None = None,
    ) -> Tensor:
        ...
```

Inputs:

```text
source_state:       [E, H]
edge_relation_type: [E]
```

Output:

```text
transformed_source: [E, H]
```

Implementations MAY include:

```text
one linear map per relation
shared transform plus relation embedding
basis decomposition
low-rank transforms
typed MLPs
```

---

# 33. Edge Normalization interface

## 33.1 Module

```text
functional_message_passing/edge_normalization.py
```

Suggested interface:

```python
class EdgeNormalizer(nn.Module):
    def forward(
        self,
        edge_index: Tensor,
        *,
        num_nodes: int,
        edge_relation_type: Tensor | None = None,
        edge_attributes: Tensor | None = None,
        semantic_edge_weight: Tensor | None = None,
    ) -> Tensor:
        ...
```

Output:

```text
normalization_weight: [E]
```

Modes MAY include:

```text
none
source_degree
target_degree
symmetric_degree
relation_specific_degree
distance_decay
semantic_weight_normalization
```

The normalizer MUST NOT overwrite semantic edge weights.

---

# 34. Message Builder interface

## 34.1 Module

```text
functional_message_passing/message_builders.py
```

Suggested interface:

```python
class MessageBuilder(nn.Module):
    def forward(
        self,
        transformed_source: Tensor,
        *,
        relation_gate_weight: Tensor | None = None,
        attention_weight: Tensor | None = None,
        normalization_weight: Tensor | None = None,
        semantic_edge_weight: Tensor | None = None,
        edge_attributes: Tensor | None = None,
    ) -> MessageBuilderOutput:
        ...
```

Inputs:

```text
transformed_source:   [E, H]
relation_gate_weight: optional [E]
attention_weight:     optional [E]
normalization_weight: optional [E]
semantic_edge_weight: optional [E]
```

Output:

```text
messages: [E, H]
```

Conceptual V2.0 formulation:

```text
message =
  transformed source state
  × relation gate
  × edge attention
  × normalization weight
  × optional semantic edge modulation
```

Optional diagnostics:

```text
message_norms
component_weights
```

---

# 35. Aggregator interface

## 35.1 Module

```text
functional_message_passing/aggregators.py
```

Suggested interface:

```python
class MessageAggregator(nn.Module):
    def forward(
        self,
        messages: Tensor,
        target_index: Tensor,
        *,
        num_nodes: int,
        edge_relation_type: Tensor | None = None,
    ) -> AggregationOutput:
        ...
```

Inputs:

```text
messages:     [E, H]
target_index: [E]
```

Required output:

```text
aggregated_messages: [N, H]
```

Optional:

```text
relation_aggregates: [N, R, H]
incoming_counts:     [N]
```

Supported modes MAY include:

```text
sum
mean
degree-normalized sum
max
relation-wise aggregation then fusion
```

V2.0 SHOULD use a simple transparent aggregator before testing more complex fusion.

---

# 36. Functional Message-Passing Layer interface

## 36.1 Module

```text
functional_message_passing/layer.py
```

## 36.2 Suggested class

```python
HazardConditionedFunctionalMessagePassingLayer
```

## 36.3 Responsibility

Coordinate:

```text
relation-family gating
relation-specific transforms
edge-level attention
normalization
message construction
aggregation
node updates
explanation trace collection
```

## 36.4 Suggested interface

```python
class HazardConditionedFunctionalMessagePassingLayer(nn.Module):
    def forward(
        self,
        node_state: Tensor,
        edge_index: Tensor,
        edge_relation_type: Tensor,
        hazard_context: Tensor,
        *,
        node_batch_index: Tensor,
        node_type: Tensor | None = None,
        edge_attributes: Tensor | None = None,
        semantic_edge_weight: Tensor | None = None,
        scenario_context: Tensor | None = None,
        relation_priors: Tensor | None = None,
        edge_mask: Tensor | None = None,
        return_explanations: bool = False,
    ) -> FunctionalMessagePassingOutput:
        ...
```

## 36.5 Input shapes

```text
node_state:          [N, H]
edge_index:          [2, E]
edge_relation_type:  [E]
hazard_context:      [N, H_hazard]
node_batch_index:    [N]
node_type:           optional [N]
edge_attributes:     optional [E, F_edge]
semantic_edge_weight:optional [E]
scenario_context:    optional [N, H_scenario]
edge_mask:           optional [E]
```

## 36.6 Output

```text
updated_node_state: [N, H]
relation_gate_output
edge_attention_output
aggregated_messages
normalization_weight
explanation_trace
regularization_terms
```

## 36.7 Internal sequence

```text
1. Validate relation IDs and graph membership.
2. Validate source and target nodes belong to compatible graphs.
3. Compute relation-family gates.
4. Gather source and target states.
5. Apply relation-specific transformations.
6. Compute edge-level attention.
7. Map relation gates to edges using declared gate scope.
8. Compute normalization weights.
9. Build edge messages.
10. Aggregate at target nodes.
11. Update node states.
12. Apply residual, normalization, and dropout as configured.
13. Return structured traces when requested.
```

## 36.8 Node update

The layer MUST preserve:

```text
[N, H] → [N, H]
```

unless an explicit output projection is configured.

V2.0 SHOULD begin with a residual normalized update such as:

```text
updated_state =
  LayerNorm(
    node_state
    + dropout(update_mlp(aggregated_messages))
  )
```

Future versions MAY use:

```text
GRU-style node updates
relation-wise fusion
gated residuals
pre-normalization
post-normalization
```

## 36.9 Zero-edge behavior

When:

```text
E = 0
```

the layer MUST return a valid result.

Recommended behavior:

```text
residual/self-state update
empty attention trace
zero aggregated message
```

It MUST NOT produce `NaN` values or crash.

## 36.10 Isolated-node behavior

Nodes with no incoming edges MUST retain a valid state through residual/self processing.

## 36.11 Ablations

```text
no graph
generic mean GraphSAGE-style layer
uniform relation gates
no edge attention
shared relation transform
hazard-blind message passing
full hazard-conditioned functional layer
```

---

# 37. Prediction Head interface

## 37.1 Module

```text
heads/prediction_heads.py
```

Suggested interface:

```python
class PredictionHead(nn.Module):
    def forward(
        self,
        node_state: Tensor,
        *,
        hazard_context: Tensor | None = None,
        forecast_horizon: Tensor | Sequence[str] | None = None,
    ) -> PredictionOutput:
        ...
```

Input:

```text
node_state: [N, H]
```

Output:

```text
prediction_mean: [N, K]
```

Possible additional fields:

```text
count_rate
ranking_score
task-specific outputs
output transformation
```

Each output MUST preserve:

```text
external node alignment
target name
forecast horizon
hazard
origin time
output transformation
higher-is-riskier orientation
```

A count head SHOULD use a nonnegative transformation such as `softplus`, unless another count distribution is explicitly modeled.

---

# 38. Prediction alignment contract

Suggested object:

```text
PredictionAlignment
```

Required fields:

```text
external_node_ids
node_batch_index
hazard_ids
origin_time
target_names
forecast_horizons
checkpoint_id
run_id
model_family_version
prediction_schema_version
```

A prediction array without alignment metadata MUST NOT be treated as a complete inference artifact.

---

# 39. Uncertainty Head interface

## 39.1 Module

```text
heads/uncertainty_heads.py
```

Suggested interface:

```python
class UncertaintyHead(nn.Module):
    def forward(
        self,
        node_state: Tensor,
        prediction_output: PredictionOutput,
        *,
        hazard_context: Tensor | None = None,
    ) -> UncertaintyOutput:
        ...
```

Possible outputs:

```text
variance:           [N, K]
standard_deviation: [N, K]
quantiles:          [N, K, Q]
lower_bound:        [N, K]
upper_bound:        [N, K]
confidence_score:   [N, K]
```

The method MUST be declared.

Examples:

```text
heteroscedastic variance
quantile prediction
ensemble variance
Monte Carlo dropout
conformal interval wrapper
```

Required evaluations MAY include:

```text
interval coverage
interval width
quantile loss
calibration error
error-versus-uncertainty association
```

---

# 40. Reporting-Bias Head interface

## 40.1 Module

```text
heads/reporting_bias_heads.py
```

Suggested interface:

```python
class ReportingBiasHead(nn.Module):
    def forward(
        self,
        node_state: Tensor,
        *,
        reporting_features: Tensor | None = None,
        hazard_context: Tensor | None = None,
    ) -> ReportingBiasOutput:
        ...
```

Possible outputs:

```text
latent_disruption:    [N, K]
reporting_propensity: [N, K]
observed_prediction:  [N, K]
```

This module is optional for V2.0.

Its assumptions MUST be documented.

It MUST NOT be presented as identifying true latent disruption without adequate identifiability evidence.

---

# 41. Explanation Trace interface

## 41.1 Module

```text
explanations/explanation_schemas.py
```

Suggested object:

```text
ExplanationTrace
```

Required minimal fields:

```text
external node IDs
hazard IDs
origin time
forecast horizon
relation-gate values
edge-attention references
layer indices
checkpoint ID
run ID
relation-registry version
hazard-registry version
explanation-schema version
```

Optional:

```text
temporal attention
pathway scores
top neighbors
counterfactual deltas
uncertainty summary
feature attribution
```

## 41.2 Internal versus serialized form

Internal traces MAY remain tensors:

```text
gate_values:        [N, R]
attention_weight:   [E]
temporal_attention: [N, T]
```

Exporters MUST convert these into serializable domain-aligned records.

---

# 42. Relation Gate Exporter interface

## 42.1 Module

```text
explanations/relation_gate_exporter.py
```

Suggested interface:

```python
def export_relation_gates(
    gate_output: RelationGateOutput,
    *,
    relation_registry: RelationRegistry,
    external_node_ids: Sequence[str],
    node_batch_index: Tensor,
    hazard_ids: Sequence[str],
    checkpoint_id: str,
    run_id: str,
    top_k: int | None = None,
) -> list[RelationGateRecord]:
    ...
```

Each record MUST include:

```text
node or graph scope
hazard
relation ID
relation name
gate value
gate scope
is control relation
explanation allowed
checkpoint/run ID
registry versions
```

---

# 43. Attention Exporter interface

## 43.1 Module

```text
explanations/attention_exporter.py
```

Suggested interface:

```python
def export_edge_attention(
    attention_output: EdgeAttentionOutput,
    *,
    edge_index: Tensor,
    edge_relation_type: Tensor,
    external_node_ids: Sequence[str],
    relation_registry: RelationRegistry,
    hazard_ids: Sequence[str],
    node_batch_index: Tensor,
    checkpoint_id: str,
    run_id: str,
    top_k_per_node: int | None = None,
) -> list[EdgeAttentionRecord]:
    ...
```

Records MUST include:

```text
source node
target node
relation
hazard
attention value
normalization scope
head index or head-reduction method
layer
checkpoint/run ID
registry versions
```

---

# 44. Pathway Exporter interface

## 44.1 Module

```text
explanations/pathway_exporter.py
```

Suggested interface:

```python
class PathwayExporter:
    def export(
        self,
        *,
        prediction_output: PredictionOutput,
        relation_gates: RelationGateOutput,
        edge_attention: EdgeAttentionOutput,
        temporal_attention: Tensor | None = None,
        uncertainty: UncertaintyOutput | None = None,
        metadata: TypedRunMetadata | None = None,
    ) -> PathwayExplanationBatch:
        ...
```

Possible outputs:

```text
top relation families
top specific edges
top neighboring nodes
top history periods
pathway scores
prediction summary
uncertainty summary
```

The exporter MUST NOT modify predictions.

---

# 45. Counterfactual interface

## 45.1 Module

```text
explanations/counterfactuals.py
```

Suggested operations:

```text
hazard swap
relation-family removal
selected-edge masking
memory removal
real-to-random topology replacement
scenario-feature alteration
```

Suggested interface:

```python
class CounterfactualRunner:
    def run_hazard_swap(
        self,
        model: nn.Module,
        batch: UrbanGraphBatch,
        *,
        alternative_hazard_ids: Tensor,
    ) -> CounterfactualResult:
        ...
```

Output:

```text
baseline prediction
counterfactual prediction
prediction delta
relation-gate delta
attention/pathway delta
alignment metadata
```

These are model counterfactual diagnostics, not causal effects by default.

---

# 46. Benchmark Adapter interface

## 46.1 Module

```text
data/benchmark_adapters.py
```

Responsibility:

```text
map benchmark artifacts into V2 contracts
enforce time split and feature-availability rules
preserve external IDs
preserve benchmark lineage metadata
```

Suggested protocol:

```python
class BenchmarkAdapter(Protocol):
    def load(self, config: AdapterConfig) -> UrbanGraphDataset:
        ...
```

Possible implementations:

```text
MontrealTractBenchmarkAdapter
QuebecCDBenchmarkAdapter
G1ArtifactAdapter
```

The adapter MAY know benchmark file paths and columns.

The core model MUST NOT.

---

# 47. Dataset interface

## 47.1 Module

```text
data/datasets.py
```

Suggested interface:

```python
class UrbanGraphDataset(Dataset):
    def __len__(self) -> int:
        ...

    def __getitem__(self, index: int) -> UrbanGraphExample:
        ...
```

A dataset example MUST preserve:

```text
external IDs
graph/scenario identity
temporal origin and cutoff information
hazard context
node features/history
edges and relation types
optional supervision
contract versions
```

The dataset SHOULD return CPU data.

Device movement belongs elsewhere.

---

# 48. Batch Collator interface

## 48.1 Module

```text
data/batch_collators.py
```

Suggested interface:

```python
class UrbanGraphBatchCollator:
    def __call__(
        self,
        examples: Sequence[UrbanGraphExample],
    ) -> UrbanGraphBatch:
        ...
```

Responsibilities:

```text
pad temporal histories
construct history masks
pack nodes
construct node_batch_index
optionally construct graph_ptr
offset edge indices
validate within-graph edges
align graph-level hazard context
build optional supervision
preserve external identifiers
merge temporal metadata
```

The collator MUST verify that edge endpoints remain valid after offsetting.

---

# 49. Graph Loader interface

## 49.1 Module

```text
data/graph_loaders.py
```

Responsibilities:

```text
load node and edge tables
map external IDs to contiguous internal indices
validate edge endpoints
validate relation IDs
construct edge attributes
preserve semantic edge weights
enforce temporal edge validity
```

It MUST NOT reinterpret relation semantics outside the registry.

---

# 50. Feature Builder interface

## 50.1 Module

```text
data/feature_builders.py
```

Suggested interface:

```python
class FeatureBuilder:
    def fit(self, training_frame: Any) -> "FeatureBuilder":
        ...

    def transform(self, frame: Any) -> FeatureBatch:
        ...

    def fit_transform(self, training_frame: Any) -> FeatureBatch:
        ...
```

Responsibilities:

```text
feature selection
training-only scaling
missing-value handling
history-window construction
feature-availability validation
feature-name and order recording
typed node-feature grouping
```

It MUST NOT fit transformations on validation or test data.

---

# 51. Loss interface

## 51.1 Module

```text
training/losses.py
```

Suggested interface:

```python
class LossComputer:
    def __call__(
        self,
        model_output: ModelOutput,
        batch: UrbanGraphBatch,
    ) -> LossOutput:
        ...
```

`LossOutput` SHOULD contain:

```text
total_loss
prediction_loss
uncertainty_loss
reporting_loss
gate_regularization
attention_regularization
other_terms
```

All component weights MUST be configuration-driven and logged.

Supervised loss computation MUST reject missing targets.

---

# 52. Trainer interface

## 52.1 Module

```text
training/trainer.py
```

Suggested interface:

```python
class Trainer:
    def fit(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        validation_loader: DataLoader,
    ) -> TrainingResult:
        ...

    def evaluate(
        self,
        model: nn.Module,
        data_loader: DataLoader,
    ) -> EvaluationResult:
        ...
```

Responsibilities:

```text
device movement
forward pass
loss computation
backpropagation
gradient clipping
optimizer step
validation
early stopping
checkpointing
logging
contract-version recording
```

The trainer MUST NOT define model architecture.

---

# 53. Evaluation interface

## 53.1 Module

```text
training/evaluation.py
```

Suggested interface:

```python
def evaluate_predictions(
    prediction_output: PredictionOutput,
    targets: Tensor,
    target_mask: Tensor,
    *,
    alignment: PredictionAlignment,
) -> MetricBundle:
    ...
```

Required metric families:

```text
MAE
RMSE
mean Poisson deviance
Spearman
Kendall
NDCG@k
top-k overlap
```

Future metrics MAY include:

```text
uncertainty calibration
interval coverage
relation-gate stability
explanation stability
hazard-gate divergence
counterfactual consistency
```

Metrics MUST remain aligned to the declared target and forecast horizon.

---

# 54. Predictor interface

## 54.1 Module

```text
inference/predictor.py
```

Suggested interface:

```python
class Predictor:
    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: Path,
        *,
        device: str | torch.device = "cpu",
    ) -> "Predictor":
        ...

    def predict(
        self,
        batch: UrbanGraphBatch,
        *,
        return_explanations: bool = False,
    ) -> PredictionResponse:
        ...
```

The predictor MUST:

```text
load configuration and registry versions
validate compatibility
validate inference schema
run inference
detach outputs
preserve external-node alignment
return structured responses
```

It MUST NOT require targets.

---

# 55. Explanation API interface

## 55.1 Module

```text
inference/explanation_api.py
```

Suggested interface:

```python
class ExplanationAPI:
    def explain_node(
        self,
        prediction_response: PredictionResponse,
        node_id: str,
        *,
        top_k_relations: int = 5,
        top_k_edges: int = 10,
        top_k_history_periods: int = 5,
    ) -> NodeExplanation:
        ...
```

The API SHOULD return domain-facing objects rather than raw tensors.

---

# 56. UI Payload interface

## 56.1 Module

```text
inference/ui_payloads.py
```

Suggested interface:

```python
def build_ui_payload(
    prediction_response: PredictionResponse,
    *,
    include_debug_fields: bool = False,
) -> dict[str, Any]:
        ...
```

Expected high-level structure:

```text
model
scenario
predictions
explanations
uncertainty
metadata
```

UI payload creation MUST NOT modify model outputs.

Debug fields SHOULD be disabled by default.

---

# 57. Experiment registry interface

## 57.1 Package

```text
experiments/
```

Suggested functions:

```python
def register_experiment(
    name: str,
    config: ExperimentConfig,
) -> None:
    ...

def get_experiment(name: str) -> ExperimentConfig:
    ...

def list_experiments() -> tuple[str, ...]:
    ...
```

Each experiment SHOULD declare:

```text
memory type
hazard conditioning mode
relation gate scope and type
attention mode
normalization mode
graph topology
prediction head
random seed
contract versions
expected output directory
```

Minimum named experiments:

```text
tabular_feature_parity
no_edge_neural
random_edge_graph
knn_graph
real_adjacency_graph
hazard_embedding_only
hazard_conditioned_gate
hazard_conditioned_gate_attention
hazard_queried_memory
full_v2
```

---

# 58. Explanation and intermediate-state policy

The model forward pass SHOULD default to:

```python
return_explanations=False
return_intermediate_states=False
```

When explanations are disabled:

```text
large edge-level traces SHOULD not be retained
memory overhead SHOULD remain minimal
```

When enabled:

```text
relation gates MUST be preserved
edge attention MUST be preserved
temporal attention SHOULD be preserved when available
layer IDs MUST be preserved
registry and checkpoint versions MUST be preserved
```

Intermediate hidden-state retention MUST be independently configurable.

---

# 59. Device policy

Neural modules MUST NOT force tensors onto hard-coded devices.

Device movement belongs in:

```text
trainer
predictor
calling application
```

All tensors in one forward call MUST be device-compatible.

---

# 60. Serialization policy

Research and inference outputs SHOULD be convertible to:

```text
JSON
CSV
Parquet
```

without requiring live GPU tensors.

Checkpoint metadata MUST include the contract and registry versions needed to reconstruct model semantics.

---

# 61. Error-handling policy

Contract violations MUST fail early with descriptive errors.

Good:

```text
edge_relation_type contains unknown relation ID 14.
Known IDs for registry version 0.1 are 0–8.
```

Bad:

```text
index out of range
```

Modules SHOULD validate:

```text
rank
shape
dtype
device compatibility
index bounds
graph membership
registered IDs
finite values
mask compatibility
temporal causality
schema compatibility
```

---

# 62. Interface stability policy

## 62.1 Stable contracts

These SHOULD change rarely:

```text
external-node alignment
node_batch_index semantics
edge-index convention
relation IDs
hazard IDs
temporal causality invariants
prediction semantics
forecast-horizon semantics
explanation record identity fields
schema and registry versions
```

## 62.2 Experimental internals

These MAY evolve:

```text
attention equation
gate activation
relation-transform strategy
normalization method
memory architecture
node-update equation
uncertainty implementation
head architecture
```

Changes to stable contracts MUST require:

```text
documentation update
version increment
migration note
test update
```

---

# 63. Minimum V2.0 implementation

Required:

```text
UrbanGraphBatch
PredictionAlignment
ModelOutput
contract-version metadata
LagMemoryEncoder
HazardEmbedding
HazardQueryEncoder
RelationFamilyGate
simple or uniform EdgeAttention
RelationTransforms
EdgeNormalizer
MessageBuilder
MessageAggregator
HazardConditionedFunctionalMessagePassingLayer
PredictionHead
basic RelationGateExporter
basic AttentionExporter
```

Optional placeholders:

```text
TransformerMemoryEncoder
UncertaintyHead
ReportingBiasHead
CounterfactualRunner
full PathwayExporter
UI payload builder
```

Unimplemented scientific modules MUST raise:

```python
NotImplementedError
```

They MUST NOT return fabricated placeholder predictions or explanations.

---

# 64. Required shape and contract tests

At minimum:

```text
UrbanGraphBatch accepts valid shapes.
UrbanGraphBatch rejects invalid edge endpoints.
node_batch_index correctly maps nodes to graphs.
graph-level hazard context broadcasts through node_batch_index.
cross-graph edges are rejected by default.
targets are required in supervised mode.
targets may be absent in inference mode.
temporal leakage invariants are enforced.
feature-availability cutoff is enforced.
heterogeneous node projectors return packed [N, H].
memory encoder returns [N, H_memory].
hazard encoder returns node-aligned [N, H_hazard].
relation gate returns one score per configured relation.
target-node gate maps correctly onto incoming edges.
edge attention returns one score per edge.
message builder returns [E, H].
aggregator returns [N, H].
functional layer preserves [N, H].
prediction head returns [N, K].
predictions preserve external-node alignment.
masked history periods receive zero temporal attention.
unknown relation IDs are rejected.
incompatible registry versions are rejected.
no-edge graphs do not crash.
isolated nodes do not produce NaN values.
explanation records reference valid nodes and relations.
attention exports record head-reduction policy.
```

---

# 65. End-to-end single-graph example

Assume:

```text
B = 1
N = 98 census divisions
E = 420 directed edges
T = 12 months
F_static = 30
F_history = 8
R = 6 relation families
H = 64
K = 1
```

Batch:

```text
external_node_ids: 98 strings
node_batch_index:  [98] all zeros
node_features:      [98, 30]
history_sequences: [98, 12, 8]
history_mask:      [98, 12]
hazard_ids:        [1]
edge_index:        [2, 420]
edge_relation_type:[420]
edge_attributes:   [420, 4]
targets:           optional [98, 1]
target_mask:       optional [98, 1]
```

Temporal contract:

```text
origin_time:                2024-12-01
history_start_time:         2024-01-01
history_end_time:           2024-12-01
feature_availability_cutoff:2024-12-01
target_start_time:          2025-01-01
target_end_time:            2025-03-31
forecast_horizon:           next_3_months
```

Internal representations:

```text
node_state:          [98, 64]
memory_state:        [98, 64]
hazard_context:      [98, 64]
hazard_memory_state: [98, 64]
relation_gates:      [98, 6]
attention_weight:    [420]
```

Final prediction:

```text
prediction_mean: [98, 1]
```

---

# 66. End-to-end multi-graph example

Assume two independent graph/scenario instances:

```text
B = 2
graph 0 nodes: 98
graph 1 nodes: 98
N = 196
```

Graph membership:

```text
node_batch_index:
  first 98 values = 0
  next 98 values  = 1
```

Hazards:

```text
hazard_ids:
  graph 0 = flood
  graph 1 = heat
```

Graph-level hazard contexts:

```text
graph_hazard_context: [2, H_hazard]
```

Node-aligned contexts:

```python
node_hazard_context = graph_hazard_context[node_batch_index]
```

This produces:

```text
first 98 nodes receive flood context
next 98 nodes receive heat context
```

Edges MUST remain within their graph instance unless an explicit cross-graph relation is registered.

---

# 67. Heterogeneous-node example

Suppose:

```text
N_tract = 540
N_water = 80
N_green = 120
N = 740
```

Raw features:

```text
tract_features: [540, F_tract]
water_features: [80, F_water]
green_features: [120, F_green]
```

Typed projection:

```text
tract_features → tract projector → [540, H]
water_features → water projector → [80, H]
green_features → green projector → [120, H]
```

Packed state:

```text
node_state: [740, H]
node_type:  [740]
```

Functional message passing sees the shared hidden state and typed relations.

It does not require all raw node types to share one feature width.

---

# 68. Ownership summary

| Module                           | Scientific question                                             |
| -------------------------------- | --------------------------------------------------------------- |
| Input Projector                  | How are heterogeneous raw entities represented in shared space? |
| Urban Memory Encoder             | What does the node remember?                                    |
| Temporal Attention               | Which past periods matter?                                      |
| Hazard Query Encoder             | Under what hazard and scenario is the model reasoning?          |
| Hazard-Queried Memory            | Which memories are relevant to this hazard?                     |
| Relation-Family Gate             | Which mechanism families are active?                            |
| Edge-Level Attention             | Which specific connections matter?                              |
| Relation Transform               | How does information change through each mechanism?             |
| Edge Normalizer                  | How is structural scale controlled?                             |
| Message Builder                  | How are routing components combined?                            |
| Aggregator                       | How are incoming messages combined?                             |
| Functional Message-Passing Layer | How does hazard-conditioned information move through the city?  |
| Prediction Head                  | What burden or risk is predicted?                               |
| Uncertainty Head                 | How uncertain is the prediction?                                |
| Reporting-Bias Head              | How much may reflect reporting propensity?                      |
| Pathway Exporter                 | How is model reasoning summarized?                              |
| Counterfactual Runner            | How does output change when hazard or pathways change?          |

---

# 69. Final interface principle

The package must remain understandable at three levels.

## Scientific level

```text
memory
hazard
functional pathway
prediction
uncertainty
explanation
```

## Module level

```text
explicit responsibility
explicit inputs
explicit outputs
clear ownership
matching ablation
```

## Tensor and contract level

```text
known shapes
known dtypes
known graph membership
known temporal availability
known relation semantics
known gate scope
known registry versions
known prediction alignment
```

The model may become mathematically sophisticated.

Its interfaces must remain simple enough that a future researcher or ML engineer can return from a technically deep implementation session and immediately recover:

```text
what each module means
what information it consumes
what it is allowed to produce
how it is validated
how it is compared against a simpler control
```

That is the purpose of this contract.
