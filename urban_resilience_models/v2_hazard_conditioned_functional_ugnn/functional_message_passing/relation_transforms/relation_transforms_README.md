# Relation Transforms Subsystem

**Package:** `urban_resilience_models.v2_hazard_conditioned_functional_ugnn.functional_message_passing.relation_transforms`  
**Primary public entry point:** `RelationTransforms`  
**Low-level implementations:** `SharedRelationTransform`, `PerRelationTransform`  
**Input contract:** `FunctionalMessagePassingInputs`  
**Output contract:** `RelationTransformOutput`  
**Registry dependency:** `CompiledRelationRegistry`  
**Purpose:** gather the source-node state for every stored edge and transform it either with one globally shared linear map or with an independently parameterized map for each compiled relation.

---

## 1. Overview

The relation-transforms subsystem is the first trainable edge-level stage of functional message passing.

It receives a fused node representation and a graph whose edges are aligned to a compiled relation registry. For each directed edge, it gathers the source node's hidden state and maps that state into the message space used by later stages.

The package implements two bounded transformation modes:

1. **Shared transform** — every relation uses the same `Linear(H, H)` map.
2. **Per-relation transform** — each dense compiled relation owns an independent `Linear(H, H)` map.

The subsystem deliberately stops after source-state transformation. It does **not** apply structural normalization, hazard-conditioned relation gates, edge attention, semantic edge weights, aggregation, residual connections, layer normalization, or dropout.

```text
NodeStateFusionOutput.fused_state [N, H]
                    |
                    | source_index [E]
                    v
       gather source node state [E, H]
                    |
                    v
          relation transform stage
         /                         \
        /                           \
 shared Linear(H,H)       relation-specific Linear(H,H)
        \                           /
         \                         /
          transformed source [E, H]
                    |
                    v
          RelationTransformOutput
                    |
                    v
 normalization -> gate -> attention -> message builder -> aggregation
```

### Role

The shared mode is the controlled baseline. It asks whether message passing can benefit from graph structure when every relation uses the same source-state map.

The per-relation mode is the functional alternative. It allows spatial adjacency, temporal memory, hydrological exposure, service access, infrastructure dependency, placebo controls, and other compiled relations to learn different transformations.

This separation supports a direct ablation:

```text
shared transform
    versus
independent transform per relation
```

A performance gain from the per-relation mode can therefore be attributed more specifically to relation-conditioned parameterization rather than to the mere presence of edges.

---

## 2. Package layout and ownership boundaries

```text
functional_message_passing/
└── relation_transforms/
    ├── __init__.py
    ├── shared_transform.py
    ├── per_relation_transform.py
    └── relation_transforms.py
```

| File | Owns | Does not own |
|---|---|---|
| `shared_transform.py` | One shared source-state linear transform, source gathering, strict tensor validation, architecture and parameter fingerprints | Relation lookup, per-relation parameters, registry compatibility, metadata-bearing output construction, gates, attention, normalization, aggregation |
| `per_relation_transform.py` | Deterministic compiled relation ordering, one independent linear map per relation, relation lookup, source gathering, dense relation-index parameter selection, relation-specific fingerprints | Registry compilation, transform-mode dispatch, output schema construction, gates, priors, attention, aggregation |
| `relation_transforms.py` | Public mode dispatcher, config construction, exact registry compatibility, full FMP input validation, metadata-preserving `RelationTransformOutput`, global and relation-specific identities | Mathematical implementation of each transform family, registry compilation, graph construction, message multiplication, aggregation |
| `__init__.py` | Stable package exports and schema-version exports | Runtime construction, device movement, registry compilation, side effects |

Recommended dependency direction:

```text
shared_transform.py -----------+
                               |
per_relation_transform.py -----+--> relation_transforms.py --> public callers
                               |
CompiledRelationRegistry ------+

__init__.py re-exports the three public classes and schema constants.
```

---

## 3. Position in the functional message-passing layer

The complete conceptual message equation is broader than this package:

```text
message[e] =
    relation_transform(source_state[e])
  × structural_normalization[e]
  × relation_gate[e]
  × edge_attention[e]
  × optional_semantic_edge_weight[e]
```

This package owns only the first factor:

```text
u_e = relation_transform(h_{s_e})
```

where:

- `e` is a stored directed edge;
- `s_e` is the source-node index of edge `e`;
- `h_{s_e}` is the fused hidden state of the source node;
- `u_e` is the transformed source state aligned to edge `e`.

Keeping this stage isolated has several benefits:

- transform ablations remain interpretable;
- gates and attention cannot be hidden inside an opaque relation layer;
- intermediate tensors can be exported and audited;
- source-state transformation can be tested independently from scatter and aggregation logic;
- architecture and parameter fingerprints can identify exactly which transform generated an edge representation.

---

## 4. Core notation and tensor contract

Let:

- `N` = number of nodes across the packed batch;
- `E` = number of stored directed edges;
- `H` = hidden dimension;
- `R` = number of relations in the compiled registry.

The relevant tensors are:

| Tensor | Shape | Dtype | Meaning |
|---|---:|---|---|
| `node_state` | `[N, H]` | floating | Fused node representation |
| `source_index` | `[E]` | `torch.long` | Source node for every stored edge |
| `edge_relation_index` | `[E]` | `torch.long` | Dense compiled relation index for every edge |
| transformed source state | `[E, H]` | same floating dtype as module | Output of the relation-transform stage |

Index ranges:

```text
0 <= source_index[e] < N
0 <= edge_relation_index[e] < R
```

All relevant tensors and trainable parameters must share one device. Floating input dtype must exactly equal parameter dtype. The modules perform no hidden casting and no implicit device transfer.

### Stored edge direction

For an edge stored as:

```text
source_index[e] -> target_index[e]
```

this package uses only `source_index[e]` to gather the state being transformed. The target index becomes relevant later for gating, attention grouping, and aggregation.

---

## 5. Mathematical definitions

### 5.1 Shared mode

For every edge `e`:

```text
h_source[e] = node_state[source_index[e]]

u[e] = W_shared h_source[e] + b_shared
```

The same matrix and optional bias are applied regardless of:

- relation identity;
- relation family;
- hazard;
- graph membership;
- control or placebo status;
- edge attributes;
- source or target node type.

### 5.2 Per-relation mode

For every edge `e` with dense relation index `r_e`:

```text
h_source[e] = node_state[source_index[e]]

u[e] = W[r_e] h_source[e] + b[r_e]
```

Each compiled relation owns an independent `Linear(H, H)` module.

The dense relation index is interpreted through exact ordered metadata:

```text
r
→ relation_names[r]
→ stable_relation_ids[r]
→ relation_module_keys[r]
→ relation_transforms[relation_module_keys[r]]
```

Stable ontology IDs are never used directly as tensor indices.

### 5.3 No hidden operations

Neither mode includes:

```text
activation
normalization
dropout
gating
attention
edge weighting
aggregation
residual update
```

The output is an affine transformation only.

---

## 6. Identity model: dense index, stable ID, name, and module key

The per-relation implementation preserves four related identities.

### 6.1 Dense relation index

A `CompiledRelationRegistry` assigns contiguous runtime indices:

```text
0, 1, ..., R - 1
```

`edge_relation_index` uses this identity.

### 6.2 Stable relation ID

Each ontology relation has a stable sparse integer ID. Stable IDs identify the semantic concept across artifacts and releases, but are not tensor positions.

### 6.3 Canonical relation name

Each relation also has a canonical string name. Names support diagnostics, exports, human-readable fingerprints, and relation lookup.

### 6.4 State-dict-safe module key

`PerRelationTransform` derives a deterministic module key:

```text
relation_{relation_index:04d}_id_{stable_relation_id}
```

Example:

```text
relation_0000_id_200
relation_0001_id_300
relation_0002_id_710
```

The key includes both:

- dense axis position, because it determines tensor alignment;
- stable ontology ID, because it determines semantic identity.

This makes checkpoint keys more informative and reduces the risk that a parameter tensor is silently loaded under a different relation ordering.

---

## 7. Public API

Recommended import:

```python
from urban_resilience_models.v2_hazard_conditioned_functional_ugnn.functional_message_passing.relation_transforms import (
    RelationTransforms,
    SharedRelationTransform,
    PerRelationTransform,
)
```

Schema-version constants are also public:

```python
from urban_resilience_models.v2_hazard_conditioned_functional_ugnn.functional_message_passing.relation_transforms import (
    SHARED_RELATION_TRANSFORM_SCHEMA_VERSION,
    PER_RELATION_TRANSFORM_SCHEMA_VERSION,
    RELATION_TRANSFORMS_DISPATCHER_SCHEMA_VERSION,
)
```

### Recommended call boundary

Application and model code should normally call:

```python
output = relation_transforms(fmp_inputs)
```

This returns a `RelationTransformOutput` and preserves the complete upstream metadata contract.

Direct use of `SharedRelationTransform` or `PerRelationTransform` is appropriate for:

- focused unit tests;
- mathematical diagnostics;
- isolated benchmarks;
- implementation development;
- explicitly low-level tensor workflows where lineage preservation is handled elsewhere.

---

# 8. `shared_transform.py`

## 8.1 Purpose

`SharedRelationTransform` implements the simplest transform ablation: one affine map shared by every edge.

```python
module = SharedRelationTransform(
    hidden_dim=64,
    bias=True,
)
```

Its exact neural architecture is:

```text
Linear(H, H, bias=bias)
```

There is no activation, normalization, or dropout.

## 8.2 Constructor

```python
SharedRelationTransform(
    *,
    hidden_dim: int,
    bias: bool = True,
)
```

| Parameter | Contract |
|---|---|
| `hidden_dim` | Positive integer. Defines both input and output width. Boolean values are rejected. |
| `bias` | Strict Boolean. Controls whether the shared `nn.Linear` contains a bias. |

### Constructor failures

- non-integer or non-positive `hidden_dim` → `ValueError`;
- non-Boolean `bias` → `TypeError`.

## 8.3 Public properties

| Property | Meaning |
|---|---|
| `transform_mode` | Canonical shared-transform constant |
| `input_dim` | `hidden_dim` |
| `output_dim` | `hidden_dim` |
| `device` | Device of `linear.weight` |
| `dtype` | Dtype of `linear.weight` |
| `parameter_count` | Number of all parameters |
| `trainable_parameter_count` | Number of parameters with `requires_grad=True` |

Parameter count:

```text
with bias:    H² + H
without bias: H²
```

## 8.4 `gather_source_state()`

```python
source_state = module.gather_source_state(
    node_state,
    source_index,
)
```

Inputs:

```text
node_state   [N, H], floating
source_index [E],    torch.long
```

Output:

```text
source_state [E, H]
```

The method performs the same strict validation as `forward()` but does not apply trainable parameters. It is useful for focused tests and diagnostics.

Validation includes:

- `node_state` is a rank-2 tensor;
- width equals `hidden_dim`;
- dtype is floating;
- values are finite;
- `source_index` is a rank-1 `torch.long` tensor;
- source indices are in range;
- input tensors share one device;
- node state and module parameters share device and dtype.

## 8.5 `transform_source_state()`

```python
transformed = module.transform_source_state(
    source_state,
)
```

This lower-level method applies the shared affine map to a tensor already aligned to edges.

Input and output both have shape `[E, H]`.

It validates:

- rank and width;
- floating dtype;
- finiteness;
- device equality with parameters;
- dtype equality with parameters;
- exact output shape;
- finite output.

## 8.6 `forward()`

```python
transformed = module(
    node_state,
    source_index,
)
```

Execution order:

```text
validate inputs
→ gather node_state[source_index]
→ apply shared Linear(H, H)
→ validate shape and finiteness
→ return [E, H]
```

## 8.7 Empty-edge behavior

The following are valid:

```text
N > 0, E = 0
N = 0, E = 0
```

The result has shape:

```text
[0, H]
```

A nonempty `source_index` is invalid when `node_state` has zero rows.

## 8.8 Architecture identity

`architecture_dict()` returns a canonical JSON-compatible dictionary containing:

- schema version;
- transform mode;
- hidden dimension;
- bias flag;
- parameter-sharing policy;
- operation order;
- explicit absence of activation, normalization, and dropout.

`architecture_fingerprint()` hashes the canonical architecture dictionary.

The architecture fingerprint changes when an architectural contract changes, for example:

- `hidden_dim` changes;
- bias is enabled or disabled;
- the operation order changes;
- a new hidden operation is introduced.

It does not depend on learned parameter values.

## 8.9 Parameter identity

`parameter_fingerprint()` hashes the complete state dict, including:

- state-dict key;
- tensor dtype;
- tensor shape;
- exact tensor bytes.

The parameter fingerprint changes after parameter initialization changes or training updates.

## 8.10 Parameter diagnostics

```python
module.assert_finite_parameters()
```

This scans all named parameters and raises `ValueError` if any contains `NaN` or infinity.

Input validation alone does not guarantee finite parameters; call this diagnostic at model construction, checkpoint loading, or audit boundaries when corruption detection is required.

## 8.11 State-dict structure

Expected parameter keys:

```text
linear.weight
linear.bias          # only when bias=True
```

When nested under the dispatcher, keys become:

```text
implementation.linear.weight
implementation.linear.bias
```

## 8.12 Computational characteristics

| Quantity | Complexity |
|---|---:|
| Parameters | `O(H²)` |
| Source gathering | `O(EH)` |
| Linear transformation | `O(EH²)` |
| Output activation memory | `O(EH)` |

The shared mode is more parameter-efficient than the per-relation mode and is the preferred baseline for determining whether relation-specific parameterization is necessary.

---

# 9. `per_relation_transform.py`

## 9.1 Purpose

`PerRelationTransform` assigns an independent affine map to every compiled relation.

```python
module = PerRelationTransform(
    hidden_dim=64,
    relation_names=(
        "spatial_adjacency",
        "temporal_memory",
        "drainage_dependency",
    ),
    stable_relation_ids=(200, 300, 710),
    control_relation_mask=(False, False, False),
    bias=True,
)
```

The relation metadata order is the dense runtime axis.

## 9.2 Constructor

```python
PerRelationTransform(
    *,
    hidden_dim: int,
    relation_names: Sequence[str],
    stable_relation_ids: Sequence[int],
    control_relation_mask: Sequence[bool] | None = None,
    bias: bool = True,
)
```

| Parameter | Contract |
|---|---|
| `hidden_dim` | Positive integer; common input/output width `H` |
| `relation_names` | Nonempty sequence of unique nonempty strings in dense compiled order |
| `stable_relation_ids` | Unique nonnegative integers aligned exactly to `relation_names` |
| `control_relation_mask` | Optional Boolean sequence aligned to relations; defaults to all `False` |
| `bias` | Strict Boolean controlling every relation-specific linear bias |

The constructor creates exactly `R` independent `nn.Linear(H, H)` modules.

## 9.3 Construction from a compiled registry

Preferred low-level constructor:

```python
module = PerRelationTransform.from_compiled_registry(
    hidden_dim=64,
    compiled_relation_registry=compiled_registry,
    bias=True,
)
```

This method:

1. verifies the argument is a `CompiledRelationRegistry`;
2. calls `compiled_relation_registry.validate()`;
3. copies `relation_names` in dense compiled order;
4. copies `stable_relation_ids` in the same order;
5. derives the control mask from `entry.specification.is_control`;
6. constructs the per-relation modules.

This is safer than manually copying registry metadata.

## 9.4 Public metadata

| Attribute/property | Meaning |
|---|---|
| `relation_names` | Canonical names in dense runtime order |
| `stable_relation_ids` | Stable ontology IDs aligned to names |
| `control_relation_mask` | Boolean control metadata aligned to relations |
| `relation_module_keys` | Deterministic state-dict-safe keys |
| `relation_count` | Number of compiled relations `R` |
| `transform_mode` | Canonical per-relation-transform constant |
| `input_dim` / `output_dim` | `hidden_dim` |
| `relation_index_by_name` | Immutable name → dense index mapping |
| `relation_index_by_stable_id` | Immutable stable ID → dense index mapping |
| `device` / `dtype` | Derived from parameters |
| `parameter_count` | Total parameter count |
| `trainable_parameter_count` | Trainable parameter count |

The two lookup mappings are wrapped in `MappingProxyType` and cannot be mutated.

## 9.5 Relation lookup semantics

```python
index = module.relation_index("spatial_adjacency")
index = module.relation_index(200)
```

A string is interpreted as a canonical relation name.

An integer is interpreted as a **stable ontology ID**, never as an already-dense relation index.

To access a module by dense index, use:

```python
linear = module.module_for_relation_index(0)
```

To access it by name or stable ID, use:

```python
linear = module.module_for_relation("spatial_adjacency")
linear = module.module_for_relation(200)
```

This explicit distinction prevents accidental mixing of stable sparse IDs and dense runtime indices.

## 9.6 Module registration

The independent maps are registered in:

```python
module.relation_transforms  # nn.ModuleDict
```

The keys follow:

```text
relation_{dense_index:04d}_id_{stable_id}
```

The helper `OrderedRelationModules` exists only to make insertion-order intent explicit when constructing the `ModuleDict`. It adds no mathematical behavior.

## 9.7 `gather_source_state()`

```python
source_state = module.gather_source_state(
    node_state,
    source_index,
    edge_relation_index,
)
```

The output has shape `[E, H]`.

Although `edge_relation_index` does not affect source gathering itself, the method validates it so the returned edge-aligned state is already guaranteed to be compatible with the subsequent per-relation transform.

## 9.8 Stacked parameter views

### `stacked_weight()`

```python
weights = module.stacked_weight()
```

Returns:

```text
[R, H, H]
```

The relation axis exactly follows `relation_names` and `stable_relation_ids`.

The returned tensor remains connected to each registered parameter, so autograd can produce explicit zero gradients for relation groups absent from a batch.

### `stacked_bias()`

```python
biases = module.stacked_bias()
```

Returns:

```text
[R, H]
```

or `None` when `bias=False`.

## 9.9 `transform_source_state()`

```python
transformed = module.transform_source_state(
    source_state,
    edge_relation_index,
)
```

Inputs:

```text
source_state        [E, H]
edge_relation_index [E]
```

Internal vectorized operation:

```text
weights = stack(W_0, ..., W_{R-1})                 [R, H, H]
selected_weights = weights[edge_relation_index]    [E, H, H]
transformed = bmm(selected_weights, source[...,None]).squeeze(-1)
```

If bias is enabled:

```text
transformed += stacked_bias[edge_relation_index]
```

## 9.10 `forward()`

```python
transformed = module(
    node_state,
    source_index,
    edge_relation_index,
)
```

Execution order:

```text
validate node and edge indices
→ gather node_state[source_index]
→ stack relation parameters
→ select parameters by dense relation index
→ batched matrix-vector multiplication
→ add relation-specific bias if enabled
→ validate output
→ return [E, H]
```

## 9.11 Zero-edge batches

`E = 0` is valid.

The module returns `[0, H]` while preserving parameter connectivity through a mathematically zero-valued expression involving the stacked weights and optional biases.

This design ensures:

- backward can still run;
- all relation parameters remain in the autograd graph;
- absent relations receive explicit zero gradients rather than disappearing from the graph.

## 9.12 Relations absent from a nonempty batch

A compiled relation may have zero matching edges even when `E > 0`.

Its parameters remain registered and included in stacked parameter tensors. Since no edge selects them, their gradients are zero for that batch. They are not removed, remapped, or replaced by a fallback relation.

## 9.13 Control relations

Control/placebo relations receive ordinary independent linear parameters in per-relation mode.

`control_relation_mask` is metadata only. This module does not:

- zero control parameters;
- freeze them;
- exclude them from training;
- prevent their transformation;
- alter their initialization;
- suppress them from explanations.

Those policies belong to experiment configuration, registry capability, explanation policy, or downstream logic.

## 9.14 Per-relation architecture identities

```python
payload = module.relation_architecture_dict(relation_index)
fingerprint = module.relation_architecture_fingerprint(relation_index)
all_fingerprints = module.relation_architecture_fingerprints()
```

Each relation architecture payload includes:

- schema version;
- transform mode;
- dense relation index;
- relation name;
- stable relation ID;
- control flag;
- module key;
- hidden dimension;
- bias setting;
- exact operation type;
- explicit absence of activation, normalization, and dropout.

`relation_architecture_fingerprints()` returns an immutable mapping keyed by canonical relation name.

## 9.15 Per-relation parameter identities

```python
fingerprint = module.relation_parameter_fingerprint(relation_index)
all_fingerprints = module.relation_parameter_fingerprints()
```

A relation parameter fingerprint hashes only that relation's linear state dict.

This supports:

- relation-specific checkpoint audits;
- tracking which relation maps changed during training;
- comparing corresponding relations across runs;
- explanation artifact provenance;
- detecting accidental relation-module replacement.

## 9.16 Global architecture and parameter identity

`architecture_dict()` includes:

- all ordered relation metadata;
- all module keys;
- control mask;
- parameter-sharing policy;
- operation order;
- relation-level architecture fingerprints.

`architecture_fingerprint()` hashes that complete architecture.

`parameter_fingerprint()` hashes the complete state dict.

## 9.17 Parameter count

```text
with bias:    R × (H² + H)
without bias: R × H²
```

Compared with shared mode, per-relation parameters grow linearly with `R`.

## 9.18 State-dict structure

Example keys under the direct module:

```text
relation_transforms.relation_0000_id_200.weight
relation_transforms.relation_0000_id_200.bias
relation_transforms.relation_0001_id_300.weight
relation_transforms.relation_0001_id_300.bias
```

When nested under the public dispatcher:

```text
implementation.relation_transforms.relation_0000_id_200.weight
implementation.relation_transforms.relation_0000_id_200.bias
```

A different compiled relation order changes module keys and architecture identity. Checkpoints must therefore be loaded only with the intended registry contract.

## 9.19 Computational characteristics

| Quantity | Complexity |
|---|---:|
| Parameters | `O(RH²)` |
| Source gathering | `O(EH)` |
| Stack all relation weights | `O(RH²)` tensor construction/view cost |
| Select edge weights | `O(EH²)` activation memory |
| Batched transforms | `O(EH²)` |
| Output | `O(EH)` |

The vectorized implementation is simple and explicit, but selecting one full `[H, H]` matrix per edge can be memory intensive for large `E` and `H`. See the performance section for extension options.

---

# 10. `relation_transforms.py`

## 10.1 Purpose

`RelationTransforms` is the stable, metadata-preserving public entry point.

It owns the boundary between:

- configuration;
- compiled relation identity;
- full functional-message-passing inputs;
- one selected mathematical implementation;
- the typed `RelationTransformOutput` consumed downstream.

Public model code should normally use this class rather than invoke the low-level implementations directly.

## 10.2 Constructor

```python
RelationTransforms(
    *,
    mode: str,
    hidden_dim: int,
    compiled_relation_registry: CompiledRelationRegistry,
    bias: bool = True,
)
```

| Parameter | Contract |
|---|---|
| `mode` | Canonical relation-transform mode; surrounding whitespace is stripped |
| `hidden_dim` | Positive integer `H` |
| `compiled_relation_registry` | Valid nonempty compiled registry defining the exact dense relation axis |
| `bias` | Strict Boolean |

## 10.3 Mode validation

Mode validation distinguishes three cases:

| Case | Result |
|---|---|
| Known and implemented mode | Construct implementation |
| Known canonical but unavailable mode | `NotImplementedError` |
| Unknown mode | `ValueError` |

Non-string values raise `TypeError`; blank strings raise `ValueError`.

Use central constants instead of hard-coded strings:

```python
from urban_resilience_models.v2_hazard_conditioned_functional_ugnn.constants import (
    RELATION_TRANSFORM_SHARED,
    RELATION_TRANSFORM_PER_RELATION,
)
```

This avoids drift between configuration vocabulary and implementation vocabulary.

## 10.4 Registry capture

The constructor validates the compiled registry and stores:

```text
relation_names
stable_relation_ids
control_relation_mask
compiled_relation_registry_fingerprint
```

It verifies:

- at least one compiled relation exists;
- names, IDs, and entries have equal lengths;
- names are unique nonempty strings;
- stable IDs are unique nonnegative integers;
- control metadata aligns with the relation order.

Even shared mode retains the complete registry identity. Although shared mathematics do not select relation-specific parameters, the output still belongs to a specific graph ontology and dense relation axis.

## 10.5 Implementation dispatch

The selected implementation is registered under the stable attribute name:

```text
implementation
```

Mode mapping:

```text
shared       -> SharedRelationTransform
per-relation -> PerRelationTransform
```

The public property verifies that the registered child is one of the supported implementation classes.

## 10.6 Construction from configuration

```python
module = RelationTransforms.from_config(
    config=fmp_config,
    hidden_dim=hidden_dim,
    compiled_relation_registry=compiled_registry,
    bias=True,
)
```

The method:

1. requires `FunctionalMessagePassingConfig`;
2. calls `config.validate()`;
3. calls `config.assert_implemented()` when `config.enabled` is true;
4. reads `config.relation_transform_type`;
5. constructs the dispatcher with the supplied runtime hidden dimension and compiled registry.

The complete model builder remains responsible for resolving `hidden_dim` and compiling the registry.

## 10.7 Public properties

| Property | Meaning |
|---|---|
| `implementation` | Selected shared or per-relation child module |
| `input_dim` / `output_dim` | `hidden_dim` |
| `relation_count` | `R` |
| `device` / `dtype` | Delegated to implementation |
| `parameter_count` | Total dispatcher parameters |
| `trainable_parameter_count` | Trainable parameters |
| `is_shared` | Whether shared mode is active |
| `is_per_relation` | Whether per-relation mode is active |

## 10.8 Exact input compatibility

The dispatcher accepts only `FunctionalMessagePassingInputs`.

Before transformation it checks:

1. input object type;
2. hidden width equals `RelationTransforms.hidden_dim`;
3. input relation-name order equals the stored compiled order;
4. input stable relation IDs equal the stored IDs;
5. input compiled-registry fingerprint equals the stored fingerprint;
6. input control mask equals the stored control mask;
7. input device equals parameter device;
8. input node-state dtype equals parameter dtype.

This is stricter than checking only `R`. Two registries with the same relation count but different identities or ordering are rejected.

## 10.9 `transform_tensor()`

```python
transformed = module.transform_tensor(inputs)
```

This lower-level public method returns only `[E, H]` while still requiring the complete typed input contract.

Dispatch:

```text
shared:
    implementation(node_state.fused_state, source_index)

per relation:
    implementation(
        node_state.fused_state,
        source_index,
        edge_relation_index,
    )
```

Postconditions are checked defensively:

- exact shape `[num_edges, hidden_dim]`;
- unchanged device;
- unchanged dtype;
- finite values.

## 10.10 `forward()`

```python
output = module(inputs)
```

`forward()` calls `transform_tensor()` and constructs:

```python
RelationTransformOutput(
    transformed_source_state=transformed,
    source_inputs=inputs,
    transform_mode=module.mode,
    encoder_architecture_fingerprint=module.architecture_fingerprint(),
    parameter_fingerprint=module.parameter_fingerprint(),
    relation_parameter_fingerprints=(
        module.relation_parameter_fingerprints()
    ),
)
```

The complete source input object is retained by identity, not reconstructed.

## 10.11 Output semantics

`RelationTransformOutput` contains:

| Field | Meaning |
|---|---|
| `transformed_source_state` | Edge-aligned transformed states `[E, H]` |
| `source_inputs` | Exact `FunctionalMessagePassingInputs` object used |
| `transform_mode` | Selected mode |
| `encoder_architecture_fingerprint` | Dispatcher architecture identity |
| `parameter_fingerprint` | Exact full parameter-state identity |
| `relation_parameter_fingerprints` | Immutable relation-name → fingerprint mapping in per-relation mode |
| `schema_version` | Output schema identity |

Shared mode returns an empty relation-specific fingerprint mapping because one parameter set is shared by all relations.

The output schema validates shape, device, dtype, finiteness, identity strings, and that relation-specific fingerprint keys belong to the compiled registry.

## 10.12 Architecture identity

`architecture_dict()` records:

- dispatcher schema version;
- selected mode;
- hidden dimension;
- bias flag;
- complete relation names, stable IDs, and control mask;
- compiled-registry fingerprint;
- implementation class name;
- implementation architecture dictionary;
- relation-specific architecture fingerprints;
- output schema name;
- exact operation order.

The architecture fingerprint therefore binds the model not only to the transform mode, but also to the compiled relation axis.

## 10.13 Parameter identity

`parameter_fingerprint()` hashes the dispatcher state dict.

`relation_parameter_fingerprints()`:

- returns an immutable empty mapping in shared mode;
- delegates to `PerRelationTransform` in per-relation mode.

## 10.14 Finite-parameter assertion

```python
module.assert_finite_parameters()
```

The dispatcher delegates to the implementation and raises if a parameter contains `NaN` or infinity.

## 10.15 Stable state-dict boundary

The child module is always registered under `implementation`, giving a predictable top-level checkpoint namespace.

Shared example:

```text
implementation.linear.weight
implementation.linear.bias
```

Per-relation example:

```text
implementation.relation_transforms.relation_0000_id_200.weight
implementation.relation_transforms.relation_0000_id_200.bias
...
```

---

# 11. `__init__.py`

## 11.1 Purpose

The package initializer defines the intended public surface.

It exports:

```text
SharedRelationTransform
PerRelationTransform
RelationTransforms
SHARED_RELATION_TRANSFORM_SCHEMA_VERSION
PER_RELATION_TRANSFORM_SCHEMA_VERSION
RELATION_TRANSFORMS_DISPATCHER_SCHEMA_VERSION
```

## 11.2 Import behavior

Importing the package performs no:

- module construction;
- registry compilation;
- parameter initialization beyond normal class definition;
- device movement;
- graph loading;
- filesystem access;
- training or inference.

## 11.3 Why the export boundary matters

External code should prefer package-level imports. Internal implementation files may evolve while the package export surface remains stable.

`__all__` also documents the supported public symbols and prevents helper functions from becoming accidental API commitments.

---

# 12. Upstream contract: `FunctionalMessagePassingInputs`

The dispatcher consumes the complete functional-message-passing input contract rather than loose tensors.

Relevant information includes:

```text
source_graph
node_state
compiled_relation_registry
relation_names
stable_relation_ids
control_relation_mask
source_index
target_index
edge_relation_index
node_batch_index
edge_batch_index
optional edge attributes
optional semantic edge weights
optional relation-family alignment
optional hazard query
optional compiled relation priors
lineage and value fingerprints
```

For this package, the most directly used views are:

```python
inputs.node_state.fused_state
inputs.source_index
inputs.edge_relation_index
inputs.relation_names
inputs.stable_relation_ids
inputs.control_relation_mask
inputs.compiled_relation_registry
inputs.hidden_dim
inputs.num_edges
inputs.device
inputs.dtype
```

The broader metadata remains attached to `RelationTransformOutput.source_inputs` so downstream modules can verify they all consumed the same source contract.

### Why loose tensors are insufficient at the public boundary

A call such as:

```python
transform(node_state, source_index, relation_index)
```

cannot by itself prove:

- that relation indices match the intended registry;
- that stable IDs and names are aligned;
- that control metadata is preserved;
- that the graph and fused node state share node ordering;
- that later gates and attention use the same artifact lineage.

`RelationTransforms` therefore accepts only the typed input object.

---

# 13. Output contract: `RelationTransformOutput`

The output is immutable and metadata preserving.

Core shape:

```text
transformed_source_state [E, H]
```

The output retains the exact `FunctionalMessagePassingInputs` instance.

This enables downstream checks such as:

```text
relation transform, normalization, gate, and attention
must all reference the same source input object
```

Properties exposed by the schema include:

```text
num_edges
hidden_dim
```

The relation-specific parameter-fingerprint mapping is copied into an immutable mapping and may contain only names from the compiled registry.

---

# 14. Device and dtype policy

The package uses an explicit, fail-fast policy.

## 14.1 No automatic device movement

The modules do not call `.to(device)` on inputs.

Correct pattern:

```python
module = module.to(device)
inputs = inputs.to(device)  # using the appropriate upstream reconstruction helper
output = module(inputs)
```

A CPU/GPU mismatch raises `ValueError`.

## 14.2 No automatic dtype casting

The modules do not silently convert `float64` inputs to `float32`, or vice versa.

Correct pattern:

```python
module = module.double()
# upstream floating tensors must also be float64
```

Exact dtype equality makes mixed-precision behavior deliberate rather than implicit.

## 14.3 Index dtypes

```text
source_index        -> torch.long
edge_relation_index -> torch.long
```

Floating or Boolean index tensors are rejected.

## 14.4 Finite values

Inputs and outputs must not contain `NaN` or infinity. Parameters can be checked explicitly through `assert_finite_parameters()`.

---

# 15. Autograd behavior

Both implementations use ordinary PyTorch operations and preserve gradients to:

- the fused node state;
- shared or relation-specific weights;
- optional biases.

### Shared mode

Every edge contributes to the same parameter set.

### Per-relation mode

Each edge contributes only to the parameters selected by its dense relation index.

Consequently:

- relations present in a batch receive data-dependent gradients;
- compiled relations absent from a batch receive zero gradients;
- control relations are trainable if their edges are present and parameters require gradients;
- the zero-edge path remains backward-compatible and yields zero parameter gradients.

No parameter is detached during stacking or selection.

---

# 16. Empty and degenerate inputs

## 16.1 Zero edges

Both modes support:

```text
E = 0
```

and return:

```text
[0, H]
```

## 16.2 Zero nodes

Valid only when there are also zero edges:

```text
N = 0, E = 0
```

A nonempty source-index vector with `N = 0` is rejected.

## 16.3 Zero relations

The public dispatcher requires at least one compiled relation, even if a particular batch contains zero edges.

A graph topology representing no edges is not represented by an empty compiled relation ontology. The registry still defines the allowable relation axis for the model artifact.

## 16.4 Empty relation groups

In per-relation mode, some compiled relation groups may have no edges in a batch. This is valid and does not alter relation ordering.

---

# 17. Control and placebo relations

The compiled registry marks control relations through `control_relation_mask`.

This package preserves that metadata but does not impose special mathematics.

In shared mode, control and real edges use the same map.

In per-relation mode, a control relation receives its own independent map just like any other compiled relation.

This is deliberate: the graph control should be evaluated under comparable model capacity unless the experiment explicitly defines a different ablation.

Explanation filtering and interpretation remain downstream responsibilities.

---

# 18. Fingerprints and reproducibility

The subsystem distinguishes architecture identity from parameter identity.

## 18.1 Architecture fingerprint

Represents what the module **is**:

- mode;
- hidden width;
- bias setting;
- registry identity and ordering;
- parameter-sharing structure;
- operation order;
- schema versions.

It does not encode learned values.

## 18.2 Parameter fingerprint

Represents the exact current tensor state:

- key names;
- dtypes;
- shapes;
- bytes.

It changes after training or manual parameter modification.

## 18.3 Relation-specific fingerprints

Available only in per-relation mode. They permit one relation map to be audited without conflating it with every other map.

## 18.4 Fingerprint cost

Parameter fingerprinting detaches tensors, moves them to CPU, makes them contiguous, and hashes their bytes.

Therefore:

- it can synchronize accelerators;
- it can be expensive for large models;
- `forward()` currently computes the parameter fingerprint for the output contract;
- callers should account for this audit cost in performance-sensitive execution.

If future profiling shows this is too costly for the training hot path, any relaxation should be explicit and preserve an audit mode rather than silently dropping identity metadata.

---

# 19. Usage examples

The examples assume that graph construction, node-state fusion, and registry compilation have already produced valid typed objects.

## 19.1 Recommended construction from configuration

```python
from urban_resilience_models.v2_hazard_conditioned_functional_ugnn.functional_message_passing.relation_transforms import (
    RelationTransforms,
)

relation_transforms = RelationTransforms.from_config(
    config=model_config.message_passing,
    hidden_dim=model_config.hidden_dim,
    compiled_relation_registry=compiled_relation_registry,
    bias=True,
)

output = relation_transforms(fmp_inputs)

assert output.transformed_source_state.shape == (
    fmp_inputs.num_edges,
    fmp_inputs.hidden_dim,
)
assert output.source_inputs is fmp_inputs
```

## 19.2 Explicit shared baseline

```python
from urban_resilience_models.v2_hazard_conditioned_functional_ugnn.constants import (
    RELATION_TRANSFORM_SHARED,
)
from urban_resilience_models.v2_hazard_conditioned_functional_ugnn.functional_message_passing.relation_transforms import (
    RelationTransforms,
)

shared = RelationTransforms(
    mode=RELATION_TRANSFORM_SHARED,
    hidden_dim=64,
    compiled_relation_registry=compiled_relation_registry,
    bias=True,
)

shared_output = shared(fmp_inputs)
```

## 19.3 Explicit per-relation transform

```python
from urban_resilience_models.v2_hazard_conditioned_functional_ugnn.constants import (
    RELATION_TRANSFORM_PER_RELATION,
)

per_relation = RelationTransforms(
    mode=RELATION_TRANSFORM_PER_RELATION,
    hidden_dim=64,
    compiled_relation_registry=compiled_relation_registry,
    bias=True,
)

per_relation_output = per_relation(fmp_inputs)
```

## 19.4 Tensor-only diagnostic call

```python
edge_state = per_relation.transform_tensor(
    fmp_inputs
)
```

This still validates full metadata but omits output-schema construction.

## 19.5 Direct shared-transform diagnostic

```python
from urban_resilience_models.v2_hazard_conditioned_functional_ugnn.functional_message_passing.relation_transforms import (
    SharedRelationTransform,
)

transform = SharedRelationTransform(
    hidden_dim=64,
    bias=False,
).to(node_state.device)

edge_state = transform(
    node_state,
    source_index,
)
```

## 19.6 Direct per-relation construction from registry

```python
from urban_resilience_models.v2_hazard_conditioned_functional_ugnn.functional_message_passing.relation_transforms import (
    PerRelationTransform,
)

transform = PerRelationTransform.from_compiled_registry(
    hidden_dim=64,
    compiled_relation_registry=compiled_relation_registry,
    bias=True,
)

edge_state = transform(
    node_state,
    source_index,
    edge_relation_index,
)
```

## 19.7 Inspect one relation module

```python
relation_index = transform.relation_index(
    "drainage_dependency"
)
linear = transform.module_for_relation_index(
    relation_index
)

print(linear.weight.shape)
print(
    transform.relation_architecture_fingerprint(
        relation_index
    )
)
```

## 19.8 Lookup by stable ontology ID

```python
relation_index = transform.relation_index(
    stable_relation_id
)
```

Do not pass a dense index to `relation_index()`. Integers are interpreted as stable IDs.

## 19.9 Save and load a checkpoint

```python
payload = {
    "state_dict": relation_transforms.state_dict(),
    "architecture_fingerprint": (
        relation_transforms.architecture_fingerprint()
    ),
    "compiled_relation_registry_fingerprint": (
        relation_transforms
        .compiled_relation_registry_fingerprint
    ),
}

torch.save(payload, checkpoint_path)
```

On load:

```python
loaded = torch.load(
    checkpoint_path,
    map_location="cpu",
)

if loaded["architecture_fingerprint"] != (
    relation_transforms.architecture_fingerprint()
):
    raise ValueError(
        "Checkpoint relation-transform architecture mismatch."
    )

relation_transforms.load_state_dict(
    loaded["state_dict"],
    strict=True,
)
relation_transforms.assert_finite_parameters()
```

## 19.10 Zero-edge batch

```python
output = relation_transforms(empty_edge_fmp_inputs)
assert output.transformed_source_state.shape == (
    0,
    relation_transforms.hidden_dim,
)
```

---

# 20. Integration with downstream message construction

A typical downstream message builder consumes:

```text
RelationTransformOutput
StructuralEdgeNormalizationOutput
optional RelationGateOutput
optional EdgeAttentionOutput
optional semantic edge weight
```

The transform output should be multiplied by scalar edge factors, usually through explicit broadcasting:

```text
edge_messages[e, :] =
    transformed_source_state[e, :]
  × normalization[e]
  × gate[e]
  × attention[e]
  × semantic_weight[e]
```

Disabled mechanisms should contribute the exact multiplicative identity `1`, not alter the relation transform.

The relation transform should not be recomputed independently by the gate, attention, or aggregator modules.

---

# 21. Ablation guidance

## 21.1 Shared versus per-relation

This is the primary transform ablation.

```text
shared:
  tests whether one generic message projection is sufficient

per relation:
  tests whether relation-specific source interpretation adds value
```

Hold the following constant when comparing them:

- compiled registry;
- graph edges;
- hidden width;
- bias choice;
- normalization;
- gating;
- attention;
- aggregation;
- training schedule;
- random seeds where practical.

## 21.2 Parameter-count confounding

Per-relation mode has approximately `R` times more transform parameters than shared mode.

A fair study should report parameter counts and may include additional controls such as:

- width-matched or parameter-matched shared baselines;
- low-rank relation transforms;
- basis-decomposed transforms;
- shared transform plus relation embedding;
- regularized per-relation transforms.

## 21.3 Interpretation caution

Different learned relation matrices do not by themselves prove causal mechanisms.

They show that the predictive model benefited from relation-conditioned parameterization under the supplied graph, features, targets, and training procedure.

Relation gates and edge attention provide separate evidence about activation and neighbor importance.

---

# 22. Common failure modes

| Failure | Likely cause | Resolution |
|---|---|---|
| `Unknown relation transform mode` | Mode string is outside canonical vocabulary | Use constants from `constants.py` |
| Canonical mode raises `NotImplementedError` | Configuration selected a recognized future mode | Implement and test a dedicated module before enabling it |
| Hidden-width mismatch | Fused node width differs from transform `hidden_dim` | Resolve one model hidden width during construction |
| Relation ordering differs | Inputs were built from a different compiled registry or reordered metadata | Rebuild FMP inputs from the exact compiled registry |
| Stable relation IDs differ | Names may match but ontology identity does not | Preserve compiled registry metadata end to end |
| Registry fingerprint mismatch | Stale or independently recompiled registry | Persist and reuse the intended compiled registry artifact |
| Control mask mismatch | Input control metadata and registry disagree | Derive masks from compiled entries rather than manually copying |
| Device mismatch | Inputs and parameters are on different devices | Move both explicitly before calling |
| Dtype mismatch | Inputs and parameters use different floating dtypes | Cast module and upstream floating tensors deliberately |
| Out-of-range source index | Edge endpoints do not match node axis | Validate graph artifact before FMP construction |
| Out-of-range relation index | Edge relation tensor is not aligned to compiled registry | Use dense `relation_index`, not stable IDs |
| Nonfinite output | Corrupt parameter or unstable upstream state | Check inputs and call `assert_finite_parameters()` |
| Checkpoint missing/unexpected keys | Registry order, stable IDs, mode, or bias setting changed | Compare architecture and registry fingerprints before loading |
| Per-relation memory pressure | `[E,H,H]` selected weights are large | Reduce batch/hidden size or implement a tested grouped/basis method |

---

# 23. Anti-patterns

## 23.1 Using stable IDs as `edge_relation_index`

Incorrect:

```python
edge_relation_index = torch.tensor(
    [200, 710, 300],
    dtype=torch.long,
)
```

Correct:

```text
edge_relation_index contains dense positions 0..R-1
```

Stable IDs remain metadata in the compiled registry.

## 23.2 Reordering relation metadata after construction

Do not sort names independently or rebuild a mapping from an unordered container.

The dense order is part of architecture identity.

## 23.3 Calling the low-level implementation in production model code

Doing so returns only a tensor and bypasses `RelationTransformOutput` lineage.

Use `RelationTransforms.forward()` at the model boundary.

## 23.4 Adding activation or dropout inside one implementation only

That would make transform modes differ in more than parameter sharing and weaken the ablation.

Add new behavior as an explicit, consistently configured stage or a new named mode.

## 23.5 Silently moving or casting inputs

Implicit adaptation can hide artifact-construction errors and mixed-precision mistakes. Preserve the strict policy.

## 23.6 Treating control relations as ordinary explanations

They are transformed mathematically for comparability, but explanation policy must still identify or exclude controls downstream.

## 23.7 Loading a per-relation checkpoint under a new registry

Equal tensor shapes are not enough. Relation names, stable IDs, order, module keys, and registry fingerprint must match.

---

# 24. Extending the subsystem

## 24.1 Adding a new transform mode

A new mode should be introduced only when it represents a meaningful architecture or ablation.

Required steps:

1. Add the canonical mode constant in `constants.py`.
2. Add it to the canonical vocabulary.
3. Add it to the implemented vocabulary only after implementation and tests exist.
4. Create a dedicated implementation module with a bounded responsibility.
5. Preserve input/output width `[H -> H]` unless a broader contract change is intentional.
6. Expose `device`, `dtype`, `parameter_count`, `trainable_parameter_count`, `architecture_dict()`, `architecture_fingerprint()`, `parameter_fingerprint()`, and `assert_finite_parameters()`.
7. Add dispatcher construction and mathematical dispatch.
8. Preserve the exact `CompiledRelationRegistry` contract.
9. Extend `__init__.py` only for intended public classes.
10. Add focused implementation tests and dispatcher integration tests.
11. Update this README and schema versions when contracts change.

## 24.2 Candidate future modes

Possible research extensions include:

```text
basis decomposition
low-rank per-relation maps
shared map plus learned relation adapter
hypernetwork-generated relation transforms
relation-family shared maps
source-type/target-type conditioned transforms
hazard-conditioned transform modulation
```

These should not be folded silently into the current shared or per-relation equations.

## 24.3 Basis decomposition example

A parameter-efficient relation transform could use:

```text
W_r = Σ_b a[r,b] B_b
```

where `B_b` are shared basis matrices and `a[r,b]` are relation coefficients.

Such a mode would need its own:

- architecture identity;
- parameter fingerprints;
- relation-specific effective-map diagnostics;
- empty-group behavior;
- parameter-count analysis;
- ablation against both shared and fully independent transforms.

## 24.4 Hazard-conditioned modulation

The current architecture keeps hazard conditioning in gates and attention. If future work modulates transformation parameters directly, the mode must clarify whether hazard conditioning is:

- additive;
- multiplicative;
- FiLM-like;
- low-rank adaptation;
- hypernetwork generated.

It should remain distinguishable from the existing hazard-conditioned relation gate.

---

# 25. Performance considerations

## 25.1 Shared mode

The shared implementation is conventional and memory efficient.

## 25.2 Per-relation vectorization

The current implementation stacks all relation matrices and selects an `[H,H]` matrix for every edge.

Advantages:

- simple tensor semantics;
- deterministic relation alignment;
- vectorized execution;
- clean autograd behavior;
- explicit zero gradients for absent relations.

Potential cost:

```text
selected_weights shape = [E, H, H]
```

For very large edge counts or hidden dimensions, this can dominate memory.

## 25.3 Safe optimization directions

Any optimization must preserve exact numerical and identity contracts. Candidate approaches:

- group edges by relation and apply each `nn.Linear` to its edge subset;
- custom segmented linear kernels;
- basis decomposition;
- low-rank adapters;
- fused indexed matrix-vector multiplication.

Optimized implementations must retain:

- output order aligned to original edges;
- exact registry mapping;
- empty-edge and empty-group behavior;
- finite validation;
- relation-specific fingerprints;
- gradients for every parameter;
- deterministic state-dict identity.

## 25.4 Fingerprinting overhead

`forward()` computes a full parameter fingerprint for `RelationTransformOutput`. This is audit friendly but may be expensive in a hot training loop.

Profile before changing it. Any future optimization should expose explicit audit semantics rather than making reproducibility metadata silently disappear.

---

# 26. Testing checklist

## 26.1 Shared transform

- constructor accepts valid `hidden_dim` and Boolean bias;
- exact `nn.Linear(H,H)` architecture;
- source gathering matches `node_state[source_index]`;
- all relation identities would receive the same transformation;
- no activation, normalization, or dropout modules exist;
- empty edges and empty nodes/edges work;
- invalid ranks, widths, dtypes, devices, ranges, and nonfinite values fail;
- forward and backward values are finite;
- architecture fingerprint is parameter independent;
- parameter fingerprint changes with parameter values;
- state-dict round trip is exact;
- finite-parameter diagnostic catches corruption;
- optional CUDA execution matches contract.

## 26.2 Per-relation transform

- relation names, stable IDs, and controls align exactly;
- module keys are deterministic and unique;
- lookup by name and stable ID resolves correctly;
- integer lookup is never treated as dense index;
- independent relation maps produce expected outputs;
- zero-edge batches preserve autograd connectivity;
- absent relation groups receive zero gradients;
- control relations retain normal parameters and metadata;
- stacked weight/bias order matches compiled order;
- relation-specific fingerprints are deterministic;
- global fingerprints include all relation identities;
- state-dict keys contain dense index and stable ID;
- constructor from compiled registry validates and preserves order;
- device, dtype, range, and finiteness failures are strict;
- gradients are isolated to selected relation parameters.

## 26.3 Dispatcher

- package exports are complete;
- implemented modes construct correct child classes;
- unknown and canonical-unimplemented modes fail differently;
- config construction validates configuration;
- registry must be nonempty and internally aligned;
- input hidden width, names, IDs, fingerprint, and controls must match;
- input and parameter device/dtype must match;
- shared and per-relation dispatch use correct arguments;
- implementation output shape/device/dtype/finiteness is guarded;
- `forward()` returns `RelationTransformOutput`;
- `source_inputs` is preserved by identity;
- shared relation-fingerprint mapping is empty;
- per-relation mapping contains all relation fingerprints;
- state-dict namespace begins with `implementation.`;
- empty-edge behavior works through the public boundary;
- finite autograd and optional CUDA behavior are covered.

## 26.4 Integration

- compiled graph relation indices use the same registry as the transform;
- `NodeStateFusionOutput.output_dim == hidden_dim`;
- transform output is consumed without reordering edges;
- downstream normalization, gate, attention, and message objects reference the same FMP inputs;
- checkpoint artifacts preserve architecture and registry fingerprints;
- shared and per-relation experiments report parameter counts.

---

# 27. Schema versions

| Constant | Current value | Scope |
|---|---|---|
| `SHARED_RELATION_TRANSFORM_SCHEMA_VERSION` | `0.1` | Shared implementation architecture contract |
| `PER_RELATION_TRANSFORM_SCHEMA_VERSION` | `0.1` | Per-relation implementation and identity contract |
| `RELATION_TRANSFORMS_DISPATCHER_SCHEMA_VERSION` | `0.1` | Public dispatch and metadata boundary |

A schema version should change when serialized or fingerprinted contract meaning changes, not for formatting-only edits.

Examples that may require a version change:

- changing operation order;
- changing module-key semantics;
- adding hidden activation or normalization;
- changing relation-lookup interpretation;
- changing architecture-dictionary fields in a compatibility-breaking way;
- changing output lineage semantics.

---

# 28. Quick reference

## 28.1 Which class should be used?

| Goal | Class/method |
|---|---|
| Normal model execution | `RelationTransforms.forward()` |
| Full construction from FMP config | `RelationTransforms.from_config()` |
| Tensor-only result with full input validation | `RelationTransforms.transform_tensor()` |
| Shared mathematical unit test | `SharedRelationTransform` |
| Per-relation mathematical unit test | `PerRelationTransform` |
| Build per-relation module from registry | `PerRelationTransform.from_compiled_registry()` |
| Inspect one relation's parameters | `module_for_relation()` or `module_for_relation_index()` |
| Audit architecture | `architecture_dict()` / `architecture_fingerprint()` |
| Audit exact parameters | `parameter_fingerprint()` |
| Audit each relation | `relation_parameter_fingerprints()` |
| Detect corrupt parameters | `assert_finite_parameters()` |

## 28.2 Mode comparison

| Property | Shared | Per relation |
|---|---:|---:|
| Number of affine maps | `1` | `R` |
| Parameters with bias | `H² + H` | `R(H² + H)` |
| Uses `edge_relation_index` mathematically | No | Yes |
| Preserves registry metadata | Yes, through dispatcher | Yes |
| Relation-specific fingerprints | Empty mapping | One per relation |
| Control relation special math | No | No |
| Output shape | `[E,H]` | `[E,H]` |
| Activation / norm / dropout | None | None |

## 28.3 Core invariants

```text
node_state.shape == [N, H]
source_index.shape == [E]
edge_relation_index.shape == [E]
output.shape == [E, H]

source_index uses node-axis positions
edge_relation_index uses dense compiled relation positions
stable relation IDs are metadata, not tensor indices

all tensors and parameters share one device
floating input dtype equals parameter dtype
all floating values are finite
registry identity and ordering remain exact
```

---

# 29. Maintenance checklist

Before modifying this package:

1. Decide whether the change belongs in transformation, gating, attention, normalization, or aggregation.
2. Preserve the distinction between stable ontology IDs and dense runtime indices.
3. Keep the shared baseline mathematically minimal.
4. Avoid adding behavior to per-relation mode that is absent from shared mode unless it defines a new named mode.
5. Preserve strict device and dtype behavior.
6. Preserve zero-edge and empty-relation-group support.
7. Preserve metadata through `RelationTransformOutput`.
8. Update architecture dictionaries and schema versions when semantics change.
9. Add focused implementation tests and dispatcher tests.
10. Compare parameter counts and computational cost in research experiments.
11. Verify checkpoint keys and fingerprints after registry-related changes.
12. Update this README and any module-interface documentation.

---

# 30. Summary

The relation-transforms package has one bounded responsibility:

> Convert each edge's source-node state into an edge-aligned hidden representation while preserving exact relation and artifact identity.

`SharedRelationTransform` provides the generic parameter-sharing control.

`PerRelationTransform` provides independent functional transformations aligned to the compiled relation ontology.

`RelationTransforms` enforces configuration, registry, device, dtype, and lineage compatibility and returns the typed output consumed by the remaining functional message-passing stages.

The package is intentionally strict and minimal. Relation semantics come from the registry; hazard relevance comes from gates and priors; neighbor importance comes from attention; graph scaling comes from normalization; and node updates come from aggregation and the layer orchestrator. Keeping those concerns separate is what makes the V2 architecture testable, reproducible, and interpretable.
