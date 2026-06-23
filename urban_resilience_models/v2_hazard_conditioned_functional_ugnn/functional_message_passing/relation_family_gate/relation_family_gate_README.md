# Hazard-Conditioned Relation Gate

## Technical reference for `functional_message_passing/relation_family_gate/`

**Model family:** `v2_hazard_conditioned_functional_ugnn`  
**Package path:**

```text
urban_resilience_models/
└── v2_hazard_conditioned_functional_ugnn/
    └── functional_message_passing/
        └── relation_family_gate/
            ├── __init__.py
            ├── activations.py
            ├── gate_network.py
            ├── relation_family_gate.py
            ├── relation_priors.py
            └── schemas.py
```

**Implementation status:** bounded V2.0 exact-relation gating  
**Primary framework:** PyTorch  
**Recommended Python version:** Python 3.11 or newer

---

## 1. Purpose

This package implements the hazard-conditioned relation-gating stage of the V2 functional message-passing architecture.

For every target node and every exact relation in the compiled relation registry, it predicts an independent gate value in `[0, 1]`. The resulting node–relation gate matrix is then gathered onto stored edges using each edge's target node and dense compiled relation index.

The subsystem answers two related questions:

```text
At node n, under the current hazard query, how active should relation r be?

For stored edge e = (source -> target) with relation r_e,
what gate value should multiply that edge's later message?
```

The bounded V2.0 path is:

```text
FunctionalMessagePassingInputs
        │
        ├── fused node state [N, D]
        ├── node-aligned hazard query [N, Q]
        ├── exact compiled relation metadata [R]
        ├── optional compiled hazard–relation priors
        ├── target_index [E]
        └── edge_relation_index [E]
        │
        ▼
RelationGateNetwork
        │ neural_logits [N, R]
        ▼
RelationPriorContributionBuilder (optional)
        │ prior_logit_contribution [N, R]
        ▼
RelationGateActivation
        │ combined_logits [N, R]
        │ gate_values [N, R]
        ▼
exact edge lookup
        │ edge_gate_values [E]
        ▼
RelationGateOutput
```

This package does **not** build graph edges, compile registries, construct hazard queries, compute edge attention, transform source states, multiply message factors, aggregate messages, or define training losses.

---

## 2. Important naming clarification

The package and primary class retain the historical names:

```text
relation_family_gate
RelationFamilyGate
```

In bounded V2.0, however, the trainable gate axis is **not** a pooled semantic-family axis. It is the exact dense relation axis produced by `CompiledRelationRegistry`:

```text
r = 0, 1, ..., R - 1
```

Each gate column corresponds to one exact compiled relation identity:

```text
relation_index r
    -> relation_names[r]
    -> stable_relation_ids[r]
```

Semantic family metadata may also be carried by `RelationGateAxis`, but it is used only for:

- diagnostics;
- explanation grouping;
- ontology inspection;
- future hierarchical extensions.

It never pools, collapses, reorders, averages, or substitutes exact relation channels in V2.0.

This distinction is important. A gate for `drainage_dependency` is not automatically the same trainable channel as another dependency relation merely because both share a semantic family.

---

## 3. Mathematical contract

### 3.1 Symbols and dimensions

| Symbol | Meaning |
|---|---|
| `N` | number of nodes across the packed batch |
| `G` | number of packed graphs |
| `E` | number of stored directed edges |
| `R` | number of exact compiled relations |
| `D` | fused node-state width |
| `Q` | node-aligned hazard-query width |
| `H` | gate-network hidden width |
| `I` | enabled input width: `D`, `Q`, or `D + Q` |

### 3.2 Gate-network context

The network may use the fused node state, the hazard query, or both:

```text
x_n = concat(enabled node-aligned inputs)
```

With both inputs enabled:

```text
x_n = concat(node_state_n, hazard_query_n) ∈ R^(D + Q)
```

The context encoder is:

```text
z1_n = GELU(W1 x_n + b1)
z2_n = GELU(W2 z1_n + b2)
h_n  = LayerNorm(z2_n)       # when enabled
```

Without layer normalization:

```text
h_n = z2_n
```

### 3.3 Exact-relation scoring

Each compiled relation owns a learned embedding `e_r ∈ R^H` and may own a scalar bias `b_r`:

```text
neural_logit[n, r]
    = <h_n, e_r> / sqrt(H) + b_r
```

When relation bias is disabled, the final term is omitted.

The score matrix has shape:

```text
neural_logits: [N, R]
```

There is no relation-family pooling and no softmax normalization.

### 3.4 Optional prior integration

When compiled hazard–relation priors are enabled:

```text
prior_logit_contribution[n, r]
    = relation_prior_strength
      * compiled_gate_bias_logit[hazard_row(n), r]
```

The prior artifact is already responsible for confidence adjustment, ontology fallback, neutral behavior, applicability, and initialization masks. This package does not reinterpret those semantics.

The combined logit is:

```text
gate_logits = neural_logits
```

when no prior contribution is present, and:

```text
gate_logits = neural_logits + prior_logit_contribution
```

when priors are present.

### 3.5 Independent activation

Bounded V2.0 uses independent sigmoid activation:

```text
gate_values[n, r] = sigmoid(gate_logits[n, r])
```

Therefore:

```text
gate_values: [N, R]
0 <= gate_values[n, r] <= 1
```

Relations do not compete. Several relation mechanisms may be strongly active for the same node and hazard query.

### 3.6 Exact edge lookup

For each stored edge `e`:

```text
edge_gate_values[e]
    = gate_values[target_index[e], edge_relation_index[e]]
```

This produces:

```text
edge_gate_values: [E]
```

The lookup is target-node scoped because the gate answers which incoming mechanisms should be active for the receiving node.

---

## 4. Architectural position

The relation gate sits between hazard-aware node-state construction and edge-level message weighting.

A later message builder can conceptually use:

```text
message_e =
    transformed_source_state_e
    × edge_gate_values[e]
    × edge_attention[e]
    × structural_edge_normalization[e]
```

The factors have distinct meanings:

- **relation transform:** how source information is represented for the relation;
- **relation gate:** how active the mechanism is for the target node and hazard;
- **edge attention:** which specific incoming edge matters within its group;
- **structural normalization:** graph-scale or degree-related weighting.

The gate is not attention and should not be normalized as though all relation channels must sum to one.

---

## 5. Package responsibilities

| File | Primary responsibility |
|---|---|
| `schemas.py` | Immutable exact-axis, neural-output, prior-output, and activation-output contracts |
| `gate_network.py` | Neural prediction of target-node logits over exact relations |
| `relation_priors.py` | Alignment of compiled hazard–relation priors to nodes and gate-logit space |
| `activations.py` | Additive logit composition and independent sigmoid activation |
| `relation_family_gate.py` | End-to-end orchestration and edge-aligned gate lookup |
| `__init__.py` | Side-effect-free public package exports and compact aliases |

The ownership boundaries are deliberate. Callers should not add prior logic to the gate network, relation scoring to the activation module, or edge lookup to the schemas.

---

# 6. Public API

The package exports the following symbols.

## 6.1 Schema versions

```python
GATE_ACTIVATION_OUTPUT_SCHEMA_VERSION
GATE_NETWORK_OUTPUT_SCHEMA_VERSION
RELATION_FAMILY_GATE_SCHEMA_VERSION
RELATION_GATE_ACTIVATIONS_SCHEMA_VERSION
RELATION_GATE_AXIS_SCHEMA_VERSION
RELATION_GATE_NETWORK_SCHEMA_VERSION
RELATION_PRIOR_CONTRIBUTION_SCHEMA_VERSION
RELATION_PRIOR_INTEGRATION_SCHEMA_VERSION
```

These version strings identify serialized or fingerprinted contracts. A schema-version change should be treated as a compatibility event.

## 6.2 Immutable contracts

```python
RelationGateAxis
GateNetworkOutput
RelationPriorContribution
GateActivationOutput
RelationGateOutput
```

`RelationGateOutput` is defined in the parent `functional_message_passing.schemas` module and re-exported here for convenience.

## 6.3 Trainable network

```python
RelationGateNetwork
GateNetwork  # alias
```

## 6.4 Prior integration

```python
RelationPriorContributionBuilder
RelationPriorBuilder       # alias
RelationPriorIntegration   # alias
```

All three names refer to the same implementation class.

## 6.5 Activation

```python
RelationGateActivation
GateActivation  # alias
sigmoid_gate_activation
apply_relation_gate_activation
```

## 6.6 Complete orchestrator

```python
RelationFamilyGate
RelationGate  # alias
```

`RelationGate` is the shorter operational alias. `RelationFamilyGate` remains the canonical historical class name.

---

# 7. `schemas.py`

## 7.1 Purpose

`schemas.py` defines immutable, metadata-preserving intermediate contracts. It validates not only shape and dtype, but also ontology alignment, registry fingerprints, source lineage, exact equations, and value identities.

It owns no trainable computation.

The main types are:

```text
RelationGateAxis
GateNetworkOutput
RelationPriorContribution
GateActivationOutput
```

It also re-exports the parent subsystem's `RelationGateOutput`.

---

## 7.2 `RelationGateAxis`

### Role

`RelationGateAxis` is the authoritative description of the gate's dense relation columns.

For every dense relation index `r`:

```text
relation_names[r]
stable_relation_ids[r]
control_relation_mask[r]
```

must refer to the same exact relation.

### Fields

```python
RelationGateAxis(
    relation_names: tuple[str, ...],
    stable_relation_ids: tuple[int, ...],
    control_relation_mask: torch.Tensor,
    compiled_relation_registry_fingerprint: str,
    family_names: tuple[str, ...] = (),
    stable_family_ids: tuple[int, ...] = (),
    relation_family_index_by_relation: torch.Tensor | None = None,
    source_relation_registry_fingerprint: str | None = None,
    schema_version: str = RELATION_GATE_AXIS_SCHEMA_VERSION,
)
```

### Required invariants

- `relation_names` is nonempty and unique.
- `stable_relation_ids` is nonnegative, unique, and aligned one-to-one with `relation_names`.
- `control_relation_mask` has shape `[R]` and dtype `torch.bool`.
- the compiled-registry fingerprint is nonempty.
- the mask is on the same device as the runtime functional-message-passing inputs.
- relation ordering, stable IDs, control mask, and compiled-registry fingerprint must exactly match the source inputs.

### Optional family metadata

Family metadata is all-or-nothing. When any family field is supplied:

- `family_names` must be nonempty and unique;
- `stable_family_ids` must be unique and aligned with `family_names`;
- `relation_family_index_by_relation` must have shape `[R]` and dtype `torch.long`;
- every family index must be in `[0, F - 1]`;
- every declared family must be represented by at least one relation;
- `source_relation_registry_fingerprint` must be present;
- all metadata must exactly match `source_inputs.relation_families`.

Family metadata remains diagnostic. It does not change `R` or any gate equation.

### Constructors and properties

```python
axis = RelationGateAxis.from_inputs(source_inputs=inputs)
```

Useful properties:

```python
axis.num_relations
axis.num_families
axis.device
axis.has_family_metadata
axis.control_relation_names
```

### Compatibility check

```python
axis.assert_matches_inputs(inputs)
```

This is the preferred way to validate a previously constructed or loaded axis before using it with a new batch.

### Fingerprints

```python
axis.semantic_dict()
axis.semantic_fingerprint()
axis.value_fingerprint()
axis.fingerprint()
```

The final fingerprint combines semantic identity with tensor-value identity. It detects changes to relation ordering, stable IDs, control masks, family mapping, or registry provenance.

---

## 7.3 `GateNetworkOutput`

### Role

Represents neural logits before priors and activation.

### Fields

```python
GateNetworkOutput(
    logits: torch.Tensor,                         # [N, R]
    source_inputs: FunctionalMessagePassingInputs,
    axis: RelationGateAxis,
    scope: str,
    encoder_architecture_fingerprint: str,
    parameter_fingerprint: str | None = None,
    input_feature_names: tuple[str, ...] = (),
    schema_version: str = GATE_NETWORK_OUTPUT_SCHEMA_VERSION,
)
```

### Required invariants

- `logits` has shape `[N, R]`.
- logits are floating point, finite, and match the source-input dtype and device.
- `axis.assert_matches_inputs(source_inputs)` succeeds.
- bounded V2.0 scope is exactly `target_node`.
- input feature names are unique.
- architecture and optional parameter fingerprints are nonempty.

### Convenience properties

```python
output.num_nodes
output.num_relations
output.device
output.dtype
output.control_relation_mask
```

### Identity methods

```python
output.lineage_dict()
output.lineage_fingerprint()
output.value_fingerprint()
output.fingerprint()
```

The lineage includes the source-input lineage, axis identity, network scope, architecture identity, parameter identity, and input-feature selection.

---

## 7.4 `RelationPriorContribution`

### Role

Represents a node-aligned additive prior contribution in gate-logit space.

### Fields

```python
RelationPriorContribution(
    logit_contribution: torch.Tensor,             # [N, R]
    source_inputs: FunctionalMessagePassingInputs,
    axis: RelationGateAxis,
    strength: float,
    source_compiled_prior_fingerprint: str,
    prior_mean: torch.Tensor | None = None,        # [N, R]
    confidence: torch.Tensor | None = None,        # [N, R]
    initialization_mask: torch.Tensor | None = None, # bool [N, R]
    regularization_mask: torch.Tensor | None = None, # bool [N, R]
    resolution_summary: Mapping[str, int] = {},
    schema_version: str = RELATION_PRIOR_CONTRIBUTION_SCHEMA_VERSION,
)
```

### Required invariants

- `logit_contribution` is finite floating point `[N, R]` on the source-input device and dtype.
- strength is finite and nonnegative.
- the source inputs contain a compiled prior artifact.
- `source_compiled_prior_fingerprint` equals that artifact's fingerprint.
- optional means and confidence are `[N, R]` probabilities in `[0, 1]`.
- optional masks are Boolean `[N, R]` tensors.
- a strength of exactly `0.0` requires an exactly zero contribution tensor.
- `resolution_summary` becomes an immutable mapping with nonnegative integer counts.

### Interpretation

`prior_mean` and `confidence` are diagnostic values from the compiled prior artifact. The trainable network does not directly consume them. The contribution actually added to neural logits is `logit_contribution`.

`resolution_summary` counts the number of node–relation cells produced by each prior-resolution mode, after graph-level hazard rows are expanded to nodes.

---

## 7.5 `GateActivationOutput`

### Role

Represents combined logits and activated node–relation gates before edge lookup.

### Fields

```python
GateActivationOutput(
    gate_logits: torch.Tensor,                    # [N, R]
    gate_values: torch.Tensor,                    # [N, R]
    source_network_output: GateNetworkOutput,
    prior_contribution: RelationPriorContribution | None,
    activation: str,
    encoder_architecture_fingerprint: str,
    parameter_fingerprint: str | None = None,
    schema_version: str = GATE_ACTIVATION_OUTPUT_SCHEMA_VERSION,
)
```

### Equation validation

The schema validates the actual equations, not only metadata:

```text
gate_logits == neural_logits + optional_prior_contribution
gate_values == sigmoid(gate_logits)
```

Comparison uses dtype-appropriate tolerances:

- relaxed tolerances for `float16` and `bfloat16`;
- tight tolerances for `float64`;
- standard tolerances for `float32`.

### Source-object identity

When priors are present, the prior contribution and network output must reference the **exact same** `source_inputs` object, not merely equivalent values. Their axis fingerprints must also match.

### Convenience properties

```python
output.source_inputs
output.axis
output.scope
output.device
output.dtype
output.control_relation_mask
```

---

## 7.6 `RelationGateOutput`

The final output is defined in `functional_message_passing.schemas` and re-exported by this package.

The orchestrator populates at least:

```python
RelationGateOutput(
    gate_logits=...,                 # [N, R]
    gate_values=...,                 # [N, R]
    edge_gate_values=...,            # [E]
    source_inputs=...,
    scope="target_node",
    activation="sigmoid",
    encoder_architecture_fingerprint=...,
    parameter_fingerprint=...,
    prior_logit_contribution=...,    # [N, R] or None
    regularization_terms={},
)
```

The empty `regularization_terms` mapping is intentional: this package exposes prior regularization weights separately but does not invent a training-loss definition.

---

# 8. `gate_network.py`

## 8.1 Purpose

`RelationGateNetwork` predicts one neural logit for every node and every exact compiled relation.

It owns:

- selection and validation of node-aligned input features;
- context encoding;
- exact-relation embeddings;
- optional relation biases;
- relation scoring;
- network initialization;
- architecture and parameter fingerprints;
- construction of `GateNetworkOutput`.

It does not apply priors, sigmoid activation, or edge lookup.

---

## 8.2 Constructor

```python
RelationGateNetwork(
    *,
    node_state_dim: int,
    hazard_query_dim: int,
    relation_names: Sequence[str],
    stable_relation_ids: Sequence[int],
    hidden_dim: int = 64,
    scope: str = "target_node",
    use_node_state: bool = True,
    use_hazard_query: bool = True,
    layer_norm: bool = True,
    relation_bias: bool = True,
)
```

### Constructor rules

- all dimensions must be positive integers;
- relation names must be unique and nonempty;
- stable relation IDs must be unique, nonnegative integers;
- names and IDs must align one-to-one;
- at least one relation is required;
- at least one of `use_node_state` or `use_hazard_query` must be true;
- only target-node scope is implemented;
- canonical-but-unimplemented scopes raise `NotImplementedError` during normalization;
- unknown scopes raise `ValueError`.

`hazard_query_dim` must remain positive even when hazard-query input is disabled. This keeps architecture serialization explicit. In that case the dimension is non-operative.

---

## 8.3 Construction from configuration

```python
network = RelationGateNetwork.from_config(
    config=relation_config,
    source_inputs=fmp_inputs,
    use_node_state=True,
    use_hazard_query=True,
    layer_norm=True,
    relation_bias=True,
)
```

This constructor:

1. validates `RelationConfig`;
2. checks implemented capabilities when gating is enabled;
3. reads `D`, `Q`, relation names, and stable IDs from the source inputs;
4. uses `config.gate_hidden_dim` and `config.gate_scope`;
5. constructs the module;
6. moves it to the source-input device and dtype.

No hidden movement occurs during `forward`; only `from_config` performs the explicit `.to(...)` at construction.

---

## 8.4 Trainable parameters

The network contains:

```text
input_projection:       Linear(I, H)
hidden_projection:      Linear(H, H)
context_norm:           LayerNorm(H) or Identity
relation_embeddings:    Parameter[R, H]
relation_bias:          Parameter[R] or None
```

The approximate parameter count is:

```text
H*I + H                 input projection
+ H*H + H               hidden projection
+ R*H                   exact-relation embeddings
+ R                     optional relation bias
+ 2H                    optional LayerNorm scale and offset
```

Use `network.parameter_count` as the authoritative runtime count.

### Initialization

- linear weights: Xavier uniform;
- linear biases: zeros;
- relation embeddings: Xavier uniform;
- relation bias: zeros;
- LayerNorm: standard PyTorch reset.

This means the optional scalar relation biases initially contribute nothing.

---

## 8.5 Context assembly

```python
context = network.build_context(source_inputs)
```

Possible shapes:

```text
node state only:    [N, D]
hazard query only:  [N, Q]
both:               [N, D + Q]
```

The method validates exact feature widths, device, dtype, floating-point type, finiteness, and row alignment. It never casts or moves tensors.

### Ablation meanings

- `use_node_state=True`, `use_hazard_query=True`: spatially and hazard-varying gate;
- `use_node_state=False`, `use_hazard_query=True`: hazard-only gate shared across nodes with identical hazard queries;
- `use_node_state=True`, `use_hazard_query=False`: node-state gate that is not explicitly hazard-conditioned;
- both false: invalid.

---

## 8.6 Context encoding

```python
encoded = network.encode_context(context)  # [N, H]
```

The exact operation order is:

```text
Linear(I, H)
GELU
Linear(H, H)
GELU
LayerNorm(H) or Identity
```

No dropout is applied in this module.

---

## 8.7 Relation scoring

```python
logits = network.score_relations(encoded)  # [N, R]
```

The implementation uses a matrix multiplication:

```python
logits = encoded_context @ relation_embeddings.T
logits = logits / sqrt(hidden_dim)
logits = logits + relation_bias  # when enabled
```

The `1 / sqrt(H)` scale limits dot-product magnitude as hidden width grows.

Exact relation embeddings are aligned by constructor order. They must never be reordered independently from the compiled relation registry.

---

## 8.8 Forward method

```python
network_output = network(
    source_inputs,
    axis=optional_axis,
)
```

Execution order:

1. validate functional-message-passing inputs;
2. assert finite network parameters;
3. construct an axis from inputs or validate the supplied axis;
4. build node-aligned context;
5. encode context;
6. score exact relations;
7. construct `GateNetworkOutput`.

The network requires at least one node and one relation. It does not require stored edges because its output is node–relation scoped.

---

## 8.9 Public properties and diagnostics

```python
network.num_relations
network.input_dim
network.input_feature_names
network.parameter_count
network.trainable_parameter_count
network.relation_score_scale
network.architecture_dict()
network.architecture_fingerprint()
network.parameter_fingerprint()
network.assert_finite_parameters()
```

---

# 9. `relation_priors.py`

## 9.1 Purpose

`RelationPriorContributionBuilder` adapts a `CompiledHazardRelationPriors` artifact to the exact node–relation gate matrix.

It is a parameter-free adapter. It does not define prior values or ontology fallback rules. Those belong to the relations-layer prior registry and compilation process.

---

## 9.2 Constructor

```python
RelationPriorContributionBuilder(
    *,
    strength: float = 0.0,
    epsilon: float = 1e-4,
)
```

### Rules

- `strength` must be finite and nonnegative;
- `epsilon` must lie strictly inside `(0, 0.5)`;
- the module has no parameters;
- negative prior strength is intentionally forbidden.

A zero strength preserves all diagnostic prior tensors but produces an exactly zero additive logit contribution.

---

## 9.3 Construction from configuration

```python
builder = RelationPriorContributionBuilder.from_config(
    config=relation_config,
    epsilon=1e-4,
)
```

The configured strength comes from:

```python
config.relation_prior_strength
```

The larger orchestrator creates this builder only when:

```python
config.use_relation_priors is True
```

---

## 9.4 Required source-input metadata

Prior integration requires:

- `source_inputs.compiled_relation_priors`;
- an exact relation ordering matching `source_inputs.relation_names`;
- stable relation IDs matching `source_inputs.stable_relation_ids`;
- a compiled relation-registry fingerprint matching the runtime registry;
- `source_inputs.hazard_query` with preserved hazard-embedding lineage;
- `source_inputs.node_batch_index`;
- known graph-level hazard identities;
- at least one node, graph, and relation.

The hazard-query source embedding must be one of:

```text
HazardEmbeddingLookup
NodeAlignedHazardEmbeddingLookup
```

For a node-aligned lookup, its graph membership must exactly match `source_inputs.node_batch_index`.

Unknown runtime hazards are rejected. The prior builder does not guess, fall back locally, or silently use an all-hazard row.

---

## 9.5 Hazard-row alignment

```python
node_hazard_rows = builder.resolve_node_hazard_rows(source_inputs)
```

The process is:

1. recover each packed graph's hazard name and stable hazard ID;
2. resolve both to the same row in the compiled prior artifact;
3. reject disagreement between name and stable ID;
4. obtain graph-level row indices `[G]`;
5. index them by `node_batch_index` to obtain node-level rows `[N]`.

This explicit graph-to-node expansion prevents accidental broadcasting across packed graphs with different hazards.

---

## 9.6 Resolved prior tensors

For every node and exact relation, the builder resolves:

```text
prior_mean             [N, R]
confidence             [N, R]
initialization_mask    [N, R] bool
regularization_mask    [N, R] bool
base_logits            [N, R]
regularization_weights [N, R]
logit_contribution     [N, R]
```

The underlying compiled tables are tensorized on the source-input device and dtype. Boolean masks remain `torch.bool`.

The builder validates:

- prior means strictly inside `(0, 1)`;
- confidence in `[0, 1]`;
- finite base logits and weights;
- nonnegative regularization weights;
- exact zero weight outside the regularization mask;
- exact zero logit contribution when strength is zero.

---

## 9.7 Prior-to-logit conversion

The base logits come from:

```python
compiled.gate_bias_logit_matrix(
    activation=GateInitializationActivation.SIGMOID,
    epsilon=builder.epsilon,
)
```

The clipping epsilon protects the logit transform from singular probabilities at zero and one.

The final contribution is:

```python
logit_contribution = base_logits * strength
```

The operation is additive in logit space, not multiplicative in probability space.

### Consequence

A positive prior strength does not force a relation gate to a fixed value. The neural network can learn logits that offset or reinforce the prior contribution.

---

## 9.8 Regularization weights

```python
weights = builder.regularization_weights(source_inputs)  # [N, R]
```

This method exposes confidence-weighted compiled regularization metadata. It does not define a loss.

A training module may later use the weights with a separately specified objective, for example a deviation penalty between learned gates and prior means. Such a loss must be explicit, versioned, and scientifically justified outside this package.

---

## 9.9 Resolution diagnostics

The builder computes a `resolution_summary` that counts node–relation cells by prior-resolution mode.

Graph-level resolution modes are weighted by the number of nodes in each graph, so the total count equals:

```text
N * R
```

This makes inherited, explicit, all-hazard, and neutral-default usage auditable at the actual node–relation level.

---

## 9.10 Forward method

```python
prior = builder(
    source_inputs,
    axis=optional_axis,
)
```

Returns `RelationPriorContribution` containing the scaled contribution, means, confidence, masks, compiled-prior fingerprint, and resolution summary.

---

# 10. `activations.py`

## 10.1 Purpose

This module owns independent sigmoid activation and the optional addition of prior logits.

It is parameter-free.

---

## 10.2 Functional API

### `sigmoid_gate_activation`

```python
values = sigmoid_gate_activation(logits)
```

Contract:

- input shape `[N, R]`;
- floating-point dtype;
- finite values;
- same shape, device, and dtype on output;
- output lies in `[0, 1]`.

### `apply_relation_gate_activation`

```python
values = apply_relation_gate_activation(
    logits,
    activation="sigmoid",
)
```

The dispatcher distinguishes:

- unknown activation: `ValueError`;
- canonical but not implemented in V2.0: `NotImplementedError`;
- implemented sigmoid: normal execution.

It never silently substitutes a different activation.

---

## 10.3 `RelationGateActivation`

### Constructor

```python
RelationGateActivation(
    *,
    activation: str = "sigmoid",
)
```

### From configuration

```python
activation = RelationGateActivation.from_config(
    config=relation_config,
)
```

### Logit composition

```python
combined = activation.combine_logits(
    network_output,
    prior_contribution,
)
```

When the prior is absent, this method returns the exact neural-logit tensor without cloning.

When a prior is present, it requires:

- the exact same `source_inputs` object;
- equal axis fingerprints;
- identical shape, device, and dtype.

### Activation-only use

```python
values = activation.activate_tensor(logits)
```

### Metadata-preserving forward

```python
activation_output = activation(
    network_output,
    prior_contribution,
)
```

Returns `GateActivationOutput` and preserves full network and prior provenance.

### Diagnostics

```python
activation.is_sigmoid
activation.parameter_count              # 0
activation.trainable_parameter_count    # 0
activation.architecture_dict()
activation.architecture_fingerprint()
activation.parameter_fingerprint()
activation.assert_finite_parameters()
```

The parameter fingerprint is still meaningful for a parameter-free stage: it identifies the schema, implementation class, zero parameter count, and empty state-dict structure.

---

# 11. `relation_family_gate.py`

## 11.1 Purpose

`RelationFamilyGate` is the complete public orchestrator. Model code should normally interact with this class rather than manually wiring the individual stages.

It coordinates:

1. source-input validation;
2. exact-axis construction or validation;
3. neural logit prediction;
4. optional prior contribution;
5. additive logit composition;
6. sigmoid activation;
7. exact target-node/relation edge lookup;
8. final `RelationGateOutput` construction.

---

## 11.2 Constructor

```python
RelationFamilyGate(
    *,
    gate_network: RelationGateNetwork,
    activation: RelationGateActivation,
    prior_builder: RelationPriorContributionBuilder | None = None,
)
```

The constructor enforces:

- exact component types;
- target-node gate scope;
- sigmoid activation;
- optional prior builder of the correct type.

Canonical future scopes and activations must receive dedicated implementations rather than being inserted behind this orchestrator.

---

## 11.3 Construction from configuration

```python
gate = RelationFamilyGate.from_config(
    config=relation_config,
    source_inputs=fmp_inputs,
    use_node_state=True,
    use_hazard_query=True,
    layer_norm=True,
    relation_bias=True,
    prior_epsilon=1e-4,
)
```

This builds:

- a `RelationGateNetwork` aligned to the source input's exact relation ordering;
- a `RelationGateActivation` from the configured activation;
- a prior builder only when `config.use_relation_priors` is enabled.

The gate-network module is moved to the source-input device and dtype by its `from_config` constructor.

The `gate_enabled` flag is interpreted by the larger model wiring. This constructor validates the configuration and can construct the component when called; callers should avoid invoking it when the gate is disabled.

---

## 11.4 Runtime source-input validation

The orchestrator requires:

- exact relation-name ordering equal to the network's ordering;
- exact stable relation IDs equal to the network's IDs;
- equal relation counts;
- source hidden width equal to `gate_network.node_state_dim`;
- a node-aligned hazard query of the configured width when hazard input is enabled;
- at least one node and relation;
- floating-point source node state.

These checks prevent silent semantic corruption when two registries happen to have the same relation count but different orderings.

---

## 11.5 Axis resolution

```python
axis = gate.resolve_axis(
    source_inputs,
    axis=optional_axis,
)
```

When omitted, the axis is derived from the source inputs. When supplied, it must match the source inputs and the gate network exactly.

Supplying a cached axis is safe only when the compiled registry, control mask, family metadata, device, names, and stable IDs all remain unchanged.

---

## 11.6 Optional prior contribution

```python
prior = gate.build_prior_contribution(
    source_inputs,
    axis=axis,
)
```

Returns `None` when no prior builder is configured.

```python
weights = gate.prior_regularization_weights(source_inputs)
```

Returns `None` when priors are disabled. Otherwise it returns `[N, R]` weights from the compiled prior artifact.

---

## 11.7 Edge lookup

```python
edge_values = gate.lookup_edge_gate_values(
    activation_output,
)
```

The method performs differentiable advanced indexing:

```python
edge_values = gate_values[target_index, edge_relation_index]
```

It does not clone, detach, cast, or move the source tensor.

The result is validated for:

- shape `[E]`;
- floating-point dtype equal to the source-input dtype;
- source-input device;
- finite values;
- range `[0, 1]`.

A zero-edge batch is valid at this stage and produces an empty `[0]` tensor, provided the source inputs themselves satisfy their contract.

---

## 11.8 Forward method

```python
output = gate(
    source_inputs,
    axis=optional_axis,
)
```

The returned object contains both node-level and edge-level gates:

```python
output.gate_logits         # [N, R]
output.gate_values         # [N, R]
output.edge_gate_values    # [E]
```

When priors are enabled:

```python
output.prior_logit_contribution  # [N, R]
```

Otherwise the prior field is `None`.

---

## 11.9 Public properties and identity

```python
gate.scope
gate.activation_name
gate.relation_names
gate.stable_relation_ids
gate.num_relations
gate.uses_relation_priors
gate.parameter_count
gate.trainable_parameter_count

gate.architecture_dict()
gate.architecture_fingerprint()
gate.parameter_fingerprint()
gate.assert_finite_parameters()
```

The orchestrator parameter fingerprint combines the gate-network, prior-builder, and activation-stage identities. Parameter-free stages still contribute deterministic fingerprints.

---

# 12. `__init__.py`

## Purpose

The package initializer defines the supported public import surface.

Example:

```python
from urban_resilience_models.v2_hazard_conditioned_functional_ugnn.functional_message_passing.relation_family_gate import (
    RelationGate,
    RelationGateAxis,
    RelationGateNetwork,
    RelationPriorContributionBuilder,
)
```

Importing the package performs no model construction, registry compilation, device movement, or registry mutation.

Use package-level imports in most application code. Import implementation modules directly only when focused testing or low-level research instrumentation requires it.

---

# 13. Recommended usage

## 13.1 Standard model integration

```python
from urban_resilience_models.v2_hazard_conditioned_functional_ugnn.functional_message_passing.relation_family_gate import (
    RelationGate,
)

relation_gate = RelationGate.from_config(
    config=model_config.relations,
    source_inputs=fmp_inputs,
)

gate_output = relation_gate(fmp_inputs)

node_relation_gates = gate_output.gate_values       # [N, R]
edge_gates = gate_output.edge_gate_values           # [E]
```

The exact configuration attribute name may vary with the surrounding model configuration. The required object type is `RelationConfig`.

---

## 13.2 Explicit construction without priors

```python
from urban_resilience_models.v2_hazard_conditioned_functional_ugnn.functional_message_passing.relation_family_gate import (
    RelationFamilyGate,
    RelationGateActivation,
    RelationGateNetwork,
)

network = RelationGateNetwork(
    node_state_dim=fmp_inputs.hidden_dim,
    hazard_query_dim=fmp_inputs.node_hazard_query.shape[1],
    relation_names=fmp_inputs.relation_names,
    stable_relation_ids=fmp_inputs.stable_relation_ids,
    hidden_dim=64,
    use_node_state=True,
    use_hazard_query=True,
    layer_norm=True,
    relation_bias=True,
).to(device=fmp_inputs.device, dtype=fmp_inputs.dtype)

activation = RelationGateActivation(
    activation="sigmoid",
)

gate = RelationFamilyGate(
    gate_network=network,
    activation=activation,
    prior_builder=None,
)

output = gate(fmp_inputs)
```

---

## 13.3 Explicit construction with priors

```python
from urban_resilience_models.v2_hazard_conditioned_functional_ugnn.functional_message_passing.relation_family_gate import (
    RelationFamilyGate,
    RelationGateActivation,
    RelationGateNetwork,
    RelationPriorContributionBuilder,
)

prior_builder = RelationPriorContributionBuilder(
    strength=0.25,
    epsilon=1e-4,
)

gate = RelationFamilyGate(
    gate_network=network,
    activation=RelationGateActivation(activation="sigmoid"),
    prior_builder=prior_builder,
)

output = gate(fmp_inputs)
regularization_weights = gate.prior_regularization_weights(fmp_inputs)
```

The source inputs must already contain a compatible compiled prior artifact and hazard-query lineage.

---

## 13.4 Inspecting intermediate stages

```python
axis = gate.resolve_axis(fmp_inputs)
network_output = gate.gate_network(fmp_inputs, axis=axis)
prior = gate.build_prior_contribution(fmp_inputs, axis=axis)
activation_output = gate.activation(network_output, prior)
edge_values = gate.lookup_edge_gate_values(activation_output)
```

This is useful for tests, diagnostics, explanation export, and ablations. Production model code should generally call the complete orchestrator.

---

## 13.5 Hazard-only ablation

```python
gate = RelationGate.from_config(
    config=relation_config,
    source_inputs=fmp_inputs,
    use_node_state=False,
    use_hazard_query=True,
)
```

This tests whether hazard identity alone can determine relation activation without node-specific context.

---

## 13.6 Hazard-blind node-state ablation

```python
gate = RelationGate.from_config(
    config=relation_config,
    source_inputs=fmp_inputs,
    use_node_state=True,
    use_hazard_query=False,
)
```

This is a useful control but is no longer explicitly hazard-conditioned.

---

# 14. Configuration contract

The package consumes fields from `RelationConfig`, including:

```text
gate_enabled
gate_hidden_dim
gate_scope
gate_activation
use_relation_priors
relation_prior_strength
```

Construction methods call:

```python
config.validate()
```

and, when the gate is enabled:

```python
config.assert_implemented()
```

The subsystem distinguishes:

- an unknown value, which indicates an invalid configuration;
- a known canonical value that is not implemented in V2.0;
- a supported implemented value.

This avoids silently degrading future configuration modes into the current baseline.

---

# 15. Device and dtype behavior

## 15.1 No hidden casting

Forward paths require source tensors and trainable parameters to already share:

- one device;
- one floating-point dtype.

The package does not silently cast inputs or move tensors between CPU and GPU.

## 15.2 `from_config` behavior

`RelationGateNetwork.from_config` explicitly moves the newly constructed network to:

```python
source_inputs.device
source_inputs.dtype
```

This is a construction-time convenience, not hidden forward behavior.

## 15.3 Prior tensorization

Compiled Python matrices are converted to tensors on the source-input device. Floating matrices use the source-input dtype, while masks use `torch.bool` and indices use `torch.long`.

## 15.4 CUDA device comparison

The implementation normalizes CUDA device identity so an unspecified CUDA index and the current CUDA device can be treated consistently.

---

# 16. Finiteness and numerical safety

The subsystem rejects NaN and infinity in:

- node states;
- hazard queries;
- context tensors;
- network parameters;
- neural logits;
- prior tables and contributions;
- combined logits;
- activated gate values;
- edge-aligned gate values.

Sigmoid values are explicitly checked to lie in `[0, 1]`.

The prior epsilon prevents infinite sigmoid logits when compiled effective means approach zero or one.

Parameter checks are exposed through:

```python
network.assert_finite_parameters()
activation.assert_finite_parameters()
prior_builder.assert_finite_parameters()
gate.assert_finite_parameters()
```

---

# 17. Identity and reproducibility

## 17.1 Architecture fingerprints

Architecture fingerprints encode configuration and semantic structure, including:

- gate scope;
- activation;
- enabled input sources;
- input and hidden widths;
- relation ordering;
- stable relation IDs;
- relation-bias and LayerNorm choices;
- prior strength and epsilon;
- exact-axis behavior;
- family-pooling status;
- operation order;
- output schema.

They do not encode current learned tensor values.

## 17.2 Parameter fingerprints

Parameter fingerprints encode the exact state-dict tensor values, shapes, dtypes, and names for trainable stages.

Parameter-free stages produce deterministic structural fingerprints rather than returning nothing.

## 17.3 Value fingerprints

Schema value fingerprints hash runtime tensors such as logits, gate values, masks, and prior contributions.

## 17.4 Lineage fingerprints

Lineage fingerprints connect outputs to:

- the original functional-message-passing inputs;
- exact relation-axis identity;
- compiled relation registry;
- compiled prior artifact;
- architecture identity;
- parameter identity;
- prior resolution path.

## 17.5 Checkpoint rule

A checkpoint must not be interpreted solely by tensor shapes. If the relation ordering changes while `R` remains the same, relation embeddings and biases would be assigned the wrong semantics.

Persist or verify at least:

```text
relation_names
stable_relation_ids
compiled relation-registry fingerprint
architecture fingerprint
parameter fingerprint
```

before loading or comparing gate checkpoints.

---

# 18. Autograd behavior

The following operations remain differentiable:

- context concatenation;
- both linear projections;
- GELU activations;
- LayerNorm;
- relation-embedding dot products;
- relation biases;
- additive prior-logit composition with respect to neural logits;
- sigmoid activation;
- edge lookup by target and relation index.

The compiled prior contribution is parameter-free and does not receive gradients as a learned object. It shifts the logits through ordinary addition.

Edge lookup uses advanced indexing but preserves gradient flow back to the selected entries of `gate_values`, and therefore to the network parameters.

---

# 19. Computational characteristics

Let `I` be the enabled context width.

## 19.1 Gate network

Approximate time complexity:

```text
O(N * I * H)
+ O(N * H^2)
+ O(N * R * H)
```

The exact-relation scoring term may dominate when `R` is large.

Approximate activation memory:

```text
context:          O(N * I)
encoded context:  O(N * H)
logits:           O(N * R)
relation embeds:  O(R * H)
```

## 19.2 Prior integration

After compiled tables are tensorized, graph-to-node expansion and matrix indexing are approximately:

```text
O(N * R)
```

The implementation currently tensorizes compiled tables during resolution. Callers should avoid repeatedly rebuilding prior contributions outside the normal forward cadence.

## 19.3 Activation

```text
O(N * R)
```

## 19.4 Edge lookup

```text
O(E)
```

## 19.5 Explanation retention

Keeping full `[N, R]` logits and gates is intentional for interpretability, but may be significant for very large `N` and `R`. Any future sparse-gating variant requires a separate contract rather than silently dropping channels.

---

# 20. Interpretation

## 20.1 What a high gate means

A high value means:

> For this target node and current model context, the corresponding relation mechanism is strongly enabled before edge attention and aggregation.

It does not, by itself, mean:

- the relation causes the target;
- the relation increases risk rather than providing protection;
- every edge of that relation is important;
- the relation is uniquely important;
- other relations must be inactive.

A protective relation can have a high gate because the mechanism is relevant, even when its transformed message later reduces a risk representation.

## 20.2 Gate versus prior

A prior is an initial or contextual expectation expressed in logit space. The final gate remains jointly determined by learned neural logits and the prior contribution.

## 20.3 Gate versus edge attention

The gate operates at node × relation resolution:

```text
Which mechanism is active for this target node?
```

Edge attention operates at edge resolution:

```text
Which specific incoming connection matters within its attention group?
```

Both should be exported for pathway explanations.

## 20.4 Control relations

Control relations remain present on the exact axis and are marked by `control_relation_mask`. They are not removed, assigned special mathematical treatment, or prevented from receiving learned gate values.

Their interpretation and explanation policy should be handled by evaluation and explanation modules.

---

# 21. Failure modes and expected exceptions

## 21.1 Configuration errors

- unknown activation or scope: `ValueError`;
- canonical but unimplemented activation or scope: `NotImplementedError`;
- non-Boolean flags: `TypeError`;
- nonpositive dimensions: `ValueError` or `TypeError`;
- both input sources disabled: `ValueError`.

## 21.2 Axis and registry errors

- relation names reordered: `ValueError`;
- stable IDs changed or reordered: `ValueError`;
- compiled-registry fingerprint mismatch: `ValueError`;
- control mask mismatch: `ValueError`;
- incomplete or invalid family metadata: `ValueError`;
- wrong device for axis masks: `ValueError`.

## 21.3 Tensor errors

- wrong rank or shape: `ValueError`;
- wrong dtype: `ValueError`;
- device mismatch: `ValueError`;
- NaN or infinity: `ValueError` or `FloatingPointError`;
- activation changes shape, dtype, or device: `RuntimeError`.

## 21.4 Prior errors

- no compiled prior artifact: `ValueError`;
- prior relation axis mismatch: `ValueError`;
- source registry mismatch: `ValueError`;
- missing hazard-query lineage: `ValueError`;
- unknown runtime hazard: `ValueError`;
- hazard name and stable ID resolve differently: `ValueError`;
- node graph membership differs from hazard metadata: `ValueError`;
- invalid epsilon or negative strength: `ValueError`;
- zero strength with nonzero contribution: `ValueError` or `RuntimeError`.

## 21.5 Orchestrator errors

- non-target-node network scope: `ValueError`;
- non-sigmoid activation: `ValueError`;
- component implementation type mismatch: `TypeError`;
- edge lookup produces invalid values: `RuntimeError` or `FloatingPointError`.

---

# 22. Ablation ladder

The subsystem supports a clear research ladder:

```text
A. no relation gate
B. uniform gate supplied outside this package
C. node-state-only neural gate
D. hazard-only neural gate
E. node-state + hazard neural gate
F. neural gate + zero-strength prior diagnostics
G. neural gate + weak prior-logit contribution
H. relation bias disabled
I. LayerNorm disabled
```

Key comparisons include:

- hazard-conditioned versus hazard-blind gating;
- node-specific versus graph/hazard-only gating;
- learned gates with and without priors;
- real relations versus control relations;
- gate-enabled versus gate-removed message passing.

Do not claim prior benefit without comparing against the same network with priors disabled or strength zero.

---

# 23. Extension rules

## 23.1 Adding a new activation

A new activation requires:

1. a canonical constant;
2. inclusion in the implemented-capability list;
3. explicit functional implementation;
4. output-range and equation validation;
5. updated activation architecture identity;
6. tests for unknown versus canonical-unimplemented behavior;
7. scientific justification for whether relation channels compete.

Do not map a new activation name to sigmoid as a fallback.

## 23.2 Adding a new gate scope

A graph-level, edge-level, source-node, or global scope requires a distinct lookup and schema contract. It must not be inserted behind the existing `target_node` name.

## 23.3 Adding true family-level gating

A future family-level model should explicitly define:

- a dense family axis;
- relation-to-family projection;
- family score parameterization;
- expansion from family gates to exact relations;
- hierarchy overlap semantics;
- separate output schemas and fingerprints;
- appropriate ablations against exact-relation gating.

`RelationGateAxis` already preserves family metadata, but V2.0 does not perform this computation.

## 23.4 Adding hierarchical gates

A hierarchical extension may combine family and exact-relation gates, for example:

```text
exact_gate[n, r]
    = family_gate[n, family(r)]
      * within_family_gate[n, r]
```

Such an equation changes model interpretation and must be implemented as an explicit new mode.

## 23.5 Adding new prior behavior

Prior values, inheritance, evidence, confidence, and applicability belong in `relations/hazard_relation_priors.py`. This package should continue consuming a compiled immutable artifact.

## 23.6 Adding a regularization loss

A new loss belongs in the training subsystem. It should consume:

- gate outputs;
- prior means;
- regularization masks;
- regularization weights;
- an explicit coefficient and formula.

Do not hide training loss inside the prior builder or gate orchestrator.

---

# 24. Testing checklist

## 24.1 Axis tests

- construct from source inputs;
- reject empty or duplicate relation names;
- reject duplicate or negative stable IDs;
- reject non-Boolean control mask;
- reject registry fingerprint mismatch;
- reject reordered names or IDs;
- validate complete family metadata;
- reject unrepresented families;
- verify semantic, value, and combined fingerprints.

## 24.2 Gate-network tests

- node-state plus hazard-query path;
- node-state-only path;
- hazard-only path;
- both inputs disabled rejection;
- exact output shape `[N, R]`;
- relation score equals manual dot-product formula;
- optional relation bias behavior;
- LayerNorm and identity paths;
- Xavier/zero initialization expectations;
- device and dtype mismatch rejection;
- nonfinite parameter rejection;
- gradient flow to relation embeddings and context encoder;
- architecture and parameter fingerprint sensitivity.

## 24.3 Prior-builder tests

- exact registry alignment;
- graph hazard name and stable-ID alignment;
- node-aligned hazard lookup consistency;
- unknown hazard rejection;
- packed graphs with different hazards;
- graph-to-node row expansion;
- exact zero contribution at strength zero;
- scaled logit contribution at positive strength;
- regularization mask and weight consistency;
- resolution-summary count equals `N * R`;
- epsilon boundary rejection;
- parameter-free identity.

## 24.4 Activation tests

- sigmoid numerical equality;
- no-prior path reuses neural logits;
- prior path performs exact addition;
- same-source-object requirement;
- axis mismatch rejection;
- output range `[0, 1]`;
- canonical-unimplemented activation rejection;
- parameter-free fingerprint stability.

## 24.5 Orchestrator tests

- complete no-prior forward path;
- complete prior-enabled forward path;
- exact edge lookup by target and relation;
- empty edge set;
- output shape and metadata preservation;
- cached compatible axis;
- incompatible cached axis rejection;
- regularization-weight exposure;
- architecture and combined parameter fingerprints;
- gradient flow from edge gates back to network parameters.

## 24.6 Tests

- different hazards produce different relation-gate distributions;
- relation ordering permutations are rejected rather than silently accepted;
- control-relation gates are retained and auditable;
- prior strength changes logits predictably;
- learned logits can override weak priors;
- gates do not sum to one unless coincidentally;
- several relations can be simultaneously active.

---

# 25. Known bounded V2.0 limitations

The current package intentionally does not provide:

- softmax or sparsemax competition;
- family-pooled trainable gates;
- hierarchical family-to-relation gating;
- edge-specific gates;
- source-target pair conditioning inside the gate network;
- edge-attribute conditioning;
- temporal gate recurrence;
- uncertainty over gate values;
- learned prior strength;
- built-in prior regularization loss;
- causal interpretation;
- automatic unknown-hazard fallback;
- sparse storage of `[N, R]` gates.

These are possible research extensions, not hidden features of the current implementation.

---

# 26. Integration checklist

Before calling `RelationGate` in a model forward pass, verify:

1. `FunctionalMessagePassingInputs` is fully validated.
2. The fused node state has width expected by the gate network.
3. A node-aligned hazard query is present when enabled.
4. Relation names and stable IDs come from the exact compiled registry.
5. `edge_relation_index` uses dense compiled indices, not stable IDs.
6. `target_index` and `edge_relation_index` are aligned edge vectors.
7. Control-relation metadata is preserved.
8. The gate module and inputs share device and dtype.
9. Compiled priors match the same relation registry when enabled.
10. Hazard-query lineage preserves graph-level hazard identity when priors are enabled.
11. Architecture and registry fingerprints are persisted with checkpoints.
12. Downstream message construction treats gates as one factor, not as edge attention.

---

# 27. Summary

The `relation_family_gate` package is the hazard-conditioned routing controller of bounded V2.0 functional message passing.

Its key guarantees are:

```text
exact compiled relation identity
+ target-node-specific neural context
+ node-aligned hazard conditioning
+ optional auditable prior logits
+ independent sigmoid activation
+ exact target-node/relation edge lookup
+ immutable metadata and fingerprints
```

The package deliberately keeps relation mechanisms independent rather than forcing them into a probability simplex. It preserves semantic family metadata without confusing family identity with the exact trainable relation axis. Its public orchestrator returns both node–relation gate matrices and edge-aligned gate values while retaining the registry, hazard, control, and lineage information needed for reproducible experiments and pathway explanations.
