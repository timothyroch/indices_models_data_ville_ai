# Functional Message-Passing Contract

**Target path**

```text
urban_resilience_models/
└── v2_hazard_conditioned_functional_ugnn/
    └── functional_message_passing/
        └── FUNCTIONAL_MESSAGE_PASSING_CONTRACT.md
```

**Status:** bounded architecture-review draft  
**Scope:** approved V2.0 functional message-passing baseline  
**Implementation status:** no implementation is asserted by this document  
**Review gate:** resolve the questions in Section 10 before generating
`functional_message_passing/schemas.py`

---

## 0. Review scope and governing sources

This review inspected the current contracts owned by:

```text
constants.py
config.py
schemas.py                         # UrbanGraphBatch and graph-edge contracts
relations/relation_types.py
relations/relation_registry.py
relations/hazard_relation_priors.py
fusion/schemas.py
fusion/node_state_fusion.py
hazard/hazard_query_encoder.py
functional_message_passing roadmap
```

The design follows four existing repository rules:

1. Stable ontology IDs are not model tensor indices.
2. A compiled registry assigns deterministic dense runtime indices.
3. Metadata-bearing upstream results are retained rather than reduced to bare
   tensors at subsystem boundaries.
4. Configuration may recognize future research modes without claiming that
   those modes are implemented.

The document deliberately does **not** define basis decomposition, low-rank
relation adapters, sparse gates, hierarchical gates, mixture-of-experts
routing, bilinear attention, uncertainty propagation, or causal attribution
as implemented V2.0 capabilities.

---

## 1. Symbols and exact tensor contracts

### 1.1 Symbols

| Symbol | Meaning |
|---|---|
| `N` | Number of nodes in the packed batch |
| `E` | Number of directed stored edges |
| `B` | Number of packed graphs/scenarios |
| `R` | Number of active compiled relation identities |
| `F` | Number of semantic ontology families represented by those relations |
| `H` | Node-state and message width |
| `Q` | Hazard-query width |
| `D_e` | Edge-attribute width |
| `A` | Number of attention heads |
| `L` | Number of message-passing layers |

Unless a contract explicitly says otherwise:

- index tensors use `torch.long`;
- masks use `torch.bool`;
- neural values use one floating-point dtype;
- every tensor participating in one forward pass is on one device;
- no module performs hidden device movement;
- no module silently casts mismatched floating-point dtypes;
- all floating-point inputs and outputs must be finite.

The default research precision is `torch.float32`. `float16` and `bfloat16`
are permitted only through the external runtime precision/AMP policy.

---

### 1.2 Primary tensors

| Tensor or field | Shape | Dtype | Scope | Required alignment |
|---|---:|---|---|---|
| `node_state` | `[N, H]` | floating | node | Row `i` is node `i` in the packed graph |
| `node_batch_index` | `[N]` | `torch.long` | node → graph | Value in `[0, B-1]`; every graph has at least one node |
| `edge_index` | `[2, E]` | `torch.long` | edge endpoints | Row `0` is source; row `1` is target |
| `source_index` | `[E]` | `torch.long` | edge | Exact view/equivalent of `edge_index[0]` |
| `target_index` | `[E]` | `torch.long` | edge | Exact view/equivalent of `edge_index[1]` |
| `edge_batch_index` | `[E]` | `torch.long` | edge → graph | Equals graph membership of both endpoints |
| `edge_relation_index` | `[E]` | `torch.long` | edge → compiled relation | Values in `[0, R-1]` |
| `relation_stable_ids` | length `R` metadata | Python integers | compiled relation | Ordered exactly like compiled relation indices |
| `relation_names` | length `R` metadata | strings | compiled relation | Ordered exactly like compiled relation indices |
| `relation_family_index_by_relation` | `[R]` | `torch.long` | relation → family | Values in `[0, F-1]` |
| `edge_relation_family_index` | `[E]` | `torch.long` | edge → family | Derived as `relation_family_index_by_relation[edge_relation_index]` |
| `family_stable_ids` | length `F` metadata | Python integers | semantic family | Canonical family order |
| `family_names` | length `F` metadata | strings | semantic family | Canonical family order |
| `control_relation_mask` | `[R]` | `torch.bool` | compiled relation | True only for registry-declared controls |
| `control_edge_mask` | `[E]` | `torch.bool` | edge | Derived from `control_relation_mask[edge_relation_index]` |
| `edge_attributes` | `[E, D_e]` | floating | edge | Row-aligned with `edge_index`; optional |
| `semantic_edge_weight` | `[E]` | floating | edge | Optional externally supplied coefficient |
| `hazard_query` | `[N, Q]` baseline | floating | node | Row-aligned with `node_state` |
| graph hazard query | `[B, Q]` accepted upstream | floating | graph | Must be explicitly expanded using `node_batch_index` before node-scoped use |
| `relation_gate_logits` | `[N, R]` baseline recommendation | floating | target node × relation | Column order equals compiled relation order |
| `relation_gate_values` | `[N, R]` | floating | target node × relation | Sigmoid values in `[0,1]` |
| `edge_gate_values` | `[E]` | floating | edge | `relation_gate_values[target_index, edge_relation_index]` |
| `attention_logits_by_head` | `[E, A]` | floating | edge × head | Row-aligned with edges |
| `attention_weights_by_head` | `[E, A]` | floating | edge × head | Group-normalized per head |
| `edge_attention` | `[E]` | floating | edge | Result of configured head reduction |
| `structural_normalization` | `[E]` | floating | edge | Graph-structural coefficient |
| `transformed_source_state` | `[E, H]` | floating | edge | Source node state after relation transform |
| `edge_message` | `[E, H]` | floating | edge | Final message before aggregation |
| `incoming_edge_count` | `[N]` | `torch.long` | target node | Number of retained incoming edges |
| `node_aggregate` | `[N, H]` | floating | node | Reduction of edge messages by target |
| `updated_node_state` | `[N, H]` | floating | node | Layer output |
| `attention_group_id` | `[E]` | `torch.long` | edge → attention group | Deterministic encoding of grouping keys |
| `attention_group_count` | `[N * R]` or compact equivalent | integer | group | Diagnostic/grouped-operation metadata |

### 1.3 Naming rule for relation tensors

The top-level graph contract currently uses the field name
`edge_relation_type`. At the functional message-passing boundary, its semantic
meaning must be made explicit:

```text
edge_relation_type from UrbanGraphBatch
    ==
dense edge_relation_index into CompiledRelationRegistry
```

It must **never** contain sparse stable ontology IDs.

The FMP schemas should prefer the explicit name `edge_relation_index` while
retaining the original graph batch as source metadata. An adapter or
constructor may copy/alias `UrbanGraphBatch.edge_relation_type`, but it must
not remap values without preserving the compiled registry and its
fingerprint.

---

### 1.4 Alignment with fused node state

The initial node state is obtained from `NodeStateFusionOutput`:

```text
NodeStateFusionOutput.fused_state: [N, H]
NodeStateFusionOutput.alignment.item_count == N
```

The FMP input contract must retain the complete `NodeStateFusionOutput`, or at
minimum retain its:

```text
NodeAlignment
encoder architecture fingerprint
lineage fingerprint
source inputs
```

The following must agree exactly:

```text
fused_state row count
UrbanGraphBatch node count
NodeAlignment item count
node_batch_index values
external node ordering
hazard-query node ordering
```

A bare `[N, H]` tensor without alignment provenance is insufficient at the
public subsystem boundary.

---

### 1.5 Hazard-query scope

`HazardQueryEncoding` may originate from:

```text
HazardEmbeddingLookup             → graph/item rows
NodeAlignedHazardEmbeddingLookup  → node rows
```

The approved baseline gate scope is target-node scope, and hazard-conditioned
edge attention also consumes target-node context. Therefore the computational
hazard tensor used by these mechanisms is:

```text
q_node: [N, Q]
```

A graph-scoped query `[B, Q]` is valid only when:

```text
node_batch_index: [N]
graph_count == B
q_node = q_graph[node_batch_index]
```

That expansion must be explicit and validated. Ordinary PyTorch broadcasting
is forbidden. The original `HazardQueryEncoding` remains attached to outputs
for lineage and scope diagnostics.

---

## 2. Relation identity versus relation-family identity

### 2.1 Stable relation identity

A canonical `RelationSpec.relation_id` is a sparse, versioned ontology
identity. Examples may be numerically grouped by semantic domain, but their
numeric values are not usable as embedding or parameter-table rows.

```text
stable relation ID:
    durable within the declared registry version

dense relation index:
    position 0, 1, ..., R-1 in one CompiledRelationRegistry
```

The compiled relation order is deterministic and follows ascending stable
relation ID, not caller order.

Every model artifact involving a relation-indexed tensor must preserve:

```text
relation_index
→ stable relation_id
→ canonical relation name
→ registry version/fingerprint
```

---

### 2.2 Semantic relation-family identity

The current registry contains a formal parent-child hierarchy. A semantic
family is the root ancestor of a relation, including the root itself.

Proposed deterministic derivation:

```text
family(relation) = highest ancestor in the canonical relation hierarchy
```

Proposed family ordering:

```text
ascending stable relation ID of the family root
```

This creates:

```text
relation_family_index_by_relation: [R]
family_names:                       length F
family_stable_ids:                  length F
```

This mapping is useful for:

- diagnostics;
- pathway summaries;
- hierarchical ablations;
- possible future hierarchical gating;
- possible family-level attention grouping.

It must be compiled once from the same source registry used to compile
relations. It must not be inferred independently in several neural modules.

---

### 2.3 Bounded baseline recommendation

For V2.0, the **trainable gate axis should remain the active compiled relation
axis `R`**, not a newly pooled family axis `F`.

Thus:

```text
relation_gate_values: [N, R]
```

This recommendation is bounded and compatible with:

- `RelationConfig.active_relation_names`;
- `CompiledRelationRegistry`;
- `CompiledHazardRelationPriors`, whose columns are relation-aligned;
- the current normalization vocabulary
  `target_node_relation`;
- exact relation-specific transforms and ablations.

The semantic family mapping is retained as metadata but does not create a
second trainable hierarchy in the first implementation.

A true family-level gate `[N, F]` remains a recognized future research mode.
It requires an explicit prior-pooling rule and separate configuration
vocabulary before implementation.

---

### 2.4 Control and placebo relations

`random_placebo` and any other registry-declared control relation are genuine
edge relation identities and may appear in a compiled registry when allowed by
configuration.

They must retain:

```text
is_control=True
diagnostic-only or excluded explanation policy
control mask in all outputs
```

They must not be presented as substantive urban pathways.

`identity_no_edge` is not a relation identity. It is an experiment/topology
mode represented by `E = 0` or disabled message passing. It must not receive:

- a stable relation ID;
- a dense relation index;
- a gate column;
- an attention group;
- a prior column.

---

### 2.5 Parent-child overlap

The safe baseline compilation policy is `REJECT_OVERLAP`.

A broad parent and one of its descendants must not both be active unless an
explicit experiment uses another hierarchy policy and proves their edge sets
do not duplicate the same mechanism.

The FMP subsystem consumes an already compiled registry. It does not decide
hierarchy policy during forward execution.

---

## 3. Baseline mathematical contract

For each stored directed edge

```text
e = (s_e → t_e)
```

let:

```text
r_e = dense compiled relation index
h_s = source node state
h_t = target node state
q_t = node-aligned hazard query
```

### 3.1 Relation transform

The source state is transformed first:

\[
u_e = T_{r_e}(h_{s_e}) \in \mathbb{R}^{H}
\]

Recognized transform modes:

```text
shared
per_relation
```

The dispatcher chooses one implementation. No transform implementation owns
gating, attention, structural normalization, or aggregation.

---

### 3.2 Structural normalization

The graph-structural coefficient is computed independently:

\[
n_e = \operatorname{Norm}(e, \mathcal{G})
\]

The current bounded baseline is:

\[
n_e = 1
\]

because `edge_normalization_type = none`.

Source-degree, target-degree, and symmetric normalization may be recognized
future modes, but are not asserted as implemented here.

---

### 3.3 Relation gate

Under the recommended V2.0 relation-channel contract:

\[
G = \operatorname{Gate}(q_{\text{node}}, h)
\quad\text{with}\quad
G \in [0,1]^{N \times R}
\]

and the edge-aligned gate is:

\[
g_e = G_{t_e,r_e}
\]

The baseline scope is target-node scope and the baseline activation is
sigmoid. Several relations may be active simultaneously; they do not compete
through a softmax.

Configured hazard-relation priors may contribute to gate logits only through
the dedicated prior-integration module. Priors never hard-mask a relation
unless an explicit future policy is introduced.

---

### 3.4 Edge attention

For each head `a`:

\[
\ell_{e,a} =
S_a(h_{s_e}, h_{t_e}, q_{t_e}, r_e, x_e)
\]

where `x_e` denotes optional edge attributes used by the selected score
function.

The score function returns:

```text
attention_logits_by_head: [E, A]
```

For the current configured normalization name
`target_node_relation`, the group key is:

\[
\gamma(e) = (t_e, r_e)
\]

and grouped softmax is applied independently for every head:

\[
\alpha_{e,a}
=
\frac{\exp(\ell_{e,a}-m_{\gamma(e),a})}
{\sum_{j:\gamma(j)=\gamma(e)}
 \exp(\ell_{j,a}-m_{\gamma(e),a})}
\]

where the maximum subtraction is required for numerical stability.

Head reduction produces:

\[
\alpha_e =
\operatorname{HeadReduce}
(\alpha_{e,1},\ldots,\alpha_{e,A})
\]

The initial baseline uses one head, making the reduction structurally trivial.

---

### 3.5 Semantic edge coefficient

When `UrbanGraphBatch.semantic_edge_weight` is present:

\[
w_e = \text{semantic edge weight}_e
\]

Otherwise:

\[
w_e = 1
\]

This coefficient is data-provided and is distinct from both structural
normalization and learned attention.

It must not be silently interpreted as an attention logit or degree
normalizer.

---

### 3.6 Message construction

After all factors have been computed and validated, the message builder forms:

\[
m_e =
u_e
\cdot n_e
\cdot g_e
\cdot \alpha_e
\cdot w_e
\]

with scalar factors broadcast only across the final message feature axis.

Exact implementation sequence:

```text
1. relation transform
2. structural-normalization computation
3. relation-gate computation and edge lookup
4. edge-attention scoring and grouped normalization
5. semantic-edge-weight resolution
6. explicit message-factor multiplication
7. target-node aggregation
```

The individual factors must remain independently retainable when diagnostic
capture is enabled.

---

### 3.7 Aggregation

For each target node `i`, let:

```text
I(i) = {e : t_e = i}
d_i  = |I(i)|
```

Configured mean aggregation is:

\[
a_i =
\begin{cases}
\frac{1}{d_i}\sum_{e\in I(i)} m_e, & d_i > 0 \\
0, & d_i = 0
\end{cases}
\]

Output:

```text
node_aggregate: [N, H]
```

Aggregation is grouped only by target node. It does not regroup by relation,
family, hazard, or graph.

---

### 3.8 Layer update

The bounded one-layer update is:

\[
\tilde{h}_i =
\begin{cases}
h_i + \operatorname{Dropout}(a_i),
& \text{residual enabled} \\
\operatorname{Dropout}(a_i),
& \text{residual disabled}
\end{cases}
\]

and:

\[
h'_i =
\begin{cases}
\operatorname{LayerNorm}(\tilde{h}_i),
& \text{layer norm enabled} \\
\tilde{h}_i,
& \text{layer norm disabled}
\end{cases}
\]

The layer preserves shape `[N, H]`.

`stack.py` applies this contract repeatedly for `L` layers. It does not change
relation, edge, node, or graph alignment.

---

## 4. Explicit disabled-mechanism identities

Disabled mechanisms must use mathematical identities, not an undocumented
substitute.

| Mechanism | Disabled representation | Edge coefficient |
|---|---|---:|
| Structural normalization | no normalization computation required | `n_e = 1` |
| Relation gate | gate output may be `None`; materialize ones only for diagnostics | `g_e = 1` |
| Edge attention | no grouped softmax is performed | `α_e = 1` |
| Semantic edge weights absent | no data coefficient | `w_e = 1` |
| Residual disabled | omit addition of original state | not applicable |
| Layer norm disabled | return pre-normalized update | not applicable |

Important distinction:

```text
attention disabled
    → multiplicative identity 1

attention enabled with uniform mode
    → grouped uniform coefficient 1 / group_size
```

Those are not equivalent.

Likewise:

```text
gate disabled
    → multiplicative identity 1

gate enabled with zero logits and sigmoid
    → coefficient 0.5
```

Those are not equivalent.

No module may silently replace a disabled mechanism with the enabled
mechanism's neutral parameter initialization.

---

## 5. Grouping semantics

### 5.1 Grouped operations

`segment_ops.py` owns generic tensor reductions and must support:

```text
segment_sum
segment_mean
segment_max
grouped_softmax
group counts
```

It knows nothing about hazards, relation registries, controls, or model
semantics. Callers provide group IDs and `num_segments`.

---

### 5.2 Attention grouping

Under the current configuration vocabulary, attention is normalized by:

```text
target node + exact compiled relation index
```

A deterministic flat group ID is:

\[
\operatorname{group\_id}(e)
=
t_e \cdot R + r_e
\]

with range:

```text
0 ≤ group_id < N * R
```

Properties:

- each head is normalized independently;
- every nonempty group sums to one per head, within floating tolerance;
- a one-edge group receives weight exactly one, up to dtype precision;
- absent groups create no synthetic edges and no NaNs;
- packed graphs are safe because node indices are globally offset and edges
  cannot cross graphs;
- control relations are normalized like other relations but remain marked as
  controls in metadata.

A future family-level normalization mode would use:

\[
t_e \cdot F + f_e
\]

but must have a distinct configuration constant. It must not silently reuse
the existing exact-relation mode name.

---

### 5.3 Aggregation grouping

Aggregation groups only by:

```text
target_index
```

A deterministic aggregation group ID is simply `target_index`, with
`num_segments=N`.

Mean aggregation divides by the number of retained incoming edges, not by:

- the sum of attention weights;
- the number of relation identities;
- the number of semantic families;
- graph size;
- source degree.

This denominator must be exported as `incoming_edge_count` when diagnostics
are captured.

---

### 5.4 Masks and retained edges

Any future edge mask must be applied before:

- attention group counts;
- grouped softmax;
- aggregation counts.

Masked edges do not participate with zero logits. They are absent from the
effective edge set.

The baseline FMP schema should preserve both:

```text
original edge count
effective edge count
```

when masking is introduced. No mask is required by the first implementation.

---

## 6. Boundary and failure behavior

### 6.1 Zero-node and empty-graph behavior

The existing `UrbanGraphBatch` contract requires:

```text
N > 0
B > 0
every packed graph contains at least one node
```

Therefore a zero-node graph is invalid at the model boundary.

The phrase **empty graph** in FMP tests means:

```text
N > 0
E = 0
```

This is valid.

For `E = 0`:

```text
transformed_source_state: [0, H]
structural_normalization: [0]
edge_gate_values:         [0] or None when disabled
attention logits:         [0, A] or None when disabled
edge_attention:           [0] or None when disabled
edge_message:             [0, H]
incoming_edge_count:      zeros [N]
node_aggregate:           zeros [N, H]
```

The layer then applies only its configured residual and normalization path.

Generic `segment_ops.py` should independently support empty tensors and
`num_segments=0`, even though full model batches do not contain zero nodes.

---

### 6.2 Zero-edge relation groups

A compiled relation may have no edges in a particular batch.

Required behavior:

- no error solely because the group is absent;
- no dummy edge is inserted;
- no softmax is evaluated on an empty slice;
- no NaN or infinity is produced;
- relation-gate columns still exist because they are architecture-level;
- diagnostics report edge count zero for that relation;
- prior and parameter columns remain aligned to the compiled registry.

Whether all compiled relations must appear in a data artifact remains a
relation-validation policy, not an FMP forward requirement.

---

### 6.3 Isolated nodes

A node with no incoming edges receives:

```text
node_aggregate = zero vector
incoming_edge_count = 0
```

Its final state is governed by layer configuration:

- residual on, norm off: unchanged;
- residual on, norm on: normalized original state;
- residual off: zero aggregate followed by optional layer norm.

The result must remain finite.

---

### 6.4 Single-edge attention groups

For enabled grouped attention, a group containing exactly one edge must yield:

```text
attention weight = 1
```

for every head, independent of the raw finite logit.

---

### 6.5 Packed graphs

Required invariants:

```text
node_batch_index: [N], values 0..B-1
edge_batch_index: [E], values 0..B-1
node_batch_index[source_index]
    == node_batch_index[target_index]
    == edge_batch_index
```

Cross-graph edges are rejected by the bounded baseline.

Graph-level hazard queries are expanded only through `node_batch_index`.
No relation, attention, or aggregation group may combine entries from
different packed graphs.

A packed graph may have nodes but no edges. It remains valid.

---

### 6.6 Invalid IDs and indices

Reject before parameter lookup or scatter operations:

- negative node indices;
- node indices `>= N`;
- negative graph indices;
- graph indices `>= B`;
- negative relation indices;
- relation indices `>= R`;
- negative family indices;
- family indices `>= F`;
- relation-to-family mapping with wrong length;
- edge relation ordering that disagrees with the compiled registry;
- stable IDs supplied where dense relation indices are expected.

Errors must identify the invalid field, observed range, and valid range.

---

### 6.7 Device mismatches

All tensor inputs to one FMP layer must share one device, including:

```text
node state
node_batch_index
edge_index
edge_batch_index
edge relation indices
relation-to-family mapping
hazard query
edge attributes
semantic edge weights
masks
compiled tensorized priors
```

The layer and every trainable submodule must be on that device.

No FMP module calls `.cpu()`, `.cuda()`, or `.to(device)` internally during
forward execution.

---

### 6.8 Dtype mismatches

Required index dtypes:

```text
edge_index                       torch.long
node_batch_index                 torch.long
edge_batch_index                 torch.long
edge_relation_index              torch.long
relation_family_index_by_relation torch.long
edge_relation_family_index       torch.long
```

Required mask dtype:

```text
torch.bool
```

All model-consumed floating inputs in one forward pass must use the same
floating dtype unless an explicitly documented projection boundary is
introduced. Integer edge attributes are not concatenated directly into neural
score functions without explicit encoding.

No implicit promotion from `float32` to `float64`, or from integer values to
floating values, is accepted at the public contract boundary.

---

### 6.9 Non-finite values

Reject NaN and infinity in:

- node states;
- hazard queries;
- edge attributes consumed by neural modules;
- semantic edge weights;
- prior tensors;
- transform outputs;
- gate logits and values;
- attention logits and weights;
- structural coefficients;
- edge messages;
- aggregates;
- updated node states;
- scalar regularization terms.

Empty tensors are finite by definition and must not fail finiteness checks.

---

### 6.10 Numerical stability

Grouped softmax must subtract the group maximum before exponentiation.

Segment max for an absent group may use an internal sentinel, but that
sentinel must never be exposed as a valid result for an absent group.

Mean operations must clamp or branch on zero counts rather than divide by
zero.

---

## 7. Retainable diagnostics, ablations, attribution inputs, and provenance

Retention is optional because edge-level tensors may be expensive. The
configuration field `capture_intermediate_messages` controls the main
message-path capture. Explanation capture may independently request gate or
attention tensors.

### 7.1 Required retainable tensors

| Stage | Retainable tensors |
|---|---|
| Input alignment | `edge_index`, `node_batch_index`, `edge_batch_index` |
| Relation identity | relation indices, stable IDs, names, family mapping, control masks |
| Relation transform | `transformed_source_state [E,H]` |
| Structural normalization | `structural_normalization [E]` |
| Gate | logits `[N,R]`, values `[N,R]`, edge lookups `[E]`, prior contribution `[N,R]` when used |
| Attention | logits `[E,A]`, grouped weights `[E,A]`, reduced weights `[E]`, group IDs and counts |
| Message construction | each multiplicative scalar factor and final messages `[E,H]` |
| Aggregation | incoming counts `[N]`, aggregate `[N,H]` |
| Layer update | pre-residual, post-residual, pre-normalization, final node state |
| Stack | optional state after each layer `[N,H]` |

### 7.2 Required metadata

Every metadata-bearing output should preserve or reference:

```text
schema version
layer index
active relation order
stable relation IDs
semantic family order
control/placebo masks
compiled relation-registry fingerprint
source relation-registry semantic fingerprint
hierarchy compilation policy
hazard-query lineage fingerprint
node-state-fusion lineage fingerprint
FMP architecture fingerprint
parameter fingerprint where relevant
input lineage fingerprint
attention grouping mode
gate scope and activation
transform mode
normalization mode
aggregation mode
head count and reduction
capture policy
```

### 7.3 Ablation support

The retained contract must allow exact identification of:

```text
shared vs per-relation transform
normalization enabled vs identity
gate enabled vs identity
attention disabled vs uniform vs learned
real relations vs control/placebo relations
one layer vs multiple layers
residual on vs off
layer norm on vs off
```

An ablation must change explicit architecture metadata and therefore the
architecture fingerprint.

### 7.4 Attribution limits

Intermediate magnitudes, gate values, and attention weights are diagnostic
model traces. They are not automatically causal importance scores.

The FMP subsystem provides tensors needed by future attribution modules, but
it does not label them as causal contributions.

### 7.5 Reproducibility

Outputs should preserve source objects or immutable source identities rather
than copying only names.

Tensor-value fingerprints may be captured for bounded research artifacts, but
large edge tensors should not be hashed during every ordinary training
forward pass unless explicitly enabled.

---

## 8. Responsibility map for the approved 24-file roadmap

| # | Module | Owns | Explicitly does not own |
|---:|---|---|---|
| 1 | `functional_message_passing/schemas.py` | Immutable typed FMP inputs/outputs, alignment validation, schema versions, lineage metadata | Neural layers, registry construction, scatter mathematics |
| 2 | `functional_message_passing/segment_ops.py` | Generic segment sum/mean/max, counts, grouped softmax | Relation semantics, hazards, model configuration |
| 3 | `relation_transforms/shared_transform.py` | One shared node-to-message transform | Relation dispatch, gates, attention, aggregation |
| 4 | `relation_transforms/per_relation_transform.py` | Independently parameterized transforms indexed by compiled relation | Registry compilation, family gating, attention |
| 5 | `relation_transforms/relation_transforms.py` | Mode validation, implementation dispatch, metadata-bearing transform output | Internal mathematics of every transform mode |
| 6 | `relation_transforms/__init__.py` | Public relation-transform exports | Runtime logic |
| 7 | `edge_normalization.py` | Graph-structural edge coefficients and normalization-mode dispatch | Learned attention, semantic edge weights, message aggregation |
| 8 | `aggregators.py` | Reduction of final edge messages by target node; counts and aggregate output | Message scoring, gating, hazard use |
| 9 | `relation_family_gate/schemas.py` | Gate-specific typed outputs, scope, ordering, prior traces, control masks | Gate neural network |
| 10 | `relation_family_gate/activations.py` | Sigmoid baseline and clearly named future activation dispatch | Gate logits, priors, relation lookup |
| 11 | `relation_family_gate/relation_priors.py` | Alignment and combination of compiled prior contributions with gate logits | Prior registry definition, neural gate prediction |
| 12 | `relation_family_gate/gate_network.py` | Neural prediction of target-node relation logits from permitted inputs | Activation, prior compilation, edge lookup |
| 13 | `relation_family_gate/relation_family_gate.py` | Gate orchestration, alignment, activation, edge lookup, output assembly | Edge attention, message construction, aggregation |
| 14 | `relation_family_gate/__init__.py` | Public gate exports | Runtime logic |
| 15 | `edge_attention/schemas.py` | Attention-specific typed scores, weights, groups, heads, lineage | Score or softmax mathematics |
| 16 | `edge_attention/score_functions.py` | Uniform and configured learned score functions | Grouped normalization, head reduction |
| 17 | `edge_attention/attention_normalization.py` | Stable grouped normalization and normalization checks | Raw score prediction, aggregation |
| 18 | `edge_attention/multihead.py` | Head organization and configured reduction | Edge grouping semantics, score-function internals |
| 19 | `edge_attention/edge_attention.py` | Score → normalize → reduce → output orchestration | Relation transforms, message building, node aggregation |
| 20 | `edge_attention/__init__.py` | Public attention exports | Runtime logic |
| 21 | `message_builders.py` | Explicit multiplication of transformed messages and scalar factors; factor traces | Computing transforms, gates, attention, or aggregates |
| 22 | `layer.py` | One complete FMP step, residual path, layer normalization, one-layer output | Multi-layer looping, prediction heads |
| 23 | `stack.py` | `L` layer orchestration, parameter-sharing policy, intermediate layer retention, stack fingerprint | Single-layer internal mathematics, prediction |
| 24 | `functional_message_passing/__init__.py` | Stable public subsystem exports | Hidden side effects, model construction |

---

## 9. Dependency and reuse map

### 9.1 Configuration: reuse, do not duplicate

From `config.py`:

```text
RelationConfig
    active_relation_names
    gate_enabled
    gate_scope
    gate_activation
    gate_hidden_dim
    use_relation_priors
    relation_prior_strength
    allow_control_relations
    allow_control_relations_in_explanations

FunctionalMessagePassingConfig
    enabled
    num_layers
    relation_transform_type
    aggregation_type
    edge_normalization_type
    attention_enabled
    attention_mode
    attention_normalization
    attention_heads
    attention_head_reduction
    residual
    layer_norm
    dropout
    capture_intermediate_messages
```

FMP modules must not define competing configuration dataclasses.

---

### 9.2 Constants: reuse, do not restate strings

Reuse canonical and implemented vocabularies for:

```text
relation transform modes
aggregation modes
edge-normalization modes
attention modes
attention-normalization modes
attention-head reductions
relation-gate scopes
relation-gate activations
control relation names
scope names
```

Local enums may wrap these constants only when they preserve exact string
values and canonical/implemented distinction.

---

### 9.3 Relation ontology and runtime mapping

Reuse:

```text
RelationSpec
RelationRegistry
RelationRegistryEntry
CompiledRelationRegistry
HierarchyCompilationPolicy
RelationExplanationPolicy
```

The FMP subsystem must consume a `CompiledRelationRegistry`; it must not:

- compile its own relation order;
- sort relation names independently;
- assign stable IDs;
- infer control status from string patterns;
- rebuild hierarchy metadata from tags.

---

### 9.4 Hazard-relation priors

Reuse:

```text
CompiledHazardRelationPriors
gate_bias_logit_matrix()
regularization_weight_matrix()
source registry fingerprints
initialization/regularization masks
```

The prior-integration module must validate exact equality of:

```text
relation names
stable relation IDs
compiled relation fingerprint
hazard ordering
```

The FMP subsystem does not own prior values, inheritance, applicability scope,
or empirical provenance.

---

### 9.5 Fused node state and node alignment

Reuse:

```text
fusion.schemas.NodeAlignment
fusion.schemas.NodeStateFusionOutput
```

The FMP input contract should retain the complete fusion output and use
`NodeAlignment` as the authoritative node-order and graph-membership contract.

It must not invent a second incompatible node-alignment class unless a
strictly broader graph-edge alignment object embeds/references the existing
one.

---

### 9.6 Hazard query

Reuse:

```text
HazardQueryEncoding
HazardQueryTensorScope
NodeAlignedHazardEmbeddingLookup
HazardEmbeddingLookup
```

The FMP subsystem preserves query lineage and performs only explicit scope
alignment required for gating/attention. It does not re-embed hazard IDs or
reconstruct scenario context.

---

### 9.7 Graph and edge contracts

Reuse from `UrbanGraphBatch`:

```text
external_node_ids
node_batch_index
edge_index
edge_relation_type
edge_attributes
semantic_edge_weight
external_edge_ids
edge_batch_index
graph_ptr
allow_cross_graph_edges
contract versions and metadata
```

The FMP schema should reference the source graph batch or preserve its
identity/fingerprint. It must not create a competing graph loader schema.

Registry membership validation remains strict at the FMP boundary because
parameter lookup requires a specific compiled registry, even if the general
graph schema performs only structural validation.

---

### 9.8 Existing source-of-truth hierarchy

```text
constants.py
    ↓ vocabulary
config.py
    ↓ selected modes
relation_types.py
    ↓ semantic relation contracts
relation_registry.py
    ↓ compiled dense relation order
hazard_relation_priors.py
    ↓ relation-aligned prior matrices
UrbanGraphBatch
    ↓ concrete packed graph tensors
NodeStateFusionOutput
    ↓ aligned initial node state
HazardQueryEncoding
    ↓ aligned hazard context
functional_message_passing/schemas.py
    ↓ typed FMP boundary
neural FMP modules
```

No arrow in this map should be reversed through import-time construction.

---

## 10. Unresolved architectural questions before `schemas.py`

### 10.1 Is the first gate axis `R` or `F`? — **blocking**

This review recommends:

```text
baseline gate axis = active compiled relations R
semantic family F = retained metadata only
```

Reasons:

- current priors are `[hazards, R]`;
- current compiled registry is relation-indexed;
- current configuration selects relation names;
- no canonical compiled family registry exists;
- no scientifically approved prior-pooling rule exists.

A true family gate `[N,F]` would require:

```text
canonical family compilation
relation-to-family mapping artifact
family-level control semantics
relation-prior pooling rule
family-level parameter and explanation identities
```

This decision must be approved before gate-output fields are frozen in
`schemas.py`.

---

### 10.2 Exact-relation or semantic-family attention grouping? — **blocking**

The approved roadmap text says:

```text
target node + relation family
```

The current configuration constant says:

```text
target_node_relation
```

This review recommends preserving the existing exact-relation meaning:

```text
group = (target node, dense relation index)
```

A family grouping mode should receive a distinct constant and tests.

The existing constant must not be silently reinterpreted.

---

### 10.3 Interaction of grouped attention with mean aggregation — **important**

The research target currently combines:

```text
attention normalized within target-relation groups
mean aggregation across all incoming edges
```

This performs two separate normalizations and makes the final scale depend on
total target indegree after attention has already normalized each relation
group.

Three defensible policies exist:

1. keep the current exact contract;
2. use sum aggregation when grouped attention is enabled;
3. aggregate within relation first, then combine relation aggregates.

This document specifies policy 1 because it matches current configuration,
but the scientific choice should be confirmed before implementing
`aggregators.py`.

It does not block shape-only schemas if both the attention output and
aggregation mode are retained explicitly.

---

### 10.4 Where is graph-to-node hazard-query expansion owned? — **blocking**

Possible owners:

```text
functional_message_passing/schemas.py
relation_family_gate/relation_family_gate.py
edge_attention/edge_attention.py
layer.py
```

The operation must happen exactly once and remain traceable.

Recommended ownership:

```text
FunctionalMessagePassingInputs exposes a validated node-aligned hazard view,
while retaining the original HazardQueryEncoding.
```

This is alignment behavior, not trainable model behavior, but it must not
create large duplicated tensors unnecessarily.

---

### 10.5 Is `semantic_edge_weight` part of the first baseline? — **important**

This review defines it as an optional independent multiplicative factor.

Before `message_builders.py`, confirm whether:

- V2.0 consumes it when present;
- V2.0 rejects it unless explicitly configured;
- it belongs inside structural normalization instead.

It must not be silently used because its scale may dominate learned gates and
attention.

The schema can preserve it without asserting that every baseline consumes it.

---

### 10.6 Should edge attributes enter the first attention score? — **not schema-blocking**

The graph contract supports `[E,D_e]`, but the current FMP configuration does
not declare an edge-attribute input dimension or encoding policy.

Recommended bounded baseline:

```text
preserve and validate edge attributes;
do not consume them in learned scores until a configured encoder contract
exists.
```

---

### 10.7 Exact layer-update transform — **not schema-blocking**

This review uses:

```text
aggregate → dropout → residual → optional layer norm
```

No extra node-update MLP is included.

Before `layer.py`, confirm whether the baseline needs an additional learned
post-aggregation update. Adding it later changes architecture identity but
does not require changing the core edge/message schemas.

---

## 11. Review conclusion

The subsystem can proceed safely after the blocking decisions in Sections
10.1, 10.2, and 10.4 are confirmed.

The recommended bounded baseline is:

```text
metadata-preserving fused node state
→ shared or per-relation source transform
→ identity structural normalization initially
→ target-node sigmoid gate over compiled relation channels
→ single-head hazard-conditioned attention
→ grouped softmax by target node and exact relation
→ explicit multiplicative message construction
→ mean aggregation by target node
→ residual
→ optional layer normalization
```

The schema and segment-operation implementation sequence remains:

```text
functional_message_passing/schemas.py
tests/test_fmp_schemas.py
functional_message_passing/segment_ops.py
tests/test_segment_ops.py
```

No advanced hierarchical gate, family-level prior pooling, sparse attention,
basis transform, or uncertainty propagation is claimed by this contract.
