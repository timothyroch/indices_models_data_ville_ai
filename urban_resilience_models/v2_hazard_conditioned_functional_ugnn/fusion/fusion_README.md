# Node-State Fusion Subsystem

**Package:** `urban_resilience_models.v2_hazard_conditioned_functional_ugnn.fusion`  
**Primary entry point:** `NodeStateFusion`  
**Implemented fusion mode:** `concat_projection`  
**Purpose:** construct one validated, metadata-preserving node representation from static, temporal-memory, hazard-memory, hazard-context, and optional node-type components.

---

## 1. Overview

The fusion subsystem sits between upstream representation encoders and the functional message-passing stack.

Upstream modules produce semantically distinct node-aligned states:

- static urban features;
- temporal urban memory;
- hazard-queried memory;
- hazard or scenario context;
- optional node-type identifiers.

The fusion subsystem validates that these objects refer to the same rows, extracts the tensors in a deterministic order, projects every active component to a common width, concatenates the projected states, and transforms the concatenation into the initial fused node state used by later model stages.

```text
static state -------------------------┐
memory encoding ----------------------┤
hazard-memory state ------------------┤
hazard query/context -----------------┤
node-type IDs -> embedding -----------┘
                   |
                   v
          NodeStateFusionInputs
                   |
                   v
       NodeStateFusion orchestrator
          |  validate policy
          |  extract components
          |  preserve lineage
          v
       ConcatProjectionFusion
          |  ComponentProjection × K
          |  concatenate
          |  final fusion MLP
          v
          NodeStateFusionOutput
                   |
                   v
     functional message passing / heads
```

This subsystem is deliberately strict. It rejects missing components, unexpected components, reordered mappings, row-count mismatches, device mismatches, non-floating state tensors, invalid node-type IDs, non-finite values, and unsupported fusion modes rather than silently adapting the inputs.

### Role

The current implementation is a controlled baseline, not an adaptive hazard-routing mechanism. Hazard-related states may be included as input components, but `concat_projection` does not use hazard context to gate or modulate other components. Hazard-conditioned relation gates, edge attention, FiLM, experts, attribution, and uncertainty propagation belong elsewhere or require future dedicated fusion implementations.

---

## 2. Files and ownership boundaries

```text
fusion/
├── component_projection.py
├── concat_projection.py
├── node_state_fusion.py
└── schemas.py
```

| File | Owns | Does not own |
|---|---|---|
| `schemas.py` | Typed input/output contracts, node alignment, lineage/value fingerprints, device-preserving reconstruction | Trainable layers, fusion mathematics, mode dispatch |
| `component_projection.py` | Projection of one dense component to the common fusion width | Component extraction, ordering, concatenation, final fusion |
| `concat_projection.py` | Ordered multi-component projection, concatenation, final fusion MLP, low-level algorithm output | Typed upstream extraction, configuration dispatch, lineage validation |
| `node_state_fusion.py` | Public orchestration boundary, configuration construction, enabled-component policy, node-type embedding, algorithm dispatch, final output assembly, legacy checkpoint migration | Schema definitions and low-level projection mathematics |

The recommended dependency direction is:

```text
schemas.py
    ^
    |
node_state_fusion.py ---> concat_projection.py ---> component_projection.py
```

`schemas.py` imports existing upstream result contracts from the memory and hazard packages. `NodeStateFusionOutput` performs a local import of `NodeStateFusionMode` during validation to avoid an import-time cycle.

---

## 3. Recommended public API

Application and model code should normally interact with only these objects:

```python
from urban_resilience_models.v2_hazard_conditioned_functional_ugnn.fusion.schemas import (
    NodeAlignment,
    NodeStateComponent,
    NodeStateFusionInputs,
    NodeStateFusionOutput,
)
from urban_resilience_models.v2_hazard_conditioned_functional_ugnn.fusion.node_state_fusion import (
    NodeStateFusion,
    NodeStateFusionMode,
)
```

Use `NodeStateFusion.from_config(model_config)` when a complete validated `ModelConfig` is available. Use the explicit constructor primarily in unit tests, isolated experiments, and low-level integration work.

The lower-level classes are public and useful for targeted tests or ablations:

```python
from urban_resilience_models.v2_hazard_conditioned_functional_ugnn.fusion.component_projection import (
    ComponentProjection,
    ComponentProjectionActivation,
    ComponentProjectionNormalization,
)
from urban_resilience_models.v2_hazard_conditioned_functional_ugnn.fusion.concat_projection import (
    ConcatProjectionFusion,
    ConcatProjectionFusionOutput,
    canonical_component_order,
)
```

Do not pass untyped dictionaries or bare tensors to `NodeStateFusion.forward`. The stable public boundary accepts only `NodeStateFusionInputs`.

---

## 4. Canonical component model

The subsystem recognizes five semantic component names, in this exact canonical order:

```python
(
    "static_state",
    "memory_state",
    "hazard_memory_state",
    "hazard_context",
    "node_type_embedding",
)
```

The constants are defined in `concat_projection.py`:

| Constant | Value | Tensor source in `NodeStateFusion` |
|---|---|---|
| `FUSION_COMPONENT_STATIC_STATE` | `"static_state"` | `inputs.static_state.values` |
| `FUSION_COMPONENT_MEMORY_STATE` | `"memory_state"` | `inputs.memory_state.memory_state` |
| `FUSION_COMPONENT_HAZARD_MEMORY_STATE` | `"hazard_memory_state"` | `inputs.hazard_memory_state.values` |
| `FUSION_COMPONENT_HAZARD_CONTEXT` | `"hazard_context"` | `inputs.hazard_context.query` |
| `FUSION_COMPONENT_NODE_TYPE_EMBEDDING` | `"node_type_embedding"` | learned lookup from `inputs.node_type_ids` |

Only enabled components are present. Their relative order always follows the canonical sequence above.

For example:

```text
Enabled: static_state, hazard_context
Order:   ("static_state", "hazard_context")

Enabled: memory_state, node_type_embedding
Order:   ("memory_state", "node_type_embedding")
```

### Why ordering is part of the contract

Concatenation is order-sensitive. Two models with identical component widths but different orders have different semantics. The implementation therefore treats ordering as architecture identity:

- `NodeStateFusion` builds an ordered component-width mapping;
- `ConcatProjectionFusion` requires the input mapping iteration order to match `component_order` exactly;
- projected outputs preserve the same order;
- architecture fingerprints include that order.

There is no hidden reordering at forward time.

---

## 5. End-to-end mathematical contract

Assume there are `K` active components. Component `k` has an item-aligned input tensor:

```text
x_k ∈ R^(N × d_k)
```

where:

- `N` is the shared item or node count;
- `d_k` is the component-specific input width;
- `D` is the configured common fusion width.

Each component receives its own learned projection:

```text
z_k = Dropout(Norm(GELU(W_k x_k + b_k)))
```

where `Norm` is either `LayerNorm(D)` or the identity operation. Therefore:

```text
z_k ∈ R^(N × D)
```

The projected components are concatenated in canonical order:

```text
z = concat(z_1, ..., z_K) ∈ R^(N × K·D)
```

The final fusion network computes:

```text
h = FinalNorm(
        Linear_out(
            Dropout(
                GELU(
                    Linear_in(z)
                )
            )
        )
    )
```

with:

```text
Linear_in:  K·D -> D
Linear_out: D   -> D
h:          N × D
```

When layer normalization is disabled, both per-component normalization and final normalization are replaced with identity operations.

---

# 6. `schemas.py`

## 6.1 Purpose

`schemas.py` defines immutable, metadata-preserving contracts for the fusion boundary. These contracts validate alignment, row counts, graph membership, device consistency, tensor finiteness, and lineage metadata before trainable fusion logic runs.

Public objects:

- `NodeAlignment`
- `NodeStateComponent`
- `NodeStateFusionInputs`
- `NodeStateFusionOutput`

Schema-version constants:

```python
NODE_ALIGNMENT_SCHEMA_VERSION = "0.1"
NODE_STATE_COMPONENT_SCHEMA_VERSION = "0.1"
NODE_STATE_FUSION_INPUT_SCHEMA_VERSION = "0.1"
NODE_STATE_FUSION_OUTPUT_SCHEMA_VERSION = "0.1"
```

All schema dataclasses use `slots=True` and `frozen=True`. This prevents field reassignment and avoids dynamic attributes. It does **not** make contained PyTorch tensors deeply immutable; callers must still avoid in-place mutation when reproducible fingerprints or lineage are required.

---

## 6.2 `NodeAlignment`

`NodeAlignment` defines what each tensor row means and, optionally, which packed graph each row belongs to.

### Constructor

```python
NodeAlignment(
    item_count: int,
    item_ids: tuple[str, ...] = (),
    node_batch_index: torch.Tensor | None = None,
    graph_count: int | None = None,
    source_fingerprint: str | None = None,
    alignment_name: str = "node_state_alignment",
    schema_version: str = "0.1",
)
```

### Fields

| Field | Contract |
|---|---|
| `item_count` | Nonnegative integer. Number of aligned rows in every component. |
| `item_ids` | Optional unique, nonempty identifiers. If supplied, length must equal `item_count`. |
| `node_batch_index` | Optional one-dimensional `torch.long` tensor of shape `[item_count]`. Maps each row to a packed-graph ID. |
| `graph_count` | Required when `node_batch_index` exists; forbidden otherwise. Must be positive. |
| `source_fingerprint` | Optional nonempty identifier for the source node-index or batch artifact. |
| `alignment_name` | Nonempty semantic name. |
| `schema_version` | Nonempty schema identifier. |

### Packed-graph invariants

When `node_batch_index` is supplied:

1. `item_count` must be greater than zero.
2. The tensor must have shape `[item_count]`.
3. Its dtype must be exactly `torch.long`.
4. Graph IDs cannot be negative.
5. Every ID must be less than `graph_count`.
6. Observed graph IDs must be exactly `0, 1, ..., graph_count - 1`.
7. Every packed graph must have at least one represented row.

Valid example:

```python
alignment = NodeAlignment(
    item_count=5,
    item_ids=("n0", "n1", "n2", "n3", "n4"),
    node_batch_index=torch.tensor([0, 0, 1, 1, 1], dtype=torch.long),
    graph_count=2,
    source_fingerprint="batch-2026-06-23",
)
```

Invalid examples include graph IDs `[0, 2]` with `graph_count=3`, a `torch.int32` membership tensor, or `graph_count` without `node_batch_index`.

### Properties and methods

| Member | Meaning |
|---|---|
| `device` | Device of `node_batch_index`, or `None` when graph membership is absent. |
| `graph_aligned` | `True` when `node_batch_index` is present. |
| `semantic_dict()` | JSON-compatible semantic metadata excluding tensor values. |
| `semantic_fingerprint()` | SHA-256 fingerprint of `semantic_dict()`. |
| `value_fingerprint()` | Exact fingerprint of `node_batch_index`, or a deterministic no-tensor marker. |
| `fingerprint()` | Combined semantic and value fingerprint. |
| `to(device)` | Reconstructs an equivalent alignment with `node_batch_index` moved to the requested device. |

### Fingerprint interpretation

`semantic_fingerprint()` identifies the declared meaning of the alignment. `value_fingerprint()` identifies the exact graph-membership values. `fingerprint()` combines both.

The tensor fingerprint includes:

- tensor name;
- dtype;
- shape;
- exact raw bytes after detaching, moving to CPU, and making the tensor contiguous.

It is device-independent but sensitive to dtype, shape, ordering, and every value.

---

## 6.3 `NodeStateComponent`

`NodeStateComponent` is the generic wrapper for a dense node-aligned state when no dedicated upstream result schema exists.

It is currently used for:

- static state;
- hazard-conditioned memory state;
- future generic fusion components until dedicated contracts are introduced.

### Constructor

```python
NodeStateComponent(
    values: torch.Tensor,
    component_name: str,
    source_fingerprint: str | None = None,
    alignment_fingerprint: str | None = None,
    schema_version: str = "0.1",
)
```

### Tensor contract

`values` must:

- be a `torch.Tensor`;
- have shape `[items, feature_dim]`;
- use a floating-point dtype;
- have `feature_dim > 0`;
- contain only finite values.

Zero rows are permitted when the surrounding alignment is not a packed graph. A zero-width feature tensor is never permitted.

### Metadata

`component_name` should use the canonical semantic name expected by its field, such as `"static_state"` or `"hazard_memory_state"`. The schema requires a nonempty name but does not itself compare the name with the field in which the object is stored; using canonical names keeps lineage metadata coherent.

`alignment_fingerprint` should normally be set to `alignment.fingerprint()`. When present, `NodeStateFusionInputs` verifies that the component was created for the same alignment.

### Properties and methods

| Member | Meaning |
|---|---|
| `item_count` | Number of rows. |
| `feature_dim` | Input feature width. |
| `device` | Device of `values`. |
| `value_fingerprint()` | Exact tensor-value fingerprint. |
| `lineage_fingerprint()` | Fingerprint combining schema, semantic name, source metadata, alignment metadata, and exact values. |
| `to(device)` | Returns a reconstructed component with `values` moved to the requested device. |

Example:

```python
alignment = NodeAlignment(
    item_count=3,
    item_ids=("tract-1", "tract-2", "tract-3"),
)

static_state = NodeStateComponent(
    values=torch.randn(3, 24),
    component_name="static_state",
    source_fingerprint="static-feature-pipeline-v4",
    alignment_fingerprint=alignment.fingerprint(),
)
```

---

## 6.4 `NodeStateFusionInputs`

`NodeStateFusionInputs` is the only accepted input type for `NodeStateFusion.forward`.

### Constructor

```python
NodeStateFusionInputs(
    alignment: NodeAlignment,
    static_state: NodeStateComponent | None = None,
    memory_state: LagMemoryEncoding | None = None,
    hazard_memory_state: NodeStateComponent | None = None,
    hazard_context: HazardQueryEncoding | None = None,
    node_type_ids: torch.Tensor | None = None,
    source_fingerprint: str | None = None,
    schema_version: str = "0.1",
)
```

### Upstream object retention

The contract preserves complete upstream result objects rather than reducing everything to tensors:

| Field | Required type | Tensor later extracted |
|---|---|---|
| `static_state` | `NodeStateComponent` | `.values` |
| `memory_state` | `LagMemoryEncoding` | `.memory_state` |
| `hazard_memory_state` | `NodeStateComponent` | `.values` |
| `hazard_context` | `HazardQueryEncoding` | `.query` |
| `node_type_ids` | `torch.long` tensor | converted by a learned `nn.Embedding` |

This preserves upstream lineage, source objects, and alignment information in the final `NodeStateFusionOutput.source_inputs`.

### Construction-time validation

The dataclass validates all of the following immediately:

1. `alignment` is a `NodeAlignment`.
2. Each optional component has the exact expected type.
3. `node_type_ids`, when present, is a one-dimensional `torch.long` tensor.
4. Every present component has `alignment.item_count` rows.
5. Generic components with an `alignment_fingerprint` match `alignment.fingerprint()`.
6. Node-aligned hazard-query graph membership matches `NodeAlignment.node_batch_index` exactly.
7. All present tensors share one device.
8. Optional source and schema fingerprints are nonempty strings.

### Hazard-context graph-alignment rule

When `hazard_context.source_embedding` is a `NodeAlignedHazardEmbeddingLookup`, the inputs must include `alignment.node_batch_index`. The source embedding's membership tensor must:

- be on the same device as the alignment membership tensor;
- be exactly equal to it.

This prevents a hazard query aligned to one packed graph layout from being fused with node states from another layout.

### Device property

`inputs.device` examines all present tensor-bearing objects:

- alignment membership;
- static state;
- memory state;
- hazard-memory state;
- hazard query;
- node-type IDs.

It returns:

- the shared device when at least one tensor exists;
- `None` when no tensor-bearing field exists;
- a `ValueError` when devices differ.

The validation is forced during construction, so cross-device inputs fail early.

### Lineage methods

`component_lineage_dict()` builds a JSON-compatible payload containing:

- fusion-input schema version;
- alignment fingerprint;
- optional top-level source fingerprint;
- lineage fingerprint for each generic component;
- upstream lineage fields for memory and hazard query;
- an exact value fingerprint of node-type IDs.

`lineage_fingerprint()` hashes that payload.

### Moving inputs between devices

```python
moved = inputs.to("cuda")
```

The method reconstructs the entire contract and moves supported tensor-bearing objects while preserving metadata.

Important behavior:

- `NodeAlignment`, generic components, node-type IDs, and `LagMemoryEncoding` tensors are moved.
- `LagMemoryEncoding` is rebuilt with its source batch, optional lag feature states, optional lag weights, architecture fingerprint, lineage fingerprint, and schema version preserved.
- `HazardQueryEncoding` is **not** reconstructed on a new device because its complete source-embedding metadata cannot safely be fabricated. Its `.query` must already be on the requested device. Otherwise `to(device)` raises `ValueError` instructing the caller to move the hazard embedding/query upstream first.
- The method changes device only; it does not expose a dtype-conversion argument.

Recommended order when hazard context is present:

```text
1. Move the memory, hazard, and other upstream modules to the target device.
2. Move or construct alignment and generic component tensors on that device.
3. Produce HazardQueryEncoding on that device.
4. Construct NodeStateFusionInputs only after every component shares the device.
5. Move NodeStateFusion to the same device and run fusion.
```

An already constructed input object containing `HazardQueryEncoding` generally cannot be used as the mechanism that moves that query to a new device; movement must occur through the owning hazard pipeline first.

---

## 6.5 `NodeStateFusionOutput`

`NodeStateFusionOutput` is the stable high-level result returned by `NodeStateFusion.forward`.

### Fields

```python
NodeStateFusionOutput(
    fused_state: torch.Tensor,
    source_inputs: NodeStateFusionInputs,
    projected_components: Mapping[str, torch.Tensor],
    fusion_mode: NodeStateFusionMode | str,
    encoder_architecture_fingerprint: str,
    lineage_fingerprint: str,
    schema_version: str = "0.1",
)
```

| Field | Meaning |
|---|---|
| `fused_state` | Final tensor of shape `[items, output_dim]`. |
| `source_inputs` | Complete metadata-preserving input object used for this output. |
| `projected_components` | Ordered mapping from active component name to its projected tensor. |
| `fusion_mode` | Normalized to `NodeStateFusionMode` during validation. |
| `encoder_architecture_fingerprint` | Identity of the fusion architecture. |
| `lineage_fingerprint` | Identity combining fusion architecture and input lineage. |
| `schema_version` | Output schema version. |

### Validation

The output requires:

- a two-dimensional, finite floating-point `fused_state`;
- row count equal to `source_inputs.item_count`;
- device equal to the source-input device when one exists;
- at least one projected component;
- every projected component to be a finite, two-dimensional floating tensor;
- projected component rows and device to match the fused state;
- nonempty architecture, lineage, and schema fingerprints;
- a valid canonical fusion mode.

The projected mapping is copied into `MappingProxyType`, preventing key insertion, deletion, or replacement. Tensor values remain ordinary tensors and should be treated as read-only diagnostics unless mutation is explicitly intended.

### Convenience properties

```python
output.item_count
output.output_dim
output.alignment
```

`output.alignment` is a direct view of `output.source_inputs.alignment`.

---

# 7. `component_projection.py`

## 7.1 Purpose

`ComponentProjection` maps one component-specific feature space to the common fusion width.

It is independently parameterized for each active semantic component. Static features, memory, hazard memory, hazard context, and node-type embeddings therefore do not share their input projection weights.

Schema version:

```python
COMPONENT_PROJECTION_SCHEMA_VERSION = "0.1"
```

---

## 7.2 Controlled vocabularies

```python
class ComponentProjectionActivation(StrEnum):
    GELU = "gelu"

class ComponentProjectionNormalization(StrEnum):
    NONE = "none"
    LAYER_NORM = "layer_norm"
```

Only GELU is implemented. Normalization may be `layer_norm` or `none`. Unknown enum values fail during normalization; canonical-but-unimplemented branches raise `NotImplementedError` if future enum members are introduced without implementation.

---

## 7.3 Constructor

```python
ComponentProjection(
    *,
    input_dim: int,
    output_dim: int,
    component_name: str,
    activation: ComponentProjectionActivation | str = "gelu",
    normalization: ComponentProjectionNormalization | str = "layer_norm",
    dropout: float = 0.0,
)
```

### Configuration rules

- `input_dim` and `output_dim` must be positive integers; Boolean values are rejected.
- `component_name` must be nonempty.
- `dropout` must be finite and satisfy `0 <= dropout < 1`.
- `activation` must normalize to an implemented activation.
- `normalization` must normalize to an implemented strategy.

### Constructed network

```text
Linear(input_dim, output_dim)
GELU
LayerNorm(output_dim) or Identity
Dropout(dropout)
```

The module exposes its layers as stable named attributes:

```python
projection.linear
projection.activation_layer
projection.normalization_layer
projection.dropout_layer
```

These names are relevant to state dictionaries and checkpoint migration.

---

## 7.4 Baseline constructor

```python
projection = ComponentProjection.baseline(
    input_dim=24,
    output_dim=64,
    component_name="static_state",
    dropout=0.1,
    layer_norm=True,
)
```

`baseline()` constructs the exact component projection used by `ConcatProjectionFusion`: GELU plus optional layer normalization.

---

## 7.5 Forward contract

```python
projected = projection(values)
```

Input requirements:

- `values` must be a tensor;
- shape must be `[items, input_dim]`;
- dtype must be floating-point;
- device must equal `projection.device`;
- all values must be finite.

The input floating dtype does not have to equal the module parameter dtype. After validation, `values` is cast to `projection.dtype` before the linear layer. Device movement is never implicit.

Output guarantee:

```text
shape  = [items, output_dim]
dtype  = projection.dtype
device = projection.device
values = finite
```

Dropout follows normal PyTorch behavior: stochastic in training mode and disabled in evaluation mode.

---

## 7.6 Identity and diagnostics

### `architecture_dict()`

Returns a JSON-compatible description containing:

- schema version;
- component name;
- input and output widths;
- activation;
- normalization;
- dropout;
- explicit operation order.

### `architecture_fingerprint()`

Returns a deterministic SHA-256 hash of `architecture_dict()`. It changes when architecture-relevant settings change, but not when learned parameter values change.

### `parameter_fingerprint()`

Returns a deterministic hash of the exact current `state_dict()` tensor values, including names, dtypes, shapes, and bytes.

### `assert_finite_parameters()`

Scans all floating tensors in the state dictionary and raises `ValueError` if any parameter or buffer contains NaN or infinity.

### `device`, `dtype`, and `extra_repr()`

- `device` and `dtype` are derived from `linear.weight`.
- `extra_repr()` makes the semantic name and architecture visible in printed module trees.

---

# 8. `concat_projection.py`

## 8.1 Purpose

`ConcatProjectionFusion` is the mathematical implementation of the currently supported baseline. It operates on an already extracted ordered mapping of tensors.

It does not know about `NodeAlignment`, `LagMemoryEncoding`, `HazardQueryEncoding`, configuration dependencies, or upstream lineage. Those responsibilities belong to `schemas.py` and `node_state_fusion.py`.

Schema versions:

```python
CONCAT_PROJECTION_FUSION_SCHEMA_VERSION = "0.1"
CONCAT_PROJECTION_OUTPUT_SCHEMA_VERSION = "0.1"
```

---

## 8.2 `canonical_component_order()`

```python
order = canonical_component_order(
    ["hazard_context", "static_state"]
)
assert order == ("static_state", "hazard_context")
```

The helper:

- requires nonempty unique names;
- rejects unknown or experimental names;
- returns the selected known names in canonical order.

Experimental components are allowed only when an explicit `component_order` is supplied directly to `ConcatProjectionFusion`. This prevents architecture identity from depending on incidental dictionary order.

---

## 8.3 `ConcatProjectionFusion` constructor

```python
ConcatProjectionFusion(
    *,
    component_input_dims: Mapping[str, int],
    output_dim: int,
    component_order: Sequence[str] | None = None,
    dropout: float = 0.0,
    layer_norm: bool = True,
    retain_concatenated_state: bool = False,
    record_input_fingerprint: bool = False,
)
```

### `component_input_dims`

Maps each stable semantic component name to its incoming width.

```python
{
    "static_state": 24,
    "memory_state": 64,
    "hazard_context": 32,
}
```

The mapping must be nonempty. Names must be nonempty and widths must be positive integers.

### `component_order`

When omitted, every component name must be canonical and the helper resolves canonical ordering.

When supplied:

- names must be unique and nonempty;
- the order and dimension mapping must contain exactly the same names;
- custom or experimental names are permitted;
- that explicit order becomes part of architecture identity.

### Other settings

- `output_dim` is both the per-component projected width and the final fused width.
- `dropout` is used in every component projector and in the final fusion network.
- `layer_norm` controls all component normalizers and the final normalizer.
- `retain_concatenated_state` retains the `[items, K * output_dim]` intermediate in the output.
- `record_input_fingerprint` computes an exact input-value fingerprint on every forward pass.

Both diagnostic flags default to `False` because they can increase memory use, CPU transfer, synchronization, and runtime overhead.

### Constructed modules

`component_projections` is an ordered `nn.ModuleDict` containing one `ComponentProjection` per component.

`fusion_network` is a named `nn.Sequential`:

```text
linear_in:      Linear(K * output_dim, output_dim)
activation:     GELU
 dropout:       Dropout(dropout)
linear_out:     Linear(output_dim, output_dim)
normalization:  LayerNorm(output_dim) or Identity
```

Stable names make architecture inspection and checkpoint diagnostics straightforward.

---

## 8.4 Strict input mapping contract

The forward method accepts:

```python
Mapping[str, torch.Tensor]
```

The mapping must satisfy all of the following:

1. Keys are exactly the configured component names.
2. Mapping iteration order is exactly `component_order`.
3. Every value is a two-dimensional floating tensor.
4. Every tensor width matches `component_input_dims[name]`.
5. Every tensor has the same row count.
6. Every tensor is on the same device as the module.
7. Every tensor is finite.

A normal dictionary is insertion-ordered in supported Python versions, but callers should construct it deliberately. `OrderedDict` is useful when the order needs to be visually explicit.

Correct:

```python
from collections import OrderedDict

components = OrderedDict([
    ("static_state", static_tensor),
    ("memory_state", memory_tensor),
])
```

Incorrect even when names and shapes match:

```python
components = {
    "memory_state": memory_tensor,
    "static_state": static_tensor,
}
# Raises because mapping order differs from component_order.
```

The algorithm never silently reorders the caller's input.

---

## 8.5 Forward result: `ConcatProjectionFusionOutput`

```python
algorithm_output = fusion(components)
```

Fields:

| Field | Meaning |
|---|---|
| `fused_state` | Final `[items, output_dim]` representation. |
| `projected_components` | Immutable ordered mapping of each `[items, output_dim]` projected component. |
| `component_order` | Exact order used for projection and concatenation. |
| `architecture_fingerprint` | Fingerprint of this low-level algorithm architecture. |
| `concatenated_state` | Optional `[items, K * output_dim]` intermediate. |
| `input_value_fingerprint` | Optional exact fingerprint of all raw input components. |
| `schema_version` | Low-level output schema version. |

The output validates order, shape, device, floating dtype, finiteness, and nonempty fingerprints. `projected_components` is exposed through an immutable mapping proxy.

### Diagnostic cost

`record_input_fingerprint=True` detaches every input, transfers it to CPU, makes it contiguous, and hashes its raw bytes. This is appropriate for audit runs, reproducibility checks, and artifact verification, but normally unsuitable for every training step.

`retain_concatenated_state=True` retains a potentially large autograd-connected activation. Use it only when the intermediate is required.

---

## 8.6 Properties and diagnostic methods

| Member | Meaning |
|---|---|
| `component_count` | Number of active components. |
| `fusion_input_dim` | `component_count * output_dim`. |
| `device` | Device of `fusion_network.linear_in.weight`. |
| `dtype` | Dtype of `fusion_network.linear_in.weight`. |
| `architecture_dict()` | Complete algorithm architecture description. |
| `architecture_fingerprint()` | Deterministic architecture hash. |
| `parameter_fingerprint()` | Exact hash of all algorithm parameters and buffers. |
| `component_architecture_fingerprints()` | Immutable mapping of per-projector architecture hashes. |
| `component_parameter_fingerprints()` | Immutable mapping of per-projector parameter hashes. |
| `assert_finite_parameters()` | Validates the complete algorithm and every projector. |

The architecture description includes the diagnostic flags because they change the forward-output contract and execution behavior.

---

# 9. `node_state_fusion.py`

## 9.1 Purpose

`NodeStateFusion` is the stable public entry point. It joins typed schemas, model configuration, optional node-type lookup, and the selected numerical fusion algorithm.

It owns:

- canonical mode recognition;
- distinction between known and implemented modes;
- validation of enabled and disabled components;
- dimension resolution;
- construction from `ModelConfig`;
- component extraction from typed upstream objects;
- optional node-type embedding;
- low-level algorithm invocation;
- final metadata-preserving output assembly;
- architecture, parameter, and lineage identity;
- migration of checkpoints from the earlier monolithic implementation.

Schema versions:

```python
NODE_STATE_FUSION_ENCODER_SCHEMA_VERSION = "0.2"
NODE_STATE_FUSION_LEGACY_STATE_DICT_SCHEMA_VERSION = "0.1"
```

---

## 9.2 Fusion modes

```python
class NodeStateFusionMode(StrEnum):
    CONCAT_PROJECTION = "concat_projection"
    PROJECTED_SUM = "projected_sum"
    GATED_FUSION = "gated_fusion"
    FILM_CONDITIONING = "film_conditioning"
```

All four values are canonical configuration vocabulary. Only `concat_projection` is implemented.

Behavior:

- unknown string: `ValueError`;
- known but unimplemented mode: `NotImplementedError`;
- implemented mode: construction proceeds.

This distinction prevents typographical errors from being confused with planned implementation gaps.

---

## 9.3 Explicit constructor

```python
NodeStateFusion(
    *,
    mode: NodeStateFusionMode | str,
    output_dim: int,

    include_static_state: bool,
    static_input_dim: int | None,

    include_memory_state: bool,
    memory_input_dim: int | None,

    include_hazard_memory_state: bool,
    hazard_memory_input_dim: int | None,

    include_hazard_context: bool,
    hazard_context_input_dim: int | None,

    include_node_type_embedding: bool,
    node_type_count: int | None,
    node_type_embedding_dim: int = 16,

    dropout: float = 0.0,
    layer_norm: bool = True,
)
```

### Component inclusion policy

At least one component must be enabled.

For each dense component:

- when enabled, its input dimension is required and must be positive;
- when disabled, its input dimension must be `None`.

Examples:

```python
# Valid
include_static_state=True
static_input_dim=24

# Invalid: enabled without width
include_static_state=True
static_input_dim=None

# Invalid: disabled but width still supplied
include_static_state=False
static_input_dim=24
```

This symmetric rule prevents stale dimensions from remaining in configurations after a component is disabled.

### Node-type embedding policy

When node-type embedding is enabled:

- `node_type_count` is required and positive;
- `node_type_embedding_dim` is positive;
- the module creates `nn.Embedding(node_type_count, node_type_embedding_dim)`.

When it is disabled:

- `node_type_count` must be `None`;
- `node_type_embedding_dim` must still be a valid positive integer because it remains part of the explicit constructor contract, though no embedding module is created.

### Low-level algorithm configuration

The orchestrator builds `ConcatProjectionFusion` with:

```python
retain_concatenated_state=False
record_input_fingerprint=False
```

Therefore high-level `NodeStateFusionOutput` does not expose the concatenated intermediate or raw input-value fingerprint. Use `ConcatProjectionFusion` directly for those diagnostics.

---

## 9.4 `from_config()`

```python
fusion = NodeStateFusion.from_config(model_config)
```

This is the preferred construction path.

The method requires the complete validated `ModelConfig` because fusion input dimensions live across multiple sub-configurations.

### Configuration fields consumed

The method reads and validates at least the following fields:

```text
config.static_input_dim
config.node_type_count

config.memory.enabled
config.memory.hidden_dim

config.hazard.enabled
config.hazard.output_dim

config.fusion.mode
config.fusion.output_dim
config.fusion.include_static_state
config.fusion.include_memory_state
config.fusion.include_hazard_memory_state
config.fusion.include_hazard_context
config.fusion.include_node_type_embedding
config.fusion.node_type_embedding_dim
config.fusion.dropout
config.fusion.layer_norm
```

### Cross-module consistency checks

`from_config()`:

1. requires an actual `ModelConfig`;
2. calls `config.validate()`;
3. verifies that fusion mode order and named constants in `config.py` exactly match the local `NodeStateFusionMode` vocabulary;
4. rejects recognized but unimplemented modes;
5. requires resolved static width when static state is enabled;
6. requires memory to be enabled when memory state is included;
7. requires both memory and hazard conditioning when hazard-memory state is included;
8. requires hazard conditioning when hazard context is included;
9. requires resolved node-type count when node-type embedding is included.

The vocabulary-drift checks intentionally raise `RuntimeError`, because disagreement between `config.py` and the fusion package is an internal integration defect rather than invalid user data.

---

## 9.5 Enabled/disabled input enforcement

`NodeStateFusion` does not merely require enabled components. It also rejects data for disabled components.

For every field:

```text
enabled + missing  -> ValueError
disabled + present -> ValueError
```

This ensures the runtime inputs exactly match the architecture that was fingerprinted and checkpointed.

---

## 9.6 Component extraction

```python
components = fusion.extract_component_tensors(inputs)
```

This method:

1. requires `NodeStateFusionInputs`;
2. enforces enabled/disabled field policy;
3. checks that the input device matches the module device;
4. iterates through `self.component_order`;
5. extracts each raw tensor from its typed source object;
6. performs node-type embedding lookup when configured;
7. validates device, floating dtype, row count, and finiteness;
8. returns an immutable ordered mapping.

No projection or final fusion occurs in this method. It is useful for debugging the exact algorithm inputs.

### Node-type ID range validation

The schema verifies only that `node_type_ids` is a one-dimensional `torch.long` tensor with the correct row count and device. `NodeStateFusion` performs architecture-dependent range validation immediately before lookup:

```text
0 <= node_type_id < node_type_count
```

Out-of-range IDs raise `IndexError`. Empty ID tensors are permitted when the overall non-graph alignment has zero items.

---

## 9.7 Forward pass

```python
output = fusion(inputs)
```

The method performs:

```text
NodeStateFusionInputs
    -> extract_component_tensors()
    -> ConcatProjectionFusion.forward()
    -> verify returned component order
    -> NodeStateFusionOutput
```

The output includes:

- the fused state;
- the complete original typed inputs;
- every projected component;
- normalized fusion mode;
- architecture fingerprint;
- combined lineage fingerprint.

The public method rejects bare tensors, lists, tuples, and dictionaries with `TypeError`.

---

## 9.8 Public properties and compatibility views

| Property | Meaning |
|---|---|
| `component_input_dims` | Immutable low-level width mapping. |
| `component_projections` | Compatibility view of the low-level `ModuleDict`. |
| `fusion_network` | Compatibility view of the low-level final fusion network. |
| `device` | Low-level algorithm device. |
| `dtype` | Low-level algorithm dtype. |
| `component_order` | Exact active semantic order. |
| `component_count` | Number of active components. |

The compatibility properties preserve access patterns from the earlier monolithic implementation while parameters are now registered under `fusion_algorithm`.

---

## 9.9 Architecture, parameter, and lineage identity

### Architecture identity

`architecture_dict()` records:

- encoder schema version;
- selected mode;
- all canonical and implemented modes;
- output width;
- each inclusion flag and corresponding input width;
- node-type vocabulary and embedding width;
- component order;
- dropout and normalization settings;
- complete nested `ConcatProjectionFusion` architecture.

`architecture_fingerprint()` hashes this dictionary.

### Parameter identity

`parameter_fingerprint()` hashes every tensor in the complete `NodeStateFusion.state_dict()`, including:

- optional node-type embedding;
- all component projections;
- final fusion network;
- normalization parameters.

### Lineage identity

```python
lineage = fusion.lineage_fingerprint(inputs)
```

The lineage fingerprint combines:

- fusion architecture fingerprint;
- input lineage fingerprint;
- alignment fingerprint.

The alignment is represented both within input lineage and explicitly in the fusion lineage payload. This makes the final identity visibly dependent on the exact row/graph alignment.

### Per-component diagnostics

```python
fusion.component_architecture_fingerprints()
fusion.component_parameter_fingerprints()
fusion.assert_finite_parameters()
```

These methods delegate to the low-level algorithm and are useful for checkpoint comparison, experiment manifests, and fault isolation.

---

## 9.10 Legacy checkpoint migration

The earlier monolithic implementation registered component projectors and the final fusion network directly under `NodeStateFusion`. The split implementation nests them under `fusion_algorithm` and uses semantic layer names.

Use:

```python
result = fusion.load_legacy_state_dict(legacy_state_dict, strict=True)
```

or migrate without loading:

```python
upgraded = fusion.upgrade_legacy_state_dict(legacy_state_dict)
```

### Key transformations

| Legacy key pattern | Current key pattern |
|---|---|
| `component_projections.<name>.network.0.<suffix>` | `fusion_algorithm.component_projections.<name>.linear.<suffix>` |
| `component_projections.<name>.network.2.<suffix>` | `fusion_algorithm.component_projections.<name>.normalization_layer.<suffix>` |
| `fusion_network.0.<suffix>` | `fusion_algorithm.fusion_network.linear_in.<suffix>` |
| `fusion_network.3.<suffix>` | `fusion_algorithm.fusion_network.linear_out.<suffix>` |
| `fusion_network.4.<suffix>` | `fusion_algorithm.fusion_network.normalization.<suffix>` |

Keys already beginning with `fusion_algorithm.` are retained. `node_type_embedding.*` keys are retained. Unknown keys are also retained so PyTorch's normal strict-loading diagnostics can report them.

Tensor values are not copied, reshaped, or transformed. The migration changes keys only. A collision after migration raises `ValueError`.

---

# 10. Usage examples

## 10.1 Minimal static-only fusion

This example uses only contracts defined in the fusion package.

```python
import torch

from urban_resilience_models.v2_hazard_conditioned_functional_ugnn.fusion.node_state_fusion import (
    NodeStateFusion,
    NodeStateFusionMode,
)
from urban_resilience_models.v2_hazard_conditioned_functional_ugnn.fusion.schemas import (
    NodeAlignment,
    NodeStateComponent,
    NodeStateFusionInputs,
)

N = 4
STATIC_DIM = 12
FUSION_DIM = 32

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

alignment = NodeAlignment(
    item_count=N,
    item_ids=("node-0", "node-1", "node-2", "node-3"),
)

static_state = NodeStateComponent(
    values=torch.randn(N, STATIC_DIM),
    component_name="static_state",
    source_fingerprint="static-features-v1",
    alignment_fingerprint=alignment.fingerprint(),
)

inputs = NodeStateFusionInputs(
    alignment=alignment,
    static_state=static_state,
    source_fingerprint="fusion-input-example-v1",
).to(device)

fusion = NodeStateFusion(
    mode=NodeStateFusionMode.CONCAT_PROJECTION,
    output_dim=FUSION_DIM,
    include_static_state=True,
    static_input_dim=STATIC_DIM,
    include_memory_state=False,
    memory_input_dim=None,
    include_hazard_memory_state=False,
    hazard_memory_input_dim=None,
    include_hazard_context=False,
    hazard_context_input_dim=None,
    include_node_type_embedding=False,
    node_type_count=None,
    dropout=0.1,
    layer_norm=True,
).to(device)

fusion.train()
output = fusion(inputs)

assert output.fused_state.shape == (N, FUSION_DIM)
assert tuple(output.projected_components) == ("static_state",)
assert output.projected_components["static_state"].shape == (N, FUSION_DIM)
```

For deterministic inference behavior, call `fusion.eval()` and use `torch.no_grad()` as usual.

---

## 10.2 Full typed integration

The memory and hazard objects must be produced by their owning upstream modules.

```python
# Produced upstream:
# memory_encoding: LagMemoryEncoding
# hazard_memory_values: torch.Tensor [N, MEMORY_DIM]
# hazard_query_encoding: HazardQueryEncoding

alignment_fp = alignment.fingerprint()

hazard_memory_state = NodeStateComponent(
    values=hazard_memory_values,
    component_name="hazard_memory_state",
    source_fingerprint="hazard-queried-memory-v2",
    alignment_fingerprint=alignment_fp,
)

inputs = NodeStateFusionInputs(
    alignment=alignment,
    static_state=static_state,
    memory_state=memory_encoding,
    hazard_memory_state=hazard_memory_state,
    hazard_context=hazard_query_encoding,
    node_type_ids=torch.tensor([0, 0, 1, 1], dtype=torch.long, device=device),
    source_fingerprint="full-node-state-input-v1",
)

fusion = NodeStateFusion(
    mode="concat_projection",
    output_dim=64,
    include_static_state=True,
    static_input_dim=static_state.feature_dim,
    include_memory_state=True,
    memory_input_dim=memory_encoding.memory_state.shape[1],
    include_hazard_memory_state=True,
    hazard_memory_input_dim=hazard_memory_state.feature_dim,
    include_hazard_context=True,
    hazard_context_input_dim=hazard_query_encoding.query.shape[1],
    include_node_type_embedding=True,
    node_type_count=2,
    node_type_embedding_dim=16,
    dropout=0.1,
    layer_norm=True,
).to(device)

output = fusion(inputs)

assert tuple(output.projected_components) == (
    "static_state",
    "memory_state",
    "hazard_memory_state",
    "hazard_context",
    "node_type_embedding",
)
```

All upstream objects and the fusion module must already share one device.

---

## 10.3 Construction from `ModelConfig`

```python
fusion = NodeStateFusion.from_config(model_config)
fusion = fusion.to(device)
output = fusion(inputs)
```

This path should be preferred in model assembly because it centralizes dimensions and inclusion rules in the package configuration.

---

## 10.4 Direct low-level algorithm use

Use this path for isolated fusion experiments, custom components, or diagnostics that the high-level wrapper disables.

```python
from collections import OrderedDict
import torch

from urban_resilience_models.v2_hazard_conditioned_functional_ugnn.fusion.concat_projection import (
    ConcatProjectionFusion,
)

algorithm = ConcatProjectionFusion(
    component_input_dims=OrderedDict([
        ("static_state", 12),
        ("memory_state", 32),
    ]),
    output_dim=64,
    component_order=("static_state", "memory_state"),
    dropout=0.0,
    layer_norm=True,
    retain_concatenated_state=True,
    record_input_fingerprint=True,
)

components = OrderedDict([
    ("static_state", torch.randn(5, 12)),
    ("memory_state", torch.randn(5, 32)),
])

result = algorithm(components)

assert result.fused_state.shape == (5, 64)
assert result.concatenated_state is not None
assert result.concatenated_state.shape == (5, 128)
assert result.input_value_fingerprint is not None
```

Move both the module and all tensors to the same device before calling it.

---

## 10.5 Isolated component projection

```python
projection = ComponentProjection.baseline(
    input_dim=20,
    output_dim=64,
    component_name="hazard_context",
    dropout=0.1,
    layer_norm=True,
)

values = torch.randn(8, 20, device=projection.device)
projected = projection(values)
assert projected.shape == (8, 64)
```

---

## 10.6 Audit and reproducibility checks

```python
fusion.assert_finite_parameters()

manifest = {
    "fusion_architecture": fusion.architecture_dict(),
    "fusion_architecture_fingerprint": fusion.architecture_fingerprint(),
    "fusion_parameter_fingerprint": fusion.parameter_fingerprint(),
    "component_architecture_fingerprints": dict(
        fusion.component_architecture_fingerprints()
    ),
    "component_parameter_fingerprints": dict(
        fusion.component_parameter_fingerprints()
    ),
    "input_lineage_fingerprint": inputs.lineage_fingerprint(),
    "output_lineage_fingerprint": output.lineage_fingerprint,
}
```

Fingerprinting exact tensor values can be expensive because it transfers detached tensors to CPU. It should generally occur at checkpoint, artifact, or audit boundaries rather than every training step.

---

# 11. Device, dtype, autograd, and mode behavior

## 11.1 Device rules

The subsystem never silently moves input tensors to the model device.

Before a forward pass:

```text
all input-bearing objects.device == fusion.device
```

Recommended pattern:

```python
fusion = fusion.to(device)
inputs = inputs.to(device)  # subject to hazard-query restriction
output = fusion(inputs)
```

When hazard context is present, produce or move that complete upstream object on the target device before calling `inputs.to(device)`.

## 11.2 Dtype rules

- Raw state components must be floating-point.
- Node-type IDs and graph-membership indices must be exactly `torch.long`.
- `ComponentProjection` casts floating input tensors to the module parameter dtype before applying its linear layer.
- Device conversion is not implicit.
- Schema `.to()` helpers move devices only and do not provide a dtype parameter.

Mixed floating input dtypes can reach the low-level algorithm, because each component projector independently casts to the module dtype. Consistent upstream dtypes are still recommended for clarity and performance.

## 11.3 Autograd

The forward path uses normal PyTorch operations and preserves autograd. Projected components and the fused state remain connected to their source tensors and module parameters unless called under `torch.no_grad()` or explicitly detached.

Fingerprint methods detach tensors and operate on CPU copies. Fingerprints do not participate in gradients.

## 11.4 Training and evaluation modes

Dropout exists in:

- every component projector;
- the final fusion network.

Use standard PyTorch mode control:

```python
fusion.train()  # stochastic dropout
fusion.eval()   # dropout disabled
```

Layer normalization behaves consistently in both modes.

---

# 12. Fingerprint taxonomy

The subsystem uses several different fingerprints. They are not interchangeable.

| Fingerprint | Answers | Depends on exact tensor values? |
|---|---|---:|
| Alignment semantic fingerprint | What alignment was declared? | No |
| Alignment value fingerprint | What graph-membership tensor was used? | Yes |
| Alignment fingerprint | What exact semantic and value alignment was used? | Partly |
| Component value fingerprint | What exact component tensor was used? | Yes |
| Component lineage fingerprint | What component, source, alignment, and values produced this state? | Yes |
| Fusion-input lineage fingerprint | What aligned collection of source components was provided? | Yes, where available |
| Component-projection architecture fingerprint | What projection design was configured? | No |
| Component-projection parameter fingerprint | What exact learned projection parameters exist now? | Yes |
| Concat-fusion architecture fingerprint | What low-level fusion algorithm and settings were configured? | No |
| Node-state-fusion architecture fingerprint | What complete high-level fusion architecture was configured? | No |
| Node-state-fusion parameter fingerprint | What exact parameters exist across the full fusion module? | Yes |
| Output lineage fingerprint | What fusion architecture and source lineage produced this output? | Indirectly through input lineage |

Fingerprints are deterministic SHA-256 hex digests. Architecture fingerprints use canonical JSON with sorted keys and compact separators. Tensor fingerprints sort tensor names and include dtype, shape, and exact bytes.

A fingerprint is an identity and audit mechanism, not a security boundary or a substitute for artifact signing.

---

# 13. Error model

The subsystem uses exception types consistently:

| Exception | Typical cause |
|---|---|
| `TypeError` | Wrong object type, non-Boolean flag, nonnumeric dropout, non-tensor state, non-string state-dict key |
| `ValueError` | Invalid shape, width, order, device, finiteness, inclusion policy, alignment, fingerprint string, or probability range |
| `IndexError` | Node-type ID outside the configured embedding range |
| `NotImplementedError` | Canonical fusion mode or strategy exists but is not implemented |
| `RuntimeError` | Internal invariant failure, configuration vocabulary drift, impossible returned shape/order, unavailable internal component |

These errors are intended to fail close to the boundary where the invalid state first becomes observable.

---

# 14. Common integration mistakes

## 14.1 Passing raw tensors to `NodeStateFusion`

Incorrect:

```python
fusion(torch.randn(10, 32))
```

Correct:

```python
fusion(NodeStateFusionInputs(...))
```

## 14.2 Supplying a disabled component

If `include_hazard_context=False`, `inputs.hazard_context` must be `None`. Extra information is not ignored.

## 14.3 Omitting the dimension of an enabled component

Every enabled dense component requires its source width at module construction.

## 14.4 Building components for different row orders

Matching row counts are necessary but not always sufficient. Use stable `item_ids`, `source_fingerprint`, and `alignment_fingerprint` so mismatched origins can be detected.

## 14.5 Reordering a low-level mapping

`ConcatProjectionFusion` compares mapping iteration order with `component_order`. Rebuild the mapping explicitly rather than relying on set operations or arbitrary collection conversion.

## 14.6 Moving only the query tensor

A `HazardQueryEncoding` retains a complete source embedding object. Move or regenerate the metadata-preserving hazard result upstream; do not fabricate a partially moved query contract.

## 14.7 Using non-contiguous packed graph IDs

For `graph_count=3`, IDs must include exactly `0`, `1`, and `2`. IDs such as `0`, `2`, and `3` are invalid even if all are nonnegative.

## 14.8 Using `int32` IDs

Both `node_type_ids` and `node_batch_index` must use exactly `torch.long`.

## 14.9 Expecting high-level concatenated-state diagnostics

`NodeStateFusion` constructs the low-level algorithm with concatenated-state retention disabled. Use `ConcatProjectionFusion` directly when that intermediate is required.

## 14.10 Treating frozen dataclasses as deeply immutable

The dataclass fields cannot be reassigned, and mappings are exposed through immutable proxies, but PyTorch tensors can still be mutated in place. In-place changes can invalidate previously computed lineage assumptions.

---

# 15. Current limitations and deliberate non-goals

The present implementation intentionally has a narrow, controlled scope:

1. Only `concat_projection` is implemented.
2. `projected_sum`, `gated_fusion`, and `film_conditioning` are recognized but unavailable.
3. Hazard context is concatenated as a component; it does not gate other components.
4. There are no node-type experts.
5. There is no component attribution mechanism beyond access to projected intermediates.
6. There is no uncertainty propagation.
7. There is no learned missing-component substitution.
8. There is no implicit broadcasting or graph-level-to-node-level expansion inside fusion.
9. The typed memory field currently requires `LagMemoryEncoding`; a generic recurrent or transformer memory result would require a compatible schema change.
10. `HazardQueryEncoding` lacks a public general-purpose device reconstruction path, so movement remains an upstream responsibility.
11. High-level fusion disables raw input-value fingerprinting and concatenated-state retention for performance and memory discipline.

These constraints make the baseline reproducible and scientifically interpretable. New behavior should be added through explicit contracts and ablations rather than silently folded into the baseline.

---

# 16. Extension guide

## 16.1 Adding a new fusion component

A new canonical component should require coordinated changes:

1. Define a stable semantic constant.
2. Decide its position in `CANONICAL_FUSION_COMPONENT_ORDER`.
3. Add a typed field to `NodeStateFusionInputs`, preferably retaining a dedicated upstream result contract.
4. Validate type, row count, device, graph alignment, and lineage in `schemas.py`.
5. Add an inclusion flag and dimension rule to configuration and `NodeStateFusion`.
6. Add extraction logic in `extract_component_tensors()`.
7. Include the component in architecture fingerprints.
8. Define checkpoint compatibility expectations.
9. Add shape, ordering, device, finiteness, and disabled-component tests.
10. Update public documentation and ablation definitions.

Do not append unknown names automatically to canonical order. Experimental low-level components should use an explicit `component_order` until their semantics become stable.

## 16.2 Adding a fusion mode

A new fusion mode should have its own dedicated mathematical module rather than branching deeply inside `ConcatProjectionFusion`.

Required work:

1. Keep the mode synchronized with `config.py` canonical constants.
2. Add the mode to `IMPLEMENTED_NODE_STATE_FUSION_MODES` only after implementation and tests exist.
3. Create a dedicated algorithm module and typed output if its diagnostics differ.
4. Add dispatch in `NodeStateFusion`.
5. Preserve enabled-component policy and typed input boundary.
6. Define architecture and parameter fingerprints.
7. Add mode-specific ablations and checkpoint behavior.
8. Ensure unknown modes still raise `ValueError` and known unsupported modes still raise `NotImplementedError`.

## 16.3 Adding an activation or normalization strategy

Update the controlled enum and implementation together. Never change the meaning of an existing enum value. Ensure the selected strategy appears in `architecture_dict()` so architecture fingerprints change correctly.

## 16.4 Generalizing memory contracts

`NodeStateFusionInputs.memory_state` currently accepts `LagMemoryEncoding`. Supporting recurrent or transformer encodings should preserve the same principles:

- complete typed upstream result retention;
- explicit output tensor access;
- item-count contract;
- device contract;
- source and architecture lineage;
- safe reconstruction or `.to()` behavior.

A protocol or shared memory-result schema may be preferable to a union that grows without a common contract.

---

# 17. Testing checklist

At minimum, tests should cover the following.

## Schemas

- `NodeAlignment` with and without graph membership.
- Rejection of negative, missing, out-of-range, or non-contiguous graph IDs.
- Duplicate or mis-sized `item_ids`.
- `NodeStateComponent` rank, dtype, width, and finiteness checks.
- Component alignment-fingerprint mismatch.
- Row-count mismatch across every input field.
- Cross-device input rejection.
- Node-aligned hazard-context membership equality.
- Input and output lineage determinism.
- Device reconstruction behavior, including hazard-query restriction.

## Component projection

- Valid shape transformation for several item counts and widths.
- Input-width mismatch.
- Non-floating input rejection.
- device mismatch.
- NaN and infinity rejection.
- layer-normalized and identity-normalized variants.
- train/eval dropout behavior.
- architecture fingerprint stability.
- parameter fingerprint change after parameter mutation.

## Concat projection

- Canonical subset ordering.
- Unknown-name rejection without explicit order.
- Experimental-name support with explicit order.
- Missing, unexpected, and reordered input mappings.
- Per-component width and row-count mismatches.
- Projected, concatenated, and fused output shapes.
- optional concatenated-state retention.
- optional input-value fingerprinting.
- finite-parameter checks.

## Orchestrator

- Every valid enabled-component subset.
- Enabled-but-missing and disabled-but-present fields.
- constructor dimension symmetry.
- `from_config()` dependency checks.
- config vocabulary drift detection.
- node-type embedding range checks.
- exact extraction order.
- output source-input preservation.
- architecture and lineage fingerprints.
- legacy key conversion and strict loading.
- canonical-but-unimplemented mode errors.

---

# 18. Operational guidance

## Training

- Construct fusion once from the frozen experiment configuration.
- Move model and upstream encoders to the same device.
- Build `NodeAlignment` from the final node ordering after batching or packing.
- Attach alignment fingerprints to generic components.
- Produce memory and hazard results using their typed upstream modules.
- Build `NodeStateFusionInputs` only after all sources are aligned.
- Call `NodeStateFusion` and pass `output.fused_state` to the next model stage.
- Store architecture and parameter fingerprints with checkpoints or manifests.

## Inference

- Reconstruct the same architecture and component policy used at training time.
- Load the checkpoint strictly unless a documented migration is being performed.
- Use `eval()` and `torch.no_grad()`.
- Preserve item IDs and source fingerprints so predictions and explanations can be joined back to source entities.
- Retain `NodeStateFusionOutput.source_inputs` when downstream explanations need upstream metadata.

## Debugging

A useful diagnostic sequence is:

```python
print(fusion)
print(fusion.architecture_dict())
print(fusion.architecture_fingerprint())
print(inputs.component_lineage_dict())

components = fusion.extract_component_tensors(inputs)
for name, tensor in components.items():
    print(name, tuple(tensor.shape), tensor.dtype, tensor.device)

fusion.assert_finite_parameters()
output = fusion(inputs)

for name, tensor in output.projected_components.items():
    print(name, tuple(tensor.shape))
print("fused", tuple(output.fused_state.shape))
```

Use exact input-value fingerprinting only through the low-level algorithm when an audit requires it.

---

# 19. Public symbols by file

## `component_projection.py`

```text
COMPONENT_PROJECTION_SCHEMA_VERSION
ComponentProjection
ComponentProjectionActivation
ComponentProjectionNormalization
```

## `concat_projection.py`

```text
CANONICAL_FUSION_COMPONENT_ORDER
CONCAT_PROJECTION_FUSION_SCHEMA_VERSION
CONCAT_PROJECTION_OUTPUT_SCHEMA_VERSION
ConcatProjectionFusion
ConcatProjectionFusionOutput
FUSION_COMPONENT_HAZARD_CONTEXT
FUSION_COMPONENT_HAZARD_MEMORY_STATE
FUSION_COMPONENT_MEMORY_STATE
FUSION_COMPONENT_NODE_TYPE_EMBEDDING
FUSION_COMPONENT_STATIC_STATE
canonical_component_order
```

## `node_state_fusion.py`

```text
CANONICAL_NODE_STATE_FUSION_MODES
IMPLEMENTED_NODE_STATE_FUSION_MODES
NODE_STATE_FUSION_ENCODER_SCHEMA_VERSION
NODE_STATE_FUSION_LEGACY_STATE_DICT_SCHEMA_VERSION
NodeStateFusion
NodeStateFusionInputs
NodeStateFusionMode
NodeStateFusionOutput
```

`NodeStateFusionInputs` and `NodeStateFusionOutput` are re-exported from this module for convenience, though their definitions live in `schemas.py`.

## `schemas.py`

```text
NODE_ALIGNMENT_SCHEMA_VERSION
NODE_STATE_COMPONENT_SCHEMA_VERSION
NODE_STATE_FUSION_INPUT_SCHEMA_VERSION
NODE_STATE_FUSION_OUTPUT_SCHEMA_VERSION
NodeAlignment
NodeStateComponent
NodeStateFusionInputs
NodeStateFusionOutput
```

---

# 20. Runtime assumptions

The implementation requires:

- Python 3.11 or newer, because it uses `enum.StrEnum`;
- PyTorch;
- the surrounding V2 package modules that define `ModelConfig`, `LagMemoryEncoding`, `HazardQueryEncoding`, and `NodeAlignedHazardEmbeddingLookup`.

The files do not pin a specific PyTorch version. Use the version selected by the repository's environment and lock files.

---

# 21. Summary contract

The subsystem can be summarized by the following rules:

1. **Typed boundary:** high-level fusion accepts `NodeStateFusionInputs`, not arbitrary tensors.
2. **Shared alignment:** every component row refers to the same item in the same order.
3. **Exact policy:** enabled components are mandatory; disabled components are forbidden.
4. **Deterministic order:** concatenation follows an explicit semantic order.
5. **Common width:** every active component receives an independent projection to `output_dim`.
6. **Controlled baseline:** projected states are concatenated and transformed by a fixed two-linear-layer MLP.
7. **No silent movement:** all tensors and modules must already share one device.
8. **Finite values:** input states, outputs, and parameters are checked for NaN and infinity at defined boundaries.
9. **Metadata preservation:** the final output retains complete typed source inputs and projected intermediates.
10. **Reproducible identity:** schema, architecture, parameter, value, alignment, and lineage fingerprints have distinct explicit roles.
11. **Explicit evolution:** new modes, components, and strategies require dedicated contracts and tests rather than silent changes to baseline semantics.

Within the V2 architecture, the resulting `fused_state` is the initial node representation supplied to later functional message-passing and prediction modules.
