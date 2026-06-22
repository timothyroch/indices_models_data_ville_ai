# Future Fusion Research Modules

## Status

The files documented here are intentional placeholders for future research and engineering work in the node-state fusion subsystem.

They are **not implemented**, **not part of the public runtime API**, and **must not be marked as supported capabilities** until each module has:

1. an accepted design contract;
2. a concrete implementation;
3. dedicated unit tests;
4. integration tests with `NodeStateFusion`;
5. configuration support;
6. reproducibility and lineage metadata;
7. capability-manifest updates.

The currently implemented baseline remains:

```text
concat_projection
```

Its active implementation is expected to live in:

```text
fusion/component_projection.py
fusion/concat_projection.py
fusion/node_state_fusion.py
fusion/schemas.py
```

The placeholder files listed below preserve future research directions without claiming that those capabilities currently exist.

---

## General rules for placeholder modules

Until a module is implemented:

- do not import it from `fusion/__init__.py`;
- do not reference it from runtime dispatch;
- do not add it to an implemented-capability tuple;
- do not silently fall back to `concat_projection`;
- do not instantiate placeholder classes;
- do not treat the existence of the file as evidence of implementation;
- prefer an explicit `NotImplementedError` if code is later added before the capability is complete.

A future implementation should preserve the existing fusion subsystem contracts:

- typed fusion inputs;
- metadata-preserving outputs;
- deterministic component alignment;
- explicit configuration;
- finite-value checks;
- architecture fingerprints;
- parameter fingerprints;
- lineage fingerprints;
- stable graph/node alignment metadata;
- reproducible construction;
- isolated ablations.

---

# Placeholder modules

## `component_registry.py`

### Purpose

Provide a semantic registry for every fusion component once the number and behavior of components justify more than a fixed tuple of names.

Potential components include:

```text
static_state
memory_state
hazard_memory_state
hazard_context
node_type_embedding
local_weather_context
uncertainty_state
external_event_context
```

### Possible responsibilities

- stable component identifiers;
- canonical component ordering;
- expected source type;
- expected tensor rank and alignment scope;
- semantic role;
- supported fusion modes;
- attribution support;
- uncertainty semantics;
- explanation labels;
- missing-component policy;
- configuration compatibility checks.

### Research value

A registry would allow experiments to add or remove context sources without hard-coding component behavior inside every fusion implementation.

### Activation criteria

Implement this module when at least one of the following becomes true:

- more components are added;
- multiple fusion algorithms need the same component metadata;
- component-specific attribution becomes operational;
- uncertainty-bearing components are introduced;
- component behavior depends on semantic role rather than only tensor shape.

### Expected tests

```text
tests/test_fusion_component_registry.py
```

The tests should cover identity stability, canonical ordering, duplicate rejection, source-type validation, serialization, and configuration compatibility.

---

## `diagnostics.py`

### Purpose

Provide diagnostics for understanding whether the fusion subsystem is behaving meaningfully during training and inference.

### Possible responsibilities

- input and projected-component norms;
- fused-state norms;
- per-component activation statistics;
- gradient magnitudes;
- missingness rates;
- gate saturation;
- FiLM scale and shift statistics;
- expert-utilization statistics;
- component-collapse detection;
- dead-component detection;
- train/validation distribution drift;
- numerical stability summaries.

### Research value

Diagnostics can reveal whether the model is actually using temporal memory, hazard context, or node-type information rather than technically accepting them while ignoring them.

### Activation criteria

Implement when adaptive fusion, component weighting, attribution, or expert routing produces quantities worth inspecting.

### Expected tests

```text
tests/test_fusion_diagnostics.py
```

Tests should verify deterministic summaries, finite outputs, masking behavior, device preservation, and no accidental gradient detachment unless explicitly requested.

---

## `gated_fusion.py`

### Purpose

Learn node-specific or scenario-specific weights controlling how strongly each component contributes to the fused node state.

### Candidate formulation

For projected component states:

```text
z_static
z_memory
z_hazard
z_type
```

learn gates:

```text
g_static
g_memory
g_hazard
g_type
```

and compute:

```text
fused = sum(g_i * z_i)
```

The gates may be normalized with sigmoid, softmax, sparsemax, entmax, or another explicitly configured mechanism.

### Research questions

- Should gates compete through softmax or operate independently through sigmoid?
- Should gating be global, graph-level, node-level, or node-and-hazard-level?
- Should missing components renormalize the remaining gates?
- Should gates be regularized for sparsity or entropy?
- Should relation-family information affect fusion?
- Are gate values reliable enough to expose as explanations?

### Required metadata

A future output should preserve:

- gate values;
- gate normalization mode;
- active component order;
- masked components;
- gate architecture fingerprint;
- gate lineage fingerprint.

### Activation criteria

Implement after the concat-projection baseline is stable and a concrete gated-fusion experiment is defined.

### Expected tests

```text
tests/test_gated_fusion.py
```

Tests should cover normalization, masks, deterministic ordering, gradients, empty batches, disabled components, and gate-value preservation.

---

## `film_conditioning.py`

### Purpose

Use one context source—most likely the hazard query—to modulate another node representation through feature-wise affine conditioning.

### Candidate formulation

Given a base node state `x` and conditioning state `q`:

```text
gamma, beta = conditioner(q)
conditioned = gamma * x + beta
```

Variants may use:

```text
conditioned = (1 + gamma) * x + beta
```

to initialize near the identity transformation.

### Research questions

- Should the hazard query modulate static state, memory state, or the already fused state?
- Should FiLM parameters vary by node, graph, node type, or relation family?
- Should gamma be constrained or regularized?
- Should multiple contexts produce separate FiLM stages?
- Does FiLM improve generalization to unseen hazard severities?

### Required safeguards

- explicit conditioning scope;
- strict graph-to-node broadcasting;
- finite gamma and beta values;
- optional identity-preserving initialization;
- traceable modulation outputs;
- no silent broadcasting across incompatible graphs.

### Expected tests

```text
tests/test_film_conditioning.py
```

---

## `hazard_conditioned_fusion.py`

### Purpose

Make the fusion mechanism itself depend on the hazard query rather than treating the hazard query as merely another concatenated component.

### Candidate behaviors

The hazard query may control:

- component gates;
- component projectors;
- cross-component attention;
- FiLM parameters;
- mixture-of-experts routing;
- residual strengths;
- uncertainty weighting.

### Research questions

- Does flood context increase reliance on drainage memory?
- Does heat context increase reliance on demographic and green-space features?
- Should hazard conditioning occur before or after temporal-memory fusion?
- Can one fusion mechanism generalize across hazard types?
- How should unknown or fallback hazards behave?
- Can fixed hazard priors improve stability without constraining learning too strongly?

### Required contracts

The module should consume a metadata-preserving `HazardQueryEncoding`, retain its lineage, and verify node/graph alignment exactly.

### Expected tests

```text
tests/test_hazard_conditioned_fusion.py
```

---

## `node_type_experts.py`

### Purpose

Allow different node types to use specialized fusion behavior while retaining shared structure across the graph.

Potential node types include:

```text
census tract
road segment
hospital
sewer asset
power asset
green space
critical facility
```

### Candidate designs

- one fusion expert per node type;
- shared trunk with type-specific adapters;
- low-rank expert parameters;
- soft mixture-of-experts routing;
- hard routing by canonical node type;
- hierarchical experts grouped by infrastructure family.

### Research questions

- Are hard node-type experts better than continuous node-type embeddings?
- How should rare node types share statistical strength?
- Should experts be hazard-dependent?
- How can expert collapse be detected?
- How should unknown node types be handled?
- Does specialization improve transfer across cities?

### Required diagnostics

A future implementation should expose:

- routing assignments;
- expert utilization;
- per-expert gradient statistics;
- fallback-expert usage;
- load-balancing losses, when applicable.

### Expected tests

```text
tests/test_node_type_experts.py
```

---

## `component_attribution.py`

### Purpose

Estimate how much each fusion component contributes to the final node representation or downstream prediction.

### Candidate methods

- component-removal ablations;
- leave-one-component-out deltas;
- integrated gradients;
- gradient × input;
- learned gate interpretation;
- Shapley-style approximations;
- projected-component norm comparisons;
- counterfactual replacement with baselines.

### Important limitation

Projected-state magnitude or gate value alone must not automatically be presented as causal importance.

Attribution outputs should clearly identify:

- the attribution method;
- the baseline;
- the target scalar or representation;
- whether the method is local or global;
- whether gradients were used;
- known interpretation limits.

### Research questions

- Which components drive predictions for each hazard?
- Does memory matter more during rapidly evolving events?
- Are attribution results stable across random seeds?
- Do attribution rankings agree across methods?
- Can attributions detect spurious dependence on reporting intensity?

### Expected tests

```text
tests/test_component_attribution.py
```

---

## `uncertainty_fusion.py`

### Purpose

Represent and propagate uncertainty associated with different fusion components.

Potential sources include:

```text
weather-forecast uncertainty
missing-history uncertainty
hazard-identity uncertainty
measurement uncertainty
model uncertainty
data-quality uncertainty
```

### Candidate designs

- mean-and-variance component representations;
- precision-weighted fusion;
- heteroscedastic component gates;
- ensemble-aware fusion;
- probabilistic latent fusion;
- distributional embeddings;
- confidence-aware attention;
- uncertainty-conditioned residuals.

### Research questions

- Should less certain weather forecasts contribute less strongly?
- How should correlated uncertainties be handled?
- Can missing-history uncertainty be separated from low-risk history?
- Should the fusion output be a distribution or a deterministic state plus uncertainty metadata?
- How should uncertainty propagate into relation gates, edge attention, and prediction heads?

### Required contracts

Future uncertainty-bearing components should declare:

- uncertainty type;
- parameterization;
- units or scale;
- calibration status;
- alignment scope;
- independence assumptions;
- source fingerprint.

### Expected tests

```text
tests/test_uncertainty_fusion.py
```

---

# Recommended implementation order

These files should not all be implemented at once. A reasonable research sequence is:

```text
1. diagnostics.py
2. gated_fusion.py
3. component_attribution.py
4. film_conditioning.py
5. hazard_conditioned_fusion.py
6. component_registry.py
7. node_type_experts.py
8. uncertainty_fusion.py
```

This order is only a starting point. A specific research hypothesis should determine the actual sequence.

For example:

- implement `gated_fusion.py` first to test adaptive static-versus-memory weighting;
- implement `film_conditioning.py` first to test hazard modulation;
- implement `uncertainty_fusion.py` first if probabilistic weather forecasts become available.

---

# Capability-manifest policy

The existence of a placeholder file must never make a capability appear implemented.

A capability should be marked implemented only after:

```text
implementation exists
+ dedicated tests pass
+ integration tests pass
+ config supports it
+ serialization includes it
+ experiment hashes include it
+ public API exports it intentionally
```

Canonical-but-unimplemented modes should produce:

```text
NotImplementedError
```

Unknown modes should produce:

```text
ValueError
```

There must be no silent fallback to `concat_projection`.

---

# Public API policy

Until implemented, these modules must not be imported by:

```text
fusion/__init__.py
model.py
training code
inference code
capability manifests
```

The stable public API should expose only completed and tested components.

---

# Research principle

The current concat-projection path is a baseline, not the endpoint of the fusion research space.

The placeholder modules preserve a deliberate path toward:

```text
adaptive fusion
hazard-aware fusion
node-type specialization
interpretable fusion
uncertainty-aware fusion
```

Their purpose is to prevent future development from treating simple concatenation as the only possible architecture while avoiding false claims that advanced methods already exist.
