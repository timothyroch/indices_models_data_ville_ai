# Relations Subsystem

**Package:** `urban_resilience_models.v2_hazard_conditioned_functional_ugnn.relations`  
**Primary semantic entry point:** `DEFAULT_RELATION_REGISTRY`  
**Primary runtime entry point:** `CompiledRelationRegistry`  
**Primary artifact-validation entry point:** `validate_relation_edge_data()`  
**Primary prior entry point:** `DEFAULT_HAZARD_RELATION_PRIOR_REGISTRY`  
**Python requirement:** Python 3.11 or newer  
**Purpose:** define the V2 functional-relation ontology, compile stable relation identities into dense runtime indices, attach scoped hazard–relation priors, and validate concrete relation-edge artifacts before training, inference, publication, or explanation.

---

## 1. Overview

The relations subsystem is the semantic and integrity layer for graph edges in the V2 hazard-conditioned functional UGNN.

It answers four different questions:

1. **What does a relation mean?**  
   `relation_types.py` defines immutable semantic contracts such as direction, endpoint types, temporal behavior, construction mode, leakage risk, explanation policy, and edge attributes.

2. **Which relations exist in V2, and which are currently usable?**  
   `relation_registry.py` declares the canonical ontology, stable relation IDs, exact endpoint pairs, current software capabilities, hierarchy, serialization, fingerprints, and dense runtime compilation.

3. **How should a hazard provisionally weight relation families before fitting?**  
   `hazard_relation_priors.py` defines scoped, auditable, confidence-aware priors and compiles them into hazard-by-relation matrices aligned with a compiled registry.

4. **Does a concrete graph artifact obey the ontology and experiment contract?**  
   `relation_validation.py` validates tensors, endpoint types, attributes, missingness, temporal validity, provenance, controls, duplicates, registry identity, and leakage-sensitive construction.

```text
constants.py + hazard_registry.py
              |
              v
      relation_types.py
      semantic primitives
              |
              v
     relation_registry.py
 canonical ontology + capabilities
              |
              +------------------------------+
              |                              |
              v                              v
 CompiledRelationRegistry        hazard_relation_priors.py
 relation_index 0..R-1           hazard × relation matrices
              |                              |
              +---------------+--------------+
                              |
                              v
                  relation_validation.py
            validates concrete packed edge data
                              |
                              v
            functional message passing / training
```

The package is deliberately strict. It rejects ambiguous identities, invalid endpoint pairs, hierarchy overlap, unsupported relations, temporal leakage, uncontrolled missing values, stale registry mappings, unreproducible controls, and provenance mismatches rather than silently adapting artifacts.

### Role

The V2 model is intended to reason through functional urban pathways rather than use an undifferentiated spatial graph. The relations subsystem makes those pathways explicit and auditable.

A relation is not merely an integer edge type. It carries:

- a stable ontology identity;
- a human-readable meaning;
- source and target node-type constraints;
- exact permitted endpoint pairs;
- directionality;
- evidence and construction semantics;
- temporal validity rules;
- leakage risk;
- required and optional edge attributes;
- hierarchy and reverse-relation metadata;
- current implementation, training, and explanation availability.

The subsystem also keeps scientific semantics separate from current software capability. A relation may exist in the ontology while remaining unavailable for message passing in the current release.

---

## 2. Files and ownership boundaries

```text
relations/
├── hazard_relation_priors.py
├── relation_registry.py
├── relation_types.py
└── relation_validation.py
```

| File | Owns | Does not own |
|---|---|---|
| `relation_types.py` | Immutable semantic primitives, relation and edge-attribute contracts, ontology-level validation, deterministic serialization | Concrete V2 registry entries, runtime indices, graph validation, priors, message passing |
| `relation_registry.py` | Canonical V2 relation ontology, stable IDs, exact endpoint pairs, current capability manifest, hierarchy traversal, dense runtime compilation, registry fingerprints | Graph construction, edge tensors, hazard priors, message passing |
| `hazard_relation_priors.py` | Scoped hazard–relation priors, confidence and evidence metadata, two-dimensional inheritance, gate initialization matrices, prior serialization and source identities | Hazard ontology, relation ontology, learned gates, substantive causal claims |
| `relation_validation.py` | Strict validation of concrete relation-edge artifacts against a compiled registry, temporal and provenance checks, structured reports | Ontology definitions, registry compilation, graph building, imputation, scaling, training |

Recommended dependency direction:

```text
relation_types.py
      ^
      |
relation_registry.py <----- hazard_relation_priors.py
      ^
      |
relation_validation.py
```

`hazard_relation_priors.py` also depends on the hazard registry because hazard names, stable IDs, hierarchy, queryability, support status, and fallback identity are owned by `hazard/hazard_registry.py`.

---

## 3. Recommended public API

Most model and experiment code should import from the registry, prior, and validator modules rather than construct low-level semantic objects repeatedly.

```python
from urban_resilience_models.v2_hazard_conditioned_functional_ugnn.relations.relation_registry import (
    DEFAULT_RELATION_REGISTRY,
    CompiledRelationRegistry,
    HierarchyCompilationPolicy,
)

from urban_resilience_models.v2_hazard_conditioned_functional_ugnn.relations.hazard_relation_priors import (
    DEFAULT_HAZARD_RELATION_PRIOR_REGISTRY,
    PriorApplicationContext,
    PriorResolutionPolicy,
    compile_default_hazard_relation_priors,
)

from urban_resilience_models.v2_hazard_conditioned_functional_ugnn.relations.relation_validation import (
    RelationEdgeData,
    RelationValidationExpectations,
    RelationValidationOptions,
    RelationValidationProfile,
    assert_valid_relation_edge_data,
    validate_relation_edge_data,
)
```

Use the low-level semantic classes when:

- defining a new ontology relation;
- defining a new edge attribute;
- reconstructing a serialized registry;
- writing unit tests for ontology invariants;
- building a custom relation or prior registry.

```python
from urban_resilience_models.v2_hazard_conditioned_functional_ugnn.relations.relation_types import (
    EdgeAttributeKind,
    EdgeAttributeSpec,
    MissingValuePolicy,
    RelationConstructionMode,
    RelationDirection,
    RelationEvidenceType,
    RelationExplanationPolicy,
    RelationLeakageRisk,
    RelationSpec,
    RelationTemporalMode,
)
```

---

## 4. Identity model: stable IDs versus dense runtime indices

The subsystem uses two intentionally different identifiers.

### 4.1 Stable ontology identity: `relation_id`

Each canonical relation owns a stable, sparse integer ID such as:

```text
spatial_adjacency -> 200
temporal_memory   -> 300
heat_exposure     -> 430
road_access       -> 610
```

Stable IDs:

- identify ontology concepts across releases and artifacts;
- are not required to be contiguous;
- are grouped by semantic domain;
- must never be reused for a different relation;
- must not be used directly as a tensor dimension or embedding index.

### 4.2 Dense runtime identity: `relation_index`

A `CompiledRelationRegistry` assigns selected relations contiguous indices:

```text
0, 1, ..., R - 1
```

These indices are used by:

- `edge_relation_index`;
- relation embeddings;
- relation gates;
- relation-specific transforms;
- attention summaries;
- hazard–relation prior matrices;
- explanation tensors.

The order is deterministic: selected entries are sorted by stable `relation_id` after hierarchy policy is applied.

### 4.3 Required artifact mapping

Any persisted artifact containing dense relation indices should preserve enough information to reconstruct:

```text
relation_index -> stable relation_id -> canonical relation name
```

The compiled registry provides this mapping and carries fingerprints of the source registry.

### 4.4 Never infer identity from position alone

A model checkpoint, graph artifact, prior matrix, or explanation payload is unsafe if it stores relation indices without the compiled-registry identity. A different selection of relations can assign the same dense index to a different semantic relation.

---

## 5. End-to-end lifecycle

A typical experiment follows this sequence:

```text
1. Inspect the canonical RelationRegistry.
2. Select relation names for the experiment.
3. Compile them into a dense CompiledRelationRegistry.
4. Build graph edges using the compiled mapping.
5. Attach edge attributes, temporal columns, and provenance.
6. Construct RelationEdgeData.
7. Validate the artifact.
8. Optionally compile hazard–relation priors against the same relation order.
9. Train or run inference.
10. Persist registry and artifact fingerprints with outputs.
```

Example:

```python
registry = DEFAULT_RELATION_REGISTRY

compiled = registry.compile_for_training(
    (
        "spatial_adjacency",
        "temporal_memory",
        "heat_exposure",
    ),
    allow_control_relations=False,
)

edge_relation_index = compiled.encode_names(
    (
        "spatial_adjacency",
        "temporal_memory",
        "heat_exposure",
    )
)
```

`encode_names()` returns dense indices in the same order as the supplied names. The exact return container is defined by `CompiledRelationRegistry`; callers should treat the values as runtime indices, not stable IDs.

---

# 6. `relation_types.py`

## 6.1 Purpose

`relation_types.py` defines the immutable semantic building blocks from which registries are assembled.

It contains:

- controlled vocabularies;
- `EdgeAttributeSpec`;
- `RelationSpec`;
- collection-level ontology validation;
- immutable indexes by stable ID and canonical name;
- deterministic serialization and reconstruction.

It does not assign dense runtime indices and does not know which relations are currently implemented.

---

## 6.2 Controlled vocabularies

### `RelationDirection`

| Value | Meaning |
|---|---|
| `directed` | Source and target roles are semantically distinct |
| `undirected` | The relation is symmetric; storage may still use reciprocal directed arcs |

### `RelationEvidenceType`

| Value | Meaning |
|---|---|
| `observed` | Supported directly by an observed source |
| `derived` | Computed from observed data or geometry |
| `hybrid` | Combines observed and derived information |
| `expert_defined` | Defined through expert knowledge |
| `learned_from_data` | Edge existence or meaning is fitted from data |
| `synthetic_control` | Artificial relation used as an experimental control |

### `RelationConstructionMode`

| Value | Meaning |
|---|---|
| `external_static` | Loaded from an externally maintained static source |
| `geometric` | Derived through spatial or network geometry |
| `training_fitted` | Fitted from training data and therefore leakage-sensitive |
| `as_of_origin` | Recomputed using only information available at the prediction origin |
| `synthetic_control` | Generated as a placebo or topology control |

Evidence type and construction mode are separate. A geometrically constructed edge may be based on observed source data; a learned relation may be rebuilt as of each prediction origin.

### `RelationLeakageRisk`

```text
none < low < moderate < high
```

The value is declarative metadata used to determine validation and provenance expectations. It is not itself a leakage test.

### `RelationTemporalMode`

| Value | Required interpretation |
|---|---|
| `static` | Edge is treated as timeless for the modeled period |
| `snapshot` | Edge is observed or constructed at a particular time |
| `interval_valid` | Edge is valid over a start/end interval |
| `lagged` | Edge links an earlier source state to a later target state |

### `RelationExplanationPolicy`

| Value | Use |
|---|---|
| `allowed` | May appear as an ordinary scientific pathway explanation |
| `diagnostic_only` | May appear only as a diagnostic or control |
| `excluded` | Must not appear as an available explanation pathway |

### `EdgeAttributeKind`

```text
float
integer
boolean
categorical
identifier
timestamp
```

### `MissingValuePolicy`

| Value | Contract |
|---|---|
| `forbidden` | Raw and model-facing values may not be missing |
| `nullable_raw_mask_required` | Raw values may be null; model values must be imputed and accompanied by a Boolean missingness mask |

### Other constants

```python
ANY_NODE_TYPE = "*"
RELATION_SPEC_SCHEMA_VERSION = "0.2"
TOPOLOGY_ONLY_NO_EDGE_NAME = "identity_no_edge"
```

`identity_no_edge` is explicitly guarded as a topology mode, not an edge relation.

Temporal field names are centralized:

```python
TEMPORAL_FIELD_EDGE_OBSERVATION_TIME = "edge_observation_time"
TEMPORAL_FIELD_EDGE_VALID_FROM = "edge_valid_from"
TEMPORAL_FIELD_EDGE_VALID_TO = "edge_valid_to"
TEMPORAL_FIELD_EDGE_LAG = "edge_lag"
```

---

## 6.3 `EdgeAttributeSpec`

### Purpose

`EdgeAttributeSpec` describes one semantically named edge attribute before model scaling.

```python
@dataclass(slots=True, frozen=True)
class EdgeAttributeSpec:
    name: str
    description: str
    kind: EdgeAttributeKind

    unit: str | None = None
    missing_value_policy: MissingValuePolicy = MissingValuePolicy.FORBIDDEN
    model_tensor_requires_finite: bool = True

    minimum: float | None = None
    maximum: float | None = None

    categorical_closed_vocabulary: bool = True
    categorical_values: tuple[str, ...] = ()
```

### Core invariants

- `name` and `description` must be non-empty strings.
- `kind` and `missing_value_policy` must be enum instances.
- Numeric bounds must be finite.
- `minimum` cannot exceed `maximum`.
- Non-numeric attributes cannot declare numeric bounds.
- Integer bounds must be mathematically integral.
- A closed categorical attribute must declare at least one category.
- Only categorical attributes may declare `categorical_values`.
- `categorical_closed_vocabulary=False` is meaningful only for categorical attributes.
- Boolean, identifier, and timestamp attributes cannot declare units.
- Raw-nullable attributes must still require finite model-facing values after imputation.

### Properties

| Property | Meaning |
|---|---|
| `numeric` | `True` for float or integer attributes |
| `raw_values_may_be_missing` | Whether raw missingness is permitted |
| `requires_missingness_mask` | Whether model data must preserve original missingness |

### Methods

```python
spec.validate()
spec.replace(**changes)
spec.to_dict()
EdgeAttributeSpec.from_dict(payload)
```

`replace()` returns a new validated instance. The dataclass is frozen.

### Example

```python
capacity = EdgeAttributeSpec(
    name="capacity_score",
    description="Nonnegative standardized service capacity.",
    kind=EdgeAttributeKind.FLOAT,
    minimum=0.0,
    missing_value_policy=(
        MissingValuePolicy.NULLABLE_RAW_MASK_REQUIRED
    ),
)
```

A model-facing column for this attribute must contain finite imputed values and a Boolean mask identifying which source values were originally missing.

---

## 6.4 `RelationSpec`

### Purpose

`RelationSpec` is the immutable semantic definition of one relation.

```python
@dataclass(slots=True, frozen=True)
class RelationSpec:
    relation_id: int
    name: str
    display_name: str
    description: str
    semantic_role: str

    source_node_types: tuple[str, ...]
    target_node_types: tuple[str, ...]

    direction: RelationDirection
    evidence_type: RelationEvidenceType
    construction_mode: RelationConstructionMode
    leakage_risk: RelationLeakageRisk
    temporal_mode: RelationTemporalMode

    parent_relation_name: str | None = None
    reverse_relation_name: str | None = None

    implementation_state: str = IMPLEMENTATION_STATE_TARGET
    is_control: bool = False
    allow_any_node_type: bool = False
    allows_self_loops: bool = False

    message_passing_allowed: bool = False
    training_allowed: bool = False
    explanation_policy: RelationExplanationPolicy = (
        RelationExplanationPolicy.EXCLUDED
    )

    required_edge_attributes: tuple[EdgeAttributeSpec, ...] = ()
    optional_edge_attributes: tuple[EdgeAttributeSpec, ...] = ()
    tags: tuple[str, ...] = ()

    registry_version: str = RELATION_REGISTRY_VERSION
    spec_schema_version: str = RELATION_SPEC_SCHEMA_VERSION
```

The registry later separates semantic data from operational capability. The capability-neutral default relation definitions use target/unavailable operational fields; `RelationCapability` supplies current release status.

### Identity and vocabulary rules

- `relation_id` must be a nonnegative integer and cannot be a Boolean.
- `name` must already exist in `constants.py`.
- `identity_no_edge` cannot be represented by `RelationSpec`.
- `semantic_role` must be canonical.
- implementation state must be canonical.
- source and target type collections must be non-empty, unique, and canonical.
- wildcard `"*"` cannot be mixed with specific node types.
- wildcard use requires `allow_any_node_type=True`.

### Hierarchy rules

- a relation cannot be its own parent;
- the parent must be a canonical relation;
- `identity_no_edge` cannot be a parent;
- collection-level validation verifies that the parent exists and that the hierarchy is acyclic.

### Reverse-relation rules

- reverse metadata is valid only for directed relations;
- a relation cannot be its own reverse;
- the reverse name must be canonical and cannot be `identity_no_edge`;
- collection-level validation requires reciprocal declarations;
- endpoint constraints must be reversed;
- shared semantic metadata must agree.

### Control rules

- `is_control` must agree with `CONTROL_RELATION_NAMES`;
- synthetic-control evidence requires `is_control=True`;
- synthetic-control construction requires `is_control=True`;
- control relations cannot use ordinary `allowed` explanation policy.

### Construction and leakage rules

- `learned_from_data` relations must use `training_fitted` or `as_of_origin`;
- `training_fitted` requires moderate or high leakage risk;
- `as_of_origin` cannot declare zero leakage risk;
- synthetic-control construction requires synthetic-control evidence.

### Temporal rules

- lagged relations must be directed;
- relations requiring as-of-time construction cannot be static;
- `required_temporal_fields` derives the fields required by the temporal mode.

### Attribute rules

- required and optional attribute names must each be unique;
- an attribute cannot be both required and optional;
- every attribute must be an `EdgeAttributeSpec`;
- each attribute spec is revalidated.

### Availability rules

- deprecated relations cannot remain enabled for training or message passing;
- training availability requires message-passing availability;
- an unavailable relation cannot be an ordinary scientific explanation;
- an unimplemented relation cannot be marked explanation-available.

### Self-loop rule

A relation may permit self-loops only if its source and target type constraints overlap or use the wildcard.

### Properties

| Property | Meaning |
|---|---|
| `is_root_relation` | No declared parent |
| `required_attribute_names` | Ordered required attribute names |
| `optional_attribute_names` | Ordered optional attribute names |
| `allowed_attribute_names` | Required plus optional names |
| `is_functional` | Semantic role is exposure, protection, access, or dependency |
| `is_directed` / `is_undirected` | Direction helpers |
| `implemented` | Current embedded implementation state |
| `deprecated` | Whether embedded state is deprecated |
| `available_for_message_passing` | Embedded availability |
| `available_for_training` | Embedded training availability |
| `available_for_explanation` | Embedded explanation availability |
| `requires_training_fit` | Construction uses `training_fitted` |
| `requires_as_of_time` | Construction uses `as_of_origin` |
| `required_temporal_fields` | Required temporal column names |

### Endpoint helpers

```python
spec.supports_source_type(node_type)
spec.supports_target_type(node_type)
spec.supports_node_pair(source_type, target_type)
```

For undirected relations, `supports_node_pair()` also accepts the reversed broad type constraints. Exact pair restrictions are applied later by `RelationEndpointContract`.

### Attribute helpers

```python
spec.attribute_spec("distance_m")
spec.requires_attribute("distance_m")
spec.permits_attribute("distance_m")
```

Unknown attribute lookup raises `KeyError`.

### Serialization

```python
payload = spec.to_dict()
restored = RelationSpec.from_dict(payload)
```

Reconstruction rejects unknown fields, normalizes lists to tuples, reconstructs enums and nested edge-attribute specs, and validates the result.

---

## 6.5 Collection-level ontology validation

```python
validated = validate_relation_spec_collection(
    specifications,
    require_current_registry_version=True,
    require_current_spec_schema_version=True,
)
```

Collection validation checks:

- every item is a `RelationSpec`;
- every spec is individually valid;
- relation IDs are unique;
- relation names are unique;
- versions are internally consistent;
- optional current-version requirements;
- parent references exist;
- hierarchy has no cycles;
- reverse relations are reciprocal;
- reverse endpoint constraints are compatible;
- reverse shared metadata is consistent.

The function returns a tuple suitable for immutable registry construction.

### Immutable indexes

```python
by_id = relation_specs_by_id(specifications)
by_name = relation_specs_by_name(specifications)
```

Both return read-only mappings and validate the collection first.

---

# 7. `relation_registry.py`

## 7.1 Purpose

`relation_registry.py` defines the concrete V2 edge-relation ontology and converts selected stable relations into dense runtime indices.

It owns:

- stable relation IDs;
- canonical registry entries;
- exact endpoint-pair contracts;
- current capability declarations;
- ontology hierarchy traversal;
- semantic and operational fingerprints;
- deterministic runtime compilation;
- serialization and source verification.

---

## 7.2 Stable ID ranges

Stable IDs are sparse and grouped semantically.

| Range | Domain |
|---:|---|
| 100–199 | Experimental controls |
| 200–299 | Spatial and administrative structure |
| 300–399 | Temporal memory |
| 400–499 | Exposure |
| 500–599 | Protection and specialized access |
| 600–699 | Access |
| 700–799 | Infrastructure dependency |
| 800–899 | Similarity |
| 900–999 | Cross-scale hierarchy |

Removed or deprecated IDs must not be reused.

---

## 7.3 Explicitly excluded concepts

```python
EDGE_REGISTRY_EXCLUDED_CANONICAL_NAMES = frozenset(
    {
        "identity_no_edge",
        "impervious_surface_exposure",
    }
)
```

- `identity_no_edge` is an experiment topology mode representing no edges.
- `impervious_surface_exposure` remains a node feature until a legitimate source node entity is introduced.

Compilation rejects these names as non-edge concepts.

---

## 7.4 Canonical edge attributes

| Name | Kind | Unit | Bounds / vocabulary | Missingness |
|---|---|---:|---|---|
| `distance_m` | float | m | `>= 0` | forbidden |
| `shared_boundary_length_m` | float | m | `>= 0` | forbidden |
| `knn_rank` | integer | — | `>= 1` | forbidden |
| `control_generator_id` | identifier | — | non-empty identifier semantics | forbidden |
| `edge_lag` | integer | months | `>= 1` | forbidden |
| `overlap_fraction` | float | — | `[0, 1]` | forbidden |
| `exposure_intensity` | float | — | `>= 0` | forbidden |
| `elevation_difference_m` | float | m | finite | forbidden |
| `travel_time_min` | float | min | `>= 0` | forbidden |
| `accessibility_score` | float | — | `[0, 1]` | forbidden |
| `dependency_strength` | float | — | `[0, 1]` | forbidden |
| `capacity_score` | float | — | `>= 0` | raw nullable; model mask required |
| `similarity_score` | float | — | `[0, 1]` | forbidden |
| `membership_weight` | float | — | `[0, 1]` | forbidden |
| `administrative_level` | categorical | — | `municipality`, `census_division` | forbidden |

Names and units are part of the semantic contract and should be treated as stable.

---

## 7.5 `RelationEndpointContract`

### Purpose

`RelationSpec.source_node_types` and `target_node_types` define broad admissible sets. `RelationEndpointContract` can narrow their Cartesian product to exact permitted pairs.

```python
@dataclass(slots=True, frozen=True)
class RelationEndpointContract:
    allowed_pairs: tuple[tuple[str, str], ...] = ()
    schema_version: str = ENDPOINT_CONTRACT_SCHEMA_VERSION
```

An empty `allowed_pairs` tuple means the broad `RelationSpec` constraints apply without additional narrowing.

### Example

Broad constraints:

```text
source: {census_tract, municipality}
target: {municipality, census_division}
```

Naive Cartesian product would also permit invalid pairs such as tract → census division. The exact contract can restrict this to:

```text
census_tract -> municipality
municipality -> census_division
```

### Methods

```python
contract.explicit
contract.validate_against(specification)
contract.permits(specification, source_type, target_type)
contract.to_dict()
RelationEndpointContract.from_dict(payload)
```

Validation rejects malformed, duplicated, or semantically unsupported pairs.

For undirected relations, permission checks account for reversed endpoint orientation.

---

## 7.6 `RelationCapability`

### Purpose

`RelationCapability` separates current software availability from stable ontology meaning.

```python
@dataclass(slots=True, frozen=True)
class RelationCapability:
    implementation_state: str = IMPLEMENTATION_STATE_TARGET
    message_passing_allowed: bool = False
    training_allowed: bool = False
    explanation_policy: RelationExplanationPolicy = (
        RelationExplanationPolicy.EXCLUDED
    )
    schema_version: str = RELATION_CAPABILITY_SCHEMA_VERSION
```

### Why separation matters

Adding support for an existing semantic relation should change the operational fingerprint without changing the semantic fingerprint.

Current default capabilities are derived from:

```python
constants.V2_0_IMPLEMENTED_RELATION_NAMES
```

For a relation absent from that set:

```text
implementation_state = target
message_passing_allowed = False
training_allowed = False
explanation_policy = excluded
```

For an implemented non-control relation:

```text
implementation_state = implemented
message_passing_allowed = True
training_allowed = True
explanation_policy = allowed
```

For an implemented control relation:

```text
explanation_policy = diagnostic_only
```

### Methods and properties

```python
capability.implemented
capability.deprecated
capability.available_for_message_passing
capability.available_for_training
capability.validate()
capability.validate_against(specification)
capability.to_dict()
RelationCapability.from_dict(payload)
```

---

## 7.7 `RelationRegistryEntry`

A registry entry combines:

```text
RelationSpec
+ RelationEndpointContract
+ RelationCapability
```

```python
@dataclass(slots=True, frozen=True)
class RelationRegistryEntry:
    specification: RelationSpec
    endpoint_contract: RelationEndpointContract
    capability: RelationCapability
```

Convenience properties:

```python
entry.relation_id
entry.name
entry.implemented
entry.available_for_message_passing
entry.available_for_training
```

Endpoint query:

```python
entry.permits_endpoint_pair(source_type, target_type)
```

Identity views:

```python
entry.semantic_dict()
entry.operational_dict()
entry.to_dict()
RelationRegistryEntry.from_dict(payload)
```

The semantic view excludes current capability; the operational view includes it.

---

## 7.8 Canonical relation inventory

The default registry contains 21 edge relations.

### Controls and structure

| Stable ID | Name | Role | Direction | Construction | Temporal | Leakage | Required attributes |
|---:|---|---|---|---|---|---|---|
| 100 | `random_placebo` | control | directed | synthetic control | static | none | — |
| 110 | `centroid_knn` | control | directed | geometric | static | low | `distance_m`, `knn_rank` |
| 200 | `spatial_adjacency` | spatial | undirected | geometric | static | none | — |
| 210 | `administrative_membership` | administrative | directed | external static | static | none | — |

### Temporal memory

| Stable ID | Name | Parent | Direction | Construction | Temporal | Leakage | Required attributes |
|---:|---|---|---|---|---|---|---|
| 300 | `temporal_memory` | — | directed | as of origin | lagged | low | `edge_lag` |
| 310 | `historical_event_propagation` | `temporal_memory` | directed | as of origin | lagged | high | `edge_lag` |

### Exposure and protection

| Stable ID | Name | Parent | Source type | Temporal | Leakage | Required attributes |
|---:|---|---|---|---|---|---|
| 400 | `hydrological_exposure` | — | water body or flood zone | static | low | — |
| 410 | `flood_zone_exposure` | `hydrological_exposure` | flood zone | static | low | `overlap_fraction` |
| 420 | `low_elevation_exposure` | `hydrological_exposure` | water body | static | low | `distance_m`, `elevation_difference_m` |
| 430 | `heat_exposure` | — | heat island | snapshot | low | — |
| 500 | `canopy_protection` | — | green space | snapshot | low | — |

### Access

| Stable ID | Name | Parent | Source → target | Temporal | Required attributes |
|---:|---|---|---|---|---|
| 510 | `cooling_access` | `service_access` | prediction unit → service facility | interval-valid | `travel_time_min` |
| 600 | `service_access` | — | prediction unit → hospital/service facility | static | — |
| 610 | `road_access` | — | road segment → prediction unit/service node | interval-valid | — |

`road_access` is explicitly a root transport relation, not a subtype of `service_access`.

### Dependency

| Stable ID | Name | Parent | Source → target | Temporal | Required attributes |
|---:|---|---|---|---|---|
| 700 | `infrastructure_dependency` | — | critical infrastructure/drainage asset → prediction unit/service | interval-valid | — |
| 710 | `drainage_dependency` | `infrastructure_dependency` | drainage asset → prediction unit | interval-valid | — |
| 720 | `critical_facility_dependency` | `infrastructure_dependency` | critical infrastructure → hospital/service facility | interval-valid | — |

### Similarity

| Stable ID | Name | Direction | Construction | Temporal | Leakage | Required attributes |
|---:|---|---|---|---|---|---|
| 800 | `reporting_similarity` | undirected | training fitted | snapshot | high | `similarity_score` |
| 810 | `socioeconomic_similarity` | undirected | as of origin | snapshot | moderate | `similarity_score` |

### Cross-scale hierarchy

| Stable ID | Name | Parent / reverse | Source → target | Direction |
|---:|---|---|---|---|
| 900 | `cross_scale_parent` | reverse: `cross_scale_child` | tract → municipality; municipality → census division | directed |
| 910 | `cross_scale_child` | reverse: `cross_scale_parent` | municipality → tract; census division → municipality | directed |

`administrative_membership` is a semantic subtype of `cross_scale_parent`.

---

## 7.9 Exact endpoint contracts

Canonical prediction-unit node types:

```text
urban_unit
census_tract
census_division
municipality
```

For same-scale relations, the exact contract permits only same-type pairs:

```text
urban_unit -> urban_unit
census_tract -> census_tract
census_division -> census_division
municipality -> municipality
```

This applies to:

- random placebo;
- centroid kNN;
- spatial adjacency;
- temporal memory;
- historical event propagation;
- reporting similarity;
- socioeconomic similarity.

Selected heterogeneous endpoint contracts:

| Relation | Exact source → target pattern |
|---|---|
| `hydrological_exposure` | water body/flood zone → prediction unit |
| `flood_zone_exposure` | flood zone → prediction unit |
| `low_elevation_exposure` | water body → prediction unit |
| `heat_exposure` | heat island → prediction unit |
| `canopy_protection` | green space → prediction unit |
| `service_access` | prediction unit → hospital or service facility |
| `cooling_access` | prediction unit → service facility |
| `road_access` | road segment → prediction unit or service node |
| `infrastructure_dependency` | critical infrastructure or drainage asset → prediction unit or service node |
| `drainage_dependency` | drainage asset → prediction unit |
| `critical_facility_dependency` | critical infrastructure → hospital or service facility |
| `cross_scale_parent` | tract → municipality; municipality → census division |
| `cross_scale_child` | municipality → tract; census division → municipality |
| `administrative_membership` | tract → municipality; municipality → census division |

---

## 7.10 Relation-specific attribute contracts

| Relation | Required | Optional |
|---|---|---|
| `random_placebo` | — | `control_generator_id` |
| `centroid_knn` | `distance_m`, `knn_rank` | — |
| `spatial_adjacency` | — | `shared_boundary_length_m`, `distance_m` |
| `temporal_memory` | `edge_lag` | — |
| `historical_event_propagation` | `edge_lag` | `exposure_intensity`, `distance_m` |
| `hydrological_exposure` | — | `distance_m`, `overlap_fraction`, `exposure_intensity` |
| `flood_zone_exposure` | `overlap_fraction` | `distance_m`, `exposure_intensity` |
| `low_elevation_exposure` | `distance_m`, `elevation_difference_m` | `exposure_intensity` |
| `heat_exposure` | — | `overlap_fraction`, `exposure_intensity` |
| `canopy_protection` | — | `distance_m`, `overlap_fraction` |
| `service_access` | — | `distance_m`, `travel_time_min`, `accessibility_score`, `capacity_score` |
| `cooling_access` | `travel_time_min` | `distance_m`, `accessibility_score`, `capacity_score` |
| `road_access` | — | `distance_m`, `travel_time_min`, `accessibility_score` |
| `infrastructure_dependency` | — | `dependency_strength`, `capacity_score`, `distance_m` |
| `drainage_dependency` | — | `dependency_strength`, `capacity_score`, `distance_m` |
| `critical_facility_dependency` | — | `dependency_strength`, `capacity_score` |
| `reporting_similarity` | `similarity_score` | — |
| `socioeconomic_similarity` | `similarity_score` | — |
| `cross_scale_parent` | — | `membership_weight` |
| `cross_scale_child` | — | `membership_weight` |
| `administrative_membership` | — | `membership_weight`, `administrative_level` |

---

## 7.11 `RelationRegistry`

```python
@dataclass(slots=True, frozen=True)
class RelationRegistry:
    entries: tuple[RelationRegistryEntry, ...]
    registry_name: str = DEFAULT_RELATION_REGISTRY_NAME
    description: str = ...
    registry_version: str = constants.RELATION_REGISTRY_VERSION
    snapshot_schema_version: str = (
        RELATION_REGISTRY_SNAPSHOT_SCHEMA_VERSION
    )
```

Construction builds immutable indexes by stable ID and name.

### Core properties

```python
registry.specifications
registry.by_id
registry.by_name
registry.relation_ids
registry.relation_names
registry.spec_schema_version
```

### Validation

```python
registry.validate(
    require_current_registry_version=True,
    require_current_spec_schema_version=True,
)
registry.assert_current_compatibility()
```

Validation covers semantic specs, capabilities, endpoint contracts, reverse contracts, versions, canonical coverage, and identity uniqueness.

### Container behavior

```python
len(registry)
for entry in registry:
    ...
"spatial_adjacency" in registry
200 in registry
```

Membership supports canonical name or stable ID.

### Lookup

```python
registry.get_entry_by_id(200)
registry.get_entry_by_name("spatial_adjacency")
registry.get_spec_by_id(200)
registry.get_spec_by_name("spatial_adjacency")
```

Unknown identity raises `KeyError`.

### Capability filters

```python
registry.implemented()
registry.controls()
registry.non_controls()
registry.available_for_message_passing()
registry.available_for_training()
registry.available_for_explanation(
    include_diagnostic=False,
)
```

### Hierarchy traversal

```python
registry.parent_of("flood_zone_exposure")
registry.children_of("hydrological_exposure")
registry.children_of(
    "hydrological_exposure",
    recursive=True,
)
registry.ancestors_of("flood_zone_exposure")
registry.roots()
registry.family_of("flood_zone_exposure")
```

Returned order is deterministic and follows registry ordering or traversal order.

### Provenance requirements

```python
requirements = registry.provenance_requirements_for(
    "reporting_similarity"
)
```

The result combines:

- temporal fields required by the relation;
- `training_split_fingerprint` and `training_fit_cutoff` for training-fitted relations;
- `construction_as_of_time` for as-of-origin relations.

---

## 7.12 Hierarchy compilation policy

```python
class HierarchyCompilationPolicy(StrEnum):
    REJECT_OVERLAP = "reject_overlap"
    LEAF_ONLY = "leaf_only"
    ALLOW_OVERLAP = "allow_overlap"
```

### `REJECT_OVERLAP`

Rejects selections containing an ancestor and descendant together.

Example:

```text
hydrological_exposure
flood_zone_exposure
```

This is the default because selecting both may double-count the same mechanism.

### `LEAF_ONLY`

Removes selected ancestors when selected descendants exist.

### `ALLOW_OVERLAP`

Keeps both. Use only when the graph builder guarantees nonduplicated edge sets or the overlap is intentional and scientifically documented.

---

## 7.13 Runtime compilation

### General compiler

```python
compiled = registry.compile(
    relation_names,
    require_implemented=True,
    require_message_passing=True,
    require_training=False,
    allow_control_relations=True,
    hierarchy_policy=HierarchyCompilationPolicy.REJECT_OVERLAP,
)
```

Compilation rejects:

- empty selection;
- duplicate names;
- excluded non-edge concepts;
- unknown names;
- forbidden controls;
- unimplemented relations when required;
- relations unavailable for message passing;
- relations unavailable for training;
- hierarchy overlap under the selected policy.

After filtering, entries are sorted by stable ID.

### Training compiler

```python
compiled = registry.compile_for_training(
    relation_names,
    allow_control_relations=False,
    hierarchy_policy=HierarchyCompilationPolicy.REJECT_OVERLAP,
)
```

Requires implementation, message-passing support, and training support.

### Inference compiler

```python
compiled = registry.compile_for_inference(
    relation_names,
    allow_control_relations=False,
)
```

Requires implementation and message-passing support, but not training availability.

### Explanation compiler

```python
compiled = registry.compile_for_explanation(
    relation_names,
    include_diagnostic_controls=False,
)
```

Also enforces explanation policy.

---

## 7.14 Semantic and operational fingerprints

```python
registry.semantic_dict()
registry.operational_dict()

registry.semantic_fingerprint()
registry.operational_fingerprint()
registry.fingerprint()  # semantic alias
```

### Semantic fingerprint changes when

- relation meaning changes;
- IDs or names change;
- endpoints change;
- hierarchy changes;
- temporal or construction semantics change;
- edge-attribute contracts change.

### Operational fingerprint changes when

- implementation status changes;
- message-passing availability changes;
- training availability changes;
- explanation policy changes.

Adding software support for an existing relation should normally alter only the operational identity.

### Serialization

```python
payload = registry.to_dict()

restored = RelationRegistry.from_dict(
    payload,
    require_current_version=True,
    verify_serialized_fingerprints=True,
)
```

Unknown fields, invalid versions, and fingerprint mismatches are rejected.

---

## 7.15 `CompiledRelationRegistry`

### Purpose

`CompiledRelationRegistry` is the exact model-run relation vocabulary.

```python
@dataclass(slots=True, frozen=True)
class CompiledRelationRegistry:
    entries: tuple[RelationRegistryEntry, ...]
    source_registry_name: str
    source_registry_version: str
    source_semantic_fingerprint: str
    source_operational_fingerprint: str
    hierarchy_policy: HierarchyCompilationPolicy
    compiled_schema_version: str = (
        COMPILED_RELATION_REGISTRY_SCHEMA_VERSION
    )
```

It builds immutable mappings:

```text
canonical name -> dense relation_index
stable relation_id -> dense relation_index
```

### Properties

```python
compiled.specifications
compiled.relation_names
compiled.stable_relation_ids
compiled.relation_index_by_name
compiled.relation_index_by_id
```

### Lookup and encoding

```python
compiled.index_for_name("spatial_adjacency")
compiled.index_for_id(200)

compiled.entry_for_index(0)
compiled.spec_for_index(0)

compiled.encode_names(names)
compiled.encode_stable_ids(ids)

compiled.decode_indices_to_names(indices)
compiled.decode_indices_to_stable_ids(indices)
```

Invalid indices or identities raise clear errors.

### Endpoint query

```python
compiled.permits_endpoint_pair(
    relation_index,
    source_node_type,
    target_node_type,
)
```

### Runtime manifest

```python
compiled.runtime_entries()
```

This returns ordered relation-index metadata suitable for artifacts and diagnostics.

### Source verification

```python
compiled.assert_matches_source_registry(
    DEFAULT_RELATION_REGISTRY,
    require_operational_match=True,
)
```

Semantic mismatch is always invalid. Operational matching can be relaxed for artifacts whose semantics remain valid after software capability changes.

### Persistence

```python
payload = compiled.to_dict()
fingerprint = compiled.fingerprint()
restored = CompiledRelationRegistry.from_dict(payload)
```

Serialized runtime indices must be contiguous and ordered from zero.

---

# 8. `hazard_relation_priors.py`

## 8.1 Purpose

This module defines provisional expectations about relation relevance under different hazards and task scopes.

A high prior means:

> Before model fitting, this relation is provisionally expected to be comparatively relevant for this hazard and application context.

It does not mean:

- the relation is causally active;
- the relation must be selected;
- the relation increases risk;
- protective relations have positive effects;
- learned gates are forbidden from moving away from the prior.

Default values are ontology-derived, provisional, and not externally calibrated. The default registry allows weak initialization but does not approve substantive regularization.

---

## 8.2 Prior strength scale

| Strength | Mean |
|---|---:|
| `very_low` | 0.10 |
| `low` | 0.20 |
| `low_medium` | 0.35 |
| `medium` | 0.50 |
| `medium_high` | 0.65 |
| `high` | 0.80 |
| `very_high` | 0.90 |

The qualitative label determines `prior_mean` through `PRIOR_MEAN_BY_STRENGTH`.

A separate confidence value determines how strongly the prior is trusted.

---

## 8.3 Prior vocabularies

### Evidence type

```text
provisional_ontology
ontology
literature
expert
empirical
mixed
control
```

### Resolution mode

```text
explicit
relation_ancestor
hazard_ancestor
hazard_relation_ancestor
all_hazard
all_hazard_relation_ancestor
neutral_default
```

### Resolution policy

| Policy | Precedence after exact lookup |
|---|---|
| `hazard_first` | relation ancestors for exact hazard, then hazard ancestors for exact relation |
| `relation_first` | hazard ancestors for exact relation, then relation ancestors for exact hazard |
| `explicit_only` | no inheritance; missing cells become neutral unless explicitly required |

Combined hazard-and-relation ancestors and optional all-hazard fallback are considered after the one-dimensional candidates.

### Registry status

```text
provisional
reviewed
calibrated
```

### Gate initialization activation

Currently only:

```text
sigmoid
```

---

## 8.4 Applicability scope

### `PriorApplicationContext`

A concrete request context:

```python
@dataclass(slots=True, frozen=True)
class PriorApplicationContext:
    target_family: str
    target_name: str
    forecast_horizon: str
    geography_level: str
    study_region: str

    dataset_fingerprint: str | None = None
    study_id: str | None = None
```

All required strings must be non-empty.

### `PriorApplicabilityScope`

```python
@dataclass(slots=True, frozen=True)
class PriorApplicabilityScope:
    target_family: str | None = None
    target_names: tuple[str, ...] = ()
    forecast_horizons: tuple[str, ...] = ()
    geography_levels: tuple[str, ...] = ()
    study_regions: tuple[str, ...] = ()
    dataset_fingerprints: tuple[str, ...] = ()
    study_ids: tuple[str, ...] = ()
```

An empty tuple means unrestricted for that dimension.

```python
scope.matches(context)
scope.assert_applicable(context)
scope.to_dict()
scope.fingerprint()
PriorApplicabilityScope.from_dict(payload)
```

Default scope:

```text
target family: municipal_service_disruption_burden
target: water_drainage_count
horizon: next_month
geography: census_tract
region: montreal
```

Using the default prior registry outside this scope raises an applicability error.

---

## 8.5 Empirical prior provenance

`EmpiricalPriorProvenance` records leakage-sensitive origin information:

```python
@dataclass(slots=True, frozen=True)
class EmpiricalPriorProvenance:
    dataset_fingerprint: str
    split_fingerprint: str
    source_artifact_fingerprint: str
    estimation_cutoff: str
    estimator_name: str
    estimator_version: str
    held_out_estimation: bool
    random_seed: int
```

The random seed must be nonnegative. Required identity strings cannot be empty.

The object is mandatory for `empirical` prior evidence and may be included for `mixed` evidence.

---

## 8.6 `PriorCellDefinition`

```python
@dataclass(slots=True, frozen=True)
class PriorCellDefinition:
    strength: PriorStrength
    confidence: float
    evidence_type: PriorEvidenceType
    rationale: str

    caveat: str | None = None
    initialization_allowed: bool = True
    regularization_allowed: bool = False

    evidence_reference_ids: tuple[str, ...] = ()
    expert_source_ids: tuple[str, ...] = ()
    reviewed_by: tuple[str, ...] = ()
    review_date: str | None = None
    empirical_provenance: EmpiricalPriorProvenance | None = None
```

### Derived properties

```python
cell.prior_mean
cell.is_neutral
```

### Validation rules

- confidence must lie in `[0, 1]`;
- prior mean lies strictly in `(0, 1)`;
- rationale is required;
- review date uses ISO `YYYY-MM-DD`;
- zero-confidence cells cannot initialize or regularize;
- control priors must be neutral, zero-confidence, and non-operative;
- literature priors require reference IDs;
- expert priors require expert source IDs;
- empirical priors require empirical provenance;
- mixed priors require at least one auditable evidence source;
- empirical provenance is forbidden for unrelated evidence types.

### Serialization

```python
payload = cell.to_dict()
restored = PriorCellDefinition.from_dict(payload)
```

---

## 8.7 Explicit and profile-based priors

### `HazardRelationPrior`

Represents one explicit cell:

```python
@dataclass(slots=True, frozen=True)
class HazardRelationPrior:
    hazard: HazardKind
    relation_name: str
    definition: PriorCellDefinition
```

Convenience properties expose `prior_mean` and `confidence`.

### `RelationPriorProfile`

Defines one default relation prior plus hazard-specific overrides:

```python
@dataclass(slots=True, frozen=True)
class RelationPriorProfile:
    relation_name: str
    default_definition: PriorCellDefinition
    overrides: tuple[
        tuple[HazardKind, PriorCellDefinition],
        ...
    ] = ()
```

Methods:

```python
profile.override_map
profile.build_priors(hazard_registry=...)
```

The default profile is applied to all canonical hazards, then explicit hazard overrides replace selected cells.

---

## 8.8 Default prior profile summary

The following table shows default strength/confidence and the principal overrides. All default profiles are provisional unless marked as controls.

| Relation | Default | Important overrides |
|---|---|---|
| `random_placebo` | neutral, 0.00 control | none |
| `centroid_knn` | neutral, 0.00 control | none |
| `spatial_adjacency` | medium, 0.20 | pluvial flood high/0.45; flood and riverine flood medium-high/0.35; winter and civil-security variants medium-high |
| `temporal_memory` | medium-high, 0.35 | flood high/0.55; heat high/0.45; winter storm and snowstorm high/0.50 |
| `historical_event_propagation` | low-medium, 0.15 | flood high/0.40; pluvial flood high/0.45; civil security high/0.35 |
| `hydrological_exposure` | very low, 0.10 | riverine flood very high/0.75; flood very high/0.65 |
| `flood_zone_exposure` | very low, 0.10 | riverine flood very high/0.75; flood high/0.55 |
| `low_elevation_exposure` | very low, 0.10 | pluvial flood very high/0.65; flood high/0.55 |
| `heat_exposure` | very low, 0.10 | heat very high/0.70 |
| `canopy_protection` | very low, 0.10 | heat very high/0.65; protective relevance, not positive risk direction |
| `service_access` | medium-high, 0.35 | heat, road disruption, civil security, snowstorm, freezing rain high |
| `cooling_access` | very low, 0.10 | heat high/0.55 |
| `road_access` | low-medium, 0.20 | road disruption very high/0.65; snowstorm and freezing rain very high/0.65 |
| `infrastructure_dependency` | medium, 0.25 | outage very high/0.65; freezing rain very high/0.60 |
| `drainage_dependency` | very low, 0.10 | pluvial flood very high/0.70; flood high/0.55 |
| `critical_facility_dependency` | medium, 0.25 | outage high/0.55; civil security and freezing rain high/0.50 |
| `reporting_similarity` | medium, 0.15 | none; interpretation caveat for reporting propensity |
| `socioeconomic_similarity` | medium, 0.20 | heat and civil security high/0.40 |
| `cross_scale_parent` | medium-high, 0.25 | civil security and winter storm high/0.35 |
| `cross_scale_child` | medium-high, 0.25 | civil security and winter storm high/0.35 |
| `administrative_membership` | medium, 0.20 | civil security high/0.35 |

These values are initialization hypotheses, not findings.

---

## 8.9 `ResolvedHazardRelationPrior`

A resolved cell preserves both the request and the source of inherited information.

```python
@dataclass(slots=True, frozen=True)
class ResolvedHazardRelationPrior:
    hazard: HazardKind
    relation_name: str

    prior_mean: float
    confidence: float
    initialization_allowed: bool
    regularization_allowed: bool

    evidence_type: PriorEvidenceType
    rationale: str
    caveat: str | None

    resolution_mode: PriorResolutionMode
    source_hazard: HazardKind | None
    source_relation_name: str | None

    hazard_inheritance_distance: int = 0
    relation_inheritance_distance: int = 0
```

### Effective initialization mean

```python
effective = resolved.effective_initialization_mean(
    neutral_prior_mean=0.5
)
```

The effective mean shrinks the raw prior toward neutral according to confidence:

```text
effective = neutral + confidence × (prior_mean - neutral)
```

If initialization is disabled, the effective mean is neutral.

This prevents low-confidence ontology priors from creating extreme initial gate biases.

---

## 8.10 Two-dimensional prior resolution

Resolution begins with an exact cell.

If absent, candidate order depends on `PriorResolutionPolicy`.

For `HAZARD_FIRST`:

```text
1. exact hazard + relation ancestors
2. hazard ancestors + exact relation
3. hazard ancestors + relation ancestors
4. all-hazard + exact relation
5. all-hazard + relation ancestors
6. neutral default
```

For `RELATION_FIRST`, steps 1 and 2 are swapped.

Confidence is attenuated independently:

```text
resolved confidence =
    source confidence
    × hazard_decay ^ hazard_distance
    × relation_decay ^ relation_distance
```

Default decays:

```text
hazard inheritance:   0.80 per level
relation inheritance: 0.75 per level
```

A neutral default has:

```text
prior mean = registry.neutral_prior_mean
confidence = 0
initialization_allowed = False
regularization_allowed = False
```

### Resolution API

```python
resolved = prior_registry.resolve(
    hazard,
    relation_name,
    hazard_registry=DEFAULT_HAZARD_REGISTRY,
    relation_registry=DEFAULT_RELATION_REGISTRY,
    resolution_policy=PriorResolutionPolicy.HAZARD_FIRST,
    hazard_inheritance_confidence_decay=0.80,
    relation_inheritance_confidence_decay=0.75,
    allow_all_hazard_fallback=True,
    require_explicit=False,
)
```

`require_explicit=True` raises `KeyError` for a missing exact cell.

---

## 8.11 `HazardRelationPriorRegistry`

This immutable registry binds priors to:

- an applicability scope;
- a source hazard registry;
- a source relation registry name/version/fingerprint;
- source relation names and stable IDs;
- a maturity status;
- a regularization approval flag;
- a neutral prior mean.

### Lookup

```python
len(prior_registry)
for prior in prior_registry:
    ...

prior_registry.get_explicit(hazard, relation_name)
prior_registry.require_explicit(hazard, relation_name)
```

### Source validation

```python
prior_registry.validate_against_sources(
    hazard_registry=DEFAULT_HAZARD_REGISTRY,
    relation_registry=DEFAULT_RELATION_REGISTRY,
    require_complete=True,
    require_current_versions=True,
)
```

This verifies ontology coverage, names, IDs, versions, and fingerprints.

### Identity and persistence

```python
prior_registry.canonical_dict()
prior_registry.fingerprint()
prior_registry.to_dict()

restored = HazardRelationPriorRegistry.from_dict(
    payload,
    hazard_registry=...,
    relation_registry=...,
    require_current_versions=True,
    require_complete=True,
)
```

Unknown fields and incompatible sources are rejected.

---

## 8.12 Compiling priors for a model run

```python
compiled_priors = prior_registry.compile(
    compiled_relation_registry,
    source_hazard_registry=hazard_registry,
    source_relation_registry=relation_registry,
    application_context=context,
    hazards=("flood", "heat"),
    resolution_policy=PriorResolutionPolicy.HAZARD_FIRST,
    hazard_inheritance_confidence_decay=0.80,
    relation_inheritance_confidence_decay=0.75,
    allow_all_hazard_fallback=True,
    require_explicit=False,
    require_queryable_hazards=True,
    require_training_supported_hazards=False,
    allow_partially_data_backed=False,
    allow_fallback_hazard=False,
)
```

Compilation checks:

- applicability scope matches the request;
- prior registry matches source ontologies;
- compiled relation registry matches the source relation registry;
- at least one hazard is requested;
- hazard requests are unique;
- hazards are queryable or training-supported according to options;
- every hazard–relation cell resolves;
- output columns exactly match compiled relation order.

---

## 8.13 `CompiledHazardRelationPriors`

This is the dense hazard-by-relation artifact aligned to a model run.

Primary matrices, each shaped:

```text
[num_hazards][num_relations]
```

include:

- `prior_mean_matrix`;
- `confidence_matrix`;
- `effective_initialization_mean_matrix`;
- `initialization_mask`;
- `regularization_mask`;
- `resolution_mode_matrix`;
- `source_hazard_matrix`;
- `source_relation_matrix`;
- hazard inheritance distances;
- relation inheritance distances.

It also stores:

- hazard names and stable IDs;
- relation names and stable IDs;
- application context;
- resolution policy and decay parameters;
- source prior, hazard, relation, and compiled-registry identities.

### Lookup

```python
compiled_priors.num_hazards
compiled_priors.num_relations

compiled_priors.hazard_index_by_name
compiled_priors.relation_index_by_name

compiled_priors.hazard_index("flood")
compiled_priors.relation_index("spatial_adjacency")
```

### Gate initialization

```python
biases = compiled_priors.gate_bias_logit_matrix()
```

For enabled cells:

```text
bias = log(p / (1 - p))
```

where `p` is the confidence-adjusted effective mean clipped by `epsilon`.

Initialization-disabled cells receive zero bias, corresponding to a neutral sigmoid gate of 0.5.

Only sigmoid gates are currently supported.

### Regularization weights

```python
weights = compiled_priors.regularization_weight_matrix()
```

A cell receives its confidence only when regularization is permitted and globally approved; otherwise it receives zero.

The default provisional registry has `regularization_approved=False`.

### Source verification

```python
compiled_priors.assert_matches_sources(
    prior_registry=prior_registry,
    hazard_registry=hazard_registry,
    relation_registry=relation_registry,
    compiled_relation_registry=compiled,
)
```

This verifies every claimed source fingerprint, ordering, stable ID, and applicability identity.

### Persistence

```python
payload = compiled_priors.to_dict()
fingerprint = compiled_priors.fingerprint()
restored = CompiledHazardRelationPriors.from_dict(payload)
```

---

# 9. `relation_validation.py`

## 9.1 Purpose

`relation_validation.py` validates a PyTorch-normalized packed graph against a `CompiledRelationRegistry`.

It is intended to run:

- when an artifact is created;
- when a persisted artifact is loaded;
- before an experiment begins;
- in continuous integration;
- before publication or release.

It should not run inside the training hot path or once per epoch.

---

## 9.2 Validation profiles

```python
class RelationValidationProfile(StrEnum):
    DEVELOPMENT = "development"
    PERSISTED_ARTIFACT = "persisted_artifact"
    PUBLICATION = "publication"
```

### Development

Suitable for:

- graph construction;
- unit tests;
- exploratory in-memory artifacts.

Registry and graph fingerprints may be omitted.

### Persisted artifact

Requires:

- compiled-registry fingerprint;
- graph-artifact fingerprint.

### Publication

Requires:

- artifact fingerprints;
- source-registry verification;
- expected provenance identities;
- CPU validation;
- warnings promoted to errors by default.

---

## 9.3 Attribute representations

```python
class AttributeRepresentation(StrEnum):
    RAW = "raw"
    MODEL = "model"
```

### Raw representation

Permitted nulls may still be present. Missing values are checked against each attribute's raw missingness policy.

### Model representation

Values must be imputed, logically valid, bounded, and finite when required. Attributes whose raw values may be missing must carry a Boolean missingness mask recording original missingness.

A missingness mask does not excuse an invalid imputed value.

---

## 9.4 Time encodings

All temporal values in one artifact share a declared numerical encoding:

```text
month_index
unix_seconds
ordinal_day
integer_period_index
```

Calendar strings are not accepted. Raw adapters must parse them before building `RelationEdgeData`.

---

## 9.5 `ValidationIssue`

```python
@dataclass(slots=True, frozen=True)
class ValidationIssue:
    severity: ValidationSeverity
    code: str
    message: str
    relation_name: str | None = None
    graph_index: int | None = None
    edge_indices: tuple[int, ...] = ()
```

Issues are structured for CI, logging, artifact storage, and debugging.

---

## 9.6 `RelationValidationReport`

```python
@dataclass(slots=True, frozen=True)
class RelationValidationReport:
    issues: tuple[ValidationIssue, ...]
    profile: RelationValidationProfile
    schema_version: str
    num_nodes: int
    num_edges: int
    num_graphs: int
    num_relations_observed: int
```

Properties:

```python
report.errors
report.warnings
report.valid
report.counts_by_code
report.counts_by_relation
report.counts_by_severity
```

Methods:

```python
report.raise_for_errors()
report.to_dict()
```

`raise_for_errors()` raises `RelationValidationError`, which retains the full report and renders a preview of the first errors.

---

## 9.7 `EdgeAttributeColumn`

```python
@dataclass(slots=True, frozen=True)
class EdgeAttributeColumn:
    values: torch.Tensor | Sequence[Any]
    missing_mask: torch.Tensor | None = None
```

### Contract

- tensor values must be one-dimensional;
- non-tensor values must be a non-string sequence;
- sequences are normalized to tuples;
- missing masks must be one-dimensional Boolean tensors;
- mask length must equal value length.

Methods:

```python
len(column)
column.value_at(edge_index)
column.is_masked_missing(edge_index)
```

Use Python sequences for raw categorical, identifier, timestamp, or nullable values when a tensor representation is unsuitable. Use tensors for numeric model-facing columns.

---

## 9.8 `TemporalEdgeColumn`

```python
@dataclass(slots=True, frozen=True)
class TemporalEdgeColumn:
    values: torch.Tensor
    applicable_mask: torch.Tensor
```

### Contract

- values and mask are one-dimensional;
- mask uses `torch.bool`;
- lengths match;
- values use numeric tensor dtype;
- all values are finite, including non-applicable positions.

Non-applicable positions are identified by the mask rather than NaN or sentinel values.

Methods:

```python
len(column)
column.applies(edge_index)
column.value_at(edge_index)
```

---

## 9.9 Construction provenance

### `RelationConstructionProvenance`

One record applies to one relation in one packed graph:

```python
@dataclass(slots=True, frozen=True)
class RelationConstructionProvenance:
    relation_name: str
    graph_index: int

    construction_as_of_time: int | float | None = None
    training_fit_cutoff: int | float | None = None

    training_split_fingerprint: str | None = None
    source_artifact_fingerprint: str | None = None
    builder_version: str | None = None
```

Rules:

- graph index is nonnegative;
- numeric times are finite;
- identity strings are non-empty;
- training fit cutoff cannot be later than construction-as-of time.

### When provenance is required

- `training_fitted`: training split fingerprint and fit cutoff;
- `as_of_origin`: construction-as-of time;
- higher validation profiles may require expected source identities;
- builder version is recommended and may generate a warning when absent.

---

## 9.10 Control provenance

```python
@dataclass(slots=True, frozen=True)
class ControlGraphProvenance:
    relation_name: str
    graph_index: int
    control_kind: ControlGraphKind

    generator_name: str
    generator_version: str
    source_graph_fingerprint: str

    random_seed: int | None = None

    preserves_edge_count: bool = False
    preserves_degree_distribution: bool = False
    preserves_node_types: bool = True

    parameters: Mapping[
        str,
        str | int | float | bool,
    ] = {}
```

### Random placebo

Requires a nonnegative `random_seed`.

### Centroid kNN

Requires parameters:

```text
k
distance_metric
coordinate_reference_system
tie_breaking_policy
same_type_enforced
```

`same_type_enforced` must be `True`.

Parameter values must be non-empty strings, Booleans, integers, or finite floats.

---

## 9.11 External expectations

`RelationValidationExpectations` turns presence checks into identity checks.

```python
@dataclass(slots=True, frozen=True)
class RelationValidationExpectations:
    expected_compiled_registry_fingerprint: str | None = None
    expected_graph_artifact_fingerprint: str | None = None
    expected_training_split_fingerprint: str | None = None
    expected_source_artifact_fingerprint: str | None = None
    expected_time_encoding: TimeEncoding | None = None
```

Use this object when loading an artifact for a specific experiment.

---

## 9.12 Validation options

```python
@dataclass(slots=True, frozen=True)
class RelationValidationOptions:
    profile: RelationValidationProfile = DEVELOPMENT

    reject_duplicate_edges: bool = True

    require_reciprocal_undirected_storage: bool = False
    warn_on_nonreciprocal_undirected_storage: bool = True

    require_all_compiled_relations_present: bool = False
    warn_on_missing_compiled_relations: bool = True

    reject_values_for_undeclared_attributes: bool = True

    validity_end_inclusive: bool = False

    require_cpu_tensors: bool = True
    promote_warnings_to_errors: bool | None = None

    numeric_tolerance: float = 1e-6
    max_edge_indices_per_issue: int = 20
```

Derived profile properties:

```python
options.registry_fingerprint_required
options.graph_artifact_fingerprint_required
options.source_registry_required
options.expected_provenance_required
options.warnings_are_errors
options.cpu_required
```

Publication profile forces CPU validation and promotes warnings by default.

---

## 9.13 `RelationEdgeData`

### Purpose

`RelationEdgeData` is the concrete packed graph payload validated by this subsystem.

```python
@dataclass(slots=True, frozen=True)
class RelationEdgeData:
    edge_index: torch.Tensor
    edge_relation_index: torch.Tensor

    node_type_names: tuple[str, ...]
    node_batch_index: torch.Tensor

    attributes: Mapping[str, EdgeAttributeColumn] = {}
    attribute_representation: AttributeRepresentation = MODEL

    edge_ids: tuple[str, ...] = ()
    edge_stable_relation_id: torch.Tensor | None = None
    edge_batch_index: torch.Tensor | None = None

    origin_time_by_graph: torch.Tensor | None = None
    node_time: torch.Tensor | None = None

    edge_observation_time: TemporalEdgeColumn | None = None
    edge_valid_from: TemporalEdgeColumn | None = None
    edge_valid_to: TemporalEdgeColumn | None = None
    edge_lag: TemporalEdgeColumn | None = None

    construction_provenance: tuple[
        RelationConstructionProvenance, ...
    ] = ()

    control_provenance: tuple[
        ControlGraphProvenance, ...
    ] = ()

    compiled_registry_fingerprint: str | None = None
    graph_artifact_fingerprint: str | None = None

    time_encoding: TimeEncoding = TimeEncoding.MONTH_INDEX
```

### Required shapes

```text
edge_index:          [2, E]
edge_relation_index: [E]
node_batch_index:    [N]
node_type_names:     length N
```

### Recommended optional vectors

```text
edge_stable_relation_id: [E]
edge_batch_index:        [E]
origin_time_by_graph:    [G]
node_time:               [N]
```

`edge_stable_relation_id` is redundant with the compiled mapping but provides a strong artifact-integrity check.

`edge_batch_index` can be derived from endpoints when every edge remains inside one packed graph, but explicitly storing it allows direct consistency checks.

### Construction-time normalization

`RelationEdgeData.__post_init__()`:

- validates enum types;
- normalizes node types, edge IDs, and provenance sequences to tuples;
- validates attribute names and column types;
- freezes the attribute mapping;
- validates optional fingerprint strings.

Deep semantic validation is performed by `RelationValidator`, not by the dataclass constructor.

---

## 9.14 Structural validation

The validator checks:

- object and tensor types;
- `edge_index` rank, shape, and integer dtype;
- relation-index vector shape and integer dtype;
- node membership vector shape and integer dtype;
- edge and node counts;
- endpoint bounds;
- empty-edge and empty-node consistency;
- relation-index bounds;
- stable relation-ID alignment;
- edge-batch alignment;
- graph membership nonnegativity and contiguity;
- no cross-graph edges;
- node type length and canonical vocabulary;
- optional edge ID length, uniqueness, and non-empty values;
- lengths of every edge attribute and temporal column;
- origin and node-time vector lengths;
- CPU placement when required.

Only when the payload is structurally usable does validation continue to semantic checks.

---

## 9.15 Endpoint validation

For every edge:

1. decode `edge_relation_index` through the compiled registry;
2. obtain source and target node indices;
3. obtain source and target canonical node types;
4. apply the exact `RelationEndpointContract`;
5. reject forbidden self-loops.

An edge can satisfy the broad `RelationSpec` types and still fail the exact endpoint contract.

---

## 9.16 Edge-attribute validation

For each relation and declared attribute:

- required columns must exist;
- optional columns are validated when present;
- undeclared attribute values can be rejected;
- values are checked only on edges where the relation uses the attribute;
- logical kind is enforced;
- numeric values must be finite where required;
- integer values must be integral;
- bounds are enforced with `numeric_tolerance`;
- closed categorical values must belong to the declared vocabulary;
- raw missingness follows `MissingValuePolicy`;
- model-facing nullable attributes require masks;
- forbidden missingness is rejected;
- unexpected masks are rejected;
- imputed model values remain subject to type, bound, and finiteness checks.

### Mixed-relation columns

An attribute mapping is global to the packed artifact, but each column is interpreted only on edges whose relation declares that attribute. Values on unrelated edges should not be used as semantic data. With strict undeclared-value rejection enabled, semantically populated values for unrelated relations are errors.

---

## 9.17 Temporal validation

### Static relations

No relation-specific temporal column is required.

### Snapshot relations

Require applicable observation times. Snapshot time cannot be later than the graph prediction origin.

### Interval-valid relations

Require applicable `valid_from` and `valid_to` values.

The validator checks:

- valid interval ordering;
- relation is active at prediction origin;
- inclusive or exclusive end semantics according to `validity_end_inclusive`.

### Lagged relations

Require applicable positive `edge_lag`.

Checks include:

- lag is valid and positive;
- source and target times move forward;
- source time plus lag matches target time when node times are available;
- lagged observation does not occur after prediction origin;
- no future node state is consumed.

### Packed graph origins

`origin_time_by_graph[g]` supplies the prediction origin for graph `g`. Temporal and provenance checks use the edge's graph membership to obtain the correct origin.

---

## 9.18 Leakage-sensitive provenance validation

### Training-fitted relations

The validator checks:

- construction provenance exists for each relation/graph pair;
- training split fingerprint exists;
- fit cutoff exists;
- fit cutoff does not exceed prediction origin;
- expected split identity matches when supplied.

### As-of-origin relations

The validator checks:

- construction-as-of time exists;
- construction time does not exceed prediction origin.

### Source artifacts

Expected source fingerprints can be required and compared per relation/graph provenance record.

### Publication profile

Publication validation expects explicit identities, not only the presence of fields.

---

## 9.19 Control graph validation

For each control relation and packed graph:

- control provenance must exist;
- source graph fingerprint is checked;
- control kind must match relation semantics;
- random placebo must preserve node types;
- expected source fingerprint can be enforced;
- centroid-kNN parameters must be complete and same-type enforcement true;
- unused or duplicate provenance records are reported.

Control relations may be trained as experimental baselines when enabled, but they remain diagnostic rather than scientific pathway explanations.

---

## 9.20 Duplicate and undirected-storage diagnostics

### Duplicate edge identity

Duplicate detection uses relation and endpoints, scoped by graph. Depending on the relation and storage convention, repeated semantic edges are rejected when `reject_duplicate_edges=True`.

### Undirected relations

An undirected semantic relation may be stored as reciprocal arcs.

Options:

```python
require_reciprocal_undirected_storage=True
```

turns missing reciprocals into errors.

```python
warn_on_nonreciprocal_undirected_storage=True
```

emits warnings when reciprocal storage is not mandatory.

The semantic relation remains undirected regardless of physical storage.

---

## 9.21 Relation coverage

The validator compares observed relation indices with all compiled relations.

- `require_all_compiled_relations_present=True` makes missing relation edges an error.
- `warn_on_missing_compiled_relations=True` emits a warning otherwise.

A compiled relation with zero edges may be valid for a particular batch, but the absence should be explicit.

---

## 9.22 Validator execution order

`RelationValidator.validate()` performs:

```text
1. input type checks
2. compiled registry validation
3. source-registry verification
4. structural validation
5. artifact identity
6. endpoint contracts
7. edge attributes
8. temporal contracts
9. construction provenance
10. control provenance
11. duplicates
12. undirected storage
13. relation coverage
14. immutable report assembly
```

If structure is unusable, later checks are skipped to avoid misleading secondary failures.

---

## 9.23 Convenience API

Return a report without raising:

```python
report = validate_relation_edge_data(
    data,
    compiled,
    source_registry=DEFAULT_RELATION_REGISTRY,
    options=options,
    expectations=expectations,
)
```

Raise when errors exist:

```python
report = assert_valid_relation_edge_data(
    data,
    compiled,
    source_registry=DEFAULT_RELATION_REGISTRY,
    options=options,
    expectations=expectations,
)
```

The raising function still returns the report when valid.

---

# 10. Usage examples

## 10.1 Inspecting the canonical registry

```python
from urban_resilience_models.v2_hazard_conditioned_functional_ugnn.relations.relation_registry import (
    DEFAULT_RELATION_REGISTRY,
)

registry = DEFAULT_RELATION_REGISTRY

print(len(registry))
print(registry.relation_names)
print(registry.semantic_fingerprint())
print(registry.operational_fingerprint())

heat = registry.get_entry_by_name("heat_exposure")

print(heat.relation_id)
print(heat.specification.temporal_mode)
print(heat.endpoint_contract.allowed_pairs)
print(heat.capability.available_for_training)
```

---

## 10.2 Compiling a training vocabulary

```python
from urban_resilience_models.v2_hazard_conditioned_functional_ugnn.relations.relation_registry import (
    DEFAULT_RELATION_REGISTRY,
    HierarchyCompilationPolicy,
)

compiled = DEFAULT_RELATION_REGISTRY.compile_for_training(
    (
        "spatial_adjacency",
        "temporal_memory",
        "heat_exposure",
        "canopy_protection",
    ),
    allow_control_relations=False,
    hierarchy_policy=HierarchyCompilationPolicy.REJECT_OVERLAP,
)

for row in compiled.runtime_entries():
    print(row)
```

Do not select an ontology parent and descendant together under the default policy.

---

## 10.3 Encoding edge relation names

```python
relation_names = (
    "spatial_adjacency",
    "temporal_memory",
    "heat_exposure",
)

indices = compiled.encode_names(relation_names)

# Convert to the tensor dtype required by graph code.
edge_relation_index = torch.tensor(
    indices,
    dtype=torch.long,
)
```

---

## 10.4 Building a minimal static graph payload

```python
import torch

from urban_resilience_models.v2_hazard_conditioned_functional_ugnn.relations.relation_validation import (
    AttributeRepresentation,
    RelationEdgeData,
)

spatial_index = compiled.index_for_name(
    "spatial_adjacency"
)

data = RelationEdgeData(
    edge_index=torch.tensor(
        [
            [0, 1],
            [1, 0],
        ],
        dtype=torch.long,
    ),
    edge_relation_index=torch.tensor(
        [spatial_index, spatial_index],
        dtype=torch.long,
    ),
    node_type_names=(
        "census_tract",
        "census_tract",
    ),
    node_batch_index=torch.tensor(
        [0, 0],
        dtype=torch.long,
    ),
    attribute_representation=(
        AttributeRepresentation.MODEL
    ),
    compiled_registry_fingerprint=(
        compiled.fingerprint()
    ),
    graph_artifact_fingerprint=(
        "artifact-sha256-or-content-address"
    ),
)
```

---

## 10.5 Validating in development

```python
from urban_resilience_models.v2_hazard_conditioned_functional_ugnn.relations.relation_validation import (
    RelationValidationOptions,
    RelationValidationProfile,
    validate_relation_edge_data,
)

report = validate_relation_edge_data(
    data,
    compiled,
    options=RelationValidationOptions(
        profile=RelationValidationProfile.DEVELOPMENT,
        require_cpu_tensors=True,
    ),
)

print(report.valid)
print(report.counts_by_code)

report.raise_for_errors()
```

---

## 10.6 Publication-grade validation

```python
from urban_resilience_models.v2_hazard_conditioned_functional_ugnn.relations.relation_validation import (
    RelationValidationExpectations,
    RelationValidationOptions,
    RelationValidationProfile,
    assert_valid_relation_edge_data,
)

assert_valid_relation_edge_data(
    data,
    compiled,
    source_registry=DEFAULT_RELATION_REGISTRY,
    options=RelationValidationOptions(
        profile=RelationValidationProfile.PUBLICATION,
    ),
    expectations=RelationValidationExpectations(
        expected_compiled_registry_fingerprint=(
            compiled.fingerprint()
        ),
        expected_graph_artifact_fingerprint=(
            data.graph_artifact_fingerprint
        ),
        expected_time_encoding=data.time_encoding,
    ),
)
```

Publication profile requires all tensors used by validation to be on CPU.

---

## 10.7 Model-facing nullable attribute

```python
from urban_resilience_models.v2_hazard_conditioned_functional_ugnn.relations.relation_validation import (
    EdgeAttributeColumn,
)

capacity = EdgeAttributeColumn(
    values=torch.tensor(
        [0.0, 1.25, 0.75],
        dtype=torch.float32,
    ),
    missing_mask=torch.tensor(
        [True, False, False],
        dtype=torch.bool,
    ),
)
```

The first value is an imputed finite placeholder whose original missingness is preserved by the mask.

---

## 10.8 Lagged temporal relation

```python
from urban_resilience_models.v2_hazard_conditioned_functional_ugnn.relations.relation_validation import (
    TemporalEdgeColumn,
)

edge_lag = TemporalEdgeColumn(
    values=torch.tensor(
        [1, 1],
        dtype=torch.long,
    ),
    applicable_mask=torch.tensor(
        [True, True],
        dtype=torch.bool,
    ),
)

data = RelationEdgeData(
    edge_index=...,
    edge_relation_index=...,
    node_type_names=...,
    node_batch_index=...,
    node_time=torch.tensor(
        [10, 11],
        dtype=torch.long,
    ),
    origin_time_by_graph=torch.tensor(
        [11],
        dtype=torch.long,
    ),
    edge_lag=edge_lag,
    ...
)
```

The source node time, target node time, and lag must agree.

---

## 10.9 Training-fitted provenance

```python
from urban_resilience_models.v2_hazard_conditioned_functional_ugnn.relations.relation_validation import (
    RelationConstructionProvenance,
)

provenance = RelationConstructionProvenance(
    relation_name="reporting_similarity",
    graph_index=0,
    training_fit_cutoff=36,
    construction_as_of_time=36,
    training_split_fingerprint="split-fingerprint",
    source_artifact_fingerprint="source-fingerprint",
    builder_version="1.0.0",
)
```

For a prediction origin of 40, both cutoff and construction time must be at or before 40.

---

## 10.10 Random placebo provenance

```python
from urban_resilience_models.v2_hazard_conditioned_functional_ugnn.relations.relation_validation import (
    ControlGraphKind,
    ControlGraphProvenance,
)

control = ControlGraphProvenance(
    relation_name="random_placebo",
    graph_index=0,
    control_kind=ControlGraphKind.RANDOM_PLACEBO,
    generator_name="degree_aware_rewire",
    generator_version="1.2.0",
    source_graph_fingerprint="real-graph-fingerprint",
    random_seed=42,
    preserves_edge_count=True,
    preserves_degree_distribution=True,
    preserves_node_types=True,
    parameters={
        "rewire_attempts": 5000,
    },
)
```

---

## 10.11 Compiling default hazard priors

```python
from urban_resilience_models.v2_hazard_conditioned_functional_ugnn.relations.hazard_relation_priors import (
    PriorApplicationContext,
    compile_default_hazard_relation_priors,
)

context = PriorApplicationContext(
    target_family=(
        "municipal_service_disruption_burden"
    ),
    target_name="water_drainage_count",
    forecast_horizon="next_month",
    geography_level="census_tract",
    study_region="montreal",
)

compiled_priors = compile_default_hazard_relation_priors(
    compiled,
    application_context=context,
    hazards=(
        "flood",
        "pluvial_flood",
        "heat",
    ),
)

gate_biases = compiled_priors.gate_bias_logit_matrix()
```

The relation columns exactly follow `compiled.relation_names`.

---

## 10.12 Resolving one prior with provenance

```python
resolved = (
    DEFAULT_HAZARD_RELATION_PRIOR_REGISTRY
    .resolve(
        "pluvial_flood",
        "flood_zone_exposure",
        hazard_registry=DEFAULT_HAZARD_REGISTRY,
        relation_registry=DEFAULT_RELATION_REGISTRY,
    )
)

print(resolved.prior_mean)
print(resolved.confidence)
print(resolved.resolution_mode)
print(resolved.source_hazard)
print(resolved.source_relation_name)
```

Always inspect `resolution_mode` and source fields when interpreting inherited priors.

---

## 10.13 Serializing registry artifacts

```python
registry_payload = (
    DEFAULT_RELATION_REGISTRY.to_dict()
)
compiled_payload = compiled.to_dict()
prior_payload = compiled_priors.to_dict()

manifest = {
    "relation_registry": registry_payload,
    "compiled_relations": compiled_payload,
    "compiled_hazard_priors": prior_payload,
}
```

Persist fingerprints next to the payload and verify them on load.

---

# 11. Fingerprint taxonomy

| Fingerprint | Captures | Typical use |
|---|---|---|
| Relation registry semantic | Stable ontology meaning | Graph/checkpoint compatibility |
| Relation registry operational | Current software capability | Runtime availability audit |
| Compiled relation registry | Selected ordered runtime vocabulary | Tensor and checkpoint identity |
| Prior applicability scope | Permitted task/dataset scope | Prevent prior misuse |
| Prior registry | All prior cells and source identities | Reproducible initialization |
| Compiled hazard priors | Ordered hazard-by-relation matrices | Gate initialization artifact |
| Graph artifact | Concrete graph content/version | Load-time integrity |
| Training split | Data used to fit leakage-sensitive relations | Leakage audit |
| Source artifact | Upstream graph/data source | Provenance verification |
| Control source graph | Real graph from which control was generated | Placebo reproducibility |

Fingerprints are deterministic SHA-256 hashes over canonical JSON or explicit artifact values, depending on the object.

---

# 12. Error model

The package distinguishes:

- `TypeError`: wrong object or field type;
- `ValueError`: invalid semantic or structural value;
- `KeyError`: requested identity is absent;
- `IndexError`: invalid dense runtime index where applicable;
- `NotImplementedError`: canonical relation requested but unavailable under capability requirements;
- `RelationValidationError`: a completed validation report contains errors;
- `RuntimeError`: internal ontology drift, impossible state, or cycle detected after prior validation.

Validation reports are preferred over fail-fast exceptions for concrete graph artifacts because they allow multiple independent issues to be diagnosed in one pass.

---

# 13. Validation issue-code reference

## 13.1 Identity and registry

```text
compiled_registry_fingerprint_mismatch
graph_artifact_fingerprint_mismatch
missing_compiled_registry_fingerprint
missing_graph_artifact_fingerprint
missing_source_registry
source_registry_mismatch
stable_relation_id_mismatch
time_encoding_mismatch
unexpected_compiled_registry_fingerprint
```

## 13.2 Structural tensors and packed graphs

```text
edge_index_type
edge_index_shape
edge_index_dtype
edge_endpoint_bounds
relation_index_bounds
edge_without_nodes
node_batch_index errors
negative_node_batch_index
noncontiguous_graph_indices
cross_graph_edge
edge_batch_mismatch
origin_for_empty_batch
origin_graph_count_mismatch
```

The exact report code for relation-index bounds is `relation_index_bounds`.

## 13.3 Node and edge identities

```text
invalid_node_type_name
unknown_node_type
edge_id_length
invalid_edge_id
duplicate_edge_ids
duplicate_relation_edges
forbidden_self_loop
invalid_endpoint_pair
```

## 13.4 Attributes and missingness

```text
attribute_length
missing_required_attribute_column
undeclared_attribute_value
invalid_attribute_value
forbidden_attribute_missingness
missing_required_missingness_mask
unexpected_missingness_mask
```

## 13.5 Temporal integrity

```text
missing_prediction_origin
missing_temporal_column
missing_temporal_applicability
temporal_column_length
snapshot_after_origin
invalid_validity_interval
edge_inactive_at_origin
invalid_edge_lag
edge_lag_alignment
nonforward_temporal_edge
node_time_lag_mismatch
future_node_state
lagged_observation_after_origin
```

## 13.6 Construction provenance

```text
construction_provenance_type
duplicate_construction_provenance
missing_construction_provenance
missing_construction_as_of_time
construction_after_origin
missing_training_fit_cutoff
training_fit_after_origin
missing_training_split_fingerprint
training_split_fingerprint_mismatch
missing_expected_training_split_fingerprint
missing_expected_source_artifact_fingerprint
source_artifact_fingerprint_mismatch
provenance_graph_bounds
missing_builder_version
unused_construction_provenance
```

## 13.7 Control provenance

```text
control_provenance_type
duplicate_control_provenance
missing_control_provenance
control_source_fingerprint_mismatch
missing_expected_control_source_fingerprint
control_node_type_not_preserved
unused_control_provenance
```

## 13.8 Storage and coverage diagnostics

```text
missing_undirected_reciprocal
missing_compiled_relation_edges
non_cpu_validation_tensor
```

Warnings are promoted to errors under publication profile unless explicitly overridden.

---

# 14. Common integration mistakes

## 14.1 Using stable IDs as tensor indices

Incorrect:

```python
edge_relation_index = torch.tensor([200, 300, 430])
```

Correct:

```python
edge_relation_index = torch.tensor(
    compiled.encode_stable_ids(
        [200, 300, 430]
    ),
    dtype=torch.long,
)
```

---

## 14.2 Omitting the compiled registry from an artifact

Dense indices are meaningless without their ordered vocabulary. Persist the compiled registry or at least its complete runtime mapping and fingerprint.

---

## 14.3 Selecting a parent and child relation together

Under the default policy, this raises because it can double-count one mechanism.

Choose the child, use `LEAF_ONLY`, or document and explicitly allow overlap.

---

## 14.4 Treating controls as explanations

`random_placebo` and `centroid_knn` may be useful baselines. They are diagnostic controls, not urban mechanisms.

---

## 14.5 Treating a high protective prior as positive risk direction

Prior strength means expected relevance, not sign. `canopy_protection` may have high heat relevance while reducing risk.

---

## 14.6 Using default priors outside their scope

The default scope is Montreal census-tract next-month water/drainage burden. Create a reviewed custom scope and registry for other targets or regions.

---

## 14.7 Regularizing with provisional priors

The default registry explicitly disallows substantive regularization. Use it for diagnostics or weak initialization only.

---

## 14.8 Filling non-applicable temporal fields with NaN

Use `TemporalEdgeColumn.applicable_mask`. Every numeric value must remain finite.

---

## 14.9 Using a missingness mask to hide invalid model values

The imputed value must still satisfy dtype, logical kind, bounds, and finiteness.

---

## 14.10 Constructing similarity edges from full data

`reporting_similarity` is training-fitted and high-risk. Record split identity and fit cutoff, and never use validation or test outcomes to construct it.

---

## 14.11 Building as-of-origin edges after the prediction origin

`temporal_memory`, `historical_event_propagation`, and socioeconomic similarity require origin-aware construction. Future information is rejected.

---

## 14.12 Assuming undirected semantics imply one storage convention

The ontology is undirected; the graph loader may store one pair or reciprocal arcs. Configure validation consistently with the message-passing implementation.

---

## 14.13 Supplying broad endpoint types without exact contracts

Hierarchical and heterogeneous relations need explicit allowed pairs. Do not rely on a broad Cartesian product when only selected pairs are meaningful.

---

## 14.14 Running the validator every epoch

Validation is artifact-level and intentionally strict. Run it before training, not in the hot path.

---

# 15. Extension guide

## 15.1 Adding a new relation

1. Add the canonical relation name to `constants.py`.
2. Allocate a new stable ID in the correct semantic range.
3. Never reuse an old ID.
4. Define or reuse edge-attribute specs.
5. Create a capability-neutral `RelationSpec`.
6. Define an exact `RelationEndpointContract`.
7. Add the entry to `build_default_relation_entries()`.
8. Add parent or reverse metadata when applicable.
9. Update canonical coverage checks.
10. Decide whether the relation belongs in `V2_0_IMPLEMENTED_RELATION_NAMES`.
11. Add prior profile coverage.
12. Add validator tests for endpoints, attributes, temporal fields, and provenance.
13. Regenerate semantic and operational manifests intentionally.

---

## 15.2 Adding a relation subtype

Declare `parent_relation_name` and ensure:

- parent exists;
- hierarchy is acyclic;
- endpoint semantics are compatible;
- compilation overlap behavior is tested;
- prior inheritance produces intended results;
- a parent and child are not selected together unintentionally.

---

## 15.3 Adding a reverse relation

Both relations must:

- be directed;
- name each other as reverse;
- reverse endpoint constraints;
- agree on registry version, schema version, evidence type, construction mode, leakage risk, temporal mode, and control status.

---

## 15.4 Adding an edge attribute

1. Choose a stable semantic name and unit.
2. Define logical kind.
3. Define raw bounds.
4. Define categorical vocabulary where needed.
5. Define missingness policy.
6. Add it to required or optional relation collections.
7. Add raw and model representation tests.
8. Add boundary, dtype, missingness, and undeclared-use tests.
9. Update this reference and artifact schemas.

---

## 15.5 Marking a relation implemented

Update the coordinated capability source in `constants.py`, not the semantic meaning.

Verify:

- graph builder exists;
- message-passing path handles the relation;
- training is supported;
- explanation policy is appropriate;
- operational fingerprint changes;
- semantic fingerprint remains unchanged;
- checkpoint compatibility is reviewed.

---

## 15.6 Creating a custom prior registry

1. Define a precise `PriorApplicabilityScope`.
2. Build auditable `PriorCellDefinition` objects.
3. Use literature, expert, or empirical provenance fields correctly.
4. Set status to reviewed or calibrated only after the relevant process.
5. Enable regularization only with explicit scientific approval.
6. Bind the registry to exact hazard and relation source fingerprints.
7. test exact and inherited resolution;
8. compile against the intended relation vocabulary;
9. verify source identities on load.

---

## 15.7 Adding another gate activation

`CompiledHazardRelationPriors.gate_bias_logit_matrix()` currently supports sigmoid only.

A new activation requires:

- an explicit enum value;
- a mathematically defined inverse link;
- endpoint clipping rules;
- neutral initialization semantics;
- tests for masked and confidence-adjusted cells;
- serialization compatibility.

---

## 15.8 Adding a validation profile

A new profile should define:

- fingerprint requirements;
- source-registry requirements;
- provenance expectation requirements;
- CPU policy;
- warning-promotion behavior.

Avoid hidden profile behavior outside `RelationValidationOptions` properties.

---

# 16. Testing checklist

## 16.1 `relation_types.py`

- every enum round-trips through serialization;
- edge-attribute bounds and kinds are enforced;
- nullable raw attributes require finite model values;
- unknown fields are rejected;
- wildcard type rules are enforced;
- control semantics are enforced;
- construction/leakage combinations are enforced;
- lagged relations must be directed;
- required and optional attributes cannot overlap;
- reverse relations are reciprocal;
- hierarchy cycles are rejected;
- duplicate names and IDs are rejected;
- immutable indexes are read-only.

## 16.2 `relation_registry.py`

- default coverage is complete;
- excluded non-edge concepts are absent;
- stable IDs are unique and sorted after compilation;
- endpoint contracts reject invalid pair combinations;
- parent and reverse contracts are valid;
- semantic and operational fingerprints differ only for intended changes;
- compile rejects duplicates and unknown names;
- control restrictions are enforced;
- hierarchy policies behave as documented;
- encode/decode round-trips;
- serialized compiled indices are contiguous;
- source-registry verification detects drift.

## 16.3 `hazard_relation_priors.py`

- strength-to-mean mapping is stable;
- evidence-specific provenance requirements are enforced;
- applicability scope accepts and rejects correctly;
- exact resolution wins;
- hazard-first and relation-first precedence differ as intended;
- confidence decays independently by both inheritance distances;
- all-hazard fallback works;
- neutral defaults are non-operative;
- provisional cells cannot regularize;
- gate bias matrix uses zero for disabled cells;
- source identity verification detects drift;
- default registry covers the source ontology.

## 16.4 `relation_validation.py`

- malformed tensor shapes and dtypes are reported;
- out-of-bounds endpoints and relation indices are reported;
- graph membership must be contiguous;
- cross-graph edges are rejected;
- stable relation IDs match dense indices;
- exact endpoint pairs are enforced;
- self-loop policies are enforced;
- required attributes are present;
- undeclared values are rejected;
- raw/model missingness rules differ correctly;
- numeric bounds and categories are enforced;
- static, snapshot, interval, and lagged modes are tested;
- future information is rejected;
- training-fitted and as-of-origin provenance are tested;
- controls require reproducibility metadata;
- duplicates and reciprocal arcs are diagnosed;
- relation coverage options work;
- publication profile promotes warnings and requires identities;
- report serialization is stable.

---

# 17. Operational guidance

## 17.1 During graph construction

- compile the relation vocabulary before assigning edge types;
- use canonical node-type names;
- store stable relation IDs in addition to dense indices when possible;
- produce exact endpoint pairs;
- preserve raw missingness before imputation;
- attach graph-scoped construction provenance;
- compute a content-addressed artifact fingerprint;
- validate on CPU before persistence.

## 17.2 During training

- load and verify the compiled registry;
- validate persisted graph artifacts once;
- keep control relations in explicit ablation runs;
- compile priors against the exact relation order;
- preserve prior and registry fingerprints in checkpoints;
- avoid revalidating the full graph each epoch.

## 17.3 During inference

- require semantic compatibility with the checkpoint;
- verify compiled relation order exactly;
- reject unseen or reordered relation vocabularies;
- use inference compilation without controls by default;
- verify time encoding and prediction origin.

## 17.4 During explanation

- compile only explanation-available relations;
- keep diagnostic controls separate from scientific pathways;
- export dense index, stable ID, and canonical name;
- report relation hierarchy when summarizing subtypes;
- include hazard-prior resolution source only as initialization provenance, not as learned evidence.

## 17.5 During publication

- use `PUBLICATION` validation profile;
- supply source registry and all expectations;
- preserve complete validation report;
- publish semantic and compiled-registry manifests;
- document hierarchy policy;
- document control generation;
- report whether priors initialized gates and whether regularization was disabled.

---

# 18. Public symbols by file

## 18.1 `relation_types.py`

```text
ANY_NODE_TYPE
EdgeAttributeKind
EdgeAttributeSpec
FUNCTIONAL_RELATION_ROLES
MissingValuePolicy
RELATION_SPEC_SCHEMA_VERSION
RelationConstructionMode
RelationDirection
RelationEvidenceType
RelationExplanationPolicy
RelationLeakageRisk
RelationSpec
RelationTemporalMode
TEMPORAL_FIELD_EDGE_LAG
TEMPORAL_FIELD_EDGE_OBSERVATION_TIME
TEMPORAL_FIELD_EDGE_VALID_FROM
TEMPORAL_FIELD_EDGE_VALID_TO
TOPOLOGY_ONLY_NO_EDGE_NAME
relation_specs_by_id
relation_specs_by_name
validate_relation_spec_collection
```

## 18.2 `relation_registry.py`

```text
ATTRIBUTE_ACCESSIBILITY_SCORE
ATTRIBUTE_ADMINISTRATIVE_LEVEL
ATTRIBUTE_CAPACITY_SCORE
ATTRIBUTE_CONTROL_GENERATOR_ID
ATTRIBUTE_DEPENDENCY_STRENGTH
ATTRIBUTE_DISTANCE_M
ATTRIBUTE_EDGE_LAG
ATTRIBUTE_ELEVATION_DIFFERENCE_M
ATTRIBUTE_EXPOSURE_INTENSITY
ATTRIBUTE_KNN_RANK
ATTRIBUTE_MEMBERSHIP_WEIGHT
ATTRIBUTE_OVERLAP_FRACTION
ATTRIBUTE_SHARED_BOUNDARY_LENGTH_M
ATTRIBUTE_SIMILARITY_SCORE
ATTRIBUTE_TRAVEL_TIME_MIN
COMPILED_RELATION_REGISTRY_SCHEMA_VERSION
CompiledRelationRegistry
DEFAULT_RELATION_ENTRIES
DEFAULT_RELATION_REGISTRY
DEFAULT_RELATION_REGISTRY_NAME
EDGE_REGISTRY_EXCLUDED_CANONICAL_NAMES
ENDPOINT_CONTRACT_SCHEMA_VERSION
HierarchyCompilationPolicy
RELATION_CAPABILITY_SCHEMA_VERSION
RELATION_ID_ADMINISTRATIVE_MEMBERSHIP
RELATION_ID_CANOPY_PROTECTION
RELATION_ID_CENTROID_KNN
RELATION_ID_COOLING_ACCESS
RELATION_ID_CRITICAL_FACILITY_DEPENDENCY
RELATION_ID_CROSS_SCALE_CHILD
RELATION_ID_CROSS_SCALE_PARENT
RELATION_ID_DRAINAGE_DEPENDENCY
RELATION_ID_FLOOD_ZONE_EXPOSURE
RELATION_ID_HEAT_EXPOSURE
RELATION_ID_HISTORICAL_EVENT_PROPAGATION
RELATION_ID_HYDROLOGICAL_EXPOSURE
RELATION_ID_INFRASTRUCTURE_DEPENDENCY
RELATION_ID_LOW_ELEVATION_EXPOSURE
RELATION_ID_RANDOM_PLACEBO
RELATION_ID_REPORTING_SIMILARITY
RELATION_ID_ROAD_ACCESS
RELATION_ID_SERVICE_ACCESS
RELATION_ID_SOCIOECONOMIC_SIMILARITY
RELATION_ID_SPATIAL_ADJACENCY
RELATION_ID_TEMPORAL_MEMORY
RELATION_REGISTRY_SNAPSHOT_SCHEMA_VERSION
RelationCapability
RelationEndpointContract
RelationRegistry
RelationRegistryEntry
build_default_relation_entries
build_default_relation_registry
get_default_relation_registry
```

## 18.3 `hazard_relation_priors.py`

```text
COMPILED_HAZARD_PRIOR_SCHEMA_VERSION
CompiledHazardRelationPriors
DEFAULT_HAZARD_RELATION_PRIOR_REGISTRY
DEFAULT_HAZARD_RELATION_PRIOR_REGISTRY_NAME
DEFAULT_PRIOR_APPLICATION_CONTEXT
DEFAULT_PRIOR_APPLICABILITY_SCOPE
DEFAULT_RELATION_PRIOR_PROFILES
EMPIRICAL_PRIOR_PROVENANCE_SCHEMA_VERSION
EmpiricalPriorProvenance
GateInitializationActivation
HAZARD_RELATION_PRIOR_REGISTRY_VERSION
HAZARD_RELATION_PRIOR_SCHEMA_VERSION
HazardRelationPrior
HazardRelationPriorRegistry
PRIOR_APPLICABILITY_SCHEMA_VERSION
PRIOR_MEAN_BY_STRENGTH
PriorApplicationContext
PriorApplicabilityScope
PriorCellDefinition
PriorEvidenceType
PriorRegistryStatus
PriorResolutionMode
PriorResolutionPolicy
PriorStrength
RelationPriorProfile
ResolvedHazardRelationPrior
build_default_hazard_relation_prior_registry
compile_default_hazard_relation_priors
get_default_hazard_relation_prior_registry
```

## 18.4 `relation_validation.py`

```text
AttributeRepresentation
CONTROL_PROVENANCE_SCHEMA_VERSION
ControlGraphKind
ControlGraphProvenance
EdgeAttributeColumn
RELATION_PROVENANCE_SCHEMA_VERSION
RELATION_VALIDATION_SCHEMA_VERSION
RelationConstructionProvenance
RelationEdgeData
RelationValidationError
RelationValidationExpectations
RelationValidationOptions
RelationValidationProfile
RelationValidationReport
RelationValidator
TemporalEdgeColumn
TimeEncoding
ValidationIssue
ValidationSeverity
assert_valid_relation_edge_data
validate_relation_edge_data
```

---

# 19. Runtime assumptions

- Python 3.11 or newer is required.
- PyTorch is required for concrete artifact validation.
- Canonical names and software capability constants must remain coordinated with `constants.py`.
- Hazard priors require the hazard registry.
- Validation time values are numerical and share one declared encoding.
- Publication validation is CPU-oriented.
- Registry and prior dataclasses are frozen, but contained tensors are not made immutable by Python.
- Mapping proxies prevent structural mutation of mappings, not mutation of referenced tensor storage.
- SHA-256 fingerprints are deterministic identity aids, not cryptographic signatures of trust.
- Filesystem persistence is intentionally outside this package.

---

# 20. Current limitations and deliberate non-goals

- The package does not build graph edges.
- It does not scale or impute edge attributes.
- It does not define PyTorch message-passing layers.
- It does not learn hazard gates.
- Prior matrices support sigmoid gate-bias conversion only.
- Default priors are provisional and scoped narrowly.
- Current relation capability depends on the coordinated constants manifest.
- The validator does not accept calendar strings.
- The validator is not optimized for per-batch training execution.
- Attention or learned gate values are not validated here.
- Causal interpretation is outside the package.
- Impervious surface remains a node feature until a valid source entity exists.
- No-edge experiments remain topology modes rather than relations.

---

# 21. Summary contract

A correct integration preserves the following invariants:

```text
Canonical ontology:
    stable relation_id
    canonical relation name
    semantic role
    exact endpoint pairs
    temporal and construction semantics
    edge-attribute contracts

Runtime model:
    dense relation_index in [0, R)
    deterministic relation ordering
    compiled-registry fingerprint

Graph artifact:
    valid packed tensors
    canonical node types
    relation-index/stable-ID agreement
    valid attributes and missingness
    no temporal leakage
    graph-scoped provenance
    reproducible controls
    artifact fingerprint

Hazard priors:
    applicable task scope
    explicit or auditable inherited source
    confidence attenuation
    exact alignment with compiled relation order
    no provisional substantive regularization

Explanations:
    ordinary pathways only from allowed relations
    controls remain diagnostic
    indices exported with stable IDs and names
```

The core rule is:

> Relation semantics are stable ontology identities; runtime indices are compiled artifacts; hazard priors are scoped initialization evidence; concrete edges are valid only after structural, temporal, provenance, and identity checks.
