# Relation Family Ontology

**Model family:** `v2_hazard_conditioned_functional_ugnn`
**File:** `urban_resilience_models/v2_hazard_conditioned_functional_ugnn/docs/relation_family_ontology.md`
**Status:** design document / ontology draft
**Purpose:** define the stable semantic relation families used by the V2 hazard-conditioned functional urban graph model.

---

## 1. Why this ontology exists

The V2 model is not intended to be a generic spatial graph model.

It is intended to learn how different hazards activate different **functional urban pathways**.

That means the graph must distinguish relations such as:

```text
spatial adjacency
hydrological exposure
drainage dependency
heat exposure
canopy protection
service access
infrastructure dependency
reporting similarity
temporal memory
```

These relations should not be collapsed into one generic edge type.

The central principle is:

```text
Edges are not only connections.
Edges represent urban mechanisms.
```

A flood query should not activate the same relations as a heat query.

A road-disruption query should not activate the same relations as a drainage query.

The relation ontology defines the vocabulary that allows the model to ask:

```text
Which urban mechanisms are active under this hazard?
Which relation families carried the prediction?
Which pathways explain the risk?
```

---

## 2. Design principles

### 2.1 Stable IDs before implementation

Every relation family must have a stable machine-readable ID.

Good:

```text
hydrological_exposure
drainage_dependency
canopy_protection
service_access
random_placebo
```

Bad:

```text
hydro stuff
flood edge
nearby thing
random graph version 2
```

Relation IDs must remain stable across:

```text
data builders
graph loaders
relation registries
message-passing layers
explanation exports
UI payloads
tests
model cards
```

---

### 2.2 Functional meaning before geometry

A relation should describe **why information can move**, not only that two nodes are close.

For example:

```text
tract A is near tract B
```

is spatial.

But:

```text
tract A drains toward the same low-elevation basin as tract B
```

is functional.

The long-term goal is to move from:

```text
spatial graph
```

to:

```text
functional urban graph
```

Spatial relations are still useful, but they should not be the only graph structure.

---

### 2.3 Control relations must be explicitly labeled

Some relations exist for scientific controls, not because they represent real urban mechanisms.

Examples:

```text
random_placebo
centroid_knn
identity_no_edge
```

These are useful for benchmarking, but explanation exports must treat them differently.

A `random_placebo` edge should never be presented to a decision-maker as a real urban pathway.

---

### 2.4 Hazard relevance is a prior, not a hard rule

Some relations are expected to matter more for certain hazards.

Example:

```text
hydrological_exposure → flood
canopy_protection → heat
service_access → transport disruption
infrastructure_dependency → outage
```

But these expectations should guide the model, not fully constrain it.

The model should be able to learn surprising associations when supported by data.

---

### 2.5 Explanations must preserve relation meaning

Relation families are core explanation objects.

The model should eventually export:

```text
top_relation_families
relation_gate_values
relation_pathway_scores
edge_attention_by_relation
counterfactual_relation_deltas
```

The explanation layer should not only say:

```text
edge 182 mattered
```

It should say:

```text
hydrological exposure relations were highly activated under the flood query,
and specific edges connected this tract to nearby low-elevation or water-exposed areas.
```

---

## 3. Relation family contract

Every relation family should eventually be represented by a structured contract.

A relation contract should include:

```text
relation_id
display_name
description
semantic_role
source_node_types
target_node_types
directedness
is_control_relation
is_real_relation
allowed_hazards
expected_edge_attributes
required_edge_attributes
explanation_allowed
default_hazard_priors
validation_rules
```

A conceptual example:

```text
relation_id: hydrological_exposure
display_name: Hydrological exposure
description: Connects urban units to water bodies, flood zones, drainage basins, or hydrological exposure features.
semantic_role: exposure
source_node_types: tract, cd, urban_unit
target_node_types: water_body, flood_zone, basin, urban_unit
directedness: directed
is_control_relation: false
is_real_relation: true
allowed_hazards: flood, civil_security_event, all_hazard
expected_edge_attributes:
  - distance_m
  - overlap_area
  - exposure_score
  - upstream_downstream_flag
explanation_allowed: true
```

This document defines the conceptual ontology.
The code-level version should live in:

```text
relations/relation_types.py
relations/relation_registry.py
relations/hazard_relation_priors.py
relations/relation_validation.py
```

---

## 4. Relation families overview

The ontology is organized into relation groups:

```text
A. Control and baseline relations
B. Spatial and administrative relations
C. Temporal and memory relations
D. Hazard exposure relations
E. Protection and mitigation relations
F. Access and service relations
G. Infrastructure and dependency relations
H. Similarity and reporting relations
I. Cross-scale relations
```

Not all relation families need to be implemented in V2.0.

The first implementation should start small and expand carefully.

---

# 5. A. Control and baseline relations

Control relations are used to test whether the graph model is learning meaningful topology or merely benefiting from smoothing, neural capacity, or arbitrary connectivity.

They are essential for scientific discipline.

---

## 5.1 `identity_no_edge`

### Meaning

No message passing between distinct nodes.

Each node is processed independently.

### Scientific role

Tests whether neural capacity and node features alone explain performance.

### Typical use

```text
B4_no_edge_neural
hazard-conditioned no-edge neural ablation
```

### Directedness

Not applicable.

### Explanation policy

Should not be presented as an urban pathway.

### V2.0 status

Essential.

---

## 5.2 `random_placebo`

### Meaning

A randomized graph topology used as a placebo control.

Ideally, it preserves some graph statistics from the real graph, such as edge count or degree distribution.

### Scientific role

Tests whether arbitrary graph smoothing helps.

If the real graph does not beat `random_placebo`, topology-specific value is weak.

### Typical edge attributes

```text
random_seed
rewiring_method
degree_preserved
source_graph
```

### Directedness

Usually follows the directedness convention of the graph being randomized.

### Explanation policy

Must not be presented as a real pathway.

Should be labeled as a control relation.

### V2.0 status

Essential.

---

## 5.3 `centroid_knn`

### Meaning

Connects urban units to nearest neighbors by centroid distance.

### Scientific role

Tests whether generic spatial proximity is enough.

If `centroid_knn` beats real adjacency, then the model may be learning distance-based spatial smoothing rather than administrative topology.

### Typical edge attributes

```text
distance_m
rank_k
k_value
```

### Directedness

Usually directed:

```text
source node → nearest-neighbor node
```

Can be symmetrized if configured.

### Explanation policy

Can be used for diagnostic explanations, but should be clearly described as generic spatial proximity, not a specific physical mechanism.

### V2.0 status

Essential.

---

# 6. B. Spatial and administrative relations

These relations represent geographic or administrative structure.

They may be useful, but they are not always mechanistic.

---

## 6.1 `spatial_adjacency`

### Meaning

Connects urban units that share a boundary or are topologically adjacent.

Examples:

```text
tract ↔ tract
CD ↔ CD
municipality ↔ municipality
```

### Scientific role

Tests whether real spatial contiguity supports prediction.

### Typical edge attributes

```text
shared_boundary_length_m
touches
queen_adjacency
rook_adjacency
```

### Directedness

Usually undirected but stored as bidirectional directed edges:

```text
A → B
B → A
```

### Relevant hazards

```text
flood
heat
road_disruption
civil_security_event
all_hazard
```

### Explanation policy

Can be presented as spatial adjacency, but should not be overinterpreted as a specific mechanism unless paired with a functional relation.

### V2.0 status

Essential.

---

## 6.2 `administrative_membership`

### Meaning

Connects lower-level units to higher-level administrative units.

Examples:

```text
tract → CD
tract → municipality
municipality → region
```

### Scientific role

Supports multi-scale modeling.

Allows vulnerability or event information to move between spatial resolutions.

### Typical edge attributes

```text
membership_weight
area_overlap
population_overlap
```

### Directedness

Usually directed:

```text
lower-level unit → higher-level unit
higher-level unit → lower-level unit
```

These directions may have different meanings and should be represented carefully.

### Relevant hazards

```text
all_hazard
civil_security_event
```

### Explanation policy

Can be presented as a scale-aggregation or administrative context pathway.

### V2.0 status

Optional. More relevant for V3.

---

# 7. C. Temporal and memory relations

Temporal relations represent continuity or history.

They may be represented as graph edges, memory sequences, or both.

---

## 7.1 `temporal_memory`

### Meaning

Connects a node’s past states to its current state.

Examples:

```text
tract at month t-1 → tract at month t
tract at month t-3 → tract at month t
```

### Scientific role

Represents persistence and historical stress.

Useful when modeling a temporal graph directly.

### Typical edge attributes

```text
lag_months
time_delta
history_window
```

### Directedness

Directed from past to present:

```text
node_t_minus_k → node_t
```

### Relevant hazards

```text
all_hazard
flood
heat
outage
road_disruption
civil_security_event
```

### Explanation policy

Can be presented as historical memory or temporal dependency.

### V2.0 status

Optional as an edge type.
The first implementation may represent memory through features or encoders instead.

---

## 7.2 `historical_event_propagation`

### Meaning

Connects areas with past events or complaints to areas that may experience future burden.

This relation can encode empirical propagation or repeated spatial-temporal stress patterns.

### Scientific role

Supports learned propagation from past stress locations.

### Typical edge attributes

```text
past_event_count
past_event_type
lag_window
historical_cooccurrence
```

### Directedness

Usually directed from past source to current target.

### Relevant hazards

```text
flood
heat
civil_security_event
all_hazard
```

### Explanation policy

Can be presented as a historical stress pathway.

### V2.0 status

Optional. More advanced than simple lag/rolling memory.

---

# 8. D. Hazard exposure relations

Exposure relations connect urban units to hazard sources, hazard zones, or exposure surfaces.

These are central for hazard-conditioned modeling.

---

## 8.1 `hydrological_exposure`

### Meaning

Connects urban units to rivers, water bodies, drainage basins, flood-prone zones, or hydrological exposure features.

### Scientific role

Represents flood or water-related exposure pathways.

### Typical edge attributes

```text
distance_to_water_m
distance_to_river_m
distance_to_flood_zone_m
overlap_with_flood_zone
water_surface_percentage
elevation_difference
basin_id
```

### Directedness

Usually directed:

```text
urban unit → water or exposure feature
exposure feature → urban unit
```

The direction should be explicit.

### Relevant hazards

```text
flood
civil_security_event
all_hazard
```

### Expected hazard prior

High for flood.

Low for heat.

### Explanation policy

Can be presented as a flood exposure pathway.

### V2.0 status

Strong priority if flood features are available.

---

## 8.2 `flood_zone_exposure`

### Meaning

Connects urban units to mapped flood-prone or flood-risk zones.

### Scientific role

More specific version of hydrological exposure.

Useful if flood-zone polygons are available.

### Typical edge attributes

```text
overlap_area
overlap_fraction
distance_to_zone_m
flood_zone_class
return_period
```

### Directedness

Usually directed from urban unit to flood-zone feature, or bidirectional if message passing requires it.

### Relevant hazards

```text
flood
civil_security_event
```

### Expected hazard prior

Very high for flood.

### Explanation policy

Can be presented as flood-zone exposure, with caveats if the flood-zone map is indicative rather than legal/definitive.

### V2.0 status

Optional but desirable.

---

## 8.3 `low_elevation_exposure`

### Meaning

Connects urban units to low-elevation or terrain-risk features.

This can also be represented as node features rather than edges.

### Scientific role

Represents terrain-based flood susceptibility.

### Typical edge attributes

```text
mean_elevation
min_elevation
slope
topographic_position_index
flow_accumulation
```

### Directedness

If represented as edges, usually from urban unit to terrain feature.

### Relevant hazards

```text
flood
civil_security_event
```

### Expected hazard prior

High for flood.

### Explanation policy

Can be presented as terrain-related exposure.

### V2.0 status

Likely starts as features, not edges.

---

## 8.4 `heat_exposure`

### Meaning

Connects urban units to heat islands, surface-temperature zones, impervious surfaces, or thermal exposure features.

### Scientific role

Represents heat-related exposure.

### Typical edge attributes

```text
heat_island_class
land_surface_temperature
impervious_surface_fraction
distance_to_heat_island_m
overlap_with_heat_island
```

### Directedness

Usually directed:

```text
urban unit → heat exposure feature
heat exposure feature → urban unit
```

### Relevant hazards

```text
heat
civil_security_event
all_hazard
```

### Expected hazard prior

Very high for heat.

Low for flood unless heat-exposure variables proxy urbanization.

### Explanation policy

Can be presented as a heat-exposure pathway.

### V2.0 status

Strong priority if heat features are available.

---

## 8.5 `impervious_surface_exposure`

### Meaning

Connects urban units to imperviousness or built-surface exposure.

This can affect both heat and flood.

### Scientific role

Represents runoff potential and heat retention.

### Typical edge attributes

```text
impervious_fraction
built_surface_fraction
surface_class
```

### Relevant hazards

```text
flood
heat
civil_security_event
```

### Expected hazard prior

Medium-high for flood.

Medium-high for heat.

### Explanation policy

Can be presented as built-surface exposure.

### V2.0 status

Likely starts as features, later can become relation family.

---

# 9. E. Protection and mitigation relations

Protection relations represent urban features that reduce or buffer hazard impact.

---

## 9.1 `canopy_protection`

### Meaning

Connects urban units to tree canopy, vegetation, parks, or green infrastructure.

### Scientific role

Represents heat mitigation and, sometimes, stormwater absorption or runoff reduction.

### Typical edge attributes

```text
canopy_fraction
ndvi
green_space_area
distance_to_green_space_m
overlap_with_green_space
```

### Directedness

Usually directed:

```text
urban unit → canopy/green feature
canopy/green feature → urban unit
```

### Relevant hazards

```text
heat
flood
civil_security_event
```

### Expected hazard prior

Very high for heat.

Medium for flood.

### Explanation policy

Can be presented as a protection or mitigation pathway.

### V2.0 status

Strong priority for heat modeling if data is available.

---

## 9.2 `cooling_access`

### Meaning

Connects urban units to cooling centers, public indoor spaces, libraries, shelters, parks, or other heat-adaptation resources.

### Scientific role

Represents adaptive capacity during heat events.

### Typical edge attributes

```text
distance_to_cooling_site_m
travel_time
capacity
opening_hours_available
accessibility_score
```

### Directedness

Usually directed:

```text
urban unit → cooling resource
```

### Relevant hazards

```text
heat
```

### Expected hazard prior

High for heat.

### Explanation policy

Can be presented as adaptive-capacity or cooling-access pathway.

### V2.0 status

Optional. More likely V3.

---

# 10. F. Access and service relations

Access relations represent proximity or travel access to services and critical facilities.

---

## 10.1 `service_access`

### Meaning

Connects urban units to hospitals, clinics, shelters, emergency facilities, cooling centers, schools, or other service nodes.

### Scientific role

Represents the ability of a population to access services during or after disruption.

### Typical edge attributes

```text
distance_m
travel_time
service_type
capacity
accessibility_score
network_distance_m
```

### Directedness

Usually directed:

```text
urban unit → service node
service node → urban unit
```

These directions may encode different meanings.

### Relevant hazards

```text
heat
road_disruption
outage
civil_security_event
all_hazard
```

### Expected hazard prior

Medium-high for heat if service is cooling-related.

High for road disruption and civil-security access scenarios.

### Explanation policy

Can be presented as service-access pathway.

### V2.0 status

Optional. More likely V3.

---

## 10.2 `road_access`

### Meaning

Connects urban units to road segments, bridges, transit links, or transportation corridors.

### Scientific role

Represents accessibility and possible propagation of road disruption.

### Typical edge attributes

```text
distance_to_road_m
road_class
traffic_volume
network_centrality
travel_time
bridge_or_tunnel_flag
```

### Directedness

Usually directed or bidirectional depending on network representation.

### Relevant hazards

```text
road_disruption
flood
civil_security_event
outage
```

### Expected hazard prior

High for road disruption.

Medium for flood.

### Explanation policy

Can be presented as access or transport-disruption pathway.

### V2.0 status

Optional. More likely V3.

---

# 11. G. Infrastructure and dependency relations

Dependency relations connect urban units to assets whose failure or capacity affects disruption risk.

---

## 11.1 `drainage_dependency`

### Meaning

Connects urban units to drainage infrastructure, sewer outfalls, stormwater assets, catch basins, drainage zones, or modeled drainage dependencies.

### Scientific role

Represents urban drainage-system dependency.

This is especially important for flood and water/drainage targets.

### Typical edge attributes

```text
distance_to_drainage_asset_m
asset_type
catchment_id
outfall_id
flow_direction
capacity_proxy
upstream_downstream_flag
```

### Directedness

Often directed.

Possible directions:

```text
urban unit → drainage asset
drainage asset → urban unit
upstream area → downstream area
```

### Relevant hazards

```text
flood
civil_security_event
```

### Expected hazard prior

Very high for flood.

### Explanation policy

Can be presented as drainage-dependency pathway.

### V2.0 status

High scientific value, but depends on data availability.

---

## 11.2 `infrastructure_dependency`

### Meaning

Connects urban units to infrastructure assets or systems whose function affects resilience.

Examples:

```text
power infrastructure
water infrastructure
telecom assets
pumping stations
transport hubs
critical facilities
```

### Scientific role

Represents service dependency and cascading disruption.

### Typical edge attributes

```text
asset_type
distance_m
dependency_type
capacity_proxy
criticality_score
service_area_id
```

### Directedness

Often directed:

```text
urban unit → infrastructure asset
infrastructure asset → served urban unit
```

### Relevant hazards

```text
outage
flood
road_disruption
civil_security_event
all_hazard
```

### Expected hazard prior

High for outage.

Medium for flood and civil-security events.

### Explanation policy

Can be presented as infrastructure-dependency pathway.

### V2.0 status

Optional. More likely V3.

---

## 11.3 `critical_facility_dependency`

### Meaning

Connects urban units to critical facilities such as hospitals, emergency centers, fire stations, shelters, or major public-service facilities.

### Scientific role

Represents dependence on critical facilities or exposure of critical services.

### Typical edge attributes

```text
facility_type
distance_m
capacity
service_area
criticality_score
```

### Relevant hazards

```text
outage
road_disruption
heat
civil_security_event
all_hazard
```

### Expected hazard prior

Medium-high for civil-security events.

High for service-access questions.

### Explanation policy

Can be presented as critical-facility dependency.

### V2.0 status

Optional. More likely V3.

---

# 12. H. Similarity and reporting relations

These relations capture similarity between areas or reporting dynamics.

They are useful but must be interpreted carefully.

---

## 12.1 `reporting_similarity`

### Meaning

Connects areas with similar 311 reporting behavior.

### Scientific role

Controls for observed-report patterns that may not equal true disruption.

Useful when targets are complaint/report counts.

### Typical edge attributes

```text
historical_total_311_similarity
category_distribution_similarity
reporting_rate_similarity
population_adjusted_reporting_similarity
```

### Directedness

Usually undirected or bidirectional.

### Relevant hazards

```text
all_hazard
flood
heat
civil_security_event
```

### Expected hazard prior

Medium when predicting observed reports.

Lower when predicting physical events.

### Explanation policy

Should be presented carefully as reporting-pattern similarity, not physical risk propagation.

### V2.0 status

Optional. Useful if reporting-bias module is developed.

---

## 12.2 `socioeconomic_similarity`

### Meaning

Connects areas with similar social vulnerability or demographic profiles.

### Scientific role

Allows the model to share vulnerability patterns across non-adjacent but similar areas.

### Typical edge attributes

```text
svi_similarity
sovi_similarity
demographic_similarity
income_similarity
age_structure_similarity
isolation_similarity
```

### Directedness

Usually undirected or bidirectional.

### Relevant hazards

```text
heat
flood
civil_security_event
all_hazard
```

### Expected hazard prior

Medium across many hazards.

### Explanation policy

Can be presented as vulnerability-profile similarity.

Must not be confused with geographic adjacency.

### V2.0 status

Optional.

---

# 13. I. Cross-scale relations

Cross-scale relations connect entities at different spatial or functional resolutions.

---

## 13.1 `cross_scale_parent`

### Meaning

Connects a lower-level unit to a parent geography.

Examples:

```text
tract → CD
tract → municipality
municipality → administrative region
```

### Scientific role

Allows multi-scale context.

Useful when combining tract-level SVI and CD-level SoVI or civil-security event data.

### Typical edge attributes

```text
parent_type
area_overlap
population_overlap
membership_weight
```

### Directedness

Directed:

```text
child → parent
```

### Relevant hazards

```text
all_hazard
civil_security_event
```

### Explanation policy

Can be presented as multi-scale administrative context.

### V2.0 status

Optional. More likely V3.

---

## 13.2 `cross_scale_child`

### Meaning

Connects a parent geography to lower-level units.

Examples:

```text
CD → tract
municipality → tract
```

### Scientific role

Allows coarse event or vulnerability information to inform local predictions.

### Typical edge attributes

```text
child_type
area_overlap
population_overlap
membership_weight
```

### Directedness

Directed:

```text
parent → child
```

### Relevant hazards

```text
all_hazard
civil_security_event
```

### Explanation policy

Can be presented as downscaled administrative context.

### V2.0 status

Optional. More likely V3.

---

# 14. Hazard relevance matrix

This matrix gives qualitative prior expectations.

These are not hard-coded truths. They are starting priors for modeling and interpretation.

| Relation family                |       Flood |        Heat |     Outage | Road disruption | Civil-security / all-hazard |
| ------------------------------ | ----------: | ----------: | ---------: | --------------: | --------------------------: |
| `identity_no_edge`             |     control |     control |    control |         control |                     control |
| `random_placebo`               |     control |     control |    control |         control |                     control |
| `centroid_knn`                 |      medium |      medium | low-medium |          medium |                      medium |
| `spatial_adjacency`            | medium-high |      medium |     medium |     medium-high |                 medium-high |
| `temporal_memory`              |        high |        high |     medium |          medium |                        high |
| `historical_event_propagation` |        high |      medium |     medium |          medium |                        high |
| `hydrological_exposure`        |   very high |         low |        low |          medium |                      medium |
| `flood_zone_exposure`          |   very high |         low |        low |          medium |                      medium |
| `low_elevation_exposure`       |        high |         low |        low |          medium |                      medium |
| `heat_exposure`                |         low |   very high |        low |             low |                      medium |
| `impervious_surface_exposure`  | medium-high | medium-high |        low |      low-medium |                      medium |
| `canopy_protection`            |      medium |   very high |        low |             low |                      medium |
| `cooling_access`               |         low |        high | low-medium |          medium |                      medium |
| `service_access`               |  low-medium | medium-high |     medium |            high |                        high |
| `road_access`                  |      medium |         low |     medium |       very high |                        high |
| `drainage_dependency`          |   very high |         low | low-medium |          medium |                      medium |
| `infrastructure_dependency`    |      medium |  low-medium |  very high |     medium-high |                        high |
| `critical_facility_dependency` |      medium |      medium |       high |            high |                        high |
| `reporting_similarity`         |      medium |      medium |     medium |          medium |                      medium |
| `socioeconomic_similarity`     |      medium |        high |     medium |          medium |                        high |
| `cross_scale_parent`           |      medium |      medium |     medium |          medium |                        high |
| `cross_scale_child`            |      medium |      medium |     medium |          medium |                        high |

---

# 15. V2.0 immediate relation subset

The first implementation should not include every relation family.

A disciplined V2.0 relation subset should be:

```text
identity_no_edge
random_placebo
centroid_knn
spatial_adjacency
temporal_memory or lag/rolling memory features
hydrological_exposure if available
heat_exposure if available
canopy_protection if available
drainage_dependency if available
```

If exposure/protection/dependency relations are not ready as edges, they can start as node features.

The model should still define the relation IDs early so that future data can plug in without renaming the ontology.

---

# 16. Relation implementation stages

## 16.1 Stage A — Control graph stage

Implement:

```text
identity_no_edge
random_placebo
centroid_knn
spatial_adjacency
```

Purpose:

```text
preserve benchmark discipline
```

---

## 16.2 Stage B — Hazard-feature stage

Add exposure/protection signals as node features:

```text
hydrological exposure features
heat exposure features
canopy/protection features
impervious surface features
drainage proxies
```

Purpose:

```text
test hazard conditioning before full heterogeneity
```

---

## 16.3 Stage C — Functional edge stage

Convert selected hazard signals into relation families:

```text
hydrological_exposure
heat_exposure
canopy_protection
drainage_dependency
```

Purpose:

```text
test whether functional relation edges add value beyond node features
```

---

## 16.4 Stage D — Heterogeneous graph stage

Add non-tract node types:

```text
water bodies
flood zones
heat islands
green spaces
roads
hospitals
drainage assets
critical infrastructure
```

Purpose:

```text
move toward a full heterogeneous functional urban graph
```

---

# 17. Required validation checks

Every edge table should be validated before training.

Validation should check:

```text
valid relation family IDs
valid source node IDs
valid target node IDs
valid node types
required edge attributes present
no missing endpoints
no duplicate edges unless allowed
directedness convention respected
control relations labeled correctly
real relations not accidentally mixed with placebo relations
edge weights finite
distance values nonnegative
relation family allowed for current experiment
```

These checks should be implemented in:

```text
relations/relation_validation.py
data/graph_loaders.py
```

---

# 18. Explanation policy

Not every relation should be explained the same way.

## 18.1 Explanation-allowed relations

These can be presented as meaningful urban pathways:

```text
spatial_adjacency
temporal_memory
historical_event_propagation
hydrological_exposure
flood_zone_exposure
low_elevation_exposure
heat_exposure
impervious_surface_exposure
canopy_protection
cooling_access
service_access
road_access
drainage_dependency
infrastructure_dependency
critical_facility_dependency
reporting_similarity
socioeconomic_similarity
cross_scale_parent
cross_scale_child
```

## 18.2 Control relations

These should be included in diagnostic outputs but not presented as real pathways:

```text
identity_no_edge
random_placebo
centroid_knn
```

`centroid_knn` is not fully meaningless, but it should be described as generic spatial proximity rather than a specific functional mechanism.

## 18.3 Explanation fields

Relation-level explanations should include:

```text
relation_id
display_name
semantic_role
hazard_id
gate_value
mean_attention
top_edges
pathway_score
is_control_relation
is_real_relation
explanation_allowed
```

---

# 19. Relation naming conventions

Use lowercase snake case.

Good:

```text
hydrological_exposure
drainage_dependency
canopy_protection
service_access
random_placebo
```

Avoid:

```text
HydroExposure
floodEdge
canopy-protection
Service Access
randomGraph
```

Use nouns or noun phrases.

Avoid implementation-specific names.

Good:

```text
relation_family_gate
hydrological_exposure
```

Bad:

```text
mlp_gate_1
edge_type_7
```

The ontology should survive implementation changes.

---

# 20. Non-goals

This ontology should not attempt to solve every modeling question immediately.

Non-goals for the first version:

```text
perfect causal relation definitions
full infrastructure dependency map
complete Québec-wide heterogeneous graph
complete road network model
complete sewer-network topology
all hazards fully supported from day one
```

The ontology is a stable direction, not a claim that all relations are immediately available.

---

# 21. Open questions

Important open questions:

```text
Should temporal memory be represented as graph edges, memory sequences, or both?
Should hydrological exposure be node features first, edges later?
Should kNN be treated only as a control or also as a legitimate spatial relation?
How should relation priors be regularized without hard-coding hazard behavior?
How should we validate that relation gates are stable across random seeds?
How should relation-family explanations be evaluated?
When should reporting similarity become a real relation rather than a confound control?
```

These questions should be revisited after the first V2.0 experiments.

---

# 22. North-star summary

The relation ontology exists so that the model can move from:

```text
generic graph edges
```

to:

```text
hazard-conditioned functional urban pathways
```

The goal is not just to predict future burden.

The goal is to predict while answering:

```text
which urban mechanisms were activated,
under which hazard,
through which relations,
and with what uncertainty.
```

That is why relation families are central to the V2 architecture.
