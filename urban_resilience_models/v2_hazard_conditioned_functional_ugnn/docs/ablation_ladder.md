# Ablation Ladder

**Model family:** `v2_hazard_conditioned_functional_ugnn`
**File:** `urban_resilience_models/v2_hazard_conditioned_functional_ugnn/docs/ablation_ladder.md`
**Status:** official experimental-design draft
**Purpose:** define the controlled sequence of experiments required to justify each component of the V2 architecture

---

## 1. Why this document exists

The V2 architecture contains many potentially useful mechanisms:

```text
urban memory
hazard encoding
scenario encoding
hazard-queried memory
relation-family gates
edge-level attention
relation-specific transformations
functional message passing
uncertainty estimation
reporting-bias decomposition
pathway explanations
```

A complex model can outperform a baseline for many reasons that have nothing to do with its claimed contribution.

Possible confounds include:

```text
more input features
more parameters
history leakage
generic spatial smoothing
random graph connectivity
better optimization
different preprocessing
different train/validation/test splits
different target transformations
greater neural capacity
```

The ablation ladder exists to answer:

> Which specific mechanism earns its complexity, and under what evidence?

The final V2 model is scientifically credible only if its improvements can be attributed to identifiable architectural additions rather than uncontrolled differences.

---

# 2. Core experimental principle

Every major module MUST have:

```text
a simpler control
a named experiment
a stable configuration
a matching hypothesis
predefined evaluation metrics
multiple random seeds
a documented acceptance or rejection criterion
```

A module SHOULD be retained only when it contributes to at least one of:

```text
predictive performance
top-k prioritization
hazard-specific behavior
topology-specific value
temporal retrieval quality
interpretability
stability
uncertainty calibration
scientific understanding
```

A module MUST NOT be retained solely because it is architecturally sophisticated.

---

# 3. What remains fixed across the ladder

To isolate architectural effects, experiments within a comparison block MUST hold the following constant unless the experiment explicitly targets one of them:

```text
dataset version
target definition
forecast horizon
train/validation/test split
temporal origin dates
feature-availability cutoff
static feature set
history window
node set
target mask
target transformation
loss function
optimizer family
training budget
early-stopping rule
evaluation metrics
random-seed set
checkpoint-selection criterion
```

Any intentional difference MUST be declared in the experiment configuration and report.

---

# 4. Required experimental metadata

Every run MUST record:

```text
experiment name
experiment family
model-family version
model-config version
batch-schema version
feature-contract version
relation-registry version
hazard-registry version
prediction-schema version
explanation-schema version
dataset version
graph version
random seed
split identifiers
origin-time range
forecast horizon
feature-availability cutoff
node types
relation families
parameter count
training duration
best epoch
checkpoint ID
run ID
```

This metadata is necessary to distinguish a real architectural comparison from a loosely related rerun.

---

# 5. Evaluation dimensions

No single metric is sufficient.

## 5.1 Predictive accuracy

Required:

```text
MAE
RMSE
```

When appropriate:

```text
mean Poisson deviance
negative binomial deviance
log-likelihood
Huber loss
```

## 5.2 Ranking and prioritization

Required when municipal prioritization is part of the task:

```text
Spearman correlation
Kendall correlation
NDCG@k
top-k overlap
top-decile overlap
precision among highest-risk units
```

## 5.3 Topology-specific evidence

Required comparisons:

```text
no edge
random/placebo edge
centroid kNN
real adjacency
functional relation graph
```

## 5.4 Hazard-conditioned behavior

Possible diagnostics:

```text
relation-gate divergence across hazards
hazard-swap prediction delta
hazard-swap pathway delta
within-node hazard-response consistency
between-seed gate stability
```

## 5.5 Temporal-memory behavior

Possible diagnostics:

```text
lag baseline comparison
temporal-attention stability
hazard-specific memory retrieval
performance by history length
performance under missing history
```

## 5.6 Explanation quality

Possible diagnostics:

```text
stability across random seeds
stability under small input perturbations
agreement with relation removal
agreement with counterfactual edge masking
control-relation suppression
pathway sparsity
pathway consistency by hazard
```

Attention alone is not sufficient evidence.

## 5.7 Uncertainty quality

When uncertainty is active:

```text
coverage probability
interval width
calibration error
quantile loss
error versus predicted uncertainty
selective prediction performance
```

---

# 6. Statistical reporting standard

Each central result SHOULD be reported across multiple random seeds.

Minimum recommendation:

```text
5 seeds for development comparisons
10 or more seeds for central final claims when computationally feasible
```

Reports SHOULD include:

```text
mean
standard deviation
median
minimum and maximum
bootstrap confidence interval
paired run-level differences when configurations share seeds
```

A central architectural claim SHOULD NOT rely on one favorable seed.

When model differences are small, paired bootstrap or permutation analyses SHOULD be used.

---

# 7. Ladder overview

The ablation ladder is divided into stages.

```text
Stage 0 — Data and target sanity
Stage 1 — Non-neural baselines
Stage 2 — Neural capacity without graph structure
Stage 3 — Generic topology controls
Stage 4 — Hazard encoding
Stage 5 — Relation-family gating
Stage 6 — Edge-level attention
Stage 7 — Relation-specific transformations
Stage 8 — Urban memory
Stage 9 — Hazard-queried memory
Stage 10 — Functional relation expansion
Stage 11 — Uncertainty
Stage 12 — Reporting-bias decomposition
Stage 13 — Explanation and counterfactual validation
Stage 14 — Full V2 assembly
```

The ladder is cumulative in scientific logic, not necessarily in code complexity.

---

# 8. Stage 0 — Data, schema, and leakage controls

Before model comparison, the data pipeline MUST pass strict audits.

## 8.1 Required checks

```text
valid external-to-internal node mapping
valid node_batch_index
valid edge endpoints
no cross-graph edges unless explicitly allowed
known relation IDs only
known hazard IDs only
compatible registry versions
finite feature tensors
valid history masks
valid target masks
training-only scaler fitting
consistent target horizon
```

## 8.2 Temporal causality checks

For every supervised example:

```text
history_start_time <= history_end_time
history_end_time <= origin_time
feature_availability_cutoff <= origin_time
target_start_time > origin_time
target_end_time >= target_start_time
```

For dynamic edges:

```text
edge_observation_time <= origin_time
edge_valid_from <= origin_time
origin_time < edge_valid_to
```

when those fields are present.

## 8.3 Required negative controls

At least one deliberately invalid batch SHOULD be used in tests to verify that temporal leakage is rejected.

Examples:

```text
future target value inserted into history
feature availability after origin time
future-observed edge included in past graph
validation-fitted scaler used on training data
```

## 8.4 Stage-completion criterion

No model comparison is considered valid until:

```text
schema tests pass
temporal audits pass
split audits pass
feature-availability audits pass
registry compatibility checks pass
```

---

# 9. Stage 1 — Non-neural benchmark anchors

These models establish what can be achieved without neural message passing.

## A0 — Exposure and seasonality baseline

Possible inputs:

```text
month
season
population
density
basic exposure controls
historical seasonal mean
```

Purpose:

```text
establish a low-complexity operational baseline
```

## A1/B1 — Direct vulnerability-index baseline

Examples:

```text
SVI direct ranking
SoVI direct ranking
```

Purpose:

```text
test whether a static vulnerability index directly aligns with future burden
```

## A2/B2 — Calibrated vulnerability-index predictor

Examples:

```text
linear calibration
isotonic calibration
generalized linear model using the index
```

Purpose:

```text
separate index quality from poor scaling or calibration
```

## A3/B3 — Tabular feature-parity model

Candidate models:

```text
random forest
gradient boosting
XGBoost
MLP
```

Requirements:

```text
same feature set available to graph models
same split
same target
same history information where applicable
```

Purpose:

> Determine whether node features and history already explain the outcome without graph structure.

## Stage-1 claim boundary

A graph model beating only A1/B1 is not sufficient.

The graph model must be compared against the strongest appropriate feature-parity tabular model.

---

# 10. Stage 2 — Neural capacity without graph structure

## N0 — No-edge neural model

Architecture:

```text
input projection
optional memory encoder
optional hazard encoder
prediction head
no inter-node messages
```

Purpose:

> Separate neural capacity from graph value.

Required behavior:

```text
E = 0 is valid
isolated nodes remain valid
no NaN values
prediction alignment is preserved
```

## N1 — Identity/self-loop-only model

Optional variation:

```text
each node communicates only with itself
```

Purpose:

```text
test whether message-passing implementation adds value without cross-node information
```

## Stage-2 decision

Any graph model MUST outperform or meaningfully complement the no-edge neural model before a graph-specific claim is made.

---

# 11. Stage 3 — Generic topology controls

All models in this stage SHOULD use the same node features, memory representation, hidden dimension, prediction head, and training budget.

## G0 — Random/placebo graph

Relation:

```text
random_placebo
```

Possible construction:

```text
same edge count
approximately preserved degree distribution
fixed random seed
rewired real adjacency
```

Purpose:

> Test whether arbitrary connectivity or smoothing improves performance.

## G1 — Centroid kNN graph

Relation:

```text
centroid_knn
```

Purpose:

> Test whether generic geographic proximity is enough.

Required sensitivity:

```text
multiple k values
symmetrized versus directed kNN
distance-weighted versus unweighted
```

## G2 — Real spatial adjacency graph

Relation:

```text
spatial_adjacency
```

Purpose:

> Test whether real contiguity beats no-edge, random-edge, and generic-proximity controls.

## G3 — Generic shared-transform message passing

Architecture:

```text
one transform shared by all relation types
uniform relation weights
uniform or degree-normalized edge weighting
```

Purpose:

```text
establish a generic GNN control before custom functional routing
```

## Stage-3 central comparison

```text
no edge
versus random
versus kNN
versus real adjacency
```

## Topology claim criterion

A strong topology-specific claim requires:

```text
real graph > no-edge neural
real graph > random graph
real graph >= or > kNN graph
```

on predefined primary metrics and across seeds.

If kNN outperforms real adjacency, the correct interpretation may be:

```text
generic proximity is useful
real administrative topology is not yet uniquely supported
```

---

# 12. Stage 4 — Hazard encoding

This stage tests whether explicitly representing the hazard improves prediction or behavior.

## H0 — Hazard-blind model

No hazard ID or scenario features.

Purpose:

```text
establish whether one shared model can predict without explicit hazard context
```

## H1 — Hazard ID embedding

Inputs:

```text
registered hazard ID
```

No dynamic scenario features.

Purpose:

> Test whether categorical hazard identity adds value.

## H2 — Hazard plus season

Inputs:

```text
hazard ID
month
season
forecast horizon
```

Purpose:

```text
test basic hazard-context interaction
```

## H3 — Dynamic scenario encoder

Possible inputs:

```text
precipitation
temperature anomaly
river level
snowmelt proxy
event severity
scenario intensity
```

Purpose:

> Test whether event-specific conditions add value beyond hazard identity.

## Required controls

Parameter count SHOULD be approximately matched where feasible.

Hazard information MUST be available at origin time.

## Stage-4 success criteria

At least one of:

```text
improved prediction
improved ranking
different predictions under valid hazard swaps
stable hazard-specific representation differences
```

Hazard conditioning that produces no measurable behavioral difference is not yet justified.

---

# 13. Stage 5 — Relation-family gating

This stage tests the first major custom mechanism.

## RG0 — No relation gate

All active relation families receive weight 1.

## RG1 — Uniform normalized gate

Every active relation family receives the same fixed weight.

## RG2 — Learned hazard-blind gate

Gate is learned but does not receive hazard context.

Purpose:

```text
separate general relation importance from hazard-specific routing
```

## RG3 — Hazard-conditioned graph-level gate

Shape:

```text
[B, R]
```

One gate vector per graph/scenario.

## RG4 — Hazard-conditioned target-node gate

Shape:

```text
[N, R]
```

Canonical semantics:

> Each target node selects which incoming relation families are relevant under the current hazard.

Edge mapping:

```python
edge_gate = gate_values[
    target_index,
    edge_relation_type,
]
```

## RG5 — Prior-regularized hazard-conditioned gate

Uses versioned hazard-relation priors as:

```text
initial bias
soft regularization
optional feature
```

Priors MUST NOT be hard-coded from validation or test outcomes.

## Required gate diagnostics

```text
gate distribution by hazard
gate distribution by relation
gate entropy or sparsity
between-seed stability
within-hazard consistency
between-hazard divergence
control-relation gate values
```

## Stage-5 success criteria

At least one of:

```text
better predictive or ranking performance
stable hazard-specific gate differences
better relation-removal consistency
improved explanation coherence
```

A gate that changes arbitrarily across seeds is not strong evidence.

---

# 14. Stage 6 — Edge-level attention

This stage tests whether selecting specific neighbors adds value beyond family-level routing.

## EA0 — Uniform edge weights

All incoming edges within the configured normalization scope receive equal weight.

## EA1 — Semantic-weight-only edges

Uses data-provided quantities such as:

```text
distance decay
overlap fraction
shared boundary
service capacity
```

No learned attention.

## EA2 — Hazard-blind learned attention

Uses node states and edge attributes but not hazard context.

## EA3 — Hazard-conditioned edge attention

Uses:

```text
source state
target state
relation type
edge attributes
hazard context
```

## EA4 — Multi-head hazard-conditioned attention

Optional advanced variation.

Required head-reduction policy:

```text
mean
weighted mean
learned fusion
```

must be explicit for explanation exports.

## Normalization ablation

At least two modes SHOULD be tested:

```text
target_node
target_node_and_relation
```

## Required attention diagnostics

```text
attention distribution by relation
attention concentration
attention stability across seeds
attention sensitivity to edge masking
agreement with relation-removal effects
attention assigned to placebo edges
```

## Stage-6 success criteria

Edge attention should provide at least one of:

```text
improved prediction
improved ranking
more stable pathway identification
better edge-removal agreement
meaningful hazard-specific neighbor selection
```

Attention values alone do not justify the module.

---

# 15. Stage 7 — Relation-specific transformations

This stage tests whether different relation families require different transformations.

## RT0 — Shared transform

One transformation for all edges.

## RT1 — Relation embedding modulation

Shared transform conditioned on a relation embedding.

## RT2 — One transform per relation family

Separate transformation for each relation family.

## RT3 — Basis decomposition

Relation transforms are combinations of shared bases.

Purpose:

```text
control parameter growth while allowing relation-specific behavior
```

## RT4 — Low-rank or typed MLP transforms

Optional advanced variants.

## Required controls

Parameter-count differences MUST be reported.

When possible, a capacity-matched shared-transform model SHOULD be included.

## Stage-7 success criteria

Relation-specific transforms are justified when they improve:

```text
performance
stability
relation-removal sensitivity
hazard-relation consistency
```

beyond what can be explained solely by increased parameter count.

---

# 16. Stage 8 — Urban memory

This stage tests how historical stress should be encoded.

## M0 — No memory

Current/static features only.

## M1 — Lag/rolling memory

Examples:

```text
lag_1
rolling_3
rolling_6
rolling_12
seasonal historical mean
```

This is the essential transparent memory baseline.

## M2 — GRU memory

Sequence-based learned history.

## M3 — LSTM memory

Alternative recurrent encoder.

## M4 — Transformer memory

Compact temporal transformer.

## Required memory controls

All variants MUST use the same:

```text
history window
history feature set
feature-availability cutoff
origin times
target horizons
```

## Missing-history sensitivity

Experiments SHOULD evaluate:

```text
full history
shortened history
partially masked history
no history
```

## Stage-8 success criteria

A learned memory encoder is justified only if it improves over lag/rolling memory or provides additional stable interpretability.

If lag/rolling memory performs equally well, it remains the preferred simpler implementation.

---

# 17. Stage 9 — Hazard-queried memory

This stage tests whether the current hazard retrieves different parts of history.

## HQM0 — Generic memory only

Memory state is independent of hazard.

## HQM1 — Post-memory hazard concatenation

Generic memory is computed first, then concatenated with hazard context.

Purpose:

```text
control for simple hazard-memory interaction
```

## HQM2 — Hazard-query temporal attention

Hazard context attends over temporal states.

Output:

```text
hazard-conditioned memory
temporal attention
```

## HQM3 — Hazard-query cross-attention with scenario context

Uses both hazard identity and dynamic scenario conditions.

## Required diagnostics

```text
temporal attention by hazard
same node under different hazards
attention stability across seeds
historical-period masking sensitivity
retrieval agreement with known hazard-specific variables
```

## Hazard-swap test

For the same node and history:

```text
flood query
versus heat query
```

should be evaluated for:

```text
prediction change
temporal-attention change
relation-gate change
pathway change
```

## Stage-9 success criteria

The module is justified when:

```text
hazard queries retrieve measurably different histories
retrieval differences are stable
prediction or ranking improves
counterfactual hazard behavior is coherent
```

---

# 18. Stage 10 — Functional relation expansion

This stage moves beyond generic spatial relations.

## F0 — Spatial controls only

```text
random_placebo
centroid_knn
spatial_adjacency
```

## F1 — Exposure as node features only

Examples:

```text
hydrological exposure
heat exposure
impervious surface
canopy fraction
drainage proxies
```

Purpose:

> Test whether exposure information helps without functional edges.

## F2 — Functional exposure edges

Examples:

```text
hydrological_exposure
flood_zone_exposure
heat_exposure
```

## F3 — Protection relations

Examples:

```text
canopy_protection
cooling_access
```

## F4 — Access relations

Examples:

```text
service_access
road_access
```

## F5 — Dependency relations

Examples:

```text
drainage_dependency
infrastructure_dependency
critical_facility_dependency
```

## F6 — Cross-scale relations

Examples:

```text
cross_scale_parent
cross_scale_child
```

## Critical feature-versus-edge comparison

For each functional relation family, compare:

```text
information represented as node features
versus
information represented as graph edges
```

This is essential.

A functional edge is not justified if the same information works equally well as a simpler feature.

## Stage-10 success criteria

A relation family is retained if it adds:

```text
predictive value
ranking value
stable hazard-specific routing
mechanistically coherent pathway evidence
```

beyond feature-only and spatial controls.

---

# 19. Stage 11 — Node heterogeneity

This stage tests whether distinct node types improve the model.

## HT0 — Urban-unit-only graph

Nodes:

```text
tracts or CDs only
```

## HT1 — Urban units plus one functional node type

Examples:

```text
tract + water body
tract + flood zone
tract + green space
```

## HT2 — Multiple functional node types

Examples:

```text
tract
water body
green space
road
hospital
drainage asset
```

## Required controls

Each heterogeneous addition SHOULD be compared against a node-feature-only representation of the same information.

For example:

```text
distance to nearest water body as tract feature
versus
tract ↔ water-body graph relation
```

## Stage-11 success criteria

Heterogeneous nodes are justified if they provide:

```text
better prediction
better top-k ranking
clearer pathway explanations
stronger transfer across hazards
```

without unstable parameter growth.

---

# 20. Stage 12 — Uncertainty estimation

Uncertainty is evaluated only after the deterministic predictive core is stable.

## U0 — Point prediction only

No uncertainty output.

## U1 — Heteroscedastic variance head

Predicts mean and variance.

## U2 — Quantile regression

Predicts selected quantiles.

## U3 — Monte Carlo dropout

Inference-time stochastic sampling.

## U4 — Deep ensemble

Multiple independently trained models.

## U5 — Conformal interval wrapper

Applied to a fixed predictive model.

## Required uncertainty metrics

```text
coverage probability
average interval width
conditional coverage by hazard
conditional coverage by geography
calibration error
error versus uncertainty
```

## Rare-scenario analysis

Uncertainty SHOULD be evaluated for:

```text
rare hazards
underrepresented geographies
missing histories
unseen scenario intensities
sparse relation neighborhoods
```

## Stage-12 success criteria

An uncertainty module is useful if it identifies difficult or sparse cases and produces reasonably calibrated intervals without excessive width.

---

# 21. Stage 13 — Reporting-bias decomposition

This stage is optional and requires stronger assumptions.

## RB0 — Direct observed-report prediction

Predicts 311 or report burden directly.

## RB1 — Reporting propensity as an auxiliary covariate

Includes reporting-related features but no latent decomposition.

## RB2 — Multi-task reporting head

Predicts:

```text
observed reports
reporting propensity proxy
```

## RB3 — Latent disruption × reporting propensity decomposition

Conceptual form:

```text
observed burden =
latent disruption
× reporting propensity
```

## Required caution

This module MUST NOT be interpreted as recovering true latent disruption unless identification assumptions are defensible.

## Required comparisons

```text
direct observed prediction
auxiliary reporting controls
multi-task reporting prediction
latent decomposition
```

## Stage-13 success criteria

The decomposition must provide evidence beyond improved fit, such as:

```text
better transfer to external event outcomes
better calibration across reporting regimes
stable latent scores under reporting-rate perturbations
meaningful external validation
```

Without such evidence, it remains an exploratory modeling device.

---

# 22. Stage 14 — Explanation validation

Explanation modules require their own controls.

## E0 — Raw attention export

Exports relation gates, edge attention, and temporal attention.

Diagnostic only.

## E1 — Relation removal

For each relation family:

```text
remove or mask relation
measure prediction change
compare with exported gate/pathway score
```

## E2 — Edge masking

Remove highest-attention edges and compare against:

```text
random-edge removal
lowest-attention removal
equal-size relation-matched removal
```

## E3 — Temporal masking

Remove highest-attention history periods and compare against random periods.

## E4 — Hazard swap

Hold node/history fixed and change hazard query.

Measure:

```text
prediction delta
relation-gate delta
edge-attention delta
temporal-attention delta
```

## E5 — Placebo-relation test

Verify that:

```text
random_placebo
```

does not receive consistently dominant explanation scores in the final real-graph model.

## E6 — Explanation stability

Evaluate across:

```text
random seeds
small feature perturbations
bootstrap samples
neighbor perturbations
```

## Explanation acceptance criteria

An explanation signal is stronger when:

```text
high-score component removal causes larger prediction changes
signals are stable across seeds
hazard swaps produce coherent pathway shifts
control relations remain suppressed
```

Attention is not considered validated merely because it is visually plausible.

---

# 23. Full V2 cumulative assembly

The final assembly should not be the first experiment.

A candidate full V2 model may include:

```text
typed input projection
lag or learned urban memory
hazard and scenario encoder
hazard-queried memory
target-node relation-family gates
hazard-conditioned edge attention
relation-specific transforms
functional message passing
prediction head
optional uncertainty
structured pathway export
```

The full model MUST be compared against the strongest retained simpler model from every prior stage.

---

# 24. Canonical named experiments

Suggested names:

```text
a0_exposure_seasonality
a1_direct_vulnerability_index
a2_calibrated_index
a3_tabular_feature_parity

n0_no_edge_neural
n1_self_loop_only

g0_random_placebo
g1_centroid_knn
g2_real_adjacency
g3_generic_shared_transform

h0_hazard_blind
h1_hazard_embedding
h2_hazard_season
h3_dynamic_scenario

rg0_no_gate
rg1_uniform_gate
rg2_hazard_blind_gate
rg3_graph_hazard_gate
rg4_target_node_hazard_gate
rg5_prior_regularized_gate

ea0_uniform_edges
ea1_semantic_edge_weight
ea2_hazard_blind_attention
ea3_hazard_conditioned_attention
ea4_multihead_attention

rt0_shared_transform
rt1_relation_embedding
rt2_relation_specific
rt3_basis_decomposition

m0_no_memory
m1_lag_memory
m2_gru_memory
m3_lstm_memory
m4_transformer_memory

hqm0_generic_memory
hqm1_hazard_concat
hqm2_hazard_queried_memory
hqm3_scenario_queried_memory

f0_spatial_relations_only
f1_exposure_features
f2_exposure_edges
f3_protection_edges
f4_access_edges
f5_dependency_edges
f6_cross_scale_edges

u0_point_prediction
u1_variance_head
u2_quantile_head
u3_mc_dropout
u4_ensemble
u5_conformal

rb0_direct_reports
rb1_reporting_covariates
rb2_reporting_multitask
rb3_latent_reporting_decomposition

full_v2
```

---

# 25. Comparison blocks

Not every experiment should be compared directly with every other experiment.

Use controlled blocks.

## Block A — Baseline strength

```text
A0
A1
A2
A3
N0
```

Question:

```text
How much can features, history, and neural capacity achieve without graph structure?
```

## Block B — Topology value

```text
N0
G0
G1
G2
G3
```

Question:

```text
Does real topology add value beyond no-edge, random, and generic spatial controls?
```

## Block C — Hazard value

```text
H0
H1
H2
H3
```

Question:

```text
Does explicit hazard/scenario context improve prediction or behavior?
```

## Block D — Relation routing

```text
RG0
RG1
RG2
RG3
RG4
RG5
```

Question:

```text
Does hazard-conditioned relation selection matter?
```

## Block E — Edge selection

```text
EA0
EA1
EA2
EA3
EA4
```

Question:

```text
Does learned hazard-specific neighbor selection matter?
```

## Block F — Relation transformations

```text
RT0
RT1
RT2
RT3
```

Question:

```text
Do relation families need distinct transformations?
```

## Block G — Memory

```text
M0
M1
M2
M3
M4
```

Question:

```text
How should urban history be represented?
```

## Block H — Hazard-memory interaction

```text
HQM0
HQM1
HQM2
HQM3
```

Question:

```text
Does the hazard retrieve different historical information?
```

## Block I — Functional graph value

```text
F0
F1
F2
F3
F4
F5
F6
```

Question:

```text
Do functional relations add value beyond features and spatial edges?
```

---

# 26. Acceptance categories

Each module should receive one of four statuses.

## Retain

Evidence is strong across seeds and relevant metrics.

## Retain as optional

The module helps in specific hazards, targets, or explanation analyses but is not universally beneficial.

## Experimental

Evidence is promising but unstable or insufficient.

## Reject for current version

The module does not earn its complexity under current data and experimental conditions.

A rejected module MAY remain in the codebase as an explicitly documented experimental implementation.

---

# 27. Decision criteria

A module SHOULD generally be retained when:

```text
primary metric improves consistently
or top-k prioritization improves meaningfully
or hazard behavior becomes demonstrably more coherent
or explanation faithfulness improves
or uncertainty calibration improves
```

and:

```text
the effect is stable across seeds
the comparison is feature- and split-matched
temporal leakage is ruled out
parameter-count differences are documented
```

A tiny average improvement with high variance SHOULD be treated cautiously.

No universal numeric threshold is imposed in this document because practical significance depends on the target scale and operational use.

Thresholds SHOULD be declared before final experiments.

---

# 28. Complexity accounting

Every experiment report SHOULD include:

```text
trainable parameter count
peak memory usage
training time
inference time
number of graph edges
number of relation families
history length
attention-head count
message-passing depth
```

The preferred model is not automatically the most accurate one.

A simpler model may be preferred when it provides comparable performance with:

```text
lower variance
lower computational cost
better interpretability
easier deployment
```

---

# 29. Parameter-count controls

When comparing architectural mechanisms, capacity differences SHOULD be controlled where possible.

Examples:

```text
shared transform with wider hidden dimension
versus relation-specific transforms

hazard-blind MLP with equal parameter count
versus hazard-conditioned gate

uniform-attention model with additional feed-forward capacity
versus learned edge attention
```

Capacity matching will not always be exact, but parameter differences MUST be reported.

---

# 30. Depth ablation

Message-passing depth SHOULD be tested independently.

Suggested values:

```text
0 layers
1 layer
2 layers
3 layers
```

Purpose:

```text
identify oversmoothing
identify unnecessary propagation depth
separate local from multi-hop effects
```

Required diagnostics MAY include:

```text
node-state similarity
performance by depth
attention/gate stability by layer
gradient magnitude
```

---

# 31. Gate-scope ablation

Because the interface supports multiple scopes, the following comparison SHOULD be considered:

```text
graph-level gate
target-node gate
source-node gate
source-target gate
```

V2.0 default:

```text
target-node gate
```

The semantic interpretation of each scope MUST be documented.

A source-target gate should not be compared as though it had the same complexity as a graph-level gate without reporting parameter differences.

---

# 32. Aggregation ablation

Suggested variants:

```text
sum
mean
degree-normalized sum
relation-wise aggregation then sum
relation-wise aggregation then learned fusion
```

V2.0 SHOULD begin with:

```text
mean
or degree-normalized sum
```

More complex fusion is retained only if justified.

---

# 33. Normalization ablation

Possible variants:

```text
no normalization
target-degree normalization
symmetric normalization
relation-specific degree normalization
semantic distance decay
```

Normalization must remain separate from learned attention and semantic edge weights.

---

# 34. Residual and normalization controls

Suggested variants:

```text
no residual
residual
residual + LayerNorm
pre-normalization
post-normalization
```

These are implementation controls rather than central scientific contributions, but they may affect training stability.

They SHOULD be tuned before drawing negative conclusions about a major research module.

---

# 35. Regularization ablations

Possible terms:

```text
relation-gate sparsity
gate-prior regularization
attention entropy
weight decay
dropout
temporal-attention sparsity
```

Each regularization term MUST have:

```text
a named coefficient
a zero-coefficient control
logged component loss
```

A regularized model must be compared with an otherwise identical unregularized model.

---

# 36. Forecast-horizon ablation

The model SHOULD be evaluated across meaningful horizons when data permits.

Examples:

```text
next month
next 3 months
next 6 months
next season
```

Purpose:

```text
determine whether memory and graph value change with horizon
```

Hypothesis:

```text
short horizons may favor recent memory
longer horizons may favor structural vulnerability and functional relations
```

Horizons MUST remain temporally leakage-free.

---

# 37. Hazard-generalization experiments

Possible evaluations:

## Seen hazards

Train and test on the same hazard families.

## Held-out scenarios

Hold out time periods or scenario intensities.

## Held-out geography

Train on some areas and evaluate on others.

## Held-out hazard

More ambitious:

```text
train on several hazards
evaluate representation transfer to another hazard
```

A held-out hazard claim requires enough shared structure and data to be meaningful.

---

# 38. Geographic robustness

Performance SHOULD be reported by:

```text
urban versus less urban areas
high versus low population
high versus low reporting intensity
high versus low vulnerability
high versus low graph degree
data-rich versus data-sparse regions
```

This can reveal whether graph gains are concentrated in specific geography types.

---

# 39. Control-relation policy

The following relations are diagnostic controls:

```text
identity_no_edge
random_placebo
centroid_knn
```

They MUST be clearly labeled in:

```text
experiment registry
explanation exports
model reports
UI-facing debug artifacts
```

`random_placebo` MUST NOT appear as a real pathway explanation.

`centroid_knn` may be described as generic proximity, not a specific urban mechanism.

---

# 40. Failure interpretations

Negative results should produce precise conclusions.

## Real graph does not beat no-edge

Possible conclusion:

```text
features and history dominate topology under current data
```

## Real graph does not beat random graph

Possible conclusion:

```text
graph smoothing helps, but topology-specific value is not established
```

## Real graph does not beat kNN

Possible conclusion:

```text
generic spatial proximity is as useful as the chosen real topology
```

## Hazard gates do not differ by hazard

Possible conclusion:

```text
hazard conditioning is unused, underidentified, or insufficiently supervised
```

## Learned memory does not beat lag memory

Possible conclusion:

```text
transparent historical summaries are sufficient under current history length
```

## Attention is unstable

Possible conclusion:

```text
edge-level explanation is not yet reliable
```

## Functional edges do not beat feature-only representation

Possible conclusion:

```text
the available information is useful, but its graph representation is not yet justified
```

## B3/A3 remains strongest

Possible conclusion:

```text
tabular features and history currently provide more reliable signal than graph structure
```

These are scientifically meaningful outcomes.

---

# 41. Minimum publishable evidence package

A serious V2 result SHOULD include:

```text
strong tabular feature-parity baseline
no-edge neural baseline
random-edge control
kNN graph
real adjacency graph
hazard-blind versus hazard-conditioned model
no-gate versus gated model
uniform-edge versus edge-attention model
lag-memory versus learned-memory model
seed stability
topology-specific statistical comparison
relation-gate analysis
counterfactual or removal-based explanation validation
```

Uncertainty and reporting-bias modules are optional for the first paper-quality V2 result.

---

# 42. Recommended V2.0 implementation ladder

A practical initial implementation sequence is:

```text
1. A3/B3 tabular feature parity
2. N0 no-edge neural
3. G0 random graph
4. G1 kNN graph
5. G2 real adjacency graph
6. H1 hazard embedding
7. RG4 target-node hazard-conditioned relation gate
8. EA0 uniform edge weights
9. EA3 hazard-conditioned edge attention
10. RT0 shared relation transform
11. RT2 relation-specific transforms
12. M1 lag/rolling memory
13. HQM2 hazard-queried memory
14. F1 functional information as features
15. F2 selected functional relations as edges
16. full_v2
```

This sequence isolates the core scientific contribution without requiring every advanced module immediately.

---

# 43. Recommended V2.0 model candidates

## Candidate C0 — Controlled graph baseline

```text
static features
lag/rolling history
real adjacency
generic message passing
prediction head
```

## Candidate C1 — Hazard-conditioned gates

```text
C0
+ hazard embedding
+ target-node relation-family gate
```

## Candidate C2 — Hazard-conditioned attention

```text
C1
+ edge-level attention
```

## Candidate C3 — Relation-specific functional routing

```text
C2
+ relation-specific transformations
```

## Candidate C4 — Hazard-queried memory

```text
C3
+ temporal states
+ hazard-query retrieval
```

## Candidate C5 — Functional relation expansion

```text
C4
+ selected exposure/protection/dependency relations
```

C5 is the candidate full V2.0 research model.

---

# 44. Experiment registry requirements

Each named experiment SHOULD map to one immutable configuration object.

Example conceptual registry entry:

```yaml
experiment_name: rg4_target_node_hazard_gate
model_family_version: v2.0.0-dev
dataset_version: cd_month_v1
memory:
  type: lag
hazard:
  enabled: true
  scope: graph
relation_gate:
  enabled: true
  scope: target_node
  activation: sigmoid
edge_attention:
  enabled: false
relation_transform:
  type: shared
graph:
  topology: real_adjacency
training:
  seed: 42
```

Changing a material field SHOULD create a different experiment identity or configuration hash.

---

# 45. Required outputs per experiment

Every experiment SHOULD produce:

```text
resolved configuration
checkpoint
training history
validation-selection record
test metrics
node-level predictions
prediction alignment
seed metadata
parameter count
runtime summary
contract versions
```

When applicable:

```text
relation-gate export
edge-attention export
temporal-attention export
counterfactual results
uncertainty results
```

---

# 46. Model selection policy

Hyperparameters and checkpoints MUST be selected using training and validation data only.

The test set MUST NOT determine:

```text
architecture choice
gate scope
relation set
history encoder
number of layers
early-stopping epoch
regularization strength
```

Final test evaluation SHOULD occur after the candidate configuration is frozen.

When extensive iterative development has already touched the test set, the limitation MUST be documented and a fresh holdout or rolling-origin evaluation SHOULD be considered.

---

# 47. Rolling-origin evaluation

For temporal prediction, a single split may be insufficient.

A stronger evaluation MAY use multiple origins:

```text
train through t1 → validate/test after t1
train through t2 → validate/test after t2
train through t3 → validate/test after t3
```

This tests whether gains are stable across time rather than dependent on one period.

All feature-availability rules apply separately at each origin.

---

# 48. Explanation reporting standard

A final explanation report SHOULD include:

```text
gate distributions by hazard
top relation families by hazard
top edge pathways
temporal retrieval examples
hazard-swap examples
relation-removal effects
edge-removal effects
seed stability
placebo-relation diagnostics
```

Qualitative examples SHOULD be accompanied by quantitative aggregate analyses.

Cherry-picked explanation cases are not sufficient.

---

# 49. Uncertainty reporting standard

When uncertainty is enabled, reports SHOULD include:

```text
overall calibration
coverage by hazard
coverage by target magnitude
coverage by geography
interval width
high-uncertainty case analysis
uncertainty under hazard swaps
```

Uncertainty outputs should be compared against simple baselines such as empirical residual intervals.

---

# 50. Stop/go gates

The implementation should advance through explicit gates.

## Gate A — Data valid

Proceed only when:

```text
schema and temporal audits pass
```

## Gate B — Baselines strong

Proceed only when:

```text
feature-parity baselines are implemented and reproducible
```

## Gate C — Graph justified

Proceed to custom routing only when:

```text
graph family shows credible value
or the research goal explicitly tests why it does not
```

## Gate D — Hazard conditioning used

Proceed to hazard-queried memory only when:

```text
hazard encoding produces measurable and stable behavioral differences
```

## Gate E — Gates meaningful

Proceed to complex pathway export only when:

```text
relation gates are stable enough to analyze
```

## Gate F — Attention faithful enough

Present edge attention as an explanation only after masking/removal diagnostics.

## Gate G — Full V2 frozen

Freeze full V2 only after the retained components survive matched ablations.

---

# 51. What success looks like

The strongest result is:

```text
better prediction
+ better top-k municipal prioritization
+ real topology beats no-edge and placebo topology
+ functional relations add value beyond feature-only representations
+ relation gates shift plausibly and stably by hazard
+ hazard queries retrieve different relevant memories
+ edge/pathway explanations agree with removal tests
+ uncertainty identifies difficult scenarios
```

Not every first implementation will satisfy all conditions.

A defensible partial success might be:

```text
no major MAE improvement
but improved top-k prioritization
and stable hazard-specific relation pathways
```

or:

```text
lag memory remains strongest
but functional graph relations improve rare-hazard ranking
```

The claim must match the evidence.

---

# 52. What failure looks like

The full architecture should be considered unsupported under the current setup if:

```text
B3/A3 remains consistently stronger
real graph does not beat random graph
hazard gates do not change by hazard
learned memory does not beat simple lag memory
functional edges do not beat feature-only equivalents
attention and pathway exports are unstable
uncertainty is poorly calibrated
```

This does not invalidate the project.

It identifies which assumptions need better data, better ontology, or a simpler model.

---

# 53. Final ablation principle

The V2 model should not be evaluated as one indivisible object.

It should be reconstructed experimentally:

```text
features
→ neural capacity
→ graph topology
→ hazard context
→ relation-family routing
→ edge selection
→ relation-specific transformation
→ urban memory
→ hazard-queried memory
→ functional relations
→ uncertainty
→ explanation validation
```

At every step, the question is:

> What new capability was added, what simpler control does it beat, and what evidence shows that the model is using it as intended?

That discipline is what turns the architecture from an impressive diagram into a credible research contribution.
