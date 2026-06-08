# Benchmark Contract — Montréal 311 Water/Drainage v0

**Project:** `urban_graph_benchmark`  
**Contract file:** `urban_graph_benchmark/docs/benchmark_contract_mtl_311_v0.md`  
**Status:** Revised draft, not frozen  
**Benchmark ID:** `mtl_311_water_v0`  
**Primary objective:** Build a reproducible Montréal census-tract × month benchmark for testing whether graph-based urban models add predictive and explanatory value beyond SVI-like indices and strong non-graph baselines.

---

## 1. Purpose of this contract

This document defines the first benchmark target for the urban graph / HGNN research program.

The goal is not to immediately build the final heterogeneous graph neural network. The goal is to define a clean empirical arena where several model families can be compared under the same task, spatial unit, temporal unit, train/test splits, and target definition.

The benchmark should answer the first concrete research question:

> Do SVI-like vulnerability features, temporal baselines, simple spatial predictors, adjacency graph models, and eventually heterogeneous environmental graph models improve the prediction and explanation of reported water/drainage disruption in Montréal?

This contract is intentionally conservative. It prioritizes reproducibility, leakage control, scale consistency, and benchmark fairness over model complexity.

---

## 2. Scientific framing

### 2.1 What this benchmark is

This benchmark is a first empirical testbed for the broader research direction:

> Move from static vulnerability mapping toward prediction and explanation of observed urban disruption signals.

The benchmark uses Montréal 311 requests as an observed municipal reporting signal. The first target focuses on water, drainage, sewer, flood, runoff, or stormwater-related service requests.

The benchmark is designed to compare:

```text
SVI-like static vulnerability baseline
temporal / seasonal baselines
calibrated SVI predictor
feature-parity non-graph spatial or tabular model
adjacency-only graph model
later: environmental heterogeneous graph model
```

### 2.2 What this benchmark is not

This benchmark is not yet a full functional-dependency model of the city.

It does not yet claim to model:

```text
actual physical flood depth
complete sewer-network hydraulics
true damage
causal effects
road closure propagation
hospital access disruption
critical infrastructure cascading failure
```

It should be described as:

```text
reported water/drainage disruption prediction
```

not as:

```text
true flood-risk prediction
complete functional urban resilience modeling
```

The stronger functional/cascade benchmark is a later track.

---

## 3. Unit of analysis

### 3.1 Spatial unit

The spatial unit for v0 is:

```text
Montréal census tract
```

The tract-level choice is motivated by:

```text
1. It aligns naturally with the SVI-like Québec tract adaptation.
2. It gives many more spatial observations than census division.
3. It is granular enough for within-Montréal variation.
4. It avoids forcing the CD-level SoVI into an invalid tract-level comparison.
```

### 3.2 Exact study area

The exact Montréal study area must be defined from the 311 coverage.

Possible interpretations:

```text
Ville de Montréal
Montréal agglomeration
Montréal census division / territoire équivalent
all census tracts intersecting the valid 311 service territory
```

**DECISION NEEDED:** after inspecting the 311 dataset, define the official v0 study area.

Rule:

```text
Only census tracts inside the valid 311 service territory should be included.
```

If the 311 dataset only covers Ville de Montréal, do not include tracts from municipalities on the island that are not covered by the same 311 reporting system.

The dataset report must document:

```text
study_area_definition
number_of_tracts_in_scope
number_of_tracts_excluded
reason_for_exclusion
```

### 3.3 Temporal unit

The temporal unit for v0 is:

```text
calendar month
```

The core panel is:

```text
all in-scope Montréal census tracts × all months in the available 311 period
```

A complete zero-filled panel must be built. A tract-month with no relevant 311 request is an explicit zero, not a missing row.

### 3.4 Primary row key

Each row of the benchmark table should be uniquely identified by:

```text
zone_id
period_month
```

Recommended canonical columns:

```text
zone_id
census_tract_dguid
census_tract_name
year
month
period_month
period_start
period_end
```

`zone_id` should be the canonical model key. It should generally equal `census_tract_dguid`.

---

## 4. Spatial-scale rule

The benchmark must not mix incompatible spatial scales.

### 4.1 SVI

The SVI-like index is tract-level and can be used directly in this Montréal tract × month benchmark.

Valid uses:

```text
SVI direct ranking
SVI calibrated predictor
SVI score as a feature
SVI component variables as features
SVI decomposition as an explanation baseline
```

### 4.2 SoVI

The SoVI-like index is census-division-level. Montréal should not be treated as if it had multiple census divisions simply because it has multiple boroughs or arrondissements.

Therefore, SoVI is not a valid within-Montréal tract-level spatial baseline.

Invalid uses:

```text
projecting a CD-level SoVI score onto all Montréal tracts and treating it as spatially informative
comparing tract-level SVI and CD-level SoVI as if they were the same scale
using SoVI to explain within-Montréal tract-level variation
```

Potential valid later uses:

```text
Québec-wide CD-level benchmark
multi-CD event benchmark
hypothesis generation from CD-level factors
coarse contextual layer, clearly labeled as non-discriminating within Montréal
```

---

## 5. Target variable

### 5.1 Target concept

The v0 target is based on Montréal 311 requests related to:

```text
water
drainage
sewer
stormwater
runoff
flooding
catch basins / stormwater infrastructure
backflow / refoulement, if available in categories
```

Working target name:

```text
water_drainage_requests
```

The target should be interpreted as:

```text
reported water/drainage disruption signal
```

not as:

```text
objective true flood damage
```

### 5.2 Target table

The target table should contain at least:

```text
zone_id
year
month
period_month
period_start
period_end
water_drainage_count
water_drainage_binary
water_drainage_magnitude_class_exploratory
total_311_count_all
total_311_count_non_water_drainage
```

### 5.3 Count target

The raw count target is:

```text
water_drainage_count = number of relevant 311 requests in tract u during month t
```

This target supports:

```text
Poisson regression
negative-binomial regression
zero-inflated models
count-based tree / boosting models
ranking metrics
```

### 5.4 Binary target

The binary target is:

```text
water_drainage_binary = 1 if water_drainage_count > 0 else 0
```

This target supports:

```text
logistic regression
binary classification
AUROC
AUPRC
F1
precision@K
```

### 5.5 Magnitude-class target

The ordinal target is:

```text
water_drainage_magnitude_class ∈ {0, 1, 2, 3, 4}
```

Proposed interpretation:

```text
0 = no reported disruption
1 = low reported disruption
2 = moderate reported disruption
3 = high reported disruption
4 = extreme reported disruption
```

Important leakage rule:

```text
The dataset should always store raw counts.
Official magnitude classes must be generated inside the modeling pipeline using training data only.
Any global magnitude class stored in the dataset is exploratory only.
```

Possible thresholding strategies:

```text
Option A:
  class 0 = count == 0
  classes 1–4 = quartiles among positive counts in the training set

Option B:
  class 0 = count == 0
  class 1 = count == 1
  class 2 = count in [2, q50_positive]
  class 3 = count in (q50_positive, q90_positive]
  class 4 = count > q90_positive

Option C:
  domain-informed thresholds after inspection of the training count distribution
```

The selected rule must be saved per split in:

```text
urban_graph_benchmark/outputs/mtl_311_water_v0/datasets/target_thresholds_<split_name>.json
```

---

## 6. Source geometry and spatial assignment

The Montréal 311 data may be point-level, grid-level, or already aggregated. The inventory script must determine the source format before the dataset builder makes assumptions.

### 6.1 If 311 data is point-level

Preferred assignment:

```text
point-in-polygon join from 311 point to census tract
```

The spatial join audit must report:

```text
number of 311 records
number successfully assigned to tract
number outside study area
number with missing or invalid coordinates
CRS used
```

### 6.2 If 311 data is grid25m × month

Preferred assignment depends on available geometry.

If grid polygons exist:

```text
assign grid cell to tract by largest area overlap
```

If only centroids exist:

```text
assign grid cell to tract by centroid-in-polygon
```

The dataset report must document the chosen rule.

Required audit columns:

```text
grid_cell_id
centroid_x
centroid_y
assigned_zone_id
assignment_method
boundary_warning
```

A boundary warning should be raised for grid cells near tract boundaries if the assignment may be ambiguous.

### 6.3 If 311 data is already aggregated

If the data is only borough-level or otherwise too coarse, the tract-month benchmark may not be feasible.

Fallbacks:

```text
borough-month benchmark
different 311 source
different target
```

These fallbacks require a revised contract.

---

## 7. Prediction settings

The benchmark must explicitly distinguish retrospective/explanatory and forecasting settings.

### 7.1 Retrospective / explanatory setting

Purpose:

```text
Explain or model the spatial distribution of reported water/drainage requests using information observed in the same month.
```

Allowed features may include:

```text
same-month rainfall observations
same-month total_311_count_non_water_drainage as reporting-intensity control
same-month citywide request volume
month and year controls
static tract vulnerability features
static environmental features
```

Interpretation:

```text
This setting is useful for retrospective validation and mechanism discovery.
It should not be described as real-time forecasting.
```

### 7.2 Forecasting setting

Purpose:

```text
Predict future reported water/drainage disruption before the target month is fully observed.
```

Allowed features may include:

```text
static tract features
calendar month / season
past water_drainage_count lags
past total_311_count lags
past total_311_count_non_water_drainage lags
historical reporting-intensity estimates
rainfall forecast or antecedent rainfall, if available before prediction time
```

Disallowed features:

```text
same-month water_drainage_count
same-month total_311_count_all
same-month total_311_count_non_water_drainage
same-month features unavailable before prediction time
```

### 7.3 Default v0 setting

For the first dataset build, both settings can be supported by storing enough columns to create separate feature matrices later.

The first model run may be retrospective, but every report must label it explicitly.

Recommended default label:

```text
benchmark_setting = retrospective_explanatory_v0
```

---

## 8. Feature groups

Feature groups should be modular. The benchmark should be able to run models on different feature sets.

### 8.1 Identity columns

```text
zone_id
census_tract_dguid
census_tract_name
municipality_name
borough_name, if available
year
month
period_month
period_start
period_end
```

### 8.2 Static geographic columns

```text
land_area_km2
population_total_2021
population_density
centroid_x
centroid_y
centroid_lon
centroid_lat
```

### 8.3 SVI columns

The v0 benchmark should include:

```text
svi_score
svi_rank
svi_percentile
svi_class
svi_theme_scores, if available
svi_input_variables
```

Exact names should follow the existing SVI output artifacts where possible.

The SVI feature block should allow three modes:

```text
svi_score_only
svi_domains
svi_variables_full
```

### 8.4 Reporting-intensity controls

Potential columns:

```text
total_311_count_all
total_311_count_non_water_drainage
total_311_count_all_lag_1
total_311_count_non_water_drainage_lag_1
total_311_count_non_water_drainage_lag_3_mean
historical_total_311_rate
citywide_total_311_count_month
citywide_total_311_count_non_water_drainage_month
```

Important rule:

```text
Prefer total_311_count_non_water_drainage over total_311_count_all as a reporting-intensity control,
because total_311_count_all mechanically includes the target.
```

Leakage rule:

```text
same-month total_311_count_non_water_drainage is allowed only in retrospective_explanatory mode.
forecasting mode must use lagged or historical controls.
```

### 8.5 Temporal controls

```text
year
month
month_of_year
season
is_winter
is_spring
is_summer
is_fall
time_index
month_sin
month_cos
```

### 8.6 Weather / rainfall controls

These may be unavailable in v0 but should be planned.

Potential columns:

```text
monthly_total_precipitation_mm
max_daily_precipitation_mm
number_of_heavy_rain_days
antecedent_7day_rainfall_mm
antecedent_30day_rainfall_mm
rainfall_station_distance_km
```

Weather features must be labeled by availability:

```text
observed_after_the_fact
known_at_prediction_time
forecast_available
```

### 8.7 Environmental enrichment features

These are not mandatory for the first minimal dataset, but should be planned for v1.

Candidate features:

```text
cuvette_count
cuvette_area_km2
cuvette_area_pct
heat_island_area_pct
canopy_cover_pct
vegetation_index_mean
impervious_surface_pct
mineralized_surface_pct
mean_elevation
min_elevation
max_elevation
mean_slope
distance_to_water_m
overflow_structure_count
distance_to_nearest_overflow_structure_m
```

The first implementation may include placeholders in the config but should not create fake values.

---

## 9. Model ladder

The benchmark should be staged. No graph model should be treated as meaningful until simpler baselines are implemented.

### 9.1 A0 — Naive temporal baselines

Purpose:

```text
Test whether simple seasonality or historical averages already explain much of the target.
```

Candidate baselines:

```text
global monthly mean
tract historical mean
month-of-year mean
previous-month persistence
previous-year same-month value, if enough history exists
```

### 9.2 A1 — SVI direct ranking

Purpose:

```text
Test whether high SVI tracts correspond to high reported water/drainage disruption.
```

Possible evaluation:

```text
Spearman correlation between SVI and target burden
Precision@K for high-burden tracts or tract-months
NDCG@K
top-decile overlap
```

Interpretation limit:

```text
SVI is static and social. It is not expected to fully predict drainage reports.
```

### 9.3 A2 — Calibrated SVI predictor

Purpose:

```text
Test whether SVI has predictive value after calibration and controls.
```

Candidate models:

```text
logistic regression: water_drainage_binary ~ svi_score + controls
negative binomial: water_drainage_count ~ svi_score + controls
ordinal regression: magnitude_class ~ svi_score + controls
```

### 9.4 A3 — Feature-parity non-graph model

Purpose:

```text
Establish the strongest fair non-graph baseline.
```

Candidate models:

```text
negative-binomial / zero-inflated model
random forest
XGBoost / LightGBM, if allowed
MLP
GAM with spatial smooth
spatial regression
```

Feature rule:

```text
Use the same tract-level features that the first graph model uses, but no message passing.
```

### 9.5 A4 — GraphSAGE adjacency-only

Purpose:

```text
Test whether spatial message passing over census-tract adjacency adds value beyond feature-parity non-graph models.
```

Graph:

```text
node type: tract
edge type: tract adjacent_to tract
features: same as A3
target: same as A3
split: same as A3
```

### 9.6 A5 — Environmental HGNN

Purpose:

```text
Test whether typed environmental relations add value beyond adjacency-only graphs.
```

Candidate node types:

```text
tract
cuvette
heat_island_polygon
canopy_or_green_space
overflow_structure
water_body
```

Candidate edge types:

```text
tract adjacent_to tract
tract intersects cuvette
tract overlaps heat_island_polygon
tract overlaps canopy_or_green_space
tract near overflow_structure
tract near water_body
```

This is not part of the minimal Dataset v0, but the dataset should be designed so it can evolve toward this model.

### 9.7 A6 — Random/permuted graph controls

Purpose:

```text
Test whether graph structure is meaningful or merely adds capacity.
```

Controls:

```text
randomized adjacency with same degree distribution
permuted environmental edges
shuffled node features
no-edge model
```

---

## 10. Splits and leakage control

### 10.1 Required split types

The dataset should eventually support:

```text
random tract-month split
temporal split
spatial block split
```

Recommended first split artifacts:

```text
splits_random.json
splits_temporal.json
splits_spatial_block.json
```

### 10.2 Random split

Purpose:

```text
debugging and quick development
```

Risk:

```text
spatial and temporal leakage
```

This split should not be the main scientific evidence.

### 10.3 Temporal split

Example:

```text
train = early years
validation = following period
test = latest period
```

Purpose:

```text
test future generalization
```

### 10.4 Spatial block split

Example:

```text
train on some geographic areas
validate/test on held-out areas
```

Purpose:

```text
test spatial generalization and reduce neighbor leakage
```

Possible block definitions:

```text
borough / arrondissement
spatial clusters
grid blocks
administrative sectors
```

### 10.5 Graph-specific spatial leakage rule

For GraphSAGE and HGNN experiments, spatial holdout requires explicit graph leakage control.

The training pipeline must specify whether the graph setting is:

```text
transductive:
    test nodes exist in the graph during training, but test labels are hidden

inductive:
    held-out test nodes or areas are removed from the training graph
```

The main scientific spatial generalization result should use an inductive or leakage-controlled setting.

The report must document:

```text
whether test nodes were present during message passing
whether train nodes had edges to held-out test nodes
whether target labels or target-derived features could propagate across split boundaries
```

If transductive evaluation is used, it must be labeled clearly and not overstated as strict spatial generalization.

### 10.6 Threshold leakage

Magnitude-class thresholds must be fitted only on the training split.

Preferred v0:

```text
store raw counts always
store one exploratory global magnitude class only if useful for inspection
generate final split-specific classes in modeling code
```

---

## 11. Metrics

### 11.1 Count prediction metrics

```text
MAE
RMSE
mean Poisson deviance
mean negative-binomial log-likelihood, if applicable
Spearman rank correlation
Pearson correlation
```

### 11.2 Binary classification metrics

```text
AUROC
AUPRC
F1
balanced accuracy
precision
recall
Brier score
calibration curve
```

AUPRC is especially important if positive tract-months are rare.

### 11.3 Ordinal / magnitude-class metrics

```text
macro-F1
weighted-F1
quadratic weighted kappa
ordinal cross-entropy
mean absolute class error
confusion matrix
```

### 11.4 Ranking metrics

```text
Precision@K
Recall@K
NDCG@K
top-decile hit rate
Spearman rank correlation
Kendall tau
```

Ranking metrics are important because municipal decision support often cares about identifying priority zones rather than predicting exact counts.

### 11.5 Spatial diagnostics

```text
Moran's I of residuals
error maps
borough-level residual summaries
performance by vulnerability quartile
performance by reporting-intensity quartile
```

---

## 12. Output artifacts

### 12.1 Required dataset outputs

The dataset builder should produce:

```text
urban_graph_benchmark/outputs/mtl_311_water_v0/datasets/tract_month_panel.parquet
urban_graph_benchmark/outputs/mtl_311_water_v0/datasets/tract_static_features.parquet
urban_graph_benchmark/outputs/mtl_311_water_v0/datasets/target_water_drainage.parquet
urban_graph_benchmark/outputs/mtl_311_water_v0/datasets/dataset_validation.json
urban_graph_benchmark/outputs/mtl_311_water_v0/datasets/dataset_report.md
```

### 12.2 Required audit outputs

```text
feature_dictionary.csv
missingness_report.csv
spatial_join_audit.csv
category_keyword_audit.csv
provenance.json
```

The category audit is mandatory for v0 because the water/drainage target depends on selecting the correct 311 categories.

### 12.3 Optional dataset outputs

```text
target_thresholds_exploratory.json
tract_month_panel_preview.csv
```

### 12.4 Later baseline outputs

```text
baselines/svi_direct_ranking_metrics.csv
baselines/calibrated_svi_metrics.csv
baselines/temporal_baseline_metrics.csv
baselines/feature_parity_model_metrics.csv
```

### 12.5 Later graph outputs

```text
graphs/tract_adjacency_edges.parquet
graphs/tract_nodes.parquet
graphs/graph_validation.json
graphs/pyg_data.pt
```

### 12.6 Later explanation outputs

```text
explanations/explanation_cases.parquet
explanations/model_explanations.jsonl
explanations/expert_study_packet/
```

---

## 13. Data provenance and versioning

Every generated benchmark dataset must record provenance.

The provenance file should include:

```text
benchmark_id
generation_timestamp
git_commit_hash, if available
config_path
config_hash
raw_input_paths
raw_input_hashes, if feasible
raw_row_counts
filtered_row_counts
study_area_definition
CRS information
spatial_join_method
target_category_selection_method
SVI source path
SVI source version / run directory
number of tracts
number of months
number of tract-month rows
```

Recommended output:

```text
urban_graph_benchmark/outputs/mtl_311_water_v0/datasets/provenance.json
```

The dataset should not be considered reproducible without this file.

---

## 14. Input inventory requirements

The first script, `01_inventory_inputs.py`, should not build the dataset. It should inspect whether the necessary inputs exist and are usable.

It should check:

```text
Montréal 311 raw file exists
311 file source type: point, grid, or pre-aggregated
311 file has date/month column
311 file has category/type/description columns
311 file has coordinates, grid ID, or spatial reference
311 file time range
water/drainage candidate categories
tract geometry exists
tract geometry CRS
tract geometry has DGUID / tract ID
SVI output exists
SVI output has tract IDs
SVI output has score and variables
tract geometry and SVI can be joined
```

It should write:

```text
urban_graph_benchmark/outputs/mtl_311_water_v0/reports/input_inventory_report.md
urban_graph_benchmark/outputs/mtl_311_water_v0/reports/input_inventory.json
```

The inventory should explicitly mark unknowns instead of guessing.

---

## 15. Target category audit requirements

The target category selection must be auditable.

Prefer official 311 category/type/activity fields over free-text keyword matching.

If keyword matching is used, produce:

```text
included categories
excluded categories
ambiguous categories
keyword hits by field
false-positive notes
random sample of included rows
random sample of excluded near-miss rows
```

Candidate keyword families for inspection:

```text
eau
égout
egout
drainage
inondation
refoulement
ruissellement
puisard
bassin
cuvette
catch basin
water
sewer
flood
```

Warning:

```text
Some keywords such as bassin or cuvette may create false positives.
They must be audited before inclusion.
```

Required output:

```text
urban_graph_benchmark/outputs/mtl_311_water_v0/datasets/category_keyword_audit.csv
```

---

## 16. Dataset validation requirements

The dataset builder must validate:

```text
one row per tract × month
no duplicate tract-month rows
all expected tracts represented in every month
zero-filled missing target rows
target counts are nonnegative integers
date parsing is consistent
spatial assignment success rate is documented
SVI join success rate is documented
no missing zone_id
no unintended SoVI columns in Track A
no future information in forecasting feature set
reporting controls are correctly labeled as same-month, lagged, or historical
```

Validation failures should stop the script unless explicitly marked as warnings in the config.

---

## 17. Reporting bias and interpretation

311 calls are not objective incidents. They are reported service requests.

Reporting can vary with:

```text
language
income
trust in municipal services
awareness of 311
internet/phone access
housing tenure
neighborhood norms
past reporting behavior
density
tourism / commuting
```

The model may learn reporting propensity as well as physical disruption.

Mitigation strategies:

```text
include reporting-intensity controls
prefer non-water 311 controls when controlling reporting intensity
compare water/drainage requests to total non-water 311 activity
evaluate performance by socio-demographic slices
avoid causal language
seek independent validation signals later
```

Recommended language:

```text
The target measures reported municipal water/drainage disruption, not true physical flood occurrence.
```

---

## 18. Relationship to explainability

Explainability should not be an afterthought. The benchmark should preserve enough information to generate explanations later.

For every evaluated case, store:

```text
case_id
zone_id
year
month
observed target
predicted target
model name
feature vector version
split name
```

Later explanation schema:

```text
case_id
model_name
explanation_type
top_features
top_edges
top_neighbor_zones
top_relation_types
text_summary
faithfulness_score
```

### 18.1 Explanation baselines

SVI explanation:

```text
SVI score
theme/domain contribution
top contributing variables
local percentiles
```

Feature-parity model explanation:

```text
SHAP or permutation importance
partial dependence summaries
local feature attribution
```

Graph model explanation:

```text
feature attribution
edge attribution
neighbor influence
subgraph explanation
relation-type ablation
```

### 18.2 Expert evaluation is not a v0 blocker

Expert evaluation is important for the publishable contribution, but it should not block Dataset v0.

The first dataset should simply make future expert explanations possible.

---

## 19. Minimum successful v0

A successful v0 does not require any neural network.

Minimum success is:

```text
1. Montréal tract-month panel exists.
2. Water/drainage 311 target is constructed and validated.
3. Zero-filled tract-month rows are present.
4. SVI features are joined.
5. Temporal/reporting controls are present or explicitly absent.
6. Dataset report documents coverage, missingness, target distribution, and category audit.
7. Provenance is recorded.
8. Splitting strategy is planned or implemented.
```

The first scientific baseline milestone is:

```text
SVI direct ranking
naive temporal baseline
calibrated SVI predictor
feature-parity non-graph model
```

Only after this should the first GraphSAGE model be trained.

---

## 20. Non-goals for v0

Do not include these in v0 unless they are already clean and easy:

```text
all roads
all buildings
full sewer network simulation
full HGNN
sandpile cascade model
expert evaluation study
SoVI tract-level comparison
multi-hazard model
reinforcement learning intervention model
```

These are later-stage research objects.

---

## 21. Open decisions

The following decisions remain open and should be resolved after input inventory.

### 21.1 Montréal study area

Need to define whether v0 covers:

```text
Ville de Montréal
Montréal agglomeration
Montréal CD / territoire équivalent
only census tracts with valid 311 coverage
```

### 21.2 311 category selection

Need to inspect actual Montréal 311 category labels.

Final selection should be stored in config and audited.

### 21.3 Geographic assignment of 311 requests

Need to determine whether 311 data has:

```text
latitude/longitude
grid25m cell ID
grid centroid
address
borough
civic number
pre-assigned district
```

Preferred assignment depends on source type:

```text
point-in-polygon for point-level records
largest area overlap or centroid assignment for grid-level data
```

### 21.4 Time span

Need to inspect the available 311 period.

The number of months determines whether forecasting and previous-year baselines are feasible.

### 21.5 Census tract geometry

Need to confirm whether the repo already contains tract geometry and whether it can be filtered to the valid 311 study area.

### 21.6 SVI output path

Need to identify the canonical SVI tract-level output file and its expected columns.

### 21.7 Rainfall/weather source

Need to decide whether rainfall is included in v0 or postponed to v1.

### 21.8 Magnitude-class thresholds

Need to select a thresholding strategy after inspecting the training target distribution.

### 21.9 Graph evaluation mode

Need to decide, for spatial split experiments, whether graph models are evaluated in:

```text
transductive mode
inductive mode
both modes
```

The main spatial generalization claim should rely on a leakage-controlled setup.

---

## 22. Draft config expectations

The config file:

```text
urban_graph_benchmark/configs/mtl_311_water_v0.yaml
```

should eventually define:

```yaml
benchmark_id: mtl_311_water_v0
status: draft

unit:
  spatial: census_tract
  temporal: month
  study_area: DECISION_NEEDED

target:
  source: montreal_311
  concept: reported_water_drainage_disruption
  count_column: water_drainage_count
  binary_column: water_drainage_binary
  magnitude_column_exploratory: water_drainage_magnitude_class_exploratory
  keyword_families:
    - water
    - drainage
    - sewer
    - flood
    - refoulement
    - egout

spatial_assignment:
  source_geometry_type: DECISION_NEEDED
  method: DECISION_NEEDED

prediction_settings:
  default: retrospective_explanatory_v0
  allow_same_month_total_311_all_in_retrospective: false
  allow_same_month_total_311_non_water_in_retrospective: true
  allow_same_month_total_311_in_forecasting: false

features:
  include_svi: true
  include_reporting_controls: true
  prefer_non_water_reporting_control: true
  include_temporal_controls: true
  include_weather_v0: false
  include_environmental_v0: false

outputs:
  dataset_dir: urban_graph_benchmark/outputs/mtl_311_water_v0/datasets
  report_dir: urban_graph_benchmark/outputs/mtl_311_water_v0/reports
```

This is illustrative. The actual config can evolve.

---

## 23. First implementation order

The recommended first implementation order is:

```text
1. Fill benchmark_contract_mtl_311_v0.md.
2. Fill configs/mtl_311_water_v0.yaml with paths and target keyword candidates.
3. Implement utils/paths.py.
4. Implement utils/io.py.
5. Implement data/inventory.py.
6. Implement scripts/01_inventory_inputs.py.
7. Run the inventory.
8. Revise the contract/config based on actual discovered data.
9. Implement data/build_tract_month_panel.py.
10. Implement data/validate_panel.py.
11. Implement scripts/02_build_dataset_v0.py.
12. Generate the first dataset report.
```

---

## 24. Review checklist before freezing this contract

Before marking this contract as frozen, verify:

```text
actual 311 schema inspected
exact Montréal study area selected
water/drainage categories selected and audited
311 source geometry type known
spatial assignment method selected
tract geometry found and joinable
SVI tract output found and joinable
time range known
zero-filled panel feasible
retrospective vs forecasting feature rules finalized
target threshold strategy selected
first split strategy selected
graph leakage mode specified for graph experiments
output artifact names finalized
provenance fields finalized
```

Until then, this document remains a revised draft.

---

## 25. One-sentence benchmark summary

The v0 benchmark builds a Montréal census tract × month panel to test whether SVI-like vulnerability, temporal baselines, feature-parity spatial models, adjacency GNNs, and later environmental HGNNs can predict and explain reported water/drainage disruption from 311 service requests under a leakage-aware, scale-consistent, reproducible protocol.
