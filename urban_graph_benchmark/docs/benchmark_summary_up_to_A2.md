# Montréal 311 Water/Drainage Benchmark v0 — Work Summary up to A2

Generated summary date: 2026-06-09

This document summarizes the benchmark-building work completed so far for the Montréal 311 Water/Drainage v0 experiment, from input inventory through Dataset v0, split artifacts, A0/A1/A2 baselines, and comparison reports.

The project status at this point is:

```text
Dataset v0                         Done
Leakage-aware split artifacts       Done
Baseline plan                       Done
A0 naive temporal baselines         Done
A1 SVI direct-ranking baseline      Done
A2 calibrated SVI predictors        Done
A0/A1 comparison report             Done
A0/A1/A2 comparison report          Done
```

The current next methodological step after this summary is **A3 feature-parity tabular baselines**, not graph models yet.

---

## 1. Benchmark objective

The benchmark is designed around a tract-month prediction/ranking problem:

```text
Target:
  reported Montréal 311 water/drainage burden

Unit:
  census tract × month

Spatial scope:
  tracts receiving at least one assigned Montréal 311 grid25m unit

Temporal scope:
  2022-01 to 2026-05

Main target column:
  water_drainage_count
```

The target is explicitly **not objective flood occurrence**. It is a municipal reported-service-disruption signal. This distinction matters throughout the benchmark, especially when interpreting SVI performance.

The central research question so far has been:

```text
How much of future reported water/drainage 311 burden can be explained by:

1. naive historical burden,
2. static SVI vulnerability,
3. calibrated SVI with simple controls,
4. same-month reporting-intensity controls?
```

---

## 2. Current main config

We moved from the original config toward a more explicit v0.1 config:

```text
urban_graph_benchmark/configs/mtl_311_water_v0.1.yaml
```

This config encodes the canonical choices discovered through the inventory and debugging process.

Important canonical inputs:

```text
311 grid-month source:
  transformation/output/ville_ia_311_features_grid25m_monthly.parquet

Census tract geometry:
  data/spatial_frame_population_2021/output/
  clean_quebec_census_tract_spatial_frame_with_population_2021.gpkg

SVI scored map output:
  outputs/svi_quebec_2021_partial12_run/
  svi_quebec_2021_partial12_map.csv
```

Important note:

```text
The 311 source is from Ville de Montréal / municipal data,
not Statistics Canada.
```

The census geometry and SVI use Statistics Canada-style tract identifiers, but the 311 signal itself is municipal.

---

## 3. Input inventory phase

### 3.1 Initial inventory problem

The first inventory pass surfaced three important facts:

```text
1. The 311 data was found successfully.
2. Census tract geometry was initially missing from configured paths.
3. The SVI best-candidate selection was too naive.
```

The 311 inventory correctly found the grid25m-month files:

```text
transformation/output/ville_ia_311_features_grid25m_monthly.parquet
transformation/output/ville_ia_311_features_grid25m_monthly.csv
```

and inferred:

```text
source_type = grid25m_month
confidence = high
grid25m_values = yes
```

But tract geometry was initially reported as missing:

```text
census_tract_geometry | 4 candidates | 0 existing
```

SVI candidate selection originally picked a diagnostics file instead of the actual SVI feature/score table.

### 3.2 Updated inventory after adding paths

After extending the candidate paths and recursive search, the inventory found:

```text
Montréal 311 candidates: 4 existing
Census tract geometry candidates: 6 existing
SVI candidates: 19 existing
```

The final inventory summary identified:

```text
Best 311 candidate:
  transformation/requetes311.csv

But preferred canonical benchmark source:
  transformation/output/ville_ia_311_features_grid25m_monthly.parquet

Best tract geometry:
  data/spatial_frame_population_2021/output/
  clean_quebec_census_tract_spatial_frame_with_population_2021.gpkg

Best SVI candidate:
  outputs/svi_quebec_2021_partial12_run/
  svi_quebec_2021_partial12_map.csv
```

The tract geometry ↔ SVI join feasibility section became meaningful:

```text
unit_id ↔ unit_id overlap: 1480
left ratio: 1.000
right ratio: 1.000
```

This confirmed that the tract geometry and SVI output could be joined cleanly at the universe level.

---

## 4. Inspecting the 311 grid25m-month table

We inspected the canonical 311 grid-month parquet:

```bash
python - <<'PY'
import pandas as pd

path = "transformation/output/ville_ia_311_features_grid25m_monthly.parquet"
df = pd.read_parquet(path)
print(df.shape)
print(df.columns.tolist())
print(df.head(5).to_string())
PY
```

Result:

```text
shape = (760621, 29)
```

Important columns:

```text
unit_id
period_month
requests_total
complaints_total
citizen_requests_total
comments_total
urgent_total
finished_total
other_requests
road_mobility_requests
snow_winter_requests
tree_canopy_requests
waste_cleanliness_requests
water_drainage_requests
share_complaints_total
share_urgent_total
share_water_drainage_requests
share_road_mobility_requests
share_tree_canopy_requests
share_snow_winter_requests
share_waste_cleanliness_requests
avg_resolution_delay_hours
median_resolution_delay_hours
unique_activity_count
unique_responsible_units
x_centroid
y_centroid
lat_centroid
lon_centroid
```

This confirmed that the 311 source is already a **grid25m × month aggregate**, not point-level raw records.

---

## 5. Dataset builder: `build_tract_month_panel.py`

Main file:

```text
urban_graph_benchmark/src/ville_hgnn/data/build_tract_month_panel.py
```

Main wrapper:

```text
urban_graph_benchmark/scripts/02_build_dataset_v0.py
```

Main output folder:

```text
urban_graph_benchmark/outputs/mtl_311_water_v0/datasets/
```

### 5.1 First major failure: no grid rows assigned to tracts

The first run failed with:

```text
DatasetBuildError: No 311 grid-month rows could be assigned to census tracts.
```

We diagnosed the issue by comparing tract bounds and grid coordinate bounds:

```text
tract CRS: EPSG:3347
tract bounds:
  [7412545.56857147 1182721.63142861 7791634.08571433 1648185.32000003]

grid x/y bounds:
  x around 267475 to 306409
  y around 5,029,293 to 5,062,642

grid lon/lat transformed to tract CRS bounds:
  [7603100.30697657 1225372.35063217 7636429.91411634 1268339.1661192]
```

Conclusion:

```text
x_centroid / y_centroid were not in the same CRS as tract geometry.
lon_centroid / lat_centroid were valid EPSG:4326 and should be used instead.
```

Patch:

```text
Use lon_centroid / lat_centroid as EPSG:4326,
then reproject points to tract CRS EPSG:3347 before spatial join.
```

This fixed the spatial assignment issue.

### 5.2 First successful Dataset v0 build

After fixing the coordinate CRS issue, the dataset builder produced:

```text
Dataset v0 build completed.
Status: warning initially, then pass after SVI join patch
Rows: 28620
Zones: 540
Months: 53
Period: 2022-01 to 2026-05
Grid assignment success: about 0.9995
SVI join success: eventually 1.0
```

Output artifacts:

```text
tract_month_panel.parquet
tract_static_features.parquet
target_water_drainage.parquet
dataset_validation.json
dataset_report.md
spatial_join_audit.csv
missingness_report.csv
feature_dictionary.csv
provenance.json
```

### 5.3 SVI join issue: DGUID mismatch

Initial SVI join success was low:

```text
SVI join success: 0.361111...
```

We found the issue by checking identifiers:

```python
static_ids = set(static["census_tract_dguid"].astype(str))
svi_ids = set(svi["statcan_dguid"].astype(str))
```

Result:

```text
static DGUID count: 540
svi DGUID count: 1480
DGUID overlap: 0
static examples:
  4620001.00, 4620002.00, ...
```

The field called `census_tract_dguid` in the static panel was effectively the numeric tract identifier, not a full StatCan DGUID string.

Then we normalized `zone_id` and `unit_id`:

```python
def norm_unit_id(x):
    if pd.isna(x):
        return None
    s = str(x).strip()
    try:
        return f"{float(s):.2f}"
    except Exception:
        return s
```

Result:

```text
static normalized count: 540
svi normalized count: 1480
normalized overlap: 540
static not in svi examples: []
```

Patch:

```text
Join SVI using normalized tract unit IDs rather than full DGUID strings.
```

After the patch:

```text
SVI join success: 1.0
```

### 5.4 Multi-coordinate grid unit issue

The dataset report initially showed:

```text
units_with_multiple_coordinate_rows = 17726
```

This raised concern because the first implementation assigned each `unit_id` once after dropping duplicate coordinate rows.

We audited coordinate variability:

```text
distinct coordinate rows: 84351
unique grid units: 42752
units with >1 coordinate row: 17726
units assigned to >1 tract: 538
```

Coordinate variation summary:

```text
mean max_axis_range_m: 0.58 m
75%: 0.03 m
max: 28.70 m
```

The largest coordinate variation was around 25–29 meters, roughly the size of a grid cell. Some grid units near tract boundaries could be assigned to different tracts depending on the coordinate row.

Impact of multi-coordinate unit IDs:

```text
rows affected by multi-coordinate unit_ids: 479570
share rows affected: 0.6305
requests_total affected: 1109709
share requests_total affected: 0.7067
water_drainage affected: 101625
share water_drainage affected: 0.6756
```

Conclusion:

```text
Even if most coordinate variations were small, the affected rows represented
a very large share of the data. Dropping to one coordinate per unit_id was too crude.
```

Patch:

```text
Assign coordinate rows, not just unit_id rows.
Create a coordinate_row_id for each distinct unit_id/lon/lat coordinate row.
Spatially join each coordinate row to a tract.
Merge original grid-month rows back to assignment via coordinate_row_id.
```

After this patch:

```text
Assignment audit rows: 83829
assigned rows: 83805
unassigned rows: 24
coordinate_row_id unique: 83829

source water total: 150425
panel water total: 150422
difference: 3

source requests total: 1570335
panel requests total: 1570303
difference: 32
```

So the final dataset loses only:

```text
3 water/drainage requests
32 total 311 requests
```

due to unassigned coordinate rows, which is negligible.

### 5.5 Final Dataset v0 report

Final dataset status:

```text
Validation status: pass
n_zones: 540
n_months: 53
expected_rows: 28620
actual_rows: 28620
zero_filled_tract_month_rows: 1549
period_month_min: 2022-01
period_month_max: 2026-05
```

Spatial assignment:

```text
total_unique_grid_units: 42752
assigned_unique_grid_units: 42731
unassigned_unique_grid_units: 21
assignment_success_rate_unique_grid_units: 0.9995087949101796

total_unique_coordinate_rows: 83829
assigned_unique_coordinate_rows: 83805
unassigned_unique_coordinate_rows: 24
assignment_success_rate_coordinate_rows: 0.9997137028951795

coordinate_source: lon_lat_centroid_epsg4326_to_tract_crs
coordinate_source_crs: EPSG:4326
coordinate_target_crs: EPSG:3347
spatial_join_method:
  grid_coordinate_row_centroid_in_polygon_using_lon_lat_epsg4326_reprojected_to_tract_crs
```

SVI join:

```text
static_tracts_in_scope: 540
svi_rows: 1480
matched_static_tracts: 540
missing_svi_rows: 0
svi_join_success_rate: 1.0
```

Target summary:

```text
panel shape: (28620, 65)
zones: 540
months: 53

water_drainage_count:
  mean: 5.2558
  std: 5.1764
  min: 0
  25%: 2
  50%: 4
  75%: 7
  max: 71

positive rate: about 0.849
total water/drainage requests: 150422
```

Interpretation:

```text
The target is dense: about 85% of tract-months have at least one water/drainage request.
So binary classification is not the main scientific target.
The better targets are count prediction, burden ranking, and ordinal/magnitude prediction.
```

---

## 6. Dataset wrapper: `02_build_dataset_v0.py`

We added a thin wrapper:

```text
urban_graph_benchmark/scripts/02_build_dataset_v0.py
```

Purpose:

```text
Only parse CLI arguments,
bootstrap the package path,
call run_build_dataset(),
print a summary,
and print output paths.
```

No dataset-building logic is inside the wrapper.

Command:

```bash
python urban_graph_benchmark/scripts/02_build_dataset_v0.py \
  --config urban_graph_benchmark/configs/mtl_311_water_v0.1.yaml
```

The wrapper ran successfully and provenance confirmed:

```text
config path:
  urban_graph_benchmark/configs/mtl_311_water_v0.1.yaml

validation status:
  pass

panel shape:
  (28620, 65)

SVI missing rows:
  0
```

---

## 7. Split artifacts: `build_splits.py`

Main file:

```text
urban_graph_benchmark/src/ville_hgnn/data/build_splits.py
```

Wrapper:

```text
urban_graph_benchmark/scripts/03_build_splits_v0.py
```

Main output folder:

```text
urban_graph_benchmark/outputs/mtl_311_water_v0/datasets/splits/
```

### 7.1 Split strategy

We created leakage-aware split artifacts before running any baseline.

The primary scientific split is temporal:

```text
train:
  2022-01 to 2024-12
  36 months
  19440 rows

validation:
  2025-01 to 2025-08
  8 months
  4320 rows

test:
  2025-09 to 2026-05
  9 months
  4860 rows
```

Other splits:

```text
random_debug_split:
  debugging only, not scientific evidence

spatial_block_split:
  preliminary spatial split by tract centroid quantile-grid blocks
  graph-specific leakage handling still later
```

### 7.2 Split outputs

The split builder produced:

```text
split_assignments.parquet
split_metadata.json
split_report.md
split_validation.json
target_thresholds_temporal.json
target_thresholds_random_debug.json
target_thresholds_spatial_block.json
```

Command:

```bash
PYTHONPATH=urban_graph_benchmark/src \
python urban_graph_benchmark/src/ville_hgnn/data/build_splits.py \
  --config urban_graph_benchmark/configs/mtl_311_water_v0.1.yaml
```

Output:

```text
Split artifacts built.
Status: pass
Rows: 28620
Zones: 540
Months: 53
Period: 2022-01 to 2026-05

Temporal rows:
  train: 19440
  validation: 4320
  test: 4860

Random-debug rows:
  train: 20034
  validation: 4293
  test: 4293

Spatial block available: True
```

### 7.3 Magnitude thresholds

Magnitude classes are train-only and split-specific.

For the temporal split:

```text
class_0_rule: y == 0
class_1_max: 3.0
class_2_max: 5.0
class_3_max: 8.0
class_4_rule: >8.0
```

Validation confirmed:

```text
class 0 nonzero rows: 0
zero rows not class 0: 0
```

Train magnitude class distribution:

```text
0: 2848
1: 5689
2: 3576
3: 3514
4: 3813
```

---

## 8. Baseline plan

We created a research-level baseline plan:

```text
urban_graph_benchmark/docs/baseline_plan_mtl_311_v0.md
```

Core staged design:

```text
A0 naive temporal/exposure baselines
A1 SVI direct-ranking baseline
A2 calibrated SVI predictor
A3 feature-parity tabular model
GraphSAGE / HGNN later
```

The baseline plan emphasized:

```text
1. Do not jump to graph models too early.
2. Establish leakage-aware splits first.
3. Stabilize metric schemas and report format.
4. Compare graph models against strong non-graph baselines, not weak baselines.
5. Separate forecasting from retrospective explanatory models.
```

---

## 9. Shared evaluation layer: `metrics.py`

Main file:

```text
urban_graph_benchmark/src/ville_hgnn/evaluation/metrics.py
```

Purpose:

```text
Provide model-agnostic metrics reused by A0, A1, A2, A3, and graph models.
```

Metric groups:

```text
count metrics:
  MAE
  RMSE
  mean Poisson deviance
  bias / mean error
  mean observed / predicted

ranking metrics:
  Spearman correlation
  Kendall correlation
  NDCG@K
  top-K overlap
  top-fraction overlap

binary diagnostics:
  if positive vs zero is needed
```

Important schema choices:

```text
n_rows is used, not n_eval.
top_10pct_overlap_rate is used, not top_10pct_overlap_precision.
```

This naming was standardized across A0, A1, and A2.

---

## 10. Shared baseline utilities: `common.py`

Main file:

```text
urban_graph_benchmark/src/ville_hgnn/baselines/common.py
```

Purpose:

```text
Load config and benchmark frame,
merge panel with split artifacts,
resolve paths,
build run contexts,
evaluate prediction frames,
enforce basic leakage guards,
and standardize output paths.
```

Important final API:

```python
config, root, resolved_config_path, panel_path, split_path, frame = load_benchmark_frame(...)
```

We moved A0, A1, and A2 to this API instead of older patterns like:

```python
load_config_and_inputs()
merge_panel_with_splits()
```

This kept all baseline scripts consistent.

---

## 11. A0 naive temporal/exposure baselines

Main file:

```text
urban_graph_benchmark/src/ville_hgnn/baselines/a0_naive_temporal.py
```

Wrapper:

```text
Used directly via module execution and later through:
urban_graph_benchmark/scripts/04_run_a0_a1_baselines.py
```

Output folder:

```text
urban_graph_benchmark/outputs/mtl_311_water_v0/baselines/A0_naive_temporal/
```

### 11.1 A0 implemented baselines

A0 implemented 8 baselines:

```text
A0_1_global_train_mean
A0_2_month_of_year_train_mean
A0_3_tract_train_mean
A0_4_tract_month_of_year_train_mean
A0_5_previous_month_persistence
A0_6_previous_year_same_month_persistence
A0_7_population_exposure_train_rate
A0_8_non_water_311_reporting_exposure_retrospective
```

Methodological notes:

```text
A0_1 to A0_4:
  train-only means

A0_5 and A0_6:
  strictly past observed-history baselines

A0_7:
  population exposure train-rate baseline

A0_8:
  same-month non-water 311 reporting exposure
  retrospective/explanatory only, not strict forecasting
```

### 11.2 A0 run

Command:

```bash
PYTHONPATH=urban_graph_benchmark/src \
python -m ville_hgnn.baselines.a0_naive_temporal \
  --config urban_graph_benchmark/configs/mtl_311_water_v0.1.yaml \
  --split-scheme temporal
```

Output:

```text
A0 naive temporal baselines completed.
Status: completed
Split scheme: temporal
Models: 8
Metric rows: 480
Prediction rows: 228960
```

The prediction row count checks out:

```text
28620 panel rows × 8 models = 228960
```

### 11.3 A0 main result

Strongest A0 test model:

```text
A0_3_tract_train_mean
```

Temporal test metrics:

```text
MAE: 2.489209
RMSE: 3.462372
Poisson deviance: 2.143367
Spearman: 0.694235
NDCG@100: 0.748363
Top-10% overlap: 0.483539
```

Interpretation:

```text
Historical tract-level burden is extremely strong.
Some tracts consistently report much more water/drainage 311 burden than others.
Any future model must beat A0_3, not just global mean.
```

Warnings:

```text
ConstantInputWarning for Spearman correlation
```

This is expected for the global mean baseline, because constant predictions have undefined rank correlation.

---

## 12. A1 SVI direct-ranking baseline

Main file:

```text
urban_graph_benchmark/src/ville_hgnn/baselines/a1_svi_direct_ranking.py
```

Output folder:

```text
urban_graph_benchmark/outputs/mtl_311_water_v0/baselines/A1_svi_direct_ranking/
```

### 12.1 A1 purpose

A1 asks:

```text
Does static SVI rank the tracts/months that later have high reported water/drainage 311 burden?
```

A1 is not a count predictor. It is a direct ranking baseline.

Evaluation views:

```text
1. tract-month ranking:
   repeat static SVI score across months

2. tract-level total burden ranking:
   aggregate validation/test burden per tract

3. tract-level mean monthly burden ranking

4. burden rate per 1,000 population when available

5. top-K and top-fraction overlap diagnostics
```

### 12.2 SVI columns

The SVI map output had columns including:

```text
svi_percentile
svi_score_raw
svi_rank
svi_class
svi_scored
svi_quality_flag
svi_missing_count
svi_reproduction_level
```

We initially included `svi_scored` by mistake.

Audit showed:

```text
svi_scored min: 0
svi_scored max: 1
svi_scored mean: about 0.987
```

Conclusion:

```text
svi_scored is a boolean-like scoring-status indicator,
not a vulnerability score.
```

Patch:

```text
Exclude svi_scored and other metadata/status columns.
```

Final A1 score columns:

```text
Primary continuous:
  svi_percentile
  svi_score_raw

Diagnostic:
  svi_rank
  svi_class
```

Important interpretation:

```text
svi_class is ordinal and diagnostic only.
It should not be treated as the primary SVI result.
```

### 12.3 A1 run

Command:

```bash
PYTHONPATH=urban_graph_benchmark/src \
python -m ville_hgnn.baselines.a1_svi_direct_ranking \
  --config urban_graph_benchmark/configs/mtl_311_water_v0.1.yaml \
  --split-scheme temporal
```

Final output after excluding `svi_scored`:

```text
A1 SVI direct-ranking baseline completed.
Status: completed
Split scheme: temporal
SVI score columns: 4
Primary SVI columns: ['svi_percentile', 'svi_score_raw']
Diagnostic SVI columns: ['svi_rank', 'svi_class']
Metric rows: 544
Prediction rows: 114480
Tract ranking rows: 4246
Top-K rows: 120
```

Prediction row count:

```text
28620 panel rows × 4 SVI scores = 114480
```

### 12.4 A1 result

Primary continuous SVI on temporal test:

```text
A1_svi_direct_ranking__svi_percentile

Spearman: 0.160639
NDCG@100: 0.220560
Top-10% overlap: 0.052411
```

A1 raw SVI score:

```text
A1_svi_direct_ranking__svi_score_raw

Spearman: 0.160630
NDCG@100: 0.220560
Top-10% overlap: 0.052411
```

Diagnostic class:

```text
A1_svi_direct_ranking__svi_class__ordinal_class_diagnostic

Spearman: 0.185642
NDCG@100: 0.222316
Top-10% overlap: 0.150943
```

Interpretation:

```text
SVI has a positive but weak direct-ranking relationship with reported water/drainage 311 burden.
It does not recover the highest-burden tracts/months well.
The class label performs slightly better in top-10% overlap, but remains diagnostic.
```

---

## 13. A0/A1 wrapper

Wrapper:

```text
urban_graph_benchmark/scripts/04_run_a0_a1_baselines.py
```

Purpose:

```text
Run A0 and A1 sequentially from one command.
```

No methodology is implemented in the wrapper.

Command:

```bash
python urban_graph_benchmark/scripts/04_run_a0_a1_baselines.py \
  --config urban_graph_benchmark/configs/mtl_311_water_v0.1.yaml \
  --split-scheme temporal
```

Output confirmed:

```text
A0:
  models: 8
  metric rows: 480
  prediction rows: 228960

A1:
  score columns: 4
  primary: svi_percentile, svi_score_raw
  diagnostic: svi_rank, svi_class
  metric rows: 544
  prediction rows: 114480
```

---

## 14. A0/A1 comparison report

Script:

```text
urban_graph_benchmark/scripts/05_compare_a0_a1_results.py
```

Output folder:

```text
urban_graph_benchmark/outputs/mtl_311_water_v0/baselines/A0_A1_comparison/
```

Outputs:

```text
a0_a1_metrics_long.csv
a0_a1_tract_month_ranking_wide.csv
a0_a1_test_headline_table.csv
a0_a1_comparison_report.md
comparison_metadata.json
```

Core comparison:

```text
A0_3_tract_train_mean vs A1_svi_percentile
```

Temporal test:

```text
A0_3:
  Spearman: 0.694235
  NDCG@100: 0.748363
  Top-10% overlap: 0.483539

A1_svi_percentile:
  Spearman: 0.160639
  NDCG@100: 0.220560
  Top-10% overlap: 0.052411
```

Conclusion:

```text
Historical tract burden dominates raw SVI direct ranking.
SVI is a weak but positive vulnerability-prior signal.
```

---

## 15. A2 calibrated SVI predictors

Main file:

```text
urban_graph_benchmark/src/ville_hgnn/baselines/a2_calibrated_svi.py
```

Output folder:

```text
urban_graph_benchmark/outputs/mtl_311_water_v0/baselines/A2_calibrated_svi/
```

### 15.1 A2 purpose

A2 asks:

```text
Does SVI become more useful after train-only calibration and basic controls?
```

This is closer to regression-style validation in the literature than A1 raw ranking.

A2 is still not full tabular ML. It is a calibrated SVI baseline.

### 15.2 A2 model family

A2 uses:

```text
ridge-regularized log-count linear model
```

Target transform:

```text
log1p(water_drainage_count)
```

Inverse transform:

```text
expm1(predicted_log_count), clipped at zero
```

Implementation details:

```text
train-only median imputation
train-only standardization
ridge penalty with unpenalized intercept
coefficient export
prediction parquet outputs
metrics/report/metadata
actual ridge_alpha stored in metadata
```

Default ridge penalty:

```text
ridge_alpha = 1.0
```

### 15.3 A2 feature sets

For each SVI score column, A2 fits:

```text
A2_svi_only
A2_svi_plus_calendar
A2_svi_plus_static
A2_svi_plus_reporting_retrospective
```

Feature set details:

```text
A2_svi_only:
  SVI score only

A2_svi_plus_calendar:
  SVI score + month-of-year dummies

A2_svi_plus_static:
  SVI score + month-of-year + static controls
  such as population, land area, density when available

A2_svi_plus_reporting_retrospective:
  SVI score + calendar/static controls
  + same-month total_311_count_non_water_drainage
  retrospective/explanatory only
```

Important leakage distinction:

```text
A2_svi_plus_reporting_retrospective is not a strict forecasting model,
because it uses same-month non-water 311 activity.
```

### 15.4 A2 run

Command:

```bash
PYTHONPATH=urban_graph_benchmark/src \
python -m ville_hgnn.baselines.a2_calibrated_svi \
  --config urban_graph_benchmark/configs/mtl_311_water_v0.1.yaml \
  --split-scheme temporal
```

Output:

```text
A2 calibrated SVI predictors completed.
Status: completed
Split scheme: temporal
SVI score columns available: 4
SVI score columns used: 4
Primary only: False
Feature sets: 4
Models: 16
Metric rows: 960
Prediction rows: 457920
```

Prediction row count:

```text
28620 panel rows × 16 models = 457920
```

Outputs:

```text
metrics.csv
model_metadata.json
baseline_report.md
coefficients.csv
svi_score_audit.csv
svi_static_score_audit.csv
feature_set_audit.csv
predictions_validation.parquet
predictions_test.parquet
```

### 15.5 A2 result

Core temporal-test comparison:

```text
A2_svi_only__svi_percentile:
  MAE: 3.475547
  RMSE: 4.938114
  Poisson deviance: 4.801117
  Spearman: 0.159008
  NDCG@100: 0.220560
  Top-10% overlap: 0.057613

A2_svi_plus_static__svi_percentile:
  MAE: 3.434751
  RMSE: 4.936054
  Poisson deviance: 4.952062
  Spearman: 0.236370
  NDCG@100: 0.193140
  Top-10% overlap: 0.121399

A2_svi_plus_reporting_retrospective__svi_percentile:
  MAE: 2.522434
  RMSE: 3.663199
  Poisson deviance: 2.522685
  Spearman: 0.671736
  NDCG@100: 0.702508
  Top-10% overlap: 0.462963
```

Best strict forecasting A2 model on test:

```text
A2_svi_plus_static__svi_class__diagnostic_svi

MAE: 3.430960
Spearman: 0.240694
Top-10% overlap: 0.133745
```

But because this uses `svi_class`, it is diagnostic. For primary continuous SVI, the key strict model is:

```text
A2_svi_plus_static__svi_percentile
```

Interpretation:

```text
A2_svi_plus_static improves over raw A1 ranking:
  Spearman: 0.1606 → 0.2364
  Top-10% overlap: 0.0524 → 0.1214

But it remains much weaker than A0_3:
  A0_3 Spearman: 0.6942
  A0_3 Top-10% overlap: 0.4835
```

The retrospective reporting-control model is very strong:

```text
A2_svi_plus_reporting_retrospective__svi_percentile:
  MAE: 2.5224
  Spearman: 0.6717
  Top-10% overlap: 0.4630
```

But it is not a strict forecasting baseline.

Interpretation:

```text
Same-month non-water 311 reporting activity explains a large amount of water/drainage 311 burden.
This likely captures reporting intensity, municipal activity, population/activity density,
and neighborhood/service-contact patterns.
```

---

## 16. A0/A1/A2 comparison report

Script:

```text
urban_graph_benchmark/scripts/06_compare_a0_a1_a2_results.py
```

Output folder:

```text
urban_graph_benchmark/outputs/mtl_311_water_v0/baselines/A0_A1_A2_comparison/
```

Outputs:

```text
a0_a1_a2_metrics_long.csv
a0_a1_a2_test_headline_table.csv
a0_a1_a2_core_model_table.csv
a0_a1_a2_tract_month_ranking_wide.csv
a0_a1_a2_best_by_stage.csv
a0_a1_a2_comparison_report.md
comparison_metadata.json
```

### 16.1 Core temporal-test comparison

```text
A0 history:
  model: A0_3_tract_train_mean
  MAE: 2.489209
  RMSE: 3.462372
  Poisson deviance: 2.143367
  Spearman: 0.694235
  NDCG@100: 0.748363
  Top-10% overlap: 0.483539

A1 raw SVI percentile:
  MAE: NA
  RMSE: NA
  Poisson deviance: NA
  Spearman: 0.160639
  NDCG@100: 0.220560
  Top-10% overlap: 0.052411

A1 raw SVI score:
  Spearman: 0.160630
  NDCG@100: 0.220560
  Top-10% overlap: 0.052411

A2 SVI-only percentile:
  MAE: 3.475547
  RMSE: 4.938114
  Poisson deviance: 4.801117
  Spearman: 0.159008
  NDCG@100: 0.220560
  Top-10% overlap: 0.057613

A2 SVI + static percentile:
  MAE: 3.434751
  RMSE: 4.936054
  Poisson deviance: 4.952062
  Spearman: 0.236370
  NDCG@100: 0.193140
  Top-10% overlap: 0.121399

A2 SVI + reporting retrospective:
  MAE: 2.522434
  RMSE: 3.663199
  Poisson deviance: 2.522685
  Spearman: 0.671736
  NDCG@100: 0.702508
  Top-10% overlap: 0.462963
```

### 16.2 Main scientific conclusion

The benchmark story up to A2 is:

```text
1. A0 tract-history is the strongest strict forecasting baseline.

2. A1 raw SVI is positive but weak as a standalone ranking signal.

3. A2 calibrated SVI with static controls improves over raw SVI,
   but remains far below A0 tract-history.

4. A2 with same-month non-water 311 reporting is strong,
   but retrospective/explanatory only.

5. Therefore, before graph models, we need A3 feature-parity tabular baselines
   to test whether non-graph ML can beat A0 and A2.
```

The most compact statement:

```text
Historical tract-level 311 burden is much more predictive of future water/drainage
311 burden than raw or calibrated SVI. SVI contributes a weak-to-modest vulnerability
signal, especially after calibration/static controls, while same-month non-water 311
activity captures strong retrospective reporting intensity.
```

---

## 17. Literature comparison: Flanagan-style SVI validation

We compared our SVI results with a validation paper using Flanagan-style SVI.

Important caveat:

```text
The paper's 0.0893 is a regression coefficient.
Our 0.16–0.24 values are rank correlations.
Our 0.42/0.22 values are NDCG.
Our 0.05/0.12 values are top-10% overlap.

These are not directly comparable scales.
```

Correct comparison:

```text
The paper found SVI statistically significant in expected direction for some broad disaster outcomes.
Our benchmark finds SVI positively associated but weak as a direct raw/calibrated predictor of Montréal 311 water/drainage burden.
```

This is not contradictory because:

```text
The paper evaluated broad disaster damages/fatalities with regression controls.
We evaluate a narrow municipal reporting proxy using direct ranking and count prediction.
```

Our result is plausible, not surprising:

```text
311 burden depends on infrastructure, reporting behavior, municipal activity,
drainage conditions, population/activity density, service access, and social vulnerability.
SVI alone should not be expected to dominate.
```

---

## 18. What we should not claim

Avoid claiming:

```text
SVI failed.
```

Better:

```text
Raw SVI is weak as a standalone direct-ranking baseline for this target.
Calibrated SVI with static controls improves but remains weaker than historical tract burden.
```

Avoid claiming:

```text
A2 reporting-control model is a forecasting model.
```

Better:

```text
A2 reporting-control model is retrospective/explanatory because it uses same-month non-water 311 activity.
```

Avoid claiming:

```text
Graph models should be built immediately.
```

Better:

```text
A3 feature-parity tabular baselines are needed before graph models,
so we can distinguish graph-structure gains from ordinary tabular/feature gains.
```

---

## 19. Current file inventory

### Configs

```text
urban_graph_benchmark/configs/mtl_311_water_v0.1.yaml
```

### Dataset and split modules

```text
urban_graph_benchmark/src/ville_hgnn/data/build_tract_month_panel.py
urban_graph_benchmark/src/ville_hgnn/data/build_splits.py
```

### Evaluation and baseline utilities

```text
urban_graph_benchmark/src/ville_hgnn/evaluation/metrics.py
urban_graph_benchmark/src/ville_hgnn/baselines/common.py
```

### Baseline modules

```text
urban_graph_benchmark/src/ville_hgnn/baselines/a0_naive_temporal.py
urban_graph_benchmark/src/ville_hgnn/baselines/a1_svi_direct_ranking.py
urban_graph_benchmark/src/ville_hgnn/baselines/a2_calibrated_svi.py
```

### Wrappers and comparison scripts

```text
urban_graph_benchmark/scripts/02_build_dataset_v0.py
urban_graph_benchmark/scripts/03_build_splits_v0.py
urban_graph_benchmark/scripts/04_run_a0_a1_baselines.py
urban_graph_benchmark/scripts/05_compare_a0_a1_results.py
urban_graph_benchmark/scripts/06_compare_a0_a1_a2_results.py
```

### Reports and outputs

```text
urban_graph_benchmark/outputs/mtl_311_water_v0/datasets/
urban_graph_benchmark/outputs/mtl_311_water_v0/datasets/splits/
urban_graph_benchmark/outputs/mtl_311_water_v0/baselines/A0_naive_temporal/
urban_graph_benchmark/outputs/mtl_311_water_v0/baselines/A1_svi_direct_ranking/
urban_graph_benchmark/outputs/mtl_311_water_v0/baselines/A2_calibrated_svi/
urban_graph_benchmark/outputs/mtl_311_water_v0/baselines/A0_A1_comparison/
urban_graph_benchmark/outputs/mtl_311_water_v0/baselines/A0_A1_A2_comparison/
```

---

## 20. Recommended next step

The next methodological step is:

```text
urban_graph_benchmark/src/ville_hgnn/baselines/a3_tabular_feature_parity.py
```

A3 should answer:

```text
Can ordinary non-graph ML using tabular tract/month/static/reporting features beat A0 and A2?
```

Why A3 before graph models:

```text
If a tabular model already beats A0/A2, then graph models must beat A3.
If graph models only beat SVI but not tabular ML, then graph structure is not yet justified.
A3 is the necessary non-graph ML control before GraphSAGE/HGNN.
```

A3 should probably include:

```text
Strict forecasting feature sets:
  historical lag features available at prediction time
  month/calendar features
  tract static features
  SVI features
  maybe rolling history computed from past months only

Retrospective feature sets:
  same-month non-water 311 reporting controls
  clearly labeled as retrospective/explanatory

Models:
  regularized linear/log-count model
  random forest or gradient boosting if available
  maybe HistGradientBoostingRegressor from sklearn if installed

Outputs:
  predictions_validation.parquet
  predictions_test.parquet
  metrics.csv
  feature_importance.csv if available
  model_metadata.json
  baseline_report.md
```

A3 should preserve the same methodological discipline:

```text
train-only fitting
train-only imputation/scaling
temporal split as primary
strict separation between forecasting and retrospective controls
no target leakage
same metric schema as A0/A1/A2
```
