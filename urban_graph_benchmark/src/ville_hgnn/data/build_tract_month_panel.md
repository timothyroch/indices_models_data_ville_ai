# `build_tract_month_panel.py` Implementation Notes

## Montréal 311 Water/Drainage Dataset v0

**Project:** `urban_graph_benchmark`  
**Benchmark ID:** `mtl_311_water_v0`  
**Main module:** `urban_graph_benchmark/src/ville_hgnn/data/build_tract_month_panel.py`  
**Main output folder:** `urban_graph_benchmark/outputs/mtl_311_water_v0/datasets/`  
**Current final status:** Dataset v0 builds successfully with `Validation status: pass`.

---

## 1. Purpose of this module

The goal of `build_tract_month_panel.py` is to build the first empirical benchmark dataset for the Montréal 311 water/drainage modeling task.

The target dataset is:

```text
Montréal census tract × month
target = reported municipal 311 water/drainage requests
```

This dataset is meant to support the first benchmark phase:

```text
SVI baseline
simple temporal/exposure baselines
simple spatial/tabular model
later adjacency GNN
later HGNN
```

This module is deliberately **not** responsible for:

```text
baseline model training
GraphSAGE
HGNN
explainability
OSM routing
road-network distance calculation
population-weighted centroids
expert-based validation
SoVI integration
```

Those belong in later modules.

---

## 2. Input data used

### 2.1 Direct 311 benchmark input

For Dataset v0, we use the derived grid25m-month 311 feature table:

```text
transformation/output/ville_ia_311_features_grid25m_monthly.parquet
```

This file is the direct input to Dataset v0.

Manual inspection showed:

```text
shape = (760621, 29)
```

Columns:

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

Important interpretation:

```text
This is not the raw 311 source.
It is a derived grid25m × month feature table.
```

The upstream provenance/raw source remains:

```text
transformation/requetes311.csv
```

The benchmark config should treat the grid25m-month parquet as the direct v0 benchmark input, while preserving `requetes311.csv` as provenance.

---

### 2.2 Census tract geometry

Resolved canonical path:

```text
data/spatial_frame_population_2021/output/clean_quebec_census_tract_spatial_frame_with_population_2021.gpkg
```

Inventory showed:

```text
CRS: EPSG:3347
usable tract identifiers
population columns available
land area columns available
```

This file is used to:

```text
define tract polygons
assign 311 grid coordinate rows to tracts
build static tract features
compute geometric tract centroids
join population / land area / density
```

---

### 2.3 SVI scored output

Resolved canonical path:

```text
outputs/svi_quebec_2021_partial12_run/svi_quebec_2021_partial12_map.csv
```

This is the SVI score/baseline output.

It includes columns such as:

```text
unit_id
svi_percentile
svi_score_raw
svi_rank
svi_class
svi_quality_flag
svi_missing_count
svi_reproduction_level
svi_scored
svi_vulnerability_label
svi_color
svi_color_value_0_1
```

This file is joined to the tract-month panel so that SVI can be evaluated as a baseline.

---

## 3. Configuration assumptions

The important config decisions after inventory were:

```yaml
inputs:
  montreal_311:
    raw_path: transformation/output/ville_ia_311_features_grid25m_monthly.parquet
    source_type: grid25m_month
    upstream_raw_source:
      path: transformation/requetes311.csv

  census_tract_geometry:
    path: data/spatial_frame_population_2021/output/clean_quebec_census_tract_spatial_frame_with_population_2021.gpkg
    crs: EPSG:3347
    id_column: unit_id

  svi:
    path: outputs/svi_quebec_2021_partial12_run/svi_quebec_2021_partial12_map.csv
    id_column: unit_id

spatial_assignment:
  method: grid_centroid_in_polygon

target:
  category_selection:
    method: precomputed_aggregated_columns
  columns:
    count: water_drainage_requests
```

The phrase `precomputed_aggregated_columns` means:

```text
The v0 311 file already contains the target column water_drainage_requests.
Therefore Dataset v0 does not reconstruct water/drainage categories from raw 311 text/category fields.
```

Raw-category reconstruction is left as a later reproducibility audit.

---

## 4. Output artifacts

The module writes:

```text
urban_graph_benchmark/outputs/mtl_311_water_v0/datasets/tract_month_panel.parquet
urban_graph_benchmark/outputs/mtl_311_water_v0/datasets/tract_static_features.parquet
urban_graph_benchmark/outputs/mtl_311_water_v0/datasets/target_water_drainage.parquet
urban_graph_benchmark/outputs/mtl_311_water_v0/datasets/dataset_validation.json
urban_graph_benchmark/outputs/mtl_311_water_v0/datasets/dataset_report.md
urban_graph_benchmark/outputs/mtl_311_water_v0/datasets/spatial_join_audit.csv
urban_graph_benchmark/outputs/mtl_311_water_v0/datasets/missingness_report.csv
urban_graph_benchmark/outputs/mtl_311_water_v0/datasets/feature_dictionary.csv
urban_graph_benchmark/outputs/mtl_311_water_v0/datasets/provenance.json
```

---

## 5. High-level pipeline

The final pipeline is:

```text
1. Load config.
2. Load grid25m-month 311 table.
3. Load census tract geometry.
4. Load SVI scored output.
5. Assign each distinct grid coordinate row to a census tract.
6. Merge spatial assignments back to grid-month rows using coordinate keys.
7. Aggregate grid-month features to tract × month.
8. Build complete tract × month panel.
9. Zero-fill missing tract-month target/count rows.
10. Join static tract features.
11. Join SVI score columns.
12. Create canonical targets and reporting controls.
13. Validate dataset.
14. Write parquet, CSV, JSON, and Markdown artifacts.
```

---

## 6. Methodological decisions

### 6.1 Why Dataset v0 uses grid25m-month data

We initially considered using `transformation/requetes311.csv` as the direct input because it is closer to raw municipal 311 data.

However, the inventory showed that the derived grid25m-month file is already clean and directly usable for the tract-month benchmark:

```text
unit_id
period_month
water_drainage_requests
requests_total
lat_centroid
lon_centroid
x_centroid
y_centroid
```

For Dataset v0, we therefore use:

```text
direct benchmark input = grid25m-month parquet
upstream provenance source = raw requetes311.csv
```

This keeps v0 feasible while preserving provenance.

### 6.2 Why the target is a count/magnitude target, not just binary

The final target distribution showed:

```text
panel shape: (28620, 65)
zones: 540
months: 53

water_drainage_count summary:
count    28620
mean         5.255835
std          5.176364
min          0
25%          2
50%          4
75%          7
max         71

positive rate: approximately 84.9%
total water/drainage requests: 150422
```

Because the binary target `water_drainage_count > 0` is positive for about 85% of tract-months, it is probably not the best primary modeling target.

The useful benchmark targets are more likely:

```text
count prediction
magnitude prediction
ranking of high-burden tract-months
top-k detection
distributional or ordinal target later
```

This aligns with the idea that the model should not merely predict a single event/no-event signal, but rather a magnitude/distribution-like signal.

### 6.3 Why SVI is joined to tract-month rows

SVI is static at census-tract level. The target is dynamic at tract × month level.

Dataset v0 joins SVI to every month of the corresponding tract.

This enables:

```text
SVI direct ranking baseline
SVI calibrated predictor
comparison of SVI score against observed 311 water/drainage burden
```

The benchmark does not compare SVI and SoVI under one universal setting here. Track A is tract-level Montréal and therefore uses SVI only.

### 6.4 Why SoVI is excluded from Track A

SoVI was reproduced at census-division level, and many SoVI variables are only available at CD scale. It should not be forced into a tract-month Montréal benchmark.

Dataset v0 explicitly validates that there are no SoVI columns in Track A:

```text
no_sovi_columns_in_track_a = True
```

---

## 7. First major failure: no grid cells assigned

### 7.1 Error

First build attempt failed with:

```text
DatasetBuildError: No 311 grid-month rows could be assigned to census tracts.
```

This occurred inside:

```text
assign_grid_units_to_tracts()
```

The first version of the code assumed:

```text
x_centroid / y_centroid are EPSG:3347
```

and built points directly from those coordinates.

### 7.2 CRS diagnostic

We ran a CRS diagnostic.

Tract geometry:

```text
tract CRS: EPSG:3347
tract bounds:
[7412545.56857147 1182721.63142861 7791634.08571433 1648185.32000003]
```

Grid `x_centroid/y_centroid` bounds:

```text
x ≈ 267,475 to 306,410
y ≈ 5,029,293 to 5,062,642
```

Those values do not overlap the tract geometry bounds in EPSG:3347.

Then we tested `lon_centroid/lat_centroid` transformed from EPSG:4326 to EPSG:3347:

```text
grid lon/lat transformed bounds:
[7603100.30697657 1225372.35063217 7636429.91411634 1268339.1661192]
```

These transformed bounds overlap the tract geometry bounds.

### 7.3 Conclusion

The assumption that `x_centroid/y_centroid` were EPSG:3347 was false.

The correct v0 approach is:

```text
use lon_centroid / lat_centroid as EPSG:4326
reproject to tract geometry CRS
then perform centroid-in-polygon
```

### 7.4 Patch implemented

`assign_grid_units_to_tracts()` was patched to prefer:

```text
lon_centroid / lat_centroid
source CRS = EPSG:4326
target CRS = tracts.crs, currently EPSG:3347
```

The audit keeps both coordinate systems, but does not assume `x_centroid/y_centroid` are EPSG:3347.

---

## 8. Second major failure: SVI join success only 36%

### 8.1 Symptom

After the CRS fix, the build ran, but status was `warning`.

SVI join summary:

```text
static_tracts_in_scope = 540
svi_rows = 1480
matched_static_tracts = 195
missing_svi_rows = 345
svi_join_success_rate = 0.3611111111111111
```

### 8.2 Diagnostic

We inspected IDs.

Static output examples:

```text
4620001.00
4620002.00
4620003.00
```

SVI CSV examples:

```text
4620001.0
4620002.0
```

This was caused by pandas reading tract unit IDs from CSV as numeric-like values and collapsing trailing zeroes.

A DGUID attempt failed because the static file's `census_tract_dguid` was not a real StatCan DGUID in the built output. It had been created as a fallback equal to `zone_id`.

We then tested canonicalized unit IDs:

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

### 8.3 Patch implemented

Added:

```python
def canonicalize_tract_unit_id(value: Any) -> str:
    if pd.isna(value):
        return ""

    text = str(value).strip()

    if text == "":
        return ""

    try:
        return f"{float(text):.2f}"
    except Exception:
        return text
```

Used it in:

```text
build_tract_static_features()
load_and_canonicalize_svi()
```

Specifically:

```python
gdf["zone_id"] = gdf[tract_id_col].map(canonicalize_tract_unit_id)
out["zone_id"] = svi[id_col].map(canonicalize_tract_unit_id)
```

After this patch:

```text
SVI join success = 1.0
missing_svi_rows = 0
```

---

## 9. Third methodological issue: multiple coordinate rows per grid unit

### 9.1 Initial concern

The first successful build reported:

```text
units_with_multiple_coordinate_rows = 17,726
```

This raised a concern:

```text
If the same unit_id has multiple centroid rows, and if those centroid rows can fall into different tracts,
then assigning one tract per unit_id using the first coordinate is not reliable.
```

The existing code did:

```text
unit_id → first coordinate row → one assigned tract forever
```

### 9.2 Audit performed

We audited distinct coordinate rows.

Results:

```text
distinct coordinate rows: 84,351
unique grid units: 42,752
units with >1 coordinate row: 17,726
units assigned to >1 tract: 538
```

Coordinate variation summary, in meters:

```text
count    42752
mean         0.580818
std          2.533244
min          0
25%          0
50%          0
75%          0.031764
max         28.704479
```

Interpretation:

Most coordinate variation is very small, but 538 grid unit IDs can cross tract assignment boundaries depending on the coordinate row.

Examples included grid units assigned to neighboring tracts such as:

```text
grid25m_273250_5040875 → 4620550.02 and 4620550.04
grid25m_273925_5035875 → 4620515.01 and 4620515.02
grid25m_275000_5038750 → 4620550.02 and 4620550.04
```

### 9.3 Impact proxy

We then computed how much data belongs to unit IDs with multiple coordinate rows:

```text
rows affected by multi-coordinate unit_ids: 479,570
share rows affected: 0.6304979746812144

requests_total affected: 1,109,709
share requests_total affected: 0.7066702327847243

water_drainage affected: 101,625
share water_drainage affected: 0.6755858401196609
```

Important nuance:

This does not mean 67.6% of water/drainage requests were wrongly assigned. Most multi-coordinate units stay inside the same tract. But it means the phenomenon is large enough that the shortcut should not be kept.

### 9.4 Methodological conclusion

The correct assignment unit should be:

```text
unit_id + coordinate row
```

not just:

```text
unit_id
```

Therefore the pipeline should be:

```text
distinct coordinate row
→ spatial join to tract
→ merge assignment back to grid rows using coordinate key
→ aggregate to tract × month
```

---

## 10. Coordinate-row spatial assignment patch

### 10.1 Updated `assign_grid_units_to_tracts()`

The function was updated to assign one row per distinct coordinate row.

Preferred key:

```text
unit_id + lon_centroid + lat_centroid
```

Fallback key:

```text
unit_id + x_centroid + y_centroid
```

The function now reports:

```text
total_unique_grid_units
assigned_unique_grid_units
unassigned_unique_grid_units
assignment_success_rate_unique_grid_units

total_unique_coordinate_rows
assigned_unique_coordinate_rows
unassigned_unique_coordinate_rows
assignment_success_rate_coordinate_rows

units_with_multiple_coordinate_rows
units_with_multiple_assigned_tracts
coordinate_source
coordinate_source_crs
coordinate_target_crs
spatial_join_method
```

### 10.2 Index/column collision bug

After the coordinate-row patch, the build failed with:

```text
ValueError: cannot insert coordinate_row_id, already exists
```

Cause:

```python
points = points.set_index("coordinate_row_id", drop=False)
```

This made `coordinate_row_id` both an index name and a normal column. GeoPandas internally calls `reset_index()` during `sjoin()`, causing a collision.

Fix:

```python
points = points.set_index("coordinate_row_id", drop=True)
points.index.name = "coordinate_row_id"
```

Then after `sjoin()`:

```python
if "coordinate_row_id" not in joined.columns:
    joined = joined.reset_index()
```

### 10.3 Valid mask bug

We also fixed valid-coordinate mask logic by computing the valid mask **after** `coordinate_rows` had been deduplicated and reset.

This avoids relying on the previous grid index after deduplication.

---

## 11. Updated aggregation patch

### 11.1 Previous aggregation behavior

The old aggregation merged assignment using only:

```text
unit_id
```

Example old code:

```python
assignment = assignment_audit[["unit_id", "assigned_zone_id", "assigned"]].copy()
assigned_grid = grid.merge(assignment, on="unit_id", how="left", validate="many_to_one")
```

This became invalid once assignment was coordinate-row based.

### 11.2 New aggregation behavior

`aggregate_grid_to_tract_month()` now detects and merges by coordinate key.

Preferred:

```text
unit_id + lon_centroid + lat_centroid
```

Fallback:

```text
unit_id + x_centroid + y_centroid
```

It validates that assignment keys are unique:

```text
duplicate assignment keys → hard DatasetBuildError
```

It reports:

```text
assignment_merge_key_cols
rows_without_assignment_record_after_coordinate_merge
assigned_grid_month_rows_used
unassigned_grid_month_rows_excluded
grid_units_with_multiple_assigned_tracts
```

This ensures each grid-month row is assigned to the tract corresponding to its own centroid row.

---

## 12. Final successful build

Final build command:

```bash
PYTHONPATH=urban_graph_benchmark/src \
python urban_graph_benchmark/src/ville_hgnn/data/build_tract_month_panel.py \
  --config urban_graph_benchmark/configs/mtl_311_water_v0.yaml
```

Final terminal output:

```text
Dataset v0 build completed.
Status: pass
Rows: 28620
Zones: 540
Months: 53
Period: 2022-01 to 2026-05
Grid assignment success: 0.9995087949101796
SVI join success: 1.0
```

---

## 13. Final dataset report summary

Final `dataset_report.md` says:

```text
Validation status: pass
n_zones = 540
n_months = 53
expected_rows = 28620
actual_rows = 28620
zero_filled_tract_month_rows = 1549
period_month_min = 2022-01
period_month_max = 2026-05
```

Spatial assignment:

```text
total_unique_grid_units = 42752
assigned_unique_grid_units = 42731
unassigned_unique_grid_units = 21
assignment_success_rate_unique_grid_units = 0.9995087949101796

total_unique_coordinate_rows = 83829
assigned_unique_coordinate_rows = 83805
unassigned_unique_coordinate_rows = 24
assignment_success_rate_coordinate_rows = 0.9997137028951795

coordinate_source = lon_lat_centroid_epsg4326_to_tract_crs
coordinate_source_crs = EPSG:4326
coordinate_target_crs = EPSG:3347

spatial_join_method =
grid_coordinate_row_centroid_in_polygon_using_lon_lat_epsg4326_reprojected_to_tract_crs

units_with_multiple_coordinate_rows = 17655
units_with_multiple_assigned_tracts = 538
within_duplicate_assignments = 0
boundary_fallback_duplicate_assignments = 0

n_assigned_zone_ids = 540
n_static_tracts_in_scope = 540
```

Aggregation:

```text
assigned_grid_month_rows_used = 760591
unassigned_grid_month_rows_excluded = 30
```

SVI join:

```text
static_tracts_in_scope = 540
svi_rows = 1480
matched_static_tracts = 540
missing_svi_rows = 0
svi_join_success_rate = 1.0
```

Validation:

```text
one_row_per_zone_month = True
all_expected_tracts_represented_in_every_month = True
zero_filled_missing_target_rows = True
water_drainage_count_nonnegative = True
total_311_count_non_water_drainage_nonnegative = True
no_sovi_columns_in_track_a = True
no_missing_zone_id = True
target_table_row_count_matches_panel = True
```

---

## 14. Final data consistency checks

### 14.1 Panel and target table

```text
panel shape: (28620, 65)
target shape: (28620, 10)
zones: 540
months: 53
duplicate tract-month rows: 0
```

### 14.2 Assignment audit

```text
audit rows: 83829
assigned rows: 83805
unassigned rows: 24
coordinate_row_id unique: 83829
assignment methods:
  centroid_within_polygon: 83805
  unassigned: 24
```

### 14.3 Target distribution

```text
water_drainage_count:
count    28620
mean         5.255835
std          5.176364
min          0
25%          2
50%          4
75%          7
max         71

water total: 150422
positive rate: approximately 0.84888
```

### 14.4 SVI check

```text
SVI columns in panel: 18
rows missing all SVI: 0
```

### 14.5 Source-total preservation

Source vs panel totals:

```text
source water total: 150425
panel water total: 150422
difference: 3

source requests total: 1570335
panel requests total: 1570303
difference: 32
```

Interpretation:

Only 3 water/drainage requests and 32 total 311 requests are lost because 30 grid-month rows remain unassigned. This is negligible.

---

## 15. Current known limitations

### 15.1 Dataset uses reported municipal 311 demand, not objective flood occurrence

The target is a reported municipal 311 signal.

It captures:

```text
water/drainage-related public service requests
municipal reporting behavior
service demand
possibly localized disruption
```

It does not directly prove:

```text
objective flooding
physical infrastructure failure
true exposure
true damage
```

This should be explicit in any paper or benchmark description.

### 15.2 Direct input is derived, not raw

Dataset v0 directly uses:

```text
ville_ia_311_features_grid25m_monthly.parquet
```

This is derived from upstream 311 data.

Future reproducibility work should rebuild the grid25m-month file from:

```text
transformation/requetes311.csv
```

or from the official municipal source, if appropriate.

### 15.3 In-scope tracts are defined empirically

Current rule:

```text
tracts_with_at_least_one_assigned_311_grid25m_unit
```

This is a v0 empirical service-territory proxy.

It is not yet a formal Ville de Montréal or Montréal agglomeration service-boundary definition.

Future improvement:

```text
derive exact Montréal/agglomeration tract set using official boundary
compare against empirical 311-covered tract set
```

### 15.4 No population-weighted centroids yet

The static features include geometric tract centroids and reserved/null population-weighted centroid columns.

Population-weighted corrected centers are planned for v1 because they matter for:

```text
distance to hospitals
distance to parks/cooling resources
distance to critical services
accessibility graph edges
road-network features
```

This was inspired by a Laval-style methodology using population-weighted centers and network travel distances.

### 15.5 No road-network distances yet

Dataset v0 does not implement:

```text
OpenStreetMap routing
OSRM/OSMR
distance by car
walking distance
distance to nearest hospital/park/library/cooling resource
critical infrastructure access
```

These should be later modules, for example:

```text
population_weighted_centroids.py
accessibility_features.py
network_distance_features.py
```

### 15.6 Unique activity/responsible-unit counts omitted

The source file has:

```text
unique_activity_count
unique_responsible_units
```

These are omitted in v0 because exact tract-month recomputation from grid-level aggregates is not reliable.

Summing unique counts across grid cells would overcount.

### 15.7 Magnitude classes not created yet

Dataset v0 keeps:

```text
water_drainage_count
water_drainage_binary
```

But it does not create official magnitude classes.

Reason:

```text
magnitude thresholds should be split-specific
thresholds should avoid train/test leakage
```

This belongs in the modeling/split pipeline.

### 15.8 Reporting controls are retrospective-only

The panel includes:

```text
total_311_count_all
total_311_count_non_water_drainage
```

Important caveat:

```text
total_311_count_all contains the target
```

Therefore, it should not be used in forecasting settings.

Preferred same-month reporting control for retrospective models:

```text
total_311_count_non_water_drainage
```

Even this is retrospective, not necessarily valid for strict future prediction unless lagged.

---

## 16. What we should do next

### 16.1 Engineering next step

Create the thin wrapper:

```text
urban_graph_benchmark/scripts/02_build_dataset_v0.py
```

This should mirror `01_inventory_inputs.py`.

It should:

```text
bootstrap urban_graph_benchmark/src if needed
parse --config
parse --repo-root
call ville_hgnn.data.build_tract_month_panel.run_build_dataset()
print build_brief(result)
print output paths
```

It should not duplicate dataset-building logic.

### 16.2 Scientific/modeling next step

Start with baselines, not GraphSAGE/HGNN.

Recommended first benchmark ladder:

```text
A0 naive temporal/exposure baseline
A1 SVI direct ranking baseline
A2 calibrated SVI predictor
A3 simple feature-parity spatial/tabular model
```

Graph models should come after the non-graph baselines are implemented and evaluated.

---

## 17. Final status statement

As of the final build:

```text
Dataset v0 is valid and ready for first baseline development.
```

The dataset now has:

```text
coordinate-row spatial assignment
complete SVI join
validated tract-month panel
negligible unassigned loss
clear provenance
documented limitations
```

The major methodological bugs encountered during development were resolved:

```text
wrong CRS assumption for x/y centroids
SVI ID formatting mismatch
unit_id-level spatial assignment despite multiple coordinate rows
coordinate_row_id index/column collision during GeoPandas spatial join
```