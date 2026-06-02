# SVI 2021 — Québec Census-Tract SVI-Like Index

This folder contains the Québec 2021 census-tract workflow for a CDC/ATSDR-inspired SVI-like social vulnerability index.

The workflow has four stages:

```text
1. Inspect available SVI input sources
2. Build a clean canonical SVI input table
3. Run the SVI-like index engine
4. Create map-ready spatial outputs
```

The current implementation is a **partial12 SVI-like local adaptation**, not a strict 15-variable SVI reproduction.

## Current status

Completed:

```text
SVI input source inspection
Canonical SVI input table
Partial12 SVI-like scoring run
Run report and metadata outputs
Map-generation script for GeoJSON/GPKG/CSV outputs
```

Current scored run:

```text
Index name: svi_like
Recipe: recipes/svi_quebec_2021_partial12.yaml
Run directory: outputs/svi_quebec_2021_partial12_run
Reproduction level: partial_svi_like
Input spatial units: 1480 Québec census tracts
Output spatial units: 1470 scored census tracts
Zero/invalid-population units excluded: 10
```

The SVI input table still contains all 15 canonical SVI columns, but the scoring recipe currently uses only the 12 usable variables.

Excluded from the partial12 scoring recipe:

```text
pct_disability
pct_no_vehicle
pct_group_quarters
```

These are excluded because they are exact-missing placeholders in the clean SVI input table.

## Main folder structure

```text
svi_2021/
├── output/
│   ├── svi_input_source_file_inventory_2021.csv
│   ├── svi_input_variable_availability_2021.csv
│   ├── svi_input_candidate_column_diagnostics_2021.csv
│   ├── svi_input_join_diagnostics_2021.csv
│   ├── svi_input_availability_summary_2021.csv
│   ├── clean_quebec_census_tract_svi_input_2021.csv
│   ├── clean_quebec_census_tract_svi_input_2021.parquet
│   ├── clean_quebec_census_tract_svi_input_2021.geojson
│   ├── clean_quebec_census_tract_svi_input_2021.gpkg
│   ├── clean_quebec_census_tract_svi_input_variable_metadata_2021.csv
│   ├── clean_quebec_census_tract_svi_input_missingness_report_2021.csv
│   └── clean_quebec_census_tract_svi_input_join_report_2021.csv
│
├── inspect_svi_input_sources_2021.py
├── clean_quebec_census_tract_svi_input_2021.py
├── create_svi_geojson_map_2021.py
└── README.md
```

Related recipe and output folders:

```text
recipes/svi_quebec_2021.yaml
recipes/svi_quebec_2021_partial12.yaml

outputs/svi_quebec_2021_partial12_run/
```

## Base spatial frame

The SVI input table is built on:

```text
spatial_frame_population_2021/output/clean_quebec_census_tract_spatial_frame_with_population_2021.parquet
```

This base layer provides:

```text
unit_id
statcan_dguid
unit_name
unit_type
census_year
province_id
province_name
land_area_km2
population_total
has_positive_population
geometry
```

The SVI cleaner creates convenience columns:

```text
zone_id = statcan_dguid
population = population_total
```

The SVI recipe uses:

```yaml
spatial_id_column: zone_id
population_column: population
```

## Geographic scope

The current SVI is built at the **census tract** level.

Important limitation:

```text
Census tracts do not cover all of Québec.
They mainly cover larger urban/metropolitan and tracted population areas.
```

So the SVI map should be described as:

```text
Québec census-tract SVI-like map for tracted areas only
```

not as:

```text
wall-to-wall Québec SVI map
```

The base input table has:

```text
1480 Québec census tract geometries
1470 positive-population census tracts
10 zero/non-positive/unknown-population tracts
```

The scoring recipe excludes the 10 zero/invalid-population tracts before ranking.

## CRS

The census-tract geometry source is in:

```text
EPSG:3347
NAD83 / Statistics Canada Lambert
```

This is a projected CRS in metres.

For web-map GeoJSON outputs, the map-generation script reprojects to:

```text
EPSG:4326
WGS84 longitude/latitude
```

This is necessary for MapLibre, Leaflet, and most web GeoJSON viewers.

## Methodology reminder

The SVI-like engine uses the CDC/ATSDR-style rank-domain method:

```text
raw variables
→ variable percentile ranks
→ domain raw sums
→ domain percentile ranks
→ overall sum of domain percentiles
→ final overall SVI percentile
→ flags
```

The current SVI implementation does **not** use PCA or factor analysis.

That distinguishes it from SoVI, which uses PCA/factor analysis with varimax rotation.

## Canonical 15 SVI variables

The clean SVI input table contains all 15 canonical columns:

```text
pct_below_poverty
pct_unemployed
per_capita_income
pct_no_high_school

pct_age_65_plus
pct_age_17_or_younger
pct_disability
pct_single_parent_households

pct_minority
pct_limited_language

pct_multiunit_structures
pct_mobile_homes
pct_crowding
pct_no_vehicle
pct_group_quarters
```

Three of these are currently all-NA exact placeholders:

```text
pct_disability
pct_no_vehicle
pct_group_quarters
```

A weak no-vehicle proxy candidate is preserved separately:

```text
pct_no_vehicle_weak_proxy_candidate
```

It is **not** mapped into `pct_no_vehicle` in the primary partial12 run.

## Partial12 scoring variables

The current scored recipe uses 12 variables.

### Socioeconomic status

```text
pct_below_poverty
pct_unemployed
per_capita_income
pct_no_high_school
```

### Household composition / disability

```text
pct_age_65_plus
pct_age_17_or_younger
pct_single_parent_households
```

### Minority status / language

```text
pct_minority
pct_limited_language
```

### Housing / transportation

```text
pct_multiunit_structures
pct_mobile_homes
pct_crowding
```

## Variable mapping

| SVI variable | Source folder | Source column | Status |
|---|---|---|---|
| `pct_below_poverty` | `low_income_2021` | `pct_low_income_lim_at` | Strong Canadian proxy |
| `pct_unemployed` | `unemployment_2021` | `pct_unemployed` | Direct / strong proxy |
| `per_capita_income` | `income_2021` | `income_measure_default` | Local adaptation proxy |
| `pct_no_high_school` | `education_2021` | `education_measure_default` | Local adaptation proxy |
| `pct_age_65_plus` | `age_structure_2021` | `pct_age_65_plus` | Direct / strong proxy |
| `pct_age_17_or_younger` | `age_structure_2021` | `pct_age_0_14` | Local adaptation proxy |
| `pct_single_parent_households` | `household_family_2021` | `single_parent_measure_default` | Local adaptation proxy |
| `pct_minority` | `immigration_ethnocultural_2021` | `ethnocultural_measure_default` | Local adaptation proxy |
| `pct_limited_language` | `language_2021` | `language_barrier_measure_default` | Local adaptation proxy |
| `pct_multiunit_structures` | `housing_type_2021` | `multiunit_measure_default` | Strong proxy |
| `pct_mobile_homes` | `housing_type_2021` | `mobile_home_measure_default` | Strong proxy |
| `pct_crowding` | `housing_suitability_crowding_2021` | `crowding_measure_default` | Strong proxy |

Missing/excluded variables:

| Canonical SVI variable | Current status |
|---|---|
| `pct_disability` | Missing; no acceptable cleaned census-tract proxy yet |
| `pct_no_vehicle` | Exact variable missing; weak commuting proxy preserved separately |
| `pct_group_quarters` | Missing; needs census-tract population in group quarters / collective dwellings |

## Input inspection script

Script:

```text
inspect_svi_input_sources_2021.py
```

Purpose:

```text
Validate candidate cleaned feature sources before building the final SVI input table.
```

It checks:

```text
source file existence
source file readability
row counts
join keys
candidate columns
missingness
join compatibility with the census-tract base frame
```

Main outputs:

```text
output/svi_input_source_file_inventory_2021.csv
output/svi_input_variable_availability_2021.csv
output/svi_input_candidate_column_diagnostics_2021.csv
output/svi_input_join_diagnostics_2021.csv
output/svi_input_availability_summary_2021.csv
```

Validated finding:

```text
12 variables ready for clean table
2 variables missing with no current proxy
1 variable with weak proxy candidate only
```

## Clean SVI input table script

Script:

```text
clean_quebec_census_tract_svi_input_2021.py
```

Purpose:

```text
Build the canonical wide input table for SVI-like scoring.
```

Main outputs:

```text
output/clean_quebec_census_tract_svi_input_2021.csv
output/clean_quebec_census_tract_svi_input_2021.parquet
output/clean_quebec_census_tract_svi_input_2021.geojson
output/clean_quebec_census_tract_svi_input_2021.gpkg
```

Audit outputs:

```text
output/clean_quebec_census_tract_svi_input_variable_metadata_2021.csv
output/clean_quebec_census_tract_svi_input_missingness_report_2021.csv
output/clean_quebec_census_tract_svi_input_join_report_2021.csv
```

Validated result:

```text
Rows: 1480
Positive-population rows: 1470
Canonical SVI columns: 15
Ready/main variable columns: 12
Missing/exact-placeholder variables: 3
```

The three exact-placeholder variables are:

```text
pct_disability
pct_no_vehicle
pct_group_quarters
```

## Scoring recipe

Primary current scoring recipe:

```text
recipes/svi_quebec_2021_partial12.yaml
```

This recipe excludes the three exact-missing variables and runs the SVI-style method on the 12 currently usable variables.

Important settings:

```yaml
name: svi_like
reproduction_level: partial_svi_like
spatial_id_column: zone_id
population_column: population

zero_population:
  action: exclude
  threshold: 0

ranking:
  method: percentile_rank
  formula: "(rank - 1) / (N - 1)"
  tie_method: min

missing_data:
  strategy: keep_missing_with_flags
  add_missing_flags: true

classification:
  method: quantile
  n_classes: 5
```

Earlier full 15-variable recipe:

```text
recipes/svi_quebec_2021.yaml
```

This recipe is useful as documentation, but it currently fails validation because three variables are all-NA placeholders:

```text
pct_disability
pct_no_vehicle
pct_group_quarters
```

## Running the SVI score

From the project root:

```bash
PYTHONPATH=src python -m ville_indices.run \
  --index svi_like \
  --recipe recipes/svi_quebec_2021_partial12.yaml \
  --features data/svi_2021/output/clean_quebec_census_tract_svi_input_2021.csv \
  --output-dir outputs/svi_quebec_2021_partial12_run
```

Main scoring outputs:

```text
outputs/svi_quebec_2021_partial12_run/standard_output.csv
outputs/svi_quebec_2021_partial12_run/intermediate_output.csv
outputs/svi_quebec_2021_partial12_run/metadata.json
outputs/svi_quebec_2021_partial12_run/metadata.yaml
outputs/svi_quebec_2021_partial12_run/validation_report.json
outputs/svi_quebec_2021_partial12_run/validation_report.yaml
outputs/svi_quebec_2021_partial12_run/missing_data_report.json
outputs/svi_quebec_2021_partial12_run/run_report.md
```

Validated run summary:

```text
Spatial units input: 1480
Spatial units output: 1470
Zero/invalid population units excluded: 10
SVI overall percentile minimum: 0.0
SVI overall percentile mean: approximately 0.5
SVI overall percentile maximum: 1.0
```

## Map-generation script

Script:

```text
create_svi_geojson_map_2021.py
```

Purpose:

```text
Join SVI score outputs back to census-tract geometry and create map-ready files.
```

Important design choices:

```text
Reads clean SVI geometry from the data folder.
Reads SVI scores from outputs/svi_quebec_2021_partial12_run.
Does not modify source data or boundary files.
Keeps full native audit output in EPSG:3347.
Exports web GeoJSON in EPSG:4326.
Uses continuous color mapping based on exact SVI percentile.
Does not use hardcoded vulnerability-class colors.
```

Run from `data/`:

```bash
python svi_2021/create_svi_geojson_map_2021.py
```

Or from the project root:

```bash
python data/svi_2021/create_svi_geojson_map_2021.py
```

Expected map outputs:

```text
outputs/svi_quebec_2021_partial12_run/svi_quebec_2021_partial12_map_web.geojson
outputs/svi_quebec_2021_partial12_run/svi_quebec_2021_partial12_map_native.gpkg
outputs/svi_quebec_2021_partial12_run/svi_quebec_2021_partial12_map_audit.csv
outputs/svi_quebec_2021_partial12_run/svi_quebec_2021_partial12_map_legend.csv
```

Use this file for MapLibre / Leaflet / web viewers:

```text
outputs/svi_quebec_2021_partial12_run/svi_quebec_2021_partial12_map_web.geojson
```

Reason:

```text
It is reprojected to EPSG:4326, which web maps expect.
```

Use this file for QGIS / audit / high-precision spatial work:

```text
outputs/svi_quebec_2021_partial12_run/svi_quebec_2021_partial12_map_native.gpkg
```

Reason:

```text
It keeps the native EPSG:3347 Statistics Canada Lambert CRS.
```

## Map color logic

The web GeoJSON includes continuous styling fields:

```text
svi_percentile
svi_color
svi_color_value_0_1
fill
fill-opacity
stroke
stroke-width
stroke-opacity
svi_vulnerability_label
```

The actual color is generated continuously from the exact SVI percentile.

This means:

```text
0.29 and 0.31 receive different colors.
```

The vulnerability labels are still included for human interpretation:

```text
Very low
Low
Moderate
High
Very high
No percentile
Not scored
```

But the color is not assigned by these classes. The color is assigned by the continuous percentile value.

## Handling unscored and missing-percentile polygons

The map layer distinguishes:

```text
Not scored
```

from:

```text
No percentile
```

`Not scored` means:

```text
The tract exists in the geometry layer but was excluded from SVI scoring,
usually because population <= 0.
```

`No percentile` means:

```text
The tract was included in the SVI scoring output but has a missing final percentile.
```

The 10 zero-population tracts are retained in the map layer as unscored polygons, rather than silently dropped.

## Interpretation warning

This table and map are **area-level** outputs.

They should not be interpreted as saying:

```text
every person in a high-SVI census tract is vulnerable
```

The correct interpretation is:

```text
higher SVI percentile means the census tract has higher relative social vulnerability
according to the selected partial12 SVI-like variable set.
```

## Important methodological warning

This is not a strict CDC/ATSDR SVI reproduction.

Reasons:

```text
Canadian/Québec local proxies are used.
The geography is Québec census tracts, not U.S. census tracts/counties.
Three original SVI variables are missing and excluded from the current scoring recipe.
The run is partial_svi_like.
```

The current output should be described as:

```text
Québec 2021 census-tract partial12 SVI-like local adaptation
```

not as:

```text
complete 15-variable SVI
```

## Relationship to SoVI

This SVI workflow is different from the SoVI workflow.

SVI:

```text
fixed conceptual variables
percentile ranks
domain sums
domain percentile ranks
overall percentile
```

SoVI:

```text
large variable matrix
normalization to percentages/per-capita/densities
standardization
PCA/factor analysis
varimax rotation
factor orientation
additive factor-score sum
standard-deviation classes
```

The current SVI outputs can later be compared against SoVI-style outputs as one benchmark index.

## Recommended next work

Immediate:

```text
Confirm the final web GeoJSON opens correctly in MapLibre or another web viewer.
Optionally inspect the native GPKG in QGIS.
```

Then:

```text
Continue with the SoVI variable inventory and SoVI input-table construction.
```

Later SVI improvements:

```text
Find or build pct_disability.
Find or build pct_no_vehicle.
Find or build pct_group_quarters.
Run a full 15-variable local-adaptation SVI if those variables become available.
Run a sensitivity version using pct_no_vehicle_weak_proxy_candidate.
Compare partial12 SVI against full/alternative SVI variants.
```