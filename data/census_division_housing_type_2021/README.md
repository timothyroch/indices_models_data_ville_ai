# Census Division Housing Type Features 2021

This folder contains the cleaned Québec census-division housing-type feature table derived from the 2021 Census Profile at the census-division level.

This feature block was created for the SoVI-like census-division index.

## Current status

This section is complete.

The cleaner successfully extracts one SoVI housing-type variable for all Québec census divisions:

```text
PCTMOBL90 -> pct_mobile_homes
```

Validated final run:

```text
Rows: 98 Québec census divisions
pct_mobile_homes: 98 non-missing, 0 missing
```

The final run confirmed:

```text
clean_rows: 98
unique_census_divisions: 98
all_variables_complete: True
base_names_with_mojibake: 0
profile_names_with_mojibake: 0
display_names_with_mojibake: 0
```

## Main input source

The main raw source is shared with other census-division Census Profile feature blocks:

```text
census_profile_census_division_2021/raw/98-401-X2021004_English_CSV_data.csv
```

This is the Statistics Canada 2021 Census Profile file for census divisions.

The cleaner reads this file directly. The raw file is **not copied** into this folder.

## Base geography

The cleaner joins to the reusable Québec census-division spatial/population frame:

```text
census_division_spatial_frame_population_2021/output/clean_quebec_census_division_spatial_frame_with_population_2021.csv
```

Validated geography:

```text
98 Québec census divisions
98 unique census-division DGUIDs
0 duplicated census-division keys
```

## Encoding note

The raw Statistics Canada Census Profile file is read with:

```text
cp1252
```

The existing cleaned base frame is read with UTF-8-first encoding.

The cleaner uses:

```text
Raw StatCan file: cp1252 first
Cleaned pipeline files: utf-8 first
```

The cleaner also keeps both the base and profile geography names and creates a safe display name:

```text
census_division_name_base_original
profile_census_division_name
census_division_name_display
```

## Scripts

### 1. Inspection script

```text
inspect_census_division_housing_type_2021.py
```

Purpose:

```text
Inspect the housing-type section of the 2021 Census Profile at census-division level.
Validate the movable-dwelling candidate characteristic, value column, symbols, coverage, and join compatibility.
```

SoVI variable inspected:

```text
PCTMOBL90 -> pct_mobile_homes
```

Important inspection outputs:

```text
output/housing_type_candidate_characteristics_2021.csv
output/housing_type_target_characteristic_summary_2021.csv
output/housing_type_target_values_long_2021.csv
output/housing_type_target_values_wide_2021.csv
output/housing_type_symbol_counts_2021.csv
output/housing_type_inspection_summary_2021.csv
```

The inspection confirmed that the target variable had complete coverage for all 98 Québec census divisions.

### 2. Cleaning script

```text
clean_census_division_housing_type_2021.py
```

Purpose:

```text
Create the final clean Québec census-division housing-type feature table.
```

Main output:

```text
output/clean_census_division_housing_type_2021.csv
```

Audit outputs:

```text
output/clean_census_division_housing_type_source_long_2021.csv
output/clean_census_division_housing_type_variable_metadata_2021.csv
output/clean_census_division_housing_type_summary_2021.csv
```

Note:

```text
The script attempts to save Parquet, but Parquet output requires pyarrow or fastparquet.
If those packages are not installed, the CSV and audit outputs are still created successfully.
```

## How to run

From the `data/` folder:

```bash
python census_division_housing_type_2021/inspect_census_division_housing_type_2021.py
python census_division_housing_type_2021/clean_census_division_housing_type_2021.py
```

## Clean variable

### `pct_mobile_homes`

Original SoVI variable:

```text
PCTMOBL90
Percent mobile homes
```

Local Census Profile proxy:

```text
CHARACTERISTIC_ID: 49
CHARACTERISTIC_NAME: Movable dwelling
VALUE_COLUMN: C10_RATE_TOTAL
SYMBOL_COLUMN: SYMBOL.3
```

Clean output column:

```text
pct_mobile_homes
```

Readable alias also included:

```text
pct_movable_dwelling
```

Validated range:

```text
min: 0.0
max: 10.7
mean: approximately 1.223
median: 0.9
```

Unit:

```text
percent
```

## Methodological choice

The original SoVI variable uses percent mobile homes.

The Canadian Census Profile does not use exactly the same naming in the relevant structural-type category. The closest local 2021 Census Profile category is:

```text
Movable dwelling
```

Therefore, this cleaner uses:

```text
49 — Movable dwelling
```

as the Canadian proxy for `PCTMOBL90`.

This is a proxy, not an exact historical reproduction of the original U.S. SoVI mobile-homes variable.

## Output table structure

The main clean table includes identity/geography columns:

```text
census_division_code
census_division_dguid
census_division_name
census_division_type
province_code
province_name
geography_level
census_year
population_total_2021
land_area_km2
has_positive_population
```

It also includes Census Profile geography/audit columns:

```text
profile_dguid
profile_census_division_code
profile_census_division_name
profile_geo_level
profile_tnr_sf
profile_tnr_lf
profile_data_quality_flag
```

Name encoding diagnostics:

```text
census_division_name_base_original
census_division_name_display
census_division_name_base_had_mojibake
profile_census_division_name_had_mojibake
```

Clean feature columns:

```text
pct_mobile_homes
pct_movable_dwelling
```

Per-variable audit columns:

```text
pct_mobile_homes__symbol
pct_mobile_homes__is_missing
pct_mobile_homes__source_characteristic_id
pct_mobile_homes__source_value_column
pct_mobile_homes__unit
pct_mobile_homes__sovi_role
pct_mobile_homes__methodological_choice
```

Block-level diagnostics:

```text
housing_type_feature_count
housing_type_missing_count
housing_type_complete
source
source_section
source_encoding
```

## Units

`pct_mobile_homes` is stored as a **percent value**, not a proportion.

Example:

```text
1.2 means 1.2%, not 0.012
```

This is appropriate for the SoVI-like PCA workflow, because variables will later be standardized before factor analysis.

## SoVI integration

This variable can be mapped into the SoVI input table as:

```text
PCTMOBL90 -> pct_mobile_homes
```

It should be treated as:

```text
movable_dwelling_proxy_for_mobile_homes
```

## Interpretation warning

This is an area-level census-division housing-type indicator.

It should not be interpreted as an individual-level measure.

The cleaned variable is a Canadian Census Profile proxy for the original SoVI mobile-homes variable, not an exact historical reproduction.

## Remaining work

None for this block.

Next recommended feature block:

```text
census_division_household_family_2021/
```

Likely target variables:

```text
AVGPERHH  -> avg_people_per_household
PCTF_HH90 -> pct_female_headed_households
```