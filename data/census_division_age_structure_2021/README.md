# Census Division Age Structure Features 2021

This folder contains the cleaned Québec census-division age/sex feature table derived from the 2021 Census Profile at the census-division level.

This feature block was created for the SoVI-like census-division index.

## Current status

This section is complete.

The cleaner successfully extracts four SoVI age/sex variables for all Québec census divisions:

```text
MED_AGE90  -> median_age
PCTKIDS90  -> pct_under_5
PCTOLD90   -> pct_over_65
PCTFEM90   -> pct_female
```

Validated final run:

```text
Rows: 98 Québec census divisions
median_age: 98 non-missing, 0 missing
pct_under_5: 98 non-missing, 0 missing
pct_over_65: 98 non-missing, 0 missing
pct_female: 98 non-missing, 0 missing
```

All four variables passed validation with complete coverage and no duplicated census-division rows.

The final run also confirmed that geography names no longer contain mojibake/encoding corruption:

```text
Base names with possible mojibake: 0
Profile names with possible mojibake: 0
Display names with possible mojibake: 0
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
98 matched between Census Profile and base frame
0 duplicated census-division keys
```

## Encoding note

The raw Statistics Canada Census Profile file is read with:

```text
cp1252
```

The existing cleaned base frame is read with UTF-8-first encoding.

This distinction matters because the raw StatCan file and our cleaned pipeline outputs may not use the same preferred encoding. The cleaner keeps the same defensive encoding strategy used in the labour-force block:

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
inspect_census_division_age_structure_2021.py
```

Purpose:

```text
Inspect the age/sex section of the 2021 Census Profile at census-division level.
Validate target characteristic IDs, value columns, symbols, coverage, derived formulas, and join compatibility.
```

SoVI variables inspected:

```text
MED_AGE90  -> median_age
PCTKIDS90  -> pct_under_5
PCTOLD90   -> pct_over_65
PCTFEM90   -> pct_female
```

Important inspection outputs:

```text
output/age_structure_candidate_characteristics_2021.csv
output/age_structure_target_characteristic_summary_2021.csv
output/age_structure_target_values_long_2021.csv
output/age_structure_target_values_wide_2021.csv
output/age_structure_symbol_counts_2021.csv
output/age_structure_inspection_summary_2021.csv
```

The inspection confirmed that all four target variables were ready for the cleaner.

### 2. Cleaning script

```text
clean_census_division_age_structure_2021.py
```

Purpose:

```text
Create the final clean Québec census-division age/sex feature table.
```

Main output:

```text
output/clean_census_division_age_structure_2021.csv
```

Audit outputs:

```text
output/clean_census_division_age_structure_source_long_2021.csv
output/clean_census_division_age_structure_variable_metadata_2021.csv
output/clean_census_division_age_structure_summary_2021.csv
```

Note:

```text
The script attempts to save Parquet, but Parquet output requires pyarrow or fastparquet.
If those packages are not installed, the CSV and audit outputs are still created successfully.
```

## How to run

From the `data/` folder:

```bash
python census_division_age_structure_2021/inspect_census_division_age_structure_2021.py
python census_division_age_structure_2021/clean_census_division_age_structure_2021.py
```

## Clean variables

### `median_age`

Original SoVI variable:

```text
MED_AGE90
Median age
```

Census Profile source:

```text
CHARACTERISTIC_ID: 40
CHARACTERISTIC_NAME: Median age of the population
VALUE_COLUMN: C1_COUNT_TOTAL
SYMBOL_COLUMN: SYMBOL
```

Clean output column:

```text
median_age
```

Validated range:

```text
min: 29.8
max: 58.0
mean: approximately 47.529
```

Unit:

```text
years
```

### `pct_under_5`

Original SoVI variable:

```text
PCTKIDS90
Percent population under five years old
```

Census Profile source:

```text
CHARACTERISTIC_ID: 10
CHARACTERISTIC_NAME: 0 to 4 years
VALUE_COLUMN: C10_RATE_TOTAL
SYMBOL_COLUMN: SYMBOL.3
```

Clean output column:

```text
pct_under_5
```

Validated range:

```text
min: 3.0
max: 9.6
mean: approximately 4.801
```

Unit:

```text
percent
```

### `pct_over_65`

Original SoVI variable:

```text
PCTOLD90
Percent population over 65
```

Census Profile source:

```text
CHARACTERISTIC_ID: 24
CHARACTERISTIC_NAME: 65 years and over
VALUE_COLUMN: C10_RATE_TOTAL
SYMBOL_COLUMN: SYMBOL.3
```

Clean output column:

```text
pct_over_65
```

Validated range:

```text
min: 9.1
max: 33.9
mean: approximately 24.022
```

Unit:

```text
percent
```

### `pct_female`

Original SoVI variable:

```text
PCTFEM90
Percent females
```

Census Profile source:

```text
CHARACTERISTIC_ID: 8
CHARACTERISTIC_NAME: Total - Age groups of the population - 100% data
```

Formula:

```text
100 * C3_COUNT_WOMEN+ / C1_COUNT_TOTAL
```

Clean output column:

```text
pct_female
```

Validated range:

```text
min: approximately 48.196
max: approximately 52.091
mean: approximately 49.856
```

Unit:

```text
percent
```

Note:

```text
This variable is derived from the Women+ count divided by the total population count.
```

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
median_age
pct_under_5
pct_over_65
pct_female
```

Per-variable audit columns:

```text
<variable>__symbol
<variable>__is_missing
<variable>__source_characteristic_id
<variable>__source_value_column_or_formula
<variable>__unit
<variable>__sovi_role
```

Block-level diagnostics:

```text
age_structure_feature_count
age_structure_missing_count
age_structure_complete
source
source_section
source_encoding
```

## Units

The age-share columns are stored as **percent values**, not proportions.

Example:

```text
4.8 means 4.8%, not 0.048
```

This is appropriate for the SoVI-like PCA workflow, because variables will later be standardized before factor analysis.

## SoVI integration

These four variables can be mapped into the SoVI input table as:

```text
MED_AGE90  -> median_age
PCTKIDS90  -> pct_under_5
PCTOLD90   -> pct_over_65
PCTFEM90   -> pct_female
```

They should be treated as:

```text
direct_or_strong_proxy
```

except for `pct_female`, which is a derived ratio from the Women+ count over total population.

## Interpretation warning

These are area-level census-division demographic indicators.

They should not be interpreted as individual-level measures.

## Remaining work

None for this block.

Next recommended feature block:

```text
census_division_income_2021/
```

Likely target variables:

```text
PERCAP89   -> per_capita_income / income proxy
PCTHH7589  -> pct_high_income_households
```