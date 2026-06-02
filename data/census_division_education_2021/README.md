# Census Division Education Features 2021

This folder contains the cleaned Québec census-division education feature table derived from the 2021 Census Profile at the census-division level.

This feature block was created for the SoVI-like census-division index.

## Current status

This section is complete.

The cleaner successfully extracts one preferred SoVI education variable and three audit/sensitivity variables for all Québec census divisions.

Preferred SoVI variable:

```text
PCTNOHSDP90 -> pct_no_high_school_diploma
```

Audit variables:

```text
pct_no_high_school_diploma_25_64
pct_no_certificate_diploma_or_degree_15_plus
pct_no_certificate_diploma_or_degree_25_64
```

Validated final run:

```text
Rows: 98 Québec census divisions
pct_no_high_school_diploma: 98 non-missing, 0 missing
pct_no_high_school_diploma_25_64: 98 non-missing, 0 missing
pct_no_certificate_diploma_or_degree_15_plus: 98 non-missing, 0 missing
pct_no_certificate_diploma_or_degree_25_64: 98 non-missing, 0 missing
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
inspect_census_division_education_2021.py
```

Purpose:

```text
Inspect the education section of the 2021 Census Profile at census-division level.
Identify candidate characteristics for the SoVI no-high-school-diploma variable.
Validate coverage, value columns, symbols, parent universes, and join compatibility.
```

SoVI variable inspected:

```text
PCTNOHSDP90 -> pct_no_high_school_diploma
```

Important inspection outputs:

```text
output/education_candidate_characteristics_2021.csv
output/education_target_characteristic_summary_2021.csv
output/education_target_values_long_2021.csv
output/education_target_values_wide_2021.csv
output/education_symbol_counts_2021.csv
output/education_inspection_summary_2021.csv
```

The inspection found multiple full-coverage education candidates, so the final cleaner required an explicit methodological choice.

### 2. Cleaning script

```text
clean_census_division_education_2021.py
```

Purpose:

```text
Create the final clean Québec census-division education feature table.
```

Main output:

```text
output/clean_census_division_education_2021.csv
```

Audit outputs:

```text
output/clean_census_division_education_source_long_2021.csv
output/clean_census_division_education_variable_metadata_2021.csv
output/clean_census_division_education_summary_2021.csv
```

Note:

```text
The script attempts to save Parquet, but Parquet output requires pyarrow or fastparquet.
If those packages are not installed, the CSV and audit outputs are still created successfully.
```

## How to run

From the `data/` folder:

```bash
python census_division_education_2021/inspect_census_division_education_2021.py
python census_division_education_2021/clean_census_division_education_2021.py
```

## Clean variables

### `pct_no_high_school_diploma`

Original SoVI variable:

```text
PCTNOHSDP90
Percent persons with no high school diploma
```

Selected local Census Profile source:

```text
CHARACTERISTIC_ID: 1993
CHARACTERISTIC_NAME: No high school diploma or equivalency certificate
PARENT_CHARACTERISTIC_ID: 1992
PARENT_CONTEXT: Total - Secondary (high) school diploma or equivalency certificate for the population aged 15 years and over in private households - 25% sample data
VALUE_COLUMN: C10_RATE_TOTAL
SYMBOL_COLUMN: SYMBOL.3
```

Clean output column:

```text
pct_no_high_school_diploma
```

Validated range:

```text
min: 13.5
max: 49.8
mean: approximately 26.260
median: 26.8
```

Unit:

```text
percent
```

#### Methodological choice

The original SoVI variable is a general population education vulnerability variable. Therefore, the cleaner uses the population aged 15 years and over universe:

```text
1993 — No high school diploma or equivalency certificate
```

rather than the narrower 25-to-64 universe.

This is the preferred SoVI-compatible education variable.

### `pct_no_high_school_diploma_25_64`

Audit/sensitivity variable:

```text
CHARACTERISTIC_ID: 1996
CHARACTERISTIC_NAME: No high school diploma or equivalency certificate
PARENT_CHARACTERISTIC_ID: 1995
PARENT_CONTEXT: Total - Secondary (high) school diploma or equivalency certificate for the population aged 25 to 64 years in private households - 25% sample data
VALUE_COLUMN: C10_RATE_TOTAL
SYMBOL_COLUMN: SYMBOL.3
```

Validated range:

```text
min: 8.3
max: 43.8
mean: approximately 19.630
median: 19.75
```

Unit:

```text
percent
```

This variable is retained because the 25-to-64 universe may better capture completed adult educational attainment. It is not the default SoVI proxy.

### `pct_no_certificate_diploma_or_degree_15_plus`

Audit/sensitivity variable:

```text
CHARACTERISTIC_ID: 1999
CHARACTERISTIC_NAME: No certificate, diploma or degree
PARENT_CHARACTERISTIC_ID: 1998
PARENT_CONTEXT: Total - Highest certificate, diploma or degree for the population aged 15 years and over in private households - 25% sample data
VALUE_COLUMN: C10_RATE_TOTAL
SYMBOL_COLUMN: SYMBOL.3
```

Validated range:

```text
min: 11.2
max: 44.0
mean: approximately 23.035
median: 23.45
```

Unit:

```text
percent
```

This variable is broader than the selected no-high-school-diploma variable. It measures no certificate, diploma, or degree, not specifically no high school diploma.

### `pct_no_certificate_diploma_or_degree_25_64`

Audit/sensitivity variable:

```text
CHARACTERISTIC_ID: 2015
CHARACTERISTIC_NAME: No certificate, diploma or degree
PARENT_CHARACTERISTIC_ID: 2014
PARENT_CONTEXT: Total - Highest certificate, diploma or degree for the population aged 25 to 64 years in private households - 25% sample data
VALUE_COLUMN: C10_RATE_TOTAL
SYMBOL_COLUMN: SYMBOL.3
```

Validated range:

```text
min: 6.1
max: 36.4
mean: approximately 16.048
median: 16.25
```

Unit:

```text
percent
```

This variable is a 25-to-64 no-credential audit alternative. It is not the default SoVI proxy.

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
pct_no_high_school_diploma
pct_no_high_school_diploma_25_64
pct_no_certificate_diploma_or_degree_15_plus
pct_no_certificate_diploma_or_degree_25_64
```

Per-variable audit columns:

```text
<variable>__symbol
<variable>__is_missing
<variable>__source_characteristic_id
<variable>__source_parent_characteristic_id
<variable>__source_parent_context
<variable>__source_value_column
<variable>__unit
<variable>__sovi_role
<variable>__preferred
<variable>__methodological_choice
```

Block-level diagnostics:

```text
education_feature_count
education_preferred_feature_count
education_audit_feature_count
education_missing_count
education_complete
source
source_section
source_encoding
```

## Units

All education variables are stored as **percent values**, not proportions.

Example:

```text
26.3 means 26.3%, not 0.263
```

This is appropriate for the SoVI-like PCA workflow, because variables will later be standardized before factor analysis.

## SoVI integration

The main SoVI input mapping should be:

```text
PCTNOHSDP90 -> pct_no_high_school_diploma
```

The following variables should be treated as audit/sensitivity alternatives, not default SoVI inputs:

```text
pct_no_high_school_diploma_25_64
pct_no_certificate_diploma_or_degree_15_plus
pct_no_certificate_diploma_or_degree_25_64
```

## Interpretation warning

These are area-level census-division education indicators.

They should not be interpreted as individual-level measures.

The preferred cleaned variable is a Canadian Census Profile proxy for the original SoVI education variable, not an exact historical reproduction.

## Remaining work

None for this block.

Next recommended feature block:

```text
census_division_indigenous_identity_2021/
```

Likely target variable:

```text
PCTNATIVE90 -> pct_indigenous_identity
```