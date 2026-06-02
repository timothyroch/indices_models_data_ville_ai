# Census Division Labour Force Features 2021

This folder contains the cleaned Québec census-division labour-force feature table derived from the 2021 Census Profile at the census-division level.

This feature block was created for the SoVI-like census-division index.

## Current status

This section is complete.

The cleaner successfully extracts three SoVI labour-force variables for all Québec census divisions:

```text
PCTVLUN91  -> pct_unemployed
CVBRPC91   -> labor_force_participation_rate
FEMLBR90   -> female_labor_force_participation_rate
```

Validated final run:

```text
Rows: 98 Québec census divisions
pct_unemployed: 98 non-missing, 0 missing
labor_force_participation_rate: 98 non-missing, 0 missing
female_labor_force_participation_rate: 98 non-missing, 0 missing
```

All three variables passed validation with complete coverage and no duplicated census-division rows. The final run also confirmed that geography names no longer contain mojibake/encoding corruption. :contentReference[oaicite:0]{index=0}

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
0 unmatched base rows
0 profile-only rows
```

## Encoding note

The raw Statistics Canada Census Profile file must be read with:

```text
cp1252
```

The existing cleaned base frame should be read with UTF-8-first encoding.

This distinction matters. If the cleaned base frame is read as `cp1252`, accented Québec place names can become corrupted, for example:

```text
La CÃ´te-de-GaspÃ©
```

The final cleaner handles this by using separate encoding priorities:

```text
Raw StatCan file: cp1252 first
Cleaned pipeline files: utf-8 first
```

The final run reported:

```text
Base names with possible mojibake: 0
Profile names with possible mojibake: 0
Display names with possible mojibake: 0
```

## Scripts

### 1. Inspection script

```text
inspect_census_division_labour_force_2021.py
```

Purpose:

```text
Inspect the labour-force section of the 2021 Census Profile at census-division level.
Validate target characteristic IDs, value columns, symbols, coverage, and join compatibility.
```

SoVI variables inspected:

```text
PCTVLUN91  -> pct_unemployed
CVBRPC91   -> labor_force_participation_rate
FEMLBR90   -> female_labor_force_participation_rate
```

Important inspection outputs:

```text
output/labour_force_candidate_characteristics_2021.csv
output/labour_force_target_characteristic_summary_2021.csv
output/labour_force_target_values_long_2021.csv
output/labour_force_target_values_wide_2021.csv
output/labour_force_symbol_counts_2021.csv
output/labour_force_inspection_summary_2021.csv
```

The inspection confirmed that all three target variables were ready for the cleaner.

### 2. Cleaning script

```text
clean_census_division_labour_force_2021.py
```

Purpose:

```text
Create the final clean Québec census-division labour-force feature table.
```

Main outputs:

```text
output/clean_census_division_labour_force_2021.csv
output/clean_census_division_labour_force_2021.parquet
```

Audit outputs:

```text
output/clean_census_division_labour_force_source_long_2021.csv
output/clean_census_division_labour_force_variable_metadata_2021.csv
output/clean_census_division_labour_force_summary_2021.csv
```

## How to run

From the `data/` folder:

```bash
python census_division_labour_force_2021/inspect_census_division_labour_force_2021.py
python census_division_labour_force_2021/clean_census_division_labour_force_2021.py
```

## Clean variables

### `pct_unemployed`

Original SoVI variable:

```text
PCTVLUN91
Percent civilian labour force unemployed
```

Census Profile source:

```text
CHARACTERISTIC_ID: 2230
CHARACTERISTIC_NAME: Unemployment rate
VALUE_COLUMN: C10_RATE_TOTAL
SYMBOL_COLUMN: SYMBOL.3
```

Clean output column:

```text
pct_unemployed
```

Validated range:

```text
min: 4.0
max: 13.8
mean: approximately 6.995
```

### `labor_force_participation_rate`

Original SoVI variable:

```text
CVBRPC91
Percent population participating in labour force
```

Census Profile source:

```text
CHARACTERISTIC_ID: 2228
CHARACTERISTIC_NAME: Participation rate
VALUE_COLUMN: C10_RATE_TOTAL
SYMBOL_COLUMN: SYMBOL.3
```

Clean output column:

```text
labor_force_participation_rate
```

Validated range:

```text
min: 46.2
max: 75.2
mean: approximately 60.835
```

### `female_labor_force_participation_rate`

Original SoVI variable:

```text
FEMLBR90
Percent females participating in civilian labour force
```

Census Profile source:

```text
CHARACTERISTIC_ID: 2228
CHARACTERISTIC_NAME: Participation rate
VALUE_COLUMN: C12_RATE_WOMEN+
SYMBOL_COLUMN: SYMBOL.5
```

Clean output column:

```text
female_labor_force_participation_rate
```

Validated range:

```text
min: 46.1
max: 74.2
mean: approximately 57.992
```

Note:

```text
This variable uses the Women+ rate column from the same Participation rate characteristic.
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
pct_unemployed
labor_force_participation_rate
female_labor_force_participation_rate
```

Per-variable audit columns:

```text
<variable>__symbol
<variable>__is_missing
<variable>__source_characteristic_id
<variable>__source_value_column
<variable>__unit
<variable>__sovi_role
```

Block-level diagnostics:

```text
labour_force_feature_count
labour_force_missing_count
labour_force_complete
source
source_section
source_encoding
```

## Units

The three cleaned feature columns are stored as **percent values**, not proportions.

Example:

```text
6.5 means 6.5%, not 0.065
```

This is appropriate for the SoVI-like PCA workflow, because variables will later be standardized before factor analysis.

## SoVI integration

These three variables can be mapped into the SoVI input table as:

```text
PCTVLUN91  -> pct_unemployed
CVBRPC91   -> labor_force_participation_rate
FEMLBR90   -> female_labor_force_participation_rate
```

They should be treated as:

```text
direct_or_strong_proxy
```

except for `female_labor_force_participation_rate`, which is a sex-specific rate extracted from the Women+ rate column of the participation-rate characteristic.

## Interpretation warning

These are area-level census-division labour-force indicators.

They should not be interpreted as individual-level measures.