# Census Division Income Features 2021

This folder contains the cleaned Québec census-division income feature table derived from the 2021 Census Profile at the census-division level.

This feature block was created for the SoVI-like census-division index.

## Current status

This section is complete.

The cleaner successfully extracts two SoVI income-related variables for all Québec census divisions:

```text
PERCAP89   -> income_measure_default
PCTHH7589  -> pct_high_income_households
```

Validated final run:

```text
Rows: 98 Québec census divisions
income_measure_default: 98 non-missing, 0 missing
pct_high_income_households: 98 non-missing, 0 missing
```

All variables passed validation with complete coverage and no duplicated census-division rows.

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
98 unique census-division DGUIDs
0 duplicated census-division keys
```

## Encoding note

The raw Statistics Canada Census Profile file is read with:

```text
cp1252
```

The existing cleaned base frame is read with UTF-8-first encoding.

This distinction matters because the raw StatCan file and our cleaned pipeline outputs may not use the same preferred encoding. The cleaner uses:

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
inspect_census_division_income_2021.py
```

Purpose:

```text
Inspect the income section of the 2021 Census Profile at census-division level.
Identify candidate income characteristics for the SoVI-like income variables.
Validate coverage, value columns, symbols, and join compatibility.
```

SoVI variables inspected:

```text
PERCAP89   -> income proxy
PCTHH7589  -> high-income household proxy
```

Important inspection outputs:

```text
output/income_candidate_characteristics_2021.csv
output/income_target_characteristic_summary_2021.csv
output/income_target_values_long_2021.csv
output/income_target_values_wide_2021.csv
output/income_symbol_counts_2021.csv
output/income_inspection_summary_2021.csv
```

The inspection confirmed that multiple full-coverage income candidates existed. The final cleaner therefore required explicit methodological choices.

### 2. Cleaning script

```text
clean_census_division_income_2021.py
```

Purpose:

```text
Create the final clean Québec census-division income feature table.
```

Main output:

```text
output/clean_census_division_income_2021.csv
```

Audit outputs:

```text
output/clean_census_division_income_source_long_2021.csv
output/clean_census_division_income_variable_metadata_2021.csv
output/clean_census_division_income_summary_2021.csv
```

Note:

```text
The script attempts to save Parquet, but Parquet output requires pyarrow or fastparquet.
If those packages are not installed, the CSV and audit outputs are still created successfully.
```

## How to run

From the `data/` folder:

```bash
python census_division_income_2021/inspect_census_division_income_2021.py
python census_division_income_2021/clean_census_division_income_2021.py
```

## Clean variables

### `income_measure_default`

Original SoVI variable:

```text
PERCAP89
Per capita income
```

Local Census Profile proxy:

```text
CHARACTERISTIC_ID: 128
CHARACTERISTIC_NAME: Average total income in 2020 among recipients ($)
VALUE_COLUMN: C1_COUNT_TOTAL
SYMBOL_COLUMN: SYMBOL
```

Clean output column:

```text
income_measure_default
```

Readable alias also included:

```text
average_total_income_2020_recipients
```

Validated range:

```text
min: 38640.0
max: 62950.0
mean: approximately 47407.143
median: 46240.0
```

Unit:

```text
dollars
```

#### Methodological choice

The original SoVI variable `PERCAP89` is per capita income.

The 2021 Canadian Census Profile does not provide a direct per-capita-income row in the same simple way as the original SoVI variable. The inspected full-coverage person-level candidates included:

```text
113  Median total income in 2020 among recipients ($)
115  Median after-tax income in 2020 among recipients ($)
128  Average total income in 2020 among recipients ($)
130  Average after-tax income in 2020 among recipients ($)
```

The cleaner uses:

```text
128 — Average total income in 2020 among recipients ($)
```

as the default income proxy because it is the closest broad person-level income-capacity measure among the available Census Profile candidates.

This is a proxy, not an exact reproduction of the original SoVI per-capita-income variable.

### `pct_high_income_households`

Original SoVI variable:

```text
PCTHH7589
Percent households earning more than $75,000
```

Local Census Profile proxy:

```text
CHARACTERISTIC_ID: 276
CHARACTERISTIC_NAME: $100,000 and over
VALUE_COLUMN: C10_RATE_TOTAL
SYMBOL_COLUMN: SYMBOL.3
```

Clean output column:

```text
pct_high_income_households
```

Readable alias also included:

```text
pct_households_income_100k_plus
```

Validated range:

```text
min: 18.5
max: 55.9
mean: approximately 30.981
median: 28.4
```

Unit:

```text
percent
```

#### Methodological choice

The original SoVI variable uses a threshold of more than `$75,000` in 1989 U.S. dollars.

For 2021 Canadian Census Profile data, the cleaner uses:

```text
276 — $100,000 and over
```

as a modern high-income household proxy.

The inspection found several full-coverage household-income candidates, including:

```text
$100,000 and over
$100,000 to $124,999
$125,000 to $149,999
$150,000 to $199,999
$200,000 and over
```

The `$100,000 and over` household category was selected because it gives a broad modern high-income threshold that is closer in spirit to the original SoVI variable than narrower upper-income brackets.

This is a proxy, not an exact reproduction of the original 1989 threshold.

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
income_measure_default
pct_high_income_households
average_total_income_2020_recipients
pct_households_income_100k_plus
```

Per-variable audit columns:

```text
<variable>__symbol
<variable>__is_missing
<variable>__source_characteristic_id
<variable>__source_value_column
<variable>__unit
<variable>__sovi_role
<variable>__methodological_choice
```

Block-level diagnostics:

```text
income_feature_count
income_missing_count
income_complete
source
source_section
source_encoding
```

## Units

`income_measure_default` is stored in dollars.

`pct_high_income_households` is stored as a **percent value**, not a proportion.

Example:

```text
30.98 means 30.98%, not 0.3098
```

This is appropriate for the SoVI-like PCA workflow, because variables will later be standardized before factor analysis.

## SoVI integration

These two variables can be mapped into the SoVI input table as:

```text
PERCAP89   -> income_measure_default
PCTHH7589  -> pct_high_income_households
```

They should be treated as:

```text
local_income_proxy
modern_high_income_household_proxy
```

respectively.

## Interpretation warning

These are area-level census-division income indicators.

They should not be interpreted as individual-level measures.

The two cleaned variables are proxies for original SoVI variables, not exact historical reproductions.

## Remaining work

None for this block.

Next recommended feature block:

```text
census_division_low_income_2021/
```

Likely target variable:

```text
PCTPOV90 -> pct_poverty_or_low_income
```