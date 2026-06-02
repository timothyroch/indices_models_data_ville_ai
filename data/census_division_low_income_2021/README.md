# Census Division Low-Income Features 2021

This folder contains the cleaned Québec census-division low-income feature table derived from the 2021 Census Profile at the census-division level.

This feature block was created for the SoVI-like census-division index.

## Current status

This section is complete.

The cleaner successfully extracts the main SoVI poverty/low-income proxy and one alternative audit variable for all Québec census divisions:

```text
PCTPOV90 -> pct_poverty_or_low_income
ALT      -> pct_low_income_lico_at
```

Validated final run:

```text
Rows: 98 Québec census divisions
pct_poverty_or_low_income: 98 non-missing, 0 missing
pct_low_income_lico_at: 98 non-missing, 0 missing
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
inspect_census_division_low_income_2021.py
```

Purpose:

```text
Inspect the low-income section of the 2021 Census Profile at census-division level.
Validate LIM-AT and LICO-AT candidate characteristics, value columns, symbols, coverage, and join compatibility.
```

SoVI variable inspected:

```text
PCTPOV90 -> pct_poverty_or_low_income
```

Candidate measures inspected:

```text
LIM-AT prevalence
LICO-AT prevalence
```

Important inspection outputs:

```text
output/low_income_candidate_characteristics_2021.csv
output/low_income_target_characteristic_summary_2021.csv
output/low_income_target_values_long_2021.csv
output/low_income_target_values_wide_2021.csv
output/low_income_symbol_counts_2021.csv
output/low_income_inspection_summary_2021.csv
```

The inspection confirmed that both LIM-AT and LICO-AT candidates had complete coverage for all 98 Québec census divisions.

### 2. Cleaning script

```text
clean_census_division_low_income_2021.py
```

Purpose:

```text
Create the final clean Québec census-division low-income feature table.
```

Main output:

```text
output/clean_census_division_low_income_2021.csv
```

Audit outputs:

```text
output/clean_census_division_low_income_source_long_2021.csv
output/clean_census_division_low_income_variable_metadata_2021.csv
output/clean_census_division_low_income_summary_2021.csv
```

Note:

```text
The script attempts to save Parquet, but Parquet output requires pyarrow or fastparquet.
If those packages are not installed, the CSV and audit outputs are still created successfully.
```

## How to run

From the `data/` folder:

```bash
python census_division_low_income_2021/inspect_census_division_low_income_2021.py
python census_division_low_income_2021/clean_census_division_low_income_2021.py
```

## Clean variables

### `pct_poverty_or_low_income`

Original SoVI variable:

```text
PCTPOV90
Percent persons in poverty
```

Local Census Profile proxy:

```text
CHARACTERISTIC_ID: 345
CHARACTERISTIC_NAME: Prevalence of low income based on the Low-income measure, after tax (LIM-AT) (%)
VALUE_COLUMN: C10_RATE_TOTAL
SYMBOL_COLUMN: SYMBOL.3
```

Clean output column:

```text
pct_poverty_or_low_income
```

Readable alias also included:

```text
pct_low_income_lim_at
```

Validated range:

```text
min: 4.0
max: 19.3
mean: approximately 12.299
median: 12.35
```

Unit:

```text
percent
```

#### Methodological choice

The original SoVI variable `PCTPOV90` uses a U.S. poverty concept.

The Canadian Census Profile does not use the same poverty definition. For the Québec 2021 census-division SoVI-like table, this cleaner uses:

```text
Prevalence of low income based on LIM-AT
```

as the default Canadian low-income / poverty proxy.

This is a proxy, not an exact reproduction of the original U.S. poverty variable.

### `pct_low_income_lico_at`

Alternative audit variable:

```text
CHARACTERISTIC_ID: 360
CHARACTERISTIC_NAME: Prevalence of low income based on the Low-income cut-offs, after tax (LICO-AT) (%)
VALUE_COLUMN: C10_RATE_TOTAL
SYMBOL_COLUMN: SYMBOL.3
```

Clean output column:

```text
pct_low_income_lico_at
```

Validated range:

```text
min: 0.7
max: 10.9
mean: approximately 2.756
median: 2.5
```

Unit:

```text
percent
```

#### Interpretation

This variable is kept for audit and possible sensitivity analysis.

It is **not** the preferred SoVI poverty input unless a later methodology document explicitly chooses LICO-AT instead of LIM-AT.

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
pct_poverty_or_low_income
pct_low_income_lim_at
pct_low_income_lico_at
```

Per-variable audit columns:

```text
<variable>__symbol
<variable>__is_missing
<variable>__source_characteristic_id
<variable>__source_value_column
<variable>__unit
<variable>__sovi_role
<variable>__preferred
<variable>__methodological_choice
```

Block-level diagnostics:

```text
low_income_feature_count
low_income_missing_count
low_income_complete
source
source_section
source_encoding
```

## Units

Both low-income columns are stored as **percent values**, not proportions.

Example:

```text
12.3 means 12.3%, not 0.123
```

This is appropriate for the SoVI-like PCA workflow, because variables will later be standardized before factor analysis.

## SoVI integration

The main SoVI input mapping should be:

```text
PCTPOV90 -> pct_poverty_or_low_income
```

The variable should be treated as:

```text
canadian_low_income_proxy_for_poverty
```

The alternative LICO-AT column can be kept for audit or sensitivity analysis:

```text
pct_low_income_lico_at
```

but it should not replace the preferred LIM-AT variable unless explicitly justified later.

## Interpretation warning

These are area-level census-division low-income indicators.

They should not be interpreted as individual-level measures.

The preferred cleaned variable is a Canadian low-income proxy for the original SoVI poverty variable, not an exact historical reproduction.

## Remaining work

None for this block.

Next recommended feature block:

```text
census_division_housing_tenure_costs_2021/
```

Likely target variables:

```text
PCTRENTER90 -> pct_renter_occupied
MEDRENT90   -> median_rent
```