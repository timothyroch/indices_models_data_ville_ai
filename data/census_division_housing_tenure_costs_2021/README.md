# Census Division Housing Tenure/Costs Features 2021

This folder contains the cleaned Québec census-division housing tenure and housing-cost feature table derived from the 2021 Census Profile at the census-division level.

This feature block was created for the SoVI-like census-division index.

## Current status

This section is complete.

The cleaner successfully extracts three SoVI housing tenure/cost variables for all Québec census divisions:

```text
PCTRENTER90 -> pct_renter_occupied
MEDRENT90   -> median_rent
MVALOO90    -> median_owner_occupied_housing_value
```

Validated final run:

```text
Rows: 98 Québec census divisions
pct_renter_occupied: 98 non-missing, 0 missing
median_rent: 98 non-missing, 0 missing
median_owner_occupied_housing_value: 98 non-missing, 0 missing
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
inspect_census_division_housing_tenure_costs_2021.py
```

Purpose:

```text
Inspect the housing tenure and housing-cost section of the 2021 Census Profile at census-division level.
Identify candidate characteristics for renter tenure, median rent, and owner-occupied housing value.
Validate coverage, value columns, symbols, and join compatibility.
```

SoVI variables inspected:

```text
PCTRENTER90 -> pct_renter_occupied
MEDRENT90   -> median_rent
MVALOO90    -> median_owner_occupied_housing_value
```

Important inspection outputs:

```text
output/housing_tenure_costs_candidate_characteristics_2021.csv
output/housing_tenure_costs_target_characteristic_summary_2021.csv
output/housing_tenure_costs_target_values_long_2021.csv
output/housing_tenure_costs_target_values_wide_2021.csv
output/housing_tenure_costs_symbol_counts_2021.csv
output/housing_tenure_costs_inspection_summary_2021.csv
```

The inspection found full-coverage candidates for all three target variables.

### 2. Cleaning script

```text
clean_census_division_housing_tenure_costs_2021.py
```

Purpose:

```text
Create the final clean Québec census-division housing tenure/costs feature table.
```

Main output:

```text
output/clean_census_division_housing_tenure_costs_2021.csv
```

Audit outputs:

```text
output/clean_census_division_housing_tenure_costs_source_long_2021.csv
output/clean_census_division_housing_tenure_costs_variable_metadata_2021.csv
output/clean_census_division_housing_tenure_costs_summary_2021.csv
```

Note:

```text
The script attempts to save Parquet, but Parquet output requires pyarrow or fastparquet.
If those packages are not installed, the CSV and audit outputs are still created successfully.
```

## How to run

From the `data/` folder:

```bash
python census_division_housing_tenure_costs_2021/inspect_census_division_housing_tenure_costs_2021.py
python census_division_housing_tenure_costs_2021/clean_census_division_housing_tenure_costs_2021.py
```

## Clean variables

### `pct_renter_occupied`

Original SoVI variable:

```text
PCTRENTER90
Percent renter-occupied housing units
```

Local Census Profile proxy:

```text
CHARACTERISTIC_ID: 1416
CHARACTERISTIC_NAME: Renter
VALUE_COLUMN: C10_RATE_TOTAL
SYMBOL_COLUMN: SYMBOL.3
```

Clean output column:

```text
pct_renter_occupied
```

Readable alias also included:

```text
pct_renter_households
```

Validated range:

```text
min: 10.8
max: 60.4
mean: approximately 28.879
median: 26.85
```

Unit:

```text
percent
```

#### Methodological choice

The inspection found several rows containing the word `tenant`, but only one row corresponds to the broad renter tenure variable needed for SoVI:

```text
1416 — Renter
```

Other rows matched by the inspection were rejected:

```text
1465 — Total - Owner and tenant households with household total income greater than zero...
1479 — Total - Owner and tenant households with household total income greater than zero and shelter-cost-to-income ratio less than 100%...
1490 — Total - Tenant households in non-farm, non-reserve private dwellings...
```

Rows `1465` and `1479` are denominator/universe rows with rates of 100% and are not meaningful as renter-tenure shares. Row `1490` had no usable rate values in `C10_RATE_TOTAL`.

Therefore, the cleaner uses:

```text
1416 — Renter
```

as the selected source for `pct_renter_occupied`.

### `median_rent`

Original SoVI variable:

```text
MEDRENT90
Median rent
```

Local Census Profile proxy:

```text
CHARACTERISTIC_ID: 1494
CHARACTERISTIC_NAME: Median monthly shelter costs for rented dwellings ($)
VALUE_COLUMN: C1_COUNT_TOTAL
SYMBOL_COLUMN: SYMBOL
```

Clean output column:

```text
median_rent
```

Readable alias also included:

```text
median_monthly_shelter_costs_rented
```

Validated range:

```text
min: 464.0
max: 980.0
mean: approximately 670.929
median: 636.0
```

Unit:

```text
dollars
```

#### Methodological choice

The original SoVI variable uses median rent. The Canadian Census Profile proxy uses:

```text
Median monthly shelter costs for rented dwellings
```

This is a shelter-cost proxy for median rent and is the closest available Census Profile equivalent found in the inspection.

### `median_owner_occupied_housing_value`

Original SoVI variable:

```text
MVALOO90
Median dollar value of owner-occupied housing
```

Local Census Profile proxy:

```text
CHARACTERISTIC_ID: 1488
CHARACTERISTIC_NAME: Median value of dwellings ($)
VALUE_COLUMN: C1_COUNT_TOTAL
SYMBOL_COLUMN: SYMBOL
```

Clean output column:

```text
median_owner_occupied_housing_value
```

Readable alias also included:

```text
median_value_of_dwellings
```

Validated range:

```text
min: 120000.0
max: 544000.0
mean: approximately 238632.653
median: 220000.0
```

Unit:

```text
dollars
```

#### Methodological choice

The original SoVI variable uses the median dollar value of owner-occupied housing.

The Canadian Census Profile proxy uses:

```text
Median value of dwellings
```

This is treated as the owner-housing-value proxy for the SoVI-like census-division table.

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
pct_renter_occupied
median_rent
median_owner_occupied_housing_value
pct_renter_households
median_monthly_shelter_costs_rented
median_value_of_dwellings
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
housing_tenure_costs_feature_count
housing_tenure_costs_missing_count
housing_tenure_costs_complete
source
source_section
source_encoding
```

## Units

`pct_renter_occupied` is stored as a **percent value**, not a proportion.

Example:

```text
28.9 means 28.9%, not 0.289
```

`median_rent` and `median_owner_occupied_housing_value` are stored in dollars.

This is appropriate for the SoVI-like PCA workflow, because variables will later be standardized before factor analysis.

## SoVI integration

These variables can be mapped into the SoVI input table as:

```text
PCTRENTER90 -> pct_renter_occupied
MEDRENT90   -> median_rent
MVALOO90    -> median_owner_occupied_housing_value
```

They should be treated as:

```text
renter_tenure_proxy
renter_housing_cost_proxy
owner_housing_value_proxy
```

respectively.

## Interpretation warning

These are area-level census-division housing indicators.

They should not be interpreted as individual-level measures.

The cleaned variables are Canadian Census Profile proxies for the original SoVI housing variables, not exact historical reproductions.

## Remaining work

None for this block.

Next recommended feature block:

```text
census_division_housing_type_2021/
```

Likely target variable:

```text
PCTMOBL90 -> pct_mobile_homes
```