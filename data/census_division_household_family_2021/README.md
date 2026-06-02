# Census Division Household / Family Features 2021

This folder contains the cleaned Québec census-division household/family feature table derived from the 2021 Census Profile at the census-division level.

This feature block was created for the SoVI-like census-division index.

## Current status

This section is partially complete.

The cleaner successfully extracts one validated SoVI household variable for all Québec census divisions:

```text
AVGPERHH -> avg_people_per_household
```

One SoVI family/household variable remains unresolved:

```text
PCTF_HH90 -> pct_female_headed_households
```

The unresolved variable is **not included as a numeric feature** in the clean table.

Validated final run:

```text
Rows: 98 Québec census divisions
avg_people_per_household: 98 non-missing, 0 missing
pct_female_headed_households: unresolved, not cleaned as numeric feature
```

The final run confirmed:

```text
clean_rows: 98
unique_census_divisions: 98
all_cleaned_variables_complete: True
pct_female_headed_households_included_as_numeric_feature: False
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

### 1. General inspection script

```text
inspect_census_division_household_family_2021.py
```

Purpose:

```text
Inspect the household/family section of the 2021 Census Profile at census-division level.
Identify candidates for average household size and female-headed-household proxies.
Validate candidate coverage, value columns, symbols, and join compatibility.
```

SoVI variables inspected:

```text
AVGPERHH  -> avg_people_per_household
PCTF_HH90 -> pct_female_headed_households
```

Important inspection outputs:

```text
output/household_family_candidate_characteristics_2021.csv
output/household_family_target_characteristic_summary_2021.csv
output/household_family_target_values_long_2021.csv
output/household_family_target_values_wide_2021.csv
output/household_family_derived_formula_audit_2021.csv
output/household_family_symbol_counts_2021.csv
output/household_family_inspection_summary_2021.csv
```

The inspection confirmed that `avg_people_per_household` was available and complete, but did not find a defensible female-headed-household proxy.

### 2. Targeted female-headed-household proxy inspection

```text
inspect_census_division_female_headed_household_proxy_2021.py
```

Purpose:

```text
Run a broader targeted search for possible Canadian Census Profile proxies for PCTF_HH90.
Search for rows involving female parent, women household maintainers, lone-parent families, household maintainers, and possible denominators.
```

Important targeted inspection outputs:

```text
output/female_headed_household_proxy_all_keyword_matches_2021.csv
output/female_headed_household_proxy_characteristic_summary_2021.csv
output/female_headed_household_proxy_candidate_classification_2021.csv
output/female_headed_household_proxy_target_values_long_2021.csv
output/female_headed_household_proxy_target_values_wide_2021.csv
output/female_headed_household_proxy_derived_formula_audit_2021.csv
output/female_headed_household_proxy_symbol_counts_2021.csv
output/female_headed_household_proxy_inspection_summary_2021.csv
```

The targeted inspection scanned the CD Census Profile successfully:

```text
total_rows_scanned: 770883
quebec_cd_rows_scanned: 257838
unique_quebec_census_divisions: 98
matched_base_cd_rows: 98
keyword_match_rows: 4214
candidate_characteristics_found: 43
primary_candidate_characteristics_found: 25
```

However, the derived formula audit did not confirm a usable numerator:

```text
published_female_parent_lone_parent_rate: unavailable
published_female_or_women_household_maintainer_rate: unavailable
100 * female_parent_lone_parent_family_count / total_private_households: unavailable
100 * female_parent_lone_parent_family_count / total_lone_parent_families: unavailable
```

The inspection found possible household denominator rows, but no complete female-parent or women-maintainer numerator suitable for a defensible numeric proxy.

### 3. Cleaning script

```text
clean_census_division_household_family_2021.py
```

Purpose:

```text
Create the final clean Québec census-division household/family feature table using only validated numeric features.
```

Main output:

```text
output/clean_census_division_household_family_2021.csv
```

Audit outputs:

```text
output/clean_census_division_household_family_source_long_2021.csv
output/clean_census_division_household_family_variable_metadata_2021.csv
output/clean_census_division_household_family_unresolved_variables_2021.csv
output/clean_census_division_household_family_summary_2021.csv
```

Note:

```text
The script attempts to save Parquet, but Parquet output requires pyarrow or fastparquet.
If those packages are not installed, the CSV and audit outputs are still created successfully.
```

## How to run

From the `data/` folder:

```bash
python census_division_household_family_2021/inspect_census_division_household_family_2021.py
python census_division_household_family_2021/inspect_census_division_female_headed_household_proxy_2021.py
python census_division_household_family_2021/clean_census_division_household_family_2021.py
```

## Clean variable

### `avg_people_per_household`

Original SoVI variable:

```text
AVGPERHH
Average number of persons per household
```

Local Census Profile source:

```text
CHARACTERISTIC_ID: 57
CHARACTERISTIC_NAME: Average household size
VALUE_COLUMN: C1_COUNT_TOTAL
SYMBOL_COLUMN: SYMBOL
```

Clean output column:

```text
avg_people_per_household
```

Readable alias also included:

```text
avg_household_size
```

Validated range:

```text
min: 1.9
max: 3.1
mean: approximately 2.205
median: 2.2
```

Unit:

```text
persons_per_household
```

## Unresolved variable

### `pct_female_headed_households`

Original SoVI variable:

```text
PCTF_HH90
Percent female-headed households
```

Current status:

```text
unresolved_not_cleaned
```

This variable is **not included as a numeric feature** in the clean table.

Reason:

```text
The broad household/family inspection and targeted female-headed-household proxy inspection did not find a defensible numerator in the CD-level 2021 Census Profile file.
```

The inspections found several possible private-household denominator rows, including 100% and 25% sample data denominator candidates, but did not find a complete female-parent or women household-maintainer numerator suitable for a clean proxy.

Therefore, the cleaner records unresolved metadata columns:

```text
pct_female_headed_households__status
pct_female_headed_households__included_as_numeric_feature
pct_female_headed_households__reason
```

but does not create:

```text
pct_female_headed_households
```

as a numeric feature.

## Methodological choice

The clean table is intentionally conservative.

Rather than forcing a weak or misleading proxy for `PCTF_HH90`, this section only includes the validated household-size variable and documents the unresolved variable separately.

This prevents the SoVI input table from silently treating an unsupported proxy as if it were equivalent to the original female-headed-household variable.

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
avg_people_per_household
avg_household_size
```

Per-variable audit columns for `avg_people_per_household`:

```text
avg_people_per_household__symbol
avg_people_per_household__is_missing
avg_people_per_household__source_characteristic_id
avg_people_per_household__source_value_column
avg_people_per_household__unit
avg_people_per_household__sovi_role
avg_people_per_household__methodological_choice
```

Unresolved-variable metadata columns:

```text
pct_female_headed_households__status
pct_female_headed_households__included_as_numeric_feature
pct_female_headed_households__reason
```

Block-level diagnostics:

```text
household_family_feature_count
household_family_unresolved_feature_count
household_family_missing_count
household_family_complete
source
source_section
source_encoding
```

## SoVI integration

The current SoVI input mapping should include:

```text
AVGPERHH -> avg_people_per_household
```

The current SoVI input mapping should **not** include a numeric value for:

```text
PCTF_HH90 -> pct_female_headed_households
```

unless a later source provides a defensible proxy.

## Interpretation warning

This is an area-level census-division household indicator.

It should not be interpreted as an individual-level measure.

The unresolved female-headed-household variable should not be imputed from unrelated variables unless the methodology explicitly accepts that proxy.

## Remaining work

The validated work in this block is done.

Unresolved item:

```text
PCTF_HH90 -> pct_female_headed_households
```

Possible future sources to inspect:

```text
custom Census family tables
household-maintainer tables
family-composition tables
additional StatCan cross-tabulations
```

Next recommended feature block:

```text
census_division_education_2021/
```

Likely target variable:

```text
PCTNOHSDP90 -> pct_no_high_school_diploma
```