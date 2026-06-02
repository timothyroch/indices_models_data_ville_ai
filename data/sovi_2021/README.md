# SoVI 2021 Census-Division Input Table

This folder contains the current 2021 Québec census-division input table work for the SoVI-like index.

The purpose of this section is to assemble a canonical SoVI-style feature table with one row per Québec census division and one column per expected SoVI variable.

This README documents the current state of the work before adding the next missing-variable blocks.

## Current status

The current SoVI input table is partially built.

The latest inspection successfully created a draft SoVI input table with:

```text
98 Québec census divisions
42 expected SoVI variables
19 variables currently filled or partially filled
23 variables still missing / unmapped
```

Latest validated inspection summary:

```text
Expected SoVI variables: 42
Base rows: 98
Variables ready for draft table: 19
  Full coverage: 18
  Partial documented coverage: 1
Variables not ready / unmapped: 23
```

The current draft table is:

```text
sovi_2021/output/draft_clean_census_division_sovi_input_2021.csv
```

This table is an inspection/draft artifact, not yet the final clean SoVI input table.

## Main script

The current main inspection script is:

```text
sovi_2021/inspect_census_division_sovi_input_sources_2021.py
```

Run from the `data/` folder:

```bash
python sovi_2021/inspect_census_division_sovi_input_sources_2021.py
```

This script:

```text
1. Defines the 42 expected SoVI-like canonical variables.
2. Loads the Québec census-division base frame.
3. Checks all completed source blocks.
4. Joins available variables into a draft SoVI-wide table.
5. Marks missing, unresolved, and partially documented variables.
6. Writes source-audit and missing-variable reports.
```

## Base geography

The SoVI draft table uses the reusable Québec census-division base frame:

```text
census_division_spatial_frame_population_2021/output/clean_quebec_census_division_spatial_frame_with_population_2021.csv
```

Validated geography:

```text
98 Québec census divisions
98 unique census-division DGUIDs
```

The base frame includes useful population/geography columns such as:

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
population_total_2016
population_change_pct_2016_2021
land_area_km2
population_density_per_km2_2021
has_positive_population
```

## Current output files

The inspection script writes:

```text
sovi_2021/output/census_division_sovi_expected_columns_2021.csv
sovi_2021/output/census_division_sovi_source_file_inventory_2021.csv
sovi_2021/output/census_division_sovi_input_source_audit_2021.csv
sovi_2021/output/census_division_sovi_missing_or_unresolved_variables_2021.csv
sovi_2021/output/draft_clean_census_division_sovi_input_2021.csv
sovi_2021/output/census_division_sovi_input_source_inspection_summary_2021.csv
```

### `census_division_sovi_expected_columns_2021.csv`

Lists the 42 expected SoVI variables, their original SoVI codes, descriptions, expected units, and current mapping status.

### `census_division_sovi_source_file_inventory_2021.csv`

Lists each mapped source file that the inspection script attempted to use.

### `census_division_sovi_input_source_audit_2021.csv`

Main audit file.

For each expected variable, it records:

```text
source folder
resolved source file
selected source column
matched rows
missing rows
coverage status
proxy quality
methodological note
whether it was inserted into the draft table
```

### `census_division_sovi_missing_or_unresolved_variables_2021.csv`

Subset of the audit table containing variables that are not yet ready.

### `draft_clean_census_division_sovi_input_2021.csv`

Current draft SoVI-wide feature table.

It contains:

```text
98 rows
identity/geography columns
42 SoVI variable columns
```

Currently, 19 variables are filled or partially filled.

### `census_division_sovi_input_source_inspection_summary_2021.csv`

Compact summary of the inspection run.

## Variables currently ready

The following variables are currently inserted into the draft SoVI table.

### Full-coverage variables

These 18 variables have 98 / 98 non-missing values:

```text
med_age
per_capita_income
median_home_value
median_rent
pct_under_5
pct_over_65
pct_unemployed
avg_people_per_household
pct_high_income_households
pct_poverty
pct_renter
pct_mobile_homes
pct_no_high_school
labor_force_participation
female_labor_force_participation
nursing_home_residents_per_capita
hospitals_per_capita
pct_female
```

### Partial documented variable

This variable is inserted into the draft table with a documented missing value:

```text
physicians_per_100k
```

Coverage:

```text
97 non-missing
1 missing
```

The missing census division is expected:

```text
Nord-du-Québec
```

This is because the physician source is CIHI health-region-native, while Nord-du-Québec as a census division is too coarse relative to the northern CIHI health-region split.

## Current ready-variable mapping

### `med_age`

Original SoVI code:

```text
MED_AGE90
```

Source:

```text
census_division_age_structure_2021/output/clean_census_division_age_structure_2021.csv
```

Selected source column:

```text
median_age
```

Proxy quality:

```text
direct_or_strong_proxy
```

### `per_capita_income`

Original SoVI code:

```text
PERCAP89
```

Source:

```text
census_division_income_2021/output/clean_census_division_income_2021.csv
```

Selected source column:

```text
income_measure_default
```

Proxy quality:

```text
income_proxy_needs_method_note
```

Methodological note:

```text
This is the selected Canadian income proxy from the income cleaner. It should not be treated as an exact historical reproduction of the original U.S. per-capita-income variable without a method note.
```

### `median_home_value`

Original SoVI code:

```text
MVALOO90
```

Source:

```text
census_division_housing_tenure_costs_2021/output/clean_census_division_housing_tenure_costs_2021.csv
```

Selected source column:

```text
median_owner_occupied_housing_value
```

Proxy quality:

```text
direct_or_strong_proxy
```

### `median_rent`

Original SoVI code:

```text
MEDRENT90
```

Source:

```text
census_division_housing_tenure_costs_2021/output/clean_census_division_housing_tenure_costs_2021.csv
```

Selected source column:

```text
median_rent
```

Proxy quality:

```text
direct_or_strong_proxy
```

### `physicians_per_100k`

Original SoVI code:

```text
PHYSICN90
```

Source:

```text
doctors_per_100khabs/output/clean_census_division_doctors_per_100k_proxy_2024.csv
```

Selected source column:

```text
physicians_per_100k
```

Coverage:

```text
97 / 98 non-missing
```

Proxy quality:

```text
health_region_proxy_one_documented_missing
```

Methodological note:

```text
This is a CIHI health-region physician-rate proxy assigned to census divisions through a crosswalk. It is not a direct census-division-native measurement. Nord-du-Québec remains missing because it is unresolved in the health-region crosswalk.
```

### `pct_under_5`

Original SoVI code:

```text
PCTKIDS90
```

Source:

```text
census_division_age_structure_2021/output/clean_census_division_age_structure_2021.csv
```

Selected source column:

```text
pct_under_5
```

Proxy quality:

```text
direct_or_strong_proxy
```

### `pct_over_65`

Original SoVI code:

```text
PCTOLD90
```

Source:

```text
census_division_age_structure_2021/output/clean_census_division_age_structure_2021.csv
```

Selected source column:

```text
pct_over_65
```

Proxy quality:

```text
direct_or_strong_proxy
```

### `pct_unemployed`

Original SoVI code:

```text
PCTVLUN91
```

Source:

```text
census_division_labour_force_2021/output/clean_census_division_labour_force_2021.csv
```

Selected source column:

```text
pct_unemployed
```

Proxy quality:

```text
direct_or_strong_proxy
```

### `avg_people_per_household`

Original SoVI code:

```text
AVGPERHH
```

Source:

```text
census_division_household_family_2021/output/clean_census_division_household_family_2021.csv
```

Selected source column:

```text
avg_people_per_household
```

Proxy quality:

```text
direct_or_strong_proxy
```

### `pct_high_income_households`

Original SoVI code:

```text
PCTHH7589
```

Source:

```text
census_division_income_2021/output/clean_census_division_income_2021.csv
```

Selected source column:

```text
pct_high_income_households
```

Proxy quality:

```text
threshold_proxy
```

### `pct_poverty`

Original SoVI code:

```text
PCTPOV90
```

Source:

```text
census_division_low_income_2021/output/clean_census_division_low_income_2021.csv
```

Selected source column:

```text
pct_poverty_or_low_income
```

Proxy quality:

```text
canadian_low_income_proxy
```

Methodological note:

```text
Uses LIM-AT low-income prevalence as the Canadian Census Profile proxy for the original SoVI poverty variable.
```

### `pct_renter`

Original SoVI code:

```text
PCTRENTER90
```

Source:

```text
census_division_housing_tenure_costs_2021/output/clean_census_division_housing_tenure_costs_2021.csv
```

Selected source column:

```text
pct_renter_occupied
```

Proxy quality:

```text
direct_or_strong_proxy
```

### `pct_mobile_homes`

Original SoVI code:

```text
PCTMOBL90
```

Source:

```text
census_division_housing_type_2021/output/clean_census_division_housing_type_2021.csv
```

Selected source column:

```text
pct_mobile_homes
```

Proxy quality:

```text
movable_dwelling_proxy
```

Methodological note:

```text
Uses the Canadian Census Profile category “Movable dwelling” as the proxy for the original SoVI mobile-homes variable.
```

### `pct_no_high_school`

Original SoVI code:

```text
PCTNOHS90
```

Source:

```text
census_division_education_2021/output/clean_census_division_education_2021.csv
```

Selected source column:

```text
pct_no_high_school_diploma_25_64
```

Proxy quality:

```text
age_universe_proxy
```

Methodological note:

```text
The draft table uses the 25–64 no-high-school variable because the SoVI/YAML concept is age-25-plus-oriented. This is a 25–64 proxy, not an exact 25+ reproduction.
```

### `labor_force_participation`

Original SoVI code:

```text
CVBRPC91
```

Source:

```text
census_division_labour_force_2021/output/clean_census_division_labour_force_2021.csv
```

Selected source column:

```text
labor_force_participation_rate
```

Proxy quality:

```text
direct_or_strong_proxy
```

### `female_labor_force_participation`

Original SoVI code:

```text
FEMLBR90
```

Source:

```text
census_division_labour_force_2021/output/clean_census_division_labour_force_2021.csv
```

Selected source column:

```text
female_labor_force_participation_rate
```

Proxy quality:

```text
direct_or_strong_proxy
```

### `nursing_home_residents_per_capita`

Original SoVI code:

```text
NRRESPC91
```

Source:

```text
residential_care_per_capita/output/clean_census_division_residential_care_per_100k_population_odhf_2021.csv
```

Selected source column:

```text
residential_care_facilities_per_100k_population_odhf
```

Proxy quality:

```text
facility_density_proxy_not_residents
```

Methodological note:

```text
This is an ODHF residential-care facility-density proxy, not literal nursing-home residents per capita.
```

### `hospitals_per_capita`

Original SoVI code:

```text
HOSPTPC91
```

Source:

```text
hospitals_per_capita/output/clean_census_division_hospitals_per_100k_population_odhf_2021.csv
```

Selected source column:

```text
hospitals_per_100k_population_odhf
```

Proxy quality:

```text
odhf_hospital_facility_density_proxy
```

Methodological note:

```text
This is an ODHF hospital facility-density proxy, not necessarily identical to original SoVI community hospitals per capita.
```

### `pct_female`

Original SoVI code:

```text
PCTFEM90
```

Source:

```text
census_division_age_structure_2021/output/clean_census_division_age_structure_2021.csv
```

Selected source column:

```text
pct_female
```

Proxy quality:

```text
direct_or_strong_proxy
```

## Variables still missing or unresolved

The following 23 variables are not yet ready:

```text
pct_vote_leading_party
birth_rate
net_international_migration
pct_land_farms
pct_black
pct_indigenous
pct_asian
pct_hispanic
pct_rural_farm
debt_revenue_ratio
housing_unit_density
housing_permit_density
manufacturing_density
earnings_density
commercial_density
property_value_density
pct_extractive_employment
pct_transport_utility_employment
pct_service_employment
pct_population_change
pct_urban
pct_female_headed_households
social_security_recipients_per_capita
```

## Known unresolved variable

### `pct_female_headed_households`

Original SoVI code:

```text
PCTF_HH90
```

Status:

```text
unresolved_not_numeric
```

The household/family block found no defensible numeric proxy in the current Census Profile source.

The SoVI draft table should not silently fabricate this variable. It remains missing unless a better source or defensible proxy is added later.

## Recommended next missing-variable clusters

### 1. Ethnocultural / Indigenous identity cluster

Likely next block:

```text
census_division_ethnocultural_identity_2021/
```

Target variables:

```text
pct_black
pct_indigenous
pct_asian
pct_hispanic
```

These are likely available from the Census Profile CD-level file and should be inspected carefully as a group.

### 2. Employment-sector cluster

Likely next block:

```text
census_division_employment_sector_2021/
```

Target variables:

```text
pct_extractive_employment
pct_transport_utility_employment
pct_service_employment
```

These may be available from Census Profile labour / industry / occupation sections.

### 3. Population / urban-rural cluster

Possible next block:

```text
census_division_population_urban_rural_2021/
```

Target variables:

```text
pct_population_change
pct_urban
pct_rural_farm
```

Some of this information may already exist in the census-division base frame, especially:

```text
population_change_pct_2016_2021
```

### 4. Density / economic landscape cluster

Possible target variables:

```text
housing_unit_density
housing_permit_density
manufacturing_density
earnings_density
commercial_density
property_value_density
```

Some partial inputs may already exist in the base frame, especially:

```text
land_area_km2
total_private_dwellings_2021
private_dwellings_occupied_by_usual_residents_2021
```

Other variables likely require additional business, permits, property, or economic data sources.

## Important methodological caveats

This table is a Canadian/Québec census-division adaptation of the original SoVI variable structure.

Several variables are direct or strong proxies, but others are Canadian source-specific substitutions.

Important current proxy caveats:

```text
per_capita_income:
    Uses income_measure_default from the income cleaner. Requires method note.

pct_no_high_school:
    Uses 25–64 no-high-school-diploma rate as an age-universe proxy.

physicians_per_100k:
    Uses CIHI health-region physician rate assigned to census divisions.
    One documented missing value remains for Nord-du-Québec.

nursing_home_residents_per_capita:
    Uses ODHF residential-care facilities per 100k population.
    This is not literal nursing-home residents per capita.

hospitals_per_capita:
    Uses ODHF hospitals per 100k population.
    This is a facility-density proxy.
```

## Current interpretation

The current draft table is good enough to guide the next stages of variable construction.

It should not yet be treated as the final SoVI input table because:

```text
1. 23 variables remain missing or unresolved.
2. Several included variables require methodological notes.
3. The final clean SoVI input cleaner has not yet been generated.
```

## Next step

The recommended next step is to continue filling missing variables block by block.

Suggested immediate next block:

```text
census_division_ethnocultural_identity_2021/
```

Suggested target variables:

```text
pct_black
pct_indigenous
pct_asian
pct_hispanic
```

After each new block is cleaned, rerun:

```bash
python sovi_2021/inspect_census_division_sovi_input_sources_2021.py
```

to update:

```text
sovi_2021/output/draft_clean_census_division_sovi_input_2021.csv
```

and the source-audit files.