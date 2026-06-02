# SoVI-like Québec Census-Division Replication, 2021

This README documents the construction of the **SoVI-like Québec census-division 2021 input table and oriented SoVI-38 run**.

The project adapts the original Social Vulnerability Index methodology of Cutter, Boruff, and Shirley to Québec census divisions using Canadian 2021-era public datasets. The result is not an exact reproduction of the original U.S. county SoVI, but a transparent local adaptation with explicit source mappings, proxy labels, missing-variable documentation, PCA/varimax factor extraction, factor orientation, additive aggregation, and map-ready spatial outputs.

The final operational run documented here is:

```text
SoVI-like Québec CD 2021, 38-variable local adaptation
Spatial units: 98 Québec census divisions
Input table: data/sovi_2021/output/draft_clean_census_division_sovi_input_2021.csv
Recipe: recipes/sovi_like_quebec_cd_2021_38var.yaml
Run directory: data/sovi_2021/output/sovi_like_quebec_cd_2021_38var_oriented_run/
```

The oriented run used 38 variables, retained 7 varimax-rotated factors using the eigenvalue-greater-than-1 rule, explained approximately 82.78% of the variance, oriented factors according to SoVI methodology, and summed the oriented factor scores additively.

---

## 1. What this project reproduces

The original SoVI methodology is not a fixed hand-weighted formula. It is a factor-analytic pipeline:

```text
raw socioeconomic variables
→ compute percentages / per-capita rates / densities
→ handle missing values
→ standardize variables
→ PCA / factor analysis
→ retain factors with eigenvalue > 1
→ varimax rotation
→ interpret factors
→ orient factor scores so higher = more vulnerable
→ use absolute value for ambiguous factors
→ sum factor scores additively
→ classify / map scores
```

The original paper used 42 independent variables and retained 11 factors. The final score was produced by adding oriented factor scores with equal factor weights. The methodology explicitly relies on interpretive factor orientation: signs are adjusted so that positive values increase vulnerability, and ambiguous factors can be converted to absolute values before the additive sum.

This project follows the same methodological structure, but adapts the variable sources and proxies to Canadian / Québec 2021 census-division data.

---

## 2. Final status

The final SoVI-like table currently includes:

```text
38 / 42 original SoVI-inspired variables
36 variables with full 98/98 Québec CD coverage
2 variables with documented partial coverage
4 variables unresolved
```

The four unresolved variables are not silently imputed as if they existed. They are documented and excluded from the active 38-variable SoVI run.

The final active input table has:

```text
98 rows
1 row per Québec census division
zone_id = census_division_dguid
38 active numeric SoVI variables
```

The oriented run has:

```text
Spatial units input: 98
Spatial units output: 98
Variables used: 38
Variables missing from recipe: None
PCA method: pca
Rotation: varimax
Factor retention: eigenvalue > 1
Factors retained: 7
Retained explained variance: 0.8277898764660341
Aggregation: equal additive factor scores
Score direction: higher_is_more_vulnerable
```

---

## 3. Main project paths

### Source-input builder

```text
data/sovi_2021/inspect_census_division_sovi_input_sources_2021.py
```

This script loads all YAML source mappings from:

```text
data/sovi_2021/mappings/
```

It joins completed source variables onto the Québec census-division base frame and writes the wide draft input table:

```text
data/sovi_2021/output/draft_clean_census_division_sovi_input_2021.csv
```

### SoVI recipe

```text
recipes/sovi_like_quebec_cd_2021_38var.yaml
```

This recipe defines the active 38-variable SoVI-like run. It excludes the four unresolved original reference variables, but documents them separately.

### Final oriented run

```text
data/sovi_2021/output/sovi_like_quebec_cd_2021_38var_oriented_run/
```

Important files in the oriented run folder:

```text
standard_output.csv
intermediate_output.csv
metadata.json
metadata.yaml
validation_report.json
validation_report.yaml
missing_data_report.json
run_report.md
sovi_eigenvalues.csv
sovi_explained_variance.csv
sovi_loadings_unrotated.csv
sovi_loadings_rotated.csv
sovi_factor_scores.csv
sovi_factor_summary.csv
sovi_standardized_variables.csv
```

### Map-ready spatial outputs

After running the GeoJSON script, the oriented run folder should also contain:

```text
sovi_like_quebec_cd_2021_38var_map_web.geojson
sovi_like_quebec_cd_2021_38var_map_native.gpkg
sovi_like_quebec_cd_2021_38var_map_audit.csv
sovi_like_quebec_cd_2021_38var_map_legend.csv
```

---

## 4. Base geography

The spatial unit is the **Québec census division**.

The base frame is:

```text
data/census_division_spatial_frame_population_2021/output/
clean_quebec_census_division_spatial_frame_with_population_2021.csv
```

and, for mapping, the corresponding spatial file is expected to be one of:

```text
data/census_division_spatial_frame_population_2021/output/
clean_quebec_census_division_spatial_frame_with_population_2021.geojson

data/census_division_spatial_frame_population_2021/output/
clean_quebec_census_division_spatial_frame_with_population_2021.gpkg
```

The key join field throughout the SoVI pipeline is:

```text
census_division_dguid
```

The SoVI runner uses:

```text
zone_id = census_division_dguid
```

The base frame also supplies:

```text
census_division_code
census_division_name
population_total_2021
land_area_km2
```

`land_area_km2` is the denominator for several density variables.

---

## 5. Pipeline overview

The project pipeline has five main layers.

### Layer 1 — variable-specific cleaners

Each completed SoVI variable or variable group has a source folder such as:

```text
data/census_division_demographic_estimates_2021/
data/census_division_agriculture_2021/
data/census_division_business_establishments_2021/
data/census_division_property_value_density_2021/
```

Most folders contain:

```text
raw/
output/
inspect_*.py
clean_*.py
README.md
```

The inspection scripts identify candidate rows, check coverage, test formulas, and avoid silently accepting bad proxies. The cleaning scripts write final clean variable tables and audit metadata.

### Layer 2 — source mapping YAML files

Completed variables are registered in YAML files under:

```text
data/sovi_2021/mappings/
```

Each mapping declares:

```yaml
source_folder: ...
file_candidates:
  - output/clean_*.csv
column_candidates:
  - canonical_variable
status_class: ...
proxy_quality: ...
allow_partial_coverage: ...
note: ...
```

The mapping files are the bridge between completed source blocks and the draft SoVI input table.

### Layer 3 — draft SoVI input table

Run:

```bash
cd /home/tim/Documents/ville_ai/indices/data

python sovi_2021/inspect_census_division_sovi_input_sources_2021.py
```

This creates:

```text
sovi_2021/output/draft_clean_census_division_sovi_input_2021.csv
```

It also writes audit outputs:

```text
sovi_2021/output/census_division_sovi_input_source_audit_2021.csv
sovi_2021/output/census_division_sovi_missing_or_unresolved_variables_2021.csv
sovi_2021/output/census_division_sovi_expected_columns_2021.csv
sovi_2021/output/census_division_sovi_source_file_inventory_2021.csv
sovi_2021/output/census_division_sovi_mapping_file_inventory_2021.csv
sovi_2021/output/census_division_sovi_input_source_inspection_summary_2021.csv
```

### Layer 4 — SoVI PCA / varimax run

From project root:

```bash
cd /home/tim/Documents/ville_ai/indices

PYTHONPATH=src python -m ville_indices.run \
  --index sovi_like \
  --recipe recipes/sovi_like_quebec_cd_2021_38var.yaml \
  --features data/sovi_2021/output/draft_clean_census_division_sovi_input_2021.csv \
  --output-dir data/sovi_2021/output/sovi_like_quebec_cd_2021_38var_oriented_run
```

### Layer 5 — map-ready GeoJSON

Run:

```bash
cd /home/tim/Documents/ville_ai/indices

python data/sovi_2021/create_sovi_geojson_map_2021.py
```

The GeoJSON script joins the final `standard_output.csv` back to Québec census-division geometry, writes a web GeoJSON in EPSG:4326, a native GeoPackage, an audit CSV, and a legend CSV.

---

## 6. Active 38 variables

The active SoVI-38 run uses the following variables.

| Original SoVI code | Canonical variable | Status | Main source / method |
|---|---|---|---|
| `MED_AGE90` | `med_age` | ready | 2021 Census Profile, median age proxy |
| `PERCAP89` | `per_capita_income` | ready | 2021 Census Profile income proxy |
| `MVALOO90` | `median_home_value` | ready | Census Profile median value of dwellings |
| `MEDRENT90` | `median_rent` | ready | Census Profile median monthly shelter costs for rented dwellings |
| `PHYSICN90` | `physicians_per_100k` | partial | CIHI / health-region physician proxy joined to CDs |
| `PCTVOTE92` | `pct_vote_leading_party` | ready | Area-weighted Canadian federal-election leading-party proxy |
| `BRATE90` | `birth_rate` | ready | StatCan Annual Demographic Estimates, births / population |
| `MIGRA_97` | `net_international_migration` | ready | StatCan demographic components |
| `PCTFARMS92` | `pct_land_farms` | partial | Census of Agriculture land use, total farm area / land area |
| `PCTBLACK90` | `pct_black` | ready | Census Profile visible-minority proxy |
| `PCTINDIAN90` | `pct_indigenous` | ready | Census Profile Indigenous identity proxy |
| `PCTASIAN90` | `pct_asian` | ready | Census Profile derived Asian visible-minority group proxy |
| `PCTHISPANIC90` | `pct_hispanic` | ready | Census Profile Latin American proxy |
| `PCTKIDS90` | `pct_under_5` | ready | Census Profile age group |
| `PCTOLD90` | `pct_over_65` | ready | Census Profile age group |
| `PCTVLUN91` | `pct_unemployed` | ready | Census Profile labour-force unemployment proxy |
| `AVGPERHH` | `avg_people_per_household` | ready | Census Profile household size |
| `PCTHH7589` | `pct_high_income_households` | ready | Census Profile high-income threshold proxy |
| `PCTPOV90` | `pct_poverty` | ready | Canadian low-income proxy |
| `PCTRENTER90` | `pct_renter` | ready | Census Profile renter tenure row |
| `PCTMOBL90` | `pct_mobile_homes` | ready | Census Profile movable dwelling proxy |
| `PCTNOHS90` | `pct_no_high_school` | ready | Census Profile no-high-school diploma proxy |
| `HODENUT90` | `housing_unit_density` | ready | Derived from housing counts / land area |
| `MAESDEN92` | `manufacturing_density` | ready | Business Counts dashboard, NAICS 31-33, CSD aggregated to CD |
| `EARNDEN90` | `earnings_density` | ready | Census Profile derived employment-income density |
| `COMDEVDN92` | `commercial_density` | ready | Business Counts dashboard, NAICS 41 + 44-45 + 72 |
| `RPROPDEN92` | `property_value_density` | ready | Weak owner-occupied residential property-value-density proxy |
| `CVBRPC91` | `labor_force_participation` | ready | Census Profile labour-force participation |
| `FEMLBR90` | `female_labor_force_participation` | ready | Census Profile female labour-force participation |
| `AGRIPC90` | `pct_extractive_employment` | ready | Census Profile primary/extractive employment proxy |
| `TRANPC90` | `pct_transport_utility_employment` | ready | Census Profile transportation / utility / communication employment proxy |
| `SERVPC90` | `pct_service_employment` | ready | Census Profile sales-and-service occupation proxy |
| `NRRESPC91` | `nursing_home_residents_per_capita` | ready | ODHF residential-care facility-density proxy |
| `HOSPTPC91` | `hospitals_per_capita` | ready | ODHF hospital facility-density proxy |
| `PCCHGPOP90` | `pct_population_change` | ready | Census population change proxy |
| `PCTFEM90` | `pct_female` | ready | Census Profile sex composition |
| `PCTF_HH90` | `pct_female_headed_households` | ready | Census Profile female-parent one-parent family proxy |
| `SSBENPC90` | `social_security_recipients_per_capita` | ready | Census Profile government-transfer recipient proxy |

---

## 7. Unresolved variables

Four original SoVI variables remain unresolved and are excluded from the 38-variable run.

| Original code | Canonical variable | Reason unresolved |
|---|---|---|
| `PCTRFRM90` | `pct_rural_farm` | Census of Agriculture public tables support land in farms, farm area, and farms reporting, but not a direct rural farm population share. Mapping a farm-area variable to rural farm population would be conceptually wrong. |
| `DEBREV92` | `debt_revenue_ratio` | A full-coverage, harmonized Québec census-division local government debt-to-revenue source was not integrated. This likely requires municipal finance / MAMH fiscal data and nontrivial aggregation from municipalities to CDs. |
| `HUPTDEN90` | `housing_permit_density` | StatCan table 34-10-0292-01 was inspected. It had building-permit measures, but no CD or CSD geography usable for 98 Québec census divisions. |
| `PCTURB90` | `pct_urban` | The Census Profile inspection did not find a defensible direct urban/rural population-share row for CDs. Population density was available but was not accepted as equivalent to percent urban. |

These variables should stay documented as unresolved rather than being replaced with weak or misleading proxies without a clear methodological decision.

---

## 8. Important variable blocks

### 8.1 Demographic estimates

Folder:

```text
data/census_division_demographic_estimates_2021/
```

Raw source:

```text
raw/population_estimates_for_canada_subprovincial_areas.xlsx
```

This Excel workbook from the Annual Demographic Estimates subprovincial areas release contained sheets for population and demographic components at the census-division scale.

Cleaned variables:

```text
BRATE90  -> birth_rate
MIGRA_97 -> net_international_migration
```

The birth-rate formula is:

```text
birth_rate =
    1000 * births_2020_2021 / population_2021
```

The net international migration candidate formula is:

```text
net_international_migration =
    immigrants_2020_2021
    - emigrants_2020_2021
    + returning_emigrants_2020_2021
    - net_temporary_emigrants_2020_2021
    + net_non_permanent_residents_2020_2021
```

Negative values are possible and meaningful because this is a net flow measure.

---

### 8.2 Agriculture

Folder:

```text
data/census_division_agriculture_2021/
```

Final useful source:

```text
raw/land_use_32100249_2021.csv
raw/land_use_32100249_2021_MetaData.csv
```

Cleaned variable:

```text
PCTFARMS92 -> pct_land_farms
```

Formula:

```text
pct_land_farms =
    100 * total_farm_area_km2 / land_area_km2
```

`total_farm_area_km2` is derived from the Census of Agriculture `Total farm area` row, using `Hectares` as the preferred unit.

Coverage is partial:

```text
94 / 98 Québec census divisions
```

The missing CDs had positive farm-reporting counts but suppressed or unavailable total farm area values. These missing values are documented and later handled by the SoVI missing-data rule.

This variable is a strong adaptation of `PCTFARMS92` because both measure land in farms relative to total land. It is not suitable for `PCTRFRM90`, which refers to rural farm population.

---

### 8.3 Housing tenure and costs

Folder:

```text
data/census_division_housing_tenure_costs_2021/
```

Source:

```text
data/census_profile_census_division_2021/raw/98-401-X2021004_English_CSV_data.csv
```

Cleaned variables:

```text
PCTRENTER90 -> pct_renter
MEDRENT90   -> median_rent
MVALOO90    -> median_home_value
```

Important Census Profile characteristic IDs:

```text
1416 -> Renter
1494 -> Median monthly shelter costs for rented dwellings ($)
1488 -> Median value of dwellings ($)
```

The cleaned file provides full coverage:

```text
98 / 98 Québec census divisions
```

---

### 8.4 Property value density

Folder:

```text
data/census_division_property_value_density_2021/
```

Cleaned variable:

```text
RPROPDEN92 -> property_value_density
```

This is a documented weak proxy.

Formula:

```text
property_value_density =
    median_owner_occupied_housing_value
    * owner_households_direct_count
    / land_area_km2
```

Main components:

```text
Census Profile ID 1415 -> Owner
Census Profile ID 1488 -> Median value of dwellings ($)
```

The numerator is retained as:

```text
estimated_owner_occupied_residential_property_value =
    median_owner_occupied_housing_value
    * owner_households_direct_count
```

Proxy quality:

```text
weak_residential_owner_occupied_property_value_density_proxy
```

This does not reproduce total assessed property value. It excludes rental, commercial, industrial, institutional, farm, and land-assessment values. It is retained only because no stronger full-coverage property-assessment source was integrated.

Coverage:

```text
98 / 98 Québec census divisions
```

---

### 8.5 Business establishments

Folder:

```text
data/census_division_business_establishments_2021/
```

Final source:

```text
raw/canada_rural_business_counts_dashboard.csv
```

This dashboard extract was used because the first Canadian Business Counts table inspected only covered 41/98 Québec CDs after CSD-to-CD aggregation. The dashboard extract provided 2021 census-subdivision DGUIDs and complete aggregation to 98 CDs.

Cleaned variables:

```text
MAESDEN92  -> manufacturing_density
COMDEVDN92 -> commercial_density
```

Selected period:

```text
2022-01
```

The 2022-01 period is the earliest available period in the dashboard extract and the closest available business-count period to the 2021 SoVI-like table.

Formulas:

```text
manufacturing_density =
    manufacturing_business_count / land_area_km2
```

where:

```text
manufacturing_business_count = NAICS 31-33 Manufacturing, total with employees
```

and:

```text
commercial_density =
    (
        wholesale_trade_business_count
        + retail_trade_business_count
        + accommodation_food_services_business_count
    )
    / land_area_km2
```

where:

```text
wholesale_trade_business_count = NAICS 41
retail_trade_business_count = NAICS 44-45
accommodation_food_services_business_count = NAICS 72
```

The cleaner also keeps a narrower audit variable:

```text
commercial_trade_only_density_per_km2 =
    (wholesale_trade_business_count + retail_trade_business_count) / land_area_km2
```

Coverage:

```text
98 / 98 Québec census divisions
```

---

### 8.6 Earnings density

Folder:

```text
data/census_division_earnings_density_2021/
```

Cleaned variable:

```text
EARNDEN90 -> earnings_density
```

No direct aggregate employment-income amount was found in the public Census Profile table. The adopted proxy derives aggregate employment income from a recipient-count times average-income formula and divides by land area.

Conceptual formula:

```text
earnings_density =
    derived_aggregate_employment_income_2020 / land_area_km2
```

Proxy quality:

```text
derived_canadian_employment_income_density_proxy
```

Coverage:

```text
98 / 98 Québec census divisions
```

---

### 8.7 Social Security recipients

Folder:

```text
data/census_division_social_security_recipients_2021/
```

Cleaned variable:

```text
SSBENPC90 -> social_security_recipients_per_capita
```

The original U.S. variable refers to Social Security recipients. A direct Canadian equivalent does not exist in the Census Profile. The adopted broad proxy is:

```text
Number of government transfers recipients aged 15 years and over in private households in 2020
/
population_total_2021
```

Proxy quality:

```text
broad_canadian_government_transfer_recipient_proxy
```

This is broader than U.S. Social Security and includes a wider Canadian government-transfer concept. It is accepted as a documented proxy.

Coverage:

```text
98 / 98 Québec census divisions
```

---

### 8.8 Female-headed households

Folder:

```text
data/census_division_female_headed_households_2021/
```

Cleaned variable:

```text
PCTF_HH90 -> pct_female_headed_households
```

The Canadian proxy uses a female-parent one-parent census-family measure. It is not a perfect one-to-one reproduction of the U.S. “female-headed households, no spouse present” concept, but it captures a closely related family-structure vulnerability dimension.

Proxy quality:

```text
female_parent_one_parent_census_family_proxy
```

Coverage:

```text
98 / 98 Québec census divisions
```

---

### 8.9 Housing permits

Folder:

```text
data/census_division_housing_permits_2021/
```

Inspected source:

```text
raw/building_permits_by_type_structure_work_3410029201.csv
raw/building_permits_by_type_structure_work_3410029201_MetaData.csv
```

Target variable:

```text
HUPTDEN90 -> housing_permit_density
```

Inspection result:

```text
not usable for CD-scale SoVI
```

Reason:

```text
direct CD rows total: 0
CSD rows total: 0
Québec direct CD geographies: 0
Québec CSD geographies: 0
full coverage candidate count: 0
```

The table contains useful building-permit measures, but its public geography is not census division or census subdivision. It cannot be joined or aggregated to 98 Québec census divisions.

No cleaner was generated.

---

### 8.10 Urban population share

Target variable:

```text
PCTURB90 -> pct_urban
```

A Census Profile inspection searched for urban, population-centre, rural, and inverse-rural candidates. It did not find a defensible direct CD-level percent-urban population share.

The base frame includes population density, but density was not accepted as equivalent to percent urban.

Status:

```text
unresolved
```

---

### 8.11 Debt-to-revenue ratio

Target variable:

```text
DEBREV92 -> debt_revenue_ratio
```

This was not resolved in the current pipeline. A defensible version would likely require Québec municipal finance data, such as debt and revenue from municipal financial reports, followed by aggregation from municipalities to census divisions.

Status:

```text
unresolved
```

---

### 8.12 Rural farm population

Target variable:

```text
PCTRFRM90 -> pct_rural_farm
```

The agriculture tables inspected contained farm area, land use, and farms-reporting measures. They did not provide a rural farm population share.

`pct_land_farms` should not be reused for this because land in farms and rural farm population are different constructs.

Status:

```text
unresolved
```

---

## 9. Missing-data treatment

The SoVI recipe uses:

```text
zero_imputation
```

The oriented run reported:

```text
Affected spatial units: 5
```

The affected missingness is driven by documented partial-coverage variables, especially:

```text
physicians_per_100k
pct_land_farms
```

This follows the original SoVI-style missing-data choice of preserving all spatial units rather than dropping them. It should be documented as a methodological limitation because zero imputation can distort factor structure when zero is not substantively meaningful.

A future robustness section should compare:

```text
zero imputation
mean imputation
median imputation
complete-case analysis
possibly multiple imputation
```

For the current SoVI-like benchmark, zero imputation is retained because it is closest to the original SoVI workflow.

---

## 10. Normalization and standardization

The input variables are already normalized into meaningful rates, percentages, per-capita values, or densities before entering the SoVI runner.

Examples:

```text
pct_* variables -> percent-style variables
physicians_per_100k -> rate per 100,000 population
hospitals_per_capita -> facility rate
manufacturing_density -> business count / land area
earnings_density -> derived dollars / land area
property_value_density -> estimated dollars / land area
```

Then the SoVI runner applies:

```text
zscore standardization
```

to every active variable before PCA.

This is important because the variables have heterogeneous units: dollars, percentages, densities, per-capita rates, and counts/rates. Z-score standardization makes the PCA effectively operate on standardized variables rather than raw units.

---

## 11. PCA, factor retention, and rotation

The oriented run uses:

```text
PCA method: pca
Factor retention: eigenvalue_gt
Rotation: varimax
```

The retained factor count is:

```text
7
```

The retained explained variance is:

```text
0.8277898764660341
```

The original SoVI paper retained 11 factors for the U.S. county case. This Québec adaptation retained 7 factors because the retained factor count is data-dependent. The number of retained factors can change with:

```text
the country / region
the year
the spatial scale
the variable set
the proxy definitions
the missing-data rule
the standardization
the factor-retention rule
the rotation implementation
```

Therefore, retaining 7 factors is not a failure. It is the result of applying the SoVI factor-retention logic to the Québec 2021 38-variable dataset.

---

## 12. Factor orientation

The first exploratory run used placeholder positive orientations. That run was not treated as the final map.

The oriented run applies manual factor orientation using the original SoVI convention:

```text
positive direction if high factor scores increase vulnerability
negative direction if high factor scores decrease vulnerability
absolute value if factor interpretation is ambiguous
```

The final orientation decisions were:

| Factor | Orientation | Interpretation |
|---|---|---|
| `factor_1` | negative | Personal wealth / socioeconomic advantage factor. Reversed so resource disadvantage contributes to vulnerability. |
| `factor_2` | negative | Density of built environment / economic activity factor. Reversed so high density/economic centrality contributes positively, consistent with the original SoVI density factor. |
| `factor_3` | negative | Social marginalization / Indigenous identity / education / female-headed-household factor. Reversed so higher vulnerability-relevant indicators increase vulnerability. |
| `factor_4` | absolute | Ambiguous demographic / institutional / rural-economic contrast factor. |
| `factor_5` | positive | Tenancy / service / female-share factor. |
| `factor_6` | absolute | Ambiguous agriculture / unemployment / service-access factor. |
| `factor_7` | absolute | Weak / ambiguous infrastructure-tenancy contrast factor. |

The dominant variables reported in the oriented run were:

```text
factor_1:
  pct_high_income_households, female_labor_force_participation,
  pct_poverty, social_security_recipients_per_capita,
  labor_force_participation

factor_2:
  earnings_density, housing_unit_density, commercial_density,
  property_value_density, manufacturing_density

factor_3:
  pct_indigenous, pct_female_headed_households,
  pct_no_high_school, birth_rate, pct_mobile_homes

factor_4:
  pct_population_change, physicians_per_100k,
  pct_extractive_employment, hospitals_per_capita,
  pct_vote_leading_party

factor_5:
  pct_female, pct_service_employment, pct_renter,
  pct_extractive_employment, pct_female_headed_households

factor_6:
  pct_land_farms, pct_unemployed, physicians_per_100k,
  hospitals_per_capita, med_age

factor_7:
  pct_transport_utility_employment, pct_vote_leading_party,
  pct_renter, pct_mobile_homes, median_rent
```

Because factors 4, 6, and 7 are mixed and ambiguous, their absolute values are used before aggregation.

---

## 13. Aggregation

The aggregation method is:

```text
additive_factor_sum
```

The weighting is:

```text
equal additive factor scores
```

The score formula is conceptually:

```text
SoVI_i =
    oriented_factor_1_i
    + oriented_factor_2_i
    + ...
    + oriented_factor_7_i
```

There are no expert weights, no AHP weights, no variance weights, and no geometric aggregation in the main SoVI-like run.

The oriented run reported the score distribution:

```text
Minimum score: -3.7518151527990415
Mean score: 2.3893957187525188
Maximum score: 9.645657244538988
Standard deviation: 2.192484916812319
```

The mean is not zero because some factors use absolute value. This is expected under the original SoVI convention for ambiguous factors.

---

## 14. Interpretation of final score

The score direction is:

```text
higher_is_more_vulnerable
```

The score is an area-level social vulnerability score. It should not be interpreted as an individual-level diagnosis or as a direct measure of disaster loss.

The final score is best interpreted as:

```text
A relative, area-level composite vulnerability index for Québec census divisions,
derived from a 38-variable Canadian adaptation of the original SoVI framework.
```

High values indicate census divisions that score high on the oriented combination of socioeconomic disadvantage, built-environment density, social marginalization proxies, demographic/institutional contrasts, tenancy/service structure, and ambiguous rural/service-access contrasts.

---

## 15. Oriented run top and bottom zones

The oriented run reported the top 5 highest-score zones:

```text
1. 2021A00032466
2. 2021A00032499
3. 2021A00032498
4. 2021A00032404
5. 2021A00032443
```

The top zone is:

```text
2021A00032466 -> Montréal
```

The top 5 lowest-score zones were:

```text
1. 2021A00032422
2. 2021A00032482
3. 2021A00032420
4. 2021A00032442
5. 2021A00032419
```

The lowest-score zone is:

```text
2021A00032422 -> L'Île-d'Orléans
```

These rankings should be interpreted only within the current SoVI-38 local adaptation and its proxy choices.

---

## 16. Validation warnings and what they mean

The oriented run produced warnings that should be documented rather than ignored.

### 16.1 Percentage variables declared as 0–1 proportions

Many variables are encoded as 0–100 percentages, but the recipe metadata labels them as 0–1 proportions. The runner warns that these variables fall outside the 0–1 range.

Examples include:

```text
pct_black
pct_indigenous
pct_asian
pct_hispanic
pct_under_5
pct_over_65
pct_unemployed
pct_poverty
pct_renter
pct_no_high_school
labor_force_participation
female_labor_force_participation
pct_female
pct_female_headed_households
```

This does not change the PCA result because the runner applies z-score standardization. Multiplying a variable by 100 does not change its standardized z-scores, except for numerical roundoff. However, the recipe metadata should eventually be cleaned so these are declared as percent-style variables, not 0–1 proportions.

### 16.2 Negative values in net variables

The runner warned that:

```text
net_international_migration
pct_population_change
```

have negative values while declared nonnegative.

This is conceptually acceptable. Net migration and population change can be negative. The recipe metadata should eventually permit signed values for these variables.

### 16.3 Partial variables and zero imputation

The runner warned that:

```text
physicians_per_100k
pct_land_farms
```

contain missing values.

This is expected and documented. The run uses zero imputation, affecting 5 spatial units. This keeps all 98 census divisions in the PCA, matching the original SoVI preference for spatial completeness, but it should be evaluated in robustness checks.

---

## 17. Map outputs

The GeoJSON mapping script is:

```text
data/sovi_2021/create_sovi_geojson_map_2021.py
```

It reads:

```text
data/sovi_2021/output/sovi_like_quebec_cd_2021_38var_oriented_run/standard_output.csv
```

and joins it to the Québec census-division geometry.

It produces:

```text
sovi_like_quebec_cd_2021_38var_map_web.geojson
```

This is reprojected to:

```text
EPSG:4326
```

for web mapping in tools such as Leaflet, MapLibre, QGIS web viewers, or browser-based GeoJSON viewers.

It also produces:

```text
sovi_like_quebec_cd_2021_38var_map_native.gpkg
```

which keeps the native CRS for GIS workflows.

The map styling uses:

```text
sovi_score_normalized_0_1
```

to generate continuous colors and vulnerability labels:

```text
Very low
Low
Moderate
High
Very high
```

These labels are for map readability. The underlying score remains continuous.

---

## 18. How to reproduce from the current repository state

### Step 1 — Rebuild the draft input table

From the data folder:

```bash
cd /home/tim/Documents/ville_ai/indices/data

python sovi_2021/inspect_census_division_sovi_input_sources_2021.py
```

Expected high-level result:

```text
Variables with mapping config: 38
Variables ready for draft table: 38
Variables not ready / unmapped: 4
```

### Step 2 — Run the oriented SoVI-38 recipe

From the project root:

```bash
cd /home/tim/Documents/ville_ai/indices

PYTHONPATH=src python -m ville_indices.run \
  --index sovi_like \
  --recipe recipes/sovi_like_quebec_cd_2021_38var.yaml \
  --features data/sovi_2021/output/draft_clean_census_division_sovi_input_2021.csv \
  --output-dir data/sovi_2021/output/sovi_like_quebec_cd_2021_38var_oriented_run
```

### Step 3 — Generate map outputs

From the project root:

```bash
python data/sovi_2021/create_sovi_geojson_map_2021.py
```

---

## 19. Files to inspect after a successful run

The most important output files are:

```text
standard_output.csv
run_report.md
sovi_loadings_rotated.csv
sovi_factor_summary.csv
sovi_factor_scores.csv
sovi_eigenvalues.csv
validation_report.yaml
missing_data_report.json
sovi_like_quebec_cd_2021_38var_map_web.geojson
```

Suggested quick checks:

```bash
head -5 data/sovi_2021/output/sovi_like_quebec_cd_2021_38var_oriented_run/standard_output.csv

cat data/sovi_2021/output/sovi_like_quebec_cd_2021_38var_oriented_run/run_report.md

python - <<'PY'
import pandas as pd
from pathlib import Path

out = Path("data/sovi_2021/output/sovi_like_quebec_cd_2021_38var_oriented_run")

for name in [
    "sovi_eigenvalues.csv",
    "sovi_factor_summary.csv",
    "sovi_loadings_rotated.csv",
    "standard_output.csv",
]:
    path = out / name
    print("\n" + "=" * 80)
    print(name)
    print("=" * 80)
    df = pd.read_csv(path)
    print(df.head(20).to_string(index=False))
PY
```

---

## 20. How to add one of the missing variables later

The intended extension workflow is:

```text
1. Create a new source folder.
2. Download raw data into raw/.
3. Write inspect_*.py.
4. Review candidate rows and coverage.
5. Write clean_*.py only if the candidate is defensible.
6. Write README.md for the source block.
7. Add a mapping YAML in data/sovi_2021/mappings/.
8. Rerun inspect_census_division_sovi_input_sources_2021.py.
9. Update the active SoVI recipe if the variable becomes available.
10. Rerun SoVI and regenerate GeoJSON.
```

A new variable should not be added directly to the SoVI table without a source-specific README, cleaner, mapping file, and audit outputs.

---

## 21. Suggested robustness checks

The current oriented SoVI-38 run is a valid first local adaptation. For a publication-quality benchmark, add robustness checks.

### 21.1 Missing-data robustness

Compare:

```text
zero imputation
mean imputation
median imputation
complete-case analysis
```

Then compare:

```text
score correlation
rank correlation
top-decile overlap
class stability
factor-loadings stability
```

### 21.2 Proxy robustness

Run alternative versions excluding weak proxy variables such as:

```text
property_value_density
social_security_recipients_per_capita
nursing_home_residents_per_capita
hospitals_per_capita
pct_vote_leading_party
```

Compare the resulting ranks and factor structures.

### 21.3 Partial-variable robustness

Run a version excluding:

```text
physicians_per_100k
pct_land_farms
```

because these are partial-coverage variables requiring imputation.

### 21.4 Orientation robustness

Compare the final oriented run to:

```text
all-positive exploratory run
manual sign-only run
manual sign + absolute ambiguous factors
```

The current oriented run uses the final option.

### 21.5 Factor-retention robustness

Compare:

```text
eigenvalue > 1
fixed 7 factors
fixed 6 factors
fixed 8 factors
parallel analysis, if implemented
```

---

## 22. Methodological limitations

This project is transparent, but it is not an exact reproduction.

The major limitations are:

```text
1. Four original variables remain unresolved.
2. Several variables are Canadian proxies rather than direct analogues.
3. Some public datasets are not available at census-division or census-subdivision geography.
4. Two variables have documented partial coverage.
5. Zero imputation preserves coverage but can distort factor structure.
6. Factor signs require human interpretation.
7. Some factors are mixed and require absolute-value treatment.
8. The spatial unit is Québec census division, not U.S. county.
9. The temporal reference is approximately 2021, with some closest-available source periods such as 2022-01 for business counts.
10. Density variables can be dominated by Montréal and other high-density urban geographies.
```

These limitations are not hidden. They are part of the methodological documentation.

---

## 23. Recommended language for reporting

A concise methods paragraph:

```text
We constructed a Québec census-division SoVI-like index for 2021 by adapting the original Cutter, Boruff, and Shirley SoVI framework to Canadian public data. Of the 42 original SoVI-inspired variables, 38 were operationalized at the Québec census-division scale. Variables were computed as percentages, per-capita rates, densities, or documented proxies, then standardized using z-scores. Missing values in partial-coverage variables were replaced with zero to preserve full spatial coverage, following the original SoVI convention. We applied PCA with varimax rotation and retained factors with eigenvalues greater than one. Seven factors were retained, explaining approximately 82.8% of the variance. Factor scores were oriented so higher values corresponded to higher vulnerability; ambiguous factors were transformed using absolute values. The final SoVI score is the equal-weight additive sum of oriented factor scores.
```

A concise limitations paragraph:

```text
This index is a local SoVI-like adaptation rather than an exact reproduction of the original U.S. SoVI. Four original variables could not be operationalized with defensible full-coverage sources: rural farm population, local government debt-to-revenue ratio, housing permit density, and percent urban population. Several included variables rely on Canadian proxy definitions, including government-transfer recipients for Social Security recipients, facility-density proxies for hospitals and residential-care institutions, and an owner-occupied residential property-value-density proxy for property value density. The results should therefore be interpreted as a transparent comparative vulnerability index for Québec census divisions, not as a literal reproduction of the original U.S. county SoVI.
```

---

## 24. Recommended final archive checklist

Before freezing the result, keep the following files together:

```text
recipes/sovi_like_quebec_cd_2021_38var.yaml

data/sovi_2021/output/draft_clean_census_division_sovi_input_2021.csv
data/sovi_2021/output/census_division_sovi_input_source_audit_2021.csv
data/sovi_2021/output/census_division_sovi_missing_or_unresolved_variables_2021.csv
data/sovi_2021/output/census_division_sovi_input_source_inspection_summary_2021.csv

data/sovi_2021/output/sovi_like_quebec_cd_2021_38var_oriented_run/standard_output.csv
data/sovi_2021/output/sovi_like_quebec_cd_2021_38var_oriented_run/run_report.md
data/sovi_2021/output/sovi_like_quebec_cd_2021_38var_oriented_run/metadata.yaml
data/sovi_2021/output/sovi_like_quebec_cd_2021_38var_oriented_run/validation_report.yaml
data/sovi_2021/output/sovi_like_quebec_cd_2021_38var_oriented_run/missing_data_report.json
data/sovi_2021/output/sovi_like_quebec_cd_2021_38var_oriented_run/sovi_loadings_rotated.csv
data/sovi_2021/output/sovi_like_quebec_cd_2021_38var_oriented_run/sovi_factor_summary.csv
data/sovi_2021/output/sovi_like_quebec_cd_2021_38var_oriented_run/sovi_factor_scores.csv
data/sovi_2021/output/sovi_like_quebec_cd_2021_38var_oriented_run/sovi_eigenvalues.csv
data/sovi_2021/output/sovi_like_quebec_cd_2021_38var_oriented_run/sovi_like_quebec_cd_2021_38var_map_web.geojson
data/sovi_2021/output/sovi_like_quebec_cd_2021_38var_oriented_run/sovi_like_quebec_cd_2021_38var_map_native.gpkg
```

Also keep each variable source block README and mapping YAML file, because those document how the 38 variables were constructed.

---

## 25. Bottom line

This project produced a complete, auditable, map-ready **SoVI-like Québec census-division 2021 index** using 38 adapted variables.

The final oriented run is the main result. The first all-positive run should be treated only as an exploratory factor-sign audit.

The final output should be described as:

```text
A 38-variable, PCA/varimax, SoVI-like local adaptation for Québec census divisions in 2021,
with documented Canadian proxies, four unresolved original variables, zero-imputation
for partial coverage, manual SoVI-consistent factor orientation, additive factor-score
aggregation, and map-ready GeoJSON output.
```
