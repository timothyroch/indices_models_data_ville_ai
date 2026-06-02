# Run Report: sovi_like

- Run ID: `sovi_like-c630c585df2c`
- Index version: `0.1`
- Reproduction level: `local_adaptation`
- Construct measured: `social_vulnerability`
- Score direction: `higher_is_more_vulnerable`
- Input feature table: `data/sovi_2021/output/draft_clean_census_division_sovi_input_2021.csv`
- Spatial units input: `98`
- Spatial units output: `98`
- Source method: `{'citation': 'Cutter, Boruff & Shirley 2003, Social Vulnerability to Environmental Hazards', 'journal': 'Social Science Quarterly'}`
- PCA method: `pca`
- Factor retention: `eigenvalue_gt`
- Factors retained: `7`
- Rotation method: `varimax`

## Variables Used

- med_age
- per_capita_income
- median_home_value
- median_rent
- physicians_per_100k
- pct_vote_leading_party
- birth_rate
- net_international_migration
- pct_land_farms
- pct_black
- pct_indigenous
- pct_asian
- pct_hispanic
- pct_under_5
- pct_over_65
- pct_unemployed
- avg_people_per_household
- pct_high_income_households
- pct_poverty
- pct_renter
- pct_mobile_homes
- pct_no_high_school
- housing_unit_density
- manufacturing_density
- earnings_density
- commercial_density
- property_value_density
- labor_force_participation
- female_labor_force_participation
- pct_extractive_employment
- pct_transport_utility_employment
- pct_service_employment
- nursing_home_residents_per_capita
- hospitals_per_capita
- pct_population_change
- pct_female
- pct_female_headed_households
- social_security_recipients_per_capita

## Variables Missing

- None

## Warnings

- Variable 'physicians_per_100k' contains missing values.
- Variable 'pct_vote_leading_party' is declared as a 0-1 proportion but falls outside that range.
- Variable 'pct_vote_leading_party' may be encoded as 0-100 percent instead of 0-1 proportion.
- Variable 'net_international_migration' has negative values but is declared nonnegative.
- Variable 'pct_land_farms' contains missing values.
- Variable 'pct_land_farms' is declared as a 0-1 proportion but falls outside that range.
- Variable 'pct_land_farms' may be encoded as 0-100 percent instead of 0-1 proportion.
- Variable 'pct_black' is declared as a 0-1 proportion but falls outside that range.
- Variable 'pct_black' may be encoded as 0-100 percent instead of 0-1 proportion.
- Variable 'pct_indigenous' is declared as a 0-1 proportion but falls outside that range.
- Variable 'pct_indigenous' may be encoded as 0-100 percent instead of 0-1 proportion.
- Variable 'pct_asian' is declared as a 0-1 proportion but falls outside that range.
- Variable 'pct_asian' may be encoded as 0-100 percent instead of 0-1 proportion.
- Variable 'pct_hispanic' is declared as a 0-1 proportion but falls outside that range.
- Variable 'pct_hispanic' may be encoded as 0-100 percent instead of 0-1 proportion.
- Variable 'pct_under_5' is declared as a 0-1 proportion but falls outside that range.
- Variable 'pct_under_5' may be encoded as 0-100 percent instead of 0-1 proportion.
- Variable 'pct_over_65' is declared as a 0-1 proportion but falls outside that range.
- Variable 'pct_over_65' may be encoded as 0-100 percent instead of 0-1 proportion.
- Variable 'pct_unemployed' is declared as a 0-1 proportion but falls outside that range.
- Variable 'pct_unemployed' may be encoded as 0-100 percent instead of 0-1 proportion.
- Variable 'pct_high_income_households' is declared as a 0-1 proportion but falls outside that range.
- Variable 'pct_high_income_households' may be encoded as 0-100 percent instead of 0-1 proportion.
- Variable 'pct_poverty' is declared as a 0-1 proportion but falls outside that range.
- Variable 'pct_poverty' may be encoded as 0-100 percent instead of 0-1 proportion.
- Variable 'pct_renter' is declared as a 0-1 proportion but falls outside that range.
- Variable 'pct_renter' may be encoded as 0-100 percent instead of 0-1 proportion.
- Variable 'pct_mobile_homes' is declared as a 0-1 proportion but falls outside that range.
- Variable 'pct_mobile_homes' may be encoded as 0-100 percent instead of 0-1 proportion.
- Variable 'pct_no_high_school' is declared as a 0-1 proportion but falls outside that range.
- Variable 'pct_no_high_school' may be encoded as 0-100 percent instead of 0-1 proportion.
- Variable 'labor_force_participation' is declared as a 0-1 proportion but falls outside that range.
- Variable 'labor_force_participation' may be encoded as 0-100 percent instead of 0-1 proportion.
- Variable 'female_labor_force_participation' is declared as a 0-1 proportion but falls outside that range.
- Variable 'female_labor_force_participation' may be encoded as 0-100 percent instead of 0-1 proportion.
- Variable 'pct_extractive_employment' is declared as a 0-1 proportion but falls outside that range.
- Variable 'pct_extractive_employment' may be encoded as 0-100 percent instead of 0-1 proportion.
- Variable 'pct_transport_utility_employment' is declared as a 0-1 proportion but falls outside that range.
- Variable 'pct_transport_utility_employment' may be encoded as 0-100 percent instead of 0-1 proportion.
- Variable 'pct_service_employment' is declared as a 0-1 proportion but falls outside that range.
- Variable 'pct_service_employment' may be encoded as 0-100 percent instead of 0-1 proportion.
- Variable 'pct_population_change' has negative values but is declared nonnegative.
- Variable 'pct_population_change' is declared as percent but falls outside 0-100.
- Variable 'pct_female' is declared as a 0-1 proportion but falls outside that range.
- Variable 'pct_female' may be encoded as 0-100 percent instead of 0-1 proportion.
- Variable 'pct_female_headed_households' is declared as a 0-1 proportion but falls outside that range.
- Variable 'pct_female_headed_households' may be encoded as 0-100 percent instead of 0-1 proportion.
- Factor signs are arbitrary and require orientation decisions.
- SoVI is an area-level social vulnerability score and should not be interpreted as an individual-level diagnosis.
- This run is a SoVI-like local/partial adaptation, not exact original SoVI.
- Zero imputation was used; this may distort factor structure when zero is not substantively meaningful.

## Proxies Used

- per_capita_income -> per_capita_income (income_proxy_needs_method_note)
- physicians_per_100k -> physicians_per_100k (health_region_proxy_one_documented_missing)
- pct_vote_leading_party -> pct_vote_leading_party (area_weighted_federal_election_proxy)
- birth_rate -> birth_rate (crude_birth_rate_from_demographic_estimates)
- net_international_migration -> net_international_migration (net_international_migration_count_from_demographic_components)
- pct_land_farms -> pct_land_farms (strong_canadian_land_in_farms_adaptation_partial_coverage)
- pct_black -> pct_black (visible_minority_population_group_proxy)
- pct_indigenous -> pct_indigenous (indigenous_identity_proxy)
- pct_asian -> pct_asian (derived_visible_minority_asian_group_proxy)
- pct_hispanic -> pct_hispanic (latin_american_proxy_for_hispanic)
- pct_high_income_households -> pct_high_income_households (threshold_proxy)
- pct_poverty -> pct_poverty (canadian_low_income_proxy)
- pct_mobile_homes -> pct_mobile_homes (movable_dwelling_proxy)
- pct_no_high_school -> pct_no_high_school (age_universe_proxy)
- housing_unit_density -> housing_unit_density (derived_density_from_clean_base_frame)
- manufacturing_density -> manufacturing_density (naics_31_33_manufacturing_business_count_density_proxy)
- earnings_density -> earnings_density (derived_canadian_employment_income_density_proxy)
- commercial_density -> commercial_density (constructed_naics_commercial_business_count_density_proxy)
- property_value_density -> property_value_density (weak_residential_owner_occupied_property_value_density_proxy)
- pct_extractive_employment -> pct_extractive_employment (derived_primary_extractive_industry_proxy)
- pct_transport_utility_employment -> pct_transport_utility_employment (derived_transportation_utility_communication_industry_proxy)
- pct_service_employment -> pct_service_employment (sales_and_service_occupation_proxy)
- nursing_home_residents_per_capita -> nursing_home_residents_per_capita (facility_density_proxy_not_residents)
- hospitals_per_capita -> hospitals_per_capita (odhf_hospital_facility_density_proxy)
- pct_population_change -> pct_population_change (direct_population_change_proxy)
- pct_female_headed_households -> pct_female_headed_households (female_parent_one_parent_census_family_proxy)
- social_security_recipients_per_capita -> social_security_recipients_per_capita (broad_canadian_government_transfer_recipient_proxy)

## Missing Data Summary

- Strategy: `zero_imputation`
- Affected spatial units: `5`

## Normalization Summary

- `med_age` / `med_age`: `zscore`
- `per_capita_income` / `per_capita_income`: `zscore`
- `median_home_value` / `median_home_value`: `zscore`
- `median_rent` / `median_rent`: `zscore`
- `physicians_per_100k` / `physicians_per_100k`: `zscore`
- `pct_vote_leading_party` / `pct_vote_leading_party`: `zscore`
- `birth_rate` / `birth_rate`: `zscore`
- `net_international_migration` / `net_international_migration`: `zscore`
- `pct_land_farms` / `pct_land_farms`: `zscore`
- `pct_black` / `pct_black`: `zscore`
- `pct_indigenous` / `pct_indigenous`: `zscore`
- `pct_asian` / `pct_asian`: `zscore`
- `pct_hispanic` / `pct_hispanic`: `zscore`
- `pct_under_5` / `pct_under_5`: `zscore`
- `pct_over_65` / `pct_over_65`: `zscore`
- `pct_unemployed` / `pct_unemployed`: `zscore`
- `avg_people_per_household` / `avg_people_per_household`: `zscore`
- `pct_high_income_households` / `pct_high_income_households`: `zscore`
- `pct_poverty` / `pct_poverty`: `zscore`
- `pct_renter` / `pct_renter`: `zscore`
- `pct_mobile_homes` / `pct_mobile_homes`: `zscore`
- `pct_no_high_school` / `pct_no_high_school`: `zscore`
- `housing_unit_density` / `housing_unit_density`: `zscore`
- `manufacturing_density` / `manufacturing_density`: `zscore`
- `earnings_density` / `earnings_density`: `zscore`
- `commercial_density` / `commercial_density`: `zscore`
- `property_value_density` / `property_value_density`: `zscore`
- `labor_force_participation` / `labor_force_participation`: `zscore`
- `female_labor_force_participation` / `female_labor_force_participation`: `zscore`
- `pct_extractive_employment` / `pct_extractive_employment`: `zscore`
- `pct_transport_utility_employment` / `pct_transport_utility_employment`: `zscore`
- `pct_service_employment` / `pct_service_employment`: `zscore`
- `nursing_home_residents_per_capita` / `nursing_home_residents_per_capita`: `zscore`
- `hospitals_per_capita` / `hospitals_per_capita`: `zscore`
- `pct_population_change` / `pct_population_change`: `zscore`
- `pct_female` / `pct_female`: `zscore`
- `pct_female_headed_households` / `pct_female_headed_households`: `zscore`
- `social_security_recipients_per_capita` / `social_security_recipients_per_capita`: `zscore`

## Aggregation Summary

- Method: `additive_factor_sum`
- Weighting: `equal additive factor scores`

## SoVI Factor Analysis

- Standardization: `zscore`
- PCA method: `pca`
- Factor-retention rule: `eigenvalue_gt`
- Retained factors: `7`
- Retained explained variance: `0.8277898764660341`
- Rotation: `varimax`

## SoVI Factor Orientation Summary

- `factor_1`: `negative`; Personal wealth / socioeconomic advantage factor. High income, high labour-force participation, and high-income households load positively, while poverty and government-transfer recipients load negatively. Reversed so poverty/resource disadvantage contributes positively to vulnerability.
- `factor_2`: `negative`; Density of built environment / economic activity factor. Housing, earnings, commercial, property, and manufacturing densities load strongly together with negative signs. Reversed so higher built-environment/economic density contributes positively to vulnerability, consistent with the original SoVI density factor.
- `factor_3`: `negative`; Social marginalization / Indigenous identity / education / female-headed-household factor. Indigenous identity, female-headed households, and no-high-school load negatively. Reversed so higher values of these vulnerability-relevant indicators increase vulnerability.
- `factor_4`: `absolute`; Ambiguous demographic/institutional/rural-economic contrast factor. Population change loads negatively, while physicians, hospitals, extractive employment, and voting proxy load positively. Following original SoVI treatment of ambiguous factors, use absolute value.
- `factor_5`: `positive`; Tenancy/service/female-share factor. Female share, service employment, renter share, and female-headed households load positively. Kept positive so higher values increase vulnerability.
- `factor_6`: `absolute`; Ambiguous agriculture/unemployment/service-access factor. Land in farms loads negatively, while unemployment, physicians, hospitals, and median age load positively. Following original SoVI treatment of ambiguous factors, use absolute value.
- `factor_7`: `absolute`; Weak/ambiguous infrastructure-tenancy contrast factor. Transportation/utility employment and voting proxy load negatively while renter share loads positively. Following original SoVI treatment of ambiguous factors, use absolute value.

## SoVI Dominant Variables

- `factor_1`: pct_high_income_households:0.9334; female_labor_force_participation:0.9206; pct_poverty:-0.9153; social_security_recipients_per_capita:-0.9096; labor_force_participation:0.8920
- `factor_2`: earnings_density:-0.9902; housing_unit_density:-0.9892; commercial_density:-0.9892; property_value_density:-0.9883; manufacturing_density:-0.9783
- `factor_3`: pct_indigenous:-0.8725; pct_female_headed_households:-0.7412; pct_no_high_school:-0.7188; birth_rate:-0.4598; pct_mobile_homes:-0.4463
- `factor_4`: pct_population_change:-0.7851; physicians_per_100k:0.6188; pct_extractive_employment:0.6101; hospitals_per_capita:0.5334; pct_vote_leading_party:0.5315
- `factor_5`: pct_female:0.8428; pct_service_employment:0.7227; pct_renter:0.5750; pct_extractive_employment:-0.4522; pct_female_headed_households:0.4004
- `factor_6`: pct_land_farms:-0.7842; pct_unemployed:0.6161; physicians_per_100k:0.5247; hospitals_per_capita:0.4057; med_age:0.3245
- `factor_7`: pct_transport_utility_employment:-0.7557; pct_vote_leading_party:-0.4268; pct_renter:0.3378; pct_mobile_homes:-0.2768; median_rent:-0.2016

## SoVI Score Distribution

- Minimum score: `-3.7518151527990415`
- Mean score: `2.3893957187525188`
- Maximum score: `9.645657244538988`
- Standard deviation: `2.192484916812319`

## Interpretation Warning

SoVI is an area-level social vulnerability score and should not be interpreted as an individual-level diagnosis.

## Output Files

- `standard_output`: `data/sovi_2021/output/sovi_like_quebec_cd_2021_38var_oriented_run/standard_output.csv`
- `intermediate_output`: `data/sovi_2021/output/sovi_like_quebec_cd_2021_38var_oriented_run/intermediate_output.csv`
- `metadata_json`: `data/sovi_2021/output/sovi_like_quebec_cd_2021_38var_oriented_run/metadata.json`
- `metadata_yaml`: `data/sovi_2021/output/sovi_like_quebec_cd_2021_38var_oriented_run/metadata.yaml`
- `validation_report_json`: `data/sovi_2021/output/sovi_like_quebec_cd_2021_38var_oriented_run/validation_report.json`
- `validation_report_yaml`: `data/sovi_2021/output/sovi_like_quebec_cd_2021_38var_oriented_run/validation_report.yaml`
- `missing_data_report_json`: `data/sovi_2021/output/sovi_like_quebec_cd_2021_38var_oriented_run/missing_data_report.json`
- `run_report`: `data/sovi_2021/output/sovi_like_quebec_cd_2021_38var_oriented_run/run_report.md`
- `sovi_eigenvalues`: `data/sovi_2021/output/sovi_like_quebec_cd_2021_38var_oriented_run/sovi_eigenvalues.csv`
- `sovi_explained_variance`: `data/sovi_2021/output/sovi_like_quebec_cd_2021_38var_oriented_run/sovi_explained_variance.csv`
- `sovi_loadings_unrotated`: `data/sovi_2021/output/sovi_like_quebec_cd_2021_38var_oriented_run/sovi_loadings_unrotated.csv`
- `sovi_loadings_rotated`: `data/sovi_2021/output/sovi_like_quebec_cd_2021_38var_oriented_run/sovi_loadings_rotated.csv`
- `sovi_factor_scores`: `data/sovi_2021/output/sovi_like_quebec_cd_2021_38var_oriented_run/sovi_factor_scores.csv`
- `sovi_factor_summary`: `data/sovi_2021/output/sovi_like_quebec_cd_2021_38var_oriented_run/sovi_factor_summary.csv`
- `sovi_standardized_variables`: `data/sovi_2021/output/sovi_like_quebec_cd_2021_38var_oriented_run/sovi_standardized_variables.csv`

## Top 5 Highest-Score Zones

| zone_id | score_raw | rank |
| --- | --- | --- |
| 2021A00032466 | 9.645657244538988 | 1 |
| 2021A00032499 | 6.763848806596924 | 2 |
| 2021A00032498 | 6.564586670164603 | 3 |
| 2021A00032404 | 6.030410969882683 | 4 |
| 2021A00032443 | 5.924793262612694 | 5 |

## Top 5 Lowest-Score Zones

| zone_id | score_raw | rank |
| --- | --- | --- |
| 2021A00032422 | -3.7518151527990415 | 98 |
| 2021A00032482 | -1.6576979940932546 | 97 |
| 2021A00032420 | -1.441391207907861 | 96 |
| 2021A00032442 | -1.3427449541801697 | 95 |
| 2021A00032419 | -1.264076526546748 | 94 |
