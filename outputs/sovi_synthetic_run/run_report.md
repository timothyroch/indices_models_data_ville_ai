# Run Report: sovi_like

- Run ID: `sovi_like-1665ad202680`
- Index version: `0.1-synthetic`
- Reproduction level: `local_adaptation`
- Construct measured: `social_vulnerability`
- Score direction: `higher_is_more_vulnerable`
- Input feature table: `data/example/synthetic_sovi_feature_table.csv`
- Spatial units input: `24`
- Spatial units output: `24`
- Source method: `synthetic fixed-factor SoVI-like example`
- PCA method: `pca`
- Factor retention: `fixed_n`
- Factors retained: `3`
- Rotation method: `varimax`

## Variables Used

- median_income
- pct_poverty
- pct_unemployed
- pct_no_high_school
- pct_age_65_plus
- pct_children
- pct_renter
- pct_mobile_homes
- housing_density
- commercial_density
- pct_service_employment
- pct_female_headed_households

## Variables Missing

- None

## Warnings

- Variable 'pct_mobile_homes' contains missing values.
- Factor signs are arbitrary and require orientation decisions.
- SoVI is an area-level social vulnerability score and should not be interpreted as an individual-level diagnosis.
- This run is a SoVI-like local/partial adaptation, not exact original SoVI.
- Zero imputation was used; this may distort factor structure when zero is not substantively meaningful.

## Proxies Used

- None

## Missing Data Summary

- Strategy: `zero_imputation`
- Affected spatial units: `1`

## Normalization Summary

- `median_income` / `median_income`: `zscore`
- `pct_poverty` / `pct_poverty`: `zscore`
- `pct_unemployed` / `pct_unemployed`: `zscore`
- `pct_no_high_school` / `pct_no_high_school`: `zscore`
- `pct_age_65_plus` / `pct_age_65_plus`: `zscore`
- `pct_children` / `pct_children`: `zscore`
- `pct_renter` / `pct_renter`: `zscore`
- `pct_mobile_homes` / `pct_mobile_homes`: `zscore`
- `housing_density` / `housing_density`: `zscore`
- `commercial_density` / `commercial_density`: `zscore`
- `pct_service_employment` / `pct_service_employment`: `zscore`
- `pct_female_headed_households` / `pct_female_headed_households`: `zscore`

## Aggregation Summary

- Method: `additive_factor_sum`
- Weighting: `equal additive factor scores`

## SoVI Factor Analysis

- Standardization: `zscore`
- PCA method: `pca`
- Factor-retention rule: `fixed_n`
- Retained factors: `3`
- Retained explained variance: `0.9995834689326782`
- Rotation: `varimax`

## SoVI Factor Orientation Summary

- `factor_1`: `positive`; Synthetic socioeconomic disadvantage factor increases vulnerability.
- `factor_2`: `negative`; Synthetic factor is flipped to test negative orientation.
- `factor_3`: `absolute`; Synthetic ambiguous density factor uses absolute value.

## SoVI Dominant Variables

- `factor_1`: median_income:-0.8658; pct_renter:0.8215; pct_age_65_plus:0.8160; pct_children:0.8160; pct_unemployed:0.8160
- `factor_2`: commercial_density:0.8430; housing_density:0.8290; pct_poverty:0.6367; pct_service_employment:0.6367; pct_female_headed_households:0.6367
- `factor_3`: pct_mobile_homes:-0.3242; pct_poverty:-0.1281; pct_female_headed_households:-0.1281; pct_service_employment:-0.1281; pct_no_high_school:-0.1281

## SoVI Score Distribution

- Minimum score: `-2.462350813023899`
- Mean score: `0.4749416257456353`
- Maximum score: `4.405675349634734`
- Standard deviation: `1.6259867149927327`

## Interpretation Warning

SoVI is an area-level social vulnerability score and should not be interpreted as an individual-level diagnosis.

## Output Files

- `standard_output`: `outputs/sovi_synthetic_run/standard_output.csv`
- `intermediate_output`: `outputs/sovi_synthetic_run/intermediate_output.csv`
- `metadata_json`: `outputs/sovi_synthetic_run/metadata.json`
- `metadata_yaml`: `outputs/sovi_synthetic_run/metadata.yaml`
- `validation_report_json`: `outputs/sovi_synthetic_run/validation_report.json`
- `validation_report_yaml`: `outputs/sovi_synthetic_run/validation_report.yaml`
- `missing_data_report_json`: `outputs/sovi_synthetic_run/missing_data_report.json`
- `run_report`: `outputs/sovi_synthetic_run/run_report.md`
- `sovi_eigenvalues`: `outputs/sovi_synthetic_run/sovi_eigenvalues.csv`
- `sovi_explained_variance`: `outputs/sovi_synthetic_run/sovi_explained_variance.csv`
- `sovi_loadings_unrotated`: `outputs/sovi_synthetic_run/sovi_loadings_unrotated.csv`
- `sovi_loadings_rotated`: `outputs/sovi_synthetic_run/sovi_loadings_rotated.csv`
- `sovi_factor_scores`: `outputs/sovi_synthetic_run/sovi_factor_scores.csv`
- `sovi_factor_summary`: `outputs/sovi_synthetic_run/sovi_factor_summary.csv`
- `sovi_standardized_variables`: `outputs/sovi_synthetic_run/sovi_standardized_variables.csv`

## Top 5 Highest-Score Zones

| zone_id | score_raw | rank |
| --- | --- | --- |
| Z07 | 4.405675349634734 | 1 |
| Z13 | 2.172043238970012 | 2 |
| Z15 | 2.064818010127197 | 3 |
| Z11 | 1.971920987665951 | 4 |
| Z14 | 1.7408760350718369 | 5 |

## Top 5 Lowest-Score Zones

| zone_id | score_raw | rank |
| --- | --- | --- |
| Z24 | -2.462350813023899 | 24 |
| Z01 | -2.058356974145247 | 23 |
| Z02 | -2.0104922793880537 | 22 |
| Z23 | -1.2941931424282704 | 21 |
| Z04 | -1.1593697537644092 | 20 |
