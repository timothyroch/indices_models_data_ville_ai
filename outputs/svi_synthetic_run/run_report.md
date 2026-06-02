# Run Report: svi_like

- Run ID: `svi_like-08ade61aafc3`
- Index version: `0.1`
- Reproduction level: `local_adaptation`
- Construct measured: `social_vulnerability`
- Score direction: `higher_is_more_vulnerable`
- Input feature table: `data/example/synthetic_svi_feature_table.csv`
- Spatial units input: `6`
- Spatial units output: `6`
- Source method: `{'citation': 'Flanagan et al. 2011, A Social Vulnerability Index for Disaster Management', 'doi': '10.2202/1547-7355.1792'}`
- Comparison scope: `global`
- Zero/invalid population units excluded: `0`
- Ranking formula: `(rank - 1) / (N - 1)`
- Tie method: `min`
- Flag threshold: `0.9`

## Variables Used

- pct_below_poverty
- pct_unemployed
- per_capita_income
- pct_no_high_school
- pct_age_65_plus
- pct_age_17_or_younger
- pct_disability
- pct_single_parent_households
- pct_minority
- pct_limited_language
- pct_multiunit_structures
- pct_mobile_homes
- pct_crowding
- pct_no_vehicle
- pct_group_quarters

## Variables Missing

- None

## Warnings

- None

## Proxies Used

- None

## Missing Data Summary

- Strategy: `error`
- Affected spatial units: `0`

## Normalization Summary

- `pct_below_poverty` / `pct_below_poverty`: `percentile_rank`
- `pct_unemployed` / `pct_unemployed`: `percentile_rank`
- `per_capita_income` / `per_capita_income`: `percentile_rank`
- `pct_no_high_school` / `pct_no_high_school`: `percentile_rank`
- `pct_age_65_plus` / `pct_age_65_plus`: `percentile_rank`
- `pct_age_17_or_younger` / `pct_age_17_or_younger`: `percentile_rank`
- `pct_disability` / `pct_disability`: `percentile_rank`
- `pct_single_parent_households` / `pct_single_parent_households`: `percentile_rank`
- `pct_minority` / `pct_minority`: `percentile_rank`
- `pct_limited_language` / `pct_limited_language`: `percentile_rank`
- `pct_multiunit_structures` / `pct_multiunit_structures`: `percentile_rank`
- `pct_mobile_homes` / `pct_mobile_homes`: `percentile_rank`
- `pct_crowding` / `pct_crowding`: `percentile_rank`
- `pct_no_vehicle` / `pct_no_vehicle`: `percentile_rank`
- `pct_group_quarters` / `pct_group_quarters`: `percentile_rank`

## Aggregation Summary

- Method: `svi_domain_sum_then_domain_percentile_then_overall_percentile`
- Weighting: `SVI equal variable ranks within domains; equal domain percentiles overall`

## SVI Distribution

- Minimum overall percentile: `0.0`
- Mean overall percentile: `0.5`
- Maximum overall percentile: `1.0`

## SVI Domain Summary

- `socioeconomic`: mean percentile `0.5`
- `household_disability`: mean percentile `0.5`
- `minority_language`: mean percentile `0.5`
- `housing_transportation`: mean percentile `0.5`

## Interpretation Warning

SVI is an area-level index and should not be interpreted as saying that every person in a high-SVI zone is vulnerable.

## Output Files

- `standard_output`: `outputs/svi_synthetic_run/standard_output.csv`
- `intermediate_output`: `outputs/svi_synthetic_run/intermediate_output.csv`
- `metadata_json`: `outputs/svi_synthetic_run/metadata.json`
- `metadata_yaml`: `outputs/svi_synthetic_run/metadata.yaml`
- `validation_report_json`: `outputs/svi_synthetic_run/validation_report.json`
- `validation_report_yaml`: `outputs/svi_synthetic_run/validation_report.yaml`
- `missing_data_report_json`: `outputs/svi_synthetic_run/missing_data_report.json`
- `run_report`: `outputs/svi_synthetic_run/run_report.md`

## Top 5 Highest-Score Zones

| zone_id | score_raw | rank |
| --- | --- | --- |
| HIGH | 4.0 | 1 |
| MID4 | 3.2 | 2 |
| MID3 | 2.4 | 3 |
| MID2 | 1.6 | 4 |
| MID1 | 0.8 | 5 |

## Top 5 Lowest-Score Zones

| zone_id | score_raw | rank |
| --- | --- | --- |
| LOW | 0.0 | 6 |
| MID1 | 0.8 | 5 |
| MID2 | 1.6 | 4 |
| MID3 | 2.4 | 3 |
| MID4 | 3.2 | 2 |
