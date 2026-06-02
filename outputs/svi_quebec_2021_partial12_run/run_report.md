# Run Report: svi_like

- Run ID: `svi_like-e520385289e0`
- Index version: `0.3-quebec-2021-partial12`
- Reproduction level: `partial_svi_like`
- Construct measured: `social_vulnerability`
- Score direction: `higher_is_more_vulnerable`
- Input feature table: `data/svi_2021/output/clean_quebec_census_tract_svi_input_2021.csv`
- Spatial units input: `1480`
- Spatial units output: `1470`
- Source method: `{'citation': 'Flanagan et al. 2011, A Social Vulnerability Index for Disaster Management', 'doi': '10.2202/1547-7355.1792'}`
- Comparison scope: `global`
- Zero/invalid population units excluded: `10`
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
- pct_single_parent_households
- pct_minority
- pct_limited_language
- pct_multiunit_structures
- pct_mobile_homes
- pct_crowding

## Variables Missing

- None

## Warnings

- Variable 'pct_below_poverty' contains missing values.
- Variable 'pct_unemployed' contains missing values.
- Variable 'per_capita_income' contains missing values.
- Variable 'pct_no_high_school' contains missing values.
- Variable 'pct_age_65_plus' contains missing values.
- Variable 'pct_age_17_or_younger' contains missing values.
- Variable 'pct_single_parent_households' contains missing values.
- Variable 'pct_minority' contains missing values.
- Variable 'pct_limited_language' contains missing values.
- Variable 'pct_multiunit_structures' contains missing values.
- Variable 'pct_mobile_homes' contains missing values.
- Variable 'pct_crowding' contains missing values.
- SVI will apply the configured zero/invalid population rule to 10 unit(s).
- This run is partial_svi_like and is not a full SVI-like reproduction.
- One or more SVI variables use documented local proxies.
- 10 unit(s) excluded because population <= 0.0.

## Proxies Used

- pct_below_poverty -> pct_below_poverty (high)
- per_capita_income -> per_capita_income (medium)
- pct_no_high_school -> pct_no_high_school (medium_high)
- pct_age_17_or_younger -> pct_age_17_or_younger (medium)
- pct_single_parent_households -> pct_single_parent_households (medium)
- pct_minority -> pct_minority (medium)
- pct_limited_language -> pct_limited_language (medium)
- pct_multiunit_structures -> pct_multiunit_structures (high)
- pct_mobile_homes -> pct_mobile_homes (high)
- pct_crowding -> pct_crowding (high)

## Missing Data Summary

- Strategy: `keep_missing_with_flags`
- Affected spatial units: `22`

## Normalization Summary

- `pct_below_poverty` / `pct_below_poverty`: `percentile_rank`
- `pct_unemployed` / `pct_unemployed`: `percentile_rank`
- `per_capita_income` / `per_capita_income`: `percentile_rank`
- `pct_no_high_school` / `pct_no_high_school`: `percentile_rank`
- `pct_age_65_plus` / `pct_age_65_plus`: `percentile_rank`
- `pct_age_17_or_younger` / `pct_age_17_or_younger`: `percentile_rank`
- `pct_single_parent_households` / `pct_single_parent_households`: `percentile_rank`
- `pct_minority` / `pct_minority`: `percentile_rank`
- `pct_limited_language` / `pct_limited_language`: `percentile_rank`
- `pct_multiunit_structures` / `pct_multiunit_structures`: `percentile_rank`
- `pct_mobile_homes` / `pct_mobile_homes`: `percentile_rank`
- `pct_crowding` / `pct_crowding`: `percentile_rank`

## Aggregation Summary

- Method: `svi_domain_sum_then_domain_percentile_then_overall_percentile`
- Weighting: `SVI equal variable ranks within domains; equal domain percentiles overall`

## SVI Distribution

- Minimum overall percentile: `0.0`
- Mean overall percentile: `0.4999376482611834`
- Maximum overall percentile: `1.0`

## SVI Domain Summary

- `socioeconomic`: mean percentile `0.5`
- `household_disability`: mean percentile `0.4998645138908423`
- `minority_language`: mean percentile `0.4997487178044686`
- `housing_transportation`: mean percentile `0.4998635762707097`

## Interpretation Warning

SVI is an area-level index and should not be interpreted as saying that every person in a high-SVI zone is vulnerable.

## Output Files

- `standard_output`: `outputs/svi_quebec_2021_partial12_run/standard_output.csv`
- `intermediate_output`: `outputs/svi_quebec_2021_partial12_run/intermediate_output.csv`
- `metadata_json`: `outputs/svi_quebec_2021_partial12_run/metadata.json`
- `metadata_yaml`: `outputs/svi_quebec_2021_partial12_run/metadata.yaml`
- `validation_report_json`: `outputs/svi_quebec_2021_partial12_run/validation_report.json`
- `validation_report_yaml`: `outputs/svi_quebec_2021_partial12_run/validation_report.yaml`
- `missing_data_report_json`: `outputs/svi_quebec_2021_partial12_run/missing_data_report.json`
- `run_report`: `outputs/svi_quebec_2021_partial12_run/run_report.md`

## Top 5 Highest-Score Zones

| zone_id | score_raw | rank |
| --- | --- | --- |
| 2021S05074620646.01 | 3.9273972602739726 | 1 |
| 2021S05074620257.00 | 3.913013698630137 | 2 |
| 2021S05074620616.00 | 3.882876712328767 | 3 |
| 2021S05074620259.00 | 3.860958904109589 | 4 |
| 2021S05074620262.00 | 3.841095890410959 | 5 |

## Top 5 Lowest-Score Zones

| zone_id | score_raw | rank |
| --- | --- | --- |
| 2021S05075050841.09 | 0.2705479452054794 | 1461 |
| 2021S05074080111.04 | 0.3404109589041096 | 1460 |
| 2021S05074210320.05 | 0.3856164383561644 | 1459 |
| 2021S05074210600.05 | 0.4178082191780822 | 1458 |
| 2021S05074622203.00 | 0.4541095890410959 | 1457 |
