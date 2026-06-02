# Run Report: dummy_additive_index

- Run ID: `dummy_additive_index-5a2566a75e94`
- Index version: `0.1`
- Reproduction level: `toy_validation_only`
- Construct measured: `synthetic_vulnerability`
- Score direction: `higher_is_more_vulnerable`
- Input feature table: `data/example/synthetic_feature_table.csv`
- Spatial units input: `5`
- Spatial units output: `5`

## Variables Used

- median_household_income
- pct_65_plus
- floodplain_pct

## Variables Missing

- None

## Warnings

- None

## Missing Data Summary

- Strategy: `error`
- Affected spatial units: `0`

## Normalization Summary

- `income` / `median_household_income`: `minmax`
- `age` / `pct_65_plus`: `minmax`
- `flood` / `floodplain_pct`: `minmax`

## Aggregation Summary

- Method: `weighted_sum`
- Weighting: `variable_weights_from_recipe`

## Output Files

- `standard_output`: `outputs/dummy_run/standard_output.csv`
- `intermediate_output`: `outputs/dummy_run/intermediate_output.csv`
- `metadata_json`: `outputs/dummy_run/metadata.json`
- `metadata_yaml`: `outputs/dummy_run/metadata.yaml`
- `validation_report_json`: `outputs/dummy_run/validation_report.json`
- `validation_report_yaml`: `outputs/dummy_run/validation_report.yaml`
- `missing_data_report_json`: `outputs/dummy_run/missing_data_report.json`
- `run_report`: `outputs/dummy_run/run_report.md`

## Top 5 Highest-Score Zones

| zone_id | score_raw | rank |
| --- | --- | --- |
| B | 1.0 | 1 |
| D | 0.7050000000000001 | 2 |
| C | 0.47 | 3 |
| E | 0.1975 | 4 |
| A | 0.0 | 5 |

## Top 5 Lowest-Score Zones

| zone_id | score_raw | rank |
| --- | --- | --- |
| A | 0.0 | 5 |
| E | 0.1975 | 4 |
| C | 0.47 | 3 |
| D | 0.7050000000000001 | 2 |
| B | 1.0 | 1 |
