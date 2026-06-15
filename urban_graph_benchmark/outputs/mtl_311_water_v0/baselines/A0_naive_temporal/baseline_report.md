# A0 Naive Temporal/Exposure Baselines — Montréal 311 Water/Drainage v0

Generated at: `2026-06-15T17:17:33.595889+00:00`

Split scheme: `temporal`

Split type: `temporal`

## Purpose

A0 establishes the minimum non-graph benchmark: train-set means, seasonality, tract history, temporal persistence, population exposure, and a retrospective same-month non-water 311 reporting exposure baseline. These models provide the floor that calibrated SVI, tabular ML, GraphSAGE, and HGNN variants must beat.

## Row counts

| Partition | Rows |
|---|---:|
| `train` | 19440 |
| `validation` | 4320 |
| `test` | 4860 |

## Implemented baselines

| Model | Prediction setting | Feature set | Notes |
|---|---|---|---|
| `A0_1_global_train_mean` | `forecasting_v0` | `global_train_target_mean` | Constant prediction equal to the training-set mean count. |
| `A0_2_month_of_year_train_mean` | `forecasting_v0` | `month_of_year_train_target_mean` | Training-set month-of-year mean with global fallback. |
| `A0_3_tract_train_mean` | `forecasting_v0` | `tract_train_target_mean` | Training-set tract mean with global fallback. |
| `A0_4_tract_month_of_year_train_mean` | `forecasting_v0` | `tract_month_of_year_train_target_mean` | Training-set tract × month-of-year mean with hierarchical fallback. |
| `A0_5_previous_month_persistence` | `one_step_observed_history_v0` | `lag1_observed_target_with_train_fallback` | Previous observed month for the same tract with train-only fallback. |
| `A0_6_previous_year_same_month_persistence` | `one_step_observed_history_v0` | `lag12_observed_target_with_train_fallback` | Previous-year same-month observed count with train-only fallback. |
| `A0_7_population_exposure_train_rate` | `forecasting_v0` | `population_exposure_train_rate` | Global train target rate per person-month multiplied by tract population. |
| `A0_8_non_water_311_reporting_exposure_retrospective` | `retrospective_explanatory_v0` | `same_month_non_water_311_reporting_exposure` | Same-month non-water 311 reporting exposure. Retrospective only. |

## Compact metrics summary

```text
   split_name           prediction_setting                                          model_name                  metric_name  metric_value  higher_is_better  n_rows
temporal_test               forecasting_v0                              A0_1_global_train_mean                   count__mae      3.689900             False    4860
temporal_test               forecasting_v0                       A0_2_month_of_year_train_mean                   count__mae      3.654103             False    4860
temporal_test               forecasting_v0                               A0_3_tract_train_mean                   count__mae      2.489209             False    4860
temporal_test               forecasting_v0                 A0_4_tract_month_of_year_train_mean                   count__mae      2.739026             False    4860
temporal_test one_step_observed_history_v0                     A0_5_previous_month_persistence                   count__mae      3.163786             False    4860
temporal_test one_step_observed_history_v0           A0_6_previous_year_same_month_persistence                   count__mae      3.220988             False    4860
temporal_test               forecasting_v0                 A0_7_population_exposure_train_rate                   count__mae      3.501857             False    4860
temporal_test retrospective_explanatory_v0 A0_8_non_water_311_reporting_exposure_retrospective                   count__mae      3.011912             False    4860
temporal_test               forecasting_v0                              A0_1_global_train_mean count__mean_poisson_deviance      4.432920             False    4860
temporal_test               forecasting_v0                       A0_2_month_of_year_train_mean count__mean_poisson_deviance      4.530709             False    4860
temporal_test               forecasting_v0                               A0_3_tract_train_mean count__mean_poisson_deviance      2.143367             False    4860
temporal_test               forecasting_v0                 A0_4_tract_month_of_year_train_mean count__mean_poisson_deviance      3.333048             False    4860
temporal_test one_step_observed_history_v0                     A0_5_previous_month_persistence count__mean_poisson_deviance     11.224499             False    4860
temporal_test one_step_observed_history_v0           A0_6_previous_year_same_month_persistence count__mean_poisson_deviance     16.101542             False    4860
temporal_test               forecasting_v0                 A0_7_population_exposure_train_rate count__mean_poisson_deviance      5.058144             False    4860
temporal_test retrospective_explanatory_v0 A0_8_non_water_311_reporting_exposure_retrospective count__mean_poisson_deviance      2.843425             False    4860
temporal_test               forecasting_v0                              A0_1_global_train_mean                  count__rmse      4.790362             False    4860
temporal_test               forecasting_v0                       A0_2_month_of_year_train_mean                  count__rmse      4.846836             False    4860
temporal_test               forecasting_v0                               A0_3_tract_train_mean                  count__rmse      3.462372             False    4860
temporal_test               forecasting_v0                 A0_4_tract_month_of_year_train_mean                  count__rmse      3.914971             False    4860
temporal_test one_step_observed_history_v0                     A0_5_previous_month_persistence                  count__rmse      4.542031             False    4860
temporal_test one_step_observed_history_v0           A0_6_previous_year_same_month_persistence                  count__rmse      4.745585             False    4860
temporal_test               forecasting_v0                 A0_7_population_exposure_train_rate                  count__rmse      4.671557             False    4860
temporal_test retrospective_explanatory_v0 A0_8_non_water_311_reporting_exposure_retrospective                  count__rmse      4.370518             False    4860
temporal_test               forecasting_v0                              A0_1_global_train_mean        ranking__kendall_corr           NaN              True    4860
temporal_test               forecasting_v0                       A0_2_month_of_year_train_mean        ranking__kendall_corr      0.010439              True    4860
temporal_test               forecasting_v0                               A0_3_tract_train_mean        ranking__kendall_corr      0.531148              True    4860
temporal_test               forecasting_v0                 A0_4_tract_month_of_year_train_mean        ranking__kendall_corr      0.469377              True    4860
temporal_test one_step_observed_history_v0                     A0_5_previous_month_persistence        ranking__kendall_corr      0.455903              True    4860
temporal_test one_step_observed_history_v0           A0_6_previous_year_same_month_persistence        ranking__kendall_corr      0.407876              True    4860
temporal_test               forecasting_v0                 A0_7_population_exposure_train_rate        ranking__kendall_corr      0.198212              True    4860
temporal_test retrospective_explanatory_v0 A0_8_non_water_311_reporting_exposure_retrospective        ranking__kendall_corr      0.475295              True    4860
temporal_test               forecasting_v0                              A0_1_global_train_mean          ranking__ndcg_at_10      0.150196              True    4860
temporal_test               forecasting_v0                       A0_2_month_of_year_train_mean          ranking__ndcg_at_10      0.112734              True    4860
temporal_test               forecasting_v0                               A0_3_tract_train_mean          ranking__ndcg_at_10      0.729174              True    4860
temporal_test               forecasting_v0                 A0_4_tract_month_of_year_train_mean          ranking__ndcg_at_10      0.476199              True    4860
temporal_test one_step_observed_history_v0                     A0_5_previous_month_persistence          ranking__ndcg_at_10      0.521926              True    4860
temporal_test one_step_observed_history_v0           A0_6_previous_year_same_month_persistence          ranking__ndcg_at_10      0.393614              True    4860
temporal_test               forecasting_v0                 A0_7_population_exposure_train_rate          ranking__ndcg_at_10      0.201897              True    4860
temporal_test retrospective_explanatory_v0 A0_8_non_water_311_reporting_exposure_retrospective          ranking__ndcg_at_10      0.755168              True    4860
temporal_test               forecasting_v0                              A0_1_global_train_mean         ranking__ndcg_at_100      0.310433              True    4860
temporal_test               forecasting_v0                       A0_2_month_of_year_train_mean         ranking__ndcg_at_100      0.133566              True    4860
temporal_test               forecasting_v0                               A0_3_tract_train_mean         ranking__ndcg_at_100      0.748363              True    4860
temporal_test               forecasting_v0                 A0_4_tract_month_of_year_train_mean         ranking__ndcg_at_100      0.636779              True    4860
temporal_test one_step_observed_history_v0                     A0_5_previous_month_persistence         ranking__ndcg_at_100      0.598551              True    4860
temporal_test one_step_observed_history_v0           A0_6_previous_year_same_month_persistence         ranking__ndcg_at_100      0.536562              True    4860
temporal_test               forecasting_v0                 A0_7_population_exposure_train_rate         ranking__ndcg_at_100      0.261060              True    4860
temporal_test retrospective_explanatory_v0 A0_8_non_water_311_reporting_exposure_retrospective         ranking__ndcg_at_100      0.662923              True    4860
temporal_test               forecasting_v0                              A0_1_global_train_mean          ranking__ndcg_at_25      0.227206              True    4860
temporal_test               forecasting_v0                       A0_2_month_of_year_train_mean          ranking__ndcg_at_25      0.117494              True    4860
temporal_test               forecasting_v0                               A0_3_tract_train_mean          ranking__ndcg_at_25      0.684787              True    4860
temporal_test               forecasting_v0                 A0_4_tract_month_of_year_train_mean          ranking__ndcg_at_25      0.551875              True    4860
temporal_test one_step_observed_history_v0                     A0_5_previous_month_persistence          ranking__ndcg_at_25      0.514557              True    4860
temporal_test one_step_observed_history_v0           A0_6_previous_year_same_month_persistence          ranking__ndcg_at_25      0.434591              True    4860
temporal_test               forecasting_v0                 A0_7_population_exposure_train_rate          ranking__ndcg_at_25      0.124945              True    4860
temporal_test retrospective_explanatory_v0 A0_8_non_water_311_reporting_exposure_retrospective          ranking__ndcg_at_25      0.704381              True    4860
temporal_test               forecasting_v0                              A0_1_global_train_mean          ranking__ndcg_at_50      0.241498              True    4860
temporal_test               forecasting_v0                       A0_2_month_of_year_train_mean          ranking__ndcg_at_50      0.125536              True    4860
temporal_test               forecasting_v0                               A0_3_tract_train_mean          ranking__ndcg_at_50      0.721416              True    4860
temporal_test               forecasting_v0                 A0_4_tract_month_of_year_train_mean          ranking__ndcg_at_50      0.597714              True    4860
temporal_test one_step_observed_history_v0                     A0_5_previous_month_persistence          ranking__ndcg_at_50      0.569246              True    4860
temporal_test one_step_observed_history_v0           A0_6_previous_year_same_month_persistence          ranking__ndcg_at_50      0.501258              True    4860
temporal_test               forecasting_v0                 A0_7_population_exposure_train_rate          ranking__ndcg_at_50      0.218093              True    4860
temporal_test retrospective_explanatory_v0 A0_8_non_water_311_reporting_exposure_retrospective          ranking__ndcg_at_50      0.696355              True    4860
temporal_test               forecasting_v0                              A0_1_global_train_mean       ranking__spearman_corr           NaN              True    4860
temporal_test               forecasting_v0                       A0_2_month_of_year_train_mean       ranking__spearman_corr      0.014116              True    4860
temporal_test               forecasting_v0                               A0_3_tract_train_mean       ranking__spearman_corr      0.694235              True    4860
temporal_test               forecasting_v0                 A0_4_tract_month_of_year_train_mean       ranking__spearman_corr      0.616683              True    4860
temporal_test one_step_observed_history_v0                     A0_5_previous_month_persistence       ranking__spearman_corr      0.593867              True    4860
temporal_test one_step_observed_history_v0           A0_6_previous_year_same_month_persistence       ranking__spearman_corr      0.533484              True    4860
temporal_test               forecasting_v0                 A0_7_population_exposure_train_rate       ranking__spearman_corr      0.274328              True    4860
temporal_test retrospective_explanatory_v0 A0_8_non_water_311_reporting_exposure_retrospective       ranking__spearman_corr      0.631671              True    4860
temporal_test               forecasting_v0                              A0_1_global_train_mean ranking__top100_overlap_rate      0.060000              True    4860
temporal_test               forecasting_v0                       A0_2_month_of_year_train_mean ranking__top100_overlap_rate      0.000000              True    4860
temporal_test               forecasting_v0                               A0_3_tract_train_mean ranking__top100_overlap_rate      0.400000              True    4860
temporal_test               forecasting_v0                 A0_4_tract_month_of_year_train_mean ranking__top100_overlap_rate      0.300000              True    4860
temporal_test one_step_observed_history_v0                     A0_5_previous_month_persistence ranking__top100_overlap_rate      0.270000              True    4860
temporal_test one_step_observed_history_v0           A0_6_previous_year_same_month_persistence ranking__top100_overlap_rate      0.210000              True    4860
temporal_test               forecasting_v0                 A0_7_population_exposure_train_rate ranking__top100_overlap_rate      0.080000              True    4860
temporal_test retrospective_explanatory_v0 A0_8_non_water_311_reporting_exposure_retrospective ranking__top100_overlap_rate      0.290000              True    4860
```

## Output artifacts

| Artifact | Path |
|---|---|
| `metrics` | `/home/tim/Documents/ville_ai/indices_BACKUP_before_clean_push/urban_graph_benchmark/outputs/mtl_311_water_v0/baselines/A0_naive_temporal/metrics.csv` |
| `model_metadata` | `/home/tim/Documents/ville_ai/indices_BACKUP_before_clean_push/urban_graph_benchmark/outputs/mtl_311_water_v0/baselines/A0_naive_temporal/model_metadata.json` |
| `baseline_report` | `/home/tim/Documents/ville_ai/indices_BACKUP_before_clean_push/urban_graph_benchmark/outputs/mtl_311_water_v0/baselines/A0_naive_temporal/baseline_report.md` |
| `predictions_validation` | `/home/tim/Documents/ville_ai/indices_BACKUP_before_clean_push/urban_graph_benchmark/outputs/mtl_311_water_v0/baselines/A0_naive_temporal/predictions_validation.parquet` |
| `predictions_test` | `/home/tim/Documents/ville_ai/indices_BACKUP_before_clean_push/urban_graph_benchmark/outputs/mtl_311_water_v0/baselines/A0_naive_temporal/predictions_test.parquet` |

## Leakage notes

- A0.1–A0.4 and A0.7 are fitted using training rows only.
- A0.5 and A0.6 use strictly past observed target history for persistence baselines.
- A0.8 uses same-month `total_311_count_non_water_drainage`; it is retrospective/explanatory only, not a strict forecasting baseline.
- No SoVI columns are used in Track A.
