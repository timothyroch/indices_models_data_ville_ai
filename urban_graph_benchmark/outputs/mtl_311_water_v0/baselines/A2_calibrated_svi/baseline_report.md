# A2 Calibrated SVI Predictors — Montréal 311 Water/Drainage v0

Generated at: `2026-06-09T14:38:24.398802+00:00`

Split scheme: `temporal`

Split type: `temporal`

Ridge alpha: `1.0`

Primary-only mode: `False`

## Purpose

A2 tests whether static SVI has predictive value after simple calibration and basic controls. The model family is a ridge-regularized linear model fit on `log1p(water_drainage_count)` using training rows only. Predictions are converted back to count scale with `expm1` and clipped at zero.

## Row counts

| Partition | Rows |
|---|---:|
| `train` | 19440 |
| `validation` | 4320 |
| `test` | 4860 |

## Primary SVI scores used

| Source column | Oriented score column | Orientation |
|---|---|---|
| `svi_percentile` | `svi_percentile__higher_more_vulnerable` | `positive_higher_more_vulnerable` |
| `svi_score_raw` | `svi_score_raw__higher_more_vulnerable` | `positive_higher_more_vulnerable` |

## Diagnostic SVI scores used

| Source column | Score role | Oriented score column | Orientation |
|---|---|---|---|
| `svi_rank` | `diagnostic_rank_reversed_score` | `svi_rank__rank_reversed_for_vulnerability` | `negative_rank_reversed` |
| `svi_class` | `diagnostic_ordinal_class_score_not_primary` | `svi_class__higher_more_vulnerable` | `positive_higher_more_vulnerable` |

## Model families

| Feature set | Prediction setting | Count | Description |
|---|---|---:|---|
| `A2_svi_only` | `forecasting_v0` | 4 | SVI score only. |
| `A2_svi_plus_calendar` | `forecasting_v0` | 4 | SVI score plus month-of-year controls. |
| `A2_svi_plus_reporting_retrospective` | `retrospective_explanatory_v0` | 4 | SVI score plus static/calendar controls and same-month non-water 311 reporting exposure. Retrospective only. |
| `A2_svi_plus_static` | `forecasting_v0` | 4 | SVI score plus month-of-year, population, land area, and density controls when available. |

## Best models by validation/test MAE

```text
    benchmark_id dataset_version          split_name split_type           prediction_setting       model_stage                                                     model_name          target_name target_type                    feature_set_name metric_name  metric_value  higher_is_better  n_rows  n_train  n_validation  n_test                     notes
mtl_311_water_v0      dataset_v0       temporal_test   temporal retrospective_explanatory_v0 A2_calibrated_svi A2_svi_plus_reporting_retrospective__svi_class__diagnostic_svi water_drainage_count       count A2_svi_plus_reporting_retrospective  count__mae      2.522280             False    4860    19440          4320    4860       eval_partition=test
mtl_311_water_v0      dataset_v0       temporal_test   temporal retrospective_explanatory_v0 A2_calibrated_svi            A2_svi_plus_reporting_retrospective__svi_percentile water_drainage_count       count A2_svi_plus_reporting_retrospective  count__mae      2.522434             False    4860    19440          4320    4860       eval_partition=test
mtl_311_water_v0      dataset_v0       temporal_test   temporal retrospective_explanatory_v0 A2_calibrated_svi  A2_svi_plus_reporting_retrospective__svi_rank__diagnostic_svi water_drainage_count       count A2_svi_plus_reporting_retrospective  count__mae      2.523753             False    4860    19440          4320    4860       eval_partition=test
mtl_311_water_v0      dataset_v0       temporal_test   temporal retrospective_explanatory_v0 A2_calibrated_svi             A2_svi_plus_reporting_retrospective__svi_score_raw water_drainage_count       count A2_svi_plus_reporting_retrospective  count__mae      2.524465             False    4860    19440          4320    4860       eval_partition=test
mtl_311_water_v0      dataset_v0       temporal_test   temporal               forecasting_v0 A2_calibrated_svi                  A2_svi_plus_static__svi_class__diagnostic_svi water_drainage_count       count                  A2_svi_plus_static  count__mae      3.430960             False    4860    19440          4320    4860       eval_partition=test
mtl_311_water_v0      dataset_v0       temporal_test   temporal               forecasting_v0 A2_calibrated_svi                   A2_svi_plus_static__svi_rank__diagnostic_svi water_drainage_count       count                  A2_svi_plus_static  count__mae      3.434079             False    4860    19440          4320    4860       eval_partition=test
mtl_311_water_v0      dataset_v0       temporal_test   temporal               forecasting_v0 A2_calibrated_svi                             A2_svi_plus_static__svi_percentile water_drainage_count       count                  A2_svi_plus_static  count__mae      3.434751             False    4860    19440          4320    4860       eval_partition=test
mtl_311_water_v0      dataset_v0       temporal_test   temporal               forecasting_v0 A2_calibrated_svi                              A2_svi_plus_static__svi_score_raw water_drainage_count       count                  A2_svi_plus_static  count__mae      3.438431             False    4860    19440          4320    4860       eval_partition=test
mtl_311_water_v0      dataset_v0       temporal_test   temporal               forecasting_v0 A2_calibrated_svi                          A2_svi_only__svi_rank__diagnostic_svi water_drainage_count       count                         A2_svi_only  count__mae      3.466399             False    4860    19440          4320    4860       eval_partition=test
mtl_311_water_v0      dataset_v0       temporal_test   temporal               forecasting_v0 A2_calibrated_svi                         A2_svi_only__svi_class__diagnostic_svi water_drainage_count       count                         A2_svi_only  count__mae      3.468756             False    4860    19440          4320    4860       eval_partition=test
mtl_311_water_v0      dataset_v0 temporal_validation   temporal retrospective_explanatory_v0 A2_calibrated_svi            A2_svi_plus_reporting_retrospective__svi_percentile water_drainage_count       count A2_svi_plus_reporting_retrospective  count__mae      2.610554             False    4320    19440          4320    4860 eval_partition=validation
mtl_311_water_v0      dataset_v0 temporal_validation   temporal retrospective_explanatory_v0 A2_calibrated_svi A2_svi_plus_reporting_retrospective__svi_class__diagnostic_svi water_drainage_count       count A2_svi_plus_reporting_retrospective  count__mae      2.610572             False    4320    19440          4320    4860 eval_partition=validation
mtl_311_water_v0      dataset_v0 temporal_validation   temporal retrospective_explanatory_v0 A2_calibrated_svi  A2_svi_plus_reporting_retrospective__svi_rank__diagnostic_svi water_drainage_count       count A2_svi_plus_reporting_retrospective  count__mae      2.611130             False    4320    19440          4320    4860 eval_partition=validation
mtl_311_water_v0      dataset_v0 temporal_validation   temporal retrospective_explanatory_v0 A2_calibrated_svi             A2_svi_plus_reporting_retrospective__svi_score_raw water_drainage_count       count A2_svi_plus_reporting_retrospective  count__mae      2.611218             False    4320    19440          4320    4860 eval_partition=validation
mtl_311_water_v0      dataset_v0 temporal_validation   temporal               forecasting_v0 A2_calibrated_svi                  A2_svi_plus_static__svi_class__diagnostic_svi water_drainage_count       count                  A2_svi_plus_static  count__mae      3.481833             False    4320    19440          4320    4860 eval_partition=validation
mtl_311_water_v0      dataset_v0 temporal_validation   temporal               forecasting_v0 A2_calibrated_svi                   A2_svi_plus_static__svi_rank__diagnostic_svi water_drainage_count       count                  A2_svi_plus_static  count__mae      3.483578             False    4320    19440          4320    4860 eval_partition=validation
mtl_311_water_v0      dataset_v0 temporal_validation   temporal               forecasting_v0 A2_calibrated_svi                             A2_svi_plus_static__svi_percentile water_drainage_count       count                  A2_svi_plus_static  count__mae      3.483931             False    4320    19440          4320    4860 eval_partition=validation
mtl_311_water_v0      dataset_v0 temporal_validation   temporal               forecasting_v0 A2_calibrated_svi                              A2_svi_plus_static__svi_score_raw water_drainage_count       count                  A2_svi_plus_static  count__mae      3.484898             False    4320    19440          4320    4860 eval_partition=validation
mtl_311_water_v0      dataset_v0 temporal_validation   temporal               forecasting_v0 A2_calibrated_svi                 A2_svi_plus_calendar__svi_rank__diagnostic_svi water_drainage_count       count                A2_svi_plus_calendar  count__mae      3.589516             False    4320    19440          4320    4860 eval_partition=validation
mtl_311_water_v0      dataset_v0 temporal_validation   temporal               forecasting_v0 A2_calibrated_svi                A2_svi_plus_calendar__svi_class__diagnostic_svi water_drainage_count       count                A2_svi_plus_calendar  count__mae      3.593123             False    4320    19440          4320    4860 eval_partition=validation
```

## Compact metrics summary

```text
         split_name           prediction_setting                                          model_name                    feature_set_name                     metric_name  metric_value  higher_is_better  n_rows
temporal_validation               forecasting_v0                         A2_svi_only__svi_percentile                         A2_svi_only                      count__mae      3.713655             False    4320
temporal_validation               forecasting_v0                         A2_svi_only__svi_percentile                         A2_svi_only                     count__rmse      5.528800             False    4320
temporal_validation               forecasting_v0                         A2_svi_only__svi_percentile                         A2_svi_only    count__mean_poisson_deviance      5.441169             False    4320
temporal_validation               forecasting_v0                         A2_svi_only__svi_percentile                         A2_svi_only          ranking__spearman_corr      0.156450              True    4320
temporal_validation               forecasting_v0                         A2_svi_only__svi_percentile                         A2_svi_only            ranking__ndcg_at_100      0.193465              True    4320
temporal_validation               forecasting_v0                         A2_svi_only__svi_percentile                         A2_svi_only ranking__top_10pct_overlap_rate      0.067130              True    4320
      temporal_test               forecasting_v0                         A2_svi_only__svi_percentile                         A2_svi_only                      count__mae      3.475547             False    4860
      temporal_test               forecasting_v0                         A2_svi_only__svi_percentile                         A2_svi_only                     count__rmse      4.938114             False    4860
      temporal_test               forecasting_v0                         A2_svi_only__svi_percentile                         A2_svi_only    count__mean_poisson_deviance      4.801117             False    4860
      temporal_test               forecasting_v0                         A2_svi_only__svi_percentile                         A2_svi_only          ranking__spearman_corr      0.159008              True    4860
      temporal_test               forecasting_v0                         A2_svi_only__svi_percentile                         A2_svi_only            ranking__ndcg_at_100      0.220560              True    4860
      temporal_test               forecasting_v0                         A2_svi_only__svi_percentile                         A2_svi_only ranking__top_10pct_overlap_rate      0.057613              True    4860
temporal_validation               forecasting_v0                A2_svi_plus_calendar__svi_percentile                A2_svi_plus_calendar                      count__mae      3.599660             False    4320
temporal_validation               forecasting_v0                A2_svi_plus_calendar__svi_percentile                A2_svi_plus_calendar                     count__rmse      5.362666             False    4320
temporal_validation               forecasting_v0                A2_svi_plus_calendar__svi_percentile                A2_svi_plus_calendar    count__mean_poisson_deviance      5.008630             False    4320
temporal_validation               forecasting_v0                A2_svi_plus_calendar__svi_percentile                A2_svi_plus_calendar          ranking__spearman_corr      0.283787              True    4320
temporal_validation               forecasting_v0                A2_svi_plus_calendar__svi_percentile                A2_svi_plus_calendar            ranking__ndcg_at_100      0.221594              True    4320
temporal_validation               forecasting_v0                A2_svi_plus_calendar__svi_percentile                A2_svi_plus_calendar ranking__top_10pct_overlap_rate      0.134259              True    4320
      temporal_test               forecasting_v0                A2_svi_plus_calendar__svi_percentile                A2_svi_plus_calendar                      count__mae      3.535689             False    4860
      temporal_test               forecasting_v0                A2_svi_plus_calendar__svi_percentile                A2_svi_plus_calendar                     count__rmse      5.040499             False    4860
      temporal_test               forecasting_v0                A2_svi_plus_calendar__svi_percentile                A2_svi_plus_calendar    count__mean_poisson_deviance      5.119893             False    4860
      temporal_test               forecasting_v0                A2_svi_plus_calendar__svi_percentile                A2_svi_plus_calendar          ranking__spearman_corr      0.127134              True    4860
      temporal_test               forecasting_v0                A2_svi_plus_calendar__svi_percentile                A2_svi_plus_calendar            ranking__ndcg_at_100      0.121777              True    4860
      temporal_test               forecasting_v0                A2_svi_plus_calendar__svi_percentile                A2_svi_plus_calendar ranking__top_10pct_overlap_rate      0.069959              True    4860
temporal_validation               forecasting_v0                  A2_svi_plus_static__svi_percentile                  A2_svi_plus_static                      count__mae      3.483931             False    4320
temporal_validation               forecasting_v0                  A2_svi_plus_static__svi_percentile                  A2_svi_plus_static                     count__rmse      5.242381             False    4320
temporal_validation               forecasting_v0                  A2_svi_plus_static__svi_percentile                  A2_svi_plus_static    count__mean_poisson_deviance      4.793326             False    4320
temporal_validation               forecasting_v0                  A2_svi_plus_static__svi_percentile                  A2_svi_plus_static          ranking__spearman_corr      0.369378              True    4320
temporal_validation               forecasting_v0                  A2_svi_plus_static__svi_percentile                  A2_svi_plus_static            ranking__ndcg_at_100      0.293624              True    4320
temporal_validation               forecasting_v0                  A2_svi_plus_static__svi_percentile                  A2_svi_plus_static ranking__top_10pct_overlap_rate      0.240741              True    4320
      temporal_test               forecasting_v0                  A2_svi_plus_static__svi_percentile                  A2_svi_plus_static                      count__mae      3.434751             False    4860
      temporal_test               forecasting_v0                  A2_svi_plus_static__svi_percentile                  A2_svi_plus_static                     count__rmse      4.936054             False    4860
      temporal_test               forecasting_v0                  A2_svi_plus_static__svi_percentile                  A2_svi_plus_static    count__mean_poisson_deviance      4.952062             False    4860
      temporal_test               forecasting_v0                  A2_svi_plus_static__svi_percentile                  A2_svi_plus_static          ranking__spearman_corr      0.236370              True    4860
      temporal_test               forecasting_v0                  A2_svi_plus_static__svi_percentile                  A2_svi_plus_static            ranking__ndcg_at_100      0.193140              True    4860
      temporal_test               forecasting_v0                  A2_svi_plus_static__svi_percentile                  A2_svi_plus_static ranking__top_10pct_overlap_rate      0.121399              True    4860
temporal_validation retrospective_explanatory_v0 A2_svi_plus_reporting_retrospective__svi_percentile A2_svi_plus_reporting_retrospective                      count__mae      2.610554             False    4320
temporal_validation retrospective_explanatory_v0 A2_svi_plus_reporting_retrospective__svi_percentile A2_svi_plus_reporting_retrospective                     count__rmse      3.996411             False    4320
temporal_validation retrospective_explanatory_v0 A2_svi_plus_reporting_retrospective__svi_percentile A2_svi_plus_reporting_retrospective    count__mean_poisson_deviance      2.441922             False    4320
temporal_validation retrospective_explanatory_v0 A2_svi_plus_reporting_retrospective__svi_percentile A2_svi_plus_reporting_retrospective          ranking__spearman_corr      0.711523              True    4320
temporal_validation retrospective_explanatory_v0 A2_svi_plus_reporting_retrospective__svi_percentile A2_svi_plus_reporting_retrospective            ranking__ndcg_at_100      0.641888              True    4320
temporal_validation retrospective_explanatory_v0 A2_svi_plus_reporting_retrospective__svi_percentile A2_svi_plus_reporting_retrospective ranking__top_10pct_overlap_rate      0.509259              True    4320
      temporal_test retrospective_explanatory_v0 A2_svi_plus_reporting_retrospective__svi_percentile A2_svi_plus_reporting_retrospective                      count__mae      2.522434             False    4860
      temporal_test retrospective_explanatory_v0 A2_svi_plus_reporting_retrospective__svi_percentile A2_svi_plus_reporting_retrospective                     count__rmse      3.663199             False    4860
      temporal_test retrospective_explanatory_v0 A2_svi_plus_reporting_retrospective__svi_percentile A2_svi_plus_reporting_retrospective    count__mean_poisson_deviance      2.522685             False    4860
      temporal_test retrospective_explanatory_v0 A2_svi_plus_reporting_retrospective__svi_percentile A2_svi_plus_reporting_retrospective          ranking__spearman_corr      0.671736              True    4860
      temporal_test retrospective_explanatory_v0 A2_svi_plus_reporting_retrospective__svi_percentile A2_svi_plus_reporting_retrospective            ranking__ndcg_at_100      0.702508              True    4860
      temporal_test retrospective_explanatory_v0 A2_svi_plus_reporting_retrospective__svi_percentile A2_svi_plus_reporting_retrospective ranking__top_10pct_overlap_rate      0.462963              True    4860
temporal_validation               forecasting_v0                          A2_svi_only__svi_score_raw                         A2_svi_only                      count__mae      3.720264             False    4320
temporal_validation               forecasting_v0                          A2_svi_only__svi_score_raw                         A2_svi_only                     count__rmse      5.541045             False    4320
temporal_validation               forecasting_v0                          A2_svi_only__svi_score_raw                         A2_svi_only    count__mean_poisson_deviance      5.473238             False    4320
temporal_validation               forecasting_v0                          A2_svi_only__svi_score_raw                         A2_svi_only          ranking__spearman_corr      0.156464              True    4320
temporal_validation               forecasting_v0                          A2_svi_only__svi_score_raw                         A2_svi_only            ranking__ndcg_at_100      0.193465              True    4320
temporal_validation               forecasting_v0                          A2_svi_only__svi_score_raw                         A2_svi_only ranking__top_10pct_overlap_rate      0.067130              True    4320
      temporal_test               forecasting_v0                          A2_svi_only__svi_score_raw                         A2_svi_only                      count__mae      3.486610             False    4860
      temporal_test               forecasting_v0                          A2_svi_only__svi_score_raw                         A2_svi_only                     count__rmse      4.952153             False    4860
      temporal_test               forecasting_v0                          A2_svi_only__svi_score_raw                         A2_svi_only    count__mean_poisson_deviance      4.834633             False    4860
      temporal_test               forecasting_v0                          A2_svi_only__svi_score_raw                         A2_svi_only          ranking__spearman_corr      0.159019              True    4860
      temporal_test               forecasting_v0                          A2_svi_only__svi_score_raw                         A2_svi_only            ranking__ndcg_at_100      0.220560              True    4860
      temporal_test               forecasting_v0                          A2_svi_only__svi_score_raw                         A2_svi_only ranking__top_10pct_overlap_rate      0.057613              True    4860
temporal_validation               forecasting_v0                 A2_svi_plus_calendar__svi_score_raw                A2_svi_plus_calendar                      count__mae      3.607856             False    4320
temporal_validation               forecasting_v0                 A2_svi_plus_calendar__svi_score_raw                A2_svi_plus_calendar                     count__rmse      5.375903             False    4320
temporal_validation               forecasting_v0                 A2_svi_plus_calendar__svi_score_raw                A2_svi_plus_calendar    count__mean_poisson_deviance      5.039918             False    4320
temporal_validation               forecasting_v0                 A2_svi_plus_calendar__svi_score_raw                A2_svi_plus_calendar          ranking__spearman_corr      0.279234              True    4320
temporal_validation               forecasting_v0                 A2_svi_plus_calendar__svi_score_raw                A2_svi_plus_calendar            ranking__ndcg_at_100      0.212026              True    4320
temporal_validation               forecasting_v0                 A2_svi_plus_calendar__svi_score_raw                A2_svi_plus_calendar ranking__top_10pct_overlap_rate      0.134259              True    4320
      temporal_test               forecasting_v0                 A2_svi_plus_calendar__svi_score_raw                A2_svi_plus_calendar                      count__mae      3.545252             False    4860
      temporal_test               forecasting_v0                 A2_svi_plus_calendar__svi_score_raw                A2_svi_plus_calendar                     count__rmse      5.053910             False    4860
      temporal_test               forecasting_v0                 A2_svi_plus_calendar__svi_score_raw                A2_svi_plus_calendar    count__mean_poisson_deviance      5.154349             False    4860
      temporal_test               forecasting_v0                 A2_svi_plus_calendar__svi_score_raw                A2_svi_plus_calendar          ranking__spearman_corr      0.113222              True    4860
      temporal_test               forecasting_v0                 A2_svi_plus_calendar__svi_score_raw                A2_svi_plus_calendar            ranking__ndcg_at_100      0.120880              True    4860
      temporal_test               forecasting_v0                 A2_svi_plus_calendar__svi_score_raw                A2_svi_plus_calendar ranking__top_10pct_overlap_rate      0.084362              True    4860
temporal_validation               forecasting_v0                   A2_svi_plus_static__svi_score_raw                  A2_svi_plus_static                      count__mae      3.484898             False    4320
temporal_validation               forecasting_v0                   A2_svi_plus_static__svi_score_raw                  A2_svi_plus_static                     count__rmse      5.248224             False    4320
temporal_validation               forecasting_v0                   A2_svi_plus_static__svi_score_raw                  A2_svi_plus_static    count__mean_poisson_deviance      4.810630             False    4320
temporal_validation               forecasting_v0                   A2_svi_plus_static__svi_score_raw                  A2_svi_plus_static          ranking__spearman_corr      0.370049              True    4320
temporal_validation               forecasting_v0                   A2_svi_plus_static__svi_score_raw                  A2_svi_plus_static            ranking__ndcg_at_100      0.307473              True    4320
temporal_validation               forecasting_v0                   A2_svi_plus_static__svi_score_raw                  A2_svi_plus_static ranking__top_10pct_overlap_rate      0.231481              True    4320
      temporal_test               forecasting_v0                   A2_svi_plus_static__svi_score_raw                  A2_svi_plus_static                      count__mae      3.438431             False    4860
      temporal_test               forecasting_v0                   A2_svi_plus_static__svi_score_raw                  A2_svi_plus_static                     count__rmse      4.942759             False    4860
      temporal_test               forecasting_v0                   A2_svi_plus_static__svi_score_raw                  A2_svi_plus_static    count__mean_poisson_deviance      4.991985             False    4860
      temporal_test               forecasting_v0                   A2_svi_plus_static__svi_score_raw                  A2_svi_plus_static          ranking__spearman_corr      0.230229              True    4860
      temporal_test               forecasting_v0                   A2_svi_plus_static__svi_score_raw                  A2_svi_plus_static            ranking__ndcg_at_100      0.188147              True    4860
      temporal_test               forecasting_v0                   A2_svi_plus_static__svi_score_raw                  A2_svi_plus_static ranking__top_10pct_overlap_rate      0.123457              True    4860
temporal_validation retrospective_explanatory_v0  A2_svi_plus_reporting_retrospective__svi_score_raw A2_svi_plus_reporting_retrospective                      count__mae      2.611218             False    4320
temporal_validation retrospective_explanatory_v0  A2_svi_plus_reporting_retrospective__svi_score_raw A2_svi_plus_reporting_retrospective                     count__rmse      3.996675             False    4320
temporal_validation retrospective_explanatory_v0  A2_svi_plus_reporting_retrospective__svi_score_raw A2_svi_plus_reporting_retrospective    count__mean_poisson_deviance      2.443276             False    4320
temporal_validation retrospective_explanatory_v0  A2_svi_plus_reporting_retrospective__svi_score_raw A2_svi_plus_reporting_retrospective          ranking__spearman_corr      0.711119              True    4320
temporal_validation retrospective_explanatory_v0  A2_svi_plus_reporting_retrospective__svi_score_raw A2_svi_plus_reporting_retrospective            ranking__ndcg_at_100      0.646822              True    4320
temporal_validation retrospective_explanatory_v0  A2_svi_plus_reporting_retrospective__svi_score_raw A2_svi_plus_reporting_retrospective ranking__top_10pct_overlap_rate      0.504630              True    4320
      temporal_test retrospective_explanatory_v0  A2_svi_plus_reporting_retrospective__svi_score_raw A2_svi_plus_reporting_retrospective                      count__mae      2.524465             False    4860
      temporal_test retrospective_explanatory_v0  A2_svi_plus_reporting_retrospective__svi_score_raw A2_svi_plus_reporting_retrospective                     count__rmse      3.664711             False    4860
      temporal_test retrospective_explanatory_v0  A2_svi_plus_reporting_retrospective__svi_score_raw A2_svi_plus_reporting_retrospective    count__mean_poisson_deviance      2.524697             False    4860
      temporal_test retrospective_explanatory_v0  A2_svi_plus_reporting_retrospective__svi_score_raw A2_svi_plus_reporting_retrospective          ranking__spearman_corr      0.671258              True    4860
      temporal_test retrospective_explanatory_v0  A2_svi_plus_reporting_retrospective__svi_score_raw A2_svi_plus_reporting_retrospective            ranking__ndcg_at_100      0.705331              True    4860
      temporal_test retrospective_explanatory_v0  A2_svi_plus_reporting_retrospective__svi_score_raw A2_svi_plus_reporting_retrospective ranking__top_10pct_overlap_rate      0.462963              True    4860
temporal_validation               forecasting_v0               A2_svi_only__svi_rank__diagnostic_svi                         A2_svi_only                      count__mae      3.704128             False    4320
temporal_validation               forecasting_v0               A2_svi_only__svi_rank__diagnostic_svi                         A2_svi_only                     count__rmse      5.516761             False    4320
temporal_validation               forecasting_v0               A2_svi_only__svi_rank__diagnostic_svi                         A2_svi_only    count__mean_poisson_deviance      5.409548             False    4320
temporal_validation               forecasting_v0               A2_svi_only__svi_rank__diagnostic_svi                         A2_svi_only          ranking__spearman_corr      0.167490              True    4320
temporal_validation               forecasting_v0               A2_svi_only__svi_rank__diagnostic_svi                         A2_svi_only            ranking__ndcg_at_100      0.193465              True    4320
temporal_validation               forecasting_v0               A2_svi_only__svi_rank__diagnostic_svi                         A2_svi_only ranking__top_10pct_overlap_rate      0.067130              True    4320
      temporal_test               forecasting_v0               A2_svi_only__svi_rank__diagnostic_svi                         A2_svi_only                      count__mae      3.466399             False    4860
      temporal_test               forecasting_v0               A2_svi_only__svi_rank__diagnostic_svi                         A2_svi_only                     count__rmse      4.926365             False    4860
      temporal_test               forecasting_v0               A2_svi_only__svi_rank__diagnostic_svi                         A2_svi_only    count__mean_poisson_deviance      4.772599             False    4860
      temporal_test               forecasting_v0               A2_svi_only__svi_rank__diagnostic_svi                         A2_svi_only          ranking__spearman_corr      0.169765              True    4860
      temporal_test               forecasting_v0               A2_svi_only__svi_rank__diagnostic_svi                         A2_svi_only            ranking__ndcg_at_100      0.220560              True    4860
      temporal_test               forecasting_v0               A2_svi_only__svi_rank__diagnostic_svi                         A2_svi_only ranking__top_10pct_overlap_rate      0.057613              True    4860
temporal_validation               forecasting_v0      A2_svi_plus_calendar__svi_rank__diagnostic_svi                A2_svi_plus_calendar                      count__mae      3.589516             False    4320
temporal_validation               forecasting_v0      A2_svi_plus_calendar__svi_rank__diagnostic_svi                A2_svi_plus_calendar                     count__rmse      5.349837             False    4320
temporal_validation               forecasting_v0      A2_svi_plus_calendar__svi_rank__diagnostic_svi                A2_svi_plus_calendar    count__mean_poisson_deviance      4.978100             False    4320
temporal_validation               forecasting_v0      A2_svi_plus_calendar__svi_rank__diagnostic_svi                A2_svi_plus_calendar          ranking__spearman_corr      0.292492              True    4320
temporal_validation               forecasting_v0      A2_svi_plus_calendar__svi_rank__diagnostic_svi                A2_svi_plus_calendar            ranking__ndcg_at_100      0.221594              True    4320
temporal_validation               forecasting_v0      A2_svi_plus_calendar__svi_rank__diagnostic_svi                A2_svi_plus_calendar ranking__top_10pct_overlap_rate      0.134259              True    4320
      temporal_test               forecasting_v0      A2_svi_plus_calendar__svi_rank__diagnostic_svi                A2_svi_plus_calendar                      count__mae      3.526431             False    4860
      temporal_test               forecasting_v0      A2_svi_plus_calendar__svi_rank__diagnostic_svi                A2_svi_plus_calendar                     count__rmse      5.028900             False    4860
      temporal_test               forecasting_v0      A2_svi_plus_calendar__svi_rank__diagnostic_svi                A2_svi_plus_calendar    count__mean_poisson_deviance      5.090171             False    4860
      temporal_test               forecasting_v0      A2_svi_plus_calendar__svi_rank__diagnostic_svi                A2_svi_plus_calendar          ranking__spearman_corr      0.144592              True    4860
      temporal_test               forecasting_v0      A2_svi_plus_calendar__svi_rank__diagnostic_svi                A2_svi_plus_calendar            ranking__ndcg_at_100      0.121777              True    4860
      temporal_test               forecasting_v0      A2_svi_plus_calendar__svi_rank__diagnostic_svi                A2_svi_plus_calendar ranking__top_10pct_overlap_rate      0.074074              True    4860
```

## Static-score audit

```text
 source_column                              score_column                                 score_role  zones  zones_with_multiple_values  max_unique_values_within_zone                status examples
svi_percentile    svi_percentile__higher_more_vulnerable     primary_continuous_svi_score_candidate    540                           0                              1 ok_static_within_zone       []
 svi_score_raw     svi_score_raw__higher_more_vulnerable     primary_continuous_svi_score_candidate    540                           0                              1 ok_static_within_zone       []
      svi_rank svi_rank__rank_reversed_for_vulnerability             diagnostic_rank_reversed_score    540                           0                              1 ok_static_within_zone       []
     svi_class         svi_class__higher_more_vulnerable diagnostic_ordinal_class_score_not_primary    540                           0                              1 ok_static_within_zone       []
```

## Coefficients preview

```text
                                         model_name                    feature_set_name           prediction_setting source_svi_column                             score_role                                  feature  coefficient
                        A2_svi_only__svi_percentile                         A2_svi_only               forecasting_v0    svi_percentile primary_continuous_svi_score_candidate                                intercept     1.513892
                        A2_svi_only__svi_percentile                         A2_svi_only               forecasting_v0    svi_percentile primary_continuous_svi_score_candidate                      svi__svi_percentile     0.122136
               A2_svi_plus_calendar__svi_percentile                A2_svi_plus_calendar               forecasting_v0    svi_percentile primary_continuous_svi_score_candidate                                intercept     1.513892
               A2_svi_plus_calendar__svi_percentile                A2_svi_plus_calendar               forecasting_v0    svi_percentile primary_continuous_svi_score_candidate                      svi__svi_percentile     0.122136
               A2_svi_plus_calendar__svi_percentile                A2_svi_plus_calendar               forecasting_v0    svi_percentile primary_continuous_svi_score_candidate                              month_is_02     0.013368
               A2_svi_plus_calendar__svi_percentile                A2_svi_plus_calendar               forecasting_v0    svi_percentile primary_continuous_svi_score_candidate                              month_is_03     0.026929
               A2_svi_plus_calendar__svi_percentile                A2_svi_plus_calendar               forecasting_v0    svi_percentile primary_continuous_svi_score_candidate                              month_is_04     0.047787
               A2_svi_plus_calendar__svi_percentile                A2_svi_plus_calendar               forecasting_v0    svi_percentile primary_continuous_svi_score_candidate                              month_is_05     0.096381
               A2_svi_plus_calendar__svi_percentile                A2_svi_plus_calendar               forecasting_v0    svi_percentile primary_continuous_svi_score_candidate                              month_is_06     0.101780
               A2_svi_plus_calendar__svi_percentile                A2_svi_plus_calendar               forecasting_v0    svi_percentile primary_continuous_svi_score_candidate                              month_is_07     0.093311
               A2_svi_plus_calendar__svi_percentile                A2_svi_plus_calendar               forecasting_v0    svi_percentile primary_continuous_svi_score_candidate                              month_is_08     0.115747
               A2_svi_plus_calendar__svi_percentile                A2_svi_plus_calendar               forecasting_v0    svi_percentile primary_continuous_svi_score_candidate                              month_is_09     0.074907
               A2_svi_plus_calendar__svi_percentile                A2_svi_plus_calendar               forecasting_v0    svi_percentile primary_continuous_svi_score_candidate                              month_is_10     0.062981
               A2_svi_plus_calendar__svi_percentile                A2_svi_plus_calendar               forecasting_v0    svi_percentile primary_continuous_svi_score_candidate                              month_is_11     0.037579
               A2_svi_plus_calendar__svi_percentile                A2_svi_plus_calendar               forecasting_v0    svi_percentile primary_continuous_svi_score_candidate                              month_is_12    -0.013798
                 A2_svi_plus_static__svi_percentile                  A2_svi_plus_static               forecasting_v0    svi_percentile primary_continuous_svi_score_candidate                                intercept     1.513892
                 A2_svi_plus_static__svi_percentile                  A2_svi_plus_static               forecasting_v0    svi_percentile primary_continuous_svi_score_candidate                      svi__svi_percentile     0.067276
                 A2_svi_plus_static__svi_percentile                  A2_svi_plus_static               forecasting_v0    svi_percentile primary_continuous_svi_score_candidate                              month_is_02     0.013368
                 A2_svi_plus_static__svi_percentile                  A2_svi_plus_static               forecasting_v0    svi_percentile primary_continuous_svi_score_candidate                              month_is_03     0.026929
                 A2_svi_plus_static__svi_percentile                  A2_svi_plus_static               forecasting_v0    svi_percentile primary_continuous_svi_score_candidate                              month_is_04     0.047787
                 A2_svi_plus_static__svi_percentile                  A2_svi_plus_static               forecasting_v0    svi_percentile primary_continuous_svi_score_candidate                              month_is_05     0.096381
                 A2_svi_plus_static__svi_percentile                  A2_svi_plus_static               forecasting_v0    svi_percentile primary_continuous_svi_score_candidate                              month_is_06     0.101780
                 A2_svi_plus_static__svi_percentile                  A2_svi_plus_static               forecasting_v0    svi_percentile primary_continuous_svi_score_candidate                              month_is_07     0.093311
                 A2_svi_plus_static__svi_percentile                  A2_svi_plus_static               forecasting_v0    svi_percentile primary_continuous_svi_score_candidate                              month_is_08     0.115747
                 A2_svi_plus_static__svi_percentile                  A2_svi_plus_static               forecasting_v0    svi_percentile primary_continuous_svi_score_candidate                              month_is_09     0.074907
                 A2_svi_plus_static__svi_percentile                  A2_svi_plus_static               forecasting_v0    svi_percentile primary_continuous_svi_score_candidate                              month_is_10     0.062981
                 A2_svi_plus_static__svi_percentile                  A2_svi_plus_static               forecasting_v0    svi_percentile primary_continuous_svi_score_candidate                              month_is_11     0.037579
                 A2_svi_plus_static__svi_percentile                  A2_svi_plus_static               forecasting_v0    svi_percentile primary_continuous_svi_score_candidate                              month_is_12    -0.013798
                 A2_svi_plus_static__svi_percentile                  A2_svi_plus_static               forecasting_v0    svi_percentile primary_continuous_svi_score_candidate              log1p_population_total_2021     0.403102
                 A2_svi_plus_static__svi_percentile                  A2_svi_plus_static               forecasting_v0    svi_percentile primary_continuous_svi_score_candidate                      log1p_land_area_km2    -0.230687
                 A2_svi_plus_static__svi_percentile                  A2_svi_plus_static               forecasting_v0    svi_percentile primary_continuous_svi_score_candidate                 log1p_population_density    -0.273814
A2_svi_plus_reporting_retrospective__svi_percentile A2_svi_plus_reporting_retrospective retrospective_explanatory_v0    svi_percentile primary_continuous_svi_score_candidate                                intercept     1.513892
A2_svi_plus_reporting_retrospective__svi_percentile A2_svi_plus_reporting_retrospective retrospective_explanatory_v0    svi_percentile primary_continuous_svi_score_candidate                      svi__svi_percentile    -0.014800
A2_svi_plus_reporting_retrospective__svi_percentile A2_svi_plus_reporting_retrospective retrospective_explanatory_v0    svi_percentile primary_continuous_svi_score_candidate                              month_is_02     0.033170
A2_svi_plus_reporting_retrospective__svi_percentile A2_svi_plus_reporting_retrospective retrospective_explanatory_v0    svi_percentile primary_continuous_svi_score_candidate                              month_is_03     0.044464
A2_svi_plus_reporting_retrospective__svi_percentile A2_svi_plus_reporting_retrospective retrospective_explanatory_v0    svi_percentile primary_continuous_svi_score_candidate                              month_is_04     0.007907
A2_svi_plus_reporting_retrospective__svi_percentile A2_svi_plus_reporting_retrospective retrospective_explanatory_v0    svi_percentile primary_continuous_svi_score_candidate                              month_is_05     0.040799
A2_svi_plus_reporting_retrospective__svi_percentile A2_svi_plus_reporting_retrospective retrospective_explanatory_v0    svi_percentile primary_continuous_svi_score_candidate                              month_is_06     0.055637
A2_svi_plus_reporting_retrospective__svi_percentile A2_svi_plus_reporting_retrospective retrospective_explanatory_v0    svi_percentile primary_continuous_svi_score_candidate                              month_is_07     0.049626
A2_svi_plus_reporting_retrospective__svi_percentile A2_svi_plus_reporting_retrospective retrospective_explanatory_v0    svi_percentile primary_continuous_svi_score_candidate                              month_is_08     0.068929
A2_svi_plus_reporting_retrospective__svi_percentile A2_svi_plus_reporting_retrospective retrospective_explanatory_v0    svi_percentile primary_continuous_svi_score_candidate                              month_is_09     0.052752
A2_svi_plus_reporting_retrospective__svi_percentile A2_svi_plus_reporting_retrospective retrospective_explanatory_v0    svi_percentile primary_continuous_svi_score_candidate                              month_is_10     0.054786
A2_svi_plus_reporting_retrospective__svi_percentile A2_svi_plus_reporting_retrospective retrospective_explanatory_v0    svi_percentile primary_continuous_svi_score_candidate                              month_is_11     0.043522
A2_svi_plus_reporting_retrospective__svi_percentile A2_svi_plus_reporting_retrospective retrospective_explanatory_v0    svi_percentile primary_continuous_svi_score_candidate                              month_is_12     0.013745
A2_svi_plus_reporting_retrospective__svi_percentile A2_svi_plus_reporting_retrospective retrospective_explanatory_v0    svi_percentile primary_continuous_svi_score_candidate              log1p_population_total_2021     0.221989
A2_svi_plus_reporting_retrospective__svi_percentile A2_svi_plus_reporting_retrospective retrospective_explanatory_v0    svi_percentile primary_continuous_svi_score_candidate                      log1p_land_area_km2    -0.031577
A2_svi_plus_reporting_retrospective__svi_percentile A2_svi_plus_reporting_retrospective retrospective_explanatory_v0    svi_percentile primary_continuous_svi_score_candidate                 log1p_population_density    -0.192195
A2_svi_plus_reporting_retrospective__svi_percentile A2_svi_plus_reporting_retrospective retrospective_explanatory_v0    svi_percentile primary_continuous_svi_score_candidate log1p_total_311_count_non_water_drainage     0.621098
                         A2_svi_only__svi_score_raw                         A2_svi_only               forecasting_v0     svi_score_raw primary_continuous_svi_score_candidate                                intercept     1.513892
                         A2_svi_only__svi_score_raw                         A2_svi_only               forecasting_v0     svi_score_raw primary_continuous_svi_score_candidate                       svi__svi_score_raw     0.108702
                A2_svi_plus_calendar__svi_score_raw                A2_svi_plus_calendar               forecasting_v0     svi_score_raw primary_continuous_svi_score_candidate                                intercept     1.513892
                A2_svi_plus_calendar__svi_score_raw                A2_svi_plus_calendar               forecasting_v0     svi_score_raw primary_continuous_svi_score_candidate                       svi__svi_score_raw     0.108702
                A2_svi_plus_calendar__svi_score_raw                A2_svi_plus_calendar               forecasting_v0     svi_score_raw primary_continuous_svi_score_candidate                              month_is_02     0.013368
                A2_svi_plus_calendar__svi_score_raw                A2_svi_plus_calendar               forecasting_v0     svi_score_raw primary_continuous_svi_score_candidate                              month_is_03     0.026929
                A2_svi_plus_calendar__svi_score_raw                A2_svi_plus_calendar               forecasting_v0     svi_score_raw primary_continuous_svi_score_candidate                              month_is_04     0.047787
                A2_svi_plus_calendar__svi_score_raw                A2_svi_plus_calendar               forecasting_v0     svi_score_raw primary_continuous_svi_score_candidate                              month_is_05     0.096381
                A2_svi_plus_calendar__svi_score_raw                A2_svi_plus_calendar               forecasting_v0     svi_score_raw primary_continuous_svi_score_candidate                              month_is_06     0.101780
                A2_svi_plus_calendar__svi_score_raw                A2_svi_plus_calendar               forecasting_v0     svi_score_raw primary_continuous_svi_score_candidate                              month_is_07     0.093311
                A2_svi_plus_calendar__svi_score_raw                A2_svi_plus_calendar               forecasting_v0     svi_score_raw primary_continuous_svi_score_candidate                              month_is_08     0.115747
                A2_svi_plus_calendar__svi_score_raw                A2_svi_plus_calendar               forecasting_v0     svi_score_raw primary_continuous_svi_score_candidate                              month_is_09     0.074907
                A2_svi_plus_calendar__svi_score_raw                A2_svi_plus_calendar               forecasting_v0     svi_score_raw primary_continuous_svi_score_candidate                              month_is_10     0.062981
                A2_svi_plus_calendar__svi_score_raw                A2_svi_plus_calendar               forecasting_v0     svi_score_raw primary_continuous_svi_score_candidate                              month_is_11     0.037579
                A2_svi_plus_calendar__svi_score_raw                A2_svi_plus_calendar               forecasting_v0     svi_score_raw primary_continuous_svi_score_candidate                              month_is_12    -0.013798
                  A2_svi_plus_static__svi_score_raw                  A2_svi_plus_static               forecasting_v0     svi_score_raw primary_continuous_svi_score_candidate                                intercept     1.513892
                  A2_svi_plus_static__svi_score_raw                  A2_svi_plus_static               forecasting_v0     svi_score_raw primary_continuous_svi_score_candidate                       svi__svi_score_raw     0.045238
                  A2_svi_plus_static__svi_score_raw                  A2_svi_plus_static               forecasting_v0     svi_score_raw primary_continuous_svi_score_candidate                              month_is_02     0.013368
                  A2_svi_plus_static__svi_score_raw                  A2_svi_plus_static               forecasting_v0     svi_score_raw primary_continuous_svi_score_candidate                              month_is_03     0.026929
                  A2_svi_plus_static__svi_score_raw                  A2_svi_plus_static               forecasting_v0     svi_score_raw primary_continuous_svi_score_candidate                              month_is_04     0.047787
                  A2_svi_plus_static__svi_score_raw                  A2_svi_plus_static               forecasting_v0     svi_score_raw primary_continuous_svi_score_candidate                              month_is_05     0.096381
                  A2_svi_plus_static__svi_score_raw                  A2_svi_plus_static               forecasting_v0     svi_score_raw primary_continuous_svi_score_candidate                              month_is_06     0.101780
                  A2_svi_plus_static__svi_score_raw                  A2_svi_plus_static               forecasting_v0     svi_score_raw primary_continuous_svi_score_candidate                              month_is_07     0.093311
                  A2_svi_plus_static__svi_score_raw                  A2_svi_plus_static               forecasting_v0     svi_score_raw primary_continuous_svi_score_candidate                              month_is_08     0.115747
                  A2_svi_plus_static__svi_score_raw                  A2_svi_plus_static               forecasting_v0     svi_score_raw primary_continuous_svi_score_candidate                              month_is_09     0.074907
                  A2_svi_plus_static__svi_score_raw                  A2_svi_plus_static               forecasting_v0     svi_score_raw primary_continuous_svi_score_candidate                              month_is_10     0.062981
                  A2_svi_plus_static__svi_score_raw                  A2_svi_plus_static               forecasting_v0     svi_score_raw primary_continuous_svi_score_candidate                              month_is_11     0.037579
                  A2_svi_plus_static__svi_score_raw                  A2_svi_plus_static               forecasting_v0     svi_score_raw primary_continuous_svi_score_candidate                              month_is_12    -0.013798
                  A2_svi_plus_static__svi_score_raw                  A2_svi_plus_static               forecasting_v0     svi_score_raw primary_continuous_svi_score_candidate              log1p_population_total_2021     0.432975
                  A2_svi_plus_static__svi_score_raw                  A2_svi_plus_static               forecasting_v0     svi_score_raw primary_continuous_svi_score_candidate                      log1p_land_area_km2    -0.253286
                  A2_svi_plus_static__svi_score_raw                  A2_svi_plus_static               forecasting_v0     svi_score_raw primary_continuous_svi_score_candidate                 log1p_population_density    -0.309085
 A2_svi_plus_reporting_retrospective__svi_score_raw A2_svi_plus_reporting_retrospective retrospective_explanatory_v0     svi_score_raw primary_continuous_svi_score_candidate                                intercept     1.513892
```

## Output artifacts

| Artifact | Path |
|---|---|
| `metrics` | `/home/tim/Documents/ville_ai/indices_BACKUP_before_clean_push/urban_graph_benchmark/outputs/mtl_311_water_v0/baselines/A2_calibrated_svi/metrics.csv` |
| `model_metadata` | `/home/tim/Documents/ville_ai/indices_BACKUP_before_clean_push/urban_graph_benchmark/outputs/mtl_311_water_v0/baselines/A2_calibrated_svi/model_metadata.json` |
| `baseline_report` | `/home/tim/Documents/ville_ai/indices_BACKUP_before_clean_push/urban_graph_benchmark/outputs/mtl_311_water_v0/baselines/A2_calibrated_svi/baseline_report.md` |
| `coefficients` | `/home/tim/Documents/ville_ai/indices_BACKUP_before_clean_push/urban_graph_benchmark/outputs/mtl_311_water_v0/baselines/A2_calibrated_svi/coefficients.csv` |
| `svi_score_audit` | `/home/tim/Documents/ville_ai/indices_BACKUP_before_clean_push/urban_graph_benchmark/outputs/mtl_311_water_v0/baselines/A2_calibrated_svi/svi_score_audit.csv` |
| `svi_static_score_audit` | `/home/tim/Documents/ville_ai/indices_BACKUP_before_clean_push/urban_graph_benchmark/outputs/mtl_311_water_v0/baselines/A2_calibrated_svi/svi_static_score_audit.csv` |
| `feature_set_audit` | `/home/tim/Documents/ville_ai/indices_BACKUP_before_clean_push/urban_graph_benchmark/outputs/mtl_311_water_v0/baselines/A2_calibrated_svi/feature_set_audit.csv` |
| `predictions_validation` | `/home/tim/Documents/ville_ai/indices_BACKUP_before_clean_push/urban_graph_benchmark/outputs/mtl_311_water_v0/baselines/A2_calibrated_svi/predictions_validation.parquet` |
| `predictions_test` | `/home/tim/Documents/ville_ai/indices_BACKUP_before_clean_push/urban_graph_benchmark/outputs/mtl_311_water_v0/baselines/A2_calibrated_svi/predictions_test.parquet` |

## Leakage and interpretation notes

- All calibration coefficients, imputations, and standardization parameters are fitted on training rows only.
- `A2_svi_plus_reporting_retrospective` uses same-month `total_311_count_non_water_drainage`; it is retrospective/explanatory only.
- The strict forecasting A2 feature sets do not use same-month reporting controls or target-derived share/count columns.
- Primary A2 interpretation should focus on `svi_percentile` and `svi_score_raw` models.
- `svi_rank` and `svi_class`, when included, are diagnostic robustness checks.
- No SoVI columns are used in Track A.
