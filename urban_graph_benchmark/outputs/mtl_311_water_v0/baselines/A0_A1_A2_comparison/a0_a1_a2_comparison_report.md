# A0/A1/A2 Baseline Comparison — Montréal 311 Water/Drainage v0

Generated at: `2026-06-09T14:47:14.541607+00:00`

Config path: `/home/tim/Documents/ville_ai/indices_BACKUP_before_clean_push/urban_graph_benchmark/configs/mtl_311_water_v0.1.yaml`

## Purpose

This report compares A0 naive temporal baselines, A1 raw SVI direct-ranking baselines, and A2 calibrated SVI predictors. It introduces no new modeling methodology; it only summarizes metrics already produced by the baseline modules.

## Headline conclusion

A2 shows that calibration and static controls improve SVI relative to raw A1, but strict-forecasting A2 still does not beat the strongest A0 historical tract baseline. The retrospective A2 reporting-control variant is strong, but it uses same-month non-water 311 reporting and must not be treated as a forecasting model.

## Temporal-test core comparison

```text
                        label                                          model_name                                 role      mae     rmse  poisson_deviance  spearman_corr  ndcg_at_100  top_10pct_overlap_rate
                   A0 history                               A0_3_tract_train_mean                 strong_naive_history 2.489209 3.462372          2.143367       0.694235     0.748363                0.483539
        A1 raw SVI percentile               A1_svi_direct_ranking__svi_percentile              raw_primary_svi_ranking      NaN      NaN               NaN       0.160639     0.220560                0.052411
             A1 raw SVI score                A1_svi_direct_ranking__svi_score_raw              raw_primary_svi_ranking      NaN      NaN               NaN       0.160630     0.220560                0.052411
       A2 SVI-only percentile                         A2_svi_only__svi_percentile   calibrated_forecasting_primary_svi 3.475547 4.938114          4.801117       0.159008     0.220560                0.057613
   A2 SVI + static percentile                  A2_svi_plus_static__svi_percentile   calibrated_forecasting_primary_svi 3.434751 4.936054          4.952062       0.236370     0.193140                0.121399
A2 SVI + reporting percentile A2_svi_plus_reporting_retrospective__svi_percentile calibrated_retrospective_primary_svi 2.522434 3.663199          2.522685       0.671736     0.702508                0.462963
```

Strict forecasting count MAE: `A0_3_tract_train_mean` = `2.4892`, `A2_svi_plus_static__svi_percentile` = `3.4348`.

Retrospective count MAE: `A2_svi_plus_reporting_retrospective__svi_percentile` = `2.5224`. This is strong but uses same-month reporting exposure.

Ranking Spearman: raw A1 SVI percentile = `0.1606`, calibrated A2 SVI+static percentile = `0.2364`, A0 tract history = `0.6942`.

Retrospective A2 SVI+reporting Spearman = `0.6717`.

## Best rows by temporal-test metric and stage

```text
source_stage               scope                 metric                                                     best_model                              model_role  metric_value  higher_is_better  n_rows
          A0    count_prediction                    mae                                          A0_3_tract_train_mean                    strong_naive_history      2.489209             False    4860
          A0    count_prediction                   rmse                                          A0_3_tract_train_mean                    strong_naive_history      3.462372             False    4860
          A0 tract_month_ranking          spearman_corr                                          A0_3_tract_train_mean                    strong_naive_history      0.694235              True    4860
          A0 tract_month_ranking            ndcg_at_100                                          A0_3_tract_train_mean                    strong_naive_history      0.748363              True    4860
          A0 tract_month_ranking top_10pct_overlap_rate                                          A0_3_tract_train_mean                    strong_naive_history      0.483539              True    4860
          A1 tract_month_ranking          spearman_corr     A1_svi_direct_ranking__svi_class__ordinal_class_diagnostic                raw_diagnostic_svi_class      0.185642              True    4770
          A1 tract_month_ranking            ndcg_at_100     A1_svi_direct_ranking__svi_class__ordinal_class_diagnostic                raw_diagnostic_svi_class      0.222316              True    4770
          A1 tract_month_ranking top_10pct_overlap_rate     A1_svi_direct_ranking__svi_class__ordinal_class_diagnostic                raw_diagnostic_svi_class      0.150943              True    4770
          A2    count_prediction                    mae A2_svi_plus_reporting_retrospective__svi_class__diagnostic_svi calibrated_retrospective_diagnostic_svi      2.522280             False    4860
          A2    count_prediction                   rmse            A2_svi_plus_reporting_retrospective__svi_percentile    calibrated_retrospective_primary_svi      3.663199             False    4860
          A2 tract_month_ranking          spearman_corr A2_svi_plus_reporting_retrospective__svi_class__diagnostic_svi calibrated_retrospective_diagnostic_svi      0.671807              True    4860
          A2 tract_month_ranking            ndcg_at_100  A2_svi_plus_reporting_retrospective__svi_rank__diagnostic_svi calibrated_retrospective_diagnostic_svi      0.705511              True    4860
          A2 tract_month_ranking top_10pct_overlap_rate A2_svi_plus_reporting_retrospective__svi_class__diagnostic_svi calibrated_retrospective_diagnostic_svi      0.465021              True    4860
```

## Overall best temporal-test rows

| Metric | Best model | Value | Scope | Higher is better |
|---|---|---:|---|:---:|
| Count MAE | `A0_3_tract_train_mean` | 2.4892 | `count_prediction` | `False` |
| Tract-month Spearman | `A0_3_tract_train_mean` | 0.6942 | `tract_month_ranking` | `True` |
| Tract-month top-10% overlap | `A0_3_tract_train_mean` | 0.4835 | `tract_month_ranking` | `True` |

## Headline metric rows

```text
source_stage                                                     model_name                              model_role           prediction_setting          target_name         target_type                              display_metric  metric_value  higher_is_better  n_rows
          A0                                          A0_3_tract_train_mean                    strong_naive_history               forecasting_v0 water_drainage_count               count                       count_prediction__mae      2.489209             False    4860
          A0                            A0_4_tract_month_of_year_train_mean                  seasonal_tract_history               forecasting_v0 water_drainage_count               count                       count_prediction__mae      2.739026             False    4860
          A0                                A0_5_previous_month_persistence                     persistence_history one_step_observed_history_v0 water_drainage_count               count                       count_prediction__mae      3.163786             False    4860
          A0            A0_8_non_water_311_reporting_exposure_retrospective        retrospective_reporting_exposure retrospective_explanatory_v0 water_drainage_count               count                       count_prediction__mae      3.011912             False    4860
          A2                                    A2_svi_only__svi_percentile      calibrated_forecasting_primary_svi               forecasting_v0 water_drainage_count               count                       count_prediction__mae      3.475547             False    4860
          A2                           A2_svi_plus_calendar__svi_percentile      calibrated_forecasting_primary_svi               forecasting_v0 water_drainage_count               count                       count_prediction__mae      3.535689             False    4860
          A2 A2_svi_plus_reporting_retrospective__svi_class__diagnostic_svi calibrated_retrospective_diagnostic_svi retrospective_explanatory_v0 water_drainage_count               count                       count_prediction__mae      2.522280             False    4860
          A2            A2_svi_plus_reporting_retrospective__svi_percentile    calibrated_retrospective_primary_svi retrospective_explanatory_v0 water_drainage_count               count                       count_prediction__mae      2.522434             False    4860
          A2                  A2_svi_plus_static__svi_class__diagnostic_svi   calibrated_forecasting_diagnostic_svi               forecasting_v0 water_drainage_count               count                       count_prediction__mae      3.430960             False    4860
          A2                             A2_svi_plus_static__svi_percentile      calibrated_forecasting_primary_svi               forecasting_v0 water_drainage_count               count                       count_prediction__mae      3.434751             False    4860
          A0                                          A0_3_tract_train_mean                    strong_naive_history               forecasting_v0 water_drainage_count               count     count_prediction__mean_poisson_deviance      2.143367             False    4860
          A0                            A0_4_tract_month_of_year_train_mean                  seasonal_tract_history               forecasting_v0 water_drainage_count               count     count_prediction__mean_poisson_deviance      3.333048             False    4860
          A0                                A0_5_previous_month_persistence                     persistence_history one_step_observed_history_v0 water_drainage_count               count     count_prediction__mean_poisson_deviance     11.224499             False    4860
          A0            A0_8_non_water_311_reporting_exposure_retrospective        retrospective_reporting_exposure retrospective_explanatory_v0 water_drainage_count               count     count_prediction__mean_poisson_deviance      2.843425             False    4860
          A2                                    A2_svi_only__svi_percentile      calibrated_forecasting_primary_svi               forecasting_v0 water_drainage_count               count     count_prediction__mean_poisson_deviance      4.801117             False    4860
          A2                           A2_svi_plus_calendar__svi_percentile      calibrated_forecasting_primary_svi               forecasting_v0 water_drainage_count               count     count_prediction__mean_poisson_deviance      5.119893             False    4860
          A2 A2_svi_plus_reporting_retrospective__svi_class__diagnostic_svi calibrated_retrospective_diagnostic_svi retrospective_explanatory_v0 water_drainage_count               count     count_prediction__mean_poisson_deviance      2.522860             False    4860
          A2            A2_svi_plus_reporting_retrospective__svi_percentile    calibrated_retrospective_primary_svi retrospective_explanatory_v0 water_drainage_count               count     count_prediction__mean_poisson_deviance      2.522685             False    4860
          A2                  A2_svi_plus_static__svi_class__diagnostic_svi   calibrated_forecasting_diagnostic_svi               forecasting_v0 water_drainage_count               count     count_prediction__mean_poisson_deviance      4.926046             False    4860
          A2                             A2_svi_plus_static__svi_percentile      calibrated_forecasting_primary_svi               forecasting_v0 water_drainage_count               count     count_prediction__mean_poisson_deviance      4.952062             False    4860
          A0                                          A0_3_tract_train_mean                    strong_naive_history               forecasting_v0 water_drainage_count               count                      count_prediction__rmse      3.462372             False    4860
          A0                            A0_4_tract_month_of_year_train_mean                  seasonal_tract_history               forecasting_v0 water_drainage_count               count                      count_prediction__rmse      3.914971             False    4860
          A0                                A0_5_previous_month_persistence                     persistence_history one_step_observed_history_v0 water_drainage_count               count                      count_prediction__rmse      4.542031             False    4860
          A0            A0_8_non_water_311_reporting_exposure_retrospective        retrospective_reporting_exposure retrospective_explanatory_v0 water_drainage_count               count                      count_prediction__rmse      4.370518             False    4860
          A2                                    A2_svi_only__svi_percentile      calibrated_forecasting_primary_svi               forecasting_v0 water_drainage_count               count                      count_prediction__rmse      4.938114             False    4860
          A2                           A2_svi_plus_calendar__svi_percentile      calibrated_forecasting_primary_svi               forecasting_v0 water_drainage_count               count                      count_prediction__rmse      5.040499             False    4860
          A2 A2_svi_plus_reporting_retrospective__svi_class__diagnostic_svi calibrated_retrospective_diagnostic_svi retrospective_explanatory_v0 water_drainage_count               count                      count_prediction__rmse      3.663603             False    4860
          A2            A2_svi_plus_reporting_retrospective__svi_percentile    calibrated_retrospective_primary_svi retrospective_explanatory_v0 water_drainage_count               count                      count_prediction__rmse      3.663199             False    4860
          A2                  A2_svi_plus_static__svi_class__diagnostic_svi   calibrated_forecasting_diagnostic_svi               forecasting_v0 water_drainage_count               count                      count_prediction__rmse      4.929193             False    4860
          A2                             A2_svi_plus_static__svi_percentile      calibrated_forecasting_primary_svi               forecasting_v0 water_drainage_count               count                      count_prediction__rmse      4.936054             False    4860
          A0                                          A0_3_tract_train_mean                    strong_naive_history               forecasting_v0 water_drainage_count               count            tract_month_ranking__ndcg_at_100      0.748363              True    4860
          A0                            A0_4_tract_month_of_year_train_mean                  seasonal_tract_history               forecasting_v0 water_drainage_count               count            tract_month_ranking__ndcg_at_100      0.636779              True    4860
          A0                                A0_5_previous_month_persistence                     persistence_history one_step_observed_history_v0 water_drainage_count               count            tract_month_ranking__ndcg_at_100      0.598551              True    4860
          A0            A0_8_non_water_311_reporting_exposure_retrospective        retrospective_reporting_exposure retrospective_explanatory_v0 water_drainage_count               count            tract_month_ranking__ndcg_at_100      0.662923              True    4860
          A1     A1_svi_direct_ranking__svi_class__ordinal_class_diagnostic                raw_diagnostic_svi_class static_svi_direct_ranking_v0 water_drainage_count tract_month_ranking            tract_month_ranking__ndcg_at_100      0.222316              True    4770
          A1                          A1_svi_direct_ranking__svi_percentile                 raw_primary_svi_ranking static_svi_direct_ranking_v0 water_drainage_count tract_month_ranking            tract_month_ranking__ndcg_at_100      0.220560              True    4770
          A1                           A1_svi_direct_ranking__svi_score_raw                 raw_primary_svi_ranking static_svi_direct_ranking_v0 water_drainage_count tract_month_ranking            tract_month_ranking__ndcg_at_100      0.220560              True    4770
          A2                                    A2_svi_only__svi_percentile      calibrated_forecasting_primary_svi               forecasting_v0 water_drainage_count               count            tract_month_ranking__ndcg_at_100      0.220560              True    4860
          A2                           A2_svi_plus_calendar__svi_percentile      calibrated_forecasting_primary_svi               forecasting_v0 water_drainage_count               count            tract_month_ranking__ndcg_at_100      0.121777              True    4860
          A2 A2_svi_plus_reporting_retrospective__svi_class__diagnostic_svi calibrated_retrospective_diagnostic_svi retrospective_explanatory_v0 water_drainage_count               count            tract_month_ranking__ndcg_at_100      0.702629              True    4860
          A2            A2_svi_plus_reporting_retrospective__svi_percentile    calibrated_retrospective_primary_svi retrospective_explanatory_v0 water_drainage_count               count            tract_month_ranking__ndcg_at_100      0.702508              True    4860
          A2                  A2_svi_plus_static__svi_class__diagnostic_svi   calibrated_forecasting_diagnostic_svi               forecasting_v0 water_drainage_count               count            tract_month_ranking__ndcg_at_100      0.221966              True    4860
          A2                             A2_svi_plus_static__svi_percentile      calibrated_forecasting_primary_svi               forecasting_v0 water_drainage_count               count            tract_month_ranking__ndcg_at_100      0.193140              True    4860
          A0                                          A0_3_tract_train_mean                    strong_naive_history               forecasting_v0 water_drainage_count               count          tract_month_ranking__spearman_corr      0.694235              True    4860
          A0                            A0_4_tract_month_of_year_train_mean                  seasonal_tract_history               forecasting_v0 water_drainage_count               count          tract_month_ranking__spearman_corr      0.616683              True    4860
          A0                                A0_5_previous_month_persistence                     persistence_history one_step_observed_history_v0 water_drainage_count               count          tract_month_ranking__spearman_corr      0.593867              True    4860
          A0            A0_8_non_water_311_reporting_exposure_retrospective        retrospective_reporting_exposure retrospective_explanatory_v0 water_drainage_count               count          tract_month_ranking__spearman_corr      0.631671              True    4860
          A1     A1_svi_direct_ranking__svi_class__ordinal_class_diagnostic                raw_diagnostic_svi_class static_svi_direct_ranking_v0 water_drainage_count tract_month_ranking          tract_month_ranking__spearman_corr      0.185642              True    4770
          A1                          A1_svi_direct_ranking__svi_percentile                 raw_primary_svi_ranking static_svi_direct_ranking_v0 water_drainage_count tract_month_ranking          tract_month_ranking__spearman_corr      0.160639              True    4770
          A1                           A1_svi_direct_ranking__svi_score_raw                 raw_primary_svi_ranking static_svi_direct_ranking_v0 water_drainage_count tract_month_ranking          tract_month_ranking__spearman_corr      0.160630              True    4770
          A2                                    A2_svi_only__svi_percentile      calibrated_forecasting_primary_svi               forecasting_v0 water_drainage_count               count          tract_month_ranking__spearman_corr      0.159008              True    4860
          A2                           A2_svi_plus_calendar__svi_percentile      calibrated_forecasting_primary_svi               forecasting_v0 water_drainage_count               count          tract_month_ranking__spearman_corr      0.127134              True    4860
          A2 A2_svi_plus_reporting_retrospective__svi_class__diagnostic_svi calibrated_retrospective_diagnostic_svi retrospective_explanatory_v0 water_drainage_count               count          tract_month_ranking__spearman_corr      0.671807              True    4860
          A2            A2_svi_plus_reporting_retrospective__svi_percentile    calibrated_retrospective_primary_svi retrospective_explanatory_v0 water_drainage_count               count          tract_month_ranking__spearman_corr      0.671736              True    4860
          A2                  A2_svi_plus_static__svi_class__diagnostic_svi   calibrated_forecasting_diagnostic_svi               forecasting_v0 water_drainage_count               count          tract_month_ranking__spearman_corr      0.240694              True    4860
          A2                             A2_svi_plus_static__svi_percentile      calibrated_forecasting_primary_svi               forecasting_v0 water_drainage_count               count          tract_month_ranking__spearman_corr      0.236370              True    4860
          A0                                          A0_3_tract_train_mean                    strong_naive_history               forecasting_v0 water_drainage_count               count tract_month_ranking__top_10pct_overlap_rate      0.483539              True    4860
          A0                            A0_4_tract_month_of_year_train_mean                  seasonal_tract_history               forecasting_v0 water_drainage_count               count tract_month_ranking__top_10pct_overlap_rate      0.448560              True    4860
          A0                                A0_5_previous_month_persistence                     persistence_history one_step_observed_history_v0 water_drainage_count               count tract_month_ranking__top_10pct_overlap_rate      0.390947              True    4860
          A0            A0_8_non_water_311_reporting_exposure_retrospective        retrospective_reporting_exposure retrospective_explanatory_v0 water_drainage_count               count tract_month_ranking__top_10pct_overlap_rate      0.403292              True    4860
          A1     A1_svi_direct_ranking__svi_class__ordinal_class_diagnostic                raw_diagnostic_svi_class static_svi_direct_ranking_v0 water_drainage_count tract_month_ranking tract_month_ranking__top_10pct_overlap_rate      0.150943              True    4770
          A1                          A1_svi_direct_ranking__svi_percentile                 raw_primary_svi_ranking static_svi_direct_ranking_v0 water_drainage_count tract_month_ranking tract_month_ranking__top_10pct_overlap_rate      0.052411              True    4770
          A1                           A1_svi_direct_ranking__svi_score_raw                 raw_primary_svi_ranking static_svi_direct_ranking_v0 water_drainage_count tract_month_ranking tract_month_ranking__top_10pct_overlap_rate      0.052411              True    4770
          A2                                    A2_svi_only__svi_percentile      calibrated_forecasting_primary_svi               forecasting_v0 water_drainage_count               count tract_month_ranking__top_10pct_overlap_rate      0.057613              True    4860
          A2                           A2_svi_plus_calendar__svi_percentile      calibrated_forecasting_primary_svi               forecasting_v0 water_drainage_count               count tract_month_ranking__top_10pct_overlap_rate      0.069959              True    4860
          A2 A2_svi_plus_reporting_retrospective__svi_class__diagnostic_svi calibrated_retrospective_diagnostic_svi retrospective_explanatory_v0 water_drainage_count               count tract_month_ranking__top_10pct_overlap_rate      0.465021              True    4860
          A2            A2_svi_plus_reporting_retrospective__svi_percentile    calibrated_retrospective_primary_svi retrospective_explanatory_v0 water_drainage_count               count tract_month_ranking__top_10pct_overlap_rate      0.462963              True    4860
          A2                  A2_svi_plus_static__svi_class__diagnostic_svi   calibrated_forecasting_diagnostic_svi               forecasting_v0 water_drainage_count               count tract_month_ranking__top_10pct_overlap_rate      0.133745              True    4860
          A2                             A2_svi_plus_static__svi_percentile      calibrated_forecasting_primary_svi               forecasting_v0 water_drainage_count               count tract_month_ranking__top_10pct_overlap_rate      0.121399              True    4860
```

## Wide tract-month ranking table

```text
         split_name      comparison_metric  A0_3_tract_train_mean  A0_4_tract_month_of_year_train_mean  A0_5_previous_month_persistence  A0_8_non_water_311_reporting_exposure_retrospective  A1_svi_direct_ranking__svi_percentile  A1_svi_direct_ranking__svi_score_raw  A1_svi_direct_ranking__svi_class__ordinal_class_diagnostic  A2_svi_only__svi_percentile  A2_svi_plus_calendar__svi_percentile  A2_svi_plus_static__svi_percentile  A2_svi_plus_reporting_retrospective__svi_percentile  A2_svi_plus_static__svi_class__diagnostic_svi  A2_svi_plus_reporting_retrospective__svi_class__diagnostic_svi
      temporal_test            ndcg_at_100               0.748363                             0.636779                         0.598551                                             0.662923                               0.220560                              0.220560                                                    0.222316                     0.220560                              0.121777                            0.193140                                             0.702508                                       0.221966                                                        0.702629
      temporal_test          spearman_corr               0.694235                             0.616683                         0.593867                                             0.631671                               0.160639                              0.160630                                                    0.185642                     0.159008                              0.127134                            0.236370                                             0.671736                                       0.240694                                                        0.671807
      temporal_test top_10pct_overlap_rate               0.483539                             0.448560                         0.390947                                             0.403292                               0.052411                              0.052411                                                    0.150943                     0.057613                              0.069959                            0.121399                                             0.462963                                       0.133745                                                        0.465021
temporal_validation            ndcg_at_100               0.691490                             0.715413                         0.649880                                             0.637579                               0.193465                              0.193465                                                    0.172898                     0.193465                              0.221594                            0.293624                                             0.641888                                       0.313128                                                        0.638654
temporal_validation          spearman_corr               0.687738                             0.676538                         0.637232                                             0.680659                               0.158606                              0.158599                                                    0.174728                     0.156450                              0.283787                            0.369378                                             0.711523                                       0.369574                                                        0.711635
temporal_validation top_10pct_overlap_rate               0.497685                             0.513889                         0.456019                                             0.465278                               0.066038                              0.066038                                                    0.110849                     0.067130                              0.134259                            0.240741                                             0.509259                                       0.252315                                                        0.504630
```

## Interpretation guardrails

- A0 and A2 strict-forecasting rows are fair forecasting comparisons.
- A1 is not a count predictor; it is a raw static SVI ranking baseline.
- A2 reporting-control models are retrospective/explanatory because they use same-month `total_311_count_non_water_drainage`.
- Primary SVI interpretation should focus on `svi_percentile` and `svi_score_raw`; rank/class SVI variants are diagnostics.
- The current result supports building A3 feature-parity tabular baselines before graph models.

## Output artifacts

| Artifact | Path |
|---|---|
| `metrics_long` | `/home/tim/Documents/ville_ai/indices_BACKUP_before_clean_push/urban_graph_benchmark/outputs/mtl_311_water_v0/baselines/A0_A1_A2_comparison/a0_a1_a2_metrics_long.csv` |
| `test_headline_table` | `/home/tim/Documents/ville_ai/indices_BACKUP_before_clean_push/urban_graph_benchmark/outputs/mtl_311_water_v0/baselines/A0_A1_A2_comparison/a0_a1_a2_test_headline_table.csv` |
| `core_model_table` | `/home/tim/Documents/ville_ai/indices_BACKUP_before_clean_push/urban_graph_benchmark/outputs/mtl_311_water_v0/baselines/A0_A1_A2_comparison/a0_a1_a2_core_model_table.csv` |
| `tract_month_ranking_wide` | `/home/tim/Documents/ville_ai/indices_BACKUP_before_clean_push/urban_graph_benchmark/outputs/mtl_311_water_v0/baselines/A0_A1_A2_comparison/a0_a1_a2_tract_month_ranking_wide.csv` |
| `best_by_stage` | `/home/tim/Documents/ville_ai/indices_BACKUP_before_clean_push/urban_graph_benchmark/outputs/mtl_311_water_v0/baselines/A0_A1_A2_comparison/a0_a1_a2_best_by_stage.csv` |
| `comparison_report` | `/home/tim/Documents/ville_ai/indices_BACKUP_before_clean_push/urban_graph_benchmark/outputs/mtl_311_water_v0/baselines/A0_A1_A2_comparison/a0_a1_a2_comparison_report.md` |
| `comparison_metadata` | `/home/tim/Documents/ville_ai/indices_BACKUP_before_clean_push/urban_graph_benchmark/outputs/mtl_311_water_v0/baselines/A0_A1_A2_comparison/comparison_metadata.json` |
