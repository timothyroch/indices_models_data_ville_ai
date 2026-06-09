# A0/A1 Baseline Comparison — Montréal 311 Water/Drainage v0

Generated at: `2026-06-09T14:21:47.275971+00:00`

Config path: `/home/tim/Documents/ville_ai/indices_BACKUP_before_clean_push/urban_graph_benchmark/configs/mtl_311_water_v0.1.yaml`

## Purpose

This report compares the first two baseline layers: A0 naive temporal/exposure baselines and A1 static SVI direct-ranking baselines. It introduces no new modeling methodology; it only summarizes metrics already produced by A0 and A1.

## Headline conclusion

The strongest A0 history baseline is much stronger than raw SVI for tract-month ranking of future water/drainage 311 burden. SVI has a positive but weak standalone ranking signal. This supports using A1 as a vulnerability-prior diagnostic, not as the main predictive benchmark.

## Test split: core ranking comparison

```text
                        label                                                 model_name  spearman_corr  ndcg_at_100  top_10pct_overlap_rate
A0 strongest history baseline                                      A0_3_tract_train_mean       0.694235     0.748363                0.483539
    A1 primary SVI percentile                      A1_svi_direct_ranking__svi_percentile       0.160639     0.220560                0.052411
     A1 primary SVI raw score                       A1_svi_direct_ranking__svi_score_raw       0.160630     0.220560                0.052411
      A1 diagnostic SVI class A1_svi_direct_ranking__svi_class__ordinal_class_diagnostic       0.185642     0.222316                0.150943
```

On temporal test, `A0_3_tract_train_mean` has Spearman `0.6942`; `A1_svi_percentile` has Spearman `0.1606`. Ratio: `4.3217`.

For top-10% overlap, `A0_3_tract_train_mean` has `0.4835`; `A1_svi_percentile` has `0.0524`. Ratio: `9.2259`.

## Best rows by temporal-test metric

| Metric | Best model | Value | Scope | Higher is better |
|---|---|---:|---|:---:|
| Tract-month Spearman | `A0_3_tract_train_mean` | 0.6942 | `tract_month_ranking` | `True` |
| Tract-month NDCG@100 | `A0_3_tract_train_mean` | 0.7484 | `tract_month_ranking` | `True` |
| Tract-month top-10% overlap | `A0_3_tract_train_mean` | 0.4835 | `tract_month_ranking` | `True` |
| Count MAE | `A0_3_tract_train_mean` | 2.4892 | `count_prediction` | `False` |

## Headline metric rows

```text
source_stage                                                 model_name                       model_role          target_name         target_type                              display_metric  metric_value  higher_is_better  n_rows
          A0                                      A0_3_tract_train_mean             strong_naive_history water_drainage_count               count                       count_prediction__mae      2.489209             False    4860
          A0                        A0_4_tract_month_of_year_train_mean           seasonal_tract_history water_drainage_count               count                       count_prediction__mae      2.739026             False    4860
          A0                            A0_5_previous_month_persistence              persistence_history water_drainage_count               count                       count_prediction__mae      3.163786             False    4860
          A0        A0_8_non_water_311_reporting_exposure_retrospective retrospective_reporting_exposure water_drainage_count               count                       count_prediction__mae      3.011912             False    4860
          A0                                      A0_3_tract_train_mean             strong_naive_history water_drainage_count               count     count_prediction__mean_poisson_deviance      2.143367             False    4860
          A0                        A0_4_tract_month_of_year_train_mean           seasonal_tract_history water_drainage_count               count     count_prediction__mean_poisson_deviance      3.333048             False    4860
          A0                            A0_5_previous_month_persistence              persistence_history water_drainage_count               count     count_prediction__mean_poisson_deviance     11.224499             False    4860
          A0        A0_8_non_water_311_reporting_exposure_retrospective retrospective_reporting_exposure water_drainage_count               count     count_prediction__mean_poisson_deviance      2.843425             False    4860
          A0                                      A0_3_tract_train_mean             strong_naive_history water_drainage_count               count                      count_prediction__rmse      3.462372             False    4860
          A0                        A0_4_tract_month_of_year_train_mean           seasonal_tract_history water_drainage_count               count                      count_prediction__rmse      3.914971             False    4860
          A0                            A0_5_previous_month_persistence              persistence_history water_drainage_count               count                      count_prediction__rmse      4.542031             False    4860
          A0        A0_8_non_water_311_reporting_exposure_retrospective retrospective_reporting_exposure water_drainage_count               count                      count_prediction__rmse      4.370518             False    4860
          A0                                      A0_3_tract_train_mean             strong_naive_history water_drainage_count               count            tract_month_ranking__ndcg_at_100      0.748363              True    4860
          A0                        A0_4_tract_month_of_year_train_mean           seasonal_tract_history water_drainage_count               count            tract_month_ranking__ndcg_at_100      0.636779              True    4860
          A0                            A0_5_previous_month_persistence              persistence_history water_drainage_count               count            tract_month_ranking__ndcg_at_100      0.598551              True    4860
          A0        A0_8_non_water_311_reporting_exposure_retrospective retrospective_reporting_exposure water_drainage_count               count            tract_month_ranking__ndcg_at_100      0.662923              True    4860
          A1 A1_svi_direct_ranking__svi_class__ordinal_class_diagnostic             diagnostic_svi_class water_drainage_count tract_month_ranking            tract_month_ranking__ndcg_at_100      0.222316              True    4770
          A1                      A1_svi_direct_ranking__svi_percentile           primary_continuous_svi water_drainage_count tract_month_ranking            tract_month_ranking__ndcg_at_100      0.220560              True    4770
          A1                       A1_svi_direct_ranking__svi_score_raw           primary_continuous_svi water_drainage_count tract_month_ranking            tract_month_ranking__ndcg_at_100      0.220560              True    4770
          A0                                      A0_3_tract_train_mean             strong_naive_history water_drainage_count               count          tract_month_ranking__spearman_corr      0.694235              True    4860
          A0                        A0_4_tract_month_of_year_train_mean           seasonal_tract_history water_drainage_count               count          tract_month_ranking__spearman_corr      0.616683              True    4860
          A0                            A0_5_previous_month_persistence              persistence_history water_drainage_count               count          tract_month_ranking__spearman_corr      0.593867              True    4860
          A0        A0_8_non_water_311_reporting_exposure_retrospective retrospective_reporting_exposure water_drainage_count               count          tract_month_ranking__spearman_corr      0.631671              True    4860
          A1 A1_svi_direct_ranking__svi_class__ordinal_class_diagnostic             diagnostic_svi_class water_drainage_count tract_month_ranking          tract_month_ranking__spearman_corr      0.185642              True    4770
          A1                      A1_svi_direct_ranking__svi_percentile           primary_continuous_svi water_drainage_count tract_month_ranking          tract_month_ranking__spearman_corr      0.160639              True    4770
          A1                       A1_svi_direct_ranking__svi_score_raw           primary_continuous_svi water_drainage_count tract_month_ranking          tract_month_ranking__spearman_corr      0.160630              True    4770
          A0                                      A0_3_tract_train_mean             strong_naive_history water_drainage_count               count tract_month_ranking__top_10pct_overlap_rate      0.483539              True    4860
          A0                        A0_4_tract_month_of_year_train_mean           seasonal_tract_history water_drainage_count               count tract_month_ranking__top_10pct_overlap_rate      0.448560              True    4860
          A0                            A0_5_previous_month_persistence              persistence_history water_drainage_count               count tract_month_ranking__top_10pct_overlap_rate      0.390947              True    4860
          A0        A0_8_non_water_311_reporting_exposure_retrospective retrospective_reporting_exposure water_drainage_count               count tract_month_ranking__top_10pct_overlap_rate      0.403292              True    4860
          A1 A1_svi_direct_ranking__svi_class__ordinal_class_diagnostic             diagnostic_svi_class water_drainage_count tract_month_ranking tract_month_ranking__top_10pct_overlap_rate      0.150943              True    4770
          A1                      A1_svi_direct_ranking__svi_percentile           primary_continuous_svi water_drainage_count tract_month_ranking tract_month_ranking__top_10pct_overlap_rate      0.052411              True    4770
          A1                       A1_svi_direct_ranking__svi_score_raw           primary_continuous_svi water_drainage_count tract_month_ranking tract_month_ranking__top_10pct_overlap_rate      0.052411              True    4770
```

## Wide tract-month ranking table

```text
         split_name      comparison_metric  A0_3_tract_train_mean  A0_4_tract_month_of_year_train_mean  A0_5_previous_month_persistence  A0_8_non_water_311_reporting_exposure_retrospective  A1_svi_direct_ranking__svi_percentile  A1_svi_direct_ranking__svi_score_raw  A1_svi_direct_ranking__svi_class__ordinal_class_diagnostic  A0_1_global_train_mean  A0_2_month_of_year_train_mean  A0_6_previous_year_same_month_persistence  A0_7_population_exposure_train_rate  A1_svi_direct_ranking__svi_rank__rank_reversed
      temporal_test            ndcg_at_100               0.748363                             0.636779                         0.598551                                             0.662923                               0.220560                              0.220560                                                    0.222316                0.310433                       0.133566                                   0.536562                             0.261060                                        0.220560
      temporal_test          spearman_corr               0.694235                             0.616683                         0.593867                                             0.631671                               0.160639                              0.160630                                                    0.185642                     NaN                       0.014116                                   0.533484                             0.274328                                        0.170839
      temporal_test top_10pct_overlap_rate               0.483539                             0.448560                         0.390947                                             0.403292                               0.052411                              0.052411                                                    0.150943                0.100823                       0.026749                                   0.388889                             0.263374                                        0.054167
temporal_validation            ndcg_at_100               0.691490                             0.715413                         0.649880                                             0.637579                               0.193465                              0.193465                                                    0.172898                0.302150                       0.232108                                   0.597957                             0.200621                                        0.193465
temporal_validation          spearman_corr               0.687738                             0.676538                         0.637232                                             0.680659                               0.158606                              0.158599                                                    0.174728                     NaN                       0.213148                                   0.597096                             0.268426                                        0.168950
temporal_validation top_10pct_overlap_rate               0.497685                             0.513889                         0.456019                                             0.465278                               0.066038                              0.066038                                                    0.110849                0.085648                       0.136574                                   0.451389                             0.256944                                        0.065574
```

## Interpretation guardrails

- A0 and A1 are not the same type of baseline: A0 includes historical target information, while A1 is a static vulnerability score.
- A1 should not be expected to beat historical tract burden on direct monthly 311 prediction. Its fair role is vulnerability-prior ranking.
- `svi_class` is diagnostic because it is an ordinal class label; primary A1 interpretation should use `svi_percentile` or `svi_score_raw`.
- This comparison supports A2: calibrated SVI/regression-style baselines are the fairer analogue to literature that validates SVI with controls.

## Output artifacts

| Artifact | Path |
|---|---|
| `metrics_long` | `/home/tim/Documents/ville_ai/indices_BACKUP_before_clean_push/urban_graph_benchmark/outputs/mtl_311_water_v0/baselines/A0_A1_comparison/a0_a1_metrics_long.csv` |
| `tract_month_ranking_wide` | `/home/tim/Documents/ville_ai/indices_BACKUP_before_clean_push/urban_graph_benchmark/outputs/mtl_311_water_v0/baselines/A0_A1_comparison/a0_a1_tract_month_ranking_wide.csv` |
| `test_headline_table` | `/home/tim/Documents/ville_ai/indices_BACKUP_before_clean_push/urban_graph_benchmark/outputs/mtl_311_water_v0/baselines/A0_A1_comparison/a0_a1_test_headline_table.csv` |
| `comparison_report` | `/home/tim/Documents/ville_ai/indices_BACKUP_before_clean_push/urban_graph_benchmark/outputs/mtl_311_water_v0/baselines/A0_A1_comparison/a0_a1_comparison_report.md` |
| `comparison_metadata` | `/home/tim/Documents/ville_ai/indices_BACKUP_before_clean_push/urban_graph_benchmark/outputs/mtl_311_water_v0/baselines/A0_A1_comparison/comparison_metadata.json` |
