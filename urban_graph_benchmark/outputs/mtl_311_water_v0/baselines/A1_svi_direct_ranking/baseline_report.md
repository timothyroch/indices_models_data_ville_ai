# A1 SVI Direct-Ranking Baseline — Montréal 311 Water/Drainage v0

Generated at: `2026-06-15T17:24:03.234365+00:00`

Split scheme: `temporal`

Split type: `temporal`

## Purpose

A1 evaluates static SVI as a direct prioritization score. It does not fit a count model and does not calibrate SVI to 311 outcomes. It asks whether tracts with higher SVI scores are also tracts with higher reported water/drainage 311 burden in validation/test windows.

## Row counts

| Partition | Rows |
|---|---:|
| `train` | 19440 |
| `validation` | 4320 |
| `test` | 4860 |

## Primary recommended SVI score columns

| Source column | Oriented score column | Orientation | Non-missing |
|---|---|---|---:|
| `svi_percentile` | `svi_percentile__higher_more_vulnerable` | `positive_higher_more_vulnerable` | 28090 |
| `svi_score_raw` | `svi_score_raw__higher_more_vulnerable` | `positive_higher_more_vulnerable` | 28090 |

## Diagnostic SVI score columns

| Source column | Score role | Oriented score column | Non-missing |
|---|---|---|---:|
| `svi_rank` | `diagnostic_rank_reversed_score` | `svi_rank__rank_reversed_for_vulnerability` | 28249 |
| `svi_class` | `diagnostic_ordinal_class_score_not_primary` | `svi_class__higher_more_vulnerable` | 28090 |

## Static-score audit

```text
 source_column                              score_column                                 score_role  zones  zones_with_multiple_values  max_unique_values_within_zone                status examples
svi_percentile    svi_percentile__higher_more_vulnerable     primary_continuous_svi_score_candidate    540                           0                              1 ok_static_within_zone       []
 svi_score_raw     svi_score_raw__higher_more_vulnerable     primary_continuous_svi_score_candidate    540                           0                              1 ok_static_within_zone       []
      svi_rank svi_rank__rank_reversed_for_vulnerability             diagnostic_rank_reversed_score    540                           0                              1 ok_static_within_zone       []
     svi_class         svi_class__higher_more_vulnerable diagnostic_ordinal_class_score_not_primary    540                           0                              1 ok_static_within_zone       []
```

## Compact metrics summary

```text
         split_name                            model_name                                               target_name         target_type                                                                                            metric_name  metric_value  higher_is_better  n_rows
temporal_validation A1_svi_direct_ranking__svi_percentile                                      water_drainage_count tract_month_ranking                                                                     tract_month_ranking__spearman_corr      0.158606              True    4240
temporal_validation A1_svi_direct_ranking__svi_percentile                                      water_drainage_count tract_month_ranking                                                                      tract_month_ranking__kendall_corr      0.108709              True    4240
temporal_validation A1_svi_direct_ranking__svi_percentile                                      water_drainage_count tract_month_ranking                                                                tract_month_ranking__top10_overlap_rate      0.000000              True    4240
temporal_validation A1_svi_direct_ranking__svi_percentile                                      water_drainage_count tract_month_ranking                                                                        tract_month_ranking__ndcg_at_10      0.147259              True    4240
temporal_validation A1_svi_direct_ranking__svi_percentile                                      water_drainage_count tract_month_ranking                                                                tract_month_ranking__top25_overlap_rate      0.000000              True    4240
temporal_validation A1_svi_direct_ranking__svi_percentile                                      water_drainage_count tract_month_ranking                                                                        tract_month_ranking__ndcg_at_25      0.201063              True    4240
temporal_validation A1_svi_direct_ranking__svi_percentile                                      water_drainage_count tract_month_ranking                                                                tract_month_ranking__top50_overlap_rate      0.000000              True    4240
temporal_validation A1_svi_direct_ranking__svi_percentile                                      water_drainage_count tract_month_ranking                                                                        tract_month_ranking__ndcg_at_50      0.189843              True    4240
temporal_validation A1_svi_direct_ranking__svi_percentile                                      water_drainage_count tract_month_ranking                                                               tract_month_ranking__top100_overlap_rate      0.020000              True    4240
temporal_validation A1_svi_direct_ranking__svi_percentile                                      water_drainage_count tract_month_ranking                                                                       tract_month_ranking__ndcg_at_100      0.193465              True    4240
temporal_validation A1_svi_direct_ranking__svi_percentile                                      water_drainage_count tract_month_ranking                                                             tract_month_ranking__top_5pct_overlap_rate      0.023585              True    4240
temporal_validation A1_svi_direct_ranking__svi_percentile                                      water_drainage_count tract_month_ranking                                                            tract_month_ranking__top_10pct_overlap_rate      0.066038              True    4240
      temporal_test A1_svi_direct_ranking__svi_percentile                                      water_drainage_count tract_month_ranking                                                                     tract_month_ranking__spearman_corr      0.160639              True    4770
      temporal_test A1_svi_direct_ranking__svi_percentile                                      water_drainage_count tract_month_ranking                                                                      tract_month_ranking__kendall_corr      0.111152              True    4770
      temporal_test A1_svi_direct_ranking__svi_percentile                                      water_drainage_count tract_month_ranking                                                                tract_month_ranking__top10_overlap_rate      0.000000              True    4770
      temporal_test A1_svi_direct_ranking__svi_percentile                                      water_drainage_count tract_month_ranking                                                                        tract_month_ranking__ndcg_at_10      0.188418              True    4770
      temporal_test A1_svi_direct_ranking__svi_percentile                                      water_drainage_count tract_month_ranking                                                                tract_month_ranking__top25_overlap_rate      0.000000              True    4770
      temporal_test A1_svi_direct_ranking__svi_percentile                                      water_drainage_count tract_month_ranking                                                                        tract_month_ranking__ndcg_at_25      0.254751              True    4770
      temporal_test A1_svi_direct_ranking__svi_percentile                                      water_drainage_count tract_month_ranking                                                                tract_month_ranking__top50_overlap_rate      0.000000              True    4770
      temporal_test A1_svi_direct_ranking__svi_percentile                                      water_drainage_count tract_month_ranking                                                                        tract_month_ranking__ndcg_at_50      0.238587              True    4770
      temporal_test A1_svi_direct_ranking__svi_percentile                                      water_drainage_count tract_month_ranking                                                               tract_month_ranking__top100_overlap_rate      0.000000              True    4770
      temporal_test A1_svi_direct_ranking__svi_percentile                                      water_drainage_count tract_month_ranking                                                                       tract_month_ranking__ndcg_at_100      0.220560              True    4770
      temporal_test A1_svi_direct_ranking__svi_percentile                                      water_drainage_count tract_month_ranking                                                             tract_month_ranking__top_5pct_overlap_rate      0.016736              True    4770
      temporal_test A1_svi_direct_ranking__svi_percentile                                      water_drainage_count tract_month_ranking                                                            tract_month_ranking__top_10pct_overlap_rate      0.052411              True    4770
temporal_validation A1_svi_direct_ranking__svi_percentile                          water_drainage_count_tract_total tract_level_ranking                                   tract_level_ranking__water_drainage_count_tract_total__spearman_corr      0.208687              True     530
temporal_validation A1_svi_direct_ranking__svi_percentile                          water_drainage_count_tract_total tract_level_ranking                                    tract_level_ranking__water_drainage_count_tract_total__kendall_corr      0.135876              True     530
temporal_validation A1_svi_direct_ranking__svi_percentile                          water_drainage_count_tract_total tract_level_ranking                              tract_level_ranking__water_drainage_count_tract_total__top10_overlap_rate      0.000000              True     530
temporal_validation A1_svi_direct_ranking__svi_percentile                          water_drainage_count_tract_total tract_level_ranking                                      tract_level_ranking__water_drainage_count_tract_total__ndcg_at_10      0.270682              True     530
temporal_validation A1_svi_direct_ranking__svi_percentile                          water_drainage_count_tract_total tract_level_ranking                              tract_level_ranking__water_drainage_count_tract_total__top25_overlap_rate      0.000000              True     530
temporal_validation A1_svi_direct_ranking__svi_percentile                          water_drainage_count_tract_total tract_level_ranking                                      tract_level_ranking__water_drainage_count_tract_total__ndcg_at_25      0.281395              True     530
temporal_validation A1_svi_direct_ranking__svi_percentile                          water_drainage_count_tract_total tract_level_ranking                              tract_level_ranking__water_drainage_count_tract_total__top50_overlap_rate      0.020000              True     530
temporal_validation A1_svi_direct_ranking__svi_percentile                          water_drainage_count_tract_total tract_level_ranking                                      tract_level_ranking__water_drainage_count_tract_total__ndcg_at_50      0.326038              True     530
temporal_validation A1_svi_direct_ranking__svi_percentile                          water_drainage_count_tract_total tract_level_ranking                             tract_level_ranking__water_drainage_count_tract_total__top100_overlap_rate      0.120000              True     530
temporal_validation A1_svi_direct_ranking__svi_percentile                          water_drainage_count_tract_total tract_level_ranking                                     tract_level_ranking__water_drainage_count_tract_total__ndcg_at_100      0.413050              True     530
temporal_validation A1_svi_direct_ranking__svi_percentile                          water_drainage_count_tract_total tract_level_ranking                           tract_level_ranking__water_drainage_count_tract_total__top_5pct_overlap_rate      0.000000              True     530
temporal_validation A1_svi_direct_ranking__svi_percentile                          water_drainage_count_tract_total tract_level_ranking                          tract_level_ranking__water_drainage_count_tract_total__top_10pct_overlap_rate      0.037736              True     530
temporal_validation A1_svi_direct_ranking__svi_percentile                     water_drainage_count_tract_month_mean tract_level_ranking                              tract_level_ranking__water_drainage_count_tract_month_mean__spearman_corr      0.208687              True     530
temporal_validation A1_svi_direct_ranking__svi_percentile                     water_drainage_count_tract_month_mean tract_level_ranking                               tract_level_ranking__water_drainage_count_tract_month_mean__kendall_corr      0.135876              True     530
temporal_validation A1_svi_direct_ranking__svi_percentile                     water_drainage_count_tract_month_mean tract_level_ranking                         tract_level_ranking__water_drainage_count_tract_month_mean__top10_overlap_rate      0.000000              True     530
temporal_validation A1_svi_direct_ranking__svi_percentile                     water_drainage_count_tract_month_mean tract_level_ranking                                 tract_level_ranking__water_drainage_count_tract_month_mean__ndcg_at_10      0.270682              True     530
temporal_validation A1_svi_direct_ranking__svi_percentile                     water_drainage_count_tract_month_mean tract_level_ranking                         tract_level_ranking__water_drainage_count_tract_month_mean__top25_overlap_rate      0.000000              True     530
temporal_validation A1_svi_direct_ranking__svi_percentile                     water_drainage_count_tract_month_mean tract_level_ranking                                 tract_level_ranking__water_drainage_count_tract_month_mean__ndcg_at_25      0.281395              True     530
temporal_validation A1_svi_direct_ranking__svi_percentile                     water_drainage_count_tract_month_mean tract_level_ranking                         tract_level_ranking__water_drainage_count_tract_month_mean__top50_overlap_rate      0.020000              True     530
temporal_validation A1_svi_direct_ranking__svi_percentile                     water_drainage_count_tract_month_mean tract_level_ranking                                 tract_level_ranking__water_drainage_count_tract_month_mean__ndcg_at_50      0.326038              True     530
temporal_validation A1_svi_direct_ranking__svi_percentile                     water_drainage_count_tract_month_mean tract_level_ranking                        tract_level_ranking__water_drainage_count_tract_month_mean__top100_overlap_rate      0.120000              True     530
temporal_validation A1_svi_direct_ranking__svi_percentile                     water_drainage_count_tract_month_mean tract_level_ranking                                tract_level_ranking__water_drainage_count_tract_month_mean__ndcg_at_100      0.413050              True     530
temporal_validation A1_svi_direct_ranking__svi_percentile                     water_drainage_count_tract_month_mean tract_level_ranking                      tract_level_ranking__water_drainage_count_tract_month_mean__top_5pct_overlap_rate      0.000000              True     530
temporal_validation A1_svi_direct_ranking__svi_percentile                     water_drainage_count_tract_month_mean tract_level_ranking                     tract_level_ranking__water_drainage_count_tract_month_mean__top_10pct_overlap_rate      0.037736              True     530
temporal_validation A1_svi_direct_ranking__svi_percentile water_drainage_count_tract_total_rate_per_1000_population tract_level_ranking          tract_level_ranking__water_drainage_count_tract_total_rate_per_1000_population__spearman_corr     -0.142233              True     530
temporal_validation A1_svi_direct_ranking__svi_percentile water_drainage_count_tract_total_rate_per_1000_population tract_level_ranking           tract_level_ranking__water_drainage_count_tract_total_rate_per_1000_population__kendall_corr     -0.094486              True     530
temporal_validation A1_svi_direct_ranking__svi_percentile water_drainage_count_tract_total_rate_per_1000_population tract_level_ranking     tract_level_ranking__water_drainage_count_tract_total_rate_per_1000_population__top10_overlap_rate      0.000000              True     530
temporal_validation A1_svi_direct_ranking__svi_percentile water_drainage_count_tract_total_rate_per_1000_population tract_level_ranking             tract_level_ranking__water_drainage_count_tract_total_rate_per_1000_population__ndcg_at_10      0.180578              True     530
temporal_validation A1_svi_direct_ranking__svi_percentile water_drainage_count_tract_total_rate_per_1000_population tract_level_ranking     tract_level_ranking__water_drainage_count_tract_total_rate_per_1000_population__top25_overlap_rate      0.000000              True     530
temporal_validation A1_svi_direct_ranking__svi_percentile water_drainage_count_tract_total_rate_per_1000_population tract_level_ranking             tract_level_ranking__water_drainage_count_tract_total_rate_per_1000_population__ndcg_at_25      0.204284              True     530
temporal_validation A1_svi_direct_ranking__svi_percentile water_drainage_count_tract_total_rate_per_1000_population tract_level_ranking     tract_level_ranking__water_drainage_count_tract_total_rate_per_1000_population__top50_overlap_rate      0.000000              True     530
temporal_validation A1_svi_direct_ranking__svi_percentile water_drainage_count_tract_total_rate_per_1000_population tract_level_ranking             tract_level_ranking__water_drainage_count_tract_total_rate_per_1000_population__ndcg_at_50      0.252697              True     530
temporal_validation A1_svi_direct_ranking__svi_percentile water_drainage_count_tract_total_rate_per_1000_population tract_level_ranking    tract_level_ranking__water_drainage_count_tract_total_rate_per_1000_population__top100_overlap_rate      0.050000              True     530
temporal_validation A1_svi_direct_ranking__svi_percentile water_drainage_count_tract_total_rate_per_1000_population tract_level_ranking            tract_level_ranking__water_drainage_count_tract_total_rate_per_1000_population__ndcg_at_100      0.331929              True     530
temporal_validation A1_svi_direct_ranking__svi_percentile water_drainage_count_tract_total_rate_per_1000_population tract_level_ranking  tract_level_ranking__water_drainage_count_tract_total_rate_per_1000_population__top_5pct_overlap_rate      0.000000              True     530
temporal_validation A1_svi_direct_ranking__svi_percentile water_drainage_count_tract_total_rate_per_1000_population tract_level_ranking tract_level_ranking__water_drainage_count_tract_total_rate_per_1000_population__top_10pct_overlap_rate      0.000000              True     530
      temporal_test A1_svi_direct_ranking__svi_percentile                          water_drainage_count_tract_total tract_level_ranking                                   tract_level_ranking__water_drainage_count_tract_total__spearman_corr      0.198890              True     530
      temporal_test A1_svi_direct_ranking__svi_percentile                          water_drainage_count_tract_total tract_level_ranking                                    tract_level_ranking__water_drainage_count_tract_total__kendall_corr      0.131574              True     530
      temporal_test A1_svi_direct_ranking__svi_percentile                          water_drainage_count_tract_total tract_level_ranking                              tract_level_ranking__water_drainage_count_tract_total__top10_overlap_rate      0.000000              True     530
      temporal_test A1_svi_direct_ranking__svi_percentile                          water_drainage_count_tract_total tract_level_ranking                                      tract_level_ranking__water_drainage_count_tract_total__ndcg_at_10      0.294005              True     530
      temporal_test A1_svi_direct_ranking__svi_percentile                          water_drainage_count_tract_total tract_level_ranking                              tract_level_ranking__water_drainage_count_tract_total__top25_overlap_rate      0.000000              True     530
      temporal_test A1_svi_direct_ranking__svi_percentile                          water_drainage_count_tract_total tract_level_ranking                                      tract_level_ranking__water_drainage_count_tract_total__ndcg_at_25      0.286569              True     530
      temporal_test A1_svi_direct_ranking__svi_percentile                          water_drainage_count_tract_total tract_level_ranking                              tract_level_ranking__water_drainage_count_tract_total__top50_overlap_rate      0.060000              True     530
      temporal_test A1_svi_direct_ranking__svi_percentile                          water_drainage_count_tract_total tract_level_ranking                                      tract_level_ranking__water_drainage_count_tract_total__ndcg_at_50      0.331540              True     530
      temporal_test A1_svi_direct_ranking__svi_percentile                          water_drainage_count_tract_total tract_level_ranking                             tract_level_ranking__water_drainage_count_tract_total__top100_overlap_rate      0.160000              True     530
      temporal_test A1_svi_direct_ranking__svi_percentile                          water_drainage_count_tract_total tract_level_ranking                                     tract_level_ranking__water_drainage_count_tract_total__ndcg_at_100      0.419181              True     530
      temporal_test A1_svi_direct_ranking__svi_percentile                          water_drainage_count_tract_total tract_level_ranking                           tract_level_ranking__water_drainage_count_tract_total__top_5pct_overlap_rate      0.000000              True     530
      temporal_test A1_svi_direct_ranking__svi_percentile                          water_drainage_count_tract_total tract_level_ranking                          tract_level_ranking__water_drainage_count_tract_total__top_10pct_overlap_rate      0.056604              True     530
      temporal_test A1_svi_direct_ranking__svi_percentile                     water_drainage_count_tract_month_mean tract_level_ranking                              tract_level_ranking__water_drainage_count_tract_month_mean__spearman_corr      0.198890              True     530
      temporal_test A1_svi_direct_ranking__svi_percentile                     water_drainage_count_tract_month_mean tract_level_ranking                               tract_level_ranking__water_drainage_count_tract_month_mean__kendall_corr      0.131574              True     530
      temporal_test A1_svi_direct_ranking__svi_percentile                     water_drainage_count_tract_month_mean tract_level_ranking                         tract_level_ranking__water_drainage_count_tract_month_mean__top10_overlap_rate      0.000000              True     530
      temporal_test A1_svi_direct_ranking__svi_percentile                     water_drainage_count_tract_month_mean tract_level_ranking                                 tract_level_ranking__water_drainage_count_tract_month_mean__ndcg_at_10      0.294005              True     530
      temporal_test A1_svi_direct_ranking__svi_percentile                     water_drainage_count_tract_month_mean tract_level_ranking                         tract_level_ranking__water_drainage_count_tract_month_mean__top25_overlap_rate      0.000000              True     530
      temporal_test A1_svi_direct_ranking__svi_percentile                     water_drainage_count_tract_month_mean tract_level_ranking                                 tract_level_ranking__water_drainage_count_tract_month_mean__ndcg_at_25      0.286569              True     530
      temporal_test A1_svi_direct_ranking__svi_percentile                     water_drainage_count_tract_month_mean tract_level_ranking                         tract_level_ranking__water_drainage_count_tract_month_mean__top50_overlap_rate      0.060000              True     530
      temporal_test A1_svi_direct_ranking__svi_percentile                     water_drainage_count_tract_month_mean tract_level_ranking                                 tract_level_ranking__water_drainage_count_tract_month_mean__ndcg_at_50      0.331540              True     530
      temporal_test A1_svi_direct_ranking__svi_percentile                     water_drainage_count_tract_month_mean tract_level_ranking                        tract_level_ranking__water_drainage_count_tract_month_mean__top100_overlap_rate      0.160000              True     530
      temporal_test A1_svi_direct_ranking__svi_percentile                     water_drainage_count_tract_month_mean tract_level_ranking                                tract_level_ranking__water_drainage_count_tract_month_mean__ndcg_at_100      0.419181              True     530
      temporal_test A1_svi_direct_ranking__svi_percentile                     water_drainage_count_tract_month_mean tract_level_ranking                      tract_level_ranking__water_drainage_count_tract_month_mean__top_5pct_overlap_rate      0.000000              True     530
      temporal_test A1_svi_direct_ranking__svi_percentile                     water_drainage_count_tract_month_mean tract_level_ranking                     tract_level_ranking__water_drainage_count_tract_month_mean__top_10pct_overlap_rate      0.056604              True     530
      temporal_test A1_svi_direct_ranking__svi_percentile water_drainage_count_tract_total_rate_per_1000_population tract_level_ranking          tract_level_ranking__water_drainage_count_tract_total_rate_per_1000_population__spearman_corr     -0.147282              True     530
      temporal_test A1_svi_direct_ranking__svi_percentile water_drainage_count_tract_total_rate_per_1000_population tract_level_ranking           tract_level_ranking__water_drainage_count_tract_total_rate_per_1000_population__kendall_corr     -0.101133              True     530
      temporal_test A1_svi_direct_ranking__svi_percentile water_drainage_count_tract_total_rate_per_1000_population tract_level_ranking     tract_level_ranking__water_drainage_count_tract_total_rate_per_1000_population__top10_overlap_rate      0.000000              True     530
      temporal_test A1_svi_direct_ranking__svi_percentile water_drainage_count_tract_total_rate_per_1000_population tract_level_ranking             tract_level_ranking__water_drainage_count_tract_total_rate_per_1000_population__ndcg_at_10      0.161915              True     530
      temporal_test A1_svi_direct_ranking__svi_percentile water_drainage_count_tract_total_rate_per_1000_population tract_level_ranking     tract_level_ranking__water_drainage_count_tract_total_rate_per_1000_population__top25_overlap_rate      0.000000              True     530
      temporal_test A1_svi_direct_ranking__svi_percentile water_drainage_count_tract_total_rate_per_1000_population tract_level_ranking             tract_level_ranking__water_drainage_count_tract_total_rate_per_1000_population__ndcg_at_25      0.176529              True     530
      temporal_test A1_svi_direct_ranking__svi_percentile water_drainage_count_tract_total_rate_per_1000_population tract_level_ranking     tract_level_ranking__water_drainage_count_tract_total_rate_per_1000_population__top50_overlap_rate      0.020000              True     530
      temporal_test A1_svi_direct_ranking__svi_percentile water_drainage_count_tract_total_rate_per_1000_population tract_level_ranking             tract_level_ranking__water_drainage_count_tract_total_rate_per_1000_population__ndcg_at_50      0.230340              True     530
      temporal_test A1_svi_direct_ranking__svi_percentile water_drainage_count_tract_total_rate_per_1000_population tract_level_ranking    tract_level_ranking__water_drainage_count_tract_total_rate_per_1000_population__top100_overlap_rate      0.080000              True     530
      temporal_test A1_svi_direct_ranking__svi_percentile water_drainage_count_tract_total_rate_per_1000_population tract_level_ranking            tract_level_ranking__water_drainage_count_tract_total_rate_per_1000_population__ndcg_at_100      0.310888              True     530
      temporal_test A1_svi_direct_ranking__svi_percentile water_drainage_count_tract_total_rate_per_1000_population tract_level_ranking  tract_level_ranking__water_drainage_count_tract_total_rate_per_1000_population__top_5pct_overlap_rate      0.000000              True     530
      temporal_test A1_svi_direct_ranking__svi_percentile water_drainage_count_tract_total_rate_per_1000_population tract_level_ranking tract_level_ranking__water_drainage_count_tract_total_rate_per_1000_population__top_10pct_overlap_rate      0.018868              True     530
temporal_validation  A1_svi_direct_ranking__svi_score_raw                                      water_drainage_count tract_month_ranking                                                                     tract_month_ranking__spearman_corr      0.158599              True    4240
temporal_validation  A1_svi_direct_ranking__svi_score_raw                                      water_drainage_count tract_month_ranking                                                                      tract_month_ranking__kendall_corr      0.108704              True    4240
temporal_validation  A1_svi_direct_ranking__svi_score_raw                                      water_drainage_count tract_month_ranking                                                                tract_month_ranking__top10_overlap_rate      0.000000              True    4240
temporal_validation  A1_svi_direct_ranking__svi_score_raw                                      water_drainage_count tract_month_ranking                                                                        tract_month_ranking__ndcg_at_10      0.147259              True    4240
temporal_validation  A1_svi_direct_ranking__svi_score_raw                                      water_drainage_count tract_month_ranking                                                                tract_month_ranking__top25_overlap_rate      0.000000              True    4240
temporal_validation  A1_svi_direct_ranking__svi_score_raw                                      water_drainage_count tract_month_ranking                                                                        tract_month_ranking__ndcg_at_25      0.201063              True    4240
temporal_validation  A1_svi_direct_ranking__svi_score_raw                                      water_drainage_count tract_month_ranking                                                                tract_month_ranking__top50_overlap_rate      0.000000              True    4240
temporal_validation  A1_svi_direct_ranking__svi_score_raw                                      water_drainage_count tract_month_ranking                                                                        tract_month_ranking__ndcg_at_50      0.189843              True    4240
temporal_validation  A1_svi_direct_ranking__svi_score_raw                                      water_drainage_count tract_month_ranking                                                               tract_month_ranking__top100_overlap_rate      0.020000              True    4240
temporal_validation  A1_svi_direct_ranking__svi_score_raw                                      water_drainage_count tract_month_ranking                                                                       tract_month_ranking__ndcg_at_100      0.193465              True    4240
temporal_validation  A1_svi_direct_ranking__svi_score_raw                                      water_drainage_count tract_month_ranking                                                             tract_month_ranking__top_5pct_overlap_rate      0.023585              True    4240
temporal_validation  A1_svi_direct_ranking__svi_score_raw                                      water_drainage_count tract_month_ranking                                                            tract_month_ranking__top_10pct_overlap_rate      0.066038              True    4240
      temporal_test  A1_svi_direct_ranking__svi_score_raw                                      water_drainage_count tract_month_ranking                                                                     tract_month_ranking__spearman_corr      0.160630              True    4770
      temporal_test  A1_svi_direct_ranking__svi_score_raw                                      water_drainage_count tract_month_ranking                                                                      tract_month_ranking__kendall_corr      0.111146              True    4770
      temporal_test  A1_svi_direct_ranking__svi_score_raw                                      water_drainage_count tract_month_ranking                                                                tract_month_ranking__top10_overlap_rate      0.000000              True    4770
      temporal_test  A1_svi_direct_ranking__svi_score_raw                                      water_drainage_count tract_month_ranking                                                                        tract_month_ranking__ndcg_at_10      0.188418              True    4770
      temporal_test  A1_svi_direct_ranking__svi_score_raw                                      water_drainage_count tract_month_ranking                                                                tract_month_ranking__top25_overlap_rate      0.000000              True    4770
      temporal_test  A1_svi_direct_ranking__svi_score_raw                                      water_drainage_count tract_month_ranking                                                                        tract_month_ranking__ndcg_at_25      0.254751              True    4770
      temporal_test  A1_svi_direct_ranking__svi_score_raw                                      water_drainage_count tract_month_ranking                                                                tract_month_ranking__top50_overlap_rate      0.000000              True    4770
      temporal_test  A1_svi_direct_ranking__svi_score_raw                                      water_drainage_count tract_month_ranking                                                                        tract_month_ranking__ndcg_at_50      0.238587              True    4770
      temporal_test  A1_svi_direct_ranking__svi_score_raw                                      water_drainage_count tract_month_ranking                                                               tract_month_ranking__top100_overlap_rate      0.000000              True    4770
      temporal_test  A1_svi_direct_ranking__svi_score_raw                                      water_drainage_count tract_month_ranking                                                                       tract_month_ranking__ndcg_at_100      0.220560              True    4770
      temporal_test  A1_svi_direct_ranking__svi_score_raw                                      water_drainage_count tract_month_ranking                                                             tract_month_ranking__top_5pct_overlap_rate      0.016736              True    4770
      temporal_test  A1_svi_direct_ranking__svi_score_raw                                      water_drainage_count tract_month_ranking                                                            tract_month_ranking__top_10pct_overlap_rate      0.052411              True    4770
```

## Output artifacts

| Artifact | Path |
|---|---|
| `metrics` | `/home/tim/Documents/ville_ai/indices_BACKUP_before_clean_push/urban_graph_benchmark/outputs/mtl_311_water_v0/baselines/A1_svi_direct_ranking/metrics.csv` |
| `model_metadata` | `/home/tim/Documents/ville_ai/indices_BACKUP_before_clean_push/urban_graph_benchmark/outputs/mtl_311_water_v0/baselines/A1_svi_direct_ranking/model_metadata.json` |
| `baseline_report` | `/home/tim/Documents/ville_ai/indices_BACKUP_before_clean_push/urban_graph_benchmark/outputs/mtl_311_water_v0/baselines/A1_svi_direct_ranking/baseline_report.md` |
| `tract_ranking_validation` | `/home/tim/Documents/ville_ai/indices_BACKUP_before_clean_push/urban_graph_benchmark/outputs/mtl_311_water_v0/baselines/A1_svi_direct_ranking/tract_ranking_validation.csv` |
| `tract_ranking_test` | `/home/tim/Documents/ville_ai/indices_BACKUP_before_clean_push/urban_graph_benchmark/outputs/mtl_311_water_v0/baselines/A1_svi_direct_ranking/tract_ranking_test.csv` |
| `topk_overlap` | `/home/tim/Documents/ville_ai/indices_BACKUP_before_clean_push/urban_graph_benchmark/outputs/mtl_311_water_v0/baselines/A1_svi_direct_ranking/topk_overlap.csv` |
| `svi_score_audit` | `/home/tim/Documents/ville_ai/indices_BACKUP_before_clean_push/urban_graph_benchmark/outputs/mtl_311_water_v0/baselines/A1_svi_direct_ranking/svi_score_audit.csv` |
| `svi_static_score_audit` | `/home/tim/Documents/ville_ai/indices_BACKUP_before_clean_push/urban_graph_benchmark/outputs/mtl_311_water_v0/baselines/A1_svi_direct_ranking/svi_static_score_audit.csv` |
| `predictions_validation` | `/home/tim/Documents/ville_ai/indices_BACKUP_before_clean_push/urban_graph_benchmark/outputs/mtl_311_water_v0/baselines/A1_svi_direct_ranking/predictions_validation.parquet` |
| `predictions_test` | `/home/tim/Documents/ville_ai/indices_BACKUP_before_clean_push/urban_graph_benchmark/outputs/mtl_311_water_v0/baselines/A1_svi_direct_ranking/predictions_test.parquet` |

## Interpretation notes

- A1 is a static ranking baseline, not a calibrated predictor.
- Continuous SVI score/percentile columns should be interpreted as the primary A1 result when available.
- SVI class-label columns are diagnostic ordinal fallbacks only; their spacing is not interval-scaled.
- SVI status/quality fields such as `svi_scored` are excluded from evaluation as vulnerability scores.
- SVI is not hazard-specific; weak alignment with water/drainage 311 burden does not invalidate SVI.
- The target is reported municipal 311 burden, not objective flood occurrence.
- Rank-column orientation is deterministic and not tuned on validation/test outcomes.
- SoVI is excluded from Track A because the SoVI reproduction is census-division scale, while this benchmark is census tract × month.
