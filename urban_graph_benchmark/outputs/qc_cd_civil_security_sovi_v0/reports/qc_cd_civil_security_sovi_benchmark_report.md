# Québec CD civil-security / SoVI benchmark report

Generated: **2026-06-17 17:15 UTC**

## Executive answer

**Verdict:** Strong evidence that real graph structure adds value.

The real adjacency graph beats B3 feature-parity, no-edge neural, and random/placebo graph controls on most primary metrics, including MAE.

This report compares direct SoVI validation, history-only baselines, calibrated SoVI, tabular feature-parity ML, no-edge neural controls, random/placebo graph controls, kNN graph controls, and the real CD adjacency graph.

## What would count as graph value?

The real graph claim is strongest only if **B4_real_cd_graph** improves over all of the following:

1. **B3_tabular_feature_parity**: same non-graph node features with strong tabular ML.
2. **B4_no_edge_neural**: same neural architecture family but no message passing.
3. **B4_random_edge_graph**: same graph architecture but placebo topology.
4. **B4_knn_graph**: generic spatial-proximity topology.

If the real graph only beats B1/B2, that is not enough: it may simply be learning from history/features. If it beats no-edge but not random graph, topology-specific value is weak. If it beats random/no-edge but not B3, graph message passing may help neural modeling but not yet surpass feature-parity tabular ML.

## Compact benchmark comparison

| primary_metric_rank | baseline_stage | display_name | model_name | graph_kind | split | mae | rmse | mean_poisson_deviance | spearman | ndcg_at_25 | top10_overlap_rate |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1.0000 | B4 | B4 kNN graph | B4_knn_graph | knn | test | 1.2889 | 2.2958 | 2.7890 | 0.3007 | 0.2880 | 0.0899 |
| 2.0000 | B4 | B4 real CD adjacency graph | B4_real_cd_graph | adjacency | test | 1.2912 | 2.2937 | 2.7789 | 0.2741 | 0.2383 | 0.0787 |
| 3.0000 | B4 | B4 random-edge graph | B4_random_edge_graph | random | test | 1.2913 | 2.2995 | 2.8130 | 0.2452 | 0.2012 | 0.1011 |
| 4.0000 | B4 | B4 no-edge neural | B4_no_edge_neural | none | test | 1.3271 | 2.3354 | 3.0005 | -0.0625 | 0.1080 | 0.1348 |
| 5.0000 | B0 | B0 history-only | previous_month | none | test | 1.4308 | 2.6234 | 53.3926 | 0.1143 | 0.2054 | 0.1236 |
| 6.0000 | B2 | B2 calibrated SoVI | poisson_sovi | none | test | 1.6295 | 2.3122 | 2.7744 | -0.0123 | 0.1185 | 0.0337 |
| 7.0000 | B2 | B2 calibrated SoVI | ridge_sovi | none | test | 1.6400 | 2.3342 | 2.8332 | -0.0123 | 0.1185 | 0.0337 |
| 8.0000 | B2 | B2 calibrated SoVI | linear_sovi | none | test | 1.6401 | 2.3344 | 2.8337 | -0.0123 | 0.1185 | 0.0337 |
| 9.0000 | B2 | B2 calibrated SoVI | poisson_sovi_seasonal | none | test | 1.6468 | 2.3606 | 2.8155 | 0.1870 | 0.1370 | 0.1124 |
| 10.0000 | B2 | B2 calibrated SoVI | ridge_sovi_seasonal | none | test | 1.6513 | 2.3602 | 2.8493 | 0.1857 | 0.1366 | 0.1011 |
| 11.0000 | B2 | B2 calibrated SoVI | linear_sovi_seasonal | none | test | 1.6517 | 2.3608 | 2.8508 | 0.1857 | 0.1366 | 0.1011 |
| 12.0000 | B0 | B0 history-only | rolling_3_months | none | test | 1.7687 | 2.9715 | 28.8301 | 0.1312 | 0.1413 | 0.1124 |
| 13.0000 | B0 | B0 history-only | seasonal_historical_mean | none | test | 2.0482 | 3.5215 | 8.3901 | 0.2878 | 0.1348 | 0.1685 |
| 14.0000 | B3 | B3 tabular feature parity | random_forest | none | test | 2.0695 | 2.8982 | 3.3092 | 0.2272 | 0.1575 | 0.1685 |
| 15.0000 | B3 | B3 tabular feature parity | hist_gradient_boosting | none | test | 2.5957 | 3.8067 | 4.1960 | 0.2003 | 0.1390 | 0.1685 |
| 16.0000 | B0 | B0 history-only | rolling_6_months | none | test | 2.7698 | 5.8661 | 18.4983 | 0.1034 | 0.1801 | 0.1461 |
| 17.0000 | B3 | B3 tabular feature parity | ridge | none | test | 3.1321 | 4.1664 | 5.0784 | 0.2177 | 0.1658 | 0.1798 |
| 18.0000 | B0 | B0 history-only | rolling_12_months | none | test | 7.1871 | 14.3187 | 16.1860 | 0.2440 | 0.1302 | 0.1573 |
| 19.0000 | B1 | B1 direct SoVI validation | B1_sovi_direct_validation | none | cumulative_static | — | — | — | 0.2534 | 0.4287 | 0.0000 |

## Graph-value checklist

| comparator_label | metric | direction | real_graph_value | comparator_value | improvement_absolute | improvement_relative | real_graph_beats_comparator | available |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| B3 tabular feature parity | mae | lower_is_better | 1.2912 | 2.0695 | 0.7783 | 0.3761 | 1.0000 | 1.0000 |
| B3 tabular feature parity | rmse | lower_is_better | 2.2937 | 2.8982 | 0.6045 | 0.2086 | 1.0000 | 1.0000 |
| B3 tabular feature parity | spearman | higher_is_better | 0.2741 | 0.2272 | 0.0469 | 0.2063 | 1.0000 | 1.0000 |
| B3 tabular feature parity | ndcg_at_25 | higher_is_better | 0.2383 | 0.1575 | 0.0808 | 0.5133 | 1.0000 | 1.0000 |
| B4 no-edge neural | mae | lower_is_better | 1.2912 | 1.3271 | 0.0359 | 0.0271 | 1.0000 | 1.0000 |
| B4 no-edge neural | rmse | lower_is_better | 2.2937 | 2.3354 | 0.0417 | 0.0178 | 1.0000 | 1.0000 |
| B4 no-edge neural | spearman | higher_is_better | 0.2741 | -0.0625 | 0.3365 | 5.3870 | 1.0000 | 1.0000 |
| B4 no-edge neural | ndcg_at_25 | higher_is_better | 0.2383 | 0.1080 | 0.1304 | 1.2073 | 1.0000 | 1.0000 |
| B4 random/placebo graph | mae | lower_is_better | 1.2912 | 1.2913 | 0.0001 | 0.0001 | 1.0000 | 1.0000 |
| B4 random/placebo graph | rmse | lower_is_better | 2.2937 | 2.2995 | 0.0058 | 0.0025 | 1.0000 | 1.0000 |
| B4 random/placebo graph | spearman | higher_is_better | 0.2741 | 0.2452 | 0.0289 | 0.1178 | 1.0000 | 1.0000 |
| B4 random/placebo graph | ndcg_at_25 | higher_is_better | 0.2383 | 0.2012 | 0.0371 | 0.1845 | 1.0000 | 1.0000 |
| B4 kNN graph | mae | lower_is_better | 1.2912 | 1.2889 | -0.0023 | -0.0018 | 0.0000 | 1.0000 |
| B4 kNN graph | rmse | lower_is_better | 2.2937 | 2.2958 | 0.0021 | 0.0009 | 1.0000 | 1.0000 |
| B4 kNN graph | spearman | higher_is_better | 0.2741 | 0.3007 | -0.0266 | -0.0884 | 0.0000 | 1.0000 |
| B4 kNN graph | ndcg_at_25 | higher_is_better | 0.2383 | 0.2880 | -0.0497 | -0.1726 | 0.0000 | 1.0000 |

## Metric winners

| split | metric | direction | winner_display_name | winner_model_name | winner_graph_kind | winner_value | runner_up_value | winner_margin | n_methods_compared |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| all | kendall_tau | higher_is_better | B3 tabular feature parity | hist_gradient_boosting | none | 0.3350 | 0.2738 | 0.0611 | 18.0000 |
| all | mae | lower_is_better | B4 real CD adjacency graph | B4_real_cd_graph | adjacency | 1.7481 | 1.7492 | 0.0011 | 18.0000 |
| all | mean_poisson_deviance | lower_is_better | B3 tabular feature parity | hist_gradient_boosting | none | 3.1784 | 3.4435 | 0.2651 | 18.0000 |
| all | ndcg_at_10 | higher_is_better | B3 tabular feature parity | hist_gradient_boosting | none | 0.3964 | 0.2809 | 0.1156 | 18.0000 |
| all | ndcg_at_100 | higher_is_better | B3 tabular feature parity | hist_gradient_boosting | none | 0.3829 | 0.3209 | 0.0620 | 18.0000 |
| all | ndcg_at_25 | higher_is_better | B3 tabular feature parity | hist_gradient_boosting | none | 0.3927 | 0.3100 | 0.0827 | 18.0000 |
| all | rmse | lower_is_better | B3 tabular feature parity | hist_gradient_boosting | none | 3.9069 | 3.9877 | 0.0809 | 18.0000 |
| all | spearman | higher_is_better | B3 tabular feature parity | hist_gradient_boosting | none | 0.4343 | 0.3608 | 0.0736 | 18.0000 |
| all | top10_overlap_rate | higher_is_better | B3 tabular feature parity | hist_gradient_boosting | none | 0.4347 | 0.3596 | 0.0751 | 18.0000 |
| all | top25_overlap_rate | higher_is_better | B3 tabular feature parity | hist_gradient_boosting | none | 0.5025 | 0.4524 | 0.0501 | 18.0000 |
| cumulative_static | kendall_tau | higher_is_better | B1 direct SoVI validation | B1_sovi_direct_validation | none | 0.2772 | 0.1766 | 0.1006 | 22.0000 |
| cumulative_static | ndcg_at_10 | higher_is_better | B1 direct SoVI validation | B1_sovi_direct_validation | none | 0.3818 | 0.3505 | 0.0313 | 22.0000 |
| cumulative_static | ndcg_at_25 | higher_is_better | B1 direct SoVI validation | B1_sovi_direct_validation | none | 0.4823 | 0.4480 | 0.0343 | 22.0000 |
| cumulative_static | spearman | higher_is_better | B1 direct SoVI validation | B1_sovi_direct_validation | none | 0.3770 | 0.2579 | 0.1191 | 22.0000 |
| cumulative_static | top10_overlap_rate | higher_is_better | B1 direct SoVI validation | B1_sovi_direct_validation | none | 0.2000 | 0.1000 | 0.1000 | 22.0000 |
| cumulative_static | top25_overlap_rate | higher_is_better | B1 direct SoVI validation | B1_sovi_direct_validation | none | 0.4000 | 0.4000 | 0.0000 | 22.0000 |
| test | kendall_tau | higher_is_better | B0 history-only | seasonal_historical_mean | none | 0.2232 | 0.2229 | 0.0004 | 18.0000 |
| test | mae | lower_is_better | B4 kNN graph | B4_knn_graph | knn | 1.2889 | 1.2912 | 0.0023 | 18.0000 |
| test | mean_poisson_deviance | lower_is_better | B2 calibrated SoVI | poisson_sovi | none | 2.7744 | 2.7789 | 0.0045 | 18.0000 |
| test | ndcg_at_10 | higher_is_better | B4 kNN graph | B4_knn_graph | knn | 0.3667 | 0.2665 | 0.1003 | 18.0000 |
| test | ndcg_at_100 | higher_is_better | B4 kNN graph | B4_knn_graph | knn | 0.3020 | 0.2971 | 0.0049 | 18.0000 |
| test | ndcg_at_25 | higher_is_better | B4 kNN graph | B4_knn_graph | knn | 0.2880 | 0.2383 | 0.0497 | 18.0000 |
| test | rmse | lower_is_better | B4 real CD adjacency graph | B4_real_cd_graph | adjacency | 2.2937 | 2.2958 | 0.0021 | 18.0000 |
| test | spearman | higher_is_better | B4 kNN graph | B4_knn_graph | knn | 0.3007 | 0.2878 | 0.0128 | 18.0000 |
| test | top10_overlap_rate | higher_is_better | B3 tabular feature parity | ridge | none | 0.1798 | 0.1685 | 0.0112 | 18.0000 |
| test | top25_overlap_rate | higher_is_better | B0 history-only | seasonal_historical_mean | none | 0.4253 | 0.4118 | 0.0136 | 18.0000 |
| validation | kendall_tau | higher_is_better | B3 tabular feature parity | random_forest | none | 0.2541 | 0.2429 | 0.0112 | 18.0000 |
| validation | mae | lower_is_better | B4 kNN graph | B4_knn_graph | knn | 2.2492 | 2.2540 | 0.0048 | 18.0000 |
| validation | mean_poisson_deviance | lower_is_better | B3 tabular feature parity | random_forest | none | 6.2572 | 6.8078 | 0.5506 | 18.0000 |
| validation | ndcg_at_10 | higher_is_better | B4 real CD adjacency graph | B4_real_cd_graph | adjacency | 0.1418 | 0.1290 | 0.0128 | 18.0000 |
| validation | ndcg_at_100 | higher_is_better | B3 tabular feature parity | random_forest | none | 0.2314 | 0.2157 | 0.0157 | 18.0000 |
| validation | ndcg_at_25 | higher_is_better | B4 real CD adjacency graph | B4_real_cd_graph | adjacency | 0.1551 | 0.1276 | 0.0275 | 18.0000 |
| validation | rmse | lower_is_better | B3 tabular feature parity | random_forest | none | 6.8530 | 6.9956 | 0.1426 | 18.0000 |
| validation | spearman | higher_is_better | B3 tabular feature parity | random_forest | none | 0.3342 | 0.3133 | 0.0209 | 18.0000 |
| validation | top10_overlap_rate | higher_is_better | B4 kNN graph | B4_knn_graph | knn | 0.2797 | 0.2627 | 0.0169 | 18.0000 |
| validation | top25_overlap_rate | higher_is_better | B3 tabular feature parity | random_forest | none | 0.4320 | 0.4286 | 0.0034 | 18.0000 |

## Interpretation guide

Read the benchmark in layers:

- **B1 → B2** tests whether raw/static SoVI becomes operationally useful after calibration.
- **B0** tests how much simple temporal history already explains.
- **B3** is the key non-graph feature-parity baseline. It is the hardest non-graph benchmark.
- **B4_no_edge_neural** tests whether neural capacity alone helps.
- **B4_random_edge_graph** tests whether arbitrary graph smoothing helps; the real graph should beat this.
- **B4_knn_graph** tests whether generic spatial proximity is enough.
- **B4_real_cd_graph** supports a topology-value claim only if it improves over the controls above.

## Missing or skipped inputs

| baseline_family | status | path | rows |
| --- | --- | --- | --- |
| B1_sovi_direct_validation | loaded_metrics_file | urban_graph_benchmark/outputs/qc_cd_civil_security_sovi_v0/baselines/B1_sovi_direct_validation/sovi_civil_security_validation_metrics.csv | 22.0000 |
| B0_history_only | loaded_predictions_recomputed_metrics | urban_graph_benchmark/outputs/qc_cd_civil_security_sovi_v0/baselines/B0_history_only/predictions.parquet | 20.0000 |
| B2_calibrated_sovi | loaded_predictions_recomputed_metrics | urban_graph_benchmark/outputs/qc_cd_civil_security_sovi_v0/baselines/B2_calibrated_sovi/predictions.parquet | 24.0000 |
| B3_tabular_feature_parity | loaded_predictions_recomputed_metrics | urban_graph_benchmark/outputs/qc_cd_civil_security_sovi_v0/baselines/B3_tabular_feature_parity/predictions.parquet | 12.0000 |
| B4_no_edge_neural | loaded_predictions_recomputed_metrics | urban_graph_benchmark/outputs/qc_cd_civil_security_sovi_v0/baselines/B4_no_edge_neural/predictions.parquet | 4.0000 |
| B4_random_edge_graph | loaded_predictions_recomputed_metrics | urban_graph_benchmark/outputs/qc_cd_civil_security_sovi_v0/baselines/B4_random_edge_graph/predictions.parquet | 4.0000 |
| B4_knn_graph | loaded_predictions_recomputed_metrics | urban_graph_benchmark/outputs/qc_cd_civil_security_sovi_v0/baselines/B4_knn_graph/predictions.parquet | 4.0000 |
| B4_real_cd_graph | loaded_predictions_recomputed_metrics | urban_graph_benchmark/outputs/qc_cd_civil_security_sovi_v0/baselines/B4_real_cd_graph/predictions.parquet | 4.0000 |

## Output files

- `benchmark_comparison.csv`: wide comparison table, one row per method/model/split where available.
- `benchmark_comparison_compact.csv`: primary comparison rows used for report ranking.
- `metrics_long.csv`: long-form metric table.
- `metric_winners.csv`: winner by metric and split.
- `qc_cd_civil_security_sovi_benchmark_report.md`: this report.

## Reproducibility notes

- Baselines directory: `urban_graph_benchmark/outputs/qc_cd_civil_security_sovi_v0/baselines`
- Comparisons directory: `urban_graph_benchmark/outputs/qc_cd_civil_security_sovi_v0/comparisons`
- Reports directory: `urban_graph_benchmark/outputs/qc_cd_civil_security_sovi_v0/reports`
- Primary split for compact ranking: `test`
- Primary metric for compact ranking: `mae`
