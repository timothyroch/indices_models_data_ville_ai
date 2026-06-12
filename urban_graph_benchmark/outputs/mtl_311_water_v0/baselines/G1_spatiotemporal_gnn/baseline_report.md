# G1 Spatiotemporal Tract GNN Baseline

Generated at: `2026-06-12T13:52:51.521543+00:00`

Graph artifact directory: `urban_graph_benchmark/outputs/mtl_311_water_v0/graphs/G1_tract_month_graph`

Output directory: `urban_graph_benchmark/outputs/mtl_311_water_v0/baselines/G1_spatiotemporal_gnn`

## Purpose

G1 is the first controlled graph-value test after the frozen A0--A3 tabular benchmark layer. It evaluates whether message passing over tract topology/time improves prediction and ranking of reported water/drainage 311 burden beyond feature-parity tabular baselines.

## Important interpretation rule

A graph result is meaningful only relative to A3 feature-parity baselines and to no-edge neural controls. Beating raw SVI or weak naive baselines is not enough after A3.

## Training configuration

```json
{
  "hidden_dim": 128,
  "num_layers": 2,
  "dropout": 0.15,
  "activation": "relu",
  "normalization": "layernorm",
  "residual": true,
  "backend": "manual",
  "relation_combine": "mean",
  "max_epochs": 20,
  "patience": 5,
  "learning_rate": 0.001,
  "weight_decay": 0.0001,
  "grad_clip_norm": 5.0,
  "min_delta": 1e-05,
  "seed": 20240610,
  "device": "auto",
  "count_min": 0.0,
  "monitor_metric": "validation_mae",
  "save_checkpoints": true,
  "save_embeddings": "none"
}
```

## Trial summary

```text
                                                                       model_name    status split_scheme   feature_regime      edge_regime edge_mask_regime  best_epoch  best_validation_mae  n_edges_used  n_relations_used  elapsed_seconds
        G1__temporal__lagged_reporting__no_edges__all_edges__h128_L2_seed20240610 completed     temporal lagged_reporting         no_edges        all_edges          15             2.660322             0                 0         2.098393
   G1__temporal__lagged_reporting__temporal_only__all_edges__h128_L2_seed20240610 completed     temporal lagged_reporting    temporal_only        all_edges          19             2.828268         50220                 2         0.644406
G1__temporal__lagged_reporting__spatial_temporal__all_edges__h128_L2_seed20240610 completed     temporal lagged_reporting spatial_temporal        all_edges          15             2.704427        279180                 3         1.166032
```

## Validation-only model selection

```text
                                                                       model_name split_scheme   feature_regime      edge_regime edge_mask_regime  validation_mae  validation_spearman  test_mae  test_spearman  selected_overall_for_split  selected_for_feature_regime  selected_for_test_summary
        G1__temporal__lagged_reporting__no_edges__all_edges__h128_L2_seed20240610     temporal lagged_reporting         no_edges        all_edges        2.660322             0.703769  2.766027       0.614693                        True                         True                       True
G1__temporal__lagged_reporting__spatial_temporal__all_edges__h128_L2_seed20240610     temporal lagged_reporting spatial_temporal        all_edges        2.704427             0.699489  2.834472       0.602422                       False                        False                      False
   G1__temporal__lagged_reporting__temporal_only__all_edges__h128_L2_seed20240610     temporal lagged_reporting    temporal_only        all_edges        2.828268             0.679727  2.814636       0.588745                       False                        False                      False
```

## Compact metrics for selected models

```text
          model_stage                                                                model_name split_scheme split_name   feature_regime edge_regime edge_mask_regime                     metric_name  metric_value  higher_is_better  n_rows     seed
G1_spatiotemporal_gnn G1__temporal__lagged_reporting__no_edges__all_edges__h128_L2_seed20240610     temporal       test lagged_reporting    no_edges        all_edges                      count__mae      2.766027             False    4860 20240610
G1_spatiotemporal_gnn G1__temporal__lagged_reporting__no_edges__all_edges__h128_L2_seed20240610     temporal       test lagged_reporting    no_edges        all_edges    count__mean_poisson_deviance      2.950863             False    4860 20240610
G1_spatiotemporal_gnn G1__temporal__lagged_reporting__no_edges__all_edges__h128_L2_seed20240610     temporal       test lagged_reporting    no_edges        all_edges                     count__rmse      3.985315             False    4860 20240610
G1_spatiotemporal_gnn G1__temporal__lagged_reporting__no_edges__all_edges__h128_L2_seed20240610     temporal       test lagged_reporting    no_edges        all_edges            ranking__ndcg_at_100      0.608128              True    4860 20240610
G1_spatiotemporal_gnn G1__temporal__lagged_reporting__no_edges__all_edges__h128_L2_seed20240610     temporal       test lagged_reporting    no_edges        all_edges          ranking__spearman_corr      0.614693              True    4860 20240610
G1_spatiotemporal_gnn G1__temporal__lagged_reporting__no_edges__all_edges__h128_L2_seed20240610     temporal       test lagged_reporting    no_edges        all_edges ranking__top_10pct_overlap_rate      0.397119              True    4860 20240610
G1_spatiotemporal_gnn G1__temporal__lagged_reporting__no_edges__all_edges__h128_L2_seed20240610     temporal      train lagged_reporting    no_edges        all_edges                      count__mae      2.662469             False   19440 20240610
G1_spatiotemporal_gnn G1__temporal__lagged_reporting__no_edges__all_edges__h128_L2_seed20240610     temporal      train lagged_reporting    no_edges        all_edges    count__mean_poisson_deviance      2.461787             False   19440 20240610
G1_spatiotemporal_gnn G1__temporal__lagged_reporting__no_edges__all_edges__h128_L2_seed20240610     temporal      train lagged_reporting    no_edges        all_edges                     count__rmse      4.002053             False   19440 20240610
G1_spatiotemporal_gnn G1__temporal__lagged_reporting__no_edges__all_edges__h128_L2_seed20240610     temporal      train lagged_reporting    no_edges        all_edges            ranking__ndcg_at_100      0.448612              True   19440 20240610
G1_spatiotemporal_gnn G1__temporal__lagged_reporting__no_edges__all_edges__h128_L2_seed20240610     temporal      train lagged_reporting    no_edges        all_edges          ranking__spearman_corr      0.688796              True   19440 20240610
G1_spatiotemporal_gnn G1__temporal__lagged_reporting__no_edges__all_edges__h128_L2_seed20240610     temporal      train lagged_reporting    no_edges        all_edges ranking__top_10pct_overlap_rate      0.484053              True   19440 20240610
G1_spatiotemporal_gnn G1__temporal__lagged_reporting__no_edges__all_edges__h128_L2_seed20240610     temporal validation lagged_reporting    no_edges        all_edges                      count__mae      2.660322             False    4320 20240610
G1_spatiotemporal_gnn G1__temporal__lagged_reporting__no_edges__all_edges__h128_L2_seed20240610     temporal validation lagged_reporting    no_edges        all_edges    count__mean_poisson_deviance      2.628910             False    4320 20240610
G1_spatiotemporal_gnn G1__temporal__lagged_reporting__no_edges__all_edges__h128_L2_seed20240610     temporal validation lagged_reporting    no_edges        all_edges                     count__rmse      4.071943             False    4320 20240610
G1_spatiotemporal_gnn G1__temporal__lagged_reporting__no_edges__all_edges__h128_L2_seed20240610     temporal validation lagged_reporting    no_edges        all_edges            ranking__ndcg_at_100      0.648490              True    4320 20240610
G1_spatiotemporal_gnn G1__temporal__lagged_reporting__no_edges__all_edges__h128_L2_seed20240610     temporal validation lagged_reporting    no_edges        all_edges          ranking__spearman_corr      0.703769              True    4320 20240610
G1_spatiotemporal_gnn G1__temporal__lagged_reporting__no_edges__all_edges__h128_L2_seed20240610     temporal validation lagged_reporting    no_edges        all_edges ranking__top_10pct_overlap_rate      0.467593              True    4320 20240610
```

## Graph-regime audit

Edge masks define the message-passing graph used by each trial. `all_edges` is transductive; `no_test_incident_edges` removes all edges touching test nodes; `train_train_edges` uses only train-train edges when requested.

```text
  feature_regime split_scheme      edge_regime edge_mask_regime  n_edges_total_graph  n_edges_used  n_edge_types_used                                                 edge_types_used  uses_temporal_edges  uses_spatial_edges  uses_placebo_edges
lagged_reporting     temporal         no_edges        all_edges               508140             0                  0                                                                                False               False               False
lagged_reporting     temporal    temporal_only        all_edges               508140         50220                  2                        temporal_self_lag_1,temporal_self_lag_12                 True               False               False
lagged_reporting     temporal spatial_temporal        all_edges               508140        279180                  3 spatial_knn_same_month,temporal_self_lag_1,temporal_self_lag_12                 True                True               False
```

## Feature preprocessing audit preview

Feature imputation and scaling are fit on train nodes only for each split/feature-regime combination.

```text
  feature_regime split_scheme                                                                  feature  train_median    train_mean  train_std
lagged_reporting     temporal              reporting_history__total_311_count_non_water_drainage_lag_1  4.200000e+01  4.814552e+01  45.325371
lagged_reporting     temporal              reporting_history__total_311_count_non_water_drainage_lag_2  4.200000e+01  4.822551e+01  44.937428
lagged_reporting     temporal              reporting_history__total_311_count_non_water_drainage_lag_3  4.200000e+01  4.811574e+01  44.398109
lagged_reporting     temporal             reporting_history__total_311_count_non_water_drainage_lag_12  4.100000e+01  4.556194e+01  38.070087
lagged_reporting     temporal  reporting_history__total_311_count_non_water_drainage_roll3_mean_shift1  4.266667e+01  4.803017e+01  43.495918
lagged_reporting     temporal  reporting_history__total_311_count_non_water_drainage_roll6_mean_shift1  4.266667e+01  4.756662e+01  42.713730
lagged_reporting     temporal reporting_history__total_311_count_non_water_drainage_roll12_mean_shift1  4.275000e+01  4.700904e+01  42.308849
lagged_reporting     temporal                                   requests_history__requests_total_lag_1  4.700000e+01  5.349805e+01  48.463203
lagged_reporting     temporal                                   requests_history__requests_total_lag_3  4.700000e+01  5.346903e+01  47.452221
lagged_reporting     temporal                                  requests_history__requests_total_lag_12  4.600000e+01  5.085334e+01  40.632614
lagged_reporting     temporal                       requests_history__requests_total_roll3_mean_shift1  4.766667e+01  5.338688e+01  46.569595
lagged_reporting     temporal                       requests_history__requests_total_roll6_mean_shift1  4.766667e+01  5.289816e+01  45.651058
lagged_reporting     temporal                      requests_history__requests_total_roll12_mean_shift1  4.783333e+01  5.235844e+01  45.133419
lagged_reporting     temporal                                                    calendar__month_is_02  0.000000e+00  8.333334e-02   0.276385
lagged_reporting     temporal                                                    calendar__month_is_03  0.000000e+00  8.333334e-02   0.276385
lagged_reporting     temporal                                                    calendar__month_is_04  0.000000e+00  8.333334e-02   0.276385
lagged_reporting     temporal                                                    calendar__month_is_05  0.000000e+00  8.333334e-02   0.276385
lagged_reporting     temporal                                                    calendar__month_is_06  0.000000e+00  8.333334e-02   0.276385
lagged_reporting     temporal                                                    calendar__month_is_07  0.000000e+00  8.333334e-02   0.276385
lagged_reporting     temporal                                                    calendar__month_is_08  0.000000e+00  8.333334e-02   0.276385
lagged_reporting     temporal                                                    calendar__month_is_09  0.000000e+00  8.333334e-02   0.276385
lagged_reporting     temporal                                                    calendar__month_is_10  0.000000e+00  8.333334e-02   0.276385
lagged_reporting     temporal                                                    calendar__month_is_11  0.000000e+00  8.333334e-02   0.276385
lagged_reporting     temporal                                                    calendar__month_is_12  0.000000e+00  8.333334e-02   0.276385
lagged_reporting     temporal                                                      calendar__month_sin -6.123234e-17 -7.812680e-18   0.707107
lagged_reporting     temporal                                                      calendar__month_cos -6.123234e-17 -7.310110e-18   0.707107
lagged_reporting     temporal                                       calendar__period_index_since_start  1.750000e+01  1.750000e+01  10.388294
lagged_reporting     temporal              reporting_history__total_311_count_non_water_drainage_lag_1  4.200000e+01  4.814552e+01  45.325371
lagged_reporting     temporal              reporting_history__total_311_count_non_water_drainage_lag_2  4.200000e+01  4.822551e+01  44.937428
lagged_reporting     temporal              reporting_history__total_311_count_non_water_drainage_lag_3  4.200000e+01  4.811574e+01  44.398109
lagged_reporting     temporal             reporting_history__total_311_count_non_water_drainage_lag_12  4.100000e+01  4.556194e+01  38.070087
lagged_reporting     temporal  reporting_history__total_311_count_non_water_drainage_roll3_mean_shift1  4.266667e+01  4.803017e+01  43.495918
lagged_reporting     temporal  reporting_history__total_311_count_non_water_drainage_roll6_mean_shift1  4.266667e+01  4.756662e+01  42.713730
lagged_reporting     temporal reporting_history__total_311_count_non_water_drainage_roll12_mean_shift1  4.275000e+01  4.700904e+01  42.308849
lagged_reporting     temporal                                   requests_history__requests_total_lag_1  4.700000e+01  5.349805e+01  48.463203
lagged_reporting     temporal                                   requests_history__requests_total_lag_3  4.700000e+01  5.346903e+01  47.452221
lagged_reporting     temporal                                  requests_history__requests_total_lag_12  4.600000e+01  5.085334e+01  40.632614
lagged_reporting     temporal                       requests_history__requests_total_roll3_mean_shift1  4.766667e+01  5.338688e+01  46.569595
lagged_reporting     temporal                       requests_history__requests_total_roll6_mean_shift1  4.766667e+01  5.289816e+01  45.651058
lagged_reporting     temporal                      requests_history__requests_total_roll12_mean_shift1  4.783333e+01  5.235844e+01  45.133419
lagged_reporting     temporal                                                    calendar__month_is_02  0.000000e+00  8.333334e-02   0.276385
lagged_reporting     temporal                                                    calendar__month_is_03  0.000000e+00  8.333334e-02   0.276385
lagged_reporting     temporal                                                    calendar__month_is_04  0.000000e+00  8.333334e-02   0.276385
lagged_reporting     temporal                                                    calendar__month_is_05  0.000000e+00  8.333334e-02   0.276385
lagged_reporting     temporal                                                    calendar__month_is_06  0.000000e+00  8.333334e-02   0.276385
lagged_reporting     temporal                                                    calendar__month_is_07  0.000000e+00  8.333334e-02   0.276385
lagged_reporting     temporal                                                    calendar__month_is_08  0.000000e+00  8.333334e-02   0.276385
lagged_reporting     temporal                                                    calendar__month_is_09  0.000000e+00  8.333334e-02   0.276385
lagged_reporting     temporal                                                    calendar__month_is_10  0.000000e+00  8.333334e-02   0.276385
lagged_reporting     temporal                                                    calendar__month_is_11  0.000000e+00  8.333334e-02   0.276385
lagged_reporting     temporal                                                    calendar__month_is_12  0.000000e+00  8.333334e-02   0.276385
lagged_reporting     temporal                                                      calendar__month_sin -6.123234e-17 -7.812680e-18   0.707107
lagged_reporting     temporal                                                      calendar__month_cos -6.123234e-17 -7.310110e-18   0.707107
lagged_reporting     temporal                                       calendar__period_index_since_start  1.750000e+01  1.750000e+01  10.388294
lagged_reporting     temporal              reporting_history__total_311_count_non_water_drainage_lag_1  4.200000e+01  4.814552e+01  45.325371
lagged_reporting     temporal              reporting_history__total_311_count_non_water_drainage_lag_2  4.200000e+01  4.822551e+01  44.937428
lagged_reporting     temporal              reporting_history__total_311_count_non_water_drainage_lag_3  4.200000e+01  4.811574e+01  44.398109
lagged_reporting     temporal             reporting_history__total_311_count_non_water_drainage_lag_12  4.100000e+01  4.556194e+01  38.070087
lagged_reporting     temporal  reporting_history__total_311_count_non_water_drainage_roll3_mean_shift1  4.266667e+01  4.803017e+01  43.495918
lagged_reporting     temporal  reporting_history__total_311_count_non_water_drainage_roll6_mean_shift1  4.266667e+01  4.756662e+01  42.713730
lagged_reporting     temporal reporting_history__total_311_count_non_water_drainage_roll12_mean_shift1  4.275000e+01  4.700904e+01  42.308849
lagged_reporting     temporal                                   requests_history__requests_total_lag_1  4.700000e+01  5.349805e+01  48.463203
lagged_reporting     temporal                                   requests_history__requests_total_lag_3  4.700000e+01  5.346903e+01  47.452221
lagged_reporting     temporal                                  requests_history__requests_total_lag_12  4.600000e+01  5.085334e+01  40.632614
lagged_reporting     temporal                       requests_history__requests_total_roll3_mean_shift1  4.766667e+01  5.338688e+01  46.569595
lagged_reporting     temporal                       requests_history__requests_total_roll6_mean_shift1  4.766667e+01  5.289816e+01  45.651058
lagged_reporting     temporal                      requests_history__requests_total_roll12_mean_shift1  4.783333e+01  5.235844e+01  45.133419
lagged_reporting     temporal                                                    calendar__month_is_02  0.000000e+00  8.333334e-02   0.276385
lagged_reporting     temporal                                                    calendar__month_is_03  0.000000e+00  8.333334e-02   0.276385
lagged_reporting     temporal                                                    calendar__month_is_04  0.000000e+00  8.333334e-02   0.276385
lagged_reporting     temporal                                                    calendar__month_is_05  0.000000e+00  8.333334e-02   0.276385
lagged_reporting     temporal                                                    calendar__month_is_06  0.000000e+00  8.333334e-02   0.276385
lagged_reporting     temporal                                                    calendar__month_is_07  0.000000e+00  8.333334e-02   0.276385
lagged_reporting     temporal                                                    calendar__month_is_08  0.000000e+00  8.333334e-02   0.276385
lagged_reporting     temporal                                                    calendar__month_is_09  0.000000e+00  8.333334e-02   0.276385
lagged_reporting     temporal                                                    calendar__month_is_10  0.000000e+00  8.333334e-02   0.276385
lagged_reporting     temporal                                                    calendar__month_is_11  0.000000e+00  8.333334e-02   0.276385
lagged_reporting     temporal                                                    calendar__month_is_12  0.000000e+00  8.333334e-02   0.276385
lagged_reporting     temporal                                                      calendar__month_sin -6.123234e-17 -7.812680e-18   0.707107
lagged_reporting     temporal                                                      calendar__month_cos -6.123234e-17 -7.310110e-18   0.707107
```

## Output artifacts

| Artifact | Path |
|---|---|
| `metrics` | `urban_graph_benchmark/outputs/mtl_311_water_v0/baselines/G1_spatiotemporal_gnn/metrics.csv` |
| `predictions_validation` | `urban_graph_benchmark/outputs/mtl_311_water_v0/baselines/G1_spatiotemporal_gnn/predictions_validation.parquet` |
| `predictions_test` | `urban_graph_benchmark/outputs/mtl_311_water_v0/baselines/G1_spatiotemporal_gnn/predictions_test.parquet` |
| `predictions_all_evaluated` | `urban_graph_benchmark/outputs/mtl_311_water_v0/baselines/G1_spatiotemporal_gnn/predictions_all_evaluated.parquet` |
| `training_curves` | `urban_graph_benchmark/outputs/mtl_311_water_v0/baselines/G1_spatiotemporal_gnn/training_curves.csv` |
| `trial_audit` | `urban_graph_benchmark/outputs/mtl_311_water_v0/baselines/G1_spatiotemporal_gnn/trial_audit.csv` |
| `model_selection_audit` | `urban_graph_benchmark/outputs/mtl_311_water_v0/baselines/G1_spatiotemporal_gnn/model_selection_audit.csv` |
| `feature_preprocessing_audit` | `urban_graph_benchmark/outputs/mtl_311_water_v0/baselines/G1_spatiotemporal_gnn/feature_preprocessing_audit.csv` |
| `graph_regime_audit` | `urban_graph_benchmark/outputs/mtl_311_water_v0/baselines/G1_spatiotemporal_gnn/graph_regime_audit.csv` |
| `model_metadata` | `urban_graph_benchmark/outputs/mtl_311_water_v0/baselines/G1_spatiotemporal_gnn/model_metadata.json` |
| `baseline_report` | `urban_graph_benchmark/outputs/mtl_311_water_v0/baselines/G1_spatiotemporal_gnn/baseline_report.md` |
| `checkpoints_dir` | `urban_graph_benchmark/outputs/mtl_311_water_v0/baselines/G1_spatiotemporal_gnn/checkpoints` |
| `embeddings_dir` | `urban_graph_benchmark/outputs/mtl_311_water_v0/baselines/G1_spatiotemporal_gnn/embeddings` |

## Benchmark handoff

Temporal graph results should be compared against A0_3_tract_train_mean and A3_all_forecasting RF. Spatial-block graph results should be compared against A3_lagged_reporting_forecasting HGB. The most convincing graph claim would improve both count error and high-burden ranking metrics under spatial-block evaluation.
