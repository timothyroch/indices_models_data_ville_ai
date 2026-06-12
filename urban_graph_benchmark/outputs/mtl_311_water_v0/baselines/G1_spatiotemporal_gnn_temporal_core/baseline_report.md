# G1 Spatiotemporal Tract GNN Baseline

Generated at: `2026-06-12T13:54:46.891549+00:00`

Graph artifact directory: `urban_graph_benchmark/outputs/mtl_311_water_v0/graphs/G1_tract_month_graph`

Output directory: `urban_graph_benchmark/outputs/mtl_311_water_v0/baselines/G1_spatiotemporal_gnn_temporal_core`

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
  "max_epochs": 250,
  "patience": 30,
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
                                                                              model_name    status split_scheme    feature_regime            edge_regime edge_mask_regime  best_epoch  best_validation_mae  n_edges_used  n_relations_used  elapsed_seconds
                G1__temporal__all_forecasting__no_edges__all_edges__h128_L2_seed20240610 completed     temporal   all_forecasting               no_edges        all_edges          32             2.515681             0                 0         1.309667
           G1__temporal__all_forecasting__temporal_only__all_edges__h128_L2_seed20240610 completed     temporal   all_forecasting          temporal_only        all_edges          39             2.488583         50220                 2         1.472706
            G1__temporal__all_forecasting__spatial_only__all_edges__h128_L2_seed20240610 completed     temporal   all_forecasting           spatial_only        all_edges          20             2.608148        228960                 1         1.832340
        G1__temporal__all_forecasting__spatial_temporal__all_edges__h128_L2_seed20240610 completed     temporal   all_forecasting       spatial_temporal        all_edges          15             2.620335        279180                 3         2.162316
  G1__temporal__all_forecasting__random_spatial_placebo__all_edges__h128_L2_seed20240610 completed     temporal   all_forecasting random_spatial_placebo        all_edges          15             2.611212        279180                 3         2.201660
               G1__temporal__lagged_reporting__no_edges__all_edges__h128_L2_seed20240610 completed     temporal  lagged_reporting               no_edges        all_edges          15             2.660322             0                 0         0.289570
          G1__temporal__lagged_reporting__temporal_only__all_edges__h128_L2_seed20240610 completed     temporal  lagged_reporting          temporal_only        all_edges          38             2.744961         50220                 2         1.394554
           G1__temporal__lagged_reporting__spatial_only__all_edges__h128_L2_seed20240610 completed     temporal  lagged_reporting           spatial_only        all_edges          35             2.740415        228960                 1         2.322820
       G1__temporal__lagged_reporting__spatial_temporal__all_edges__h128_L2_seed20240610 completed     temporal  lagged_reporting       spatial_temporal        all_edges          15             2.704445        279180                 3         2.135255
 G1__temporal__lagged_reporting__random_spatial_placebo__all_edges__h128_L2_seed20240610 completed     temporal  lagged_reporting random_spatial_placebo        all_edges          20             2.708284        279180                 3         2.412874
              G1__temporal__no_target_history__no_edges__all_edges__h128_L2_seed20240610 completed     temporal no_target_history               no_edges        all_edges          32             2.679423             0                 0         0.388952
         G1__temporal__no_target_history__temporal_only__all_edges__h128_L2_seed20240610 completed     temporal no_target_history          temporal_only        all_edges          37             2.689674         50220                 2         1.374894
          G1__temporal__no_target_history__spatial_only__all_edges__h128_L2_seed20240610 completed     temporal no_target_history           spatial_only        all_edges          47             2.640088        228960                 1         2.714197
      G1__temporal__no_target_history__spatial_temporal__all_edges__h128_L2_seed20240610 completed     temporal no_target_history       spatial_temporal        all_edges          43             2.737175        279180                 3         3.378419
G1__temporal__no_target_history__random_spatial_placebo__all_edges__h128_L2_seed20240610 completed     temporal no_target_history random_spatial_placebo        all_edges          34             2.735814        279180                 3         3.037041
```

## Validation-only model selection

```text
                                                                              model_name split_scheme    feature_regime            edge_regime edge_mask_regime  validation_mae  validation_spearman  test_mae  test_spearman  selected_overall_for_split  selected_for_feature_regime  selected_for_test_summary
           G1__temporal__all_forecasting__temporal_only__all_edges__h128_L2_seed20240610     temporal   all_forecasting          temporal_only        all_edges        2.488583             0.738506  2.545413       0.679158                        True                         True                       True
                G1__temporal__all_forecasting__no_edges__all_edges__h128_L2_seed20240610     temporal   all_forecasting               no_edges        all_edges        2.515681             0.728339  2.492699       0.688743                       False                        False                      False
            G1__temporal__all_forecasting__spatial_only__all_edges__h128_L2_seed20240610     temporal   all_forecasting           spatial_only        all_edges        2.608148             0.714735  2.628518       0.653188                       False                        False                      False
  G1__temporal__all_forecasting__random_spatial_placebo__all_edges__h128_L2_seed20240610     temporal   all_forecasting random_spatial_placebo        all_edges        2.611212             0.720368  2.662570       0.651317                       False                        False                      False
        G1__temporal__all_forecasting__spatial_temporal__all_edges__h128_L2_seed20240610     temporal   all_forecasting       spatial_temporal        all_edges        2.620335             0.718785  2.666741       0.650926                       False                        False                      False
          G1__temporal__no_target_history__spatial_only__all_edges__h128_L2_seed20240610     temporal no_target_history           spatial_only        all_edges        2.640088             0.717949  2.757825       0.628680                       False                         True                       True
               G1__temporal__lagged_reporting__no_edges__all_edges__h128_L2_seed20240610     temporal  lagged_reporting               no_edges        all_edges        2.660322             0.703769  2.766027       0.614693                       False                         True                       True
              G1__temporal__no_target_history__no_edges__all_edges__h128_L2_seed20240610     temporal no_target_history               no_edges        all_edges        2.679422             0.703707  2.669598       0.637856                       False                        False                      False
         G1__temporal__no_target_history__temporal_only__all_edges__h128_L2_seed20240610     temporal no_target_history          temporal_only        all_edges        2.689674             0.712676  2.749322       0.617184                       False                        False                      False
       G1__temporal__lagged_reporting__spatial_temporal__all_edges__h128_L2_seed20240610     temporal  lagged_reporting       spatial_temporal        all_edges        2.704445             0.699489  2.834482       0.602419                       False                        False                      False
 G1__temporal__lagged_reporting__random_spatial_placebo__all_edges__h128_L2_seed20240610     temporal  lagged_reporting random_spatial_placebo        all_edges        2.708285             0.704988  2.863012       0.605736                       False                        False                      False
G1__temporal__no_target_history__random_spatial_placebo__all_edges__h128_L2_seed20240610     temporal no_target_history random_spatial_placebo        all_edges        2.735814             0.701334  2.788870       0.607679                       False                        False                      False
      G1__temporal__no_target_history__spatial_temporal__all_edges__h128_L2_seed20240610     temporal no_target_history       spatial_temporal        all_edges        2.737175             0.704686  2.772380       0.622025                       False                        False                      False
           G1__temporal__lagged_reporting__spatial_only__all_edges__h128_L2_seed20240610     temporal  lagged_reporting           spatial_only        all_edges        2.740415             0.706784  2.915464       0.603356                       False                        False                      False
          G1__temporal__lagged_reporting__temporal_only__all_edges__h128_L2_seed20240610     temporal  lagged_reporting          temporal_only        all_edges        2.744961             0.698837  2.706240       0.624730                       False                        False                      False
```

## Compact metrics for selected models

```text
          model_stage                                                                     model_name split_scheme split_name    feature_regime   edge_regime edge_mask_regime                     metric_name  metric_value  higher_is_better  n_rows     seed
G1_spatiotemporal_gnn  G1__temporal__all_forecasting__temporal_only__all_edges__h128_L2_seed20240610     temporal       test   all_forecasting temporal_only        all_edges                      count__mae      2.545413             False    4860 20240610
G1_spatiotemporal_gnn  G1__temporal__all_forecasting__temporal_only__all_edges__h128_L2_seed20240610     temporal       test   all_forecasting temporal_only        all_edges    count__mean_poisson_deviance      2.596564             False    4860 20240610
G1_spatiotemporal_gnn  G1__temporal__all_forecasting__temporal_only__all_edges__h128_L2_seed20240610     temporal       test   all_forecasting temporal_only        all_edges                     count__rmse      3.744627             False    4860 20240610
G1_spatiotemporal_gnn  G1__temporal__all_forecasting__temporal_only__all_edges__h128_L2_seed20240610     temporal       test   all_forecasting temporal_only        all_edges            ranking__ndcg_at_100      0.564742              True    4860 20240610
G1_spatiotemporal_gnn  G1__temporal__all_forecasting__temporal_only__all_edges__h128_L2_seed20240610     temporal       test   all_forecasting temporal_only        all_edges          ranking__spearman_corr      0.679158              True    4860 20240610
G1_spatiotemporal_gnn  G1__temporal__all_forecasting__temporal_only__all_edges__h128_L2_seed20240610     temporal       test   all_forecasting temporal_only        all_edges ranking__top_10pct_overlap_rate      0.483539              True    4860 20240610
G1_spatiotemporal_gnn  G1__temporal__all_forecasting__temporal_only__all_edges__h128_L2_seed20240610     temporal      train   all_forecasting temporal_only        all_edges                      count__mae      2.421843             False   19440 20240610
G1_spatiotemporal_gnn  G1__temporal__all_forecasting__temporal_only__all_edges__h128_L2_seed20240610     temporal      train   all_forecasting temporal_only        all_edges    count__mean_poisson_deviance      2.285099             False   19440 20240610
G1_spatiotemporal_gnn  G1__temporal__all_forecasting__temporal_only__all_edges__h128_L2_seed20240610     temporal      train   all_forecasting temporal_only        all_edges                     count__rmse      3.853960             False   19440 20240610
G1_spatiotemporal_gnn  G1__temporal__all_forecasting__temporal_only__all_edges__h128_L2_seed20240610     temporal      train   all_forecasting temporal_only        all_edges            ranking__ndcg_at_100      0.514769              True   19440 20240610
G1_spatiotemporal_gnn  G1__temporal__all_forecasting__temporal_only__all_edges__h128_L2_seed20240610     temporal      train   all_forecasting temporal_only        all_edges          ranking__spearman_corr      0.745855              True   19440 20240610
G1_spatiotemporal_gnn  G1__temporal__all_forecasting__temporal_only__all_edges__h128_L2_seed20240610     temporal      train   all_forecasting temporal_only        all_edges ranking__top_10pct_overlap_rate      0.542695              True   19440 20240610
G1_spatiotemporal_gnn  G1__temporal__all_forecasting__temporal_only__all_edges__h128_L2_seed20240610     temporal validation   all_forecasting temporal_only        all_edges                      count__mae      2.488583             False    4320 20240610
G1_spatiotemporal_gnn  G1__temporal__all_forecasting__temporal_only__all_edges__h128_L2_seed20240610     temporal validation   all_forecasting temporal_only        all_edges    count__mean_poisson_deviance      2.423168             False    4320 20240610
G1_spatiotemporal_gnn  G1__temporal__all_forecasting__temporal_only__all_edges__h128_L2_seed20240610     temporal validation   all_forecasting temporal_only        all_edges                     count__rmse      3.904949             False    4320 20240610
G1_spatiotemporal_gnn  G1__temporal__all_forecasting__temporal_only__all_edges__h128_L2_seed20240610     temporal validation   all_forecasting temporal_only        all_edges            ranking__ndcg_at_100      0.654797              True    4320 20240610
G1_spatiotemporal_gnn  G1__temporal__all_forecasting__temporal_only__all_edges__h128_L2_seed20240610     temporal validation   all_forecasting temporal_only        all_edges          ranking__spearman_corr      0.738506              True    4320 20240610
G1_spatiotemporal_gnn  G1__temporal__all_forecasting__temporal_only__all_edges__h128_L2_seed20240610     temporal validation   all_forecasting temporal_only        all_edges ranking__top_10pct_overlap_rate      0.550926              True    4320 20240610
G1_spatiotemporal_gnn      G1__temporal__lagged_reporting__no_edges__all_edges__h128_L2_seed20240610     temporal       test  lagged_reporting      no_edges        all_edges                      count__mae      2.766027             False    4860 20240610
G1_spatiotemporal_gnn      G1__temporal__lagged_reporting__no_edges__all_edges__h128_L2_seed20240610     temporal       test  lagged_reporting      no_edges        all_edges    count__mean_poisson_deviance      2.950863             False    4860 20240610
G1_spatiotemporal_gnn      G1__temporal__lagged_reporting__no_edges__all_edges__h128_L2_seed20240610     temporal       test  lagged_reporting      no_edges        all_edges                     count__rmse      3.985315             False    4860 20240610
G1_spatiotemporal_gnn      G1__temporal__lagged_reporting__no_edges__all_edges__h128_L2_seed20240610     temporal       test  lagged_reporting      no_edges        all_edges            ranking__ndcg_at_100      0.608128              True    4860 20240610
G1_spatiotemporal_gnn      G1__temporal__lagged_reporting__no_edges__all_edges__h128_L2_seed20240610     temporal       test  lagged_reporting      no_edges        all_edges          ranking__spearman_corr      0.614693              True    4860 20240610
G1_spatiotemporal_gnn      G1__temporal__lagged_reporting__no_edges__all_edges__h128_L2_seed20240610     temporal       test  lagged_reporting      no_edges        all_edges ranking__top_10pct_overlap_rate      0.397119              True    4860 20240610
G1_spatiotemporal_gnn      G1__temporal__lagged_reporting__no_edges__all_edges__h128_L2_seed20240610     temporal      train  lagged_reporting      no_edges        all_edges                      count__mae      2.662469             False   19440 20240610
G1_spatiotemporal_gnn      G1__temporal__lagged_reporting__no_edges__all_edges__h128_L2_seed20240610     temporal      train  lagged_reporting      no_edges        all_edges    count__mean_poisson_deviance      2.461787             False   19440 20240610
G1_spatiotemporal_gnn      G1__temporal__lagged_reporting__no_edges__all_edges__h128_L2_seed20240610     temporal      train  lagged_reporting      no_edges        all_edges                     count__rmse      4.002053             False   19440 20240610
G1_spatiotemporal_gnn      G1__temporal__lagged_reporting__no_edges__all_edges__h128_L2_seed20240610     temporal      train  lagged_reporting      no_edges        all_edges            ranking__ndcg_at_100      0.448612              True   19440 20240610
G1_spatiotemporal_gnn      G1__temporal__lagged_reporting__no_edges__all_edges__h128_L2_seed20240610     temporal      train  lagged_reporting      no_edges        all_edges          ranking__spearman_corr      0.688796              True   19440 20240610
G1_spatiotemporal_gnn      G1__temporal__lagged_reporting__no_edges__all_edges__h128_L2_seed20240610     temporal      train  lagged_reporting      no_edges        all_edges ranking__top_10pct_overlap_rate      0.484053              True   19440 20240610
G1_spatiotemporal_gnn      G1__temporal__lagged_reporting__no_edges__all_edges__h128_L2_seed20240610     temporal validation  lagged_reporting      no_edges        all_edges                      count__mae      2.660322             False    4320 20240610
G1_spatiotemporal_gnn      G1__temporal__lagged_reporting__no_edges__all_edges__h128_L2_seed20240610     temporal validation  lagged_reporting      no_edges        all_edges    count__mean_poisson_deviance      2.628910             False    4320 20240610
G1_spatiotemporal_gnn      G1__temporal__lagged_reporting__no_edges__all_edges__h128_L2_seed20240610     temporal validation  lagged_reporting      no_edges        all_edges                     count__rmse      4.071943             False    4320 20240610
G1_spatiotemporal_gnn      G1__temporal__lagged_reporting__no_edges__all_edges__h128_L2_seed20240610     temporal validation  lagged_reporting      no_edges        all_edges            ranking__ndcg_at_100      0.648490              True    4320 20240610
G1_spatiotemporal_gnn      G1__temporal__lagged_reporting__no_edges__all_edges__h128_L2_seed20240610     temporal validation  lagged_reporting      no_edges        all_edges          ranking__spearman_corr      0.703769              True    4320 20240610
G1_spatiotemporal_gnn      G1__temporal__lagged_reporting__no_edges__all_edges__h128_L2_seed20240610     temporal validation  lagged_reporting      no_edges        all_edges ranking__top_10pct_overlap_rate      0.467593              True    4320 20240610
G1_spatiotemporal_gnn G1__temporal__no_target_history__spatial_only__all_edges__h128_L2_seed20240610     temporal       test no_target_history  spatial_only        all_edges                      count__mae      2.757825             False    4860 20240610
G1_spatiotemporal_gnn G1__temporal__no_target_history__spatial_only__all_edges__h128_L2_seed20240610     temporal       test no_target_history  spatial_only        all_edges    count__mean_poisson_deviance      3.037779             False    4860 20240610
G1_spatiotemporal_gnn G1__temporal__no_target_history__spatial_only__all_edges__h128_L2_seed20240610     temporal       test no_target_history  spatial_only        all_edges                     count__rmse      4.065525             False    4860 20240610
G1_spatiotemporal_gnn G1__temporal__no_target_history__spatial_only__all_edges__h128_L2_seed20240610     temporal       test no_target_history  spatial_only        all_edges            ranking__ndcg_at_100      0.511270              True    4860 20240610
G1_spatiotemporal_gnn G1__temporal__no_target_history__spatial_only__all_edges__h128_L2_seed20240610     temporal       test no_target_history  spatial_only        all_edges          ranking__spearman_corr      0.628680              True    4860 20240610
G1_spatiotemporal_gnn G1__temporal__no_target_history__spatial_only__all_edges__h128_L2_seed20240610     temporal       test no_target_history  spatial_only        all_edges ranking__top_10pct_overlap_rate      0.425926              True    4860 20240610
G1_spatiotemporal_gnn G1__temporal__no_target_history__spatial_only__all_edges__h128_L2_seed20240610     temporal      train no_target_history  spatial_only        all_edges                      count__mae      2.621749             False   19440 20240610
G1_spatiotemporal_gnn G1__temporal__no_target_history__spatial_only__all_edges__h128_L2_seed20240610     temporal      train no_target_history  spatial_only        all_edges    count__mean_poisson_deviance      2.745643             False   19440 20240610
G1_spatiotemporal_gnn G1__temporal__no_target_history__spatial_only__all_edges__h128_L2_seed20240610     temporal      train no_target_history  spatial_only        all_edges                     count__rmse      4.204673             False   19440 20240610
G1_spatiotemporal_gnn G1__temporal__no_target_history__spatial_only__all_edges__h128_L2_seed20240610     temporal      train no_target_history  spatial_only        all_edges            ranking__ndcg_at_100      0.504587              True   19440 20240610
G1_spatiotemporal_gnn G1__temporal__no_target_history__spatial_only__all_edges__h128_L2_seed20240610     temporal      train no_target_history  spatial_only        all_edges          ranking__spearman_corr      0.702593              True   19440 20240610
G1_spatiotemporal_gnn G1__temporal__no_target_history__spatial_only__all_edges__h128_L2_seed20240610     temporal      train no_target_history  spatial_only        all_edges ranking__top_10pct_overlap_rate      0.494342              True   19440 20240610
G1_spatiotemporal_gnn G1__temporal__no_target_history__spatial_only__all_edges__h128_L2_seed20240610     temporal validation no_target_history  spatial_only        all_edges                      count__mae      2.640088             False    4320 20240610
G1_spatiotemporal_gnn G1__temporal__no_target_history__spatial_only__all_edges__h128_L2_seed20240610     temporal validation no_target_history  spatial_only        all_edges    count__mean_poisson_deviance      2.739140             False    4320 20240610
G1_spatiotemporal_gnn G1__temporal__no_target_history__spatial_only__all_edges__h128_L2_seed20240610     temporal validation no_target_history  spatial_only        all_edges                     count__rmse      4.136995             False    4320 20240610
G1_spatiotemporal_gnn G1__temporal__no_target_history__spatial_only__all_edges__h128_L2_seed20240610     temporal validation no_target_history  spatial_only        all_edges            ranking__ndcg_at_100      0.646767              True    4320 20240610
G1_spatiotemporal_gnn G1__temporal__no_target_history__spatial_only__all_edges__h128_L2_seed20240610     temporal validation no_target_history  spatial_only        all_edges          ranking__spearman_corr      0.717949              True    4320 20240610
G1_spatiotemporal_gnn G1__temporal__no_target_history__spatial_only__all_edges__h128_L2_seed20240610     temporal validation no_target_history  spatial_only        all_edges ranking__top_10pct_overlap_rate      0.497685              True    4320 20240610
```

## Graph-regime audit

Edge masks define the message-passing graph used by each trial. `all_edges` is transductive; `no_test_incident_edges` removes all edges touching test nodes; `train_train_edges` uses only train-train edges when requested.

```text
   feature_regime split_scheme            edge_regime edge_mask_regime  n_edges_total_graph  n_edges_used  n_edge_types_used                                                                edge_types_used  uses_temporal_edges  uses_spatial_edges  uses_placebo_edges
  all_forecasting     temporal               no_edges        all_edges               508140             0                  0                                                                                               False               False               False
  all_forecasting     temporal          temporal_only        all_edges               508140         50220                  2                                       temporal_self_lag_1,temporal_self_lag_12                 True               False               False
  all_forecasting     temporal           spatial_only        all_edges               508140        228960                  1                                                         spatial_knn_same_month                False                True               False
  all_forecasting     temporal       spatial_temporal        all_edges               508140        279180                  3                spatial_knn_same_month,temporal_self_lag_1,temporal_self_lag_12                 True                True               False
  all_forecasting     temporal random_spatial_placebo        all_edges               508140        279180                  3 spatial_knn_same_month_random_placebo,temporal_self_lag_1,temporal_self_lag_12                 True                True                True
 lagged_reporting     temporal               no_edges        all_edges               508140             0                  0                                                                                               False               False               False
 lagged_reporting     temporal          temporal_only        all_edges               508140         50220                  2                                       temporal_self_lag_1,temporal_self_lag_12                 True               False               False
 lagged_reporting     temporal           spatial_only        all_edges               508140        228960                  1                                                         spatial_knn_same_month                False                True               False
 lagged_reporting     temporal       spatial_temporal        all_edges               508140        279180                  3                spatial_knn_same_month,temporal_self_lag_1,temporal_self_lag_12                 True                True               False
 lagged_reporting     temporal random_spatial_placebo        all_edges               508140        279180                  3 spatial_knn_same_month_random_placebo,temporal_self_lag_1,temporal_self_lag_12                 True                True                True
no_target_history     temporal               no_edges        all_edges               508140             0                  0                                                                                               False               False               False
no_target_history     temporal          temporal_only        all_edges               508140         50220                  2                                       temporal_self_lag_1,temporal_self_lag_12                 True               False               False
no_target_history     temporal           spatial_only        all_edges               508140        228960                  1                                                         spatial_knn_same_month                False                True               False
no_target_history     temporal       spatial_temporal        all_edges               508140        279180                  3                spatial_knn_same_month,temporal_self_lag_1,temporal_self_lag_12                 True                True               False
no_target_history     temporal random_spatial_placebo        all_edges               508140        279180                  3 spatial_knn_same_month_random_placebo,temporal_self_lag_1,temporal_self_lag_12                 True                True                True
```

## Feature preprocessing audit preview

Feature imputation and scaling are fit on train nodes only for each split/feature-regime combination.

```text
 feature_regime split_scheme                                                                  feature  train_median    train_mean   train_std
all_forecasting     temporal                               target_history__water_drainage_count_lag_1  4.000000e+00  5.324743e+00    5.205296
all_forecasting     temporal                               target_history__water_drainage_count_lag_2  4.000000e+00  5.303189e+00    5.125669
all_forecasting     temporal                               target_history__water_drainage_count_lag_3  4.000000e+00  5.269959e+00    5.056834
all_forecasting     temporal                               target_history__water_drainage_count_lag_6  4.000000e+00  5.065278e+00    4.675610
all_forecasting     temporal                              target_history__water_drainage_count_lag_12  4.000000e+00  4.958076e+00    4.333220
all_forecasting     temporal                   target_history__water_drainage_count_roll3_mean_shift1  4.666667e+00  5.347454e+00    4.384836
all_forecasting     temporal                    target_history__water_drainage_count_roll3_sum_shift1  1.300000e+01  1.559234e+01   13.034337
all_forecasting     temporal                   target_history__water_drainage_count_roll6_mean_shift1  4.666667e+00  5.322275e+00    4.044493
all_forecasting     temporal                    target_history__water_drainage_count_roll6_sum_shift1  2.600000e+01  2.971260e+01   23.749451
all_forecasting     temporal                  target_history__water_drainage_count_roll12_mean_shift1  4.750000e+00  5.340138e+00    3.843044
all_forecasting     temporal                   target_history__water_drainage_count_roll12_sum_shift1  4.700000e+01  5.372917e+01   42.979145
all_forecasting     temporal               target_history__water_drainage_count_expanding_mean_shift1  4.985294e+00  5.494469e+00    3.863878
all_forecasting     temporal                                               target_train_summary__mean  4.847222e+00  5.309568e+00    3.666113
all_forecasting     temporal                                             target_train_summary__median  4.000000e+00  4.762037e+00    3.443370
all_forecasting     temporal                                                target_train_summary__p90  8.500000e+00  9.185185e+00    5.694699
all_forecasting     temporal                                      target_train_summary__positive_rate  9.722222e-01  8.534979e-01    0.276043
all_forecasting     temporal              reporting_history__total_311_count_non_water_drainage_lag_1  4.200000e+01  4.814552e+01   45.325371
all_forecasting     temporal              reporting_history__total_311_count_non_water_drainage_lag_2  4.200000e+01  4.822551e+01   44.937428
all_forecasting     temporal              reporting_history__total_311_count_non_water_drainage_lag_3  4.200000e+01  4.811574e+01   44.398109
all_forecasting     temporal             reporting_history__total_311_count_non_water_drainage_lag_12  4.100000e+01  4.556194e+01   38.070087
all_forecasting     temporal  reporting_history__total_311_count_non_water_drainage_roll3_mean_shift1  4.266667e+01  4.803017e+01   43.495918
all_forecasting     temporal  reporting_history__total_311_count_non_water_drainage_roll6_mean_shift1  4.266667e+01  4.756662e+01   42.713730
all_forecasting     temporal reporting_history__total_311_count_non_water_drainage_roll12_mean_shift1  4.275000e+01  4.700904e+01   42.308849
all_forecasting     temporal                                   requests_history__requests_total_lag_1  4.700000e+01  5.349805e+01   48.463203
all_forecasting     temporal                                   requests_history__requests_total_lag_3  4.700000e+01  5.346903e+01   47.452221
all_forecasting     temporal                                  requests_history__requests_total_lag_12  4.600000e+01  5.085334e+01   40.632614
all_forecasting     temporal                       requests_history__requests_total_roll3_mean_shift1  4.766667e+01  5.338688e+01   46.569595
all_forecasting     temporal                       requests_history__requests_total_roll6_mean_shift1  4.766667e+01  5.289816e+01   45.651058
all_forecasting     temporal                      requests_history__requests_total_roll12_mean_shift1  4.783333e+01  5.235844e+01   45.133419
all_forecasting     temporal                                              svi_primary__svi_percentile  7.386986e-01  6.830530e-01    0.242299
all_forecasting     temporal                                               svi_primary__svi_score_raw  2.485274e+00  2.513813e+00    0.746725
all_forecasting     temporal                                      static__log1p_population_total_2021  8.180451e+00  7.995710e+00    1.111241
all_forecasting     temporal                                              static__log1p_land_area_km2  3.905545e-01  5.139575e-01    0.435412
all_forecasting     temporal                                         static__log1p_population_density  8.958465e+00  8.668377e+00    1.378286
all_forecasting     temporal                                         static_spatial__tract_centroid_x  7.628184e+06  7.626816e+06 5892.627441
all_forecasting     temporal                                         static_spatial__tract_centroid_y  1.246208e+06  1.245772e+06 6999.055176
all_forecasting     temporal                                       static_spatial__tract_centroid_lon -7.360283e+01 -7.362295e+01    0.084887
all_forecasting     temporal                                       static_spatial__tract_centroid_lat  4.552404e+01  4.552384e+01    0.054920
all_forecasting     temporal                                                    calendar__month_is_02  0.000000e+00  8.333334e-02    0.276385
all_forecasting     temporal                                                    calendar__month_is_03  0.000000e+00  8.333334e-02    0.276385
all_forecasting     temporal                                                    calendar__month_is_04  0.000000e+00  8.333334e-02    0.276385
all_forecasting     temporal                                                    calendar__month_is_05  0.000000e+00  8.333334e-02    0.276385
all_forecasting     temporal                                                    calendar__month_is_06  0.000000e+00  8.333334e-02    0.276385
all_forecasting     temporal                                                    calendar__month_is_07  0.000000e+00  8.333334e-02    0.276385
all_forecasting     temporal                                                    calendar__month_is_08  0.000000e+00  8.333334e-02    0.276385
all_forecasting     temporal                                                    calendar__month_is_09  0.000000e+00  8.333334e-02    0.276385
all_forecasting     temporal                                                    calendar__month_is_10  0.000000e+00  8.333334e-02    0.276385
all_forecasting     temporal                                                    calendar__month_is_11  0.000000e+00  8.333334e-02    0.276385
all_forecasting     temporal                                                    calendar__month_is_12  0.000000e+00  8.333334e-02    0.276385
all_forecasting     temporal                                                      calendar__month_sin -6.123234e-17 -7.812680e-18    0.707107
all_forecasting     temporal                                                      calendar__month_cos -6.123234e-17 -7.310110e-18    0.707107
all_forecasting     temporal                                       calendar__period_index_since_start  1.750000e+01  1.750000e+01   10.388294
all_forecasting     temporal                               target_history__water_drainage_count_lag_1  4.000000e+00  5.324743e+00    5.205296
all_forecasting     temporal                               target_history__water_drainage_count_lag_2  4.000000e+00  5.303189e+00    5.125669
all_forecasting     temporal                               target_history__water_drainage_count_lag_3  4.000000e+00  5.269959e+00    5.056834
all_forecasting     temporal                               target_history__water_drainage_count_lag_6  4.000000e+00  5.065278e+00    4.675610
all_forecasting     temporal                              target_history__water_drainage_count_lag_12  4.000000e+00  4.958076e+00    4.333220
all_forecasting     temporal                   target_history__water_drainage_count_roll3_mean_shift1  4.666667e+00  5.347454e+00    4.384836
all_forecasting     temporal                    target_history__water_drainage_count_roll3_sum_shift1  1.300000e+01  1.559234e+01   13.034337
all_forecasting     temporal                   target_history__water_drainage_count_roll6_mean_shift1  4.666667e+00  5.322275e+00    4.044493
all_forecasting     temporal                    target_history__water_drainage_count_roll6_sum_shift1  2.600000e+01  2.971260e+01   23.749451
all_forecasting     temporal                  target_history__water_drainage_count_roll12_mean_shift1  4.750000e+00  5.340138e+00    3.843044
all_forecasting     temporal                   target_history__water_drainage_count_roll12_sum_shift1  4.700000e+01  5.372917e+01   42.979145
all_forecasting     temporal               target_history__water_drainage_count_expanding_mean_shift1  4.985294e+00  5.494469e+00    3.863878
all_forecasting     temporal                                               target_train_summary__mean  4.847222e+00  5.309568e+00    3.666113
all_forecasting     temporal                                             target_train_summary__median  4.000000e+00  4.762037e+00    3.443370
all_forecasting     temporal                                                target_train_summary__p90  8.500000e+00  9.185185e+00    5.694699
all_forecasting     temporal                                      target_train_summary__positive_rate  9.722222e-01  8.534979e-01    0.276043
all_forecasting     temporal              reporting_history__total_311_count_non_water_drainage_lag_1  4.200000e+01  4.814552e+01   45.325371
all_forecasting     temporal              reporting_history__total_311_count_non_water_drainage_lag_2  4.200000e+01  4.822551e+01   44.937428
all_forecasting     temporal              reporting_history__total_311_count_non_water_drainage_lag_3  4.200000e+01  4.811574e+01   44.398109
all_forecasting     temporal             reporting_history__total_311_count_non_water_drainage_lag_12  4.100000e+01  4.556194e+01   38.070087
all_forecasting     temporal  reporting_history__total_311_count_non_water_drainage_roll3_mean_shift1  4.266667e+01  4.803017e+01   43.495918
all_forecasting     temporal  reporting_history__total_311_count_non_water_drainage_roll6_mean_shift1  4.266667e+01  4.756662e+01   42.713730
all_forecasting     temporal reporting_history__total_311_count_non_water_drainage_roll12_mean_shift1  4.275000e+01  4.700904e+01   42.308849
all_forecasting     temporal                                   requests_history__requests_total_lag_1  4.700000e+01  5.349805e+01   48.463203
all_forecasting     temporal                                   requests_history__requests_total_lag_3  4.700000e+01  5.346903e+01   47.452221
all_forecasting     temporal                                  requests_history__requests_total_lag_12  4.600000e+01  5.085334e+01   40.632614
all_forecasting     temporal                       requests_history__requests_total_roll3_mean_shift1  4.766667e+01  5.338688e+01   46.569595
all_forecasting     temporal                       requests_history__requests_total_roll6_mean_shift1  4.766667e+01  5.289816e+01   45.651058
```

## Output artifacts

| Artifact | Path |
|---|---|
| `metrics` | `urban_graph_benchmark/outputs/mtl_311_water_v0/baselines/G1_spatiotemporal_gnn_temporal_core/metrics.csv` |
| `predictions_validation` | `urban_graph_benchmark/outputs/mtl_311_water_v0/baselines/G1_spatiotemporal_gnn_temporal_core/predictions_validation.parquet` |
| `predictions_test` | `urban_graph_benchmark/outputs/mtl_311_water_v0/baselines/G1_spatiotemporal_gnn_temporal_core/predictions_test.parquet` |
| `predictions_all_evaluated` | `urban_graph_benchmark/outputs/mtl_311_water_v0/baselines/G1_spatiotemporal_gnn_temporal_core/predictions_all_evaluated.parquet` |
| `training_curves` | `urban_graph_benchmark/outputs/mtl_311_water_v0/baselines/G1_spatiotemporal_gnn_temporal_core/training_curves.csv` |
| `trial_audit` | `urban_graph_benchmark/outputs/mtl_311_water_v0/baselines/G1_spatiotemporal_gnn_temporal_core/trial_audit.csv` |
| `model_selection_audit` | `urban_graph_benchmark/outputs/mtl_311_water_v0/baselines/G1_spatiotemporal_gnn_temporal_core/model_selection_audit.csv` |
| `feature_preprocessing_audit` | `urban_graph_benchmark/outputs/mtl_311_water_v0/baselines/G1_spatiotemporal_gnn_temporal_core/feature_preprocessing_audit.csv` |
| `graph_regime_audit` | `urban_graph_benchmark/outputs/mtl_311_water_v0/baselines/G1_spatiotemporal_gnn_temporal_core/graph_regime_audit.csv` |
| `model_metadata` | `urban_graph_benchmark/outputs/mtl_311_water_v0/baselines/G1_spatiotemporal_gnn_temporal_core/model_metadata.json` |
| `baseline_report` | `urban_graph_benchmark/outputs/mtl_311_water_v0/baselines/G1_spatiotemporal_gnn_temporal_core/baseline_report.md` |
| `checkpoints_dir` | `urban_graph_benchmark/outputs/mtl_311_water_v0/baselines/G1_spatiotemporal_gnn_temporal_core/checkpoints` |
| `embeddings_dir` | `urban_graph_benchmark/outputs/mtl_311_water_v0/baselines/G1_spatiotemporal_gnn_temporal_core/embeddings` |

## Benchmark handoff

Temporal graph results should be compared against A0_3_tract_train_mean and A3_all_forecasting RF. Spatial-block graph results should be compared against A3_lagged_reporting_forecasting HGB. The most convincing graph claim would improve both count error and high-burden ranking metrics under spatial-block evaluation.
