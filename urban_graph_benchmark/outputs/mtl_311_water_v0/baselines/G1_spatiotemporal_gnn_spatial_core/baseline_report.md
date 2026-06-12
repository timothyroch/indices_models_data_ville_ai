# G1 Spatiotemporal Tract GNN Baseline

Generated at: `2026-06-12T13:58:31.189567+00:00`

Graph artifact directory: `urban_graph_benchmark/outputs/mtl_311_water_v0/graphs/G1_tract_month_graph`

Output directory: `urban_graph_benchmark/outputs/mtl_311_water_v0/baselines/G1_spatiotemporal_gnn_spatial_core`

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
                                                                                                model_name    status  split_scheme    feature_regime            edge_regime       edge_mask_regime  best_epoch  best_validation_mae  n_edges_used  n_relations_used  elapsed_seconds
                             G1__spatial_block__all_forecasting__no_edges__all_edges__h128_L2_seed20240610 completed spatial_block   all_forecasting               no_edges              all_edges          32             2.727494             0                 0         1.314530
                        G1__spatial_block__all_forecasting__temporal_only__all_edges__h128_L2_seed20240610 completed spatial_block   all_forecasting          temporal_only              all_edges          40             2.701370         50220                 2         1.483593
           G1__spatial_block__all_forecasting__temporal_only__no_test_incident_edges__h128_L2_seed20240610 completed spatial_block   all_forecasting          temporal_only no_test_incident_edges          40             2.701405         46593                 2         1.494213
                         G1__spatial_block__all_forecasting__spatial_only__all_edges__h128_L2_seed20240610 completed spatial_block   all_forecasting           spatial_only              all_edges          39             2.814959        228960                 1         2.473590
            G1__spatial_block__all_forecasting__spatial_only__no_test_incident_edges__h128_L2_seed20240610 completed spatial_block   all_forecasting           spatial_only no_test_incident_edges          39             2.818847        205746                 1         2.355161
                     G1__spatial_block__all_forecasting__spatial_temporal__all_edges__h128_L2_seed20240610 completed spatial_block   all_forecasting       spatial_temporal              all_edges          15             2.845467        279180                 3         2.141425
        G1__spatial_block__all_forecasting__spatial_temporal__no_test_incident_edges__h128_L2_seed20240610 completed spatial_block   all_forecasting       spatial_temporal no_test_incident_edges          15             2.845386        252339                 3         2.062777
               G1__spatial_block__all_forecasting__random_spatial_placebo__all_edges__h128_L2_seed20240610 completed spatial_block   all_forecasting random_spatial_placebo              all_edges          15             2.835845        279180                 3         2.192078
  G1__spatial_block__all_forecasting__random_spatial_placebo__no_test_incident_edges__h128_L2_seed20240610 completed spatial_block   all_forecasting random_spatial_placebo no_test_incident_edges          15             2.837360        243568                 3         2.055587
                            G1__spatial_block__lagged_reporting__no_edges__all_edges__h128_L2_seed20240610 completed spatial_block  lagged_reporting               no_edges              all_edges          38             2.829357             0                 0         0.413350
                       G1__spatial_block__lagged_reporting__temporal_only__all_edges__h128_L2_seed20240610 completed spatial_block  lagged_reporting          temporal_only              all_edges          27             2.887125         50220                 2         1.181536
          G1__spatial_block__lagged_reporting__temporal_only__no_test_incident_edges__h128_L2_seed20240610 completed spatial_block  lagged_reporting          temporal_only no_test_incident_edges          27             2.887117         46593                 2         1.199456
                        G1__spatial_block__lagged_reporting__spatial_only__all_edges__h128_L2_seed20240610 completed spatial_block  lagged_reporting           spatial_only              all_edges          27             2.876456        228960                 1         2.045585
           G1__spatial_block__lagged_reporting__spatial_only__no_test_incident_edges__h128_L2_seed20240610 completed spatial_block  lagged_reporting           spatial_only no_test_incident_edges          27             2.872665        205746                 1         1.960180
                    G1__spatial_block__lagged_reporting__spatial_temporal__all_edges__h128_L2_seed20240610 completed spatial_block  lagged_reporting       spatial_temporal              all_edges          21             2.891814        279180                 3         2.388146
       G1__spatial_block__lagged_reporting__spatial_temporal__no_test_incident_edges__h128_L2_seed20240610 completed spatial_block  lagged_reporting       spatial_temporal no_test_incident_edges          21             2.892538        252339                 3         2.296980
              G1__spatial_block__lagged_reporting__random_spatial_placebo__all_edges__h128_L2_seed20240610 completed spatial_block  lagged_reporting random_spatial_placebo              all_edges          22             2.880180        279180                 3         2.492145
 G1__spatial_block__lagged_reporting__random_spatial_placebo__no_test_incident_edges__h128_L2_seed20240610 completed spatial_block  lagged_reporting random_spatial_placebo no_test_incident_edges          22             2.881727        243568                 3         2.397389
                           G1__spatial_block__no_target_history__no_edges__all_edges__h128_L2_seed20240610 completed spatial_block no_target_history               no_edges              all_edges          22             2.912395             0                 0         0.328412
                      G1__spatial_block__no_target_history__temporal_only__all_edges__h128_L2_seed20240610 completed spatial_block no_target_history          temporal_only              all_edges          29             2.883242         50220                 2         1.220986
         G1__spatial_block__no_target_history__temporal_only__no_test_incident_edges__h128_L2_seed20240610 completed spatial_block no_target_history          temporal_only no_test_incident_edges          29             2.883216         46593                 2         1.257883
                       G1__spatial_block__no_target_history__spatial_only__all_edges__h128_L2_seed20240610 completed spatial_block no_target_history           spatial_only              all_edges          28             2.860664        228960                 1         2.154597
          G1__spatial_block__no_target_history__spatial_only__no_test_incident_edges__h128_L2_seed20240610 completed spatial_block no_target_history           spatial_only no_test_incident_edges          28             2.867044        205746                 1         1.993463
                   G1__spatial_block__no_target_history__spatial_temporal__all_edges__h128_L2_seed20240610 completed spatial_block no_target_history       spatial_temporal              all_edges          34             2.906983        279180                 3         2.978219
      G1__spatial_block__no_target_history__spatial_temporal__no_test_incident_edges__h128_L2_seed20240610 completed spatial_block no_target_history       spatial_temporal no_test_incident_edges          34             2.903661        252339                 3         2.870023
             G1__spatial_block__no_target_history__random_spatial_placebo__all_edges__h128_L2_seed20240610 completed spatial_block no_target_history random_spatial_placebo              all_edges          33             2.925234        279180                 3         3.013329
G1__spatial_block__no_target_history__random_spatial_placebo__no_test_incident_edges__h128_L2_seed20240610 completed spatial_block no_target_history random_spatial_placebo no_test_incident_edges          33             2.923119        243568                 3         2.807245
```

## Validation-only model selection

```text
                                                                                                model_name  split_scheme    feature_regime            edge_regime       edge_mask_regime  validation_mae  validation_spearman  test_mae  test_spearman  selected_overall_for_split  selected_for_feature_regime  selected_for_test_summary
                        G1__spatial_block__all_forecasting__temporal_only__all_edges__h128_L2_seed20240610 spatial_block   all_forecasting          temporal_only              all_edges        2.701370             0.655859  2.443677       0.754360                        True                         True                       True
           G1__spatial_block__all_forecasting__temporal_only__no_test_incident_edges__h128_L2_seed20240610 spatial_block   all_forecasting          temporal_only no_test_incident_edges        2.701405             0.655860  2.447381       0.749069                       False                        False                      False
                             G1__spatial_block__all_forecasting__no_edges__all_edges__h128_L2_seed20240610 spatial_block   all_forecasting               no_edges              all_edges        2.727494             0.646632  2.422627       0.751735                       False                        False                      False
                         G1__spatial_block__all_forecasting__spatial_only__all_edges__h128_L2_seed20240610 spatial_block   all_forecasting           spatial_only              all_edges        2.814959             0.647181  2.580073       0.737918                       False                        False                      False
            G1__spatial_block__all_forecasting__spatial_only__no_test_incident_edges__h128_L2_seed20240610 spatial_block   all_forecasting           spatial_only no_test_incident_edges        2.818847             0.647196  2.485036       0.746335                       False                        False                      False
                            G1__spatial_block__lagged_reporting__no_edges__all_edges__h128_L2_seed20240610 spatial_block  lagged_reporting               no_edges              all_edges        2.829357             0.617654  2.538687       0.747131                       False                         True                       True
               G1__spatial_block__all_forecasting__random_spatial_placebo__all_edges__h128_L2_seed20240610 spatial_block   all_forecasting random_spatial_placebo              all_edges        2.835845             0.630473  2.555845       0.747222                       False                        False                      False
  G1__spatial_block__all_forecasting__random_spatial_placebo__no_test_incident_edges__h128_L2_seed20240610 spatial_block   all_forecasting random_spatial_placebo no_test_incident_edges        2.837360             0.630351  2.834995       0.702400                       False                        False                      False
        G1__spatial_block__all_forecasting__spatial_temporal__no_test_incident_edges__h128_L2_seed20240610 spatial_block   all_forecasting       spatial_temporal no_test_incident_edges        2.845387             0.627910  2.853737       0.701469                       False                        False                      False
                     G1__spatial_block__all_forecasting__spatial_temporal__all_edges__h128_L2_seed20240610 spatial_block   all_forecasting       spatial_temporal              all_edges        2.845467             0.627958  2.566587       0.748125                       False                        False                      False
                       G1__spatial_block__no_target_history__spatial_only__all_edges__h128_L2_seed20240610 spatial_block no_target_history           spatial_only              all_edges        2.860664             0.603658  2.674980       0.726135                       False                         True                       True
          G1__spatial_block__no_target_history__spatial_only__no_test_incident_edges__h128_L2_seed20240610 spatial_block no_target_history           spatial_only no_test_incident_edges        2.867044             0.602370  2.641255       0.720787                       False                        False                      False
           G1__spatial_block__lagged_reporting__spatial_only__no_test_incident_edges__h128_L2_seed20240610 spatial_block  lagged_reporting           spatial_only no_test_incident_edges        2.872665             0.610079  2.661529       0.722409                       False                        False                      False
                        G1__spatial_block__lagged_reporting__spatial_only__all_edges__h128_L2_seed20240610 spatial_block  lagged_reporting           spatial_only              all_edges        2.876457             0.609717  2.653636       0.727265                       False                        False                      False
              G1__spatial_block__lagged_reporting__random_spatial_placebo__all_edges__h128_L2_seed20240610 spatial_block  lagged_reporting random_spatial_placebo              all_edges        2.880180             0.609161  2.614144       0.734590                       False                        False                      False
 G1__spatial_block__lagged_reporting__random_spatial_placebo__no_test_incident_edges__h128_L2_seed20240610 spatial_block  lagged_reporting random_spatial_placebo no_test_incident_edges        2.881727             0.608709  2.749321       0.716701                       False                        False                      False
         G1__spatial_block__no_target_history__temporal_only__no_test_incident_edges__h128_L2_seed20240610 spatial_block no_target_history          temporal_only no_test_incident_edges        2.883216             0.603755  2.826518       0.677184                       False                        False                      False
                      G1__spatial_block__no_target_history__temporal_only__all_edges__h128_L2_seed20240610 spatial_block no_target_history          temporal_only              all_edges        2.883242             0.603758  2.733084       0.708461                       False                        False                      False
          G1__spatial_block__lagged_reporting__temporal_only__no_test_incident_edges__h128_L2_seed20240610 spatial_block  lagged_reporting          temporal_only no_test_incident_edges        2.887117             0.607492  2.709219       0.720667                       False                        False                      False
                       G1__spatial_block__lagged_reporting__temporal_only__all_edges__h128_L2_seed20240610 spatial_block  lagged_reporting          temporal_only              all_edges        2.887125             0.607503  2.636127       0.736686                       False                        False                      False
                    G1__spatial_block__lagged_reporting__spatial_temporal__all_edges__h128_L2_seed20240610 spatial_block  lagged_reporting       spatial_temporal              all_edges        2.891814             0.609184  2.626826       0.731762                       False                        False                      False
       G1__spatial_block__lagged_reporting__spatial_temporal__no_test_incident_edges__h128_L2_seed20240610 spatial_block  lagged_reporting       spatial_temporal no_test_incident_edges        2.892538             0.609074  2.763096       0.714598                       False                        False                      False
      G1__spatial_block__no_target_history__spatial_temporal__no_test_incident_edges__h128_L2_seed20240610 spatial_block no_target_history       spatial_temporal no_test_incident_edges        2.903662             0.609733  2.697990       0.713080                       False                        False                      False
                   G1__spatial_block__no_target_history__spatial_temporal__all_edges__h128_L2_seed20240610 spatial_block no_target_history       spatial_temporal              all_edges        2.906983             0.609200  2.678327       0.720703                       False                        False                      False
                           G1__spatial_block__no_target_history__no_edges__all_edges__h128_L2_seed20240610 spatial_block no_target_history               no_edges              all_edges        2.912396             0.595864  2.681391       0.722203                       False                        False                      False
G1__spatial_block__no_target_history__random_spatial_placebo__no_test_incident_edges__h128_L2_seed20240610 spatial_block no_target_history random_spatial_placebo no_test_incident_edges        2.923119             0.605269  2.755042       0.708139                       False                        False                      False
             G1__spatial_block__no_target_history__random_spatial_placebo__all_edges__h128_L2_seed20240610 spatial_block no_target_history random_spatial_placebo              all_edges        2.925234             0.605329  2.689676       0.719704                       False                        False                      False
```

## Compact metrics for selected models

```text
          model_stage                                                                          model_name  split_scheme split_name    feature_regime   edge_regime edge_mask_regime                     metric_name  metric_value  higher_is_better  n_rows     seed
G1_spatiotemporal_gnn  G1__spatial_block__all_forecasting__temporal_only__all_edges__h128_L2_seed20240610 spatial_block       test   all_forecasting temporal_only        all_edges                      count__mae      2.443677             False    2067 20240610
G1_spatiotemporal_gnn  G1__spatial_block__all_forecasting__temporal_only__all_edges__h128_L2_seed20240610 spatial_block       test   all_forecasting temporal_only        all_edges    count__mean_poisson_deviance      2.401075             False    2067 20240610
G1_spatiotemporal_gnn  G1__spatial_block__all_forecasting__temporal_only__all_edges__h128_L2_seed20240610 spatial_block       test   all_forecasting temporal_only        all_edges                     count__rmse      3.953976             False    2067 20240610
G1_spatiotemporal_gnn  G1__spatial_block__all_forecasting__temporal_only__all_edges__h128_L2_seed20240610 spatial_block       test   all_forecasting temporal_only        all_edges            ranking__ndcg_at_100      0.813935              True    2067 20240610
G1_spatiotemporal_gnn  G1__spatial_block__all_forecasting__temporal_only__all_edges__h128_L2_seed20240610 spatial_block       test   all_forecasting temporal_only        all_edges          ranking__spearman_corr      0.754360              True    2067 20240610
G1_spatiotemporal_gnn  G1__spatial_block__all_forecasting__temporal_only__all_edges__h128_L2_seed20240610 spatial_block       test   all_forecasting temporal_only        all_edges ranking__top_10pct_overlap_rate      0.603865              True    2067 20240610
G1_spatiotemporal_gnn  G1__spatial_block__all_forecasting__temporal_only__all_edges__h128_L2_seed20240610 spatial_block      train   all_forecasting temporal_only        all_edges                      count__mae      2.384673             False   21200 20240610
G1_spatiotemporal_gnn  G1__spatial_block__all_forecasting__temporal_only__all_edges__h128_L2_seed20240610 spatial_block      train   all_forecasting temporal_only        all_edges    count__mean_poisson_deviance      2.291257             False   21200 20240610
G1_spatiotemporal_gnn  G1__spatial_block__all_forecasting__temporal_only__all_edges__h128_L2_seed20240610 spatial_block      train   all_forecasting temporal_only        all_edges                     count__rmse      3.795458             False   21200 20240610
G1_spatiotemporal_gnn  G1__spatial_block__all_forecasting__temporal_only__all_edges__h128_L2_seed20240610 spatial_block      train   all_forecasting temporal_only        all_edges            ranking__ndcg_at_100      0.437783              True   21200 20240610
G1_spatiotemporal_gnn  G1__spatial_block__all_forecasting__temporal_only__all_edges__h128_L2_seed20240610 spatial_block      train   all_forecasting temporal_only        all_edges          ranking__spearman_corr      0.747989              True   21200 20240610
G1_spatiotemporal_gnn  G1__spatial_block__all_forecasting__temporal_only__all_edges__h128_L2_seed20240610 spatial_block      train   all_forecasting temporal_only        all_edges ranking__top_10pct_overlap_rate      0.525943              True   21200 20240610
G1_spatiotemporal_gnn  G1__spatial_block__all_forecasting__temporal_only__all_edges__h128_L2_seed20240610 spatial_block validation   all_forecasting temporal_only        all_edges                      count__mae      2.701370             False    5353 20240610
G1_spatiotemporal_gnn  G1__spatial_block__all_forecasting__temporal_only__all_edges__h128_L2_seed20240610 spatial_block validation   all_forecasting temporal_only        all_edges    count__mean_poisson_deviance      2.496756             False    5353 20240610
G1_spatiotemporal_gnn  G1__spatial_block__all_forecasting__temporal_only__all_edges__h128_L2_seed20240610 spatial_block validation   all_forecasting temporal_only        all_edges                     count__rmse      4.039091             False    5353 20240610
G1_spatiotemporal_gnn  G1__spatial_block__all_forecasting__temporal_only__all_edges__h128_L2_seed20240610 spatial_block validation   all_forecasting temporal_only        all_edges            ranking__ndcg_at_100      0.624064              True    5353 20240610
G1_spatiotemporal_gnn  G1__spatial_block__all_forecasting__temporal_only__all_edges__h128_L2_seed20240610 spatial_block validation   all_forecasting temporal_only        all_edges          ranking__spearman_corr      0.655859              True    5353 20240610
G1_spatiotemporal_gnn  G1__spatial_block__all_forecasting__temporal_only__all_edges__h128_L2_seed20240610 spatial_block validation   all_forecasting temporal_only        all_edges ranking__top_10pct_overlap_rate      0.529851              True    5353 20240610
G1_spatiotemporal_gnn      G1__spatial_block__lagged_reporting__no_edges__all_edges__h128_L2_seed20240610 spatial_block       test  lagged_reporting      no_edges        all_edges                      count__mae      2.538687             False    2067 20240610
G1_spatiotemporal_gnn      G1__spatial_block__lagged_reporting__no_edges__all_edges__h128_L2_seed20240610 spatial_block       test  lagged_reporting      no_edges        all_edges    count__mean_poisson_deviance      2.669218             False    2067 20240610
G1_spatiotemporal_gnn      G1__spatial_block__lagged_reporting__no_edges__all_edges__h128_L2_seed20240610 spatial_block       test  lagged_reporting      no_edges        all_edges                     count__rmse      4.161396             False    2067 20240610
G1_spatiotemporal_gnn      G1__spatial_block__lagged_reporting__no_edges__all_edges__h128_L2_seed20240610 spatial_block       test  lagged_reporting      no_edges        all_edges            ranking__ndcg_at_100      0.770606              True    2067 20240610
G1_spatiotemporal_gnn      G1__spatial_block__lagged_reporting__no_edges__all_edges__h128_L2_seed20240610 spatial_block       test  lagged_reporting      no_edges        all_edges          ranking__spearman_corr      0.747131              True    2067 20240610
G1_spatiotemporal_gnn      G1__spatial_block__lagged_reporting__no_edges__all_edges__h128_L2_seed20240610 spatial_block       test  lagged_reporting      no_edges        all_edges ranking__top_10pct_overlap_rate      0.589372              True    2067 20240610
G1_spatiotemporal_gnn      G1__spatial_block__lagged_reporting__no_edges__all_edges__h128_L2_seed20240610 spatial_block      train  lagged_reporting      no_edges        all_edges                      count__mae      2.528663             False   21200 20240610
G1_spatiotemporal_gnn      G1__spatial_block__lagged_reporting__no_edges__all_edges__h128_L2_seed20240610 spatial_block      train  lagged_reporting      no_edges        all_edges    count__mean_poisson_deviance      2.534174             False   21200 20240610
G1_spatiotemporal_gnn      G1__spatial_block__lagged_reporting__no_edges__all_edges__h128_L2_seed20240610 spatial_block      train  lagged_reporting      no_edges        all_edges                     count__rmse      3.965415             False   21200 20240610
G1_spatiotemporal_gnn      G1__spatial_block__lagged_reporting__no_edges__all_edges__h128_L2_seed20240610 spatial_block      train  lagged_reporting      no_edges        all_edges            ranking__ndcg_at_100      0.496449              True   21200 20240610
G1_spatiotemporal_gnn      G1__spatial_block__lagged_reporting__no_edges__all_edges__h128_L2_seed20240610 spatial_block      train  lagged_reporting      no_edges        all_edges          ranking__spearman_corr      0.702767              True   21200 20240610
G1_spatiotemporal_gnn      G1__spatial_block__lagged_reporting__no_edges__all_edges__h128_L2_seed20240610 spatial_block      train  lagged_reporting      no_edges        all_edges ranking__top_10pct_overlap_rate      0.479245              True   21200 20240610
G1_spatiotemporal_gnn      G1__spatial_block__lagged_reporting__no_edges__all_edges__h128_L2_seed20240610 spatial_block validation  lagged_reporting      no_edges        all_edges                      count__mae      2.829357             False    5353 20240610
G1_spatiotemporal_gnn      G1__spatial_block__lagged_reporting__no_edges__all_edges__h128_L2_seed20240610 spatial_block validation  lagged_reporting      no_edges        all_edges    count__mean_poisson_deviance      2.792279             False    5353 20240610
G1_spatiotemporal_gnn      G1__spatial_block__lagged_reporting__no_edges__all_edges__h128_L2_seed20240610 spatial_block validation  lagged_reporting      no_edges        all_edges                     count__rmse      4.250847             False    5353 20240610
G1_spatiotemporal_gnn      G1__spatial_block__lagged_reporting__no_edges__all_edges__h128_L2_seed20240610 spatial_block validation  lagged_reporting      no_edges        all_edges            ranking__ndcg_at_100      0.585850              True    5353 20240610
G1_spatiotemporal_gnn      G1__spatial_block__lagged_reporting__no_edges__all_edges__h128_L2_seed20240610 spatial_block validation  lagged_reporting      no_edges        all_edges          ranking__spearman_corr      0.617654              True    5353 20240610
G1_spatiotemporal_gnn      G1__spatial_block__lagged_reporting__no_edges__all_edges__h128_L2_seed20240610 spatial_block validation  lagged_reporting      no_edges        all_edges ranking__top_10pct_overlap_rate      0.440299              True    5353 20240610
G1_spatiotemporal_gnn G1__spatial_block__no_target_history__spatial_only__all_edges__h128_L2_seed20240610 spatial_block       test no_target_history  spatial_only        all_edges                      count__mae      2.674980             False    2067 20240610
G1_spatiotemporal_gnn G1__spatial_block__no_target_history__spatial_only__all_edges__h128_L2_seed20240610 spatial_block       test no_target_history  spatial_only        all_edges    count__mean_poisson_deviance      2.847513             False    2067 20240610
G1_spatiotemporal_gnn G1__spatial_block__no_target_history__spatial_only__all_edges__h128_L2_seed20240610 spatial_block       test no_target_history  spatial_only        all_edges                     count__rmse      4.303394             False    2067 20240610
G1_spatiotemporal_gnn G1__spatial_block__no_target_history__spatial_only__all_edges__h128_L2_seed20240610 spatial_block       test no_target_history  spatial_only        all_edges            ranking__ndcg_at_100      0.649851              True    2067 20240610
G1_spatiotemporal_gnn G1__spatial_block__no_target_history__spatial_only__all_edges__h128_L2_seed20240610 spatial_block       test no_target_history  spatial_only        all_edges          ranking__spearman_corr      0.726135              True    2067 20240610
G1_spatiotemporal_gnn G1__spatial_block__no_target_history__spatial_only__all_edges__h128_L2_seed20240610 spatial_block       test no_target_history  spatial_only        all_edges ranking__top_10pct_overlap_rate      0.536232              True    2067 20240610
G1_spatiotemporal_gnn G1__spatial_block__no_target_history__spatial_only__all_edges__h128_L2_seed20240610 spatial_block      train no_target_history  spatial_only        all_edges                      count__mae      2.566401             False   21200 20240610
G1_spatiotemporal_gnn G1__spatial_block__no_target_history__spatial_only__all_edges__h128_L2_seed20240610 spatial_block      train no_target_history  spatial_only        all_edges    count__mean_poisson_deviance      2.517263             False   21200 20240610
G1_spatiotemporal_gnn G1__spatial_block__no_target_history__spatial_only__all_edges__h128_L2_seed20240610 spatial_block      train no_target_history  spatial_only        all_edges                     count__rmse      3.969222             False   21200 20240610
G1_spatiotemporal_gnn G1__spatial_block__no_target_history__spatial_only__all_edges__h128_L2_seed20240610 spatial_block      train no_target_history  spatial_only        all_edges            ranking__ndcg_at_100      0.481167              True   21200 20240610
G1_spatiotemporal_gnn G1__spatial_block__no_target_history__spatial_only__all_edges__h128_L2_seed20240610 spatial_block      train no_target_history  spatial_only        all_edges          ranking__spearman_corr      0.693294              True   21200 20240610
G1_spatiotemporal_gnn G1__spatial_block__no_target_history__spatial_only__all_edges__h128_L2_seed20240610 spatial_block      train no_target_history  spatial_only        all_edges ranking__top_10pct_overlap_rate      0.458019              True   21200 20240610
G1_spatiotemporal_gnn G1__spatial_block__no_target_history__spatial_only__all_edges__h128_L2_seed20240610 spatial_block validation no_target_history  spatial_only        all_edges                      count__mae      2.860664             False    5353 20240610
G1_spatiotemporal_gnn G1__spatial_block__no_target_history__spatial_only__all_edges__h128_L2_seed20240610 spatial_block validation no_target_history  spatial_only        all_edges    count__mean_poisson_deviance      2.762956             False    5353 20240610
G1_spatiotemporal_gnn G1__spatial_block__no_target_history__spatial_only__all_edges__h128_L2_seed20240610 spatial_block validation no_target_history  spatial_only        all_edges                     count__rmse      4.306349             False    5353 20240610
G1_spatiotemporal_gnn G1__spatial_block__no_target_history__spatial_only__all_edges__h128_L2_seed20240610 spatial_block validation no_target_history  spatial_only        all_edges            ranking__ndcg_at_100      0.567474              True    5353 20240610
G1_spatiotemporal_gnn G1__spatial_block__no_target_history__spatial_only__all_edges__h128_L2_seed20240610 spatial_block validation no_target_history  spatial_only        all_edges          ranking__spearman_corr      0.603658              True    5353 20240610
G1_spatiotemporal_gnn G1__spatial_block__no_target_history__spatial_only__all_edges__h128_L2_seed20240610 spatial_block validation no_target_history  spatial_only        all_edges ranking__top_10pct_overlap_rate      0.425373              True    5353 20240610
```

## Graph-regime audit

Edge masks define the message-passing graph used by each trial. `all_edges` is transductive; `no_test_incident_edges` removes all edges touching test nodes; `train_train_edges` uses only train-train edges when requested.

```text
   feature_regime  split_scheme            edge_regime       edge_mask_regime  n_edges_total_graph  n_edges_used  n_edge_types_used                                                                edge_types_used  uses_temporal_edges  uses_spatial_edges  uses_placebo_edges
  all_forecasting spatial_block               no_edges              all_edges               508140             0                  0                                                                                               False               False               False
  all_forecasting spatial_block          temporal_only              all_edges               508140         50220                  2                                       temporal_self_lag_1,temporal_self_lag_12                 True               False               False
  all_forecasting spatial_block          temporal_only no_test_incident_edges               508140         46593                  2                                       temporal_self_lag_1,temporal_self_lag_12                 True               False               False
  all_forecasting spatial_block           spatial_only              all_edges               508140        228960                  1                                                         spatial_knn_same_month                False                True               False
  all_forecasting spatial_block           spatial_only no_test_incident_edges               508140        205746                  1                                                         spatial_knn_same_month                False                True               False
  all_forecasting spatial_block       spatial_temporal              all_edges               508140        279180                  3                spatial_knn_same_month,temporal_self_lag_1,temporal_self_lag_12                 True                True               False
  all_forecasting spatial_block       spatial_temporal no_test_incident_edges               508140        252339                  3                spatial_knn_same_month,temporal_self_lag_1,temporal_self_lag_12                 True                True               False
  all_forecasting spatial_block random_spatial_placebo              all_edges               508140        279180                  3 spatial_knn_same_month_random_placebo,temporal_self_lag_1,temporal_self_lag_12                 True                True                True
  all_forecasting spatial_block random_spatial_placebo no_test_incident_edges               508140        243568                  3 spatial_knn_same_month_random_placebo,temporal_self_lag_1,temporal_self_lag_12                 True                True                True
 lagged_reporting spatial_block               no_edges              all_edges               508140             0                  0                                                                                               False               False               False
 lagged_reporting spatial_block          temporal_only              all_edges               508140         50220                  2                                       temporal_self_lag_1,temporal_self_lag_12                 True               False               False
 lagged_reporting spatial_block          temporal_only no_test_incident_edges               508140         46593                  2                                       temporal_self_lag_1,temporal_self_lag_12                 True               False               False
 lagged_reporting spatial_block           spatial_only              all_edges               508140        228960                  1                                                         spatial_knn_same_month                False                True               False
 lagged_reporting spatial_block           spatial_only no_test_incident_edges               508140        205746                  1                                                         spatial_knn_same_month                False                True               False
 lagged_reporting spatial_block       spatial_temporal              all_edges               508140        279180                  3                spatial_knn_same_month,temporal_self_lag_1,temporal_self_lag_12                 True                True               False
 lagged_reporting spatial_block       spatial_temporal no_test_incident_edges               508140        252339                  3                spatial_knn_same_month,temporal_self_lag_1,temporal_self_lag_12                 True                True               False
 lagged_reporting spatial_block random_spatial_placebo              all_edges               508140        279180                  3 spatial_knn_same_month_random_placebo,temporal_self_lag_1,temporal_self_lag_12                 True                True                True
 lagged_reporting spatial_block random_spatial_placebo no_test_incident_edges               508140        243568                  3 spatial_knn_same_month_random_placebo,temporal_self_lag_1,temporal_self_lag_12                 True                True                True
no_target_history spatial_block               no_edges              all_edges               508140             0                  0                                                                                               False               False               False
no_target_history spatial_block          temporal_only              all_edges               508140         50220                  2                                       temporal_self_lag_1,temporal_self_lag_12                 True               False               False
no_target_history spatial_block          temporal_only no_test_incident_edges               508140         46593                  2                                       temporal_self_lag_1,temporal_self_lag_12                 True               False               False
no_target_history spatial_block           spatial_only              all_edges               508140        228960                  1                                                         spatial_knn_same_month                False                True               False
no_target_history spatial_block           spatial_only no_test_incident_edges               508140        205746                  1                                                         spatial_knn_same_month                False                True               False
no_target_history spatial_block       spatial_temporal              all_edges               508140        279180                  3                spatial_knn_same_month,temporal_self_lag_1,temporal_self_lag_12                 True                True               False
no_target_history spatial_block       spatial_temporal no_test_incident_edges               508140        252339                  3                spatial_knn_same_month,temporal_self_lag_1,temporal_self_lag_12                 True                True               False
no_target_history spatial_block random_spatial_placebo              all_edges               508140        279180                  3 spatial_knn_same_month_random_placebo,temporal_self_lag_1,temporal_self_lag_12                 True                True                True
no_target_history spatial_block random_spatial_placebo no_test_incident_edges               508140        243568                  3 spatial_knn_same_month_random_placebo,temporal_self_lag_1,temporal_self_lag_12                 True                True                True
```

## Feature preprocessing audit preview

Feature imputation and scaling are fit on train nodes only for each split/feature-regime combination.

```text
 feature_regime  split_scheme                                                                  feature  train_median    train_mean   train_std
all_forecasting spatial_block                               target_history__water_drainage_count_lag_1  4.000000e+00  5.148255e+00    5.104562
all_forecasting spatial_block                               target_history__water_drainage_count_lag_2  4.000000e+00  5.106462e+00    5.042885
all_forecasting spatial_block                               target_history__water_drainage_count_lag_3  4.000000e+00  5.081132e+00    4.997454
all_forecasting spatial_block                               target_history__water_drainage_count_lag_6  4.000000e+00  5.057783e+00    4.897671
all_forecasting spatial_block                              target_history__water_drainage_count_lag_12  4.000000e+00  4.866887e+00    4.566316
all_forecasting spatial_block                   target_history__water_drainage_count_roll3_mean_shift1  4.333333e+00  5.135660e+00    4.304965
all_forecasting spatial_block                    target_history__water_drainage_count_roll3_sum_shift1  1.300000e+01  1.512830e+01   12.820516
all_forecasting spatial_block                   target_history__water_drainage_count_roll6_mean_shift1  4.500000e+00  5.146700e+00    4.009442
all_forecasting spatial_block                    target_history__water_drainage_count_roll6_sum_shift1  2.600000e+01  2.942203e+01   23.665339
all_forecasting spatial_block                  target_history__water_drainage_count_roll12_mean_shift1  4.583333e+00  5.178527e+00    3.830589
all_forecasting spatial_block                   target_history__water_drainage_count_roll12_sum_shift1  4.900000e+01  5.532198e+01   43.919407
all_forecasting spatial_block               target_history__water_drainage_count_expanding_mean_shift1  4.866667e+00  5.302842e+00    3.778401
all_forecasting spatial_block                                               target_train_summary__mean  4.833333e+00  5.210764e+00    3.686425
all_forecasting spatial_block                                             target_train_summary__median  4.000000e+00  4.665000e+00    3.438426
all_forecasting spatial_block                                                target_train_summary__p90  8.500000e+00  8.996250e+00    5.726090
all_forecasting spatial_block                                      target_train_summary__positive_rate  9.722222e-01  8.381250e-01    0.295279
all_forecasting spatial_block              reporting_history__total_311_count_non_water_drainage_lag_1  4.300000e+01  4.930075e+01   47.203907
all_forecasting spatial_block              reporting_history__total_311_count_non_water_drainage_lag_2  4.300000e+01  4.881028e+01   46.635502
all_forecasting spatial_block              reporting_history__total_311_count_non_water_drainage_lag_3  4.300000e+01  4.871736e+01   46.365837
all_forecasting spatial_block             reporting_history__total_311_count_non_water_drainage_lag_12  4.100000e+01  4.626080e+01   42.712704
all_forecasting spatial_block  reporting_history__total_311_count_non_water_drainage_roll3_mean_shift1  4.433333e+01  4.883573e+01   44.991226
all_forecasting spatial_block  reporting_history__total_311_count_non_water_drainage_roll6_mean_shift1  4.450000e+01  4.848863e+01   44.362473
all_forecasting spatial_block reporting_history__total_311_count_non_water_drainage_roll12_mean_shift1  4.483333e+01  4.807013e+01   44.404480
all_forecasting spatial_block                                   requests_history__requests_total_lag_1  4.800000e+01  5.446788e+01   50.282215
all_forecasting spatial_block                                   requests_history__requests_total_lag_3  4.800000e+01  5.385509e+01   49.365654
all_forecasting spatial_block                                  requests_history__requests_total_lag_12  4.600000e+01  5.135410e+01   45.403763
all_forecasting spatial_block                       requests_history__requests_total_roll3_mean_shift1  4.900000e+01  5.397768e+01   48.021931
all_forecasting spatial_block                       requests_history__requests_total_roll6_mean_shift1  4.933333e+01  5.364162e+01   47.305401
all_forecasting spatial_block                      requests_history__requests_total_roll12_mean_shift1  4.966667e+01  5.325338e+01   47.252377
all_forecasting spatial_block                                              svi_primary__svi_percentile  7.178082e-01  6.717038e-01    0.243419
all_forecasting spatial_block                                               svi_primary__svi_score_raw  2.451370e+00  2.479075e+00    0.750677
all_forecasting spatial_block                                      static__log1p_population_total_2021  8.157084e+00  7.989863e+00    1.042116
all_forecasting spatial_block                                              static__log1p_land_area_km2  3.882510e-01  5.071181e-01    0.441121
all_forecasting spatial_block                                         static__log1p_population_density  8.951216e+00  8.690145e+00    1.326585
all_forecasting spatial_block                                         static_spatial__tract_centroid_x  7.627618e+06  7.626040e+06 6529.291016
all_forecasting spatial_block                                         static_spatial__tract_centroid_y  1.245054e+06  1.244221e+06 6201.333984
all_forecasting spatial_block                                       static_spatial__tract_centroid_lon -7.360869e+01 -7.363787e+01    0.089755
all_forecasting spatial_block                                       static_spatial__tract_centroid_lat  4.551492e+01  4.551259e+01    0.048858
all_forecasting spatial_block                                                    calendar__month_is_02  0.000000e+00  9.433962e-02    0.292301
all_forecasting spatial_block                                                    calendar__month_is_03  0.000000e+00  9.433962e-02    0.292301
all_forecasting spatial_block                                                    calendar__month_is_04  0.000000e+00  9.433962e-02    0.292301
all_forecasting spatial_block                                                    calendar__month_is_05  0.000000e+00  9.433962e-02    0.292301
all_forecasting spatial_block                                                    calendar__month_is_06  0.000000e+00  7.547170e-02    0.264151
all_forecasting spatial_block                                                    calendar__month_is_07  0.000000e+00  7.547170e-02    0.264151
all_forecasting spatial_block                                                    calendar__month_is_08  0.000000e+00  7.547170e-02    0.264151
all_forecasting spatial_block                                                    calendar__month_is_09  0.000000e+00  7.547170e-02    0.264151
all_forecasting spatial_block                                                    calendar__month_is_10  0.000000e+00  7.547170e-02    0.264151
all_forecasting spatial_block                                                    calendar__month_is_11  0.000000e+00  7.547170e-02    0.264151
all_forecasting spatial_block                                                    calendar__month_is_12  0.000000e+00  7.547170e-02    0.264151
all_forecasting spatial_block                                                      calendar__month_sin  1.224647e-16  7.041606e-02    0.710264
all_forecasting spatial_block                                                      calendar__month_cos  6.123234e-17 -1.028527e-17    0.700404
all_forecasting spatial_block                                       calendar__period_index_since_start  2.600000e+01  2.600000e+01   15.297058
all_forecasting spatial_block                               target_history__water_drainage_count_lag_1  4.000000e+00  5.148255e+00    5.104562
all_forecasting spatial_block                               target_history__water_drainage_count_lag_2  4.000000e+00  5.106462e+00    5.042885
all_forecasting spatial_block                               target_history__water_drainage_count_lag_3  4.000000e+00  5.081132e+00    4.997454
all_forecasting spatial_block                               target_history__water_drainage_count_lag_6  4.000000e+00  5.057783e+00    4.897671
all_forecasting spatial_block                              target_history__water_drainage_count_lag_12  4.000000e+00  4.866887e+00    4.566316
all_forecasting spatial_block                   target_history__water_drainage_count_roll3_mean_shift1  4.333333e+00  5.135660e+00    4.304965
all_forecasting spatial_block                    target_history__water_drainage_count_roll3_sum_shift1  1.300000e+01  1.512830e+01   12.820516
all_forecasting spatial_block                   target_history__water_drainage_count_roll6_mean_shift1  4.500000e+00  5.146700e+00    4.009442
all_forecasting spatial_block                    target_history__water_drainage_count_roll6_sum_shift1  2.600000e+01  2.942203e+01   23.665339
all_forecasting spatial_block                  target_history__water_drainage_count_roll12_mean_shift1  4.583333e+00  5.178527e+00    3.830589
all_forecasting spatial_block                   target_history__water_drainage_count_roll12_sum_shift1  4.900000e+01  5.532198e+01   43.919407
all_forecasting spatial_block               target_history__water_drainage_count_expanding_mean_shift1  4.866667e+00  5.302842e+00    3.778401
all_forecasting spatial_block                                               target_train_summary__mean  4.833333e+00  5.210764e+00    3.686425
all_forecasting spatial_block                                             target_train_summary__median  4.000000e+00  4.665000e+00    3.438426
all_forecasting spatial_block                                                target_train_summary__p90  8.500000e+00  8.996250e+00    5.726090
all_forecasting spatial_block                                      target_train_summary__positive_rate  9.722222e-01  8.381250e-01    0.295279
all_forecasting spatial_block              reporting_history__total_311_count_non_water_drainage_lag_1  4.300000e+01  4.930075e+01   47.203907
all_forecasting spatial_block              reporting_history__total_311_count_non_water_drainage_lag_2  4.300000e+01  4.881028e+01   46.635502
all_forecasting spatial_block              reporting_history__total_311_count_non_water_drainage_lag_3  4.300000e+01  4.871736e+01   46.365837
all_forecasting spatial_block             reporting_history__total_311_count_non_water_drainage_lag_12  4.100000e+01  4.626080e+01   42.712704
all_forecasting spatial_block  reporting_history__total_311_count_non_water_drainage_roll3_mean_shift1  4.433333e+01  4.883573e+01   44.991226
all_forecasting spatial_block  reporting_history__total_311_count_non_water_drainage_roll6_mean_shift1  4.450000e+01  4.848863e+01   44.362473
all_forecasting spatial_block reporting_history__total_311_count_non_water_drainage_roll12_mean_shift1  4.483333e+01  4.807013e+01   44.404480
all_forecasting spatial_block                                   requests_history__requests_total_lag_1  4.800000e+01  5.446788e+01   50.282215
all_forecasting spatial_block                                   requests_history__requests_total_lag_3  4.800000e+01  5.385509e+01   49.365654
all_forecasting spatial_block                                  requests_history__requests_total_lag_12  4.600000e+01  5.135410e+01   45.403763
all_forecasting spatial_block                       requests_history__requests_total_roll3_mean_shift1  4.900000e+01  5.397768e+01   48.021931
all_forecasting spatial_block                       requests_history__requests_total_roll6_mean_shift1  4.933333e+01  5.364162e+01   47.305401
```

## Output artifacts

| Artifact | Path |
|---|---|
| `metrics` | `urban_graph_benchmark/outputs/mtl_311_water_v0/baselines/G1_spatiotemporal_gnn_spatial_core/metrics.csv` |
| `predictions_validation` | `urban_graph_benchmark/outputs/mtl_311_water_v0/baselines/G1_spatiotemporal_gnn_spatial_core/predictions_validation.parquet` |
| `predictions_test` | `urban_graph_benchmark/outputs/mtl_311_water_v0/baselines/G1_spatiotemporal_gnn_spatial_core/predictions_test.parquet` |
| `predictions_all_evaluated` | `urban_graph_benchmark/outputs/mtl_311_water_v0/baselines/G1_spatiotemporal_gnn_spatial_core/predictions_all_evaluated.parquet` |
| `training_curves` | `urban_graph_benchmark/outputs/mtl_311_water_v0/baselines/G1_spatiotemporal_gnn_spatial_core/training_curves.csv` |
| `trial_audit` | `urban_graph_benchmark/outputs/mtl_311_water_v0/baselines/G1_spatiotemporal_gnn_spatial_core/trial_audit.csv` |
| `model_selection_audit` | `urban_graph_benchmark/outputs/mtl_311_water_v0/baselines/G1_spatiotemporal_gnn_spatial_core/model_selection_audit.csv` |
| `feature_preprocessing_audit` | `urban_graph_benchmark/outputs/mtl_311_water_v0/baselines/G1_spatiotemporal_gnn_spatial_core/feature_preprocessing_audit.csv` |
| `graph_regime_audit` | `urban_graph_benchmark/outputs/mtl_311_water_v0/baselines/G1_spatiotemporal_gnn_spatial_core/graph_regime_audit.csv` |
| `model_metadata` | `urban_graph_benchmark/outputs/mtl_311_water_v0/baselines/G1_spatiotemporal_gnn_spatial_core/model_metadata.json` |
| `baseline_report` | `urban_graph_benchmark/outputs/mtl_311_water_v0/baselines/G1_spatiotemporal_gnn_spatial_core/baseline_report.md` |
| `checkpoints_dir` | `urban_graph_benchmark/outputs/mtl_311_water_v0/baselines/G1_spatiotemporal_gnn_spatial_core/checkpoints` |
| `embeddings_dir` | `urban_graph_benchmark/outputs/mtl_311_water_v0/baselines/G1_spatiotemporal_gnn_spatial_core/embeddings` |

## Benchmark handoff

Temporal graph results should be compared against A0_3_tract_train_mean and A3_all_forecasting RF. Spatial-block graph results should be compared against A3_lagged_reporting_forecasting HGB. The most convincing graph claim would improve both count error and high-burden ranking metrics under spatial-block evaluation.
