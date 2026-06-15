# G1 Spatiotemporal Tract GNN Baseline

Generated at: `2026-06-15T18:19:13.198813+00:00`

Graph artifact directory: `urban_graph_benchmark/outputs/mtl_311_water_v0/graphs/G1_tract_month_graph`

Output directory: `urban_graph_benchmark/outputs/mtl_311_water_v0/baselines/G1_5_validation_sweep_spatial_ndcg/runs/h128_L1_do0p05_layernorm_res_relsum_manual_lr0p001_wd0p0001`

## Purpose

G1 is the first controlled graph-value test after the frozen A0--A3 tabular benchmark layer. It evaluates whether message passing over tract topology/time improves prediction and ranking of reported water/drainage 311 burden beyond feature-parity tabular baselines.

## Important interpretation rule

A graph result is meaningful only relative to A3 feature-parity baselines and to no-edge neural controls. Beating raw SVI or weak naive baselines is not enough after A3.

## Training configuration

Checkpointing and validation selection monitor: `validation_ndcg_at_100` (higher is better).

```json
{
  "hidden_dim": 128,
  "num_layers": 1,
  "dropout": 0.05,
  "activation": "relu",
  "normalization": "layernorm",
  "residual": true,
  "backend": "manual",
  "relation_combine": "sum",
  "max_epochs": 250,
  "patience": 40,
  "learning_rate": 0.001,
  "weight_decay": 0.0001,
  "grad_clip_norm": 5.0,
  "min_delta": 1e-05,
  "seed": 20240610,
  "device": "auto",
  "count_min": 0.0,
  "monitor_metric": "validation_ndcg_at_100",
  "save_checkpoints": true,
  "save_embeddings": "none"
}
```

## Trial summary

```text
                                                                                                                                                  model_name    status  split_scheme  feature_regime            edge_regime       edge_mask_regime  best_epoch    best_monitor_metric  best_monitor_value  best_validation_mae  best_validation_ndcg_at_100  n_edges_used  n_relations_used  elapsed_seconds
                           G1__spatial_block__all_forecasting__no_edges__all_edges__h128_L1_do0p05_layernorm_res_relsum_manual_lr0p001_wd0p0001_seed20240610 completed spatial_block all_forecasting               no_edges              all_edges         248 validation_ndcg_at_100            0.674711             2.635669                     0.674711             0                 0         1.565151
                      G1__spatial_block__all_forecasting__temporal_only__all_edges__h128_L1_do0p05_layernorm_res_relsum_manual_lr0p001_wd0p0001_seed20240610 completed spatial_block all_forecasting          temporal_only              all_edges         157 validation_ndcg_at_100            0.695238             2.635010                     0.695238         50220                 2         2.808355
         G1__spatial_block__all_forecasting__temporal_only__no_test_incident_edges__h128_L1_do0p05_layernorm_res_relsum_manual_lr0p001_wd0p0001_seed20240610 completed spatial_block all_forecasting          temporal_only no_test_incident_edges         157 validation_ndcg_at_100            0.694809             2.635451                     0.694809         46593                 2         2.803109
                   G1__spatial_block__all_forecasting__spatial_temporal__all_edges__h128_L1_do0p05_layernorm_res_relsum_manual_lr0p001_wd0p0001_seed20240610 completed spatial_block all_forecasting       spatial_temporal              all_edges         246 validation_ndcg_at_100            0.678673             2.609206                     0.678673        279180                 3         6.733640
      G1__spatial_block__all_forecasting__spatial_temporal__no_test_incident_edges__h128_L1_do0p05_layernorm_res_relsum_manual_lr0p001_wd0p0001_seed20240610 completed spatial_block all_forecasting       spatial_temporal no_test_incident_edges          77 validation_ndcg_at_100            0.647831             2.636653                     0.647831        252339                 3         3.101368
             G1__spatial_block__all_forecasting__random_spatial_placebo__all_edges__h128_L1_do0p05_layernorm_res_relsum_manual_lr0p001_wd0p0001_seed20240610 completed spatial_block all_forecasting random_spatial_placebo              all_edges          55 validation_ndcg_at_100            0.621906             2.661458                     0.621906        279180                 3         2.704825
G1__spatial_block__all_forecasting__random_spatial_placebo__no_test_incident_edges__h128_L1_do0p05_layernorm_res_relsum_manual_lr0p001_wd0p0001_seed20240610 completed spatial_block all_forecasting random_spatial_placebo no_test_incident_edges          47 validation_ndcg_at_100            0.612269             2.657449                     0.612269        243568                 3         2.345286
```

## Validation-only model selection

Trial representatives are selected using `validation_ndcg_at_100` on validation nodes only. Test metrics are reported after selection.

```text
                                                                                                                                                  model_name  split_scheme  feature_regime            edge_regime       edge_mask_regime       selection_metric  selection_metric_value  validation_mae  validation_spearman  validation_ndcg_at_10  validation_ndcg_at_25  validation_ndcg_at_50  validation_ndcg_at_100  validation_top10_overlap_rate  validation_top25_overlap_rate  validation_top50_overlap_rate  validation_top100_overlap_rate  validation_top_5pct_overlap_rate  validation_top_10pct_overlap_rate  test_mae  test_spearman  test_ndcg_at_10  test_ndcg_at_25  test_ndcg_at_50  test_ndcg_at_100  test_top10_overlap_rate  test_top25_overlap_rate  test_top50_overlap_rate  test_top100_overlap_rate  test_top_5pct_overlap_rate  test_top_10pct_overlap_rate  selected_overall_for_split  selected_for_feature_regime  selected_for_test_summary
                      G1__spatial_block__all_forecasting__temporal_only__all_edges__h128_L1_do0p05_layernorm_res_relsum_manual_lr0p001_wd0p0001_seed20240610 spatial_block all_forecasting          temporal_only              all_edges validation_ndcg_at_100                0.695238        2.635010             0.675085               0.600302               0.610512               0.644560                0.695238                            0.2                           0.20                           0.28                            0.35                          0.402985                           0.522388  2.346261       0.771959         0.692141         0.737622         0.782232          0.826168                      0.1                     0.28                     0.56                      0.66                    0.673077                     0.628019                        True                         True                       True
         G1__spatial_block__all_forecasting__temporal_only__no_test_incident_edges__h128_L1_do0p05_layernorm_res_relsum_manual_lr0p001_wd0p0001_seed20240610 spatial_block all_forecasting          temporal_only no_test_incident_edges validation_ndcg_at_100                0.694809        2.635451             0.675027               0.600244               0.600834               0.644050                0.694809                            0.2                           0.20                           0.28                            0.35                          0.402985                           0.524254  2.647294       0.733266         0.500548         0.582128         0.658768          0.695872                      0.1                     0.28                     0.40                      0.51                    0.519231                     0.574879                       False                        False                      False
                   G1__spatial_block__all_forecasting__spatial_temporal__all_edges__h128_L1_do0p05_layernorm_res_relsum_manual_lr0p001_wd0p0001_seed20240610 spatial_block all_forecasting       spatial_temporal              all_edges validation_ndcg_at_100                0.678673        2.609206             0.676135               0.592728               0.602812               0.613479                0.678673                            0.3                           0.20                           0.22                            0.33                          0.440299                           0.526119  2.321852       0.767798         0.621234         0.688447         0.761604          0.802370                      0.0                     0.28                     0.58                      0.67                    0.673077                     0.632850                       False                        False                      False
                           G1__spatial_block__all_forecasting__no_edges__all_edges__h128_L1_do0p05_layernorm_res_relsum_manual_lr0p001_wd0p0001_seed20240610 spatial_block all_forecasting               no_edges              all_edges validation_ndcg_at_100                0.674711        2.635669             0.675850               0.605463               0.640931               0.653176                0.674711                            0.1                           0.24                           0.32                            0.31                          0.410448                           0.526119  2.352912       0.772845         0.617723         0.676406         0.755200          0.796551                      0.1                     0.24                     0.58                      0.65                    0.644231                     0.628019                       False                        False                      False
      G1__spatial_block__all_forecasting__spatial_temporal__no_test_incident_edges__h128_L1_do0p05_layernorm_res_relsum_manual_lr0p001_wd0p0001_seed20240610 spatial_block all_forecasting       spatial_temporal no_test_incident_edges validation_ndcg_at_100                0.647831        2.636653             0.672921               0.533260               0.568342               0.611815                0.647831                            0.2                           0.20                           0.26                            0.30                          0.436567                           0.520522  2.638707       0.736332         0.425559         0.445064         0.542114          0.615002                      0.1                     0.12                     0.32                      0.44                    0.461538                     0.521739                       False                        False                      False
             G1__spatial_block__all_forecasting__random_spatial_placebo__all_edges__h128_L1_do0p05_layernorm_res_relsum_manual_lr0p001_wd0p0001_seed20240610 spatial_block all_forecasting random_spatial_placebo              all_edges validation_ndcg_at_100                0.621906        2.661458             0.665255               0.488203               0.556023               0.566049                0.621906                            0.2                           0.28                           0.22                            0.30                          0.417910                           0.522388  2.421082       0.765856         0.555446         0.663131         0.733755          0.779185                      0.0                     0.24                     0.50                      0.65                    0.644231                     0.613527                       False                        False                      False
G1__spatial_block__all_forecasting__random_spatial_placebo__no_test_incident_edges__h128_L1_do0p05_layernorm_res_relsum_manual_lr0p001_wd0p0001_seed20240610 spatial_block all_forecasting random_spatial_placebo no_test_incident_edges validation_ndcg_at_100                0.612269        2.657449             0.663897               0.433989               0.527193               0.563822                0.612269                            0.1                           0.24                           0.24                            0.29                          0.410448                           0.524254  2.673974       0.734831         0.229696         0.349281         0.454124          0.548497                      0.0                     0.16                     0.30                      0.41                    0.423077                     0.516908                       False                        False                      False
```

## Compact metrics for selected models

```text
          model_stage                                                                                                                             model_name  split_scheme split_name  feature_regime   edge_regime edge_mask_regime                     metric_name  metric_value  higher_is_better  n_rows     seed
G1_spatiotemporal_gnn G1__spatial_block__all_forecasting__temporal_only__all_edges__h128_L1_do0p05_layernorm_res_relsum_manual_lr0p001_wd0p0001_seed20240610 spatial_block       test all_forecasting temporal_only        all_edges                      count__mae      2.346261             False    2067 20240610
G1_spatiotemporal_gnn G1__spatial_block__all_forecasting__temporal_only__all_edges__h128_L1_do0p05_layernorm_res_relsum_manual_lr0p001_wd0p0001_seed20240610 spatial_block       test all_forecasting temporal_only        all_edges    count__mean_poisson_deviance      2.292293             False    2067 20240610
G1_spatiotemporal_gnn G1__spatial_block__all_forecasting__temporal_only__all_edges__h128_L1_do0p05_layernorm_res_relsum_manual_lr0p001_wd0p0001_seed20240610 spatial_block       test all_forecasting temporal_only        all_edges                     count__rmse      3.817675             False    2067 20240610
G1_spatiotemporal_gnn G1__spatial_block__all_forecasting__temporal_only__all_edges__h128_L1_do0p05_layernorm_res_relsum_manual_lr0p001_wd0p0001_seed20240610 spatial_block       test all_forecasting temporal_only        all_edges           ranking__kendall_corr      0.602758              True    2067 20240610
G1_spatiotemporal_gnn G1__spatial_block__all_forecasting__temporal_only__all_edges__h128_L1_do0p05_layernorm_res_relsum_manual_lr0p001_wd0p0001_seed20240610 spatial_block       test all_forecasting temporal_only        all_edges             ranking__ndcg_at_10      0.692141              True    2067 20240610
G1_spatiotemporal_gnn G1__spatial_block__all_forecasting__temporal_only__all_edges__h128_L1_do0p05_layernorm_res_relsum_manual_lr0p001_wd0p0001_seed20240610 spatial_block       test all_forecasting temporal_only        all_edges            ranking__ndcg_at_100      0.826168              True    2067 20240610
G1_spatiotemporal_gnn G1__spatial_block__all_forecasting__temporal_only__all_edges__h128_L1_do0p05_layernorm_res_relsum_manual_lr0p001_wd0p0001_seed20240610 spatial_block       test all_forecasting temporal_only        all_edges             ranking__ndcg_at_25      0.737622              True    2067 20240610
G1_spatiotemporal_gnn G1__spatial_block__all_forecasting__temporal_only__all_edges__h128_L1_do0p05_layernorm_res_relsum_manual_lr0p001_wd0p0001_seed20240610 spatial_block       test all_forecasting temporal_only        all_edges             ranking__ndcg_at_50      0.782232              True    2067 20240610
G1_spatiotemporal_gnn G1__spatial_block__all_forecasting__temporal_only__all_edges__h128_L1_do0p05_layernorm_res_relsum_manual_lr0p001_wd0p0001_seed20240610 spatial_block       test all_forecasting temporal_only        all_edges          ranking__spearman_corr      0.771959              True    2067 20240610
G1_spatiotemporal_gnn G1__spatial_block__all_forecasting__temporal_only__all_edges__h128_L1_do0p05_layernorm_res_relsum_manual_lr0p001_wd0p0001_seed20240610 spatial_block       test all_forecasting temporal_only        all_edges    ranking__top100_overlap_rate      0.660000              True    2067 20240610
G1_spatiotemporal_gnn G1__spatial_block__all_forecasting__temporal_only__all_edges__h128_L1_do0p05_layernorm_res_relsum_manual_lr0p001_wd0p0001_seed20240610 spatial_block       test all_forecasting temporal_only        all_edges     ranking__top10_overlap_rate      0.100000              True    2067 20240610
G1_spatiotemporal_gnn G1__spatial_block__all_forecasting__temporal_only__all_edges__h128_L1_do0p05_layernorm_res_relsum_manual_lr0p001_wd0p0001_seed20240610 spatial_block       test all_forecasting temporal_only        all_edges     ranking__top25_overlap_rate      0.280000              True    2067 20240610
G1_spatiotemporal_gnn G1__spatial_block__all_forecasting__temporal_only__all_edges__h128_L1_do0p05_layernorm_res_relsum_manual_lr0p001_wd0p0001_seed20240610 spatial_block       test all_forecasting temporal_only        all_edges     ranking__top50_overlap_rate      0.560000              True    2067 20240610
G1_spatiotemporal_gnn G1__spatial_block__all_forecasting__temporal_only__all_edges__h128_L1_do0p05_layernorm_res_relsum_manual_lr0p001_wd0p0001_seed20240610 spatial_block       test all_forecasting temporal_only        all_edges ranking__top_10pct_overlap_rate      0.628019              True    2067 20240610
G1_spatiotemporal_gnn G1__spatial_block__all_forecasting__temporal_only__all_edges__h128_L1_do0p05_layernorm_res_relsum_manual_lr0p001_wd0p0001_seed20240610 spatial_block       test all_forecasting temporal_only        all_edges  ranking__top_5pct_overlap_rate      0.673077              True    2067 20240610
G1_spatiotemporal_gnn G1__spatial_block__all_forecasting__temporal_only__all_edges__h128_L1_do0p05_layernorm_res_relsum_manual_lr0p001_wd0p0001_seed20240610 spatial_block      train all_forecasting temporal_only        all_edges                      count__mae      2.201071             False   21200 20240610
G1_spatiotemporal_gnn G1__spatial_block__all_forecasting__temporal_only__all_edges__h128_L1_do0p05_layernorm_res_relsum_manual_lr0p001_wd0p0001_seed20240610 spatial_block      train all_forecasting temporal_only        all_edges    count__mean_poisson_deviance      1.910975             False   21200 20240610
G1_spatiotemporal_gnn G1__spatial_block__all_forecasting__temporal_only__all_edges__h128_L1_do0p05_layernorm_res_relsum_manual_lr0p001_wd0p0001_seed20240610 spatial_block      train all_forecasting temporal_only        all_edges                     count__rmse      3.499104             False   21200 20240610
G1_spatiotemporal_gnn G1__spatial_block__all_forecasting__temporal_only__all_edges__h128_L1_do0p05_layernorm_res_relsum_manual_lr0p001_wd0p0001_seed20240610 spatial_block      train all_forecasting temporal_only        all_edges           ranking__kendall_corr      0.614369              True   21200 20240610
G1_spatiotemporal_gnn G1__spatial_block__all_forecasting__temporal_only__all_edges__h128_L1_do0p05_layernorm_res_relsum_manual_lr0p001_wd0p0001_seed20240610 spatial_block      train all_forecasting temporal_only        all_edges             ranking__ndcg_at_10      0.478286              True   21200 20240610
G1_spatiotemporal_gnn G1__spatial_block__all_forecasting__temporal_only__all_edges__h128_L1_do0p05_layernorm_res_relsum_manual_lr0p001_wd0p0001_seed20240610 spatial_block      train all_forecasting temporal_only        all_edges            ranking__ndcg_at_100      0.618193              True   21200 20240610
G1_spatiotemporal_gnn G1__spatial_block__all_forecasting__temporal_only__all_edges__h128_L1_do0p05_layernorm_res_relsum_manual_lr0p001_wd0p0001_seed20240610 spatial_block      train all_forecasting temporal_only        all_edges             ranking__ndcg_at_25      0.521225              True   21200 20240610
G1_spatiotemporal_gnn G1__spatial_block__all_forecasting__temporal_only__all_edges__h128_L1_do0p05_layernorm_res_relsum_manual_lr0p001_wd0p0001_seed20240610 spatial_block      train all_forecasting temporal_only        all_edges             ranking__ndcg_at_50      0.573607              True   21200 20240610
G1_spatiotemporal_gnn G1__spatial_block__all_forecasting__temporal_only__all_edges__h128_L1_do0p05_layernorm_res_relsum_manual_lr0p001_wd0p0001_seed20240610 spatial_block      train all_forecasting temporal_only        all_edges          ranking__spearman_corr      0.782918              True   21200 20240610
G1_spatiotemporal_gnn G1__spatial_block__all_forecasting__temporal_only__all_edges__h128_L1_do0p05_layernorm_res_relsum_manual_lr0p001_wd0p0001_seed20240610 spatial_block      train all_forecasting temporal_only        all_edges    ranking__top100_overlap_rate      0.240000              True   21200 20240610
G1_spatiotemporal_gnn G1__spatial_block__all_forecasting__temporal_only__all_edges__h128_L1_do0p05_layernorm_res_relsum_manual_lr0p001_wd0p0001_seed20240610 spatial_block      train all_forecasting temporal_only        all_edges     ranking__top10_overlap_rate      0.100000              True   21200 20240610
G1_spatiotemporal_gnn G1__spatial_block__all_forecasting__temporal_only__all_edges__h128_L1_do0p05_layernorm_res_relsum_manual_lr0p001_wd0p0001_seed20240610 spatial_block      train all_forecasting temporal_only        all_edges     ranking__top25_overlap_rate      0.200000              True   21200 20240610
G1_spatiotemporal_gnn G1__spatial_block__all_forecasting__temporal_only__all_edges__h128_L1_do0p05_layernorm_res_relsum_manual_lr0p001_wd0p0001_seed20240610 spatial_block      train all_forecasting temporal_only        all_edges     ranking__top50_overlap_rate      0.200000              True   21200 20240610
G1_spatiotemporal_gnn G1__spatial_block__all_forecasting__temporal_only__all_edges__h128_L1_do0p05_layernorm_res_relsum_manual_lr0p001_wd0p0001_seed20240610 spatial_block      train all_forecasting temporal_only        all_edges ranking__top_10pct_overlap_rate      0.556604              True   21200 20240610
G1_spatiotemporal_gnn G1__spatial_block__all_forecasting__temporal_only__all_edges__h128_L1_do0p05_layernorm_res_relsum_manual_lr0p001_wd0p0001_seed20240610 spatial_block      train all_forecasting temporal_only        all_edges  ranking__top_5pct_overlap_rate      0.481132              True   21200 20240610
G1_spatiotemporal_gnn G1__spatial_block__all_forecasting__temporal_only__all_edges__h128_L1_do0p05_layernorm_res_relsum_manual_lr0p001_wd0p0001_seed20240610 spatial_block validation all_forecasting temporal_only        all_edges                      count__mae      2.635010             False    5353 20240610
G1_spatiotemporal_gnn G1__spatial_block__all_forecasting__temporal_only__all_edges__h128_L1_do0p05_layernorm_res_relsum_manual_lr0p001_wd0p0001_seed20240610 spatial_block validation all_forecasting temporal_only        all_edges    count__mean_poisson_deviance      2.353783             False    5353 20240610
G1_spatiotemporal_gnn G1__spatial_block__all_forecasting__temporal_only__all_edges__h128_L1_do0p05_layernorm_res_relsum_manual_lr0p001_wd0p0001_seed20240610 spatial_block validation all_forecasting temporal_only        all_edges                     count__rmse      3.955355             False    5353 20240610
G1_spatiotemporal_gnn G1__spatial_block__all_forecasting__temporal_only__all_edges__h128_L1_do0p05_layernorm_res_relsum_manual_lr0p001_wd0p0001_seed20240610 spatial_block validation all_forecasting temporal_only        all_edges           ranking__kendall_corr      0.507038              True    5353 20240610
G1_spatiotemporal_gnn G1__spatial_block__all_forecasting__temporal_only__all_edges__h128_L1_do0p05_layernorm_res_relsum_manual_lr0p001_wd0p0001_seed20240610 spatial_block validation all_forecasting temporal_only        all_edges             ranking__ndcg_at_10      0.600302              True    5353 20240610
G1_spatiotemporal_gnn G1__spatial_block__all_forecasting__temporal_only__all_edges__h128_L1_do0p05_layernorm_res_relsum_manual_lr0p001_wd0p0001_seed20240610 spatial_block validation all_forecasting temporal_only        all_edges            ranking__ndcg_at_100      0.695238              True    5353 20240610
G1_spatiotemporal_gnn G1__spatial_block__all_forecasting__temporal_only__all_edges__h128_L1_do0p05_layernorm_res_relsum_manual_lr0p001_wd0p0001_seed20240610 spatial_block validation all_forecasting temporal_only        all_edges             ranking__ndcg_at_25      0.610512              True    5353 20240610
G1_spatiotemporal_gnn G1__spatial_block__all_forecasting__temporal_only__all_edges__h128_L1_do0p05_layernorm_res_relsum_manual_lr0p001_wd0p0001_seed20240610 spatial_block validation all_forecasting temporal_only        all_edges             ranking__ndcg_at_50      0.644560              True    5353 20240610
G1_spatiotemporal_gnn G1__spatial_block__all_forecasting__temporal_only__all_edges__h128_L1_do0p05_layernorm_res_relsum_manual_lr0p001_wd0p0001_seed20240610 spatial_block validation all_forecasting temporal_only        all_edges          ranking__spearman_corr      0.675085              True    5353 20240610
G1_spatiotemporal_gnn G1__spatial_block__all_forecasting__temporal_only__all_edges__h128_L1_do0p05_layernorm_res_relsum_manual_lr0p001_wd0p0001_seed20240610 spatial_block validation all_forecasting temporal_only        all_edges    ranking__top100_overlap_rate      0.350000              True    5353 20240610
G1_spatiotemporal_gnn G1__spatial_block__all_forecasting__temporal_only__all_edges__h128_L1_do0p05_layernorm_res_relsum_manual_lr0p001_wd0p0001_seed20240610 spatial_block validation all_forecasting temporal_only        all_edges     ranking__top10_overlap_rate      0.200000              True    5353 20240610
G1_spatiotemporal_gnn G1__spatial_block__all_forecasting__temporal_only__all_edges__h128_L1_do0p05_layernorm_res_relsum_manual_lr0p001_wd0p0001_seed20240610 spatial_block validation all_forecasting temporal_only        all_edges     ranking__top25_overlap_rate      0.200000              True    5353 20240610
G1_spatiotemporal_gnn G1__spatial_block__all_forecasting__temporal_only__all_edges__h128_L1_do0p05_layernorm_res_relsum_manual_lr0p001_wd0p0001_seed20240610 spatial_block validation all_forecasting temporal_only        all_edges     ranking__top50_overlap_rate      0.280000              True    5353 20240610
G1_spatiotemporal_gnn G1__spatial_block__all_forecasting__temporal_only__all_edges__h128_L1_do0p05_layernorm_res_relsum_manual_lr0p001_wd0p0001_seed20240610 spatial_block validation all_forecasting temporal_only        all_edges ranking__top_10pct_overlap_rate      0.522388              True    5353 20240610
G1_spatiotemporal_gnn G1__spatial_block__all_forecasting__temporal_only__all_edges__h128_L1_do0p05_layernorm_res_relsum_manual_lr0p001_wd0p0001_seed20240610 spatial_block validation all_forecasting temporal_only        all_edges  ranking__top_5pct_overlap_rate      0.402985              True    5353 20240610
```

## Graph-regime audit

Edge masks define the message-passing graph used by each trial. `all_edges` is transductive; `no_test_incident_edges` removes all edges touching test nodes; `train_train_edges` uses only train-train edges when requested.

```text
 feature_regime  split_scheme            edge_regime       edge_mask_regime  n_edges_total_graph  n_edges_used  n_edge_types_used                                                                edge_types_used  uses_temporal_edges  uses_spatial_edges  uses_placebo_edges
all_forecasting spatial_block               no_edges              all_edges               508140             0                  0                                                                                               False               False               False
all_forecasting spatial_block          temporal_only              all_edges               508140         50220                  2                                       temporal_self_lag_1,temporal_self_lag_12                 True               False               False
all_forecasting spatial_block          temporal_only no_test_incident_edges               508140         46593                  2                                       temporal_self_lag_1,temporal_self_lag_12                 True               False               False
all_forecasting spatial_block       spatial_temporal              all_edges               508140        279180                  3                spatial_knn_same_month,temporal_self_lag_1,temporal_self_lag_12                 True                True               False
all_forecasting spatial_block       spatial_temporal no_test_incident_edges               508140        252339                  3                spatial_knn_same_month,temporal_self_lag_1,temporal_self_lag_12                 True                True               False
all_forecasting spatial_block random_spatial_placebo              all_edges               508140        279180                  3 spatial_knn_same_month_random_placebo,temporal_self_lag_1,temporal_self_lag_12                 True                True                True
all_forecasting spatial_block random_spatial_placebo no_test_incident_edges               508140        243568                  3 spatial_knn_same_month_random_placebo,temporal_self_lag_1,temporal_self_lag_12                 True                True                True
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
| `metrics` | `urban_graph_benchmark/outputs/mtl_311_water_v0/baselines/G1_5_validation_sweep_spatial_ndcg/runs/h128_L1_do0p05_layernorm_res_relsum_manual_lr0p001_wd0p0001/metrics.csv` |
| `predictions_validation` | `urban_graph_benchmark/outputs/mtl_311_water_v0/baselines/G1_5_validation_sweep_spatial_ndcg/runs/h128_L1_do0p05_layernorm_res_relsum_manual_lr0p001_wd0p0001/predictions_validation.parquet` |
| `predictions_test` | `urban_graph_benchmark/outputs/mtl_311_water_v0/baselines/G1_5_validation_sweep_spatial_ndcg/runs/h128_L1_do0p05_layernorm_res_relsum_manual_lr0p001_wd0p0001/predictions_test.parquet` |
| `predictions_all_evaluated` | `urban_graph_benchmark/outputs/mtl_311_water_v0/baselines/G1_5_validation_sweep_spatial_ndcg/runs/h128_L1_do0p05_layernorm_res_relsum_manual_lr0p001_wd0p0001/predictions_all_evaluated.parquet` |
| `training_curves` | `urban_graph_benchmark/outputs/mtl_311_water_v0/baselines/G1_5_validation_sweep_spatial_ndcg/runs/h128_L1_do0p05_layernorm_res_relsum_manual_lr0p001_wd0p0001/training_curves.csv` |
| `trial_audit` | `urban_graph_benchmark/outputs/mtl_311_water_v0/baselines/G1_5_validation_sweep_spatial_ndcg/runs/h128_L1_do0p05_layernorm_res_relsum_manual_lr0p001_wd0p0001/trial_audit.csv` |
| `model_selection_audit` | `urban_graph_benchmark/outputs/mtl_311_water_v0/baselines/G1_5_validation_sweep_spatial_ndcg/runs/h128_L1_do0p05_layernorm_res_relsum_manual_lr0p001_wd0p0001/model_selection_audit.csv` |
| `feature_preprocessing_audit` | `urban_graph_benchmark/outputs/mtl_311_water_v0/baselines/G1_5_validation_sweep_spatial_ndcg/runs/h128_L1_do0p05_layernorm_res_relsum_manual_lr0p001_wd0p0001/feature_preprocessing_audit.csv` |
| `graph_regime_audit` | `urban_graph_benchmark/outputs/mtl_311_water_v0/baselines/G1_5_validation_sweep_spatial_ndcg/runs/h128_L1_do0p05_layernorm_res_relsum_manual_lr0p001_wd0p0001/graph_regime_audit.csv` |
| `model_metadata` | `urban_graph_benchmark/outputs/mtl_311_water_v0/baselines/G1_5_validation_sweep_spatial_ndcg/runs/h128_L1_do0p05_layernorm_res_relsum_manual_lr0p001_wd0p0001/model_metadata.json` |
| `baseline_report` | `urban_graph_benchmark/outputs/mtl_311_water_v0/baselines/G1_5_validation_sweep_spatial_ndcg/runs/h128_L1_do0p05_layernorm_res_relsum_manual_lr0p001_wd0p0001/baseline_report.md` |
| `checkpoints_dir` | `urban_graph_benchmark/outputs/mtl_311_water_v0/baselines/G1_5_validation_sweep_spatial_ndcg/runs/h128_L1_do0p05_layernorm_res_relsum_manual_lr0p001_wd0p0001/checkpoints` |
| `embeddings_dir` | `urban_graph_benchmark/outputs/mtl_311_water_v0/baselines/G1_5_validation_sweep_spatial_ndcg/runs/h128_L1_do0p05_layernorm_res_relsum_manual_lr0p001_wd0p0001/embeddings` |

## Benchmark handoff

Temporal graph results should be compared against A0_3_tract_train_mean and A3_all_forecasting RF. Spatial-block graph results should be compared against A3_lagged_reporting_forecasting HGB. The most convincing graph claim would improve both count error and high-burden ranking metrics under spatial-block evaluation.
