# G1 Spatiotemporal Tract GNN Baseline

Generated at: `2026-06-15T18:15:26.001742+00:00`

Graph artifact directory: `urban_graph_benchmark/outputs/mtl_311_water_v0/graphs/G1_tract_month_graph`

Output directory: `urban_graph_benchmark/outputs/mtl_311_water_v0/baselines/G1_5_validation_sweep_spatial_ndcg/runs/h64_L3_do0_layernorm_res_relsum_manual_lr0p001_wd0p0001`

## Purpose

G1 is the first controlled graph-value test after the frozen A0--A3 tabular benchmark layer. It evaluates whether message passing over tract topology/time improves prediction and ranking of reported water/drainage 311 burden beyond feature-parity tabular baselines.

## Important interpretation rule

A graph result is meaningful only relative to A3 feature-parity baselines and to no-edge neural controls. Beating raw SVI or weak naive baselines is not enough after A3.

## Training configuration

Checkpointing and validation selection monitor: `validation_ndcg_at_100` (higher is better).

```json
{
  "hidden_dim": 64,
  "num_layers": 3,
  "dropout": 0.0,
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
                           G1__spatial_block__all_forecasting__no_edges__all_edges__h64_L3_do0_layernorm_res_relsum_manual_lr0p001_wd0p0001_seed20240610 completed spatial_block all_forecasting               no_edges              all_edges         164 validation_ndcg_at_100            0.654699             2.628653                     0.654699             0                 0         1.346956
                      G1__spatial_block__all_forecasting__temporal_only__all_edges__h64_L3_do0_layernorm_res_relsum_manual_lr0p001_wd0p0001_seed20240610 completed spatial_block all_forecasting          temporal_only              all_edges         221 validation_ndcg_at_100            0.670504             2.635335                     0.670504         50220                 2         3.799880
         G1__spatial_block__all_forecasting__temporal_only__no_test_incident_edges__h64_L3_do0_layernorm_res_relsum_manual_lr0p001_wd0p0001_seed20240610 completed spatial_block all_forecasting          temporal_only no_test_incident_edges         197 validation_ndcg_at_100            0.667390             2.634580                     0.667390         46593                 2         3.598093
                   G1__spatial_block__all_forecasting__spatial_temporal__all_edges__h64_L3_do0_layernorm_res_relsum_manual_lr0p001_wd0p0001_seed20240610 completed spatial_block all_forecasting       spatial_temporal              all_edges         107 validation_ndcg_at_100            0.644932             2.668042                     0.644932        279180                 3         5.220951
      G1__spatial_block__all_forecasting__spatial_temporal__no_test_incident_edges__h64_L3_do0_layernorm_res_relsum_manual_lr0p001_wd0p0001_seed20240610 completed spatial_block all_forecasting       spatial_temporal no_test_incident_edges         208 validation_ndcg_at_100            0.664066             2.645737                     0.664066        252339                 3         8.176522
             G1__spatial_block__all_forecasting__random_spatial_placebo__all_edges__h64_L3_do0_layernorm_res_relsum_manual_lr0p001_wd0p0001_seed20240610 completed spatial_block all_forecasting random_spatial_placebo              all_edges          85 validation_ndcg_at_100            0.653900             2.674706                     0.653900        279180                 3         4.452938
G1__spatial_block__all_forecasting__random_spatial_placebo__no_test_incident_edges__h64_L3_do0_layernorm_res_relsum_manual_lr0p001_wd0p0001_seed20240610 completed spatial_block all_forecasting random_spatial_placebo no_test_incident_edges          92 validation_ndcg_at_100            0.655891             2.673733                     0.655891        243568                 3         4.376927
```

## Validation-only model selection

Trial representatives are selected using `validation_ndcg_at_100` on validation nodes only. Test metrics are reported after selection.

```text
                                                                                                                                              model_name  split_scheme  feature_regime            edge_regime       edge_mask_regime       selection_metric  selection_metric_value  validation_mae  validation_spearman  validation_ndcg_at_10  validation_ndcg_at_25  validation_ndcg_at_50  validation_ndcg_at_100  validation_top10_overlap_rate  validation_top25_overlap_rate  validation_top50_overlap_rate  validation_top100_overlap_rate  validation_top_5pct_overlap_rate  validation_top_10pct_overlap_rate  test_mae  test_spearman  test_ndcg_at_10  test_ndcg_at_25  test_ndcg_at_50  test_ndcg_at_100  test_top10_overlap_rate  test_top25_overlap_rate  test_top50_overlap_rate  test_top100_overlap_rate  test_top_5pct_overlap_rate  test_top_10pct_overlap_rate  selected_overall_for_split  selected_for_feature_regime  selected_for_test_summary
                      G1__spatial_block__all_forecasting__temporal_only__all_edges__h64_L3_do0_layernorm_res_relsum_manual_lr0p001_wd0p0001_seed20240610 spatial_block all_forecasting          temporal_only              all_edges validation_ndcg_at_100                0.670504        2.635335             0.665190               0.542751               0.574941               0.605958                0.670504                            0.1                           0.20                           0.22                            0.31                          0.406716                           0.524254  2.348636       0.764371         0.649296         0.697657         0.764277          0.799381                      0.2                     0.32                     0.58                      0.65                    0.644231                     0.642512                        True                         True                       True
         G1__spatial_block__all_forecasting__temporal_only__no_test_incident_edges__h64_L3_do0_layernorm_res_relsum_manual_lr0p001_wd0p0001_seed20240610 spatial_block all_forecasting          temporal_only no_test_incident_edges validation_ndcg_at_100                0.667390        2.634580             0.665890               0.523262               0.566056               0.600287                0.667390                            0.1                           0.20                           0.22                            0.30                          0.402985                           0.522388  2.885024       0.669019         0.438877         0.516328         0.538804          0.575949                      0.0                     0.20                     0.34                      0.40                    0.403846                     0.478261                       False                        False                      False
      G1__spatial_block__all_forecasting__spatial_temporal__no_test_incident_edges__h64_L3_do0_layernorm_res_relsum_manual_lr0p001_wd0p0001_seed20240610 spatial_block all_forecasting       spatial_temporal no_test_incident_edges validation_ndcg_at_100                0.664066        2.645737             0.660730               0.545539               0.567289               0.601752                0.664066                            0.3                           0.20                           0.26                            0.29                          0.432836                           0.498134  2.914173       0.688431         0.273272         0.277501         0.389340          0.447495                      0.1                     0.04                     0.12                      0.21                    0.230769                     0.458937                       False                        False                      False
G1__spatial_block__all_forecasting__random_spatial_placebo__no_test_incident_edges__h64_L3_do0_layernorm_res_relsum_manual_lr0p001_wd0p0001_seed20240610 spatial_block all_forecasting random_spatial_placebo no_test_incident_edges validation_ndcg_at_100                0.655891        2.673733             0.656855               0.549861               0.566360               0.620997                0.655891                            0.1                           0.20                           0.32                            0.28                          0.417910                           0.490672  2.930548       0.674008         0.199859         0.249648         0.317322          0.408042                      0.1                     0.04                     0.10                      0.20                    0.201923                     0.396135                       False                        False                      False
                           G1__spatial_block__all_forecasting__no_edges__all_edges__h64_L3_do0_layernorm_res_relsum_manual_lr0p001_wd0p0001_seed20240610 spatial_block all_forecasting               no_edges              all_edges validation_ndcg_at_100                0.654699        2.628653             0.672393               0.512994               0.539337               0.603026                0.654699                            0.2                           0.12                           0.32                            0.29                          0.447761                           0.516791  2.349807       0.767166         0.684505         0.707053         0.784305          0.812191                      0.2                     0.24                     0.56                      0.64                    0.634615                     0.623188                       False                        False                      False
             G1__spatial_block__all_forecasting__random_spatial_placebo__all_edges__h64_L3_do0_layernorm_res_relsum_manual_lr0p001_wd0p0001_seed20240610 spatial_block all_forecasting random_spatial_placebo              all_edges validation_ndcg_at_100                0.653900        2.674706             0.656981               0.525425               0.554025               0.617857                0.653900                            0.1                           0.16                           0.32                            0.28                          0.421642                           0.485075  2.404253       0.759065         0.572202         0.682138         0.749288          0.796577                      0.0                     0.36                     0.56                      0.65                    0.653846                     0.589372                       False                        False                      False
                   G1__spatial_block__all_forecasting__spatial_temporal__all_edges__h64_L3_do0_layernorm_res_relsum_manual_lr0p001_wd0p0001_seed20240610 spatial_block all_forecasting       spatial_temporal              all_edges validation_ndcg_at_100                0.644932        2.668042             0.658489               0.543078               0.615874               0.612686                0.644932                            0.1                           0.28                           0.24                            0.25                          0.402985                           0.507463  2.365707       0.761900         0.562526         0.672437         0.761251          0.798828                      0.1                     0.36                     0.58                      0.67                    0.673077                     0.594203                       False                        False                      False
```

## Compact metrics for selected models

```text
          model_stage                                                                                                                         model_name  split_scheme split_name  feature_regime   edge_regime edge_mask_regime                     metric_name  metric_value  higher_is_better  n_rows     seed
G1_spatiotemporal_gnn G1__spatial_block__all_forecasting__temporal_only__all_edges__h64_L3_do0_layernorm_res_relsum_manual_lr0p001_wd0p0001_seed20240610 spatial_block       test all_forecasting temporal_only        all_edges                      count__mae      2.348636             False    2067 20240610
G1_spatiotemporal_gnn G1__spatial_block__all_forecasting__temporal_only__all_edges__h64_L3_do0_layernorm_res_relsum_manual_lr0p001_wd0p0001_seed20240610 spatial_block       test all_forecasting temporal_only        all_edges    count__mean_poisson_deviance      2.335947             False    2067 20240610
G1_spatiotemporal_gnn G1__spatial_block__all_forecasting__temporal_only__all_edges__h64_L3_do0_layernorm_res_relsum_manual_lr0p001_wd0p0001_seed20240610 spatial_block       test all_forecasting temporal_only        all_edges                     count__rmse      3.782199             False    2067 20240610
G1_spatiotemporal_gnn G1__spatial_block__all_forecasting__temporal_only__all_edges__h64_L3_do0_layernorm_res_relsum_manual_lr0p001_wd0p0001_seed20240610 spatial_block       test all_forecasting temporal_only        all_edges           ranking__kendall_corr      0.595439              True    2067 20240610
G1_spatiotemporal_gnn G1__spatial_block__all_forecasting__temporal_only__all_edges__h64_L3_do0_layernorm_res_relsum_manual_lr0p001_wd0p0001_seed20240610 spatial_block       test all_forecasting temporal_only        all_edges             ranking__ndcg_at_10      0.649296              True    2067 20240610
G1_spatiotemporal_gnn G1__spatial_block__all_forecasting__temporal_only__all_edges__h64_L3_do0_layernorm_res_relsum_manual_lr0p001_wd0p0001_seed20240610 spatial_block       test all_forecasting temporal_only        all_edges            ranking__ndcg_at_100      0.799381              True    2067 20240610
G1_spatiotemporal_gnn G1__spatial_block__all_forecasting__temporal_only__all_edges__h64_L3_do0_layernorm_res_relsum_manual_lr0p001_wd0p0001_seed20240610 spatial_block       test all_forecasting temporal_only        all_edges             ranking__ndcg_at_25      0.697657              True    2067 20240610
G1_spatiotemporal_gnn G1__spatial_block__all_forecasting__temporal_only__all_edges__h64_L3_do0_layernorm_res_relsum_manual_lr0p001_wd0p0001_seed20240610 spatial_block       test all_forecasting temporal_only        all_edges             ranking__ndcg_at_50      0.764277              True    2067 20240610
G1_spatiotemporal_gnn G1__spatial_block__all_forecasting__temporal_only__all_edges__h64_L3_do0_layernorm_res_relsum_manual_lr0p001_wd0p0001_seed20240610 spatial_block       test all_forecasting temporal_only        all_edges          ranking__spearman_corr      0.764371              True    2067 20240610
G1_spatiotemporal_gnn G1__spatial_block__all_forecasting__temporal_only__all_edges__h64_L3_do0_layernorm_res_relsum_manual_lr0p001_wd0p0001_seed20240610 spatial_block       test all_forecasting temporal_only        all_edges    ranking__top100_overlap_rate      0.650000              True    2067 20240610
G1_spatiotemporal_gnn G1__spatial_block__all_forecasting__temporal_only__all_edges__h64_L3_do0_layernorm_res_relsum_manual_lr0p001_wd0p0001_seed20240610 spatial_block       test all_forecasting temporal_only        all_edges     ranking__top10_overlap_rate      0.200000              True    2067 20240610
G1_spatiotemporal_gnn G1__spatial_block__all_forecasting__temporal_only__all_edges__h64_L3_do0_layernorm_res_relsum_manual_lr0p001_wd0p0001_seed20240610 spatial_block       test all_forecasting temporal_only        all_edges     ranking__top25_overlap_rate      0.320000              True    2067 20240610
G1_spatiotemporal_gnn G1__spatial_block__all_forecasting__temporal_only__all_edges__h64_L3_do0_layernorm_res_relsum_manual_lr0p001_wd0p0001_seed20240610 spatial_block       test all_forecasting temporal_only        all_edges     ranking__top50_overlap_rate      0.580000              True    2067 20240610
G1_spatiotemporal_gnn G1__spatial_block__all_forecasting__temporal_only__all_edges__h64_L3_do0_layernorm_res_relsum_manual_lr0p001_wd0p0001_seed20240610 spatial_block       test all_forecasting temporal_only        all_edges ranking__top_10pct_overlap_rate      0.642512              True    2067 20240610
G1_spatiotemporal_gnn G1__spatial_block__all_forecasting__temporal_only__all_edges__h64_L3_do0_layernorm_res_relsum_manual_lr0p001_wd0p0001_seed20240610 spatial_block       test all_forecasting temporal_only        all_edges  ranking__top_5pct_overlap_rate      0.644231              True    2067 20240610
G1_spatiotemporal_gnn G1__spatial_block__all_forecasting__temporal_only__all_edges__h64_L3_do0_layernorm_res_relsum_manual_lr0p001_wd0p0001_seed20240610 spatial_block      train all_forecasting temporal_only        all_edges                      count__mae      2.149734             False   21200 20240610
G1_spatiotemporal_gnn G1__spatial_block__all_forecasting__temporal_only__all_edges__h64_L3_do0_layernorm_res_relsum_manual_lr0p001_wd0p0001_seed20240610 spatial_block      train all_forecasting temporal_only        all_edges    count__mean_poisson_deviance      1.788551             False   21200 20240610
G1_spatiotemporal_gnn G1__spatial_block__all_forecasting__temporal_only__all_edges__h64_L3_do0_layernorm_res_relsum_manual_lr0p001_wd0p0001_seed20240610 spatial_block      train all_forecasting temporal_only        all_edges                     count__rmse      3.388339             False   21200 20240610
G1_spatiotemporal_gnn G1__spatial_block__all_forecasting__temporal_only__all_edges__h64_L3_do0_layernorm_res_relsum_manual_lr0p001_wd0p0001_seed20240610 spatial_block      train all_forecasting temporal_only        all_edges           ranking__kendall_corr      0.622535              True   21200 20240610
G1_spatiotemporal_gnn G1__spatial_block__all_forecasting__temporal_only__all_edges__h64_L3_do0_layernorm_res_relsum_manual_lr0p001_wd0p0001_seed20240610 spatial_block      train all_forecasting temporal_only        all_edges             ranking__ndcg_at_10      0.388515              True   21200 20240610
G1_spatiotemporal_gnn G1__spatial_block__all_forecasting__temporal_only__all_edges__h64_L3_do0_layernorm_res_relsum_manual_lr0p001_wd0p0001_seed20240610 spatial_block      train all_forecasting temporal_only        all_edges            ranking__ndcg_at_100      0.599411              True   21200 20240610
G1_spatiotemporal_gnn G1__spatial_block__all_forecasting__temporal_only__all_edges__h64_L3_do0_layernorm_res_relsum_manual_lr0p001_wd0p0001_seed20240610 spatial_block      train all_forecasting temporal_only        all_edges             ranking__ndcg_at_25      0.499380              True   21200 20240610
G1_spatiotemporal_gnn G1__spatial_block__all_forecasting__temporal_only__all_edges__h64_L3_do0_layernorm_res_relsum_manual_lr0p001_wd0p0001_seed20240610 spatial_block      train all_forecasting temporal_only        all_edges             ranking__ndcg_at_50      0.561770              True   21200 20240610
G1_spatiotemporal_gnn G1__spatial_block__all_forecasting__temporal_only__all_edges__h64_L3_do0_layernorm_res_relsum_manual_lr0p001_wd0p0001_seed20240610 spatial_block      train all_forecasting temporal_only        all_edges          ranking__spearman_corr      0.791420              True   21200 20240610
G1_spatiotemporal_gnn G1__spatial_block__all_forecasting__temporal_only__all_edges__h64_L3_do0_layernorm_res_relsum_manual_lr0p001_wd0p0001_seed20240610 spatial_block      train all_forecasting temporal_only        all_edges    ranking__top100_overlap_rate      0.250000              True   21200 20240610
G1_spatiotemporal_gnn G1__spatial_block__all_forecasting__temporal_only__all_edges__h64_L3_do0_layernorm_res_relsum_manual_lr0p001_wd0p0001_seed20240610 spatial_block      train all_forecasting temporal_only        all_edges     ranking__top10_overlap_rate      0.100000              True   21200 20240610
G1_spatiotemporal_gnn G1__spatial_block__all_forecasting__temporal_only__all_edges__h64_L3_do0_layernorm_res_relsum_manual_lr0p001_wd0p0001_seed20240610 spatial_block      train all_forecasting temporal_only        all_edges     ranking__top25_overlap_rate      0.160000              True   21200 20240610
G1_spatiotemporal_gnn G1__spatial_block__all_forecasting__temporal_only__all_edges__h64_L3_do0_layernorm_res_relsum_manual_lr0p001_wd0p0001_seed20240610 spatial_block      train all_forecasting temporal_only        all_edges     ranking__top50_overlap_rate      0.200000              True   21200 20240610
G1_spatiotemporal_gnn G1__spatial_block__all_forecasting__temporal_only__all_edges__h64_L3_do0_layernorm_res_relsum_manual_lr0p001_wd0p0001_seed20240610 spatial_block      train all_forecasting temporal_only        all_edges ranking__top_10pct_overlap_rate      0.568396              True   21200 20240610
G1_spatiotemporal_gnn G1__spatial_block__all_forecasting__temporal_only__all_edges__h64_L3_do0_layernorm_res_relsum_manual_lr0p001_wd0p0001_seed20240610 spatial_block      train all_forecasting temporal_only        all_edges  ranking__top_5pct_overlap_rate      0.501887              True   21200 20240610
G1_spatiotemporal_gnn G1__spatial_block__all_forecasting__temporal_only__all_edges__h64_L3_do0_layernorm_res_relsum_manual_lr0p001_wd0p0001_seed20240610 spatial_block validation all_forecasting temporal_only        all_edges                      count__mae      2.635335             False    5353 20240610
G1_spatiotemporal_gnn G1__spatial_block__all_forecasting__temporal_only__all_edges__h64_L3_do0_layernorm_res_relsum_manual_lr0p001_wd0p0001_seed20240610 spatial_block validation all_forecasting temporal_only        all_edges    count__mean_poisson_deviance      2.317746             False    5353 20240610
G1_spatiotemporal_gnn G1__spatial_block__all_forecasting__temporal_only__all_edges__h64_L3_do0_layernorm_res_relsum_manual_lr0p001_wd0p0001_seed20240610 spatial_block validation all_forecasting temporal_only        all_edges                     count__rmse      3.891252             False    5353 20240610
G1_spatiotemporal_gnn G1__spatial_block__all_forecasting__temporal_only__all_edges__h64_L3_do0_layernorm_res_relsum_manual_lr0p001_wd0p0001_seed20240610 spatial_block validation all_forecasting temporal_only        all_edges           ranking__kendall_corr      0.498804              True    5353 20240610
G1_spatiotemporal_gnn G1__spatial_block__all_forecasting__temporal_only__all_edges__h64_L3_do0_layernorm_res_relsum_manual_lr0p001_wd0p0001_seed20240610 spatial_block validation all_forecasting temporal_only        all_edges             ranking__ndcg_at_10      0.542751              True    5353 20240610
G1_spatiotemporal_gnn G1__spatial_block__all_forecasting__temporal_only__all_edges__h64_L3_do0_layernorm_res_relsum_manual_lr0p001_wd0p0001_seed20240610 spatial_block validation all_forecasting temporal_only        all_edges            ranking__ndcg_at_100      0.670504              True    5353 20240610
G1_spatiotemporal_gnn G1__spatial_block__all_forecasting__temporal_only__all_edges__h64_L3_do0_layernorm_res_relsum_manual_lr0p001_wd0p0001_seed20240610 spatial_block validation all_forecasting temporal_only        all_edges             ranking__ndcg_at_25      0.574941              True    5353 20240610
G1_spatiotemporal_gnn G1__spatial_block__all_forecasting__temporal_only__all_edges__h64_L3_do0_layernorm_res_relsum_manual_lr0p001_wd0p0001_seed20240610 spatial_block validation all_forecasting temporal_only        all_edges             ranking__ndcg_at_50      0.605958              True    5353 20240610
G1_spatiotemporal_gnn G1__spatial_block__all_forecasting__temporal_only__all_edges__h64_L3_do0_layernorm_res_relsum_manual_lr0p001_wd0p0001_seed20240610 spatial_block validation all_forecasting temporal_only        all_edges          ranking__spearman_corr      0.665190              True    5353 20240610
G1_spatiotemporal_gnn G1__spatial_block__all_forecasting__temporal_only__all_edges__h64_L3_do0_layernorm_res_relsum_manual_lr0p001_wd0p0001_seed20240610 spatial_block validation all_forecasting temporal_only        all_edges    ranking__top100_overlap_rate      0.310000              True    5353 20240610
G1_spatiotemporal_gnn G1__spatial_block__all_forecasting__temporal_only__all_edges__h64_L3_do0_layernorm_res_relsum_manual_lr0p001_wd0p0001_seed20240610 spatial_block validation all_forecasting temporal_only        all_edges     ranking__top10_overlap_rate      0.100000              True    5353 20240610
G1_spatiotemporal_gnn G1__spatial_block__all_forecasting__temporal_only__all_edges__h64_L3_do0_layernorm_res_relsum_manual_lr0p001_wd0p0001_seed20240610 spatial_block validation all_forecasting temporal_only        all_edges     ranking__top25_overlap_rate      0.200000              True    5353 20240610
G1_spatiotemporal_gnn G1__spatial_block__all_forecasting__temporal_only__all_edges__h64_L3_do0_layernorm_res_relsum_manual_lr0p001_wd0p0001_seed20240610 spatial_block validation all_forecasting temporal_only        all_edges     ranking__top50_overlap_rate      0.220000              True    5353 20240610
G1_spatiotemporal_gnn G1__spatial_block__all_forecasting__temporal_only__all_edges__h64_L3_do0_layernorm_res_relsum_manual_lr0p001_wd0p0001_seed20240610 spatial_block validation all_forecasting temporal_only        all_edges ranking__top_10pct_overlap_rate      0.524254              True    5353 20240610
G1_spatiotemporal_gnn G1__spatial_block__all_forecasting__temporal_only__all_edges__h64_L3_do0_layernorm_res_relsum_manual_lr0p001_wd0p0001_seed20240610 spatial_block validation all_forecasting temporal_only        all_edges  ranking__top_5pct_overlap_rate      0.406716              True    5353 20240610
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
| `metrics` | `urban_graph_benchmark/outputs/mtl_311_water_v0/baselines/G1_5_validation_sweep_spatial_ndcg/runs/h64_L3_do0_layernorm_res_relsum_manual_lr0p001_wd0p0001/metrics.csv` |
| `predictions_validation` | `urban_graph_benchmark/outputs/mtl_311_water_v0/baselines/G1_5_validation_sweep_spatial_ndcg/runs/h64_L3_do0_layernorm_res_relsum_manual_lr0p001_wd0p0001/predictions_validation.parquet` |
| `predictions_test` | `urban_graph_benchmark/outputs/mtl_311_water_v0/baselines/G1_5_validation_sweep_spatial_ndcg/runs/h64_L3_do0_layernorm_res_relsum_manual_lr0p001_wd0p0001/predictions_test.parquet` |
| `predictions_all_evaluated` | `urban_graph_benchmark/outputs/mtl_311_water_v0/baselines/G1_5_validation_sweep_spatial_ndcg/runs/h64_L3_do0_layernorm_res_relsum_manual_lr0p001_wd0p0001/predictions_all_evaluated.parquet` |
| `training_curves` | `urban_graph_benchmark/outputs/mtl_311_water_v0/baselines/G1_5_validation_sweep_spatial_ndcg/runs/h64_L3_do0_layernorm_res_relsum_manual_lr0p001_wd0p0001/training_curves.csv` |
| `trial_audit` | `urban_graph_benchmark/outputs/mtl_311_water_v0/baselines/G1_5_validation_sweep_spatial_ndcg/runs/h64_L3_do0_layernorm_res_relsum_manual_lr0p001_wd0p0001/trial_audit.csv` |
| `model_selection_audit` | `urban_graph_benchmark/outputs/mtl_311_water_v0/baselines/G1_5_validation_sweep_spatial_ndcg/runs/h64_L3_do0_layernorm_res_relsum_manual_lr0p001_wd0p0001/model_selection_audit.csv` |
| `feature_preprocessing_audit` | `urban_graph_benchmark/outputs/mtl_311_water_v0/baselines/G1_5_validation_sweep_spatial_ndcg/runs/h64_L3_do0_layernorm_res_relsum_manual_lr0p001_wd0p0001/feature_preprocessing_audit.csv` |
| `graph_regime_audit` | `urban_graph_benchmark/outputs/mtl_311_water_v0/baselines/G1_5_validation_sweep_spatial_ndcg/runs/h64_L3_do0_layernorm_res_relsum_manual_lr0p001_wd0p0001/graph_regime_audit.csv` |
| `model_metadata` | `urban_graph_benchmark/outputs/mtl_311_water_v0/baselines/G1_5_validation_sweep_spatial_ndcg/runs/h64_L3_do0_layernorm_res_relsum_manual_lr0p001_wd0p0001/model_metadata.json` |
| `baseline_report` | `urban_graph_benchmark/outputs/mtl_311_water_v0/baselines/G1_5_validation_sweep_spatial_ndcg/runs/h64_L3_do0_layernorm_res_relsum_manual_lr0p001_wd0p0001/baseline_report.md` |
| `checkpoints_dir` | `urban_graph_benchmark/outputs/mtl_311_water_v0/baselines/G1_5_validation_sweep_spatial_ndcg/runs/h64_L3_do0_layernorm_res_relsum_manual_lr0p001_wd0p0001/checkpoints` |
| `embeddings_dir` | `urban_graph_benchmark/outputs/mtl_311_water_v0/baselines/G1_5_validation_sweep_spatial_ndcg/runs/h64_L3_do0_layernorm_res_relsum_manual_lr0p001_wd0p0001/embeddings` |

## Benchmark handoff

Temporal graph results should be compared against A0_3_tract_train_mean and A3_all_forecasting RF. Spatial-block graph results should be compared against A3_lagged_reporting_forecasting HGB. The most convincing graph claim would improve both count error and high-burden ranking metrics under spatial-block evaluation.
