# G1 Spatiotemporal Tract GNN Baseline

Generated at: `2026-06-15T18:13:33.319575+00:00`

Graph artifact directory: `urban_graph_benchmark/outputs/mtl_311_water_v0/graphs/G1_tract_month_graph`

Output directory: `urban_graph_benchmark/outputs/mtl_311_water_v0/baselines/G1_5_validation_sweep_spatial_ndcg/runs/h64_L2_do0p05_layernorm_res_relsum_manual_lr0p001_wd0p0001`

## Purpose

G1 is the first controlled graph-value test after the frozen A0--A3 tabular benchmark layer. It evaluates whether message passing over tract topology/time improves prediction and ranking of reported water/drainage 311 burden beyond feature-parity tabular baselines.

## Important interpretation rule

A graph result is meaningful only relative to A3 feature-parity baselines and to no-edge neural controls. Beating raw SVI or weak naive baselines is not enough after A3.

## Training configuration

Checkpointing and validation selection monitor: `validation_ndcg_at_100` (higher is better).

```json
{
  "hidden_dim": 64,
  "num_layers": 2,
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
                           G1__spatial_block__all_forecasting__no_edges__all_edges__h64_L2_do0p05_layernorm_res_relsum_manual_lr0p001_wd0p0001_seed20240610 completed spatial_block all_forecasting               no_edges              all_edges         151 validation_ndcg_at_100            0.683941             2.654501                     0.683941             0                 0         1.146923
                      G1__spatial_block__all_forecasting__temporal_only__all_edges__h64_L2_do0p05_layernorm_res_relsum_manual_lr0p001_wd0p0001_seed20240610 completed spatial_block all_forecasting          temporal_only              all_edges         203 validation_ndcg_at_100            0.693295             2.617420                     0.693295         50220                 2         2.912069
         G1__spatial_block__all_forecasting__temporal_only__no_test_incident_edges__h64_L2_do0p05_layernorm_res_relsum_manual_lr0p001_wd0p0001_seed20240610 completed spatial_block all_forecasting          temporal_only no_test_incident_edges         200 validation_ndcg_at_100            0.694453             2.619593                     0.694453         46593                 2         2.896424
                   G1__spatial_block__all_forecasting__spatial_temporal__all_edges__h64_L2_do0p05_layernorm_res_relsum_manual_lr0p001_wd0p0001_seed20240610 completed spatial_block all_forecasting       spatial_temporal              all_edges          81 validation_ndcg_at_100            0.654618             2.663333                     0.654618        279180                 3         3.152564
      G1__spatial_block__all_forecasting__spatial_temporal__no_test_incident_edges__h64_L2_do0p05_layernorm_res_relsum_manual_lr0p001_wd0p0001_seed20240610 completed spatial_block all_forecasting       spatial_temporal no_test_incident_edges         224 validation_ndcg_at_100            0.691003             2.625194                     0.691003        252339                 3         6.036422
             G1__spatial_block__all_forecasting__random_spatial_placebo__all_edges__h64_L2_do0p05_layernorm_res_relsum_manual_lr0p001_wd0p0001_seed20240610 completed spatial_block all_forecasting random_spatial_placebo              all_edges         240 validation_ndcg_at_100            0.681548             2.605051                     0.681548        279180                 3         6.341877
G1__spatial_block__all_forecasting__random_spatial_placebo__no_test_incident_edges__h64_L2_do0p05_layernorm_res_relsum_manual_lr0p001_wd0p0001_seed20240610 completed spatial_block all_forecasting random_spatial_placebo no_test_incident_edges         235 validation_ndcg_at_100            0.675513             2.607309                     0.675513        243568                 3         5.911709
```

## Validation-only model selection

Trial representatives are selected using `validation_ndcg_at_100` on validation nodes only. Test metrics are reported after selection.

```text
                                                                                                                                                 model_name  split_scheme  feature_regime            edge_regime       edge_mask_regime       selection_metric  selection_metric_value  validation_mae  validation_spearman  validation_ndcg_at_10  validation_ndcg_at_25  validation_ndcg_at_50  validation_ndcg_at_100  validation_top10_overlap_rate  validation_top25_overlap_rate  validation_top50_overlap_rate  validation_top100_overlap_rate  validation_top_5pct_overlap_rate  validation_top_10pct_overlap_rate  test_mae  test_spearman  test_ndcg_at_10  test_ndcg_at_25  test_ndcg_at_50  test_ndcg_at_100  test_top10_overlap_rate  test_top25_overlap_rate  test_top50_overlap_rate  test_top100_overlap_rate  test_top_5pct_overlap_rate  test_top_10pct_overlap_rate  selected_overall_for_split  selected_for_feature_regime  selected_for_test_summary
         G1__spatial_block__all_forecasting__temporal_only__no_test_incident_edges__h64_L2_do0p05_layernorm_res_relsum_manual_lr0p001_wd0p0001_seed20240610 spatial_block all_forecasting          temporal_only no_test_incident_edges validation_ndcg_at_100                0.694453        2.619593             0.674008               0.562918               0.601589               0.639878                0.694453                            0.2                           0.20                           0.28                            0.34                          0.436567                           0.529851  2.626761       0.752316         0.523990         0.638360         0.657235          0.731127                      0.1                     0.32                     0.44                      0.60                    0.605769                     0.613527                        True                         True                       True
                      G1__spatial_block__all_forecasting__temporal_only__all_edges__h64_L2_do0p05_layernorm_res_relsum_manual_lr0p001_wd0p0001_seed20240610 spatial_block all_forecasting          temporal_only              all_edges validation_ndcg_at_100                0.693295        2.617420             0.674068               0.562918               0.597477               0.639813                0.693295                            0.2                           0.20                           0.28                            0.33                          0.436567                           0.531716  2.351789       0.771093         0.645138         0.700656         0.761452          0.805716                      0.3                     0.32                     0.60                      0.64                    0.634615                     0.623188                       False                        False                      False
      G1__spatial_block__all_forecasting__spatial_temporal__no_test_incident_edges__h64_L2_do0p05_layernorm_res_relsum_manual_lr0p001_wd0p0001_seed20240610 spatial_block all_forecasting       spatial_temporal no_test_incident_edges validation_ndcg_at_100                0.691003        2.625194             0.676984               0.634109               0.627991               0.669669                0.691003                            0.3                           0.20                           0.36                            0.30                          0.414179                           0.527985  2.739455       0.705548         0.492827         0.568834         0.626558          0.688549                      0.0                     0.20                     0.40                      0.53                    0.528846                     0.565217                       False                        False                      False
                           G1__spatial_block__all_forecasting__no_edges__all_edges__h64_L2_do0p05_layernorm_res_relsum_manual_lr0p001_wd0p0001_seed20240610 spatial_block all_forecasting               no_edges              all_edges validation_ndcg_at_100                0.683941        2.654501             0.668729               0.552822               0.591507               0.609954                0.683941                            0.1                           0.20                           0.24                            0.34                          0.417910                           0.531716  2.411409       0.763784         0.661877         0.701481         0.767052          0.805768                      0.1                     0.32                     0.58                      0.68                    0.663462                     0.628019                       False                        False                      False
             G1__spatial_block__all_forecasting__random_spatial_placebo__all_edges__h64_L2_do0p05_layernorm_res_relsum_manual_lr0p001_wd0p0001_seed20240610 spatial_block all_forecasting random_spatial_placebo              all_edges validation_ndcg_at_100                0.681548        2.605051             0.679929               0.577140               0.619518               0.657839                0.681548                            0.1                           0.36                           0.34                            0.32                          0.432836                           0.542910  2.321417       0.775933         0.644854         0.686067         0.747059          0.811123                      0.2                     0.28                     0.56                      0.67                    0.673077                     0.628019                       False                        False                      False
G1__spatial_block__all_forecasting__random_spatial_placebo__no_test_incident_edges__h64_L2_do0p05_layernorm_res_relsum_manual_lr0p001_wd0p0001_seed20240610 spatial_block all_forecasting random_spatial_placebo no_test_incident_edges validation_ndcg_at_100                0.675513        2.607309             0.679405               0.557516               0.619971               0.649453                0.675513                            0.1                           0.36                           0.34                            0.32                          0.436567                           0.541045  2.766095       0.692884         0.487926         0.521795         0.563965          0.651903                      0.0                     0.12                     0.28                      0.48                    0.471154                     0.512077                       False                        False                      False
                   G1__spatial_block__all_forecasting__spatial_temporal__all_edges__h64_L2_do0p05_layernorm_res_relsum_manual_lr0p001_wd0p0001_seed20240610 spatial_block all_forecasting       spatial_temporal              all_edges validation_ndcg_at_100                0.654618        2.663333             0.665465               0.538118               0.605693               0.632577                0.654618                            0.2                           0.32                           0.36                            0.30                          0.395522                           0.529851  2.398913       0.764408         0.547733         0.643711         0.735885          0.785416                      0.0                     0.20                     0.54                      0.66                    0.653846                     0.623188                       False                        False                      False
```

## Compact metrics for selected models

```text
          model_stage                                                                                                                                         model_name  split_scheme split_name  feature_regime   edge_regime       edge_mask_regime                     metric_name  metric_value  higher_is_better  n_rows     seed
G1_spatiotemporal_gnn G1__spatial_block__all_forecasting__temporal_only__no_test_incident_edges__h64_L2_do0p05_layernorm_res_relsum_manual_lr0p001_wd0p0001_seed20240610 spatial_block       test all_forecasting temporal_only no_test_incident_edges                      count__mae      2.626761             False    2067 20240610
G1_spatiotemporal_gnn G1__spatial_block__all_forecasting__temporal_only__no_test_incident_edges__h64_L2_do0p05_layernorm_res_relsum_manual_lr0p001_wd0p0001_seed20240610 spatial_block       test all_forecasting temporal_only no_test_incident_edges    count__mean_poisson_deviance      3.043146             False    2067 20240610
G1_spatiotemporal_gnn G1__spatial_block__all_forecasting__temporal_only__no_test_incident_edges__h64_L2_do0p05_layernorm_res_relsum_manual_lr0p001_wd0p0001_seed20240610 spatial_block       test all_forecasting temporal_only no_test_incident_edges                     count__rmse      4.428671             False    2067 20240610
G1_spatiotemporal_gnn G1__spatial_block__all_forecasting__temporal_only__no_test_incident_edges__h64_L2_do0p05_layernorm_res_relsum_manual_lr0p001_wd0p0001_seed20240610 spatial_block       test all_forecasting temporal_only no_test_incident_edges           ranking__kendall_corr      0.583957              True    2067 20240610
G1_spatiotemporal_gnn G1__spatial_block__all_forecasting__temporal_only__no_test_incident_edges__h64_L2_do0p05_layernorm_res_relsum_manual_lr0p001_wd0p0001_seed20240610 spatial_block       test all_forecasting temporal_only no_test_incident_edges             ranking__ndcg_at_10      0.523990              True    2067 20240610
G1_spatiotemporal_gnn G1__spatial_block__all_forecasting__temporal_only__no_test_incident_edges__h64_L2_do0p05_layernorm_res_relsum_manual_lr0p001_wd0p0001_seed20240610 spatial_block       test all_forecasting temporal_only no_test_incident_edges            ranking__ndcg_at_100      0.731127              True    2067 20240610
G1_spatiotemporal_gnn G1__spatial_block__all_forecasting__temporal_only__no_test_incident_edges__h64_L2_do0p05_layernorm_res_relsum_manual_lr0p001_wd0p0001_seed20240610 spatial_block       test all_forecasting temporal_only no_test_incident_edges             ranking__ndcg_at_25      0.638360              True    2067 20240610
G1_spatiotemporal_gnn G1__spatial_block__all_forecasting__temporal_only__no_test_incident_edges__h64_L2_do0p05_layernorm_res_relsum_manual_lr0p001_wd0p0001_seed20240610 spatial_block       test all_forecasting temporal_only no_test_incident_edges             ranking__ndcg_at_50      0.657235              True    2067 20240610
G1_spatiotemporal_gnn G1__spatial_block__all_forecasting__temporal_only__no_test_incident_edges__h64_L2_do0p05_layernorm_res_relsum_manual_lr0p001_wd0p0001_seed20240610 spatial_block       test all_forecasting temporal_only no_test_incident_edges          ranking__spearman_corr      0.752316              True    2067 20240610
G1_spatiotemporal_gnn G1__spatial_block__all_forecasting__temporal_only__no_test_incident_edges__h64_L2_do0p05_layernorm_res_relsum_manual_lr0p001_wd0p0001_seed20240610 spatial_block       test all_forecasting temporal_only no_test_incident_edges    ranking__top100_overlap_rate      0.600000              True    2067 20240610
G1_spatiotemporal_gnn G1__spatial_block__all_forecasting__temporal_only__no_test_incident_edges__h64_L2_do0p05_layernorm_res_relsum_manual_lr0p001_wd0p0001_seed20240610 spatial_block       test all_forecasting temporal_only no_test_incident_edges     ranking__top10_overlap_rate      0.100000              True    2067 20240610
G1_spatiotemporal_gnn G1__spatial_block__all_forecasting__temporal_only__no_test_incident_edges__h64_L2_do0p05_layernorm_res_relsum_manual_lr0p001_wd0p0001_seed20240610 spatial_block       test all_forecasting temporal_only no_test_incident_edges     ranking__top25_overlap_rate      0.320000              True    2067 20240610
G1_spatiotemporal_gnn G1__spatial_block__all_forecasting__temporal_only__no_test_incident_edges__h64_L2_do0p05_layernorm_res_relsum_manual_lr0p001_wd0p0001_seed20240610 spatial_block       test all_forecasting temporal_only no_test_incident_edges     ranking__top50_overlap_rate      0.440000              True    2067 20240610
G1_spatiotemporal_gnn G1__spatial_block__all_forecasting__temporal_only__no_test_incident_edges__h64_L2_do0p05_layernorm_res_relsum_manual_lr0p001_wd0p0001_seed20240610 spatial_block       test all_forecasting temporal_only no_test_incident_edges ranking__top_10pct_overlap_rate      0.613527              True    2067 20240610
G1_spatiotemporal_gnn G1__spatial_block__all_forecasting__temporal_only__no_test_incident_edges__h64_L2_do0p05_layernorm_res_relsum_manual_lr0p001_wd0p0001_seed20240610 spatial_block       test all_forecasting temporal_only no_test_incident_edges  ranking__top_5pct_overlap_rate      0.605769              True    2067 20240610
G1_spatiotemporal_gnn G1__spatial_block__all_forecasting__temporal_only__no_test_incident_edges__h64_L2_do0p05_layernorm_res_relsum_manual_lr0p001_wd0p0001_seed20240610 spatial_block      train all_forecasting temporal_only no_test_incident_edges                      count__mae      2.247483             False   21200 20240610
G1_spatiotemporal_gnn G1__spatial_block__all_forecasting__temporal_only__no_test_incident_edges__h64_L2_do0p05_layernorm_res_relsum_manual_lr0p001_wd0p0001_seed20240610 spatial_block      train all_forecasting temporal_only no_test_incident_edges    count__mean_poisson_deviance      2.007617             False   21200 20240610
G1_spatiotemporal_gnn G1__spatial_block__all_forecasting__temporal_only__no_test_incident_edges__h64_L2_do0p05_layernorm_res_relsum_manual_lr0p001_wd0p0001_seed20240610 spatial_block      train all_forecasting temporal_only no_test_incident_edges                     count__rmse      3.595446             False   21200 20240610
G1_spatiotemporal_gnn G1__spatial_block__all_forecasting__temporal_only__no_test_incident_edges__h64_L2_do0p05_layernorm_res_relsum_manual_lr0p001_wd0p0001_seed20240610 spatial_block      train all_forecasting temporal_only no_test_incident_edges           ranking__kendall_corr      0.604484              True   21200 20240610
G1_spatiotemporal_gnn G1__spatial_block__all_forecasting__temporal_only__no_test_incident_edges__h64_L2_do0p05_layernorm_res_relsum_manual_lr0p001_wd0p0001_seed20240610 spatial_block      train all_forecasting temporal_only no_test_incident_edges             ranking__ndcg_at_10      0.314807              True   21200 20240610
G1_spatiotemporal_gnn G1__spatial_block__all_forecasting__temporal_only__no_test_incident_edges__h64_L2_do0p05_layernorm_res_relsum_manual_lr0p001_wd0p0001_seed20240610 spatial_block      train all_forecasting temporal_only no_test_incident_edges            ranking__ndcg_at_100      0.528066              True   21200 20240610
G1_spatiotemporal_gnn G1__spatial_block__all_forecasting__temporal_only__no_test_incident_edges__h64_L2_do0p05_layernorm_res_relsum_manual_lr0p001_wd0p0001_seed20240610 spatial_block      train all_forecasting temporal_only no_test_incident_edges             ranking__ndcg_at_25      0.397045              True   21200 20240610
G1_spatiotemporal_gnn G1__spatial_block__all_forecasting__temporal_only__no_test_incident_edges__h64_L2_do0p05_layernorm_res_relsum_manual_lr0p001_wd0p0001_seed20240610 spatial_block      train all_forecasting temporal_only no_test_incident_edges             ranking__ndcg_at_50      0.461523              True   21200 20240610
G1_spatiotemporal_gnn G1__spatial_block__all_forecasting__temporal_only__no_test_incident_edges__h64_L2_do0p05_layernorm_res_relsum_manual_lr0p001_wd0p0001_seed20240610 spatial_block      train all_forecasting temporal_only no_test_incident_edges          ranking__spearman_corr      0.773361              True   21200 20240610
G1_spatiotemporal_gnn G1__spatial_block__all_forecasting__temporal_only__no_test_incident_edges__h64_L2_do0p05_layernorm_res_relsum_manual_lr0p001_wd0p0001_seed20240610 spatial_block      train all_forecasting temporal_only no_test_incident_edges    ranking__top100_overlap_rate      0.180000              True   21200 20240610
G1_spatiotemporal_gnn G1__spatial_block__all_forecasting__temporal_only__no_test_incident_edges__h64_L2_do0p05_layernorm_res_relsum_manual_lr0p001_wd0p0001_seed20240610 spatial_block      train all_forecasting temporal_only no_test_incident_edges     ranking__top10_overlap_rate      0.000000              True   21200 20240610
G1_spatiotemporal_gnn G1__spatial_block__all_forecasting__temporal_only__no_test_incident_edges__h64_L2_do0p05_layernorm_res_relsum_manual_lr0p001_wd0p0001_seed20240610 spatial_block      train all_forecasting temporal_only no_test_incident_edges     ranking__top25_overlap_rate      0.040000              True   21200 20240610
G1_spatiotemporal_gnn G1__spatial_block__all_forecasting__temporal_only__no_test_incident_edges__h64_L2_do0p05_layernorm_res_relsum_manual_lr0p001_wd0p0001_seed20240610 spatial_block      train all_forecasting temporal_only no_test_incident_edges     ranking__top50_overlap_rate      0.100000              True   21200 20240610
G1_spatiotemporal_gnn G1__spatial_block__all_forecasting__temporal_only__no_test_incident_edges__h64_L2_do0p05_layernorm_res_relsum_manual_lr0p001_wd0p0001_seed20240610 spatial_block      train all_forecasting temporal_only no_test_incident_edges ranking__top_10pct_overlap_rate      0.541038              True   21200 20240610
G1_spatiotemporal_gnn G1__spatial_block__all_forecasting__temporal_only__no_test_incident_edges__h64_L2_do0p05_layernorm_res_relsum_manual_lr0p001_wd0p0001_seed20240610 spatial_block      train all_forecasting temporal_only no_test_incident_edges  ranking__top_5pct_overlap_rate      0.469811              True   21200 20240610
G1_spatiotemporal_gnn G1__spatial_block__all_forecasting__temporal_only__no_test_incident_edges__h64_L2_do0p05_layernorm_res_relsum_manual_lr0p001_wd0p0001_seed20240610 spatial_block validation all_forecasting temporal_only no_test_incident_edges                      count__mae      2.619593             False    5353 20240610
G1_spatiotemporal_gnn G1__spatial_block__all_forecasting__temporal_only__no_test_incident_edges__h64_L2_do0p05_layernorm_res_relsum_manual_lr0p001_wd0p0001_seed20240610 spatial_block validation all_forecasting temporal_only no_test_incident_edges    count__mean_poisson_deviance      2.347368             False    5353 20240610
G1_spatiotemporal_gnn G1__spatial_block__all_forecasting__temporal_only__no_test_incident_edges__h64_L2_do0p05_layernorm_res_relsum_manual_lr0p001_wd0p0001_seed20240610 spatial_block validation all_forecasting temporal_only no_test_incident_edges                     count__rmse      3.940648             False    5353 20240610
G1_spatiotemporal_gnn G1__spatial_block__all_forecasting__temporal_only__no_test_incident_edges__h64_L2_do0p05_layernorm_res_relsum_manual_lr0p001_wd0p0001_seed20240610 spatial_block validation all_forecasting temporal_only no_test_incident_edges           ranking__kendall_corr      0.506864              True    5353 20240610
G1_spatiotemporal_gnn G1__spatial_block__all_forecasting__temporal_only__no_test_incident_edges__h64_L2_do0p05_layernorm_res_relsum_manual_lr0p001_wd0p0001_seed20240610 spatial_block validation all_forecasting temporal_only no_test_incident_edges             ranking__ndcg_at_10      0.562918              True    5353 20240610
G1_spatiotemporal_gnn G1__spatial_block__all_forecasting__temporal_only__no_test_incident_edges__h64_L2_do0p05_layernorm_res_relsum_manual_lr0p001_wd0p0001_seed20240610 spatial_block validation all_forecasting temporal_only no_test_incident_edges            ranking__ndcg_at_100      0.694453              True    5353 20240610
G1_spatiotemporal_gnn G1__spatial_block__all_forecasting__temporal_only__no_test_incident_edges__h64_L2_do0p05_layernorm_res_relsum_manual_lr0p001_wd0p0001_seed20240610 spatial_block validation all_forecasting temporal_only no_test_incident_edges             ranking__ndcg_at_25      0.601589              True    5353 20240610
G1_spatiotemporal_gnn G1__spatial_block__all_forecasting__temporal_only__no_test_incident_edges__h64_L2_do0p05_layernorm_res_relsum_manual_lr0p001_wd0p0001_seed20240610 spatial_block validation all_forecasting temporal_only no_test_incident_edges             ranking__ndcg_at_50      0.639878              True    5353 20240610
G1_spatiotemporal_gnn G1__spatial_block__all_forecasting__temporal_only__no_test_incident_edges__h64_L2_do0p05_layernorm_res_relsum_manual_lr0p001_wd0p0001_seed20240610 spatial_block validation all_forecasting temporal_only no_test_incident_edges          ranking__spearman_corr      0.674008              True    5353 20240610
G1_spatiotemporal_gnn G1__spatial_block__all_forecasting__temporal_only__no_test_incident_edges__h64_L2_do0p05_layernorm_res_relsum_manual_lr0p001_wd0p0001_seed20240610 spatial_block validation all_forecasting temporal_only no_test_incident_edges    ranking__top100_overlap_rate      0.340000              True    5353 20240610
G1_spatiotemporal_gnn G1__spatial_block__all_forecasting__temporal_only__no_test_incident_edges__h64_L2_do0p05_layernorm_res_relsum_manual_lr0p001_wd0p0001_seed20240610 spatial_block validation all_forecasting temporal_only no_test_incident_edges     ranking__top10_overlap_rate      0.200000              True    5353 20240610
G1_spatiotemporal_gnn G1__spatial_block__all_forecasting__temporal_only__no_test_incident_edges__h64_L2_do0p05_layernorm_res_relsum_manual_lr0p001_wd0p0001_seed20240610 spatial_block validation all_forecasting temporal_only no_test_incident_edges     ranking__top25_overlap_rate      0.200000              True    5353 20240610
G1_spatiotemporal_gnn G1__spatial_block__all_forecasting__temporal_only__no_test_incident_edges__h64_L2_do0p05_layernorm_res_relsum_manual_lr0p001_wd0p0001_seed20240610 spatial_block validation all_forecasting temporal_only no_test_incident_edges     ranking__top50_overlap_rate      0.280000              True    5353 20240610
G1_spatiotemporal_gnn G1__spatial_block__all_forecasting__temporal_only__no_test_incident_edges__h64_L2_do0p05_layernorm_res_relsum_manual_lr0p001_wd0p0001_seed20240610 spatial_block validation all_forecasting temporal_only no_test_incident_edges ranking__top_10pct_overlap_rate      0.529851              True    5353 20240610
G1_spatiotemporal_gnn G1__spatial_block__all_forecasting__temporal_only__no_test_incident_edges__h64_L2_do0p05_layernorm_res_relsum_manual_lr0p001_wd0p0001_seed20240610 spatial_block validation all_forecasting temporal_only no_test_incident_edges  ranking__top_5pct_overlap_rate      0.436567              True    5353 20240610
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
| `metrics` | `urban_graph_benchmark/outputs/mtl_311_water_v0/baselines/G1_5_validation_sweep_spatial_ndcg/runs/h64_L2_do0p05_layernorm_res_relsum_manual_lr0p001_wd0p0001/metrics.csv` |
| `predictions_validation` | `urban_graph_benchmark/outputs/mtl_311_water_v0/baselines/G1_5_validation_sweep_spatial_ndcg/runs/h64_L2_do0p05_layernorm_res_relsum_manual_lr0p001_wd0p0001/predictions_validation.parquet` |
| `predictions_test` | `urban_graph_benchmark/outputs/mtl_311_water_v0/baselines/G1_5_validation_sweep_spatial_ndcg/runs/h64_L2_do0p05_layernorm_res_relsum_manual_lr0p001_wd0p0001/predictions_test.parquet` |
| `predictions_all_evaluated` | `urban_graph_benchmark/outputs/mtl_311_water_v0/baselines/G1_5_validation_sweep_spatial_ndcg/runs/h64_L2_do0p05_layernorm_res_relsum_manual_lr0p001_wd0p0001/predictions_all_evaluated.parquet` |
| `training_curves` | `urban_graph_benchmark/outputs/mtl_311_water_v0/baselines/G1_5_validation_sweep_spatial_ndcg/runs/h64_L2_do0p05_layernorm_res_relsum_manual_lr0p001_wd0p0001/training_curves.csv` |
| `trial_audit` | `urban_graph_benchmark/outputs/mtl_311_water_v0/baselines/G1_5_validation_sweep_spatial_ndcg/runs/h64_L2_do0p05_layernorm_res_relsum_manual_lr0p001_wd0p0001/trial_audit.csv` |
| `model_selection_audit` | `urban_graph_benchmark/outputs/mtl_311_water_v0/baselines/G1_5_validation_sweep_spatial_ndcg/runs/h64_L2_do0p05_layernorm_res_relsum_manual_lr0p001_wd0p0001/model_selection_audit.csv` |
| `feature_preprocessing_audit` | `urban_graph_benchmark/outputs/mtl_311_water_v0/baselines/G1_5_validation_sweep_spatial_ndcg/runs/h64_L2_do0p05_layernorm_res_relsum_manual_lr0p001_wd0p0001/feature_preprocessing_audit.csv` |
| `graph_regime_audit` | `urban_graph_benchmark/outputs/mtl_311_water_v0/baselines/G1_5_validation_sweep_spatial_ndcg/runs/h64_L2_do0p05_layernorm_res_relsum_manual_lr0p001_wd0p0001/graph_regime_audit.csv` |
| `model_metadata` | `urban_graph_benchmark/outputs/mtl_311_water_v0/baselines/G1_5_validation_sweep_spatial_ndcg/runs/h64_L2_do0p05_layernorm_res_relsum_manual_lr0p001_wd0p0001/model_metadata.json` |
| `baseline_report` | `urban_graph_benchmark/outputs/mtl_311_water_v0/baselines/G1_5_validation_sweep_spatial_ndcg/runs/h64_L2_do0p05_layernorm_res_relsum_manual_lr0p001_wd0p0001/baseline_report.md` |
| `checkpoints_dir` | `urban_graph_benchmark/outputs/mtl_311_water_v0/baselines/G1_5_validation_sweep_spatial_ndcg/runs/h64_L2_do0p05_layernorm_res_relsum_manual_lr0p001_wd0p0001/checkpoints` |
| `embeddings_dir` | `urban_graph_benchmark/outputs/mtl_311_water_v0/baselines/G1_5_validation_sweep_spatial_ndcg/runs/h64_L2_do0p05_layernorm_res_relsum_manual_lr0p001_wd0p0001/embeddings` |

## Benchmark handoff

Temporal graph results should be compared against A0_3_tract_train_mean and A3_all_forecasting RF. Spatial-block graph results should be compared against A3_lagged_reporting_forecasting HGB. The most convincing graph claim would improve both count error and high-burden ranking metrics under spatial-block evaluation.
