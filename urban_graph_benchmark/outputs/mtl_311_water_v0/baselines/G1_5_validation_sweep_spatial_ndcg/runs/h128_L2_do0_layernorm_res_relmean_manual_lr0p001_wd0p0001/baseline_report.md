# G1 Spatiotemporal Tract GNN Baseline

Generated at: `2026-06-15T18:20:39.112399+00:00`

Graph artifact directory: `urban_graph_benchmark/outputs/mtl_311_water_v0/graphs/G1_tract_month_graph`

Output directory: `urban_graph_benchmark/outputs/mtl_311_water_v0/baselines/G1_5_validation_sweep_spatial_ndcg/runs/h128_L2_do0_layernorm_res_relmean_manual_lr0p001_wd0p0001`

## Purpose

G1 is the first controlled graph-value test after the frozen A0--A3 tabular benchmark layer. It evaluates whether message passing over tract topology/time improves prediction and ranking of reported water/drainage 311 burden beyond feature-parity tabular baselines.

## Important interpretation rule

A graph result is meaningful only relative to A3 feature-parity baselines and to no-edge neural controls. Beating raw SVI or weak naive baselines is not enough after A3.

## Training configuration

Checkpointing and validation selection monitor: `validation_ndcg_at_100` (higher is better).

```json
{
  "hidden_dim": 128,
  "num_layers": 2,
  "dropout": 0.0,
  "activation": "relu",
  "normalization": "layernorm",
  "residual": true,
  "backend": "manual",
  "relation_combine": "mean",
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
                           G1__spatial_block__all_forecasting__no_edges__all_edges__h128_L2_do0_layernorm_res_relmean_manual_lr0p001_wd0p0001_seed20240610 completed spatial_block all_forecasting               no_edges              all_edges         100 validation_ndcg_at_100            0.674683             2.599412                     0.674683             0                 0         1.060985
                      G1__spatial_block__all_forecasting__temporal_only__all_edges__h128_L2_do0_layernorm_res_relmean_manual_lr0p001_wd0p0001_seed20240610 completed spatial_block all_forecasting          temporal_only              all_edges          83 validation_ndcg_at_100            0.652655             2.635928                     0.652655         50220                 2         2.670016
         G1__spatial_block__all_forecasting__temporal_only__no_test_incident_edges__h128_L2_do0_layernorm_res_relmean_manual_lr0p001_wd0p0001_seed20240610 completed spatial_block all_forecasting          temporal_only no_test_incident_edges          83 validation_ndcg_at_100            0.652556             2.635975                     0.652556         46593                 2         2.652657
                   G1__spatial_block__all_forecasting__spatial_temporal__all_edges__h128_L2_do0_layernorm_res_relmean_manual_lr0p001_wd0p0001_seed20240610 completed spatial_block all_forecasting       spatial_temporal              all_edges          76 validation_ndcg_at_100            0.662973             2.637110                     0.662973        279180                 3         5.507796
      G1__spatial_block__all_forecasting__spatial_temporal__no_test_incident_edges__h128_L2_do0_layernorm_res_relmean_manual_lr0p001_wd0p0001_seed20240610 completed spatial_block all_forecasting       spatial_temporal no_test_incident_edges          66 validation_ndcg_at_100            0.653209             2.636331                     0.653209        252339                 3         4.782879
             G1__spatial_block__all_forecasting__random_spatial_placebo__all_edges__h128_L2_do0_layernorm_res_relmean_manual_lr0p001_wd0p0001_seed20240610 completed spatial_block all_forecasting random_spatial_placebo              all_edges         177 validation_ndcg_at_100            0.669229             2.624680                     0.669229        279180                 3        10.369634
G1__spatial_block__all_forecasting__random_spatial_placebo__no_test_incident_edges__h128_L2_do0_layernorm_res_relmean_manual_lr0p001_wd0p0001_seed20240610 completed spatial_block all_forecasting random_spatial_placebo no_test_incident_edges          87 validation_ndcg_at_100            0.662264             2.619570                     0.662264        243568                 3         5.680809
```

## Validation-only model selection

Trial representatives are selected using `validation_ndcg_at_100` on validation nodes only. Test metrics are reported after selection.

```text
                                                                                                                                                model_name  split_scheme  feature_regime            edge_regime       edge_mask_regime       selection_metric  selection_metric_value  validation_mae  validation_spearman  validation_ndcg_at_10  validation_ndcg_at_25  validation_ndcg_at_50  validation_ndcg_at_100  validation_top10_overlap_rate  validation_top25_overlap_rate  validation_top50_overlap_rate  validation_top100_overlap_rate  validation_top_5pct_overlap_rate  validation_top_10pct_overlap_rate  test_mae  test_spearman  test_ndcg_at_10  test_ndcg_at_25  test_ndcg_at_50  test_ndcg_at_100  test_top10_overlap_rate  test_top25_overlap_rate  test_top50_overlap_rate  test_top100_overlap_rate  test_top_5pct_overlap_rate  test_top_10pct_overlap_rate  selected_overall_for_split  selected_for_feature_regime  selected_for_test_summary
                           G1__spatial_block__all_forecasting__no_edges__all_edges__h128_L2_do0_layernorm_res_relmean_manual_lr0p001_wd0p0001_seed20240610 spatial_block all_forecasting               no_edges              all_edges validation_ndcg_at_100                0.674683        2.599412             0.673159               0.569124               0.586309               0.620329                0.674683                            0.1                           0.24                           0.26                            0.31                          0.395522                           0.535448  2.303005       0.767077         0.662239         0.761188         0.784108          0.824939                      0.1                     0.48                     0.56                      0.68                    0.673077                     0.657005                        True                         True                       True
             G1__spatial_block__all_forecasting__random_spatial_placebo__all_edges__h128_L2_do0_layernorm_res_relmean_manual_lr0p001_wd0p0001_seed20240610 spatial_block all_forecasting random_spatial_placebo              all_edges validation_ndcg_at_100                0.669229        2.624680             0.663175               0.489743               0.559558               0.599079                0.669229                            0.1                           0.16                           0.24                            0.30                          0.414179                           0.518657  2.325751       0.761183         0.606877         0.687904         0.759663          0.799780                      0.0                     0.32                     0.56                      0.66                    0.644231                     0.632850                       False                        False                      False
                   G1__spatial_block__all_forecasting__spatial_temporal__all_edges__h128_L2_do0_layernorm_res_relmean_manual_lr0p001_wd0p0001_seed20240610 spatial_block all_forecasting       spatial_temporal              all_edges validation_ndcg_at_100                0.662973        2.637110             0.664284               0.582627               0.626084               0.653342                0.662973                            0.2                           0.24                           0.34                            0.28                          0.429104                           0.524254  2.331436       0.763750         0.582875         0.682595         0.769707          0.792021                      0.0                     0.32                     0.60                      0.67                    0.663462                     0.623188                       False                        False                      False
G1__spatial_block__all_forecasting__random_spatial_placebo__no_test_incident_edges__h128_L2_do0_layernorm_res_relmean_manual_lr0p001_wd0p0001_seed20240610 spatial_block all_forecasting random_spatial_placebo no_test_incident_edges validation_ndcg_at_100                0.662264        2.619570             0.666775               0.486136               0.571719               0.621434                0.662264                            0.1                           0.24                           0.36                            0.30                          0.417910                           0.527985  2.524904       0.722972         0.582327         0.653439         0.709223          0.746600                      0.0                     0.24                     0.46                      0.62                    0.615385                     0.574879                       False                        False                      False
      G1__spatial_block__all_forecasting__spatial_temporal__no_test_incident_edges__h128_L2_do0_layernorm_res_relmean_manual_lr0p001_wd0p0001_seed20240610 spatial_block all_forecasting       spatial_temporal no_test_incident_edges validation_ndcg_at_100                0.653209        2.636331             0.663300               0.548669               0.592764               0.629621                0.653209                            0.2                           0.24                           0.32                            0.29                          0.432836                           0.514925  2.542357       0.724975         0.572367         0.640493         0.701665          0.732662                      0.0                     0.20                     0.42                      0.59                    0.596154                     0.574879                       False                        False                      False
                      G1__spatial_block__all_forecasting__temporal_only__all_edges__h128_L2_do0_layernorm_res_relmean_manual_lr0p001_wd0p0001_seed20240610 spatial_block all_forecasting          temporal_only              all_edges validation_ndcg_at_100                0.652655        2.635928             0.661533               0.450279               0.548924               0.604515                0.652655                            0.1                           0.16                           0.28                            0.34                          0.421642                           0.529851  2.331647       0.763607         0.605019         0.706757         0.768489          0.802156                      0.1                     0.32                     0.58                      0.65                    0.644231                     0.618357                       False                        False                      False
         G1__spatial_block__all_forecasting__temporal_only__no_test_incident_edges__h128_L2_do0_layernorm_res_relmean_manual_lr0p001_wd0p0001_seed20240610 spatial_block all_forecasting          temporal_only no_test_incident_edges validation_ndcg_at_100                0.652556        2.635975             0.661535               0.450279               0.548754               0.604343                0.652556                            0.1                           0.16                           0.28                            0.34                          0.421642                           0.529851  2.787934       0.721018         0.582626         0.656394         0.713026          0.762737                      0.1                     0.32                     0.44                      0.62                    0.615385                     0.589372                       False                        False                      False
```

## Compact metrics for selected models

```text
          model_stage                                                                                                                      model_name  split_scheme split_name  feature_regime edge_regime edge_mask_regime                     metric_name  metric_value  higher_is_better  n_rows     seed
G1_spatiotemporal_gnn G1__spatial_block__all_forecasting__no_edges__all_edges__h128_L2_do0_layernorm_res_relmean_manual_lr0p001_wd0p0001_seed20240610 spatial_block       test all_forecasting    no_edges        all_edges                      count__mae      2.303005             False    2067 20240610
G1_spatiotemporal_gnn G1__spatial_block__all_forecasting__no_edges__all_edges__h128_L2_do0_layernorm_res_relmean_manual_lr0p001_wd0p0001_seed20240610 spatial_block       test all_forecasting    no_edges        all_edges    count__mean_poisson_deviance      2.153243             False    2067 20240610
G1_spatiotemporal_gnn G1__spatial_block__all_forecasting__no_edges__all_edges__h128_L2_do0_layernorm_res_relmean_manual_lr0p001_wd0p0001_seed20240610 spatial_block       test all_forecasting    no_edges        all_edges                     count__rmse      3.714394             False    2067 20240610
G1_spatiotemporal_gnn G1__spatial_block__all_forecasting__no_edges__all_edges__h128_L2_do0_layernorm_res_relmean_manual_lr0p001_wd0p0001_seed20240610 spatial_block       test all_forecasting    no_edges        all_edges           ranking__kendall_corr      0.598352              True    2067 20240610
G1_spatiotemporal_gnn G1__spatial_block__all_forecasting__no_edges__all_edges__h128_L2_do0_layernorm_res_relmean_manual_lr0p001_wd0p0001_seed20240610 spatial_block       test all_forecasting    no_edges        all_edges             ranking__ndcg_at_10      0.662239              True    2067 20240610
G1_spatiotemporal_gnn G1__spatial_block__all_forecasting__no_edges__all_edges__h128_L2_do0_layernorm_res_relmean_manual_lr0p001_wd0p0001_seed20240610 spatial_block       test all_forecasting    no_edges        all_edges            ranking__ndcg_at_100      0.824939              True    2067 20240610
G1_spatiotemporal_gnn G1__spatial_block__all_forecasting__no_edges__all_edges__h128_L2_do0_layernorm_res_relmean_manual_lr0p001_wd0p0001_seed20240610 spatial_block       test all_forecasting    no_edges        all_edges             ranking__ndcg_at_25      0.761188              True    2067 20240610
G1_spatiotemporal_gnn G1__spatial_block__all_forecasting__no_edges__all_edges__h128_L2_do0_layernorm_res_relmean_manual_lr0p001_wd0p0001_seed20240610 spatial_block       test all_forecasting    no_edges        all_edges             ranking__ndcg_at_50      0.784108              True    2067 20240610
G1_spatiotemporal_gnn G1__spatial_block__all_forecasting__no_edges__all_edges__h128_L2_do0_layernorm_res_relmean_manual_lr0p001_wd0p0001_seed20240610 spatial_block       test all_forecasting    no_edges        all_edges          ranking__spearman_corr      0.767077              True    2067 20240610
G1_spatiotemporal_gnn G1__spatial_block__all_forecasting__no_edges__all_edges__h128_L2_do0_layernorm_res_relmean_manual_lr0p001_wd0p0001_seed20240610 spatial_block       test all_forecasting    no_edges        all_edges    ranking__top100_overlap_rate      0.680000              True    2067 20240610
G1_spatiotemporal_gnn G1__spatial_block__all_forecasting__no_edges__all_edges__h128_L2_do0_layernorm_res_relmean_manual_lr0p001_wd0p0001_seed20240610 spatial_block       test all_forecasting    no_edges        all_edges     ranking__top10_overlap_rate      0.100000              True    2067 20240610
G1_spatiotemporal_gnn G1__spatial_block__all_forecasting__no_edges__all_edges__h128_L2_do0_layernorm_res_relmean_manual_lr0p001_wd0p0001_seed20240610 spatial_block       test all_forecasting    no_edges        all_edges     ranking__top25_overlap_rate      0.480000              True    2067 20240610
G1_spatiotemporal_gnn G1__spatial_block__all_forecasting__no_edges__all_edges__h128_L2_do0_layernorm_res_relmean_manual_lr0p001_wd0p0001_seed20240610 spatial_block       test all_forecasting    no_edges        all_edges     ranking__top50_overlap_rate      0.560000              True    2067 20240610
G1_spatiotemporal_gnn G1__spatial_block__all_forecasting__no_edges__all_edges__h128_L2_do0_layernorm_res_relmean_manual_lr0p001_wd0p0001_seed20240610 spatial_block       test all_forecasting    no_edges        all_edges ranking__top_10pct_overlap_rate      0.657005              True    2067 20240610
G1_spatiotemporal_gnn G1__spatial_block__all_forecasting__no_edges__all_edges__h128_L2_do0_layernorm_res_relmean_manual_lr0p001_wd0p0001_seed20240610 spatial_block       test all_forecasting    no_edges        all_edges  ranking__top_5pct_overlap_rate      0.673077              True    2067 20240610
G1_spatiotemporal_gnn G1__spatial_block__all_forecasting__no_edges__all_edges__h128_L2_do0_layernorm_res_relmean_manual_lr0p001_wd0p0001_seed20240610 spatial_block      train all_forecasting    no_edges        all_edges                      count__mae      2.236232             False   21200 20240610
G1_spatiotemporal_gnn G1__spatial_block__all_forecasting__no_edges__all_edges__h128_L2_do0_layernorm_res_relmean_manual_lr0p001_wd0p0001_seed20240610 spatial_block      train all_forecasting    no_edges        all_edges    count__mean_poisson_deviance      1.914105             False   21200 20240610
G1_spatiotemporal_gnn G1__spatial_block__all_forecasting__no_edges__all_edges__h128_L2_do0_layernorm_res_relmean_manual_lr0p001_wd0p0001_seed20240610 spatial_block      train all_forecasting    no_edges        all_edges                     count__rmse      3.525749             False   21200 20240610
G1_spatiotemporal_gnn G1__spatial_block__all_forecasting__no_edges__all_edges__h128_L2_do0_layernorm_res_relmean_manual_lr0p001_wd0p0001_seed20240610 spatial_block      train all_forecasting    no_edges        all_edges           ranking__kendall_corr      0.604514              True   21200 20240610
G1_spatiotemporal_gnn G1__spatial_block__all_forecasting__no_edges__all_edges__h128_L2_do0_layernorm_res_relmean_manual_lr0p001_wd0p0001_seed20240610 spatial_block      train all_forecasting    no_edges        all_edges             ranking__ndcg_at_10      0.515099              True   21200 20240610
G1_spatiotemporal_gnn G1__spatial_block__all_forecasting__no_edges__all_edges__h128_L2_do0_layernorm_res_relmean_manual_lr0p001_wd0p0001_seed20240610 spatial_block      train all_forecasting    no_edges        all_edges            ranking__ndcg_at_100      0.593428              True   21200 20240610
G1_spatiotemporal_gnn G1__spatial_block__all_forecasting__no_edges__all_edges__h128_L2_do0_layernorm_res_relmean_manual_lr0p001_wd0p0001_seed20240610 spatial_block      train all_forecasting    no_edges        all_edges             ranking__ndcg_at_25      0.520771              True   21200 20240610
G1_spatiotemporal_gnn G1__spatial_block__all_forecasting__no_edges__all_edges__h128_L2_do0_layernorm_res_relmean_manual_lr0p001_wd0p0001_seed20240610 spatial_block      train all_forecasting    no_edges        all_edges             ranking__ndcg_at_50      0.545545              True   21200 20240610
G1_spatiotemporal_gnn G1__spatial_block__all_forecasting__no_edges__all_edges__h128_L2_do0_layernorm_res_relmean_manual_lr0p001_wd0p0001_seed20240610 spatial_block      train all_forecasting    no_edges        all_edges          ranking__spearman_corr      0.774113              True   21200 20240610
G1_spatiotemporal_gnn G1__spatial_block__all_forecasting__no_edges__all_edges__h128_L2_do0_layernorm_res_relmean_manual_lr0p001_wd0p0001_seed20240610 spatial_block      train all_forecasting    no_edges        all_edges    ranking__top100_overlap_rate      0.210000              True   21200 20240610
G1_spatiotemporal_gnn G1__spatial_block__all_forecasting__no_edges__all_edges__h128_L2_do0_layernorm_res_relmean_manual_lr0p001_wd0p0001_seed20240610 spatial_block      train all_forecasting    no_edges        all_edges     ranking__top10_overlap_rate      0.100000              True   21200 20240610
G1_spatiotemporal_gnn G1__spatial_block__all_forecasting__no_edges__all_edges__h128_L2_do0_layernorm_res_relmean_manual_lr0p001_wd0p0001_seed20240610 spatial_block      train all_forecasting    no_edges        all_edges     ranking__top25_overlap_rate      0.160000              True   21200 20240610
G1_spatiotemporal_gnn G1__spatial_block__all_forecasting__no_edges__all_edges__h128_L2_do0_layernorm_res_relmean_manual_lr0p001_wd0p0001_seed20240610 spatial_block      train all_forecasting    no_edges        all_edges     ranking__top50_overlap_rate      0.160000              True   21200 20240610
G1_spatiotemporal_gnn G1__spatial_block__all_forecasting__no_edges__all_edges__h128_L2_do0_layernorm_res_relmean_manual_lr0p001_wd0p0001_seed20240610 spatial_block      train all_forecasting    no_edges        all_edges ranking__top_10pct_overlap_rate      0.545283              True   21200 20240610
G1_spatiotemporal_gnn G1__spatial_block__all_forecasting__no_edges__all_edges__h128_L2_do0_layernorm_res_relmean_manual_lr0p001_wd0p0001_seed20240610 spatial_block      train all_forecasting    no_edges        all_edges  ranking__top_5pct_overlap_rate      0.472642              True   21200 20240610
G1_spatiotemporal_gnn G1__spatial_block__all_forecasting__no_edges__all_edges__h128_L2_do0_layernorm_res_relmean_manual_lr0p001_wd0p0001_seed20240610 spatial_block validation all_forecasting    no_edges        all_edges                      count__mae      2.599412             False    5353 20240610
G1_spatiotemporal_gnn G1__spatial_block__all_forecasting__no_edges__all_edges__h128_L2_do0_layernorm_res_relmean_manual_lr0p001_wd0p0001_seed20240610 spatial_block validation all_forecasting    no_edges        all_edges    count__mean_poisson_deviance      2.254911             False    5353 20240610
G1_spatiotemporal_gnn G1__spatial_block__all_forecasting__no_edges__all_edges__h128_L2_do0_layernorm_res_relmean_manual_lr0p001_wd0p0001_seed20240610 spatial_block validation all_forecasting    no_edges        all_edges                     count__rmse      3.880373             False    5353 20240610
G1_spatiotemporal_gnn G1__spatial_block__all_forecasting__no_edges__all_edges__h128_L2_do0_layernorm_res_relmean_manual_lr0p001_wd0p0001_seed20240610 spatial_block validation all_forecasting    no_edges        all_edges           ranking__kendall_corr      0.506076              True    5353 20240610
G1_spatiotemporal_gnn G1__spatial_block__all_forecasting__no_edges__all_edges__h128_L2_do0_layernorm_res_relmean_manual_lr0p001_wd0p0001_seed20240610 spatial_block validation all_forecasting    no_edges        all_edges             ranking__ndcg_at_10      0.569124              True    5353 20240610
G1_spatiotemporal_gnn G1__spatial_block__all_forecasting__no_edges__all_edges__h128_L2_do0_layernorm_res_relmean_manual_lr0p001_wd0p0001_seed20240610 spatial_block validation all_forecasting    no_edges        all_edges            ranking__ndcg_at_100      0.674683              True    5353 20240610
G1_spatiotemporal_gnn G1__spatial_block__all_forecasting__no_edges__all_edges__h128_L2_do0_layernorm_res_relmean_manual_lr0p001_wd0p0001_seed20240610 spatial_block validation all_forecasting    no_edges        all_edges             ranking__ndcg_at_25      0.586309              True    5353 20240610
G1_spatiotemporal_gnn G1__spatial_block__all_forecasting__no_edges__all_edges__h128_L2_do0_layernorm_res_relmean_manual_lr0p001_wd0p0001_seed20240610 spatial_block validation all_forecasting    no_edges        all_edges             ranking__ndcg_at_50      0.620329              True    5353 20240610
G1_spatiotemporal_gnn G1__spatial_block__all_forecasting__no_edges__all_edges__h128_L2_do0_layernorm_res_relmean_manual_lr0p001_wd0p0001_seed20240610 spatial_block validation all_forecasting    no_edges        all_edges          ranking__spearman_corr      0.673159              True    5353 20240610
G1_spatiotemporal_gnn G1__spatial_block__all_forecasting__no_edges__all_edges__h128_L2_do0_layernorm_res_relmean_manual_lr0p001_wd0p0001_seed20240610 spatial_block validation all_forecasting    no_edges        all_edges    ranking__top100_overlap_rate      0.310000              True    5353 20240610
G1_spatiotemporal_gnn G1__spatial_block__all_forecasting__no_edges__all_edges__h128_L2_do0_layernorm_res_relmean_manual_lr0p001_wd0p0001_seed20240610 spatial_block validation all_forecasting    no_edges        all_edges     ranking__top10_overlap_rate      0.100000              True    5353 20240610
G1_spatiotemporal_gnn G1__spatial_block__all_forecasting__no_edges__all_edges__h128_L2_do0_layernorm_res_relmean_manual_lr0p001_wd0p0001_seed20240610 spatial_block validation all_forecasting    no_edges        all_edges     ranking__top25_overlap_rate      0.240000              True    5353 20240610
G1_spatiotemporal_gnn G1__spatial_block__all_forecasting__no_edges__all_edges__h128_L2_do0_layernorm_res_relmean_manual_lr0p001_wd0p0001_seed20240610 spatial_block validation all_forecasting    no_edges        all_edges     ranking__top50_overlap_rate      0.260000              True    5353 20240610
G1_spatiotemporal_gnn G1__spatial_block__all_forecasting__no_edges__all_edges__h128_L2_do0_layernorm_res_relmean_manual_lr0p001_wd0p0001_seed20240610 spatial_block validation all_forecasting    no_edges        all_edges ranking__top_10pct_overlap_rate      0.535448              True    5353 20240610
G1_spatiotemporal_gnn G1__spatial_block__all_forecasting__no_edges__all_edges__h128_L2_do0_layernorm_res_relmean_manual_lr0p001_wd0p0001_seed20240610 spatial_block validation all_forecasting    no_edges        all_edges  ranking__top_5pct_overlap_rate      0.395522              True    5353 20240610
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
| `metrics` | `urban_graph_benchmark/outputs/mtl_311_water_v0/baselines/G1_5_validation_sweep_spatial_ndcg/runs/h128_L2_do0_layernorm_res_relmean_manual_lr0p001_wd0p0001/metrics.csv` |
| `predictions_validation` | `urban_graph_benchmark/outputs/mtl_311_water_v0/baselines/G1_5_validation_sweep_spatial_ndcg/runs/h128_L2_do0_layernorm_res_relmean_manual_lr0p001_wd0p0001/predictions_validation.parquet` |
| `predictions_test` | `urban_graph_benchmark/outputs/mtl_311_water_v0/baselines/G1_5_validation_sweep_spatial_ndcg/runs/h128_L2_do0_layernorm_res_relmean_manual_lr0p001_wd0p0001/predictions_test.parquet` |
| `predictions_all_evaluated` | `urban_graph_benchmark/outputs/mtl_311_water_v0/baselines/G1_5_validation_sweep_spatial_ndcg/runs/h128_L2_do0_layernorm_res_relmean_manual_lr0p001_wd0p0001/predictions_all_evaluated.parquet` |
| `training_curves` | `urban_graph_benchmark/outputs/mtl_311_water_v0/baselines/G1_5_validation_sweep_spatial_ndcg/runs/h128_L2_do0_layernorm_res_relmean_manual_lr0p001_wd0p0001/training_curves.csv` |
| `trial_audit` | `urban_graph_benchmark/outputs/mtl_311_water_v0/baselines/G1_5_validation_sweep_spatial_ndcg/runs/h128_L2_do0_layernorm_res_relmean_manual_lr0p001_wd0p0001/trial_audit.csv` |
| `model_selection_audit` | `urban_graph_benchmark/outputs/mtl_311_water_v0/baselines/G1_5_validation_sweep_spatial_ndcg/runs/h128_L2_do0_layernorm_res_relmean_manual_lr0p001_wd0p0001/model_selection_audit.csv` |
| `feature_preprocessing_audit` | `urban_graph_benchmark/outputs/mtl_311_water_v0/baselines/G1_5_validation_sweep_spatial_ndcg/runs/h128_L2_do0_layernorm_res_relmean_manual_lr0p001_wd0p0001/feature_preprocessing_audit.csv` |
| `graph_regime_audit` | `urban_graph_benchmark/outputs/mtl_311_water_v0/baselines/G1_5_validation_sweep_spatial_ndcg/runs/h128_L2_do0_layernorm_res_relmean_manual_lr0p001_wd0p0001/graph_regime_audit.csv` |
| `model_metadata` | `urban_graph_benchmark/outputs/mtl_311_water_v0/baselines/G1_5_validation_sweep_spatial_ndcg/runs/h128_L2_do0_layernorm_res_relmean_manual_lr0p001_wd0p0001/model_metadata.json` |
| `baseline_report` | `urban_graph_benchmark/outputs/mtl_311_water_v0/baselines/G1_5_validation_sweep_spatial_ndcg/runs/h128_L2_do0_layernorm_res_relmean_manual_lr0p001_wd0p0001/baseline_report.md` |
| `checkpoints_dir` | `urban_graph_benchmark/outputs/mtl_311_water_v0/baselines/G1_5_validation_sweep_spatial_ndcg/runs/h128_L2_do0_layernorm_res_relmean_manual_lr0p001_wd0p0001/checkpoints` |
| `embeddings_dir` | `urban_graph_benchmark/outputs/mtl_311_water_v0/baselines/G1_5_validation_sweep_spatial_ndcg/runs/h128_L2_do0_layernorm_res_relmean_manual_lr0p001_wd0p0001/embeddings` |

## Benchmark handoff

Temporal graph results should be compared against A0_3_tract_train_mean and A3_all_forecasting RF. Spatial-block graph results should be compared against A3_lagged_reporting_forecasting HGB. The most convincing graph claim would improve both count error and high-burden ranking metrics under spatial-block evaluation.
