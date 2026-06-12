# Index vs ML vs Graph Benchmark Comparison

Generated at: `2026-06-12T16:33:37.356258+00:00`

Output directory: `/home/tim/Documents/ville_ai/indices_BACKUP_before_clean_push/urban_graph_benchmark/outputs/mtl_311_water_v0/comparisons/08_index_ml_graph_benchmark`

## Purpose

This report consolidates already-produced benchmark outputs into a single index → tabular ML → graph/neural comparison. It does not retrain models. The goal is to summarize whether learned graph/neural benchmarks improve over static composite vulnerability indices and the A3 tabular ML layer.

## Compact comparison

```text
                               label        comparison_group                        family                              selection_policy         split_name      MAE  Spearman  NDCG@100  Top-10% overlap                                                                    role
            A0 tract historical mean Naive temporal baseline                            A0                 predefined strong A0 baseline      temporal_test 2.489209  0.694235  0.748363         0.483539                                    Strong naive tract-history baseline.
                    A1 raw SVI class         Composite index                           SVI                    diagnostic SVI class score      temporal_test      NaN  0.185642  0.222316         0.150943 Raw static SVI-style vulnerability class used directly as risk ranking.
               A1 raw SVI percentile         Composite index                           SVI                  predefined primary SVI score      temporal_test      NaN  0.160639  0.220560         0.052411     Raw static SVI-style composite score used directly as risk ranking.
     A2 calibrated SVI retrospective        Calibrated index SVI + retrospective reporting diagnostic retrospective calibrated SVI model      temporal_test 2.522434  0.671736  0.702508         0.462963                      Retrospective calibrated SVI/reporting diagnostic.
            A2 calibrated SVI static        Calibrated index             SVI + calibration               predefined calibrated SVI model      temporal_test 3.434751  0.236370  0.193140         0.121399                       Supervised calibration of static SVI-style score.
              A3 selected tabular ML              Tabular ML                 A3 tabular ML    A3 validation-selected spatial-block model spatial_block_test 2.339043  0.762380  0.784789         0.618357 Frozen feature-parity tabular ML baseline for spatial-block comparison.
G1.5 selected no-edge neural control          Neural control                      no_edges                        validation_ndcg_at_100               test 2.411409  0.763784  0.805768         0.628019                                                    G1 selected no_edges
           G1 pilot spatial_temporal            Graph/neural              spatial_temporal                        validation_ndcg_at_100               test 2.678841  0.761745  0.805970         0.623188                                Selected G1 pilot family representative.
             G1.5 selected A3_frozen            Graph/neural                     A3_frozen                           frozen_A3_selection               test 2.339043  0.762380  0.784789         0.618357                               A3 frozen selected spatial-block baseline
G1.5 selected spatial-temporal graph            Graph/neural              spatial_temporal                        validation_ndcg_at_100               test 2.418032  0.741807  0.802691         0.603865                                            G1 selected spatial_temporal
        G1.5 selected temporal graph            Graph/neural                 temporal_only                        validation_ndcg_at_100               test 2.682400  0.766072  0.796453         0.628019                                               G1 selected temporal_only
G1.5 selected random spatial placebo         Placebo control        random_spatial_placebo                        validation_ndcg_at_100               test 2.293357  0.777173  0.799908         0.642512                                      G1 selected random_spatial_placebo
```

## Metric winners

```text
                metric    metric_label  higher_is_better                         winner_label    winner_group          winner_family                                                                                                                                       winner_model_name  winner_value
                   mae             MAE             False G1.5 selected random spatial placebo Placebo control random_spatial_placebo        G1__spatial_block__all_forecasting__random_spatial_placebo__all_edges__h128_L2_do0p05_layernorm_res_relmean_manual_lr0p001_wd0p0001_seed20240610      2.293357
              spearman        Spearman              True G1.5 selected random spatial placebo Placebo control random_spatial_placebo        G1__spatial_block__all_forecasting__random_spatial_placebo__all_edges__h128_L2_do0p05_layernorm_res_relmean_manual_lr0p001_wd0p0001_seed20240610      0.777173
           ndcg_at_100        NDCG@100              True            G1 pilot spatial_temporal    Graph/neural       spatial_temporal G1__spatial_block__all_forecasting__spatial_temporal__no_test_incident_edges__h128_L2_do0p15_layernorm_res_relmean_manual_lr0p001_wd0p0001_seed20240610      0.805970
top_10pct_overlap_rate Top-10% overlap              True G1.5 selected random spatial placebo Placebo control random_spatial_placebo        G1__spatial_block__all_forecasting__random_spatial_placebo__all_edges__h128_L2_do0p05_layernorm_res_relmean_manual_lr0p001_wd0p0001_seed20240610      0.642512
```

## Key margins

```text
                                comparison                 metric    metric_label  higher_is_better                           left_label  left_value                          right_label  right_value  positive_margin_means_left_better
Graph family vs composite/calibrated index                    mae             MAE             False              G1.5 selected A3_frozen    2.339043      A2 calibrated SVI retrospective     2.522434                           0.183391
             Graph family vs A3 tabular ML                    mae             MAE             False              G1.5 selected A3_frozen    2.339043               A3 selected tabular ML     2.339043                           0.000000
  Graph family vs no-edge/placebo controls                    mae             MAE             False              G1.5 selected A3_frozen    2.339043 G1.5 selected random spatial placebo     2.293357                          -0.045687
           Control family vs A3 tabular ML                    mae             MAE             False G1.5 selected random spatial placebo    2.293357               A3 selected tabular ML     2.339043                           0.045687
Graph family vs composite/calibrated index               spearman        Spearman              True         G1.5 selected temporal graph    0.766072      A2 calibrated SVI retrospective     0.671736                           0.094336
             Graph family vs A3 tabular ML               spearman        Spearman              True         G1.5 selected temporal graph    0.766072               A3 selected tabular ML     0.762380                           0.003691
  Graph family vs no-edge/placebo controls               spearman        Spearman              True         G1.5 selected temporal graph    0.766072 G1.5 selected random spatial placebo     0.777173                          -0.011101
           Control family vs A3 tabular ML               spearman        Spearman              True G1.5 selected random spatial placebo    0.777173               A3 selected tabular ML     0.762380                           0.014793
Graph family vs composite/calibrated index            ndcg_at_100        NDCG@100              True            G1 pilot spatial_temporal    0.805970      A2 calibrated SVI retrospective     0.702508                           0.103462
             Graph family vs A3 tabular ML            ndcg_at_100        NDCG@100              True            G1 pilot spatial_temporal    0.805970               A3 selected tabular ML     0.784789                           0.021181
  Graph family vs no-edge/placebo controls            ndcg_at_100        NDCG@100              True            G1 pilot spatial_temporal    0.805970 G1.5 selected no-edge neural control     0.805768                           0.000202
           Control family vs A3 tabular ML            ndcg_at_100        NDCG@100              True G1.5 selected no-edge neural control    0.805768               A3 selected tabular ML     0.784789                           0.020979
Graph family vs composite/calibrated index top_10pct_overlap_rate Top-10% overlap              True         G1.5 selected temporal graph    0.628019      A2 calibrated SVI retrospective     0.462963                           0.165056
             Graph family vs A3 tabular ML top_10pct_overlap_rate Top-10% overlap              True         G1.5 selected temporal graph    0.628019               A3 selected tabular ML     0.618357                           0.009662
  Graph family vs no-edge/placebo controls top_10pct_overlap_rate Top-10% overlap              True         G1.5 selected temporal graph    0.628019 G1.5 selected random spatial placebo     0.642512                          -0.014493
           Control family vs A3 tabular ML top_10pct_overlap_rate Top-10% overlap              True G1.5 selected random spatial placebo    0.642512               A3 selected tabular ML     0.618357                           0.024155
```

## Interpretation

- Best graph-family NDCG@100 row: `G1 pilot spatial_temporal` (0.8060). Best index row: `A2 calibrated SVI retrospective` (0.7025). Graph-family margin: `0.1035`.

- Against A3 on NDCG@100, best graph-family row is `G1 pilot spatial_temporal` (0.8060) versus `A3 selected tabular ML` (0.7848); margin `0.0212`.

- The selected graph family beats the best no-edge/placebo control on NDCG@100 in this table. This is the stronger topology-specific pattern, but it should still be checked across seeds/splits.

- Public wording should distinguish `graph/neural benchmark improves over static composite indices` from `real spatial topology is validated`. The former may be supported even when no-edge/placebo controls remain strong; the latter requires the graph family to beat those controls.

## Missing or unavailable inputs

```text
source                                                                                                                                                                                                                                                issue
  SOVI SoVI metrics not found at /home/tim/Documents/ville_ai/indices_BACKUP_before_clean_push/urban_graph_benchmark/outputs/mtl_311_water_v0/baselines/A1_sovi_direct_ranking/metrics.csv. This is expected until the SoVI benchmark adapter is generated.
```

Missing optional rows are expected while SoVI or future benchmark layers are still being produced. Re-run this script after adding the corresponding metrics files.

## Inputs

```text
                          input                                                                                                                                                                   path  exists  required_for_core_comparison                                                                                                                                                                                                                                               status
                     A0 metrics                   /home/tim/Documents/ville_ai/indices_BACKUP_before_clean_push/urban_graph_benchmark/outputs/mtl_311_water_v0/baselines/A0_naive_temporal/metrics.csv    True                         False                                                                                                                                                                                                                                            available
                 A1 SVI metrics               /home/tim/Documents/ville_ai/indices_BACKUP_before_clean_push/urban_graph_benchmark/outputs/mtl_311_water_v0/baselines/A1_svi_direct_ranking/metrics.csv    True                          True                                                                                                                                                                                                                                            available
      A2 calibrated SVI metrics                   /home/tim/Documents/ville_ai/indices_BACKUP_before_clean_push/urban_graph_benchmark/outputs/mtl_311_water_v0/baselines/A2_calibrated_svi/metrics.csv    True                         False                                                                                                                                                                                                                                            available
                   SoVI metrics              /home/tim/Documents/ville_ai/indices_BACKUP_before_clean_push/urban_graph_benchmark/outputs/mtl_311_water_v0/baselines/A1_sovi_direct_ranking/metrics.csv   False                         False                                                                                                                                                                                                                                              missing
           A3 spatial directory         /home/tim/Documents/ville_ai/indices_BACKUP_before_clean_push/urban_graph_benchmark/outputs/mtl_311_water_v0/baselines/A3_feature_parity_tabular_spatial_block    True                          True                                                                                                                                                                                                                                            available
           G1 spatial directory /home/tim/Documents/ville_ai/indices_BACKUP_before_clean_push/urban_graph_benchmark/outputs/mtl_311_water_v0/baselines/G1_spatiotemporal_gnn_spatial_core_ndcg_monitor    True                         False                                                                                                                                                                                                                                            available
G1.5 validation-sweep directory              /home/tim/Documents/ville_ai/indices_BACKUP_before_clean_push/urban_graph_benchmark/outputs/mtl_311_water_v0/baselines/G1_5_validation_sweep_spatial_ndcg    True                          True                                                                                                                                                                                                                                            available
               logical_row:SOVI                                                                                                                                                                          False                         False SoVI metrics not found at /home/tim/Documents/ville_ai/indices_BACKUP_before_clean_push/urban_graph_benchmark/outputs/mtl_311_water_v0/baselines/A1_sovi_direct_ranking/metrics.csv. This is expected until the SoVI benchmark adapter is generated.
```

## Output files

```text
                   artifact                                                                                                                                                                                  path
               metrics_long  /home/tim/Documents/ville_ai/indices_BACKUP_before_clean_push/urban_graph_benchmark/outputs/mtl_311_water_v0/comparisons/08_index_ml_graph_benchmark/index_ml_graph_metrics_long.csv
                 comparison         /home/tim/Documents/ville_ai/indices_BACKUP_before_clean_push/urban_graph_benchmark/outputs/mtl_311_water_v0/comparisons/08_index_ml_graph_benchmark/benchmark_comparison.csv
                    compact /home/tim/Documents/ville_ai/indices_BACKUP_before_clean_push/urban_graph_benchmark/outputs/mtl_311_water_v0/comparisons/08_index_ml_graph_benchmark/benchmark_comparison_compact.csv
                    winners               /home/tim/Documents/ville_ai/indices_BACKUP_before_clean_push/urban_graph_benchmark/outputs/mtl_311_water_v0/comparisons/08_index_ml_graph_benchmark/metric_winners.csv
                    margins          /home/tim/Documents/ville_ai/indices_BACKUP_before_clean_push/urban_graph_benchmark/outputs/mtl_311_water_v0/comparisons/08_index_ml_graph_benchmark/family_margin_table.csv
                    missing          /home/tim/Documents/ville_ai/indices_BACKUP_before_clean_push/urban_graph_benchmark/outputs/mtl_311_water_v0/comparisons/08_index_ml_graph_benchmark/missing_input_audit.csv
                   metadata         /home/tim/Documents/ville_ai/indices_BACKUP_before_clean_push/urban_graph_benchmark/outputs/mtl_311_water_v0/comparisons/08_index_ml_graph_benchmark/comparison_metadata.json
                     report      /home/tim/Documents/ville_ai/indices_BACKUP_before_clean_push/urban_graph_benchmark/outputs/mtl_311_water_v0/comparisons/08_index_ml_graph_benchmark/benchmark_interpretation.md
                   plot_mae                    /home/tim/Documents/ville_ai/indices_BACKUP_before_clean_push/urban_graph_benchmark/outputs/mtl_311_water_v0/comparisons/08_index_ml_graph_benchmark/plots/mae.png
              plot_spearman               /home/tim/Documents/ville_ai/indices_BACKUP_before_clean_push/urban_graph_benchmark/outputs/mtl_311_water_v0/comparisons/08_index_ml_graph_benchmark/plots/spearman.png
           plot_ndcg_at_100            /home/tim/Documents/ville_ai/indices_BACKUP_before_clean_push/urban_graph_benchmark/outputs/mtl_311_water_v0/comparisons/08_index_ml_graph_benchmark/plots/ndcg_at_100.png
plot_top_10pct_overlap_rate /home/tim/Documents/ville_ai/indices_BACKUP_before_clean_push/urban_graph_benchmark/outputs/mtl_311_water_v0/comparisons/08_index_ml_graph_benchmark/plots/top_10pct_overlap_rate.png
```
