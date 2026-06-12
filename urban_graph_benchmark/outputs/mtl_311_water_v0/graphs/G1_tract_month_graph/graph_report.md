# G1 typed spatiotemporal tract-month graph

Generated at: `2026-06-12T13:20:46.278334+00:00`

## Purpose

This artifact converts the A3 tract-month panel into an auditable typed spatiotemporal graph for G1 graph baselines. It does not train a GNN. It freezes graph nodes, feature regimes, edge types, split masks, and leakage audits.

## Core graph definition

```text
node = census tract × month
target = water_drainage_count
primary graph question = does typed message passing improve beyond frozen A3 tabular baselines?
```

## Graph dimensions

| Quantity | Value |
|---|---:|
| Nodes | 28,620 |
| Unique tracts | 540 |
| Unique months | 53 |
| Edges | 508,140 |

## Feature regimes

| Regime | Number of features |
|---|---:|
| `all_forecasting` | 52 |
| `lagged_reporting` | 27 |
| `no_target_history` | 36 |

## Edge audit

```text
                            edge_type  n_edges  n_unique_sources  n_unique_targets  mean_out_degree_over_active_sources  max_out_degree  temporal_direction_violations  same_month_spatial_violations  is_placebo
               spatial_knn_same_month   228960             28620             28567                                  8.0               8                              0                              0       False
spatial_knn_same_month_random_placebo   228960             28620             28613                                  8.0               8                              0                              0        True
                  temporal_self_lag_1    28080             28080             28080                                  1.0               1                              0                              0       False
                 temporal_self_lag_12    22140             22140             22140                                  1.0               1                              0                              0       False
```

## Split summary

```text
 split_scheme  partition  n_nodes  target_total  target_mean  target_positive_rate
     temporal      train    19440      103218.0     5.309568              0.853498
     temporal validation     4320       22874.0     5.294907              0.840741
     temporal       test     4860       24330.0     5.006173              0.837654
 random_debug      train    20034      105622.0     5.272137              0.849656
 random_debug validation     4293       22423.0     5.223154              0.848125
 random_debug       test     4293       22377.0     5.212439              0.846028
spatial_block      train    21200      108751.0     5.129764              0.834151
spatial_block validation     5353       31377.0     5.861573              0.928078
spatial_block       test     2067       10294.0     4.980164              0.794872
```

## Message-passing edge masks

This graph artifact supports multiple message-passing regimes. Edge masks are stored in `edge_mask_by_split_regime.npz` and are aligned with the row order of `edge_table.parquet`.

```text
                           edge_mask  n_edges_allowed  n_edges_total  share_edges_allowed
                  temporal_all_edges           508140         508140             1.000000
          temporal_train_train_edges           342900         508140             0.674814
     temporal_no_test_incident_edges           420660         508140             0.827843
              random_debug_all_edges           508140         508140             1.000000
      random_debug_train_train_edges           248892         508140             0.489810
 random_debug_no_test_incident_edges           367211         508140             0.722657
             spatial_block_all_edges           508140         508140             1.000000
     spatial_block_train_train_edges           321294         508140             0.632294
spatial_block_no_test_incident_edges           449314         508140             0.884233
```

## Leakage audit

```text
                                     check_name status   severity  n_violations                                                                                                          details
           forecasting_feature_forbidden_tokens passed   critical             0                                                                                                                 
temporal_edges_do_not_point_from_future_to_past passed   critical             0                                                                        source_period_index > target_period_index
                   spatial_edges_are_same_month passed   critical             0                                                                       source_period_index != target_period_index
          temporal_cross_split_edges_disclosure   info disclosure         10260 Cross-split edges are allowed only under explicit transductive/inference assumptions; labels must remain masked.
      random_debug_cross_split_edges_disclosure   info disclosure        236396 Cross-split edges are allowed only under explicit transductive/inference assumptions; labels must remain masked.
     spatial_block_cross_split_edges_disclosure   info disclosure        127710 Cross-split edges are allowed only under explicit transductive/inference assumptions; labels must remain masked.
```

## Feature audit preview

```text
   feature_regime                                                                  feature        feature_family  uses_target_history  uses_reporting_history  uses_same_month_information  is_strict_forecasting_safe  global_missing_rate leakage_status
  all_forecasting                               target_history__water_drainage_count_lag_1        target_history                 True                   False                        False                        True             0.018868         passed
  all_forecasting                               target_history__water_drainage_count_lag_2        target_history                 True                   False                        False                        True             0.037736         passed
  all_forecasting                               target_history__water_drainage_count_lag_3        target_history                 True                   False                        False                        True             0.056604         passed
  all_forecasting                               target_history__water_drainage_count_lag_6        target_history                 True                   False                        False                        True             0.113208         passed
  all_forecasting                              target_history__water_drainage_count_lag_12        target_history                 True                   False                        False                        True             0.226415         passed
  all_forecasting                   target_history__water_drainage_count_roll3_mean_shift1        target_history                 True                   False                        False                        True             0.018868         passed
  all_forecasting                    target_history__water_drainage_count_roll3_sum_shift1        target_history                 True                   False                        False                        True             0.018868         passed
  all_forecasting                   target_history__water_drainage_count_roll6_mean_shift1        target_history                 True                   False                        False                        True             0.018868         passed
  all_forecasting                    target_history__water_drainage_count_roll6_sum_shift1        target_history                 True                   False                        False                        True             0.018868         passed
  all_forecasting                  target_history__water_drainage_count_roll12_mean_shift1        target_history                 True                   False                        False                        True             0.018868         passed
  all_forecasting                   target_history__water_drainage_count_roll12_sum_shift1        target_history                 True                   False                        False                        True             0.018868         passed
  all_forecasting               target_history__water_drainage_count_expanding_mean_shift1        target_history                 True                   False                        False                        True             0.018868         passed
  all_forecasting                                               target_train_summary__mean  target_train_summary                 True                   False                        False                        True             0.000000         passed
  all_forecasting                                             target_train_summary__median  target_train_summary                 True                   False                        False                        True             0.000000         passed
  all_forecasting                                                target_train_summary__p90  target_train_summary                 True                   False                        False                        True             0.000000         passed
  all_forecasting                                      target_train_summary__positive_rate  target_train_summary                 True                   False                        False                        True             0.000000         passed
  all_forecasting              reporting_history__total_311_count_non_water_drainage_lag_1      lagged_reporting                False                    True                        False                        True             0.018868         passed
  all_forecasting              reporting_history__total_311_count_non_water_drainage_lag_2      lagged_reporting                False                    True                        False                        True             0.037736         passed
  all_forecasting              reporting_history__total_311_count_non_water_drainage_lag_3      lagged_reporting                False                    True                        False                        True             0.056604         passed
  all_forecasting             reporting_history__total_311_count_non_water_drainage_lag_12      lagged_reporting                False                    True                        False                        True             0.226415         passed
  all_forecasting  reporting_history__total_311_count_non_water_drainage_roll3_mean_shift1      lagged_reporting                False                    True                        False                        True             0.018868         passed
  all_forecasting  reporting_history__total_311_count_non_water_drainage_roll6_mean_shift1      lagged_reporting                False                    True                        False                        True             0.018868         passed
  all_forecasting reporting_history__total_311_count_non_water_drainage_roll12_mean_shift1      lagged_reporting                False                    True                        False                        True             0.018868         passed
  all_forecasting                                   requests_history__requests_total_lag_1 lagged_requests_total                False                    True                        False                        True             0.018868         passed
  all_forecasting                                   requests_history__requests_total_lag_3 lagged_requests_total                False                    True                        False                        True             0.056604         passed
  all_forecasting                                  requests_history__requests_total_lag_12 lagged_requests_total                False                    True                        False                        True             0.226415         passed
  all_forecasting                       requests_history__requests_total_roll3_mean_shift1 lagged_requests_total                False                    True                        False                        True             0.018868         passed
  all_forecasting                       requests_history__requests_total_roll6_mean_shift1 lagged_requests_total                False                    True                        False                        True             0.018868         passed
  all_forecasting                      requests_history__requests_total_roll12_mean_shift1 lagged_requests_total                False                    True                        False                        True             0.018868         passed
  all_forecasting                                              svi_primary__svi_percentile           svi_primary                False                   False                        False                        True             0.018519         passed
  all_forecasting                                               svi_primary__svi_score_raw           svi_primary                False                   False                        False                        True             0.018519         passed
  all_forecasting                                      static__log1p_population_total_2021                static                False                   False                        False                        True             0.000000         passed
  all_forecasting                                              static__log1p_land_area_km2                static                False                   False                        False                        True             0.000000         passed
  all_forecasting                                         static__log1p_population_density                static                False                   False                        False                        True             0.000000         passed
  all_forecasting                                         static_spatial__tract_centroid_x        static_spatial                False                   False                        False                        True             0.000000         passed
  all_forecasting                                         static_spatial__tract_centroid_y        static_spatial                False                   False                        False                        True             0.000000         passed
  all_forecasting                                       static_spatial__tract_centroid_lon        static_spatial                False                   False                        False                        True             0.000000         passed
  all_forecasting                                       static_spatial__tract_centroid_lat        static_spatial                False                   False                        False                        True             0.000000         passed
  all_forecasting                                                    calendar__month_is_02              calendar                False                   False                        False                        True             0.000000         passed
  all_forecasting                                                    calendar__month_is_03              calendar                False                   False                        False                        True             0.000000         passed
  all_forecasting                                                    calendar__month_is_04              calendar                False                   False                        False                        True             0.000000         passed
  all_forecasting                                                    calendar__month_is_05              calendar                False                   False                        False                        True             0.000000         passed
  all_forecasting                                                    calendar__month_is_06              calendar                False                   False                        False                        True             0.000000         passed
  all_forecasting                                                    calendar__month_is_07              calendar                False                   False                        False                        True             0.000000         passed
  all_forecasting                                                    calendar__month_is_08              calendar                False                   False                        False                        True             0.000000         passed
  all_forecasting                                                    calendar__month_is_09              calendar                False                   False                        False                        True             0.000000         passed
  all_forecasting                                                    calendar__month_is_10              calendar                False                   False                        False                        True             0.000000         passed
  all_forecasting                                                    calendar__month_is_11              calendar                False                   False                        False                        True             0.000000         passed
  all_forecasting                                                    calendar__month_is_12              calendar                False                   False                        False                        True             0.000000         passed
  all_forecasting                                                      calendar__month_sin              calendar                False                   False                        False                        True             0.000000         passed
  all_forecasting                                                      calendar__month_cos              calendar                False                   False                        False                        True             0.000000         passed
  all_forecasting                                       calendar__period_index_since_start              calendar                False                   False                        False                        True             0.000000         passed
 lagged_reporting              reporting_history__total_311_count_non_water_drainage_lag_1      lagged_reporting                False                    True                        False                        True             0.018868         passed
 lagged_reporting              reporting_history__total_311_count_non_water_drainage_lag_2      lagged_reporting                False                    True                        False                        True             0.037736         passed
 lagged_reporting              reporting_history__total_311_count_non_water_drainage_lag_3      lagged_reporting                False                    True                        False                        True             0.056604         passed
 lagged_reporting             reporting_history__total_311_count_non_water_drainage_lag_12      lagged_reporting                False                    True                        False                        True             0.226415         passed
 lagged_reporting  reporting_history__total_311_count_non_water_drainage_roll3_mean_shift1      lagged_reporting                False                    True                        False                        True             0.018868         passed
 lagged_reporting  reporting_history__total_311_count_non_water_drainage_roll6_mean_shift1      lagged_reporting                False                    True                        False                        True             0.018868         passed
 lagged_reporting reporting_history__total_311_count_non_water_drainage_roll12_mean_shift1      lagged_reporting                False                    True                        False                        True             0.018868         passed
 lagged_reporting                                   requests_history__requests_total_lag_1 lagged_requests_total                False                    True                        False                        True             0.018868         passed
 lagged_reporting                                   requests_history__requests_total_lag_3 lagged_requests_total                False                    True                        False                        True             0.056604         passed
 lagged_reporting                                  requests_history__requests_total_lag_12 lagged_requests_total                False                    True                        False                        True             0.226415         passed
 lagged_reporting                       requests_history__requests_total_roll3_mean_shift1 lagged_requests_total                False                    True                        False                        True             0.018868         passed
 lagged_reporting                       requests_history__requests_total_roll6_mean_shift1 lagged_requests_total                False                    True                        False                        True             0.018868         passed
 lagged_reporting                      requests_history__requests_total_roll12_mean_shift1 lagged_requests_total                False                    True                        False                        True             0.018868         passed
 lagged_reporting                                                    calendar__month_is_02              calendar                False                   False                        False                        True             0.000000         passed
 lagged_reporting                                                    calendar__month_is_03              calendar                False                   False                        False                        True             0.000000         passed
 lagged_reporting                                                    calendar__month_is_04              calendar                False                   False                        False                        True             0.000000         passed
 lagged_reporting                                                    calendar__month_is_05              calendar                False                   False                        False                        True             0.000000         passed
 lagged_reporting                                                    calendar__month_is_06              calendar                False                   False                        False                        True             0.000000         passed
 lagged_reporting                                                    calendar__month_is_07              calendar                False                   False                        False                        True             0.000000         passed
 lagged_reporting                                                    calendar__month_is_08              calendar                False                   False                        False                        True             0.000000         passed
 lagged_reporting                                                    calendar__month_is_09              calendar                False                   False                        False                        True             0.000000         passed
 lagged_reporting                                                    calendar__month_is_10              calendar                False                   False                        False                        True             0.000000         passed
 lagged_reporting                                                    calendar__month_is_11              calendar                False                   False                        False                        True             0.000000         passed
 lagged_reporting                                                    calendar__month_is_12              calendar                False                   False                        False                        True             0.000000         passed
 lagged_reporting                                                      calendar__month_sin              calendar                False                   False                        False                        True             0.000000         passed
 lagged_reporting                                                      calendar__month_cos              calendar                False                   False                        False                        True             0.000000         passed
 lagged_reporting                                       calendar__period_index_since_start              calendar                False                   False                        False                        True             0.000000         passed
no_target_history              reporting_history__total_311_count_non_water_drainage_lag_1      lagged_reporting                False                    True                        False                        True             0.018868         passed
```

## Edge construction notes

```json
{
  "spatial_knn_tract_pairs": 4320
}
```

## Diagnostic plots

| Plot | Path |
|---|---|
| `edge_type_counts` | `/home/tim/Documents/ville_ai/indices_BACKUP_before_clean_push/urban_graph_benchmark/outputs/mtl_311_water_v0/graphs/G1_tract_month_graph/plots/edge_type_counts.png` |
| `node_counts_by_split` | `/home/tim/Documents/ville_ai/indices_BACKUP_before_clean_push/urban_graph_benchmark/outputs/mtl_311_water_v0/graphs/G1_tract_month_graph/plots/node_counts_by_split.png` |
| `degree_distribution` | `/home/tim/Documents/ville_ai/indices_BACKUP_before_clean_push/urban_graph_benchmark/outputs/mtl_311_water_v0/graphs/G1_tract_month_graph/plots/degree_distribution_by_edge_type.png` |
| `target_mean_by_month` | `/home/tim/Documents/ville_ai/indices_BACKUP_before_clean_push/urban_graph_benchmark/outputs/mtl_311_water_v0/graphs/G1_tract_month_graph/plots/target_mean_by_month.png` |
| `spatial_knn_preview` | `/home/tim/Documents/ville_ai/indices_BACKUP_before_clean_push/urban_graph_benchmark/outputs/mtl_311_water_v0/graphs/G1_tract_month_graph/plots/spatial_preview__spatial_knn_same_month.png` |

## Output artifacts

| Artifact | Path |
|---|---|
| `node_table` | `/home/tim/Documents/ville_ai/indices_BACKUP_before_clean_push/urban_graph_benchmark/outputs/mtl_311_water_v0/graphs/G1_tract_month_graph/node_table.parquet` |
| `edge_table` | `/home/tim/Documents/ville_ai/indices_BACKUP_before_clean_push/urban_graph_benchmark/outputs/mtl_311_water_v0/graphs/G1_tract_month_graph/edge_table.parquet` |
| `target_vector` | `/home/tim/Documents/ville_ai/indices_BACKUP_before_clean_push/urban_graph_benchmark/outputs/mtl_311_water_v0/graphs/G1_tract_month_graph/target_vector.npy` |
| `binary_target_vector` | `/home/tim/Documents/ville_ai/indices_BACKUP_before_clean_push/urban_graph_benchmark/outputs/mtl_311_water_v0/graphs/G1_tract_month_graph/binary_target_vector.npy` |
| `split_masks` | `/home/tim/Documents/ville_ai/indices_BACKUP_before_clean_push/urban_graph_benchmark/outputs/mtl_311_water_v0/graphs/G1_tract_month_graph/split_masks.npz` |
| `edge_mask_by_split_regime` | `/home/tim/Documents/ville_ai/indices_BACKUP_before_clean_push/urban_graph_benchmark/outputs/mtl_311_water_v0/graphs/G1_tract_month_graph/edge_mask_by_split_regime.npz` |
| `edge_index_by_type` | `/home/tim/Documents/ville_ai/indices_BACKUP_before_clean_push/urban_graph_benchmark/outputs/mtl_311_water_v0/graphs/G1_tract_month_graph/edge_index_by_type.npz` |
| `edge_weight_by_type` | `/home/tim/Documents/ville_ai/indices_BACKUP_before_clean_push/urban_graph_benchmark/outputs/mtl_311_water_v0/graphs/G1_tract_month_graph/edge_weight_by_type.npz` |
| `graph_metadata` | `/home/tim/Documents/ville_ai/indices_BACKUP_before_clean_push/urban_graph_benchmark/outputs/mtl_311_water_v0/graphs/G1_tract_month_graph/graph_metadata.json` |
| `feature_audit` | `/home/tim/Documents/ville_ai/indices_BACKUP_before_clean_push/urban_graph_benchmark/outputs/mtl_311_water_v0/graphs/G1_tract_month_graph/feature_audit.csv` |
| `edge_audit` | `/home/tim/Documents/ville_ai/indices_BACKUP_before_clean_push/urban_graph_benchmark/outputs/mtl_311_water_v0/graphs/G1_tract_month_graph/edge_audit.csv` |
| `leakage_audit` | `/home/tim/Documents/ville_ai/indices_BACKUP_before_clean_push/urban_graph_benchmark/outputs/mtl_311_water_v0/graphs/G1_tract_month_graph/leakage_audit.csv` |
| `feature_matrix_metadata` | `/home/tim/Documents/ville_ai/indices_BACKUP_before_clean_push/urban_graph_benchmark/outputs/mtl_311_water_v0/graphs/G1_tract_month_graph/feature_matrix_metadata.json` |
| `plots_dir` | `/home/tim/Documents/ville_ai/indices_BACKUP_before_clean_push/urban_graph_benchmark/outputs/mtl_311_water_v0/graphs/G1_tract_month_graph/plots` |
| `feature_matrix:all_forecasting` | `/home/tim/Documents/ville_ai/indices_BACKUP_before_clean_push/urban_graph_benchmark/outputs/mtl_311_water_v0/graphs/G1_tract_month_graph/feature_matrix__all_forecasting__raw.npy` |
| `feature_matrix:lagged_reporting` | `/home/tim/Documents/ville_ai/indices_BACKUP_before_clean_push/urban_graph_benchmark/outputs/mtl_311_water_v0/graphs/G1_tract_month_graph/feature_matrix__lagged_reporting__raw.npy` |
| `feature_matrix:no_target_history` | `/home/tim/Documents/ville_ai/indices_BACKUP_before_clean_push/urban_graph_benchmark/outputs/mtl_311_water_v0/graphs/G1_tract_month_graph/feature_matrix__no_target_history__raw.npy` |
| `feature_columns:all_forecasting` | `/home/tim/Documents/ville_ai/indices_BACKUP_before_clean_push/urban_graph_benchmark/outputs/mtl_311_water_v0/graphs/G1_tract_month_graph/feature_columns__all_forecasting.json` |
| `feature_columns:lagged_reporting` | `/home/tim/Documents/ville_ai/indices_BACKUP_before_clean_push/urban_graph_benchmark/outputs/mtl_311_water_v0/graphs/G1_tract_month_graph/feature_columns__lagged_reporting.json` |
| `feature_columns:no_target_history` | `/home/tim/Documents/ville_ai/indices_BACKUP_before_clean_push/urban_graph_benchmark/outputs/mtl_311_water_v0/graphs/G1_tract_month_graph/feature_columns__no_target_history.json` |
| `feature_stats:all_forecasting` | `/home/tim/Documents/ville_ai/indices_BACKUP_before_clean_push/urban_graph_benchmark/outputs/mtl_311_water_v0/graphs/G1_tract_month_graph/feature_stats__all_forecasting.csv` |
| `feature_stats:lagged_reporting` | `/home/tim/Documents/ville_ai/indices_BACKUP_before_clean_push/urban_graph_benchmark/outputs/mtl_311_water_v0/graphs/G1_tract_month_graph/feature_stats__lagged_reporting.csv` |
| `feature_stats:no_target_history` | `/home/tim/Documents/ville_ai/indices_BACKUP_before_clean_push/urban_graph_benchmark/outputs/mtl_311_water_v0/graphs/G1_tract_month_graph/feature_stats__no_target_history.csv` |

## Reproduction metadata

```json
{
  "config_path": "/home/tim/Documents/ville_ai/indices_BACKUP_before_clean_push/urban_graph_benchmark/configs/mtl_311_water_v0.1.yaml",
  "config_hash": "581f0cdb52f7a6722c3718b06933823ca364e8a1cfb09a27401c430bbe8810ca",
  "panel_path": "/home/tim/Documents/ville_ai/indices_BACKUP_before_clean_push/urban_graph_benchmark/outputs/mtl_311_water_v0/datasets/tract_month_panel.parquet",
  "panel_sha256": "5192920f9705e28b422f53203fbab0e86e907078db2ad9b2c294496c87af9c5b",
  "split_path": "/home/tim/Documents/ville_ai/indices_BACKUP_before_clean_push/urban_graph_benchmark/outputs/mtl_311_water_v0/datasets/splits/split_assignments.parquet",
  "split_sha256": "32a280aed074450d65d3f1cd86f391e9e31710a089b62e41ec0ef2e08cd48b6c",
  "graph_build_config": {
    "config_path": "urban_graph_benchmark/configs/mtl_311_water_v0.1.yaml",
    "repo_root": null,
    "output_suffix": null,
    "feature_regimes": [
      "all_forecasting",
      "lagged_reporting",
      "no_target_history"
    ],
    "split_schemes": [
      "temporal",
      "random_debug",
      "spatial_block"
    ],
    "spatial_knn_k": 8,
    "spatial_weighting": "rbf",
    "include_knn_edges": true,
    "include_adjacency_edges": false,
    "include_temporal_lag1_edges": true,
    "include_temporal_lag12_edges": true,
    "include_random_placebo_edges": true,
    "random_seed": 42,
    "tract_geometry_path": null,
    "tract_geometry_id_col": null,
    "generate_diagnostic_plots": true,
    "plot_format": "png",
    "strict_leakage": true,
    "write_pyg_placeholders": true
  }
}
```

## Interpretation warnings

- This graph artifact supports multiple message-passing regimes. The default temporal experiment may use transductive node features with masked labels. Spatial-block experiments must report whether they use all edges, train-train edges, or no-test-incident edges.
- Cross-split edges may exist in the graph artifact. This is acceptable only under explicit transductive/inference assumptions where labels remain masked.
- Feature matrices are raw numeric matrices with NaNs preserved. Training scripts should fit imputation/scaling on train nodes only.
- Randomized placebo edges are for topology ablation, not for final predictive deployment.
- This graph is a typed spatiotemporal tract graph, not yet a full environmental HGNN with roads, drainage assets, green infrastructure, or critical-facility nodes.
