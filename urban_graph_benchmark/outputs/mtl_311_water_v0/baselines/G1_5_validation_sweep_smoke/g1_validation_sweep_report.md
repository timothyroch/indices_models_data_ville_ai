# G1.5 Validation-Selected Architecture Sweep

Generated at: `2026-06-12T15:50:00.686006+00:00`

Graph directory: `urban_graph_benchmark/outputs/mtl_311_water_v0/graphs/G1_tract_month_graph`

A3 comparison directory: `urban_graph_benchmark/outputs/mtl_311_water_v0/baselines/A3_feature_parity_tabular_spatial_block`

Output directory: `urban_graph_benchmark/outputs/mtl_311_water_v0/baselines/G1_5_validation_sweep_smoke`

## Purpose

This sweep is a bounded validation-selected G1 architecture/model-selection step. It tests whether G1 performance under spatial-block evaluation is sensitive to architecture and validation monitor choice. It is not a new graph construction and does not replace the frozen A3 baseline.

## Selection protocol

Primary validation selection metric: `validation_ndcg_at_100` (higher is better).

Each edge-regime family is selected using validation data only. Test metrics are joined only after selection for reporting. The no-edge neural control and random spatial placebo are retained as first-class controls.

## Sweep space

```json
{
  "feature_regimes": [
    "all_forecasting"
  ],
  "split_scheme": "spatial_block",
  "edge_regimes": [
    "no_edges",
    "temporal_only",
    "spatial_temporal",
    "random_spatial_placebo"
  ],
  "edge_mask_regimes": [
    "all_edges",
    "no_test_incident_edges"
  ],
  "hidden_dims": [
    128
  ],
  "num_layers": [
    1,
    2
  ],
  "dropouts": [
    0.15
  ],
  "normalizations": [
    "layernorm"
  ],
  "residual_options": [
    true
  ],
  "relation_combines": [
    "mean"
  ],
  "backends": [
    "manual"
  ],
  "seeds": [
    20240610
  ]
}
```

## Run summary

```text
                                                      run_id    status  hidden_dim  num_layers  dropout normalization  residual relation_combine backend  trial_count  completed_trial_count  failed_trial_count  elapsed_seconds
h128_L1_do0p15_layernorm_res_relmean_manual_lr0p001_wd0p0001 completed         128           1     0.15     layernorm      True             mean  manual            7                      7                   0        13.666784
h128_L2_do0p15_layernorm_res_relmean_manual_lr0p001_wd0p0001 completed         128           2     0.15     layernorm      True             mean  manual            7                      7                   0        19.749729
```

## Selected representatives by family

```text
                family                                                                                                                                           model_name  feature_regime            edge_regime       edge_mask_regime  hidden_dim  num_layers  dropout normalization  residual relation_combine     seed  validation_ndcg_at_100  validation_spearman  validation_mae  test_ndcg_at_100  test_spearman  test_top_10pct_overlap_rate  test_mae
              no_edges                   G1__spatial_block__all_forecasting__no_edges__all_edges__h128_L1_do0p15_layernorm_res_relmean_manual_lr0p001_wd0p0001_seed20240610 all_forecasting               no_edges              all_edges         128           1     0.15     layernorm      True             mean 20240610                0.638574             0.661020        2.800278          0.800671       0.755050                     0.599034  2.530838
random_spatial_placebo     G1__spatial_block__all_forecasting__random_spatial_placebo__all_edges__h128_L2_do0p15_layernorm_res_relmean_manual_lr0p001_wd0p0001_seed20240610 all_forecasting random_spatial_placebo              all_edges         128           2     0.15     layernorm      True             mean 20240610                0.658058             0.669718        2.902435          0.804378       0.770923                     0.618357  2.562693
      spatial_temporal           G1__spatial_block__all_forecasting__spatial_temporal__all_edges__h128_L2_do0p15_layernorm_res_relmean_manual_lr0p001_wd0p0001_seed20240610 all_forecasting       spatial_temporal              all_edges         128           2     0.15     layernorm      True             mean 20240610                0.653769             0.670475        2.870272          0.807463       0.771639                     0.613527  2.544260
         temporal_only G1__spatial_block__all_forecasting__temporal_only__no_test_incident_edges__h128_L2_do0p15_layernorm_res_relmean_manual_lr0p001_wd0p0001_seed20240610 all_forecasting          temporal_only no_test_incident_edges         128           2     0.15     layernorm      True             mean 20240610                0.672348             0.668677        2.780659          0.806343       0.753733                     0.599034  2.476317
```

## Final comparison

```text
                          comparison_role                 family            edge_regime       edge_mask_regime  hidden_dim  num_layers  dropout  test_mae  test_spearman  test_ndcg_at_100  test_top_10pct_overlap_rate
A3 frozen selected spatial-block baseline              A3_frozen             A3_tabular                    NaN         NaN         NaN      NaN  2.339043       0.762380          0.784789                     0.618357
                     G1 selected no_edges               no_edges               no_edges              all_edges       128.0         1.0     0.15  2.530838       0.755050          0.800671                     0.599034
       G1 selected random_spatial_placebo random_spatial_placebo random_spatial_placebo              all_edges       128.0         2.0     0.15  2.562693       0.770923          0.804378                     0.618357
             G1 selected spatial_temporal       spatial_temporal       spatial_temporal              all_edges       128.0         2.0     0.15  2.544260       0.771639          0.807463                     0.613527
                G1 selected temporal_only          temporal_only          temporal_only no_test_incident_edges       128.0         2.0     0.15  2.476317       0.753733          0.806343                     0.599034
```

## Metric winners

```text
                     metric  higher_is_better                               winner_role    winner_family                                                                                                                          winner_model_name  winner_value
                   test_mae             False A3 frozen selected spatial-block baseline        A3_frozen                                                            hist_gradient_boosting_poisson__A3_lagged_reporting_forecasting__hgb_poisson_02      2.339043
              test_spearman              True              G1 selected spatial_temporal spatial_temporal G1__spatial_block__all_forecasting__spatial_temporal__all_edges__h128_L2_do0p15_layernorm_res_relmean_manual_lr0p001_wd0p0001_seed20240610      0.771639
           test_ndcg_at_100              True              G1 selected spatial_temporal spatial_temporal G1__spatial_block__all_forecasting__spatial_temporal__all_edges__h128_L2_do0p15_layernorm_res_relmean_manual_lr0p001_wd0p0001_seed20240610      0.807463
test_top_10pct_overlap_rate              True A3 frozen selected spatial-block baseline        A3_frozen                                                            hist_gradient_boosting_poisson__A3_lagged_reporting_forecasting__hgb_poisson_02      0.618357
```

## Factor summary preview

```text
          factor                  value  n_trials  mean_validation_ndcg_at_100  best_validation_ndcg_at_100  mean_test_ndcg_at_100  best_test_ndcg_at_100  mean_test_mae
  feature_regime        all_forecasting        14                     0.634483                     0.672348               0.784000               0.831887       2.566032
     edge_regime               no_edges         2                     0.602908                     0.638574               0.771103               0.800671       2.635759
     edge_regime random_spatial_placebo         4                     0.641372                     0.658058               0.788504               0.804378       2.564301
     edge_regime       spatial_temporal         4                     0.637979                     0.653769               0.782079               0.807463       2.559076
     edge_regime          temporal_only         4                     0.639885                     0.672348               0.787866               0.831887       2.539854
edge_mask_regime              all_edges         8                     0.630968                     0.670950               0.791997               0.831887       2.576180
edge_mask_regime no_test_incident_edges         6                     0.639170                     0.672348               0.773337               0.806343       2.552501
      hidden_dim                    128        14                     0.634483                     0.672348               0.784000               0.831887       2.566032
      num_layers                      1         7                     0.621570                     0.638574               0.769153               0.800671       2.578705
      num_layers                      2         7                     0.647396                     0.672348               0.798847               0.831887       2.553358
         dropout                   0.15        14                     0.634483                     0.672348               0.784000               0.831887       2.566032
   normalization              layernorm        14                     0.634483                     0.672348               0.784000               0.831887       2.566032
        residual                   True        14                     0.634483                     0.672348               0.784000               0.831887       2.566032
relation_combine                   mean        14                     0.634483                     0.672348               0.784000               0.831887       2.566032
         backend                 manual        14                     0.634483                     0.672348               0.784000               0.831887       2.566032
            seed               20240610        14                     0.634483                     0.672348               0.784000               0.831887       2.566032
```

## Output artifacts

| Artifact | Path |
|---|---|
| `sweep_manifest` | `urban_graph_benchmark/outputs/mtl_311_water_v0/baselines/G1_5_validation_sweep_smoke/sweep_manifest.csv` |
| `sweep_run_results` | `urban_graph_benchmark/outputs/mtl_311_water_v0/baselines/G1_5_validation_sweep_smoke/sweep_run_results.csv` |
| `sweep_metrics` | `urban_graph_benchmark/outputs/mtl_311_water_v0/baselines/G1_5_validation_sweep_smoke/sweep_metrics.csv` |
| `sweep_model_selection_audit` | `urban_graph_benchmark/outputs/mtl_311_water_v0/baselines/G1_5_validation_sweep_smoke/sweep_model_selection_audit.csv` |
| `sweep_trial_audit` | `urban_graph_benchmark/outputs/mtl_311_water_v0/baselines/G1_5_validation_sweep_smoke/sweep_trial_audit.csv` |
| `sweep_training_curves` | `urban_graph_benchmark/outputs/mtl_311_water_v0/baselines/G1_5_validation_sweep_smoke/sweep_training_curves.csv` |
| `sweep_graph_regime_audit` | `urban_graph_benchmark/outputs/mtl_311_water_v0/baselines/G1_5_validation_sweep_smoke/sweep_graph_regime_audit.csv` |
| `selection_by_family` | `urban_graph_benchmark/outputs/mtl_311_water_v0/baselines/G1_5_validation_sweep_smoke/selection_by_family.csv` |
| `final_comparison` | `urban_graph_benchmark/outputs/mtl_311_water_v0/baselines/G1_5_validation_sweep_smoke/final_comparison.csv` |
| `metric_winners` | `urban_graph_benchmark/outputs/mtl_311_water_v0/baselines/G1_5_validation_sweep_smoke/metric_winners.csv` |
| `factor_summary` | `urban_graph_benchmark/outputs/mtl_311_water_v0/baselines/G1_5_validation_sweep_smoke/factor_summary.csv` |
| `sweep_report` | `urban_graph_benchmark/outputs/mtl_311_water_v0/baselines/G1_5_validation_sweep_smoke/g1_validation_sweep_report.md` |
| `sweep_metadata` | `urban_graph_benchmark/outputs/mtl_311_water_v0/baselines/G1_5_validation_sweep_smoke/sweep_metadata.json` |

## Interpretation guardrails

A G1.5 result should be interpreted as graph-specific only if a graph edge-regime family beats the validation-selected no-edge neural control and the random spatial placebo, not only the frozen A3 tabular model. If the no-edge control also wins, the result supports neural ranking-oriented model selection more strongly than graph-specific message passing.
