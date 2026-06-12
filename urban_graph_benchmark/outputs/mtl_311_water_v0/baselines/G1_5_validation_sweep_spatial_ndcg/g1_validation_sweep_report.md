# G1.5 Validation-Selected Architecture Sweep

Generated at: `2026-06-12T16:10:34.716112+00:00`

Graph directory: `urban_graph_benchmark/outputs/mtl_311_water_v0/graphs/G1_tract_month_graph`

A3 comparison directory: `urban_graph_benchmark/outputs/mtl_311_water_v0/baselines/A3_feature_parity_tabular_spatial_block`

Output directory: `urban_graph_benchmark/outputs/mtl_311_water_v0/baselines/G1_5_validation_sweep_spatial_ndcg`

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
    64,
    128
  ],
  "num_layers": [
    1,
    2,
    3
  ],
  "dropouts": [
    0.0,
    0.05,
    0.15
  ],
  "normalizations": [
    "layernorm"
  ],
  "residual_options": [
    true
  ],
  "relation_combines": [
    "mean",
    "sum"
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
    h64_L1_do0_layernorm_res_relmean_manual_lr0p001_wd0p0001 completed          64           1     0.00     layernorm      True             mean  manual            7                      7                   0        16.673693
     h64_L1_do0_layernorm_res_relsum_manual_lr0p001_wd0p0001 completed          64           1     0.00     layernorm      True              sum  manual            7                      7                   0        16.537505
 h64_L1_do0p05_layernorm_res_relmean_manual_lr0p001_wd0p0001 completed          64           1     0.05     layernorm      True             mean  manual            7                      7                   0        15.443384
  h64_L1_do0p05_layernorm_res_relsum_manual_lr0p001_wd0p0001 completed          64           1     0.05     layernorm      True              sum  manual            7                      7                   0        17.594319
 h64_L1_do0p15_layernorm_res_relmean_manual_lr0p001_wd0p0001 completed          64           1     0.15     layernorm      True             mean  manual            7                      7                   0        13.787945
  h64_L1_do0p15_layernorm_res_relsum_manual_lr0p001_wd0p0001 completed          64           1     0.15     layernorm      True              sum  manual            7                      7                   0        13.232497
    h64_L2_do0_layernorm_res_relmean_manual_lr0p001_wd0p0001 completed          64           2     0.00     layernorm      True             mean  manual            7                      7                   0        29.826272
     h64_L2_do0_layernorm_res_relsum_manual_lr0p001_wd0p0001 completed          64           2     0.00     layernorm      True              sum  manual            7                      7                   0        25.399945
 h64_L2_do0p05_layernorm_res_relmean_manual_lr0p001_wd0p0001 completed          64           2     0.05     layernorm      True             mean  manual            7                      7                   0        28.609184
  h64_L2_do0p05_layernorm_res_relsum_manual_lr0p001_wd0p0001 completed          64           2     0.05     layernorm      True              sum  manual            7                      7                   0        28.426520
 h64_L2_do0p15_layernorm_res_relmean_manual_lr0p001_wd0p0001 completed          64           2     0.15     layernorm      True             mean  manual            7                      7                   0        31.203710
  h64_L2_do0p15_layernorm_res_relsum_manual_lr0p001_wd0p0001 completed          64           2     0.15     layernorm      True              sum  manual            7                      7                   0        22.463059
    h64_L3_do0_layernorm_res_relmean_manual_lr0p001_wd0p0001 completed          64           3     0.00     layernorm      True             mean  manual            7                      7                   0        28.616733
     h64_L3_do0_layernorm_res_relsum_manual_lr0p001_wd0p0001 completed          64           3     0.00     layernorm      True              sum  manual            7                      7                   0        27.767135
 h64_L3_do0p05_layernorm_res_relmean_manual_lr0p001_wd0p0001 completed          64           3     0.05     layernorm      True             mean  manual            7                      7                   0        38.146771
  h64_L3_do0p05_layernorm_res_relsum_manual_lr0p001_wd0p0001 completed          64           3     0.05     layernorm      True              sum  manual            7                      7                   0        31.333045
 h64_L3_do0p15_layernorm_res_relmean_manual_lr0p001_wd0p0001 completed          64           3     0.15     layernorm      True             mean  manual            7                      7                   0        36.806007
  h64_L3_do0p15_layernorm_res_relsum_manual_lr0p001_wd0p0001 completed          64           3     0.15     layernorm      True              sum  manual            7                      7                   0        30.366102
   h128_L1_do0_layernorm_res_relmean_manual_lr0p001_wd0p0001 completed         128           1     0.00     layernorm      True             mean  manual            7                      7                   0        26.518804
    h128_L1_do0_layernorm_res_relsum_manual_lr0p001_wd0p0001 completed         128           1     0.00     layernorm      True              sum  manual            7                      7                   0        25.568957
h128_L1_do0p05_layernorm_res_relmean_manual_lr0p001_wd0p0001 completed         128           1     0.05     layernorm      True             mean  manual            7                      7                   0        21.590243
 h128_L1_do0p05_layernorm_res_relsum_manual_lr0p001_wd0p0001 completed         128           1     0.05     layernorm      True              sum  manual            7                      7                   0        22.247074
h128_L1_do0p15_layernorm_res_relmean_manual_lr0p001_wd0p0001 completed         128           1     0.15     layernorm      True             mean  manual            7                      7                   0        24.194904
 h128_L1_do0p15_layernorm_res_relsum_manual_lr0p001_wd0p0001 completed         128           1     0.15     layernorm      True              sum  manual            7                      7                   0        30.412684
   h128_L2_do0_layernorm_res_relmean_manual_lr0p001_wd0p0001 completed         128           2     0.00     layernorm      True             mean  manual            7                      7                   0        31.896137
    h128_L2_do0_layernorm_res_relsum_manual_lr0p001_wd0p0001 completed         128           2     0.00     layernorm      True              sum  manual            7                      7                   0        33.971053
h128_L2_do0p05_layernorm_res_relmean_manual_lr0p001_wd0p0001 completed         128           2     0.05     layernorm      True             mean  manual            7                      7                   0        55.315001
 h128_L2_do0p05_layernorm_res_relsum_manual_lr0p001_wd0p0001 completed         128           2     0.05     layernorm      True              sum  manual            7                      7                   0        48.668782
h128_L2_do0p15_layernorm_res_relmean_manual_lr0p001_wd0p0001 completed         128           2     0.15     layernorm      True             mean  manual            7                      7                   0        52.283010
 h128_L2_do0p15_layernorm_res_relsum_manual_lr0p001_wd0p0001 completed         128           2     0.15     layernorm      True              sum  manual            7                      7                   0        43.683372
   h128_L3_do0_layernorm_res_relmean_manual_lr0p001_wd0p0001 completed         128           3     0.00     layernorm      True             mean  manual            7                      7                   0        42.808954
    h128_L3_do0_layernorm_res_relsum_manual_lr0p001_wd0p0001 completed         128           3     0.00     layernorm      True              sum  manual            7                      7                   0        66.219771
h128_L3_do0p05_layernorm_res_relmean_manual_lr0p001_wd0p0001 completed         128           3     0.05     layernorm      True             mean  manual            7                      7                   0        53.511696
 h128_L3_do0p05_layernorm_res_relsum_manual_lr0p001_wd0p0001 completed         128           3     0.05     layernorm      True              sum  manual            7                      7                   0        51.460417
h128_L3_do0p15_layernorm_res_relmean_manual_lr0p001_wd0p0001 completed         128           3     0.15     layernorm      True             mean  manual            7                      7                   0        60.535613
 h128_L3_do0p15_layernorm_res_relsum_manual_lr0p001_wd0p0001 completed         128           3     0.15     layernorm      True              sum  manual            7                      7                   0        74.690602
```

## Selected representatives by family

```text
                family                                                                                                                                              model_name  feature_regime            edge_regime       edge_mask_regime  hidden_dim  num_layers  dropout normalization  residual relation_combine     seed  validation_ndcg_at_100  validation_spearman  validation_mae  test_ndcg_at_100  test_spearman  test_top_10pct_overlap_rate  test_mae
              no_edges                       G1__spatial_block__all_forecasting__no_edges__all_edges__h64_L2_do0p05_layernorm_res_relmean_manual_lr0p001_wd0p0001_seed20240610 all_forecasting               no_edges              all_edges          64           2     0.05     layernorm      True             mean 20240610                0.683941             0.668729        2.654501          0.805768       0.763784                     0.628019  2.411409
random_spatial_placebo        G1__spatial_block__all_forecasting__random_spatial_placebo__all_edges__h128_L2_do0p05_layernorm_res_relmean_manual_lr0p001_wd0p0001_seed20240610 all_forecasting random_spatial_placebo              all_edges         128           2     0.05     layernorm      True             mean 20240610                0.704806             0.676240        2.620738          0.799908       0.777173                     0.642512  2.293357
      spatial_temporal G1__spatial_block__all_forecasting__spatial_temporal__no_test_incident_edges__h128_L2_do0p05_layernorm_res_relmean_manual_lr0p001_wd0p0001_seed20240610 all_forecasting       spatial_temporal no_test_incident_edges         128           2     0.05     layernorm      True             mean 20240610                0.706902             0.675649        2.620016          0.802691       0.741807                     0.603865  2.418032
         temporal_only    G1__spatial_block__all_forecasting__temporal_only__no_test_incident_edges__h128_L3_do0p15_layernorm_res_relmean_manual_lr0p001_wd0p0001_seed20240610 all_forecasting          temporal_only no_test_incident_edges         128           3     0.15     layernorm      True             mean 20240610                0.712044             0.682324        2.683592          0.796453       0.766072                     0.628019  2.682400
```

## Final comparison

```text
                          comparison_role                 family            edge_regime       edge_mask_regime  hidden_dim  num_layers  dropout  test_mae  test_spearman  test_ndcg_at_100  test_top_10pct_overlap_rate
A3 frozen selected spatial-block baseline              A3_frozen             A3_tabular                    NaN         NaN         NaN      NaN  2.339043       0.762380          0.784789                     0.618357
                     G1 selected no_edges               no_edges               no_edges              all_edges        64.0         2.0     0.05  2.411409       0.763784          0.805768                     0.628019
       G1 selected random_spatial_placebo random_spatial_placebo random_spatial_placebo              all_edges       128.0         2.0     0.05  2.293357       0.777173          0.799908                     0.642512
             G1 selected spatial_temporal       spatial_temporal       spatial_temporal no_test_incident_edges       128.0         2.0     0.05  2.418032       0.741807          0.802691                     0.603865
                G1 selected temporal_only          temporal_only          temporal_only no_test_incident_edges       128.0         3.0     0.15  2.682400       0.766072          0.796453                     0.628019
```

## Metric winners

```text
                     metric  higher_is_better                        winner_role          winner_family                                                                                                                                winner_model_name  winner_value
                   test_mae             False G1 selected random_spatial_placebo random_spatial_placebo G1__spatial_block__all_forecasting__random_spatial_placebo__all_edges__h128_L2_do0p05_layernorm_res_relmean_manual_lr0p001_wd0p0001_seed20240610      2.293357
              test_spearman              True G1 selected random_spatial_placebo random_spatial_placebo G1__spatial_block__all_forecasting__random_spatial_placebo__all_edges__h128_L2_do0p05_layernorm_res_relmean_manual_lr0p001_wd0p0001_seed20240610      0.777173
           test_ndcg_at_100              True               G1 selected no_edges               no_edges                G1__spatial_block__all_forecasting__no_edges__all_edges__h64_L2_do0p05_layernorm_res_relmean_manual_lr0p001_wd0p0001_seed20240610      0.805768
test_top_10pct_overlap_rate              True G1 selected random_spatial_placebo random_spatial_placebo G1__spatial_block__all_forecasting__random_spatial_placebo__all_edges__h128_L2_do0p05_layernorm_res_relmean_manual_lr0p001_wd0p0001_seed20240610      0.642512
```

## Factor summary preview

```text
          factor                  value  n_trials  mean_validation_ndcg_at_100  best_validation_ndcg_at_100  mean_test_ndcg_at_100  best_test_ndcg_at_100  mean_test_mae
  feature_regime        all_forecasting       252                     0.665802                     0.712044               0.763546               0.834718       2.517354
     edge_regime               no_edges        36                     0.657956                     0.683941               0.806505               0.824939       2.409641
     edge_regime random_spatial_placebo        72                     0.669728                     0.704806               0.750892               0.826082       2.536589
     edge_regime       spatial_temporal        72                     0.665384                     0.706902               0.756396               0.822787       2.525030
     edge_regime          temporal_only        72                     0.666216                     0.712044               0.761870               0.834718       2.544298
edge_mask_regime              all_edges       144                     0.665028                     0.711740               0.802920               0.834718       2.401209
edge_mask_regime no_test_incident_edges       108                     0.666833                     0.712044               0.711047               0.815973       2.672214
      hidden_dim                     64       126                     0.660529                     0.701230               0.758264               0.824496       2.523065
      hidden_dim                    128       126                     0.671074                     0.712044               0.768828               0.834718       2.511642
      num_layers                      1        84                     0.652780                     0.695238               0.757192               0.826168       2.504491
      num_layers                      2        84                     0.673769                     0.706902               0.768692               0.826212       2.526923
      num_layers                      3        84                     0.670857                     0.712044               0.764753               0.834718       2.520647
         dropout                    0.0        84                     0.662020                     0.693661               0.737113               0.826082       2.519069
         dropout                   0.05        84                     0.673933                     0.706902               0.772432               0.826168       2.455606
         dropout                   0.15        84                     0.661452                     0.712044               0.781093               0.834718       2.577386
   normalization              layernorm       252                     0.665802                     0.712044               0.763546               0.834718       2.517354
        residual                   True       252                     0.665802                     0.712044               0.763546               0.834718       2.517354
relation_combine                   mean       126                     0.662459                     0.712044               0.779818               0.826212       2.475889
relation_combine                    sum       126                     0.669145                     0.710708               0.747274               0.834718       2.558819
         backend                 manual       252                     0.665802                     0.712044               0.763546               0.834718       2.517354
            seed               20240610       252                     0.665802                     0.712044               0.763546               0.834718       2.517354
```

## Output artifacts

| Artifact | Path |
|---|---|
| `sweep_manifest` | `urban_graph_benchmark/outputs/mtl_311_water_v0/baselines/G1_5_validation_sweep_spatial_ndcg/sweep_manifest.csv` |
| `sweep_run_results` | `urban_graph_benchmark/outputs/mtl_311_water_v0/baselines/G1_5_validation_sweep_spatial_ndcg/sweep_run_results.csv` |
| `sweep_metrics` | `urban_graph_benchmark/outputs/mtl_311_water_v0/baselines/G1_5_validation_sweep_spatial_ndcg/sweep_metrics.csv` |
| `sweep_model_selection_audit` | `urban_graph_benchmark/outputs/mtl_311_water_v0/baselines/G1_5_validation_sweep_spatial_ndcg/sweep_model_selection_audit.csv` |
| `sweep_trial_audit` | `urban_graph_benchmark/outputs/mtl_311_water_v0/baselines/G1_5_validation_sweep_spatial_ndcg/sweep_trial_audit.csv` |
| `sweep_training_curves` | `urban_graph_benchmark/outputs/mtl_311_water_v0/baselines/G1_5_validation_sweep_spatial_ndcg/sweep_training_curves.csv` |
| `sweep_graph_regime_audit` | `urban_graph_benchmark/outputs/mtl_311_water_v0/baselines/G1_5_validation_sweep_spatial_ndcg/sweep_graph_regime_audit.csv` |
| `selection_by_family` | `urban_graph_benchmark/outputs/mtl_311_water_v0/baselines/G1_5_validation_sweep_spatial_ndcg/selection_by_family.csv` |
| `final_comparison` | `urban_graph_benchmark/outputs/mtl_311_water_v0/baselines/G1_5_validation_sweep_spatial_ndcg/final_comparison.csv` |
| `metric_winners` | `urban_graph_benchmark/outputs/mtl_311_water_v0/baselines/G1_5_validation_sweep_spatial_ndcg/metric_winners.csv` |
| `factor_summary` | `urban_graph_benchmark/outputs/mtl_311_water_v0/baselines/G1_5_validation_sweep_spatial_ndcg/factor_summary.csv` |
| `sweep_report` | `urban_graph_benchmark/outputs/mtl_311_water_v0/baselines/G1_5_validation_sweep_spatial_ndcg/g1_validation_sweep_report.md` |
| `sweep_metadata` | `urban_graph_benchmark/outputs/mtl_311_water_v0/baselines/G1_5_validation_sweep_spatial_ndcg/sweep_metadata.json` |

## Interpretation guardrails

A G1.5 result should be interpreted as graph-specific only if a graph edge-regime family beats the validation-selected no-edge neural control and the random spatial placebo, not only the frozen A3 tabular model. If the no-edge control also wins, the result supports neural ranking-oriented model selection more strongly than graph-specific message passing.
