# G1.5 Validation-Selected Architecture Sweep

Generated at: `2026-06-15T18:30:17.796024+00:00`

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
    h64_L1_do0_layernorm_res_relmean_manual_lr0p001_wd0p0001 completed          64           1     0.00     layernorm      True             mean  manual            7                      7                   0        16.542801
     h64_L1_do0_layernorm_res_relsum_manual_lr0p001_wd0p0001 completed          64           1     0.00     layernorm      True              sum  manual            7                      7                   0        17.745773
 h64_L1_do0p05_layernorm_res_relmean_manual_lr0p001_wd0p0001 completed          64           1     0.05     layernorm      True             mean  manual            7                      7                   0        15.455271
  h64_L1_do0p05_layernorm_res_relsum_manual_lr0p001_wd0p0001 completed          64           1     0.05     layernorm      True              sum  manual            7                      7                   0        16.899248
 h64_L1_do0p15_layernorm_res_relmean_manual_lr0p001_wd0p0001 completed          64           1     0.15     layernorm      True             mean  manual            7                      7                   0        13.963421
  h64_L1_do0p15_layernorm_res_relsum_manual_lr0p001_wd0p0001 completed          64           1     0.15     layernorm      True              sum  manual            7                      7                   0        13.632804
    h64_L2_do0_layernorm_res_relmean_manual_lr0p001_wd0p0001 completed          64           2     0.00     layernorm      True             mean  manual            7                      7                   0        28.773962
     h64_L2_do0_layernorm_res_relsum_manual_lr0p001_wd0p0001 completed          64           2     0.00     layernorm      True              sum  manual            7                      7                   0        23.044939
 h64_L2_do0p05_layernorm_res_relmean_manual_lr0p001_wd0p0001 completed          64           2     0.05     layernorm      True             mean  manual            7                      7                   0        29.126972
  h64_L2_do0p05_layernorm_res_relsum_manual_lr0p001_wd0p0001 completed          64           2     0.05     layernorm      True              sum  manual            7                      7                   0        28.718658
 h64_L2_do0p15_layernorm_res_relmean_manual_lr0p001_wd0p0001 completed          64           2     0.15     layernorm      True             mean  manual            7                      7                   0        31.490216
  h64_L2_do0p15_layernorm_res_relsum_manual_lr0p001_wd0p0001 completed          64           2     0.15     layernorm      True              sum  manual            7                      7                   0        22.343075
    h64_L3_do0_layernorm_res_relmean_manual_lr0p001_wd0p0001 completed          64           3     0.00     layernorm      True             mean  manual            7                      7                   0        27.329010
     h64_L3_do0_layernorm_res_relsum_manual_lr0p001_wd0p0001 completed          64           3     0.00     layernorm      True              sum  manual            7                      7                   0        31.292686
 h64_L3_do0p05_layernorm_res_relmean_manual_lr0p001_wd0p0001 completed          64           3     0.05     layernorm      True             mean  manual            7                      7                   0        38.304220
  h64_L3_do0p05_layernorm_res_relsum_manual_lr0p001_wd0p0001 completed          64           3     0.05     layernorm      True              sum  manual            7                      7                   0        31.435185
 h64_L3_do0p15_layernorm_res_relmean_manual_lr0p001_wd0p0001 completed          64           3     0.15     layernorm      True             mean  manual            7                      7                   0        36.614860
  h64_L3_do0p15_layernorm_res_relsum_manual_lr0p001_wd0p0001 completed          64           3     0.15     layernorm      True              sum  manual            7                      7                   0        30.198757
   h128_L1_do0_layernorm_res_relmean_manual_lr0p001_wd0p0001 completed         128           1     0.00     layernorm      True             mean  manual            7                      7                   0        24.278136
    h128_L1_do0_layernorm_res_relsum_manual_lr0p001_wd0p0001 completed         128           1     0.00     layernorm      True              sum  manual            7                      7                   0        24.430349
h128_L1_do0p05_layernorm_res_relmean_manual_lr0p001_wd0p0001 completed         128           1     0.05     layernorm      True             mean  manual            7                      7                   0        19.094161
 h128_L1_do0p05_layernorm_res_relsum_manual_lr0p001_wd0p0001 completed         128           1     0.05     layernorm      True              sum  manual            7                      7                   0        22.373138
h128_L1_do0p15_layernorm_res_relmean_manual_lr0p001_wd0p0001 completed         128           1     0.15     layernorm      True             mean  manual            7                      7                   0        20.805382
 h128_L1_do0p15_layernorm_res_relsum_manual_lr0p001_wd0p0001 completed         128           1     0.15     layernorm      True              sum  manual            7                      7                   0        31.902581
   h128_L2_do0_layernorm_res_relmean_manual_lr0p001_wd0p0001 completed         128           2     0.00     layernorm      True             mean  manual            7                      7                   0        33.033844
    h128_L2_do0_layernorm_res_relsum_manual_lr0p001_wd0p0001 completed         128           2     0.00     layernorm      True              sum  manual            7                      7                   0        32.794482
h128_L2_do0p05_layernorm_res_relmean_manual_lr0p001_wd0p0001 completed         128           2     0.05     layernorm      True             mean  manual            7                      7                   0        54.924230
 h128_L2_do0p05_layernorm_res_relsum_manual_lr0p001_wd0p0001 completed         128           2     0.05     layernorm      True              sum  manual            7                      7                   0        49.222753
h128_L2_do0p15_layernorm_res_relmean_manual_lr0p001_wd0p0001 completed         128           2     0.15     layernorm      True             mean  manual            7                      7                   0        47.063867
 h128_L2_do0p15_layernorm_res_relsum_manual_lr0p001_wd0p0001 completed         128           2     0.15     layernorm      True              sum  manual            7                      7                   0        45.857793
   h128_L3_do0_layernorm_res_relmean_manual_lr0p001_wd0p0001 completed         128           3     0.00     layernorm      True             mean  manual            7                      7                   0        42.713439
    h128_L3_do0_layernorm_res_relsum_manual_lr0p001_wd0p0001 completed         128           3     0.00     layernorm      True              sum  manual            7                      7                   0        66.317715
h128_L3_do0p05_layernorm_res_relmean_manual_lr0p001_wd0p0001 completed         128           3     0.05     layernorm      True             mean  manual            7                      7                   0        52.666015
 h128_L3_do0p05_layernorm_res_relsum_manual_lr0p001_wd0p0001 completed         128           3     0.05     layernorm      True              sum  manual            7                      7                   0        50.505814
h128_L3_do0p15_layernorm_res_relmean_manual_lr0p001_wd0p0001 completed         128           3     0.15     layernorm      True             mean  manual            7                      7                   0        60.228808
 h128_L3_do0p15_layernorm_res_relsum_manual_lr0p001_wd0p0001 completed         128           3     0.15     layernorm      True              sum  manual            7                      7                   0        74.443755
```

## Selected representatives by family

```text
                family                                                                                                                                                    model_name  feature_regime            edge_regime       edge_mask_regime  hidden_dim  num_layers  dropout normalization  residual relation_combine     seed  validation_ndcg_at_10  validation_ndcg_at_25  validation_ndcg_at_50  validation_ndcg_at_100  validation_spearman  validation_mae  test_ndcg_at_10  test_ndcg_at_25  test_ndcg_at_50  test_ndcg_at_100  test_top10_overlap_rate  test_top25_overlap_rate  test_top50_overlap_rate  test_top100_overlap_rate  test_top_5pct_overlap_rate  test_top_10pct_overlap_rate  test_spearman  test_mae
              no_edges                             G1__spatial_block__all_forecasting__no_edges__all_edges__h64_L2_do0p05_layernorm_res_relmean_manual_lr0p001_wd0p0001_seed20240610 all_forecasting               no_edges              all_edges          64           2     0.05     layernorm      True             mean 20240610               0.552822               0.591507               0.609954                0.683941             0.668729        2.654501         0.661877         0.701481         0.767052          0.805768                      0.1                     0.32                     0.58                      0.68                    0.663462                     0.628019       0.763784  2.411409
random_spatial_placebo G1__spatial_block__all_forecasting__random_spatial_placebo__no_test_incident_edges__h128_L2_do0p05_layernorm_res_relmean_manual_lr0p001_wd0p0001_seed20240610 all_forecasting random_spatial_placebo no_test_incident_edges         128           2     0.05     layernorm      True             mean 20240610               0.648982               0.618520               0.650445                0.702533             0.676086        2.611960         0.668102         0.706299         0.735120          0.793265                      0.2                     0.32                     0.46                      0.63                    0.625000                     0.618357       0.731470  2.489836
      spatial_temporal       G1__spatial_block__all_forecasting__spatial_temporal__no_test_incident_edges__h128_L2_do0p15_layernorm_res_relmean_manual_lr0p001_wd0p0001_seed20240610 all_forecasting       spatial_temporal no_test_incident_edges         128           2     0.15     layernorm      True             mean 20240610               0.612838               0.623593               0.663764                0.708770             0.683357        2.693654         0.599979         0.656591         0.722124          0.805646                      0.0                     0.24                     0.46                      0.68                    0.673077                     0.628019       0.761439  2.683340
         temporal_only          G1__spatial_block__all_forecasting__temporal_only__no_test_incident_edges__h128_L3_do0p15_layernorm_res_relmean_manual_lr0p001_wd0p0001_seed20240610 all_forecasting          temporal_only no_test_incident_edges         128           3     0.15     layernorm      True             mean 20240610               0.640156               0.653132               0.683644                0.712044             0.682324        2.683592         0.560740         0.666059         0.738881          0.796453                      0.1                     0.28                     0.52                      0.68                    0.692308                     0.628019       0.766072  2.682400
```

## Final comparison

```text
                          comparison_role                 family            edge_regime       edge_mask_regime  hidden_dim  num_layers  dropout  test_mae  test_spearman  test_ndcg_at_10  test_ndcg_at_25  test_ndcg_at_50  test_ndcg_at_100  test_top10_overlap_rate  test_top25_overlap_rate  test_top50_overlap_rate  test_top100_overlap_rate  test_top_5pct_overlap_rate  test_top_10pct_overlap_rate
A3 frozen selected spatial-block baseline              A3_frozen             A3_tabular                    NaN         NaN         NaN      NaN  2.339043       0.762380         0.547230         0.634233         0.751944          0.784789                      0.0                     0.12                     0.54                      0.66                    0.663462                     0.618357
                     G1 selected no_edges               no_edges               no_edges              all_edges        64.0         2.0     0.05  2.411409       0.763784         0.661877         0.701481         0.767052          0.805768                      0.1                     0.32                     0.58                      0.68                    0.663462                     0.628019
       G1 selected random_spatial_placebo random_spatial_placebo random_spatial_placebo no_test_incident_edges       128.0         2.0     0.05  2.489836       0.731470         0.668102         0.706299         0.735120          0.793265                      0.2                     0.32                     0.46                      0.63                    0.625000                     0.618357
             G1 selected spatial_temporal       spatial_temporal       spatial_temporal no_test_incident_edges       128.0         2.0     0.15  2.683340       0.761439         0.599979         0.656591         0.722124          0.805646                      0.0                     0.24                     0.46                      0.68                    0.673077                     0.628019
                G1 selected temporal_only          temporal_only          temporal_only no_test_incident_edges       128.0         3.0     0.15  2.682400       0.766072         0.560740         0.666059         0.738881          0.796453                      0.1                     0.28                     0.52                      0.68                    0.692308                     0.628019
```

## Metric winners

```text
                     metric  higher_is_better                               winner_role          winner_family                                                                                                                                             winner_model_name  winner_value
                   test_mae             False A3 frozen selected spatial-block baseline              A3_frozen                                                                               hist_gradient_boosting_poisson__A3_lagged_reporting_forecasting__hgb_poisson_02      2.339043
              test_spearman              True                 G1 selected temporal_only          temporal_only          G1__spatial_block__all_forecasting__temporal_only__no_test_incident_edges__h128_L3_do0p15_layernorm_res_relmean_manual_lr0p001_wd0p0001_seed20240610      0.766072
            test_ndcg_at_10              True        G1 selected random_spatial_placebo random_spatial_placebo G1__spatial_block__all_forecasting__random_spatial_placebo__no_test_incident_edges__h128_L2_do0p05_layernorm_res_relmean_manual_lr0p001_wd0p0001_seed20240610      0.668102
            test_ndcg_at_25              True        G1 selected random_spatial_placebo random_spatial_placebo G1__spatial_block__all_forecasting__random_spatial_placebo__no_test_incident_edges__h128_L2_do0p05_layernorm_res_relmean_manual_lr0p001_wd0p0001_seed20240610      0.706299
            test_ndcg_at_50              True                      G1 selected no_edges               no_edges                             G1__spatial_block__all_forecasting__no_edges__all_edges__h64_L2_do0p05_layernorm_res_relmean_manual_lr0p001_wd0p0001_seed20240610      0.767052
           test_ndcg_at_100              True                      G1 selected no_edges               no_edges                             G1__spatial_block__all_forecasting__no_edges__all_edges__h64_L2_do0p05_layernorm_res_relmean_manual_lr0p001_wd0p0001_seed20240610      0.805768
    test_top10_overlap_rate              True        G1 selected random_spatial_placebo random_spatial_placebo G1__spatial_block__all_forecasting__random_spatial_placebo__no_test_incident_edges__h128_L2_do0p05_layernorm_res_relmean_manual_lr0p001_wd0p0001_seed20240610      0.200000
test_top_10pct_overlap_rate              True                      G1 selected no_edges               no_edges                             G1__spatial_block__all_forecasting__no_edges__all_edges__h64_L2_do0p05_layernorm_res_relmean_manual_lr0p001_wd0p0001_seed20240610      0.628019
```

## Factor summary preview

```text
          factor                  value  n_trials  mean_validation_ndcg_at_10  best_validation_ndcg_at_10  mean_validation_ndcg_at_25  best_validation_ndcg_at_25  mean_validation_ndcg_at_50  best_validation_ndcg_at_50  mean_validation_ndcg_at_100  best_validation_ndcg_at_100  mean_test_ndcg_at_10  best_test_ndcg_at_10  mean_test_ndcg_at_25  best_test_ndcg_at_25  mean_test_ndcg_at_50  best_test_ndcg_at_50  mean_test_ndcg_at_100  best_test_ndcg_at_100  mean_test_mae
  feature_regime        all_forecasting       252                    0.550572                    0.670536                    0.581876                    0.653132                    0.617136                    0.683644                     0.665336                     0.712044              0.588595              0.727680              0.653767              0.761188              0.715888              0.823267               0.763688               0.834718       2.515603
     edge_regime               no_edges        36                    0.543565                    0.605463                    0.575652                    0.640931                    0.606421                    0.653176                     0.657956                     0.683941              0.642325              0.727680              0.702142              0.761188              0.767401              0.786356               0.806505               0.824939       2.409641
     edge_regime random_spatial_placebo        72                    0.547974                    0.648982                    0.583305                    0.647267                    0.618590                    0.657839                     0.668132                     0.702533              0.569381              0.715194              0.635428              0.760844              0.697367              0.796223               0.750856               0.826510       2.531137
     edge_regime       spatial_temporal        72                    0.553834                    0.670536                    0.584788                    0.639174                    0.621111                    0.669669                     0.665349                     0.708770              0.580617              0.663502              0.644661              0.730523              0.709568              0.795768               0.756929               0.820615       2.524356
     edge_regime          temporal_only        72                    0.553413                    0.654424                    0.580648                    0.653132                    0.617065                    0.683644                     0.666216                     0.712044              0.588920              0.722181              0.657025              0.758784              0.714972              0.823267               0.761870               0.834718       2.544298
edge_mask_regime              all_edges       144                    0.551560                    0.657013                    0.582070                    0.653037                    0.615601                    0.668609                     0.664530                     0.711740              0.623635              0.727680              0.696841              0.761188              0.762710              0.823267               0.802732               0.834718       2.401999
edge_mask_regime no_test_incident_edges       108                    0.549256                    0.670536                    0.581617                    0.653132                    0.619183                    0.683644                     0.666410                     0.712044              0.541873              0.710394              0.596335              0.730523              0.653458              0.776356               0.711630               0.816051       2.667076
      hidden_dim                     64       126                    0.540490                    0.634109                    0.572964                    0.647267                    0.613842                    0.669669                     0.660404                     0.701258              0.582882              0.727680              0.647831              0.733984              0.707688              0.798653               0.758379               0.823347       2.523486
      hidden_dim                    128       126                    0.560655                    0.670536                    0.590788                    0.653132                    0.620430                    0.683644                     0.670268                     0.712044              0.594307              0.722181              0.659704              0.761188              0.724088              0.823267               0.768997               0.834718       2.507721
      num_layers                      1        84                    0.536716                    0.630978                    0.568698                    0.640931                    0.604601                    0.658034                     0.652292                     0.695238              0.578337              0.722181              0.642926              0.746510              0.706854              0.795991               0.757263               0.826168       2.506693
      num_layers                      2        84                    0.555466                    0.670536                    0.593338                    0.639174                    0.629376                    0.669669                     0.672991                     0.708770              0.587677              0.700728              0.659332              0.761188              0.721777              0.823267               0.768739               0.826212       2.526346
      num_layers                      3        84                    0.559535                    0.654424                    0.583592                    0.653132                    0.617432                    0.683644                     0.670724                     0.712044              0.599770              0.727680              0.659045              0.760844              0.719032              0.808360               0.765062               0.834718       2.513771
         dropout                    0.0        84                    0.545685                    0.633768                    0.575361                    0.646932                    0.611236                    0.658034                     0.661284                     0.693661              0.559333              0.715194              0.627982              0.761188              0.688820              0.795724               0.737035               0.826510       2.512977
         dropout                   0.05        84                    0.570154                    0.670536                    0.593917                    0.647267                    0.625856                    0.669669                     0.673620                     0.707930              0.595761              0.727680              0.661471              0.741010              0.721864              0.808360               0.772493               0.826168       2.455974
         dropout                   0.15        84                    0.535878                    0.654424                    0.576351                    0.653132                    0.614318                    0.683644                     0.661102                     0.712044              0.610690              0.722181              0.671849              0.758784              0.736979              0.823267               0.781537               0.834718       2.577859
   normalization              layernorm       252                    0.550572                    0.670536                    0.581876                    0.653132                    0.617136                    0.683644                     0.665336                     0.712044              0.588595              0.727680              0.653767              0.761188              0.715888              0.823267               0.763688               0.834718       2.515603
        residual                   True       252                    0.550572                    0.670536                    0.581876                    0.653132                    0.617136                    0.683644                     0.665336                     0.712044              0.588595              0.727680              0.653767              0.761188              0.715888              0.823267               0.763688               0.834718       2.515603
relation_combine                   mean       126                    0.545810                    0.670536                    0.578268                    0.653132                    0.612625                    0.683644                     0.661721                     0.712044              0.602579              0.727680              0.668994              0.761188              0.732619              0.823267               0.779751               0.826510       2.475152
relation_combine                    sum       126                    0.555335                    0.640809                    0.585485                    0.647267                    0.621648                    0.669669                     0.668951                     0.710708              0.574610              0.727680              0.638540              0.761188              0.699157              0.807176               0.747626               0.834718       2.556054
         backend                 manual       252                    0.550572                    0.670536                    0.581876                    0.653132                    0.617136                    0.683644                     0.665336                     0.712044              0.588595              0.727680              0.653767              0.761188              0.715888              0.823267               0.763688               0.834718       2.515603
            seed               20240610       252                    0.550572                    0.670536                    0.581876                    0.653132                    0.617136                    0.683644                     0.665336                     0.712044              0.588595              0.727680              0.653767              0.761188              0.715888              0.823267               0.763688               0.834718       2.515603
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
