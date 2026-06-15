# A1 SVI Sensitivity Analysis — Montréal 311 Water/Drainage v0

Generated at: `2026-06-15T16:39:09.250024+00:00`

## Purpose

This block expands the raw A1 SVI benchmark with non-fitted sensitivity variants. The primary A1 test compares static tract-level SVI to monthly tract-level water/drainage 311 burden. That strict tract-month test is useful, but harsh: SVI is static and social-vulnerability-oriented, while 311 burden is dynamic, hazard-dependent, and affected by reporting behavior. The variants below test whether the weak strict A1 result is robust to tract-level, population-normalized, and reporting-normalized burden definitions.

**Methodological boundary:** no regression, calibration, or ML model is fitted from SVI to the 311 target. SVI remains the direct ranking score in every variant, so these remain A1-style index baselines rather than A2 calibrated SVI models.

## Sensitivity variants

| Variant | Unit | Target definition | Fitted model? |
|---|---|---|---|
| `A1a` | tract-month | monthly raw water/drainage count | no |
| `A1b` | tract | mean monthly water/drainage count | no |
| `A1c` | tract | total water/drainage count | no |
| `A1d` | tract | water/drainage reports per 1,000 residents | no |
| `A1e` | tract | water/drainage reports as a share of all 311 reports | no |
| `A1f` | tract | excess water/drainage burden over the all-311 reporting baseline | no |
| `A1g` | tract | data-defined water/drainage surge-window burden | no |

## Row counts by split scheme

| Split scheme | Train | Validation | Test |
|---|---:|---:|---:|
| `temporal` | 19440 | 4320 | 4860 |
| `spatial_block` | 21200 | 5353 | 2067 |

## Primary SVI score columns

| Source column | Oriented score column | Non-missing |
|---|---|---:|
| `svi_percentile` | `svi_percentile__higher_more_vulnerable` | 28090 |
| `svi_score_raw` | `svi_score_raw__higher_more_vulnerable` | 28090 |

## Denominator audit

| Field | Value |
|---|---|
| `population_source_column` | `population_total_2021` |
| `all311_source_column` | `total_311_count_all` |
| `non_water_311_source_column` | `None` |
| `all311_method` | `direct_all311_column` |
| `min_population` | `100.0` |
| `min_all311` | `10.0` |
| `all311_warning` | `None` |
| `population_non_missing_rows` | `28620` |
| `all311_non_missing_rows` | `28620` |
| `surge_months_by_scheme` | `{'temporal': (6, 8), 'spatial_block': (5, 8)}` |
| `a1f_reporting_baseline` | `train-derived water share of all 311 activity` |

## Compact sensitivity metrics

```text
              split_name                     sensitivity_variant_id source_svi_column  spearman_corr  kendall_corr  ndcg_at_10  ndcg_at_25  ndcg_at_50  ndcg_at_100  top_10_overlap_rate  top_25_overlap_rate  top_50_overlap_rate  top_100_overlap_rate  top_5pct_overlap_rate  top_10pct_overlap_rate  n_valid_for_variant
      spatial_block_test           A1a_strict_tract_month_raw_count    svi_percentile       0.098558      0.069047    0.067009    0.073543    0.072441     0.149799                  0.0                 0.00                 0.00                  0.01               0.010101                0.030457                 1961
      spatial_block_test           A1a_strict_tract_month_raw_count     svi_score_raw       0.098558      0.069047    0.067009    0.073543    0.072441     0.149799                  0.0                 0.00                 0.00                  0.01               0.010101                0.030457                 1961
      spatial_block_test                      A1b_tract_mean_burden    svi_percentile       0.105039      0.060150    0.397163    0.542315    0.688923     0.688923                  0.2                 0.72                 1.00                  1.00               0.000000                0.000000                   37
      spatial_block_test                      A1b_tract_mean_burden     svi_score_raw       0.105039      0.060150    0.397163    0.542315    0.688923     0.688923                  0.2                 0.72                 1.00                  1.00               0.000000                0.000000                   37
      spatial_block_test                     A1c_tract_total_burden    svi_percentile       0.105039      0.060150    0.397163    0.542315    0.688923     0.688923                  0.2                 0.72                 1.00                  1.00               0.000000                0.000000                   37
      spatial_block_test                     A1c_tract_total_burden     svi_score_raw       0.105039      0.060150    0.397163    0.542315    0.688923     0.688923                  0.2                 0.72                 1.00                  1.00               0.000000                0.000000                   37
      spatial_block_test           A1d_population_normalized_burden    svi_percentile      -0.066983     -0.057143    0.433624    0.590272    0.748207     0.748207                  0.0                 0.68                 1.00                  1.00               0.000000                0.000000                   37
      spatial_block_test           A1d_population_normalized_burden     svi_score_raw      -0.066983     -0.057143    0.433624    0.590272    0.748207     0.748207                  0.0                 0.68                 1.00                  1.00               0.000000                0.000000                   37
      spatial_block_test             A1e_reporting_normalized_share    svi_percentile      -0.024652     -0.026913    0.740402    0.830498    0.906253     0.906253                  0.2                 0.76                 1.00                  1.00               0.000000                0.000000                   35
      spatial_block_test             A1e_reporting_normalized_share     svi_score_raw      -0.024652     -0.026913    0.740402    0.830498    0.906253     0.906253                  0.2                 0.76                 1.00                  1.00               0.000000                0.000000                   35
      spatial_block_test            A1f_reporting_excess_difference    svi_percentile      -0.011065     -0.016821    0.347684    0.590670    0.691911     0.691911                  0.2                 0.72                 1.00                  1.00               0.000000                0.000000                   35
      spatial_block_test            A1f_reporting_excess_difference     svi_score_raw      -0.011065     -0.016821    0.347684    0.590670    0.691911     0.691911                  0.2                 0.72                 1.00                  1.00               0.000000                0.000000                   35
      spatial_block_test                 A1f_reporting_excess_ratio    svi_percentile      -0.024652     -0.026913    0.740402    0.830498    0.906253     0.906253                  0.2                 0.76                 1.00                  1.00               0.000000                0.000000                   35
      spatial_block_test                 A1f_reporting_excess_ratio     svi_score_raw      -0.024652     -0.026913    0.740402    0.830498    0.906253     0.906253                  0.2                 0.76                 1.00                  1.00               0.000000                0.000000                   35
      spatial_block_test  A1g_data_defined_surge_window_mean_burden    svi_percentile       0.040081      0.007536    0.348232    0.493553    0.657455     0.657455                  0.3                 0.68                 1.00                  1.00               0.000000                0.000000                   37
      spatial_block_test  A1g_data_defined_surge_window_mean_burden     svi_score_raw       0.040081      0.007536    0.348232    0.493553    0.657455     0.657455                  0.3                 0.68                 1.00                  1.00               0.000000                0.000000                   37
      spatial_block_test A1g_data_defined_surge_window_total_burden    svi_percentile       0.040081      0.007536    0.348232    0.493553    0.657455     0.657455                  0.3                 0.68                 1.00                  1.00               0.000000                0.000000                   37
      spatial_block_test A1g_data_defined_surge_window_total_burden     svi_score_raw       0.040081      0.007536    0.348232    0.493553    0.657455     0.657455                  0.3                 0.68                 1.00                  1.00               0.000000                0.000000                   37
spatial_block_validation           A1a_strict_tract_month_raw_count    svi_percentile       0.024326      0.017129    0.013025    0.017818    0.021980     0.052407                  0.0                 0.00                 0.00                  0.00               0.003774                0.007547                 5300
spatial_block_validation           A1a_strict_tract_month_raw_count     svi_score_raw       0.024326      0.017129    0.013025    0.017818    0.021980     0.052407                  0.0                 0.00                 0.00                  0.00               0.003774                0.007547                 5300
spatial_block_validation                      A1b_tract_mean_burden    svi_percentile       0.002652      0.006876    0.209777    0.368451    0.561268     0.754615                  0.0                 0.12                 0.50                  1.00               0.000000                0.000000                  100
spatial_block_validation                      A1b_tract_mean_burden     svi_score_raw       0.002652      0.006876    0.209777    0.368451    0.561268     0.754615                  0.0                 0.12                 0.50                  1.00               0.000000                0.000000                  100
spatial_block_validation                     A1c_tract_total_burden    svi_percentile       0.002652      0.006876    0.209777    0.368451    0.561268     0.754615                  0.0                 0.12                 0.50                  1.00               0.000000                0.000000                  100
spatial_block_validation                     A1c_tract_total_burden     svi_score_raw       0.002652      0.006876    0.209777    0.368451    0.561268     0.754615                  0.0                 0.12                 0.50                  1.00               0.000000                0.000000                  100
spatial_block_validation           A1d_population_normalized_burden    svi_percentile      -0.469583     -0.321212    0.199649    0.340491    0.492942     0.739916                  0.0                 0.08                 0.28                  1.00               0.000000                0.000000                  100
spatial_block_validation           A1d_population_normalized_burden     svi_score_raw      -0.469583     -0.321212    0.199649    0.340491    0.492942     0.739916                  0.0                 0.08                 0.28                  1.00               0.000000                0.000000                  100
spatial_block_validation             A1e_reporting_normalized_share    svi_percentile       0.122527      0.076169    0.621854    0.727418    0.818006     0.915832                  0.0                 0.24                 0.58                  1.00               0.000000                0.000000                  100
spatial_block_validation             A1e_reporting_normalized_share     svi_score_raw       0.122527      0.076169    0.621854    0.727418    0.818006     0.915832                  0.0                 0.24                 0.58                  1.00               0.000000                0.000000                  100
spatial_block_validation            A1f_reporting_excess_difference    svi_percentile       0.120300      0.060202    0.385587    0.493529    0.655119     0.811280                  0.0                 0.20                 0.58                  1.00               0.000000                0.000000                  100
spatial_block_validation            A1f_reporting_excess_difference     svi_score_raw       0.120300      0.060202    0.385587    0.493529    0.655119     0.811280                  0.0                 0.20                 0.58                  1.00               0.000000                0.000000                  100
spatial_block_validation                 A1f_reporting_excess_ratio    svi_percentile       0.122527      0.076169    0.621854    0.727418    0.818006     0.915832                  0.0                 0.24                 0.58                  1.00               0.000000                0.000000                  100
spatial_block_validation                 A1f_reporting_excess_ratio     svi_score_raw       0.122527      0.076169    0.621854    0.727418    0.818006     0.915832                  0.0                 0.24                 0.58                  1.00               0.000000                0.000000                  100
spatial_block_validation  A1g_data_defined_surge_window_mean_burden    svi_percentile       0.032956      0.028633    0.195297    0.361742    0.547538     0.744621                  0.0                 0.16                 0.50                  1.00               0.000000                0.000000                  100
spatial_block_validation  A1g_data_defined_surge_window_mean_burden     svi_score_raw       0.032956      0.028633    0.195297    0.361742    0.547538     0.744621                  0.0                 0.16                 0.50                  1.00               0.000000                0.000000                  100
spatial_block_validation A1g_data_defined_surge_window_total_burden    svi_percentile       0.032956      0.028633    0.195297    0.361742    0.547538     0.744621                  0.0                 0.16                 0.50                  1.00               0.000000                0.000000                  100
spatial_block_validation A1g_data_defined_surge_window_total_burden     svi_score_raw       0.032956      0.028633    0.195297    0.361742    0.547538     0.744621                  0.0                 0.16                 0.50                  1.00               0.000000                0.000000                  100
           temporal_test           A1a_strict_tract_month_raw_count    svi_percentile       0.160639      0.111152    0.188418    0.254751    0.238587     0.220560                  0.0                 0.00                 0.00                  0.00               0.016736                0.052411                 4770
           temporal_test           A1a_strict_tract_month_raw_count     svi_score_raw       0.160630      0.111146    0.188418    0.254751    0.238587     0.220560                  0.0                 0.00                 0.00                  0.00               0.016736                0.052411                 4770
           temporal_test                      A1b_tract_mean_burden    svi_percentile       0.198890      0.131574    0.294005    0.286569    0.331540     0.419181                  0.0                 0.00                 0.06                  0.16               0.000000                0.056604                  530
           temporal_test                      A1b_tract_mean_burden     svi_score_raw       0.198879      0.131582    0.294005    0.286572    0.331542     0.419183                  0.0                 0.00                 0.06                  0.16               0.000000                0.056604                  530
           temporal_test                     A1c_tract_total_burden    svi_percentile       0.198890      0.131574    0.294005    0.286569    0.331540     0.419181                  0.0                 0.00                 0.06                  0.16               0.000000                0.056604                  530
           temporal_test                     A1c_tract_total_burden     svi_score_raw       0.198879      0.131582    0.294005    0.286572    0.331542     0.419183                  0.0                 0.00                 0.06                  0.16               0.000000                0.056604                  530
           temporal_test           A1d_population_normalized_burden    svi_percentile      -0.147254     -0.101103    0.211860    0.211793    0.264366     0.343124                  0.0                 0.00                 0.02                  0.08               0.000000                0.018868                  530
           temporal_test           A1d_population_normalized_burden     svi_score_raw      -0.147265     -0.101097    0.211860    0.211818    0.264384     0.343138                  0.0                 0.00                 0.02                  0.08               0.000000                0.018868                  530
           temporal_test             A1e_reporting_normalized_share    svi_percentile       0.299356      0.199288    0.513569    0.520814    0.602143     0.685287                  0.0                 0.04                 0.14                  0.33               0.040000                0.140000                  494
           temporal_test             A1e_reporting_normalized_share     svi_score_raw       0.299359      0.199289    0.513569    0.520926    0.602218     0.685337                  0.0                 0.04                 0.14                  0.33               0.040000                0.140000                  494
           temporal_test            A1f_reporting_excess_difference    svi_percentile       0.278035      0.178867    0.772929    0.797393    0.844616     0.885998                  0.0                 0.04                 0.08                  0.32               0.040000                0.080000                  494
           temporal_test            A1f_reporting_excess_difference     svi_score_raw       0.278039      0.178868    0.772929    0.797451    0.844654     0.886022                  0.0                 0.04                 0.08                  0.32               0.040000                0.080000                  494
           temporal_test                 A1f_reporting_excess_ratio    svi_percentile       0.299350      0.199268    0.513569    0.520814    0.602143     0.685287                  0.0                 0.04                 0.14                  0.33               0.040000                0.140000                  494
           temporal_test                 A1f_reporting_excess_ratio     svi_score_raw       0.299353      0.199270    0.513569    0.520926    0.602218     0.685337                  0.0                 0.04                 0.14                  0.33               0.040000                0.140000                  494
           temporal_test  A1g_data_defined_surge_window_mean_burden    svi_percentile            NaN           NaN         NaN         NaN         NaN          NaN                  NaN                  NaN                  NaN                   NaN                    NaN                     NaN                    0
           temporal_test  A1g_data_defined_surge_window_mean_burden     svi_score_raw            NaN           NaN         NaN         NaN         NaN          NaN                  NaN                  NaN                  NaN                   NaN                    NaN                     NaN                    0
           temporal_test A1g_data_defined_surge_window_total_burden    svi_percentile            NaN           NaN         NaN         NaN         NaN          NaN                  0.0                 0.04                 0.02                  0.03               0.037037                0.018868                  530
           temporal_test A1g_data_defined_surge_window_total_burden     svi_score_raw            NaN           NaN         NaN         NaN         NaN          NaN                  0.0                 0.04                 0.02                  0.03               0.037037                0.018868                  530
     temporal_validation           A1a_strict_tract_month_raw_count    svi_percentile       0.158606      0.108709    0.147259    0.201063    0.189843     0.193465                  0.0                 0.00                 0.00                  0.02               0.023585                0.066038                 4240
     temporal_validation           A1a_strict_tract_month_raw_count     svi_score_raw       0.158599      0.108704    0.147259    0.201063    0.189843     0.193465                  0.0                 0.00                 0.00                  0.02               0.023585                0.066038                 4240
     temporal_validation                      A1b_tract_mean_burden    svi_percentile       0.208687      0.135876    0.270682    0.281395    0.326038     0.413050                  0.0                 0.00                 0.02                  0.12               0.000000                0.037736                  530
     temporal_validation                      A1b_tract_mean_burden     svi_score_raw       0.208678      0.135885    0.270682    0.281399    0.326041     0.413052                  0.0                 0.00                 0.02                  0.12               0.000000                0.037736                  530
     temporal_validation                     A1c_tract_total_burden    svi_percentile       0.208687      0.135876    0.270682    0.281395    0.326038     0.413050                  0.0                 0.00                 0.02                  0.12               0.000000                0.037736                  530
     temporal_validation                     A1c_tract_total_burden     svi_score_raw       0.208678      0.135885    0.270682    0.281399    0.326041     0.413052                  0.0                 0.00                 0.02                  0.12               0.000000                0.037736                  530
     temporal_validation           A1d_population_normalized_burden    svi_percentile      -0.142253     -0.094527    0.235403    0.243676    0.287436     0.364940                  0.0                 0.00                 0.00                  0.05               0.000000                0.000000                  530
     temporal_validation           A1d_population_normalized_burden     svi_score_raw      -0.142261     -0.094521    0.235403    0.243704    0.287456     0.364954                  0.0                 0.00                 0.00                  0.05               0.000000                0.000000                  530
     temporal_validation             A1e_reporting_normalized_share    svi_percentile       0.226621      0.149540    0.562276    0.573512    0.633423     0.691225                  0.0                 0.04                 0.16                  0.28               0.040000                0.160000                  496
     temporal_validation             A1e_reporting_normalized_share     svi_score_raw       0.226629      0.149541    0.562276    0.573612    0.633490     0.691270                  0.0                 0.04                 0.16                  0.28               0.040000                0.160000                  496
     temporal_validation            A1f_reporting_excess_difference    svi_percentile       0.222324      0.143878    0.810574    0.841395    0.876344     0.905846                  0.0                 0.04                 0.10                  0.29               0.040000                0.100000                  496
     temporal_validation            A1f_reporting_excess_difference     svi_score_raw       0.222332      0.143879    0.810574    0.841425    0.876363     0.905859                  0.0                 0.04                 0.10                  0.29               0.040000                0.100000                  496
     temporal_validation                 A1f_reporting_excess_ratio    svi_percentile       0.226568      0.149495    0.562276    0.573512    0.633423     0.691225                  0.0                 0.04                 0.16                  0.28               0.040000                0.160000                  496
     temporal_validation                 A1f_reporting_excess_ratio     svi_score_raw       0.226576      0.149496    0.562276    0.573612    0.633490     0.691270                  0.0                 0.04                 0.16                  0.28               0.040000                0.160000                  496
     temporal_validation  A1g_data_defined_surge_window_mean_burden    svi_percentile       0.187510      0.123681    0.212885    0.232491    0.284186     0.359872                  0.0                 0.00                 0.00                  0.08               0.000000                0.000000                  530
     temporal_validation  A1g_data_defined_surge_window_mean_burden     svi_score_raw       0.187501      0.123682    0.212885    0.232491    0.284186     0.359872                  0.0                 0.00                 0.00                  0.08               0.000000                0.000000                  530
     temporal_validation A1g_data_defined_surge_window_total_burden    svi_percentile       0.187510      0.123681    0.212885    0.232491    0.284186     0.359872                  0.0                 0.00                 0.00                  0.08               0.000000                0.000000                  530
     temporal_validation A1g_data_defined_surge_window_total_burden     svi_score_raw       0.187501      0.123682    0.212885    0.232491    0.284186     0.359872                  0.0                 0.00                 0.00                  0.08               0.000000                0.000000                  530
```

## Target audit

```text
 split_scheme split_partition                                 variant_id                              target_column      status  non_missing  missing         min         max       mean
     temporal      validation                      A1b_tract_mean_burden                                 water_mean          ok        540.0      0.0    0.000000   24.875000   5.294907
     temporal      validation                     A1c_tract_total_burden                                water_total          ok        540.0      0.0    0.000000  199.000000  42.359259
     temporal      validation           A1d_population_normalized_burden                  water_per_1000_population          ok        530.0     10.0    0.000000   44.018184  12.308167
     temporal      validation             A1e_reporting_normalized_share                     water_share_of_all_311          ok        505.0     35.0    0.000000    0.173534   0.088161
     temporal      validation            A1f_reporting_excess_difference       water_excess_over_reporting_expected          ok        505.0     35.0 -242.420595   76.581908  -4.990549
     temporal      validation                 A1f_reporting_excess_ratio water_excess_ratio_over_reporting_expected          ok        505.0     35.0    0.000000    1.737221   0.882565
     temporal      validation  A1g_data_defined_surge_window_mean_burden                          hazard_water_mean          ok        540.0      0.0    0.000000   34.500000   6.062037
     temporal      validation A1g_data_defined_surge_window_total_burden                         hazard_water_total          ok        540.0      0.0    0.000000   69.000000  12.124074
     temporal      validation               denominator_audit_population                                 population       audit          NaN      NaN         NaN         NaN        NaN
     temporal      validation                   denominator_audit_all311                               all311_total       audit          NaN      NaN         NaN         NaN        NaN
     temporal      validation            data_defined_surge_window_audit                                        NaN       audit          NaN      NaN         NaN         NaN        NaN
     temporal            test                      A1b_tract_mean_burden                                 water_mean          ok        540.0      0.0    0.000000   25.333333   5.006173
     temporal            test                     A1c_tract_total_burden                                water_total          ok        540.0      0.0    0.000000  228.000000  45.055556
     temporal            test           A1d_population_normalized_burden                  water_per_1000_population          ok        530.0     10.0    0.000000   52.957132  13.248757
     temporal            test             A1e_reporting_normalized_share                     water_share_of_all_311          ok        504.0     36.0    0.000000    0.179078   0.086588
     temporal            test            A1f_reporting_excess_difference       water_excess_over_reporting_expected          ok        504.0     36.0 -187.249018   58.383210  -7.754441
     temporal            test                 A1f_reporting_excess_ratio water_excess_ratio_over_reporting_expected          ok        504.0     36.0    0.000000    1.792727   0.866823
     temporal            test  A1g_data_defined_surge_window_mean_burden                          hazard_water_mean all_missing          0.0    540.0         NaN         NaN        NaN
     temporal            test A1g_data_defined_surge_window_total_burden                         hazard_water_total          ok        540.0      0.0    0.000000    0.000000   0.000000
     temporal            test               denominator_audit_population                                 population       audit          NaN      NaN         NaN         NaN        NaN
     temporal            test                   denominator_audit_all311                               all311_total       audit          NaN      NaN         NaN         NaN        NaN
     temporal            test            data_defined_surge_window_audit                                        NaN       audit          NaN      NaN         NaN         NaN        NaN
spatial_block      validation                      A1b_tract_mean_burden                                 water_mean          ok        101.0      0.0    0.150943   17.981132   5.861573
spatial_block      validation                     A1c_tract_total_burden                                water_total          ok        101.0      0.0    8.000000  953.000000 310.663366
spatial_block      validation           A1d_population_normalized_burden                  water_per_1000_population          ok        100.0      1.0    6.836659  258.237766  89.871703
spatial_block      validation             A1e_reporting_normalized_share                     water_share_of_all_311          ok        101.0      0.0    0.029412    0.157039   0.098187
spatial_block      validation            A1f_reporting_excess_difference       water_excess_over_reporting_expected          ok        101.0      0.0 -119.382089  300.657362  14.615457
spatial_block      validation                 A1f_reporting_excess_ratio water_excess_ratio_over_reporting_expected          ok        101.0      0.0    0.311776    1.664667   1.040814
spatial_block      validation  A1g_data_defined_surge_window_mean_burden                          hazard_water_mean          ok        101.0      0.0    0.222222   22.222222   6.815182
spatial_block      validation A1g_data_defined_surge_window_total_burden                         hazard_water_total          ok        101.0      0.0    2.000000  200.000000  61.336634
spatial_block      validation               denominator_audit_population                                 population       audit          NaN      NaN         NaN         NaN        NaN
spatial_block      validation                   denominator_audit_all311                               all311_total       audit          NaN      NaN         NaN         NaN        NaN
spatial_block      validation            data_defined_surge_window_audit                                        NaN       audit          NaN      NaN         NaN         NaN        NaN
spatial_block            test                      A1b_tract_mean_burden                                 water_mean          ok         39.0      0.0    0.000000   20.320755   4.980164
spatial_block            test                     A1c_tract_total_burden                                water_total          ok         39.0      0.0    0.000000 1077.000000 263.948718
spatial_block            test           A1d_population_normalized_burden                  water_per_1000_population          ok         37.0      2.0    0.000000  155.689912  62.449716
spatial_block            test             A1e_reporting_normalized_share                     water_share_of_all_311          ok         37.0      2.0    0.022472    0.138645   0.097467
spatial_block            test            A1f_reporting_excess_difference       water_excess_over_reporting_expected          ok         37.0      2.0  -67.197447  317.215307  21.896795
spatial_block            test                 A1f_reporting_excess_ratio water_excess_ratio_over_reporting_expected          ok         37.0      2.0    0.238211    1.469690   1.033187
spatial_block            test  A1g_data_defined_surge_window_mean_burden                          hazard_water_mean          ok         39.0      0.0    0.000000   26.777778   6.045584
spatial_block            test A1g_data_defined_surge_window_total_burden                         hazard_water_total          ok         39.0      0.0    0.000000  241.000000  54.410256
spatial_block            test               denominator_audit_population                                 population       audit          NaN      NaN         NaN         NaN        NaN
spatial_block            test                   denominator_audit_all311                               all311_total       audit          NaN      NaN         NaN         NaN        NaN
spatial_block            test            data_defined_surge_window_audit                                        NaN       audit          NaN      NaN         NaN         NaN        NaN
```

## Static-score audit

```text
 source_column                              score_column                                 score_role  zones  zones_with_multiple_values  max_unique_values_within_zone                status examples
svi_percentile    svi_percentile__higher_more_vulnerable     primary_continuous_svi_score_candidate    540                           0                              1 ok_static_within_zone       []
 svi_score_raw     svi_score_raw__higher_more_vulnerable     primary_continuous_svi_score_candidate    540                           0                              1 ok_static_within_zone       []
      svi_rank svi_rank__rank_reversed_for_vulnerability             diagnostic_rank_reversed_score    540                           0                              1 ok_static_within_zone       []
     svi_class         svi_class__higher_more_vulnerable diagnostic_ordinal_class_score_not_primary    540                           0                              1 ok_static_within_zone       []
```

## Output artifacts

| Artifact | Path |
|---|---|
| `metrics` | `/home/tim/Documents/ville_ai/indices_BACKUP_before_clean_push/urban_graph_benchmark/outputs/mtl_311_water_v0/baselines/A1_svi_sensitivity/metrics.csv` |
| `model_metadata` | `/home/tim/Documents/ville_ai/indices_BACKUP_before_clean_push/urban_graph_benchmark/outputs/mtl_311_water_v0/baselines/A1_svi_sensitivity/model_metadata.json` |
| `baseline_report` | `/home/tim/Documents/ville_ai/indices_BACKUP_before_clean_push/urban_graph_benchmark/outputs/mtl_311_water_v0/baselines/A1_svi_sensitivity/baseline_report.md` |
| `sensitivity_table` | `/home/tim/Documents/ville_ai/indices_BACKUP_before_clean_push/urban_graph_benchmark/outputs/mtl_311_water_v0/baselines/A1_svi_sensitivity/sensitivity_table.csv` |
| `tract_level_targets` | `/home/tim/Documents/ville_ai/indices_BACKUP_before_clean_push/urban_graph_benchmark/outputs/mtl_311_water_v0/baselines/A1_svi_sensitivity/tract_level_targets.csv` |
| `top_decile_overlap_by_variant` | `/home/tim/Documents/ville_ai/indices_BACKUP_before_clean_push/urban_graph_benchmark/outputs/mtl_311_water_v0/baselines/A1_svi_sensitivity/top_decile_overlap_by_variant.csv` |
| `target_audit` | `/home/tim/Documents/ville_ai/indices_BACKUP_before_clean_push/urban_graph_benchmark/outputs/mtl_311_water_v0/baselines/A1_svi_sensitivity/target_audit.csv` |
| `svi_score_audit` | `/home/tim/Documents/ville_ai/indices_BACKUP_before_clean_push/urban_graph_benchmark/outputs/mtl_311_water_v0/baselines/A1_svi_sensitivity/svi_score_audit.csv` |
| `svi_static_score_audit` | `/home/tim/Documents/ville_ai/indices_BACKUP_before_clean_push/urban_graph_benchmark/outputs/mtl_311_water_v0/baselines/A1_svi_sensitivity/svi_static_score_audit.csv` |

## Interpretation guide

- If A1a is weak but A1b/A1c improve, SVI may contain static spatial burden signal while remaining weak for monthly operational prioritization.
- If A1d improves, raw call volume may be partly dominated by population size rather than relative social burden.
- If A1e/A1f improve, raw water/drainage call volume may be partly dominated by general 311 reporting intensity.
- If all variants remain weak, then raw SVI is weakly aligned with this specific Montréal water/drainage 311 target even under more SVI-native target definitions.
- A1 sensitivity results should not be read as invalidating SVI. The target is reported municipal service burden, not objective flood occurrence or disaster impact.
