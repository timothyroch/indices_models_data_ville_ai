# SoVI direct external-validation against Québec civil-security event burden

## Scope

This benchmark is a direct external-validation benchmark. It asks whether the static SoVI-like CD score ranks Québec census divisions in a way that aligns with observed civil-security event burden.

It does **not** run supervised ML, graph modeling, calibration, or target-based tuning. The SoVI score is used directly as a ranking signal.

Selected SoVI score column:

```text
score_normalized_0_1
```

The selected SoVI score was evaluated as higher = more vulnerable.

## Primary target

The primary target is:

```text
event_count_2021_2025_all
```

Interpretation:

```text
2021–2025 = descriptive / near-snapshot validation
2022–2025 = cleaner forward-looking validation from mostly-2021 SoVI variables
```

The primary result should be read as external alignment evidence, not causal proof and not proof that SoVI predicts future operational disruption.

## Primary result: SoVI vs 2021–2025 all civil-security events

- Spearman: 0.253
- Kendall: 0.159
- NDCG@10: 0.284
- NDCG@25: 0.429
- Top-10 overlap: 0.000
- Top-25 overlap: 0.360

Headline metrics are Spearman, Kendall, NDCG@10, NDCG@25, top-10 overlap, and top-25 overlap.

NDCG@50 and top-50 overlap are still reported in the CSV files for consistency with the SVI benchmark family, but they should not be headlined strongly because there are only 98 CDs. Top-50 is roughly half the province, so it is a broad/global diagnostic rather than a sharp prioritization metric.

## Target summary

| target_id                | target_label                             | target_family      |   total_events |   nonzero_cds |   nonzero_rate |   median |     max |
|:-------------------------|:-----------------------------------------|:-------------------|---------------:|--------------:|---------------:|---------:|--------:|
| B1a                      | 2021–2025 all events                     | main_cumulative    |           3426 |            98 |          1.000 |   27.000 | 150.000 |
| B1b                      | 2022–2025 all events                     | forward_cumulative |           3076 |            98 |          1.000 |   24.000 | 145.000 |
| B1c                      | 2021–2025 precise or very precise events | precision_filter   |           1670 |            93 |          0.949 |   11.000 | 109.000 |
| B1d                      | 2021–2025 very precise events            | precision_filter   |           1304 |            92 |          0.939 |    6.500 |  98.000 |
| B1e                      | 2021–2025 flood/water events             | hazard_group       |            852 |            97 |          0.990 |    6.000 |  52.000 |
| B1f_land_ground          | 2021–2025 land/ground events             | hazard_group       |           1183 |            89 |          0.908 |    5.500 |  98.000 |
| B1f_weather_climate      | 2021–2025 weather/climate events         | hazard_group       |            375 |            75 |          0.765 |    2.000 |  23.000 |
| B1f_infrastructure       | 2021–2025 infrastructure events          | hazard_group       |            726 |            89 |          0.908 |    5.000 |  33.000 |
| B1f_wildfire             | 2021–2025 wildfire events                | hazard_group       |            109 |            31 |          0.316 |    0.000 |  23.000 |
| B1g_moderate_or_worse    | 2021–2025 moderate-or-worse events       | severity_threshold |            840 |            89 |          0.908 |    5.000 |  44.000 |
| B1g_important_or_extreme | 2021–2025 important-or-extreme events    | severity_threshold |            101 |            44 |          0.449 |    0.000 |  10.000 |

## SoVI target-sensitivity results

| target_id                | target_label                             | target_family      |   spearman |   kendall |   ndcg_at_10 |   ndcg_at_25 |   top10_overlap_rate |   top25_overlap_rate |
|:-------------------------|:-----------------------------------------|:-------------------|-----------:|----------:|-------------:|-------------:|---------------------:|---------------------:|
| B1a                      | 2021–2025 all events                     | main_cumulative    |      0.253 |     0.159 |        0.284 |        0.429 |                0.000 |                0.360 |
| B1b                      | 2022–2025 all events                     | forward_cumulative |      0.201 |     0.126 |        0.256 |        0.398 |                0.000 |                0.320 |
| B1c                      | 2021–2025 precise or very precise events | precision_filter   |      0.191 |     0.118 |        0.193 |        0.322 |                0.100 |                0.280 |
| B1d                      | 2021–2025 very precise events            | precision_filter   |      0.164 |     0.106 |        0.140 |        0.257 |                0.100 |                0.240 |
| B1e                      | 2021–2025 flood/water events             | hazard_group       |     -0.042 |    -0.023 |        0.129 |        0.316 |                0.000 |                0.280 |
| B1f_infrastructure       | 2021–2025 infrastructure events          | hazard_group       |      0.206 |     0.150 |        0.306 |        0.412 |                0.000 |                0.280 |
| B1f_land_ground          | 2021–2025 land/ground events             | hazard_group       |      0.085 |     0.055 |        0.118 |        0.224 |                0.100 |                0.240 |
| B1f_weather_climate      | 2021–2025 weather/climate events         | hazard_group       |      0.377 |     0.277 |        0.350 |        0.482 |                0.100 |                0.400 |
| B1f_wildfire             | 2021–2025 wildfire events                | hazard_group       |      0.082 |     0.067 |        0.382 |        0.448 |                0.200 |                0.280 |
| B1g_important_or_extreme | 2021–2025 important-or-extreme events    | severity_threshold |      0.236 |     0.177 |        0.277 |        0.389 |                0.100 |                0.320 |
| B1g_moderate_or_worse    | 2021–2025 moderate-or-worse events       | severity_threshold |      0.258 |     0.176 |        0.307 |        0.411 |                0.000 |                0.400 |

## Optional B0 sanity/null controls

These controls are included only as small sanity checks when the required columns are already present in the aligned frame. They are not a supervised exposure model and should not become the center of the benchmark.

| scorer_id         |   spearman |   kendall |   ndcg_at_10 |   ndcg_at_25 |   top10_overlap_rate |   top25_overlap_rate |
|:------------------|-----------:|----------:|-------------:|-------------:|---------------------:|---------------------:|
| B0_random_seed_42 |     -0.225 |    -0.145 |        0.194 |        0.241 |                0.000 |                0.040 |

## Recommended interpretation

Use this benchmark to answer:

```text
Do Québec CDs with higher SoVI-like vulnerability scores tend to rank higher in observed civil-security event burden, and is this relationship stable across target definitions, hazard groups, precision filters, and severity thresholds?
```

A strong result would show stable positive rank association across the primary target, the 2022–2025 forward-looking target, localization-precision targets, and at least some hazard/severity targets.

A weak or unstable result would not invalidate SoVI as a social vulnerability index. It would mean that this particular event-burden layer is only partially aligned with the social-vulnerability construct, or that observed event counts are driven heavily by exposure, hazard geography, reporting practices, administrative definitions, and event-detection processes.

## Caveats

- Civil-security event burden is not the same construct as social vulnerability.
- CD-level event counts mix hazard exposure, reporting intensity, administrative practice, and vulnerability.
- The current benchmark is ranking-based because SoVI is an index, not a calibrated event-count predictor.
- No SoVI score was tuned using the civil-security target.
- Monthly CD targets are intentionally not the main target because earlier density checks showed sparse CD-month cells.
