# Split Report — Montréal 311 Water/Drainage v0

Generated at: `2026-06-08T15:30:24.699575+00:00`

Benchmark ID: `mtl_311_water_v0`

Panel path: `/home/tim/Documents/ville_ai/indices_BACKUP_before_clean_push/urban_graph_benchmark/outputs/mtl_311_water_v0/datasets/tract_month_panel.parquet`

Validation status: `pass`

## Summary

| Metric | Value |
|---|---:|
| rows | 28620 |
| zones | 540 |
| months | 53 |
| period min | 2022-01 |
| period max | 2026-05 |

## Temporal split — primary scientific split

| Split | Period | Rows | Months |
|---|---|---:|---:|
| `train` | 2022-01 to 2024-12 | 19440 | 36 |
| `validation` | 2025-01 to 2025-08 | 4320 | 8 |
| `test` | 2025-09 to 2026-05 | 4860 | 9 |

## Random debug split

Random split is for implementation checks only and is not main scientific evidence.

| Split | Rows |
|---|---:|
| `train` | 20034 |
| `validation` | 4293 |
| `test` | 4293 |

## Spatial block split

Available: `True`

This is a preliminary spatial split based on tract centroid quantile-grid blocks. Graph-specific leakage control remains a later modeling responsibility.

| Split | Rows |
|---|---:|
| `train` | 21200 |
| `validation` | 5353 |
| `test` | 2067 |

## Train-only magnitude thresholds

### `temporal`

| Threshold | Value |
|---|---:|
| `class_0_rule` | `y == 0` |
| `class_1_max` | `3.0` |
| `class_2_max` | `5.0` |
| `class_3_max` | `8.0` |
| `class_4_rule` | `>8.0` |

### `random_debug`

| Threshold | Value |
|---|---:|
| `class_0_rule` | `y == 0` |
| `class_1_max` | `3.0` |
| `class_2_max` | `5.0` |
| `class_3_max` | `8.0` |
| `class_4_rule` | `>8.0` |

### `spatial_block`

| Threshold | Value |
|---|---:|
| `class_0_rule` | `y == 0` |
| `class_1_max` | `3.0` |
| `class_2_max` | `5.0` |
| `class_3_max` | `8.0` |
| `class_4_rule` | `>8.0` |

## Validation checks

| Check | Passed | Severity | Details |
|---|:---:|---|---|
| `one_row_per_zone_month_in_split_assignments` | `True` | `error` | `{"duplicate_rows": 0}` |
| `temporal_split_covers_every_row` | `True` | `error` | `{"missing_rows": 0, "values": ["test", "train", "validation"]}` |
| `temporal_split_has_train_validation_test` | `True` | `error` | `{"counts": {"train": 19440, "test": 4860, "validation": 4320}}` |
| `random_debug_split_covers_every_row` | `True` | `error` | `{"missing_rows": 0, "values": ["test", "train", "validation"]}` |
| `random_debug_split_has_train_validation_test` | `True` | `error` | `{"counts": {"train": 20034, "validation": 4293, "test": 4293}}` |
| `spatial_block_split_status_documented` | `True` | `info` | `{"available": true, "missing_rows": 0}` |
| `spatial_block_split_has_train_validation_test` | `True` | `warning` | `{"counts": {"train": 21200, "validation": 5353, "test": 2067}}` |
| `spatial_block_split_constant_within_zone` | `True` | `error` | `{"bad_zone_count": 0, "examples": []}` |
| `temporal_split_month_counts_match_plan` | `True` | `error` | `{"month_counts": {"test": 9, "train": 36, "validation": 8}}` |
| `magnitude_class_temporal_valid_0_to_4` | `True` | `error` | `{"missing_rows": 0, "values": [0, 1, 2, 3, 4]}` |
| `magnitude_class_random_debug_valid_0_to_4` | `True` | `error` | `{"missing_rows": 0, "values": [0, 1, 2, 3, 4]}` |
| `magnitude_class_spatial_block_valid_0_to_4` | `True` | `error` | `{"missing_rows": 0, "values": [0, 1, 2, 3, 4]}` |

## Output artifacts

| Artifact | Path |
|---|---|
| `split_assignments` | `/home/tim/Documents/ville_ai/indices_BACKUP_before_clean_push/urban_graph_benchmark/outputs/mtl_311_water_v0/datasets/splits/split_assignments.parquet` |
| `split_metadata` | `/home/tim/Documents/ville_ai/indices_BACKUP_before_clean_push/urban_graph_benchmark/outputs/mtl_311_water_v0/datasets/splits/split_metadata.json` |
| `split_report` | `/home/tim/Documents/ville_ai/indices_BACKUP_before_clean_push/urban_graph_benchmark/outputs/mtl_311_water_v0/datasets/splits/split_report.md` |
| `split_validation` | `/home/tim/Documents/ville_ai/indices_BACKUP_before_clean_push/urban_graph_benchmark/outputs/mtl_311_water_v0/datasets/splits/split_validation.json` |
| `target_thresholds_temporal` | `/home/tim/Documents/ville_ai/indices_BACKUP_before_clean_push/urban_graph_benchmark/outputs/mtl_311_water_v0/datasets/splits/target_thresholds_temporal.json` |
| `target_thresholds_random_debug` | `/home/tim/Documents/ville_ai/indices_BACKUP_before_clean_push/urban_graph_benchmark/outputs/mtl_311_water_v0/datasets/splits/target_thresholds_random_debug.json` |
| `target_thresholds_spatial_block` | `/home/tim/Documents/ville_ai/indices_BACKUP_before_clean_push/urban_graph_benchmark/outputs/mtl_311_water_v0/datasets/splits/target_thresholds_spatial_block.json` |

## Leakage notes

- Magnitude thresholds are fitted on training rows only for each split scheme.
- Magnitude class 0 is strictly `water_drainage_count == 0`.
- Positive magnitude classes 1–4 use quantiles of positive training counts only.
- Random split is debugging-only and must not be used as primary scientific evidence.
- Same-month target-derived columns must be excluded from model features.
- Spatial block split is preliminary; graph evaluation must later document inductive vs transductive handling.
