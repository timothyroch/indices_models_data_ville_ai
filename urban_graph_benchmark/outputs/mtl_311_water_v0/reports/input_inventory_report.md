# Input Inventory Report — Montréal 311 Water/Drainage v0

Generated at: `2026-06-05T16:15:36.035494+00:00`

Benchmark ID: `mtl_311_water_v0`

Config path: `/home/tim/Documents/ville_ai/indices_BACKUP_before_clean_push/urban_graph_benchmark/configs/mtl_311_water_v0.yaml`

Config hash: `e01703b33032b639889a32eaea07601539611c4b9fc8a662bca31b427ee6ce1b`

Repository root: `/home/tim/Documents/ville_ai/indices_BACKUP_before_clean_push`

## Summary

| Input | Candidates | Existing | Best candidate |
|---|---:|---:|---|
| `montreal_311` | 4 | 4 | `/home/tim/Documents/ville_ai/indices_BACKUP_before_clean_push/transformation/requetes311.csv` |
| `census_tract_geometry` | 6 | 6 | `/home/tim/Documents/ville_ai/indices_BACKUP_before_clean_push/data/spatial_frame_population_2021/output/clean_quebec_census_tract_spatial_frame_with_population_2021.gpkg` |
| `svi` | 19 | 19 | `/home/tim/Documents/ville_ai/indices_BACKUP_before_clean_push/outputs/svi_quebec_2021_partial12_run/svi_quebec_2021_partial12_map.csv` |

## Open decisions

| Decision | Current value |
|---|---|
| `study_area` | `DECISION_NEEDED` |
| `311_source_type` | `DECISION_NEEDED` |
| `spatial_assignment_method` | `DECISION_NEEDED` |
| `target_category_selection_method` | `DECISION_NEEDED` |
| `magnitude_threshold_strategy` | `DECISION_NEEDED` |

## 311 source-type hints

| Inferred source type | Candidate count |
|---|---:|
| `pre_aggregated` | 1 |
| `grid25m_month` | 2 |

## Tract geometry ↔ SVI join feasibility

Status: `computed`

| Left column | Right column | Overlap | Left ratio | Right ratio |
|---|---|---:|---:|---:|
| `unit_id` | `unit_id` | 1480 | 1.000 | 1.000 |

## Montréal 311 candidates

| # | Exists | Status | Kind | Path | Notes |
|---:|:---:|---|---|---|---|
| 1 | True | inspected | table | `/home/tim/Documents/ville_ai/indices_BACKUP_before_clean_push/transformation/requetes311.csv` | source_type=pre_aggregated<br>confidence=low<br>id_cols=1 |
| 2 | True | inspected | table | `/home/tim/Documents/ville_ai/indices_BACKUP_before_clean_push/transformation/output/ville_ia_311_features_grid25m_monthly.parquet` | source_type=grid25m_month<br>confidence=high<br>grid25m_values=yes<br>id_cols=5 |
| 3 | True | inspected | table | `/home/tim/Documents/ville_ai/indices_BACKUP_before_clean_push/transformation/output/ville_ia_311_features_grid25m_monthly.csv` | source_type=grid25m_month<br>confidence=high<br>grid25m_values=yes<br>id_cols=5 |
| 4 | True | read_failed | spatial | `/home/tim/Documents/ville_ai/indices_BACKUP_before_clean_push/transformation/output/ville_ia_311_features_grid25m_monthly.geojson` |  |

## Census tract geometry candidates

| # | Exists | Status | Kind | Path | Notes |
|---:|:---:|---|---|---|---|
| 1 | True | inspected | spatial | `/home/tim/Documents/ville_ai/indices_BACKUP_before_clean_push/data/spatial_frame_population_2021/output/clean_quebec_census_tract_spatial_frame_with_population_2021.gpkg` | id_cols=3<br>crs=EPSG:3347 |
| 2 | True | inspected | spatial | `/home/tim/Documents/ville_ai/indices_BACKUP_before_clean_push/data/spatial_frame_population_2021/output/clean_quebec_census_tract_spatial_frame_with_population_2021.geojson` | id_cols=3<br>crs=EPSG:3347 |
| 3 | True | inspected | table | `/home/tim/Documents/ville_ai/indices_BACKUP_before_clean_push/data/spatial_frame_population_2021/output/clean_quebec_census_tract_spatial_frame_with_population_2021.parquet` | id_cols=3 |
| 4 | True | inspected | spatial | `/home/tim/Documents/ville_ai/indices_BACKUP_before_clean_push/data/2021-census-boundaries-file/output/clean_quebec_census_tracts_2021.gpkg` | id_cols=3<br>crs=EPSG:3347 |
| 5 | True | inspected | spatial | `/home/tim/Documents/ville_ai/indices_BACKUP_before_clean_push/data/2021-census-boundaries-file/output/clean_quebec_census_tracts_2021.geojson` | id_cols=3<br>crs=EPSG:3347 |
| 6 | True | inspected | table | `/home/tim/Documents/ville_ai/indices_BACKUP_before_clean_push/data/2021-census-boundaries-file/output/clean_quebec_census_tracts_2021.parquet` | id_cols=3 |

## SVI candidates

| # | Exists | Status | Kind | Path | Notes |
|---:|:---:|---|---|---|---|
| 1 | True | inspected | table | `/home/tim/Documents/ville_ai/indices_BACKUP_before_clean_push/outputs/svi_quebec_2021_partial12_run/svi_quebec_2021_partial12_map.csv` | id_cols=6<br>score_cols=5 |
| 2 | True | read_failed | spatial | `/home/tim/Documents/ville_ai/indices_BACKUP_before_clean_push/outputs/svi_quebec_2021_partial12_run/svi_quebec_2021_partial12_map.geojson` |  |
| 3 | True | read_failed | spatial | `/home/tim/Documents/ville_ai/indices_BACKUP_before_clean_push/outputs/svi_quebec_2021_partial12_run/svi_quebec_2021_partial12_map.gpkg` |  |
| 4 | True | read_failed | spatial | `/home/tim/Documents/ville_ai/indices_BACKUP_before_clean_push/outputs/svi_quebec_2021_partial12_run/svi_quebec_2021_partial12_map_web.geojson` |  |
| 5 | True | inspected | table | `/home/tim/Documents/ville_ai/indices_BACKUP_before_clean_push/data/svi_2021/output/clean_quebec_census_tract_svi_input_2021.parquet` | id_cols=5 |
| 6 | True | inspected | table | `/home/tim/Documents/ville_ai/indices_BACKUP_before_clean_push/data/svi_2021/output/clean_quebec_census_tract_svi_input_2021.csv` | id_cols=5 |
| 7 | True | read_failed | spatial | `/home/tim/Documents/ville_ai/indices_BACKUP_before_clean_push/data/svi_2021/output/clean_quebec_census_tract_svi_input_2021.geojson` |  |
| 8 | True | read_failed | spatial | `/home/tim/Documents/ville_ai/indices_BACKUP_before_clean_push/data/svi_2021/output/clean_quebec_census_tract_svi_input_2021.gpkg` |  |
| 9 | True | inspected | table | `/home/tim/Documents/ville_ai/indices_BACKUP_before_clean_push/data/svi_2021/output/svi_input_candidate_column_diagnostics_2021.csv` | id_cols=2<br>score_cols=1 |
| 10 | True | inspected | table | `/home/tim/Documents/ville_ai/indices_BACKUP_before_clean_push/data/svi_2021/output/svi_input_join_diagnostics_2021.csv` |  |
| 11 | True | inspected | table | `/home/tim/Documents/ville_ai/indices_BACKUP_before_clean_push/data/svi_2021/output/clean_quebec_census_tract_svi_input_missingness_report_2021.csv` |  |
| 12 | True | inspected | table | `/home/tim/Documents/ville_ai/indices_BACKUP_before_clean_push/data/svi_2021/output/svi_input_source_file_inventory_2021.csv` | id_cols=2 |
| 13 | True | inspected | table | `/home/tim/Documents/ville_ai/indices_BACKUP_before_clean_push/data/svi_2021/output/clean_quebec_census_tract_svi_input_variable_metadata_2021.csv` |  |
| 14 | True | inspected | table | `/home/tim/Documents/ville_ai/indices_BACKUP_before_clean_push/data/svi_2021/output/svi_input_variable_availability_2021.csv` | id_cols=4<br>score_cols=1 |
| 15 | True | inspected | table | `/home/tim/Documents/ville_ai/indices_BACKUP_before_clean_push/data/svi_2021/output/svi_input_availability_summary_2021.csv` |  |
| 16 | True | inspected | table | `/home/tim/Documents/ville_ai/indices_BACKUP_before_clean_push/data/svi_2021/output/clean_quebec_census_tract_svi_input_join_report_2021.csv` |  |
| 17 | True | inspected | table | `/home/tim/Documents/ville_ai/indices_BACKUP_before_clean_push/outputs/svi_quebec_2021_partial12_run/svi_quebec_2021_partial12_map_legend.csv` | score_cols=1 |
| 18 | True | read_failed | spatial | `/home/tim/Documents/ville_ai/indices_BACKUP_before_clean_push/outputs/svi_quebec_2021_partial12_run/svi_quebec_2021_partial12_map_native.gpkg` |  |
| 19 | True | inspected | table | `/home/tim/Documents/ville_ai/indices_BACKUP_before_clean_push/outputs/svi_quebec_2021_partial12_run/svi_quebec_2021_partial12_map_audit.csv` | id_cols=6<br>score_cols=5 |

## Next steps

1. Confirm the canonical 311 file path and whether it is point-level, grid25m-month, or already aggregated.
2. Confirm the exact Montréal study area based on valid 311 coverage.
3. Confirm the tract geometry file and SVI output file.
4. Review candidate 311 categories before building the target.
5. Revise the config values currently marked `DECISION_NEEDED`.
