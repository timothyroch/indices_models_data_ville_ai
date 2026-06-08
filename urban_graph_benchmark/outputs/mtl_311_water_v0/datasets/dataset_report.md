# Dataset Report — Montréal 311 Water/Drainage v0

Generated at: `2026-06-08T14:04:33.787893+00:00`

Benchmark ID: `mtl_311_water_v0`

Validation status: `pass`

## Dataset scope

Dataset v0 uses the derived grid25m-month 311 table and assigns grid-cell centroids to census tracts using lon/lat centroids in EPSG:4326 reprojected to the tract CRS before centroid-in-polygon. The in-scope tract set is the set of tracts receiving at least one assigned 311 grid cell. This is a v0 empirical 311 service-territory proxy, not a formal service-boundary definition.

Population-weighted centroids and road-network accessibility features are not computed in v0. Null population-weighted centroid columns are reserved for v1.

## Panel summary

| Metric | Value |
|---|---:|
| `n_zones` | `540` |
| `n_months` | `53` |
| `expected_rows` | `28620` |
| `actual_rows` | `28620` |
| `zero_filled_tract_month_rows` | `1549` |
| `period_month_min` | `2022-01` |
| `period_month_max` | `2026-05` |

## Spatial assignment summary

| Metric | Value |
|---|---:|
| `total_unique_grid_units` | `42752` |
| `assigned_unique_grid_units` | `42731` |
| `unassigned_unique_grid_units` | `21` |
| `assignment_success_rate_unique_grid_units` | `0.9995087949101796` |
| `total_unique_coordinate_rows` | `83829` |
| `assigned_unique_coordinate_rows` | `83805` |
| `unassigned_unique_coordinate_rows` | `24` |
| `assignment_success_rate_coordinate_rows` | `0.9997137028951795` |
| `coordinate_source` | `lon_lat_centroid_epsg4326_to_tract_crs` |
| `coordinate_source_crs` | `EPSG:4326` |
| `coordinate_target_crs` | `EPSG:3347` |
| `spatial_join_method` | `grid_coordinate_row_centroid_in_polygon_using_lon_lat_epsg4326_reprojected_to_tract_crs` |
| `units_with_multiple_coordinate_rows` | `17655` |
| `units_with_multiple_assigned_tracts` | `538` |
| `within_duplicate_assignments` | `0` |
| `boundary_fallback_duplicate_assignments` | `0` |
| `study_area_rule_v0` | `tracts_with_at_least_one_assigned_311_grid25m_unit` |
| `n_assigned_zone_ids` | `540` |
| `n_static_tracts_in_scope` | `540` |
| `note` | `This is a v0 empirical service-territory proxy. It does not yet use a formal Ville de Montréal/agglomeration service boundary.` |

## Aggregation summary

| Item | Value |
|---|---|
| `assigned_grid_month_rows_used` | `760591` |
| `unassigned_grid_month_rows_excluded` | `30` |
| `sum_columns` | `['requests_total', 'complaints_total', 'citizen_requests_total', 'comments_total', 'urgent_total', 'finished_total', 'other_requests', 'road_mobility_requests', 'snow_winter_requests', 'tree_canopy_requests', 'waste_cleanliness_requests', 'water_drainage_requests']` |
| `delay_columns_aggregated` | `['avg_resolution_delay_hours', 'median_resolution_delay_hours_grid_weighted_mean_not_true_median']` |
| `share_columns_recomputed` | `['share_complaints_total', 'share_urgent_total', 'share_water_drainage_requests', 'share_road_mobility_requests', 'share_tree_canopy_requests', 'share_snow_winter_requests', 'share_waste_cleanliness_requests']` |
| `omitted_grid_columns` | `['unique_activity_count', 'unique_responsible_units']` |

## SVI join summary

| Metric | Value |
|---|---:|
| `panel_rows_before_svi_join` | `28620` |
| `panel_rows_after_svi_join` | `28620` |
| `static_tracts_in_scope` | `540` |
| `svi_rows` | `1480` |
| `matched_static_tracts` | `540` |
| `missing_svi_rows` | `0` |
| `svi_join_success_rate` | `1.0` |

## Validation checks

| Check | Passed | Severity | Details |
|---|:---:|---|---|
| `one_row_per_zone_month` | `True` | `error` | `{"duplicate_rows": 0}` |
| `all_expected_tracts_represented_in_every_month` | `True` | `error` | `{"expected_rows": 28620, "actual_rows": 28620}` |
| `zero_filled_missing_target_rows` | `True` | `error` | `{"missing_water_drainage_count": 0}` |
| `water_drainage_count_nonnegative` | `True` | `error` | `{"negative_rows": 0}` |
| `total_311_count_non_water_drainage_nonnegative` | `True` | `error` | `{"negative_rows": 0}` |
| `spatial_assignment_success_rate_reported` | `True` | `error` | `{"total_unique_grid_units": 42752, "assigned_unique_grid_units": 42731, "unassigned_unique_grid_units": 21, "assignment_success_rate_unique_grid_units": 0.9995087949101796, "total_unique_coordinate_rows": 83829, "assigned_unique_coordinate_rows": 83805, "unassigned_unique_coordinate_rows": 24, "assi` |
| `unassigned_grid_cells_reported` | `True` | `warning` | `{"unassigned_unique_grid_units": 21}` |
| `svi_join_success_rate_reported` | `True` | `error` | `{"panel_rows_before_svi_join": 28620, "panel_rows_after_svi_join": 28620, "static_tracts_in_scope": 540, "svi_rows": 1480, "matched_static_tracts": 540, "missing_svi_rows": 0, "svi_join_success_rate": 1.0, "svi_columns_joined": ["svi_input_year", "svi_input_source", "svi_available_variable_count", "` |
| `svi_join_complete_for_in_scope_tracts` | `True` | `warning` | `{"panel_rows_before_svi_join": 28620, "panel_rows_after_svi_join": 28620, "static_tracts_in_scope": 540, "svi_rows": 1480, "matched_static_tracts": 540, "missing_svi_rows": 0, "svi_join_success_rate": 1.0, "svi_columns_joined": ["svi_input_year", "svi_input_source", "svi_available_variable_count", "` |
| `no_sovi_columns_in_track_a` | `True` | `error` | `{"sovi_like_columns": []}` |
| `no_missing_zone_id` | `True` | `error` | `{"missing_zone_id_rows": 0}` |
| `target_table_row_count_matches_panel` | `True` | `error` | `{"target_rows": 28620, "panel_rows": 28620}` |

## Output artifacts

| Artifact | Path |
|---|---|
| `tract_month_panel` | `/home/tim/Documents/ville_ai/indices_BACKUP_before_clean_push/urban_graph_benchmark/outputs/mtl_311_water_v0/datasets/tract_month_panel.parquet` |
| `tract_static_features` | `/home/tim/Documents/ville_ai/indices_BACKUP_before_clean_push/urban_graph_benchmark/outputs/mtl_311_water_v0/datasets/tract_static_features.parquet` |
| `target_water_drainage` | `/home/tim/Documents/ville_ai/indices_BACKUP_before_clean_push/urban_graph_benchmark/outputs/mtl_311_water_v0/datasets/target_water_drainage.parquet` |
| `dataset_validation` | `/home/tim/Documents/ville_ai/indices_BACKUP_before_clean_push/urban_graph_benchmark/outputs/mtl_311_water_v0/datasets/dataset_validation.json` |
| `dataset_report` | `/home/tim/Documents/ville_ai/indices_BACKUP_before_clean_push/urban_graph_benchmark/outputs/mtl_311_water_v0/datasets/dataset_report.md` |
| `spatial_join_audit` | `/home/tim/Documents/ville_ai/indices_BACKUP_before_clean_push/urban_graph_benchmark/outputs/mtl_311_water_v0/datasets/spatial_join_audit.csv` |
| `missingness_report` | `/home/tim/Documents/ville_ai/indices_BACKUP_before_clean_push/urban_graph_benchmark/outputs/mtl_311_water_v0/datasets/missingness_report.csv` |
| `feature_dictionary` | `/home/tim/Documents/ville_ai/indices_BACKUP_before_clean_push/urban_graph_benchmark/outputs/mtl_311_water_v0/datasets/feature_dictionary.csv` |
| `provenance` | `/home/tim/Documents/ville_ai/indices_BACKUP_before_clean_push/urban_graph_benchmark/outputs/mtl_311_water_v0/datasets/provenance.json` |

## Notes and limitations

- The target is a reported municipal 311 disruption signal, not objective flood occurrence.
- `total_311_count_all` contains the water/drainage target and is retrospective-only.
- `total_311_count_non_water_drainage` is the preferred same-month reporting-control proxy for retrospective models.
- Official magnitude classes are not generated here; they must be split-specific in the modeling pipeline.
- Unique activity/responsible-unit counts are omitted from v0 because exact tract-month recomputation is impossible from grid-level aggregates.
- Road-network travel distances, OSM routing, population-weighted centroids, and accessibility features are planned for later modules.
