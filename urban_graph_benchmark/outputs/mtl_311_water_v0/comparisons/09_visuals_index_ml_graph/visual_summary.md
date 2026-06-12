# Benchmark Visuals

Generated at: `2026-06-12T17:16:34.423660+00:00`

## Purpose

This folder contains post-hoc visuals for the index → tabular ML → graph/neural benchmark layer. The script reads existing benchmark outputs and graph artifacts; it does not retrain or reselect models.

## Inputs

- Comparison directory: `urban_graph_benchmark/outputs/mtl_311_water_v0/comparisons/08_index_ml_graph_benchmark`
- G1.5 sweep directory: `urban_graph_benchmark/outputs/mtl_311_water_v0/baselines/G1_5_validation_sweep_spatial_ndcg`
- G1 pilot directory: `urban_graph_benchmark/outputs/mtl_311_water_v0/baselines/G1_spatiotemporal_gnn_spatial_core_ndcg_monitor`
- Graph artifact directory: `urban_graph_benchmark/outputs/mtl_311_water_v0/graphs/G1_tract_month_graph`

## Visual artifacts

| Artifact | Path |
|---|---|
| `fig_01_benchmark_metric_panels` | `urban_graph_benchmark/outputs/mtl_311_water_v0/comparisons/09_visuals_index_ml_graph/figures/01_benchmark_metric_panels.png` |
| `fig_02_index_vs_learned_ranking_gap` | `urban_graph_benchmark/outputs/mtl_311_water_v0/comparisons/09_visuals_index_ml_graph/figures/02_index_vs_learned_ranking_gap.png` |
| `fig_03_family_margin_panels` | `urban_graph_benchmark/outputs/mtl_311_water_v0/comparisons/09_visuals_index_ml_graph/figures/03_family_margin_panels.png` |
| `fig_04_g1_family_comparison` | `urban_graph_benchmark/outputs/mtl_311_water_v0/comparisons/09_visuals_index_ml_graph/figures/04_g1_family_comparison.png` |
| `fig_05_g1_validation_sweep_heatmap` | `urban_graph_benchmark/outputs/mtl_311_water_v0/comparisons/09_visuals_index_ml_graph/figures/05_g1_validation_sweep_heatmap.png` |
| `fig_06_benchmark_pipeline_schema` | `urban_graph_benchmark/outputs/mtl_311_water_v0/comparisons/09_visuals_index_ml_graph/figures/06_benchmark_pipeline_schema.png` |
| `fig_07_tract_month_graph_sample` | `urban_graph_benchmark/outputs/mtl_311_water_v0/comparisons/09_visuals_index_ml_graph/figures/07_tract_month_graph_sample.png` |
| `cytoscape_nodes` | `urban_graph_benchmark/outputs/mtl_311_water_v0/comparisons/09_visuals_index_ml_graph/cytoscape/tract_month_graph_sample_nodes.csv` |
| `cytoscape_edges` | `urban_graph_benchmark/outputs/mtl_311_water_v0/comparisons/09_visuals_index_ml_graph/cytoscape/tract_month_graph_sample_edges.csv` |
| `cytoscape_cyjs` | `urban_graph_benchmark/outputs/mtl_311_water_v0/comparisons/09_visuals_index_ml_graph/cytoscape/tract_month_graph_sample.cyjs` |
| `cytoscape_graphml` | `urban_graph_benchmark/outputs/mtl_311_water_v0/comparisons/09_visuals_index_ml_graph/cytoscape/tract_month_graph_sample.graphml` |
| `fig_08_one_month_spatial_graph_dense` | `urban_graph_benchmark/outputs/mtl_311_water_v0/comparisons/09_visuals_index_ml_graph/figures/08_one_month_spatial_graph_dense.png` |
| `one_month_spatial_graph_dense_nodes` | `urban_graph_benchmark/outputs/mtl_311_water_v0/comparisons/09_visuals_index_ml_graph/cytoscape/one_month_spatial_graph_dense_nodes.csv` |
| `one_month_spatial_graph_dense_edges` | `urban_graph_benchmark/outputs/mtl_311_water_v0/comparisons/09_visuals_index_ml_graph/cytoscape/one_month_spatial_graph_dense_edges.csv` |
| `one_month_spatial_graph_dense_cyjs` | `urban_graph_benchmark/outputs/mtl_311_water_v0/comparisons/09_visuals_index_ml_graph/cytoscape/one_month_spatial_graph_dense.cyjs` |
| `fig_09_full_tract_month_graph_spatial_cloud` | `urban_graph_benchmark/outputs/mtl_311_water_v0/comparisons/09_visuals_index_ml_graph/figures/09_full_tract_month_graph_spatial_cloud.png` |
| `full_tract_month_graph_spatial_cloud_nodes` | `urban_graph_benchmark/outputs/mtl_311_water_v0/comparisons/09_visuals_index_ml_graph/cytoscape/full_tract_month_graph_spatial_cloud_nodes.csv` |
| `full_tract_month_graph_spatial_cloud_edges` | `urban_graph_benchmark/outputs/mtl_311_water_v0/comparisons/09_visuals_index_ml_graph/cytoscape/full_tract_month_graph_spatial_cloud_edges.csv` |
| `full_tract_month_graph_spatial_cloud_cyjs` | `urban_graph_benchmark/outputs/mtl_311_water_v0/comparisons/09_visuals_index_ml_graph/cytoscape/full_tract_month_graph_spatial_cloud.cyjs` |
| `fig_10_full_artifact_graph_spatial_cloud_with_placebo` | `urban_graph_benchmark/outputs/mtl_311_water_v0/comparisons/09_visuals_index_ml_graph/figures/10_full_artifact_graph_spatial_cloud_with_placebo.png` |
| `full_artifact_graph_spatial_cloud_with_placebo_nodes` | `urban_graph_benchmark/outputs/mtl_311_water_v0/comparisons/09_visuals_index_ml_graph/cytoscape/full_artifact_graph_spatial_cloud_with_placebo_nodes.csv` |
| `full_artifact_graph_spatial_cloud_with_placebo_edges` | `urban_graph_benchmark/outputs/mtl_311_water_v0/comparisons/09_visuals_index_ml_graph/cytoscape/full_artifact_graph_spatial_cloud_with_placebo_edges.csv` |
| `full_artifact_graph_spatial_cloud_with_placebo_cyjs` | `urban_graph_benchmark/outputs/mtl_311_water_v0/comparisons/09_visuals_index_ml_graph/cytoscape/full_artifact_graph_spatial_cloud_with_placebo.cyjs` |
| `metadata` | `urban_graph_benchmark/outputs/mtl_311_water_v0/comparisons/09_visuals_index_ml_graph/visual_metadata.json` |

## Interpretation note

Use these visuals to show that learned models are far stronger than static composite indices. Keep the no-edge and random-placebo controls visible: they are central to the scientific interpretation and prevent overclaiming the current spatial topology.