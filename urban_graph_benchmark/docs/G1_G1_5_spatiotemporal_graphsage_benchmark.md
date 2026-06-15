# G1 / G1.5 Spatiotemporal GraphSAGE Benchmark Documentation

**Project:** VILLE_IA municipal vulnerability / resilience benchmark  
**Benchmark stream:** Montréal 311 water/drainage tract-month prediction and high-risk ranking  
**Document purpose:** technical and interpretive reference for the graph-model benchmark layer, to support later preprint writing  
**Generated:** 2026-06-12  
**Status:** research documentation / benchmark record

---

## 1. Executive summary

This document records the construction, evaluation, and interpretation of the G1 and G1.5 spatiotemporal graph benchmark layer. The purpose of this layer was to test whether a graph-based model can improve over static composite vulnerability indices, calibrated index baselines, and tabular supervised ML baselines for the task of predicting and ranking high-burden Montréal census tract-months for water/drainage-related 311 reports.

The central conclusion is nuanced but useful:

> The G1/G1.5 benchmark layer clearly demonstrates that learned neural/graph-style models are far stronger than raw SVI-style composite indices for this 311 burden-ranking task. It also shows that high-risk ranking metrics such as NDCG@100 and top-10% overlap are more aligned with the municipal decision-support objective than MAE alone. However, the current real spatial graph topology is not yet conclusively validated, because no-edge neural controls and random spatial placebo controls remain highly competitive and in some cases stronger.

The first G1 pilot showed that a narrow GraphSAGE configuration selected by MAE/loss was not enough to establish strong graph value. A later G1.5 validation-selected architecture sweep, using validation NDCG@100 as the primary selection metric, showed that learned models can beat the A3 tabular ML baseline on high-risk ranking metrics. The selected graph/neural rows substantially outperform raw SVI-style direct ranking and calibrated SVI baselines.

The strongest public-safe claim is therefore:

> Moving from static composite vulnerability indices to supervised learned models, including graph/neural models, produces a large improvement in high-risk tract-month ranking. The graph benchmark layer is valuable and should be retained, but the current centroid/kNN spatial topology should be treated as a first graph construction rather than as a fully validated causal or infrastructural topology.

The current graph benchmark is satisfactory as a first research layer because it establishes: (1) a reproducible graph artifact; (2) feature-parity comparisons against tabular ML; (3) no-edge and placebo controls; (4) ranking-oriented evaluation; and (5) benchmark visuals and comparison outputs. The next research step should be to add the SoVI benchmark layer and then move toward a more advanced graph construction/model rather than indefinitely tuning G1.

---

## 2. Original research question and benchmark evolution

### 2.1 Original goal

The original goal was not to beat every possible supervised ML model. The initial research objective was more direct:

> Compare a graph-based model against social vulnerability index baselines such as SVI and SoVI, and test whether learned relational urban structure can outperform static composite vulnerability scoring for municipal risk prioritization.

This makes the graph benchmark meaningful even if stronger non-graph supervised ML baselines are later added. The research question is about the limitation of static composite indices and the potential value of learned spatiotemporal representations.

### 2.2 Why additional baselines were added

During development, additional baselines were inserted between the raw index baselines and the graph model. This created a more rigorous ladder:

1. **A0:** naive temporal / historical tract baseline.
2. **A1:** raw SVI-like direct ranking.
3. **A2:** calibrated SVI-style supervised model.
4. **A3:** feature-parity tabular ML baseline.
5. **G1:** initial spatiotemporal GraphSAGE pilot.
6. **G1.5:** validation-selected graph/neural architecture sweep.

This was scientifically useful because it separated several possible explanations:

- The graph could beat raw SVI simply because raw SVI is weak.
- The graph could beat calibrated SVI but not a strong tabular ML model.
- The graph could beat tabular ML only on ranking-oriented metrics, not on average count prediction.
- A no-edge neural model could explain part of the gain, implying that the architecture or training objective matters independently of graph topology.
- A placebo graph could match or beat the real spatial graph, implying that the current topology construction is not yet validated.

The benchmark ladder therefore made the comparison more honest. It also prevented the project from overclaiming graph value prematurely.

---

## 3. Data task and prediction target

### 3.1 Unit of observation

The primary graph unit is a **census tract-month**:

\[
\text{node} = (\text{census tract}, \text{month})
\]

The constructed graph contains:

| Quantity | Value |
|---|---:|
| Census tracts | 540 |
| Months | 53 |
| Tract-month nodes | 28,620 |
| Time range | 2022-01 through 2026-05 |

### 3.2 Target variable

The target is:

```text
water_drainage_count
```

This is the monthly count of water/drainage-related 311 reports in a census tract.

The supervised graph model is trained on:

```text
log1p(water_drainage_count)
```

and evaluated in count space after inverse transformation:

```text
prediction_count = expm1(predicted_log_count), clipped at zero
```

This matches the A3-style count prediction benchmark while enabling stable neural training.

### 3.3 Evaluation objective

There are two evaluation objectives:

1. **Count prediction:** how close predicted counts are to observed counts.
2. **High-risk prioritization:** how well the model ranks the most burdened tract-months.

The second objective is particularly important for municipal decision support. A model that ranks the top high-burden tract-months well may be operationally useful even if its average count calibration is imperfect.

---

## 4. Graph construction

### 4.1 Graph artifact location

The graph artifact is produced by:

```text
urban_graph_benchmark/src/ville_hgnn/graphs/build_tract_month_graph.py
```

Default artifact directory:

```text
urban_graph_benchmark/outputs/mtl_311_water_v0/graphs/G1_tract_month_graph/
```

Key files include:

```text
node_table.parquet
edge_table.parquet
target_vector.npy
binary_target_vector.npy
split_masks.npz
edge_mask_by_split_regime.npz
feature_matrix__<regime>__raw.npy
feature_columns__<regime>.json
graph_metadata.json
```

### 4.2 Nodes

Each node represents one census tract in one month. The node table stores spatial coordinates, tract identifiers, month identifiers, target values, and split membership metadata.

The node count is:

```text
28,620 = 540 tracts × 53 months
```

### 4.3 Edge types

The graph artifact currently includes four edge-type families:

| Edge type | Count | Meaning |
|---|---:|---|
| `temporal_self_lag_1` | 28,080 | Same tract, previous month → current month |
| `temporal_self_lag_12` | 22,140 | Same tract, same month previous year → current month |
| `spatial_knn_same_month` | 228,960 | Same month, k-nearest spatial tract links |
| `spatial_knn_same_month_random_placebo` | 228,960 | Randomized spatial-placebo links with same broad edge budget |
| **Total** | **508,140** | Full artifact including placebo-control edges |

The **real graph** excludes random placebo edges and contains:

```text
28,620 nodes
279,180 real edges
```

The full artifact including placebo-control edges contains:

```text
28,620 nodes
508,140 total edges
```

### 4.4 Edge regimes

The G1/G1.5 runner supports multiple edge regimes:

| Regime | Interpretation |
|---|---|
| `no_edges` | No message passing; MLP neural control |
| `temporal_only` | Temporal lag edges only |
| `spatial_only` | Same-month spatial kNN edges only |
| `spatial_temporal` | Real temporal + real spatial edges |
| `random_spatial_placebo` | Temporal + randomized spatial placebo edges |
| `all_edges` | Full artifact including controls when requested |

The key scientific controls are:

- **No-edge control:** tests whether the gain comes from neural architecture/training rather than graph structure.
- **Random spatial placebo:** tests whether spatial topology matters, or whether any dense/random connectivity can produce comparable smoothing/regularization.

### 4.5 Edge masks

The graph includes split-aware edge masks:

| Edge mask regime | Meaning |
|---|---|
| `all_edges` | Transductive edge use, including edges touching validation/test nodes |
| `train_train_edges` | Only edges where both endpoints are training nodes |
| `no_test_incident_edges` | Removes all edges touching test nodes |

The most important conservative mask for spatial-block evaluation is:

```text
no_test_incident_edges
```

This prevents message passing directly into or out of test nodes, making the graph result less dependent on transductive test-node connectivity.

---

## 5. Feature regimes

The graph benchmark uses A3-compatible feature regimes. The goal is feature parity: when comparing graph vs tabular ML, the graph should not win simply because it has more or different tabular features.

The main feature regimes are:

| Feature regime | Meaning |
|---|---|
| `all_forecasting` | Full forecasting-safe feature set |
| `lagged_reporting` | Lagged non-target/general reporting features |
| `no_target_history` | Forecasting features excluding direct target-history signals |

For the most recent G1.5 validation sweep, the compact sweep focused primarily on:

```text
all_forecasting
```

This was appropriate because the goal was to compare graph/neural models against the strongest A3-style feature-parity tabular baseline.

---

## 6. Splits

### 6.1 Temporal split

The temporal split evaluates future-period generalization.

Earlier A0/A1/A2 outputs mostly report temporal-test results. This is useful but not perfectly aligned with the later A3/G1 spatial-block comparisons.

### 6.2 Spatial-block split

The spatial-block split evaluates held-out geographic regions / tract blocks.

This is the main split used for A3, G1, and G1.5 comparison because it is closer to the question:

> Can the model generalize risk ranking across spatial areas rather than only forward in time within the same tracts?

The A3 spatial-block split has approximately:

| Partition | Rows |
|---|---:|
| Train | 21,200 |
| Validation | 5,353 |
| Test | 2,067 |

The most important current graph comparison is on:

```text
spatial_block_test
```

or equivalent `test` rows in G1/G1.5 output folders.

---

## 7. Model architecture: G1 GraphSAGE-style benchmark

### 7.1 Files

The main model code is in:

```text
urban_graph_benchmark/src/ville_hgnn/models/spatiotemporal_graphsage.py
```

The runner is:

```text
urban_graph_benchmark/src/ville_hgnn/baselines/g1_spatiotemporal_gnn.py
```

The validation sweep wrapper is:

```text
urban_graph_benchmark/src/ville_hgnn/baselines/g1_validation_sweep.py
```

### 7.2 Model family

The graph model is a typed, weighted GraphSAGE-style message-passing model.

At a high level, for a node \(i\), layer \(\ell\), and relation type \(r\):

\[
\mathbf{m}^{(\ell)}_{i,r} = \text{WeightedMean}\left(\{\mathbf{h}^{(\ell)}_j : (j \rightarrow i) \in E_r\}\right)
\]

Then relation-specific neighbor representations and the self representation are combined:

\[
\mathbf{h}^{(\ell+1)}_i = \sigma\left(W_{self}\mathbf{h}^{(\ell)}_i + \text{Combine}_r(W_r\mathbf{m}^{(\ell)}_{i,r})\right)
\]

The implemented model supports:

- typed relation-specific message passing;
- weighted mean aggregation;
- residual connections;
- layer normalization or no normalization;
- dropout;
- `mean` or `sum` relation combination;
- no-edge MLP fallback/control.

### 7.3 Why manual backend was retained

PyG / GraphGym were considered. However, the current manual backend preserves domain-specific requirements:

- typed edge regimes;
- edge weights;
- split-aware edge masks;
- A3-compatible features and metrics;
- custom benchmark outputs;
- direct no-edge and placebo controls.

The conclusion was that GraphGym can be useful later, but the immediate bottleneck was not library infrastructure. It was model-selection alignment and architecture search inside the existing controlled benchmark.

---

## 8. Metrics

The graph benchmark reports both count and ranking metrics.

| Metric | Direction | Purpose |
|---|---|---|
| MAE | lower is better | Count-space average absolute error |
| RMSE | lower is better | Count-space squared-error sensitivity |
| Mean Poisson deviance | lower is better | Count-like predictive distribution fit |
| Spearman | higher is better | Global rank correlation |
| Kendall | higher is better | Global rank correlation alternative |
| NDCG@100 | higher is better | Quality of top-100 high-burden ranking |
| Top-10% overlap | higher is better | Overlap between predicted and observed top decile |

### 8.1 Why NDCG@100 became central

The graph model’s most promising signal was not average count calibration. It was high-risk prioritization. Therefore, selection by validation MAE was misaligned with the most relevant graph story.

G1.5 introduced:

```text
--monitor-metric validation_ndcg_at_100
```

This enabled checkpoint and model selection based on validation NDCG@100.

This is scientifically legitimate because the selection metric is declared before the architecture sweep and only validation performance is used for selection.

---

## 9. G1 pilot benchmark

### 9.1 Initial G1 setup

The initial G1 pilot used a narrow GraphSAGE setting:

```text
hidden_dim = 128
num_layers = 2
dropout = 0.15
normalization = layernorm
residual = true
relation_combine = mean
backend = manual
seed = 20240610
```

The initial runner selected checkpoints primarily by validation MAE / validation loss. This produced a valid pipeline but likely under-explored the graph model space and under-emphasized the ranking objective.

### 9.2 G1 spatial-core run under NDCG monitor

After patching the runner to support ranking monitors, a spatial-core G1 run used:

```text
--preset spatial_core
--monitor-metric validation_ndcg_at_100
--max-epochs 250
--patience 40
```

It completed 27/27 trials with 0 failures.

The validation-selected overall G1 model was:

```text
G1 spatial_block / all_forecasting / spatial_temporal / no_test_incident_edges
```

Performance:

| Metric | Value |
|---|---:|
| Validation NDCG@100 | 0.706894 |
| Test MAE | 2.678841 |
| Test Spearman | 0.761745 |
| Test NDCG@100 | 0.805970 |
| Test top-10% overlap | 0.623188 |

This was meaningful because it beat A3 on NDCG@100 and top-10% overlap while nearly tying Spearman, although it lost clearly on MAE.

### 9.3 G1 pilot comparison against A3

A3 spatial-block selected HGB:

| Metric | A3 value |
|---|---:|
| Test MAE | 2.339043 |
| Test Spearman | 0.762380 |
| Test NDCG@100 | 0.784789 |
| Test top-10% overlap | 0.618357 |

G1 pilot spatial-temporal NDCG-selected model:

| Metric | G1 pilot value | Difference vs A3 |
|---|---:|---:|
| MAE | 2.678841 | worse |
| Spearman | 0.761745 | approximately tied / slightly lower |
| NDCG@100 | 0.805970 | +0.021181 |
| Top-10% overlap | 0.623188 | +0.004831 |

Interpretation:

> Switching G1 selection from validation MAE to validation NDCG@100 revealed a stronger ranking-oriented graph signal. The initial MAE-selected G1 pilot understated the ranking value of the graph/neural layer.

---

## 10. G1.5 validation-selected architecture sweep

### 10.1 Purpose


```text
G1.5 = validation-selected G1 architecture sweep
primary selection metric = validation NDCG@100
final comparison = spatial-block test metrics
controls = A3, no_edges MLP, random spatial placebo
```

### 10.2 Files

The G1.5 wrapper is:

```text
urban_graph_benchmark/src/ville_hgnn/baselines/g1_validation_sweep.py
```

The output directory used for the compact spatial NDCG sweep is:

```text
urban_graph_benchmark/outputs/mtl_311_water_v0/baselines/G1_5_validation_sweep_spatial_ndcg/
```

Important outputs:

```text
final_comparison.csv
selection_by_family.csv
metric_winners.csv
sweep_model_selection_audit.csv
factor_summary.csv
g1_validation_sweep_report.md
```

### 10.3 Architecture search space

The compact sweep used 36 architecture configurations. The primary dimensions included:

| Dimension | Values |
|---|---|
| hidden_dim | 64, 128 |
| num_layers | 1, 2, 3 |
| dropout | 0.0, 0.05, 0.15 |
| normalization | layernorm |
| residual | true |
| relation_combine | mean, sum |
| backend | manual |
| learning_rate | 0.001 |
| weight_decay | 0.0001 |
| seed | 20240610 |

For each architecture, the wrapper ran family-level variants:

```text
no_edges
spatial_temporal
temporal_only
random_spatial_placebo
```

with relevant edge masks.

### 10.4 Final selected family representatives

The final selected representatives were:

#### No-edge neural control

```text
G1__spatial_block__all_forecasting__no_edges__all_edges__h64_L2_do0p05_layernorm_res_relmean_manual_lr0p001_wd0p0001_seed20240610
```

| Metric | Value |
|---|---:|
| Test MAE | 2.411409 |
| Test Spearman | 0.763784 |
| Test NDCG@100 | 0.805768 |
| Test top-10% overlap | 0.628019 |

#### Random spatial placebo control

```text
G1__spatial_block__all_forecasting__random_spatial_placebo__all_edges__h128_L2_do0p05_layernorm_res_relmean_manual_lr0p001_wd0p0001_seed20240610
```

| Metric | Value |
|---|---:|
| Test MAE | 2.293357 |
| Test Spearman | 0.777173 |
| Test NDCG@100 | 0.799908 |
| Test top-10% overlap | 0.642512 |

#### Real spatial-temporal graph

```text
G1__spatial_block__all_forecasting__spatial_temporal__no_test_incident_edges__h128_L2_do0p05_layernorm_res_relmean_manual_lr0p001_wd0p0001_seed20240610
```

| Metric | Value |
|---|---:|
| Test MAE | 2.418032 |
| Test Spearman | 0.741807 |
| Test NDCG@100 | 0.802691 |
| Test top-10% overlap | 0.603865 |

#### Temporal-only graph

```text
G1__spatial_block__all_forecasting__temporal_only__no_test_incident_edges__h128_L3_do0p15_layernorm_res_relmean_manual_lr0p001_wd0p0001_seed20240610
```

| Metric | Value |
|---|---:|
| Test MAE | 2.682400 |
| Test Spearman | 0.766072 |
| Test NDCG@100 | 0.796453 |
| Test top-10% overlap | 0.628019 |

### 10.5 Metric winners

Among the selected final representatives:

| Metric | Winner | Value |
|---|---|---:|
| MAE | G1.5 random spatial placebo | 2.293357 |
| Spearman | G1.5 random spatial placebo | 0.777173 |
| NDCG@100 | G1.5 no-edge neural control | 0.805768 |
| Top-10% overlap | G1.5 random spatial placebo | 0.642512 |

When the G1 pilot spatial-temporal row is included in the broader comparison, it has the highest NDCG@100:

| Row | NDCG@100 |
|---|---:|
| G1 pilot spatial-temporal | 0.805970 |
| G1.5 no-edge control | 0.805768 |
| Difference | +0.000202 |

This margin is extremely small and should not be overinterpreted without multi-seed confirmation.

### 10.6 Interpretation of G1.5

G1.5 demonstrates that neural/graph-style models can beat A3 and dramatically outperform SVI-style composite indices on ranking metrics. However, the real spatial-temporal graph is not the strongest family after controlled validation selection.

The most important conclusion is:

> G1.5 confirms that ranking-oriented neural model selection is valuable. It does not yet prove that the current centroid/kNN spatial topology is the cause of the improvement.

The random placebo and no-edge controls are not a problem; they are a sign that the benchmark is working scientifically. They reveal that the current graph construction is not yet sufficiently distinct from neural representation learning or generic connectivity effects.

---

## 11. Index → ML → Graph comparison

### 11.1 Comparison script

The consolidated comparison is generated by:

```text
urban_graph_benchmark/scripts/08_compare_index_ml_graph_benchmark.py
```

Output directory:

```text
urban_graph_benchmark/outputs/mtl_311_water_v0/comparisons/08_index_ml_graph_benchmark/
```

Important outputs:

```text
index_ml_graph_metrics_long.csv
benchmark_comparison.csv
benchmark_comparison_compact.csv
metric_winners.csv
family_margin_table.csv
missing_input_audit.csv
benchmark_interpretation.md
comparison_metadata.json
plots/
```

### 11.2 Current comparison rows

The compact comparison currently includes:

| Row | Group |
|---|---|
| A0 tract historical mean | naive temporal baseline |
| A1 raw SVI class | composite index |
| A1 raw SVI percentile | composite index |
| A2 calibrated SVI static | calibrated index |
| A2 calibrated SVI retrospective | calibrated/retrospective diagnostic |
| A3 selected tabular ML | feature-parity tabular ML |
| G1 pilot spatial-temporal | graph/neural pilot |
| G1.5 selected no-edge neural control | neural control |
| G1.5 selected spatial-temporal graph | graph/neural |
| G1.5 selected temporal graph | graph/neural |
| G1.5 selected random spatial placebo | placebo control |
| G1.5 selected A3_frozen | frozen A3 reference row |

SoVI is expected but currently missing:

```text
urban_graph_benchmark/outputs/mtl_311_water_v0/baselines/A1_sovi_direct_ranking/metrics.csv
```

This is explicitly recorded in `missing_input_audit.csv` and is expected until the SoVI benchmark adapter is generated.

### 11.3 Current compact comparison values

Selected values:

| Label | MAE | Spearman | NDCG@100 | Top-10% overlap |
|---|---:|---:|---:|---:|
| A1 raw SVI percentile | — | 0.160639 | 0.220560 | 0.052411 |
| A1 raw SVI class | — | 0.185642 | 0.222316 | 0.150943 |
| A2 calibrated SVI static | 3.434751 | 0.236370 | 0.193140 | 0.121399 |
| A2 calibrated SVI retrospective | 2.522434 | 0.671736 | 0.702508 | 0.462963 |
| A0 tract historical mean | 2.489209 | 0.694235 | 0.748363 | 0.483539 |
| A3 selected tabular ML | 2.339043 | 0.762380 | 0.784789 | 0.618357 |
| G1 pilot spatial-temporal | 2.678841 | 0.761745 | 0.805970 | 0.623188 |
| G1.5 no-edge neural control | 2.411409 | 0.763784 | 0.805768 | 0.628019 |
| G1.5 spatial-temporal graph | 2.418032 | 0.741807 | 0.802691 | 0.603865 |
| G1.5 temporal graph | 2.682400 | 0.766072 | 0.796453 | 0.628019 |
| G1.5 random spatial placebo | 2.293357 | 0.777173 | 0.799908 | 0.642512 |

### 11.4 Major margins

Relative to the best calibrated SVI diagnostic:

| Comparison | Metric | Best graph/neural value | Best index value | Margin |
|---|---|---:|---:|---:|
| Graph family vs calibrated index | NDCG@100 | 0.805970 | 0.702508 | +0.103462 |
| Graph family vs calibrated index | Top-10% overlap | 0.628019 | 0.462963 | +0.165056 |
| Graph family vs calibrated index | Spearman | 0.766072 | 0.671736 | +0.094336 |

Relative to A3:

| Comparison | Metric | Best graph/neural value | A3 value | Margin |
|---|---|---:|---:|---:|
| Graph family vs A3 | NDCG@100 | 0.805970 | 0.784789 | +0.021181 |
| Graph family vs A3 | Top-10% overlap | 0.628019 | 0.618357 | +0.009662 |
| Graph family vs A3 | Spearman | 0.766072 | 0.762380 | +0.003691 |

Relative to no-edge/placebo controls:

| Comparison | Metric | Best graph/neural value | Best control value | Margin |
|---|---|---:|---:|---:|
| Graph family vs controls | NDCG@100 | 0.805970 | 0.805768 | +0.000202 |
| Graph family vs controls | Spearman | 0.766072 | 0.777173 | -0.011101 |
| Graph family vs controls | Top-10% overlap | 0.628019 | 0.642512 | -0.014493 |
| Graph family vs controls | MAE | 2.339043 / 2.418032 depending row | 2.293357 | control better |

The NDCG advantage over controls is tiny and comes from the G1 pilot row. It should be treated as suggestive, not conclusive.

---

## 12. Benchmark visuals

### 12.1 Visual script

The visual script is:

```text
urban_graph_benchmark/scripts/09_generate_benchmark_visuals.py
```

Output directory:

```text
urban_graph_benchmark/outputs/mtl_311_water_v0/comparisons/09_visuals_index_ml_graph/
```

Important output groups:

```text
figures/
cytoscape/
visual_manifest.csv
visual_metadata.json
visual_summary.md
```

### 12.2 Comparison visuals

The script generates post-hoc comparison visuals:

```text
01_benchmark_metric_panels.png
02_index_vs_learned_ranking_gap.png
03_family_margin_panels.png
04_g1_family_comparison.png
05_g1_validation_sweep_heatmap.png
06_benchmark_pipeline_schema.png
```

These are useful for the benchmark/results section.

### 12.3 Graph visuals

The graph visual iterations produced several formats.

#### Local sample graph

```text
07_tract_month_graph_sample.png
```

This is a small explanatory ego/subgraph view. It is useful for explaining local neighborhoods, but it is not visually representative of the full graph scale.

#### Dense one-month spatial graph

```text
08_one_month_spatial_graph_dense.png
```

This shows one monthly spatial layer:

```text
540 tract nodes
4,320 spatial kNN edges
```


#### Full real graph hairball / spatial-cloud views

The first full-graph attempt used a layered x/y representation:

```text
x-axis = normalized tract spatial x-coordinate
y-axis = month layer
```

This showed the full real graph:

```text
28,620 nodes
279,180 real edges
```

However, it looked like a snowball / spreadsheet-like temporal lattice. It was useful for scale but not aesthetically appropriate for the main visual.

The updated objective is a **spatial-cloud full-graph view**, inspired by dense biological/network visualizations. The preferred full graph visual should:

- remove visible x/y axes;
- avoid a temporal spreadsheet layout;
- use a 2D/2.5D cloud-like layout;
- preserve geography-like structure;
- draw many transparent edges;
- color nodes by burden;
- highlight highest-burden tract-months;
- create a visually impressive dense graph while remaining connected to the actual graph artifact.

Key updated output names:

```text
09_full_tract_month_graph_spatial_cloud.png
10_full_artifact_graph_spatial_cloud_with_placebo.png
```

The real graph version excludes placebo edges. The artifact version includes placebo edges and should be labeled clearly as including controls.

### 12.4 Cytoscape exports

The visual script also produces Cytoscape-compatible exports:

```text
cytoscape/*.csv
cytoscape/*.cyjs
cytoscape/*.graphml  # optional when NetworkX is available
```

The revised spatial-cloud export does not require NetworkX. NetworkX is only needed for optional GraphML export.

---

## 13. What can be claimed safely

### 13.1 Strong supported claims

The following claims are well supported by the current benchmark:

1. **Raw SVI-style direct ranking is weak for this water/drainage 311 burden task.**  
   SVI percentile has Spearman ≈ 0.16 and NDCG@100 ≈ 0.22, far below learned models.

2. **Calibrating an index helps but remains below the learned benchmark layer.**  
   The retrospective calibrated SVI diagnostic is much stronger than raw SVI but still below A3/G1/G1.5 on high-risk ranking metrics.

3. **A3 establishes a strong feature-parity tabular ML baseline.**  
   A3 is stronger than raw/composite indices and must be retained as the main non-graph ML comparator.

4. **G1/G1.5 demonstrates that neural/graph benchmark layers can beat A3 on ranking-oriented metrics.**  
   The strongest graph/neural rows improve NDCG@100 and top-10 overlap relative to A3.

5. **Validation NDCG@100 is a more aligned selection metric than validation MAE for the high-risk ranking story.**  
   G1 performance improved materially when checkpoint/model selection was aligned with the ranking objective.

6. **No-edge and random-placebo controls are essential.**  
   These controls prevent overclaiming and reveal that the current spatial topology is not yet conclusively responsible for the gains.

### 13.2 Claims that should be avoided or qualified

The following claims should **not** be made without qualification:

1. **“The spatial graph topology is validated.”**  
   Too strong. Random placebo and no-edge controls remain competitive or stronger.

2. **“G1.5 proves graph message passing is the source of improvement.”**  
   Too strong. Some of the best results come from no-edge or placebo controls.

3. **“The graph beats all ML baselines overall.”**  
   Too broad. The best result depends on metric and family.

4. **“Spatial kNN is the correct urban topology.”**  
   Not yet supported. The current kNN graph should be treated as a first spatial graph construction.

### 13.3 Expected wording


> The graph/neural benchmark layer substantially outperforms raw SVI-style composite ranking and calibrated SVI baselines for water/drainage 311 burden prioritization. Under spatial-block evaluation, G1/G1.5 models improve high-risk ranking metrics relative to the A3 tabular ML baseline in selected configurations, especially NDCG@100. However, no-edge and random-placebo controls remain highly competitive, indicating that the current centroid/kNN spatial topology should be interpreted as a first graph construction rather than conclusive evidence of spatial-topological mechanism.

Shorter version:

> G1/G1.5 validates the value of moving beyond static composite indices and motivates graph/neural modeling, while also showing that stronger spatial topology design is needed before claiming topology-specific effects.

---

## 14. What remains to complete the benchmark layer

### 14.1 Add SoVI benchmark

The current comparison script expects SoVI at:

```text
urban_graph_benchmark/outputs/mtl_311_water_v0/baselines/A1_sovi_direct_ranking/metrics.csv
```

This does not exist yet. The next benchmark task should be:

```text
A1_sovi_direct_ranking
```

or equivalent SoVI adapter, producing metrics in the same format as the SVI direct-ranking baseline.

The SoVI result should be added to:

```text
08_compare_index_ml_graph_benchmark.py
```

without changing the comparison structure.

### 14.2 Align index baselines to spatial-block split

The current consolidated comparison mixes split regimes:

- A1/A2 SVI rows are mostly temporal-test outputs.
- A3/G1/G1.5 rows are spatial-block test outputs.

This is acceptable for preliminary benchmark consolidation, but the final preprint should ideally compare:

```text
SVI vs SoVI vs A3 vs G1/G1.5 on the same spatial-block test set
```

Therefore, after SoVI is added, the next refinement should be to ensure that SVI/SoVI direct rankings are also evaluated on the spatial-block test nodes.

### 14.3 Multi-seed confirmation

The G1.5 compact sweep used one seed:

```text
20240610
```

Before making strong graph-specific claims, the selected families should be confirmed across multiple seeds. This is especially important because margins between graph, no-edge, and placebo controls are small on some ranking metrics.

### 14.4 Improve graph topology

The current spatial topology is centroid/kNN. Future graph construction should test more meaningful urban/infrastructure relations, such as:

- drainage/sewer catchments;
- hydrological flow or elevation/slope-based adjacency;
- road network connectivity;
- impervious-surface adjacency;
- service-network dependencies;
- proximity to critical drainage infrastructure;
- learned or multi-relation urban graph construction.

The current results suggest that graph topology design, not just GraphSAGE architecture, is the next bottleneck.

---

## 15. Reproducibility commands

### 15.1 Build graph artifact

The graph artifact was created by the graph builder. The canonical command pattern is:

```bash
PYTHONPATH=urban_graph_benchmark/src python -m ville_hgnn.graphs.build_tract_month_graph \
  --config urban_graph_benchmark/configs/mtl_311_water_v0.1.yaml \
  --overwrite
```

Exact arguments may differ depending on current project CLI, but the output directory should be:

```text
urban_graph_benchmark/outputs/mtl_311_water_v0/graphs/G1_tract_month_graph/
```

### 15.2 Run G1 spatial-core NDCG monitor

```bash
PYTHONPATH=urban_graph_benchmark/src python -m ville_hgnn.baselines.g1_spatiotemporal_gnn \
  --graph-dir urban_graph_benchmark/outputs/mtl_311_water_v0/graphs/G1_tract_month_graph \
  --output-dir urban_graph_benchmark/outputs/mtl_311_water_v0/baselines/G1_spatiotemporal_gnn_spatial_core_ndcg_monitor \
  --preset spatial_core \
  --max-epochs 250 \
  --patience 40 \
  --monitor-metric validation_ndcg_at_100 \
  --overwrite
```

### 15.3 Run G1.5 validation sweep

```bash
PYTHONPATH=urban_graph_benchmark/src python -m ville_hgnn.baselines.g1_validation_sweep \
  --graph-dir urban_graph_benchmark/outputs/mtl_311_water_v0/graphs/G1_tract_month_graph \
  --a3-dir urban_graph_benchmark/outputs/mtl_311_water_v0/baselines/A3_feature_parity_tabular_spatial_block \
  --output-dir urban_graph_benchmark/outputs/mtl_311_water_v0/baselines/G1_5_validation_sweep_spatial_ndcg \
  --sweep-preset compact \
  --max-epochs 250 \
  --patience 40 \
  --monitor-metric validation_ndcg_at_100 \
  --overwrite
```

### 15.4 Generate consolidated comparison

```bash
PYTHONPATH=urban_graph_benchmark/src python urban_graph_benchmark/scripts/08_compare_index_ml_graph_benchmark.py
```

### 15.5 Generate visuals

```bash
PYTHONPATH=urban_graph_benchmark/src python urban_graph_benchmark/scripts/09_generate_benchmark_visuals.py \
  --overwrite \
  --full-graph-edge-mode both \
  --full-cyjs-edge-limit 700000
```

---

## 16. Suggested preprint structure for this benchmark section

### 16.1 Methods subsection

Possible heading:

```text
Spatiotemporal graph benchmark
```

Content:

- Define tract-month nodes.
- Define target as water/drainage 311 count.
- Describe temporal lag edges and same-month spatial kNN edges.
- Describe placebo graph.
- Describe feature-parity with A3.
- Describe GraphSAGE-style typed weighted message passing.
- Describe validation NDCG@100 model selection.
- Describe no-edge and placebo controls.

### 16.2 Results subsection

Possible heading:

```text
Graph/neural models improve high-risk ranking beyond composite indices
```

Content:

- Show raw SVI is weak.
- Show calibrated SVI improves but remains below A3/G1/G1.5.
- Show A3 is strong.
- Show G1/G1.5 improves NDCG@100/top-risk ranking in selected rows.
- Explicitly state that no-edge/placebo controls remain strong.

### 16.3 Discussion subsection

Possible heading:

```text
Graph value and topology controls
```

Content:

- State that graph/neural benchmark is justified.
- State that current spatial kNN topology is not yet conclusively validated.
- Motivate future work on infrastructure/hydrological graph construction.

---

## 17. Final interpretation

The G1/G1.5 benchmark is a successful first graph benchmark layer. It should not be judged only by whether the current spatial-temporal graph dominates every control. Its main contribution is that it created a rigorous ladder from composite indices to calibrated indices, tabular ML, graph/neural models, no-edge controls, and placebo graph controls.

The research now has a coherent result:

1. Static SVI-style ranking is far too weak for this task.
2. Calibrated SVI improves but remains below learned models.
3. A3 is a strong tabular ML baseline.
4. G1/G1.5 can improve high-risk ranking metrics and is worth developing.
5. The current spatial kNN topology is not enough to claim definitive topology-specific value.
6. The next benchmark task is SoVI inclusion.
7. The next modeling task is a more meaningful graph construction / advanced graph model.

This is a satisfactory stopping point for the first G1/G1.5 GraphSAGE benchmark cycle.
