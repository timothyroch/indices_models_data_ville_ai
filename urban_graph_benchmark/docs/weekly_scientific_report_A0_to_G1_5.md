# Weekly technical report — Benchmark layer from A0 to G1/G1.5

**Author:** Timothy Roch  
**Project:** VILLE_IA — resilience/vulnerability benchmark and graph-based modelling  
**Period covered:** Current weekly development cycle  
**Scope:** A0 through G1/G1.5 benchmark layer for tract-month water/drainage 311 burden

---

## 1. Executive summary

This week I built a first controlled benchmark layer for predicting and ranking water/drainage-related 311 burden at the tract-month level. The work moved from static composite vulnerability indices to learned tabular models and then to a first graph/neural benchmark layer. The main objective was not to produce the final research model, but to establish whether the graph direction is meaningful enough to justify moving beyond static index reproduction.

The benchmark now contains five main layers:

1. **A0 — historical tract baseline.** A strong naive temporal baseline using tract-level historical reporting patterns.
2. **A1 — raw SVI-style direct ranking.** A static composite vulnerability score used directly as a risk ranking signal.
3. **A2 — calibrated SVI variants.** Supervised calibration of SVI-style information, including a retrospective diagnostic variant.
4. **A3 — feature-parity tabular ML.** A stronger supervised non-graph baseline using lagged reporting and forecasting-safe features.
5. **G1/G1.5 — graph/neural benchmark layer.** A tract-month spatiotemporal graph and GraphSAGE-style message-passing models, with no-edge and random-placebo controls.

The most important result is that the raw SVI-style index is very weak as a direct predictor/ranker of water/drainage 311 burden, whereas learned models are much stronger. The graph/neural layer substantially outperforms static composite index baselines and can improve high-risk ranking over A3 on NDCG@100. However, the current spatial topology is not yet conclusively validated: no-edge and random-placebo neural controls remain highly competitive, and in some metrics they outperform the real spatial-temporal graph. Therefore, the correct interpretation is that the benchmark strongly supports moving beyond static indices and continuing the graph/neural direction, but the current centroid/k-nearest-neighbor spatial graph should be treated as a first graph construction rather than as a validated final topology.

The immediate next step is to add the SoVI-style benchmark to the same comparison table and then move toward a more advanced graph model or improved graph construction.

---

## 2. Prediction task and evaluation frame

The task is formulated at the level of **census tract × month**. Each observation corresponds to one tract in one month. The target is the water/drainage 311 reporting burden for that tract-month. Models are evaluated in count space and ranking space.

The evaluation emphasizes two distinct questions:

- **Prediction accuracy:** Can the model estimate the number of water/drainage reports? Main metrics: MAE, RMSE, mean Poisson deviance.
- **Priority ranking:** Can the model identify the tract-months with the highest burden? Main metrics: Spearman correlation, NDCG@100, and top-10% overlap.

The ranking metrics are particularly important for municipal decision support because the practical question is often not only “what exact count will occur?”, but also “which tract-months should be prioritized?” This distinction became important when comparing A3 and G1/G1.5: the graph/neural layer is more promising on high-risk ranking than on average count calibration.

A caveat is that some early index baselines are currently evaluated under temporal-test conditions, whereas the strongest A3/G1/G1.5 comparisons are spatial-block tests. This is acceptable for the current internal benchmark summary, but the next clean benchmark consolidation should place SVI and SoVI on the same spatial-block comparison whenever possible.

---

## 3. Benchmark layers produced this week

### 3.1 A0 — historical tract baseline

A0 is a strong naive baseline based on historical tract behavior. Its role is to prevent overclaiming against weak baselines. In this task, a tract’s past reporting level is already informative, so a graph or machine-learning model must be compared not only against raw indices, but also against tract-history baselines.

A0 performs strongly relative to raw vulnerability indices. This confirms that temporal/reporting history is a major component of the water/drainage 311 signal.

### 3.2 A1 — raw SVI-style direct ranking

A1 uses a static SVI-style composite score directly as a risk ranking signal. This is the closest representation of the original “index versus graph” question: if a vulnerability index is used directly to prioritize areas, how well does it rank observed water/drainage burden?

The result is weak. The raw SVI percentile and class both have very low correlation and NDCG relative to learned models. This does not mean SVI is useless as a social vulnerability construct; rather, it means that the static composite score is not sufficient as a direct operational ranking model for this specific 311 burden target.

### 3.3 A2 — calibrated SVI variants

A2 tests whether SVI-style information can become more useful when supervised calibration is allowed. The static calibrated SVI model remains weak, while the retrospective diagnostic variant performs much better because it includes information closer to observed reporting dynamics.

The retrospective A2 result is useful diagnostically: reporting history and calibration can recover substantial signal. However, it is not the cleanest forecasting-style model. It should be interpreted as a bridge between static index reproduction and supervised prediction, not as the main final benchmark.

### 3.4 A3 — feature-parity tabular ML

A3 is the strongest non-graph supervised baseline. It uses forecasting-safe tabular features, including lagged reporting features and spatial/static covariates, while avoiding same-month target leakage. This is the key baseline that makes the graph comparison scientifically serious.

The best spatial-block A3 model is a histogram gradient boosting Poisson model using lagged reporting features. It performs strongly on both count and ranking metrics and is currently the strongest non-graph tabular baseline.

The A3 analysis also produced interpretable response-surface figures. These are useful because they show what the tabular model is learning from lagged reporting features and spatial/reporting histories.

![A3 response surface: lagged non-water reporting and lagged total request history](../outputs/mtl_311_water_v0/baselines/A3_feature_parity_tabular_spatial_block_with_plots/plots/surface_3d__hist_gradient_boosting_poisson_A3_lagged_reporting_forecasting_hgb_poisson_02__reporting_history_total_311_count_non_water_drainage_lag_1__requests_history_requests_total_lag_1.png)

![A3 response surface: rolling non-water reporting and rolling total request history](../outputs/mtl_311_water_v0/baselines/A3_feature_parity_tabular_spatial_block_with_plots/plots/surface_3d__hist_gradient_boosting_poisson_A3_lagged_reporting_forecasting_hgb_poisson_02__reporting_history_total_311_count_non_water_drainage_roll3_mean_shift1__requests_history_requests_total_roll3_mean_shift1.png)

![A3 response surface: target-history and non-water reporting history](../outputs/mtl_311_water_v0/baselines/A3_feature_parity_tabular_with_plots/plots/surface_3d__random_forest_log_count_A3_all_forecasting_rf_log_count_conservative__target_history_water_drainage_count_lag_1__reporting_history_total_311_count_non_water_drainage_lag_1.png)

### 3.5 G1 — first spatiotemporal graph benchmark

G1 constructs a tract-month graph. Each node is a tract-month. The graph contains temporal self-lag edges and same-month spatial k-nearest-neighbor edges between tracts. The initial graph artifact contains approximately 28,620 nodes and hundreds of thousands of typed edges, depending on whether placebo/control edges are included.

The first graph model is a GraphSAGE-style neural model with typed/weighted message passing. The purpose of G1 is not to be the final model, but to test whether graph message passing adds value beyond A3-compatible features. The G1 experiments include important controls:

- **No-edge neural control:** same neural architecture without graph edges.
- **Temporal-only graph:** temporal self-lag edges only.
- **Spatial-only graph:** same-month tract spatial edges only.
- **Spatial-temporal graph:** both spatial and temporal edges.
- **Random spatial placebo:** randomized spatial edges used as a control for generic message-passing effects.

The graph representation is large-scale enough to be meaningful as a benchmark object.

![Full tract-month graph spatial cloud](../outputs/mtl_311_water_v0/comparisons/09_visuals_index_ml_graph/figures/09_full_tract_month_graph_spatial_cloud.png)

### 3.6 G1.5 — validation-selected architecture sweep

G1.5 is a controlled architecture-selection step around G1. It was added after the first G1 run showed that the initial architecture and selection metric were probably too narrow. The key change was to select checkpoints and candidate models using **validation NDCG@100** rather than validation MAE, because the most promising graph/neural signal is high-risk ranking rather than average count calibration.

G1.5 remains an internal model-selection layer, not a new conceptual model. It is useful because it tests whether the graph/neural direction remains competitive after allowing a bounded validation-only architecture sweep. The sweep varies architecture choices such as depth, hidden dimension, dropout, relation aggregation, and edge regime, while keeping the evaluation controlled through validation selection, held-out testing, no-edge controls, and random-placebo controls.

The main G1.5 result is nuanced. G1.5 confirms that neural ranking models can outperform A3 on some ranking metrics. However, the no-edge and random-placebo controls remain very strong. Therefore, the current evidence supports the graph/neural benchmark direction, but it does not yet prove that the current real spatial topology is the causal source of the improvement.

---

## 4. Consolidated results

The table below summarizes the current benchmark comparison. Lower MAE is better; higher Spearman, NDCG@100, and top-10% overlap are better. Some rows are evaluated on temporal-test outputs and others on spatial-block test outputs; this will be harmonized in the next benchmark consolidation step.

| Model / representative row | Benchmark group | Test split | MAE | Spearman | NDCG@100 | Top-10% overlap | Interpretation |
|---|---|---:|---:|---:|---:|---:|---|
| A0 tract historical mean | Naive temporal baseline | temporal | 2.4892 | 0.6942 | 0.7484 | 0.4835 | Strong tract-history baseline. |
| A1 raw SVI percentile | Composite index | temporal | — | 0.1606 | 0.2206 | 0.0524 | Raw static SVI-style score used directly as ranking signal. |
| A1 raw SVI class | Composite index | temporal | — | 0.1856 | 0.2223 | 0.1509 | Diagnostic class-based SVI ranking. |
| A2 calibrated SVI static | Calibrated index | temporal | 3.4348 | 0.2364 | 0.1931 | 0.1214 | Static SVI calibration remains weak. |
| A2 calibrated SVI retrospective | Calibrated index / diagnostic | temporal | 2.5224 | 0.6717 | 0.7025 | 0.4630 | Stronger but retrospective/diagnostic. |
| A3 selected tabular ML | Tabular ML | spatial-block | 2.3390 | 0.7624 | 0.7848 | 0.6184 | Strongest frozen non-graph supervised baseline. |
| G1 pilot spatial-temporal | Graph/neural | spatial-block | 2.6788 | 0.7617 | 0.8060 | 0.6232 | Strong ranking result, especially NDCG@100. |
| G1.5 selected no-edge control | Neural control | spatial-block | 2.4114 | 0.7638 | 0.8058 | 0.6280 | Shows neural ranking value without graph topology. |
| G1.5 selected spatial-temporal graph | Graph/neural | spatial-block | 2.4180 | 0.7418 | 0.8027 | 0.6039 | Beats A3 on NDCG@100, but not controls overall. |
| G1.5 selected temporal graph | Graph/neural | spatial-block | 2.6824 | 0.7661 | 0.7965 | 0.6280 | Ranking-competitive, but weak count calibration. |
| G1.5 selected random spatial placebo | Placebo control | spatial-block | **2.2934** | **0.7772** | 0.7999 | **0.6425** | Strongest overall row, but it is a control. |

### Main quantitative takeaways

1. **Raw SVI-style ranking is weak for this target.** The raw SVI percentile has Spearman 0.1606 and NDCG@100 0.2206, far below A3/G1/G1.5.
2. **A3 is a strong supervised non-graph baseline.** It reaches MAE 2.3390, Spearman 0.7624, and NDCG@100 0.7848 on the spatial-block test.
3. **G1/G1.5 can improve high-risk ranking.** The G1 pilot spatial-temporal row reaches NDCG@100 0.8060, a margin of roughly +0.0212 over A3.
4. **The strongest current row is a control.** The random spatial placebo obtains the best MAE, Spearman, and top-10% overlap. This prevents overclaiming about the current spatial topology.
5. **The no-edge control is also strong.** This indicates that part of the gain comes from neural model selection and ranking-oriented checkpoint selection, not necessarily from graph structure.

---

## 5. Interpretation for the research direction

The original research question was whether graph-based modelling can improve over static vulnerability indices such as SVI-style scores. On that question, the answer is clearly encouraging: the graph/neural benchmark layer and the supervised ML baselines are much stronger than the raw composite index.

The more demanding question is whether the **current spatial graph topology itself** is responsible for the improvement. The answer is not yet. The no-edge and random-placebo controls remain too strong to claim that the centroid/kNN spatial topology is validated. This is a useful result rather than a failure. It suggests that the next research gains may come from improving graph construction rather than only tuning the GNN architecture.

The current benchmark therefore supports the following careful statement:

> Static SVI-style composite scores are weak direct predictors/rankers of tract-month water/drainage 311 burden. Supervised tabular models and graph/neural benchmark layers are substantially stronger, especially for high-risk ranking. The current G1/G1.5 experiments show that graph/neural models can improve NDCG@100 over A3, but no-edge and random-placebo controls remain highly competitive, so the present centroid-based spatial topology should be treated as a first graph construction rather than as a validated final urban graph.

This is a satisfactory stopping point for the first benchmark layer. The benchmark now provides enough evidence to justify moving toward a more advanced model, while preserving scientific caution about what has and has not been validated.

---

## 6. Technical issues and methodological safeguards

Several safeguards were added to avoid over-interpreting early results:

- **Validation-only model selection.** G1.5 selects configurations using validation NDCG@100, then reports held-out test metrics.
- **No-edge neural control.** This separates neural model capacity from graph topology.
- **Random spatial placebo.** This tests whether message passing over arbitrary spatial-like edges performs as well as or better than real spatial edges.
- **Ranking-oriented metrics.** NDCG@100 and top-10% overlap are reported alongside MAE and Spearman because municipal prioritization is partly a ranking problem.
- **Explicit interpretation boundary.** The current results support learned benchmarking and graph/neural exploration, but not yet a strong claim that the current spatial topology is the source of improvement.

One methodological caveat remains: not every benchmark row is currently evaluated under the exact same split scheme. The clean next step is to place SVI and SoVI baselines under the same spatial-block comparison as A3/G1/G1.5.

---

## 7. What I would like feedback on

The current results raise several technical questions where feedback would be useful:

1. **Choice of graph topology.** The centroid/kNN spatial graph is easy to construct and auditable, but the random-placebo result suggests that it may not encode the right urban dependency structure. A more meaningful graph may require hydrological, drainage, road, slope, impervious-surface, sewer-catchment, or infrastructure-network information.
2. **Target framing.** The ranking metrics appear more aligned with decision support than MAE alone. I would like feedback on whether high-risk ranking should be treated as the primary evaluation axis.
3. **SoVI integration.** The next missing benchmark is a SoVI-style index comparison. Adding SoVI will make the “index versus graph/neural model” story more complete.
4. **Advanced model direction.** The benchmark suggests that moving beyond static indices is justified, but the next model should likely improve graph construction and/or learn graph structure rather than only re-tune the current GraphSAGE layer.

---

## 8. Next steps

The immediate next steps are:

1. **Add a SoVI-style benchmark row.** This should be integrated into the same comparison table as SVI, A3, and G1/G1.5.
2. **Harmonize split comparisons.** SVI and SoVI should be evaluated on the same spatial-block test setting when possible.
3. **Freeze the first benchmark layer.** A0–G1/G1.5 now provides a useful baseline ladder from static indices to learned models.
4. **Move toward a more advanced graph model.** The next graph model should focus less on generic architecture tuning and more on better graph construction or learned dependencies.
5. **Preserve control results in the report.** The no-edge and placebo controls should remain visible because they make the analysis more credible and clarify what the current graph does and does not prove.

---

## 9. Proposed concise conclusion

This week’s work produced a complete first benchmark ladder from static vulnerability indices to tabular ML and graph/neural models. The main conclusion is that raw SVI-style composite scores are not competitive as direct predictors or rankers of tract-month water/drainage 311 burden, whereas learned models are substantially stronger. A3 provides a strong tabular baseline, and G1/G1.5 shows that graph/neural models can improve high-risk ranking, particularly NDCG@100. However, because no-edge and random-placebo controls remain very strong, the current spatial graph topology should be interpreted as an initial graph construction rather than a validated final topology. The benchmark is now strong enough to justify moving forward, with the next priority being SoVI integration and a more advanced graph model with improved urban-dependency structure.
