# A3 Tabular Benchmark Report  
## Montréal 311 Water/Drainage Benchmark — Temporal and Spatial-Block Baselines

**Status:** frozen baseline layer before graph modeling  
**Benchmark target:** `water_drainage_count`  
**Unit of analysis:** census tract × month  
**Primary purpose:** establish the strongest non-graph tabular baselines before GraphSAGE/HGNN modeling

---

## 1. Why this README exists

This document freezes the A3 tabular benchmark layer before moving to graph models. It records the methodological choices, result interpretation, artifact locations, and graph-model comparison targets established by the A0–A3 baseline sequence.

The key reason for freezing A3 now is scientific discipline. Once graph models are introduced, the benchmark target should not keep moving. The graph model should be judged against a fixed non-graph baseline suite.

The A3 benchmark answers:

> Can ordinary non-graph tabular machine learning, using static tract features, SVI, spatial coordinates, calendar variables, lagged target history, and lagged reporting history, beat the strongest naive historical baselines?

The answer is split-dependent:

- **Temporal split:** A3 nearly matches A0 on count error, but A0 remains stronger for ranking.
- **Spatial-block split:** A3’s lagged-reporting HGB model becomes the dominant transferable tabular model, while static/SVI/target-history-heavy models degrade strongly.

This means future graph models must be compared against **A0 and A3**, not only against raw SVI or weak naive baselines.

---

## 2. Benchmark context

### 2.1 Prediction target

The prediction target is:

```text
water_drainage_count
```

This is the number of Montréal 311 water/drainage service requests assigned to a census tract-month.

Important interpretation:

```text
This target measures reported municipal water/drainage burden.
It is not an objective flood occurrence measure.
```

The target reflects both underlying physical/service conditions and reporting behavior. It may be influenced by drainage infrastructure, rainfall exposure, municipal service access, population/activity density, neighborhood reporting propensity, and persistent spatial burden.

### 2.2 Dataset unit

```text
census tract × month
```

### 2.3 Benchmark panel

The full panel used in A3 contains:

```text
540 tracts
53 months
28,620 tract-month rows
```

The temporal coverage is:

```text
2022-01 to 2026-05
```

---

## 3. Baseline sequence up to A3

The baseline suite is intentionally layered.

### A0 — naive temporal/exposure baselines

A0 establishes simple non-ML baselines, especially:

```text
A0_3_tract_train_mean
```

This model predicts future burden using the tract’s train-period historical mean. It is simple but extremely strong because water/drainage reporting burden is highly persistent at the tract level.

### A1 — raw SVI direct ranking

A1 tests whether raw SVI scores directly rank water/drainage burden.

Main conclusion:

```text
Raw SVI has positive but weak direct-ranking signal for water/drainage 311 burden.
```

A1 is useful as a vulnerability-index baseline, but it is not competitive with historical burden baselines.

### A2 — calibrated SVI

A2 calibrates SVI through simple supervised models and adds static/reporting controls.

Main conclusion:

```text
Calibrated SVI improves on raw SVI, but remains below the historical A0 tract baseline in strict forecasting.
Retrospective same-month non-water reporting is powerful, but not a forecasting feature.
```

### A3 — feature-parity tabular ML

A3 is the main non-graph ML floor. It asks whether a strong tabular model using rich non-graph features can beat A0 before graph models are introduced.

Main conclusion:

```text
A3 substantially raises the benchmark floor.
Temporal A3 nearly ties A0 on MAE.
Spatial-block A3 reveals that lagged reporting history is the most transferable tabular signal.
```

---

## 4. A3 design principles

### 4.1 Feature parity before graphs

A3 is based on the feature-parity principle:

```text
Before giving node features to a graph model, test whether ordinary tabular ML can already exploit them.
```

This prevents unfair graph claims. A graph model should not appear successful only because it receives richer features than the tabular baseline.

### 4.2 Prediction-setting separation

A3 separates three prediction settings.

#### `forecasting_v0`

Static/calendar/train-summary features only. No rolling target history from validation/test months.

Examples:

```text
SVI
population
land area
density
tract centroid coordinates
calendar variables
train-period tract summaries
```

#### `rolling_observed_history_v0`

Uses lagged and rolling observed history. This is valid for rolling monthly forecasting, where previous months are known when predicting the next month.

Examples:

```text
water_drainage_count lag 1
water_drainage_count rolling 3-month mean shifted by 1
non-water 311 lag 1
requests_total lag 1
lagged reporting rolling means shifted by 1
```

Important caveat:

```text
rolling_observed_history_v0 is not the same as predicting the entire future horizon from the train endpoint.
```

#### `retrospective_explanatory_v0`

Uses same-month non-water reporting controls.

Example:

```text
same-month total_311_count_non_water_drainage
```

This is explanatory, not forecasting. It helps understand reporting intensity but is not available before the target month.

### 4.3 Leakage control

A3 forbids same-month target-derived features in forecasting models, including:

```text
water_drainage_count
water_drainage_binary
share_water_drainage_requests
total_311_count_all
same-month target-containing aggregates
```

Rolling features are built with:

```text
shift first, then roll
```

This ensures the current month’s target does not leak into lagged/rolling features.

### 4.4 Validation-only model selection

A3 selects models using validation MAE only. Test metrics are reported after selection.

This matters because A3 evaluates many candidates:

```text
9 feature sets × 6 candidates = 54 models
```

Without validation-only selection, the process could become an informal test-set leaderboard.

---

## 5. A3 feature sets

A3 evaluates nine feature sets.

| Feature set | Prediction setting | Purpose |
|---|---|---|
| `A3_static_svi_calendar_forecasting` | `forecasting_v0` | Static SVI, tract demographics, spatial coordinates, calendar |
| `A3_target_history_forecasting` | `rolling_observed_history_v0` | Lagged/rolling target history and train-period tract summaries |
| `A3_target_history_svi_static_forecasting` | `rolling_observed_history_v0` | Target history plus SVI/static/spatial controls |
| `A3_lagged_reporting_forecasting` | `rolling_observed_history_v0` | Lagged general 311 reporting history without target history |
| `A3_target_history_lagged_reporting_forecasting` | `rolling_observed_history_v0` | Target history plus lagged reporting history |
| `A3_all_forecasting` | `rolling_observed_history_v0` | Main all-feature strict/rolling tabular ML baseline |
| `A3_all_forecasting_diagnostic_svi_expanded` | `rolling_observed_history_v0` | Diagnostic all-feature model with `svi_rank` and `svi_class` |
| `A3_reporting_retrospective` | `retrospective_explanatory_v0` | All forecasting features plus same-month non-water reporting |
| `A3_reporting_retrospective_diagnostic_svi_expanded` | `retrospective_explanatory_v0` | Diagnostic retrospective model with expanded SVI encodings |

Primary feature sets exclude diagnostic-only SVI encodings unless explicitly noted.

---

## 6. A3 model families

A3 evaluates three model families.

### 6.1 Ridge log-count

```text
ridge_log_count
```

Target transform:

```text
log1p(water_drainage_count)
```

Inverse transform:

```text
expm1(prediction), clipped at zero
```

Candidate alphas:

```text
0.1
1.0
10.0
```

Purpose:

```text
transparent linear-in-transformed-space baseline
stable under correlated features
useful coefficient and feature diagnostics
```

### 6.2 Poisson HistGradientBoosting

```text
hist_gradient_boosting_poisson
```

Purpose:

```text
nonlinear model aligned with count outcomes
can capture tabular interactions
primary nonlinear count model candidate
```

### 6.3 RandomForest log-count

```text
random_forest_log_count
```

Target transform:

```text
log1p(water_drainage_count)
```

Purpose:

```text
robust nonlinear diagnostic model
strong tabular baseline
useful feature-importance reference
```

---

## 7. Temporal A3 benchmark

### 7.1 Temporal split

The temporal split asks:

```text
Can we forecast future months for known tracts?
```

Partition sizes:

| Partition | Rows |
|---|---:|
| train | 19,440 |
| validation | 4,320 |
| test | 4,860 |

Temporal split interpretation:

```text
The same tracts appear in train, validation, and test.
The model can learn persistent tract-level burden structure.
```

### 7.2 Temporal A3 run summary

The full temporal A3 run completed with:

```text
Feature sets: 9
Candidates per feature set: 6
Models: 54
Metric rows: 3,240
Prediction rows: 1,545,480
Feature importance rows: 1,531
```

### 7.3 Main temporal result

The strongest primary temporal A3 strict/rolling model was:

```text
random_forest_log_count__A3_all_forecasting__rf_log_count_conservative
```

Approximate temporal test performance:

| Model | Test MAE | Test Spearman | Interpretation |
|---|---:|---:|---|
| `A0_3_tract_train_mean` | 2.489209 | 0.694235 | Strong historical tract baseline |
| `A3_all_forecasting` RF | 2.491364 | 0.679111 | Nearly ties A0 on MAE, weaker ranking |
| `A3_all_forecasting_diagnostic_svi_expanded` RF | 2.489811 | 0.679431 | Diagnostic-expanded version, almost identical to primary |

Temporal conclusion:

```text
A3 nearly matches A0 on count MAE, but A0 remains stronger for ranking.
```

### 7.4 Temporal interpretation

The temporal result shows that for known tracts, persistent historical burden is extremely informative. A nonlinear tabular model using target history, lagged reporting, static SVI, population, density, coordinates, and calendar features can nearly match A0 on MAE, but it does not clearly surpass A0’s ranking strength.

This implies:

```text
Historical tract-level burden is a very hard temporal forecasting baseline.
```

### 7.5 Temporal graph-model target

A temporal graph model should be judged against:

```text
A0_3_tract_train_mean
A3_all_forecasting RandomForest
```

A graph model is not convincing if it only beats raw SVI or calibrated SVI.

---

## 8. Spatial-block A3 benchmark

### 8.1 Spatial-block split

The spatial-block split asks:

```text
Can the model generalize to spatially held-out regions?
```

The current split is labeled:

```text
spatial_block_preliminary
```

Partition sizes:

| Partition | Rows |
|---|---:|
| train | 21,200 |
| validation | 5,353 |
| test | 2,067 |

Important caveat:

```text
This is a preliminary spatial robustness diagnostic, not yet the final definitive spatial generalization protocol.
```

### 8.2 Spatial-block run summary

The full spatial-block A3 run completed with:

```text
Feature sets: 9
Candidates per feature set: 6
Models: 54
Metric rows: 3,240
Prediction rows: 1,545,480
Feature importance rows: 1,531
```

The results were written to:

```text
urban_graph_benchmark/outputs/mtl_311_water_v0/baselines/A3_feature_parity_tabular_spatial_block/
```

The full paper-grade plotting run was written to:

```text
urban_graph_benchmark/outputs/mtl_311_water_v0/baselines/A3_feature_parity_tabular_spatial_block_with_plots/
```

### 8.3 Main spatial-block result

The validation-selected strict/rolling spatial-block model was:

```text
hist_gradient_boosting_poisson__A3_lagged_reporting_forecasting__hgb_poisson_02
```

Performance:

| Metric | Validation | Test |
|---|---:|---:|
| MAE | 2.707307 | 2.339043 |
| Spearman | 0.647544 | 0.762380 |
| NDCG@100 | 0.604147 | 0.784789 |
| Top-10% overlap | 0.503731 | 0.618357 |

Main spatial-block conclusion:

```text
Lagged general 311 reporting history is the strongest transferable A3 signal under the spatial-block split.
```

### 8.4 Spatial-block comparison to all-feature models

In spatial-block evaluation, the full all-feature models did not dominate.

Examples:

| Model | Test MAE | Test Spearman |
|---|---:|---:|
| `A3_lagged_reporting_forecasting` HGB | 2.339043 | 0.762380 |
| `A3_all_forecasting` HGB | 3.509423 | 0.545077 |
| `A3_all_forecasting` RF | 3.508593 | 0.500382 |

This is one of the most important A3 findings.

Spatial-block interpretation:

```text
Adding target history, static SVI, spatial coordinates, and all forecasting features hurts spatial-block transfer relative to the cleaner lagged-reporting HGB model.
```

### 8.5 Static/SVI features under spatial holdout

The static-only RandomForest model was much weaker spatially:

```text
A3_static_svi_calendar_forecasting RF
test MAE:      3.528722
test Spearman: 0.370744
```

This contrasts strongly with the temporal split, where static/SVI/spatial features were much more competitive.

Interpretation:

```text
Static SVI/spatial features are useful for known-tract temporal forecasting,
but they transfer poorly to held-out spatial regions.
```

### 8.6 Target-history features under spatial holdout

Target-history-heavy models also degraded in the spatial-block split.

Examples:

```text
A3_target_history_forecasting Ridge
test MAE:      3.827482
test Spearman: 0.034875

A3_target_history_forecasting HGB
test MAE:      3.873117
test Spearman: 0.183117

A3_target_history_forecasting RF
test MAE:      3.653418
test Spearman: 0.331914
```

Interpretation:

```text
Target history alone does not transfer well to held-out spatial regions under this preliminary split.
```

### 8.7 Retrospective spatial-block result

The selected retrospective spatial-block model was:

```text
hist_gradient_boosting_poisson__A3_reporting_retrospective_diagnostic_svi_expanded__hgb_poisson_01
```

Approximate performance:

```text
validation MAE: 3.260574
test MAE:       3.387679
test Spearman:  0.626208
```

This is weaker than the strict/rolling lagged-reporting HGB model. In the spatial-block setting, same-month retrospective reporting was informative but not dominant.

Important interpretation:

```text
Retrospective same-month reporting remains an explanatory diagnostic,
but it is not the strongest spatial-transfer model and should not be used as a forecasting claim.
```

---

## 9. Temporal vs spatial-block scientific interpretation

A3 reveals two different prediction regimes.

### 9.1 Temporal regime

Temporal question:

```text
Can we forecast future months for known tracts?
```

Dominant signal:

```text
persistent tract-level historical burden
```

Best references:

```text
A0_3_tract_train_mean
A3_all_forecasting RF
```

Scientific meaning:

```text
For known tracts, long-run tract burden is highly persistent.
```

### 9.2 Spatial-transfer regime

Spatial-block question:

```text
Can we generalize to spatially held-out regions?
```

Dominant signal:

```text
lagged general 311 reporting history
```

Best reference:

```text
A3_lagged_reporting_forecasting HGB
```

Scientific meaning:

```text
General reporting dynamics transfer better across space than target-history-heavy or static spatial/SVI models.
```

### 9.3 Combined A3 conclusion

```text
A3 is not a single story.
Temporal and spatial-block splits reveal different dominant signals.
```

This is exactly why both splits must be preserved before graph modeling.

---

## 10. Interpretation of A3 plots

A3-with-plots and A3-spatial-block-with-plots produce diagnostic figures.

### 10.1 Temporal plots

Temporal plots are stored in:

```text
urban_graph_benchmark/outputs/mtl_311_water_v0/baselines/A3_feature_parity_tabular_with_plots/
```

These include:

```text
validation-vs-test MAE scatter
selected-candidate MAE leaderboard
observed-vs-predicted calibration
monthly aggregate observed-vs-predicted curve
residual-by-decile plots
feature importance
PDP/ICE curves
2D response heatmaps
3D response surfaces
```

### 10.2 Spatial-block plots

Spatial-block plots are stored in:

```text
urban_graph_benchmark/outputs/mtl_311_water_v0/baselines/A3_feature_parity_tabular_spatial_block_with_plots/
```

The paper-grade spatial plotting script focuses on:

```text
validation-winning strict spatial-block model
A3_lagged_reporting_forecasting HGB
true PDP/ICE curves
true 2D and 3D response surfaces by feature perturbation
heatmaps centered on transferable lagged-reporting behavior
```

### 10.3 Plot interpretation warning

PDP/ICE curves, response surfaces, feature importances, and heatmaps are model-behavior diagnostics.

They show:

```text
how the fitted model responds to controlled feature perturbations
```

They do not show:

```text
causal effects
```

---

## 11. Frozen graph-model comparison targets

The graph stage must beat the frozen A3 layer.

### 11.1 Temporal graph comparison target

For temporal graph evaluation, compare against:

```text
A0_3_tract_train_mean
A3_all_forecasting RandomForest
```

Success criteria:

```text
The graph model should improve count metrics and/or ranking metrics over both A0 and A3.
```

Most convincing temporal success:

```text
lower MAE than A0_3 and A3_all_forecasting RF
higher Spearman/NDCG/top-10 overlap than A0_3
```

### 11.2 Spatial-block graph comparison target

For spatial-block graph evaluation, compare against:

```text
A3_lagged_reporting_forecasting HGB
```

Success criteria:

```text
The graph model must beat the lagged-reporting HGB spatial-block baseline,
especially on ranking metrics and high-burden identification.
```

Most convincing spatial success:

```text
lower MAE than A3_lagged_reporting_forecasting HGB
higher Spearman/NDCG/top-10 overlap than A3_lagged_reporting_forecasting HGB
```

### 11.3 What not to claim

Do not claim graph value only because the graph model beats:

```text
raw SVI
calibrated SVI
global mean
weak linear baselines
```

Those comparisons are insufficient after A3.

---

## 12. Artifact inventory

### 12.1 Core A3 temporal results

```text
urban_graph_benchmark/outputs/mtl_311_water_v0/baselines/A3_feature_parity_tabular/
```

### 12.2 Temporal backup

```text
urban_graph_benchmark/outputs/mtl_311_water_v0/baselines/A3_feature_parity_tabular_temporal_backup/
```

### 12.3 A3 temporal with plots

```text
urban_graph_benchmark/outputs/mtl_311_water_v0/baselines/A3_feature_parity_tabular_with_plots/
```

### 12.4 A3 spatial-block results

```text
urban_graph_benchmark/outputs/mtl_311_water_v0/baselines/A3_feature_parity_tabular_spatial_block/
```

### 12.5 A3 spatial-block with plots

```text
urban_graph_benchmark/outputs/mtl_311_water_v0/baselines/A3_feature_parity_tabular_spatial_block_with_plots/
```

### 12.6 Expected files in each A3 result folder

```text
metrics.csv
baseline_report.md
model_metadata.json
feature_set_audit.csv
feature_lineage_audit.csv
feature_importance.csv
model_audit.csv
model_selection_audit.csv
svi_score_audit.csv
svi_static_score_audit.csv
predictions_validation.parquet
predictions_test.parquet
```

Plot-enabled folders additionally include:

```text
plot_index.md
plots/
```

or equivalent plot index/report files depending on the plotting script version.

---

## 13. Reproduction commands

### 13.1 Temporal A3

```bash
PYTHONPATH=urban_graph_benchmark/src python -m ville_hgnn.baselines.a3_tabular_feature_parity \
  --config urban_graph_benchmark/configs/mtl_311_water_v0.1.yaml \
  --split-scheme temporal \
  --hgb-grid small
```

### 13.2 Temporal A3 with plots

```bash
PYTHONPATH=urban_graph_benchmark/src python -m ville_hgnn.baselines.a3_tabular_feature_parity_with_plots \
  --config urban_graph_benchmark/configs/mtl_311_water_v0.1.yaml \
  --split-scheme temporal \
  --hgb-grid small
```

### 13.3 Spatial-block A3

```bash
PYTHONPATH=urban_graph_benchmark/src python -m ville_hgnn.baselines.a3_tabular_feature_parity_spatial_block \
  --config urban_graph_benchmark/configs/mtl_311_water_v0.1.yaml \
  --split-scheme spatial_block \
  --hgb-grid small
```

### 13.4 Spatial-block A3 with plots

```bash
PYTHONPATH=urban_graph_benchmark/src python -m ville_hgnn.baselines.a3_tabular_feature_parity_spatial_block_with_plots \
  --config urban_graph_benchmark/configs/mtl_311_water_v0.1.yaml \
  --split-scheme spatial_block \
  --hgb-grid small
```

### 13.5 A0/A1/A2/A3 comparison

```bash
python urban_graph_benchmark/scripts/07_compare_a0_a1_a2_a3_results.py \
  --config urban_graph_benchmark/configs/mtl_311_water_v0.1.yaml
```

---

## 14. Limitations

### 14.1 Reported 311 burden is not objective flooding

The benchmark target is reported water/drainage 311 burden. It is not direct flood damage, hydrological hazard, or infrastructure failure.

### 14.2 Temporal split does not test unseen-tract generalization

Temporal evaluation is operationally useful for known tracts, but it does not test generalization to new spatial areas.

### 14.3 Spatial-block split is preliminary

The spatial-block split is currently labeled preliminary. It should be treated as a spatial robustness diagnostic, not final proof of geographic generalization.

### 14.4 Rolling observed-history assumptions

Rolling-history features assume that previous months are observed before later months are predicted. This is valid for rolling monthly deployment but not for one-shot multi-month forecasting from the training endpoint.

### 14.5 Retrospective features are not forecasting features

Same-month non-water reporting controls are explanatory and should not be used as future-month forecasting evidence.

### 14.6 Feature importance is predictive, not causal

Ridge coefficients, RandomForest importances, HGB importances, PDPs, ICE curves, and response surfaces describe fitted model behavior. They do not identify causal mechanisms.

---

## 15. Final frozen A3 conclusions

### 15.1 Temporal conclusion

```text
For known tracts over future months, A3 full tabular ML nearly matches A0 on MAE, but A0 remains the stronger ranking baseline.
```

### 15.2 Spatial-block conclusion

```text
For held-out spatial regions, lagged general 311 reporting history is the strongest transferable A3 signal.
```

### 15.3 Graph handoff conclusion

```text
The graph model must beat the frozen A3 benchmark layer.
Temporal graph models must beat A0_3 and A3_all_forecasting RF.
Spatial-block graph models must beat A3_lagged_reporting_forecasting HGB.
```

---

## 16. Immediate next step: first graph baseline

The next stage should not begin with the full HGNN. It should begin with a simpler graph baseline.

Recommended next model:

```text
G0/G1 tract graph baseline
```

Suggested setup:

```text
Nodes:
  census tracts

Edges:
  tract adjacency / contiguity
  and/or k-nearest-neighbor centroid graph

Node-month features:
  same feature families as A3_all_forecasting
  with no retrospective leakage

Target:
  water_drainage_count

Splits:
  temporal
  spatial_block

Models:
  GraphSAGE
  GCN
  possibly MLP-on-node-features as internal control
```

First graph research question:

```text
Does spatial message passing improve beyond A3 tabular ML and A0 tract history?
```

If the answer is no, the graph architecture should not be expanded yet. If the answer is yes, the project can move toward richer heterogeneous graph modeling.

---

## 17. Frozen benchmark rule

From this point forward:

```text
A0–A3 are frozen as the non-graph benchmark layer.
Graph models are evaluated against this frozen layer.
Any new baseline improvements should be versioned separately and not silently replace these results.
```
