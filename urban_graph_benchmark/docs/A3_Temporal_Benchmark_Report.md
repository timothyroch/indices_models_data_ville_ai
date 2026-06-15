# A3 Feature-Parity Tabular Baselines for the Montréal 311 Water/Drainage Benchmark

## 1. Executive Summary

This report documents the design, implementation, and interpretation of the **A3 feature-parity tabular baseline suite** for the Montréal 311 water/drainage benchmark. A3 was introduced after A0, A1, and A2 to establish a strong non-graph machine learning baseline before developing GraphSAGE or HGNN models.

The central question answered by A3 is:

```text
Can ordinary non-graph tabular machine learning, using tract-level static features, SVI, spatial coordinates, calendar variables, lagged target history, and lagged reporting signals, beat the best naive historical baseline?
```

The answer from the full A3 run is nuanced:

```text
A3 nearly matches A0_3 on count prediction, but it does not clearly beat A0_3 as a ranking model.
```

The strongest A3 strict/rolling forecasting model selected by validation MAE was:

```text
random_forest_log_count__A3_all_forecasting_diagnostic_svi_expanded__rf_log_count_conservative
```

with:

```text
test MAE:      2.489811
test Spearman: 0.679431
```

The primary non-diagnostic equivalent was:

```text
random_forest_log_count__A3_all_forecasting__rf_log_count_conservative
```

with:

```text
test MAE:      2.491364
test Spearman: 0.679111
```

For comparison, the strongest A0 model, `A0_3_tract_train_mean`, achieved approximately:

```text
test MAE:       2.489209
test Spearman:  0.694235
NDCG@100:       0.748363
Top-10% overlap: 0.483539
```

Thus, the best A3 model is essentially tied with A0 on MAE but remains weaker on rank-based performance. This is scientifically important: a simple tract-history baseline remains extremely difficult to beat.

A3 also showed that same-month non-water 311 reporting activity is a strong retrospective explanatory signal. The best A3 retrospective model achieved:

```text
test MAE:      2.431769
test Spearman: 0.685681
```

However, this model uses same-month non-water 311 information and therefore must not be described as a forecasting model. It is an explanatory model of same-month reporting intensity.

The major conclusion is:

```text
A3 establishes a serious non-graph ML floor. Future graph models must beat both A0_3 and A3 RandomForest, not merely A1/A2 SVI baselines.
```

---

## 2. Context and Motivation

The broader project aims to construct a publishable urban vulnerability and resilience benchmark using Montréal 311 water/drainage requests. The benchmark is organized around census tract-month prediction and ranking of reported water/drainage service burden.

Before A3, the baseline sequence was:

```text
A0: naive temporal and exposure baselines
A1: raw SVI direct-ranking baseline
A2: calibrated SVI regression-style baseline
A3: feature-parity tabular machine learning baseline
```

A0 showed that historical tract-level burden is a very strong predictor. A1 showed that raw SVI has only weak direct-ranking power for this target. A2 showed that calibrated SVI and static controls improve on raw SVI but still remain weaker than A0. A2 also showed that same-month non-water 311 activity is powerful retrospectively, but not valid as a strict forecasting feature.

Therefore, A3 was necessary before graph modeling. Without A3, a graph model could appear strong simply because it beats weak SVI baselines. A3 forces a more rigorous comparison:

```text
Does graph structure improve over a strong non-graph model with similar node/month features?
```

This is the core feature-parity principle.

---

## 3. Prediction Task

### 3.1 Target

The prediction target is:

```text
water_drainage_count
```

This is the number of reported Montréal 311 water/drainage requests assigned to a census tract-month.

The target is a **reported municipal service burden**, not an objective flood occurrence measure. It is affected by physical infrastructure, weather, drainage conditions, reporting behavior, municipal service access, neighborhood activity, population density, and vulnerability.

### 3.2 Unit of analysis

```text
census tract × month
```

### 3.3 Temporal coverage

```text
2022-01 to 2026-05
```

### 3.4 Spatial coverage

The dataset contains 540 census tracts after assigning Montréal 311 grid25m-month observations to census tracts.

### 3.5 Primary split

The primary split used for A3 is temporal:

```text
train:      19,440 rows
validation: 4,320 rows
test:       4,860 rows
```

These row counts are confirmed in the A3 report.

---

## 4. Why A3 Was Needed

A3 was designed to fill the methodological gap between SVI-based baselines and future graph models.

A1 and A2 answer:

```text
How predictive is SVI, either directly or after calibration?
```

A3 answers:

```text
How predictive is a strong non-graph ML model using the same broad feature families that a graph model might later receive?
```

This matters because future GraphSAGE/HGNN models should not merely beat:

```text
raw SVI
calibrated SVI
global mean
```

They should beat:

```text
tract-history baselines
non-graph tabular ML
spatial-coordinate tabular models
lagged reporting-history tabular models
```

A3 therefore acts as a **modeling floor**. If a graph model cannot beat A3, then the evidence for graph structure is weak. If a graph model beats A3, the result becomes much more convincing.

---

## 5. Methodological Design Principles

### 5.1 Feature parity

The first principle is feature parity:

```text
Any feature later given to a graph model should first be tested in a flat non-graph model.
```

This prevents an unfair comparison where the graph model receives rich node features while the tabular baseline receives only weak static predictors.

### 5.2 Strict separation of prediction settings

A3 separates three prediction settings:

```text
forecasting_v0
rolling_observed_history_v0
retrospective_explanatory_v0
```

This distinction is critical.

#### forecasting_v0

These models use static and calendar features only, plus train-period summaries where applicable. They do not use rolling target history from validation/test months.

Examples:

```text
SVI
population
density
land area
spatial coordinates
calendar features
train-period historical tract summaries
```

#### rolling_observed_history_v0

These models use lagged and rolling observed history. For example, when predicting a given month, the model may use previous months’ observed water/drainage counts or previous months’ non-water reporting counts.

This is valid for a rolling monthly forecasting protocol:

```text
At each month, predict the next month using all observations available up to the previous month.
```

It is not equivalent to forecasting the whole test horizon from the end of the training period.

#### retrospective_explanatory_v0

These models use same-month non-water 311 reporting exposure:

```text
total_311_count_non_water_drainage
```

This variable is highly informative but is not available before the target month. Therefore, these models are retrospective or explanatory, not forecasting models.

### 5.3 Leakage prevention

The A3 design explicitly forbids target-derived same-month variables in forecasting models, including:

```text
water_drainage_count
water_drainage_binary
water_drainage_requests
share_water_drainage_requests
total_311_count_all
```

The variable `total_311_count_all` is forbidden because it includes the target category. Lagged versions of reporting variables are allowed, but same-month target-containing variables are not.

Rolling features are constructed with the rule:

```text
shift first, then roll
```

This prevents the current month’s target from entering rolling means or rolling sums.

### 5.4 Validation-only model selection

A3 uses validation MAE to select models. Test metrics are reported only after selection and are not used to choose the model.

This is important because A3 evaluates 54 models. Without a validation-only selection rule, there would be a risk of informal test-set overfitting.

---

## 6. Feature Engineering

A3 uses nine feature sets. These are organized to answer specific scientific questions.

### 6.1 Static SVI/calendar forecasting

Feature set:

```text
A3_static_svi_calendar_forecasting
```

Prediction setting:

```text
forecasting_v0
```

Number of features:

```text
23
```

Feature families:

```text
calendar
static
static_spatial
svi_primary
```

This feature set includes:

```text
svi_percentile
svi_score_raw
population_total_2021
land_area_km2
population_density
tract centroid x/y
tract centroid lon/lat
month indicators
month sine/cosine
period index
```

Purpose:

```text
Test how much can be predicted from static vulnerability, demography, geography, and calendar features, without rolling target history.
```

This feature set is scientifically important because it approximates what a static vulnerability model can do once spatial and demographic controls are included.

### 6.2 Target-history forecasting

Feature set:

```text
A3_target_history_forecasting
```

Prediction setting:

```text
rolling_observed_history_v0
```

Number of features:

```text
30
```

Feature families:

```text
calendar
target_history
target_train_summary
```

This includes:

```text
water_drainage_count lag 1
water_drainage_count lag 2
water_drainage_count lag 3
water_drainage_count lag 6
water_drainage_count lag 12
rolling means shifted by one month
rolling sums shifted by one month
expanding mean shifted by one month
train-period tract mean
train-period tract median
train-period tract p90
train-period positive rate
calendar features
```

Purpose:

```text
Test whether past water/drainage burden alone is enough to approach or beat A0.
```

This is one of the most important A3 ablations because A0 already demonstrated strong tract persistence.

### 6.3 Target history + SVI/static forecasting

Feature set:

```text
A3_target_history_svi_static_forecasting
```

Prediction setting:

```text
rolling_observed_history_v0
```

Number of features:

```text
39
```

Feature families:

```text
calendar
static
static_spatial
svi_primary
target_history
target_train_summary
```

Purpose:

```text
Test whether SVI, population, density, land area, and spatial coordinates add predictive value beyond target history.
```

This feature set is important for assessing whether SVI adds value once historical burden is known.

### 6.4 Lagged reporting forecasting

Feature set:

```text
A3_lagged_reporting_forecasting
```

Prediction setting:

```text
rolling_observed_history_v0
```

Number of features:

```text
27
```

Feature families:

```text
calendar
lagged_reporting
lagged_requests_total
```

This uses:

```text
lagged non-water 311 count
rolling lagged non-water 311 count
lagged requests_total
rolling lagged requests_total
calendar
```

Purpose:

```text
Test whether past general reporting behavior predicts future water/drainage burden without using water/drainage target history.
```

This is a key diagnostic of reporting-intensity persistence.

### 6.5 Target history + lagged reporting forecasting

Feature set:

```text
A3_target_history_lagged_reporting_forecasting
```

Prediction setting:

```text
rolling_observed_history_v0
```

Number of features:

```text
43
```

Purpose:

```text
Test whether past general reporting history adds value beyond past water/drainage history.
```

### 6.6 All forecasting features

Feature set:

```text
A3_all_forecasting
```

Prediction setting:

```text
rolling_observed_history_v0
```

Number of features:

```text
52
```

Feature families:

```text
calendar
lagged_reporting
lagged_requests_total
static
static_spatial
svi_primary
target_history
target_train_summary
```

This is the main primary A3 feature set. It includes all strict/rolling forecasting-safe predictors, but excludes diagnostic SVI rank/class encodings and excludes same-month retrospective reporting controls.

Purpose:

```text
Define the main non-graph ML benchmark.
```

### 6.7 Diagnostic SVI-expanded all forecasting

Feature set:

```text
A3_all_forecasting_diagnostic_svi_expanded
```

Prediction setting:

```text
rolling_observed_history_v0
```

Number of features:

```text
54
```

This adds:

```text
svi_rank
svi_class
```

Purpose:

```text
Check whether diagnostic SVI encodings materially change performance.
```

This should not be treated as the primary model because `svi_rank` and `svi_class` are diagnostic encodings. However, it is useful for robustness.

### 6.8 Retrospective reporting

Feature set:

```text
A3_reporting_retrospective
```

Prediction setting:

```text
retrospective_explanatory_v0
```

Number of features:

```text
53
```

This is `A3_all_forecasting` plus:

```text
same-month total_311_count_non_water_drainage
```

Purpose:

```text
Measure the explanatory value of same-month non-water 311 reporting exposure.
```

This model should not be interpreted as a future-month forecasting model.

### 6.9 Retrospective diagnostic SVI-expanded

Feature set:

```text
A3_reporting_retrospective_diagnostic_svi_expanded
```

Prediction setting:

```text
retrospective_explanatory_v0
```

Number of features:

```text
55
```

This adds diagnostic SVI encodings to the retrospective model.

Purpose:

```text
Robustness check for retrospective modeling.
```

---

## 7. Model Families

A3 evaluates three model families.

### 7.1 Ridge log-count model

Model family:

```text
ridge_log_count
```

Target transformation:

```text
log1p(water_drainage_count)
```

Inverse transformation:

```text
expm1(prediction), clipped at zero
```

Candidate alphas:

```text
0.1
1.0
10.0
```

Ridge is included because it is transparent, stable, and interpretable. It extends the modeling logic of A2 while allowing richer A3 feature sets.

Advantages:

```text
simple
fast
interpretable
stable under correlated features
useful coefficient diagnostics
```

Limitations:

```text
linear in transformed space
limited nonlinear interaction modeling
may underfit complex spatial/reporting relationships
```

### 7.2 Poisson HistGradientBoosting

Model family:

```text
hist_gradient_boosting_poisson
```

This model uses a Poisson loss and is naturally aligned with nonnegative count outcomes.

Advantages:

```text
nonlinear
can model feature interactions
well-suited to count outcomes through Poisson loss
fast for tabular data
```

Limitations:

```text
less transparent than ridge
sensitive to hyperparameters
may not dominate if the signal is mostly stable tract-level persistence
```

In the full A3 run, HGB was useful but did not become the best model.

### 7.3 RandomForest log-count model

Model family:

```text
random_forest_log_count
```

Target transformation:

```text
log1p(water_drainage_count)
```

The RandomForest was configured conservatively as a diagnostic nonlinear model.

Advantages:

```text
nonlinear
robust
captures interactions
useful feature-importance diagnostics
strong empirical performance in this run
```

Limitations:

```text
less interpretable than ridge
feature importances are predictive, not causal
can memorize structure if unconstrained
```

In the full A3 run, RandomForest was the best-performing model family by validation-selected MAE.

---

## 8. Evaluation Metrics

A3 uses the same metric schema as A0/A1/A2.

### 8.1 Count metrics

```text
count__mae
count__rmse
count__mean_poisson_deviance
```

MAE is the primary model-selection metric.

### 8.2 Ranking metrics

```text
ranking__spearman_corr
ranking__ndcg_at_100
ranking__top_10pct_overlap_rate
```

These metrics matter because municipal decision support may care more about identifying high-burden tracts/months than predicting the exact count everywhere.

### 8.3 Validation vs test discipline

Validation metrics are used for model selection. Test metrics are reported after selection. This distinction is essential because A3 includes many models and feature sets.

---

## 9. Run Summary

The full A3 run completed successfully.

```text
Feature sets: 9
Candidates per feature set: 6
Models: 54
Metric rows: 3240
Prediction rows: 1545480
Feature importance rows: 1531
```

The model count is coherent:

```text
9 feature sets × 6 candidates per feature set = 54 models
```

The prediction row count is coherent:

```text
28,620 panel rows × 54 models = 1,545,480 prediction rows
```

This confirms that the run executed across the full candidate suite.

---

## 10. Main Results

### 10.1 Best strict/rolling A3 model

The validation-selected strict/rolling model was:

```text
random_forest_log_count__A3_all_forecasting_diagnostic_svi_expanded__rf_log_count_conservative
```

Performance:

```text
validation MAE: 2.390825
test MAE:       2.489811
test Spearman:  0.679431
```

This model uses diagnostic SVI-expanded features, so the primary model without diagnostic SVI is more appropriate for the main paper result.

### 10.2 Best primary strict/rolling A3 model

The primary equivalent is:

```text
random_forest_log_count__A3_all_forecasting__rf_log_count_conservative
```

Performance:

```text
validation MAE: 2.394284
test MAE:       2.491364
test Spearman:  0.679111
```

This is the main A3 result to emphasize.

### 10.3 Comparison to A0

A0_3 achieved approximately:

```text
test MAE:       2.489209
test Spearman:  0.694235
NDCG@100:       0.748363
Top-10% overlap: 0.483539
```

A3 primary RF all-forecasting achieved:

```text
test MAE:      2.491364
test Spearman: 0.679111
```

The MAE difference is tiny:

```text
A3 primary RF all forecasting MAE - A0_3 MAE ≈ 0.002155
```

This is effectively a tie on MAE, but A0 remains stronger in ranking.

Interpretation:

```text
A3 reaches the A0 level in count prediction but does not clearly improve ranking performance.
```

### 10.4 Static SVI/calendar forecasting result

The best static-only model was:

```text
random_forest_log_count__A3_static_svi_calendar_forecasting__rf_log_count_conservative
```

Performance:

```text
validation MAE: 2.443337
test MAE:       2.512990
test Spearman:  0.678924
```

This is surprisingly strong because it does not use rolling target history. It uses static SVI, population, density, land area, coordinates, and calendar features.

Interpretation:

```text
Static spatial/demographic structure explains a large share of future reported water/drainage burden.
```

This is a very important finding. It means that a non-graph tabular model with spatial coordinates can already learn much of the spatial burden surface.

### 10.5 Target-history result

The target-history RandomForest model achieved:

```text
test MAE:      2.542770
test Spearman: 0.666642
```

The Ridge target-history model achieved:

```text
test MAE:      2.570638
test Spearman: 0.676583
```

Interpretation:

```text
Target history alone is strong, but static/spatial information and reporting features help improve count prediction.
```

### 10.6 Lagged reporting result

The lagged-reporting-only models are weaker on count prediction but informative for ranking. Ridge lagged reporting achieved:

```text
test MAE:      3.073577
test Spearman: 0.676840
```

RandomForest lagged reporting achieved:

```text
test MAE:      2.598438
test Spearman: 0.646495
```

Interpretation:

```text
Lagged reporting behavior carries meaningful rank information, but it does not fully calibrate counts without target history or static burden structure.
```

### 10.7 Retrospective result

The best retrospective model was:

```text
random_forest_log_count__A3_reporting_retrospective_diagnostic_svi_expanded__rf_log_count_conservative
```

Performance:

```text
validation MAE: 2.387977
test MAE:       2.431769
test Spearman:  0.685681
```

The primary retrospective model was:

```text
random_forest_log_count__A3_reporting_retrospective__rf_log_count_conservative
```

Performance:

```text
validation MAE: 2.389727
test MAE:       2.435450
test Spearman:  0.685485
```

This beats A0 on MAE but uses same-month non-water 311 reporting exposure.

Interpretation:

```text
Same-month non-water 311 reporting is a strong explanatory variable for same-month water/drainage reports, but it cannot be used for strict forecasting.
```

---

## 11. Interpretation of the Results

### 11.1 A0 remains a hard baseline

The most important conclusion is that A0_3 remains extremely competitive. Even with nonlinear tabular ML, static SVI, coordinates, lagged target history, lagged reporting history, and calendar features, A3 does not clearly beat A0_3.

This implies that the water/drainage 311 burden is highly persistent at the tract level. A simple train-period tract mean captures a large share of the signal.

### 11.2 A3 improves the credibility of the benchmark

Even though A3 does not clearly beat A0, it greatly improves the benchmark.

Without A3, future graph models could be compared mainly against SVI and naive baselines. With A3, graph models must beat a much stronger non-graph ML reference.

This makes the benchmark much more publishable.

### 11.3 Static spatial structure is highly informative

The static SVI/calendar RandomForest model is one of the most surprising results. It reaches:

```text
test MAE:      2.512990
test Spearman: 0.678924
```

This suggests that static tract attributes and spatial coordinates encode much of the long-run spatial burden pattern.

This has two implications:

```text
1. Graph models must prove they add value beyond spatial coordinates.
2. The observed 311 burden is strongly structured by persistent tract-level characteristics.
```

### 11.4 SVI alone is weak, but SVI within a richer static model is more useful

A1 showed raw SVI is weak as a direct ranking signal. A2 showed calibrated SVI improves but remains below A0. A3 shows that static features including SVI, demographics, and coordinates can become much stronger inside nonlinear models.

However, the strong A3 static model should not be interpreted as “SVI works strongly.” It is a combined static spatial/demographic/SVI model. The contribution of SVI specifically requires feature importance, ablation, or PDP analysis.

### 11.5 Retrospective reporting controls are powerful

The retrospective A3 result confirms a pattern already seen in A2: same-month non-water 311 reporting activity is highly informative.

This may capture:

```text
general neighborhood reporting propensity
municipal activity
population/activity density
service access
seasonal reporting shocks
local administrative patterns
```

But it is not available before the month being predicted. Therefore, it should be framed as a reporting-intensity explanation, not a forecasting model.

---

## 12. Model-Family Interpretation

### 12.1 Ridge

Ridge remained useful but was not the best model once nonlinear options were installed.

The best Ridge all-forecasting model achieved approximately:

```text
test MAE:      2.538869
test Spearman: 0.679863
```

Ridge is transparent and useful for coefficients, but it underperformed RandomForest in count prediction.

### 12.2 Poisson HistGradientBoosting

Poisson HGB ran successfully but did not dominate.

For `A3_all_forecasting`, HGB achieved:

```text
test MAE:      2.512332
test Spearman: 0.677990
```

This was good, but RandomForest was slightly better on MAE.

Possible reasons:

```text
the signal may be dominated by stable tract-level burden rather than complex nonlinear temporal interactions
the HGB grid was intentionally small
the conservative RF configuration may better capture persistent spatial heterogeneity
```

### 12.3 RandomForest

RandomForest was the strongest A3 model family in this run.

It performed especially well in:

```text
A3_static_svi_calendar_forecasting
A3_all_forecasting
A3_reporting_retrospective
```

This suggests that nonlinear interactions among spatial coordinates, static tract attributes, and history features matter.

However, RandomForest should be interpreted carefully. Its feature importances are predictive, not causal. It may also be very strong in temporal splits because the same tracts appear in train and test.

---

## 13. Relationship to A0, A1, and A2

### 13.1 A1 raw SVI

A1 showed that raw SVI percentile and raw SVI score have weak direct-ranking performance for water/drainage 311 burden.

A3 does not contradict this. A3 shows that SVI may become more useful when embedded in a richer nonlinear static/spatial model, but SVI alone is not the main predictive driver.

### 13.2 A2 calibrated SVI

A2 showed that calibrated SVI + static controls improves over raw SVI but remains weaker than A0.

A3 confirms this broader story. Richer static/spatial and nonlinear modeling improves further, but the strongest historical baseline remains hard to beat.

### 13.3 A0 tract history

A0_3 remains the central reference point. A3 nearly matches it on MAE, but A0 remains stronger in ranking.

This suggests:

```text
the benchmark is difficult
historical tract burden is a strong baseline
graph models must provide real additional value
```

---

## 14. Implications for Future Graph Models

A3 changes the standard for evaluating graph models.

A future graph model must be compared against:

```text
A0_3_tract_train_mean
A3_all_forecasting RandomForest
A3_static_svi_calendar RandomForest
A3_reporting_retrospective only as explanatory reference
```

A graph model should not be considered successful merely because it beats:

```text
A1 raw SVI
A2 calibrated SVI
global mean
```

A strong graph result would need to show improvement on:

```text
MAE
RMSE
Poisson deviance
Spearman rank correlation
NDCG@100
Top-10% overlap
```

The most convincing outcome would be:

```text
Graph model beats A0_3 and A3_all_forecasting under the same temporal split, without retrospective same-month reporting leakage.
```

A weaker but still interesting outcome would be:

```text
Graph model improves ranking metrics while only matching A3 on MAE.
```

This could still matter for municipal triage if high-burden identification is the operational priority.

---

## 15. Limitations

### 15.1 The target is reported burden, not objective hazard

The target measures reported water/drainage 311 requests. It is not direct flood exposure or infrastructure failure. It reflects both underlying events and reporting behavior.

### 15.2 Temporal split uses same tracts across train and test

The temporal split evaluates future months for known tracts. This is appropriate for operational forecasting but does not test generalization to unseen tracts.

Spatial-block evaluation should be performed later.

### 15.3 Rolling observed-history models are not horizon-wide forecasts

Models using lagged target history are valid in a rolling monthly setting, but they assume that previous validation/test months are observed by the time later months are predicted.

They should not be described as forecasting the entire future horizon from the train endpoint.

### 15.4 Retrospective models are not forecasting models

Same-month non-water 311 reporting controls are useful diagnostically, but they are not available before the target month. These models should not be used for forecasting claims.

### 15.5 Feature importance is not causal

RandomForest feature importance, Ridge coefficients, PDPs, and response surfaces describe model behavior. They do not identify causal effects.

### 15.6 HGB grid was intentionally small

The HGB model may improve with a broader hyperparameter search. However, the current grid was intentionally limited to avoid leaderboard-style validation overfitting.

---

## 16. Considered Alternatives

### 16.1 Larger hyperparameter search

A larger grid could have been used for HGB and RF. This was not done initially because A3 is intended as a disciplined benchmark, not an exhaustive leaderboard competition.

A future extension could run:

```text
larger HGB grid
more RF depth/min-leaf variants
Poisson GLM
ElasticNet
LightGBM or XGBoost
```

But these should remain validation-selected and clearly documented.

### 16.2 Zone fixed effects

Raw tract ID one-hot encoding was intentionally avoided in the primary A3 design. It could improve temporal-split performance but risks turning the model into a tract memorization engine.

Instead, A3 uses train-period tract summaries and spatial coordinates. This is more interpretable and more relevant for future graph comparison.

### 16.3 Same-month total 311 requests

Same-month `total_311_count_all` was not used because it contains the water/drainage target. This would leak the target.

### 16.4 Same-month non-water 311 requests

Same-month non-water 311 was used only in retrospective models. This was considered useful because it quantifies the explanatory value of same-month reporting intensity while preserving the forecasting/explanatory distinction.

### 16.5 Autograd Poisson regression

A fully custom Poisson regression with gradient descent could be implemented, as in educational Poisson regression/autograd examples. However, A3 already includes a transparent Ridge log-count model and Poisson HistGradientBoosting. A custom autograd Poisson model may be useful pedagogically, but it is not necessary for the benchmark unless we want a fully controlled Poisson GLM implementation.

---

## 17. Recommended Reporting Language

A good scientific summary sentence is:

```text
A3 feature-parity tabular baselines substantially raise the non-graph benchmark floor. A conservative RandomForest model using static, spatial, SVI, lagged target, and lagged reporting features nearly matches the strongest A0 tract-history baseline in MAE, but does not clearly outperform it on ranking metrics. Retrospective same-month non-water reporting controls further improve count accuracy, but these models are explanatory rather than forecasting baselines.
```

A shorter version:

```text
A3 nearly ties A0 on count prediction, but A0 remains the stronger ranking baseline.
```

And the graph-model implication:

```text
Future graph models must beat both A0_3 and A3 RandomForest under the strict/rolling forecasting setting to demonstrate graph-specific value.
```

---

## 18. Recommended Next Steps

### 18.1 Generate the A0/A1/A2/A3 comparison report

The next comparison script should consolidate:

```text
A0
A1
A2
A3
```

and report canonical model rows.

### 18.2 Generate A3 interpretation plots

The augmented script `a3_tabular_feature_parity_with_plots.py` should be run after the manual API fix. It should produce:

```text
calibration plots
monthly observed-vs-predicted curves
residual-by-decile plots
feature importance plots
1D partial dependence / ICE-style curves
2D response heatmaps
3D response surfaces
```

These are useful for moving from benchmark comparison to scientific interpretation.

### 18.3 Run spatial-block robustness

After temporal results are stabilized, A3 should be rerun on the spatial-block split.

This will answer:

```text
Do static/spatial/tabular models generalize to held-out spatial regions?
```

This is important before making claims about spatial robustness.

### 18.4 Prepare graph baseline specification

Only after A3 is fully summarized should the graph model be specified. The graph model should be compared against:

```text
A0_3
A3_all_forecasting RandomForest
A3_static_svi_calendar RandomForest
```

### 18.5 Decide whether ranking or count is the primary operational metric

The current results show that MAE and ranking metrics can tell different stories. A0 is stronger in ranking, while A3 nearly ties count MAE. The project should define whether the priority is:

```text
accurate counts
high-burden tract/month identification
operational triage
retrospective explanation
```

This will influence how graph models are judged.

---

## 19. Final Conclusion

A3 is a successful and necessary stage of the benchmark. It establishes that non-graph tabular ML can nearly match the strongest naive historical baseline, but it does not clearly surpass it. The strongest strict/rolling A3 models, especially conservative RandomForest models using all forecasting-safe features, reach A0-level MAE but remain weaker than A0 on rank-based metrics.

This is not a failure. It is a valuable scientific result. It shows that:

```text
1. Montréal water/drainage 311 burden is highly persistent at the tract level.
2. Static spatial/demographic structure is surprisingly informative.
3. SVI alone is weak, but richer static/spatial models are much stronger.
4. Same-month non-water 311 reporting is powerful retrospectively.
5. Future graph models now face a serious non-graph ML benchmark.
```
