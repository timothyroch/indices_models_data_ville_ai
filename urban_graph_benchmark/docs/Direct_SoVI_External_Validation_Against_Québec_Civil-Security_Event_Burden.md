# B1 — Direct SoVI External Validation Against Québec Civil-Security Event Burden

## 1. Purpose of this benchmark

This benchmark evaluates whether a static SoVI-like social vulnerability score at the Québec census-division level aligns with observed civil-security event burden.

The central question is:

> Do Québec census divisions with higher SoVI-like vulnerability scores tend to rank higher in observed civil-security event burden?

This benchmark is intentionally limited in scope. It is not a supervised machine-learning model, not a graph model, and not a calibrated predictor. It is a direct external-validation test of a classical vulnerability index against observed civil-security event outcomes.

The benchmark is useful because it establishes the empirical value and limitations of a static social vulnerability index before moving to richer models that include temporal history, hazard exposure, spatial context, or graph structure.

---

## 2. Inputs and benchmark configuration

### Input table

```text
data/external/quebec_civil_security_events/processed/
cd_civil_security_sovi_validation_targets_cumulative.parquet
```

### Output directory

```text
urban_graph_benchmark/outputs/mtl_311_water_v0/benchmarks/
sovi_civil_security_validation
```

### Spatial unit

The benchmark is evaluated at the Québec census-division level.

```text
n = 98 census divisions
```

### SoVI score

```text
score_normalized_0_1
```

The score is interpreted as:

```text
higher score = more vulnerable
```

### Hardened uncertainty configuration

The final run included:

```text
10,000 permutation-null runs
2,000 bootstrap samples
```

The additional hardening outputs were:

```text
sovi_civil_security_validation_permutation_null.csv
sovi_civil_security_validation_bootstrap_ci.csv
sovi_civil_security_validation_topk_diagnostics.csv
sovi_civil_security_validation_metric_audit.csv
```

---

## 3. Target definitions

The benchmark evaluates SoVI against multiple cumulative civil-security event targets.

The primary target is:

```text
B1a = 2021–2025 all civil-security events
```

Additional targets test robustness across time windows, localization precision, hazard families, and severity thresholds.

| Target ID                | Target label                             | Role                                      |
| ------------------------ | ---------------------------------------- | ----------------------------------------- |
| B1a                      | 2021–2025 all events                     | Primary descriptive validation target     |
| B1b                      | 2022–2025 all events                     | Cleaner forward-looking robustness target |
| B1c                      | 2021–2025 precise or very precise events | Localization-precision robustness         |
| B1d                      | 2021–2025 very precise events            | Strict localization-precision robustness  |
| B1e                      | 2021–2025 flood/water events             | Hazard-specific robustness                |
| B1f_land_ground          | 2021–2025 land/ground events             | Hazard-specific robustness                |
| B1f_weather_climate      | 2021–2025 weather/climate events         | Hazard-specific robustness                |
| B1f_infrastructure       | 2021–2025 infrastructure events          | Hazard-specific robustness                |
| B1f_wildfire             | 2021–2025 wildfire events                | Sparse hazard-specific robustness         |
| B1g_moderate_or_worse    | 2021–2025 moderate-or-worse events       | Severity-threshold robustness             |
| B1g_important_or_extreme | 2021–2025 important-or-extreme events    | Strict severity-threshold robustness      |

---

## 4. Metrics

The benchmark uses ranking-oriented metrics because SoVI is an index, not a calibrated event-count predictor.

The headline metrics are:

```text
Spearman rank correlation
Kendall rank correlation
NDCG@10
NDCG@25
Top-10 overlap rate
Top-25 overlap rate
```

The hardening run additionally reports:

```text
bootstrap confidence intervals
permutation-null means and standard deviations
empirical permutation p-values
top-k diagnostic tables
independent metric audit
```

This distinction matters. The benchmark does not only ask whether the observed metric is positive. It asks whether the observed alignment is stronger than what would be expected from random rankings and whether the result is stable under resampling of census divisions.

---

## 5. Metric audit

The independent metric audit passed.

```text
All audited metrics pass.
```

This is important because it reduces the risk that the results are caused by an implementation error, ranking-orientation error, duplicated row issue, or metric-computation bug.

The benchmark can therefore be interpreted as a valid direct-validation result.

---

## 6. Primary result: B1a, all civil-security events

For the primary target, 2021–2025 all civil-security events, the SoVI score shows modest but statistically meaningful global rank alignment.

### Primary headline metrics

```text
Spearman = 0.253
Kendall = 0.159
NDCG@10 = 0.284
NDCG@25 = 0.429
Top-10 overlap = 0.000
Top-25 overlap = 0.360
```

### Hardened interpretation

For the primary target, the Spearman result is:

```text
Observed Spearman = 0.253
Bootstrap 95% CI = [0.078, 0.418]
Permutation null mean ≈ 0.000
Permutation null std ≈ 0.101
Empirical one-sided p ≈ 0.0066
Empirical two-sided p ≈ 0.0121
```

This indicates that the observed SoVI/event-burden rank association is not simply random noise. Higher-SoVI census divisions do tend, on average, to have higher civil-security event burden.

However, the operational prioritization results are weaker:

```text
Observed NDCG@25 = 0.429
NDCG@25 permutation p ≈ 0.306

Observed Top-25 overlap = 0.360
Top-25 overlap permutation p ≈ 0.131

Top-10 overlap = 0.000
```

This means that the primary result should not be interpreted as evidence that raw SoVI accurately identifies the highest-event census divisions. The global rank association is statistically meaningful, but the top-k prioritization capacity is weak.

### Primary scientific interpretation

The correct interpretation is:

> Raw SoVI contains a modest global signal related to observed civil-security event burden, but it is not a strong standalone operational ranking tool.

This is exactly the type of result expected from a static social vulnerability index when tested against observed event burden. SoVI captures social and structural vulnerability, but observed civil-security events also depend on hazard geography, exposure, infrastructure conditions, reporting practices, administrative definitions, and temporal event dynamics.

---

## 7. Target-wise Spearman results

The Spearman results provide the clearest global view of SoVI alignment across target definitions.

| Target                  | Observed Spearman | Bootstrap 95% CI | One-sided permutation p | Interpretation                                       |
| ----------------------- | ----------------: | ---------------: | ----------------------: | ---------------------------------------------------- |
| Weather/climate         |             0.377 |   [0.189, 0.548] |                  0.0001 | Strongest and most robust positive signal            |
| Moderate-or-worse       |             0.258 |   [0.071, 0.439] |                  0.0044 | Meaningful positive signal                           |
| All events, 2021–2025   |             0.253 |   [0.078, 0.418] |                  0.0066 | Primary modest positive signal                       |
| Important-or-extreme    |             0.236 |   [0.044, 0.426] |                  0.0079 | Meaningful but sparse severity signal                |
| Infrastructure          |             0.206 |  [-0.017, 0.407] |                  0.0220 | Weak-to-moderate positive signal, CI crosses zero    |
| All events, 2022–2025   |             0.201 |   [0.026, 0.378] |                  0.0260 | Forward-looking positive signal, weaker than B1a     |
| Precise or very precise |             0.191 |   [0.008, 0.359] |                  0.0295 | Positive signal remains under localization filtering |
| Very precise            |             0.164 |  [-0.013, 0.336] |                  0.0553 | Weak positive signal, borderline                     |
| Land/ground             |             0.085 |  [-0.090, 0.268] |                  0.2076 | No clear meaningful alignment                        |
| Wildfire                |             0.082 |  [-0.121, 0.295] |                  0.2086 | No stable global rank signal; sparse target          |
| Flood/water             |            -0.042 |  [-0.237, 0.156] |                  0.6611 | No alignment                                         |

### Interpretation

The target-wise pattern is scientifically informative.

The strongest result is for weather/climate events. This is conceptually plausible because weather and climate impacts often interact with social vulnerability: older populations, low-income households, housing vulnerability, social isolation, service access, and other social conditions can influence how strongly weather-related stressors translate into reported civil-security events.

The primary all-event target is positive and significant, but modest. This suggests that SoVI has external validity as a broad vulnerability signal, but only partial explanatory power for civil-security event burden.

Flood/water events show no meaningful alignment. This is a valuable negative result. It suggests that flood/water event burden is likely driven more by physical exposure, hydrology, terrain, drainage infrastructure, watershed conditions, and localized hazard mechanisms than by social vulnerability alone.

---

## 8. NDCG@25 results

NDCG@25 evaluates whether CDs with high target burden are ranked highly by SoVI. Unlike Spearman, it focuses more strongly on the upper part of the ranking.

| Target                  | Observed NDCG@25 | Bootstrap 95% CI | Null mean | One-sided permutation p | Interpretation                                  |
| ----------------------- | ---------------: | ---------------: | --------: | ----------------------: | ----------------------------------------------- |
| Weather/climate         |            0.482 |   [0.293, 0.630] |     0.303 |                  0.0188 | Strongest top-25 ranking evidence               |
| Wildfire                |            0.448 |   [0.148, 0.782] |     0.166 |                  0.0173 | High but unstable due to sparse target          |
| All events, 2021–2025   |            0.429 |   [0.334, 0.547] |     0.399 |                  0.3056 | Not clearly better than null for top-25 ranking |
| Infrastructure          |            0.412 |   [0.286, 0.527] |     0.377 |                  0.2934 | Not clearly better than null                    |
| Moderate-or-worse       |            0.411 |   [0.273, 0.567] |     0.321 |                  0.1220 | Suggestive but not strong                       |
| All events, 2022–2025   |            0.398 |   [0.301, 0.520] |     0.387 |                  0.4036 | Not clearly better than null                    |
| Important-or-extreme    |            0.389 |   [0.195, 0.648] |     0.238 |                  0.0456 | Statistically meaningful but sparse             |
| Precise or very precise |            0.322 |   [0.231, 0.465] |     0.323 |                  0.4706 | Similar to null                                 |
| Flood/water             |            0.316 |   [0.197, 0.451] |     0.350 |                  0.6634 | Worse than null tendency                        |
| Very precise            |            0.257 |   [0.168, 0.397] |     0.288 |                  0.6252 | No meaningful top-25 signal                     |
| Land/ground             |            0.224 |   [0.130, 0.375] |     0.272 |                  0.7109 | No meaningful top-25 signal                     |

### Interpretation

NDCG@25 reinforces the distinction between global rank correlation and sharp prioritization.

For B1a, the Spearman signal is significant, but NDCG@25 is not significantly better than the permutation null. This means SoVI has a weak monotonic relationship with event burden across the whole province, but does not reliably concentrate the highest-event CDs near the top.

Weather/climate is the clearest exception. It shows both strong Spearman alignment and significant NDCG@25. This makes it the most convincing target family for direct SoVI external validity.

Wildfire has high NDCG@25, but the result should be interpreted cautiously because wildfire is sparse and has only 31 nonzero CDs. A sparse target can produce high top-k metrics without a stable global rank relationship.

---

## 9. Top-25 overlap results

Top-25 overlap measures the fraction of CDs that appear both in the SoVI top 25 and the target-burden top 25.

| Target                  | Observed top-25 overlap | Bootstrap 95% CI | Null mean | One-sided permutation p | Interpretation                          |
| ----------------------- | ----------------------: | ---------------: | --------: | ----------------------: | --------------------------------------- |
| Weather/climate         |                   0.400 |   [0.240, 0.600] |     0.256 |                  0.0553 | Borderline evidence of top-25 alignment |
| Moderate-or-worse       |                   0.400 |   [0.200, 0.560] |     0.256 |                  0.0536 | Borderline evidence of top-25 alignment |
| All events, 2021–2025   |                   0.360 |   [0.160, 0.480] |     0.256 |                  0.1311 | Weak top-25 overlap evidence            |
| All events, 2022–2025   |                   0.320 |   [0.120, 0.480] |     0.255 |                  0.2695 | Not clearly above null                  |
| Important-or-extreme    |                   0.320 |   [0.160, 0.520] |     0.257 |                  0.2737 | Not clearly above null                  |
| Precise or very precise |                   0.280 |   [0.120, 0.440] |     0.254 |                  0.4619 | Similar to null                         |
| Flood/water             |                   0.280 |   [0.080, 0.480] |     0.255 |                  0.4645 | Similar to null                         |
| Infrastructure          |                   0.280 |   [0.120, 0.440] |     0.256 |                  0.4691 | Similar to null                         |
| Wildfire                |                   0.280 |   [0.200, 0.520] |     0.255 |                  0.4629 | Similar to null                         |
| Very precise            |                   0.240 |   [0.080, 0.400] |     0.255 |                  0.6699 | No top-25 advantage                     |
| Land/ground             |                   0.240 |   [0.080, 0.400] |     0.256 |                  0.6767 | No top-25 advantage                     |

### Interpretation

Top-25 overlap is weaker than Spearman. This is important.

The benchmark shows that SoVI is better interpreted as a broad vulnerability gradient than as a sharp operational prioritization tool. Even when the rank correlation is positive, the overlap between the highest-SoVI CDs and the highest-event CDs is modest.

This is particularly clear for the primary target:

```text
B1a top-25 overlap = 0.360
Permutation p ≈ 0.131
```

This is above random expectation, but not strongly enough to claim robust top-25 prioritization.

---

## 10. Top-10 diagnostic for B1a

The primary target has:

```text
Top-10 overlap = 0.000
```

This means that none of the ten highest-SoVI CDs were among the ten highest-event-burden CDs for the 2021–2025 all-event target.

### CDs selected by SoVI but not by event burden

| CD                                  | SoVI rank | Target rank | SoVI score | Event count |
| ----------------------------------- | --------: | ----------: | ---------: | ----------: |
| Montréal                            |         1 |          31 |      1.000 |          36 |
| Nord-du-Québec                      |         2 |          21 |      0.785 |          51 |
| Minganie--Le Golfe-du-Saint-Laurent |         3 |          67 |      0.770 |          18 |
| La Haute-Gaspésie                   |         4 |          33 |      0.730 |          35 |
| Sherbrooke                          |         5 |          71 |      0.722 |          16 |
| Manicouagan                         |         6 |          44 |      0.715 |          29 |
| Shawinigan                          |         7 |          15 |      0.697 |          57 |
| Le Rocher-Percé                     |         8 |          75 |      0.681 |          14 |
| La Haute-Côte-Nord                  |         9 |          80 |      0.680 |          13 |
| Laval                               |        10 |          52 |      0.678 |          26 |

### CDs selected by event burden but not by SoVI

| CD                       | SoVI rank | Target rank | SoVI score | Event count |
| ------------------------ | --------: | ----------: | ---------: | ----------: |
| Maskinongé               |        61 |           1 |      0.399 |         150 |
| Le Saguenay-et-son-Fjord |        63 |           2 |      0.392 |         119 |
| Matawinie                |        26 |           3 |      0.572 |         112 |
| Pierre-De Saurel         |        42 |           4 |      0.492 |         106 |
| Les Laurentides          |        22 |           5 |      0.584 |         105 |
| D'Autray                 |        53 |           6 |      0.438 |          95 |
| L'Assomption             |        38 |           7 |      0.515 |          89 |
| Vaudreuil-Soulanges      |        41 |           8 |      0.494 |          88 |
| Joliette                 |        13 |           9 |      0.649 |          81 |
| Antoine-Labelle          |        39 |          10 |      0.512 |          80 |

### Interpretation of the top-10 mismatch

This diagnostic table is one of the most scientifically important outputs of the benchmark.

SoVI identifies CDs that are socially vulnerable or structurally vulnerable according to the index construction. However, the highest civil-security event counts are concentrated in different CDs. These target-heavy CDs may be driven by localized exposure, hazard frequency, hydrological conditions, infrastructure networks, land-use patterns, municipal reporting intensity, or event-detection procedures.

This mismatch does not invalidate SoVI. Instead, it clarifies its role.

SoVI is not an event-exposure model. It is a social vulnerability index. Therefore, it should not be expected to perfectly identify the highest observed civil-security event burden when that burden is partly controlled by physical hazards, geography, infrastructure, and reporting processes.

The top-10 failure is a strong argument for moving beyond static social vulnerability scores toward models that incorporate:

```text
hazard exposure
geographic context
historical event burden
infrastructure and service features
spatial dependencies
graph structure
```

---

## 11. Scientific interpretation by target family

### 11.1 All-event burden

The all-event targets show modest positive signal.

```text
B1a Spearman = 0.253
B1b Spearman = 0.201
```

Both are positive. B1a is statistically significant, and B1b remains positive in the cleaner 2022–2025 forward-looking window.

This suggests that SoVI has some external validity as a broad vulnerability proxy. However, the signal weakens under the forward-looking window, and top-k prioritization remains weak.

### 11.2 Localization-precision targets

The precision-filter targets remain positive but weaker.

```text
B1c Spearman = 0.191
B1d Spearman = 0.164
```

This suggests that the primary SoVI signal is not simply an artifact of imprecise localization. However, the signal becomes weaker as the target is restricted to better-localized events.

This may indicate that precise event locations capture more localized hazard and infrastructure mechanisms that are not represented by CD-level SoVI.

### 11.3 Weather/climate events

Weather/climate events are the strongest target family.

```text
Spearman = 0.377
NDCG@25 = 0.482
Top-25 overlap = 0.400
```

This is the most convincing evidence of direct SoVI external validity in the benchmark.

A plausible interpretation is that weather/climate impacts are more socially mediated than some other hazard families. Social vulnerability may affect preparedness, exposure, sensitivity, capacity to respond, housing conditions, and the likelihood that hazards become reported civil-security events.

### 11.4 Flood/water events

Flood/water events show no meaningful SoVI alignment.

```text
Spearman = -0.042
Bootstrap 95% CI = [-0.237, 0.156]
```

This result is scientifically valuable because it shows that SoVI does not function as a universal hazard-burden predictor.

Flood/water events likely require explicit exposure and infrastructure variables:

```text
floodplain intersection
river proximity
watershed position
elevation
slope
drainage infrastructure
impervious surface
culvert / sewer capacity
historical flood recurrence
```

This result directly motivates GIS-derived exposure features and graph-structured modeling.

### 11.5 Infrastructure events

Infrastructure events show a weak-to-moderate positive signal.

```text
Spearman = 0.206
Bootstrap 95% CI = [-0.017, 0.407]
```

The result is suggestive but uncertain. Infrastructure events may depend on both social vulnerability and infrastructure exposure/condition. A static SoVI score alone is likely insufficient.

### 11.6 Severity-threshold targets

Severity-filtered targets show meaningful positive Spearman signal.

```text
Moderate-or-worse Spearman = 0.258
Important-or-extreme Spearman = 0.236
```

This suggests that SoVI may align better with more consequential or socially mediated events than with raw all-event burden alone.

However, strict severity targets are sparser, and uncertainty should be interpreted carefully.

### 11.7 Wildfire events

Wildfire has an interesting metric profile:

```text
Spearman = 0.082
NDCG@25 = 0.448
```

This means that SoVI does not produce a stable global rank correlation for wildfire burden, but some high-wildfire-burden CDs may appear in upper SoVI-ranked positions.

Because wildfire has only 31 nonzero CDs, this result should be treated cautiously. Sparse targets can produce unstable top-k behavior.

---

## 12. Overall conclusion

The hardened B1 benchmark supports the following conclusion:

> The static SoVI-like index has modest but statistically meaningful external validity against Québec civil-security event burden, especially for weather/climate and severity-filtered targets. However, raw SoVI is not sufficient as a sharp operational prioritization model, as shown by weak top-k overlap and zero top-10 overlap for the primary all-event target.

This is a strong result because it is nuanced. It neither dismisses SoVI nor overstates its predictive power.

The benchmark shows that SoVI is useful as a baseline vulnerability signal, but incomplete as a model of observed disruption burden.

---

## 13. Implications for the broader benchmark ladder

The B1 result motivates the next benchmark stage.

The correct lesson is not:

```text
SoVI fails, so use a complex model.
```

The correct lesson is:

```text
SoVI contains real but incomplete signal. Therefore, richer models should be tested against it and against stronger non-graph baselines.
```

The next benchmark stages should test whether performance improves when the model receives additional information that SoVI lacks:

```text
historical event burden
seasonality
hazard-specific histories
exposure variables
GIS-derived physical context
spatial adjacency
graph structure
functional dependencies
```

A natural model ladder is:

```text
B1: raw SoVI direct validation
B2: calibrated SoVI-only model
B3: tabular feature-parity model
B4: history-aware tabular forecasting model
B5: no-edge neural control
B6: random-edge graph control
B7: real CD spatial graph model
```

If future graph models improve over the tabular, no-edge, and random-edge controls, then the project can claim evidence that graph structure adds value beyond static vulnerability and temporal history.

---

## 14. Bridge to temporal modeling and Claudia’s LSTM direction

The broader SVI/311 benchmark suggested that historical reporting patterns are highly informative for predicting operational municipal burden. The SoVI/civil-security benchmark now shows that static vulnerability alone has only partial explanatory power.

Together, these findings create a principled bridge toward temporal node encoders.

The mature version of the idea is not simply:

```text
LSTM + HGNN because it looks advanced.
```

The mature version is:

```text
Past node-level event histories are informative. Therefore, a temporal encoder can summarize each node's local history before a graph model tests whether spatial or functional dependencies add further predictive value.
```

In this framing:

```text
LSTM / GRU / temporal encoder:
summarizes local node histories.

HGNN / graph model:
propagates temporally enriched information across spatial or functional dependencies.

Output:
future civil-security event burden, disruption burden, or vulnerability-related impact.
```

The required ablations are:

```text
history-only baseline
tabular lag/rolling-history model
LSTM-only model
no-edge neural temporal control
random-edge graph temporal control
real graph temporal model
```

This makes temporal graph learning a principled staged extension rather than an arbitrary architectural complication.

---

## 15. Limitations

Several limitations should be explicitly acknowledged.

### Construct mismatch

Civil-security event burden is not the same construct as social vulnerability. It combines vulnerability, exposure, hazard frequency, administrative reporting, municipal practice, and event-detection processes.

### Spatial scale

The benchmark is conducted at the census-division level. This is useful for Québec-wide validation, but it may be too coarse for some hazards, especially flood/water events and localized infrastructure disruptions.

### Event-count target

The target is an observed event burden, not a direct measure of impact, damage, loss, service disruption, or recovery time.

### Hazard heterogeneity

Different hazard families behave differently. A single SoVI score should not be expected to align equally with weather/climate, flood/water, infrastructure, wildfire, and land/ground events.

### Top-k instability

Top-k metrics are sensitive with only 98 CDs. This is why bootstrap intervals, permutation-null comparisons, and diagnostic tables are necessary.

### No causal interpretation

The benchmark is correlational and ranking-based. It does not establish that social vulnerability causes civil-security events.

---

## 16. Final benchmark status

The B1 direct SoVI external-validation benchmark can be considered frozen.

Reasons:

```text
The metric audit passes.
The primary target shows statistically meaningful Spearman alignment.
The result is robust enough to support a modest external-validity claim.
The top-k diagnostics reveal a meaningful limitation.
The target-wise pattern is scientifically interpretable.
The results motivate the next benchmark layer.
```

The frozen research statement is:

> Static SoVI has modest external validity as a broad social vulnerability signal for Québec civil-security event burden, especially for weather/climate and severity-filtered targets. However, it does not provide strong top-k operational prioritization and fails to align with flood/water event burden. This motivates exposure-aware, history-aware, and graph-aware models as the next stage of the benchmark.
