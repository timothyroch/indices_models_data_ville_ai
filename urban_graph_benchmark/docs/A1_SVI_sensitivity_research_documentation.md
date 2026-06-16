# A1 SVI Sensitivity Analysis — Research Documentation

**Project:** VILLE_IA / Montréal 311 water-drainage benchmark  
**Benchmark block:** A1 / A1a–A1g  
**Main object tested:** Static tract-level SVI-style vulnerability score  
**Primary data unit:** Montréal census tract × month  
**Primary target family:** 311 water/drainage burden  
**Status:** Completed as a non-fitted sensitivity benchmark block  
**Generated documentation date:** 2026-06-16

---

## 1. Executive summary

The A1a–A1g sensitivity analysis was built to answer a reviewer-facing robustness question:

> Is raw SVI weak because it truly has little alignment with Montréal 311 water/drainage burden, or because the first A1 test compared a static tract-level vulnerability score to a dynamic monthly operational target?

The original A1 benchmark compared a static SVI score to monthly tract-level water/drainage 311 counts. That is a useful strict test, but it is also harsh. SVI is static, social-vulnerability-oriented, and built from census variables. The 311 water/drainage target is dynamic, operational, event-dependent, seasonal, and affected by reporting behavior. Therefore, A1a–A1g tested whether the conclusion “raw SVI is weak” survives when the target is reformulated into more index-compatible definitions.

The main result is:

> Raw SVI remains weak for strict raw 311 forecasting, but becomes more aligned when the target is reframed as reporting-normalized or reporting-excess water/drainage burden. This suggests that SVI is not a short-term incident predictor, but may encode social/reporting-adjusted burden patterns.

The analysis also clarified what raw SVI can and cannot claim in the broader research program:

- Raw SVI is **not** a strong direct predictor of monthly operational 311 water/drainage activity.
- SVI has more relevance when the outcome is reframed as **relative water/drainage burden** rather than raw call volume.
- Population normalization did **not** rescue SVI; in several splits it made the association negative.
- Reporting-normalized and reporting-excess targets are scientifically important because they attempt to separate vulnerability-related burden from general reporting intensity.
- A1a–A1g should be treated as **index-validation sensitivity tests**, not as calibrated prediction models.

---

## 2. Scientific motivation

### 2.1 Why this analysis was needed

The larger project asks whether graph/neural urban models can improve the prediction and explanation of functional urban disruption beyond classical static vulnerability indices. But before comparing SVI to A3/G1/G1.5 or future HGNNs, we needed to understand what the raw SVI score is actually being asked to do.

A naive comparison is:

```text
SVI percentile  → monthly water/drainage 311 count
```

This is simple, but it is not necessarily fair to SVI. SVI was not designed as a monthly incident forecasting model. It is a static vulnerability index intended to describe social vulnerability patterns. The 311 target is not pure hazard impact; it combines:

```text
actual water/drainage issue
+ reporting propensity
+ access to municipal services
+ local behavior around complaint submission
+ seasonality and weather/hydrology
+ infrastructure state
```

Therefore, A1a–A1g decomposed the original question into several target definitions.

### 2.2 The core scientific tension

The strict version asks:

> Do high-SVI tracts have more water/drainage 311 calls in each month?

But a more vulnerability-sensitive version asks:

> Do high-SVI tracts carry a disproportionate water/drainage burden relative to their population or general 311 reporting volume?

These are not the same scientific question. A tract with many 311 calls may simply be dense, highly active, or high-reporting. A tract with a high share of water/drainage calls relative to all its 311 activity may indicate a more specific disruption burden.

A1a–A1g was therefore not an attempt to “save” SVI. It was an attempt to determine which claim about SVI is empirically defensible.

---

## 3. Methodological boundary

A1a–A1g obeyed a strict boundary:

```text
No regression.
No calibration.
No machine learning model.
No target-fitted transformation of SVI.
No parameter learning from 311.
```

The SVI score remained the direct ranking score in every variant. Only the **target definition** changed.

This matters because it separates A1 from A2:

| Stage | Meaning | Uses fitted model? | Role |
|---|---|---:|---|
| A1 / A1a–A1g | Direct SVI ranking tests | No | Tests raw index alignment |
| A2 | Calibrated SVI model | Yes | Tests whether SVI can be supervised/calibrated |
| A3 | Tabular ML baseline | Yes | Tests operational feature model |
| G1/G1.5 | Graph/neural models | Yes | Tests graph/neural operational models |

So A1a–A1g should not be judged as if they are trying to maximize prediction. They test whether the raw SVI ranking aligns with different interpretations of 311 burden.

---

## 4. Data and split design

### 4.1 Primary unit

The base panel is:

```text
census tract × month
```

A static SVI score is attached to each tract and repeated across months.

### 4.2 Split schemes

Two split schemes were used:

| Split scheme | Train rows | Validation rows | Test rows | Scientific purpose |
|---|---:|---:|---:|---|
| temporal | 19,440 | 4,320 | 4,860 | Future-period generalization |
| spatial_block | 21,200 | 5,353 | 2,067 | Spatial held-out/block robustness |

The split design matters because a static SVI score may behave differently under temporal and spatial evaluation. In the temporal split, all tracts can remain represented but months differ. In the spatial-block split, the test set may contain a small number of held-out tracts, which makes tract-level metrics more fragile.

### 4.3 Primary SVI score columns

Two primary continuous SVI columns were evaluated:

| Source column | Oriented score column | Non-missing rows |
|---|---|---:|
| `svi_percentile` | `svi_percentile__higher_more_vulnerable` | 28,090 |
| `svi_score_raw` | `svi_score_raw__higher_more_vulnerable` | 28,090 |

Both were treated as higher = more vulnerable.

### 4.4 Denominator audit

The denominator audit confirmed the sources needed for population-normalized and reporting-normalized variants:

| Field | Value |
|---|---|
| population source | `population_total_2021` |
| all-311 source | `total_311_count_all` |
| all-311 method | direct all-311 column |
| minimum population | 100 |
| minimum all-311 denominator | 10 |
| population non-missing rows | 28,620 |
| all-311 non-missing rows | 28,620 |
| A1f reporting baseline | train-derived water share of all 311 activity |
| surge months by temporal split | June and August |
| surge months by spatial-block split | May and August |

The train-derived reporting baseline is important: A1f does not use future/test water share to define expected burden.

---

## 5. Sensitivity variants A1a–A1g

### Overview

| Variant | Unit | Target definition | Target family | Fitted model? |
|---|---|---|---|---:|
| A1a | tract-month | monthly raw water/drainage count | raw monthly | no |
| A1b | tract | mean monthly water/drainage count | raw aggregate | no |
| A1c | tract | total water/drainage count | raw aggregate | no |
| A1d | tract | water/drainage reports per 1,000 residents | population-normalized | no |
| A1e | tract | water/drainage reports as share of all 311 reports | reporting-normalized | no |
| A1f difference | tract | observed water count minus expected count from reporting baseline | reporting-excess | no |
| A1f ratio | tract | observed water count divided by expected count from reporting baseline | reporting-excess | no |
| A1g mean | tract | mean burden during data-defined surge months | surge-window | no |
| A1g total | tract | total burden during data-defined surge months | surge-window | no |

### A1a — strict tract-month raw-count ranking

**Definition:** Compare repeated static SVI score `s_i` to monthly water/drainage count `y_{i,t}`.

**Scientific question:**

> Does static tract-level SVI rank monthly operational water/drainage 311 burden?

**Why it matters:**

This is the strictest version of A1 and the most directly comparable to the original 311 prediction benchmark. But it is also the harshest because SVI cannot vary by month.

**Expected behavior:** Weak-to-moderate association at best.

### A1b — tract-level mean burden

**Definition:** Aggregate monthly water/drainage calls into mean monthly burden per tract.

**Scientific question:**

> Does SVI rank tracts by their average water/drainage burden over the evaluation window?

**Why it matters:**

This removes month-to-month volatility and tests whether SVI aligns with stable spatial burden patterns.

### A1c — tract-level total burden

**Definition:** Aggregate water/drainage calls into total burden per tract.

**Scientific question:**

> Does SVI rank tracts by total water/drainage activity over the evaluation window?

**Why it matters:**

This is similar to A1b, but emphasizes cumulative burden rather than average monthly burden. In many split windows, A1b and A1c produce identical rankings because total = mean × constant number of months.

### A1d — population-normalized burden

**Definition:** Total water/drainage reports per 1,000 residents.

**Scientific question:**

> Does SVI align with per-capita water/drainage burden?

**Why it matters:**

This tests a tempting hypothesis: maybe SVI does not align with raw counts because raw counts are dominated by population size. If that hypothesis were true, normalizing by population would improve the association.

**Result:** This hypothesis was not supported. A1d often became negative, especially in the temporal test and spatial-block validation.

### A1e — reporting-normalized water share

**Definition:** Water/drainage reports divided by all 311 reports.

**Scientific question:**

> Does SVI align with the share of municipal reporting devoted specifically to water/drainage problems?

**Why it matters:**

This attempts to control for general reporting propensity. If one tract reports everything more often, raw 311 volume may not isolate water/drainage burden. A1e asks whether water/drainage is disproportionately present within the local reporting profile.

**Result:** A1e improved substantially in temporal evaluation.

### A1f — reporting-baseline excess burden

**Definition:** Compare observed water/drainage reports to expected water/drainage reports implied by all-311 reporting volume and a train-derived water-share baseline.

Two forms were used:

1. **Difference:** observed water count minus expected water count.
2. **Ratio:** observed water count divided by expected water count.

**Scientific question:**

> Does SVI align with water/drainage burden above what would be expected from general reporting volume?

**Why it matters:**

This is the closest A1 variant to the idea of disproportionate burden. It is still not causal and still not calibrated ML, but it is more aligned with a vulnerability interpretation than raw counts.

**Result:** A1f was among the strongest target definitions in temporal evaluation, especially the excess-difference variant on NDCG.

### A1g — data-defined surge-window burden

**Definition:** Identify high-water-burden calendar months using the training partition only, then evaluate water/drainage burden during those months in validation/test partitions.

Default surge months:

| Split scheme | Data-defined surge months |
|---|---|
| temporal | June, August |
| spatial_block | May, August |

**Scientific question:**

> Does SVI align better with burden during high water/drainage seasonal windows?

**Why it matters:**

A1g is a step toward event/hazard conditioning. However, it remains internally 311-defined, not externally weather-defined. It should be interpreted as a diagnostic, not as a true rainfall/flood hazard test.

**Result:** Mixed and fragile. Temporal test surge-window metrics were not reliable for all A1g variants, including cases with no valid mean-burden rows or missing rank correlation.

---

## 6. Metrics

### 6.1 Original metric emphasis

The initial sensitivity table emphasized Spearman correlation because A1 is a direct ranking test:

```text
SVI score ranking vs target burden ranking
```

Spearman is appropriate because SVI is not a calibrated count prediction. It tests monotonic association rather than exact count accuracy.

### 6.2 Expanded ranking metric grid

The expanded metric grid later added a fuller ranking view:

| Metric | Interpretation |
|---|---|
| Spearman correlation | Monotonic rank association across all valid units |
| Kendall correlation | Pairwise rank concordance, usually more conservative than Spearman |
| NDCG@10 | Quality of top-10 ranking, with relevance discounting |
| NDCG@25 | Quality of top-25 ranking |
| NDCG@50 | Quality of top-50 ranking |
| NDCG@100 | Quality of top-100 ranking |
| Top-10 overlap | Fraction overlap between predicted and true top 10 |
| Top-25 overlap | Fraction overlap between predicted and true top 25 |
| Top-50 overlap | Fraction overlap between predicted and true top 50 |
| Top-100 overlap | Fraction overlap between predicted and true top 100 |
| Top-5% overlap | Fraction overlap in top 5% of units |
| Top-10% overlap | Fraction overlap in top 10% of units |

This matters because a vulnerability index might have weak global rank correlation but still identify some high-burden areas. The expanded metrics test that possibility.

### 6.3 Why count metrics were not primary for A1

A1 scores are index scores, not calibrated count predictions. Therefore, metrics like MAE, RMSE, and mean Poisson deviance are not conceptually primary for A1a–A1g. They become meaningful in A2/A3/G1 where models output count predictions or count-like expected values.

For A1, ranking metrics are the correct primary family.

---

## 7. Core results — Spearman sensitivity table

The table below uses `svi_percentile` as the primary SVI score.

### 7.1 Temporal split

| Variant | Validation Spearman | Test Spearman | Interpretation |
|---|---:|---:|---|
| A1a strict tract-month raw count | 0.1586 | 0.1606 | Weak positive association with raw monthly burden |
| A1b tract mean burden | 0.2087 | 0.1989 | Slightly stronger after aggregating to tract-level burden |
| A1c tract total burden | 0.2087 | 0.1989 | Same as A1b because total/mean ranking is similar |
| A1d population-normalized burden | -0.1423 | -0.1473 | Population normalization reverses association |
| A1e reporting-normalized share | 0.2266 | 0.2994 | Stronger alignment when controlling for all-311 reporting |
| A1f excess difference | 0.2223 | 0.2780 | Stronger alignment with reporting-excess burden |
| A1f excess ratio | 0.2266 | 0.2994 | Similar to reporting-normalized share |
| A1g surge-window mean | 0.1875 | NaN / no valid mean rows | Diagnostic only; temporal test unstable |
| A1g surge-window total | 0.1875 | NaN | Diagnostic only; temporal test unstable |

### 7.2 Spatial-block split

| Variant | Validation Spearman | Test Spearman | Interpretation |
|---|---:|---:|---|
| A1a strict tract-month raw count | 0.0243 | 0.0986 | Very weak spatial-block signal |
| A1b tract mean burden | 0.0027 | 0.1050 | Nearly no validation signal; weak positive test signal |
| A1c tract total burden | 0.0027 | 0.1050 | Same as A1b |
| A1d population-normalized burden | -0.4696 | -0.0670 | Strongly negative in validation; weakly negative in test |
| A1e reporting-normalized share | 0.1225 | -0.0247 | Not stable across spatial-block validation/test |
| A1f excess difference | 0.1203 | -0.0111 | Not stable across spatial-block validation/test |
| A1f excess ratio | 0.1225 | -0.0247 | Not stable across spatial-block validation/test |
| A1g surge-window mean | 0.0330 | 0.0401 | Very weak |
| A1g surge-window total | 0.0330 | 0.0401 | Very weak |

### 7.3 Interpretation of Spearman results

The temporal split supports a clear pattern:

```text
raw monthly burden      ≈ weak association
tract-level raw burden  ≈ slightly stronger association
population-normalized   ≈ negative association
reporting-normalized    ≈ strongest association
reporting-excess        ≈ strongest association
```

The spatial-block split is much weaker and less stable. This may reflect a combination of:

- fewer held-out tracts in the spatial-block test target variants;
- spatial heterogeneity in Montréal neighborhoods;
- SVI's limited ability to generalize to spatially held-out areas for this specific 311 outcome;
- instability of tract-level normalized targets when the number of spatial-block units is small.

---

## 8. Expanded metric results for A1a–A1g

The expanded metric results are particularly useful because they ask whether SVI identifies top-burden tracts even if global rank correlation is weak.

### 8.1 Temporal test — selected expanded metrics for `svi_percentile`

| Variant | Spearman | NDCG@10 | NDCG@50 | NDCG@100 | Top-10% overlap | n valid |
|---|---:|---:|---:|---:|---:|---:|
| A1a strict tract-month raw count | 0.1606 | 0.1884 | 0.2386 | 0.2206 | 0.0524 | 4,770 |
| A1b tract mean burden | 0.1989 | 0.2940 | 0.3315 | 0.4192 | 0.0566 | 530 |
| A1c tract total burden | 0.1989 | 0.2940 | 0.3315 | 0.4192 | 0.0566 | 530 |
| A1d population-normalized burden | -0.1473 | 0.2119 | 0.2644 | 0.3431 | 0.0189 | 530 |
| A1e reporting-normalized share | 0.2994 | 0.5136 | 0.6021 | 0.6853 | 0.1400 | 494 |
| A1f excess difference | 0.2780 | 0.7729 | 0.8446 | 0.8860 | 0.0800 | 494 |
| A1f excess ratio | 0.2994 | 0.5136 | 0.6021 | 0.6853 | not captured | 494 |
| A1g surge-window mean | NaN | not reliable | not reliable | not reliable | not reliable | 0 |
| A1g surge-window total | NaN | not reliable | not reliable | not reliable | not reliable | 530 |

### 8.2 Spatial-block test — selected expanded metrics for `svi_percentile`

| Variant | Spearman | NDCG@10 | NDCG@50 | NDCG@100 | Top-10% overlap | n valid |
|---|---:|---:|---:|---:|---:|---:|
| A1a strict tract-month raw count | 0.0986 | 0.0670 | 0.0724 | 0.1498 | 0.0305 | 1,961 |
| A1b tract mean burden | 0.1050 | 0.3972 | 0.6889 | 0.6889 | 0.0000 | 37 |
| A1c tract total burden | 0.1050 | 0.3972 | 0.6889 | 0.6889 | 0.0000 | 37 |
| A1d population-normalized burden | -0.0670 | 0.4336 | 0.7482 | 0.7482 | 0.0000 | 37 |
| A1e reporting-normalized share | -0.0247 | 0.7404 | 0.9063 | 0.9063 | 0.0000 | 35 |
| A1f excess difference | -0.0111 | 0.3477 | 0.6919 | 0.6919 | 0.0000 | 35 |
| A1f excess ratio | -0.0247 | 0.7404 | 0.9063 | 0.9063 | 0.0000 | 35 |
| A1g surge-window mean | 0.0401 | 0.3482 | 0.6575 | 0.6575 | 0.0000 | 37 |
| A1g surge-window total | 0.0401 | 0.3482 | 0.6575 | 0.6575 | 0.0000 | 37 |

### 8.3 Important caution about NDCG on small spatial-block tract targets

Some spatial-block test NDCG values look high even when Spearman and top-overlap are weak or zero. This is a warning sign that spatial-block tract-level evaluation has small `n` for many target variants, often around 35–37 valid tracts.

When `n` is small, NDCG@50 and NDCG@100 can become less intuitive because `K` is larger than the number of valid units. Therefore, spatial-block NDCG should be interpreted together with:

- valid `n`;
- Spearman/Kendall;
- top-overlap;
- target distribution;
- whether the target is dense or sparse.

For spatial-block test A1e, for example, NDCG@100 is high but top-10% overlap is zero. That means the ranking has some graded relevance structure but is not successfully capturing the very top decile.

---

## 9. Final benchmark A1 raw SVI rows with expanded metric grid

The final cleaned benchmark comparison includes A1 direct-ranking rows for raw SVI class and raw SVI percentile. These rows use the expanded metric grid.

| Row | Spearman | Kendall | NDCG@10 | NDCG@25 | NDCG@50 | NDCG@100 | Top-10 | Top-25 | Top-50 | Top-100 | Top-5% | Top-10% |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| A1 raw SVI class | 0.1856 | 0.1449 | 0.2588 | 0.2876 | 0.2356 | 0.2223 | 0.0000 | 0.0000 | 0.0200 | 0.0200 | 0.0586 | 0.1509 |
| A1 raw SVI percentile | 0.1606 | 0.1112 | 0.1884 | 0.2548 | 0.2386 | 0.2206 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0167 | 0.0524 |

Interpretation:

- Raw SVI percentile is weak under the strict operational target.
- SVI class performs slightly better than percentile on top-10% overlap, but still remains weak overall.
- The expanded grid confirms that raw SVI is not simply failing because of one metric choice; it is weak across global correlation, NDCG, and top-overlap under the strict direct-ranking task.

---

## 10. Scientific interpretation

### 10.1 What A1a–A1g tells us about SVI

The sensitivity analysis did not show that SVI is useless. It showed that SVI is being asked different questions under different target definitions.

The strongest interpretation is:

> SVI is weak as a direct short-term operational predictor of raw monthly 311 water/drainage counts, but has more alignment with reporting-normalized and reporting-excess definitions of water/drainage burden.

This supports a distinction between:

```text
operational prediction
```

and:

```text
vulnerability-context alignment
```

SVI is closer to the second.

### 10.2 Why reporting-normalized targets matter

A1e and A1f are scientifically important because 311 data is a reporting process, not a pure hazard sensor. High raw 311 volume may reflect:

- more residents;
- more active reporting culture;
- easier access to reporting channels;
- borough/municipal-service differences;
- more general municipal issues, not specifically water/drainage vulnerability.

By normalizing water/drainage reports by total 311 activity, A1e and A1f attempt to test whether water/drainage is disproportionate relative to local reporting volume.

The improvement from A1a/A1b to A1e/A1f suggests that SVI may be more related to relative or disproportionate burden than absolute service-request volume.

### 10.3 Why population normalization did not help

A1d was negative in temporal test and strongly negative in spatial-block validation. This means the hypothesis “SVI aligns better with per-capita water/drainage burden” was not supported.

This result matters because population normalization is an obvious first correction. Its failure suggests that simple per-capita burden is not the right vulnerability target for SVI in this dataset.

### 10.4 Why A1g remains diagnostic only

A1g tries to move toward hazard conditioning, but it uses data-defined 311 surge months rather than external rainfall/flood/hazard events. That makes it useful as a seasonal diagnostic, but not yet a true hazard-conditioned vulnerability validation.

A stronger future version would use external hazard windows such as heavy rainfall, flooding alerts, hydrometric levels, civil-security events, or weather observations.

---

## 11. Hypotheses tested and outcomes

### H1 — Raw SVI should weakly rank raw monthly water/drainage burden

**Result:** Weakly supported. Temporal test Spearman was about 0.16. Spatial-block test Spearman was about 0.10.

**Interpretation:** SVI has some weak positive association with raw burden, but not enough to be considered a strong operational predictor.

### H2 — Aggregating monthly burden to tract-level burden should improve alignment

**Result:** Partly supported in temporal split. A1b/A1c increased temporal test Spearman to about 0.20. Spatial-block results remained weak.

**Interpretation:** SVI aligns slightly better with stable spatial burden than with month-by-month burden.

### H3 — Population normalization should improve alignment

**Result:** Not supported. A1d was negative in temporal test and spatial-block validation.

**Interpretation:** SVI does not appear to rank per-capita water/drainage reporting burden in this benchmark.

### H4 — Reporting normalization should improve alignment

**Result:** Supported in temporal evaluation. A1e achieved temporal test Spearman around 0.299 and NDCG@100 around 0.685.

**Interpretation:** SVI aligns better with water/drainage share of all 311 than with raw water/drainage count.

### H5 — Reporting-excess burden should improve alignment

**Result:** Supported in temporal evaluation. A1f excess difference had temporal test Spearman around 0.278 and NDCG@100 around 0.886.

**Interpretation:** SVI may be informative about areas with water/drainage burden above a general-reporting baseline.

### H6 — Surge-window burden should reveal stronger SVI alignment

**Result:** Not clearly supported. A1g was fragile and unstable, especially in temporal test.

**Interpretation:** A1g requires a more principled external hazard definition before it can support strong claims.

---

## 12. Relationship to A2, A3, and graph/neural benchmarks

A1a–A1g clarifies what raw SVI can do before calibration or ML.

A2 asks a different question:

> If we allow supervised calibration using SVI/reporting features, can SVI become more predictive?

A3 asks:

> If we allow a full tabular supervised model with operational/dynamic features, how much better can prediction become?

G1/G1.5 ask:

> Does graph/neural structure improve prediction beyond tabular and no-edge controls?

Therefore, A1a–A1g should not be viewed as competitors to A3/G1/G1.5 on equal informational footing. A3/G1/G1.5 can use dynamic operational history and learned functions. A1 uses a static index score only.

The correct scientific relation is:

```text
A1a–A1g = external validity / target-definition sensitivity of raw SVI.
A2 = supervised calibration of SVI-like signal.
A3/G = operational prediction benchmark.
```

---

## 13. Methodological lessons for future B1 / static vulnerability graph work

A1a–A1g later became relevant for the proposed B1 static vulnerability representation work. The key lesson is that a static score should be evaluated using the same target-family logic.

If a future B1 model outputs one static tract score, it should be evaluated against:

- A1a as a diagnostic monthly raw-burden test;
- A1b/A1c as stable tract burden tests;
- A1e/A1f as more vulnerability-aligned reporting-normalized/excess burden tests;
- A1g only after surge/hazard windows are better defined.

This prevents a future model from being evaluated only on the harshest raw monthly target.

---

## 14. Limitations

### 14.1 311 is a proxy label

311 calls are not pure measures of infrastructure failure or social harm. They are observed reporting events. Therefore, the benchmark evaluates alignment with observed reporting burden, not true latent vulnerability.

### 14.2 Static SVI cannot model time

SVI does not vary by month. A1a is intentionally harsh because it compares a static tract score to a dynamic monthly target.

### 14.3 A1e/A1f change the target concept

Reporting-normalized and reporting-excess variants are not the same as raw burden. They answer a different but arguably more vulnerability-relevant question.

### 14.4 Spatial-block tract-level targets have small valid sample sizes

Some spatial-block test variants have only about 35–37 valid tract-level units. Metrics on these should be interpreted cautiously.

### 14.5 A1g is not externally hazard-conditioned

A1g uses data-defined high-burden calendar months derived from training 311 data, not independent rainfall/flood/hazard observations.

---

## 15. Main conclusions

1. The strict raw SVI-to-monthly-311 test is weak but not meaningless.
2. Aggregating from tract-month to tract-level burden modestly improves temporal alignment.
3. Population normalization does not rescue SVI and can reverse the association.
4. Reporting-normalized and reporting-excess burden definitions produce the strongest SVI alignment.
5. The expanded metric grid confirms that raw SVI is weak under strict operational ranking, not just under Spearman.
6. NDCG can look high in small spatial-block tract-level settings, so top-overlap and valid sample size must be considered.
7. The scientifically honest statement is not “SVI predicts 311 well.” It is:

> SVI is weak as a raw monthly operational predictor but more informative when the target is reframed as disproportionate/reporting-adjusted water-drainage burden.

---

## 16. Recommended language for reports/papers

A concise research-report formulation:

> We evaluated raw SVI as a direct non-fitted ranking score against seven 311 water/drainage burden definitions. The strict tract-month comparison produced weak positive association, confirming that static SVI is not a strong monthly operational predictor. However, SVI aligned more strongly with reporting-normalized and reporting-excess burden, suggesting that vulnerability indices may be better interpreted as contextual indicators of disproportionate burden rather than direct short-term incident forecasts.

A stronger methodological formulation:

> The A1 sensitivity block demonstrates that conclusions about composite vulnerability indices depend strongly on the operational target definition. Raw count prediction, per-capita burden, reporting-normalized burden, and reporting-excess burden are not interchangeable validation targets. For urban resilience benchmarking, static indices should therefore be evaluated across multiple outcome definitions before being declared empirically weak or strong.

---

## 17. Source artifacts and reproducibility notes

Primary source outputs from the A1 sensitivity run:

```text
urban_graph_benchmark/outputs/mtl_311_water_v0/baselines/A1_svi_sensitivity/metrics.csv
urban_graph_benchmark/outputs/mtl_311_water_v0/baselines/A1_svi_sensitivity/sensitivity_table.csv
urban_graph_benchmark/outputs/mtl_311_water_v0/baselines/A1_svi_sensitivity/baseline_report.md
urban_graph_benchmark/outputs/mtl_311_water_v0/baselines/A1_svi_sensitivity/tract_level_targets.csv
urban_graph_benchmark/outputs/mtl_311_water_v0/baselines/A1_svi_sensitivity/top_decile_overlap_by_variant.csv
urban_graph_benchmark/outputs/mtl_311_water_v0/baselines/A1_svi_sensitivity/target_audit.csv
urban_graph_benchmark/outputs/mtl_311_water_v0/baselines/A1_svi_sensitivity/svi_score_audit.csv
urban_graph_benchmark/outputs/mtl_311_water_v0/baselines/A1_svi_sensitivity/svi_static_score_audit.csv
```

Run command used:

```bash
PYTHONPATH=urban_graph_benchmark/src python -m ville_hgnn.baselines.a1_svi_sensitivity \
  --split-schemes temporal spatial_block \
  --enable-surge-window \
  --surge-quantile 0.90 \
  --tract-ndcg-k 50 \
  --min-population 100 \
  --min-all311 10 \
  --winsorize-rate-quantile 0.99 \
  --no-plots \
  --overwrite
```

Run summary:

| Output count | Value |
|---|---:|
| SVI score columns | 4 |
| Primary SVI columns | 2 |
| Metric rows | 1,728 |
| Sensitivity summary rows | 72 |
| Tract target rows | 1,220 |
| Top-K rows | 744 |
| Plot count | 0 |

---

## 18. Practical next steps

1. Preserve A1a–A1g as a frozen benchmark block.
2. Use A1e/A1f as the main target families when discussing SVI as vulnerability-context alignment.
3. Do not overclaim A1g until external hazard windows are available.
4. When adding future B1 static vulnerability models, evaluate them using the same A1a–A1g framework.
5. When adding Track B SoVI/CD-level validation, build analogous sensitivity target definitions appropriate to civil-security event burden.
6. For any final paper/report, clearly separate:
   - raw index validity;
   - calibrated index prediction;
   - operational ML prediction;
   - graph topology value;
   - functional/HGNN disruption modeling.

