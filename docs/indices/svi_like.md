# SVI-Like Social Vulnerability Index

The SVI-like index measures area-level social vulnerability for disaster management. It is not a hazard-specific risk index: it does not estimate flood depth, heat exposure, infrastructure failure, or physical damage. It summarizes socioeconomic and demographic conditions that may affect a community's capacity to prepare for, respond to, and recover from disasters.

This implementation follows the CDC/ATSDR SVI method described by Flanagan et al. (2011): 15 variables, four conceptual domains, percentile ranks, domain sums, domain percentile ranks, an overall domain-percentile sum, a final overall percentile rank, and flags for extreme variable percentiles.

SVI should not be interpreted at the individual level. A high-SVI zone is an area-level signal; it does not mean that every person in that zone is vulnerable.

## Variables And Domains

Socioeconomic status:

- `pct_below_poverty`
- `pct_unemployed`
- `per_capita_income`
- `pct_no_high_school`

Household composition / disability:

- `pct_age_65_plus`
- `pct_age_17_or_younger`
- `pct_disability`
- `pct_single_parent_households`

Minority status / language:

- `pct_minority`
- `pct_limited_language`

Housing / transportation:

- `pct_multiunit_structures`
- `pct_mobile_homes`
- `pct_crowding`
- `pct_no_vehicle`
- `pct_group_quarters`

## Percentile Ranks

Each variable is converted to a vulnerability percentile rank:

```text
PR = (rank - 1) / (N - 1)
```

Ties use the minimum rank by default. For a comparison scope with `N = 1`, the default implementation returns `0.0` and records that behavior in metadata.

For positive-direction variables, higher raw values receive higher vulnerability percentiles. For `per_capita_income`, the direction is negative: lower income receives the higher vulnerability percentile.

## Aggregation

SVI is not the direct sum or average of all 15 variable percentiles.

The implemented sequence is:

```text
raw variables
-> variable percentile ranks
-> domain raw sums
-> domain percentile ranks
-> final sum of the four domain percentile ranks
-> final overall SVI percentile rank
```

This makes variables equal within domains and gives the four domains equal final-stage weight.

## Flags

Variable flags are assigned when:

```text
variable percentile >= 0.90
```

The threshold is configurable in `recipes/svi_like.yaml`. Domain flag counts and total flag counts are interpretability outputs, not replacements for the SVI score.

## Reproduction Levels

`strict_original_like` requires all 15 canonical SVI variables and follows the original rank/domain procedure.

`local_adaptation` uses the same methodology but allows documented local proxies in the recipe. For example, `per_capita_income` can be mapped to `median_household_income` only if the proxy is explicitly declared with quality and conceptual-risk notes.

`partial_svi_like` is only allowed when explicitly configured. It reports incomplete variables/domains and marks output quality flags accordingly. Partial output is not a full SVI-like reproduction.

## Output Columns

The standard benchmark output contains:

- `zone_id`
- `index_name`
- `run_id`
- `score_raw`
- `score_normalized_0_1`
- `score_direction`
- `rank`
- `percentile`
- `class`
- `missing_count`
- `quality_flag`
- `reproduction_level`

For SVI, `score_raw` is `svi_overall_sum`, while `score_normalized_0_1` and `percentile` are `svi_overall_percentile`.

The detailed intermediate output contains raw variable copies, `svi_pr_*` variable percentiles, `svi_*_sum` domain sums, `svi_*_percentile` domain percentiles, `svi_overall_sum`, `svi_overall_percentile`, variable flags, domain flag counts, `svi_total_flag_count`, proxy flags, missing flags, and quality flags.

## Run The Synthetic Example

```bash
PYTHONPATH=src python -m ville_indices.run \
  --index svi_like \
  --recipe recipes/svi_like.yaml \
  --features data/example/synthetic_svi_feature_table.csv \
  --output-dir outputs/svi_synthetic_run
```

The runner writes standard output, detailed intermediate output, metadata, validation reports, missing-data reports, and a Markdown run report.

## Adapting To VILLE_IA Data

Prepare a canonical feature table with one row per spatial unit and one column per SVI variable or documented proxy. Then update only the recipe:

- set `spatial_id_column`;
- set `population_column` if available;
- map each SVI variable's `canonical_name` to the cleaned feature-table column;
- document any proxy with `proxy_used`, `proxy_quality`, `status`, and `conceptual_risk`;
- keep `reproduction_level` honest: use `local_adaptation` for documented proxies and `partial_svi_like` only when explicitly approved.

The SVI index class should not need to change for a new cleaned feature table.

## Limitations And Open Decisions

- Minority/language variables require careful Canadian and Quebec-specific methodological review.
- Mobile homes may have low relevance or poor availability in Montreal-area data.
- The original paper does not specify an imputation rule after zero-population exclusions; this implementation defaults to failing on missing values.
- Grouped comparison scopes are represented in the recipe structure, but the reference implementation currently computes the global scope.
- SVI captures social vulnerability only. It should be combined carefully with hazard, exposure, infrastructure, service, and resource data for risk analysis.
