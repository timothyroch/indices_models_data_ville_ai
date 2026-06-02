# SoVI-Like Social Vulnerability Index

The SoVI-like index measures area-level social vulnerability to environmental hazards using latent dimensions extracted from socioeconomic, demographic, housing, infrastructure, and community variables.

It is inspired by Cutter, Boruff & Shirley's original Social Vulnerability Index methodology. Exact reproduction requires the original U.S. county dataset, years, transformations, and 42 variables. For VILLE_IA / Montreal / Quebec / Canadian municipal data, this implementation should normally be treated as a `local_adaptation`.

SoVI is an area-level score. It should not be interpreted as an individual-level diagnosis.

## Difference From SVI

SVI-like is a deterministic rank/domain index: 15 predefined variables, four domains, percentile ranks, domain sums, and a final percentile.

SoVI-like is a multivariate factor-analysis index: many configured variables, z-score standardization, PCA, factor retention, varimax rotation, factor scores, factor orientation, additive factor-score aggregation, and standard-deviation classification.

SoVI does not use SVI percentile-domain aggregation, TOPSIS, OWA, AHP, fuzzy AHP, entropy weighting, or expert weights.

## Original 42-Variable Target

The original SoVI paper used 42 independent variables:

`MED_AGE90`, `PERCAP89`, `MVALOO90`, `MEDRENT90`, `PHYSICN90`, `PCTVOTE92`, `BRATE90`, `MIGRA_97`, `PCTFARMS92`, `PCTBLACK90`, `PCTINDIAN90`, `PCTASIAN90`, `PCTHISPANIC90`, `PCTKIDS90`, `PCTOLD90`, `PCTVLUN91`, `AVGPERHH`, `PCTHH7589`, `PCTPOV90`, `PCTRENTER90`, `PCTRFRM90`, `DEBREV92`, `PCTMOBL90`, `PCTNOHS90`, `HODENUT90`, `HUPTDEN90`, `MAESDEN92`, `EARNDEN90`, `COMDEVDN92`, `RPROPDEN92`, `CVBRPC91`, `FEMLBR90`, `AGRIPC90`, `TRANPC90`, `SERVPC90`, `NRRESPC91`, `HOSPTPC91`, `PCCHGPOP90`, `PCTURB90`, `PCTFEM90`, `PCTF_HH90`, and `SSBENPC90`.

The reference recipe [recipes/sovi_like.yaml](../../recipes/sovi_like.yaml) represents this target structure with local canonical column names.

## Method

The implemented pipeline is:

```text
canonical feature table
-> missing-data handling
-> z-score standardization
-> PCA by SVD
-> factor retention
-> varimax rotation
-> projection-based rotated factor scores
-> recipe-driven factor orientation
-> additive sum of oriented factor scores
-> standard-deviation class bands
```

## Missing Data

The original SoVI paper replaced missing values with zero because factor analysis cannot operate on missing values. This implementation supports `zero_imputation`, `mean_imputation`, `median_imputation`, `drop_units`, and `error`.

Zero imputation is metadata-visible and warned about because zero can distort factor structure when it is not substantively meaningful.

## Standardization

Variables are standardized before PCA:

```text
z = (x - mean) / std
```

This is necessary because SoVI variables mix percentages, dollars, per-capita measures, densities, ratios, and other units. Constant variables are explicitly dropped by default and reported in metadata.

## PCA And Factor Retention

PCA is run on the standardized matrix. The framework saves eigenvalues, explained variance ratios, cumulative explained variance, unrotated loadings, and unrotated scores.

Supported factor-retention rules:

- `eigenvalue_gt`, usually threshold `1.0`;
- `fixed_n`, useful for deterministic synthetic tests and local research decisions.

The implementation does not hardcode the original 11 factors.

## Varimax Rotation

Varimax rotation is the default. It rotates retained loadings to improve interpretability. Rotation can be disabled only with an explicit recipe variant:

```yaml
rotation:
  method: none
```

If rotation is skipped, metadata marks it as a methodological variant.

## Factor Scores

Rotated factor scores are computed with a projection-based approximation: standardized variables are projected onto coefficients derived from the rotated loading matrix. The method is documented in metadata because rotated factor scores can be computed in multiple defensible ways.

## Factor Orientation

Factor signs are arbitrary. A factor can be multiplied by `-1` and remain mathematically equivalent. Therefore factor orientation is recipe-driven.

Supported methods:

- `positive`: keep factor score as-is;
- `negative`: multiply by `-1`;
- `absolute`: use absolute value for ambiguous factors;
- `none`: leave as-is when explicitly allowed;
- `auto_by_anchor_variable`: orient sign using a configured anchor variable loading.

Factor names and orientations require researcher interpretation of rotated loadings.

## Aggregation

The final SoVI score is:

```text
sovi_score_raw = sum(oriented_factor_scores)
```

Factors are not weighted by explained variance by default.

## Classification

The framework preserves the continuous score and also computes a z-score:

```text
sovi_score_z = (sovi_score_raw - mean) / std
```

Default standard-deviation bands:

- `least_vulnerable`: z < -1.0
- `low_vulnerability`: -1.0 <= z < -0.5
- `moderate_vulnerability`: -0.5 <= z <= 0.5
- `high_vulnerability`: 0.5 < z <= 1.0
- `most_vulnerable`: z > 1.0

Benchmark percentile and normalized score fields are produced for comparison convenience; they are not the original SoVI construction method.

## Run The Synthetic Example

```bash
PYTHONPATH=src python -m ville_indices.run \
  --index sovi_like \
  --recipe recipes/sovi_like_synthetic.yaml \
  --features data/example/synthetic_sovi_feature_table.csv \
  --output-dir outputs/sovi_synthetic_run
```

The runner writes standard output, detailed output, metadata, validation reports, missing-data reports, a Markdown report, eigenvalues, explained variance, unrotated loadings, rotated loadings, factor scores, standardized variables, and a factor summary.

## Adapting To VILLE_IA Data

Prepare a canonical feature table with one row per spatial unit and one column per configured SoVI variable. Then update the recipe:

- set `spatial_id_column`;
- map each variable's `canonical_name` to the cleaned feature-table column;
- document proxies with `proxy_used`, `proxy_quality`, `status`, and `conceptual_risk`;
- choose a missing-data strategy;
- choose `eigenvalue_gt` or `fixed_n` retention;
- inspect rotated loadings;
- configure factor orientations with rationales.

The SoVI class should not need to change for a new cleaned feature table.

## Limitations

- Local SoVI-like factors may not match the original 11 SoVI factors.
- Factor signs are arbitrary.
- Factor names require human interpretation.
- More variables than rows can produce unstable factor structures.
- Missing-data choices can materially alter PCA results.
- Constant variables are excluded from PCA and reported.
- A SoVI score is area-level and should not be interpreted as vulnerability of every person in the area.
