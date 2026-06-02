# VILLE_IA Composite-Index Benchmark Framework

This repository contains the reusable foundation for reproducing and comparing multiple composite vulnerability and resilience indices in a common benchmark framework.

The framework is designed for multiple composite indices with different methodologies. It does not assume all indices are weighted sums.

## Architecture

The core design rule is that an index never depends directly on raw files. Index classes consume a canonical feature table: one row per spatial unit and one column per canonical variable.

The framework separates:

- raw data ingestion, which is intentionally not implemented yet;
- feature engineering, which should create canonical feature tables;
- recipe loading, where scientific and methodological decisions are declared in YAML;
- feature-table validation;
- index computation;
- metadata and validation reporting;
- standardized benchmark outputs;
- intermediate outputs for explainability and future ML/HGNN use.

## Project Layout

```text
src/ville_indices/
  core/          Base interface, recipes, validation, metadata, outputs, registry
  operations/    Normalization, orientation, ranking, aggregation, classification, missing data
  indices/       Index implementations; currently only a toy dummy index
  reporting/     Markdown run report generation
recipes/         YAML recipes
data/example/    Synthetic canonical feature table
outputs/         Generated benchmark outputs
tests/           Deterministic pytest coverage
```

## Run The Dummy Index

The included dummy index is only a toy architecture validation example. It is not SVI, SoVI, BRIC, TOPSIS, LOFVI, or any scientific index.

```bash
python -m ville_indices.run \
  --recipe recipes/dummy_index.yaml \
  --features data/example/synthetic_feature_table.csv \
  --output-dir outputs/dummy_run
```

If the package is installed, the same runner is available as:

```bash
ville-indices \
  --recipe recipes/dummy_index.yaml \
  --features data/example/synthetic_feature_table.csv \
  --output-dir outputs/dummy_run
```

The runner writes:

- `standard_output.csv`;
- `intermediate_output.csv`;
- `metadata.json`;
- `metadata.yaml`;
- `validation_report.json`;
- `validation_report.yaml`;
- `missing_data_report.json`;
- `run_report.md`.

## Run Tests

```bash
pytest
```

For an editable local install:

```bash
python -m pip install -e ".[test]"
pytest
```

## Recipes

Recipes are YAML files that declare the methodological contract for an index run: variables, canonical column mappings, normalization, direction/orientation, missing-data handling, weighting, aggregation, classification, output options, and assumption/decision metadata.

The schema is intentionally flexible enough to support future non-additive methods such as:

- SVI-style percentile ranks and domain sums;
- SoVI-style PCA or factor-score workflows;
- BRIC-style subcomponent scores;
- Yang-style TOPSIS and Shannon entropy correction;
- Kaur-style OWA, fuzzy AHP, and LOFVI workflows.

Only generic utilities and the dummy additive validation index are implemented now.

## Fit/Transform Design

Normalization and missing-data handling use fit/transform semantics. Descriptive full-dataset indices may fit on the whole canonical feature table. Future predictive or ML/HGNN workflows can fit parameters on training data and apply them to validation/test data to avoid leakage.

## Validation And Metadata

Validation returns structured reports instead of only raising exceptions. Blocking errors include missing spatial IDs, duplicate spatial IDs, missing required variables under strict strategies, and nonnumeric required variables for numeric operations. Warnings cover issues such as optional missing variables, constant columns, high missingness, suspicious percentage/proportion scales, and out-of-range values.

Every run records structured metadata, including recipe hash, variables used/missing, validation summary, missing-data strategy, normalization parameters, orientation decisions, aggregation/classification methods, assumptions, warnings, and output files.

## Adding A Future Index

A future SVI-like implementation should add:

- one recipe file;
- one index class under `src/ville_indices/indices/`;
- registration in `src/ville_indices/indices/__init__.py`;
- SVI-specific tests;
- any SVI-specific aggregation utilities needed, such as domain sum then rank.

It should reuse the existing recipe loader, validation reports, missing-data module, ranking utilities, output schema, metadata export, and run report generation.

## Implemented Scientific Indices

- SVI-like social vulnerability index: see [docs/indices/svi_like.md](docs/indices/svi_like.md) and [recipes/svi_like.yaml](recipes/svi_like.yaml).
- SoVI-like factor-analysis social vulnerability index: see [docs/indices/sovi_like.md](docs/indices/sovi_like.md), [recipes/sovi_like.yaml](recipes/sovi_like.yaml), and [recipes/sovi_like_synthetic.yaml](recipes/sovi_like_synthetic.yaml).

## Intentionally Not Implemented Yet

This initialization does not implement SVI, SoVI, BRIC, TOPSIS, OWA, fuzzy AHP, CKAN ingestion, shapefile/raster ingestion, APIs, notebook workflows, or fake scientific outputs. Those should be added as real methodology-specific modules on top of this foundation.
