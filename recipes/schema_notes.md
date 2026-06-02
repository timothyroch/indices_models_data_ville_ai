# Recipe Schema Notes

Recipes are scientific/methodological specifications for composite-index runs. The current loader accepts the common fields below and preserves unknown fields under `extra` so future index modules can extend the schema without rewriting the core.

Common top-level fields:

- `name`
- `version`
- `source`
- `method_reference`
- `construct_measured`
- `score_direction`
- `reproduction_level`
- `spatial_id_column`
- `variables`
- `missing_data`
- `aggregation`
- `classification`
- `outputs`
- `assumptions`
- `decisions`

Variable entries should include:

- `canonical_name`
- `required`
- `unit`
- `scale`
- `direction`
- `normalization`
- `weight`
- `numeric`
- `nonnegative`
- `decisions`

Decision status values are intentionally broad:

- `paper_explicit`
- `paper_implicit`
- `implementation_assumption`
- `local_adaptation`
- `researcher_decision_required`
- `method_family_default`

Future index modules may add method-specific sections such as `domains`, `pca`, `topsis`, `entropy`, `owa`, `ahp`, or `proxy_variables`.
