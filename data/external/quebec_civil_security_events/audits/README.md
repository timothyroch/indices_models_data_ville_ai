# Québec civil-security events inspection

Raw file inspected:

```text
data/external/quebec_civil_security_events/raw/quebec_civil_security_events.json
```

This directory contains inspection/audit outputs only. No cleaning, aggregation,
spatial join, SoVI construction, or target construction is performed here.

Detected format: `geojson_feature_collection`
Rows: `4147`
Columns: `27`

Core outputs:

- `inspection_summary.json`
- `schema_audit.csv`
- `core_field_quality_audit.csv`
- `row_core_missing_examples.csv`
- `date_coverage_audit.csv`
- `monthly_counts_by_date_column.csv`
- `coordinate_audit.csv`
- `coordinate_problem_examples.csv`
- `duration_audit.csv`
- `negative_duration_examples.csv`
- `duplicate_candidate_audit.csv`
- `value_counts__*.csv`
- `crosstab__*.csv`
- `municipality_event_counts.csv`
- `normalized_preview.csv`

Project note: this dataset is best treated as an aléa/sinistre/security-civil
event layer for external validation or target/context construction. It does not
replace the already-built CD-level SoVI index run.
