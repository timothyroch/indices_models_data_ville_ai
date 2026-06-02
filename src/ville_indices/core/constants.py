"""Shared constants for the composite-index framework."""

DECISION_STATUSES = {
    "paper_explicit",
    "paper_implicit",
    "implementation_assumption",
    "local_adaptation",
    "researcher_decision_required",
    "method_family_default",
    "implementation_example",
}

SCORE_DIRECTIONS = {
    "higher_is_more_vulnerable",
    "higher_is_less_vulnerable",
    "higher_is_more_resilient",
    "higher_is_less_resilient",
    "custom",
}

VARIABLE_DIRECTIONS = {"positive", "negative", "none", "custom"}

MISSING_DATA_STRATEGIES = {
    "error",
    "drop_units",
    "median_imputation",
    "mean_imputation",
    "zero_imputation",
    "keep_missing_with_flags",
}

NORMALIZATION_METHODS = {
    "minmax",
    "zscore",
    "percentile_rank",
    "vector_normalization",
    "none",
}

AGGREGATION_METHODS = {
    "sum",
    "mean",
    "weighted_sum",
    "custom",
}

CLASSIFICATION_METHODS = {
    "quantile",
    "equal_interval",
    "none",
    "natural_breaks",
    "standard_deviation",
}

STANDARD_OUTPUT_COLUMNS = [
    "zone_id",
    "index_name",
    "run_id",
    "score_raw",
    "score_normalized_0_1",
    "score_direction",
    "rank",
    "percentile",
    "missing_count",
    "quality_flag",
    "reproduction_level",
]
