"""Human-readable Markdown run report generation."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from ville_indices.core.metadata import RunMetadata


def _markdown_table(frame: pd.DataFrame, columns: list[str]) -> str:
    selected = frame[columns].copy()
    header = "| " + " | ".join(columns) + " |"
    separator = "| " + " | ".join(["---"] * len(columns)) + " |"
    rows = []
    for _, row in selected.iterrows():
        rows.append("| " + " | ".join(str(row[column]) for column in columns) + " |")
    return "\n".join([header, separator, *rows])


def _bullet_list(items: list[str]) -> str:
    if not items:
        return "- None\n"
    return "".join(f"- {item}\n" for item in items)


def generate_run_report(
    *,
    metadata: RunMetadata,
    standard_output: pd.DataFrame,
    output_files: dict[str, str],
) -> str:
    top = standard_output.sort_values("score_raw", ascending=False).head(5)
    bottom = standard_output.sort_values("score_raw", ascending=True).head(5)
    svi_extra = metadata.extra.get("svi", {})
    sovi_extra = metadata.extra.get("sovi", {})
    ranking_extra = metadata.extra.get("ranking", {})
    flags_extra = metadata.extra.get("flags", {})
    population_extra = metadata.extra.get("population_filter", {})
    proxied_variables = metadata.extra.get("variables_proxied", [])

    lines = [
        f"# Run Report: {metadata.index_name}",
        "",
        f"- Run ID: `{metadata.run_id}`",
        f"- Index version: `{metadata.index_version}`",
        f"- Reproduction level: `{metadata.reproduction_level}`",
        f"- Construct measured: `{metadata.construct_measured}`",
        f"- Score direction: `{metadata.score_direction}`",
        f"- Input feature table: `{metadata.input_feature_table_path}`",
        f"- Spatial units input: `{metadata.number_input_units}`",
        f"- Spatial units output: `{metadata.number_output_units}`",
    ]
    if metadata.index_name == "svi_like":
        lines.extend(
            [
                f"- Source method: `{metadata.extra.get('source_reference', metadata.index_name)}`",
                f"- Comparison scope: `{metadata.extra.get('comparison', {}).get('default_scope', 'global')}`",
                f"- Zero/invalid population units excluded: `{population_extra.get('excluded_count', 0)}`",
                f"- Ranking formula: `{ranking_extra.get('formula', '(rank - 1) / (N - 1)')}`",
                f"- Tie method: `{ranking_extra.get('tie_method', 'min')}`",
                f"- Flag threshold: `{flags_extra.get('variable_flag_threshold', 0.9)}`",
            ]
        )
    if metadata.index_name == "sovi_like":
        factor_retention = metadata.extra.get("factor_retention", {})
        factor_analysis = metadata.extra.get("factor_analysis", {})
        rotation = metadata.extra.get("rotation", {})
        lines.extend(
            [
                f"- Source method: `{metadata.extra.get('source_reference', metadata.index_name)}`",
                f"- PCA method: `{factor_analysis.get('method', 'pca')}`",
                f"- Factor retention: `{factor_retention.get('method')}`",
                f"- Factors retained: `{factor_retention.get('n_factors_retained')}`",
                f"- Rotation method: `{rotation.get('method')}`",
            ]
        )
    lines.extend(
        [
            "",
            "## Variables Used",
            "",
            _bullet_list(metadata.variables_used),
            "## Variables Missing",
            "",
            _bullet_list(metadata.variables_missing + metadata.optional_variables_missing),
            "## Warnings",
            "",
            _bullet_list(metadata.warnings),
        ]
    )
    if metadata.index_name in {"svi_like", "sovi_like"}:
        lines.extend(
            [
                "## Proxies Used",
                "",
                _bullet_list(
                    [
                        (
                            f"{item.get('variable')} -> {item.get('canonical_name')} "
                            f"({item.get('proxy_quality') or 'quality not specified'})"
                        )
                        for item in proxied_variables
                    ]
                ),
            ]
        )
    lines.extend(
        [
            "## Missing Data Summary",
            "",
            f"- Strategy: `{metadata.missing_data_strategy}`",
            f"- Affected spatial units: `{metadata.missing_data_report.get('affected_spatial_units_count', 0)}`",
            "",
            "## Normalization Summary",
            "",
        ]
    )

    for key, config in metadata.normalization.items():
        lines.append(
            f"- `{key}` / `{config.get('canonical_name')}`: `{config.get('method')}`"
        )

    lines.extend(
        [
            "",
            "## Aggregation Summary",
            "",
            f"- Method: `{metadata.aggregation_method}`",
            f"- Weighting: `{metadata.weighting_method}`",
            "",
        ]
    )

    if metadata.index_name == "svi_like":
        percentile_summary = svi_extra.get("overall_percentile_summary", {})
        domain_summary = svi_extra.get("domain_summary", {})
        lines.extend(
            [
                "## SVI Distribution",
                "",
                f"- Minimum overall percentile: `{percentile_summary.get('min')}`",
                f"- Mean overall percentile: `{percentile_summary.get('mean')}`",
                f"- Maximum overall percentile: `{percentile_summary.get('max')}`",
                "",
                "## SVI Domain Summary",
                "",
            ]
        )
        for domain_name, summary in domain_summary.items():
            lines.append(
                f"- `{domain_name}`: mean percentile `{summary.get('mean_percentile')}`"
            )
        lines.extend(
            [
                "",
                "## Interpretation Warning",
                "",
                svi_extra.get(
                    "area_level_warning",
                    "SVI is an area-level index and should not be interpreted at the individual level.",
                ),
                "",
            ]
        )
    if metadata.index_name == "sovi_like":
        standardization = metadata.extra.get("standardization", {})
        factor_retention = metadata.extra.get("factor_retention", {})
        factor_analysis = metadata.extra.get("factor_analysis", {})
        rotation = metadata.extra.get("rotation", {})
        score_summary = sovi_extra.get("score_summary", {})
        dominant = sovi_extra.get("dominant_variables_by_factor", [])
        retained_variance = sovi_extra.get("retained_cumulative_explained_variance")
        lines.extend(
            [
                "## SoVI Factor Analysis",
                "",
                f"- Standardization: `{standardization.get('method', 'zscore')}`",
                f"- PCA method: `{factor_analysis.get('method', 'pca')}`",
                f"- Factor-retention rule: `{factor_retention.get('method')}`",
                f"- Retained factors: `{factor_retention.get('n_factors_retained')}`",
                f"- Retained explained variance: `{retained_variance}`",
                f"- Rotation: `{rotation.get('method')}`",
                "",
                "## SoVI Factor Orientation Summary",
                "",
            ]
        )
        for factor_name, config in metadata.orientation.items():
            lines.append(
                f"- `{factor_name}`: `{config.get('method')}`; {config.get('rationale')}"
            )
        lines.extend(["", "## SoVI Dominant Variables", ""])
        for item in dominant[:10]:
            lines.append(
                f"- `{item.get('factor')}`: {item.get('dominant_variables_by_abs_loading')}"
            )
        lines.extend(
            [
                "",
                "## SoVI Score Distribution",
                "",
                f"- Minimum score: `{score_summary.get('min')}`",
                f"- Mean score: `{score_summary.get('mean')}`",
                f"- Maximum score: `{score_summary.get('max')}`",
                f"- Standard deviation: `{score_summary.get('std')}`",
                "",
                "## Interpretation Warning",
                "",
                sovi_extra.get(
                    "area_level_warning",
                    "SoVI is an area-level index and should not be interpreted at the individual level.",
                ),
                "",
            ]
        )

    lines.extend(["## Output Files", ""])
    for label, path in output_files.items():
        lines.append(f"- `{label}`: `{path}`")

    lines.extend(["", "## Top 5 Highest-Score Zones", ""])
    lines.append(_markdown_table(top, ["zone_id", "score_raw", "rank"]))
    lines.extend(["", "## Top 5 Lowest-Score Zones", ""])
    lines.append(_markdown_table(bottom, ["zone_id", "score_raw", "rank"]))
    lines.append("")
    return "\n".join(lines)


def write_run_report(
    *,
    path: str | Path,
    metadata: RunMetadata,
    standard_output: pd.DataFrame,
    output_files: dict[str, str],
) -> None:
    Path(path).write_text(
        generate_run_report(
            metadata=metadata,
            standard_output=standard_output,
            output_files=output_files,
        ),
        encoding="utf-8",
    )
