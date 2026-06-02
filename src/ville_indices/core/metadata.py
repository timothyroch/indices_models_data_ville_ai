"""Run metadata models and export helpers."""

from __future__ import annotations

import hashlib
import subprocess
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from ville_indices.core.recipe import Recipe
from ville_indices.core.serialization import to_jsonable, write_json


def new_run_id(prefix: str | None = None) -> str:
    token = uuid.uuid4().hex[:12]
    return f"{prefix}-{token}" if prefix else token


def file_sha256(path: str | Path | None) -> str | None:
    if path is None:
        return None
    file_path = Path(path)
    if not file_path.exists():
        return None
    digest = hashlib.sha256()
    with file_path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()


def code_version() -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
    except Exception:
        return None
    return result.stdout.strip()


@dataclass
class RunMetadata:
    index_name: str
    index_version: str
    run_id: str
    created_at: str
    code_version: str | None
    recipe_path: str | None
    recipe_hash: str | None
    input_feature_table_path: str | None
    spatial_id_column: str
    number_input_units: int
    number_output_units: int
    variables_requested: list[str]
    variables_used: list[str]
    variables_missing: list[str]
    optional_variables_missing: list[str]
    missing_data_strategy: str
    missing_data_report: dict[str, Any]
    normalization: dict[str, Any]
    orientation: dict[str, Any]
    weighting_method: str | None
    aggregation_method: str | None
    classification_method: str | None
    score_direction: str
    construct_measured: str
    reproduction_level: str
    assumptions: list[dict[str, Any] | str] = field(default_factory=list)
    decisions: list[dict[str, Any]] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    validation_summary: dict[str, Any] = field(default_factory=dict)
    output_files: dict[str, str] = field(default_factory=dict)
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return to_jsonable(asdict(self))

    def to_json(self, path: str | Path) -> None:
        write_json(self.to_dict(), path)

    def to_yaml(self, path: str | Path) -> None:
        with Path(path).open("w", encoding="utf-8") as handle:
            yaml.safe_dump(self.to_dict(), handle, sort_keys=False)


def build_run_metadata(
    *,
    recipe: Recipe,
    run_id: str,
    number_input_units: int,
    number_output_units: int,
    recipe_path: str | Path | None = None,
    input_feature_table_path: str | Path | None = None,
    validation_report: Any | None = None,
    missing_report: Any | None = None,
    process_metadata: dict[str, Any] | None = None,
    output_files: dict[str, str] | None = None,
) -> RunMetadata:
    process_metadata = process_metadata or {}
    validation_dict = validation_report.to_dict() if validation_report is not None else {}
    missing_dict = missing_report.to_dict() if missing_report is not None else {}
    warnings = [
        issue["message"]
        for issue in validation_dict.get("issues", [])
        if issue.get("severity") == "warning"
    ]
    warnings.extend(process_metadata.get("warnings", []))
    extra = {
        "aggregation": process_metadata.get("aggregation", {}),
        "classification": process_metadata.get("classification", {}),
    }
    for key in [
        "svi",
        "sovi",
        "standardization",
        "factor_analysis",
        "factor_retention",
        "rotation",
        "factor_scores",
        "ranking",
        "flags",
        "domains",
        "comparison",
        "population_filter",
        "variables_proxied",
        "variables_missing",
        "methodology_note",
        "source_reference",
        "population_column",
        "variables_used",
        "variables_dropped",
    ]:
        if key in process_metadata:
            extra[key] = process_metadata[key]
    for key in ["source_reference", "population_column", "comparison", "domains", "ranking", "flags"]:
        if key in recipe.extra and key not in extra:
            extra[key] = recipe.extra[key]
    return RunMetadata(
        index_name=recipe.name,
        index_version=recipe.version,
        run_id=run_id,
        created_at=datetime.now(timezone.utc).isoformat(),
        code_version=code_version(),
        recipe_path=str(recipe_path) if recipe_path is not None else None,
        recipe_hash=file_sha256(recipe_path),
        input_feature_table_path=(
            str(input_feature_table_path) if input_feature_table_path is not None else None
        ),
        spatial_id_column=recipe.spatial_id_column,
        number_input_units=int(number_input_units),
        number_output_units=int(number_output_units),
        variables_requested=recipe.variable_columns,
        variables_used=validation_dict.get("variables_present", []),
        variables_missing=validation_dict.get("variables_missing", []),
        optional_variables_missing=validation_dict.get("optional_variables_missing", []),
        missing_data_strategy=recipe.missing_data.get("strategy", "error"),
        missing_data_report=missing_dict,
        normalization=process_metadata.get("normalization", {}),
        orientation=process_metadata.get("orientation", {}),
        weighting_method=process_metadata.get("weighting_method"),
        aggregation_method=recipe.aggregation.get("method"),
        classification_method=recipe.classification.get("method"),
        score_direction=recipe.score_direction,
        construct_measured=recipe.construct_measured,
        reproduction_level=recipe.reproduction_level,
        assumptions=recipe.assumptions,
        decisions=recipe.decisions,
        warnings=warnings,
        validation_summary={
            "is_valid": validation_dict.get("is_valid"),
            "error_count": len(
                [
                    issue
                    for issue in validation_dict.get("issues", [])
                    if issue.get("severity") == "error"
                ]
            ),
            "warning_count": len(warnings),
            "missingness": validation_dict.get("missingness", {}),
        },
        output_files=output_files or {},
        extra=extra,
    )
