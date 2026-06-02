"""Command-line runner for index benchmark runs."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from ville_indices.core.metadata import build_run_metadata
from ville_indices.core.recipe import load_recipe
from ville_indices.core.registry import get_index_class
from ville_indices.reporting.run_report import write_run_report


def run_index(
    recipe_path: str | Path = "recipes/dummy_index.yaml",
    feature_table_path: str | Path = "data/example/synthetic_feature_table.csv",
    output_dir: str | Path = "outputs/dummy_run",
    index_name: str | None = None,
) -> dict[str, Path]:
    """Run an index recipe against a canonical feature table."""

    # Import built-in index modules so they register themselves.
    import ville_indices.indices  # noqa: F401

    recipe_path = Path(recipe_path)
    feature_table_path = Path(feature_table_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    recipe = load_recipe(recipe_path)
    feature_table = pd.read_csv(feature_table_path)
    if index_name is not None and index_name != recipe.name:
        raise ValueError(
            f"Requested index '{index_name}' does not match recipe name '{recipe.name}'."
        )
    index_class = get_index_class(index_name or recipe.name)
    index = index_class(recipe)
    standard_output = index.fit_transform(feature_table)
    if index.intermediate_output is None:
        raise RuntimeError("Index did not produce intermediate output.")
    if index.validation_report is None:
        raise RuntimeError("Index did not produce a validation report.")
    if index.missing_report is None:
        raise RuntimeError("Index did not produce a missing-data report.")

    output_files = {
        "standard_output": str(output_dir / "standard_output.csv"),
        "intermediate_output": str(output_dir / "intermediate_output.csv"),
        "metadata_json": str(output_dir / "metadata.json"),
        "metadata_yaml": str(output_dir / "metadata.yaml"),
        "validation_report_json": str(output_dir / "validation_report.json"),
        "validation_report_yaml": str(output_dir / "validation_report.yaml"),
        "missing_data_report_json": str(output_dir / "missing_data_report.json"),
        "run_report": str(output_dir / "run_report.md"),
    }
    for label, artifact in getattr(index, "extra_outputs", {}).items():
        output_files[label] = str(output_dir / f"{label}.csv")

    standard_output.to_csv(output_files["standard_output"], index=False)
    index.intermediate_output.to_csv(output_files["intermediate_output"], index=False)
    for label, artifact in getattr(index, "extra_outputs", {}).items():
        if hasattr(artifact, "to_csv"):
            artifact.to_csv(output_files[label], index=False)
        else:
            raise TypeError(f"Extra output '{label}' is not a DataFrame-like object.")
    index.validation_report.to_json(output_files["validation_report_json"])
    index.validation_report.to_yaml(output_files["validation_report_yaml"])
    index.missing_report.to_json(output_files["missing_data_report_json"])

    metadata = build_run_metadata(
        recipe=recipe,
        run_id=index.run_id,
        recipe_path=recipe_path,
        input_feature_table_path=feature_table_path,
        number_input_units=len(feature_table),
        number_output_units=len(standard_output),
        validation_report=index.validation_report,
        missing_report=index.missing_report,
        process_metadata=index.process_metadata,
        output_files=output_files,
    )
    metadata.to_json(output_files["metadata_json"])
    metadata.to_yaml(output_files["metadata_yaml"])
    write_run_report(
        path=output_files["run_report"],
        metadata=metadata,
        standard_output=standard_output,
        output_files=output_files,
    )
    return {key: Path(path) for key, path in output_files.items()}


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Run a VILLE composite-index recipe.")
    parser.add_argument("--index", required=False, help="Optional registered index name.")
    parser.add_argument("--recipe", required=True, help="Path to index recipe YAML.")
    parser.add_argument("--features", required=True, help="Path to canonical feature table CSV.")
    parser.add_argument("--output-dir", required=True, help="Directory for benchmark outputs.")
    args = parser.parse_args(argv)
    outputs = run_index(
        index_name=args.index,
        recipe_path=args.recipe,
        feature_table_path=args.features,
        output_dir=args.output_dir,
    )
    print("Generated outputs:")
    for label, path in outputs.items():
        print(f"- {label}: {path}")


if __name__ == "__main__":
    main()
