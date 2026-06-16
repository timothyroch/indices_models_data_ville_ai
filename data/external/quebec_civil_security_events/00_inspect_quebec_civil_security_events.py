#!/usr/bin/env python3
"""
Inspect Québec civil-security event data.

Default input:
    data/external/quebec_civil_security_events/raw/quebec_civil_security_events.json

Default output:
    data/external/quebec_civil_security_events/audits/

This is an inspection/audit script only. It does not clean, aggregate, spatially
join, construct SoVI, or use any 311/outcome label.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any, Mapping

import pandas as pd


DEFAULT_RAW_PATH = Path(
    "data/external/quebec_civil_security_events/raw/quebec_civil_security_events.json"
)
DEFAULT_OUTPUT_DIR = Path("data/external/quebec_civil_security_events/audits")

EXPECTED_CORE_COLUMNS = [
    "code_alea",
    "alea",
    "code_municipalite",
    "municipalite",
    "precision_localisation",
    "info_compl_localisation",
    "code_severite",
    "severite",
    "date_signalement",
    "date_debut",
    "date_debut_imprecise",
    "commentaire_date_debut",
    "date_fin",
    "coord_x",
    "coord_y",
]

DATE_COLUMNS = ["date_signalement", "date_debut", "date_fin"]

CATEGORICAL_COLUMNS = [
    "code_alea",
    "alea",
    "code_municipalite",
    "municipalite",
    "precision_localisation",
    "code_severite",
    "severite",
    "date_debut_imprecise",
    "_geometry_type",
]

# Loose WGS84 sanity bounds for Québec; used only for audit flags, not cleaning.
QUEBEC_LON_MIN = -80.5
QUEBEC_LON_MAX = -56.0
QUEBEC_LAT_MIN = 44.0
QUEBEC_LAT_MAX = 63.5


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Inspect Québec civil-security event JSON and write audit tables."
    )
    parser.add_argument("--raw-path", type=Path, default=DEFAULT_RAW_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--max-value-count-rows", type=int, default=500)
    parser.add_argument("--max-problem-rows", type=int, default=2000)
    parser.add_argument("--preview-rows", type=int, default=50)
    return parser.parse_args()


def read_json(path: Path) -> Any:
    if not path.exists():
        raise FileNotFoundError(f"Raw JSON file does not exist: {path}")
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def flatten_mapping(obj: Mapping[str, Any], prefix: str = "", sep: str = "__") -> dict[str, Any]:
    out: dict[str, Any] = {}

    def walk(value: Any, key_prefix: str, depth: int) -> None:
        if isinstance(value, Mapping) and depth < 5:
            for k, v in value.items():
                next_key = f"{key_prefix}{sep}{k}" if key_prefix else str(k)
                walk(v, next_key, depth + 1)
        elif isinstance(value, list):
            out[key_prefix] = json.dumps(value, ensure_ascii=False, default=str)
        else:
            out[key_prefix] = value

    walk(obj, prefix, 0)
    return out


def extract_geojson_feature(feature: Mapping[str, Any], index: int) -> dict[str, Any]:
    row: dict[str, Any] = {}
    props = feature.get("properties")
    if isinstance(props, Mapping):
        row.update(flatten_mapping(props))
    else:
        row["properties_raw"] = json.dumps(props, ensure_ascii=False, default=str)

    geom = feature.get("geometry")
    row["_feature_index"] = index
    row["_geometry_present"] = isinstance(geom, Mapping)
    row["_geometry_type"] = geom.get("type") if isinstance(geom, Mapping) else None

    if isinstance(geom, Mapping):
        coords = geom.get("coordinates")
        row["_geometry_coordinates_raw"] = json.dumps(coords, ensure_ascii=False, default=str)
        if geom.get("type") == "Point" and isinstance(coords, list) and len(coords) >= 2:
            row["_geometry_x"] = coords[0]
            row["_geometry_y"] = coords[1]

    return row


def extract_records(obj: Any) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    meta: dict[str, Any] = {
        "top_level_type": type(obj).__name__,
        "detected_format": None,
        "top_level_keys": sorted(map(str, obj.keys())) if isinstance(obj, Mapping) else None,
    }

    if isinstance(obj, Mapping) and obj.get("type") == "FeatureCollection":
        features = obj.get("features") or []
        if not isinstance(features, list):
            raise ValueError("GeoJSON FeatureCollection has non-list features.")
        meta["detected_format"] = "geojson_feature_collection"
        meta["feature_count"] = len(features)
        meta["top_level_metadata"] = {k: v for k, v in obj.items() if k != "features"}
        return [extract_geojson_feature(feat, i) for i, feat in enumerate(features)], meta

    if isinstance(obj, list):
        meta["detected_format"] = "json_list"
        rows = []
        for i, item in enumerate(obj):
            row = flatten_mapping(item) if isinstance(item, Mapping) else {"value": item}
            row["_record_index"] = i
            rows.append(row)
        return rows, meta

    if isinstance(obj, Mapping):
        for key in ["features", "records", "data", "results", "items"]:
            value = obj.get(key)
            if isinstance(value, list):
                meta["detected_format"] = f"dict_with_{key}"
                meta["record_container_key"] = key
                meta["record_count"] = len(value)
                rows = []
                for i, item in enumerate(value):
                    if key == "features" and isinstance(item, Mapping) and (
                        "properties" in item or "geometry" in item
                    ):
                        row = extract_geojson_feature(item, i)
                    elif isinstance(item, Mapping):
                        row = flatten_mapping(item)
                        row["_record_index"] = i
                    else:
                        row = {"value": item, "_record_index": i}
                    rows.append(row)
                meta["top_level_metadata"] = {k: v for k, v in obj.items() if k != key}
                return rows, meta
        meta["detected_format"] = "single_dict"
        return [flatten_mapping(obj)], meta

    raise ValueError(f"Unsupported JSON top-level type: {type(obj).__name__}")


def safe_to_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, quoting=csv.QUOTE_MINIMAL)


def safe_json_dump(data: Mapping[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False, default=str)


def value_sample(series: pd.Series, n: int = 8) -> str:
    values = (
        series.dropna()
        .astype(str)
        .replace({"": pd.NA})
        .dropna()
        .drop_duplicates()
        .head(n)
        .tolist()
    )
    return "; ".join(values)


def schema_audit(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    n = len(df)
    for col in df.columns:
        s = df[col]
        non_missing = int(s.notna().sum())
        missing = int(n - non_missing)
        rows.append(
            {
                "column": col,
                "dtype": str(s.dtype),
                "non_missing": non_missing,
                "missing": missing,
                "missing_rate": missing / n if n else math.nan,
                "n_unique": int(s.nunique(dropna=True)),
                "sample_values": value_sample(s),
                "is_expected_core_column": col in EXPECTED_CORE_COLUMNS,
                "is_date_candidate": col in DATE_COLUMNS or "date" in col.lower(),
                "is_coordinate_candidate": col in {"coord_x", "coord_y", "_geometry_x", "_geometry_y"},
            }
        )
    return pd.DataFrame(rows)


def value_count_audit(df: pd.DataFrame, col: str, limit: int) -> pd.DataFrame:
    counts = (
        df[col]
        .astype("string")
        .fillna("<MISSING>")
        .replace("", "<EMPTY>")
        .value_counts(dropna=False)
        .head(limit)
        .rename_axis(col)
        .reset_index(name="n")
    )
    counts["share"] = counts["n"] / len(df) if len(df) else math.nan
    return counts


def parse_date_column(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series, errors="coerce", utc=False)


def date_audit(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    date_cols = [c for c in DATE_COLUMNS if c in df.columns]
    for col in df.columns:
        if "date" in col.lower() and col not in date_cols:
            date_cols.append(col)

    coverage_rows = []
    monthly_rows = []
    problem_rows = []

    for col in date_cols:
        raw_non_missing = int(df[col].notna().sum())
        parsed = parse_date_column(df[col])
        parsed_non_missing = int(parsed.notna().sum())
        coverage_rows.append(
            {
                "column": col,
                "raw_non_missing": raw_non_missing,
                "parsed_non_missing": parsed_non_missing,
                "n_failed_parse": raw_non_missing - parsed_non_missing,
                "parse_success_rate_among_non_missing": parsed_non_missing / raw_non_missing if raw_non_missing else math.nan,
                "min_date": parsed.min(),
                "max_date": parsed.max(),
            }
        )

        if parsed.notna().any():
            month_counts = (
                parsed.dropna()
                .dt.to_period("M")
                .astype(str)
                .value_counts()
                .sort_index()
                .rename_axis("period_month")
                .reset_index(name="n")
            )
            month_counts["date_column"] = col
            monthly_rows.append(month_counts)

        failed = df[df[col].notna() & parsed.isna()].copy()
        if not failed.empty:
            failed = failed[[col]].head(200).copy()
            failed["date_column"] = col
            problem_rows.append(failed)

    coverage = pd.DataFrame(coverage_rows)
    monthly = pd.concat(monthly_rows, ignore_index=True) if monthly_rows else pd.DataFrame()
    problems = pd.concat(problem_rows, ignore_index=True) if problem_rows else pd.DataFrame()
    return coverage, monthly, problems


def coordinate_audit(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    candidate_pairs = [
        ("coord_x", "coord_y", "declared_coord_columns"),
        ("_geometry_x", "_geometry_y", "geojson_point_geometry"),
    ]
    summary_rows = []
    flagged_rows = []

    for x_col, y_col, source in candidate_pairs:
        if x_col not in df.columns or y_col not in df.columns:
            summary_rows.append({"source": source, "x_col": x_col, "y_col": y_col, "columns_present": False})
            continue

        x = pd.to_numeric(df[x_col], errors="coerce")
        y = pd.to_numeric(df[y_col], errors="coerce")
        valid_pair = x.notna() & y.notna()
        in_quebec_bbox = (
            valid_pair
            & (x >= QUEBEC_LON_MIN)
            & (x <= QUEBEC_LON_MAX)
            & (y >= QUEBEC_LAT_MIN)
            & (y <= QUEBEC_LAT_MAX)
        )
        possible_swapped = (
            valid_pair
            & (y >= QUEBEC_LON_MIN)
            & (y <= QUEBEC_LON_MAX)
            & (x >= QUEBEC_LAT_MIN)
            & (x <= QUEBEC_LAT_MAX)
        )
        zero_zero = valid_pair & (x.abs() < 1e-12) & (y.abs() < 1e-12)
        outside = valid_pair & ~in_quebec_bbox

        summary_rows.append(
            {
                "source": source,
                "x_col": x_col,
                "y_col": y_col,
                "columns_present": True,
                "n_rows": int(len(df)),
                "n_valid_pairs": int(valid_pair.sum()),
                "valid_pair_rate": float(valid_pair.mean()) if len(df) else math.nan,
                "n_inside_loose_quebec_bbox": int(in_quebec_bbox.sum()),
                "inside_loose_quebec_bbox_rate_among_valid": float(in_quebec_bbox.sum() / valid_pair.sum()) if valid_pair.sum() else math.nan,
                "n_outside_loose_quebec_bbox": int(outside.sum()),
                "n_possible_swapped_lon_lat": int(possible_swapped.sum()),
                "n_zero_zero": int(zero_zero.sum()),
                "x_min": float(x.min()) if x.notna().any() else math.nan,
                "x_max": float(x.max()) if x.notna().any() else math.nan,
                "y_min": float(y.min()) if y.notna().any() else math.nan,
                "y_max": float(y.max()) if y.notna().any() else math.nan,
            }
        )

        flags = pd.DataFrame(
            {
                "row_index": df.index,
                "source": source,
                "x_col": x_col,
                "y_col": y_col,
                "x": x,
                "y": y,
                "valid_pair": valid_pair,
                "inside_loose_quebec_bbox": in_quebec_bbox,
                "possible_swapped_lon_lat": possible_swapped,
                "zero_zero": zero_zero,
                "outside_loose_quebec_bbox": outside,
            }
        )
        flagged = flags[valid_pair & (outside | possible_swapped | zero_zero)].copy()
        if not flagged.empty:
            context_cols = [
                c for c in [
                    "alea", "severite", "precision_localisation", "municipalite",
                    "code_municipalite", "date_signalement", "date_debut", "date_fin",
                ] if c in df.columns
            ]
            flagged = flagged.join(df[context_cols])
            flagged_rows.append(flagged)

    return pd.DataFrame(summary_rows), pd.concat(flagged_rows, ignore_index=True) if flagged_rows else pd.DataFrame()


def duration_audit(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    if "date_debut" not in df.columns or "date_fin" not in df.columns:
        return pd.DataFrame(), pd.DataFrame()

    start = parse_date_column(df["date_debut"])
    end = parse_date_column(df["date_fin"])
    valid = start.notna() & end.notna()
    duration_days = (end - start).dt.total_seconds() / 86400.0

    summary = pd.DataFrame([
        {
            "n_rows": int(len(df)),
            "n_valid_start_end": int(valid.sum()),
            "n_open_or_missing_end": int(start.notna().sum() - valid.sum()),
            "n_negative_duration": int((valid & (duration_days < 0)).sum()),
            "duration_min_days": float(duration_days[valid].min()) if valid.any() else math.nan,
            "duration_p25_days": float(duration_days[valid].quantile(0.25)) if valid.any() else math.nan,
            "duration_median_days": float(duration_days[valid].median()) if valid.any() else math.nan,
            "duration_p75_days": float(duration_days[valid].quantile(0.75)) if valid.any() else math.nan,
            "duration_max_days": float(duration_days[valid].max()) if valid.any() else math.nan,
        }
    ])

    problem = pd.DataFrame({
        "row_index": df.index,
        "date_debut": df.get("date_debut"),
        "date_fin": df.get("date_fin"),
        "duration_days": duration_days,
        "negative_duration": valid & (duration_days < 0),
    })
    problem = problem[problem["negative_duration"]].copy()
    context_cols = [c for c in ["alea", "severite", "municipalite", "code_municipalite"] if c in df.columns]
    if not problem.empty:
        problem = problem.join(df[context_cols])
    return summary, problem


def core_field_quality_audit(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    for col in EXPECTED_CORE_COLUMNS:
        if col in df.columns:
            missing = df[col].isna() | (df[col].astype("string").fillna("").str.strip() == "")
            rows.append({
                "column": col,
                "present": True,
                "non_missing": int((~missing).sum()),
                "missing": int(missing.sum()),
                "missing_rate": float(missing.mean()) if len(df) else math.nan,
            })
        else:
            rows.append({
                "column": col,
                "present": False,
                "non_missing": 0,
                "missing": int(len(df)),
                "missing_rate": 1.0 if len(df) else math.nan,
            })

    present_core = [c for c in EXPECTED_CORE_COLUMNS if c in df.columns]
    problem = pd.DataFrame({"row_index": df.index})
    for col in present_core:
        problem[f"missing__{col}"] = df[col].isna() | (df[col].astype("string").fillna("").str.strip() == "")

    missing_cols = [c for c in problem.columns if c.startswith("missing__")]
    if missing_cols:
        problem["n_missing_core_fields"] = problem[missing_cols].sum(axis=1)
        problem = problem[problem["n_missing_core_fields"] > 0].copy()
        context_cols = [
            c for c in [
                "alea", "severite", "precision_localisation", "municipalite", "code_municipalite",
                "date_signalement", "date_debut", "date_fin", "coord_x", "coord_y",
            ] if c in df.columns
        ]
        if not problem.empty:
            problem = problem.join(df[context_cols])
    else:
        problem = pd.DataFrame()

    return pd.DataFrame(rows), problem


def duplicate_candidate_audit(df: pd.DataFrame) -> pd.DataFrame:
    keys = [
        c for c in [
            "code_alea", "alea", "code_municipalite", "municipalite",
            "date_signalement", "date_debut", "date_fin", "coord_x", "coord_y",
        ] if c in df.columns
    ]
    if not keys:
        return pd.DataFrame()
    tmp = df[keys].copy()
    for c in keys:
        tmp[c] = tmp[c].astype("string").fillna("<MISSING>")
    counts = tmp.value_counts(dropna=False).reset_index(name="n_duplicate_rows")
    return counts[counts["n_duplicate_rows"] > 1].sort_values("n_duplicate_rows", ascending=False)


def write_crosstabs(df: pd.DataFrame, out_dir: Path) -> None:
    pairs = [
        ("alea", "severite"),
        ("alea", "precision_localisation"),
        ("severite", "precision_localisation"),
    ]
    for left, right in pairs:
        if left in df.columns and right in df.columns:
            tab = pd.crosstab(
                df[left].astype("string").fillna("<MISSING>"),
                df[right].astype("string").fillna("<MISSING>"),
            ).reset_index()
            safe_to_csv(tab, out_dir / f"crosstab__{left}_by_{right}.csv")


def municipality_event_counts(df: pd.DataFrame) -> pd.DataFrame:
    if "code_municipalite" not in df.columns and "municipalite" not in df.columns:
        return pd.DataFrame()
    muni_col = "code_municipalite" if "code_municipalite" in df.columns else "municipalite"
    out = (
        df[muni_col].astype("string").fillna("<MISSING>")
        .value_counts(dropna=False)
        .rename_axis(muni_col)
        .reset_index(name="n_events")
    )
    if muni_col != "municipalite" and "municipalite" in df.columns:
        names = (
            df[[muni_col, "municipalite"]]
            .dropna()
            .astype("string")
            .drop_duplicates()
            .groupby(muni_col)["municipalite"]
            .apply(lambda s: "; ".join(sorted(set(s.astype(str)))[:5]))
            .reset_index(name="municipalite_examples")
        )
        out = out.merge(names, on=muni_col, how="left")
    return out


def write_readme(out_dir: Path, raw_path: Path, summary: Mapping[str, Any]) -> None:
    readme = f"""# Québec civil-security events inspection

Raw file inspected:

```text
{raw_path}
```

This directory contains inspection/audit outputs only. No cleaning, aggregation,
spatial join, SoVI construction, or target construction is performed here.

Detected format: `{summary.get('detected_format')}`
Rows: `{summary.get('rows')}`
Columns: `{summary.get('columns')}`

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
"""
    (out_dir / "README.md").write_text(readme, encoding="utf-8")


def main() -> None:
    args = parse_args()
    raw_path = Path(args.raw_path)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    raw_obj = read_json(raw_path)
    records, source_meta = extract_records(raw_obj)
    df = pd.DataFrame(records)
    df.columns = [str(c) for c in df.columns]
    df = df.reset_index(drop=True)

    if df.empty:
        raise ValueError(f"No records extracted from {raw_path}")

    safe_to_csv(df.head(args.preview_rows), out_dir / "normalized_preview.csv")
    safe_to_csv(schema_audit(df), out_dir / "schema_audit.csv")

    core_summary, core_problems = core_field_quality_audit(df)
    safe_to_csv(core_summary, out_dir / "core_field_quality_audit.csv")
    safe_to_csv(core_problems.head(args.max_problem_rows), out_dir / "row_core_missing_examples.csv")

    for col in CATEGORICAL_COLUMNS:
        if col in df.columns:
            safe_to_csv(value_count_audit(df, col, args.max_value_count_rows), out_dir / f"value_counts__{col}.csv")

    # Extra value counts for low-cardinality string columns not listed above.
    for col in df.columns:
        if col in CATEGORICAL_COLUMNS:
            continue
        nunique = df[col].nunique(dropna=True)
        if 1 < nunique <= 100 and (df[col].dtype == "object" or str(df[col].dtype).startswith("string")):
            safe_to_csv(value_count_audit(df, col, args.max_value_count_rows), out_dir / f"value_counts__{col}.csv")

    date_coverage, monthly_counts, date_problems = date_audit(df)
    safe_to_csv(date_coverage, out_dir / "date_coverage_audit.csv")
    safe_to_csv(monthly_counts, out_dir / "monthly_counts_by_date_column.csv")
    safe_to_csv(date_problems.head(args.max_problem_rows), out_dir / "date_parse_problem_examples.csv")

    duration_summary, negative_duration = duration_audit(df)
    safe_to_csv(duration_summary, out_dir / "duration_audit.csv")
    safe_to_csv(negative_duration.head(args.max_problem_rows), out_dir / "negative_duration_examples.csv")

    coord_summary, coord_problems = coordinate_audit(df)
    safe_to_csv(coord_summary, out_dir / "coordinate_audit.csv")
    safe_to_csv(coord_problems.head(args.max_problem_rows), out_dir / "coordinate_problem_examples.csv")

    dupes = duplicate_candidate_audit(df)
    safe_to_csv(dupes.head(args.max_problem_rows), out_dir / "duplicate_candidate_audit.csv")

    write_crosstabs(df, out_dir)

    muni = municipality_event_counts(df)
    safe_to_csv(muni, out_dir / "municipality_event_counts.csv")

    if "alea" in df.columns:
        safe_to_csv(value_count_audit(df, "alea", args.max_value_count_rows), out_dir / "hazard_alea_summary.csv")
    if "severite" in df.columns:
        safe_to_csv(value_count_audit(df, "severite", args.max_value_count_rows), out_dir / "severity_summary.csv")

    expected_present = {col: col in df.columns for col in EXPECTED_CORE_COLUMNS}
    summary = {
        "raw_path": str(raw_path),
        "output_dir": str(out_dir),
        "rows": int(len(df)),
        "columns": int(len(df.columns)),
        "source_metadata": source_meta,
        "detected_format": source_meta.get("detected_format"),
        "expected_core_columns_present": expected_present,
        "missing_expected_core_columns": [col for col, present in expected_present.items() if not present],
        "date_columns_detected": [col for col in df.columns if "date" in col.lower()],
        "coordinate_columns_detected": [
            col for col in df.columns
            if col in {"coord_x", "coord_y", "_geometry_x", "_geometry_y"} or "coord" in col.lower()
        ],
        "categorical_columns_detected": [col for col in CATEGORICAL_COLUMNS if col in df.columns],
        "inspection_config": {
            "max_value_count_rows": args.max_value_count_rows,
            "max_problem_rows": args.max_problem_rows,
            "preview_rows": args.preview_rows,
        },
    }
    safe_json_dump(summary, out_dir / "inspection_summary.json")
    write_readme(out_dir, raw_path, summary)

    print("Québec civil-security events inspection completed.")
    print(f"Raw file: {raw_path}")
    print(f"Rows: {len(df):,}")
    print(f"Columns: {len(df.columns):,}")
    print(f"Audit directory: {out_dir}")
    print("Core expected columns:")
    for col in EXPECTED_CORE_COLUMNS:
        print(f"  {'OK' if col in df.columns else 'MISSING'}  {col}")


if __name__ == "__main__":
    main()
