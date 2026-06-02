from pathlib import Path
import pandas as pd


# ============================================================
# Clean Census Division Voter Turnout / Vote Proxy 2021
# ============================================================
#
# Purpose:
#   Spatially allocate Québec federal electoral district vote-proxy variables
#   to Québec census divisions.
#
# Inputs:
#   1. Federal electoral district boundaries, 2013 Representation Order:
#      census_division_voter_turnout_2021/raw/lfed000a16a_e/lfed000a16a_e.shp
#
#   2. Clean Québec federal electoral district vote proxy:
#      census_division_voter_turnout_2021/output/clean_quebec_federal_district_vote_proxy_2021.csv
#
#   3. Québec census-division base frame with geometry:
#      census_division_spatial_frame_population_2021/output/
#      clean_quebec_census_division_spatial_frame_with_population_2021.gpkg
#
# Outputs:
#   census_division_voter_turnout_2021/output/clean_census_division_voter_turnout_2021.csv
#   census_division_voter_turnout_2021/output/clean_census_division_voter_turnout_2021.parquet
#   census_division_voter_turnout_2021/output/clean_census_division_voter_turnout_2021.geojson
#   census_division_voter_turnout_2021/output/census_division_voter_turnout_spatial_intersections_2021.csv
#   census_division_voter_turnout_2021/output/clean_census_division_voter_turnout_summary_2021.csv
#   census_division_voter_turnout_2021/output/clean_census_division_voter_turnout_variable_metadata_2021.csv
#
# Method:
#   Area-weighted allocation.
#
#   For each census division:
#
#       value_cd = sum(value_fed * overlap_area_cd_fed)
#                  / sum(overlap_area_cd_fed)
#
#   This is a first-pass spatial proxy. It is not population-weighted.
#
# Run from data/:
#   python census_division_voter_turnout_2021/clean_census_division_voter_turnout_2021.py
#
# ============================================================


# -----------------------------
# Paths
# -----------------------------

DATA_DIR = Path(__file__).resolve().parent.parent

SECTION_DIR = DATA_DIR / "census_division_voter_turnout_2021"
RAW_DIR = SECTION_DIR / "raw"
OUTPUT_DIR = SECTION_DIR / "output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

FED_SHAPEFILE = RAW_DIR / "lfed000a16a_e" / "lfed000a16a_e.shp"

FED_VOTE_PROXY_CSV = OUTPUT_DIR / "clean_quebec_federal_district_vote_proxy_2021.csv"

BASE_CD_CANDIDATES = [
    DATA_DIR
    / "census_division_spatial_frame_population_2021"
    / "output"
    / "clean_quebec_census_division_spatial_frame_with_population_2021.gpkg",
    DATA_DIR
    / "census_division_spatial_frame_population_2021"
    / "output"
    / "clean_quebec_census_division_spatial_frame_with_population_2021.geojson",
    DATA_DIR
    / "census_division_spatial_frame_population_2021"
    / "output"
    / "clean_quebec_census_division_spatial_frame_with_population_2021.parquet",
]

OUTPUT_CSV = OUTPUT_DIR / "clean_census_division_voter_turnout_2021.csv"
OUTPUT_PARQUET = OUTPUT_DIR / "clean_census_division_voter_turnout_2021.parquet"
OUTPUT_GEOJSON = OUTPUT_DIR / "clean_census_division_voter_turnout_2021.geojson"
OUTPUT_INTERSECTIONS = OUTPUT_DIR / "census_division_voter_turnout_spatial_intersections_2021.csv"
OUTPUT_SUMMARY = OUTPUT_DIR / "clean_census_division_voter_turnout_summary_2021.csv"
OUTPUT_METADATA = OUTPUT_DIR / "clean_census_division_voter_turnout_variable_metadata_2021.csv"


# -----------------------------
# Config
# -----------------------------

EXPECTED_QC_CD_COUNT = 98
EXPECTED_QC_FED_COUNT = 78

TARGET_CRS = "EPSG:3347"

FED_VALUE_COLUMNS = [
    "voter_turnout_pct_federal_2021",
    "pct_vote_leading_candidate_federal_2021",
    "pct_vote_leading_party_federal_2021",
    "majority_pct",
]

FED_COUNT_COLUMNS_FOR_DIAGNOSTICS = [
    "registered_electors",
    "ballots_cast",
    "valid_ballots",
    "rejected_ballots",
    "leading_candidate_votes",
    "district_valid_votes_from_candidate_sum",
]

CD_IDENTITY_COLUMNS = [
    "census_division_code",
    "census_division_dguid",
    "census_division_name",
    "census_division_type",
    "province_code",
    "province_name",
    "geography_level",
    "census_year",
    "population_total_2021",
    "land_area_km2",
    "has_positive_population",
]


# -----------------------------
# Helpers
# -----------------------------

def safe_relative(path: Path | None) -> str:
    if path is None:
        return ""

    try:
        return str(path.relative_to(DATA_DIR))
    except ValueError:
        return str(path)


def first_existing_path(paths: list[Path]) -> Path | None:
    for path in paths:
        if path.exists():
            return path
    return None


def clean_text(series: pd.Series) -> pd.Series:
    return series.astype("string").str.strip()


def clean_numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(
        series.astype("string")
        .str.replace("\u00a0", "", regex=False)
        .str.replace(",", "", regex=False)
        .str.replace("%", "", regex=False)
        .str.replace("$", "", regex=False)
        .str.strip(),
        errors="coerce",
    )


def numeric_summary(series: pd.Series) -> dict:
    numeric = clean_numeric(series)

    return {
        "non_missing": int(numeric.notna().sum()),
        "missing": int(numeric.isna().sum()),
        "min": numeric.min(skipna=True),
        "max": numeric.max(skipna=True),
        "mean": numeric.mean(skipna=True),
        "median": numeric.median(skipna=True),
    }


def make_valid_geometries(gdf):
    out = gdf.copy()

    invalid_count = int((~out.geometry.is_valid).sum())
    if invalid_count == 0:
        return out, invalid_count

    try:
        from shapely.validation import make_valid
        out["geometry"] = out.geometry.apply(
            lambda geom: make_valid(geom) if geom is not None and not geom.is_valid else geom
        )
    except Exception:
        out["geometry"] = out.geometry.buffer(0)

    return out, invalid_count


def read_spatial_file(path: Path):
    import geopandas as gpd

    suffix = path.suffix.lower()

    if suffix in [".gpkg", ".geojson", ".shp"]:
        return gpd.read_file(path)

    if suffix == ".parquet":
        return gpd.read_parquet(path)

    raise ValueError(f"Unsupported spatial file format: {path}")


def weighted_mean(group: pd.DataFrame, value_col: str, weight_col: str) -> float:
    values = clean_numeric(group[value_col])
    weights = clean_numeric(group[weight_col])

    mask = values.notna() & weights.notna() & (weights > 0)

    if not mask.any():
        return pd.NA

    return float((values[mask] * weights[mask]).sum() / weights[mask].sum())


def weighted_sum_allocated_count(group: pd.DataFrame, value_col: str, fed_area_col: str, overlap_area_col: str) -> float:
    """
    Diagnostic-only approximate count allocation.

    A federal district count is apportioned into census divisions by:

        count_fed * overlap_area / fed_area

    This is not used as the main SoVI variable, but it is useful for audit.
    """

    values = clean_numeric(group[value_col])
    fed_area = clean_numeric(group[fed_area_col])
    overlap_area = clean_numeric(group[overlap_area_col])

    mask = values.notna() & fed_area.notna() & overlap_area.notna() & (fed_area > 0) & (overlap_area > 0)

    if not mask.any():
        return pd.NA

    return float((values[mask] * overlap_area[mask] / fed_area[mask]).sum())


# -----------------------------
# Validate inputs
# -----------------------------

if not FED_SHAPEFILE.exists():
    raise FileNotFoundError(f"Missing federal electoral district shapefile:\n{FED_SHAPEFILE}")

if not FED_VOTE_PROXY_CSV.exists():
    raise FileNotFoundError(
        "Missing clean federal district vote proxy table. "
        "Run clean_federal_district_vote_proxy_2021.py first.\n"
        f"Expected:\n{FED_VOTE_PROXY_CSV}"
    )

base_cd_path = first_existing_path(BASE_CD_CANDIDATES)

if base_cd_path is None:
    raise FileNotFoundError(
        "Could not find Québec census-division base frame with geometry. Expected one of:\n"
        + "\n".join(str(path) for path in BASE_CD_CANDIDATES)
    )

try:
    import geopandas as gpd
except ImportError as exc:
    raise ImportError("geopandas is required for this spatial allocation script.") from exc


# -----------------------------
# Load inputs
# -----------------------------

fed_boundaries = gpd.read_file(FED_SHAPEFILE)
fed_votes = pd.read_csv(FED_VOTE_PROXY_CSV, dtype=str, low_memory=False)
cd = read_spatial_file(base_cd_path)

fed_boundaries.columns = [str(col).strip() for col in fed_boundaries.columns]
fed_votes.columns = [str(col).strip() for col in fed_votes.columns]
cd.columns = [str(col).strip() for col in cd.columns]

print("\nLoaded inputs")
print("FED boundaries:", safe_relative(FED_SHAPEFILE), "rows:", len(fed_boundaries))
print("FED vote proxy:", safe_relative(FED_VOTE_PROXY_CSV), "rows:", len(fed_votes))
print("CD base frame:", safe_relative(base_cd_path), "rows:", len(cd))


# -----------------------------
# Validate required columns
# -----------------------------

required_fed_boundary_cols = [
    "FEDUID",
    "FEDNAME",
    "FEDENAME",
    "FEDFNAME",
    "PRUID",
    "PRNAME",
    "geometry",
]

missing_fed_boundary_cols = [
    col for col in required_fed_boundary_cols
    if col not in fed_boundaries.columns
]

if missing_fed_boundary_cols:
    raise ValueError(
        "FED boundary file missing required columns:\n"
        + "\n".join(missing_fed_boundary_cols)
        + "\n\nAvailable columns:\n"
        + "\n".join(fed_boundaries.columns)
    )

required_fed_vote_cols = [
    "federal_electoral_district_id",
    "federal_electoral_district_name",
    "voter_turnout_pct_federal_2021",
    "pct_vote_leading_candidate_federal_2021",
    "pct_vote_leading_party_federal_2021",
]

missing_fed_vote_cols = [
    col for col in required_fed_vote_cols
    if col not in fed_votes.columns
]

if missing_fed_vote_cols:
    raise ValueError(
        "FED vote proxy table missing required columns:\n"
        + "\n".join(missing_fed_vote_cols)
        + "\n\nAvailable columns:\n"
        + "\n".join(fed_votes.columns)
    )

required_cd_cols = [
    "census_division_code",
    "census_division_dguid",
    "census_division_name",
    "geometry",
]

missing_cd_cols = [
    col for col in required_cd_cols
    if col not in cd.columns
]

if missing_cd_cols:
    raise ValueError(
        "Census-division base frame missing required columns:\n"
        + "\n".join(missing_cd_cols)
        + "\n\nAvailable columns:\n"
        + "\n".join(cd.columns)
    )


# -----------------------------
# Normalize identifiers
# -----------------------------

fed_boundaries["FEDUID"] = clean_text(fed_boundaries["FEDUID"])
fed_boundaries["FEDNAME"] = clean_text(fed_boundaries["FEDNAME"])
fed_boundaries["PRUID"] = clean_text(fed_boundaries["PRUID"])

fed_votes["federal_electoral_district_id"] = clean_text(
    fed_votes["federal_electoral_district_id"]
)

fed_votes["federal_electoral_district_name"] = clean_text(
    fed_votes["federal_electoral_district_name"]
)

cd["census_division_code"] = clean_text(cd["census_division_code"])
cd["census_division_dguid"] = clean_text(cd["census_division_dguid"])
cd["census_division_name"] = clean_text(cd["census_division_name"])


# -----------------------------
# Filter Québec FEDs and join vote proxy
# -----------------------------

qc_fed = fed_boundaries[fed_boundaries["PRUID"] == "24"].copy()

if len(qc_fed) != EXPECTED_QC_FED_COUNT:
    raise ValueError(
        f"Expected {EXPECTED_QC_FED_COUNT} Québec FED boundaries, got {len(qc_fed)}."
    )

if len(fed_votes) != EXPECTED_QC_FED_COUNT:
    raise ValueError(
        f"Expected {EXPECTED_QC_FED_COUNT} Québec FED vote rows, got {len(fed_votes)}."
    )

if qc_fed["FEDUID"].duplicated().any():
    raise ValueError("Duplicate FEDUID values in Québec FED boundary file.")

if fed_votes["federal_electoral_district_id"].duplicated().any():
    raise ValueError("Duplicate federal_electoral_district_id values in FED vote proxy table.")

fed_vote_keep_cols = [
    col for col in [
        "federal_electoral_district_id",
        "federal_electoral_district_name",
        "voter_turnout_pct_federal_2021",
        "pct_vote_leading_candidate_federal_2021",
        "pct_vote_leading_party_federal_2021",
        "majority_pct",
        "registered_electors",
        "ballots_cast",
        "valid_ballots",
        "rejected_ballots",
        "leading_candidate_name",
        "leading_party_label_raw",
        "leading_party_label_english_or_first",
        "leading_party_label_french_or_second",
        "leading_candidate_votes",
        "district_valid_votes_from_candidate_sum",
        "so_vi_proxy_recommended",
        "so_vi_proxy_alternative",
        "method_note",
    ]
    if col in fed_votes.columns
]

fed_votes_small = fed_votes[fed_vote_keep_cols].copy()
fed_votes_small = fed_votes_small.rename(
    columns={"federal_electoral_district_id": "FEDUID"}
)

for col in FED_VALUE_COLUMNS + FED_COUNT_COLUMNS_FOR_DIAGNOSTICS:
    if col in fed_votes_small.columns:
        fed_votes_small[col] = clean_numeric(fed_votes_small[col])

qc_fed = qc_fed.merge(
    fed_votes_small,
    on="FEDUID",
    how="left",
    validate="one_to_one",
    indicator=True,
)

fed_vote_missing = int((qc_fed["_merge"] != "both").sum())

if fed_vote_missing != 0:
    missing = qc_fed[qc_fed["_merge"] != "both"][["FEDUID", "FEDNAME", "_merge"]]
    raise ValueError(
        "Some Québec FED boundaries did not match the vote proxy table:\n"
        + missing.to_string(index=False)
    )

qc_fed = qc_fed.drop(columns=["_merge"])


# -----------------------------
# Validate CD base
# -----------------------------

if len(cd) != EXPECTED_QC_CD_COUNT:
    raise ValueError(
        f"Expected {EXPECTED_QC_CD_COUNT} Québec census divisions, got {len(cd)}."
    )

if cd["census_division_dguid"].duplicated().any():
    dupes = cd[cd["census_division_dguid"].duplicated(keep=False)]
    raise ValueError(
        "Duplicate census_division_dguid values in CD base frame:\n"
        + dupes[["census_division_code", "census_division_dguid", "census_division_name"]].to_string(index=False)
    )


# -----------------------------
# CRS and geometry preparation
# -----------------------------

if qc_fed.crs is None:
    raise ValueError("FED boundary file has no CRS.")

if cd.crs is None:
    raise ValueError("Census-division base frame has no CRS.")

fed_original_crs = str(qc_fed.crs)
cd_original_crs = str(cd.crs)

qc_fed = qc_fed.to_crs(TARGET_CRS)
cd = cd.to_crs(TARGET_CRS)

qc_fed, fed_invalid_before = make_valid_geometries(qc_fed)
cd, cd_invalid_before = make_valid_geometries(cd)

qc_fed["fed_area_m2"] = qc_fed.geometry.area
cd["cd_area_m2"] = cd.geometry.area

# Remove empty geometries defensively.
qc_fed = qc_fed[~qc_fed.geometry.is_empty & qc_fed.geometry.notna()].copy()
cd = cd[~cd.geometry.is_empty & cd.geometry.notna()].copy()


# -----------------------------
# Spatial intersection
# -----------------------------

cd_identity_cols = [col for col in CD_IDENTITY_COLUMNS if col in cd.columns]

cd_for_overlay = cd[
    cd_identity_cols + [
        "census_division_dguid",
        "census_division_code",
        "census_division_name",
        "cd_area_m2",
        "geometry",
    ]
].copy()

# Avoid duplicate columns if identity cols already include required columns.
cd_for_overlay = cd_for_overlay.loc[:, ~cd_for_overlay.columns.duplicated()].copy()

fed_for_overlay_cols = [
    "FEDUID",
    "FEDNAME",
    "FEDENAME",
    "FEDFNAME",
    "fed_area_m2",
    "voter_turnout_pct_federal_2021",
    "pct_vote_leading_candidate_federal_2021",
    "pct_vote_leading_party_federal_2021",
    "majority_pct",
    "registered_electors",
    "ballots_cast",
    "valid_ballots",
    "rejected_ballots",
    "leading_candidate_name",
    "leading_party_label_english_or_first",
    "leading_candidate_votes",
    "district_valid_votes_from_candidate_sum",
    "geometry",
]

fed_for_overlay_cols = [col for col in fed_for_overlay_cols if col in qc_fed.columns]
fed_for_overlay = qc_fed[fed_for_overlay_cols].copy()

print("\nRunning spatial overlay...")
intersections = gpd.overlay(
    cd_for_overlay,
    fed_for_overlay,
    how="intersection",
    keep_geom_type=False,
)

intersections = intersections[
    intersections.geometry.notna() & ~intersections.geometry.is_empty
].copy()

intersections["overlap_area_m2"] = intersections.geometry.area

# Remove zero / microscopic invalid area rows.
intersections = intersections[intersections["overlap_area_m2"] > 0].copy()

if intersections.empty:
    raise ValueError("Spatial intersection produced no overlap rows.")

intersections["weight_within_census_division"] = (
    intersections["overlap_area_m2"]
    / intersections.groupby("census_division_dguid")["overlap_area_m2"].transform("sum")
)

intersections["weight_within_federal_district"] = (
    intersections["overlap_area_m2"]
    / intersections["fed_area_m2"]
)


# -----------------------------
# Aggregate to census divisions
# -----------------------------

allocated_rows = []

for cd_dguid, group in intersections.groupby("census_division_dguid", dropna=False):
    row = {
        "census_division_dguid": cd_dguid,
        "fed_overlap_count": int(group["FEDUID"].nunique()),
        "fed_overlap_area_m2": group["overlap_area_m2"].sum(),
        "fed_overlap_area_weight_sum": group["weight_within_census_division"].sum(),
        "dominant_federal_electoral_district_id": group.sort_values(
            "overlap_area_m2",
            ascending=False,
        )["FEDUID"].iloc[0],
        "dominant_federal_electoral_district_name": group.sort_values(
            "overlap_area_m2",
            ascending=False,
        )["FEDNAME"].iloc[0],
        "dominant_federal_electoral_district_area_share": group.sort_values(
            "overlap_area_m2",
            ascending=False,
        )["weight_within_census_division"].iloc[0],
    }

    for value_col in FED_VALUE_COLUMNS:
        if value_col in group.columns:
            out_col = f"{value_col}_area_weighted"
            row[out_col] = weighted_mean(
                group=group,
                value_col=value_col,
                weight_col="overlap_area_m2",
            )

    for count_col in FED_COUNT_COLUMNS_FOR_DIAGNOSTICS:
        if count_col in group.columns:
            out_col = f"{count_col}_area_allocated_diagnostic"
            row[out_col] = weighted_sum_allocated_count(
                group=group,
                value_col=count_col,
                fed_area_col="fed_area_m2",
                overlap_area_col="overlap_area_m2",
            )

    allocated_rows.append(row)

allocated = pd.DataFrame(allocated_rows)


# -----------------------------
# Join allocated features back to all census divisions
# -----------------------------

clean = cd.drop(columns=["geometry"]).copy()

clean = clean.merge(
    allocated,
    on="census_division_dguid",
    how="left",
    validate="one_to_one",
)

# Coverage diagnostics.
clean["census_division_area_m2"] = clean["cd_area_m2"]
clean["spatial_allocation_coverage_ratio"] = (
    clean["fed_overlap_area_m2"] / clean["census_division_area_m2"]
)

# Main SoVI aliases.
clean["pct_vote_leading_party"] = clean[
    "pct_vote_leading_party_federal_2021_area_weighted"
]

clean["pct_vote_leading_party_federal_2021"] = clean[
    "pct_vote_leading_party_federal_2021_area_weighted"
]

clean["pct_vote_leading_candidate_federal_2021"] = clean[
    "pct_vote_leading_candidate_federal_2021_area_weighted"
]

clean["voter_turnout_pct_federal_2021"] = clean[
    "voter_turnout_pct_federal_2021_area_weighted"
]

if "majority_pct_area_weighted" in clean.columns:
    clean["majority_pct_federal_2021"] = clean["majority_pct_area_weighted"]

clean["allocation_method"] = "area_weighted_fed_2013_to_cd_2021"
clean["source_geography"] = "Federal electoral district, 2013 Representation Order"
clean["target_geography"] = "Statistics Canada census division, 2021"
clean["election_year"] = 2021
clean["source_organization"] = "Elections Canada / Statistics Canada"
clean["source_boundary_file"] = safe_relative(FED_SHAPEFILE)
clean["source_vote_proxy_file"] = safe_relative(FED_VOTE_PROXY_CSV)
clean["source_cd_frame"] = safe_relative(base_cd_path)
clean["method_note"] = (
    "Area-weighted allocation from 2013 Representation Order federal electoral districts "
    "to 2021 Québec census divisions. The recommended SoVI proxy pct_vote_leading_party "
    "uses the federal district leading-party/candidate vote-share proxy from Elections Canada "
    "Table 12, spatially averaged by polygon overlap area. This is not population-weighted."
)

clean["voter_proxy_complete"] = clean["pct_vote_leading_party"].notna()


# -----------------------------
# Spatial output version
# -----------------------------

clean_geo = cd[["census_division_dguid", "geometry"]].merge(
    clean,
    on="census_division_dguid",
    how="left",
    validate="one_to_one",
)

clean_geo = gpd.GeoDataFrame(clean_geo, geometry="geometry", crs=TARGET_CRS)


# -----------------------------
# Save intersection audit
# -----------------------------

intersection_audit_cols = [
    "census_division_code",
    "census_division_dguid",
    "census_division_name",
    "FEDUID",
    "FEDNAME",
    "overlap_area_m2",
    "weight_within_census_division",
    "weight_within_federal_district",
    "voter_turnout_pct_federal_2021",
    "pct_vote_leading_candidate_federal_2021",
    "pct_vote_leading_party_federal_2021",
    "majority_pct",
]

intersection_audit_cols = [
    col for col in intersection_audit_cols
    if col in intersections.columns
]

intersections_audit = pd.DataFrame(
    intersections[intersection_audit_cols].drop(columns=["geometry"], errors="ignore")
)

intersections_audit.to_csv(OUTPUT_INTERSECTIONS, index=False, encoding="utf-8")


# -----------------------------
# Validation
# -----------------------------

if len(clean) != EXPECTED_QC_CD_COUNT:
    raise ValueError(
        f"Clean CD voter proxy output has {len(clean)} rows; expected {EXPECTED_QC_CD_COUNT}."
    )

if clean["census_division_dguid"].duplicated().any():
    dupes = clean[clean["census_division_dguid"].duplicated(keep=False)]
    raise ValueError(
        "Duplicate census_division_dguid rows in clean output:\n"
        + dupes[["census_division_code", "census_division_dguid", "census_division_name"]].to_string(index=False)
    )

missing_main = int(clean["pct_vote_leading_party"].isna().sum())

if missing_main != 0:
    missing = clean[clean["pct_vote_leading_party"].isna()][
        [
            "census_division_code",
            "census_division_dguid",
            "census_division_name",
        ]
    ]
    raise ValueError(
        f"Unexpected missing pct_vote_leading_party values: {missing_main}\n"
        + missing.to_string(index=False)
    )

# We allow coverage ratios to differ from 1 because of 2016/2021 boundary-year
# differences and because both boundary files are digital/full-extent products.
coverage_min = clean["spatial_allocation_coverage_ratio"].min(skipna=True)
coverage_max = clean["spatial_allocation_coverage_ratio"].max(skipna=True)


# -----------------------------
# Metadata
# -----------------------------

metadata_rows = [
    {
        "variable": "pct_vote_leading_party",
        "description": "Recommended SoVI voting proxy at census-division level",
        "source": "Area-weighted federal district leading-party/candidate vote-share proxy",
        "unit": "percent",
        "role": "recommended_sovi_pctvote_proxy",
        "notes": (
            "Maps the original SoVI PCTVOTE92 concept to a Canadian federal election proxy. "
            "The federal-district value is area-weighted into census divisions."
        ),
    },
    {
        "variable": "pct_vote_leading_party_federal_2021_area_weighted",
        "description": "Area-weighted leading-party/candidate vote-share proxy",
        "source": "Elections Canada Table 12 + FED 2013 boundary overlay",
        "unit": "percent",
        "role": "main_area_weighted_source_variable",
        "notes": "Uses parsed leading-party label from the winning candidate field.",
    },
    {
        "variable": "pct_vote_leading_candidate_federal_2021_area_weighted",
        "description": "Area-weighted leading-candidate vote share",
        "source": "Elections Canada Table 12 + FED 2013 boundary overlay",
        "unit": "percent",
        "role": "candidate_vote_share_audit_variable",
        "notes": "Numerically equivalent to leading-party proxy under one-leading-candidate-per-district logic.",
    },
    {
        "variable": "voter_turnout_pct_federal_2021_area_weighted",
        "description": "Area-weighted voter turnout percentage",
        "source": "Elections Canada Table 11 + FED 2013 boundary overlay",
        "unit": "percent",
        "role": "civic_participation_alternative_proxy",
        "notes": "Alternative civic participation proxy, not the recommended SoVI PCTVOTE92 mapping.",
    },
    {
        "variable": "fed_overlap_count",
        "description": "Number of federal electoral districts intersecting the census division",
        "source": "Spatial overlay",
        "unit": "count",
        "role": "allocation_diagnostic",
        "notes": "Higher values indicate more spatial mixing in the allocation.",
    },
    {
        "variable": "spatial_allocation_coverage_ratio",
        "description": "Total FED-CD overlap area divided by census-division geometry area",
        "source": "Spatial overlay",
        "unit": "ratio",
        "role": "coverage_diagnostic",
        "notes": "May differ from 1 because the source boundary years differ.",
    },
]

metadata = pd.DataFrame(metadata_rows)


# -----------------------------
# Summary
# -----------------------------

summary_rows = [
    {"metric": "fed_boundary_file", "value": safe_relative(FED_SHAPEFILE)},
    {"metric": "fed_vote_proxy_file", "value": safe_relative(FED_VOTE_PROXY_CSV)},
    {"metric": "base_cd_frame", "value": safe_relative(base_cd_path)},
    {"metric": "fed_original_crs", "value": fed_original_crs},
    {"metric": "cd_original_crs", "value": cd_original_crs},
    {"metric": "target_crs", "value": TARGET_CRS},
    {"metric": "fed_invalid_geometries_before_fix", "value": fed_invalid_before},
    {"metric": "cd_invalid_geometries_before_fix", "value": cd_invalid_before},
    {"metric": "qc_fed_rows", "value": len(qc_fed)},
    {"metric": "qc_cd_rows", "value": len(cd)},
    {"metric": "intersection_rows", "value": len(intersections)},
    {"metric": "clean_rows", "value": len(clean)},
    {"metric": "unique_census_divisions", "value": clean["census_division_dguid"].nunique()},
    {"metric": "pct_vote_leading_party_non_missing", "value": int(clean["pct_vote_leading_party"].notna().sum())},
    {"metric": "pct_vote_leading_party_missing", "value": int(clean["pct_vote_leading_party"].isna().sum())},
    {"metric": "voter_proxy_complete_all_rows", "value": bool(clean["voter_proxy_complete"].all())},
    {"metric": "fed_overlap_count_min", "value": clean["fed_overlap_count"].min(skipna=True)},
    {"metric": "fed_overlap_count_max", "value": clean["fed_overlap_count"].max(skipna=True)},
    {"metric": "fed_overlap_count_mean", "value": clean["fed_overlap_count"].mean(skipna=True)},
    {"metric": "dominant_fed_area_share_min", "value": clean["dominant_federal_electoral_district_area_share"].min(skipna=True)},
    {"metric": "dominant_fed_area_share_max", "value": clean["dominant_federal_electoral_district_area_share"].max(skipna=True)},
    {"metric": "spatial_allocation_coverage_ratio_min", "value": coverage_min},
    {"metric": "spatial_allocation_coverage_ratio_max", "value": coverage_max},
]

for variable in [
    "pct_vote_leading_party",
    "pct_vote_leading_candidate_federal_2021",
    "voter_turnout_pct_federal_2021",
    "majority_pct_federal_2021" if "majority_pct_federal_2021" in clean.columns else None,
]:
    if variable is None:
        continue

    stats = numeric_summary(clean[variable])

    for key, value in stats.items():
        summary_rows.append(
            {
                "metric": f"{variable}_{key}",
                "value": value,
            }
        )

summary_rows.append(
    {
        "metric": "recommended_next_step",
        "value": (
            "Review the clean census-division voter turnout output. If it looks valid, "
            "update sovi_2021/inspect_census_division_sovi_input_sources_2021.py so "
            "pct_vote_leading_party maps to census_division_voter_turnout_2021/output/"
            "clean_census_division_voter_turnout_2021.csv."
        ),
    }
)

summary = pd.DataFrame(summary_rows)


# -----------------------------
# Save outputs
# -----------------------------

clean.to_csv(OUTPUT_CSV, index=False, encoding="utf-8")

try:
    clean.to_parquet(OUTPUT_PARQUET, index=False)
    parquet_saved = True
except Exception as exc:
    parquet_saved = False
    print("\nWARNING: Could not save Parquet output.")
    print("Reason:", exc)

# GeoJSON must be WGS84 for web mapping.
clean_geo_wgs84 = clean_geo.to_crs(epsg=4326)
clean_geo_wgs84.to_file(OUTPUT_GEOJSON, driver="GeoJSON")

metadata.to_csv(OUTPUT_METADATA, index=False, encoding="utf-8")
summary.to_csv(OUTPUT_SUMMARY, index=False, encoding="utf-8")


# -----------------------------
# Console report
# -----------------------------

print("\n" + "=" * 72)
print("CLEAN CENSUS DIVISION VOTER TURNOUT / VOTE PROXY 2021")
print("=" * 72)

print("\nInputs:")
print("FED boundaries:", safe_relative(FED_SHAPEFILE))
print("FED vote proxy:", safe_relative(FED_VOTE_PROXY_CSV))
print("CD base frame:", safe_relative(base_cd_path))

print("\nCRS:")
print("FED original CRS:", fed_original_crs)
print("CD original CRS:", cd_original_crs)
print("Target CRS:", TARGET_CRS)

print("\nSpatial overlay:")
print("Québec FED rows:", len(qc_fed))
print("Québec CD rows:", len(cd))
print("Intersection rows:", len(intersections))

print("\nClean output:")
print("Rows:", len(clean))
print("Unique CDs:", clean["census_division_dguid"].nunique())
print("pct_vote_leading_party non-missing:", int(clean["pct_vote_leading_party"].notna().sum()))
print("pct_vote_leading_party missing:", int(clean["pct_vote_leading_party"].isna().sum()))
print("All complete:", bool(clean["voter_proxy_complete"].all()))

print("\nAllocation diagnostics:")
print("FED overlap count:")
print(clean["fed_overlap_count"].describe())
print("\nDominant FED area share:")
print(clean["dominant_federal_electoral_district_area_share"].describe())
print("\nSpatial coverage ratio:")
print(clean["spatial_allocation_coverage_ratio"].describe())

print("\nMain variable summaries:")
for variable in [
    "pct_vote_leading_party",
    "pct_vote_leading_candidate_federal_2021",
    "voter_turnout_pct_federal_2021",
]:
    print("\n", variable)
    print(clean[variable].describe())

print("\nPreview:")
preview_cols = [
    "census_division_code",
    "census_division_name",
    "pct_vote_leading_party",
    "voter_turnout_pct_federal_2021",
    "fed_overlap_count",
    "dominant_federal_electoral_district_name",
    "dominant_federal_electoral_district_area_share",
    "spatial_allocation_coverage_ratio",
]
preview_cols = [col for col in preview_cols if col in clean.columns]
print(clean[preview_cols].head(25).to_string(index=False))

print("\nSaved:")
print(OUTPUT_CSV)
if parquet_saved:
    print(OUTPUT_PARQUET)
print(OUTPUT_GEOJSON)
print(OUTPUT_INTERSECTIONS)
print(OUTPUT_METADATA)
print(OUTPUT_SUMMARY)

print("\nDone.")