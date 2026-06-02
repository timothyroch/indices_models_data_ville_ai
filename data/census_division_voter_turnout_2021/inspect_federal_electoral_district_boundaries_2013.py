from pathlib import Path
import pandas as pd


# ============================================================
# Inspect Federal Electoral District Boundaries 2013
# ============================================================
#
# Purpose:
#   Inspect the Statistics Canada 2016 Census Federal Electoral District
#   boundary file based on the 2013 Representation Order.
#
# This file is needed to spatially allocate federal electoral district vote
# proxies to Québec census divisions.
#
# Inputs:
#   census_division_voter_turnout_2021/raw/lfed000a16a_e/lfed000a16a_e.shp
#   census_division_voter_turnout_2021/output/clean_quebec_federal_district_vote_proxy_2021.csv
#   census_division_spatial_frame_population_2021/output/clean_quebec_census_division_spatial_frame_with_population_2021.*
#
# Outputs:
#   census_division_voter_turnout_2021/output/federal_electoral_district_boundary_inventory_2013.csv
#   census_division_voter_turnout_2021/output/quebec_federal_electoral_district_boundary_inventory_2013.csv
#   census_division_voter_turnout_2021/output/quebec_federal_district_vote_boundary_join_audit_2021.csv
#   census_division_voter_turnout_2021/output/federal_electoral_district_boundary_inspection_summary_2013.csv
#   census_division_voter_turnout_2021/output/quebec_federal_electoral_district_boundaries_2013.geojson
#
# Run from data/:
#   python census_division_voter_turnout_2021/inspect_federal_electoral_district_boundaries_2013.py
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

VOTE_PROXY_CSV = OUTPUT_DIR / "clean_quebec_federal_district_vote_proxy_2021.csv"

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
    DATA_DIR
    / "census_division_spatial_frame_population_2021"
    / "output"
    / "clean_quebec_census_division_spatial_frame_with_population_2021.csv",
]

OUTPUT_FED_INVENTORY = OUTPUT_DIR / "federal_electoral_district_boundary_inventory_2013.csv"
OUTPUT_QC_FED_INVENTORY = OUTPUT_DIR / "quebec_federal_electoral_district_boundary_inventory_2013.csv"
OUTPUT_JOIN_AUDIT = OUTPUT_DIR / "quebec_federal_district_vote_boundary_join_audit_2021.csv"
OUTPUT_QC_BOUNDARIES_GEOJSON = OUTPUT_DIR / "quebec_federal_electoral_district_boundaries_2013.geojson"
OUTPUT_SUMMARY = OUTPUT_DIR / "federal_electoral_district_boundary_inspection_summary_2013.csv"


# -----------------------------
# Constants
# -----------------------------

EXPECTED_CANADA_FED_COUNT = 338
EXPECTED_QUEBEC_FED_COUNT = 78

EXPECTED_FED_COLUMNS = [
    "FEDUID",
    "FEDNAME",
    "FEDENAME",
    "FEDFNAME",
    "PRUID",
    "PRNAME",
]

EXPECTED_QC_PRUID = "24"

VOTE_PROXY_ID_CANDIDATES = [
    "federal_electoral_district_id",
    "FEDUID",
    "feduid",
]

VOTE_PROXY_NAME_CANDIDATES = [
    "federal_electoral_district_name",
    "FEDNAME",
    "fedname",
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


def find_first_existing_path(paths: list[Path]) -> Path | None:
    for path in paths:
        if path.exists():
            return path
    return None


def normalize_text(series: pd.Series) -> pd.Series:
    return series.astype("string").str.strip()


def find_first_existing_column(columns: list[str], candidates: list[str]) -> str | None:
    for col in candidates:
        if col in columns:
            return col
    return None


def read_base_cd_frame(path: Path):
    suffix = path.suffix.lower()

    if suffix in [".gpkg", ".geojson", ".shp"]:
        import geopandas as gpd
        return gpd.read_file(path)

    if suffix == ".parquet":
        try:
            import geopandas as gpd
            return gpd.read_parquet(path)
        except Exception:
            return pd.read_parquet(path)

    if suffix == ".csv":
        return pd.read_csv(path, low_memory=False)

    raise ValueError(f"Unsupported base frame format: {path}")


def geometry_area_summary(gdf, area_col: str) -> dict:
    return {
        f"{area_col}_non_missing": int(gdf[area_col].notna().sum()),
        f"{area_col}_min": gdf[area_col].min(skipna=True),
        f"{area_col}_max": gdf[area_col].max(skipna=True),
        f"{area_col}_mean": gdf[area_col].mean(skipna=True),
        f"{area_col}_median": gdf[area_col].median(skipna=True),
    }


# -----------------------------
# Load inputs
# -----------------------------

if not FED_SHAPEFILE.exists():
    raise FileNotFoundError(f"Missing FED shapefile:\n{FED_SHAPEFILE}")

if not VOTE_PROXY_CSV.exists():
    raise FileNotFoundError(
        "Missing cleaned Québec federal district vote proxy table. "
        "Run clean_federal_district_vote_proxy_2021.py first.\n"
        f"Expected:\n{VOTE_PROXY_CSV}"
    )

base_cd_path = find_first_existing_path(BASE_CD_CANDIDATES)
if base_cd_path is None:
    raise FileNotFoundError(
        "Could not find cleaned Québec census-division base frame. Expected one of:\n"
        + "\n".join(str(path) for path in BASE_CD_CANDIDATES)
    )

try:
    import geopandas as gpd
except ImportError as exc:
    raise ImportError(
        "geopandas is required for this boundary inspection script."
    ) from exc

fed = gpd.read_file(FED_SHAPEFILE)
vote_proxy = pd.read_csv(VOTE_PROXY_CSV, dtype=str, low_memory=False)
base_cd = read_base_cd_frame(base_cd_path)

print("\nLoaded inputs")
print("FED shapefile:", safe_relative(FED_SHAPEFILE))
print("FED rows:", len(fed))
print("Vote proxy:", safe_relative(VOTE_PROXY_CSV))
print("Vote proxy rows:", len(vote_proxy))
print("Base CD frame:", safe_relative(base_cd_path))
print("Base CD rows:", len(base_cd))


# -----------------------------
# Basic FED validation
# -----------------------------

fed.columns = [str(col).strip() for col in fed.columns]
vote_proxy.columns = [str(col).strip() for col in vote_proxy.columns]

missing_expected_cols = [col for col in EXPECTED_FED_COLUMNS if col not in fed.columns]

if missing_expected_cols:
    raise ValueError(
        "FED boundary file is missing expected columns:\n"
        + "\n".join(missing_expected_cols)
        + "\n\nAvailable columns:\n"
        + "\n".join(fed.columns)
    )

fed["FEDUID"] = normalize_text(fed["FEDUID"])
fed["FEDNAME"] = normalize_text(fed["FEDNAME"])
fed["FEDENAME"] = normalize_text(fed["FEDENAME"])
fed["FEDFNAME"] = normalize_text(fed["FEDFNAME"])
fed["PRUID"] = normalize_text(fed["PRUID"])
fed["PRNAME"] = normalize_text(fed["PRNAME"])

if fed["FEDUID"].duplicated().any():
    duplicates = fed[fed["FEDUID"].duplicated(keep=False)]
    raise ValueError(
        "Duplicated FEDUID values in boundary file:\n"
        + duplicates[["FEDUID", "FEDNAME", "PRUID", "PRNAME"]].to_string(index=False)
    )

canada_fed_count = len(fed)
qc_fed = fed[fed["PRUID"] == EXPECTED_QC_PRUID].copy()
qc_fed_count = len(qc_fed)

if qc_fed["FEDUID"].duplicated().any():
    duplicates = qc_fed[qc_fed["FEDUID"].duplicated(keep=False)]
    raise ValueError(
        "Duplicated Québec FEDUID values:\n"
        + duplicates[["FEDUID", "FEDNAME"]].to_string(index=False)
    )

qc_fed["fed_geometry_area_native"] = qc_fed.geometry.area


# -----------------------------
# CRS diagnostics
# -----------------------------

fed_crs = str(fed.crs)
qc_fed_crs = str(qc_fed.crs)

# The 2016 StatCan boundary guide says Lambert conformal conic, NAD83, metres.
crs_is_projected = bool(fed.crs and fed.crs.is_projected)
crs_linear_unit = ""
try:
    crs_linear_unit = fed.crs.axis_info[0].unit_name if fed.crs else ""
except Exception:
    crs_linear_unit = ""

# WGS84 copy for web-map viewing only.
qc_fed_wgs84 = qc_fed.to_crs(epsg=4326)
qc_fed_wgs84.to_file(OUTPUT_QC_BOUNDARIES_GEOJSON, driver="GeoJSON")


# -----------------------------
# Save boundary inventories
# -----------------------------

fed_inventory_cols = [
    col for col in [
        "FEDUID",
        "FEDNAME",
        "FEDENAME",
        "FEDFNAME",
        "PRUID",
        "PRNAME",
    ]
    if col in fed.columns
]

fed_inventory = pd.DataFrame(fed[fed_inventory_cols].copy())
fed_inventory.to_csv(OUTPUT_FED_INVENTORY, index=False, encoding="utf-8")

qc_fed_inventory = pd.DataFrame(
    qc_fed[
        fed_inventory_cols + ["fed_geometry_area_native"]
    ].copy()
)
qc_fed_inventory.to_csv(OUTPUT_QC_FED_INVENTORY, index=False, encoding="utf-8")


# -----------------------------
# Join boundary inventory to vote proxy
# -----------------------------

vote_proxy_id_col = find_first_existing_column(
    list(vote_proxy.columns),
    VOTE_PROXY_ID_CANDIDATES,
)

vote_proxy_name_col = find_first_existing_column(
    list(vote_proxy.columns),
    VOTE_PROXY_NAME_CANDIDATES,
)

if vote_proxy_id_col is None:
    raise ValueError(
        "Could not find federal electoral district ID column in vote proxy table.\n"
        + "Available columns:\n"
        + "\n".join(vote_proxy.columns)
    )

vote_proxy[vote_proxy_id_col] = normalize_text(vote_proxy[vote_proxy_id_col])

if vote_proxy_name_col is not None:
    vote_proxy[vote_proxy_name_col] = normalize_text(vote_proxy[vote_proxy_name_col])

boundary_small = qc_fed[
    [
        "FEDUID",
        "FEDNAME",
        "FEDENAME",
        "FEDFNAME",
        "PRUID",
        "PRNAME",
    ]
].copy()

vote_small_cols = [
    col for col in [
        vote_proxy_id_col,
        vote_proxy_name_col,
        "voter_turnout_pct_federal_2021",
        "pct_vote_leading_candidate_federal_2021",
        "pct_vote_leading_party_federal_2021",
        "leading_candidate_name",
        "leading_party_label_english_or_first",
        "registered_electors",
        "valid_ballots",
        "ballots_cast",
    ]
    if col is not None and col in vote_proxy.columns
]

vote_small = vote_proxy[vote_small_cols].copy()

if vote_proxy_id_col != "FEDUID":
    vote_small = vote_small.rename(columns={vote_proxy_id_col: "FEDUID"})

joined = boundary_small.merge(
    vote_small,
    on="FEDUID",
    how="outer",
    indicator=True,
    validate="one_to_one",
)

joined["in_boundary"] = joined["_merge"].isin(["both", "left_only"])
joined["in_vote_proxy"] = joined["_merge"].isin(["both", "right_only"])
joined["matched_boundary_to_vote_proxy"] = joined["_merge"].eq("both")

joined.to_csv(OUTPUT_JOIN_AUDIT, index=False, encoding="utf-8")

matched_rows = int(joined["matched_boundary_to_vote_proxy"].sum())
boundary_only_rows = int((joined["_merge"] == "left_only").sum())
vote_only_rows = int((joined["_merge"] == "right_only").sum())


# -----------------------------
# Base CD compatibility diagnostics
# -----------------------------

base_has_geometry = hasattr(base_cd, "geometry") and "geometry" in base_cd.columns

base_cd_crs = ""
base_cd_rows = len(base_cd)
base_cd_geometry_ready = False
base_cd_qc_rows = None

if base_has_geometry:
    base_cd_crs = str(base_cd.crs)
    base_cd_geometry_ready = base_cd.crs is not None

    if "province_code" in base_cd.columns:
        base_cd_qc_rows = int((base_cd["province_code"].astype(str).str.strip() == "24").sum())
    elif "PRUID" in base_cd.columns:
        base_cd_qc_rows = int((base_cd["PRUID"].astype(str).str.strip() == "24").sum())
    else:
        base_cd_qc_rows = base_cd_rows


# -----------------------------
# Summary
# -----------------------------

summary_rows = [
    {"metric": "fed_shapefile", "value": safe_relative(FED_SHAPEFILE)},
    {"metric": "vote_proxy_csv", "value": safe_relative(VOTE_PROXY_CSV)},
    {"metric": "base_cd_frame", "value": safe_relative(base_cd_path)},

    {"metric": "fed_rows_canada", "value": canada_fed_count},
    {"metric": "expected_fed_rows_canada", "value": EXPECTED_CANADA_FED_COUNT},
    {"metric": "fed_rows_canada_matches_expected", "value": canada_fed_count == EXPECTED_CANADA_FED_COUNT},

    {"metric": "fed_rows_quebec", "value": qc_fed_count},
    {"metric": "expected_fed_rows_quebec", "value": EXPECTED_QUEBEC_FED_COUNT},
    {"metric": "fed_rows_quebec_matches_expected", "value": qc_fed_count == EXPECTED_QUEBEC_FED_COUNT},

    {"metric": "fed_columns", "value": ", ".join(fed.columns)},
    {"metric": "missing_expected_fed_columns", "value": ", ".join(missing_expected_cols)},

    {"metric": "fed_crs", "value": fed_crs},
    {"metric": "fed_crs_is_projected", "value": crs_is_projected},
    {"metric": "fed_crs_linear_unit", "value": crs_linear_unit},

    {"metric": "vote_proxy_rows", "value": len(vote_proxy)},
    {"metric": "vote_proxy_id_column", "value": vote_proxy_id_col},
    {"metric": "vote_proxy_name_column", "value": vote_proxy_name_col or ""},

    {"metric": "boundary_vote_proxy_matched_rows", "value": matched_rows},
    {"metric": "boundary_only_rows", "value": boundary_only_rows},
    {"metric": "vote_proxy_only_rows", "value": vote_only_rows},
    {"metric": "boundary_vote_proxy_full_match", "value": matched_rows == EXPECTED_QUEBEC_FED_COUNT and boundary_only_rows == 0 and vote_only_rows == 0},

    {"metric": "base_cd_rows", "value": base_cd_rows},
    {"metric": "base_cd_has_geometry", "value": base_has_geometry},
    {"metric": "base_cd_crs", "value": base_cd_crs},
    {"metric": "base_cd_geometry_ready", "value": base_cd_geometry_ready},
    {"metric": "base_cd_qc_rows_detected", "value": base_cd_qc_rows if base_cd_qc_rows is not None else ""},

    {"metric": "output_qc_fed_boundaries_geojson", "value": safe_relative(OUTPUT_QC_BOUNDARIES_GEOJSON)},
]

for key, value in geometry_area_summary(qc_fed, "fed_geometry_area_native").items():
    summary_rows.append({"metric": key, "value": value})

summary_rows.append(
    {
        "metric": "recommended_next_step",
        "value": (
            "If the boundary-vote proxy join is 78/78 and the base census-division frame has usable geometry, "
            "generate the spatial allocation script to intersect federal electoral districts with Québec census divisions "
            "and area-weight vote proxy variables to census divisions."
        ),
    }
)

summary = pd.DataFrame(summary_rows)
summary.to_csv(OUTPUT_SUMMARY, index=False, encoding="utf-8")


# -----------------------------
# Console report
# -----------------------------

print("\n" + "=" * 72)
print("FEDERAL ELECTORAL DISTRICT BOUNDARY INSPECTION 2013")
print("=" * 72)

print("\nBoundary file:")
print("Path:", safe_relative(FED_SHAPEFILE))
print("Rows Canada:", canada_fed_count)
print("Rows Québec:", qc_fed_count)
print("CRS:", fed_crs)
print("Projected:", crs_is_projected)
print("Linear unit:", crs_linear_unit)

print("\nColumns:")
print(list(fed.columns))

print("\nQuébec boundary preview:")
print(qc_fed[["FEDUID", "FEDNAME", "FEDENAME", "FEDFNAME", "PRUID", "PRNAME"]].head(20).to_string(index=False))

print("\nVote proxy join:")
print("Vote proxy rows:", len(vote_proxy))
print("ID column:", vote_proxy_id_col)
print("Matched rows:", matched_rows)
print("Boundary-only rows:", boundary_only_rows)
print("Vote-proxy-only rows:", vote_only_rows)

if boundary_only_rows or vote_only_rows:
    print("\nJoin mismatches:")
    print(joined[joined["_merge"] != "both"].to_string(index=False))

print("\nBase census-division frame:")
print("Path:", safe_relative(base_cd_path))
print("Rows:", base_cd_rows)
print("Has geometry:", base_has_geometry)
print("CRS:", base_cd_crs)
print("Geometry ready:", base_cd_geometry_ready)

print("\nSummary:")
print(summary.to_string(index=False))

print("\nSaved:")
print(OUTPUT_FED_INVENTORY)
print(OUTPUT_QC_FED_INVENTORY)
print(OUTPUT_JOIN_AUDIT)
print(OUTPUT_QC_BOUNDARIES_GEOJSON)
print(OUTPUT_SUMMARY)

print("\nDone.")