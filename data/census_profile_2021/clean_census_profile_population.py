from pathlib import Path
import pandas as pd


# -----------------------------
# Configuration
# -----------------------------

RAW_PROFILE_PATH = Path("98-401-X2021007_English_CSV_data.csv")

OUTPUT_DIR = Path("output")
OUTPUT_DIR.mkdir(exist_ok=True)

OUTPUT_CSV = OUTPUT_DIR / "clean_census_tract_population_2021.csv"
OUTPUT_PARQUET = OUTPUT_DIR / "clean_census_tract_population_2021.parquet"


# -----------------------------
# Load only needed columns
# -----------------------------

usecols = [
    "CENSUS_YEAR",
    "DGUID",
    "GEO_LEVEL",
    "GEO_NAME",
    "CHARACTERISTIC_NAME",
    "C1_COUNT_TOTAL",
    "SYMBOL",
]

df = pd.read_csv(
    RAW_PROFILE_PATH,
    usecols=usecols,
    encoding="iso-8859-1",
    low_memory=False,
)

print("Loaded Census Profile")
print("Rows:", len(df))


# -----------------------------
# Filter to census tract population rows
# -----------------------------

pop = df[
    (df["GEO_LEVEL"].astype(str).str.strip() == "Census tract")
    & (df["CHARACTERISTIC_NAME"].astype(str).str.strip() == "Population, 2021")
].copy()

print("Census tract population rows:", len(pop))

if pop.empty:
    raise ValueError("No census tract population rows found.")


# -----------------------------
# Convert population values safely
# -----------------------------

pop["population_total"] = pd.to_numeric(
    pop["C1_COUNT_TOTAL"],
    errors="coerce",
)

missing = pop[pop["population_total"].isna()].copy()

if not missing.empty:
    print("\nWarning: some census tracts have missing/non-numeric population values.")
    print("Missing rows:", len(missing))
    print("\nMissing population rows preview:")
    print(
        missing[
            [
                "DGUID",
                "GEO_NAME",
                "C1_COUNT_TOTAL",
                "SYMBOL",
            ]
        ].head(30).to_string(index=False)
    )

    print("\nSYMBOL counts for missing rows:")
    print(missing["SYMBOL"].value_counts(dropna=False))

    # Conservative rule for now:
    # keep the rows, but set missing population to 0.
    # These can later be excluded from SVI because SVI requires non-zero population.
    pop["population_total"] = pop["population_total"].fillna(0)


# -----------------------------
# Create clean schema
# -----------------------------

clean = pop.rename(
    columns={
        "DGUID": "statcan_dguid",
        "GEO_NAME": "geo_name",
        "CENSUS_YEAR": "census_year",
    }
).copy()

clean["unit_type"] = "census_tract"
clean["source"] = "Statistics Canada Census Profile, 2021 Census of Population"

clean["population_total"] = clean["population_total"].astype("Int64")

clean = clean[
    [
        "statcan_dguid",
        "geo_name",
        "unit_type",
        "census_year",
        "population_total",
        "source",
    ]
]


# -----------------------------
# Validation
# -----------------------------

if clean["statcan_dguid"].duplicated().any():
    duplicated = clean[clean["statcan_dguid"].duplicated(keep=False)]
    raise ValueError(
        "Duplicated statcan_dguid values found:\n"
        + duplicated.to_string(index=False)
    )

if clean["population_total"].isna().any():
    raise ValueError("Some census tracts still have missing population_total.")

print("\nPopulation summary:")
print(clean["population_total"].describe())

zero_pop_count = int((clean["population_total"] == 0).sum())
print("\nZero-population tracts:", zero_pop_count)

print("\nClean table preview:")
print(clean.head(10).to_string(index=False))


# -----------------------------
# Save outputs
# -----------------------------

clean.to_csv(OUTPUT_CSV, index=False)
clean.to_parquet(OUTPUT_PARQUET, index=False)

print("\nSaved:")
print(OUTPUT_CSV)
print(OUTPUT_PARQUET)