from pathlib import Path
import pandas as pd

RAW_PROFILE_PATH = Path("98-401-X2021007_English_CSV_data.csv")

usecols = [
    "DGUID",
    "GEO_LEVEL",
    "GEO_NAME",
    "CHARACTERISTIC_ID",
    "CHARACTERISTIC_NAME",
    "C1_COUNT_TOTAL",
    "C10_RATE_TOTAL",
    "SYMBOL",
]

df = pd.read_csv(
    RAW_PROFILE_PATH,
    usecols=usecols,
    encoding="iso-8859-1",
    low_memory=False,
)

# Search only census tract characteristics.
tracts = df[df["GEO_LEVEL"].astype(str).str.strip() == "Census tract"].copy()

# Search for low-income related rows.
mask = (
    tracts["CHARACTERISTIC_NAME"]
    .astype(str)
    .str.contains("low-income|low income|LIM|LICO", case=False, na=False)
)

matches = tracts.loc[
    mask,
    [
        "CHARACTERISTIC_ID",
        "CHARACTERISTIC_NAME",
    ],
].drop_duplicates().sort_values("CHARACTERISTIC_ID")

print("\n--- LOW-INCOME CHARACTERISTIC CANDIDATES ---")
print(matches.to_string(index=False))

print("\nNumber of candidate characteristics:", len(matches))