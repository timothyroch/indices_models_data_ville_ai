import pandas as pd

path = "98-401-X2021007_English_CSV_data.csv"

df = pd.read_csv(
    path,
    usecols=[
        "DGUID",
        "GEO_LEVEL",
        "GEO_NAME",
        "CHARACTERISTIC_NAME",
        "C1_COUNT_TOTAL",
    ],
    low_memory=False,
    encoding='iso-8859-1'  # <-- This right here stops the UTF-8 crash
)

pop = df[df["CHARACTERISTIC_NAME"].str.strip() == "Population, 2021"].copy()

print("\n--- POPULATION ROWS BY GEO_LEVEL ---")
print(pop["GEO_LEVEL"].value_counts())

print("\n--- SAMPLE CENSUS TRACT POPULATION ROWS ---")
tract_pop = pop[pop["GEO_LEVEL"].str.contains("Census tract", case=False, na=False)].copy()
print(tract_pop.head(20).to_string())

print("\nNumber of census tract population rows:", len(tract_pop))