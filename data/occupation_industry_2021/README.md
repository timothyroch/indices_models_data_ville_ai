# Occupation and Industry 2021

This folder contains the cleaned occupation, industry, and adjacent labour-market structure feature table derived from the 2021 Statistics Canada Census Profile.

The table is designed as a reusable feature source for SoVI-style socioeconomic variables, HGNN node features, labour-market vulnerability analysis, and other model-ready datasets. It does **not** compute any index directly.

## Source data

Original source file:

```text
data/census_profile_2021/98-401-X2021007_English_CSV_data.csv
````

Source product:

```text
Statistics Canada
Census Profile, 2021 Census of Population
Census Metropolitan Areas, Tracted Census Agglomerations and Census Tracts
```

Geographic level used:

```text
GEO_LEVEL == "Census tract"
```

Encoding note:

```python
pd.read_csv(path, encoding="iso-8859-1", low_memory=False)
```

## Scope of this table

Although this folder is named `occupation_industry_2021`, the clean table also includes adjacent labour-market structure variables.

The extracted block is:

```text
2223–2230 = labour force status
2231–2236 = work activity during the reference year
2237–2245 = class of worker and job permanency
2246–2258 = occupation, broad NOC 2021 categories
2259–2281 = industry, NAICS 2017 sectors
```

The inspection confirmed that:

```text
2180–2222 = location of study variables
2282+ = language-at-work variables
```

Those sections are not included in this cleaner.

## Extracted characteristics

### Labour force status

| Characteristic ID | Characteristic name                                                                  | Output field                                   |
| ----------------: | ------------------------------------------------------------------------------------ | ---------------------------------------------- |
|            `2223` | `Total - Population aged 15 years and over by labour force status - 25% sample data` | `labour_force_status_total_population_15_plus` |
|            `2224` | `In the labour force`                                                                | `in_labour_force`                              |
|            `2225` | `Employed`                                                                           | `employed`                                     |
|            `2226` | `Unemployed`                                                                         | `unemployed`                                   |
|            `2227` | `Not in the labour force`                                                            | `not_in_labour_force`                          |
|            `2228` | `Participation rate`                                                                 | `published_participation_rate`                 |
|            `2229` | `Employment rate`                                                                    | `published_employment_rate`                    |
|            `2230` | `Unemployment rate`                                                                  | `published_unemployment_rate`                  |

### Work activity during the reference year

| Characteristic ID | Characteristic name                                                                                      | Output field                             |
| ----------------: | -------------------------------------------------------------------------------------------------------- | ---------------------------------------- |
|            `2231` | `Total - Population aged 15 years and over by work activity during the reference year - 25% sample data` | `work_activity_total_population_15_plus` |
|            `2232` | `Did not work`                                                                                           | `did_not_work_reference_year`            |
|            `2233` | `Worked`                                                                                                 | `worked_reference_year`                  |
|            `2234` | `Worked full year full time`                                                                             | `worked_full_year_full_time`             |
|            `2235` | `Worked part year and/or part time`                                                                      | `worked_part_year_or_part_time`          |
|            `2236` | `Average weeks worked in reference year`                                                                 | `average_weeks_worked_reference_year`    |

### Class of worker and job permanency

| Characteristic ID | Characteristic name                                                                                         | Output field                          |
| ----------------: | ----------------------------------------------------------------------------------------------------------- | ------------------------------------- |
|            `2237` | `Total - Labour force aged 15 years and over by class of worker including job permanency - 25% sample data` | `class_worker_total_labour_force`     |
|            `2238` | `Class of worker - not applicable`                                                                          | `class_worker_not_applicable`         |
|            `2239` | `All classes of workers`                                                                                    | `all_classes_of_workers`              |
|            `2240` | `Employee`                                                                                                  | `employee`                            |
|            `2241` | `Permanent position`                                                                                        | `permanent_position`                  |
|            `2242` | `Temporary position`                                                                                        | `temporary_position`                  |
|            `2243` | `Fixed term (1 year or more)`                                                                               | `fixed_term_position_1_year_or_more`  |
|            `2244` | `Casual, seasonal or short-term position (less than 1 year)`                                                | `casual_seasonal_short_term_position` |
|            `2245` | `Self-employed`                                                                                             | `self_employed`                       |

### Occupation, broad NOC 2021 categories

| Characteristic ID | Characteristic name                                                                                                                              | Output field                                           |
| ----------------: | ------------------------------------------------------------------------------------------------------------------------------------------------ | ------------------------------------------------------ |
|            `2246` | `Total - Labour force aged 15 years and over by occupation - Broad category - National Occupational Classification (NOC) 2021 - 25% sample data` | `occupation_total_labour_force`                        |
|            `2247` | `Occupation - not applicable`                                                                                                                    | `occupation_not_applicable`                            |
|            `2248` | `All occupations`                                                                                                                                | `all_occupations`                                      |
|            `2249` | `0 Legislative and senior management occupations`                                                                                                | `occupation_management`                                |
|            `2250` | `1 Business, finance and administration occupations`                                                                                             | `occupation_business_finance_administration`           |
|            `2251` | `2 Natural and applied sciences and related occupations`                                                                                         | `occupation_natural_applied_sciences`                  |
|            `2252` | `3 Health occupations`                                                                                                                           | `occupation_health`                                    |
|            `2253` | `4 Occupations in education, law and social, community and government services`                                                                  | `occupation_education_law_social_community_government` |
|            `2254` | `5 Occupations in art, culture, recreation and sport`                                                                                            | `occupation_art_culture_recreation_sport`              |
|            `2255` | `6 Sales and service occupations`                                                                                                                | `occupation_sales_service`                             |
|            `2256` | `7 Trades, transport and equipment operators and related occupations`                                                                            | `occupation_trades_transport_equipment_operators`      |
|            `2257` | `8 Natural resources, agriculture and related production occupations`                                                                            | `occupation_natural_resources_agriculture_production`  |
|            `2258` | `9 Occupations in manufacturing and utilities`                                                                                                   | `occupation_manufacturing_utilities`                   |

### Industry, NAICS 2017 sectors

| Characteristic ID | Characteristic name                                                                                                                                | Output field                                          |
| ----------------: | -------------------------------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------- |
|            `2259` | `Total - Labour force aged 15 years and over by industry - Sectors - North American Industry Classification System (NAICS) 2017 - 25% sample data` | `industry_total_labour_force`                         |
|            `2260` | `Industry - not applicable`                                                                                                                        | `industry_not_applicable`                             |
|            `2261` | `All industries`                                                                                                                                   | `all_industries`                                      |
|            `2262` | `11 Agriculture, forestry, fishing and hunting`                                                                                                    | `industry_agriculture_forestry_fishing_hunting`       |
|            `2263` | `21 Mining, quarrying, and oil and gas extraction`                                                                                                 | `industry_mining_quarrying_oil_gas`                   |
|            `2264` | `22 Utilities`                                                                                                                                     | `industry_utilities`                                  |
|            `2265` | `23 Construction`                                                                                                                                  | `industry_construction`                               |
|            `2266` | `31-33 Manufacturing`                                                                                                                              | `industry_manufacturing`                              |
|            `2267` | `41 Wholesale trade`                                                                                                                               | `industry_wholesale_trade`                            |
|            `2268` | `44-45 Retail trade`                                                                                                                               | `industry_retail_trade`                               |
|            `2269` | `48-49 Transportation and warehousing`                                                                                                             | `industry_transportation_warehousing`                 |
|            `2270` | `51 Information and cultural industries`                                                                                                           | `industry_information_cultural`                       |
|            `2271` | `52 Finance and insurance`                                                                                                                         | `industry_finance_insurance`                          |
|            `2272` | `53 Real estate and rental and leasing`                                                                                                            | `industry_real_estate_rental_leasing`                 |
|            `2273` | `54 Professional, scientific and technical services`                                                                                               | `industry_professional_scientific_technical`          |
|            `2274` | `55 Management of companies and enterprises`                                                                                                       | `industry_management_companies_enterprises`           |
|            `2275` | `56 Administrative and support, waste management and remediation services`                                                                         | `industry_admin_support_waste_management_remediation` |
|            `2276` | `61 Educational services`                                                                                                                          | `industry_educational_services`                       |
|            `2277` | `62 Health care and social assistance`                                                                                                             | `industry_health_care_social_assistance`              |
|            `2278` | `71 Arts, entertainment and recreation`                                                                                                            | `industry_arts_entertainment_recreation`              |
|            `2279` | `72 Accommodation and food services`                                                                                                               | `industry_accommodation_food_services`                |
|            `2280` | `81 Other services except public administration`                                                                                                   | `industry_other_services_except_public_admin`         |
|            `2281` | `91 Public administration`                                                                                                                         | `industry_public_administration`                      |

## Clean output

The script creates:

```text
output/clean_census_tract_occupation_industry_2021.csv
output/clean_census_tract_occupation_industry_2021.parquet
```

The clean table contains one row per census tract.

In the validated run, the output contained:

```text
6247 census tracts
```

## Derived fields

The script computes reusable proportions, including:

| Column                                                | Description                                                                                                                     |
| ----------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------- |
| `participation_rate_published`                        | Published participation rate converted from `0–100` to `0–1`                                                                    |
| `employment_rate_published`                           | Published employment rate converted from `0–100` to `0–1`                                                                       |
| `unemployment_rate_published`                         | Published unemployment rate converted from `0–100` to `0–1`                                                                     |
| `pct_worked_full_year_full_time`                      | Share of population 15+ who worked full year full time                                                                          |
| `pct_worked_part_year_or_part_time`                   | Share of population 15+ who worked part year and/or part time                                                                   |
| `pct_part_time_or_part_year_among_workers`            | Share of workers who worked part year and/or part time                                                                          |
| `pct_temporary_position`                              | Share of employees in temporary positions                                                                                       |
| `pct_self_employed`                                   | Share of all classes of workers who are self-employed                                                                           |
| `pct_occupation_sales_service`                        | Share of all occupations in sales and service                                                                                   |
| `pct_occupation_trades_transport_equipment_operators` | Share of all occupations in trades, transport, and equipment operators                                                          |
| `pct_occupation_frontline_service`                    | Composite occupation share: sales/service, trades/transport, manufacturing/utilities, natural resources/agriculture             |
| `pct_occupation_knowledge_professional`               | Composite occupation share: business/finance/admin, natural/applied sciences, health, education/law/social/community/government |
| `pct_industry_service_retail_accommodation`           | Composite industry share: retail, accommodation/food, arts/recreation, and other services                                       |
| `pct_industry_physical_infrastructure`                | Composite industry share: construction, manufacturing, transportation/warehousing, utilities, agriculture, mining/oil/gas       |
| `pct_industry_professional_knowledge`                 | Composite industry share: information/cultural, finance/insurance, professional/scientific/technical, management companies      |
| `pct_industry_public_essential_services`              | Composite industry share: education, health care/social assistance, public administration                                       |

Computed `pct_*` columns are stored as proportions between `0` and `1`.

The script uses safe bounded division. If a denominator is zero or missing, the derived value is kept as missing. Derived proportions are clipped to the valid range:

```text
0 <= proportion <= 1
```

Raw counts and published `0–100` rates are preserved unchanged.

## Default measures

The script defines:

```text
sales_service_occupation_measure_default = pct_occupation_sales_service
```

This captures the share of the labour force in sales and service occupations.

The script defines:

```text
trades_transport_occupation_measure_default = pct_occupation_trades_transport_equipment_operators
```

This captures the share of the labour force in trades, transport, equipment operator, and related occupations.

The script defines:

```text
temporary_work_measure_default = pct_temporary_position
```

This captures the share of employees in temporary positions.

The script defines:

```text
part_time_or_part_year_work_measure_default = pct_part_time_or_part_year_among_workers
```

This captures the share of workers who worked part year and/or part time.

The script defines:

```text
service_retail_accommodation_industry_measure_default = pct_industry_service_retail_accommodation
```

This captures the share of the labour force working in retail, accommodation/food, arts/recreation, and other services.

## Validation notes

The validated run confirmed that the table was created successfully with:

```text
6247 census tracts
```

The main default variables had the following summaries:

```text
unemployment_rate_published mean                         0.107386
sales_service_occupation_measure_default mean            0.249622
trades_transport_occupation_measure_default mean         0.161268
temporary_work_measure_default mean                      0.164101
part_time_or_part_year_work_measure_default mean         0.468382
service_retail_accommodation_industry_measure_default    0.233206
pct_industry_physical_infrastructure mean                0.229887
pct_industry_professional_knowledge mean                 0.164546
```

A few variables can reach `1.0` in very small or specialized tracts. This is not treated as an error because the proportions remain bounded and the overall summaries are coherent.

## Missing values

The script preserves all census tracts.

In the validated run, almost every selected raw variable had:

```text
89 missing values
```

The field:

```text
average_weeks_worked_reference_year
```

had:

```text
90 missing values
```

All missing values in the selected block were marked with:

```text
SYMBOL = x
```

Missing values are kept as missing. They are not dropped or imputed at this stage.

## Role in the pipeline

This folder produces a clean feature table only.

The broader pipeline is:

```text
raw Census Profile
    ↓
clean occupation / industry / labour-market feature table
    ↓
master census-tract feature table
    ↓
SoVI / BRIC / HGNN-specific inputs
```

For HGNN and SoVI-style labour-market analysis, useful fields include:

```text
unemployment_rate_published
sales_service_occupation_measure_default
trades_transport_occupation_measure_default
temporary_work_measure_default
part_time_or_part_year_work_measure_default
service_retail_accommodation_industry_measure_default
pct_occupation_frontline_service
pct_occupation_knowledge_professional
pct_industry_public_essential_services
pct_industry_physical_infrastructure
pct_industry_professional_knowledge
```

```

This matches the validated run: 6,247 census tracts preserved, expected 25% sample suppression kept as missing, and coherent labour-market, occupation, and industry summaries generated.