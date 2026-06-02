Yes. The paper’s **42 independent variables** are in **Table 2**, and the **11 factors** are in **Table 3** plus the explanatory text immediately after it. The paper says they originally collected more than 250 variables, reduced those to 85 raw/computed variables after multicollinearity testing, and then used **42 independent variables** in the statistical analysis. 

## The 42 independent variables in SoVI

| Code            | Variable description                                                                  |
| --------------- | ------------------------------------------------------------------------------------- |
| `MED_AGE90`     | Median age, 1990                                                                      |
| `PERCAP89`      | Per capita income, 1989                                                               |
| `MVALOO90`      | Median dollar value of owner-occupied housing, 1990                                   |
| `MEDRENT90`     | Median rent for renter-occupied housing units, 1990                                   |
| `PHYSICN90`     | Number of physicians per 100,000 population, 1990                                     |
| `PCTVOTE92`     | Percent vote cast for president for the leading party, Democratic, 1992               |
| `BRATE90`       | Birth rate, births per 1,000 population, 1990                                         |
| `MIGRA_97`      | Net international migration, 1990–1997                                                |
| `PCTFARMS92`    | Land in farms as a percent of total land, 1992                                        |
| `PCTBLACK90`    | Percent African American, 1990                                                        |
| `PCTINDIAN90`   | Percent Native American, 1990                                                         |
| `PCTASIAN90`    | Percent Asian, 1990                                                                   |
| `PCTHISPANIC90` | Percent Hispanic, 1990                                                                |
| `PCTKIDS90`     | Percent of population under five years old, 1990                                      |
| `PCTOLD90`      | Percent of population over 65 years, 1990                                             |
| `PCTVLUN91`     | Percent of civilian labor force unemployed, 1991                                      |
| `AVGPERHH`      | Average number of people per household, 1990                                          |
| `PCTHH7589`     | Percent of households earning more than $75,000, 1989                                 |
| `PCTPOV90`      | Percent living in poverty, 1990                                                       |
| `PCTRENTER90`   | Percent renter-occupied housing units, 1990                                           |
| `PCTRFRM90`     | Percent rural farm population, 1990                                                   |
| `DEBREV92`      | General local government debt-to-revenue ratio, 1992                                  |
| `PCTMOBL90`     | Percent of housing units that are mobile homes, 1990                                  |
| `PCTNOHS90`     | Percent of population age 25+ with no high school diploma, 1990                       |
| `HODENUT90`     | Number of housing units per square mile, 1990                                         |
| `HUPTDEN90`     | Number of housing permits per new residential construction per square mile, 1990      |
| `MAESDEN92`     | Number of manufacturing establishments per square mile, 1992                          |
| `EARNDEN90`     | Earnings, in $1,000, in all industries per square mile, 1990                          |
| `COMDEVDN92`    | Number of commercial establishments per square mile, 1990                             |
| `RPROPDEN92`    | Value of all property and farm products sold per square mile, 1990                    |
| `CVBRPC91`      | Percent of the population participating in the labor force, 1990                      |
| `FEMLBR90`      | Percent females participating in civilian labor force, 1990                           |
| `AGRIPC90`      | Percent employed in primary extractive industries: farming, fishing, mining, forestry |
| `TRANPC90`      | Percent employed in transportation, communications, and other public utilities, 1990  |
| `SERVPC90`      | Percent employed in service occupations, 1990                                         |
| `NRRESPC91`     | Per capita residents in nursing homes, 1991                                           |
| `HOSPTPC91`     | Per capita number of community hospitals, 1991                                        |
| `PCCHGPOP90`    | Percent population change, 1980–1990                                                  |
| `PCTURB90`      | Percent urban population, 1990                                                        |
| `PCTFEM90`      | Percent females, 1990                                                                 |
| `PCTF_HH90`     | Percent female-headed households, no spouse present, 1990                             |
| `SSBENPC90`     | Per capita Social Security recipients, 1990                                           |

These are directly listed in Table 2 of the paper. 

## The 11 SoVI factors

The paper uses principal components/factor analysis with **varimax rotation** and keeps components with **eigenvalues greater than 1.00**, producing 11 factors that explain **76.4%** of the variance.  

|  # | Factor name                       | Variance explained | Dominant variable                                                        | Correlation |
| -: | --------------------------------- | -----------------: | ------------------------------------------------------------------------ | ----------: |
|  1 | Personal wealth                   |              12.4% | Per capita income                                                        |       +0.87 |
|  2 | Age                               |              11.9% | Median age                                                               |       −0.90 |
|  3 | Density of the built environment  |              11.2% | Number of commercial establishments per square mile                      |       +0.98 |
|  4 | Single-sector economic dependence |               8.6% | Percent employed in extractive industries                                |       +0.80 |
|  5 | Housing stock and tenancy         |               7.0% | Percent housing units that are mobile homes                              |       −0.75 |
|  6 | Race — African American           |               6.9% | Percent African American                                                 |       +0.80 |
|  7 | Ethnicity — Hispanic              |               4.2% | Percent Hispanic                                                         |       +0.89 |
|  8 | Ethnicity — Native American       |               4.1% | Percent Native American                                                  |       +0.75 |
|  9 | Race — Asian                      |               3.9% | Percent Asian                                                            |       +0.71 |
| 10 | Occupation                        |               3.2% | Percent employed in service occupations                                  |       +0.76 |
| 11 | Infrastructure dependence         |               2.9% | Percent employed in transportation, communications, and public utilities |       +0.77 |

This is the exact Table 3 summary. 

A key detail: their final SoVI is not a weighted index by these percentages. They add the factor scores in an **additive model**, treating each factor as equally contributing to vulnerability because they say they had no defensible basis for assigning different weights. They also orient factors so positive values increase vulnerability and negative values decrease it; when the sign is ambiguous, they use the absolute value. 

For your benchmark, the faithful “SoVI-like” recipe is therefore:

```text
42-variable-inspired social table
→ normalize to percentages / per-capita / density
→ handle missing values
→ PCA / factor analysis
→ varimax rotation
→ keep eigenvalues > 1
→ orient factors toward vulnerability
→ sum factor scores additively
→ classify by standard-deviation bands
```


Yes. The paper is **Cutter, Boruff & Shirley’s original SoVI paper**, and its methodology is much simpler than the full OECD/Nardo-style composite-indicator workflow you listed. It uses some of those steps, but not all of them with the same depth. The core pipeline is:

**raw socioeconomic variables → transformed/normalized variables → factor analysis/PCA with varimax rotation → 11 factor scores → sign correction/absolute value when needed → additive sum → SoVI score → standard deviation classification/map**. 

Below is how each step is actually done in the paper, and how you would replicate it.

---

# 1. Traiter les données manquantes

## What the paper does

The paper does **not** use advanced imputation. It does not use multiple imputation, regression imputation, maximum likelihood imputation, or uncertainty propagation.

It says explicitly that **factor analysis cannot be performed with missing values**, so when a county had a missing value, the authors **substituted zero**.

Their justification is spatial/comparative: they preferred to keep **all 3,141 U.S. counties** in the analysis rather than dropping counties with missing data, especially because many missing cases were in Alaska and Hawaii. They acknowledge that replacing missing values with zero may not represent the true vulnerability and may **underestimate vulnerability** for affected counties. 

So the missing-data method is:

[
x_{ij}^{missing} \leftarrow 0
]

where (x_{ij}) is variable (j) for county (i).

## Interpretation

This is a **very crude imputation strategy**. It is not statistically elegant. It preserves spatial coverage, but it creates bias if zero is not a meaningful value for the variable.

For example, if a county has missing data for physicians per capita, replacing it with zero makes the county look like it has no physicians. That could artificially increase vulnerability or distort the factor structure.

The authors know this, but they prioritize keeping every county.

## How to replicate

For a strict reproduction:

1. Build the county-by-variable matrix.
2. After computing all derived variables, check for missing values.
3. Replace all missing entries with zero.
4. Run PCA/factor analysis on the completed matrix.

In Python-style pseudocode:

```python
X = county_variable_matrix.copy()
X = X.fillna(0)
```

But for your benchmark, I would document this as:

> Original SoVI missing-data treatment: zero substitution for missing values, justified by the desire to preserve all spatial units. This may underestimate vulnerability in affected counties.

## Stronger modern variant

For your own benchmark, I would reproduce the original zero-imputation version, but also include sensitivity tests:

```text
SoVI-original: missing values → 0
SoVI-mean: missing values → variable mean
SoVI-median: missing values → variable median
SoVI-MICE: missing values → multiple imputation
SoVI-complete-case: drop counties with missing values
```

That would let you quantify whether the original SoVI is robust to this questionable choice.

---

# 2. Faire une analyse multivariée

This is the heart of the paper.

## What the paper does

The authors start with **more than 250 variables**. Then they test for multicollinearity and reduce the set to **85 raw and computed variables**. After additional transformations and normalization, they keep **42 independent variables** for the statistical analysis. 

Then they use **factor analysis, specifically principal components analysis**, to reduce those 42 variables into **11 independent factors**.

The 11 factors explain **76.4% of the variance** among U.S. counties. 

They also use **varimax rotation** to simplify the factor structure. The paper says varimax rotation was used to make the underlying dimensions more independent and interpretable. In practice, varimax tries to make each variable load strongly on a small number of factors and weakly on others. 

The number of factors is chosen using:

1. **Eigenvalues greater than 1.00**, and
2. A **scree diagram** showing a distinct break.

So the multivariate procedure is:

[
42 \text{ variables} \rightarrow PCA/factor analysis \rightarrow varimax rotation \rightarrow 11 factors
]

## The 42 variables

The paper lists the 42 variables in Table 2. They include variables such as median age, per capita income, median housing value, median rent, physicians per 100,000 population, birth rate, migration, percent Black, percent Native American, percent Asian, percent Hispanic, percent children under 5, percent over 65, unemployment, poverty, renters, mobile homes, no high school diploma, housing density, commercial density, manufacturing density, labor-force participation, service employment, nursing homes, hospitals, population change, percent urban, female-headed households, and Social Security recipients. 

So the empirical matrix is:

[
X \in \mathbb{R}^{3141 \times 42}
]

where rows are counties and columns are vulnerability indicators.

## The 11 factors

The paper’s 11 factors are:

| Factor | Name                              | Main interpretation                                                      |
| -----: | --------------------------------- | ------------------------------------------------------------------------ |
|      1 | Personal wealth                   | Income, high-income households, housing value, rent, poverty             |
|      2 | Age                               | Children, birth rate, median age, elderly, Social Security               |
|      3 | Density of built environment      | Commercial, manufacturing, housing, and construction density             |
|      4 | Single-sector economic dependence | Rural farm population, extractive industries                             |
|      5 | Housing stock and tenancy         | Mobile homes, renters, urban population                                  |
|      6 | Race: African American            | Percent African American, female-headed households                       |
|      7 | Ethnicity: Hispanic               | Percent Hispanic                                                         |
|      8 | Ethnicity: Native American        | Percent Native American                                                  |
|      9 | Race: Asian                       | Percent Asian                                                            |
|     10 | Occupation                        | Service occupations                                                      |
|     11 | Infrastructure dependence         | Debt-to-revenue ratio, transportation/communication/utilities employment |

The factor names are not mechanically produced by PCA. They are **interpretive labels** assigned by the authors after inspecting which variables load most strongly on each factor. 

## Important replication point

You asked earlier whether you should expect to get the exact same 11 groups if you rerun factor analysis. The answer is: **not necessarily**.

You may get something different because PCA/factor analysis depends on:

1. the exact dataset,
2. the exact year,
3. the exact variables,
4. transformations,
5. standardization,
6. treatment of missing values,
7. rotation method,
8. software implementation,
9. sign conventions,
10. factor-retention rule.

Even factor signs are arbitrary: a factor can be multiplied by (-1) and remain mathematically equivalent. So if you replicate it, you should not expect perfect semantic identity unless you use the exact same data and preprocessing.

## How to replicate

A faithful reproduction would be:

```python
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
# For varimax, use factor_analyzer or implement varimax manually.

X = df[variables_42].fillna(0)

# Standardize before PCA/factor analysis
Z = StandardScaler().fit_transform(X)

# Run PCA
pca = PCA()
scores = pca.fit_transform(Z)

# Keep components with eigenvalue > 1
eigenvalues = pca.explained_variance_
k = sum(eigenvalues > 1.0)

# Rerun PCA with k components
pca = PCA(n_components=k)
scores = pca.fit_transform(Z)
loadings = pca.components_.T
```

Then apply varimax rotation to the loadings and compute rotated factor scores.

Conceptually:

[
Z = \text{standardized}(X)
]

[
Z \approx F \Lambda^\top
]

where (F) are county factor scores and (\Lambda) are variable loadings.

After varimax:

[
\Lambda_{rot} = \Lambda R
]

where (R) is the rotation matrix.

Then obtain factor scores for each county.

---

# 3. Normaliser les données

## What the paper does

The paper says that after computations and normalization, the 42 variables were used in the statistical analysis. The normalization described is mostly **substantive unit normalization**, not necessarily final index scaling.

They convert variables into comparable forms such as:

1. percentages,
2. per capita measures,
3. density functions.

The paper says:

> “After all the computations and normalization of data — to percentages, per capita, or density functions — 42 independent variables were used…”

So the normalization step is mainly about turning raw counts into meaningful ratios.

Examples:

| Raw concept                      | Normalized form                   |
| -------------------------------- | --------------------------------- |
| Number of physicians             | Physicians per 100,000 population |
| Number of hospitals              | Hospitals per capita              |
| Number of nursing home residents | Nursing home residents per capita |
| Commercial establishments        | Establishments per square mile    |
| Housing units                    | Housing units per square mile     |
| African American population      | Percent African American          |
| Elderly population               | Percent over 65                   |
| Poverty count                    | Percent living in poverty         |
| Social Security recipients       | Recipients per capita             |

This is different from the OECD guide’s menu of min-max, z-score, rank, distance-to-target, categorical scales, etc.

## Likely additional standardization before PCA

Although the paper does not spell out every computational detail, PCA/factor analysis on variables with very different units normally requires standardization, usually using the correlation matrix. Given that variables include dollars, percentages, densities, ratios, and per-capita values, a faithful modern replication should standardize the 42 variables before PCA.

That means:

[
z_{ij} = \frac{x_{ij} - \mu_j}{\sigma_j}
]

where:

* (x_{ij}) is county (i)’s value for variable (j),
* (\mu_j) is the mean of variable (j),
* (\sigma_j) is the standard deviation of variable (j).

This gives each variable mean 0 and standard deviation 1.

However, the paper’s explicit normalization discussion is not as detailed as the OECD guide. So in your reproduction, you should distinguish:

```text
Explicitly stated in paper:
- Convert raw variables into percentages, per-capita values, or densities.

Statistically necessary / likely:
- Standardize variables before PCA/factor analysis, usually equivalent to PCA on the correlation matrix.
```

## How to replicate

For each variable:

### Percent variables

[
\text{Percent group} = \frac{\text{group population}}{\text{total population}} \times 100
]

Example:

[
\text{PCTOLD90} = \frac{\text{population over 65}}{\text{total population}} \times 100
]

### Per-capita variables

[
\text{Per capita variable} = \frac{\text{count}}{\text{population}}
]

Example:

[
\text{SSBENPC90} = \frac{\text{Social Security recipients}}{\text{total population}}
]

### Density variables

[
\text{Density} = \frac{\text{count or value}}{\text{land area}}
]

Example:

[
\text{COMDEVDN92} = \frac{\text{commercial establishments}}{\text{square miles}}
]

### Standardization before PCA

```python
from sklearn.preprocessing import StandardScaler

Z = StandardScaler().fit_transform(X)
```

This is the matrix you feed into PCA/factor analysis.

---

# 4. Pondérer et agréger

This is another central part of the paper.

## What the paper does for weighting

The paper does **not** use AHP, DEA, Benefit of the Doubt, public opinion weights, budget allocation, conjoint analysis, or expert weights.

It uses **equal weighting of factors**.

After the factor analysis, the authors add the 11 factor scores to the original county file as 11 new variables. Then they place those 11 factors in an **additive model**.

They explicitly say that they selected an additive model because they did not want to make an a priori assumption about the importance of each factor. Each factor is therefore treated as having an equal contribution to overall vulnerability. 

So the SoVI score is basically:

[
SoVI_i = \sum_{k=1}^{11} F_{ik}
]

where:

* (SoVI_i) is the social vulnerability score of county (i),
* (F_{ik}) is the score of county (i) on factor (k).

This is an **equal-weight additive aggregation** of factor scores.

## Direction/sign correction

Before summing, the authors adjust the factor scores so that:

* positive values indicate **higher vulnerability**,
* negative values indicate **lower vulnerability**.

They also say that when a factor’s effect was ambiguous, they used the **absolute value**. 

So for each factor (k), they decide whether the factor increases or decreases vulnerability.

If a factor’s high values clearly increase vulnerability:

[
F_{ik}^{*} = F_{ik}
]

If a factor’s high values decrease vulnerability:

[
F_{ik}^{*} = -F_{ik}
]

If the direction is ambiguous:

[
F_{ik}^{*} = |F_{ik}|
]

Then:

[
SoVI_i = \sum_{k=1}^{11} F_{ik}^{*}
]

## Why this matters

This is a very important methodological choice.

PCA itself does not know whether a factor is “good” or “bad.” For example, a “wealth” factor may load positively on income and housing value. Wealth can decrease vulnerability because rich communities recover more easily, but it can also increase potential dollar losses because there is more property at risk.

The authors handle this interpretively.

For factors with clear vulnerability direction, they flip signs if necessary. For ambiguous factors, they use absolute values.

This means the SoVI is not purely statistical. It is:

```text
statistical extraction of factors
+
theoretical interpretation of vulnerability direction
+
equal-weight additive aggregation
```

## Aggregation method

The aggregation is **linear/additive**:

[
SoVI_i = F_{i1}^{*} + F_{i2}^{*} + \cdots + F_{i11}^{*}
]

There is no geometric aggregation:

[
\prod_k F_{ik}^{w_k}
]

There is no non-compensatory multicriteria method.

SoVI is compensatory: a county can have high vulnerability on one factor and lower vulnerability on another, and the additive sum allows these to offset each other.

## Classification/mapping

After computing SoVI, they map the scores using standard deviations from the mean. The map has five classes:

1. below (-1) standard deviation,
2. (-1) to (-0.5) standard deviation,
3. (-0.5) to (0.5) standard deviation,
4. (0.5) to (1) standard deviation,
5. above (1) standard deviation.

They report that the SoVI scores range from **-9.6** to **49.51**, with a mean of **1.54** and standard deviation of **3.38**. Counties above (+1) standard deviation are treated as most vulnerable; counties below (-1) standard deviation are least vulnerable. 

So classification is:

[
z_i = \frac{SoVI_i - \overline{SoVI}}{sd(SoVI)}
]

Then:

```text
z < -1.0             least vulnerable
-1.0 <= z < -0.5     low vulnerability
-0.5 <= z <= 0.5     moderate vulnerability
0.5 < z <= 1.0       high vulnerability
z > 1.0              most vulnerable
```

---

# Full reproduction recipe

Here is the clean reproducible version.

## Step 0 — Define spatial units

Use all U.S. counties:

[
n = 3141
]

Each row is one county.

## Step 1 — Collect variables

Collect the 42 variables from Table 2.

Examples include:

```text
MED_AGE90
PERCAP89
MVALOO90
MEDRENT90
PHYSICN90
PCTVOTE92
BRATE90
MIGRA_97
PCTFARMS92
PCTBLACK90
PCTINDIAN90
PCTASIAN90
PCTHISPANIC90
PCTKIDS90
PCTOLD90
PCTVLUN91
AVGPERHH
PCTHH7589
PCTPOV90
PCTRENTER90
PCTRFRM90
DEBREV92
PCTMOBL90
PCTNOHS90
HODENUT90
HUPTDEN90
MAESDEN92
EARNDEN90
COMDEVDN92
RPROPDEN92
CVBRPC91
FEMLBR90
AGRIPC90
TRANPC90
SERVPC90
NRRESPC91
HOSPTPC91
PCCHGPOP90
PCTURB90
PCTFEM90
PCTF_HH90
SSBENPC90
```

## Step 2 — Transform raw variables

Convert raw counts to:

```text
percentages
per-capita values
density values
ratios
```

## Step 3 — Handle missing values

Original paper:

```python
X = X.fillna(0)
```

Document the limitation:

```text
This preserves all counties but may underestimate vulnerability.
```

## Step 4 — Standardize variables

Use z-scores:

[
z_{ij} = \frac{x_{ij} - \mu_j}{\sigma_j}
]

```python
Z = StandardScaler().fit_transform(X)
```

## Step 5 — Run PCA/factor analysis

Use PCA/factor analysis on the standardized variables.

Keep factors with eigenvalue (> 1), supported by the scree plot.

The target result is:

```text
11 factors
approximately 76.4% explained variance
```

## Step 6 — Apply varimax rotation

Use varimax rotation to simplify interpretation.

```text
Goal: variables load strongly on one factor and weakly on others.
```

## Step 7 — Interpret and name factors

Inspect the rotated loadings.

Assign names based on dominant variables:

```text
Personal wealth
Age
Density of built environment
Single-sector economic dependence
Housing stock and tenancy
Race: African American
Ethnicity: Hispanic
Ethnicity: Native American
Race: Asian
Occupation
Infrastructure dependence
```

## Step 8 — Compute factor scores

For each county, compute its score on each of the 11 factors.

You now have:

[
F \in \mathbb{R}^{3141 \times 11}
]

## Step 9 — Orient factor signs

Make sure high values mean high vulnerability.

For each factor:

```text
if high factor score means high vulnerability:
    keep sign
if high factor score means low vulnerability:
    multiply by -1
if ambiguous:
    take absolute value
```

Mathematically:

[
F_{ik}^{*} =
\begin{cases}
F_{ik}, & \text{if positive direction increases vulnerability} \
-F_{ik}, & \text{if positive direction decreases vulnerability} \
|F_{ik}|, & \text{if ambiguous}
\end{cases}
]

## Step 10 — Additive aggregation

Compute:

[
SoVI_i = \sum_{k=1}^{11} F_{ik}^{*}
]

No extra weights.

Equivalent:

```python
sovi = oriented_factor_scores.sum(axis=1)
```

## Step 11 — Classify counties

Compute standard deviation categories:

```python
z_sovi = (sovi - sovi.mean()) / sovi.std()
```

Then classify:

```text
z < -1.0
-1.0 to -0.5
-0.5 to 0.5
0.5 to 1.0
z > 1.0
```

---

# How each OECD-style step maps to the SoVI paper

| Composite-indicator step | What SoVI actually does                                                                             |
| ------------------------ | --------------------------------------------------------------------------------------------------- |
| Missing data             | Replaces missing values with zero                                                                   |
| Multivariate analysis    | PCA/factor analysis on 42 variables; varimax rotation; 11 factors retained                          |
| Normalization            | Converts raw variables to percentages, per-capita values, densities; likely standardizes before PCA |
| Weighting                | Equal weighting of the 11 factor scores                                                             |
| Aggregation              | Additive sum of factor scores                                                                       |
| Directionality           | Signs adjusted so positive means more vulnerable; ambiguous effects use absolute value              |
| Classification           | Standard deviations from mean into five vulnerability classes                                       |

---

# Important methodological warning for your benchmark

For your reproduction benchmark, I would not treat SoVI as a single fixed formula. It is better to treat it as a **methodological family**:

```text
SoVI = variable selection + missing-data rule + standardization + PCA/factor analysis + rotation + sign orientation + additive aggregation
```

The fragile parts are:

1. **missing-value substitution with zero**,
2. **factor instability**,
3. **interpretive naming of factors**,
4. **sign/orientation decisions**,
5. **equal weighting**,
6. **additive compensation**.

So if you implement this for VILLE_IA, your reproduction should probably include:

```text
SoVI-original reproduction:
    missing = 0
    PCA/factor analysis
    varimax
    eigenvalue > 1
    equal additive factor sum

SoVI-robustness variants:
    median/multiple imputation
    alternative number of factors
    no rotation vs varimax
    equal weights vs variance-weighted factors
    additive vs geometric aggregation
    sensitivity of final rankings/classes
```

That would make your benchmark much stronger than just copying the index.