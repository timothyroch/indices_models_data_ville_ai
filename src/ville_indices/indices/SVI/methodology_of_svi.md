Yes — the **CDC/ATSDR SVI paper** uses a much simpler and more operational methodology than the SoVI paper. The core pipeline is:

**15 Census variables → percentile ranks → four domain percentile ranks → overall SVI percentile rank → flags for extreme variables.** 

It does **not** use PCA/factor analysis, varimax rotation, z-scores, eigenvalues, or latent factors like SoVI.

---

# 1. Définir le cadre théorique

The SVI is designed to measure **social vulnerability for disaster management**. Social vulnerability is framed as the socioeconomic and demographic conditions that affect how well communities can prepare for, respond to, and recover from disasters. The index is not specific to one hazard. It is meant to support emergency management generally: hurricanes, floods, earthquakes, chemical spills, heat events, and other disasters.

The paper situates SVI inside this risk equation:

[
Risk = Hazard \times (Vulnerability - Resources)
]

The point is that emergency planning often focuses on the **hazard** itself — flood depth, wind speed, infrastructure damage — but ignores the social side: poverty, age, disability, language barriers, lack of vehicles, poor housing, etc. The SVI isolates that social-vulnerability component so emergency managers can target support before, during, and after an event. 

Important warning: the paper explicitly warns against the **ecological fallacy**. A census tract can be socially vulnerable, but that does not mean every individual in that tract is helpless or vulnerable in the same way. The index is about **population groups and places**, not deterministic claims about individuals. 

---

# 2. Sélectionner les variables

The paper uses data from the **2000 U.S. Census of Population and Housing** at the **census tract level**. The authors choose census tracts because they are smaller than counties and are commonly used for public-health and planning analysis. The paper uses only tracts with **non-zero population**, giving:

[
N = 65{,}081
]

The 15 variables are grouped into **four predefined domains**:

1. Socioeconomic status
2. Household composition / disability
3. Minority status / language
4. Housing / transportation

These domains are not statistically discovered. They are conceptually chosen from the vulnerability and disaster-management literature. 

## Table of the 15 SVI variables

| Domain                             | Variable                                                |                      Type | Direction                   | Why it matters                                                                                               |
| ---------------------------------- | ------------------------------------------------------- | ------------------------: | --------------------------- | ------------------------------------------------------------------------------------------------------------ |
| Socioeconomic status               | Percent individuals below poverty                       |                Percentage | Higher = more vulnerable    | Poor households have fewer resources to prepare, evacuate, insure losses, and recover.                       |
| Socioeconomic status               | Percent civilian unemployed                             |                Percentage | Higher = more vulnerable    | Unemployment reduces financial resilience and access to employer-linked benefits.                            |
| Socioeconomic status               | Per capita income                                       | Per-capita dollar measure | **Lower = more vulnerable** | Higher income generally increases access to resources, insurance, mobility, and recovery capacity.           |
| Socioeconomic status               | Percent persons with no high school diploma             |                Percentage | Higher = more vulnerable    | Lower education can make warning information, bureaucracy, and recovery resources harder to navigate.        |
| Household composition / disability | Percent persons 65+                                     |                Percentage | Higher = more vulnerable    | Older adults may need mobility, medical, or care assistance during disasters.                                |
| Household composition / disability | Percent persons 17 or younger                           |                Percentage | Higher = more vulnerable    | Children depend on adults and require special planning in emergencies.                                       |
| Household composition / disability | Percent persons 5+ with a disability                    |                Percentage | Higher = more vulnerable    | Disabled residents may require transportation, medical, sensory, or daily-living assistance.                 |
| Household composition / disability | Percent single-parent households with children under 18 |                Percentage | Higher = more vulnerable    | One adult carries care, work, evacuation, and recovery responsibilities.                                     |
| Minority status / language         | Percent minority                                        |                Percentage | Higher = more vulnerable    | The paper links minority status to social/economic marginalization and unequal access to recovery resources. |
| Minority status / language         | Percent persons 5+ who speak English less than “well”   |                Percentage | Higher = more vulnerable    | Emergency communication and access to assistance can be harder with limited English proficiency.             |
| Housing / transportation           | Percent multi-unit structures                           |                Percentage | Higher = more vulnerable    | Dense or high-rise housing complicates evacuation and response.                                              |
| Housing / transportation           | Percent mobile homes                                    |                Percentage | Higher = more vulnerable    | Mobile homes are especially exposed to storms, flooding, and structural damage.                              |
| Housing / transportation           | Crowding                                                |                Percentage | Higher = more vulnerable    | More than one person per room can complicate sheltering, evacuation, and recovery.                           |
| Housing / transportation           | No vehicle available                                    |                Percentage | Higher = more vulnerable    | Lack of vehicle access makes evacuation difficult.                                                           |
| Housing / transportation           | Percent persons in group quarters                       |                Percentage | Higher = more vulnerable    | Institutions, dormitories, prisons, nursing homes, etc. require special evacuation planning.                 |

The Appendix defines the variables precisely, including Census table sources. For example, poverty uses individuals below the federal poverty line; crowding means occupied housing units with more than one person per room; limited English means people who speak English “not well” or “not at all”; and group quarters include both institutionalized and non-institutionalized group quarters. 

---

# 3. Traiter les données manquantes

The paper does **not** describe a detailed missing-data strategy. It does not mention multiple imputation, regression imputation, maximum likelihood, mean imputation, or deletion based on missing variables.

What it clearly states is that the index is computed across **all census tracts in the United States with a non-zero population**. So the explicit exclusion rule is:

[
\text{Keep tract } i \quad \text{if population}_i > 0
]

[
\text{Exclude tract } i \quad \text{if population}_i = 0
]

This is important because many variables are percentages or ratios. A zero-population tract would create undefined denominators or meaningless vulnerability values.

For faithful replication, the missing-data handling should be described as:

```text
Explicitly stated:
- Exclude zero-population census tracts.
- Use 65,081 non-zero-population tracts.

Not specified:
- No formal imputation method is described.
- No detailed rule is given for missing Census values after excluding zero-population tracts.
```

So, unlike SoVI, this SVI paper does **not** say “replace missing values with zero.” It mostly avoids the issue by using standardized Census variables and excluding zero-population tracts. 

---

# 4. Faire une analyse multivariée

The SVI does **not** use PCA, factor analysis, clustering, Cronbach’s alpha, varimax rotation, or eigenvalue criteria.

This is the major difference from SoVI.

In **SoVI**, the authors start with many variables, reduce them through factor analysis/PCA, interpret the resulting factors, and aggregate factor scores.

In **SVI**, the authors start with a predefined conceptual structure:

[
15 \text{ variables} \rightarrow 4 \text{ domains} \rightarrow overall SVI
]

The four domains are not latent factors discovered by the data. They are **theoretical categories**:

[
D_1 = \text{Socioeconomic Status}
]

[
D_2 = \text{Household Composition/Disability}
]

[
D_3 = \text{Minority Status/Language}
]

[
D_4 = \text{Housing/Transportation}
]

The paper does include some empirical checking in the Katrina case study: it compares elderly SVI values with Katrina drowning mortality patterns, examines mail delivery recovery data, reports that total flags and SVI are correlated at (r = 0.58), and mentions an initial regression where drowning probabilities plus the SVI household-composition domain explain 33% of variance in mail delivery data. But this is **not** a multivariate construction method for the index itself. It is more of an applied validation/illustration. 

Methodological implication: SVI is easier to reproduce and explain than SoVI, but less data-adaptive. It does not learn which variables cluster together empirically. It imposes a domain structure from theory and emergency-management relevance.

---

# 5. Normaliser les données

This is the central technical step.

The SVI uses **percentile ranks**.

For each of the 15 variables, every non-zero-population census tract is ranked relative to all other tracts. Then each rank is converted into a percentile score between 0 and 1.

The paper gives the formula:

[
\text{Percentile Rank} = \frac{Rank - 1}{N - 1}
]

where:

[
N = 65{,}081
]

for the national calculation.

Ties are handled by assigning all tied values the **smallest of the corresponding ranks**. 

## Direction of ranking

For 14 of the 15 variables, higher raw values mean higher vulnerability. So higher poverty, higher unemployment, higher disability, higher mobile-home percentage, higher no-vehicle percentage, etc., should produce a higher vulnerability percentile.

The exception is **per capita income**. Higher income indicates **lower** vulnerability, so income is ranked in the opposite direction. In practical terms, low-income tracts should receive high vulnerability percentiles.

So the direction logic is:

[
\text{For most variables: high raw value} \Rightarrow \text{high vulnerability}
]

[
\text{For income: low raw value} \Rightarrow \text{high vulnerability}
]

A clean implementation would define an “oriented vulnerability value”:

[
v_{ij} =
\begin{cases}
x_{ij}, & \text{if higher } x \text{ means higher vulnerability} \
-x_{ij}, & \text{if higher } x \text{ means lower vulnerability}
\end{cases}
]

Then compute percentile ranks on (v_{ij}), so high percentile always means high vulnerability.

## National and state calculations

The paper says this percentile-ranking process is done twice:

1. **Nationally**, across all U.S. census tracts with non-zero population.
2. **State-by-state**, within each state.

So each tract can have:

```text
national variable percentiles
national domain percentiles
national overall SVI

state variable percentiles
state domain percentiles
state overall SVI
```

The state-based version exists because emergency managers often need within-state comparisons, not only national comparisons. 

## What the SVI does not use

It does **not** use:

```text
z-scores
min-max normalization
distance-to-target
PCA scores
factor scores
geometric scaling
standard-deviation classes
```

It is a rank-based normalization method.

---

# 6. Pondérer et agréger

The SVI uses **additive aggregation of percentile ranks**.

It does not use explicit expert weights, AHP, PCA weights, DEA, entropy weights, TOPSIS, Benefit of the Doubt, or conjoint analysis.

## Step 1: Variable-level percentile ranks

For each tract (i) and variable (j):

[
PR_{ij} = \frac{Rank_{ij} - 1}{N - 1}
]

This gives 15 variable-level percentile ranks.

## Step 2: Domain raw sums

For each domain, sum the percentile ranks of variables in that domain.

For example, socioeconomic status has four variables:

[
S_{i,SES} =
PR_{poverty,i}
+
PR_{unemployment,i}
+
PR_{income,i}
+
PR_{noHS,i}
]

Household composition/disability has four variables:

[
S_{i,HHD} =
PR_{age65,i}
+
PR_{age17,i}
+
PR_{disability,i}
+
PR_{singleParent,i}
]

Minority/language has two variables:

[
S_{i,ML} =
PR_{minority,i}
+
PR_{limitedEnglish,i}
]

Housing/transportation has five variables:

[
S_{i,HT} =
PR_{multiUnit,i}
+
PR_{mobileHome,i}
+
PR_{crowding,i}
+
PR_{noVehicle,i}
+
PR_{groupQuarters,i}
]

## Step 3: Domain percentile ranks

The domain sums are then themselves converted into percentile ranks.

So for each domain:

[
D_{ik} = PercentileRank(S_{ik})
]

where (k) is one of the four domains.

This means a tract’s socioeconomic-domain score is not just the raw sum. It is the **percentile rank of that raw domain sum** relative to other tracts.

## Step 4: Overall raw sum

Then the four domain percentile ranks are summed:

[
S_{i,overall}
=============

D_{i,SES}
+
D_{i,HHD}
+
D_{i,ML}
+
D_{i,HT}
]

## Step 5: Overall SVI percentile rank

Finally, the overall sum is converted into an overall percentile rank:

[
SVI_i = PercentileRank(S_{i,overall})
]

So the full aggregation sequence is:

```text
raw Census variables
→ variable percentile ranks
→ sum variable percentiles within each domain
→ percentile rank of each domain sum
→ sum four domain percentile ranks
→ percentile rank of final sum
→ overall SVI
```

This is additive and rank-based.

## Are variables equally weighted?

Not perfectly, because the domains contain different numbers of variables.

Within each domain, variables are effectively equally weighted because their percentile ranks are simply summed.

At the final level, the four domains are effectively equally weighted because the method sums the four **domain percentile ranks**, not all 15 variables directly.

This means the minority/language domain has two variables but still contributes one full domain percentile to the final SVI, just like the housing/transportation domain with five variables. So the method gives roughly equal importance to domains, not equal importance to every individual variable.

In other words:

```text
Within-domain weighting: equal variable weights.
Across-domain weighting: equal domain weights.
```

This is different from simply summing all 15 variable percentiles.

---

# 7. Flag system

The SVI also includes a second approach: **flags**.

A flag identifies whether a tract is extremely vulnerable on a specific variable.

The threshold is:

[
PR_{ij} \geq 0.90
]

or the 90th percentile and above.

So:

[
Flag_{ij} =
\begin{cases}
1, & \text{if } PR_{ij} \geq 0.90 \
0, & \text{otherwise}
\end{cases}
]

The paper says the purpose of flags is to detect cases where a tract may have a very high value on one vulnerable characteristic, but that vulnerability gets masked by low values on other variables during averaging or summation. For example, a tract might have a very high percentage of people without vehicles, but moderate values elsewhere. Its overall SVI may not look extreme, but the no-vehicle flag reveals a very concrete evacuation problem. 

Flags are calculated at several levels:

```text
1. Individual variable flags:
   Is the tract at or above the 90th percentile for this variable?

2. Domain flag counts:
   How many variables inside this domain are flagged?

3. Overall flag count:
   How many of the 15 variables are flagged overall?
```

The paper says the toolkit provides 40 measures per tract: SVI values for the 15 variables, four domains, and overall result, plus corresponding flag measures. 

---

# 8. Tester l’incertitude et la sensibilité

The paper does **not** perform a formal sensitivity analysis.

It does not test:

```text
alternative variable sets
alternative weights
alternative domain structures
alternative normalization methods
alternative flag thresholds
alternative imputation methods
```

The Katrina case study is best understood as an **illustration / partial validation**, not a full validation.

The authors use Hurricane Katrina to show that SVI components can help explain response and recovery patterns. They examine:

```text
Katrina-related drowning deaths
flood zones deeper than 2 feet
elderly SVI component
mail delivery recovery data
socioeconomic-domain SVI
levee breach locations
tract-level death rates
Poisson probabilities for rare drowning deaths
```

Figure 2 overlays Katrina-related drowning deaths with the elderly SVI value. It shows that several tracts with statistically high drowning death rates were also in the highest elderly-vulnerability category. But the authors explicitly say they cannot prove a tract-level association with certainty because they do not have all the data needed for a complete quantitative analysis. 

Figure 3 overlays socioeconomic-domain SVI with mail-delivery recovery. The interpretation is that heavily flooded areas recovered slowly regardless of SVI, but socioeconomically vulnerable areas also recovered slowly, and areas with both heavy damage and socioeconomic vulnerability were slowest to recover. 

The limitations they acknowledge are important:

```text
1. Census data can become outdated between censuses.
2. Census data describe where people live, not necessarily where they work, study, or spend time.
3. SVI is only one component of risk; hazard, infrastructure vulnerability, and resources also matter.
4. Facility-level vulnerabilities, such as nursing homes and hospitals, may be hidden inside tract averages.
```

The nursing-home example is especially important: some elderly people died in facilities that did not evacuate, but tract-level elderly SVI can mask facility-level risk. The authors therefore recommend adding facility layers such as nursing homes, hospitals, schools, and similar locations. 

---

# 9. Revenir aux données de base

The SVI is designed to preserve interpretability better than SoVI.

Because the index keeps:

```text
variable-level percentile ranks
domain-level percentile ranks
overall SVI percentile rank
variable-level flags
domain flag counts
overall flag count
```

a user can inspect why a tract is vulnerable.

For example, two tracts can both have high overall SVI, but for different reasons:

```text
Tract A:
- high poverty
- high unemployment
- low income
- no vehicle access

Tract B:
- high elderly population
- high disability
- high group quarters
- high limited English proficiency
```

The final score tells you **where** vulnerability is high. The variables, domains, and flags tell you **why**.

The authors also mention that toolkit users wanted raw Census data, such as the number of persons in poverty, not only percentile ranks. The paper says raw Census data would be added to the toolkit for targeted interventions. 

---

# Full reconstruction of the SVI methodology

Here is the full reproducible pipeline.

## Step 0 — Choose spatial units

Use U.S. census tracts.

Exclude zero-population tracts.

[
N = 65{,}081
]

## Step 1 — Collect the 15 Census variables

From the 2000 Census:

```text
Socioeconomic status:
1. Percent below poverty
2. Percent civilian unemployed
3. Per capita income
4. Percent no high school diploma

Household composition/disability:
5. Percent age 65+
6. Percent age 17 or younger
7. Percent age 5+ with disability
8. Percent single-parent households with children under 18

Minority status/language:
9. Percent minority
10. Percent speak English less than well

Housing/transportation:
11. Percent multi-unit structures
12. Percent mobile homes
13. Crowding
14. No vehicle available
15. Percent in group quarters
```

## Step 2 — Orient variables by vulnerability direction

For 14 variables:

[
\text{higher value} = \text{higher vulnerability}
]

For per capita income:

[
\text{lower income} = \text{higher vulnerability}
]

## Step 3 — Compute variable percentile ranks

For each variable (j), rank all tracts.

Then compute:

[
PR_{ij} = \frac{Rank_{ij} - 1}{N - 1}
]

where (PR_{ij}) is the percentile rank of tract (i) on variable (j).

After this step:

[
PR_{ij} \in [0,1]
]

and higher values mean higher vulnerability.

## Step 4 — Compute raw domain sums

For each domain:

[
S_{ik} = \sum_{j \in k} PR_{ij}
]

where (k) is one of the four domains.

## Step 5 — Convert domain sums into domain percentile ranks

For each domain (k):

[
D_{ik} = PercentileRank(S_{ik})
]

So each tract receives four domain scores:

```text
Socioeconomic SVI percentile
Household composition/disability SVI percentile
Minority/language SVI percentile
Housing/transportation SVI percentile
```

## Step 6 — Compute overall raw sum

[
S_{i,overall}
=============

D_{i,SES}
+
D_{i,HHD}
+
D_{i,ML}
+
D_{i,HT}
]

## Step 7 — Convert final sum into overall SVI percentile rank

[
SVI_i = PercentileRank(S_{i,overall})
]

This is the final overall SVI.

## Step 8 — Repeat nationally and by state

Do the whole process:

```text
once across all U.S. tracts
once separately within each state
```

So the same tract can have both national and state-relative vulnerability scores.

## Step 9 — Compute variable flags

For each variable:

[
Flag_{ij} =
\begin{cases}
1, & PR_{ij} \geq 0.90 \
0, & PR_{ij} < 0.90
\end{cases}
]

## Step 10 — Compute domain and overall flag counts

For each domain:

[
FlagCount_{ik} = \sum_{j \in k} Flag_{ij}
]

Overall:

[
TotalFlags_i = \sum_{j=1}^{15} Flag_{ij}
]

---

# The most compact version

The SVI method is:

```text
1. Use 2000 Census tract data.
2. Exclude zero-population tracts.
3. Select 15 variables grouped into 4 conceptual domains.
4. Convert each variable to a percentile rank.
5. Reverse income direction so low income = high vulnerability.
6. Sum variable percentiles within each domain.
7. Percentile-rank each domain sum.
8. Sum the four domain percentile ranks.
9. Percentile-rank that final sum to get overall SVI.
10. Create flags for variables at or above the 90th percentile.
11. Count flags within domains and overall.
12. Repeat nationally and within each state.
```

So compared with SoVI:

```text
SoVI = PCA/factor-analysis index.
SVI = percentile-rank additive index.
```

For your benchmark reproduction, this SVI is much easier to implement faithfully than SoVI because the domains and aggregation rules are explicit. The fragile parts are mostly **ranking direction, tie handling, state-vs-national comparison, and whether you preserve variable/domain/flag outputs instead of only the final score**.