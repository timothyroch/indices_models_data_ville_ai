Voici le **master guide** que je proposerais de donner au codeur ou à l’agent IA qui va implémenter les différents indices. Il est volontairement général : il ne dit pas “comment coder SoVI seulement”, mais **comment reproduire n’importe quel indice composite de façon propre, réutilisable, documentée et compatible avec ton futur benchmark / ML / HGNN**.

---

# Master Guide — Bonnes pratiques pour reproduire les indices composites du benchmark VILLE_IA

## 0. Objectif du document

Ce document définit les règles que doit suivre tout codeur ou agent IA chargé de reproduire un indice composite dans le cadre du benchmark VILLE_IA.

Les indices concernés peuvent inclure, entre autres :

```text
SoVI-like
SVI-like
BRIC-like
Yang-style TOPSIS / Shannon entropy
Kaur-style LOFVI / fuzzy AHP / OWA
VILLE_IA current heuristic indices
```

Les analyses méthodologiques détaillées de chaque indice existent séparément. Elles expliquent :

```text
le cadre théorique
les variables
la normalisation
la pondération
l’agrégation
les limites de chaque méthode
```

Le présent document ne remplace pas ces analyses. Il précise **comment transformer ces méthodologies en code réutilisable**, sans rendre les indices dépendants d’un dataset particulier, d’un fichier particulier ou d’un format particulier.

L’objectif final est que chaque indice puisse être appliqué à une table de données propre, documentée et interchangeable, puis comparé aux autres indices et aux futurs modèles ML / GNN / HGNN dans un benchmark commun.

---

# 1. Principe fondamental

La règle centrale est :

> **Un indice ne doit jamais dépendre directement des fichiers bruts.**

Un indice ne devrait pas savoir si les données viennent de :

```text
CSV
JSON
GeoJSON
API CKAN
shapefile
Excel
raster
output du prototype
base de données
```

L’indice doit recevoir une table propre, déjà préparée, par exemple :

```text
zone_id | median_income | pct_65_plus | canopy_pct | floodplain_pct | no_vehicle_pct
001     | 52000         | 0.18        | 0.22       | 0.35           | 0.12
002     | 61000         | 0.12        | 0.35       | 0.04           | 0.08
003     | 43000         | 0.25        | 0.10       | 0.51           | 0.21
```

L’indice applique ensuite des opérations génériques :

```text
prendre une colonne
normaliser
orienter
pondérer
agréger
classer
exporter les résultats
```

Le code de l’indice doit donc ressembler conceptuellement à :

```python
for variable in index_recipe.variables:
    x = feature_table[variable.name]
    x_norm = normalize(x, variable.normalization)
    x_oriented = orient(x_norm, variable.direction)
    contribution = variable.weight * x_oriented
    score += contribution
```

et non à :

```python
open("specific_file_from_specific_dataset.geojson")
read_this_exact_column()
apply_this_one_hardcoded_rule()
```

---

# 2. Séparer trois choses : données brutes, features, indice

Il faut toujours distinguer trois niveaux.

## 2.1 Données brutes

Les données brutes sont ce que le laboratoire, CKAN, une API ou un portail public fournit directement.

Exemples :

```text
un GeoJSON de canopée
un CSV StatCan
un JSON météo
un fichier de zones inondables
un shapefile d’aires de diffusion
un output du prototype VILLE_IA
```

Ces données peuvent avoir n’importe quel format, n’importe quel nom de colonne, n’importe quelle structure.

## 2.2 Feature table

La feature table est une table propre, avec une ligne par unité spatiale et une colonne par variable utile.

Exemple :

```text
zone_id | income | pct_65_plus | gini | heat_island_pct | canopy_pct | floodplain_pct
```

C’est cette table que les indices doivent lire.

## 2.3 Indice

L’indice est une méthode qui transforme certaines colonnes de la feature table en score.

Exemple :

```text
SVI-like:
income, poverty, unemployment, age, disability, housing, transport
→ percentile ranks
→ domain scores
→ overall SVI
```

ou :

```text
BRIC-like:
social, economic, institutional, infrastructure, community variables
→ min-max
→ sub-index averages
→ final resilience score
```

---

# 3. Architecture générale recommandée

La structure logique du projet devrait ressembler à ceci :

```text
raw data
   ↓
ingestion
   ↓
standardisation
   ↓
feature engineering
   ↓
canonical feature table
   ↓
index recipes
   ↓
index outputs
   ↓
benchmark / ML / HGNN
```

En pratique :

```text
data/
  raw/
  processed/
  features/
  indices/
  graph/

src/
  ingestion/
  features/
  indices/
  benchmark/
  graph/
```

Les indices devraient être dans un module séparé :

```text
src/indices/
  base.py
  normalization.py
  aggregation.py
  classification.py
  sovi.py
  svi.py
  bric.py
  topsis.py
  owa.py
```

Les recettes devraient être dans des fichiers de configuration :

```text
recipes/
  sovi_like.yaml
  svi_like.yaml
  bric_like.yaml
  yang_topsis_entropy.yaml
  kaur_lofvi.yaml
  ville_ia_current.yaml
```

---

# 4. Les indices doivent être pilotés par des recettes

Chaque indice doit être séparé en deux parties :

```text
1. La logique méthodologique générale
2. La recette spécifique qui dit quelles variables utiliser et comment
```

Exemple pour un indice additif :

```yaml
name: example_flood_index
spatial_unit: zone_id

variables:
  floodplain_pct:
    normalization: minmax
    direction: positive
    weight: 0.4

  imperviousness_pct:
    normalization: minmax
    direction: positive
    weight: 0.3

  median_income:
    normalization: minmax
    direction: negative
    weight: 0.3

aggregation:
  method: weighted_sum

classification:
  method: quantile
  n_classes: 5
```

Le code lit la recette, puis applique la méthode.

Avantage :

```text
la logique de l’indice reste stable
les variables peuvent changer
le dataset peut changer
le benchmark reste reproductible
```

---

# 5. Aucun nom de colonne brut ne doit être hardcodé dans l’indice

Le code de l’indice ne doit pas contenir des choses comme :

```python
df["revenu_median_menage"]
df["Temp_Class"]
df["CruesCl"]
```

Ces noms appartiennent aux datasets sources, pas à l’indice.

Il faut d’abord les mapper vers des noms canoniques :

```text
revenu_median_menage → median_household_income
Temp_Class → heat_island_class
CruesCl → fluvial_flood_class
```

Ensuite l’indice utilise uniquement :

```text
median_household_income
heat_island_class
fluvial_flood_class
```

---

# 6. Construire un dictionnaire de variables canoniques

Le projet doit avoir un dictionnaire central des variables.

Exemple :

```yaml
median_household_income:
  description: Median household income
  unit: CAD
  type: numeric
  expected_range: [0, null]
  vulnerability_direction: negative
  source_candidates:
    - StatCan
    - municipal census table

pct_65_plus:
  description: Share of population aged 65 and older
  unit: proportion
  type: numeric
  expected_range: [0, 1]
  vulnerability_direction: positive

canopy_pct:
  description: Share of zone covered by tree canopy
  unit: proportion
  type: numeric
  expected_range: [0, 1]
  heat_vulnerability_direction: negative
```

Ce dictionnaire sert à :

```text
éviter les ambiguïtés
documenter les unités
documenter les directions
faciliter les substitutions
faciliter les tests de qualité
```

---

# 7. Chaque indice doit déclarer ses variables requises

Chaque classe ou pipeline d’indice doit pouvoir retourner :

```text
variables requises
variables optionnelles
variables manquantes
variables substituées
variables utilisées effectivement
```

Exemple :

```text
SVI-like expected variables:
- pct_poverty
- pct_unemployed
- income
- pct_no_high_school
- pct_65_plus
- pct_children
- pct_disability
- pct_single_parent
- pct_minority
- pct_limited_language
- pct_multiunit
- pct_mobile_home
- pct_crowding
- pct_no_vehicle
- pct_group_quarters
```

Si une variable manque, le code ne doit pas planter silencieusement ni l’ignorer sans trace. Il doit produire un rapport :

```text
pct_disability: missing
pct_group_quarters: missing
pct_no_vehicle: available
income: available but proxy = median_household_income instead of per_capita_income
```

---

# 8. Niveaux de reproduction : exact, adapté, inspiré

Tous les indices ne peuvent pas être reproduits avec le même niveau de fidélité.

Il faut donc classifier chaque reproduction.

## 8.1 Reproduction exacte

On utilise :

```text
les mêmes variables
le même type d’unité spatiale
les mêmes formules
les mêmes paramètres
les mêmes règles d’agrégation
```

Ce sera rare.

## 8.2 Adaptation locale

La méthodologie est reproduite, mais les variables sont adaptées au contexte local.

Exemple :

```text
SoVI-like Montréal:
même logique PCA/factor analysis
mais variables canadiennes/montréalaises plutôt que les 42 variables américaines de 1990
```

## 8.3 Reproduction de famille méthodologique

On reproduit une famille de méthode, pas l’article exact.

Exemple :

```text
Yang-style TOPSIS flood baseline:
TOPSIS + exposure/sensitivity/adaptive capacity
mais avec des proxies locaux si le modèle hydrodynamique original n’existe pas
```

Chaque output doit indiquer son niveau :

```yaml
reproduction_level: local_adaptation
```

ou :

```yaml
reproduction_level: method_family
```

---

# 9. Gérer l’ambiguïté méthodologique

Un article scientifique ne spécifie pas toujours tous les détails nécessaires pour coder.

Règle :

> **Aucune ambiguïté ne doit être transformée silencieusement en décision de code.**

Si une méthode ne précise pas un détail, il faut l’indiquer.

Exemple :

```text
Le papier ne précise pas comment les valeurs manquantes sont traitées.
Décision par défaut : median imputation.
Statut : assumption.
Sensibilité recommandée : comparer avec drop rows et zero imputation.
```

Les décisions doivent être classées :

```text
paper_explicit
paper_implicit
implementation_assumption
local_adaptation
researcher_decision_required
```

Exemple :

```yaml
missing_data:
  method: median_imputation
  status: implementation_assumption
  reason: paper does not specify missing-value handling
```

---

# 10. Gérer les valeurs manquantes

Chaque indice doit avoir une stratégie explicite de gestion des valeurs manquantes.

Options possibles :

```text
drop_units
drop_variables
zero_imputation
mean_imputation
median_imputation
domain_partial_score
missingness_flag
multiple_imputation
```

La stratégie choisie doit être configurable.

Exemple :

```yaml
missing_data:
  strategy: median_imputation
  add_missing_flags: true
```

Le code doit produire un rapport :

```text
nombre de valeurs manquantes par variable
nombre d’unités affectées
stratégie utilisée
variables exclues
unités exclues
```

Dans un benchmark sérieux, il est recommandé de tester plusieurs stratégies de missing data lorsque cela peut affecter les résultats.

---

# 11. Normalisation : toujours configurable et documentée

Chaque variable doit déclarer sa méthode de normalisation.

Méthodes possibles :

```text
minmax
zscore
percentile_rank
vector_normalization
distance_to_goal
binary
ordinal
none
custom_transformation
```

Exemple :

```yaml
normalization:
  method: minmax
  scope: full_dataset
```

ou :

```yaml
normalization:
  method: percentile_rank
  ranking_direction: descending
```

Le code doit sauvegarder les paramètres de normalisation :

```text
min
max
mean
std
ranking scope
number of units
treatment of ties
```

Pourquoi c’est important :

```text
sans ces paramètres, on ne peut pas reproduire le score
sans ces paramètres, on risque de comparer des choses incohérentes
sans ces paramètres, on ne peut pas appliquer l’indice à un nouveau dataset proprement
```

---

# 12. Orientation des variables

Chaque variable doit avoir une direction.

Exemples :

```text
positive:
plus la variable est haute, plus la vulnérabilité est haute

negative:
plus la variable est haute, plus la vulnérabilité est basse

ambiguous:
la direction demande une décision théorique

nonlinear:
la relation n’est pas simplement croissante ou décroissante
```

Exemple :

```yaml
median_income:
  direction: negative

pct_65_plus:
  direction: positive

canopy_pct:
  direction: negative_for_heat_vulnerability

floodplain_pct:
  direction: positive
```

Le code doit appliquer l’orientation après normalisation, sauf si la méthode originale indique autre chose.

Exemple :

```python
if direction == "negative":
    x = 1 - x
```

Pour les méthodes comme SVI, l’orientation peut être gérée par le sens du ranking.

Pour SoVI, l’orientation des facteurs doit être documentée séparément, car elle dépend de l’interprétation des facteurs.

---

# 13. Pondération

Les poids ne doivent pas être hardcodés dans la logique centrale.

Ils doivent venir :

```text
de la recette
de la méthode
d’une table de poids
d’une procédure calculée
```

Types de poids possibles :

```text
equal
manual
expert
AHP
fuzzy_AHP
entropy
PCA/factor loadings
OWA positional weights
learned weights
none
```

Chaque poids doit avoir une provenance :

```yaml
weight:
  value: 0.25
  source: paper
  status: paper_explicit
```

ou :

```yaml
weight:
  method: equal
  source: implementation
  status: method_family_default
```

---

# 14. Agrégation

Chaque indice doit déclarer explicitement son mode d’agrégation.

Méthodes possibles :

```text
sum
mean
weighted_sum
geometric_mean
domain_sum_then_rank
TOPSIS
OWA
multiplicative
PCA/factor_score_sum
custom
```

Exemples :

```text
SoVI:
factor scores → sign orientation → additive sum

SVI:
variable percentile ranks → domain sums → domain ranks → overall rank

BRIC:
variables → sub-index averages → sum of five sub-indices

Yang:
TOPSIS closeness for E/S/AC → multiplicative vulnerability → entropy inhomogeneity correction

Kaur:
normalized indicators → OWA dimension scores → OWA final LOFVI
```

Le code doit conserver les scores intermédiaires.

---

# 15. Ne jamais seulement exporter le score final

Un indice composite ne doit pas produire uniquement :

```text
zone_id | final_score
```

Il doit aussi produire :

```text
variables normalisées
variables orientées
contributions pondérées
sub-index scores
domain scores
classification
missingness flags
metadata
```

Exemple de sortie minimale :

```text
zone_id
final_score
final_class
social_subscore
physical_subscore
economic_subscore
environmental_subscore
missing_count
quality_flag
```

Exemple de sortie détaillée :

```text
zone_id
raw_income
norm_income
oriented_income
contribution_income
raw_pct_65_plus
norm_pct_65_plus
contribution_pct_65_plus
...
final_score
```

Cette granularité est essentielle pour :

```text
debugger
expliquer les résultats
comparer les indices
faire de la sensibilité
alimenter le HGNN
```

---

# 16. Métadonnées obligatoires

Chaque exécution d’un indice doit produire un fichier metadata.

Exemple :

```yaml
index_name: SVI-like
run_id: 2026-05-15_svi_v1
created_at: 2026-05-15
input_file: zone_features_v0.parquet
recipe_file: recipes/svi_like.yaml
spatial_unit: ADIDU
n_units: 3182

variables_used:
  - median_income
  - pct_65_plus
  - pct_no_vehicle

variables_missing:
  - pct_disability

normalization:
  method: percentile_rank
  scope: dataset
  ties: smallest_rank

missing_data:
  strategy: median_imputation

aggregation:
  method: domain_sum_then_percentile_rank

classification:
  method: quantile
  n_classes: 5

reproduction_level: local_adaptation
```

Sans metadata, le résultat n’est pas research-grade.

---

# 17. Le code doit être reproductible

Chaque run doit pouvoir être reproduit.

Il faut sauvegarder :

```text
input dataset path
input dataset version
recipe version
code commit hash si possible
normalization parameters
random seed si applicable
output path
timestamp
```

Pour les méthodes stochastiques ou les algorithmes avec hasard :

```text
random seed obligatoire
```

Pour les indices déterministes :

```text
le même input + la même recette doivent produire exactement le même output
```

---

# 18. Validation de la feature table avant calcul

Avant de calculer un indice, le code doit valider la feature table.

Checks minimaux :

```text
la colonne ID existe
les variables requises existent ou sont déclarées manquantes
les types sont numériques lorsque nécessaire
les proportions sont dans un range plausible
les valeurs ne sont pas toutes constantes
les valeurs manquantes sont détectées
les doublons d’ID sont détectés
```

Exemple :

```text
pct_65_plus doit être entre 0 et 1 ou entre 0 et 100
income doit être >= 0
zone_id doit être unique
```

Si une variable est constante, certaines normalisations deviennent impossibles. Le code doit gérer ce cas explicitement.

---

# 19. Gérer les unités

Chaque variable doit avoir une unité déclarée.

Exemples :

```text
CAD
percent
proportion
persons/km²
km
m
ordinal_class
binary
count
```

Le code ne doit pas deviner silencieusement si une variable est en pourcentage ou en proportion.

Exemple problématique :

```text
pct_65_plus = 18
```

Est-ce 18 % ou 18.0 comme proportion impossible?

Il faut une règle claire :

```yaml
pct_65_plus:
  unit: percent
  scale: 0_100
```

ou :

```yaml
pct_65_plus:
  unit: proportion
  scale: 0_1
```

---

# 20. Gérer les unités spatiales

Chaque indice doit déclarer à quelle unité spatiale il s’applique.

Exemples :

```text
dissemination area
census tract
municipality
neighborhood
grid cell
building
road segment
```

La table d’entrée doit avoir une colonne stable :

```text
zone_id
ADIDU
CSDUID
neighborhood_id
```

Le code ne doit pas utiliser un `range(len(df))` comme identifiant principal, sauf comme identifiant temporaire.

Un identifiant stable est nécessaire pour :

```text
joindre les datasets
comparer les indices
construire les graphes
faire des splits train/test
reproduire les résultats
```

---

# 21. Spécificité des indices spatiaux

Pour les indices qui dépendent de couches spatiales, comme les indices de crue, chaleur ou inondation, il faut séparer :

```text
calcul spatial
calcul indiciel
```

Exemple :

```text
canopy polygons + zones
→ canopy_pct par zone
```

Ensuite :

```text
canopy_pct
→ normalisation / orientation / score
```

L’indice ne devrait pas lui-même faire directement tous les overlays spatiaux, sauf si c’est explicitement sa responsabilité dans un pipeline séparé.

Les features spatiales doivent être précalculées autant que possible :

```text
floodplain_coverage_pct
canopy_coverage_pct
heat_island_coverage_pct
distance_to_nearest_hospital
road_density
service_count_within_1km
```

---

# 22. Gestion des proxys

Dans une reproduction locale, une variable originale peut ne pas exister.

Exemple :

```text
SVI original: per capita income
local data: median household income
```

Il faut documenter :

```yaml
original_variable: per_capita_income
local_proxy: median_household_income
proxy_quality: medium
reason: per capita income unavailable in current feature table
conceptual_risk: household income is not equivalent to individual income
```

Aucun proxy ne doit être utilisé sans trace.

Chaque proxy doit avoir :

```text
nom original
proxy local
justification
risque conceptuel
niveau de confiance
```

---

# 23. Sensibilité et variantes

Chaque indice devrait idéalement pouvoir être exécuté en plusieurs variantes.

Exemples :

```text
missing data = zero / median / drop
normalization = minmax / zscore / percentile
classification = quantile / fixed / std dev
weights = original / equal / sensitivity perturbation
```

Le but n’est pas de compliquer le code dès le départ, mais de permettre :

```text
robustness checks
benchmark sérieux
analyse de fragilité
```

Exemple :

```yaml
variants:
  - name: original_like
    missing_data: zero
  - name: median_imputation
    missing_data: median
  - name: equal_weights
    weights: equal
```

---

# 24. Comparabilité entre indices

Tous les indices ne mesurent pas exactement la même chose.

Exemples :

```text
SoVI mesure la vulnérabilité sociale
SVI mesure la vulnérabilité sociale opérationnelle
BRIC mesure la résilience communautaire
Yang mesure la vulnérabilité aux inondations
Kaur mesure la vulnérabilité locale aux inondations
```

Le code ne doit pas forcer tous les scores à être interprétés de la même façon.

Chaque output doit déclarer :

```text
construct_measured: social_vulnerability / resilience / flood_vulnerability / heat_vulnerability
score_direction: higher_is_worse / higher_is_better
score_range
```

Exemple :

```yaml
score_direction: higher_is_more_vulnerable
```

ou :

```yaml
score_direction: higher_is_more_resilient
```

C’est essentiel, surtout pour BRIC où un score élevé signifie plus de résilience, alors que pour SoVI/SVI un score élevé signifie plus de vulnérabilité.

---

# 25. Sorties standardisées pour le benchmark

Même si les indices ont des logiques différentes, leurs outputs doivent avoir une structure commune.

Exemple :

```text
zone_id
index_name
score_raw
score_normalized_0_1
score_direction
rank
percentile
class
reproduction_level
run_id
```

Cela permet de comparer :

```text
corrélations
rankings
top-k overlap
cartes de désaccord
robustesse
```

Si l’indice produit un score entre 0 et 5, 0 et 1, ou un score non borné, on conserve le score original et on ajoute une version standardisée pour comparaison.

---

# 26. Ne pas confondre score, label et feature

Un score d’indice peut servir de trois façons différentes :

```text
1. résultat final d’un indice
2. feature pour un modèle ML/HGNN
3. weak label / pseudo-label
```

Le code doit garder cette distinction claire.

Exemple :

```text
score_svi
```

peut être :

```text
benchmark output
```

mais aussi :

```text
node feature in HGNN
```

ou :

```text
weak target for a preliminary model
```

Il faut éviter de réutiliser un score comme feature lorsque ce même score est la target, sauf si l’objectif est explicitement de tester une reproduction triviale.

---

# 27. Tests nécessaires

Chaque indice doit avoir des tests.

Tests minimaux :

```text
test des colonnes requises
test de normalisation
test d’orientation
test de pondération
test d’agrégation
test sur mini-dataset synthétique
test des valeurs manquantes
test que le score final a la bonne direction
test que les métadonnées sont exportées
```

Exemple de mini-dataset synthétique :

```text
zone A: faible vulnérabilité attendue
zone B: forte vulnérabilité attendue
```

Le test doit vérifier que :

```text
score_B > score_A
```

pour un indice de vulnérabilité.

---

# 28. Mini-dataset de validation obligatoire

Chaque indice devrait avoir un mini-dataset artificiel avec 3 à 5 lignes pour vérifier la logique.

Exemple :

```text
zone_id | income | pct_65_plus | flood_pct
A       | 90000  | 0.05        | 0.00
B       | 30000  | 0.30        | 0.80
C       | 60000  | 0.15        | 0.40
```

On doit pouvoir prédire qualitativement :

```text
A faible vulnérabilité
B forte vulnérabilité
C moyenne
```

Ce test ne valide pas scientifiquement l’indice, mais il valide que le code n’est pas inversé.

---

# 29. Erreurs à éviter absolument

## 29.1 Hardcoder les colonnes source

Mauvais :

```python
df["revenu_median_menage"]
```

Bon :

```python
df[recipe.variables["income"].canonical_name]
```

## 29.2 Mélanger nettoyage et indice

Mauvais :

```python
class SVI:
    def compute():
        download_csv()
        parse_json()
        clean_income()
        compute_percentile()
```

Bon :

```text
ingestion pipeline → feature table
SVI pipeline → score
```

## 29.3 Ne pas sauvegarder les paramètres de normalisation

Mauvais :

```python
x = (x - x.min()) / (x.max() - x.min())
```

sans garder min/max.

Bon :

```text
sauvegarder min, max, scope, date, dataset
```

## 29.4 Ignorer une variable manquante silencieusement

Mauvais :

```python
if var not in df:
    continue
```

Bon :

```text
report variable missing
apply configured missing strategy
save metadata
```

## 29.5 Changer la méthodologie sans le dire

Mauvais :

```text
le papier utilise percentile rank, mais le code utilise minmax parce que c’est plus simple
```

Bon :

```text
paper method: percentile rank
implemented method: minmax
status: variant, not faithful reproduction
```

---

# 30. Structure recommandée d’une classe d’indice

Une classe d’indice devrait idéalement avoir une interface du genre :

```python
class CompositeIndex:
    def __init__(self, recipe):
        self.recipe = recipe

    def validate_inputs(self, feature_table):
        ...

    def fit(self, feature_table):
        ...

    def transform(self, feature_table):
        ...

    def fit_transform(self, feature_table):
        ...

    def export_metadata(self):
        ...

    def explain(self, zone_id):
        ...
```

Pour certains indices, `fit` est nécessaire :

```text
PCA
factor analysis
minmax fitted on dataset
percentile ranks
entropy weights
```

Pour d’autres, la méthode est purement déterministe avec paramètres fixes.

---

# 31. Indices avec apprentissage ou calcul global

Certains indices ont besoin de calculer des paramètres à partir de tout le dataset.

Exemples :

```text
min/max pour minmax
mean/std pour zscore
ranks pour percentile
PCA loadings pour SoVI-like
entropy weights pour entropy methods
OWA weights selon n
```

Ces paramètres doivent être considérés comme des paramètres “appris” ou “fittés” sur le dataset.

Donc :

```python
index.fit(train_data)
index.transform(test_data)
```

sera nécessaire si on utilise les indices dans un cadre ML avec train/test split.

Pour une simple carte descriptive, on peut fit sur tout le dataset.
Pour un benchmark prédictif, il faudra éviter la fuite de données.

---

# 32. Compatibilité avec benchmark ML

Si les indices sont utilisés dans un benchmark ML, il faut éviter le data leakage.

Exemple dangereux :

```text
normaliser avec min/max de tout le dataset
puis évaluer sur test set
```

Dans un benchmark prédictif, il faut :

```text
fit normalization on train
apply to validation/test
```

Même chose pour :

```text
PCA
factor analysis
entropy weights
percentile ranks
imputation values
```

Pour une reproduction descriptive de l’indice, ce n’est pas forcément nécessaire. Mais pour une évaluation ML, c’est essentiel.

Le code doit donc supporter deux modes :

```yaml
mode: descriptive_full_dataset
```

et :

```yaml
mode: predictive_train_test
```

---

# 33. Compatibilité avec HGNN

Les indices doivent produire des outputs utilisables par le HGNN.

Exemples :

```text
score_svi
score_sovi
score_bric
physical_subscore
social_subscore
economic_subscore
environmental_subscore
```

Ces scores peuvent devenir des features de nœuds :

```text
data["zone"].x
```

Mais le HGNN aura besoin de plus que les scores finaux. Il aura besoin de :

```text
variables brutes
variables normalisées
subscores
labels/pseudo-labels
spatial IDs
geometry / relations
```

Donc il faut préserver autant que possible les données intermédiaires.

---

# 34. Explicabilité

Chaque indice doit pouvoir expliquer un score.

Fonction souhaitée :

```python
index.explain(zone_id)
```

Sortie possible :

```text
zone_id: 001
final_score: 0.78
main_contributors:
  - floodplain_pct: +0.22
  - pct_65_plus: +0.18
  - low_income: +0.14
protective_contributors:
  - canopy_pct: -0.08
missing_variables:
  - pct_disability
```

Cela sera très utile pour comparer avec les explications du HGNN.

---

# 35. Documentation minimale par indice

Chaque indice implémenté doit avoir un petit document ou rapport généré automatiquement :

```text
nom de l’indice
article source
reproduction level
construct measured
variables used
variables missing
normalization method
weighting method
aggregation method
classification method
score direction
limitations
```

Exemple :

```text
SVI-like
Source: Flanagan et al. 2011
Reproduction: local adaptation
Construct: social vulnerability
Method: percentile ranks + predefined domains
Score direction: higher = more vulnerable
```

---

# 36. Ordre de travail recommandé pour chaque indice

Pour chaque indice, le codeur doit suivre cet ordre :

```text
1. Lire le methodological brief de l’indice
2. Identifier les variables requises
3. Créer la recette YAML
4. Vérifier la feature table disponible
5. Mapper variables originales → variables canoniques
6. Implémenter ou réutiliser les opérations nécessaires
7. Produire score + sous-scores + contributions
8. Exporter metadata
9. Tester sur mini-dataset synthétique
10. Tester sur vraie feature table
11. Générer rapport de run
12. Comparer distributions et rankings
```

---

# 37. Critères d’acceptation d’une implémentation

Une implémentation est acceptable si :

```text
elle ne dépend pas directement des fichiers bruts
elle utilise une recette/config
elle valide les colonnes d’entrée
elle gère les valeurs manquantes explicitement
elle sauvegarde les paramètres de normalisation
elle exporte les sous-scores/contributions
elle exporte les métadonnées
elle a des tests sur mini-dataset
elle documente les écarts avec le papier original
elle peut être réutilisée sur un nouveau dataset avec une nouvelle recette
```

Une implémentation n’est pas acceptable si :

```text
elle hardcode les colonnes du dataset actuel
elle ignore silencieusement les variables manquantes
elle produit seulement un score final
elle ne sauvegarde pas les paramètres
elle mélange ingestion brute et calcul indiciel
elle ne distingue pas reproduction exacte vs adaptation locale
elle inverse des directions sans justification
```

---

# 38. Relation entre les cinq indices choisis

Chaque indice couvre une famille méthodologique différente.

```text
SoVI-like:
PCA / factor analysis / latent social vulnerability

SVI-like:
percentile ranks / predefined social domains / flags

BRIC-like:
resilience capacities / minmax / equal-weight subindices

Yang-style:
flood vulnerability / TOPSIS / Delphi weights / entropy inhomogeneity

Kaur-style LOFVI:
localized flood vulnerability / fuzzy AHP / OWA aggregation
```

Le code doit donc éviter de forcer tous les indices dans une même formule additive.

Il faut plutôt créer des briques communes :

```text
normalization
orientation
weighting
aggregation
classification
metadata
```

et laisser chaque indice combiner ces briques différemment.

---

# 39. Le rôle du codeur ou agent IA face aux ambiguïtés

Le codeur ne doit pas prendre des décisions scientifiques lourdes seul.

Il doit classer les ambiguïtés :

```text
peut être résolu par le papier
peut être résolu par la recette
nécessite une décision du chercheur
nécessite une variante de sensibilité
```

Exemple :

```text
Le papier ne précise pas comment gérer les missing values.
→ proposer options
→ choisir default configurable
→ marquer assumption
```

Exemple :

```text
Une variable originale n’existe pas localement.
→ proposer proxy
→ marquer proxy
→ demander validation
```

---

# 40. Résumé opérationnel pour le codeur

Le codeur doit retenir ceci :

> Tu ne codes pas cinq scripts jetables. Tu construis une petite infrastructure de reproduction d’indices composites.

Chaque indice doit être :

```text
configurable
documenté
reproductible
séparé des données brutes
capable d’exporter ses intermédiaires
compatible avec un benchmark
compatible avec une future feature table HGNN
```

Le pipeline idéal est :

```text
raw data
→ canonical feature table
→ index recipe
→ index computation
→ score + sub-scores + metadata
→ benchmark comparison
→ ML/HGNN features
```

La bonne question n’est pas :

> “Comment coder cet indice pour ce fichier précis?”

La bonne question est :

> “Comment coder cette méthodologie pour qu’elle puisse être appliquée à une table de features propre, peu importe d’où viennent les données?”

C’est ce principe qui doit guider toutes les implémentations.