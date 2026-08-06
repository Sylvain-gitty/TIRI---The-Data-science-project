# Tier 1b lexical block — results and the shuffled-brief control

`1852` labelled papers, 6 use cases, 22 lexical features. Plain LogisticRegression throughout.

> Read `recall_at_10pct` against its ceiling, not against 1.0. These pools are ~58% positive, so reviewing the top 10% of a pool can recover at most ~0.17 of the positives even with perfect ranking. The number becomes informative at production prevalence, not here.

## 1. Leave-one-use-case-out (chooses central defaults, not a production number)

| index                                       | roc_auc | pr_auc | recall_at_10pct |
|---------------------------------------------|---------|--------|-----------------|
| paper embedding (384-d, current baseline)   | 0.537   | 0.598  | 0.091           |
| Tier 1b lexical — real briefs               | 0.642   | 0.696  | 0.131           |
| Tier 1b + embedding                         | 0.576   | 0.632  | 0.106           |
| Tier 1b lexical — SHUFFLED briefs (control) | 0.488   | 0.577  | 0.092           |

Per use case, ROC-AUC:

| use_case         | paper embedding (384-d, current baseline) | Tier 1b lexical — real briefs | Tier 1b + embedding | Tier 1b lexical — SHUFFLED briefs (control) |
|------------------|-------------------------------------------|-------------------------------|---------------------|---------------------------------------------|
| carbon_capture   | 0.616                                     | 0.744                         | 0.653               | 0.473                                       |
| cement_binders   | 0.645                                     | 0.890                         | 0.741               | 0.521                                       |
| ner              | 0.473                                     | 0.673                         | 0.532               | 0.483                                       |
| soil_microbiome  | 0.527                                     | 0.594                         | 0.538               | 0.490                                       |
| solar_leo        | 0.443                                     | 0.413                         | 0.424               | 0.455                                       |
| tech_forecasting | 0.515                                     | 0.540                         | 0.565               | 0.504                                       |

**Control:** real briefs beat wrong briefs on **5/6** use cases, mean ROC-AUC gap **+0.155** (between-derangement sd 0.037).

## 2. Within-silo, author-grouped 5-fold (this is the production shape)

| use_case         | paper embedding | Tier 1b lexical | Tier 1b + embedding | Tier 1b — SHUFFLED (control) |
|------------------|-----------------|-----------------|---------------------|------------------------------|
| carbon_capture   | 0.754           | 0.762           | 0.786               | 0.609                        |
| cement_binders   | 0.824           | 0.881           | 0.848               | 0.679                        |
| ner              | 0.727           | 0.717           | 0.733               | 0.527                        |
| soil_microbiome  | 0.695           | 0.679           | 0.692               | 0.613                        |
| solar_leo        | 0.662           | 0.610           | 0.675               | 0.562                        |
| tech_forecasting | 0.702           | 0.657           | 0.711               | 0.551                        |

Seed-to-seed sd (Tier 1b): carbon_capture 0.011, cement_binders 0.010, ner 0.008, soil_microbiome 0.011, solar_leo 0.009, tech_forecasting 0.010

## 3. Which part of the block carries it (LOGO mean ROC-AUC)

| index                       | mean LOGO ROC-AUC |
|-----------------------------|-------------------|
| bm25                        | 0.637             |
| overlap                     | 0.623             |
| length_normalised           | 0.550             |
| all                         | 0.642             |
| all minus length_normalised | 0.638             |

