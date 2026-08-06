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
| carbon_capture   | 0.616                                     | 0.744                         | 0.654               | 0.473                                       |
| cement_binders   | 0.645                                     | 0.890                         | 0.741               | 0.521                                       |
| ner              | 0.473                                     | 0.673                         | 0.532               | 0.483                                       |
| soil_microbiome  | 0.527                                     | 0.594                         | 0.538               | 0.490                                       |
| solar_leo        | 0.443                                     | 0.413                         | 0.423               | 0.455                                       |
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

## 4. Warm-start curve — ROC-AUC vs. number of labels from the target use case

`n=0` is the transfer baseline (trained on the other five use cases). Labels are drawn at random, not stratified, because that is what arriving labels look like. Mean of 25 seeds.

Mean across all six use cases:

| n_labels | Tier 1b | Tier 1b + embedding | embedding |
|----------|---------|---------------------|-----------|
| 0        | 0.642   | 0.576               | 0.537     |
| 25       | 0.635   | 0.681               | 0.670     |
| 50       | 0.665   | 0.711               | 0.699     |
| 100      | 0.689   | 0.733               | 0.720     |
| 200      | 0.717   | 0.742               | 0.726     |

Per use case — Tier 1b + embedding:

| use_case         | 0     | 25    | 50    | 100   | 200   |
|------------------|-------|-------|-------|-------|-------|
| carbon_capture   | 0.654 | 0.746 | 0.766 | 0.784 | 0.789 |
| cement_binders   | 0.741 | 0.840 | 0.855 | 0.859 | 0.859 |
| ner              | 0.532 | 0.650 | 0.681 | 0.699 | 0.736 |
| soil_microbiome  | 0.538 | 0.662 | 0.693 | 0.712 | 0.695 |
| solar_leo        | 0.423 | 0.562 | 0.615 | 0.653 | 0.663 |
| tech_forecasting | 0.565 | 0.627 | 0.658 | 0.688 | 0.709 |

Per use case — Tier 1b:

| use_case         | 0     | 25    | 50    | 100   | 200   |
|------------------|-------|-------|-------|-------|-------|
| carbon_capture   | 0.744 | 0.676 | 0.718 | 0.740 | 0.778 |
| cement_binders   | 0.890 | 0.823 | 0.854 | 0.866 | 0.877 |
| ner              | 0.673 | 0.635 | 0.655 | 0.681 | 0.719 |
| soil_microbiome  | 0.594 | 0.533 | 0.589 | 0.642 | 0.682 |
| solar_leo        | 0.413 | 0.572 | 0.581 | 0.593 | 0.605 |
| tech_forecasting | 0.540 | 0.568 | 0.590 | 0.611 | 0.639 |

Per use case — embedding:

| use_case         | 0     | 25    | 50    | 100   | 200   |
|------------------|-------|-------|-------|-------|-------|
| carbon_capture   | 0.616 | 0.718 | 0.737 | 0.758 | 0.758 |
| cement_binders   | 0.645 | 0.830 | 0.841 | 0.843 | 0.836 |
| ner              | 0.473 | 0.633 | 0.666 | 0.682 | 0.722 |
| soil_microbiome  | 0.527 | 0.663 | 0.691 | 0.709 | 0.693 |
| solar_leo        | 0.443 | 0.553 | 0.607 | 0.644 | 0.656 |
| tech_forecasting | 0.515 | 0.623 | 0.652 | 0.682 | 0.689 |

