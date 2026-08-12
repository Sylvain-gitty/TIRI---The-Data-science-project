# SYNERGY validation — the recall lens at realistic prevalence

> **Status: current.** External validation at realistic prevalence (1.7–14.8% positive) on systematic reviews this project had no hand in labelling — the least self-graded evidence in the repo.

| use_case_key      | n    | n_included | prevalence | wss_ceiling | exp_pos_in_25_random |
|-------------------|------|------------|------------|-------------|----------------------|
| Menon_2022        | 975  | 74         | 0.076      | 0.878       | 1.900                |
| Sep_2021          | 271  | 40         | 0.148      | 0.810       | 3.690                |
| van_der_Waal_2022 | 1970 | 33         | 0.017      | 0.934       | 0.420                |

`exp_pos_in_25_random` is the expected number of positives in a random 25-label bootstrap. Below 1.0 it means most such draws contain no positive at all and cannot train a classifier — see §3.

## 1. Within-review (each SYNERGY review is a silo)

**wss_at_95**

| index                               | Sep_2021 | Menon_2022 | van_der_Waal_2022 | MEAN  |
|-------------------------------------|----------|------------|-------------------|-------|
| qwen3-8b: embedding                 | 0.308    | 0.711      | 0.532             | 0.517 |
| jasper: embedding + lexical         | 0.245    | 0.661      | 0.624             | 0.510 |
| qwen3-8b: embedding + lexical       | 0.149    | 0.619      | 0.606             | 0.458 |
| qwen3-4b: embedding + lexical       | 0.163    | 0.609      | 0.582             | 0.451 |
| jasper: embedding                   | 0.212    | 0.681      | 0.421             | 0.438 |
| qwen3-4b: embedding                 | 0.182    | 0.698      | 0.245             | 0.375 |
| lexical — SHUFFLED briefs (control) | 0.092    | 0.035      | 0.319             | 0.149 |
| lexical only (Tier 1b)              | 0.017    | 0.020      | 0.119             | 0.052 |

**recall_at_10pct**

| index                               | Sep_2021 | Menon_2022 | van_der_Waal_2022 | MEAN  |
|-------------------------------------|----------|------------|-------------------|-------|
| qwen3-8b: embedding + lexical       | 0.370    | 0.722      | 0.776             | 0.622 |
| qwen3-8b: embedding                 | 0.370    | 0.776      | 0.715             | 0.620 |
| qwen3-4b: embedding                 | 0.335    | 0.778      | 0.745             | 0.620 |
| qwen3-4b: embedding + lexical       | 0.320    | 0.719      | 0.812             | 0.617 |
| jasper: embedding + lexical         | 0.315    | 0.724      | 0.745             | 0.595 |
| jasper: embedding                   | 0.330    | 0.743      | 0.655             | 0.576 |
| lexical only (Tier 1b)              | 0.140    | 0.081      | 0.455             | 0.225 |
| lexical — SHUFFLED briefs (control) | 0.100    | 0.127      | 0.279             | 0.169 |

**roc_auc**

| index                               | Sep_2021 | Menon_2022 | van_der_Waal_2022 | MEAN  |
|-------------------------------------|----------|------------|-------------------|-------|
| qwen3-8b: embedding                 | 0.798    | 0.958      | 0.907             | 0.888 |
| qwen3-8b: embedding + lexical       | 0.775    | 0.944      | 0.928             | 0.883 |
| qwen3-4b: embedding + lexical       | 0.759    | 0.944      | 0.936             | 0.879 |
| jasper: embedding + lexical         | 0.748    | 0.947      | 0.932             | 0.876 |
| qwen3-4b: embedding                 | 0.773    | 0.958      | 0.891             | 0.874 |
| jasper: embedding                   | 0.767    | 0.949      | 0.883             | 0.866 |
| lexical — SHUFFLED briefs (control) | 0.570    | 0.566      | 0.739             | 0.625 |
| lexical only (Tier 1b)              | 0.552    | 0.535      | 0.753             | 0.613 |

## 2. Shuffled-brief control, at realistic prevalence

| review            | real_brief | wrong_brief | gap    |
|-------------------|------------|-------------|--------|
| Sep_2021          | 0.552      | 0.570       | -0.019 |
| Menon_2022        | 0.535      | 0.566       | -0.031 |
| van_der_Waal_2022 | 0.753      | 0.739       | 0.015  |

Real briefs beat wrong briefs on **1/3** reviews, mean gap **-0.012** ROC-AUC.

## 3. Warm-start at low prevalence — and how often 25 random labels are unusable

Share of random label draws with no positive at all (so no model can be fit):

| review            | 25    | 50    | 100   | 200   |
|-------------------|-------|-------|-------|-------|
| Menon_2022        | 0.150 | 0.000 | 0.000 | 0.000 |
| Sep_2021          | 0.050 | 0.000 | 0.000 | 0.000 |
| van_der_Waal_2022 | 0.570 | 0.280 | 0.220 | 0.050 |

WSS@95 by budget (mean over reviews, unusable draws excluded):

| variant                             | 25    | 50    | 100   | 200   |
|-------------------------------------|-------|-------|-------|-------|
| jasper: embedding                   | 0.344 | 0.382 | 0.420 | 0.437 |
| jasper: embedding + lexical         | 0.222 | 0.282 | 0.354 | 0.397 |
| lexical only (Tier 1b)              | 0.035 | 0.047 | 0.053 | 0.050 |
| lexical — SHUFFLED briefs (control) | 0.099 | 0.094 | 0.109 | 0.119 |
| qwen3-4b: embedding                 | 0.344 | 0.381 | 0.431 | 0.477 |
| qwen3-4b: embedding + lexical       | 0.225 | 0.319 | 0.371 | 0.419 |
| qwen3-8b: embedding                 | 0.356 | 0.383 | 0.436 | 0.480 |
| qwen3-8b: embedding + lexical       | 0.218 | 0.308 | 0.382 | 0.414 |

