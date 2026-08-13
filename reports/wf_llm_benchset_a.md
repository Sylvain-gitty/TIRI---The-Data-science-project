# Set A — LLM screening at 2.19% prevalence

8 SYNERGY collections, 62,229 papers, 1,362 relevant. Scored on the 9,993-row case-control sample and reweighted to population prevalence (`scripts/benchset_metrics.py`); the sampling gate is `reports/wf_llm_benchset_a_baselines.md`.

**The bar:** the cold-start cosine-to-brief ranker already on disk — mean AUC **0.763**, mean WSS@95 **0.357**. Marking everything relevant scores F2 **0.101**.

## 1. Ranking — per collection, against the cosine baseline

| use_case          | cosine | gemma-4-31b-it | gpt-oss-20b | nemotron-3-super-120b-a12b | qwen3.5-397b-a17b |
|-------------------|--------|----------------|-------------|----------------------------|-------------------|
| brouwer_2019      | 0.993  | 0.938          | 0.916       | 0.992                      | 0.984             |
| van_dis_2020      | 0.881  | 0.959          | 0.931       | 0.933                      | 0.938             |
| leenaars_2020     | 0.862  | 0.758          | 0.797       | 0.878                      | 0.753             |
| van_der_valk_2021 | 0.853  | 0.848          | 0.808       | 0.841                      | 0.828             |
| muthu_2021        | 0.750  | 0.610          | 0.729       | 0.698                      | 0.727             |
| nelson_2002       | 0.714  | 0.761          | 0.786       | 0.797                      | 0.774             |
| sep_2021          | 0.613  | 0.907          | 0.886       | 0.840                      | 0.899             |
| moran_2021        | 0.441  | 0.408          | 0.413       | 0.451                      | 0.422             |

Against the 0.03 noise floor, of 8 collections:

| index                      | beats cosine | loses to cosine | within floor |
|----------------------------|--------------|-----------------|--------------|
| gemma-4-31b-it             | 3            | 4               | 1            |
| gpt-oss-20b                | 3            | 3               | 2            |
| nemotron-3-super-120b-a12b | 3            | 1               | 4            |
| qwen3.5-397b-a17b          | 3            | 1               | 4            |

## 2. Headline, P2, all 9,993 sampled rows

| model                      | mean_auc | mean_wss95 | mean_f2_star | mean_f2_own | recall_own | screened_own | tie_frac | coverage | beats_cosine |
|----------------------------|----------|------------|--------------|-------------|------------|--------------|----------|----------|--------------|
| gemma-4-31b-it             | 0.774    | 0.308      | 0.490        | 0.293       | 0.486      | 0.148        | 0.999    | 1.000    | 3.000        |
| gpt-oss-20b                | 0.783    | 0.211      | 0.504        | 0.402       | 0.490      | 0.103        | 0.998    | 1.000    | 3.000        |
| nemotron-3-super-120b-a12b | 0.804    | 0.353      | 0.487        | 0.389       | 0.552      | 0.121        | 0.997    | 1.000    | 3.000        |
| qwen3.5-397b-a17b          | 0.790    | 0.368      | 0.502        | 0.395       | 0.490      | 0.131        | 0.999    | 1.000    | 3.000        |
| cosine baseline            | 0.763    | 0.357      | 0.459        | nan         | nan        | nan          | 0.000    | 1.000    | nan          |

`f2_star` is an **oracle** on both sides (91 thresholds swept on the rows scored). `f2_own` is the model's own verdict with nothing tuned — the cosine has no equivalent, which is the point: a ranker cannot decide, only order. `screened_own` is the fraction of the corpus a reviewer would read at that verdict.

`mean_auc` is dominated by `brouwer_2019`, which is 60% of set A and which the cosine already scores 0.99 on. Read the win/loss table above it, not the mean.

### Where the advantage lives

The pilot's rule was **diversity only pays where the dissenting branch is competent** — the LLM helped where it was strong, not where it was different. The same question here is whether the LLM covers the cosine's weak collections or merely re-wins its strong ones.

| use_case          | cosine_auc | gemma-4-31b-it | gpt-oss-20b | nemotron-3-super-120b-a12b | qwen3.5-397b-a17b |
|-------------------|------------|----------------|-------------|----------------------------|-------------------|
| brouwer_2019      | 0.993      | -0.055         | -0.077      | -0.001                     | -0.009            |
| leenaars_2020     | 0.862      | -0.104         | -0.065      | 0.016                      | -0.109            |
| moran_2021        | 0.441      | -0.033         | -0.028      | 0.010                      | -0.019            |
| muthu_2021        | 0.750      | -0.140         | -0.021      | -0.052                     | -0.023            |
| nelson_2002       | 0.714      | 0.047          | 0.072       | 0.083                      | 0.060             |
| sep_2021          | 0.613      | 0.294          | 0.273       | 0.227                      | 0.286             |
| van_der_valk_2021 | 0.853      | -0.005         | -0.045      | -0.012                     | -0.025            |
| van_dis_2020      | 0.881      | 0.078          | 0.050       | 0.052                      | 0.057             |

Correlation between the cosine's own AUC and the LLM's gain over it: **gemma-4-31b-it** -0.29 · **gpt-oss-20b** -0.41 · **nemotron-3-super-120b-a12b** -0.33 · **qwen3.5-397b-a17b** -0.34. A strong negative means the LLM picks up exactly where the cheap ranker gives out — which is the complementarity a second rung would need.

## 3. Does "when uncertain, include" survive the prevalence drop?

The pilot's most transferable finding was that stating the F2 asymmetry moved F2@own by +0.31 to +0.37 — at 26–77% prevalence, where including a doubtful paper is nearly free. At 2.19% each marginal inclusion costs ~45 false positives per true one. This is that instruction re-priced.

| model          | f2_own_P1 | f2_own_P2 | recall_P1 | recall_P2 | screened_P1 | screened_P2 | auc_P1 | auc_P2 | d_f2_own | d_auc |
|----------------|-----------|-----------|-----------|-----------|-------------|-------------|--------|--------|----------|-------|
| gemma-4-31b-it | 0.333     | 0.293     | 0.388     | 0.486     | 0.069       | 0.148       | 0.759  | 0.774  | -0.040   | 0.015 |
| gpt-oss-20b    | 0.382     | 0.402     | 0.434     | 0.490     | 0.086       | 0.103       | 0.772  | 0.783  | 0.020    | 0.011 |

## 4. The brief-format ladder

Same prompt (P2), four brief sets. Scored on **test + validate only**, because B2 was induced from train labels and every arm must sit on identical unseen rows. B0 strips the LLM-written term lists back to the review's own title and abstract — the format `CONTEXT.md` §4 says made the shuffled-brief control fail on old SYNERGY.

| model                      | arm         | mean_auc | mean_f2_own | recall_own | screened_own | mean_wss95 | coverage |
|----------------------------|-------------|----------|-------------|------------|--------------|------------|----------|
| gemma-4-31b-it             | B0 raw      | 0.777    | 0.349       | 0.559      | 0.147        | 0.250      | 1.000    |
| gemma-4-31b-it             | B1 supplied | 0.777    | 0.285       | 0.484      | 0.148        | 0.331      | 1.000    |
| gemma-4-31b-it             | B2 induced  | 0.834    | 0.488       | 0.674      | 0.169        | 0.338      | 1.000    |
| gemma-4-31b-it             | B✗ shuffled | 0.498    | 0.000       | 0.000      | 0.000        | -0.016     | 1.000    |
| gpt-oss-20b                | B1 supplied | 0.802    | 0.400       | 0.510      | 0.099        | 0.254      | 1.000    |
| nemotron-3-super-120b-a12b | B1 supplied | 0.815    | 0.404       | 0.574      | 0.117        | 0.312      | 1.000    |
| qwen3.5-397b-a17b          | B1 supplied | 0.793    | 0.398       | 0.508      | 0.124        | 0.334      | 1.000    |

#### `gemma-4-31b-it` per collection

**ROC-AUC**

| use_case          | cosine | B0 raw | B1 supplied | B2 induced | B✗ shuffled |
|-------------------|--------|--------|-------------|------------|-------------|
| brouwer_2019      | 0.993  | 0.998  | 0.970       | 0.998      | 0.494       |
| van_dis_2020      | 0.881  | 0.989  | 0.982       | 0.983      | 0.500       |
| leenaars_2020     | 0.862  | 0.782  | 0.766       | 0.888      | 0.500       |
| van_der_valk_2021 | 0.853  | 0.767  | 0.853       | 0.866      | 0.498       |
| muthu_2021        | 0.750  | 0.642  | 0.604       | 0.749      | 0.495       |
| nelson_2002       | 0.714  | 0.675  | 0.705       | 0.797      | 0.500       |
| sep_2021          | 0.613  | 0.877  | 0.884       | 0.929      | 0.500       |
| moran_2021        | 0.441  | 0.488  | 0.453       | 0.462      | 0.500       |

**recall at own verdict**

| use_case          | B0 raw | B1 supplied | B2 induced | B✗ shuffled |
|-------------------|--------|-------------|------------|-------------|
| brouwer_2019      | 1.000  | 0.880       | 0.880      | 0.000       |
| van_dis_2020      | 0.966  | 0.966       | 0.966      | 0.000       |
| leenaars_2020     | 0.083  | 0.043       | 0.713      | 0.000       |
| van_der_valk_2021 | 0.571  | 0.571       | 0.600      | 0.000       |
| muthu_2021        | 0.351  | 0.022       | 0.485      | 0.000       |
| nelson_2002       | 0.406  | 0.375       | 0.719      | 0.000       |
| sep_2021          | 0.938  | 0.875       | 0.938      | 0.000       |
| moran_2021        | 0.159  | 0.136       | 0.091      | 0.000       |

**fraction of corpus read**

| use_case          | B0 raw | B1 supplied | B2 induced | B✗ shuffled |
|-------------------|--------|-------------|------------|-------------|
| brouwer_2019      | 0.023  | 0.015       | 0.010      | 0.000       |
| van_dis_2020      | 0.084  | 0.104       | 0.082      | 0.000       |
| leenaars_2020     | 0.013  | 0.010       | 0.127      | 0.000       |
| van_der_valk_2021 | 0.187  | 0.197       | 0.183      | 0.000       |
| muthu_2021        | 0.147  | 0.007       | 0.189      | 0.000       |
| nelson_2002       | 0.210  | 0.259       | 0.336      | 0.000       |
| sep_2021          | 0.352  | 0.426       | 0.296      | 0.000       |
| moran_2021        | 0.157  | 0.163       | 0.128      | 0.000       |

- **B0 raw** vs cosine: 3 wins, 4 losses, 1 inside the floor
- **B1 supplied** vs cosine: 2 wins, 2 losses, 4 inside the floor
- **B2 induced** vs cosine: 3 wins, 0 losses, 5 inside the floor

**Falsification check.** The shuffled arm must collapse: mean AUC **0.498** (0.5 = no ranking), predicted-positive rate **0.000**. If that number is not near chance, every other result on this page is measuring generic paper quality and should be discarded.

## 5. Pre-registered bar

| condition                                                                                 | result                                                   | verdict  |
|-------------------------------------------------------------------------------------------|----------------------------------------------------------|----------|
| Zero-shot replaces the cold-start rung: beat cosine 0.763 by >0.03 on ≥6 of 8 collections | best is nemotron-3-super-120b-a12b at 0.804, winning 3/8 | FAIL     |
| Stop early if best mean AUC < 0.70                                                        | 0.804                                                    | continue |
| Induced brief (B2, fitted on train) replaces the rung — `gemma-4-31b-it`                  | mean AUC 0.834, 3 wins / 0 losses of 8                   | FAIL     |

