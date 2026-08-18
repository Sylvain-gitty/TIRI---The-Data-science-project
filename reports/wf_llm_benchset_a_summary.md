# Set A — every method, every metric, one table

8 SYNERGY collections, 62,229 papers, **2.19% relevant**. All rows scored on **test + validate** (3,997 of the 9,993-row case-control sample) and **weighted back to population prevalence** — precision, F2, WSS@95 and recall@10% are wrong by 5-25x without that weight. Every figure is the **mean over the 8 collections**, not a pooled number: `brouwer_2019` is 60% of set A and trivially easy, so a pool is mostly that one collection (`CONTEXT.md` §5).

**Marking every paper relevant scores F2 = 0.101.** That is the floor every F2 below sits on.

## 1. Ranking quality and the two threshold-free metrics

`wss95` = fraction of the corpus a reviewer can skip while still finding 95% of relevant papers, minus the 5% chance would give. `recall@10%` = share of relevant papers found in the top tenth of the ranking.

| labels  | method                                    | auc   | wss95  | recall@10% |
|---------|-------------------------------------------|-------|--------|------------|
| in-silo | Ensemble blend, induced brief + cosine    | 0.893 | 0.494  | 0.606      |
| in-silo | Ensemble blend, induced brief             | 0.892 | 0.506  | 0.615      |
| in-silo | Ensemble blend, supplied brief            | 0.891 | 0.491  | 0.600      |
| 60      | LogReg on Qwen3-4B embedding              | 0.844 | 0.417  | 0.489      |
| 60      | LLM gemma-4-31b-it P2 · B2 induced brief  | 0.834 | 0.338  | 0.544      |
| 0       | LLM nemotron-3-super-120b-a12b P2         | 0.815 | 0.312  | 0.512      |
| 0       | LLM gpt-oss-20b P2                        | 0.802 | 0.254  | 0.493      |
| 0       | LLM qwen3.5-397b-a17b P2                  | 0.793 | 0.334  | 0.449      |
| 0       | LLM gpt-oss-20b P1                        | 0.781 | 0.197  | 0.477      |
| 60      | LogReg on query-conditioned block         | 0.777 | 0.382  | 0.408      |
| 0       | LLM gemma-4-31b-it P2                     | 0.777 | 0.331  | 0.459      |
| 0       | LLM gemma-4-31b-it P2 · B0 raw brief      | 0.777 | 0.250  | 0.451      |
| 0       | cosine-to-brief (Qwen3-4B)                | 0.774 | 0.377  | 0.413      |
| 0       | cosine-to-brief (Jasper)                  | 0.768 | 0.343  | 0.379      |
| 60      | LogReg on lexical block, induced brief    | 0.763 | 0.334  | 0.394      |
| 0       | LLM gemma-4-31b-it P1                     | 0.762 | 0.225  | 0.423      |
| 60      | LogReg on lexical block, supplied brief   | 0.719 | 0.269  | 0.318      |
| 0       | BM25 must-terms, supplied brief           | 0.697 | 0.265  | 0.306      |
| 0       | LLM gemma-4-31b-it P2 · B✗ shuffled brief | 0.498 | -0.016 | 0.097      |

## 2. At a fixed, untuned threshold of 0.5

What you get with no calibration. `read` is the fraction of the corpus the reviewer must read. Cosine and BM25 rows are min-max rescaled to [0,1] first — a cosine has no natural 0.5, so read those two rows as indicative only.

| labels  | method                                    | f2@0.5 | prec@0.5 | rec@0.5 | read@0.5 |
|---------|-------------------------------------------|--------|----------|---------|----------|
| 0       | BM25 must-terms, supplied brief           | 0.175  | 0.192    | 0.278   | 0.106    |
| 0       | LLM gemma-4-31b-it P1                     | 0.339  | 0.277    | 0.410   | 0.070    |
| 0       | LLM gemma-4-31b-it P2                     | 0.300  | 0.252    | 0.470   | 0.112    |
| 0       | LLM gemma-4-31b-it P2 · B0 raw brief      | 0.345  | 0.324    | 0.530   | 0.123    |
| 0       | LLM gemma-4-31b-it P2 · B✗ shuffled brief | 0.000  | 0.000    | 0.000   | 0.000    |
| 0       | LLM gpt-oss-20b P1                        | 0.396  | 0.411    | 0.455   | 0.084    |
| 0       | LLM gpt-oss-20b P2                        | 0.401  | 0.357    | 0.509   | 0.098    |
| 0       | LLM nemotron-3-super-120b-a12b P2         | 0.402  | 0.338    | 0.577   | 0.118    |
| 0       | LLM qwen3.5-397b-a17b P2                  | 0.382  | 0.519    | 0.400   | 0.079    |
| 0       | cosine-to-brief (Jasper)                  | 0.326  | 0.134    | 0.802   | 0.531    |
| 0       | cosine-to-brief (Qwen3-4B)                | 0.338  | 0.142    | 0.782   | 0.452    |
| 60      | LLM gemma-4-31b-it P2 · B2 induced brief  | 0.503  | 0.328    | 0.665   | 0.151    |
| 60      | LogReg on Qwen3-4B embedding              | 0.413  | 0.201    | 0.790   | 0.303    |
| 60      | LogReg on lexical block, induced brief    | 0.334  | 0.167    | 0.679   | 0.320    |
| 60      | LogReg on lexical block, supplied brief   | 0.303  | 0.138    | 0.680   | 0.371    |
| 60      | LogReg on query-conditioned block         | 0.359  | 0.175    | 0.732   | 0.321    |
| in-silo | Ensemble blend, induced brief             | 0.508  | 0.395    | 0.599   | 0.112    |
| in-silo | Ensemble blend, induced brief + cosine    | 0.507  | 0.412    | 0.590   | 0.106    |
| in-silo | Ensemble blend, supplied brief            | 0.502  | 0.391    | 0.602   | 0.111    |

## 3. At the best of 91 swept thresholds (oracle — upper bound)

`t_star` is the mean per-collection optimum. This metric picks its threshold on the same rows it scores, so it flatters every method equally and is quoted for all of them or none. It is the rule that produced the ensemble's published 0.893.

| labels  | method                                    | t_star | f2@t* | prec@t* | rec@t* | read@t* |
|---------|-------------------------------------------|--------|-------|---------|--------|---------|
| 0       | BM25 must-terms, supplied brief           | 0.266  | 0.374 | 0.153   | 0.732  | 0.504   |
| 0       | LLM gemma-4-31b-it P1                     | 0.229  | 0.525 | 0.332   | 0.801  | 0.383   |
| 0       | LLM gemma-4-31b-it P2                     | 0.353  | 0.512 | 0.334   | 0.711  | 0.330   |
| 0       | LLM gemma-4-31b-it P2 · B0 raw brief      | 0.340  | 0.531 | 0.328   | 0.695  | 0.265   |
| 0       | LLM gemma-4-31b-it P2 · B✗ shuffled brief | 0.010  | 0.000 | 0.000   | 0.000  | 0.003   |
| 0       | LLM gpt-oss-20b P1                        | 0.248  | 0.520 | 0.312   | 0.679  | 0.220   |
| 0       | LLM gpt-oss-20b P2                        | 0.220  | 0.537 | 0.257   | 0.781  | 0.293   |
| 0       | LLM nemotron-3-super-120b-a12b P2         | 0.296  | 0.531 | 0.383   | 0.710  | 0.265   |
| 0       | LLM qwen3.5-397b-a17b P2                  | 0.204  | 0.533 | 0.334   | 0.812  | 0.420   |
| 0       | cosine-to-brief (Jasper)                  | 0.626  | 0.507 | 0.326   | 0.684  | 0.317   |
| 0       | cosine-to-brief (Qwen3-4B)                | 0.559  | 0.498 | 0.304   | 0.753  | 0.376   |
| 60      | LLM gemma-4-31b-it P2 · B2 induced brief  | 0.421  | 0.579 | 0.411   | 0.766  | 0.276   |
| 60      | LogReg on Qwen3-4B embedding              | 0.641  | 0.539 | 0.312   | 0.759  | 0.250   |
| 60      | LogReg on lexical block, induced brief    | 0.568  | 0.451 | 0.301   | 0.705  | 0.379   |
| 60      | LogReg on lexical block, supplied brief   | 0.570  | 0.396 | 0.154   | 0.708  | 0.426   |
| 60      | LogReg on query-conditioned block         | 0.591  | 0.473 | 0.314   | 0.757  | 0.384   |
| in-silo | Ensemble blend, induced brief             | 0.414  | 0.646 | 0.479   | 0.760  | 0.182   |
| in-silo | Ensemble blend, induced brief + cosine    | 0.406  | 0.649 | 0.470   | 0.776  | 0.187   |
| in-silo | Ensemble blend, supplied brief            | 0.388  | 0.647 | 0.444   | 0.789  | 0.197   |

## 4. At the model's own verdict — nothing fitted

The honest operating point, and **only prompted LLMs have one**. A ranker orders papers but cannot say where to stop; turning one into a decision costs labels that the LLM did not need. That is the asymmetry the F2 columns above hide.

| labels | method                                    | f2@own | prec@own | rec@own | read@own | auc   |
|--------|-------------------------------------------|--------|----------|---------|----------|-------|
| 0      | LLM gemma-4-31b-it P1                     | 0.361  | 0.360    | 0.410   | 0.067    | 0.762 |
| 0      | LLM gemma-4-31b-it P2                     | 0.285  | 0.243    | 0.484   | 0.148    | 0.777 |
| 0      | LLM gemma-4-31b-it P2 · B0 raw brief      | 0.349  | 0.278    | 0.559   | 0.147    | 0.777 |
| 0      | LLM gemma-4-31b-it P2 · B✗ shuffled brief | 0.000  | 0.000    | 0.000   | 0.000    | 0.498 |
| 0      | LLM gpt-oss-20b P1                        | 0.396  | 0.411    | 0.455   | 0.084    | 0.781 |
| 0      | LLM gpt-oss-20b P2                        | 0.400  | 0.357    | 0.510   | 0.099    | 0.802 |
| 0      | LLM nemotron-3-super-120b-a12b P2         | 0.404  | 0.338    | 0.574   | 0.117    | 0.815 |
| 0      | LLM qwen3.5-397b-a17b P2                  | 0.398  | 0.353    | 0.508   | 0.124    | 0.793 |
| 60     | LLM gemma-4-31b-it P2 · B2 induced brief  | 0.488  | 0.302    | 0.674   | 0.169    | 0.834 |

## 5. Headline comparison, best configuration per label budget

| labels  | method                                   | auc   | wss95 | recall@10% | f2@0.5 | prec@0.5 | rec@0.5 | read@0.5 | t_star | f2@t* | prec@t* | rec@t* | read@t* | f2@own | prec@own | rec@own | read@own |
|---------|------------------------------------------|-------|-------|------------|--------|----------|---------|----------|--------|-------|---------|--------|---------|--------|----------|---------|----------|
| 0       | cosine-to-brief (Qwen3-4B)               | 0.774 | 0.377 | 0.413      | 0.338  | 0.142    | 0.782   | 0.452    | 0.559  | 0.498 | 0.304   | 0.753  | 0.376   | nan    | nan      | nan     | nan      |
| 0       | LLM gemma-4-31b-it P2                    | 0.777 | 0.331 | 0.459      | 0.300  | 0.252    | 0.470   | 0.112    | 0.353  | 0.512 | 0.334   | 0.711  | 0.330   | 0.285  | 0.243    | 0.484   | 0.148    |
| 0       | LLM nemotron-3-super-120b-a12b P2        | 0.815 | 0.312 | 0.512      | 0.402  | 0.338    | 0.577   | 0.118    | 0.296  | 0.531 | 0.383   | 0.710  | 0.265   | 0.404  | 0.338    | 0.574   | 0.117    |
| 60      | LogReg on lexical block, induced brief   | 0.763 | 0.334 | 0.394      | 0.334  | 0.167    | 0.679   | 0.320    | 0.568  | 0.451 | 0.301   | 0.705  | 0.379   | nan    | nan      | nan     | nan      |
| 60      | LLM gemma-4-31b-it P2 · B2 induced brief | 0.834 | 0.338 | 0.544      | 0.503  | 0.328    | 0.665   | 0.151    | 0.421  | 0.579 | 0.411   | 0.766  | 0.276   | 0.488  | 0.302    | 0.674   | 0.169    |
| 60      | LogReg on Qwen3-4B embedding             | 0.844 | 0.417 | 0.489      | 0.413  | 0.201    | 0.790   | 0.303    | 0.641  | 0.539 | 0.312   | 0.759  | 0.250   | nan    | nan      | nan     | nan      |
| in-silo | Ensemble blend, supplied brief           | 0.891 | 0.491 | 0.600      | 0.502  | 0.391    | 0.602   | 0.111    | 0.388  | 0.647 | 0.444   | 0.789  | 0.197   | nan    | nan      | nan     | nan      |

## 6. Per collection — the two configurations that matter

**LLM gemma-4-31b-it P2 · B2 induced brief**

| use_case          | auc   | wss95  | f2@0.5 | f2@t* | t_star | prec@t* | rec@t* | floor_f2 |
|-------------------|-------|--------|--------|-------|--------|---------|--------|----------|
| brouwer_2019      | 0.998 | 0.938  | 0.496  | 0.690 | 0.910  | 1.000   | 0.640  | 0.008    |
| leenaars_2020     | 0.888 | 0.204  | 0.657  | 0.689 | 0.210  | 0.419   | 0.822  | 0.319    |
| moran_2021        | 0.462 | -0.011 | 0.053  | 0.078 | 0.010  | 0.018   | 0.455  | 0.098    |
| muthu_2021        | 0.749 | 0.080  | 0.416  | 0.515 | 0.010  | 0.205   | 0.828  | 0.416    |
| nelson_2002       | 0.797 | 0.033  | 0.657  | 0.667 | 0.860  | 0.595   | 0.688  | 0.590    |
| sep_2021          | 0.929 | 0.080  | 0.824  | 0.824 | 0.410  | 0.556   | 0.938  | 0.465    |
| van_der_valk_2021 | 0.866 | 0.497  | 0.535  | 0.639 | 0.010  | 0.261   | 1.000  | 0.413    |
| van_dis_2020      | 0.983 | 0.888  | 0.386  | 0.526 | 0.950  | 0.237   | 0.759  | 0.039    |

**Ensemble blend, supplied brief**

| use_case          | auc   | wss95  | f2@0.5 | f2@t* | t_star | prec@t* | rec@t* | floor_f2 |
|-------------------|-------|--------|--------|-------|--------|---------|--------|----------|
| brouwer_2019      | 1.000 | 0.947  | 0.557  | 0.902 | 0.800  | 1.000   | 0.880  | 0.008    |
| leenaars_2020     | 0.952 | 0.673  | 0.723  | 0.735 | 0.570  | 0.491   | 0.839  | 0.319    |
| moran_2021        | 0.875 | 0.284  | 0.133  | 0.406 | 0.280  | 0.140   | 0.773  | 0.098    |
| muthu_2021        | 0.841 | 0.371  | 0.526  | 0.598 | 0.320  | 0.346   | 0.731  | 0.416    |
| nelson_2002       | 0.872 | 0.419  | 0.667  | 0.767 | 0.180  | 0.475   | 0.906  | 0.590    |
| sep_2021          | 0.761 | -0.004 | 0.533  | 0.640 | 0.190  | 0.500   | 0.688  | 0.465    |
| van_der_valk_2021 | 0.874 | 0.503  | 0.480  | 0.647 | 0.080  | 0.287   | 0.943  | 0.413    |
| van_dis_2020      | 0.952 | 0.737  | 0.394  | 0.478 | 0.680  | 0.311   | 0.552  | 0.039    |

## 7. Reading notes

- **`floor_f2` is per collection and ranges 0.008-0.590**, because prevalence does (0.2%-22%). A collection-level F2 near its own floor is measuring prevalence, not skill. The pooled floor is 0.101.
- **`f2@t*` is an oracle** and always beats `f2@0.5` and `f2@own` by construction. The gap between `f2@t*` and `f2@own` is what a model gives up by having to decide without seeing the answers.
- **Precision is low everywhere** — at 2.19% prevalence, a screen tuned for recall necessarily returns mostly irrelevant papers. That is the correct behaviour when a miss costs 5x a false positive; `read@` is the column that says what it costs.
- **The shuffled-brief row must collapse to ~0.5 AUC and 0 recall.** It does. If it ever does not, nothing else on this page is measuring brief-conditioned relevance.

