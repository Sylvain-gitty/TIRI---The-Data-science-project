# Set A — cold-start baselines, and the case-control sampling gate

`benchset_v1_large_set_a`: **62,229 rows / 8 collections / 1,362 positive (0.0219)**. All SYNERGY; briefs carry populated `terms_*` lists, which old SYNERGY did not.

## The incumbent bar — cold-start cosine-to-brief, all 62,229 rows

Ranking metrics from `cos_brief_qwen4b`. `floor_f2` is F2 for marking *everything* relevant — the number every F2 below has to be read against.

| use_case          | n     | prev  | floor_f2 | auc_jasper | auc_qwen4b | auc_bm25_must | wss95 | recall@10% | f2_star |
|-------------------|-------|-------|----------|------------|------------|---------------|-------|------------|---------|
| brouwer_2019      | 37401 | 0.002 | 0.008    | 0.994      | 0.993      | 0.969         | 0.916 | 0.984      | 0.558   |
| van_dis_2020      | 8938  | 0.008 | 0.039    | 0.865      | 0.895      | 0.840         | 0.551 | 0.611      | 0.227   |
| leenaars_2020     | 6711  | 0.086 | 0.319    | 0.831      | 0.861      | 0.674         | 0.470 | 0.406      | 0.562   |
| moran_2021        | 5154  | 0.021 | 0.099    | 0.434      | 0.445      | 0.539         | 0.007 | 0.081      | 0.101   |
| muthu_2021        | 2687  | 0.124 | 0.415    | 0.770      | 0.753      | 0.718         | 0.211 | 0.242      | 0.531   |
| van_der_valk_2021 | 710   | 0.122 | 0.411    | 0.845      | 0.853      | 0.735         | 0.418 | 0.402      | 0.616   |
| nelson_2002       | 358   | 0.224 | 0.590    | 0.727      | 0.714      | 0.581         | 0.263 | 0.188      | 0.675   |
| sep_2021          | 270   | 0.148 | 0.465    | 0.587      | 0.613      | 0.548         | 0.054 | 0.225      | 0.477   |

**Mean AUC 0.766** (jasper 0.757, bm25_must 0.701) · **mean WSS@95 0.361**. Pooled floor F2 **0.101** at 2.19% prevalence, against 0.872 on TIRI's pools — this is the surface where F2 means something.

Two collections to keep in view. **`brouwer_2019` is 60% of set A by volume and trivially easy** (AUC 0.994, 62 positives in 37,401 rows), so it will dominate any pooled number and flatter anything that ranks at all. **`moran_2021` is below chance** (0.445) — every embedding and the lex block agree, so the brief points away from the labels there. It is kept: it is the one collection where reading the abstract could beat a cosine outright, and the one that could expose a corpus defect.

## The gate — 9,993-row case-control sample vs all 62,229 rows

Same metric, same ranker, three seeds. `_samp` columns are weighted by `w = negatives_in_population / negatives_sampled`; `_full` scores every row. `floor_samp` recovering `floor_full` is the direct check that the weights restore population prevalence.

| use_case          | auc_full | auc_samp | auc_sd | d_auc  | wss_full | wss_samp | wss_sd | d_wss  | f2_full | f2_samp | d_f2   | floor_full | floor_samp |
|-------------------|----------|----------|--------|--------|----------|----------|--------|--------|---------|---------|--------|------------|------------|
| brouwer_2019      | 0.993    | 0.993    | 0.000  | 0.000  | 0.916    | 0.911    | 0.004  | -0.005 | 0.558   | 0.541   | -0.017 | 0.008      | 0.008      |
| van_dis_2020      | 0.895    | 0.891    | 0.011  | -0.004 | 0.551    | 0.545    | 0.020  | -0.006 | 0.227   | 0.220   | -0.007 | 0.039      | 0.039      |
| leenaars_2020     | 0.861    | 0.861    | 0.003  | 0.000  | 0.470    | 0.469    | 0.004  | -0.001 | 0.562   | 0.558   | -0.004 | 0.319      | 0.319      |
| moran_2021        | 0.445    | 0.445    | 0.004  | 0.000  | 0.007    | 0.004    | 0.006  | -0.003 | 0.101   | 0.101   | 0.000  | 0.099      | 0.099      |
| muthu_2021        | 0.753    | 0.756    | 0.006  | 0.003  | 0.211    | 0.209    | 0.006  | -0.002 | 0.531   | 0.531   | 0.000  | 0.415      | 0.415      |
| van_der_valk_2021 | 0.853    | 0.853    | 0.000  | 0.000  | 0.418    | 0.418    | 0.000  | 0.000  | 0.616   | 0.616   | 0.000  | 0.411      | 0.411      |
| nelson_2002       | 0.714    | 0.714    | 0.000  | 0.000  | 0.263    | 0.263    | 0.000  | 0.000  | 0.675   | 0.675   | 0.000  | 0.590      | 0.590      |
| sep_2021          | 0.613    | 0.613    | 0.000  | 0.000  | 0.054    | 0.054    | 0.000  | 0.000  | 0.477   | 0.477   | 0.000  | 0.465      | 0.465      |

Largest absolute discrepancy: AUC **0.004**, WSS@95 **0.006**, F2@t\* **0.017**. Prevalence recovery is exact to **0.0000**. Seed-to-seed sd on AUC is **0.003**, against the repo's ~0.010 noise floor.

**Gate: PASS.** The sample reproduces the corpus inside the noise floor; LLM spend is authorised against it.

