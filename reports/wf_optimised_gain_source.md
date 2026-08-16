# Where does the optimised-spec gain come from?

Status: **measured.** Follow-up to [`wf_optimised_usecase_baseline.md`](wf_optimised_usecase_baseline.md), which established that enriching a use-case objective is worth **+0.014** mean zero-label ROC-AUC and clears the 0.03 floor on 2 of 6 use cases — but not *which* 2, or why.

🟢 clears the 0.03 noise floor · 🟡 real but under it · ⚪ engineering finding

## 0. The hypothesis, and how it can be tested without fooling yourself

> *"It improves the least well-defined use cases and has less effect on the rest."*

That predicts two separable things:

**(a) the six use cases end up closer together** — the weak ones come up, the strong ones stay put, so the spread narrows.
**(b) the gain lands on the badly-written specs** — measured on something independent of the outcome.

🔴 **The tempting version of (b) is circular and this report keeps it visible so it can be dismissed on the record.** "Which use case scored worst before?" is not a measure of the spec — it is the outcome. Ranking gain against it is regression to the mean by construction: *any* noisy quantity gains most where it started lowest, whether or not briefs matter at all. So five of the six measures below are properties of the **spec text alone**, and never see a paper or a label.

⚠️ **n = 6, and nothing here licenses a correlation claim.** Spearman rho is printed because ranking six things two ways and asking whether the orders agree is a fair description of six data points. It is not evidence about a population. Read the ranks.

⚠️ Everything below is the **zero-label** regime, which is the only one where the answer is non-zero — the fitted baseline cannot separate the arms at all (`wf_optimised_usecase_baseline.md` §4).

## 1. (a) Does the spread narrow?

`sd` and `range` are taken across the six use cases' zero-label ROC-AUC, once per encoder. If the rewrite lifts the laggards, both shrink.

| encoder | sd_before | sd_after | d_sd   | range_before | range_after | d_range | worst_before | worst_after | d_worst | d_mean |
|---------|-----------|----------|--------|--------------|-------------|---------|--------------|-------------|---------|--------|
| jasper  | 0.113     | 0.109    | -0.004 | 0.299        | 0.270       | -0.030  | 0.529        | 0.549       | 0.020   | 0.005  |
| qwen4b  | 0.126     | 0.130    | 0.004  | 0.315        | 0.326       | 0.011   | 0.519        | 0.511       | -0.008  | 0.010  |
| qwen8b  | 0.127     | 0.135    | 0.008  | 0.326        | 0.338       | 0.012   | 0.517        | 0.512       | -0.005  | 0.026  |

*ELI18: `sd` is how far apart the six use cases' scores are. A negative `d_sd` means they converged; a positive one means the rewrite pushed them further apart. `d_worst` is what happened to the worst-performing use case specifically — the one the hypothesis says should benefit most.*

## 2. (b) Does the gain land on the worst-defined specs?

| use_case         | baseline_auc | gain   | encoders_up | linter_findings | checkability | objective_words | n_terms | reservoir_chars | dose_chars |
|------------------|--------------|--------|-------------|-----------------|--------------|-----------------|---------|-----------------|------------|
| carbon_capture   | 0.759        | 0.045  | 3/3         | 1               | 0.647        | 17              | 8       | 601             | 292        |
| cement_binders   | 0.835        | -0.000 | 2/3         | 1               | 1.000        | 11              | 6       | 322             | 212        |
| ner              | 0.749        | 0.011  | 2/3         | 1               | 0.788        | 25              | 10      | 1120            | 786        |
| soil_microbiome  | 0.550        | -0.006 | 1/3         | 0               | 0.735        | 23              | 15      | 774             | 701        |
| solar_leo        | 0.546        | 0.032  | 3/3         | 2               | 0.500        | 53              | 3       | 1265            | 861        |
| tech_forecasting | 0.647        | 0.001  | 1/3         | 0               | 0.786        | 21              | 12      | 769             | 564        |

**Rank agreement.** Each measure orders the six use cases worst-spec-first; `rho` says how well that order matches the gain order. **Positive rho supports the hypothesis.**

| measure         | 'worse' means | worst-defined spec | 2nd worst       | best-defined spec | rho vs gain |
|-----------------|---------------|--------------------|-----------------|-------------------|-------------|
| baseline_auc    | lower value   | solar_leo          | soil_microbiome | cement_binders    | -0.030      |
| linter_findings | higher value  | solar_leo          | carbon_capture  | tech_forecasting  | 0.650       |
| checkability    | lower value   | solar_leo          | carbon_capture  | cement_binders    | 0.540       |
| objective_words | lower value   | cement_binders     | carbon_capture  | solar_leo         | -0.200      |
| n_terms         | lower value   | solar_leo          | cement_binders  | soil_microbiome   | 0.540       |
| reservoir_chars | higher value  | solar_leo          | ner             | cement_binders    | 0.200       |
| dose_chars      | higher value  | solar_leo          | ner             | cement_binders    | 0.200       |

For the record, the two numbers the circular version turns on: the use case with the **lowest** baseline gained **+0.032**, and the one with the **highest** gained **-0.000**. The largest gain of all went to **`carbon_capture`**.

### 2a. ⚠️ How much of that rests on one use case

`solar_leo` ranks worst-defined on five of the seven measures *and* is a large gainer, so it could be carrying every positive rho on its own. Dropping each use case in turn:

| measure         | all 6  | min (drop 1) | max (drop 1) | worst when dropping |
|-----------------|--------|--------------|--------------|---------------------|
| baseline_auc    | -0.030 | -0.400       | 0.300        | cement_binders      |
| linter_findings | 0.650  | 0.450        | 0.790        | soil_microbiome     |
| checkability    | 0.540  | 0.300        | 0.800        | carbon_capture      |
| objective_words | -0.200 | -0.700       | 0.100        | carbon_capture      |
| n_terms         | 0.540  | 0.200        | 0.900        | soil_microbiome     |
| reservoir_chars | 0.200  | -0.100       | 0.700        | cement_binders      |
| dose_chars      | 0.200  | -0.100       | 0.700        | cement_binders      |

🔴 **The limit of this test, stated because it has bitten this repo before.** `wf_foreign_brief_validity_setb.md` records it: leave-one-out shows whether a single *point* carries a correlation. It cannot show whether the whole *surface* does. Six analyst-written specs from one team is a surface, and it took a second surface to kill the foreign-brief margin after leave-one-out had passed it. §5 says what the second surface would be here.

## 3. (c) Which papers moved, and where in the list

Percentile rank is zero-sum inside a use case, so "relevant papers rose" and "irrelevant papers fell" are one fact stated twice. `d_recall_at_10pct` is the one that matters operationally: ROC-AUC counts a relevant paper climbing from rank 900 to 600 exactly as much as one climbing from 50 to 20, and only the second changes what an analyst sees.

| use_case         | d_auc  | pos_rank_shift | neg_rank_shift | share_pos_up | d_recall_at_10pct |
|------------------|--------|----------------|----------------|--------------|-------------------|
| carbon_capture   | 0.045  | 0.023          | -0.022         | 0.537        | 0.000             |
| solar_leo        | 0.032  | 0.007          | -0.025         | 0.458        | 0.001             |
| ner              | 0.011  | 0.003          | -0.008         | 0.478        | -0.000            |
| tech_forecasting | 0.001  | 0.000          | -0.001         | 0.480        | -0.006            |
| cement_binders   | -0.000 | -0.000         | 0.000          | 0.471        | 0.004             |
| soil_microbiome  | -0.006 | -0.004         | 0.001          | 0.440        | -0.014            |

*ELI18: `pos_rank_shift` +0.02 means the average relevant paper moved up 2 percentiles — out of 100 papers, past two of them. `share_pos_up` is the fraction of relevant papers that moved up at all; 0.500 would mean the rewrite shuffled them without helping.*

### 3a. The depth profile — the finding that outranks the rest

ROC-AUC is the area under the whole ranked list, so an AUC gain says nothing about **where** it happened. Only the shallow end is a screening workflow: an analyst reads the top of the list and stops. Change in recall, by how far down the list you read:

| use_case         | 5%     | 10%    | 20%    | 30%    | 50%    | 75%    |
|------------------|--------|--------|--------|--------|--------|--------|
| carbon_capture   | 0.009  | 0.000  | 0.005  | 0.014  | 0.046  | 0.041  |
| cement_binders   | 0.000  | 0.004  | 0.004  | 0.000  | 0.000  | -0.004 |
| ner              | 0.000  | 0.000  | -0.004 | 0.001  | 0.006  | 0.001  |
| soil_microbiome  | -0.004 | -0.014 | -0.014 | -0.011 | -0.007 | -0.014 |
| solar_leo        | 0.000  | 0.001  | 0.004  | 0.010  | 0.012  | 0.013  |
| tech_forecasting | -0.006 | -0.006 | 0.000  | -0.006 | 0.004  | 0.002  |
| ALL SIX (mean)   | -0.000 | -0.003 | -0.001 | 0.001  | 0.010  | 0.007  |

*ELI18: each cell is "how many more of the relevant papers you would have found, as a share of all of them, if you read that far down the list with the new brief instead of the old one". 0.000 means the rewrite made no difference to what you would have found by then.*
