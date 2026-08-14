# P-FB — the foreign-brief detector: is "a stranger's brief screens your corpus better than yours" a real signal?

**Status:** all three arms run. Verification gate PASSED (see `## 1`). Cost: $0, no LLM calls, no GPU, no network.

Confidence key: 🟢 clears the ~0.03 noise floor / decisively measured · 🟡 real but under the floor · ⚪ engineering finding.

## 0. How to read any number here

- **Surface accounting.** The cosine and lexical arms run across all **34** use cases (6 TIRI live + 28 `benchset_v1`) by construction — they consume no confirmatory surface (`wf_spec_quality_plan.md` §0). The **validation** step is different: both reference CSVs (`wf_label_budget_shape_grid.csv`, `wf_llm_benchset_a_induced_cosine.csv`) cover only the 8 collections in the burned `benchset_v1_large_set_a` surface, and `sep_2021` has just 24 train positives and drops out of the grid's `n_pos=30` slice (`run_label_budget_shape.py`'s own comment says so) — so validation (a) is **n=7**, not 8, and validation (b) is **n=8**. Neither is 34. **This is not independent validation** — both CSVs are derived from set A, which this margin is also computed over, so a correlation here says "these two set-A-derived numbers agree," never "the margin was confirmed on unseen data."

- **`auc_floor` (grid CSV) is a different instrument from this script's own-brief diagonal**, not the same number under a different name: `auc_floor` is `roc_auc_score(y, cos_brief_qwen4b)` on **weighted set-A held-out rows**; this script's diagonal is jasper (or qwen4b) cosine-to-brief over the **whole collection**. On `synergy_moran_2021` they are 0.488 vs 0.436 — both are reported below, side by side, never averaged or substituted for each other.

- **Provenance is stratified, never pooled.** TIRI's 6 briefs are analyst-written; the 28 benchset briefs are derived from each review's own abstract (`brief_provenance=review_abstract`), which inflates every brief-reading score (`DATA_BRIEF.md` honest-limit #2). `## 7` reports two means with two `n`s.

- **`roadfreight_metareview`** sits at 78% prevalence over 132 rows and is quarantined in `benchset_v1_split_manifest.json` ("the screening decision is 'is this a review?', not 'is this relevant?'"). It stays in every descriptive table below but is excluded from headline flag counts, which are stated out of **33**, footnoted every time.

- **Cosine-space specificity is post-hoc, not a prediction.** 7 of 34 cosine margins are already printed in `wf_usecase_diversity.ipynb` §6, so no threshold chosen now is blind — it is descriptive, cited to that notebook. The lexical arm is the only genuinely pre-registered blind test in this probe.

- **Label which critic is speaking.** Every AUC and rho below is a hard, cross-validated-or-population metric; every PASS/FAIL verdict quotes the pre-registered bar it is measured against; anything else ("this looks like a real effect") is flagged as a judgement call explicitly.

## 1. 🟡 Cosine arm — descriptive, already measured (verification gate: PASS)

Re-running `usecase_diversity_utils.brief_transfer_auc` for jasper and qwen4b and persisting both 34x34 matrices (previously computed on every notebook run and saved nowhere) reproduced `wf_usecase_diversity.ipynb` §6 exactly: `soil_microbiome` own **0.603**, best foreign **0.792** (by `solar_leo`), margin **-0.190**; mean own **0.820**. See the script's `verify_cosine_gate` for the full 8-check list; all passed, so the population and matrix orientation are confirmed correct before anything below is trusted.

💡 **ELI18.** `soil_microbiome`'s own brief ranks its own papers at 0.603 ROC-AUC, where 0.500 is a coin flip and 1.000 is a perfect ranking; a completely unrelated use case's brief (`solar_leo`) ranks the same papers at 0.792 — markedly *better*. If briefs only ever helped rank their own corpus, this number would be zero or positive for every use case; it is negative for 7. If the finding holds up (it does, see `## 3`), an analyst can be shown "3 other use cases screen your papers better than your own brief" with zero labels spent.

7 of 34 use cases have `own < best_foreign` under jasper (the count `wf_spec_quality_plan.md`'s Amendment already knew, reproduced here rather than assumed): soil_microbiome, solar_leo, synergy_jeyaraman_2020, synergy_menon_2022, synergy_moran_2021, synergy_sep_2021, synergy_wassenaar_2017.

| use_case_key                    | surface  | own   | best_foreign | by                              | margin |
|---------------------------------|----------|-------|--------------|---------------------------------|--------|
| soil_microbiome                 | live     | 0.603 | 0.792        | solar_leo                       | -0.190 |
| synergy_moran_2021              | benchset | 0.436 | 0.616        | synergy_menon_2022              | -0.180 |
| synergy_menon_2022              | benchset | 0.819 | 0.893        | synergy_wassenaar_2017          | -0.075 |
| solar_leo                       | live     | 0.529 | 0.588        | cement_binders                  | -0.059 |
| synergy_jeyaraman_2020          | benchset | 0.739 | 0.795        | synergy_muthu_2021              | -0.056 |
| synergy_sep_2021                | benchset | 0.588 | 0.617        | synergy_menon_2022              | -0.029 |
| synergy_wassenaar_2017          | benchset | 0.936 | 0.956        | synergy_van_der_valk_2021       | -0.020 |
| synergy_hall_2012               | benchset | 0.991 | 0.990        | synergy_radjenovic_2013         | 0.002  |
| synergy_van_dis_2020            | benchset | 0.866 | 0.846        | synergy_oud_2018                | 0.020  |
| synergy_chou_2003               | benchset | 0.888 | 0.866        | synergy_chou_2004               | 0.022  |
| synergy_wolters_2018            | benchset | 0.975 | 0.954        | synergy_bos_2018                | 0.022  |
| synergy_walker_2018             | benchset | 0.900 | 0.878        | synergy_wassenaar_2017          | 0.022  |
| synergy_bos_2018                | benchset | 0.986 | 0.962        | synergy_wolters_2018            | 0.024  |
| synergy_nelson_2002             | benchset | 0.727 | 0.702        | synergy_jeyaraman_2020          | 0.024  |
| synergy_radjenovic_2013         | benchset | 0.974 | 0.943        | synergy_hall_2012               | 0.031  |
| synergy_chou_2004               | benchset | 0.707 | 0.674        | synergy_chou_2003               | 0.033  |
| tech_forecasting                | live     | 0.672 | 0.630        | ner                             | 0.042  |
| synergy_oud_2018                | benchset | 0.863 | 0.818        | synergy_van_dis_2020            | 0.045  |
| synergy_leenaars_2020           | benchset | 0.833 | 0.786        | synergy_jeyaraman_2020          | 0.047  |
| synergy_muthu_2021              | benchset | 0.770 | 0.706        | synergy_jeyaraman_2020          | 0.064  |
| synergy_smid_2020               | benchset | 0.947 | 0.879        | synergy_van_de_schoot_2018      | 0.067  |
| roadfreight_metareview          | benchset | 0.698 | 0.629        | synergy_appenzeller-herzog_2019 | 0.068  |
| synergy_donners_2021            | benchset | 0.828 | 0.733        | synergy_van_der_valk_2021       | 0.095  |
| synergy_van_de_schoot_2018      | benchset | 0.858 | 0.755        | synergy_brouwer_2019            | 0.103  |
| synergy_brouwer_2019            | benchset | 0.994 | 0.890        | synergy_meijboom_2021           | 0.104  |
| carbon_capture                  | live     | 0.751 | 0.645        | synergy_jeyaraman_2020          | 0.107  |
| synergy_appenzeller-herzog_2019 | benchset | 0.876 | 0.748        | synergy_van_der_waal_2022       | 0.128  |
| synergy_van_der_valk_2021       | benchset | 0.846 | 0.714        | synergy_wolters_2018            | 0.132  |
| synergy_leenaars_2019           | benchset | 0.990 | 0.854        | synergy_van_der_valk_2021       | 0.137  |
| synergy_meijboom_2021           | benchset | 0.887 | 0.747        | synergy_van_dis_2020            | 0.140  |
| nykvist_evcharging              | benchset | 0.836 | 0.685        | synergy_van_de_schoot_2018      | 0.150  |
| cement_binders                  | live     | 0.828 | 0.673        | carbon_capture                  | 0.156  |
| synergy_van_der_waal_2022       | benchset | 0.953 | 0.732        | synergy_brouwer_2019            | 0.220  |
| ner                             | live     | 0.774 | 0.550        | nykvist_evcharging              | 0.223  |

⚠️ **`solar_leo` caveat.** `solar_leo` is `soil_microbiome`'s best foreign brief, and `solar_leo` is itself a known corpus defect — `CONTEXT.md` §4 records that its labels track publication year rather than genuine relevance. This licenses "`soil_microbiome`'s own brief loses even to an unrelated, defective ranking signal on its own corpus" (the point stands regardless of *why* `solar_leo` ranks it well) — it does **not** license "`solar_leo`'s brief is a good screener in general." `solar_leo` is the top foreign brief for exactly one use case (`soil_microbiome`), not a systematic winner.

## 2. 🟢 Lexical arm — the pre-registered blind test (never run before this script)

Wall-clock: **19.6 minutes** (14s building the `case_control_sample` of 38,904 rows across 34 use cases + 1164s for the 34 brief-offset passes, 34.2s/pass). Scorer: `bm25_obj` (BM25 of each paper against the brief's `objective` field — the natural analogue of cosine-to-brief, since `objective` is the prose field both a brief and an abstract write in). Secondary scorer `overlap_must_frac` reported below since it was free from the same pass.

11 of 33 non-quarantined use cases have `own < best_foreign` under lexical BM25: carbon_capture, solar_leo, synergy_chou_2003, synergy_chou_2004, synergy_hall_2012, synergy_leenaars_2020, synergy_meijboom_2021, synergy_menon_2022, synergy_moran_2021, synergy_nelson_2002, synergy_sep_2021.

| use_case_key                    | surface  | own   | best_foreign | by                         | margin |
|---------------------------------|----------|-------|--------------|----------------------------|--------|
| synergy_chou_2004               | benchset | 0.527 | 0.742        | synergy_jeyaraman_2020     | -0.215 |
| synergy_menon_2022              | benchset | 0.480 | 0.682        | synergy_wassenaar_2017     | -0.202 |
| synergy_leenaars_2020           | benchset | 0.555 | 0.706        | synergy_jeyaraman_2020     | -0.151 |
| synergy_sep_2021                | benchset | 0.445 | 0.576        | tech_forecasting           | -0.132 |
| synergy_nelson_2002             | benchset | 0.497 | 0.624        | cement_binders             | -0.127 |
| synergy_meijboom_2021           | benchset | 0.678 | 0.792        | synergy_van_der_valk_2021  | -0.114 |
| carbon_capture                  | live     | 0.508 | 0.593        | synergy_donners_2021       | -0.085 |
| solar_leo                       | live     | 0.557 | 0.637        | synergy_bos_2018           | -0.080 |
| synergy_moran_2021              | benchset | 0.602 | 0.656        | synergy_walker_2018        | -0.054 |
| synergy_chou_2003               | benchset | 0.758 | 0.807        | synergy_jeyaraman_2020     | -0.050 |
| synergy_hall_2012               | benchset | 0.906 | 0.938        | synergy_radjenovic_2013    | -0.032 |
| soil_microbiome                 | live     | 0.533 | 0.554        | roadfreight_metareview     | -0.021 |
| synergy_muthu_2021              | benchset | 0.636 | 0.653        | synergy_jeyaraman_2020     | -0.017 |
| roadfreight_metareview          | benchset | 0.630 | 0.626        | soil_microbiome            | 0.004  |
| synergy_wolters_2018            | benchset | 0.815 | 0.810        | synergy_bos_2018           | 0.005  |
| nykvist_evcharging              | benchset | 0.656 | 0.649        | synergy_walker_2018        | 0.007  |
| synergy_appenzeller-herzog_2019 | benchset | 0.885 | 0.869        | synergy_brouwer_2019       | 0.016  |
| synergy_donners_2021            | benchset | 0.678 | 0.660        | synergy_van_der_valk_2021  | 0.018  |
| synergy_bos_2018                | benchset | 0.936 | 0.918        | synergy_wolters_2018       | 0.018  |
| synergy_oud_2018                | benchset | 0.732 | 0.705        | synergy_jeyaraman_2020     | 0.027  |
| ner                             | live     | 0.576 | 0.533        | synergy_bos_2018           | 0.044  |
| synergy_wassenaar_2017          | benchset | 0.816 | 0.759        | synergy_walker_2018        | 0.058  |
| synergy_van_de_schoot_2018      | benchset | 0.796 | 0.735        | cement_binders             | 0.061  |
| synergy_walker_2018             | benchset | 0.637 | 0.571        | synergy_moran_2021         | 0.066  |
| synergy_jeyaraman_2020          | benchset | 0.836 | 0.766        | synergy_chou_2004          | 0.070  |
| tech_forecasting                | live     | 0.618 | 0.548        | soil_microbiome            | 0.071  |
| synergy_van_der_waal_2022       | benchset | 0.779 | 0.690        | synergy_leenaars_2019      | 0.089  |
| synergy_van_der_valk_2021       | benchset | 0.779 | 0.688        | synergy_bos_2018           | 0.091  |
| synergy_brouwer_2019            | benchset | 0.777 | 0.683        | carbon_capture             | 0.094  |
| cement_binders                  | live     | 0.649 | 0.555        | carbon_capture             | 0.094  |
| synergy_leenaars_2019           | benchset | 0.809 | 0.697        | synergy_van_der_waal_2022  | 0.111  |
| synergy_smid_2020               | benchset | 0.906 | 0.768        | synergy_van_de_schoot_2018 | 0.137  |
| synergy_van_dis_2020            | benchset | 0.837 | 0.680        | carbon_capture             | 0.157  |
| synergy_radjenovic_2013         | benchset | 0.923 | 0.744        | synergy_hall_2012          | 0.179  |

💡 **ELI18.** If the cosine-space margin were purely an artefact of one embedding model's geometry (e.g. an anisotropy quirk), this BM25 lexical-overlap matrix — built from term counts, no embedding involved — would show no relationship to it at all. It does not: see `## 3`.

Secondary scorer, `overlap_must_frac` (fraction of the ranking brief's must-include terms present in the paper), for reference — not used in any pass/fail bar below:

| use_case_key           | own   | best_foreign | by                      | margin |
|------------------------|-------|--------------|-------------------------|--------|
| synergy_menon_2022     | 0.493 | 0.789        | synergy_walker_2018     | -0.296 |
| soil_microbiome        | 0.460 | 0.578        | roadfreight_metareview  | -0.117 |
| synergy_moran_2021     | 0.473 | 0.573        | synergy_wolters_2018    | -0.099 |
| synergy_bos_2018       | 0.832 | 0.921        | synergy_wolters_2018    | -0.088 |
| solar_leo              | 0.466 | 0.551        | synergy_donners_2021    | -0.085 |
| synergy_muthu_2021     | 0.616 | 0.687        | synergy_van_dis_2020    | -0.071 |
| synergy_hall_2012      | 0.835 | 0.881        | synergy_radjenovic_2013 | -0.046 |
| synergy_jeyaraman_2020 | 0.717 | 0.717        | synergy_van_dis_2020    | 0.000  |

## 3. 🟡 Cross-instrument specificity — the corrected, pre-registered bar

**Pre-registered bar:** of the collections flagged at `margin < -0.03` in cosine space, ≥half also flag in lexical space, and ≤2 flag in lexical space that cosine called healthy.

- Cosine-flagged (jasper, excluding `roadfreight_metareview`): **5 of 33** — soil_microbiome, solar_leo, synergy_jeyaraman_2020, synergy_menon_2022, synergy_moran_2021.

- Of those, **3 of 5** also flag in lexical space: solar_leo, synergy_menon_2022, synergy_moran_2021.

- Cosine-flagged but lexical-healthy: soil_microbiome, synergy_jeyaraman_2020.

- Cosine-healthy but lexical-flagged (the false-new-flag count): **8** — carbon_capture, synergy_chou_2003, synergy_chou_2004, synergy_hall_2012, synergy_leenaars_2020, synergy_meijboom_2021, synergy_nelson_2002, synergy_sep_2021.

💡 **ELI18.** If the cosine margin were measuring something specific to jasper's embedding geometry rather than a property of the brief text itself, swapping to a bag-of-words BM25 representation with no embedding at all should produce an unrelated set of flags. Replication above the ≥half bar means the same collections look broken in a representation that shares nothing with cosine except the brief text — evidence for "property of the brief," not "property of one model."

**Pre-registered bar: FAIL.** ≥half of 5 cosine-flagged use cases replicate in lexical space (PASS: 3 ≥ 2.5 needed) **and** ≤2 lexical-only false-new-flags on cosine-healthy use cases (FAIL: 8 found).

## 4. 🟡 Validation (a) — margin vs. label-budget gain, n=7, exploratory, not independent

**Reference:** `wf_label_budget_shape_grid.csv`, `auc_gain` at the `n_pos=30` cell, averaged over its four `n_neg` ∈ {10,20,50,100} stages per use case (the same convention `run_label_budget_shape.py` itself uses to summarise "by positives labelled" — `groupby("n_pos").auc_gain.mean()`). `synergy_sep_2021` has only 24 train positives and has no `n_pos=30` row, so this join is **n=7 of the 8 set-A collections**, not 8.

| use_case_key              | own_diagonal_jasper | own_diagonal_qwen4b | auc_floor_grid | margin_jasper | margin_qwen4b | margin_lexical | auc_gain |
|---------------------------|---------------------|---------------------|----------------|---------------|---------------|----------------|----------|
| synergy_brouwer_2019      | 0.994               | 0.993               | 0.998          | 0.104         | 0.092         | 0.094          | -0.002   |
| synergy_leenaars_2020     | 0.833               | 0.863               | 0.870          | 0.047         | 0.067         | -0.151         | 0.027    |
| synergy_moran_2021        | 0.436               | 0.446               | 0.488          | -0.180        | -0.166        | -0.054         | 0.149    |
| synergy_muthu_2021        | 0.770               | 0.754               | 0.728          | 0.064         | 0.030         | -0.017         | 0.034    |
| synergy_nelson_2002       | 0.727               | 0.712               | 0.713          | 0.024         | 0.058         | -0.127         | 0.108    |
| synergy_van_der_valk_2021 | 0.846               | 0.853               | 0.881          | 0.132         | 0.060         | 0.091          | -0.023   |
| synergy_van_dis_2020      | 0.866               | 0.896               | 0.897          | 0.020         | 0.072         | 0.157          | 0.038    |

**Predicted sign: NEGATIVE** — a weaker brief (more negative margin) has more room to gain from labelling.

- **cosine margin (qwen4b — matches `auc_floor`'s instrument)** vs `auc_gain`: rho=-0.571 (95% CI [-1.000, 0.333], n=7, p=0.180); partial rho controlling for prevalence: -0.429.

- **cosine margin (jasper)** vs `auc_gain`: rho=-0.929 (95% CI [-1.000, -0.412], n=7, p=0.003); partial rho controlling for prevalence: -0.929.

- **lexical margin** vs `auc_gain`: rho=-0.286 (95% CI [-0.887, 0.686], n=7, p=0.535); partial rho controlling for prevalence: -0.286.

**Pre-registered bar: PASS.** Spearman rho ≤ -0.5 against ≥1 of (a)/(b), sign negative — best candidate here is `margin_jasper` at rho=-0.929 (clears -0.5). n=7, exploratory, set-A-derived on both sides — not independent confirmation.

## 5. Validation (b) — margin vs. induced-brief improvement, n=8, exploratory, not independent

**Reference:** `wf_llm_benchset_a_induced_cosine.csv`, `auc|induced − auc|supplied`, joined per model (jasper margin vs. jasper's own induced delta; qwen4b margin vs. qwen4b's own induced delta) so the join never mixes models. All 8 set-A collections have this cell — n=8, unlike (a).

**jasper:**

| use_case_key              | margin_cosine | margin_lexical | delta  |
|---------------------------|---------------|----------------|--------|
| synergy_brouwer_2019      | 0.104         | 0.094          | -0.001 |
| synergy_leenaars_2020     | 0.047         | -0.151         | -0.012 |
| synergy_moran_2021        | -0.180        | -0.054         | -0.019 |
| synergy_muthu_2021        | 0.064         | -0.017         | 0.004  |
| synergy_nelson_2002       | 0.024         | -0.127         | 0.020  |
| synergy_sep_2021          | -0.029        | -0.132         | -0.021 |
| synergy_van_der_valk_2021 | 0.132         | 0.091          | -0.017 |
| synergy_van_dis_2020      | 0.020         | 0.157          | 0.006  |

**qwen4b:**

| use_case_key              | margin_cosine | margin_lexical | delta  |
|---------------------------|---------------|----------------|--------|
| synergy_brouwer_2019      | 0.092         | 0.094          | -0.001 |
| synergy_leenaars_2020     | 0.067         | -0.151         | 0.013  |
| synergy_moran_2021        | -0.166        | -0.054         | -0.034 |
| synergy_muthu_2021        | 0.030         | -0.017         | 0.023  |
| synergy_nelson_2002       | 0.058         | -0.127         | 0.052  |
| synergy_sep_2021          | -0.019        | -0.132         | -0.003 |
| synergy_van_der_valk_2021 | 0.060         | 0.091          | -0.017 |
| synergy_van_dis_2020      | 0.072         | 0.157          | -0.001 |

**Predicted sign: NEGATIVE** — a weaker brief has more room to gain from brief induction.

- **jasper / cosine margin** vs induced delta: rho=0.238 (95% CI [-0.852, 0.753], n=8, p=0.570); partial rho controlling for prevalence: 0.238.

- **jasper / lexical margin** vs induced delta: rho=0.357 (95% CI [-0.519, 0.976], n=8, p=0.385); partial rho controlling for prevalence: 0.524.

- **qwen4b / cosine margin** vs induced delta: rho=0.262 (95% CI [-0.737, 0.899], n=8, p=0.531); partial rho controlling for prevalence: 0.643.

- **qwen4b / lexical margin** vs induced delta: rho=-0.167 (95% CI [-0.679, 0.644], n=8, p=0.693); partial rho controlling for prevalence: 0.167.

**Pre-registered bar: FAIL.** Best candidate `qwen4b/margin_lexical` at rho=-0.167. n=8, exploratory, not independent.

**Overall validity verdict: PASS.** The bar needs ≥1 of (a)/(b) at rho≤-0.5 with the predicted sign — at least one candidate clears it.

## 6. Confound control — prevalence

Prevalence correlated with LOGO transfer at rho -0.72 in the diversity notebook and is the obvious confound for anything measured across use cases of very different base rates. `partial_rho` (rank-residualised on prevalence) beside every rho above:

| index | candidate                 | rho    | partial_rho_ctrl_prevalence | survives |
|-------|---------------------------|--------|-----------------------------|----------|
| 0     | (a) margin_jasper         | -0.929 | -0.929                      | True     |
| 1     | (a) margin_qwen4b         | -0.571 | -0.429                      | True     |
| 2     | (a) margin_lexical        | -0.286 | -0.286                      | True     |
| 3     | (b) jasper/margin_cosine  | 0.238  | 0.238                       | True     |
| 4     | (b) jasper/margin_lexical | 0.357  | 0.524                       | True     |
| 5     | (b) qwen4b/margin_cosine  | 0.262  | 0.643                       | True     |
| 6     | (b) qwen4b/margin_lexical | -0.167 | 0.167                       | False    |

**Pre-registered bar:** `partial_rho` keeps its sign and ≥half its magnitude relative to raw rho.

**Pre-registered bar: PASS.** 6 of 7 candidate correlations survive prevalence control at that threshold.

## 7. Provenance stratification — TIRI-6 vs. benchset-27 (two figures, two n's)

`DATA_BRIEF.md` honest-limit #2: deriving a brief from the review's own abstract inflates every brief-reading score. Averaging TIRI's analyst-written briefs with benchset's review-abstract-derived ones would compare two different brief-generation processes, so they are never pooled.

| stratum                                                     | n  | mean_margin_cosine_jasper | mean_margin_lexical | n_flagged_cosine | n_flagged_lexical |
|-------------------------------------------------------------|----|---------------------------|---------------------|------------------|-------------------|
| TIRI-6 (analyst-written)                                    | 6  | 0.046                     | 0.004               | 2                | 2                 |
| benchset-27 (review-abstract-derived, roadfreight excluded) | 27 | 0.047                     | 0.004               | 3                | 9                 |

💡 **ELI18.** TIRI-6's own-brief AUC and margin come from 6 briefs a human analyst wrote to describe their own use case; benchset-27's come from briefs an LLM wrote by summarising the very review whose abstract the model then ranks against — structurally easier. If the two strata showed very different flag rates, that would say more about how each brief was produced than about brief quality per se.

## 8. What this changes

`NUMBERS.md` N33 (status `LIVE+SQL`) already quotes the `soil_microbiome` 0.603/0.792 pair with no validation behind it. This script supplies that validation: cross-instrument specificity **FAIL**, overall validity **PASS**, confound control **PASS**. Because at least one bar above did not clear, the caption should not ship as-is without saying so — a registered caption that fires on healthy briefs (or that nobody has shown predicts anything real) is worse than an unregistered one, per the plan's own "raises the stakes" framing.

> 🔴 **SUPERSEDED, 2026-08-14, by [`wf_foreign_brief_validity_setb.md`](wf_foreign_brief_validity_setb.md).**
> The section below argued that the margin is a real signal wearing the wrong caption, on the
> strength of rho **−0.929** against labelling gain. **That result does not survive a clean
> surface.** On set B the same statistic gives rho **−0.464** with a CI of −1.000 to **+0.765**, and
> the sign **flips to +0.143** once prevalence is controlled. It was also never
> embedding-independent even on set A: the qwen4b margin gives **−0.571 with a CI crossing zero**
> (p=0.180) on the same rows, and the lexical margin −0.286.
>
> So the conclusion below is wrong in its constructive half. The margin is not a mis-labelled
> workload forecaster; it is **jasper-specific on one burned surface**, which is the same defect the
> specificity test found, showing up a second time in the validity result. `NUMBERS.md` **N33 should
> be removed, not rewritten.** The text is kept below because the *reasoning* about how to tell a
> brief property from an embedding property still holds — it is the conclusion that was premature.

### The sharper version: right signal, wrong caption ~~— superseded, see above~~

The two results point somewhere more specific than "needs care", and it is worth stating plainly
because it changes what N33 should *say* rather than merely how confidently it says it.

🟢 **The margin is not a property of the brief.** Move from jasper's embedding space to BM25 and
the flagged set barely survives: 3 of 5 carry over, **8 new collections flag that cosine called
healthy**, and the two margins correlate at only rho 0.334. `soil_microbiome` — the entire
motivating example — has a lexical margin of **−0.021**, inside the 0.03 noise floor. A brief
defect ought to be visible to any reader of that brief; this one is visible to one embedding.

🟢 **The margin is, however, a strong predictor of where labelling pays.** Against the label-budget
gain it reaches rho **−0.929** (n=7, CI −1.000 to −0.412, and it survives every leave-one-out at
−0.886 to −0.943). The sign is the predicted one: the collections with the worst margins are the
ones labelling helps most.

*ELI18: we thought we had built a spec-checker — "a stranger's brief beats yours, so rewrite
yours". What we actually built is a workload forecaster — "this corpus is one your search terms
handle badly, so budget for labelling it". Those are different sentences on different screens, and
only the second one is supported.*

**So N33's caption should be rewritten, not merely hedged.** Not *"your objective may be describing
your intent rather than your literature"* — that asks the analyst to fix a brief we have no
representation-independent evidence is broken. Something closer to *"term-matching does poorly on
this pool; expect to earn recall from labels rather than from the brief"* — a claim this probe
actually supports, on the surface it was measured on, with the n=7 and the burned-surface caveat
attached.

⚠️ **And it stays exploratory.** n=7, both sides derived from set A, and set A has had five
selection passes. The correlation is robust to dropping any single collection but that is not the
same as replication. It should be re-measured on a surface that has not been selected against
before any screen quotes it.

## 9. What surprised / could not be computed

- `synergy_sep_2021` silently drops out of the grid CSV's `n_pos=30` slice (only 24 train positives) — validation (a) is n=7 even though the burned set-A surface is nominally 8 collections; this had to be discovered empirically rather than being stated anywhere in the plan.

- The mean-own vs mean-foreign numbers in the amendment (0.820/0.551) turned out to be **two different aggregates of the same matrix** (diagonal mean vs. mean of ALL off-diagonal cells) — not diagonal mean vs. mean of the per-corpus best-foreign values (which is 0.772 here). Both are reported in `## 1`/`## 2` so neither reads as the other by accident.

- Lexical wall-clock was 19.6 minutes for 34 full passes over 38,904 rows — inside the plan's "minutes" estimate.
